"""
imu/ekf.py

Error-State Extended Kalman Filter (ES-EKF) for fusing IMU measurements with
pose, position, and velocity observations.

Background
----------
The ES-EKF splits the state into a *nominal* (large-signal) component and a
small *error* (perturbation) component.  The nominal state is propagated by
direct IMU integration; the 15-dimensional error state captures the
linearised deviations that are tracked by the Kalman filter:

.. code-block:: text

    δx = [δp (3), δv (3), δθ (3), δba (3), δbg (3)]

where:

* ``δp``  – position error (metres)
* ``δv``  – velocity error (m/s)
* ``δθ``  – orientation error as a rotation vector in SO(3) tangent space (rad)
* ``δba`` – accelerometer bias error (m/s²)
* ``δbg`` – gyroscope bias error (rad/s)

Nominal state
-------------
The nominal state maintained inside :class:`EkfState` contains:

* ``position``   – 3-D position in the world/map frame (metres).
* ``velocity``   – 3-D velocity in the world/map frame (m/s).
* ``quaternion`` – unit quaternion ``[w, x, y, z]`` representing the
  body-to-world orientation.
* ``accel_bias`` – accelerometer bias in m/s².
* ``gyro_bias``  – gyroscope bias in rad/s.
* ``covariance`` – 15×15 error-state covariance matrix P.

Prediction step (IMU)
---------------------
A single IMU sample (corrected accelerometer ``a``, corrected gyroscope ``ω``,
time step ``dt``) drives the nominal-state equations:

.. code-block:: text

    p  ← p + v·dt + ½(R·a_c + g)·dt²
    v  ← v + (R·a_c + g)·dt
    q  ← q ⊗ Exp(ω_c·dt)
    ba ← ba   (random-walk model)
    bg ← bg

where ``a_c = a − ba``, ``ω_c = ω − bg``, ``R`` is the rotation matrix from
``q``, ``g`` is the gravity vector, and ``Exp`` is the SO(3) exponential map.

The 15×15 linearised state-transition matrix F and the discrete process-noise
matrix Q are used to propagate the error-state covariance:

.. code-block:: text

    P ← F P Fᵀ + Q

Update step (observation)
-------------------------
A measurement ``z = H·δx + noise(R_noise)`` is fused via:

.. code-block:: text

    K = P·Hᵀ·(H·P·Hᵀ + R_noise)⁻¹
    δx = K·(z − ẑ)
    P  ← (I − K·H)·P·(I − K·H)ᵀ + K·R_noise·Kᵀ   [Joseph form]

The error state ``δx`` is then injected into the nominal state and reset to
zero.

Pre-built measurement models
-----------------------------
:meth:`ImuEkf.position_update`
    3-D position from GPS or LiDAR odometry.

:meth:`ImuEkf.velocity_update`
    3-D velocity from wheel odometry or GPS-Doppler.

:meth:`ImuEkf.pose_update`
    6-DOF position + orientation from LiDAR scan matching.

All functions operate on plain NumPy arrays and depend only on ``numpy``
(already required by ``sensor_transposition``), so no additional dependencies
are needed.

Typical use-case
----------------
::

    from sensor_transposition.imu.ekf import ImuEkf, EkfState
    import numpy as np

    ekf = ImuEkf(gravity=np.array([0.0, 0.0, -9.81]))
    state = EkfState(timestamp=0.0)

    # IMU prediction loop
    for t, a, w in imu_stream:
        dt = t - state.timestamp
        state = ekf.predict(state, a, w, dt)

    # Fuse a GPS position fix
    state = ekf.position_update(
        state,
        position=np.array([x, y, z]),
        noise=np.eye(3) * 0.5 ** 2,
    )

    # Fuse a LiDAR odometry pose
    state = ekf.pose_update(
        state,
        position=np.array([px, py, pz]),
        quaternion=np.array([qw, qx, qy, qz]),
        position_noise=np.eye(3) * 0.1 ** 2,
        orientation_noise=np.eye(3) * np.deg2rad(1) ** 2,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# SO(3) / quaternion helpers
# ---------------------------------------------------------------------------


def _skew(v: np.ndarray) -> np.ndarray:
    """Return the 3×3 skew-symmetric matrix for the cross-product with *v*."""
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def _exp_so3(phi: np.ndarray) -> np.ndarray:
    """SO(3) exponential map: rotation vector → 3×3 rotation matrix.

    Uses the Rodrigues formula.  For near-zero angles the first-order
    approximation ``I + [φ]×`` is used to avoid division by zero.

    Args:
        phi: Rotation vector with magnitude equal to the rotation angle (rad).

    Returns:
        3×3 rotation matrix.
    """
    angle = float(np.linalg.norm(phi))
    if angle < 1e-10:
        return np.eye(3) + _skew(phi)
    K = _skew(phi / angle)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _rot_from_quat(q: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix from a unit quaternion ``[w, x, y, z]``."""
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z),  2.0 * (x * y - w * z),        2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z),         1.0 - 2.0 * (x * x + z * z),  2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y),         2.0 * (y * z + w * x),         1.0 - 2.0 * (x * x + y * y)],
    ])


def _quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two ``[w, x, y, z]`` unit quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_from_rotvec(phi: np.ndarray) -> np.ndarray:
    """Unit quaternion ``[w, x, y, z]`` from a rotation vector.

    For small angles the first-order approximation is used to avoid
    numerical issues.

    Args:
        phi: Rotation vector (rad).

    Returns:
        Unit quaternion ``[w, x, y, z]``.
    """
    angle = float(np.linalg.norm(phi))
    if angle < 1e-10:
        return np.array([1.0, 0.5 * phi[0], 0.5 * phi[1], 0.5 * phi[2]])
    half = 0.5 * angle
    axis = phi / angle
    return np.array([np.cos(half), *(np.sin(half) * axis)])


def _quat_norm(q: np.ndarray) -> np.ndarray:
    """Return a normalised copy of quaternion *q*."""
    n = np.linalg.norm(q)
    if n < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


# ---------------------------------------------------------------------------
# EkfState
# ---------------------------------------------------------------------------


@dataclass
class EkfState:
    """Nominal state and error-state covariance for the ES-EKF.

    The nominal state stores:

    * **position** – ``(3,)`` metres in the world/map frame.
    * **velocity** – ``(3,)`` m/s in the world/map frame.
    * **quaternion** – ``(4,)`` unit quaternion ``[w, x, y, z]``
      representing the body-to-world orientation.
    * **accel_bias** – ``(3,)`` accelerometer bias in m/s².
    * **gyro_bias** – ``(3,)`` gyroscope bias in rad/s.

    The error-state covariance **P** is a ``(15, 15)`` matrix corresponding
    to the error state
    ``δx = [δp (3), δv (3), δθ (3), δba (3), δbg (3)]``.

    Attributes:
        position: ``(3,)`` position in the world frame (metres).
        velocity: ``(3,)`` velocity in the world frame (m/s).
        quaternion: ``(4,)`` unit quaternion ``[w, x, y, z]``.
        accel_bias: ``(3,)`` accelerometer bias (m/s²).
        gyro_bias: ``(3,)`` gyroscope bias (rad/s).
        covariance: ``(15, 15)`` error-state covariance matrix.
        timestamp: UNIX timestamp of this state in seconds.
    """

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(15) * 1e-4)
    timestamp: float = 0.0

    @property
    def rotation_matrix(self) -> np.ndarray:
        """3×3 body-to-world rotation matrix derived from :attr:`quaternion`."""
        return _rot_from_quat(self.quaternion)


# ---------------------------------------------------------------------------
# ImuEkf
# ---------------------------------------------------------------------------


class ImuEkf:
    """Error-State EKF for fusing IMU measurements with pose/position/velocity
    observations.

    The filter maintains a 15-dimensional error state
    ``δx = [δp, δv, δθ, δba, δbg]`` alongside the nominal state stored in
    :class:`EkfState`.

    Args:
        gravity: Gravity vector in the world/map frame (default:
            ``[0, 0, −9.81]`` m/s²).
        accel_noise_density: Accelerometer white-noise density in
            m/s²/√Hz (default: ``0.01``).
        gyro_noise_density: Gyroscope white-noise density in rad/s/√Hz
            (default: ``0.001``).
        accel_bias_noise: Accelerometer bias random-walk noise density in
            m/s²·√Hz (default: ``0.0001``).
        gyro_bias_noise: Gyroscope bias random-walk noise density in
            rad/s·√Hz (default: ``0.00001``).

    Example::

        from sensor_transposition.imu.ekf import ImuEkf, EkfState
        import numpy as np

        ekf = ImuEkf()
        state = EkfState(timestamp=0.0)

        # --- Prediction loop (call once per IMU sample) ---
        for t, accel, gyro in imu_samples:
            dt = t - state.timestamp
            state = ekf.predict(state, accel, gyro, dt)

        # --- Position update (e.g. from GPS) ---
        state = ekf.position_update(
            state,
            position=np.array([1.0, 2.0, 0.0]),
            noise=np.eye(3) * 0.25,
        )

        # --- Pose update (e.g. from LiDAR scan matching) ---
        state = ekf.pose_update(
            state,
            position=np.array([1.0, 2.0, 0.0]),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            position_noise=np.eye(3) * 0.01,
            orientation_noise=np.eye(3) * 1e-4,
        )
    """

    def __init__(
        self,
        gravity: np.ndarray | None = None,
        accel_noise_density: float = 0.01,
        gyro_noise_density: float = 0.001,
        accel_bias_noise: float = 0.0001,
        gyro_bias_noise: float = 0.00001,
    ) -> None:
        self._g = np.asarray(
            gravity if gravity is not None else [0.0, 0.0, -9.81],
            dtype=float,
        )
        if self._g.shape != (3,):
            raise ValueError(
                f"gravity must have shape (3,), got {self._g.shape}."
            )
        self._accel_nd = float(accel_noise_density)
        self._gyro_nd = float(gyro_noise_density)
        self._accel_bn = float(accel_bias_noise)
        self._gyro_bn = float(gyro_bias_noise)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def gravity(self) -> np.ndarray:
        """Gravity vector in the world frame (m/s²)."""
        return self._g.copy()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        state: EkfState,
        accel: np.ndarray,
        gyro: np.ndarray,
        dt: float,
    ) -> EkfState:
        """Propagate the EKF state forward by one IMU sample.

        Uses the nominal-state kinematic equations and the linearised
        state-transition Jacobian F to advance both the nominal state and
        the 15×15 error-state covariance.

        Args:
            state: Current :class:`EkfState`.
            accel: ``(3,)`` accelerometer measurement in m/s² (body frame).
            gyro: ``(3,)`` gyroscope measurement in rad/s (body frame).
            dt: Time step in seconds (must be positive).

        Returns:
            New :class:`EkfState` with updated nominal state, covariance,
            and timestamp ``state.timestamp + dt``.

        Raises:
            ValueError: If *dt* is not positive.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")

        a = np.asarray(accel, dtype=float)
        w = np.asarray(gyro, dtype=float)

        # Bias-corrected measurements.
        a_c = a - state.accel_bias
        w_c = w - state.gyro_bias

        R = state.rotation_matrix

        # ---- Nominal-state propagation ----
        world_accel = R @ a_c + self._g
        new_p = state.position + state.velocity * dt + 0.5 * world_accel * dt ** 2
        new_v = state.velocity + world_accel * dt
        dq = _quat_from_rotvec(w_c * dt)
        new_q = _quat_norm(_quat_mult(state.quaternion, dq))
        new_ba = state.accel_bias.copy()
        new_bg = state.gyro_bias.copy()

        # ---- Error-state Jacobian F (15×15) ----
        # Linearised discrete-time transition; each sub-block is 3×3.
        # Error state ordering: [δp, δv, δθ, δba, δbg]
        F = np.eye(15)
        # δp row
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = -0.5 * R @ _skew(a_c) * dt ** 2
        F[0:3, 9:12] = -0.5 * R * dt ** 2
        # δv row
        F[3:6, 6:9] = -R @ _skew(a_c) * dt
        F[3:6, 9:12] = -R * dt
        # δθ row
        F[6:9, 6:9] = _exp_so3(-w_c * dt)
        F[6:9, 12:15] = -np.eye(3) * dt
        # δba and δbg rows: identity (already set by np.eye(15))

        # ---- Discrete process-noise matrix Q (15×15) ----
        # Spectral densities scaled by dt to give per-step variances.
        qa = self._accel_nd ** 2 * dt   # accel white-noise variance / step
        qg = self._gyro_nd ** 2 * dt    # gyro white-noise variance / step
        qab = self._accel_bn ** 2 * dt  # accel bias random-walk variance / step
        qgb = self._gyro_bn ** 2 * dt   # gyro bias random-walk variance / step

        Q = np.zeros((15, 15))
        Q[3:6, 3:6] = np.eye(3) * qa    # velocity noise (R @ I @ Rᵀ = I)
        Q[6:9, 6:9] = np.eye(3) * qg    # orientation noise
        Q[9:12, 9:12] = np.eye(3) * qab
        Q[12:15, 12:15] = np.eye(3) * qgb

        # ---- Covariance propagation ----
        new_P = F @ state.covariance @ F.T + Q
        new_P = 0.5 * (new_P + new_P.T)  # enforce symmetry

        return EkfState(
            position=new_p,
            velocity=new_v,
            quaternion=new_q,
            accel_bias=new_ba,
            gyro_bias=new_bg,
            covariance=new_P,
            timestamp=state.timestamp + dt,
        )

    # ------------------------------------------------------------------
    # Internal update helper
    # ------------------------------------------------------------------

    def _update(
        self,
        state: EkfState,
        residual: np.ndarray,
        H: np.ndarray,
        R_noise: np.ndarray,
    ) -> EkfState:
        """Generic EKF measurement update (Joseph covariance form).

        Args:
            state: Current :class:`EkfState`.
            residual: Measurement residual vector ``z − ẑ``.
            H: Observation Jacobian matrix ``(m, 15)``.
            R_noise: Measurement noise covariance ``(m, m)``.

        Returns:
            Updated :class:`EkfState`.
        """
        P = state.covariance
        S = H @ P @ H.T + R_noise
        K = P @ H.T @ np.linalg.inv(S)
        delta_x = K @ residual

        # Inject error state into nominal state.
        dp = delta_x[0:3]
        dv = delta_x[3:6]
        dtheta = delta_x[6:9]
        dba = delta_x[9:12]
        dbg = delta_x[12:15]

        new_p = state.position + dp
        new_v = state.velocity + dv
        new_q = _quat_norm(_quat_mult(state.quaternion, _quat_from_rotvec(dtheta)))
        new_ba = state.accel_bias + dba
        new_bg = state.gyro_bias + dbg

        # Joseph form for numerical stability.
        I_KH = np.eye(15) - K @ H
        new_P = I_KH @ P @ I_KH.T + K @ R_noise @ K.T
        new_P = 0.5 * (new_P + new_P.T)

        return EkfState(
            position=new_p,
            velocity=new_v,
            quaternion=new_q,
            accel_bias=new_ba,
            gyro_bias=new_bg,
            covariance=new_P,
            timestamp=state.timestamp,
        )

    # ------------------------------------------------------------------
    # Measurement updates
    # ------------------------------------------------------------------

    def position_update(
        self,
        state: EkfState,
        position: np.ndarray,
        noise: np.ndarray,
    ) -> EkfState:
        """Fuse a 3-D position measurement (e.g. from GPS or LiDAR odometry).

        Args:
            state: Current :class:`EkfState`.
            position: ``(3,)`` measured position in the world frame (metres).
            noise: ``(3, 3)`` measurement noise covariance (m²).

        Returns:
            Updated :class:`EkfState`.

        Raises:
            ValueError: If *position* or *noise* have unexpected shapes.
        """
        pos = np.asarray(position, dtype=float)
        R_noise = np.asarray(noise, dtype=float)
        if pos.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {pos.shape}.")
        if R_noise.shape != (3, 3):
            raise ValueError(f"noise must have shape (3, 3), got {R_noise.shape}.")

        H = np.zeros((3, 15))
        H[0:3, 0:3] = np.eye(3)
        residual = pos - state.position
        return self._update(state, residual, H, R_noise)

    def velocity_update(
        self,
        state: EkfState,
        velocity: np.ndarray,
        noise: np.ndarray,
    ) -> EkfState:
        """Fuse a 3-D velocity measurement (e.g. from wheel odometry or
        GPS-Doppler).

        Args:
            state: Current :class:`EkfState`.
            velocity: ``(3,)`` measured velocity in the world frame (m/s).
            noise: ``(3, 3)`` measurement noise covariance ((m/s)²).

        Returns:
            Updated :class:`EkfState`.

        Raises:
            ValueError: If *velocity* or *noise* have unexpected shapes.
        """
        vel = np.asarray(velocity, dtype=float)
        R_noise = np.asarray(noise, dtype=float)
        if vel.shape != (3,):
            raise ValueError(f"velocity must have shape (3,), got {vel.shape}.")
        if R_noise.shape != (3, 3):
            raise ValueError(f"noise must have shape (3, 3), got {R_noise.shape}.")

        H = np.zeros((3, 15))
        H[0:3, 3:6] = np.eye(3)
        residual = vel - state.velocity
        return self._update(state, residual, H, R_noise)

    def pose_update(
        self,
        state: EkfState,
        position: np.ndarray,
        quaternion: np.ndarray,
        position_noise: np.ndarray,
        orientation_noise: np.ndarray,
    ) -> EkfState:
        """Fuse a full 6-DOF pose measurement (e.g. from LiDAR scan matching).

        The orientation residual is expressed as the rotation vector of the
        relative rotation between the measured and predicted orientations.

        Args:
            state: Current :class:`EkfState`.
            position: ``(3,)`` measured position in metres.
            quaternion: ``(4,)`` measured orientation as ``[w, x, y, z]`` unit
                quaternion.
            position_noise: ``(3, 3)`` position measurement noise covariance
                (m²).
            orientation_noise: ``(3, 3)`` orientation measurement noise
                covariance (rad²), expressed in rotation-vector units.

        Returns:
            Updated :class:`EkfState`.

        Raises:
            ValueError: If any argument has an unexpected shape.
        """
        pos = np.asarray(position, dtype=float)
        q_meas = _quat_norm(np.asarray(quaternion, dtype=float))
        R_pos = np.asarray(position_noise, dtype=float)
        R_ori = np.asarray(orientation_noise, dtype=float)

        if pos.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {pos.shape}.")
        if q_meas.shape != (4,):
            raise ValueError(f"quaternion must have shape (4,), got {q_meas.shape}.")
        if R_pos.shape != (3, 3):
            raise ValueError(
                f"position_noise must have shape (3, 3), got {R_pos.shape}."
            )
        if R_ori.shape != (3, 3):
            raise ValueError(
                f"orientation_noise must have shape (3, 3), got {R_ori.shape}."
            )

        # 6-row observation Jacobian: [δp block | δθ block].
        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)   # position rows
        H[3:6, 6:9] = np.eye(3)   # orientation rows

        # Position residual.
        dp = pos - state.position

        # Orientation residual as a rotation vector:
        #   dq = q_meas ⊗ q_pred*
        q_pred = state.quaternion
        q_pred_conj = np.array([q_pred[0], -q_pred[1], -q_pred[2], -q_pred[3]])
        dq = _quat_norm(_quat_mult(q_meas, q_pred_conj))
        # Ensure the scalar part is non-negative (pick the shorter arc).
        if dq[0] < 0.0:
            dq = -dq
        angle = 2.0 * np.arccos(np.clip(dq[0], -1.0, 1.0))
        vec_norm = np.linalg.norm(dq[1:4])
        if angle < 1e-10 or vec_norm < 1e-10:
            dtheta = np.zeros(3)
        else:
            dtheta = angle * dq[1:4] / vec_norm

        residual = np.concatenate([dp, dtheta])

        R_noise = np.zeros((6, 6))
        R_noise[0:3, 0:3] = R_pos
        R_noise[3:6, 3:6] = R_ori

        return self._update(state, residual, H, R_noise)
