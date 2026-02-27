"""
imu/preintegration.py

IMU pre-integration for high-rate relative-pose prediction between
LiDAR/camera frames and for IMU-LiDAR tight coupling.

Pre-integration accumulates raw accelerometer and gyroscope measurements
between two keyframe timestamps into compact relative-motion increments
(ΔR, Δv, Δp) expressed in the *initial body frame* at the start of the
integration window.  These increments can be composed with an initial pose
to propagate the ego state forward without replaying the full measurement
stream.

Typical use-cases
-----------------
* **High-rate pose prediction**: bridge the gap between consecutive LiDAR or
  camera keyframes with IMU-rate relative poses.
* **IMU–LiDAR tight coupling**: supply an accurate initial pose guess for
  scan-matching and form IMU residuals inside a factor-graph optimiser.

Algorithm
---------
Integration uses the *midpoint* (trapezoidal) method for improved accuracy
over simple Euler integration:

.. code-block:: text

    For consecutive samples k and k+1 with interval dt:

        a_mid   = 0.5 * (a_k + a_{k+1}) - accel_bias
        ω_mid   = 0.5 * (ω_k + ω_{k+1}) - gyro_bias
        ΔR_prev = ΔR
        ΔR      = ΔR_prev @ Exp(ω_mid * dt)
        Δv      = Δv + ΔR_prev @ a_mid * dt
        Δp      = Δp + Δv_prev * dt + 0.5 * ΔR_prev @ a_mid * dt²

where ``Exp`` is the SO(3) matrix exponential (Rodrigues formula).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# SO(3) helpers
# ---------------------------------------------------------------------------


def _skew(v: np.ndarray) -> np.ndarray:
    """Return the 3×3 skew-symmetric matrix for cross-product with *v*."""
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def _exp_so3(phi: np.ndarray) -> np.ndarray:
    """SO(3) matrix exponential: rotation vector → 3×3 rotation matrix.

    Uses the Rodrigues formula.  For near-zero angles the first-order
    approximation ``I + [φ]×`` is used to avoid division by zero.

    Args:
        phi: Rotation vector ``ω * dt`` with magnitude equal to the rotation
            angle in radians.

    Returns:
        3×3 rotation matrix.
    """
    angle = float(np.linalg.norm(phi))
    if angle < 1e-10:
        return np.eye(3) + _skew(phi)
    K = _skew(phi / angle)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


# ---------------------------------------------------------------------------
# PreintegrationResult
# ---------------------------------------------------------------------------


@dataclass
class PreintegrationResult:
    """Relative-motion increments produced by IMU pre-integration.

    All quantities are expressed in the *body frame at the start* of the
    integration window (often called the *i*-frame or reference frame).

    Attributes:
        delta_rotation: 3×3 rotation matrix representing the orientation
            change over the integration window (body_i → body_j).
        delta_velocity: (3,) velocity increment in m/s, in the body_i frame.
        delta_position: (3,) position increment in metres, in the body_i frame.
        duration: Total integration interval in seconds.
        num_samples: Number of IMU samples used during integration.
    """

    delta_rotation: np.ndarray
    delta_velocity: np.ndarray
    delta_position: np.ndarray
    duration: float
    num_samples: int

    @property
    def delta_quaternion(self) -> np.ndarray:
        """Unit quaternion ``[w, x, y, z]`` equivalent of *delta_rotation*.

        Convenience property for interfacing with :class:`~sensor_transposition.frame_pose.FramePose`
        or other quaternion-based APIs.

        Returns:
            1-D numpy array ``[w, x, y, z]``.
        """
        from scipy.spatial.transform import Rotation as _R

        q = _R.from_matrix(self.delta_rotation).as_quat()  # returns [x, y, z, w]
        return np.array([q[3], q[0], q[1], q[2]])          # reorder to [w, x, y, z]


# ---------------------------------------------------------------------------
# ImuPreintegrator
# ---------------------------------------------------------------------------


class ImuPreintegrator:
    """Integrates IMU measurements into relative-pose increments.

    Accumulates accelerometer and gyroscope readings using the midpoint
    (trapezoidal) integration method to produce a
    :class:`PreintegrationResult` containing ΔR, Δv, and Δp.

    Args:
        accel_bias: (3,) accelerometer bias in m/s² (subtracted from each
            measurement before integration).  Defaults to ``[0, 0, 0]``.
        gyro_bias: (3,) gyroscope bias in rad/s (subtracted from each
            measurement before integration).  Defaults to ``[0, 0, 0]``.

    Example::

        from sensor_transposition.imu.preintegration import ImuPreintegrator

        integrator = ImuPreintegrator()
        result = integrator.integrate(timestamps, accel, gyro)
        print(result.delta_rotation)   # 3×3 rotation matrix
        print(result.delta_velocity)   # (3,) m/s
        print(result.delta_position)   # (3,) metres
    """

    def __init__(
        self,
        accel_bias: np.ndarray | None = None,
        gyro_bias: np.ndarray | None = None,
    ) -> None:
        self._accel_bias = np.asarray(
            accel_bias if accel_bias is not None else [0.0, 0.0, 0.0],
            dtype=float,
        )
        self._gyro_bias = np.asarray(
            gyro_bias if gyro_bias is not None else [0.0, 0.0, 0.0],
            dtype=float,
        )
        if self._accel_bias.shape != (3,):
            raise ValueError(
                f"accel_bias must have shape (3,), got {self._accel_bias.shape}."
            )
        if self._gyro_bias.shape != (3,):
            raise ValueError(
                f"gyro_bias must have shape (3,), got {self._gyro_bias.shape}."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def accel_bias(self) -> np.ndarray:
        """Accelerometer bias vector in m/s²."""
        return self._accel_bias.copy()

    @property
    def gyro_bias(self) -> np.ndarray:
        """Gyroscope bias vector in rad/s."""
        return self._gyro_bias.copy()

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def integrate(
        self,
        timestamps: np.ndarray,
        accel: np.ndarray,
        gyro: np.ndarray,
    ) -> PreintegrationResult:
        """Integrate IMU measurements using the midpoint method.

        Args:
            timestamps: Shape ``(N,)`` UNIX timestamps in seconds.  Must be
                strictly increasing.
            accel: Shape ``(N, 3)`` linear accelerations in m/s² expressed in
                the IMU body frame.
            gyro: Shape ``(N, 3)`` angular velocities in rad/s expressed in
                the IMU body frame.

        Returns:
            :class:`PreintegrationResult` containing ΔR, Δv, Δp, duration,
            and the number of samples used.

        Raises:
            ValueError: If fewer than two samples are provided, array shapes
                are inconsistent, or timestamps are not strictly increasing.
        """
        ts = np.asarray(timestamps, dtype=float)
        a = np.asarray(accel, dtype=float)
        w = np.asarray(gyro, dtype=float)

        n = ts.shape[0]
        if n < 2:
            raise ValueError(
                f"At least 2 IMU samples are required for pre-integration, got {n}."
            )
        if a.shape != (n, 3):
            raise ValueError(
                f"accel must have shape ({n}, 3), got {a.shape}."
            )
        if w.shape != (n, 3):
            raise ValueError(
                f"gyro must have shape ({n}, 3), got {w.shape}."
            )

        dt_all = np.diff(ts)
        if np.any(dt_all <= 0):
            raise ValueError("timestamps must be strictly increasing.")

        # Initialise integration state in the body frame at t=ts[0].
        delta_R = np.eye(3)
        delta_v = np.zeros(3)
        delta_p = np.zeros(3)

        for k in range(n - 1):
            dt = dt_all[k]
            dt2 = dt * dt

            # Midpoint (trapezoidal) estimates with bias subtraction.
            a_mid = 0.5 * (a[k] + a[k + 1]) - self._accel_bias
            w_mid = 0.5 * (w[k] + w[k + 1]) - self._gyro_bias

            # Save state at the start of this step for velocity/position updates.
            delta_R_k = delta_R
            delta_v_k = delta_v

            # Rotation update via SO(3) exponential map.
            delta_R = delta_R_k @ _exp_so3(w_mid * dt)

            # Velocity update (rotate accel into the i-frame using delta_R_k).
            delta_v = delta_v_k + delta_R_k @ a_mid * dt

            # Position update.
            delta_p = delta_p + delta_v_k * dt + 0.5 * (delta_R_k @ a_mid) * dt2

        return PreintegrationResult(
            delta_rotation=delta_R,
            delta_velocity=delta_v,
            delta_position=delta_p,
            duration=float(ts[-1] - ts[0]),
            num_samples=n,
        )
