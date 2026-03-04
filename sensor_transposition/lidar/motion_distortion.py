"""
lidar/motion_distortion.py

LiDAR point-cloud motion-distortion correction (deskewing) using IMU data.

When a spinning or MEMS LiDAR acquires a single "frame", individual points are
measured at slightly different physical times spread across the scan interval
(typically ~100 ms for a 10 Hz rotating LiDAR).  If the sensor platform is
moving, the scan is *distorted* because each point was observed from a
different pose.

This module provides :func:`deskew_scan`, which uses IMU measurements to
estimate the relative body-frame transform at each point's acquisition time
and corrects the point coordinates to a single reference timestamp — usually
the start or the end of the scan.

Algorithm
---------
1. IMU gyroscope and accelerometer readings are integrated step-by-step using
   the midpoint (trapezoidal) method, starting from the first IMU sample with
   an identity pose and an optional initial velocity.  This produces a discrete
   rotation + position trajectory at each IMU sample time.

2. Continuous poses are obtained by **SLERP** (Spherical Linear Interpolation)
   for rotations and **linear interpolation** for positions at any time within
   the scan window.

3. The relative transform at each point's acquisition time with respect to the
   reference time is::

       R_rel = R_ref.T  @  R(t_i)
       t_rel = R_ref.T  @  (p(t_i) − p(t_ref))

   where ``R_ref``, ``p(t_ref)`` are the rotation and position at the reference
   time.

4. Each point is corrected by applying the relative transform::

       p_corrected = R_rel @ p_i + t_rel

   The result is a point cloud in which every point is expressed as if the
   sensor had been stationary at its pose at ``ref_time``.

Notes
-----
* **Rotation correction** is always physically meaningful and depends only on
  the gyroscope (together with gyro bias).

* **Position correction** additionally depends on the accelerometer, gravity
  vector, and the initial velocity at the start of the IMU window.  If
  ``initial_velocity`` is not provided (defaults to zero), position correction
  will be inaccurate for platforms with significant translational motion.
  For best results, supply the velocity estimate from the EKF or odometry
  filter at the start of the scan.  Alternatively, set ``gravity`` to a zero
  vector to disable the gravity term if the initial velocity is unknown.

* Point and IMU timestamps must share the same time reference (e.g. both from
  the same hardware clock or already synchronised via :mod:`sensor_transposition.sync`).

* Point timestamps that fall outside the IMU time window are clamped to the
  nearest boundary, meaning no correction is applied to those points (identity
  transform).

Example::

    from sensor_transposition.lidar.motion_distortion import deskew_scan

    # imu_times: (K,) timestamps; imu_accel: (K, 3); imu_gyro: (K, 3)
    # points: (N, 3) LiDAR points in sensor frame; point_times: (N,)
    corrected = deskew_scan(
        points, point_times, imu_times, imu_accel, imu_gyro,
        ref_time=point_times[0],          # deskew to scan-start
    )

References
----------
* Zhang & Singh, "LOAM: Lidar Odometry and Mapping in Real-time", RSS 2014.
* Xu *et al.*, "FAST-LIO2: Fast Direct LiDAR-Inertial Odometry", T-RO 2022.
* Shan *et al.*, "LIO-SAM: Tightly-coupled Lidar Inertial Odometry via
  Smoothing and Mapping", IROS 2020.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _Rotation
from scipy.spatial.transform import Slerp as _Slerp


# ---------------------------------------------------------------------------
# SO(3) helpers (intentionally self-contained; mirrors imu/preintegration.py)
# ---------------------------------------------------------------------------


def _skew(v: np.ndarray) -> np.ndarray:
    """Return the 3×3 skew-symmetric matrix for the cross-product with *v*."""
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def _exp_so3(phi: np.ndarray) -> np.ndarray:
    """SO(3) matrix exponential: rotation vector → 3×3 rotation matrix.

    Uses the Rodrigues formula.  For near-zero angles the first-order
    approximation ``I + [φ]×`` is used to avoid division by zero.
    """
    angle = float(np.linalg.norm(phi))
    if angle < 1e-10:
        return np.eye(3) + _skew(phi)
    K = _skew(phi / angle)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


# ---------------------------------------------------------------------------
# Trajectory builder
# ---------------------------------------------------------------------------


def _build_trajectory(
    imu_times: np.ndarray,
    imu_accel: np.ndarray,
    imu_gyro: np.ndarray,
    accel_bias: np.ndarray,
    gyro_bias: np.ndarray,
    gravity: np.ndarray,
    initial_velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate IMU data to produce a rotation + position trajectory.

    The starting pose is an identity rotation and zero position at
    ``imu_times[0]``, and the starting velocity is ``initial_velocity``.
    Integration uses the midpoint (trapezoidal) method identical to
    :mod:`sensor_transposition.imu.preintegration`.

    Args:
        imu_times: ``(K,)`` strictly increasing UNIX timestamps in seconds.
        imu_accel: ``(K, 3)`` accelerometer measurements in m/s² (body frame).
        imu_gyro: ``(K, 3)`` gyroscope measurements in rad/s (body frame).
        accel_bias: ``(3,)`` accelerometer bias to subtract (m/s²).
        gyro_bias: ``(3,)`` gyroscope bias to subtract (rad/s).
        gravity: ``(3,)`` gravity vector in the world frame (m/s²).
        initial_velocity: ``(3,)`` body-frame velocity at ``imu_times[0]``
            expressed in the world frame (m/s).

    Returns:
        Tuple ``(rotations, positions)`` where

        * ``rotations`` is ``(K, 3, 3)`` – body-to-world rotation matrices.
        * ``positions`` is ``(K, 3)`` – body origin positions in the world
          frame (metres), with the world frame centred at the body origin at
          ``imu_times[0]``.
    """
    K = len(imu_times)
    rotations = np.empty((K, 3, 3))
    positions = np.empty((K, 3))

    rotations[0] = np.eye(3)
    positions[0] = np.zeros(3)
    velocity = initial_velocity.copy()

    for k in range(K - 1):
        dt = imu_times[k + 1] - imu_times[k]

        # Midpoint (trapezoidal) estimates with bias subtraction.
        a_mid = 0.5 * (imu_accel[k] + imu_accel[k + 1]) - accel_bias
        w_mid = 0.5 * (imu_gyro[k] + imu_gyro[k + 1]) - gyro_bias

        R_k = rotations[k]
        v_k = velocity.copy()

        # Rotation update via SO(3) exponential map.
        rotations[k + 1] = R_k @ _exp_so3(w_mid * dt)

        # Velocity and position update (gravity included as a world-frame
        # correction to the specific force).
        world_accel = R_k @ a_mid + gravity
        velocity = v_k + world_accel * dt
        positions[k + 1] = positions[k] + v_k * dt + 0.5 * world_accel * dt ** 2

    return rotations, positions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def deskew_scan(
    points: np.ndarray,
    point_times: np.ndarray,
    imu_times: np.ndarray,
    imu_accel: np.ndarray,
    imu_gyro: np.ndarray,
    ref_time: float,
    *,
    accel_bias: np.ndarray | None = None,
    gyro_bias: np.ndarray | None = None,
    gravity: np.ndarray | None = None,
    initial_velocity: np.ndarray | None = None,
) -> np.ndarray:
    """Correct LiDAR motion distortion using IMU measurements.

    Transforms each point from its acquisition pose to the pose at
    ``ref_time``, producing a deskewed point cloud in which all points are
    expressed as if the sensor had been stationary at its ``ref_time`` pose.

    Args:
        points: ``(N, 3)`` float array of LiDAR points in the sensor body
            frame (metres).
        point_times: ``(N,)`` float array of per-point acquisition timestamps
            in seconds.  Must share the same time base as ``imu_times``.
        imu_times: ``(K,)`` float array of IMU sample timestamps in seconds.
            Must be strictly increasing and must span ``ref_time`` as well as
            the range of ``point_times``.  At least **2** samples are required.
        imu_accel: ``(K, 3)`` float array of accelerometer measurements in
            m/s² (body frame).
        imu_gyro: ``(K, 3)`` float array of gyroscope measurements in rad/s
            (body frame).
        ref_time: Reference timestamp (seconds) to which all points are
            deskewed.  Choose this as either the **start**
            (``point_times.min()``) or the **end** (``point_times.max()``)
            of the LiDAR sweep:

            * **Start** – the deskewed cloud is expressed in the sensor frame
              at the beginning of the sweep.  This is typical for
              LOAM-style front-end pipelines.
            * **End** – the deskewed cloud is expressed in the sensor frame at
              the end of the sweep.  Some systems prefer this because the last
              IMU sample falls closest to the next processing step.

            Using the **midpoint** (``(min + max) / 2``) is also valid and
            minimises the maximum per-point correction magnitude.

            .. important::
                ``ref_time`` and all values in ``point_times`` must be in the
                **same hardware clock** as ``imu_times``.  If the clocks
                differ, apply :func:`~sensor_transposition.sync.apply_time_offset`
                to align them before calling this function.
        accel_bias: Optional ``(3,)`` accelerometer bias in m/s².  Defaults to
            ``[0, 0, 0]`` (no correction).
        gyro_bias: Optional ``(3,)`` gyroscope bias in rad/s.  Defaults to
            ``[0, 0, 0]`` (no correction).
        gravity: Optional ``(3,)`` gravity vector in the world frame (m/s²).
            Defaults to ``[0, 0, −9.81]``.  Set to ``np.zeros(3)`` to
            disable gravity-induced position correction when the initial
            platform velocity is unknown.
        initial_velocity: Optional ``(3,)`` body velocity at
            ``imu_times[0]`` in the world frame (m/s).  Defaults to
            ``[0, 0, 0]``.  Providing the velocity from an EKF or odometry
            filter significantly improves position correction accuracy for
            fast-moving platforms.

    Returns:
        ``(N, 3)`` float array of deskewed LiDAR points expressed in the
        sensor body frame at ``ref_time``.

    Raises:
        ValueError: If input arrays have incompatible shapes, ``imu_times``
            has fewer than 2 samples, timestamps are not strictly increasing,
            or bias / gravity / velocity arrays have the wrong shape.

    Notes:
        * Point timestamps outside the IMU window are clamped to the nearest
          boundary; those points receive an identity (no-op) correction.
        * The primary correction comes from gyroscope-driven rotation.
          Accelerometer-driven position correction is only meaningful when
          ``initial_velocity`` matches the true velocity at scan start.

    Example::

        import numpy as np
        from sensor_transposition.lidar.motion_distortion import deskew_scan
        from sensor_transposition.imu.ekf import ImuEkf, EkfState

        # Assume `ekf_state` is an EkfState at the start of the scan.
        corrected = deskew_scan(
            points=lidar_points,
            point_times=lidar_point_times,
            imu_times=imu_timestamps,
            imu_accel=imu_accel_data,
            imu_gyro=imu_gyro_data,
            ref_time=lidar_point_times[0],
            initial_velocity=ekf_state.velocity,
        )
    """
    # ------------------------------------------------------------------
    # Input conversion
    # ------------------------------------------------------------------
    pts = np.asarray(points, dtype=float)
    pt_ts = np.asarray(point_times, dtype=float)
    imu_ts = np.asarray(imu_times, dtype=float)
    a = np.asarray(imu_accel, dtype=float)
    w = np.asarray(imu_gyro, dtype=float)

    # ------------------------------------------------------------------
    # Shape validation
    # ------------------------------------------------------------------
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(
            f"points must be an (N, 3) array, got shape {pts.shape}."
        )
    N = pts.shape[0]
    if N < 1:
        raise ValueError("points must contain at least one point.")

    if pt_ts.shape != (N,):
        raise ValueError(
            f"point_times must have shape ({N},), got {pt_ts.shape}."
        )

    K = imu_ts.shape[0]
    if imu_ts.ndim != 1 or K < 2:
        raise ValueError(
            f"imu_times must be a 1-D array with at least 2 elements, "
            f"got shape {imu_ts.shape}."
        )
    if a.shape != (K, 3):
        raise ValueError(
            f"imu_accel must have shape ({K}, 3), got {a.shape}."
        )
    if w.shape != (K, 3):
        raise ValueError(
            f"imu_gyro must have shape ({K}, 3), got {w.shape}."
        )
    if np.any(np.diff(imu_ts) <= 0.0):
        raise ValueError("imu_times must be strictly increasing.")

    # ------------------------------------------------------------------
    # Optional parameter defaults and validation
    # ------------------------------------------------------------------
    _ab = (
        np.zeros(3) if accel_bias is None
        else np.asarray(accel_bias, dtype=float)
    )
    _gb = (
        np.zeros(3) if gyro_bias is None
        else np.asarray(gyro_bias, dtype=float)
    )
    _g = (
        np.array([0.0, 0.0, -9.81]) if gravity is None
        else np.asarray(gravity, dtype=float)
    )
    _v0 = (
        np.zeros(3) if initial_velocity is None
        else np.asarray(initial_velocity, dtype=float)
    )

    for name, arr in [("accel_bias", _ab), ("gyro_bias", _gb),
                      ("gravity", _g), ("initial_velocity", _v0)]:
        if arr.shape != (3,):
            raise ValueError(
                f"{name} must have shape (3,), got {arr.shape}."
            )

    # ------------------------------------------------------------------
    # Build IMU trajectory
    # ------------------------------------------------------------------
    rotations, positions = _build_trajectory(
        imu_ts, a, w, _ab, _gb, _g, _v0
    )

    # ------------------------------------------------------------------
    # Build interpolators
    # ------------------------------------------------------------------
    slerp = _Slerp(imu_ts, _Rotation.from_matrix(rotations))

    # Clamp all query times to the valid IMU window.
    ref_clipped = float(np.clip(ref_time, imu_ts[0], imu_ts[-1]))
    pt_ts_clipped = np.clip(pt_ts, imu_ts[0], imu_ts[-1])

    # ------------------------------------------------------------------
    # Interpolate pose at ref_time
    # ------------------------------------------------------------------
    R_ref = slerp(ref_clipped).as_matrix()          # (3, 3)
    p_ref = np.array([
        np.interp(ref_clipped, imu_ts, positions[:, i])
        for i in range(3)
    ])                                               # (3,)

    # ------------------------------------------------------------------
    # Interpolate pose at each point's acquisition time
    # ------------------------------------------------------------------
    R_at_pts = slerp(pt_ts_clipped).as_matrix()     # (N, 3, 3)
    p_at_pts = np.column_stack([
        np.interp(pt_ts_clipped, imu_ts, positions[:, i])
        for i in range(3)
    ])                                               # (N, 3)

    # ------------------------------------------------------------------
    # Compute relative transforms: T_rel(t_i) = T_ref^{-1} ⊗ T(t_i)
    #
    #   R_rel[i] = R_ref.T  @  R_at_pts[i]
    #   t_rel[i] = R_ref.T  @  (p_at_pts[i] − p_ref)
    #
    # einsum 'ji,njk->nik' computes R_ref.T @ R_at_pts[n] for each n.
    # (p_at_pts − p_ref) @ R_ref  is equivalent to (R_ref.T @ Δp.T).T.
    # ------------------------------------------------------------------
    R_rel = np.einsum("ji,njk->nik", R_ref, R_at_pts)   # (N, 3, 3)
    t_rel = (p_at_pts - p_ref) @ R_ref                   # (N, 3)

    # ------------------------------------------------------------------
    # Apply deskewing: p_corrected[i] = R_rel[i] @ pts[i] + t_rel[i]
    # einsum 'nij,nj->ni' applies the per-point rotation matrix.
    # ------------------------------------------------------------------
    return np.einsum("nij,nj->ni", R_rel, pts) + t_rel
