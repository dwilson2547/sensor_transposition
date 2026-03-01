"""
radar/radar_odometry.py

Radar odometry using two complementary approaches:

1. **Doppler-based ego-velocity estimation** – each FMCW radar detection
   provides a radial (line-of-sight) velocity measurement.  With multiple
   simultaneous detections the 3-D sensor ego-velocity can be recovered by
   least squares.

2. **Scan-to-scan ICP matching** – consecutive radar Cartesian point clouds
   (produced by :meth:`RadarParser.xyz`) are aligned with point-to-point ICP
   (reuses :func:`~sensor_transposition.lidar.scan_matching.icp_align`).
   The resulting 4×4 rigid transform represents the relative ego-motion
   between two frames.

Doppler model
-------------
For a stationary target the measured radial velocity equals the projection of
the *negative* sensor velocity onto the unit direction vector from the sensor
to the target::

    velocity_i = -(d_i · v_ego)

where ``d_i`` is the unit direction vector, ``v_ego`` is the 3-D sensor
velocity in m/s, and ``velocity_i`` is the signed Doppler reading (negative
means the target is approaching, i.e. the sensor is moving towards it).

Rearranging gives the over-determined linear system solved by
:func:`estimate_ego_velocity`::

    D @ v_ego = -velocity           (D is the (N, 3) direction matrix)

Typical use-cases
-----------------
* **Ego-motion prior** – supply ``v_ego`` as a motion constraint for an EKF
  or pose-graph node.
* **Weather-robust odometry** – radar penetrates fog, rain, and dust where
  cameras and LiDARs degrade.
* **Complementary fusion** – combine the Doppler velocity estimate with
  wheel odometry or IMU integration via the existing :class:`ImuEkf`.

Example – Doppler velocity::

    from sensor_transposition.radar import RadarParser
    from sensor_transposition.radar.radar_odometry import estimate_ego_velocity

    detections = RadarParser("frame.bin").read()
    result = estimate_ego_velocity(detections, min_snr=10.0)
    if result.valid:
        print("Ego velocity (m/s):", result.velocity)

Example – scan matching::

    from sensor_transposition.radar import RadarParser
    from sensor_transposition.radar.radar_odometry import RadarOdometer

    odom = RadarOdometer()
    for path in sorted(frame_paths):
        xyz = RadarParser(path).xyz()
        result = odom.add_frame(xyz, timestamp)
        if result is not None:
            print("Transform:\\n", result.transform)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sensor_transposition.lidar.scan_matching import IcpResult, icp_align
from sensor_transposition.radar.radar import RADAR_DETECTION_DTYPE


# ---------------------------------------------------------------------------
# EgoVelocityResult
# ---------------------------------------------------------------------------


@dataclass
class EgoVelocityResult:
    """Result of a Doppler-based ego-velocity estimation.

    Attributes:
        velocity: Estimated 3-D sensor velocity vector ``[vx, vy, vz]`` in
            m/s in the sensor frame (x = forward, y = left, z = up).
        residuals: Per-detection residuals ``(D @ velocity + measured_velocity)``
            in m/s.  Empty array if the estimation was not performed.
        num_inliers: Number of detections used in the final least-squares fit.
        valid: ``True`` if there were enough detections to solve for a
            velocity estimate.
    """

    velocity: np.ndarray
    residuals: np.ndarray
    num_inliers: int
    valid: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_ego_velocity(
    detections: np.ndarray,
    *,
    min_snr: float = 0.0,
    min_inliers: int = 3,
) -> EgoVelocityResult:
    """Estimate the sensor ego-velocity from Doppler radar detections.

    Uses a least-squares solution of the overdetermined linear system::

        D @ v_ego = -velocity

    where ``D`` is the ``(N, 3)`` matrix of unit direction vectors to each
    detection and ``velocity`` is the ``(N,)`` array of measured radial
    velocities (negative = approaching).

    Args:
        detections: Structured numpy array with dtype
            :data:`~sensor_transposition.radar.radar.RADAR_DETECTION_DTYPE`
            (fields ``range``, ``azimuth``, ``elevation``, ``velocity``,
            ``snr``).  Typically the output of :meth:`RadarParser.read`.
        min_snr: Minimum signal-to-noise ratio (dB) for a detection to be
            included in the estimation.  Defaults to ``0.0`` (accept all).
        min_inliers: Minimum number of detections required to attempt the
            estimate.  Defaults to ``3`` (the minimum for a well-conditioned
            3-D system).

    Returns:
        :class:`EgoVelocityResult` with the estimated velocity, per-detection
        residuals, inlier count, and a validity flag.

    Raises:
        ValueError: If *detections* is not a structured array with the
            expected dtype.
    """
    if detections.dtype != RADAR_DETECTION_DTYPE:
        raise ValueError(
            f"detections must have dtype RADAR_DETECTION_DTYPE, "
            f"got {detections.dtype}."
        )

    # SNR filter.
    mask = detections["snr"] >= min_snr
    filtered = detections[mask]

    if len(filtered) < min_inliers:
        return EgoVelocityResult(
            velocity=np.zeros(3),
            residuals=np.empty(0),
            num_inliers=0,
            valid=False,
        )

    # Build direction matrix D: each row is the unit vector [dx, dy, dz]
    # from the sensor to the detection in the sensor Cartesian frame.
    az_rad = np.radians(filtered["azimuth"].astype(float))
    el_rad = np.radians(filtered["elevation"].astype(float))
    cos_el = np.cos(el_rad)
    dx = cos_el * np.cos(az_rad)
    dy = cos_el * np.sin(az_rad)
    dz = np.sin(el_rad)
    D = np.column_stack([dx, dy, dz])          # (N, 3)

    # Doppler model: D @ v_ego = -velocity
    rhs = -filtered["velocity"].astype(float)  # (N,)

    # Least-squares estimate (ignore residuals, rank, and singular values).
    v_ego, _lstsq_res, _lstsq_rank, _lstsq_sv = np.linalg.lstsq(D, rhs, rcond=None)

    residuals = D @ v_ego - rhs

    return EgoVelocityResult(
        velocity=v_ego,
        residuals=residuals,
        num_inliers=int(len(filtered)),
        valid=True,
    )


def radar_scan_match(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    *,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_dist: float = float("inf"),
    initial_transform: np.ndarray | None = None,
) -> IcpResult:
    """Align two radar Cartesian point clouds using point-to-point ICP.

    This is a thin wrapper around
    :func:`~sensor_transposition.lidar.scan_matching.icp_align` that is
    provided as a named entry-point for radar odometry pipelines.  The
    underlying algorithm is identical; see :func:`icp_align` for full
    parameter documentation.

    Args:
        source_xyz: ``(N, 3)`` float array – the moving radar point cloud
            (e.g. the current frame).
        target_xyz: ``(M, 3)`` float array – the fixed reference radar point
            cloud (e.g. the previous frame or a local map).
        max_iterations: Maximum number of ICP iterations (default ``50``).
        tolerance: Convergence threshold on the change in MSE between
            successive iterations (default ``1e-6``).
        max_correspondence_dist: Reject source–target pairs whose Euclidean
            distance exceeds this value in metres.  Defaults to ``inf``.
        initial_transform: Optional 4×4 homogeneous matrix applied to
            *source* before the first ICP iteration.

    Returns:
        :class:`~sensor_transposition.lidar.scan_matching.IcpResult` with the
        cumulative 4×4 transform, convergence flag, iteration count, and final
        mean squared error.
    """
    return icp_align(
        source_xyz,
        target_xyz,
        max_iterations=max_iterations,
        tolerance=tolerance,
        max_correspondence_dist=max_correspondence_dist,
        initial_transform=initial_transform,
    )


# ---------------------------------------------------------------------------
# RadarOdometer
# ---------------------------------------------------------------------------


@dataclass
class _RadarFrame:
    """Internal storage for a single radar frame."""
    xyz: np.ndarray      # (N, 3) Cartesian points
    timestamp: float     # seconds


class RadarOdometer:
    """Frame-to-frame radar odometry using ICP scan matching.

    The odometer maintains a running world-frame pose (4×4 homogeneous matrix)
    by aligning each new radar frame against the previous one with
    point-to-point ICP.

    Args:
        max_iterations: ICP iteration limit per frame pair (default ``50``).
        tolerance: ICP convergence tolerance (default ``1e-6``).
        max_correspondence_dist: Maximum point-pair distance accepted by ICP
            in metres.  Defaults to ``inf`` (accept all).

    Example::

        odom = RadarOdometer(max_correspondence_dist=5.0)
        for xyz, ts in zip(frames_xyz, timestamps):
            result = odom.add_frame(xyz, ts)
            if result is not None:
                print("relative transform:", result.transform)
        print("world pose:\\n", odom.pose)
    """

    def __init__(
        self,
        *,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        max_correspondence_dist: float = float("inf"),
    ) -> None:
        if max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1, got {max_iterations}."
            )
        if tolerance <= 0.0:
            raise ValueError(f"tolerance must be > 0, got {tolerance}.")

        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._max_correspondence_dist = max_correspondence_dist

        self._prev_frame: _RadarFrame | None = None
        self._pose: np.ndarray = np.eye(4, dtype=float)
        self._transforms: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_frame(
        self,
        xyz: np.ndarray,
        timestamp: float,
    ) -> IcpResult | None:
        """Add a new radar frame and update the odometry estimate.

        On the first call the frame is stored as the reference and ``None``
        is returned.  On subsequent calls the new frame is aligned against
        the previous one; the resulting relative transform is composed with
        the accumulated world-frame pose.

        Args:
            xyz: ``(N, 3)`` float array of Cartesian radar points (typically
                from :meth:`~sensor_transposition.radar.RadarParser.xyz`).
            timestamp: Frame timestamp in seconds.

        Returns:
            :class:`~sensor_transposition.lidar.scan_matching.IcpResult` for
            the current pair, or ``None`` if this is the first frame.

        Raises:
            ValueError: If *xyz* has the wrong shape.
        """
        xyz = np.asarray(xyz, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(
                f"xyz must be an (N, 3) array, got shape {xyz.shape}."
            )
        if xyz.shape[0] < 1:
            raise ValueError("xyz must contain at least one point.")

        frame = _RadarFrame(xyz=xyz, timestamp=timestamp)

        if self._prev_frame is None:
            self._prev_frame = frame
            return None

        # Align previous frame (source) against current frame (target) to
        # obtain the relative motion transform T that maps previous-frame
        # points into the current frame: p_cur ≈ T @ p_prev.
        # Accumulating this gives the forward world-frame pose.
        result = icp_align(
            self._prev_frame.xyz,
            frame.xyz,
            max_iterations=self._max_iterations,
            tolerance=self._tolerance,
            max_correspondence_dist=self._max_correspondence_dist,
        )

        # Accumulate pose: T_world_cur = T_world_prev * T_prev_cur
        self._pose = self._pose @ result.transform
        self._transforms.append(result.transform.copy())
        self._prev_frame = frame

        return result

    def reset(self) -> None:
        """Reset the odometer to its initial state (identity pose, no frames)."""
        self._prev_frame = None
        self._pose = np.eye(4, dtype=float)
        self._transforms.clear()

    @property
    def pose(self) -> np.ndarray:
        """Current world-frame pose as a 4×4 homogeneous matrix."""
        return self._pose.copy()

    @property
    def transforms(self) -> list[np.ndarray]:
        """List of per-frame-pair relative transforms (4×4 matrices)."""
        return [T.copy() for T in self._transforms]


# ---------------------------------------------------------------------------
# Convenience functional API
# ---------------------------------------------------------------------------


def integrate_radar_odometry(
    frames_xyz: list[np.ndarray],
    timestamps: np.ndarray | list[float],
    *,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_dist: float = float("inf"),
) -> tuple[np.ndarray, list[IcpResult]]:
    """Integrate a sequence of radar frames into a world-frame pose.

    Args:
        frames_xyz: List of ``(N_i, 3)`` float arrays, one per frame.
        timestamps: 1-D array (or list) of frame timestamps in seconds.
            Must have the same length as *frames_xyz*.
        max_iterations: ICP iteration limit per frame pair (default ``50``).
        tolerance: ICP convergence tolerance (default ``1e-6``).
        max_correspondence_dist: Maximum ICP correspondence distance in metres.
            Defaults to ``inf``.

    Returns:
        A tuple ``(pose, icp_results)`` where *pose* is the final 4×4
        world-frame pose and *icp_results* is a list of
        :class:`~sensor_transposition.lidar.scan_matching.IcpResult` objects,
        one per consecutive frame pair.

    Raises:
        ValueError: If *frames_xyz* and *timestamps* have different lengths,
            or if fewer than two frames are provided.
    """
    ts = list(timestamps)
    if len(frames_xyz) != len(ts):
        raise ValueError(
            f"frames_xyz and timestamps must have the same length, "
            f"got {len(frames_xyz)} and {len(ts)}."
        )
    if len(frames_xyz) < 2:
        raise ValueError(
            "At least 2 frames are required for odometry integration."
        )

    odom = RadarOdometer(
        max_iterations=max_iterations,
        tolerance=tolerance,
        max_correspondence_dist=max_correspondence_dist,
    )
    icp_results: list[IcpResult] = []

    for xyz, timestamp in zip(frames_xyz, ts):
        result = odom.add_frame(xyz, timestamp)
        if result is not None:
            icp_results.append(result)

    return odom.pose, icp_results
