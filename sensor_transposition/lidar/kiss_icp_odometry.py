"""
lidar/kiss_icp_odometry.py

KISS-ICP (Keep It Small and Simple ICP) odometry module.

KISS-ICP is a minimal-parameter LiDAR odometry algorithm described in:

    Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Louis Wiesmann,
    Jens Behley, and Cyrill Stachniss.  "KISS-ICP: In Defense of
    Point-to-Point ICP – Simple, Accurate, and Robust Registration if
    Done the Right Way."  *IEEE Robotics and Automation Letters*, 2023.

The key ideas implemented here (in pure NumPy/SciPy, no additional
dependencies) are:

1. **Adaptive correspondence threshold** – instead of a fixed
   ``max_correspondence_dist``, KISS-ICP estimates the threshold
   automatically from the ICP residuals.  The running estimate
   ``σ̂`` is updated each frame via an exponential moving average of the
   mean point-to-point residuals, and the threshold is set to ``3σ̂``.
   This means the algorithm self-tunes to the sensor noise level without
   manual tuning.

2. **Voxel-hashed local map** – the odometry reference is not the
   immediately preceding scan but a *local map* of the last
   ``local_map_size`` keyframes.  Each frame is down-sampled to one
   representative point per voxel cell and inserted into a hash-map
   keyed by voxel index.  Older voxels are evicted to keep memory
   bounded.

3. **Point-to-point ICP with the adaptive threshold** – standard
   point-to-point ICP (Kabsch SVD) using the adaptive threshold as the
   correspondence-distance filter.

Usage::

    from sensor_transposition.lidar.kiss_icp_odometry import KissIcpOdometry

    odometry = KissIcpOdometry(voxel_size=0.5)

    for scan in lidar_scans:
        pose = odometry.register_frame(scan)
        print("Current pose:\\n", pose)          # 4×4 homogeneous matrix
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from sensor_transposition.lidar.scan_matching import _apply_transform, _kabsch, _rt_to_matrix


# ---------------------------------------------------------------------------
# VoxelHashMap
# ---------------------------------------------------------------------------


# Stride used to encode (ix, iy, iz) voxel indices into a single int64 key.
# 100_003 is a prime chosen large enough to avoid collisions for coordinate
# ranges typical in autonomous-driving and robotics (< ±500 m at any voxel
# size ≥ 0.01 m).
_VOXEL_STRIDE = np.array([1, 100_003, 100_003**2], dtype=np.int64)


class VoxelHashMap:
    """Voxel-hashed point-cloud local map.

    Stores at most one representative point per voxel cell in a Python
    ``dict`` keyed by the integer voxel index tuple.  This provides O(1)
    average-case insertion and lookup and constant memory per cell.

    Args:
        voxel_size: Edge length of each cubic voxel (metres).
        max_voxels: Maximum number of voxels to retain.  When the map
            reaches this capacity, the oldest-inserted voxels are evicted.
            Defaults to ``100 000``.
    """

    def __init__(self, voxel_size: float, max_voxels: int = 100_000) -> None:
        if voxel_size <= 0:
            raise ValueError(f"voxel_size must be > 0, got {voxel_size}.")
        self._voxel_size = float(voxel_size)
        self._max_voxels = max_voxels
        # Maps (ix, iy, iz) → representative point (3,)
        self._map: Dict[Tuple[int, int, int], np.ndarray] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _voxel_index(self, point: np.ndarray) -> Tuple[int, int, int]:
        """Convert a 3-D point to its integer voxel index."""
        idx = np.floor(point / self._voxel_size).astype(np.int64)
        return int(idx[0]), int(idx[1]), int(idx[2])

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_points(self, points: np.ndarray) -> None:
        """Insert *points* into the map, one representative per voxel.

        If a voxel is already occupied, the existing representative is
        kept (first-insertion wins).  When the map exceeds *max_voxels*
        after insertion, the oldest entries are evicted.

        Args:
            points: ``(N, 3)`` float array of 3-D points.
        """
        for pt in points:
            key = self._voxel_index(pt)
            if key not in self._map:
                self._map[key] = pt.copy()

        # Evict oldest entries when over capacity.
        while len(self._map) > self._max_voxels:
            oldest_key = next(iter(self._map))
            del self._map[oldest_key]

    def clear(self) -> None:
        """Remove all voxels from the map."""
        self._map.clear()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_points(self) -> np.ndarray:
        """Return all representative points as an ``(N, 3)`` array.

        Returns an empty ``(0, 3)`` array if the map is empty.
        """
        if not self._map:
            return np.empty((0, 3), dtype=float)
        return np.array(list(self._map.values()), dtype=float)

    def __len__(self) -> int:
        return len(self._map)


# ---------------------------------------------------------------------------
# AdaptiveThreshold
# ---------------------------------------------------------------------------


class AdaptiveThreshold:
    """Online estimator for the KISS-ICP adaptive correspondence threshold.

    The threshold is initialised to *initial_threshold* and updated each
    frame as an exponential moving average of the per-frame mean ICP
    residual.  The correspondence threshold applied to ICP is ``3 × σ̂``
    so that it adapts to the sensor noise level automatically.

    Args:
        initial_threshold: Starting threshold in metres (default ``2.0``).
        min_threshold: Minimum allowed threshold (default ``0.1 m``).
        max_threshold: Maximum allowed threshold (default ``10.0 m``).
        alpha: EMA smoothing factor in ``(0, 1]`` (default ``0.9``).
    """

    def __init__(
        self,
        initial_threshold: float = 2.0,
        min_threshold: float = 0.1,
        max_threshold: float = 10.0,
        alpha: float = 0.9,
    ) -> None:
        self._sigma: float = initial_threshold
        self._min = min_threshold
        self._max = max_threshold
        self._alpha = alpha

    @property
    def threshold(self) -> float:
        """Current correspondence threshold (3 × estimated σ̂)."""
        return float(np.clip(3.0 * self._sigma, self._min, self._max))

    def update(self, mean_residual: float) -> None:
        """Update the running σ̂ estimate with *mean_residual*.

        Args:
            mean_residual: Mean point-to-point ICP residual (RMS in metres)
                from the most recent frame registration.
        """
        self._sigma = self._alpha * self._sigma + (1.0 - self._alpha) * mean_residual


# ---------------------------------------------------------------------------
# KissIcpOdometry
# ---------------------------------------------------------------------------


class KissIcpOdometry:
    """KISS-ICP LiDAR odometry estimator.

    Registers each incoming scan against a voxel-hashed local map using
    point-to-point ICP with an adaptive correspondence-distance threshold.
    The current ego pose is accumulated as a 4×4 homogeneous matrix.

    Args:
        voxel_size: Voxel edge length for both input down-sampling and the
            local map (metres, default ``0.5``).
        max_iterations: Maximum ICP iterations per frame (default ``50``).
        tolerance: ICP convergence tolerance (default ``1e-4``).
        initial_threshold: Initial adaptive-threshold sigma in metres
            (default ``2.0``).
        min_threshold: Minimum correspondence threshold (metres).
        max_threshold: Maximum correspondence threshold (metres).
        local_map_max_voxels: Maximum voxels in the local map
            (default ``100 000``).

    Example::

        from sensor_transposition.lidar.kiss_icp_odometry import KissIcpOdometry

        odometry = KissIcpOdometry(voxel_size=0.5)

        poses = []
        for scan in lidar_scans:
            pose = odometry.register_frame(scan)
            poses.append(pose)
    """

    def __init__(
        self,
        *,
        voxel_size: float = 0.5,
        max_iterations: int = 50,
        tolerance: float = 1e-4,
        initial_threshold: float = 2.0,
        min_threshold: float = 0.1,
        max_threshold: float = 10.0,
        local_map_max_voxels: int = 100_000,
    ) -> None:
        if voxel_size <= 0:
            raise ValueError(f"voxel_size must be > 0, got {voxel_size}.")
        self._voxel_size = float(voxel_size)
        self._max_iterations = max_iterations
        self._tolerance = tolerance

        self._local_map = VoxelHashMap(
            voxel_size=voxel_size,
            max_voxels=local_map_max_voxels,
        )
        self._threshold = AdaptiveThreshold(
            initial_threshold=initial_threshold,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
        )

        # Accumulated ego pose (world ← sensor).
        self._pose: np.ndarray = np.eye(4, dtype=float)
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pose(self) -> np.ndarray:
        """Current ego pose as a 4×4 homogeneous matrix (world ← sensor)."""
        return self._pose.copy()

    @property
    def local_map(self) -> VoxelHashMap:
        """The internal :class:`VoxelHashMap` used as the odometry reference."""
        return self._local_map

    @property
    def adaptive_threshold(self) -> AdaptiveThreshold:
        """The :class:`AdaptiveThreshold` estimator."""
        return self._threshold

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def register_frame(self, scan: np.ndarray) -> np.ndarray:
        """Register one LiDAR scan and return the updated ego pose.

        The scan is first voxel-downsampled, then aligned to the current
        local map (or treated as the initial keyframe if the map is empty).
        The adaptive threshold is updated from the ICP residual, the
        local map is extended with the registered scan, and the cumulative
        pose is returned.

        Args:
            scan: ``(N, 3)`` float array of LiDAR points in the sensor frame.

        Returns:
            4×4 homogeneous matrix representing the current ego pose in the
            world frame (world ← sensor).
        """
        pts = np.asarray(scan, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 1:
            raise ValueError(
                f"scan must be an (N, 3) array with N >= 1, got shape {pts.shape}."
            )

        # 1. Voxel-downsample the incoming scan.
        downsampled = _voxel_downsample(pts, self._voxel_size)

        if self._frame_count == 0 or len(self._local_map) == 0:
            # First frame: insert into the local map, no ICP.
            self._local_map.add_points(downsampled)
            self._frame_count += 1
            return self._pose.copy()

        # 2. Run ICP against the local map.
        map_points = self._local_map.get_points()
        R_inc, t_inc, mean_residual = _icp_one_frame(
            source=downsampled,
            target=map_points,
            max_iterations=self._max_iterations,
            tolerance=self._tolerance,
            max_dist=self._threshold.threshold,
        )

        # 3. Update adaptive threshold.
        if mean_residual < float("inf"):
            self._threshold.update(float(np.sqrt(mean_residual)))

        # 4. Accumulate pose.
        T_inc = _rt_to_matrix(R_inc, t_inc)
        self._pose = self._pose @ np.linalg.inv(T_inc)

        # 5. Transform scan to world frame and add to local map.
        world_pts = _apply_transform(self._pose, downsampled)
        self._local_map.add_points(world_pts)

        self._frame_count += 1
        return self._pose.copy()

    def reset(self) -> None:
        """Reset the odometry state (pose, local map, and threshold)."""
        self._local_map.clear()
        self._pose = np.eye(4, dtype=float)
        self._frame_count = 0


# ---------------------------------------------------------------------------
# Module-level helpers (private)
# ---------------------------------------------------------------------------


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Down-sample *points* to one representative point per voxel.

    The representative is the centroid of all points within each voxel.

    Args:
        points: ``(N, 3)`` float array.
        voxel_size: Voxel edge length (metres).

    Returns:
        ``(M, 3)`` float array with M ≤ N.
    """
    indices = np.floor(points / voxel_size).astype(np.int64)
    # Encode (ix, iy, iz) as a single integer for efficient grouping using the
    # module-level prime stride to avoid key collisions.
    keys = indices @ _VOXEL_STRIDE

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    centroids = np.zeros((len(unique_keys), 3), dtype=float)
    counts = np.zeros(len(unique_keys), dtype=int)
    np.add.at(centroids, inverse, points)
    np.add.at(counts, inverse, 1)
    centroids /= counts[:, None]
    return centroids


def _icp_one_frame(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int,
    tolerance: float,
    max_dist: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Single ICP run returning (R, t, mean_squared_residual).

    A lightweight point-to-point ICP loop using the Kabsch algorithm.
    Returns the identity transform if fewer than 3 inlier correspondences
    are found.
    """
    src = source.copy()
    tree = cKDTree(target)
    R_cum = np.eye(3, dtype=float)
    t_cum = np.zeros(3, dtype=float)
    prev_mse = float("inf")

    for _ in range(max_iterations):
        dists, indices = tree.query(src, workers=1)
        mask = dists <= max_dist
        if mask.sum() < 3:
            break

        src_in = src[mask]
        tgt_in = target[indices[mask]]

        R_inc, t_inc = _kabsch(src_in, tgt_in)
        src = (R_inc @ src.T).T + t_inc
        R_cum = R_inc @ R_cum
        t_cum = R_inc @ t_cum + t_inc

        mse = float(np.mean(dists[mask] ** 2))
        if abs(prev_mse - mse) < tolerance:
            break
        prev_mse = mse

    # Compute final MSE.
    dists_f, _ = tree.query(src, workers=1)
    mask_f = dists_f <= max_dist
    final_mse = float(np.mean(dists_f[mask_f] ** 2)) if mask_f.sum() > 0 else float("inf")

    return R_cum, t_cum, final_mse
