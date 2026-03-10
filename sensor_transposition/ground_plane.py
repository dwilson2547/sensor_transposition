"""
ground_plane.py

Ground plane segmentation utilities for LiDAR point clouds.

Three progressively more robust methods are provided:

**Height threshold**
    The simplest approach: any point whose Z coordinate (in a frame where the
    ground is at z ≈ 0) falls below *threshold* is classified as ground.
    Works well on flat, paved environments where the sensor height above the
    ground is known.

**RANSAC plane fitting**
    Fits a plane model using Random Sample Consensus.  Robust to outliers
    (walls, cars, trees) and works even when the true ground surface is
    slightly tilted or the sensor height is not precisely known.

**Normal-based filtering**
    Estimates a per-point surface normal via PCA on the k-nearest neighbours
    and classifies a point as ground when its normal is approximately vertical
    (dot product with +Z exceeds *verticality_threshold*).  Handles undulating
    terrain better than a single-plane RANSAC fit at the cost of higher
    computational expense.

All methods assume the point cloud is in a coordinate frame where the
**ground normal points toward +Z** (e.g. the FLU ego frame or the ENU frame).

See ``docs/ground_plane_identification.md`` for a detailed guide with worked
examples.

Typical usage::

    import numpy as np
    from sensor_transposition.ground_plane import (
        height_threshold_segment,
        ransac_ground_plane,
        normal_based_segment,
    )

    xyz = np.load("frame.npy")  # (N, 3) array in the ego frame

    # Fast height-threshold segmentation
    ground_mask, non_ground_mask = height_threshold_segment(xyz, threshold=0.3)

    # RANSAC plane fitting (more robust)
    ground_mask, plane = ransac_ground_plane(xyz, distance_threshold=0.2)
    a, b, c, d = plane  # ax + by + cz + d = 0

    # Normal-based segmentation (best for uneven terrain)
    ground_mask, normals = normal_based_segment(xyz, k=20)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Method 1 – Height Threshold
# ---------------------------------------------------------------------------


def height_threshold_segment(
    cloud: np.ndarray,
    threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment ground points using a simple Z-axis height threshold.

    Any point whose Z coordinate is less than *threshold* is classified as
    ground.  The method assumes the point cloud is in a frame where the ground
    surface lies at z ≈ 0 and points above ground have z > 0 (e.g. the FLU
    ego frame).

    Args:
        cloud: ``(N, 3)`` float array of 3-D points.
        threshold: Maximum Z value (in metres) for a point to be classified as
            ground.  Defaults to ``0.3`` m.

    Returns:
        ground_mask: ``(N,)`` boolean array — ``True`` for ground points.
        non_ground_mask: ``(N,)`` boolean array — ``True`` for non-ground
            points (complement of *ground_mask*).

    Example::

        ground_mask, non_ground_mask = height_threshold_segment(xyz, threshold=0.3)
        ground_points = xyz[ground_mask]
        obstacles     = xyz[non_ground_mask]
    """
    cloud = np.asarray(cloud, dtype=float)
    ground_mask = cloud[:, 2] < threshold
    return ground_mask, ~ground_mask


# ---------------------------------------------------------------------------
# Method 2 – RANSAC Plane Fitting
# ---------------------------------------------------------------------------


def ransac_ground_plane(
    cloud: np.ndarray,
    distance_threshold: float = 0.2,
    max_iterations: int = 1000,
    normal_threshold: float = 0.9,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment ground points using RANSAC plane fitting.

    Iteratively samples three random points, fits a plane, and counts inliers
    (points within *distance_threshold* of the plane).  A *normal_threshold*
    guard rejects planes whose normal deviates too far from the +Z axis,
    preventing walls or vertical structures from being mistaken for the ground.

    Args:
        cloud: ``(N, 3)`` float array of 3-D points in a frame where the
            ground normal is approximately ``+Z``.
        distance_threshold: Maximum distance from the plane (in metres) for a
            point to be counted as an inlier.  Defaults to ``0.2`` m.
        max_iterations: Number of RANSAC iterations.  More iterations improve
            robustness at the cost of speed.  Defaults to ``1000``.
        normal_threshold: Minimum dot product between the candidate plane
            normal and the ``+Z`` axis.  Planes whose normal has a Z component
            below this value are rejected as non-horizontal.  Defaults to
            ``0.9`` (≈ 26° from vertical).
        rng: Optional :class:`numpy.random.Generator` for reproducibility.
            If ``None``, a default generator is created.

    Returns:
        ground_mask: ``(N,)`` boolean array — ``True`` for ground inliers of
            the best-fit plane.
        plane_coefficients: ``(4,)`` float array ``[a, b, c, d]`` of the best
            plane equation ``ax + by + cz + d = 0`` where ``(a, b, c)`` is the
            unit normal (oriented toward ``+Z``).

    Example::

        ground_mask, plane = ransac_ground_plane(xyz, distance_threshold=0.2)
        a, b, c, d = plane
        print(f"Ground plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
    """
    cloud = np.asarray(cloud, dtype=float)
    n = len(cloud)
    if n < 3:
        return np.zeros(n, dtype=bool), np.zeros(4)

    if rng is None:
        rng = np.random.default_rng()

    best_inliers = np.zeros(n, dtype=bool)
    best_plane = np.zeros(4)

    for _ in range(max_iterations):
        # 1. Sample three random points
        idx = rng.choice(n, size=3, replace=False)
        p1, p2, p3 = cloud[idx]

        # 2. Compute the plane normal via cross product
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm

        # Ensure the normal points upward (+Z)
        if normal[2] < 0:
            normal = -normal

        # 3. Reject planes whose normal deviates too far from +Z
        if normal[2] < normal_threshold:
            continue

        # 4. Compute signed distances from all points to the plane
        plane_offset = -float(np.dot(normal, p1))
        distances = np.abs(cloud @ normal + plane_offset)

        # 5. Count inliers and keep the best model
        inliers = distances < distance_threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_plane = np.array([normal[0], normal[1], normal[2], plane_offset])

    return best_inliers, best_plane


# ---------------------------------------------------------------------------
# Method 3 – Normal-Based Filtering
# ---------------------------------------------------------------------------


def normal_based_segment(
    cloud: np.ndarray,
    k: int = 20,
    verticality_threshold: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment ground points by classifying per-point surface normals.

    Estimates a surface normal at each point using PCA on its *k* nearest
    neighbours.  A point is classified as ground when the dot product between
    its normal and the ``+Z`` axis exceeds *verticality_threshold*, i.e. the
    local surface is approximately horizontal.

    This method handles undulating terrain better than a single-plane RANSAC
    fit, because it evaluates ground-ness locally rather than globally.

    Args:
        cloud: ``(N, 3)`` float array of 3-D points in a frame where ground
            normals point approximately toward ``+Z``.
        k: Number of nearest neighbours to use for normal estimation.
            Defaults to ``20``.
        verticality_threshold: Minimum dot product of the estimated normal
            with ``+Z`` for a point to be classified as ground.  Must be in
            ``(0, 1]``.  Defaults to ``0.85`` (≈ 32° from vertical).

    Returns:
        ground_mask: ``(N,)`` boolean array — ``True`` for ground points.
        normals: ``(N, 3)`` float array of estimated unit normal vectors
            (oriented toward ``+Z``).

    Example::

        ground_mask, normals = normal_based_segment(xyz, k=20)
        ground_points = xyz[ground_mask]
    """
    cloud = np.asarray(cloud, dtype=float)
    n = len(cloud)
    normals = np.empty((n, 3), dtype=float)

    if n == 0:
        return np.zeros(n, dtype=bool), normals

    # Clamp k to the number of available points
    k_actual = min(k, n)

    tree = cKDTree(cloud)
    _, idx = tree.query(cloud, k=k_actual)

    for i in range(n):
        local = cloud[idx[i]]
        cov = np.cov(local, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Smallest eigenvalue → eigvecs[:, 0] is the surface normal
        normal = eigvecs[:, 0]
        # Orient toward +Z
        if normal[2] < 0:
            normal = -normal
        normals[i] = normal

    # Ground points have normals approximately equal to [0, 0, 1]
    verticality = normals[:, 2]
    ground_mask = verticality > verticality_threshold

    return ground_mask, normals
