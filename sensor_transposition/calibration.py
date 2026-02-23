"""
calibration.py

Camera–LiDAR target-based extrinsic calibration.

The calibration workflow requires a planar calibration target (e.g. a
checkerboard or ArUco board) that is visible simultaneously in both the camera
image and the LiDAR point cloud.  For each target pose you collect one
*observation* consisting of:

Camera side
-----------
Use any PnP solver (e.g. ``cv2.solvePnP``) to obtain the board's 3-D normal
vector and its signed plane distance from the camera origin in the camera
frame.

LiDAR side
----------
Use :func:`ransac_plane` (or :func:`fit_plane` after manually cropping the
board region) to extract the board plane from the point cloud, obtaining the
normal vector and signed distance in the LiDAR frame.

Solve
-----
Collect N ≥ 3 such observations and pass them to
:func:`calibrate_lidar_camera`, which returns the 4×4 rigid transform
**T** (LiDAR → camera) using a closed-form SVD-based algorithm.

See ``docs/camera_lidar_extrinsic_calibration.md`` for a complete worked
example.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Plane fitting
# ---------------------------------------------------------------------------


def fit_plane(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit a plane to a set of 3-D points using PCA (SVD).

    The plane is found by computing the centroid and performing SVD on the
    mean-centred point matrix.  The right singular vector corresponding to the
    *smallest* singular value is the plane normal.

    Sign convention: the returned normal is flipped (if necessary) so that the
    signed distance ``distance = normal · centroid`` is non-negative.  This
    makes the sign deterministic and independent of point ordering.

    Args:
        points: ``(N, 3)`` float array of 3-D points (N ≥ 3).

    Returns:
        normal: Unit normal vector ``(3,)`` of the fitted plane.
        distance: Signed distance from the world origin to the plane along
            *normal*, defined as ``normal · centroid`` (always ≥ 0).

    Raises:
        ValueError: If *points* has fewer than 3 rows, is not ``(N, 3)``, or
            the points are collinear/coincident.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be shape (N, 3), got {pts.shape}.")
    if pts.shape[0] < 3:
        raise ValueError(
            f"At least 3 points are required to fit a plane, got {pts.shape[0]}."
        )

    centroid = pts.mean(axis=0)
    centered = pts - centroid

    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]  # row corresponding to smallest singular value

    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        raise ValueError(
            "Cannot determine plane normal: points appear to be collinear or coincident."
        )
    normal = normal / norm

    distance = float(np.dot(normal, centroid))

    # Enforce non-negative distance for a deterministic sign convention.
    if distance < 0.0:
        normal = -normal
        distance = -distance

    return normal, distance


def ransac_plane(
    points: np.ndarray,
    distance_threshold: float = 0.05,
    max_iterations: int = 100,
    min_inliers: int = 10,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Robustly fit a plane to noisy 3-D points using RANSAC.

    On each iteration, three random points are sampled to define a candidate
    plane.  All points within *distance_threshold* of that plane are counted
    as inliers.  The candidate with the most inliers is retained, then refined
    by calling :func:`fit_plane` on all inliers.

    Args:
        points: ``(N, 3)`` float array of 3-D points (N ≥ 3).
        distance_threshold: Maximum perpendicular distance (metres) from a
            point to the plane for the point to count as an inlier.
            Default: ``0.05`` m.
        max_iterations: Number of RANSAC iterations.  Default: ``100``.
        min_inliers: Minimum number of inliers required for a plane hypothesis
            to be accepted.  Default: ``10``.
        rng: Random number generator seed or ``np.random.Generator`` instance
            for reproducibility.  Pass an integer seed for a fixed result.
            If ``None``, a fresh generator with an unpredictable seed is used.

    Returns:
        normal: Unit normal vector ``(3,)`` of the best-fit plane (with the
            same non-negative-distance sign convention as :func:`fit_plane`).
        distance: Signed distance from the world origin to the plane (≥ 0).
        inlier_mask: ``(N,)`` boolean array; ``True`` for inlier points.

    Raises:
        ValueError: If *points* is not ``(N, 3)``, has fewer than 3 rows, or
            no plane with at least *min_inliers* inliers is found within
            *max_iterations*.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be shape (N, 3), got {pts.shape}.")
    n = pts.shape[0]
    if n < 3:
        raise ValueError(f"At least 3 points are required, got {n}.")

    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    elif rng is None:
        rng = np.random.default_rng()

    best_count = 0
    best_normal: Optional[np.ndarray] = None
    best_distance: float = 0.0
    best_mask: Optional[np.ndarray] = None

    for _ in range(max_iterations):
        idx = rng.choice(n, size=3, replace=False)
        sample = pts[idx]

        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        candidate_normal = np.cross(v1, v2)
        norm = np.linalg.norm(candidate_normal)
        if norm < 1e-12:
            continue  # collinear sample
        candidate_normal = candidate_normal / norm
        d = float(np.dot(candidate_normal, sample[0]))

        perp_dists = np.abs(pts @ candidate_normal - d)
        mask = perp_dists <= distance_threshold
        count = int(mask.sum())

        if count > best_count:
            best_count = count
            best_normal = candidate_normal
            best_distance = d
            best_mask = mask

    if best_normal is None or best_count < min_inliers:
        raise ValueError(
            f"RANSAC failed: best plane has {best_count} inlier(s) "
            f"(minimum required: {min_inliers}). "
            "Consider increasing max_iterations or distance_threshold, "
            "or reducing min_inliers."
        )

    # Refine by fitting a plane through all inliers.
    inlier_pts = pts[best_mask]
    if len(inlier_pts) >= 3:
        best_normal, best_distance = fit_plane(inlier_pts)

    # Recompute inlier mask using the refined plane.
    perp_dists = np.abs(pts @ best_normal - best_distance)
    best_mask = perp_dists <= distance_threshold

    return best_normal, best_distance, best_mask


# ---------------------------------------------------------------------------
# Extrinsic calibration
# ---------------------------------------------------------------------------


def calibrate_lidar_camera(
    lidar_normals: np.ndarray,
    lidar_distances: np.ndarray,
    camera_normals: np.ndarray,
    camera_distances: np.ndarray,
) -> np.ndarray:
    """Estimate the LiDAR-to-camera extrinsic transform from plane correspondences.

    Given N observations of a planar calibration target, each described by a
    unit normal vector and signed plane distance in the LiDAR frame **and** in
    the camera frame, this function returns the 4×4 rigid transform **T** such
    that::

        p_camera = T @ p_lidar          (homogeneous coordinates)

    The algorithm uses two closed-form steps (Geiger *et al.*, 2012):

    1. **Rotation** – solved with an SVD-based orthogonal Procrustes fit that
       maps the LiDAR normals onto the camera normals.

    2. **Translation** – once the rotation R is known, the plane equation
       constraint provides one scalar equation per observation::

           camera_normal_i · t  =  camera_distance_i − lidar_distance_i

       These N equations form a linear system that is solved in the
       least-squares sense with ``numpy.linalg.lstsq``.

    **Normal sign convention**: for each observation, both normals must point
    to *the same side* of the calibration board (typically, toward their
    respective sensor).  Use :func:`fit_plane` or :func:`ransac_plane` for the
    LiDAR side; see ``docs/camera_lidar_extrinsic_calibration.md`` for the
    camera-side procedure with OpenCV.

    Args:
        lidar_normals: ``(N, 3)`` float array of unit normal vectors for each
            target observation in the LiDAR frame.  N ≥ 3.
        lidar_distances: ``(N,)`` float array of signed plane distances in the
            LiDAR frame (as returned by :func:`fit_plane` or
            :func:`ransac_plane`).
        camera_normals: ``(N, 3)`` float array of unit normal vectors for the
            same N observations in the camera frame.
        camera_distances: ``(N,)`` float array of signed plane distances in the
            camera frame.

    Returns:
        T: ``(4, 4)`` float array — the rigid transform from the LiDAR frame
            to the camera frame.

    Raises:
        ValueError: If input arrays have incompatible shapes, or if N < 3.
    """
    nl = np.asarray(lidar_normals, dtype=float)
    dl = np.asarray(lidar_distances, dtype=float)
    nc = np.asarray(camera_normals, dtype=float)
    dc = np.asarray(camera_distances, dtype=float)

    if nl.ndim != 2 or nl.shape[1] != 3:
        raise ValueError(f"lidar_normals must be shape (N, 3), got {nl.shape}.")
    if nc.ndim != 2 or nc.shape[1] != 3:
        raise ValueError(f"camera_normals must be shape (N, 3), got {nc.shape}.")
    n = nl.shape[0]
    if n < 3:
        raise ValueError(
            f"At least 3 plane correspondences are required, got {n}."
        )
    if nc.shape[0] != n:
        raise ValueError(
            f"lidar_normals and camera_normals must have the same number of rows, "
            f"got {n} and {nc.shape[0]}."
        )
    if dl.shape != (n,):
        raise ValueError(f"lidar_distances must be shape ({n},), got {dl.shape}.")
    if dc.shape != (n,):
        raise ValueError(f"camera_distances must be shape ({n},), got {dc.shape}.")

    # ── Step 1: Rotation via SVD Procrustes ───────────────────────────────────
    # Find R that minimises  sum_i ||R @ nl_i - nc_i||^2.
    # Equivalent to maximising  trace(R @ H)  where H = sum_i nl_i nc_i^T.
    H = nl.T @ nc  # (3, 3)
    U, _S, Vt = np.linalg.svd(H)
    # Enforce det(R) = +1 (proper rotation, not reflection).
    det_sign = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, det_sign]) @ U.T  # (3, 3)

    # ── Step 2: Translation via least-squares ─────────────────────────────────
    # Plane-equation constraint: nc_i · t = dc_i - dl_i  for all i.
    b = dc - dl  # (N,)
    t, _, _, _ = np.linalg.lstsq(nc, b, rcond=None)  # (3,)

    # ── Assemble 4×4 transform ────────────────────────────────────────────────
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
