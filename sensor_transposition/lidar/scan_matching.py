"""
lidar/scan_matching.py

Scan-matching utilities for LiDAR odometry: point-to-point ICP,
point-to-plane ICP, and per-point normal estimation.

Point-to-point ICP estimates the rigid-body transform (rotation + translation)
that best aligns a *source* point cloud to a *target* point cloud by
iteratively:

1. Finding the closest point in the target for each source point.
2. Rejecting correspondences beyond *max_correspondence_dist*.
3. Solving for the optimal transform via SVD (Kabsch algorithm).
4. Applying the incremental transform to the source cloud.
5. Repeating until the per-iteration change falls below *tolerance* or
   *max_iterations* is reached.

Point-to-plane ICP minimises the sum of squared distances from each source
point to the *tangent plane* of its nearest target point.  It converges faster
than point-to-point ICP on structured scenes (walls, floors, roads) and is the
standard variant used in production LiDAR SLAM systems (Cartographer, LOAM,
LIO-SAM).  Per-point normals on the target cloud are estimated by PCA on the
k-nearest neighbors.

Typical use-cases
-----------------
* **LiDAR frame-to-frame odometry**: align consecutive LiDAR scans to
  estimate relative ego motion between frames.
* **Map-to-scan registration**: align an incoming scan against an
  accumulated local map for localisation.

Examples::

    from sensor_transposition.lidar.scan_matching import (
        icp_align,
        icp_align_point_to_plane,
        point_cloud_normals,
    )

    # Point-to-point ICP
    result = icp_align(source_xyz, target_xyz, max_iterations=50)
    if result.converged:
        print("Transform:\\n", result.transform)
        print("MSE:", result.mean_squared_error)

    # Point-to-plane ICP (faster convergence on planar scenes)
    result = icp_align_point_to_plane(source_xyz, target_xyz, max_iterations=50)
    if result.converged:
        print("Point-to-plane transform:\\n", result.transform)

    # Compute per-point normals
    normals = point_cloud_normals(target_xyz, k=20)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# IcpResult
# ---------------------------------------------------------------------------


@dataclass
class IcpResult:
    """Result of an ICP scan-matching run.

    Attributes:
        transform: 4×4 homogeneous matrix that maps *source* points into the
            *target* frame: ``p_target ≈ transform @ p_source_h``.
        converged: ``True`` if the algorithm converged within
            *max_iterations*.
        num_iterations: Number of iterations actually performed.
        mean_squared_error: Mean squared point-to-point distance (in the
            units of the input clouds, typically metres) over the inlier
            correspondences from the final iteration.
    """

    transform: np.ndarray
    converged: bool
    num_iterations: int
    mean_squared_error: float

    def __repr__(self) -> str:
        return (
            f"IcpResult(converged={self.converged}, "
            f"mse={self.mean_squared_error:.6g}, "
            f"n_iter={self.num_iterations})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def icp_align(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_dist: float = float("inf"),
    initial_transform: np.ndarray | None = None,
    callback: Optional[Callable[[int, float], None]] = None,
) -> IcpResult:
    """Align *source* to *target* using point-to-point ICP.

    Both clouds must be ``(N, 3)`` or ``(M, 3)`` float arrays with at least
    one point each.  The returned transform maps *source* points into the
    *target* frame.

    Args:
        source: ``(N, 3)`` float array – the moving cloud to be aligned.
        target: ``(M, 3)`` float array – the fixed reference cloud.
        max_iterations: Maximum number of ICP iterations (default ``50``).
        tolerance: Convergence threshold on the change in mean squared error
            between successive iterations (default ``1e-6``).  Must be
            strictly positive.
        max_correspondence_dist: Reject source–target point pairs whose
            Euclidean distance exceeds this value (metres).  Defaults to
            ``inf`` (accept all correspondences).
        initial_transform: Optional 4×4 homogeneous matrix applied to
            *source* before the first iteration.  Defaults to the identity.
        callback: Optional callable invoked at the end of every iteration
            with ``(iteration: int, mean_squared_error: float)``.  Can be
            used to monitor progress or implement early stopping::

                result = icp_align(
                    src, tgt,
                    callback=lambda i, c: print(f"iter {i}: mse={c:.6f}"),
                )

    Returns:
        :class:`IcpResult` with the cumulative 4×4 transform, convergence
        flag, iteration count, and final mean squared error.

    Raises:
        ValueError: If *source* or *target* have the wrong shape, are empty,
            or if *max_iterations* is less than 1 or *tolerance* is not
            strictly positive.
    """
    src = np.asarray(source, dtype=float)
    tgt = np.asarray(target, dtype=float)

    _validate_cloud(src, "source")
    _validate_cloud(tgt, "target")

    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}.")
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be > 0, got {tolerance}.")

    # Initialise the cumulative transform.
    if initial_transform is not None:
        T_cum = np.asarray(initial_transform, dtype=float)
        if T_cum.shape != (4, 4):
            raise ValueError(
                f"initial_transform must be 4×4, got {T_cum.shape}."
            )
        # Apply the initial transform to the source cloud.
        src = _apply_transform(T_cum, src)
    else:
        T_cum = np.eye(4, dtype=float)

    tree = cKDTree(tgt)
    prev_mse = float("inf")
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        # Step 1 – nearest-neighbour correspondence.
        dists, indices = tree.query(src, workers=1)

        # Step 2 – correspondence rejection.
        mask = dists <= max_correspondence_dist
        if mask.sum() < 3:
            # Fewer than 3 correspondences: cannot solve for a rigid transform.
            break

        src_inliers = src[mask]
        tgt_inliers = tgt[indices[mask]]

        # Step 3 – solve for optimal R, t (Kabsch algorithm).
        R, t = _kabsch(src_inliers, tgt_inliers)

        # Step 4 – accumulate incremental transform.
        T_inc = _rt_to_matrix(R, t)
        T_cum = T_inc @ T_cum

        # Step 5 – apply incremental transform to source cloud.
        src = _apply_transform(T_inc, src)

        # Step 6 – check convergence.
        mse = float(np.mean(dists[mask] ** 2))
        if callback is not None:
            callback(iteration, mse)
        if abs(prev_mse - mse) < tolerance:
            converged = True
            break
        prev_mse = mse

    # Compute final MSE.
    dists_final, indices_final = tree.query(src, workers=1)
    mask_final = dists_final <= max_correspondence_dist
    if mask_final.sum() > 0:
        final_mse = float(np.mean(dists_final[mask_final] ** 2))
    else:
        final_mse = float("inf")

    return IcpResult(
        transform=T_cum,
        converged=converged,
        num_iterations=iteration,
        mean_squared_error=final_mse,
    )


def point_cloud_normals(cloud: np.ndarray, k: int = 20) -> np.ndarray:
    """Estimate per-point surface normals via PCA on k-nearest neighbours.

    For each point in *cloud* the k-nearest neighbours are found with a
    KD-tree query.  The normal is the eigenvector corresponding to the
    smallest eigenvalue of the 3×3 covariance matrix of those neighbours
    (i.e. the direction of *least* variance, which is perpendicular to the
    local surface).

    Normals are flipped so they all point toward the centroid of the full
    cloud (a simple global orientation heuristic suitable for indoor/outdoor
    scenes where the sensor is roughly centred).

    Args:
        cloud: ``(N, 3)`` float array of 3-D points.
        k: Number of nearest neighbours used for each normal estimate
            (default ``20``).  Must be at least 3.

    Returns:
        ``(N, 3)`` float array of unit normals, one per input point.

    Raises:
        ValueError: If *cloud* has the wrong shape, is empty, or *k* < 3.

    Example::

        normals = point_cloud_normals(target_xyz, k=20)
        # normals[i] is the estimated surface normal at target_xyz[i]
    """
    pts = np.asarray(cloud, dtype=float)
    _validate_cloud(pts, "cloud")
    if k < 3:
        raise ValueError(f"k must be >= 3 for PCA, got {k}.")

    n = pts.shape[0]
    # Clamp k so it never exceeds the number of available points.
    k_eff = min(k, n)

    tree = cKDTree(pts)
    # Include the point itself in the neighbour query.
    _, indices = tree.query(pts, k=k_eff, workers=1)

    # Fallback: if the cloud has fewer than 3 points PCA is ill-posed.
    if k_eff < 3:
        return np.tile([0.0, 0.0, 1.0], (n, 1))

    # Centroid used for global orientation.
    cloud_centroid = pts.mean(axis=0)

    normals = np.empty((n, 3), dtype=float)
    for i in range(n):
        nbrs = pts[indices[i]]          # (k_eff, 3)
        centred = nbrs - nbrs.mean(axis=0)
        cov = centred.T @ centred       # (3, 3)
        # Eigenvector of the smallest eigenvalue = surface normal.
        eigvals, eigvecs = np.linalg.eigh(cov)
        # np.linalg.eigh returns eigenvalues in ascending order, so index 0
        # is the smallest eigenvalue and its eigenvector is the surface normal.
        normal = eigvecs[:, 0]
        # Orient toward the cloud centroid.
        if np.dot(normal, cloud_centroid - pts[i]) < 0:
            normal = -normal
        normals[i] = normal

    return normals


def icp_align_point_to_plane(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_dist: float = float("inf"),
    initial_transform: np.ndarray | None = None,
    target_normals: np.ndarray | None = None,
    normals_k: int = 20,
    callback: Optional[Callable[[int, float], None]] = None,
) -> IcpResult:
    """Align *source* to *target* using point-to-plane ICP.

    Point-to-plane ICP minimises the sum of squared point-to-plane distances::

        Σ (nᵢ · (R·pᵢ + t − qᵢ))²

    where **nᵢ** is the surface normal at the nearest target point **qᵢ**,
    and **pᵢ** is the (current) source point.  At each iteration the
    non-linear objective is linearised by approximating R ≈ I + [ω]×, yielding
    a 6×6 linear system solved in closed form via ``numpy.linalg.solve``.

    This variant converges in roughly half as many iterations as point-to-point
    ICP on structured scenes (walls, floors, roads) and is the standard approach
    in production LiDAR SLAM systems (Cartographer, LOAM, LIO-SAM).

    Args:
        source: ``(N, 3)`` float array – the moving cloud to be aligned.
        target: ``(M, 3)`` float array – the fixed reference cloud.
        max_iterations: Maximum number of ICP iterations (default ``50``).
        tolerance: Convergence threshold on the change in mean squared
            point-to-plane distance between successive iterations
            (default ``1e-6``).  Must be strictly positive.
        max_correspondence_dist: Reject source–target point pairs whose
            Euclidean distance exceeds this value (metres).  Defaults to
            ``inf`` (accept all correspondences).
        initial_transform: Optional 4×4 homogeneous matrix applied to
            *source* before the first iteration.  Defaults to the identity.
        target_normals: Pre-computed ``(M, 3)`` unit normals for the target
            cloud.  If ``None`` (the default) normals are estimated
            automatically using :func:`point_cloud_normals` with *normals_k*
            neighbours.
        normals_k: Number of nearest neighbours for automatic normal
            estimation (ignored when *target_normals* is supplied).
        callback: Optional callable invoked at the end of every iteration
            with ``(iteration: int, mean_squared_error: float)``.

    Returns:
        :class:`IcpResult` with the cumulative 4×4 transform, convergence
        flag, iteration count, and final mean squared **point-to-point**
        distance (for comparability with :func:`icp_align`).

    Raises:
        ValueError: If the inputs have the wrong shape, if *max_iterations*
            is less than 1, or if *tolerance* is not strictly positive.

    Example::

        from sensor_transposition.lidar.scan_matching import (
            icp_align_point_to_plane,
        )

        result = icp_align_point_to_plane(source_xyz, target_xyz)
        if result.converged:
            R = result.transform[:3, :3]
            t = result.transform[:3, 3]
            aligned = (R @ source_xyz.T).T + t
    """
    src = np.asarray(source, dtype=float)
    tgt = np.asarray(target, dtype=float)

    _validate_cloud(src, "source")
    _validate_cloud(tgt, "target")

    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}.")
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be > 0, got {tolerance}.")

    # Compute or validate target normals.
    if target_normals is not None:
        tgt_normals = np.asarray(target_normals, dtype=float)
        if tgt_normals.shape != tgt.shape:
            raise ValueError(
                f"target_normals shape {tgt_normals.shape} must match "
                f"target shape {tgt.shape}."
            )
    else:
        tgt_normals = point_cloud_normals(tgt, k=normals_k)

    # Initialise the cumulative transform.
    if initial_transform is not None:
        T_cum = np.asarray(initial_transform, dtype=float)
        if T_cum.shape != (4, 4):
            raise ValueError(
                f"initial_transform must be 4×4, got {T_cum.shape}."
            )
        src = _apply_transform(T_cum, src)
    else:
        T_cum = np.eye(4, dtype=float)

    tree = cKDTree(tgt)
    prev_mse = float("inf")
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        # Nearest-neighbour correspondence.
        dists, indices = tree.query(src, workers=1)

        # Correspondence rejection.
        mask = dists <= max_correspondence_dist
        if mask.sum() < 6:
            # Need at least 6 inliers to solve the 6-DOF linear system.
            break

        src_in = src[mask]
        tgt_in = tgt[indices[mask]]
        n_in = tgt_normals[indices[mask]]   # (K, 3)

        # Build the 6×6 system for [ω, t] (linearised point-to-plane).
        # aᵢ = [(pᵢ × nᵢ), nᵢ],  bᵢ = nᵢ · (pᵢ − qᵢ)
        cross = np.cross(src_in, n_in)      # (K, 3)
        A_rows = np.hstack([cross, n_in])   # (K, 6)
        b_vec = np.einsum("ij,ij->i", n_in, src_in - tgt_in)  # (K,)

        ATA = A_rows.T @ A_rows             # (6, 6)
        ATb = A_rows.T @ b_vec              # (6,)

        try:
            x = np.linalg.solve(ATA, -ATb)
        except np.linalg.LinAlgError:
            break

        omega = x[:3]   # infinitesimal rotation vector
        t_inc = x[3:]   # translation

        # Convert infinitesimal rotation to a full rotation matrix.
        angle = np.linalg.norm(omega)
        if angle > 1e-12:
            axis = omega / angle
            K_mat = np.array([
                [0.0,      -axis[2],  axis[1]],
                [axis[2],   0.0,     -axis[0]],
                [-axis[1],  axis[0],  0.0],
            ])
            R_inc = (
                np.eye(3)
                + np.sin(angle) * K_mat
                + (1.0 - np.cos(angle)) * (K_mat @ K_mat)
            )
        else:
            R_inc = np.eye(3)

        T_inc = _rt_to_matrix(R_inc, t_inc)
        T_cum = T_inc @ T_cum
        src = _apply_transform(T_inc, src)

        # Convergence check on point-to-plane MSE.
        mse = float(np.mean(b_vec ** 2))
        if callback is not None:
            callback(iteration, mse)
        if abs(prev_mse - mse) < tolerance:
            converged = True
            break
        prev_mse = mse

    # Final point-to-point MSE for consistency with icp_align output.
    dists_final, _ = tree.query(src, workers=1)
    mask_final = dists_final <= max_correspondence_dist
    if mask_final.sum() > 0:
        final_mse = float(np.mean(dists_final[mask_final] ** 2))
    else:
        final_mse = float("inf")

    return IcpResult(
        transform=T_cum,
        converged=converged,
        num_iterations=iteration,
        mean_squared_error=final_mse,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_cloud(cloud: np.ndarray, name: str) -> None:
    """Raise ``ValueError`` if *cloud* is not an (N, 3) array with N >= 1."""
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(
            f"{name} must be an (N, 3) array, got shape {cloud.shape}."
        )
    if cloud.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one point.")


def _kabsch(src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch algorithm: find optimal rotation R and translation t.

    Minimises ``sum ||tgt_i - (R @ src_i + t)||²``.

    Args:
        src: ``(N, 3)`` inlier source points.
        tgt: ``(N, 3)`` corresponding inlier target points.

    Returns:
        Tuple ``(R, t)`` where ``R`` is a 3×3 rotation matrix and ``t`` is
        a (3,) translation vector.
    """
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)

    src_c = src - src_mean
    tgt_c = tgt - tgt_mean

    H = src_c.T @ tgt_c
    U, _, Vt = np.linalg.svd(H)

    # Ensure a proper rotation (det = +1); correct reflections.
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    t = tgt_mean - R @ src_mean
    return R, t


def _apply_transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4×4 homogeneous matrix to an ``(N, 3)`` point array."""
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ points.T).T + t


def _rt_to_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pack a 3×3 rotation and (3,) translation into a 4×4 matrix."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
