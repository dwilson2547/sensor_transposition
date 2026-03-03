"""
lidar/scan_matching.py

Point-to-point Iterative Closest Point (ICP) scan matching for LiDAR odometry.

ICP estimates the rigid-body transform (rotation + translation) that best
aligns a *source* point cloud to a *target* point cloud by iteratively:

1. Finding the closest point in the target for each source point.
2. Rejecting correspondences beyond *max_correspondence_dist*.
3. Solving for the optimal transform via SVD (Kabsch algorithm).
4. Applying the incremental transform to the source cloud.
5. Repeating until the per-iteration change falls below *tolerance* or
   *max_iterations* is reached.

Typical use-case
----------------
* **LiDAR frame-to-frame odometry**: align consecutive LiDAR scans to
  estimate relative ego motion between frames.
* **Map-to-scan registration**: align an incoming scan against an
  accumulated local map for localisation.

Example::

    from sensor_transposition.lidar.scan_matching import icp_align

    result = icp_align(source_xyz, target_xyz, max_iterations=50)
    if result.converged:
        print("Transform:\\n", result.transform)
        print("MSE:", result.mean_squared_error)
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
