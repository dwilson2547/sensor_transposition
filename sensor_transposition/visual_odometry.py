"""
visual_odometry.py

Monocular visual odometry primitives: essential-matrix estimation, camera-pose
recovery, and Perspective-n-Point (PnP) pose estimation.

These functions form the core geometry back-end of a monocular (or stereo)
visual odometry pipeline:

1. **Essential matrix estimation** – given a set of matched pixel coordinates
   from two camera views (same intrinsic matrix), compute the essential matrix
   *E* via the normalised 8-point algorithm wrapped in RANSAC for robustness to
   outliers.

2. **Pose recovery from E** – decompose *E* into the four (R, t) candidate
   solutions and select the physically valid solution by requiring the majority
   of triangulated points to lie in front of both cameras.

3. **Perspective-n-Point (PnP) solver** – given a set of 3-D world points and
   their corresponding 2-D pixel observations in a new view, estimate the
   6-DOF camera pose via the Direct Linear Transform (DLT), with an optional
   RANSAC outer loop for robustness.

All functions operate on plain NumPy arrays and depend only on ``numpy`` and
``scipy`` (already required by ``sensor_transposition``), so no additional
dependencies are needed.

Typical use-case
----------------
* **Frame-to-frame ego-motion**: detect and match feature keypoints between
  consecutive frames using :mod:`sensor_transposition.feature_detection`
  (pure NumPy/SciPy, no external dependencies) or any external library
  (ORB, SIFT, etc.) and supply the matched pixel pairs to
  :func:`estimate_essential_matrix` and :func:`recover_pose_from_essential`.
* **Map-based localisation**: use :func:`solve_pnp` to estimate the pose of a
  new frame given 3-D map-point observations.

Complete end-to-end example (no external dependencies)::

    import numpy as np
    from sensor_transposition.feature_detection import (
        detect_harris_corners,
        compute_patch_descriptor,
        match_features,
    )
    from sensor_transposition.visual_odometry import (
        estimate_essential_matrix,
        recover_pose_from_essential,
        solve_pnp,
    )

    # Camera intrinsic matrix
    K = np.array([[718.856, 0, 607.193],
                  [0, 718.856, 185.216],
                  [0, 0, 1.0]])

    # 1. Detect Harris corners
    kp1 = detect_harris_corners(gray1, threshold=0.01, max_corners=500)
    kp2 = detect_harris_corners(gray2, threshold=0.01, max_corners=500)

    # 2. Compute patch descriptors and match (Lowe's ratio test)
    desc1 = compute_patch_descriptor(gray1, kp1, patch_size=11)
    desc2 = compute_patch_descriptor(gray2, kp2, patch_size=11)
    matches = match_features(desc1, desc2, ratio_threshold=0.75)

    # (row, col) → pixel (u, v) coordinate convention
    pts1 = kp1[matches[:, 0], ::-1].astype(float)
    pts2 = kp2[matches[:, 1], ::-1].astype(float)

    # 3. Estimate E from matched pixel pairs
    result = estimate_essential_matrix(pts1, pts2, K)
    E, mask = result.essential_matrix, result.inlier_mask

    # 4. Recover camera motion
    R, t = recover_pose_from_essential(E, pts1[mask], pts2[mask], K)

    # 5. Estimate pose from 3-D / 2-D correspondences (map-based localisation)
    pnp = solve_pnp(points_3d, pixels_2d, K)
    if pnp.success:
        print("Camera rotation:\\n", pnp.rotation)
        print("Camera translation:", pnp.translation)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EssentialMatrixResult:
    """Result of essential-matrix estimation.

    Attributes:
        essential_matrix: 3×3 essential matrix *E* such that
            ``x2.T @ E @ x1 ≈ 0`` for corresponding normalised image
            coordinates.
        inlier_mask: Boolean array of length *N* (one entry per input
            correspondence).  ``True`` marks inliers that are consistent
            with *E* to within *inlier_threshold* pixels.
        num_inliers: Number of inlier correspondences.
    """

    essential_matrix: np.ndarray
    inlier_mask: np.ndarray
    num_inliers: int


@dataclass
class PnPResult:
    """Result of a PnP pose estimation.

    Attributes:
        rotation: 3×3 rotation matrix *R* mapping world points to the
            camera frame: ``p_cam = R @ p_world + t``.
        translation: (3,) translation vector *t* in metres.
        inlier_mask: Boolean array of length *N*.  ``True`` marks
            correspondences used as inliers in the final solution.
        num_inliers: Number of inlier correspondences.
        success: ``True`` if a valid pose was found.
    """

    rotation: np.ndarray
    translation: np.ndarray
    inlier_mask: np.ndarray
    num_inliers: int
    success: bool


# ---------------------------------------------------------------------------
# Essential matrix
# ---------------------------------------------------------------------------


def estimate_essential_matrix(
    points1: np.ndarray,
    points2: np.ndarray,
    K: np.ndarray,
    *,
    inlier_threshold: float = 1.0,
    max_ransac_iterations: int = 1000,
    confidence: float = 0.999,
    rng: np.random.Generator | None = None,
) -> EssentialMatrixResult:
    """Estimate the essential matrix from two sets of matching pixel coordinates.

    Uses the **normalised 8-point algorithm** inside a **RANSAC** loop to
    produce a robust estimate:

    1. Randomly sample 8 correspondences.
    2. Normalise pixel coordinates by *K* to obtain normalised image
       coordinates.
    3. Build and solve the 9×9 linear system ``A e = 0`` via SVD.
    4. Enforce the essential-matrix rank-2 constraint (singular values
       ``[s, s, 0]``) via an additional SVD.
    5. Count inliers using the Sampson error as the reprojection
       approximation.
    6. Return the *E* estimated from all inliers of the best hypothesis.

    Args:
        points1: ``(N, 2)`` float array of pixel coordinates in image 1.
        points2: ``(N, 2)`` float array of pixel coordinates in image 2
            (same indexing as *points1*).
        K: 3×3 camera intrinsic matrix (same for both views).
        inlier_threshold: Maximum Sampson error (in normalised-coordinate
            units, roughly pixels / focal_length) for a correspondence to be
            counted as an inlier.  Default ``1.0``.
        max_ransac_iterations: Upper bound on RANSAC iterations.  The
            algorithm may terminate early when the adaptive stopping
            criterion is met.  Default ``1000``.
        confidence: Desired RANSAC confidence level in ``(0, 1)``.
            Controls the adaptive iteration count.  Default ``0.999``.
        rng: Optional :class:`numpy.random.Generator` for reproducibility.

    Returns:
        :class:`EssentialMatrixResult` containing the 3×3 essential matrix,
        the boolean inlier mask, and the inlier count.

    Raises:
        ValueError: If the input arrays have wrong shapes, *N* < 8, or *K*
            is not 3×3.
    """
    pts1 = np.asarray(points1, dtype=float)
    pts2 = np.asarray(points2, dtype=float)
    K = np.asarray(K, dtype=float)

    _validate_pixel_pairs(pts1, pts2, min_points=8)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3×3, got {K.shape}.")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")
    if inlier_threshold <= 0.0:
        raise ValueError(
            f"inlier_threshold must be positive, got {inlier_threshold}."
        )
    if max_ransac_iterations < 1:
        raise ValueError(
            f"max_ransac_iterations must be >= 1, got {max_ransac_iterations}."
        )

    if rng is None:
        rng = np.random.default_rng()

    # Normalise pixel coordinates to normalised image plane.
    K_inv = np.linalg.inv(K)
    n1 = _pixels_to_normalised(pts1, K_inv)  # (N, 2)
    n2 = _pixels_to_normalised(pts2, K_inv)  # (N, 2)

    N = n1.shape[0]
    threshold_sq = inlier_threshold ** 2

    best_E = np.zeros((3, 3))
    best_mask = np.zeros(N, dtype=bool)
    best_inliers = 0

    max_iter = max_ransac_iterations

    for iteration in range(max_iter):
        # Sample 8 correspondences.
        indices = rng.choice(N, size=8, replace=False)
        E_cand = _eight_point(n1[indices], n2[indices])
        if E_cand is None:
            continue

        # Count inliers via Sampson error.
        errors = _sampson_error(E_cand, n1, n2)  # (N,)
        mask = errors < threshold_sq
        n_inliers = int(mask.sum())

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_mask = mask
            best_E = E_cand

            # Adaptive RANSAC: update iteration count estimate.
            inlier_ratio = n_inliers / N
            if inlier_ratio >= 1.0:
                break
            denom = 1.0 - inlier_ratio ** 8
            if denom < 1e-12:
                break
            max_iter = min(
                max_ransac_iterations,
                int(np.ceil(np.log(1.0 - confidence) / np.log(denom))),
            )

    # Re-estimate E from all inliers of the best model.
    if best_inliers >= 8:
        E_refined = _eight_point(n1[best_mask], n2[best_mask])
        if E_refined is not None:
            best_E = E_refined
            # Recompute inlier mask with the refined E.
            errors = _sampson_error(best_E, n1, n2)
            best_mask = errors < threshold_sq
            best_inliers = int(best_mask.sum())

    return EssentialMatrixResult(
        essential_matrix=best_E,
        inlier_mask=best_mask,
        num_inliers=best_inliers,
    )


def recover_pose_from_essential(
    E: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the relative camera pose (R, t) from an essential matrix.

    Decomposes *E* via SVD to produce four (R, t) candidate solutions, then
    selects the unique physically valid solution by requiring that the
    majority of triangulated points lie in front of **both** cameras
    (positive depth test).

    Args:
        E: 3×3 essential matrix.
        points1: ``(N, 2)`` pixel coordinates in view 1 (inliers only).
        points2: ``(N, 2)`` pixel coordinates in view 2 (inliers only).
        K: 3×3 camera intrinsic matrix (same for both views).

    Returns:
        Tuple ``(R, t)`` where *R* is a 3×3 rotation matrix and *t* is a
        unit-norm (3,) translation vector.  The translation is recovered up
        to scale (baseline unknown for monocular).

    Raises:
        ValueError: If inputs have wrong shapes or fewer than 5 point pairs.
    """
    pts1 = np.asarray(points1, dtype=float)
    pts2 = np.asarray(points2, dtype=float)
    E = np.asarray(E, dtype=float)
    K = np.asarray(K, dtype=float)

    _validate_pixel_pairs(pts1, pts2, min_points=5)
    if E.shape != (3, 3):
        raise ValueError(f"E must be 3×3, got {E.shape}.")
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3×3, got {K.shape}.")

    K_inv = np.linalg.inv(K)
    n1 = _pixels_to_normalised(pts1, K_inv)
    n2 = _pixels_to_normalised(pts2, K_inv)

    R1, R2, t = _decompose_essential(E)

    # Camera 1 is at the world origin.
    P1 = np.eye(3, 4)
    best_R, best_t = R1, t
    best_count = -1

    for R_cand, t_cand in [(R1, t), (R1, -t), (R2, t), (R2, -t)]:
        P2 = np.hstack([R_cand, t_cand.reshape(3, 1)])
        pts_3d = _triangulate_points(P1, P2, n1, n2)  # (N, 3)

        # Positive depth in camera 1.
        depth1 = pts_3d[:, 2]
        # Positive depth in camera 2: z2 = (R @ pts_3d.T + t)[2]
        depth2 = (R_cand @ pts_3d.T + t_cand.reshape(3, 1))[2]

        count = int(np.sum((depth1 > 0) & (depth2 > 0)))
        if count > best_count:
            best_count = count
            best_R = R_cand
            best_t = t_cand

    return best_R, best_t


# ---------------------------------------------------------------------------
# Perspective-n-Point
# ---------------------------------------------------------------------------


def solve_pnp(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
    *,
    inlier_threshold: float = 2.0,
    max_ransac_iterations: int = 500,
    confidence: float = 0.999,
    rng: np.random.Generator | None = None,
) -> PnPResult:
    """Estimate camera pose from 3-D / 2-D point correspondences (PnP).

    Solves for the 6-DOF camera pose ``(R, t)`` given a set of world-frame
    3-D points and their corresponding pixel observations using the
    **Direct Linear Transform (DLT)** inside a RANSAC loop.

    The DLT formulation builds a ``(2N, 12)`` linear system from the
    projection equations and solves it via SVD.  With ``N >= 6`` non-coplanar
    points it produces a full perspective solution.  For robustness against
    feature-matching outliers, the DLT is embedded in RANSAC using a minimum
    sample of 6 correspondences.

    Args:
        points_3d: ``(N, 3)`` world-frame coordinates of the 3-D points.
        points_2d: ``(N, 2)`` pixel coordinates of the corresponding
            image observations.
        K: 3×3 camera intrinsic matrix.
        inlier_threshold: Maximum reprojection error in **pixels** for a
            correspondence to be considered an inlier.  Default ``2.0``.
        max_ransac_iterations: Upper bound on RANSAC iterations.  Default
            ``500``.
        confidence: Desired RANSAC confidence level in ``(0, 1)``.  Default
            ``0.999``.
        rng: Optional :class:`numpy.random.Generator` for reproducibility.

    Returns:
        :class:`PnPResult` with the rotation matrix, translation vector,
        inlier mask, inlier count, and a success flag.  If fewer than 6
        non-degenerate correspondences are available ``success`` is
        ``False``.

    Raises:
        ValueError: If input arrays have the wrong shape, *N* < 6, or *K*
            is not 3×3.
    """
    pts3 = np.asarray(points_3d, dtype=float)
    pts2 = np.asarray(points_2d, dtype=float)
    K = np.asarray(K, dtype=float)

    if pts3.ndim != 2 or pts3.shape[1] != 3:
        raise ValueError(
            f"points_3d must be (N, 3), got {pts3.shape}."
        )
    if pts2.ndim != 2 or pts2.shape[1] != 2:
        raise ValueError(
            f"points_2d must be (N, 2), got {pts2.shape}."
        )
    if pts3.shape[0] != pts2.shape[0]:
        raise ValueError(
            "points_3d and points_2d must have the same number of rows, "
            f"got {pts3.shape[0]} and {pts2.shape[0]}."
        )
    if pts3.shape[0] < 6:
        raise ValueError(
            f"At least 6 point correspondences are required, "
            f"got {pts3.shape[0]}."
        )
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3×3, got {K.shape}.")
    if inlier_threshold <= 0.0:
        raise ValueError(
            f"inlier_threshold must be positive, got {inlier_threshold}."
        )
    if max_ransac_iterations < 1:
        raise ValueError(
            f"max_ransac_iterations must be >= 1, got {max_ransac_iterations}."
        )
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")

    if rng is None:
        rng = np.random.default_rng()

    N = pts3.shape[0]
    threshold_sq = inlier_threshold ** 2

    best_R = np.eye(3)
    best_t = np.zeros(3)
    best_mask = np.zeros(N, dtype=bool)
    best_inliers = 0
    success = False

    max_iter = max_ransac_iterations

    for _ in range(max_iter):
        indices = rng.choice(N, size=6, replace=False)
        result = _dlt_pnp(pts3[indices], pts2[indices], K)
        if result is None:
            continue
        R_cand, t_cand = result

        # Count inliers by reprojection error.
        errors_sq = _reprojection_error_sq(R_cand, t_cand, K, pts3, pts2)
        mask = errors_sq < threshold_sq
        n_inliers = int(mask.sum())

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_mask = mask
            best_R = R_cand
            best_t = t_cand
            success = True

            # Adaptive stopping criterion.
            inlier_ratio = n_inliers / N
            if inlier_ratio >= 1.0:
                break
            denom = 1.0 - inlier_ratio ** 6
            if denom < 1e-12:
                break
            max_iter = min(
                max_ransac_iterations,
                int(np.ceil(np.log(1.0 - confidence) / np.log(denom))),
            )

    # Re-estimate from all inliers of the best model.
    if best_inliers >= 6:
        refined = _dlt_pnp(pts3[best_mask], pts2[best_mask], K)
        if refined is not None:
            best_R, best_t = refined
            errors_sq = _reprojection_error_sq(
                best_R, best_t, K, pts3, pts2
            )
            best_mask = errors_sq < threshold_sq
            best_inliers = int(best_mask.sum())

    return PnPResult(
        rotation=best_R,
        translation=best_t,
        inlier_mask=best_mask,
        num_inliers=best_inliers,
        success=success,
    )


# ---------------------------------------------------------------------------
# Internal helpers – essential matrix
# ---------------------------------------------------------------------------


def _pixels_to_normalised(pixels: np.ndarray, K_inv: np.ndarray) -> np.ndarray:
    """Convert (N, 2) pixel array to (N, 2) normalised image coordinates."""
    N = pixels.shape[0]
    hom = np.column_stack([pixels, np.ones(N)])  # (N, 3)
    norm = (K_inv @ hom.T).T                     # (N, 3)
    return norm[:, :2]


def _eight_point(
    n1: np.ndarray,
    n2: np.ndarray,
) -> np.ndarray | None:
    """Normalised 8-point algorithm: estimate E from ≥ 8 normalised pairs.

    Args:
        n1: ``(N, 2)`` normalised coordinates in view 1.
        n2: ``(N, 2)`` normalised coordinates in view 2.

    Returns:
        3×3 essential matrix, or ``None`` if the system is degenerate.
    """
    N = n1.shape[0]

    # Hartley normalisation: zero-mean, RMS distance √2 to origin.
    T1, n1_norm = _hartley_normalise(n1)
    T2, n2_norm = _hartley_normalise(n2)

    # Build the (N, 9) epipolar constraint matrix.
    x1, y1 = n1_norm[:, 0], n1_norm[:, 1]
    x2, y2 = n2_norm[:, 0], n2_norm[:, 1]
    A = np.column_stack([
        x2 * x1, x2 * y1, x2,
        y2 * x1, y2 * y1, y2,
        x1,      y1,      np.ones(N),
    ])  # (N, 9)

    _, _, Vt = np.linalg.svd(A)
    e = Vt[-1]           # right singular vector of smallest singular value
    E_raw = e.reshape(3, 3)

    # Enforce the essential-matrix rank-2 constraint.
    U, s, Vt2 = np.linalg.svd(E_raw)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt2) < 0:
        Vt2 = -Vt2
    s_mean = (s[0] + s[1]) / 2.0
    E_constrained = U @ np.diag([s_mean, s_mean, 0.0]) @ Vt2

    # Denormalise.
    E_final = T2.T @ E_constrained @ T1

    # Normalise to unit Frobenius norm for consistent scale.
    norm = np.linalg.norm(E_final)
    if norm < 1e-12:
        return None
    return E_final / norm


def _hartley_normalise(
    pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation for 2-D point sets.

    Translates points to zero mean and scales so the mean distance to the
    origin is √2.

    Args:
        pts: ``(N, 2)`` point array.

    Returns:
        Tuple ``(T, pts_normalised)`` where *T* is the 3×3 normalisation
        matrix and *pts_normalised* is the ``(N, 2)`` normalised array.
    """
    mean = pts.mean(axis=0)
    shifted = pts - mean
    mean_dist = np.sqrt((shifted ** 2).sum(axis=1)).mean()
    scale = np.sqrt(2.0) / (mean_dist + 1e-12)

    T = np.array([
        [scale, 0.0,   -scale * mean[0]],
        [0.0,   scale, -scale * mean[1]],
        [0.0,   0.0,    1.0],
    ])
    pts_n = shifted * scale
    return T, pts_n


def _sampson_error(
    E: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
) -> np.ndarray:
    """Compute the squared Sampson error for each correspondence.

    The Sampson error is a first-order approximation to the reprojection
    error and is cheap to compute::

        err² = (x2.T E x1)² / (|E x1|[0]² + |E x1|[1]² +
                                 |E.T x2|[0]² + |E.T x2|[1]²)

    Args:
        E: 3×3 essential matrix.
        n1: ``(N, 2)`` normalised coordinates in view 1.
        n2: ``(N, 2)`` normalised coordinates in view 2.

    Returns:
        ``(N,)`` array of squared Sampson errors.
    """
    N = n1.shape[0]
    x1 = np.column_stack([n1, np.ones(N)])  # (N, 3)
    x2 = np.column_stack([n2, np.ones(N)])  # (N, 3)

    Ex1 = (E @ x1.T).T      # (N, 3)
    Etx2 = (E.T @ x2.T).T   # (N, 3)

    numerator = np.sum(x2 * Ex1, axis=1) ** 2   # (N,)
    denom = (
        Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2
        + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2
    )
    # Avoid division by zero.
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return numerator / denom


def _decompose_essential(
    E: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose *E* into two rotation candidates and one translation direction.

    The standard SVD decomposition of the essential matrix yields::

        E = U diag(1, 1, 0) Vt
        R1 = U W  Vt
        R2 = U W.T Vt
        t  = U[:, 2]   (unit vector; sign ambiguity resolved externally)

    where ``W = [[0,-1,0],[1,0,0],[0,0,1]]``.

    Args:
        E: 3×3 essential matrix.

    Returns:
        Tuple ``(R1, R2, t)`` – two rotation matrices and the translation
        direction.
    """
    U, _, Vt = np.linalg.svd(E)

    # Ensure proper rotations.
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    W = np.array([[0.0, -1.0, 0.0],
                  [1.0,  0.0, 0.0],
                  [0.0,  0.0, 1.0]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    # Correct sign so det(R) = +1.
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    return R1, R2, t


def _triangulate_points(
    P1: np.ndarray,
    P2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    """Linear triangulation (DLT) for a batch of point pairs.

    Args:
        P1: 3×4 projection matrix for camera 1.
        P2: 3×4 projection matrix for camera 2.
        pts1: ``(N, 2)`` normalised coordinates in view 1.
        pts2: ``(N, 2)`` normalised coordinates in view 2.

    Returns:
        ``(N, 3)`` Euclidean 3-D points (in the coordinate frame of camera 1).
    """
    N = pts1.shape[0]
    pts_3d = np.zeros((N, 3))

    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A = np.array([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        if abs(X[3]) < 1e-12:
            pts_3d[i] = np.array([np.nan, np.nan, np.nan])
        else:
            pts_3d[i] = X[:3] / X[3]

    return pts_3d


# ---------------------------------------------------------------------------
# Internal helpers – PnP / DLT
# ---------------------------------------------------------------------------


def _dlt_pnp(
    pts3: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Direct Linear Transform for the Perspective-n-Point problem.

    Solves the full-perspective PnP by building a ``(2N, 12)`` linear
    system and extracting the 3×4 projection matrix *P* via SVD.

    The camera projection model is::

        λ [u, v, 1]^T = K [R | t] [X, Y, Z, 1]^T

    which is re-written as a homogeneous linear system in the 12 entries of
    the 3×4 matrix ``M = K [R | t]``.

    Args:
        pts3: ``(N, 3)`` world-frame 3-D points.
        pts2: ``(N, 2)`` observed pixel coordinates.
        K: 3×3 camera intrinsic matrix.

    Returns:
        Tuple ``(R, t)`` on success, or ``None`` if the system is degenerate
        or the extracted matrix is not a valid camera pose.
    """
    N = pts3.shape[0]

    # Normalise 3-D points for numerical stability.
    mean3 = pts3.mean(axis=0)
    scale3 = np.sqrt(((pts3 - mean3) ** 2).sum(axis=1)).mean() + 1e-12
    pts3_n = (pts3 - mean3) / scale3

    # Normalise 2-D points (Hartley).
    T2d, pts2_n = _hartley_normalise(pts2)

    # Build the (2N, 12) DLT matrix in normalised coordinates.
    rows = []
    for i in range(N):
        X, Y, Z = pts3_n[i]
        u, v = pts2_n[i]
        rows.append([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u])
        rows.append([0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v])

    A = np.array(rows, dtype=float)
    _, _, Vt = np.linalg.svd(A)
    p = Vt[-1]
    P_n = p.reshape(3, 4)

    # Denormalise: P_true = T2d_inv @ P_n @ T3d  (in normalised-K space)
    T3d = np.eye(4)
    T3d[:3, :3] *= scale3
    T3d[:3, 3] = mean3
    T2d_inv = np.linalg.inv(T2d)

    # Map back to pixel space then to normalised image plane.
    K_inv = np.linalg.inv(K)
    P_pix = T2d_inv @ P_n @ np.linalg.inv(T3d)
    P_cam = K_inv @ P_pix

    # Decompose P_cam = [R | t] via RQ decomposition (using QR on reversed P).
    M = P_cam[:3, :3]
    t_raw = P_cam[:3, 3]

    # Enforce positive depth: if det(M) < 0 flip sign.
    if np.linalg.det(M) < 0:
        M = -M
        t_raw = -t_raw

    # Polar decomposition: M = R * sqrt(M.T M) → extract R.
    U, _, Vt2 = np.linalg.svd(M)
    R = U @ Vt2

    if np.linalg.det(R) < 0:
        R = -R
        t_raw = -t_raw

    # Scale t by the inverse of the scale factor embedded in M.
    # Estimate scale as the mean singular value.
    _, s, _ = np.linalg.svd(M)
    scale = s.mean()
    if scale < 1e-12:
        return None

    t = t_raw / scale

    return R, t


def _reprojection_error_sq(
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    pts3: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    """Compute squared reprojection error for each 3-D / 2-D correspondence.

    Args:
        R: 3×3 rotation matrix.
        t: (3,) translation vector.
        K: 3×3 camera intrinsic matrix.
        pts3: ``(N, 3)`` world-frame 3-D points.
        pts2: ``(N, 2)`` observed pixel coordinates.

    Returns:
        ``(N,)`` array of squared reprojection errors in pixels².
    """
    # Project: p_cam = R @ X + t
    p_cam = (R @ pts3.T).T + t  # (N, 3)

    # Avoid division by near-zero depth.
    z = p_cam[:, 2]
    valid = np.abs(z) > 1e-8
    z_safe = np.where(valid, z, 1.0)

    x_n = p_cam[:, 0] / z_safe
    y_n = p_cam[:, 1] / z_safe

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u_proj = fx * x_n + cx
    v_proj = fy * y_n + cy

    du = u_proj - pts2[:, 0]
    dv = v_proj - pts2[:, 1]

    errors_sq = du ** 2 + dv ** 2
    # Mark points behind the camera with infinite error.
    errors_sq = np.where(valid, errors_sq, np.inf)
    return errors_sq


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------


def _validate_pixel_pairs(
    pts1: np.ndarray,
    pts2: np.ndarray,
    min_points: int,
) -> None:
    """Raise ``ValueError`` if *pts1* / *pts2* are not matching (N, 2) arrays."""
    if pts1.ndim != 2 or pts1.shape[1] != 2:
        raise ValueError(f"points1 must be (N, 2), got {pts1.shape}.")
    if pts2.ndim != 2 or pts2.shape[1] != 2:
        raise ValueError(f"points2 must be (N, 2), got {pts2.shape}.")
    if pts1.shape[0] != pts2.shape[0]:
        raise ValueError(
            "points1 and points2 must have the same number of rows, "
            f"got {pts1.shape[0]} and {pts2.shape[0]}."
        )
    if pts1.shape[0] < min_points:
        raise ValueError(
            f"At least {min_points} point pairs are required, "
            f"got {pts1.shape[0]}."
        )
