"""
feature_detection.py

Minimal pure-NumPy/SciPy image feature detection and descriptor matching for
visual odometry pipelines that need to operate without OpenCV.

This module provides a complete pipeline from raw grayscale images to matched
point pairs that can be fed directly into
:func:`sensor_transposition.visual_odometry.estimate_essential_matrix`:

1. **Harris corner detection** – :func:`detect_harris_corners` computes the
   Harris corner response at every pixel using a Sobel approximation of image
   gradients, then extracts local maxima above a configurable threshold with
   non-maximum suppression.

2. **Patch descriptor** – :func:`compute_patch_descriptor` extracts a
   normalised flat intensity patch around each keypoint as a compact
   appearance descriptor.  The patch is mean-centred and unit-normalised so
   that L2 distance corresponds to a correlation-like similarity.

3. **Feature matching** – :func:`match_features` performs brute-force L2
   nearest-neighbour matching between two descriptor sets and applies Lowe's
   ratio test to filter ambiguous (and therefore likely false) matches.

All functions operate on plain NumPy arrays and depend only on ``numpy`` and
``scipy`` (already required by ``sensor_transposition``), so no additional
dependencies are needed.

Typical use-case
----------------
Complete pipeline from two grayscale images to a relative camera pose::

    import numpy as np
    from sensor_transposition.feature_detection import (
        detect_harris_corners,
        compute_patch_descriptor,
        match_features,
    )
    from sensor_transposition.visual_odometry import (
        estimate_essential_matrix,
        recover_pose_from_essential,
    )

    # 1. Detect corners in both frames
    kp1 = detect_harris_corners(gray1)
    kp2 = detect_harris_corners(gray2)

    # 2. Compute descriptors
    desc1 = compute_patch_descriptor(gray1, kp1)
    desc2 = compute_patch_descriptor(gray2, kp2)

    # 3. Match features with Lowe's ratio test
    matches = match_features(desc1, desc2, ratio_threshold=0.75)
    pts1 = kp1[matches[:, 0]]
    pts2 = kp2[matches[:, 1]]

    # 4. Estimate essential matrix and recover pose
    result = estimate_essential_matrix(pts1, pts2, K)
    E, mask = result.essential_matrix, result.inlier_mask
    R, t = recover_pose_from_essential(E, pts1[mask], pts2[mask], K)

"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import maximum_filter

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Sobel kernels for gradient approximation.
_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D convolution via NumPy stride tricks (no SciPy signal dependency).

    Performs 'same'-mode convolution: the output has the same shape as
    *image*.  Border pixels that would require out-of-bounds reads are set to
    zero.

    Args:
        image: 2-D float array of shape ``(H, W)``.
        kernel: 2-D float array of shape ``(kH, kW)``.  Both dimensions must
            be odd.

    Returns:
        2-D float array of shape ``(H, W)`` — the convolution result.
    """
    H, W = image.shape
    kH, kW = kernel.shape
    pH, pW = kH // 2, kW // 2

    # Pad image with zeros.
    padded = np.pad(image, ((pH, pH), (pW, pW)), mode="constant")

    # Build output using vectorised sliding-window multiplication.
    out = np.zeros((H, W), dtype=float)
    for di in range(kH):
        for dj in range(kW):
            out += padded[di: di + H, dj: dj + W] * kernel[di, dj]
    return out


def _gaussian_kernel(sigma: float, radius: int | None = None) -> np.ndarray:
    """Construct a 2-D isotropic Gaussian kernel.

    Args:
        sigma: Standard deviation in pixels.  Must be > 0.
        radius: Half-width of the kernel in pixels.  If ``None`` the radius
            is set to ``ceil(3 * sigma)``.

    Returns:
        2-D float array of shape ``(2*radius+1, 2*radius+1)`` that sums to 1.
    """
    import math

    if radius is None:
        radius = int(math.ceil(3.0 * sigma))
    size = 2 * radius + 1
    ax = np.arange(-radius, radius + 1, dtype=float)
    g1d = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel = np.outer(g1d, g1d)
    return kernel / kernel.sum()


# ---------------------------------------------------------------------------
# Harris corner detection
# ---------------------------------------------------------------------------


def detect_harris_corners(
    image: np.ndarray,
    *,
    k: float = 0.05,
    sigma: float = 1.0,
    threshold: float = 0.01,
    nms_radius: int = 5,
    max_corners: int | None = None,
) -> np.ndarray:
    """Detect Harris corners in a grayscale image.

    **Algorithm:**

    1. Compute image gradients ``Ix`` and ``Iy`` using 3×3 Sobel kernels.
    2. Compute the second-moment matrix elements
       ``Ixx = Ix²``, ``Ixy = Ix·Iy``, ``Iyy = Iy²``.
    3. Smooth each element with a Gaussian of standard deviation *sigma*.
    4. Compute the corner response
       ``R = det(M) − k · trace(M)²``
       where ``M = [[Ixx, Ixy], [Ixy, Iyy]]``.
    5. Threshold *R* at *threshold* × max(R) and suppress non-maxima within
       a square neighbourhood of radius *nms_radius*.

    Args:
        image: Grayscale image as a 2-D NumPy array.  Values need not be
            normalised; float or uint8 are both accepted.
        k: Harris sensitivity parameter (``0.04``–``0.06`` is typical).
            Default ``0.05``.
        sigma: Standard deviation (pixels) of the Gaussian window used to
            smooth the second-moment matrix elements.  Default ``1.0``.
        threshold: Relative threshold in ``(0, 1]`` applied to the maximum
            response value.  Only pixels with
            ``R ≥ threshold × max(R)`` are kept.  Default ``0.01``.
        nms_radius: Half-width of the non-maximum suppression window in
            pixels.  Keypoints whose response is not a strict local maximum
            within this neighbourhood are discarded.  Default ``5``.
        max_corners: If given, return only the top *max_corners* keypoints
            ranked by response value.

    Returns:
        ``(M, 2)`` integer array of ``(row, col)`` corner coordinates, sorted
        by descending response.  Returns an empty ``(0, 2)`` array when no
        corners are found.

    Raises:
        ValueError: If *image* is not 2-D, *k* or *threshold* are not in
            ``(0, 1]``, or *nms_radius* < 1.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError(
            f"image must be a 2-D (grayscale) array, got shape {img.shape}."
        )
    if not (0.0 < k <= 1.0):
        raise ValueError(f"k must be in (0, 1], got {k}.")
    if not (0.0 < threshold <= 1.0):
        raise ValueError(f"threshold must be in (0, 1], got {threshold}.")
    if nms_radius < 1:
        raise ValueError(f"nms_radius must be >= 1, got {nms_radius}.")

    # 1. Sobel gradients.
    Ix = _convolve2d(img, _SOBEL_X)
    Iy = _convolve2d(img, _SOBEL_Y)

    # 2. Second-moment matrix elements.
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy

    # 3. Gaussian smoothing.
    g = _gaussian_kernel(sigma)
    Sxx = _convolve2d(Ixx, g)
    Sxy = _convolve2d(Ixy, g)
    Syy = _convolve2d(Iyy, g)

    # 4. Harris response.
    det_M = Sxx * Syy - Sxy * Sxy
    trace_M = Sxx + Syy
    R = det_M - k * trace_M * trace_M

    # 5. Threshold.
    r_max = R.max()
    if r_max <= 0.0:
        return np.empty((0, 2), dtype=int)

    # Non-maximum suppression: keep only pixels that are local maxima.
    nms_size = 2 * nms_radius + 1
    R_max_filtered = maximum_filter(R, size=nms_size)
    is_local_max = (R == R_max_filtered) & (R >= threshold * r_max)

    rows, cols = np.nonzero(is_local_max)
    if rows.size == 0:
        return np.empty((0, 2), dtype=int)

    # Sort by descending response.
    responses = R[rows, cols]
    order = np.argsort(-responses)
    rows, cols = rows[order], cols[order]

    if max_corners is not None:
        rows = rows[:max_corners]
        cols = cols[:max_corners]

    return np.stack([rows, cols], axis=1)


# ---------------------------------------------------------------------------
# Patch descriptor
# ---------------------------------------------------------------------------


def compute_patch_descriptor(
    image: np.ndarray,
    keypoints: np.ndarray,
    *,
    patch_size: int = 11,
) -> np.ndarray:
    """Extract normalised intensity-patch descriptors at keypoint locations.

    For each keypoint a square patch of size *patch_size × patch_size* is
    extracted from *image*, mean-centred, and L2-normalised.  The resulting
    flat vector is used as a simple but effective appearance descriptor for
    brute-force matching.

    Keypoints too close to the image border (within *patch_size // 2* pixels)
    are represented by zero-vectors and will be filtered out by
    :func:`match_features`.

    Args:
        image: Grayscale 2-D float or uint8 array.
        keypoints: ``(N, 2)`` integer array of ``(row, col)`` coordinates as
            returned by :func:`detect_harris_corners`.
        patch_size: Side length of the square patch in pixels.  Must be a
            positive odd integer.  Default ``11``.

    Returns:
        ``(N, patch_size²)`` float array of L2-normalised descriptors.  Rows
        corresponding to border keypoints are all-zero.

    Raises:
        ValueError: If *image* is not 2-D, *keypoints* is not ``(N, 2)``, or
            *patch_size* is not a positive odd integer.
    """
    img = np.asarray(image, dtype=float)
    kps = np.asarray(keypoints, dtype=int)

    if img.ndim != 2:
        raise ValueError(
            f"image must be a 2-D (grayscale) array, got shape {img.shape}."
        )
    if kps.ndim != 2 or kps.shape[1] != 2:
        raise ValueError(
            f"keypoints must be an (N, 2) array, got shape {kps.shape}."
        )
    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError(
            f"patch_size must be a positive odd integer, got {patch_size}."
        )

    H, W = img.shape
    half = patch_size // 2
    N = len(kps)
    descriptors = np.zeros((N, patch_size * patch_size), dtype=float)

    for i, (r, c) in enumerate(kps):
        r0, r1 = r - half, r + half + 1
        c0, c1 = c - half, c + half + 1
        if r0 < 0 or r1 > H or c0 < 0 or c1 > W:
            # Border keypoint – leave as zero vector.
            continue
        patch = img[r0:r1, c0:c1].ravel().astype(float)
        patch -= patch.mean()
        norm = np.linalg.norm(patch)
        if norm > 1e-12:
            patch /= norm
        descriptors[i] = patch

    return descriptors


# ---------------------------------------------------------------------------
# Feature matching
# ---------------------------------------------------------------------------


def match_features(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    *,
    ratio_threshold: float = 0.75,
) -> np.ndarray:
    """Match two sets of descriptors using brute-force L2 + Lowe's ratio test.

    For each descriptor in *descriptors1* the two nearest neighbours in
    *descriptors2* are found by brute-force L2 distance.  A match is accepted
    only when the ratio of the best to the second-best distance is below
    *ratio_threshold* (Lowe's ratio test [Lowe 2004]).  This rejects
    ambiguous matches where two potential neighbours are similarly close,
    which are likely to be false positives.

    Zero-vector descriptors (e.g. border keypoints from
    :func:`compute_patch_descriptor`) are excluded from both the query and
    reference sets.

    Args:
        descriptors1: ``(N, D)`` float array of query descriptors.
        descriptors2: ``(M, D)`` float array of reference descriptors.
        ratio_threshold: Maximum allowed ratio ``d_best / d_second`` for a
            match to be accepted.  Must be in ``(0, 1]``.  Default ``0.75``.

    Returns:
        ``(K, 2)`` integer array of accepted matches.  Each row
        ``[i, j]`` means descriptor *i* from *descriptors1* matches
        descriptor *j* from *descriptors2*.  Returns an empty ``(0, 2)``
        array when no matches survive the ratio test.

    Raises:
        ValueError: If arrays have incompatible descriptor dimensions or
            *ratio_threshold* is not in ``(0, 1]``.
    """
    d1 = np.asarray(descriptors1, dtype=float)
    d2 = np.asarray(descriptors2, dtype=float)

    if d1.ndim != 2 or d2.ndim != 2:
        raise ValueError("descriptors1 and descriptors2 must be 2-D arrays.")
    if d1.shape[1] != d2.shape[1]:
        raise ValueError(
            f"Descriptor dimension mismatch: {d1.shape[1]} vs {d2.shape[1]}."
        )
    if not (0.0 < ratio_threshold <= 1.0):
        raise ValueError(
            f"ratio_threshold must be in (0, 1], got {ratio_threshold}."
        )

    if d1.shape[0] == 0 or d2.shape[0] == 0:
        return np.empty((0, 2), dtype=int)

    # Identify non-zero query and reference descriptors.
    valid1 = np.linalg.norm(d1, axis=1) > 1e-12
    valid2 = np.linalg.norm(d2, axis=1) > 1e-12
    idx1 = np.where(valid1)[0]
    idx2 = np.where(valid2)[0]

    if idx1.size == 0 or idx2.size < 2:
        return np.empty((0, 2), dtype=int)

    d1_valid = d1[idx1]  # (N', D)
    d2_valid = d2[idx2]  # (M', D)

    # Compute pairwise squared L2 distance matrix via expanded dot product.
    # ||a - b||² = ||a||² + ||b||² - 2 a·b
    d1_sq = (d1_valid ** 2).sum(axis=1, keepdims=True)   # (N', 1)
    d2_sq = (d2_valid ** 2).sum(axis=1, keepdims=True)   # (M', 1)
    dist_sq = d1_sq + d2_sq.T - 2.0 * (d1_valid @ d2_valid.T)  # (N', M')
    dist_sq = np.clip(dist_sq, 0.0, None)  # numerical noise guard

    # For each query, find the two nearest neighbours.
    if d2_valid.shape[0] < 2:
        return np.empty((0, 2), dtype=int)

    # Partial sort: get indices of the two smallest distances per row.
    # Use argpartition for efficiency.
    part = np.argpartition(dist_sq, kth=min(1, dist_sq.shape[1] - 1), axis=1)
    nn1_local = part[:, 0]
    nn2_local = part[:, 1]

    d_best = np.sqrt(dist_sq[np.arange(len(idx1)), nn1_local])
    d_second = np.sqrt(dist_sq[np.arange(len(idx1)), nn2_local])

    # Ratio test.
    # Guard against zero second-best distances.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(d_second > 1e-12, d_best / d_second, 1.0)

    keep = ratio < ratio_threshold

    matched_i = idx1[keep]
    matched_j = idx2[nn1_local[keep]]

    if matched_i.size == 0:
        return np.empty((0, 2), dtype=int)

    return np.stack([matched_i, matched_j], axis=1)
