"""
stereo.py

Stereo camera utilities: rectification, disparity computation, and metric
depth recovery via stereo triangulation.

A stereo camera rig consists of two cameras (left and right) with known
intrinsics and a calibrated relative pose (rotation *R* and translation *t*,
which is typically just the baseline vector along the x-axis).  This module
provides:

1. **Stereo rectification** – :func:`stereo_rectify` computes the rotation
   matrices that warp both image planes to be coplanar and epipolar lines
   to be horizontal, using the standard Bouguet/OpenCV algorithm.  The
   output can be used to build pixel-coordinate look-up tables for image
   remapping.

2. **Block-matching disparity** – :func:`compute_disparity_sgbm` estimates a
   dense disparity map by sliding a square matching window (Sum of Absolute
   Differences) over the rectified left image and searching horizontally
   in the right image within the specified disparity range.  This implements
   a simple but effective block-matching approach in pure NumPy.

3. **Stereo triangulation** – :func:`triangulate_stereo` converts matched
   pixel pairs from the left and right images (or a dense disparity map)
   into metric 3-D points using the stereo camera geometry.

All functions operate on plain NumPy arrays and depend only on ``numpy`` and
``scipy`` (already required by ``sensor_transposition``), so no additional
dependencies are needed.

Typical use-case
----------------
::

    from sensor_transposition.stereo import (
        stereo_rectify,
        compute_disparity_sgbm,
        triangulate_stereo,
    )
    import numpy as np

    # Camera intrinsics and stereo geometry (from calibration)
    K = np.array([[718.856, 0, 607.193],
                  [0, 718.856, 185.216],
                  [0, 0, 1.0]])
    baseline = 0.537  # metres
    R_stereo = np.eye(3)                  # cameras are parallel
    t_stereo = np.array([-baseline, 0.0, 0.0])

    # Rectification
    R1, R2, P1, P2 = stereo_rectify(K, (), K, (), R_stereo, t_stereo,
                                     image_size=(375, 1242))

    # Compute disparity from a rectified stereo pair
    disp = compute_disparity_sgbm(img_left_rect, img_right_rect,
                                   block_size=11, num_disparities=64)

    # Convert disparity to 3-D points
    pts3d = triangulate_stereo(disp=disp, K=K, baseline=baseline)

"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Stereo rectification
# ---------------------------------------------------------------------------


def stereo_rectify(
    K1: np.ndarray,
    D1: Tuple[float, ...],
    K2: np.ndarray,
    D2: Tuple[float, ...],
    R: np.ndarray,
    t: np.ndarray,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute stereo rectification transforms for a calibrated camera pair.

    This implements the **Bouguet algorithm** (used internally by OpenCV's
    ``cv2.stereoRectify``):

    1. Compute the rotation *R_rect* that bisects *R* (the rotation from
       camera 1 to camera 2), giving each camera an equal half-rotation.
    2. Set the new x-axis of both rectified frames to be parallel to the
       stereo baseline vector *t*.
    3. Compute the resulting 3×3 rotation matrices *R1*, *R2* (one per
       camera) and ideal pinhole projection matrices *P1*, *P2*.

    For a typical **horizontal stereo rig** (cameras side by side, *t* ≈
    ``[−baseline, 0, 0]``) the result is that:

    * Epipolar lines are exactly horizontal in both rectified images.
    * Corresponding points differ only in their x-coordinate (disparity).
    * *P1* and *P2* share the same focal length and vertical principal point.

    Distortion coefficients *D1* / *D2* are accepted but currently not used in
    the rectification rotation computation (the rotation is derived from the
    calibrated extrinsic geometry only).  To build a full pixel-remapping
    look-up table, apply the returned *R1*/*R2* and *P1*/*P2* together with
    the distortion coefficients in a separate undistort-and-remap step.

    Args:
        K1: 3×3 intrinsic matrix of the left camera.
        D1: Brown–Conrady distortion coefficients for the left camera
            ``(k1, k2, p1, p2, k3)``.  May be empty.
        K2: 3×3 intrinsic matrix of the right camera.
        D2: Brown–Conrady distortion coefficients for the right camera.
            May be empty.
        R: 3×3 rotation matrix from camera 1 to camera 2.
        t: (3,) translation vector from camera 1 to camera 2 (in metres).
        image_size: ``(height, width)`` of the images in pixels.

    Returns:
        Tuple ``(R1, R2, P1, P2)`` where:

        * *R1*: 3×3 rectification rotation for the left camera.  Rotates
          camera 1's frame into the common rectified frame.
        * *R2*: 3×3 rectification rotation for the right camera.
        * *P1*: 3×4 projection matrix for the left rectified camera
          ``[f, 0, cx, 0; 0, f, cy, 0; 0, 0, 1, 0]``.
        * *P2*: 3×4 projection matrix for the right rectified camera
          ``[f, 0, cx, −f·baseline; 0, f, cy, 0; 0, 0, 1, 0]``.

    Raises:
        ValueError: If *K1*, *K2* are not 3×3; *R* is not 3×3; *t* is not
            shape (3,); or *image_size* has non-positive values.
    """
    K1 = np.asarray(K1, dtype=float)
    K2 = np.asarray(K2, dtype=float)
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).ravel()

    if K1.shape != (3, 3):
        raise ValueError(f"K1 must be 3×3, got {K1.shape}.")
    if K2.shape != (3, 3):
        raise ValueError(f"K2 must be 3×3, got {K2.shape}.")
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3×3, got {R.shape}.")
    if t.shape != (3,):
        raise ValueError(f"t must be shape (3,), got {t.shape}.")
    if len(image_size) != 2 or image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError(
            f"image_size must be (height, width) with positive values, "
            f"got {image_size}."
        )

    # Step 1: Bouguet bisection rotation.
    # R_rect rotates camera 1's frame so that the rotation to camera 2 is
    # symmetric; each camera gets half the rotation.
    # Compute the axis-angle of R, halve the angle.
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if abs(angle) < 1e-10:
        # Cameras are already co-planar.
        R_rect = np.eye(3)
    else:
        # Rodrigues axis (skew-symmetric part of R).
        rx = (R[2, 1] - R[1, 2]) / (2.0 * np.sin(angle))
        ry = (R[0, 2] - R[2, 0]) / (2.0 * np.sin(angle))
        rz = (R[1, 0] - R[0, 1]) / (2.0 * np.sin(angle))
        axis = np.array([rx, ry, rz])
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-12:
            axis /= axis_norm
        half_angle = angle / 2.0
        # Rodrigues formula for the half-rotation.
        K_cross = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
        R_rect = (
            np.eye(3)
            + np.sin(half_angle) * K_cross
            + (1.0 - np.cos(half_angle)) * (K_cross @ K_cross)
        )

    # Step 2: New x-axis is parallel to the baseline vector.
    e1 = t.copy()
    e1_norm = np.linalg.norm(e1)
    if e1_norm < 1e-12:
        raise ValueError(
            "Baseline vector t has near-zero length; cannot compute "
            "epipolar geometry."
        )
    e1 /= e1_norm  # unit baseline direction

    # New y-axis: perpendicular to e1, in the plane spanned by e1 and
    # camera 1's z-axis (optical axis [0, 0, 1] rotated by R_rect).
    z_cam = R_rect @ np.array([0.0, 0.0, 1.0])
    e2 = np.cross(z_cam, e1)
    e2_norm = np.linalg.norm(e2)
    if e2_norm < 1e-12:
        # Degenerate: baseline is parallel to optical axis; use y-axis.
        e2 = np.cross(np.array([0.0, 1.0, 0.0]), e1)
        e2_norm = np.linalg.norm(e2)
    e2 /= max(e2_norm, 1e-12)

    # New z-axis: e3 = e1 × e2.
    e3 = np.cross(e1, e2)

    # Rectification rotation (rows are the new axes).
    R_new = np.stack([e1, e2, e3], axis=0)  # (3, 3)

    # Per-camera rectification matrices.
    R1 = R_new @ R_rect             # left camera: undo R_rect, apply R_new
    R2 = R_new @ (R_rect @ R.T)    # right camera: undo R_rect, apply R_new

    # Step 3: Build projection matrices.
    # Use the average focal length and the mean principal point.
    f = float((K1[0, 0] + K1[1, 1] + K2[0, 0] + K2[1, 1]) / 4.0)
    cx = float((K1[0, 2] + K2[0, 2]) / 2.0)
    cy = float((K1[1, 2] + K2[1, 2]) / 2.0)

    # Baseline is the component of t along e1.
    baseline = float(t @ e1)  # signed; usually negative (right camera is at -Tx)

    P1 = np.array([
        [f, 0.0, cx, 0.0],
        [0.0, f, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    P2 = np.array([
        [f, 0.0, cx, -f * baseline],
        [0.0, f, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])

    return R1, R2, P1, P2


# ---------------------------------------------------------------------------
# Block-matching disparity
# ---------------------------------------------------------------------------


def compute_disparity_sgbm(
    img_left: np.ndarray,
    img_right: np.ndarray,
    *,
    block_size: int = 11,
    num_disparities: int = 64,
    min_disparity: int = 0,
) -> np.ndarray:
    """Compute a dense disparity map from a rectified stereo pair.

    Implements **block matching** (Sum of Absolute Differences) in pure
    NumPy.  For each pixel ``(r, c)`` in the left image the best matching
    position in the right image is found by searching horizontally in the
    range ``[c − min_disparity − num_disparities + 1, c − min_disparity]``,
    which matches the OpenCV convention where disparity ``d`` means the left
    pixel at column ``c`` corresponds to the right pixel at column ``c − d``.

    This is a simple but correct block-matching implementation.  For
    production use with OpenCV available, use
    ``cv2.StereoSGBM_create(...).compute(...)`` instead.

    Args:
        img_left: Rectified left grayscale image as a 2-D array (H × W).
            Float or uint8 values accepted.
        img_right: Rectified right grayscale image, same shape as *img_left*.
        block_size: Side length of the square matching window in pixels.
            Must be a positive odd integer.  Default ``11``.
        num_disparities: Number of disparity values to search.  The search
            range is ``[min_disparity, min_disparity + num_disparities − 1]``.
            Must be a positive multiple of 16 (following OpenCV convention).
            Default ``64``.
        min_disparity: Minimum disparity value to search (default ``0``).

    Returns:
        ``(H, W)`` float array of disparity values.  Pixels where no valid
        match was found (near the image border or within the disparity
        margin) are set to ``0``.

    Raises:
        ValueError: If the images are not 2-D, have different shapes, or
            *block_size* is not a positive odd integer, or
            *num_disparities* ≤ 0.
    """
    left = np.asarray(img_left, dtype=float)
    right = np.asarray(img_right, dtype=float)

    if left.ndim != 2:
        raise ValueError(
            f"img_left must be a 2-D (grayscale) array, got shape {left.shape}."
        )
    if right.ndim != 2:
        raise ValueError(
            f"img_right must be a 2-D (grayscale) array, got shape {right.shape}."
        )
    if left.shape != right.shape:
        raise ValueError(
            f"img_left and img_right must have the same shape, "
            f"got {left.shape} and {right.shape}."
        )
    if block_size < 1 or block_size % 2 == 0:
        raise ValueError(
            f"block_size must be a positive odd integer, got {block_size}."
        )
    if num_disparities <= 0:
        raise ValueError(
            f"num_disparities must be > 0, got {num_disparities}."
        )

    H, W = left.shape
    half = block_size // 2
    disparity = np.zeros((H, W), dtype=float)

    # Pre-compute integral images for fast box-sum computation.
    # integral[r, c] = sum of left[0:r, 0:c]
    # Not needed for SAD: we use direct summation over blocks.
    # For efficiency we iterate over disparity values and compute SAD maps.

    best_cost = np.full((H, W), np.inf)

    for d in range(min_disparity, min_disparity + num_disparities):
        if d < 0:
            continue
        # Shift the right image to the right by d pixels.
        right_shifted = np.zeros_like(right)
        if d < W:
            right_shifted[:, d:] = right[:, :W - d]

        diff = np.abs(left - right_shifted)

        # Compute box-sum of diff over block_size × block_size windows using
        # cumulative sums.
        cum = np.cumsum(diff, axis=0)
        cum = np.pad(cum, ((1, 0), (0, 0)), mode="constant")
        row_sum = cum[block_size:, :] - cum[:-block_size, :]  # sum over rows

        cum2 = np.cumsum(row_sum, axis=1)
        cum2 = np.pad(cum2, ((0, 0), (1, 0)), mode="constant")
        sad = cum2[:, block_size:] - cum2[:, :-block_size]   # sum over cols

        # sad has shape (H - block_size + 1, W - block_size + 1).
        # Map back to full image coordinates (top-left of valid region).
        r_start = half
        r_end = H - half
        c_start = half
        c_end = W - half

        rows_out = r_end - r_start
        cols_out = c_end - c_start

        if rows_out <= 0 or cols_out <= 0:
            continue

        sad_crop = sad[:rows_out, :cols_out]
        cost_crop = best_cost[r_start:r_end, c_start:c_end]

        improved = sad_crop < cost_crop
        cost_crop[improved] = sad_crop[improved]
        disparity[r_start:r_end, c_start:c_end][improved] = float(d)
        best_cost[r_start:r_end, c_start:c_end] = cost_crop

    return disparity


# ---------------------------------------------------------------------------
# Stereo triangulation
# ---------------------------------------------------------------------------


def triangulate_stereo(
    pts_left: np.ndarray | None = None,
    pts_right: np.ndarray | None = None,
    *,
    K: np.ndarray,
    baseline: float,
    disp: np.ndarray | None = None,
) -> np.ndarray:
    """Convert stereo pixel matches (or a disparity map) to metric 3-D points.

    Two calling modes are supported:

    **Point-pair mode** (supply *pts_left* and *pts_right*):
        Given matched pixel coordinates ``(ul, vl)`` in the left image and
        ``(ur, vr)`` in the right image, compute depth as::

            Z = f * baseline / (ul - ur)

        and unproject to 3-D::

            X = (ul - cx) * Z / f
            Y = (vl - cy) * Z / f

    **Disparity-map mode** (supply *disp*):
        For every pixel ``(r, c)`` with non-zero disparity, compute::

            Z = f * baseline / disp[r, c]
            X = (c - cx) * Z / f
            Y = (r - cy) * Z / f

        and return all valid (positive Z) 3-D points.

    Exactly one of (*pts_left* + *pts_right*) or *disp* must be provided.

    Args:
        pts_left: ``(N, 2)`` float array of ``(u, v)`` pixel coordinates in
            the *left* rectified image.  Required in point-pair mode.
        pts_right: ``(N, 2)`` float array of corresponding ``(u, v)`` pixel
            coordinates in the *right* rectified image.  Required in
            point-pair mode.
        K: 3×3 camera intrinsic matrix (shared by left and right after
            rectification).
        baseline: Stereo baseline in metres (positive scalar).  This is the
            physical distance between the two camera optical centres.
        disp: ``(H, W)`` float disparity map as returned by
            :func:`compute_disparity_sgbm`.  Required in disparity-map mode.

    Returns:
        **Point-pair mode**: ``(N, 3)`` float array of metric XYZ points in
        the left camera frame.  Points with zero or negative disparity (i.e.
        ``ul ≤ ur``) are set to ``[NaN, NaN, NaN]``.

        **Disparity-map mode**: ``(M, 3)`` float array of metric XYZ points
        for all pixels with positive disparity.

    Raises:
        ValueError: If neither or both modes are specified; if *K* is not
            3×3; if *baseline* ≤ 0; or if input arrays have wrong shapes.
    """
    K_arr = np.asarray(K, dtype=float)
    if K_arr.shape != (3, 3):
        raise ValueError(f"K must be 3×3, got {K_arr.shape}.")
    if baseline <= 0.0:
        raise ValueError(f"baseline must be positive, got {baseline}.")

    have_pts = pts_left is not None and pts_right is not None
    have_disp = disp is not None

    if have_pts and have_disp:
        raise ValueError(
            "Provide either (pts_left, pts_right) OR disp, not both."
        )
    if not have_pts and not have_disp:
        raise ValueError(
            "Provide either (pts_left, pts_right) or disp."
        )

    f = float(K_arr[0, 0])  # assume fx ≈ fy after rectification
    cx = float(K_arr[0, 2])
    cy = float(K_arr[1, 2])

    if have_pts:
        pl = np.asarray(pts_left, dtype=float)
        pr = np.asarray(pts_right, dtype=float)
        if pl.ndim != 2 or pl.shape[1] != 2:
            raise ValueError(
                f"pts_left must be (N, 2), got {pl.shape}."
            )
        if pr.ndim != 2 or pr.shape[1] != 2:
            raise ValueError(
                f"pts_right must be (N, 2), got {pr.shape}."
            )
        if pl.shape[0] != pr.shape[0]:
            raise ValueError(
                f"pts_left and pts_right must have the same length, "
                f"got {pl.shape[0]} and {pr.shape[0]}."
            )

        disp_vals = pl[:, 0] - pr[:, 0]  # ul - ur
        pts3d = np.full((pl.shape[0], 3), np.nan, dtype=float)

        valid = disp_vals > 1e-12
        safe_disp = np.where(valid, disp_vals, 1.0)  # avoid divide-by-zero
        Z = np.where(valid, f * baseline / safe_disp, np.nan)
        pts3d[:, 0] = np.where(valid, (pl[:, 0] - cx) * Z / f, np.nan)
        pts3d[:, 1] = np.where(valid, (pl[:, 1] - cy) * Z / f, np.nan)
        pts3d[:, 2] = Z
        return pts3d

    else:  # have_disp
        disp_arr = np.asarray(disp, dtype=float)
        if disp_arr.ndim != 2:
            raise ValueError(
                f"disp must be a 2-D array, got shape {disp_arr.shape}."
            )
        H, W = disp_arr.shape
        rows, cols = np.indices((H, W))

        valid = disp_arr > 1e-12
        rows_v = rows[valid].astype(float)
        cols_v = cols[valid].astype(float)
        disp_v = disp_arr[valid]

        Z = f * baseline / disp_v
        X = (cols_v - cx) * Z / f
        Y = (rows_v - cy) * Z / f

        # Filter out negative depths (can arise with min_disparity > 0).
        pos = Z > 0.0
        return np.stack([X[pos], Y[pos], Z[pos]], axis=1)
