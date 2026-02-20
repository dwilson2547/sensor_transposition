"""
lidar_camera.py

Tools for fusing LiDAR point-cloud data with camera imagery:

* :func:`project_lidar_to_image` – project 3-D LiDAR points onto the 2-D image
  plane of a calibrated camera.
* :func:`color_lidar_from_image` – sample image colour data at each projected
  LiDAR point location, effectively "painting" the point cloud with colours from
  the image.

Both functions operate purely on numpy arrays so they integrate naturally with
:class:`~sensor_transposition.sensor_collection.SensorCollection` extrinsic /
intrinsic data or with any other calibration source.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def project_lidar_to_image(
    points: np.ndarray,
    lidar_to_camera: np.ndarray,
    camera_matrix: np.ndarray,
    image_width: int,
    image_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project LiDAR points onto a camera image plane.

    Transforms each 3-D point from the LiDAR coordinate frame to the camera
    coordinate frame using *lidar_to_camera*, then applies the pinhole
    projection model defined by *camera_matrix* to obtain 2-D pixel
    coordinates.  Only points that are in front of the camera (positive depth)
    and whose projections fall within the image bounds are marked as valid.

    Args:
        points: ``(N, 3)`` float array of 3-D points in the LiDAR frame
            ``[x, y, z]``.
        lidar_to_camera: 4×4 homogeneous transform matrix that converts
            points from the LiDAR frame to the camera frame.
        camera_matrix: 3×3 camera intrinsic matrix **K**::

                [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]

        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        pixel_coords: ``(N, 2)`` float array of ``(u, v)`` pixel coordinates.
            Entries for invalid points are set to ``(0, 0)`` and should be
            ignored.
        valid_mask: ``(N,)`` boolean array.  ``True`` where the projected point
            is in front of the camera *and* within the image bounds.

    Raises:
        ValueError: If *points* is not a 2-D array with 3 columns, or if
            *lidar_to_camera* is not 4×4, or if *camera_matrix* is not 3×3.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(
            f"points must be shape (N, 3), got {pts.shape}."
        )
    T = np.asarray(lidar_to_camera, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(
            f"lidar_to_camera must be shape (4, 4), got {T.shape}."
        )
    K = np.asarray(camera_matrix, dtype=float)
    if K.shape != (3, 3):
        raise ValueError(
            f"camera_matrix must be shape (3, 3), got {K.shape}."
        )
    if image_width <= 0:
        raise ValueError(f"image_width must be positive, got {image_width}.")
    if image_height <= 0:
        raise ValueError(f"image_height must be positive, got {image_height}.")

    n = pts.shape[0]

    # Homogeneous coordinates: (N, 4)
    ones = np.ones((n, 1), dtype=float)
    pts_h = np.hstack([pts, ones])

    # Transform to camera frame: (N, 3)
    pts_cam = (T @ pts_h.T).T[:, :3]

    # Depth (Z in camera frame)
    depth = pts_cam[:, 2]
    in_front = depth > 0.0

    # Initialize outputs
    pixel_coords = np.zeros((n, 2), dtype=float)
    valid_mask = np.zeros(n, dtype=bool)

    if not np.any(in_front):
        return pixel_coords, valid_mask

    # Project only in-front points to avoid division by zero
    idx = np.where(in_front)[0]
    pts_front = pts_cam[idx]          # (M, 3)

    # Apply K: project to normalised then to pixel coords
    projected = (K @ pts_front.T).T  # (M, 3)
    u = projected[:, 0] / projected[:, 2]
    v = projected[:, 1] / projected[:, 2]

    # Check image bounds
    in_bounds = (
        (u >= 0.0) & (u < image_width)
        & (v >= 0.0) & (v < image_height)
    )

    valid_idx = idx[in_bounds]
    pixel_coords[valid_idx, 0] = u[in_bounds]
    pixel_coords[valid_idx, 1] = v[in_bounds]
    valid_mask[valid_idx] = True

    return pixel_coords, valid_mask


def color_lidar_from_image(
    points: np.ndarray,
    lidar_to_camera: np.ndarray,
    camera_matrix: np.ndarray,
    image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample image colour data at projected LiDAR point locations.

    Projects each LiDAR point onto the image plane (nearest-neighbour
    sampling) and returns the pixel colour at that location.  Points that do
    not project onto the image receive a zero colour value and are flagged in
    the *valid_mask*.

    Args:
        points: ``(N, 3)`` float array of 3-D points in the LiDAR frame.
        lidar_to_camera: 4×4 homogeneous transform matrix that converts
            points from the LiDAR frame to the camera frame.
        camera_matrix: 3×3 camera intrinsic matrix **K**.
        image: ``(H, W, C)`` or ``(H, W)`` numpy array representing the image.
            The dtype is preserved in the output.

    Returns:
        colors: ``(N, C)`` array (or ``(N,)`` for a single-channel image) of
            colour values sampled at each projected point.  Invalid points
            contain zeros.
        valid_mask: ``(N,)`` boolean array.  ``True`` where a colour was
            successfully sampled from the image.

    Raises:
        ValueError: If *points* is not a 2-D array with 3 columns, or if the
            image array does not have 2 or 3 dimensions.
    """
    img = np.asarray(image)
    if img.ndim not in (2, 3):
        raise ValueError(
            f"image must be 2-D (H, W) or 3-D (H, W, C), got ndim={img.ndim}."
        )

    image_height, image_width = img.shape[:2]

    pixel_coords, valid_mask = project_lidar_to_image(
        points, lidar_to_camera, camera_matrix, image_width, image_height
    )

    n = pixel_coords.shape[0]

    if img.ndim == 3:
        num_channels = img.shape[2]
        colors = np.zeros((n, num_channels), dtype=img.dtype)
    else:
        colors = np.zeros(n, dtype=img.dtype)

    if np.any(valid_mask):
        # Round to nearest pixel
        u_px = np.round(pixel_coords[valid_mask, 0]).astype(int)
        v_px = np.round(pixel_coords[valid_mask, 1]).astype(int)

        # Clamp to valid range (should already be within bounds, but be safe)
        u_px = np.clip(u_px, 0, image_width - 1)
        v_px = np.clip(v_px, 0, image_height - 1)

        colors[valid_mask] = img[v_px, u_px]

    return colors, valid_mask
