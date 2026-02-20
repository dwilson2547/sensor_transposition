"""
camera_intrinsics.py

Tools for calculating pinhole camera intrinsic parameters.

The pinhole camera model maps a 3-D point (X, Y, Z) in the camera frame to a
2-D pixel (u, v) via::

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

Focal lengths *fx* and *fy* can be derived from the sensor geometry or from
the diagonal / horizontal / vertical field-of-view angles.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Focal-length derivation
# ---------------------------------------------------------------------------


def focal_length_from_fov(image_size: int, fov_deg: float) -> float:
    """Calculate focal length in pixels from image size and field-of-view.

    Uses the relation::

        f = (image_size / 2) / tan(fov / 2)

    Args:
        image_size: Image width (for horizontal FOV) or height (for vertical
            FOV) in pixels.
        fov_deg: Field of view in degrees corresponding to *image_size*.

    Returns:
        Focal length in pixels.
    """
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}.")
    if not (0.0 < fov_deg < 180.0):
        raise ValueError(f"fov_deg must be in (0, 180), got {fov_deg}.")
    return (image_size / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def focal_length_from_sensor(
    image_size_px: int,
    sensor_size_mm: float,
    focal_length_mm: float,
) -> float:
    """Calculate focal length in pixels from physical sensor geometry.

    Uses the relation::

        f_px = focal_length_mm * (image_size_px / sensor_size_mm)

    Args:
        image_size_px: Image width or height in pixels.
        sensor_size_mm: Physical sensor width or height in millimetres.
        focal_length_mm: Physical focal length in millimetres.

    Returns:
        Focal length in pixels.
    """
    if image_size_px <= 0:
        raise ValueError(f"image_size_px must be positive, got {image_size_px}.")
    if sensor_size_mm <= 0:
        raise ValueError(f"sensor_size_mm must be positive, got {sensor_size_mm}.")
    if focal_length_mm <= 0:
        raise ValueError(f"focal_length_mm must be positive, got {focal_length_mm}.")
    return focal_length_mm * (image_size_px / sensor_size_mm)


def fov_from_focal_length(focal_length_px: float, image_size_px: int) -> float:
    """Calculate field-of-view in degrees from focal length and image size.

    Args:
        focal_length_px: Focal length in pixels.
        image_size_px: Image width or height in pixels.

    Returns:
        Field of view in degrees.
    """
    if focal_length_px <= 0:
        raise ValueError(f"focal_length_px must be positive, got {focal_length_px}.")
    if image_size_px <= 0:
        raise ValueError(f"image_size_px must be positive, got {image_size_px}.")
    return math.degrees(2.0 * math.atan((image_size_px / 2.0) / focal_length_px))


# ---------------------------------------------------------------------------
# Camera matrix helpers
# ---------------------------------------------------------------------------


def camera_matrix(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Construct the 3×3 camera intrinsic matrix K.

    Args:
        fx: Focal length along x in pixels.
        fy: Focal length along y in pixels.
        cx: Principal point x-coordinate in pixels.
        cy: Principal point y-coordinate in pixels.

    Returns:
        3×3 numpy array::

            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
    """
    return np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )


def project_point(
    K: np.ndarray,
    point_camera: np.ndarray,
) -> Tuple[float, float]:
    """Project a 3-D point in the camera frame to 2-D pixel coordinates.

    Args:
        K: 3×3 camera intrinsic matrix.
        point_camera: (3,) array [X, Y, Z] in the camera frame.  Z must be > 0.

    Returns:
        (u, v) pixel coordinates.
    """
    pt = np.asarray(point_camera, dtype=float)
    if pt.shape != (3,):
        raise ValueError(f"point_camera must be shape (3,), got {pt.shape}.")
    if pt[2] <= 0:
        raise ValueError(f"Point is behind the camera (Z={pt[2]}).")
    projected = K @ pt
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def unproject_pixel(
    K: np.ndarray,
    pixel: Tuple[float, float],
    depth: float,
) -> np.ndarray:
    """Unproject a 2-D pixel at a given depth to a 3-D camera-frame point.

    Args:
        K: 3×3 camera intrinsic matrix.
        pixel: (u, v) pixel coordinates.
        depth: Depth (Z value) in the camera frame.

    Returns:
        (3,) array [X, Y, Z] in the camera frame.
    """
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}.")
    u, v = float(pixel[0]), float(pixel[1])
    K_inv = np.linalg.inv(K)
    return K_inv @ (np.array([u, v, 1.0]) * depth)


# ---------------------------------------------------------------------------
# Distortion / undistortion (Brown–Conrady model)
# ---------------------------------------------------------------------------


def distort_point(
    point_normalised: np.ndarray,
    dist_coeffs: Tuple[float, ...],
) -> np.ndarray:
    """Apply radial/tangential lens distortion to a normalised image point.

    Uses the Brown–Conrady model with up to five coefficients
    ``(k1, k2, p1, p2, k3)``.

    Args:
        point_normalised: (2,) normalised (undistorted) coordinates [x_n, y_n]
            where ``x_n = (u - cx) / fx``.
        dist_coeffs: Distortion coefficients ``(k1, k2, p1, p2, k3)``.

    Returns:
        (2,) distorted normalised coordinates.
    """
    x, y = float(point_normalised[0]), float(point_normalised[1])
    k1, k2, p1, p2, k3 = (list(dist_coeffs) + [0.0] * 5)[:5]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.array([x_d, y_d], dtype=float)


def undistort_point(
    point_distorted: np.ndarray,
    dist_coeffs: Tuple[float, ...],
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Remove lens distortion from a normalised image point (iterative method).

    Args:
        point_distorted: (2,) distorted normalised coordinates.
        dist_coeffs: Distortion coefficients ``(k1, k2, p1, p2, k3)``.
        max_iterations: Maximum number of Newton-Raphson iterations.
        tolerance: Convergence tolerance.

    Returns:
        (2,) undistorted normalised coordinates.
    """
    x_d = np.asarray(point_distorted, dtype=float).copy()
    x_u = x_d.copy()
    for _ in range(max_iterations):
        x_d_hat = distort_point(x_u, dist_coeffs)
        error = x_d - x_d_hat
        x_u = x_u + error
        if np.linalg.norm(error) < tolerance:
            break
    return x_u
