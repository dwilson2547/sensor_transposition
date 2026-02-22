"""
camera_intrinsics.py

Tools for calculating pinhole and fisheye camera intrinsic parameters.

The pinhole camera model maps a 3-D point (X, Y, Z) in the camera frame to a
2-D pixel (u, v) via::

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

Focal lengths *fx* and *fy* can be derived from the sensor geometry or from
the diagonal / horizontal / vertical field-of-view angles.

The fisheye (Kannala-Brandt) model uses the equidistant projection::

    r = f * θ_d
    θ_d = θ * (1 + k1*θ² + k2*θ⁴ + k3*θ⁶ + k4*θ⁸)
    θ = atan2(sqrt(X² + Y²), Z)

where *r* is the distance from the principal point to the projected pixel.
This model supports fields of view up to 360°, making it suitable for
omnidirectional cameras.
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


# ---------------------------------------------------------------------------
# Fisheye / omnidirectional camera model (Kannala-Brandt)
# ---------------------------------------------------------------------------


def fisheye_focal_length_from_fov(image_size: int, fov_deg: float) -> float:
    """Calculate focal length in pixels for an equidistant fisheye camera.

    The equidistant (equiangular) projection maps the incidence angle *θ* to
    the image radius *r* via ``r = f * θ``.  At the image edge
    ``r = image_size / 2`` and ``θ = fov_rad / 2``, giving::

        f = (image_size / 2) / (fov_rad / 2)

    This model is valid for fields of view up to 360°, making it suitable for
    omnidirectional cameras.

    Args:
        image_size: Image width (for horizontal FOV) or height (for vertical
            FOV) in pixels.
        fov_deg: Field of view in degrees.  Must be in ``(0, 360)``.

    Returns:
        Focal length in pixels.
    """
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}.")
    if not (0.0 < fov_deg < 360.0):
        raise ValueError(f"fov_deg must be in (0, 360), got {fov_deg}.")
    fov_rad = math.radians(fov_deg)
    return (image_size / 2.0) / (fov_rad / 2.0)


def fisheye_distort_point(
    point_normalised: np.ndarray,
    dist_coeffs: Tuple[float, ...] = (),
) -> np.ndarray:
    """Apply Kannala-Brandt fisheye distortion to a normalised image point.

    The Kannala-Brandt model maps the incidence angle *θ* to the distorted
    angle *θ_d*::

        θ_d = θ * (1 + k1*θ² + k2*θ⁴ + k3*θ⁶ + k4*θ⁸)

    where ``θ = atan(r)`` and ``r = sqrt(x_n² + y_n²)`` for normalised
    coordinates ``(x_n, y_n) = (X/Z, Y/Z)``.

    Args:
        point_normalised: (2,) undistorted normalised coordinates
            ``[x_n, y_n]`` where ``x_n = X/Z``, ``y_n = Y/Z``.
        dist_coeffs: Kannala-Brandt coefficients ``(k1, k2, k3, k4)``.
            Defaults to no distortion.

    Returns:
        (2,) distorted normalised coordinates.
    """
    x, y = float(point_normalised[0]), float(point_normalised[1])
    k1, k2, k3, k4 = (list(dist_coeffs) + [0.0] * 4)[:4]
    r = math.sqrt(x * x + y * y)
    if r < 1e-8:
        return np.array([x, y], dtype=float)
    theta = math.atan(r)
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
    scale = theta_d / r
    return np.array([x * scale, y * scale], dtype=float)


def fisheye_undistort_point(
    point_distorted: np.ndarray,
    dist_coeffs: Tuple[float, ...] = (),
    max_iterations: int = 20,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Remove Kannala-Brandt fisheye distortion from a normalised image point.

    Inverts the distortion polynomial using Newton-Raphson iteration to recover
    the undistorted normalised coordinates.

    Args:
        point_distorted: (2,) distorted normalised coordinates.
        dist_coeffs: Kannala-Brandt coefficients ``(k1, k2, k3, k4)``.
            Defaults to no distortion.
        max_iterations: Maximum Newton-Raphson iterations.
        tolerance: Convergence tolerance on the angle correction.

    Returns:
        (2,) undistorted normalised coordinates.
    """
    x_d, y_d = float(point_distorted[0]), float(point_distorted[1])
    k1, k2, k3, k4 = (list(dist_coeffs) + [0.0] * 4)[:4]
    r_d = math.sqrt(x_d * x_d + y_d * y_d)
    if r_d < 1e-8:
        return np.array([x_d, y_d], dtype=float)
    # Newton-Raphson: find θ such that θ_d(θ) = r_d
    theta = r_d
    for _ in range(max_iterations):
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
        f_prime = 1.0 + 3.0 * k1 * theta2 + 5.0 * k2 * theta4 + 7.0 * k3 * theta6 + 9.0 * k4 * theta8
        delta = (theta_d - r_d) / f_prime if abs(f_prime) > 1e-12 else 0.0
        theta -= delta
        if abs(delta) < tolerance:
            break
    # Recover undistorted normalised radius: r = tan(θ)
    r_u = math.tan(theta)
    scale = r_u / r_d
    return np.array([x_d * scale, y_d * scale], dtype=float)


def fisheye_project_point(
    K: np.ndarray,
    point_camera: np.ndarray,
    dist_coeffs: Tuple[float, ...] = (),
) -> Tuple[float, float]:
    """Project a 3-D point to pixel coordinates using the fisheye camera model.

    Uses the Kannala-Brandt equidistant projection::

        θ   = atan2(sqrt(X² + Y²), Z)
        θ_d = θ * (1 + k1*θ² + k2*θ⁴ + k3*θ⁶ + k4*θ⁸)
        u   = fx * θ_d * X / sqrt(X² + Y²) + cx
        v   = fy * θ_d * Y / sqrt(X² + Y²) + cy

    Unlike the pinhole model, this projection works for points with large
    off-axis angles (including angles ≥ 90°), making it suitable for
    fisheye and omnidirectional cameras.

    Args:
        K: 3×3 camera intrinsic matrix.
        point_camera: (3,) array ``[X, Y, Z]`` in the camera frame.
        dist_coeffs: Kannala-Brandt coefficients ``(k1, k2, k3, k4)``.
            Defaults to no distortion.

    Returns:
        (u, v) pixel coordinates.
    """
    pt = np.asarray(point_camera, dtype=float)
    if pt.shape != (3,):
        raise ValueError(f"point_camera must be shape (3,), got {pt.shape}.")
    X, Y, Z = pt
    r_xy = math.sqrt(X * X + Y * Y)
    theta = math.atan2(r_xy, Z)
    k1, k2, k3, k4 = (list(dist_coeffs) + [0.0] * 4)[:4]
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
    if r_xy < 1e-8:
        x_d, y_d = 0.0, 0.0
    else:
        x_d = theta_d * X / r_xy
        y_d = theta_d * Y / r_xy
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return float(fx * x_d + cx), float(fy * y_d + cy)


def fisheye_unproject_pixel(
    K: np.ndarray,
    pixel: Tuple[float, float],
    depth: float,
    dist_coeffs: Tuple[float, ...] = (),
    max_iterations: int = 20,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Unproject a fisheye pixel to a 3-D camera-frame point.

    Inverts the Kannala-Brandt projection.  *depth* is the Euclidean distance
    from the camera origin to the 3-D point (i.e. ``‖[X, Y, Z]‖``).

    Args:
        K: 3×3 camera intrinsic matrix.
        pixel: (u, v) pixel coordinates.
        depth: Euclidean distance from the camera origin in metres.  Must be
            positive.
        dist_coeffs: Kannala-Brandt coefficients ``(k1, k2, k3, k4)``.
            Defaults to no distortion.
        max_iterations: Maximum Newton-Raphson iterations for distortion
            inversion.
        tolerance: Convergence tolerance.

    Returns:
        (3,) array ``[X, Y, Z]`` in the camera frame.
    """
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}.")
    u, v = float(pixel[0]), float(pixel[1])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_d = (u - cx) / fx
    y_d = (v - cy) / fy
    r_d = math.sqrt(x_d * x_d + y_d * y_d)
    if r_d < 1e-8:
        return np.array([0.0, 0.0, depth], dtype=float)
    # Invert the distortion polynomial: find θ such that θ_d(θ) = r_d
    k1, k2, k3, k4 = (list(dist_coeffs) + [0.0] * 4)[:4]
    theta = r_d
    for _ in range(max_iterations):
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
        f_prime = 1.0 + 3.0 * k1 * theta2 + 5.0 * k2 * theta4 + 7.0 * k3 * theta6 + 9.0 * k4 * theta8
        delta = (theta_d - r_d) / f_prime if abs(f_prime) > 1e-12 else 0.0
        theta -= delta
        if abs(delta) < tolerance:
            break
    # Reconstruct the 3-D direction unit vector
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    direction = np.array([
        sin_theta * x_d / r_d,
        sin_theta * y_d / r_d,
        cos_theta,
    ], dtype=float)
    return direction * depth
