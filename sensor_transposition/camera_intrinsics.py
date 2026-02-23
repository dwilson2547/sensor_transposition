"""
camera_intrinsics.py

Tools for calculating pinhole and fisheye camera intrinsic parameters, and for
modelling rolling-shutter cameras.

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

The rolling-shutter model corrects for the per-row exposure delay present in
CMOS sensors.  Each row is captured at a slightly different time, so a moving
camera introduces geometric distortion.  The first-order correction shifts a
camera-frame point by the camera's instantaneous velocity scaled by the row's
time offset from the frame start::

    p_corrected ≈ p - t_row * (v + ω × p)
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


# ---------------------------------------------------------------------------
# Rolling-shutter model
# ---------------------------------------------------------------------------


def rolling_shutter_row_time(
    row: int,
    image_height: int,
    readout_time: float,
) -> float:
    """Return the time offset from frame start for a given image row.

    A rolling-shutter sensor exposes one row at a time.  The first row is
    captured at ``t = 0`` and the last row at ``t = readout_time``.  Each
    intermediate row is captured at a linearly interpolated time::

        t_row = (row / (image_height - 1)) * readout_time

    Args:
        row: Row index (0-indexed).  Must satisfy ``0 <= row < image_height``.
        image_height: Total number of rows in the image.  Must be >= 2.
        readout_time: Total time to read out the full frame in seconds.
            Must be non-negative.  Pass ``0`` to model a global-shutter camera.

    Returns:
        Time offset in seconds from the frame start for the given row.
    """
    if image_height < 2:
        raise ValueError(f"image_height must be >= 2, got {image_height}.")
    if not (0 <= row < image_height):
        raise ValueError(
            f"row must be in [0, {image_height - 1}], got {row}."
        )
    if readout_time < 0:
        raise ValueError(
            f"readout_time must be non-negative, got {readout_time}."
        )
    return (row / (image_height - 1)) * readout_time


def rolling_shutter_correct_point(
    point_camera: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    row_time: float,
) -> np.ndarray:
    """Apply first-order rolling-shutter motion correction to a camera-frame point.

    Models the apparent shift of a 3-D point due to camera motion between
    the frame-start reference time and the time at which the point's row is
    captured.  Under a constant-velocity assumption and a first-order
    (small-angle) approximation the corrected point is::

        p_corrected ≈ p - row_time * (v + ω × p)

    where *v* is the camera's linear velocity and *ω* is its angular velocity,
    both expressed in the camera frame.

    Args:
        point_camera: (3,) array ``[X, Y, Z]`` in the camera frame at the
            frame-start reference time.
        linear_velocity: (3,) linear velocity ``[vx, vy, vz]`` of the camera
            expressed in the camera frame, in metres per second.
        angular_velocity: (3,) angular velocity ``[ωx, ωy, ωz]`` of the
            camera expressed in the camera frame, in radians per second.
        row_time: Time offset from the frame start in seconds for the row at
            which the point is captured (see :func:`rolling_shutter_row_time`).

    Returns:
        (3,) corrected camera-frame point.
    """
    p = np.asarray(point_camera, dtype=float)
    v = np.asarray(linear_velocity, dtype=float)
    omega = np.asarray(angular_velocity, dtype=float)
    if p.shape != (3,):
        raise ValueError(f"point_camera must be shape (3,), got {p.shape}.")
    if v.shape != (3,):
        raise ValueError(
            f"linear_velocity must be shape (3,), got {v.shape}."
        )
    if omega.shape != (3,):
        raise ValueError(
            f"angular_velocity must be shape (3,), got {omega.shape}."
        )
    return p - row_time * (v + np.cross(omega, p))


def rolling_shutter_project_point(
    K: np.ndarray,
    point_camera: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    image_height: int,
    readout_time: float,
    dist_coeffs: Tuple[float, ...] = (),
    max_iterations: int = 10,
    tolerance: float = 1e-6,
) -> Tuple[float, float]:
    """Project a camera-frame point to pixel coordinates with rolling-shutter correction.

    Iteratively solves for the pixel row at which the scene point is captured.
    Because the row determines the row_time, which determines the corrected
    camera-frame point, which in turn determines the projected row, a
    fixed-point iteration is used:

    1. Project the undistorted point at reference time to obtain an initial
       row estimate.
    2. Compute the ``row_time`` for that row using
       :func:`rolling_shutter_row_time`.
    3. Apply :func:`rolling_shutter_correct_point` to obtain the point in the
       camera frame at ``row_time``.
    4. Re-project (with optional distortion) and repeat until the v-coordinate
       converges.

    Setting ``readout_time=0`` recovers the standard global-shutter pinhole
    projection (possibly with Brown–Conrady distortion).

    Args:
        K: 3×3 camera intrinsic matrix.
        point_camera: (3,) array ``[X, Y, Z]`` in the camera frame at the
            frame-start reference time.  Z must be > 0.
        linear_velocity: (3,) linear velocity of the camera in the camera
            frame, in metres per second.
        angular_velocity: (3,) angular velocity of the camera in the camera
            frame, in radians per second.
        image_height: Total number of rows in the image.  Must be >= 2.
        readout_time: Total frame readout time in seconds.  Must be
            non-negative.
        dist_coeffs: Brown–Conrady distortion coefficients
            ``(k1, k2, p1, p2, k3)``.  Defaults to no distortion.
        max_iterations: Maximum fixed-point iterations for row convergence.
        tolerance: Convergence tolerance on the v (row) coordinate in pixels.

    Returns:
        ``(u, v)`` pixel coordinates after rolling-shutter correction.
    """
    pt = np.asarray(point_camera, dtype=float)
    if pt.shape != (3,):
        raise ValueError(f"point_camera must be shape (3,), got {pt.shape}.")
    if pt[2] <= 0:
        raise ValueError(f"Point is behind the camera (Z={pt[2]}).")
    if image_height < 2:
        raise ValueError(f"image_height must be >= 2, got {image_height}.")
    if readout_time < 0:
        raise ValueError(
            f"readout_time must be non-negative, got {readout_time}."
        )

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def _project(p3d: np.ndarray) -> Tuple[float, float]:
        x_n, y_n = p3d[0] / p3d[2], p3d[1] / p3d[2]
        if dist_coeffs:
            x_n, y_n = distort_point(np.array([x_n, y_n]), dist_coeffs)
        return fx * x_n + cx, fy * y_n + cy

    # Seed: project at reference time (row_time = 0)
    u, v = _project(pt)

    for _ in range(max_iterations):
        v_clamped = float(np.clip(v, 0.0, image_height - 1))
        row_time = (v_clamped / (image_height - 1)) * readout_time
        pt_corr = rolling_shutter_correct_point(
            pt, linear_velocity, angular_velocity, row_time
        )
        if pt_corr[2] <= 0:
            raise ValueError(
                "Rolling-shutter corrected point is behind the camera "
                f"(Z={pt_corr[2]})."
            )
        u_new, v_new = _project(pt_corr)
        if abs(v_new - v) < tolerance:
            u, v = u_new, v_new
            break
        u, v = u_new, v_new

    return float(u), float(v)


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
