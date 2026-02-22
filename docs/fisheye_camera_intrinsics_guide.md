# Fisheye / Omnidirectional Camera Intrinsics Guide

A step-by-step guide to calculating intrinsic parameters for **fisheye and
omnidirectional cameras** using the Kannala-Brandt equidistant model and the
`sensor_transposition` library.

---

## Table of Contents

1. [Why a Different Model?](#why-a-different-model)
2. [The Kannala-Brandt Equidistant Projection](#the-kannala-brandt-equidistant-projection)
3. [Prerequisites](#prerequisites)
4. [Step 1 – Gather Your Camera Specifications](#step-1--gather-your-camera-specifications)
5. [Step 2 – Calculate the Focal Length](#step-2--calculate-the-focal-length)
6. [Step 3 – Determine the Principal Point](#step-3--determine-the-principal-point)
7. [Step 4 – Assemble the Camera Matrix K](#step-4--assemble-the-camera-matrix-k)
8. [Step 5 – Project and Unproject Points](#step-5--project-and-unproject-points)
9. [Step 6 – Account for Fisheye Distortion](#step-6--account-for-fisheye-distortion)
10. [Omnidirectional Cameras (FOV > 180°)](#omnidirectional-cameras-fov--180)
11. [Putting It All Together](#putting-it-all-together)
12. [Obtaining Coefficients with OpenCV](#obtaining-coefficients-with-opencv)
13. [Troubleshooting](#troubleshooting)

---

## Why a Different Model?

The standard **pinhole model** maps a 3-D point to a pixel via
``u = fx * (X/Z) + cx``. This formula breaks down when the angle from the
optical axis approaches 90° (``Z → 0``). Fisheye and omnidirectional cameras
can cover fields of view from 180° up to 360°, so they require a model that
maps the incidence angle *θ* — not the tangent of the angle — to the image
radius.

| Model          | Projection formula  | Max FOV |
|----------------|---------------------|---------|
| Pinhole        | ``r = f · tan(θ)``  | < 180°  |
| Equidistant    | ``r = f · θ``       | ≤ 360°  |
| Equisolid      | ``r = 2f · sin(θ/2)``| ≤ 360°  |
| Stereographic  | ``r = 2f · tan(θ/2)``| < 360°  |

The `sensor_transposition` library implements the **equidistant
(Kannala-Brandt)** model, which is also used by OpenCV's `cv2.fisheye`
module and is the de-facto standard for wide-FOV robotics cameras.

---

## The Kannala-Brandt Equidistant Projection

For a 3-D point ``(X, Y, Z)`` in the camera frame:

```
θ     = atan2(sqrt(X² + Y²), Z)          # incidence angle
θ_d   = θ · (1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)   # distorted angle
r     = sqrt(X² + Y²)

u = fx · θ_d · X/r + cx
v = fy · θ_d · Y/r + cy
```

The four coefficients ``(k1, k2, k3, k4)`` describe how the actual lens
deviates from a perfect equidistant lens.  A perfect equidistant lens has
``k1 = k2 = k3 = k4 = 0``.

For the focal-length calculation, the equidistant relation ``r = f · θ``
gives:

```
f = (image_size / 2) / (fov_rad / 2)
```

---

## Prerequisites

Install `sensor_transposition` if you have not already:

```bash
pip install sensor_transposition
```

Or, from the repository source:

```bash
pip install -e ".[dev]"
```

---

## Step 1 – Gather Your Camera Specifications

You need the following values from the camera data sheet:

| Value | Example |
|-------|---------|
| Image width in pixels | 1920 |
| Image height in pixels | 1080 |
| Diagonal (or horizontal) field of view | 190 ° |

> **Tip:** Many fisheye lenses are specified by their **diagonal FOV**.  If
> you only have the diagonal FOV, you can approximate the horizontal and
> vertical FOV from the sensor's aspect ratio:
>
> ```python
> import math
> diag_fov_deg = 190.0       # diagonal FOV from the data sheet
> width, height = 1920, 1080
> # For equidistant fisheye, use the equidistant approximation:
> diag_px   = math.hypot(width, height)
> diag_fov_rad = math.radians(diag_fov_deg)
> # equidistant focal length from diagonal
> f_diag = (diag_px / 2.0) / (diag_fov_rad / 2.0)
> # back-compute horizontal / vertical FOV
> hfov = math.degrees(2 * (width  / 2.0) / f_diag)
> vfov = math.degrees(2 * (height / 2.0) / f_diag)
> print(f"HFOV: {hfov:.2f}°  VFOV: {vfov:.2f}°")
> ```

---

## Step 2 – Calculate the Focal Length

Use `fisheye_focal_length_from_fov` with the image dimension and the
corresponding FOV angle.  The function supports FOV values up to 360°.

```python
from sensor_transposition.camera_intrinsics import fisheye_focal_length_from_fov

width  = 1920   # pixels
height = 1080   # pixels
hfov   = 190.0  # horizontal field of view, degrees (may exceed 180°)
vfov   = 130.0  # vertical field of view, degrees

fx = fisheye_focal_length_from_fov(width,  hfov)
fy = fisheye_focal_length_from_fov(height, vfov)

print(f"fx = {fx:.2f} px")
print(f"fy = {fy:.2f} px")
```

> **Comparison with the pinhole formula:**
> The pinhole function `focal_length_from_fov` uses ``f = (size/2) / tan(fov/2)``
> and is restricted to FOV < 180°.  The fisheye function uses
> ``f = (size/2) / (fov_rad/2)`` and supports any FOV up to 360°.

---

## Step 3 – Determine the Principal Point

The principal point ``(cx, cy)`` is the pixel where the optical axis
pierces the image plane.  For a well-aligned fisheye sensor it is close to
the image centre:

```python
cx = width  / 2.0   # e.g. 960.0 px
cy = height / 2.0   # e.g. 540.0 px
```

For omnidirectional cameras with non-centred optics, or after running an
OpenCV fisheye calibration, use the calibrated ``(cx, cy)`` values.

---

## Step 4 – Assemble the Camera Matrix K

The same 3×3 **K** matrix is used for both pinhole and fisheye models:

```python
from sensor_transposition.camera_intrinsics import camera_matrix

K = camera_matrix(fx=fx, fy=fy, cx=cx, cy=cy)
print(K)
```

---

## Step 5 – Project and Unproject Points

Use `fisheye_project_point` and `fisheye_unproject_pixel` instead of their
pinhole counterparts:

```python
from sensor_transposition.camera_intrinsics import (
    fisheye_project_point,
    fisheye_unproject_pixel,
)
import numpy as np

# Project a 3-D point in the camera frame to pixel coordinates
point_3d = np.array([0.5, 0.3, 2.0])   # [X, Y, Z] in metres
u, v = fisheye_project_point(K, point_3d)
print(f"Projected: ({u:.1f}, {v:.1f})")

# Unproject a pixel back to a 3-D point at a known Euclidean distance (depth)
depth = float(np.linalg.norm(point_3d))   # Euclidean distance, not Z-depth
recovered = fisheye_unproject_pixel(K, (u, v), depth=depth)
print(f"Recovered: {recovered}")
print(f"Round-trip error: {np.linalg.norm(recovered - point_3d):.2e}")
```

> **Important:** `fisheye_unproject_pixel` takes the **Euclidean distance**
> from the camera origin to the point (i.e. ``‖[X, Y, Z]‖``), *not* the
> Z-coordinate depth used by `unproject_pixel` in the pinhole model.  This
> is because the equidistant model assigns equal scale to all directions.

---

## Step 6 – Account for Fisheye Distortion

Real fisheye lenses deviate from the ideal equidistant model.  The
Kannala-Brandt distortion coefficients ``(k1, k2, k3, k4)`` capture this
deviation.  If you have obtained them from a calibration tool, pass them as
`dist_coeffs`:

```python
from sensor_transposition.camera_intrinsics import (
    fisheye_distort_point,
    fisheye_undistort_point,
    fisheye_project_point,
    fisheye_unproject_pixel,
)
import numpy as np

# Coefficients from an OpenCV fisheye calibration
dist_coeffs = (0.05, -0.02, 0.003, -0.001)   # (k1, k2, k3, k4)

# Project with distortion
point_3d = np.array([0.8, 0.4, 3.0])
u, v = fisheye_project_point(K, point_3d, dist_coeffs=dist_coeffs)

# Unproject with distortion
depth = float(np.linalg.norm(point_3d))
recovered = fisheye_unproject_pixel(K, (u, v), depth=depth, dist_coeffs=dist_coeffs)
print(f"Round-trip error: {np.linalg.norm(recovered - point_3d):.2e}")

# Apply / remove distortion on normalised coordinates directly
# Normalised coordinates: (x_n, y_n) = (X/Z, Y/Z)
x_n, y_n = point_3d[0] / point_3d[2], point_3d[1] / point_3d[2]
point_norm = np.array([x_n, y_n])

point_dist   = fisheye_distort_point(point_norm, dist_coeffs)
point_undist = fisheye_undistort_point(point_dist, dist_coeffs)
print(f"Distortion round-trip error: {np.linalg.norm(point_undist - point_norm):.2e}")
```

> **If you do not have distortion coefficients:** Start with
> ``dist_coeffs = ()`` (or pass no argument).  This assumes a perfect
> equidistant lens.  For high-precision applications, obtain the
> coefficients through a calibration run (see
> [Obtaining Coefficients with OpenCV](#obtaining-coefficients-with-opencv)).

---

## Omnidirectional Cameras (FOV > 180°)

The equidistant model naturally handles fields of view greater than 180°,
including cameras that can see behind themselves.  Simply pass the full FOV
to `fisheye_focal_length_from_fov` (up to but not including 360°):

```python
# 270° omnidirectional camera on a 2000×2000 image
f = fisheye_focal_length_from_fov(image_size=2000, fov_deg=270.0)
K_omni = camera_matrix(fx=f, fy=f, cx=1000.0, cy=1000.0)

# A point *behind* the camera at θ = 135° is still within the FOV
point_behind = np.array([1.0, 0.0, -1.0])   # Z < 0
u, v = fisheye_project_point(K_omni, point_behind)
print(f"Point behind camera projects to ({u:.1f}, {v:.1f})")
```

> **Note:** Points directly behind the camera on the optical axis
> (``X = Y = 0, Z < 0``) project to ``θ = π`` and map to the image centre
> at ``(cx, cy)`` only if the FOV is exactly 360° (which is not supported
> due to the open upper bound).  For all other points at ``Z < 0``, the
> projection is well-defined as long as ``θ < fov/2``.

---

## Putting It All Together

Below is a complete, self-contained script for a fisheye camera.

```python
"""
fisheye_intrinsics_example.py
Build a fisheye camera matrix and verify it with a round-trip projection.
"""
import math
import numpy as np
from sensor_transposition.camera_intrinsics import (
    fisheye_focal_length_from_fov,
    camera_matrix,
    fisheye_project_point,
    fisheye_unproject_pixel,
)

# ── 1. Camera specifications ─────────────────────────────────────────────────
WIDTH, HEIGHT = 1920, 1080     # image resolution in pixels
HFOV_DEG      = 190.0          # horizontal FOV (may exceed 180° for fisheye)
VFOV_DEG      = 130.0          # vertical FOV

# ── 2. Focal lengths ─────────────────────────────────────────────────────────
fx = fisheye_focal_length_from_fov(WIDTH,  HFOV_DEG)
fy = fisheye_focal_length_from_fov(HEIGHT, VFOV_DEG)
print(f"fx = {fx:.2f} px,  fy = {fy:.2f} px")

# ── 3. Principal point ───────────────────────────────────────────────────────
cx, cy = WIDTH / 2.0, HEIGHT / 2.0

# ── 4. Camera matrix ─────────────────────────────────────────────────────────
K = camera_matrix(fx, fy, cx, cy)
print("Camera matrix K:")
print(K)

# ── 5. Sanity check: point on optical axis → principal point ─────────────────
u, v = fisheye_project_point(K, [0.0, 0.0, 1.0])
print(f"\nOn-axis point projects to ({u:.1f}, {v:.1f}),  expected ({cx}, {cy})")

# ── 6. Sanity check: point at 90° off-axis maps to image edge ────────────────
u_edge, _ = fisheye_project_point(K, [1.0, 0.0, 0.0])
print(f"90°-off-axis point: u = {u_edge:.1f},  expected {cx + fx * math.pi / 2:.1f}")

# ── 7. Round-trip projection ─────────────────────────────────────────────────
dist_coeffs = (0.05, -0.02, 0.003, -0.001)
test_point  = np.array([0.8, -0.4, 3.0])
depth       = float(np.linalg.norm(test_point))

u2, v2    = fisheye_project_point(K, test_point, dist_coeffs)
recovered = fisheye_unproject_pixel(K, (u2, v2), depth=depth,
                                    dist_coeffs=dist_coeffs)
print(f"\n3-D point {test_point} → pixel ({u2:.2f}, {v2:.2f})")
print(f"Unprojected:  {recovered}")
print(f"Round-trip error: {np.linalg.norm(recovered - test_point):.2e}")
```

---

## Obtaining Coefficients with OpenCV

The easiest way to calibrate a fisheye camera is to use OpenCV's
`cv2.fisheye` module with a printed checkerboard pattern.

### Quick-start script

```python
import cv2
import numpy as np
import glob

# Checkerboard dimensions (inner corners)
BOARD_W, BOARD_H = 9, 6
SQUARE_MM = 25.0   # physical square size in mm

# Prepare 3-D corner coordinates
objp = np.zeros((BOARD_W * BOARD_H, 1, 3), dtype=np.float64)
objp[:, 0, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2) * SQUARE_MM

objpoints, imgpoints = [], []

for fname in glob.glob("calibration_images/*.jpg"):
    img  = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (BOARD_W, BOARD_H), None)
    if ret:
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
        )
        objpoints.append(objp)
        imgpoints.append(corners2)

K_cv  = np.zeros((3, 3))
D_cv  = np.zeros((4, 1))          # (k1, k2, k3, k4)
rvecs = [np.zeros((1, 1, 3))] * len(objpoints)
tvecs = [np.zeros((1, 1, 3))] * len(objpoints)

cv2.fisheye.calibrate(
    objpoints, imgpoints, gray.shape[::-1],
    K_cv, D_cv, rvecs, tvecs,
    flags=(cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
           cv2.fisheye.CALIB_CHECK_COND |
           cv2.fisheye.CALIB_FIX_SKEW),
)

print("Calibrated K (OpenCV):")
print(K_cv)
print("Distortion coefficients D (k1, k2, k3, k4):")
print(D_cv.ravel())
```

### Using the result in sensor_transposition

```python
from sensor_transposition.camera_intrinsics import (
    camera_matrix, fisheye_project_point, fisheye_unproject_pixel,
)

# K_cv and D_cv are from the OpenCV calibration above
fx, fy = K_cv[0, 0], K_cv[1, 1]
cx, cy = K_cv[0, 2], K_cv[1, 2]
K = camera_matrix(fx, fy, cx, cy)

dist_coeffs = tuple(D_cv.ravel())   # (k1, k2, k3, k4)
```

> **Tip:** Collect at least **20–30 images** of the checkerboard at a
> variety of positions, angles, and distances throughout the full FOV.
> Include images where the checkerboard is near the edges and corners of
> the image to constrain the distortion model near large incidence angles.

---

## Troubleshooting

### `ValueError: fov_deg must be in (0, 360)`

The FOV you provided is ≤ 0° or ≥ 360°.  The equidistant model is defined
for any strictly positive FOV less than 360°.  Check the camera data sheet —
typical values for fisheye lenses are in the range 100° – 270°.

### `ValueError: image_size must be positive`

Make sure you are passing the pixel count (e.g. `1920`), not metres or mm.

### Projected points are shifted from expected positions

- Verify that the principal point is correct.  For cameras with
  non-centred optics (common in omnidirectional rigs), the optical axis
  may not pass through the image centre.  Use the ``(cx, cy)`` values from
  a full calibration run.
- Ensure the 3-D point coordinates are in the **camera frame** (x right,
  y down, z forward for a standard RDF camera), not in the ego or world
  frame.

### Round-trip error is large

- Make sure you pass the same ``dist_coeffs`` to both `fisheye_project_point`
  and `fisheye_unproject_pixel`.
- Ensure that `depth` is the **Euclidean distance** (``‖[X, Y, Z]‖``) rather
  than the Z-component.

### Fisheye vs. pinhole: which to use?

| Criterion                  | Fisheye model         | Pinhole model     |
|----------------------------|-----------------------|-------------------|
| FOV > 90°                  | ✓ Correct             | ✗ Inaccurate      |
| FOV > 180°                 | ✓ Supported           | ✗ Not supported   |
| Standard automotive camera | May be overkill       | ✓ Usually correct |
| Calibrated via `cv2.fisheye`| ✓ Use this model     | ✗ Wrong coefficients|
| Calibrated via `calibrateCamera` | ✗ Wrong model  | ✓ Use Brown–Conrady|
