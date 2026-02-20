# Camera Intrinsics Guide

A step-by-step guide to calculating your camera's intrinsic parameters using
the **pinhole model** and the `sensor_transposition` library.

---

## Table of Contents

1. [What Are Camera Intrinsics?](#what-are-camera-intrinsics)
2. [Prerequisites](#prerequisites)
3. [Step 1 – Gather Your Camera Specifications](#step-1--gather-your-camera-specifications)
4. [Step 2 – Calculate the Focal Lengths](#step-2--calculate-the-focal-lengths)
   - [Method A: From Field of View](#method-a-from-field-of-view)
   - [Method B: From Physical Sensor Geometry](#method-b-from-physical-sensor-geometry)
5. [Step 3 – Determine the Principal Point](#step-3--determine-the-principal-point)
6. [Step 4 – Assemble the Camera Matrix K](#step-4--assemble-the-camera-matrix-k)
7. [Step 5 – Verify With Point Projection](#step-5--verify-with-point-projection)
8. [Step 6 – Account for Lens Distortion](#step-6--account-for-lens-distortion)
9. [Putting It All Together](#putting-it-all-together)
10. [Troubleshooting](#troubleshooting)

---

## What Are Camera Intrinsics?

The **pinhole camera model** describes how a 3-D point `(X, Y, Z)` in the
camera frame maps to a 2-D pixel `(u, v)` in the image:

```
u = fx * (X / Z) + cx
v = fy * (Y / Z) + cy
```

The four numbers that describe this mapping are called the *intrinsic
parameters*:

| Parameter | Meaning |
|-----------|---------|
| `fx` | Focal length along the x-axis, in pixels |
| `fy` | Focal length along the y-axis, in pixels |
| `cx` | Principal-point x-coordinate (optical axis in the image), in pixels |
| `cy` | Principal-point y-coordinate (optical axis in the image), in pixels |

Together they form the **3×3 camera matrix K**:

```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
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

You need at least **one** of the following pieces of information from your
camera's data sheet or manufacturer documentation:

**Option A – Field of View (FOV)**

| Value | Example |
|-------|---------|
| Image width in pixels | 1920 |
| Image height in pixels | 1080 |
| Horizontal FOV in degrees | 90 ° |
| Vertical FOV in degrees | 59 ° |

> **Tip:** If only a diagonal FOV is provided, you can derive horizontal and
> vertical FOV from the image aspect ratio:
>
> ```python
> import math
> diag_fov_deg = 110.0       # diagonal FOV from the data sheet
> width, height = 1920, 1080
> diag_px = math.hypot(width, height)
> hfov = math.degrees(2 * math.atan(math.tan(math.radians(diag_fov_deg) / 2) * width  / diag_px))
> vfov = math.degrees(2 * math.atan(math.tan(math.radians(diag_fov_deg) / 2) * height / diag_px))
> print(f"HFOV: {hfov:.2f}°  VFOV: {vfov:.2f}°")
> ```

**Option B – Physical Sensor Size and Focal Length**

| Value | Example |
|-------|---------|
| Image width in pixels | 1920 |
| Image height in pixels | 1080 |
| Sensor width in mm | 6.40 mm |
| Sensor height in mm | 4.80 mm |
| Focal length in mm | 4.35 mm |

> **Tip:** Sensor size and focal length in mm are usually listed in the
> camera's technical specification sheet.  For a smartphone or embedded
> camera, the EXIF data embedded in a raw photo often contains the focal
> length and sensor crop factor needed to derive the physical sensor size.

---

## Step 2 – Calculate the Focal Lengths

### Method A: From Field of View

Use `focal_length_from_fov` when you know the image resolution and the FOV
angle for that axis.

```python
from sensor_transposition.camera_intrinsics import focal_length_from_fov

width  = 1920   # pixels
height = 1080   # pixels
hfov   = 90.0   # horizontal field of view, degrees
vfov   = 59.0   # vertical field of view, degrees

fx = focal_length_from_fov(width,  hfov)
fy = focal_length_from_fov(height, vfov)

print(f"fx = {fx:.2f} px")   # → fx = 960.00 px
print(f"fy = {fy:.2f} px")   # → fy = 954.45 px
```

> **Note:** For an ideal lens with square pixels `fx ≈ fy`.  Small
> differences are normal; a large discrepancy (> 5 %) suggests the FOV
> values are inconsistent with the aspect ratio — double-check the data
> sheet.

### Method B: From Physical Sensor Geometry

Use `focal_length_from_sensor` when you know the physical focal length and
sensor size in mm.

```python
from sensor_transposition.camera_intrinsics import focal_length_from_sensor

width_px       = 1920   # pixels
height_px      = 1080   # pixels
sensor_w_mm    = 6.40   # physical sensor width in mm
sensor_h_mm    = 4.80   # physical sensor height in mm
focal_len_mm   = 4.35   # physical focal length in mm

fx = focal_length_from_sensor(width_px,  sensor_w_mm, focal_len_mm)
fy = focal_length_from_sensor(height_px, sensor_h_mm, focal_len_mm)

print(f"fx = {fx:.2f} px")   # → fx = 1305.00 px
print(f"fy = {fy:.2f} px")   # → fy = 978.75 px
```

> **Pixel pitch check:** The pixel pitch (mm per pixel) is
> `sensor_size_mm / image_size_px`.  If `fx / fy` does not match
> `(sensor_h_mm / height_px) / (sensor_w_mm / width_px)` the pixels are
> non-square, which is unusual for modern sensors.

---

## Step 3 – Determine the Principal Point

The **principal point** `(cx, cy)` is the pixel where the optical axis
pierces the image plane.  For a well-aligned sensor it is very close to the
image centre:

```python
cx = width  / 2.0   # 960.0 px for a 1920-wide image
cy = height / 2.0   # 540.0 px for a 1080-tall image
```

Use the image-centre approximation unless you have run a full
checkerboard calibration (e.g. with OpenCV's `calibrateCamera`) that gives
you a more precise value.

---

## Step 4 – Assemble the Camera Matrix K

Pass the four values to `camera_matrix` to obtain the standard 3×3 **K**
matrix:

```python
from sensor_transposition.camera_intrinsics import camera_matrix

K = camera_matrix(fx=fx, fy=fy, cx=cx, cy=cy)
print(K)
# [[ 960.      0.    960. ]
#  [   0.    954.45  540. ]
#  [   0.      0.      1. ]]
```

`K` is a plain `numpy.ndarray` (dtype `float64`) that you can pass directly
to any downstream function in this library or to OpenCV.

---

## Step 5 – Verify With Point Projection

A quick sanity check: project a point that you know should land at a
predictable pixel location.

```python
from sensor_transposition.camera_intrinsics import project_point, unproject_pixel
import numpy as np

# A point 1 metre directly in front of the camera (on the optical axis)
# should project to the principal point (cx, cy).
point_on_axis = np.array([0.0, 0.0, 1.0])
u, v = project_point(K, point_on_axis)
print(f"Projected: ({u:.1f}, {v:.1f})")   # → (960.0, 540.0)  ✓

# Round-trip: unproject back to 3-D at depth 1 m
recovered = unproject_pixel(K, (u, v), depth=1.0)
print(f"Recovered: {recovered}")   # → [0. 0. 1.]  ✓

# A point 0.5 m to the right of the axis (X = 0.5 m, Z = 2 m)
point_right = np.array([0.5, 0.0, 2.0])
u2, v2 = project_point(K, point_right)
print(f"Right-of-axis: ({u2:.1f}, {v2:.1f})")
# Expected: u ≈ cx + fx * (0.5 / 2) = 960 + 240 = 1200
```

---

## Step 6 – Account for Lens Distortion

Real lenses introduce radial and tangential distortion that shifts pixels
away from where the ideal pinhole model places them.  The library uses the
**Brown–Conrady model** with coefficients `(k1, k2, p1, p2, k3)`.

If you obtained distortion coefficients from a calibration tool (e.g. OpenCV
`calibrateCamera`, MATLAB Camera Calibrator, or ROS `camera_calibration`),
you can apply and remove distortion as follows:

```python
from sensor_transposition.camera_intrinsics import distort_point, undistort_point
import numpy as np

# Example coefficients from a calibration run
dist_coeffs = (-0.25, 0.12, 0.0, 0.0, 0.0)   # (k1, k2, p1, p2, k3)

# Convert a pixel to normalised coordinates before calling distort/undistort
u_raw, v_raw = 1100.0, 620.0
x_n = (u_raw - cx) / fx
y_n = (v_raw - cy) / fy
point_norm = np.array([x_n, y_n])

# Apply distortion (normalised → distorted normalised)
point_dist = distort_point(point_norm, dist_coeffs)

# Remove distortion (distorted normalised → undistorted normalised)
point_undist = undistort_point(point_dist, dist_coeffs)
print(f"Round-trip error: {np.linalg.norm(point_undist - point_norm):.2e}")
# → should be < 1e-7

# Convert undistorted normalised coordinates back to pixels
u_corr = point_undist[0] * fx + cx
v_corr = point_undist[1] * fy + cy
print(f"Corrected pixel: ({u_corr:.2f}, {v_corr:.2f})")
```

> **If you do not have distortion coefficients:** Start with all zeros
> `(0, 0, 0, 0, 0)`.  For cameras with wide-angle or fisheye lenses the
> uncorrected images will look noticeably barrel-distorted; in that case
> obtain the coefficients through a checkerboard calibration session.

---

## Putting It All Together

Below is a complete, self-contained script that builds a camera matrix and
performs a projection round-trip.

```python
"""
calculate_intrinsics.py
Complete example: build a camera matrix and verify it with a projection round-trip.
"""
import math
import numpy as np
from sensor_transposition.camera_intrinsics import (
    focal_length_from_fov,
    focal_length_from_sensor,
    fov_from_focal_length,
    camera_matrix,
    project_point,
    unproject_pixel,
)

# ── 1. Camera specifications ──────────────────────────────────────────────────
WIDTH, HEIGHT = 1920, 1080   # image resolution in pixels

# Choose ONE of the two methods below:

# Method A – from FOV
HFOV_DEG = 90.0
VFOV_DEG = 59.0
fx = focal_length_from_fov(WIDTH,  HFOV_DEG)
fy = focal_length_from_fov(HEIGHT, VFOV_DEG)

# Method B – from physical sensor geometry (comment out Method A to use this)
# fx = focal_length_from_sensor(WIDTH,  6.40, 4.35)
# fy = focal_length_from_sensor(HEIGHT, 4.80, 4.35)

# ── 2. Principal point ────────────────────────────────────────────────────────
cx, cy = WIDTH / 2.0, HEIGHT / 2.0

# ── 3. Camera matrix ──────────────────────────────────────────────────────────
K = camera_matrix(fx, fy, cx, cy)
print("Camera matrix K:")
print(K)

# ── 4. Sanity check: recover FOV from focal lengths ───────────────────────────
hfov_check = fov_from_focal_length(fx, WIDTH)
vfov_check = fov_from_focal_length(fy, HEIGHT)
print(f"\nRecovered HFOV: {hfov_check:.2f}°  (expected {HFOV_DEG}°)")
print(f"Recovered VFOV: {vfov_check:.2f}°  (expected {VFOV_DEG}°)")

# ── 5. Projection round-trip ─────────────────────────────────────────────────
test_point = np.array([1.0, -0.5, 3.0])
u, v = project_point(K, test_point)
print(f"\n3-D point {test_point} → pixel ({u:.2f}, {v:.2f})")

recovered = unproject_pixel(K, (u, v), depth=test_point[2])
print(f"Unprojected back to 3-D: {recovered}")
print(f"Round-trip error: {np.linalg.norm(recovered - test_point):.2e}")
```

---

## Troubleshooting

### `ValueError: fov_deg must be in (0, 180)`

The FOV value you provided is outside the valid range.  Double-check the
camera spec — FOV is always strictly between 0° and 180°.

### `ValueError: image_size must be positive`

Make sure you are passing the pixel count (e.g. `1920`), not metres or mm.

### `fx` and `fy` differ by more than a few percent

- If using Method A (FOV): check that the horizontal and vertical FOV values
  correspond to the correct image axes.  Some manufacturers quote diagonal FOV
  — use the conversion snippet in [Step 1](#step-1--gather-your-camera-specifications).
- If using Method B (sensor size): verify that the sensor width/height values
  match the image aspect ratio (`sensor_w / sensor_h ≈ width / height`).

### Projected points are systematically offset from expected positions

The principal point may not be at the exact image centre.  Run a full
checkerboard calibration (e.g. with `opencv-python`'s `calibrateCamera`) to
obtain a precise `(cx, cy)` as well as distortion coefficients.

### `ValueError: Point is behind the camera`

`project_point` requires `Z > 0` (the point must be in front of the camera).
Verify that your 3-D point coordinates are expressed in the camera frame, not
in a world or body frame.
