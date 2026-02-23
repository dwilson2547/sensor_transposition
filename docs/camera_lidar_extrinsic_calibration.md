# Camera–LiDAR Extrinsic Calibration Guide

A step-by-step guide to computing the 6-DoF rigid-body transform between a
LiDAR and a camera using a **planar calibration target** (checkerboard or
ArUco board) and the `sensor_transposition` library.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step 1 – Prepare the Calibration Target](#step-1--prepare-the-calibration-target)
4. [Step 2 – Collect Calibration Data](#step-2--collect-calibration-data)
5. [Step 3 – Extract the Board Plane from LiDAR](#step-3--extract-the-board-plane-from-lidar)
6. [Step 4 – Extract the Board Pose from Camera](#step-4--extract-the-board-pose-from-camera)
7. [Step 5 – Solve for the Extrinsic Transform](#step-5--solve-for-the-extrinsic-transform)
8. [Step 6 – Save the Result to SensorCollection](#step-6--save-the-result-to-sensorcollection)
9. [Complete Example Script](#complete-example-script)
10. [Tips and Best Practices](#tips-and-best-practices)
11. [Troubleshooting](#troubleshooting)

---

## Overview

The extrinsic transform **T** (LiDAR → camera) encodes how a 3-D point
measured in the LiDAR frame maps to the camera frame:

```
p_camera = T @ p_lidar        (homogeneous coordinates)
```

This library uses a **plane-correspondence approach** (Geiger *et al.*,
2012): for each pose of a planar calibration board, both sensors observe the
same plane.  Collecting N ≥ 3 such observations provides enough constraints
to solve for **T** in closed form.

---

## Prerequisites

**Hardware**

* A rigid planar calibration target — a printed checkerboard (recommended: 8×6
  inner corners, 0.1 m square side length) or an ArUco board large enough to
  produce ≥ 20 LiDAR returns.
* The camera and LiDAR mounted together in a **fixed, rigid rig** — no motion
  between them is allowed during calibration.

**Software**

```bash
pip install sensor_transposition
pip install opencv-python          # needed for camera corner detection
```

**Camera intrinsics**

You need the camera's intrinsic matrix **K** and distortion coefficients
before running extrinsic calibration.  See
[`camera_intrinsics_guide.md`](camera_intrinsics_guide.md) for details.

---

## Step 1 – Prepare the Calibration Target

Print a checkerboard at a known physical size.  Record:

| Parameter | Example |
|-----------|---------|
| `pattern_size` | `(8, 6)` — *inner corner* count (cols, rows) |
| `square_size_m` | `0.10` — physical side length in metres |

The board must be large enough that the LiDAR scans it with a good density
of returns (≥ 20 points) across its entire surface.

---

## Step 2 – Collect Calibration Data

Mount the calibration board on a stable surface (e.g. a wall or tripod) and
capture **10–20 different board poses** by moving the board or the sensor rig
between captures.  For each pose, record a synchronised pair:

* One camera image (JPEG/PNG or numpy array).
* One LiDAR frame (numpy `(N, 3)` array of `[x, y, z]` points in the LiDAR
  frame — see the `sensor_transposition.lidar` parsers).

**Diversity matters**: vary the board's distance, tilt, and orientation across
poses to make the calibration problem well-conditioned.  Include poses where the
board is tilted > 30° along different axes.

---

## Step 3 – Extract the Board Plane from LiDAR

For each capture, crop the LiDAR frame to a bounding box that contains only
the calibration board, then call `ransac_plane` to robustly fit the board
plane.

```python
import numpy as np
from sensor_transposition.calibration import ransac_plane

# lidar_frame: (N, 3) array of all points in one LiDAR scan

# --- Crop to the board region (adjust bounds for your setup) ---
mask = (
    (lidar_frame[:, 0] > 1.0) & (lidar_frame[:, 0] < 4.0) &   # forward range
    (lidar_frame[:, 1] > -0.8) & (lidar_frame[:, 1] < 0.8) &  # lateral range
    (lidar_frame[:, 2] > -0.5) & (lidar_frame[:, 2] < 1.0)    # height range
)
board_pts = lidar_frame[mask]

# --- Fit the board plane using RANSAC ---
normal_lidar, distance_lidar, inlier_mask = ransac_plane(
    board_pts,
    distance_threshold=0.02,   # 2 cm — adjust to your LiDAR noise level
    min_inliers=20,
    rng=42,                    # set a seed for reproducibility
)

print(f"Board normal (LiDAR):  {normal_lidar}")
print(f"Plane distance (LiDAR): {distance_lidar:.4f} m")
print(f"Inlier count: {inlier_mask.sum()} / {len(board_pts)}")
```

> **Sign convention**: `ransac_plane` (and `fit_plane`) always return a normal
> with `distance ≥ 0`.  To ensure the normal points **toward the LiDAR**
> (required for `calibrate_lidar_camera`) check whether the returned normal
> has a positive dot product with the vector from the board centroid to the
> LiDAR origin.  If not, flip both the normal and the distance sign:
>
> ```python
> lidar_origin = np.array([0.0, 0.0, 0.0])
> centroid = board_pts[inlier_mask].mean(axis=0)
> if np.dot(normal_lidar, lidar_origin - centroid) < 0:
>     normal_lidar  = -normal_lidar
>     distance_lidar = -distance_lidar
> ```

---

## Step 4 – Extract the Board Pose from Camera

Use OpenCV to detect checkerboard corners in the image and solve the
Perspective-n-Point (PnP) problem to get the board pose in the camera frame.
From the pose, extract the board's normal vector and plane distance.

```python
import cv2
import numpy as np

# Camera intrinsics (from your prior intrinsic calibration)
K = np.array([[fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1]], dtype=float)
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=float)

pattern_size = (8, 6)     # inner corners (cols, rows)
square_size_m = 0.10      # metres

# 3-D object points: board lies in the Z=0 plane
obj_pts = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
obj_pts[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
obj_pts *= square_size_m

# Detect corners
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
found, corners = cv2.findChessboardCorners(gray, pattern_size)
if not found:
    raise RuntimeError("Checkerboard not found in image — check the image and pattern_size.")

corners = cv2.cornerSubPix(
    gray, corners, (11, 11), (-1, -1),
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
)

# Solve PnP: board-to-camera rotation and translation
ok, rvec, tvec = cv2.solvePnP(obj_pts, corners, K, dist_coeffs)
if not ok:
    raise RuntimeError("solvePnP failed.")

R_board, _ = cv2.Rodrigues(rvec)      # (3, 3) rotation matrix
tvec = tvec.ravel()                   # (3,)

# Board normal in camera frame = third column of R_board
# (the board's local Z-axis, which is perpendicular to the board surface)
normal_camera = R_board[:, 2]

# Plane distance: n_c · (board origin in camera frame) = n_c · tvec
distance_camera = float(np.dot(normal_camera, tvec))

# Ensure the normal points toward the camera (positive Z should give d > 0
# when the board is in front of the camera).
if distance_camera < 0:
    normal_camera  = -normal_camera
    distance_camera = -distance_camera

print(f"Board normal (camera):   {normal_camera}")
print(f"Plane distance (camera): {distance_camera:.4f} m")
```

---

## Step 5 – Solve for the Extrinsic Transform

Collect all N observations into arrays and call `calibrate_lidar_camera`.

```python
from sensor_transposition.calibration import calibrate_lidar_camera

# Accumulated over N captures:
lidar_normals    = np.array([nl_1, nl_2, ..., nl_N])   # (N, 3)
lidar_distances  = np.array([dl_1, dl_2, ..., dl_N])   # (N,)
camera_normals   = np.array([nc_1, nc_2, ..., nc_N])   # (N, 3)
camera_distances = np.array([dc_1, dc_2, ..., dc_N])   # (N,)

T_lidar_to_camera = calibrate_lidar_camera(
    lidar_normals, lidar_distances,
    camera_normals, camera_distances,
)

print("LiDAR → Camera transform:")
print(T_lidar_to_camera)
```

The result is a 4×4 homogeneous transform.  Verify it by projecting LiDAR
points onto the camera image with
`sensor_transposition.lidar_camera.project_lidar_to_image`.

---

## Step 6 – Save the Result to SensorCollection

Convert the 4×4 transform to a quaternion + translation and store it in a
`SensorCollection` YAML file.

```python
from scipy.spatial.transform import Rotation
from sensor_transposition import SensorCollection, Sensor, CameraIntrinsics

# Extract rotation and translation from T
R = T_lidar_to_camera[:3, :3]
t = T_lidar_to_camera[:3, 3]

# Convert rotation matrix to quaternion [w, x, y, z]
rot = Rotation.from_matrix(R)
quat_xyzw = rot.as_quat()               # scipy uses [x, y, z, w]
quat_wxyz = [quat_xyzw[3], *quat_xyzw[:3]]

print(f"translation: {t.tolist()}")
print(f"quaternion [w,x,y,z]: {quat_wxyz}")

# Build or update a SensorCollection
collection = SensorCollection.from_yaml("sensors.yaml")

# Update the LiDAR sensor's extrinsics so that they express
# the LiDAR-to-camera transform (relative to the camera sensor).
# By convention, extrinsics are stored as sensor-to-ego transforms.
# Adjust as needed for your coordinate-frame convention.
lidar_sensor = collection.get_sensor("front_lidar")
lidar_sensor.translation = t.tolist()
lidar_sensor.rotation    = quat_wxyz

collection.to_yaml("sensors_calibrated.yaml")
print("Saved calibrated sensor collection to sensors_calibrated.yaml")
```

---

## Complete Example Script

```python
"""
calibrate_lidar_camera.py

End-to-end Camera–LiDAR extrinsic calibration from a set of captures.
Each capture is a (camera_image, lidar_frame) pair taken with a
checkerboard visible to both sensors.
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from sensor_transposition.calibration import ransac_plane, calibrate_lidar_camera
from sensor_transposition.lidar_camera import project_lidar_to_image

# ── Camera intrinsics ─────────────────────────────────────────────────────────
K = np.array([[800.0, 0.0, 640.0],
              [  0.0, 800.0, 360.0],
              [  0.0,   0.0,   1.0]], dtype=float)
dist_coeffs = np.zeros(5, dtype=float)

# ── Calibration target ────────────────────────────────────────────────────────
PATTERN_SIZE   = (8, 6)      # (cols, rows) of inner corners
SQUARE_SIZE_M  = 0.10        # physical side length of each square in metres

obj_pts = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), dtype=np.float32)
obj_pts[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
obj_pts *= SQUARE_SIZE_M

# ── Accumulate observations ───────────────────────────────────────────────────
lidar_normals_list    = []
lidar_distances_list  = []
camera_normals_list   = []
camera_distances_list = []

# Replace this list with your actual (image, lidar_frame) pairs.
captures = []   # [(image_np_array, lidar_np_array), ...]

for image, lidar_frame in captures:
    # --- Camera side ---
    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE)
    if not found:
        print("Board not found in image; skipping this capture.")
        continue
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    ok, rvec, tvec = cv2.solvePnP(obj_pts, corners, K, dist_coeffs)
    if not ok:
        continue
    R_board, _ = cv2.Rodrigues(rvec)
    tvec = tvec.ravel()
    nc = R_board[:, 2]
    dc = float(np.dot(nc, tvec))
    if dc < 0:
        nc, dc = -nc, -dc

    # --- LiDAR side ---
    # Crop to the board region (adjust bounds for your setup).
    mask = (
        (lidar_frame[:, 0] > 1.0) & (lidar_frame[:, 0] < 4.0)
        & (lidar_frame[:, 1] > -0.8) & (lidar_frame[:, 1] < 0.8)
        & (lidar_frame[:, 2] > -0.5) & (lidar_frame[:, 2] < 1.0)
    )
    board_pts = lidar_frame[mask]
    if len(board_pts) < 20:
        print("Too few LiDAR points on board; skipping this capture.")
        continue
    nl, dl, inlier_mask = ransac_plane(board_pts, distance_threshold=0.02, min_inliers=10)
    centroid = board_pts[inlier_mask].mean(axis=0)
    if np.dot(nl, -centroid) < 0:      # ensure normal points toward LiDAR
        nl, dl = -nl, -dl

    lidar_normals_list.append(nl)
    lidar_distances_list.append(dl)
    camera_normals_list.append(nc)
    camera_distances_list.append(dc)

if len(lidar_normals_list) < 3:
    raise RuntimeError("Need at least 3 valid observations; collect more data.")

# ── Solve ─────────────────────────────────────────────────────────────────────
T = calibrate_lidar_camera(
    np.array(lidar_normals_list),
    np.array(lidar_distances_list),
    np.array(camera_normals_list),
    np.array(camera_distances_list),
)
print("Estimated LiDAR → Camera transform:")
print(T)

# ── Verify ────────────────────────────────────────────────────────────────────
# Project the first LiDAR scan onto the camera image and inspect the overlay.
if captures:
    sample_image, sample_lidar = captures[0]
    H, W = sample_image.shape[:2]
    pixels, valid = project_lidar_to_image(sample_lidar, T, K, W, H)
    print(f"Projected {valid.sum()} / {len(sample_lidar)} points onto image.")
```

---

## Tips and Best Practices

* **Number of captures**: 10–20 diverse board poses give a well-conditioned
  system.  Fewer than 5 poses often lead to poor translation estimates.

* **Board size**: The board must cover at least 5–10 LiDAR scan lines.  For a
  VLP-16 at 5 m range, a 0.6 m × 0.8 m board is a practical minimum.

* **Tilt diversity**: Include poses where the board is tilted 30–60° in pitch
  *and* yaw relative to the LiDAR.  Flat-on poses constrain only one normal
  direction.

* **LiDAR crop**: Crop tightly to the board to avoid including ground, walls,
  or other surfaces in the RANSAC input.  If the cropped region contains fewer
  than 20 points, move the board closer or choose a coarser crop.

* **Sign consistency**: Both the LiDAR normal and the camera normal for the
  same capture must point to the same side of the board (toward their
  respective sensor).  A sign mismatch produces a large residual — check if
  `camera_distance_i ≈ lidar_distance_i + camera_normal_i · t` for your
  final calibration.

* **Verification**: After calibration, project several LiDAR frames onto the
  camera image and visually inspect the alignment.  Edges of the calibration
  board in the image should line up with the corresponding point-cloud boundary.

---

## Troubleshooting

### `ValueError: RANSAC failed`

* The board region crop may include too many non-board points.  Tighten the
  bounding-box filter.
* Try increasing `distance_threshold` (e.g. `0.05`) or `max_iterations`
  (e.g. `200`).
* Ensure `min_inliers` is not larger than the expected number of LiDAR points
  on the board.

### `ValueError: At least 3 plane correspondences are required`

You have fewer than 3 valid captures.  Re-collect data; ensure the board is
visible in both modalities for each pose.

### Projected LiDAR points are clearly offset from the image edges

* Verify the sign convention: normals for all observations must point toward
  their respective sensors.
* Check that the camera intrinsics **K** and distortion coefficients are
  accurate — extrinsic calibration errors can be caused by poor intrinsics.
* Collect more diverse board poses (especially tilted at different angles).

### `solvePnP` returns `ok=False`

* Ensure `findChessboardCorners` succeeded (`found=True`) before calling
  `solvePnP`.
* Check that `obj_pts` and `corners` have the same number of rows.
* Verify the `pattern_size` matches the printed board exactly (inner corners,
  not total squares).
