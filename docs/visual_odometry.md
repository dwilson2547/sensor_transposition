# Visual Odometry

`visual_odometry.py` provides the core geometry primitives for a monocular
visual odometry front-end.  All functions operate on plain NumPy arrays and
require no additional dependencies beyond `numpy` and `scipy`.

---

## Overview

A monocular visual odometry pipeline estimates the **incremental motion** of a
camera by comparing successive images.  The typical sequence is:

1. **Detect and match** keypoints between frame *k* and frame *k+1* using any
   descriptor matcher (ORB, SIFT, AKAZE, …).
2. **Estimate the essential matrix** from the matched pixel pairs using
   [`estimate_essential_matrix`](#estimate_essential_matrix).
3. **Recover the relative pose** (R, t) from the essential matrix using
   [`recover_pose_from_essential`](#recover_pose_from_essential).
4. When 3-D map points are available (after triangulation or from a depth
   sensor), refine or re-localise the pose with
   [`solve_pnp`](#solve_pnp).

---

## Functions

### `estimate_essential_matrix`

```python
from sensor_transposition.visual_odometry import estimate_essential_matrix

result = estimate_essential_matrix(
    points1,          # (N, 2) pixel coords in image 1
    points2,          # (N, 2) pixel coords in image 2
    K,                # 3×3 camera intrinsic matrix
    inlier_threshold=1.0,        # Sampson error threshold (normalised units)
    max_ransac_iterations=1000,  # upper bound on RANSAC iterations
    confidence=0.999,            # desired RANSAC confidence
    rng=None,                    # optional numpy.random.Generator
)

E      = result.essential_matrix  # 3×3 essential matrix
mask   = result.inlier_mask       # boolean (N,) array
ninl   = result.num_inliers       # number of inliers
```

Computes the **essential matrix** *E* satisfying `x2.T @ E @ x1 ≈ 0` for
normalised image coordinates, using the **normalised 8-point algorithm** inside
a **RANSAC** loop.

**Algorithm:**
1. Randomly sample 8 correspondences.
2. Normalise pixel coordinates by *K⁻¹* (Hartley normalisation for numerical
   stability).
3. Solve the 9-element epipolar linear system via SVD.
4. Enforce the rank-2 essential-matrix constraint.
5. Count inliers using the **Sampson error** as a reprojection proxy.
6. Adaptive RANSAC stopping criterion.
7. Final non-linear re-estimation from all inliers.

**Requirements:** `N ≥ 8` point correspondences.

---

### `recover_pose_from_essential`

```python
from sensor_transposition.visual_odometry import recover_pose_from_essential

R, t = recover_pose_from_essential(
    E,        # 3×3 essential matrix (from estimate_essential_matrix)
    points1,  # (N, 2) inlier pixel coords in image 1
    points2,  # (N, 2) inlier pixel coords in image 2
    K,        # 3×3 camera intrinsic matrix
)
# R : 3×3 rotation matrix  (camera 1 → camera 2 orientation)
# t : (3,) unit translation vector  (scale ambiguous for monocular)
```

Decomposes *E* via SVD into the **four (R, t) candidate solutions** and
selects the unique physically valid one by triangulating a subset of points and
applying the **cheirality test** (both cameras must see the point in front).

**Note:** The translation `t` is recovered **up to scale** — the baseline
length is unknown for a monocular camera.  Use stereo, depth sensors, or
external scale cues to recover metric scale.

**Requirements:** `N ≥ 5` inlier correspondences.

---

### `solve_pnp`

```python
from sensor_transposition.visual_odometry import solve_pnp

result = solve_pnp(
    points_3d,           # (N, 3) world-frame 3-D points
    points_2d,           # (N, 2) observed pixel coordinates
    K,                   # 3×3 camera intrinsic matrix
    inlier_threshold=2.0,        # reprojection error threshold in pixels
    max_ransac_iterations=500,
    confidence=0.999,
    rng=None,
)

R    = result.rotation     # 3×3 rotation  (world → camera frame)
t    = result.translation  # (3,) translation in metres
mask = result.inlier_mask  # boolean (N,) inlier mask
ok   = result.success      # True if a valid pose was found
```

Solves the **Perspective-n-Point (PnP)** problem: given *N* 3-D world points
and their 2-D pixel observations, estimate the 6-DOF camera pose that minimises
reprojection error.

**Algorithm:**
1. Sample 6 correspondences.
2. Solve the 12-DOF linear projection system via **Direct Linear Transform
   (DLT)**.
3. Extract *R* and *t* from the solution via polar decomposition / SVD.
4. Count inliers by reprojection error.
5. Adaptive RANSAC stopping.
6. Final re-estimation from all inliers.

**Requirements:** `N ≥ 6` point correspondences; points should be
non-coplanar for a full perspective solution.

---

## Data Classes

### `EssentialMatrixResult`

| Field | Type | Description |
|-------|------|-------------|
| `essential_matrix` | `(3, 3) ndarray` | The estimated essential matrix *E* |
| `inlier_mask` | `(N,) bool ndarray` | True for inlier correspondences |
| `num_inliers` | `int` | Number of inliers |

### `PnPResult`

| Field | Type | Description |
|-------|------|-------------|
| `rotation` | `(3, 3) ndarray` | Rotation matrix (world → camera) |
| `translation` | `(3,) ndarray` | Translation vector in metres |
| `inlier_mask` | `(N,) bool ndarray` | True for inlier correspondences |
| `num_inliers` | `int` | Number of inliers |
| `success` | `bool` | True if a valid pose was found |

---

## Full Example

```python
import numpy as np
from sensor_transposition.camera_intrinsics import camera_matrix
from sensor_transposition.visual_odometry import (
    estimate_essential_matrix,
    recover_pose_from_essential,
    solve_pnp,
)

# Camera intrinsics
K = camera_matrix(fx=800.0, fy=800.0, cx=640.0, cy=360.0)

# --- Step 1: matched pixel pairs from frame k and frame k+1 ----
# (Replace with real matches from your keypoint detector/matcher)
pts1 = np.array(...)  # (N, 2) pixels in frame k
pts2 = np.array(...)  # (N, 2) pixels in frame k+1

# --- Step 2: estimate essential matrix ---
em_result = estimate_essential_matrix(
    pts1, pts2, K,
    inlier_threshold=1.0,
    rng=np.random.default_rng(42),
)
print(f"Inliers: {em_result.num_inliers} / {len(pts1)}")

# --- Step 3: recover relative pose ---
inliers1 = pts1[em_result.inlier_mask]
inliers2 = pts2[em_result.inlier_mask]

R, t = recover_pose_from_essential(
    em_result.essential_matrix, inliers1, inliers2, K
)
print("Rotation:\n", R)
print("Translation (unit):", t)  # scale unknown for monocular

# --- Step 4: PnP localisation (when 3-D map points are available) ---
map_pts_3d = np.array(...)   # (M, 3) world-frame 3-D points
obs_pts_2d = np.array(...)   # (M, 2) corresponding pixel observations

pnp = solve_pnp(map_pts_3d, obs_pts_2d, K, inlier_threshold=2.0)
if pnp.success:
    print("Camera rotation:\n", pnp.rotation)
    print("Camera position (world frame):", -pnp.rotation.T @ pnp.translation)
```

---

## Notes

* **No OpenCV required.** The implementation uses only `numpy` and `scipy`
  (already required by `sensor_transposition`).
* **Keypoint detection and matching** are *not* included — supply matched pixel
  coordinates from any detector (ORB, SIFT, SuperPoint, …).
* **Monocular scale ambiguity:** `recover_pose_from_essential` returns a
  unit-norm translation.  Metric scale requires stereo, a depth sensor, or an
  IMU.
* **Planar scenes** cause the PnP DLT to become degenerate.  For predominantly
  planar environments consider a homography-based estimator instead.
