# Ground Plane Identification in Sensor Datasets

A practical guide to identifying and segmenting the ground plane in
point-cloud data captured by **LiDAR** and **radar** sensors, using the
`sensor_transposition` library together with standard NumPy / SciPy
operations.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Coordinate System Conventions](#coordinate-system-conventions)
4. [Method 1 – Height Threshold](#method-1--height-threshold)
   - [Single Sensor (Sensor Frame)](#single-sensor-sensor-frame)
   - [After Transforming to the Ego Frame](#after-transforming-to-the-ego-frame)
5. [Method 2 – RANSAC Plane Fitting](#method-2--ransac-plane-fitting)
6. [Method 3 – Normal-Based Filtering](#method-3--normal-based-filtering)
7. [Working With Radar Data](#working-with-radar-data)
8. [Combining Methods](#combining-methods)
9. [Verifying Your Ground Segmentation](#verifying-your-ground-segmentation)
10. [Troubleshooting](#troubleshooting)

---

## Overview

Many autonomous-vehicle and robotics pipelines need to separate **ground
points** from **non-ground points** (obstacles, vegetation, buildings, etc.)
before downstream processing such as object detection, clustering, or
mapping.

This guide covers three progressively more robust approaches:

| Method | Complexity | When to use |
|--------|-----------|-------------|
| **Height threshold** | Low | Flat terrain, calibrated sensor height is known |
| **RANSAC plane fitting** | Medium | Uneven terrain or unknown sensor height |
| **Normal-based filtering** | Medium–High | Hilly environments with varying slope |

All examples assume you have already captured and loaded a point cloud
using one of the library's parsers (`VelodyneParser`, `OusterParser`,
`LivoxParser`, or `RadarParser`).

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

The examples below also use **NumPy** (installed automatically as a
dependency) and optionally **SciPy** (for RANSAC and normal estimation).

---

## Coordinate System Conventions

Ground plane identification relies heavily on knowing which axis points
**up**.  The library's supported coordinate systems define the vertical
axis as follows:

| Convention | Up axis | Ground plane equation |
|------------|---------|----------------------|
| **FLU** (Forward, Left, Up) | +Z | `z ≈ 0` at ground level |
| **FRD** (Forward, Right, Down) | −Z | `z ≈ 0` at sensor level; ground is at positive Z |
| **ENU** (East, North, Up) | +Z | `z ≈ 0` at the reference altitude |
| **NED** (North, East, Down) | −Z | `z ≈ 0` at the reference altitude; ground is at positive Z |
| **RDF** (Right, Down, Forward) | −Y | `y ≈ 0` at sensor level; ground is at positive Y |

The **ego frame** used by this library is **FLU** with the origin at the
centre of the rear axle projected onto the ground plane.  In this frame
the ground surface satisfies **z ≈ 0** and points above ground have
**z > 0**.

> **Tip:** Always transform point clouds into the ego frame before
> applying a ground plane method.  This makes thresholds and plane
> parameters consistent across sensors with different mounting positions
> and orientations.

---

## Method 1 – Height Threshold

The simplest approach: after transforming the point cloud into the ego
frame (where the ground is at z ≈ 0), classify any point with
z < *threshold* as ground.

### Single Sensor (Sensor Frame)

If the sensor's mounting height above the ground is known, you can
threshold directly in the sensor frame.  For an FLU sensor mounted at
height *h*, ground points satisfy `z ≈ −h`:

```python
import numpy as np
from sensor_transposition.lidar.velodyne import VelodyneParser

parser = VelodyneParser("frame_000000.bin")
xyz = parser.xyz()  # (N, 3) in sensor frame (FLU)

sensor_height = 1.91  # metres above ground (from extrinsics)
threshold = 0.3       # tolerance in metres

ground_mask = xyz[:, 2] < (-sensor_height + threshold)
ground_points = xyz[ground_mask]
non_ground_points = xyz[~ground_mask]

print(f"Ground points    : {ground_mask.sum()}")
print(f"Non-ground points: {(~ground_mask).sum()}")
```

### After Transforming to the Ego Frame

Using `SensorCollection` to transform the cloud into the ego frame
first is more reliable, because the ego frame origin is defined at
ground level:

```python
import numpy as np
from sensor_transposition import SensorCollection
from sensor_transposition.lidar.velodyne import VelodyneParser

col = SensorCollection.from_yaml("examples/sensor_collection.yaml")
parser = VelodyneParser("frame_000000.bin")
xyz_sensor = parser.xyz()  # (N, 3) in the sensor's native frame

# Transform to the ego frame
T = col.get_sensor("front_lidar").get_transform()
xyz_ego = T.apply_to_points(xyz_sensor)

# In the ego frame, ground is at z ≈ 0
height_threshold = 0.3  # metres above ground
ground_mask = xyz_ego[:, 2] < height_threshold

ground_points = xyz_ego[ground_mask]
non_ground_points = xyz_ego[~ground_mask]

print(f"Ground points    : {ground_mask.sum()}")
print(f"Non-ground points: {(~ground_mask).sum()}")
```

> **When to use:** Flat, paved environments (parking lots, highways)
> where the ground surface is approximately level.  On hilly terrain or
> rough off-road surfaces, a fixed height threshold will misclassify
> elevated ground as obstacles and low obstacles as ground.

---

## Method 2 – RANSAC Plane Fitting

**RANSAC** (Random Sample Consensus) fits a plane model to the point
cloud without being distorted by outliers (walls, cars, trees, etc.).
It works well even when the true ground surface is tilted or the sensor
height is not precisely known.

A plane in 3-D is described by `ax + by + cz + d = 0`, where
`(a, b, c)` is the unit normal.  For a ground plane in the FLU ego
frame, the normal should point approximately upward: `(a, b, c) ≈ (0, 0, 1)`.

```python
import numpy as np
from sensor_transposition.lidar.velodyne import VelodyneParser

def fit_ground_plane_ransac(
    points: np.ndarray,
    distance_threshold: float = 0.2,
    max_iterations: int = 1000,
    normal_threshold: float = 0.9,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a ground plane using RANSAC.

    Parameters
    ----------
    points : (N, 3) array
        Point cloud in a frame where the ground normal is approximately +Z.
    distance_threshold : float
        Maximum distance from the plane for a point to be considered an inlier.
    max_iterations : int
        Number of RANSAC iterations.
    normal_threshold : float
        Minimum dot product between the candidate plane normal and the +Z
        axis.  Planes that are too tilted are rejected.
    rng : numpy random Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    ground_mask : (N,) bool array
        True for points classified as ground.
    plane : (4,) array
        Plane coefficients [a, b, c, d] where ax + by + cz + d = 0.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(points)
    best_inliers = np.zeros(n, dtype=bool)
    best_plane = np.zeros(4)

    for _ in range(max_iterations):
        # 1. Sample three random points
        idx = rng.choice(n, size=3, replace=False)
        p1, p2, p3 = points[idx]

        # 2. Compute the plane normal
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm

        # Ensure the normal points upward (+Z)
        if normal[2] < 0:
            normal = -normal

        # 3. Reject planes whose normal deviates too far from +Z
        if normal[2] < normal_threshold:
            continue

        # 4. Compute signed distances from all points to the plane
        d = -np.dot(normal, p1)
        distances = np.abs(points @ normal + d)

        # 5. Count inliers
        inliers = distances < distance_threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_plane = np.array([normal[0], normal[1], normal[2], d])

    return best_inliers, best_plane


# ── Example usage ─────────────────────────────────────────────────────────
parser = VelodyneParser("frame_000000.bin")
xyz = parser.xyz()

ground_mask, plane = fit_ground_plane_ransac(xyz, distance_threshold=0.2)

a, b, c, d = plane
print(f"Plane: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
print(f"Ground points    : {ground_mask.sum()}")
print(f"Non-ground points: {(~ground_mask).sum()}")
```

### Tuning Parameters

| Parameter | Effect of increasing | Typical range |
|-----------|---------------------|---------------|
| `distance_threshold` | More points classified as ground | 0.1 – 0.4 m |
| `max_iterations` | More robust fit, slower | 200 – 2000 |
| `normal_threshold` | Rejects more tilted planes (stricter) | 0.85 – 0.95 |

> **When to use:** General-purpose; works on flat and moderately sloped
> terrain.  For highly uneven surfaces (e.g. steep hills, ramps), consider
> splitting the point cloud into spatial bins and fitting a plane per bin.

---

## Method 3 – Normal-Based Filtering

Instead of fitting a single global plane, compute a **local surface
normal** at each point and classify it as ground if the normal is
approximately vertical.  This handles undulating terrain better than a
single-plane RANSAC fit.

Computing per-point normals requires a local neighbourhood query.  The
example below uses SciPy's `cKDTree` for neighbour lookups:

```python
import numpy as np
from scipy.spatial import cKDTree
from sensor_transposition.lidar.velodyne import VelodyneParser


def estimate_normals(points: np.ndarray, k: int = 20) -> np.ndarray:
    """Estimate per-point normals using PCA on k-nearest neighbours.

    Parameters
    ----------
    points : (N, 3) array
    k : int
        Number of nearest neighbours to use for each point.

    Returns
    -------
    normals : (N, 3) array
        Unit normal vectors (oriented toward +Z).
    """
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)
    normals = np.empty_like(points)

    for i, neighbours in enumerate(idx):
        local = points[neighbours]
        cov = np.cov(local, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Smallest eigenvalue corresponds to the surface normal
        normal = eigvecs[:, 0]
        # Orient toward +Z
        if normal[2] < 0:
            normal = -normal
        normals[i] = normal

    return normals


# ── Example usage ─────────────────────────────────────────────────────────
parser = VelodyneParser("frame_000000.bin")
xyz = parser.xyz()

normals = estimate_normals(xyz, k=20)

# Ground points have normals approximately equal to [0, 0, 1]
verticality = normals[:, 2]  # dot product with +Z
ground_mask = verticality > 0.85  # threshold on cos(angle)

ground_points = xyz[ground_mask]
non_ground_points = xyz[~ground_mask]

print(f"Ground points    : {ground_mask.sum()}")
print(f"Non-ground points: {(~ground_mask).sum()}")
```

> **When to use:** Hilly or uneven terrain where the ground surface
> cannot be described by a single plane.  The trade-off is higher
> computational cost due to the per-point neighbour search.

---

## Working With Radar Data

Radar sensors like the Continental ARS 408-21 output 2-D detections
(range and azimuth only; elevation is always zero).  Because there is no
height information, classical ground plane methods do not apply directly.

However, you can still separate ground-level returns from elevated
objects using context clues:

```python
import numpy as np
from sensor_transposition.radar.radar import RadarParser

parser = RadarParser("radar_frame.bin")
detections = parser.read()

# Detections with very low RCS at short range are often ground clutter
rcs = detections["snr"]
rng = detections["range"]

ground_clutter_mask = (rcs < -10.0) & (rng < 30.0)

non_clutter = detections[~ground_clutter_mask]
print(f"Detections after clutter removal: {len(non_clutter)}")
```

For radars that provide elevation angle (e.g. 4-D imaging radars), the
RANSAC and height-threshold methods described above apply directly to
the `(x, y, z)` output of `RadarParser.xyz()`.

---

## Combining Methods

For best results, combine a coarse height threshold with RANSAC
refinement:

```python
import numpy as np
from sensor_transposition import SensorCollection
from sensor_transposition.lidar.velodyne import VelodyneParser

col = SensorCollection.from_yaml("examples/sensor_collection.yaml")
parser = VelodyneParser("frame_000000.bin")
xyz_sensor = parser.xyz()

# 1. Transform to ego frame
T = col.get_sensor("front_lidar").get_transform()
xyz_ego = T.apply_to_points(xyz_sensor)

# 2. Coarse filter: keep only low points as RANSAC candidates
low_mask = xyz_ego[:, 2] < 0.5  # rough upper bound for ground
candidates = xyz_ego[low_mask]

# 3. RANSAC on the candidate subset
ground_of_candidates, plane = fit_ground_plane_ransac(
    candidates, distance_threshold=0.15
)

# 4. Map back to the full cloud
ground_mask = np.zeros(len(xyz_ego), dtype=bool)
ground_indices = np.where(low_mask)[0]
ground_mask[ground_indices[ground_of_candidates]] = True

print(f"Total points     : {len(xyz_ego)}")
print(f"Ground points    : {ground_mask.sum()}")
print(f"Non-ground points: {(~ground_mask).sum()}")
```

Pre-filtering with a height threshold reduces the number of points that
RANSAC must process, which makes the plane fit both faster and more
robust.

---

## Verifying Your Ground Segmentation

After segmentation, a quick sanity check helps confirm the results are
reasonable:

```python
import numpy as np

# Assuming ground_points and non_ground_points are (M, 3) and (K, 3) arrays
print(f"Ground points    : {len(ground_points)}")
print(f"Non-ground points: {len(non_ground_points)}")

# Ground fraction – typically 30–60 % for an outdoor LiDAR scan
total = len(ground_points) + len(non_ground_points)
print(f"Ground fraction  : {len(ground_points) / total:.1%}")

# Height statistics of ground points (in ego frame)
z_ground = ground_points[:, 2]
print(f"Ground Z mean    : {z_ground.mean():.3f} m")
print(f"Ground Z std     : {z_ground.std():.3f} m")
print(f"Ground Z range   : [{z_ground.min():.3f}, {z_ground.max():.3f}] m")

# Sanity checks
assert len(ground_points) > 0, "No ground points found – check thresholds"
assert z_ground.std() < 0.5, "Ground points have high Z variance – segmentation may be noisy"
```

**Expected values for a typical outdoor LiDAR scan:**

| Metric | Typical value |
|--------|---------------|
| Ground fraction | 30 – 60 % |
| Ground Z mean (ego frame) | −0.1 m to 0.1 m |
| Ground Z standard deviation | < 0.15 m on flat terrain |

---

## Troubleshooting

### Too few ground points

- **Lower the `distance_threshold`** in RANSAC or increase the
  `height_threshold` — the ground surface may not be perfectly flat.
- Verify that the point cloud is in the correct coordinate frame.  If
  the Z axis does not point up, the height-based methods will fail
  silently.
- Check the sensor's extrinsic calibration: an incorrect mounting height
  shifts the entire cloud vertically.

### Too many non-ground points classified as ground

- **Tighten the `distance_threshold`** in RANSAC (e.g. 0.1 m instead of
  0.2 m).
- Use the combined method (height pre-filter + RANSAC) so that tall
  objects are excluded before plane fitting.
- Increase `normal_threshold` in RANSAC or the verticality threshold in
  the normal-based method to reject sloped surfaces.

### RANSAC returns a vertical plane instead of the ground

- The `normal_threshold` guard should prevent this.  If it still
  happens, ensure you are working in a frame where ground normals point
  along +Z (the FLU ego frame).
- Increase `max_iterations` — with very few ground points relative to
  walls or buildings, more iterations are needed.

### Ground segmentation is slow

- Downsample the cloud before running RANSAC (e.g. every 4th point):
  `xyz_down = xyz[::4]`.
- For the normal-based method, reduce `k` (fewer neighbours) or use a
  voxel grid to downsample.
- The combined method (height pre-filter + RANSAC) is usually the
  fastest because it reduces the candidate set before the expensive
  fitting step.

### Radar clutter filtering removes real objects

- Raise the RCS threshold or narrow the range window.  Clutter filtering
  is inherently heuristic for 2-D radars; combine with camera or LiDAR
  data for more reliable classification.
