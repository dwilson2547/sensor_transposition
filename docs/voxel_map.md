# Voxel Map / TSDF Volumetric Representation

## Overview

`sensor_transposition.voxel_map` provides :class:`TSDFVolume`, a
memory-efficient volumetric map based on the **Truncated Signed Distance
Function (TSDF)** introduced by KinectFusion (Newcombe *et al.*, 2011).

The environment is represented as a regular 3-D grid of voxels.  Each voxel
stores:

* A **normalised TSDF value** in ``[-1, 1]``: the signed distance to the
  nearest observed surface, divided by the truncation radius.
* A **fusion weight**: the number of observations that have contributed to
  that voxel.

New LiDAR scans are integrated via a **running weighted average**.  Surface
points can be extracted by finding voxels near the zero-crossing of the TSDF.

The implementation is pure NumPy and introduces no additional dependencies
beyond those already required by ``sensor_transposition``.

---

## Quick Start

```python
import numpy as np
from sensor_transposition.voxel_map import TSDFVolume
from sensor_transposition.frame_pose import FramePoseSequence

# 10-cm voxels, 20 m × 20 m × 5 m volume.
volume = TSDFVolume(
    voxel_size=0.10,
    origin=np.array([-10.0, -10.0, -0.5]),
    dims=(200, 200, 50),
)

# Load trajectory (e.g. output of optimize_pose_graph).
trajectory = FramePoseSequence.from_yaml("trajectory.yaml")

# Integrate all LiDAR scans.
for frame_pose, lidar_scan in zip(trajectory, lidar_scans):
    volume.integrate(lidar_scan, frame_pose.transform)

# Extract surface points near the zero-crossing.
surface_pts = volume.extract_surface_points(threshold=0.1)
print(f"Surface: {surface_pts.shape[0]:,} points")

# Access the raw TSDF and weight arrays.
tsdf    = volume.get_tsdf()     # (200, 200, 50) float64; NaN = unseen
weights = volume.get_weights()  # (200, 200, 50) float64
```

---

## TSDF Sign Convention

The sign follows the KinectFusion convention:

| TSDF value | Meaning |
|------------|---------|
| `> 0` | Voxel is **closer to the sensor** than the surface (free space). |
| `= 0` | Voxel is **on the surface**. |
| `< 0` | Voxel is **behind the surface** (solid interior). |

The stored value is `sdf / truncation`, so it always lies in `[-1, 1]`.
Unseen voxels contain `NaN`.

---

## Integration Model

For each surface point **P** observed from sensor origin **O**:

1. Compute the unit ray direction **d** = (*P* − *O*) / ‖*P* − *O*‖.
2. For each voxel **V** whose centre lies within `truncation` metres of **P**:

   ```
   sdf(V) = ‖P − O‖ − (V_centre − O) · d
   ```

3. Normalise: `sdf_norm = clip(sdf / truncation, −1, 1)`.
4. Update the running weighted average:

   ```
   tsdf_new(V) = (weight(V) × tsdf(V) + sdf_norm) / (weight(V) + 1)
   weight_new(V) = weight(V) + 1
   ```

Only voxels with `|sdf| ≤ truncation` are updated per observation.

---

## API Reference

### `TSDFVolume`

```python
TSDFVolume(
    voxel_size: float,
    origin: Optional[np.ndarray] = None,   # shape (3,); default [0, 0, 0]
    dims: Tuple[int, int, int] = (100, 100, 100),
    truncation: Optional[float] = None,    # default = 3 × voxel_size
)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `voxel_size` | `float` | Side length of each voxel in metres. |
| `origin` | `(3,)` ndarray | World-frame corner of voxel `(0, 0, 0)`. |
| `dims` | `(int, int, int)` | Number of voxels `(nx, ny, nz)`. |
| `truncation` | `float` | Truncation radius in metres. |

#### `integrate(points, ego_to_world)`

Fuse a new point-cloud observation into the volume.

| Parameter | Type | Description |
|-----------|------|-------------|
| `points` | `(N, 3)` float | XYZ in the sensor body frame (metres). |
| `ego_to_world` | `(4, 4)` float | Homogeneous transform from body to world frame. |

#### `get_tsdf() → np.ndarray`

Returns a `(nx, ny, nz)` float64 copy of the TSDF values.  Unseen voxels
contain `NaN`; observed voxels contain values in `[-1, 1]`.

#### `get_weights() → np.ndarray`

Returns a `(nx, ny, nz)` float64 copy of the fusion weights.  Zero for
unobserved voxels; equal to the number of contributing observations otherwise.

#### `extract_surface_points(threshold=0.1) → np.ndarray`

Returns a `(M, 3)` float64 array of world-frame voxel centres satisfying
`weight > 0` and `|tsdf| ≤ threshold`.  Returns `(0, 3)` when no such voxels
exist.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | `0.1` | Normalised TSDF threshold in `[0, 1]`. |

#### `voxel_to_world(ix, iy, iz) → (x, y, z)`

Return the world-frame coordinates of the **centre** of voxel `(ix, iy, iz)`.

#### `world_to_voxel(x, y, z) → (fx, fy, fz)`

Return the (possibly non-integer) voxel-space coordinates for a world-frame
position.  The integer part gives the voxel index; values outside
`[0, nx/ny/nz)` indicate positions outside the volume.

#### `clear()`

Reset all voxels to the unobserved state (`NaN` TSDF, zero weight).

---

## Coordinate Frame

```
world z ↑                  dims = (nx, ny, nz) voxels
         │
         │      ┌────────────────────────────┐  ← z = origin[2] + nz × voxel_size
         │      │ voxel (0, 0, nz-1) ...    │
         │      │ ...                        │  nz × voxel_size
         │      │ voxel (0, 0, 0) at corner  │
         │      └────────────────────────────┘  ← z = origin[2]
         │      ↑ origin
         └──────────────────────────────→ world x
```

* `origin` is the world-frame position of the **corner** (not centre) of
  voxel `(0, 0, 0)`.
* `voxel_to_world(ix, iy, iz)` returns the voxel **centre**:
  `x = origin[0] + (ix + 0.5) × voxel_size`.

---

## Choosing Parameters

| Parameter | Guidance |
|-----------|----------|
| `voxel_size` | 5 – 10 cm for indoor mapping; 10 – 20 cm for outdoor. |
| `dims` | Cover the expected map extent: `nx = map_width / voxel_size`. |
| `truncation` | 2 – 5 × `voxel_size`; larger values smooth the surface more. |

Memory usage per volume: `nx × ny × nz × 16` bytes (two float64 arrays).

---

## Integration with Other SLAM Modules

```
LiDAR scan (sensor frame)
        │
        ▼
 deskew_scan()                 ← lidar/motion_distortion.py
        │
        ▼
 icp_align()                   ← lidar/scan_matching.py
        │ (relative transform)
        ▼
 optimize_pose_graph()         ← pose_graph.py
        │ (world-frame poses)
        ▼
 TSDFVolume.integrate()        ← voxel_map.py
        │
        ▼
 extract_surface_points()
        │
        ▼
 Dense surface reconstruction
```

```python
from sensor_transposition.voxel_map import TSDFVolume
from sensor_transposition.point_cloud_map import PointCloudMap

tsdf_map = TSDFVolume(voxel_size=0.10, origin=np.array([-10., -10., -1.]),
                      dims=(200, 200, 20))
pcd_map  = PointCloudMap()

for frame_pose, scan in zip(trajectory, scans):
    T = frame_pose.transform
    tsdf_map.integrate(scan, T)   # dense volumetric reconstruction
    pcd_map.add_scan(scan, T)     # raw point-cloud accumulation
```

---

## References

* Newcombe *et al.*, "KinectFusion: Real-time dense surface mapping and
  tracking", ISMAR 2011.
* Whelan *et al.*, "ElasticFusion: Dense SLAM without a pose graph", RSS 2015.
* Oleynikova *et al.*, "Voxblox: Incremental 3D Euclidean Signed Distance
  Fields for on-board MAV planning", IROS 2017.
