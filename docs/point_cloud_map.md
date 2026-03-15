# Accumulated Point-Cloud Map

## Overview

`sensor_transposition.point_cloud_map` provides :class:`PointCloudMap`, a
lightweight incremental accumulator that assembles a global point-cloud map
from successive LiDAR scans.  As the sensor platform moves, each scan is
transformed from the sensor body frame into a fixed world/map frame using the
ego pose produced by the SLAM front-end and appended to a growing buffer.
An optional voxel grid downsampling step keeps memory bounded for long
trajectories.

---

## Why an Accumulated Map?

Individual LiDAR frames capture only the local environment visible from one
sensor pose.  A **global point-cloud map** stitches together the partial views
from every keyframe into a single consistent representation of the entire
environment.  Downstream consumers of the map include:

* **Loop closure verification** – matching incoming scans against a local
  region of the global map.
* **Localisation** – re-using a pre-built map for subsequent traversals.
* **Obstacle detection / path planning** – building a high-resolution 3-D
  occupancy model from the dense point cloud.
* **Inspection and visualisation** – exporting the coloured map for review in
  tools such as CloudCompare, MeshLab, or Open3D.

---

## Quick Start

```python
from sensor_transposition.point_cloud_map import PointCloudMap
from sensor_transposition.frame_pose import FramePoseSequence

# Load a pre-built trajectory (e.g. output of optimize_pose_graph).
trajectory = FramePoseSequence.from_yaml("trajectory.yaml")

pcd_map = PointCloudMap()

for frame_pose, lidar_scan in zip(trajectory, lidar_scans):
    pcd_map.add_scan(lidar_scan, frame_pose.transform)

# Reduce map density to 5-cm voxels.
pcd_map.voxel_downsample(voxel_size=0.05)

world_points = pcd_map.get_points()   # (N, 3) float64, world frame
print(f"Map contains {len(pcd_map):,} points after downsampling.")
```

---

## Adding Coloured Scans

Use
[`color_lidar_from_image`](camera_lidar_extrinsic_calibration.md)
to paint each scan with colours from a synchronised camera image before
accumulation:

```python
from sensor_transposition.lidar_camera import (
    project_lidar_to_image,
    color_lidar_from_image,
)
from sensor_transposition.point_cloud_map import PointCloudMap

pcd_map = PointCloudMap()

for frame_pose, lidar_scan, camera_image in zip(trajectory, scans, images):
    # Project LiDAR points onto the image plane.
    pixel_coords, valid = project_lidar_to_image(
        lidar_scan, lidar_to_camera_transform, camera_K,
        img_w, img_h,
    )
    # Sample RGB colour at each projected pixel.
    colors = color_lidar_from_image(camera_image, pixel_coords, valid)

    pcd_map.add_scan(lidar_scan, frame_pose.transform, colors=colors)

world_pts = pcd_map.get_points()   # (N, 3) float64
rgb       = pcd_map.get_colors()   # (N, 3) uint8, or None for uncoloured maps
```

> **Note:** Either *all* calls to `add_scan` must supply `colors` or *none*
> may.  Mixing coloured and uncoloured scans raises `ValueError`.

---

## Voxel Grid Downsampling

`voxel_downsample(voxel_size)` partitions 3-D space into axis-aligned cubes
of side length `voxel_size` (metres) and replaces all points inside each
occupied voxel with their centroid.  Per-point colours are averaged over the
voxel.

```python
pcd_map.voxel_downsample(voxel_size=0.10)   # 10-cm voxels
```

Choosing an appropriate voxel size:

| Use-case | Recommended voxel size |
|---|---|
| Dense indoor mapping | 1 – 5 cm |
| Outdoor/urban mapping | 5 – 20 cm |
| Long-range road mapping | 20 – 50 cm |

---

## Bounded-Memory Operation

Pass `max_points` to cap the internal buffer at a fixed number of points.
When the buffer is full, the oldest points are discarded (FIFO) to make
room for new ones.

```python
pcd_map = PointCloudMap(max_points=5_000_000)   # ~190 MB at float64
```

---

## Map Serialisation

### Saving

`save_pcd(path)` writes an ASCII PCD (version 0.7) file, compatible with
the [Point Cloud Library (PCL)](https://pointclouds.org/) and most
point-cloud viewers:

```python
pcd_map.save_pcd("map.pcd")
```

`save_ply(path)` writes an ASCII PLY file, compatible with CloudCompare,
MeshLab, and Open3D:

```python
pcd_map.save_ply("map.ply")
```

Both formats include per-point RGB colour when present.

### Loading

Previously saved maps can be reloaded as a new `PointCloudMap` instance:

```python
from sensor_transposition.point_cloud_map import PointCloudMap

# Reload from PCD
map_pcd = PointCloudMap.from_pcd("map.pcd")

# Reload from PLY
map_ply = PointCloudMap.from_ply("map.ply")

print(f"Loaded {len(map_pcd):,} points from PCD.")
print(f"Loaded {len(map_ply):,} points from PLY.")
```

> **Note:** Only ASCII-encoded files are supported.  Binary PCD or binary
> PLY files will raise `ValueError`.

---

## Dynamic Object Filtering

Moving objects (pedestrians, vehicles, cyclists) that appear in multiple
consecutive scans leave "ghost" trails in the accumulated map that degrade
localisation accuracy.  `sensor_transposition.point_cloud_map` provides
two complementary filters to remove transient points before they enter the
map.

### Doppler Velocity Filter – `filter_dynamic_points`

When a co-registered radar or Doppler-LiDAR scan is available, each point
carries a **radial velocity** measurement.  Points whose absolute Doppler
speed exceeds a threshold are classified as dynamic and should be excluded.

```python
from sensor_transposition.point_cloud_map import filter_dynamic_points

# doppler_velocities is an (N,) array of signed radial speeds (m/s)
# from a co-registered radar or Doppler LiDAR scan.
static_mask = filter_dynamic_points(
    lidar_cloud,          # (N, 3) XYZ
    doppler_velocities,   # (N,)  m/s
    doppler_threshold=0.5,
)

static_cloud = lidar_cloud[static_mask]
pcd_map.add_scan(static_cloud, ego_to_world)
```

> **Rule of thumb:** 0.5 m/s is a good starting threshold for urban driving.
> Lower the threshold in pedestrian-heavy environments; raise it on
> high-speed roads where even stationary objects may appear to move due to
> ego-motion estimation error.

### Consistency Filter – `consistency_filter`

When no Doppler data is available, transient objects can be detected by
checking that each point in one scan has support from a *reference* scan
of the same area (e.g. a second traversal or an adjacent keyframe).
Points that are not supported within `threshold_m` are assumed to belong
to transient obstacles.

```python
from sensor_transposition.point_cloud_map import consistency_filter

# scan_a: current scan to be added to the map.
# scan_b: a trusted reference scan of the same region.
static_mask = consistency_filter(
    scan_a,
    scan_b,
    threshold_m=0.5,   # metres
)

static_points = scan_a[static_mask]
pcd_map.add_scan(static_points, ego_to_world)
```

> **Note:** The consistency filter requires two overlapping views of the
> same region.  For long trajectories, the reference scan can be drawn from
> the global map's local neighbourhood around the current pose.

### Combined workflow

Both filters can be applied in sequence:

```python
from sensor_transposition.point_cloud_map import (
    filter_dynamic_points,
    consistency_filter,
    PointCloudMap,
)

pcd_map = PointCloudMap()

for lidar_cloud, doppler_v, reference_cloud, ego_to_world in data_stream:
    # 1. Remove Doppler-detected dynamic points.
    static1 = filter_dynamic_points(lidar_cloud, doppler_v, 0.5)
    cloud1 = lidar_cloud[static1]

    # 2. Remove consistency-check failures against the previous keyframe.
    static2 = consistency_filter(cloud1, reference_cloud, 0.5)
    cloud2 = cloud1[static2]

    pcd_map.add_scan(cloud2, ego_to_world)
```

---

## API Reference

### `PointCloudMap`

| Method / Property | Description |
|---|---|
| `add_scan(points, ego_to_world, *, colors=None)` | Transform *points* to the world frame and append to the map. |
| `get_points()` | Return a copy of the accumulated `(N, 3)` float64 world-frame points. |
| `get_colors()` | Return a copy of the accumulated `(N, 3)` uint8 RGB colours, or `None`. |
| `voxel_downsample(voxel_size)` | Downsample the map in place using a voxel grid filter. |
| `save_pcd(path)` | Save the map to an ASCII PCD file (PCL-compatible, v0.7). |
| `save_ply(path)` | Save the map to an ASCII PLY file (compatible with CloudCompare, MeshLab, Open3D). |
| `from_pcd(path)` *(classmethod)* | Load a map from an ASCII PCD file. |
| `from_ply(path)` *(classmethod)* | Load a map from an ASCII PLY file. |
| `clear()` | Remove all accumulated points and colours. |
| `__len__()` | Number of points currently stored. |

### Dynamic filtering utilities

| Function | Description |
|---|---|
| `filter_dynamic_points(cloud, velocity_map, doppler_threshold=0.5)` | Return a boolean mask keeping only static points (those with `|velocity| ≤ doppler_threshold`). |
| `consistency_filter(cloud, reference_cloud, threshold_m=0.5)` | Return a boolean mask keeping only points in *cloud* that have at least one neighbour within `threshold_m` in *reference_cloud*. |

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
 PoseGraph + optimize_pose_graph()  ← pose_graph.py
        │ (world-frame poses)
        ▼
 PointCloudMap.add_scan()      ← point_cloud_map.py
        │
        ▼
 voxel_downsample()
        │
        ▼
 Global point-cloud map
```

---

## References

* Zhang & Singh, "LOAM: Lidar Odometry and Mapping in Real-time", RSS 2014.
* Shan *et al.*, "LIO-SAM: Tightly-coupled Lidar Inertial Odometry via
  Smoothing and Mapping", IROS 2020.
* Rusu & Cousins, "3D is here: Point Cloud Library (PCL)", ICRA 2011.
