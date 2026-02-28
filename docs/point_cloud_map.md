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

## API Reference

### `PointCloudMap`

| Method / Property | Description |
|---|---|
| `add_scan(points, ego_to_world, *, colors=None)` | Transform *points* to the world frame and append to the map. |
| `get_points()` | Return a copy of the accumulated `(N, 3)` float64 world-frame points. |
| `get_colors()` | Return a copy of the accumulated `(N, 3)` uint8 RGB colours, or `None`. |
| `voxel_downsample(voxel_size)` | Downsample the map in place using a voxel grid filter. |
| `clear()` | Remove all accumulated points and colours. |
| `__len__()` | Number of points currently stored. |

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
