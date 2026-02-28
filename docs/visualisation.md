# Multi-Sensor Synchronised Visualisation

`sensor_transposition.visualisation` provides a lightweight, **pure-NumPy**
toolkit for visualising synchronised multi-sensor data.  No display framework
is required — all renderers return plain NumPy arrays (`uint8` RGB images)
that can be written to disk, shown by any image viewer, or passed straight to
external tools such as **Open3D**, **RViz**, or **rerun.io**.

---

## Overview

| Function / Class | Purpose |
|---|---|
| `colour_by_height` | Map z-values to a jet-like RGB palette |
| `render_birdseye_view` | Project a 3-D point cloud onto a BEV canvas |
| `render_trajectory_birdseye` | Draw a pose trajectory onto a BEV canvas |
| `overlay_lidar_on_image` | Overlay depth-coded LiDAR dots on a camera image |
| `export_point_cloud_open3d` | Serialise a point cloud to an Open3D-compatible dict |
| `export_trajectory_rviz` | Serialise a trajectory to RViz `Marker` dicts |
| `SensorFrameVisualiser` | Per-frame multi-sensor container with rendering helpers |

---

## Quick-start

```python
from sensor_transposition.visualisation import (
    SensorFrameVisualiser,
    render_birdseye_view,
    overlay_lidar_on_image,
)
from sensor_transposition.lidar_camera import project_lidar_to_image
```

### 1. Bird's-eye view of an accumulated point-cloud map

```python
from sensor_transposition.point_cloud_map import PointCloudMap

pcd_map = PointCloudMap()
for frame_pose, scan in zip(trajectory, lidar_scans):
    pcd_map.add_scan(scan, frame_pose.transform)

bev = render_birdseye_view(pcd_map.get_points(), resolution=0.10)
# bev is (H, W, 3) uint8 — save or display with any tool:
# imageio.imwrite("map_bev.png", bev)
```

### 2. LiDAR depth overlay on a camera image

```python
pixel_coords, valid = project_lidar_to_image(
    lidar_scan, lidar_to_cam_T, K, img_w, img_h
)
depth = lidar_scan[:, 0]   # use X-distance as proxy for depth
overlay = overlay_lidar_on_image(camera_image, pixel_coords, valid, depth)
```

### 3. Combined frame visualiser

```python
vis = SensorFrameVisualiser()
vis.set_point_cloud(lidar_scan)
vis.set_camera_image(camera_image)
vis.set_trajectory(trajectory_xy)   # (M, 2) XY world positions

# Bird's-eye view with all layers stacked
bev_frame = vis.render_birdseye(resolution=0.10)

# Camera with LiDAR depth overlay
cam_frame = vis.render_camera_with_lidar(pixel_coords, valid, depth)
```

---

## API Reference

### `colour_by_height(z_values, *, z_min=None, z_max=None)`

Maps height values to a **jet-like** 5-colour ramp (blue → cyan → green →
yellow → red) and returns an `(N, 3)` `uint8` RGB array.

```python
from sensor_transposition.visualisation import colour_by_height
import numpy as np

z = np.array([-2.0, 0.0, 2.0])
colours = colour_by_height(z)
# array([[  0,   0, 255],   # blue  (lowest)
#        [  0, 255,   0],   # green (mid)
#        [255,   0,   0]], dtype=uint8)  # red (highest)
```

---

### `render_birdseye_view(points, *, resolution=0.10, colors=None, ...)`

Projects a `(N, 3)` point cloud onto a 2-D canvas coloured by z-height (or
supplied per-point colours).  Returns an `(H, W, 3)` `uint8` image.

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `resolution` | `0.10` | Metres per pixel |
| `colors` | `None` | `(N, 3)` per-point RGB; if omitted, z-height is used |
| `z_min` / `z_max` | auto | Colour-scale bounds |
| `canvas_size` | auto | `(width, height)` in pixels |
| `origin` | auto | World XY at bottom-left pixel |
| `background` | `(0,0,0)` | Background RGB |

```python
bev = render_birdseye_view(scan, resolution=0.05, z_min=-2.0, z_max=3.0)
```

---

### `render_trajectory_birdseye(positions, *, resolution=0.10, ...)`

Draws trajectory waypoints as filled squares onto a BEV canvas.  Returns an
`(H, W, 3)` `uint8` image.

```python
import numpy as np
from sensor_transposition.visualisation import render_trajectory_birdseye

# Extract XY positions from a FramePoseSequence
xy = np.array([fp.translation[:2] for fp in seq.poses])
traj_img = render_trajectory_birdseye(xy, resolution=0.10, color=(255, 127, 0))
```

---

### `overlay_lidar_on_image(image, pixel_coords, valid, depth, *, ...)`

Overlays LiDAR depth-coded dots on top of a camera image.  A copy of the
image is returned; the input is not modified.

```python
pixel_coords, valid = project_lidar_to_image(scan, T_lidar_cam, K, W, H)
depth = np.linalg.norm(scan[:, :3], axis=1)   # Euclidean range
result = overlay_lidar_on_image(
    camera_image, pixel_coords, valid, depth,
    d_min=0.5, d_max=50.0, radius=3
)
```

---

### `export_point_cloud_open3d(points, colors=None)`

Serialises a point cloud to the dict format consumed by Open3D's
`PointCloud.from_dict` / `to_dict`.  `uint8` colours are automatically
normalised to `[0, 1]` as required by Open3D.

```python
import open3d as o3d
from sensor_transposition.visualisation import export_point_cloud_open3d

d = export_point_cloud_open3d(points, colors)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(d["points"])
if d["colors"] is not None:
    pcd.colors = o3d.utility.Vector3dVector(d["colors"])
o3d.visualization.draw_geometries([pcd])
```

---

### `export_trajectory_rviz(frame_poses, *, frame_id="map", ...)`

Converts a list of `FramePose` objects to RViz-compatible
`visualization_msgs/Marker` dicts (type `SPHERE`, action `ADD`).

```python
from sensor_transposition.visualisation import export_trajectory_rviz

markers = export_trajectory_rviz(seq.poses, frame_id="map", scale=0.3)
# Publish each dict as a visualization_msgs/Marker via rospy or rclpy.
```

---

### `SensorFrameVisualiser`

A per-frame container that holds up to four sensor streams and exposes
rendering helpers:

| Method | Description |
|---|---|
| `set_point_cloud(points, colors=None)` | Set the LiDAR point cloud |
| `set_camera_image(image)` | Set the camera image `(H, W, 3)` uint8 |
| `set_trajectory(positions)` | Set `(M, 2+)` trajectory XY positions |
| `set_radar_points(points)` | Set `(K, 2+)` radar returns |
| `render_birdseye(*, resolution=0.10, ...)` | Render layered BEV image |
| `render_camera_with_lidar(pixel_coords, valid, depth, ...)` | Camera + LiDAR overlay |

The BEV render layers points (bottom), radar returns, then trajectory on top.
An unset stream is simply omitted from the output.

---

## Integration with other modules

```
                    ┌─────────────────────────────────────────────┐
                    │           SensorFrameVisualiser              │
                    │                                             │
 LiDAR scan ───────►│ set_point_cloud()  ──► render_birdseye()   │──► (H,W,3) uint8
 Camera image ─────►│ set_camera_image() ──► render_camera_with_ │──► (H,W,3) uint8
 Trajectory ───────►│ set_trajectory()       lidar()             │
 Radar scan ───────►│ set_radar_points()                         │
                    └─────────────────────────────────────────────┘

 PointCloudMap ────► render_birdseye_view()            ──────────────► (H,W,3) uint8
 FramePoseSequence ► render_trajectory_birdseye()      ──────────────► (H,W,3) uint8
 project_lidar_to_image + camera ► overlay_lidar_on_image() ─────────► (H,W,3) uint8
 PointCloudMap ────► export_point_cloud_open3d()       ──────────────► Open3D dict
 FramePoseSequence ► export_trajectory_rviz()          ──────────────► RViz Marker list
```

### Compositing two BEV layers

Because all renderers return NumPy arrays, layers can be composited manually:

```python
import numpy as np
from sensor_transposition.visualisation import (
    render_birdseye_view,
    render_trajectory_birdseye,
)

pts_bev = render_birdseye_view(map_points, resolution=0.10, canvas_size=(500, 500))
traj_bev = render_trajectory_birdseye(xy, resolution=0.10, canvas_size=(500, 500),
                                       origin=(ox, oy))

# Composite: trajectory dots overwrite point cloud pixels.
mask = np.any(traj_bev != 0, axis=2)
composite = pts_bev.copy()
composite[mask] = traj_bev[mask]
```
