# Map Management: Submap Division and Keyframe Selection

## Overview

`sensor_transposition.submap_manager` provides two cooperating components for
managing maps in large-scale or long-duration SLAM sessions:

| Class | Description |
|-------|-------------|
| `KeyframeSelector` | Selects keyframes from a stream of poses based on translation and rotation thresholds |
| `SubmapManager` | Divides accepted keyframes into bounded local submaps with optional overlap |

These components solve a core scalability problem: for long traversals, storing
and optimising every incoming frame is computationally prohibitive.  Keyframe
selection reduces the problem size by discarding redundant near-duplicate
poses; submap division bounds the per-submap memory and optimisation cost and
enables parallel or hierarchical back-end processing.

---

## Why Map Management?

A full SLAM session might accumulate tens of thousands of frames.  Without
management:

* **Pose graph optimisation** cost grows as O(*n*³) in the number of keyframes.
* **Point cloud maps** consume unbounded memory.
* **Loop closure search** slows as the database grows.

Map management addresses all three by keeping only a sparse, well-spaced set
of *keyframes* and grouping them into *submaps* of bounded size:

```
Raw frames (100 Hz)
        │  KeyframeSelector
        │  (translation ≥ 1 m  OR  rotation ≥ 10°)
        ▼
Keyframe stream (≈ 1 Hz)
        │  SubmapManager
        │  (max 50 keyframes per submap, overlap 5)
        ▼
Submap 0  [kf 0 … 49]   ──┐
Submap 1  [kf 45 … 94]  ──┤── independent local maps
Submap 2  [kf 90 … 139] ──┘   (5-keyframe overlap for alignment)
```

---

## Quick Start

```python
import numpy as np
from sensor_transposition.submap_manager import KeyframeSelector, SubmapManager

# Accept a new keyframe every ≥ 1 m or ≥ 10° rotation.
selector = KeyframeSelector(translation_threshold=1.0, rotation_threshold_deg=10.0)

# Divide keyframes into submaps of at most 50, with 5-keyframe overlap.
manager = SubmapManager(max_keyframes_per_submap=50, overlap=5)

keyframe_id = 0
for pose, lidar_scan in zip(frame_poses, lidar_scans):
    if selector.check_and_accept(pose.transform):
        manager.add_keyframe(keyframe_id, pose.transform, lidar_scan)
        keyframe_id += 1

submaps = manager.get_all_submaps()
print(f"Accepted {selector.keyframe_count} keyframes → {manager.num_submaps} submaps.")
```

---

## Keyframe Selection

### Algorithm

`KeyframeSelector` maintains the pose of the last accepted keyframe and
compares each incoming pose against two thresholds:

| Criterion | Condition |
|-----------|-----------|
| Translation | Euclidean distance from last keyframe ≥ `translation_threshold` |
| Rotation | Rotation angle from last keyframe ≥ `rotation_threshold_deg` |

A pose is accepted when **either** condition is met.  The very first pose
presented is always accepted.

### Two-phase API

The selector exposes a read-only query and a separate state-update method so
that the caller can interleave acceptance decisions with other logic:

```python
selector = KeyframeSelector(translation_threshold=2.0, rotation_threshold_deg=15.0)

# --- Option A: combined check + accept (most common) ---
if selector.check_and_accept(current_pose):
    process_keyframe(current_pose)

# --- Option B: query then decide ---
if selector.should_add_keyframe(current_pose):
    if extra_condition():
        selector.mark_accepted(current_pose)
        process_keyframe(current_pose)
```

### Choosing thresholds

| Environment | Suggested `translation_threshold` | Suggested `rotation_threshold_deg` |
|-------------|-----------------------------------|------------------------------------|
| Indoor, dense mapping | 0.3 – 0.5 m | 5 – 10° |
| Outdoor, urban | 1 – 2 m | 10 – 15° |
| Highway / long-range | 3 – 5 m | 15 – 20° |

---

## Submap Division

### Algorithm

`SubmapManager` accumulates accepted keyframes into the current `Submap`.
When the current submap reaches `max_keyframes_per_submap` entries, a new
submap is created.  If `overlap > 0`, the last *overlap* keyframes of the
finishing submap are replayed into the start of the new one:

```
max_keyframes_per_submap = 5, overlap = 2

submap 0:  [k0  k1  k2  k3  k4]        ← filled at k4
submap 1:  [k3  k4  k5  k6  k7]        ← starts with k3, k4 (overlap)
submap 2:  [k6  k7  k8  k9  k10]       ← starts with k6, k7
```

### Adding keyframes with colours

```python
from sensor_transposition.lidar_camera import project_lidar_to_image, color_lidar_from_image

for pose, scan, image in zip(poses, scans, images):
    if selector.check_and_accept(pose.transform):
        pixel_coords, valid = project_lidar_to_image(
            scan, lidar_to_camera, K, img_w, img_h
        )
        colors = color_lidar_from_image(image, pixel_coords, valid)
        manager.add_keyframe(kf_id, pose.transform, scan, colors=colors)
        kf_id += 1
```

### Accessing submaps

```python
# All submaps (including the one in progress).
all_submaps = manager.get_all_submaps()

# The submap currently being built.
active = manager.get_current_submap()

# A specific submap by ID.
sm = manager.get_submap(submap_id=2)
print(sm.keyframe_ids)          # list of keyframe IDs in this submap
print(sm.size)                  # number of keyframe entries (including overlap)
world_pts = sm.point_cloud.get_points()  # (N, 3) float64
```

### Choosing `max_keyframes_per_submap` and `overlap`

| Scenario | `max_keyframes_per_submap` | `overlap` |
|----------|---------------------------|-----------|
| Indoor rooms (short corridors) | 20 – 50 | 3 – 5 |
| Outdoor urban blocks | 50 – 100 | 5 – 10 |
| Long-range road mapping | 100 – 200 | 10 – 20 |

A larger overlap gives more shared observations between adjacent submaps for
inter-submap loop closure and alignment, at the cost of extra point-cloud
storage.

---

## API Reference

### `KeyframeSelector`

```python
KeyframeSelector(
    translation_threshold: float = 1.0,
    rotation_threshold_deg: float = 10.0,
)
```

| Parameter | Description |
|-----------|-------------|
| `translation_threshold` | Minimum Euclidean translation (m) from the last keyframe to trigger acceptance.  Must be > 0. |
| `rotation_threshold_deg` | Minimum rotation angle (°) from the last keyframe to trigger acceptance.  Must be ≥ 0; set to `0.0` to disable the rotation criterion. |

#### Methods and properties

| Name | Description |
|------|-------------|
| `should_add_keyframe(transform)` | Return `True` if *transform* qualifies as a new keyframe.  Read-only — does not update state. |
| `mark_accepted(transform)` | Record *transform* as the most recently accepted keyframe pose and increment `keyframe_count`. |
| `check_and_accept(transform)` | Call `should_add_keyframe`; if `True`, also call `mark_accepted`.  Returns acceptance result. |
| `keyframe_count` *(property)* | Total number of keyframes accepted and marked so far. |
| `last_pose` *(property)* | Copy of the most recently accepted keyframe pose (4×4), or `None`. |

---

### `Submap`

| Attribute | Type | Description |
|-----------|------|-------------|
| `submap_id` | `int` | Unique identifier. |
| `keyframe_ids` | `list[int]` | Ordered keyframe IDs belonging to this submap (may include overlap copies). |
| `origin_pose` | `ndarray (4×4)` | World-frame pose of the first keyframe in this submap. |
| `point_cloud` | `PointCloudMap` | Accumulated world-frame point cloud for this submap. |
| `size` *(property)* | `int` | Number of keyframe entries (`len(keyframe_ids)`). |

---

### `SubmapManager`

```python
SubmapManager(
    max_keyframes_per_submap: int = 50,
    overlap: int = 0,
)
```

| Parameter | Description |
|-----------|-------------|
| `max_keyframes_per_submap` | Maximum keyframe entries per submap before a new one is created.  Must be ≥ 1. |
| `overlap` | Keyframes from the end of the finishing submap that are replayed at the start of the next.  Must be ≥ 0 and < `max_keyframes_per_submap`. |

#### `add_keyframe(keyframe_id, transform, points, *, colors=None) → Submap`

Add a keyframe scan.  Creates a new submap (seeded with overlap frames if
configured) whenever the current submap is full.

#### Other methods and properties

| Name | Description |
|------|-------------|
| `get_current_submap()` | Return the active `Submap`, or `None` before the first keyframe. |
| `get_all_submaps()` | Return all `Submap` objects (including the one in progress). |
| `get_submap(submap_id)` | Return the `Submap` with the given ID; raises `KeyError` if not found. |
| `num_submaps` *(property)* | Number of submaps created so far. |
| `total_keyframes` *(property)* | Total keyframe entries across all submaps (overlap copies counted per submap). |

---

## Integration in the SLAM Pipeline

```
LiDAR scan (100 Hz)  +  Ego pose (from ICP / EKF)
          │
          ▼
  KeyframeSelector.check_and_accept(pose.transform)
          │  accept (≈ 1–5 Hz)
          ▼
  SubmapManager.add_keyframe(kf_id, transform, scan)
          │
          ├──► Submap.point_cloud    ← per-submap PointCloudMap
          │
          ├──► PoseGraph (per submap) + optimize_pose_graph()
          │         ← lidar/scan_matching.icp_align for odometry edges
          │         ← loop_closure.ScanContextDatabase for loop edges
          │
          └──► SlidingWindowSmoother (global trajectory)
```

### Recommended workflow for large-scale sessions

1. **Collect keyframes** using `KeyframeSelector`.
2. **Build submaps** with `SubmapManager`.
3. **Optimise each submap independently** using `PoseGraph` + `optimize_pose_graph`.
4. **Detect inter-submap loop closures** by querying `ScanContextDatabase`
   with the descriptor of each new submap's last keyframe against descriptors
   from completed submaps.
5. **Build a global pose graph** whose nodes are submap origins and whose
   edges are the inter-submap constraints, then run a final global
   optimisation.

---

## References

* Bosse, M., Newman, P., Leonard, J., & Teller, S. (2004). "An Atlas framework
  for scalable mapping." *IEEE ICRA 2003*, 1899–1906.
* Stachniss, C., Hähnel, D., & Burgard, W. (2004). "Exploration with active
  loop-closing for FastSLAM." *IEEE IROS 2004*, 1505–1510.
* Shan, T., Englot, B., Meyers, D., Wang, W., Ratti, C., & Rus, D. (2020).
  "LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and
  Mapping." *IEEE IROS 2020*.
