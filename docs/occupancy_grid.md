# Occupancy Grid

A 2-D occupancy grid is a probabilistic map that partitions the horizontal
plane into a regular grid of square cells.  Each cell stores an estimate of
the probability that it is occupied by an obstacle.  Occupancy grids are the
standard map representation for path planning and obstacle avoidance in
ground-robot SLAM.

---

## Overview

`sensor_transposition` provides [`OccupancyGrid`](#occupancygrid) in
`sensor_transposition/occupancy_grid.py`.  Key features:

| Feature | Detail |
|---------|--------|
| **Probabilistic update** | Log-odds model; cells are updated by addition, so the cost per observation is O(1). |
| **Ray-casting (Bresenham)** | Cells between the sensor origin and each hit point are marked *free*; the hit cell is marked *occupied*. |
| **Height-band filter** | Only points with Z ∈ `[z_min, z_max]` are projected; suppresses ground and ceiling returns. |
| **ROS-compatible output** | `get_grid()` returns `int8` values: `−1` unknown, `0` free, `100` occupied. |
| **Pure NumPy** | No additional dependencies beyond those already required by `sensor_transposition`. |

---

## Quick Start

```python
import numpy as np
from sensor_transposition.occupancy_grid import OccupancyGrid

# 200 × 200 grid with 10 cm cells, centred on the world origin.
grid = OccupancyGrid(
    resolution=0.10,          # metres per cell
    width=200,                # cells in x (20 m)
    height=200,               # cells in y (20 m)
    origin=np.array([-10.0, -10.0]),  # world coords of cell (0, 0) bottom-left
    z_min=-0.3,               # ignore ground returns below -0.3 m
    z_max=2.5,                # ignore overhead structure above 2.5 m
)

# Insert LiDAR scans as the robot moves.
for frame_pose, lidar_scan in zip(trajectory, scans):
    grid.insert_scan(lidar_scan, frame_pose.transform)

# Read the result.
occupancy = grid.get_grid()   # (height, width) int8 array
probs     = grid.to_probability()  # (height, width) float64 in [0, 1]
```

---

## API Reference

### `OccupancyGrid`

```python
OccupancyGrid(
    resolution: float,
    width: int,
    height: int,
    origin: Optional[np.ndarray] = None,  # shape (2,); default [0, 0]
    *,
    z_min: float = -inf,
    z_max: float = +inf,
    log_odds_hit: float  = 0.85,   # update applied to the hit cell
    log_odds_miss: float = -0.40,  # update applied to free-ray cells
    log_odds_min: float  = -5.0,   # lower clamp (≈ 0.7 % probability)
    log_odds_max: float  = 5.0,    # upper clamp (≈ 99.3 % probability)
)
```

#### `insert_scan(points, ego_to_world, sensor_origin=None)`

Insert one LiDAR scan into the grid.

| Parameter | Type | Description |
|-----------|------|-------------|
| `points` | `(N, 3)` float | XYZ in the sensor body frame (metres). |
| `ego_to_world` | `(4, 4)` float | Homogeneous transform from body to world frame. |
| `sensor_origin` | `(3,)` or `(2,)` float, optional | World-frame sensor position used as the ray origin.  Defaults to the translation of `ego_to_world`. |

1. Points are transformed to world frame.
2. Points outside `[z_min, z_max]` are discarded.
3. For each remaining point a Bresenham ray is traced from `sensor_origin` to
   the hit cell; intermediate cells receive the *miss* log-odds update, and
   the hit cell receives the *hit* log-odds update.

#### `get_grid() → np.ndarray`

Returns a `(height, width)` `int8` array using the ROS
`nav_msgs/OccupancyGrid` convention:

| Value | Meaning |
|-------|---------|
| `-1` | Unknown — cell has never been observed. |
| `0` | Free — ray-cast evidence that the cell is unoccupied. |
| `100` | Occupied — at least one LiDAR return in this cell. |

#### `to_probability() → np.ndarray`

Returns a `(height, width)` float64 array with occupancy probabilities in
`[0, 1]`.  Unknown cells map to `0.5`.

#### `world_to_cell(x, y) → (col, row)`

Convert world-frame coordinates to grid cell indices.  The returned indices
may be outside `[0, width)` / `[0, height)` if the coordinate lies outside
the map extent.

#### `cell_to_world(col, row) → (x, y)`

Return the world-frame coordinates of the **centre** of a cell.

#### `clear()`

Reset all cells to the unknown state (log-odds = 0).

---

## Coordinate Frame

```
world y ↑
        │
        │   ┌───────────────────────┐
        │   │  (col=0,row=H-1)      │  height × resolution
        │   │  ...                  │
        │   │  (col=0,row=0)        │
        │   └───────────────────────┘
        │   origin                  width × resolution
        └──────────────────────────────→ world x
```

* `origin` is the world-frame position of the **bottom-left corner** of cell `(col=0, row=0)`.
* Column index increases with world-frame **x**; row index increases with world-frame **y**.
* `cell_to_world` returns the **centre** of a cell:
  `x = origin[0] + (col + 0.5) * resolution`.

---

## Integration with `PointCloudMap`

The occupancy grid and the accumulated point-cloud map complement each other:

```python
from sensor_transposition.point_cloud_map import PointCloudMap
from sensor_transposition.occupancy_grid import OccupancyGrid

pcd_map = PointCloudMap()
occ_map = OccupancyGrid(resolution=0.10, width=400, height=400,
                        origin=np.array([-20.0, -20.0]),
                        z_min=0.1, z_max=1.5)

for frame_pose, scan in zip(trajectory, scans):
    T = frame_pose.transform
    pcd_map.add_scan(scan, T)          # build dense 3-D map
    occ_map.insert_scan(scan, T)       # build 2-D occupancy map for planning
```

---

## Integration with `FramePoseSequence` and `PoseGraph`

After pose-graph optimisation, update each scan with its refined pose:

```python
from sensor_transposition.pose_graph import optimize_pose_graph
from sensor_transposition.occupancy_grid import OccupancyGrid

result = optimize_pose_graph(pose_graph)

occ_map = OccupancyGrid(resolution=0.10, width=400, height=400,
                        origin=np.array([-20.0, -20.0]))
occ_map.clear()  # start fresh with optimised poses

for node_id, scan in zip(node_ids, scans):
    optimised_transform = result.optimized_poses[node_id]["transform"]
    occ_map.insert_scan(scan, optimised_transform)
```

---

## Log-Odds Update Model

The implementation uses the standard inverse-sensor-model log-odds formulation:

```
L(m_i | z_{1:t}) = L(m_i | z_t) + L(m_i | z_{1:t-1}) - L(m_i)
```

where `L(x) = log(x / (1 - x))` is the log-odds transform.

The prior `L(m_i)` is zero (50 % probability), so the update simplifies to:

```
log_odds[cell] += log_odds_hit   # for the hit cell
log_odds[cell] += log_odds_miss  # for each free-ray cell
```

Both values are clamped to `[log_odds_min, log_odds_max]` after each update
to prevent numerical saturation and allow the map to adapt if the environment
changes.

---

## Tuning Parameters

| Parameter | Effect |
|-----------|--------|
| `resolution` | Smaller → finer detail, more memory and computation. |
| `log_odds_hit` | Larger → cells become occupied faster. |
| `log_odds_miss` | More negative → cells become free faster. |
| `log_odds_min/max` | Wider range → slower to recover from mis-classifications; narrower → adapts faster to change. |
| `z_min / z_max` | Set to the height of the obstacles of interest to suppress irrelevant returns. |
