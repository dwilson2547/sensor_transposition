# Sliding-Window / Fixed-Lag Smoother

## Overview

`sensor_transposition.sliding_window` provides a **fixed-lag sliding-window
smoother** for online SLAM — a bounded-cost alternative to full batch pose
graph optimisation that is suitable for long-duration autonomous navigation.

| Class | Description |
|-------|-------------|
| `SlidingWindowSmoother` | Online fixed-lag smoother with world-frame prior factors |

---

## Motivation

The batch `optimize_pose_graph` function processes the entire trajectory at
every step, with cost that grows as O(*n*³) in the number of keyframes.
On a long run this quickly becomes intractable.

A **sliding-window smoother** caps the cost by keeping only the *w* most
recent keyframes in the active window:

```
time →  0   1   2   3   4   5   6   7 ...
             [──── window (w=4) ────]
                              ↑ new frame added → oldest evicted
```

When a keyframe *leaves* the window it is **marginalised**: any information
it contributed about its still-active neighbours is converted into a
*world-frame prior factor* that is carried forward in future optimisation
steps.  The per-step cost is then O(*w*³), independent of total trajectory
length.

---

## Algorithm

### Active window

At each step the smoother maintains a set of at most `window_size` active
nodes.  All relative-pose edges (from odometry or loop-closure detection)
between active nodes are included in the optimisation.

### Marginalisation and prior factors

When the oldest node `m` exits the window:

1. Its most recent optimised world-frame pose **T_m** is recorded.
2. For every edge `m → a` (where `a` is still active) with measured relative
   transform **T̂_{ma}** and information **Ω**:
   - A *prior factor* is created: node `a` should be at world-frame pose
     **T_m · T̂_{ma}** with information **Ω**.
3. For every edge `a → m` with measured relative transform **T̂_{am}**:
   - A prior factor is created: node `a` should be at **T_m · T̂_{am}⁻¹**.
4. The node, its edges, and any priors previously accumulated on it are
   removed.

Prior factors are represented as edges from a fixed **world anchor node**
(ID `−1`, placed at the world origin) to the constrained active nodes.
The anchor is always inserted as the *first* node in the pose graph, so
`optimize_pose_graph` holds it fixed throughout the optimisation.

### Optimisation

Each call to `optimize()`:

1. Builds a `PoseGraph` with the active nodes, their mutual edges, and all
   outstanding prior factors.
2. Calls `optimize_pose_graph` on this reduced graph.
3. Updates the stored node poses with the optimised values (so that
   subsequent marginalisation steps use the best available estimate).
4. Returns an `OptimizationResult` containing **only the active nodes**
   (the anchor node is excluded).

---

## Usage

### Incremental keyframe insertion

```python
import numpy as np
from sensor_transposition.sliding_window import SlidingWindowSmoother

smoother = SlidingWindowSmoother(window_size=5)

# Relative-pose measurement: 1 m step forward in x.
T_step = np.eye(4)
T_step[0, 3] = 1.0
info = np.eye(6) * 200.0   # high confidence in the step

for i in range(20):
    # Add new keyframe with dead-reckoning initial estimate.
    smoother.add_node(i, translation=[float(i), 0.0, 0.0])

    # Add odometry edge to the previous node (after the first).
    if i > 0:
        smoother.add_edge(i - 1, i, transform=T_step, information=info)

    # Optimise the active window and retrieve the latest poses.
    result = smoother.optimize()
    if result.success:
        pos = result.optimized_poses[i]["translation"]
        print(f"node {i:2d}  x={pos[0]:.3f}  y={pos[1]:.3f}")
```

### Accessing the active window

```python
print("Active nodes:", smoother.active_node_ids)
# e.g. Active nodes: [15, 16, 17, 18, 19]
```

### Using a 4×4 transform as the initial pose

```python
from sensor_transposition.transform import Transform

T_world = np.eye(4)
T_world[0, 3] = 5.0   # 5 m in x

smoother.add_node(42, transform=T_world)
```

### Integrating with scan matching

```python
from sensor_transposition.lidar.scan_matching import icp_align

result_icp = icp_align(prev_cloud, curr_cloud)
smoother.add_edge(
    prev_id,
    curr_id,
    transform=result_icp.transform,
    information=np.eye(6) * (1.0 / result_icp.mean_squared_error + 1e-9),
)
```

### Integrating with loop-closure detection

```python
from sensor_transposition.loop_closure import ScanContextDatabase

db = ScanContextDatabase()
# ... populate the database ...

match_id, score = db.query(curr_descriptor, exclude_recent=10)
if score < 0.2:
    loop_tf = icp_align(clouds[match_id], curr_cloud).transform
    smoother.add_edge(
        match_id,
        curr_id,
        transform=loop_tf,
        information=np.eye(6) * 50.0,
    )
```

### Recovering the full trajectory for very long runs

For arbitrarily long trajectories the active window contains only the most
recent `window_size` keyframes.  Access the complete history via
`marginalised_poses` or `full_trajectory()`:

```python
smoother = SlidingWindowSmoother(window_size=10)

# ... run for thousands of keyframes ...

# Inspect all poses that have been evicted from the window.
for node_id, pose in smoother.marginalised_poses.items():
    print(node_id, pose["translation"])

# Get the entire trajectory in one call (marginalised + active nodes).
full_traj = smoother.full_trajectory()
for node_id in sorted(full_traj):
    t = full_traj[node_id]["translation"]
    print(f"node {node_id:5d}  ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})")
```

The `full_trajectory()` dict can be fed directly into `PointCloudMap`,
`FramePoseSequence`, or any other component that accepts world-frame poses.

---

## API Reference

### `SlidingWindowSmoother`

```python
SlidingWindowSmoother(
    window_size: int = 10,
    max_iterations: int = 20,
    tolerance: float = 1e-6,
    damping: float = 1e-6,
)
```

| Parameter | Description |
|-----------|-------------|
| `window_size` | Maximum number of simultaneously active keyframe nodes. |
| `max_iterations` | Maximum Gauss-Newton iterations per `optimize()` call. |
| `tolerance` | Convergence threshold on the update step norm `‖Δx‖`. |
| `damping` | Levenberg-Marquardt diagonal damping for numerical robustness. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `window_size` | `int` | Maximum number of active nodes. |
| `active_node_ids` | `list[int]` | Ordered list of currently active node IDs (oldest first). |
| `latest_result` | `OptimizationResult \| None` | Result from the most recent `optimize()` call. |
| `marginalised_poses` | `dict[int, dict]` | World-frame poses of all nodes evicted from the window (historical record). |

#### `add_node(node_id, *, translation=None, quaternion=None, transform=None)`

Add a new keyframe.  If adding the node would overflow the window, the
oldest active node is automatically marginalised first.

Node ID `−1` is reserved for the internal world anchor and must not be used
as a keyframe identifier.

#### `add_edge(from_id, to_id, *, transform, information=None)`

Add a relative-pose constraint.  Edges may reference nodes that have already
been marginalised; they are silently ignored in `optimize()` if either
endpoint is no longer active.

#### `optimize() → OptimizationResult`

Optimise the active window.  Returns an
[`OptimizationResult`](./pose_graph_optimisation.md) with poses for the
active nodes only.

#### `marginalised_poses → dict[int, dict]`

Read-only property.  Maps every node ID that has been evicted from the window
to a dict with keys `"translation"`, `"quaternion"`, and `"transform"`,
preserving each node's last optimised world-frame pose at the time of
eviction.  The dict grows monotonically as nodes are marginalised and is
never modified retroactively.

Useful for building a complete map or trajectory log without keeping all
nodes in memory at once.

#### `full_trajectory() → dict[int, dict]`

Returns the complete trajectory across the entire run: the union of
:attr:`marginalised_poses` (historical nodes) and the current active-window
poses (updated after the latest `optimize()` call).  Each entry has the same
`"translation"` / `"quaternion"` / `"transform"` keys as `marginalised_poses`
and `OptimizationResult`.

---

## OptimizationResult fields

Returned by `optimize()`:

| Field | Type | Description |
|-------|------|-------------|
| `optimized_poses` | `dict[int, dict]` | Maps node ID to `{"translation": list, "quaternion": list, "transform": ndarray}`. |
| `final_cost` | `float` | Weighted squared error at convergence. |
| `iterations` | `int` | Number of Gauss-Newton iterations performed. |
| `success` | `bool` | `True` if the solver converged or the window has ≤ 1 node. |

---

## Integration in the SLAM Pipeline

```
                   ┌─────────────────────────────────────────┐
  Sensor stream    │          SlidingWindowSmoother           │
  ─────────────►   │                                          │
  (LiDAR scan)     │  add_node(i, translation=dead_reckoning) │
                   │  add_edge(i-1, i, transform=icp_result)  │
  (loop closure)   │  add_edge(match, i, transform=loop_tf)   │
                   │  result = optimize()                     │
                   └──────────────┬──────────────────────────┘
                                  │ optimized_poses
                                  ▼
                   ┌──────────────────────────┐
                   │  PointCloudMap.add_scan  │
                   │  GpsFuser.update         │
                   │  FramePoseSequence       │
                   └──────────────────────────┘
```

### Recommended update rate

Call `optimize()` after every keyframe addition.  The sliding window bounds
the per-step optimisation cost to O(`window_size`³), making real-time
operation feasible for typical window sizes (5–20 nodes).

### Choosing `window_size`

| Scenario | Suggested `window_size` |
|----------|------------------------|
| Indoor robot (short corridors) | 5–10 |
| Outdoor vehicle (long runs) | 10–20 |
| High-rate LiDAR (10 Hz+) | 5–8 (keep cost low) |
| Infrequent loop closures | 15–20 (more context) |

---

## Limitations

- **Approximate marginalisation**: the prior factors are derived by composing
  the marginalised node's pose with the incident edge measurements.  In the
  nonlinear case (large rotations between consecutive keyframes) this is an
  approximation of the theoretically exact Schur-complement factor.  For
  automotive or robotic platforms with modest inter-frame rotations the
  approximation is accurate.

- **No loop closure across the window boundary**: a loop-closure edge whose
  older endpoint has already been marginalised will be ignored by `optimize()`
  (both endpoints must be active).  To detect and handle inter-window loops,
  use `ScanContextDatabase` to find the match and add the edge *before* the
  older node is evicted, or supplement with a separate full-batch
  re-localisation step.

- **Pure NumPy/SciPy**: no external SLAM library is required, but performance
  is limited compared to native-code solvers such as g2o, GTSAM, or Ceres.
  For production deployments consider wrapping those libraries.