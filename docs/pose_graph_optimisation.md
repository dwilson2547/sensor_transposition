# Pose Graph Optimisation

This guide explains the pose graph data structure and Gauss-Newton optimisation
back-end provided by `sensor_transposition.pose_graph`.  A pose graph is the
standard *back-end* of a SLAM pipeline: it takes all relative-pose constraints
(from odometry and loop closures) and finds the globally consistent set of
absolute poses that best satisfies them all simultaneously.

---

## Background

Front-end components (LiDAR odometry, visual odometry, loop-closure detection)
each produce *local*, *pairwise* pose constraints.  However, small errors in
each constraint accumulate over time, causing the trajectory to drift.  When a
loop closure is detected the drift is resolved by *redistributing* the error
across all poses in the graph — which is exactly what the pose graph optimiser
does.

The total cost the optimiser minimises is:

```
F(x) = ½ Σ  eᵢⱼ(x)ᵀ Ωᵢⱼ eᵢⱼ(x)
```

where the sum runs over all edges, `eᵢⱼ` is the **6-D pose error** for the
edge, and `Ωᵢⱼ` is its **6×6 information matrix** (inverse covariance).

### Error formulation

For an edge from node `i` to node `j` with measured relative transform `T̂ᵢⱼ`
and current absolute poses `Tᵢ`, `Tⱼ`:

```
T_err = T̂ᵢⱼ⁻¹ · Tᵢ⁻¹ · Tⱼ

eᵢⱼ = [ T_err[:3, 3]        ]   ← translation error  (3-D)
       [ log(T_err[:3, :3])  ]   ← rotation error     (rotation vector, 3-D)
```

### Algorithm

The optimiser runs **Gauss-Newton** iterations, fixing the first node to remove
the gauge freedom of the global reference frame:

```
1. For each edge, compute eᵢⱼ and numerical Jacobians Jᵢ, Jⱼ.
2. Build the reduced Hessian and gradient over free nodes.
3. Solve H · Δx = −b  (with LM damping for robustness).
4. Apply the update to all free nodes.
5. Repeat until ‖Δx‖ < tolerance or max_iterations reached.
```

Jacobians are computed by **central finite differences** (step 10⁻⁷), which
requires no manual derivation and is numerically stable for this problem size.

---

## API

### `PoseGraph`

```python
from sensor_transposition.pose_graph import PoseGraph

graph = PoseGraph()
```

#### `add_node`

```python
graph.add_node(
    node_id,               # int — unique keyframe identifier
    translation=[x, y, z], # initial position in world frame (metres)
    quaternion=[w, x, y, z],  # initial orientation (unit quaternion)
)
# OR pass a 4×4 SE(3) transform directly:
graph.add_node(node_id, transform=T)
```

Adds a keyframe node.  Each `node_id` must be unique.  Returns a
`PoseGraphNode`.

#### `add_edge`

```python
graph.add_edge(
    from_id,               # int — source node ID
    to_id,                 # int — target node ID
    transform=T_relative,  # 4×4 SE(3): from_frame → to_frame
    information=Omega,     # 6×6 information matrix (optional, default I)
)
```

Adds a relative-pose constraint (edge).  Returns a `PoseGraphEdge`.

The column/row ordering of the information matrix is
`[tx, ty, tz, rx, ry, rz]` — translation first, then rotation.

#### Properties

| Property | Description |
|----------|-------------|
| `nodes`  | `dict[int, PoseGraphNode]` — copy of all nodes |
| `edges`  | `list[PoseGraphEdge]` — copy of all edges |
| `len(g)` | Number of nodes |

---

### `optimize_pose_graph`

```python
from sensor_transposition.pose_graph import optimize_pose_graph

result = optimize_pose_graph(
    graph,
    max_iterations=20,   # Gauss-Newton iteration limit (default 20)
    tolerance=1e-6,      # convergence threshold on ‖Δx‖ (default 1e-6)
    damping=1e-6,        # LM diagonal damping (default 1e-6)
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | `20` | Maximum Gauss-Newton iterations |
| `tolerance` | `1e-6` | Step-norm convergence threshold |
| `damping` | `1e-6` | Levenberg-Marquardt diagonal damping |

Returns an `OptimizationResult`:

| Field | Type | Description |
|-------|------|-------------|
| `optimized_poses` | `dict[int, dict]` | Per-node optimised pose (see below) |
| `final_cost` | `float` | Total weighted squared error at convergence |
| `iterations` | `int` | Number of iterations performed |
| `success` | `bool` | `True` if step norm fell below `tolerance` |

Each value in `optimized_poses` is a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `"translation"` | `list[float]` | `[x, y, z]` in world frame (metres) |
| `"quaternion"` | `list[float]` | `[w, x, y, z]` unit quaternion |
| `"transform"` | `np.ndarray` | 4×4 SE(3) numpy array |

---

## Usage Examples

### LiDAR-odometry pose graph

```python
from sensor_transposition.pose_graph import PoseGraph, optimize_pose_graph
from sensor_transposition.lidar.scan_matching import icp_align
import numpy as np

graph = PoseGraph()

# Initialise from identity.
graph.add_node(0)

prev_cloud = lidar_frames[0]
current_pose = np.eye(4)

for i, cloud in enumerate(lidar_frames[1:], start=1):
    # Estimate incremental motion via ICP.
    result = icp_align(prev_cloud, cloud)
    T_inc = result.transform if result.converged else np.eye(4)

    # Compose to get the new absolute pose (dead-reckoning).
    current_pose = current_pose @ T_inc

    # Add node with the dead-reckoned pose.
    graph.add_node(i, transform=current_pose)

    # Add odometry edge (high information = low uncertainty).
    graph.add_edge(
        from_id=i - 1, to_id=i,
        transform=T_inc,
        information=np.eye(6) * 100.0,
    )

    prev_cloud = cloud

# Optimise.
opt = optimize_pose_graph(graph, max_iterations=30)
```

### Adding a loop-closure edge

After a loop closure is detected (e.g. via `ScanContextDatabase.query`) and
verified geometrically (`icp_align`):

```python
from sensor_transposition.loop_closure import ScanContextDatabase, compute_scan_context

db = ScanContextDatabase(num_rings=20, num_sectors=60, max_range=80.0,
                         exclusion_window=50)

for i, cloud in enumerate(lidar_frames):
    desc = compute_scan_context(cloud, num_rings=20, num_sectors=60, max_range=80.0)
    candidates = db.query(desc, top_k=1)
    db.add(desc, frame_id=i)

    if candidates and candidates[0].distance < 0.15:
        j = candidates[0].match_frame_id
        # Verify with ICP.
        verification = icp_align(lidar_frames[j], cloud)
        if verification.converged:
            graph.add_edge(
                from_id=j, to_id=i,
                transform=verification.transform,
                information=np.eye(6) * 50.0,  # lower weight than odometry
            )

# Optimise after all loop closures have been added.
opt = optimize_pose_graph(graph, max_iterations=50)
```

### Updating `FramePoseSequence` from optimisation result

```python
from sensor_transposition.frame_pose import FramePose, FramePoseSequence

sequence = FramePoseSequence(frame_duration=0.1)
for node_id in sorted(opt.optimized_poses):
    pose = opt.optimized_poses[node_id]
    sequence.add_pose(FramePose(
        timestamp=timestamps[node_id],
        translation=pose["translation"],
        rotation=pose["quaternion"],
    ))
```

---

## Information Matrix Tuning

The information matrix `Ω` is the inverse covariance of the relative-pose
constraint.  It should reflect your confidence in the measurement:

| Edge type | Suggested information | Rationale |
|-----------|----------------------|-----------|
| LiDAR ICP (short range) | `np.eye(6) * 100` – `1000` | High-quality scan matching |
| Visual odometry | `np.eye(6) * 50` – `500` | Depends on feature quality |
| Loop closure (ICP verified) | `np.eye(6) * 20` – `100` | Lower weight: larger uncertainty |
| GPS anchor | `diag([1/σ², 1/σ², 1/σ², ∞, ∞, ∞])` | Position only; no orientation |

Use anisotropic matrices to weight translation and rotation components
differently when the uncertainties are unequal.

---

## Integration with the SLAM Pipeline

```
ImuPreintegrator      →  high-rate relative-pose prediction
     ↓
  ImuEkf.predict       →  continuous-time state propagation
     ↓
icp_align / VO        →  odometry edges  ──────────────────────┐
     ↓                                                          │
ScanContextDatabase   →  loop-closure candidates               │
     ↓                                                          │
icp_align (verify)    →  loop-closure edges ──────────────────▶│
                                                                │
                                                     PoseGraph  │
                                                         ↓      │
                                           optimize_pose_graph ◀┘
                                                         ↓
                                             FramePoseSequence
                                         (globally consistent trajectory)
```

The optimised `FramePoseSequence` can then be used to accumulate a consistent
point-cloud map (see *Accumulated point-cloud map* in `TODO.md`).
