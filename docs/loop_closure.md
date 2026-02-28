# Loop Closure Detection

Loop closure detection is the process of recognising when a mobile sensor platform revisits a previously observed location.  When a closure is confirmed the trajectory estimate can be corrected and the accumulated drift from odometry eliminated.

`sensor_transposition` provides two complementary **appearance-based LiDAR descriptors** plus a **database and query engine** that together form the front-end of a complete loop-closure pipeline.

---

## Descriptors

### Scan Context (`compute_scan_context`)

Scan Context [Kim & Kim, IROS 2018] is a rotation-aware descriptor optimised for ground-vehicle LiDAR.

**Algorithm:**
1. The horizontal plane around the sensor is divided into `num_rings` concentric radial annuli and `num_sectors` angular sectors, forming a `(num_rings × num_sectors)` polar grid.
2. Each cell stores the **maximum z-height** of the LiDAR points that fall into it.  Empty cells are set to zero.
3. A compact **ring key** — the row-wise mean of the descriptor matrix — provides a fast `O(num_rings)` pre-filter for nearest-neighbour search.
4. **Rotation invariance** is achieved at query time by testing all column shifts and returning the shift that minimises the normalised cosine distance.

**Typical parameters:**

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_rings` | 20 | More rings → finer radial resolution |
| `num_sectors` | 60 | More sectors → finer angular resolution |
| `max_range` | 80.0 m | Points beyond this range are ignored |
| `min_z` / `max_z` | `None` | Optional height-clipping to suppress ground or ceiling |

```python
from sensor_transposition.loop_closure import compute_scan_context, scan_context_distance

desc_a = compute_scan_context(points_a, num_rings=20, num_sectors=60, max_range=80.0)
desc_b = compute_scan_context(points_b, num_rings=20, num_sectors=60, max_range=80.0)

distance, yaw_shift = scan_context_distance(desc_a, desc_b)
print(f"Distance: {distance:.4f}  (yaw offset ≈ {yaw_shift} sectors)")
```

---

### M2DP (`compute_m2dp`)

M2DP (Multi-view 2D Projection) [He et al., IROS 2016] is a viewpoint-insensitive descriptor that does not rely on a well-defined ground plane and therefore complements Scan Context well, e.g. for aerial LiDAR, multi-floor environments, or sensors mounted at unusual roll/pitch angles.

**Algorithm:**
1. The point cloud is centred at its centroid.
2. For each of `num_elevation × num_azimuth` oriented planes (sampled uniformly in elevation and azimuth):
   a. Every point is projected orthogonally onto the plane.
   b. The 2-D projected positions are converted to polar coordinates and binned into a `(num_rings × num_sectors)` point-density histogram.
   c. The flattened histogram forms one row of the `(L, P)` signature matrix, where `L = num_elevation × num_azimuth` and `P = num_rings × num_sectors`.
3. A thin SVD is computed on the signature matrix.  The descriptor vector is the concatenation of the **first left singular vector** (length `L`) and the **first right singular vector** (length `P`).

The resulting descriptor has length `L + P = num_elevation × num_azimuth + num_rings × num_sectors`.

**Typical parameters:**

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_azimuth` | 4 | Azimuthal projection directions |
| `num_elevation` | 16 | Elevation projection directions |
| `num_rings` | 4 | Radial bins in each 2-D projection |
| `num_sectors` | 16 | Angular bins in each 2-D projection |

```python
from sensor_transposition.loop_closure import compute_m2dp, m2dp_distance

desc_a = compute_m2dp(points_a, num_azimuth=4, num_elevation=16,
                      num_rings=4, num_sectors=16)
desc_b = compute_m2dp(points_b, num_azimuth=4, num_elevation=16,
                      num_rings=4, num_sectors=16)

dist = m2dp_distance(desc_a, desc_b)
print(f"M2DP distance: {dist:.4f}")
```

---

## Database and Query (`ScanContextDatabase`)

`ScanContextDatabase` maintains an incremental collection of Scan Context descriptors and exposes an efficient two-stage nearest-neighbour search.

**Two-stage search:**
1. **Ring key pre-filter** — L1 distance between ring keys narrows the candidate set to `candidate_pool_size` entries in `O(n × num_rings)` time.
2. **Full descriptor distance** — the rotation-invariant Scan Context distance (with exhaustive column-shift search) is computed only for the candidates, returning the `top_k` closest matches.

An **exclusion window** suppresses matches against recently added frames to avoid spurious self-matches along the current trajectory.

```python
from sensor_transposition.loop_closure import (
    ScanContextDatabase,
    compute_scan_context,
)

db = ScanContextDatabase(
    num_rings=20,
    num_sectors=60,
    max_range=80.0,
    exclusion_window=50,    # skip the 50 most recent frames
    candidate_pool_size=25, # ring-key shortlist size
)

for frame_id, cloud in enumerate(lidar_frames):
    desc = compute_scan_context(cloud, num_rings=20, num_sectors=60, max_range=80.0)
    candidates = db.query(desc, top_k=1)
    db.add(desc, frame_id=frame_id)

    if candidates and candidates[0].distance < 0.15:
        loop_frame = candidates[0].match_frame_id
        yaw_offset = candidates[0].yaw_shift_sectors
        print(f"Loop closure: frame {frame_id} ↔ frame {loop_frame} "
              f"(dist={candidates[0].distance:.3f}, yaw_shift={yaw_offset})")
```

---

## Full Pipeline Integration

After a loop candidate is found, pass the two point clouds to
`sensor_transposition.lidar.scan_matching.icp_align` for geometric verification
and to obtain the relative transform for the pose-graph edge:

```python
from sensor_transposition.lidar.scan_matching import icp_align
from sensor_transposition.pose_graph import PoseGraph, PoseGraphEdge

pg = PoseGraph()

# ... build nodes via odometry ...

if candidates and candidates[0].distance < 0.15:
    result = icp_align(cloud_query, cloud_match, max_iterations=50)
    if result.converged:
        pg.add_edge(PoseGraphEdge(
            from_id=loop_frame,
            to_id=current_frame,
            transform=result.transform,
        ))
```

See `docs/pose_graph_optimisation.md` for details on running the back-end
optimiser once the graph is built.

---

## Choosing Between Scan Context and M2DP

| Criterion | Scan Context | M2DP |
|-----------|-------------|------|
| Designed for | Ground-vehicle LiDAR | General 3-D LiDAR (any orientation) |
| Rotation invariance | Column-shift search over yaw | Implicit in multi-view projection |
| Descriptor length | `num_rings × num_sectors` matrix + ring key | `L + P` vector (compact) |
| Speed | Fast ring-key pre-filter in `ScanContextDatabase` | No built-in database; use cosine distance directly |
| Sensitivity to ground plane | Encodes height information explicitly | Invariant to up-direction |

For typical outdoor ground-vehicle SLAM, **Scan Context** combined with
`ScanContextDatabase` is the recommended choice.  **M2DP** is best suited as a
complementary or standalone descriptor for environments where the ground-plane
assumption does not hold or a single compact descriptor is needed for downstream
machine-learning pipelines.
