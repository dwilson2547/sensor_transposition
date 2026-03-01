# Radar Odometry

This document describes the radar odometry utilities in
`sensor_transposition.radar.radar_odometry`.  Two complementary approaches
are provided for estimating ego-motion from radar data:

1. **Doppler-based ego-velocity estimation** – recovers the sensor's 3-D
   velocity vector from radial Doppler measurements in a single frame.
2. **ICP scan matching** – aligns consecutive radar Cartesian point clouds to
   estimate the 6-DOF relative pose between frames.

---

## Why Radar for Odometry?

Radar is uniquely suited to adverse conditions that degrade camera and LiDAR
performance:

* **All-weather** – penetrates fog, rain, snow, and dust.
* **Long range** – typical automotive radars detect objects at 50–250 m.
* **Doppler velocity** – FMCW radars measure the radial velocity of each
  detection directly, providing an ego-motion signal that requires no
  frame-to-frame matching.

Radar odometry can therefore serve as a reliable fallback or complementary
source when camera or LiDAR-based estimates are unreliable.

---

## Doppler Ego-Velocity Estimation

### Physical Model

For a stationary target, the measured radial (Doppler) velocity equals the
projection of the *negative* sensor velocity onto the unit direction vector
from the sensor to the target:

```
velocity_i = -(d_i · v_ego)
```

where:

* `d_i = [cos(el_i)·cos(az_i), cos(el_i)·sin(az_i), sin(el_i)]` is the unit
  direction vector to detection *i*.
* `v_ego` is the 3-D sensor velocity in m/s in the sensor frame
  (`x` = forward, `y` = left, `z` = up).
* `velocity_i` is the signed Doppler reading (negative = approaching).

With *N* detections this becomes the over-determined linear system:

```
D @ v_ego = -velocity
```

solved in the least-squares sense to recover `v_ego`.

### API

```python
from sensor_transposition.radar import RadarParser
from sensor_transposition.radar.radar_odometry import estimate_ego_velocity

detections = RadarParser("frame_0001.bin").read()
result = estimate_ego_velocity(detections, min_snr=10.0)

if result.valid:
    vx, vy, vz = result.velocity        # m/s in sensor frame
    print(f"Ego velocity: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f} m/s")
    print(f"Used {result.num_inliers} detections")
    print(f"Max residual: {abs(result.residuals).max():.4f} m/s")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_snr` | `0.0` dB | Minimum SNR to include a detection. |
| `min_inliers` | `3` | Minimum detections required for a valid estimate. |

### Return Value (`EgoVelocityResult`)

| Field | Type | Description |
|-------|------|-------------|
| `velocity` | `ndarray (3,)` | Estimated ego-velocity `[vx, vy, vz]` in m/s. |
| `residuals` | `ndarray (N,)` | Per-detection fitting residuals in m/s. |
| `num_inliers` | `int` | Number of detections used. |
| `valid` | `bool` | `False` if too few inliers to solve. |

### Fusion with EKF

The estimated velocity integrates directly with the `ImuEkf` via its
`velocity_update` method:

```python
from sensor_transposition.imu.ekf import ImuEkf
from sensor_transposition.radar.radar_odometry import estimate_ego_velocity

ekf = ImuEkf()
# ... populate ekf with IMU data ...

result = estimate_ego_velocity(detections, min_snr=12.0)
if result.valid:
    # velocity_update expects the 3-D velocity in the world frame;
    # transform from sensor frame to world frame first.
    v_world = ekf.state.rotation @ result.velocity
    ekf.velocity_update(v_world, noise_std=0.1)
```

---

## Scan-to-Scan ICP Matching

### Algorithm

Consecutive radar Cartesian point clouds (obtained via
[`RadarParser.xyz()`](../sensor_transposition/radar/radar.py)) are aligned
with point-to-point ICP.  The implementation reuses
[`lidar.scan_matching.icp_align`](lidar_motion_distortion.md) internally:

1. Build a KD-tree from the previous (target) point cloud.
2. Find the closest target point for each source point.
3. Reject correspondences beyond `max_correspondence_dist`.
4. Solve for the optimal rigid transform with the Kabsch SVD algorithm.
5. Apply the incremental transform and repeat until convergence.

### RadarOdometer

`RadarOdometer` is a stateful class that maintains a running world-frame pose
by composing per-frame-pair transforms:

```python
from sensor_transposition.radar import RadarParser
from sensor_transposition.radar.radar_odometry import RadarOdometer

odom = RadarOdometer(max_correspondence_dist=8.0, max_iterations=50)

for path, timestamp in zip(sorted(bin_files), timestamps):
    xyz = RadarParser(path).xyz()
    result = odom.add_frame(xyz, timestamp)
    if result is not None:
        status = "converged" if result.converged else "not converged"
        print(f"t={timestamp:.2f}s  MSE={result.mean_squared_error:.4f}  {status}")

print("Final world pose:\n", odom.pose)
```

### Functional API

`integrate_radar_odometry` processes a complete list of frames in one call:

```python
from sensor_transposition.radar.radar_odometry import integrate_radar_odometry

frames_xyz = [RadarParser(p).xyz() for p in sorted(bin_files)]
timestamps = [i * 0.1 for i in range(len(frames_xyz))]

pose, icp_results = integrate_radar_odometry(
    frames_xyz,
    timestamps,
    max_correspondence_dist=8.0,
    max_iterations=50,
)
print("Accumulated pose:\n", pose)
```

### API Reference

#### `RadarOdometer`

| Method / Property | Description |
|-------------------|-------------|
| `add_frame(xyz, timestamp)` | Add a frame; returns `IcpResult` or `None` (first frame). |
| `reset()` | Reset pose to identity and clear stored frames. |
| `pose` | Current world-frame 4×4 pose matrix. |
| `transforms` | List of per-pair relative 4×4 transforms. |

Constructor parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | `50` | ICP iteration limit per frame pair. |
| `tolerance` | `1e-6` | ICP convergence threshold (MSE change). |
| `max_correspondence_dist` | `∞` | Max point-pair distance in metres. |

#### `radar_scan_match`

A named wrapper around `icp_align` for use in radar odometry pipelines:

```python
from sensor_transposition.radar.radar_odometry import radar_scan_match

result = radar_scan_match(
    source_xyz,
    target_xyz,
    max_correspondence_dist=5.0,
)
print(result.transform)   # 4×4 rigid transform
print(result.converged)   # True/False
```

---

## Integration with the SLAM Pipeline

| Stage | Integration point |
|-------|-------------------|
| **Odometry front-end** | Use `RadarOdometer` or `estimate_ego_velocity` to provide a motion prior between LiDAR keyframes. |
| **EKF state estimator** | Feed `result.velocity` from `estimate_ego_velocity` into `ImuEkf.velocity_update`. |
| **Pose graph** | Use `RadarOdometer.transforms` as pose-graph edge constraints alongside LiDAR ICP edges. |
| **Adverse weather** | Swap in radar odometry when LiDAR or camera degrades (fog, rain, direct sunlight). |

---

## Tips and Limitations

* **SNR filtering** – always set `min_snr` to a value appropriate for your
  sensor (commonly 10–15 dB) to exclude ghost detections.
* **Min inliers** – increase `min_inliers` beyond 3 for more robust
  velocity estimates.  Six or more spatially diverse detections are
  recommended.
* **Sparse clouds** – radar point clouds are sparser than LiDAR.  Reduce
  `max_correspondence_dist` conservatively (e.g. 5–10 m) to avoid false
  correspondences.
* **Dynamic objects** – moving targets violate the stationary-target
  assumption of the Doppler model.  Consider RANSAC-based outlier rejection
  for environments with many dynamic objects.
* **2-D vs 3-D** – many automotive radars produce detections in a 2-D
  horizontal plane (elevation ≈ 0).  In that case, set all elevation values
  to zero and the velocity estimate will still recover `vx` and `vy`
  correctly.
