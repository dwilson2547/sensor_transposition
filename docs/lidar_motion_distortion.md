# LiDAR Motion-Distortion Correction (Deskewing)

This guide explains how to use `sensor_transposition.lidar.motion_distortion.deskew_scan`
to remove motion-induced distortion from LiDAR point clouds using IMU measurements.

---

## Background

A rotating LiDAR (e.g. Velodyne VLP-16, Ouster OS1, Livox Mid-360) collects
points continuously as its laser(s) sweep through a full 360° rotation.  For a
10 Hz LiDAR, one complete "frame" spans **100 ms**.  During that interval a
moving vehicle may travel several metres and rotate by several degrees, so
points at the start and end of the scan were acquired from **different sensor
poses**.

The result is a *distorted* point cloud: straight walls appear curved, flat
ground appears wavy, and objects appear stretched or duplicated.  This
distortion is visually small at low speeds but causes measurable degradation in
scan-matching accuracy even at pedestrian speeds (1–3 m/s).

**Deskewing** (also called *motion-distortion correction*) undoes this effect
by transforming every point from its true acquisition pose to a single,
consistent *reference pose* — typically the pose at the start or end of the
scan.

---

## How It Works

The correction uses the IMU to track the sensor's changing pose during the
scan window.

### Step 1 — IMU trajectory integration

Starting from the first IMU sample with an identity pose (and an optional
initial velocity), the gyroscope and accelerometer are integrated using the
**midpoint (trapezoidal) method** — the same scheme used in
`sensor_transposition.imu.preintegration`:

```
For consecutive samples k and k+1 with interval dt:

    a_mid = 0.5 × (a_k + a_{k+1}) − accel_bias
    ω_mid = 0.5 × (ω_k + ω_{k+1}) − gyro_bias

    R_{k+1} = R_k × Exp(ω_mid × dt)                  # SO(3) Rodrigues
    v_{k+1} = v_k + (R_k × a_mid + g) × dt
    p_{k+1} = p_k + v_k × dt + ½(R_k × a_mid + g) × dt²
```

This produces a discrete trajectory of **(R, p)** pairs at each IMU sample
time, with the world frame anchored to the sensor body at the first IMU
sample.

### Step 2 — Continuous pose interpolation

Poses at arbitrary times within the scan window are obtained by:

- **SLERP** (Spherical Linear Interpolation, via `scipy.spatial.transform.Slerp`)
  for the rotation component.
- **Linear interpolation** (`numpy.interp`) for the position component.

### Step 3 — Relative-transform computation

The relative transform from the reference time **t_ref** to each point's
acquisition time **t_i** is:

```
R_rel = R(t_ref)ᵀ × R(t_i)
t_rel = R(t_ref)ᵀ × (p(t_i) − p(t_ref))
```

This transform describes *how much the sensor body moved* between the
reference pose and the point's acquisition pose.

### Step 4 — Point correction

Each point is expressed in the body frame at `t_ref` by applying the
relative transform:

```
p_corrected = R_rel × p_i + t_rel
```

---

## API

### `deskew_scan`

```python
from sensor_transposition.lidar.motion_distortion import deskew_scan
```

```python
corrected = deskew_scan(
    points,
    point_times,
    imu_times,
    imu_accel,
    imu_gyro,
    ref_time,
    *,
    accel_bias=None,
    gyro_bias=None,
    gravity=None,
    initial_velocity=None,
)
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `points` | `(N, 3)` | LiDAR points in the sensor body frame (m) |
| `point_times` | `(N,)` | Per-point acquisition timestamps (s) |
| `imu_times` | `(K,)` | IMU sample timestamps — strictly increasing (s) |
| `imu_accel` | `(K, 3)` | Accelerometer measurements in body frame (m/s²) |
| `imu_gyro` | `(K, 3)` | Gyroscope measurements in body frame (rad/s) |
| `ref_time` | scalar | Timestamp to deskew all points to (s) |
| `accel_bias` | `(3,)` | Accelerometer bias; default `[0, 0, 0]` |
| `gyro_bias` | `(3,)` | Gyroscope bias; default `[0, 0, 0]` |
| `gravity` | `(3,)` | Gravity vector in world frame; default `[0, 0, −9.81]` |
| `initial_velocity` | `(3,)` | Body velocity at `imu_times[0]` (m/s); default `[0, 0, 0]` |

**Returns:** `(N, 3)` float64 array of deskewed points in the sensor body
frame at `ref_time`.

> **Tip:** Point timestamps outside the IMU window are silently **clamped** to
> the nearest boundary — those points receive no correction (identity
> transform), which is appropriate when a few points arrive slightly early or
> late.

---

## Usage Examples

### Minimal example — rotation-only correction

When only the scan start / end timestamp and IMU gyroscope data are available
(no velocity estimate), rotation-only deskewing is obtained by setting
`gravity=np.zeros(3)`:

```python
import numpy as np
from sensor_transposition.lidar.motion_distortion import deskew_scan

# --- Simulated inputs ---
# LiDAR frame: 1000 points over 100 ms at 10 Hz
point_times = np.linspace(t_scan_start, t_scan_start + 0.1, 1000)

# IMU at 200 Hz spanning the scan window (plus a small margin)
imu_slice_mask = (imu_timestamps >= t_scan_start - 0.01) & \
                 (imu_timestamps <= t_scan_start + 0.11)
imu_t = imu_timestamps[imu_slice_mask]
imu_a = imu_accel[imu_slice_mask]
imu_g = imu_gyro[imu_slice_mask]

# Deskew to scan start (rotation-only; gravity disabled for cleanliness
# when initial velocity is unknown)
deskewed = deskew_scan(
    points=lidar_xyz,
    point_times=point_times,
    imu_times=imu_t,
    imu_accel=imu_a,
    imu_gyro=imu_g,
    ref_time=t_scan_start,
    gravity=np.zeros(3),
)
```

### Full correction with EKF velocity

For best accuracy, supply the velocity from the Error-State EKF at the
beginning of the scan.  This enables physically correct position correction
in addition to rotation correction:

```python
import numpy as np
from sensor_transposition.lidar.motion_distortion import deskew_scan
from sensor_transposition.imu.ekf import ImuEkf, EkfState

# Assume `ekf_state` is an EkfState valid at `t_scan_start`.
deskewed = deskew_scan(
    points=lidar_xyz,
    point_times=point_times,
    imu_times=imu_t,
    imu_accel=imu_a,
    imu_gyro=imu_g,
    ref_time=t_scan_start,
    accel_bias=ekf_state.accel_bias,
    gyro_bias=ekf_state.gyro_bias,
    gravity=np.array([0.0, 0.0, -9.81]),
    initial_velocity=ekf_state.velocity,
)
```

### Integration with ICP scan matching

Deskewing is typically applied immediately before scan matching:

```python
from sensor_transposition.lidar.motion_distortion import deskew_scan
from sensor_transposition.lidar.scan_matching import icp_align

# 1. Deskew the incoming scan.
src_deskewed = deskew_scan(
    src_points, src_point_times,
    imu_t, imu_a, imu_g,
    ref_time=src_point_times[0],
    gravity=np.zeros(3),
    initial_velocity=ekf_state.velocity,
)

# 2. Deskew the previous (reference) scan the same way.
tgt_deskewed = deskew_scan(
    tgt_points, tgt_point_times,
    imu_t_prev, imu_a_prev, imu_g_prev,
    ref_time=tgt_point_times[0],
    gravity=np.zeros(3),
    initial_velocity=prev_ekf_state.velocity,
)

# 3. Run ICP on the corrected clouds.
result = icp_align(src_deskewed, tgt_deskewed, max_iterations=50)
if result.converged:
    print("Relative transform:\n", result.transform)
```

---

## Accuracy Notes

### Rotation vs. position correction

| Component | Depends on | Typical effect |
|-----------|-----------|----------------|
| Rotation | Gyroscope + gyro bias | Dominant effect; always physically meaningful |
| Position | Accelerometer + gravity + initial velocity | Meaningful only when `initial_velocity` is known |

For a platform moving at **10 m/s** over a **100 ms** scan:
- Rotational distortion at **10 deg/s** yaw: up to **~0.17 rad × range** in lateral offset.
- Translational distortion: **10 m/s × 0.1 s = 1 m** from start to end of scan.

Both effects are significant for scan-matching at urban driving speeds.

### IMU rate requirements

| LiDAR rate | Scan duration | Recommended IMU rate |
|-----------|--------------|----------------------|
| 10 Hz | 100 ms | ≥ 100 Hz (preferably 200 Hz) |
| 20 Hz | 50 ms | ≥ 100 Hz |
| 100 Hz (solid-state) | 10 ms | ≥ 200 Hz |

The IMU window should extend **at least 5 ms before and after** the LiDAR scan
window to avoid edge-clamping artefacts on early and late points.

### Bias and noise

Gyroscope bias has the largest impact on rotation correction.  For a bias of
**0.1 deg/s** over a 100 ms scan window, the orientation error is only
**0.01 deg** — negligible.  However, accumulated over many frames without EKF
bias updates, the drift will compound.

Use `accel_bias` and `gyro_bias` from the :class:`~sensor_transposition.imu.ekf.EkfState`
or from an offline Allan variance calibration
(see `ImuParameters` in `sensor_collection.py`) for best results.

---

## Integration with the SLAM Pipeline

```
Raw LiDAR scan (distorted)
          ↓
  deskew_scan()          ← This module (uses IMU + EKF velocity)
          ↓
Deskewed point cloud
          ↓
  icp_align()            ← sensor_transposition.lidar.scan_matching
          ↓
Relative odometry transform
          ↓
  ImuEkf.pose_update()   ← sensor_transposition.imu.ekf
          ↓
Updated EkfState (new velocity for next scan's deskewing)
```

After deskewing, the resulting cloud can also be fed into the
:func:`~sensor_transposition.loop_closure.ScanContextDatabase` for loop
closure detection before scan matching.
