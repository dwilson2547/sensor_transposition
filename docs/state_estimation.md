# State Estimation: Error-State EKF for IMU + Odometry Fusion

This guide explains the Error-State Extended Kalman Filter (ES-EKF) provided
by `sensor_transposition.imu.ekf`.  The filter fuses high-rate IMU measurements
with lower-rate pose, position, or velocity observations (from GPS, LiDAR
odometry, or wheel encoders) to produce a smooth, continuous state estimate.

---

## Background

Estimating the pose and velocity of a moving platform from noisy sensors
requires a principled fusion strategy.  The **Error-State EKF** is the
standard approach used in commercial INS and open-source SLAM systems
(e.g. MSCKF, VINS-Mono, LIO-SAM) because it:

* Integrates IMU measurements at high rate (100–1000 Hz) without accumulating
  linearisation error.
* Handles the non-Euclidean SO(3) orientation manifold correctly by tracking
  a *small* orientation error in the tangent space.
* Is numerically robust and computationally cheap (15×15 matrices).

---

## State Representation

The filter maintains two complementary representations:

### Nominal state (`EkfState`)

| Field | Symbol | Units | Description |
|-------|--------|-------|-------------|
| `position` | **p** | m | Position in the world/map frame |
| `velocity` | **v** | m/s | Velocity in the world/map frame |
| `quaternion` | **q** | — | Body-to-world orientation `[w, x, y, z]` |
| `accel_bias` | **bₐ** | m/s² | Accelerometer bias |
| `gyro_bias` | **bᵍ** | rad/s | Gyroscope bias |
| `covariance` | **P** | — | 15×15 error-state covariance |
| `timestamp` | *t* | s | UNIX timestamp |

### Error state (internal, 15-dimensional)

```
δx = [δp (3),  δv (3),  δθ (3),  δbₐ (3),  δbᵍ (3)]
```

`δθ` is a rotation vector in the SO(3) tangent space — a 3-D vector whose
magnitude is the rotation angle and whose direction is the rotation axis.

---

## Algorithms

### Prediction (IMU propagation)

For each IMU sample `(a, ω, dt)`:

```
a_c = a − bₐ                    # corrected accelerometer
ω_c = ω − bᵍ                   # corrected gyroscope

p  ← p + v·dt + ½(R·a_c + g)·dt²
v  ← v + (R·a_c + g)·dt
q  ← q ⊗ Exp(ω_c·dt)           # SO(3) exponential map (Rodrigues)
bₐ ← bₐ    (random-walk model)
bᵍ ← bᵍ
```

The 15×15 linearised transition Jacobian **F** and discrete process-noise
matrix **Q** advance the covariance:

```
P ← F P Fᵀ + Q
```

### Update (measurement fusion)

Any observation that can be written as a linear function of the error state
`z = H δx + n` (where `n ~ N(0, R_noise)`) is fused by:

```
S   = H P Hᵀ + R_noise
K   = P Hᵀ S⁻¹                  # Kalman gain
δx  = K (z − ẑ)                 # error-state correction
```

The error state is then **injected** into the nominal state and reset to zero.
The covariance is updated in the **Joseph form** for numerical stability:

```
P ← (I − KH) P (I − KH)ᵀ + K R_noise Kᵀ
```

---

## API

### `ImuEkf`

```python
from sensor_transposition.imu.ekf import ImuEkf, EkfState
import numpy as np

ekf = ImuEkf(
    gravity=np.array([0.0, 0.0, -9.81]),  # world-frame gravity (m/s²)
    accel_noise_density=0.01,              # m/s²/√Hz
    gyro_noise_density=0.001,             # rad/s/√Hz
    accel_bias_noise=0.0001,              # m/s²·√Hz
    gyro_bias_noise=0.00001,              # rad/s·√Hz
)
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `gravity` | Gravity vector in the world frame | `[0, 0, −9.81]` |
| `accel_noise_density` | Accelerometer white-noise density (m/s²/√Hz) | `0.01` |
| `gyro_noise_density` | Gyroscope white-noise density (rad/s/√Hz) | `0.001` |
| `accel_bias_noise` | Accelerometer bias random-walk (m/s²·√Hz) | `0.0001` |
| `gyro_bias_noise` | Gyroscope bias random-walk (rad/s·√Hz) | `0.00001` |

Noise densities should match the values from your IMU datasheet or Allan
variance analysis (see `ImuParameters` in `sensor_collection.py`).

---

### `ImuEkf.predict`

```python
state = ekf.predict(state, accel, gyro, dt)
```

Propagates the nominal state and covariance forward by one IMU sample.

| Argument | Shape | Description |
|----------|-------|-------------|
| `state` | `EkfState` | Current filter state |
| `accel` | `(3,)` | Accelerometer reading in m/s² (body frame) |
| `gyro` | `(3,)` | Gyroscope reading in rad/s (body frame) |
| `dt` | scalar | Time step in seconds (must be > 0) |

---

### `ImuEkf.position_update`

```python
state = ekf.position_update(state, position, noise)
```

Fuses a 3-D position measurement (e.g. from GPS or a LiDAR odometry
translation estimate).

| Argument | Shape | Description |
|----------|-------|-------------|
| `position` | `(3,)` | Measured position in the world frame (m) |
| `noise` | `(3, 3)` | Measurement noise covariance (m²) |

---

### `ImuEkf.velocity_update`

```python
state = ekf.velocity_update(state, velocity, noise)
```

Fuses a 3-D velocity measurement (e.g. from wheel odometry or GPS-Doppler).

| Argument | Shape | Description |
|----------|-------|-------------|
| `velocity` | `(3,)` | Measured velocity in the world frame (m/s) |
| `noise` | `(3, 3)` | Measurement noise covariance ((m/s)²) |

---

### `ImuEkf.pose_update`

```python
state = ekf.pose_update(
    state,
    position,
    quaternion,
    position_noise,
    orientation_noise,
)
```

Fuses a full 6-DOF pose measurement (e.g. from LiDAR scan matching via
`sensor_transposition.lidar.scan_matching.icp_align`).

| Argument | Shape | Description |
|----------|-------|-------------|
| `position` | `(3,)` | Measured position in metres |
| `quaternion` | `(4,)` | Measured orientation `[w, x, y, z]` |
| `position_noise` | `(3, 3)` | Position noise covariance (m²) |
| `orientation_noise` | `(3, 3)` | Orientation noise covariance (rad²) |

---

## Usage Examples

### GPS-aided INS

```python
from sensor_transposition.imu.ekf import ImuEkf, EkfState
from sensor_transposition.gps.converter import geodetic_to_enu
import numpy as np

ekf = ImuEkf()

# Initialise state at the GPS reference point.
state = EkfState(timestamp=imu_timestamps[0])

# Run prediction loop at IMU rate.
for i in range(1, len(imu_timestamps)):
    dt = imu_timestamps[i] - imu_timestamps[i - 1]
    state = ekf.predict(state, accel[i], gyro[i], dt)

    # Fuse GPS at 1 Hz when a new fix is available.
    if new_gps_fix_available(imu_timestamps[i]):
        lat, lon, alt = get_latest_gps_fix()
        enu = geodetic_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)
        state = ekf.position_update(
            state,
            position=np.array(enu),
            noise=np.diag([0.5**2, 0.5**2, 1.0**2]),  # 0.5 m horizontal, 1 m vertical
        )
```

### LiDAR odometry fusion

```python
from sensor_transposition.imu.ekf import ImuEkf, EkfState
from sensor_transposition.lidar.scan_matching import icp_align
import numpy as np

ekf = ImuEkf()
state = EkfState(timestamp=t0)

prev_cloud = lidar_frames[0]

for t, cloud, accel, gyro in sensor_stream:
    dt = t - state.timestamp

    # --- IMU prediction ---
    state = ekf.predict(state, accel, gyro, dt)

    # --- LiDAR odometry update (at LiDAR frame rate) ---
    if new_lidar_frame:
        result = icp_align(prev_cloud, cloud)
        if result.converged:
            R_inc = result.transform[:3, :3]
            t_inc = result.transform[:3, 3]

            # Convert incremental transform to world-frame pose.
            R_world = state.rotation_matrix @ R_inc
            p_world = state.position + state.rotation_matrix @ t_inc

            from scipy.spatial.transform import Rotation
            q_scipy = Rotation.from_matrix(R_world).as_quat()   # [x, y, z, w]
            q_world = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

            state = ekf.pose_update(
                state,
                position=p_world,
                quaternion=q_world,
                position_noise=np.eye(3) * 0.05**2,
                orientation_noise=np.eye(3) * np.deg2rad(0.5)**2,
            )
            prev_cloud = cloud
```

---

## Noise Parameter Tuning

The filter behaviour is governed by two sets of noise parameters:

**Process noise** (IMU noise densities, set at construction):

* Larger `accel_noise_density` / `gyro_noise_density` → covariance grows faster
  during prediction → more weight given to external measurements.
* Larger `accel_bias_noise` / `gyro_bias_noise` → biases allowed to change more
  rapidly.

**Measurement noise** (passed to each update call):

* Smaller noise covariance → filter trusts the measurement more.
* Set values to the *actual* expected measurement uncertainty (e.g. GPS CEP,
  ICP alignment residual).

IMU noise density values can be obtained from:

1. The IMU datasheet (look for "noise density" or "ARW / VRW" specifications).
2. Allan variance analysis of a static IMU log
   (see `ImuParameters` in `sensor_collection.py`).

---

## Integration with the SLAM Pipeline

The ES-EKF slots naturally into the sensor-transposition SLAM workflow:

```
ImuPreintegrator          →  high-rate relative-pose prediction
         ↓
      ImuEkf.predict       →  continuous-time state propagation
         ↓
icp_align / ScanContext    →  odometry and loop-closure edges
         ↓
ImuEkf.pose_update         →  correct drift with geometry constraints
         ↓
      EkfState             →  smooth trajectory for FramePoseSequence
```

For global consistency after loop closures, pass the corrected trajectory to
a pose-graph optimiser (see `TODO.md` — back-end graph optimisation is the
next pipeline stage after state estimation).
