# Wheel Odometry

## Overview

`sensor_transposition.wheel_odometry` provides two planar kinematic models for
dead-reckoning vehicle pose estimation from wheel measurements:

| Model | Class | Description |
|-------|-------|-------------|
| Differential drive | `DifferentialDriveOdometer` | Two independently driven wheels (robot / skid-steer) |
| Ackermann (bicycle) | `AckermannOdometer` | Car-like steering with front-axle steering angle |

Both models integrate a sequence of time-stamped measurements using the
**midpoint (trapezoidal) method** and return an `OdometryResult` containing
the accumulated **SE(2) pose** `(x, y, θ)` together with a **4×4 homogeneous
transform**.

---

## Coordinate Frame Convention

- The vehicle starts at the **origin** with heading along the **+x axis**.
- **x** – forward displacement (metres).
- **y** – lateral displacement, positive to the left (metres).
- **θ** – heading change, positive counter-clockwise (radians).

---

## Differential-Drive Model

### Kinematic equations

For a vehicle with track width *L* (metres) and left/right wheel linear
speeds *v_L* and *v_R*:

```
v     = (v_R + v_L) / 2          # forward speed of the vehicle centre
ω     = (v_R - v_L) / L          # angular rate

θ̇     = ω
ẋ     = v · cos(θ)
ẏ     = v · sin(θ)
```

### Usage – wheel speeds

```python
import numpy as np
from sensor_transposition.wheel_odometry import DifferentialDriveOdometer

# Robot with 54 cm track width
odom = DifferentialDriveOdometer(wheel_base=0.54)

timestamps = np.linspace(0.0, 5.0, 51)       # 5 s at 10 Hz
left_speeds  = np.full(51, 0.5)               # 0.5 m/s
right_speeds = np.full(51, 0.5)               # 0.5 m/s (straight line)

result = odom.integrate(timestamps, left_speeds, right_speeds)
print(f"x={result.x:.3f} m, y={result.y:.3f} m, θ={result.theta:.4f} rad")
# x=2.500 m, y=0.000 m, θ=0.0000 rad
```

### Usage – encoder ticks

Pass cumulative encoder tick counts and set `ticks_per_revolution`:

```python
from sensor_transposition.wheel_odometry import DifferentialDriveOdometer

odom = DifferentialDriveOdometer(wheel_base=0.54, wheel_radius=0.1)

# Cumulative tick counts (e.g. from a 360 tick/rev encoder)
left_ticks  = np.arange(0, 3601, 36, dtype=float)   # 100 ticks/step
right_ticks = np.arange(0, 3601, 36, dtype=float)

result = odom.integrate(
    timestamps,
    left_ticks,
    right_ticks,
    ticks_per_revolution=360,
)
```

The conversion formula is:

```
arc_per_tick = 2π · r / ticks_per_revolution
Δarc = Δticks · arc_per_tick
speed = Δarc / Δt
```

### Functional API

```python
from sensor_transposition.wheel_odometry import integrate_differential_drive

result = integrate_differential_drive(
    timestamps, left_speeds, right_speeds,
    wheel_base=0.54,
)
```

---

## Ackermann (Bicycle) Model

### Kinematic equations

For a car with wheel-base *L* (front-to-rear axle distance), rear-axle
reference point, forward speed *v*, and front steering angle *δ*:

```
ω  = v · tan(δ) / L      # yaw rate

θ̇  = ω
ẋ  = v · cos(θ)
ẏ  = v · sin(θ)
```

### Usage

```python
import numpy as np
from sensor_transposition.wheel_odometry import AckermannOdometer

# Car with 2.7 m wheel-base
odom = AckermannOdometer(wheel_base=2.7)

timestamps     = np.linspace(0.0, 10.0, 101)   # 10 s at 10 Hz
speeds         = np.full(101, 10.0)             # 10 m/s constant speed
steering_angles = np.full(101, 0.1)             # 0.1 rad steering → gentle left turn

result = odom.integrate(timestamps, speeds, steering_angles)
print(f"x={result.x:.2f} m, y={result.y:.2f} m, θ={result.theta:.4f} rad")
```

### Functional API

```python
from sensor_transposition.wheel_odometry import integrate_ackermann

result = integrate_ackermann(
    timestamps, speeds, steering_angles,
    wheel_base=2.7,
)
```

---

## OdometryResult

Both models return an `OdometryResult` dataclass:

| Attribute | Type | Description |
|-----------|------|-------------|
| `x` | `float` | Forward displacement (metres) |
| `y` | `float` | Lateral displacement (metres) |
| `theta` | `float` | Heading change (radians) |
| `duration` | `float` | Integration duration (seconds) |
| `num_samples` | `int` | Number of measurement samples |
| `translation` | `np.ndarray` | `[x, y, 0]` displacement vector |
| `rotation_matrix` | `np.ndarray` | 3×3 SO(3) rotation for `theta` |
| `transform` | `np.ndarray` | 4×4 homogeneous SE(2) transform |

The `transform` property is directly composable with
`FramePose.transform` to propagate a world-frame trajectory:

```python
import numpy as np
from sensor_transposition.frame_pose import FramePose
from sensor_transposition.wheel_odometry import DifferentialDriveOdometer

odom = DifferentialDriveOdometer(wheel_base=0.54)
result = odom.integrate(timestamps, left_speeds, right_speeds)

# Compose with the current world pose
current_pose = FramePose(timestamp=0.0)
new_transform = current_pose.transform @ result.transform
```

---

## Integration in the SLAM Pipeline

Wheel odometry fills the **dead-reckoning** role between higher-rate sensor
updates:

```
┌─────────────┐    dead-reckoning    ┌───────────────┐
│  Wheel ODom │ ──────────────────►  │ Initial guess │
└─────────────┘                      └──────┬────────┘
                                            │ ICP refinement
                                     ┌──────▼────────┐
                                     │  Scan Match   │
                                     └──────┬────────┘
                                            │
                                     ┌──────▼────────┐
                                     │  EKF predict  │
                                     └───────────────┘
```

- **Motion prior for ICP**: supply `result.transform` as the initial guess
  to `lidar.scan_matching.icp_align`.
- **EKF prediction step**: use `result.x`, `result.y`, and `result.theta`
  as the motion model input to `imu.ekf.ImuEkf.predict`.
- **Pose graph odometry edge**: add a
  `pose_graph.PoseGraphEdge` from the `result.transform` to constrain
  consecutive keyframe nodes.

---

## Limitations

- **2-D planar motion only**: the models do not account for pitch or roll.
  For 3-D motion (ramps, rough terrain) supplement with IMU pre-integration
  (`imu.preintegration.ImuPreintegrator`).
- **No slip modelling**: both models assume perfect rolling contact.
  Wheel-slip errors accumulate over time and should be corrected by an
  absolute position source such as GPS or LiDAR scan-matching.
- **Ackermann model uses rear-axle reference**: if your speed sensor is at
  the front axle or the vehicle CoM, adjust the input speed accordingly.
