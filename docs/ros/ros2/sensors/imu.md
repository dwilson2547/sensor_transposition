# ROS 2 — IMU Integration

Guide to integrating Inertial Measurement Units (IMUs) with ROS 2, covering
driver setup, topics, filtering, QoS, and data recording.

---

## Table of Contents

1. [Supported IMUs](#supported-imus)
2. [Installation](#installation)
3. [Launching the Driver](#launching-the-driver)
4. [Topics & Message Types](#topics--message-types)
5. [IMU Filtering & Orientation](#imu-filtering--orientation)
6. [Coordinate Frames & Conventions](#coordinate-frames--conventions)
7. [QoS Considerations](#qos-considerations)
8. [Visualizing in rviz2](#visualizing-in-rviz2)
9. [Recording Data](#recording-data)
10. [Troubleshooting](#troubleshooting)
11. [Further Reading](#further-reading)

---

## Supported IMUs

| IMU | Interface | Package |
|-----|-----------|---------|
| Xsens MTi series | USB / Serial | `bluespace_ai_xsens_mti_driver` — build from source |
| Microstrain (3DM-GX5, CV5) | USB / Serial | `microstrain_inertial` — build from source |
| VectorNav (VN-100, VN-300) | USB / Serial | `vectornav` — build from source |
| Bosch BNO055 | I²C / UART | Community packages |
| PhidgetSpatial | USB | `phidgets_imu` — `sudo apt install ros-humble-phidgets-imu` |
| WitMotion | Serial / USB | Community `witmotion_ros2` — build from source |

---

## Installation

### Phidgets IMU (apt)

```bash
sudo apt install ros-humble-phidgets-imu
```

### Building from source (e.g., Microstrain)

```bash
cd ~/ros2_ws/src
git clone https://github.com/LORD-MicroStrain/microstrain_inertial.git -b ros2
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-up-to microstrain_inertial_driver
source install/setup.bash
```

### Serial permissions

```bash
sudo usermod -aG dialout $USER
# Log out and back in
```

---

## Launching the Driver

### Phidgets

```bash
ros2 launch phidgets_imu imu.launch.py
```

### Microstrain

```bash
ros2 launch microstrain_inertial_driver microstrain_launch.py
```

---

## Topics & Message Types

| Topic (typical) | Message Type | Content |
|-----------------|-------------|---------|
| `/imu/data` | `sensor_msgs/msg/Imu` | Orientation + angular velocity + linear acceleration |
| `/imu/data_raw` | `sensor_msgs/msg/Imu` | Angular velocity + linear acceleration (no orientation) |
| `/imu/mag` | `sensor_msgs/msg/MagneticField` | Magnetometer reading (if available) |

### `sensor_msgs/msg/Imu` key fields

| Field | Type | Description |
|-------|------|-------------|
| `orientation` | `geometry_msgs/msg/Quaternion` | Estimated orientation (may be identity if not computed) |
| `angular_velocity` | `geometry_msgs/msg/Vector3` | Gyroscope (rad/s) |
| `linear_acceleration` | `geometry_msgs/msg/Vector3` | Accelerometer (m/s²) |
| `*_covariance` | `float64[9]` | 3×3 row-major covariance matrix |

Inspect:

```bash
ros2 topic echo /imu/data
ros2 topic hz /imu/data
```

---

## IMU Filtering & Orientation

Raw IMU data often needs filtering to produce stable orientation. ROS 2
provides the same well-known filter nodes:

### imu_complementary_filter

```bash
sudo apt install ros-humble-imu-complementary-filter
ros2 run imu_complementary_filter complementary_filter_node \
  --ros-args -p do_bias_estimation:=true -p use_mag:=true
```

### imu_filter_madgwick

```bash
sudo apt install ros-humble-imu-filter-madgwick
ros2 run imu_filter_madgwick imu_filter_madgwick_node \
  --ros-args -p use_mag:=true -p publish_tf:=true
```

Both subscribe to `/imu/data_raw` (and optionally `/imu/mag`) and publish
a filtered `/imu/data` with a populated orientation field.

---

## Coordinate Frames & Conventions

- **Frame:** The IMU driver typically publishes in a frame called `imu_link`.
- **Axes:** ROS 2 follows the same conventions as ROS 1 — **ENU**
  (East-North-Up) for global frames, **FLU** (Forward-Left-Up) for body
  frames.
- **REP-145** remains the standard for IMU orientation.

Static transform from `base_link` to `imu_link`:

```bash
ros2 run tf2_ros static_transform_publisher \
  0 0 0.1 0 0 0 base_link imu_link
```

Or in a launch file:

```python
from launch_ros.actions import Node

Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'imu_link'],
)
```

---

## QoS Considerations

IMUs publish at high rates (100–400 Hz). Use **Best Effort** QoS to avoid
unnecessary retransmissions:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy

imu_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)
self.create_subscription(Imu, '/imu/data', self.cb, imu_qos)
```

---

## Visualizing in rviz2

1. Open: `rviz2`
2. Set **Fixed Frame** to `base_link` or `imu_link`.
3. Install the IMU plugin:
   ```bash
   sudo apt install ros-humble-rviz-imu-plugin
   ```
4. Add → **Imu** display → set topic to `/imu/data`.

---

## Recording Data

```bash
ros2 bag record /imu/data /imu/data_raw /imu/mag -o imu_session
```

---

## Troubleshooting

| Symptom | Possible Cause | Fix |
|---------|---------------|-----|
| Orientation all zeros | Driver does not compute orientation | Use a filter node (see above) |
| Drift over time | Gyro bias / no magnetometer | Enable magnetometer fusion or bias estimation |
| High noise | Vibrations | Mount with vibration-damping pads |
| Serial permission denied | User not in `dialout` group | See [Serial permissions](#serial-permissions) |
| Subscriber receives nothing | QoS mismatch | Check with `ros2 topic info /imu/data --verbose` |

---

## Further Reading

- [sensor_msgs/msg/Imu](https://docs.ros.org/en/humble/p/sensor_msgs/interfaces/msg/Imu.html)
- [imu_filter_madgwick ROS 2 (GitHub)](https://github.com/CCNYRoboticsLab/imu_tools/tree/humble)
- [imu_complementary_filter ROS 2 (GitHub)](https://github.com/CCNYRoboticsLab/imu_tools/tree/humble)
- [robot_localization ROS 2 (EKF with IMU)](https://docs.ros.org/en/humble/p/robot_localization/)
- [REP-145: Conventions for IMU Sensor Drivers](https://www.ros.org/reps/rep-0145.html)
