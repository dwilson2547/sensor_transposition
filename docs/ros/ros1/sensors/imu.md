# ROS 1 — IMU Integration

Guide to integrating Inertial Measurement Units (IMUs) with ROS 1, covering
driver setup, topics, filtering, and data recording.

---

## Table of Contents

1. [Supported IMUs](#supported-imus)
2. [Installation](#installation)
3. [Launching the Driver](#launching-the-driver)
4. [Topics & Message Types](#topics--message-types)
5. [IMU Filtering & Orientation](#imu-filtering--orientation)
6. [Coordinate Frames & Conventions](#coordinate-frames--conventions)
7. [Visualizing in rviz](#visualizing-in-rviz)
8. [Recording Data](#recording-data)
9. [Troubleshooting](#troubleshooting)
10. [Further Reading](#further-reading)

---

## Supported IMUs

| IMU | Interface | Package |
|-----|-----------|---------|
| Xsens MTi series | USB / Serial | `xsens_mti_driver` — build from source |
| Microstrain (3DM-GX5, CV5) | USB / Serial | `microstrain_inertial` — build from source |
| VectorNav (VN-100, VN-300) | USB / Serial | `vectornav` — build from source |
| Bosch BNO055 | I²C / UART | `ros-noetic-imu-bno055` or community packages |
| PhidgetSpatial | USB | `phidgets_imu` — `sudo apt install ros-noetic-phidgets-imu` |
| Generic serial IMU | Serial | `imu_complementary_filter` + custom parser |

---

## Installation

### Phidgets IMU (apt)

```bash
sudo apt install ros-noetic-phidgets-imu
```

### Building from source (e.g., Microstrain)

```bash
cd ~/catkin_ws/src
git clone https://github.com/LORD-MicroStrain/microstrain_inertial.git
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash
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
roslaunch phidgets_imu imu.launch
```

### Microstrain

```bash
roslaunch microstrain_inertial_driver microstrain.launch
```

---

## Topics & Message Types

| Topic (typical) | Message Type | Content |
|-----------------|-------------|---------|
| `/imu/data` | `sensor_msgs/Imu` | Orientation (quaternion), angular velocity, linear acceleration |
| `/imu/data_raw` | `sensor_msgs/Imu` | Angular velocity + linear acceleration only (no orientation) |
| `/imu/mag` | `sensor_msgs/MagneticField` | Magnetometer reading (if available) |

### `sensor_msgs/Imu` key fields

| Field | Type | Description |
|-------|------|-------------|
| `orientation` | `geometry_msgs/Quaternion` | Estimated orientation (may be zeros if not computed) |
| `angular_velocity` | `geometry_msgs/Vector3` | Gyroscope (rad/s) |
| `linear_acceleration` | `geometry_msgs/Vector3` | Accelerometer (m/s²) |
| `*_covariance` | `float64[9]` | 3×3 row-major covariance matrix |

Inspect:

```bash
rostopic echo /imu/data
rostopic hz /imu/data
```

---

## IMU Filtering & Orientation

Raw IMU data often requires filtering to produce a stable orientation estimate.
ROS 1 provides ready-made filter nodes:

### imu_complementary_filter

```bash
sudo apt install ros-noetic-imu-complementary-filter
rosrun imu_complementary_filter complementary_filter_node \
  _do_bias_estimation:=true _use_mag:=true
```

### imu_filter_madgwick

```bash
sudo apt install ros-noetic-imu-filter-madgwick
rosrun imu_filter_madgwick imu_filter_node \
  _use_mag:=true _publish_tf:=true
```

Both subscribe to `/imu/data_raw` (and optionally `/imu/mag`) and publish
a filtered `/imu/data` with a populated orientation field.

---

## Coordinate Frames & Conventions

- **Frame:** The IMU driver typically publishes in a frame called `imu_link`.
- **Axes:** ROS uses the **ENU** (East-North-Up) or **FLU**
  (Forward-Left-Up) convention depending on context.
- **REP-145** defines the standard for IMU orientation:
  - If the IMU is level, the orientation quaternion should represent the
    identity rotation.
  - Gravity should read approximately `(0, 0, 9.81)` in `linear_acceleration`.

Make sure your TF tree connects `imu_link` to your robot's `base_link`:

```xml
<node pkg="tf" type="static_transform_publisher" name="imu_tf"
      args="0 0 0.1 0 0 0 base_link imu_link 100" />
```

---

## Visualizing in rviz

1. Open rviz: `rosrun rviz rviz`
2. Set **Fixed Frame** to `base_link` or `imu_link`.
3. Add → **Imu** display (from `rviz_imu_plugin`):
   ```bash
   sudo apt install ros-noetic-rviz-imu-plugin
   ```
4. Set the topic to `/imu/data`.

---

## Recording Data

```bash
rosbag record /imu/data /imu/data_raw /imu/mag -O imu_session.bag
```

---

## Troubleshooting

| Symptom | Possible Cause | Fix |
|---------|---------------|-----|
| Orientation all zeros | Driver does not estimate orientation | Use a filter node (see above) |
| Drift over time | Gyro bias / no magnetometer | Enable magnetometer fusion or use `do_bias_estimation` |
| High noise | Vibrations | Mount the IMU with vibration-damping pads |
| Serial permission denied | User not in `dialout` group | See [Serial permissions](#serial-permissions) above |

---

## Further Reading

- [sensor_msgs/Imu](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html)
- [imu_filter_madgwick ROS wiki](http://wiki.ros.org/imu_filter_madgwick)
- [imu_complementary_filter ROS wiki](http://wiki.ros.org/imu_complementary_filter)
- [robot_localization (EKF with IMU)](http://wiki.ros.org/robot_localization)
- [REP-145: Conventions for IMU Sensor Drivers](https://www.ros.org/reps/rep-0145.html)
