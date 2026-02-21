# ROS 1 — Radar Integration

Guide to integrating radar sensors with ROS 1, covering driver setup, topics,
message types, and data recording.

---

## Table of Contents

1. [Overview](#overview)
2. [Supported Radars](#supported-radars)
3. [Installation](#installation)
4. [Launching the Driver](#launching-the-driver)
5. [Topics & Message Types](#topics--message-types)
6. [Visualizing Radar Data](#visualizing-radar-data)
7. [Recording Data](#recording-data)
8. [Working with Radar Data in Python](#working-with-radar-data-in-python)
9. [Troubleshooting](#troubleshooting)
10. [Further Reading](#further-reading)

---

## Overview

Automotive and robotic radar sensors typically output a list of **detections**
(range, azimuth, velocity, RCS) at each scan cycle. Unlike LiDAR, radar
provides **Doppler velocity** per detection, making it especially useful for
tracking moving objects.

ROS 1 does not define a single standard radar message in `sensor_msgs`. Most
drivers use one of:

- A custom message type (e.g., `RadarDetection`, `RadarScan`).
- `sensor_msgs/PointCloud2` with extra fields for velocity and RCS.
- `visualization_msgs/MarkerArray` for display.

---

## Supported Radars

| Radar | Interface | Package |
|-------|-----------|---------|
| Continental ARS408 | CAN bus | `ars408_ros` — community packages |
| TI AWR1843 / IWR6843 | Serial / USB | `ti_mmwave_rospkg` — [TI mmWave ROS](https://github.com/radar-lab/ti_mmwave_rospkg) |
| Smartmicro (UMRR) | Ethernet / CAN | `umrr_ros_driver` — build from source |
| Ainstein (T-79, K-79) | Serial / CAN | `ainstein_radar` — `sudo apt install ros-noetic-ainstein-radar` |

---

## Installation

### TI mmWave (build from source)

```bash
cd ~/catkin_ws/src
git clone https://github.com/radar-lab/ti_mmwave_rospkg.git
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash
```

### Ainstein (apt)

```bash
sudo apt install ros-noetic-ainstein-radar
```

### CAN bus setup (for Continental, Smartmicro)

```bash
sudo apt install can-utils
sudo ip link set can0 type can bitrate 500000
sudo ip link set up can0
```

---

## Launching the Driver

### TI mmWave

```bash
roslaunch ti_mmwave_rospkg 1843_multi_3d_0.launch
```

### Ainstein

```bash
roslaunch ainstein_radar_drivers t79_node.launch
```

---

## Topics & Message Types

Because there is no single standard, topics and messages vary by driver. Common
patterns:

| Topic (typical) | Message Type | Content |
|-----------------|-------------|---------|
| `/ti_mmwave/radar_scan` | `sensor_msgs/PointCloud2` | x, y, z, velocity, intensity |
| `/radar/detections` | Custom `RadarDetection[]` | range, azimuth, elevation, velocity, RCS |
| `/radar/tracks` | Custom `RadarTrack[]` | Tracked object state (position, velocity, ID) |
| `/radar/markers` | `visualization_msgs/MarkerArray` | Visualization markers |

### PointCloud2 radar fields (TI mmWave example)

| Field | Type | Description |
|-------|------|-------------|
| `x`, `y`, `z` | float32 | Detection position (m) |
| `velocity` | float32 | Radial (Doppler) velocity (m/s) |
| `intensity` | float32 | Signal-to-noise ratio or RCS (dBsm) |

---

## Visualizing Radar Data

### rviz with PointCloud2

1. Open rviz: `rosrun rviz rviz`
2. Set **Fixed Frame** to the radar frame (e.g., `ti_mmwave_0`).
3. Add → **PointCloud2** on the radar topic.
4. Color by the **velocity** or **intensity** field.

### rviz with MarkerArray

Some drivers publish a `MarkerArray` for direct visualization. Add →
**MarkerArray** and select the topic.

---

## Recording Data

```bash
rosbag record /ti_mmwave/radar_scan -O radar_session.bag
```

---

## Working with Radar Data in Python

```python
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

def callback(msg):
    for point in pc2.read_points(msg, field_names=("x", "y", "z", "velocity")):
        x, y, z, vel = point
        print(f"x={x:.2f} y={y:.2f} z={z:.2f} vel={vel:.2f} m/s")

rospy.init_node('radar_listener')
rospy.Subscriber('/ti_mmwave/radar_scan', PointCloud2, callback)
rospy.spin()
```

---

## Troubleshooting

| Symptom | Possible Cause | Fix |
|---------|---------------|-----|
| No data on topic | Serial port not detected | Check `ls /dev/ttyACM*` and update the `port` parameter |
| CAN interface down | CAN bus not configured | Run `sudo ip link set up can0` (see above) |
| Ghost detections | Multi-path reflections | Tune CFAR threshold or filter by velocity / RCS |
| High noise / clutter | Indoor environment | Test outdoors or increase detection threshold |

---

## Further Reading

- [TI mmWave ROS package (GitHub)](https://github.com/radar-lab/ti_mmwave_rospkg)
- [Continental ARS408 ROS driver (community)](https://github.com/Project-MANAS/ars408_ros)
- [ainstein_radar ROS wiki](http://wiki.ros.org/ainstein_radar)
- [sensor_msgs/PointCloud2](http://wiki.ros.org/sensor_msgs)
- [Radar fundamentals for robotics (overview)](https://www.ros.org/news/2019/06/radar-in-ros.html)
