# ROS 2 — Radar Integration

Guide to integrating radar sensors with ROS 2, covering driver setup, message
types, QoS, visualization, and data recording.

---

## Table of Contents

1. [Overview](#overview)
2. [Supported Radars](#supported-radars)
3. [Installation](#installation)
4. [Launching the Driver](#launching-the-driver)
5. [Topics & Message Types](#topics--message-types)
6. [QoS Considerations](#qos-considerations)
7. [Visualizing Radar Data](#visualizing-radar-data)
8. [Recording Data](#recording-data)
9. [Working with Radar Data in Python](#working-with-radar-data-in-python)
10. [Troubleshooting](#troubleshooting)
11. [Further Reading](#further-reading)

---

## Overview

Automotive and robotic radar sensors output a list of **detections** (range,
azimuth, velocity, RCS) per scan cycle. Unlike LiDAR, radar provides **Doppler
velocity** per detection, which is valuable for tracking moving objects.

ROS 2 introduced the **`radar_msgs`** package with standardized message types:

- `radar_msgs/msg/RadarReturn` — a single detection.
- `radar_msgs/msg/RadarScan` — a collection of returns for one scan cycle.

Many drivers also publish `sensor_msgs/msg/PointCloud2` with extra fields for
velocity and RCS, ensuring compatibility with standard point-cloud tools.

---

## Supported Radars

| Radar | Interface | Package |
|-------|-----------|---------|
| Continental ARS408 | CAN bus | `ars408_driver` — community packages |
| TI AWR1843 / IWR6843 | Serial / USB | `ti_mmwave_ros2` — build from source |
| Smartmicro (UMRR) | Ethernet / CAN | `umrr_ros2_driver` — build from source |
| Ainstein (T-79, K-79) | Serial / CAN | `ainstein_radar` — build from source for ROS 2 |
| Oculii Eagle / Falcon | Ethernet | Vendor SDK + custom wrapper |

---

## Installation

### radar_msgs (standard messages)

```bash
sudo apt install ros-humble-radar-msgs
```

### TI mmWave (build from source)

```bash
cd ~/ros2_ws/src
git clone https://github.com/radar-lab/ti_mmwave_ros2.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-up-to ti_mmwave_ros2
source install/setup.bash
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
ros2 launch ti_mmwave_ros2 1843_multi_3d.launch.py
```

### Ainstein

```bash
ros2 launch ainstein_radar_drivers t79_node.launch.py
```

---

## Topics & Message Types

### Standard radar_msgs

| Topic (typical) | Message Type | Content |
|-----------------|-------------|---------|
| `/radar/scan` | `radar_msgs/msg/RadarScan` | Array of `RadarReturn` for one cycle |

#### RadarReturn fields

| Field | Type | Description |
|-------|------|-------------|
| `range` | float32 | Distance to target (m) |
| `azimuth` | float32 | Horizontal angle (rad) |
| `elevation` | float32 | Vertical angle (rad) |
| `doppler_velocity` | float32 | Radial velocity (m/s) |
| `amplitude` | float32 | Reflected signal strength (dB) |

### PointCloud2 variant

| Topic | Message Type | Fields |
|-------|-------------|--------|
| `/radar/points` | `sensor_msgs/msg/PointCloud2` | x, y, z, velocity, intensity |

Inspect:

```bash
ros2 topic list | grep radar
ros2 topic echo /radar/scan
ros2 topic hz /radar/scan
```

---

## QoS Considerations

Radar scans are typically 10–20 Hz. A **Best Effort** profile works well:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy

radar_qos = QoSProfile(
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)
self.create_subscription(RadarScan, '/radar/scan', self.cb, radar_qos)
```

---

## Visualizing Radar Data

### rviz2 with PointCloud2

1. Open: `rviz2`
2. Set **Fixed Frame** to the radar frame.
3. Add → **PointCloud2** on the radar point-cloud topic.
4. Color by `velocity` or `intensity`.

### rviz2 with MarkerArray

Some drivers publish a `visualization_msgs/msg/MarkerArray`. Add →
**MarkerArray** in rviz2 for a quick look.

### Custom visualization

Convert `RadarScan` to `MarkerArray` in a small node for full control over
colours, sizes, and labels.

---

## Recording Data

```bash
ros2 bag record /radar/scan /radar/points -o radar_session
```

---

## Working with Radar Data in Python

### Using RadarScan

```python
import rclpy
from rclpy.node import Node
from radar_msgs.msg import RadarScan

class RadarListener(Node):
    def __init__(self):
        super().__init__('radar_listener')
        self.create_subscription(RadarScan, '/radar/scan', self.cb, 10)

    def cb(self, msg):
        for ret in msg.returns:
            self.get_logger().info(
                f'range={ret.range:.2f}m  az={ret.azimuth:.2f}rad  '
                f'vel={ret.doppler_velocity:.2f}m/s  amp={ret.amplitude:.1f}dB'
            )

rclpy.init()
rclpy.spin(RadarListener())
```

### Using PointCloud2

```python
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

def cb(self, msg):
    for pt in pc2.read_points(msg, field_names=('x', 'y', 'z', 'velocity')):
        x, y, z, vel = pt
        # process ...
```

---

## Troubleshooting

| Symptom | Possible Cause | Fix |
|---------|---------------|-----|
| No data on topic | Serial port not detected | Check `ls /dev/ttyACM*`; update `port` parameter |
| CAN interface down | CAN bus not configured | Run `sudo ip link set up can0` |
| QoS mismatch | Subscriber settings differ from publisher | Inspect with `ros2 topic info --verbose` |
| Ghost detections | Multi-path reflections | Tune CFAR threshold or filter by velocity / amplitude |

---

## Further Reading

- [radar_msgs (GitHub)](https://github.com/ros-perception/radar_msgs)
- [TI mmWave ROS 2 package (GitHub)](https://github.com/radar-lab/ti_mmwave_ros2)
- [Continental ARS408 ROS 2 driver (community)](https://github.com/Project-MANAS/ars408_ros)
- [sensor_msgs/msg/PointCloud2](https://docs.ros.org/en/humble/p/sensor_msgs/interfaces/msg/PointCloud2.html)
- [ROS 2 QoS concepts](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html)
