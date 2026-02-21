# ROS 2 — LiDAR Integration

Guide to integrating LiDAR sensors with ROS 2, covering driver installation,
topics, QoS configuration, visualization, and recording.

---

## Table of Contents

1. [Supported Sensors](#supported-sensors)
2. [Installation](#installation)
3. [Launching the Driver](#launching-the-driver)
4. [Topics & Message Types](#topics--message-types)
5. [QoS Considerations](#qos-considerations)
6. [Visualizing in rviz2](#visualizing-in-rviz2)
7. [Recording Data](#recording-data)
8. [Working with PointCloud2](#working-with-pointcloud2)
9. [Troubleshooting](#troubleshooting)
10. [Further Reading](#further-reading)

---

## Supported Sensors

| Sensor Family | Package | Install Command |
|---------------|---------|-----------------|
| Velodyne (VLP-16, VLP-32, HDL-64) | `velodyne` | `sudo apt install ros-humble-velodyne` |
| Ouster (OS0, OS1, OS2) | `ouster-ros` | Build from source — [ouster-ros](https://github.com/ouster-lidar/ouster-ros) |
| Livox (Mid-360, HAP, Avia) | `livox_ros_driver2` | Build from source — [livox_ros_driver2](https://github.com/Livox-SDK/livox_ros_driver2) |
| SICK (TiM, LMS, multiScan) | `sick_scan_xd` | Build from source — [sick_scan_xd](https://github.com/SICKAG/sick_scan_xd) |
| Hesai (Pandar, AT128) | `hesai_ros_driver` | Build from source — [HesaiTechnology/HesaiLidar_ROS_2.0](https://github.com/HesaiTechnology/HesaiLidar_ROS_2.0) |

---

## Installation

### Velodyne (apt)

```bash
sudo apt install ros-humble-velodyne
```

### Building from source (e.g., Ouster)

```bash
cd ~/ros2_ws/src
git clone --recurse-submodules https://github.com/ouster-lidar/ouster-ros.git -b ros2
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-up-to ouster_ros
source install/setup.bash
```

---

## Launching the Driver

### Velodyne VLP-16

```bash
ros2 launch velodyne velodyne-all-nodes-VLP16-launch.py
```

### Ouster OS1

```bash
ros2 launch ouster_ros sensor.launch.xml sensor_hostname:=os1-XXXXXXXXXXXX.local
```

### Livox Mid-360

```bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

---

## Topics & Message Types

| Topic (typical) | Message Type | Content |
|-----------------|-------------|---------|
| `/velodyne_points` | `sensor_msgs/msg/PointCloud2` | XYZ + intensity |
| `/ouster/points` | `sensor_msgs/msg/PointCloud2` | XYZ + intensity + ring + range |
| `/livox/lidar` | `sensor_msgs/msg/PointCloud2` | XYZ + intensity |
| `/scan` | `sensor_msgs/msg/LaserScan` | 2-D scan (some 2-D LiDARs) |

Inspect:

```bash
ros2 topic list | grep -i point
ros2 topic echo /velodyne_points --no-arr   # header only
ros2 topic hz /velodyne_points
```

---

## QoS Considerations

LiDAR drivers often publish with **Best Effort** reliability and **Volatile**
durability to minimize latency. Make sure your subscriber uses a compatible QoS
profile:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

lidar_qos = QoSProfile(
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
self.create_subscription(PointCloud2, '/velodyne_points', self.cb, lidar_qos)
```

> **Tip:** If you see a subscriber with no data, a QoS mismatch is the most
> common cause. Use `ros2 topic info /velodyne_points --verbose` to check the
> publisher's QoS settings.

---

## Visualizing in rviz2

1. Open: `rviz2`
2. Set **Fixed Frame** to the LiDAR frame (e.g., `velodyne`).
3. Add → **By topic** → `/velodyne_points` → **PointCloud2**.
4. Adjust **Size**, **Color Transformer**, and **Decay Time**.

---

## Recording Data

```bash
# Record specific topics
ros2 bag record /velodyne_points -o lidar_session

# Record everything
ros2 bag record -a
```

### Inspect

```bash
ros2 bag info lidar_session
```

### Replay

```bash
ros2 bag play lidar_session
```

---

## Working with PointCloud2

### Python (rclpy + sensor_msgs)

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

class LidarListener(Node):
    def __init__(self):
        super().__init__('lidar_listener')
        self.create_subscription(PointCloud2, '/velodyne_points', self.cb, 10)

    def cb(self, msg):
        for point in pc2.read_points(msg, field_names=('x', 'y', 'z', 'intensity')):
            x, y, z, intensity = point
            # process point ...

rclpy.init()
rclpy.spin(LidarListener())
```

> In ROS 2, use the **`sensor_msgs_py`** package (pure Python) instead of the
> C++-backed `sensor_msgs.point_cloud2` helper from ROS 1.

---

## Troubleshooting

| Symptom | Possible Cause | Fix |
|---------|---------------|-----|
| No data on topic | Sensor not on same subnet | Set a static IP on the same subnet as the LiDAR |
| Subscriber receives nothing | QoS mismatch | Match the publisher's reliability & durability settings |
| `rviz2` shows nothing | Wrong fixed frame | Set the fixed frame to the LiDAR's `frame_id` |
| Low publish rate | Network bandwidth | Use a direct Ethernet connection |
| DDS discovery issues | Multicast blocked | Set `ROS_LOCALHOST_ONLY=1` or configure DDS peer list |

---

## Further Reading

- [Velodyne ROS 2 driver (GitHub)](https://github.com/ros-drivers/velodyne/tree/ros2)
- [Ouster ROS 2 driver (GitHub)](https://github.com/ouster-lidar/ouster-ros)
- [Livox ROS 2 driver (GitHub)](https://github.com/Livox-SDK/livox_ros_driver2)
- [sensor_msgs/msg/PointCloud2](https://docs.ros.org/en/humble/p/sensor_msgs/interfaces/msg/PointCloud2.html)
- [ROS 2 QoS concepts](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html)
