# ROS 1 — LiDAR Integration

Guide to integrating LiDAR sensors with ROS 1, covering driver installation,
topic structure, visualization, and recording.

---

## Table of Contents

1. [Supported Sensors](#supported-sensors)
2. [Installation](#installation)
3. [Launching the Driver](#launching-the-driver)
4. [Topics & Message Types](#topics--message-types)
5. [Visualizing in rviz](#visualizing-in-rviz)
6. [Recording Data](#recording-data)
7. [Working with PointCloud2](#working-with-pointcloud2)
8. [Troubleshooting](#troubleshooting)
9. [Further Reading](#further-reading)

---

## Supported Sensors

The most common ROS 1 LiDAR driver packages:

| Sensor Family | Package | Install Command |
|---------------|---------|-----------------|
| Velodyne (VLP-16, VLP-32, HDL-64) | `velodyne` | `sudo apt install ros-noetic-velodyne` |
| Ouster (OS0, OS1, OS2) | `ouster_ros` | Build from source — [ouster-ros](https://github.com/ouster-lidar/ouster-ros) |
| Livox (Mid-40, Mid-70, Horizon, Avia) | `livox_ros_driver` | Build from source — [livox_ros_driver](https://github.com/Livox-SDK/livox_ros_driver) |
| SICK (TiM, LMS) | `sick_scan` | `sudo apt install ros-noetic-sick-scan` |

---

## Installation

### Velodyne example

```bash
sudo apt install ros-noetic-velodyne
```

### Building from source (e.g., Ouster)

```bash
cd ~/catkin_ws/src
git clone --recurse-submodules https://github.com/ouster-lidar/ouster-ros.git
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash
```

---

## Launching the Driver

### Velodyne VLP-16

```bash
roslaunch velodyne_pointcloud VLP16_points.launch
```

### Ouster OS1

```bash
roslaunch ouster_ros sensor.launch sensor_hostname:=os1-XXXXXXXXXXXX.local
```

The driver node connects to the sensor over Ethernet and begins publishing
point-cloud data.

---

## Topics & Message Types

| Topic (typical) | Message Type | Content |
|-----------------|-------------|---------|
| `/velodyne_points` | `sensor_msgs/PointCloud2` | XYZ + intensity point cloud |
| `/ouster/points` | `sensor_msgs/PointCloud2` | XYZ + intensity + ring + range |
| `/livox/lidar` | `sensor_msgs/PointCloud2` | XYZ + intensity |
| `/scan` | `sensor_msgs/LaserScan` | 2-D scan (some 2-D LiDARs) |

Inspect available topics:

```bash
rostopic list | grep -i point
rostopic echo /velodyne_points --noarr   # print header without array data
rostopic hz /velodyne_points             # check publish rate
```

---

## Visualizing in rviz

1. Open rviz: `rosrun rviz rviz`
2. Set **Fixed Frame** to the LiDAR frame (e.g., `velodyne`).
3. Click **Add → By topic → /velodyne_points → PointCloud2**.
4. Adjust **Size**, **Color Transformer** (intensity or axis), and **Decay
   Time** to taste.

---

## Recording Data

```bash
# Record only the point-cloud topic
rosbag record /velodyne_points -O lidar_session.bag

# Record everything
rosbag record -a
```

### Inspecting the bag

```bash
rosbag info lidar_session.bag
```

---

## Working with PointCloud2

### Python (rospy + sensor_msgs)

```python
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

def callback(msg):
    for point in pc2.read_points(msg, field_names=("x", "y", "z", "intensity")):
        x, y, z, intensity = point
        # process point ...

rospy.init_node('lidar_listener')
rospy.Subscriber('/velodyne_points', PointCloud2, callback)
rospy.spin()
```

---

## Troubleshooting

| Symptom | Possible Cause | Fix |
|---------|---------------|-----|
| No data on topic | Sensor not on same subnet | Set a static IP on the same subnet as the LiDAR (e.g., `192.168.1.100`) |
| `rviz` shows nothing | Wrong fixed frame | Set the fixed frame to the LiDAR's `frame_id` |
| Low publish rate | Network bandwidth | Use a direct Ethernet connection, not Wi-Fi |
| `rosdep` errors during build | Missing keys | Run `rosdep update` then retry |

---

## Further Reading

- [Velodyne ROS wiki](http://wiki.ros.org/velodyne)
- [Ouster ROS driver (GitHub)](https://github.com/ouster-lidar/ouster-ros)
- [Livox ROS driver (GitHub)](https://github.com/Livox-SDK/livox_ros_driver)
- [sensor_msgs/PointCloud2](http://wiki.ros.org/sensor_msgs)
- [rviz PointCloud2 display](http://wiki.ros.org/rviz/DisplayTypes/PointCloud)
