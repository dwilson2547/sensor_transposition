# ROS 2 — GPS Integration

Guide to integrating GPS/GNSS receivers with ROS 2, covering driver setup,
topics, coordinate handling, QoS, and data recording.

---

## Table of Contents

1. [Supported Receivers](#supported-receivers)
2. [Installation](#installation)
3. [Launching the Driver](#launching-the-driver)
4. [Topics & Message Types](#topics--message-types)
5. [Coordinate Frames & Transforms](#coordinate-frames--transforms)
6. [QoS Considerations](#qos-considerations)
7. [Visualizing GPS Data](#visualizing-gps-data)
8. [Recording Data](#recording-data)
9. [Converting to Local Coordinates](#converting-to-local-coordinates)
10. [Troubleshooting](#troubleshooting)
11. [Further Reading](#further-reading)

---

## Supported Receivers

| Receiver | Interface | Package |
|----------|-----------|---------|
| u-blox (NEO-M8, ZED-F9P) | Serial / USB | `ublox` — build from source for ROS 2 |
| NMEA-compatible (generic) | Serial | `nmea_navsat_driver` — `sudo apt install ros-humble-nmea-navsat-driver` |
| Septentrio (mosaic, AsteRx) | Serial / Ethernet | `septentrio_gnss_driver` — build from source |
| SwiftNav (Piksi, Duro) | Serial / Ethernet | `swiftnav_ros2` — build from source |

---

## Installation

### NMEA driver (apt — works with most receivers)

```bash
sudo apt install ros-humble-nmea-navsat-driver
```

### Building from source (e.g., u-blox)

```bash
cd ~/ros2_ws/src
git clone https://github.com/KumarRobotics/ublox.git -b ros2
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-up-to ublox_gps
source install/setup.bash
```

### Serial permissions

```bash
sudo usermod -aG dialout $USER
# Log out and back in
```

---

## Launching the Driver

### NMEA driver

```bash
ros2 launch nmea_navsat_driver nmea_serial_driver.launch.py \
  port:=/dev/ttyUSB0 baud:=115200
```

### u-blox

```bash
ros2 launch ublox_gps ublox_gps_node-launch.py
```

---

## Topics & Message Types

| Topic (typical) | Message Type | Content |
|-----------------|-------------|---------|
| `/fix` or `/gps/fix` | `sensor_msgs/msg/NavSatFix` | Latitude, longitude, altitude, covariance |
| `/vel` or `/gps/vel` | `geometry_msgs/msg/TwistStamped` | Velocity from GPS |
| `/time_reference` | `sensor_msgs/msg/TimeReference` | GPS UTC time |

### NavSatFix key fields

| Field | Description |
|-------|-------------|
| `latitude` | Degrees (WGS-84) |
| `longitude` | Degrees (WGS-84) |
| `altitude` | Metres above the WGS-84 ellipsoid |
| `status.status` | -1 = no fix, 0 = fix, 1 = SBAS, 2 = GBAS |
| `position_covariance` | 3×3 covariance (m²) |

Inspect:

```bash
ros2 topic echo /fix
ros2 topic hz /fix
```

---

## Coordinate Frames & Transforms

To fuse GPS data with other sensors in a local Cartesian frame:

1. **`navsat_transform_node`** (from `robot_localization`) converts
   `NavSatFix` → local `nav_msgs/msg/Odometry`.
2. Publishes a transform from a world frame (e.g., `map`) to `gps_link`.

```bash
sudo apt install ros-humble-robot-localization
ros2 launch robot_localization navsat_transform.launch.py
```

Configure through YAML parameters — see the
[robot_localization docs](https://docs.ros.org/en/humble/p/robot_localization/).

---

## QoS Considerations

GPS data is typically low-frequency (1–10 Hz). A **Reliable / Transient Local**
QoS profile is often appropriate so late-joining subscribers receive the last
known fix:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

gps_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
self.create_subscription(NavSatFix, '/fix', self.cb, gps_qos)
```

---

## Visualizing GPS Data

### mapviz (ROS 2 port)

```bash
sudo apt install ros-humble-mapviz ros-humble-mapviz-plugins \
  ros-humble-tile-map
ros2 launch mapviz mapviz.launch.py
```

Add a **NavSatFix** plugin and point it at `/fix`.

### rviz2

After converting to local coordinates (see above), add an **Odometry** or
**Path** display in `rviz2` to visualize the trajectory.

---

## Recording Data

```bash
ros2 bag record /fix /vel /time_reference -o gps_session
```

---

## Converting to Local Coordinates

Offline UTM conversion example:

```python
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import NavSatFix
from pyproj import Proj

reader = SequentialReader()
reader.open(
    StorageOptions(uri='gps_session', storage_id='mcap'),
    ConverterOptions(input_serialization_format='cdr',
                     output_serialization_format='cdr'),
)

proj = Proj(proj='utm', zone=17, ellps='WGS84')  # adjust zone

while reader.has_next():
    topic, data, timestamp = reader.read_next()
    if topic == '/fix':
        msg = deserialize_message(data, NavSatFix)
        easting, northing = proj(msg.longitude, msg.latitude)
        print(f"E={easting:.2f}  N={northing:.2f}  alt={msg.altitude:.2f}")
```

---

## Troubleshooting

| Symptom | Possible Cause | Fix |
|---------|---------------|-----|
| No data on `/fix` | Serial port permission | Add user to `dialout` group |
| `status = -1` (no fix) | Weak sky view | Move outdoors; wait for satellite lock |
| Large covariance | Single-point GPS | Use RTK / SBAS for better accuracy |
| QoS mismatch | Publisher/subscriber disagree | Check with `ros2 topic info --verbose` |

---

## Further Reading

- [nmea_navsat_driver ROS 2 (GitHub)](https://github.com/ros-drivers/nmea_navsat_driver/tree/ros2)
- [u-blox ROS 2 driver (GitHub)](https://github.com/KumarRobotics/ublox/tree/ros2)
- [robot_localization ROS 2](https://docs.ros.org/en/humble/p/robot_localization/)
- [sensor_msgs/msg/NavSatFix](https://docs.ros.org/en/humble/p/sensor_msgs/interfaces/msg/NavSatFix.html)
- [mapviz ROS 2](https://github.com/swri-robotics/mapviz)
