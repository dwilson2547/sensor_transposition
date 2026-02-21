# ROS 1 — GPS Integration

Guide to integrating GPS/GNSS receivers with ROS 1, covering driver setup,
topics, coordinate handling, and data recording.

---

## Table of Contents

1. [Supported Receivers](#supported-receivers)
2. [Installation](#installation)
3. [Launching the Driver](#launching-the-driver)
4. [Topics & Message Types](#topics--message-types)
5. [Coordinate Frames & Transforms](#coordinate-frames--transforms)
6. [Visualizing GPS Data](#visualizing-gps-data)
7. [Recording Data](#recording-data)
8. [Converting to Local Coordinates](#converting-to-local-coordinates)
9. [Troubleshooting](#troubleshooting)
10. [Further Reading](#further-reading)

---

## Supported Receivers

| Receiver | Interface | Package |
|----------|-----------|---------|
| u-blox (NEO-M8, ZED-F9P) | Serial / USB | `ublox` — `sudo apt install ros-noetic-ublox` |
| NMEA-compatible (generic) | Serial | `nmea_navsat_driver` — `sudo apt install ros-noetic-nmea-navsat-driver` |
| Septentrio (mosaic, AsteRx) | Serial / Ethernet | `septentrio_gnss_driver` — build from source |
| Trimble / NovAtel | Serial | `nmea_navsat_driver` or vendor-specific |

---

## Installation

### NMEA driver (works with most receivers)

```bash
sudo apt install ros-noetic-nmea-navsat-driver
```

### u-blox driver

```bash
sudo apt install ros-noetic-ublox
```

### Permissions

Make sure your user can access the serial port:

```bash
sudo usermod -aG dialout $USER
# Log out and back in for the change to take effect
```

---

## Launching the Driver

### NMEA driver

```bash
roslaunch nmea_navsat_driver nmea_serial_driver.launch \
  port:=/dev/ttyUSB0 baud:=115200
```

### u-blox

```bash
roslaunch ublox_gps ublox_device.launch
```

---

## Topics & Message Types

| Topic (typical) | Message Type | Content |
|-----------------|-------------|---------|
| `/fix` or `/gps/fix` | `sensor_msgs/NavSatFix` | Latitude, longitude, altitude, covariance |
| `/vel` or `/gps/vel` | `geometry_msgs/TwistStamped` | Velocity from GPS |
| `/time_reference` | `sensor_msgs/TimeReference` | GPS time |
| `/navsat/nmea_sentence` | `nmea_msgs/Sentence` | Raw NMEA string |

Inspect:

```bash
rostopic echo /fix
rostopic hz /fix
```

### NavSatFix fields

| Field | Description |
|-------|-------------|
| `latitude` | Degrees (WGS-84) |
| `longitude` | Degrees (WGS-84) |
| `altitude` | Metres above the WGS-84 ellipsoid |
| `status.status` | -1 = no fix, 0 = fix, 1 = SBAS, 2 = GBAS |
| `position_covariance` | 3×3 covariance (m²) |

---

## Coordinate Frames & Transforms

GPS coordinates are geodetic (latitude / longitude). To use them alongside
LiDAR or camera data in a local Cartesian frame you need a conversion:

1. **`navsat_transform_node`** (from `robot_localization`) converts
   `NavSatFix` → local `nav_msgs/Odometry`.
2. Publishes a transform from a world frame (e.g., `map`) to `gps_link`.

```bash
sudo apt install ros-noetic-robot-localization
roslaunch robot_localization navsat_transform_template.launch
```

---

## Visualizing GPS Data

### mapviz (map overlay)

```bash
sudo apt install ros-noetic-mapviz ros-noetic-mapviz-plugins \
  ros-noetic-tile-map
roslaunch mapviz mapviz.launch
```

Add a **NavSatFix** plugin pointing at `/fix` to see your position on a map.

### rviz

After converting to local coordinates with `navsat_transform_node`, visualize
the odometry path in **rviz** with an **Odometry** or **Path** display.

---

## Recording Data

```bash
rosbag record /fix /vel /time_reference -O gps_session.bag
```

---

## Converting to Local Coordinates

If you prefer an offline conversion without `robot_localization`, you can use
a UTM projection:

```python
import rosbag
from pyproj import Proj

proj = Proj(proj='utm', zone=17, ellps='WGS84')  # adjust zone

bag = rosbag.Bag('gps_session.bag')
for topic, msg, t in bag.read_messages(topics=['/fix']):
    easting, northing = proj(msg.longitude, msg.latitude)
    print(f"E={easting:.2f}  N={northing:.2f}  alt={msg.altitude:.2f}")
bag.close()
```

---

## Troubleshooting

| Symptom | Possible Cause | Fix |
|---------|---------------|-----|
| No data on `/fix` | Serial port permission | Add user to `dialout` group (see above) |
| `status = -1` (no fix) | Weak sky view | Move outdoors with clear sky; wait for satellite lock |
| Large covariance | Single-point GPS | Use RTK or SBAS for centimetre-level accuracy |
| Time jumps | GPS time vs system time | Enable NTP synchronization or use `chrony` with GPS PPS |

---

## Further Reading

- [nmea_navsat_driver ROS wiki](http://wiki.ros.org/nmea_navsat_driver)
- [ublox ROS wiki](http://wiki.ros.org/ublox)
- [robot_localization navsat_transform](http://wiki.ros.org/robot_localization)
- [sensor_msgs/NavSatFix](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/NavSatFix.html)
- [mapviz](http://wiki.ros.org/mapviz)
