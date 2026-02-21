# ROS 2 Overview

A high-level introduction to **ROS 2** covering what it is, core concepts,
sensor integration, common commands, and setup instructions.

---

## Table of Contents

1. [What Is ROS 2?](#what-is-ros-2)
2. [Core Concepts](#core-concepts)
3. [Setup & Installation](#setup--installation)
4. [Workspace & Package Basics](#workspace--package-basics)
5. [Common Commands](#common-commands)
6. [Quality of Service (QoS)](#quality-of-service-qos)
7. [Sensor Integration at a Glance](#sensor-integration-at-a-glance)
8. [Data Collection with ros2 bag](#data-collection-with-ros2-bag)
9. [Further Reading](#further-reading)

---

## What Is ROS 2?

ROS 2 is the next-generation **open-source middleware framework** for robotics.
Built on top of **DDS** (Data Distribution Service), it provides:

- **Decentralized** peer-to-peer discovery — no central master required.
- Configurable **Quality of Service** (QoS) per topic.
- First-class **real-time**, **multi-platform**, and **security** support.
- Publish/subscribe (topics), request/reply (services), and goal-based
  (actions) communication — same concepts as ROS 1, modernized.

> **Current LTS:** *Humble Hawksbill* (May 2022 – May 2027, Ubuntu 22.04).
> *Jazzy Jalisco* (May 2024, Ubuntu 24.04) is the latest LTS.

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Node** | A single-purpose process (e.g., a LiDAR driver, an IMU filter). |
| **Topic** | A named bus for streaming messages (many-to-many pub/sub). |
| **Message** | A typed data structure defined in `.msg` / `.idl` files. |
| **Service** | A synchronous request/reply call between two nodes. |
| **Action** | An asynchronous goal/feedback/result pattern. |
| **Lifecycle Node** | A managed node with well-defined states (unconfigured → active → finalized). |
| **Component** | A node that can be loaded into a shared process (composable). |
| **QoS Profile** | Per-topic settings for reliability, durability, deadline, etc. |
| **Parameter** | Per-node key/value configuration (no global Parameter Server). |

---

## Setup & Installation

### Ubuntu 22.04 — Humble Hawksbill

```bash
# 1. Set locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# 2. Add the ROS 2 apt repository
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu jammy main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 3. Install desktop-full
sudo apt update
sudo apt install ros-humble-desktop

# 4. Source the setup script (add to ~/.bashrc for persistence)
source /opt/ros/humble/setup.bash

# 5. Install build tools
sudo apt install python3-colcon-common-extensions python3-rosdep
sudo rosdep init
rosdep update
```

> For other platforms and distributions see the
> [official ROS 2 installation guides](https://docs.ros.org/en/humble/Installation.html).

---

## Workspace & Package Basics

### Create a colcon workspace

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build
source install/setup.bash
```

### Create a new Python package

```bash
cd ~/ros2_ws/src
ros2 pkg create my_sensor_pkg --build-type ament_python \
  --dependencies rclpy sensor_msgs
cd ~/ros2_ws
colcon build --packages-select my_sensor_pkg
```

### Create a new C++ package

```bash
cd ~/ros2_ws/src
ros2 pkg create my_sensor_pkg --build-type ament_cmake \
  --dependencies rclcpp sensor_msgs
```

---

## Common Commands

| Command | Purpose |
|---------|---------|
| `ros2 run <pkg> <executable>` | Run a single node. |
| `ros2 launch <pkg> <file>` | Launch multiple nodes from a launch file. |
| `ros2 topic list` | List all active topics. |
| `ros2 topic echo /topic` | Print messages on a topic. |
| `ros2 topic hz /topic` | Show the publishing rate. |
| `ros2 node list` | List running nodes. |
| `ros2 node info /node` | Show detailed info about a node. |
| `ros2 param list` | List parameters for running nodes. |
| `ros2 interface show <MsgType>` | Display the fields of a message type. |
| `ros2 service list` | List available services. |
| `ros2 bag record` | Record topic data to an `mcap` / `sqlite3` bag. |
| `ros2 bag play <bag_dir>` | Replay recorded data. |
| `rviz2` | Open the 3-D visualization tool. |
| `rqt_graph` | Show the node/topic computation graph. |

---

## Quality of Service (QoS)

QoS profiles let you tune per-topic communication behaviour. Key policies:

| Policy | Options | Typical sensor use |
|--------|---------|--------------------|
| **Reliability** | `RELIABLE` / `BEST_EFFORT` | Camera streams → best-effort; config services → reliable |
| **Durability** | `VOLATILE` / `TRANSIENT_LOCAL` | Use transient-local for latched topics (e.g., static transforms) |
| **History** | `KEEP_LAST(N)` / `KEEP_ALL` | LiDAR at 10 Hz → keep_last(5) is often sufficient |
| **Deadline** | duration | Detect stale sensor data |

Set QoS when creating a subscriber:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy

qos = QoSProfile(
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)
self.create_subscription(PointCloud2, '/lidar_points', self.cb, qos)
```

---

## Sensor Integration at a Glance

The same pattern applies to most hardware sensors in ROS 2:

1. **Install the driver package** (e.g., `ros-humble-velodyne`).
2. **Launch the driver node** which publishes standard `sensor_msgs` messages:
   - `sensor_msgs/msg/PointCloud2` — LiDAR
   - `sensor_msgs/msg/Image` — Camera
   - `sensor_msgs/msg/NavSatFix` — GPS
   - `sensor_msgs/msg/Imu` — IMU
   - `radar_msgs/msg/RadarScan` (or custom) — Radar
3. **Record the data** using `ros2 bag record`.
4. **Replay & analyze** with `ros2 bag play`, `rviz2`, or offline scripts.

See the individual sensor guides in the [sensors/](sensors/) folder for
detailed integration steps.

---

## Data Collection with ros2 bag

`ros2 bag` is the built-in tool for recording and playing back topic data.

### Record

```bash
# Record specific topics
ros2 bag record /lidar_points /imu/data /gps/fix -o my_session

# Record everything
ros2 bag record -a
```

### Inspect

```bash
ros2 bag info my_session
```

### Replay

```bash
ros2 bag play my_session
# Play at half speed:
ros2 bag play my_session --rate 0.5
```

Bag files default to the **MCAP** storage format (Humble+) and can be
post-processed with the Python `rosbag2_py` API or converted to other formats.

---

## Further Reading

- [Official ROS 2 documentation](https://docs.ros.org/en/humble/)
- [ROS 2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)
- [ROS 2 QoS concepts](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html)
- [sensor_msgs package](https://index.ros.org/p/sensor_msgs/)
- [ros2 bag CLI reference](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html)
- [rviz2 user guide](https://docs.ros.org/en/humble/Tutorials/Intermediate/RViz/RViz-User-Guide/RViz-User-Guide-Main.html)
