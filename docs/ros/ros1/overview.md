# ROS 1 Overview

A high-level introduction to **ROS 1** (Robot Operating System) covering what it
is, core concepts, sensor integration, common commands, and setup instructions.

---

## Table of Contents

1. [What Is ROS 1?](#what-is-ros-1)
2. [Core Concepts](#core-concepts)
3. [Setup & Installation](#setup--installation)
4. [Workspace & Package Basics](#workspace--package-basics)
5. [Common Commands](#common-commands)
6. [Sensor Integration at a Glance](#sensor-integration-at-a-glance)
7. [Data Collection with rosbag](#data-collection-with-rosbag)
8. [Further Reading](#further-reading)

---

## What Is ROS 1?

ROS 1 is an open-source **middleware framework** for robot software development.
It provides:

- A **publish / subscribe** messaging layer (topics).
- A **request / reply** pattern (services).
- A **parameter server** for shared configuration.
- Tools for recording, replaying, and visualizing data (`rosbag`, `rviz`,
  `rqt`).

> **Note:** The final ROS 1 distribution is **Noetic Ninjemys** (Ubuntu 20.04,
> Python 3), which reached end-of-life in May 2025. New projects should
> consider [ROS 2](../ros2/overview.md).

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Node** | A single-purpose process (e.g., a LiDAR driver, an IMU filter). |
| **Topic** | A named bus for streaming messages (many-to-many pub/sub). |
| **Message** | A typed data structure (e.g., `sensor_msgs/PointCloud2`). |
| **Service** | A synchronous request/reply call between two nodes. |
| **Action** | An asynchronous, preemptable goal/feedback/result pattern. |
| **Master (`roscore`)** | Central registry that all nodes must connect to. |
| **Parameter Server** | Key/value store shared across nodes (hosted by the Master). |

---

## Setup & Installation

The recommended way to install ROS 1 Noetic on **Ubuntu 20.04**:

```bash
# 1. Configure the apt repository
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" \
  > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
  | sudo apt-key add -

# 2. Install the desktop-full variant
sudo apt update
sudo apt install ros-noetic-desktop-full

# 3. Source the setup script (add to ~/.bashrc for persistence)
source /opt/ros/noetic/setup.bash

# 4. Install build tools
sudo apt install python3-rosdep python3-rosinstall python3-catkin-tools
sudo rosdep init
rosdep update
```

> For the full installation walkthrough see the
> [official ROS 1 Noetic install guide](http://wiki.ros.org/noetic/Installation/Ubuntu).

---

## Workspace & Package Basics

### Create a catkin workspace

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin_make          # or: catkin build
source devel/setup.bash
```

### Create a new package

```bash
cd ~/catkin_ws/src
catkin_create_pkg my_sensor_pkg std_msgs sensor_msgs rospy roscpp
cd ~/catkin_ws
catkin_make
```

---

## Common Commands

| Command | Purpose |
|---------|---------|
| `roscore` | Start the Master, Parameter Server, and logging node. |
| `rosrun <pkg> <node>` | Run a single node. |
| `roslaunch <pkg> <file.launch>` | Launch multiple nodes from a launch file. |
| `rostopic list` | List all active topics. |
| `rostopic echo /topic` | Print messages on a topic in real time. |
| `rostopic hz /topic` | Show the publishing rate of a topic. |
| `rosnode list` | List running nodes. |
| `rosnode info /node` | Show detailed info about a node. |
| `rosparam list` | List all parameters on the Parameter Server. |
| `rosmsg show <MsgType>` | Display the fields of a message type. |
| `rosservice list` | List available services. |
| `rviz` | Open the 3-D visualization tool. |
| `rqt_graph` | Show the node/topic graph. |

---

## Sensor Integration at a Glance

Most hardware sensors in ROS 1 follow the same pattern:

1. **Install the driver package** (e.g., `ros-noetic-velodyne`).
2. **Launch the driver node** which publishes standard `sensor_msgs` messages:
   - `sensor_msgs/PointCloud2` — LiDAR
   - `sensor_msgs/Image` — Camera
   - `sensor_msgs/NavSatFix` — GPS
   - `sensor_msgs/Imu` — IMU
   - `sensor_msgs/Range` or custom — Radar
3. **Record the data** using `rosbag`.
4. **Replay & analyze** with `rosbag play`, `rviz`, or offline scripts.

See the individual sensor guides in the [sensors/](sensors/) folder for
detailed integration steps.

---

## Data Collection with rosbag

`rosbag` is the standard tool for recording and playing back topic data.

### Record

```bash
# Record specific topics
rosbag record /velodyne_points /imu/data /gps/fix -O my_session.bag

# Record everything
rosbag record -a
```

### Inspect

```bash
rosbag info my_session.bag
```

### Replay

```bash
rosbag play my_session.bag
# Play at half speed:
rosbag play -r 0.5 my_session.bag
```

Bag files store the raw serialized messages and can be post-processed with
Python (`rosbag` API) or converted to other formats.

---

## Further Reading

- [ROS 1 Wiki](http://wiki.ros.org/)
- [ROS 1 Tutorials](http://wiki.ros.org/ROS/Tutorials)
- [sensor_msgs documentation](http://wiki.ros.org/sensor_msgs)
- [rosbag command-line reference](http://wiki.ros.org/rosbag/Commandline)
- [rviz user guide](http://wiki.ros.org/rviz/UserGuide)
