# ROS 1 vs ROS 2

A concise comparison of the **Robot Operating System** versions to help you
choose the right one for your sensor-integration project.

---

## Table of Contents

1. [Background](#background)
2. [Key Differences at a Glance](#key-differences-at-a-glance)
3. [Architecture](#architecture)
4. [Communication Middleware](#communication-middleware)
5. [Build System](#build-system)
6. [Launch System](#launch-system)
7. [Lifecycle & Node Management](#lifecycle--node-management)
8. [Platform & OS Support](#platform--os-support)
9. [Security](#security)
10. [Ecosystem & Community](#ecosystem--community)
11. [Which Should I Use?](#which-should-i-use)
12. [Further Reading](#further-reading)

---

## Background

**ROS 1** (first release: 2007) established the dominant middleware layer for
robotics research.  
**ROS 2** (first stable release: 2017, LTS *Humble Hawksbill* 2022) is a
ground-up redesign that targets production-grade, real-time, and multi-robot
systems while keeping the core concepts—nodes, topics, services, and
actions—familiar.

> **Tip:** ROS 1 reached end-of-life with **Noetic Ninjemys** (May 2025).
> All new projects are encouraged to start on ROS 2.

---

## Key Differences at a Glance

| Feature | ROS 1 | ROS 2 |
|---------|-------|-------|
| Middleware | Custom TCP/UDP (`TCPROS`/`UDPROS`) | DDS (Data Distribution Service) |
| Master / Discovery | Central `roscore` required | Decentralized peer-to-peer discovery |
| Supported OS | Ubuntu Linux (primary) | Linux, macOS, Windows |
| Real-time support | Not designed for it | First-class real-time support |
| Build tool | `catkin` / `catkin_make` | `colcon` / `ament` |
| Launch files | XML only | Python, XML, or YAML |
| Node composition | One node per process | Multiple nodes per process (components) |
| Lifecycle nodes | No | Yes — managed state transitions |
| Security | None built-in | DDS-Security (authentication, encryption, access control) |
| Python version | Python 2 (Melodic) / Python 3 (Noetic) | Python 3 only |
| QoS policies | Limited | Full DDS QoS (reliability, durability, deadline, …) |

---

## Architecture

### ROS 1

```
roscore (Master + Parameter Server)
   │
   ├── Node A  ──topic──▶  Node B
   └── Node C  ◀─service─  Node D
```

Every ROS 1 system needs a running `roscore`. Nodes register with the Master,
which brokers connections. If `roscore` dies, new connections cannot be made.

### ROS 2

```
Node A  ──DDS──▶  Node B
Node C  ◀─DDS──  Node D
        (peer-to-peer, no central master)
```

ROS 2 uses **DDS** for peer-to-peer discovery and transport. There is no single
point of failure.

---

## Communication Middleware

| Aspect | ROS 1 | ROS 2 |
|--------|-------|-------|
| Transport | `TCPROS` / `UDPROS` | DDS (multiple vendor implementations) |
| Discovery | Master lookup | DDS multicast / unicast discovery |
| Serialization | Custom (`rosmsg`) | CDR (Common Data Representation) via DDS |
| QoS | Best-effort only | Configurable per-topic (reliable, best-effort, history depth, etc.) |

QoS profiles are especially important for sensor data—you can choose between
low-latency best-effort delivery (e.g., live camera streams) or reliable
delivery (e.g., configuration commands).

---

## Build System

| | ROS 1 | ROS 2 |
|---|-------|-------|
| Workspace tool | `catkin_make` / `catkin build` | `colcon build` |
| Package manifest | `package.xml` (format 1 or 2) | `package.xml` (format 3) |
| CMake macros | `catkin` macros | `ament_cmake` macros |
| Python packages | `setup.py` + `catkin` | `setup.py` / `setup.cfg` + `ament_python` |

---

## Launch System

**ROS 1** uses XML-only launch files:

```xml
<launch>
  <node pkg="velodyne_driver" type="velodyne_node" name="vlp16" output="screen">
    <param name="model" value="VLP16" />
  </node>
</launch>
```

**ROS 2** supports Python, XML, and YAML. Python launch files offer full
programmatic control:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='velodyne_driver',
            executable='velodyne_driver_node',
            name='vlp16',
            output='screen',
            parameters=[{'model': 'VLP16'}],
        ),
    ])
```

---

## Lifecycle & Node Management

ROS 2 introduces **managed (lifecycle) nodes** with well-defined states:

```
Unconfigured → Inactive → Active → Finalized
```

This makes it possible to configure a sensor driver, verify its parameters, and
then activate it—without restarting the entire system. ROS 1 has no equivalent
built-in mechanism.

---

## Platform & OS Support

| OS | ROS 1 | ROS 2 |
|----|-------|-------|
| Ubuntu Linux | ✅ Primary | ✅ Primary |
| macOS | Community / partial | ✅ Tier 2 |
| Windows | Community / partial | ✅ Tier 1 (some distros) |
| RTOS | ❌ | ✅ (micro-ROS) |

---

## Security

ROS 1 has **no built-in security layer**—all traffic is unencrypted and
unauthenticated.

ROS 2 leverages **DDS-Security** to provide:

- **Authentication** – verify node identity.
- **Encryption** – protect data in transit.
- **Access control** – restrict who may publish or subscribe to a topic.

Enable security with the ROS 2 `SROS2` tooling:

```bash
ros2 security create_keystore ~/sros2_keystore
ros2 security create_enclave ~/sros2_keystore /my_node
```

---

## Ecosystem & Community

- Many mature ROS 1 packages have been **ported to ROS 2** (Nav2, MoveIt 2,
  `robot_localization`, sensor drivers for Velodyne, Ouster, etc.).
- The **`ros1_bridge`** package allows ROS 1 and ROS 2 nodes to communicate
  side-by-side during migration.
- New development is overwhelmingly happening on ROS 2.

---

## Which Should I Use?

| Scenario | Recommendation |
|----------|---------------|
| New project, no legacy code | **ROS 2** |
| Existing ROS 1 codebase, hard to migrate | Continue ROS 1 while planning migration; use `ros1_bridge` |
| Real-time or embedded requirements | **ROS 2** (with micro-ROS for MCUs) |
| Multi-robot or fleet systems | **ROS 2** (DDS domain IDs, decentralized discovery) |
| Need Windows or macOS support | **ROS 2** |

---

## Further Reading

- [Official ROS 2 documentation](https://docs.ros.org/en/rolling/)
- [ROS 1 → ROS 2 migration guide](https://docs.ros.org/en/rolling/How-To-Guides/Migrating-from-ROS1.html)
- [ROS 2 design articles](https://design.ros2.org/)
- [ros1_bridge](https://github.com/ros2/ros1_bridge)
