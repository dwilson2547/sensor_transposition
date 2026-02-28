# Rosbag / MCAP Recording and Playback

## Overview

`sensor_transposition.rosbag` provides a lightweight, self-contained bag
file format for **recording** and **replaying** timestamped multi-sensor
data — the data-capture primitive needed before any offline SLAM pipeline
can run.

The format is inspired by ROS bag / MCAP concepts but requires **no ROS
installation** and **no external Python dependencies** beyond the standard
library.  All message payloads are serialised as UTF-8 JSON, so bag files
can be inspected with any text editor (after skipping the binary header).

---

## Why a Bag File?

In a typical SLAM workflow the sensors are first driven in the real world
and all their outputs are **captured to disk**.  The recorded data is then
replayed offline — potentially many times — while algorithms are developed
and tuned.  A bag file:

* **Decouples recording from processing** — sensors run at full rate; the
  algorithm can replay at any speed.
* **Enables reproducibility** — the same raw data drives every experiment.
* **Supports multi-sensor fusion** — all streams share a common timestamp
  axis so they can be aligned by
  [`SensorSynchroniser`](../sensor_transposition/sync.py).

---

## Quick Start

### Recording

```python
from sensor_transposition.rosbag import BagWriter

with BagWriter("session.sbag") as bag:
    # LiDAR scan (convert NumPy array to nested list for JSON serialisation)
    bag.write("/lidar/points",  frame_pose.timestamp,
              {"xyz": lidar_scan.tolist()})

    # Ego pose from scan-matching
    bag.write("/pose/ego",      frame_pose.timestamp,
              frame_pose.to_dict())

    # Raw IMU measurement
    bag.write("/imu/data",      imu_timestamp,
              {"accel": accel.tolist(), "gyro": gyro.tolist()})

    # GPS fix
    bag.write("/gps/fix",       gps_timestamp,
              {"lat": lat, "lon": lon, "alt": alt})
```

### Playback

```python
from sensor_transposition.rosbag import BagReader

with BagReader("session.sbag") as bag:
    print("Topics  :", bag.topics)
    print("Messages:", bag.message_count)
    print("Duration:", bag.end_time - bag.start_time, "s")

    for msg in bag.read_messages(topics=["/pose/ego"]):
        print(msg.topic, msg.timestamp, msg.data)
```

---

## Filtering Playback

`read_messages` supports three independent filter parameters that can be
combined freely:

| Parameter | Type | Effect |
|-----------|------|--------|
| `topics` | `str` or `list[str]` | Only yield messages whose topic is in this set. |
| `start_time` | `float` | Skip messages with `timestamp < start_time`. |
| `end_time` | `float` | Skip messages with `timestamp > end_time`. |

```python
# Only LiDAR messages in the second minute of the recording.
for msg in bag.read_messages(topics="/lidar/points",
                              start_time=60.0, end_time=120.0):
    process(msg.data)
```

---

## Integration with sensor_transposition

### Replaying poses into `FramePoseSequence`

```python
from sensor_transposition.frame_pose import FramePose, FramePoseSequence
from sensor_transposition.rosbag import BagReader

sequence = FramePoseSequence()

with BagReader("session.sbag") as bag:
    for msg in bag.read_messages(topics="/pose/ego"):
        sequence.add_pose(FramePose.from_dict(msg.data))
```

### Re-synchronising streams with `SensorSynchroniser`

```python
import numpy as np
from sensor_transposition.sync import SensorSynchroniser
from sensor_transposition.rosbag import BagReader

lidar_times, lidar_data = [], []
imu_times,   imu_data   = [], []

with BagReader("session.sbag") as bag:
    for msg in bag.read_messages(topics="/lidar/points"):
        lidar_times.append(msg.timestamp)
        lidar_data.append(msg.data["xyz"])

    for msg in bag.read_messages(topics="/imu/data"):
        imu_times.append(msg.timestamp)
        imu_data.append(msg.data["accel"])

sync = SensorSynchroniser()
sync.add_stream("lidar", lidar_times, np.array(lidar_data))
sync.add_stream("imu",   imu_times,   np.array(imu_data))
aligned = sync.synchronise(lidar_times)
```

---

## File Format

Every `.sbag` file starts with a 10-byte header:

```
[MAGIC  : 9 bytes] 0x89 'S' 'B' 'A' 'G' 0x0D 0x0A 0x1A 0x0A
[VERSION: 1 byte ] 0x01
```

Each message is encoded as a sequential record:

```
record_body_size : uint32 LE  — byte count of the remaining fields
topic_len        : uint16 LE  — byte length of the topic string
topic            : topic_len bytes, UTF-8
timestamp        : float64 LE — seconds on the reference clock
payload          : (record_body_size - 2 - topic_len - 8) bytes, UTF-8 JSON
```

A 4-byte sentinel `SEND` is appended on clean close.  Readers tolerate
files without the sentinel (e.g. from a crash during recording).

---

## API Reference

### `BagMessage`

| Attribute | Type | Description |
|-----------|------|-------------|
| `topic` | `str` | Channel name (e.g. `"/lidar/points"`). |
| `timestamp` | `float` | Message time in seconds (reference clock). |
| `data` | `dict` | JSON-serialisable payload. |

---

### `BagWriter`

| Method | Description |
|--------|-------------|
| `BagWriter(path)` | Create (or truncate) a bag file at *path*. |
| `write(topic, timestamp, data)` | Append one message. |
| `close()` | Flush, write footer, and close the file. |
| `message_count` | Number of messages written so far. |

Supports the context-manager protocol (`with BagWriter(...) as bag`).

---

### `BagReader`

| Property / Method | Description |
|-------------------|-------------|
| `BagReader(path)` | Open a bag file and build the in-memory index. |
| `topics` | Sorted list of unique topic names. |
| `message_count` | Total number of messages in the bag. |
| `start_time` | Timestamp of the earliest message. |
| `end_time` | Timestamp of the latest message. |
| `topic_message_counts` | `dict` mapping topic → message count. |
| `read_messages(topics, start_time, end_time)` | Iterate filtered messages. |
| `close()` | Close the underlying file. |

Supports the context-manager protocol (`with BagReader(...) as bag`).

---

## SLAM Pipeline Integration

```
Sensor drivers (LiDAR, camera, IMU, GPS, radar)
        │
        ▼  (real-time)
 BagWriter.write()              ← rosbag.py
        │
        ▼  (file on disk)
 BagReader.read_messages()      ← rosbag.py
        │
        ├─────────────────────────────────────────────┐
        ▼                                             ▼
 SensorSynchroniser()           ← sync.py     FramePoseSequence
        │
        ▼
 ImuPreintegrator / icp_align / estimate_essential_matrix
        │
        ▼
 PoseGraph + optimize_pose_graph()
        │
        ▼
 PointCloudMap
```

---

## References

* [ROS bag file format](http://wiki.ros.org/Bags/Format/2.0) — the
  production format that inspired this lightweight implementation.
* [MCAP format](https://mcap.dev/) — a modern, schema-aware multi-channel
  logging format for robotics.
* [`sensor_transposition.sync`](../sensor_transposition/sync.py) —
  multi-sensor time synchronisation and interpolation.
* [`sensor_transposition.frame_pose`](../sensor_transposition/frame_pose.py)
  — trajectory storage and YAML I/O.
