# Synchronizing Multiple LiDAR Sensors

A guide to aligning point clouds from two or more **Velodyne**, **Ouster**, or
**Livox** LiDAR sensors in time and space so that they can be merged into a
single, consistent point cloud using the `sensor_transposition` library.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Hardware Synchronization](#hardware-synchronization)
   - [GPS / PPS Synchronization](#gps--pps-synchronization)
   - [Hardware Trigger Cables](#hardware-trigger-cables)
4. [Software Timestamp Alignment](#software-timestamp-alignment)
   - [Nearest-Frame Matching](#nearest-frame-matching)
   - [Linear Interpolation of Transforms](#linear-interpolation-of-transforms)
5. [Configuring a Multi-LiDAR SensorCollection](#configuring-a-multi-lidar-sensorcollection)
6. [Merging Point Clouds into the Ego Frame](#merging-point-clouds-into-the-ego-frame)
   - [Loading Individual Frames](#loading-individual-frames)
   - [Transforming to the Ego Frame](#transforming-to-the-ego-frame)
   - [Concatenating the Clouds](#concatenating-the-clouds)
7. [Complete End-to-End Example](#complete-end-to-end-example)
8. [Verifying the Merged Cloud](#verifying-the-merged-cloud)
9. [Troubleshooting](#troubleshooting)

---

## Overview

A single LiDAR sensor provides coverage over its own field of view only.
Mounting a second (or third) sensor fills blind spots, extends range, and
improves point density.  Before the clouds can be merged you must solve two
independent problems:

| Problem | Solution |
|---------|----------|
| **Temporal** – each sensor scans at its own clock rate and may not fire at the same instant. | Hardware PPS/GPS sync **or** software nearest-frame matching. |
| **Spatial** – each sensor lives at a different position and orientation on the vehicle. | Extrinsic calibration stored in a `SensorCollection` YAML and applied via `transform_between`. |

This guide addresses both problems.

---

## Prerequisites

### Python environment

```bash
pip install sensor_transposition
```

Or, from the repository source:

```bash
pip install -e ".[dev]"
```

### Additional packages used in examples

```bash
pip install numpy
```

---

## Hardware Synchronization

Hardware synchronization eliminates timing error entirely by forcing all sensors
to begin each rotation (or frame) at the same instant.  This is the recommended
approach for high-accuracy applications such as HD mapping.

### GPS / PPS Synchronization

Both Velodyne and Ouster sensors accept a **PPS (Pulse-Per-Second)** signal from
a GPS receiver together with an NMEA time string.  When connected, the sensor
locks its internal clock to UTC and timestamps every point with sub-microsecond
accuracy.

**Velodyne wiring (VLP-16 / VLP-32C / HDL-32E):**

1. Wire the GPS receiver's PPS output to pin 5 of the Velodyne 12-pin
   interface cable, and the NMEA Tx line (9600 baud) to pin 6.
2. In the Velodyne web interface (`http://192.168.1.201`) navigate to
   **GPS → Synchronization** and verify that the *GPS status* field reads
   `LOCKED` within 60 s of applying power.

**Ouster wiring (OS0 / OS1 / OS2):**

1. Connect a PPS signal (3.3 V or 5 V, 100 ms pulse width) to the Ouster
   SYNC_PULSE_IN pin exposed on the sensor connector.
2. Send an NMEA GGA or RMC sentence to the sensor's UDP configuration port.
3. Confirm synchronization:

   ```bash
   curl http://<SENSOR_IP>/api/v1/sensor/metadata | python3 -m json.tool | grep -i sync
   ```

   Look for `"sync_pulse_in": "LOCKED"`.

**Livox (Mid-360 / HAP / Avia):**

Livox sensors use the Livox SDK2's `timesync` configuration block.  Set
`timesync_type` to `"PTP"` (IEEE 1588) in `config.json` to synchronize all
sensors on the same network switch to a common PTP grand-master clock.

> **Tip:** When all sensors share the same PTP or GPS/PPS time source, their
> per-point timestamps are directly comparable.  You can then match frames by
> choosing the frame whose start timestamp is closest to a desired epoch.

---

### Hardware Trigger Cables

Some deployments use a microcontroller or function generator to send a GPIO
pulse simultaneously to each sensor's **trigger input**, forcing all sensors to
begin a new rotation at the same instant.

- Velodyne sensors do **not** support an external rotation trigger; use GPS/PPS
  instead.
- Ouster sensors can be triggered externally via `SYNC_PULSE_IN` when
  `operating_mode` is set to `"STANDBY"` and `multipurpose_io_mode` to
  `"INPUT_NMEA_UART"` or `"INPUT_TRIGGER"` via the HTTP configuration API.
- Livox Mid-360 / HAP support trigger-based synchronization through the
  SDK2 `lidar_cfg` parameter `external_trigger_enable`.

---

## Software Timestamp Alignment

When hardware synchronization is not available, you can align frames in
software by matching each sensor's frames by timestamp.

### Nearest-Frame Matching

The simplest approach: for each frame from the *primary* sensor, pick the
frame from every *secondary* sensor whose timestamp is closest in time.

```python
"""
nearest_frame_match.py
Match frames from two sensors by closest timestamp.
"""
import bisect


def nearest_frame(primary_ts: float, secondary_timestamps: list[float]) -> int:
    """Return the index in *secondary_timestamps* closest to *primary_ts*."""
    idx = bisect.bisect_left(secondary_timestamps, primary_ts)

    if idx == 0:
        return 0
    if idx == len(secondary_timestamps):
        return len(secondary_timestamps) - 1

    before = secondary_timestamps[idx - 1]
    after  = secondary_timestamps[idx]
    return idx - 1 if abs(primary_ts - before) <= abs(primary_ts - after) else idx


# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------

# Timestamps in seconds for each sensor's frames
front_lidar_ts = [0.00, 0.10, 0.20, 0.30, 0.40]   # 10 Hz
rear_lidar_ts  = [0.00, 0.12, 0.24, 0.36, 0.48]   # ~8.3 Hz (slightly offset)

for primary_idx, ts in enumerate(front_lidar_ts):
    secondary_idx = nearest_frame(ts, rear_lidar_ts)
    dt_ms = abs(ts - rear_lidar_ts[secondary_idx]) * 1000
    print(f"Front frame {primary_idx} (t={ts:.2f} s) → "
          f"Rear frame {secondary_idx} (t={rear_lidar_ts[secondary_idx]:.2f} s, "
          f"Δt={dt_ms:.1f} ms)")
```

> **Acceptable time difference:** For a vehicle travelling at 30 m/s (108 km/h)
> a 10 ms timestamp mismatch introduces up to 0.3 m of positional error in
> the merged cloud.  Keep Δt below 5 ms for HD-map-quality results.

### Linear Interpolation of Transforms

When the vehicle is moving, even a small timestamp difference causes the two
clouds to be captured from slightly different vehicle poses.  If you have
odometry or IMU data, you can compute a pose for each sensor at a common
reference time and use that pose to motion-correct the secondary cloud before
merging.

```python
import numpy as np
from sensor_transposition.transform import Transform


def interpolate_transform(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    """Linearly interpolate between two 4×4 homogeneous transforms.

    Args:
        T0: Transform at time t0 (4×4 numpy array).
        T1: Transform at time t1 (4×4 numpy array).
        alpha: Interpolation factor in [0, 1].  0 → T0, 1 → T1.

    Returns:
        Interpolated 4×4 numpy array.

    Note:
        This uses simple linear interpolation of the matrix entries, which is
        only accurate for small rotations.  For large rotations use SLERP on
        the rotation component.
    """
    return (1.0 - alpha) * T0 + alpha * T1


# Example: front_lidar captures at t=0.00 s, rear_lidar at t=0.03 s.
# We have ego poses at t=0.00 and t=0.10 (10 Hz odometry).
# Motion-correct the rear cloud to t=0.00.

t_front   = 0.00   # reference time
t_rear    = 0.03   # rear scan time
t0, t1    = 0.00, 0.10  # odometry timestamps bracketing t_rear

# T_ego_at_t0 and T_ego_at_t1 would come from your odometry source
T_ego_t0 = np.eye(4)                           # placeholder: ego at t=0.00
T_ego_t1 = Transform.from_translation([0.3, 0.0, 0.0]).matrix  # 0.3 m forward

alpha = (t_rear - t0) / (t1 - t0)
T_ego_at_rear = interpolate_transform(T_ego_t0, T_ego_t1, alpha)

# Transform that moves points from their captured pose back to the reference pose
T_correction = np.linalg.inv(T_ego_t0) @ T_ego_at_rear
```

---

## Configuring a Multi-LiDAR SensorCollection

Use a YAML file to record the extrinsic pose (position + orientation relative
to the vehicle ego frame) of every LiDAR sensor.  The `SensorCollection` class
reads this file and provides the transforms needed to merge point clouds.

```yaml
# multi_lidar.yaml
#
# Ego frame: origin at the centre of the rear axle projected onto the ground
# plane.  +x forward, +y left, +z up (FLU convention).

sensors:
  front_lidar:
    type: lidar
    coordinate_system: FLU
    extrinsics:
      translation: [1.84, 0.00, 1.91]   # roof-mounted, 1.84 m ahead, 1.91 m up
      rotation:
        quaternion: [1.0, 0.0, 0.0, 0.0]  # aligned with ego frame

  rear_lidar:
    type: lidar
    coordinate_system: FLU
    extrinsics:
      translation: [-0.50, 0.00, 1.85]  # 0.5 m behind the rear axle, 1.85 m up
      rotation:
        quaternion: [0.0, 0.0, 0.0, 1.0]  # 180° yaw – sensor points backward

  left_lidar:
    type: lidar
    coordinate_system: FLU
    extrinsics:
      translation: [0.00, 0.90, 1.50]   # 0.9 m to the left, 1.5 m up
      rotation:
        quaternion: [0.7071, 0.0, 0.0, 0.7071]  # 90° yaw – sensor points left
```

Load the collection in Python:

```python
from sensor_transposition.sensor_collection import SensorCollection

collection = SensorCollection.from_yaml("multi_lidar.yaml")

print("Sensors:", collection.sensor_names)
# → Sensors: ['front_lidar', 'left_lidar', 'rear_lidar']
```

---

## Merging Point Clouds into the Ego Frame

### Loading Individual Frames

```python
import numpy as np
from sensor_transposition.lidar.velodyne import VelodyneParser   # or OusterParser / LivoxParser

front_parser = VelodyneParser("front_lidar/frame_000050.bin")
rear_parser  = VelodyneParser("rear_lidar/frame_000050.bin")

front_xyz = front_parser.xyz()  # (N, 3) in the front_lidar frame
rear_xyz  = rear_parser.xyz()   # (M, 3) in the rear_lidar frame
```

### Transforming to the Ego Frame

```python
from sensor_transposition.sensor_collection import SensorCollection
from sensor_transposition.transform import Transform

collection = SensorCollection.from_yaml("multi_lidar.yaml")

# Retrieve 4×4 homogeneous transform matrices: sensor frame → ego frame
T_front_to_ego = collection.transform_to_ego("front_lidar")  # (4, 4)
T_rear_to_ego  = collection.transform_to_ego("rear_lidar")   # (4, 4)

# Apply transforms using the Transform helper
front_in_ego = Transform(T_front_to_ego).apply_to_points(front_xyz)  # (N, 3)
rear_in_ego  = Transform(T_rear_to_ego).apply_to_points(rear_xyz)    # (M, 3)
```

### Concatenating the Clouds

```python
merged = np.vstack([front_in_ego, rear_in_ego])  # (N+M, 3)
print(f"Merged cloud: {len(merged)} points")
```

To keep intensity or other per-point attributes alongside the XYZ coordinates,
stack them together before merging:

```python
front_xyzi = front_parser.xyz_intensity()  # (N, 4) – x, y, z, intensity
rear_xyzi  = rear_parser.xyz_intensity()   # (M, 4)

front_xyz_ego  = Transform(T_front_to_ego).apply_to_points(front_xyzi[:, :3])
rear_xyz_ego   = Transform(T_rear_to_ego).apply_to_points(rear_xyzi[:, :3])

# Re-attach intensity after transforming coordinates
front_merged = np.hstack([front_xyz_ego, front_xyzi[:, 3:]])
rear_merged  = np.hstack([rear_xyz_ego,  rear_xyzi[:, 3:]])

merged_xyzi = np.vstack([front_merged, rear_merged])  # (N+M, 4)
```

---

## Complete End-to-End Example

The script below shows the full pipeline: load two Velodyne frames, align them
to the nearest timestamp, transform both clouds into the ego frame, and save
the merged result.

```python
"""
merge_lidar_frames.py

Merge synchronized point clouds from a front and rear Velodyne LiDAR into
a single ego-frame cloud.

Usage:
    python merge_lidar_frames.py

Requires:
    sensor_transposition, numpy
"""

import bisect
from pathlib import Path

import numpy as np

from sensor_transposition.lidar.velodyne import VelodyneParser
from sensor_transposition.sensor_collection import SensorCollection
from sensor_transposition.transform import Transform

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

YAML_PATH        = "multi_lidar.yaml"
FRONT_LIDAR_DIR  = Path("front_lidar")
REAR_LIDAR_DIR   = Path("rear_lidar")
OUTPUT_DIR       = Path("merged_frames")

# Timestamps (seconds) for each sensor's frames; in a real system these
# come from the file metadata or a sidecar timestamp file.
FRONT_TIMESTAMPS = [i * 0.10 for i in range(50)]   # 10 Hz, starting at 0 s
REAR_TIMESTAMPS  = [i * 0.10 + 0.012 for i in range(50)]  # 10 Hz, 12 ms offset

OUTPUT_DIR.mkdir(exist_ok=True)


def nearest_frame_idx(ts: float, timestamps: list) -> int:
    """Return the index in *timestamps* whose value is closest to *ts*."""
    idx = bisect.bisect_left(timestamps, ts)
    if idx == 0:
        return 0
    if idx == len(timestamps):
        return len(timestamps) - 1
    before, after = timestamps[idx - 1], timestamps[idx]
    return idx - 1 if abs(ts - before) <= abs(ts - after) else idx


# ------------------------------------------------------------------
# Load extrinsic configuration
# ------------------------------------------------------------------

collection = SensorCollection.from_yaml(YAML_PATH)

T_front_to_ego = collection.transform_to_ego("front_lidar")
T_rear_to_ego  = collection.transform_to_ego("rear_lidar")

# ------------------------------------------------------------------
# Process frames
# ------------------------------------------------------------------

for front_idx, front_ts in enumerate(FRONT_TIMESTAMPS):
    rear_idx = nearest_frame_idx(front_ts, REAR_TIMESTAMPS)
    dt_ms    = abs(front_ts - REAR_TIMESTAMPS[rear_idx]) * 1000

    front_bin = FRONT_LIDAR_DIR / f"frame_{front_idx:06d}.bin"
    rear_bin  = REAR_LIDAR_DIR  / f"frame_{rear_idx:06d}.bin"

    if not front_bin.exists() or not rear_bin.exists():
        continue   # skip missing frames

    # Load point clouds
    front_xyzi = VelodyneParser(front_bin).xyz_intensity()  # (N, 4)
    rear_xyzi  = VelodyneParser(rear_bin).xyz_intensity()   # (M, 4)

    # Transform coordinates to ego frame
    front_ego = Transform(T_front_to_ego).apply_to_points(front_xyzi[:, :3])
    rear_ego  = Transform(T_rear_to_ego).apply_to_points(rear_xyzi[:, :3])

    # Reassemble with intensity and merge
    merged = np.vstack([
        np.hstack([front_ego, front_xyzi[:, 3:]]),
        np.hstack([rear_ego,  rear_xyzi[:, 3:]]),
    ])

    out_path = OUTPUT_DIR / f"frame_{front_idx:06d}.bin"
    merged.astype(np.float32).tofile(out_path)

    print(f"Frame {front_idx:04d}  front={len(front_xyzi):6d} pts  "
          f"rear={len(rear_xyzi):6d} pts  Δt={dt_ms:.1f} ms  "
          f"→ {len(merged):6d} pts  {out_path}")
```

---

## Verifying the Merged Cloud

After merging, confirm that the two clouds are geometrically consistent: points
near the boundary between the two sensor volumes should overlap smoothly, with
no visible seam or double-wall artifact.

```python
import numpy as np
from sensor_transposition.lidar.velodyne import VelodyneParser
from sensor_transposition.sensor_collection import SensorCollection
from sensor_transposition.transform import Transform

collection     = SensorCollection.from_yaml("multi_lidar.yaml")
T_front        = collection.transform_to_ego("front_lidar")
T_rear         = collection.transform_to_ego("rear_lidar")

front_xyz      = VelodyneParser("front_lidar/frame_000000.bin").xyz()
rear_xyz       = VelodyneParser("rear_lidar/frame_000000.bin").xyz()

front_ego      = Transform(T_front).apply_to_points(front_xyz)
rear_ego       = Transform(T_rear).apply_to_points(rear_xyz)
merged         = np.vstack([front_ego, rear_ego])

# Basic statistics
for label, pts in [("Front", front_ego), ("Rear", rear_ego), ("Merged", merged)]:
    print(f"{label:8s}  N={len(pts):7d}  "
          f"X=[{pts[:,0].min():6.1f}, {pts[:,0].max():6.1f}] m  "
          f"Y=[{pts[:,1].min():6.1f}, {pts[:,1].max():6.1f}] m  "
          f"Z=[{pts[:,2].min():6.1f}, {pts[:,2].max():6.1f}] m")

# Sanity checks
assert not np.any(np.isnan(merged)),  "NaN values in merged cloud"
assert not np.any(np.isinf(merged)),  "Inf values in merged cloud"
assert len(merged) == len(front_ego) + len(rear_ego), "Point count mismatch"
```

**Expected output for a typical outdoor scene:**

| Cloud   | Points      | X range         | Y range         |
|---------|-------------|-----------------|-----------------|
| Front   | 30 000–70 000 | 0 m to +120 m  | −50 m to +50 m  |
| Rear    | 20 000–50 000 | −60 m to +10 m | −40 m to +40 m  |
| Merged  | 50 000–120 000 | −60 m to +120 m | −50 m to +50 m |

---

## Troubleshooting

### Double-wall artifacts at the sensor boundary

Points near the overlap region appear twice, offset from each other.  The most
common causes are:

- **Incorrect extrinsics** – re-measure the sensor mounting positions and
  update `multi_lidar.yaml`.
- **Large timestamp offset** – reduce Δt by using hardware PPS sync or by
  motion-correcting the secondary cloud (see
  [Linear Interpolation of Transforms](#linear-interpolation-of-transforms)).

### One cloud is rotated or mirrored relative to the other

Check the `coordinate_system` field and the `rotation` quaternion in the YAML.
A common mistake is setting the rear sensor's yaw to `90°` instead of `180°`.
For a rear-facing sensor that is physically rotated 180° around the vertical
axis the correct quaternion is `[0.0, 0.0, 0.0, 1.0]` (180° around Z in the
`[w, x, y, z]` convention).

### `KeyError: Sensor 'X' not found in collection`

The sensor name in your Python code does not match the key in the YAML file.
Print `collection.sensor_names` to see the exact names loaded:

```python
from sensor_transposition.sensor_collection import SensorCollection
collection = SensorCollection.from_yaml("multi_lidar.yaml")
print(collection.sensor_names)
```

### Merged cloud has far fewer points than expected

One or more `.bin` files may be empty or contain only a few points.  Check that
the conversion pipeline completed successfully and that the file sizes are
non-zero:

```bash
ls -lh front_lidar/ rear_lidar/
```

Reload the raw cloud and check the point count before transformation:

```python
from sensor_transposition.lidar.velodyne import VelodyneParser
print(len(VelodyneParser("rear_lidar/frame_000000.bin").read()))
```

### Timestamp offsets larger than expected

- **Velodyne / Ouster:** Confirm PPS lock in the sensor web interface.  If no
  GPS is available, synchronize both host computers to the same NTP server.
- **Livox:** Verify that all sensors and the host PC are on the same PTP domain
  and that the PTP grand master is reachable.
