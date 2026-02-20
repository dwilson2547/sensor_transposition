# Radar Data Collection Guide

A practical how-to guide for capturing real detection data from the
**Continental ARS 408-21** radar sensor without ROS, and loading the results
with the `sensor_transposition` library.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Hardware Setup](#hardware-setup)
4. [Capturing CAN Data](#capturing-can-data)
   - [Linux – using `candump`](#linux--using-candump)
   - [Python – direct CAN capture](#python--direct-can-capture)
5. [Converting CAN Frames to `.bin`](#converting-can-frames-to-bin)
6. [Loading with sensor_transposition](#loading-with-sensor_transposition)
7. [Verifying Your Data](#verifying-your-data)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The Continental ARS 408-21 is a short-to-medium range automotive radar
(76–77 GHz) with a maximum range of approximately 250 m.  It communicates
exclusively over a **CAN bus** at 500 kbps and periodically transmits a list
of tracked objects, each carrying position, relative velocity, and radar cross
section (RCS) information.

Because the sensor uses a 2-D flat-scan geometry, the elevation angle is
always zero.  The table below summarises the mapping from native CAN fields to
the binary format expected by the `sensor_transposition` library:

| CAN field          | `sensor_transposition` field | Notes                                    |
|--------------------|------------------------------|------------------------------------------|
| `Obj_DistLong`     | range                        | Derived together with `Obj_DistLat`      |
| `Obj_DistLat`      | azimuth                      | Derived together with `Obj_DistLong`     |
| *(always zero)*    | elevation                    | ARS 408-21 is a 2-D radar                |
| `Obj_VrelLong`     | velocity                     | Radial (longitudinal) relative velocity  |
| `Obj_RCS`          | snr                          | Used as a signal-strength proxy          |

---

## Prerequisites

### Python environment

```bash
pip install sensor_transposition
```

Or, if you are working from the repository source:

```bash
pip install -e ".[dev]"
```

### CAN libraries

```bash
pip install python-can cantools
```

| Package      | Purpose                                              |
|--------------|------------------------------------------------------|
| `python-can` | Low-level CAN bus access (read/write frames)         |
| `cantools`   | Decode CAN frames using a DBC database file          |

### ARS 408-21 DBC file

Continental provides a DBC file with the ARS 408-21 delivery package
(`ARS408-21_CAN_Interface.dbc`).  Place it alongside your capture scripts.
The DBC file defines all message IDs and signal encodings required by
`cantools`.

---

## Hardware Setup

1. **Mount the sensor** on a rigid bracket facing the target scene.  The sensor
   connector is a 12-pin Deutsch DTM series plug; the delivery kit includes a
   mating harness.

2. **Wire power and CAN:**
   - Supply **+12 V DC** (typically 10–16 V) on the power lines.
   - Connect **CAN-H** and **CAN-L** from the sensor harness to a
     **USB-to-CAN adapter** (e.g. PEAK PCAN-USB, IXXAT USB-to-CAN, or a
     SocketCAN-compatible device such as the Kvaser Leaf Light).
   - Place a **120 Ω termination resistor** across CAN-H and CAN-L at each
     end of the bus.  Most USB-to-CAN adapters have a built-in switchable
     terminator; enable it if the adapter is at one end of the cable.

3. **Configure the CAN interface (Linux):**

   ```bash
   # Replace can0 with the interface name shown by `ip link`
   sudo ip link set can0 type can bitrate 500000
   sudo ip link set can0 up
   ```

   Verify the interface is up:

   ```bash
   ip -details link show can0
   ```

   You should see `bitrate 500000` in the output.

4. **Confirm frames are arriving:**

   ```bash
   candump can0
   ```

   Once the sensor has powered up (allow ~1 s), you should see a continuous
   stream of CAN frames.  The ARS 408-21 transmits object-list frames roughly
   every 60 ms (at the default 18.5 Hz update rate).  Press **Ctrl-C** to
   stop.

> **Windows / macOS note:** Use the vendor's driver and application
> (e.g. PEAK PCAN-View) to verify connectivity.  The Python capture scripts
> below work on all platforms via `python-can`'s `pcan`, `ixxat`, or
> `kvaser` backends—replace `"socketcan"` with the appropriate interface
> string for your adapter.

---

## Capturing CAN Data

The ARS 408-21 object list is transmitted as a burst of CAN frames for each
radar cycle.  A cycle begins with message **`Obj_0_Status`** (ID `0x60A`),
which carries the number of objects in the list, followed by one
**`Obj_1_General`** frame (ID `0x60B`) per object and one **`Obj_2_Quality`**
frame (ID `0x60C`) per object.

### Linux – using `candump`

`candump` (part of `can-utils`) records raw frames to an ASCII log file:

```bash
# Install can-utils if not already present
sudo apt install can-utils   # Debian / Ubuntu

# Capture for 10 seconds to a log file
timeout 10 candump -l can0
# Creates a file named candump-YYYY-MM-DD_HHMMSS.log
```

The log format is one frame per line:
```
(1700000000.123456) can0 60B#0100803F00000000
```

### Python – direct CAN capture

The script below collects complete radar cycles and writes each cycle's
decoded objects to a structured NumPy binary file (``.bin``), one file per
cycle:

```python
"""
capture_ars408.py
Capture one or more radar cycles from a Continental ARS 408-21 sensor
and save each cycle as a .bin file for sensor_transposition.

Requires:
    pip install python-can cantools numpy
"""

import math
import struct
from pathlib import Path

import cantools
import can
import numpy as np

# ---------------------------------------------------------------------------
# Configuration – adjust as needed
# ---------------------------------------------------------------------------

DBC_FILE     = "ARS408-21_CAN_Interface.dbc"  # path to the DBC file
INTERFACE    = "socketcan"                      # python-can interface string
CHANNEL      = "can0"                           # CAN channel / device
BITRATE      = 500_000                          # 500 kbps
NUM_CYCLES   = 50                               # number of cycles to capture
OUT_DIR      = Path("radar_frames")

# ARS 408-21 message IDs (decimal)
MSG_OBJ_STATUS  = 0x60A   # Obj_0_Status  – carries object count
MSG_OBJ_GENERAL = 0x60B   # Obj_1_General – one per object per cycle
MSG_OBJ_QUALITY = 0x60C   # Obj_2_Quality – one per object per cycle

RECORD_DTYPE = np.dtype([
    ("range",     np.float32),
    ("azimuth",   np.float32),
    ("elevation", np.float32),
    ("velocity",  np.float32),
    ("snr",       np.float32),
])

# ---------------------------------------------------------------------------

def spherical_from_cart(dist_long: float, dist_lat: float):
    """Convert longitudinal/lateral distances to (range_m, azimuth_deg)."""
    rng = math.sqrt(dist_long ** 2 + dist_lat ** 2)
    az  = math.degrees(math.atan2(dist_lat, dist_long))
    return rng, az


def main():
    OUT_DIR.mkdir(exist_ok=True)
    db = cantools.database.load_file(DBC_FILE)

    bus = can.interface.Bus(interface=INTERFACE, channel=CHANNEL,
                            bitrate=BITRATE)
    print(f"Connected to {CHANNEL}.  Capturing {NUM_CYCLES} cycles …")

    cycle_idx = 0
    objects   = {}  # obj_id -> {"dist_long", "dist_lat", "vrel_long", "rcs"}

    try:
        for msg in bus:
            arb_id = msg.arbitration_id

            if arb_id == MSG_OBJ_STATUS:
                # Start of a new cycle – flush previous cycle if non-empty
                if objects:
                    _save_cycle(objects, cycle_idx, OUT_DIR, RECORD_DTYPE)
                    cycle_idx += 1
                    print(f"  Cycle {cycle_idx:04d}: {len(objects)} objects")
                    if cycle_idx >= NUM_CYCLES:
                        break
                objects = {}

            elif arb_id == MSG_OBJ_GENERAL:
                decoded   = db.decode_message(arb_id, msg.data)
                obj_id    = int(decoded["Obj_ID"])
                objects.setdefault(obj_id, {})
                objects[obj_id]["dist_long"] = float(decoded["Obj_DistLong"])
                objects[obj_id]["dist_lat"]  = float(decoded["Obj_DistLat"])
                objects[obj_id]["vrel_long"] = float(decoded["Obj_VrelLong"])

            elif arb_id == MSG_OBJ_QUALITY:
                decoded = db.decode_message(arb_id, msg.data)
                obj_id  = int(decoded["Obj_ID"])
                objects.setdefault(obj_id, {})
                objects[obj_id]["rcs"] = float(decoded.get("Obj_RCS", 0.0))

    finally:
        bus.shutdown()

    print(f"Saved {cycle_idx} cycle(s) to {OUT_DIR}/")


def _save_cycle(objects: dict, idx: int, out_dir: Path, dtype) -> None:
    valid = [o for o in objects.values()
             if "dist_long" in o and "dist_lat" in o]
    if not valid:
        return

    records = np.empty(len(valid), dtype=dtype)
    for i, obj in enumerate(valid):
        rng, az = spherical_from_cart(obj["dist_long"], obj["dist_lat"])
        records[i]["range"]     = rng
        records[i]["azimuth"]   = az
        records[i]["elevation"] = 0.0
        records[i]["velocity"]  = obj.get("vrel_long", 0.0)
        records[i]["snr"]       = obj.get("rcs", 0.0)

    out_path = out_dir / f"frame_{idx:06d}.bin"
    records.tofile(out_path)


if __name__ == "__main__":
    main()
```

Run the capture script:

```bash
python capture_ars408.py
```

Each `.bin` file contains the detections for one radar update cycle.

---

## Converting CAN Frames to `.bin`

If you already have a `candump` log file and want to convert it offline, the
script below replays the log and produces the same `.bin` files as the live
capture script:

```python
"""
convert_ars408_log.py
Convert a candump ASCII log from the ARS 408-21 to per-cycle .bin files.

Requires:
    pip install python-can cantools numpy
"""

import math
from pathlib import Path

import cantools
import can
import numpy as np

DBC_FILE  = "ARS408-21_CAN_Interface.dbc"
LOG_FILE  = "candump-2024-01-01_120000.log"   # path to your candump log
OUT_DIR   = Path("radar_frames")

MSG_OBJ_STATUS  = 0x60A
MSG_OBJ_GENERAL = 0x60B
MSG_OBJ_QUALITY = 0x60C

RECORD_DTYPE = np.dtype([
    ("range", np.float32), ("azimuth", np.float32),
    ("elevation", np.float32), ("velocity", np.float32), ("snr", np.float32),
])


def spherical_from_cart(dist_long, dist_lat):
    rng = math.sqrt(dist_long ** 2 + dist_lat ** 2)
    az  = math.degrees(math.atan2(dist_lat, dist_long))
    return rng, az


OUT_DIR.mkdir(exist_ok=True)
db  = cantools.database.load_file(DBC_FILE)
log = can.LogReader(LOG_FILE)

cycle_idx = 0
objects   = {}

for msg in log:
    arb_id = msg.arbitration_id

    if arb_id == MSG_OBJ_STATUS:
        if objects:
            valid = [o for o in objects.values() if "dist_long" in o]
            if valid:
                records = np.empty(len(valid), dtype=RECORD_DTYPE)
                for i, obj in enumerate(valid):
                    rng, az = spherical_from_cart(obj["dist_long"], obj["dist_lat"])
                    records[i] = (rng, az, 0.0, obj.get("vrel_long", 0.0), obj.get("rcs", 0.0))
                records.tofile(OUT_DIR / f"frame_{cycle_idx:06d}.bin")
                print(f"Cycle {cycle_idx:04d}: {len(valid)} objects")
            cycle_idx += 1
        objects = {}

    elif arb_id == MSG_OBJ_GENERAL:
        decoded = db.decode_message(arb_id, msg.data)
        oid = int(decoded["Obj_ID"])
        objects.setdefault(oid, {})
        objects[oid]["dist_long"] = float(decoded["Obj_DistLong"])
        objects[oid]["dist_lat"]  = float(decoded["Obj_DistLat"])
        objects[oid]["vrel_long"] = float(decoded["Obj_VrelLong"])

    elif arb_id == MSG_OBJ_QUALITY:
        decoded = db.decode_message(arb_id, msg.data)
        oid = int(decoded["Obj_ID"])
        objects.setdefault(oid, {})
        objects[oid]["rcs"] = float(decoded.get("Obj_RCS", 0.0))

print(f"Converted {cycle_idx} cycle(s) → {OUT_DIR}/")
```

---

## Loading with sensor_transposition

```python
from sensor_transposition.radar.radar import RadarParser

parser     = RadarParser("radar_frames/frame_000000.bin")
detections = parser.read()

print(f"Detections : {len(detections)}")
print(f"Fields     : {detections.dtype.names}")

# Access individual fields
ranges     = detections["range"]      # slant range in metres
azimuths   = detections["azimuth"]    # azimuth angle in degrees
velocities = detections["velocity"]   # radial velocity in m/s (negative = closing)
rcs        = detections["snr"]        # RCS used as signal-strength proxy

# Cartesian coordinates (x = forward, y = left, z = up)
xyz = parser.xyz()   # (N, 3) float32 array
print(f"XYZ shape  : {xyz.shape}")
print(f"First detection: x={xyz[0, 0]:.2f} m, y={xyz[0, 1]:.2f} m")

# Module-level convenience function
from sensor_transposition.radar.radar import load_radar_bin

detections = load_radar_bin("radar_frames/frame_000000.bin")
```

---

## Verifying Your Data

After loading, a quick sanity check confirms the data looks reasonable for
outdoor driving conditions:

```python
import numpy as np
from sensor_transposition.radar.radar import RadarParser

parser     = RadarParser("radar_frames/frame_000000.bin")
detections = parser.read()
xyz        = parser.xyz()

print(f"Total detections : {len(detections)}")
print(f"Range   min/max  : {detections['range'].min():.1f} / {detections['range'].max():.1f} m")
print(f"Azimuth min/max  : {detections['azimuth'].min():.1f} / {detections['azimuth'].max():.1f} °")
print(f"Velocity min/max : {detections['velocity'].min():.1f} / {detections['velocity'].max():.1f} m/s")
print(f"RCS      min/max : {detections['snr'].min():.1f} / {detections['snr'].max():.1f} dBsm")

assert not np.any(np.isnan(xyz)), "NaN values present – check the conversion script"
assert detections["range"].min() >= 0,   "Negative range – conversion error"
assert detections["range"].max() <= 260, "Range exceeds sensor maximum (250 m)"
```

**Expected ranges for a typical outdoor scene:**

| Metric | Typical value |
|--------|---------------|
| Detections per cycle | 1 – 64 (up to 96 raw clusters) |
| Range | 0.2 m to 250 m |
| Azimuth | −60° to +60° (±9° in long-range mode) |
| Velocity (relative) | −70 m/s to +70 m/s |
| RCS | −50 dBsm to +50 dBsm |

---

## Troubleshooting

### No frames visible in `candump`

- Confirm the CAN interface is up: `ip -details link show can0`
- Verify the bit rate matches the sensor (500 kbps for ARS 408-21).
- Check that **termination resistors** are present at both ends of the bus.
  A missing terminator is the most common cause of a silent bus.
- Measure CAN-H and CAN-L with a multimeter: in the recessive (idle) state
  both lines should be at ≈ 2.5 V; CAN-H rises to ≈ 3.5 V and CAN-L drops to
  ≈ 1.5 V during a dominant bit.

### `can.CanError: Failed to transmit` or interface errors

The SocketCAN kernel module may not be loaded.  Run:

```bash
sudo modprobe can
sudo modprobe can_raw
sudo modprobe can_dev
```

Then re-run the `ip link set` commands from the [Hardware Setup](#hardware-setup)
section.

### `FileNotFoundError` when loading

Verify the path passed to `RadarParser`:

```python
from pathlib import Path
print(Path("radar_frames/frame_000000.bin").exists())   # must print True
```

### `ValueError: Radar binary file size … is not divisible by the record size`

The `.bin` file is truncated or corrupted.  Re-run the capture script and
check for disk-space issues.

### Empty detections list (zero objects)

- The ARS 408-21 may still be in its start-up phase (allow ~2 s after power-on).
- Confirm the sensor has a clear field of view; a lens cover or pointed away
  from objects will yield zero detections.
- Check that the `Obj_0_Status` frame (ID `0x60A`) is actually arriving by
  filtering: `candump can0 60A#`.

### Coordinate system

The ARS 408-21 uses a **sensor-centric Cartesian frame**:

- **X (forward):** positive in the direction the sensor faces.
- **Y (left):** positive to the left of the sensor boresight.

`RadarParser.xyz()` applies the same convention: `x = range × cos(az)`,
`y = range × sin(az)`, `z = 0`.  When fusing with other sensors, use the
extrinsic calibration support in `sensor_transposition.sensor_collection`
(see `examples/sensor_collection.yaml`) to transform detections into a common
ego frame.
