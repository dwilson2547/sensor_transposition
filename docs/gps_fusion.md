# GPS Absolute-Position Fusion

This guide explains how to use `sensor_transposition.gps.fusion` to anchor the
local SLAM map to a global geodetic frame and fuse GPS fixes into either the
Error-State EKF or a `FramePoseSequence` trajectory.

---

## Overview

A typical SLAM pipeline builds a **local** Cartesian map in the East-North-Up
(ENU) frame centred on an arbitrary origin.  GPS receivers, by contrast,
report positions in **geodetic** coordinates (latitude / longitude / altitude
above the WGS-84 ellipsoid).  Fusing these two representations requires:

1. **Choosing a reference origin** – a fixed geodetic point that defines where
   `(east=0, north=0, up=0)` of the local map is located on Earth.
2. **Converting each fix** from geodetic to the local ENU frame using the
   `GpsFuser.fix_to_enu` method.
3. **Fusing the ENU position** into the state estimator
   (`ImuEkf.position_update`) or directly into the trajectory
   (`FramePoseSequence`).

This workflow *ties* the local map to a global coordinate frame and enables
multi-session SLAM, absolute pose retrieval, and comparison of trajectories
against ground-truth GPS tracks.

---

## Key Components

| Symbol | Description |
|--------|-------------|
| `GpsFuser` | Main class; stores the ENU origin and converts / fuses fixes. |
| `hdop_to_noise` | Converts HDOP from a GGA sentence to a 3×3 position noise covariance. |
| `GgaFix` | GGA sentence dataclass from `sensor_transposition.gps.nmea`. |
| `RmcFix` | RMC sentence dataclass from `sensor_transposition.gps.nmea`. |

---

## Quick-Start Example

```python
from sensor_transposition.gps.fusion import GpsFuser, hdop_to_noise
from sensor_transposition.gps.nmea import NmeaParser
from sensor_transposition.frame_pose import FramePoseSequence
from sensor_transposition.imu.ekf import ImuEkf, EkfState

# 1. Load GPS fixes from an NMEA log file.
fixes = NmeaParser("gps_log.nmea").gga_fixes()

# 2. Use the first fix as the map origin.
origin = fixes[0]
fuser = GpsFuser(
    ref_lat=origin.latitude,
    ref_lon=origin.longitude,
    ref_alt=origin.altitude,
)

# 3a. Build a trajectory from GPS alone.
seq = FramePoseSequence()
for i, fix in enumerate(fixes):
    fuser.fuse_into_sequence(seq, timestamp=float(i) * 0.1, fix=fix)

# 3b. Or fuse GPS into an EKF that has already been propagated by IMU.
ekf   = ImuEkf()
state = EkfState()
for fix in fixes:
    state = fuser.fuse_into_ekf(ekf, state, fix, noise=hdop_to_noise(fix.hdop))
```

---

## Choosing the Reference Origin

The ENU origin should be fixed for the entire session (or across sessions if
you want multi-session consistency).  Good choices include:

* **The first valid GPS fix** – simple and automatic; useful for single-session
  mapping.
* **A known survey point** – set `ref_lat/ref_lon/ref_alt` from a benchmark
  coordinate; enables absolute positioning.
* **The start of the dataset** – pick any convenient fix near the centre of the
  expected trajectory to keep ENU coordinates small.

```python
fuser = GpsFuser(
    ref_lat=51.5080,   # Trafalgar Square, London
    ref_lon=-0.1281,
    ref_alt=10.0,      # approximate ellipsoidal height
)
```

---

## Converting a Fix to Local ENU

```python
east, north, up = fuser.fix_to_enu(fix)
# or, as a NumPy array:
pos_enu = fuser.fix_to_enu_array(fix)   # shape (3,), dtype float64
```

`GgaFix` records include an altitude field that is used for the Up component.
`RmcFix` records carry no altitude; the reference altitude is substituted so
that the Up component is approximately zero at the map origin.

---

## Measurement Noise from HDOP

GPS accuracy is reported through the **Horizontal Dilution of Precision**
(HDOP) field in GGA sentences.  `hdop_to_noise` converts this to a 3×3
diagonal covariance matrix:

```python
import numpy as np
from sensor_transposition.gps.fusion import hdop_to_noise

# Standard GPS (≈3 m horizontal, ≈5 m vertical at HDOP=1)
noise = hdop_to_noise(fix.hdop)

# High-precision RTK receiver (≈2 cm horizontal at HDOP=1)
rtk_noise = hdop_to_noise(fix.hdop, base_sigma_m=0.02, vertical_sigma_m=0.05)
```

The matrix is `diag(σ_E², σ_N², σ_U²)` where `σ_E = σ_N = HDOP × base_sigma_m`
and `σ_U = vertical_sigma_m`.

---

## Fusing into the EKF

`GpsFuser.fuse_into_ekf` is a thin wrapper around
`ImuEkf.position_update` that handles the geodetic-to-ENU conversion
automatically:

```python
from sensor_transposition.imu.ekf import ImuEkf, EkfState

ekf   = ImuEkf()
state = EkfState(timestamp=0.0)

# --- IMU prediction loop (call at ~100–1000 Hz) ---
for t, accel, gyro in imu_stream:
    dt = t - state.timestamp
    state = ekf.predict(state, accel, gyro, dt)

# --- GPS update (call at ~1–10 Hz) ---
if new_gps_fix_available:
    state = fuser.fuse_into_ekf(
        ekf, state, gga_fix,
        noise=hdop_to_noise(gga_fix.hdop),
    )
```

The EKF corrects the position, velocity, orientation, and IMU biases using
the GPS measurement.  See `docs/state_estimation.md` for a detailed
description of the filter.

---

## Fusing into a FramePoseSequence

`GpsFuser.fuse_into_sequence` adds GPS-derived positions directly to a
`FramePoseSequence` trajectory:

```python
from sensor_transposition.frame_pose import FramePoseSequence

seq = FramePoseSequence(frame_duration=0.1)

for i, fix in enumerate(fixes):
    pose = fuser.fuse_into_sequence(seq, timestamp=float(i) * 0.1, fix=fix)
    print(f"Frame {i}: east={pose.translation[0]:.2f} m, "
          f"north={pose.translation[1]:.2f} m, "
          f"up={pose.translation[2]:.2f} m")
```

If a frame already covers the given timestamp (within its `frame_duration`
window), its translation is updated in place rather than a new frame being
created.  The orientation quaternion is left unchanged (identity for new
frames), so GPS fusion can be combined with orientation estimates from IMU
or LiDAR scan-matching.

---

## Multi-Session SLAM

To align two sessions recorded at different times, both sessions must use the
**same reference origin**:

```python
# Session A – drive around the block.
fuser = GpsFuser(ref_lat=51.5080, ref_lon=-0.1281, ref_alt=10.0)
seq_a = FramePoseSequence()
for i, fix in enumerate(fixes_a):
    fuser.fuse_into_sequence(seq_a, timestamp=float(i) * 0.1, fix=fix)

# Session B – same origin.
seq_b = FramePoseSequence()
for i, fix in enumerate(fixes_b):
    fuser.fuse_into_sequence(seq_b, timestamp=float(i) * 0.1, fix=fix)

# Both trajectories are now in the same ENU coordinate system.
```

---

## GNSS Outage Handling

During GNSS outages (tunnels, urban canyons, multi-storey car parks) the GPS
receiver stops producing valid fixes.  If `fuse_into_ekf` is called with a
stale fix, the EKF will incorporate outdated position information, degrading
the state estimate.

### Enabling outage detection

Pass `max_fix_age_sec` and an optional `on_outage` callback when constructing
`GpsFuser`:

```python
from sensor_transposition.gps.fusion import GpsFuser, hdop_to_noise
from sensor_transposition.imu.ekf import ImuEkf, EkfState

def handle_outage(age_sec: float) -> None:
    """Called when a GPS update is skipped because the fix is stale."""
    print(f"GNSS outage detected — last fix was {age_sec:.1f} s ago. "
          f"Relying on IMU/wheel-odometry dead-reckoning.")

fuser = GpsFuser(
    ref_lat=51.5080,
    ref_lon=-0.1281,
    max_fix_age_sec=2.0,       # skip GPS updates older than 2 s
    on_outage=handle_outage,   # called each time an update is skipped
)
```

### Fusing with a timestamp

Pass `current_timestamp` to `fuse_into_ekf` so the fuser can track fix age:

```python
ekf   = ImuEkf()
state = EkfState()
current_time = 0.0

for t, fix, accel, gyro, dt in sensor_stream:
    # IMU prediction (always runs)
    state = ekf.predict(state, accel, gyro, dt)
    current_time += dt

    # GPS update — skipped automatically during outages
    noise = hdop_to_noise(fix.hdop)
    state = fuser.fuse_into_ekf(
        ekf, state, fix, noise,
        current_timestamp=current_time,
    )
```

When `age > max_fix_age_sec`:

1. `fuse_into_ekf` returns the **unchanged** state (EKF is not updated).
2. `on_outage(age)` is called with the age in seconds (if provided).
3. The EKF continues to propagate using IMU measurements alone.

### Querying fix age manually

Use `fix_age(current_timestamp)` to poll the staleness of the last fix:

```python
age = fuser.fix_age(current_time)   # None if no fix has been fused yet

if age is not None and age > 5.0:
    print("GPS fix is very stale — increase dead-reckoning covariance")
```

### Recommended pattern: fall back to LiDAR/wheel odometry

```python
from sensor_transposition.gps.fusion import GpsFuser, hdop_to_noise
from sensor_transposition.imu.ekf import ImuEkf, EkfState

_use_gps = True

def on_outage(age_sec: float) -> None:
    global _use_gps
    _use_gps = False
    print(f"GNSS outage ({age_sec:.1f} s) — switching to LiDAR odometry")

fuser = GpsFuser(
    ref_lat=51.5080, ref_lon=-0.1281,
    max_fix_age_sec=1.0,
    on_outage=on_outage,
)

# When a fresh fix arrives again, the fuser will resume updating the EKF
# automatically — no manual re-enable needed.
```

---

## API Reference

### `hdop_to_noise(hdop, base_sigma_m=3.0, vertical_sigma_m=5.0)`

Returns a `(3, 3)` diagonal numpy array representing the ENU position
measurement noise covariance.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hdop` | — | Horizontal dilution of precision (from GGA sentence). |
| `base_sigma_m` | `3.0` | Horizontal σ in metres when HDOP = 1. |
| `vertical_sigma_m` | `5.0` | Vertical (Up) σ in metres, constant. |

### `GpsFuser(ref_lat, ref_lon, ref_alt=0.0, max_fix_age_sec=None, on_outage=None)`

| Method | Returns | Description |
|--------|---------|-------------|
| `fix_to_enu(fix)` | `(east, north, up)` | Convert a fix to ENU tuple (metres). |
| `fix_to_enu_array(fix)` | `np.ndarray (3,)` | Same as above, as a NumPy array. |
| `fuse_into_ekf(ekf, state, fix, noise, current_timestamp=None)` | `EkfState` | Fuse via EKF position update; skips update if fix is stale. |
| `fuse_into_sequence(seq, timestamp, fix)` | `FramePose` | Add/update frame in trajectory. |
| `fix_age(current_timestamp)` | `float \| None` | Age of the most recent fused fix (s), or `None` if no fix yet. |

| Property | Type | Description |
|----------|------|-------------|
| `ref_lat` | `float` | Latitude of ENU origin (degrees). |
| `ref_lon` | `float` | Longitude of ENU origin (degrees). |
| `ref_alt` | `float` | Altitude of ENU origin (metres). |
| `max_fix_age_sec` | `float \| None` | Outage threshold in seconds (`None` = disabled). |
| `last_fix_timestamp` | `float \| None` | Timestamp of the most recently fused fix. |
