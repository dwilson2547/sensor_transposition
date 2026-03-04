# sensor_transposition

A Python toolkit for multi-sensor calibration, data parsing, and coordinate-frame management in autonomous-vehicle and robotics pipelines.

---

## Contents

- [Features](#features)
- [Installation](#installation)
- [Coordinate Systems](#coordinate-systems)
- [Modules](#modules)
  - [SensorCollection](#sensorcollection)
  - [Sensor Synchroniser](#sensor-synchroniser)
  - [Camera Intrinsics](#camera-intrinsics)
  - [Fisheye / Omnidirectional Camera](#fisheye--omnidirectional-camera)
  - [Transform](#transform)
  - [LiDAR–Camera Fusion](#lidar-camera-fusion)
  - [LiDAR Parsers](#lidar-parsers)
  - [LiDAR Scan Matching](#lidar-scan-matching)
  - [LiDAR Motion Distortion Correction](#lidar-motion-distortion-correction)
  - [GPS / GNSS](#gps--gnss)
  - [GPS Coordinate-Frame Converter](#gps-coordinate-frame-converter)
  - [GPS Fusion](#gps-fusion)
  - [IMU](#imu)
  - [IMU Error-State EKF](#imu-error-state-ekf)
  - [IMU Pre-integration](#imu-pre-integration)
  - [Radar](#radar)
  - [Radar Odometry](#radar-odometry)
  - [Visual Odometry](#visual-odometry)
  - [Wheel Odometry](#wheel-odometry)
  - [Loop Closure](#loop-closure)
  - [Pose Graph](#pose-graph)
  - [Sliding-Window Smoother](#sliding-window-smoother)
  - [Submap Manager](#submap-manager)
  - [Occupancy Grid](#occupancy-grid)
  - [Voxel Map (TSDF)](#voxel-map-tsdf)
  - [Point-Cloud Map](#point-cloud-map)
  - [Visualisation](#visualisation)
  - [Bag Recorder / Player](#bag-recorder--player)
  - [Camera–LiDAR Extrinsic Calibration](#cameralidar-extrinsic-calibration)
  - [SLAM Session (Pipeline Orchestration)](#slam-session-pipeline-orchestration)
- [Error Handling](#error-handling)
- [Configuration Example](#configuration-example)
- [Quick-Start: End-to-End SLAM Pipeline](#quick-start-end-to-end-slam-pipeline)
- [ROS Examples](#ros-examples)

---

## Features

- **Sensor collection** – YAML-driven multi-sensor configuration storing extrinsics (translation + quaternion), **temporal extrinsic calibration** (time offset per sensor), and per-sensor intrinsics/parameters for cameras, LiDARs, radars, GPS, and IMUs.
- **Pinhole camera model** – focal-length derivation from FOV or physical sensor geometry, camera matrix construction, point projection / unprojection, and Brown–Conrady lens distortion/undistortion.
- **Fisheye / omnidirectional camera model** – Kannala-Brandt equidistant projection supporting fields of view up to 360°, with focal-length derivation, point projection / unprojection, and Kannala-Brandt distortion/undistortion (compatible with OpenCV's `cv2.fisheye` module).
- **Homogeneous transforms** – composable 4×4 `Transform` objects with helpers for building from quaternions or rotation matrices, inverting, and applying to point clouds.
- **LiDAR–camera fusion** – project 3-D LiDAR point clouds onto a camera image plane and colour the cloud by sampling pixel values.
- **LiDAR parsers** – binary readers for Velodyne (KITTI `.bin`), Ouster (4-column and 8-column `.bin`), and Livox (LVX / LVX2) file formats.
- **LiDAR scan matching** – point-to-point ICP (Iterative Closest Point) with the Kabsch SVD algorithm and a KD-tree nearest-neighbour search; supports a maximum correspondence-distance filter, an optional initial transform, and a configurable convergence tolerance.
- **LiDAR motion distortion correction** – IMU-based scan deskewing that corrects per-point timestamps across a spinning LiDAR sweep.
- **GPS / GNSS** – NMEA 0183 parser supporting GGA and RMC sentence types, plus a coordinate-frame converter for ECEF ↔ ENU and geodetic ↔ UTM conversions.
- **GPS fusion** – `GpsFuser` converts GPS fixes to local ENU and integrates them into an `ImuEkf` state or a `FramePoseSequence`; `hdop_to_noise` converts HDOP to a 3×3 position noise covariance.
- **IMU** – binary parser for 32-byte (accel + gyro) and 48-byte (accel + gyro + quaternion) records.
- **IMU Error-State EKF** – 15-state error-state extended Kalman filter fusing IMU, GPS, and pose observations.
- **IMU pre-integration** – accumulates raw IMU measurements between keyframes into compact (ΔR, Δv, Δp) increments for tight IMU–LiDAR coupling.
- **Radar** – binary parser for 5-field (range, azimuth, elevation, velocity, SNR) detection records with spherical → Cartesian conversion.
- **Radar odometry** – Doppler-based ego-velocity estimation and scan-to-scan ICP radar odometry.
- **Visual odometry** – essential-matrix estimation (normalised 8-point + RANSAC), pose recovery, and Perspective-n-Point (PnP) solver.
- **Wheel odometry** – differential-drive and Ackermann (bicycle-model) dead-reckoning with midpoint integration.
- **Loop closure** – Scan Context and M2DP place-recognition descriptors with `ScanContextDatabase` for efficient loop-closure candidate retrieval.
- **Pose graph** – pose graph data structure and Gauss-Newton optimisation back-end for graph-SLAM.
- **Sliding-window smoother** – fixed-lag online SLAM smoother that bounds per-step cost to O(window_size³).
- **Submap manager** – keyframe selection and submap division for large-scale long-duration SLAM.
- **Occupancy grid** – 2-D probabilistic occupancy grid with log-odds ray-casting; exports to ROS `nav_msgs/OccupancyGrid` int8 format.
- **Voxel map (TSDF)** – Truncated Signed-Distance Function volumetric map for dense 3-D reconstruction.
- **Point-cloud map** – accumulated coloured point-cloud map with voxel-grid downsampling and PCD / PLY I/O.
- **Visualisation** – BEV rendering, trajectory overlay, LiDAR-on-image overlay, Open3D and RViz export helpers.
- **Bag recorder / player** – lightweight multi-topic binary bag format (`.sbag`) with streaming write and indexed playback; no external dependencies.  `sbag_to_rosbag()` converts to MCAP for use with `ros2 bag` (requires `pip install ".[mcap]"`).
- **SLAM session** – `SLAMSession` orchestration class that wires ICP odometry, Scan Context loop closure, pose-graph optimisation, and point-cloud map accumulation into a single object with sensible defaults and per-topic callbacks.
- **Camera–LiDAR extrinsic calibration** – target-based extrinsic calibration using plane correspondences (`fit_plane`, `ransac_plane`, `calibrate_lidar_camera`).
- **ROS examples** – ready-to-use launch / parameter files for Velodyne and Ouster LiDARs in both ROS 1 and ROS 2.

---

## Installation

```bash
pip install .
```

Development dependencies (pytest):

```bash
pip install ".[dev]"
```

Requires Python ≥ 3.9 and `numpy`, `pyyaml`, and `scipy`.

### Optional extras

Install optional extras to unlock additional integrations:

| Extra | Package installed | Enables |
|-------|------------------|---------|
| `open3d` | `open3d>=0.17` | `export_point_cloud_open3d()` and Open3D visualisation |
| `rerun` | `rerun-sdk>=0.14` | rerun.io visualisation logging |
| `mcap` | `mcap>=1.0` | `sbag_to_rosbag()` MCAP conversion |

```bash
pip install ".[open3d]"   # Open3D support
pip install ".[rerun]"    # rerun.io support
pip install ".[mcap]"     # MCAP / ROS 2 bag conversion
```

---

## Coordinate Systems

All sensors declare their native coordinate frame convention.  The ego (vehicle) frame is **FLU** – x forward, y left, z up.

| Convention | Axes               | Typical use                        |
|------------|--------------------|------------------------------------|
| **FLU**    | Forward, Left, Up  | Ego frame; ROS; Velodyne / Ouster  |
| **RDF**    | Right, Down, Fwd   | Camera optical frame               |
| **FRD**    | Forward, Right, Dwn| NED-like robotics frames           |
| **ENU**    | East, North, Up    | GPS / local tangent plane          |
| **NED**    | North, East, Down  | Aviation / navigation              |

---

## Modules

### SensorCollection

Loads a YAML file describing a rig of sensors and provides transform queries between any two sensors.

```python
from sensor_transposition import SensorCollection

col = SensorCollection.from_yaml("examples/sensor_collection.yaml")

# 4×4 transform: front_lidar → front_camera
T = col.transform_between("front_lidar", "front_camera")

# Modify and save
col.to_yaml("my_rig.yaml")
```

**YAML sensor types:** `camera`, `lidar`, `radar`, `gps`, `imu`.  
Each entry specifies `coordinate_system`, `extrinsics` (translation + quaternion + optional `time_offset_sec`), and optional type-specific parameter blocks.

**`from_yaml()` validation** – when loading from YAML, `SensorCollection.validate()` is automatically called to catch common mistakes (e.g. a `camera` sensor missing its `intrinsics` block, or non-positive `fx`/`fy`/`width`/`height` values) before they cause cryptic errors downstream:

```python
# Raises ValueError immediately with a clear message:
# "Camera sensor 'front_camera' is missing a required 'intrinsics' block."
col = SensorCollection.from_yaml("bad_config.yaml")

# You can also call validate() explicitly on a programmatically-built collection:
col.validate()
```

**Temporal extrinsic calibration** – each sensor can declare a `time_offset_sec` in its `extrinsics` block that captures the delay between the sensor's hardware clock and the reference/ego clock.  Use `SensorCollection.time_offset_between(source, target)` to compute the time difference between any two sensors:

```python
# How many seconds ahead is the camera clock relative to the LiDAR clock?
dt = col.time_offset_between("front_lidar", "front_camera")   # e.g. +0.033

# Synchronise a LiDAR timestamp to the camera's timebase
t_camera_equiv = t_lidar + dt
```

---

### Sensor Synchroniser

Resample multiple sensor streams onto a common reference timeline.

```python
from sensor_transposition.sync import (
    SensorSynchroniser,
    SensorSynchronizer,   # American-spelling alias
)
import numpy as np

sync = SensorSynchroniser()
sync.add_stream("lidar", lidar_times, lidar_data,
                time_offset_sec=col.get_sensor("front_lidar").time_offset_sec)
sync.add_stream("imu",   imu_times,   imu_accel,
                time_offset_sec=col.get_sensor("imu").time_offset_sec)

# Check that the streams actually overlap before resampling:
overlap = sync.temporal_overlap()
if overlap is None:
    raise RuntimeError("Streams do not overlap — check time offsets.")
t_start, t_end = overlap

# Resample all streams at LiDAR reference times (synchronise / synchronize both work):
aligned = sync.synchronise(lidar_times)
imu_at_lidar_times = aligned["imu"]
```

| Method | Description |
|--------|-------------|
| `add_stream(name, times, data)` | Register a named sensor stream |
| `temporal_overlap()` | Return `(start, end)` of the common overlap, or `None` |
| `stream_start_time(name)` / `stream_end_time(name)` | Per-stream time bounds |
| `synchronise(ref_times)` / `synchronize(ref_times)` | Resample all streams |
| `interpolate(name, query_times)` | Resample a single stream |

---

### Camera Intrinsics

Pinhole model utilities.

```python
from sensor_transposition.camera_intrinsics import (
    focal_length_from_fov,
    focal_length_from_sensor,
    fov_from_focal_length,
    camera_matrix,
    project_point,
    unproject_pixel,
    distort_point,
    undistort_point,
)

# Focal length from horizontal FOV
fx = focal_length_from_fov(image_size=1920, fov_deg=90.0)

# Focal length from physical sensor geometry
fx = focal_length_from_sensor(image_size_px=1920, sensor_size_mm=6.4, focal_length_mm=4.0)

# Build 3×3 K matrix
K = camera_matrix(fx=1266.4, fy=1266.4, cx=816.0, cy=612.0)

# Project a 3-D point to pixel coordinates
u, v = project_point(K, [1.0, 0.5, 5.0])

# Unproject a pixel at a known depth back to 3-D
xyz = unproject_pixel(K, (u, v), depth=5.0)

# Apply / remove Brown–Conrady distortion
coeffs = (-0.05, 0.08, 0.0, 0.0, -0.03)  # (k1, k2, p1, p2, k3)
distorted = distort_point([x_n, y_n], coeffs)
undistorted = undistort_point(distorted, coeffs)
```

---

### Fisheye / Omnidirectional Camera

Kannala-Brandt equidistant projection for wide-FOV and omnidirectional
cameras.  Supports fields of view up to 360° and is compatible with
coefficients from OpenCV's `cv2.fisheye.calibrate`.

```python
from sensor_transposition.camera_intrinsics import (
    fisheye_focal_length_from_fov,
    fisheye_project_point,
    fisheye_unproject_pixel,
    fisheye_distort_point,
    fisheye_undistort_point,
    camera_matrix,
)
import numpy as np

# Focal length for a 190° horizontal FOV on a 1920-pixel-wide sensor
fx = fisheye_focal_length_from_fov(image_size=1920, fov_deg=190.0)
fy = fisheye_focal_length_from_fov(image_size=1080, fov_deg=130.0)

K = camera_matrix(fx=fx, fy=fy, cx=960.0, cy=540.0)

# Distortion coefficients (k1, k2, k3, k4) from cv2.fisheye.calibrate
dist_coeffs = (0.05, -0.02, 0.003, -0.001)

# Project a 3-D point (including points with large off-axis angles)
point_3d = np.array([0.8, 0.4, 3.0])
u, v = fisheye_project_point(K, point_3d, dist_coeffs=dist_coeffs)

# Unproject: depth is the Euclidean distance ‖[X, Y, Z]‖, not Z
depth = float(np.linalg.norm(point_3d))
recovered = fisheye_unproject_pixel(K, (u, v), depth=depth,
                                    dist_coeffs=dist_coeffs)

# Apply / remove Kannala-Brandt distortion on normalised coordinates
x_n, y_n = point_3d[0] / point_3d[2], point_3d[1] / point_3d[2]
distorted   = fisheye_distort_point([x_n, y_n], dist_coeffs)
undistorted = fisheye_undistort_point(distorted, dist_coeffs)
```

See [`docs/fisheye_camera_intrinsics_guide.md`](docs/fisheye_camera_intrinsics_guide.md)
for a full guide including how to calibrate with OpenCV and how to handle
omnidirectional cameras with FOV > 180°.

---

### Transform

Composable 4×4 homogeneous transformation matrices.

```python
from sensor_transposition.transform import Transform, sensor_to_sensor
import numpy as np

# Build from a quaternion [w, x, y, z] and translation [x, y, z]
T = Transform.from_quaternion([1.0, 0.0, 0.0, 0.0], translation=[1.84, 0.0, 1.91])

# Compose transforms
T_combined = T1 @ T2

# Apply to a single point or an (N, 3) cloud
pt_ego = T.apply_to_point([0.0, 0.0, 0.0])
cloud_ego = T.apply_to_points(lidar_cloud_xyz)

# Invert
T_ego_to_sensor = T.inverse()

# Compute sensor-to-sensor transform from two sensor-to-ego matrices
T_src_to_tgt = sensor_to_sensor(T_src_to_ego, T_tgt_to_ego)
```

---

### LiDAR–Camera Fusion

Project a 3-D LiDAR point cloud onto a camera image plane, or colour the cloud from an image.

```python
from sensor_transposition.lidar_camera import project_lidar_to_image, color_lidar_from_image

# pixel_coords: (N, 2) [u, v]; valid_mask: (N,) bool
pixel_coords, valid_mask = project_lidar_to_image(
    points=lidar_xyz,          # (N, 3) float array
    lidar_to_camera=T,         # 4×4 extrinsic matrix
    camera_matrix=K,           # 3×3 intrinsic matrix
    image_width=1632,
    image_height=1224,
)

# colours: (N, C); valid_mask: (N,) bool
colours, valid_mask = color_lidar_from_image(lidar_xyz, T, K, image)
```

Both methods are also exposed through `SensorCollection`:

```python
pixel_coords, valid = col.project_lidar_to_image("front_lidar", "front_camera", xyz)
colours, valid     = col.color_lidar_from_image("front_lidar", "front_camera", xyz, image)
```

---

### LiDAR Parsers

#### Velodyne (KITTI `.bin`)

```python
from sensor_transposition.lidar.velodyne import VelodyneParser, load_velodyne_bin

parser = VelodyneParser("frame0000.bin")
cloud  = parser.read()          # structured array: x, y, z, intensity
xyz    = parser.xyz()           # (N, 3) float32
xyzI   = parser.xyz_intensity() # (N, 4) float32

# Or as a one-liner
cloud = load_velodyne_bin("frame0000.bin")
```

Supported models: HDL-32E, HDL-64E, VLP-16, VLP-32C (all share the same KITTI binary format).

#### Ouster (`.bin`)

```python
from sensor_transposition.lidar.ouster import OusterParser, load_ouster_bin

parser = OusterParser("ouster_frame.bin")
cloud  = parser.read()   # 4-column (x,y,z,intensity) or 8-column variant auto-detected
xyz    = parser.xyz()
```

The 8-column extended format additionally provides `t`, `reflectivity`, `ring`, and `ambient` fields.

#### Livox (LVX / LVX2)

```python
from sensor_transposition.lidar.livox import LivoxParser, load_livox_lvx

parser = LivoxParser("recording.lvx2")
cloud  = parser.read()   # structured array: x, y, z, intensity (coordinates in metres)
xyz    = parser.xyz()
```

Supported data types: Cartesian float32 (type 0), Spherical float32 (type 1), Cartesian int32 mm-precision (type 2, LVX2).

---

### LiDAR Scan Matching

Point-to-point ICP (Iterative Closest Point) scan matching for LiDAR frame-to-frame odometry and map-to-scan localisation.

```python
from sensor_transposition.lidar.scan_matching import icp_align
import numpy as np

# Align source cloud to target cloud
result = icp_align(source_xyz, target_xyz, max_iterations=50)

if result.converged:
    print("Transform:\n", result.transform)   # 4×4 homogeneous matrix
    print("MSE:", result.mean_squared_error)  # mean squared point distance

# Apply the recovered transform to the source points
R = result.transform[:3, :3]
t = result.transform[:3, 3]
aligned = (R @ source_xyz.T).T + t
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | `50` | Maximum ICP iterations |
| `tolerance` | `1e-6` | Convergence threshold on MSE change per iteration |
| `max_correspondence_dist` | `inf` | Reject source–target pairs farther apart than this (metres) |
| `initial_transform` | `None` | Optional 4×4 initial guess applied before the first iteration |
| `callback` | `None` | Optional callable `(iteration, mse)` for progress monitoring |

```python
# Monitor ICP progress with a callback:
result = icp_align(
    source_xyz, target_xyz,
    callback=lambda i, mse: print(f"iter {i}: mse={mse:.6f}"),
)
```

The returned `IcpResult` contains:
- `transform` – 4×4 homogeneous matrix mapping source → target frame
- `converged` – ``True`` if the tolerance condition was met
- `num_iterations` – iterations actually performed
- `mean_squared_error` – final mean squared point-to-point distance (inliers only)

---

### LiDAR Motion Distortion Correction

Correct per-point motion distortion in a spinning LiDAR scan using IMU data.

```python
from sensor_transposition.lidar.motion_distortion import deskew_scan
from sensor_transposition.imu.imu import ImuParser

imu_data  = ImuParser("imu.bin").read()
imu_times = imu_data["timestamp"].astype(float)
accel     = imu_data[["ax", "ay", "az"]].view(float).reshape(-1, 3).astype(float)
gyro      = imu_data[["wx", "wy", "wz"]].view(float).reshape(-1, 3).astype(float)

# per_point_timestamps: (N,) float array in the same clock as the IMU
# ref_timestamp: the scan's reference time (start or end of sweep)
deskewed = deskew_scan(
    points=lidar_scan,
    per_point_timestamps=per_point_timestamps,
    imu_timestamps=imu_times,
    imu_accel=accel,
    imu_gyro=gyro,
    ref_timestamp=sweep_end_time,
)
```

Both `per_point_timestamps` and `ref_timestamp` must be in the same reference
clock as the IMU (apply `apply_time_offset` from `sync.py` first if needed).
`ref_timestamp` is typically the **end** of the sweep so that the corrected
cloud represents the sensor pose at the time the last point was captured.

---

### GPS / GNSS

Parse NMEA 0183 log files (GGA and RMC sentences).

```python
from sensor_transposition.gps.nmea import NmeaParser, GgaFix, RmcFix, load_nmea

parser = NmeaParser("gps_log.nmea")

for record in parser.records():
    if isinstance(record, GgaFix):
        print(record.latitude, record.longitude, record.altitude)
    elif isinstance(record, RmcFix) and record.is_valid:
        print(record.speed_knots, record.course)

# Filtered helpers
gga_fixes = parser.gga_fixes()
rmc_fixes = parser.rmc_fixes()

# One-liner
records = load_nmea("gps_log.nmea")
```

---

### GPS Coordinate-Frame Converter

Convert raw GPS latitude / longitude / altitude into local Cartesian frames
needed for SLAM initialisation, global constraints, or sensor-fusion.

#### ECEF ↔ Geodetic

```python
from sensor_transposition.gps.converter import geodetic_to_ecef, ecef_to_geodetic

# Geodetic (lat/lon/alt) → ECEF Cartesian
X, Y, Z = geodetic_to_ecef(lat_deg=51.5074, lon_deg=-0.1278, alt_m=11.0)

# ECEF → Geodetic
lat, lon, alt = ecef_to_geodetic(X, Y, Z)
```

#### ECEF / Geodetic ↔ ENU (local tangent plane)

```python
from sensor_transposition.gps.converter import (
    ecef_to_enu,
    enu_to_ecef,
    geodetic_to_enu,
)

# Define a local ENU origin (e.g. the first GPS fix)
lat0, lon0, alt0 = 51.5074, -0.1278, 11.0

# Geodetic → ENU (convenience wrapper)
east, north, up = geodetic_to_enu(
    lat_deg=51.5075, lon_deg=-0.1279, alt_m=11.0,
    lat0_deg=lat0, lon0_deg=lon0, alt0_m=alt0,
)

# ECEF → ENU
east, north, up = ecef_to_enu(X, Y, Z, lat0, lon0, alt0)

# ENU → ECEF (inversion)
X2, Y2, Z2 = enu_to_ecef(east, north, up, lat0, lon0, alt0)
```

#### Geodetic ↔ UTM

```python
from sensor_transposition.gps.converter import geodetic_to_utm, utm_to_geodetic

# Geodetic → UTM
easting, northing, zone_number, zone_letter = geodetic_to_utm(51.5074, -0.1278)
# → (699_330.6, 5_710_155.4, 30, 'U')

# UTM → Geodetic
lat, lon = utm_to_geodetic(easting, northing, zone_number, zone_letter)
```

---

### GPS Fusion

Integrate GPS fixes into the local ENU map frame used by the SLAM pipeline.

```python
from sensor_transposition.gps.fusion import GpsFuser, hdop_to_noise
from sensor_transposition.gps.nmea import NmeaParser
from sensor_transposition.imu.ekf import ImuEkf, EkfState
from sensor_transposition.frame_pose import FramePoseSequence

# Load GPS fixes.
fixes = NmeaParser("gps_log.nmea").gga_fixes()

# 1. Anchor the local ENU map frame to the first valid fix.
origin = fixes[0]
fuser = GpsFuser(
    ref_lat=origin.latitude,
    ref_lon=origin.longitude,
    ref_alt=origin.altitude,
)

# 2. Fuse all fixes into a FramePoseSequence (GPS-only trajectory).
seq = FramePoseSequence()
for i, fix in enumerate(fixes):
    fuser.fuse_into_sequence(seq, timestamp=float(i) * 0.1, fix=fix)

# 3. Fuse GPS into a running EKF state (after IMU prediction steps).
ekf   = ImuEkf()
state = EkfState()
for fix in fixes:
    noise = hdop_to_noise(fix.hdop)   # 3×3 ENU position covariance
    state = fuser.fuse_into_ekf(ekf, state, fix, noise)
```

`hdop_to_noise(hdop)` converts the HDOP field from a GGA sentence into a 3×3
diagonal ENU position noise covariance matrix, ready to pass to
`ImuEkf.position_update`.

**`FramePoseSequence` convenience accessors:**

```python
# Get trajectory as numpy arrays for analysis or export:
positions   = seq.positions    # (N, 3) float64 array of [x, y, z]
quaternions = seq.quaternions  # (N, 4) float64 array of [w, x, y, z]

# Export to / load from CSV (timestamp, x, y, z, qw, qx, qy, qz):
seq.to_csv("trajectory.csv")
seq2 = FramePoseSequence.from_csv("trajectory.csv")
```

---

### IMU

Parse binary IMU data files.

```python
from sensor_transposition.imu.imu import ImuParser, load_imu_bin

parser = ImuParser("imu_data.bin")
data   = parser.read()          # structured array: timestamp, ax, ay, az, wx, wy, wz [, qw, qx, qy, qz]

accel  = parser.linear_acceleration()  # (N, 3) float32 – m/s²
gyro   = parser.angular_velocity()     # (N, 3) float32 – rad/s
times  = parser.timestamps()           # (N,)   float64 – UNIX seconds
```

Two record formats are auto-detected by file size:
- **32 bytes/record** – timestamp (f64) + accelerometer (3×f32) + gyroscope (3×f32)
- **48 bytes/record** – above + orientation quaternion (4×f32)

---

### IMU Error-State EKF

15-state Error-State Extended Kalman Filter (ES-EKF) for fusing IMU with GPS
and pose observations.

```python
from sensor_transposition.imu.ekf import ImuEkf, EkfState
import numpy as np

ekf   = ImuEkf(gyro_noise=1e-4, accel_noise=1e-3)
state = EkfState()   # starts at origin, zero velocity, identity orientation

# Predict with IMU sample
accel = np.array([0.0, 0.0, 9.81])  # m/s²
gyro  = np.array([0.0, 0.0, 0.01])  # rad/s
dt    = 0.01                          # seconds
state = ekf.predict(state, accel, gyro, dt)

# Update with a GPS position observation
z_pos   = np.array([1.0, 0.0, 0.0])  # ENU position (metres)
R_noise = np.eye(3) * 0.25           # 3×3 position noise covariance
state   = ekf.position_update(state, z_pos, R_noise)
```

The 15-D error state is ``[δp (3), δv (3), δθ (3), δba (3), δbg (3)]``.
See `docs/state_estimation.md` for a full derivation.

---

### IMU Pre-integration

Accumulate raw IMU measurements between two keyframe timestamps into compact
``(ΔR, Δv, Δp)`` relative-motion increments.

```python
from sensor_transposition.imu.preintegration import ImuPreintegrator
import numpy as np

integrator = ImuPreintegrator()

for accel, gyro, dt in imu_samples:
    integrator.integrate(accel, gyro, dt)

result = integrator.get_result()
print("ΔR:\n", result.delta_rotation)   # (3, 3) rotation matrix
print("Δv:", result.delta_velocity)      # (3,)   m/s
print("Δp:", result.delta_position)      # (3,)   metres
```

---

### Radar

Parse binary radar detection files.

```python
from sensor_transposition.radar.radar import RadarParser, load_radar_bin

parser     = RadarParser("radar_frame.bin")
detections = parser.read()   # structured array: range, azimuth, elevation, velocity, snr
xyz        = parser.xyz()    # (N, 3) Cartesian float32 converted from spherical coords
```

Each 20-byte record contains: `range` (m), `azimuth` (°), `elevation` (°), `velocity` (m/s, negative = approaching), `snr` (dB).

---

### Radar Odometry

Estimate ego-velocity from Doppler measurements, or run scan-to-scan ICP
radar odometry.

```python
from sensor_transposition.radar.radar import RadarParser
from sensor_transposition.radar.radar_odometry import (
    estimate_ego_velocity,
    RadarOdometer,
)

# Doppler-based ego-velocity (single frame)
detections = RadarParser("frame.bin").read()
result = estimate_ego_velocity(detections, min_snr=10.0)
if result.valid:
    print("Ego velocity (m/s):", result.velocity)   # (3,) m/s

# Scan-to-scan odometry over a sequence
odom = RadarOdometer()
for path in sorted(frame_paths):
    xyz = RadarParser(path).xyz()
    step = odom.update(xyz)
    if step is not None:
        print("Relative transform:\n", step.transform)
```

---

### Visual Odometry

Essential-matrix estimation, pose recovery, and Perspective-n-Point (PnP)
solver for monocular or stereo visual odometry.

```python
from sensor_transposition.visual_odometry import (
    estimate_essential_matrix,
    recover_pose_from_essential,
    solve_pnp,
)

# 1. Estimate E from matched pixel pairs (same camera)
result = estimate_essential_matrix(pts1, pts2, K)
E, mask = result.essential_matrix, result.inlier_mask

# 2. Recover relative camera pose
R, t = recover_pose_from_essential(E, pts1[mask], pts2[mask], K)

# 3. Estimate absolute pose from 3-D / 2-D correspondences
pnp = solve_pnp(points_3d, pixels_2d, K)
if pnp.success:
    print("R:\n", pnp.rotation)
    print("t:", pnp.translation)
```

---

### Wheel Odometry

Dead-reckoning pose estimation for differential-drive and Ackermann vehicles.

```python
from sensor_transposition.wheel_odometry import (
    DifferentialDriveOdometer,
    AckermannOdometer,
)

# Differential drive (encoder ticks)
odom = DifferentialDriveOdometer(wheel_base=0.54, wheel_radius=0.1)
result = odom.integrate(timestamps, left_ticks, right_ticks,
                        ticks_per_revolution=360)
print(result.x, result.y, result.theta)   # SE(2) pose

# Ackermann / bicycle model
odom = AckermannOdometer(wheel_base=2.7)
result = odom.integrate(timestamps, speeds, steering_angles)
print(result.x, result.y, result.theta)
```

---

### Loop Closure

Place recognition using Scan Context and M2DP descriptors.

```python
from sensor_transposition.loop_closure import (
    ScanContextDatabase,
    compute_m2dp,
)

db = ScanContextDatabase(num_rings=20, num_sectors=60, max_range=80.0)

for frame_id, cloud in enumerate(lidar_frames):
    # compute_descriptor() uses the database's own parameters — no duplication:
    desc = db.compute_descriptor(cloud)
    candidates = db.query(desc, top_k=1)
    db.add(desc, frame_id=frame_id)

    if candidates and candidates[0].distance < 0.15:
        loop_from = candidates[0].match_frame_id
        print(f"Loop closure: {frame_id} ↔ {loop_from}, "
              f"d={candidates[0].distance:.3f}")

# M2DP as a viewpoint-insensitive alternative
from sensor_transposition.loop_closure import m2dp_distance
desc_a = compute_m2dp(cloud_a)
desc_b = compute_m2dp(cloud_b)
print("M2DP distance:", m2dp_distance(desc_a, desc_b))
```

---

### Pose Graph

Pose graph construction and Gauss-Newton optimisation for graph-SLAM.

```python
from sensor_transposition.pose_graph import PoseGraph, optimize_pose_graph
import numpy as np

graph = PoseGraph()

# Add keyframe nodes
for i, (t, q) in enumerate(zip(translations, quaternions)):
    graph.add_node(i, translation=t, quaternion=q)

# Add odometry edges from ICP
for i in range(len(translations) - 1):
    result = icp_align(clouds[i], clouds[i + 1])
    graph.add_edge(from_id=i, to_id=i + 1,
                   transform=result.transform,
                   information=np.eye(6) * 100.0)

# Add a loop-closure edge
graph.add_edge(from_id=loop_from, to_id=loop_to,
               transform=loop_transform,
               information=np.eye(6) * 50.0)

opt = optimize_pose_graph(graph)
if opt.success:
    for node_id, pose in opt.optimized_poses.items():
        print(node_id, pose["translation"])

# Optional progress callback (iteration number + current cost):
opt = optimize_pose_graph(
    graph,
    callback=lambda i, c: print(f"iter {i}: cost={c:.4f}"),
)
```

---

### Sliding-Window Smoother

Fixed-lag online SLAM smoother that keeps only the most recent `window_size`
keyframes in the active optimisation window.

```python
from sensor_transposition.sliding_window import SlidingWindowSmoother
import numpy as np

smoother = SlidingWindowSmoother(window_size=5)

for i, (trans, rel_tf) in enumerate(keyframe_stream):
    smoother.add_node(i, translation=trans)
    if i > 0:
        smoother.add_edge(i - 1, i, transform=rel_tf,
                          information=np.eye(6) * 200.0)
    result = smoother.optimize()
    if result.success:
        print(f"Node {i}: {result.optimized_poses[i]['translation']}")

# Progress callback is supported:
result = smoother.optimize(callback=lambda i, c: print(f"iter {i}: cost={c:.4f}"))
```

---

### Submap Manager

Keyframe selection and submap division for large-scale or long-duration SLAM.

```python
from sensor_transposition.submap_manager import KeyframeSelector, SubmapManager
import numpy as np

selector = KeyframeSelector(translation_threshold=1.0,
                            rotation_threshold_deg=10.0)
manager  = SubmapManager(max_keyframes_per_submap=20, overlap=2)

for frame_id, (pose, scan) in enumerate(zip(frame_poses, lidar_scans)):
    if selector.check_and_accept(pose.transform):
        manager.add_keyframe(frame_id, pose.transform, scan)

submaps = manager.get_all_submaps()
print(f"Created {len(submaps)} submaps.")
```

---

### Occupancy Grid

2-D probabilistic occupancy grid built from LiDAR scans using log-odds
ray-casting.

```python
from sensor_transposition.occupancy_grid import OccupancyGrid
import numpy as np

grid = OccupancyGrid(
    resolution=0.10,                    # 10 cm/cell
    width=200, height=200,              # 20 m × 20 m
    origin=np.array([-10.0, -10.0]),    # world coords of cell (0, 0)
)

for frame_pose, lidar_scan in zip(trajectory, scans):
    sensor_origin = frame_pose.transform[:3, 3]
    grid.insert_scan(lidar_scan, frame_pose.transform, sensor_origin)

occupancy = grid.get_grid()       # (height, width) int8 — ROS convention
probs     = grid.to_probability() # (height, width) float64 in [0, 1]

# ROS nav_msgs/OccupancyGrid compatibility:
# grid.to_ros_int8() returns a 2-D int8 array using the ROS convention:
#   -1 = unknown,  0 = free,  100 = occupied
ros_grid = grid.to_ros_int8()
```

---

### Voxel Map (TSDF)

Truncated Signed-Distance Function (TSDF) volumetric map for dense 3-D
reconstruction.

```python
import numpy as np
from sensor_transposition.voxel_map import TSDFVolume

volume = TSDFVolume(
    voxel_size=0.10,
    origin=np.array([-10.0, -10.0, -0.5]),
    dims=(200, 200, 50),   # 200×200×50 voxels
)

for frame_pose, lidar_scan in zip(trajectory, scans):
    volume.integrate(lidar_scan, frame_pose.transform)

surface_pts  = volume.extract_surface_points(threshold=0.1)
tsdf_array   = volume.get_tsdf()     # (nx, ny, nz) float64; NaN = unseen
weight_array = volume.get_weights()  # (nx, ny, nz) float64
```

---

### Point-Cloud Map

Accumulated coloured point-cloud map assembled from successive LiDAR scans,
with voxel-grid downsampling and PCD / PLY file I/O.

```python
from sensor_transposition.point_cloud_map import PointCloudMap

pcd_map = PointCloudMap()

for frame_pose, lidar_scan in zip(trajectory, scans):
    pcd_map.add_scan(lidar_scan, frame_pose.transform)

# Downsample to 10-cm voxels
pcd_map.voxel_downsample(voxel_size=0.10)

world_points = pcd_map.get_points()   # (N, 3) float64
world_colors = pcd_map.get_colors()   # (N, 3) uint8, or None

# Save / load
pcd_map.save_pcd("map.pcd")
pcd_map.save_ply("map.ply")
loaded = PointCloudMap.from_pcd("map.pcd")
```

---

### Visualisation

BEV rendering, trajectory overlay, LiDAR-on-image overlay, and export helpers
for Open3D and RViz.  All renderers return plain NumPy `uint8` RGB arrays.

```python
from sensor_transposition.visualisation import (
    render_birdseye_view,
    render_trajectory_birdseye,
    overlay_lidar_on_image,
    export_point_cloud_open3d,
    colour_by_height,
    color_by_height,      # American-spelling alias
    SensorFrameVisualiser,
)

# Bird's-eye view of the accumulated map
bev = render_birdseye_view(map_points, resolution=0.10)

# Colour a point cloud by height (both spellings accepted)
colours = colour_by_height(points[:, 2])   # (N, 3) uint8
colours = color_by_height(points[:, 2])    # identical result

# Overlay depth-coded LiDAR on a camera image
from sensor_transposition.lidar_camera import project_lidar_to_image
pixel_coords, valid = project_lidar_to_image(lidar_scan, T, K, W, H)
depth   = lidar_scan[:, 0]
overlay = overlay_lidar_on_image(camera_image, pixel_coords, valid, depth)

# Per-frame container — populate field-by-field:
vis = SensorFrameVisualiser()
vis.set_point_cloud(lidar_scan)
vis.set_camera_image(camera_image)
bev_frame = vis.render_birdseye(resolution=0.10)

# Or construct from a bag message payload in one step:
vis = SensorFrameVisualiser.from_dict({
    "point_cloud": lidar_scan,
    "camera_image": camera_image,
    "trajectory": trajectory_xy,
    "radar_scan": radar_points,
})
bev_frame = vis.render_birdseye(resolution=0.10)

# Export for Open3D
o3d_dict = export_point_cloud_open3d(map_points)
```

---

### Bag Recorder / Player

Lightweight multi-topic binary bag format (`.sbag`) for recording and replaying
multi-sensor data.  No external dependencies — pure Python standard library.

```python
from sensor_transposition.rosbag import BagWriter, BagReader
import numpy as np

# Record — numpy arrays are automatically converted (no .tolist() needed):
with BagWriter("session.sbag") as bag:
    bag.write("/lidar/points", timestamp, {"xyz": lidar_pts})  # numpy OK
    bag.write("/imu/data",     timestamp, {"accel": accel_arr, "gyro": gyro_arr})

# Replay
with BagReader("session.sbag") as bag:
    print("Topics:", bag.topics)
    for msg in bag.read_messages(topics=["/lidar/points"]):
        xyz = msg.data["xyz"]   # plain Python list after round-trip
```

The `.sbag` extension stands for **sensor bag**.  Each message record stores a
UTF-8 JSON payload, so files can be inspected with any text editor after
stripping the binary header.  See `docs/rosbag.md` for the full format
specification.

#### Converting to MCAP (ROS 2 compatible)

Use `sbag_to_rosbag` to convert a `.sbag` file to
[MCAP](https://mcap.dev/) format, which can be opened with `ros2 bag`,
the MCAP CLI, or Foxglove Studio (requires `pip install ".[mcap]"`):

```python
from sensor_transposition.rosbag import sbag_to_rosbag

sbag_to_rosbag("session.sbag", "session.mcap")
# Inspect with: ros2 bag info session.mcap
```

---

### Camera–LiDAR Extrinsic Calibration

Target-based extrinsic calibration using plane correspondences observed from
both the camera and the LiDAR.

```python
from sensor_transposition.calibration import (
    fit_plane,
    ransac_plane,
    calibrate_lidar_camera,
)
# Also importable directly from sensor_transposition:
from sensor_transposition import fit_plane, ransac_plane, calibrate_lidar_camera
import numpy as np

# For each pose of a planar calibration target:
lidar_normals    = []
lidar_distances  = []
camera_normals   = []
camera_distances = []

# LiDAR side — fit a plane to the board region
normal, dist, inliers = ransac_plane(board_lidar_pts, distance_threshold=0.02)
lidar_normals.append(normal)
lidar_distances.append(dist)

# Camera side — use a PnP solver (e.g. cv2.solvePnP) to obtain the
# board normal and signed distance in the camera frame, then append.

# Solve for the 4×4 LiDAR → camera transform (need ≥ 3 observations)
T_lidar_to_cam = calibrate_lidar_camera(
    np.array(lidar_normals),   np.array(lidar_distances),
    np.array(camera_normals),  np.array(camera_distances),
)
```

See [`docs/camera_lidar_extrinsic_calibration.md`](docs/camera_lidar_extrinsic_calibration.md)
for a complete worked example including the camera-side PnP procedure.

---

### SLAM Session (Pipeline Orchestration)

:class:`SLAMSession` wires together the core SLAM modules — ICP scan matching,
Scan Context loop closure, pose-graph optimisation, and point-cloud map
accumulation — into a single object with sensible defaults.

```python
from sensor_transposition.rosbag import BagReader
from sensor_transposition.slam_session import SLAMSession

# Run the pipeline
with BagReader("session.sbag") as bag:
    session = SLAMSession(
        icp_max_iterations=50,
        loop_closure_threshold=0.15,
    )
    session.run(bag, lidar_topic="/lidar/points")

# Optimise pose graph and rebuild map with corrected poses
session.optimize()

# Save outputs
session.point_cloud_map.voxel_downsample(voxel_size=0.10)
session.point_cloud_map.save_pcd("map.pcd")
session.trajectory.to_csv("trajectory.csv")
```

Register callbacks for other sensor topics:

```python
session = SLAMSession()

@session.on_topic("/imu/data")
def handle_imu(msg):
    print(f"IMU @ {msg.timestamp:.3f}s: accel={msg.data['accel']}")

with BagReader("session.sbag") as bag:
    session.run(bag)
```

All internal components are accessible as properties for advanced use:
`session.pose_graph`, `session.loop_db`, `session.trajectory`,
`session.point_cloud_map`.

---

## Error Handling

`sensor_transposition` defines a small hierarchy of library-specific exceptions
in `sensor_transposition.exceptions`.  Every exception inherits from the base
class **`SensorTranspositionError`** so a single `except` clause can catch any
library error.  Each subclass also inherits from the appropriate standard
Python exception so existing `except KeyError / RuntimeError / ValueError`
handlers continue to work without change.

| Exception | Inherits from | Raised when |
|---|---|---|
| `SensorTranspositionError` | `Exception` | Base class for all library errors |
| `SensorNotFoundError` | `SensorTranspositionError`, `KeyError` | `SensorCollection.get_sensor()` — sensor name not found |
| `BagError` | `SensorTranspositionError`, `RuntimeError` | `BagWriter.write()` / `BagReader.read_messages()` — writer or reader is closed |
| `CalibrationError` | `SensorTranspositionError`, `ValueError` | Calibration operations that fail with invalid input |

```python
from sensor_transposition.exceptions import (
    SensorTranspositionError,
    SensorNotFoundError,
    BagError,
)

# Catch a specific library exception:
try:
    sensor = collection.get_sensor("unknown_lidar")
except SensorNotFoundError as exc:
    print(f"Sensor not found: {exc}")

# Or catch all library exceptions at once:
try:
    bag.write("/lidar/points", timestamp, payload)
except SensorTranspositionError as exc:
    print(f"sensor_transposition error: {exc}")

# Existing KeyError / RuntimeError handlers also work unchanged:
try:
    sensor = collection.get_sensor("front_camera")
except KeyError:
    pass  # SensorNotFoundError is a KeyError
```

All four classes are exported from the top-level package:

```python
import sensor_transposition as st
st.SensorNotFoundError
st.BagError
st.CalibrationError
st.SensorTranspositionError
```

---

## Configuration Example

See [`examples/sensor_collection.yaml`](examples/sensor_collection.yaml) for a full vehicle rig including a front camera, front/rear LiDARs, a front radar, a GNSS antenna, and an IMU.

```yaml
sensors:
  front_lidar:
    type: lidar
    coordinate_system: FLU
    extrinsics:
      translation: [1.84, 0.00, 1.91]
      rotation:
        quaternion: [1.0, 0.0, 0.0, 0.0]
      time_offset_sec: 0.0              # LiDAR is the reference clock

  front_camera:
    type: camera
    coordinate_system: RDF
    extrinsics:
      translation: [1.80, 0.00, 1.45]
      rotation:
        quaternion: [0.5, -0.5, 0.5, -0.5]
      time_offset_sec: 0.033            # camera triggers ~33 ms after reference
    intrinsics:
      fx: 1266.417
      fy: 1266.417
      cx: 816.0
      cy: 612.0
      width: 1632
      height: 1224
      distortion_coefficients: [-0.05, 0.08, 0.0, 0.0, -0.03]
```

---

## Quick-Start: End-to-End SLAM Pipeline

[`examples/slam_pipeline.py`](examples/slam_pipeline.py) demonstrates a
complete offline SLAM pipeline using **synthetic** (randomly generated) data,
so it runs immediately without any real sensor hardware:

```
python examples/slam_pipeline.py
```

The script walks through each stage of the pipeline:

1. **Generate** synthetic LiDAR scans and IMU samples.
2. **Record** data to a `.sbag` file with `BagWriter`.
3. **Replay** the bag with `BagReader` and extract LiDAR scans.
4. **ICP odometry** – align consecutive scans with `icp_align`.
5. **Loop closure** – detect revisits with `ScanContextDatabase`.
6. **Pose graph optimisation** – correct accumulated drift with
   `optimize_pose_graph`.
7. **Map accumulation** – build a `PointCloudMap` from optimised poses and
   save it as a `.pcd` file.

---

## ROS Examples

Ready-to-use driver configurations for Velodyne and Ouster LiDARs are provided in the [`ros_examples/`](ros_examples/) directory.

```
ros_examples/
├── ros/
│   ├── velodyne.launch       # ROS 1 – VLP-16 / VLP-32C / HDL-32E / HDL-64E
│   └── ouster.launch         # ROS 1 – OS0 / OS1 / OS2
└── ros2/
    ├── velodyne_params.yaml  # ROS 2 – velodyne_driver + velodyne_pointcloud
    └── ouster_params.yaml    # ROS 2 – ouster_ros (ROS 2 branch)
```

### ROS 1

```bash
# Velodyne VLP-16 (live sensor)
roslaunch ros_examples/ros/velodyne.launch device_ip:=192.168.1.201 model:=VLP16

# Ouster OS1
roslaunch ros_examples/ros/ouster.launch sensor_hostname:=os-<serial>.local
```

### ROS 2

```bash
# Velodyne VLP-16
ros2 launch velodyne velodyne-all-nodes-VLP16-launch.py \
    params_file:=ros_examples/ros2/velodyne_params.yaml

# Ouster OS1
ros2 launch ouster_ros driver.launch.py \
    params_file:=ros_examples/ros2/ouster_params.yaml
```

Both ROS 2 YAML files use the `ros__parameters` namespace convention and can be passed directly via `params_file:=`.
