# sensor_transposition

A Python toolkit for multi-sensor calibration, data parsing, and coordinate-frame management in autonomous-vehicle and robotics pipelines.

---

## Contents

- [Features](#features)
- [Installation](#installation)
- [Coordinate Systems](#coordinate-systems)
- [Modules](#modules)
  - [SensorCollection](#sensorcollection)
  - [Camera Intrinsics](#camera-intrinsics)
  - [Fisheye / Omnidirectional Camera](#fisheye--omnidirectional-camera)
  - [Transform](#transform)
  - [LiDAR–Camera Fusion](#lidar-camera-fusion)
  - [LiDAR Parsers](#lidar-parsers)
  - [GPS / GNSS](#gps--gnss)
  - [IMU](#imu)
  - [Radar](#radar)
- [Configuration Example](#configuration-example)
- [ROS Examples](#ros-examples)

---

## Features

- **Sensor collection** – YAML-driven multi-sensor configuration storing extrinsics (translation + quaternion) and per-sensor intrinsics/parameters for cameras, LiDARs, radars, GPS, and IMUs.
- **Pinhole camera model** – focal-length derivation from FOV or physical sensor geometry, camera matrix construction, point projection / unprojection, and Brown–Conrady lens distortion/undistortion.
- **Fisheye / omnidirectional camera model** – Kannala-Brandt equidistant projection supporting fields of view up to 360°, with focal-length derivation, point projection / unprojection, and Kannala-Brandt distortion/undistortion (compatible with OpenCV's `cv2.fisheye` module).
- **Homogeneous transforms** – composable 4×4 `Transform` objects with helpers for building from quaternions or rotation matrices, inverting, and applying to point clouds.
- **LiDAR–camera fusion** – project 3-D LiDAR point clouds onto a camera image plane and colour the cloud by sampling pixel values.
- **LiDAR parsers** – binary readers for Velodyne (KITTI `.bin`), Ouster (4-column and 8-column `.bin`), and Livox (LVX / LVX2) file formats.
- **GPS / GNSS** – NMEA 0183 parser supporting GGA and RMC sentence types.
- **IMU** – binary parser for 32-byte (accel + gyro) and 48-byte (accel + gyro + quaternion) records.
- **Radar** – binary parser for 5-field (range, azimuth, elevation, velocity, SNR) detection records with spherical → Cartesian conversion.
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
Each entry specifies `coordinate_system`, `extrinsics` (translation + quaternion), and optional type-specific parameter blocks.

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

  front_camera:
    type: camera
    coordinate_system: RDF
    extrinsics:
      translation: [1.80, 0.00, 1.45]
      rotation:
        quaternion: [0.5, -0.5, 0.5, -0.5]
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
