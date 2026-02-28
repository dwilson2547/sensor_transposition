# SLAM Workflow Gap Analysis

This document analyses the current `sensor_transposition` toolset and identifies what is present, what is partially present, and what is missing for a **full Simultaneous Localisation and Mapping (SLAM) pipeline**.

---

## What the Repository Currently Provides

| Area | Module(s) | Status |
|------|-----------|--------|
| Multi-sensor rig configuration (YAML I/O) | `sensor_collection.py` | ✅ Complete |
| Homogeneous transform math (4×4, compose, invert, apply) | `transform.py` | ✅ Complete |
| Pinhole camera model (focal length, projection, Brown–Conrady distortion) and fisheye / omnidirectional model (Kannala-Brandt, equidistant, FOV up to 360°) | `camera_intrinsics.py` | ✅ Complete |
| LiDAR–camera projection and point-cloud colouring | `lidar_camera.py` | ✅ Complete |
| LiDAR binary parsers (Velodyne KITTI, Ouster 4/8-col, Livox LVX/LVX2) | `lidar/` | ✅ Complete |
| NMEA 0183 GPS parser (GGA, RMC) | `gps/nmea.py` | ✅ Complete |
| GPS coordinate-frame converter (ECEF ↔ ENU / UTM) | `gps/converter.py` | ✅ Complete |
| IMU binary parser (32-byte and 48-byte records) | `imu/imu.py` | ✅ Complete |
| IMU pre-integration (ΔR, Δv, Δp via midpoint method) | `imu/preintegration.py` | ✅ Complete |
| Radar binary parser (spherical → Cartesian) | `radar/radar.py` | ✅ Complete |
| Trajectory storage (`FramePose`, `FramePoseSequence`, YAML I/O) | `frame_pose.py` | ✅ Complete |
| Multi-sensor time synchronisation and interpolation | `sync.py` | ✅ Complete |
| ROS 1 & 2 launch/parameter files for LiDAR, camera, GPS, IMU, and radar sensors | `ros_examples/` | ✅ Complete |
| Calibration and data-collection documentation | `docs/` | ✅ Complete |

---

## Full SLAM Workflow — Stage-by-Stage Gap Analysis

A production SLAM pipeline is typically divided into the stages below. Each gap is rated by priority: **High** (blocks the pipeline), **Medium** (limits quality or generality), **Low** (nice-to-have).

---

### 1. Sensor Setup & Calibration

**Present:**
- Extrinsic calibration (sensor-to-ego and sensor-to-sensor transforms via `SensorCollection`)
- Intrinsic calibration model (pinhole + Brown–Conrady) for perspective cameras
- Example YAML rig definition with camera, LiDAR, radar, GPS, and IMU

**Gaps:**

| Gap | Priority | Notes |
|-----|----------|-------|
| Fisheye / omnidirectional camera model (Kannala–Brandt / equidistant) | High | ✅ Added `fisheye_focal_length_from_fov`, `fisheye_distort_point`, `fisheye_undistort_point`, `fisheye_project_point`, and `fisheye_unproject_pixel` to `camera_intrinsics.py`; supports FOV up to 360°. |
| Time-offset (temporal extrinsic) calibration between sensors | High | ✅ Added `time_offset_sec` field to `Sensor` and `time_offset_between()` to `SensorCollection`; `sync.py` applies the offset when aligning streams. |
| Camera-LiDAR target-based extrinsic calibration tooling | Medium | ✅ Added `docs/camera_lidar_extrinsic_calibration.md` with a full checkerboard/ArUco-target workflow for computing and saving the extrinsic transform. |
| IMU-to-vehicle extrinsic + bias/scale calibration | Medium | ✅ Added `ImuParameters` dataclass to `SensorCollection` with Allan-variance noise-density and random-walk fields, plus calibrated bias and scale-factor vectors. |
| Rolling-shutter model for cameras | Medium | ✅ Added `rolling_shutter_row_time`, `rolling_shutter_correct_point`, and `rolling_shutter_project_point` to `camera_intrinsics.py`; first-order constant-velocity correction with fixed-point iteration for row convergence. |

---

### 2. Sensor Data Acquisition & Synchronisation

**Present:**
- Binary and text parsers for all major sensor types
- Frame-duration concept in `FramePoseSequence`

**Gaps:**

| Gap | Priority | Notes |
|-----|----------|-------|
| Multi-sensor time synchronisation / interpolation | High | ✅ Added `sync.py` with `SensorSynchroniser` class and standalone `apply_time_offset`, `find_nearest_indices`, and `interpolate_timestamps` helpers; integrates with `Sensor.time_offset_sec`. |
| ROS launch/parameter files for cameras, GPS, IMU, and radar | High | ✅ Added ROS 1 launch files and ROS 2 parameter files for USB camera (`usb_cam`), NMEA GPS, MicroStrain IMU, and TI mmWave radar in `ros_examples/`. |
| GPS / GNSS ROS driver integration and RTK setup guide | High | `gps/nmea.py` parses log files but there is no live-driver config or RTK correction pipeline. Noted in `TODO.md`. |
| IMU ROS driver integration | Medium | ✅ Added `ros_examples/ros/microstrain_imu.launch` (ROS 1) and `ros_examples/ros2/microstrain_imu_params.yaml` (ROS 2) for the MicroStrain IMU driver. |
| Radar ROS driver integration | Medium | ✅ Added `ros_examples/ros/ti_mmwave_radar.launch` (ROS 1) and `ros_examples/ros2/ti_mmwave_radar_params.yaml` (ROS 2) for the TI mmWave radar driver. |
| Rosbag / MCAP recording and playback utilities | Medium | ✅ Added `rosbag.py` with `BagWriter` (streaming write), `BagReader` (indexed read), and `BagMessage` dataclass; simple binary bag format with topic + timestamp + JSON payload; topic and time-range filtering; no external dependencies. See `docs/rosbag.md`. |
| Data capture instructions and intrinsic calculation guide | Medium | ✅ Added data collection guides in `docs/` for cameras (pinhole: `camera_intrinsics_guide.md`; fisheye: `fisheye_camera_intrinsics_guide.md`), GPS (`gps_data_collection.md`), LiDAR (`lidar_data_collection.md`), and radar (`radar_data_collection.md`). |

---

### 3. Motion Estimation (Odometry Front-End)

**Present:**
- Trajectory storage structure (`FramePose`, `FramePoseSequence`)
- Coordinate-frame transforms for stitching sensor observations

**Gaps:**

| Gap | Priority | Notes |
|-----|----------|-------|
| LiDAR odometry / scan matching (ICP, NDT, or LOAM-style) | High | ✅ Added `lidar/scan_matching.py` with point-to-point ICP (`icp_align`) using the Kabsch SVD algorithm and a scipy KD-tree for nearest-neighbour search; supports max-correspondence distance filtering, an optional initial transform, and convergence tolerance. |
| IMU pre-integration | High | ✅ Added `imu/preintegration.py` with `ImuPreintegrator` class; uses midpoint (trapezoidal) method to accumulate ΔR, Δv, Δp between keyframe timestamps. |
| Visual odometry (feature tracking / direct methods) | High | ✅ Added `visual_odometry.py` with `estimate_essential_matrix` (normalised 8-point algorithm + RANSAC), `recover_pose_from_essential` (SVD decomposition + cheirality test), and `solve_pnp` (DLT + RANSAC); pure NumPy/SciPy implementation. |
| Wheel odometry / vehicle kinematic model | Medium | ✅ Added `wheel_odometry.py` with `DifferentialDriveOdometer` (midpoint integration from wheel speeds or encoder ticks) and `AckermannOdometer` (bicycle model from speed + steering angle); both produce an `OdometryResult` with SE(2) pose, 4×4 transform, duration, and sample count. Functional wrappers `integrate_differential_drive` and `integrate_ackermann` also provided. See `docs/wheel_odometry.md`. |
| GPS-to-local-frame converter (ECEF ↔ ENU / UTM) | Medium | ✅ Added `gps/converter.py` with `geodetic_to_ecef`, `ecef_to_geodetic`, `ecef_to_enu`, `enu_to_ecef`, `geodetic_to_enu`, `geodetic_to_utm`, and `utm_to_geodetic`. |

---

### 4. Loop Closure Detection

**Present:**
- Scan Context descriptor (`compute_scan_context`, `ScanContextDatabase`) for appearance-based loop closure
- M2DP descriptor (`compute_m2dp`, `m2dp_distance`) for viewpoint-insensitive place recognition

**Gaps:**

| Gap | Priority | Notes |
|-----|----------|-------|
| Place recognition / appearance-based descriptor (DBoW, NetVLAD, etc.) | High | ✅ Added `loop_closure.py` with `compute_scan_context` (Scan Context polar-grid descriptor), `scan_context_distance` (rotation-invariant normalised cosine distance with column-shift search), and `ScanContextDatabase` (two-stage ring-key pre-filter + full descriptor search with exclusion window). Pure NumPy/SciPy; no additional dependencies. |
| Geometry-based loop closure verification (ICP/NDT re-alignment) | High | ✅ Handled by the existing `lidar/scan_matching.py` `icp_align` function; after a loop candidate is found via `ScanContextDatabase.query`, callers pass the two point clouds to `icp_align` for geometric verification and edge estimation. |
| LiDAR descriptor matching (Scan Context, M2DP, etc.) | Medium | ✅ Added `compute_m2dp`, `M2dpDescriptor`, and `m2dp_distance` to `loop_closure.py`; projects the point cloud onto multiple oriented planes, bins point density into polar histograms, and compresses via SVD (first left + right singular vectors). Complements Scan Context for environments where the ground-plane assumption does not hold. See `docs/loop_closure.md`. |

---

### 5. State Estimation & Sensor Fusion

**Present:**
- Individual sensor parsers providing raw measurements.

**Gaps:**

| Gap | Priority | Notes |
|-----|----------|-------|
| Extended / Unscented Kalman Filter (EKF/UKF) for pose estimation | High | ✅ Added `imu/ekf.py` with `EkfState` and `ImuEkf`; implements a 15-state Error-State EKF with `predict` (IMU propagation), `position_update` (GPS/odometry), `velocity_update` (wheel odometry/GPS-Doppler), and `pose_update` (6-DOF LiDAR scan-matching); Joseph-form covariance update for numerical stability; pure NumPy. See `docs/state_estimation.md`. |
| IMU-LiDAR tightly/loosely coupled fusion | High | ✅ Added `lidar/motion_distortion.py` with `deskew_scan`; integrates IMU gyro + accelerometer via midpoint method, SLERP for rotation and linear interpolation for position, producing per-point relative transforms w.r.t. a configurable reference timestamp. See `docs/lidar_motion_distortion.md`. |
| GPS / GNSS absolute-position fusion | Medium | ✅ Added `GpsFuser` class and `hdop_to_noise` helper to `gps/fusion.py`; converts GGA/RMC fixes to local ENU, fuses into `ImuEkf` via `position_update`, and adds/updates `FramePose` entries in a `FramePoseSequence`. See `docs/gps_fusion.md`. |
| Radar-odometry or radar-SLAM integration | Low | Radar can complement LiDAR in adverse weather; no fusion mechanism exists. |

---

### 6. Back-End Pose Graph Optimisation

**Present:**
- `FramePoseSequence` stores poses but applies no optimisation.

**Gaps:**

| Gap | Priority | Notes |
|-----|----------|-------|
| Pose graph construction (nodes = poses, edges = relative constraints) | High | ✅ Added `pose_graph.py` with `PoseGraph` (`add_node`, `add_edge`) and `PoseGraphNode` / `PoseGraphEdge` dataclasses; integrates with `FramePoseSequence`, `icp_align`, and `ScanContextDatabase`. |
| Non-linear graph optimisation (g2o, GTSAM, Ceres, or iSAM2) | High | ✅ Added `optimize_pose_graph` in `pose_graph.py`; Gauss-Newton with numerical Jacobians, LM damping, and gauge-freedom fix (first node held constant); pure NumPy/SciPy. See `docs/pose_graph_optimisation.md`. |
| Marginalisation / sliding-window optimisation for online SLAM | Medium | ✅ Added `sliding_window.py` with `SlidingWindowSmoother`; fixed-lag smoother with world-frame prior factors derived from marginalised nodes; bounds per-step cost to O(window_size³); pure NumPy/SciPy. See `docs/sliding_window.md`. |
| Uncertainty / covariance propagation in `FramePose` | Medium | `FramePose` stores only the mean pose; no covariance matrix is tracked. |

---

### 7. Map Representation

**Present:**
- Nothing — the library is calibration/parsing focused and produces no persistent map.

**Gaps:**

| Gap | Priority | Notes |
|-----|----------|-------|
| Accumulated point-cloud / surfel map | High | ✅ Added `point_cloud_map.py` with `PointCloudMap`; supports per-scan ego-to-world transform, optional per-point RGB colour (uint8 or float), FIFO `max_points` cap, and voxel-grid centroid downsampling. See `docs/point_cloud_map.md`. |
| Occupancy grid (2-D or 3-D) | Medium | Required for path planning and obstacle avoidance. |
| Voxel map / TSDF (TSDFusion, VDB, etc.) | Medium | Memory-efficient volumetric representation used for dense reconstruction. |
| Map serialisation (PCD, PLY, or custom binary format) | Medium | No mechanism to save or load a built map. |
| Map management (submap division, keyframe selection) | Low | Needed for large-scale or long-duration sessions. |

---

### 8. Visualisation

**Present:**
- No visualisation tooling.

**Gaps:**

| Gap | Priority | Notes |
|-----|----------|-------|
| Multi-sensor data viewer (point clouds, images, trajectory) | High | ✅ Added `visualisation.py` with `render_birdseye_view` (height-coded BEV image), `render_trajectory_birdseye`, `overlay_lidar_on_image`, `export_point_cloud_open3d`, `export_trajectory_rviz`, and `SensorFrameVisualiser` (per-frame multi-sensor container); pure NumPy. See `docs/visualisation.md`. |
| Real-time trajectory and map visualisation (RViz, Open3D, rerun.io) | High | ✅ `export_point_cloud_open3d` and `export_trajectory_rviz` in `visualisation.py` serialise data to Open3D-compatible dicts and RViz `Marker` dicts respectively; `SensorFrameVisualiser.render_birdseye` provides a dependency-free BEV view for debugging. |
| LiDAR–camera overlay visualisation | Medium | ✅ `overlay_lidar_on_image` in `visualisation.py` depth-codes projected LiDAR points and overlays them on a camera image; `SensorFrameVisualiser.render_camera_with_lidar` exposes the same helper through the frame container. |

---

## Summary Checklist

The following is a consolidated list of all identified gaps, ordered roughly by the impact they have on completing a working SLAM pipeline:

### Blocking / High Priority
- [X] Fisheye/omnidirectional camera model (Kannala–Brandt)
- [X] Multi-sensor time synchronisation / interpolation utilities
- [X] ROS launch/parameter files for cameras, GPS, IMU, and radar
- [X] GPS/RTK driver integration and ECEF/ENU/UTM conversion
- [X] IMU pre-integration
- [X] LiDAR odometry / scan matching (ICP, NDT; Kiss-ICP integration)
- [X] Visual odometry (feature tracking, essential matrix, PnP)
- [X] Place recognition / loop closure detection
- [X] EKF/UKF state estimator for IMU + odometry fusion
- [X] IMU-LiDAR tightly coupled fusion (motion-distortion correction)
- [X] Pose graph data structure and optimisation back-end (g2o / GTSAM / Ceres)
- [X] Accumulated point-cloud or surfel map output
- [X] Multi-sensor synchronised visualisation platform

### Medium Priority
- [X] Temporal extrinsic (clock-offset) calibration field in `SensorCollection`
- [X] Camera-LiDAR target-based extrinsic calibration workflow
- [X] IMU noise model / Allan-variance parameters in `SensorCollection`
- [X] Rolling-shutter camera model and correction
- [X] IMU and radar ROS driver examples
- [X] Rosbag / MCAP recording and playback utilities
- [X] Data capture and intrinsic calculation guide
- [X] Wheel odometry / vehicle kinematic model
- [X] LiDAR descriptor-based place recognition (Scan Context, M2DP)
- [X] GPS absolute-position fusion into local map
- [X] Sliding-window / fixed-lag smoother for online SLAM
- [ ] Covariance tracking in `FramePose`
- [ ] Occupancy grid (2-D / 3-D)
- [ ] Voxel map / TSDF volumetric representation
- [ ] Map serialisation (PCD, PLY)
- [X] LiDAR–camera overlay display utility

### Low Priority
- [ ] Radar-odometry / radar-SLAM integration
- [ ] Map management (submap division, keyframe selection)
- [ ] Marginalisation for very long trajectories
