# SLAM Workflow Gap Analysis

This document analyses the current `sensor_transposition` toolset and identifies what is present, what is partially present, and what is missing for a **full Simultaneous Localisation and Mapping (SLAM) pipeline**.

---

## What the Repository Currently Provides

| Area | Module(s) | Status |
|------|-----------|--------|
| Multi-sensor rig configuration (YAML I/O) | `sensor_collection.py` | ✅ Complete |
| Homogeneous transform math (4×4, compose, invert, apply) | `transform.py` | ✅ Complete |
| Pinhole camera model (focal length, projection, Brown–Conrady distortion) | `camera_intrinsics.py` | ✅ Complete |
| LiDAR–camera projection and point-cloud colouring | `lidar_camera.py` | ✅ Complete |
| LiDAR binary parsers (Velodyne KITTI, Ouster 4/8-col, Livox LVX/LVX2) | `lidar/` | ✅ Complete |
| NMEA 0183 GPS parser (GGA, RMC) | `gps/nmea.py` | ✅ Complete |
| IMU binary parser (32-byte and 48-byte records) | `imu/imu.py` | ✅ Complete |
| Radar binary parser (spherical → Cartesian) | `radar/radar.py` | ✅ Complete |
| Trajectory storage (`FramePose`, `FramePoseSequence`, YAML I/O) | `frame_pose.py` | ✅ Complete |
| ROS 1 & 2 launch/parameter files for Velodyne and Ouster | `ros_examples/` | ✅ Complete |
| Calibration and data-collection documentation | `docs/` | ✅ Partial |

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
| Fisheye / omnidirectional camera model (Kannala–Brandt / equidistant) | High | Wide-angle and 270° fisheye lenses are common in automotive SLAM; the pinhole model is insufficient at large FOVs. Noted in `TODO.md`. |
| Time-offset (temporal extrinsic) calibration between sensors | High | Hardware-timestamped sensors may have a fixed clock offset; there is no field or utility for this in `SensorCollection`. |
| Camera-LiDAR target-based extrinsic calibration tooling | Medium | The library stores calibration results but provides no workflow (e.g., checkerboard or ArUco target detection) to compute them. `docs/camera_intrinsics_guide.md` covers intrinsics only. |
| IMU-to-vehicle extrinsic + bias/scale calibration | Medium | `SensorCollection` stores IMU extrinsics but there is no IMU calibration or noise-model field (Allan-variance parameters). |
| Rolling-shutter model for cameras | Medium | High-speed cameras used in SLAM often have a rolling shutter; no model or correction exists. |

---

### 2. Sensor Data Acquisition & Synchronisation

**Present:**
- Binary and text parsers for all major sensor types
- Frame-duration concept in `FramePoseSequence`

**Gaps:**

| Gap | Priority | Notes |
|-----|----------|-------|
| Multi-sensor time synchronisation / interpolation | High | No utility to align data from different sensors to a common timeline. Needed before any sensor-fusion step. |
| ROS launch/parameter files for cameras, GPS, IMU, and radar | High | Only LiDAR ROS examples exist. Noted in `TODO.md`. |
| GPS / GNSS ROS driver integration and RTK setup guide | High | `gps/nmea.py` parses log files but there is no live-driver config or RTK correction pipeline. Noted in `TODO.md`. |
| IMU ROS driver integration | Medium | Parsing works on offline files; no ROS node config or live-streaming example. Noted in `TODO.md`. |
| Radar ROS driver integration | Medium | Same situation as IMU. Noted in `TODO.md`. |
| Rosbag / MCAP recording and playback utilities | Medium | No tooling to record or replay multi-sensor data in a ROS-compatible bag format. |
| Data capture instructions and intrinsic calculation guide | Medium | Noted in `TODO.md`. |

---

### 3. Motion Estimation (Odometry Front-End)

**Present:**
- Trajectory storage structure (`FramePose`, `FramePoseSequence`)
- Coordinate-frame transforms for stitching sensor observations

**Gaps:**

| Gap | Priority | Notes |
|-----|----------|-------|
| LiDAR odometry / scan matching (ICP, NDT, or LOAM-style) | High | The most common front-end for outdoor SLAM. Kiss-ICP integration is noted in `TODO.md` but not yet implemented. |
| IMU pre-integration | High | Needed for high-rate relative-pose prediction between LiDAR/camera frames, and for IMU-LiDAR tight coupling. |
| Visual odometry (feature tracking / direct methods) | High | No optical flow, ORB/SIFT keypoint extraction, essential-matrix estimation, or PnP solver. |
| Wheel odometry / vehicle kinematic model | Medium | No differential-drive or Ackermann model for dead-reckoning between frames. |
| GPS-to-local-frame converter (ECEF ↔ ENU / UTM) | Medium | `gps/nmea.py` returns raw latitude/longitude/altitude but provides no conversion to a local Cartesian frame needed for initialisation or global constraints. |

---

### 4. Loop Closure Detection

**Present:**
- Nothing directly applicable.

**Gaps:**

| Gap | Priority | Notes |
|-----|----------|-------|
| Place recognition / appearance-based descriptor (DBoW, NetVLAD, etc.) | High | Required to detect revisited locations and close loops. |
| Geometry-based loop closure verification (ICP/NDT re-alignment) | High | After a candidate loop is found, a geometric check validates it before adding a constraint to the graph. |
| LiDAR descriptor matching (Scan Context, M2DP, etc.) | Medium | LiDAR-based place recognition can complement or replace visual bag-of-words approaches. |

---

### 5. State Estimation & Sensor Fusion

**Present:**
- Individual sensor parsers providing raw measurements.

**Gaps:**

| Gap | Priority | Notes |
|-----|----------|-------|
| Extended / Unscented Kalman Filter (EKF/UKF) for pose estimation | High | Standard approach for fusing IMU, GPS, and odometry into a smooth state estimate. |
| IMU-LiDAR tightly/loosely coupled fusion | High | Needed to compensate for LiDAR motion distortion and produce continuous odometry at IMU rate. |
| GPS / GNSS absolute-position fusion | Medium | Ties the local map to a global coordinate frame; useful for multi-session SLAM. |
| Radar-odometry or radar-SLAM integration | Low | Radar can complement LiDAR in adverse weather; no fusion mechanism exists. |

---

### 6. Back-End Pose Graph Optimisation

**Present:**
- `FramePoseSequence` stores poses but applies no optimisation.

**Gaps:**

| Gap | Priority | Notes |
|-----|----------|-------|
| Pose graph construction (nodes = poses, edges = relative constraints) | High | No graph data structure or API for adding odometry and loop-closure edges. |
| Non-linear graph optimisation (g2o, GTSAM, Ceres, or iSAM2) | High | Essential for global consistency after loop closures are detected. |
| Marginalisation / sliding-window optimisation for online SLAM | Medium | Full batch optimisation becomes intractable on long trajectories; a sliding-window or fixed-lag smoother is needed. |
| Uncertainty / covariance propagation in `FramePose` | Medium | `FramePose` stores only the mean pose; no covariance matrix is tracked. |

---

### 7. Map Representation

**Present:**
- Nothing — the library is calibration/parsing focused and produces no persistent map.

**Gaps:**

| Gap | Priority | Notes |
|-----|----------|-------|
| Accumulated point-cloud / surfel map | High | A basic coloured point-cloud map assembled from successive LiDAR scans. |
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
| Multi-sensor data viewer (point clouds, images, trajectory) | High | A viewer that can synchronise and display data from all sensors is noted in `TODO.md`. |
| Real-time trajectory and map visualisation (RViz, Open3D, rerun.io) | High | Essential for debugging odometry and loop closures. |
| LiDAR–camera overlay visualisation | Medium | `lidar_camera.py` projects points but there is no rendering or display utility. |

---

## Summary Checklist

The following is a consolidated list of all identified gaps, ordered roughly by the impact they have on completing a working SLAM pipeline:

### Blocking / High Priority
- [ ] Fisheye/omnidirectional camera model (Kannala–Brandt)
- [ ] Multi-sensor time synchronisation / interpolation utilities
- [ ] ROS launch/parameter files for cameras, GPS, IMU, and radar
- [ ] GPS/RTK driver integration and ECEF/ENU/UTM conversion
- [ ] IMU pre-integration
- [ ] LiDAR odometry / scan matching (ICP, NDT; Kiss-ICP integration)
- [ ] Visual odometry (feature tracking, essential matrix, PnP)
- [ ] Place recognition / loop closure detection
- [ ] EKF/UKF state estimator for IMU + odometry fusion
- [ ] IMU-LiDAR tightly coupled fusion (motion-distortion correction)
- [ ] Pose graph data structure and optimisation back-end (g2o / GTSAM / Ceres)
- [ ] Accumulated point-cloud or surfel map output
- [ ] Multi-sensor synchronised visualisation platform

### Medium Priority
- [ ] Temporal extrinsic (clock-offset) calibration field in `SensorCollection`
- [ ] Camera-LiDAR target-based extrinsic calibration workflow
- [ ] IMU noise model / Allan-variance parameters in `SensorCollection`
- [ ] Rolling-shutter camera model and correction
- [ ] IMU and radar ROS driver examples
- [ ] Rosbag / MCAP recording and playback utilities
- [ ] Data capture and intrinsic calculation guide
- [ ] Wheel odometry / vehicle kinematic model
- [ ] LiDAR descriptor-based place recognition (Scan Context, M2DP)
- [ ] GPS absolute-position fusion into local map
- [ ] Sliding-window / fixed-lag smoother for online SLAM
- [ ] Covariance tracking in `FramePose`
- [ ] Occupancy grid (2-D / 3-D)
- [ ] Voxel map / TSDF volumetric representation
- [ ] Map serialisation (PCD, PLY)
- [ ] LiDAR–camera overlay display utility

### Low Priority
- [ ] Radar-odometry / radar-SLAM integration
- [ ] Map management (submap division, keyframe selection)
- [ ] Marginalisation for very long trajectories
