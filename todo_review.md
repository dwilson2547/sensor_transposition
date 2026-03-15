# SLAM Functionality Review

This document is a re-analysis of the `sensor_transposition` project after the
work recorded in `todo_slam.md` and `todo_useability.md` has been completed.
All items in both predecessor documents are marked as implemented.  This review
looks at the **current state of the code** and identifies gaps that still exist
for a production-quality SLAM pipeline.

Items are grouped by theme and ordered by priority within each group:
**High** (blocks a real deployment), **Medium** (limits quality or generality),
**Low** (nice-to-have polish).

---

## 1. Scan Matching & Odometry Front-End

### 1.1 Only point-to-point ICP is implemented — no point-to-plane ICP or NDT

**Gap:** `lidar/scan_matching.py` implements the Kabsch SVD algorithm for
point-to-point ICP.  This variant treats every point as a featureless sphere;
it works but converges slowly on planar scenes (walls, floors, roads) and is
sensitive to density differences between source and target.

Point-to-plane ICP (align each source point to the *tangent plane* of its
nearest target point) is the standard variant used in every production LiDAR
SLAM system (Cartographer, LOAM, LIO-SAM) because it converges in roughly
half as many iterations and produces more accurate results on structured
indoor/outdoor scenes.

NDT (Normal Distributions Transform), also mentioned in `todo_slam.md` as an
alternative but never implemented, is even more robust to noise and outliers.

- [x] Add `point_cloud_normals(cloud, k=20)` utility to
  `lidar/scan_matching.py` (or a new `lidar/normals.py`) that returns per-point
  normals via PCA on k-nearest neighbors using SciPy's `cKDTree`.
- [x] Add `icp_align_point_to_plane(source, target, ...)` to
  `lidar/scan_matching.py` that uses the computed target normals to set up and
  solve the point-to-plane linear system via SVD at each iteration.
- [ ] (Optional / Medium) Add `ndt_align(source, target, voxel_size, ...)`
  implementing the Normal Distributions Transform scan matcher as an
  alternative to ICP.

---

### 1.2 Odometry is frame-to-frame only — no scan-to-submap registration

**Gap:** `SLAMSession.run()` and `examples/slam_pipeline.py` match each new
scan only against the *immediately preceding* scan.  This means small ICP
errors compound frame-by-frame, producing drift that is only corrected after a
full pose-graph optimisation.  In state-of-the-art systems (e.g. LOAM, HDL
Graph SLAM) the odometry front-end maintains a local *sliding submap* of the
last N scans and matches each new scan against the submap.  This leverages
more constraints simultaneously, significantly reducing odometry drift before
it enters the pose graph.

- [x] Add a `LocalMap` helper class (or extend `PointCloudMap`) that maintains
  a downsampled accumulation of the last N keyframe scans (configurable window)
  in the sensor/ego frame.
- [x] Modify `SLAMSession.run()` with a `use_local_map=True` option that
  matches each incoming scan against the `LocalMap` rather than only the
  previous scan.

---

### 1.3 No KISS-ICP integration

**Gap:** The original `TODO.md` listed "integration with kiss icp pub sub" as
a to-do item that was never addressed.  KISS-ICP is a widely-adopted,
minimal-parameter LiDAR odometry algorithm that uses an adaptive
correspondence-distance threshold and a voxel-based local map; it outperforms
vanilla ICP on most public benchmarks with essentially no tuning.

- [x] Add a `kiss_icp_odometry.py` module (or extend `lidar/scan_matching.py`)
  that wraps the key KISS-ICP ideas: adaptive threshold estimation, voxel-hashed
  local map, and point-to-point ICP with the adaptive threshold.  Pure NumPy/
  SciPy implementation so no additional dependency is required.
- [x] Add a `kiss_icp` optional extra to `pyproject.toml` that pulls in the
  upstream `kiss-icp` package for users who want the full reference
  implementation.
- [x] Document in the README under a new **KISS-ICP** sub-section.

---

## 2. Ground Plane Segmentation Module

### 2.1 Documentation exists but no Python module

**Gap:** `docs/ground_plane_identification.md` provides an excellent guide
describing three methods (height threshold, RANSAC plane fitting, and
normal-based filtering) with inline code snippets.  However, none of these
methods exist as importable, tested functions in the `sensor_transposition`
package.  Users must copy-paste code from the documentation, which is error-
prone and defeats the purpose of having a library.

- [x] Create `sensor_transposition/ground_plane.py` with the following public
  API:
  - `height_threshold_segment(cloud, threshold=0.3)` → `(ground_mask, non_ground_mask)`
  - `ransac_ground_plane(cloud, distance_threshold=0.2, max_iterations=1000, normal_threshold=0.9)` → `(ground_mask, plane_coefficients)`
  - `normal_based_segment(cloud, k=20, verticality_threshold=0.85)` → `(ground_mask, normals)`
- [x] Export the new functions from `sensor_transposition/__init__.py`.
- [x] Add unit tests in `tests/test_ground_plane.py`.
- [x] Add a **Ground Plane Segmentation** section to the README that references
  the existing guide in `docs/ground_plane_identification.md`.

---

## 3. Visual Odometry & Visual SLAM

### 3.1 No visual feature extraction — callers must supply pre-matched pairs

**Gap:** `visual_odometry.py` provides the geometric back-end (essential
matrix estimation, pose recovery, PnP solver) but the module docstring
explicitly tells users they must supply "matched feature keypoints" from "any
descriptor matcher (e.g. ORB, SIFT, or your own)".  There is no feature
detection or matching code in the library.

This means the visual odometry module cannot stand alone: a new user must
integrate an external library (OpenCV, kornia, etc.) before they can get any
output.  The gap is especially large because the rest of the SLAM pipeline
(ICP, EKF, pose graph) is fully self-contained in pure NumPy/SciPy.

- [x] Add a `feature_detection.py` module with a minimal, pure-NumPy/SciPy
  Harris corner detector (`detect_harris_corners`) and a simple descriptor
  (intensity patches or simplified BRIEF).
- [x] Add `match_features(desc1, desc2, ratio_threshold=0.75)` using brute-
  force L2 distance with Lowe's ratio test to filter outlier matches.
- [x] Update the `visual_odometry.py` docstring and README to show a complete
  example that goes from two grayscale images all the way to a relative pose.
- [x] (Optional / Medium) Add an `opencv` optional extra to `pyproject.toml`
  so users who want production-quality ORB/SIFT features can `pip install
  ".[opencv]"` and use `cv2` in their own code.

---

### 3.2 No visual loop closure

**Gap:** `loop_closure.py` provides two LiDAR-based descriptors (Scan Context
and M2DP) but no image-based descriptor for visual loop closure.  In camera-
heavy platforms (e.g. autonomous cars with surround cameras but no LiDAR) or
in systems where the LiDAR loop-closure database is sparse, a visual loop
closure fallback is needed.

- [x] Add a `compute_image_descriptor(image, grid=(4,4), bins=8)` function to
  `loop_closure.py` that computes a compact HOG-like bag-of-visual-words
  descriptor from a grayscale image — pure NumPy, no external dependencies.
- [x] Add an `ImageLoopClosureDatabase` class mirroring the
  `ScanContextDatabase` API (`add`, `query`, `compute_descriptor`) for image
  descriptors.
- [x] Document in `docs/loop_closure.md` and the README.

---

### 3.3 No stereo camera support

**Gap:** `visual_odometry.py` mentions "monocular (or stereo)" in its
docstring, but stereo-specific operations (rectification, disparity
computation, direct metric depth from stereo baseline) are absent.  Monocular
VO produces pose estimates only up to a scale factor; stereo VO recovers the
metric scale directly from the baseline.

- [x] Add `stereo_rectify(K1, D1, K2, D2, R, t, image_size)` to
  a new `stereo.py`.
- [x] Add `compute_disparity_sgbm(img_left, img_right, block_size=11,
  num_disparities=64)` (pure NumPy block-matching / SAD implementation).
- [x] Add `triangulate_stereo(pts_left, pts_right, K, baseline)` that converts
  stereo pixel matches to metric 3-D points.
- [x] Add a **Stereo Camera** section to the README.

---

## 4. Map Localization (Localization against a pre-built map)

### 4.1 No localization-only mode — SLAMSession always builds a new map

**Gap:** `SLAMSession` always runs in mapping mode: it starts with an empty
map and builds it incrementally from scratch.  Real autonomous systems
frequently need a *localization-only* mode where a previously built (and
possibly hand-annotated or survey-grade) map is loaded, and the system
localises the vehicle within it without modifying the map.

Neither `SLAMSession` nor any other module supports this workflow:

- Loading a saved PCD/PLY map as the fixed reference.
- Running ICP or NDT to match each incoming scan against the fixed map.
- Returning a stream of ego-poses without adding nodes to the pose graph.

- [x] Add a `SLAMSession.load_map(path)` method that loads a PCD/PLY file into
  `_point_cloud_map` and sets a `_localization_only` flag.
- [x] When `_localization_only=True`, `SLAMSession.run()` should skip pose
  graph construction and instead match each scan directly against the loaded
  map using `icp_align`, accumulating poses into `_trajectory` only.
- [x] Add a `LocalizationSession` convenience subclass (or factory function)
  that wraps this workflow with a cleaner API.
- [x] Document in the README under a new **Localization Against a Pre-Built
  Map** section.

---

## 5. GPS / GNSS Integration

### 5.1 No RTK GPS setup guide or NTRIP integration

**Gap:** The original `TODO.md` listed "rtk setup and instructions" as an
unchecked item.  The current GPS module handles NMEA 0183 log file parsing and
coordinate-frame conversion, but there is:

- No documentation of how to set up an RTK base station or subscribe to an
  NTRIP correction service.
- No RTCM message parser for processing correction streams.
- No `GpsFuser` integration with an RTK-grade position covariance (centimetre
  vs. metre level).

- [x] Add `docs/rtk_gps_setup.md` covering RTK base-station requirements,
  NTRIP client configuration (using common tools such as `str2str` or
  `rtklib`), and how to set the resulting centimetre-level sigma in
  `hdop_to_noise`.
- [x] Add an `rtcm.py` module (or a section of `gps/nmea.py`) that can parse
  RTCM 3.x Message Type 1005 (base-station antenna position) and MSM4/MSM7
  (raw observations), returning structured records.
- [x] Update the README GPS section with an RTK sub-section linking to the new
  guide.

---

### 5.2 No explicit GNSS outage / dead-reckoning handling

**Gap:** `GpsFuser.fuse_into_ekf` passes every GPS fix to the EKF without
checking whether the fix is stale or whether a long gap exists between
consecutive fixes.  During GNSS outages (tunnels, urban canyons) the EKF drifts
on IMU alone with no indication that GPS data is absent, and no automatic
switch to a wheel-odometry or LiDAR-odometry only mode.

- [x] Add a `max_fix_age_sec` parameter to `GpsFuser` (default `1.0`).  If the
  time since the last valid fix exceeds this threshold, `fuse_into_ekf` should
  skip the update and optionally call a user-supplied `on_outage` callback.
- [x] Add a `GpsFuser.fix_age(current_timestamp)` property so callers can check
  how stale the most recent fix is before deciding whether to trust it.
- [x] Document the recommended pattern (fall back to wheel odometry / LiDAR
  odometry) in `docs/gps_fusion.md`.

---

## 6. IMU Integration in the Pose Graph

### 6.1 IMU pre-integration is not wired as pose-graph edge factors

**Gap:** `imu/preintegration.py` computes compact (ΔR, Δv, Δp) increments
between keyframes, which is exactly the data required for an IMU binary factor
in a pose graph.  However, `pose_graph.py` only supports generic 6-DOF edges
(relative SE(3) transforms).  There is no `ImuFactor` edge type and no code
path that connects `ImuPreintegrator` output to `PoseGraph.add_edge`.

Without proper IMU factors, the pose graph back-end ignores IMU measurements
entirely during optimisation, giving up the tightly-coupled accuracy that
pre-integration was designed to provide.

- [x] Define an `ImuFactor` dataclass in `pose_graph.py` (or a new
  `imu_factor.py`) that stores a pre-integration result and the two keyframe
  IDs it connects.
- [x] Add `PoseGraph.add_imu_factor(from_id, to_id, preint_result)` that
  converts the pre-integrated (ΔR, Δv, Δp) into the 6-DOF error formulation
  used by the existing Gauss-Newton optimiser.
- [x] Update `SLAMSession.run()` to accumulate IMU messages (via
  `ImuPreintegrator`) between LiDAR keyframes and add `ImuFactor` edges to the
  pose graph.
- [x] Add unit tests for the `ImuFactor` to `tests/test_pose_graph.py`.

---

## 7. Multi-Session SLAM / Map Merging

### 7.1 No support for extending or merging existing maps

**Gap:** Each `SLAMSession.run()` call starts with an empty map.  In practice,
robots often need to:

1. Resume a previous session from a saved map (add new areas without re-mapping
   existing ones).
2. Merge two independently built maps (e.g. from two different robots or two
   separate survey runs) when a loop closure between them is detected.

Neither workflow is currently possible.

- [x] Add `SLAMSession.save(path)` and `SLAMSession.load(path)` methods that
  serialise and deserialise the pose graph, trajectory, Scan Context database,
  and point-cloud map together (e.g. as a ZIP archive containing PCD, YAML,
  and a binary pose-graph file).
- [x] Add a `merge_sessions(session_a, session_b, loop_edge)` utility function
  that merges two sessions given a known inter-session loop-closure edge.
- [x] Document in the README under a new **Multi-Session and Map Merging**
  section.

---

## 8. Dynamic Object Handling

### 8.1 No filtering of moving objects from the map

**Gap:** `PointCloudMap.add_scan()` accumulates all points from every scan,
including points on moving objects (pedestrians, other vehicles).  Over time
these produce "ghost" trails in the map that degrade localization accuracy.

No filtering mechanism currently exists — neither a simple velocity-based
filter using radar Doppler measurements nor a consistency-check between
overlapping scans.

- [x] Add a `filter_dynamic_points(cloud, velocity_map, ...)` utility to
  `point_cloud_map.py` that removes points whose Doppler velocity (from a
  co-registered radar scan) exceeds a threshold.
- [x] Add a `consistency_filter(cloud, reference_cloud, threshold_m=0.5)` that
  removes points in `cloud` that are not supported by any point within
  `threshold_m` in `reference_cloud` — a simple but effective way to remove
  transient objects when two scans of the same area are available.
- [x] Document both filters in `docs/point_cloud_map.md`.

---

## 9. Map Efficiency & Representation

### 9.1 All map representations use fixed-resolution grids — no adaptive resolution

**Gap:** `OccupancyGrid`, `TSDFVolume`, and `PointCloudMap` all allocate memory
for a uniform voxel grid over the entire specified bounding box, regardless of
whether a region has been observed.  For large outdoor environments (e.g. a
city block) the memory usage becomes prohibitive before any sensor data is
processed.

- [x] Add a `SparseOccupancyGrid` class (or extend `OccupancyGrid`) that stores
  only observed voxels in a Python `dict` keyed by voxel index, identical
  external API.
- [x] Add a `SparseTSDFVolume` (or extend `TSDFVolume`) with the same approach:
  hash map over occupied voxels, lazy allocation on first integration.
- [x] Benchmark memory usage of sparse vs. dense representations in the
  docstring.

---

## Summary Checklist

### High Priority
- [x] Add `point_cloud_normals()` utility and `icp_align_point_to_plane()` to
      `lidar/scan_matching.py`
- [x] Create `ground_plane.py` module with height-threshold, RANSAC, and
      normal-based segmentation functions (currently only in docs)
- [x] Add visual feature detection / matching to support a complete visual
      odometry pipeline without external dependencies
- [x] Add map-based localization mode to `SLAMSession` (load existing map,
      localize without modifying it)
- [x] Implement scan-to-local-submap odometry to reduce frame-to-frame drift in
      `SLAMSession`

### Medium Priority
- [x] Implement KISS-ICP adaptive-threshold odometry
      (pure NumPy/SciPy; reference original `TODO.md`)
- [x] Add `ImuFactor` edge type and wire `ImuPreintegrator` output into the
      pose graph for tightly-coupled IMU optimisation
- [x] Add visual loop closure support (`compute_image_descriptor`,
      `ImageLoopClosureDatabase`) to `loop_closure.py`
- [x] Add stereo camera utilities (rectification, disparity, triangulation)
- [x] Add multi-session SLAM: `SLAMSession.save()` / `load()` and
      `merge_sessions()` utility
- [x] Add RTK GPS setup guide (`docs/rtk_gps_setup.md`) and RTCM 3.x parser
- [x] Add GNSS outage handling to `GpsFuser` (`max_fix_age_sec`, `on_outage`
      callback, `fix_age` property)
- [ ] Add NDT (Normal Distributions Transform) scan matcher as an alternative
      to ICP

### Low Priority
- [x] Add dynamic-object filtering utilities to `point_cloud_map.py` (Doppler
      filter + consistency filter)
- [x] Add sparse (hash-map based) variants of `OccupancyGrid` and `TSDFVolume`
      for large-scale outdoor environments
- [ ] Add `on_outage` sensor-health monitoring hooks to `SLAMSession` for
      detecting IMU, LiDAR, and GPS dropouts
- [ ] Add semantic SLAM integration hooks (per-point label fields in
      `PointCloudMap`, label-aware loop closure)
