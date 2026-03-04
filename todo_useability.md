# Usability Review — `sensor_transposition`

This document records usability findings after reviewing all changes described in
`todo_slam.md`.  Each item describes a friction point in the user workflow, explains
why it is a problem, and suggests how it could be addressed.  Items are grouped by
theme and ordered roughly by impact.

---

## 1. Documentation & Discoverability

### 1.1 README is significantly out of date

**Problem:** The README still only covers the original eight modules
(SensorCollection, Camera Intrinsics, Transform, LiDAR-Camera Fusion, LiDAR Parsers,
LiDAR Scan Matching, GPS, IMU, Radar).  Roughly fifteen new modules added during the
SLAM sprint are not mentioned at all:

- `imu/ekf.py` – 15-state Error-State EKF
- `imu/preintegration.py` – IMU pre-integration
- `loop_closure.py` – Scan Context / M2DP place recognition
- `pose_graph.py` – pose graph construction and Gauss-Newton optimization
- `sliding_window.py` – fixed-lag online SLAM smoother
- `submap_manager.py` – keyframe selection and submap division
- `occupancy_grid.py` – 2-D/3-D probabilistic occupancy grid
- `voxel_map.py` – TSDF volumetric map
- `point_cloud_map.py` – accumulated colored point-cloud map (PCD / PLY I/O)
- `visualisation.py` – BEV rendering, trajectory overlay, Open3D / RViz export
- `rosbag.py` – custom binary bag recorder/player
- `wheel_odometry.py` – differential-drive and Ackermann dead-reckoning
- `gps/fusion.py` – GPS/ENU fusion into EKF and FramePoseSequence
- `lidar/motion_distortion.py` – IMU-based LiDAR scan deskewing
- `radar/radar_odometry.py` – Doppler-based ego-velocity and radar odometry
- `calibration.py` – camera–LiDAR target-based extrinsic calibration

**Suggested fix:** Add a new section to the README for each of the above modules,
following the existing pattern (short description + minimal code snippet + table of
key parameters).

---

### 1.2 No end-to-end quick-start example

**Problem:** A user wanting to run a complete SLAM pipeline (load sensor config →
record a bag → replay and synchronize → run ICP odometry → detect loop closures →
optimize pose graph → save map) must individually discover and wire together fifteen
modules.  The `examples/` directory contains only a sensor config YAML.

**Suggested fix:** Add `examples/slam_pipeline.py` showing a minimal but complete
pipeline from bag playback to map output.  Even a synthetic (random point cloud)
version would reduce the onboarding time dramatically.

---

### 1.3 `calibration.py` is absent from both the README and `__init__.py` exports

**Problem:** `calibration.py` provides camera–LiDAR extrinsic calibration
(`fit_plane`, `ransac_plane`, `calibrate_lidar_camera`) but is not mentioned in the
README and the top-level `__all__` list in `__init__.py` does not export its
functions — only the module itself.  Users must search the source tree to discover it.

**Suggested fix:** Add a README section and export `fit_plane`, `ransac_plane`,
`calibrate_lidar_camera` in `__init__.py` alongside the existing named exports.

---

### 1.4 `gps/fusion.py` (GpsFuser) absent from the README GPS section

**Problem:** The README GPS section covers NMEA parsing and coordinate frame
conversion but says nothing about `GpsFuser` or `hdop_to_noise`, which are the
classes that actually integrate GPS into the SLAM pipeline.

**Suggested fix:** Add a `GPS Fusion` sub-section to the README with a short example
showing `GpsFuser` being used with `NmeaParser` and `ImuEkf`.

---

## 2. API Design

### 2.1 Scan Context parameters must be specified twice (easy to mismatch) ✓

**Status:** Implemented — `ScanContextDatabase.compute_descriptor(cloud)` added.

**Problem:** `compute_scan_context()` is a standalone function that takes
`num_rings`, `num_sectors`, and `max_range` directly, while `ScanContextDatabase` is
constructed with the same three parameters.  A user who creates the database with
`num_rings=20` but accidentally calls `compute_scan_context(..., num_rings=25, ...)`
will silently produce incompatible descriptors and get nonsensical distances without
any error.

```python
# Current API — easy to get wrong:
db = ScanContextDatabase(num_rings=20, num_sectors=60, max_range=80.0)
for cloud in frames:
    desc = compute_scan_context(cloud, num_rings=20, num_sectors=60, max_range=80.0)  # must match!
    db.add(desc, frame_id=i)
```

**Suggested fix:** Give `ScanContextDatabase` a `compute_descriptor(cloud)` method
that uses the stored parameters, so the caller never has to repeat them.

---

### 2.2 `FramePoseSequence` has no convenience trajectory array accessors ✓

**Status:** Implemented — `positions`, `quaternions` properties and `to_csv()`/`from_csv()` added.

**Problem:** Users frequently need the trajectory as numpy arrays (e.g., for
visualisation, analysis, or exporting to CSV) but must write verbose boilerplate:

```python
# Current — verbose:
positions = np.array([p.translation for p in sequence])
quaternions = np.array([p.rotation for p in sequence])
```

There is no dedicated property and no CSV/numpy export method.

**Suggested fix:** Add `positions` and `quaternions` properties returning `(N, 3)`
and `(N, 4)` arrays respectively, and a `to_csv(path)` / `from_csv(path)` method pair
for easy analysis with pandas or numpy.

---

### 2.3 `BagWriter` does not accept numpy arrays directly ✓

**Status:** Implemented — `_to_json_serializable()` helper added; numpy arrays are auto-converted.

**Problem:** The bag payload is stored as JSON.  NumPy arrays must be manually
converted with `.tolist()` before calling `bag.write()`.  Forgetting this raises a
cryptic `TypeError` from the JSON serialiser deep in the call stack, not a clear
`ValueError` from `BagWriter` itself.

```python
# Fails with an unhelpful error:
bag.write("/lidar/points", t, {"xyz": lidar_xyz})       # numpy array → TypeError
# Must do:
bag.write("/lidar/points", t, {"xyz": lidar_xyz.tolist()})
```

**Suggested fix:** Add a lightweight serialisation helper inside `BagWriter.write()`
that recursively converts numpy scalars and arrays to Python lists/floats, mirroring
the behaviour of `json.JSONEncoder` subclasses that handle numpy.

---

### 2.4 `SensorCollection` silently accepts invalid sensor YAML ✓

**Status:** Implemented — `SensorCollection.validate()` added and called from `from_yaml()`.

**Problem:** `SensorCollection.from_yaml()` uses `data.get(key, default)` everywhere,
so malformed YAML (wrong key names, missing required fields, wrong value types) is
silently loaded with default values.  A camera sensor missing `intrinsics` will load
without error, then fail only later when `project_lidar_to_image` raises `ValueError`.

**Suggested fix:** Add a `validate()` method (and call it inside `from_yaml()`) that
checks type-specific required fields — e.g., that every `type: camera` sensor has a
complete `intrinsics` block with positive `fx`, `fy`, valid `width`/`height`, etc.

---

### 2.5 No progress reporting in long-running iterative algorithms ✓

**Status:** Implemented — optional `callback(iteration, cost)` parameter added to `icp_align`, `optimize_pose_graph`, and `SlidingWindowSmoother.optimize`.

**Problem:** ICP (`icp_align`), pose graph optimization (`optimize_pose_graph`), and
the sliding window smoother (`SlidingWindowSmoother.optimize`) give no feedback
during execution.  On large point clouds or dense graphs they can run for many seconds
with no indication of progress, making it impossible to distinguish a slow run from a
hang.

**Suggested fix:** Add an optional `callback` parameter (callable that receives the
current iteration number and cost) to each of the three functions.  A one-liner
`callback=lambda i, c: print(f"iter {i}: cost={c:.4f}")` would already be very
useful for debugging without adding a logging dependency.

---

### 2.6 `SensorSynchroniser` provides no overlap diagnostics ✓

**Status:** Implemented — `temporal_overlap()`, `stream_start_time()`, and `stream_end_time()` added.

**Problem:** A common mistake is to register streams whose time ranges don't overlap,
resulting in all interpolated values being boundary-clamped.  There is no method to
check whether registered streams have sufficient temporal overlap before calling
`synchronise()`.

**Suggested fix:** Add a `temporal_overlap()` method (or expose `start_time` /
`end_time` properties per stream) so users can verify overlap before resampling.

---

### 2.7 British vs. American spelling inconsistency in the public API ✓

**Status:** Implemented — `color_by_height` alias for `colour_by_height` added in `visualisation.py`; `synchronize` method alias and `SensorSynchronizer` class alias added in `sync.py`.

**Problem:** The library mixes British and American spelling across the public API:

- `SensorSynchroniser` / `synchronise()` (British)
- `colour_by_height()` (British spelling — inconsistent with the rest of the API)
- `color_lidar_from_image()` (American — note: this is in `lidar_camera.py` and
  `SensorCollection`)

This makes the API hard to remember and autocompletion unreliable.

**Suggested fix:** Pick one spelling convention (American is the broader Python-world
standard: `color`, `synchronize`, `Synchronizer`) and add aliases for the alternate
spelling to avoid a breaking change.

---

### 2.8 `SensorFrameVisualiser` has no factory for batch data ✓

**Status:** Implemented — `SensorFrameVisualiser.from_dict(data)` class method added.

**Problem:** `SensorFrameVisualiser` must be populated field-by-field with separate
`set_*` calls.  There is no constructor or class method that accepts a `BagMessage`
dictionary or a frame snapshot dict, making it tedious to replay bag files for
visualisation.

**Suggested fix:** Add a `SensorFrameVisualiser.from_dict(data)` class method that
accepts a dictionary with optional `"point_cloud"`, `"camera_image"`, `"trajectory"`,
and `"radar_scan"` keys — the same shape as a bag payload — and populates the
visualiser in one step.

---

## 3. Integration & Interoperability

### 3.1 Custom bag format (`.sbag`) has no bridge to standard ROS bag tools

**Status:** Implemented — `sbag_to_rosbag(sbag_path, output_path)` added to `rosbag.py`; writes MCAP format (JSON-encoded messages) compatible with `ros2 bag` and Foxglove Studio.  Requires `pip install "sensor_transposition[mcap]"`.  Documented in `rosbag.py` module docstring and the README Bag Recorder section.

---

### 3.2 No standard way to wire modules together for a complete pipeline

**Status:** Implemented — `SLAMSession` class added in `sensor_transposition/slam_session.py`.  It accepts a `BagReader`, exposes per-topic callbacks via `on_topic()`, manages the loop-closure database, pose graph, trajectory, and point-cloud map internally, and exposes them as properties for advanced use.  Exported from `sensor_transposition.__init__` and documented in the README.  `examples/slam_pipeline.py` also serves as a lower-level reference.

---

### 3.3 No optional-dependency declarations for external integrations

**Status:** Implemented — `open3d = ["open3d>=0.17"]`, `rerun = ["rerun-sdk>=0.14"]`, and `mcap = ["mcap>=1.0"]` optional extras added to `pyproject.toml`.  The README Installation section now documents all three extras with a summary table.

---

## 4. Error Handling & Robustness

### 4.1 Inconsistent exception types across the library ✓

**Status:** Implemented — `sensor_transposition/exceptions.py` added with
`SensorTranspositionError` (base), `SensorNotFoundError` (inherits `KeyError`),
`BagError` (inherits `RuntimeError`), and `CalibrationError` (inherits `ValueError`).
`SensorCollection.get_sensor()` now raises `SensorNotFoundError`;
`BagWriter.write()` and `BagReader.read_messages()` now raise `BagError` when
the writer/reader is closed.  All subclasses also inherit from the corresponding
standard exception so existing `except KeyError / RuntimeError / ValueError`
handlers continue to work.  All four exception classes are exported from the
top-level `sensor_transposition` package.

**Problem:** Different modules raise different exception types for similar situations:
- `SensorCollection.get_sensor()` raises `KeyError`
- `BagWriter.write()` raises `RuntimeError` for a closed writer and `ValueError` for
  bad arguments
- `ScanContextDatabase.query()` raises `RuntimeError` for an empty database
- `icp_align()` raises `ValueError` for bad inputs

There is no documented exception hierarchy, making `try/except` blocks fragile.

**Suggested fix:** Define a small set of library-specific exceptions in
`sensor_transposition/exceptions.py` (e.g., `SensorNotFoundError`,
`CalibrationError`, `BagError`) that inherit from standard Python exceptions, and
migrate the most user-facing error sites to use them.

---

### 4.2 Pose graph optimization has no maximum-iteration / timeout safeguard ✓

**Status:** Implemented — `max_iterations` is exposed as a public keyword argument
(default ``20``) to `optimize_pose_graph()`.  The returned `OptimizationResult`
includes a `success` field that is ``False`` when the solver did not converge
within *max_iterations* steps, allowing downstream code to detect and handle
non-converged results.

---

## 5. Minor / Low-effort Improvements

### 5.1 `FramePose.transform` is a property but not cached

**Status:** Implemented — `transform` is now a `functools.cached_property`; the
4×4 matrix is computed once and reused on subsequent accesses.

---

### 5.2 `lidar/motion_distortion.py` deskew timestamps require the user to know the sweep duration

**Status:** Implemented — the `ref_time` parameter docstring now explains the
choice between start / end / midpoint of the sweep and includes an explicit note
that all timestamps must share the same hardware clock.

---

### 5.3 `occupancy_grid.py` ROS `int8` export is not explained in the README

**Status:** Implemented — `OccupancyGrid.to_ros_int8()` method added (explicit
alias for `get_grid()`).  The README occupancy grid snippet now shows the
value-to-ROS-convention mapping and the `.ravel()` step for populating a
`nav_msgs/OccupancyGrid` message.

---

### 5.4 No `__repr__` on key dataclasses

**Status:** Implemented — compact `__repr__` added to `FramePose`
(`FramePose(t=…, xyz=[…])`), `IcpResult`
(`IcpResult(converged=…, mse=…, n_iter=…)`), and `OdometryResult`
(`OdometryResult(x=…, y=…, theta=…, dur=…s)`).

---

### 5.5 Bag file extension `.sbag` is not documented or registered

**Problem:** The `BagWriter` and `BagReader` accept any path string, but the
extension used in all examples is `.sbag`.  This extension is never defined or
explained in the README or in `rosbag.py`.

**Suggested fix:** Document in `rosbag.py` and the README that `.sbag` is the
recommended extension ("sensor bag") and what it stands for.

---

## Summary Checklist

### High Impact
- [ ] Update README to document all 15+ new modules added during the SLAM sprint
- [ ] Add an end-to-end `examples/slam_pipeline.py` quick-start script
- [ ] Export `calibration.py` functions in `__init__.py` and README
- [ ] Add `GpsFuser` documentation to the README GPS section
- [x] Give `ScanContextDatabase` a `compute_descriptor(cloud)` method to avoid
      duplicated parameters
- [x] Add `BagWriter` numpy-array auto-conversion to prevent cryptic `TypeError`
- [x] Add `FramePoseSequence.positions` / `.quaternions` array properties and `to_csv()` / `from_csv()` methods

### Medium Impact
- [x] Add `SensorCollection.validate()` and call it from `from_yaml()`
- [x] Add optional `callback` parameter to `icp_align`, `optimize_pose_graph`, and
      `SlidingWindowSmoother.optimize`
- [x] Add `SensorSynchroniser.temporal_overlap()` diagnostics
- [x] Standardise British/American spelling (add aliases, pick one canonical form)
- [x] Add `SensorFrameVisualiser.from_dict()` factory method
- [x] Add optional `open3d` and `rerun` extras to `pyproject.toml`
- [x] Document `.sbag` extension in README and `rosbag.py`

### Low Impact
- [x] Cache `FramePose.transform` with `functools.cached_property`
- [x] Clarify `deskew_scan` `ref_timestamp` convention in docstring
- [x] Add ROS `int8` export note to README occupancy grid section
- [x] Add compact `__repr__` to `FramePose`, `IcpResult`, and `OdometryResult`
- [x] Add `sbag_to_rosbag` conversion helper (or MCAP writer option)
- [x] Add `Pipeline` / `SLAMSession` orchestration class or reference example
- [x] Define a small library-specific exception hierarchy
- [x] Expose `max_iterations` in `optimize_pose_graph` public API
