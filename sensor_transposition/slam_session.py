"""
slam_session.py

Thin orchestration layer for offline SLAM pipelines.

:class:`SLAMSession` wires the core ``sensor_transposition`` modules together
into a single object with sensible defaults, so users can get a working
pipeline running with minimal boilerplate.  It is deliberately *thin* — every
internal component is accessible as a public property so advanced users can
customise or replace it.

Typical use-case
----------------
::

    import numpy as np
    from sensor_transposition.rosbag import BagReader
    from sensor_transposition.slam_session import SLAMSession

    with BagReader("session.sbag") as bag:
        session = SLAMSession()
        session.run(bag, lidar_topic="/lidar/points")

    session.point_cloud_map.voxel_downsample(voxel_size=0.10)
    session.point_cloud_map.save_pcd("map.pcd")
    session.trajectory.to_csv("trajectory.csv")

Custom callbacks
----------------
Register a per-topic callback to inject custom processing at any stage::

    session = SLAMSession()

    @session.on_topic("/imu/data")
    def handle_imu(msg):
        # feed into your own EKF, log to a file, etc.
        pass

    session.run(bag)

Architecture
------------
On each LiDAR message ``SLAMSession`` performs the following steps:

1. Run ICP scan matching against the previous scan (or the local submap when
   ``use_local_map=True``).
2. Accumulate the relative transform to maintain a running ego pose.
3. Add the new keyframe to the :class:`~sensor_transposition.pose_graph.PoseGraph`.
4. Query the :class:`~sensor_transposition.loop_closure.ScanContextDatabase`
   for loop-closure candidates.
5. If a candidate passes the distance threshold, run ICP verification and add a
   loop-closure edge to the pose graph.
6. Append the current pose to the :class:`~sensor_transposition.frame_pose.FramePoseSequence`.
7. Accumulate the scan into the :class:`~sensor_transposition.point_cloud_map.PointCloudMap`.

After :meth:`run` returns, call :meth:`optimize` to run Gauss-Newton pose-graph
optimisation and rebuild the map with the corrected poses.

Local-submap odometry
---------------------
Pass ``use_local_map=True`` to :meth:`run` to match each incoming scan against
a downsampled accumulation of the last *N* keyframe scans (a ``LocalMap``)
rather than only the immediately preceding scan.  This reduces frame-to-frame
odometry drift before it enters the pose graph::

    with BagReader("session.sbag") as bag:
        session = SLAMSession()
        session.run(bag, use_local_map=True, local_map_size=10,
                    local_map_voxel_size=0.2)

Tightly-coupled IMU integration
--------------------------------
Pass ``imu_topic`` to :meth:`run` to accumulate IMU measurements between
consecutive LiDAR keyframes and add an :class:`~sensor_transposition.pose_graph.ImuFactor`
edge to the pose graph for each inter-keyframe window.  Each factor encodes
the pre-integrated (ΔR, Δv, Δp) increment as an additional 6-DOF relative-pose
constraint alongside the ICP odometry edge, providing tighter coupling between
IMU and LiDAR in the pose-graph back-end::

    with BagReader("session.sbag") as bag:
        session = SLAMSession()
        session.run(
            bag,
            lidar_topic="/lidar/points",
            imu_topic="/imu/data",
        )

    session.optimize()

IMU messages are expected to carry ``"accel"`` (``[ax, ay, az]`` in m/s²) and
``"gyro"`` (``[ωx, ωy, ωz]`` in rad/s) keys in their payload dict.  Use the
*imu_accel_key* and *imu_gyro_key* arguments to :meth:`run` if different key
names are needed.  A factor is only added when at least two IMU samples arrive
between consecutive LiDAR keyframes.

Localization against a pre-built map
-------------------------------------
Call :meth:`~SLAMSession.load_map` before :meth:`~SLAMSession.run` to switch
the session into *localization-only* mode.  In this mode each incoming scan is
matched against the fixed pre-built map via ICP and the resulting pose is
appended to :attr:`~SLAMSession.trajectory`, but the map is never modified and
no pose-graph nodes or loop-closure edges are added::

    session = SLAMSession()
    session.load_map("map.pcd")          # activates localization-only mode

    with BagReader("live.sbag") as bag:
        session.run(bag, lidar_topic="/lidar/points")

    # Trajectory contains ego poses in the pre-built map frame.
    session.trajectory.to_csv("localization_trajectory.csv")

The convenience subclass :class:`LocalizationSession` wraps this workflow::

    with BagReader("live.sbag") as bag:
        session = LocalizationSession("map.pcd")
        session.run(bag)

    session.trajectory.to_csv("localization_trajectory.csv")
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import numpy as np

from sensor_transposition.frame_pose import FramePose, FramePoseSequence
from sensor_transposition.imu.preintegration import ImuPreintegrator
from sensor_transposition.lidar.scan_matching import icp_align
from sensor_transposition.loop_closure import ScanContextDatabase, compute_scan_context
from sensor_transposition.point_cloud_map import PointCloudMap
from sensor_transposition.pose_graph import PoseGraph, optimize_pose_graph
from sensor_transposition.rosbag import BagReader


# ---------------------------------------------------------------------------
# LocalMap
# ---------------------------------------------------------------------------


class LocalMap:
    """Sliding-window local submap for scan-to-submap ICP odometry.

    Maintains a downsampled accumulation of the last *window_size* keyframe
    scans in the **sensor/ego frame** (centred on the most recent keyframe).
    Matching each new scan against this richer reference rather than only the
    immediately preceding scan provides more constraints simultaneously and
    significantly reduces frame-to-frame odometry drift.

    Args:
        window_size: Number of most-recent keyframes to retain (default
            ``10``).  Must be at least 1.
        voxel_size: Voxel-grid edge length (metres) used to downsample each
            keyframe before adding it to the submap (default ``0.1``).
            Set to ``0.0`` or ``None`` to skip downsampling.

    Example::

        local_map = LocalMap(window_size=10, voxel_size=0.2)

        # Add each keyframe scan (in sensor frame) as it arrives.
        local_map.add_keyframe(scan)

        # Retrieve the current submap for ICP.
        submap = local_map.get_submap()          # (M, 3) float array
    """

    def __init__(
        self,
        window_size: int = 10,
        voxel_size: float = 0.1,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}.")
        self._window_size = window_size
        self._voxel_size = float(voxel_size) if voxel_size else 0.0
        self._frames: List[np.ndarray] = []     # raw (or downsampled) scans

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_keyframe(self, scan: np.ndarray) -> None:
        """Add a new keyframe scan to the local map.

        The scan is optionally voxel-downsampled before storage.  When the
        window is full, the oldest keyframe is evicted.

        Args:
            scan: ``(N, 3)`` float array of LiDAR points in the sensor frame.
        """
        pts = np.asarray(scan, dtype=float)
        if self._voxel_size > 0 and pts.shape[0] > 0:
            pts = _voxel_downsample_simple(pts, self._voxel_size)
        self._frames.append(pts)
        if len(self._frames) > self._window_size:
            self._frames.pop(0)

    def get_submap(self) -> np.ndarray:
        """Return all keyframe points concatenated as an ``(M, 3)`` array.

        Returns an empty ``(0, 3)`` array if no keyframes have been added yet.
        """
        if not self._frames:
            return np.empty((0, 3), dtype=float)
        return np.concatenate(self._frames, axis=0)

    def __len__(self) -> int:
        """Return the number of keyframes currently in the window."""
        return len(self._frames)

    def clear(self) -> None:
        """Remove all keyframes from the local map."""
        self._frames.clear()

# Stride used to encode (ix, iy, iz) voxel indices into a single int64 key.
# 100_003 is a prime chosen large enough to avoid collisions for coordinate
# ranges typical in autonomous-driving and robotics (< ±500 m at any voxel
# size ≥ 0.01 m).
_VOXEL_STRIDE = np.array([1, 100_003, 100_003**2], dtype=np.int64)


def _voxel_downsample_simple(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel-grid downsample *points* to one centroid per cell (pure NumPy)."""
    indices = np.floor(points / voxel_size).astype(np.int64)
    keys = indices @ _VOXEL_STRIDE
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    centroids = np.zeros((len(unique_keys), 3), dtype=float)
    counts = np.zeros(len(unique_keys), dtype=int)
    np.add.at(centroids, inverse, points)
    np.add.at(counts, inverse, 1)
    return centroids / counts[:, None]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a unit quaternion ``[w, x, y, z]``.

    Uses Shepperd's method, which is numerically stable for all rotations.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


class SLAMSession:
    """Orchestrates an offline SLAM pipeline from bag replay to map output.

    Args:
        icp_max_iterations: Maximum ICP iterations per scan pair.
        icp_max_distance: Maximum correspondence distance for ICP (metres).
            ``None`` disables the filter.
        sc_num_rings: Scan Context ring count.
        sc_num_sectors: Scan Context sector count.
        sc_max_range: Scan Context maximum range (metres).
        loop_closure_threshold: Scan Context distance threshold below which a
            candidate is considered a loop closure (lower = stricter).
        loop_closure_min_gap: Minimum keyframe separation (in frames) required
            before a loop closure is accepted.  Prevents accepting the previous
            keyframe as a "loop".
        pg_max_iterations: Maximum Gauss-Newton iterations for pose-graph
            optimisation.

    Example::

        from sensor_transposition.rosbag import BagReader
        from sensor_transposition.slam_session import SLAMSession

        with BagReader("session.sbag") as bag:
            session = SLAMSession()
            session.run(bag)
        session.optimize()
        session.point_cloud_map.save_pcd("map.pcd")
    """

    def __init__(
        self,
        *,
        icp_max_iterations: int = 50,
        icp_max_distance: Optional[float] = None,
        sc_num_rings: int = 20,
        sc_num_sectors: int = 60,
        sc_max_range: float = 80.0,
        loop_closure_threshold: float = 0.15,
        loop_closure_min_gap: int = 5,
        pg_max_iterations: int = 50,
    ) -> None:
        self._icp_max_iterations = icp_max_iterations
        self._icp_max_distance = icp_max_distance
        self._loop_closure_threshold = loop_closure_threshold
        self._loop_closure_min_gap = loop_closure_min_gap
        self._pg_max_iterations = pg_max_iterations

        # Core SLAM components — accessible as public properties.
        self._loop_db = ScanContextDatabase(
            num_rings=sc_num_rings,
            num_sectors=sc_num_sectors,
            max_range=sc_max_range,
        )
        self._pose_graph = PoseGraph()
        self._trajectory = FramePoseSequence()
        self._point_cloud_map = PointCloudMap()

        # Per-topic user callbacks registered via on_topic().
        self._callbacks: Dict[str, List[Callable]] = {}

        # Internal state.
        self._scans: List[np.ndarray] = []          # raw scans in sensor frame
        self._opt_result = None                     # last optimize_pose_graph result
        self._localization_only: bool = False       # set by load_map()

    # ------------------------------------------------------------------
    # Properties — expose internal components for advanced use
    # ------------------------------------------------------------------

    @property
    def loop_db(self) -> ScanContextDatabase:
        """The internal :class:`~sensor_transposition.loop_closure.ScanContextDatabase`."""
        return self._loop_db

    @property
    def pose_graph(self) -> PoseGraph:
        """The internal :class:`~sensor_transposition.pose_graph.PoseGraph`."""
        return self._pose_graph

    @property
    def trajectory(self) -> FramePoseSequence:
        """Trajectory as a :class:`~sensor_transposition.frame_pose.FramePoseSequence`."""
        return self._trajectory

    @property
    def point_cloud_map(self) -> PointCloudMap:
        """The accumulated :class:`~sensor_transposition.point_cloud_map.PointCloudMap`."""
        return self._point_cloud_map

    @property
    def localization_only(self) -> bool:
        """``True`` after :meth:`load_map` has been called; ``False`` by default."""
        return self._localization_only

    # ------------------------------------------------------------------
    # Map loading (localization-only mode)
    # ------------------------------------------------------------------

    def load_map(self, path: str) -> None:
        """Load a pre-built point-cloud map and switch to localization-only mode.

        After calling this method, :meth:`run` will match each incoming LiDAR
        scan against the fixed loaded map using ICP and accumulate the resulting
        pose into :attr:`trajectory`.  The map is never modified; pose-graph
        construction and loop-closure detection are skipped.

        Args:
            path: Path to a ``.pcd`` or ``.ply`` map file previously saved by
                :meth:`~sensor_transposition.point_cloud_map.PointCloudMap.save_pcd`
                or
                :meth:`~sensor_transposition.point_cloud_map.PointCloudMap.save_ply`.

        Raises:
            ValueError: If the file extension is not ``.pcd`` or ``.ply``.
            RuntimeError: If the file cannot be parsed by
                :class:`~sensor_transposition.point_cloud_map.PointCloudMap`.

        Example::

            session = SLAMSession()
            session.load_map("map.pcd")        # activates localization mode
            with BagReader("live.sbag") as bag:
                session.run(bag)
            session.trajectory.to_csv("localization_trajectory.csv")
        """
        path_lower = path.lower()
        if path_lower.endswith(".pcd"):
            self._point_cloud_map = PointCloudMap.from_pcd(path)
        elif path_lower.endswith(".ply"):
            self._point_cloud_map = PointCloudMap.from_ply(path)
        else:
            raise ValueError(
                f"Unsupported map file format: {path!r}.  "
                "Expected a .pcd or .ply file."
            )
        self._localization_only = True

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_topic(self, topic: str) -> Callable:
        """Decorator that registers a callback for *topic*.

        The callback receives a single
        :class:`~sensor_transposition.rosbag.BagMessage` argument each time a
        matching message is replayed.  Multiple callbacks can be registered for
        the same topic; they are called in registration order.

        Args:
            topic: Topic name (e.g. ``"/imu/data"``).

        Returns:
            A decorator that registers the decorated function.

        Example::

            session = SLAMSession()

            @session.on_topic("/imu/data")
            def handle_imu(msg):
                print(msg.timestamp, msg.data["accel"])
        """
        def decorator(fn: Callable) -> Callable:
            self._callbacks.setdefault(topic, []).append(fn)
            return fn
        return decorator

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def run(
        self,
        bag: BagReader,
        lidar_topic: str = "/lidar/points",
        xyz_key: str = "xyz",
        use_local_map: bool = False,
        local_map_size: int = 10,
        local_map_voxel_size: float = 0.1,
        imu_topic: Optional[str] = None,
        imu_accel_key: str = "accel",
        imu_gyro_key: str = "gyro",
        imu_information: float = 10.0,
    ) -> None:
        """Replay *bag* and process every LiDAR frame.

        All non-LiDAR topics are forwarded to any registered callbacks.  The
        LiDAR topic is also forwarded to registered callbacks *after* SLAM
        processing.

        Args:
            bag: An open :class:`~sensor_transposition.rosbag.BagReader`.
            lidar_topic: Topic name that carries LiDAR point-cloud messages.
                The payload dict must contain an ``xyz_key`` entry with a 2-D
                array of shape ``(N, 3)`` (list-of-lists or numpy array).
            xyz_key: Key inside the LiDAR message payload that holds the
                ``(N, 3)`` point array.  Defaults to ``"xyz"``.
            use_local_map: When ``True``, each incoming scan is matched
                against a sliding-window local submap (a
                :class:`LocalMap` of the last *local_map_size* keyframes)
                rather than only the immediately preceding scan.  This
                reduces frame-to-frame odometry drift by providing more
                constraints to ICP simultaneously.  Default ``False``.
                Ignored when the session is in localization-only mode (see
                :meth:`load_map`).
            local_map_size: Number of most-recent keyframes retained in the
                local submap when ``use_local_map=True`` (default ``10``).
            local_map_voxel_size: Voxel edge length (metres) used to
                downsample each keyframe before adding it to the local map
                (default ``0.1``).
            imu_topic: Optional topic name that carries IMU messages.  When
                set, IMU readings arriving between consecutive LiDAR keyframes
                are buffered and integrated via
                :class:`~sensor_transposition.imu.preintegration.ImuPreintegrator`
                to produce an
                :class:`~sensor_transposition.pose_graph.ImuFactor` edge in
                the pose graph alongside the ICP odometry edge.  Each IMU
                message payload is expected to contain *imu_accel_key*
                (``[ax, ay, az]`` m/s²) and *imu_gyro_key* (``[ωx, ωy, ωz]``
                rad/s) arrays.  ``None`` disables IMU integration (default).
                Ignored when the session is in localization-only mode.
            imu_accel_key: Key in the IMU message payload dict for the
                accelerometer reading.  Defaults to ``"accel"``.
            imu_gyro_key: Key in the IMU message payload dict for the
                gyroscope reading.  Defaults to ``"gyro"``.
            imu_information: Scalar multiplier for the 6×6 identity
                information matrix assigned to each
                :class:`~sensor_transposition.pose_graph.ImuFactor`
                (default ``10.0``).  Lower than the ICP default of 100 to
                reflect the higher positional uncertainty of IMU-only
                dead-reckoning between keyframes.
        """
        if self._localization_only:
            self._run_localization(bag, lidar_topic, xyz_key)
            return
        pose_translation = np.zeros(3)
        pose_rotation = np.eye(3)  # 3×3 rotation matrix, updated each frame

        local_map: Optional[LocalMap] = (
            LocalMap(window_size=local_map_size, voxel_size=local_map_voxel_size)
            if use_local_map
            else None
        )

        # IMU buffering state: (timestamp, accel, gyro) triples accumulated
        # between consecutive LiDAR keyframes.
        _imu_buf: List[tuple] = []

        for msg in bag.read_messages():
            # Forward every message to user callbacks first.
            for cb in self._callbacks.get(msg.topic, []):
                cb(msg)

            # Buffer IMU readings for tight IMU–LiDAR coupling.
            if imu_topic and msg.topic == imu_topic:
                accel = np.asarray(msg.data[imu_accel_key], dtype=float)
                gyro = np.asarray(msg.data[imu_gyro_key], dtype=float)
                _imu_buf.append((msg.timestamp, accel, gyro))

            if msg.topic != lidar_topic:
                continue

            scan = np.asarray(msg.data[xyz_key], dtype=float)
            frame_id = len(self._scans)

            # Compute Scan Context descriptor once; reuse for both loop-closure
            # check and database insertion.
            desc = self._loop_db.compute_descriptor(scan)

            if frame_id == 0:
                # First keyframe: identity pose.
                self._pose_graph.add_node(
                    frame_id,
                    translation=pose_translation.copy(),
                    quaternion=_rotation_to_quaternion(pose_rotation),
                )
                if local_map is not None:
                    local_map.add_keyframe(scan)
            else:
                # Determine ICP reference: local submap or previous scan.
                if local_map is not None and len(local_map) > 0:
                    reference = local_map.get_submap()
                else:
                    reference = self._scans[-1]

                # ICP scan matching.
                kwargs: Dict = {"max_iterations": self._icp_max_iterations}
                if self._icp_max_distance is not None:
                    kwargs["max_distance"] = self._icp_max_distance
                result = icp_align(scan, reference, **kwargs)

                # Accumulate rotation and translation from the ICP transform.
                R_inc = result.transform[:3, :3]
                t_inc = result.transform[:3, 3]
                pose_translation = R_inc @ pose_translation + t_inc
                pose_rotation = R_inc @ pose_rotation

                self._pose_graph.add_node(
                    frame_id,
                    translation=pose_translation.copy(),
                    quaternion=_rotation_to_quaternion(pose_rotation),
                )
                self._pose_graph.add_edge(
                    from_id=frame_id - 1,
                    to_id=frame_id,
                    transform=result.transform,
                    information=np.eye(6) * (100.0 if result.converged else 1.0),
                )

                # Add IMU factor if enough samples were buffered.
                if imu_topic and len(_imu_buf) >= 2:
                    ts_arr = np.array([t for t, _, _ in _imu_buf])
                    accel_arr = np.vstack([a for _, a, _ in _imu_buf])
                    gyro_arr = np.vstack([g for _, _, g in _imu_buf])
                    preint = ImuPreintegrator().integrate(ts_arr, accel_arr, gyro_arr)
                    self._pose_graph.add_imu_factor(
                        from_id=frame_id - 1,
                        to_id=frame_id,
                        preint_result=preint,
                        information=np.eye(6) * imu_information,
                    )

                # Carry the last IMU sample forward as the anchor of the next
                # window.  Its original timestamp is intentionally kept so that
                # ImuPreintegrator computes the correct Δt to the first new
                # sample in the following gap, ensuring seamless continuity of
                # the integration chain across keyframe boundaries.
                _imu_buf = [_imu_buf[-1]] if _imu_buf else []

                # Update local map with the new scan.
                if local_map is not None:
                    local_map.add_keyframe(scan)

                # Loop-closure check — reuse the already-computed descriptor.
                candidates = self._loop_db.query(desc, top_k=1)
                for cand in candidates:
                    if (
                        cand.distance < self._loop_closure_threshold
                        and frame_id - cand.match_frame_id > self._loop_closure_min_gap
                    ):
                        lc_result = icp_align(
                            scan,
                            self._scans[cand.match_frame_id],
                            max_iterations=self._icp_max_iterations,
                        )
                        if lc_result.converged:
                            self._pose_graph.add_edge(
                                from_id=cand.match_frame_id,
                                to_id=frame_id,
                                transform=lc_result.transform,
                                information=np.eye(6) * 50.0,
                            )

            # Add descriptor to loop-closure database.
            self._loop_db.add(desc, frame_id=frame_id)

            # Record pose and scan.
            self._trajectory.add_pose(
                FramePose(
                    timestamp=msg.timestamp,
                    translation=pose_translation.copy(),
                    rotation=_rotation_to_quaternion(pose_rotation),
                )
            )
            T = np.eye(4)
            T[:3, 3] = pose_translation
            self._point_cloud_map.add_scan(scan, T)
            self._scans.append(scan)

    def optimize(self, *, max_iterations: Optional[int] = None) -> None:
        """Run Gauss-Newton pose-graph optimisation and rebuild the map.

        After optimisation the :attr:`point_cloud_map` is rebuilt from scratch
        using the corrected poses and the stored raw scans.

        Args:
            max_iterations: Override the default maximum Gauss-Newton
                iterations set at construction.
        """
        if not self._pose_graph.nodes:
            return

        n_iters = max_iterations if max_iterations is not None else self._pg_max_iterations
        self._opt_result = optimize_pose_graph(
            self._pose_graph, max_iterations=n_iters
        )

        # Rebuild map with corrected poses.
        self._point_cloud_map = PointCloudMap()
        for i, scan in enumerate(self._scans):
            node_pose = self._opt_result.optimized_poses.get(i, {"translation": np.zeros(3)})
            world_t = np.asarray(node_pose["translation"])
            T = np.eye(4)
            T[:3, 3] = world_t
            self._point_cloud_map.add_scan(scan, T)

    # ------------------------------------------------------------------
    # Localization-only run loop
    # ------------------------------------------------------------------

    def _run_localization(
        self,
        bag: BagReader,
        lidar_topic: str,
        xyz_key: str,
    ) -> None:
        """Replay *bag* in localization-only mode.

        Each LiDAR scan is matched against the fixed pre-built map stored in
        :attr:`point_cloud_map` via ICP.  The resulting ego pose (in the
        pre-built map frame) is appended to :attr:`trajectory`.  The map is
        never modified; pose-graph construction, loop-closure detection, and
        map accumulation are all skipped.

        Args:
            bag: An open :class:`~sensor_transposition.rosbag.BagReader`.
            lidar_topic: LiDAR topic name (same semantics as in :meth:`run`).
            xyz_key: Key inside each LiDAR message payload that holds the
                ``(N, 3)`` point array.
        """
        map_cloud = self._point_cloud_map.get_points()
        if map_cloud.shape[0] == 0:
            raise RuntimeError(
                "The loaded map is empty.  Call load_map() with a "
                "non-empty .pcd or .ply file before running in "
                "localization-only mode."
            )

        for msg in bag.read_messages():
            # Forward every message to user callbacks first.
            for cb in self._callbacks.get(msg.topic, []):
                cb(msg)

            if msg.topic != lidar_topic:
                continue

            scan = np.asarray(msg.data[xyz_key], dtype=float)

            # ICP against the fixed pre-built map.  The returned transform
            # maps the sensor frame directly into the world (map) frame, so
            # the result is the absolute pose — no incremental composition
            # is needed.
            kwargs: Dict = {"max_iterations": self._icp_max_iterations}
            if self._icp_max_distance is not None:
                kwargs["max_distance"] = self._icp_max_distance
            result = icp_align(scan, map_cloud, **kwargs)

            pose_rotation = result.transform[:3, :3]
            pose_translation = result.transform[:3, 3]

            # Record pose only — map and pose graph are not modified.
            # Scans are not stored in localization mode because the raw
            # scan data is not needed for pose-graph rebuilding.
            self._trajectory.add_pose(
                FramePose(
                    timestamp=msg.timestamp,
                    translation=pose_translation.copy(),
                    rotation=_rotation_to_quaternion(pose_rotation),
                )
            )


# ---------------------------------------------------------------------------
# LocalizationSession
# ---------------------------------------------------------------------------


class LocalizationSession(SLAMSession):
    """Convenience subclass for localization against a pre-built map.

    Loads a PCD or PLY map file at construction and immediately activates
    *localization-only* mode.  Calling :meth:`run` then matches each incoming
    LiDAR scan against the fixed map via ICP and accumulates the ego poses in
    :attr:`~SLAMSession.trajectory` — the map is never modified.

    This is equivalent to::

        session = SLAMSession(...)
        session.load_map(map_path)

    but wraps both steps in a single constructor call for convenience.

    Args:
        map_path: Path to a ``.pcd`` or ``.ply`` map file.
        **kwargs: Additional keyword arguments forwarded to
            :class:`SLAMSession` (e.g. ``icp_max_iterations``,
            ``icp_max_distance``).

    Example::

        from sensor_transposition.rosbag import BagReader
        from sensor_transposition.slam_session import LocalizationSession

        with BagReader("live.sbag") as bag:
            session = LocalizationSession("map.pcd", icp_max_iterations=30)
            session.run(bag, lidar_topic="/lidar/points")

        # Trajectory contains ego poses in the pre-built map frame.
        session.trajectory.to_csv("localization_trajectory.csv")

    Note:
        :meth:`optimize` is a no-op in localization-only mode because no
        pose graph is built.
    """

    def __init__(self, map_path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_map(map_path)
