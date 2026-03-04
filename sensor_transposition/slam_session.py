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

1. Run ICP scan matching against the previous scan.
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
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import numpy as np

from sensor_transposition.frame_pose import FramePose, FramePoseSequence
from sensor_transposition.lidar.scan_matching import icp_align
from sensor_transposition.loop_closure import ScanContextDatabase, compute_scan_context
from sensor_transposition.point_cloud_map import PointCloudMap
from sensor_transposition.pose_graph import PoseGraph, optimize_pose_graph
from sensor_transposition.rosbag import BagReader


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
        """
        pose_translation = np.zeros(3)
        pose_rotation = np.eye(3)  # 3×3 rotation matrix, updated each frame

        for msg in bag.read_messages():
            # Forward every message to user callbacks first.
            for cb in self._callbacks.get(msg.topic, []):
                cb(msg)

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
            else:
                # ICP scan matching against the previous scan.
                kwargs: Dict = {"max_iterations": self._icp_max_iterations}
                if self._icp_max_distance is not None:
                    kwargs["max_distance"] = self._icp_max_distance
                result = icp_align(scan, self._scans[-1], **kwargs)

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
