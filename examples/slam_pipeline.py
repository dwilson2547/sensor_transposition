"""
slam_pipeline.py

Minimal end-to-end SLAM pipeline using synthetic (random) data.

This script demonstrates how to wire together the sensor_transposition modules
to build a complete offline SLAM pipeline:

    1. Generate synthetic sensor data (random point clouds + IMU samples).
    2. Record data to a bag file with BagWriter.
    3. Replay the bag with BagReader and synchronise LiDAR + IMU streams.
    4. Run ICP odometry between consecutive scans.
    5. Detect loop closures with ScanContextDatabase.
    6. Build and optimise a pose graph.
    7. Accumulate the final map with PointCloudMap and save as PCD.

Run with:

    python examples/slam_pipeline.py
"""

import tempfile
from pathlib import Path

import numpy as np

from sensor_transposition.rosbag import BagWriter, BagReader
from sensor_transposition.lidar.scan_matching import icp_align
from sensor_transposition.loop_closure import compute_scan_context, ScanContextDatabase
from sensor_transposition.pose_graph import PoseGraph, optimize_pose_graph
from sensor_transposition.point_cloud_map import PointCloudMap
from sensor_transposition.frame_pose import FramePose, FramePoseSequence

# ---------------------------------------------------------------------------
# 1. Synthetic data generation
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
NUM_FRAMES = 10           # number of LiDAR keyframes
POINTS_PER_SCAN = 200     # points per synthetic scan
DT = 0.1                  # seconds between frames


def make_scan(centre: np.ndarray) -> np.ndarray:
    """Return a random point cloud centred around *centre*."""
    return centre + RNG.standard_normal((POINTS_PER_SCAN, 3)) * 2.0


def make_imu_sample(timestamp: float) -> dict:
    """Return a synthetic IMU record as a JSON-serialisable dict."""
    return {
        "accel": RNG.standard_normal(3).tolist(),
        "gyro": (RNG.standard_normal(3) * 0.01).tolist(),
    }


# ---------------------------------------------------------------------------
# 2. Record data to a bag file
# ---------------------------------------------------------------------------

bag_path = Path(tempfile.mkdtemp()) / "pipeline.sbag"
print(f"Recording bag to: {bag_path}")

# Ground-truth translations (straight line with small lateral noise).
gt_translations = [
    np.array([i * 1.0, RNG.uniform(-0.05, 0.05), 0.0])
    for i in range(NUM_FRAMES)
]

with BagWriter(bag_path) as bag:
    for i in range(NUM_FRAMES):
        t = float(i) * DT
        scan = make_scan(gt_translations[i])
        bag.write("/lidar/points", t, {"xyz": scan.tolist()})
        bag.write("/imu/data",     t, make_imu_sample(t))

print(f"Wrote {NUM_FRAMES} LiDAR frames and {NUM_FRAMES} IMU samples.")

# ---------------------------------------------------------------------------
# 3. Replay and synchronise
# ---------------------------------------------------------------------------

reader = BagReader(bag_path)
lidar_msgs = list(reader.read_messages(topics=["/lidar/points"]))
imu_msgs   = list(reader.read_messages(topics=["/imu/data"]))

lidar_scans = [np.array(m.data["xyz"]) for m in lidar_msgs]
print(f"Replayed {len(lidar_scans)} LiDAR scans.")

# ---------------------------------------------------------------------------
# 4. ICP odometry between consecutive frames
# ---------------------------------------------------------------------------

SC_NUM_RINGS   = 20
SC_NUM_SECTORS = 60
SC_MAX_RANGE   = 20.0

loop_db = ScanContextDatabase(num_rings=SC_NUM_RINGS,
                               num_sectors=SC_NUM_SECTORS,
                               max_range=SC_MAX_RANGE)
graph   = PoseGraph()
trajectory = FramePoseSequence()

# First keyframe: identity pose.
pose_translation = np.zeros(3)
pose_quaternion  = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
graph.add_node(0, translation=pose_translation, quaternion=pose_quaternion)
trajectory.add_pose(FramePose(timestamp=0.0,
                             translation=pose_translation.copy(),
                             rotation=pose_quaternion.copy()))

# Add Scan Context descriptor for frame 0.
descriptor = compute_scan_context(lidar_scans[0],
                                  num_rings=SC_NUM_RINGS,
                                  num_sectors=SC_NUM_SECTORS,
                                  max_range=SC_MAX_RANGE)
loop_db.add(descriptor, frame_id=0)

for i in range(1, NUM_FRAMES):
    result = icp_align(lidar_scans[i], lidar_scans[i - 1], max_iterations=30)

    # Accumulate odometry translation.
    R = result.transform[:3, :3]
    t = result.transform[:3, 3]
    pose_translation = (R @ pose_translation) + t
    # Keep quaternion as identity for this synthetic demo (no rotation).
    graph.add_node(i, translation=pose_translation.copy(),
                   quaternion=pose_quaternion.copy())
    graph.add_edge(
        from_id=i - 1,
        to_id=i,
        transform=result.transform,
        information=np.eye(6) * (100.0 if result.converged else 1.0),
    )
    trajectory.add_pose(FramePose(timestamp=float(i) * DT,
                                  translation=pose_translation.copy(),
                                  rotation=pose_quaternion.copy()))

    # Scan Context loop-closure check.
    descriptor = compute_scan_context(lidar_scans[i],
                                      num_rings=SC_NUM_RINGS,
                                      num_sectors=SC_NUM_SECTORS,
                                      max_range=SC_MAX_RANGE)
    candidates = loop_db.query(descriptor, top_k=1)
    for cand in candidates:
        if cand.distance < 0.15 and i - cand.match_frame_id > 3:
            print(f"Loop closure detected: frame {i} ↔ frame "
                  f"{cand.match_frame_id} (d={cand.distance:.3f})")
            lc_result = icp_align(lidar_scans[i],
                                  lidar_scans[cand.match_frame_id],
                                  max_iterations=30)
            if lc_result.converged:
                graph.add_edge(
                    from_id=cand.match_frame_id,
                    to_id=i,
                    transform=lc_result.transform,
                    information=np.eye(6) * 50.0,
                )
    loop_db.add(descriptor, frame_id=i)

print(f"Built pose graph with {len(graph.nodes)} nodes and "
      f"{len(graph.edges)} edges.")

# ---------------------------------------------------------------------------
# 5. Optimise pose graph
# ---------------------------------------------------------------------------

opt_result = optimize_pose_graph(graph, max_iterations=30)
print(f"Pose graph optimisation converged={opt_result.success}, "
      f"final cost={opt_result.final_cost:.4f}")

# ---------------------------------------------------------------------------
# 6. Accumulate point-cloud map
# ---------------------------------------------------------------------------

pcd_map = PointCloudMap()
for i, scan in enumerate(lidar_scans):
    node_pose = opt_result.optimized_poses.get(i, {"translation": np.zeros(3)})
    world_t = np.asarray(node_pose["translation"])
    T = np.eye(4)
    T[:3, 3] = world_t
    pcd_map.add_scan(scan, T)

pcd_map.voxel_downsample(voxel_size=0.5)
print(f"Accumulated map: {pcd_map.get_points().shape[0]:,} points after downsampling.")

# ---------------------------------------------------------------------------
# 7. Save map
# ---------------------------------------------------------------------------

map_path = bag_path.parent / "map.pcd"
pcd_map.save_pcd(map_path)
print(f"Map saved to: {map_path}")

print("\nPipeline complete.")
