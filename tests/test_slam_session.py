"""Tests for SLAMSession LocalMap helper class and use_local_map option."""

import os
import tempfile

import numpy as np
import pytest

from sensor_transposition.point_cloud_map import PointCloudMap
from sensor_transposition.rosbag import BagWriter, BagReader
from sensor_transposition.slam_session import (
    LocalMap,
    LocalizationSession,
    SLAMSession,
    _voxel_downsample_simple,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_cloud(n: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-5.0, 5.0, (n, 3))


def _write_pcd_map(path: str, cloud: np.ndarray) -> None:
    """Save a PointCloudMap as a PCD file at *path*."""
    m = PointCloudMap()
    m.add_scan(cloud, np.eye(4))
    m.save_pcd(path)


def _write_bag_with_scans(path: str, scans) -> None:
    """Write a .sbag file with one /lidar/points message per scan in *scans*."""
    with BagWriter(path) as bag:
        for i, scan in enumerate(scans):
            bag.write("/lidar/points", float(i), {"xyz": scan.tolist()})


# ---------------------------------------------------------------------------
# LocalMap
# ---------------------------------------------------------------------------


class TestLocalMap:
    def test_empty_returns_empty_array(self):
        lm = LocalMap(window_size=5)
        sub = lm.get_submap()
        assert sub.shape == (0, 3)

    def test_len_zero_initially(self):
        lm = LocalMap()
        assert len(lm) == 0

    def test_add_one_keyframe(self):
        lm = LocalMap(window_size=5, voxel_size=0.0)
        cloud = _random_cloud(100)
        lm.add_keyframe(cloud)
        assert len(lm) == 1
        sub = lm.get_submap()
        assert sub.shape == (100, 3)

    def test_window_evicts_old_frames(self):
        lm = LocalMap(window_size=3, voxel_size=0.0)
        for i in range(5):
            lm.add_keyframe(_random_cloud(50, seed=i))
        assert len(lm) == 3

    def test_submap_contains_all_window_frames(self):
        lm = LocalMap(window_size=3, voxel_size=0.0)
        for i in range(3):
            lm.add_keyframe(_random_cloud(50, seed=i))
        sub = lm.get_submap()
        assert sub.shape == (150, 3)

    def test_downsampling_reduces_points(self):
        lm = LocalMap(window_size=5, voxel_size=1.0)
        cloud = _random_cloud(500)
        lm.add_keyframe(cloud)
        sub = lm.get_submap()
        assert sub.shape[0] < 500

    def test_clear_resets_map(self):
        lm = LocalMap(window_size=5)
        lm.add_keyframe(_random_cloud(100))
        lm.clear()
        assert len(lm) == 0
        assert lm.get_submap().shape == (0, 3)

    def test_window_size_less_than_1_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            LocalMap(window_size=0)

    def test_window_size_1(self):
        """window_size=1 should keep only the last frame."""
        lm = LocalMap(window_size=1, voxel_size=0.0)
        lm.add_keyframe(_random_cloud(50, seed=0))
        lm.add_keyframe(_random_cloud(60, seed=1))
        assert len(lm) == 1
        sub = lm.get_submap()
        # Should contain the second frame's 60 points (no downsampling).
        assert sub.shape[0] == 60


# ---------------------------------------------------------------------------
# _voxel_downsample_simple
# ---------------------------------------------------------------------------


class TestVoxelDownsampleSimple:
    def test_output_shape(self):
        cloud = _random_cloud(500)
        ds = _voxel_downsample_simple(cloud, voxel_size=0.5)
        assert ds.shape[1] == 3
        assert ds.shape[0] <= 500

    def test_one_voxel_centroid(self):
        pts = np.array([[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]])
        ds = _voxel_downsample_simple(pts, voxel_size=1.0)
        assert ds.shape == (1, 3)
        np.testing.assert_allclose(ds[0], pts.mean(axis=0))


# ---------------------------------------------------------------------------
# SLAMSession.load_map
# ---------------------------------------------------------------------------


class TestLoadMap:
    def test_load_pcd_sets_localization_only(self):
        cloud = _random_cloud(50)
        with tempfile.NamedTemporaryFile(suffix=".pcd", delete=False) as f:
            path = f.name
        try:
            _write_pcd_map(path, cloud)
            session = SLAMSession()
            assert not session.localization_only
            session.load_map(path)
            assert session.localization_only
        finally:
            os.unlink(path)

    def test_load_pcd_populates_point_cloud_map(self):
        cloud = _random_cloud(50)
        with tempfile.NamedTemporaryFile(suffix=".pcd", delete=False) as f:
            path = f.name
        try:
            _write_pcd_map(path, cloud)
            session = SLAMSession()
            session.load_map(path)
            loaded_pts = session.point_cloud_map.get_points()
            assert loaded_pts.shape == (50, 3)
        finally:
            os.unlink(path)

    def test_load_ply_sets_localization_only(self):
        cloud = _random_cloud(40)
        m = PointCloudMap()
        m.add_scan(cloud, np.eye(4))
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            path = f.name
        try:
            m.save_ply(path)
            session = SLAMSession()
            session.load_map(path)
            assert session.localization_only
            assert session.point_cloud_map.get_points().shape == (40, 3)
        finally:
            os.unlink(path)

    def test_load_map_unsupported_extension_raises(self):
        session = SLAMSession()
        with pytest.raises(ValueError, match="Unsupported"):
            session.load_map("map.xyz")

    def test_load_map_missing_file_raises(self):
        session = SLAMSession()
        with pytest.raises(Exception):
            session.load_map("/nonexistent/path/map.pcd")


# ---------------------------------------------------------------------------
# SLAMSession.run in localization-only mode
# ---------------------------------------------------------------------------


class TestLocalizationRun:
    def _setup(self, n_scans: int = 3, cloud_size: int = 80):
        """Create a map PCD and a bag file, returning their temp paths."""
        map_cloud = _random_cloud(cloud_size, seed=99)

        map_fd, map_path = tempfile.mkstemp(suffix=".pcd")
        os.close(map_fd)
        _write_pcd_map(map_path, map_cloud)

        scans = [_random_cloud(50, seed=i) for i in range(n_scans)]
        bag_fd, bag_path = tempfile.mkstemp(suffix=".sbag")
        os.close(bag_fd)
        _write_bag_with_scans(bag_path, scans)

        return map_path, bag_path, n_scans

    def test_trajectory_length_equals_scan_count(self):
        map_path, bag_path, n_scans = self._setup(n_scans=3)
        try:
            session = SLAMSession()
            session.load_map(map_path)
            with BagReader(bag_path) as bag:
                session.run(bag, lidar_topic="/lidar/points")
            assert len(session.trajectory) == n_scans
        finally:
            os.unlink(map_path)
            os.unlink(bag_path)

    def test_pose_graph_not_modified(self):
        map_path, bag_path, _ = self._setup(n_scans=2)
        try:
            session = SLAMSession()
            session.load_map(map_path)
            with BagReader(bag_path) as bag:
                session.run(bag, lidar_topic="/lidar/points")
            # No nodes should have been added to the pose graph.
            assert len(session.pose_graph.nodes) == 0
        finally:
            os.unlink(map_path)
            os.unlink(bag_path)

    def test_point_cloud_map_unchanged_after_run(self):
        map_path, bag_path, _ = self._setup()
        try:
            session = SLAMSession()
            session.load_map(map_path)
            n_before = len(session.point_cloud_map)
            with BagReader(bag_path) as bag:
                session.run(bag, lidar_topic="/lidar/points")
            assert len(session.point_cloud_map) == n_before
        finally:
            os.unlink(map_path)
            os.unlink(bag_path)

    def test_empty_map_raises_runtime_error(self):
        """Running against an empty map should raise RuntimeError."""
        # Build an empty PCD by writing a map with no scans.
        m = PointCloudMap()
        with tempfile.NamedTemporaryFile(suffix=".pcd", delete=False) as f:
            path = f.name
        try:
            m.save_pcd(path)
            # Manually patch the map to be empty but flag set.
            session = SLAMSession()
            session._point_cloud_map = PointCloudMap()
            session._localization_only = True

            scans = [_random_cloud(30)]
            bag_fd, bag_path = tempfile.mkstemp(suffix=".sbag")
            os.close(bag_fd)
            _write_bag_with_scans(bag_path, scans)
            try:
                with BagReader(bag_path) as bag:
                    with pytest.raises(RuntimeError, match="empty"):
                        session.run(bag)
            finally:
                os.unlink(bag_path)
        finally:
            os.unlink(path)

    def test_callbacks_fired_in_localization_mode(self):
        map_path, bag_path, n_scans = self._setup(n_scans=2)
        try:
            session = SLAMSession()
            session.load_map(map_path)

            received = []

            @session.on_topic("/lidar/points")
            def cb(msg):
                received.append(msg.timestamp)

            with BagReader(bag_path) as bag:
                session.run(bag, lidar_topic="/lidar/points")

            assert len(received) == n_scans
        finally:
            os.unlink(map_path)
            os.unlink(bag_path)


# ---------------------------------------------------------------------------
# LocalizationSession
# ---------------------------------------------------------------------------


class TestLocalizationSession:
    def test_constructor_sets_localization_only(self):
        cloud = _random_cloud(60)
        with tempfile.NamedTemporaryFile(suffix=".pcd", delete=False) as f:
            path = f.name
        try:
            _write_pcd_map(path, cloud)
            session = LocalizationSession(path)
            assert session.localization_only
        finally:
            os.unlink(path)

    def test_run_produces_trajectory(self):
        map_cloud = _random_cloud(80, seed=10)
        map_fd, map_path = tempfile.mkstemp(suffix=".pcd")
        os.close(map_fd)
        _write_pcd_map(map_path, map_cloud)

        scans = [_random_cloud(50, seed=i) for i in range(4)]
        bag_fd, bag_path = tempfile.mkstemp(suffix=".sbag")
        os.close(bag_fd)
        _write_bag_with_scans(bag_path, scans)

        try:
            session = LocalizationSession(map_path)
            with BagReader(bag_path) as bag:
                session.run(bag, lidar_topic="/lidar/points")
            assert len(session.trajectory) == 4
        finally:
            os.unlink(map_path)
            os.unlink(bag_path)

    def test_kwargs_forwarded_to_slam_session(self):
        cloud = _random_cloud(50)
        with tempfile.NamedTemporaryFile(suffix=".pcd", delete=False) as f:
            path = f.name
        try:
            _write_pcd_map(path, cloud)
            session = LocalizationSession(path, icp_max_iterations=5)
            assert session._icp_max_iterations == 5
        finally:
            os.unlink(path)

    def test_unsupported_map_raises_at_construction(self):
        with pytest.raises(ValueError, match="Unsupported"):
            LocalizationSession("map.bin")


# ---------------------------------------------------------------------------
# SLAMSession.run with IMU integration
# ---------------------------------------------------------------------------


def _write_bag_with_scans_and_imu(path: str, scans, imu_readings_per_gap: int = 3) -> None:
    """Write a .sbag file interleaving /lidar/points and /imu/data messages.

    Each LiDAR frame at integer timestamp i is preceded by
    *imu_readings_per_gap* IMU messages evenly spaced in [i-1, i].
    """
    with BagWriter(path) as bag:
        for i, scan in enumerate(scans):
            # Write IMU messages between frames.
            if i > 0:
                for k in range(imu_readings_per_gap):
                    ts = float(i - 1) + (k + 1) / (imu_readings_per_gap + 1)
                    bag.write(
                        "/imu/data",
                        ts,
                        {
                            "accel": [0.0, 0.0, 0.0],
                            "gyro": [0.0, 0.0, 0.0],
                        },
                    )
            bag.write("/lidar/points", float(i), {"xyz": scan.tolist()})


class TestSLAMSessionImuIntegration:
    def _setup(self, n_scans: int = 4, imu_per_gap: int = 3):
        scans = [_random_cloud(50, seed=i) for i in range(n_scans)]
        bag_fd, bag_path = tempfile.mkstemp(suffix=".sbag")
        os.close(bag_fd)
        _write_bag_with_scans_and_imu(bag_path, scans, imu_per_gap)
        return bag_path, n_scans

    def test_imu_factors_added_to_pose_graph(self):
        """With imu_topic set, an ImuFactor should be added for each non-first keyframe."""
        bag_path, n_scans = self._setup(n_scans=4, imu_per_gap=3)
        try:
            session = SLAMSession()
            with BagReader(bag_path) as bag:
                session.run(bag, lidar_topic="/lidar/points",
                            imu_topic="/imu/data")
            # Expect (n_scans - 1) IMU factors (one per inter-keyframe gap).
            assert len(session.pose_graph.imu_factors) == n_scans - 1
        finally:
            os.unlink(bag_path)

    def test_imu_factors_connect_consecutive_keyframes(self):
        """Each ImuFactor must connect consecutive keyframe IDs."""
        bag_path, n_scans = self._setup(n_scans=3, imu_per_gap=3)
        try:
            session = SLAMSession()
            with BagReader(bag_path) as bag:
                session.run(bag, lidar_topic="/lidar/points",
                            imu_topic="/imu/data")
            factors = session.pose_graph.imu_factors
            for k, factor in enumerate(factors):
                assert factor.from_id == k
                assert factor.to_id == k + 1
        finally:
            os.unlink(bag_path)

    def test_no_imu_factors_without_imu_topic(self):
        """Without imu_topic, the pose graph must contain no ImuFactors."""
        bag_path, _ = self._setup()
        try:
            session = SLAMSession()
            with BagReader(bag_path) as bag:
                session.run(bag, lidar_topic="/lidar/points")
            assert len(session.pose_graph.imu_factors) == 0
        finally:
            os.unlink(bag_path)

    def test_trajectory_length_unchanged_with_imu(self):
        """IMU integration must not affect the trajectory length."""
        bag_path, n_scans = self._setup()
        try:
            session = SLAMSession()
            with BagReader(bag_path) as bag:
                session.run(bag, lidar_topic="/lidar/points",
                            imu_topic="/imu/data")
            assert len(session.trajectory) == n_scans
        finally:
            os.unlink(bag_path)

    def test_no_imu_factor_when_too_few_imu_samples(self):
        """If no IMU samples arrive between keyframes, no ImuFactor is added."""
        scans = [_random_cloud(50, seed=i) for i in range(3)]
        bag_fd, bag_path = tempfile.mkstemp(suffix=".sbag")
        os.close(bag_fd)
        # Write bags with no IMU messages at all.
        with BagWriter(bag_path) as bag:
            for i, scan in enumerate(scans):
                bag.write("/lidar/points", float(i), {"xyz": scan.tolist()})
        try:
            session = SLAMSession()
            with BagReader(bag_path) as bag:
                session.run(bag, lidar_topic="/lidar/points",
                            imu_topic="/imu/data")
            assert len(session.pose_graph.imu_factors) == 0
        finally:
            os.unlink(bag_path)

    def test_imu_information_parameter_used(self):
        """The *imu_information* scalar is forwarded to the ImuFactor information matrix."""
        bag_path, _ = self._setup(n_scans=3, imu_per_gap=3)
        try:
            session = SLAMSession()
            with BagReader(bag_path) as bag:
                session.run(bag, lidar_topic="/lidar/points",
                            imu_topic="/imu/data",
                            imu_information=25.0)
            for factor in session.pose_graph.imu_factors:
                np.testing.assert_allclose(factor.information, np.eye(6) * 25.0)
        finally:
            os.unlink(bag_path)
