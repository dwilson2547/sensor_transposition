"""Tests for KISS-ICP LiDAR odometry module."""

import math

import numpy as np
import pytest

from sensor_transposition.lidar.kiss_icp_odometry import (
    AdaptiveThreshold,
    KissIcpOdometry,
    VoxelHashMap,
    _voxel_downsample,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_cloud(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-5.0, 5.0, (n, 3))


def _rotation_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# VoxelHashMap
# ---------------------------------------------------------------------------


class TestVoxelHashMap:
    def test_empty_map_returns_empty_array(self):
        vhm = VoxelHashMap(voxel_size=0.5)
        pts = vhm.get_points()
        assert pts.shape == (0, 3)

    def test_add_points_and_retrieve(self):
        vhm = VoxelHashMap(voxel_size=0.5)
        cloud = _random_cloud(100)
        vhm.add_points(cloud)
        pts = vhm.get_points()
        assert pts.shape[1] == 3
        assert pts.shape[0] >= 1
        assert pts.shape[0] <= 100

    def test_one_point_per_voxel(self):
        """Two points in the same voxel should produce one representative."""
        vhm = VoxelHashMap(voxel_size=1.0)
        vhm.add_points(np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]))
        assert len(vhm) == 1

    def test_different_voxels(self):
        vhm = VoxelHashMap(voxel_size=1.0)
        vhm.add_points(np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]))
        assert len(vhm) == 2

    def test_max_voxels_eviction(self):
        """Map should not exceed max_voxels."""
        vhm = VoxelHashMap(voxel_size=0.01, max_voxels=50)
        rng = np.random.default_rng(42)
        big_cloud = rng.uniform(-10, 10, (1000, 3))
        vhm.add_points(big_cloud)
        assert len(vhm) <= 50

    def test_clear(self):
        vhm = VoxelHashMap(voxel_size=0.5)
        vhm.add_points(_random_cloud(50))
        vhm.clear()
        assert len(vhm) == 0

    def test_invalid_voxel_size_raises(self):
        with pytest.raises(ValueError, match="voxel_size"):
            VoxelHashMap(voxel_size=0.0)


# ---------------------------------------------------------------------------
# AdaptiveThreshold
# ---------------------------------------------------------------------------


class TestAdaptiveThreshold:
    def test_initial_threshold(self):
        at = AdaptiveThreshold(initial_threshold=2.0)
        assert at.threshold == pytest.approx(6.0)   # 3 × sigma

    def test_update_decreases_sigma(self):
        at = AdaptiveThreshold(initial_threshold=2.0, alpha=0.9)
        at.update(0.1)
        assert at.threshold < 6.0

    def test_threshold_clamped_by_min(self):
        at = AdaptiveThreshold(initial_threshold=0.001, min_threshold=0.5)
        assert at.threshold >= 0.5

    def test_threshold_clamped_by_max(self):
        at = AdaptiveThreshold(initial_threshold=100.0, max_threshold=5.0)
        assert at.threshold <= 5.0


# ---------------------------------------------------------------------------
# _voxel_downsample
# ---------------------------------------------------------------------------


class TestVoxelDownsample:
    def test_output_within_bounds(self):
        cloud = _random_cloud(500)
        ds = _voxel_downsample(cloud, voxel_size=0.5)
        assert ds.shape[1] == 3
        assert ds.shape[0] <= cloud.shape[0]

    def test_single_point_unchanged(self):
        pt = np.array([[1.23, 4.56, 7.89]])
        ds = _voxel_downsample(pt, voxel_size=1.0)
        assert ds.shape == (1, 3)

    def test_all_same_voxel(self):
        pts = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])
        ds = _voxel_downsample(pts, voxel_size=1.0)
        assert ds.shape == (1, 3)
        np.testing.assert_allclose(ds[0], pts.mean(axis=0))


# ---------------------------------------------------------------------------
# KissIcpOdometry
# ---------------------------------------------------------------------------


class TestKissIcpOdometry:
    def test_first_frame_returns_identity(self):
        odom = KissIcpOdometry(voxel_size=0.5)
        cloud = _random_cloud(200)
        pose = odom.register_frame(cloud)
        np.testing.assert_array_equal(pose, np.eye(4))

    def test_pose_shape(self):
        odom = KissIcpOdometry(voxel_size=0.5)
        cloud = _random_cloud(100)
        pose = odom.register_frame(cloud)
        assert pose.shape == (4, 4)

    def test_second_frame_returns_matrix(self):
        odom = KissIcpOdometry(voxel_size=0.5)
        cloud = _random_cloud(200)
        odom.register_frame(cloud)
        shifted = cloud + np.array([0.1, 0.0, 0.0])
        pose = odom.register_frame(shifted)
        assert pose.shape == (4, 4)

    def test_invalid_scan_raises(self):
        odom = KissIcpOdometry(voxel_size=0.5)
        with pytest.raises(ValueError, match="\\(N, 3\\)"):
            odom.register_frame(np.ones((10, 4)))

    def test_invalid_voxel_size_raises(self):
        with pytest.raises(ValueError, match="voxel_size"):
            KissIcpOdometry(voxel_size=-1.0)

    def test_reset_clears_state(self):
        odom = KissIcpOdometry(voxel_size=0.5)
        cloud = _random_cloud(100)
        odom.register_frame(cloud)
        odom.register_frame(cloud + 0.1)
        odom.reset()
        # After reset, the next frame should give identity again.
        pose = odom.register_frame(cloud)
        np.testing.assert_array_equal(pose, np.eye(4))

    def test_pose_property(self):
        odom = KissIcpOdometry(voxel_size=0.5)
        pose_ref = odom.pose
        assert pose_ref.shape == (4, 4)

    def test_local_map_property(self):
        odom = KissIcpOdometry(voxel_size=0.5)
        assert isinstance(odom.local_map, VoxelHashMap)

    def test_adaptive_threshold_property(self):
        odom = KissIcpOdometry(voxel_size=0.5)
        assert isinstance(odom.adaptive_threshold, AdaptiveThreshold)

    def test_multiple_frames_no_crash(self):
        """Registering many frames should not raise."""
        odom = KissIcpOdometry(voxel_size=0.3)
        rng = np.random.default_rng(99)
        for i in range(10):
            cloud = rng.uniform(-5, 5, (100, 3))
            odom.register_frame(cloud)
