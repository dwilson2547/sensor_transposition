"""Tests for SLAMSession LocalMap helper class and use_local_map option."""

import numpy as np
import pytest

from sensor_transposition.slam_session import LocalMap, _voxel_downsample_simple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_cloud(n: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-5.0, 5.0, (n, 3))


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
