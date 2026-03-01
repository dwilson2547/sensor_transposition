"""Tests for point_cloud_map: PointCloudMap accumulator and voxel downsampling."""

import math
import os
import tempfile

import numpy as np
import pytest

from sensor_transposition.point_cloud_map import PointCloudMap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity() -> np.ndarray:
    """Return a 4×4 identity transform."""
    return np.eye(4, dtype=float)


def _translation_tf(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    """Return a 4×4 pure-translation transform."""
    T = np.eye(4, dtype=float)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


def _rot_z(angle_rad: float) -> np.ndarray:
    """Return a 4×4 pure-rotation about z."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    T = np.eye(4, dtype=float)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    return T


def _random_scan(n: int = 20, seed: int = 0) -> np.ndarray:
    """Random (N, 3) float scan in a 10 m cube."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-5.0, 5.0, (n, 3))


def _random_colors_uint8(n: int = 20, seed: int = 0) -> np.ndarray:
    """Random (N, 3) uint8 RGB colours."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (n, 3), dtype=np.uint8)


def _random_colors_float(n: int = 20, seed: int = 0) -> np.ndarray:
    """Random (N, 3) float RGB colours in [0, 1]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, (n, 3)).astype(float)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_empty_on_creation(self):
        m = PointCloudMap()
        assert len(m) == 0

    def test_get_points_empty(self):
        m = PointCloudMap()
        pts = m.get_points()
        assert pts.shape == (0, 3)

    def test_get_colors_empty_returns_none(self):
        m = PointCloudMap()
        assert m.get_colors() is None

    def test_max_points_none_is_default(self):
        m = PointCloudMap()
        assert m._max_points is None

    def test_max_points_zero_raises(self):
        with pytest.raises(ValueError, match="max_points"):
            PointCloudMap(max_points=0)

    def test_max_points_negative_raises(self):
        with pytest.raises(ValueError, match="max_points"):
            PointCloudMap(max_points=-1)


# ---------------------------------------------------------------------------
# add_scan — basic
# ---------------------------------------------------------------------------


class TestAddScanBasic:
    def test_single_scan_identity_transform(self):
        m = PointCloudMap()
        pts = _random_scan(10, seed=1)
        m.add_scan(pts, _identity())
        assert len(m) == 10
        np.testing.assert_allclose(m.get_points(), pts, atol=1e-12)

    def test_two_scans_accumulate(self):
        m = PointCloudMap()
        pts1 = _random_scan(10, seed=2)
        pts2 = _random_scan(10, seed=3)
        m.add_scan(pts1, _identity())
        m.add_scan(pts2, _identity())
        assert len(m) == 20

    def test_translation_transform_applied(self):
        m = PointCloudMap()
        pts = np.array([[1.0, 0.0, 0.0]])
        m.add_scan(pts, _translation_tf(tx=5.0))
        world_pts = m.get_points()
        np.testing.assert_allclose(world_pts[0], [6.0, 0.0, 0.0], atol=1e-12)

    def test_rotation_transform_applied(self):
        """A 90° z-rotation should move [1, 0, 0] to [0, 1, 0]."""
        m = PointCloudMap()
        pts = np.array([[1.0, 0.0, 0.0]])
        m.add_scan(pts, _rot_z(math.radians(90)))
        world_pts = m.get_points()
        np.testing.assert_allclose(world_pts[0], [0.0, 1.0, 0.0], atol=1e-12)

    def test_get_points_returns_copy(self):
        """Mutating the returned array must not affect the internal buffer."""
        m = PointCloudMap()
        m.add_scan(_random_scan(5, seed=4), _identity())
        pts = m.get_points()
        original = pts.copy()
        pts[:] = 999.0
        np.testing.assert_allclose(m.get_points(), original, atol=1e-12)


# ---------------------------------------------------------------------------
# add_scan — colour handling
# ---------------------------------------------------------------------------


class TestAddScanColors:
    def test_uint8_colors_stored(self):
        m = PointCloudMap()
        pts = _random_scan(5, seed=5)
        clr = _random_colors_uint8(5, seed=5)
        m.add_scan(pts, _identity(), colors=clr)
        stored = m.get_colors()
        assert stored is not None
        assert stored.dtype == np.uint8
        np.testing.assert_array_equal(stored, clr)

    def test_float_colors_converted_to_uint8(self):
        m = PointCloudMap()
        n = 5
        pts = _random_scan(n, seed=6)
        clr_float = np.ones((n, 3), dtype=float)  # all white → 255
        m.add_scan(pts, _identity(), colors=clr_float)
        stored = m.get_colors()
        assert stored is not None
        assert stored.dtype == np.uint8
        np.testing.assert_array_equal(stored, np.full((n, 3), 255, dtype=np.uint8))

    def test_no_colors_returns_none(self):
        m = PointCloudMap()
        m.add_scan(_random_scan(5, seed=7), _identity())
        assert m.get_colors() is None

    def test_get_colors_returns_copy(self):
        m = PointCloudMap()
        n = 4
        pts = _random_scan(n, seed=8)
        clr = _random_colors_uint8(n, seed=8)
        m.add_scan(pts, _identity(), colors=clr)
        stored = m.get_colors()
        original = stored.copy()
        stored[:] = 0
        np.testing.assert_array_equal(m.get_colors(), original)

    def test_colors_accumulate(self):
        m = PointCloudMap()
        pts1 = _random_scan(4, seed=9)
        clr1 = _random_colors_uint8(4, seed=9)
        pts2 = _random_scan(6, seed=10)
        clr2 = _random_colors_uint8(6, seed=10)
        m.add_scan(pts1, _identity(), colors=clr1)
        m.add_scan(pts2, _identity(), colors=clr2)
        stored = m.get_colors()
        assert stored is not None
        assert stored.shape == (10, 3)

    def test_mix_colored_uncolored_raises(self):
        """Adding an uncoloured scan after a coloured one must raise."""
        m = PointCloudMap()
        pts = _random_scan(4, seed=11)
        clr = _random_colors_uint8(4, seed=11)
        m.add_scan(pts, _identity(), colors=clr)
        with pytest.raises(ValueError, match="mix"):
            m.add_scan(_random_scan(4, seed=12), _identity())

    def test_mix_uncolored_colored_raises(self):
        """Adding a coloured scan after an uncoloured one must raise."""
        m = PointCloudMap()
        pts = _random_scan(4, seed=13)
        m.add_scan(pts, _identity())
        with pytest.raises(ValueError, match="mix"):
            clr = _random_colors_uint8(4, seed=13)
            m.add_scan(_random_scan(4, seed=14), _identity(), colors=clr)


# ---------------------------------------------------------------------------
# add_scan — input validation
# ---------------------------------------------------------------------------


class TestAddScanValidation:
    def test_wrong_shape_raises(self):
        m = PointCloudMap()
        with pytest.raises(ValueError, match="\\(N, 3\\)"):
            m.add_scan(np.zeros((5, 2)), _identity())

    def test_1d_array_raises(self):
        m = PointCloudMap()
        with pytest.raises(ValueError, match="\\(N, 3\\)"):
            m.add_scan(np.zeros(9), _identity())

    def test_empty_scan_raises(self):
        m = PointCloudMap()
        with pytest.raises(ValueError, match="at least one point"):
            m.add_scan(np.zeros((0, 3)), _identity())

    def test_wrong_transform_shape_raises(self):
        m = PointCloudMap()
        with pytest.raises(ValueError, match="4×4"):
            m.add_scan(_random_scan(5), np.eye(3))

    def test_color_shape_mismatch_raises(self):
        m = PointCloudMap()
        pts = _random_scan(5, seed=15)
        bad_colors = _random_colors_uint8(3, seed=15)  # wrong N
        with pytest.raises(ValueError, match="colors"):
            m.add_scan(pts, _identity(), colors=bad_colors)


# ---------------------------------------------------------------------------
# max_points cap
# ---------------------------------------------------------------------------


class TestMaxPoints:
    def test_cap_enforced(self):
        m = PointCloudMap(max_points=10)
        m.add_scan(_random_scan(8, seed=20), _identity())
        m.add_scan(_random_scan(8, seed=21), _identity())  # 16 total → cap to 10
        assert len(m) <= 10

    def test_cap_oldest_discarded(self):
        """After the cap, the newest points should be retained."""
        m = PointCloudMap(max_points=5)
        # First scan: all points at y=0
        pts1 = np.zeros((5, 3))
        pts1[:, 1] = 0.0
        m.add_scan(pts1, _identity())
        # Second scan: all points at y=1
        pts2 = np.zeros((5, 3))
        pts2[:, 1] = 1.0
        m.add_scan(pts2, _identity())
        world_pts = m.get_points()
        # Only the newest 5 points (y=1) should remain.
        np.testing.assert_allclose(world_pts[:, 1], np.ones(5), atol=1e-12)

    def test_max_points_with_colors(self):
        m = PointCloudMap(max_points=6)
        clr1 = _random_colors_uint8(4, seed=22)
        clr2 = _random_colors_uint8(4, seed=23)
        m.add_scan(_random_scan(4, seed=22), _identity(), colors=clr1)
        m.add_scan(_random_scan(4, seed=23), _identity(), colors=clr2)
        assert len(m) == 6
        stored_colors = m.get_colors()
        assert stored_colors is not None
        assert len(stored_colors) == 6


# ---------------------------------------------------------------------------
# voxel_downsample
# ---------------------------------------------------------------------------


class TestVoxelDownsample:
    def test_noop_on_empty_map(self):
        m = PointCloudMap()
        m.voxel_downsample(voxel_size=0.5)  # should not raise
        assert len(m) == 0

    def test_invalid_voxel_size_raises(self):
        m = PointCloudMap()
        with pytest.raises(ValueError, match="voxel_size"):
            m.voxel_downsample(voxel_size=0.0)

    def test_invalid_voxel_size_negative_raises(self):
        m = PointCloudMap()
        with pytest.raises(ValueError, match="voxel_size"):
            m.voxel_downsample(voxel_size=-1.0)

    def test_reduces_point_count(self):
        """Downsampling a dense cloud should reduce point count."""
        m = PointCloudMap()
        # 1000 random points in a 10 m cube → many will share the same 1-m voxel.
        rng = np.random.default_rng(30)
        pts = rng.uniform(0.0, 10.0, (1000, 3))
        m.add_scan(pts, _identity())
        before = len(m)
        m.voxel_downsample(voxel_size=1.0)
        assert len(m) < before

    def test_single_point_unchanged(self):
        m = PointCloudMap()
        pts = np.array([[1.0, 2.0, 3.0]])
        m.add_scan(pts, _identity())
        m.voxel_downsample(voxel_size=0.5)
        np.testing.assert_allclose(m.get_points()[0], [1.0, 2.0, 3.0], atol=1e-12)

    def test_points_in_same_voxel_become_centroid(self):
        """Two points in the same voxel should be replaced by their centroid."""
        m = PointCloudMap()
        pts = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])  # both in voxel [0]
        m.add_scan(pts, _identity())
        m.voxel_downsample(voxel_size=1.0)
        assert len(m) == 1
        np.testing.assert_allclose(m.get_points()[0], [0.15, 0.0, 0.0], atol=1e-12)

    def test_points_in_different_voxels_preserved(self):
        """Points in separate voxels should each produce one output point."""
        m = PointCloudMap()
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])  # 2 m apart, 1 m voxels
        m.add_scan(pts, _identity())
        m.voxel_downsample(voxel_size=1.0)
        assert len(m) == 2

    def test_color_averaging_in_voxel(self):
        """Colours of points in the same voxel should be averaged."""
        m = PointCloudMap()
        pts = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
        clr = np.array([[0, 0, 0], [200, 200, 200]], dtype=np.uint8)
        m.add_scan(pts, _identity(), colors=clr)
        m.voxel_downsample(voxel_size=1.0)
        stored = m.get_colors()
        assert stored is not None
        assert stored.shape == (1, 3)
        # Average of 0 and 200 is 100.
        np.testing.assert_array_equal(stored[0], [100, 100, 100])

    def test_large_voxel_collapses_all_to_one(self):
        """A voxel size that encompasses the entire cloud should yield 1 point."""
        m = PointCloudMap()
        rng = np.random.default_rng(31)
        pts = rng.uniform(0.0, 5.0, (100, 3))
        m.add_scan(pts, _identity())
        m.voxel_downsample(voxel_size=100.0)
        assert len(m) == 1
        np.testing.assert_allclose(m.get_points()[0], pts.mean(axis=0), atol=1e-12)


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_empties_map(self):
        m = PointCloudMap()
        m.add_scan(_random_scan(5, seed=40), _identity())
        m.clear()
        assert len(m) == 0

    def test_clear_points_returns_empty(self):
        m = PointCloudMap()
        m.add_scan(_random_scan(5, seed=41), _identity())
        m.clear()
        assert m.get_points().shape == (0, 3)

    def test_clear_colors_returns_none(self):
        m = PointCloudMap()
        pts = _random_scan(5, seed=42)
        clr = _random_colors_uint8(5, seed=42)
        m.add_scan(pts, _identity(), colors=clr)
        m.clear()
        assert m.get_colors() is None

    def test_can_add_after_clear(self):
        m = PointCloudMap()
        m.add_scan(_random_scan(5, seed=43), _identity())
        m.clear()
        m.add_scan(_random_scan(3, seed=44), _identity())
        assert len(m) == 3


# ---------------------------------------------------------------------------
# Integration: multiple scans with different transforms
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_two_scans_at_different_poses(self):
        """Points from two scans at different poses should be separated in
        world space."""
        m = PointCloudMap()
        # Both scans: the origin in their respective body frames.
        pts = np.array([[0.0, 0.0, 0.0]])

        m.add_scan(pts, _translation_tf(tx=0.0))
        m.add_scan(pts, _translation_tf(tx=10.0))

        world_pts = m.get_points()
        assert world_pts.shape == (2, 3)
        # First point should be at origin, second at x=10.
        np.testing.assert_allclose(world_pts[0], [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(world_pts[1], [10.0, 0.0, 0.0], atol=1e-12)

    def test_accumulated_then_downsampled(self):
        """A typical workflow: accumulate scans, then downsample the map."""
        m = PointCloudMap()
        rng = np.random.default_rng(50)
        for i in range(5):
            pts = rng.uniform(-1.0, 1.0, (50, 3))
            T = _translation_tf(tx=float(i) * 5.0)
            m.add_scan(pts, T)
        before = len(m)
        m.voxel_downsample(voxel_size=0.5)
        assert len(m) <= before
        assert len(m) >= 1


# ---------------------------------------------------------------------------
# Serialisation: PCD
# ---------------------------------------------------------------------------


class TestSavePcd:
    def test_save_and_reload_no_color(self):
        """Round-trip: save then load an uncoloured map via PCD."""
        m = PointCloudMap()
        pts = _random_scan(10, seed=60)
        m.add_scan(pts, _identity())

        with tempfile.NamedTemporaryFile(suffix=".pcd", delete=False) as f:
            path = f.name
        try:
            m.save_pcd(path)
            m2 = PointCloudMap.from_pcd(path)
        finally:
            os.unlink(path)

        assert len(m2) == 10
        np.testing.assert_allclose(m2.get_points(), m.get_points(), atol=1e-5)
        assert m2.get_colors() is None

    def test_save_and_reload_with_color(self):
        """Round-trip: save then load a coloured map via PCD."""
        m = PointCloudMap()
        pts = _random_scan(8, seed=61)
        clr = _random_colors_uint8(8, seed=61)
        m.add_scan(pts, _identity(), colors=clr)

        with tempfile.NamedTemporaryFile(suffix=".pcd", delete=False) as f:
            path = f.name
        try:
            m.save_pcd(path)
            m2 = PointCloudMap.from_pcd(path)
        finally:
            os.unlink(path)

        assert len(m2) == 8
        np.testing.assert_allclose(m2.get_points(), m.get_points(), atol=1e-5)
        stored = m2.get_colors()
        assert stored is not None
        np.testing.assert_array_equal(stored, clr)

    def test_save_empty_map_pcd(self):
        """Saving an empty map should produce a valid zero-point PCD file."""
        m = PointCloudMap()
        with tempfile.NamedTemporaryFile(suffix=".pcd", delete=False) as f:
            path = f.name
        try:
            m.save_pcd(path)
            m2 = PointCloudMap.from_pcd(path)
        finally:
            os.unlink(path)
        assert len(m2) == 0

    def test_pcd_file_contains_header(self):
        """The saved PCD file must include required header lines."""
        m = PointCloudMap()
        m.add_scan(_random_scan(3, seed=62), _identity())
        with tempfile.NamedTemporaryFile(suffix=".pcd", delete=False, mode="w") as f:
            path = f.name
        try:
            m.save_pcd(path)
            content = open(path).read()
        finally:
            os.unlink(path)
        assert "VERSION 0.7" in content
        assert "DATA ascii" in content
        assert "FIELDS x y z" in content

    def test_from_pcd_non_ascii_raises(self):
        """Loading a binary PCD file should raise ValueError."""
        content = "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 1\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 1\nDATA binary\n"
        with tempfile.NamedTemporaryFile(suffix=".pcd", delete=False, mode="w") as f:
            f.write(content)
            path = f.name
        try:
            with pytest.raises(ValueError, match="ASCII"):
                PointCloudMap.from_pcd(path)
        finally:
            os.unlink(path)

    def test_from_pcd_missing_xyz_raises(self):
        """Loading a PCD file without x/y/z fields should raise ValueError."""
        content = "# .PCD v0.7\nVERSION 0.7\nFIELDS intensity\nSIZE 4\nTYPE F\nCOUNT 1\nWIDTH 0\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 0\nDATA ascii\n"
        with tempfile.NamedTemporaryFile(suffix=".pcd", delete=False, mode="w") as f:
            f.write(content)
            path = f.name
        try:
            with pytest.raises(ValueError, match="x/y/z"):
                PointCloudMap.from_pcd(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Serialisation: PLY
# ---------------------------------------------------------------------------


class TestSavePly:
    def test_save_and_reload_no_color(self):
        """Round-trip: save then load an uncoloured map via PLY."""
        m = PointCloudMap()
        pts = _random_scan(10, seed=70)
        m.add_scan(pts, _identity())

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            path = f.name
        try:
            m.save_ply(path)
            m2 = PointCloudMap.from_ply(path)
        finally:
            os.unlink(path)

        assert len(m2) == 10
        np.testing.assert_allclose(m2.get_points(), m.get_points(), atol=1e-5)
        assert m2.get_colors() is None

    def test_save_and_reload_with_color(self):
        """Round-trip: save then load a coloured map via PLY."""
        m = PointCloudMap()
        pts = _random_scan(8, seed=71)
        clr = _random_colors_uint8(8, seed=71)
        m.add_scan(pts, _identity(), colors=clr)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            path = f.name
        try:
            m.save_ply(path)
            m2 = PointCloudMap.from_ply(path)
        finally:
            os.unlink(path)

        assert len(m2) == 8
        np.testing.assert_allclose(m2.get_points(), m.get_points(), atol=1e-5)
        stored = m2.get_colors()
        assert stored is not None
        np.testing.assert_array_equal(stored, clr)

    def test_save_empty_map_ply(self):
        """Saving an empty map should produce a valid zero-vertex PLY file."""
        m = PointCloudMap()
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            path = f.name
        try:
            m.save_ply(path)
            m2 = PointCloudMap.from_ply(path)
        finally:
            os.unlink(path)
        assert len(m2) == 0

    def test_ply_file_contains_header(self):
        """The saved PLY file must include the standard ply header."""
        m = PointCloudMap()
        m.add_scan(_random_scan(3, seed=72), _identity())
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False, mode="w") as f:
            path = f.name
        try:
            m.save_ply(path)
            content = open(path).read()
        finally:
            os.unlink(path)
        assert content.startswith("ply\n")
        assert "format ascii 1.0" in content
        assert "element vertex 3" in content
        assert "end_header" in content

    def test_from_ply_non_ascii_raises(self):
        """Loading a binary PLY file should raise ValueError."""
        content = "ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False, mode="w") as f:
            f.write(content)
            path = f.name
        try:
            with pytest.raises(ValueError, match="ASCII"):
                PointCloudMap.from_ply(path)
        finally:
            os.unlink(path)

    def test_from_ply_missing_xyz_raises(self):
        """Loading a PLY file without x/y/z vertex properties should raise."""
        content = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float intensity\nend_header\n1.0\n"
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False, mode="w") as f:
            f.write(content)
            path = f.name
        try:
            with pytest.raises(ValueError, match="x/y/z"):
                PointCloudMap.from_ply(path)
        finally:
            os.unlink(path)
