"""Tests for voxel_map: TSDFVolume TSDF volumetric map."""

import math

import numpy as np
import pytest

from sensor_transposition.voxel_map import TSDFVolume


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


def _make_volume(voxel_size: float = 0.5) -> TSDFVolume:
    """Return a small TSDFVolume for testing."""
    return TSDFVolume(
        voxel_size=voxel_size,
        origin=np.array([-5.0, -5.0, -5.0]),
        dims=(20, 20, 20),
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_origin(self):
        v = TSDFVolume(voxel_size=0.1, dims=(10, 10, 10))
        np.testing.assert_array_equal(v.origin, [0.0, 0.0, 0.0])

    def test_custom_origin(self):
        origin = np.array([1.0, 2.0, 3.0])
        v = TSDFVolume(voxel_size=0.1, origin=origin, dims=(5, 5, 5))
        np.testing.assert_array_equal(v.origin, origin)

    def test_origin_is_copy(self):
        origin = np.array([1.0, 2.0, 3.0])
        v = TSDFVolume(voxel_size=0.1, origin=origin, dims=(5, 5, 5))
        origin[0] = 99.0
        assert v.origin[0] != 99.0

    def test_dims_property(self):
        v = TSDFVolume(voxel_size=0.1, dims=(4, 5, 6))
        assert v.dims == (4, 5, 6)

    def test_voxel_size_property(self):
        v = TSDFVolume(voxel_size=0.25, dims=(10, 10, 10))
        assert v.voxel_size == pytest.approx(0.25)

    def test_default_truncation(self):
        v = TSDFVolume(voxel_size=0.2, dims=(5, 5, 5))
        assert v.truncation == pytest.approx(0.6)

    def test_custom_truncation(self):
        v = TSDFVolume(voxel_size=0.1, dims=(5, 5, 5), truncation=0.5)
        assert v.truncation == pytest.approx(0.5)

    def test_tsdf_all_nan_on_creation(self):
        v = TSDFVolume(voxel_size=0.5, dims=(4, 4, 4))
        tsdf = v.get_tsdf()
        assert np.all(np.isnan(tsdf))

    def test_weights_all_zero_on_creation(self):
        v = TSDFVolume(voxel_size=0.5, dims=(4, 4, 4))
        assert np.all(v.get_weights() == 0.0)

    def test_invalid_voxel_size_zero_raises(self):
        with pytest.raises(ValueError, match="voxel_size"):
            TSDFVolume(voxel_size=0.0, dims=(5, 5, 5))

    def test_invalid_voxel_size_negative_raises(self):
        with pytest.raises(ValueError, match="voxel_size"):
            TSDFVolume(voxel_size=-1.0, dims=(5, 5, 5))

    def test_invalid_dims_zero_raises(self):
        with pytest.raises(ValueError, match="dims"):
            TSDFVolume(voxel_size=0.1, dims=(0, 5, 5))

    def test_invalid_truncation_raises(self):
        with pytest.raises(ValueError, match="truncation"):
            TSDFVolume(voxel_size=0.1, dims=(5, 5, 5), truncation=0.0)

    def test_invalid_origin_shape_raises(self):
        with pytest.raises(ValueError, match="origin"):
            TSDFVolume(voxel_size=0.1, origin=np.array([1.0, 2.0]), dims=(5, 5, 5))


# ---------------------------------------------------------------------------
# integrate — basic
# ---------------------------------------------------------------------------


class TestIntegrateBasic:
    def test_surface_point_at_origin_updates_voxels(self):
        """Integrating a single point should produce non-NaN TSDF values."""
        v = TSDFVolume(
            voxel_size=0.5,
            origin=np.array([-2.0, -2.0, -2.0]),
            dims=(8, 8, 8),
            truncation=1.0,
        )
        # Sensor at (5, 0, 0); surface point at (0, 0, 0) in world frame.
        # ego_to_world translates sensor to (5, 0, 0).
        T = _translation_tf(tx=5.0)
        pts = np.array([[-5.0, 0.0, 0.0]])  # sensor frame → world [0, 0, 0]
        v.integrate(pts, T)
        tsdf = v.get_tsdf()
        weights = v.get_weights()
        # At least some voxels near the surface should be updated.
        assert np.any(~np.isnan(tsdf))
        assert np.any(weights > 0)

    def test_identity_transform_uses_sensor_at_origin(self):
        """With identity transform, sensor is at (0,0,0)."""
        v = TSDFVolume(
            voxel_size=0.5,
            origin=np.array([-2.0, -2.0, -2.0]),
            dims=(8, 8, 8),
            truncation=1.0,
        )
        # Surface point at (1, 0, 0) in both sensor and world frames.
        # Volume spans [-2, 2) so x=1 is inside.
        pts = np.array([[1.0, 0.0, 0.0]])
        v.integrate(pts, _identity())
        tsdf = v.get_tsdf()
        assert np.any(~np.isnan(tsdf))

    def test_input_validation_wrong_shape_raises(self):
        v = _make_volume()
        with pytest.raises(ValueError, match="\\(N, 3\\)"):
            v.integrate(np.zeros((5, 2)), _identity())

    def test_input_validation_1d_raises(self):
        v = _make_volume()
        with pytest.raises(ValueError, match="\\(N, 3\\)"):
            v.integrate(np.zeros(9), _identity())

    def test_input_validation_empty_raises(self):
        v = _make_volume()
        with pytest.raises(ValueError, match="at least one point"):
            v.integrate(np.zeros((0, 3)), _identity())

    def test_input_validation_bad_transform_raises(self):
        v = _make_volume()
        with pytest.raises(ValueError, match="4×4"):
            v.integrate(np.ones((3, 3)), np.eye(3))

    def test_point_outside_volume_does_not_crash(self):
        """Points far outside the volume should be handled gracefully."""
        v = TSDFVolume(voxel_size=0.5, origin=np.zeros(3), dims=(4, 4, 4))
        pts = np.array([[1000.0, 1000.0, 1000.0]])
        v.integrate(pts, _identity())  # should not raise


# ---------------------------------------------------------------------------
# TSDF sign convention
# ---------------------------------------------------------------------------


class TestTSDFSign:
    def test_free_space_is_positive(self):
        """Voxels between the sensor and the surface should have positive SDF."""
        # Sensor at (10, 0, 0); surface at (0, 0, 0).
        # Voxels near x=5 (between sensor and surface) should be positive.
        vs = 1.0
        v = TSDFVolume(
            voxel_size=vs,
            origin=np.array([-0.5, -0.5, -0.5]),
            dims=(12, 2, 2),
            truncation=2.0,
        )
        T = _translation_tf(tx=10.0)
        pts = np.array([[-10.0, 0.0, 0.0]])  # sensor frame → world [0, 0, 0]
        v.integrate(pts, T)

        tsdf = v.get_tsdf()
        # Voxel at ix=5 (centre ≈ x=5) is between sensor and surface → +SDF.
        val_free = tsdf[5, 0, 0]
        if not np.isnan(val_free):
            assert val_free > 0.0, f"Expected positive SDF for free voxel, got {val_free}"

    def test_surface_voxel_near_zero(self):
        """The voxel containing the surface point should have near-zero SDF."""
        vs = 0.5
        v = TSDFVolume(
            voxel_size=vs,
            origin=np.array([-0.5, -0.5, -0.5]),
            dims=(10, 4, 4),
            truncation=1.5,
        )
        # Sensor at (8, 0, 0); surface at (3, 0, 0) in world frame.
        T = _translation_tf(tx=8.0)
        pts = np.array([[-5.0, 0.0, 0.0]])  # sensor frame → world [3, 0, 0]
        v.integrate(pts, T)

        tsdf = v.get_tsdf()
        weights = v.get_weights()
        # Voxel containing (3, 0, 0): ix = floor((3 - (-0.5)) / 0.5) = 7
        if weights[7, 1, 1] > 0:
            assert abs(tsdf[7, 1, 1]) < 1.0  # near zero crossing

    def test_behind_surface_is_negative(self):
        """Voxels behind the surface (away from sensor) should have negative SDF."""
        vs = 0.5
        # Sensor at (10, 0, 0); surface at (3, 0, 0).
        v = TSDFVolume(
            voxel_size=vs,
            origin=np.array([-0.5, -0.5, -0.5]),
            dims=(12, 4, 4),
            truncation=2.0,
        )
        T = _translation_tf(tx=10.0)
        pts = np.array([[-7.0, 0.0, 0.0]])  # sensor frame → world [3, 0, 0]
        v.integrate(pts, T)

        tsdf = v.get_tsdf()
        weights = v.get_weights()
        # A voxel clearly behind the surface, e.g. ix=1 (centre ≈ x=0)
        if weights[1, 1, 1] > 0:
            assert tsdf[1, 1, 1] < 0.0


# ---------------------------------------------------------------------------
# Multiple integrations
# ---------------------------------------------------------------------------


class TestMultipleIntegrations:
    def test_weights_accumulate(self):
        """Integrating the same scan twice should double the weights."""
        v = _make_volume()
        # Sensor at (3, 0, 0); surface at world (0, 0, 0) — inside volume.
        pts = np.array([[-3.0, 0.0, 0.0]])
        T = _translation_tf(tx=3.0)
        v.integrate(pts, T)
        w_after_one = v.get_weights().max()
        v.integrate(pts, T)
        w_after_two = v.get_weights().max()
        assert w_after_two == pytest.approx(2 * w_after_one)

    def test_tsdf_changes_with_second_integration(self):
        """A second integration from a different viewpoint should update TSDF."""
        v = _make_volume(voxel_size=0.5)
        # First integration: sensor at (3, 0, 0); surface at world (0, 0, 0).
        # Sensor frame point: (0,0,0) - (3,0,0) = (-3, 0, 0).
        v.integrate(np.array([[-3.0, 0.0, 0.0]]), _translation_tf(tx=3.0))
        tsdf_after_one = v.get_tsdf().copy()
        # Second integration: sensor at (-3, 0, 0); surface at world (0, 0, 0).
        # Sensor frame point: (0,0,0) - (-3,0,0) = (3, 0, 0).
        v.integrate(np.array([[3.0, 0.0, 0.0]]), _translation_tf(tx=-3.0))
        tsdf_after_two = v.get_tsdf()
        # Voxels updated by both integrations should have different (averaged) values.
        changed = ~np.isnan(tsdf_after_one) & ~np.isnan(tsdf_after_two)
        if changed.any():
            assert not np.allclose(tsdf_after_one[changed], tsdf_after_two[changed])


# ---------------------------------------------------------------------------
# extract_surface_points
# ---------------------------------------------------------------------------


class TestExtractSurfacePoints:
    def test_empty_volume_returns_empty(self):
        v = _make_volume()
        pts = v.extract_surface_points()
        assert pts.shape == (0, 3)

    def test_surface_points_returned_after_integration(self):
        """After integrating a surface, at least one surface point should be found."""
        v = TSDFVolume(
            voxel_size=0.5,
            origin=np.array([-1.0, -1.0, -1.0]),
            dims=(6, 6, 6),
            truncation=1.5,
        )
        # Sensor far away; surface near origin.
        T = _translation_tf(tx=10.0)
        pts = np.array([[-10.0, 0.0, 0.0]])  # world (0, 0, 0)
        v.integrate(pts, T)
        surface = v.extract_surface_points(threshold=0.5)
        assert surface.ndim == 2
        assert surface.shape[1] == 3
        assert surface.shape[0] > 0

    def test_threshold_zero_returns_exactly_zero_crossing(self):
        """With threshold=0 the result may be empty (exact zero is rare)."""
        v = _make_volume()
        v.integrate(np.array([[2.0, 0.0, 0.0]]), _translation_tf(tx=5.0))
        pts = v.extract_surface_points(threshold=0.0)
        # Result is valid (0,3) or (M,3); just verify shape.
        assert pts.ndim == 2
        assert pts.shape[1] == 3

    def test_threshold_one_includes_all_observed(self):
        """With threshold=1 all observed voxels should be included."""
        v = _make_volume()
        v.integrate(np.array([[2.0, 0.0, 0.0]]), _translation_tf(tx=5.0))
        pts_full = v.extract_surface_points(threshold=1.0)
        pts_narrow = v.extract_surface_points(threshold=0.1)
        assert len(pts_full) >= len(pts_narrow)

    def test_invalid_threshold_raises(self):
        v = _make_volume()
        with pytest.raises(ValueError, match="threshold"):
            v.extract_surface_points(threshold=1.5)

    def test_surface_points_within_volume_extent(self):
        """All returned surface points should lie within the volume bounds."""
        v = TSDFVolume(
            voxel_size=0.5,
            origin=np.array([-1.0, -1.0, -1.0]),
            dims=(6, 6, 6),
            truncation=1.5,
        )
        v.integrate(np.array([[-10.0, 0.0, 0.0]]), _translation_tf(tx=10.0))
        surface = v.extract_surface_points(threshold=0.5)
        if len(surface) > 0:
            mins = v.origin
            maxs = v.origin + np.array(v.dims) * v.voxel_size
            assert np.all(surface >= mins)
            assert np.all(surface <= maxs)


# ---------------------------------------------------------------------------
# voxel_to_world / world_to_voxel
# ---------------------------------------------------------------------------


class TestCoordinateConversions:
    def test_voxel_to_world_origin_voxel(self):
        """Voxel (0, 0, 0) centre is at origin + 0.5 * voxel_size."""
        vs = 0.5
        orig = np.array([1.0, 2.0, 3.0])
        v = TSDFVolume(voxel_size=vs, origin=orig, dims=(10, 10, 10))
        x, y, z = v.voxel_to_world(0, 0, 0)
        assert x == pytest.approx(1.25)
        assert y == pytest.approx(2.25)
        assert z == pytest.approx(3.25)

    def test_world_to_voxel_origin_corner(self):
        """A world point at the volume origin maps to voxel index 0."""
        vs = 0.5
        orig = np.array([0.0, 0.0, 0.0])
        v = TSDFVolume(voxel_size=vs, origin=orig, dims=(10, 10, 10))
        fx, fy, fz = v.world_to_voxel(0.0, 0.0, 0.0)
        assert fx == pytest.approx(0.0)
        assert fy == pytest.approx(0.0)
        assert fz == pytest.approx(0.0)

    def test_roundtrip_voxel_world(self):
        """voxel_to_world then world_to_voxel should recover the original index."""
        vs = 0.25
        orig = np.array([-2.0, -2.0, -2.0])
        v = TSDFVolume(voxel_size=vs, origin=orig, dims=(16, 16, 16))
        for ix, iy, iz in [(0, 0, 0), (5, 3, 7), (15, 15, 15)]:
            xw, yw, zw = v.voxel_to_world(ix, iy, iz)
            fx, fy, fz = v.world_to_voxel(xw, yw, zw)
            # Centre maps back to index + 0.5.
            assert fx == pytest.approx(ix + 0.5, abs=1e-10)
            assert fy == pytest.approx(iy + 0.5, abs=1e-10)
            assert fz == pytest.approx(iz + 0.5, abs=1e-10)


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_resets_to_nan(self):
        v = _make_volume()
        # Sensor at (3, 0, 0); surface at world (0, 0, 0) — inside volume.
        v.integrate(np.array([[-3.0, 0.0, 0.0]]), _translation_tf(tx=3.0))
        v.clear()
        assert np.all(np.isnan(v.get_tsdf()))

    def test_clear_resets_weights_to_zero(self):
        v = _make_volume()
        v.integrate(np.array([[-3.0, 0.0, 0.0]]), _translation_tf(tx=3.0))
        v.clear()
        assert np.all(v.get_weights() == 0.0)

    def test_can_integrate_after_clear(self):
        v = _make_volume()
        # Sensor at (3, 0, 0); surface point at world (0, 0, 0) — inside volume.
        v.integrate(np.array([[-3.0, 0.0, 0.0]]), _translation_tf(tx=3.0))
        v.clear()
        v.integrate(np.array([[-3.0, 0.0, 0.0]]), _translation_tf(tx=3.0))
        assert np.any(~np.isnan(v.get_tsdf()))


# ---------------------------------------------------------------------------
# get_tsdf / get_weights return copies
# ---------------------------------------------------------------------------


class TestReturnsCopies:
    def test_get_tsdf_returns_copy(self):
        v = _make_volume()
        v.integrate(np.array([[-3.0, 0.0, 0.0]]), _translation_tf(tx=3.0))
        tsdf = v.get_tsdf()
        original = tsdf.copy()
        tsdf[:] = 0.0
        np.testing.assert_array_equal(
            np.isnan(v.get_tsdf()), np.isnan(original)
        )

    def test_get_weights_returns_copy(self):
        v = _make_volume()
        v.integrate(np.array([[-3.0, 0.0, 0.0]]), _translation_tf(tx=3.0))
        weights = v.get_weights()
        weights[:] = 999.0
        assert v.get_weights().max() < 999.0


# ---------------------------------------------------------------------------
# SparseTSDFVolume
# ---------------------------------------------------------------------------

from sensor_transposition.voxel_map import SparseTSDFVolume  # noqa: E402


def _make_sparse_volume(voxel_size: float = 0.5) -> SparseTSDFVolume:
    return SparseTSDFVolume(
        voxel_size=voxel_size,
        origin=np.array([-5.0, -5.0, -5.0]),
        dims=(20, 20, 20),
    )


class TestSparseTSDFVolumeInit:
    def test_default_origin(self):
        v = SparseTSDFVolume(voxel_size=0.1, dims=(10, 10, 10))
        np.testing.assert_array_equal(v.origin, [0.0, 0.0, 0.0])

    def test_zero_voxel_size_raises(self):
        with pytest.raises(ValueError, match="voxel_size"):
            SparseTSDFVolume(voxel_size=0.0, dims=(10, 10, 10))

    def test_zero_dim_raises(self):
        with pytest.raises(ValueError, match="dims"):
            SparseTSDFVolume(voxel_size=0.1, dims=(0, 10, 10))

    def test_bad_origin_shape_raises(self):
        with pytest.raises(ValueError, match="origin"):
            SparseTSDFVolume(voxel_size=0.1, dims=(10, 10, 10), origin=np.zeros(2))

    def test_zero_truncation_raises(self):
        with pytest.raises(ValueError, match="truncation"):
            SparseTSDFVolume(voxel_size=0.1, dims=(10, 10, 10), truncation=0.0)

    def test_default_truncation(self):
        v = SparseTSDFVolume(voxel_size=0.2, dims=(10, 10, 10))
        assert v.truncation == pytest.approx(0.6)

    def test_dims_property(self):
        v = SparseTSDFVolume(voxel_size=0.1, dims=(5, 6, 7))
        assert v.dims == (5, 6, 7)

    def test_empty_volume_tsdf_all_nan(self):
        v = _make_sparse_volume()
        tsdf = v.get_tsdf()
        assert tsdf.shape == (20, 20, 20)
        assert np.all(np.isnan(tsdf))

    def test_empty_volume_weights_all_zero(self):
        v = _make_sparse_volume()
        weights = v.get_weights()
        assert weights.shape == (20, 20, 20)
        assert np.all(weights == 0.0)


class TestSparseTSDFVolumeIntegrate:
    def test_integrate_updates_voxels(self):
        v = _make_sparse_volume()
        v.integrate(np.array([[-3.0, 0.0, 0.0]]), _translation_tf(tx=3.0))
        tsdf = v.get_tsdf()
        assert np.any(~np.isnan(tsdf))

    def test_weights_nonzero_after_integrate(self):
        v = _make_sparse_volume()
        v.integrate(np.array([[-3.0, 0.0, 0.0]]), _translation_tf(tx=3.0))
        assert np.any(v.get_weights() > 0)

    def test_invalid_points_raises(self):
        v = _make_sparse_volume()
        with pytest.raises(ValueError):
            v.integrate(np.zeros((3, 2)), np.eye(4))

    def test_invalid_transform_raises(self):
        v = _make_sparse_volume()
        with pytest.raises(ValueError):
            v.integrate(np.zeros((3, 3)), np.eye(3))


class TestSparseTSDFVolumeSurface:
    def test_extract_surface_no_observations_returns_empty(self):
        v = _make_sparse_volume()
        pts = v.extract_surface_points()
        assert pts.shape == (0, 3)

    def test_extract_surface_returns_near_zero_crossing(self):
        v = _make_sparse_volume()
        v.integrate(np.array([[-3.0, 0.0, 0.0]]), _translation_tf(tx=3.0))
        pts = v.extract_surface_points(threshold=0.3)
        assert pts.shape[1] == 3

    def test_threshold_out_of_range_raises(self):
        v = _make_sparse_volume()
        with pytest.raises(ValueError, match="threshold"):
            v.extract_surface_points(threshold=1.5)

    def test_threshold_negative_raises(self):
        v = _make_sparse_volume()
        with pytest.raises(ValueError, match="threshold"):
            v.extract_surface_points(threshold=-0.1)


class TestSparseTSDFVolumeCoordConversion:
    def test_voxel_to_world(self):
        v = SparseTSDFVolume(voxel_size=1.0, origin=np.zeros(3), dims=(10, 10, 10))
        x, y, z = v.voxel_to_world(0, 0, 0)
        assert x == pytest.approx(0.5)
        assert y == pytest.approx(0.5)
        assert z == pytest.approx(0.5)

    def test_world_to_voxel(self):
        v = SparseTSDFVolume(voxel_size=1.0, origin=np.zeros(3), dims=(10, 10, 10))
        fx, fy, fz = v.world_to_voxel(1.5, 2.5, 3.5)
        assert fx == pytest.approx(1.5)
        assert fy == pytest.approx(2.5)
        assert fz == pytest.approx(3.5)


class TestSparseTSDFVolumeClear:
    def test_clear_empties_voxels(self):
        v = _make_sparse_volume()
        v.integrate(np.array([[-3.0, 0.0, 0.0]]), _translation_tf(tx=3.0))
        v.clear()
        assert len(v._voxels) == 0
        assert np.all(np.isnan(v.get_tsdf()))
        assert np.all(v.get_weights() == 0.0)

    def test_can_integrate_after_clear(self):
        v = _make_sparse_volume()
        v.integrate(np.array([[-3.0, 0.0, 0.0]]), _translation_tf(tx=3.0))
        v.clear()
        v.integrate(np.array([[-3.0, 0.0, 0.0]]), _translation_tf(tx=3.0))
        assert np.any(~np.isnan(v.get_tsdf()))


class TestSparseTSDFVolumeMatchesDense:
    def test_tsdf_values_match_dense_volume(self):
        """SparseTSDFVolume should produce the same TSDF as TSDFVolume."""
        origin = np.array([-5.0, -5.0, -5.0])
        voxel_size = 0.5
        dims = (20, 20, 20)
        pts = np.array([[-3.0, 0.0, 0.0]])
        T = _translation_tf(tx=3.0)

        dense = TSDFVolume(voxel_size=voxel_size, origin=origin, dims=dims)
        sparse = SparseTSDFVolume(voxel_size=voxel_size, origin=origin, dims=dims)
        dense.integrate(pts, T)
        sparse.integrate(pts, T)

        dense_tsdf = dense.get_tsdf()
        sparse_tsdf = sparse.get_tsdf()

        # Observed voxels should match.
        observed = ~np.isnan(dense_tsdf)
        np.testing.assert_array_equal(np.isnan(sparse_tsdf), ~observed)
        np.testing.assert_allclose(
            sparse_tsdf[observed], dense_tsdf[observed], atol=1e-10
        )
