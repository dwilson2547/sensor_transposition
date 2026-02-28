"""Tests for LiDAR point-to-point ICP scan matching."""

import math

import numpy as np
import pytest

from sensor_transposition.lidar.scan_matching import (
    IcpResult,
    _apply_transform,
    _kabsch,
    _rt_to_matrix,
    _validate_cloud,
    icp_align,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rotation_z(angle_rad: float) -> np.ndarray:
    """3×3 rotation matrix about the z-axis."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _random_cloud(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-5.0, 5.0, (n, 3))


# ---------------------------------------------------------------------------
# _validate_cloud
# ---------------------------------------------------------------------------


class TestValidateCloud:
    def test_valid_cloud_passes(self):
        _validate_cloud(np.ones((10, 3)), "test")

    def test_wrong_columns_raises(self):
        with pytest.raises(ValueError, match="\\(N, 3\\)"):
            _validate_cloud(np.ones((10, 4)), "src")

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="\\(N, 3\\)"):
            _validate_cloud(np.ones(3), "src")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one point"):
            _validate_cloud(np.empty((0, 3)), "src")


# ---------------------------------------------------------------------------
# _kabsch
# ---------------------------------------------------------------------------


class TestKabsch:
    def test_identity_transform(self):
        """When src == tgt, Kabsch should return R=I and t=0."""
        pts = _random_cloud(50)
        R, t = _kabsch(pts, pts)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(t, np.zeros(3), atol=1e-10)

    def test_pure_translation(self):
        pts = _random_cloud(50)
        t_true = np.array([1.0, -2.0, 0.5])
        R, t = _kabsch(pts, pts + t_true)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(t, t_true, atol=1e-10)

    def test_pure_rotation_about_z(self):
        pts = _random_cloud(100)
        R_true = _rotation_z(math.pi / 4)
        tgt = (R_true @ pts.T).T
        R, t = _kabsch(pts, tgt)
        np.testing.assert_allclose(R, R_true, atol=1e-10)
        np.testing.assert_allclose(t, np.zeros(3), atol=1e-10)

    def test_rotation_and_translation(self):
        pts = _random_cloud(80)
        R_true = _rotation_z(0.3)
        t_true = np.array([0.5, 1.0, -0.2])
        tgt = (R_true @ pts.T).T + t_true
        R, t = _kabsch(pts, tgt)
        np.testing.assert_allclose(R, R_true, atol=1e-10)
        np.testing.assert_allclose(t, t_true, atol=1e-10)

    def test_result_is_valid_rotation(self):
        pts = _random_cloud(60)
        R_true = _rotation_z(1.2)
        R, _ = _kabsch(pts, (R_true @ pts.T).T)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# _apply_transform / _rt_to_matrix
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_apply_transform_identity(self):
        pts = _random_cloud(20)
        out = _apply_transform(np.eye(4), pts)
        np.testing.assert_allclose(out, pts, atol=1e-15)

    def test_rt_to_matrix_structure(self):
        R = _rotation_z(0.5)
        t = np.array([1.0, 2.0, 3.0])
        T = _rt_to_matrix(R, t)
        assert T.shape == (4, 4)
        np.testing.assert_array_equal(T[:3, :3], R)
        np.testing.assert_array_equal(T[:3, 3], t)
        np.testing.assert_array_equal(T[3, :], [0.0, 0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# icp_align – basic convergence tests
# ---------------------------------------------------------------------------


class TestIcpAlignIdentity:
    """Source == target: ICP should return near-identity transform."""

    def test_returns_icp_result(self):
        pts = _random_cloud(100)
        result = icp_align(pts, pts)
        assert isinstance(result, IcpResult)

    def test_transform_shape(self):
        pts = _random_cloud(50)
        result = icp_align(pts, pts)
        assert result.transform.shape == (4, 4)

    def test_transform_is_identity_when_src_eq_tgt(self):
        pts = _random_cloud(100)
        result = icp_align(pts, pts)
        np.testing.assert_allclose(result.transform, np.eye(4), atol=1e-6)

    def test_converged_flag(self):
        pts = _random_cloud(100)
        result = icp_align(pts, pts)
        assert result.converged

    def test_mse_near_zero(self):
        pts = _random_cloud(100)
        result = icp_align(pts, pts)
        assert result.mean_squared_error < 1e-10


class TestIcpAlignTranslation:
    """Source is a translated version of target."""

    def setup_method(self):
        self.pts = _random_cloud(200, seed=1)
        self.t_true = np.array([0.5, -0.3, 0.1])
        self.src = self.pts + self.t_true
        self.tgt = self.pts

    def test_recovers_translation(self):
        result = icp_align(self.src, self.tgt, max_iterations=100)
        recovered = _apply_transform(result.transform, self.src)
        np.testing.assert_allclose(recovered, self.tgt, atol=1e-4)

    def test_converged(self):
        result = icp_align(self.src, self.tgt, max_iterations=100)
        assert result.converged

    def test_mse_low(self):
        result = icp_align(self.src, self.tgt, max_iterations=100)
        assert result.mean_squared_error < 1e-6


class TestIcpAlignSmallRotation:
    """Source is rotated by a small angle; ICP should converge."""

    def test_small_rotation_about_z(self):
        pts = _random_cloud(300, seed=2)
        R_true = _rotation_z(math.radians(5))
        src = (R_true @ pts.T).T
        result = icp_align(src, pts, max_iterations=100)
        recovered = _apply_transform(result.transform, src)
        np.testing.assert_allclose(recovered, pts, atol=1e-3)

    def test_iterations_recorded(self):
        pts = _random_cloud(100)
        result = icp_align(pts, pts)
        assert result.num_iterations >= 1


# ---------------------------------------------------------------------------
# icp_align – initial_transform
# ---------------------------------------------------------------------------


class TestIcpInitialTransform:
    def test_initial_transform_applied(self):
        """Pass the true transform as initial_transform; ICP should refine to
        near-identity incremental correction."""
        pts = _random_cloud(150, seed=5)
        R_true = _rotation_z(math.radians(8))
        t_true = np.array([0.3, 0.1, 0.0])
        T_true = _make_transform(R_true, t_true)
        src = _apply_transform(T_true, pts)

        result = icp_align(src, pts, initial_transform=np.linalg.inv(T_true),
                           max_iterations=50)
        recovered = _apply_transform(result.transform, src)
        np.testing.assert_allclose(recovered, pts, atol=1e-3)

    def test_wrong_initial_transform_shape_raises(self):
        pts = _random_cloud(20)
        with pytest.raises(ValueError, match="initial_transform"):
            icp_align(pts, pts, initial_transform=np.eye(3))


# ---------------------------------------------------------------------------
# icp_align – max_correspondence_dist
# ---------------------------------------------------------------------------


class TestIcpCorrespondenceDist:
    def test_tight_dist_still_converges_on_nearby_cloud(self):
        pts = _random_cloud(100, seed=3)
        src = pts + 0.01  # tiny shift
        result = icp_align(src, pts, max_correspondence_dist=1.0,
                           max_iterations=50)
        assert result.mean_squared_error < 1e-4

    def test_zero_inliers_stops_early(self):
        """When no correspondences are within max_correspondence_dist, the
        loop should exit gracefully without raising."""
        src = _random_cloud(10, seed=0) + 1000.0  # far from target
        tgt = _random_cloud(10, seed=1)
        result = icp_align(src, tgt, max_correspondence_dist=0.001,
                           max_iterations=10)
        assert isinstance(result, IcpResult)


# ---------------------------------------------------------------------------
# icp_align – input validation
# ---------------------------------------------------------------------------


class TestIcpInputValidation:
    def test_source_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="source"):
            icp_align(np.ones((10, 4)), np.ones((10, 3)))

    def test_target_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="target"):
            icp_align(np.ones((10, 3)), np.ones((5, 2)))

    def test_max_iterations_zero_raises(self):
        pts = _random_cloud(10)
        with pytest.raises(ValueError, match="max_iterations"):
            icp_align(pts, pts, max_iterations=0)

    def test_negative_tolerance_raises(self):
        pts = _random_cloud(10)
        with pytest.raises(ValueError, match="tolerance"):
            icp_align(pts, pts, tolerance=-1.0)


# ---------------------------------------------------------------------------
# IcpResult dataclass
# ---------------------------------------------------------------------------


class TestIcpResult:
    def test_fields_accessible(self):
        T = np.eye(4)
        result = IcpResult(transform=T, converged=True,
                           num_iterations=5, mean_squared_error=0.001)
        assert result.converged is True
        assert result.num_iterations == 5
        assert result.mean_squared_error == pytest.approx(0.001)
        np.testing.assert_array_equal(result.transform, T)
