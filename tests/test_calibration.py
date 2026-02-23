"""Tests for the calibration module (target-based extrinsic calibration)."""

import math

import numpy as np
import pytest

from sensor_transposition.calibration import (
    calibrate_lidar_camera,
    fit_plane,
    ransac_plane,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rotation_z(angle_deg: float) -> np.ndarray:
    """Return a 3×3 rotation matrix for a rotation about the Z-axis."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _points_on_plane(normal: np.ndarray, d: float, n_pts: int = 20,
                     rng: np.random.Generator = None) -> np.ndarray:
    """Generate *n_pts* 3-D points that lie on the plane ``normal · p = d``."""
    if rng is None:
        rng = np.random.default_rng(0)
    # Build two orthogonal vectors in the plane.
    u = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(u, normal)) > 0.9:
        u = np.array([0.0, 1.0, 0.0])
    u = u - np.dot(u, normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    coords_uv = rng.standard_normal((n_pts, 2))
    # A point on the plane: p0 = normal * d
    p0 = normal * d
    pts = p0 + coords_uv[:, 0:1] * u + coords_uv[:, 1:2] * v
    return pts


def _synthetic_calibration_data(
    T_true: np.ndarray,
    lidar_normals_in: np.ndarray,
    lidar_distances_in: np.ndarray,
) -> tuple:
    """Transform LiDAR plane observations to camera frame using *T_true*.

    Returns (lidar_normals, lidar_distances, camera_normals, camera_distances).
    """
    R = T_true[:3, :3]
    t = T_true[:3, 3]

    camera_normals = (R @ lidar_normals_in.T).T  # (N, 3)
    # d_camera = d_lidar + n_camera · t
    camera_distances = lidar_distances_in + camera_normals @ t

    return lidar_normals_in, lidar_distances_in, camera_normals, camera_distances


# ---------------------------------------------------------------------------
# fit_plane
# ---------------------------------------------------------------------------


class TestFitPlane:
    def test_xy_plane_returns_z_normal(self):
        """Points on z=0 plane should give normal [0,0,1] and distance 0."""
        rng = np.random.default_rng(1)
        pts = np.hstack([rng.standard_normal((30, 2)), np.zeros((30, 1))])
        n, d = fit_plane(pts)
        assert abs(abs(n[2]) - 1.0) < 1e-10
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_known_plane(self):
        """Points on the plane normal·p = 5 (a tilted plane with d = 5.0)."""
        normal_true = np.array([1.0, 1.0, 1.0]) / math.sqrt(3)
        d_true = 5.0
        pts = _points_on_plane(normal_true, d_true, n_pts=50, rng=np.random.default_rng(2))
        n, d = fit_plane(pts)
        # The normal may be flipped – check abs alignment.
        assert abs(abs(np.dot(n, normal_true)) - 1.0) < 1e-10
        assert abs(abs(d) - d_true) < 1e-10

    def test_distance_is_nonnegative(self):
        """The returned distance is always ≥ 0."""
        rng = np.random.default_rng(3)
        for _ in range(10):
            normal = rng.standard_normal(3)
            normal /= np.linalg.norm(normal)
            d_signed = rng.uniform(-10, 10)
            pts = _points_on_plane(normal, d_signed, n_pts=20, rng=rng)
            _, d = fit_plane(pts)
            assert d >= 0.0

    def test_normal_is_unit_length(self):
        rng = np.random.default_rng(4)
        pts = rng.standard_normal((20, 3))
        n, _ = fit_plane(pts)
        assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-12)

    def test_returns_correct_distance_value(self):
        """d should equal normal · centroid (within floating point)."""
        rng = np.random.default_rng(5)
        normal_true = np.array([0.0, 0.0, 1.0])
        d_true = 3.5
        pts = _points_on_plane(normal_true, d_true, n_pts=40, rng=rng)
        n, d = fit_plane(pts)
        assert d == pytest.approx(d_true, abs=1e-10)
        assert np.dot(n, pts.mean(axis=0)) == pytest.approx(d, abs=1e-10)

    def test_noisy_points_close_to_true_plane(self):
        """Small Gaussian noise should give a normal close to ground truth."""
        rng = np.random.default_rng(6)
        normal_true = np.array([1.0, 0.0, 0.0])
        d_true = 2.0
        pts = _points_on_plane(normal_true, d_true, n_pts=100, rng=rng)
        pts += rng.standard_normal(pts.shape) * 0.001  # small noise
        n, d = fit_plane(pts)
        assert abs(np.dot(n, normal_true)) == pytest.approx(1.0, abs=1e-3)
        assert d == pytest.approx(d_true, abs=1e-2)

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="3 points"):
            fit_plane(np.ones((2, 3)))

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            fit_plane(np.ones((10, 4)))

    def test_1d_input_raises(self):
        with pytest.raises(ValueError):
            fit_plane(np.ones(3))


# ---------------------------------------------------------------------------
# ransac_plane
# ---------------------------------------------------------------------------


class TestRansacPlane:
    def _plane_with_outliers(self, n_inliers=40, n_outliers=10,
                              normal=None, d=2.0, noise=0.01):
        rng = np.random.default_rng(42)
        if normal is None:
            normal = np.array([0.0, 0.0, 1.0])
        inlier_pts = _points_on_plane(normal, d, n_pts=n_inliers, rng=rng)
        inlier_pts += rng.standard_normal(inlier_pts.shape) * noise
        outlier_pts = rng.standard_normal((n_outliers, 3)) * 5 + 20
        pts = np.vstack([inlier_pts, outlier_pts])
        perm = rng.permutation(len(pts))
        return pts[perm], normal, d, n_inliers

    def test_finds_correct_plane(self):
        pts, normal_true, d_true, _ = self._plane_with_outliers()
        n, d, mask = ransac_plane(pts, distance_threshold=0.05, min_inliers=5, rng=0)
        assert abs(np.dot(n, normal_true)) == pytest.approx(1.0, abs=0.05)
        assert d == pytest.approx(d_true, abs=0.1)

    def test_inlier_mask_shape(self):
        pts, _, _, _ = self._plane_with_outliers()
        _, _, mask = ransac_plane(pts, distance_threshold=0.05, min_inliers=5, rng=0)
        assert mask.shape == (len(pts),)
        assert mask.dtype == bool

    def test_most_inliers_selected(self):
        """Inlier count should be at least the known inlier set size."""
        pts, _, _, n_inliers = self._plane_with_outliers(n_inliers=40, n_outliers=10)
        _, _, mask = ransac_plane(pts, distance_threshold=0.05, min_inliers=5, rng=0)
        assert mask.sum() >= n_inliers * 0.9  # allow a few missed due to noise

    def test_distance_is_nonnegative(self):
        pts, _, _, _ = self._plane_with_outliers()
        _, d, _ = ransac_plane(pts, distance_threshold=0.05, min_inliers=5, rng=0)
        assert d >= 0.0

    def test_normal_is_unit_length(self):
        pts, _, _, _ = self._plane_with_outliers()
        n, _, _ = ransac_plane(pts, distance_threshold=0.05, min_inliers=5, rng=0)
        assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-10)

    def test_integer_rng_seed_is_reproducible(self):
        pts, _, _, _ = self._plane_with_outliers()
        n1, d1, m1 = ransac_plane(pts, distance_threshold=0.05, min_inliers=5, rng=7)
        n2, d2, m2 = ransac_plane(pts, distance_threshold=0.05, min_inliers=5, rng=7)
        np.testing.assert_array_equal(n1, n2)
        assert d1 == d2
        np.testing.assert_array_equal(m1, m2)

    def test_generator_rng_accepted(self):
        pts, _, _, _ = self._plane_with_outliers()
        gen = np.random.default_rng(99)
        n, d, mask = ransac_plane(pts, distance_threshold=0.05, min_inliers=5, rng=gen)
        assert mask.shape == (len(pts),)

    def test_fails_when_min_inliers_too_high(self):
        rng = np.random.default_rng(0)
        pts = rng.standard_normal((20, 3)) * 10  # scattered, no dominant plane
        with pytest.raises(ValueError, match="RANSAC failed"):
            ransac_plane(pts, distance_threshold=0.001, max_iterations=10,
                         min_inliers=100, rng=1)

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="3 points"):
            ransac_plane(np.ones((2, 3)), rng=0)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            ransac_plane(np.ones((10, 2)), rng=0)


# ---------------------------------------------------------------------------
# calibrate_lidar_camera
# ---------------------------------------------------------------------------


class TestCalibrateLidarCamera:
    def _make_normals(self) -> np.ndarray:
        """Six diverse unit normals covering all axes and diagonals."""
        vecs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ], dtype=float)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def _make_T(self, angle_deg=30.0, translation=(0.5, 0.2, 0.1)) -> np.ndarray:
        T = np.eye(4, dtype=float)
        T[:3, :3] = _make_rotation_z(angle_deg)
        T[:3, 3] = translation
        return T

    def test_identity_transform(self):
        """When LiDAR and camera are co-located (identity T), result is I."""
        nl = self._make_normals()
        dl = np.array([2.0, 3.0, 1.5, 2.5, 2.0, 1.8])
        T_true = np.eye(4, dtype=float)
        nl, dl, nc, dc = _synthetic_calibration_data(T_true, nl, dl)
        T_est = calibrate_lidar_camera(nl, dl, nc, dc)
        np.testing.assert_allclose(T_est, T_true, atol=1e-10)

    def test_pure_rotation(self):
        """Recover a known rotation with zero translation."""
        nl = self._make_normals()
        dl = np.array([2.0, 3.0, 1.5, 2.5, 2.0, 1.8])
        T_true = self._make_T(angle_deg=45.0, translation=(0.0, 0.0, 0.0))
        nl, dl, nc, dc = _synthetic_calibration_data(T_true, nl, dl)
        T_est = calibrate_lidar_camera(nl, dl, nc, dc)
        np.testing.assert_allclose(T_est, T_true, atol=1e-10)

    def test_pure_translation(self):
        """Recover a known translation with identity rotation."""
        nl = self._make_normals()
        dl = np.array([2.0, 3.0, 1.5, 2.5, 2.0, 1.8])
        T_true = np.eye(4, dtype=float)
        T_true[:3, 3] = [0.3, -0.1, 0.8]
        nl, dl, nc, dc = _synthetic_calibration_data(T_true, nl, dl)
        T_est = calibrate_lidar_camera(nl, dl, nc, dc)
        np.testing.assert_allclose(T_est, T_true, atol=1e-10)

    def test_rotation_and_translation(self):
        """Recover a combined rigid transform."""
        nl = self._make_normals()
        dl = np.array([2.0, 3.0, 1.5, 2.5, 2.0, 1.8])
        T_true = self._make_T(angle_deg=30.0, translation=(0.5, 0.2, 0.1))
        nl, dl, nc, dc = _synthetic_calibration_data(T_true, nl, dl)
        T_est = calibrate_lidar_camera(nl, dl, nc, dc)
        np.testing.assert_allclose(T_est, T_true, atol=1e-10)

    def test_result_is_proper_rotation(self):
        """The rotation block of the result must be a proper rotation matrix."""
        nl = self._make_normals()
        dl = np.ones(6) * 2.0
        T_true = self._make_T(angle_deg=123.0, translation=(1.0, -0.5, 0.25))
        nl, dl, nc, dc = _synthetic_calibration_data(T_true, nl, dl)
        T_est = calibrate_lidar_camera(nl, dl, nc, dc)
        R = T_est[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)

    def test_last_row_is_homogeneous(self):
        """The last row of T must be [0, 0, 0, 1]."""
        nl = self._make_normals()
        dl = np.ones(6) * 2.0
        T_true = self._make_T()
        nl, dl, nc, dc = _synthetic_calibration_data(T_true, nl, dl)
        T_est = calibrate_lidar_camera(nl, dl, nc, dc)
        np.testing.assert_allclose(T_est[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)

    def test_minimum_three_correspondences(self):
        """Three observations should still work (minimum required)."""
        nl = self._make_normals()[:3]
        dl = np.array([2.0, 3.0, 1.5])
        T_true = self._make_T(angle_deg=15.0, translation=(0.1, 0.2, 0.3))
        nl, dl, nc, dc = _synthetic_calibration_data(T_true, nl, dl)
        T_est = calibrate_lidar_camera(nl, dl, nc, dc)
        np.testing.assert_allclose(T_est, T_true, atol=1e-8)

    def test_too_few_correspondences_raises(self):
        nl = self._make_normals()[:2]
        dl = np.ones(2)
        nc = nl.copy()
        dc = dl.copy()
        with pytest.raises(ValueError, match="3 plane correspondences"):
            calibrate_lidar_camera(nl, dl, nc, dc)

    def test_mismatched_row_counts_raises(self):
        nl = self._make_normals()         # 6 rows
        nc = self._make_normals()[:4]     # 4 rows
        with pytest.raises(ValueError, match="same number of rows"):
            calibrate_lidar_camera(nl, np.ones(6), nc, np.ones(4))

    def test_wrong_lidar_normals_shape_raises(self):
        with pytest.raises(ValueError, match="lidar_normals"):
            calibrate_lidar_camera(np.ones((4, 2)), np.ones(4),
                                   np.ones((4, 3)), np.ones(4))

    def test_wrong_camera_normals_shape_raises(self):
        with pytest.raises(ValueError, match="camera_normals"):
            calibrate_lidar_camera(np.ones((4, 3)), np.ones(4),
                                   np.ones((4, 2)), np.ones(4))

    def test_wrong_lidar_distances_shape_raises(self):
        nl = self._make_normals()
        with pytest.raises(ValueError, match="lidar_distances"):
            calibrate_lidar_camera(nl, np.ones(4), nl, np.ones(6))

    def test_wrong_camera_distances_shape_raises(self):
        nl = self._make_normals()
        with pytest.raises(ValueError, match="camera_distances"):
            calibrate_lidar_camera(nl, np.ones(6), nl, np.ones(4))

    def test_output_shape(self):
        nl = self._make_normals()
        dl = np.ones(6)
        T_true = self._make_T()
        nl, dl, nc, dc = _synthetic_calibration_data(T_true, nl, dl)
        T_est = calibrate_lidar_camera(nl, dl, nc, dc)
        assert T_est.shape == (4, 4)
        assert T_est.dtype == float


# ---------------------------------------------------------------------------
# Integration: fit_plane → calibrate_lidar_camera
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Verify that fit_plane output feeds correctly into calibrate_lidar_camera."""

    def test_fit_then_calibrate(self):
        """Fit planes to synthetic point clouds, then calibrate and check T."""
        rng = np.random.default_rng(2025)
        R = _make_rotation_z(20.0)
        t = np.array([0.3, -0.15, 0.5])
        T_true = np.eye(4)
        T_true[:3, :3] = R
        T_true[:3, 3] = t

        lidar_normals_list = []
        lidar_distances_list = []
        camera_normals_list = []
        camera_distances_list = []

        base_normals = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ], dtype=float)
        base_normals /= np.linalg.norm(base_normals, axis=1, keepdims=True)
        base_distances = np.array([2.0, 3.0, 1.5, 2.5])

        for nl_true, dl_true in zip(base_normals, base_distances):
            # Generate LiDAR inlier points on the plane with small noise.
            pts_lidar = _points_on_plane(nl_true, dl_true, n_pts=30, rng=rng)
            pts_lidar += rng.standard_normal(pts_lidar.shape) * 0.001

            nl, dl = fit_plane(pts_lidar)
            # Ensure consistent sign with the true normal.
            if np.dot(nl, nl_true) < 0:
                nl, dl = -nl, -dl

            # Derive camera plane from known T.
            nc = R @ nl
            dc = dl + np.dot(nc, t)

            lidar_normals_list.append(nl)
            lidar_distances_list.append(dl)
            camera_normals_list.append(nc)
            camera_distances_list.append(dc)

        T_est = calibrate_lidar_camera(
            np.array(lidar_normals_list),
            np.array(lidar_distances_list),
            np.array(camera_normals_list),
            np.array(camera_distances_list),
        )

        np.testing.assert_allclose(T_est[:3, :3], T_true[:3, :3], atol=1e-3)
        np.testing.assert_allclose(T_est[:3, 3], T_true[:3, 3], atol=1e-2)
