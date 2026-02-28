"""Tests for visual odometry: essential matrix, pose recovery, and PnP."""

import math

import numpy as np
import pytest

from sensor_transposition.visual_odometry import (
    EssentialMatrixResult,
    PnPResult,
    _decompose_essential,
    _eight_point,
    _hartley_normalise,
    _pixels_to_normalised,
    _reprojection_error_sq,
    _sampson_error,
    _triangulate_points,
    estimate_essential_matrix,
    recover_pose_from_essential,
    solve_pnp,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _rotation_y(angle_rad: float) -> np.ndarray:
    """3×3 rotation matrix about the y-axis."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_z(angle_rad: float) -> np.ndarray:
    """3×3 rotation matrix about the z-axis."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _make_K(fx: float = 500.0, fy: float = 500.0,
             cx: float = 320.0, cy: float = 240.0) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _project(K: np.ndarray, R: np.ndarray, t: np.ndarray,
              pts3: np.ndarray) -> np.ndarray:
    """Project (N,3) world points into pixels using K, R, t."""
    p_cam = (R @ pts3.T).T + t          # (N, 3)
    p_hom = (K @ p_cam.T).T            # (N, 3)
    return p_hom[:, :2] / p_hom[:, 2:3]  # (N, 2)


def _random_scene(
    n: int = 50,
    seed: int = 42,
    depth_min: float = 3.0,
    depth_max: float = 10.0,
) -> np.ndarray:
    """Return (N, 3) random world points in front of the camera."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-2.0, 2.0, (n, 3))
    pts[:, 2] = rng.uniform(depth_min, depth_max, n)
    return pts


# ---------------------------------------------------------------------------
# _hartley_normalise
# ---------------------------------------------------------------------------


class TestHartleyNormalise:
    def test_output_shape(self):
        pts = np.random.default_rng(0).uniform(-100, 100, (30, 2))
        _, normed = _hartley_normalise(pts)
        assert normed.shape == (30, 2)

    def test_zero_mean(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        _, normed = _hartley_normalise(pts)
        np.testing.assert_allclose(normed.mean(axis=0), [0.0, 0.0], atol=1e-10)

    def test_transform_matrix_shape(self):
        pts = np.random.default_rng(1).uniform(0, 640, (20, 2))
        T, _ = _hartley_normalise(pts)
        assert T.shape == (3, 3)

    def test_transform_is_invertible(self):
        pts = np.random.default_rng(2).uniform(0, 480, (40, 2))
        T, normed = _hartley_normalise(pts)
        N = pts.shape[0]
        hom = np.column_stack([normed, np.ones(N)])
        reconstructed = (np.linalg.inv(T) @ hom.T).T[:, :2]
        np.testing.assert_allclose(reconstructed, pts, atol=1e-8)


# ---------------------------------------------------------------------------
# _sampson_error
# ---------------------------------------------------------------------------


class TestSampsonError:
    def test_perfect_correspondences_near_zero(self):
        """Sampson error should be near-zero for exact epipolar pairs."""
        K = _make_K()
        R = _rotation_y(math.radians(5))
        t = np.array([0.5, 0.0, 0.0])

        pts3 = _random_scene(30, seed=1)
        pts1 = _project(K, np.eye(3), np.zeros(3), pts3)
        pts2 = _project(K, R, t, pts3)

        K_inv = np.linalg.inv(K)
        n1 = (K_inv @ np.column_stack([pts1, np.ones(30)]).T).T[:, :2]
        n2 = (K_inv @ np.column_stack([pts2, np.ones(30)]).T).T[:, :2]

        # Compute the true essential matrix: E = [t]× R
        tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        E_true = tx @ R

        errors = _sampson_error(E_true, n1, n2)
        np.testing.assert_array_less(errors, 1e-6 * np.ones_like(errors))

    def test_output_shape(self):
        E = np.eye(3)
        n1 = np.random.default_rng(0).uniform(-1, 1, (20, 2))
        n2 = np.random.default_rng(1).uniform(-1, 1, (20, 2))
        errors = _sampson_error(E, n1, n2)
        assert errors.shape == (20,)

    def test_errors_non_negative(self):
        E = np.eye(3)
        n1 = np.random.default_rng(2).uniform(-1, 1, (15, 2))
        n2 = np.random.default_rng(3).uniform(-1, 1, (15, 2))
        errors = _sampson_error(E, n1, n2)
        assert np.all(errors >= 0)


# ---------------------------------------------------------------------------
# _eight_point
# ---------------------------------------------------------------------------


class TestEightPoint:
    def test_returns_3x3(self):
        K = _make_K()
        R = _rotation_y(math.radians(5))
        t = np.array([0.5, 0.0, 0.0])
        pts3 = _random_scene(20)
        pts1 = _project(K, np.eye(3), np.zeros(3), pts3)
        pts2 = _project(K, R, t, pts3)
        K_inv = np.linalg.inv(K)
        n1 = (K_inv @ np.column_stack([pts1, np.ones(20)]).T).T[:, :2]
        n2 = (K_inv @ np.column_stack([pts2, np.ones(20)]).T).T[:, :2]
        E = _eight_point(n1, n2)
        assert E is not None
        assert E.shape == (3, 3)

    def test_rank_two(self):
        """Essential matrix must have rank 2 (one zero singular value)."""
        K = _make_K()
        R = _rotation_y(math.radians(8))
        t = np.array([0.3, 0.1, 0.0])
        pts3 = _random_scene(40)
        pts1 = _project(K, np.eye(3), np.zeros(3), pts3)
        pts2 = _project(K, R, t, pts3)
        K_inv = np.linalg.inv(K)
        n1 = (K_inv @ np.column_stack([pts1, np.ones(40)]).T).T[:, :2]
        n2 = (K_inv @ np.column_stack([pts2, np.ones(40)]).T).T[:, :2]
        E = _eight_point(n1, n2)
        assert E is not None
        s = np.linalg.svd(E, compute_uv=False)
        # Smallest singular value should be near zero.
        assert s[2] < 0.01 * s[0]

    def test_epipolar_constraint(self):
        """x2.T E x1 ≈ 0 for exact correspondences."""
        K = _make_K()
        R = _rotation_y(math.radians(10))
        t = np.array([1.0, 0.0, 0.0])
        pts3 = _random_scene(30, seed=5)
        pts1 = _project(K, np.eye(3), np.zeros(3), pts3)
        pts2 = _project(K, R, t, pts3)
        K_inv = np.linalg.inv(K)
        n1 = (K_inv @ np.column_stack([pts1, np.ones(30)]).T).T[:, :2]
        n2 = (K_inv @ np.column_stack([pts2, np.ones(30)]).T).T[:, :2]
        E = _eight_point(n1, n2)
        assert E is not None

        # Check average Sampson error is small.
        errors = _sampson_error(E, n1, n2)
        assert errors.mean() < 1e-3


# ---------------------------------------------------------------------------
# _decompose_essential
# ---------------------------------------------------------------------------


class TestDecomposeEssential:
    def _make_E(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        E = tx @ R
        return E / np.linalg.norm(E)

    def test_returns_valid_rotations(self):
        R_true = _rotation_y(math.radians(10))
        t_true = np.array([1.0, 0.0, 0.0])
        E = self._make_E(R_true, t_true)
        R1, R2, t = _decompose_essential(E)
        for R_cand in (R1, R2):
            np.testing.assert_allclose(R_cand @ R_cand.T, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(R_cand), 1.0, atol=1e-10)

    def test_translation_unit_norm(self):
        R_true = _rotation_z(math.radians(5))
        t_true = np.array([0.5, 0.3, 0.0])
        t_true /= np.linalg.norm(t_true)
        E = self._make_E(R_true, t_true)
        _, _, t = _decompose_essential(E)
        np.testing.assert_allclose(np.linalg.norm(t), 1.0, atol=1e-10)

    def test_four_candidates_cover_true_solution(self):
        """The true (R, t) must be one of the four (R1/R2, ±t) combinations."""
        R_true = _rotation_y(math.radians(7))
        t_true = np.array([1.0, 0.0, 0.0])
        t_true /= np.linalg.norm(t_true)
        E = self._make_E(R_true, t_true)
        R1, R2, t = _decompose_essential(E)

        candidates = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
        found = False
        for R_cand, t_cand in candidates:
            if (
                np.allclose(R_cand, R_true, atol=1e-5)
                and np.allclose(np.abs(t_cand), np.abs(t_true), atol=1e-5)
            ):
                found = True
                break
        assert found, "True (R, t) not among the four candidates."


# ---------------------------------------------------------------------------
# _triangulate_points
# ---------------------------------------------------------------------------


class TestTriangulatePoints:
    def test_triangulation_accuracy(self):
        K = _make_K()
        R = _rotation_y(math.radians(5))
        t = np.array([0.5, 0.0, 0.0])

        pts3 = _random_scene(20, seed=7)
        K_inv = np.linalg.inv(K)
        pts1 = _project(K, np.eye(3), np.zeros(3), pts3)
        pts2 = _project(K, R, t, pts3)
        n1 = (K_inv @ np.column_stack([pts1, np.ones(20)]).T).T[:, :2]
        n2 = (K_inv @ np.column_stack([pts2, np.ones(20)]).T).T[:, :2]

        P1 = np.eye(3, 4)
        P2 = np.hstack([R, t.reshape(3, 1)])
        tri = _triangulate_points(P1, P2, n1, n2)

        assert tri.shape == (20, 3)
        np.testing.assert_allclose(tri, pts3, atol=1e-4)

    def test_positive_depths(self):
        K = _make_K()
        R = _rotation_y(math.radians(3))
        t = np.array([0.3, 0.0, 0.0])
        pts3 = _random_scene(15)
        K_inv = np.linalg.inv(K)
        pts1 = _project(K, np.eye(3), np.zeros(3), pts3)
        pts2 = _project(K, R, t, pts3)
        n1 = (K_inv @ np.column_stack([pts1, np.ones(15)]).T).T[:, :2]
        n2 = (K_inv @ np.column_stack([pts2, np.ones(15)]).T).T[:, :2]
        P1 = np.eye(3, 4)
        P2 = np.hstack([R, t.reshape(3, 1)])
        tri = _triangulate_points(P1, P2, n1, n2)
        assert np.all(tri[:, 2] > 0)


# ---------------------------------------------------------------------------
# _reprojection_error_sq
# ---------------------------------------------------------------------------


class TestReprojectionErrorSq:
    def test_zero_error_for_exact_projection(self):
        K = _make_K()
        R = _rotation_y(math.radians(5))
        t = np.array([0.2, 0.0, 0.0])
        pts3 = _random_scene(20)
        pts2 = _project(K, R, t, pts3)

        errors = _reprojection_error_sq(R, t, K, pts3, pts2)
        np.testing.assert_allclose(errors, 0.0, atol=1e-8)

    def test_non_zero_error_for_wrong_pose(self):
        K = _make_K()
        R = _rotation_y(math.radians(5))
        t = np.array([0.2, 0.0, 0.0])
        pts3 = _random_scene(20)
        pts2 = _project(K, R, t, pts3)

        R_wrong = _rotation_y(math.radians(15))
        errors = _reprojection_error_sq(R_wrong, t, K, pts3, pts2)
        assert np.mean(errors) > 1.0

    def test_output_shape(self):
        K = _make_K()
        pts3 = _random_scene(10)
        pts2 = _project(K, np.eye(3), np.zeros(3), pts3)
        errors = _reprojection_error_sq(np.eye(3), np.zeros(3), K, pts3, pts2)
        assert errors.shape == (10,)


# ---------------------------------------------------------------------------
# estimate_essential_matrix
# ---------------------------------------------------------------------------


class TestEstimateEssentialMatrix:
    def _setup(
        self,
        n_inliers: int = 60,
        n_outliers: int = 0,
        angle_deg: float = 5.0,
        t: np.ndarray | None = None,
        seed: int = 42,
    ):
        K = _make_K()
        R = _rotation_y(math.radians(angle_deg))
        t_vec = np.array([0.5, 0.0, 0.0]) if t is None else t
        pts3 = _random_scene(n_inliers, seed=seed)
        pts1 = _project(K, np.eye(3), np.zeros(3), pts3)
        pts2 = _project(K, R, t_vec, pts3)

        if n_outliers > 0:
            rng = np.random.default_rng(seed + 1)
            out1 = rng.uniform(0, 640, (n_outliers, 2))
            out2 = rng.uniform(0, 480, (n_outliers, 2))
            pts1 = np.vstack([pts1, out1])
            pts2 = np.vstack([pts2, out2])

        return K, R, t_vec, pts1, pts2

    def test_returns_essential_matrix_result(self):
        K, _, _, pts1, pts2 = self._setup()
        result = estimate_essential_matrix(pts1, pts2, K)
        assert isinstance(result, EssentialMatrixResult)

    def test_essential_matrix_shape(self):
        K, _, _, pts1, pts2 = self._setup()
        result = estimate_essential_matrix(pts1, pts2, K)
        assert result.essential_matrix.shape == (3, 3)

    def test_inlier_mask_shape(self):
        K, _, _, pts1, pts2 = self._setup(n_inliers=60)
        result = estimate_essential_matrix(pts1, pts2, K)
        assert result.inlier_mask.shape == (60,)

    def test_num_inliers_matches_mask(self):
        K, _, _, pts1, pts2 = self._setup()
        result = estimate_essential_matrix(pts1, pts2, K)
        assert result.num_inliers == int(result.inlier_mask.sum())

    def test_high_inlier_ratio_clean_data(self):
        """With clean data almost all correspondences should be inliers."""
        K, _, _, pts1, pts2 = self._setup(n_inliers=80)
        result = estimate_essential_matrix(
            pts1, pts2, K,
            inlier_threshold=2.0,
            rng=np.random.default_rng(0),
        )
        assert result.num_inliers >= 70

    def test_essential_matrix_rank_two(self):
        K, _, _, pts1, pts2 = self._setup(n_inliers=50)
        result = estimate_essential_matrix(
            pts1, pts2, K, rng=np.random.default_rng(0)
        )
        s = np.linalg.svd(result.essential_matrix, compute_uv=False)
        assert s[2] < 0.05 * s[0]

    def test_ransac_handles_outliers(self):
        """With 30 % outliers, RANSAC should still find a majority of inliers."""
        K, _, _, pts1, pts2 = self._setup(n_inliers=70, n_outliers=30)
        result = estimate_essential_matrix(
            pts1, pts2, K,
            inlier_threshold=2.0,
            rng=np.random.default_rng(5),
        )
        # At least 60 of 70 true inliers should be recovered.
        assert result.num_inliers >= 60

    def test_epipolar_constraint_for_inliers(self):
        """Sampson error should be small for inlier correspondences."""
        K, _, _, pts1, pts2 = self._setup(n_inliers=60)
        result = estimate_essential_matrix(
            pts1, pts2, K, rng=np.random.default_rng(0)
        )
        K_inv = np.linalg.inv(K)
        n1 = _pixels_to_normalised(pts1[result.inlier_mask], K_inv)
        n2 = _pixels_to_normalised(pts2[result.inlier_mask], K_inv)
        errors = _sampson_error(result.essential_matrix, n1, n2)
        assert np.mean(errors) < 0.01


class TestEstimateEssentialMatrixInputValidation:
    def _good_pts(self, n=20):
        K = _make_K()
        pts3 = _random_scene(n)
        pts1 = _project(K, np.eye(3), np.zeros(3), pts3)
        pts2 = _project(K, _rotation_y(0.1), np.array([0.1, 0, 0]), pts3)
        return K, pts1, pts2

    def test_too_few_points_raises(self):
        K, pts1, pts2 = self._good_pts(n=20)
        with pytest.raises(ValueError, match="8 point"):
            estimate_essential_matrix(pts1[:7], pts2[:7], K)

    def test_wrong_K_shape_raises(self):
        K, pts1, pts2 = self._good_pts()
        with pytest.raises(ValueError, match="K must be 3×3"):
            estimate_essential_matrix(pts1, pts2, np.eye(4))

    def test_mismatched_row_count_raises(self):
        K, pts1, pts2 = self._good_pts()
        with pytest.raises(ValueError, match="same number of rows"):
            estimate_essential_matrix(pts1, pts2[:-1], K)

    def test_wrong_columns_raises(self):
        K, pts1, _ = self._good_pts()
        with pytest.raises(ValueError, match="\\(N, 2\\)"):
            estimate_essential_matrix(pts1, pts1.reshape(-1, 1), K)

    def test_invalid_confidence_raises(self):
        K, pts1, pts2 = self._good_pts()
        with pytest.raises(ValueError, match="confidence"):
            estimate_essential_matrix(pts1, pts2, K, confidence=1.5)

    def test_invalid_threshold_raises(self):
        K, pts1, pts2 = self._good_pts()
        with pytest.raises(ValueError, match="inlier_threshold"):
            estimate_essential_matrix(pts1, pts2, K, inlier_threshold=-1.0)

    def test_invalid_max_iterations_raises(self):
        K, pts1, pts2 = self._good_pts()
        with pytest.raises(ValueError, match="max_ransac_iterations"):
            estimate_essential_matrix(pts1, pts2, K, max_ransac_iterations=0)


# ---------------------------------------------------------------------------
# recover_pose_from_essential
# ---------------------------------------------------------------------------


class TestRecoverPoseFromEssential:
    def _setup(
        self,
        angle_deg: float = 8.0,
        t_true: np.ndarray | None = None,
        n: int = 60,
        seed: int = 42,
    ):
        K = _make_K()
        R_true = _rotation_y(math.radians(angle_deg))
        t_vec = np.array([0.5, 0.0, 0.0]) if t_true is None else t_true
        pts3 = _random_scene(n, seed=seed)
        pts1 = _project(K, np.eye(3), np.zeros(3), pts3)
        pts2 = _project(K, R_true, t_vec, pts3)
        return K, R_true, t_vec, pts1, pts2

    def test_returns_rotation_and_translation(self):
        K, _, _, pts1, pts2 = self._setup()
        result = estimate_essential_matrix(
            pts1, pts2, K, rng=np.random.default_rng(0)
        )
        R, t = recover_pose_from_essential(
            result.essential_matrix,
            pts1[result.inlier_mask],
            pts2[result.inlier_mask],
            K,
        )
        assert R.shape == (3, 3)
        assert t.shape == (3,)

    def test_rotation_is_valid(self):
        K, _, _, pts1, pts2 = self._setup()
        result = estimate_essential_matrix(
            pts1, pts2, K, rng=np.random.default_rng(0)
        )
        R, _ = recover_pose_from_essential(
            result.essential_matrix,
            pts1[result.inlier_mask],
            pts2[result.inlier_mask],
            K,
        )
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-8)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-8)

    def test_translation_is_unit_norm(self):
        K, _, _, pts1, pts2 = self._setup()
        result = estimate_essential_matrix(
            pts1, pts2, K, rng=np.random.default_rng(0)
        )
        _, t = recover_pose_from_essential(
            result.essential_matrix,
            pts1[result.inlier_mask],
            pts2[result.inlier_mask],
            K,
        )
        np.testing.assert_allclose(np.linalg.norm(t), 1.0, atol=1e-8)

    def test_recovers_correct_rotation_direction(self):
        """Recovered R should be close to the true R."""
        K, R_true, _, pts1, pts2 = self._setup(angle_deg=5.0, n=80)
        result = estimate_essential_matrix(
            pts1, pts2, K, rng=np.random.default_rng(1)
        )
        R, _ = recover_pose_from_essential(
            result.essential_matrix,
            pts1[result.inlier_mask],
            pts2[result.inlier_mask],
            K,
        )
        # Angular distance between R and R_true: trace(R_true.T @ R) ∈ [-1, 3]
        trace_val = np.trace(R_true.T @ R)
        angle_diff = math.acos(np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0))
        assert angle_diff < math.radians(5.0)


class TestRecoverPoseInputValidation:
    def _good_input(self):
        K = _make_K()
        R = _rotation_y(0.1)
        t = np.array([0.2, 0.0, 0.0])
        tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        E = tx @ R
        pts3 = _random_scene(20)
        pts1 = _project(K, np.eye(3), np.zeros(3), pts3)
        pts2 = _project(K, R, t, pts3)
        return E, pts1, pts2, K

    def test_too_few_points_raises(self):
        E, pts1, pts2, K = self._good_input()
        with pytest.raises(ValueError, match="5 point"):
            recover_pose_from_essential(E, pts1[:4], pts2[:4], K)

    def test_wrong_E_shape_raises(self):
        E, pts1, pts2, K = self._good_input()
        with pytest.raises(ValueError, match="E must be 3×3"):
            recover_pose_from_essential(np.eye(4), pts1, pts2, K)

    def test_wrong_K_shape_raises(self):
        E, pts1, pts2, K = self._good_input()
        with pytest.raises(ValueError, match="K must be 3×3"):
            recover_pose_from_essential(E, pts1, pts2, np.eye(2))


# ---------------------------------------------------------------------------
# solve_pnp
# ---------------------------------------------------------------------------


class TestSolvePnP:
    def _setup(
        self,
        n: int = 40,
        angle_deg: float = 10.0,
        t_true: np.ndarray | None = None,
        n_outliers: int = 0,
        seed: int = 0,
    ):
        K = _make_K()
        R_true = _rotation_y(math.radians(angle_deg))
        t_vec = np.array([1.0, 0.0, 0.0]) if t_true is None else t_true
        pts3 = _random_scene(n, seed=seed)
        pts2 = _project(K, R_true, t_vec, pts3)

        if n_outliers > 0:
            rng = np.random.default_rng(seed + 100)
            out2 = rng.uniform(0, 640, (n_outliers, 2))
            out3 = _random_scene(n_outliers, seed=seed + 200)
            pts3 = np.vstack([pts3, out3])
            pts2 = np.vstack([pts2, out2])

        return K, R_true, t_vec, pts3, pts2

    def test_returns_pnp_result(self):
        K, _, _, pts3, pts2 = self._setup()
        result = solve_pnp(pts3, pts2, K)
        assert isinstance(result, PnPResult)

    def test_success_flag_true_clean_data(self):
        K, _, _, pts3, pts2 = self._setup()
        result = solve_pnp(pts3, pts2, K, rng=np.random.default_rng(0))
        assert result.success

    def test_rotation_shape(self):
        K, _, _, pts3, pts2 = self._setup()
        result = solve_pnp(pts3, pts2, K, rng=np.random.default_rng(0))
        assert result.rotation.shape == (3, 3)

    def test_translation_shape(self):
        K, _, _, pts3, pts2 = self._setup()
        result = solve_pnp(pts3, pts2, K, rng=np.random.default_rng(0))
        assert result.translation.shape == (3,)

    def test_inlier_mask_shape(self):
        n = 40
        K, _, _, pts3, pts2 = self._setup(n=n)
        result = solve_pnp(pts3, pts2, K, rng=np.random.default_rng(0))
        assert result.inlier_mask.shape == (n,)

    def test_num_inliers_matches_mask(self):
        K, _, _, pts3, pts2 = self._setup()
        result = solve_pnp(pts3, pts2, K, rng=np.random.default_rng(0))
        assert result.num_inliers == int(result.inlier_mask.sum())

    def test_low_reprojection_error_clean_data(self):
        """Reprojection error for inliers should be sub-pixel on clean data."""
        K, R_true, t_true, pts3, pts2 = self._setup(n=50)
        result = solve_pnp(
            pts3, pts2, K,
            inlier_threshold=2.0,
            rng=np.random.default_rng(0),
        )
        assert result.success
        errors_sq = _reprojection_error_sq(
            result.rotation, result.translation, K,
            pts3[result.inlier_mask], pts2[result.inlier_mask],
        )
        assert np.sqrt(errors_sq.mean()) < 2.0

    def test_rotation_is_valid(self):
        K, _, _, pts3, pts2 = self._setup(n=50)
        result = solve_pnp(pts3, pts2, K, rng=np.random.default_rng(0))
        R = result.rotation
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)

    def test_handles_outliers(self):
        """With 20 % outliers, RANSAC should still succeed."""
        K, _, _, pts3, pts2 = self._setup(n=40, n_outliers=10)
        result = solve_pnp(
            pts3, pts2, K,
            inlier_threshold=2.0,
            rng=np.random.default_rng(0),
        )
        assert result.success
        assert result.num_inliers >= 30


class TestSolvePnPInputValidation:
    def _good_input(self, n: int = 20):
        K = _make_K()
        R = _rotation_y(0.1)
        t = np.array([0.5, 0.0, 0.0])
        pts3 = _random_scene(n)
        pts2 = _project(K, R, t, pts3)
        return K, pts3, pts2

    def test_too_few_points_raises(self):
        K, pts3, pts2 = self._good_input(n=20)
        with pytest.raises(ValueError, match="6 point"):
            solve_pnp(pts3[:5], pts2[:5], K)

    def test_wrong_pts3_shape_raises(self):
        K, pts3, pts2 = self._good_input()
        with pytest.raises(ValueError, match="points_3d must be"):
            solve_pnp(pts3.reshape(-1, 1), pts2, K)

    def test_wrong_pts2_shape_raises(self):
        K, pts3, pts2 = self._good_input()
        with pytest.raises(ValueError, match="points_2d must be"):
            solve_pnp(pts3, pts3, K)

    def test_mismatched_row_count_raises(self):
        K, pts3, pts2 = self._good_input()
        with pytest.raises(ValueError, match="same number of rows"):
            solve_pnp(pts3, pts2[:-1], K)

    def test_wrong_K_shape_raises(self):
        K, pts3, pts2 = self._good_input()
        with pytest.raises(ValueError, match="K must be 3×3"):
            solve_pnp(pts3, pts2, np.eye(2))

    def test_invalid_threshold_raises(self):
        K, pts3, pts2 = self._good_input()
        with pytest.raises(ValueError, match="inlier_threshold"):
            solve_pnp(pts3, pts2, K, inlier_threshold=0.0)

    def test_invalid_confidence_raises(self):
        K, pts3, pts2 = self._good_input()
        with pytest.raises(ValueError, match="confidence"):
            solve_pnp(pts3, pts2, K, confidence=2.0)

    def test_invalid_max_iterations_raises(self):
        K, pts3, pts2 = self._good_input()
        with pytest.raises(ValueError, match="max_ransac_iterations"):
            solve_pnp(pts3, pts2, K, max_ransac_iterations=0)


# ---------------------------------------------------------------------------
# Dataclass fields
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_essential_matrix_result_fields(self):
        E = np.eye(3)
        mask = np.array([True, False, True])
        r = EssentialMatrixResult(essential_matrix=E, inlier_mask=mask,
                                  num_inliers=2)
        np.testing.assert_array_equal(r.essential_matrix, E)
        np.testing.assert_array_equal(r.inlier_mask, mask)
        assert r.num_inliers == 2

    def test_pnp_result_fields(self):
        R = np.eye(3)
        t = np.zeros(3)
        mask = np.array([True, True])
        r = PnPResult(rotation=R, translation=t, inlier_mask=mask,
                      num_inliers=2, success=True)
        np.testing.assert_array_equal(r.rotation, R)
        np.testing.assert_array_equal(r.translation, t)
        assert r.success is True
        assert r.num_inliers == 2
