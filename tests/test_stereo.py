"""Tests for stereo.py: stereo_rectify, compute_disparity_sgbm, triangulate_stereo."""

import numpy as np
import pytest

from sensor_transposition.stereo import (
    compute_disparity_sgbm,
    stereo_rectify,
    triangulate_stereo,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_K(f: float = 500.0, cx: float = 320.0, cy: float = 240.0) -> np.ndarray:
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])


def _make_stereo_rig() -> tuple:
    """Return (K1, K2, R, t) for a typical horizontal stereo rig."""
    K = _make_K()
    R = np.eye(3)
    t = np.array([-0.12, 0.0, 0.0])  # 12 cm baseline to the right
    return K, K, R, t


# ---------------------------------------------------------------------------
# stereo_rectify
# ---------------------------------------------------------------------------


class TestStereoRectify:
    def test_returns_four_matrices(self):
        K1, K2, R, t = _make_stereo_rig()
        result = stereo_rectify(K1, (), K2, (), R, t, image_size=(480, 640))
        assert len(result) == 4

    def test_R1_R2_are_3x3(self):
        K1, K2, R, t = _make_stereo_rig()
        R1, R2, P1, P2 = stereo_rectify(K1, (), K2, (), R, t, image_size=(480, 640))
        assert R1.shape == (3, 3)
        assert R2.shape == (3, 3)

    def test_P1_P2_are_3x4(self):
        K1, K2, R, t = _make_stereo_rig()
        R1, R2, P1, P2 = stereo_rectify(K1, (), K2, (), R, t, image_size=(480, 640))
        assert P1.shape == (3, 4)
        assert P2.shape == (3, 4)

    def test_R1_R2_are_rotation_matrices(self):
        """Rectification rotations must be orthogonal with determinant +1."""
        K1, K2, R, t = _make_stereo_rig()
        R1, R2, P1, P2 = stereo_rectify(K1, (), K2, (), R, t, image_size=(480, 640))
        np.testing.assert_allclose(R1 @ R1.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(R2 @ R2.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R1) - 1.0) < 1e-10
        assert abs(np.linalg.det(R2) - 1.0) < 1e-10

    def test_P1_has_zero_tx(self):
        """P1 should have no x-translation (left camera at origin)."""
        K1, K2, R, t = _make_stereo_rig()
        R1, R2, P1, P2 = stereo_rectify(K1, (), K2, (), R, t, image_size=(480, 640))
        assert abs(P1[0, 3]) < 1e-10

    def test_P2_has_nonzero_tx(self):
        """P2 should encode the baseline as a non-zero x-offset."""
        K1, K2, R, t = _make_stereo_rig()
        R1, R2, P1, P2 = stereo_rectify(K1, (), K2, (), R, t, image_size=(480, 640))
        assert abs(P2[0, 3]) > 1e-3

    def test_P1_P2_share_focal_and_cy(self):
        """After rectification, both projection matrices share f and cy."""
        K1, K2, R, t = _make_stereo_rig()
        R1, R2, P1, P2 = stereo_rectify(K1, (), K2, (), R, t, image_size=(480, 640))
        assert abs(P1[0, 0] - P2[0, 0]) < 1e-6  # fx
        assert abs(P1[1, 1] - P2[1, 1]) < 1e-6  # fy
        assert abs(P1[1, 2] - P2[1, 2]) < 1e-6  # cy

    def test_identity_R_gives_valid_result(self):
        """Cameras already co-planar (R = I) should produce valid output."""
        K = _make_K()
        R = np.eye(3)
        t = np.array([-0.1, 0.0, 0.0])
        R1, R2, P1, P2 = stereo_rectify(K, (), K, (), R, t, image_size=(480, 640))
        assert R1.shape == (3, 3)
        assert P1.shape == (3, 4)

    def test_invalid_K1_shape(self):
        K1 = np.eye(4)
        K2 = _make_K()
        with pytest.raises(ValueError, match="K1"):
            stereo_rectify(K1, (), K2, (), np.eye(3), np.array([0.1, 0, 0]),
                           image_size=(480, 640))

    def test_invalid_K2_shape(self):
        K1 = _make_K()
        K2 = np.eye(4)
        with pytest.raises(ValueError, match="K2"):
            stereo_rectify(K1, (), K2, (), np.eye(3), np.array([0.1, 0, 0]),
                           image_size=(480, 640))

    def test_invalid_R_shape(self):
        K = _make_K()
        with pytest.raises(ValueError, match="R must be"):
            stereo_rectify(K, (), K, (), np.eye(4), np.array([0.1, 0, 0]),
                           image_size=(480, 640))

    def test_invalid_t_shape(self):
        K = _make_K()
        with pytest.raises(ValueError, match="t must be"):
            stereo_rectify(K, (), K, (), np.eye(3), np.array([0.1, 0, 0, 1.0]),
                           image_size=(480, 640))

    def test_zero_baseline_raises(self):
        K = _make_K()
        with pytest.raises(ValueError, match="near-zero"):
            stereo_rectify(K, (), K, (), np.eye(3), np.array([0.0, 0.0, 0.0]),
                           image_size=(480, 640))

    def test_invalid_image_size(self):
        K = _make_K()
        with pytest.raises(ValueError, match="image_size"):
            stereo_rectify(K, (), K, (), np.eye(3), np.array([0.1, 0, 0]),
                           image_size=(-1, 640))


# ---------------------------------------------------------------------------
# compute_disparity_sgbm
# ---------------------------------------------------------------------------


def _make_stereo_pair(H: int = 32, W: int = 64, d_true: int = 8):
    """Create a synthetic rectified stereo pair with known disparity *d_true*."""
    rng = np.random.default_rng(0)
    left = rng.uniform(0, 255, (H, W))
    # Right image is left shifted by d_true pixels.
    right = np.zeros_like(left)
    right[:, :W - d_true] = left[:, d_true:]
    return left, right, d_true


class TestComputeDisparitysgbm:
    def test_output_shape(self):
        left, right, _ = _make_stereo_pair(32, 64)
        disp = compute_disparity_sgbm(left, right, block_size=5, num_disparities=16)
        assert disp.shape == left.shape

    def test_known_disparity(self):
        """Interior pixels should recover the known disparity."""
        d_true = 8
        left, right, _ = _make_stereo_pair(H=40, W=80, d_true=d_true)
        disp = compute_disparity_sgbm(left, right, block_size=5, num_disparities=16,
                                      min_disparity=0)
        # Check a region well away from borders.
        interior = disp[10:30, d_true + 5: 70]
        assert np.median(interior[interior > 0]) == pytest.approx(float(d_true), abs=1.0)

    def test_output_nonnegative(self):
        left, right, _ = _make_stereo_pair()
        disp = compute_disparity_sgbm(left, right, block_size=5, num_disparities=16)
        assert np.all(disp >= 0)

    def test_invalid_left_shape(self):
        with pytest.raises(ValueError, match="img_left"):
            compute_disparity_sgbm(np.ones((10, 10, 3)), np.ones((10, 10)))

    def test_invalid_right_shape(self):
        with pytest.raises(ValueError, match="img_right"):
            compute_disparity_sgbm(np.ones((10, 10)), np.ones((10, 10, 3)))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            compute_disparity_sgbm(np.ones((10, 10)), np.ones((10, 12)))

    def test_invalid_block_size_even(self):
        with pytest.raises(ValueError, match="block_size"):
            compute_disparity_sgbm(np.ones((20, 20)), np.ones((20, 20)),
                                   block_size=10)

    def test_invalid_num_disparities(self):
        with pytest.raises(ValueError, match="num_disparities"):
            compute_disparity_sgbm(np.ones((20, 20)), np.ones((20, 20)),
                                   num_disparities=0)


# ---------------------------------------------------------------------------
# triangulate_stereo
# ---------------------------------------------------------------------------


class TestTriangulateStereo:
    def test_point_pair_mode_basic(self):
        K = _make_K(f=500.0, cx=320.0, cy=240.0)
        baseline = 0.12
        # A point at (X=0, Y=0, Z=5) projects to (320, 240) in left.
        # In right: u_r = u_l - f * baseline / Z = 320 - 500*0.12/5 = 308
        pts_left = np.array([[320.0, 240.0]])
        pts_right = np.array([[308.0, 240.0]])
        pts3d = triangulate_stereo(pts_left, pts_right, K=K, baseline=baseline)
        assert pts3d.shape == (1, 3)
        np.testing.assert_allclose(pts3d[0, 2], 5.0, rtol=1e-4)
        np.testing.assert_allclose(pts3d[0, 0], 0.0, atol=1e-3)
        np.testing.assert_allclose(pts3d[0, 1], 0.0, atol=1e-3)

    def test_point_pair_multiple_points(self):
        K = _make_K(f=500.0)
        baseline = 0.10
        Z_vals = np.array([3.0, 5.0, 8.0])
        disp_vals = 500.0 * baseline / Z_vals
        pts_left = np.column_stack([np.full(3, 320.0), np.full(3, 240.0)])
        pts_right = np.column_stack([pts_left[:, 0] - disp_vals, pts_left[:, 1]])
        pts3d = triangulate_stereo(pts_left, pts_right, K=K, baseline=baseline)
        np.testing.assert_allclose(pts3d[:, 2], Z_vals, rtol=1e-4)

    def test_zero_disparity_gives_nan(self):
        """Points with ul == ur (zero disparity) should give NaN depth."""
        K = _make_K()
        pts_left = np.array([[320.0, 240.0]])
        pts_right = np.array([[320.0, 240.0]])  # same → zero disparity
        pts3d = triangulate_stereo(pts_left, pts_right, K=K, baseline=0.1)
        assert np.isnan(pts3d[0, 2])

    def test_disparity_map_mode(self):
        K = _make_K(f=500.0, cx=32.0, cy=24.0)
        baseline = 0.12
        H, W = 48, 64
        Z_true = 5.0
        disp_arr = np.zeros((H, W), dtype=float)
        disp_arr[24, 32] = 500.0 * 0.12 / Z_true  # 12.0
        pts3d = triangulate_stereo(K=K, baseline=baseline, disp=disp_arr)
        assert pts3d.shape[1] == 3
        assert len(pts3d) >= 1
        # Find the point closest to column 32 → X ≈ 0
        np.testing.assert_allclose(pts3d[0, 2], Z_true, rtol=1e-4)

    def test_disparity_map_mode_no_valid_pixels(self):
        K = _make_K()
        disp_arr = np.zeros((10, 10), dtype=float)  # all zero
        pts3d = triangulate_stereo(K=K, baseline=0.1, disp=disp_arr)
        assert pts3d.shape[0] == 0
        assert pts3d.shape[1] == 3

    def test_invalid_both_modes(self):
        K = _make_K()
        pts = np.array([[320.0, 240.0]])
        disp = np.zeros((10, 10))
        with pytest.raises(ValueError, match="not both"):
            triangulate_stereo(pts, pts, K=K, baseline=0.1, disp=disp)

    def test_invalid_neither_mode(self):
        K = _make_K()
        with pytest.raises(ValueError, match="Provide either"):
            triangulate_stereo(K=K, baseline=0.1)

    def test_invalid_K_shape(self):
        with pytest.raises(ValueError, match="K must be"):
            triangulate_stereo(np.array([[320.0, 240.0]]),
                               np.array([[310.0, 240.0]]),
                               K=np.eye(4), baseline=0.1)

    def test_invalid_baseline(self):
        K = _make_K()
        with pytest.raises(ValueError, match="baseline must be"):
            triangulate_stereo(np.array([[320.0, 240.0]]),
                               np.array([[310.0, 240.0]]),
                               K=K, baseline=0.0)

    def test_pts_shape_mismatch(self):
        K = _make_K()
        with pytest.raises(ValueError, match="same length"):
            triangulate_stereo(np.array([[320.0, 240.0], [310.0, 230.0]]),
                               np.array([[310.0, 240.0]]),
                               K=K, baseline=0.1)

    def test_invalid_pts_left_shape(self):
        K = _make_K()
        with pytest.raises(ValueError, match="pts_left"):
            triangulate_stereo(np.array([320.0, 240.0]),
                               np.array([[310.0, 240.0]]),
                               K=K, baseline=0.1)
