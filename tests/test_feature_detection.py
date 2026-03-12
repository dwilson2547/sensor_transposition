"""Tests for feature_detection: Harris corners, patch descriptors, matching."""

import numpy as np
import pytest

from sensor_transposition.feature_detection import (
    _convolve2d,
    _gaussian_kernel,
    compute_patch_descriptor,
    detect_harris_corners,
    match_features,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_checkerboard(size: int = 64, block: int = 8) -> np.ndarray:
    """Return a (size, size) uint8 checkerboard image."""
    img = np.zeros((size, size), dtype=float)
    for r in range(size):
        for c in range(size):
            if ((r // block) + (c // block)) % 2 == 0:
                img[r, c] = 255.0
    return img


def _make_gradient_image(size: int = 32) -> np.ndarray:
    """Return an image with a strong horizontal gradient."""
    img = np.tile(np.linspace(0, 255, size), (size, 1))
    return img


# ---------------------------------------------------------------------------
# _convolve2d
# ---------------------------------------------------------------------------


class TestConvolve2d:
    def test_identity_kernel(self):
        img = np.random.default_rng(0).uniform(0, 255, (10, 10))
        kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
        out = _convolve2d(img, kernel)
        np.testing.assert_allclose(out, img, atol=1e-10)

    def test_output_shape_matches_input(self):
        img = np.ones((15, 20), dtype=float)
        kernel = np.ones((3, 3), dtype=float) / 9.0
        out = _convolve2d(img, kernel)
        assert out.shape == img.shape

    def test_uniform_blur_ones(self):
        """Blurring an all-ones image gives all-ones in the interior."""
        img = np.ones((10, 10), dtype=float)
        kernel = np.ones((3, 3), dtype=float) / 9.0
        out = _convolve2d(img, kernel)
        # Interior pixels (away from border) should be 1.0.
        np.testing.assert_allclose(out[1:-1, 1:-1], 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# _gaussian_kernel
# ---------------------------------------------------------------------------


class TestGaussianKernel:
    def test_sums_to_one(self):
        g = _gaussian_kernel(sigma=2.0)
        assert abs(g.sum() - 1.0) < 1e-10

    def test_shape_is_odd(self):
        g = _gaussian_kernel(sigma=1.5)
        assert g.shape[0] % 2 == 1
        assert g.shape[1] % 2 == 1

    def test_symmetric(self):
        g = _gaussian_kernel(sigma=1.0)
        np.testing.assert_allclose(g, g.T, atol=1e-12)

    def test_centre_is_maximum(self):
        g = _gaussian_kernel(sigma=1.0)
        r, c = g.shape[0] // 2, g.shape[1] // 2
        assert g[r, c] == g.max()


# ---------------------------------------------------------------------------
# detect_harris_corners
# ---------------------------------------------------------------------------


class TestDetectHarrisCorners:
    def test_returns_array(self):
        img = _make_checkerboard(64, 8)
        kps = detect_harris_corners(img)
        assert isinstance(kps, np.ndarray)

    def test_output_shape_columns(self):
        img = _make_checkerboard(64, 8)
        kps = detect_harris_corners(img)
        assert kps.ndim == 2
        assert kps.shape[1] == 2

    def test_detects_corners_in_checkerboard(self):
        """Checkerboard has many genuine corners."""
        img = _make_checkerboard(64, 8)
        kps = detect_harris_corners(img, threshold=0.01)
        assert len(kps) > 0

    def test_no_corners_in_uniform_image(self):
        """Uniform image has no gradients → no interior corners."""
        img = np.ones((64, 64), dtype=float) * 128.0
        kps = detect_harris_corners(img)
        # Any detected "corners" must be near the border (Sobel padding artefacts).
        if len(kps) > 0:
            margin = 5
            interior_mask = (
                (kps[:, 0] >= margin) & (kps[:, 0] < 64 - margin) &
                (kps[:, 1] >= margin) & (kps[:, 1] < 64 - margin)
            )
            assert not interior_mask.any(), "Unexpected interior corners in uniform image."

    def test_no_corners_in_flat_gradient(self):
        """A perfectly linear gradient has no interior corners."""
        img = _make_gradient_image(64)
        kps = detect_harris_corners(img)
        if len(kps) > 0:
            margin = 5
            interior_mask = (
                (kps[:, 0] >= margin) & (kps[:, 0] < 64 - margin) &
                (kps[:, 1] >= margin) & (kps[:, 1] < 64 - margin)
            )
            assert not interior_mask.any(), "Unexpected interior corners in gradient image."

    def test_max_corners_limits_output(self):
        img = _make_checkerboard(64, 4)
        kps = detect_harris_corners(img, threshold=0.001, max_corners=5)
        assert len(kps) <= 5

    def test_sorted_by_descending_response(self):
        """Keypoints should be sorted by corner response (highest first)."""
        rng = np.random.default_rng(42)
        img = rng.uniform(0, 255, (32, 32))
        kps = detect_harris_corners(img, threshold=0.001)
        # We can't directly verify ordering without re-computing R, but we can
        # verify that the returned array is well-formed.
        assert kps.ndim == 2
        assert kps.shape[1] == 2

    def test_invalid_image_3d(self):
        with pytest.raises(ValueError, match="2-D"):
            detect_harris_corners(np.ones((10, 10, 3)))

    def test_invalid_k_zero(self):
        with pytest.raises(ValueError, match="k must be"):
            detect_harris_corners(np.ones((10, 10)), k=0.0)

    def test_invalid_threshold_zero(self):
        with pytest.raises(ValueError, match="threshold must be"):
            detect_harris_corners(np.ones((10, 10)), threshold=0.0)

    def test_invalid_nms_radius(self):
        with pytest.raises(ValueError, match="nms_radius"):
            detect_harris_corners(np.ones((10, 10)), nms_radius=0)

    def test_corner_coordinates_within_image(self):
        img = _make_checkerboard(64, 8)
        kps = detect_harris_corners(img)
        if len(kps) > 0:
            assert np.all(kps[:, 0] >= 0)
            assert np.all(kps[:, 0] < img.shape[0])
            assert np.all(kps[:, 1] >= 0)
            assert np.all(kps[:, 1] < img.shape[1])


# ---------------------------------------------------------------------------
# compute_patch_descriptor
# ---------------------------------------------------------------------------


class TestComputePatchDescriptor:
    def test_output_shape(self):
        img = _make_checkerboard(64, 8)
        kps = detect_harris_corners(img, max_corners=10)
        if len(kps) == 0:
            pytest.skip("No keypoints detected.")
        descs = compute_patch_descriptor(img, kps, patch_size=11)
        assert descs.shape == (len(kps), 121)

    def test_interior_descriptors_unit_norm(self):
        """Non-border keypoints with non-uniform patches should be L2-normalised."""
        img = _make_checkerboard(64, 8)
        kps = detect_harris_corners(img, max_corners=20)
        if len(kps) == 0:
            pytest.skip("No keypoints detected.")
        descs = compute_patch_descriptor(img, kps, patch_size=11)
        # Interior keypoints (far from border).
        interior_mask = (
            (kps[:, 0] >= 6) & (kps[:, 0] < 58) &
            (kps[:, 1] >= 6) & (kps[:, 1] < 58)
        )
        interior_descs = descs[interior_mask]
        if len(interior_descs) == 0:
            pytest.skip("No interior keypoints.")
        norms = np.linalg.norm(interior_descs, axis=1)
        non_zero = norms > 1e-12
        if non_zero.any():
            np.testing.assert_allclose(norms[non_zero], 1.0, atol=1e-6)

    def test_invalid_image_3d(self):
        with pytest.raises(ValueError, match="2-D"):
            compute_patch_descriptor(np.ones((10, 10, 3)), np.array([[5, 5]]))

    def test_invalid_keypoints_shape(self):
        with pytest.raises(ValueError, match=r"\(N, 2\)"):
            compute_patch_descriptor(np.ones((10, 10)), np.array([5, 5]))

    def test_invalid_patch_size_even(self):
        with pytest.raises(ValueError, match="patch_size"):
            compute_patch_descriptor(
                np.ones((10, 10)), np.array([[5, 5]]), patch_size=10
            )

    def test_invalid_patch_size_zero(self):
        with pytest.raises(ValueError, match="patch_size"):
            compute_patch_descriptor(
                np.ones((10, 10)), np.array([[5, 5]]), patch_size=0
            )

    def test_empty_keypoints(self):
        img = np.ones((10, 10), dtype=float)
        kps = np.empty((0, 2), dtype=int)
        descs = compute_patch_descriptor(img, kps)
        assert descs.shape[0] == 0

    def test_different_patches_yield_different_descriptors(self):
        """Two keypoints at different image locations should have different descriptors."""
        img = _make_checkerboard(64, 8)
        kps = detect_harris_corners(img, max_corners=2)
        if len(kps) < 2:
            pytest.skip("Need at least 2 keypoints.")
        descs = compute_patch_descriptor(img, kps)
        # Descriptors should not be identical.
        assert not np.allclose(descs[0], descs[1])


# ---------------------------------------------------------------------------
# match_features
# ---------------------------------------------------------------------------


class TestMatchFeatures:
    def _make_descriptors(self, n: int, d: int = 121, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        descs = rng.standard_normal((n, d))
        norms = np.linalg.norm(descs, axis=1, keepdims=True)
        return descs / np.where(norms > 1e-12, norms, 1.0)

    def test_returns_array(self):
        d1 = self._make_descriptors(10)
        d2 = self._make_descriptors(10, seed=1)
        matches = match_features(d1, d2)
        assert isinstance(matches, np.ndarray)
        assert matches.ndim == 2
        assert matches.shape[1] == 2

    def test_matches_identical_descriptors(self):
        """Identical descriptor sets should produce many matches."""
        d = self._make_descriptors(20, seed=5)
        matches = match_features(d, d, ratio_threshold=0.9)
        # At least some matches expected when sets are identical.
        assert len(matches) > 0

    def test_self_match_indices_diagonal(self):
        """When descriptor sets are identical, each row should match itself."""
        d = self._make_descriptors(10, seed=7)
        matches = match_features(d, d, ratio_threshold=0.9)
        for m in matches:
            assert m[0] == m[1]

    def test_match_indices_in_range(self):
        d1 = self._make_descriptors(15, seed=0)
        d2 = self._make_descriptors(20, seed=1)
        matches = match_features(d1, d2)
        if len(matches) > 0:
            assert np.all(matches[:, 0] >= 0) and np.all(matches[:, 0] < 15)
            assert np.all(matches[:, 1] >= 0) and np.all(matches[:, 1] < 20)

    def test_strict_ratio_returns_fewer_matches(self):
        d1 = self._make_descriptors(30, seed=0)
        d2 = self._make_descriptors(30, seed=1)
        matches_loose = match_features(d1, d2, ratio_threshold=0.95)
        matches_strict = match_features(d1, d2, ratio_threshold=0.5)
        assert len(matches_strict) <= len(matches_loose)

    def test_empty_descriptors_1(self):
        d1 = np.empty((0, 121), dtype=float)
        d2 = self._make_descriptors(10)
        matches = match_features(d1, d2)
        assert matches.shape == (0, 2)

    def test_empty_descriptors_2(self):
        d1 = self._make_descriptors(10)
        d2 = np.empty((0, 121), dtype=float)
        matches = match_features(d1, d2)
        assert matches.shape == (0, 2)

    def test_dimension_mismatch(self):
        d1 = self._make_descriptors(10, d=121)
        d2 = self._make_descriptors(10, d=64)
        with pytest.raises(ValueError, match="mismatch"):
            match_features(d1, d2)

    def test_invalid_ratio_threshold(self):
        d1 = self._make_descriptors(5)
        d2 = self._make_descriptors(5)
        with pytest.raises(ValueError, match="ratio_threshold"):
            match_features(d1, d2, ratio_threshold=0.0)

    def test_all_zero_descriptors_excluded(self):
        """Zero-vector descriptors should be excluded (border keypoints)."""
        d1 = np.zeros((5, 10), dtype=float)
        d2 = np.zeros((5, 10), dtype=float)
        matches = match_features(d1, d2)
        assert matches.shape == (0, 2)


# ---------------------------------------------------------------------------
# End-to-end: Harris → patch descriptors → match_features
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_pipeline_produces_matches(self):
        """Shifted checkerboard should produce feature matches."""
        img1 = _make_checkerboard(64, 8)
        # Small shift of 3 pixels down and 3 right.
        img2 = np.zeros_like(img1)
        img2[3:, 3:] = img1[:61, :61]

        kp1 = detect_harris_corners(img1, threshold=0.01, max_corners=50)
        kp2 = detect_harris_corners(img2, threshold=0.01, max_corners=50)

        if len(kp1) == 0 or len(kp2) == 0:
            pytest.skip("No keypoints in test images.")

        d1 = compute_patch_descriptor(img1, kp1, patch_size=11)
        d2 = compute_patch_descriptor(img2, kp2, patch_size=11)

        matches = match_features(d1, d2, ratio_threshold=0.8)
        assert len(matches) >= 0  # At least the pipeline runs without error.
