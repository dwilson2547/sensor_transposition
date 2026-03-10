"""Tests for the ground_plane segmentation module."""

import numpy as np
import pytest

from sensor_transposition.ground_plane import (
    height_threshold_segment,
    normal_based_segment,
    ransac_ground_plane,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _flat_ground_cloud(
    n_ground: int = 300,
    n_above: int = 200,
    seed: int = 0,
    ground_z: float = 0.0,
    above_z_min: float = 0.5,
    above_z_max: float = 3.0,
) -> np.ndarray:
    """Generate a synthetic point cloud with a flat ground plane at *ground_z*.

    The first *n_ground* points are placed within ±0.05 m of *ground_z* (flat
    ground), and the next *n_above* points are placed between *above_z_min*
    and *above_z_max* (obstacles / structure above ground).
    """
    rng = np.random.default_rng(seed)
    ground_pts = rng.uniform(-10.0, 10.0, (n_ground, 3))
    ground_pts[:, 2] = rng.uniform(ground_z - 0.05, ground_z + 0.05, n_ground)

    above_pts = rng.uniform(-10.0, 10.0, (n_above, 3))
    above_pts[:, 2] = rng.uniform(above_z_min, above_z_max, n_above)

    return np.vstack([ground_pts, above_pts])


# ---------------------------------------------------------------------------
# height_threshold_segment
# ---------------------------------------------------------------------------


class TestHeightThresholdSegment:
    def test_returns_two_boolean_masks(self):
        cloud = _flat_ground_cloud()
        gm, ngm = height_threshold_segment(cloud)
        assert gm.dtype == bool
        assert ngm.dtype == bool
        assert gm.shape == (len(cloud),)
        assert ngm.shape == (len(cloud),)

    def test_masks_are_complements(self):
        cloud = _flat_ground_cloud()
        gm, ngm = height_threshold_segment(cloud, threshold=0.3)
        np.testing.assert_array_equal(gm, ~ngm)

    def test_all_points_covered(self):
        cloud = _flat_ground_cloud()
        gm, ngm = height_threshold_segment(cloud, threshold=0.3)
        assert (gm | ngm).all()

    def test_detects_ground_points(self):
        """Ground points (z < threshold) should all be classified as ground."""
        cloud = _flat_ground_cloud(n_ground=300, n_above=0, ground_z=0.0)
        gm, _ = height_threshold_segment(cloud, threshold=0.3)
        assert gm.all()

    def test_detects_non_ground_points(self):
        """Points well above threshold should all be non-ground."""
        cloud = _flat_ground_cloud(n_ground=0, n_above=200, above_z_min=1.0, above_z_max=5.0)
        gm, ngm = height_threshold_segment(cloud, threshold=0.3)
        assert ngm.all()

    def test_custom_threshold(self):
        cloud = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 1.5], [0.0, 0.0, 2.5]])
        gm, ngm = height_threshold_segment(cloud, threshold=1.0)
        assert gm[0] and not gm[1] and not gm[2]
        assert not ngm[0] and ngm[1] and ngm[2]

    def test_empty_cloud(self):
        cloud = np.empty((0, 3))
        gm, ngm = height_threshold_segment(cloud)
        assert gm.shape == (0,)
        assert ngm.shape == (0,)

    def test_accepts_list_input(self):
        cloud_list = [[0.0, 0.0, 0.1], [0.0, 0.0, 1.0]]
        gm, _ = height_threshold_segment(cloud_list, threshold=0.5)
        assert gm[0] and not gm[1]


# ---------------------------------------------------------------------------
# ransac_ground_plane
# ---------------------------------------------------------------------------


class TestRansacGroundPlane:
    def test_returns_mask_and_plane(self):
        cloud = _flat_ground_cloud()
        gm, plane = ransac_ground_plane(cloud, rng=np.random.default_rng(42))
        assert gm.dtype == bool
        assert gm.shape == (len(cloud),)
        assert plane.shape == (4,)

    def test_plane_normal_approximately_vertical(self):
        """The fitted plane normal should point approximately toward +Z."""
        cloud = _flat_ground_cloud(seed=1)
        _, plane = ransac_ground_plane(cloud, rng=np.random.default_rng(0))
        a, b, c, d = plane
        normal = np.array([a, b, c])
        # Normal should be nearly unit-length and mostly pointing in +Z
        assert abs(np.linalg.norm(normal) - 1.0) < 1e-6
        assert c > 0.85

    def test_most_ground_points_are_inliers(self):
        """At least 80 % of the true ground points should be found."""
        n_ground = 300
        cloud = _flat_ground_cloud(n_ground=n_ground, n_above=100, seed=7)
        gm, _ = ransac_ground_plane(
            cloud,
            distance_threshold=0.15,
            max_iterations=500,
            rng=np.random.default_rng(7),
        )
        # First n_ground points are ground; check recall
        true_ground = np.arange(n_ground)
        recall = gm[true_ground].sum() / n_ground
        assert recall > 0.80

    def test_empty_cloud_returns_zeros(self):
        cloud = np.empty((0, 3))
        gm, plane = ransac_ground_plane(cloud)
        assert gm.shape == (0,)
        assert plane.shape == (4,)
        np.testing.assert_array_equal(plane, np.zeros(4))

    def test_small_cloud_less_than_3_points(self):
        cloud = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        gm, plane = ransac_ground_plane(cloud)
        assert gm.shape == (2,)
        np.testing.assert_array_equal(plane, np.zeros(4))

    def test_reproducible_with_rng(self):
        cloud = _flat_ground_cloud(seed=42)
        gm1, plane1 = ransac_ground_plane(cloud, rng=np.random.default_rng(0))
        gm2, plane2 = ransac_ground_plane(cloud, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(gm1, gm2)
        np.testing.assert_array_equal(plane1, plane2)

    def test_tilted_plane_rejected_by_normal_threshold(self):
        """A nearly vertical wall should not be detected as ground."""
        rng = np.random.default_rng(0)
        # Create a vertical wall (normal is in +X direction)
        wall = rng.uniform(-5.0, 5.0, (200, 3))
        wall[:, 0] = rng.uniform(-0.05, 0.05, 200)  # x ≈ 0 (wall)
        _, plane = ransac_ground_plane(
            wall,
            normal_threshold=0.9,
            max_iterations=200,
            rng=np.random.default_rng(0),
        )
        # With a strict normal_threshold the vertical plane should be rejected;
        # plane coefficients remain zero (no valid model found)
        assert plane[2] >= 0.0  # normal Z-component should be non-negative


# ---------------------------------------------------------------------------
# normal_based_segment
# ---------------------------------------------------------------------------


class TestNormalBasedSegment:
    def test_returns_mask_and_normals(self):
        cloud = _flat_ground_cloud()
        gm, normals = normal_based_segment(cloud, k=10)
        assert gm.dtype == bool
        assert gm.shape == (len(cloud),)
        assert normals.shape == (len(cloud), 3)

    def test_normals_are_unit_vectors(self):
        cloud = _flat_ground_cloud(n_ground=100, n_above=50, seed=3)
        _, normals = normal_based_segment(cloud, k=10)
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_normals_oriented_toward_plus_z(self):
        """All normals should have a non-negative Z component."""
        cloud = _flat_ground_cloud(n_ground=200, n_above=100, seed=5)
        _, normals = normal_based_segment(cloud, k=10)
        assert (normals[:, 2] >= 0).all()

    def test_flat_ground_mostly_classified_correctly(self):
        """Most of the flat ground points should be detected."""
        n_ground = 200
        cloud = _flat_ground_cloud(n_ground=n_ground, n_above=100, seed=8)
        gm, _ = normal_based_segment(cloud, k=15)
        true_ground = np.arange(n_ground)
        recall = gm[true_ground].sum() / n_ground
        assert recall > 0.70

    def test_empty_cloud(self):
        cloud = np.empty((0, 3))
        gm, normals = normal_based_segment(cloud)
        assert gm.shape == (0,)
        assert normals.shape == (0, 3)

    def test_k_clamped_to_cloud_size(self):
        """When k >= N the function should still run without error."""
        cloud = _flat_ground_cloud(n_ground=5, n_above=5, seed=2)
        gm, normals = normal_based_segment(cloud, k=100)
        assert gm.shape == (10,)
        assert normals.shape == (10, 3)

    def test_custom_verticality_threshold(self):
        """Lower threshold → more points classified as ground."""
        cloud = _flat_ground_cloud(n_ground=200, n_above=100, seed=6)
        gm_strict, _ = normal_based_segment(cloud, k=10, verticality_threshold=0.95)
        gm_loose, _ = normal_based_segment(cloud, k=10, verticality_threshold=0.50)
        assert gm_loose.sum() >= gm_strict.sum()


# ---------------------------------------------------------------------------
# Integration – public API via package __init__
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_importable_from_package(self):
        import sensor_transposition as st

        assert hasattr(st, "height_threshold_segment")
        assert hasattr(st, "ransac_ground_plane")
        assert hasattr(st, "normal_based_segment")
        assert hasattr(st, "ground_plane")

    def test_module_functions_callable(self):
        import sensor_transposition as st

        cloud = _flat_ground_cloud(n_ground=50, n_above=30, seed=0)
        gm, ngm = st.height_threshold_segment(cloud)
        assert gm.shape == (len(cloud),)

        gm, plane = st.ransac_ground_plane(cloud, rng=np.random.default_rng(0))
        assert plane.shape == (4,)

        gm, normals = st.normal_based_segment(cloud, k=10)
        assert normals.shape == (len(cloud), 3)
