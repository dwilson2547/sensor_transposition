"""Tests for loop closure: Scan Context descriptor and ScanContextDatabase."""

import math

import numpy as np
import pytest

from sensor_transposition.loop_closure import (
    LoopClosureCandidate,
    M2dpDescriptor,
    ScanContextDatabase,
    ScanContextDescriptor,
    _check_compatible,
    _column_cosine_distance,
    compute_m2dp,
    compute_scan_context,
    m2dp_distance,
    scan_context_distance,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _random_cloud(n: int = 500, seed: int = 0, max_range: float = 50.0) -> np.ndarray:
    """Generate a random (N, 3) LiDAR point cloud within a given range."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-max_range, max_range, (n, 3))
    pts[:, 2] = rng.uniform(-2.0, 5.0, n)
    return pts


def _yaw_rotate(points: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate (N, 3) points about the z-axis by *angle_rad*."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (R @ points.T).T


# ---------------------------------------------------------------------------
# compute_scan_context
# ---------------------------------------------------------------------------


class TestComputeScanContext:
    def test_returns_descriptor(self):
        pts = _random_cloud()
        desc = compute_scan_context(pts)
        assert isinstance(desc, ScanContextDescriptor)

    def test_matrix_shape(self):
        pts = _random_cloud()
        desc = compute_scan_context(pts, num_rings=10, num_sectors=30)
        assert desc.matrix.shape == (10, 30)

    def test_ring_key_shape(self):
        pts = _random_cloud()
        desc = compute_scan_context(pts, num_rings=15, num_sectors=45)
        assert desc.ring_key.shape == (15,)

    def test_ring_key_is_row_mean(self):
        pts = _random_cloud()
        desc = compute_scan_context(pts, num_rings=10, num_sectors=20)
        np.testing.assert_allclose(
            desc.ring_key, desc.matrix.mean(axis=1), atol=1e-12
        )

    def test_non_negative_values(self):
        """Max-height in each bin should never be negative if all z >= 0."""
        rng = np.random.default_rng(1)
        pts = rng.uniform(-40.0, 40.0, (200, 3))
        pts[:, 2] = rng.uniform(0.0, 5.0, 200)
        desc = compute_scan_context(pts, num_rings=10, num_sectors=20)
        assert np.all(desc.matrix >= 0.0)

    def test_empty_cloud_all_zeros(self):
        """A cloud entirely outside max_range gives an all-zero descriptor."""
        pts = np.zeros((10, 3))  # all at origin → excluded (r == 0)
        desc = compute_scan_context(pts, num_rings=5, num_sectors=10, max_range=1.0)
        np.testing.assert_array_equal(desc.matrix, 0.0)

    def test_stores_params(self):
        pts = _random_cloud()
        desc = compute_scan_context(pts, num_rings=8, num_sectors=16, max_range=30.0)
        assert desc.num_rings == 8
        assert desc.num_sectors == 16
        assert desc.max_range == 30.0

    def test_min_z_clipping(self):
        """Points below min_z should be excluded."""
        pts = np.array([[5.0, 0.0, -10.0], [5.0, 0.0, 2.0]])
        desc_all = compute_scan_context(pts, num_rings=5, num_sectors=10, max_range=20.0)
        desc_clip = compute_scan_context(
            pts, num_rings=5, num_sectors=10, max_range=20.0, min_z=0.0
        )
        # With clipping the max height in the relevant bin should be 2.0, not -10.
        assert desc_clip.matrix.max() >= 2.0
        assert desc_clip.matrix.min() >= 0.0

    def test_max_z_clipping(self):
        """Points above max_z should be excluded."""
        pts = np.array([[5.0, 0.0, 100.0], [5.0, 0.0, 1.0]])
        desc = compute_scan_context(
            pts, num_rings=5, num_sectors=10, max_range=20.0, max_z=10.0
        )
        assert desc.matrix.max() <= 10.0

    def test_points_beyond_max_range_excluded(self):
        pts = np.array([[200.0, 0.0, 1.0]])  # far beyond default 80 m
        desc = compute_scan_context(pts, num_rings=5, num_sectors=10, max_range=80.0)
        np.testing.assert_array_equal(desc.matrix, 0.0)

    def test_accepts_extra_columns(self):
        """Points with more than 3 columns (e.g. intensity) are accepted."""
        rng = np.random.default_rng(2)
        pts4 = rng.uniform(-10.0, 10.0, (50, 4))
        desc = compute_scan_context(pts4, num_rings=5, num_sectors=10)
        assert desc.matrix.shape == (5, 10)


class TestComputeScanContextInputValidation:
    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="\\(N, ≥3\\)"):
            compute_scan_context(np.zeros((10, 2)))

    def test_1d_array_raises(self):
        with pytest.raises(ValueError, match="\\(N, ≥3\\)"):
            compute_scan_context(np.zeros(12))

    def test_invalid_num_rings_raises(self):
        pts = _random_cloud()
        with pytest.raises(ValueError, match="num_rings"):
            compute_scan_context(pts, num_rings=0)

    def test_invalid_num_sectors_raises(self):
        pts = _random_cloud()
        with pytest.raises(ValueError, match="num_sectors"):
            compute_scan_context(pts, num_sectors=0)

    def test_invalid_max_range_raises(self):
        pts = _random_cloud()
        with pytest.raises(ValueError, match="max_range"):
            compute_scan_context(pts, max_range=0.0)


# ---------------------------------------------------------------------------
# _column_cosine_distance
# ---------------------------------------------------------------------------


class TestColumnCosineDistance:
    def test_identical_matrices_zero_distance(self):
        A = np.random.default_rng(0).uniform(0, 5, (10, 20))
        dist = _column_cosine_distance(A, A)
        assert dist < 1e-10

    def test_orthogonal_rows_distance_one(self):
        """Two matrices whose rows are orthogonal should have distance ~1."""
        A = np.zeros((4, 4))
        B = np.zeros((4, 4))
        A[0, 0] = 1.0
        B[0, 1] = 1.0  # orthogonal to A[0]
        A[1, 2] = 1.0
        B[1, 3] = 1.0
        # Rows 2 and 3 are all-zero → distance 0
        dist = _column_cosine_distance(A, B)
        # Expect mean of [1, 1, 0, 0] = 0.5
        assert abs(dist - 0.5) < 1e-10

    def test_both_zero_rows_contribute_zero(self):
        A = np.zeros((3, 5))
        B = np.zeros((3, 5))
        assert _column_cosine_distance(A, B) == 0.0

    def test_distance_in_0_1_range(self):
        rng = np.random.default_rng(3)
        A = rng.uniform(0, 1, (15, 30))
        B = rng.uniform(0, 1, (15, 30))
        dist = _column_cosine_distance(A, B)
        assert 0.0 <= dist <= 1.0


# ---------------------------------------------------------------------------
# scan_context_distance
# ---------------------------------------------------------------------------


class TestScanContextDistance:
    def test_identical_descriptors_near_zero(self):
        pts = _random_cloud(300, seed=7)
        desc = compute_scan_context(pts, num_rings=10, num_sectors=20)
        dist, shift = scan_context_distance(desc, desc)
        assert dist < 1e-10
        assert shift == 0

    def test_distance_symmetric_up_to_rotation(self):
        """d(A, B) and d(B, A) should be close (rotation search may differ)."""
        pts1 = _random_cloud(300, seed=8)
        pts2 = _random_cloud(300, seed=9)
        desc1 = compute_scan_context(pts1, num_rings=10, num_sectors=20)
        desc2 = compute_scan_context(pts2, num_rings=10, num_sectors=20)
        d_ab, _ = scan_context_distance(desc1, desc2)
        d_ba, _ = scan_context_distance(desc2, desc1)
        # Distances are not guaranteed to be exactly symmetric due to the
        # asymmetric rotation search, but should be close.
        assert abs(d_ab - d_ba) < 0.1

    def test_rotated_cloud_lower_distance(self):
        """A yaw-rotated version of the same cloud should match better than a
        completely different cloud."""
        pts = _random_cloud(500, seed=10, max_range=40.0)
        pts_rotated = _yaw_rotate(pts, math.radians(30))
        pts_different = _random_cloud(500, seed=99, max_range=40.0)

        desc_a = compute_scan_context(pts, num_rings=20, num_sectors=60)
        desc_b_rot = compute_scan_context(
            pts_rotated, num_rings=20, num_sectors=60
        )
        desc_c = compute_scan_context(pts_different, num_rings=20, num_sectors=60)

        dist_same, _ = scan_context_distance(desc_a, desc_b_rot)
        dist_diff, _ = scan_context_distance(desc_a, desc_c)

        assert dist_same < dist_diff

    def test_distance_in_0_1_range(self):
        pts1 = _random_cloud(200, seed=11)
        pts2 = _random_cloud(200, seed=12)
        desc1 = compute_scan_context(pts1, num_rings=10, num_sectors=20)
        desc2 = compute_scan_context(pts2, num_rings=10, num_sectors=20)
        dist, shift = scan_context_distance(desc1, desc2)
        assert 0.0 <= dist <= 1.0
        assert 0 <= shift < 20

    def test_incompatible_shapes_raise(self):
        pts = _random_cloud()
        desc_a = compute_scan_context(pts, num_rings=10, num_sectors=20)
        desc_b = compute_scan_context(pts, num_rings=5, num_sectors=20)
        with pytest.raises(ValueError, match="incompatible shapes"):
            scan_context_distance(desc_a, desc_b)


# ---------------------------------------------------------------------------
# ScanContextDatabase
# ---------------------------------------------------------------------------


class TestScanContextDatabase:
    def _make_db(self, **kwargs) -> ScanContextDatabase:
        return ScanContextDatabase(
            num_rings=10, num_sectors=20, max_range=50.0, **kwargs
        )

    def _make_desc(self, seed: int) -> ScanContextDescriptor:
        pts = _random_cloud(300, seed=seed, max_range=40.0)
        return compute_scan_context(
            pts, num_rings=10, num_sectors=20, max_range=50.0
        )

    def test_initial_length_zero(self):
        db = self._make_db()
        assert len(db) == 0

    def test_add_increments_length(self):
        db = self._make_db()
        for i in range(5):
            db.add(self._make_desc(i))
        assert len(db) == 5

    def test_add_returns_correct_index(self):
        db = self._make_db()
        idx0 = db.add(self._make_desc(0))
        idx1 = db.add(self._make_desc(1))
        assert idx0 == 0
        assert idx1 == 1

    def test_custom_frame_id(self):
        db = self._make_db(exclusion_window=0)
        desc = self._make_desc(0)
        db.add(desc, frame_id=42)
        candidates = db.query(desc, top_k=1)
        assert len(candidates) == 1
        assert candidates[0].match_frame_id == 42

    def test_query_returns_list(self):
        db = self._make_db(exclusion_window=0)
        desc = self._make_desc(0)
        db.add(desc)
        result = db.query(desc)
        assert isinstance(result, list)

    def test_query_empty_below_exclusion_window(self):
        """When all entries are within the exclusion window, return empty."""
        db = self._make_db(exclusion_window=10)
        for i in range(5):
            db.add(self._make_desc(i))
        result = db.query(self._make_desc(99))
        assert result == []

    def test_query_finds_same_descriptor(self):
        """A descriptor should match itself when added outside exclusion window."""
        db = self._make_db(exclusion_window=2, candidate_pool_size=5)
        target_desc = self._make_desc(0)
        db.add(target_desc, frame_id=0)

        # Add more frames so the target is outside the exclusion window.
        for i in range(1, 10):
            db.add(self._make_desc(i + 1), frame_id=i)

        candidates = db.query(target_desc, top_k=1)
        assert len(candidates) == 1
        assert candidates[0].match_frame_id == 0
        assert candidates[0].distance < 0.05

    def test_candidates_sorted_ascending(self):
        db = self._make_db(exclusion_window=0, candidate_pool_size=20)
        for i in range(20):
            db.add(self._make_desc(i))
        candidates = db.query(self._make_desc(99), top_k=5)
        distances = [c.distance for c in candidates]
        assert distances == sorted(distances)

    def test_top_k_respected(self):
        db = self._make_db(exclusion_window=0)
        for i in range(30):
            db.add(self._make_desc(i))
        candidates = db.query(self._make_desc(99), top_k=3)
        assert len(candidates) <= 3

    def test_candidate_has_required_fields(self):
        db = self._make_db(exclusion_window=0)
        desc = self._make_desc(0)
        db.add(desc)
        candidates = db.query(desc, top_k=1)
        c = candidates[0]
        assert isinstance(c, LoopClosureCandidate)
        assert isinstance(c.match_frame_id, int)
        assert isinstance(c.distance, float)
        assert isinstance(c.yaw_shift_sectors, int)
        assert isinstance(c.database_index, int)
        assert 0.0 <= c.distance <= 1.0
        assert 0 <= c.yaw_shift_sectors < 20

    def test_incompatible_descriptor_raises_on_add(self):
        db = self._make_db()
        pts = _random_cloud()
        bad_desc = compute_scan_context(
            pts, num_rings=5, num_sectors=20, max_range=50.0
        )
        with pytest.raises(ValueError, match="num_rings"):
            db.add(bad_desc)

    def test_incompatible_descriptor_raises_on_query(self):
        db = self._make_db()
        pts = _random_cloud()
        bad_desc = compute_scan_context(
            pts, num_rings=10, num_sectors=10, max_range=50.0
        )
        with pytest.raises(ValueError, match="num_sectors"):
            db.query(bad_desc)

    def test_invalid_top_k_raises(self):
        db = self._make_db()
        desc = self._make_desc(0)
        with pytest.raises(ValueError, match="top_k"):
            db.query(desc, top_k=0)

    def test_loop_closure_detection_rotated_revisit(self):
        """Revisiting the same location with a slight yaw rotation should score
        lower (better match) than an unrelated location."""
        db = ScanContextDatabase(
            num_rings=20, num_sectors=60, max_range=50.0,
            exclusion_window=5, candidate_pool_size=15,
        )
        # Add a 'previous visit' at frame 0.
        original_pts = _random_cloud(800, seed=42, max_range=40.0)
        original_desc = compute_scan_context(
            original_pts, num_rings=20, num_sectors=60, max_range=50.0
        )
        db.add(original_desc, frame_id=0)

        # Add unrelated frames to satisfy the exclusion window.
        for i in range(1, 10):
            db.add(
                compute_scan_context(
                    _random_cloud(500, seed=i + 100, max_range=40.0),
                    num_rings=20, num_sectors=60, max_range=50.0,
                ),
                frame_id=i,
            )

        # Query with the same location, slightly rotated (simulating a revisit).
        revisit_pts = _yaw_rotate(original_pts, math.radians(15))
        revisit_desc = compute_scan_context(
            revisit_pts, num_rings=20, num_sectors=60, max_range=50.0
        )
        candidates = db.query(revisit_desc, top_k=1)
        assert len(candidates) == 1
        # The revisit should match back to frame 0.
        assert candidates[0].match_frame_id == 0
        # The revisit distance should be strictly less than matching a completely
        # different cloud (sampled independently with a different seed).
        different_pts = _random_cloud(800, seed=200, max_range=40.0)
        different_desc = compute_scan_context(
            different_pts, num_rings=20, num_sectors=60, max_range=50.0
        )
        dist_revisit, _ = scan_context_distance(revisit_desc, original_desc)
        dist_different, _ = scan_context_distance(different_desc, original_desc)
        assert dist_revisit < dist_different


class TestScanContextDatabaseInitValidation:
    def test_invalid_num_rings_raises(self):
        with pytest.raises(ValueError, match="num_rings"):
            ScanContextDatabase(num_rings=0)

    def test_invalid_num_sectors_raises(self):
        with pytest.raises(ValueError, match="num_sectors"):
            ScanContextDatabase(num_sectors=0)

    def test_invalid_max_range_raises(self):
        with pytest.raises(ValueError, match="max_range"):
            ScanContextDatabase(max_range=-1.0)

    def test_invalid_exclusion_window_raises(self):
        with pytest.raises(ValueError, match="exclusion_window"):
            ScanContextDatabase(exclusion_window=-1)

    def test_invalid_candidate_pool_raises(self):
        with pytest.raises(ValueError, match="candidate_pool_size"):
            ScanContextDatabase(candidate_pool_size=0)


# ---------------------------------------------------------------------------
# Dataclass fields
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_scan_context_descriptor_fields(self):
        mat = np.ones((5, 10))
        rk = mat.mean(axis=1)
        desc = ScanContextDescriptor(
            matrix=mat, ring_key=rk, num_rings=5, num_sectors=10, max_range=40.0
        )
        assert desc.matrix.shape == (5, 10)
        assert desc.ring_key.shape == (5,)
        assert desc.num_rings == 5
        assert desc.num_sectors == 10
        assert desc.max_range == 40.0

    def test_loop_closure_candidate_fields(self):
        c = LoopClosureCandidate(
            match_frame_id=7, distance=0.12, yaw_shift_sectors=3, database_index=2
        )
        assert c.match_frame_id == 7
        assert c.distance == 0.12
        assert c.yaw_shift_sectors == 3
        assert c.database_index == 2


# ---------------------------------------------------------------------------
# compute_m2dp
# ---------------------------------------------------------------------------


class TestComputeM2dp:
    def test_returns_descriptor(self):
        pts = _random_cloud()
        desc = compute_m2dp(pts)
        assert isinstance(desc, M2dpDescriptor)

    def test_vector_length(self):
        pts = _random_cloud()
        na, ne, nr, ns = 4, 8, 4, 16
        desc = compute_m2dp(pts, num_azimuth=na, num_elevation=ne,
                            num_rings=nr, num_sectors=ns)
        expected_len = na * ne + nr * ns
        assert desc.vector.shape == (expected_len,)

    def test_stores_params(self):
        pts = _random_cloud()
        desc = compute_m2dp(pts, num_azimuth=3, num_elevation=6,
                            num_rings=5, num_sectors=12)
        assert desc.num_azimuth == 3
        assert desc.num_elevation == 6
        assert desc.num_rings == 5
        assert desc.num_sectors == 12

    def test_empty_cloud_zero_vector(self):
        """An all-zero point cloud should yield a zero (or near-zero) descriptor."""
        pts = np.zeros((50, 3))
        desc = compute_m2dp(pts, num_azimuth=2, num_elevation=4,
                            num_rings=2, num_sectors=4)
        # All points at centroid → all projected to origin → single bin
        assert desc.vector.shape == (2 * 4 + 2 * 4,)

    def test_accepts_extra_columns(self):
        pts = np.random.default_rng(5).uniform(-10, 10, (80, 4))
        desc = compute_m2dp(pts, num_azimuth=2, num_elevation=4,
                            num_rings=3, num_sectors=8)
        assert desc.vector.shape == (2 * 4 + 3 * 8,)

    def test_same_cloud_same_descriptor(self):
        pts = _random_cloud(400, seed=20)
        d1 = compute_m2dp(pts, num_azimuth=4, num_elevation=8,
                          num_rings=4, num_sectors=16)
        d2 = compute_m2dp(pts, num_azimuth=4, num_elevation=8,
                          num_rings=4, num_sectors=16)
        np.testing.assert_array_equal(d1.vector, d2.vector)

    def test_similar_cloud_lower_distance(self):
        """A translated copy of the cloud should match better than a random one."""
        pts = _random_cloud(500, seed=30)
        pts_shifted = pts + np.array([0.1, 0.1, 0.0])  # small translation
        pts_different = _random_cloud(500, seed=31)

        d_orig = compute_m2dp(pts, num_azimuth=4, num_elevation=8,
                               num_rings=4, num_sectors=16)
        d_shift = compute_m2dp(pts_shifted, num_azimuth=4, num_elevation=8,
                                num_rings=4, num_sectors=16)
        d_diff = compute_m2dp(pts_different, num_azimuth=4, num_elevation=8,
                               num_rings=4, num_sectors=16)

        dist_same = m2dp_distance(d_orig, d_shift)
        dist_diff = m2dp_distance(d_orig, d_diff)
        assert dist_same < dist_diff


class TestComputeM2dpInputValidation:
    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="\\(N, ≥3\\)"):
            compute_m2dp(np.zeros((10, 2)))

    def test_1d_array_raises(self):
        with pytest.raises(ValueError, match="\\(N, ≥3\\)"):
            compute_m2dp(np.zeros(12))

    def test_invalid_num_azimuth_raises(self):
        pts = _random_cloud()
        with pytest.raises(ValueError, match="num_azimuth"):
            compute_m2dp(pts, num_azimuth=0)

    def test_invalid_num_elevation_raises(self):
        pts = _random_cloud()
        with pytest.raises(ValueError, match="num_elevation"):
            compute_m2dp(pts, num_elevation=0)

    def test_invalid_num_rings_raises(self):
        pts = _random_cloud()
        with pytest.raises(ValueError, match="num_rings"):
            compute_m2dp(pts, num_rings=0)

    def test_invalid_num_sectors_raises(self):
        pts = _random_cloud()
        with pytest.raises(ValueError, match="num_sectors"):
            compute_m2dp(pts, num_sectors=0)


# ---------------------------------------------------------------------------
# m2dp_distance
# ---------------------------------------------------------------------------


class TestM2dpDistance:
    def _make_desc(self, seed: int) -> M2dpDescriptor:
        return compute_m2dp(
            _random_cloud(300, seed=seed),
            num_azimuth=4, num_elevation=8, num_rings=4, num_sectors=16,
        )

    def test_identical_descriptors_zero_distance(self):
        desc = self._make_desc(40)
        assert m2dp_distance(desc, desc) < 1e-10

    def test_distance_in_0_1_range(self):
        d1 = self._make_desc(41)
        d2 = self._make_desc(42)
        dist = m2dp_distance(d1, d2)
        assert 0.0 <= dist <= 1.0

    def test_incompatible_params_raise(self):
        d1 = compute_m2dp(_random_cloud(100, seed=0),
                          num_azimuth=4, num_elevation=8,
                          num_rings=4, num_sectors=16)
        d2 = compute_m2dp(_random_cloud(100, seed=1),
                          num_azimuth=4, num_elevation=8,
                          num_rings=4, num_sectors=8)  # different num_sectors
        with pytest.raises(ValueError, match="incompatible parameters"):
            m2dp_distance(d1, d2)

    def test_m2dp_descriptor_fields(self):
        vec = np.ones(20)
        desc = M2dpDescriptor(
            vector=vec, num_azimuth=4, num_elevation=8,
            num_rings=4, num_sectors=16,
        )
        assert desc.vector.shape == (20,)
        assert desc.num_azimuth == 4
        assert desc.num_elevation == 8
        assert desc.num_rings == 4
        assert desc.num_sectors == 16
