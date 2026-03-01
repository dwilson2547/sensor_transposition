"""Tests for submap_manager: KeyframeSelector, Submap, SubmapManager."""

import math

import numpy as np
import pytest

from sensor_transposition.submap_manager import (
    KeyframeSelector,
    Submap,
    SubmapManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity() -> np.ndarray:
    return np.eye(4, dtype=float)


def _translation_tf(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


def _rot_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    T = np.eye(4, dtype=float)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    return T


def _scan(n: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, (n, 3))


# ---------------------------------------------------------------------------
# KeyframeSelector — construction
# ---------------------------------------------------------------------------


class TestKeySelectorInit:
    def test_defaults(self):
        sel = KeyframeSelector()
        assert sel.keyframe_count == 0
        assert sel.last_pose is None

    def test_zero_translation_threshold_raises(self):
        with pytest.raises(ValueError, match="translation_threshold"):
            KeyframeSelector(translation_threshold=0.0)

    def test_negative_translation_threshold_raises(self):
        with pytest.raises(ValueError, match="translation_threshold"):
            KeyframeSelector(translation_threshold=-1.0)

    def test_negative_rotation_threshold_raises(self):
        with pytest.raises(ValueError, match="rotation_threshold_deg"):
            KeyframeSelector(rotation_threshold_deg=-1.0)

    def test_zero_rotation_threshold_allowed(self):
        sel = KeyframeSelector(rotation_threshold_deg=0.0)
        assert sel.keyframe_count == 0


# ---------------------------------------------------------------------------
# KeyframeSelector — should_add_keyframe / mark_accepted / check_and_accept
# ---------------------------------------------------------------------------


class TestKeySelectorLogic:
    def test_first_pose_always_accepted(self):
        sel = KeyframeSelector(translation_threshold=5.0)
        assert sel.should_add_keyframe(_identity()) is True

    def test_should_add_does_not_update_state(self):
        sel = KeyframeSelector(translation_threshold=1.0)
        sel.should_add_keyframe(_identity())
        # State should still be uninitialised.
        assert sel.last_pose is None
        assert sel.keyframe_count == 0

    def test_mark_accepted_updates_state(self):
        sel = KeyframeSelector(translation_threshold=1.0)
        T = _translation_tf(tx=1.0)
        sel.mark_accepted(T)
        assert sel.keyframe_count == 1
        np.testing.assert_array_equal(sel.last_pose, T)

    def test_last_pose_returns_copy(self):
        sel = KeyframeSelector(translation_threshold=1.0)
        sel.mark_accepted(_identity())
        pose = sel.last_pose
        assert pose is not None
        pose[:] = 99.0
        np.testing.assert_array_equal(sel.last_pose, np.eye(4))

    def test_pose_below_both_thresholds_rejected(self):
        sel = KeyframeSelector(translation_threshold=2.0, rotation_threshold_deg=20.0)
        sel.mark_accepted(_identity())
        # 0.5 m, 5° — below both thresholds.
        T = _translation_tf(tx=0.5) @ _rot_z(math.radians(5.0))
        assert sel.should_add_keyframe(T) is False

    def test_pose_exceeds_translation_threshold(self):
        sel = KeyframeSelector(translation_threshold=1.0, rotation_threshold_deg=45.0)
        sel.mark_accepted(_identity())
        T = _translation_tf(tx=1.5)  # 1.5 m > 1.0 m threshold
        assert sel.should_add_keyframe(T) is True

    def test_pose_exceeds_rotation_threshold(self):
        sel = KeyframeSelector(translation_threshold=5.0, rotation_threshold_deg=15.0)
        sel.mark_accepted(_identity())
        T = _rot_z(math.radians(20.0))  # 20° > 15° threshold, 0 m translation
        assert sel.should_add_keyframe(T) is True

    def test_zero_rotation_threshold_only_translation_matters(self):
        sel = KeyframeSelector(translation_threshold=1.0, rotation_threshold_deg=0.0)
        sel.mark_accepted(_identity())
        # Large rotation but no translation → should be rejected.
        T = _rot_z(math.radians(90.0))
        assert sel.should_add_keyframe(T) is False
        # Sufficient translation → accepted.
        T2 = _translation_tf(tx=2.0)
        assert sel.should_add_keyframe(T2) is True

    def test_check_and_accept_returns_true_and_updates(self):
        sel = KeyframeSelector(translation_threshold=1.0)
        accepted = sel.check_and_accept(_identity())
        assert accepted is True
        assert sel.keyframe_count == 1

    def test_check_and_accept_returns_false_when_too_close(self):
        sel = KeyframeSelector(translation_threshold=2.0, rotation_threshold_deg=30.0)
        sel.check_and_accept(_identity())
        # Very close pose — should be rejected.
        accepted = sel.check_and_accept(_translation_tf(tx=0.1))
        assert accepted is False
        assert sel.keyframe_count == 1  # count unchanged

    def test_invalid_transform_shape_raises(self):
        sel = KeyframeSelector()
        with pytest.raises(ValueError, match="4×4"):
            sel.should_add_keyframe(np.eye(3))

    def test_mark_accepted_invalid_raises(self):
        sel = KeyframeSelector()
        with pytest.raises(ValueError, match="4×4"):
            sel.mark_accepted(np.zeros((3, 3)))

    def test_sequence_of_keyframes(self):
        """Walk in steps of 0.5 m; threshold 1.0 m → accept every other pose."""
        sel = KeyframeSelector(translation_threshold=1.0, rotation_threshold_deg=180.0)
        accepted_positions = []
        for i in range(10):
            T = _translation_tf(tx=float(i) * 0.5)
            if sel.check_and_accept(T):
                accepted_positions.append(i * 0.5)
        # Steps 0, 2, 4, 6, 8 (every 1.0 m).
        assert accepted_positions == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert sel.keyframe_count == 5


# ---------------------------------------------------------------------------
# Submap — dataclass
# ---------------------------------------------------------------------------


class TestSubmap:
    def test_default_fields(self):
        sm = Submap(submap_id=0)
        assert sm.submap_id == 0
        assert sm.keyframe_ids == []
        assert sm.size == 0
        np.testing.assert_array_equal(sm.origin_pose, np.eye(4))

    def test_size_reflects_keyframe_ids(self):
        sm = Submap(submap_id=1, keyframe_ids=[0, 1, 2])
        assert sm.size == 3


# ---------------------------------------------------------------------------
# SubmapManager — construction
# ---------------------------------------------------------------------------


class TestSubmapManagerInit:
    def test_defaults(self):
        mgr = SubmapManager()
        assert mgr.num_submaps == 0
        assert mgr.total_keyframes == 0
        assert mgr.get_current_submap() is None

    def test_invalid_max_keyframes_raises(self):
        with pytest.raises(ValueError, match="max_keyframes_per_submap"):
            SubmapManager(max_keyframes_per_submap=0)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            SubmapManager(overlap=-1)

    def test_overlap_equal_to_max_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            SubmapManager(max_keyframes_per_submap=3, overlap=3)

    def test_overlap_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            SubmapManager(max_keyframes_per_submap=3, overlap=5)


# ---------------------------------------------------------------------------
# SubmapManager — add_keyframe basic
# ---------------------------------------------------------------------------


class TestSubmapManagerAddKeyframe:
    def test_first_keyframe_creates_submap(self):
        mgr = SubmapManager(max_keyframes_per_submap=5)
        mgr.add_keyframe(0, _identity(), _scan(3, seed=0))
        assert mgr.num_submaps == 1
        assert mgr.total_keyframes == 1

    def test_returns_current_submap(self):
        mgr = SubmapManager(max_keyframes_per_submap=5)
        sm = mgr.add_keyframe(0, _identity(), _scan(3, seed=0))
        assert sm is mgr.get_current_submap()

    def test_keyframes_accumulate_within_submap(self):
        mgr = SubmapManager(max_keyframes_per_submap=5)
        for i in range(4):
            mgr.add_keyframe(i, _translation_tf(tx=float(i)), _scan(3, seed=i))
        sm = mgr.get_current_submap()
        assert sm is not None
        assert sm.size == 4
        assert mgr.num_submaps == 1

    def test_new_submap_created_at_max(self):
        mgr = SubmapManager(max_keyframes_per_submap=3)
        for i in range(3):
            mgr.add_keyframe(i, _translation_tf(tx=float(i)), _scan(3, seed=i))
        # 3rd keyframe fills the submap; adding a 4th triggers a new one.
        mgr.add_keyframe(3, _translation_tf(tx=3.0), _scan(3, seed=3))
        assert mgr.num_submaps == 2

    def test_get_all_submaps_length(self):
        mgr = SubmapManager(max_keyframes_per_submap=2)
        for i in range(6):
            mgr.add_keyframe(i, _translation_tf(tx=float(i)), _scan(3, seed=i))
        assert mgr.num_submaps == 3
        assert len(mgr.get_all_submaps()) == 3

    def test_invalid_transform_raises(self):
        mgr = SubmapManager()
        with pytest.raises(ValueError, match="4×4"):
            mgr.add_keyframe(0, np.eye(3), _scan(3))

    def test_get_submap_by_id(self):
        mgr = SubmapManager(max_keyframes_per_submap=2)
        for i in range(4):
            mgr.add_keyframe(i, _translation_tf(tx=float(i)), _scan(3, seed=i))
        sm = mgr.get_submap(1)
        assert sm.submap_id == 1

    def test_get_submap_missing_id_raises(self):
        mgr = SubmapManager()
        with pytest.raises(KeyError):
            mgr.get_submap(99)

    def test_origin_pose_is_first_keyframe_transform(self):
        mgr = SubmapManager(max_keyframes_per_submap=3)
        T0 = _translation_tf(tx=5.0)
        mgr.add_keyframe(0, T0, _scan(3, seed=0))
        sm0 = mgr.get_submap(0)
        np.testing.assert_array_almost_equal(sm0.origin_pose, T0)

    def test_second_submap_origin_is_first_keyframe_of_new_submap(self):
        """Without overlap, the new submap's origin should be the first fresh frame."""
        mgr = SubmapManager(max_keyframes_per_submap=2, overlap=0)
        for i in range(2):
            mgr.add_keyframe(i, _translation_tf(tx=float(i)), _scan(3, seed=i))
        T_new = _translation_tf(tx=10.0)
        mgr.add_keyframe(2, T_new, _scan(3, seed=2))
        sm1 = mgr.get_submap(1)
        np.testing.assert_array_almost_equal(sm1.origin_pose, T_new)

    def test_total_keyframes_counts_all_entries(self):
        mgr = SubmapManager(max_keyframes_per_submap=3, overlap=0)
        for i in range(7):
            mgr.add_keyframe(i, _translation_tf(tx=float(i)), _scan(3, seed=i))
        # 3 + 3 + 1 = 7
        assert mgr.total_keyframes == 7

    def test_point_cloud_populated(self):
        mgr = SubmapManager(max_keyframes_per_submap=5)
        pts = _scan(10, seed=42)
        mgr.add_keyframe(0, _identity(), pts)
        sm = mgr.get_current_submap()
        assert sm is not None
        assert len(sm.point_cloud) == 10


# ---------------------------------------------------------------------------
# SubmapManager — overlap
# ---------------------------------------------------------------------------


class TestSubmapManagerOverlap:
    def test_overlap_keyframes_appear_in_new_submap(self):
        """With overlap=1 and max=3, kf2 should appear at the start of submap1."""
        mgr = SubmapManager(max_keyframes_per_submap=3, overlap=1)
        for i in range(4):
            mgr.add_keyframe(i, _translation_tf(tx=float(i)), _scan(3, seed=i))
        # submap0 = [0, 1, 2] (filled), submap1 starts with overlap kf2, then kf3.
        sm0 = mgr.get_submap(0)
        sm1 = mgr.get_submap(1)
        assert sm0.keyframe_ids == [0, 1, 2]
        assert sm1.keyframe_ids[0] == 2   # overlap frame
        assert 3 in sm1.keyframe_ids

    def test_overlap_2_carries_two_keyframes(self):
        mgr = SubmapManager(max_keyframes_per_submap=5, overlap=2)
        for i in range(6):
            mgr.add_keyframe(i, _translation_tf(tx=float(i)), _scan(3, seed=i))
        sm1 = mgr.get_submap(1)
        # First two IDs should be the last two from sm0 (i.e. 3, 4).
        assert sm1.keyframe_ids[:2] == [3, 4]

    def test_overlap_origin_is_first_overlap_frame(self):
        mgr = SubmapManager(max_keyframes_per_submap=3, overlap=1)
        T0 = _translation_tf(tx=0.0)
        T1 = _translation_tf(tx=1.0)
        T2 = _translation_tf(tx=2.0)
        T3 = _translation_tf(tx=3.0)
        mgr.add_keyframe(0, T0, _scan(3, seed=0))
        mgr.add_keyframe(1, T1, _scan(3, seed=1))
        mgr.add_keyframe(2, T2, _scan(3, seed=2))
        mgr.add_keyframe(3, T3, _scan(3, seed=3))
        sm1 = mgr.get_submap(1)
        # Origin should be T2 (the overlap frame).
        np.testing.assert_array_almost_equal(sm1.origin_pose, T2)

    def test_point_cloud_includes_overlap_points(self):
        """The new submap's point cloud must include points from the overlap scan."""
        mgr = SubmapManager(max_keyframes_per_submap=2, overlap=1)
        pts0 = _scan(5, seed=0)
        pts1 = _scan(5, seed=1)
        pts2 = _scan(5, seed=2)
        mgr.add_keyframe(0, _identity(), pts0)
        mgr.add_keyframe(1, _identity(), pts1)
        mgr.add_keyframe(2, _translation_tf(tx=1.0), pts2)
        sm1 = mgr.get_current_submap()
        assert sm1 is not None
        # kf1 (overlap) + kf2 = 10 points.
        assert len(sm1.point_cloud) == 10

    def test_no_overlap_no_shared_keyframes(self):
        mgr = SubmapManager(max_keyframes_per_submap=3, overlap=0)
        for i in range(6):
            mgr.add_keyframe(i, _translation_tf(tx=float(i)), _scan(3, seed=i))
        sm0 = mgr.get_submap(0)
        sm1 = mgr.get_submap(1)
        # No IDs should overlap.
        assert len(set(sm0.keyframe_ids) & set(sm1.keyframe_ids)) == 0


# ---------------------------------------------------------------------------
# SubmapManager — integration with KeyframeSelector
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_selector_and_manager_together(self):
        """Simulate a robot driving in a straight line: select keyframes and
        build submaps."""
        sel = KeyframeSelector(translation_threshold=1.0, rotation_threshold_deg=45.0)
        mgr = SubmapManager(max_keyframes_per_submap=5, overlap=1)

        kf_id = 0
        for step in range(20):
            T = _translation_tf(tx=float(step) * 0.5)
            if sel.check_and_accept(T):
                mgr.add_keyframe(kf_id, T, _scan(4, seed=step))
                kf_id += 1

        assert sel.keyframe_count == kf_id
        assert mgr.num_submaps >= 1
        assert mgr.total_keyframes >= kf_id

    def test_get_all_submaps_returns_independent_list(self):
        mgr = SubmapManager(max_keyframes_per_submap=2)
        for i in range(4):
            mgr.add_keyframe(i, _translation_tf(tx=float(i)), _scan(3, seed=i))
        submaps = mgr.get_all_submaps()
        submaps.clear()  # should not affect internal state
        assert mgr.num_submaps == 2
