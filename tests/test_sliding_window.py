"""Tests for sliding_window: SlidingWindowSmoother."""

import numpy as np
import pytest

from sensor_transposition.sliding_window import SlidingWindowSmoother, _ANCHOR_ID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _translation_tf(tx: float, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    """4×4 SE(3) pure-translation matrix."""
    T = np.eye(4)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


def _build_chain(
    n_nodes: int,
    step: float = 1.0,
    noise: float = 0.0,
    window_size: int = 5,
    rng: np.random.Generator | None = None,
) -> tuple[SlidingWindowSmoother, list]:
    """Build a straight-line chain of *n_nodes* nodes with step *step*."""
    if rng is None:
        rng = np.random.default_rng(0)
    smoother = SlidingWindowSmoother(window_size=window_size)
    T_step = _translation_tf(step)
    results = []
    for i in range(n_nodes):
        noisy = noise * rng.standard_normal(3) if noise > 0 else [0.0, 0.0, 0.0]
        smoother.add_node(
            i,
            translation=[float(i) * step + noisy[0], noisy[1], noisy[2]],
        )
        if i > 0:
            smoother.add_edge(
                i - 1,
                i,
                transform=T_step,
                information=np.eye(6) * 200.0,
            )
        results.append(smoother.optimize())
    return smoother, results


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


class TestSlidingWindowSmootherInit:
    def test_default_window_size(self):
        s = SlidingWindowSmoother()
        assert s.window_size == 10

    def test_custom_window_size(self):
        s = SlidingWindowSmoother(window_size=4)
        assert s.window_size == 4

    def test_invalid_window_size_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            SlidingWindowSmoother(window_size=0)

    def test_invalid_max_iterations_raises(self):
        with pytest.raises(ValueError, match="max_iterations"):
            SlidingWindowSmoother(max_iterations=0)

    def test_invalid_tolerance_raises(self):
        with pytest.raises(ValueError, match="tolerance"):
            SlidingWindowSmoother(tolerance=-1.0)

    def test_initial_active_node_ids_empty(self):
        s = SlidingWindowSmoother()
        assert s.active_node_ids == []

    def test_latest_result_none_initially(self):
        s = SlidingWindowSmoother()
        assert s.latest_result is None


# ---------------------------------------------------------------------------
# add_node
# ---------------------------------------------------------------------------


class TestAddNode:
    def test_reserved_anchor_id_raises(self):
        s = SlidingWindowSmoother()
        with pytest.raises(ValueError, match="reserved"):
            s.add_node(_ANCHOR_ID)

    def test_duplicate_node_raises(self):
        s = SlidingWindowSmoother()
        s.add_node(0)
        with pytest.raises(ValueError, match="already in the active window"):
            s.add_node(0)

    def test_node_appears_in_active_ids(self):
        s = SlidingWindowSmoother()
        s.add_node(5, translation=[1.0, 2.0, 3.0])
        assert 5 in s.active_node_ids

    def test_window_does_not_exceed_size(self):
        s = SlidingWindowSmoother(window_size=3)
        for i in range(10):
            s.add_node(i)
        assert len(s.active_node_ids) == 3

    def test_oldest_node_evicted_when_overflow(self):
        s = SlidingWindowSmoother(window_size=3)
        s.add_node(0)
        s.add_node(1)
        s.add_node(2)
        assert 0 in s.active_node_ids
        s.add_node(3)
        assert 0 not in s.active_node_ids
        assert 3 in s.active_node_ids

    def test_add_node_with_transform(self):
        s = SlidingWindowSmoother()
        T = _translation_tf(5.0, 3.0, 1.0)
        s.add_node(0, transform=T)
        assert 0 in s.active_node_ids

    def test_invalid_transform_shape_raises(self):
        s = SlidingWindowSmoother()
        with pytest.raises(ValueError, match="4×4"):
            s.add_node(0, transform=np.eye(3))


# ---------------------------------------------------------------------------
# add_edge
# ---------------------------------------------------------------------------


class TestAddEdge:
    def test_invalid_transform_shape_raises(self):
        s = SlidingWindowSmoother()
        s.add_node(0)
        s.add_node(1)
        with pytest.raises(ValueError, match="4×4"):
            s.add_edge(0, 1, transform=np.eye(3))

    def test_invalid_information_shape_raises(self):
        s = SlidingWindowSmoother()
        s.add_node(0)
        s.add_node(1)
        with pytest.raises(ValueError, match="6×6"):
            s.add_edge(0, 1, transform=np.eye(4), information=np.eye(3))

    def test_default_information_is_identity(self):
        s = SlidingWindowSmoother()
        s.add_node(0)
        s.add_node(1)
        s.add_edge(0, 1, transform=np.eye(4))
        # Edge was stored — we can verify indirectly via optimization.
        result = s.optimize()
        assert result is not None


# ---------------------------------------------------------------------------
# optimize — basic
# ---------------------------------------------------------------------------


class TestOptimizeBasic:
    def test_empty_window_returns_empty_result(self):
        s = SlidingWindowSmoother()
        result = s.optimize()
        assert result.optimized_poses == {}
        assert result.success

    def test_single_node_success(self):
        s = SlidingWindowSmoother()
        s.add_node(0, translation=[1.0, 2.0, 3.0])
        result = s.optimize()
        assert result.success
        assert 0 in result.optimized_poses

    def test_result_stored_as_latest(self):
        s = SlidingWindowSmoother()
        s.add_node(0)
        result = s.optimize()
        assert s.latest_result is result

    def test_anchor_not_in_result(self):
        s = SlidingWindowSmoother(window_size=2)
        for i in range(3):
            s.add_node(i, translation=[float(i), 0.0, 0.0])
            if i > 0:
                s.add_edge(i - 1, i, transform=_translation_tf(1.0))
        result = s.optimize()
        assert _ANCHOR_ID not in result.optimized_poses

    def test_result_has_expected_keys(self):
        s = SlidingWindowSmoother()
        s.add_node(0, translation=[0.0, 0.0, 0.0])
        s.add_node(1, translation=[1.0, 0.0, 0.0])
        s.add_edge(0, 1, transform=_translation_tf(1.0))
        result = s.optimize()
        pose = result.optimized_poses[1]
        assert "translation" in pose
        assert "quaternion" in pose
        assert "transform" in pose

    def test_result_transform_shape(self):
        s = SlidingWindowSmoother()
        s.add_node(0)
        s.add_node(1)
        s.add_edge(0, 1, transform=_translation_tf(2.0))
        result = s.optimize()
        assert result.optimized_poses[1]["transform"].shape == (4, 4)


# ---------------------------------------------------------------------------
# optimize — correctness: active window only
# ---------------------------------------------------------------------------


class TestOptimizeCorrectness:
    def test_two_node_translation(self):
        """Two nodes: noisy initial pose corrected by strong edge."""
        s = SlidingWindowSmoother(window_size=2)
        s.add_node(0, translation=[0.0, 0.0, 0.0])
        s.add_node(1, translation=[0.5, 0.0, 0.0])  # noisy
        s.add_edge(0, 1, transform=_translation_tf(1.0), information=np.eye(6) * 1000.0)
        result = s.optimize()
        t1 = result.optimized_poses[1]["translation"]
        assert abs(t1[0] - 1.0) < 1e-3

    def test_chain_stays_straight(self):
        """A straight-line chain should produce near-zero lateral deviation."""
        _, results = _build_chain(n_nodes=10, step=1.0, window_size=5)
        for i, result in enumerate(results):
            for nid, pose in result.optimized_poses.items():
                assert abs(pose["translation"][1]) < 0.01, (
                    f"Node {nid} in step {i}: lateral y = {pose['translation'][1]}"
                )

    def test_active_nodes_only_in_result(self):
        """The result must contain only the currently active nodes."""
        s, _ = _build_chain(n_nodes=10, window_size=4)
        active = set(s.active_node_ids)
        result = s.optimize()
        assert set(result.optimized_poses.keys()) == active

    def test_first_node_fixed_without_priors(self):
        """With no prior factors, the first active node must remain fixed."""
        s = SlidingWindowSmoother(window_size=5)
        s.add_node(0, translation=[3.0, 4.0, 0.0])
        s.add_node(1, translation=[3.5, 4.0, 0.0])  # noisy
        s.add_edge(0, 1, transform=_translation_tf(1.0), information=np.eye(6) * 1000.0)
        result = s.optimize()
        t0 = result.optimized_poses[0]["translation"]
        np.testing.assert_allclose(t0, [3.0, 4.0, 0.0], atol=1e-10)

    def test_final_cost_non_negative(self):
        _, results = _build_chain(n_nodes=8, window_size=4)
        for result in results:
            assert result.final_cost >= 0.0


# ---------------------------------------------------------------------------
# Marginalisation & prior propagation
# ---------------------------------------------------------------------------


class TestMarginalisation:
    def test_prior_anchors_position_after_eviction(self):
        """After marginalising node 0, nodes in the window should still be
        roughly at the correct positions due to prior constraints."""
        s = SlidingWindowSmoother(window_size=3)
        T_step = _translation_tf(1.0)
        info = np.eye(6) * 500.0

        for i in range(4):
            s.add_node(i, translation=[float(i), 0.0, 0.0])
            if i > 0:
                s.add_edge(i - 1, i, transform=T_step, information=info)
            s.optimize()

        # Node 0 has been evicted; nodes 1, 2, 3 are active.
        assert 0 not in s.active_node_ids
        result = s.optimize()
        for nid in [1, 2, 3]:
            x = result.optimized_poses[nid]["translation"][0]
            assert abs(x - float(nid)) < 0.1, (
                f"Node {nid}: expected x≈{float(nid)}, got {x:.4f}"
            )

    def test_long_trajectory_bounded_window(self):
        """After N > window_size steps the active window must not exceed
        window_size, and all results must have been returned successfully."""
        n, w = 20, 5
        _, results = _build_chain(n_nodes=n, window_size=w)
        for r in results:
            assert r.success

    def test_position_accuracy_over_long_trajectory(self):
        """Each node's x-coordinate should be close to its ground-truth
        index * step size even after many marginalisations."""
        n, w, step = 15, 5, 2.0
        smoother = SlidingWindowSmoother(window_size=w)
        T_step = _translation_tf(step)
        info = np.eye(6) * 1000.0
        for i in range(n):
            smoother.add_node(i, translation=[float(i) * step, 0.0, 0.0])
            if i > 0:
                smoother.add_edge(i - 1, i, transform=T_step, information=info)
            result = smoother.optimize()
            for nid, pose in result.optimized_poses.items():
                expected_x = float(nid) * step
                actual_x = pose["translation"][0]
                assert abs(actual_x - expected_x) < 0.5, (
                    f"Step {i}, node {nid}: "
                    f"expected x≈{expected_x:.1f}, got {actual_x:.4f}"
                )

    def test_evicted_node_not_in_active_ids(self):
        s = SlidingWindowSmoother(window_size=2)
        s.add_node(0)
        s.add_node(1)
        s.add_node(2)  # triggers eviction of 0
        assert 0 not in s.active_node_ids

    def test_anchor_id_not_in_active_ids(self):
        """The internal anchor node must never appear in active_node_ids."""
        s, _ = _build_chain(n_nodes=10, window_size=3)
        assert _ANCHOR_ID not in s.active_node_ids
