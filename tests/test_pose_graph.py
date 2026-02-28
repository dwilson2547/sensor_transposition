"""Tests for pose_graph: PoseGraph data structure and optimize_pose_graph."""

import math

import numpy as np
import pytest

from sensor_transposition.pose_graph import (
    OptimizationResult,
    PoseGraph,
    PoseGraphEdge,
    PoseGraphNode,
    _edge_error,
    _exp_so3,
    _log_so3,
    _params_to_transform,
    _rotmat_to_quat,
    _transform_to_params,
    optimize_pose_graph,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rot_z(angle_rad: float) -> np.ndarray:
    """4×4 SE(3) pure-rotation about z-axis."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    T = np.eye(4)
    T[0, 0] = c; T[0, 1] = -s
    T[1, 0] = s; T[1, 1] = c
    return T


def _translation_tf(tx: float, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    """4×4 SE(3) pure-translation."""
    T = np.eye(4)
    T[0, 3] = tx; T[1, 3] = ty; T[2, 3] = tz
    return T


# ---------------------------------------------------------------------------
# SO(3) helpers
# ---------------------------------------------------------------------------


class TestSO3Helpers:
    def test_exp_log_roundtrip(self):
        """log(exp(φ)) should recover φ for a generic rotation vector."""
        phi = np.array([0.1, -0.2, 0.3])
        R = _exp_so3(phi)
        phi2 = _log_so3(R)
        np.testing.assert_allclose(phi2, phi, atol=1e-12)

    def test_exp_identity_for_zero(self):
        R = _exp_so3(np.zeros(3))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_log_identity_returns_zero(self):
        phi = _log_so3(np.eye(3))
        np.testing.assert_allclose(phi, np.zeros(3), atol=1e-12)

    def test_exp_so3_is_rotation_matrix(self):
        phi = np.array([0.5, -0.3, 0.7])
        R = _exp_so3(phi)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12


class TestRotmatToQuat:
    def test_identity_returns_identity_quat(self):
        q = _rotmat_to_quat(np.eye(3))
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_unit_norm(self):
        phi = np.array([0.3, -0.5, 0.2])
        R = _exp_so3(phi)
        q = _rotmat_to_quat(R)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-12

    def test_roundtrip_through_exp_so3(self):
        """R → q → R should reproduce the original rotation matrix."""
        from sensor_transposition.pose_graph import _quat_to_rotmat
        phi = np.array([0.1, 0.2, -0.3])
        R = _exp_so3(phi)
        q = _rotmat_to_quat(R)
        R2 = _quat_to_rotmat(q)
        np.testing.assert_allclose(R2, R, atol=1e-12)


# ---------------------------------------------------------------------------
# Param / transform helpers
# ---------------------------------------------------------------------------


class TestParamsTransform:
    def test_identity_roundtrip(self):
        T = np.eye(4)
        p = _transform_to_params(T)
        T2 = _params_to_transform(p)
        np.testing.assert_allclose(T2, T, atol=1e-12)

    def test_translation_preserved(self):
        T = _translation_tf(3.0, -1.0, 2.0)
        p = _transform_to_params(T)
        T2 = _params_to_transform(p)
        np.testing.assert_allclose(T2[:3, 3], [3.0, -1.0, 2.0], atol=1e-12)

    def test_rotation_preserved(self):
        phi = np.array([0.2, -0.1, 0.4])
        T = np.eye(4)
        T[:3, :3] = _exp_so3(phi)
        p = _transform_to_params(T)
        T2 = _params_to_transform(p)
        np.testing.assert_allclose(T2[:3, :3], T[:3, :3], atol=1e-12)


# ---------------------------------------------------------------------------
# _edge_error
# ---------------------------------------------------------------------------


class TestEdgeError:
    def _state(self, T_i: np.ndarray, T_j: np.ndarray) -> np.ndarray:
        """Build a 12-D state vector for two nodes."""
        x = np.zeros(12)
        x[:6] = _transform_to_params(T_i)
        x[6:12] = _transform_to_params(T_j)
        return x

    def test_zero_error_for_exact_measurement(self):
        """When T_meas equals the actual relative pose, the error should be 0."""
        T_i = _translation_tf(1.0, 2.0, 0.0)
        T_j = _translation_tf(3.0, 2.0, 0.0)
        T_meas = np.linalg.inv(T_i) @ T_j  # exact relative transform
        x = self._state(T_i, T_j)
        e = _edge_error(x, 0, 1, T_meas)
        np.testing.assert_allclose(e, np.zeros(6), atol=1e-10)

    def test_translation_error_detected(self):
        """A wrong translation measurement should produce a nonzero error."""
        T_i = np.eye(4)
        T_j = _translation_tf(2.0)
        T_meas = _translation_tf(1.0)  # measurement says 1 m, actual is 2 m
        x = self._state(T_i, T_j)
        e = _edge_error(x, 0, 1, T_meas)
        assert np.linalg.norm(e) > 0.5

    def test_error_length_six(self):
        x = np.zeros(12)
        T_meas = np.eye(4)
        e = _edge_error(x, 0, 1, T_meas)
        assert e.shape == (6,)


# ---------------------------------------------------------------------------
# PoseGraphNode
# ---------------------------------------------------------------------------


class TestPoseGraphNode:
    def test_default_translation_and_quat(self):
        node = PoseGraphNode(node_id=0)
        assert node.translation == [0.0, 0.0, 0.0]
        assert node.quaternion == [1.0, 0.0, 0.0, 0.0]

    def test_transform_is_identity_for_default(self):
        node = PoseGraphNode(node_id=0)
        np.testing.assert_allclose(node.transform, np.eye(4), atol=1e-12)

    def test_transform_encodes_translation(self):
        node = PoseGraphNode(node_id=1, translation=[1.0, 2.0, 3.0])
        T = node.transform
        np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0], atol=1e-12)


# ---------------------------------------------------------------------------
# PoseGraphEdge
# ---------------------------------------------------------------------------


class TestPoseGraphEdge:
    def test_default_transform_and_information(self):
        edge = PoseGraphEdge(from_id=0, to_id=1)
        np.testing.assert_allclose(edge.transform, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(edge.information, np.eye(6), atol=1e-12)

    def test_fields(self):
        T = _translation_tf(1.0)
        Omega = np.eye(6) * 2.0
        edge = PoseGraphEdge(from_id=3, to_id=7, transform=T, information=Omega)
        assert edge.from_id == 3
        assert edge.to_id == 7
        np.testing.assert_allclose(edge.transform, T, atol=1e-12)
        np.testing.assert_allclose(edge.information, Omega, atol=1e-12)


# ---------------------------------------------------------------------------
# PoseGraph
# ---------------------------------------------------------------------------


class TestPoseGraph:
    def test_empty_graph_len_zero(self):
        g = PoseGraph()
        assert len(g) == 0

    def test_add_node_increments_len(self):
        g = PoseGraph()
        g.add_node(0)
        g.add_node(1)
        assert len(g) == 2

    def test_add_node_returns_node(self):
        g = PoseGraph()
        node = g.add_node(5, translation=[1.0, 2.0, 3.0])
        assert isinstance(node, PoseGraphNode)
        assert node.node_id == 5
        assert node.translation == [1.0, 2.0, 3.0]

    def test_add_node_with_transform(self):
        g = PoseGraph()
        T = _translation_tf(4.0, 5.0, 6.0)
        node = g.add_node(0, transform=T)
        np.testing.assert_allclose(node.translation, [4.0, 5.0, 6.0], atol=1e-12)

    def test_duplicate_node_raises(self):
        g = PoseGraph()
        g.add_node(0)
        with pytest.raises(ValueError, match="already exists"):
            g.add_node(0)

    def test_get_node(self):
        g = PoseGraph()
        g.add_node(3, translation=[1.0, 0.0, 0.0])
        node = g.get_node(3)
        assert node.node_id == 3

    def test_get_missing_node_raises(self):
        g = PoseGraph()
        with pytest.raises(KeyError):
            g.get_node(99)

    def test_add_edge_returns_edge(self):
        g = PoseGraph()
        g.add_node(0)
        g.add_node(1)
        edge = g.add_edge(0, 1, transform=np.eye(4))
        assert isinstance(edge, PoseGraphEdge)
        assert edge.from_id == 0
        assert edge.to_id == 1

    def test_add_edge_default_information_is_identity(self):
        g = PoseGraph()
        g.add_node(0)
        g.add_node(1)
        edge = g.add_edge(0, 1, transform=np.eye(4))
        np.testing.assert_allclose(edge.information, np.eye(6), atol=1e-12)

    def test_add_edge_missing_node_raises(self):
        g = PoseGraph()
        g.add_node(0)
        with pytest.raises(ValueError, match="from_id"):
            g.add_edge(99, 0, transform=np.eye(4))

    def test_add_edge_bad_transform_shape_raises(self):
        g = PoseGraph()
        g.add_node(0)
        g.add_node(1)
        with pytest.raises(ValueError, match="4×4"):
            g.add_edge(0, 1, transform=np.eye(3))

    def test_add_edge_bad_information_shape_raises(self):
        g = PoseGraph()
        g.add_node(0)
        g.add_node(1)
        with pytest.raises(ValueError, match="6×6"):
            g.add_edge(0, 1, transform=np.eye(4), information=np.eye(3))

    def test_nodes_property_returns_copy(self):
        g = PoseGraph()
        g.add_node(0)
        nodes = g.nodes
        nodes[99] = None  # mutating the copy should not affect the graph
        assert 99 not in g._nodes

    def test_edges_property_returns_list(self):
        g = PoseGraph()
        g.add_node(0)
        g.add_node(1)
        g.add_edge(0, 1, transform=np.eye(4))
        assert isinstance(g.edges, list)
        assert len(g.edges) == 1


# ---------------------------------------------------------------------------
# optimize_pose_graph — basic
# ---------------------------------------------------------------------------


class TestOptimizePoseGraphBasic:
    def test_empty_graph_returns_empty(self):
        g = PoseGraph()
        result = optimize_pose_graph(g)
        assert isinstance(result, OptimizationResult)
        assert result.optimized_poses == {}
        assert result.success

    def test_single_node_no_edges(self):
        g = PoseGraph()
        g.add_node(0, translation=[1.0, 2.0, 3.0])
        result = optimize_pose_graph(g)
        assert result.success
        assert len(result.optimized_poses) == 1

    def test_result_has_expected_keys(self):
        g = PoseGraph()
        g.add_node(0)
        g.add_node(1)
        g.add_edge(0, 1, transform=_translation_tf(1.0))
        result = optimize_pose_graph(g)
        pose = result.optimized_poses[1]
        assert "translation" in pose
        assert "quaternion" in pose
        assert "transform" in pose

    def test_result_transform_is_4x4(self):
        g = PoseGraph()
        g.add_node(0)
        g.add_node(1)
        g.add_edge(0, 1, transform=_translation_tf(1.0))
        result = optimize_pose_graph(g)
        T = result.optimized_poses[1]["transform"]
        assert T.shape == (4, 4)

    def test_invalid_max_iterations_raises(self):
        g = PoseGraph()
        with pytest.raises(ValueError, match="max_iterations"):
            optimize_pose_graph(g, max_iterations=0)

    def test_invalid_tolerance_raises(self):
        g = PoseGraph()
        with pytest.raises(ValueError, match="tolerance"):
            optimize_pose_graph(g, tolerance=-1.0)


# ---------------------------------------------------------------------------
# optimize_pose_graph — correctness
# ---------------------------------------------------------------------------


class TestOptimizePoseGraphCorrectness:
    def test_single_edge_corrects_translation_error(self):
        """A noisy initial pose should be pulled toward the measurement."""
        g = PoseGraph()
        g.add_node(0, translation=[0.0, 0.0, 0.0])
        # Initial pose is slightly off (0.5 m instead of the expected 1.0 m).
        g.add_node(1, translation=[0.5, 0.0, 0.0])
        # Edge says the true relative pose is 1 m in x.
        g.add_edge(0, 1,
                   transform=_translation_tf(1.0),
                   information=np.eye(6) * 1000.0)
        result = optimize_pose_graph(g, max_iterations=50)
        t1 = result.optimized_poses[1]["translation"]
        assert abs(t1[0] - 1.0) < 1e-3, f"Expected x≈1.0, got {t1[0]:.6f}"

    def test_chain_of_three_nodes(self):
        """Three nodes in a chain should be arranged correctly after optimisation."""
        g = PoseGraph()
        g.add_node(0, translation=[0.0, 0.0, 0.0])
        g.add_node(1, translation=[0.8, 0.0, 0.0])   # noisy
        g.add_node(2, translation=[2.2, 0.0, 0.0])   # noisy
        g.add_edge(0, 1, transform=_translation_tf(1.0), information=np.eye(6) * 500.0)
        g.add_edge(1, 2, transform=_translation_tf(1.0), information=np.eye(6) * 500.0)
        result = optimize_pose_graph(g, max_iterations=50)
        t1 = result.optimized_poses[1]["translation"]
        t2 = result.optimized_poses[2]["translation"]
        assert abs(t1[0] - 1.0) < 0.05
        assert abs(t2[0] - 2.0) < 0.05

    def test_loop_closure_reduces_cost(self):
        """Adding a loop-closure edge should reduce the total cost vs no loop."""
        # Build a square loop: 0→1→2→3→0
        positions = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        noise = 0.1
        rng = np.random.default_rng(42)

        def _make_graph(add_loop_closure: bool) -> PoseGraph:
            g = PoseGraph()
            for i, (px, py) in enumerate(positions):
                g.add_node(i, translation=[px + (rng.uniform(-noise, noise) if i > 0 else 0),
                                           py + (rng.uniform(-noise, noise) if i > 0 else 0),
                                           0.0])
            for i in range(4):
                j = (i + 1) % 4
                px_i, py_i = positions[i]
                px_j, py_j = positions[j]
                dx, dy = px_j - px_i, py_j - py_i
                T = _translation_tf(dx, dy)
                if j != 0:  # skip the loop edge for the no-loop case
                    g.add_edge(i, j, transform=T, information=np.eye(6) * 100.0)
                elif add_loop_closure:
                    g.add_edge(i, j, transform=T, information=np.eye(6) * 100.0)
            return g

        g_no_loop = _make_graph(add_loop_closure=False)
        g_loop = _make_graph(add_loop_closure=True)

        result_no_loop = optimize_pose_graph(g_no_loop, max_iterations=50)
        result_loop = optimize_pose_graph(g_loop, max_iterations=50)

        assert result_loop.final_cost <= result_no_loop.final_cost + 1e-6

    def test_pure_rotation_edge(self):
        """An edge with a pure rotation constraint should align orientations."""
        g = PoseGraph()
        g.add_node(0, translation=[0.0, 0.0, 0.0])  # identity orientation
        # Initial orientation: identity (but the edge says 90° around z).
        g.add_node(1, translation=[0.0, 0.0, 0.0])
        # Measurement: 90° rotation around z.
        T_meas = _rot_z(math.radians(90))
        g.add_edge(0, 1, transform=T_meas, information=np.eye(6) * 1000.0)
        result = optimize_pose_graph(g, max_iterations=50)
        T_opt = result.optimized_poses[1]["transform"]
        expected_R = _rot_z(math.radians(90))[:3, :3]
        np.testing.assert_allclose(T_opt[:3, :3], expected_R, atol=1e-3)

    def test_first_node_stays_fixed(self):
        """The first node's pose must not change during optimisation."""
        g = PoseGraph()
        g.add_node(0, translation=[5.0, 3.0, 1.0])
        g.add_node(1, translation=[0.0, 0.0, 0.0])
        g.add_edge(0, 1, transform=_translation_tf(1.0), information=np.eye(6) * 1.0)
        result = optimize_pose_graph(g, max_iterations=20)
        t0 = result.optimized_poses[0]["translation"]
        np.testing.assert_allclose(t0, [5.0, 3.0, 1.0], atol=1e-12)

    def test_final_cost_is_non_negative(self):
        g = PoseGraph()
        g.add_node(0); g.add_node(1); g.add_node(2)
        g.add_edge(0, 1, transform=_translation_tf(1.0))
        g.add_edge(1, 2, transform=_translation_tf(1.0))
        result = optimize_pose_graph(g)
        assert result.final_cost >= 0.0

    def test_consistent_graph_zero_cost(self):
        """When all initial poses are consistent with all edges, cost should be ~0."""
        g = PoseGraph()
        g.add_node(0, translation=[0.0, 0.0, 0.0])
        g.add_node(1, translation=[2.0, 0.0, 0.0])
        g.add_node(2, translation=[4.0, 0.0, 0.0])
        T_01 = np.linalg.inv(g.get_node(0).transform) @ g.get_node(1).transform
        T_12 = np.linalg.inv(g.get_node(1).transform) @ g.get_node(2).transform
        g.add_edge(0, 1, transform=T_01, information=np.eye(6) * 100.0)
        g.add_edge(1, 2, transform=T_12, information=np.eye(6) * 100.0)
        result = optimize_pose_graph(g, max_iterations=5)
        assert result.final_cost < 1e-10
