"""Tests for the transform module."""

import math

import numpy as np
import pytest

from sensor_transposition.transform import Transform, sensor_to_sensor


# ---------------------------------------------------------------------------
# Transform construction
# ---------------------------------------------------------------------------


class TestTransformConstruction:
    def test_identity_default(self):
        T = Transform()
        np.testing.assert_allclose(T.matrix, np.eye(4), atol=1e-10)

    def test_identity_classmethod(self):
        T = Transform.identity()
        np.testing.assert_allclose(T.matrix, np.eye(4), atol=1e-10)

    def test_from_translation(self):
        T = Transform.from_translation([1.0, 2.0, 3.0])
        np.testing.assert_allclose(T.translation, [1.0, 2.0, 3.0], atol=1e-10)
        np.testing.assert_allclose(T.rotation, np.eye(3), atol=1e-10)

    def test_from_rotation_matrix(self):
        R = np.eye(3)
        R[0, 0] = -1.0
        R[1, 1] = -1.0  # 180° about Z
        T = Transform.from_rotation_matrix(R, translation=[1.0, 0.0, 0.0])
        np.testing.assert_allclose(T.rotation, R, atol=1e-10)
        np.testing.assert_allclose(T.translation, [1.0, 0.0, 0.0], atol=1e-10)

    def test_from_quaternion_identity(self):
        T = Transform.from_quaternion([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(T.rotation, np.eye(3), atol=1e-10)

    def test_from_quaternion_90_z(self):
        angle = math.pi / 2
        w, z = math.cos(angle / 2), math.sin(angle / 2)
        T = Transform.from_quaternion([w, 0.0, 0.0, z])
        x_rotated = T.rotation @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(x_rotated, [0.0, 1.0, 0.0], atol=1e-10)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="4×4"):
            Transform(np.eye(3))


# ---------------------------------------------------------------------------
# Applying transforms to points
# ---------------------------------------------------------------------------


class TestTransformApply:
    def test_apply_to_point_identity(self):
        T = Transform.identity()
        result = T.apply_to_point([1.0, 2.0, 3.0])
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0], atol=1e-10)

    def test_apply_to_point_translation_only(self):
        T = Transform.from_translation([1.0, 2.0, 3.0])
        result = T.apply_to_point([0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0], atol=1e-10)

    def test_apply_to_point_rotation(self):
        angle = math.pi / 2
        w, z = math.cos(angle / 2), math.sin(angle / 2)
        T = Transform.from_quaternion([w, 0.0, 0.0, z])
        result = T.apply_to_point([1.0, 0.0, 0.0])
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-10)

    def test_apply_to_points_batch(self):
        T = Transform.from_translation([10.0, 0.0, 0.0])
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        result = T.apply_to_points(pts)
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result[0], [10.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(result[1], [11.0, 1.0, 1.0], atol=1e-10)

    def test_apply_to_points_homogeneous_input(self):
        T = Transform.from_translation([1.0, 2.0, 3.0])
        pts = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
        result = T.apply_to_points(pts)
        assert result.shape == (2, 3)

    def test_apply_to_point_wrong_size_raises(self):
        T = Transform.identity()
        with pytest.raises(ValueError):
            T.apply_to_point([1.0, 2.0])


# ---------------------------------------------------------------------------
# Composition and inverse
# ---------------------------------------------------------------------------


class TestTransformComposition:
    def test_compose_with_inverse_is_identity(self):
        T = Transform.from_translation([1.0, 2.0, 3.0])
        np.testing.assert_allclose((T @ T.inverse()).matrix, np.eye(4), atol=1e-10)

    def test_compose_translations(self):
        T1 = Transform.from_translation([1.0, 0.0, 0.0])
        T2 = Transform.from_translation([0.0, 1.0, 0.0])
        T12 = T1 @ T2
        np.testing.assert_allclose(T12.translation, [1.0, 1.0, 0.0], atol=1e-10)

    def test_matmul_non_transform_returns_notimplemented(self):
        T = Transform.identity()
        result = T.__matmul__(np.eye(4))
        assert result is NotImplemented

    def test_equality(self):
        T1 = Transform.from_translation([1.0, 2.0, 3.0])
        T2 = Transform.from_translation([1.0, 2.0, 3.0])
        assert T1 == T2

    def test_inequality(self):
        T1 = Transform.from_translation([1.0, 2.0, 3.0])
        T2 = Transform.from_translation([4.0, 5.0, 6.0])
        assert T1 != T2


# ---------------------------------------------------------------------------
# sensor_to_sensor convenience function
# ---------------------------------------------------------------------------


class TestSensorToSensor:
    def test_same_sensor_is_identity(self):
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        result = sensor_to_sensor(T, T)
        np.testing.assert_allclose(result, np.eye(4), atol=1e-10)

    def test_translation_only_sensors(self):
        """Two sensors with identity rotation, different translations."""
        T_a = np.eye(4)
        T_a[:3, 3] = [1.0, 0.0, 0.0]
        T_b = np.eye(4)
        T_b[:3, 3] = [3.0, 0.0, 0.0]

        # A point at origin in sensor_a should be at [1-3, 0, 0] = [-2, 0, 0]
        # in sensor_b (sensor_a is 2 m behind sensor_b relative to ego)
        T_ab = sensor_to_sensor(T_a, T_b)
        point_in_a = np.array([0.0, 0.0, 0.0, 1.0])
        point_in_b = T_ab @ point_in_a
        np.testing.assert_allclose(point_in_b[:3], [-2.0, 0.0, 0.0], atol=1e-10)

    def test_a_to_b_and_b_to_a_are_inverses(self):
        T_a = np.eye(4)
        T_a[:3, 3] = [1.0, 0.0, 0.0]
        T_b = np.eye(4)
        T_b[:3, 3] = [0.0, 2.0, 0.0]
        T_ab = sensor_to_sensor(T_a, T_b)
        T_ba = sensor_to_sensor(T_b, T_a)
        np.testing.assert_allclose(T_ab @ T_ba, np.eye(4), atol=1e-10)
