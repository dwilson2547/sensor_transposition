"""Tests for IMU pre-integration."""

import math

import numpy as np
import pytest

from sensor_transposition.imu.preintegration import (
    ImuPreintegrator,
    PreintegrationResult,
    _exp_so3,
    _skew,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_inputs(n: int, dt: float = 0.1, accel=(0.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0)):
    """Build uniform-interval timestamps with constant accel/gyro."""
    timestamps = np.arange(n, dtype=float) * dt
    accel_arr = np.tile(np.array(accel, dtype=float), (n, 1))
    gyro_arr = np.tile(np.array(gyro, dtype=float), (n, 1))
    return timestamps, accel_arr, gyro_arr


# ---------------------------------------------------------------------------
# SO(3) helper tests
# ---------------------------------------------------------------------------


class TestSkew:
    def test_skew_anti_symmetric(self):
        v = np.array([1.0, 2.0, 3.0])
        K = _skew(v)
        np.testing.assert_allclose(K, -K.T)

    def test_skew_cross_product(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        # a × b should equal [0, 0, 1]
        np.testing.assert_allclose(_skew(a) @ b, np.cross(a, b))


class TestExpSO3:
    def test_zero_vector_gives_identity(self):
        R = _exp_so3(np.zeros(3))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_90_deg_about_z(self):
        phi = np.array([0.0, 0.0, math.pi / 2])
        R = _exp_so3(phi)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_180_deg_about_z(self):
        phi = np.array([0.0, 0.0, math.pi])
        R = _exp_so3(phi)
        expected = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_result_is_rotation_matrix(self):
        phi = np.array([0.1, 0.2, 0.3])
        R = _exp_so3(phi)
        # Orthogonality: R @ R^T = I
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
        # Determinant = 1
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-14)


# ---------------------------------------------------------------------------
# ImuPreintegrator construction tests
# ---------------------------------------------------------------------------


class TestImuPreintegratorInit:
    def test_default_biases_are_zero(self):
        integrator = ImuPreintegrator()
        np.testing.assert_array_equal(integrator.accel_bias, np.zeros(3))
        np.testing.assert_array_equal(integrator.gyro_bias, np.zeros(3))

    def test_custom_biases_stored(self):
        ab = np.array([0.1, 0.2, 0.3])
        gb = np.array([0.01, 0.02, 0.03])
        integrator = ImuPreintegrator(accel_bias=ab, gyro_bias=gb)
        np.testing.assert_allclose(integrator.accel_bias, ab)
        np.testing.assert_allclose(integrator.gyro_bias, gb)

    def test_bias_returns_copy(self):
        integrator = ImuPreintegrator()
        integrator.accel_bias[:] = 99.0  # should not mutate internal state
        np.testing.assert_array_equal(integrator.accel_bias, np.zeros(3))

    def test_invalid_accel_bias_shape_raises(self):
        with pytest.raises(ValueError, match="accel_bias"):
            ImuPreintegrator(accel_bias=np.zeros(4))

    def test_invalid_gyro_bias_shape_raises(self):
        with pytest.raises(ValueError, match="gyro_bias"):
            ImuPreintegrator(gyro_bias=np.zeros(2))


# ---------------------------------------------------------------------------
# Zero-motion tests
# ---------------------------------------------------------------------------


class TestZeroMotion:
    def setup_method(self):
        self.integrator = ImuPreintegrator()
        self.ts, self.a, self.w = _make_inputs(5)

    def test_delta_rotation_is_identity(self):
        result = self.integrator.integrate(self.ts, self.a, self.w)
        np.testing.assert_allclose(result.delta_rotation, np.eye(3), atol=1e-15)

    def test_delta_velocity_is_zero(self):
        result = self.integrator.integrate(self.ts, self.a, self.w)
        np.testing.assert_allclose(result.delta_velocity, np.zeros(3), atol=1e-15)

    def test_delta_position_is_zero(self):
        result = self.integrator.integrate(self.ts, self.a, self.w)
        np.testing.assert_allclose(result.delta_position, np.zeros(3), atol=1e-15)

    def test_duration(self):
        result = self.integrator.integrate(self.ts, self.a, self.w)
        np.testing.assert_allclose(result.duration, 0.4)  # 4 steps × 0.1 s

    def test_num_samples(self):
        result = self.integrator.integrate(self.ts, self.a, self.w)
        assert result.num_samples == 5


# ---------------------------------------------------------------------------
# Constant-acceleration tests (no rotation)
# ---------------------------------------------------------------------------


class TestConstantAcceleration:
    """Constant accel along x, zero gyro – verify kinematic equations."""

    def setup_method(self):
        self.integrator = ImuPreintegrator()
        # 3 samples, dt = 0.1 s, accel = [1, 0, 0] m/s²
        self.ts, self.a, self.w = _make_inputs(3, dt=0.1, accel=(1.0, 0.0, 0.0))

    def test_velocity(self):
        result = self.integrator.integrate(self.ts, self.a, self.w)
        # v = a * t = 1 * 0.2 = 0.2 m/s
        np.testing.assert_allclose(result.delta_velocity, [0.2, 0.0, 0.0], atol=1e-12)

    def test_position(self):
        result = self.integrator.integrate(self.ts, self.a, self.w)
        # p = 0.5 * a * t² = 0.5 * 1 * 0.04 = 0.02 m
        np.testing.assert_allclose(result.delta_position, [0.02, 0.0, 0.0], atol=1e-12)

    def test_rotation_unchanged(self):
        result = self.integrator.integrate(self.ts, self.a, self.w)
        np.testing.assert_allclose(result.delta_rotation, np.eye(3), atol=1e-14)


# ---------------------------------------------------------------------------
# Constant-rotation tests (no acceleration)
# ---------------------------------------------------------------------------


class TestConstantRotation:
    """Constant yaw rate (rotation about z), no translation."""

    def test_180_degree_rotation_about_z(self):
        integrator = ImuPreintegrator()
        ts = np.array([0.0, 1.0])
        a = np.zeros((2, 3))
        # pi rad/s about z → after 1 second: 180° rotation
        w = np.tile([0.0, 0.0, math.pi], (2, 1))
        result = integrator.integrate(ts, a, w)
        expected_R = np.array([[-1.0, 0.0, 0.0],
                                [0.0, -1.0, 0.0],
                                [0.0,  0.0, 1.0]])
        np.testing.assert_allclose(result.delta_rotation, expected_R, atol=1e-10)

    def test_360_degree_rotation_returns_identity(self):
        integrator = ImuPreintegrator()
        # Two equal steps each contributing 180°: total 360° → identity
        ts = np.array([0.0, 1.0, 2.0])
        a = np.zeros((3, 3))
        w = np.tile([0.0, 0.0, math.pi], (3, 1))
        result = integrator.integrate(ts, a, w)
        np.testing.assert_allclose(result.delta_rotation, np.eye(3), atol=1e-10)

    def test_result_is_valid_rotation_matrix(self):
        integrator = ImuPreintegrator()
        ts = np.array([0.0, 0.1, 0.2, 0.3])
        a = np.zeros((4, 3))
        w = np.random.default_rng(0).uniform(-1.0, 1.0, (4, 3))
        result = integrator.integrate(ts, a, w)
        R = result.delta_rotation
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-13)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-13)


# ---------------------------------------------------------------------------
# Bias correction tests
# ---------------------------------------------------------------------------


class TestBiasCorrection:
    def test_accel_bias_corrected(self):
        """Measurements with a known bias, when the bias is set, should give
        the same result as zero-bias measurements without the bias added."""
        bias = np.array([0.5, 0.0, 0.0])
        ts = np.array([0.0, 0.1, 0.2])
        # "True" accel = [1, 0, 0]; biased measurement = [1.5, 0, 0]
        a_biased = np.tile([1.5, 0.0, 0.0], (3, 1))
        a_true = np.tile([1.0, 0.0, 0.0], (3, 1))
        w = np.zeros((3, 3))

        result_biased = ImuPreintegrator(accel_bias=bias).integrate(ts, a_biased, w)
        result_true = ImuPreintegrator().integrate(ts, a_true, w)

        np.testing.assert_allclose(result_biased.delta_velocity, result_true.delta_velocity, atol=1e-12)
        np.testing.assert_allclose(result_biased.delta_position, result_true.delta_position, atol=1e-12)

    def test_gyro_bias_corrected(self):
        """Measurements with a known gyro bias should yield the same rotation
        as zero-bias measurements after bias subtraction."""
        bias = np.array([0.0, 0.0, 0.1])
        ts = np.array([0.0, 1.0])
        a = np.zeros((2, 3))
        # True omega = pi rad/s about z; biased = pi + 0.1
        w_biased = np.tile([0.0, 0.0, math.pi + 0.1], (2, 1))
        w_true = np.tile([0.0, 0.0, math.pi], (2, 1))

        result_biased = ImuPreintegrator(gyro_bias=bias).integrate(ts, a, w_biased)
        result_true = ImuPreintegrator().integrate(ts, a, w_true)

        np.testing.assert_allclose(result_biased.delta_rotation, result_true.delta_rotation, atol=1e-12)


# ---------------------------------------------------------------------------
# delta_quaternion tests
# ---------------------------------------------------------------------------


class TestDeltaQuaternion:
    def test_identity_rotation_gives_unit_quaternion(self):
        integrator = ImuPreintegrator()
        ts, a, w = _make_inputs(3)
        result = integrator.integrate(ts, a, w)
        q = result.delta_quaternion
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_quaternion_norm_is_one(self):
        integrator = ImuPreintegrator()
        ts = np.array([0.0, 0.1, 0.2, 0.3])
        a = np.zeros((4, 3))
        w = np.random.default_rng(42).uniform(-1.0, 1.0, (4, 3))
        result = integrator.integrate(ts, a, w)
        q = result.delta_quaternion
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-10)

    def test_quaternion_consistent_with_rotation_matrix(self):
        """R computed from the returned quaternion must match delta_rotation."""
        from scipy.spatial.transform import Rotation as ScipyR

        integrator = ImuPreintegrator()
        ts = np.array([0.0, 0.5, 1.0])
        a = np.zeros((3, 3))
        w = np.tile([0.0, 0.0, math.pi / 2], (3, 1))
        result = integrator.integrate(ts, a, w)

        q = result.delta_quaternion  # [w, x, y, z]
        R_from_q = ScipyR.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()  # scipy wants [x,y,z,w]
        np.testing.assert_allclose(R_from_q, result.delta_rotation, atol=1e-10)


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    def setup_method(self):
        self.integrator = ImuPreintegrator()

    def test_single_sample_raises(self):
        ts = np.array([0.0])
        a = np.zeros((1, 3))
        w = np.zeros((1, 3))
        with pytest.raises(ValueError, match="At least 2"):
            self.integrator.integrate(ts, a, w)

    def test_wrong_accel_shape_raises(self):
        ts = np.array([0.0, 0.1, 0.2])
        a = np.zeros((2, 3))  # mismatched: should be (3, 3)
        w = np.zeros((3, 3))
        with pytest.raises(ValueError, match="accel"):
            self.integrator.integrate(ts, a, w)

    def test_wrong_gyro_shape_raises(self):
        ts = np.array([0.0, 0.1, 0.2])
        a = np.zeros((3, 3))
        w = np.zeros((4, 3))  # mismatched
        with pytest.raises(ValueError, match="gyro"):
            self.integrator.integrate(ts, a, w)

    def test_non_monotonic_timestamps_raises(self):
        ts = np.array([0.0, 0.2, 0.1])  # not strictly increasing
        a = np.zeros((3, 3))
        w = np.zeros((3, 3))
        with pytest.raises(ValueError, match="strictly increasing"):
            self.integrator.integrate(ts, a, w)

    def test_equal_timestamps_raises(self):
        ts = np.array([0.0, 0.1, 0.1])  # duplicate
        a = np.zeros((3, 3))
        w = np.zeros((3, 3))
        with pytest.raises(ValueError, match="strictly increasing"):
            self.integrator.integrate(ts, a, w)


# ---------------------------------------------------------------------------
# PreintegrationResult dataclass tests
# ---------------------------------------------------------------------------


class TestPreintegrationResult:
    def test_fields_accessible(self):
        R = np.eye(3)
        v = np.zeros(3)
        p = np.zeros(3)
        result = PreintegrationResult(
            delta_rotation=R, delta_velocity=v, delta_position=p,
            duration=0.2, num_samples=3,
        )
        assert result.duration == 0.2
        assert result.num_samples == 3
        np.testing.assert_array_equal(result.delta_rotation, R)
        np.testing.assert_array_equal(result.delta_velocity, v)
        np.testing.assert_array_equal(result.delta_position, p)
