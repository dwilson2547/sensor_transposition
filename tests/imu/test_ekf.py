"""Tests for the Error-State EKF (imu/ekf.py)."""

import math

import numpy as np
import pytest

from sensor_transposition.imu.ekf import (
    EkfState,
    ImuEkf,
    _exp_so3,
    _quat_from_rotvec,
    _quat_mult,
    _quat_norm,
    _rot_from_quat,
    _skew,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_state(t: float = 0.0) -> EkfState:
    """Return an EkfState at the origin with identity orientation."""
    return EkfState(timestamp=t)


def _make_ekf(**kwargs) -> ImuEkf:
    return ImuEkf(gravity=np.array([0.0, 0.0, -9.81]), **kwargs)


# ---------------------------------------------------------------------------
# SO(3) / quaternion helper tests
# ---------------------------------------------------------------------------


class TestSkew:
    def test_anti_symmetric(self):
        v = np.array([1.0, 2.0, 3.0])
        K = _skew(v)
        np.testing.assert_allclose(K, -K.T)

    def test_cross_product(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(_skew(a) @ b, np.cross(a, b))


class TestExpSO3:
    def test_zero_vector_gives_identity(self):
        np.testing.assert_allclose(_exp_so3(np.zeros(3)), np.eye(3), atol=1e-15)

    def test_90_deg_about_z(self):
        phi = np.array([0.0, 0.0, math.pi / 2])
        R = _exp_so3(phi)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_result_is_rotation_matrix(self):
        phi = np.array([0.1, 0.2, 0.3])
        R = _exp_so3(phi)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-14)


class TestRotFromQuat:
    def test_identity_quaternion_gives_identity_matrix(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(_rot_from_quat(q), np.eye(3), atol=1e-15)

    def test_90_deg_about_z(self):
        angle = math.pi / 2
        q = np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
        R = _rot_from_quat(q)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_is_rotation_matrix(self):
        angle = math.pi / 3
        q = _quat_from_rotvec(np.array([0.1, 0.2, angle]))
        R = _rot_from_quat(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-14)


class TestQuatMult:
    def test_identity_times_identity(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(_quat_mult(q, q), q)

    def test_90_about_z_twice_is_180_about_z(self):
        # 90° about z
        angle = math.pi / 2
        q90 = np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
        q180_expected = np.array([0.0, 0.0, 0.0, 1.0])
        result = _quat_norm(_quat_mult(q90, q90))
        np.testing.assert_allclose(np.abs(result), np.abs(q180_expected), atol=1e-10)


class TestQuatFromRotvec:
    def test_zero_vector_gives_identity(self):
        q = _quat_from_rotvec(np.zeros(3))
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_unit_norm(self):
        q = _quat_from_rotvec(np.array([0.1, 0.2, 0.3]))
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-12)

    def test_consistent_with_exp_so3(self):
        """R from _exp_so3(phi) should equal R from _rot_from_quat(_quat_from_rotvec(phi))."""
        phi = np.array([0.2, -0.1, 0.5])
        R_exp = _exp_so3(phi)
        R_quat = _rot_from_quat(_quat_from_rotvec(phi))
        np.testing.assert_allclose(R_exp, R_quat, atol=1e-12)


# ---------------------------------------------------------------------------
# EkfState tests
# ---------------------------------------------------------------------------


class TestEkfState:
    def test_defaults(self):
        s = EkfState()
        np.testing.assert_array_equal(s.position, np.zeros(3))
        np.testing.assert_array_equal(s.velocity, np.zeros(3))
        np.testing.assert_array_equal(s.quaternion, [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(s.accel_bias, np.zeros(3))
        np.testing.assert_array_equal(s.gyro_bias, np.zeros(3))
        assert s.covariance.shape == (15, 15)
        assert s.timestamp == 0.0

    def test_rotation_matrix_identity(self):
        s = EkfState()
        np.testing.assert_allclose(s.rotation_matrix, np.eye(3), atol=1e-15)

    def test_rotation_matrix_90_deg_z(self):
        angle = math.pi / 2
        q = np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
        s = EkfState(quaternion=q)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(s.rotation_matrix, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# ImuEkf construction tests
# ---------------------------------------------------------------------------


class TestImuEkfInit:
    def test_default_gravity(self):
        ekf = ImuEkf()
        np.testing.assert_allclose(ekf.gravity, [0.0, 0.0, -9.81])

    def test_custom_gravity(self):
        g = np.array([0.0, 0.0, -9.80665])
        ekf = ImuEkf(gravity=g)
        np.testing.assert_allclose(ekf.gravity, g)

    def test_gravity_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="gravity"):
            ImuEkf(gravity=np.array([0.0, 0.0]))


# ---------------------------------------------------------------------------
# ImuEkf.predict tests
# ---------------------------------------------------------------------------


class TestPredict:
    def test_zero_dt_raises(self):
        ekf = _make_ekf()
        state = _identity_state()
        with pytest.raises(ValueError, match="dt"):
            ekf.predict(state, np.zeros(3), np.zeros(3), 0.0)

    def test_negative_dt_raises(self):
        ekf = _make_ekf()
        state = _identity_state()
        with pytest.raises(ValueError, match="dt"):
            ekf.predict(state, np.zeros(3), np.zeros(3), -0.01)

    def test_timestamp_advances(self):
        ekf = _make_ekf()
        state = _identity_state(t=1.0)
        new_state = ekf.predict(state, np.zeros(3), np.zeros(3), dt=0.1)
        np.testing.assert_allclose(new_state.timestamp, 1.1)

    def test_covariance_grows_with_time(self):
        """Repeated predictions with no updates should increase trace(P)."""
        ekf = _make_ekf()
        state = _identity_state()
        trace_initial = np.trace(state.covariance)
        for _ in range(10):
            state = ekf.predict(state, np.zeros(3), np.zeros(3), dt=0.1)
        assert np.trace(state.covariance) > trace_initial

    def test_covariance_symmetric(self):
        ekf = _make_ekf()
        state = _identity_state()
        new_state = ekf.predict(state, np.array([0.1, 0.0, 0.0]), np.zeros(3), dt=0.1)
        np.testing.assert_allclose(
            new_state.covariance, new_state.covariance.T, atol=1e-15
        )

    def test_zero_input_only_gravity_effect(self):
        """With zero IMU input the position should drop due to gravity (no initial v)."""
        ekf = _make_ekf()
        state = _identity_state()
        # Zero accel (no body-frame force) → gravity acts downward.
        new_state = ekf.predict(state, np.zeros(3), np.zeros(3), dt=1.0)
        # p_z = 0.5 * g_z * t^2 = 0.5 * (-9.81) * 1.0 = -4.905
        np.testing.assert_allclose(new_state.position[2], -4.905, atol=1e-10)
        np.testing.assert_allclose(new_state.velocity[2], -9.81, atol=1e-10)

    def test_constant_accel_x(self):
        """Body-frame accel [a, 0, 0] with identity orientation → world frame."""
        ekf = ImuEkf(gravity=np.zeros(3))  # no gravity for clean kinematics
        state = _identity_state()
        a = 2.0
        dt = 0.5
        new_state = ekf.predict(state, np.array([a, 0.0, 0.0]), np.zeros(3), dt=dt)
        np.testing.assert_allclose(new_state.velocity[0], a * dt, atol=1e-12)
        np.testing.assert_allclose(new_state.position[0], 0.5 * a * dt ** 2, atol=1e-12)

    def test_pure_yaw_rotation(self):
        """Constant yaw rate → quaternion norm preserved and z-rotation accumulated."""
        ekf = ImuEkf(gravity=np.zeros(3))
        state = _identity_state()
        omega_z = math.pi / 2  # 90°/s
        dt = 1.0
        new_state = ekf.predict(state, np.zeros(3), np.array([0.0, 0.0, omega_z]), dt=dt)
        # Quaternion must remain a unit quaternion.
        np.testing.assert_allclose(np.linalg.norm(new_state.quaternion), 1.0, atol=1e-12)
        # Rotation should be 90° about z.
        R = new_state.rotation_matrix
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_bias_correction(self):
        """Predictions with biased input + matching bias parameter == no-bias + true input."""
        ekf = ImuEkf(gravity=np.zeros(3))
        ba = np.array([0.3, 0.0, 0.0])
        true_a = np.array([1.0, 0.0, 0.0])
        biased_a = true_a + ba

        state_true = EkfState(accel_bias=np.zeros(3))
        state_biased = EkfState(accel_bias=ba)

        new_true = ekf.predict(state_true, true_a, np.zeros(3), dt=0.1)
        new_biased = ekf.predict(state_biased, biased_a, np.zeros(3), dt=0.1)

        np.testing.assert_allclose(new_true.velocity, new_biased.velocity, atol=1e-12)
        np.testing.assert_allclose(new_true.position, new_biased.position, atol=1e-12)


# ---------------------------------------------------------------------------
# ImuEkf.position_update tests
# ---------------------------------------------------------------------------


class TestPositionUpdate:
    def test_reduces_position_uncertainty(self):
        ekf = _make_ekf()
        # Inflate covariance so the update has a clear effect.
        state = EkfState(covariance=np.eye(15) * 10.0)
        pos_noise = np.eye(3) * 0.01
        updated = ekf.position_update(state, np.array([1.0, 0.0, 0.0]), pos_noise)
        # Position-block variance should decrease.
        assert np.trace(updated.covariance[0:3, 0:3]) < np.trace(state.covariance[0:3, 0:3])

    def test_position_corrected_toward_measurement(self):
        ekf = _make_ekf()
        state = EkfState(
            position=np.array([0.0, 0.0, 0.0]),
            covariance=np.eye(15) * 5.0,
        )
        meas = np.array([2.0, 0.0, 0.0])
        updated = ekf.position_update(state, meas, noise=np.eye(3) * 0.01)
        # After update, position should have moved toward the measurement.
        assert updated.position[0] > state.position[0]

    def test_covariance_symmetric_after_update(self):
        ekf = _make_ekf()
        state = EkfState(covariance=np.eye(15))
        updated = ekf.position_update(state, np.zeros(3), np.eye(3))
        np.testing.assert_allclose(
            updated.covariance, updated.covariance.T, atol=1e-14
        )

    def test_wrong_position_shape_raises(self):
        ekf = _make_ekf()
        state = _identity_state()
        with pytest.raises(ValueError, match="position"):
            ekf.position_update(state, np.zeros(2), np.eye(3))

    def test_wrong_noise_shape_raises(self):
        ekf = _make_ekf()
        state = _identity_state()
        with pytest.raises(ValueError, match="noise"):
            ekf.position_update(state, np.zeros(3), np.eye(2))

    def test_high_trust_measurement_snaps_position(self):
        """Very low noise → position should closely match the measurement."""
        ekf = _make_ekf()
        state = EkfState(
            position=np.zeros(3),
            covariance=np.eye(15) * 100.0,
        )
        meas = np.array([5.0, 3.0, 1.0])
        updated = ekf.position_update(state, meas, noise=np.eye(3) * 1e-6)
        np.testing.assert_allclose(updated.position, meas, atol=1e-3)


# ---------------------------------------------------------------------------
# ImuEkf.velocity_update tests
# ---------------------------------------------------------------------------


class TestVelocityUpdate:
    def test_velocity_corrected_toward_measurement(self):
        ekf = _make_ekf()
        state = EkfState(
            velocity=np.zeros(3),
            covariance=np.eye(15) * 5.0,
        )
        meas = np.array([3.0, 0.0, 0.0])
        updated = ekf.velocity_update(state, meas, noise=np.eye(3) * 0.01)
        assert updated.velocity[0] > state.velocity[0]

    def test_covariance_symmetric_after_update(self):
        ekf = _make_ekf()
        state = EkfState(covariance=np.eye(15))
        updated = ekf.velocity_update(state, np.zeros(3), np.eye(3))
        np.testing.assert_allclose(
            updated.covariance, updated.covariance.T, atol=1e-14
        )

    def test_wrong_velocity_shape_raises(self):
        ekf = _make_ekf()
        state = _identity_state()
        with pytest.raises(ValueError, match="velocity"):
            ekf.velocity_update(state, np.zeros(4), np.eye(3))

    def test_wrong_noise_shape_raises(self):
        ekf = _make_ekf()
        state = _identity_state()
        with pytest.raises(ValueError, match="noise"):
            ekf.velocity_update(state, np.zeros(3), np.eye(4))

    def test_high_trust_measurement_snaps_velocity(self):
        ekf = _make_ekf()
        state = EkfState(
            velocity=np.zeros(3),
            covariance=np.eye(15) * 100.0,
        )
        meas = np.array([2.0, -1.0, 0.5])
        updated = ekf.velocity_update(state, meas, noise=np.eye(3) * 1e-6)
        np.testing.assert_allclose(updated.velocity, meas, atol=1e-3)


# ---------------------------------------------------------------------------
# ImuEkf.pose_update tests
# ---------------------------------------------------------------------------


class TestPoseUpdate:
    def _identity_pose_noise(self):
        return np.eye(3) * 0.01, np.eye(3) * 1e-4

    def test_position_corrected(self):
        ekf = _make_ekf()
        state = EkfState(
            position=np.zeros(3),
            covariance=np.eye(15) * 5.0,
        )
        p_noise, o_noise = self._identity_pose_noise()
        meas_pos = np.array([1.0, 0.0, 0.0])
        meas_q = np.array([1.0, 0.0, 0.0, 0.0])
        updated = ekf.pose_update(state, meas_pos, meas_q, p_noise, o_noise)
        assert updated.position[0] > state.position[0]

    def test_orientation_corrected(self):
        """Pose update with a 90° yaw measurement should rotate the state."""
        ekf = _make_ekf()
        state = EkfState(covariance=np.eye(15) * 10.0)
        # 90° about z.
        angle = math.pi / 2
        q_meas = np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
        p_noise, o_noise = self._identity_pose_noise()
        updated = ekf.pose_update(
            state, np.zeros(3), q_meas,
            np.eye(3) * 1e-6, np.eye(3) * 1e-6,
        )
        # The updated quaternion should have a non-zero z component.
        assert abs(updated.quaternion[3]) > 0.1

    def test_covariance_symmetric(self):
        ekf = _make_ekf()
        state = EkfState(covariance=np.eye(15))
        p_noise, o_noise = self._identity_pose_noise()
        updated = ekf.pose_update(
            state, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]),
            p_noise, o_noise,
        )
        np.testing.assert_allclose(
            updated.covariance, updated.covariance.T, atol=1e-14
        )

    def test_wrong_position_shape_raises(self):
        ekf = _make_ekf()
        state = _identity_state()
        p_noise, o_noise = self._identity_pose_noise()
        with pytest.raises(ValueError, match="position"):
            ekf.pose_update(state, np.zeros(2), np.array([1, 0, 0, 0]), p_noise, o_noise)

    def test_wrong_quaternion_shape_raises(self):
        ekf = _make_ekf()
        state = _identity_state()
        p_noise, o_noise = self._identity_pose_noise()
        with pytest.raises(ValueError, match="quaternion"):
            ekf.pose_update(state, np.zeros(3), np.array([1, 0, 0]), p_noise, o_noise)

    def test_wrong_position_noise_shape_raises(self):
        ekf = _make_ekf()
        state = _identity_state()
        _, o_noise = self._identity_pose_noise()
        with pytest.raises(ValueError, match="position_noise"):
            ekf.pose_update(
                state, np.zeros(3), np.array([1, 0, 0, 0]),
                np.eye(2), o_noise,
            )

    def test_wrong_orientation_noise_shape_raises(self):
        ekf = _make_ekf()
        state = _identity_state()
        p_noise, _ = self._identity_pose_noise()
        with pytest.raises(ValueError, match="orientation_noise"):
            ekf.pose_update(
                state, np.zeros(3), np.array([1, 0, 0, 0]),
                p_noise, np.eye(2),
            )

    def test_identity_measurement_no_position_change(self):
        """Pose update at origin with identity quaternion should not drift position."""
        ekf = _make_ekf()
        state = EkfState(covariance=np.eye(15) * 1.0)
        updated = ekf.pose_update(
            state,
            np.zeros(3),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.eye(3) * 0.01,
            np.eye(3) * 1e-4,
        )
        np.testing.assert_allclose(updated.position, np.zeros(3), atol=1e-12)


# ---------------------------------------------------------------------------
# Integration test: predict then update converges
# ---------------------------------------------------------------------------


class TestPredictUpdateCycle:
    def test_position_converges_with_repeated_gps_updates(self):
        """Run several predict-update cycles and verify the estimated position
        converges toward the ground-truth GPS measurement."""
        ekf = ImuEkf(gravity=np.zeros(3))
        state = EkfState(
            position=np.zeros(3),
            covariance=np.eye(15) * 1.0,
        )
        true_pos = np.array([3.0, 0.0, 0.0])

        for _ in range(20):
            state = ekf.predict(state, np.zeros(3), np.zeros(3), dt=0.1)
            state = ekf.position_update(state, true_pos, noise=np.eye(3) * 0.1)

        np.testing.assert_allclose(state.position, true_pos, atol=0.1)

    def test_covariance_decreases_with_updates(self):
        """After several pose updates the covariance trace should decrease."""
        ekf = _make_ekf()
        state = EkfState(covariance=np.eye(15) * 10.0)
        p_noise = np.eye(3) * 0.01
        o_noise = np.eye(3) * 1e-4
        q_id = np.array([1.0, 0.0, 0.0, 0.0])

        trace_initial = np.trace(state.covariance)
        for _ in range(5):
            state = ekf.predict(state, np.zeros(3), np.zeros(3), dt=0.1)
            state = ekf.pose_update(state, np.zeros(3), q_id, p_noise, o_noise)

        assert np.trace(state.covariance) < trace_initial


# ---------------------------------------------------------------------------
# Import from package namespace
# ---------------------------------------------------------------------------


class TestPackageImport:
    def test_import_from_imu_package(self):
        from sensor_transposition.imu import EkfState, ImuEkf  # noqa: F401

    def test_import_from_imu_ekf(self):
        from sensor_transposition.imu.ekf import EkfState, ImuEkf  # noqa: F401
