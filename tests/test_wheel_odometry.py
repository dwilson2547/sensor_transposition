"""Tests for wheel odometry: differential-drive and Ackermann kinematic models."""

import math

import numpy as np
import pytest

from sensor_transposition.wheel_odometry import (
    AckermannOdometer,
    DifferentialDriveOdometer,
    OdometryResult,
    _integrate_ackermann,
    _integrate_diff_drive,
    _ticks_to_speeds,
    integrate_ackermann,
    integrate_differential_drive,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linspace_ts(duration: float = 1.0, n: int = 11) -> np.ndarray:
    """Return N evenly-spaced timestamps over [0, duration]."""
    return np.linspace(0.0, duration, n)


# ---------------------------------------------------------------------------
# OdometryResult
# ---------------------------------------------------------------------------


class TestOdometryResult:
    def _make(self, x=1.0, y=2.0, theta=0.5) -> OdometryResult:
        return OdometryResult(x=x, y=y, theta=theta, duration=1.0, num_samples=5)

    def test_translation_shape(self):
        r = self._make()
        t = r.translation
        assert t.shape == (3,)

    def test_translation_values(self):
        r = self._make(x=3.0, y=-1.0)
        np.testing.assert_allclose(r.translation, [3.0, -1.0, 0.0])

    def test_rotation_matrix_shape(self):
        r = self._make()
        assert r.rotation_matrix.shape == (3, 3)

    def test_rotation_matrix_is_valid(self):
        r = self._make(theta=math.pi / 4)
        R = r.rotation_matrix
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_rotation_matrix_identity_at_zero(self):
        r = self._make(theta=0.0)
        np.testing.assert_allclose(r.rotation_matrix, np.eye(3), atol=1e-10)

    def test_transform_shape(self):
        r = self._make()
        assert r.transform.shape == (4, 4)

    def test_transform_homogeneous_row(self):
        r = self._make()
        np.testing.assert_allclose(r.transform[3], [0.0, 0.0, 0.0, 1.0])

    def test_transform_rotation_block(self):
        theta = math.pi / 3
        r = self._make(theta=theta)
        T = r.transform
        np.testing.assert_allclose(T[:3, :3], r.rotation_matrix, atol=1e-10)

    def test_transform_translation_block(self):
        r = self._make(x=5.0, y=-2.0)
        T = r.transform
        np.testing.assert_allclose(T[:3, 3], [5.0, -2.0, 0.0])


# ---------------------------------------------------------------------------
# DifferentialDriveOdometer – construction
# ---------------------------------------------------------------------------


class TestDifferentialDriveOdometerInit:
    def test_stores_wheel_base(self):
        odom = DifferentialDriveOdometer(wheel_base=0.54)
        assert odom.wheel_base == pytest.approx(0.54)

    def test_stores_wheel_radius(self):
        odom = DifferentialDriveOdometer(wheel_base=0.54, wheel_radius=0.1)
        assert odom.wheel_radius == pytest.approx(0.1)

    def test_default_wheel_radius(self):
        odom = DifferentialDriveOdometer(wheel_base=0.5)
        assert odom.wheel_radius == pytest.approx(1.0)

    def test_zero_wheel_base_raises(self):
        with pytest.raises(ValueError, match="wheel_base must be positive"):
            DifferentialDriveOdometer(wheel_base=0.0)

    def test_negative_wheel_base_raises(self):
        with pytest.raises(ValueError, match="wheel_base must be positive"):
            DifferentialDriveOdometer(wheel_base=-1.0)

    def test_zero_wheel_radius_raises(self):
        with pytest.raises(ValueError, match="wheel_radius must be positive"):
            DifferentialDriveOdometer(wheel_base=0.5, wheel_radius=0.0)


# ---------------------------------------------------------------------------
# DifferentialDriveOdometer – straight-line motion
# ---------------------------------------------------------------------------


class TestDifferentialDriveStraightLine:
    """Both wheels turning at the same speed → pure forward translation."""

    def _run(
        self,
        speed: float = 1.0,
        duration: float = 1.0,
        n: int = 11,
        wheel_base: float = 0.5,
    ) -> OdometryResult:
        ts = _linspace_ts(duration, n)
        left = np.full(n, speed)
        right = np.full(n, speed)
        odom = DifferentialDriveOdometer(wheel_base=wheel_base)
        return odom.integrate(ts, left, right)

    def test_x_displacement(self):
        result = self._run(speed=1.0, duration=2.0)
        assert result.x == pytest.approx(2.0, abs=1e-9)

    def test_y_displacement_near_zero(self):
        result = self._run(speed=1.0, duration=2.0)
        assert result.y == pytest.approx(0.0, abs=1e-9)

    def test_theta_near_zero(self):
        result = self._run(speed=1.0, duration=2.0)
        assert result.theta == pytest.approx(0.0, abs=1e-9)

    def test_duration_stored(self):
        result = self._run(duration=3.0)
        assert result.duration == pytest.approx(3.0)

    def test_num_samples_stored(self):
        result = self._run(n=21)
        assert result.num_samples == 21

    def test_negative_speed_reverses(self):
        result = self._run(speed=-1.0, duration=1.0)
        assert result.x == pytest.approx(-1.0, abs=1e-9)
        assert result.y == pytest.approx(0.0, abs=1e-9)

    def test_zero_speed_no_motion(self):
        result = self._run(speed=0.0, duration=1.0)
        assert result.x == pytest.approx(0.0, abs=1e-9)
        assert result.y == pytest.approx(0.0, abs=1e-9)
        assert result.theta == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# DifferentialDriveOdometer – turning motion
# ---------------------------------------------------------------------------


class TestDifferentialDriveTurn:
    """One wheel stopped → vehicle pivots around the stopped wheel."""

    def test_spin_in_place(self):
        """Left stopped, right at v: should rotate CW by roughly omega*t."""
        wheel_base = 1.0
        speed = math.pi   # so one full rotation in 2 s
        duration = 2.0
        n = 201
        ts = _linspace_ts(duration, n)
        left = np.zeros(n)
        right = np.full(n, speed)

        odom = DifferentialDriveOdometer(wheel_base=wheel_base)
        result = odom.integrate(ts, left, right)

        # omega = (v_r - v_l) / L = pi/1 = pi rad/s → theta after 2 s = 2*pi
        assert result.theta == pytest.approx(2.0 * math.pi, abs=1e-4)

    def test_circular_arc_closes(self):
        """Equal-speed, symmetric radii: drive a full circle and return to origin."""
        wheel_base = 1.0
        radius = 2.0  # turning radius of the vehicle centre
        # v_l = omega * (r - L/2), v_r = omega * (r + L/2)
        omega = 1.0  # rad/s
        v_l = omega * (radius - wheel_base / 2.0)
        v_r = omega * (radius + wheel_base / 2.0)
        duration = 2.0 * math.pi / omega  # one full circle
        n = 1001
        ts = _linspace_ts(duration, n)
        left = np.full(n, v_l)
        right = np.full(n, v_r)

        odom = DifferentialDriveOdometer(wheel_base=wheel_base)
        result = odom.integrate(ts, left, right)

        # After a full circle the heading increment should be 2π and
        # the position should return to the origin.
        assert result.theta == pytest.approx(2.0 * math.pi, abs=1e-3)
        assert result.x == pytest.approx(0.0, abs=1e-2)
        assert result.y == pytest.approx(0.0, abs=1e-2)


# ---------------------------------------------------------------------------
# DifferentialDriveOdometer – encoder ticks
# ---------------------------------------------------------------------------


class TestDifferentialDriveEncoderTicks:
    def test_straight_line_from_ticks(self):
        """Verify straight-line integration from encoder counts."""
        wheel_radius = 0.1   # 10 cm
        tpr = 360.0          # ticks per revolution
        arc_per_tick = 2.0 * math.pi * wheel_radius / tpr

        n = 11
        ts = _linspace_ts(1.0, n)
        # Move at 1 tick per 0.1 s step → speed = arc_per_tick / 0.1 per step
        ticks_per_step = 10.0
        total_ticks = ticks_per_step * (n - 1)
        left_ticks = np.linspace(0, total_ticks, n)
        right_ticks = np.linspace(0, total_ticks, n)

        expected_distance = total_ticks * arc_per_tick

        odom = DifferentialDriveOdometer(wheel_base=0.5, wheel_radius=wheel_radius)
        result = odom.integrate(
            ts, left_ticks, right_ticks, ticks_per_revolution=tpr
        )

        assert result.x == pytest.approx(expected_distance, rel=1e-4)
        assert result.y == pytest.approx(0.0, abs=1e-9)

    def test_zero_ticks_no_motion(self):
        ts = _linspace_ts(1.0)
        left_ticks = np.zeros(11)
        right_ticks = np.zeros(11)
        odom = DifferentialDriveOdometer(wheel_base=0.5, wheel_radius=0.1)
        result = odom.integrate(
            ts, left_ticks, right_ticks, ticks_per_revolution=360
        )
        assert result.x == pytest.approx(0.0, abs=1e-9)
        assert result.y == pytest.approx(0.0, abs=1e-9)

    def test_invalid_ticks_per_revolution_raises(self):
        ts = _linspace_ts(1.0)
        lw = np.zeros(11)
        rw = np.zeros(11)
        odom = DifferentialDriveOdometer(wheel_base=0.5, wheel_radius=0.1)
        with pytest.raises(ValueError, match="ticks_per_revolution must be positive"):
            odom.integrate(ts, lw, rw, ticks_per_revolution=0.0)


# ---------------------------------------------------------------------------
# DifferentialDriveOdometer – input validation
# ---------------------------------------------------------------------------


class TestDifferentialDriveInputValidation:
    def test_too_few_samples_raises(self):
        odom = DifferentialDriveOdometer(wheel_base=0.5)
        with pytest.raises(ValueError, match="At least 2 samples"):
            odom.integrate(np.array([0.0]), np.array([0.0]), np.array([0.0]))

    def test_wrong_left_wheel_shape_raises(self):
        odom = DifferentialDriveOdometer(wheel_base=0.5)
        ts = _linspace_ts()
        with pytest.raises(ValueError, match="left_wheel"):
            odom.integrate(ts, np.zeros(5), np.zeros(11))

    def test_wrong_right_wheel_shape_raises(self):
        odom = DifferentialDriveOdometer(wheel_base=0.5)
        ts = _linspace_ts()
        with pytest.raises(ValueError, match="right_wheel"):
            odom.integrate(ts, np.zeros(11), np.zeros(3))

    def test_non_increasing_timestamps_raises(self):
        odom = DifferentialDriveOdometer(wheel_base=0.5)
        ts = np.array([0.0, 0.2, 0.1, 0.3])
        with pytest.raises(ValueError, match="strictly increasing"):
            odom.integrate(ts, np.zeros(4), np.zeros(4))

    def test_equal_timestamps_raises(self):
        odom = DifferentialDriveOdometer(wheel_base=0.5)
        ts = np.array([0.0, 0.1, 0.1, 0.3])
        with pytest.raises(ValueError, match="strictly increasing"):
            odom.integrate(ts, np.zeros(4), np.zeros(4))


# ---------------------------------------------------------------------------
# integrate_differential_drive (functional API)
# ---------------------------------------------------------------------------


class TestIntegrateDifferentialDriveFunctional:
    def test_matches_class_api(self):
        ts = _linspace_ts()
        speed = 1.0
        lw = np.full(11, speed)
        rw = np.full(11, speed)
        result_cls = DifferentialDriveOdometer(wheel_base=0.5).integrate(ts, lw, rw)
        result_fn = integrate_differential_drive(ts, lw, rw, wheel_base=0.5)
        assert result_fn.x == pytest.approx(result_cls.x)
        assert result_fn.y == pytest.approx(result_cls.y)
        assert result_fn.theta == pytest.approx(result_cls.theta)


# ---------------------------------------------------------------------------
# AckermannOdometer – construction
# ---------------------------------------------------------------------------


class TestAckermannOdometerInit:
    def test_stores_wheel_base(self):
        odom = AckermannOdometer(wheel_base=2.7)
        assert odom.wheel_base == pytest.approx(2.7)

    def test_zero_wheel_base_raises(self):
        with pytest.raises(ValueError, match="wheel_base must be positive"):
            AckermannOdometer(wheel_base=0.0)

    def test_negative_wheel_base_raises(self):
        with pytest.raises(ValueError, match="wheel_base must be positive"):
            AckermannOdometer(wheel_base=-2.0)


# ---------------------------------------------------------------------------
# AckermannOdometer – straight-line motion
# ---------------------------------------------------------------------------


class TestAckermannStraightLine:
    def _run(
        self,
        speed: float = 1.0,
        duration: float = 1.0,
        n: int = 11,
        wheel_base: float = 2.7,
    ) -> OdometryResult:
        ts = _linspace_ts(duration, n)
        speeds = np.full(n, speed)
        steers = np.zeros(n)
        odom = AckermannOdometer(wheel_base=wheel_base)
        return odom.integrate(ts, speeds, steers)

    def test_x_displacement(self):
        result = self._run(speed=2.0, duration=3.0)
        assert result.x == pytest.approx(6.0, abs=1e-9)

    def test_y_displacement_near_zero(self):
        result = self._run(speed=2.0, duration=3.0)
        assert result.y == pytest.approx(0.0, abs=1e-9)

    def test_theta_near_zero(self):
        result = self._run(speed=2.0, duration=3.0)
        assert result.theta == pytest.approx(0.0, abs=1e-9)

    def test_duration_stored(self):
        result = self._run(duration=5.0)
        assert result.duration == pytest.approx(5.0)

    def test_num_samples_stored(self):
        result = self._run(n=51)
        assert result.num_samples == 51

    def test_reverse_motion(self):
        result = self._run(speed=-1.0, duration=2.0)
        assert result.x == pytest.approx(-2.0, abs=1e-9)
        assert result.y == pytest.approx(0.0, abs=1e-9)

    def test_zero_speed_no_motion(self):
        result = self._run(speed=0.0, duration=1.0)
        assert result.x == pytest.approx(0.0, abs=1e-9)
        assert result.y == pytest.approx(0.0, abs=1e-9)
        assert result.theta == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# AckermannOdometer – turning motion
# ---------------------------------------------------------------------------


class TestAckermannTurn:
    def test_constant_steering_accumulates_heading(self):
        """Constant speed + constant steering → constant angular rate."""
        wheel_base = 2.7
        speed = 1.0
        # Choose steering angle so omega = 0.5 rad/s.
        # omega = v * tan(delta) / L  →  tan(delta) = omega * L / v
        omega = 0.5
        delta = math.atan(omega * wheel_base / speed)
        duration = 2.0
        n = 201
        ts = _linspace_ts(duration, n)
        speeds = np.full(n, speed)
        steers = np.full(n, delta)

        odom = AckermannOdometer(wheel_base=wheel_base)
        result = odom.integrate(ts, speeds, steers)

        expected_theta = omega * duration
        assert result.theta == pytest.approx(expected_theta, rel=1e-3)

    def test_circular_arc_closes(self):
        """Drive a full circle and return to (approximately) the origin."""
        wheel_base = 2.7
        omega = 1.0   # rad/s
        speed = 3.0   # m/s
        # tan(delta) = omega * L / v
        delta = math.atan(omega * wheel_base / speed)
        duration = 2.0 * math.pi / omega   # one full revolution
        n = 2001
        ts = _linspace_ts(duration, n)
        speeds = np.full(n, speed)
        steers = np.full(n, delta)

        odom = AckermannOdometer(wheel_base=wheel_base)
        result = odom.integrate(ts, speeds, steers)

        assert result.theta == pytest.approx(2.0 * math.pi, abs=1e-2)
        assert result.x == pytest.approx(0.0, abs=0.1)
        assert result.y == pytest.approx(0.0, abs=0.1)


# ---------------------------------------------------------------------------
# AckermannOdometer – input validation
# ---------------------------------------------------------------------------


class TestAckermannInputValidation:
    def test_too_few_samples_raises(self):
        odom = AckermannOdometer(wheel_base=2.7)
        with pytest.raises(ValueError, match="At least 2 samples"):
            odom.integrate(
                np.array([0.0]), np.array([0.0]), np.array([0.0])
            )

    def test_wrong_speeds_shape_raises(self):
        odom = AckermannOdometer(wheel_base=2.7)
        ts = _linspace_ts()
        with pytest.raises(ValueError, match="speeds"):
            odom.integrate(ts, np.zeros(5), np.zeros(11))

    def test_wrong_steering_angles_shape_raises(self):
        odom = AckermannOdometer(wheel_base=2.7)
        ts = _linspace_ts()
        with pytest.raises(ValueError, match="steering_angles"):
            odom.integrate(ts, np.zeros(11), np.zeros(3))

    def test_non_increasing_timestamps_raises(self):
        odom = AckermannOdometer(wheel_base=2.7)
        ts = np.array([0.0, 0.2, 0.1, 0.3])
        with pytest.raises(ValueError, match="strictly increasing"):
            odom.integrate(ts, np.zeros(4), np.zeros(4))


# ---------------------------------------------------------------------------
# integrate_ackermann (functional API)
# ---------------------------------------------------------------------------


class TestIntegrateAckermannFunctional:
    def test_matches_class_api(self):
        ts = _linspace_ts()
        speeds = np.full(11, 2.0)
        steers = np.zeros(11)
        result_cls = AckermannOdometer(wheel_base=2.7).integrate(ts, speeds, steers)
        result_fn = integrate_ackermann(ts, speeds, steers, wheel_base=2.7)
        assert result_fn.x == pytest.approx(result_cls.x)
        assert result_fn.y == pytest.approx(result_cls.y)
        assert result_fn.theta == pytest.approx(result_cls.theta)


# ---------------------------------------------------------------------------
# _ticks_to_speeds helper
# ---------------------------------------------------------------------------


class TestTicksToSpeeds:
    def test_constant_rate_gives_constant_speed(self):
        """Uniform tick increments → uniform speed output."""
        wheel_radius = 0.1
        tpr = 360.0
        arc_per_tick = 2.0 * math.pi * wheel_radius / tpr

        n = 6
        dt = 0.1
        dt_all = np.full(n - 1, dt)
        ticks_per_step = 5.0
        ticks = np.arange(n, dtype=float) * ticks_per_step
        expected_speed = (ticks_per_step * arc_per_tick) / dt

        left_s, right_s = _ticks_to_speeds(
            ticks, ticks, dt_all,
            ticks_per_revolution=tpr,
            wheel_radius=wheel_radius,
        )

        np.testing.assert_allclose(left_s[1:], expected_speed, rtol=1e-9)
        np.testing.assert_allclose(right_s[1:], expected_speed, rtol=1e-9)

    def test_output_shapes(self):
        n = 10
        dt_all = np.full(n - 1, 0.1)
        ticks = np.arange(n, dtype=float)
        left_s, right_s = _ticks_to_speeds(
            ticks, ticks, dt_all,
            ticks_per_revolution=100.0,
            wheel_radius=0.05,
        )
        assert left_s.shape == (n,)
        assert right_s.shape == (n,)


# ---------------------------------------------------------------------------
# _integrate_diff_drive and _integrate_ackermann kernels
# ---------------------------------------------------------------------------


class TestIntegrateDiffDriveKernel:
    def test_zero_speeds_no_motion(self):
        dt_all = np.full(9, 0.1)
        left = np.zeros(10)
        right = np.zeros(10)
        x, y, theta = _integrate_diff_drive(dt_all, left, right, wheel_base=0.5)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)
        assert theta == pytest.approx(0.0)

    def test_straight_line_exact(self):
        n = 11
        speed = 2.0
        dt = 0.1
        dt_all = np.full(n - 1, dt)
        left = np.full(n, speed)
        right = np.full(n, speed)
        x, y, theta = _integrate_diff_drive(dt_all, left, right, wheel_base=0.5)
        assert x == pytest.approx(speed * dt * (n - 1), abs=1e-9)
        assert y == pytest.approx(0.0, abs=1e-9)
        assert theta == pytest.approx(0.0, abs=1e-9)


class TestIntegrateAckermannKernel:
    def test_zero_speed_no_motion(self):
        dt_all = np.full(9, 0.1)
        speeds = np.zeros(10)
        steers = np.zeros(10)
        x, y, theta = _integrate_ackermann(dt_all, speeds, steers, wheel_base=2.7)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)
        assert theta == pytest.approx(0.0)

    def test_straight_line_exact(self):
        n = 11
        speed = 3.0
        dt = 0.2
        dt_all = np.full(n - 1, dt)
        speeds = np.full(n, speed)
        steers = np.zeros(n)
        x, y, theta = _integrate_ackermann(dt_all, speeds, steers, wheel_base=2.7)
        assert x == pytest.approx(speed * dt * (n - 1), abs=1e-9)
        assert y == pytest.approx(0.0, abs=1e-9)
        assert theta == pytest.approx(0.0, abs=1e-9)
