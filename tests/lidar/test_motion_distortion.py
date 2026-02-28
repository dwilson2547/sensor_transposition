"""Tests for LiDAR motion-distortion correction (deskewing)."""

import math

import numpy as np
import pytest

from sensor_transposition.lidar.motion_distortion import (
    _build_trajectory,
    _exp_so3,
    _skew,
    deskew_scan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_imu(
    n: int,
    dt: float = 0.01,
    accel=(0.0, 0.0, 0.0),
    gyro=(0.0, 0.0, 0.0),
    t0: float = 0.0,
):
    """Build constant-value IMU arrays over ``n`` samples at ``dt`` intervals."""
    times = t0 + np.arange(n, dtype=float) * dt
    a = np.tile(np.array(accel, dtype=float), (n, 1))
    w = np.tile(np.array(gyro, dtype=float), (n, 1))
    return times, a, w


def _rotation_z(angle_rad: float) -> np.ndarray:
    """3×3 rotation matrix about the z-axis."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# SO(3) helpers
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


# ---------------------------------------------------------------------------
# _build_trajectory
# ---------------------------------------------------------------------------


class TestBuildTrajectory:
    def test_identity_at_first_sample(self):
        times, a, w = _make_imu(5)
        rotations, positions = _build_trajectory(
            times, a, w,
            np.zeros(3), np.zeros(3),
            np.zeros(3), np.zeros(3),
        )
        np.testing.assert_allclose(rotations[0], np.eye(3), atol=1e-15)
        np.testing.assert_allclose(positions[0], np.zeros(3), atol=1e-15)

    def test_output_shapes(self):
        times, a, w = _make_imu(10)
        rotations, positions = _build_trajectory(
            times, a, w,
            np.zeros(3), np.zeros(3),
            np.zeros(3), np.zeros(3),
        )
        assert rotations.shape == (10, 3, 3)
        assert positions.shape == (10, 3)

    def test_all_rotations_are_valid(self):
        """Each rotation matrix must be orthonormal with det=+1."""
        times, a, w = _make_imu(20, gyro=(0.1, 0.2, 0.3))
        rotations, _ = _build_trajectory(
            times, a, w,
            np.zeros(3), np.zeros(3),
            np.zeros(3), np.zeros(3),
        )
        for R in rotations:
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)

    def test_zero_inputs_no_motion(self):
        """With zero accel, zero gyro, and zero gravity, positions stay at origin."""
        times, a, w = _make_imu(5)
        rotations, positions = _build_trajectory(
            times, a, w,
            np.zeros(3), np.zeros(3),
            np.zeros(3), np.zeros(3),
        )
        np.testing.assert_allclose(positions, np.zeros((5, 3)), atol=1e-15)
        for R in rotations:
            np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_constant_velocity(self):
        """With zero accel+gyro+gravity but non-zero initial velocity, positions
        should grow linearly."""
        v0 = np.array([1.0, 0.0, 0.0])
        times, a, w = _make_imu(5, dt=0.1)
        _, positions = _build_trajectory(
            times, a, w,
            np.zeros(3), np.zeros(3),
            np.zeros(3), v0,
        )
        expected_x = np.arange(5) * 0.1  # 1 m/s × k×0.1 s
        np.testing.assert_allclose(positions[:, 0], expected_x, atol=1e-12)
        np.testing.assert_allclose(positions[:, 1:], 0.0, atol=1e-12)

    def test_pure_yaw_rotation(self):
        """Constant yaw-rate should accumulate rotation about z."""
        omega_z = math.pi / 2  # 90 deg/s
        dt = 0.1
        n = 11  # 1 second total
        times, a, w = _make_imu(n, dt=dt, gyro=(0.0, 0.0, omega_z))
        rotations, _ = _build_trajectory(
            times, a, w,
            np.zeros(3), np.zeros(3),
            np.zeros(3), np.zeros(3),
        )
        # After 10 steps × 0.1 s × π/2 rad/s = π/2 rad total
        R_final = rotations[-1]
        expected = _rotation_z(omega_z * dt * (n - 1))
        np.testing.assert_allclose(R_final, expected, atol=1e-6)

    def test_bias_subtraction(self):
        """Gyro bias should be subtracted: integration with (true + bias) input
        and matching gyro_bias should equal integration with true input only."""
        true_gyro = np.array([0.1, 0.0, 0.0])
        bias = np.array([0.05, 0.0, 0.0])
        times, a, _ = _make_imu(10)
        _, w_true = _make_imu(10, gyro=tuple(true_gyro))[1:]
        _, w_biased = _make_imu(10, gyro=tuple(true_gyro + bias))[1:]

        R_true, _ = _build_trajectory(
            times, a, w_true, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
        )
        R_biased, _ = _build_trajectory(
            times, a, w_biased, np.zeros(3), bias, np.zeros(3), np.zeros(3)
        )
        np.testing.assert_allclose(R_true, R_biased, atol=1e-12)


# ---------------------------------------------------------------------------
# deskew_scan – input validation
# ---------------------------------------------------------------------------


class TestDeskewInputValidation:
    def _valid_inputs(self):
        """Return a minimal valid set of inputs."""
        times, a, w = _make_imu(5, dt=0.01, t0=0.0)
        pts = np.ones((3, 3))
        pt_ts = np.array([0.01, 0.02, 0.03])
        return pts, pt_ts, times, a, w, 0.02

    def test_wrong_points_shape_raises(self):
        pts, pt_ts, imu_t, a, w, ref = self._valid_inputs()
        with pytest.raises(ValueError, match="points"):
            deskew_scan(np.ones((5, 4)), pt_ts, imu_t, a, w, ref)

    def test_1d_points_raises(self):
        _, pt_ts, imu_t, a, w, ref = self._valid_inputs()
        with pytest.raises(ValueError, match="points"):
            deskew_scan(np.ones(3), pt_ts, imu_t, a, w, ref)

    def test_point_times_wrong_length_raises(self):
        pts, _, imu_t, a, w, ref = self._valid_inputs()
        with pytest.raises(ValueError, match="point_times"):
            deskew_scan(pts, np.zeros(10), imu_t, a, w, ref)

    def test_too_few_imu_samples_raises(self):
        pts, pt_ts, imu_t, a, w, ref = self._valid_inputs()
        with pytest.raises(ValueError, match="imu_times"):
            deskew_scan(pts, pt_ts, imu_t[:1], a[:1], w[:1], ref)

    def test_imu_accel_wrong_shape_raises(self):
        pts, pt_ts, imu_t, _, w, ref = self._valid_inputs()
        with pytest.raises(ValueError, match="imu_accel"):
            deskew_scan(pts, pt_ts, imu_t, np.zeros((5, 2)), w, ref)

    def test_imu_gyro_wrong_shape_raises(self):
        pts, pt_ts, imu_t, a, _, ref = self._valid_inputs()
        with pytest.raises(ValueError, match="imu_gyro"):
            deskew_scan(pts, pt_ts, imu_t, a, np.zeros((5, 2)), ref)

    def test_non_monotonic_imu_times_raises(self):
        pts, pt_ts, imu_t, a, w, ref = self._valid_inputs()
        imu_t_bad = imu_t.copy()
        imu_t_bad[2] = imu_t_bad[1]  # duplicate timestamp
        with pytest.raises(ValueError, match="strictly increasing"):
            deskew_scan(pts, pt_ts, imu_t_bad, a, w, ref)

    def test_accel_bias_wrong_shape_raises(self):
        pts, pt_ts, imu_t, a, w, ref = self._valid_inputs()
        with pytest.raises(ValueError, match="accel_bias"):
            deskew_scan(pts, pt_ts, imu_t, a, w, ref, accel_bias=np.zeros(2))

    def test_gyro_bias_wrong_shape_raises(self):
        pts, pt_ts, imu_t, a, w, ref = self._valid_inputs()
        with pytest.raises(ValueError, match="gyro_bias"):
            deskew_scan(pts, pt_ts, imu_t, a, w, ref, gyro_bias=np.zeros(4))

    def test_gravity_wrong_shape_raises(self):
        pts, pt_ts, imu_t, a, w, ref = self._valid_inputs()
        with pytest.raises(ValueError, match="gravity"):
            deskew_scan(pts, pt_ts, imu_t, a, w, ref, gravity=np.zeros(2))

    def test_initial_velocity_wrong_shape_raises(self):
        pts, pt_ts, imu_t, a, w, ref = self._valid_inputs()
        with pytest.raises(ValueError, match="initial_velocity"):
            deskew_scan(pts, pt_ts, imu_t, a, w, ref, initial_velocity=np.zeros(6))


# ---------------------------------------------------------------------------
# deskew_scan – output shape and dtype
# ---------------------------------------------------------------------------


class TestDeskewOutputShape:
    def test_output_shape(self):
        times, a, w = _make_imu(10, dt=0.01)
        pts = np.random.default_rng(0).uniform(-5, 5, (50, 3))
        pt_ts = np.linspace(times[0], times[-1], 50)
        out = deskew_scan(pts, pt_ts, times, a, w, ref_time=times[0],
                          gravity=np.zeros(3))
        assert out.shape == (50, 3)

    def test_output_dtype_float(self):
        times, a, w = _make_imu(5)
        pts = np.ones((4, 3), dtype=np.float32)
        pt_ts = np.linspace(times[0], times[-1], 4)
        out = deskew_scan(pts, pt_ts, times, a, w, ref_time=times[0],
                          gravity=np.zeros(3))
        assert out.dtype == np.float64

    def test_single_point(self):
        times, a, w = _make_imu(5)
        pts = np.array([[1.0, 2.0, 3.0]])
        pt_ts = np.array([times[2]])
        out = deskew_scan(pts, pt_ts, times, a, w, ref_time=times[0],
                          gravity=np.zeros(3))
        assert out.shape == (1, 3)


# ---------------------------------------------------------------------------
# deskew_scan – identity / no-motion cases
# ---------------------------------------------------------------------------


class TestDeskewNoMotion:
    """With zero gyro and zero accel (+ gravity = 0), no correction is applied."""

    def test_zero_motion_returns_original_points(self):
        times, a, w = _make_imu(10, dt=0.01)
        pts = np.random.default_rng(1).uniform(-5, 5, (30, 3))
        pt_ts = np.linspace(times[0], times[-1], 30)
        out = deskew_scan(pts, pt_ts, times, a, w, ref_time=times[0],
                          gravity=np.zeros(3))
        np.testing.assert_allclose(out, pts, atol=1e-12)

    def test_all_points_at_ref_time_returns_original(self):
        """If all points are at the reference time, no correction should apply."""
        times, a, w = _make_imu(10)
        pts = np.random.default_rng(2).uniform(-2, 2, (20, 3))
        pt_ts = np.full(20, times[5])  # all at sample 5
        out = deskew_scan(pts, pt_ts, times, a, w, ref_time=times[5],
                          gravity=np.zeros(3))
        np.testing.assert_allclose(out, pts, atol=1e-12)

    def test_ref_at_start(self):
        """ref_time == imu_times[0]: T_ref = I, points should be unchanged for
        zero-motion IMU."""
        times, a, w = _make_imu(8)
        pts = np.eye(3)
        pt_ts = np.array([times[2], times[4], times[6]])
        out = deskew_scan(pts, pt_ts, times, a, w, ref_time=times[0],
                          gravity=np.zeros(3))
        np.testing.assert_allclose(out, pts, atol=1e-12)

    def test_ref_at_end(self):
        """ref_time == imu_times[-1]: T_ref is the final pose (still identity
        for zero-motion), so all points should be unchanged."""
        times, a, w = _make_imu(8)
        pts = np.ones((5, 3))
        pt_ts = np.linspace(times[0], times[-1], 5)
        out = deskew_scan(pts, pt_ts, times, a, w, ref_time=times[-1],
                          gravity=np.zeros(3))
        np.testing.assert_allclose(out, pts, atol=1e-12)


# ---------------------------------------------------------------------------
# deskew_scan – rotation-only correction (primary use case)
# ---------------------------------------------------------------------------


class TestDeskewRotation:
    """Verify that pure rotation in the IMU is correctly removed."""

    def test_constant_yaw_corrects_to_ref_frame(self):
        """Simulate a constant yaw rate over a 100 ms scan.
        The deskewed cloud at the scan-start reference should match the cloud
        manually rotated back to the sensor's orientation at scan start."""
        omega_z = math.pi  # 180 deg/s
        dt = 0.01
        n = 11  # 100 ms
        times, a_arr, w_arr = _make_imu(
            n, dt=dt, gyro=(0.0, 0.0, omega_z)
        )

        # Create a simple cloud: axis-aligned unit vectors.
        pts = np.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]])
        # Each point is sampled at a different time.
        pt_ts = np.array([times[0], times[5], times[10]])
        ref = times[0]

        out = deskew_scan(pts, pt_ts, times, a_arr, w_arr, ref,
                          gravity=np.zeros(3))

        # Point 0 is at ref_time → no correction.
        np.testing.assert_allclose(out[0], pts[0], atol=1e-10)

        # Point 1 is at times[5] (50 ms): body has rotated by omega_z * 0.05 rad.
        # Expected correction: rotate back by that amount.
        R_accumulated = _rotation_z(omega_z * 5 * dt)
        expected_1 = R_accumulated @ pts[1]
        np.testing.assert_allclose(out[1], expected_1, atol=1e-6)

        # Point 2 is at times[10] (100 ms): body has rotated by omega_z * 0.1 rad.
        R_accumulated_2 = _rotation_z(omega_z * 10 * dt)
        expected_2 = R_accumulated_2 @ pts[2]
        np.testing.assert_allclose(out[2], expected_2, atol=1e-6)

    def test_deskew_collapses_rotating_cloud(self):
        """A point cloud that appears distorted due to a known rotation should
        become consistent (all points at the same effective range) after deskewing."""
        omega_z = 0.5  # rad/s
        dt = 0.005
        n = 21  # 100 ms
        times, a_arr, w_arr = _make_imu(n, dt=dt, gyro=(0.0, 0.0, omega_z))

        rng = np.random.default_rng(42)
        # All true points at radius 1 from z-axis.
        angles = np.linspace(0, 2 * math.pi, 20, endpoint=False)
        true_pts = np.column_stack([
            np.cos(angles), np.sin(angles), np.zeros(20)
        ])
        pt_ts = np.linspace(times[0], times[-1], 20)

        # Distort: rotate each point by the accumulated yaw at its acquisition time.
        distorted_pts = np.empty_like(true_pts)
        for i, t in enumerate(pt_ts):
            alpha = (t - times[0]) / (times[-1] - times[0])
            angle = omega_z * (t - times[0])
            R = _rotation_z(angle)
            distorted_pts[i] = R @ true_pts[i]

        corrected = deskew_scan(
            distorted_pts, pt_ts, times, a_arr, w_arr,
            ref_time=times[0], gravity=np.zeros(3),
        )
        # After correction, all points should be back near the unit circle.
        radii = np.linalg.norm(corrected[:, :2], axis=1)
        np.testing.assert_allclose(radii, 1.0, atol=1e-5)

    def test_reverse_deskew_ref_at_end(self):
        """Using ref_time at scan end produces a different but equally valid result."""
        omega_z = 0.3
        dt = 0.01
        n = 11
        times, a_arr, w_arr = _make_imu(n, dt=dt, gyro=(0.0, 0.0, omega_z))
        pts = np.eye(3)
        pt_ts = np.array([times[0], times[5], times[10]])

        out_start = deskew_scan(pts, pt_ts, times, a_arr, w_arr,
                                ref_time=times[0], gravity=np.zeros(3))
        out_end = deskew_scan(pts, pt_ts, times, a_arr, w_arr,
                              ref_time=times[-1], gravity=np.zeros(3))

        # The two deskewed clouds should differ (different reference).
        assert not np.allclose(out_start, out_end, atol=1e-6)
        # But point 2 (at times[10] = ref for out_end) should be unchanged.
        np.testing.assert_allclose(out_end[2], pts[2], atol=1e-10)
        # And point 0 (at times[0] = ref for out_start) should be unchanged.
        np.testing.assert_allclose(out_start[0], pts[0], atol=1e-10)


# ---------------------------------------------------------------------------
# deskew_scan – translation correction
# ---------------------------------------------------------------------------


class TestDeskewTranslation:
    """Verify position correction when initial velocity is provided."""

    def test_pure_translation_corrected(self):
        """Constant velocity (zero gyro, zero accel, known initial velocity)
        should move all points back to the reference position."""
        v0 = np.array([1.0, 0.0, 0.0])   # 1 m/s along x
        dt = 0.01
        n = 11  # 100 ms window
        times, a_arr, w_arr = _make_imu(n, dt=dt)
        ref = times[0]

        pts = np.tile(np.array([5.0, 0.0, 0.0]), (n, 1))  # cloud at x=5
        pt_ts = times.copy()

        out = deskew_scan(
            pts, pt_ts, times, a_arr, w_arr, ref,
            gravity=np.zeros(3),
            initial_velocity=v0,
        )
        # With gravity=0 and zero accel, position[k] = v0 * (times[k] - times[0]).
        # At ref (times[0]): p_ref = 0, at times[k]: p_k = v0 * k * dt.
        # t_rel[k] = R_ref.T @ (p_k - 0) = p_k (R_ref = I).
        # p_corrected[k] = pts[k] + t_rel[k] = [5, 0, 0] + [k*dt, 0, 0].
        expected_x = 5.0 + np.arange(n) * dt
        np.testing.assert_allclose(out[:, 0], expected_x, atol=1e-12)
        np.testing.assert_allclose(out[:, 1:], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# deskew_scan – edge cases
# ---------------------------------------------------------------------------


class TestDeskewEdgeCases:
    def test_points_outside_imu_window_clamped(self):
        """Points before/after the IMU window should not raise; they are clamped
        to the nearest boundary and receive the boundary correction (identity
        for zero-motion IMU)."""
        times, a, w = _make_imu(5, dt=0.01, t0=1.0)
        pts = np.ones((3, 3))
        pt_ts = np.array([0.5, 1.02, 2.0])  # 0.5 and 2.0 are outside [1.0, 1.04]
        out = deskew_scan(pts, pt_ts, times, a, w, ref_time=times[0],
                          gravity=np.zeros(3))
        assert out.shape == (3, 3)
        # No-motion: all outputs equal original points.
        np.testing.assert_allclose(out, pts, atol=1e-12)

    def test_many_points_large_cloud(self):
        """Ensure the vectorised path handles large clouds without error."""
        n_imu = 20
        times, a, w = _make_imu(n_imu, dt=0.005)
        rng = np.random.default_rng(99)
        pts = rng.uniform(-10, 10, (5000, 3))
        pt_ts = np.linspace(times[0], times[-1], 5000)
        out = deskew_scan(pts, pt_ts, times, a, w, ref_time=times[0],
                          gravity=np.zeros(3))
        assert out.shape == (5000, 3)
        assert np.all(np.isfinite(out))

    def test_default_gravity_applied(self):
        """Without explicit gravity=zeros, the default [0, 0, -9.81] gravity
        changes the result compared to gravity=zeros for the same inputs."""
        times, a, w = _make_imu(10, dt=0.01)
        pts = np.array([[0.0, 0.0, 1.0]])
        pt_ts = np.array([times[-1]])
        ref = times[0]

        out_with_g = deskew_scan(pts, pt_ts, times, a, w, ref)
        out_no_g = deskew_scan(pts, pt_ts, times, a, w, ref, gravity=np.zeros(3))

        # The z components will differ because gravity accumulates a position offset.
        assert not np.allclose(out_with_g, out_no_g)

    def test_two_imu_samples_minimum(self):
        """Exactly 2 IMU samples should work without error."""
        times, a, w = _make_imu(2, dt=0.1)
        pts = np.array([[1.0, 0.0, 0.0]])
        pt_ts = np.array([times[0]])
        out = deskew_scan(pts, pt_ts, times, a, w, ref_time=times[0],
                          gravity=np.zeros(3))
        assert out.shape == (1, 3)


# ---------------------------------------------------------------------------
# Import from package namespace
# ---------------------------------------------------------------------------


class TestPackageImport:
    def test_import_from_lidar_package(self):
        from sensor_transposition.lidar import deskew_scan  # noqa: F401

    def test_import_from_motion_distortion_module(self):
        from sensor_transposition.lidar.motion_distortion import deskew_scan  # noqa: F401
