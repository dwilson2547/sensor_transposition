"""Tests for radar odometry: Doppler ego-velocity estimation and ICP scan matching."""

import math

import numpy as np
import pytest

from sensor_transposition.radar.radar import RADAR_DETECTION_DTYPE
from sensor_transposition.radar.radar_odometry import (
    EgoVelocityResult,
    RadarOdometer,
    estimate_ego_velocity,
    integrate_radar_odometry,
    radar_scan_match,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detections(
    azimuths_deg: list[float],
    elevations_deg: list[float],
    velocities: list[float],
    ranges: list[float] | None = None,
    snrs: list[float] | None = None,
) -> np.ndarray:
    """Build a structured radar detection array from lists of values."""
    n = len(azimuths_deg)
    if ranges is None:
        ranges = [10.0] * n
    if snrs is None:
        snrs = [20.0] * n
    out = np.empty(n, dtype=RADAR_DETECTION_DTYPE)
    out["range"] = np.array(ranges, dtype=np.float32)
    out["azimuth"] = np.array(azimuths_deg, dtype=np.float32)
    out["elevation"] = np.array(elevations_deg, dtype=np.float32)
    out["velocity"] = np.array(velocities, dtype=np.float32)
    out["snr"] = np.array(snrs, dtype=np.float32)
    return out


def _detections_from_ego_velocity(
    v_ego: np.ndarray,
    azimuths_deg: list[float],
    elevations_deg: list[float],
    snr: float = 20.0,
) -> np.ndarray:
    """Synthesise detections consistent with a known ego-velocity."""
    az = np.radians(np.array(azimuths_deg))
    el = np.radians(np.array(elevations_deg))
    cos_el = np.cos(el)
    dx = cos_el * np.cos(az)
    dy = cos_el * np.sin(az)
    dz = np.sin(el)
    D = np.column_stack([dx, dy, dz])
    # velocity_i = -(d_i · v_ego)
    vel = -(D @ v_ego)
    return _make_detections(
        azimuths_deg, elevations_deg, vel.tolist(), snrs=[snr] * len(azimuths_deg)
    )


def _make_point_cloud(n: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3)).astype(float) * 5.0


# ---------------------------------------------------------------------------
# estimate_ego_velocity
# ---------------------------------------------------------------------------


class TestEstimateEgoVelocity:
    def test_returns_ego_velocity_result(self):
        dets = _detections_from_ego_velocity(
            np.array([1.0, 0.0, 0.0]),
            [0.0, 90.0, 180.0, -90.0],
            [0.0, 0.0, 0.0, 0.0],
        )
        result = estimate_ego_velocity(dets)
        assert isinstance(result, EgoVelocityResult)

    def test_pure_forward_motion(self):
        """Pure forward motion (vx=2) → detections ahead approaches, behind recedes."""
        v_ego = np.array([2.0, 0.0, 0.0])
        azimuths = [0.0, 90.0, 180.0, -90.0, 45.0, -45.0]
        elevs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        dets = _detections_from_ego_velocity(v_ego, azimuths, elevs)
        result = estimate_ego_velocity(dets)
        assert result.valid
        np.testing.assert_allclose(result.velocity, v_ego, atol=1e-6)

    def test_lateral_motion(self):
        """Pure lateral motion (vy=3)."""
        v_ego = np.array([0.0, 3.0, 0.0])
        azimuths = [0.0, 90.0, 180.0, -90.0, 45.0, -45.0, 135.0, -135.0]
        elevs = [0.0] * 8
        dets = _detections_from_ego_velocity(v_ego, azimuths, elevs)
        result = estimate_ego_velocity(dets)
        assert result.valid
        np.testing.assert_allclose(result.velocity, v_ego, atol=1e-6)

    def test_3d_velocity(self):
        """3-D velocity with non-zero vertical component."""
        v_ego = np.array([1.5, -0.5, 0.8])
        azimuths = [0.0, 90.0, 180.0, -90.0, 45.0, -45.0]
        elevs = [10.0, 0.0, -10.0, 5.0, 15.0, -5.0]
        dets = _detections_from_ego_velocity(v_ego, azimuths, elevs)
        result = estimate_ego_velocity(dets)
        assert result.valid
        np.testing.assert_allclose(result.velocity, v_ego, atol=1e-6)

    def test_velocity_shape(self):
        dets = _detections_from_ego_velocity(
            np.array([1.0, 0.0, 0.0]),
            [0.0, 90.0, 180.0],
            [0.0, 0.0, 0.0],
        )
        result = estimate_ego_velocity(dets)
        assert result.velocity.shape == (3,)

    def test_residuals_shape(self):
        azimuths = [0.0, 90.0, 180.0, -90.0]
        dets = _detections_from_ego_velocity(
            np.array([1.0, 0.5, 0.0]), azimuths, [0.0] * 4
        )
        result = estimate_ego_velocity(dets)
        assert result.residuals.shape == (4,)

    def test_num_inliers(self):
        azimuths = [0.0, 90.0, 180.0, -90.0, 45.0]
        dets = _detections_from_ego_velocity(
            np.array([1.0, 0.0, 0.0]), azimuths, [0.0] * 5
        )
        result = estimate_ego_velocity(dets)
        assert result.num_inliers == 5

    def test_snr_filter_removes_low_snr(self):
        v_ego = np.array([1.0, 0.0, 0.0])
        azimuths = [0.0, 90.0, 180.0, -90.0, 45.0]
        elevs = [0.0] * 5
        # Last detection has very low SNR.
        snrs = [20.0, 20.0, 20.0, 20.0, 2.0]
        dets = _detections_from_ego_velocity(v_ego, azimuths, elevs)
        dets["snr"] = np.array(snrs, dtype=np.float32)
        result = estimate_ego_velocity(dets, min_snr=10.0)
        assert result.num_inliers == 4

    def test_too_few_inliers_returns_invalid(self):
        # Only 2 detections after SNR filter (need >= 3).
        dets = _make_detections([0.0, 90.0], [0.0, 0.0], [1.0, 0.0])
        result = estimate_ego_velocity(dets, min_inliers=3)
        assert not result.valid
        assert result.num_inliers == 0

    def test_all_filtered_returns_invalid(self):
        dets = _make_detections([0.0, 90.0, 180.0], [0.0, 0.0, 0.0], [1.0, 0.0, -1.0])
        dets["snr"] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        result = estimate_ego_velocity(dets, min_snr=50.0)
        assert not result.valid

    def test_wrong_dtype_raises(self):
        bad = np.zeros(5, dtype=np.float32)
        with pytest.raises(ValueError, match="RADAR_DETECTION_DTYPE"):
            estimate_ego_velocity(bad)

    def test_stationary_vehicle(self):
        """Zero ego-velocity → all Doppler readings should be zero."""
        v_ego = np.array([0.0, 0.0, 0.0])
        dets = _detections_from_ego_velocity(
            v_ego, [0.0, 60.0, 120.0, 180.0, 240.0, 300.0], [0.0] * 6
        )
        result = estimate_ego_velocity(dets)
        assert result.valid
        np.testing.assert_allclose(result.velocity, np.zeros(3), atol=1e-10)

    def test_residuals_near_zero_for_noiseless_data(self):
        v_ego = np.array([1.0, -1.0, 0.5])
        dets = _detections_from_ego_velocity(
            v_ego,
            [0.0, 45.0, 90.0, 135.0, 180.0, -45.0],
            [5.0, -5.0, 5.0, 0.0, 0.0, 5.0],
        )
        result = estimate_ego_velocity(dets)
        np.testing.assert_allclose(result.residuals, np.zeros(6), atol=1e-6)


# ---------------------------------------------------------------------------
# radar_scan_match
# ---------------------------------------------------------------------------


class TestRadarScanMatch:
    def test_identity_transform_on_identical_clouds(self):
        xyz = _make_point_cloud(30, seed=1)
        result = radar_scan_match(xyz, xyz)
        np.testing.assert_allclose(result.transform, np.eye(4), atol=1e-6)

    def test_recovers_pure_translation(self):
        source = _make_point_cloud(50, seed=2)
        t_true = np.array([1.0, 0.5, -0.3])
        target = source + t_true
        result = radar_scan_match(source, target, max_iterations=100)
        assert result.converged
        np.testing.assert_allclose(
            result.transform[:3, 3], t_true, atol=1e-4
        )

    def test_result_is_icp_result(self):
        from sensor_transposition.lidar.scan_matching import IcpResult
        xyz = _make_point_cloud(20, seed=3)
        result = radar_scan_match(xyz, xyz)
        assert isinstance(result, IcpResult)

    def test_max_correspondence_dist_kwarg(self):
        """Passing max_correspondence_dist should not raise."""
        xyz = _make_point_cloud(20, seed=4)
        result = radar_scan_match(xyz, xyz, max_correspondence_dist=10.0)
        assert result.mean_squared_error < 1e-8


# ---------------------------------------------------------------------------
# RadarOdometer – construction
# ---------------------------------------------------------------------------


class TestRadarOdometerInit:
    def test_default_pose_is_identity(self):
        odom = RadarOdometer()
        np.testing.assert_array_equal(odom.pose, np.eye(4))

    def test_transforms_empty_on_init(self):
        odom = RadarOdometer()
        assert odom.transforms == []

    def test_zero_max_iterations_raises(self):
        with pytest.raises(ValueError, match="max_iterations"):
            RadarOdometer(max_iterations=0)

    def test_zero_tolerance_raises(self):
        with pytest.raises(ValueError, match="tolerance"):
            RadarOdometer(tolerance=0.0)


# ---------------------------------------------------------------------------
# RadarOdometer – first frame
# ---------------------------------------------------------------------------


class TestRadarOdometerFirstFrame:
    def test_first_frame_returns_none(self):
        odom = RadarOdometer()
        xyz = _make_point_cloud(20, seed=5)
        result = odom.add_frame(xyz, timestamp=0.0)
        assert result is None

    def test_pose_unchanged_after_first_frame(self):
        odom = RadarOdometer()
        xyz = _make_point_cloud(20, seed=6)
        odom.add_frame(xyz, timestamp=0.0)
        np.testing.assert_array_equal(odom.pose, np.eye(4))

    def test_wrong_xyz_shape_raises(self):
        odom = RadarOdometer()
        with pytest.raises(ValueError, match="\\(N, 3\\)"):
            odom.add_frame(np.zeros((5, 2)), timestamp=0.0)

    def test_empty_xyz_raises(self):
        odom = RadarOdometer()
        with pytest.raises(ValueError):
            odom.add_frame(np.zeros((0, 3)), timestamp=0.0)


# ---------------------------------------------------------------------------
# RadarOdometer – consecutive frames
# ---------------------------------------------------------------------------


class TestRadarOdometerConsecutiveFrames:
    def test_second_frame_returns_icp_result(self):
        from sensor_transposition.lidar.scan_matching import IcpResult
        odom = RadarOdometer()
        xyz = _make_point_cloud(30, seed=7)
        odom.add_frame(xyz, timestamp=0.0)
        result = odom.add_frame(xyz, timestamp=0.1)
        assert isinstance(result, IcpResult)

    def test_identical_frames_give_identity_pose(self):
        odom = RadarOdometer()
        xyz = _make_point_cloud(40, seed=8)
        odom.add_frame(xyz, timestamp=0.0)
        odom.add_frame(xyz, timestamp=0.1)
        np.testing.assert_allclose(odom.pose, np.eye(4), atol=1e-6)

    def test_translation_accumulates(self):
        """Shift point cloud by a fixed translation and check pose accumulates."""
        odom = RadarOdometer(max_correspondence_dist=20.0, max_iterations=100)
        base = _make_point_cloud(60, seed=9)
        t = np.array([1.0, 0.0, 0.0])
        odom.add_frame(base, 0.0)
        odom.add_frame(base + t, 0.1)
        odom.add_frame(base + 2 * t, 0.2)
        # After two shifts of +1 in x, the world pose should reflect ~+2 in x.
        np.testing.assert_allclose(odom.pose[0, 3], 2.0, atol=0.05)

    def test_transforms_list_grows(self):
        odom = RadarOdometer()
        xyz = _make_point_cloud(20, seed=10)
        odom.add_frame(xyz, 0.0)
        odom.add_frame(xyz, 0.1)
        odom.add_frame(xyz, 0.2)
        assert len(odom.transforms) == 2

    def test_transforms_are_copies(self):
        odom = RadarOdometer()
        xyz = _make_point_cloud(20, seed=11)
        odom.add_frame(xyz, 0.0)
        odom.add_frame(xyz, 0.1)
        t1 = odom.transforms[0]
        t1[0, 3] = 9999.0
        assert odom.transforms[0][0, 3] != 9999.0

    def test_pose_is_copy(self):
        odom = RadarOdometer()
        xyz = _make_point_cloud(20, seed=12)
        odom.add_frame(xyz, 0.0)
        p = odom.pose
        p[0, 3] = 9999.0
        assert odom.pose[0, 3] != 9999.0


# ---------------------------------------------------------------------------
# RadarOdometer – reset
# ---------------------------------------------------------------------------


class TestRadarOdometerReset:
    def test_reset_clears_pose(self):
        odom = RadarOdometer()
        xyz = _make_point_cloud(20, seed=13)
        t = np.array([1.0, 0.0, 0.0])
        odom.add_frame(xyz, 0.0)
        odom.add_frame(xyz + t, 0.1)
        odom.reset()
        np.testing.assert_array_equal(odom.pose, np.eye(4))

    def test_reset_clears_transforms(self):
        odom = RadarOdometer()
        xyz = _make_point_cloud(20, seed=14)
        odom.add_frame(xyz, 0.0)
        odom.add_frame(xyz, 0.1)
        odom.reset()
        assert odom.transforms == []

    def test_reset_allows_new_trajectory(self):
        odom = RadarOdometer(max_correspondence_dist=20.0, max_iterations=100)
        base = _make_point_cloud(40, seed=15)
        t = np.array([0.5, 0.0, 0.0])
        odom.add_frame(base, 0.0)
        odom.add_frame(base + t, 0.1)
        odom.reset()
        odom.add_frame(base, 0.0)
        result = odom.add_frame(base + t, 0.1)
        assert result is not None


# ---------------------------------------------------------------------------
# integrate_radar_odometry
# ---------------------------------------------------------------------------


class TestIntegrateRadarOdometry:
    def _make_sequence(self, n_frames: int = 4, seed: int = 20):
        base = _make_point_cloud(40, seed=seed)
        frames = [base + np.array([i * 0.5, 0.0, 0.0]) for i in range(n_frames)]
        timestamps = [float(i) * 0.1 for i in range(n_frames)]
        return frames, timestamps

    def test_returns_pose_and_icp_results(self):
        frames, ts = self._make_sequence(4)
        pose, results = integrate_radar_odometry(
            frames, ts, max_correspondence_dist=20.0, max_iterations=100
        )
        assert pose.shape == (4, 4)
        assert len(results) == 3

    def test_accumulated_translation(self):
        """3 frames, each shifted +1 in x → final pose should show ~+2 in x."""
        frames, ts = self._make_sequence(3)
        pose, _ = integrate_radar_odometry(
            frames, ts, max_correspondence_dist=20.0, max_iterations=100
        )
        np.testing.assert_allclose(pose[0, 3], 1.0, atol=0.05)

    def test_identical_frames_give_identity_pose(self):
        xyz = _make_point_cloud(30, seed=21)
        frames = [xyz] * 3
        ts = [0.0, 0.1, 0.2]
        pose, results = integrate_radar_odometry(frames, ts)
        np.testing.assert_allclose(pose, np.eye(4), atol=1e-6)

    def test_mismatched_lengths_raises(self):
        frames = [_make_point_cloud(20)] * 3
        ts = [0.0, 0.1]
        with pytest.raises(ValueError, match="same length"):
            integrate_radar_odometry(frames, ts)

    def test_too_few_frames_raises(self):
        frames = [_make_point_cloud(20)]
        ts = [0.0]
        with pytest.raises(ValueError, match="At least 2 frames"):
            integrate_radar_odometry(frames, ts)

    def test_functional_matches_class_api(self):
        """integrate_radar_odometry should produce the same result as RadarOdometer."""
        xyz = _make_point_cloud(30, seed=22)
        frames = [xyz, xyz, xyz]
        ts = [0.0, 0.1, 0.2]

        pose_fn, _ = integrate_radar_odometry(frames, ts)

        odom = RadarOdometer()
        for frame, t in zip(frames, ts):
            odom.add_frame(frame, t)

        np.testing.assert_allclose(pose_fn, odom.pose, atol=1e-10)
