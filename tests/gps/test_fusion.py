"""Tests for gps/fusion.py: GpsFuser and hdop_to_noise."""

import math

import numpy as np
import pytest

from sensor_transposition.gps.fusion import GpsFuser, hdop_to_noise
from sensor_transposition.gps.nmea import GgaFix, RmcFix
from sensor_transposition.frame_pose import FramePose, FramePoseSequence
from sensor_transposition.imu.ekf import ImuEkf, EkfState

# ---------------------------------------------------------------------------
# Reference location: Trafalgar Square, London (approximate)
# ---------------------------------------------------------------------------

_LAT = 51.5080
_LON = -0.1281
_ALT = 10.0


def _make_gga(lat: float = _LAT, lon: float = _LON, alt: float = _ALT,
              hdop: float = 1.0) -> GgaFix:
    return GgaFix(
        timestamp="120000.00",
        latitude=lat,
        longitude=lon,
        fix_quality=1,
        num_satellites=8,
        hdop=hdop,
        altitude=alt,
        geoid_separation=47.0,
    )


def _make_rmc(lat: float = _LAT, lon: float = _LON) -> RmcFix:
    return RmcFix(
        timestamp="120000.00",
        status="A",
        latitude=lat,
        longitude=lon,
        speed_knots=0.0,
        course=0.0,
        date="010124",
    )


# ===========================================================================
# hdop_to_noise
# ===========================================================================


class TestHdopToNoise:
    def test_returns_3x3_array(self):
        R = hdop_to_noise(1.0)
        assert R.shape == (3, 3)

    def test_diagonal_matrix(self):
        R = hdop_to_noise(1.0)
        off_diag = R.copy()
        np.fill_diagonal(off_diag, 0.0)
        np.testing.assert_array_equal(off_diag, np.zeros((3, 3)))

    def test_hdop1_default_sigma(self):
        """HDOP=1 → horizontal variance = (3.0 m)² = 9.0."""
        R = hdop_to_noise(1.0, base_sigma_m=3.0, vertical_sigma_m=5.0)
        assert R[0, 0] == pytest.approx(9.0)
        assert R[1, 1] == pytest.approx(9.0)
        assert R[2, 2] == pytest.approx(25.0)

    def test_hdop_scales_horizontal_variance(self):
        """Doubling HDOP should quadruple the horizontal variance."""
        R1 = hdop_to_noise(1.0)
        R2 = hdop_to_noise(2.0)
        assert R2[0, 0] == pytest.approx(4.0 * R1[0, 0])
        assert R2[1, 1] == pytest.approx(4.0 * R1[1, 1])

    def test_vertical_variance_constant(self):
        """Vertical variance must not change with HDOP."""
        R1 = hdop_to_noise(1.0, vertical_sigma_m=5.0)
        R2 = hdop_to_noise(5.0, vertical_sigma_m=5.0)
        assert R1[2, 2] == pytest.approx(R2[2, 2])

    def test_rtk_small_sigma(self):
        """RTK sigma of 0.02 m → variance = 0.0004 m²."""
        R = hdop_to_noise(1.0, base_sigma_m=0.02, vertical_sigma_m=0.05)
        assert R[0, 0] == pytest.approx(0.0004, rel=1e-6)
        assert R[2, 2] == pytest.approx(0.0025, rel=1e-6)


# ===========================================================================
# GpsFuser – construction
# ===========================================================================


class TestGpsFuserInit:
    def test_stores_ref_coords(self):
        fuser = GpsFuser(ref_lat=51.5, ref_lon=-0.1, ref_alt=20.0)
        assert fuser.ref_lat == pytest.approx(51.5)
        assert fuser.ref_lon == pytest.approx(-0.1)
        assert fuser.ref_alt == pytest.approx(20.0)

    def test_default_ref_alt_is_zero(self):
        fuser = GpsFuser(ref_lat=0.0, ref_lon=0.0)
        assert fuser.ref_alt == pytest.approx(0.0)


# ===========================================================================
# GpsFuser – fix_to_enu
# ===========================================================================


class TestFixToEnu:
    def test_origin_fix_gives_zero_enu(self):
        """A fix at the reference origin should produce (0, 0, 0)."""
        fuser = GpsFuser(ref_lat=_LAT, ref_lon=_LON, ref_alt=_ALT)
        fix = _make_gga()
        e, n, u = fuser.fix_to_enu(fix)
        assert e == pytest.approx(0.0, abs=0.01)
        assert n == pytest.approx(0.0, abs=0.01)
        assert u == pytest.approx(0.0, abs=0.01)

    def test_north_displacement(self):
        """A fix 100 m north should yield (e≈0, n≈100, u≈0)."""
        fuser = GpsFuser(ref_lat=_LAT, ref_lon=_LON, ref_alt=_ALT)
        lat2 = _LAT + 100.0 / 6_378_137.0 * (180.0 / math.pi)
        fix = _make_gga(lat=lat2, lon=_LON, alt=_ALT)
        e, n, u = fuser.fix_to_enu(fix)
        assert e == pytest.approx(0.0, abs=0.5)
        assert n == pytest.approx(100.0, rel=1e-3)

    def test_east_displacement(self):
        """A fix 100 m east should yield (e≈100, n≈0, u≈0)."""
        fuser = GpsFuser(ref_lat=_LAT, ref_lon=_LON, ref_alt=_ALT)
        lon2 = _LON + 100.0 / (6_378_137.0 * math.cos(math.radians(_LAT)) * math.radians(1.0))
        fix = _make_gga(lat=_LAT, lon=lon2, alt=_ALT)
        e, n, u = fuser.fix_to_enu(fix)
        assert e == pytest.approx(100.0, rel=5e-3)
        assert n == pytest.approx(0.0, abs=0.5)

    def test_altitude_in_up_component(self):
        """An additional 50 m altitude should give u≈50."""
        fuser = GpsFuser(ref_lat=_LAT, ref_lon=_LON, ref_alt=_ALT)
        fix = _make_gga(lat=_LAT, lon=_LON, alt=_ALT + 50.0)
        _, _, u = fuser.fix_to_enu(fix)
        assert u == pytest.approx(50.0, rel=1e-4)

    def test_rmc_up_uses_ref_alt(self):
        """RmcFix has no altitude; Up should be close to zero at the origin."""
        fuser = GpsFuser(ref_lat=_LAT, ref_lon=_LON, ref_alt=_ALT)
        fix = _make_rmc()
        e, n, u = fuser.fix_to_enu(fix)
        # RMC fix at the origin: e≈0, n≈0; up ≈ 0 because ref_alt is used
        assert e == pytest.approx(0.0, abs=0.01)
        assert n == pytest.approx(0.0, abs=0.01)
        assert u == pytest.approx(0.0, abs=0.01)


class TestFixToEnuArray:
    def test_returns_numpy_array(self):
        fuser = GpsFuser(ref_lat=_LAT, ref_lon=_LON, ref_alt=_ALT)
        arr = fuser.fix_to_enu_array(_make_gga())
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)

    def test_matches_fix_to_enu(self):
        fuser = GpsFuser(ref_lat=_LAT, ref_lon=_LON, ref_alt=_ALT)
        fix = _make_gga(lat=_LAT + 0.001, lon=_LON + 0.001, alt=_ALT + 5.0)
        e, n, u = fuser.fix_to_enu(fix)
        arr = fuser.fix_to_enu_array(fix)
        np.testing.assert_allclose(arr, [e, n, u], atol=1e-12)


# ===========================================================================
# GpsFuser – fuse_into_ekf
# ===========================================================================


class TestFuseIntoEkf:
    def test_returns_ekf_state(self):
        fuser = GpsFuser(ref_lat=_LAT, ref_lon=_LON, ref_alt=_ALT)
        ekf   = ImuEkf()
        state = EkfState()
        fix   = _make_gga()
        noise = hdop_to_noise(1.5)
        new_state = fuser.fuse_into_ekf(ekf, state, fix, noise)
        assert isinstance(new_state, EkfState)

    def test_position_updated_toward_measurement(self):
        """After fusing the origin fix, state position should move toward (0,0,0)."""
        fuser = GpsFuser(ref_lat=_LAT, ref_lon=_LON, ref_alt=_ALT)
        ekf   = ImuEkf()
        # Start with a large position error.
        state = EkfState(position=np.array([100.0, 200.0, 50.0]))
        fix   = _make_gga()  # origin → ENU = (0, 0, 0)
        noise = hdop_to_noise(1.0)
        new_state = fuser.fuse_into_ekf(ekf, state, fix, noise)
        # The Kalman update should pull the estimate toward (0, 0, 0).
        old_err = np.linalg.norm(state.position)
        new_err = np.linalg.norm(new_state.position)
        assert new_err < old_err

    def test_covariance_reduced_after_update(self):
        """Fusing a measurement should reduce the position covariance."""
        fuser = GpsFuser(ref_lat=_LAT, ref_lon=_LON, ref_alt=_ALT)
        ekf   = ImuEkf()
        state = EkfState(covariance=np.eye(15) * 100.0)
        fix   = _make_gga()
        noise = hdop_to_noise(1.0)
        new_state = fuser.fuse_into_ekf(ekf, state, fix, noise)
        old_trace = np.trace(state.covariance[0:3, 0:3])
        new_trace = np.trace(new_state.covariance[0:3, 0:3])
        assert new_trace < old_trace

    def test_rmc_fix_also_works(self):
        """fuse_into_ekf should accept RmcFix without raising."""
        fuser = GpsFuser(ref_lat=_LAT, ref_lon=_LON, ref_alt=_ALT)
        ekf   = ImuEkf()
        state = EkfState()
        fix   = _make_rmc()
        noise = hdop_to_noise(2.0)
        new_state = fuser.fuse_into_ekf(ekf, state, fix, noise)
        assert isinstance(new_state, EkfState)


# ===========================================================================
# GpsFuser – fuse_into_sequence
# ===========================================================================


class TestFuseIntoSequence:
    def test_appends_new_pose(self):
        fuser = GpsFuser(ref_lat=_LAT, ref_lon=_LON, ref_alt=_ALT)
        seq = FramePoseSequence()
        fix = _make_gga()
        pose = fuser.fuse_into_sequence(seq, timestamp=0.0, fix=fix)
        assert len(seq) == 1
        assert isinstance(pose, FramePose)

    def test_pose_translation_matches_enu(self):
        fuser = GpsFuser(ref_lat=_LAT, ref_lon=_LON, ref_alt=_ALT)
        seq = FramePoseSequence()
        lat2 = _LAT + 100.0 / 6_378_137.0 * (180.0 / math.pi)
        fix = _make_gga(lat=lat2, lon=_LON, alt=_ALT)
        pose = fuser.fuse_into_sequence(seq, timestamp=0.0, fix=fix)
        e, n, u = fuser.fix_to_enu(fix)
        assert pose.translation[0] == pytest.approx(e, abs=1e-9)
        assert pose.translation[1] == pytest.approx(n, abs=1e-9)
        assert pose.translation[2] == pytest.approx(u, abs=1e-9)

    def test_identity_orientation_on_new_pose(self):
        """A freshly added pose should have an identity quaternion."""
        fuser = GpsFuser(ref_lat=_LAT, ref_lon=_LON, ref_alt=_ALT)
        seq = FramePoseSequence()
        pose = fuser.fuse_into_sequence(seq, timestamp=0.0, fix=_make_gga())
        np.testing.assert_allclose(pose.rotation, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_updates_existing_pose_in_place(self):
        """If a frame already covers the timestamp, its translation is updated."""
        fuser = GpsFuser(ref_lat=_LAT, ref_lon=_LON, ref_alt=_ALT)
        seq = FramePoseSequence(frame_duration=1.0)
        fix1 = _make_gga()
        fuser.fuse_into_sequence(seq, timestamp=0.0, fix=fix1)
        assert len(seq) == 1

        # A second fix with timestamp still within the same frame window.
        lat2 = _LAT + 100.0 / 6_378_137.0 * (180.0 / math.pi)
        fix2 = _make_gga(lat=lat2)
        fuser.fuse_into_sequence(seq, timestamp=0.5, fix=fix2)
        # The sequence should still have only one frame.
        assert len(seq) == 1
        e2, n2, u2 = fuser.fix_to_enu(fix2)
        assert seq.get_pose(0).translation[1] == pytest.approx(n2, abs=1e-9)

    def test_appends_multiple_poses_for_different_windows(self):
        fuser = GpsFuser(ref_lat=_LAT, ref_lon=_LON, ref_alt=_ALT)
        seq = FramePoseSequence(frame_duration=0.1)
        for i in range(5):
            fuser.fuse_into_sequence(seq, timestamp=float(i) * 0.1, fix=_make_gga())
        assert len(seq) == 5

    def test_rmc_fix_appended_to_sequence(self):
        fuser = GpsFuser(ref_lat=_LAT, ref_lon=_LON, ref_alt=_ALT)
        seq = FramePoseSequence()
        pose = fuser.fuse_into_sequence(seq, timestamp=0.0, fix=_make_rmc())
        assert len(seq) == 1
        assert isinstance(pose, FramePose)


# ===========================================================================
# Integration test: GPS → EKF → FramePoseSequence
# ===========================================================================


class TestIntegration:
    def test_pipeline_with_multiple_fixes(self):
        """Simulate a short trajectory with 5 GPS fixes."""
        ref_lat, ref_lon, ref_alt = _LAT, _LON, _ALT
        fuser = GpsFuser(ref_lat=ref_lat, ref_lon=ref_lon, ref_alt=ref_alt)
        ekf   = ImuEkf()
        state = EkfState()
        seq   = FramePoseSequence()

        for i in range(5):
            # Simulate moving north by 10 m per step.
            lat_i = ref_lat + float(i) * 10.0 / 6_378_137.0 * (180.0 / math.pi)
            fix = _make_gga(lat=lat_i)
            noise = hdop_to_noise(fix.hdop)
            state = fuser.fuse_into_ekf(ekf, state, fix, noise)
            fuser.fuse_into_sequence(seq, timestamp=float(i) * 0.1, fix=fix)

        assert len(seq) == 5
        # Each frame should be ~10 m further north than the previous.
        poses = list(seq)
        for j in range(1, 5):
            dn = poses[j].translation[1] - poses[j - 1].translation[1]
            assert dn == pytest.approx(10.0, rel=1e-3)
