"""Tests for camera_intrinsics module."""

import math

import numpy as np
import pytest

from sensor_transposition.camera_intrinsics import (
    camera_matrix,
    distort_point,
    fisheye_distort_point,
    fisheye_focal_length_from_fov,
    fisheye_project_point,
    fisheye_undistort_point,
    fisheye_unproject_pixel,
    focal_length_from_fov,
    focal_length_from_sensor,
    fov_from_focal_length,
    project_point,
    undistort_point,
    unproject_pixel,
)


# ---------------------------------------------------------------------------
# Focal length / FOV derivations
# ---------------------------------------------------------------------------


class TestFocalLengthFromFov:
    def test_90_degree_fov_1024px(self):
        """For 90° FOV the focal length equals half the image size."""
        f = focal_length_from_fov(image_size=1024, fov_deg=90.0)
        assert f == pytest.approx(512.0, rel=1e-6)

    def test_60_degree_fov_640px(self):
        expected = (640 / 2) / math.tan(math.radians(60) / 2)
        assert focal_length_from_fov(640, 60.0) == pytest.approx(expected, rel=1e-6)

    def test_invalid_image_size(self):
        with pytest.raises(ValueError, match="image_size"):
            focal_length_from_fov(0, 90.0)

    def test_invalid_fov_zero(self):
        with pytest.raises(ValueError, match="fov_deg"):
            focal_length_from_fov(640, 0.0)

    def test_invalid_fov_180(self):
        with pytest.raises(ValueError, match="fov_deg"):
            focal_length_from_fov(640, 180.0)


class TestFocalLengthFromSensor:
    def test_standard_values(self):
        """50mm lens on 36mm sensor at 7200px → f = 50*(7200/36) = 10000px."""
        f = focal_length_from_sensor(image_size_px=7200, sensor_size_mm=36.0, focal_length_mm=50.0)
        assert f == pytest.approx(10000.0, rel=1e-6)

    def test_invalid_focal_length(self):
        with pytest.raises(ValueError, match="focal_length_mm"):
            focal_length_from_sensor(1920, 36.0, 0.0)

    def test_invalid_sensor_size(self):
        with pytest.raises(ValueError, match="sensor_size_mm"):
            focal_length_from_sensor(1920, 0.0, 50.0)


class TestFovFromFocalLength:
    def test_round_trip_with_focal_length_from_fov(self):
        fov_in = 75.0
        f = focal_length_from_fov(1920, fov_in)
        fov_out = fov_from_focal_length(f, 1920)
        assert fov_out == pytest.approx(fov_in, rel=1e-6)

    def test_invalid_focal_length(self):
        with pytest.raises(ValueError):
            fov_from_focal_length(0.0, 1920)


# ---------------------------------------------------------------------------
# Camera matrix
# ---------------------------------------------------------------------------


class TestCameraMatrix:
    def test_shape_and_values(self):
        K = camera_matrix(fx=800.0, fy=600.0, cx=320.0, cy=240.0)
        assert K.shape == (3, 3)
        assert K[0, 0] == 800.0
        assert K[1, 1] == 600.0
        assert K[0, 2] == 320.0
        assert K[1, 2] == 240.0
        assert K[2, 2] == 1.0
        assert K[0, 1] == 0.0

    def test_dtype_float64(self):
        K = camera_matrix(800.0, 800.0, 640.0, 360.0)
        assert K.dtype == np.float64


# ---------------------------------------------------------------------------
# Projection and unprojection
# ---------------------------------------------------------------------------


class TestProjectUnproject:
    def setup_method(self):
        self.K = camera_matrix(800.0, 800.0, 640.0, 360.0)

    def test_project_principal_point(self):
        """A point on the optical axis should project to the principal point."""
        u, v = project_point(self.K, [0.0, 0.0, 1.0])
        assert u == pytest.approx(640.0, rel=1e-6)
        assert v == pytest.approx(360.0, rel=1e-6)

    def test_project_offset_point(self):
        u, v = project_point(self.K, [1.0, 0.5, 2.0])
        # u = 800 * (1/2) + 640 = 1040
        # v = 800 * (0.5/2) + 360 = 560
        assert u == pytest.approx(1040.0, rel=1e-6)
        assert v == pytest.approx(560.0, rel=1e-6)

    def test_project_behind_camera_raises(self):
        with pytest.raises(ValueError, match="behind"):
            project_point(self.K, [0.0, 0.0, -1.0])

    def test_unproject_principal_point(self):
        pt = unproject_pixel(self.K, (640.0, 360.0), depth=2.0)
        np.testing.assert_allclose(pt, [0.0, 0.0, 2.0], atol=1e-10)

    def test_project_unproject_round_trip(self):
        """project then unproject should recover the original 3D point."""
        original = np.array([0.5, -0.3, 3.0])
        u, v = project_point(self.K, original)
        recovered = unproject_pixel(self.K, (u, v), depth=original[2])
        np.testing.assert_allclose(recovered, original, atol=1e-8)

    def test_unproject_negative_depth_raises(self):
        with pytest.raises(ValueError, match="depth"):
            unproject_pixel(self.K, (640.0, 360.0), depth=-1.0)


# ---------------------------------------------------------------------------
# Distortion / undistortion
# ---------------------------------------------------------------------------


class TestDistortion:
    def test_zero_coefficients_no_distortion(self):
        pt = np.array([0.1, 0.2])
        pt_d = distort_point(pt, (0.0, 0.0, 0.0, 0.0, 0.0))
        np.testing.assert_allclose(pt_d, pt, atol=1e-10)

    def test_distort_undistort_round_trip(self):
        """Applying distortion then undistortion should recover the original."""
        dist = (-0.3, 0.1, 0.001, -0.001, 0.0)
        pt = np.array([0.15, -0.1])
        pt_d = distort_point(pt, dist)
        pt_u = undistort_point(pt_d, dist)
        np.testing.assert_allclose(pt_u, pt, atol=1e-7)

    def test_radial_distortion_expands_point(self):
        """Positive k1 (barrel) distortion moves points away from centre."""
        dist = (0.5, 0.0, 0.0, 0.0, 0.0)
        pt = np.array([0.1, 0.0])
        pt_d = distort_point(pt, dist)
        assert abs(pt_d[0]) > abs(pt[0])


# ---------------------------------------------------------------------------
# Fisheye / omnidirectional camera model
# ---------------------------------------------------------------------------


class TestFisheyeFocalLengthFromFov:
    def test_180_degree_fov(self):
        """For 180° FOV the focal length equals image_size / π."""
        f = fisheye_focal_length_from_fov(image_size=1000, fov_deg=180.0)
        assert f == pytest.approx(1000.0 / math.pi, rel=1e-6)

    def test_90_degree_fov(self):
        """For 90° FOV: f = (1000/2) / (π/4) = 2000/π."""
        f = fisheye_focal_length_from_fov(image_size=1000, fov_deg=90.0)
        assert f == pytest.approx(1000.0 / (math.pi / 2.0), rel=1e-6)

    def test_270_degree_fov(self):
        """Supports omnidirectional cameras with FOV > 180°."""
        f = fisheye_focal_length_from_fov(image_size=2000, fov_deg=270.0)
        expected = (2000 / 2.0) / (math.radians(270.0) / 2.0)
        assert f == pytest.approx(expected, rel=1e-6)

    def test_invalid_image_size(self):
        with pytest.raises(ValueError, match="image_size"):
            fisheye_focal_length_from_fov(0, 180.0)

    def test_invalid_fov_zero(self):
        with pytest.raises(ValueError, match="fov_deg"):
            fisheye_focal_length_from_fov(1000, 0.0)

    def test_invalid_fov_360(self):
        with pytest.raises(ValueError, match="fov_deg"):
            fisheye_focal_length_from_fov(1000, 360.0)


class TestFisheyeDistortUndistort:
    def test_zero_coefficients_equidistant(self):
        """Zero k-coefficients apply the pure equidistant mapping (r → θ = atan(r))."""
        pt = np.array([0.2, -0.15])
        pt_d = fisheye_distort_point(pt, ())
        r = float(np.linalg.norm(pt))
        theta = math.atan(r)
        expected = pt * (theta / r)
        np.testing.assert_allclose(pt_d, expected, atol=1e-10)

    def test_origin_unchanged(self):
        """A point at the origin remains at the origin."""
        pt = np.array([0.0, 0.0])
        pt_d = fisheye_distort_point(pt, (0.1, -0.05, 0.0, 0.0))
        np.testing.assert_allclose(pt_d, np.array([0.0, 0.0]), atol=1e-10)

    def test_distort_undistort_round_trip(self):
        """Applying distortion then undistortion should recover the original."""
        dist = (0.05, -0.02, 0.003, -0.001)
        pt = np.array([0.3, -0.2])
        pt_d = fisheye_distort_point(pt, dist)
        pt_u = fisheye_undistort_point(pt_d, dist)
        np.testing.assert_allclose(pt_u, pt, atol=1e-7)

    def test_positive_k1_compresses_point(self):
        """Positive k1 in the fisheye model compresses the radius (maps
        larger angles inward), so the distorted radius is smaller."""
        dist = (0.1, 0.0, 0.0, 0.0)
        pt = np.array([0.5, 0.0])
        pt_d = fisheye_distort_point(pt, dist)
        # theta = atan(0.5) ≈ 0.4636 rad; with k1>0, theta_d < theta_d_undistorted
        # so the distorted radius should be smaller than tan(theta) = 0.5
        assert abs(pt_d[0]) < abs(pt[0])

    def test_undistort_reverses_distort_zero_coefficients(self):
        """undistort(distort(pt)) == pt even with zero coefficients."""
        pt = np.array([0.1, 0.3])
        pt_d = fisheye_distort_point(pt, ())
        pt_u = fisheye_undistort_point(pt_d, ())
        np.testing.assert_allclose(pt_u, pt, atol=1e-8)


class TestFisheyeProjectUnproject:
    def setup_method(self):
        f = fisheye_focal_length_from_fov(image_size=1000, fov_deg=180.0)
        self.K = camera_matrix(fx=f, fy=f, cx=500.0, cy=500.0)

    def test_project_on_axis(self):
        """A point on the optical axis projects to the principal point."""
        u, v = fisheye_project_point(self.K, [0.0, 0.0, 1.0])
        assert u == pytest.approx(500.0, abs=1e-6)
        assert v == pytest.approx(500.0, abs=1e-6)

    def test_project_90_degrees_off_axis(self):
        """A point at exactly 90° from the optical axis maps to the image edge."""
        # FOV=180° → f = image_size/π.  Point at θ=90° → r = f*π/2 = image_size/2.
        u, v = fisheye_project_point(self.K, [1.0, 0.0, 0.0])
        assert u == pytest.approx(500.0 + 500.0, abs=1e-4)
        assert v == pytest.approx(500.0, abs=1e-4)

    def test_project_behind_camera(self):
        """Points behind the camera (Z < 0) can be projected for wide-FOV lenses."""
        # For a 270° FOV camera, a point at Z < 0 is within the field of view.
        f270 = fisheye_focal_length_from_fov(image_size=2000, fov_deg=270.0)
        K270 = camera_matrix(fx=f270, fy=f270, cx=1000.0, cy=1000.0)
        # Point at θ = 135° (directly behind and to the side)
        u, v = fisheye_project_point(K270, [1.0, 0.0, -1.0])
        assert math.isfinite(u) and math.isfinite(v)

    def test_project_unproject_round_trip_no_distortion(self):
        """project then unproject should recover the original point."""
        pt = np.array([0.3, -0.2, 1.0])
        pt_normed = pt / np.linalg.norm(pt)
        depth = np.linalg.norm(pt)
        u, v = fisheye_project_point(self.K, pt)
        recovered = fisheye_unproject_pixel(self.K, (u, v), depth=depth)
        np.testing.assert_allclose(recovered, pt, atol=1e-6)

    def test_project_unproject_round_trip_with_distortion(self):
        """Round-trip with Kannala-Brandt distortion coefficients."""
        dist = (0.05, -0.02, 0.003, -0.001)
        pt = np.array([0.5, 0.3, 2.0])
        depth = float(np.linalg.norm(pt))
        u, v = fisheye_project_point(self.K, pt, dist)
        recovered = fisheye_unproject_pixel(self.K, (u, v), depth=depth, dist_coeffs=dist)
        np.testing.assert_allclose(recovered, pt, atol=1e-6)

    def test_unproject_invalid_depth(self):
        with pytest.raises(ValueError, match="depth"):
            fisheye_unproject_pixel(self.K, (500.0, 500.0), depth=-1.0)

    def test_project_invalid_shape(self):
        with pytest.raises(ValueError, match="shape"):
            fisheye_project_point(self.K, [0.0, 1.0])
