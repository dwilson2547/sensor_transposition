"""Tests for sensor_collection module."""

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from sensor_transposition.sensor_collection import (
    CameraIntrinsics,
    GpsParameters,
    ImuParameters,
    RadarParameters,
    Sensor,
    SensorCollection,
    _quaternion_to_rotation_matrix,
    _make_transform,
)


# ---------------------------------------------------------------------------
# CameraIntrinsics tests
# ---------------------------------------------------------------------------


class TestCameraIntrinsics:
    def test_camera_matrix_shape(self):
        ci = CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0, width=1280, height=720)
        K = ci.camera_matrix
        assert K.shape == (3, 3)

    def test_camera_matrix_values(self):
        ci = CameraIntrinsics(fx=800.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
        K = ci.camera_matrix
        assert K[0, 0] == 800.0
        assert K[1, 1] == 600.0
        assert K[0, 2] == 320.0
        assert K[1, 2] == 240.0
        assert K[2, 2] == 1.0
        assert K[0, 1] == 0.0  # no skew

    def test_default_distortion_coefficients(self):
        ci = CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0, width=1280, height=720)
        assert ci.distortion_coefficients == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_round_trip_dict(self):
        ci = CameraIntrinsics(
            fx=1266.4, fy=1266.4, cx=816.0, cy=612.0, width=1632, height=1224,
            distortion_coefficients=[-0.05, 0.08, 0.0, 0.0, -0.03],
        )
        ci2 = CameraIntrinsics.from_dict(ci.to_dict())
        assert ci2.fx == pytest.approx(ci.fx)
        assert ci2.fy == pytest.approx(ci.fy)
        assert ci2.distortion_coefficients == pytest.approx(ci.distortion_coefficients)


# ---------------------------------------------------------------------------
# GpsParameters tests
# ---------------------------------------------------------------------------


class TestGpsParameters:
    def test_defaults(self):
        gps = GpsParameters()
        assert gps.reference_latitude == 0.0
        assert gps.reference_longitude == 0.0
        assert gps.reference_altitude == 0.0
        assert gps.coordinate_frame == "ENU"

    def test_round_trip_dict(self):
        gps = GpsParameters(
            reference_latitude=37.7749,
            reference_longitude=-122.4194,
            reference_altitude=16.0,
            coordinate_frame="ENU",
        )
        gps2 = GpsParameters.from_dict(gps.to_dict())
        assert gps2.reference_latitude == pytest.approx(gps.reference_latitude)
        assert gps2.reference_longitude == pytest.approx(gps.reference_longitude)
        assert gps2.reference_altitude == pytest.approx(gps.reference_altitude)
        assert gps2.coordinate_frame == gps.coordinate_frame

    def test_from_dict_missing_keys_use_defaults(self):
        gps = GpsParameters.from_dict({})
        assert gps.reference_latitude == 0.0
        assert gps.coordinate_frame == "ENU"


# ---------------------------------------------------------------------------
# ImuParameters tests
# ---------------------------------------------------------------------------


class TestImuParameters:
    def test_defaults(self):
        imu = ImuParameters()
        assert imu.accelerometer_noise_density == 0.0
        assert imu.gyroscope_noise_density == 0.0
        assert imu.update_rate == 100.0

    def test_round_trip_dict(self):
        imu = ImuParameters(
            accelerometer_noise_density=0.003924,
            gyroscope_noise_density=0.000205,
            accelerometer_random_walk=0.004330,
            gyroscope_random_walk=0.0000438,
            update_rate=200.0,
        )
        imu2 = ImuParameters.from_dict(imu.to_dict())
        assert imu2.accelerometer_noise_density == pytest.approx(imu.accelerometer_noise_density)
        assert imu2.gyroscope_noise_density == pytest.approx(imu.gyroscope_noise_density)
        assert imu2.update_rate == pytest.approx(imu.update_rate)

    def test_from_dict_missing_keys_use_defaults(self):
        imu = ImuParameters.from_dict({})
        assert imu.update_rate == 100.0


# ---------------------------------------------------------------------------
# RadarParameters tests
# ---------------------------------------------------------------------------


class TestRadarParameters:
    def test_defaults(self):
        radar = RadarParameters()
        assert radar.max_range == 0.0
        assert radar.azimuth_fov == 0.0

    def test_round_trip_dict(self):
        radar = RadarParameters(
            max_range=200.0,
            range_resolution=0.4,
            azimuth_fov=90.0,
            elevation_fov=14.0,
            velocity_resolution=0.1,
        )
        radar2 = RadarParameters.from_dict(radar.to_dict())
        assert radar2.max_range == pytest.approx(radar.max_range)
        assert radar2.range_resolution == pytest.approx(radar.range_resolution)
        assert radar2.azimuth_fov == pytest.approx(radar.azimuth_fov)
        assert radar2.velocity_resolution == pytest.approx(radar.velocity_resolution)

    def test_from_dict_missing_keys_use_defaults(self):
        radar = RadarParameters.from_dict({})
        assert radar.max_range == 0.0


# ---------------------------------------------------------------------------
# Sensor tests
# ---------------------------------------------------------------------------


class TestSensor:
    def _make_sensor(self):
        return Sensor(
            name="front_lidar",
            sensor_type="lidar",
            coordinate_system="FLU",
            translation=[1.0, 0.0, 1.5],
            rotation=[1.0, 0.0, 0.0, 0.0],  # identity
        )

    def test_transform_to_ego_identity_rotation(self):
        s = self._make_sensor()
        T = s.transform_to_ego
        assert T.shape == (4, 4)
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-10)
        np.testing.assert_allclose(T[:3, 3], [1.0, 0.0, 1.5], atol=1e-10)

    def test_transform_homogeneous_row(self):
        s = self._make_sensor()
        T = s.transform_to_ego
        np.testing.assert_allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10)

    def test_round_trip_dict(self):
        s = self._make_sensor()
        d = s.to_dict()
        s2 = Sensor.from_dict(s.name, d)
        assert s2.name == s.name
        assert s2.sensor_type == s.sensor_type
        assert s2.coordinate_system == s.coordinate_system
        assert s2.translation == pytest.approx(s.translation)
        assert s2.rotation == pytest.approx(s.rotation)
        assert s2.intrinsics is None

    def test_round_trip_dict_with_intrinsics(self):
        ci = CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0, width=1280, height=720)
        s = Sensor(
            name="front_camera",
            sensor_type="camera",
            coordinate_system="RDF",
            translation=[1.0, 0.0, 1.4],
            rotation=[1.0, 0.0, 0.0, 0.0],
            intrinsics=ci,
        )
        s2 = Sensor.from_dict(s.name, s.to_dict())
        assert s2.intrinsics is not None
        assert s2.intrinsics.fx == pytest.approx(800.0)

    def test_gps_sensor_round_trip(self):
        gps_params = GpsParameters(
            reference_latitude=37.7749,
            reference_longitude=-122.4194,
            reference_altitude=16.0,
        )
        s = Sensor(
            name="gnss",
            sensor_type="gps",
            coordinate_system="ENU",
            translation=[0.0, 0.0, 1.6],
            rotation=[1.0, 0.0, 0.0, 0.0],
            gps_parameters=gps_params,
        )
        s2 = Sensor.from_dict(s.name, s.to_dict())
        assert s2.sensor_type == "gps"
        assert s2.gps_parameters is not None
        assert s2.gps_parameters.reference_latitude == pytest.approx(37.7749)
        assert s2.gps_parameters.coordinate_frame == "ENU"

    def test_imu_sensor_round_trip(self):
        imu_params = ImuParameters(
            accelerometer_noise_density=0.003924,
            gyroscope_noise_density=0.000205,
            update_rate=200.0,
        )
        s = Sensor(
            name="imu",
            sensor_type="imu",
            coordinate_system="FLU",
            translation=[0.0, 0.0, 0.5],
            rotation=[1.0, 0.0, 0.0, 0.0],
            imu_parameters=imu_params,
        )
        s2 = Sensor.from_dict(s.name, s.to_dict())
        assert s2.sensor_type == "imu"
        assert s2.imu_parameters is not None
        assert s2.imu_parameters.update_rate == pytest.approx(200.0)

    def test_radar_sensor_round_trip(self):
        radar_params = RadarParameters(
            max_range=200.0,
            range_resolution=0.4,
            azimuth_fov=90.0,
        )
        s = Sensor(
            name="front_radar",
            sensor_type="radar",
            coordinate_system="FLU",
            translation=[2.1, 0.0, 0.55],
            rotation=[1.0, 0.0, 0.0, 0.0],
            radar_parameters=radar_params,
        )
        s2 = Sensor.from_dict(s.name, s.to_dict())
        assert s2.sensor_type == "radar"
        assert s2.radar_parameters is not None
        assert s2.radar_parameters.max_range == pytest.approx(200.0)

    def test_sensor_no_parameters_returns_none(self):
        s = Sensor(
            name="front_lidar",
            sensor_type="lidar",
            coordinate_system="FLU",
            translation=[1.8, 0.0, 1.9],
            rotation=[1.0, 0.0, 0.0, 0.0],
        )
        assert s.gps_parameters is None
        assert s.imu_parameters is None
        assert s.radar_parameters is None


# ---------------------------------------------------------------------------
# SensorCollection tests
# ---------------------------------------------------------------------------


class TestSensorCollection:
    def _make_collection(self) -> SensorCollection:
        cam = Sensor(
            name="front_camera",
            sensor_type="camera",
            coordinate_system="RDF",
            translation=[1.5, 0.0, 1.4],
            rotation=[1.0, 0.0, 0.0, 0.0],
            intrinsics=CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0, width=1280, height=720),
        )
        lidar = Sensor(
            name="front_lidar",
            sensor_type="lidar",
            coordinate_system="FLU",
            translation=[1.8, 0.0, 1.9],
            rotation=[1.0, 0.0, 0.0, 0.0],
        )
        return SensorCollection([cam, lidar])

    def test_add_and_get_sensor(self):
        col = self._make_collection()
        s = col.get_sensor("front_camera")
        assert s.name == "front_camera"

    def test_sensor_names(self):
        col = self._make_collection()
        assert "front_camera" in col.sensor_names
        assert "front_lidar" in col.sensor_names

    def test_len(self):
        col = self._make_collection()
        assert len(col) == 2

    def test_contains(self):
        col = self._make_collection()
        assert "front_lidar" in col
        assert "rear_lidar" not in col

    def test_get_missing_sensor_raises(self):
        col = self._make_collection()
        with pytest.raises(KeyError, match="rear_lidar"):
            col.get_sensor("rear_lidar")

    def test_transform_to_ego(self):
        col = self._make_collection()
        T = col.transform_to_ego("front_lidar")
        assert T.shape == (4, 4)
        np.testing.assert_allclose(T[:3, 3], [1.8, 0.0, 1.9], atol=1e-10)

    def test_transform_from_ego(self):
        col = self._make_collection()
        T_to = col.transform_to_ego("front_lidar")
        T_from = col.transform_from_ego("front_lidar")
        product = T_to @ T_from
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)

    def test_transform_between_identity_when_same(self):
        """Transform from sensor A to sensor A should be identity."""
        col = self._make_collection()
        T = col.transform_between("front_lidar", "front_lidar")
        np.testing.assert_allclose(T, np.eye(4), atol=1e-10)

    def test_transform_between_inverse(self):
        """T(a→b) and T(b→a) should be inverses."""
        col = self._make_collection()
        T_ab = col.transform_between("front_camera", "front_lidar")
        T_ba = col.transform_between("front_lidar", "front_camera")
        np.testing.assert_allclose(T_ab @ T_ba, np.eye(4), atol=1e-10)

    def test_transform_between_translation_only(self):
        """Two sensors both at identity rotation; transform is a pure translation."""
        col = self._make_collection()
        T = col.transform_between("front_camera", "front_lidar")
        # Camera at [1.5,0,1.4], lidar at [1.8,0,1.9]
        # T_cam→lidar translates camera coords into lidar frame:
        #   expected translation = T_cam→ego * p, then T_ego→lidar
        # With identity rotations: t_cam_in_lidar = t_cam - t_lidar
        expected_translation = np.array([1.5, 0.0, 1.4]) - np.array([1.8, 0.0, 1.9])
        np.testing.assert_allclose(T[:3, 3], expected_translation, atol=1e-10)

    def test_yaml_round_trip(self):
        col = self._make_collection()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            col.to_yaml(path)
            col2 = SensorCollection.from_yaml(path)
            assert len(col2) == len(col)
            for name in col.sensor_names:
                s1, s2 = col.get_sensor(name), col2.get_sensor(name)
                assert s1.sensor_type == s2.sensor_type
                assert s1.translation == pytest.approx(s2.translation)
                assert s1.rotation == pytest.approx(s2.rotation)
        finally:
            os.unlink(path)

    def test_load_example_yaml(self):
        """Smoke-test loading the bundled example YAML."""
        example_path = Path(__file__).parent.parent / "examples" / "sensor_collection.yaml"
        col = SensorCollection.from_yaml(example_path)
        assert len(col) >= 5
        front_cam = col.get_sensor("front_camera")
        assert front_cam.sensor_type == "camera"
        assert front_cam.intrinsics is not None
        # Verify GPS sensor
        gnss = col.get_sensor("gnss")
        assert gnss.sensor_type == "gps"
        assert gnss.gps_parameters is not None
        assert gnss.gps_parameters.coordinate_frame == "ENU"
        # Verify IMU sensor
        imu = col.get_sensor("imu")
        assert imu.sensor_type == "imu"
        assert imu.imu_parameters is not None
        assert imu.imu_parameters.update_rate == pytest.approx(200.0)
        # Verify radar sensor
        radar = col.get_sensor("front_radar")
        assert radar.sensor_type == "radar"
        assert radar.radar_parameters is not None
        assert radar.radar_parameters.max_range == pytest.approx(200.0)

    def test_remove_sensor(self):
        col = self._make_collection()
        col.remove_sensor("front_lidar")
        assert "front_lidar" not in col
        assert len(col) == 1


# ---------------------------------------------------------------------------
# Quaternion / transform helper tests
# ---------------------------------------------------------------------------


class TestQuaternionHelpers:
    def test_identity_quaternion(self):
        R = _quaternion_to_rotation_matrix([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_90_degree_rotation_z(self):
        """Quaternion for 90° rotation about Z: w=cos(45°), z=sin(45°)."""
        angle = math.pi / 2
        w = math.cos(angle / 2)
        z = math.sin(angle / 2)
        R = _quaternion_to_rotation_matrix([w, 0.0, 0.0, z])
        # x-axis should map to y-axis
        x_rotated = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(x_rotated, [0.0, 1.0, 0.0], atol=1e-10)

    def test_unnormalised_quaternion_is_normalised(self):
        """_quaternion_to_rotation_matrix should normalise its input."""
        R1 = _quaternion_to_rotation_matrix([1.0, 0.0, 0.0, 0.0])
        R2 = _quaternion_to_rotation_matrix([2.0, 0.0, 0.0, 0.0])  # un-normalised
        np.testing.assert_allclose(R1, R2, atol=1e-10)

    def test_make_transform_shape(self):
        T = _make_transform([1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
        assert T.shape == (4, 4)
        np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0], atol=1e-10)
        np.testing.assert_allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10)
