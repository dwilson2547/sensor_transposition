"""Tests for the IMU binary data parser."""

import os
import struct
import tempfile

import numpy as np
import pytest

from sensor_transposition.imu.imu import (
    IMU_ORIENTATION_DTYPE,
    IMU_POINT_DTYPE,
    ImuParser,
    load_imu_bin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_imu_bin(path: str, records: list, with_orientation: bool = False) -> None:
    """Write a list of IMU records to a binary file."""
    dtype = IMU_ORIENTATION_DTYPE if with_orientation else IMU_POINT_DTYPE
    arr = np.array(records, dtype=dtype)
    arr.tofile(path)


class TestImuParser:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        self.tmp.close()
        self.path = self.tmp.name
        # 4 records without orientation (4 × 32 = 128 bytes, uniquely divisible by
        # IMU_POINT_DTYPE.itemsize=32, detected as basic format)
        self.records = [
            (1000.0, 0.1, -0.2, 9.8, 0.01, -0.02, 0.03),
            (1001.0, 0.2, -0.1, 9.7, 0.02, -0.01, 0.04),
            (1002.0, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0),
            (1003.0, -0.1, 0.1, 9.75, -0.01, 0.01, -0.01),
        ]
        _write_imu_bin(self.path, self.records)

    def teardown_method(self):
        os.unlink(self.path)

    def test_read_returns_structured_array(self):
        parser = ImuParser(self.path)
        data = parser.read()
        assert data.dtype == IMU_POINT_DTYPE
        assert data.shape == (4,)

    def test_read_timestamp_values(self):
        data = ImuParser(self.path).read()
        np.testing.assert_allclose(data["timestamp"], [r[0] for r in self.records], rtol=1e-6)

    def test_read_acceleration_values(self):
        data = ImuParser(self.path).read()
        np.testing.assert_allclose(data["ax"], [r[1] for r in self.records], rtol=1e-5)
        np.testing.assert_allclose(data["ay"], [r[2] for r in self.records], rtol=1e-5)
        np.testing.assert_allclose(data["az"], [r[3] for r in self.records], rtol=1e-5)

    def test_read_gyro_values(self):
        data = ImuParser(self.path).read()
        np.testing.assert_allclose(data["wx"], [r[4] for r in self.records], rtol=1e-5)
        np.testing.assert_allclose(data["wy"], [r[5] for r in self.records], rtol=1e-5)
        np.testing.assert_allclose(data["wz"], [r[6] for r in self.records], rtol=1e-5)

    def test_linear_acceleration_shape(self):
        accel = ImuParser(self.path).linear_acceleration()
        assert accel.shape == (4, 3)

    def test_angular_velocity_shape(self):
        gyro = ImuParser(self.path).angular_velocity()
        assert gyro.shape == (4, 3)

    def test_timestamps(self):
        ts = ImuParser(self.path).timestamps()
        assert ts.shape == (4,)
        np.testing.assert_allclose(ts, [r[0] for r in self.records], rtol=1e-6)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            ImuParser("/nonexistent/path/imu.bin")

    def test_load_imu_bin(self):
        data = load_imu_bin(self.path)
        assert data.dtype == IMU_POINT_DTYPE
        assert len(data) == 4

    def test_empty_file(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        tmp.close()
        try:
            data = ImuParser(tmp.name).read()
            assert len(data) == 0
        finally:
            os.unlink(tmp.name)

    def test_path_property(self):
        parser = ImuParser(self.path)
        assert str(parser.path) == self.path


class TestImuParserWithOrientation:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        self.tmp.close()
        self.path = self.tmp.name
        # 3 records with orientation (3 × 48 = 144 bytes, uniquely divisible by
        # IMU_ORIENTATION_DTYPE.itemsize=48, detected as orientation format)
        self.records = [
            (2000.0, 0.1, -0.2, 9.8, 0.01, -0.02, 0.03, 1.0, 0.0, 0.0, 0.0),
            (2001.0, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0, 0.707, 0.707, 0.0, 0.0),
            (2002.0, 0.05, -0.05, 9.79, 0.005, -0.005, 0.001, 0.0, 0.0, 1.0, 0.0),
        ]
        _write_imu_bin(self.path, self.records, with_orientation=True)

    def teardown_method(self):
        os.unlink(self.path)

    def test_read_returns_orientation_dtype(self):
        data = ImuParser(self.path).read()
        assert data.dtype == IMU_ORIENTATION_DTYPE
        assert data.shape == (3,)

    def test_quaternion_values(self):
        data = ImuParser(self.path).read()
        np.testing.assert_allclose(data["qw"][0], 1.0, rtol=1e-5)
        np.testing.assert_allclose(data["qw"][1], 0.707, rtol=1e-3)

    def test_linear_acceleration_shape(self):
        accel = ImuParser(self.path).linear_acceleration()
        assert accel.shape == (3, 3)


class TestImuInvalidFile:
    def test_invalid_file_size_raises(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        tmp.write(b"\x00" * 13)  # 13 bytes – not divisible by any record size
        tmp.close()
        try:
            with pytest.raises(ValueError, match="record size"):
                ImuParser(tmp.name).read()
        finally:
            os.unlink(tmp.name)
