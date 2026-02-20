"""Tests for the Ouster LiDAR parser."""

import os
import tempfile

import numpy as np
import pytest

from sensor_transposition.lidar.ouster import (
    OUSTER_EXTENDED_POINT_DTYPE,
    OUSTER_POINT_DTYPE,
    OusterParser,
    load_ouster_bin,
    _float32_to_structured,
)


def _write_bin(path: str, data: np.ndarray) -> None:
    data.astype(np.float32).tofile(path)


class TestOusterParser4Col:
    """Tests for the 4-column (x, y, z, intensity) KITTI-style format."""

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        self.tmp.close()
        self.path = self.tmp.name
        self.data = np.array(
            [[1.0, 2.0, 3.0, 0.5],
             [4.0, 5.0, 6.0, 0.8],
             [7.0, 8.0, 9.0, 0.3]],
            dtype=np.float32,
        )
        _write_bin(self.path, self.data)

    def teardown_method(self):
        os.unlink(self.path)

    def test_read_dtype(self):
        cloud = OusterParser(self.path).read()
        assert cloud.dtype == OUSTER_POINT_DTYPE

    def test_read_values(self):
        cloud = OusterParser(self.path).read()
        np.testing.assert_allclose(cloud["x"], self.data[:, 0])
        np.testing.assert_allclose(cloud["intensity"], self.data[:, 3])

    def test_xyz_shape(self):
        xyz = OusterParser(self.path).xyz()
        assert xyz.shape == (3, 3)

    def test_xyz_intensity_shape(self):
        xyzr = OusterParser(self.path).xyz_intensity()
        assert xyzr.shape == (3, 4)


class TestOusterParser8Col:
    """Tests for the 8-column extended format."""

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        self.tmp.close()
        self.path = self.tmp.name
        # 3 points × 8 columns
        self.data = np.random.default_rng(42).random((3, 8)).astype(np.float32)
        _write_bin(self.path, self.data)

    def teardown_method(self):
        os.unlink(self.path)

    def test_read_dtype(self):
        cloud = OusterParser(self.path).read()
        assert cloud.dtype == OUSTER_EXTENDED_POINT_DTYPE

    def test_read_n_points(self):
        cloud = OusterParser(self.path).read()
        assert len(cloud) == 3

    def test_xyz_shape(self):
        xyz = OusterParser(self.path).xyz()
        assert xyz.shape == (3, 3)

    def test_ring_field_present(self):
        cloud = OusterParser(self.path).read()
        assert "ring" in cloud.dtype.names


class TestOusterEdgeCases:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            OusterParser("/nonexistent/ouster.bin")

    def test_bad_file_size_raises(self):
        """A file whose size is not divisible by 4 or 8 columns should raise."""
        tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        tmp.write(b"\x00" * 7)  # 7 bytes – not valid
        tmp.close()
        try:
            with pytest.raises(ValueError, match="Cannot determine"):
                OusterParser(tmp.name).read()
        finally:
            os.unlink(tmp.name)

    def test_load_ouster_bin(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        tmp.close()
        # Use 3 points × 4 cols = 48 bytes (not divisible by 8-col point size of 32)
        data = np.ones((3, 4), dtype=np.float32)
        _write_bin(tmp.name, data)
        try:
            cloud = load_ouster_bin(tmp.name)
            assert len(cloud) == 3
        finally:
            os.unlink(tmp.name)


class TestOusterFloatToStructured:
    def test_4col(self):
        raw = np.array([[1.0, 2.0, 3.0, 0.9]], dtype=np.float32)
        out = _float32_to_structured(raw, OUSTER_POINT_DTYPE)
        assert out["x"][0] == pytest.approx(1.0)

    def test_8col(self):
        raw = np.zeros((1, 8), dtype=np.float32)
        raw[0, 6] = 5.0  # ring field
        out = _float32_to_structured(raw, OUSTER_EXTENDED_POINT_DTYPE)
        assert out["ring"][0] == pytest.approx(5.0)
