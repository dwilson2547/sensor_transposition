"""Tests for the Velodyne LiDAR parser."""

import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sensor_transposition.lidar.velodyne import (
    VELODYNE_POINT_DTYPE,
    VelodyneParser,
    load_velodyne_bin,
    _float32_to_structured,
)


def _write_bin(path: str, points: np.ndarray) -> None:
    """Write (N, 4) float32 array to a KITTI-style .bin file."""
    points.astype(np.float32).tofile(path)


class TestVelodyneParser:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        self.tmp.close()
        self.path = self.tmp.name
        # 5 points
        self.data = np.array(
            [
                [1.0, 2.0, 3.0, 0.5],
                [4.0, 5.0, 6.0, 0.8],
                [0.0, 0.0, 0.0, 0.0],
                [-1.0, -2.0, -3.0, 1.0],
                [10.0, 20.0, 30.0, 0.1],
            ],
            dtype=np.float32,
        )
        _write_bin(self.path, self.data)

    def teardown_method(self):
        os.unlink(self.path)

    def test_read_returns_structured_array(self):
        parser = VelodyneParser(self.path)
        cloud = parser.read()
        assert cloud.dtype == VELODYNE_POINT_DTYPE
        assert cloud.shape == (5,)

    def test_read_values(self):
        cloud = VelodyneParser(self.path).read()
        np.testing.assert_allclose(cloud["x"], self.data[:, 0])
        np.testing.assert_allclose(cloud["y"], self.data[:, 1])
        np.testing.assert_allclose(cloud["z"], self.data[:, 2])
        np.testing.assert_allclose(cloud["intensity"], self.data[:, 3])

    def test_xyz_shape(self):
        xyz = VelodyneParser(self.path).xyz()
        assert xyz.shape == (5, 3)

    def test_xyz_values(self):
        xyz = VelodyneParser(self.path).xyz()
        np.testing.assert_allclose(xyz, self.data[:, :3])

    def test_xyz_intensity_shape(self):
        xyzr = VelodyneParser(self.path).xyz_intensity()
        assert xyzr.shape == (5, 4)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            VelodyneParser("/nonexistent/path/file.bin")

    def test_load_velodyne_bin(self):
        cloud = load_velodyne_bin(self.path)
        assert cloud.dtype == VELODYNE_POINT_DTYPE
        assert len(cloud) == 5

    def test_empty_file(self):
        empty = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        empty.close()
        try:
            cloud = VelodyneParser(empty.name).read()
            assert len(cloud) == 0
        finally:
            os.unlink(empty.name)


class TestFloatToStructured:
    def test_basic(self):
        raw = np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32)
        out = _float32_to_structured(raw)
        assert out["x"][0] == pytest.approx(1.0)
        assert out["y"][0] == pytest.approx(2.0)
        assert out["z"][0] == pytest.approx(3.0)
        assert out["intensity"][0] == pytest.approx(0.5)
