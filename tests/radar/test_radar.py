"""Tests for the radar detection binary parser."""

import os
import tempfile

import numpy as np
import pytest

from sensor_transposition.radar.radar import (
    RADAR_DETECTION_DTYPE,
    RadarParser,
    load_radar_bin,
    _float32_to_structured,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_radar_bin(path: str, data: np.ndarray) -> None:
    """Write (N, 5) float32 array to a radar .bin file."""
    data.astype(np.float32).tofile(path)


class TestRadarParser:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        self.tmp.close()
        self.path = self.tmp.name
        # 4 detections: [range, azimuth, elevation, velocity, snr]
        self.data = np.array(
            [
                [50.0, 10.0, 2.0, -5.0, 15.0],
                [120.0, -15.0, 0.0, 10.0, 20.0],
                [30.0, 5.0, -1.0, 0.0, 30.0],
                [80.0, 0.0, 0.0, -3.5, 25.0],
            ],
            dtype=np.float32,
        )
        _write_radar_bin(self.path, self.data)

    def teardown_method(self):
        os.unlink(self.path)

    def test_read_returns_structured_array(self):
        parser = RadarParser(self.path)
        detections = parser.read()
        assert detections.dtype == RADAR_DETECTION_DTYPE
        assert detections.shape == (4,)

    def test_read_range_values(self):
        detections = RadarParser(self.path).read()
        np.testing.assert_allclose(detections["range"], self.data[:, 0])

    def test_read_azimuth_values(self):
        detections = RadarParser(self.path).read()
        np.testing.assert_allclose(detections["azimuth"], self.data[:, 1])

    def test_read_velocity_values(self):
        detections = RadarParser(self.path).read()
        np.testing.assert_allclose(detections["velocity"], self.data[:, 3])

    def test_read_snr_values(self):
        detections = RadarParser(self.path).read()
        np.testing.assert_allclose(detections["snr"], self.data[:, 4])

    def test_xyz_shape(self):
        xyz = RadarParser(self.path).xyz()
        assert xyz.shape == (4, 3)

    def test_xyz_zero_azimuth_elevation(self):
        """A target at range R, azimuth 0, elevation 0 should map to x=R, y=0, z=0."""
        data = np.array([[100.0, 0.0, 0.0, 0.0, 10.0]], dtype=np.float32)
        tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        tmp.close()
        try:
            _write_radar_bin(tmp.name, data)
            xyz = RadarParser(tmp.name).xyz()
            np.testing.assert_allclose(xyz[0], [100.0, 0.0, 0.0], atol=1e-5)
        finally:
            os.unlink(tmp.name)

    def test_xyz_90_degree_azimuth(self):
        """A target at range R, azimuth 90°, elevation 0 should map to x≈0, y=R."""
        data = np.array([[50.0, 90.0, 0.0, 0.0, 10.0]], dtype=np.float32)
        tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        tmp.close()
        try:
            _write_radar_bin(tmp.name, data)
            xyz = RadarParser(tmp.name).xyz()
            np.testing.assert_allclose(xyz[0, 0], 0.0, atol=1e-5)
            np.testing.assert_allclose(xyz[0, 1], 50.0, atol=1e-5)
            np.testing.assert_allclose(xyz[0, 2], 0.0, atol=1e-5)
        finally:
            os.unlink(tmp.name)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            RadarParser("/nonexistent/path/radar.bin")

    def test_load_radar_bin(self):
        detections = load_radar_bin(self.path)
        assert detections.dtype == RADAR_DETECTION_DTYPE
        assert len(detections) == 4

    def test_empty_file(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        tmp.close()
        try:
            detections = RadarParser(tmp.name).read()
            assert len(detections) == 0
        finally:
            os.unlink(tmp.name)

    def test_invalid_file_size_raises(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        tmp.write(b"\x00" * 7)  # 7 bytes – not divisible by 20
        tmp.close()
        try:
            with pytest.raises(ValueError, match="record size"):
                RadarParser(tmp.name).read()
        finally:
            os.unlink(tmp.name)

    def test_path_property(self):
        parser = RadarParser(self.path)
        assert str(parser.path) == self.path


class TestFloatToStructured:
    def test_basic(self):
        raw = np.array([[10.0, 5.0, 1.0, -2.0, 20.0]], dtype=np.float32)
        out = _float32_to_structured(raw)
        assert out["range"][0] == pytest.approx(10.0)
        assert out["azimuth"][0] == pytest.approx(5.0)
        assert out["elevation"][0] == pytest.approx(1.0)
        assert out["velocity"][0] == pytest.approx(-2.0)
        assert out["snr"][0] == pytest.approx(20.0)
