"""Tests for the Livox LiDAR parser."""

import io
import os
import struct
import tempfile

import numpy as np
import pytest

from sensor_transposition.lidar.livox import (
    LIVOX_POINT_DTYPE,
    LivoxParser,
    load_livox_lvx,
    _parse_lvx,
)


# ---------------------------------------------------------------------------
# Helpers to build minimal valid LVX binary data
# ---------------------------------------------------------------------------

_MAGIC = b"livox_tech"
_MAGIC_CODE = 0xAC0EA767


def _build_file_header(version_major: int = 1, device_count: int = 1) -> bytes:
    """Build a minimal 24-byte LVX file header.

    Layout (24 bytes):
      0-9   magic signature (10 bytes)
      10-13 version major/minor/patch/build (4 × uint8)
      14-17 magic_code (uint32)
      18-21 reserved (4 bytes)
      22    device_count (uint8)
      23    reserved (1 byte)
    """
    return (
        _MAGIC
        + struct.pack("<BBBB", version_major, 0, 0, 0)  # version
        + struct.pack("<I", _MAGIC_CODE)                # magic_code
        + b"\x00\x00\x00\x00"                          # 4 reserved bytes (offsets 18-21)
        + struct.pack("<BB", device_count, 0)           # device_count + rsvd (offsets 22-23)
    )


def _build_device_info_v1() -> bytes:
    """Build a 59-byte LVX1 device info entry (all zeros except lidar_type)."""
    return b"\x00" * 59


def _build_package_v1(points_xyz_intensity: list, data_type: int = 0) -> bytes:
    """Build a minimal LVX1 package with the given points.

    Data type 0: Cartesian float32 – x, y, z in mm (int-like float), intensity.
    """
    # Build point data
    point_bytes = b""
    for x, y, z, intensity in points_xyz_intensity:
        # type 0: x,y,z in mm as float32, intensity as float32
        point_bytes += struct.pack("<ffff", x * 1000.0, y * 1000.0, z * 1000.0, intensity * 255.0)

    pkg_header_size = 19
    pkg_length = pkg_header_size + len(point_bytes)

    header = (
        struct.pack("<BB", 0, data_type)   # device_index, data_type
        + struct.pack("<B", 0)             # timestamp_type
        + struct.pack("<Q", 0)             # timestamp
        + struct.pack("<I", 0)             # udp_counter
        + struct.pack("<I", pkg_length)    # length
    )
    assert len(header) == pkg_header_size, f"header size {len(header)}"
    return header + point_bytes


def _build_frame_v1(packages: list) -> bytes:
    """Wrap packages in an LVX1 frame header."""
    pkg_data = b"".join(packages)
    frame_header_size = 24
    # We'll write offsets as dummy since _parse_frame uses them for sizing
    # current_offset=0, next_offset=frame_header_size+len(pkg_data), frame_index=0
    total = frame_header_size + len(pkg_data)
    frame_hdr = struct.pack("<QQQ", 0, total, 0)
    assert len(frame_hdr) == frame_header_size
    return frame_hdr + pkg_data


def _build_lvx1_file(points: list) -> bytes:
    """Build a minimal complete LVX1 file with one frame and one package."""
    file_hdr = _build_file_header(version_major=1, device_count=1)
    dev_info = _build_device_info_v1()
    pkg = _build_package_v1(points, data_type=0)
    frame = _build_frame_v1([pkg])
    return file_hdr + dev_info + frame


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLivoxParserFromBytes:
    def test_parse_single_point(self):
        pts = [(1.0, 2.0, 3.0, 0.5)]
        data = _build_lvx1_file(pts)
        cloud = _parse_lvx(data)
        assert len(cloud) == 1
        assert cloud.dtype == LIVOX_POINT_DTYPE
        assert cloud["x"][0] == pytest.approx(1.0, abs=1e-3)
        assert cloud["y"][0] == pytest.approx(2.0, abs=1e-3)
        assert cloud["z"][0] == pytest.approx(3.0, abs=1e-3)

    def test_parse_multiple_points(self):
        pts = [(i * 0.1, 0.0, 0.0, 0.5) for i in range(10)]
        data = _build_lvx1_file(pts)
        cloud = _parse_lvx(data)
        assert len(cloud) == 10

    def test_empty_frame(self):
        """A file with no points should return an empty structured array."""
        data = _build_lvx1_file([])
        cloud = _parse_lvx(data)
        assert len(cloud) == 0

    def test_wrong_magic_raises(self):
        bad_data = b"not_livox" + b"\x00" * 100
        with pytest.raises(ValueError, match="livox"):
            _parse_lvx(bad_data)

    def test_truncated_header_raises(self):
        with pytest.raises(ValueError, match="too short"):
            _parse_lvx(b"\x00" * 10)


class TestLivoxParserFileIO:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".lvx", delete=False)
        pts = [(1.0, 2.0, 3.0, 0.8), (4.0, 5.0, 6.0, 0.2)]
        self.tmp.write(_build_lvx1_file(pts))
        self.tmp.close()
        self.path = self.tmp.name

    def teardown_method(self):
        os.unlink(self.path)

    def test_parser_read(self):
        parser = LivoxParser(self.path)
        cloud = parser.read()
        assert len(cloud) == 2

    def test_xyz_shape(self):
        xyz = LivoxParser(self.path).xyz()
        assert xyz.shape == (2, 3)

    def test_xyz_intensity_shape(self):
        xyzr = LivoxParser(self.path).xyz_intensity()
        assert xyzr.shape == (2, 4)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            LivoxParser("/nonexistent/file.lvx")

    def test_load_livox_lvx(self):
        cloud = load_livox_lvx(self.path)
        assert len(cloud) == 2
        assert cloud.dtype == LIVOX_POINT_DTYPE
