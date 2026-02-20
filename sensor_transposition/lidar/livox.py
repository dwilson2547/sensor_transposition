"""
lidar/livox.py

Parser for Livox LiDAR point-cloud data.

Supported formats
-----------------
* **LVX (v1.0)** – The original Livox file format used by Livox Viewer.
* **LVX2 (v2.0)** – The updated format used by newer Livox sensors
  (Mid-360, HAP, Avia).

Both formats share the same outer file structure:

::

    [ File Header (24 bytes) ]
    [ Device Info Block       ]
    [ Point Data Block        ]

Inside the Point Data Block, data is organised into *frames*, each of which
contains one or more *packages*.  A package holds a fixed number of points in
one of several sub-formats.

This module implements a pure-Python / numpy parser that handles the most
common LVX/LVX2 point data types:

=====  ============================================================
 Type  Description
=====  ============================================================
  0    Cartesian (float32) – x, y, z, intensity (LVX1 "raw")
  1    Spherical (float32) – depth, zenith, azimuth, intensity
  2    Cartesian (int32, mm precision) – x, y, z, intensity, tag,
       line (LVX2 standard)
=====  ============================================================

All coordinate values are returned in metres.  Intensity is normalised to
the range [0, 1] when the raw value exceeds 1.

References
----------
* https://github.com/Livox-SDK/LivoxSDK/wiki
* Livox SDK2 data format specification (public documentation).
"""

from __future__ import annotations

import io
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LVX_MAGIC = b"livox_tech"
_FILE_HEADER_SIZE = 24  # bytes

_DEVICE_INFO_SIZE_V1 = 59  # bytes per device entry in LVX1
_DEVICE_INFO_SIZE_V2 = 63  # bytes per device entry in LVX2

# Frame / package header sizes
_FRAME_HEADER_SIZE = 24   # bytes
_PKG_HEADER_SIZE_V1 = 19  # bytes
_PKG_HEADER_SIZE_V2 = 22  # bytes

# Point sizes (bytes) for each data type
_POINT_SIZE: dict[int, int] = {
    0: 16,  # Cartesian float32  (x,y,z,intensity)
    1: 16,  # Spherical float32  (depth,zenith,azimuth,intensity)
    2: 14,  # Cartesian int32 mm (x,y,z,intensity,tag,line)
}

# Output structured dtype for all point types (xyz in metres, float32)
LIVOX_POINT_DTYPE = np.dtype(
    [("x", np.float32), ("y", np.float32), ("z", np.float32), ("intensity", np.float32)]
)

# ---------------------------------------------------------------------------
# File header
# ---------------------------------------------------------------------------


@dataclass
class LvxFileHeader:
    """Parsed LVX file header."""

    magic: bytes
    version_major: int
    version_minor: int
    version_patch: int
    version_build: int
    magic_code: int
    device_count: int


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


class LivoxParser:
    """Parser for Livox LVX / LVX2 point-cloud files.

    Args:
        path: Path to the ``.lvx`` or ``.lvx2`` file.

    Example::

        parser = LivoxParser("recording.lvx2")
        cloud = parser.read()
        # Returns a structured numpy array with fields x, y, z, intensity
    """

    def __init__(self, path: str | os.PathLike) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Livox file not found: {self._path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> np.ndarray:
        """Read all points from the LVX/LVX2 file.

        Returns:
            Structured numpy array with dtype
            ``[('x', f4), ('y', f4), ('z', f4), ('intensity', f4)]``
            and shape ``(N,)``.  Coordinates are in metres.
        """
        with open(self._path, "rb") as fh:
            data = fh.read()
        return _parse_lvx(data)

    def xyz(self) -> np.ndarray:
        """Return an ``(N, 3)`` float32 array of ``[x, y, z]`` coordinates."""
        cloud = self.read()
        return np.column_stack([cloud["x"], cloud["y"], cloud["z"]])

    def xyz_intensity(self) -> np.ndarray:
        """Return an ``(N, 4)`` float32 array of ``[x, y, z, intensity]``."""
        cloud = self.read()
        return np.column_stack([cloud["x"], cloud["y"], cloud["z"], cloud["intensity"]])

    @property
    def path(self) -> Path:
        return self._path


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def load_livox_lvx(path: str | os.PathLike) -> np.ndarray:
    """Load a Livox LVX / LVX2 file.

    Args:
        path: Path to the ``.lvx`` or ``.lvx2`` file.

    Returns:
        Structured numpy array with fields ``('x', 'y', 'z', 'intensity')``.
    """
    return LivoxParser(path).read()


# ---------------------------------------------------------------------------
# Internal parsing implementation
# ---------------------------------------------------------------------------


def _parse_lvx(data: bytes) -> np.ndarray:
    """Parse raw LVX/LVX2 bytes and return a structured point array."""
    buf = io.BytesIO(data)

    header = _read_file_header(buf)
    version = header.version_major

    # Read device info block
    device_info_size = _DEVICE_INFO_SIZE_V2 if version >= 2 else _DEVICE_INFO_SIZE_V1
    buf.seek(_FILE_HEADER_SIZE + device_info_size * header.device_count)

    # Read point data block
    all_points: List[np.ndarray] = []
    pkg_header_size = _PKG_HEADER_SIZE_V2 if version >= 2 else _PKG_HEADER_SIZE_V1

    while True:
        # Try to read a frame header
        frame_hdr_bytes = buf.read(_FRAME_HEADER_SIZE)
        if len(frame_hdr_bytes) < _FRAME_HEADER_SIZE:
            break
        frame_pts = _parse_frame(buf, frame_hdr_bytes, pkg_header_size, data)
        if frame_pts is not None and len(frame_pts) > 0:
            all_points.append(frame_pts)

    if not all_points:
        return np.empty(0, dtype=LIVOX_POINT_DTYPE)
    return np.concatenate(all_points)


def _read_file_header(buf: io.BytesIO) -> LvxFileHeader:
    """Read and validate the 24-byte LVX file header."""
    raw = buf.read(_FILE_HEADER_SIZE)
    if len(raw) < _FILE_HEADER_SIZE:
        raise ValueError("File too short to contain a valid LVX header.")
    magic = raw[:10]
    if magic != _LVX_MAGIC:
        raise ValueError(
            f"Not a Livox LVX file (magic bytes {magic!r} != {_LVX_MAGIC!r})."
        )
    ver_maj, ver_min, ver_patch, ver_build = struct.unpack_from("<BBBB", raw, 10)
    magic_code = struct.unpack_from("<I", raw, 14)[0]
    device_count = struct.unpack_from("<B", raw, 22)[0]
    return LvxFileHeader(
        magic=magic,
        version_major=ver_maj,
        version_minor=ver_min,
        version_patch=ver_patch,
        version_build=ver_build,
        magic_code=magic_code,
        device_count=device_count,
    )


def _parse_frame(
    buf: io.BytesIO,
    frame_hdr_bytes: bytes,
    pkg_header_size: int,
    full_data: bytes,
) -> Optional[np.ndarray]:
    """Parse one frame from the current buffer position.

    The frame header layout (24 bytes):

    ======  ====  ====================================================
    Offset  Size  Field
    ======  ====  ====================================================
      0      8    current_offset (uint64) – absolute offset in file
      8      8    next_offset    (uint64) – absolute offset of next frame
     16      8    frame_index    (uint64)
    ======  ====  ====================================================
    """
    if len(frame_hdr_bytes) < _FRAME_HEADER_SIZE:
        return None

    current_offset, next_offset, _frame_index = struct.unpack_from("<QQQ", frame_hdr_bytes, 0)
    frame_data_size = int(next_offset) - int(current_offset) - _FRAME_HEADER_SIZE
    if frame_data_size <= 0:
        return None

    frame_data = buf.read(frame_data_size)
    all_pts: List[np.ndarray] = []
    pos = 0

    while pos + pkg_header_size <= len(frame_data):
        pkg_pts, consumed = _parse_package(frame_data, pos, pkg_header_size)
        if consumed <= 0:
            break
        if pkg_pts is not None and len(pkg_pts) > 0:
            all_pts.append(pkg_pts)
        pos += consumed

    if not all_pts:
        return None
    return np.concatenate(all_pts)


def _parse_package(
    data: bytes,
    offset: int,
    pkg_header_size: int,
) -> Tuple[Optional[np.ndarray], int]:
    """Parse one data package starting at *offset*.

    LVX1 package header (19 bytes):

    ======  ====  =========================================================
    Offset  Size  Field
    ======  ====  =========================================================
      0      1    device_index
      1      1    data_type
      2      1    timestamp_type
      3      8    timestamp (uint64)
     11      4    udp_counter (uint32)
     15      4    length (uint32) – total package length including header
    ======  ====  =========================================================

    LVX2 package header (22 bytes): same layout but 3 additional bytes for
    lidar_type, lidar_id, and rsvd at offsets 19–21.
    """
    if offset + pkg_header_size > len(data):
        return None, 0

    data_type = struct.unpack_from("<B", data, offset + 1)[0]
    pkg_length = struct.unpack_from("<I", data, offset + pkg_header_size - 4)[0]

    if pkg_length == 0 or offset + pkg_length > len(data):
        return None, 0

    point_data_start = offset + pkg_header_size
    point_data_end = offset + pkg_length
    point_data = data[point_data_start:point_data_end]

    pts = _parse_points(point_data, data_type)
    return pts, int(pkg_length)


def _parse_points(data: bytes, data_type: int) -> Optional[np.ndarray]:
    """Parse raw point bytes for the given *data_type*."""
    point_size = _POINT_SIZE.get(data_type)
    if point_size is None or len(data) == 0:
        return None

    n_pts = len(data) // point_size
    if n_pts == 0:
        return None

    out = np.empty(n_pts, dtype=LIVOX_POINT_DTYPE)

    if data_type == 0:
        # Cartesian float32: x, y, z (mm → m), intensity (uint8 in float32)
        raw = np.frombuffer(data[: n_pts * 16], dtype=np.float32).reshape(n_pts, 4)
        out["x"] = raw[:, 0] / 1000.0
        out["y"] = raw[:, 1] / 1000.0
        out["z"] = raw[:, 2] / 1000.0
        out["intensity"] = raw[:, 3] / 255.0

    elif data_type == 1:
        # Spherical float32: depth (mm), zenith (0.01°), azimuth (0.01°), intensity
        raw = np.frombuffer(data[: n_pts * 16], dtype=np.float32).reshape(n_pts, 4)
        depth = raw[:, 0] / 1000.0
        zenith = np.radians(raw[:, 1] * 0.01)
        azimuth = np.radians(raw[:, 2] * 0.01)
        out["x"] = (depth * np.sin(zenith) * np.cos(azimuth)).astype(np.float32)
        out["y"] = (depth * np.sin(zenith) * np.sin(azimuth)).astype(np.float32)
        out["z"] = (depth * np.cos(zenith)).astype(np.float32)
        out["intensity"] = raw[:, 3] / 255.0

    elif data_type == 2:
        # Cartesian int32 mm: x(4),y(4),z(4),reflectivity(1),tag(1) = 14 bytes
        fmt = "<iiibb"
        for i in range(n_pts):
            start = i * point_size
            vals = struct.unpack_from("<iiibb", data, start)
            out["x"][i] = vals[0] / 1000.0
            out["y"][i] = vals[1] / 1000.0
            out["z"][i] = vals[2] / 1000.0
            out["intensity"][i] = (vals[3] & 0xFF) / 255.0
    else:
        return None

    return out
