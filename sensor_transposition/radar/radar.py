"""
radar/radar.py

Parser for radar detection binary data files.

Binary format
-------------
Each detection record is a fixed-size little-endian binary structure:

==========  ====  ========  =============================================
Field       Type  Size (B)  Description
==========  ====  ========  =============================================
range       f32      4      Slant range to target in metres
azimuth     f32      4      Azimuth angle in degrees (positive = right)
elevation   f32      4      Elevation angle in degrees (positive = up)
velocity    f32      4      Radial (Doppler) velocity in m/s
            		         (negative = approaching)
snr         f32      4      Signal-to-noise ratio in dB
==========  ====  ========  =============================================

Each record is 20 bytes.  The output is a structured numpy array with the
fields listed above.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Structured dtype
# ---------------------------------------------------------------------------

RADAR_DETECTION_DTYPE = np.dtype([
    ("range", np.float32),
    ("azimuth", np.float32),
    ("elevation", np.float32),
    ("velocity", np.float32),
    ("snr", np.float32),
])


class RadarParser:
    """Parser for radar detection binary files.

    Args:
        path: Path to the ``.bin`` radar data file.

    Example::

        parser = RadarParser("radar_frame.bin")
        detections = parser.read()
        # detections["range"]     – slant range in metres
        # detections["azimuth"]   – azimuth in degrees
        # detections["velocity"]  – Doppler velocity in m/s
        xyz = parser.xyz()
    """

    def __init__(self, path: str | os.PathLike) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Radar binary file not found: {self._path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> np.ndarray:
        """Read all radar detections from the file.

        Returns:
            Structured numpy array with dtype
            ``[('range', f4), ('azimuth', f4), ('elevation', f4),
            ('velocity', f4), ('snr', f4)]`` and shape ``(N,)``.

        Raises:
            ValueError: If the file size is not divisible by the record size.
        """
        file_size = self._path.stat().st_size
        if file_size == 0:
            return np.empty(0, dtype=RADAR_DETECTION_DTYPE)

        record_size = RADAR_DETECTION_DTYPE.itemsize
        if file_size % record_size != 0:
            raise ValueError(
                f"Radar binary file size {file_size} bytes is not divisible "
                f"by the record size {record_size} bytes."
            )

        n_detections = file_size // record_size
        raw = np.fromfile(self._path, dtype=np.float32).reshape(n_detections, 5)
        return _float32_to_structured(raw)

    def xyz(self) -> np.ndarray:
        """Return an ``(N, 3)`` float32 array of Cartesian ``[x, y, z]`` coordinates.

        Converts spherical radar coordinates (range, azimuth, elevation) to
        Cartesian using:

        * ``x = range * cos(elevation) * cos(azimuth)``   (forward)
        * ``y = range * cos(elevation) * sin(azimuth)``   (left/right)
        * ``z = range * sin(elevation)``                   (up/down)
        """
        detections = self.read()
        az_rad = np.radians(detections["azimuth"].astype(np.float32))
        el_rad = np.radians(detections["elevation"].astype(np.float32))
        r = detections["range"].astype(np.float32)
        cos_el = np.cos(el_rad)
        x = r * cos_el * np.cos(az_rad)
        y = r * cos_el * np.sin(az_rad)
        z = r * np.sin(el_rad)
        return np.column_stack([x, y, z])

    @property
    def path(self) -> Path:
        return self._path


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def load_radar_bin(path: str | os.PathLike) -> np.ndarray:
    """Load a radar detection binary ``.bin`` file.

    Args:
        path: Path to the ``.bin`` file.

    Returns:
        Structured numpy array with fields
        ``('range', 'azimuth', 'elevation', 'velocity', 'snr')``.
    """
    return RadarParser(path).read()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _float32_to_structured(raw: np.ndarray) -> np.ndarray:
    """Convert an ``(N, 5)`` float32 array to a structured detection array."""
    n = raw.shape[0]
    out = np.empty(n, dtype=RADAR_DETECTION_DTYPE)
    out["range"] = raw[:, 0]
    out["azimuth"] = raw[:, 1]
    out["elevation"] = raw[:, 2]
    out["velocity"] = raw[:, 3]
    out["snr"] = raw[:, 4]
    return out
