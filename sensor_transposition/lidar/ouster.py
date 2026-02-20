"""
lidar/ouster.py

Parser for Ouster LiDAR point-cloud data.

Supported formats
-----------------
* **KITTI-style binary** (``.bin``): Four ``float32`` columns per point –
  ``(x, y, z, intensity)``.  This is the simplest interchange format and is
  produced by the Ouster SDK's ``pcap-to-las / pcap-to-csv`` utilities as well
  as most third-party converters.
* **Ouster extended binary** (``.bin`` with ring + timestamp): Eight
  ``float32`` columns per point – ``(x, y, z, intensity, t, reflectivity,
  ring, ambient)``.  Written by some Ouster SDK examples.

The parser auto-detects which variant is present based on the column count.

Output is a structured numpy array; column layout depends on the detected
format (see :attr:`OUSTER_POINT_DTYPE` and
:attr:`OUSTER_EXTENDED_POINT_DTYPE`).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

OUSTER_POINT_DTYPE = np.dtype(
    [("x", np.float32), ("y", np.float32), ("z", np.float32), ("intensity", np.float32)]
)

OUSTER_EXTENDED_POINT_DTYPE = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("intensity", np.float32),
        ("t", np.float32),
        ("reflectivity", np.float32),
        ("ring", np.float32),
        ("ambient", np.float32),
    ]
)

# Number of float32 columns that map to each dtype, checked largest-first so
# that 8-column files are not misidentified as 4-column files.
_DTYPE_BY_COLS: list[tuple[int, np.dtype]] = [
    (8, OUSTER_EXTENDED_POINT_DTYPE),
    (4, OUSTER_POINT_DTYPE),
]


class OusterParser:
    """Parser for Ouster LiDAR binary files.

    Args:
        path: Path to the ``.bin`` file.

    Example::

        parser = OusterParser("ouster_frame.bin")
        cloud = parser.read()
        xyz = parser.xyz()
    """

    def __init__(self, path: str | os.PathLike) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Ouster binary file not found: {self._path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> np.ndarray:
        """Read all points from the binary file.

        The format (4-column or 8-column) is auto-detected from file size.

        Returns:
            Structured numpy array.  Fields are ``('x', 'y', 'z',
            'intensity')`` for the 4-column variant, or additionally
            ``('t', 'reflectivity', 'ring', 'ambient')`` for the 8-column
            variant.

        Raises:
            ValueError: If the file size is not divisible by a known column
                count × 4 bytes.
        """
        file_size = self._path.stat().st_size
        for n_cols, dtype in _DTYPE_BY_COLS:
            point_bytes = n_cols * 4
            if file_size % point_bytes == 0:
                n_points = file_size // point_bytes
                raw = np.fromfile(self._path, dtype=np.float32).reshape(n_points, n_cols)
                return _float32_to_structured(raw, dtype)
        raise ValueError(
            f"Cannot determine Ouster binary format: file size {file_size} bytes is not "
            f"divisible by any known point size ({[c for c, _ in _DTYPE_BY_COLS]} × 4 bytes)."
        )

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


def load_ouster_bin(path: str | os.PathLike) -> np.ndarray:
    """Load an Ouster binary ``.bin`` point-cloud file.

    Args:
        path: Path to the ``.bin`` file.

    Returns:
        Structured numpy array.
    """
    return OusterParser(path).read()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _float32_to_structured(raw: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Convert an ``(N, K)`` float32 array to a structured array."""
    n = raw.shape[0]
    out = np.empty(n, dtype=dtype)
    for i, name in enumerate(dtype.names):
        out[name] = raw[:, i]
    return out
