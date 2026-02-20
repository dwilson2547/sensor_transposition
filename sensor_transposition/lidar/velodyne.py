"""
lidar/velodyne.py

Parser for Velodyne LiDAR point-cloud data.

Supported formats
-----------------
* **KITTI binary** (``.bin``): Raw binary dump used by the KITTI dataset and
  widely supported by Velodyne (HDL-32E, HDL-64E, VLP-16, VLP-32C) and
  Ouster recorders.  Each point is stored as four consecutive ``float32``
  values: ``(x, y, z, intensity)``.

The output is a structured numpy array with fields
``('x', 'y', 'z', 'intensity')``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

# Structured dtype for a Velodyne point (KITTI convention)
VELODYNE_POINT_DTYPE = np.dtype(
    [("x", np.float32), ("y", np.float32), ("z", np.float32), ("intensity", np.float32)]
)


class VelodyneParser:
    """Parser for Velodyne LiDAR binary (KITTI-format) files.

    Args:
        path: Path to the ``.bin`` file.

    Example::

        parser = VelodyneParser("frame0000.bin")
        cloud = parser.read()
        # cloud is a structured numpy array with fields x, y, z, intensity
        xyz = parser.xyz()
    """

    def __init__(self, path: str | os.PathLike) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Velodyne binary file not found: {self._path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> np.ndarray:
        """Read all points from the binary file.

        Returns:
            Structured numpy array with dtype ``[('x', f4), ('y', f4),
            ('z', f4), ('intensity', f4)]`` and shape ``(N,)``.
        """
        raw = np.fromfile(self._path, dtype=np.float32).reshape(-1, 4)
        return _float32_to_structured(raw)

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


def load_velodyne_bin(path: str | os.PathLike) -> np.ndarray:
    """Load a Velodyne KITTI-format ``.bin`` file.

    Args:
        path: Path to the ``.bin`` file.

    Returns:
        Structured numpy array with fields ``('x', 'y', 'z', 'intensity')``.
    """
    return VelodyneParser(path).read()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _float32_to_structured(raw: np.ndarray) -> np.ndarray:
    """Convert an ``(N, 4)`` float32 array to a structured array."""
    out = np.empty(raw.shape[0], dtype=VELODYNE_POINT_DTYPE)
    out["x"] = raw[:, 0]
    out["y"] = raw[:, 1]
    out["z"] = raw[:, 2]
    out["intensity"] = raw[:, 3]
    return out
