"""
imu/imu.py

Parser for IMU (Inertial Measurement Unit) binary data files.

Binary format
-------------
Each record is a fixed-size binary structure written in little-endian byte
order:

==========  ====  ========  ============================================
Field       Type  Size (B)  Description
==========  ====  ========  ============================================
timestamp   f64      8      UNIX timestamp in seconds
ax          f32      4      Linear acceleration X  (m/s²)
ay          f32      4      Linear acceleration Y  (m/s²)
az          f32      4      Linear acceleration Z  (m/s²)
wx          f32      4      Angular velocity X     (rad/s)
wy          f32      4      Angular velocity Y     (rad/s)
wz          f32      4      Angular velocity Z     (rad/s)
qw          f32      4      Orientation quaternion W  (optional)
qx          f32      4      Orientation quaternion X  (optional)
qy          f32      4      Orientation quaternion Y  (optional)
qz          f32      4      Orientation quaternion Z  (optional)
==========  ====  ========  ============================================

Two record sizes are supported:

* **32 bytes** (no orientation): timestamp(f64) + accel(3×f32) + gyro(3×f32)
* **48 bytes** (with orientation): timestamp(f64) + accel(3×f32) + gyro(3×f32) + quaternion(4×f32)

The format is auto-detected from the file size.  Output is a structured numpy
array; available fields depend on the detected format.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Structured dtypes
# ---------------------------------------------------------------------------

IMU_POINT_DTYPE = np.dtype([
    ("timestamp", np.float64),
    ("ax", np.float32),
    ("ay", np.float32),
    ("az", np.float32),
    ("wx", np.float32),
    ("wy", np.float32),
    ("wz", np.float32),
])

IMU_ORIENTATION_DTYPE = np.dtype([
    ("timestamp", np.float64),
    ("ax", np.float32),
    ("ay", np.float32),
    ("az", np.float32),
    ("wx", np.float32),
    ("wy", np.float32),
    ("wz", np.float32),
    ("qw", np.float32),
    ("qx", np.float32),
    ("qy", np.float32),
    ("qz", np.float32),
])

# Detect format by record size (bytes).  Check basic format first so that
# files whose size is divisible by both record sizes are treated as basic.
_DTYPE_BY_SIZE: list[tuple[int, np.dtype]] = [
    (IMU_POINT_DTYPE.itemsize, IMU_POINT_DTYPE),
    (IMU_ORIENTATION_DTYPE.itemsize, IMU_ORIENTATION_DTYPE),
]


class ImuParser:
    """Parser for IMU binary data files.

    Args:
        path: Path to the ``.bin`` IMU data file.

    Example::

        parser = ImuParser("imu_data.bin")
        data = parser.read()
        # data["ax"], data["ay"], data["az"] – accelerometer (m/s²)
        # data["wx"], data["wy"], data["wz"] – gyroscope (rad/s)
        accel = parser.linear_acceleration()
        gyro = parser.angular_velocity()
    """

    def __init__(self, path: str | os.PathLike) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"IMU binary file not found: {self._path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> np.ndarray:
        """Read all IMU records from the file.

        Returns:
            Structured numpy array.  Fields are always
            ``('timestamp', 'ax', 'ay', 'az', 'wx', 'wy', 'wz')``;
            records written with orientation additionally carry
            ``('qw', 'qx', 'qy', 'qz')``.

        Raises:
            ValueError: If the file size is not divisible by a known record
                size.
        """
        file_size = self._path.stat().st_size
        if file_size == 0:
            return np.empty(0, dtype=IMU_POINT_DTYPE)

        for record_size, dtype in _DTYPE_BY_SIZE:
            if file_size % record_size == 0:
                n_records = file_size // record_size
                raw = np.fromfile(self._path, dtype=np.uint8)
                return np.frombuffer(raw.tobytes(), dtype=dtype).copy()

        raise ValueError(
            f"Cannot determine IMU binary format: file size {file_size} bytes is not "
            f"divisible by any known record size "
            f"({[s for s, _ in _DTYPE_BY_SIZE]} bytes)."
        )

    def linear_acceleration(self) -> np.ndarray:
        """Return an ``(N, 3)`` float32 array of ``[ax, ay, az]`` in m/s²."""
        data = self.read()
        return np.column_stack([data["ax"], data["ay"], data["az"]])

    def angular_velocity(self) -> np.ndarray:
        """Return an ``(N, 3)`` float32 array of ``[wx, wy, wz]`` in rad/s."""
        data = self.read()
        return np.column_stack([data["wx"], data["wy"], data["wz"]])

    def timestamps(self) -> np.ndarray:
        """Return a 1-D float64 array of UNIX timestamps in seconds."""
        return self.read()["timestamp"].copy()

    @property
    def path(self) -> Path:
        return self._path


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def load_imu_bin(path: str | os.PathLike) -> np.ndarray:
    """Load an IMU binary ``.bin`` data file.

    Args:
        path: Path to the ``.bin`` file.

    Returns:
        Structured numpy array with IMU records.
    """
    return ImuParser(path).read()
