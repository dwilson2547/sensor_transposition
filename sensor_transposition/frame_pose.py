"""
frame_pose.py

Defines FramePose and FramePoseSequence for tracking the position and
orientation of a sensor collection over time.

A *frame* is a span of time containing data from one or more sensors.  The
default frame duration is 0.1 s, which aligns with the rotation period of
typical Velodyne LiDAR sensors.  Each :class:`FramePose` records the ego-frame
pose (position + orientation) in a fixed world/map reference frame at a given
timestamp, and optionally a 6×6 pose-covariance matrix.

The covariance matrix follows the ordering ``[x, y, z, rx, ry, rz]``,
consistent with the information matrix convention in
:class:`sensor_transposition.pose_graph.PoseGraphEdge`.  Sources such as
:class:`sensor_transposition.imu.ekf.ImuEkf` and
:func:`sensor_transposition.pose_graph.optimize_pose_graph` can populate
this field to propagate uncertainty through the SLAM pipeline.

:class:`FramePoseSequence` is an ordered container of frame poses representing
the trajectory of the sensor collection – a building block for SLAM
(Simultaneous Localisation and Mapping) pipelines.
"""

from __future__ import annotations

import csv
import functools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

from sensor_transposition.sensor_collection import _make_transform


# ---------------------------------------------------------------------------
# FramePose
# ---------------------------------------------------------------------------


@dataclass
class FramePose:
    """Position, orientation, and optional covariance of the sensor collection
    at a given frame.

    A frame represents a configurable span of time (see
    :class:`FramePoseSequence`) containing data from one or more sensors.
    The pose describes the ego-frame position and orientation in a fixed
    world/map frame at the given timestamp.

    Attributes:
        timestamp: Start time of the frame in seconds.
        translation: ``[x, y, z]`` position in the world/map frame (metres).
        rotation: Unit quaternion ``[w, x, y, z]`` describing the orientation
            of the ego frame in the world/map frame.
        covariance: Optional 6×6 pose-covariance matrix.  Rows and columns
            are ordered ``[x, y, z, rx, ry, rz]`` — translation components
            first, then rotation (rotation vector / SO(3) tangent space).
            ``None`` means no covariance estimate is available.  This
            convention is consistent with the information-matrix ordering
            used by :class:`sensor_transposition.pose_graph.PoseGraphEdge`.
    """

    timestamp: float
    translation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    covariance: Optional[np.ndarray] = field(default=None)

    @functools.cached_property
    def transform(self) -> np.ndarray:
        """4×4 homogeneous transform: ego frame → world/map frame.

        The result is computed once and cached on first access.
        """
        return _make_transform(self.rotation, self.translation)

    def __repr__(self) -> str:
        xyz = [round(float(v), 4) for v in self.translation]
        q = [round(float(v), 4) for v in self.rotation]
        return f"FramePose(t={self.timestamp}, xyz={xyz}, q={q})"

    def to_dict(self) -> dict:
        d = {
            "timestamp": float(self.timestamp),
            "translation": [float(v) for v in self.translation],
            "rotation": {
                "quaternion": [float(v) for v in self.rotation],
            },
        }
        if self.covariance is not None:
            cov = np.asarray(self.covariance, dtype=float)
            d["covariance"] = cov.tolist()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "FramePose":
        rotation_data = data.get("rotation", {})
        cov_raw = data.get("covariance")
        covariance = np.array(cov_raw, dtype=float) if cov_raw is not None else None
        return cls(
            timestamp=float(data["timestamp"]),
            translation=[float(v) for v in data.get("translation", [0.0, 0.0, 0.0])],
            rotation=[float(v) for v in rotation_data.get("quaternion", [1.0, 0.0, 0.0, 0.0])],
            covariance=covariance,
        )


# ---------------------------------------------------------------------------
# FramePoseSequence
# ---------------------------------------------------------------------------


class FramePoseSequence:
    """An ordered sequence of frame poses tracking the trajectory of a
    sensor collection over time.

    Each frame covers a configurable time span (*frame_duration*) and records
    the ego pose (position + orientation) in a fixed world/map reference
    frame.  The default duration of **0.1 s** is chosen to align with the
    rotation period of typical Velodyne LiDAR sensors.

    This structure is designed to assist with SLAM (Simultaneous Localisation
    and Mapping) workflows.

    Args:
        frame_duration: Duration of each frame in seconds (default ``0.1``).
        poses: Optional initial list of :class:`FramePose` objects.
    """

    def __init__(
        self,
        frame_duration: float = 0.1,
        poses: Optional[List[FramePose]] = None,
    ) -> None:
        if frame_duration <= 0:
            raise ValueError(
                f"frame_duration must be positive, got {frame_duration}"
            )
        self._frame_duration = float(frame_duration)
        self._poses: List[FramePose] = list(poses) if poses else []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def frame_duration(self) -> float:
        """Duration of each frame in seconds."""
        return self._frame_duration

    @property
    def timestamps(self) -> np.ndarray:
        """1-D array of timestamps for every pose in the sequence."""
        return np.array([p.timestamp for p in self._poses], dtype=float)

    @property
    def positions(self) -> np.ndarray:
        """``(N, 3)`` float array of ``[x, y, z]`` translations for every pose."""
        return np.array([p.translation for p in self._poses], dtype=float)

    @property
    def quaternions(self) -> np.ndarray:
        """``(N, 4)`` float array of ``[w, x, y, z]`` quaternions for every pose."""
        return np.array([p.rotation for p in self._poses], dtype=float)

    # ------------------------------------------------------------------
    # Sequence management
    # ------------------------------------------------------------------

    def add_pose(self, pose: FramePose) -> None:
        """Append a :class:`FramePose` to the sequence."""
        self._poses.append(pose)

    def get_pose(self, index: int) -> FramePose:
        """Retrieve a pose by its integer index."""
        return self._poses[index]

    def get_pose_at_timestamp(self, timestamp: float) -> Optional[FramePose]:
        """Return the pose whose frame contains *timestamp*, or ``None``.

        A frame with start time *t* contains *timestamp* when
        ``t <= timestamp < t + frame_duration``.
        """
        for pose in self._poses:
            if pose.timestamp <= timestamp < pose.timestamp + self._frame_duration:
                return pose
        return None

    def __len__(self) -> int:
        return len(self._poses)

    def __iter__(self):
        return iter(self._poses)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "frame_duration": float(self._frame_duration),
            "poses": [p.to_dict() for p in self._poses],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FramePoseSequence":
        frame_duration = float(data.get("frame_duration", 0.1))
        poses = [FramePose.from_dict(p) for p in data.get("poses", [])]
        return cls(frame_duration=frame_duration, poses=poses)

    def to_yaml(self, path: str | os.PathLike) -> None:
        """Write the pose sequence to a YAML file."""
        Path(path).write_text(
            yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        )

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "FramePoseSequence":
        """Load a :class:`FramePoseSequence` from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(raw)

    def to_csv(self, path: str | os.PathLike) -> None:
        """Write the pose sequence to a CSV file.

        Each row encodes one :class:`FramePose` with columns:

        ``timestamp, x, y, z, qw, qx, qy, qz``

        The ``frame_duration`` is stored in the first comment line so
        it can be recovered by :meth:`from_csv`.

        Args:
            path: Destination file path (will be created or overwritten).
        """
        with open(Path(path), "w", newline="") as f:
            f.write(f"# frame_duration={self._frame_duration}\n")
            writer = csv.writer(f)
            writer.writerow(["timestamp", "x", "y", "z", "qw", "qx", "qy", "qz"])
            for p in self._poses:
                t = p.translation
                r = p.rotation
                writer.writerow([p.timestamp, t[0], t[1], t[2], r[0], r[1], r[2], r[3]])

    @classmethod
    def from_csv(cls, path: str | os.PathLike) -> "FramePoseSequence":
        """Load a :class:`FramePoseSequence` from a CSV file written by :meth:`to_csv`.

        Args:
            path: Path to the CSV file.

        Returns:
            :class:`FramePoseSequence` reconstructed from the file.
        """
        frame_duration = 0.1
        poses: List[FramePose] = []
        with open(Path(path), "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                first = row[0].strip()
                if first.startswith("# frame_duration="):
                    try:
                        frame_duration = float(first.split("=", 1)[1])
                    except ValueError:
                        pass
                    continue
                if first == "timestamp":
                    continue  # header row
                poses.append(FramePose(
                    timestamp=float(row[0]),
                    translation=[float(row[1]), float(row[2]), float(row[3])],
                    rotation=[float(row[4]), float(row[5]), float(row[6]), float(row[7])],
                ))
        return cls(frame_duration=frame_duration, poses=poses)
