"""
frame_pose.py

Defines FramePose and FramePoseSequence for tracking the position and
orientation of a sensor collection over time.

A *frame* is a span of time containing data from one or more sensors.  The
default frame duration is 0.1 s, which aligns with the rotation period of
typical Velodyne LiDAR sensors.  Each :class:`FramePose` records the ego-frame
pose (position + orientation) in a fixed world/map reference frame at a given
timestamp.

:class:`FramePoseSequence` is an ordered container of frame poses representing
the trajectory of the sensor collection – a building block for SLAM
(Simultaneous Localisation and Mapping) pipelines.
"""

from __future__ import annotations

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
    """Position and orientation of the sensor collection at a given frame.

    A frame represents a configurable span of time (see
    :class:`FramePoseSequence`) containing data from one or more sensors.
    The pose describes the ego-frame position and orientation in a fixed
    world/map frame at the given timestamp.

    Attributes:
        timestamp: Start time of the frame in seconds.
        translation: ``[x, y, z]`` position in the world/map frame (metres).
        rotation: Unit quaternion ``[w, x, y, z]`` describing the orientation
            of the ego frame in the world/map frame.
    """

    timestamp: float
    translation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])

    @property
    def transform(self) -> np.ndarray:
        """4×4 homogeneous transform: ego frame → world/map frame."""
        return _make_transform(self.rotation, self.translation)

    def to_dict(self) -> dict:
        return {
            "timestamp": float(self.timestamp),
            "translation": [float(v) for v in self.translation],
            "rotation": {
                "quaternion": [float(v) for v in self.rotation],
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FramePose":
        rotation_data = data.get("rotation", {})
        return cls(
            timestamp=float(data["timestamp"]),
            translation=[float(v) for v in data.get("translation", [0.0, 0.0, 0.0])],
            rotation=[float(v) for v in rotation_data.get("quaternion", [1.0, 0.0, 0.0, 0.0])],
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
