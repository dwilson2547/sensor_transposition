"""
sensor_collection.py

Defines the SensorCollection configuration, which holds extrinsic parameters
(sensor-to-ego transform) and, for cameras, intrinsic parameters for each sensor.
The configuration can be loaded from and saved to YAML files.

Coordinate systems use standard naming conventions:
  FLU  – Forward, Left, Up   (ROS / ego vehicle convention)
  RDF  – Right, Down, Forward (camera optical convention)
  FRD  – Forward, Right, Down (NED-like convention)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters.

    Attributes:
        fx: Focal length in pixels along the x-axis.
        fy: Focal length in pixels along the y-axis.
        cx: Principal point x-coordinate (pixels).
        cy: Principal point y-coordinate (pixels).
        width: Image width in pixels.
        height: Image height in pixels.
        distortion_coefficients: Radial/tangential distortion coefficients
            [k1, k2, p1, p2, k3].  Defaults to all zeros.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_coefficients: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])

    @property
    def camera_matrix(self) -> np.ndarray:
        """Return the 3×3 camera (intrinsic) matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=float,
        )

    def to_dict(self) -> dict:
        return {
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "width": int(self.width),
            "height": int(self.height),
            "distortion_coefficients": [float(v) for v in self.distortion_coefficients],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraIntrinsics":
        return cls(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            width=int(data["width"]),
            height=int(data["height"]),
            distortion_coefficients=[float(v) for v in data.get("distortion_coefficients", [0.0] * 5)],
        )


@dataclass
class Sensor:
    """Represents a single sensor in a collection.

    Attributes:
        name: Unique sensor name.
        sensor_type: "camera", "lidar", "radar", or any custom string.
        coordinate_system: The coordinate frame convention used by this sensor
            (e.g. "FLU", "RDF", "FRD").
        translation: [x, y, z] translation from the sensor origin to the ego
            origin, expressed in the ego frame (metres).
        rotation: Unit quaternion [w, x, y, z] describing the rotation from
            sensor frame to ego frame.
        intrinsics: Camera intrinsic parameters.  Only populated for camera
            sensors.
    """

    name: str
    sensor_type: str
    coordinate_system: str
    translation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    intrinsics: Optional[CameraIntrinsics] = None

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    @property
    def transform_to_ego(self) -> np.ndarray:
        """4×4 homogeneous transform matrix: sensor frame → ego frame.

        Built from the stored quaternion rotation and translation.
        """
        return _make_transform(self.rotation, self.translation)

    def to_dict(self) -> dict:
        data: dict = {
            "type": self.sensor_type,
            "coordinate_system": self.coordinate_system,
            "extrinsics": {
                "translation": [float(v) for v in self.translation],
                "rotation": {
                    "quaternion": [float(v) for v in self.rotation],
                },
            },
        }
        if self.intrinsics is not None:
            data["intrinsics"] = self.intrinsics.to_dict()
        return data

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "Sensor":
        extrinsics = data.get("extrinsics", {})
        translation = extrinsics.get("translation", [0.0, 0.0, 0.0])
        rotation_data = extrinsics.get("rotation", {})
        rotation = rotation_data.get("quaternion", [1.0, 0.0, 0.0, 0.0])

        intrinsics: Optional[CameraIntrinsics] = None
        if "intrinsics" in data:
            intrinsics = CameraIntrinsics.from_dict(data["intrinsics"])

        return cls(
            name=name,
            sensor_type=data.get("type", "unknown"),
            coordinate_system=data.get("coordinate_system", "FLU"),
            translation=list(translation),
            rotation=list(rotation),
            intrinsics=intrinsics,
        )


# ---------------------------------------------------------------------------
# SensorCollection
# ---------------------------------------------------------------------------


class SensorCollection:
    """A collection of named sensors with their extrinsic (and optional
    intrinsic) parameters.

    Example::

        collection = SensorCollection.from_yaml("sensors.yaml")
        T = collection.transform_between("front_camera", "front_lidar")
        point_in_lidar = T @ point_in_camera

    The ego frame is the common reference frame for all extrinsic transforms.
    """

    def __init__(self, sensors: Optional[List[Sensor]] = None) -> None:
        self._sensors: dict[str, Sensor] = {}
        for sensor in sensors or []:
            self.add_sensor(sensor)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def add_sensor(self, sensor: Sensor) -> None:
        """Add or replace a sensor in the collection."""
        self._sensors[sensor.name] = sensor

    def remove_sensor(self, name: str) -> None:
        """Remove a sensor by name."""
        del self._sensors[name]

    def get_sensor(self, name: str) -> Sensor:
        """Retrieve a sensor by name."""
        if name not in self._sensors:
            raise KeyError(f"Sensor '{name}' not found in collection. "
                           f"Available: {list(self._sensors)}")
        return self._sensors[name]

    @property
    def sensor_names(self) -> List[str]:
        """Return a sorted list of sensor names in the collection."""
        return sorted(self._sensors.keys())

    def __len__(self) -> int:
        return len(self._sensors)

    def __contains__(self, name: str) -> bool:
        return name in self._sensors

    # ------------------------------------------------------------------
    # Transform accessors
    # ------------------------------------------------------------------

    def transform_to_ego(self, sensor_name: str) -> np.ndarray:
        """Return the 4×4 homogeneous transform: *sensor_name* → ego."""
        return self.get_sensor(sensor_name).transform_to_ego

    def transform_from_ego(self, sensor_name: str) -> np.ndarray:
        """Return the 4×4 homogeneous transform: ego → *sensor_name*."""
        return np.linalg.inv(self.get_sensor(sensor_name).transform_to_ego)

    def transform_between(self, source: str, target: str) -> np.ndarray:
        """Return the 4×4 homogeneous transform: *source* → *target*.

        Computed as::

            T_source_to_target = inv(T_target_to_ego) @ T_source_to_ego
        """
        T_src = self.transform_to_ego(source)
        T_tgt = self.transform_to_ego(target)
        return np.linalg.inv(T_tgt) @ T_src

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"sensors": {name: sensor.to_dict() for name, sensor in self._sensors.items()}}

    def to_yaml(self, path: str | os.PathLike) -> None:
        """Write the collection to a YAML file."""
        Path(path).write_text(yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False))

    @classmethod
    def from_dict(cls, data: dict) -> "SensorCollection":
        sensors_data = data.get("sensors", {})
        sensors = [Sensor.from_dict(name, sdata) for name, sdata in sensors_data.items()]
        return cls(sensors)

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "SensorCollection":
        """Load a SensorCollection from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(raw)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _quaternion_to_rotation_matrix(q: List[float]) -> np.ndarray:
    """Convert a unit quaternion [w, x, y, z] to a 3×3 rotation matrix."""
    w, x, y, z = q
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-10:
        raise ValueError("Quaternion has near-zero norm; cannot normalise.")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - w * z),     2 * (x * z + w * y)],
        [    2 * (x * y + w * z), 1 - 2 * (x * x + z * z),     2 * (y * z - w * x)],
        [    2 * (x * z - w * y),     2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def _make_transform(quaternion: List[float], translation: List[float]) -> np.ndarray:
    """Build a 4×4 homogeneous transform from a quaternion and translation."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = _quaternion_to_rotation_matrix(quaternion)
    T[:3, 3] = translation
    return T
