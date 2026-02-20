"""
sensor_collection.py

Defines the SensorCollection configuration, which holds extrinsic parameters
(sensor-to-ego transform) and, for cameras, intrinsic parameters for each sensor.
The configuration can be loaded from and saved to YAML files.

Coordinate systems use standard naming conventions:
  FLU  – Forward, Left, Up   (ROS / ego vehicle convention)
  RDF  – Right, Down, Forward (camera optical convention)
  FRD  – Forward, Right, Down (NED-like convention)
  ENU  – East, North, Up     (GPS / local tangent plane convention)
  NED  – North, East, Down   (aviation / navigation convention)
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
class GpsParameters:
    """Parameters specific to a GPS/GNSS sensor.

    Attributes:
        reference_latitude: Reference latitude in decimal degrees.
        reference_longitude: Reference longitude in decimal degrees.
        reference_altitude: Reference altitude above sea level in metres.
        coordinate_frame: Local coordinate frame convention for ENU output
            (e.g. "ENU", "NED").
    """

    reference_latitude: float = 0.0
    reference_longitude: float = 0.0
    reference_altitude: float = 0.0
    coordinate_frame: str = "ENU"

    def to_dict(self) -> dict:
        return {
            "reference_latitude": float(self.reference_latitude),
            "reference_longitude": float(self.reference_longitude),
            "reference_altitude": float(self.reference_altitude),
            "coordinate_frame": self.coordinate_frame,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GpsParameters":
        return cls(
            reference_latitude=float(data.get("reference_latitude", 0.0)),
            reference_longitude=float(data.get("reference_longitude", 0.0)),
            reference_altitude=float(data.get("reference_altitude", 0.0)),
            coordinate_frame=str(data.get("coordinate_frame", "ENU")),
        )


@dataclass
class ImuParameters:
    """Parameters specific to an IMU (Inertial Measurement Unit) sensor.

    Attributes:
        accelerometer_noise_density: Accelerometer noise density in m/s²/√Hz.
        gyroscope_noise_density: Gyroscope noise density in rad/s/√Hz.
        accelerometer_random_walk: Accelerometer bias random walk in m/s³/√Hz.
        gyroscope_random_walk: Gyroscope bias random walk in rad/s²/√Hz.
        update_rate: Nominal sensor update rate in Hz.
    """

    accelerometer_noise_density: float = 0.0
    gyroscope_noise_density: float = 0.0
    accelerometer_random_walk: float = 0.0
    gyroscope_random_walk: float = 0.0
    update_rate: float = 100.0

    def to_dict(self) -> dict:
        return {
            "accelerometer_noise_density": float(self.accelerometer_noise_density),
            "gyroscope_noise_density": float(self.gyroscope_noise_density),
            "accelerometer_random_walk": float(self.accelerometer_random_walk),
            "gyroscope_random_walk": float(self.gyroscope_random_walk),
            "update_rate": float(self.update_rate),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImuParameters":
        return cls(
            accelerometer_noise_density=float(data.get("accelerometer_noise_density", 0.0)),
            gyroscope_noise_density=float(data.get("gyroscope_noise_density", 0.0)),
            accelerometer_random_walk=float(data.get("accelerometer_random_walk", 0.0)),
            gyroscope_random_walk=float(data.get("gyroscope_random_walk", 0.0)),
            update_rate=float(data.get("update_rate", 100.0)),
        )


@dataclass
class RadarParameters:
    """Parameters specific to a radar sensor.

    Attributes:
        max_range: Maximum detection range in metres.
        range_resolution: Range resolution in metres.
        azimuth_fov: Azimuth field of view in degrees (full angle).
        elevation_fov: Elevation field of view in degrees (full angle).
        velocity_resolution: Radial velocity resolution in m/s.
    """

    max_range: float = 0.0
    range_resolution: float = 0.0
    azimuth_fov: float = 0.0
    elevation_fov: float = 0.0
    velocity_resolution: float = 0.0

    def to_dict(self) -> dict:
        return {
            "max_range": float(self.max_range),
            "range_resolution": float(self.range_resolution),
            "azimuth_fov": float(self.azimuth_fov),
            "elevation_fov": float(self.elevation_fov),
            "velocity_resolution": float(self.velocity_resolution),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RadarParameters":
        return cls(
            max_range=float(data.get("max_range", 0.0)),
            range_resolution=float(data.get("range_resolution", 0.0)),
            azimuth_fov=float(data.get("azimuth_fov", 0.0)),
            elevation_fov=float(data.get("elevation_fov", 0.0)),
            velocity_resolution=float(data.get("velocity_resolution", 0.0)),
        )


@dataclass
class Sensor:
    """Represents a single sensor in a collection.

    Attributes:
        name: Unique sensor name.
        sensor_type: "camera", "lidar", "radar", "gps", "imu", or any custom string.
        coordinate_system: The coordinate frame convention used by this sensor
            (e.g. "FLU", "RDF", "FRD", "ENU", "NED").
        translation: [x, y, z] translation from the sensor origin to the ego
            origin, expressed in the ego frame (metres).
        rotation: Unit quaternion [w, x, y, z] describing the rotation from
            sensor frame to ego frame.
        intrinsics: Camera intrinsic parameters.  Only populated for camera
            sensors.
        gps_parameters: GPS-specific parameters.  Only populated for GPS
            sensors.
        imu_parameters: IMU-specific parameters.  Only populated for IMU
            sensors.
        radar_parameters: Radar-specific parameters.  Only populated for radar
            sensors.
    """

    name: str
    sensor_type: str
    coordinate_system: str
    translation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    intrinsics: Optional[CameraIntrinsics] = None
    gps_parameters: Optional[GpsParameters] = None
    imu_parameters: Optional[ImuParameters] = None
    radar_parameters: Optional[RadarParameters] = None

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
        if self.gps_parameters is not None:
            data["gps_parameters"] = self.gps_parameters.to_dict()
        if self.imu_parameters is not None:
            data["imu_parameters"] = self.imu_parameters.to_dict()
        if self.radar_parameters is not None:
            data["radar_parameters"] = self.radar_parameters.to_dict()
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

        gps_parameters: Optional[GpsParameters] = None
        if "gps_parameters" in data:
            gps_parameters = GpsParameters.from_dict(data["gps_parameters"])

        imu_parameters: Optional[ImuParameters] = None
        if "imu_parameters" in data:
            imu_parameters = ImuParameters.from_dict(data["imu_parameters"])

        radar_parameters: Optional[RadarParameters] = None
        if "radar_parameters" in data:
            radar_parameters = RadarParameters.from_dict(data["radar_parameters"])

        return cls(
            name=name,
            sensor_type=data.get("type", "unknown"),
            coordinate_system=data.get("coordinate_system", "FLU"),
            translation=list(translation),
            rotation=list(rotation),
            intrinsics=intrinsics,
            gps_parameters=gps_parameters,
            imu_parameters=imu_parameters,
            radar_parameters=radar_parameters,
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
    # LiDAR-camera fusion
    # ------------------------------------------------------------------

    def project_lidar_to_image(
        self,
        lidar_name: str,
        camera_name: str,
        points: "np.ndarray",
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Project LiDAR points onto the image plane of a camera sensor.

        Looks up the extrinsic transform between *lidar_name* and *camera_name*
        and the intrinsic matrix of *camera_name*, then delegates to
        :func:`~sensor_transposition.lidar_camera.project_lidar_to_image`.

        Args:
            lidar_name: Name of the source LiDAR sensor in this collection.
            camera_name: Name of the target camera sensor in this collection.
                The sensor must have ``intrinsics`` set.
            points: ``(N, 3)`` float array of 3-D points in the LiDAR frame.

        Returns:
            pixel_coords: ``(N, 2)`` float array of ``(u, v)`` pixel coordinates.
            valid_mask: ``(N,)`` boolean array; ``True`` for points that project
                onto the image.

        Raises:
            KeyError: If either sensor is not found.
            ValueError: If the camera sensor has no intrinsics.
        """
        from sensor_transposition.lidar_camera import project_lidar_to_image as _project

        camera = self.get_sensor(camera_name)
        if camera.intrinsics is None:
            raise ValueError(
                f"Camera sensor '{camera_name}' has no intrinsics set."
            )
        T = self.transform_between(lidar_name, camera_name)
        K = camera.intrinsics.camera_matrix
        return _project(points, T, K, camera.intrinsics.width, camera.intrinsics.height)

    def color_lidar_from_image(
        self,
        lidar_name: str,
        camera_name: str,
        points: "np.ndarray",
        image: "np.ndarray",
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Sample image colour at projected LiDAR point locations.

        Looks up the extrinsic transform between *lidar_name* and *camera_name*
        and the intrinsic matrix of *camera_name*, then delegates to
        :func:`~sensor_transposition.lidar_camera.color_lidar_from_image`.

        Args:
            lidar_name: Name of the source LiDAR sensor in this collection.
            camera_name: Name of the target camera sensor in this collection.
                The sensor must have ``intrinsics`` set.
            points: ``(N, 3)`` float array of 3-D points in the LiDAR frame.
            image: ``(H, W, C)`` or ``(H, W)`` numpy array representing the image.

        Returns:
            colors: ``(N, C)`` or ``(N,)`` array of sampled colour values.
                Invalid points contain zeros.
            valid_mask: ``(N,)`` boolean array; ``True`` where colour was sampled.

        Raises:
            KeyError: If either sensor is not found.
            ValueError: If the camera sensor has no intrinsics.
        """
        from sensor_transposition.lidar_camera import color_lidar_from_image as _color

        camera = self.get_sensor(camera_name)
        if camera.intrinsics is None:
            raise ValueError(
                f"Camera sensor '{camera_name}' has no intrinsics set."
            )
        T = self.transform_between(lidar_name, camera_name)
        K = camera.intrinsics.camera_matrix
        return _color(points, T, K, image)

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
