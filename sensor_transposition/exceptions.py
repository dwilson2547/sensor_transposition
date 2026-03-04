"""
exceptions.py

Library-specific exception hierarchy for ``sensor_transposition``.

All public exceptions inherit from :class:`SensorTranspositionError` so that
callers can catch every library error with a single ``except`` clause while
still being able to distinguish specific failure modes.  Each subclass also
inherits from an appropriate built-in exception so that code already catching
standard exceptions (e.g. ``except KeyError``) continues to work without
modification.

Example::

    from sensor_transposition.exceptions import SensorNotFoundError, BagError

    try:
        sensor = collection.get_sensor("missing_camera")
    except SensorNotFoundError as exc:
        print(f"Sensor lookup failed: {exc}")

    try:
        bag.write("/topic", timestamp, payload)
    except BagError as exc:
        print(f"Bag operation failed: {exc}")
"""

from __future__ import annotations

__all__ = [
    "SensorTranspositionError",
    "SensorNotFoundError",
    "BagError",
    "CalibrationError",
]


class SensorTranspositionError(Exception):
    """Base exception for all ``sensor_transposition`` errors.

    Catching this class will intercept any error raised by the library.
    """


class SensorNotFoundError(SensorTranspositionError, KeyError):
    """Raised when a requested sensor is not found in a :class:`~sensor_transposition.sensor_collection.SensorCollection`.

    Inherits from both :class:`SensorTranspositionError` and :class:`KeyError`
    so that existing ``except KeyError`` handlers continue to work.

    Example::

        from sensor_transposition.exceptions import SensorNotFoundError

        try:
            sensor = collection.get_sensor("unknown")
        except SensorNotFoundError as exc:
            print(exc)  # "Sensor 'unknown' not found …"
    """


class BagError(SensorTranspositionError, RuntimeError):
    """Raised for errors related to bag file operations (:class:`~sensor_transposition.rosbag.BagWriter` / :class:`~sensor_transposition.rosbag.BagReader`).

    Inherits from both :class:`SensorTranspositionError` and
    :class:`RuntimeError` so that existing ``except RuntimeError`` handlers
    continue to work.

    Typical causes include writing to a closed :class:`~sensor_transposition.rosbag.BagWriter`
    or reading from a closed :class:`~sensor_transposition.rosbag.BagReader`.

    Example::

        from sensor_transposition.exceptions import BagError

        try:
            bag.write("/lidar", timestamp, payload)
        except BagError as exc:
            print(f"Bag write failed: {exc}")
    """


class CalibrationError(SensorTranspositionError, ValueError):
    """Raised for errors in calibration operations (e.g. :mod:`sensor_transposition.calibration`).

    Inherits from both :class:`SensorTranspositionError` and
    :class:`ValueError` so that existing ``except ValueError`` handlers
    continue to work.

    Example::

        from sensor_transposition.exceptions import CalibrationError

        try:
            result = calibrate_lidar_camera(lidar_pts, image_pts)
        except CalibrationError as exc:
            print(f"Calibration failed: {exc}")
    """
