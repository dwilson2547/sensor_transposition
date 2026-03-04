"""Tests for the sensor_transposition exception hierarchy."""

import pytest

from sensor_transposition.exceptions import (
    BagError,
    CalibrationError,
    SensorNotFoundError,
    SensorTranspositionError,
)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """Verify that each exception type has the correct inheritance chain."""

    def test_base_inherits_from_exception(self):
        assert issubclass(SensorTranspositionError, Exception)

    def test_sensor_not_found_inherits_from_base(self):
        assert issubclass(SensorNotFoundError, SensorTranspositionError)

    def test_sensor_not_found_inherits_from_key_error(self):
        """Backward-compatibility: existing ``except KeyError`` blocks still work."""
        assert issubclass(SensorNotFoundError, KeyError)

    def test_bag_error_inherits_from_base(self):
        assert issubclass(BagError, SensorTranspositionError)

    def test_bag_error_inherits_from_runtime_error(self):
        """Backward-compatibility: existing ``except RuntimeError`` blocks still work."""
        assert issubclass(BagError, RuntimeError)

    def test_calibration_error_inherits_from_base(self):
        assert issubclass(CalibrationError, SensorTranspositionError)

    def test_calibration_error_inherits_from_value_error(self):
        """Backward-compatibility: existing ``except ValueError`` blocks still work."""
        assert issubclass(CalibrationError, ValueError)

    def test_base_catches_all_library_exceptions(self):
        """A single ``except SensorTranspositionError`` catches every subclass."""
        for exc_cls in (SensorNotFoundError, BagError, CalibrationError):
            with pytest.raises(SensorTranspositionError):
                raise exc_cls("test message")


# ---------------------------------------------------------------------------
# SensorNotFoundError raised by SensorCollection.get_sensor
# ---------------------------------------------------------------------------


class TestSensorNotFoundError:
    def test_get_sensor_raises_sensor_not_found(self):
        from sensor_transposition.sensor_collection import SensorCollection

        col = SensorCollection()
        with pytest.raises(SensorNotFoundError, match="missing_cam"):
            col.get_sensor("missing_cam")

    def test_get_sensor_still_raises_key_error(self):
        """Backward-compatible: existing code that catches KeyError still works."""
        from sensor_transposition.sensor_collection import SensorCollection

        col = SensorCollection()
        with pytest.raises(KeyError):
            col.get_sensor("no_such_sensor")


# ---------------------------------------------------------------------------
# BagError raised by BagWriter / BagReader
# ---------------------------------------------------------------------------


class TestBagError:
    def test_bag_writer_closed_raises_bag_error(self, tmp_path):
        from sensor_transposition.rosbag import BagWriter

        bag = BagWriter(tmp_path / "test.sbag")
        bag.close()
        with pytest.raises(BagError, match="closed"):
            bag.write("/topic", 1.0, {"v": 1})

    def test_bag_writer_closed_still_raises_runtime_error(self, tmp_path):
        """Backward-compatible: existing code that catches RuntimeError still works."""
        from sensor_transposition.rosbag import BagWriter

        bag = BagWriter(tmp_path / "test.sbag")
        bag.close()
        with pytest.raises(RuntimeError):
            bag.write("/topic", 1.0, {"v": 1})

    def test_bag_reader_closed_raises_bag_error(self, tmp_path):
        from sensor_transposition.rosbag import BagReader, BagWriter

        path = tmp_path / "test.sbag"
        with BagWriter(path):
            pass

        reader = BagReader(path)
        reader.close()
        with pytest.raises(BagError, match="closed"):
            list(reader.read_messages())

    def test_bag_reader_closed_still_raises_runtime_error(self, tmp_path):
        """Backward-compatible: existing code that catches RuntimeError still works."""
        from sensor_transposition.rosbag import BagReader, BagWriter

        path = tmp_path / "test.sbag"
        with BagWriter(path):
            pass

        reader = BagReader(path)
        reader.close()
        with pytest.raises(RuntimeError):
            list(reader.read_messages())


# ---------------------------------------------------------------------------
# Exported from top-level package
# ---------------------------------------------------------------------------


class TestTopLevelExports:
    def test_exceptions_exported_from_package(self):
        import sensor_transposition as st

        assert hasattr(st, "SensorTranspositionError")
        assert hasattr(st, "SensorNotFoundError")
        assert hasattr(st, "BagError")
        assert hasattr(st, "CalibrationError")
        assert hasattr(st, "exceptions")

    def test_exceptions_in_all(self):
        import sensor_transposition as st

        for name in ("SensorTranspositionError", "SensorNotFoundError", "BagError", "CalibrationError"):
            assert name in st.__all__, f"{name} missing from __all__"
