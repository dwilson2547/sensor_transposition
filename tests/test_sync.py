"""Tests for the sync module (multi-sensor time synchronisation)."""

import numpy as np
import pytest

from sensor_transposition.sync import (
    SensorSynchroniser,
    apply_time_offset,
    find_nearest_indices,
    interpolate_timestamps,
)


# ---------------------------------------------------------------------------
# apply_time_offset
# ---------------------------------------------------------------------------


class TestApplyTimeOffset:
    def test_zero_offset_unchanged(self):
        ts = np.array([0.0, 1.0, 2.0])
        result = apply_time_offset(ts, 0.0)
        np.testing.assert_array_equal(result, ts)

    def test_positive_offset_subtracted(self):
        ts = np.array([1.0, 2.0, 3.0])
        result = apply_time_offset(ts, 0.5)
        np.testing.assert_allclose(result, [0.5, 1.5, 2.5])

    def test_negative_offset(self):
        ts = np.array([1.0, 2.0, 3.0])
        result = apply_time_offset(ts, -0.1)
        np.testing.assert_allclose(result, [1.1, 2.1, 3.1])

    def test_returns_float64(self):
        ts = [0, 1, 2]
        result = apply_time_offset(ts, 0.0)
        assert result.dtype == np.float64

    def test_list_input(self):
        result = apply_time_offset([0.0, 1.0], 0.25)
        np.testing.assert_allclose(result, [-0.25, 0.75])

    def test_single_element(self):
        result = apply_time_offset([5.0], 1.0)
        np.testing.assert_allclose(result, [4.0])

    def test_empty_array(self):
        result = apply_time_offset([], 1.0)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# find_nearest_indices
# ---------------------------------------------------------------------------


class TestFindNearestIndices:
    def _times(self):
        return np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    def test_exact_match(self):
        idx = find_nearest_indices(self._times(), np.array([2.0]))
        assert idx[0] == 2

    def test_closer_to_left(self):
        idx = find_nearest_indices(self._times(), np.array([1.3]))
        assert idx[0] == 1

    def test_closer_to_right(self):
        idx = find_nearest_indices(self._times(), np.array([1.7]))
        assert idx[0] == 2

    def test_equidistant_returns_left(self):
        idx = find_nearest_indices(self._times(), np.array([1.5]))
        assert idx[0] == 1

    def test_below_range_clamps_to_first(self):
        idx = find_nearest_indices(self._times(), np.array([-100.0]))
        assert idx[0] == 0

    def test_above_range_clamps_to_last(self):
        idx = find_nearest_indices(self._times(), np.array([100.0]))
        assert idx[0] == 4

    def test_multiple_queries(self):
        # 0.1 → nearest is 0 (index 0); 1.9 → nearest is 2.0 (index 2);
        # 3.5 is equidistant between 3.0 and 4.0 → returns left (index 3)
        idx = find_nearest_indices(self._times(), np.array([0.1, 1.9, 3.5]))
        assert list(idx) == [0, 2, 3]

    def test_single_element_times(self):
        idx = find_nearest_indices(np.array([5.0]), np.array([0.0, 10.0]))
        np.testing.assert_array_equal(idx, [0, 0])

    def test_empty_sorted_times_raises(self):
        with pytest.raises(ValueError, match="empty"):
            find_nearest_indices([], [1.0])

    def test_returns_integer_array(self):
        idx = find_nearest_indices(self._times(), np.array([1.0]))
        assert np.issubdtype(idx.dtype, np.integer)


# ---------------------------------------------------------------------------
# interpolate_timestamps
# ---------------------------------------------------------------------------


class TestInterpolateTimestamps:
    def _scalar_stream(self):
        times = np.array([0.0, 1.0, 2.0, 3.0])
        data = np.array([10.0, 20.0, 30.0, 40.0])
        return times, data

    def _vector_stream(self):
        times = np.array([0.0, 1.0, 2.0])
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        return times, data

    # --- linear scalar ---

    def test_linear_scalar_exact_timestamps(self):
        times, data = self._scalar_stream()
        result = interpolate_timestamps(times, data, times, method="linear")
        np.testing.assert_allclose(result, data)

    def test_linear_scalar_midpoint(self):
        times, data = self._scalar_stream()
        result = interpolate_timestamps(times, data, np.array([0.5]), method="linear")
        np.testing.assert_allclose(result, [15.0])

    def test_linear_scalar_clamps_below(self):
        times, data = self._scalar_stream()
        result = interpolate_timestamps(times, data, np.array([-1.0]), method="linear")
        np.testing.assert_allclose(result, [10.0])

    def test_linear_scalar_clamps_above(self):
        times, data = self._scalar_stream()
        result = interpolate_timestamps(times, data, np.array([10.0]), method="linear")
        np.testing.assert_allclose(result, [40.0])

    # --- linear vector ---

    def test_linear_vector_exact_timestamps(self):
        times, data = self._vector_stream()
        result = interpolate_timestamps(times, data, times, method="linear")
        np.testing.assert_allclose(result, data)

    def test_linear_vector_midpoint(self):
        times, data = self._vector_stream()
        result = interpolate_timestamps(times, data, np.array([0.5]), method="linear")
        np.testing.assert_allclose(result, [[2.0, 3.0]])

    def test_linear_vector_output_shape(self):
        times, data = self._vector_stream()
        result = interpolate_timestamps(times, data, np.array([0.0, 1.0, 2.0]))
        assert result.shape == (3, 2)

    # --- nearest scalar ---

    def test_nearest_scalar_exact(self):
        times, data = self._scalar_stream()
        result = interpolate_timestamps(times, data, times, method="nearest")
        np.testing.assert_allclose(result, data)

    def test_nearest_scalar_midpoint_rounds_left(self):
        times, data = self._scalar_stream()
        # 0.5 is equidistant from 0.0 and 1.0 → should pick left (index 0)
        result = interpolate_timestamps(times, data, np.array([0.5]), method="nearest")
        np.testing.assert_allclose(result, [10.0])

    def test_nearest_scalar_closer_to_right(self):
        times, data = self._scalar_stream()
        result = interpolate_timestamps(times, data, np.array([0.8]), method="nearest")
        np.testing.assert_allclose(result, [20.0])

    # --- nearest vector ---

    def test_nearest_vector(self):
        times, data = self._vector_stream()
        result = interpolate_timestamps(times, data, np.array([0.4]), method="nearest")
        np.testing.assert_allclose(result, [[1.0, 2.0]])

    # --- error cases ---

    def test_unknown_method_raises(self):
        times, data = self._scalar_stream()
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            interpolate_timestamps(times, data, times, method="cubic")

    def test_mismatched_length_raises(self):
        times = np.array([0.0, 1.0, 2.0])
        data = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="same length"):
            interpolate_timestamps(times, data, times)

    def test_default_method_is_linear(self):
        times, data = self._scalar_stream()
        result = interpolate_timestamps(times, data, np.array([0.5]))
        np.testing.assert_allclose(result, [15.0])


# ---------------------------------------------------------------------------
# SensorSynchroniser
# ---------------------------------------------------------------------------


class TestSensorSynchroniser:
    def _sync_with_two_streams(self) -> SensorSynchroniser:
        sync = SensorSynchroniser()
        # lidar at 10 Hz
        lidar_times = np.array([0.0, 0.1, 0.2, 0.3])
        lidar_data = np.array([1.0, 2.0, 3.0, 4.0])
        sync.add_stream("lidar", lidar_times, lidar_data, time_offset_sec=0.0)
        # imu at 10 Hz, 0.05 s ahead of reference
        imu_times = np.array([0.05, 0.15, 0.25, 0.35])
        imu_data = np.array([10.0, 20.0, 30.0, 40.0])
        sync.add_stream("imu", imu_times, imu_data, time_offset_sec=0.05)
        return sync

    def test_len(self):
        sync = self._sync_with_two_streams()
        assert len(sync) == 2

    def test_contains(self):
        sync = self._sync_with_two_streams()
        assert "lidar" in sync
        assert "imu" in sync
        assert "radar" not in sync

    def test_stream_names_sorted(self):
        sync = self._sync_with_two_streams()
        assert sync.stream_names == ["imu", "lidar"]

    def test_get_timestamps_no_offset(self):
        sync = SensorSynchroniser()
        ts = np.array([1.0, 2.0, 3.0])
        sync.add_stream("cam", ts, np.zeros(3))
        np.testing.assert_allclose(sync.get_timestamps("cam"), ts)

    def test_get_timestamps_with_offset(self):
        sync = SensorSynchroniser()
        ts = np.array([1.05, 2.05, 3.05])
        sync.add_stream("cam", ts, np.zeros(3), time_offset_sec=0.05)
        np.testing.assert_allclose(sync.get_timestamps("cam"), [1.0, 2.0, 3.0])

    def test_get_timestamps_unknown_stream_raises(self):
        sync = SensorSynchroniser()
        with pytest.raises(KeyError, match="'missing'"):
            sync.get_timestamps("missing")

    def test_add_and_remove_stream(self):
        sync = SensorSynchroniser()
        sync.add_stream("s1", [0.0], [1.0])
        assert "s1" in sync
        sync.remove_stream("s1")
        assert "s1" not in sync

    # --- interpolate ---

    def test_interpolate_linear(self):
        sync = self._sync_with_two_streams()
        query = np.array([0.0, 0.1, 0.2, 0.3])
        result = sync.interpolate("lidar", query)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0])

    def test_interpolate_unknown_stream_raises(self):
        sync = self._sync_with_two_streams()
        with pytest.raises(KeyError, match="'ghost'"):
            sync.interpolate("ghost", np.array([0.0]))

    def test_interpolate_imu_time_offset_applied(self):
        """IMU offset converts its timestamps to reference clock."""
        sync = self._sync_with_two_streams()
        # After offset correction imu times are [0.0, 0.1, 0.2, 0.3]
        result = sync.interpolate("imu", np.array([0.0, 0.1]))
        np.testing.assert_allclose(result, [10.0, 20.0])

    def test_interpolate_nearest(self):
        sync = self._sync_with_two_streams()
        result = sync.interpolate("lidar", np.array([0.04]), method="nearest")
        np.testing.assert_allclose(result, [1.0])

    def test_interpolate_method_override(self):
        sync = SensorSynchroniser(method="nearest")
        sync.add_stream("s", np.array([0.0, 1.0]), np.array([0.0, 10.0]))
        # Override to linear
        result = sync.interpolate("s", np.array([0.5]), method="linear")
        np.testing.assert_allclose(result, [5.0])

    # --- synchronise ---

    def test_synchronise_returns_all_streams(self):
        sync = self._sync_with_two_streams()
        ref = np.array([0.0, 0.1, 0.2, 0.3])
        aligned = sync.synchronise(ref)
        assert set(aligned.keys()) == {"lidar", "imu"}

    def test_synchronise_correct_length(self):
        sync = self._sync_with_two_streams()
        ref = np.array([0.0, 0.1, 0.2])
        aligned = sync.synchronise(ref)
        assert aligned["lidar"].shape == (3,)
        assert aligned["imu"].shape == (3,)

    def test_synchronise_lidar_values(self):
        sync = self._sync_with_two_streams()
        ref = np.array([0.0, 0.1, 0.2, 0.3])
        aligned = sync.synchronise(ref)
        np.testing.assert_allclose(aligned["lidar"], [1.0, 2.0, 3.0, 4.0])

    def test_synchronise_imu_values_after_offset(self):
        sync = self._sync_with_two_streams()
        ref = np.array([0.0, 0.1, 0.2, 0.3])
        aligned = sync.synchronise(ref)
        np.testing.assert_allclose(aligned["imu"], [10.0, 20.0, 30.0, 40.0])

    def test_synchronise_nearest_method(self):
        sync = self._sync_with_two_streams()
        ref = np.array([0.04, 0.14])
        aligned = sync.synchronise(ref, method="nearest")
        np.testing.assert_allclose(aligned["lidar"], [1.0, 2.0])

    def test_synchronise_vector_data(self):
        sync = SensorSynchroniser()
        times = np.array([0.0, 1.0, 2.0])
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        sync.add_stream("pts", times, data)
        aligned = sync.synchronise(np.array([0.0, 0.5, 1.0]))
        assert aligned["pts"].shape == (3, 2)
        np.testing.assert_allclose(aligned["pts"][1], [2.0, 3.0])

    def test_empty_synchroniser(self):
        sync = SensorSynchroniser()
        aligned = sync.synchronise(np.array([0.0, 1.0]))
        assert aligned == {}

    # --- constructor method validation ---

    def test_invalid_default_method_raises(self):
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            SensorSynchroniser(method="cubic")

    def test_default_method_nearest(self):
        sync = SensorSynchroniser(method="nearest")
        sync.add_stream("s", np.array([0.0, 1.0]), np.array([0.0, 10.0]))
        result = sync.interpolate("s", np.array([0.4]))
        # nearest to 0.4 is index 0 (value 0.0)
        np.testing.assert_allclose(result, [0.0])

    # --- integration: use with SensorCollection time offsets ---

    def test_integration_with_sensor_collection_offsets(self):
        """Verify SensorSynchroniser correctly consumes SensorCollection offsets."""
        from sensor_transposition.sensor_collection import Sensor, SensorCollection

        lidar = Sensor(
            name="lidar",
            sensor_type="lidar",
            coordinate_system="FLU",
            time_offset_sec=0.0,
        )
        imu = Sensor(
            name="imu",
            sensor_type="imu",
            coordinate_system="FLU",
            time_offset_sec=0.02,
        )
        collection = SensorCollection([lidar, imu])

        sync = SensorSynchroniser()
        lidar_times = np.array([0.0, 0.1, 0.2])
        sync.add_stream(
            "lidar",
            lidar_times,
            np.array([1.0, 2.0, 3.0]),
            time_offset_sec=collection.get_sensor("lidar").time_offset_sec,
        )
        imu_times = np.array([0.02, 0.12, 0.22])
        sync.add_stream(
            "imu",
            imu_times,
            np.array([10.0, 20.0, 30.0]),
            time_offset_sec=collection.get_sensor("imu").time_offset_sec,
        )

        # After offset correction imu times should be [0.0, 0.1, 0.2]
        np.testing.assert_allclose(
            sync.get_timestamps("imu"), [0.0, 0.1, 0.2], atol=1e-12
        )

        aligned = sync.synchronise(lidar_times)
        np.testing.assert_allclose(aligned["imu"], [10.0, 20.0, 30.0])
