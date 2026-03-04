"""
sync.py

Multi-sensor time synchronisation and interpolation utilities.

Provides tools to align timestamped data from different sensors onto a
common reference timeline, taking into account per-sensor clock offsets
(``time_offset_sec``) as defined in
:class:`~sensor_transposition.sensor_collection.Sensor`.

Two usage patterns are supported:

1. **Standalone functions** – operate directly on numpy arrays:

   - :func:`apply_time_offset` – convert sensor-local timestamps to the
     reference clock.
   - :func:`find_nearest_indices` – for each query time find the index of
     the nearest sample in a sorted timestamp array.
   - :func:`interpolate_timestamps` – resample a data array at arbitrary
     query timestamps using nearest-neighbour or linear interpolation.

2. **SensorSynchroniser class** – manages multiple named streams and
   resamples them all to a shared reference timeline in one call.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def apply_time_offset(
    timestamps: Union[np.ndarray, List[float]],
    time_offset_sec: float,
) -> np.ndarray:
    """Convert sensor-local timestamps to the common reference timeline.

    Subtracts *time_offset_sec* from each element of *timestamps*, yielding
    the equivalent time on the reference clock.  This matches the convention
    used in :attr:`~sensor_transposition.sensor_collection.Sensor.time_offset_sec`,
    where a **positive** offset means the sensor clock is **ahead** of the
    reference::

        t_reference = t_sensor - time_offset_sec

    Args:
        timestamps: 1-D array (or list) of sensor-local timestamps in seconds.
        time_offset_sec: The sensor's clock offset relative to the reference
            clock in seconds (from
            :attr:`~sensor_transposition.sensor_collection.Sensor.time_offset_sec`).

    Returns:
        1-D ``float64`` array of timestamps on the reference clock.
    """
    return np.asarray(timestamps, dtype=float) - float(time_offset_sec)


def find_nearest_indices(
    sorted_times: Union[np.ndarray, List[float]],
    query_times: Union[np.ndarray, List[float]],
) -> np.ndarray:
    """Find the index of the nearest sample for each query timestamp.

    For every element of *query_times*, returns the index *i* in
    *sorted_times* such that ``|sorted_times[i] - query_time|`` is minimised.
    When two adjacent samples are equidistant the left (earlier) index is
    returned.

    Args:
        sorted_times: 1-D array of **monotonically increasing** sample
            timestamps.
        query_times: 1-D array of timestamps at which the nearest index is
            required.  Need not be sorted.

    Returns:
        1-D ``int64`` array of indices into *sorted_times*, one per element
        of *query_times*.

    Raises:
        ValueError: If *sorted_times* is empty.
    """
    sorted_times = np.asarray(sorted_times, dtype=float)
    query_times = np.asarray(query_times, dtype=float)

    if sorted_times.size == 0:
        raise ValueError("sorted_times must not be empty.")

    # searchsorted gives the index where query_time would be inserted to keep
    # sorted order.  The nearest sample is either at that index or the one
    # immediately to the left.
    right_idx = np.searchsorted(sorted_times, query_times, side="left")
    right_idx = np.clip(right_idx, 0, len(sorted_times) - 1)
    left_idx = np.clip(right_idx - 1, 0, len(sorted_times) - 1)

    right_diff = np.abs(sorted_times[right_idx] - query_times)
    left_diff = np.abs(sorted_times[left_idx] - query_times)

    return np.where(left_diff <= right_diff, left_idx, right_idx).astype(np.int64)


def interpolate_timestamps(
    times: Union[np.ndarray, List[float]],
    data: Union[np.ndarray, List],
    query_times: Union[np.ndarray, List[float]],
    method: str = "linear",
) -> np.ndarray:
    """Resample sensor data at arbitrary query timestamps.

    Supports both scalar (1-D) and vector (2-D) data arrays.  For 2-D arrays
    each column is interpolated independently.

    When *query_times* fall outside the range of *times*, the boundary sample
    values are returned (no extrapolation).

    Args:
        times: 1-D array of **monotonically increasing** sample timestamps
            (in seconds, on the reference clock).
        data: Array of sample values.  Shape ``(N,)`` for scalar data or
            ``(N, D)`` for *D*-dimensional vector data, where *N* matches
            ``len(times)``.
        query_times: 1-D array of timestamps at which resampled values are
            required.
        method: Interpolation strategy.  One of:

            - ``"linear"`` *(default)* – linearly interpolate between the
              two bounding samples.
            - ``"nearest"`` – return the value of the nearest sample.

    Returns:
        Resampled array.  Shape ``(M,)`` for scalar data or ``(M, D)`` for
        vector data, where *M* is ``len(query_times)``.

    Raises:
        ValueError: If *method* is not ``"linear"`` or ``"nearest"``.
        ValueError: If *times* and *data* have incompatible lengths.
    """
    times = np.asarray(times, dtype=float)
    data = np.asarray(data, dtype=float)
    query_times = np.asarray(query_times, dtype=float)

    if len(times) != len(data):
        raise ValueError(
            f"times and data must have the same length; "
            f"got {len(times)} and {len(data)}."
        )

    if method == "nearest":
        idx = find_nearest_indices(times, query_times)
        return data[idx]

    if method == "linear":
        if data.ndim == 1:
            return np.interp(query_times, times, data)
        # 2-D: interpolate each column independently
        result = np.empty((len(query_times), data.shape[1]), dtype=float)
        for col in range(data.shape[1]):
            result[:, col] = np.interp(query_times, times, data[:, col])
        return result

    raise ValueError(
        f"Unknown interpolation method {method!r}. "
        "Expected 'linear' or 'nearest'."
    )


# ---------------------------------------------------------------------------
# SensorSynchroniser
# ---------------------------------------------------------------------------


class SensorSynchroniser:
    """Aligns timestamped data from multiple sensors onto a common timeline.

    Each sensor stream is registered with its raw timestamps and a
    ``time_offset_sec`` (taken directly from
    :attr:`~sensor_transposition.sensor_collection.Sensor.time_offset_sec`).
    The offset is applied once on registration so that all stored timestamps
    are on the shared reference clock.

    Example::

        sync = SensorSynchroniser()
        sync.add_stream("lidar", lidar_times, lidar_points,
                        time_offset_sec=collection.get_sensor("lidar").time_offset_sec)
        sync.add_stream("imu", imu_times, imu_accel,
                        time_offset_sec=collection.get_sensor("imu").time_offset_sec)

        # Interpolate all streams at LiDAR timestamps
        aligned = sync.synchronise(lidar_times)
        imu_at_lidar_times = aligned["imu"]

    Args:
        method: Default interpolation method (``"linear"`` or ``"nearest"``)
            used by :meth:`synchronise` when no per-call override is given.
    """

    def __init__(self, method: str = "linear") -> None:
        if method not in ("linear", "nearest"):
            raise ValueError(
                f"Unknown interpolation method {method!r}. "
                "Expected 'linear' or 'nearest'."
            )
        self._default_method = method
        self._streams: Dict[str, tuple] = {}

    # ------------------------------------------------------------------
    # Stream management
    # ------------------------------------------------------------------

    def add_stream(
        self,
        name: str,
        timestamps: Union[np.ndarray, List[float]],
        data: Union[np.ndarray, List],
        time_offset_sec: float = 0.0,
    ) -> None:
        """Register a sensor data stream.

        The sensor's clock offset is subtracted from *timestamps* so that
        all stored timestamps represent times on the common reference clock.

        Args:
            name: Unique stream identifier (typically the sensor name).
            timestamps: 1-D array of sensor-local sample timestamps in
                seconds (must be monotonically increasing).
            data: Sample values.  Shape ``(N,)`` for scalars or ``(N, D)``
                for vector data.
            time_offset_sec: Sensor's clock offset in seconds (see
                :attr:`~sensor_transposition.sensor_collection.Sensor.time_offset_sec`).
                Defaults to ``0.0``.
        """
        corrected = apply_time_offset(timestamps, time_offset_sec)
        self._streams[name] = (corrected, np.asarray(data, dtype=float))

    def remove_stream(self, name: str) -> None:
        """Remove a previously registered stream."""
        del self._streams[name]

    @property
    def stream_names(self) -> List[str]:
        """Sorted list of registered stream names."""
        return sorted(self._streams.keys())

    def __len__(self) -> int:
        return len(self._streams)

    def __contains__(self, name: str) -> bool:
        return name in self._streams

    # ------------------------------------------------------------------
    # Timestamp access
    # ------------------------------------------------------------------

    def get_timestamps(self, name: str) -> np.ndarray:
        """Return the reference-clock timestamps for a stream.

        These are the original sensor timestamps with the time offset
        already subtracted.

        Args:
            name: Stream name as passed to :meth:`add_stream`.

        Returns:
            1-D ``float64`` array of corrected timestamps.

        Raises:
            KeyError: If *name* is not a registered stream.
        """
        if name not in self._streams:
            raise KeyError(
                f"Stream '{name}' not found. "
                f"Available: {self.stream_names}"
            )
        return self._streams[name][0]

    def stream_start_time(self, name: str) -> float:
        """Return the earliest timestamp in stream *name* on the reference clock.

        Args:
            name: Stream name as passed to :meth:`add_stream`.

        Returns:
            Minimum timestamp (float, seconds) in the stream.

        Raises:
            KeyError: If *name* is not a registered stream.
        """
        return float(self.get_timestamps(name).min())

    def stream_end_time(self, name: str) -> float:
        """Return the latest timestamp in stream *name* on the reference clock.

        Args:
            name: Stream name as passed to :meth:`add_stream`.

        Returns:
            Maximum timestamp (float, seconds) in the stream.

        Raises:
            KeyError: If *name* is not a registered stream.
        """
        return float(self.get_timestamps(name).max())

    def temporal_overlap(self) -> Optional[tuple[float, float]]:
        """Return the time range over which **all** registered streams overlap.

        The overlap is the intersection of each stream's time range:
        ``[max(start_times), min(end_times)]``.  This is the range that can
        be safely passed to :meth:`synchronise` without boundary clamping.

        Returns:
            ``(overlap_start, overlap_end)`` tuple of reference-clock
            timestamps if the overlap is non-empty (i.e.
            ``overlap_start < overlap_end``), or ``None`` if there is no
            overlap or fewer than two streams are registered.

        Example::

            sync = SensorSynchroniser()
            sync.add_stream("lidar", lidar_times, lidar_data)
            sync.add_stream("imu", imu_times, imu_data)

            result = sync.temporal_overlap()
            if result is None:
                raise RuntimeError("Streams do not overlap — check time offsets.")
            t_start, t_end = result
        """
        if len(self._streams) < 2:
            return None
        starts = [float(ts.min()) for ts, _ in self._streams.values()]
        ends = [float(ts.max()) for ts, _ in self._streams.values()]
        overlap_start = max(starts)
        overlap_end = min(ends)
        if overlap_start >= overlap_end:
            return None
        return (overlap_start, overlap_end)

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def interpolate(
        self,
        name: str,
        query_times: Union[np.ndarray, List[float]],
        method: Optional[str] = None,
    ) -> np.ndarray:
        """Interpolate a single stream at the given query timestamps.

        Args:
            name: Stream name as passed to :meth:`add_stream`.
            query_times: 1-D array of reference-clock timestamps at which
                resampled values are required.
            method: Interpolation method (``"linear"`` or ``"nearest"``).
                Defaults to the method set in the constructor.

        Returns:
            Resampled array aligned to *query_times*.

        Raises:
            KeyError: If *name* is not a registered stream.
        """
        if name not in self._streams:
            raise KeyError(
                f"Stream '{name}' not found. "
                f"Available: {self.stream_names}"
            )
        m = method if method is not None else self._default_method
        times, data = self._streams[name]
        return interpolate_timestamps(times, data, query_times, method=m)

    def synchronise(
        self,
        reference_times: Union[np.ndarray, List[float]],
        method: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Resample all registered streams at a common set of timestamps.

        Args:
            reference_times: 1-D array of reference-clock timestamps at which
                every stream is resampled.
            method: Interpolation method (``"linear"`` or ``"nearest"``).
                Defaults to the method set in the constructor.

        Returns:
            Dictionary mapping stream name → resampled data array.  Each
            array has the same number of rows as *reference_times*.
        """
        ref = np.asarray(reference_times, dtype=float)
        return {name: self.interpolate(name, ref, method=method) for name in self._streams}

    # American-spelling alias for synchronise.
    synchronize = synchronise


# American-spelling alias for SensorSynchroniser.
SensorSynchronizer = SensorSynchroniser
