"""
rosbag.py

Lightweight multi-sensor bag recording and playback utilities.

This module provides a self-contained bag file format for recording and
replaying timestamped, topic-based multi-sensor data — the data-capture
primitive needed before a full SLAM pipeline can run offline.

The bag format is a simple binary container inspired by ROS bag / MCAP
concepts.  It does **not** require a ROS installation or any external
dependency beyond the Python standard library; all data payloads are
serialised as UTF-8 JSON so the files can be inspected with any text editor
(after stripping the binary header).

Design goals
------------
* **No extra dependencies** – pure Python standard library (``struct``,
  ``json``, ``io``, ``pathlib``).
* **Topic-based** – each message is associated with a named channel (e.g.
  ``"/lidar/points"``, ``"/camera/image"``, ``"/imu/data"``).
* **Timestamped** – every message carries a ``float64`` timestamp (seconds
  on the reference clock) so :class:`~sensor_transposition.sync.SensorSynchroniser`
  can later realign the streams.
* **Streaming write** – :class:`BagWriter` appends messages one-by-one;
  the file is never held fully in memory.
* **Random-access read** – :class:`BagReader` builds a lightweight in-memory
  index on open so that :meth:`~BagReader.read_messages` can efficiently
  filter by topic and / or time range.

File format
-----------
::

    [MAGIC  : 9 bytes] b'\\x89SBAG\\r\\n\\x1a\\n'
    [VERSION: 1 byte ] 0x01
    [MESSAGES: variable]
        Each message record:
            record_size : uint32 LE  — bytes following this field
            topic_len   : uint16 LE  — byte length of the topic string
            topic       : topic_len bytes, UTF-8
            timestamp   : float64 LE — seconds since reference epoch
            payload     : (record_size - 2 - topic_len - 8) bytes, UTF-8 JSON

The trailing part of the file (after all message records) is an optional
4-byte footer sentinel ``b'SEND'`` written by :class:`BagWriter` on clean
close.  Readers tolerate files without the footer (e.g. from a crash during
recording).

Typical usage
-------------
**Recording**::

    from sensor_transposition.rosbag import BagWriter

    with BagWriter("session.sbag") as bag:
        for frame_pose, lidar_pts in zip(poses, lidar_scans):
            bag.write("/lidar/points",   frame_pose.timestamp,
                      {"xyz": lidar_pts.tolist()})
            bag.write("/pose/ego",        frame_pose.timestamp,
                      frame_pose.to_dict())

**Playback**::

    from sensor_transposition.rosbag import BagReader

    with BagReader("session.sbag") as bag:
        print("Topics:", bag.topics)
        print("Duration:", bag.end_time - bag.start_time, "s")

        for msg in bag.read_messages(topics=["/pose/ego"],
                                     start_time=100.0, end_time=200.0):
            print(msg.topic, msg.timestamp, msg.data)

Integration with sensor_transposition
--------------------------------------
::

    from sensor_transposition.frame_pose import FramePose
    from sensor_transposition.rosbag import BagReader

    poses = []
    with BagReader("session.sbag") as bag:
        for msg in bag.read_messages(topics=["/pose/ego"]):
            poses.append(FramePose.from_dict(msg.data))
"""

from __future__ import annotations

import io
import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Union


# ---------------------------------------------------------------------------
# File format constants
# ---------------------------------------------------------------------------

#: Magic bytes at the start of every bag file (9 bytes).
_MAGIC: bytes = b"\x89SBAG\r\n\x1a\n"

#: Current file format version.
_VERSION: int = 1

#: Footer sentinel written by :class:`BagWriter` on clean close (4 bytes).
_FOOTER: bytes = b"SEND"

# Struct formats (all little-endian):
#   record prefix  → uint32 record_size + uint16 topic_len = 6 bytes
#   timestamp      → float64 = 8 bytes
_RECORD_PREFIX_FMT = "<IH"   # uint32 + uint16
_RECORD_PREFIX_SIZE = struct.calcsize(_RECORD_PREFIX_FMT)   # 6

_TIMESTAMP_FMT = "<d"        # float64
_TIMESTAMP_SIZE = struct.calcsize(_TIMESTAMP_FMT)            # 8


# ---------------------------------------------------------------------------
# BagMessage
# ---------------------------------------------------------------------------


@dataclass
class BagMessage:
    """A single timestamped message recorded on a named topic.

    Attributes:
        topic: Channel name, e.g. ``"/lidar/points"`` or ``"/imu/data"``.
        timestamp: Message timestamp in seconds on the reference clock.
        data: Message payload as a plain Python :class:`dict`.  All values
            must be JSON-serialisable (numbers, strings, lists, nested dicts).
    """

    topic: str
    timestamp: float
    data: dict


# ---------------------------------------------------------------------------
# BagWriter
# ---------------------------------------------------------------------------


class BagWriter:
    """Writes timestamped messages to a bag file.

    Messages are appended to the file sequentially; the file is never held
    fully in memory.  :class:`BagWriter` supports the context-manager
    protocol — the file is closed (and the footer written) on ``__exit__``.

    Args:
        path: Filesystem path for the output bag file.  The file is created
            or **truncated** on open.

    Raises:
        IOError: If the file cannot be opened for writing.

    Example::

        import numpy as np
        from sensor_transposition.rosbag import BagWriter

        with BagWriter("recording.sbag") as bag:
            bag.write("/lidar/points", 1.0, {"xyz": [[0, 1, 2], [3, 4, 5]]})
            bag.write("/imu/data",     1.0, {"accel": [0.0, 0.0, 9.81],
                                             "gyro":  [0.0, 0.0, 0.0]})
    """

    def __init__(self, path: Union[str, os.PathLike]) -> None:
        self._path = Path(path)
        self._file: Optional[io.BufferedWriter] = None
        self._message_count: int = 0
        self._open()

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "BagWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def message_count(self) -> int:
        """Number of messages written so far."""
        return self._message_count

    def write(self, topic: str, timestamp: float, data: dict) -> None:
        """Append a message to the bag file.

        Args:
            topic: Channel name (e.g. ``"/lidar/points"``).  Must be a
                non-empty string.
            timestamp: Message timestamp in seconds (reference clock).
            data: Payload dict.  Must be JSON-serialisable.

        Raises:
            ValueError: If *topic* is empty or *data* is not a ``dict``.
            TypeError: If *data* contains non-JSON-serialisable values.
            RuntimeError: If the writer has already been closed.
        """
        if self._file is None or self._file.closed:
            raise RuntimeError("BagWriter is closed.")
        if not topic:
            raise ValueError("topic must be a non-empty string.")
        if not isinstance(data, dict):
            raise ValueError(
                f"data must be a dict, got {type(data).__name__!r}."
            )

        topic_bytes = topic.encode("utf-8")
        payload_bytes = json.dumps(data, separators=(",", ":")).encode("utf-8")

        # The uint32 record_size field counts bytes *after* itself.
        # Layout after the uint32: uint16 topic_len | topic | timestamp | payload
        record_body_size = (
            2  # uint16 topic_len
            + len(topic_bytes)
            + _TIMESTAMP_SIZE
            + len(payload_bytes)
        )

        header = struct.pack(
            _RECORD_PREFIX_FMT,
            record_body_size,
            len(topic_bytes),
        )
        ts_bytes = struct.pack(_TIMESTAMP_FMT, float(timestamp))

        self._file.write(header)
        self._file.write(topic_bytes)
        self._file.write(ts_bytes)
        self._file.write(payload_bytes)
        self._message_count += 1

    def close(self) -> None:
        """Flush and close the bag file, writing the footer sentinel.

        Calling :meth:`close` more than once is safe (subsequent calls are
        no-ops).
        """
        if self._file is not None and not self._file.closed:
            self._file.write(_FOOTER)
            self._file.flush()
            self._file.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Create the file and write the bag header."""
        self._file = open(self._path, "wb")  # noqa: WPS515
        self._file.write(_MAGIC)
        self._file.write(bytes([_VERSION]))


# ---------------------------------------------------------------------------
# BagReader
# ---------------------------------------------------------------------------


@dataclass
class _IndexEntry:
    """Internal index record for one message in a bag file."""

    topic: str
    timestamp: float
    offset: int          # byte offset of the record_size uint32


class BagReader:
    """Reads and replays messages from a bag file.

    On construction (or :meth:`open`) the reader performs a single sequential
    scan of the file to build an in-memory index.  Subsequent calls to
    :meth:`read_messages` use the index to return only the requested messages.

    Args:
        path: Path to the bag file produced by :class:`BagWriter`.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file does not start with the expected magic bytes.

    Example::

        from sensor_transposition.rosbag import BagReader

        with BagReader("recording.sbag") as bag:
            print("Topics:", bag.topics)
            for msg in bag.read_messages(topics=["/lidar/points"]):
                points = msg.data["xyz"]
    """

    def __init__(self, path: Union[str, os.PathLike]) -> None:
        self._path = Path(path)
        self._file: Optional[io.BufferedReader] = None
        self._index: List[_IndexEntry] = []
        self._open()

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "BagReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def topics(self) -> List[str]:
        """Sorted list of unique topic names present in the bag."""
        return sorted({e.topic for e in self._index})

    @property
    def message_count(self) -> int:
        """Total number of messages in the bag."""
        return len(self._index)

    @property
    def start_time(self) -> Optional[float]:
        """Timestamp of the earliest message, or ``None`` if the bag is empty."""
        if not self._index:
            return None
        return min(e.timestamp for e in self._index)

    @property
    def end_time(self) -> Optional[float]:
        """Timestamp of the latest message, or ``None`` if the bag is empty."""
        if not self._index:
            return None
        return max(e.timestamp for e in self._index)

    @property
    def topic_message_counts(self) -> Dict[str, int]:
        """Mapping from topic name to the number of messages on that topic."""
        counts: Dict[str, int] = {}
        for entry in self._index:
            counts[entry.topic] = counts.get(entry.topic, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_messages(
        self,
        topics: Optional[Union[str, List[str]]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Iterator[BagMessage]:
        """Iterate over messages in the bag, optionally filtered.

        Messages are yielded in the order they were written (which is
        typically chronological, but the order is not enforced).

        Args:
            topics: A single topic name or a list of topic names to include.
                ``None`` (default) returns messages from *all* topics.
            start_time: If provided, only messages with
                ``timestamp >= start_time`` are returned.
            end_time: If provided, only messages with
                ``timestamp <= end_time`` are returned.

        Yields:
            :class:`BagMessage` objects in recording order.

        Raises:
            RuntimeError: If the reader has been closed.
        """
        if self._file is None or self._file.closed:
            raise RuntimeError("BagReader is closed.")

        # Normalise topics filter.
        topic_set: Optional[set[str]] = None
        if topics is not None:
            if isinstance(topics, str):
                topic_set = {topics}
            else:
                topic_set = set(topics)

        for entry in self._index:
            # Apply filters.
            if topic_set is not None and entry.topic not in topic_set:
                continue
            if start_time is not None and entry.timestamp < start_time:
                continue
            if end_time is not None and entry.timestamp > end_time:
                continue

            # Seek to the record and decode it.
            self._file.seek(entry.offset)
            msg = self._read_record()
            if msg is not None:
                yield msg

    def close(self) -> None:
        """Close the underlying file.  Safe to call multiple times."""
        if self._file is not None and not self._file.closed:
            self._file.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Open the bag file and build the in-memory index."""
        self._file = open(self._path, "rb")  # noqa: WPS515

        # Validate magic bytes.
        magic = self._file.read(len(_MAGIC))
        if magic != _MAGIC:
            self._file.close()
            raise ValueError(
                f"File {self._path!r} does not appear to be a valid bag file "
                "(magic bytes mismatch)."
            )

        # Read version byte.
        version_byte = self._file.read(1)
        if not version_byte:
            return
        version = version_byte[0]
        if version != _VERSION:
            # Future versions may be added; warn but do not raise.
            pass

        # Sequential scan to build the index.
        self._index = []
        while True:
            offset = self._file.tell()
            prefix_bytes = self._file.read(_RECORD_PREFIX_SIZE)

            if len(prefix_bytes) < _RECORD_PREFIX_SIZE:
                # End of file (or footer sentinel).
                break

            # Check if we've hit the SEND footer.
            if prefix_bytes[:4] == _FOOTER:
                break

            record_body_size, topic_len = struct.unpack(
                _RECORD_PREFIX_FMT, prefix_bytes
            )

            topic_bytes = self._file.read(topic_len)
            if len(topic_bytes) < topic_len:
                break   # truncated file

            ts_bytes = self._file.read(_TIMESTAMP_SIZE)
            if len(ts_bytes) < _TIMESTAMP_SIZE:
                break

            timestamp = struct.unpack(_TIMESTAMP_FMT, ts_bytes)[0]
            topic = topic_bytes.decode("utf-8")

            self._index.append(
                _IndexEntry(topic=topic, timestamp=timestamp, offset=offset)
            )

            # Skip the payload bytes to advance to the next record.
            payload_size = record_body_size - 2 - topic_len - _TIMESTAMP_SIZE
            self._file.seek(payload_size, io.SEEK_CUR)

    def _read_record(self) -> Optional[BagMessage]:
        """Read and decode the record at the current file position."""
        prefix_bytes = self._file.read(_RECORD_PREFIX_SIZE)
        if len(prefix_bytes) < _RECORD_PREFIX_SIZE:
            return None

        record_body_size, topic_len = struct.unpack(
            _RECORD_PREFIX_FMT, prefix_bytes
        )

        topic_bytes = self._file.read(topic_len)
        if len(topic_bytes) < topic_len:
            return None

        ts_bytes = self._file.read(_TIMESTAMP_SIZE)
        if len(ts_bytes) < _TIMESTAMP_SIZE:
            return None

        payload_size = record_body_size - 2 - topic_len - _TIMESTAMP_SIZE
        payload_bytes = self._file.read(payload_size)
        if len(payload_bytes) < payload_size:
            return None

        topic = topic_bytes.decode("utf-8")
        timestamp = struct.unpack(_TIMESTAMP_FMT, ts_bytes)[0]
        data = json.loads(payload_bytes.decode("utf-8"))

        return BagMessage(topic=topic, timestamp=timestamp, data=data)
