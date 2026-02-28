"""Tests for rosbag: BagWriter and BagReader."""

import json
import struct
import tempfile
from pathlib import Path

import pytest

from sensor_transposition.rosbag import (
    BagMessage,
    BagReader,
    BagWriter,
    _MAGIC,
    _VERSION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_bag(path, messages):
    """Write a list of (topic, timestamp, data) tuples to a bag file."""
    with BagWriter(path) as bag:
        for topic, ts, data in messages:
            bag.write(topic, ts, data)


def _tmp_bag(tmp_path, suffix=".sbag"):
    return tmp_path / f"test{suffix}"


# ---------------------------------------------------------------------------
# BagWriter — basic write
# ---------------------------------------------------------------------------


class TestBagWriter:
    def test_creates_file(self, tmp_path):
        p = _tmp_bag(tmp_path)
        with BagWriter(p):
            pass
        assert p.exists()

    def test_magic_bytes_present(self, tmp_path):
        p = _tmp_bag(tmp_path)
        with BagWriter(p):
            pass
        raw = p.read_bytes()
        assert raw[: len(_MAGIC)] == _MAGIC

    def test_version_byte(self, tmp_path):
        p = _tmp_bag(tmp_path)
        with BagWriter(p):
            pass
        raw = p.read_bytes()
        assert raw[len(_MAGIC)] == _VERSION

    def test_message_count_starts_zero(self, tmp_path):
        p = _tmp_bag(tmp_path)
        with BagWriter(p) as bag:
            assert bag.message_count == 0

    def test_message_count_increments(self, tmp_path):
        p = _tmp_bag(tmp_path)
        with BagWriter(p) as bag:
            bag.write("/a", 1.0, {"v": 1})
            assert bag.message_count == 1
            bag.write("/b", 2.0, {"v": 2})
            assert bag.message_count == 2

    def test_empty_topic_raises(self, tmp_path):
        p = _tmp_bag(tmp_path)
        with BagWriter(p) as bag:
            with pytest.raises(ValueError, match="topic"):
                bag.write("", 1.0, {"v": 1})

    def test_non_dict_data_raises(self, tmp_path):
        p = _tmp_bag(tmp_path)
        with BagWriter(p) as bag:
            with pytest.raises(ValueError, match="dict"):
                bag.write("/a", 1.0, [1, 2, 3])  # type: ignore[arg-type]

    def test_write_after_close_raises(self, tmp_path):
        p = _tmp_bag(tmp_path)
        bag = BagWriter(p)
        bag.close()
        with pytest.raises(RuntimeError, match="closed"):
            bag.write("/a", 1.0, {"v": 1})

    def test_close_is_idempotent(self, tmp_path):
        p = _tmp_bag(tmp_path)
        bag = BagWriter(p)
        bag.close()
        bag.close()  # second close must not raise

    def test_context_manager_closes_file(self, tmp_path):
        p = _tmp_bag(tmp_path)
        with BagWriter(p) as bag:
            bag.write("/x", 0.0, {})
        assert bag._file.closed


# ---------------------------------------------------------------------------
# BagReader — basic read
# ---------------------------------------------------------------------------


class TestBagReader:
    def test_open_valid_file(self, tmp_path):
        p = _tmp_bag(tmp_path)
        _write_bag(p, [("/a", 1.0, {"k": "v"})])
        with BagReader(p):
            pass  # should not raise

    def test_wrong_magic_raises(self, tmp_path):
        p = _tmp_bag(tmp_path)
        p.write_bytes(b"NOTABAG")
        with pytest.raises(ValueError, match="magic"):
            BagReader(p)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BagReader(tmp_path / "does_not_exist.sbag")

    def test_topics_property(self, tmp_path):
        p = _tmp_bag(tmp_path)
        _write_bag(p, [
            ("/lidar", 1.0, {}),
            ("/camera", 2.0, {}),
            ("/lidar", 3.0, {}),
        ])
        with BagReader(p) as bag:
            assert bag.topics == ["/camera", "/lidar"]

    def test_message_count_property(self, tmp_path):
        p = _tmp_bag(tmp_path)
        _write_bag(p, [("/a", 1.0, {}), ("/b", 2.0, {}), ("/c", 3.0, {})])
        with BagReader(p) as bag:
            assert bag.message_count == 3

    def test_start_end_time(self, tmp_path):
        p = _tmp_bag(tmp_path)
        _write_bag(p, [("/a", 5.0, {}), ("/b", 2.0, {}), ("/c", 8.0, {})])
        with BagReader(p) as bag:
            assert bag.start_time == pytest.approx(2.0)
            assert bag.end_time == pytest.approx(8.0)

    def test_empty_bag_start_end_none(self, tmp_path):
        p = _tmp_bag(tmp_path)
        _write_bag(p, [])
        with BagReader(p) as bag:
            assert bag.start_time is None
            assert bag.end_time is None

    def test_topic_message_counts(self, tmp_path):
        p = _tmp_bag(tmp_path)
        _write_bag(p, [
            ("/a", 1.0, {}),
            ("/a", 2.0, {}),
            ("/b", 3.0, {}),
        ])
        with BagReader(p) as bag:
            counts = bag.topic_message_counts
        assert counts["/a"] == 2
        assert counts["/b"] == 1

    def test_context_manager_closes_file(self, tmp_path):
        p = _tmp_bag(tmp_path)
        _write_bag(p, [])
        with BagReader(p) as bag:
            pass
        assert bag._file.closed

    def test_close_is_idempotent(self, tmp_path):
        p = _tmp_bag(tmp_path)
        _write_bag(p, [])
        bag = BagReader(p)
        bag.close()
        bag.close()  # second close must not raise

    def test_read_after_close_raises(self, tmp_path):
        p = _tmp_bag(tmp_path)
        _write_bag(p, [("/a", 1.0, {})])
        bag = BagReader(p)
        bag.close()
        with pytest.raises(RuntimeError, match="closed"):
            list(bag.read_messages())


# ---------------------------------------------------------------------------
# BagReader — read_messages filtering
# ---------------------------------------------------------------------------


class TestReadMessages:
    def _bag_path(self, tmp_path):
        p = tmp_path / "filter.sbag"
        _write_bag(p, [
            ("/lidar", 1.0, {"scan": "a"}),
            ("/camera", 1.5, {"img": "b"}),
            ("/lidar", 2.0, {"scan": "c"}),
            ("/imu",   2.5, {"accel": [0, 0, 9.81]}),
            ("/lidar", 3.0, {"scan": "d"}),
        ])
        return p

    def test_read_all_messages(self, tmp_path):
        p = self._bag_path(tmp_path)
        with BagReader(p) as bag:
            msgs = list(bag.read_messages())
        assert len(msgs) == 5

    def test_filter_by_single_topic_string(self, tmp_path):
        p = self._bag_path(tmp_path)
        with BagReader(p) as bag:
            msgs = list(bag.read_messages(topics="/lidar"))
        assert len(msgs) == 3
        assert all(m.topic == "/lidar" for m in msgs)

    def test_filter_by_topic_list(self, tmp_path):
        p = self._bag_path(tmp_path)
        with BagReader(p) as bag:
            msgs = list(bag.read_messages(topics=["/lidar", "/camera"]))
        assert len(msgs) == 4
        topics_found = {m.topic for m in msgs}
        assert topics_found == {"/lidar", "/camera"}

    def test_filter_by_start_time(self, tmp_path):
        p = self._bag_path(tmp_path)
        with BagReader(p) as bag:
            msgs = list(bag.read_messages(start_time=2.0))
        assert all(m.timestamp >= 2.0 for m in msgs)
        assert len(msgs) == 3

    def test_filter_by_end_time(self, tmp_path):
        p = self._bag_path(tmp_path)
        with BagReader(p) as bag:
            msgs = list(bag.read_messages(end_time=2.0))
        assert all(m.timestamp <= 2.0 for m in msgs)
        assert len(msgs) == 3

    def test_filter_by_time_range(self, tmp_path):
        p = self._bag_path(tmp_path)
        with BagReader(p) as bag:
            msgs = list(bag.read_messages(start_time=1.5, end_time=2.5))
        assert all(1.5 <= m.timestamp <= 2.5 for m in msgs)
        assert len(msgs) == 3

    def test_filter_topic_and_time(self, tmp_path):
        p = self._bag_path(tmp_path)
        with BagReader(p) as bag:
            msgs = list(bag.read_messages(topics="/lidar", start_time=2.0))
        assert len(msgs) == 2
        assert all(m.topic == "/lidar" and m.timestamp >= 2.0 for m in msgs)

    def test_nonexistent_topic_returns_empty(self, tmp_path):
        p = self._bag_path(tmp_path)
        with BagReader(p) as bag:
            msgs = list(bag.read_messages(topics="/nonexistent"))
        assert msgs == []

    def test_message_data_roundtrip(self, tmp_path):
        p = tmp_path / "rt.sbag"
        payload = {"xyz": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "label": "scan"}
        _write_bag(p, [("/lidar", 42.5, payload)])
        with BagReader(p) as bag:
            msgs = list(bag.read_messages())
        assert len(msgs) == 1
        assert msgs[0].topic == "/lidar"
        assert msgs[0].timestamp == pytest.approx(42.5)
        assert msgs[0].data == payload

    def test_message_type(self, tmp_path):
        p = tmp_path / "type.sbag"
        _write_bag(p, [("/a", 0.0, {"x": 1})])
        with BagReader(p) as bag:
            msgs = list(bag.read_messages())
        assert isinstance(msgs[0], BagMessage)


# ---------------------------------------------------------------------------
# Roundtrip — many messages
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_many_messages(self, tmp_path):
        p = tmp_path / "many.sbag"
        original = [
            (f"/topic/{i % 3}", float(i) * 0.1, {"index": i, "value": float(i) ** 2})
            for i in range(200)
        ]
        _write_bag(p, original)
        with BagReader(p) as bag:
            recovered = list(bag.read_messages())
        assert len(recovered) == len(original)
        for (topic, ts, data), msg in zip(original, recovered):
            assert msg.topic == topic
            assert msg.timestamp == pytest.approx(ts)
            assert msg.data == data

    def test_unicode_topic_and_data(self, tmp_path):
        p = tmp_path / "unicode.sbag"
        payload = {"label": "héllo wörld", "values": [1, 2, 3]}
        _write_bag(p, [("/sensor/ñ", 0.5, payload)])
        with BagReader(p) as bag:
            msgs = list(bag.read_messages())
        assert msgs[0].topic == "/sensor/ñ"
        assert msgs[0].data["label"] == "héllo wörld"

    def test_nested_data(self, tmp_path):
        p = tmp_path / "nested.sbag"
        payload = {
            "pose": {"translation": [1.0, 2.0, 3.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
            "covariance": [[1.0, 0.0], [0.0, 1.0]],
        }
        _write_bag(p, [("/pose", 99.0, payload)])
        with BagReader(p) as bag:
            msgs = list(bag.read_messages())
        assert msgs[0].data == payload

    def test_multiple_reads_consistent(self, tmp_path):
        """read_messages can be called multiple times on the same open reader."""
        p = tmp_path / "multi_read.sbag"
        _write_bag(p, [("/a", 1.0, {"n": 1}), ("/b", 2.0, {"n": 2})])
        with BagReader(p) as bag:
            first = list(bag.read_messages())
            second = list(bag.read_messages())
        assert len(first) == len(second) == 2
        assert first[0].data == second[0].data

    def test_empty_payload_dict(self, tmp_path):
        p = tmp_path / "empty_payload.sbag"
        _write_bag(p, [("/empty", 0.0, {})])
        with BagReader(p) as bag:
            msgs = list(bag.read_messages())
        assert msgs[0].data == {}

    def test_large_payload(self, tmp_path):
        """A payload with many float values should round-trip without truncation."""
        import math
        p = tmp_path / "large.sbag"
        data = {"points": [[math.sin(i * 0.01), math.cos(i * 0.01), float(i)]
                            for i in range(500)]}
        _write_bag(p, [("/lidar", 1.0, data)])
        with BagReader(p) as bag:
            msgs = list(bag.read_messages())
        assert len(msgs[0].data["points"]) == 500


# ---------------------------------------------------------------------------
# Integration with FramePose
# ---------------------------------------------------------------------------


class TestFramePoseIntegration:
    def test_frame_pose_roundtrip(self, tmp_path):
        from sensor_transposition.frame_pose import FramePose

        p = tmp_path / "pose.sbag"
        original_pose = FramePose(
            timestamp=1.23,
            translation=[1.0, 2.0, 3.0],
            rotation=[0.7071, 0.0, 0.7071, 0.0],
        )
        with BagWriter(p) as bag:
            bag.write("/pose/ego", original_pose.timestamp, original_pose.to_dict())

        with BagReader(p) as bag:
            msgs = list(bag.read_messages(topics="/pose/ego"))

        assert len(msgs) == 1
        recovered = FramePose.from_dict(msgs[0].data)
        assert recovered.timestamp == pytest.approx(original_pose.timestamp)
        assert recovered.translation == pytest.approx(original_pose.translation)
        assert recovered.rotation == pytest.approx(original_pose.rotation)
