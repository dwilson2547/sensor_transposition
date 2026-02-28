"""Tests for frame_pose module."""

import math
import os
import tempfile

import numpy as np
import pytest
import yaml

from sensor_transposition.frame_pose import FramePose, FramePoseSequence


# ---------------------------------------------------------------------------
# FramePose tests
# ---------------------------------------------------------------------------


class TestFramePose:
    def _make_pose(self, timestamp: float = 0.0) -> FramePose:
        return FramePose(
            timestamp=timestamp,
            translation=[1.0, 2.0, 3.0],
            rotation=[1.0, 0.0, 0.0, 0.0],  # identity
        )

    def test_defaults(self):
        pose = FramePose(timestamp=0.0)
        assert pose.timestamp == 0.0
        assert pose.translation == [0.0, 0.0, 0.0]
        assert pose.rotation == [1.0, 0.0, 0.0, 0.0]

    def test_transform_shape(self):
        pose = self._make_pose()
        T = pose.transform
        assert T.shape == (4, 4)

    def test_transform_identity_rotation(self):
        pose = self._make_pose()
        T = pose.transform
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-10)
        np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0], atol=1e-10)

    def test_transform_homogeneous_row(self):
        pose = self._make_pose()
        T = pose.transform
        np.testing.assert_allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10)

    def test_transform_with_rotation(self):
        """90° rotation about Z axis."""
        angle = math.pi / 2
        w = math.cos(angle / 2)
        z = math.sin(angle / 2)
        pose = FramePose(timestamp=0.5, translation=[0.0, 0.0, 0.0], rotation=[w, 0.0, 0.0, z])
        T = pose.transform
        # x-axis should map to y-axis
        x_rotated = T[:3, :3] @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(x_rotated, [0.0, 1.0, 0.0], atol=1e-10)

    def test_round_trip_dict(self):
        pose = self._make_pose(timestamp=1.5)
        d = pose.to_dict()
        pose2 = FramePose.from_dict(d)
        assert pose2.timestamp == pytest.approx(pose.timestamp)
        assert pose2.translation == pytest.approx(pose.translation)
        assert pose2.rotation == pytest.approx(pose.rotation)

    def test_to_dict_structure(self):
        pose = self._make_pose(timestamp=2.0)
        d = pose.to_dict()
        assert "timestamp" in d
        assert "translation" in d
        assert "rotation" in d
        assert "quaternion" in d["rotation"]

    def test_from_dict_defaults(self):
        pose = FramePose.from_dict({"timestamp": 0.0})
        assert pose.translation == [0.0, 0.0, 0.0]
        assert pose.rotation == [1.0, 0.0, 0.0, 0.0]

    def test_covariance_default_is_none(self):
        pose = FramePose(timestamp=0.0)
        assert pose.covariance is None

    def test_covariance_set(self):
        cov = np.eye(6) * 0.01
        pose = FramePose(timestamp=0.0, covariance=cov)
        assert pose.covariance is not None
        np.testing.assert_array_equal(pose.covariance, cov)

    def test_to_dict_without_covariance(self):
        pose = FramePose(timestamp=0.0)
        d = pose.to_dict()
        assert "covariance" not in d

    def test_to_dict_with_covariance(self):
        cov = np.eye(6) * 0.5
        pose = FramePose(timestamp=0.0, covariance=cov)
        d = pose.to_dict()
        assert "covariance" in d
        assert len(d["covariance"]) == 6
        assert len(d["covariance"][0]) == 6
        np.testing.assert_allclose(np.array(d["covariance"]), cov)

    def test_round_trip_dict_with_covariance(self):
        cov = np.diag([0.1, 0.2, 0.3, 0.01, 0.02, 0.03])
        pose = FramePose(timestamp=1.0, translation=[1.0, 2.0, 3.0], rotation=[1.0, 0.0, 0.0, 0.0], covariance=cov)
        d = pose.to_dict()
        pose2 = FramePose.from_dict(d)
        assert pose2.covariance is not None
        np.testing.assert_allclose(pose2.covariance, cov)

    def test_from_dict_no_covariance_key(self):
        """from_dict without covariance key yields covariance=None."""
        pose = FramePose.from_dict({"timestamp": 1.0, "translation": [0.0, 0.0, 0.0], "rotation": {"quaternion": [1.0, 0.0, 0.0, 0.0]}})
        assert pose.covariance is None

    def test_covariance_shape(self):
        cov = np.eye(6)
        pose = FramePose(timestamp=0.0, covariance=cov)
        assert pose.covariance.shape == (6, 6)


# ---------------------------------------------------------------------------
# FramePoseSequence tests
# ---------------------------------------------------------------------------


class TestFramePoseSequence:
    def _make_sequence(self) -> FramePoseSequence:
        poses = [
            FramePose(timestamp=0.0, translation=[0.0, 0.0, 0.0], rotation=[1.0, 0.0, 0.0, 0.0]),
            FramePose(timestamp=0.1, translation=[1.0, 0.0, 0.0], rotation=[1.0, 0.0, 0.0, 0.0]),
            FramePose(timestamp=0.2, translation=[2.0, 0.0, 0.0], rotation=[1.0, 0.0, 0.0, 0.0]),
        ]
        return FramePoseSequence(frame_duration=0.1, poses=poses)

    def test_default_frame_duration(self):
        seq = FramePoseSequence()
        assert seq.frame_duration == pytest.approx(0.1)

    def test_custom_frame_duration(self):
        seq = FramePoseSequence(frame_duration=0.05)
        assert seq.frame_duration == pytest.approx(0.05)

    def test_invalid_frame_duration_raises(self):
        with pytest.raises(ValueError, match="positive"):
            FramePoseSequence(frame_duration=0.0)
        with pytest.raises(ValueError, match="positive"):
            FramePoseSequence(frame_duration=-1.0)

    def test_len(self):
        seq = self._make_sequence()
        assert len(seq) == 3

    def test_len_empty(self):
        seq = FramePoseSequence()
        assert len(seq) == 0

    def test_add_pose(self):
        seq = FramePoseSequence()
        seq.add_pose(FramePose(timestamp=0.0))
        assert len(seq) == 1

    def test_get_pose(self):
        seq = self._make_sequence()
        pose = seq.get_pose(1)
        assert pose.timestamp == pytest.approx(0.1)
        assert pose.translation == pytest.approx([1.0, 0.0, 0.0])

    def test_get_pose_index_error(self):
        seq = self._make_sequence()
        with pytest.raises(IndexError):
            seq.get_pose(10)

    def test_timestamps(self):
        seq = self._make_sequence()
        ts = seq.timestamps
        np.testing.assert_allclose(ts, [0.0, 0.1, 0.2], atol=1e-10)

    def test_timestamps_empty(self):
        seq = FramePoseSequence()
        ts = seq.timestamps
        assert len(ts) == 0

    def test_get_pose_at_timestamp(self):
        seq = self._make_sequence()
        # Exactly at the start of frame 1
        pose = seq.get_pose_at_timestamp(0.1)
        assert pose is not None
        assert pose.timestamp == pytest.approx(0.1)

    def test_get_pose_at_timestamp_within_frame(self):
        seq = self._make_sequence()
        # Midway through frame 0
        pose = seq.get_pose_at_timestamp(0.05)
        assert pose is not None
        assert pose.timestamp == pytest.approx(0.0)

    def test_get_pose_at_timestamp_not_found(self):
        seq = self._make_sequence()
        # After all frames
        pose = seq.get_pose_at_timestamp(1.0)
        assert pose is None

    def test_get_pose_at_timestamp_before_first(self):
        seq = self._make_sequence()
        pose = seq.get_pose_at_timestamp(-0.5)
        assert pose is None

    def test_iter(self):
        seq = self._make_sequence()
        poses = list(seq)
        assert len(poses) == 3
        assert poses[0].timestamp == pytest.approx(0.0)
        assert poses[2].timestamp == pytest.approx(0.2)

    def test_round_trip_dict(self):
        seq = self._make_sequence()
        d = seq.to_dict()
        seq2 = FramePoseSequence.from_dict(d)
        assert seq2.frame_duration == pytest.approx(seq.frame_duration)
        assert len(seq2) == len(seq)
        for p1, p2 in zip(seq, seq2):
            assert p1.timestamp == pytest.approx(p2.timestamp)
            assert p1.translation == pytest.approx(p2.translation)
            assert p1.rotation == pytest.approx(p2.rotation)

    def test_to_dict_structure(self):
        seq = self._make_sequence()
        d = seq.to_dict()
        assert "frame_duration" in d
        assert "poses" in d
        assert len(d["poses"]) == 3

    def test_from_dict_default_frame_duration(self):
        """from_dict uses default 0.1s when frame_duration is absent."""
        seq = FramePoseSequence.from_dict({"poses": []})
        assert seq.frame_duration == pytest.approx(0.1)

    def test_yaml_round_trip(self):
        seq = self._make_sequence()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            seq.to_yaml(path)
            seq2 = FramePoseSequence.from_yaml(path)
            assert seq2.frame_duration == pytest.approx(seq.frame_duration)
            assert len(seq2) == len(seq)
            for p1, p2 in zip(seq, seq2):
                assert p1.timestamp == pytest.approx(p2.timestamp)
                assert p1.translation == pytest.approx(p2.translation)
                assert p1.rotation == pytest.approx(p2.rotation)
        finally:
            os.unlink(path)

    def test_yaml_file_contents(self):
        """Verify the YAML output is human-readable and correct."""
        seq = FramePoseSequence(
            frame_duration=0.1,
            poses=[FramePose(timestamp=0.0, translation=[1.0, 2.0, 3.0], rotation=[1.0, 0.0, 0.0, 0.0])],
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            seq.to_yaml(path)
            with open(path) as f:
                raw = yaml.safe_load(f)
            assert raw["frame_duration"] == pytest.approx(0.1)
            assert len(raw["poses"]) == 1
            assert raw["poses"][0]["timestamp"] == pytest.approx(0.0)
        finally:
            os.unlink(path)

    def test_yaml_round_trip_with_covariance(self):
        """Pose covariance survives a YAML serialisation round-trip."""
        cov = np.diag([0.1, 0.2, 0.3, 0.01, 0.02, 0.03])
        seq = FramePoseSequence(
            frame_duration=0.1,
            poses=[
                FramePose(timestamp=0.0, translation=[0.0, 0.0, 0.0],
                          rotation=[1.0, 0.0, 0.0, 0.0], covariance=cov),
                FramePose(timestamp=0.1, translation=[1.0, 0.0, 0.0],
                          rotation=[1.0, 0.0, 0.0, 0.0]),  # no covariance
            ],
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            seq.to_yaml(path)
            seq2 = FramePoseSequence.from_yaml(path)
            p0 = seq2.get_pose(0)
            assert p0.covariance is not None
            np.testing.assert_allclose(p0.covariance, cov, atol=1e-12)
            p1 = seq2.get_pose(1)
            assert p1.covariance is None
        finally:
            os.unlink(path)
