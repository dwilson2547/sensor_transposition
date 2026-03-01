"""
submap_manager.py

Map management: keyframe selection and submap division.

For large-scale or long-duration SLAM sessions, storing and optimising every
incoming frame is computationally prohibitive.  Map management reduces the
problem size through two complementary strategies:

1. **Keyframe selection** – only retaining frames whose pose differs
   sufficiently from the last accepted keyframe (by translation or rotation),
   so the keyframe set captures the essential geometry of the trajectory
   without near-duplicate frames.

2. **Submap division** – partitioning the keyframe sequence into overlapping
   local submaps, each with its own origin pose and independent
   :class:`~sensor_transposition.point_cloud_map.PointCloudMap`.  This bounds
   the per-submap memory and optimisation cost and allows the global map to be
   assembled from independently-optimised local pieces.

Classes
-------
KeyframeSelector
    Stateful selector that accepts or rejects incoming poses based on
    translation and rotation thresholds relative to the most recently
    accepted keyframe.

Submap
    Lightweight container holding the keyframe IDs, the origin pose, and
    the accumulated local
    :class:`~sensor_transposition.point_cloud_map.PointCloudMap` for one
    submap segment.

SubmapManager
    Higher-level manager that feeds accepted keyframes into the current
    submap and creates a new submap when the current one has grown beyond a
    configurable limit.  An optional *overlap* parameter carries the last
    *k* keyframes of the finishing submap into the start of the new one,
    providing shared observations for inter-submap loop-closure and
    alignment.

Typical use-case
----------------
::

    from sensor_transposition.submap_manager import KeyframeSelector, SubmapManager
    import numpy as np

    selector = KeyframeSelector(translation_threshold=1.0, rotation_threshold_deg=10.0)
    manager  = SubmapManager(max_keyframes_per_submap=20, overlap=2)

    for frame_id, (pose, scan) in enumerate(zip(frame_poses, lidar_scans)):
        if selector.check_and_accept(pose.transform):
            manager.add_keyframe(frame_id, pose.transform, scan)

    submaps = manager.get_all_submaps()
    print(f"Created {len(submaps)} submaps with "
          f"{manager.total_keyframes} total keyframe entries.")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from sensor_transposition.point_cloud_map import PointCloudMap


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_transform(T: np.ndarray, name: str = "transform") -> None:
    """Raise ``ValueError`` if *T* is not a 4×4 matrix."""
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be a 4×4 matrix, got {T.shape}.")


def _rotation_angle_between(T1: np.ndarray, T2: np.ndarray) -> float:
    """Rotation angle (radians) between the orientations of two SE(3) matrices.

    Uses the axis-angle formula:
    ``angle = arccos((trace(R₁ᵀ R₂) − 1) / 2)``.
    """
    R_rel = T1[:3, :3].T @ T2[:3, :3]
    cos_angle = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def _translation_distance(T1: np.ndarray, T2: np.ndarray) -> float:
    """Euclidean distance between the translation parts of two SE(3) matrices."""
    return float(np.linalg.norm(T2[:3, 3] - T1[:3, 3]))


# ---------------------------------------------------------------------------
# KeyframeSelector
# ---------------------------------------------------------------------------


class KeyframeSelector:
    """Select keyframes from a stream of incoming poses.

    A new keyframe is accepted when **either**:

    * The Euclidean translation distance from the last accepted keyframe
      exceeds *translation_threshold* (metres), **or**
    * The rotation angle from the last accepted keyframe exceeds
      *rotation_threshold_deg* (degrees).

    The very first pose presented is always accepted.

    Acceptance and state-update are kept separate so that the caller can
    decide whether to commit a candidate pose:

    * :meth:`should_add_keyframe` – query without changing state.
    * :meth:`mark_accepted`       – record a pose as the new reference.
    * :meth:`check_and_accept`    – query **and** conditionally update state
      in one step (most common usage).

    Args:
        translation_threshold: Minimum translation distance (metres) between
            successive keyframes.  Must be strictly positive.
        rotation_threshold_deg: Minimum rotation angle (degrees) from the last
            keyframe to trigger acceptance.  Must be non-negative.
            Set to ``0.0`` to accept every pose that exceeds the translation
            threshold regardless of rotation.
    """

    def __init__(
        self,
        translation_threshold: float = 1.0,
        rotation_threshold_deg: float = 10.0,
    ) -> None:
        if translation_threshold <= 0.0:
            raise ValueError(
                f"translation_threshold must be > 0, got {translation_threshold}."
            )
        if rotation_threshold_deg < 0.0:
            raise ValueError(
                f"rotation_threshold_deg must be >= 0, got {rotation_threshold_deg}."
            )
        self._t_thresh = float(translation_threshold)
        self._r_thresh = float(np.radians(rotation_threshold_deg))
        self._last_pose: Optional[np.ndarray] = None
        self._count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_add_keyframe(self, transform: np.ndarray) -> bool:
        """Decide whether *transform* qualifies as a new keyframe.

        The first call always returns ``True``.  Subsequent calls return
        ``True`` when the translation or rotation distance from the most
        recently accepted keyframe meets or exceeds the configured thresholds.

        This method is **read-only** — it does not update the stored reference
        pose.  Call :meth:`mark_accepted` (or use :meth:`check_and_accept`)
        to update the state after deciding to accept a pose.

        Args:
            transform: 4×4 SE(3) ego-to-world transform.

        Returns:
            ``True`` if the pose should be accepted as a new keyframe.

        Raises:
            ValueError: If *transform* is not a 4×4 matrix.
        """
        T = np.asarray(transform, dtype=float)
        _validate_transform(T)

        if self._last_pose is None:
            return True
        if _translation_distance(self._last_pose, T) >= self._t_thresh:
            return True
        if self._r_thresh > 0.0 and _rotation_angle_between(self._last_pose, T) >= self._r_thresh:
            return True
        return False

    def mark_accepted(self, transform: np.ndarray) -> None:
        """Record *transform* as the most recently accepted keyframe pose.

        Increments the internal keyframe counter and updates the stored
        reference pose used by :meth:`should_add_keyframe`.

        Args:
            transform: 4×4 SE(3) ego-to-world transform.

        Raises:
            ValueError: If *transform* is not a 4×4 matrix.
        """
        T = np.asarray(transform, dtype=float)
        _validate_transform(T)
        self._last_pose = T.copy()
        self._count += 1

    def check_and_accept(self, transform: np.ndarray) -> bool:
        """Query and, if accepted, record *transform* as a new keyframe.

        Equivalent to calling :meth:`should_add_keyframe` and, when it
        returns ``True``, calling :meth:`mark_accepted`.

        Args:
            transform: 4×4 SE(3) ego-to-world transform.

        Returns:
            ``True`` if the pose was accepted as a new keyframe.

        Raises:
            ValueError: If *transform* is not a 4×4 matrix.
        """
        if self.should_add_keyframe(transform):
            self.mark_accepted(transform)
            return True
        return False

    @property
    def keyframe_count(self) -> int:
        """Total number of keyframes accepted (and marked) so far."""
        return self._count

    @property
    def last_pose(self) -> Optional[np.ndarray]:
        """The most recently accepted keyframe pose (4×4 copy), or ``None``."""
        return None if self._last_pose is None else self._last_pose.copy()


# ---------------------------------------------------------------------------
# Submap
# ---------------------------------------------------------------------------


@dataclass
class Submap:
    """A local map segment assembled from a contiguous set of keyframes.

    Attributes:
        submap_id: Unique integer identifier for this submap.
        keyframe_ids: Ordered list of integer keyframe IDs belonging to
            this submap.  When *overlap* > 0 in :class:`SubmapManager`,
            the first few IDs are shared with the preceding submap.
        origin_pose: 4×4 SE(3) world-frame pose of the first keyframe in
            this submap, used as the local coordinate origin.
        point_cloud: Accumulated point cloud (points stored in the
            *world* frame, consistent with
            :class:`~sensor_transposition.point_cloud_map.PointCloudMap`).
    """

    submap_id: int
    keyframe_ids: List[int] = field(default_factory=list)
    origin_pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    point_cloud: PointCloudMap = field(default_factory=PointCloudMap)

    @property
    def size(self) -> int:
        """Number of keyframe entries in this submap (including overlap copies)."""
        return len(self.keyframe_ids)


# ---------------------------------------------------------------------------
# SubmapManager
# ---------------------------------------------------------------------------


class SubmapManager:
    """Divide a keyframe stream into a sequence of local submaps.

    A new submap is created when the current one reaches
    *max_keyframes_per_submap* keyframes.  An optional *overlap* parameter
    causes the last *overlap* keyframes of the finishing submap to be
    replayed into the start of the new one, providing shared observations
    for inter-submap loop closure and alignment.

    Args:
        max_keyframes_per_submap: Maximum number of keyframes per submap
            before a new one is created.  Must be >= 1.
        overlap: Number of keyframes from the end of the current submap
            that are duplicated at the start of the next submap.  Must be
            >= 0 and < *max_keyframes_per_submap*.  Default ``0`` (no
            overlap).

    Raises:
        ValueError: If *max_keyframes_per_submap* < 1, *overlap* < 0, or
            *overlap* >= *max_keyframes_per_submap*.
    """

    def __init__(
        self,
        max_keyframes_per_submap: int = 50,
        overlap: int = 0,
    ) -> None:
        if max_keyframes_per_submap < 1:
            raise ValueError(
                f"max_keyframes_per_submap must be >= 1, "
                f"got {max_keyframes_per_submap}."
            )
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}.")
        if overlap >= max_keyframes_per_submap:
            raise ValueError(
                f"overlap ({overlap}) must be < max_keyframes_per_submap "
                f"({max_keyframes_per_submap})."
            )

        self._max_kf = max_keyframes_per_submap
        self._overlap = overlap

        self._submaps: List[Submap] = []
        self._current: Optional[Submap] = None
        self._next_id: int = 0

        # Circular buffer of the last *overlap* keyframe entries for carry-over.
        # Each entry is (keyframe_id, transform_4x4, points_Nx3, colors_or_None).
        self._overlap_buf: List[Tuple] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_keyframe(
        self,
        keyframe_id: int,
        transform: np.ndarray,
        points: np.ndarray,
        *,
        colors: Optional[np.ndarray] = None,
    ) -> Submap:
        """Add a keyframe scan to the current submap.

        If the current submap is at capacity a new submap is created first,
        optionally seeded with the overlap keyframes from the previous submap.

        Args:
            keyframe_id: Unique integer identifier for this keyframe.
            transform: 4×4 SE(3) ego-to-world transform at this keyframe.
            points: ``(N, 3)`` float array of LiDAR points in the sensor
                body frame (will be transformed to the world frame by
                :class:`~sensor_transposition.point_cloud_map.PointCloudMap`).
            colors: Optional ``(N, 3)`` per-point RGB colours.  See
                :meth:`~sensor_transposition.point_cloud_map.PointCloudMap.add_scan`
                for dtype rules.

        Returns:
            The :class:`Submap` to which this keyframe was added.

        Raises:
            ValueError: If *transform* is not 4×4 or *points* / *colors*
                fail :class:`PointCloudMap` validation.
        """
        T = np.asarray(transform, dtype=float)
        _validate_transform(T)

        # Start the first submap or a new one when the current is full.
        if self._current is None or self._current.size >= self._max_kf:
            self._start_new_submap(T)

        # Add this keyframe to the active submap.
        assert self._current is not None
        self._current.keyframe_ids.append(keyframe_id)
        self._current.point_cloud.add_scan(points, T, colors=colors)

        # Maintain the sliding overlap buffer (last *overlap* entries).
        if self._overlap > 0:
            self._overlap_buf.append((keyframe_id, T.copy(), points, colors))
            if len(self._overlap_buf) > self._overlap:
                self._overlap_buf.pop(0)

        return self._current

    def get_current_submap(self) -> Optional[Submap]:
        """Return the submap currently being populated, or ``None``."""
        return self._current

    def get_all_submaps(self) -> List[Submap]:
        """Return all submaps, including the one currently being built."""
        return list(self._submaps)

    def get_submap(self, submap_id: int) -> Submap:
        """Return the submap with the given *submap_id*.

        Raises:
            KeyError: If no submap with *submap_id* exists.
        """
        for sm in self._submaps:
            if sm.submap_id == submap_id:
                return sm
        raise KeyError(f"Submap {submap_id} not found.")

    @property
    def num_submaps(self) -> int:
        """Number of submaps created so far (including the current one)."""
        return len(self._submaps)

    @property
    def total_keyframes(self) -> int:
        """Total number of keyframe entries across all submaps.

        Overlap copies are counted once per submap in which they appear.
        """
        return sum(sm.size for sm in self._submaps)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_new_submap(self, incoming_transform: np.ndarray) -> None:
        """Close the current submap and open a new one.

        If *overlap* > 0, the buffered keyframes from the previous submap
        are replayed into the new submap before the caller's frame is added,
        and the new submap's *origin_pose* is set to the pose of the first
        replayed frame (or to *incoming_transform* when the buffer is empty).
        """
        # Choose origin: first overlap frame if available, else the incoming pose.
        if self._overlap_buf:
            origin = self._overlap_buf[0][1]
        else:
            origin = incoming_transform

        submap = Submap(
            submap_id=self._next_id,
            origin_pose=origin.copy(),
        )
        self._next_id += 1
        self._submaps.append(submap)
        self._current = submap

        # Replay the overlap keyframes into the new submap.
        for (kf_id, kf_T, kf_pts, kf_clr) in self._overlap_buf:
            submap.keyframe_ids.append(kf_id)
            submap.point_cloud.add_scan(kf_pts, kf_T, colors=kf_clr)
