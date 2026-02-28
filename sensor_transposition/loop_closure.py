"""
loop_closure.py

Place recognition and loop-closure detection using the Scan Context descriptor.

Scan Context [Kim & Kim, IROS 2018] encodes a 360° LiDAR scan into a compact
2-D polar grid descriptor suitable for efficient place recognition:

* The horizontal plane around the sensor is divided into ``num_rings`` concentric
  radial bins and ``num_sectors`` angular bins, forming an
  ``(num_rings × num_sectors)`` descriptor matrix.
* Each cell stores the **maximum z-height** of points that fall inside it.
  Cells with no points are set to zero.
* **Rotation invariance** is achieved at query time by trying all column
  shifts (yaw rotations) and returning the shift that minimises the normalised
  cosine distance.
* A compact **ring key** (column-wise mean of each row) provides an
  *O*(num_rings) pre-filter before the full *O*(num_rings × num_sectors)
  distance is computed, which substantially reduces the number of full
  descriptor comparisons required.

Together, :func:`compute_scan_context` and :class:`ScanContextDatabase` form
the *appearance-based* front-end of a loop-closure pipeline.  After a
candidate loop is found, the caller should pass the two point clouds to
:func:`sensor_transposition.lidar.scan_matching.icp_align` for a
geometry-based verification step and pose-graph edge estimation.

All functions operate on plain NumPy arrays and depend only on ``numpy`` and
``scipy`` (already required by ``sensor_transposition``), so no additional
dependencies are needed.

Typical use-case
----------------
::

    from sensor_transposition.loop_closure import (
        compute_scan_context,
        ScanContextDatabase,
    )

    db = ScanContextDatabase(num_rings=20, num_sectors=60, max_range=80.0)

    for frame_id, point_cloud in enumerate(lidar_frames):
        descriptor = compute_scan_context(
            point_cloud, num_rings=20, num_sectors=60, max_range=80.0
        )
        candidates = db.query(descriptor, top_k=1)
        db.add(descriptor, frame_id=frame_id)

        if candidates and candidates[0].distance < 0.15:
            print(f"Loop closure: frame {frame_id} ↔ frame "
                  f"{candidates[0].match_frame_id} "
                  f"(distance={candidates[0].distance:.3f}, "
                  f"yaw_shift={candidates[0].yaw_shift_sectors} sectors)")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ScanContextDescriptor:
    """A Scan Context descriptor computed from one LiDAR scan.

    Attributes:
        matrix: ``(num_rings, num_sectors)`` float array.  Each cell holds the
            maximum z-height (metres) of points that project into that
            (ring, sector) bin.  Empty cells are zero.
        ring_key: ``(num_rings,)`` float array — the mean of each row of
            *matrix*.  Used as a fast rotation-invariant pre-filter.
        num_rings: Number of radial bins.
        num_sectors: Number of angular (azimuth) bins.
        max_range: Maximum radial range (metres) used when building the
            descriptor.  Points beyond this radius are ignored.
    """

    matrix: np.ndarray
    ring_key: np.ndarray
    num_rings: int
    num_sectors: int
    max_range: float


@dataclass
class LoopClosureCandidate:
    """A candidate loop closure returned by :class:`ScanContextDatabase`.

    Attributes:
        match_frame_id: Frame identifier of the matching keyframe in the
            database (as supplied to :meth:`ScanContextDatabase.add`).
        distance: Normalised cosine distance in ``[0, 1]`` between the query
            and the matched descriptor.  Lower means more similar; values
            below ~0.15 typically indicate a genuine revisit.
        yaw_shift_sectors: Number of sector columns the matched descriptor
            was shifted to minimise the distance.  Multiply by
            ``(2π / num_sectors)`` to obtain the estimated yaw offset in
            radians.
        database_index: Index of the match within the database's internal
            descriptor list.
    """

    match_frame_id: int
    distance: float
    yaw_shift_sectors: int
    database_index: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_scan_context(
    points: np.ndarray,
    *,
    num_rings: int = 20,
    num_sectors: int = 60,
    max_range: float = 80.0,
    min_z: float | None = None,
    max_z: float | None = None,
) -> ScanContextDescriptor:
    """Compute a Scan Context descriptor from a 3-D LiDAR point cloud.

    Args:
        points: ``(N, 3)`` float array of XYZ coordinates in the sensor
            frame.  Only the ``(x, y)`` planar position and ``z`` height are
            used; intensity and other columns are ignored if present.
        num_rings: Number of concentric radial bins.  Must be ≥ 1.
            Default ``20``.
        num_sectors: Number of angular bins covering 0–2π.  Must be ≥ 1.
            Default ``60``.
        max_range: Maximum radial distance (metres) to consider.  Points
            beyond this radius are discarded.  Must be > 0.  Default ``80.0``.
        min_z: Optional lower z-clipping plane (metres).  Points below this
            height are discarded before building the descriptor.  ``None``
            means no lower clip.
        max_z: Optional upper z-clipping plane (metres).  Points above this
            height are discarded before building the descriptor.  ``None``
            means no upper clip.

    Returns:
        :class:`ScanContextDescriptor` containing the ``(num_rings,
        num_sectors)`` matrix and the ``(num_rings,)`` ring key.

    Raises:
        ValueError: If *points* is not at least ``(N, 3)``, *num_rings* < 1,
            *num_sectors* < 1, or *max_range* ≤ 0.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(
            f"points must be an (N, ≥3) array, got shape {pts.shape}."
        )
    if num_rings < 1:
        raise ValueError(f"num_rings must be >= 1, got {num_rings}.")
    if num_sectors < 1:
        raise ValueError(f"num_sectors must be >= 1, got {num_sectors}.")
    if max_range <= 0.0:
        raise ValueError(f"max_range must be > 0, got {max_range}.")

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    # Optional height clipping.
    mask = np.ones(len(pts), dtype=bool)
    if min_z is not None:
        mask &= z >= min_z
    if max_z is not None:
        mask &= z <= max_z
    x, y, z = x[mask], y[mask], z[mask]

    # Compute planar range and reject points beyond max_range.
    r = np.sqrt(x ** 2 + y ** 2)
    valid = (r > 0.0) & (r <= max_range)
    x, y, z, r = x[valid], y[valid], z[valid], r[valid]

    descriptor = np.zeros((num_rings, num_sectors), dtype=float)

    if r.size > 0:
        # Ring index: equally-spaced in [0, max_range].
        ring_idx = np.floor(r / max_range * num_rings).astype(int)
        ring_idx = np.clip(ring_idx, 0, num_rings - 1)

        # Sector index: equally-spaced in [0, 2π).
        angle = np.arctan2(y, x) % (2.0 * np.pi)  # map to [0, 2π)
        sector_idx = np.floor(angle / (2.0 * np.pi) * num_sectors).astype(int)
        sector_idx = np.clip(sector_idx, 0, num_sectors - 1)

        # For each (ring, sector) bin keep the maximum z height.
        np.maximum.at(descriptor, (ring_idx, sector_idx), z)

    ring_key = descriptor.mean(axis=1)  # (num_rings,)

    return ScanContextDescriptor(
        matrix=descriptor,
        ring_key=ring_key,
        num_rings=num_rings,
        num_sectors=num_sectors,
        max_range=max_range,
    )


def scan_context_distance(
    desc_a: ScanContextDescriptor,
    desc_b: ScanContextDescriptor,
) -> tuple[float, int]:
    """Compute the rotation-invariant Scan Context distance between two descriptors.

    The distance is the **minimum normalised cosine distance** over all
    possible column shifts of *desc_b*:

    .. code-block:: text

        d(A, B) = min_{shift} mean_{rings} cos_dist(A[ring], B_shifted[ring])

    where ``cos_dist(u, v) = 1 − (u·v) / (||u|| ||v||)`` (clipped to
    ``[0, 1]``).

    Args:
        desc_a: First :class:`ScanContextDescriptor`.
        desc_b: Second :class:`ScanContextDescriptor`.

    Returns:
        Tuple ``(distance, yaw_shift_sectors)`` where *distance* is in
        ``[0, 1]`` (lower = more similar) and *yaw_shift_sectors* is the
        column shift applied to *desc_b* that achieves the minimum.

    Raises:
        ValueError: If the two descriptors have incompatible shapes.
    """
    _check_compatible(desc_a, desc_b)

    A = desc_a.matrix   # (R, S)
    B = desc_b.matrix   # (R, S)

    best_dist = float("inf")
    best_shift = 0

    # Use FFT-based circular correlation for efficient shift search when
    # num_sectors is large; fall back to direct iteration otherwise.
    # For simplicity (and to avoid fftpack quirks with small sizes), we always
    # use the direct approach here – O(num_sectors * num_rings) is cheap for
    # typical values (20 × 60 = 1200 operations per shift × 60 shifts = 72k).
    num_sectors = desc_a.num_sectors
    for shift in range(num_sectors):
        B_shifted = np.roll(B, shift, axis=1)
        dist = _column_cosine_distance(A, B_shifted)
        if dist < best_dist:
            best_dist = dist
            best_shift = shift

    return best_dist, best_shift


class ScanContextDatabase:
    """An incremental database of Scan Context descriptors for loop closure.

    Descriptors are added one by one via :meth:`add`; the database is then
    queried via :meth:`query` which returns the top-*k* nearest neighbours
    (by Scan Context distance) from a configurable *exclusion window* of
    recent entries.

    A two-stage search is used for efficiency:

    1. **Ring key pre-filter**: Compute the L1 distance between the query's
       ring key and all stored ring keys and retain the *candidate_pool_size*
       nearest candidates.
    2. **Full Scan Context distance**: Compute the rotation-invariant distance
       for each candidate, returning the *top_k* closest matches.

    Args:
        num_rings: Radial bins expected for all descriptors (default ``20``).
        num_sectors: Angular bins expected for all descriptors (default ``60``).
        max_range: Maximum range used when building descriptors (default
            ``80.0`` metres).
        exclusion_window: Minimum number of recent frames to skip when
            searching for loop closures, to avoid matching against
            near-consecutive frames.  Default ``50``.
        candidate_pool_size: Number of ring-key nearest neighbours to
            evaluate with the full distance before selecting the *top_k*
            results.  Default ``25``.
    """

    def __init__(
        self,
        *,
        num_rings: int = 20,
        num_sectors: int = 60,
        max_range: float = 80.0,
        exclusion_window: int = 50,
        candidate_pool_size: int = 25,
    ) -> None:
        if num_rings < 1:
            raise ValueError(f"num_rings must be >= 1, got {num_rings}.")
        if num_sectors < 1:
            raise ValueError(f"num_sectors must be >= 1, got {num_sectors}.")
        if max_range <= 0.0:
            raise ValueError(f"max_range must be > 0, got {max_range}.")
        if exclusion_window < 0:
            raise ValueError(
                f"exclusion_window must be >= 0, got {exclusion_window}."
            )
        if candidate_pool_size < 1:
            raise ValueError(
                f"candidate_pool_size must be >= 1, got {candidate_pool_size}."
            )

        self.num_rings = num_rings
        self.num_sectors = num_sectors
        self.max_range = max_range
        self.exclusion_window = exclusion_window
        self.candidate_pool_size = candidate_pool_size

        self._descriptors: List[ScanContextDescriptor] = []
        self._frame_ids: List[int] = []
        # Cached ring key matrix: shape (n_stored, num_rings) for fast L1 search.
        self._ring_keys: np.ndarray = np.empty((0, num_rings), dtype=float)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def add(
        self,
        descriptor: ScanContextDescriptor,
        *,
        frame_id: int | None = None,
    ) -> int:
        """Add a descriptor to the database.

        Args:
            descriptor: A :class:`ScanContextDescriptor` computed with the
                same ``num_rings``, ``num_sectors``, and ``max_range`` as the
                database.
            frame_id: Optional integer identifier for the keyframe.  If
                ``None``, the current length of the database (before insertion)
                is used as the frame ID.

        Returns:
            The database index assigned to the newly added descriptor.

        Raises:
            ValueError: If *descriptor* has incompatible shape parameters.
        """
        self._check_descriptor_params(descriptor)

        idx = len(self._descriptors)
        if frame_id is None:
            frame_id = idx

        self._descriptors.append(descriptor)
        self._frame_ids.append(frame_id)

        # Append ring key to the cached matrix.
        rk = descriptor.ring_key.reshape(1, -1)
        self._ring_keys = np.vstack([self._ring_keys, rk])

        return idx

    def query(
        self,
        descriptor: ScanContextDescriptor,
        *,
        top_k: int = 1,
    ) -> List[LoopClosureCandidate]:
        """Find the top-*k* loop closure candidates for a query descriptor.

        Args:
            descriptor: The query :class:`ScanContextDescriptor`.
            top_k: Number of best matches to return.  Default ``1``.

        Returns:
            List of up to *top_k* :class:`LoopClosureCandidate` objects sorted
            by ascending distance.  An empty list is returned when the
            database contains no eligible entries (i.e. all entries are within
            the *exclusion_window*).

        Raises:
            ValueError: If *descriptor* has incompatible shape parameters or
                *top_k* < 1.
        """
        self._check_descriptor_params(descriptor)
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")

        n = len(self._descriptors)
        eligible_end = n - self.exclusion_window
        if eligible_end <= 0:
            return []

        eligible_indices = np.arange(eligible_end)

        # Stage 1: ring key L1 pre-filter.
        rk_query = descriptor.ring_key  # (num_rings,)
        rk_matrix = self._ring_keys[:eligible_end]  # (eligible, num_rings)
        l1_dists = np.abs(rk_matrix - rk_query).sum(axis=1)  # (eligible,)

        pool_size = min(self.candidate_pool_size, eligible_end)
        pool_indices = np.argpartition(l1_dists, pool_size - 1)[:pool_size]
        pool_indices = eligible_indices[pool_indices]

        # Stage 2: full Scan Context distance with rotation search.
        candidates: List[LoopClosureCandidate] = []
        for db_idx in pool_indices:
            dist, shift = scan_context_distance(descriptor, self._descriptors[db_idx])
            candidates.append(
                LoopClosureCandidate(
                    match_frame_id=self._frame_ids[db_idx],
                    distance=dist,
                    yaw_shift_sectors=shift,
                    database_index=int(db_idx),
                )
            )

        candidates.sort(key=lambda c: c.distance)
        return candidates[:top_k]

    def __len__(self) -> int:
        """Return the number of descriptors currently stored."""
        return len(self._descriptors)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_descriptor_params(self, descriptor: ScanContextDescriptor) -> None:
        """Raise ``ValueError`` if descriptor parameters are incompatible."""
        if descriptor.num_rings != self.num_rings:
            raise ValueError(
                f"Descriptor num_rings={descriptor.num_rings} does not match "
                f"database num_rings={self.num_rings}."
            )
        if descriptor.num_sectors != self.num_sectors:
            raise ValueError(
                f"Descriptor num_sectors={descriptor.num_sectors} does not match "
                f"database num_sectors={self.num_sectors}."
            )
        if descriptor.max_range != self.max_range:
            raise ValueError(
                f"Descriptor max_range={descriptor.max_range} does not match "
                f"database max_range={self.max_range}."
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _column_cosine_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Mean per-row cosine distance between two ``(R, S)`` matrices.

    For each row ``r``: ``cos_dist = 1 − (A[r] · B[r]) / (||A[r]|| ||B[r]||)``.
    Rows where both vectors are zero contribute 0 (identical empty cells).
    The mean over all rows is returned.

    Args:
        A: ``(R, S)`` matrix.
        B: ``(R, S)`` matrix (same shape as *A*).

    Returns:
        Mean cosine distance in ``[0, 1]``.
    """
    norm_a = np.linalg.norm(A, axis=1)  # (R,)
    norm_b = np.linalg.norm(B, axis=1)  # (R,)

    dot = np.sum(A * B, axis=1)  # (R,)
    denom = norm_a * norm_b

    # For rows where both norms are zero the vectors are identical (all zeros),
    # so cosine distance is 0.  Rows where only one norm is zero have cosine
    # similarity 0 (since denom = 0), giving cos_dist = 1 — maximum distance.
    both_zero = (norm_a < 1e-12) & (norm_b < 1e-12)

    # Compute cosine similarity safely: avoid division by zero by substituting
    # 1.0 for near-zero denominators (the result is overridden by np.where).
    safe_denom = np.where(denom > 1e-12, denom, 1.0)
    cos_sim = np.where(denom > 1e-12, dot / safe_denom, 0.0)
    cos_dist = 1.0 - np.clip(cos_sim, -1.0, 1.0)
    # Rows where both vectors are zero are identical → distance 0.
    # Rows where only one vector is zero → maximally different → distance 1.
    cos_dist = np.where(both_zero, 0.0, cos_dist)

    return float(cos_dist.mean())


def _check_compatible(
    a: ScanContextDescriptor,
    b: ScanContextDescriptor,
) -> None:
    """Raise ``ValueError`` if two descriptors have incompatible shapes."""
    if a.num_rings != b.num_rings or a.num_sectors != b.num_sectors:
        raise ValueError(
            f"Descriptors have incompatible shapes: "
            f"({a.num_rings}, {a.num_sectors}) vs ({b.num_rings}, {b.num_sectors})."
        )
