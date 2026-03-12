"""
loop_closure.py

Place recognition and loop-closure detection using the Scan Context and M2DP
descriptors.

**Scan Context** [Kim & Kim, IROS 2018] encodes a 360° LiDAR scan into a
compact 2-D polar grid descriptor suitable for efficient place recognition:

* The horizontal plane around the sensor is divided into ``num_rings``
  concentric radial bins and ``num_sectors`` angular bins, forming an
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

**M2DP** (Multi-view 2D Projection) [He et al., IROS 2016] builds a compact
global descriptor by projecting the point cloud onto multiple oriented 2-D
planes and summarising the point-density distribution on each plane:

* For each of ``num_elevation × num_azimuth`` oriented planes a polar-bin
  density histogram is computed over the projected 2-D point positions.
* The resulting ``(num_elevation × num_azimuth, num_rings × num_sectors)``
  signature matrix is decomposed via SVD; the first left and right singular
  vectors are concatenated to form the final descriptor vector.
* M2DP is viewpoint-insensitive (works well even without a strong ground
  plane) and complements Scan Context, which is optimised for ground-vehicle
  LiDAR.

Together, :func:`compute_scan_context`, :func:`compute_m2dp`, and
:class:`ScanContextDatabase` form the *appearance-based* front-end of a
loop-closure pipeline.  After a candidate loop is found, the caller should
pass the two point clouds to
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
        compute_m2dp,
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

    # M2DP can be used stand-alone for viewpoint-insensitive matching:
    desc_a = compute_m2dp(point_cloud_a)
    desc_b = compute_m2dp(point_cloud_b)
    from sensor_transposition.loop_closure import m2dp_distance
    print(f"M2DP distance: {m2dp_distance(desc_a, desc_b):.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Threshold for detecting when a unit vector is near-vertical (|n_z| > this
# value), used when choosing a reference vector for orthonormal basis
# construction in compute_m2dp.
_VERTICAL_THRESHOLD: float = 0.99

# Below this L2 norm a descriptor vector is considered a zero vector.
_ZERO_NORM_THRESHOLD: float = 1e-12


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


@dataclass
class M2dpDescriptor:
    """An M2DP descriptor computed from one LiDAR scan.

    M2DP (Multi-view 2D Projection) [He et al., IROS 2016] builds a compact
    global descriptor by projecting the point cloud onto
    ``num_elevation × num_azimuth`` oriented 2-D planes, computing a polar
    point-density histogram on each plane, and reducing the resulting
    signature matrix via SVD.

    Attributes:
        vector: 1-D float array of length
            ``num_elevation * num_azimuth + num_rings * num_sectors``.
            Formed by concatenating the first left and right singular vectors
            of the signature matrix.
        num_azimuth: Number of azimuthal projection directions.
        num_elevation: Number of elevation projection directions.
        num_rings: Number of radial bins in each 2-D projection.
        num_sectors: Number of angular bins in each 2-D projection.
    """

    vector: np.ndarray
    num_azimuth: int
    num_elevation: int
    num_rings: int
    num_sectors: int


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


def compute_m2dp(
    points: np.ndarray,
    *,
    num_azimuth: int = 4,
    num_elevation: int = 16,
    num_rings: int = 4,
    num_sectors: int = 16,
) -> M2dpDescriptor:
    """Compute an M2DP descriptor from a 3-D LiDAR point cloud.

    The algorithm [He et al., IROS 2016]:

    1. Centre the point cloud at its centroid.
    2. For each of ``num_elevation × num_azimuth`` oriented planes (defined by
       sweeping elevation θ over (−π/2, π/2) and azimuth φ over [0, 2π)):

       a. Project the 3-D points orthogonally onto the plane using two
          orthonormal basis vectors **e₁**, **e₂** constructed from the plane
          normal.
       b. Convert the 2-D projected coordinates to polar form and bin into a
          ``(num_rings, num_sectors)`` density histogram (point counts).
       c. Store the flattened histogram as one row of the signature matrix
          **A** of shape ``(L, P)`` where
          ``L = num_elevation × num_azimuth`` and
          ``P = num_rings × num_sectors``.

    3. Compute the thin SVD of **A**.  The descriptor vector is formed by
       concatenating the first left singular vector **u₁** (length *L*) and
       the first right singular vector **v₁** (length *P*), giving a vector
       of length ``L + P``.

    Args:
        points: ``(N, 3)`` (or wider) float array of XYZ coordinates.
            Only the first three columns are used.
        num_azimuth: Number of azimuthal projection directions.  Must be ≥ 1.
            Default ``4``.
        num_elevation: Number of elevation projection directions.  Must be ≥ 1.
            Default ``16``.
        num_rings: Number of radial bins in the 2-D polar histogram.  Must be
            ≥ 1.  Default ``4``.
        num_sectors: Number of angular bins in the 2-D polar histogram.  Must
            be ≥ 1.  Default ``16``.

    Returns:
        :class:`M2dpDescriptor` whose ``vector`` has length
        ``num_elevation * num_azimuth + num_rings * num_sectors``.

    Raises:
        ValueError: If *points* is not at least ``(N, 3)``, or any parameter
            is less than 1.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(
            f"points must be an (N, ≥3) array, got shape {pts.shape}."
        )
    if num_azimuth < 1:
        raise ValueError(f"num_azimuth must be >= 1, got {num_azimuth}.")
    if num_elevation < 1:
        raise ValueError(f"num_elevation must be >= 1, got {num_elevation}.")
    if num_rings < 1:
        raise ValueError(f"num_rings must be >= 1, got {num_rings}.")
    if num_sectors < 1:
        raise ValueError(f"num_sectors must be >= 1, got {num_sectors}.")

    xyz = pts[:, :3]
    # Centre at the centroid.
    xyz = xyz - xyz.mean(axis=0)

    L = num_elevation * num_azimuth   # number of projection planes
    P = num_rings * num_sectors        # bins per projection

    signature = np.zeros((L, P), dtype=float)

    plane_idx = 0
    for ei in range(num_elevation):
        # Elevation angle θ sweeps −π/2 … π/2 (exclusive) via midpoint rule.
        theta = np.pi * (ei + 0.5) / num_elevation - np.pi / 2.0
        for ai in range(num_azimuth):
            phi = 2.0 * np.pi * ai / num_azimuth  # azimuth 0 … 2π

            # Plane normal vector.
            cos_t = np.cos(theta)
            n = np.array([
                cos_t * np.cos(phi),
                cos_t * np.sin(phi),
                np.sin(theta),
            ])

            # Build an orthonormal basis {e1, e2} for the plane.
            # Choose a reference vector that is not parallel to n.
            ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < _VERTICAL_THRESHOLD else np.array([1.0, 0.0, 0.0])
            e1 = np.cross(n, ref)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(n, e1)
            e2 /= np.linalg.norm(e2)

            # Orthogonal projection of every point onto the plane.
            u = xyz @ e1  # (N,)
            v = xyz @ e2  # (N,)

            # Convert to polar and bin into (num_rings × num_sectors) histogram.
            r = np.sqrt(u ** 2 + v ** 2)
            max_r_val = r.max()
            max_r = max_r_val if max_r_val > _ZERO_NORM_THRESHOLD else 1.0

            ring_idx = np.floor(r / max_r * num_rings).astype(int)
            ring_idx = np.clip(ring_idx, 0, num_rings - 1)

            angle = np.arctan2(v, u) % (2.0 * np.pi)
            sector_idx = np.floor(angle / (2.0 * np.pi) * num_sectors).astype(int)
            sector_idx = np.clip(sector_idx, 0, num_sectors - 1)

            density = np.zeros((num_rings, num_sectors), dtype=float)
            np.add.at(density, (ring_idx, sector_idx), 1)
            signature[plane_idx] = density.ravel()
            plane_idx += 1

    # SVD decomposition of the signature matrix.
    if signature.any():
        U, _s, Vt = np.linalg.svd(signature, full_matrices=False)
        u1 = U[:, 0]    # first left singular vector  (L,)
        v1 = Vt[0, :]   # first right singular vector (P,)
    else:
        u1 = np.zeros(L, dtype=float)
        v1 = np.zeros(P, dtype=float)

    return M2dpDescriptor(
        vector=np.concatenate([u1, v1]),
        num_azimuth=num_azimuth,
        num_elevation=num_elevation,
        num_rings=num_rings,
        num_sectors=num_sectors,
    )


def m2dp_distance(
    desc_a: M2dpDescriptor,
    desc_b: M2dpDescriptor,
) -> float:
    """Compute the cosine distance between two M2DP descriptors.

    Args:
        desc_a: First :class:`M2dpDescriptor`.
        desc_b: Second :class:`M2dpDescriptor`.

    Returns:
        Cosine distance in ``[0, 1]`` — lower means more similar.

    Raises:
        ValueError: If the two descriptors have incompatible parameters.
    """
    if (
        desc_a.num_azimuth != desc_b.num_azimuth
        or desc_a.num_elevation != desc_b.num_elevation
        or desc_a.num_rings != desc_b.num_rings
        or desc_a.num_sectors != desc_b.num_sectors
    ):
        raise ValueError(
            "M2DP descriptors have incompatible parameters: "
            f"({desc_a.num_azimuth}, {desc_a.num_elevation}, "
            f"{desc_a.num_rings}, {desc_a.num_sectors}) vs "
            f"({desc_b.num_azimuth}, {desc_b.num_elevation}, "
            f"{desc_b.num_rings}, {desc_b.num_sectors})."
        )

    a = desc_a.vector
    b = desc_b.vector
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < _ZERO_NORM_THRESHOLD and norm_b < _ZERO_NORM_THRESHOLD:
        return 0.0  # both zero vectors → identical (both from empty clouds)
    if norm_a < _ZERO_NORM_THRESHOLD or norm_b < _ZERO_NORM_THRESHOLD:
        return 1.0  # one is zero → maximally different

    cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
    return float(1.0 - np.clip(cos_sim, -1.0, 1.0))


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

    def compute_descriptor(
        self,
        points: np.ndarray,
        *,
        min_z: float | None = None,
        max_z: float | None = None,
    ) -> ScanContextDescriptor:
        """Compute a Scan Context descriptor using the database's own parameters.

        This is a convenience wrapper around :func:`compute_scan_context` that
        uses the ``num_rings``, ``num_sectors``, and ``max_range`` values that
        were set when the database was constructed, so the caller never has to
        repeat them.  Using this method guarantees that descriptors are always
        compatible with the database.

        Args:
            points: ``(N, 3)`` float array of XYZ LiDAR point coordinates.
            min_z: Optional lower z-clipping plane (metres).
            max_z: Optional upper z-clipping plane (metres).

        Returns:
            :class:`ScanContextDescriptor` compatible with this database.

        Example::

            db = ScanContextDatabase(num_rings=20, num_sectors=60, max_range=80.0)
            for frame_id, cloud in enumerate(lidar_frames):
                desc = db.compute_descriptor(cloud)  # parameters always match
                candidates = db.query(desc, top_k=1)
                db.add(desc, frame_id=frame_id)
        """
        return compute_scan_context(
            points,
            num_rings=self.num_rings,
            num_sectors=self.num_sectors,
            max_range=self.max_range,
            min_z=min_z,
            max_z=max_z,
        )

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
    both_zero = (norm_a < _ZERO_NORM_THRESHOLD) & (norm_b < _ZERO_NORM_THRESHOLD)

    # Compute cosine similarity safely: avoid division by zero by substituting
    # 1.0 for near-zero denominators (the result is overridden by np.where).
    safe_denom = np.where(denom > _ZERO_NORM_THRESHOLD, denom, 1.0)
    cos_sim = np.where(denom > _ZERO_NORM_THRESHOLD, dot / safe_denom, 0.0)
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


# ===========================================================================
# Visual Loop Closure
# ===========================================================================
# The functions and class below extend the loop-closure module to support
# *camera-based* place recognition alongside the existing LiDAR descriptors.
# All computations are pure NumPy; no additional dependencies are required.


@dataclass
class ImageDescriptor:
    """A compact HOG-like image descriptor for visual loop closure.

    Attributes:
        vector: 1-D float array of length ``grid_rows × grid_cols × bins``.
            The descriptor is L2-normalised so that cosine distance and
            L2 distance are equivalent up to a constant scale.
        grid_rows: Number of cell rows used when building the descriptor.
        grid_cols: Number of cell columns used when building the descriptor.
        bins: Number of gradient-orientation histogram bins per cell.
    """

    vector: np.ndarray
    grid_rows: int
    grid_cols: int
    bins: int


def compute_image_descriptor(
    image: np.ndarray,
    *,
    grid: Tuple[int, int] = (4, 4),
    bins: int = 8,
) -> ImageDescriptor:
    """Compute a compact HOG-like descriptor from a grayscale image.

    The image is divided into a ``grid[0] × grid[1]`` regular grid of cells.
    Within each cell a histogram of gradient orientations (unsigned, 0–π) is
    computed using *bins* equally-spaced bins.  Bin magnitudes are
    gradient-magnitude weighted.  All cell histograms are concatenated and
    the resulting vector is L2-normalised.

    The descriptor length is ``grid[0] × grid[1] × bins``.

    Args:
        image: Grayscale image as a 2-D NumPy array (H × W).  Floating-point
            or integer pixel values are both accepted.
        grid: ``(num_rows, num_cols)`` number of cells to divide the image
            into.  Both values must be ≥ 1.  Default ``(4, 4)``.
        bins: Number of orientation histogram bins per cell.  Must be ≥ 1.
            Default ``8``.

    Returns:
        :class:`ImageDescriptor` with a normalised vector of length
        ``grid[0] × grid[1] × bins``.

    Raises:
        ValueError: If *image* is not 2-D, *grid* values are < 1, or
            *bins* < 1.

    Example::

        from sensor_transposition.loop_closure import compute_image_descriptor

        desc_a = compute_image_descriptor(gray_frame_a)
        desc_b = compute_image_descriptor(gray_frame_b)
        dist = image_descriptor_distance(desc_a, desc_b)
        print(f"Visual similarity distance: {dist:.4f}")
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError(
            f"image must be a 2-D (grayscale) array, got shape {img.shape}."
        )
    grid_rows, grid_cols = grid
    if grid_rows < 1 or grid_cols < 1:
        raise ValueError(
            f"grid values must be >= 1, got ({grid_rows}, {grid_cols})."
        )
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}.")

    H, W = img.shape

    # Compute image gradients using simple central differences.
    # Pad image to handle borders.
    padded = np.pad(img, 1, mode="edge")
    gx = padded[1:-1, 2:] - padded[1:-1, :-2]   # horizontal gradient
    gy = padded[2:, 1:-1] - padded[:-2, 1:-1]   # vertical gradient

    magnitude = np.sqrt(gx * gx + gy * gy)
    # Unsigned orientations in [0, π).
    orientation = np.arctan2(np.abs(gy), gx) % np.pi

    # Divide into grid cells and compute histograms.
    descriptor_parts: list[np.ndarray] = []
    for ri in range(grid_rows):
        r_start = int(round(ri * H / grid_rows))
        r_end = int(round((ri + 1) * H / grid_rows))
        for ci in range(grid_cols):
            c_start = int(round(ci * W / grid_cols))
            c_end = int(round((ci + 1) * W / grid_cols))

            cell_mag = magnitude[r_start:r_end, c_start:c_end].ravel()
            cell_ori = orientation[r_start:r_end, c_start:c_end].ravel()

            # Weighted histogram of orientations (unsigned).
            hist, _ = np.histogram(
                cell_ori,
                bins=bins,
                range=(0.0, np.pi),
                weights=cell_mag,
            )
            descriptor_parts.append(hist.astype(float))

    vector = np.concatenate(descriptor_parts)

    # L2-normalise.
    norm = np.linalg.norm(vector)
    if norm > _ZERO_NORM_THRESHOLD:
        vector = vector / norm

    return ImageDescriptor(
        vector=vector,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        bins=bins,
    )


def image_descriptor_distance(
    desc_a: ImageDescriptor,
    desc_b: ImageDescriptor,
) -> float:
    """Compute the cosine distance between two :class:`ImageDescriptor` objects.

    Args:
        desc_a: First image descriptor.
        desc_b: Second image descriptor.

    Returns:
        Cosine distance in ``[0, 1]`` — lower means more similar.

    Raises:
        ValueError: If the descriptors have incompatible parameters.
    """
    if (
        desc_a.grid_rows != desc_b.grid_rows
        or desc_a.grid_cols != desc_b.grid_cols
        or desc_a.bins != desc_b.bins
    ):
        raise ValueError(
            "ImageDescriptors have incompatible parameters: "
            f"grid=({desc_a.grid_rows}, {desc_a.grid_cols}), bins={desc_a.bins} vs "
            f"grid=({desc_b.grid_rows}, {desc_b.grid_cols}), bins={desc_b.bins}."
        )
    a = desc_a.vector
    b = desc_b.vector
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < _ZERO_NORM_THRESHOLD and norm_b < _ZERO_NORM_THRESHOLD:
        return 0.0
    if norm_a < _ZERO_NORM_THRESHOLD or norm_b < _ZERO_NORM_THRESHOLD:
        return 1.0
    cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
    return float(1.0 - np.clip(cos_sim, -1.0, 1.0))


class ImageLoopClosureDatabase:
    """An incremental database of image descriptors for visual loop closure.

    Mirrors the :class:`ScanContextDatabase` API (``add``, ``query``,
    ``compute_descriptor``) but operates on :class:`ImageDescriptor` objects
    computed from grayscale camera images.

    Descriptors are compared using cosine distance (via
    :func:`image_descriptor_distance`).  An exclusion window prevents matches
    against recently added frames.

    Args:
        grid: ``(num_rows, num_cols)`` grid used when computing descriptors
            (default ``(4, 4)``).
        bins: Number of orientation histogram bins per cell (default ``8``).
        exclusion_window: Minimum number of recent frames to skip when
            searching for loop closures.  Default ``20``.
    """

    def __init__(
        self,
        *,
        grid: Tuple[int, int] = (4, 4),
        bins: int = 8,
        exclusion_window: int = 20,
    ) -> None:
        grid_rows, grid_cols = grid
        if grid_rows < 1 or grid_cols < 1:
            raise ValueError(
                f"grid values must be >= 1, got ({grid_rows}, {grid_cols})."
            )
        if bins < 1:
            raise ValueError(f"bins must be >= 1, got {bins}.")
        if exclusion_window < 0:
            raise ValueError(
                f"exclusion_window must be >= 0, got {exclusion_window}."
            )

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.bins = bins
        self.exclusion_window = exclusion_window

        self._descriptors: List[ImageDescriptor] = []
        self._frame_ids: List[int] = []
        # Cached descriptor matrix for fast L2 pre-screening: (n, D).
        self._vectors: np.ndarray = np.empty(
            (0, grid_rows * grid_cols * bins), dtype=float
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def compute_descriptor(self, image: np.ndarray) -> ImageDescriptor:
        """Compute an :class:`ImageDescriptor` using this database's parameters.

        Args:
            image: Grayscale 2-D image array.

        Returns:
            :class:`ImageDescriptor` compatible with this database.
        """
        return compute_image_descriptor(
            image,
            grid=(self.grid_rows, self.grid_cols),
            bins=self.bins,
        )

    def add(
        self,
        descriptor: ImageDescriptor,
        *,
        frame_id: int | None = None,
    ) -> int:
        """Add a descriptor to the database.

        Args:
            descriptor: An :class:`ImageDescriptor` computed with compatible
                ``grid`` and ``bins`` parameters.
            frame_id: Optional integer identifier for the keyframe.  If
                ``None``, the current database length is used.

        Returns:
            The database index assigned to the newly added descriptor.

        Raises:
            ValueError: If *descriptor* has incompatible parameters.
        """
        self._check_descriptor_params(descriptor)
        idx = len(self._descriptors)
        if frame_id is None:
            frame_id = idx
        self._descriptors.append(descriptor)
        self._frame_ids.append(frame_id)
        row = descriptor.vector.reshape(1, -1)
        self._vectors = np.vstack([self._vectors, row])
        return idx

    def query(
        self,
        descriptor: ImageDescriptor,
        *,
        top_k: int = 1,
    ) -> List[LoopClosureCandidate]:
        """Find the top-*k* loop closure candidates for a query descriptor.

        Args:
            descriptor: The query :class:`ImageDescriptor`.
            top_k: Number of best matches to return.  Default ``1``.

        Returns:
            List of up to *top_k* :class:`LoopClosureCandidate` objects sorted
            by ascending cosine distance.  An empty list is returned when no
            eligible entries exist.

        Raises:
            ValueError: If *descriptor* has incompatible parameters or
                *top_k* < 1.
        """
        self._check_descriptor_params(descriptor)
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")

        n = len(self._descriptors)
        eligible_end = n - self.exclusion_window
        if eligible_end <= 0:
            return []

        q = descriptor.vector  # (D,)
        q_norm = np.linalg.norm(q)

        candidates: List[LoopClosureCandidate] = []
        for db_idx in range(eligible_end):
            v = self._vectors[db_idx]
            v_norm = np.linalg.norm(v)
            if q_norm < _ZERO_NORM_THRESHOLD or v_norm < _ZERO_NORM_THRESHOLD:
                dist = 0.0 if (q_norm < _ZERO_NORM_THRESHOLD and v_norm < _ZERO_NORM_THRESHOLD) else 1.0
            else:
                cos_sim = float(np.dot(q, v) / (q_norm * v_norm))
                dist = float(1.0 - np.clip(cos_sim, -1.0, 1.0))
            candidates.append(
                LoopClosureCandidate(
                    match_frame_id=self._frame_ids[db_idx],
                    distance=dist,
                    yaw_shift_sectors=0,
                    database_index=db_idx,
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

    def _check_descriptor_params(self, descriptor: ImageDescriptor) -> None:
        """Raise ``ValueError`` if descriptor parameters are incompatible."""
        if descriptor.grid_rows != self.grid_rows:
            raise ValueError(
                f"Descriptor grid_rows={descriptor.grid_rows} does not match "
                f"database grid_rows={self.grid_rows}."
            )
        if descriptor.grid_cols != self.grid_cols:
            raise ValueError(
                f"Descriptor grid_cols={descriptor.grid_cols} does not match "
                f"database grid_cols={self.grid_cols}."
            )
        if descriptor.bins != self.bins:
            raise ValueError(
                f"Descriptor bins={descriptor.bins} does not match "
                f"database bins={self.bins}."
            )
