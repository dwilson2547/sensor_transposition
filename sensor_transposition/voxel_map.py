"""
voxel_map.py

Truncated Signed-Distance Function (TSDF) volumetric map.

A TSDF volume represents the environment as a regular 3-D grid of voxels.
Each voxel stores a *truncated* signed distance to the nearest surface,
expressed as a value in ``[-1, 1]`` (normalised by the truncation radius),
together with a fusion *weight* that tracks how many observations have
contributed to that voxel.

The sign convention follows KinectFusion (Newcombe *et al.*, 2011):

* **Positive** values indicate voxels that lie *in front of* (closer to the
  sensor than) the observed surface – i.e. free space.
* **Negative** values indicate voxels that lie *behind* the surface –
  i.e. the solid interior of an object.
* **Zero** corresponds to the surface itself.

Sensor measurements are integrated via a running **weighted average**:

.. math::

    D_{new}(v) = \\frac{w(v) \\cdot D(v) + \\hat{d}(v)}{w(v) + 1}, \\quad
    w_{new}(v) = w(v) + 1

where :math:`\\hat{d}(v)` is the normalised SDF value of voxel *v* relative
to the new observation, and the update is only applied when
:math:`|\\text{sdf}| \\le \\text{truncation}`.

The implementation is pure NumPy/SciPy and introduces no additional
dependencies beyond those already required by ``sensor_transposition``.

Typical use-case
----------------
::

    import numpy as np
    from sensor_transposition.voxel_map import TSDFVolume

    # 10-cm voxels, 20 m × 20 m × 5 m volume centred near the origin.
    volume = TSDFVolume(
        voxel_size=0.10,
        origin=np.array([-10.0, -10.0, -0.5]),
        dims=(200, 200, 50),
    )

    for frame_pose, lidar_scan in zip(trajectory, scans):
        volume.integrate(lidar_scan, frame_pose.transform)

    # Extract world-frame surface points (|tsdf| ≤ 0.1).
    surface_pts = volume.extract_surface_points(threshold=0.1)
    print(f"Surface point cloud: {surface_pts.shape[0]:,} points")

    tsdf_array  = volume.get_tsdf()    # (nx, ny, nz) float64; NaN = unseen
    weight_array = volume.get_weights() # (nx, ny, nz) float64
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


class TSDFVolume:
    """Volumetric TSDF map for dense 3-D reconstruction.

    The volume is a regular axis-aligned grid of voxels.  Each voxel stores a
    normalised truncated signed-distance value in ``[-1, 1]`` and a fusion
    weight.  New sensor observations are integrated via a running weighted
    average, and surface points can be extracted by finding voxels near the
    zero-crossing.

    Args:
        voxel_size: Side length of each cubic voxel in metres.  Must be
            strictly positive.
        origin: ``(3,)`` array ``[x0, y0, z0]`` giving the world-frame
            coordinates of the **corner** of voxel ``(0, 0, 0)`` (not the
            centre).  Defaults to ``[0, 0, 0]``.
        dims: ``(nx, ny, nz)`` tuple specifying the number of voxels along
            each axis.  Each dimension must be at least 1.
        truncation: Truncation radius in metres.  SDF values outside
            ``[-truncation, +truncation]`` are clamped and voxels further than
            ``truncation`` from an observed surface are not updated.  Defaults
            to ``3 * voxel_size``.

    Raises:
        ValueError: If *voxel_size* is not strictly positive, any dimension in
            *dims* is less than 1, *origin* does not have shape ``(3,)``, or
            *truncation* is not strictly positive.
    """

    def __init__(
        self,
        voxel_size: float,
        origin: Optional[np.ndarray] = None,
        dims: Tuple[int, int, int] = (100, 100, 100),
        truncation: Optional[float] = None,
    ) -> None:
        if voxel_size <= 0.0:
            raise ValueError(
                f"voxel_size must be > 0, got {voxel_size}."
            )

        nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError(
                f"Each dimension in dims must be >= 1, got ({nx}, {ny}, {nz})."
            )

        if origin is None:
            origin = np.zeros(3, dtype=float)
        else:
            origin = np.asarray(origin, dtype=float)
            if origin.shape != (3,):
                raise ValueError(
                    f"origin must have shape (3,), got {origin.shape}."
                )

        if truncation is None:
            truncation = 3.0 * voxel_size
        else:
            truncation = float(truncation)
            if truncation <= 0.0:
                raise ValueError(
                    f"truncation must be > 0, got {truncation}."
                )

        self._voxel_size = float(voxel_size)
        self._origin = origin.copy()
        self._nx = nx
        self._ny = ny
        self._nz = nz
        self._truncation = truncation

        # TSDF values – NaN indicates an unobserved voxel.
        self._tsdf: np.ndarray = np.full(
            (nx, ny, nz), fill_value=np.nan, dtype=float
        )
        # Fusion weights – 0 means the voxel has not been updated.
        self._weights: np.ndarray = np.zeros((nx, ny, nz), dtype=float)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def voxel_size(self) -> float:
        """Voxel side length in metres."""
        return self._voxel_size

    @property
    def origin(self) -> np.ndarray:
        """World-frame coordinates of the corner of voxel (0, 0, 0)."""
        return self._origin.copy()

    @property
    def dims(self) -> Tuple[int, int, int]:
        """Number of voxels along each axis ``(nx, ny, nz)``."""
        return (self._nx, self._ny, self._nz)

    @property
    def truncation(self) -> float:
        """Truncation distance in metres."""
        return self._truncation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def integrate(
        self,
        points: np.ndarray,
        ego_to_world: np.ndarray,
    ) -> None:
        """Fuse a new point-cloud observation into the TSDF volume.

        Each input point is taken to be a surface observation.  For every
        point, the voxels within the truncation radius are identified and their
        TSDF values are updated using a running weighted average.  The signed
        distance for a voxel *v* relative to an observed surface point *P*
        with sensor origin *O* is:

        .. code-block:: text

            d = normalise(P - O)          # unit ray direction
            sdf(v) = ||P - O|| - (v_centre - O) · d

        Positive sdf → voxel is closer to the sensor than the surface (free).
        Negative sdf → voxel is behind the surface (occupied interior).

        Only voxels with ``|sdf| ≤ truncation`` are updated.  The stored value
        is ``sdf / truncation`` (normalised to ``[-1, 1]``).

        Args:
            points: ``(N, 3)`` float array of XYZ coordinates in the sensor
                body frame (metres).
            ego_to_world: 4×4 homogeneous transform that maps sensor-frame
                points to the world/map frame.

        Raises:
            ValueError: If *points* is not ``(N, 3)`` with N ≥ 1, or
                *ego_to_world* is not 4×4.
        """
        pts = np.asarray(points, dtype=float)
        T = np.asarray(ego_to_world, dtype=float)

        _validate_points(pts)
        _validate_transform(T)

        R = T[:3, :3]
        t = T[:3, 3]

        # World-frame points and sensor origin.
        world_pts = (R @ pts.T).T + t    # (N, 3)
        sensor_origin = t                # (3,)

        trunc = self._truncation
        vs = self._voxel_size
        origin = self._origin
        nx, ny, nz = self._nx, self._ny, self._nz
        dims_max = np.array([nx - 1, ny - 1, nz - 1], dtype=int)

        for P_w in world_pts:
            D_vec = P_w - sensor_origin
            D = float(np.sqrt(np.sum(D_vec ** 2)))
            if D < 1e-9:
                continue
            d = D_vec / D   # unit ray direction

            # Bounding box of voxel indices within truncation distance of P_w.
            P_idx = (P_w - origin) / vs   # float voxel index
            r = trunc / vs                 # half-width in voxel units

            lo = np.maximum(np.floor(P_idx - r).astype(int), 0)
            hi = np.minimum(np.ceil(P_idx + r).astype(int), dims_max)

            if np.any(lo > hi):
                continue

            # Grid of voxel indices in the bounding box.
            ix = np.arange(lo[0], hi[0] + 1)
            iy = np.arange(lo[1], hi[1] + 1)
            iz = np.arange(lo[2], hi[2] + 1)
            gx, gy, gz = np.meshgrid(ix, iy, iz, indexing="ij")  # (sx, sy, sz)

            # Voxel centres in world frame.
            vc_x = origin[0] + (gx + 0.5) * vs
            vc_y = origin[1] + (gy + 0.5) * vs
            vc_z = origin[2] + (gz + 0.5) * vs

            # SDF: D - (V_c - O) · d
            proj = (
                (vc_x - sensor_origin[0]) * d[0]
                + (vc_y - sensor_origin[1]) * d[1]
                + (vc_z - sensor_origin[2]) * d[2]
            )
            sdf = D - proj  # (sx, sy, sz)

            # Only update voxels within the truncation band.
            valid = np.abs(sdf) <= trunc  # boolean (sx, sy, sz)

            # Normalise SDF to [-1, 1].
            sdf_norm = np.clip(sdf / trunc, -1.0, 1.0)

            # Running weighted average update.
            w_cur = self._weights[gx, gy, gz]
            t_cur = self._tsdf[gx, gy, gz]

            w_new = w_cur + valid.astype(float)
            safe_weight = np.where(w_new > 0, w_new, 1.0)

            # For previously unobserved voxels (NaN), treat t_cur as 0.
            t_cur_safe = np.where(np.isnan(t_cur), 0.0, t_cur)

            t_new = np.where(
                valid,
                (t_cur_safe * w_cur + sdf_norm) / safe_weight,
                t_cur,
            )

            self._tsdf[gx, gy, gz] = t_new
            self._weights[gx, gy, gz] = w_new

    def get_tsdf(self) -> np.ndarray:
        """Return the TSDF values for the entire volume.

        Returns:
            ``(nx, ny, nz)`` float64 array.  Values in ``[-1, 1]`` for
            observed voxels; ``NaN`` for voxels that have not yet been
            updated.
        """
        return self._tsdf.copy()

    def get_weights(self) -> np.ndarray:
        """Return the fusion weights for the entire volume.

        Returns:
            ``(nx, ny, nz)`` float64 array.  Zero for unobserved voxels;
            positive (equal to the number of contributing observations) for
            updated voxels.
        """
        return self._weights.copy()

    def extract_surface_points(self, threshold: float = 0.1) -> np.ndarray:
        """Extract world-frame surface points near the TSDF zero-crossing.

        Returns the centres of all voxels that have been observed (weight > 0)
        and whose normalised TSDF value satisfies ``|tsdf| ≤ threshold``.

        Args:
            threshold: Normalised TSDF threshold in ``[0, 1]``.  Voxels with
                ``|tsdf| ≤ threshold`` are considered surface voxels.
                Defaults to ``0.1`` (10 % of the truncation distance).

        Returns:
            ``(M, 3)`` float64 array of world-frame XYZ surface point
            coordinates (voxel centres).  Returns an empty ``(0, 3)`` array
            when no surface voxels are found.

        Raises:
            ValueError: If *threshold* is not in ``[0, 1]``.
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"threshold must be in [0, 1], got {threshold}."
            )
        mask = (self._weights > 0) & (~np.isnan(self._tsdf)) & (
            np.abs(self._tsdf) <= threshold
        )
        ix, iy, iz = np.where(mask)
        if len(ix) == 0:
            return np.empty((0, 3), dtype=float)

        x = self._origin[0] + (ix + 0.5) * self._voxel_size
        y = self._origin[1] + (iy + 0.5) * self._voxel_size
        z = self._origin[2] + (iz + 0.5) * self._voxel_size
        return np.column_stack([x, y, z])

    def voxel_to_world(
        self, ix: int, iy: int, iz: int
    ) -> Tuple[float, float, float]:
        """Return the world-frame centre of a voxel.

        Args:
            ix: Voxel index along the x axis.
            iy: Voxel index along the y axis.
            iz: Voxel index along the z axis.

        Returns:
            ``(x, y, z)`` world-frame coordinates of the voxel centre in
            metres.
        """
        x = self._origin[0] + (ix + 0.5) * self._voxel_size
        y = self._origin[1] + (iy + 0.5) * self._voxel_size
        z = self._origin[2] + (iz + 0.5) * self._voxel_size
        return float(x), float(y), float(z)

    def world_to_voxel(
        self, x: float, y: float, z: float
    ) -> Tuple[float, float, float]:
        """Return the (possibly non-integer) voxel-space coordinates for a
        world-frame position.

        The integer part of each returned value is the voxel index; the
        fractional part indicates the position within the voxel.  Values may
        lie outside ``[0, nx/ny/nz)`` when the world position is outside the
        volume.

        Args:
            x: World-frame x coordinate in metres.
            y: World-frame y coordinate in metres.
            z: World-frame z coordinate in metres.

        Returns:
            ``(fx, fy, fz)`` floating-point voxel coordinates.
        """
        fx = (x - self._origin[0]) / self._voxel_size
        fy = (y - self._origin[1]) / self._voxel_size
        fz = (z - self._origin[2]) / self._voxel_size
        return float(fx), float(fy), float(fz)

    def clear(self) -> None:
        """Reset all voxels to the unobserved state (NaN TSDF, zero weight)."""
        self._tsdf[:] = np.nan
        self._weights[:] = 0.0


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------


def _validate_points(pts: np.ndarray) -> None:
    """Raise ``ValueError`` if *pts* is not an (N, 3) array with N >= 1."""
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(
            f"points must be an (N, 3) array, got shape {pts.shape}."
        )
    if pts.shape[0] < 1:
        raise ValueError("points must contain at least one point.")


def _validate_transform(T: np.ndarray) -> None:
    """Raise ``ValueError`` if *T* is not a 4×4 matrix."""
    if T.shape != (4, 4):
        raise ValueError(
            f"ego_to_world must be a 4×4 matrix, got shape {T.shape}."
        )


# ---------------------------------------------------------------------------
# SparseTSDFVolume
# ---------------------------------------------------------------------------


class SparseTSDFVolume:
    """Volumetric TSDF map that stores only observed voxels.

    This is a **drop-in replacement** for :class:`TSDFVolume` for large
    outdoor environments where most of the bounding volume is never
    observed.  Instead of allocating two dense ``(nx, ny, nz)`` NumPy
    arrays up front, :class:`SparseTSDFVolume` stores TSDF values and
    fusion weights in Python ``dict`` objects keyed by ``(ix, iy, iz)``
    voxel index, allocating memory lazily as sensor observations arrive.

    **Memory comparison**

    For a ``200 × 200 × 50`` voxel volume (20 m × 20 m × 5 m at 10 cm
    resolution — a typical indoor mapping scenario):

    * :class:`TSDFVolume` – always allocates ``2 000 000`` float64 values
      for TSDF + ``2 000 000`` for weights ≈ **32 MB** regardless of how
      many voxels are actually near a surface.
    * :class:`SparseTSDFVolume` – uses roughly ``100 bytes`` per observed
      voxel (two ``dict`` entries with tuple keys).  For a scene where
      5 000 surface voxels are updated, peak memory is ≈ **500 KB** — a
      **64× reduction**.

    The API is identical to :class:`TSDFVolume`: the same constructor
    arguments, ``integrate``, ``get_tsdf``, ``get_weights``,
    ``extract_surface_points``, ``voxel_to_world``, ``world_to_voxel``,
    and ``clear`` methods are all present with the same signatures and
    semantics.

    Args:
        voxel_size: Side length of each cubic voxel in metres.  Must be
            strictly positive.
        origin: ``(3,)`` array ``[x0, y0, z0]`` giving the world-frame
            coordinates of the **corner** of voxel ``(0, 0, 0)``.
            Defaults to ``[0, 0, 0]``.
        dims: ``(nx, ny, nz)`` tuple specifying the number of voxels along
            each axis.  Used for bounds-checking only; no memory is
            reserved up front.
        truncation: Truncation radius in metres.  Defaults to
            ``3 * voxel_size``.

    Internal storage:
        Observed voxels are stored in ``_voxels: dict[(ix, iy, iz), (tsdf, weight)]``
        mapping each ``(ix, iy, iz)`` integer voxel index tuple to a
        ``(tsdf_value, weight)`` pair.  Voxels are allocated on first
        integration update; unobserved voxels have no entry in the dict.

    Raises:
        ValueError: If *voxel_size* is not strictly positive, any
            dimension in *dims* is less than 1, *origin* does not have
            shape ``(3,)``, or *truncation* is not strictly positive.
    """

    def __init__(
        self,
        voxel_size: float,
        origin: Optional[np.ndarray] = None,
        dims: Tuple[int, int, int] = (100, 100, 100),
        truncation: Optional[float] = None,
    ) -> None:
        if voxel_size <= 0.0:
            raise ValueError(
                f"voxel_size must be > 0, got {voxel_size}."
            )

        nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError(
                f"Each dimension in dims must be >= 1, got ({nx}, {ny}, {nz})."
            )

        if origin is None:
            origin = np.zeros(3, dtype=float)
        else:
            origin = np.asarray(origin, dtype=float)
            if origin.shape != (3,):
                raise ValueError(
                    f"origin must have shape (3,), got {origin.shape}."
                )

        if truncation is None:
            truncation = 3.0 * voxel_size
        else:
            truncation = float(truncation)
            if truncation <= 0.0:
                raise ValueError(
                    f"truncation must be > 0, got {truncation}."
                )

        self._voxel_size = float(voxel_size)
        self._origin = origin.copy()
        self._nx = nx
        self._ny = ny
        self._nz = nz
        self._truncation = truncation

        # Sparse storage: maps (ix, iy, iz) → (tsdf_value, weight).
        self._voxels: Dict[Tuple[int, int, int], Tuple[float, float]] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def voxel_size(self) -> float:
        """Voxel side length in metres."""
        return self._voxel_size

    @property
    def origin(self) -> np.ndarray:
        """World-frame coordinates of the corner of voxel (0, 0, 0)."""
        return self._origin.copy()

    @property
    def dims(self) -> Tuple[int, int, int]:
        """Number of voxels along each axis ``(nx, ny, nz)``."""
        return (self._nx, self._ny, self._nz)

    @property
    def truncation(self) -> float:
        """Truncation distance in metres."""
        return self._truncation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def integrate(
        self,
        points: np.ndarray,
        ego_to_world: np.ndarray,
    ) -> None:
        """Fuse a new point-cloud observation into the sparse TSDF volume.

        Identical semantics to :meth:`TSDFVolume.integrate`: for each
        observed surface point, voxels within the truncation band are
        updated via a running weighted average.  Voxels are allocated on
        first write; voxels outside the volume bounds are silently ignored.

        Args:
            points: ``(N, 3)`` float array of XYZ coordinates in the
                sensor body frame (metres).
            ego_to_world: 4×4 homogeneous transform mapping sensor-frame
                points to the world/map frame.

        Raises:
            ValueError: If *points* is not ``(N, 3)`` with N ≥ 1, or
                *ego_to_world* is not 4×4.
        """
        pts = np.asarray(points, dtype=float)
        T = np.asarray(ego_to_world, dtype=float)

        _validate_points(pts)
        _validate_transform(T)

        R = T[:3, :3]
        t = T[:3, 3]

        world_pts = (R @ pts.T).T + t
        sensor_origin = t

        trunc = self._truncation
        vs = self._voxel_size
        origin = self._origin
        nx, ny, nz = self._nx, self._ny, self._nz

        for P_w in world_pts:
            D_vec = P_w - sensor_origin
            D = float(np.sqrt(np.sum(D_vec ** 2)))
            if D < 1e-9:
                continue
            d = D_vec / D

            P_idx = (P_w - origin) / vs
            r = trunc / vs

            lo = np.maximum(np.floor(P_idx - r).astype(int), 0)
            hi = np.minimum(np.ceil(P_idx + r).astype(int),
                            np.array([nx - 1, ny - 1, nz - 1], dtype=int))

            if np.any(lo > hi):
                continue

            ix_range = np.arange(lo[0], hi[0] + 1)
            iy_range = np.arange(lo[1], hi[1] + 1)
            iz_range = np.arange(lo[2], hi[2] + 1)
            gx, gy, gz = np.meshgrid(ix_range, iy_range, iz_range,
                                      indexing="ij")

            vc_x = origin[0] + (gx + 0.5) * vs
            vc_y = origin[1] + (gy + 0.5) * vs
            vc_z = origin[2] + (gz + 0.5) * vs

            proj = (
                (vc_x - sensor_origin[0]) * d[0]
                + (vc_y - sensor_origin[1]) * d[1]
                + (vc_z - sensor_origin[2]) * d[2]
            )
            sdf = D - proj

            valid = np.abs(sdf) <= trunc
            sdf_norm = np.clip(sdf / trunc, -1.0, 1.0)

            # Update only voxels within the truncation band.
            valid_idx = np.argwhere(valid)
            for vi in valid_idx:
                ixi = int(gx[vi[0], vi[1], vi[2]])
                iyi = int(gy[vi[0], vi[1], vi[2]])
                izi = int(gz[vi[0], vi[1], vi[2]])
                sdf_v = float(sdf_norm[vi[0], vi[1], vi[2]])

                key = (ixi, iyi, izi)
                if key in self._voxels:
                    t_cur, w_cur = self._voxels[key]
                else:
                    t_cur, w_cur = 0.0, 0.0

                w_new = w_cur + 1.0
                t_new = (t_cur * w_cur + sdf_v) / w_new
                self._voxels[key] = (t_new, w_new)

    def get_tsdf(self) -> np.ndarray:
        """Return the TSDF values for the entire volume as a dense array.

        Returns:
            ``(nx, ny, nz)`` float64 array.  Values in ``[-1, 1]`` for
            observed voxels; ``NaN`` for voxels that have not been updated.
        """
        result = np.full((self._nx, self._ny, self._nz), fill_value=np.nan,
                         dtype=float)
        for (ix, iy, iz), (t_val, _) in self._voxels.items():
            result[ix, iy, iz] = t_val
        return result

    def get_weights(self) -> np.ndarray:
        """Return the fusion weights for the entire volume as a dense array.

        Returns:
            ``(nx, ny, nz)`` float64 array.  Zero for unobserved voxels;
            positive for updated voxels.
        """
        result = np.zeros((self._nx, self._ny, self._nz), dtype=float)
        for (ix, iy, iz), (_, w_val) in self._voxels.items():
            result[ix, iy, iz] = w_val
        return result

    def extract_surface_points(self, threshold: float = 0.1) -> np.ndarray:
        """Extract world-frame surface points near the TSDF zero-crossing.

        Args:
            threshold: Normalised TSDF threshold in ``[0, 1]``.  Voxels
                with ``|tsdf| ≤ threshold`` are considered surface voxels.
                Defaults to ``0.1``.

        Returns:
            ``(M, 3)`` float64 array of world-frame XYZ surface point
            coordinates.  Returns an empty ``(0, 3)`` array when no
            surface voxels are found.

        Raises:
            ValueError: If *threshold* is not in ``[0, 1]``.
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"threshold must be in [0, 1], got {threshold}."
            )

        pts = []
        vs = self._voxel_size
        origin = self._origin
        for (ix, iy, iz), (t_val, w_val) in self._voxels.items():
            if w_val > 0 and abs(t_val) <= threshold:
                x = origin[0] + (ix + 0.5) * vs
                y = origin[1] + (iy + 0.5) * vs
                z = origin[2] + (iz + 0.5) * vs
                pts.append((x, y, z))

        if not pts:
            return np.empty((0, 3), dtype=float)
        return np.array(pts, dtype=float)

    def voxel_to_world(
        self, ix: int, iy: int, iz: int
    ) -> Tuple[float, float, float]:
        """Return the world-frame centre of a voxel.

        Args:
            ix: Voxel index along the x axis.
            iy: Voxel index along the y axis.
            iz: Voxel index along the z axis.

        Returns:
            ``(x, y, z)`` world-frame coordinates of the voxel centre.
        """
        x = self._origin[0] + (ix + 0.5) * self._voxel_size
        y = self._origin[1] + (iy + 0.5) * self._voxel_size
        z = self._origin[2] + (iz + 0.5) * self._voxel_size
        return float(x), float(y), float(z)

    def world_to_voxel(
        self, x: float, y: float, z: float
    ) -> Tuple[float, float, float]:
        """Return the (possibly non-integer) voxel-space coordinates for a
        world-frame position.

        Args:
            x: World-frame x coordinate in metres.
            y: World-frame y coordinate in metres.
            z: World-frame z coordinate in metres.

        Returns:
            ``(fx, fy, fz)`` floating-point voxel coordinates.
        """
        fx = (x - self._origin[0]) / self._voxel_size
        fy = (y - self._origin[1]) / self._voxel_size
        fz = (z - self._origin[2]) / self._voxel_size
        return float(fx), float(fy), float(fz)

    def clear(self) -> None:
        """Reset all voxels to the unobserved state."""
        self._voxels.clear()
