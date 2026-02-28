"""
point_cloud_map.py

Accumulated point-cloud map assembled from successive LiDAR scans.

As a mobile sensor platform moves through its environment, each LiDAR scan
captures a local view of the surroundings in the sensor body frame.  By
transforming every scan into a common world/map frame (using the ego poses
produced by, for example, scan-matching + pose-graph optimisation) and
accumulating the results, a global point-cloud map of the environment can be
built incrementally.

This module provides :class:`PointCloudMap`, a lightweight accumulator that:

* **Accepts coloured or uncoloured scans** – each call to :meth:`add_scan`
  appends the scan's world-frame points (and optional per-point RGB colours)
  to the internal buffers.
* **Applies a rigid ego-to-world transform** – the 4×4 homogeneous matrix
  supplied to :meth:`add_scan` converts sensor-frame points to the world/map
  frame before accumulation.
* **Voxel grid downsampling** – :meth:`voxel_downsample` reduces the map
  to one representative point per voxel cell, keeping memory and later
  processing costs bounded.  The retained point is the centroid of all points
  inside each voxel; colours are averaged over the voxel (when present).

The implementation is pure NumPy and introduces no extra dependencies
beyond those already required by ``sensor_transposition``.

Typical use-case
----------------
::

    from sensor_transposition.point_cloud_map import PointCloudMap

    pcd_map = PointCloudMap()

    for frame_pose, lidar_scan in zip(trajectory, scans):
        pcd_map.add_scan(lidar_scan, frame_pose.transform)

    # Downsample the accumulated map to 10-cm voxels.
    pcd_map.voxel_downsample(voxel_size=0.10)

    world_points = pcd_map.get_points()   # (N, 3) float64
    world_colors = pcd_map.get_colors()   # (N, 3) uint8, or None

Integration with coloured point clouds
---------------------------------------
Use :func:`sensor_transposition.lidar_camera.color_lidar_from_image` to
colour each scan before accumulation::

    from sensor_transposition.lidar_camera import (
        project_lidar_to_image,
        color_lidar_from_image,
    )

    pixel_coords, valid = project_lidar_to_image(
        scan, lidar_to_camera, K, img_w, img_h
    )
    colors = color_lidar_from_image(image, pixel_coords, valid)
    pcd_map.add_scan(scan, ego_to_world, colors=colors)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# PointCloudMap
# ---------------------------------------------------------------------------


class PointCloudMap:
    """Incremental accumulated point-cloud map.

    Points from each incoming scan are transformed into the world/map frame
    and appended to a growing buffer.  An optional voxel-grid downsampling
    step limits memory growth for long trajectories.

    Args:
        max_points: Optional upper bound on the number of world-frame points
            stored in the map.  When the buffer reaches *max_points*, the
            oldest points are discarded (FIFO) to make room for new ones.
            ``None`` (default) means the buffer grows without bound.
    """

    def __init__(self, max_points: Optional[int] = None) -> None:
        if max_points is not None and max_points < 1:
            raise ValueError(
                f"max_points must be >= 1 or None, got {max_points}."
            )
        self._max_points = max_points

        # Internal buffers – lazily allocated on first add_scan call.
        self._points: Optional[np.ndarray] = None   # (N, 3) float64
        self._colors: Optional[np.ndarray] = None   # (N, 3) uint8, or None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_scan(
        self,
        points: np.ndarray,
        ego_to_world: np.ndarray,
        *,
        colors: Optional[np.ndarray] = None,
    ) -> None:
        """Add a LiDAR scan to the map.

        Transforms *points* from the sensor body frame into the world/map
        frame using *ego_to_world* and appends them to the internal buffer.

        Args:
            points: ``(N, 3)`` float array of XYZ coordinates in the sensor
                body frame (metres).
            ego_to_world: 4×4 homogeneous transform that maps sensor-frame
                points to the world/map frame.  Typically a
                :attr:`~sensor_transposition.frame_pose.FramePose.transform`.
            colors: Optional ``(N, 3)`` array of per-point RGB colours.  May
                be ``uint8`` (values in ``[0, 255]``) or ``float`` (values in
                ``[0, 1]``); ``uint8`` values are stored as-is, ``float``
                values are scaled to ``[0, 255]`` and cast to ``uint8``.
                If *colors* is provided, all subsequent calls to
                :meth:`add_scan` should also supply *colors*; mixing coloured
                and uncoloured scans raises :class:`ValueError`.

        Raises:
            ValueError: If *points* is not ``(N, 3)``, *ego_to_world* is not
                4×4, or *colors* has an incompatible shape or mixing mode.
        """
        pts = np.asarray(points, dtype=float)
        T = np.asarray(ego_to_world, dtype=float)

        _validate_points(pts, "points")
        _validate_transform(T)

        N = pts.shape[0]

        # Transform points to world frame.
        R = T[:3, :3]
        t = T[:3, 3]
        world_pts = (R @ pts.T).T + t   # (N, 3)

        # Handle colours.
        if colors is not None:
            clr = np.asarray(colors)
            if clr.shape != (N, 3):
                raise ValueError(
                    f"colors must have shape ({N}, 3), got {clr.shape}."
                )
            # Convert float colours to uint8.
            if np.issubdtype(clr.dtype, np.floating):
                clr = np.clip(clr * 255.0, 0, 255).astype(np.uint8)
            else:
                clr = clr.astype(np.uint8)

            if self._colors is None and self._points is not None:
                raise ValueError(
                    "Cannot mix coloured and uncoloured scans: previous scans "
                    "were added without colours."
                )
        else:
            if self._colors is not None:
                raise ValueError(
                    "Cannot mix coloured and uncoloured scans: previous scans "
                    "were added with colours."
                )
            clr = None

        # Append to buffers.
        if self._points is None:
            self._points = world_pts
            self._colors = clr
        else:
            self._points = np.concatenate([self._points, world_pts], axis=0)
            if clr is not None and self._colors is not None:
                self._colors = np.concatenate([self._colors, clr], axis=0)

        # Apply max_points cap (FIFO: discard oldest entries).
        if self._max_points is not None and len(self._points) > self._max_points:
            excess = len(self._points) - self._max_points
            self._points = self._points[excess:]
            if self._colors is not None:
                self._colors = self._colors[excess:]

    def get_points(self) -> np.ndarray:
        """Return the accumulated world-frame point cloud.

        Returns:
            ``(N, 3)`` float64 array of XYZ coordinates in the world/map
            frame.  Returns an empty ``(0, 3)`` array when no scans have been
            added.
        """
        if self._points is None:
            return np.empty((0, 3), dtype=float)
        return self._points.copy()

    def get_colors(self) -> Optional[np.ndarray]:
        """Return the accumulated per-point RGB colours, or ``None``.

        Returns:
            ``(N, 3)`` uint8 array of RGB colours, or ``None`` when no colour
            information was provided (or after :meth:`clear` is called).
        """
        if self._colors is None:
            return None
        return self._colors.copy()

    def voxel_downsample(self, voxel_size: float) -> None:
        """Downsample the accumulated map using a voxel grid filter.

        The 3-D space is partitioned into axis-aligned cubes of side length
        *voxel_size*.  All points that fall into the same voxel are replaced
        by their centroid; per-point colours (when present) are averaged over
        the voxel.

        This operation modifies the map **in place**.  It is a no-op when the
        map is empty.

        Args:
            voxel_size: Side length of each voxel in metres.  Must be
                strictly positive.

        Raises:
            ValueError: If *voxel_size* is not strictly positive.
        """
        if voxel_size <= 0.0:
            raise ValueError(
                f"voxel_size must be > 0, got {voxel_size}."
            )
        if self._points is None or len(self._points) == 0:
            return

        pts = self._points

        # Compute voxel indices for each point.
        min_coords = pts.min(axis=0)
        voxel_indices = np.floor(
            (pts - min_coords) / voxel_size
        ).astype(np.int64)

        # Encode the 3-D index as a single integer key for grouping.
        # Use a large stride in each dimension to avoid collisions for
        # realistic map extents (up to ~20 km per axis at 1 cm resolution).
        stride = int(
            np.ceil((pts.max(axis=0) - min_coords).max() / voxel_size) + 2
        )
        keys = (
            voxel_indices[:, 0] * stride * stride
            + voxel_indices[:, 1] * stride
            + voxel_indices[:, 2]
        )

        # Sort by key so that equal keys are contiguous.
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        sorted_pts = pts[order]
        sorted_clr = self._colors[order] if self._colors is not None else None

        # Find group boundaries (where the key changes).
        change = np.concatenate([[True], sorted_keys[1:] != sorted_keys[:-1]])
        group_start = np.where(change)[0]

        # Compute per-group centroids (and mean colours).
        n_voxels = len(group_start)
        new_pts = np.empty((n_voxels, 3), dtype=float)
        new_clr: Optional[np.ndarray] = (
            np.empty((n_voxels, 3), dtype=float) if sorted_clr is not None else None
        )

        group_end = np.concatenate([group_start[1:], [len(sorted_pts)]])
        for i, (s, e) in enumerate(zip(group_start, group_end)):
            new_pts[i] = sorted_pts[s:e].mean(axis=0)
            if new_clr is not None and sorted_clr is not None:
                new_clr[i] = sorted_clr[s:e].astype(float).mean(axis=0)

        self._points = new_pts
        self._colors = (
            np.clip(new_clr, 0, 255).astype(np.uint8)
            if new_clr is not None
            else None
        )

    def clear(self) -> None:
        """Remove all accumulated points and colours from the map."""
        self._points = None
        self._colors = None

    def __len__(self) -> int:
        """Return the number of points currently stored in the map."""
        if self._points is None:
            return 0
        return len(self._points)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_points(pts: np.ndarray, name: str) -> None:
    """Raise ``ValueError`` if *pts* is not an (N, 3) array with N >= 1."""
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(
            f"{name} must be an (N, 3) array, got shape {pts.shape}."
        )
    if pts.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one point.")


def _validate_transform(T: np.ndarray) -> None:
    """Raise ``ValueError`` if *T* is not a 4×4 matrix."""
    if T.shape != (4, 4):
        raise ValueError(
            f"ego_to_world must be a 4×4 matrix, got shape {T.shape}."
        )
