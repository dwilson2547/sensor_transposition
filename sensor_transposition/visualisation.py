"""
visualisation.py

Multi-sensor synchronised visualisation utilities.

This module provides lightweight, pure-NumPy helpers for rendering and
exporting synchronised multi-sensor data.  No display framework is required;
all renderers return plain NumPy arrays (``uint8`` RGB images) that can be
shown by any image viewer, written to disk, or passed to external tools such
as **Open3D**, **RViz**, or **rerun.io**.

Available helpers
-----------------
* :func:`colour_by_height` – map z-values to a jet-like RGB colour palette.
* :func:`render_birdseye_view` – project a 3-D point cloud onto a 2-D
  bird's-eye-view (BEV) canvas coloured by height or a supplied colour array.
* :func:`render_trajectory_birdseye` – draw a trajectory of XY positions onto
  a BEV canvas as coloured dots.
* :func:`overlay_lidar_on_image` – overlay depth-coded LiDAR points on top of
  a camera image.
* :func:`export_point_cloud_open3d` – serialise a point cloud to the dict
  format accepted by ``open3d.geometry.PointCloud`` (``from_dict`` /
  ``to_dict``).
* :func:`export_trajectory_rviz` – serialise a list of
  :class:`~sensor_transposition.frame_pose.FramePose` objects to a list of
  RViz-compatible ``visualization_msgs/Marker`` dicts (``SPHERE``
  sub-type).
* :class:`SensorFrameVisualiser` – a per-frame container that holds one
  synchronised set of sensor observations (point cloud, camera image,
  trajectory, radar scan) and exposes rendering helpers.

Typical use-case
----------------
::

    from sensor_transposition.visualisation import (
        SensorFrameVisualiser,
        render_birdseye_view,
        overlay_lidar_on_image,
    )
    from sensor_transposition.lidar_camera import (
        project_lidar_to_image,
    )

    # Render a standalone bird's-eye view of the accumulated map.
    bev = render_birdseye_view(map_points, resolution=0.10)

    # Overlay LiDAR depth on the current camera image.
    pixel_coords, valid = project_lidar_to_image(
        lidar_scan, lidar_to_cam, K, W, H
    )
    depth = lidar_scan[:, 0]   # use x-distance as proxy for depth
    overlay = overlay_lidar_on_image(camera_image, pixel_coords, valid, depth)

    # Use the frame container for a combined view.
    vis = SensorFrameVisualiser()
    vis.set_point_cloud(lidar_scan)
    vis.set_camera_image(camera_image)
    vis.set_trajectory(trajectory_xy)
    bev_frame = vis.render_birdseye(resolution=0.10)
    cam_frame = vis.render_camera_with_lidar(pixel_coords, valid, depth)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------


def colour_by_height(
    z_values: np.ndarray,
    *,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
) -> np.ndarray:
    """Map z (height) values to jet-like RGB colours.

    The input range ``[z_min, z_max]`` is linearly mapped to ``[0, 1]`` and
    then passed through a piecewise jet-like colour ramp::

        blue  (0.0) → cyan (0.25) → green (0.5) → yellow (0.75) → red (1.0)

    The mapping is implemented entirely with NumPy and requires no additional
    dependencies.

    Args:
        z_values: 1-D float array of height values.
        z_min: Lower bound of the colour scale.  Points at or below this
            height receive the coldest colour (blue).  Defaults to
            ``z_values.min()``.
        z_max: Upper bound of the colour scale.  Points at or above this
            height receive the warmest colour (red).  Defaults to
            ``z_values.max()``.  Must be strictly greater than *z_min* if
            both are supplied.

    Returns:
        ``(N, 3)`` ``uint8`` array of RGB colours, one row per element of
        *z_values*.

    Raises:
        ValueError: If *z_min* >= *z_max* when both are explicitly provided.
    """
    z = np.asarray(z_values, dtype=float).ravel()
    N = z.size

    lo = float(z.min()) if z_min is None else float(z_min)
    hi = float(z.max()) if z_max is None else float(z_max)

    if z_min is not None and z_max is not None and lo >= hi:
        raise ValueError(
            f"z_min must be < z_max, got z_min={lo}, z_max={hi}."
        )

    # Normalise to [0, 1]; handle the degenerate case where all heights are
    # equal by mapping everything to the midpoint colour.
    if hi - lo < 1e-12:
        t = np.full(N, 0.5, dtype=float)
    else:
        t = np.clip((z - lo) / (hi - lo), 0.0, 1.0)

    return _jet_colormap(t)


def _jet_colormap(t: np.ndarray) -> np.ndarray:
    """Convert normalised values in ``[0, 1]`` to jet-like RGB ``uint8``.

    Four-segment piecewise linear ramp:

    * ``t = 0.00`` → ``(  0,   0, 255)``  — blue
    * ``t = 0.25`` → ``(  0, 255, 255)``  — cyan
    * ``t = 0.50`` → ``(  0, 255,   0)``  — green
    * ``t = 0.75`` → ``(255, 255,   0)``  — yellow
    * ``t = 1.00`` → ``(255,   0,   0)``  — red

    Args:
        t: 1-D float array with values in ``[0, 1]``.

    Returns:
        ``(N, 3)`` ``uint8`` RGB array.
    """
    t = np.asarray(t, dtype=float).ravel()

    # Four segments [0,0.25], [0.25,0.5], [0.5,0.75], [0.75,1.0].
    r = np.where(
        t < 0.5,
        0.0,
        np.where(t < 0.75, 4.0 * (t - 0.5), 1.0),
    )
    g = np.where(
        t < 0.25,
        4.0 * t,
        np.where(t < 0.75, 1.0, 1.0 - 4.0 * (t - 0.75)),
    )
    b = np.where(
        t < 0.25,
        1.0,
        np.where(t < 0.5, 1.0 - 4.0 * (t - 0.25), 0.0),
    )

    return (np.clip(np.stack([r, g, b], axis=1), 0.0, 1.0) * 255.0).astype(np.uint8)


# American-spelling alias for colour_by_height.
color_by_height = colour_by_height


# ---------------------------------------------------------------------------
# Bird's-eye view rendering
# ---------------------------------------------------------------------------


def render_birdseye_view(
    points: np.ndarray,
    *,
    resolution: float = 0.10,
    colors: Optional[np.ndarray] = None,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    canvas_size: Optional[Tuple[int, int]] = None,
    origin: Optional[Tuple[float, float]] = None,
    background: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Render a 3-D point cloud as a 2-D bird's-eye-view (BEV) RGB image.

    Each point is projected onto the XY plane and plotted as a single pixel.
    Colour is taken from *colors* when supplied, otherwise computed from the
    point's z-height via :func:`colour_by_height`.

    The canvas coordinate system has:

    * the positive X axis pointing **right** (column direction);
    * the positive Y axis pointing **up** (decreasing row index).

    The canvas origin is placed at the *bottom-left* corner of the image by
    default (see *origin* parameter).

    Args:
        points: ``(N, 3)`` float array of XYZ point coordinates (metres).
        resolution: Metres per pixel.  Smaller values produce larger, more
            detailed images.  Must be strictly positive.  Default ``0.10``.
        colors: Optional ``(N, 3)`` ``uint8`` or float array of per-point RGB
            colours.  When ``None`` (default), colours are computed from the
            z-component using a jet-like ramp.
        z_min: Lower bound of the z-colour scale (ignored when *colors* is
            supplied).
        z_max: Upper bound of the z-colour scale (ignored when *colors* is
            supplied).
        canvas_size: ``(width, height)`` in pixels.  When ``None`` (default)
            the canvas is sized to exactly contain all projected points with a
            2-pixel border on each side.
        origin: ``(x_world, y_world)`` world-space coordinates that map to the
            bottom-left pixel of the canvas.  When ``None`` the bottom-left
            corner of the point cloud's bounding box (minus one pixel of
            margin) is used.
        background: Background colour as an ``(R, G, B)`` ``int`` tuple.
            Default ``(0, 0, 0)`` (black).

    Returns:
        ``(H, W, 3)`` ``uint8`` RGB image.

    Raises:
        ValueError: If *points* is not ``(N, 3)``, *resolution* is not
            positive, or *canvas_size* contains non-positive dimensions.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(
            f"points must be an (N, 3) array, got shape {pts.shape}."
        )
    if resolution <= 0.0:
        raise ValueError(
            f"resolution must be strictly positive, got {resolution}."
        )

    if pts.shape[0] == 0:
        W, H = canvas_size if canvas_size is not None else (1, 1)
        _check_canvas_size(W, H)
        return np.full((H, W, 3), background, dtype=np.uint8)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # Determine origin and canvas size.
    margin = resolution  # 1-pixel border
    if origin is None:
        ox = float(x.min()) - margin
        oy = float(y.min()) - margin
    else:
        ox, oy = float(origin[0]), float(origin[1])

    if canvas_size is None:
        W = int(np.ceil((float(x.max()) - ox + margin) / resolution)) + 1
        H = int(np.ceil((float(y.max()) - oy + margin) / resolution)) + 1
    else:
        W, H = canvas_size
    _check_canvas_size(W, H)

    # Compute per-point RGB colours.
    if colors is not None:
        clr = np.asarray(colors)
        if clr.shape != (pts.shape[0], 3):
            raise ValueError(
                f"colors must have shape ({pts.shape[0]}, 3), got {clr.shape}."
            )
        if np.issubdtype(clr.dtype, np.floating):
            clr = np.clip(clr * 255.0, 0, 255).astype(np.uint8)
        else:
            clr = clr.astype(np.uint8)
    else:
        clr = colour_by_height(z, z_min=z_min, z_max=z_max)

    # Project to pixel coordinates.
    col = np.floor((x - ox) / resolution).astype(int)
    row = H - 1 - np.floor((y - oy) / resolution).astype(int)

    # Clamp to canvas bounds.
    valid = (col >= 0) & (col < W) & (row >= 0) & (row < H)

    canvas = np.full((H, W, 3), background, dtype=np.uint8)
    canvas[row[valid], col[valid]] = clr[valid]
    return canvas


def render_trajectory_birdseye(
    positions: np.ndarray,
    *,
    resolution: float = 0.10,
    canvas_size: Optional[Tuple[int, int]] = None,
    origin: Optional[Tuple[float, float]] = None,
    color: Tuple[int, int, int] = (255, 127, 0),
    background: Tuple[int, int, int] = (0, 0, 0),
    dot_radius: int = 2,
) -> np.ndarray:
    """Render a sequence of XY trajectory positions as a BEV image.

    Each pose is drawn as a filled square of side ``2*dot_radius + 1`` pixels,
    coloured with *color*.  Consecutive positions are **not** connected by
    line segments; this keeps the implementation dependency-free.

    Args:
        positions: ``(M, 2)`` or ``(M, 3)`` float array of XY(Z) world-frame
            positions in metres.  Only the first two columns (X, Y) are used.
        resolution: Metres per pixel.  Must be strictly positive.  Default
            ``0.10``.
        canvas_size: ``(width, height)`` in pixels; auto-sized when ``None``.
        origin: ``(x_world, y_world)`` world coordinate at the bottom-left of
            the canvas; auto-computed when ``None``.
        color: RGB colour for the trajectory dots.  Default orange
            ``(255, 127, 0)``.
        background: Background colour.  Default black ``(0, 0, 0)``.
        dot_radius: Half-side of the filled square drawn at each pose.
            Default ``2`` pixels.

    Returns:
        ``(H, W, 3)`` ``uint8`` RGB image.

    Raises:
        ValueError: If *positions* does not have at least 2 columns, or
            *resolution* / *canvas_size* dimensions are invalid.
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] < 2:
        raise ValueError(
            f"positions must be an (M, ≥2) array, got shape {pos.shape}."
        )
    if resolution <= 0.0:
        raise ValueError(
            f"resolution must be strictly positive, got {resolution}."
        )

    x, y = pos[:, 0], pos[:, 1]

    margin = resolution
    if origin is None:
        ox = float(x.min()) - margin if len(x) > 0 else 0.0
        oy = float(y.min()) - margin if len(y) > 0 else 0.0
    else:
        ox, oy = float(origin[0]), float(origin[1])

    if canvas_size is None:
        if len(x) == 0:
            W, H = 1, 1
        else:
            W = int(np.ceil((float(x.max()) - ox + margin) / resolution)) + 1
            H = int(np.ceil((float(y.max()) - oy + margin) / resolution)) + 1
    else:
        W, H = canvas_size
    _check_canvas_size(W, H)

    canvas = np.full((H, W, 3), background, dtype=np.uint8)

    for xi, yi in zip(x, y):
        col = int(np.floor((xi - ox) / resolution))
        row = H - 1 - int(np.floor((yi - oy) / resolution))
        r0 = max(row - dot_radius, 0)
        r1 = min(row + dot_radius + 1, H)
        c0 = max(col - dot_radius, 0)
        c1 = min(col + dot_radius + 1, W)
        if r0 < r1 and c0 < c1:
            canvas[r0:r1, c0:c1] = color

    return canvas


# ---------------------------------------------------------------------------
# Camera image overlay
# ---------------------------------------------------------------------------


def overlay_lidar_on_image(
    image: np.ndarray,
    pixel_coords: np.ndarray,
    valid: np.ndarray,
    depth: np.ndarray,
    *,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
    radius: int = 2,
) -> np.ndarray:
    """Overlay depth-coded LiDAR points on a camera image.

    For each LiDAR point that projects into the camera field of view, a small
    filled square of side ``2*radius + 1`` pixels is drawn on the image, colour-
    coded by *depth* using a jet-like palette.  The original image is not
    modified; a copy is returned.

    Args:
        image: ``(H, W, 3)`` ``uint8`` RGB camera image.
        pixel_coords: ``(N, 2)`` float array of ``(u, v)`` pixel coordinates
            as returned by
            :func:`~sensor_transposition.lidar_camera.project_lidar_to_image`.
        valid: ``(N,)`` boolean mask; only points where ``valid[i]`` is
            ``True`` are drawn.
        depth: ``(N,)`` float array of depth (or range) values used for
            colour-coding.
        d_min: Lower depth clamp for the colour scale.  Defaults to the
            minimum depth among valid points.
        d_max: Upper depth clamp for the colour scale.  Defaults to the
            maximum depth among valid points.
        radius: Half-side of the filled square drawn at each point.  Default
            ``2`` pixels.

    Returns:
        ``(H, W, 3)`` ``uint8`` RGB image (a copy of *image* with the LiDAR
        overlay applied).

    Raises:
        ValueError: If *image* is not ``(H, W, 3)`` uint8, or the array
            shapes are inconsistent.
    """
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"image must be an (H, W, 3) array, got shape {img.shape}."
        )
    H, W = img.shape[:2]

    pix = np.asarray(pixel_coords, dtype=float)
    msk = np.asarray(valid, dtype=bool)
    d = np.asarray(depth, dtype=float)

    N = pix.shape[0]
    if pix.ndim != 2 or pix.shape[1] != 2:
        raise ValueError(
            f"pixel_coords must be (N, 2), got shape {pix.shape}."
        )
    if msk.shape != (N,):
        raise ValueError(
            f"valid must have shape ({N},), got {msk.shape}."
        )
    if d.shape != (N,):
        raise ValueError(
            f"depth must have shape ({N},), got {d.shape}."
        )

    out = img.astype(np.uint8).copy()

    valid_idx = np.where(msk)[0]
    if valid_idx.size == 0:
        return out

    d_valid = d[valid_idx]
    lo = float(d_valid.min()) if d_min is None else float(d_min)
    hi = float(d_valid.max()) if d_max is None else float(d_max)

    # Clamp d_min < d_max to avoid division issues.
    if hi - lo < 1e-12:
        hi = lo + 1e-12

    colours = colour_by_height(d_valid, z_min=lo, z_max=hi)  # (M, 3) uint8

    for k, idx in enumerate(valid_idx):
        u, v = pix[idx]
        col = int(round(u))
        row = int(round(v))
        r0 = max(row - radius, 0)
        r1 = min(row + radius + 1, H)
        c0 = max(col - radius, 0)
        c1 = min(col + radius + 1, W)
        if r0 < r1 and c0 < c1:
            out[r0:r1, c0:c1] = colours[k]

    return out


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def export_point_cloud_open3d(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> dict:
    """Serialise a point cloud to an Open3D-compatible dict.

    The returned dict mirrors the format produced by
    ``open3d.geometry.PointCloud.to_dict()`` / consumed by
    ``open3d.geometry.PointCloud.from_dict()``, so callers can pass it
    directly to Open3D without importing it here::

        import open3d as o3d

        d = export_point_cloud_open3d(points, colors)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(d["points"]))
        if d["colors"] is not None:
            pcd.colors = o3d.utility.Vector3dVector(np.array(d["colors"]))

    Args:
        points: ``(N, 3)`` float array of XYZ coordinates.
        colors: Optional ``(N, 3)`` array of per-point RGB colours.  ``uint8``
            values (0–255) are automatically normalised to ``[0, 1]`` as
            required by Open3D.

    Returns:
        Dict with keys:

        * ``"points"``: ``list`` of ``[x, y, z]`` float triplets.
        * ``"colors"``: ``list`` of ``[r, g, b]`` float triplets in ``[0, 1]``,
          or ``None`` when no colours were provided.

    Raises:
        ValueError: If *points* is not ``(N, 3)`` or *colors* has an
            incompatible shape.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(
            f"points must be an (N, 3) array, got shape {pts.shape}."
        )

    out: dict = {"points": pts.tolist(), "colors": None}

    if colors is not None:
        clr = np.asarray(colors)
        if clr.shape != (pts.shape[0], 3):
            raise ValueError(
                f"colors must have shape ({pts.shape[0]}, 3), got {clr.shape}."
            )
        if np.issubdtype(clr.dtype, np.unsignedinteger) or clr.max() > 1.0:
            # Convert uint8 [0, 255] → float [0, 1].
            clr = clr.astype(float) / 255.0
        out["colors"] = clr.tolist()

    return out


def export_trajectory_rviz(
    frame_poses: list,
    *,
    frame_id: str = "map",
    namespace: str = "trajectory",
    marker_id: int = 0,
    color: Tuple[float, float, float, float] = (1.0, 0.5, 0.0, 1.0),
    scale: float = 0.3,
) -> List[dict]:
    """Serialise a trajectory to a list of RViz-compatible marker dicts.

    Each :class:`~sensor_transposition.frame_pose.FramePose` is converted to a
    ``visualization_msgs/Marker`` dict with ``type=SPHERE`` so that the
    trajectory can be published to RViz without introducing a ROS dependency
    in this module::

        import rospy
        from visualization_msgs.msg import Marker
        from std_msgs.msg import Header

        markers = export_trajectory_rviz(seq.poses)
        pub = rospy.Publisher("/trajectory", Marker, queue_size=10)
        for m in markers:
            msg = Marker()
            msg.header.frame_id = m["header"]["frame_id"]
            msg.ns = m["ns"]
            msg.id = m["id"]
            msg.type = m["type"]   # 2 = SPHERE
            msg.action = m["action"]
            msg.pose.position.x = m["pose"]["position"]["x"]
            # … etc.
            pub.publish(msg)

    Args:
        frame_poses: List of
            :class:`~sensor_transposition.frame_pose.FramePose` objects (or
            any objects that have ``timestamp``, ``translation``, and
            ``rotation`` attributes).
        frame_id: TF frame name used in the ``header.frame_id`` field.
            Default ``"map"``.
        namespace: Marker namespace.  Default ``"trajectory"``.
        marker_id: Starting marker ID; each pose increments this by one.
            Default ``0``.
        color: ``(r, g, b, a)`` float tuple in ``[0, 1]`` for the sphere
            colour.  Default orange ``(1.0, 0.5, 0.0, 1.0)``.
        scale: Diameter of each sphere marker in metres.  Default ``0.3``.

    Returns:
        List of dicts, one per pose, each representing a
        ``visualization_msgs/Marker`` message with ``type=2`` (SPHERE).
    """
    r, g, b, a = (float(c) for c in color)
    markers = []

    for i, pose in enumerate(frame_poses):
        tx, ty, tz = (float(v) for v in pose.translation[:3])
        qw, qx, qy, qz = (float(v) for v in pose.rotation[:4])

        markers.append(
            {
                "header": {
                    "frame_id": frame_id,
                    "stamp": float(pose.timestamp),
                },
                "ns": namespace,
                "id": marker_id + i,
                "type": 2,           # SPHERE
                "action": 0,         # ADD
                "pose": {
                    "position": {"x": tx, "y": ty, "z": tz},
                    "orientation": {"w": qw, "x": qx, "y": qy, "z": qz},
                },
                "scale": {"x": scale, "y": scale, "z": scale},
                "color": {"r": r, "g": g, "b": b, "a": a},
            }
        )

    return markers


# ---------------------------------------------------------------------------
# SensorFrameVisualiser
# ---------------------------------------------------------------------------


class SensorFrameVisualiser:
    """Per-frame container for synchronised multi-sensor observations.

    Holds up to four sensor streams for one synchronised time-step:

    * **Point cloud** (from LiDAR or accumulated :class:`~sensor_transposition.point_cloud_map.PointCloudMap`)
    * **Camera image** (H×W×3 ``uint8`` RGB)
    * **Trajectory** (M×2 or M×3 XY(Z) positions, e.g. from
      :class:`~sensor_transposition.frame_pose.FramePoseSequence`)
    * **Radar scan** (K×2 or K×3 XY(Z) positions in the ego frame)

    All streams are optional; unset streams are simply omitted from the
    rendered output.

    Typical usage::

        vis = SensorFrameVisualiser()
        vis.set_point_cloud(lidar_scan)
        vis.set_camera_image(camera_image)
        vis.set_trajectory(trajectory_xy)

        bev = vis.render_birdseye(resolution=0.10)
        cam = vis.render_camera_with_lidar(pixel_coords, valid, depth)
    """

    def __init__(self) -> None:
        self._points: Optional[np.ndarray] = None    # (N, 3)
        self._point_colors: Optional[np.ndarray] = None  # (N, 3) uint8
        self._image: Optional[np.ndarray] = None     # (H, W, 3) uint8
        self._trajectory: Optional[np.ndarray] = None  # (M, 2+) float
        self._radar: Optional[np.ndarray] = None     # (K, 2+) float

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "SensorFrameVisualiser":
        """Create a :class:`SensorFrameVisualiser` from a frame snapshot dict.

        Accepts a dictionary with any combination of the following optional
        keys — the same shape as a ``BagMessage`` payload when replaying bag
        files:

        * ``"point_cloud"`` – ``(N, 3)`` array-like of XYZ coordinates.
        * ``"point_colors"`` – optional ``(N, 3)`` array-like of RGB colours.
        * ``"camera_image"`` – ``(H, W, 3)`` uint8 array-like.
        * ``"trajectory"``   – ``(M, 2+)`` array-like of XY(Z) positions.
        * ``"radar_scan"``   – ``(K, 2+)`` array-like of XY(Z) radar returns.

        Keys that are absent or ``None`` are silently skipped.

        Args:
            data: Dictionary with optional sensor data keys (see above).

        Returns:
            A populated :class:`SensorFrameVisualiser` instance.

        Example::

            # Replay bag messages for visualisation:
            with BagReader("session.sbag") as bag:
                for msg in bag.read_messages():
                    vis = SensorFrameVisualiser.from_dict(msg.data)
                    bev = vis.render_birdseye(resolution=0.10)
        """
        vis = cls()
        if data.get("point_cloud") is not None:
            colors = data.get("point_colors")
            vis.set_point_cloud(
                np.asarray(data["point_cloud"], dtype=float),
                colors=np.asarray(colors) if colors is not None else None,
            )
        if data.get("camera_image") is not None:
            vis.set_camera_image(np.asarray(data["camera_image"]))
        if data.get("trajectory") is not None:
            vis.set_trajectory(np.asarray(data["trajectory"], dtype=float))
        if data.get("radar_scan") is not None:
            vis.set_radar_points(np.asarray(data["radar_scan"], dtype=float))
        return vis

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_point_cloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
    ) -> None:
        """Set the current point cloud.

        Args:
            points: ``(N, 3)`` float array of XYZ coordinates.
            colors: Optional ``(N, 3)`` uint8 or float colour array.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(
                f"points must be (N, 3), got {pts.shape}."
            )
        self._points = pts

        if colors is not None:
            clr = np.asarray(colors)
            if clr.shape != (pts.shape[0], 3):
                raise ValueError(
                    f"colors must have shape ({pts.shape[0]}, 3), got {clr.shape}."
                )
            if np.issubdtype(clr.dtype, np.floating):
                clr = np.clip(clr * 255.0, 0, 255).astype(np.uint8)
            else:
                clr = clr.astype(np.uint8)
            self._point_colors = clr
        else:
            self._point_colors = None

    def set_camera_image(self, image: np.ndarray) -> None:
        """Set the current camera image.

        Args:
            image: ``(H, W, 3)`` uint8 RGB image array.

        Raises:
            ValueError: If the array is not ``(H, W, 3)``.
        """
        img = np.asarray(image)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                f"image must be (H, W, 3), got {img.shape}."
            )
        self._image = img.astype(np.uint8)

    def set_trajectory(self, positions: np.ndarray) -> None:
        """Set the trajectory positions to overlay on BEV renders.

        Args:
            positions: ``(M, 2)`` or ``(M, 3)`` float array of XY(Z)
                world-frame positions in metres.

        Raises:
            ValueError: If the array does not have at least 2 columns.
        """
        pos = np.asarray(positions, dtype=float)
        if pos.ndim != 2 or pos.shape[1] < 2:
            raise ValueError(
                f"positions must be (M, ≥2), got {pos.shape}."
            )
        self._trajectory = pos

    def set_radar_points(self, points: np.ndarray) -> None:
        """Set the current radar scan in the ego frame.

        Args:
            points: ``(K, 2)`` or ``(K, 3)`` float array of XY(Z) positions.

        Raises:
            ValueError: If the array does not have at least 2 columns.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] < 2:
            raise ValueError(
                f"radar points must be (K, ≥2), got {pts.shape}."
            )
        self._radar = pts

    # ------------------------------------------------------------------
    # Renderers
    # ------------------------------------------------------------------

    def render_birdseye(
        self,
        *,
        resolution: float = 0.10,
        canvas_size: Optional[Tuple[int, int]] = None,
        origin: Optional[Tuple[float, float]] = None,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        background: Tuple[int, int, int] = (0, 0, 0),
        trajectory_color: Tuple[int, int, int] = (255, 127, 0),
        radar_color: Tuple[int, int, int] = (0, 200, 255),
        dot_radius: int = 2,
    ) -> np.ndarray:
        """Render a combined bird's-eye-view frame.

        Layers (bottom to top):

        1. Point cloud, coloured by z-height (or supplied colours).
        2. Radar scan (if set), drawn with *radar_color*.
        3. Trajectory (if set), drawn with *trajectory_color*.

        Args:
            resolution: Metres per pixel.  Default ``0.10``.
            canvas_size: ``(width, height)`` in pixels; auto-sized when
                ``None``.
            origin: ``(x_world, y_world)`` at the bottom-left pixel; auto-
                computed when ``None``.
            z_min: Lower bound of the point-cloud z-colour scale.
            z_max: Upper bound of the point-cloud z-colour scale.
            background: Background colour.  Default black.
            trajectory_color: Dot colour for trajectory positions.
            radar_color: Dot colour for radar returns.
            dot_radius: Half-side (pixels) of dots drawn for trajectory and
                radar returns.  Default ``2``.

        Returns:
            ``(H, W, 3)`` ``uint8`` RGB image.

        Raises:
            RuntimeError: If no sensor data has been set on this visualiser.
        """
        if self._points is None and self._trajectory is None and self._radar is None:
            raise RuntimeError(
                "No sensor data has been set.  Call set_point_cloud, "
                "set_trajectory, or set_radar_points before rendering."
            )

        # Determine world-space bounds across all available streams.
        ox, oy = _compute_origin(
            self._points, self._trajectory, self._radar, origin, resolution
        )

        # Determine canvas size from all streams.
        if canvas_size is None:
            W, H = _compute_canvas_size(
                self._points,
                self._trajectory,
                self._radar,
                ox,
                oy,
                resolution,
            )
        else:
            W, H = canvas_size

        canvas = np.full((H, W, 3), background, dtype=np.uint8)

        # Layer 1: point cloud.
        if self._points is not None and len(self._points) > 0:
            bev = render_birdseye_view(
                self._points,
                resolution=resolution,
                colors=self._point_colors,
                z_min=z_min,
                z_max=z_max,
                canvas_size=(W, H),
                origin=(ox, oy),
                background=background,
            )
            mask = np.any(bev != np.array(background, dtype=np.uint8), axis=2)
            canvas[mask] = bev[mask]

        # Layer 2: radar returns.
        if self._radar is not None and len(self._radar) > 0:
            _draw_dots(canvas, self._radar[:, :2], ox, oy, resolution, radar_color, dot_radius, H)

        # Layer 3: trajectory.
        if self._trajectory is not None and len(self._trajectory) > 0:
            _draw_dots(canvas, self._trajectory[:, :2], ox, oy, resolution, trajectory_color, dot_radius, H)

        return canvas

    def render_camera_with_lidar(
        self,
        pixel_coords: np.ndarray,
        valid: np.ndarray,
        depth: np.ndarray,
        *,
        d_min: Optional[float] = None,
        d_max: Optional[float] = None,
        radius: int = 2,
    ) -> np.ndarray:
        """Overlay LiDAR depth on the stored camera image.

        This is a thin wrapper around :func:`overlay_lidar_on_image` that uses
        the camera image previously supplied via :meth:`set_camera_image`.

        Args:
            pixel_coords: ``(N, 2)`` pixel coordinates from
                :func:`~sensor_transposition.lidar_camera.project_lidar_to_image`.
            valid: ``(N,)`` boolean validity mask.
            depth: ``(N,)`` depth / range values.
            d_min: Lower depth clamp for the colour scale.
            d_max: Upper depth clamp for the colour scale.
            radius: Half-side (pixels) of drawn dots.  Default ``2``.

        Returns:
            ``(H, W, 3)`` ``uint8`` RGB image with LiDAR overlay.

        Raises:
            RuntimeError: If no camera image has been set.
        """
        if self._image is None:
            raise RuntimeError(
                "No camera image has been set.  Call set_camera_image first."
            )
        return overlay_lidar_on_image(
            self._image,
            pixel_coords,
            valid,
            depth,
            d_min=d_min,
            d_max=d_max,
            radius=radius,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_canvas_size(W: int, H: int) -> None:
    """Raise ``ValueError`` if canvas dimensions are non-positive."""
    if W <= 0 or H <= 0:
        raise ValueError(
            f"canvas_size must have positive dimensions, got ({W}, {H})."
        )


def _compute_origin(
    points: Optional[np.ndarray],
    trajectory: Optional[np.ndarray],
    radar: Optional[np.ndarray],
    origin: Optional[Tuple[float, float]],
    resolution: float,
) -> Tuple[float, float]:
    """Compute the world-space bottom-left origin of a combined BEV canvas."""
    if origin is not None:
        return float(origin[0]), float(origin[1])

    x_mins, y_mins = [], []
    if points is not None and len(points) > 0:
        x_mins.append(float(points[:, 0].min()))
        y_mins.append(float(points[:, 1].min()))
    if trajectory is not None and len(trajectory) > 0:
        x_mins.append(float(trajectory[:, 0].min()))
        y_mins.append(float(trajectory[:, 1].min()))
    if radar is not None and len(radar) > 0:
        x_mins.append(float(radar[:, 0].min()))
        y_mins.append(float(radar[:, 1].min()))

    if not x_mins:
        return 0.0, 0.0
    return min(x_mins) - resolution, min(y_mins) - resolution


def _compute_canvas_size(
    points: Optional[np.ndarray],
    trajectory: Optional[np.ndarray],
    radar: Optional[np.ndarray],
    ox: float,
    oy: float,
    resolution: float,
) -> Tuple[int, int]:
    """Compute canvas ``(W, H)`` large enough to contain all streams."""
    x_maxs, y_maxs = [], []
    margin = resolution

    if points is not None and len(points) > 0:
        x_maxs.append(float(points[:, 0].max()))
        y_maxs.append(float(points[:, 1].max()))
    if trajectory is not None and len(trajectory) > 0:
        x_maxs.append(float(trajectory[:, 0].max()))
        y_maxs.append(float(trajectory[:, 1].max()))
    if radar is not None and len(radar) > 0:
        x_maxs.append(float(radar[:, 0].max()))
        y_maxs.append(float(radar[:, 1].max()))

    if not x_maxs:
        return 1, 1

    W = int(np.ceil((max(x_maxs) - ox + margin) / resolution)) + 1
    H = int(np.ceil((max(y_maxs) - oy + margin) / resolution)) + 1
    return max(W, 1), max(H, 1)


def _draw_dots(
    canvas: np.ndarray,
    xy: np.ndarray,
    ox: float,
    oy: float,
    resolution: float,
    color: Tuple[int, int, int],
    radius: int,
    H: int,
) -> None:
    """Draw filled square dots onto *canvas* in-place."""
    W = canvas.shape[1]
    for xi, yi in xy:
        col = int(np.floor((xi - ox) / resolution))
        row = H - 1 - int(np.floor((yi - oy) / resolution))
        r0 = max(row - radius, 0)
        r1 = min(row + radius + 1, H)
        c0 = max(col - radius, 0)
        c1 = min(col + radius + 1, W)
        if r0 < r1 and c0 < c1:
            canvas[r0:r1, c0:c1] = color
