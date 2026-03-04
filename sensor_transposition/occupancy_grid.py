"""
occupancy_grid.py

2-D occupancy grid map built from LiDAR point clouds.

An occupancy grid partitions the 2-D horizontal plane into a regular grid of
square cells and maintains a probability estimate for each cell being occupied
by an obstacle.  Cells start in the **unknown** state and are updated
incrementally as new sensor observations arrive.

The implementation follows the standard probabilistic occupancy-grid model
(Thrun, Burgard & Fox, *Probabilistic Robotics*, 2005):

* **Log-odds representation** – each cell stores a log-odds value that can be
  efficiently updated by addition.  The log-odds are clamped to
  ``[log_odds_min, log_odds_max]`` to prevent saturation.

* **Ray-casting (Bresenham)** – for each observed point, the cells along the
  line from the sensor origin to that point are marked as *free*, and the cell
  containing the point is marked as *occupied*.

* **Height-band filter** – only points whose Z coordinate falls within a
  configurable ``[z_min, z_max]`` range are projected onto the 2-D grid; this
  suppresses ground returns and overhead structure (e.g. ceilings) that would
  produce spurious obstacle cells.

Cell values returned by :meth:`OccupancyGrid.get_grid` follow the ROS
``nav_msgs/OccupancyGrid`` convention:

* ``-1``  – unknown / not yet observed
* ``0``   – free
* ``100`` – occupied

The map coordinate frame has its origin at the bottom-left corner of the grid
(cell ``(0, 0)``).  The *x*-axis points right (increasing column index) and
the *y*-axis points up (increasing row index).

The implementation is pure NumPy/SciPy and introduces no additional
dependencies beyond those already required by ``sensor_transposition``.

Typical use-case
----------------
::

    from sensor_transposition.occupancy_grid import OccupancyGrid
    import numpy as np

    grid = OccupancyGrid(
        resolution=0.10,          # 10 cm per cell
        width=200,                # 20 m in x
        height=200,               # 20 m in y
        origin=np.array([-10.0, -10.0]),   # world coords of cell (0,0)
    )

    for frame_pose, lidar_scan in zip(trajectory, scans):
        sensor_origin = frame_pose.transform[:3, 3]
        grid.insert_scan(lidar_scan, frame_pose.transform, sensor_origin)

    occupancy = grid.get_grid()   # (height, width) int8 array
    probs = grid.to_probability() # (height, width) float64 in [0, 1]
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Log-odds helpers
# ---------------------------------------------------------------------------

_LOG_ODDS_OCCUPIED: float = 0.85   # probability ≈ 70 % occupied
_LOG_ODDS_FREE: float = -0.4       # probability ≈ 60 % free (40 % occupied)
_LOG_ODDS_MIN: float = -5.0        # probability ≈ 0.007
_LOG_ODDS_MAX: float = 5.0         # probability ≈ 0.993
_LOG_ODDS_UNKNOWN: float = 0.0     # probability = 0.5


def _log_odds(p: float) -> float:
    """Convert a probability *p* to its log-odds representation."""
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return float(np.log(p / (1.0 - p)))


def _log_odds_to_probability(lo: np.ndarray) -> np.ndarray:
    """Convert an array of log-odds values to probabilities."""
    return 1.0 / (1.0 + np.exp(-lo))


# ---------------------------------------------------------------------------
# OccupancyGrid
# ---------------------------------------------------------------------------


class OccupancyGrid:
    """2-D probabilistic occupancy grid.

    The grid discretises the horizontal plane into square cells of side length
    *resolution*.  Each cell stores a log-odds occupancy estimate that is
    updated via ray-casting every time a new LiDAR scan is inserted.

    Args:
        resolution: Cell side length in metres.  Must be strictly positive.
        width: Number of cells along the *x* (column) axis.
        height: Number of cells along the *y* (row) axis.
        origin: ``(2,)`` array ``[x0, y0]`` giving the world-frame coordinates
            of the **bottom-left corner** of cell ``(row=0, col=0)`` in metres.
            Defaults to ``[0, 0]``.
        z_min: Minimum Z coordinate (metres) for points to be projected onto
            the grid.  Points below this value are ignored.  Defaults to
            ``-inf`` (accept all).
        z_max: Maximum Z coordinate (metres) for points to be projected onto
            the grid.  Points above this value are ignored.  Defaults to
            ``+inf`` (accept all).
        log_odds_hit: Log-odds increment applied to the cell containing an
            observed point.  Defaults to ``0.85``.
        log_odds_miss: Log-odds decrement applied to cells along the free-space
            ray (i.e. between the sensor and the hit cell).  Should be
            negative.  Defaults to ``-0.4``.
        log_odds_min: Lower clamp for log-odds values (prevents saturation).
            Defaults to ``-5.0``.
        log_odds_max: Upper clamp for log-odds values (prevents saturation).
            Defaults to ``5.0``.

    Raises:
        ValueError: If *resolution*, *width*, or *height* is not strictly
            positive, or if *origin* does not have shape ``(2,)``.
    """

    def __init__(
        self,
        resolution: float,
        width: int,
        height: int,
        origin: Optional[np.ndarray] = None,
        *,
        z_min: float = -np.inf,
        z_max: float = np.inf,
        log_odds_hit: float = _LOG_ODDS_OCCUPIED,
        log_odds_miss: float = _LOG_ODDS_FREE,
        log_odds_min: float = _LOG_ODDS_MIN,
        log_odds_max: float = _LOG_ODDS_MAX,
    ) -> None:
        if resolution <= 0.0:
            raise ValueError(
                f"resolution must be > 0, got {resolution}."
            )
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}.")
        if height < 1:
            raise ValueError(f"height must be >= 1, got {height}.")

        if origin is None:
            origin = np.zeros(2, dtype=float)
        else:
            origin = np.asarray(origin, dtype=float)
            if origin.shape != (2,):
                raise ValueError(
                    f"origin must have shape (2,), got {origin.shape}."
                )

        if log_odds_min >= log_odds_max:
            raise ValueError(
                "log_odds_min must be strictly less than log_odds_max."
            )

        self._resolution = float(resolution)
        self._width = int(width)
        self._height = int(height)
        self._origin = origin.copy()
        self._z_min = float(z_min)
        self._z_max = float(z_max)
        self._log_odds_hit = float(log_odds_hit)
        self._log_odds_miss = float(log_odds_miss)
        self._log_odds_min = float(log_odds_min)
        self._log_odds_max = float(log_odds_max)

        # Log-odds grid; 0.0 = unknown.
        self._log_odds: np.ndarray = np.zeros(
            (self._height, self._width), dtype=float
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def resolution(self) -> float:
        """Cell side length in metres."""
        return self._resolution

    @property
    def width(self) -> int:
        """Number of cells along the x (column) axis."""
        return self._width

    @property
    def height(self) -> int:
        """Number of cells along the y (row) axis."""
        return self._height

    @property
    def origin(self) -> np.ndarray:
        """World-frame coordinates of the bottom-left corner of cell (0, 0)."""
        return self._origin.copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert_scan(
        self,
        points: np.ndarray,
        ego_to_world: np.ndarray,
        sensor_origin: Optional[np.ndarray] = None,
    ) -> None:
        """Insert a LiDAR scan into the occupancy grid.

        Each point in *points* (given in the sensor body frame) is first
        transformed into the world frame using *ego_to_world*.  Points whose
        world-frame Z coordinate falls outside ``[z_min, z_max]`` are
        discarded.  The remaining points are projected onto the XY plane and
        used to update cell log-odds via ray-casting:

        * The cell containing the point is incremented by *log_odds_hit*.
        * All cells along the line from *sensor_origin* to (but not including)
          the hit cell are decremented by *log_odds_miss*.

        Args:
            points: ``(N, 3)`` float array of XYZ coordinates in the sensor
                body frame (metres).
            ego_to_world: 4×4 homogeneous transform mapping sensor-frame
                points to the world/map frame.
            sensor_origin: Optional ``(3,)`` or ``(2,)`` world-frame position
                of the sensor (used as the ray origin for free-space marking).
                If ``None``, the translation component of *ego_to_world* is
                used.

        Raises:
            ValueError: If *points* is not ``(N, 3)`` or *ego_to_world* is
                not 4×4.
        """
        pts = np.asarray(points, dtype=float)
        T = np.asarray(ego_to_world, dtype=float)

        _validate_points(pts)
        _validate_transform(T)

        # Determine sensor origin in world frame.
        if sensor_origin is None:
            origin_world = T[:3, 3]
        else:
            origin_world = np.asarray(sensor_origin, dtype=float).ravel()[:3]

        # Transform points to world frame.
        R = T[:3, :3]
        t = T[:3, 3]
        world_pts = (R @ pts.T).T + t   # (N, 3)

        # Filter by Z range.
        valid_mask = (world_pts[:, 2] >= self._z_min) & (
            world_pts[:, 2] <= self._z_max
        )
        world_pts = world_pts[valid_mask]

        if len(world_pts) == 0:
            return

        # Convert world-frame origin to grid cell indices.
        ox, oy = self._world_to_cell(origin_world[0], origin_world[1])

        for p in world_pts:
            px, py = self._world_to_cell(p[0], p[1])

            # Ray-cast: mark free cells along the ray.
            ray_cells = _bresenham(ox, oy, px, py)
            # All cells except the last (hit cell) are free.
            for cx, cy in ray_cells[:-1]:
                if 0 <= cy < self._height and 0 <= cx < self._width:
                    self._log_odds[cy, cx] = np.clip(
                        self._log_odds[cy, cx] + self._log_odds_miss,
                        self._log_odds_min,
                        self._log_odds_max,
                    )

            # Mark hit cell as occupied.
            if 0 <= py < self._height and 0 <= px < self._width:
                self._log_odds[py, px] = np.clip(
                    self._log_odds[py, px] + self._log_odds_hit,
                    self._log_odds_min,
                    self._log_odds_max,
                )

    def get_grid(self) -> np.ndarray:
        """Return the occupancy grid as a 2-D integer array.

        Values follow the ROS ``nav_msgs/OccupancyGrid`` convention:

        * ``-1``  – unknown (cell has not been observed)
        * ``0``   – free
        * ``100`` – occupied

        Returns:
            ``(height, width)`` ``int8`` array.
        """
        grid = np.full(
            (self._height, self._width), fill_value=-1, dtype=np.int8
        )
        # Cells with log-odds > 0 are occupied; < 0 are free; == 0 are unknown.
        occupied_mask = self._log_odds > 0.0
        free_mask = self._log_odds < 0.0

        grid[occupied_mask] = 100
        grid[free_mask] = 0

        return grid

    def to_probability(self) -> np.ndarray:
        """Return the occupancy probability for each cell.

        Converts the internal log-odds representation to probabilities in the
        range ``[0, 1]``.  Unknown cells (log-odds == 0) map to ``0.5``.

        Returns:
            ``(height, width)`` float64 array with values in ``[0, 1]``.
        """
        return _log_odds_to_probability(self._log_odds)

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world-frame coordinates to grid cell indices.

        Args:
            x: World-frame x coordinate in metres.
            y: World-frame y coordinate in metres.

        Returns:
            ``(col, row)`` integer cell indices.  Values may be outside the
            grid bounds if the world coordinate falls outside the map extent.
        """
        return self._world_to_cell(x, y)

    def cell_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Return the world-frame coordinates of the **centre** of a cell.

        Args:
            col: Column index (x direction).
            row: Row index (y direction).

        Returns:
            ``(x, y)`` world-frame coordinates of the cell centre in metres.
        """
        x = self._origin[0] + (col + 0.5) * self._resolution
        y = self._origin[1] + (row + 0.5) * self._resolution
        return float(x), float(y)

    def to_ros_int8(self) -> np.ndarray:
        """Return the occupancy grid as a 2-D ``int8`` array in ROS convention.

        This is a convenience alias for :meth:`get_grid` that makes the
        ``nav_msgs/OccupancyGrid`` compatibility explicit.  The returned array
        can be assigned directly to the ``data`` field of a ROS
        ``nav_msgs/OccupancyGrid`` message after flattening (``grid.ravel()``).

        Values follow the ROS convention:

        * ``-1``  – unknown / not yet observed
        * ``0``   – free
        * ``100`` – occupied

        Returns:
            ``(height, width)`` ``int8`` array.

        Example::

            ros_msg_data = grid.to_ros_int8().ravel()
        """
        return self.get_grid()

    def clear(self) -> None:
        """Reset all cells to the unknown state (log-odds = 0)."""
        self._log_odds[:] = _LOG_ODDS_UNKNOWN

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world-frame (x, y) to (col, row) cell indices."""
        col = int(np.floor((x - self._origin[0]) / self._resolution))
        row = int(np.floor((y - self._origin[1]) / self._resolution))
        return col, row


# ---------------------------------------------------------------------------
# Bresenham's line algorithm
# ---------------------------------------------------------------------------


def _bresenham(
    x0: int, y0: int, x1: int, y1: int
) -> list[Tuple[int, int]]:
    """Return a list of ``(col, row)`` cells along the line from (x0,y0) to (x1,y1).

    Uses Bresenham's line-drawing algorithm to enumerate all grid cells that
    the line segment passes through.  The endpoint ``(x1, y1)`` is included.

    Args:
        x0: Start column.
        y0: Start row.
        x1: End column.
        y1: End row.

    Returns:
        Ordered list of ``(col, row)`` tuples from start to end (inclusive).
    """
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 >= x0 else -1
    sy = 1 if y1 >= y0 else -1

    if dx >= dy:
        err = dx // 2
        while x != x1:
            cells.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy // 2
        while y != y1:
            cells.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    cells.append((x1, y1))
    return cells


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
