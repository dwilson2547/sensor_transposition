"""Tests for occupancy_grid: OccupancyGrid 2-D probabilistic map."""

import math

import numpy as np
import pytest

from sensor_transposition.occupancy_grid import OccupancyGrid, _bresenham


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity() -> np.ndarray:
    """Return a 4×4 identity transform."""
    return np.eye(4, dtype=float)


def _translation_tf(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    """Return a 4×4 pure-translation transform."""
    T = np.eye(4, dtype=float)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


def _make_grid(
    resolution: float = 0.5,
    width: int = 20,
    height: int = 20,
    origin: np.ndarray = None,
) -> OccupancyGrid:
    if origin is None:
        origin = np.array([-5.0, -5.0])
    return OccupancyGrid(resolution=resolution, width=width, height=height, origin=origin)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_grid_is_unknown(self):
        g = _make_grid()
        grid = g.get_grid()
        assert np.all(grid == -1)

    def test_shape_matches_params(self):
        g = OccupancyGrid(resolution=0.1, width=30, height=40)
        assert g.get_grid().shape == (40, 30)

    def test_resolution_stored(self):
        g = OccupancyGrid(resolution=0.25, width=10, height=10)
        assert g.resolution == 0.25

    def test_width_and_height_stored(self):
        g = OccupancyGrid(resolution=1.0, width=5, height=7)
        assert g.width == 5
        assert g.height == 7

    def test_default_origin_is_zero(self):
        g = OccupancyGrid(resolution=1.0, width=5, height=5)
        np.testing.assert_array_equal(g.origin, [0.0, 0.0])

    def test_custom_origin_stored(self):
        orig = np.array([-10.0, -20.0])
        g = OccupancyGrid(resolution=1.0, width=5, height=5, origin=orig)
        np.testing.assert_array_equal(g.origin, orig)

    def test_zero_resolution_raises(self):
        with pytest.raises(ValueError, match="resolution"):
            OccupancyGrid(resolution=0.0, width=10, height=10)

    def test_negative_resolution_raises(self):
        with pytest.raises(ValueError, match="resolution"):
            OccupancyGrid(resolution=-1.0, width=10, height=10)

    def test_zero_width_raises(self):
        with pytest.raises(ValueError, match="width"):
            OccupancyGrid(resolution=1.0, width=0, height=10)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError, match="height"):
            OccupancyGrid(resolution=1.0, width=10, height=0)

    def test_origin_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="origin"):
            OccupancyGrid(resolution=1.0, width=10, height=10, origin=np.zeros(3))

    def test_invalid_log_odds_bounds_raises(self):
        with pytest.raises(ValueError, match="log_odds_min"):
            OccupancyGrid(
                resolution=1.0, width=5, height=5,
                log_odds_min=5.0, log_odds_max=5.0
            )


# ---------------------------------------------------------------------------
# insert_scan — basic occupancy
# ---------------------------------------------------------------------------


class TestInsertScanOccupancy:
    def test_point_marks_cell_occupied(self):
        """A single point inside the grid should mark its cell as occupied."""
        g = OccupancyGrid(
            resolution=1.0, width=10, height=10,
            origin=np.zeros(2),
        )
        # Place a point at (0.5, 0.5, 0) → cell (0, 0).
        pts = np.array([[0.5, 0.5, 0.0]])
        sensor_origin = np.array([-5.0, 0.5, 0.0])  # off the grid → no ray update
        g.insert_scan(pts, _identity(), sensor_origin=sensor_origin)
        grid = g.get_grid()
        assert grid[0, 0] == 100

    def test_multiple_hits_on_same_cell(self):
        """Multiple hits on the same cell should keep it marked occupied."""
        g = OccupancyGrid(resolution=1.0, width=10, height=10, origin=np.zeros(2))
        pts = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
        sensor_origin = np.array([20.0, 20.0, 0.0])  # far away, ray off grid
        g.insert_scan(pts, _identity(), sensor_origin=sensor_origin)
        assert g.get_grid()[0, 0] == 100

    def test_point_outside_grid_does_not_crash(self):
        """Points that map to cells outside the grid bounds must be silently ignored.

        The occupied cell is outside the grid, so no cell is marked occupied.
        Ray cells within the grid may be marked free (correct behaviour).
        """
        g = OccupancyGrid(resolution=1.0, width=5, height=5, origin=np.zeros(2))
        pts = np.array([[100.0, 100.0, 0.0]])
        g.insert_scan(pts, _identity())   # should not raise
        # No cell should be marked occupied (the hit is outside the grid).
        assert np.all(g.get_grid() != 100)

    def test_ray_marks_free_cells(self):
        """Cells between the sensor and a hit point should be marked free."""
        g = OccupancyGrid(resolution=1.0, width=20, height=1, origin=np.zeros(2))
        # Sensor at x=0 (cell col=0), point at x=5.5 (cell col=5).
        sensor_origin = np.array([0.5, 0.5, 0.0])
        pts = np.array([[5.5, 0.5, 0.0]])
        g.insert_scan(pts, _identity(), sensor_origin=sensor_origin)
        grid = g.get_grid()
        # Cell col=5 should be occupied.
        assert grid[0, 5] == 100
        # Cell col=0 (sensor cell) should be free (it was ray-cast through).
        assert grid[0, 0] == 0

    def test_z_filter_rejects_out_of_range(self):
        """Points outside z_min/z_max should not update the grid."""
        g = OccupancyGrid(
            resolution=1.0, width=10, height=10, origin=np.zeros(2),
            z_min=0.0, z_max=2.0,
        )
        pts = np.array([[0.5, 0.5, 5.0]])  # z=5.0 outside [0, 2]
        g.insert_scan(pts, _identity())
        assert np.all(g.get_grid() == -1)

    def test_z_filter_accepts_in_range(self):
        """Points within z_min/z_max should update the grid."""
        g = OccupancyGrid(
            resolution=1.0, width=10, height=10, origin=np.zeros(2),
            z_min=0.0, z_max=2.0,
        )
        pts = np.array([[0.5, 0.5, 1.0]])  # z=1.0 within [0, 2]
        sensor_origin = np.array([20.0, 20.0, 0.0])
        g.insert_scan(pts, _identity(), sensor_origin=sensor_origin)
        assert g.get_grid()[0, 0] == 100

    def test_ego_to_world_transform_applied(self):
        """Points should be transformed into world frame before insertion."""
        g = OccupancyGrid(resolution=1.0, width=20, height=20, origin=np.zeros(2))
        # Point at (0, 0, 0) in body frame; shift by 5 in x → world (5, 0).
        pts = np.array([[0.0, 0.0, 0.0]])
        T = _translation_tf(tx=5.5, ty=0.5)
        sensor_origin = np.array([100.0, 100.0, 0.0])
        g.insert_scan(pts, T, sensor_origin=sensor_origin)
        grid = g.get_grid()
        assert grid[0, 5] == 100

    def test_sensor_origin_defaults_to_transform_translation(self):
        """When no sensor_origin is given, it should be taken from ego_to_world."""
        g = OccupancyGrid(resolution=1.0, width=20, height=1, origin=np.zeros(2))
        # Sensor at x=0.5 (via transform translation), point at x=5.5.
        T = _translation_tf(tx=0.5, ty=0.5)
        pts = np.array([[5.0, 0.0, 0.0]])  # body-frame offset → world x=5.5
        g.insert_scan(pts, T)
        grid = g.get_grid()
        assert grid[0, 5] == 100


# ---------------------------------------------------------------------------
# insert_scan — input validation
# ---------------------------------------------------------------------------


class TestInsertScanValidation:
    def test_wrong_points_shape_raises(self):
        g = _make_grid()
        with pytest.raises(ValueError, match="\\(N, 3\\)"):
            g.insert_scan(np.zeros((5, 2)), _identity())

    def test_1d_points_raises(self):
        g = _make_grid()
        with pytest.raises(ValueError, match="\\(N, 3\\)"):
            g.insert_scan(np.zeros(9), _identity())

    def test_empty_points_raises(self):
        g = _make_grid()
        with pytest.raises(ValueError, match="at least one point"):
            g.insert_scan(np.zeros((0, 3)), _identity())

    def test_wrong_transform_shape_raises(self):
        g = _make_grid()
        with pytest.raises(ValueError, match="4×4"):
            g.insert_scan(np.zeros((3, 3)), np.eye(3))


# ---------------------------------------------------------------------------
# get_grid and to_probability
# ---------------------------------------------------------------------------


class TestGetGrid:
    def test_unknown_cells_are_minus_one(self):
        g = OccupancyGrid(resolution=1.0, width=5, height=5)
        assert np.all(g.get_grid() == -1)

    def test_dtype_is_int8(self):
        g = OccupancyGrid(resolution=1.0, width=5, height=5)
        assert g.get_grid().dtype == np.int8

    def test_get_grid_returns_copy(self):
        g = OccupancyGrid(resolution=1.0, width=5, height=5)
        grid = g.get_grid()
        grid[:] = 99
        # Internal state should be unchanged.
        assert np.all(g.get_grid() == -1)

    def test_to_probability_unknown_is_half(self):
        g = OccupancyGrid(resolution=1.0, width=3, height=3)
        probs = g.to_probability()
        np.testing.assert_allclose(probs, 0.5, atol=1e-12)

    def test_to_probability_occupied_above_half(self):
        g = OccupancyGrid(resolution=1.0, width=10, height=10, origin=np.zeros(2))
        pts = np.array([[0.5, 0.5, 0.0]])
        sensor_origin = np.array([20.0, 20.0, 0.0])
        g.insert_scan(pts, _identity(), sensor_origin=sensor_origin)
        probs = g.to_probability()
        assert probs[0, 0] > 0.5

    def test_to_probability_free_below_half(self):
        g = OccupancyGrid(resolution=1.0, width=20, height=1, origin=np.zeros(2))
        sensor_origin = np.array([0.5, 0.5, 0.0])
        pts = np.array([[15.5, 0.5, 0.0]])
        g.insert_scan(pts, _identity(), sensor_origin=sensor_origin)
        probs = g.to_probability()
        # Cells between sensor and hit should have probability < 0.5.
        assert probs[0, 0] < 0.5

    def test_to_probability_shape(self):
        g = OccupancyGrid(resolution=1.0, width=8, height=6)
        assert g.to_probability().shape == (6, 8)


# ---------------------------------------------------------------------------
# world_to_cell and cell_to_world
# ---------------------------------------------------------------------------


class TestCoordinateConversion:
    def test_world_to_cell_origin(self):
        g = OccupancyGrid(resolution=1.0, width=10, height=10, origin=np.zeros(2))
        col, row = g.world_to_cell(0.5, 0.5)
        assert (col, row) == (0, 0)

    def test_world_to_cell_offset_origin(self):
        g = OccupancyGrid(
            resolution=1.0, width=10, height=10, origin=np.array([-5.0, -5.0])
        )
        col, row = g.world_to_cell(0.0, 0.0)
        assert (col, row) == (5, 5)

    def test_cell_to_world_origin_cell(self):
        g = OccupancyGrid(resolution=1.0, width=10, height=10, origin=np.zeros(2))
        x, y = g.cell_to_world(0, 0)
        assert x == pytest.approx(0.5)
        assert y == pytest.approx(0.5)

    def test_cell_to_world_roundtrip(self):
        g = OccupancyGrid(
            resolution=0.5, width=20, height=20, origin=np.array([-5.0, -5.0])
        )
        for col, row in [(0, 0), (5, 7), (19, 19)]:
            x, y = g.cell_to_world(col, row)
            c2, r2 = g.world_to_cell(x, y)
            assert (c2, r2) == (col, row)


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_resets_to_unknown(self):
        g = OccupancyGrid(resolution=1.0, width=10, height=10, origin=np.zeros(2))
        pts = np.array([[0.5, 0.5, 0.0]])
        sensor_origin = np.array([20.0, 20.0, 0.0])
        g.insert_scan(pts, _identity(), sensor_origin=sensor_origin)
        g.clear()
        assert np.all(g.get_grid() == -1)

    def test_can_insert_after_clear(self):
        g = OccupancyGrid(resolution=1.0, width=10, height=10, origin=np.zeros(2))
        pts = np.array([[0.5, 0.5, 0.0]])
        sensor_origin = np.array([20.0, 20.0, 0.0])
        g.insert_scan(pts, _identity(), sensor_origin=sensor_origin)
        g.clear()
        g.insert_scan(pts, _identity(), sensor_origin=sensor_origin)
        assert g.get_grid()[0, 0] == 100


# ---------------------------------------------------------------------------
# Log-odds clamping
# ---------------------------------------------------------------------------


class TestLogOddsClamping:
    def test_repeated_hits_capped_at_max(self):
        """Repeated observations should not push log-odds beyond max."""
        g = OccupancyGrid(
            resolution=1.0, width=5, height=5, origin=np.zeros(2),
            log_odds_max=5.0,
        )
        pts = np.array([[0.5, 0.5, 0.0]])
        sensor_origin = np.array([20.0, 20.0, 0.0])
        for _ in range(100):
            g.insert_scan(pts, _identity(), sensor_origin=sensor_origin)
        assert g._log_odds[0, 0] <= 5.0

    def test_repeated_misses_capped_at_min(self):
        """Repeated free-space rays should not push log-odds below min."""
        g = OccupancyGrid(
            resolution=1.0, width=20, height=1, origin=np.zeros(2),
            log_odds_min=-5.0,
        )
        sensor_origin = np.array([0.5, 0.5, 0.0])
        pts = np.array([[19.5, 0.5, 0.0]])
        for _ in range(100):
            g.insert_scan(pts, _identity(), sensor_origin=sensor_origin)
        # Cells along the ray (except the last) should be clamped.
        assert np.all(g._log_odds[0, :19] >= -5.0)


# ---------------------------------------------------------------------------
# Bresenham
# ---------------------------------------------------------------------------


class TestBresenham:
    def test_same_point(self):
        cells = _bresenham(3, 3, 3, 3)
        assert cells == [(3, 3)]

    def test_horizontal_line(self):
        cells = _bresenham(0, 0, 4, 0)
        assert cells == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]

    def test_vertical_line(self):
        cells = _bresenham(0, 0, 0, 3)
        assert cells == [(0, 0), (0, 1), (0, 2), (0, 3)]

    def test_diagonal_line(self):
        cells = _bresenham(0, 0, 2, 2)
        assert (0, 0) in cells
        assert (2, 2) in cells
        assert len(cells) >= 3

    def test_negative_direction(self):
        cells = _bresenham(4, 0, 0, 0)
        assert cells == [(4, 0), (3, 0), (2, 0), (1, 0), (0, 0)]

    def test_endpoint_always_included(self):
        for x1, y1 in [(5, 2), (0, 7), (3, 3), (1, 4)]:
            cells = _bresenham(0, 0, x1, y1)
            assert (x1, y1) in cells

    def test_startpoint_always_included(self):
        cells = _bresenham(2, 3, 7, 5)
        assert (2, 3) in cells


# ---------------------------------------------------------------------------
# Integration: multi-scan workflow
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_wall_detection(self):
        """Scans from multiple poses pointing at a wall should build up occupancy."""
        # 20 x 10 grid, 1 m resolution, origin at (0, 0).
        g = OccupancyGrid(resolution=1.0, width=20, height=10, origin=np.zeros(2))

        # Wall at x=10; sensor sweeping from x=0 to x=5.
        for sx in range(6):
            sensor_origin = np.array([sx + 0.5, 5.0, 0.0])
            pts = np.array([[9.5, 5.0, 0.0]])   # wall point
            T = _translation_tf(tx=0.0, ty=0.0)
            g.insert_scan(pts, T, sensor_origin=sensor_origin)

        grid = g.get_grid()
        # Wall cell (col=9, row=5) should be occupied.
        assert grid[5, 9] == 100
        # Free cells along some rays (e.g. col=3, row=5) should be free.
        assert grid[5, 3] == 0


# ---------------------------------------------------------------------------
# SparseOccupancyGrid
# ---------------------------------------------------------------------------

from sensor_transposition.occupancy_grid import SparseOccupancyGrid  # noqa: E402


def _make_sparse_grid(
    resolution: float = 0.5,
    width: int = 20,
    height: int = 20,
    origin: np.ndarray = None,
) -> SparseOccupancyGrid:
    if origin is None:
        origin = np.array([-5.0, -5.0])
    return SparseOccupancyGrid(resolution=resolution, width=width, height=height, origin=origin)


class TestSparseOccupancyGridInit:
    def test_default_grid_all_unknown(self):
        g = _make_sparse_grid()
        grid = g.get_grid()
        assert grid.shape == (20, 20)
        assert np.all(grid == -1)

    def test_shape_matches_params(self):
        g = SparseOccupancyGrid(resolution=0.1, width=30, height=40)
        assert g.get_grid().shape == (40, 30)

    def test_zero_resolution_raises(self):
        with pytest.raises(ValueError, match="resolution"):
            SparseOccupancyGrid(resolution=0.0, width=10, height=10)

    def test_zero_width_raises(self):
        with pytest.raises(ValueError, match="width"):
            SparseOccupancyGrid(resolution=1.0, width=0, height=10)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError, match="height"):
            SparseOccupancyGrid(resolution=1.0, width=10, height=0)

    def test_origin_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="origin"):
            SparseOccupancyGrid(resolution=1.0, width=10, height=10, origin=np.zeros(3))

    def test_invalid_log_odds_bounds_raises(self):
        with pytest.raises(ValueError, match="log_odds_min"):
            SparseOccupancyGrid(
                resolution=1.0, width=5, height=5,
                log_odds_min=5.0, log_odds_max=5.0
            )

    def test_properties(self):
        g = SparseOccupancyGrid(resolution=0.25, width=8, height=12)
        assert g.resolution == 0.25
        assert g.width == 8
        assert g.height == 12


class TestSparseOccupancyGridInsert:
    def test_hit_cell_becomes_occupied(self):
        g = SparseOccupancyGrid(resolution=1.0, width=10, height=10, origin=np.zeros(2))
        pts = np.array([[5.5, 5.5, 0.0]])
        T = np.eye(4)
        sensor_origin = np.array([0.5, 0.5, 0.0])
        g.insert_scan(pts, T, sensor_origin=sensor_origin)
        grid = g.get_grid()
        assert grid[5, 5] == 100

    def test_ray_cells_become_free(self):
        g = SparseOccupancyGrid(resolution=1.0, width=20, height=10, origin=np.zeros(2))
        pts = np.array([[9.5, 5.0, 0.0]])
        T = np.eye(4)
        sensor_origin = np.array([0.5, 5.0, 0.0])
        g.insert_scan(pts, T, sensor_origin=sensor_origin)
        grid = g.get_grid()
        assert grid[5, 9] == 100
        assert grid[5, 3] == 0

    def test_out_of_z_range_points_ignored(self):
        g = SparseOccupancyGrid(resolution=1.0, width=10, height=10, z_min=0.0, z_max=2.0)
        pts = np.array([[5.5, 5.5, 5.0]])
        T = np.eye(4)
        g.insert_scan(pts, T)
        assert np.all(g.get_grid() == -1)

    def test_invalid_points_raises(self):
        g = _make_sparse_grid()
        with pytest.raises(ValueError):
            g.insert_scan(np.zeros((3, 2)), np.eye(4))

    def test_invalid_transform_raises(self):
        g = _make_sparse_grid()
        with pytest.raises(ValueError):
            g.insert_scan(np.zeros((3, 3)), np.eye(3))


class TestSparseOccupancyGridMethods:
    def test_clear_resets_all_cells(self):
        g = SparseOccupancyGrid(resolution=1.0, width=10, height=10, origin=np.zeros(2))
        pts = np.array([[5.5, 5.5, 0.0]])
        g.insert_scan(pts, np.eye(4))
        g.clear()
        assert np.all(g.get_grid() == -1)
        assert len(g._cells) == 0

    def test_to_probability_unobserved_is_0_5(self):
        g = _make_sparse_grid()
        prob = g.to_probability()
        assert prob.shape == (20, 20)
        np.testing.assert_allclose(prob, 0.5)

    def test_to_probability_occupied_above_0_5(self):
        g = SparseOccupancyGrid(resolution=1.0, width=10, height=10, origin=np.zeros(2))
        pts = np.array([[5.5, 5.5, 0.0]])
        sensor_origin = np.array([0.5, 0.5, 0.0])
        for _ in range(10):
            g.insert_scan(pts, np.eye(4), sensor_origin=sensor_origin)
        prob = g.to_probability()
        assert prob[5, 5] > 0.5

    def test_world_to_cell_matches_origin(self):
        g = SparseOccupancyGrid(resolution=1.0, width=10, height=10, origin=np.zeros(2))
        col, row = g.world_to_cell(0.5, 0.5)
        assert col == 0
        assert row == 0

    def test_cell_to_world_returns_centre(self):
        g = SparseOccupancyGrid(resolution=1.0, width=10, height=10, origin=np.zeros(2))
        x, y = g.cell_to_world(2, 3)
        assert x == pytest.approx(2.5)
        assert y == pytest.approx(3.5)

    def test_to_ros_int8_same_as_get_grid(self):
        g = SparseOccupancyGrid(resolution=1.0, width=10, height=10, origin=np.zeros(2))
        pts = np.array([[5.5, 5.5, 0.0]])
        g.insert_scan(pts, np.eye(4))
        np.testing.assert_array_equal(g.to_ros_int8(), g.get_grid())

    def test_sparse_matches_dense_single_scan(self):
        """SparseOccupancyGrid and OccupancyGrid should produce identical grids."""
        origin = np.array([-5.0, -5.0])
        dense = OccupancyGrid(resolution=1.0, width=10, height=10, origin=origin)
        sparse = SparseOccupancyGrid(resolution=1.0, width=10, height=10, origin=origin)
        pts = np.array([[2.5, 2.5, 0.0]])
        sensor_origin = np.array([-4.5, -4.5, 0.0])
        T = np.eye(4)
        dense.insert_scan(pts, T, sensor_origin=sensor_origin)
        sparse.insert_scan(pts, T, sensor_origin=sensor_origin)
        np.testing.assert_array_equal(dense.get_grid(), sparse.get_grid())
