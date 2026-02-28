"""Tests for visualisation module."""

import math

import numpy as np
import pytest

from sensor_transposition.visualisation import (
    SensorFrameVisualiser,
    _jet_colormap,
    colour_by_height,
    export_point_cloud_open3d,
    export_trajectory_rviz,
    overlay_lidar_on_image,
    render_birdseye_view,
    render_trajectory_birdseye,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grid_cloud(n: int = 100, seed: int = 0) -> np.ndarray:
    """Random (N, 3) float cloud in a 20 m cube."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-10.0, 10.0, (n, 3))


def _random_image(H: int = 100, W: int = 120, seed: int = 0) -> np.ndarray:
    """Random (H, W, 3) uint8 image."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (H, W, 3), dtype=np.uint8)


class _FakePose:
    """Minimal FramePose-like object for export tests."""

    def __init__(self, t, tx=0.0, ty=0.0, tz=0.0):
        self.timestamp = t
        self.translation = [tx, ty, tz]
        self.rotation = [1.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# colour_by_height / _jet_colormap
# ---------------------------------------------------------------------------


class TestColourByHeight:
    def test_returns_uint8_rgb(self):
        z = np.linspace(-5.0, 5.0, 50)
        clr = colour_by_height(z)
        assert clr.dtype == np.uint8
        assert clr.shape == (50, 3)

    def test_values_in_range(self):
        z = np.linspace(0.0, 1.0, 100)
        clr = colour_by_height(z)
        assert clr.min() >= 0
        assert clr.max() <= 255

    def test_z_min_z_max_clamps(self):
        z = np.array([0.0, 10.0])
        # Force all points above z_max to get the warm (red) colour.
        clr = colour_by_height(z, z_min=0.0, z_max=5.0)
        assert clr.shape == (2, 3)

    def test_z_min_equal_z_max_raises(self):
        with pytest.raises(ValueError, match="z_min"):
            colour_by_height(np.array([1.0, 2.0]), z_min=3.0, z_max=3.0)

    def test_z_min_greater_than_z_max_raises(self):
        with pytest.raises(ValueError, match="z_min"):
            colour_by_height(np.array([1.0, 2.0]), z_min=5.0, z_max=1.0)

    def test_constant_z_midpoint_color(self):
        """All-equal z-values should map to the midpoint colour without error."""
        z = np.full(10, 3.0)
        clr = colour_by_height(z)
        assert clr.shape == (10, 3)
        # All rows identical
        assert np.all(clr == clr[0])


class TestJetColormap:
    def test_blue_at_zero(self):
        clr = _jet_colormap(np.array([0.0]))
        # At t=0: r=0, g=0, b=255
        assert clr[0, 2] == 255   # blue channel maximal

    def test_red_at_one(self):
        clr = _jet_colormap(np.array([1.0]))
        assert clr[0, 0] == 255   # red channel maximal

    def test_shape_preserved(self):
        t = np.linspace(0.0, 1.0, 64)
        clr = _jet_colormap(t)
        assert clr.shape == (64, 3)


# ---------------------------------------------------------------------------
# render_birdseye_view
# ---------------------------------------------------------------------------


class TestRenderBirdseyeView:
    def test_returns_uint8_rgb(self):
        pts = _grid_cloud(200)
        img = render_birdseye_view(pts)
        assert img.dtype == np.uint8
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_auto_canvas_not_empty(self):
        pts = _grid_cloud(200)
        img = render_birdseye_view(pts)
        assert img.shape[0] > 0 and img.shape[1] > 0

    def test_resolution_affects_canvas_size(self):
        pts = _grid_cloud(200, seed=1)
        img_coarse = render_birdseye_view(pts, resolution=1.0)
        img_fine = render_birdseye_view(pts, resolution=0.1)
        # Finer resolution → larger canvas.
        assert img_fine.shape[0] > img_coarse.shape[0]
        assert img_fine.shape[1] > img_coarse.shape[1]

    def test_supplied_canvas_size_honoured(self):
        pts = _grid_cloud(200, seed=2)
        img = render_birdseye_view(pts, canvas_size=(80, 60))
        assert img.shape == (60, 80, 3)

    def test_supplied_colors_used(self):
        pts = np.array([[0.0, 0.0, 0.0]])
        clr = np.array([[255, 0, 0]], dtype=np.uint8)
        img = render_birdseye_view(pts, colors=clr, resolution=1.0, canvas_size=(5, 5), origin=(-2.0, -2.0))
        # At least one red pixel should exist in the image.
        assert np.any(img[:, :, 0] == 255)

    def test_float_colors_converted(self):
        pts = np.array([[0.0, 0.0, 0.0]])
        clr = np.array([[1.0, 0.0, 0.0]], dtype=float)
        img = render_birdseye_view(pts, colors=clr, resolution=1.0, canvas_size=(5, 5), origin=(-2.0, -2.0))
        assert np.any(img[:, :, 0] == 255)

    def test_empty_points_returns_background(self):
        img = render_birdseye_view(np.empty((0, 3)), canvas_size=(10, 10))
        assert np.all(img == 0)

    def test_invalid_points_shape_raises(self):
        with pytest.raises(ValueError, match="\\(N, 3\\)"):
            render_birdseye_view(np.zeros((10, 2)))

    def test_invalid_resolution_raises(self):
        with pytest.raises(ValueError, match="resolution"):
            render_birdseye_view(np.zeros((5, 3)), resolution=0.0)

    def test_invalid_canvas_size_raises(self):
        with pytest.raises(ValueError, match="canvas_size"):
            render_birdseye_view(np.zeros((5, 3)), canvas_size=(0, 10))

    def test_color_shape_mismatch_raises(self):
        pts = _grid_cloud(10, seed=3)
        bad_clr = np.zeros((5, 3), dtype=np.uint8)  # wrong N
        with pytest.raises(ValueError, match="colors"):
            render_birdseye_view(pts, colors=bad_clr)

    def test_background_color(self):
        pts = np.array([[100.0, 100.0, 0.0]])  # single distant point
        bg = (50, 100, 150)
        img = render_birdseye_view(pts, resolution=1.0, canvas_size=(5, 5), origin=(0.0, 0.0), background=bg)
        # Ensure background is applied somewhere.
        assert np.any(np.all(img == np.array(bg, dtype=np.uint8), axis=2))

    def test_single_point_marks_canvas(self):
        pts = np.array([[0.0, 0.0, 1.0]])
        img = render_birdseye_view(pts, resolution=1.0, canvas_size=(5, 5), origin=(-2.0, -2.0))
        # The canvas must not be all-black.
        assert not np.all(img == 0)


# ---------------------------------------------------------------------------
# render_trajectory_birdseye
# ---------------------------------------------------------------------------


class TestRenderTrajectoryBirdseye:
    def test_returns_uint8_rgb(self):
        pos = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        img = render_trajectory_birdseye(pos)
        assert img.dtype == np.uint8
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_canvas_contains_trajectory_color(self):
        pos = np.array([[0.0, 0.0], [1.0, 0.0]])
        color = (200, 50, 100)
        img = render_trajectory_birdseye(pos, color=color, resolution=0.5)
        assert np.any(np.all(img == np.array(color, dtype=np.uint8), axis=2))

    def test_canvas_size_honoured(self):
        pos = np.array([[0.0, 0.0], [1.0, 1.0]])
        img = render_trajectory_birdseye(pos, canvas_size=(50, 40))
        assert img.shape == (40, 50, 3)

    def test_invalid_positions_shape_raises(self):
        with pytest.raises(ValueError, match="M"):
            render_trajectory_birdseye(np.zeros((5,)))

    def test_1d_column_raises(self):
        with pytest.raises(ValueError, match="M"):
            render_trajectory_birdseye(np.zeros((5, 1)))

    def test_invalid_resolution_raises(self):
        with pytest.raises(ValueError, match="resolution"):
            render_trajectory_birdseye(np.zeros((3, 2)), resolution=-1.0)

    def test_3d_positions_accepted(self):
        pos = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
        img = render_trajectory_birdseye(pos)
        assert img.shape[2] == 3

    def test_dot_radius_zero(self):
        pos = np.array([[0.0, 0.0], [5.0, 5.0]])
        img = render_trajectory_birdseye(pos, dot_radius=0, resolution=1.0)
        assert img.shape[2] == 3


# ---------------------------------------------------------------------------
# overlay_lidar_on_image
# ---------------------------------------------------------------------------


class TestOverlayLidarOnImage:
    def _make_inputs(self, H=60, W=80, N=50, seed=0):
        rng = np.random.default_rng(seed)
        image = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        pixel_coords = rng.uniform(5.0, min(H, W) - 5.0, (N, 2))
        valid = rng.integers(0, 2, N, dtype=bool)
        depth = rng.uniform(1.0, 50.0, N)
        return image, pixel_coords, valid, depth

    def test_returns_same_shape(self):
        image, pix, valid, depth = self._make_inputs()
        out = overlay_lidar_on_image(image, pix, valid, depth)
        assert out.shape == image.shape
        assert out.dtype == np.uint8

    def test_does_not_modify_input(self):
        image, pix, valid, depth = self._make_inputs()
        original = image.copy()
        overlay_lidar_on_image(image, pix, valid, depth)
        np.testing.assert_array_equal(image, original)

    def test_valid_points_change_image(self):
        image, pix, valid, depth = self._make_inputs(seed=1)
        valid[:] = True
        out = overlay_lidar_on_image(image, pix, valid, depth)
        assert not np.array_equal(out, image)

    def test_no_valid_points_image_unchanged(self):
        image, pix, valid, depth = self._make_inputs(seed=2)
        valid[:] = False
        out = overlay_lidar_on_image(image, pix, valid, depth)
        np.testing.assert_array_equal(out, image)

    def test_invalid_image_shape_raises(self):
        with pytest.raises(ValueError, match="image"):
            overlay_lidar_on_image(
                np.zeros((60, 80)),  # missing channel dim
                np.zeros((5, 2)),
                np.ones(5, dtype=bool),
                np.ones(5),
            )

    def test_pixel_coords_shape_mismatch_raises(self):
        image = _random_image()
        with pytest.raises(ValueError, match="pixel_coords"):
            overlay_lidar_on_image(
                image,
                np.zeros((5, 3)),  # wrong second dim
                np.ones(5, dtype=bool),
                np.ones(5),
            )

    def test_valid_shape_mismatch_raises(self):
        image = _random_image()
        with pytest.raises(ValueError, match="valid"):
            overlay_lidar_on_image(
                image,
                np.zeros((5, 2)),
                np.ones(3, dtype=bool),  # wrong N
                np.ones(5),
            )

    def test_depth_shape_mismatch_raises(self):
        image = _random_image()
        with pytest.raises(ValueError, match="depth"):
            overlay_lidar_on_image(
                image,
                np.zeros((5, 2)),
                np.ones(5, dtype=bool),
                np.ones(3),  # wrong N
            )

    def test_explicit_d_min_d_max(self):
        image = _random_image(seed=3)
        N = 10
        pix = np.full((N, 2), 50.0)
        valid = np.ones(N, dtype=bool)
        depth = np.linspace(0.0, 100.0, N)
        out = overlay_lidar_on_image(image, pix, valid, depth, d_min=10.0, d_max=90.0)
        assert out.shape == image.shape


# ---------------------------------------------------------------------------
# export_point_cloud_open3d
# ---------------------------------------------------------------------------


class TestExportPointCloudOpen3d:
    def test_basic_structure(self):
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        d = export_point_cloud_open3d(pts)
        assert "points" in d
        assert "colors" in d
        assert d["colors"] is None
        assert len(d["points"]) == 2
        assert d["points"][0] == [1.0, 2.0, 3.0]

    def test_with_uint8_colors_normalised(self):
        pts = np.array([[0.0, 0.0, 0.0]])
        clr = np.array([[255, 0, 0]], dtype=np.uint8)
        d = export_point_cloud_open3d(pts, clr)
        assert d["colors"] is not None
        # Red channel should be normalised to 1.0
        assert abs(d["colors"][0][0] - 1.0) < 1e-6

    def test_with_float_colors_passthrough(self):
        pts = np.array([[0.0, 0.0, 0.0]])
        clr = np.array([[0.5, 0.5, 0.5]], dtype=float)
        d = export_point_cloud_open3d(pts, clr)
        assert d["colors"] is not None
        assert abs(d["colors"][0][0] - 0.5) < 1e-6

    def test_invalid_points_shape_raises(self):
        with pytest.raises(ValueError, match="\\(N, 3\\)"):
            export_point_cloud_open3d(np.zeros((5, 2)))

    def test_color_shape_mismatch_raises(self):
        pts = np.zeros((5, 3))
        with pytest.raises(ValueError, match="colors"):
            export_point_cloud_open3d(pts, np.zeros((3, 3)))

    def test_empty_points(self):
        d = export_point_cloud_open3d(np.empty((0, 3)))
        assert d["points"] == []
        assert d["colors"] is None


# ---------------------------------------------------------------------------
# export_trajectory_rviz
# ---------------------------------------------------------------------------


class TestExportTrajectoryRviz:
    def _poses(self, n: int = 5) -> list:
        return [_FakePose(t=float(i), tx=float(i), ty=0.0) for i in range(n)]

    def test_returns_list_of_dicts(self):
        markers = export_trajectory_rviz(self._poses(3))
        assert isinstance(markers, list)
        assert len(markers) == 3
        assert isinstance(markers[0], dict)

    def test_marker_has_required_keys(self):
        markers = export_trajectory_rviz(self._poses(1))
        m = markers[0]
        for key in ("header", "ns", "id", "type", "action", "pose", "scale", "color"):
            assert key in m

    def test_marker_type_is_sphere(self):
        markers = export_trajectory_rviz(self._poses(2))
        for m in markers:
            assert m["type"] == 2  # SPHERE

    def test_position_values(self):
        markers = export_trajectory_rviz(self._poses(3))
        assert markers[2]["pose"]["position"]["x"] == 2.0

    def test_frame_id_propagated(self):
        markers = export_trajectory_rviz(self._poses(2), frame_id="odom")
        for m in markers:
            assert m["header"]["frame_id"] == "odom"

    def test_custom_color(self):
        color = (0.0, 1.0, 0.0, 0.8)
        markers = export_trajectory_rviz(self._poses(1), color=color)
        c = markers[0]["color"]
        assert abs(c["g"] - 1.0) < 1e-6
        assert abs(c["a"] - 0.8) < 1e-6

    def test_marker_id_offset(self):
        markers = export_trajectory_rviz(self._poses(3), marker_id=10)
        for i, m in enumerate(markers):
            assert m["id"] == 10 + i

    def test_empty_poses_returns_empty_list(self):
        assert export_trajectory_rviz([]) == []


# ---------------------------------------------------------------------------
# SensorFrameVisualiser
# ---------------------------------------------------------------------------


class TestSensorFrameVisualiserSetters:
    def test_set_point_cloud_valid(self):
        vis = SensorFrameVisualiser()
        pts = _grid_cloud(50)
        vis.set_point_cloud(pts)
        assert vis._points is not None
        assert vis._point_colors is None

    def test_set_point_cloud_with_uint8_colors(self):
        vis = SensorFrameVisualiser()
        pts = _grid_cloud(10)
        clr = np.zeros((10, 3), dtype=np.uint8)
        vis.set_point_cloud(pts, clr)
        assert vis._point_colors is not None
        assert vis._point_colors.dtype == np.uint8

    def test_set_point_cloud_with_float_colors(self):
        vis = SensorFrameVisualiser()
        pts = _grid_cloud(10)
        clr = np.zeros((10, 3), dtype=float)
        vis.set_point_cloud(pts, clr)
        assert vis._point_colors.dtype == np.uint8

    def test_set_point_cloud_invalid_shape_raises(self):
        vis = SensorFrameVisualiser()
        with pytest.raises(ValueError, match="\\(N, 3\\)"):
            vis.set_point_cloud(np.zeros((5, 2)))

    def test_set_point_cloud_color_mismatch_raises(self):
        vis = SensorFrameVisualiser()
        pts = _grid_cloud(10)
        bad_clr = np.zeros((5, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="colors"):
            vis.set_point_cloud(pts, bad_clr)

    def test_set_camera_image_valid(self):
        vis = SensorFrameVisualiser()
        img = _random_image()
        vis.set_camera_image(img)
        assert vis._image is not None
        assert vis._image.dtype == np.uint8

    def test_set_camera_image_invalid_raises(self):
        vis = SensorFrameVisualiser()
        with pytest.raises(ValueError, match="image"):
            vis.set_camera_image(np.zeros((60, 80)))  # missing channel dim

    def test_set_trajectory_valid(self):
        vis = SensorFrameVisualiser()
        pos = np.array([[0.0, 0.0], [1.0, 1.0]])
        vis.set_trajectory(pos)
        assert vis._trajectory is not None

    def test_set_trajectory_invalid_raises(self):
        vis = SensorFrameVisualiser()
        with pytest.raises(ValueError, match="M"):
            vis.set_trajectory(np.zeros((5, 1)))

    def test_set_radar_points_valid(self):
        vis = SensorFrameVisualiser()
        vis.set_radar_points(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert vis._radar is not None

    def test_set_radar_invalid_raises(self):
        vis = SensorFrameVisualiser()
        with pytest.raises(ValueError, match="K"):
            vis.set_radar_points(np.zeros((3, 1)))


class TestSensorFrameVisualiserRenderBirdseye:
    def test_no_data_raises(self):
        vis = SensorFrameVisualiser()
        with pytest.raises(RuntimeError, match="No sensor data"):
            vis.render_birdseye()

    def test_with_point_cloud_only(self):
        vis = SensorFrameVisualiser()
        vis.set_point_cloud(_grid_cloud(100))
        img = vis.render_birdseye(resolution=0.5)
        assert img.dtype == np.uint8
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_with_trajectory_only(self):
        vis = SensorFrameVisualiser()
        pos = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        vis.set_trajectory(pos)
        img = vis.render_birdseye(resolution=0.5)
        assert img.shape[2] == 3

    def test_with_all_streams(self):
        vis = SensorFrameVisualiser()
        vis.set_point_cloud(_grid_cloud(80, seed=0))
        vis.set_trajectory(np.array([[0.0, 0.0], [5.0, 5.0]]))
        vis.set_radar_points(np.array([[1.0, 1.0], [2.0, 2.0]]))
        img = vis.render_birdseye(resolution=0.5)
        assert img.shape[2] == 3

    def test_canvas_size_respected(self):
        vis = SensorFrameVisualiser()
        vis.set_point_cloud(_grid_cloud(50))
        img = vis.render_birdseye(resolution=1.0, canvas_size=(30, 25))
        assert img.shape == (25, 30, 3)

    def test_with_coloured_point_cloud(self):
        vis = SensorFrameVisualiser()
        pts = _grid_cloud(30)
        clr = np.zeros((30, 3), dtype=np.uint8)
        clr[:, 0] = 200
        vis.set_point_cloud(pts, clr)
        img = vis.render_birdseye(resolution=1.0)
        assert img.shape[2] == 3


class TestSensorFrameVisualiserRenderCameraWithLidar:
    def test_no_image_raises(self):
        vis = SensorFrameVisualiser()
        with pytest.raises(RuntimeError, match="No camera image"):
            vis.render_camera_with_lidar(
                np.zeros((5, 2)), np.ones(5, dtype=bool), np.ones(5)
            )

    def test_returns_same_shape_as_image(self):
        vis = SensorFrameVisualiser()
        H, W = 60, 80
        img = _random_image(H, W, seed=5)
        vis.set_camera_image(img)
        N = 20
        pix = np.full((N, 2), 30.0)
        valid = np.ones(N, dtype=bool)
        depth = np.linspace(1.0, 20.0, N)
        out = vis.render_camera_with_lidar(pix, valid, depth)
        assert out.shape == (H, W, 3)
        assert out.dtype == np.uint8
