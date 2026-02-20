"""Tests for the lidar_camera fusion module."""

import math

import numpy as np
import pytest

from sensor_transposition.lidar_camera import (
    project_lidar_to_image,
    color_lidar_from_image,
)
from sensor_transposition.sensor_collection import (
    CameraIntrinsics,
    Sensor,
    SensorCollection,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_T() -> np.ndarray:
    """4×4 identity transform (lidar and camera in the same frame)."""
    return np.eye(4, dtype=float)


def _simple_K(fx: float = 500.0, fy: float = 500.0,
               cx: float = 320.0, cy: float = 240.0) -> np.ndarray:
    return np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# project_lidar_to_image
# ---------------------------------------------------------------------------


class TestProjectLidarToImage:
    def setup_method(self):
        self.K = _simple_K()  # 640×480 image
        self.W, self.H = 640, 480
        self.T = _identity_T()

    # --- basic correctness ---

    def test_point_on_optical_axis_projects_to_principal_point(self):
        pts = np.array([[0.0, 0.0, 5.0]])  # straight ahead, Z=5
        pixels, mask = project_lidar_to_image(pts, self.T, self.K, self.W, self.H)
        assert mask[0]
        assert pixels[0, 0] == pytest.approx(320.0)
        assert pixels[0, 1] == pytest.approx(240.0)

    def test_known_projection(self):
        # X=1, Y=0.5, Z=2 → u = 500*(1/2)+320=570, v = 500*(0.5/2)+240=365
        pts = np.array([[1.0, 0.5, 2.0]])
        pixels, mask = project_lidar_to_image(pts, self.T, self.K, self.W, self.H)
        assert mask[0]
        assert pixels[0, 0] == pytest.approx(570.0)
        assert pixels[0, 1] == pytest.approx(365.0)

    def test_multiple_points(self):
        pts = np.array([
            [0.0, 0.0, 1.0],    # on-axis → (320, 240)
            [0.0, 0.0, 10.0],   # also on-axis
            [0.64, 0.0, 1.0],   # → u=320+500*0.64=640 → exactly at right edge
        ])
        pixels, mask = project_lidar_to_image(pts, self.T, self.K, self.W, self.H)
        assert mask[0]
        assert mask[1]
        # u=640 is out of bounds (< image_width means [0, 640))
        assert not mask[2]

    # --- behind-camera / out-of-bounds filtering ---

    def test_point_behind_camera_is_invalid(self):
        pts = np.array([[0.0, 0.0, -1.0]])
        _, mask = project_lidar_to_image(pts, self.T, self.K, self.W, self.H)
        assert not mask[0]

    def test_point_at_zero_depth_is_invalid(self):
        pts = np.array([[0.0, 0.0, 0.0]])
        _, mask = project_lidar_to_image(pts, self.T, self.K, self.W, self.H)
        assert not mask[0]

    def test_point_outside_image_right(self):
        # u = 500*(2/1)+320 = 1320 → out of 640 width
        pts = np.array([[2.0, 0.0, 1.0]])
        _, mask = project_lidar_to_image(pts, self.T, self.K, self.W, self.H)
        assert not mask[0]

    def test_point_outside_image_bottom(self):
        # v = 500*(2/1)+240 = 1240 → out of 480 height
        pts = np.array([[0.0, 2.0, 1.0]])
        _, mask = project_lidar_to_image(pts, self.T, self.K, self.W, self.H)
        assert not mask[0]

    def test_all_invalid_returns_zero_pixels(self):
        pts = np.array([[0.0, 0.0, -5.0], [0.0, 0.0, -1.0]])
        pixels, mask = project_lidar_to_image(pts, self.T, self.K, self.W, self.H)
        assert not np.any(mask)
        np.testing.assert_array_equal(pixels, np.zeros((2, 2)))

    # --- output shapes ---

    def test_output_shapes(self):
        pts = np.random.randn(10, 3)
        pts[:, 2] = np.abs(pts[:, 2]) + 0.1  # ensure positive depth
        pixels, mask = project_lidar_to_image(pts, self.T, self.K, self.W, self.H)
        assert pixels.shape == (10, 2)
        assert mask.shape == (10,)
        assert mask.dtype == bool

    # --- transform applied correctly ---

    def test_pure_translation_transform(self):
        """Shift the lidar 1 m forward relative to camera; point at origin becomes
        1 m in front of camera and should project to the principal point."""
        T = np.eye(4, dtype=float)
        T[2, 3] = 1.0  # lidar is 1 m behind camera along Z
        pts = np.array([[0.0, 0.0, 0.0]])  # lidar origin
        pixels, mask = project_lidar_to_image(pts, T, self.K, self.W, self.H)
        assert mask[0]
        assert pixels[0, 0] == pytest.approx(320.0)
        assert pixels[0, 1] == pytest.approx(240.0)

    # --- input validation ---

    def test_invalid_points_shape_raises(self):
        pts_bad = np.ones((5, 4))
        with pytest.raises(ValueError, match="points"):
            project_lidar_to_image(pts_bad, self.T, self.K, self.W, self.H)

    def test_invalid_transform_shape_raises(self):
        pts = np.ones((3, 3))
        T_bad = np.eye(3)
        with pytest.raises(ValueError, match="lidar_to_camera"):
            project_lidar_to_image(pts, T_bad, self.K, self.W, self.H)

    def test_invalid_camera_matrix_shape_raises(self):
        pts = np.ones((3, 3))
        K_bad = np.eye(4)
        with pytest.raises(ValueError, match="camera_matrix"):
            project_lidar_to_image(pts, self.T, K_bad, self.W, self.H)

    def test_invalid_image_width_raises(self):
        pts = np.ones((3, 3))
        with pytest.raises(ValueError, match="image_width"):
            project_lidar_to_image(pts, self.T, self.K, 0, self.H)

    def test_invalid_image_height_raises(self):
        pts = np.ones((3, 3))
        with pytest.raises(ValueError, match="image_height"):
            project_lidar_to_image(pts, self.T, self.K, self.W, -1)


# ---------------------------------------------------------------------------
# color_lidar_from_image
# ---------------------------------------------------------------------------


class TestColorLidarFromImage:
    def setup_method(self):
        self.K = _simple_K()
        self.T = _identity_T()
        # Solid-colour RGB image: 480×640, all pixels are (10, 20, 30)
        self.image_rgb = np.full((480, 640, 3), fill_value=[10, 20, 30], dtype=np.uint8)
        # Grayscale image
        self.image_gray = np.full((480, 640), fill_value=128, dtype=np.uint8)

    # --- basic colour sampling ---

    def test_valid_point_gets_correct_rgb_colour(self):
        pts = np.array([[0.0, 0.0, 5.0]])  # projects to principal point
        colors, mask = color_lidar_from_image(pts, self.T, self.K, self.image_rgb)
        assert mask[0]
        np.testing.assert_array_equal(colors[0], [10, 20, 30])

    def test_valid_point_gets_correct_gray_colour(self):
        pts = np.array([[0.0, 0.0, 5.0]])
        colors, mask = color_lidar_from_image(pts, self.T, self.K, self.image_gray)
        assert mask[0]
        assert colors[0] == 128

    def test_invalid_point_gets_zero_colour(self):
        pts = np.array([[0.0, 0.0, -1.0]])  # behind camera
        colors, mask = color_lidar_from_image(pts, self.T, self.K, self.image_rgb)
        assert not mask[0]
        np.testing.assert_array_equal(colors[0], [0, 0, 0])

    def test_mixed_valid_and_invalid_points(self):
        pts = np.array([
            [0.0, 0.0, 5.0],   # valid → projects to centre
            [0.0, 0.0, -1.0],  # behind camera → invalid
        ])
        colors, mask = color_lidar_from_image(pts, self.T, self.K, self.image_rgb)
        assert mask[0] and not mask[1]
        np.testing.assert_array_equal(colors[0], [10, 20, 30])
        np.testing.assert_array_equal(colors[1], [0, 0, 0])

    def test_correct_colour_at_specific_pixel(self):
        """Place a distinct colour at a known pixel and verify sampling."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[240, 320] = [255, 0, 128]  # at principal point pixel
        # Point that projects exactly to (320, 240)
        pts = np.array([[0.0, 0.0, 1.0]])
        colors, mask = color_lidar_from_image(pts, self.T, self.K, image)
        assert mask[0]
        np.testing.assert_array_equal(colors[0], [255, 0, 128])

    # --- output shapes ---

    def test_output_shape_rgb(self):
        pts = np.zeros((7, 3))
        pts[:, 2] = 1.0
        colors, mask = color_lidar_from_image(pts, self.T, self.K, self.image_rgb)
        assert colors.shape == (7, 3)
        assert mask.shape == (7,)

    def test_output_shape_grayscale(self):
        pts = np.zeros((5, 3))
        pts[:, 2] = 1.0
        colors, mask = color_lidar_from_image(pts, self.T, self.K, self.image_gray)
        assert colors.shape == (5,)

    def test_dtype_preserved(self):
        image_float = np.ones((480, 640, 3), dtype=np.float32) * 0.5
        pts = np.array([[0.0, 0.0, 1.0]])
        colors, _ = color_lidar_from_image(pts, self.T, self.K, image_float)
        assert colors.dtype == np.float32

    # --- input validation ---

    def test_invalid_image_ndim_raises(self):
        bad_image = np.ones((480, 640, 3, 2))
        pts = np.ones((3, 3))
        with pytest.raises(ValueError, match="image"):
            color_lidar_from_image(pts, self.T, self.K, bad_image)


# ---------------------------------------------------------------------------
# SensorCollection convenience methods
# ---------------------------------------------------------------------------


class TestSensorCollectionMethods:
    def _make_collection(self) -> SensorCollection:
        """Collection with a camera and a lidar both at identity extrinsics."""
        cam = Sensor(
            name="front_camera",
            sensor_type="camera",
            coordinate_system="RDF",
            translation=[0.0, 0.0, 0.0],
            rotation=[1.0, 0.0, 0.0, 0.0],
            intrinsics=CameraIntrinsics(
                fx=500.0, fy=500.0, cx=320.0, cy=240.0,
                width=640, height=480,
            ),
        )
        lidar = Sensor(
            name="front_lidar",
            sensor_type="lidar",
            coordinate_system="FLU",
            translation=[0.0, 0.0, 0.0],
            rotation=[1.0, 0.0, 0.0, 0.0],
        )
        return SensorCollection([cam, lidar])

    # --- project_lidar_to_image ---

    def test_project_returns_correct_shapes(self):
        col = self._make_collection()
        pts = np.array([[0.0, 0.0, 5.0], [1.0, 0.5, 3.0]])
        pixels, mask = col.project_lidar_to_image("front_lidar", "front_camera", pts)
        assert pixels.shape == (2, 2)
        assert mask.shape == (2,)

    def test_project_on_axis_point(self):
        col = self._make_collection()
        pts = np.array([[0.0, 0.0, 5.0]])
        pixels, mask = col.project_lidar_to_image("front_lidar", "front_camera", pts)
        assert mask[0]
        assert pixels[0, 0] == pytest.approx(320.0)
        assert pixels[0, 1] == pytest.approx(240.0)

    def test_project_raises_for_missing_intrinsics(self):
        cam = Sensor(
            name="nocam",
            sensor_type="camera",
            coordinate_system="RDF",
            translation=[0.0, 0.0, 0.0],
            rotation=[1.0, 0.0, 0.0, 0.0],
        )
        lidar = Sensor(
            name="lidar",
            sensor_type="lidar",
            coordinate_system="FLU",
            translation=[0.0, 0.0, 0.0],
            rotation=[1.0, 0.0, 0.0, 0.0],
        )
        col = SensorCollection([cam, lidar])
        with pytest.raises(ValueError, match="intrinsics"):
            col.project_lidar_to_image("lidar", "nocam", np.ones((1, 3)))

    def test_project_raises_for_unknown_sensor(self):
        col = self._make_collection()
        with pytest.raises(KeyError):
            col.project_lidar_to_image("nonexistent", "front_camera", np.ones((1, 3)))

    # --- color_lidar_from_image ---

    def test_color_returns_correct_shapes(self):
        col = self._make_collection()
        pts = np.array([[0.0, 0.0, 5.0]])
        image = np.full((480, 640, 3), 42, dtype=np.uint8)
        colors, mask = col.color_lidar_from_image("front_lidar", "front_camera", pts, image)
        assert colors.shape == (1, 3)
        assert mask.shape == (1,)

    def test_color_correct_value(self):
        col = self._make_collection()
        image = np.full((480, 640, 3), fill_value=[7, 8, 9], dtype=np.uint8)
        pts = np.array([[0.0, 0.0, 2.0]])
        colors, mask = col.color_lidar_from_image("front_lidar", "front_camera", pts, image)
        assert mask[0]
        np.testing.assert_array_equal(colors[0], [7, 8, 9])

    def test_color_raises_for_missing_intrinsics(self):
        cam = Sensor(
            name="nocam",
            sensor_type="camera",
            coordinate_system="RDF",
            translation=[0.0, 0.0, 0.0],
            rotation=[1.0, 0.0, 0.0, 0.0],
        )
        lidar = Sensor(
            name="lidar",
            sensor_type="lidar",
            coordinate_system="FLU",
            translation=[0.0, 0.0, 0.0],
            rotation=[1.0, 0.0, 0.0, 0.0],
        )
        col = SensorCollection([cam, lidar])
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="intrinsics"):
            col.color_lidar_from_image("lidar", "nocam", np.ones((1, 3)), image)
