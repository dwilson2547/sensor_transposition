"""
sensor_transposition: A suite of tools for sensor coordinate system
transposition, camera intrinsic calibration, and LiDAR data parsing.
"""

from sensor_transposition.sensor_collection import SensorCollection, Sensor, CameraIntrinsics
from sensor_transposition.sensor_collection import GpsParameters, ImuParameters, RadarParameters
from sensor_transposition.transform import Transform
from sensor_transposition.frame_pose import FramePose, FramePoseSequence
from sensor_transposition.sync import (
    SensorSynchroniser,
    SensorSynchronizer,
    apply_time_offset,
    find_nearest_indices,
    interpolate_timestamps,
)
from sensor_transposition import camera_intrinsics
from sensor_transposition import calibration
from sensor_transposition.calibration import fit_plane, ransac_plane, calibrate_lidar_camera
from sensor_transposition import lidar
from sensor_transposition import lidar_camera
from sensor_transposition import gps
from sensor_transposition import imu
from sensor_transposition import radar
from sensor_transposition import sync
from sensor_transposition import visual_odometry
from sensor_transposition import loop_closure
from sensor_transposition import pose_graph
from sensor_transposition import point_cloud_map
from sensor_transposition import visualisation
from sensor_transposition import rosbag
from sensor_transposition import wheel_odometry
from sensor_transposition import sliding_window
from sensor_transposition import occupancy_grid
from sensor_transposition import voxel_map
from sensor_transposition import submap_manager
from sensor_transposition import slam_session
from sensor_transposition.slam_session import SLAMSession, LocalMap, LocalizationSession, merge_sessions
from sensor_transposition import ground_plane
from sensor_transposition.ground_plane import (
    height_threshold_segment,
    ransac_ground_plane,
    normal_based_segment,
)
from sensor_transposition import feature_detection
from sensor_transposition.feature_detection import (
    detect_harris_corners,
    compute_patch_descriptor,
    match_features,
)
from sensor_transposition import stereo
from sensor_transposition.stereo import (
    stereo_rectify,
    compute_disparity_sgbm,
    triangulate_stereo,
)
from sensor_transposition.loop_closure import (
    compute_image_descriptor,
    image_descriptor_distance,
    ImageDescriptor,
    ImageLoopClosureDatabase,
)
from sensor_transposition import exceptions
from sensor_transposition.exceptions import (
    SensorTranspositionError,
    SensorNotFoundError,
    BagError,
    CalibrationError,
)

__all__ = [
    "SensorCollection",
    "Sensor",
    "CameraIntrinsics",
    "GpsParameters",
    "ImuParameters",
    "RadarParameters",
    "Transform",
    "FramePose",
    "FramePoseSequence",
    "SensorSynchroniser",
    "SensorSynchronizer",
    "apply_time_offset",
    "find_nearest_indices",
    "interpolate_timestamps",
    "calibration",
    "fit_plane",
    "ransac_plane",
    "calibrate_lidar_camera",
    "camera_intrinsics",
    "lidar",
    "lidar_camera",
    "gps",
    "imu",
    "radar",
    "sync",
    "visual_odometry",
    "loop_closure",
    "pose_graph",
    "point_cloud_map",
    "visualisation",
    "rosbag",
    "wheel_odometry",
    "sliding_window",
    "occupancy_grid",
    "voxel_map",
    "submap_manager",
    "slam_session",
    "SLAMSession",
    "LocalMap",
    "LocalizationSession",
    "merge_sessions",
    "ground_plane",
    "height_threshold_segment",
    "ransac_ground_plane",
    "normal_based_segment",
    "feature_detection",
    "detect_harris_corners",
    "compute_patch_descriptor",
    "match_features",
    "stereo",
    "stereo_rectify",
    "compute_disparity_sgbm",
    "triangulate_stereo",
    "compute_image_descriptor",
    "image_descriptor_distance",
    "ImageDescriptor",
    "ImageLoopClosureDatabase",
    "exceptions",
    "SensorTranspositionError",
    "SensorNotFoundError",
    "BagError",
    "CalibrationError",
]
