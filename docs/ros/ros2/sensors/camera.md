# ROS 2 — Camera Integration

Guide to integrating cameras with ROS 2, covering USB, GigE, and depth cameras,
topic structure, calibration, and recording.

---

## Table of Contents

1. [Supported Cameras](#supported-cameras)
2. [Installation](#installation)
3. [Launching the Driver](#launching-the-driver)
4. [Topics & Message Types](#topics--message-types)
5. [Camera Calibration](#camera-calibration)
6. [Visualizing in rviz2 / rqt](#visualizing-in-rviz2--rqt)
7. [Recording Data](#recording-data)
8. [Working with Images in Python](#working-with-images-in-python)
9. [Troubleshooting](#troubleshooting)
10. [Further Reading](#further-reading)

---

## Supported Cameras

| Camera Type | Package | Install Command |
|-------------|---------|-----------------|
| USB webcams (V4L2) | `usb_cam` or `v4l2_camera` | `sudo apt install ros-humble-usb-cam` |
| FLIR / Point Grey (GigE, USB3) | `spinnaker_camera_driver` | Build from source — [ros-drivers/flir_camera_driver](https://github.com/ros-drivers/flir_camera_driver/tree/humble) |
| Intel RealSense (D435, D455, L515) | `realsense2_camera` | `sudo apt install ros-humble-realsense2-camera` |
| ZED (Stereolabs) | `zed_ros2_wrapper` | Build from source — [stereolabs/zed-ros2-wrapper](https://github.com/stereolabs/zed-ros2-wrapper) |
| OAK-D (Luxonis) | `depthai_ros` | Build from source — [luxonis/depthai-ros](https://github.com/luxonis/depthai-ros) |

---

## Installation

### USB camera (apt)

```bash
sudo apt install ros-humble-usb-cam
```

### Intel RealSense

```bash
sudo apt install ros-humble-realsense2-camera
```

---

## Launching the Driver

### USB camera

```bash
ros2 launch usb_cam camera.launch.py
```

### Intel RealSense D435

```bash
ros2 launch realsense2_camera rs_launch.py
```

### v4l2_camera (lightweight V4L2 driver)

```bash
sudo apt install ros-humble-v4l2-camera
ros2 run v4l2_camera v4l2_camera_node --ros-args -p video_device:=/dev/video0
```

---

## Topics & Message Types

| Topic (typical) | Message Type | Content |
|-----------------|-------------|---------|
| `/image_raw` | `sensor_msgs/msg/Image` | Raw RGB / grayscale image |
| `/camera/color/image_raw` | `sensor_msgs/msg/Image` | Color image (RealSense) |
| `/camera/depth/image_rect_raw` | `sensor_msgs/msg/Image` | Aligned depth image |
| `/camera_info` | `sensor_msgs/msg/CameraInfo` | Intrinsics & distortion |
| `/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | JPEG-compressed image |

Inspect:

```bash
ros2 topic list | grep image
ros2 topic hz /image_raw
ros2 topic info /image_raw --verbose   # check QoS settings
```

---

## Camera Calibration

Use the ROS 2 camera calibration tool with a checkerboard:

```bash
sudo apt install ros-humble-camera-calibration
ros2 run camera_calibration cameracalibrator \
  --size 8x6 --square 0.025 \
  --ros-args --remap image:=/image_raw --remap camera:=/camera
```

This produces a YAML file with the **intrinsic matrix** (`K`), **distortion
coefficients** (`D`), and **projection matrix** (`P`).

> See also the
> [camera_intrinsics_guide](../../camera_intrinsics_guide.md) in this
> repo for a deeper dive on intrinsic parameter calculation.

---

## Visualizing in rviz2 / rqt

### rviz2

1. Open: `rviz2`
2. Add → **By topic** → select the image topic → **Image** display.

### rqt_image_view

```bash
ros2 run rqt_image_view rqt_image_view
```

Select the image topic from the dropdown.

---

## Recording Data

```bash
# Record image + camera info
ros2 bag record /image_raw /camera_info -o camera_session

# Compressed (saves disk space)
ros2 bag record /image_raw/compressed -o camera_compressed
```

> **Tip:** Raw images at 30 fps generate large bags quickly. Use compressed
> transport or reduce resolution/frame rate for long sessions.

### image_transport

For efficient transport (compressed, theora, etc.):

```bash
sudo apt install ros-humble-image-transport-plugins
```

Subscribers can then receive compressed images transparently.

---

## Working with Images in Python

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraListener(Node):
    def __init__(self):
        super().__init__('camera_listener')
        self.bridge = CvBridge()
        self.create_subscription(Image, '/image_raw', self.cb, 10)

    def cb(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.get_logger().info(f'Image shape: {cv_image.shape}')

rclpy.init()
rclpy.spin(CameraListener())
```

Install `cv_bridge`:

```bash
sudo apt install ros-humble-cv-bridge
```

---

## Troubleshooting

| Symptom | Possible Cause | Fix |
|---------|---------------|-----|
| No image published | Wrong device path | Check `ls /dev/video*`; set `video_device` param |
| QoS mismatch (subscriber gets nothing) | Publisher uses different QoS | Use `ros2 topic info --verbose` to check, then match settings |
| Green / garbled image | Pixel format mismatch | Set `pixel_format` param to `yuyv` or `mjpeg` |
| Low frame rate | USB bandwidth | Use a USB 3.0 port; lower resolution |
| `cv_bridge` import error | Package not installed | `sudo apt install ros-humble-cv-bridge` |

---

## Further Reading

- [usb_cam ROS 2 (GitHub)](https://github.com/ros-drivers/usb_cam/tree/ros2)
- [v4l2_camera (GitHub)](https://gitlab.com/boldhearts/ros2_v4l2_camera)
- [realsense2_camera ROS 2](https://github.com/IntelRealSense/realsense-ros/tree/ros2-development)
- [camera_calibration ROS 2](https://docs.ros.org/en/humble/p/camera_calibration/)
- [cv_bridge ROS 2](https://docs.ros.org/en/humble/p/cv_bridge/)
- [image_transport](https://docs.ros.org/en/humble/p/image_transport/)
