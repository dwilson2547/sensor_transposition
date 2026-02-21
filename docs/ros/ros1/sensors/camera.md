# ROS 1 — Camera Integration

Guide to integrating cameras with ROS 1, covering USB, GigE, and depth cameras,
along with topic structure, calibration, and recording.

---

## Table of Contents

1. [Supported Cameras](#supported-cameras)
2. [Installation](#installation)
3. [Launching the Driver](#launching-the-driver)
4. [Topics & Message Types](#topics--message-types)
5. [Camera Calibration](#camera-calibration)
6. [Visualizing in rviz / rqt](#visualizing-in-rviz--rqt)
7. [Recording Data](#recording-data)
8. [Working with Images in Python](#working-with-images-in-python)
9. [Troubleshooting](#troubleshooting)
10. [Further Reading](#further-reading)

---

## Supported Cameras

| Camera Type | Package | Install Command |
|-------------|---------|-----------------|
| USB webcams / UVC | `usb_cam` | `sudo apt install ros-noetic-usb-cam` |
| FLIR / Point Grey (GigE, USB3) | `spinnaker_camera_driver` | Build from source — [flir_camera_driver](https://github.com/ros-drivers/flir_camera_driver) |
| Intel RealSense (D435, D455, L515) | `realsense2_camera` | `sudo apt install ros-noetic-realsense2-camera` |
| Stereo / depth (generic) | `stereo_image_proc` | `sudo apt install ros-noetic-stereo-image-proc` |
| ZED (Stereolabs) | `zed_ros_wrapper` | Build from source — [zed-ros-wrapper](https://github.com/stereolabs/zed-ros-wrapper) |

---

## Installation

### USB camera

```bash
sudo apt install ros-noetic-usb-cam
```

### Intel RealSense

```bash
sudo apt install ros-noetic-realsense2-camera
```

---

## Launching the Driver

### USB camera

```bash
roslaunch usb_cam usb_cam-test.launch
```

### Intel RealSense D435

```bash
roslaunch realsense2_camera rs_camera.launch
```

### FLIR camera

```bash
roslaunch spinnaker_camera_driver camera.launch camera_serial:=<SERIAL>
```

---

## Topics & Message Types

| Topic (typical) | Message Type | Content |
|-----------------|-------------|---------|
| `/usb_cam/image_raw` | `sensor_msgs/Image` | Raw RGB / grayscale image |
| `/camera/color/image_raw` | `sensor_msgs/Image` | Color image (RealSense) |
| `/camera/depth/image_rect_raw` | `sensor_msgs/Image` | Depth image (RealSense) |
| `/usb_cam/camera_info` | `sensor_msgs/CameraInfo` | Intrinsics & distortion coefficients |
| `/camera/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | JPEG-compressed image |

Inspect:

```bash
rostopic list | grep image
rostopic hz /usb_cam/image_raw
```

---

## Camera Calibration

Use the built-in ROS camera calibration tool with a checkerboard:

```bash
rosrun camera_calibration cameracalibrator.py \
  --size 8x6 --square 0.025 \
  image:=/usb_cam/image_raw camera:=/usb_cam
```

This produces a `camera_info` YAML file containing the **intrinsic matrix**
(`K`), **distortion coefficients** (`D`), and **projection matrix** (`P`).

> See also the
> [camera_intrinsics_guide](../../camera_intrinsics_guide.md) in this
> repo for a deeper dive on intrinsic parameter calculation.

---

## Visualizing in rviz / rqt

### rviz

1. Open: `rosrun rviz rviz`
2. Add → **By topic** → select the image topic → **Image** display.

### rqt_image_view

```bash
rosrun rqt_image_view rqt_image_view
```

Select the image topic from the dropdown to view a live feed.

---

## Recording Data

```bash
# Record image and camera info
rosbag record /usb_cam/image_raw /usb_cam/camera_info -O camera_session.bag

# Compressed variant (saves disk space)
rosbag record /usb_cam/image_raw/compressed -O camera_compressed.bag
```

> **Tip:** Raw images at 30 fps can produce very large bag files. Consider
> recording compressed images or reducing the frame rate.

---

## Working with Images in Python

```python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()

def callback(msg):
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    # Now use cv_image as a normal OpenCV numpy array
    print(f"Image shape: {cv_image.shape}")

rospy.init_node('camera_listener')
rospy.Subscriber('/usb_cam/image_raw', Image, callback)
rospy.spin()
```

Install `cv_bridge`:

```bash
sudo apt install ros-noetic-cv-bridge
```

---

## Troubleshooting

| Symptom | Possible Cause | Fix |
|---------|---------------|-----|
| `No video device found` | Wrong device path | Set `video_device:=/dev/video0` (or check `ls /dev/video*`) |
| Green / garbled image | Pixel format mismatch | Set `pixel_format` param to `yuyv` or `mjpeg` |
| Low frame rate | USB bandwidth | Use a USB 3.0 port; reduce resolution if needed |
| `cv_bridge` import error | Package not installed | `sudo apt install ros-noetic-cv-bridge` |

---

## Further Reading

- [usb_cam ROS wiki](http://wiki.ros.org/usb_cam)
- [realsense2_camera ROS wiki](http://wiki.ros.org/realsense2_camera)
- [camera_calibration ROS wiki](http://wiki.ros.org/camera_calibration)
- [cv_bridge Tutorials](http://wiki.ros.org/cv_bridge/Tutorials)
- [image_transport](http://wiki.ros.org/image_transport)
