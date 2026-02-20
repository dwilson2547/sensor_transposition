# LiDAR Data Collection Guide

A practical how-to guide for capturing real point-cloud data from **Velodyne**,
**Ouster**, and **Livox** LiDAR sensors without ROS, and loading the results
with the `sensor_transposition` library.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Velodyne](#velodyne)
   - [Hardware Setup](#velodyne-hardware-setup)
   - [Installing Veloview](#installing-veloview)
   - [Capturing Data with Veloview](#capturing-data-with-veloview)
   - [Exporting to KITTI `.bin`](#exporting-to-kitti-bin-velodyne)
   - [Loading with sensor_transposition](#loading-velodyne-data)
4. [Ouster](#ouster)
   - [Hardware Setup](#ouster-hardware-setup)
   - [Installing the Ouster SDK](#installing-the-ouster-sdk)
   - [Capturing a PCAP Recording](#capturing-a-pcap-recording)
   - [Converting PCAP to `.bin`](#converting-pcap-to-bin)
   - [Loading with sensor_transposition](#loading-ouster-data)
5. [Livox](#livox)
   - [Hardware Setup](#livox-hardware-setup)
   - [Installing Livox Viewer](#installing-livox-viewer)
   - [Recording with Livox Viewer](#recording-with-livox-viewer)
   - [Loading with sensor_transposition](#loading-livox-data)
6. [Verifying Your Data](#verifying-your-data)
7. [Troubleshooting](#troubleshooting)

---

## Overview

All three sensor families output 3-D point clouds; they differ in their native
wire protocols and file formats:

| Sensor family | Native recording format | Format loaded by this library |
|---------------|------------------------|-------------------------------|
| Velodyne      | PCAP (UDP packets)     | KITTI `.bin` (float32 × 4)    |
| Ouster        | PCAP (UDP packets)     | KITTI `.bin` (float32 × 4 or × 8) |
| Livox         | LVX / LVX2             | `.lvx` / `.lvx2`              |

The guide takes you from plugging in the sensor to having a file on disk that
you can parse with a few lines of Python.

---

## Prerequisites

### Python environment

```bash
pip install sensor_transposition
```

Or, if you are working from the repository source:

```bash
pip install -e ".[dev]"
```

### Common networking requirements (Velodyne & Ouster)

Both Velodyne and Ouster sensors communicate over Ethernet using UDP
multicast/unicast packets.  Before you start:

1. Connect the sensor directly to a laptop NIC or to a network switch.
2. Assign a **static IP address** on the same subnet as the sensor
   (typically `192.168.1.x/24`).
3. Disable any firewall rules that block UDP on ports **2368–2369**
   (Velodyne) or **7502** (Ouster).

**Linux example – assign a static IP:**

```bash
# Replace eth0 with your actual interface name
sudo ip addr add 192.168.1.100/24 dev eth0
sudo ip link set eth0 up
```

**macOS example:**

```
System Preferences → Network → [your NIC] → Configure IPv4: Manually
IP Address : 192.168.1.100
Subnet Mask: 255.255.255.0
```

---

## Velodyne

### Velodyne Hardware Setup

1. Mount the sensor and connect power (+12 V, typically via the supplied cable
   gland or interface box).
2. Connect the RJ-45 Ethernet cable from the sensor to your computer's NIC.
3. Open a browser and navigate to **`http://192.168.1.201`** (Velodyne default
   IP) to confirm the sensor's web interface loads.  If the page does not load,
   check your NIC's IP address and subnet settings.

> **Tip:** VLP-16 and VLP-32C sensors default to `192.168.1.201`.
> HDL-32E and HDL-64E sensors may use `192.168.1.200`.  Consult the label on
> the bottom of your unit or the quick-start guide that ships with the sensor.

### Installing Veloview

[Veloview](https://www.paraview.org/veloview/) is Velodyne's free, open-source
desktop visualiser and recorder.

| Platform | Download |
|----------|----------|
| Windows  | Installer from the [Kitware release page](https://github.com/Kitware/VeloView/releases) |
| Linux    | AppImage from the same release page |
| macOS    | DMG from the same release page |

After installing, launch Veloview.  From the toolbar select
**Sensor → Connect** and choose your sensor model from the drop-down.  You
should see the live point cloud spinning in the 3-D viewport within a few
seconds.

### Capturing Data with Veloview

1. From the menu choose **File → Start Recording**.
2. Select an output directory and filename (e.g. `capture_001.pcap`).
3. Click **OK** – a red recording indicator appears.
4. Drive/move the sensor through the scene you want to capture.
5. When done, choose **File → Stop Recording**.

The result is a standard `.pcap` file containing the raw UDP packets from the
sensor.

### Exporting to KITTI `.bin` (Velodyne) {#exporting-to-kitti-bin-velodyne}

The `sensor_transposition` library's `VelodyneParser` reads **KITTI-format
binary files**: each point is four consecutive `float32` values
`(x, y, z, intensity)`.  Use the open-source
[`pcl_ros` conversion utility](https://pointclouds.org/) **or** the small
Python snippet below to convert your PCAP to `.bin` files, one file per frame:

```python
"""
convert_velodyne_pcap.py
Convert a Velodyne PCAP recording to per-frame KITTI .bin files.

Requires: velodyne_decoder  (pip install velodyne-decoder)
"""
import velodyne_decoder as vd
import numpy as np
from pathlib import Path

PCAP_FILE  = "capture_001.pcap"   # path to your PCAP
MODEL      = "VLP-16"             # sensor model: VLP-16, VLP-32C, HDL-32E, HDL-64E
OUT_DIR    = Path("velodyne_frames")

OUT_DIR.mkdir(exist_ok=True)

config = vd.Config(model=MODEL)
for frame_idx, (stamp, points) in enumerate(vd.read_pcap(PCAP_FILE, config)):
    # points is (N, 4+) numpy array; first 4 cols are x, y, z, intensity
    data = points[:, :4].astype(np.float32)
    out_path = OUT_DIR / f"frame_{frame_idx:06d}.bin"
    data.tofile(out_path)
    print(f"Wrote {len(data)} points → {out_path}")
```

Install the dependency with:

```bash
pip install velodyne-decoder
```

### Loading Velodyne Data

```python
from sensor_transposition.lidar.velodyne import VelodyneParser

parser = VelodyneParser("velodyne_frames/frame_000000.bin")

# Structured numpy array – fields: x, y, z, intensity
cloud = parser.read()
print(f"Points: {len(cloud)}")
print(f"First point: x={cloud['x'][0]:.3f}, y={cloud['y'][0]:.3f}, z={cloud['z'][0]:.3f}")

# Plain (N, 3) array – coordinates only
xyz = parser.xyz()

# Plain (N, 4) array – x, y, z, intensity
xyz_i = parser.xyz_intensity()
```

---

## Ouster

### Ouster Hardware Setup

1. Connect power (12–24 V DC) and Ethernet to the sensor.
2. Assign a static IP to your NIC on the same `/24` subnet as the sensor.
   Ouster sensors default to DHCP on first boot; if you do not have a DHCP
   server available, use the Ouster Studio or `curl` commands to assign a
   static IP to the sensor itself.
3. Verify connectivity:

   ```bash
   ping <SENSOR_IP>   # use the link-local address shown on the sensor label
   ```

4. Open the sensor's HTTP configuration page to confirm the sensor type and
   firmware version:

   ```bash
   curl http://<SENSOR_IP>/api/v1/sensor/metadata | python3 -m json.tool
   ```

> **Ouster default IP ranges:**  
> OS0, OS1, OS2, OSDome all start with a link-local address (`169.254.x.x`)
> that is printed on the sensor housing.  Use Ouster Studio or the SDK CLI to
> reassign to a static routable address when needed.

### Installing the Ouster SDK

```bash
pip install ouster-sdk
```

Verify the install:

```bash
python -c "import ouster.sdk; print(ouster.sdk.__version__)"
```

### Capturing a PCAP Recording

The Ouster SDK ships a command-line recorder.  Point it at your sensor to
start capturing:

```bash
python -m ouster.sdk.examples.record \
    --hostname <SENSOR_IP> \
    --lidar-port 7502 \
    --output capture_ouster.pcap \
    --meta   capture_ouster.json
```

The `--meta` flag saves a sidecar JSON file with sensor calibration and
configuration, which is required for accurate point de-projection.

Let the capture run for as long as needed, then press **Ctrl-C** to stop.

### Converting PCAP to `.bin`

```python
"""
convert_ouster_pcap.py
Convert an Ouster PCAP + metadata JSON to per-frame KITTI .bin files.

Requires: ouster-sdk  (pip install ouster-sdk)
"""
import numpy as np
from pathlib import Path
from ouster.sdk import open_source
from ouster.sdk import client

PCAP_FILE = "capture_ouster.pcap"
META_FILE = "capture_ouster.json"
OUT_DIR   = Path("ouster_frames")

OUT_DIR.mkdir(exist_ok=True)

with open(META_FILE) as f:
    metadata = client.SensorInfo(f.read())

source = open_source(PCAP_FILE, meta=[metadata])
xyzlut = client.XYZLut(metadata)

for frame_idx, scan in enumerate(source):
    if not isinstance(scan, client.LidarScan):
        continue
    xyz    = xyzlut(scan)                           # (H, W, 3)  metres
    signal = scan.field(client.ChanField.SIGNAL)    # (H, W)     raw signal

    # Flatten to (N, 4) and drop zero-range returns
    xyz_flat    = xyz.reshape(-1, 3)
    signal_flat = signal.reshape(-1, 1).astype(np.float32)
    mask        = np.linalg.norm(xyz_flat, axis=1) > 0
    data        = np.hstack([xyz_flat[mask], signal_flat[mask]])

    out_path = OUT_DIR / f"frame_{frame_idx:06d}.bin"
    data.astype(np.float32).tofile(out_path)
    print(f"Wrote {mask.sum()} points → {out_path}")
```

> **8-column extended format:** The `OusterParser` also supports an 8-column
> variant `(x, y, z, intensity, t, reflectivity, ring, ambient)`.  To produce
> this file, replace the `data` assembly above with:
>
> ```python
> t_flat      = scan.field(client.ChanField.TIMESTAMP).reshape(-1, 1).astype(np.float32)
> refl_flat   = scan.field(client.ChanField.REFLECTIVITY).reshape(-1, 1).astype(np.float32)
> ring_flat   = np.tile(np.arange(metadata.format.pixels_per_column),
>                       metadata.format.columns_per_frame).reshape(-1, 1).astype(np.float32)
> ambient_flat = scan.field(client.ChanField.NEAR_IR).reshape(-1, 1).astype(np.float32)
> data = np.hstack([xyz_flat[mask], signal_flat[mask], t_flat[mask],
>                   refl_flat[mask], ring_flat[mask], ambient_flat[mask]])
> ```

### Loading Ouster Data

```python
from sensor_transposition.lidar.ouster import OusterParser

parser = OusterParser("ouster_frames/frame_000000.bin")

# Auto-detects 4-column or 8-column format
cloud = parser.read()
print(f"Points : {len(cloud)}")
print(f"Fields : {cloud.dtype.names}")

xyz   = parser.xyz()          # (N, 3)
xyz_i = parser.xyz_intensity() # (N, 4)
```

---

## Livox

### Livox Hardware Setup

1. Connect the sensor to power (12–24 V DC, depending on model).
2. Connect the Ethernet cable to your computer.
3. Livox sensors ship with a default static IP in the `192.168.1.0/24` range
   (the exact address depends on the model — check the quick-start card).
   Set your NIC to a static IP on the same `/24` subnet, e.g. `192.168.1.50`.
4. Verify connectivity:

   ```bash
   ping <SENSOR_IP>   # replace with the IP shown on the sensor label
   ```

> **Model-specific default IPs:**
>
> | Model           | Default sensor IP  |
> |-----------------|--------------------|
> | Mid-40 / Mid-70 | `192.168.1.120`    |
> | Horizon         | `192.168.1.100`    |
> | Tele-15         | `192.168.1.110`    |
> | Mid-360         | see label          |
> | HAP / Avia      | see label          |

### Installing Livox Viewer

[Livox Viewer](https://www.livoxtech.com/downloads) is Livox's free desktop
application for visualising and recording data.  Download the installer for
your platform from the official website and follow the setup wizard.

After launching Livox Viewer, your sensor should appear automatically in the
device list on the left panel.  Click **Connect** to stream the live point
cloud.

> **Livox Viewer 2** is required for Mid-360, HAP, and Avia sensors (these
> use the LVX2 format).  Older sensors (Mid-40, Horizon, Tele-15) use Livox
> Viewer 1 and produce LVX1 files.

### Recording with Livox Viewer

#### Livox Viewer 1 (LVX format)

1. With the sensor streaming, click the **Record** button (red circle) in the
   toolbar.
2. Choose an output path and click **Start**.
3. Click **Stop** when you have captured enough data.

The output is a `.lvx` file.

#### Livox Viewer 2 (LVX2 format)

1. Click the **Record** icon in the upper-right toolbar.
2. Set the output path, then click **Start Recording**.
3. Click **Stop Recording** when done.

The output is a `.lvx2` file.

#### Recording with the Livox SDK (optional)

If you prefer a command-line workflow, the
[Livox SDK2](https://github.com/Livox-SDK/Livox-SDK2) includes sample
recording applications:

```bash
# Clone and build SDK2
git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd Livox-SDK2 && mkdir build && cd build
cmake .. && make -j$(nproc)

# Run the sample recorder (outputs .lvx2)
./samples/livox_lidar_quick_start/livox_lidar_quick_start \
    ../samples/livox_lidar_quick_start/config.json
```

Edit `config.json` to set your sensor's IP and the desired output path.

### Loading Livox Data

```python
from sensor_transposition.lidar.livox import LivoxParser

# Works for both .lvx (v1) and .lvx2 (v2) files
parser = LivoxParser("recording.lvx2")

# Structured numpy array – fields: x, y, z, intensity (all float32, metres)
cloud = parser.read()
print(f"Points : {len(cloud)}")
print(f"First point: x={cloud['x'][0]:.3f} m, y={cloud['y'][0]:.3f} m, z={cloud['z'][0]:.3f} m")

xyz   = parser.xyz()           # (N, 3)
xyz_i = parser.xyz_intensity() # (N, 4)
```

---

## Verifying Your Data

After loading a point cloud, a quick sanity check helps confirm the data is
sensible before you use it downstream:

```python
import numpy as np
from sensor_transposition.lidar.velodyne import VelodyneParser  # swap for Ouster/Livox as needed

parser = VelodyneParser("frame_000000.bin")
cloud  = parser.read()
xyz    = parser.xyz()

# Basic statistics
print(f"Total points : {len(cloud)}")
print(f"X range      : [{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}] m")
print(f"Y range      : [{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}] m")
print(f"Z range      : [{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}] m")
print(f"Intensity min/max: {cloud['intensity'].min():.3f} / {cloud['intensity'].max():.3f}")

# Check for NaN or Inf values
assert not np.any(np.isnan(xyz)),  "NaN values present – check conversion script"
assert not np.any(np.isinf(xyz)),  "Inf values present – check conversion script"

# Rough range check – most outdoor scenes are within ±200 m
assert xyz[:, :2].max() < 250, "Points suspiciously far away"
```

**Expected ranges for a typical outdoor scan:**

| Metric | Typical value |
|--------|---------------|
| Total points (single frame) | 10 000 – 200 000 |
| XY range | ±0.5 m to ±150 m (sensor-dependent) |
| Z range | −3 m to +10 m (ground + overhead) |
| Intensity | 0.0 – 1.0 (normalised) |

---

## Troubleshooting

### No points / empty cloud

- **Velodyne / Ouster:** Confirm the sensor's UDP packets are reaching your
  host.  Run `tcpdump -i eth0 udp port 2368` (Velodyne) or
  `tcpdump -i eth0 udp port 7502` (Ouster) and verify packets arrive.
- **Livox:** Ensure the sensor IP and your NIC IP are on the same `/24` subnet.
  Check the Livox Viewer device list – the sensor must show a green status icon
  before recording.

### `FileNotFoundError` when loading

Verify the path passed to the parser.  All three parsers raise a clear
`FileNotFoundError` if the file does not exist:

```python
from pathlib import Path
path = Path("frame_000000.bin")
print(path.exists())   # must be True
```

### `ValueError: Cannot determine Ouster binary format`

The `.bin` file's size is not a multiple of 16 bytes (4-column) or 32 bytes
(8-column).  The file may be truncated.  Re-run the conversion script and
check for disk-space issues.

### `ValueError: Not a Livox LVX file`

The file's magic bytes do not match the expected `livox_tech` header.  Make
sure the file was not corrupted during transfer and that you are using the
correct file extension (`.lvx` vs `.lvx2`).

### Coordinate system alignment

All three parsers return coordinates in the sensor's native frame:

- **Velodyne / Ouster:** Forward–Left–Up (FLU), matching the ROS REP-103
  convention.
- **Livox:** Coordinate frame varies by model.  Mid-40/70 and Horizon use
  FLU; Mid-360 and HAP use FLU by default in the SDK.

When fusing data from multiple sensors, use the extrinsic calibration
support in `sensor_transposition.sensor_collection` (see
`examples/sensor_collection.yaml`) to transform all point clouds into a common
ego frame before processing.
