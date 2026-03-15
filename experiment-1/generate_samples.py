"""
generate_samples.py

Generate small binary sensor sample files for experiment-1.

This script creates realistic sample data files for three sensor types
supported by the sensor_transposition library:

  - sensor_samples/lidar_frame.bin   Velodyne KITTI-format LiDAR point cloud
  - sensor_samples/imu_data.bin      IMU binary recording
  - sensor_samples/radar_frame.bin   Radar detection binary file

Run from the experiment-1 directory:

    python generate_samples.py

The GPS sample (sensor_samples/gps_log.nmea) is stored as a plain-text
NMEA 0183 file and does not need to be generated.
"""

from pathlib import Path

import numpy as np

SAMPLES_DIR = Path(__file__).parent / "sensor_samples"
SAMPLES_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Velodyne LiDAR sample  (KITTI binary format)
# ---------------------------------------------------------------------------
# A synthetic 32-beam scan with ~128 points per beam arranged in a
# partial hemisphere at ranges 5–30 m, resembling a single rotation of a
# Velodyne HDL-32E mounted on a vehicle roof.

NUM_BEAMS = 32
POINTS_PER_BEAM = 128

elevations = np.linspace(-30.67, 10.67, NUM_BEAMS)          # degrees (HDL-32E)
azimuths = np.linspace(0.0, 360.0, POINTS_PER_BEAM, endpoint=False)

lidar_points = []
for el_deg in elevations:
    el_rad = np.radians(el_deg)
    for az_deg in azimuths:
        az_rad = np.radians(az_deg)
        r = RNG.uniform(5.0, 30.0)
        x = r * np.cos(el_rad) * np.cos(az_rad)
        y = r * np.cos(el_rad) * np.sin(az_rad)
        z = r * np.sin(el_rad)
        intensity = RNG.uniform(0.1, 1.0)
        lidar_points.append([x, y, z, intensity])

lidar_array = np.array(lidar_points, dtype=np.float32)
lidar_path = SAMPLES_DIR / "lidar_frame.bin"
lidar_array.tofile(lidar_path)
print(f"[LiDAR]  Wrote {len(lidar_points):,} points  →  {lidar_path}")

# ---------------------------------------------------------------------------
# IMU sample (custom binary format)
# ---------------------------------------------------------------------------
# 200 Hz recording for 0.5 s (100 samples) simulating a vehicle turning left
# at ~0.3 rad/s with 9.81 m/s² gravity on the z-axis.

IMU_RECORD_DTYPE = np.dtype([
    ("timestamp", np.float64),
    ("ax", np.float32),
    ("ay", np.float32),
    ("az", np.float32),
    ("wx", np.float32),
    ("wy", np.float32),
    ("wz", np.float32),
])

NUM_IMU = 100
DT_IMU = 1.0 / 200.0
T0_IMU = 1_710_000_000.0  # arbitrary recent UNIX epoch

imu_records = np.empty(NUM_IMU, dtype=IMU_RECORD_DTYPE)
for i in range(NUM_IMU):
    imu_records["timestamp"][i] = T0_IMU + i * DT_IMU
    imu_records["ax"][i] = RNG.normal(0.0, 0.05)
    imu_records["ay"][i] = RNG.normal(0.0, 0.05)
    imu_records["az"][i] = 9.81 + RNG.normal(0.0, 0.02)
    imu_records["wx"][i] = RNG.normal(0.0, 0.002)
    imu_records["wy"][i] = RNG.normal(0.0, 0.002)
    imu_records["wz"][i] = 0.3 + RNG.normal(0.0, 0.005)

imu_path = SAMPLES_DIR / "imu_data.bin"
with open(imu_path, "wb") as f:
    f.write(imu_records.tobytes())
print(f"[IMU]    Wrote {NUM_IMU} records @ 200 Hz  →  {imu_path}")

# ---------------------------------------------------------------------------
# Radar sample (custom binary format)
# ---------------------------------------------------------------------------
# 15 detections from a forward-looking automotive radar with realistic
# range / azimuth / velocity values.

RADAR_DETECTION_DTYPE = np.dtype([
    ("range", np.float32),
    ("azimuth", np.float32),
    ("elevation", np.float32),
    ("velocity", np.float32),
    ("snr", np.float32),
])

NUM_DETECTIONS = 15
radar_records = np.empty(NUM_DETECTIONS, dtype=RADAR_DETECTION_DTYPE)
radar_records["range"] = RNG.uniform(5.0, 100.0, NUM_DETECTIONS).astype(np.float32)
radar_records["azimuth"] = RNG.uniform(-45.0, 45.0, NUM_DETECTIONS).astype(np.float32)
radar_records["elevation"] = RNG.uniform(-7.0, 7.0, NUM_DETECTIONS).astype(np.float32)
radar_records["velocity"] = RNG.uniform(-20.0, 5.0, NUM_DETECTIONS).astype(np.float32)
radar_records["snr"] = RNG.uniform(10.0, 35.0, NUM_DETECTIONS).astype(np.float32)

radar_path = SAMPLES_DIR / "radar_frame.bin"
with open(radar_path, "wb") as f:
    f.write(radar_records.tobytes())
print(f"[Radar]  Wrote {NUM_DETECTIONS} detections  →  {radar_path}")

print("\nAll sensor sample files generated successfully.")
