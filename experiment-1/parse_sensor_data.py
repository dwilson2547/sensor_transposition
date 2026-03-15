"""
parse_sensor_data.py

Autonomous-vehicle sensor data parsing experiment using sensor_transposition.

This script loads small real-format sensor samples from the sensor_samples/
directory and exercises the library parsers to prove end-to-end functionality:

  1. GPS  – NMEA 0183 text file  →  GgaFix / RmcFix records
  2. LiDAR – Velodyne KITTI .bin file  →  structured point cloud
  3. IMU   – custom binary .bin file   →  structured IMU records
  4. Radar – custom binary .bin file   →  structured detection records

Run from the experiment-1 directory:

    python parse_sensor_data.py

Or from the repository root:

    python experiment-1/parse_sensor_data.py
"""

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Resolve paths relative to this script regardless of working directory
# ---------------------------------------------------------------------------

SAMPLES_DIR = Path(__file__).parent / "sensor_samples"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# 1.  GPS – NMEA 0183
# ---------------------------------------------------------------------------

section("1. GPS  –  NMEA 0183 parser")

from sensor_transposition.gps.nmea import NmeaParser, GgaFix, RmcFix

nmea_path = SAMPLES_DIR / "gps_log.nmea"
parser = NmeaParser(nmea_path)
records = parser.records()

gga_fixes = [r for r in records if isinstance(r, GgaFix)]
rmc_fixes = [r for r in records if isinstance(r, RmcFix)]

print(f"File          : {nmea_path.name}")
print(f"Total records : {len(records)}  (GGA={len(gga_fixes)}, RMC={len(rmc_fixes)})")

if gga_fixes:
    fix = gga_fixes[0]
    print(f"\nFirst GGA fix:")
    print(f"  Time        : {fix.timestamp} UTC")
    print(f"  Latitude    : {fix.latitude:.6f}°")
    print(f"  Longitude   : {fix.longitude:.6f}°")
    print(f"  Altitude    : {fix.altitude:.1f} m MSL")
    print(f"  Satellites  : {fix.num_satellites}")
    print(f"  HDOP        : {fix.hdop}")
    print(f"  Fix quality : {fix.fix_quality}")

if rmc_fixes:
    fix = rmc_fixes[0]
    print(f"\nFirst RMC fix:")
    print(f"  Time        : {fix.timestamp} UTC")
    print(f"  Date        : {fix.date}")
    print(f"  Speed       : {fix.speed_knots:.2f} knots")
    print(f"  Course      : {fix.course:.1f}°")
    print(f"  Valid       : {fix.is_valid}")

# ---------------------------------------------------------------------------
# 2.  LiDAR – Velodyne KITTI binary
# ---------------------------------------------------------------------------

section("2. LiDAR  –  Velodyne KITTI binary parser")

from sensor_transposition.lidar.velodyne import VelodyneParser

lidar_path = SAMPLES_DIR / "lidar_frame.bin"
lidar = VelodyneParser(lidar_path)
cloud = lidar.read()
xyz = lidar.xyz()

ranges = np.linalg.norm(xyz, axis=1)

print(f"File          : {lidar_path.name}")
print(f"Total points  : {len(cloud):,}")
print(f"Fields        : {cloud.dtype.names}")
print(f"\nPoint cloud statistics:")
print(f"  X  range    : [{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}] m")
print(f"  Y  range    : [{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}] m")
print(f"  Z  range    : [{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}] m")
print(f"  Range       : [{ranges.min():.2f}, {ranges.max():.2f}] m")
print(f"  Intensity   : [{cloud['intensity'].min():.3f}, {cloud['intensity'].max():.3f}]")

# ---------------------------------------------------------------------------
# 3.  IMU – binary recording
# ---------------------------------------------------------------------------

section("3. IMU  –  binary recording parser")

from sensor_transposition.imu.imu import ImuParser

imu_path = SAMPLES_DIR / "imu_data.bin"
imu_parser = ImuParser(imu_path)
imu_data = imu_parser.read()

accel = imu_parser.linear_acceleration()
gyro = imu_parser.angular_velocity()
timestamps = imu_parser.timestamps()

duration = timestamps[-1] - timestamps[0]

print(f"File          : {imu_path.name}")
print(f"Total records : {len(imu_data)}")
print(f"Duration      : {duration:.3f} s  ({1.0 / (timestamps[1] - timestamps[0]):.0f} Hz)")
print(f"\nAccelerometer (m/s²):")
print(f"  Mean        : [{accel[:, 0].mean():.4f}, {accel[:, 1].mean():.4f}, {accel[:, 2].mean():.4f}]")
print(f"  Std dev     : [{accel[:, 0].std():.4f}, {accel[:, 1].std():.4f}, {accel[:, 2].std():.4f}]")
print(f"\nGyroscope (rad/s):")
print(f"  Mean        : [{gyro[:, 0].mean():.4f}, {gyro[:, 1].mean():.4f}, {gyro[:, 2].mean():.4f}]")
print(f"  Std dev     : [{gyro[:, 0].std():.4f}, {gyro[:, 1].std():.4f}, {gyro[:, 2].std():.4f}]")

# ---------------------------------------------------------------------------
# 4.  Radar – binary detections
# ---------------------------------------------------------------------------

section("4. Radar  –  binary detection parser")

from sensor_transposition.radar.radar import RadarParser

radar_path = SAMPLES_DIR / "radar_frame.bin"
radar_parser = RadarParser(radar_path)
detections = radar_parser.read()
xyz_radar = radar_parser.xyz()

print(f"File          : {radar_path.name}")
print(f"Detections    : {len(detections)}")
print(f"Fields        : {detections.dtype.names}")
print(f"\nDetection statistics:")
print(f"  Range       : [{detections['range'].min():.1f}, {detections['range'].max():.1f}] m")
print(f"  Azimuth     : [{detections['azimuth'].min():.1f}, {detections['azimuth'].max():.1f}]°")
print(f"  Velocity    : [{detections['velocity'].min():.2f}, {detections['velocity'].max():.2f}] m/s")
print(f"  SNR         : [{detections['snr'].min():.1f}, {detections['snr'].max():.1f}] dB")

print(f"\nCartesian coordinates (first 5 detections):")
for i in range(min(5, len(xyz_radar))):
    print(f"  [{i}]  x={xyz_radar[i, 0]:7.2f}  y={xyz_radar[i, 1]:7.2f}  z={xyz_radar[i, 2]:6.2f}  m")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

section("Summary")
print("All sensor parsers exercised successfully.")
print()
print(f"  GPS  : {len(records)} NMEA records parsed")
print(f"  LiDAR: {len(cloud):,} points loaded")
print(f"  IMU  : {len(imu_data)} records @ {1.0 / (timestamps[1] - timestamps[0]):.0f} Hz")
print(f"  Radar: {len(detections)} detections")
