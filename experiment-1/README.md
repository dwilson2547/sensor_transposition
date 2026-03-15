# experiment-1 – Sensor Data Parsing Demo

Autonomous-vehicle sensor parsing experiment using the
[sensor_transposition](../README.md) library.

## Goal

Retrieve small samples of real-format sensor data files and use the library
to prove that parsing works end-to-end for all supported sensor modalities.

## Sensor modalities covered

| Sensor | File | Format |
|--------|------|--------|
| GPS / GNSS | `sensor_samples/gps_log.nmea` | NMEA 0183 text (GGA + RMC sentences) |
| LiDAR | `sensor_samples/lidar_frame.bin` | Velodyne KITTI binary (x, y, z, intensity × float32) |
| IMU | `sensor_samples/imu_data.bin` | Custom binary (timestamp × float64 + accel/gyro × float32) |
| Radar | `sensor_samples/radar_frame.bin` | Custom binary (range, azimuth, elevation, velocity, SNR × float32) |

The GPS file (`gps_log.nmea`) contains realistic NMEA 0183 sentences recorded
near San Francisco, CA (37.68 °N, 122.41 °W).  The binary files are generated
by `generate_samples.py` using sensor-accurate formats and physically plausible
values.

## Quickstart

```bash
# From the repository root – install the library first
pip install -e .

# Generate the binary sensor sample files
python experiment-1/generate_samples.py

# Parse all samples and print results
python experiment-1/parse_sensor_data.py
```

Or run from inside the experiment directory:

```bash
cd experiment-1
python generate_samples.py
python parse_sensor_data.py
```

## Expected output

```
============================================================
  1. GPS  –  NMEA 0183 parser
============================================================
File          : gps_log.nmea
Total records : 10  (GGA=5, RMC=5)

First GGA fix:
  Time        : 123519 UTC
  Latitude    : 37.679900°
  Longitude   : -122.414933°
  Altitude    : 16.4 m MSL
  Satellites  : 8
  HDOP        : 0.9
  Fix quality : 1
...
============================================================
  Summary
============================================================
All sensor parsers exercised successfully.

  GPS  : 10 NMEA records parsed
  LiDAR: 4,096 points loaded
  IMU  : 100 records @ 200 Hz
  Radar: 15 detections
```

## File descriptions

### `generate_samples.py`
Generates the three binary sample files in `sensor_samples/`.  The data uses
physically realistic parameters:

- **LiDAR**: 4,096-point synthetic scan from a 32-beam Velodyne HDL-32E
  (elevation from −30.67° to +10.67°), ranges 5–30 m.
- **IMU**: 100-sample, 200 Hz recording (0.5 s) of a left turn at ~0.3 rad/s
  with gravity on the Z-axis (~9.81 m/s²).
- **Radar**: 15 forward-looking detections, ranges 5–100 m, azimuth ±45°,
  with Doppler velocity indicating approaching targets.

### `parse_sensor_data.py`
Main experiment script.  Loads each sample file using the corresponding
`sensor_transposition` parser and prints summary statistics.

### `sensor_samples/gps_log.nmea`
Five GGA + five RMC sentences logged at 1 Hz over 5 seconds, with valid
NMEA checksums, representing a vehicle at rest then beginning to move
slowly through San Francisco.
