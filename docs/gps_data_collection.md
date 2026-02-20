# GPS Data Collection Guide

A practical how-to guide for capturing real NMEA 0183 data from a **USB or
serial GPS receiver** without ROS, and loading the results with the
`sensor_transposition` library.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Hardware Setup](#hardware-setup)
4. [Capturing NMEA Data](#capturing-nmea-data)
   - [Linux / macOS – using `cat` or `gpspipe`](#linux--macos)
   - [Windows – using PuTTY or TeraTerm](#windows)
   - [Python – direct serial capture](#python-direct-serial-capture)
5. [Loading with sensor_transposition](#loading-with-sensor_transposition)
   - [All records](#all-records)
   - [GGA fixes only](#gga-fixes-only)
   - [RMC fixes only](#rmc-fixes-only)
6. [Verifying Your Data](#verifying-your-data)
7. [Troubleshooting](#troubleshooting)

---

## Overview

Most off-the-shelf GPS receivers output data in the
[NMEA 0183](https://www.nmea.org/content/STANDARDS/NMEA_0183_Standard) text
format over a serial (UART / USB-CDC) connection.  The two most useful sentence
types are:

| Sentence | Contents |
|----------|----------|
| **GGA**  | Position (lat/lon), altitude, fix quality, satellite count, HDOP |
| **RMC**  | Position, speed over ground, course, UTC date/time, validity flag |

The `sensor_transposition` library's `NmeaParser` reads ``.nmea`` (or ``.txt``)
files containing these sentences and returns structured Python dataclass
instances that are easy to work with downstream.

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

### Optional: `pyserial` for direct serial capture in Python

```bash
pip install pyserial
```

### Optional: `gpsd` for Linux daemon-based capture

```bash
sudo apt install gpsd gpsd-clients   # Debian / Ubuntu
sudo dnf install gpsd gpsd-clients   # Fedora / RHEL
```

---

## Hardware Setup

1. Plug the GPS receiver into a USB port (most modern receivers enumerate as a
   USB-CDC serial device).
2. Identify the serial port:
   - **Linux:** `ls /dev/ttyUSB* /dev/ttyACM*` – typically `/dev/ttyUSB0` or
     `/dev/ttyACM0`.
   - **macOS:** `ls /dev/cu.usbmodem* /dev/cu.usbserial*`
   - **Windows:** Device Manager → Ports (COM & LPT) → note the `COMx` number.
3. Most GPS receivers communicate at **9600 baud**, 8N1.  Check your
   receiver's datasheet if the port does not produce output.
4. (Linux only) Add your user to the `dialout` group so you can access the
   serial port without `sudo`:

   ```bash
   sudo usermod -aG dialout $USER
   # log out and back in for the change to take effect
   ```

---

## Capturing NMEA Data

### Linux / macOS

#### Simple `cat` redirect

```bash
# Replace /dev/ttyUSB0 and baud rate as needed
stty -F /dev/ttyUSB0 9600 raw
cat /dev/ttyUSB0 > gps_log.nmea
# Press Ctrl-C when you have captured enough data
```

#### Using `gpspipe` (requires `gpsd`)

```bash
# Start the daemon (replace /dev/ttyUSB0 as needed)
sudo gpsd /dev/ttyUSB0 -F /var/run/gpsd.sock

# Dump raw NMEA sentences to a file
gpspipe -r -n 200 > gps_log.nmea   # capture 200 sentences then exit
```

### Windows

1. Open **PuTTY** (or **TeraTerm**).
2. Select **Serial**, set the COM port and speed (9600).
3. Under **Session → Logging**, choose *All session output* and set a log
   file path (e.g. `C:\gps\gps_log.nmea`).
4. Click **Open** – NMEA sentences will scroll in the terminal and be written
   to the log file simultaneously.
5. Close PuTTY when done.

### Python – direct serial capture

```python
"""
capture_gps.py
Capture NMEA sentences from a serial GPS receiver to a file.

Requires: pyserial  (pip install pyserial)
"""
import serial
import time
from pathlib import Path

PORT      = "/dev/ttyUSB0"   # adjust for your system
BAUD_RATE = 9600
DURATION  = 60               # seconds to capture
OUT_FILE  = Path("gps_log.nmea")

start = time.monotonic()
with serial.Serial(PORT, BAUD_RATE, timeout=1) as ser, OUT_FILE.open("w") as f:
    print(f"Capturing for {DURATION} s → {OUT_FILE}")
    while time.monotonic() - start < DURATION:
        line = ser.readline().decode("ascii", errors="replace").strip()
        if line.startswith("$"):
            f.write(line + "\n")
            print(line)

print("Done.")
```

Run it:

```bash
python capture_gps.py
```

The output file contains one NMEA sentence per line, ready to be parsed.

---

## Loading with sensor_transposition

### All records

```python
from sensor_transposition.gps import NmeaParser, GgaFix, RmcFix

parser = NmeaParser("gps_log.nmea")

for record in parser.records():
    if isinstance(record, GgaFix):
        print(f"GGA  lat={record.latitude:.6f}  lon={record.longitude:.6f}"
              f"  alt={record.altitude:.1f} m  fix={record.fix_quality}"
              f"  sats={record.num_satellites}")
    elif isinstance(record, RmcFix):
        if record.is_valid:
            print(f"RMC  lat={record.latitude:.6f}  lon={record.longitude:.6f}"
                  f"  speed={record.speed_knots:.1f} kn"
                  f"  course={record.course:.1f}°")
```

### GGA fixes only

GGA sentences provide altitude and fix-quality information that RMC sentences
omit.  Use `gga_fixes()` when you need altitude or HDOP:

```python
from sensor_transposition.gps import NmeaParser

parser = NmeaParser("gps_log.nmea")
fixes  = parser.gga_fixes()

print(f"Total GGA fixes : {len(fixes)}")

for fix in fixes:
    print(f"  {fix.timestamp}  ({fix.latitude:.6f}, {fix.longitude:.6f})"
          f"  alt={fix.altitude:.1f} m  hdop={fix.hdop:.1f}")
```

### RMC fixes only

RMC sentences include speed, course, and a validity flag, making them useful
for tracking motion:

```python
from sensor_transposition.gps import NmeaParser

parser  = NmeaParser("gps_log.nmea")
fixes   = parser.rmc_fixes()
valid   = [f for f in fixes if f.is_valid]

print(f"Total RMC fixes : {len(fixes)}  (valid: {len(valid)})")

for fix in valid:
    print(f"  {fix.date} {fix.timestamp}"
          f"  ({fix.latitude:.6f}, {fix.longitude:.6f})"
          f"  speed={fix.speed_knots:.1f} kn  course={fix.course:.1f}°")
```

### Module-level convenience function

If you just need all records in one call:

```python
from sensor_transposition.gps import load_nmea

records = load_nmea("gps_log.nmea")
print(f"Parsed {len(records)} records")
```

---

## Verifying Your Data

After loading, a quick sanity check confirms the data is sensible:

```python
from sensor_transposition.gps import NmeaParser

parser = NmeaParser("gps_log.nmea")
fixes  = parser.gga_fixes()

assert fixes, "No GGA fixes found – check the log file contains $GPGGA or $GNGGA sentences"

lats = [f.latitude  for f in fixes]
lons = [f.longitude for f in fixes]
alts = [f.altitude  for f in fixes]

print(f"GGA fixes       : {len(fixes)}")
print(f"Latitude  range : [{min(lats):.6f}, {max(lats):.6f}]")
print(f"Longitude range : [{min(lons):.6f}, {max(lons):.6f}]")
print(f"Altitude  range : [{min(alts):.1f}, {max(alts):.1f}] m")
print(f"Fix quality     : {set(f.fix_quality for f in fixes)}")
print(f"Satellites      : min={min(f.num_satellites for f in fixes)}"
      f"  max={max(f.num_satellites for f in fixes)}")
```

**Expected ranges for a typical outdoor fix:**

| Metric | Typical value |
|--------|---------------|
| Fix quality | 1 (GPS) or 2 (DGPS) |
| Satellites in use | 4 – 12 |
| HDOP | < 2.0 (good), < 5.0 (acceptable) |
| Altitude | −100 m to +5 000 m (sea level ± terrain) |

---

## Troubleshooting

### No sentences in the output file

- Confirm the serial port name and baud rate match your receiver.
- On Linux, run `dmesg | tail -20` immediately after plugging in the receiver
  to see which `/dev/ttyUSBx` or `/dev/ttyACMx` node was created.
- Try a different baud rate (4 800 is common on older units; 38 400 or 115 200
  on high-update-rate units).

### `FileNotFoundError` when parsing

Verify the path passed to `NmeaParser`:

```python
from pathlib import Path
print(Path("gps_log.nmea").exists())   # must print True
```

### All fixes have `latitude=0.0, longitude=0.0`

The receiver does not yet have a valid fix (cold start can take 30–120 s
outdoors).  Check `fix_quality` in GGA sentences — a value of `0` means no
fix.  Wait for the value to become `1` or higher.

### `is_valid` is always `False` for RMC fixes

The RMC `status` field is `'V'` (void) when the receiver has no valid fix.
Again, allow time for the receiver to acquire satellites.

### Mixing GGA and RMC fixes by timestamp

Both sentence types carry a `timestamp` field in `HHMMSS.ss` format.  You can
pair them by timestamp to get the full set of fields available from both types:

```python
from sensor_transposition.gps import NmeaParser

parser = NmeaParser("gps_log.nmea")
all_records = parser.records()

gga_by_ts = {r.timestamp: r for r in all_records
             if hasattr(r, "fix_quality")}
rmc_by_ts  = {r.timestamp: r for r in all_records
             if hasattr(r, "speed_knots")}

for ts in sorted(set(gga_by_ts) & set(rmc_by_ts)):
    gga = gga_by_ts[ts]
    rmc = rmc_by_ts[ts]
    print(f"{ts}  alt={gga.altitude:.1f} m  speed={rmc.speed_knots:.1f} kn")
```
