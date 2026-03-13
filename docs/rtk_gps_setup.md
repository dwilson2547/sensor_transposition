# RTK GPS Setup and NTRIP Integration Guide

This guide explains how to set up a Real-Time Kinematic (RTK) GPS base
station, subscribe to an NTRIP correction service, and configure
`sensor_transposition` to take advantage of centimetre-level accuracy.

---

## Overview

Standard single-frequency GPS receivers achieve **2–5 m** horizontal
accuracy under open sky.  RTK GPS uses **carrier-phase measurements** and a
reference correction stream to achieve **1–5 cm** horizontal accuracy in
real time.

The two main components are:

1. **Base station (reference)** – a GNSS receiver at a precisely-known
   location that transmits RTCM 3.x correction messages.
2. **Rover (your vehicle)** – a dual-frequency GNSS receiver that receives
   the corrections and computes a centimetre-level position relative to the
   base.

Corrections are delivered over the Internet via the **NTRIP** protocol
(Networked Transport of RTCM via Internet Protocol), or directly over a
UHF/LoRa radio link for base-to-rover setups without Internet access.

---

## Hardware Requirements

| Component | Recommendation |
|-----------|---------------|
| Rover receiver | Any dual-frequency (L1/L2 or L1/L5) RTK-capable module, e.g. u-blox F9P, Septentrio mosaic-X5, Swift Navigation Piksi |
| Base receiver | Same class as rover, or a rented VRS service via NTRIP |
| Antenna | Survey-grade patch antenna with a tight phase-centre variation specification |
| Data link | 4G/LTE modem for NTRIP, or 915 MHz/868 MHz radio for direct base-to-rover |

> **Tip:** Many national geodetic survey agencies operate free CORS
> (Continuously Operating Reference Station) networks that you can
> subscribe to via NTRIP — no physical base station required.

---

## NTRIP Client Configuration

### Using `str2str` (part of RTKLIB)

`str2str` is a lightweight NTRIP client / serial bridge included with
[RTKLIB](https://rtklib.com/):

```bash
# Receive RTCM3 corrections from an NTRIP caster and forward to a serial port
str2str \
  -in  ntrip://user:password@caster.example.com:2101/MOUNTPOINT \
  -out serial://ttyUSB0:115200#rtcm3
```

| Flag | Description |
|------|-------------|
| `-in ntrip://…` | NTRIP caster URL including credentials and mount-point |
| `-out serial://ttyUSBx:baud#rtcm3` | Forward RTCM stream to the receiver's serial port |

### Using `ntripclient` (Python)

If you prefer a pure-Python approach you can use `pygnss` or build a thin
wrapper with Python's `socket` module:

```python
import socket

def ntrip_connect(host: str, port: int, mountpoint: str,
                  user: str, password: str) -> socket.socket:
    """Open an NTRIP TCP connection and return the socket."""
    import base64
    credentials = base64.b64encode(f"{user}:{password}".encode()).decode()
    request = (
        f"GET /{mountpoint} HTTP/1.0\r\n"
        f"Host: {host}\r\n"
        f"Ntrip-Version: Ntrip/2.0\r\n"
        f"Authorization: Basic {credentials}\r\n"
        f"User-Agent: sensor_transposition/0.1\r\n"
        f"\r\n"
    )
    sock = socket.create_connection((host, port), timeout=10)
    sock.sendall(request.encode())
    resp = sock.recv(1024).decode(errors="replace")
    if "200 OK" not in resp and "ICY 200 OK" not in resp:
        sock.close()
        raise ConnectionError(f"NTRIP server rejected connection: {resp[:200]}")
    return sock
```

Once connected, pipe the raw bytes into `sensor_transposition`'s RTCM
parser:

```python
import io
from sensor_transposition.gps.rtcm import RtcmParser, Rtcm1005

sock = ntrip_connect("caster.example.com", 2101, "MOUNTPOINT", "user", "pw")

# Accumulate bytes into an in-memory buffer and parse
buf = io.BytesIO()
while True:
    chunk = sock.recv(4096)
    if not chunk:
        break
    buf.write(chunk)
    buf.seek(0)
    for msg in RtcmParser(buf).messages():
        if isinstance(msg, Rtcm1005):
            print(f"Base station ECEF: X={msg.x_m:.4f} Y={msg.y_m:.4f} Z={msg.z_m:.4f}")
    buf.seek(0, 2)  # seek to end for next write
```

---

## Parsing RTCM 3.x Messages

`sensor_transposition` includes a pure-Python RTCM 3.x parser in
`sensor_transposition.gps.rtcm` that supports:

| Message Type | Description |
|-------------|-------------|
| **MT1005** | Stationary RTK Reference Station ARP (base-station ECEF position) |
| **MSM4** (1074 / 1084 / 1094 / 1124) | Compact pseudorange + phase observations for GPS / GLONASS / Galileo / BeiDou |
| **MSM7** (1077 / 1087 / 1097 / 1127) | High-resolution pseudorange + phase (full-precision variant of MSM4) |

### Parsing a file

```python
from sensor_transposition.gps.rtcm import parse_rtcm_file, Rtcm1005, RtcmMsm

messages = parse_rtcm_file("corrections.rtcm3")

for msg in messages:
    if isinstance(msg, Rtcm1005):
        print(f"Base station ECEF: X={msg.x_m:.4f} m, "
              f"Y={msg.y_m:.4f} m, Z={msg.z_m:.4f} m")
    elif isinstance(msg, RtcmMsm):
        print(f"MSM{msg.msm_type} {msg.constellation} "
              f"epoch={msg.epoch_ms} ms  "
              f"sats=0x{msg.satellite_mask:016x}")
```

### Streaming from a socket or serial port

```python
import serial
from sensor_transposition.gps.rtcm import RtcmParser, Rtcm1005

ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
parser = RtcmParser(ser)

for msg in parser.messages():
    if isinstance(msg, Rtcm1005):
        print("Base ARP:", msg.x_m, msg.y_m, msg.z_m)
```

> **Note:** `RtcmParser` accepts any file-like object opened in binary mode,
> including `serial.Serial` objects and `socket.makefile("rb")` wrappers.

### Converting the base-station position to ENU

```python
from sensor_transposition.gps.converter import ecef_to_geodetic
from sensor_transposition.gps.rtcm import parse_rtcm_file, Rtcm1005

messages = parse_rtcm_file("corrections.rtcm3")
base = next(m for m in messages if isinstance(m, Rtcm1005))

# Convert base ECEF → geodetic
lat, lon, alt = ecef_to_geodetic(base.x_m, base.y_m, base.z_m)
print(f"Base station: lat={lat:.7f}° lon={lon:.7f}° alt={alt:.3f} m")
```

---

## Setting RTK Noise Covariance in `hdop_to_noise`

With RTK corrections applied, the horizontal position accuracy improves from
**~3 m** (standard GPS) to **~2 cm**.  Update `hdop_to_noise` accordingly:

```python
from sensor_transposition.gps.fusion import hdop_to_noise

# Standard single-frequency GPS (≈3 m horizontal at HDOP=1)
noise_standard = hdop_to_noise(fix.hdop)

# RTK-corrected position (≈2 cm horizontal at HDOP=1)
noise_rtk = hdop_to_noise(fix.hdop, base_sigma_m=0.02, vertical_sigma_m=0.05)
```

| Receiver type | `base_sigma_m` | `vertical_sigma_m` |
|---------------|---------------|-------------------|
| Standard GPS (L1 only) | `3.0` m | `5.0` m |
| SBAS / DGPS-corrected | `1.0` m | `2.0` m |
| RTK float solution | `0.3` m | `0.5` m |
| RTK fixed solution | `0.02` m | `0.05` m |

Pass the resulting covariance to `GpsFuser.fuse_into_ekf`:

```python
from sensor_transposition.gps.fusion import GpsFuser, hdop_to_noise
from sensor_transposition.imu.ekf import ImuEkf, EkfState

fuser = GpsFuser(ref_lat=51.5080, ref_lon=-0.1281, ref_alt=10.0)
ekf   = ImuEkf()
state = EkfState()

for fix in rtk_fixes:
    # Use RTK sigma values for centimetre-level accuracy
    noise = hdop_to_noise(fix.hdop, base_sigma_m=0.02, vertical_sigma_m=0.05)
    state = fuser.fuse_into_ekf(ekf, state, fix, noise)
```

---

## Handling GNSS Outages

During GNSS outages (tunnels, multi-storey car parks, urban canyons) RTK
corrections and position fixes become unavailable.  Configure `GpsFuser`
with a `max_fix_age_sec` limit and an `on_outage` callback to detect outages
and switch to a dead-reckoning fallback:

```python
from sensor_transposition.gps.fusion import GpsFuser, hdop_to_noise
from sensor_transposition.imu.ekf import ImuEkf, EkfState

def on_gnss_outage(age_sec: float) -> None:
    print(f"GNSS outage: last fix was {age_sec:.1f} s ago — "
          f"relying on IMU / wheel odometry")

fuser = GpsFuser(
    ref_lat=51.5080,
    ref_lon=-0.1281,
    max_fix_age_sec=2.0,      # skip GPS updates older than 2 s
    on_outage=on_gnss_outage, # called each time an update is skipped
)

ekf   = ImuEkf()
state = EkfState()
current_time = 0.0

for t, fix, accel, gyro, dt in sensor_stream:
    # IMU prediction (always)
    state = ekf.predict(state, accel, gyro, dt)
    current_time += dt

    # GPS update (skipped automatically during outages)
    noise = hdop_to_noise(fix.hdop, base_sigma_m=0.02, vertical_sigma_m=0.05)
    state = fuser.fuse_into_ekf(
        ekf, state, fix, noise,
        current_timestamp=current_time,
    )

    # Check fix staleness externally if needed
    age = fuser.fix_age(current_time)
    if age is not None and age > 5.0:
        print(f"Warning: GPS fix is {age:.1f} s old")
```

See `docs/gps_fusion.md` for a more detailed treatment of the GPS fusion
workflow and the EKF integration pattern.

---

## Recommended Setup Checklist

1. ☐ Install RTKLIB (`str2str`, `rtkpost`, `rtkplot`) on your base-station
   computer.
2. ☐ Mount the base-station antenna on a precisely-surveyed benchmark or
   measure its position with a 30-minute static observation session processed
   in post-processing (PPP or OPUS).
3. ☐ Subscribe to a local CORS / NTRIP caster, or configure `str2str` to
   broadcast corrections from your own base receiver.
4. ☐ Verify the rover outputs RTK-fixed sentences (fix quality `4` in GGA,
   or status `4` / `5` on NMEA GPGST).
5. ☐ Set `base_sigma_m=0.02` (RTK fixed) or `base_sigma_m=0.3` (RTK float)
   when constructing the noise covariance in `hdop_to_noise`.
6. ☐ Set `max_fix_age_sec=1.0` (or `2.0` for low-speed platforms) and
   provide an `on_outage` callback to fall back to wheel odometry / LiDAR
   odometry during outages.
