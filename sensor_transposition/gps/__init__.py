"""
sensor_transposition.gps

GPS/GNSS data parsing and coordinate-frame conversion utilities.
"""

from sensor_transposition.gps.nmea import (
    GgaFix,
    RmcFix,
    NmeaParser,
    load_nmea,
)
from sensor_transposition.gps.converter import (
    geodetic_to_ecef,
    ecef_to_geodetic,
    ecef_to_enu,
    enu_to_ecef,
    geodetic_to_enu,
    geodetic_to_utm,
    utm_to_geodetic,
    utm_zone_number,
    utm_zone_letter,
)
from sensor_transposition.gps.fusion import (
    GpsFuser,
    hdop_to_noise,
)
from sensor_transposition.gps.rtcm import (
    Rtcm1005,
    RtcmMsm,
    RtcmParser,
    parse_rtcm_file,
)

__all__ = [
    "GgaFix",
    "RmcFix",
    "NmeaParser",
    "load_nmea",
    "geodetic_to_ecef",
    "ecef_to_geodetic",
    "ecef_to_enu",
    "enu_to_ecef",
    "geodetic_to_enu",
    "geodetic_to_utm",
    "utm_to_geodetic",
    "utm_zone_number",
    "utm_zone_letter",
    "GpsFuser",
    "hdop_to_noise",
    "Rtcm1005",
    "RtcmMsm",
    "RtcmParser",
    "parse_rtcm_file",
]
