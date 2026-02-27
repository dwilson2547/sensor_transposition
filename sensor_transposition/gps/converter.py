"""
gps/converter.py

Coordinate-frame conversion utilities for GPS/GNSS data.

Provides conversions between:

* **Geodetic** (latitude / longitude / altitude) and **ECEF**
  (Earth-Centred Earth-Fixed Cartesian).
* **ECEF** and **ENU** (East / North / Up) local tangent-plane frame
  centred on an arbitrary reference point.
* **Geodetic** and **UTM** (Universal Transverse Mercator) grid coordinates.

All angles are in **decimal degrees** unless otherwise stated.
All distances are in **metres**.

References
----------
* WGS-84 definition: NIMA TR8350.2, 3rd edition (2000).
* Bowring, B. R. (1985). *The geodesic line and the normal section.*
* Helmert / Krueger series for Transverse Mercator as summarised in:
  Snyder, J. P. (1987). *Map Projections – A Working Manual.* USGS.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# WGS-84 ellipsoid constants
# ---------------------------------------------------------------------------

_A = 6_378_137.0  # semi-major axis (m)
_F = 1.0 / 298.257_223_563  # flattening
_B = _A * (1.0 - _F)  # semi-minor axis (m)
_E2 = 2.0 * _F - _F**2  # first eccentricity squared  (e²)
_EP2 = _E2 / (1.0 - _E2)  # second eccentricity squared (e'²)

# ---------------------------------------------------------------------------
# UTM projection constants
# ---------------------------------------------------------------------------

_K0 = 0.9996  # UTM scale factor at central meridian
_UTM_E0 = 500_000.0  # false easting (m)
_UTM_N0_SOUTH = 10_000_000.0  # false northing for southern hemisphere (m)

# UTM latitude-band letters (south to north, 8° bands starting at −80°)
_UTM_BAND_LETTERS = "CDEFGHJKLMNPQRSTUVWX"


# ---------------------------------------------------------------------------
# ECEF <-> Geodetic
# ---------------------------------------------------------------------------


def geodetic_to_ecef(
    lat_deg: float, lon_deg: float, alt_m: float = 0.0
) -> Tuple[float, float, float]:
    """Convert geodetic coordinates to ECEF Cartesian.

    Args:
        lat_deg: Geodetic latitude in decimal degrees (−90 … +90).
        lon_deg: Longitude in decimal degrees (−180 … +180).
        alt_m: Height above the WGS-84 ellipsoid in metres (default 0).

    Returns:
        Tuple ``(X, Y, Z)`` in metres in the ECEF frame.

    Example::

        X, Y, Z = geodetic_to_ecef(51.5074, -0.1278, 11.0)
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    # Radius of curvature in the prime vertical
    N = _A / math.sqrt(1.0 - _E2 * sin_lat**2)

    X = (N + alt_m) * cos_lat * cos_lon
    Y = (N + alt_m) * cos_lat * sin_lon
    Z = (N * (1.0 - _E2) + alt_m) * sin_lat

    return X, Y, Z


def ecef_to_geodetic(
    x: float, y: float, z: float
) -> Tuple[float, float, float]:
    """Convert ECEF Cartesian coordinates to geodetic (Bowring iterative).

    Args:
        x: ECEF X in metres.
        y: ECEF Y in metres.
        z: ECEF Z in metres.

    Returns:
        Tuple ``(lat_deg, lon_deg, alt_m)`` where *lat_deg* and *lon_deg* are
        in decimal degrees and *alt_m* is height above the WGS-84 ellipsoid.

    Example::

        lat, lon, alt = ecef_to_geodetic(3_978_874.0, -18_501.0, 4_966_467.0)
    """
    lon = math.atan2(y, x)

    p = math.sqrt(x**2 + y**2)
    # Initial estimate of latitude (parametric / reduced latitude)
    lat = math.atan2(z, p * (1.0 - _E2))

    # Bowring iteration (converges in 2–3 steps)
    for _ in range(10):
        sin_lat = math.sin(lat)
        N = _A / math.sqrt(1.0 - _E2 * sin_lat**2)
        lat_new = math.atan2(z + _E2 * N * sin_lat, p)
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    N = _A / math.sqrt(1.0 - _E2 * sin_lat**2)

    if abs(cos_lat) > 1e-10:
        alt_m = p / cos_lat - N
    else:
        alt_m = abs(z) / abs(sin_lat) - N * (1.0 - _E2)

    return math.degrees(lat), math.degrees(lon), alt_m


# ---------------------------------------------------------------------------
# ECEF <-> ENU
# ---------------------------------------------------------------------------


def ecef_to_enu(
    x: float,
    y: float,
    z: float,
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float = 0.0,
) -> Tuple[float, float, float]:
    """Convert an ECEF point to a local ENU frame.

    The ENU frame is a right-handed tangent plane at the reference point
    (``lat0_deg``, ``lon0_deg``, ``alt0_m``):

    * **E** – East
    * **N** – North
    * **U** – Up (normal to ellipsoid)

    Args:
        x: ECEF X of the point to convert (m).
        y: ECEF Y of the point to convert (m).
        z: ECEF Z of the point to convert (m).
        lat0_deg: Geodetic latitude of the ENU origin (decimal degrees).
        lon0_deg: Longitude of the ENU origin (decimal degrees).
        alt0_m: Altitude of the ENU origin above the ellipsoid (m).

    Returns:
        Tuple ``(east, north, up)`` in metres relative to the origin.

    Example::

        e, n, u = ecef_to_enu(x, y, z, lat0_deg=51.5, lon0_deg=-0.1)
    """
    x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, alt0_m)
    dx, dy, dz = x - x0, y - y0, z - z0

    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    sin_lat = math.sin(lat0)
    cos_lat = math.cos(lat0)
    sin_lon = math.sin(lon0)
    cos_lon = math.cos(lon0)

    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return east, north, up


def enu_to_ecef(
    east: float,
    north: float,
    up: float,
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float = 0.0,
) -> Tuple[float, float, float]:
    """Convert a local ENU displacement to ECEF.

    Args:
        east: East component (m).
        north: North component (m).
        up: Up component (m).
        lat0_deg: Geodetic latitude of the ENU origin (decimal degrees).
        lon0_deg: Longitude of the ENU origin (decimal degrees).
        alt0_m: Altitude of the ENU origin above the ellipsoid (m).

    Returns:
        Tuple ``(X, Y, Z)`` in metres in the ECEF frame.

    Example::

        X, Y, Z = enu_to_ecef(100.0, 200.0, 0.0, lat0_deg=51.5, lon0_deg=-0.1)
    """
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    sin_lat = math.sin(lat0)
    cos_lat = math.cos(lat0)
    sin_lon = math.sin(lon0)
    cos_lon = math.cos(lon0)

    x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, alt0_m)

    dx = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
    dy = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
    dz = cos_lat * north + sin_lat * up

    return x0 + dx, y0 + dy, z0 + dz


# ---------------------------------------------------------------------------
# Geodetic <-> ENU (convenience wrappers)
# ---------------------------------------------------------------------------


def geodetic_to_enu(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float = 0.0,
) -> Tuple[float, float, float]:
    """Convert geodetic coordinates to a local ENU frame.

    Combines :func:`geodetic_to_ecef` and :func:`ecef_to_enu` for convenience.

    Args:
        lat_deg: Latitude of the point (decimal degrees).
        lon_deg: Longitude of the point (decimal degrees).
        alt_m: Altitude of the point above the ellipsoid (m).
        lat0_deg: Latitude of the ENU origin (decimal degrees).
        lon0_deg: Longitude of the ENU origin (decimal degrees).
        alt0_m: Altitude of the ENU origin (m).

    Returns:
        Tuple ``(east, north, up)`` in metres.

    Example::

        e, n, u = geodetic_to_enu(
            51.5075, -0.1279, 11.0,
            lat0_deg=51.5074, lon0_deg=-0.1278, alt0_m=11.0,
        )
    """
    x, y, z = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    return ecef_to_enu(x, y, z, lat0_deg, lon0_deg, alt0_m)


# ---------------------------------------------------------------------------
# UTM zone helpers
# ---------------------------------------------------------------------------


def utm_zone_number(lat_deg: float, lon_deg: float) -> int:
    """Return the UTM zone number (1–60) for a geodetic position.

    Special zones for Norway / Svalbard (as per the UTM standard) are handled.

    Args:
        lat_deg: Latitude in decimal degrees.
        lon_deg: Longitude in decimal degrees.

    Returns:
        Integer zone number in the range 1–60.
    """
    zone = int((lon_deg + 180.0) / 6.0) + 1

    # Norway exception: zone 32 is extended westward in 56°–64° N
    if 56.0 <= lat_deg < 64.0 and 3.0 <= lon_deg < 12.0:
        zone = 32

    # Svalbard exceptions (72°–84° N)
    if 72.0 <= lat_deg < 84.0:
        if 0.0 <= lon_deg < 9.0:
            zone = 31
        elif 9.0 <= lon_deg < 21.0:
            zone = 33
        elif 21.0 <= lon_deg < 33.0:
            zone = 35
        elif 33.0 <= lon_deg < 42.0:
            zone = 37

    return zone


def utm_zone_letter(lat_deg: float) -> str:
    """Return the UTM latitude-band letter for a given latitude.

    Args:
        lat_deg: Latitude in decimal degrees (must be in −80° … +84°).

    Returns:
        Single uppercase letter in the range C–X (excluding I and O).

    Raises:
        ValueError: If the latitude is outside the UTM coverage area.
    """
    if not (-80.0 <= lat_deg <= 84.0):
        raise ValueError(
            f"Latitude {lat_deg} is outside the UTM coverage area (−80° to +84°)."
        )
    idx = int((lat_deg + 80.0) / 8.0)
    # Clamp to the last valid band (X covers 72°–84°, i.e. 12° wide)
    idx = min(idx, len(_UTM_BAND_LETTERS) - 1)
    return _UTM_BAND_LETTERS[idx]


# ---------------------------------------------------------------------------
# Geodetic <-> UTM
# ---------------------------------------------------------------------------


def geodetic_to_utm(
    lat_deg: float, lon_deg: float
) -> Tuple[float, float, int, str]:
    """Convert geodetic coordinates to UTM (Transverse Mercator projection).

    Uses the standard Helmert/Krueger series (Snyder 1987) with the
    WGS-84 ellipsoid.

    Args:
        lat_deg: Latitude in decimal degrees (−80° … +84°).
        lon_deg: Longitude in decimal degrees (−180° … +180°).

    Returns:
        Tuple ``(easting, northing, zone_number, zone_letter)`` where:

        * *easting* – metres east of the central meridian + false easting
          (500 000 m);
        * *northing* – metres north of the equator (southern hemisphere adds
          a false northing of 10 000 000 m);
        * *zone_number* – integer 1–60;
        * *zone_letter* – latitude-band letter (C–X).

    Raises:
        ValueError: If the latitude is outside the UTM coverage area.

    Example::

        easting, northing, zone, band = geodetic_to_utm(51.5074, -0.1278)
        # → (699_330.6, 5_710_155.4, 30, 'U')
    """
    zone_num = utm_zone_number(lat_deg, lon_deg)
    zone_let = utm_zone_letter(lat_deg)

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lon0 = math.radians((zone_num - 1) * 6 - 180 + 3)  # central meridian

    # --- Transverse Mercator series (Snyder 1987, equations 8-13 to 8-15) ---
    e2 = _E2
    n_param = _A / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)  # prime vertical

    T = math.tan(lat) ** 2
    C = _EP2 * math.cos(lat) ** 2
    A_ = math.cos(lat) * (lon - lon0)

    # Meridional arc (M) via series
    e4 = e2**2
    e6 = e2**3
    M = _A * (
        (1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * lat
        - (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0)
        * math.sin(2.0 * lat)
        + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * math.sin(4.0 * lat)
        - (35.0 * e6 / 3072.0) * math.sin(6.0 * lat)
    )

    easting = (
        _K0
        * n_param
        * (
            A_
            + (1.0 - T + C) * A_**3 / 6.0
            + (5.0 - 18.0 * T + T**2 + 72.0 * C - 58.0 * _EP2) * A_**5 / 120.0
        )
        + _UTM_E0
    )

    northing = _K0 * (
        M
        + n_param
        * math.tan(lat)
        * (
            A_**2 / 2.0
            + (5.0 - T + 9.0 * C + 4.0 * C**2) * A_**4 / 24.0
            + (61.0 - 58.0 * T + T**2 + 600.0 * C - 330.0 * _EP2) * A_**6 / 720.0
        )
    )

    if lat_deg < 0.0:
        northing += _UTM_N0_SOUTH

    return easting, northing, zone_num, zone_let


def utm_to_geodetic(
    easting: float, northing: float, zone_number: int, zone_letter: str
) -> Tuple[float, float]:
    """Convert UTM coordinates back to geodetic (lat / lon).

    Uses the inverse Transverse Mercator series (Snyder 1987).

    Args:
        easting: Easting in metres (including the 500 000 m false easting).
        northing: Northing in metres (including the 10 000 000 m false
            northing for the southern hemisphere).
        zone_number: UTM zone number (1–60).
        zone_letter: UTM latitude-band letter (C–X).

    Returns:
        Tuple ``(lat_deg, lon_deg)`` in decimal degrees.

    Example::

        lat, lon = utm_to_geodetic(699_330.6, 5_710_155.4, 30, 'U')
        # → (51.5074, -0.1278)
    """
    x = easting - _UTM_E0
    y = northing
    if zone_letter.upper() < "N":
        y -= _UTM_N0_SOUTH

    lon0 = math.radians((zone_number - 1) * 6 - 180 + 3)

    e2 = _E2
    e1 = (1.0 - math.sqrt(1.0 - e2)) / (1.0 + math.sqrt(1.0 - e2))

    M = y / _K0
    mu = M / (_A * (1.0 - e2 / 4.0 - 3.0 * e2**2 / 64.0 - 5.0 * e2**3 / 256.0))

    # Footprint latitude (phi1) via series
    phi1 = (
        mu
        + (3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0) * math.sin(2.0 * mu)
        + (21.0 * e1**2 / 16.0 - 55.0 * e1**4 / 32.0) * math.sin(4.0 * mu)
        + (151.0 * e1**3 / 96.0) * math.sin(6.0 * mu)
        + (1097.0 * e1**4 / 512.0) * math.sin(8.0 * mu)
    )

    N1 = _A / math.sqrt(1.0 - e2 * math.sin(phi1) ** 2)
    T1 = math.tan(phi1) ** 2
    C1 = _EP2 * math.cos(phi1) ** 2
    R1 = _A * (1.0 - e2) / (1.0 - e2 * math.sin(phi1) ** 2) ** 1.5
    D = x / (N1 * _K0)

    lat = phi1 - (N1 * math.tan(phi1) / R1) * (
        D**2 / 2.0
        - (5.0 + 3.0 * T1 + 10.0 * C1 - 4.0 * C1**2 - 9.0 * _EP2) * D**4 / 24.0
        + (61.0 + 90.0 * T1 + 298.0 * C1 + 45.0 * T1**2 - 252.0 * _EP2 - 3.0 * C1**2)
        * D**6
        / 720.0
    )

    lon = lon0 + (
        D
        - (1.0 + 2.0 * T1 + C1) * D**3 / 6.0
        + (5.0 - 2.0 * C1 + 28.0 * T1 - 3.0 * C1**2 + 8.0 * _EP2 + 24.0 * T1**2)
        * D**5
        / 120.0
    ) / math.cos(phi1)

    return math.degrees(lat), math.degrees(lon)
