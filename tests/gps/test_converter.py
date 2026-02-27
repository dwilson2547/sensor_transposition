"""Tests for the GPS coordinate-frame converter (ECEF / ENU / UTM)."""

import math

import pytest

from sensor_transposition.gps.converter import (
    ecef_to_enu,
    ecef_to_geodetic,
    enu_to_ecef,
    geodetic_to_ecef,
    geodetic_to_enu,
    geodetic_to_utm,
    utm_to_geodetic,
    utm_zone_letter,
    utm_zone_number,
)

# ---------------------------------------------------------------------------
# Known reference values
# ---------------------------------------------------------------------------
# London, UK – Trafalgar Square (approximate)
_LAT = 51.5080
_LON = -0.1281
_ALT = 10.0

# ECEF for the above, computed from WGS-84 formulas.
_ECEF_X = 3_977_948.2
_ECEF_Y = -8_893.8
_ECEF_Z = 4_968_924.3

_REL_TOL = 1e-5  # 10 ppm relative tolerance for round-trip tests
_ABS_TOL_M = 0.01  # 1 cm absolute tolerance (metres) for positional tests


# ===========================================================================
# geodetic_to_ecef
# ===========================================================================


class TestGeodeticToEcef:
    def test_known_point_x(self):
        X, Y, Z = geodetic_to_ecef(_LAT, _LON, _ALT)
        assert X == pytest.approx(_ECEF_X, abs=10.0)

    def test_known_point_y(self):
        X, Y, Z = geodetic_to_ecef(_LAT, _LON, _ALT)
        assert Y == pytest.approx(_ECEF_Y, abs=10.0)

    def test_known_point_z(self):
        X, Y, Z = geodetic_to_ecef(_LAT, _LON, _ALT)
        assert Z == pytest.approx(_ECEF_Z, abs=10.0)

    def test_equator_prime_meridian(self):
        """At lat=0, lon=0, alt=0 the ECEF point is (a, 0, 0)."""
        X, Y, Z = geodetic_to_ecef(0.0, 0.0, 0.0)
        assert X == pytest.approx(6_378_137.0, rel=1e-9)
        assert Y == pytest.approx(0.0, abs=1e-3)
        assert Z == pytest.approx(0.0, abs=1e-3)

    def test_north_pole(self):
        """At lat=90, the point should be near (0, 0, b)."""
        b = 6_356_752.314  # WGS-84 semi-minor axis
        X, Y, Z = geodetic_to_ecef(90.0, 0.0, 0.0)
        assert math.sqrt(X**2 + Y**2) == pytest.approx(0.0, abs=1.0)
        assert Z == pytest.approx(b, rel=1e-6)

    def test_altitude_offset(self):
        """Adding altitude should increase the radial distance by the same amount."""
        X0, Y0, Z0 = geodetic_to_ecef(_LAT, _LON, 0.0)
        X1, Y1, Z1 = geodetic_to_ecef(_LAT, _LON, 1000.0)
        delta = math.sqrt((X1 - X0) ** 2 + (Y1 - Y0) ** 2 + (Z1 - Z0) ** 2)
        assert delta == pytest.approx(1000.0, rel=1e-4)


# ===========================================================================
# ecef_to_geodetic
# ===========================================================================


class TestEcefToGeodetic:
    def test_round_trip_lat(self):
        X, Y, Z = geodetic_to_ecef(_LAT, _LON, _ALT)
        lat, lon, alt = ecef_to_geodetic(X, Y, Z)
        assert lat == pytest.approx(_LAT, rel=_REL_TOL)

    def test_round_trip_lon(self):
        X, Y, Z = geodetic_to_ecef(_LAT, _LON, _ALT)
        lat, lon, alt = ecef_to_geodetic(X, Y, Z)
        assert lon == pytest.approx(_LON, rel=_REL_TOL)

    def test_round_trip_alt(self):
        X, Y, Z = geodetic_to_ecef(_LAT, _LON, _ALT)
        lat, lon, alt = ecef_to_geodetic(X, Y, Z)
        assert alt == pytest.approx(_ALT, abs=_ABS_TOL_M)

    def test_south_hemisphere(self):
        lat_in, lon_in, alt_in = -33.8688, 151.2093, 50.0  # Sydney
        X, Y, Z = geodetic_to_ecef(lat_in, lon_in, alt_in)
        lat, lon, alt = ecef_to_geodetic(X, Y, Z)
        assert lat == pytest.approx(lat_in, rel=_REL_TOL)
        assert lon == pytest.approx(lon_in, rel=_REL_TOL)
        assert alt == pytest.approx(alt_in, abs=_ABS_TOL_M)


# ===========================================================================
# ecef_to_enu / enu_to_ecef
# ===========================================================================


class TestEcefEnu:
    def test_origin_gives_zero_enu(self):
        """A point at the reference itself should give ENU = (0, 0, 0)."""
        X, Y, Z = geodetic_to_ecef(_LAT, _LON, _ALT)
        e, n, u = ecef_to_enu(X, Y, Z, _LAT, _LON, _ALT)
        assert e == pytest.approx(0.0, abs=_ABS_TOL_M)
        assert n == pytest.approx(0.0, abs=_ABS_TOL_M)
        assert u == pytest.approx(0.0, abs=_ABS_TOL_M)

    def test_100m_east(self):
        """Moving a point 100 m east should produce (e≈100, n≈0, u≈0)."""
        lon2 = _LON + 100.0 / (
            6_378_137.0 * math.radians(1.0) * math.cos(math.radians(_LAT))
        )
        X, Y, Z = geodetic_to_ecef(_LAT, lon2, _ALT)
        e, n, u = ecef_to_enu(X, Y, Z, _LAT, _LON, _ALT)
        assert e == pytest.approx(100.0, rel=5e-3)  # within 0.5 %
        assert n == pytest.approx(0.0, abs=0.1)
        assert u == pytest.approx(0.0, abs=0.1)

    def test_100m_up(self):
        """A point 100 m higher should produce (e≈0, n≈0, u≈100)."""
        X, Y, Z = geodetic_to_ecef(_LAT, _LON, _ALT + 100.0)
        e, n, u = ecef_to_enu(X, Y, Z, _LAT, _LON, _ALT)
        assert e == pytest.approx(0.0, abs=0.1)
        assert n == pytest.approx(0.0, abs=0.1)
        assert u == pytest.approx(100.0, rel=1e-4)

    def test_enu_to_ecef_round_trip(self):
        """enu_to_ecef(ecef_to_enu(P)) should return P."""
        X_in, Y_in, Z_in = geodetic_to_ecef(_LAT + 0.001, _LON + 0.001, _ALT + 50.0)
        e, n, u = ecef_to_enu(X_in, Y_in, Z_in, _LAT, _LON, _ALT)
        X_out, Y_out, Z_out = enu_to_ecef(e, n, u, _LAT, _LON, _ALT)
        assert X_out == pytest.approx(X_in, abs=_ABS_TOL_M)
        assert Y_out == pytest.approx(Y_in, abs=_ABS_TOL_M)
        assert Z_out == pytest.approx(Z_in, abs=_ABS_TOL_M)


# ===========================================================================
# geodetic_to_enu
# ===========================================================================


class TestGeodeticToEnu:
    def test_small_displacement(self):
        """A 100 m north displacement should give (e≈0, n≈100, u≈0)."""
        lat2 = _LAT + 100.0 / 6_378_137.0 * (180.0 / math.pi)
        e, n, u = geodetic_to_enu(lat2, _LON, _ALT, _LAT, _LON, _ALT)
        assert e == pytest.approx(0.0, abs=0.1)
        assert n == pytest.approx(100.0, rel=1e-3)
        assert u == pytest.approx(0.0, abs=0.1)

    def test_same_point_is_origin(self):
        e, n, u = geodetic_to_enu(_LAT, _LON, _ALT, _LAT, _LON, _ALT)
        assert e == pytest.approx(0.0, abs=_ABS_TOL_M)
        assert n == pytest.approx(0.0, abs=_ABS_TOL_M)
        assert u == pytest.approx(0.0, abs=_ABS_TOL_M)


# ===========================================================================
# utm_zone_number / utm_zone_letter
# ===========================================================================


class TestUtmZone:
    def test_london_zone_number(self):
        # London is in UTM zone 30
        assert utm_zone_number(_LAT, _LON) == 30

    def test_london_zone_letter(self):
        # London latitude ~51.5° → band U
        assert utm_zone_letter(_LAT) == "U"

    def test_sydney_zone_number(self):
        # Sydney ~33.9°S, 151.2°E → zone 56
        assert utm_zone_number(-33.8688, 151.2093) == 56

    def test_sydney_zone_letter(self):
        # −33.9° → band H
        assert utm_zone_letter(-33.8688) == "H"

    def test_norway_exception(self):
        # Bergen, Norway (~60.4°N, 5.3°E) lies in the Norway exception zone 32
        assert utm_zone_number(60.4, 5.3) == 32

    def test_invalid_latitude_raises(self):
        with pytest.raises(ValueError):
            utm_zone_letter(85.0)

    def test_invalid_latitude_south_raises(self):
        with pytest.raises(ValueError):
            utm_zone_letter(-81.0)


# ===========================================================================
# geodetic_to_utm / utm_to_geodetic
# ===========================================================================


class TestUtm:
    def test_london_easting_ballpark(self):
        easting, northing, zone, band = geodetic_to_utm(_LAT, _LON)
        # London, zone 30U: easting should be near 699 km
        assert easting == pytest.approx(699_000.0, abs=2000.0)

    def test_london_northing_ballpark(self):
        easting, northing, zone, band = geodetic_to_utm(_LAT, _LON)
        # London northing should be near 5 710 km
        assert northing == pytest.approx(5_710_000.0, abs=2000.0)

    def test_london_zone_and_band(self):
        _, _, zone, band = geodetic_to_utm(_LAT, _LON)
        assert zone == 30
        assert band == "U"

    def test_southern_hemisphere_northing(self):
        # Sydney – northing should include 10 000 000 m false northing
        easting, northing, zone, band = geodetic_to_utm(-33.8688, 151.2093)
        assert northing > 6_000_000.0  # false northing lifts it above equator

    def test_round_trip_lat(self):
        easting, northing, zone, band = geodetic_to_utm(_LAT, _LON)
        lat_out, lon_out = utm_to_geodetic(easting, northing, zone, band)
        assert lat_out == pytest.approx(_LAT, abs=1e-5)

    def test_round_trip_lon(self):
        easting, northing, zone, band = geodetic_to_utm(_LAT, _LON)
        lat_out, lon_out = utm_to_geodetic(easting, northing, zone, band)
        assert lon_out == pytest.approx(_LON, abs=1e-5)

    def test_round_trip_sydney(self):
        lat_in, lon_in = -33.8688, 151.2093
        easting, northing, zone, band = geodetic_to_utm(lat_in, lon_in)
        lat_out, lon_out = utm_to_geodetic(easting, northing, zone, band)
        assert lat_out == pytest.approx(lat_in, abs=1e-5)
        assert lon_out == pytest.approx(lon_in, abs=1e-5)

    def test_round_trip_new_york(self):
        lat_in, lon_in = 40.7128, -74.0060
        easting, northing, zone, band = geodetic_to_utm(lat_in, lon_in)
        lat_out, lon_out = utm_to_geodetic(easting, northing, zone, band)
        assert lat_out == pytest.approx(lat_in, abs=1e-5)
        assert lon_out == pytest.approx(lon_in, abs=1e-5)
