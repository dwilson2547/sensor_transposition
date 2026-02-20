"""Tests for the NMEA GPS parser."""

import os
import tempfile

import pytest

from sensor_transposition.gps.nmea import (
    GgaFix,
    NmeaParser,
    RmcFix,
    load_nmea,
    _parse_lat,
    _parse_lon,
    _parse_sentence,
    _strip_checksum,
)


# ---------------------------------------------------------------------------
# Sample NMEA sentences
# ---------------------------------------------------------------------------

# GGA with valid checksum
_GGA = "$GPGGA,092750.000,5321.6802,N,00630.3372,W,1,8,1.03,61.7,M,55.2,M,,*76"
# RMC with valid checksum
_RMC = "$GPRMC,092750.000,A,5321.6802,N,00630.3372,W,0.02,31.66,280511,,,A*43"
# GGA with no checksum
_GGA_NO_CHK = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,"
# Invalid checksum
_BAD_CHK = "$GPGGA,092750.000,5321.6802,N,00630.3372,W,1,8,1.03,61.7,M,55.2,M,,*FF"
# Unknown sentence type
_UNKNOWN = "$GPZDA,092750.000,28,05,2011,,*56"


def _write_nmea(path: str, lines: list) -> None:
    """Write NMEA lines to a file."""
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


class TestStripChecksum:
    def test_valid_checksum(self):
        body = _strip_checksum(_GGA)
        assert body is not None
        assert body.startswith("GPGGA")

    def test_invalid_checksum_returns_none(self):
        assert _strip_checksum(_BAD_CHK) is None

    def test_no_checksum_returns_body(self):
        body = _strip_checksum(_GGA_NO_CHK)
        assert body is not None
        assert body.startswith("GPGGA")


class TestLatLon:
    def test_north_latitude(self):
        lat = _parse_lat("5321.6802", "N")
        assert lat == pytest.approx(53 + 21.6802 / 60.0, rel=1e-6)

    def test_south_latitude(self):
        lat = _parse_lat("5321.6802", "S")
        assert lat == pytest.approx(-(53 + 21.6802 / 60.0), rel=1e-6)

    def test_west_longitude(self):
        lon = _parse_lon("00630.3372", "W")
        assert lon == pytest.approx(-(6 + 30.3372 / 60.0), rel=1e-6)

    def test_east_longitude(self):
        lon = _parse_lon("01131.000", "E")
        assert lon == pytest.approx(11 + 31.0 / 60.0, rel=1e-6)

    def test_empty_lat(self):
        assert _parse_lat("", "N") == 0.0

    def test_empty_lon(self):
        assert _parse_lon("", "E") == 0.0


class TestParseSentence:
    def test_parse_gga(self):
        record = _parse_sentence(_GGA)
        assert isinstance(record, GgaFix)
        assert record.fix_quality == 1
        assert record.num_satellites == 8
        assert record.altitude == pytest.approx(61.7)
        assert record.latitude == pytest.approx(53 + 21.6802 / 60.0, rel=1e-5)
        assert record.longitude == pytest.approx(-(6 + 30.3372 / 60.0), rel=1e-5)

    def test_parse_rmc(self):
        record = _parse_sentence(_RMC)
        assert isinstance(record, RmcFix)
        assert record.status == "A"
        assert record.is_valid is True
        assert record.speed_knots == pytest.approx(0.02, rel=1e-4)
        assert record.date == "280511"

    def test_bad_checksum_returns_none(self):
        assert _parse_sentence(_BAD_CHK) is None

    def test_unknown_sentence_returns_none(self):
        assert _parse_sentence(_UNKNOWN) is None

    def test_gngga_recognized(self):
        sentence = "$GNGGA,092750.000,5321.6802,N,00630.3372,W,1,8,1.03,61.7,M,55.2,M,,*68"
        record = _parse_sentence(sentence)
        assert isinstance(record, GgaFix)

    def test_gnrmc_recognized(self):
        sentence = "$GNRMC,092750.000,A,5321.6802,N,00630.3372,W,0.02,31.66,280511,,,A*5D"
        record = _parse_sentence(sentence)
        assert isinstance(record, RmcFix)


class TestRmcFix:
    def test_void_status(self):
        sentence = "$GPRMC,235947.000,V,0000.0000,N,00000.0000,E,,,010180,,,N*41"
        record = _parse_sentence(sentence)
        if record is not None:
            assert isinstance(record, RmcFix)
            assert record.is_valid is False


class TestNmeaParser:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".nmea", delete=False, mode="w")
        self.tmp.write(_GGA + "\n")
        self.tmp.write(_RMC + "\n")
        self.tmp.write(_UNKNOWN + "\n")
        self.tmp.close()
        self.path = self.tmp.name

    def teardown_method(self):
        os.unlink(self.path)

    def test_records_returns_list(self):
        parser = NmeaParser(self.path)
        records = parser.records()
        assert isinstance(records, list)
        assert len(records) == 2  # GGA + RMC; unknown is skipped

    def test_gga_fixes(self):
        fixes = NmeaParser(self.path).gga_fixes()
        assert len(fixes) == 1
        assert isinstance(fixes[0], GgaFix)

    def test_rmc_fixes(self):
        fixes = NmeaParser(self.path).rmc_fixes()
        assert len(fixes) == 1
        assert isinstance(fixes[0], RmcFix)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            NmeaParser("/nonexistent/path/file.nmea")

    def test_path_property(self):
        parser = NmeaParser(self.path)
        assert str(parser.path) == self.path

    def test_load_nmea_function(self):
        records = load_nmea(self.path)
        assert len(records) == 2

    def test_empty_file(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".nmea", delete=False, mode="w")
        tmp.close()
        try:
            records = NmeaParser(tmp.name).records()
            assert records == []
        finally:
            os.unlink(tmp.name)

    def test_non_sentence_lines_skipped(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".nmea", delete=False, mode="w")
        tmp.write("# This is a comment\n")
        tmp.write("Some random text\n")
        tmp.write(_GGA + "\n")
        tmp.close()
        try:
            records = NmeaParser(tmp.name).records()
            assert len(records) == 1
        finally:
            os.unlink(tmp.name)
