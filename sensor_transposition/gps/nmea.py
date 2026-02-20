"""
gps/nmea.py

Parser for NMEA 0183 GPS/GNSS sentences.

Supported sentence types
------------------------
* **GGA** – Global Positioning System Fix Data (position, altitude, fix quality,
  number of satellites, HDOP).
* **RMC** – Recommended Minimum Navigation Information (position, speed over
  ground, course over ground, date/time, validity).

The parser is lenient: it skips unrecognised or malformed sentences without
raising an exception.

Output is a list of :class:`GgaFix` or :class:`RmcFix` dataclass instances,
one per successfully parsed sentence.

References
----------
* NMEA 0183 Standard (version 4.11)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Union


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GgaFix:
    """Parsed GGA (GPS Fix Data) sentence.

    Attributes:
        timestamp: UTC time as a string in ``HHMMSS.ss`` format.
        latitude: Latitude in decimal degrees (negative = South).
        longitude: Longitude in decimal degrees (negative = West).
        fix_quality: Fix quality indicator (0 = invalid, 1 = GPS fix,
            2 = DGPS fix, etc.).
        num_satellites: Number of satellites in use.
        hdop: Horizontal dilution of precision.
        altitude: Antenna altitude above mean sea level in metres.
        geoid_separation: Geoid separation (undulation) in metres.
    """

    timestamp: str
    latitude: float
    longitude: float
    fix_quality: int
    num_satellites: int
    hdop: float
    altitude: float
    geoid_separation: float


@dataclass
class RmcFix:
    """Parsed RMC (Recommended Minimum Navigation) sentence.

    Attributes:
        timestamp: UTC time as a string in ``HHMMSS.ss`` format.
        status: Status character (``'A'`` = active/valid, ``'V'`` = void).
        latitude: Latitude in decimal degrees (negative = South).
        longitude: Longitude in decimal degrees (negative = West).
        speed_knots: Speed over ground in knots.
        course: True course over ground in degrees (0–360).
        date: Date string in ``DDMMYY`` format.
        is_valid: ``True`` when the fix is valid (status == ``'A'``).
    """

    timestamp: str
    status: str
    latitude: float
    longitude: float
    speed_knots: float
    course: float
    date: str

    @property
    def is_valid(self) -> bool:
        """Return ``True`` when the fix is active/valid."""
        return self.status == "A"


# Type alias for any parsed fix record
NmeaRecord = Union[GgaFix, RmcFix]


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


class NmeaParser:
    """Parser for NMEA 0183 text files.

    Args:
        path: Path to the NMEA ``.nmea`` or ``.txt`` file.

    Example::

        parser = NmeaParser("gps_log.nmea")
        for record in parser.records():
            if isinstance(record, GgaFix):
                print(record.latitude, record.longitude, record.altitude)
    """

    def __init__(self, path: str | os.PathLike) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"NMEA file not found: {self._path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def records(self) -> List[NmeaRecord]:
        """Parse all recognised sentences from the file.

        Returns:
            List of :class:`GgaFix` and :class:`RmcFix` instances in file
            order.  Unrecognised or malformed sentences are silently skipped.
        """
        results: List[NmeaRecord] = []
        for sentence in self._iter_sentences():
            record = _parse_sentence(sentence)
            if record is not None:
                results.append(record)
        return results

    def gga_fixes(self) -> List[GgaFix]:
        """Return only the GGA fix records from the file."""
        return [r for r in self.records() if isinstance(r, GgaFix)]

    def rmc_fixes(self) -> List[RmcFix]:
        """Return only the RMC fix records from the file."""
        return [r for r in self.records() if isinstance(r, RmcFix)]

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _iter_sentences(self) -> Iterator[str]:
        """Yield raw (stripped) NMEA sentence strings from the file."""
        for line in self._path.read_text(encoding="ascii", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("$"):
                yield line


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def load_nmea(path: str | os.PathLike) -> List[NmeaRecord]:
    """Load all recognisable NMEA records from a file.

    Args:
        path: Path to the NMEA ``.nmea`` or ``.txt`` file.

    Returns:
        List of :class:`GgaFix` and :class:`RmcFix` instances.
    """
    return NmeaParser(path).records()


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------


def _strip_checksum(sentence: str) -> Optional[str]:
    """Strip the ``*HH`` checksum suffix and validate it.

    Returns the sentence body (between ``$`` and ``*``) if the checksum is
    valid or absent, or ``None`` if the checksum is present but incorrect.
    """
    if "*" in sentence:
        body, checksum_str = sentence[1:].rsplit("*", 1)
        try:
            expected = int(checksum_str[:2], 16)
        except ValueError:
            return None
        actual = 0
        for ch in body:
            actual ^= ord(ch)
        if actual != expected:
            return None
        return body
    return sentence[1:]


def _parse_lat(value: str, hemisphere: str) -> float:
    """Convert NMEA latitude string (``DDMM.mmmm``) to decimal degrees."""
    if not value:
        return 0.0
    deg = float(value[:2])
    minutes = float(value[2:])
    decimal = deg + minutes / 60.0
    return -decimal if hemisphere.upper() == "S" else decimal


def _parse_lon(value: str, hemisphere: str) -> float:
    """Convert NMEA longitude string (``DDDMM.mmmm``) to decimal degrees."""
    if not value:
        return 0.0
    deg = float(value[:3])
    minutes = float(value[3:])
    decimal = deg + minutes / 60.0
    return -decimal if hemisphere.upper() == "W" else decimal


def _parse_sentence(sentence: str) -> Optional[NmeaRecord]:
    """Parse a single NMEA sentence string.

    Returns a :class:`GgaFix`, :class:`RmcFix`, or ``None`` if the sentence
    type is not supported or parsing fails.
    """
    body = _strip_checksum(sentence)
    if body is None:
        return None

    fields = body.split(",")
    if not fields:
        return None

    sentence_type = fields[0].upper()

    try:
        if sentence_type in ("GPGGA", "GNGGA"):
            return _parse_gga(fields)
        if sentence_type in ("GPRMC", "GNRMC"):
            return _parse_rmc(fields)
    except (IndexError, ValueError):
        return None

    return None


def _parse_gga(fields: List[str]) -> Optional[GgaFix]:
    """Parse a GGA sentence field list (already split on commas)."""
    # GGA: $GPGGA,hhmmss.ss,llll.ll,a,yyyyy.yy,a,x,xx,x.x,x.x,M,x.x,M,x.x,xxxx
    if len(fields) < 15:
        return None
    lat = _parse_lat(fields[2], fields[3])
    lon = _parse_lon(fields[4], fields[5])
    return GgaFix(
        timestamp=fields[1],
        latitude=lat,
        longitude=lon,
        fix_quality=int(fields[6]) if fields[6] else 0,
        num_satellites=int(fields[7]) if fields[7] else 0,
        hdop=float(fields[8]) if fields[8] else 0.0,
        altitude=float(fields[9]) if fields[9] else 0.0,
        geoid_separation=float(fields[11]) if fields[11] else 0.0,
    )


def _parse_rmc(fields: List[str]) -> Optional[RmcFix]:
    """Parse an RMC sentence field list (already split on commas)."""
    # RMC: $GPRMC,hhmmss.ss,A,llll.ll,a,yyyyy.yy,a,x.x,x.x,ddmmyy,x.x,a
    if len(fields) < 10:
        return None
    lat = _parse_lat(fields[3], fields[4])
    lon = _parse_lon(fields[5], fields[6])
    return RmcFix(
        timestamp=fields[1],
        status=fields[2],
        latitude=lat,
        longitude=lon,
        speed_knots=float(fields[7]) if fields[7] else 0.0,
        course=float(fields[8]) if fields[8] else 0.0,
        date=fields[9],
    )
