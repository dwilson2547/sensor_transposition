"""
gps/rtcm.py

Minimal pure-Python parser for selected RTCM 3.x message types used in
RTK-GPS correction streams.

Supported message types
-----------------------
* **MT1005** – Stationary RTK Reference Station ARP (base-station antenna
  position in ECEF, used to anchor the RTK correction baseline).
* **MSM4** – Multiple Signal Messages type 4 (MT1074/1084/1094/1124) carrying
  compact pseudorange and phase observations for GPS, GLONASS, Galileo, and
  BeiDou constellations.
* **MSM7** – Multiple Signal Messages type 7 (MT1077/1087/1097/1127) – the
  high-resolution variant of MSM4 with extended precision fields.

The parser is intentionally lenient: unknown or malformed frames are skipped
without raising an exception.  Only the subset of fields required for
position/correction-stream ingestion is extracted.

References
----------
* RTCM 10403.3 Standard (RTCM Special Committee 104)
* RTCM 10410.1 (NTRIP / CRS transport)

Typical use
-----------
::

    from sensor_transposition.gps.rtcm import RtcmParser, Rtcm1005, RtcmMsm

    with open("corrections.rtcm3", "rb") as fh:
        frames = RtcmParser(fh).parse_all()

    for frame in frames:
        if isinstance(frame, Rtcm1005):
            print("Base station ECEF:", frame.x_m, frame.y_m, frame.z_m)
        elif isinstance(frame, RtcmMsm):
            print(f"MSM{frame.msm_type} constellation={frame.constellation} "
                  f"sat_mask=0x{frame.satellite_mask:016x}")
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from io import RawIOBase
from typing import List, Optional, Union

# ---------------------------------------------------------------------------
# RTCM 3.x framing constants
# ---------------------------------------------------------------------------

_RTCM_PREAMBLE = 0xD3
_HEADER_LEN = 3        # preamble (1) + reserved (6 bits) + length (10 bits)
_CRC_LEN = 3           # CRC-24Q appended after payload

# ---------------------------------------------------------------------------
# CRC-24Q (used by all RTCM 3.x frames)
# ---------------------------------------------------------------------------

_CRC24_TABLE: List[int] = []


def _build_crc24_table() -> None:
    for i in range(256):
        crc = i << 16
        for _ in range(8):
            crc <<= 1
            if crc & 0x01000000:
                crc ^= 0x01864CFB
        _CRC24_TABLE.append(crc & 0xFFFFFF)


_build_crc24_table()


def _crc24q(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc = ((crc << 8) ^ _CRC24_TABLE[((crc >> 16) ^ byte) & 0xFF]) & 0xFFFFFF
    return crc


# ---------------------------------------------------------------------------
# Bit-level accessor
# ---------------------------------------------------------------------------


class _BitReader:
    """Read arbitrary-width unsigned/signed fields from a byte buffer."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0  # bit position

    def read_u(self, n: int) -> int:
        """Read *n* bits as an unsigned integer."""
        val = 0
        for _ in range(n):
            byte_idx = self._pos >> 3
            bit_idx = 7 - (self._pos & 7)
            val = (val << 1) | ((self._data[byte_idx] >> bit_idx) & 1)
            self._pos += 1
        return val

    def read_s(self, n: int) -> int:
        """Read *n* bits as a two's-complement signed integer."""
        val = self.read_u(n)
        if val >= (1 << (n - 1)):
            val -= 1 << n
        return val

    @property
    def bits_remaining(self) -> int:
        return len(self._data) * 8 - self._pos


# ---------------------------------------------------------------------------
# Data classes for parsed messages
# ---------------------------------------------------------------------------


@dataclass
class Rtcm1005:
    """Parsed RTCM MT1005 – Stationary RTK Reference Station ARP.

    Contains the ECEF X/Y/Z position of the base-station antenna reference
    point (ARP) in metres.  Pass these values to
    :func:`~sensor_transposition.gps.converter.ecef_to_enu` to convert the
    base-station position into your local ENU frame.

    Attributes:
        message_type: Always ``1005``.
        station_id: Reference station identifier (0–4095).
        itrf_realization_year: ITRF realisation year (0 = unspecified).
        x_m: ECEF X coordinate of the ARP in metres.
        y_m: ECEF Y coordinate of the ARP in metres.
        z_m: ECEF Z coordinate of the ARP in metres.
        antenna_height_m: Antenna height above the ARP in metres.
    """

    message_type: int
    station_id: int
    itrf_realization_year: int
    x_m: float
    y_m: float
    z_m: float
    antenna_height_m: float = 0.0


@dataclass
class RtcmMsm:
    """Parsed RTCM MSM4 or MSM7 header fields.

    MSM (Multiple Signal Message) frames carry raw GNSS observations
    (pseudorange and carrier-phase) from a single constellation.  This
    dataclass captures the header fields most useful for ingestion; the raw
    signal data arrays are provided but their exact interpretation depends on
    the constellation and signal type.

    Attributes:
        message_type: RTCM message number (e.g. ``1074``, ``1077``).
        msm_type: Either ``4`` or ``7``.
        constellation: Short string identifying the GNSS constellation:
            ``"GPS"``, ``"GLONASS"``, ``"Galileo"``, or ``"BeiDou"``.
        station_id: Reference station identifier.
        epoch_ms: Constellation-specific epoch time (ms) from the MSM header.
        satellite_mask: 64-bit bitmask indicating which satellite PRNs are
            present in the message.
        signal_mask: 32-bit bitmask indicating which signal types are present.
        pseudoranges_m: List of pseudorange values in metres (one per
            satellite × signal cell, or empty if decoding was skipped).
        carrier_phases_cycles: List of carrier-phase values in cycles (MSM7
            only; empty for MSM4 or if decoding was skipped).
    """

    message_type: int
    msm_type: int
    constellation: str
    station_id: int
    epoch_ms: int
    satellite_mask: int
    signal_mask: int
    pseudoranges_m: List[float] = field(default_factory=list)
    carrier_phases_cycles: List[float] = field(default_factory=list)


# Union type for all supported parsed messages
RtcmMessage = Union[Rtcm1005, RtcmMsm]

# ---------------------------------------------------------------------------
# MSM constellation lookup
# ---------------------------------------------------------------------------

_MSM_CONSTELLATIONS = {
    # MT1074-1077: GPS
    1074: "GPS",   1075: "GPS",   1076: "GPS",   1077: "GPS",
    # MT1084-1087: GLONASS
    1084: "GLONASS", 1085: "GLONASS", 1086: "GLONASS", 1087: "GLONASS",
    # MT1094-1097: Galileo
    1094: "Galileo", 1095: "Galileo", 1096: "Galileo", 1097: "Galileo",
    # MT1124-1127: BeiDou
    1124: "BeiDou",  1125: "BeiDou",  1126: "BeiDou",  1127: "BeiDou",
}

# MSM sub-type: 4 or 7
_MSM_SUBTYPE = {
    1074: 4, 1077: 7,
    1084: 4, 1087: 7,
    1094: 4, 1097: 7,
    1124: 4, 1127: 7,
}

# ---------------------------------------------------------------------------
# Parsing functions
# ---------------------------------------------------------------------------

_P2_10 = 1.0 / 1024.0            # 2^-10
_P2_24 = 1.0 / 16_777_216.0      # 2^-24
_P2_29 = 1.0 / 536_870_912.0     # 2^-29
_LIGHT_MS = 299_792.458           # speed of light in m/ms
# GPS L1 carrier wavelength: c / f_L1 = 299 792 458 / 1 575 420 000 ≈ 0.1903 m
_GPS_L1_WAVELENGTH_M = 299_792_458.0 / 1_575_420_000.0


def _parse_1005(payload: bytes) -> Optional[Rtcm1005]:
    """Parse an MT1005 payload and return an :class:`Rtcm1005`."""
    if len(payload) < 19:
        return None
    br = _BitReader(payload)
    msg_type = br.read_u(12)
    if msg_type != 1005:
        return None
    station_id = br.read_u(12)
    itrf_year = br.read_u(6)
    br.read_u(4)    # GPS/GLONASS/Galileo/Reference-Station indicator bits
    x_raw = br.read_s(38)
    br.read_u(2)    # oscillator / reserved
    y_raw = br.read_s(38)
    br.read_u(2)
    z_raw = br.read_s(38)
    x_m = x_raw * 0.0001
    y_m = y_raw * 0.0001
    z_m = z_raw * 0.0001
    return Rtcm1005(
        message_type=1005,
        station_id=station_id,
        itrf_realization_year=itrf_year,
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
    )


def _parse_msm(payload: bytes, msg_type: int) -> Optional[RtcmMsm]:
    """Parse an MSM4 or MSM7 payload and return an :class:`RtcmMsm`."""
    if msg_type not in _MSM_CONSTELLATIONS:
        return None
    if len(payload) < 14:
        return None

    constellation = _MSM_CONSTELLATIONS[msg_type]
    msm_subtype = _MSM_SUBTYPE.get(msg_type, 4)

    br = _BitReader(payload)
    actual_type = br.read_u(12)
    if actual_type != msg_type:
        return None
    station_id = br.read_u(12)
    epoch_ms = br.read_u(30)    # GNSS epoch time (ms, constellation-specific)
    br.read_u(1)                # multiple message flag
    br.read_u(3)                # issue of data station
    br.read_u(7)                # reserved / clock steering / external clock
    br.read_u(2)                # GNSS divergence-free smoothing indicator
    br.read_u(3)                # smoothing interval

    satellite_mask = br.read_u(64)
    signal_mask = br.read_u(32)

    # Cell mask: one bit per (satellite × signal) combination that is present.
    n_sats = bin(satellite_mask).count("1")
    n_sigs = bin(signal_mask).count("1")
    n_cells = n_sats * n_sigs

    if br.bits_remaining < n_cells:
        # Not enough bits for cell mask — return header-only result
        return RtcmMsm(
            message_type=msg_type,
            msm_type=msm_subtype,
            constellation=constellation,
            station_id=station_id,
            epoch_ms=epoch_ms,
            satellite_mask=satellite_mask,
            signal_mask=signal_mask,
        )

    cell_mask = br.read_u(n_cells)
    n_present = bin(cell_mask).count("1")

    pseudoranges: List[float] = []
    carrier_phases: List[float] = []

    try:
        if msm_subtype == 4:
            # MSM4: integer pseudoranges (10 bits) + fractional (15 bits) per sat
            int_ranges = [br.read_u(8) for _ in range(n_sats)]
            frac_ranges = [br.read_u(10) for _ in range(n_present)]
            for i in range(n_present):
                sat_idx = _cell_to_sat(i, n_sats, n_sigs, cell_mask)
                pr_m = (int_ranges[sat_idx] * 1.0 + frac_ranges[i] * _P2_10) * _LIGHT_MS
                pseudoranges.append(pr_m)
        else:
            # MSM7: integer pseudoranges (8 bits sat-level) + fine (20 bits cell)
            int_ranges = [br.read_u(8) for _ in range(n_sats)]
            frac_ranges_u = [br.read_u(10) for _ in range(n_sats)]
            fine_pr = [br.read_s(20) for _ in range(n_present)]
            fine_cp = [br.read_s(24) for _ in range(n_present)]
            for i in range(n_present):
                sat_idx = _cell_to_sat(i, n_sats, n_sigs, cell_mask)
                rough = (int_ranges[sat_idx] + frac_ranges_u[sat_idx] * _P2_10)
                pr_m = (rough + fine_pr[i] * _P2_29) * _LIGHT_MS
                cp_cyc = rough * _LIGHT_MS / _GPS_L1_WAVELENGTH_M + fine_cp[i] * _P2_24
                pseudoranges.append(pr_m)
                carrier_phases.append(cp_cyc)
    except (IndexError, struct.error):
        pass  # partial decode – return what we have

    return RtcmMsm(
        message_type=msg_type,
        msm_type=msm_subtype,
        constellation=constellation,
        station_id=station_id,
        epoch_ms=epoch_ms,
        satellite_mask=satellite_mask,
        signal_mask=signal_mask,
        pseudoranges_m=pseudoranges,
        carrier_phases_cycles=carrier_phases,
    )


def _cell_to_sat(cell_idx: int, n_sats: int, n_sigs: int, cell_mask: int) -> int:
    """Return the satellite index (0-based) for the *cell_idx*-th present cell."""
    present = 0
    total_cells = n_sats * n_sigs
    for c in range(total_cells):
        if cell_mask >> (total_cells - 1 - c) & 1:
            if present == cell_idx:
                return c // n_sigs
            present += 1
    return 0


# ---------------------------------------------------------------------------
# RtcmParser
# ---------------------------------------------------------------------------


class RtcmParser:
    """Parser for RTCM 3.x binary correction streams.

    Reads from any file-like object opened in binary mode and yields
    :class:`Rtcm1005` or :class:`RtcmMsm` instances for each successfully
    decoded frame.  Unsupported message types and frames with CRC errors are
    silently skipped.

    Args:
        stream: A binary-mode file-like object (e.g. ``open("file.rtcm3", "rb")``
            or any ``io.RawIOBase`` / ``io.BufferedIOBase``).

    Example::

        with open("corrections.rtcm3", "rb") as fh:
            parser = RtcmParser(fh)
            for msg in parser.messages():
                if isinstance(msg, Rtcm1005):
                    print("Base ECEF:", msg.x_m, msg.y_m, msg.z_m)
    """

    def __init__(self, stream: RawIOBase) -> None:
        self._stream = stream

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def messages(self):
        """Iterate over successfully decoded RTCM messages.

        Yields:
            :class:`Rtcm1005` or :class:`RtcmMsm` instances in stream order.
        """
        while True:
            frame = self._read_frame()
            if frame is None:
                return
            msg = self._decode_frame(frame)
            if msg is not None:
                yield msg

    def parse_all(self) -> List[RtcmMessage]:
        """Parse the entire stream and return a list of decoded messages.

        Returns:
            List of :class:`Rtcm1005` and/or :class:`RtcmMsm` instances.
        """
        return list(self.messages())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_frame(self) -> Optional[bytes]:
        """Read the next complete RTCM frame (header + payload + CRC).

        Returns ``None`` when the stream is exhausted.  Skips bytes that do
        not start with the RTCM preamble ``0xD3``.
        """
        while True:
            b = self._stream.read(1)
            if not b:
                return None
            if b[0] != _RTCM_PREAMBLE:
                continue
            # Read the 2-byte length field (bits 6–15 are the payload length)
            hdr_rest = self._stream.read(2)
            if len(hdr_rest) < 2:
                return None
            length = ((hdr_rest[0] & 0x03) << 8) | hdr_rest[1]
            payload = self._stream.read(length)
            if len(payload) < length:
                return None
            crc_bytes = self._stream.read(_CRC_LEN)
            if len(crc_bytes) < _CRC_LEN:
                return None
            # Validate CRC-24Q over preamble + 2-byte header + payload
            frame = bytes([_RTCM_PREAMBLE]) + hdr_rest + payload
            expected_crc = _crc24q(frame)
            actual_crc = (
                (crc_bytes[0] << 16) | (crc_bytes[1] << 8) | crc_bytes[2]
            )
            if expected_crc != actual_crc:
                # CRC mismatch — skip this frame
                continue
            return payload

    @staticmethod
    def _decode_frame(payload: bytes) -> Optional[RtcmMessage]:
        """Attempt to decode a payload and return a typed dataclass or None."""
        if len(payload) < 2:
            return None
        # Extract 12-bit message type from the first two bytes
        msg_type = (payload[0] << 4) | (payload[1] >> 4)
        if msg_type == 1005:
            return _parse_1005(payload)
        if msg_type in _MSM_CONSTELLATIONS:
            return _parse_msm(payload, msg_type)
        return None


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def parse_rtcm_file(path: str) -> List[RtcmMessage]:
    """Parse all supported RTCM 3.x messages from a binary file.

    Args:
        path: Path to an RTCM 3.x binary file (typically ``.rtcm3`` or
            ``.eph``).

    Returns:
        List of :class:`Rtcm1005` and :class:`RtcmMsm` instances.

    Example::

        from sensor_transposition.gps.rtcm import parse_rtcm_file, Rtcm1005

        messages = parse_rtcm_file("corrections.rtcm3")
        base_pos = next(m for m in messages if isinstance(m, Rtcm1005))
        print(f"Base station: X={base_pos.x_m:.4f} m")
    """
    with open(path, "rb") as fh:
        return RtcmParser(fh).parse_all()
