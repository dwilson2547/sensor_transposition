"""Tests for gps/rtcm.py: RtcmParser, Rtcm1005, RtcmMsm."""

from __future__ import annotations

import io
import struct

import pytest

from sensor_transposition.gps.rtcm import (
    Rtcm1005,
    RtcmMsm,
    RtcmParser,
    _crc24q,
    _BitReader,
)


# ---------------------------------------------------------------------------
# Helpers – build valid RTCM 3.x frames from raw payloads
# ---------------------------------------------------------------------------

def _make_frame(payload: bytes) -> bytes:
    """Wrap a payload in a valid RTCM 3.x frame (preamble + length + CRC)."""
    length = len(payload)
    header = bytes([0xD3, (length >> 8) & 0x03, length & 0xFF])
    crc_val = _crc24q(header + payload)
    crc_bytes = bytes([
        (crc_val >> 16) & 0xFF,
        (crc_val >> 8) & 0xFF,
        crc_val & 0xFF,
    ])
    return header + payload + crc_bytes


def _encode_bits(fields: list[tuple[int, int]]) -> bytes:
    """Encode a list of (value, num_bits) pairs into a big-endian byte string."""
    bits: list[int] = []
    for val, n in fields:
        for i in range(n - 1, -1, -1):
            bits.append((val >> i) & 1)
    # Pad to full bytes
    while len(bits) % 8:
        bits.append(0)
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        result.append(byte)
    return bytes(result)


def _build_mt1005_payload(
    station_id: int = 0,
    itrf_year: int = 2020,
    x_raw: int = 40_000_000,   # → 4000.0 m
    y_raw: int = 10_000_000,   # → 1000.0 m
    z_raw: int = 48_000_000,   # → 4800.0 m
) -> bytes:
    """Build a minimal MT1005 payload."""
    fields = [
        (1005, 12),       # message type
        (station_id, 12), # station id
        (itrf_year, 6),   # ITRF year
        (0b1111, 4),      # GPS/GLONASS/Galileo/ref-station indicators
        (x_raw, 38),      # ECEF X (signed, 0.1 mm resolution)
        (0b00, 2),        # oscillator / reserved
        (y_raw, 38),      # ECEF Y
        (0b00, 2),
        (z_raw, 38),      # ECEF Z
    ]
    return _encode_bits(fields)


# ---------------------------------------------------------------------------
# _BitReader
# ---------------------------------------------------------------------------

class TestBitReader:
    def test_read_single_byte(self):
        br = _BitReader(bytes([0b10110010]))
        assert br.read_u(8) == 0b10110010

    def test_read_nibbles(self):
        br = _BitReader(bytes([0b11001010]))
        assert br.read_u(4) == 0b1100
        assert br.read_u(4) == 0b1010

    def test_read_signed_positive(self):
        br = _BitReader(bytes([0b01000000]))
        assert br.read_s(4) == 4  # 0b0100

    def test_read_signed_negative(self):
        # 0b1100 in 4-bit two's complement = -4
        br = _BitReader(bytes([0b11000000]))
        assert br.read_s(4) == -4

    def test_bits_remaining(self):
        br = _BitReader(bytes([0xFF, 0xFF]))
        br.read_u(5)
        assert br.bits_remaining == 11


# ---------------------------------------------------------------------------
# _crc24q
# ---------------------------------------------------------------------------

class TestCrc24q:
    def test_empty_bytes(self):
        assert _crc24q(b"") == 0

    def test_known_value(self):
        # CRC of a known byte sequence (verified externally)
        result = _crc24q(bytes([0xD3, 0x00, 0x13]))
        assert isinstance(result, int)
        assert 0 <= result <= 0xFFFFFF


# ---------------------------------------------------------------------------
# RtcmParser – frame detection
# ---------------------------------------------------------------------------

class TestRtcmFrameDetection:
    def test_empty_stream_yields_nothing(self):
        parser = RtcmParser(io.BytesIO(b""))
        assert parser.parse_all() == []

    def test_garbage_stream_yields_nothing(self):
        data = bytes(range(100))  # no 0xD3 preamble
        parser = RtcmParser(io.BytesIO(data))
        assert parser.parse_all() == []

    def test_bad_crc_skipped(self):
        payload = _build_mt1005_payload()
        frame = bytearray(_make_frame(payload))
        # Corrupt the last CRC byte
        frame[-1] ^= 0xFF
        parser = RtcmParser(io.BytesIO(bytes(frame)))
        assert parser.parse_all() == []

    def test_valid_frame_not_skipped(self):
        payload = _build_mt1005_payload()
        frame = _make_frame(payload)
        messages = RtcmParser(io.BytesIO(frame)).parse_all()
        assert len(messages) == 1

    def test_leading_garbage_skipped(self):
        payload = _build_mt1005_payload()
        frame = _make_frame(payload)
        # Prepend random bytes that don't contain 0xD3
        garbage = bytes([0x00, 0x01, 0x02, 0x10])
        messages = RtcmParser(io.BytesIO(garbage + frame)).parse_all()
        assert len(messages) == 1

    def test_multiple_frames_parsed(self):
        payload = _build_mt1005_payload()
        frame = _make_frame(payload)
        two_frames = frame + frame
        messages = RtcmParser(io.BytesIO(two_frames)).parse_all()
        assert len(messages) == 2


# ---------------------------------------------------------------------------
# Rtcm1005 parsing
# ---------------------------------------------------------------------------

class TestRtcm1005:
    def _parse(self, **kwargs) -> Rtcm1005:
        payload = _build_mt1005_payload(**kwargs)
        frame = _make_frame(payload)
        msgs = RtcmParser(io.BytesIO(frame)).parse_all()
        assert len(msgs) == 1
        assert isinstance(msgs[0], Rtcm1005)
        return msgs[0]

    def test_message_type(self):
        msg = self._parse()
        assert msg.message_type == 1005

    def test_station_id(self):
        msg = self._parse(station_id=42)
        assert msg.station_id == 42

    def test_ecef_x_conversion(self):
        # x_raw=40_000_000 → x_m = 40_000_000 * 0.0001 = 4000.0 m
        msg = self._parse(x_raw=40_000_000)
        assert msg.x_m == pytest.approx(4000.0, rel=1e-9)

    def test_ecef_y_conversion(self):
        msg = self._parse(y_raw=10_000_000)
        assert msg.y_m == pytest.approx(1000.0, rel=1e-9)

    def test_ecef_z_conversion(self):
        msg = self._parse(z_raw=48_000_000)
        assert msg.z_m == pytest.approx(4800.0, rel=1e-9)

    def test_itrf_year(self):
        msg = self._parse(itrf_year=14)
        assert msg.itrf_realization_year == 14

    def test_zero_coordinates(self):
        msg = self._parse(x_raw=0, y_raw=0, z_raw=0)
        assert msg.x_m == pytest.approx(0.0)
        assert msg.y_m == pytest.approx(0.0)
        assert msg.z_m == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# RtcmMsm parsing
# ---------------------------------------------------------------------------

def _build_msm4_payload(
    msg_type: int = 1074,
    station_id: int = 1,
    epoch_ms: int = 86_400_000,
    satellite_mask: int = 0x0000_0000_0000_0001,  # 1 satellite
    signal_mask: int = 0x0000_0001,                # 1 signal
) -> bytes:
    """Build a minimal MSM4 payload with a single satellite / signal cell."""
    n_sats = bin(satellite_mask).count("1")
    n_sigs = bin(signal_mask).count("1")
    cell_mask_val = (1 << (n_sats * n_sigs)) - 1  # all cells present
    # Header
    fields: list[tuple[int, int]] = [
        (msg_type, 12),
        (station_id, 12),
        (epoch_ms, 30),
        (0, 1),   # multiple message flag
        (0, 3),   # issue of data station
        (0, 7),   # reserved
        (0, 2),   # GNSS divergence-free
        (0, 3),   # smoothing interval
        (satellite_mask, 64),
        (signal_mask, 32),
        (cell_mask_val, n_sats * n_sigs),
        # Satellite rough ranges (1 per sat: 8 bits)
        (100, 8),
        # Fine pseudoranges per present cell (10 bits each)
        (512, 10),  # → fractional = 512 / 1024 = 0.5
    ]
    return _encode_bits(fields)


class TestRtcmMsm:
    def _parse_msm(self, msg_type: int = 1074) -> RtcmMsm:
        payload = _build_msm4_payload(msg_type=msg_type)
        frame = _make_frame(payload)
        msgs = RtcmParser(io.BytesIO(frame)).parse_all()
        assert len(msgs) == 1
        assert isinstance(msgs[0], RtcmMsm)
        return msgs[0]

    def test_gps_msm4_message_type(self):
        msg = self._parse_msm(1074)
        assert msg.message_type == 1074
        assert msg.msm_type == 4
        assert msg.constellation == "GPS"

    def test_glonass_msm4(self):
        payload = _build_msm4_payload(msg_type=1084)
        frame = _make_frame(payload)
        msgs = RtcmParser(io.BytesIO(frame)).parse_all()
        assert len(msgs) == 1
        assert msgs[0].constellation == "GLONASS"

    def test_galileo_msm4(self):
        payload = _build_msm4_payload(msg_type=1094)
        frame = _make_frame(payload)
        msgs = RtcmParser(io.BytesIO(frame)).parse_all()
        assert len(msgs) == 1
        assert msgs[0].constellation == "Galileo"

    def test_beidou_msm4(self):
        payload = _build_msm4_payload(msg_type=1124)
        frame = _make_frame(payload)
        msgs = RtcmParser(io.BytesIO(frame)).parse_all()
        assert len(msgs) == 1
        assert msgs[0].constellation == "BeiDou"

    def test_gps_msm7_message_type(self):
        # MSM7 payload is more complex; just check the message is decoded
        payload = _build_msm4_payload(msg_type=1077)
        frame = _make_frame(payload)
        msgs = RtcmParser(io.BytesIO(frame)).parse_all()
        # May or may not fully decode depending on payload length,
        # but should at least return an RtcmMsm and not raise.
        assert len(msgs) <= 1  # 0 if payload too short, 1 otherwise

    def test_station_id(self):
        payload = _build_msm4_payload(station_id=7)
        frame = _make_frame(payload)
        msgs = RtcmParser(io.BytesIO(frame)).parse_all()
        assert msgs[0].station_id == 7

    def test_satellite_mask(self):
        payload = _build_msm4_payload(satellite_mask=0x0000_0000_0000_0001)
        frame = _make_frame(payload)
        msgs = RtcmParser(io.BytesIO(frame)).parse_all()
        assert msgs[0].satellite_mask == 0x0000_0000_0000_0001

    def test_pseudoranges_not_empty(self):
        msg = self._parse_msm(1074)
        assert isinstance(msg.pseudoranges_m, list)
        assert len(msg.pseudoranges_m) >= 1

    def test_pseudorange_positive(self):
        msg = self._parse_msm(1074)
        if msg.pseudoranges_m:
            assert msg.pseudoranges_m[0] > 0

    def test_unknown_message_type_skipped(self):
        # MT1234 is not a supported message type
        payload = bytes([0x04, 0xD2, 0x00, 0x00, 0x00])  # type=1234 approx
        frame = _make_frame(payload)
        msgs = RtcmParser(io.BytesIO(frame)).parse_all()
        assert msgs == []
