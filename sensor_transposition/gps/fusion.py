"""
gps/fusion.py

GPS absolute-position fusion into the local (ENU) map frame.

Ties GPS geodetic fixes to the local map frame used by the rest of the
sensor_transposition SLAM pipeline.  A reference origin (latitude, longitude,
altitude) anchors the local East-North-Up (ENU) frame to the WGS-84 ellipsoid,
so that every subsequent GPS fix can be expressed as a 3-D position in the
same local Cartesian frame that is used by :class:`FramePoseSequence`,
:class:`PointCloudMap`, and :class:`ImuEkf`.

Key classes and functions
-------------------------
:class:`GpsFuser`
    Converts GPS fixes to local ENU and fuses them into an EKF state or a
    :class:`~sensor_transposition.frame_pose.FramePoseSequence`.

:func:`hdop_to_noise`
    Converts an HDOP value (from a GGA sentence) into a 3×3 ENU position
    noise covariance matrix suitable for use with
    :meth:`~sensor_transposition.imu.ekf.ImuEkf.position_update`.

Typical use
-----------
::

    from sensor_transposition.gps.fusion import GpsFuser, hdop_to_noise
    from sensor_transposition.gps.nmea import NmeaParser
    from sensor_transposition.imu.ekf import ImuEkf, EkfState
    from sensor_transposition.frame_pose import FramePoseSequence

    # Load GPS fixes.
    fixes = NmeaParser("gps_log.nmea").gga_fixes()

    # 1. Anchor the map to the first valid fix.
    origin = fixes[0]
    fuser = GpsFuser(
        ref_lat=origin.latitude,
        ref_lon=origin.longitude,
        ref_alt=origin.altitude,
    )

    # 2. Fuse all fixes into a FramePoseSequence (trajectory).
    seq = FramePoseSequence()
    for i, fix in enumerate(fixes):
        fuser.fuse_into_sequence(seq, timestamp=float(i) * 0.1, fix=fix)

    # 3. Fuse the same fixes into an EKF-state stream.
    ekf   = ImuEkf()
    state = EkfState()
    for fix in fixes:
        noise = hdop_to_noise(fix.hdop)
        state = fuser.fuse_into_ekf(ekf, state, fix, noise)
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from sensor_transposition.gps.converter import geodetic_to_enu
from sensor_transposition.gps.nmea import GgaFix, RmcFix

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

GpsFix = Union[GgaFix, RmcFix]


# ---------------------------------------------------------------------------
# Noise helper
# ---------------------------------------------------------------------------


def hdop_to_noise(
    hdop: float,
    base_sigma_m: float = 3.0,
    vertical_sigma_m: float = 5.0,
) -> np.ndarray:
    """Convert HDOP to a 3×3 ENU position noise covariance matrix.

    The horizontal standard deviation is ``hdop × base_sigma_m``; the
    vertical standard deviation is fixed at *vertical_sigma_m* (GPS height
    accuracy is typically 1.5× worse than horizontal accuracy, and the
    fixed value avoids an unrealistically small covariance when HDOP is
    very small).

    Args:
        hdop: Horizontal dilution of precision (dimensionless; typical values
            are 1–5 for consumer GNSS receivers in open sky).
        base_sigma_m: Horizontal standard deviation in metres when HDOP = 1.
            Use ``3.0`` m for a standard single-frequency GPS receiver,
            ``0.3`` m for an RTK-corrected receiver, or ``1.0`` m for a
            multi-constellation receiver with SBAS.
        vertical_sigma_m: Vertical (Up) standard deviation in metres,
            independent of HDOP (default: ``5.0`` m).

    Returns:
        ``(3, 3)`` diagonal covariance matrix
        ``diag(σ_E², σ_N², σ_U²)`` where
        ``σ_E = σ_N = hdop × base_sigma_m`` and ``σ_U = vertical_sigma_m``.

    Example::

        noise = hdop_to_noise(1.5)          # standard GPS, HDOP 1.5
        rtk_noise = hdop_to_noise(1.0, base_sigma_m=0.02)  # RTK, σ = 2 cm
    """
    h_sigma = float(hdop) * float(base_sigma_m)
    v_sigma = float(vertical_sigma_m)
    return np.diag([h_sigma ** 2, h_sigma ** 2, v_sigma ** 2])


# ---------------------------------------------------------------------------
# GpsFuser
# ---------------------------------------------------------------------------


class GpsFuser:
    """Fuses GPS geodetic fixes into the local ENU map frame.

    A single reference origin (latitude, longitude, altitude) anchors the
    local East-North-Up (ENU) Cartesian frame to the WGS-84 ellipsoid.  All
    subsequent GPS fixes are expressed as ``(east, north, up)`` displacements
    in metres from that origin, compatible with the coordinate system used by
    :class:`~sensor_transposition.frame_pose.FramePoseSequence`,
    :class:`~sensor_transposition.point_cloud_map.PointCloudMap`, and
    :class:`~sensor_transposition.imu.ekf.ImuEkf`.

    Args:
        ref_lat: Geodetic latitude of the map origin in decimal degrees.
        ref_lon: Longitude of the map origin in decimal degrees.
        ref_alt: Altitude of the map origin above the WGS-84 ellipsoid in
            metres (default ``0.0``).

    Example::

        from sensor_transposition.gps.fusion import GpsFuser
        from sensor_transposition.gps.nmea import GgaFix

        fuser = GpsFuser(ref_lat=51.5, ref_lon=-0.1, ref_alt=10.0)

        fix = GgaFix(
            timestamp="120000.00",
            latitude=51.501,
            longitude=-0.099,
            fix_quality=1,
            num_satellites=8,
            hdop=1.2,
            altitude=11.0,
            geoid_separation=47.0,
        )

        east, north, up = fuser.fix_to_enu(fix)
        # east ≈ 65 m, north ≈ 111 m, up ≈ 1 m
    """

    def __init__(
        self,
        ref_lat: float,
        ref_lon: float,
        ref_alt: float = 0.0,
    ) -> None:
        self._ref_lat = float(ref_lat)
        self._ref_lon = float(ref_lon)
        self._ref_alt = float(ref_alt)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ref_lat(self) -> float:
        """Latitude of the ENU map origin in decimal degrees."""
        return self._ref_lat

    @property
    def ref_lon(self) -> float:
        """Longitude of the ENU map origin in decimal degrees."""
        return self._ref_lon

    @property
    def ref_alt(self) -> float:
        """Altitude of the ENU map origin above the WGS-84 ellipsoid (m)."""
        return self._ref_alt

    # ------------------------------------------------------------------
    # Core conversion
    # ------------------------------------------------------------------

    def fix_to_enu(self, fix: GpsFix) -> Tuple[float, float, float]:
        """Convert a GPS fix to local ENU coordinates.

        Args:
            fix: A :class:`~sensor_transposition.gps.nmea.GgaFix` or
                :class:`~sensor_transposition.gps.nmea.RmcFix` instance.
                For :class:`~sensor_transposition.gps.nmea.RmcFix` objects
                (which carry no altitude), the reference altitude is used for
                the Up component.

        Returns:
            Tuple ``(east, north, up)`` in metres relative to the map origin.
        """
        if isinstance(fix, GgaFix):
            alt = fix.altitude
        else:
            # RMC sentences carry no altitude; use the reference altitude so
            # the Up component is approximately zero at the map origin.
            alt = self._ref_alt
        return geodetic_to_enu(
            fix.latitude,
            fix.longitude,
            alt,
            self._ref_lat,
            self._ref_lon,
            self._ref_alt,
        )

    def fix_to_enu_array(self, fix: GpsFix) -> np.ndarray:
        """Return the local ENU position as a ``(3,)`` NumPy array.

        Convenience wrapper around :meth:`fix_to_enu`.

        Args:
            fix: A GPS fix record.

        Returns:
            ``(3,)`` float64 array ``[east, north, up]`` in metres.
        """
        e, n, u = self.fix_to_enu(fix)
        return np.array([e, n, u], dtype=float)

    # ------------------------------------------------------------------
    # EKF fusion
    # ------------------------------------------------------------------

    def fuse_into_ekf(
        self,
        ekf: "ImuEkf",  # noqa: F821
        state: "EkfState",  # noqa: F821
        fix: GpsFix,
        noise: np.ndarray,
    ) -> "EkfState":  # noqa: F821
        """Fuse a GPS fix into an EKF state via a 3-D position update.

        Converts *fix* to local ENU and calls
        :meth:`~sensor_transposition.imu.ekf.ImuEkf.position_update` to
        incorporate the measurement into the filter.

        Args:
            ekf: :class:`~sensor_transposition.imu.ekf.ImuEkf` instance.
            state: Current :class:`~sensor_transposition.imu.ekf.EkfState`.
            fix: GPS fix record to fuse.
            noise: ``(3, 3)`` measurement noise covariance in m².  Use
                :func:`hdop_to_noise` to derive this from the HDOP value
                carried in a :class:`~sensor_transposition.gps.nmea.GgaFix`.

        Returns:
            Updated :class:`~sensor_transposition.imu.ekf.EkfState`.

        Example::

            from sensor_transposition.gps.fusion import GpsFuser, hdop_to_noise
            from sensor_transposition.imu.ekf import ImuEkf, EkfState

            fuser = GpsFuser(ref_lat=51.5, ref_lon=-0.1)
            ekf   = ImuEkf()
            state = EkfState()

            state = fuser.fuse_into_ekf(
                ekf, state, gga_fix,
                noise=hdop_to_noise(gga_fix.hdop),
            )
        """
        position = self.fix_to_enu_array(fix)
        return ekf.position_update(state, position, noise)

    # ------------------------------------------------------------------
    # FramePoseSequence fusion
    # ------------------------------------------------------------------

    def fuse_into_sequence(
        self,
        sequence: "FramePoseSequence",  # noqa: F821
        timestamp: float,
        fix: GpsFix,
    ) -> "FramePose":  # noqa: F821
        """Add or update a :class:`~sensor_transposition.frame_pose.FramePose`
        with the position derived from *fix*.

        If *sequence* already contains a frame whose time window covers
        *timestamp* (i.e.
        ``frame.timestamp <= timestamp < frame.timestamp + frame_duration``),
        its :attr:`~sensor_transposition.frame_pose.FramePose.translation` is
        updated in place.  Otherwise a new
        :class:`~sensor_transposition.frame_pose.FramePose` is appended with
        an identity orientation quaternion.

        Args:
            sequence: :class:`~sensor_transposition.frame_pose.FramePoseSequence`
                to update.
            timestamp: Time at which the fix was received (seconds, e.g. a
                UNIX timestamp or elapsed time since the start of the session).
            fix: GPS fix record.

        Returns:
            The :class:`~sensor_transposition.frame_pose.FramePose` that was
            added or updated.

        Example::

            from sensor_transposition.gps.fusion import GpsFuser
            from sensor_transposition.frame_pose import FramePoseSequence

            fuser = GpsFuser(ref_lat=51.5, ref_lon=-0.1, ref_alt=10.0)
            seq   = FramePoseSequence()

            for i, fix in enumerate(gga_fixes):
                fuser.fuse_into_sequence(seq, timestamp=float(i) * 0.1, fix=fix)
        """
        from sensor_transposition.frame_pose import FramePose

        e, n, u = self.fix_to_enu(fix)

        existing = sequence.get_pose_at_timestamp(timestamp)
        if existing is not None:
            existing.translation = [e, n, u]
            return existing

        pose = FramePose(
            timestamp=timestamp,
            translation=[e, n, u],
        )
        sequence.add_pose(pose)
        return pose
