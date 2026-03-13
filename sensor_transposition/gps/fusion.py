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
    :class:`~sensor_transposition.frame_pose.FramePoseSequence`.  Supports
    GNSS outage detection via ``max_fix_age_sec`` and an optional
    ``on_outage`` callback.

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

GNSS outage handling
--------------------
Pass ``max_fix_age_sec`` and an optional ``on_outage`` callback to
:class:`GpsFuser` to detect GNSS outages (tunnels, urban canyons) and
automatically skip GPS updates when the last fix is stale::

    def handle_outage(age_sec: float) -> None:
        print(f"GPS outage – last fix was {age_sec:.1f} s ago")

    fuser = GpsFuser(
        ref_lat=51.5080, ref_lon=-0.1281,
        max_fix_age_sec=2.0,
        on_outage=handle_outage,
    )

    current_time = 1000.0  # UNIX timestamp or elapsed time (seconds)
    age = fuser.fix_age(current_time)   # seconds since the last fused fix
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

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

    **GNSS outage handling** – when ``max_fix_age_sec`` is supplied,
    :meth:`fuse_into_ekf` will skip the EKF update if the time elapsed since
    the last successfully fused fix exceeds this threshold.  An optional
    ``on_outage`` callback is invoked with the current age (in seconds) each
    time a stale-fix skip occurs.  Use :meth:`fix_age` to query the age of
    the most recent fix at any time.

    Args:
        ref_lat: Geodetic latitude of the map origin in decimal degrees.
        ref_lon: Longitude of the map origin in decimal degrees.
        ref_alt: Altitude of the map origin above the WGS-84 ellipsoid in
            metres (default ``0.0``).
        max_fix_age_sec: Maximum age of a GPS fix (seconds) before
            :meth:`fuse_into_ekf` skips the update and calls *on_outage*.
            ``None`` (default) disables the age check entirely.
        on_outage: Optional callable ``(age_sec: float) -> None`` invoked by
            :meth:`fuse_into_ekf` each time it skips an update due to a stale
            fix.  Suitable for switching to wheel-odometry or LiDAR-odometry
            only mode.

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

    GNSS outage example::

        def handle_outage(age_sec: float) -> None:
            print(f"GPS outage – switching to LiDAR odometry (age={age_sec:.1f}s)")

        fuser = GpsFuser(
            ref_lat=51.5, ref_lon=-0.1,
            max_fix_age_sec=2.0,
            on_outage=handle_outage,
        )

        # fuse_into_ekf will skip the update and call handle_outage()
        # if more than 2.0 s have elapsed since the last fused fix.
        new_state = fuser.fuse_into_ekf(
            ekf, state, fix,
            noise=hdop_to_noise(fix.hdop),
            current_timestamp=current_time,
        )
    """

    def __init__(
        self,
        ref_lat: float,
        ref_lon: float,
        ref_alt: float = 0.0,
        max_fix_age_sec: Optional[float] = None,
        on_outage: Optional[Callable[[float], None]] = None,
    ) -> None:
        self._ref_lat = float(ref_lat)
        self._ref_lon = float(ref_lon)
        self._ref_alt = float(ref_alt)
        self._max_fix_age_sec: Optional[float] = (
            float(max_fix_age_sec) if max_fix_age_sec is not None else None
        )
        self._on_outage = on_outage
        self._last_fix_timestamp: Optional[float] = None

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

    @property
    def max_fix_age_sec(self) -> Optional[float]:
        """Maximum fix age threshold in seconds, or ``None`` if disabled."""
        return self._max_fix_age_sec

    @property
    def last_fix_timestamp(self) -> Optional[float]:
        """Timestamp of the most recently fused GPS fix, or ``None`` if no fix
        has been fused yet."""
        return self._last_fix_timestamp

    def fix_age(self, current_timestamp: float) -> Optional[float]:
        """Return the age of the most recent GPS fix in seconds.

        Args:
            current_timestamp: The current time (seconds), using the same
                timebase as the ``current_timestamp`` argument of
                :meth:`fuse_into_ekf`.

        Returns:
            Age in seconds (``current_timestamp - last_fix_timestamp``), or
            ``None`` if no fix has been fused yet (i.e. this fuser has never
            successfully fused a fix).

        Example::

            age = fuser.fix_age(current_time)
            if age is not None and age > 5.0:
                print("Switching to dead-reckoning mode")
        """
        if self._last_fix_timestamp is None:
            return None
        return float(current_timestamp) - self._last_fix_timestamp

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
        current_timestamp: Optional[float] = None,
    ) -> "EkfState":  # noqa: F821
        """Fuse a GPS fix into an EKF state via a 3-D position update.

        Converts *fix* to local ENU and calls
        :meth:`~sensor_transposition.imu.ekf.ImuEkf.position_update` to
        incorporate the measurement into the filter.

        If ``max_fix_age_sec`` was set at construction time and
        ``current_timestamp`` is provided, the method checks whether the last
        fused fix is stale.  If the fix age exceeds ``max_fix_age_sec``, the
        EKF update is skipped and ``on_outage`` (if set) is called with the
        age in seconds.  When *current_timestamp* is provided and the update
        is not skipped, ``_last_fix_timestamp`` is updated to
        *current_timestamp*.

        Args:
            ekf: :class:`~sensor_transposition.imu.ekf.ImuEkf` instance.
            state: Current :class:`~sensor_transposition.imu.ekf.EkfState`.
            fix: GPS fix record to fuse.
            noise: ``(3, 3)`` measurement noise covariance in m².  Use
                :func:`hdop_to_noise` to derive this from the HDOP value
                carried in a :class:`~sensor_transposition.gps.nmea.GgaFix`.
            current_timestamp: Optional current time in seconds (same
                timebase used for ``fix_age``).  Required for outage
                detection when ``max_fix_age_sec`` is set.

        Returns:
            Updated :class:`~sensor_transposition.imu.ekf.EkfState`, or the
            unchanged *state* if the update was skipped due to a stale fix.

        Example::

            from sensor_transposition.gps.fusion import GpsFuser, hdop_to_noise
            from sensor_transposition.imu.ekf import ImuEkf, EkfState

            fuser = GpsFuser(ref_lat=51.5, ref_lon=-0.1)
            ekf   = ImuEkf()
            state = EkfState()

            state = fuser.fuse_into_ekf(
                ekf, state, gga_fix,
                noise=hdop_to_noise(gga_fix.hdop),
                current_timestamp=current_time,
            )
        """
        # ---- outage check ------------------------------------------------
        if (
            self._max_fix_age_sec is not None
            and current_timestamp is not None
            and self._last_fix_timestamp is not None
        ):
            age = float(current_timestamp) - self._last_fix_timestamp
            if age > self._max_fix_age_sec:
                if self._on_outage is not None:
                    self._on_outage(age)
                return state  # skip update

        # ---- perform EKF update ------------------------------------------
        position = self.fix_to_enu_array(fix)
        new_state = ekf.position_update(state, position, noise)

        # Record the timestamp of this successfully fused fix
        if current_timestamp is not None:
            self._last_fix_timestamp = float(current_timestamp)

        return new_state

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
