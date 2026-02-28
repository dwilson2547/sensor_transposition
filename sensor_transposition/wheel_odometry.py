"""
wheel_odometry.py

Wheel-odometry kinematic models for dead-reckoning vehicle pose estimation.

Two planar kinematic models are provided:

1. **Differential drive** – a vehicle with two independently driven wheels
   (e.g. a robot or skid-steer platform).  State is evolved from left/right
   wheel speeds using the midpoint (trapezoidal) integration method.

2. **Ackermann (bicycle model)** – a car-like vehicle whose steering geometry
   is approximated by the single-track bicycle model.  State is evolved from
   forward speed and front-axle steering angle.

Both models operate in a local, planar reference frame and accumulate a 2-D
SE(2) pose ``(x, y, θ)``.  The result can be composed with a
:class:`~sensor_transposition.frame_pose.FramePose` to obtain a world-frame
trajectory.

All functions rely only on ``numpy`` (already required by
``sensor_transposition``).

Typical use-cases
-----------------
* **Dead-reckoning between sensor frames**: bridge gaps when LiDAR or camera
  data are unavailable.
* **Initial pose guess for scan-matching**: provide a motion prior for ICP to
  refine.
* **EKF motion model**: supply the prediction step of an error-state EKF when
  GPS or LiDAR updates are absent.

Example – differential drive::

    from sensor_transposition.wheel_odometry import (
        DifferentialDriveOdometer,
        integrate_differential_drive,
    )

    odom = DifferentialDriveOdometer(wheel_base=0.54, wheel_radius=0.1)
    result = odom.integrate(timestamps, left_ticks, right_ticks,
                            ticks_per_revolution=360)
    print(result.x, result.y, result.theta)

Example – Ackermann::

    from sensor_transposition.wheel_odometry import (
        AckermannOdometer,
        integrate_ackermann,
    )

    odom = AckermannOdometer(wheel_base=2.7)
    result = odom.integrate(timestamps, speeds, steering_angles)
    print(result.x, result.y, result.theta)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# OdometryResult
# ---------------------------------------------------------------------------


@dataclass
class OdometryResult:
    """Accumulated 2-D SE(2) pose from a wheel-odometry integration run.

    The pose is expressed relative to the vehicle frame at the start of the
    integration window (i.e. the initial pose is always the origin).

    Attributes:
        x: Forward displacement in metres along the world x-axis.
        y: Lateral displacement in metres along the world y-axis.
        theta: Heading angle in radians, measured counter-clockwise from the
            initial heading direction.
        duration: Total integration interval in seconds.
        num_samples: Number of measurement samples used during integration.
    """

    x: float
    y: float
    theta: float
    duration: float
    num_samples: int

    @property
    def translation(self) -> np.ndarray:
        """``[x, y, 0]`` displacement vector (metres)."""
        return np.array([self.x, self.y, 0.0])

    @property
    def rotation_matrix(self) -> np.ndarray:
        """3×3 rotation matrix for the accumulated heading angle."""
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ])

    @property
    def transform(self) -> np.ndarray:
        """4×4 homogeneous transform representing the accumulated SE(2) pose."""
        T = np.eye(4)
        T[:3, :3] = self.rotation_matrix
        T[:3, 3] = self.translation
        return T


# ---------------------------------------------------------------------------
# Differential-drive odometry
# ---------------------------------------------------------------------------


class DifferentialDriveOdometer:
    """Dead-reckoning odometer for a differential-drive platform.

    The vehicle is modelled as two parallel wheels separated by *wheel_base*.
    Measurements may be supplied either as **wheel speeds** (m/s) or as
    **encoder tick counts** (converted internally using *ticks_per_revolution*
    and *wheel_radius*).

    Integration uses the **midpoint (trapezoidal) method** for improved
    accuracy over simple Euler steps.

    Args:
        wheel_base: Distance between the left and right wheel contact points
            in metres.
        wheel_radius: Wheel radius in metres.  Required only when integrating
            encoder ticks (ignored when speeds are supplied directly).

    Example::

        odom = DifferentialDriveOdometer(wheel_base=0.54, wheel_radius=0.1)
        result = odom.integrate(timestamps, left_ticks, right_ticks,
                                ticks_per_revolution=360)
    """

    def __init__(
        self,
        wheel_base: float,
        wheel_radius: float = 1.0,
    ) -> None:
        if wheel_base <= 0:
            raise ValueError(
                f"wheel_base must be positive, got {wheel_base}."
            )
        if wheel_radius <= 0:
            raise ValueError(
                f"wheel_radius must be positive, got {wheel_radius}."
            )
        self._wheel_base = float(wheel_base)
        self._wheel_radius = float(wheel_radius)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def wheel_base(self) -> float:
        """Distance between the two wheel contact points (metres)."""
        return self._wheel_base

    @property
    def wheel_radius(self) -> float:
        """Wheel radius (metres)."""
        return self._wheel_radius

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def integrate(
        self,
        timestamps: np.ndarray,
        left_wheel: np.ndarray,
        right_wheel: np.ndarray,
        *,
        ticks_per_revolution: float | None = None,
    ) -> OdometryResult:
        """Integrate wheel measurements into a relative SE(2) pose.

        Args:
            timestamps: Shape ``(N,)`` UNIX timestamps in seconds.  Must be
                strictly increasing.
            left_wheel: Shape ``(N,)`` measurements for the left wheel.
                Interpreted as **linear speeds in m/s** when
                *ticks_per_revolution* is ``None``, or as **encoder tick
                counts** (cumulative or per-interval) otherwise.
            right_wheel: Shape ``(N,)`` measurements for the right wheel,
                same units as *left_wheel*.
            ticks_per_revolution: If provided, *left_wheel* and *right_wheel*
                are treated as **cumulative** encoder tick counts and converted
                to arc lengths using the formula::

                    arc = (Δticks / ticks_per_revolution) * 2π * wheel_radius

                If ``None``, *left_wheel* and *right_wheel* are taken as
                instantaneous linear speeds (m/s).

        Returns:
            :class:`OdometryResult` containing the accumulated SE(2) pose,
            total duration, and sample count.

        Raises:
            ValueError: If fewer than 2 samples are provided, array shapes
                are inconsistent, or timestamps are not strictly increasing.
        """
        ts = np.asarray(timestamps, dtype=float)
        lw = np.asarray(left_wheel, dtype=float)
        rw = np.asarray(right_wheel, dtype=float)

        n = ts.shape[0]
        if n < 2:
            raise ValueError(
                f"At least 2 samples are required for integration, got {n}."
            )
        if lw.shape != (n,):
            raise ValueError(
                f"left_wheel must have shape ({n},), got {lw.shape}."
            )
        if rw.shape != (n,):
            raise ValueError(
                f"right_wheel must have shape ({n},), got {rw.shape}."
            )

        dt_all = np.diff(ts)
        if np.any(dt_all <= 0):
            raise ValueError("timestamps must be strictly increasing.")

        if ticks_per_revolution is not None:
            if ticks_per_revolution <= 0:
                raise ValueError(
                    f"ticks_per_revolution must be positive, "
                    f"got {ticks_per_revolution}."
                )
            # Convert cumulative tick counts to per-interval arc lengths.
            left_speeds, right_speeds = _ticks_to_speeds(
                lw, rw, dt_all,
                ticks_per_revolution=float(ticks_per_revolution),
                wheel_radius=self._wheel_radius,
            )
        else:
            left_speeds = lw
            right_speeds = rw

        x, y, theta = _integrate_diff_drive(
            dt_all, left_speeds, right_speeds, self._wheel_base
        )

        return OdometryResult(
            x=float(x),
            y=float(y),
            theta=float(theta),
            duration=float(ts[-1] - ts[0]),
            num_samples=n,
        )


def integrate_differential_drive(
    timestamps: np.ndarray,
    left_wheel: np.ndarray,
    right_wheel: np.ndarray,
    wheel_base: float,
    wheel_radius: float = 1.0,
    *,
    ticks_per_revolution: float | None = None,
) -> OdometryResult:
    """Functional wrapper around :class:`DifferentialDriveOdometer`.

    Args:
        timestamps: Shape ``(N,)`` timestamps in seconds.
        left_wheel: Shape ``(N,)`` left-wheel speeds (m/s) or encoder ticks.
        right_wheel: Shape ``(N,)`` right-wheel speeds (m/s) or encoder ticks.
        wheel_base: Distance between wheel contact points (metres).
        wheel_radius: Wheel radius (metres).  Used only when
            *ticks_per_revolution* is set.
        ticks_per_revolution: If provided, inputs are treated as cumulative
            encoder ticks rather than speeds.

    Returns:
        :class:`OdometryResult` with the accumulated SE(2) pose.
    """
    return DifferentialDriveOdometer(
        wheel_base=wheel_base,
        wheel_radius=wheel_radius,
    ).integrate(
        timestamps,
        left_wheel,
        right_wheel,
        ticks_per_revolution=ticks_per_revolution,
    )


# ---------------------------------------------------------------------------
# Ackermann (bicycle-model) odometry
# ---------------------------------------------------------------------------


class AckermannOdometer:
    """Dead-reckoning odometer for a car-like (Ackermann-steered) vehicle.

    Uses the **single-track (bicycle) kinematic model** where the steering
    angle is measured at the front axle and the vehicle's rear axle is the
    reference point.

    Kinematics (per time step *dt*)::

        β     = arctan(tan(δ) * rear_axle_fraction)  [side-slip at CoM]
        v_com = speed * cos(β) / cos(δ - β)           [unused in bicycle model]

        Bicycle model (rear-axle reference)::

            θ̇ = speed * tan(δ) / wheel_base
            ẋ = speed * cos(θ)
            ẏ = speed * sin(θ)

    Integration uses the **midpoint method**.

    Args:
        wheel_base: Distance between front and rear axle contact points
            (metres).

    Example::

        odom = AckermannOdometer(wheel_base=2.7)
        result = odom.integrate(timestamps, speeds, steering_angles)
    """

    def __init__(self, wheel_base: float) -> None:
        if wheel_base <= 0:
            raise ValueError(
                f"wheel_base must be positive, got {wheel_base}."
            )
        self._wheel_base = float(wheel_base)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def wheel_base(self) -> float:
        """Distance between front and rear axles (metres)."""
        return self._wheel_base

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def integrate(
        self,
        timestamps: np.ndarray,
        speeds: np.ndarray,
        steering_angles: np.ndarray,
    ) -> OdometryResult:
        """Integrate speed and steering measurements into a relative SE(2) pose.

        Args:
            timestamps: Shape ``(N,)`` UNIX timestamps in seconds.  Must be
                strictly increasing.
            speeds: Shape ``(N,)`` forward speed of the rear axle in m/s.
                Negative values indicate reverse motion.
            steering_angles: Shape ``(N,)`` front-axle steering angle in
                radians.  Positive is a left (counter-clockwise) turn.

        Returns:
            :class:`OdometryResult` containing the accumulated SE(2) pose,
            total duration, and sample count.

        Raises:
            ValueError: If fewer than 2 samples are provided, array shapes
                are inconsistent, or timestamps are not strictly increasing.
        """
        ts = np.asarray(timestamps, dtype=float)
        v = np.asarray(speeds, dtype=float)
        delta = np.asarray(steering_angles, dtype=float)

        n = ts.shape[0]
        if n < 2:
            raise ValueError(
                f"At least 2 samples are required for integration, got {n}."
            )
        if v.shape != (n,):
            raise ValueError(
                f"speeds must have shape ({n},), got {v.shape}."
            )
        if delta.shape != (n,):
            raise ValueError(
                f"steering_angles must have shape ({n},), got {delta.shape}."
            )

        dt_all = np.diff(ts)
        if np.any(dt_all <= 0):
            raise ValueError("timestamps must be strictly increasing.")

        x, y, theta = _integrate_ackermann(
            dt_all, v, delta, self._wheel_base
        )

        return OdometryResult(
            x=float(x),
            y=float(y),
            theta=float(theta),
            duration=float(ts[-1] - ts[0]),
            num_samples=n,
        )


def integrate_ackermann(
    timestamps: np.ndarray,
    speeds: np.ndarray,
    steering_angles: np.ndarray,
    wheel_base: float,
) -> OdometryResult:
    """Functional wrapper around :class:`AckermannOdometer`.

    Args:
        timestamps: Shape ``(N,)`` timestamps in seconds.
        speeds: Shape ``(N,)`` forward speed at the rear axle (m/s).
        steering_angles: Shape ``(N,)`` front-axle steering angle (radians).
        wheel_base: Distance between front and rear axles (metres).

    Returns:
        :class:`OdometryResult` with the accumulated SE(2) pose.
    """
    return AckermannOdometer(wheel_base=wheel_base).integrate(
        timestamps, speeds, steering_angles
    )


# ---------------------------------------------------------------------------
# Internal integration kernels
# ---------------------------------------------------------------------------


def _ticks_to_speeds(
    left_ticks: np.ndarray,
    right_ticks: np.ndarray,
    dt_all: np.ndarray,
    ticks_per_revolution: float,
    wheel_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert cumulative encoder tick arrays to per-interval speed arrays.

    Args:
        left_ticks: ``(N,)`` cumulative tick counts for the left wheel.
        right_ticks: ``(N,)`` cumulative tick counts for the right wheel.
        dt_all: ``(N-1,)`` time intervals between consecutive samples.
        ticks_per_revolution: Encoder ticks per full wheel revolution.
        wheel_radius: Wheel radius in metres.

    Returns:
        Tuple ``(left_speeds, right_speeds)`` each of shape ``(N,)`` in m/s.
        The first entry is the average speed over the first interval and the
        last interval respectively; internal entries are midpoint averages.
    """
    arc_per_tick = (2.0 * np.pi * wheel_radius) / ticks_per_revolution

    # Per-interval arc lengths from consecutive tick differences.
    left_arcs = np.diff(left_ticks) * arc_per_tick   # (N-1,)
    right_arcs = np.diff(right_ticks) * arc_per_tick  # (N-1,)

    # Per-interval speeds.
    left_interval_speed = left_arcs / dt_all    # (N-1,)
    right_interval_speed = right_arcs / dt_all  # (N-1,)

    # Produce per-sample speed arrays of length N by padding with edge values.
    left_speeds = np.empty(left_ticks.shape[0])
    right_speeds = np.empty(right_ticks.shape[0])
    left_speeds[0] = left_interval_speed[0]
    left_speeds[1:] = left_interval_speed
    right_speeds[0] = right_interval_speed[0]
    right_speeds[1:] = right_interval_speed

    return left_speeds, right_speeds


def _integrate_diff_drive(
    dt_all: np.ndarray,
    left_speeds: np.ndarray,
    right_speeds: np.ndarray,
    wheel_base: float,
) -> tuple[float, float, float]:
    """Core differential-drive midpoint integration.

    Args:
        dt_all: ``(N-1,)`` time intervals.
        left_speeds: ``(N,)`` left-wheel linear speeds (m/s).
        right_speeds: ``(N,)`` right-wheel linear speeds (m/s).
        wheel_base: Track width (metres).

    Returns:
        Tuple ``(x, y, theta)`` accumulated SE(2) pose.
    """
    x = 0.0
    y = 0.0
    theta = 0.0

    n = dt_all.shape[0]
    for k in range(n):
        dt = dt_all[k]

        # Midpoint speeds.
        v_l_mid = 0.5 * (left_speeds[k] + left_speeds[k + 1])
        v_r_mid = 0.5 * (right_speeds[k] + right_speeds[k + 1])

        # Linear and angular velocity of the vehicle centre.
        v = 0.5 * (v_l_mid + v_r_mid)
        omega = (v_r_mid - v_l_mid) / wheel_base

        # Midpoint heading: use the heading at mid-step for better accuracy.
        theta_mid = theta + 0.5 * omega * dt

        x += v * np.cos(theta_mid) * dt
        y += v * np.sin(theta_mid) * dt
        theta += omega * dt

    return x, y, theta


def _integrate_ackermann(
    dt_all: np.ndarray,
    speeds: np.ndarray,
    steering_angles: np.ndarray,
    wheel_base: float,
) -> tuple[float, float, float]:
    """Core Ackermann (bicycle model) midpoint integration.

    Args:
        dt_all: ``(N-1,)`` time intervals.
        speeds: ``(N,)`` forward speed at the rear axle (m/s).
        steering_angles: ``(N,)`` front-axle steering angle (radians).
        wheel_base: Axle-to-axle distance (metres).

    Returns:
        Tuple ``(x, y, theta)`` accumulated SE(2) pose.
    """
    x = 0.0
    y = 0.0
    theta = 0.0

    n = dt_all.shape[0]
    for k in range(n):
        dt = dt_all[k]

        # Midpoint values.
        v_mid = 0.5 * (speeds[k] + speeds[k + 1])
        delta_mid = 0.5 * (steering_angles[k] + steering_angles[k + 1])

        # Angular rate: ω = v * tan(δ) / L
        omega = v_mid * np.tan(delta_mid) / wheel_base

        # Midpoint heading.
        theta_mid = theta + 0.5 * omega * dt

        x += v_mid * np.cos(theta_mid) * dt
        y += v_mid * np.sin(theta_mid) * dt
        theta += omega * dt

    return x, y, theta
