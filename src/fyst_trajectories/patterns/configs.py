"""Configuration objects for scan patterns.

Each scan pattern has an associated config class that holds its
parameters. All configs inherit from :class:`ScanConfig`, which
provides the common ``timestep`` parameter. Config instances are
immutable after creation.
"""

import math
import warnings
from dataclasses import dataclass, field

from ..coordinates import SATELLITE_BODIES, SOLAR_SYSTEM_BODIES
from ..exceptions import PointingWarning
from .turnarounds import turnaround_overshoot_deg

# Advisory upper bounds, separate from the per-axis limits in
# Site.telescope_limits (whose position bounds are enforced by
# validate_trajectory_bounds and whose rate bounds are advisory as well).
# Exceeding these values emits PointingWarning.
MAX_REASONABLE_SCAN_WIDTH_DEG: float = 30.0
"""Maximum scan width/height (or azimuth throw) before a warning is issued."""

MAX_REASONABLE_DAISY_RADIUS_DEG: float = 15.0
"""Maximum Daisy scan radius before a warning is issued."""

MAX_REASONABLE_VELOCITY_DEG_S: float = 5.0
"""Maximum scan velocity before a warning is issued."""

MAX_REASONABLE_ACCELERATION_DEG_S2: float = 3.0
"""Maximum scan acceleration before a warning is issued."""


def _require_positive(value: float, name: str) -> None:
    """Raise unless ``value`` is a finite, strictly positive number.

    ``value <= 0`` alone lets NaN and infinity through: NaN fails every
    comparison, and infinity is genuinely positive. Both then surface far
    from the config that accepted them (as ``cannot convert float NaN to
    integer`` from the leg quantiser, or as an empty sample array), so the
    finiteness test belongs here with the sign test.

    Parameters
    ----------
    value : float
        The field being validated.
    name : str
        Field name used in the message.

    Raises
    ------
    ValueError
        If ``value`` is not finite or is not strictly positive.
    """
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _require_non_negative(value: float, name: str) -> None:
    """Raise unless ``value`` is a finite, non-negative number.

    The non-negative counterpart of :func:`_require_positive`; see that
    function for why finiteness is checked alongside the sign.

    Parameters
    ----------
    value : float
        The field being validated.
    name : str
        Field name used in the message.

    Raises
    ------
    ValueError
        If ``value`` is not finite or is negative.
    """
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _warn_if_unusual(value: float, threshold: float, label: str, unit: str) -> None:
    """Emit :class:`PointingWarning` if ``value`` exceeds ``threshold``.

    Shared helper for config ``__post_init__`` routines so each config
    doesn't repeat the same three-line warn-on-threshold pattern.

    Parameters
    ----------
    value : float
        Observed value.
    threshold : float
        Advisory upper bound.
    label : str
        Short human-readable description of the field (e.g. ``"Scan width"``).
    unit : str
        Unit string appended to both the value and the threshold
        (e.g. ``"deg"``, ``"deg/s"``, ``"deg/s^2"``).
    """
    if value > threshold:
        # 4 frames out: this helper, __post_init__, the dataclass-generated
        # __init__ (whose filename is "<string>"), then the caller. A 3 here
        # attributed every advisory to that generated __init__.
        warnings.warn(
            f"{label} {value} {unit} is unusually large (> {threshold} {unit}).",
            PointingWarning,
            stacklevel=4,
        )


@dataclass(frozen=True)
class ScanConfig:
    """Base configuration for all scan patterns.

    Parameters
    ----------
    timestep : float
        Time between trajectory points in seconds.

    Raises
    ------
    ValueError
        If timestep is not positive.
    """

    timestep: float

    def __post_init__(self) -> None:
        _require_positive(self.timestep, "timestep")


@dataclass(frozen=True)
class ConstantElScanConfig(ScanConfig):
    """Configuration for constant elevation scan.

    Parameters
    ----------
    az_start : float
        Starting azimuth in degrees.
    az_stop : float
        Ending azimuth in degrees.
    elevation : float
        Fixed elevation in degrees.
    az_speed : float
        Azimuth scan speed in azimuth coordinate degrees/second
        (not on-sky). The on-sky speed is
        ``az_speed * cos(elevation)``; this is the mount-frame rate the
        telescope executes.
    az_accel : float
        Azimuth acceleration in azimuth coordinate
        degrees/second^2 (not on-sky).
    timestep : float
        Time between trajectory points in seconds.

    Raises
    ------
    ValueError
        If az_speed or az_accel is not positive.

    Warns
    -----
    PointingWarning
        If the azimuth throw, speed or acceleration is unusually large, or
        if the turnaround overshoot exceeds the science throw (the scan
        would spend most of its time turning around).

    Notes
    -----
    On-sky vs azimuth-coordinate speed: at higher elevations the same
    ``az_speed`` covers proportionally less sky because azimuth lines of
    constant elevation get smaller toward the pole. The on-sky angular
    rate is ``az_speed * cos(elevation)``. Worked example for typical
    FYST elevations:

    +-----------+----------------------+--------------------+
    | Elevation | cos(el)              | On-sky / coord (%) |
    +===========+======================+====================+
    | 30°       | 0.866                | 86.6               |
    +-----------+----------------------+--------------------+
    | 45°       | 0.707                | 70.7               |
    +-----------+----------------------+--------------------+
    | 60°       | 0.500                | 50.0               |
    +-----------+----------------------+--------------------+
    | 75°       | 0.259                | 25.9               |
    +-----------+----------------------+--------------------+

    A planner that wants a fixed on-sky scan rate (e.g. for noise
    uniformity) must scale ``az_speed`` by ``1/cos(elevation)`` for
    each target; ``plan_constant_el_scan``'s ``velocity`` argument is
    the same mount-frame quantity, passed through unscaled. (Pong/daisy
    ``velocity`` is on-sky, a different frame.)
    """

    az_start: float
    az_stop: float
    elevation: float
    az_speed: float
    az_accel: float

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_positive(self.az_speed, "az_speed")
        _require_positive(self.az_accel, "az_accel")
        az_throw = abs(self.az_stop - self.az_start)
        _warn_if_unusual(az_throw, MAX_REASONABLE_SCAN_WIDTH_DEG, "Azimuth throw", "deg")
        _warn_if_unusual(self.az_speed, MAX_REASONABLE_VELOCITY_DEG_S, "Azimuth speed", "deg/s")
        _warn_if_unusual(
            self.az_accel,
            MAX_REASONABLE_ACCELERATION_DEG_S2,
            "Azimuth acceleration",
            "deg/s^2",
        )
        d_half_turn = turnaround_overshoot_deg(self.az_speed, self.az_accel)
        if d_half_turn > az_throw:
            warnings.warn(
                f"Turnaround distance ({d_half_turn:.1f} deg) exceeds science throw "
                f"({az_throw:.1f} deg). The telescope will spend most of its time "
                f"in turnarounds. Consider increasing az_accel or decreasing az_speed.",
                PointingWarning,
                # 3 frames out: __post_init__, the generated __init__, the caller.
                stacklevel=3,
            )


@dataclass(frozen=True)
class PongScanConfig(ScanConfig):
    """Configuration for Pong (curvy box) scan.

    The Pong pattern is a closed-path scan optimized for uniformly covering
    rectangular regions. It uses Fourier-approximated triangle waves to
    create smooth turnarounds at the edges while maintaining efficient
    coverage.

    Parameters
    ----------
    width : float
        Width of scan region in degrees. Must be positive.
    height : float
        Height of scan region in degrees. Must be positive.
    spacing : float
        Space between scan lines in degrees. Must be positive.
    velocity : float
        Mean diagonal scan speed in sky-offset degrees/second (the
        tangent-plane speed, not azimuth coordinate velocity). Note this is
        the *mean* cruise speed: each axis follows a truncated Fourier triangle
        wave whose slope ripples about the nominal rate, so the *peak* on-sky
        diagonal speed exceeds ``velocity``. Measured on the shipped pattern
        the overshoot oscillates roughly between 9 and 18 percent for
        ``num_terms >= 4``, varying with ``num_terms`` and the scan geometry
        rather than converging; coarser truncations are worse, about 19
        percent at ``num_terms=3``, 20 percent at ``num_terms=2`` and 27
        percent at ``num_terms=1``. A planner
        sizing ``velocity`` against axis rate limits should budget for the
        top of that band; the realized mount-frame velocity is checked by
        ``validate_trajectory_dynamics``. Must be positive.
    num_terms : int
        Fourier terms for triangle wave approximation.
    angle : float
        Rotation angle of pattern in degrees.
    timestep : float
        Time between trajectory points in seconds.

    Raises
    ------
    ValueError
        If width, height, spacing, or velocity is not positive.
        If num_terms is less than 1.

    Warns
    -----
    PointingWarning
        If the scan width, height or velocity is unusually large.

    Notes
    -----
    The scan geometry is computed using a flat-sky (tangent-plane)
    approximation.  This is accurate for scan dimensions up to about
    10 degrees; beyond that, distortion at the field edges becomes
    significant. For very large scans, consider tiling with smaller
    overlapping fields.
    """

    width: float
    height: float
    spacing: float
    velocity: float
    num_terms: int
    angle: float

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_positive(self.width, "width")
        _require_positive(self.height, "height")
        _require_positive(self.spacing, "spacing")
        _require_positive(self.velocity, "velocity")
        if self.num_terms < 1:
            raise ValueError(f"num_terms must be at least 1, got {self.num_terms}")
        _warn_if_unusual(self.width, MAX_REASONABLE_SCAN_WIDTH_DEG, "Scan width", "deg")
        _warn_if_unusual(self.height, MAX_REASONABLE_SCAN_WIDTH_DEG, "Scan height", "deg")
        _warn_if_unusual(self.velocity, MAX_REASONABLE_VELOCITY_DEG_S, "Scan velocity", "deg/s")


@dataclass(frozen=True)
class PongAltAzScanConfig(ScanConfig):
    """Configuration for a Curvy-Pong scan about a fixed AltAz center.

    Executes the same Fourier-truncated Pong pattern as
    :class:`PongScanConfig`, but about a fixed horizon-frame center
    (``az_center``, ``el_center``) with no sky tracking. The on-sky
    tangent-plane pattern is generated exactly as for the celestial Pong,
    then mapped into telescope coordinates by::

        az = x_offset / cos(radians(el_center)) + az_center
        el = y_offset + el_center

    Consequently ``width``, ``height``, ``spacing``, and ``velocity`` are
    tangent-plane (on-sky) quantities, identical in meaning to the
    :class:`PongScanConfig` fields of the same name. The azimuth coordinate
    is stretched by ``1 / cos(el_center)``: the azimuth-coordinate extent is
    ``width / cos(el_center)`` up to the vertex quantisation: the box is
    rounded up to a whole number of ``sqrt(2) * spacing`` steps, and further
    to make the two vertex counts coprime, so the realised extent runs from a
    few percent narrower at fine ``spacing`` to tens of percent wider once
    ``spacing`` is a large fraction of ``width``; the azimuth-coordinate speed
    exceeds the on-sky speed by the same ``1 / cos(el_center)`` factor. Pick
    ``velocity`` against the mount
    azimuth-rate limit with that factor (and the Pong peak-speed overshoot,
    see :class:`PongScanConfig`) in mind.

    Unlike the celestial Pong, no coordinate transform or ``start_time`` is
    needed to build the pattern (it is an :class:`AltAzPattern`); the mapping
    above is a static horizon-frame projection.

    Parameters
    ----------
    az_center : float
        Azimuth of the pattern center in degrees.
    el_center : float
        Elevation of the pattern center in degrees. Must be in the open
        interval ``(0, 90)`` so ``cos(el_center)`` is defined and nonzero.
    width : float
        On-sky width of the scan region in degrees (cross-elevation extent
        before the ``1 / cos(el_center)`` azimuth stretch). Must be positive.
    height : float
        On-sky height of the scan region in degrees (elevation extent).
        Must be positive.
    spacing : float
        On-sky spacing between scan lines in degrees. Must be positive.
    velocity : float
        Mean diagonal on-sky scan speed in degrees/second (tangent-plane,
        not azimuth-coordinate). The peak on-sky speed exceeds this (see
        :class:`PongScanConfig`); the azimuth-coordinate speed exceeds the
        on-sky speed by ``1 / cos(el_center)``. Must be positive.
    num_terms : int, optional
        Number of Fourier terms for the triangle-wave approximation.
        Default is 4 (matching :class:`PongScanConfig` usage). Must be >= 1.
    angle : float, optional
        Rotation angle of the on-sky pattern in degrees, applied in the
        tangent plane before the horizon-frame mapping. Default is 0.0.
    timestep : float, optional
        Time between trajectory points in seconds. Default is 0.1. Must be
        positive.

    Raises
    ------
    ValueError
        If width, height, spacing, or velocity is not positive, if
        num_terms is less than 1, or if el_center is not in ``(0, 90)``.

    Warns
    -----
    PointingWarning
        If the scan width, height or velocity is unusually large, or if the
        derived azimuth-coordinate speed ``velocity / cos(el_center)`` is,
        which is a quantity the caller never passed in.

    Notes
    -----
    The scan geometry uses a flat-sky (tangent-plane) approximation, so it
    is accurate for scan dimensions up to about 10 degrees; beyond that,
    field-edge distortion becomes significant.
    """

    az_center: float
    el_center: float
    width: float
    height: float
    spacing: float
    velocity: float
    # kw_only so these defaulted fields may follow the required fields above
    # despite the base class declaring ``timestep`` without a default.
    num_terms: int = field(default=4, kw_only=True)
    angle: float = field(default=0.0, kw_only=True)
    timestep: float = field(default=0.1, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_positive(self.width, "width")
        _require_positive(self.height, "height")
        _require_positive(self.spacing, "spacing")
        _require_positive(self.velocity, "velocity")
        if self.num_terms < 1:
            raise ValueError(f"num_terms must be at least 1, got {self.num_terms}")
        if not 0.0 < self.el_center < 90.0:
            raise ValueError(
                f"el_center must be in (0, 90) degrees so cos(el_center) is nonzero, "
                f"got {self.el_center}"
            )
        _warn_if_unusual(self.width, MAX_REASONABLE_SCAN_WIDTH_DEG, "Scan width", "deg")
        _warn_if_unusual(self.height, MAX_REASONABLE_SCAN_WIDTH_DEG, "Scan height", "deg")
        _warn_if_unusual(self.velocity, MAX_REASONABLE_VELOCITY_DEG_S, "Scan velocity", "deg/s")
        # The azimuth-coordinate speed is inflated by 1/cos(el_center); warn on
        # that realized quantity too, since it is what the mount must slew.
        az_coord_velocity = self.velocity / math.cos(math.radians(self.el_center))
        _warn_if_unusual(
            az_coord_velocity,
            MAX_REASONABLE_VELOCITY_DEG_S,
            "Azimuth-coordinate velocity",
            "deg/s",
        )


@dataclass(frozen=True)
class DaisyScanConfig(ScanConfig):
    """Configuration for Daisy (Constant Velocity petal) scan.

    The Daisy pattern is a constant-velocity pattern optimized for point sources.
    The telescope moves in curved petal-shaped paths that repeatedly cross
    the center of the field, ensuring good coverage of the central region.

    Parameters
    ----------
    radius : float
        Characteristic radius R0 in on-sky degrees. Must be positive.
    velocity : float
        Scan velocity in sky-offset degrees/second. This is the
        speed in the tangent plane, not azimuth coordinate
        velocity. Must be positive.
    turn_radius : float
        Radius of curvature for turns in on-sky degrees. Must be positive.
    avoidance_radius : float
        Radius to avoid near center in on-sky degrees. Must be
        non-negative.
    start_acceleration : float
        Ramp-up acceleration in on-sky degrees/second^2 (the mount-frame
        azimuth acceleration exceeds it by ``1 / cos(el)``). Must be
        positive.
    y_offset : float
        Initial y offset in on-sky degrees.
    timestep : float
        Time between trajectory points in seconds.

    Raises
    ------
    ValueError
        If radius, velocity, or turn_radius is not positive.
        If avoidance_radius is negative.
        If start_acceleration is not positive.

    Warns
    -----
    PointingWarning
        If the daisy radius, velocity or start acceleration is unusually
        large.

    Notes
    -----
    The internal simulation uses a timestep of at most ~1/150 s (the output
    timestep when that is finer) for accurate curve approximation during
    turns. Extreme parameter combinations (very high velocity with very
    tight turn_radius) may produce inaccurate curves because the Taylor
    series approximation assumes small arc lengths per step. If the arc
    length per internal step (at most velocity / 150) approaches the
    turn_radius, consider reducing velocity or increasing turn_radius.

    A Daisy built with ``.duration(D)`` samples the integrator's own grid and
    spans ``[0, D - timestep]``, so ``trajectory.duration`` reports
    ``D - timestep``: one timestep short of the other patterns (which span
    ``[0, D]``). This is deliberate: sampling on the integrator grid avoids the
    ~1% velocity bias that a stretched ``linspace(0, D)`` time axis would inject.
    The per-sample ``times`` array is internally self-consistent, so serialized
    output (``to_path_format``) and ``/path`` dispatch are unaffected.
    """

    radius: float
    velocity: float
    turn_radius: float
    avoidance_radius: float
    start_acceleration: float
    y_offset: float

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_positive(self.radius, "radius")
        _require_positive(self.velocity, "velocity")
        _require_positive(self.turn_radius, "turn_radius")
        _require_non_negative(self.avoidance_radius, "avoidance_radius")
        _require_positive(self.start_acceleration, "start_acceleration")
        _warn_if_unusual(self.radius, MAX_REASONABLE_DAISY_RADIUS_DEG, "Daisy radius", "deg")
        _warn_if_unusual(self.velocity, MAX_REASONABLE_VELOCITY_DEG_S, "Scan velocity", "deg/s")
        _warn_if_unusual(
            self.start_acceleration,
            MAX_REASONABLE_ACCELERATION_DEG_S2,
            "Start acceleration",
            "deg/s^2",
        )


@dataclass(frozen=True)
class DaisyAltAzScanConfig(ScanConfig):
    """Configuration for a Constant-Velocity Daisy scan about a fixed AltAz center.

    Executes the same Constant-Velocity petal pattern as
    :class:`DaisyScanConfig`, but about a fixed horizon-frame center
    (``az_center``, ``el_center``) with no sky tracking. The on-sky
    tangent-plane pattern is generated exactly as for the celestial Daisy,
    then mapped into telescope coordinates by::

        az = x_offset / cos(radians(el_center)) + az_center
        el = y_offset + el_center

    Consequently ``radius``, ``velocity``, ``turn_radius``,
    ``avoidance_radius``, and ``start_acceleration`` are tangent-plane
    (on-sky) quantities, identical in meaning to the :class:`DaisyScanConfig`
    fields of the same name. The azimuth coordinate is stretched by
    ``1 / cos(el_center)``: the azimuth-coordinate extent is about
    ``2 * r_max / cos(el_center)``, where the petal's reach
    ``r_max = sqrt(radius**2 + turn_radius**2) + turn_radius`` exceeds
    ``radius`` because the petal only starts to turn once it is ``radius``
    from the center, and the azimuth-coordinate speed exceeds the on-sky
    speed by the same ``1 / cos(el_center)`` factor. Pick ``velocity``
    against the mount azimuth-rate limit with that factor in mind.

    Unlike the celestial Daisy, no coordinate transform or ``start_time`` is
    needed to build the pattern (it is an :class:`AltAzPattern`); the mapping
    above is a static horizon-frame projection.

    Parameters
    ----------
    az_center : float
        Azimuth of the pattern center in degrees.
    el_center : float
        Elevation of the pattern center in degrees. Must be in the open
        interval ``(0, 90)`` so ``cos(el_center)`` is defined and nonzero.
    radius : float
        On-sky characteristic radius R0 in degrees. Must be positive.
    velocity : float
        On-sky scan velocity in degrees/second (tangent-plane, not
        azimuth-coordinate). The azimuth-coordinate speed exceeds this by
        ``1 / cos(el_center)``. Must be positive.
    turn_radius : float
        On-sky radius of curvature for turns in degrees. Must be positive.
    avoidance_radius : float
        On-sky radius to avoid near center in degrees. Must be non-negative.
    start_acceleration : float
        On-sky ramp-up acceleration in degrees/second^2. Must be positive.
    y_offset : float, optional
        Initial on-sky y offset in degrees. Default is 0.0 (start at center).
    timestep : float, optional
        Time between trajectory points in seconds. Default is 0.1. Must be
        positive.

    Raises
    ------
    ValueError
        If radius, velocity, or turn_radius is not positive, if
        avoidance_radius is negative, if start_acceleration is not positive,
        or if el_center is not in ``(0, 90)``.

    Warns
    -----
    PointingWarning
        If the daisy radius, velocity or start acceleration is unusually
        large, or if the derived azimuth-coordinate speed
        ``velocity / cos(el_center)`` is, which is a quantity the caller
        never passed in.

    Notes
    -----
    The internal simulation uses a timestep of at most ~1/150 s (the output
    timestep when that is finer) for accurate curve approximation during
    turns, exactly as for the celestial Daisy. The duration reporting
    caveat is also shared: a Daisy built with ``.duration(D)`` spans
    ``[0, D - timestep]``, see :class:`DaisyScanConfig`.
    """

    az_center: float
    el_center: float
    radius: float
    velocity: float
    turn_radius: float
    avoidance_radius: float
    start_acceleration: float
    # kw_only so these defaulted fields may follow the required fields above
    # despite the base class declaring ``timestep`` without a default.
    y_offset: float = field(default=0.0, kw_only=True)
    timestep: float = field(default=0.1, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_positive(self.radius, "radius")
        _require_positive(self.velocity, "velocity")
        _require_positive(self.turn_radius, "turn_radius")
        _require_non_negative(self.avoidance_radius, "avoidance_radius")
        _require_positive(self.start_acceleration, "start_acceleration")
        if not 0.0 < self.el_center < 90.0:
            raise ValueError(
                f"el_center must be in (0, 90) degrees so cos(el_center) is nonzero, "
                f"got {self.el_center}"
            )
        _warn_if_unusual(self.radius, MAX_REASONABLE_DAISY_RADIUS_DEG, "Daisy radius", "deg")
        _warn_if_unusual(self.velocity, MAX_REASONABLE_VELOCITY_DEG_S, "Scan velocity", "deg/s")
        _warn_if_unusual(
            self.start_acceleration,
            MAX_REASONABLE_ACCELERATION_DEG_S2,
            "Start acceleration",
            "deg/s^2",
        )
        # The azimuth-coordinate speed is inflated by 1/cos(el_center); warn on
        # that realized quantity too, since it is what the mount must slew.
        az_coord_velocity = self.velocity / math.cos(math.radians(self.el_center))
        _warn_if_unusual(
            az_coord_velocity,
            MAX_REASONABLE_VELOCITY_DEG_S,
            "Azimuth-coordinate velocity",
            "deg/s",
        )


@dataclass(frozen=True)
class SiderealTrackConfig(ScanConfig):
    """Configuration for sidereal tracking.

    Sidereal tracking follows a fixed RA/Dec position as it moves
    across the sky due to Earth's rotation.

    Parameters
    ----------
    timestep : float
        Time between trajectory points in seconds.
    """


@dataclass(frozen=True)
class PlanetTrackConfig(ScanConfig):
    """Configuration for planet tracking.

    Planet tracking follows a solar system body as it moves
    across the sky.

    Parameters
    ----------
    body : str
        Name of solar system body to track.
    timestep : float
        Time between trajectory points in seconds.

    Raises
    ------
    ValueError
        If body is not a valid solar system body name.
    """

    body: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.body.lower() not in SOLAR_SYSTEM_BODIES:
            raise ValueError(f"Unknown body '{self.body}'. Valid: {sorted(SOLAR_SYSTEM_BODIES)}")


@dataclass(frozen=True)
class SatelliteTrackConfig(ScanConfig):
    """Configuration for planetary-satellite tracking.

    Satellite tracking follows a planetary satellite (e.g. Titan) as it
    moves across the sky, used for submillimetre flux calibration. The
    satellite's apparent centroid is tracked as an unresolved point
    source; no disk model is applied (Titan's ~0.8 arcsec disk is well
    below the Prime-Cam beam).

    Unlike :class:`PlanetTrackConfig`, the body is resolved from a JPL
    satellite SPK kernel rather than astropy's builtin ephemeris. The
    kernel is supplied via ``satellite_kernel`` or, if that is ``None``,
    the ``FYST_SATELLITE_KERNEL`` environment variable.

    Parameters
    ----------
    body : str
        Name of the planetary satellite to track (e.g. ``"titan"``).
    satellite_kernel : str or None, optional
        Path to a JPL satellite SPK kernel used to resolve the body.
        If ``None``, the ``FYST_SATELLITE_KERNEL`` environment variable
        is used at generation time.
    timestep : float
        Time between trajectory points in seconds.

    Raises
    ------
    ValueError
        If body is not a valid planetary-satellite name.
    """

    body: str
    satellite_kernel: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        body = self.body.lower()
        if body not in SATELLITE_BODIES:
            raise ValueError(f"Unknown satellite '{self.body}'. Valid: {sorted(SATELLITE_BODIES)}")
        # Frozen dataclass: normalise the stored body to lower-case so the
        # pattern metadata (``target_name``) is canonical regardless of input
        # casing. ``object.__setattr__`` is the standard frozen-field idiom.
        object.__setattr__(self, "body", body)


@dataclass(frozen=True)
class LinearMotionConfig(ScanConfig):
    """Configuration for linear motion.

    Linear motion moves at constant velocity in Az/El space.

    Parameters
    ----------
    az_start : float
        Starting azimuth in degrees.
    el_start : float
        Starting elevation in degrees.
    az_velocity : float
        Azimuth velocity in azimuth coordinate degrees/second
        (not on-sky). The on-sky component is
        ``az_velocity * cos(elevation)``.
    el_velocity : float
        Elevation velocity in degrees/second.
    timestep : float
        Time between trajectory points in seconds.

    Raises
    ------
    ValueError
        If ``timestep`` is not positive, or if any of the four motion
        fields is not finite.
    """

    az_start: float
    el_start: float
    az_velocity: float
    el_velocity: float

    def __post_init__(self) -> None:
        super().__post_init__()
        # Sign and magnitude are all free here (a velocity may be zero or
        # negative, a start position may be anywhere in the mount range), so
        # finiteness is the whole contract. Without it a NaN reaches the
        # generator and produces an all-NaN trajectory that only the
        # position-bounds check refuses, with no mention of which field.
        for name, value in (
            ("az_start", self.az_start),
            ("el_start", self.el_start),
            ("az_velocity", self.az_velocity),
            ("el_velocity", self.el_velocity),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value}")
