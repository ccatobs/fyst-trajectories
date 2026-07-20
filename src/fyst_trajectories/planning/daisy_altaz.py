"""AltAz Daisy scan planner (public via :mod:`fyst_trajectories.planning`)."""

from typing import TYPE_CHECKING

from astropy.time import Time

from ..patterns.configs import DaisyAltAzScanConfig
from ..site import AtmosphericConditions, Site
from ._helpers import _build_altaz_trajectory, _coerce_start_time
from ._sun_safety import _check_altaz_center_sun_safety
from ._types import DaisyAltAzComputedParams, ScanBlock, validate_computed_params

if TYPE_CHECKING:
    from ..dispatch import SunSafePredicate
    from ..offsets import InstrumentOffset


def plan_daisy_altaz_scan(
    az_center: float,
    el_center: float,
    radius: float,
    velocity: float,
    turn_radius: float,
    avoidance_radius: float,
    start_acceleration: float,
    site: Site,
    start_time: str | Time,
    timestep: float,
    duration: float,
    y_offset: float = 0.0,
    detector_offset: "InstrumentOffset | None" = None,
    atmosphere: AtmosphericConditions | None = None,
    sun_safe: "SunSafePredicate | None" = None,
) -> ScanBlock:
    """Plan a Constant-Velocity Daisy scan about a fixed AltAz center.

    Generates the same on-sky Daisy pattern as :func:`plan_daisy_scan`, but
    executes it about a fixed horizon-frame center (``az_center``,
    ``el_center``) with no sky tracking. The on-sky tangent-plane offsets
    are mapped into telescope coordinates by::

        az = x_offset / cos(radians(el_center)) + az_center
        el = y_offset + el_center

    so ``radius``, ``velocity``, ``turn_radius``, ``avoidance_radius``, and
    ``start_acceleration`` are tangent-plane (on-sky) quantities, identical
    in meaning to the :func:`plan_daisy_scan` arguments of the same name.
    The azimuth coordinate is stretched by ``1 / cos(el_center)``: the
    azimuth-coordinate extent is ``2 * radius / cos(el_center)`` and the
    azimuth-coordinate speed exceeds the on-sky ``velocity`` by the same
    factor.

    Parameters
    ----------
    az_center : float
        Azimuth of the pattern center in degrees.
    el_center : float
        Elevation of the pattern center in degrees. Must be in ``(0, 90)``.
    radius : float
        On-sky characteristic radius R0 in degrees. Must be positive.
    velocity : float
        On-sky scan velocity in degrees/second (tangent-plane, not
        azimuth-coordinate velocity). Must be positive.
    turn_radius : float
        On-sky radius of curvature for turns in degrees. Must be positive.
    avoidance_radius : float
        On-sky radius to avoid near center in degrees. Must be non-negative.
    start_acceleration : float
        On-sky ramp-up acceleration in degrees/second^2. Must be positive.
    site : Site
        Telescope site configuration.
    start_time : str or Time
        Observation start time. Accepts an ISO string or
        ``astropy.time.Time``. Used to anchor the trajectory timestamp and
        to convert the center to RA/Dec for the sun-safety pre-flight check.
    timestep : float
        Time between trajectory points in seconds. Must be positive.
    duration : float
        Observation duration in seconds. Must be positive.
    y_offset : float, optional
        Initial on-sky y offset in degrees. Default is 0.0 (start at center).
    detector_offset : InstrumentOffset or None, optional
        If provided, adjust the trajectory so this detector tracks the
        center instead of the boresight.
    atmosphere : AtmosphericConditions or None, optional
        Not used for the AltAz geometry (no coordinate transforms), accepted
        for parity with the other planners. Default is None.
    sun_safe : SunSafePredicate or None, optional
        Sun-safety predicate implementing the
        :class:`~fyst_trajectories.dispatch.SunSafePredicate` contract,
        forwarded to the center pre-flight check. ``None`` (default) keeps
        the built-in scalar exclusion-radius check; an injected predicate is
        consulted instead, so the directional sun-avoidance model (future
        shared library) is honored end-to-end. Warn-only.

    Returns
    -------
    ScanBlock
        Planned observation containing trajectory, config, and computed
        parameters (duration, az_center, el_center).

    Raises
    ------
    ValueError
        If any config field is invalid (non-positive
        radius/velocity/turn_radius/start_acceleration, negative
        avoidance_radius, or el_center outside ``(0, 90)``).
    TrajectoryBoundsError
        If the trajectory exceeds telescope limits.

    Examples
    --------
    >>> from astropy.time import Time
    >>> from fyst_trajectories import get_fyst_site
    >>> from fyst_trajectories.planning import plan_daisy_altaz_scan
    >>> site = get_fyst_site()
    >>> block = plan_daisy_altaz_scan(
    ...     az_center=120.0,
    ...     el_center=60.0,
    ...     radius=0.5,
    ...     velocity=0.3,
    ...     turn_radius=0.2,
    ...     avoidance_radius=0.0,
    ...     start_acceleration=0.5,
    ...     site=site,
    ...     start_time=Time("2026-03-15T04:00:00", scale="utc"),
    ...     timestep=0.1,
    ...     duration=300.0,
    ... )
    """
    start_time = _coerce_start_time(start_time)

    # Building the config first validates el_center, so a bad value raises the
    # config's clear message instead of an astropy latitude error below.
    config = DaisyAltAzScanConfig(
        az_center=az_center,
        el_center=el_center,
        radius=radius,
        velocity=velocity,
        turn_radius=turn_radius,
        avoidance_radius=avoidance_radius,
        start_acceleration=start_acceleration,
        y_offset=y_offset,
        timestep=timestep,
    )

    _check_altaz_center_sun_safety(
        site=site,
        az_center=az_center,
        el_center=el_center,
        start_time=start_time,
        sun_safe=sun_safe,
    )

    trajectory = _build_altaz_trajectory(
        site=site,
        config=config,
        duration=duration,
        start_time=start_time,
        atmosphere=atmosphere,
        detector_offset=detector_offset,
    )

    computed_params: DaisyAltAzComputedParams = {
        "duration": duration,
        "az_center": az_center,
        "el_center": el_center,
    }
    validate_computed_params(computed_params, "daisy_altaz")

    summary = (
        f"AltAz Daisy scan: radius={radius:.3f} deg on-sky "
        f"at az={az_center:.3f}, el={el_center:.3f}\n"
        f"  Velocity: {velocity:.3f} deg/s (on-sky), Turn radius: {turn_radius:.3f} deg\n"
        f"  Avoidance radius: {avoidance_radius:.3f} deg, "
        f"Start acceleration: {start_acceleration:.3f} deg/s^2\n"
        f"  Duration: {duration:.1f}s, "
        f"Trajectory points: {trajectory.n_points}"
    )

    return ScanBlock(
        trajectory=trajectory,
        config=config,
        duration=duration,
        computed_params=computed_params,
        summary=summary,
    )
