"""Daisy scan planner (public via :mod:`fyst_trajectories.planning`)."""

from typing import TYPE_CHECKING

from astropy.time import Time

from ..patterns.configs import DaisyScanConfig
from ..site import AtmosphericConditions, Site
from ._helpers import _build_celestial_trajectory
from ._sun_safety import _check_field_sun_safety
from ._types import DaisyComputedParams, ScanBlock, validate_computed_params

if TYPE_CHECKING:
    from ..offsets import InstrumentOffset


def plan_daisy_scan(
    ra: float,
    dec: float,
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
) -> ScanBlock:
    """Plan a Daisy scan centered on a single RA/Dec position.

    Parameters
    ----------
    ra : float
        Right Ascension of the source in degrees.
    dec : float
        Declination of the source in degrees.
    radius : float
        Characteristic radius R0 in degrees. Must be positive.
    velocity : float
        Scan velocity in sky-offset degrees/second. This is the
        speed in the tangent plane, not azimuth coordinate
        velocity. Must be positive.
    turn_radius : float
        Radius of curvature for turns in degrees. Must be positive.
    avoidance_radius : float
        Radius to avoid near center in degrees. Must be non-negative.
    start_acceleration : float
        Ramp-up acceleration in degrees/second^2. Must be positive.
    site : Site
        Telescope site configuration.
    start_time : str or Time
        Observation start time (required for celestial patterns).
        Accepts an ISO string or ``astropy.time.Time``.
    timestep : float
        Time between trajectory points in seconds. Must be positive.
    duration : float
        Observation duration in seconds. Must be positive.
    y_offset : float, optional
        Initial y offset in degrees. Default is 0.0 (start at center).
    detector_offset : InstrumentOffset or None, optional
        If provided, adjust the trajectory for this detector offset.
    atmosphere : AtmosphericConditions or None, optional
        Atmospheric conditions for refraction correction. If None,
        no refraction is applied.

    Returns
    -------
    ScanBlock
        Planned observation containing trajectory, config, and computed
        parameters.

    Raises
    ------
    TargetNotObservableError
        If the target is not observable at the requested time.
    TrajectoryBoundsError
        If the trajectory exceeds telescope limits.

    Examples
    --------
    >>> from astropy.time import Time
    >>> from fyst_trajectories import get_fyst_site
    >>> from fyst_trajectories.planning import plan_daisy_scan
    >>> site = get_fyst_site()
    >>> block = plan_daisy_scan(
    ...     ra=180.0,
    ...     dec=-30.0,
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
    if isinstance(start_time, str):
        start_time = Time(start_time, scale="utc")

    _check_field_sun_safety(ra, dec, start_time, site)

    config = DaisyScanConfig(
        timestep=timestep,
        radius=radius,
        velocity=velocity,
        turn_radius=turn_radius,
        avoidance_radius=avoidance_radius,
        start_acceleration=start_acceleration,
        y_offset=y_offset,
    )

    trajectory = _build_celestial_trajectory(
        site=site,
        ra=ra,
        dec=dec,
        config=config,
        duration=duration,
        start_time=start_time,
        atmosphere=atmosphere,
        detector_offset=detector_offset,
    )

    computed_params: DaisyComputedParams = {
        "duration": duration,
    }
    validate_computed_params(computed_params, "daisy")

    summary = (
        f"Daisy scan: radius={radius:.3f} deg at RA={ra:.3f}, Dec={dec:.3f}\n"
        f"  Velocity: {velocity:.3f} deg/s, Turn radius: {turn_radius:.3f} deg\n"
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
