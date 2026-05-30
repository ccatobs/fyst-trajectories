"""Constant-elevation scan planner (public via :mod:`fyst_trajectories.planning`)."""

from typing import TYPE_CHECKING

from astropy.time import Time

from ..coordinates import Coordinates
from ..patterns.configs import ConstantElScanConfig
from ..site import AtmosphericConditions, Site
from ._ce_geometry import (
    _compute_ce_az_range,
    _compute_ce_duration,
    _compute_ce_duration_from_lsa,
)
from ._helpers import _build_altaz_trajectory
from ._sun_safety import _check_field_sun_safety
from ._types import ConstantElComputedParams, FieldRegion, ScanBlock, validate_computed_params

if TYPE_CHECKING:
    from ..offsets import InstrumentOffset


def plan_constant_el_scan(
    field: FieldRegion,
    elevation: float,
    velocity: float,
    site: Site,
    start_time: str | Time,
    rising: bool = True,
    angle: float = 0.0,
    az_accel: float = 1.0,
    timestep: float = 0.1,
    detector_offset: "InstrumentOffset | None" = None,
    az_padding: float = 2.0,
    atmosphere: AtmosphericConditions | None = None,
    max_search_hours: float = 12.0,
    step_seconds: float = 30.0,
    lsa_window: tuple[float, float] | list[float] | None = None,
) -> ScanBlock:
    """Plan a constant-elevation scan that covers a FieldRegion.

    Auto-computes the azimuth range and observation duration from the
    field geometry, matching the algorithm used by the FYST scan strategy
    planning tools.

    The function:

    1. Finds when the RA edges of the (optionally rotated) field cross
       the target elevation (determines start/end time and duration).
    2. Computes the azimuth range that covers the entire field at that
       elevation at the midpoint of the observation.
    3. Computes n_scans from the duration and single-leg sweep time.
    4. Builds and returns a ScanBlock.

    Parameters
    ----------
    field : FieldRegion
        Rectangular field specification (center RA/Dec, width, height).
    elevation : float
        Fixed elevation for the scan in degrees.
    velocity : float
        Azimuth scan speed in azimuth coordinate degrees/second
        (not on-sky). The on-sky speed is
        ``velocity * cos(elevation)``. This is the value sent
        directly to the Vertex ACU. Must be positive.
    site : Site
        Telescope site configuration.
    start_time : str or Time
        Approximate start time for the search window. The function
        searches up to ``max_search_hours`` forward from this time to
        find when the field edges cross the target elevation.
    rising : bool, optional
        If True (default), use the rising crossing; if False, the
        setting crossing.
    angle : float, optional
        Rotation angle of the field region in degrees. Default is 0.0.
    az_accel : float, optional
        Azimuth acceleration in degrees/second^2. Default is 1.0.
    timestep : float, optional
        Time between trajectory points in seconds. Default is 0.1.
    detector_offset : InstrumentOffset or None, optional
        If provided, adjust the trajectory for this detector offset.
    az_padding : float, optional
        Extra azimuth padding in degrees on each side of the computed
        range. Default is 2.0.
    atmosphere : AtmosphericConditions or None, optional
        Atmospheric conditions for refraction correction. If None,
        no refraction is applied.
    max_search_hours : float, optional
        Maximum time to search forward in hours for elevation crossings.
        Default is 12.0.
    step_seconds : float, optional
        Time step in seconds for the elevation crossing search.
        Default is 30.0.
    lsa_window : tuple of (min_lsa, max_lsa), optional
        Local Sidereal Angle window in degrees. When supplied, the scan
        ``start_time`` and ``duration`` are derived from this LSA window
        rather than from RA-edge elevation crossings: the planner finds
        the first time at or after ``start_time`` at which Local
        Sidereal Time (in degrees) increases through ``min_lsa``, and
        the scan runs for ``(max_lsa - min_lsa) mod 360 / 15`` hours.
        Wrap-around windows where ``max_lsa < min_lsa`` are supported
        (e.g. ``(310.0, 10.0)`` is a 60°/15 = 4 hour scan crossing the
        LSA = 0/360 boundary). Both endpoints must lie in ``[0, 360)``
        and must not be equal. ``rising`` is still honored — it
        controls the azimuth-range computation (rising/setting half of
        the field's transit) but does NOT affect the LSA-derived
        ``obs_start`` / ``obs_end``. ``max_search_hours`` and
        ``step_seconds`` still bound the search horizon. Use this for
        operator-driven LSA-windowed scheduling (e.g. ACT/Deep56-style
        constant-elevation patches). Default is ``None``, which
        preserves the elevation-crossing behavior.

    Returns
    -------
    ScanBlock
        Planned observation containing trajectory, config, and computed
        parameters (az_start, az_stop, az_throw, start_time, end_time,
        duration).

    Raises
    ------
    ValueError
        If the elevation crossings cannot be found within the search
        window, or if ``lsa_window`` is supplied with equal endpoints
        or values outside ``[0, 360)``.
    PointingError
        If ``lsa_window`` is supplied and LST never increases through
        ``min_lsa`` within ``max_search_hours`` of ``start_time``.
    AzimuthBoundsError
        If the computed azimuth range exceeds telescope limits.
    ElevationBoundsError
        If the elevation exceeds telescope limits.

    Notes
    -----
    Sun-safety pre-flight runs twice when ``lsa_window`` is supplied:
    once at ``start_time`` (the search anchor) and once at the
    resolved ``obs_start``. The LSA branch can delay the observation
    by several hours relative to ``start_time``, during which the
    Sun moves ~15°/hour — a field that is safely far from the Sun
    at ``start_time`` may not be safe by the time the LSA window
    opens. The elevation-crossing path runs the check only at
    ``start_time`` because ``obs_start`` resolves within a typical
    few-seconds search offset of the anchor.

    Examples
    --------
    Elevation-crossing mode (the default):

    >>> from astropy.time import Time
    >>> from fyst_trajectories import get_fyst_site
    >>> from fyst_trajectories.planning import FieldRegion, plan_constant_el_scan
    >>> site = get_fyst_site()
    >>> field = FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7)
    >>> block = plan_constant_el_scan(
    ...     field=field,
    ...     elevation=50.0,
    ...     velocity=0.5,
    ...     site=site,
    ...     start_time=Time("2026-03-15T17:00:00", scale="utc"),
    ...     rising=True,
    ...     angle=170.0,
    ... )

    LSA-window mode (Deep56-style, 4-hour wrap-around window):

    >>> deep56 = FieldRegion(ra_center=0.0, dec_center=-2.0, width=60.0, height=14.0)
    >>> block = plan_constant_el_scan(  # doctest: +SKIP
    ...     field=deep56,
    ...     elevation=50.0,
    ...     velocity=0.5,
    ...     site=site,
    ...     start_time=Time("2026-09-15T00:00:00", scale="utc"),
    ...     rising=True,
    ...     lsa_window=(310.0, 10.0),
    ... )
    """
    if isinstance(start_time, str):
        start_time = Time(start_time, scale="utc")

    # Pre-flight sun-safety check at the *search anchor*. For the
    # elevation-crossing path this is essentially the observation start
    # (``obs_start`` resolves within seconds of ``start_time`` in the
    # typical case). For the LSA-window path we re-check below once
    # ``obs_start`` is known — the LSA branch can delay observation by
    # hours, during which the Sun moves ~15°/h, so a field that is
    # 50° from the Sun at ``start_time`` can be 5° from it at
    # ``obs_start``.
    _check_field_sun_safety(field.ra_center, field.dec_center, start_time, site)

    coords_obj = Coordinates(site, atmosphere=atmosphere)

    if lsa_window is not None:
        obs_start, obs_end, duration = _compute_ce_duration_from_lsa(
            lsa_window,
            coords_obj,
            start_time,
            max_search_hours=max_search_hours,
            step_seconds=step_seconds,
        )
        # Re-check sun safety at the resolved ``obs_start`` — see the
        # comment above for the rationale.
        _check_field_sun_safety(field.ra_center, field.dec_center, obs_start, site)
    else:
        obs_start, obs_end, duration = _compute_ce_duration(
            field,
            angle,
            elevation,
            coords_obj,
            start_time,
            rising,
            max_search_hours=max_search_hours,
            step_seconds=step_seconds,
        )

    az_min, az_max = _compute_ce_az_range(field, angle, coords_obj, obs_start, obs_end, az_padding)

    az_throw = az_max - az_min
    scan_leg_time = az_throw / velocity
    n_scans = max(1, round(duration / scan_leg_time))

    # Compute the actual duration from the CE pattern's cycle geometry so
    # the trajectory length matches n_scans exactly, rather than using the
    # elevation-crossing duration which may differ.
    # Factor 2: trapezoidal velocity profile = ramp-up time (v/a) + ramp-down time (v/a)
    t_turnaround = 2.0 * velocity / az_accel
    t_cruise = az_throw / velocity
    # ``n_scans`` cruises with ``n_scans - 1`` inter-leg turnarounds (the
    # trailing turnaround of the final leg is unused). The original
    # ``n_scans * (t_cruise + t_turnaround)`` form over-counted by one
    # turnaround for odd ``n_scans``.
    actual_duration = n_scans * t_cruise + max(0, n_scans - 1) * t_turnaround

    config = ConstantElScanConfig(
        timestep=timestep,
        az_start=az_min,
        az_stop=az_max,
        elevation=elevation,
        az_speed=velocity,
        az_accel=az_accel,
    )

    trajectory = _build_altaz_trajectory(
        site=site,
        config=config,
        duration=actual_duration,
        start_time=obs_start,
        atmosphere=atmosphere,
        detector_offset=detector_offset,
    )

    computed_params: ConstantElComputedParams = {
        "az_start": az_min,
        "az_stop": az_max,
        "az_throw": az_throw,
        "n_scans": n_scans,
        "start_time_iso": obs_start.iso,
        "end_time_iso": obs_end.iso,
        "duration": actual_duration,
    }
    validate_computed_params(computed_params, "constant_el")

    summary = (
        f"Constant-El scan: {field.width:.2f} x {field.height:.2f} deg field "
        f"at RA={field.ra_center:.3f}, Dec={field.dec_center:.3f}\n"
        f"  Elevation: {elevation:.2f} deg, "
        f"Az range: [{az_min:.2f}, {az_max:.2f}] deg "
        f"(throw: {az_throw:.2f} deg)\n"
        f"  Velocity: {velocity:.3f} deg/s, Acceleration: {az_accel:.3f} deg/s^2\n"
        f"  {'Rising' if rising else 'Setting'} pass: "
        f"{obs_start.iso[:19]} to {obs_end.iso[:19]}\n"
        f"  Scans: {n_scans}, Duration: {actual_duration:.1f}s "
        f"({actual_duration / 60:.1f}min), "
        f"Trajectory points: {trajectory.n_points}"
    )

    return ScanBlock(
        trajectory=trajectory,
        config=config,
        duration=actual_duration,
        computed_params=computed_params,
        summary=summary,
    )
