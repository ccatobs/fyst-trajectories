"""Constant-elevation scan planner (public via :mod:`fyst_trajectories.planning`)."""

from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

from ..coordinates import Coordinates
from ..patterns.configs import ConstantElScanConfig
from ..site import AtmosphericConditions, Site
from ._ce_geometry import (
    _compute_ce_az_range,
    _compute_ce_duration,
    _compute_ce_duration_from_lsa,
    _quantize_ce_duration,
)
from ._helpers import _build_altaz_trajectory, _coerce_start_time
from ._sun_safety import (
    _SUN_SAFETY_ARC_N_SAMPLES,
    _check_arc_sun_safety,
    _check_field_sun_safety,
    _swept_arc_samples,
)
from ._types import ConstantElComputedParams, FieldRegion, ScanBlock, validate_computed_params

if TYPE_CHECKING:
    from ..dispatch import SunSafePredicate
    from ..offsets import InstrumentOffset


def plan_constant_el_scan(
    field: FieldRegion,
    elevation: float,
    velocity: float,
    site: Site,
    start_time: str | Time,
    rising: bool | None = None,
    angle: float = 0.0,
    az_accel: float = 1.0,
    timestep: float = 0.1,
    detector_offset: "InstrumentOffset | None" = None,
    az_padding: float = 2.0,
    atmosphere: AtmosphericConditions | None = None,
    max_search_hours: float = 12.0,
    step_seconds: float = 30.0,
    lsa_window: tuple[float, float] | list[float] | None = None,
    sun_safe: "SunSafePredicate | None" = None,
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
        ``velocity * cos(elevation)``; this is the mount-frame rate the
        telescope executes. Must be positive.
    site : Site
        Telescope site configuration.
    start_time : str or Time
        Approximate start time for the search window. The function
        searches up to ``max_search_hours`` forward from this time to
        find when the field edges cross the target elevation.
    rising : bool or None, optional
        Which elevation crossing to observe: ``True`` the rising one,
        ``False`` the setting one. Default ``None`` selects the rising
        crossing. Not accepted together with ``lsa_window``, which derives
        the timing from sidereal angle instead and has no crossing to
        choose.
    angle : float, optional
        Rotation angle of the field region in degrees. Default is 0.0.
    az_accel : float, optional
        Azimuth acceleration in azimuth coordinate degrees/second^2
        (mount frame, not on-sky). Default is 1.0.
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
        the scan spans ``(max_lsa - min_lsa) mod 360 / 15`` hours of
        UTC, about 0.3 percent longer than the sidereal window it names.
        ``ScanBlock.duration`` is that span quantised to whole azimuth
        legs, so it is a little shorter again.
        Wrap-around windows where ``max_lsa < min_lsa`` are supported
        (e.g. ``(310.0, 10.0)`` is a 60°/15 = 4 hour scan crossing the
        LSA = 0/360 boundary). Both endpoints must lie in ``[0, 360)``
        and must not be equal. The window fixes both the timing and the
        azimuth range, so ``rising`` has nothing left to choose and
        passing it alongside is rejected. ``max_search_hours`` and
        ``step_seconds`` still bound the search horizon. Use this for
        operator-driven LSA-windowed scheduling (e.g. ACT/Deep56-style
        constant-elevation patches). Default is ``None``, which
        preserves the elevation-crossing behavior.
    sun_safe : SunSafePredicate or None, optional
        Sun-safety predicate implementing the
        :class:`~fyst_trajectories.dispatch.SunSafePredicate` contract,
        forwarded to the field-center pre-flight check(s). ``None``
        (default) keeps the built-in scalar exclusion-radius check; an
        injected predicate is consulted instead, so the directional
        sun-avoidance model (see
        :func:`~fyst_trajectories.sun_models.make_sun_safe`) is honored
        end-to-end.
        Applied at every sun check the planner runs: the field-center
        pre-flight(s) and the swept-pass check described in the Notes.
        Warn-only.

    Returns
    -------
    ScanBlock
        Planned observation containing trajectory, config, and computed
        parameters (``az_start``, ``az_stop``, ``az_throw``, ``n_scans``,
        ``start_time_iso``, ``end_time_iso``, ``duration``).

    Raises
    ------
    ValueError
        If ``velocity`` is not positive, if the elevation crossings
        cannot be found within the search window, if ``rising`` and
        ``lsa_window`` are supplied together, or if ``lsa_window``
        is supplied with equal endpoints or values outside ``[0, 360)``.
    PointingError
        If ``lsa_window`` is supplied and LST never increases through
        ``min_lsa`` within ``max_search_hours`` of ``start_time``.
    AzimuthBoundsError
        If the computed azimuth range exceeds telescope limits.
    ElevationBoundsError
        If the elevation exceeds telescope limits.

    Notes
    -----
    Sun safety is screened in two stages. The field-center pre-flight is
    a cheap instant check that runs before the search, at ``start_time``,
    and again at the resolved ``obs_start`` when ``lsa_window`` delays the
    observation (the LSA branch can push it hours later, during which the
    Sun moves ~15°/hour). Once the pass is resolved, the planner then
    sweeps it: the commanded azimuth envelope (the swept window widened by
    the turnaround overshoot on each side) is sampled at the fixed
    ``elevation`` across the whole run, so a multi-hour block that starts
    clear of the Sun and ends inside the exclusion zone is caught. Both
    stages warn and never block.

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
    >>> block = plan_constant_el_scan(
    ...     field=deep56,
    ...     elevation=50.0,
    ...     velocity=0.5,
    ...     site=site,
    ...     start_time=Time("2026-09-15T00:00:00", scale="utc"),
    ...     lsa_window=(310.0, 10.0),
    ... )
    """
    if velocity <= 0:
        raise ValueError(f"velocity must be positive, got {velocity}")
    if lsa_window is not None and rising is not None:
        raise ValueError(
            "rising is not accepted with lsa_window: the sidereal window fixes both "
            "the timing and the azimuth range, so there is no crossing half to choose. "
            "Drop rising, or drop lsa_window to plan an elevation crossing."
        )

    start_time = _coerce_start_time(start_time)

    # Pre-flight sun-safety at the *search anchor*; the LSA path
    # re-checks at the resolved ``obs_start`` below. Rationale in the
    # Notes section.
    _check_field_sun_safety(field.ra_center, field.dec_center, start_time, site, sun_safe=sun_safe)

    coords_obj = Coordinates(site, atmosphere=atmosphere)

    if lsa_window is not None:
        obs_start, obs_end, duration = _compute_ce_duration_from_lsa(
            lsa_window,
            coords_obj,
            start_time,
            max_search_hours=max_search_hours,
            step_seconds=step_seconds,
        )
        # Re-check sun safety at the resolved ``obs_start``; see the
        # comment above for the rationale.
        _check_field_sun_safety(
            field.ra_center, field.dec_center, obs_start, site, sun_safe=sun_safe
        )
    else:
        obs_start, obs_end, duration = _compute_ce_duration(
            field,
            angle,
            elevation,
            coords_obj,
            start_time,
            True if rising is None else rising,
            max_search_hours=max_search_hours,
            step_seconds=step_seconds,
        )

    az_min, az_max = _compute_ce_az_range(field, angle, coords_obj, obs_start, obs_end, az_padding)

    az_throw = az_max - az_min
    # Quantise the elevation-crossing window into whole azimuth legs so the
    # trajectory length matches n_scans exactly, rather than using the raw
    # crossing duration which may differ.
    n_scans, actual_duration = _quantize_ce_duration(
        az_throw=az_throw,
        velocity=velocity,
        duration=duration,
        az_accel=az_accel,
    )

    # Sweep the resolved pass, not just its opening instant: a block can run
    # for hours while the Sun closes at ~15 deg/hour, so a pass that is clear
    # at ``obs_start`` can end deep inside the exclusion zone. The screened
    # span covers both the quantised trajectory length and ``obs_end``,
    # whichever runs longer, at the commanded azimuth envelope.
    swept_seconds = max(actual_duration, (obs_end - obs_start).sec)
    sweep_times = obs_start + TimeDelta(
        np.linspace(0.0, swept_seconds, _SUN_SAFETY_ARC_N_SAMPLES) * u.s
    )
    sweep_az, sweep_el, sweep_times = _swept_arc_samples(
        az_min=np.full(_SUN_SAFETY_ARC_N_SAMPLES, az_min),
        az_throw=az_throw,
        el_deg=elevation,
        times=sweep_times,
        az_speed=velocity,
        az_accel=az_accel,
    )
    _check_arc_sun_safety(
        coords_obj,
        site,
        sweep_az,
        sweep_el,
        sweep_times,
        f"constant-elevation scan at RA={field.ra_center:.3f}, Dec={field.dec_center:.3f}",
        sun_safe=sun_safe,
        stacklevel=3,
    )

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

    # Name the pass by what actually chose it: the crossing half on the
    # elevation-crossing path, the sidereal window on the LSA path (where
    # no half is chosen and claiming one would be a false report).
    if lsa_window is not None:
        pass_label = "LSA-window pass"
    else:
        pass_label = "Rising pass" if rising is None or rising else "Setting pass"

    summary = (
        f"Constant-El scan: {field.width:.2f} x {field.height:.2f} deg field "
        f"at RA={field.ra_center:.3f}, Dec={field.dec_center:.3f}\n"
        f"  Elevation: {elevation:.2f} deg, "
        f"Az range: [{az_min:.2f}, {az_max:.2f}] deg "
        f"(throw: {az_throw:.2f} deg)\n"
        f"  Velocity: {velocity:.3f} deg/s, Acceleration: {az_accel:.3f} deg/s^2\n"
        f"  {pass_label}: "
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
