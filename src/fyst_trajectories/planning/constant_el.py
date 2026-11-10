"""Constant-elevation scan planners (public via :mod:`fyst_trajectories.planning`)."""

import warnings
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

from .._validation import (
    AZ_PADDING_MAX_DEG,
    _require_finite,
    _require_non_negative,
    _require_positive,
)
from ..coordinates import Coordinates
from ..exceptions import ElevationBoundsError, PointingWarning
from ..patterns.configs import ConstantElScanConfig
from ..site import AtmosphericConditions, Site
from ._ce_geometry import (
    _CE_LONG_BLOCK_HOURS,
    _boresight_az_offset,
    _check_window_field_drift,
    _compute_ce_az_range,
    _compute_ce_duration,
    _compute_ce_duration_from_lsa,
    _place_ce_sweep,
    _window_side,
)
from ._helpers import _build_altaz_trajectory, _coerce_start_time
from ._sun_safety import (
    _SUN_SAFETY_ARC_N_SAMPLES,
    _check_arc_sun_safety,
    _check_field_sun_safety,
    _check_trajectory_sun_safety,
    _swept_arc_samples,
)
from ._types import ConstantElComputedParams, FieldRegion, ScanBlock, validate_computed_params

if TYPE_CHECKING:
    from ..offsets import InstrumentOffset
    from ..sun_protocols import SunSafePredicate


def plan_constant_el_scan(
    field: FieldRegion,
    *,
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
) -> ScanBlock[ConstantElComputedParams]:
    """Plan a constant-elevation scan that covers a FieldRegion.

    Auto-computes the azimuth range and observation duration from the field
    geometry (the duration from ``lsa_window`` instead, when given). The
    azimuth range is the projection of the whole field over the whole pass,
    which is wider than an elevation-band drift corridor; which of the two a
    constant-elevation survey scan should use awaits FYST-team confirmation.
    :func:`plan_constant_el_window_scan` sweeps a given azimuth window
    instead, from a literal start.

    The function:

    1. Finds when the RA edges of the (optionally rotated) field cross
       the target elevation (determines start/end time and duration);
       with ``lsa_window`` the window sets them instead. The edges are the
       field's own extreme corners at its central Dec, so a field across
       RA 0/360, or wider than 180 deg of RA, is timed by its own extent.
    2. Computes the azimuth range that covers the entire field at that
       elevation at the start, midpoint and end of the observation.
    3. Computes n_scans from the duration, the single-leg sweep time and
       the turnaround time between legs, and places the range on a 360 deg
       branch. The sweep reaches the range plus the overshoot
       ``5 v**2 / (8 a)`` past each edge a turnaround reaches (the first leg
       runs up from the low edge, so two legs turn only at the high edge),
       and the boresight, which the mount follows, reaches the same moved by
       the constant azimuth a ``detector_offset`` gives it. The range moves
       by a whole turn when the branch a turn away holds the trajectory's
       samples, the commanded ones and the boresight's, and its own does
       not, or holds the boresight's whole path as well where its own holds
       only the samples; a sample can stop short of a turnaround's apex.
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
        (mount frame, not on-sky). Default is 1.0. Must be positive.
    timestep : float, optional
        Time between trajectory points in seconds. Default is 0.1.
    detector_offset : InstrumentOffset or None, optional
        If provided, adjust the trajectory so this detector, rather than the
        boresight, sweeps the azimuth range.
    az_padding : float, optional
        Extra azimuth padding in degrees on each side of the computed
        range. Default is 2.0. Must lie in ``[0, 360]``.
    atmosphere : AtmosphericConditions or None, optional
        Refraction model for the elevation-crossing search and the azimuth
        range; the commanded ``elevation`` itself is never refracted.
        ``None`` (default) solves the pass on vacuum elevations, which is
        what a scan sent to the telescope must use, since refraction is
        applied downstream at execution time. Pass one only for planning
        or simulation output.
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
        (e.g. ``(310.0, 10.0)`` is a 60 deg / (15 deg/h) = 4 hour scan crossing the
        LSA = 0/360 boundary). Both endpoints must lie in ``[0, 360)``
        and must not be equal. The window fixes both the timing and the
        azimuth range, so ``rising`` has nothing left to choose and
        passing it alongside is rejected. ``max_search_hours`` and
        ``step_seconds`` still bound the search horizon. Default is
        ``None``, which times the scan from the elevation crossings.
    sun_safe : SunSafePredicate or None, optional
        Sun-safety predicate implementing the
        :class:`~fyst_trajectories.sun_protocols.SunSafePredicate` contract,
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
        If ``elevation`` is not a finite real number, if ``velocity`` or
        ``az_accel`` is not a positive one (a ``bool``, an array of one or
        more dimensions, text and ``None`` are not real numbers here), if
        ``timestep``, ``max_search_hours`` or ``step_seconds`` is not
        positive, if ``az_padding`` is not a real number in ``[0, 360]``, if
        ``rising`` and ``lsa_window`` are supplied together, if
        ``lsa_window`` is supplied with equal endpoints or values outside
        ``[0, 360)``, or if ``start_time`` holds more than one instant.
    TypeError
        If ``start_time`` is neither an ISO string nor an astropy ``Time``.
    PointingError
        If the elevation crossings cannot be found within the search window
        or the two field edges cross on different passes, if the field lies
        within 0.57 deg of a celestial pole or a corner of it lies past one,
        if, without ``lsa_window``, it spans a whole turn of RA or more at
        its central Dec, or if ``lsa_window`` is supplied and LST never
        increases through ``min_lsa`` within ``max_search_hours`` of
        ``start_time``.
    AzimuthBoundsError
        If the built trajectory exceeds the telescope azimuth limits. The
        range is first moved to a branch where the trajectory's samples fit,
        when one does (step 3).
    OffsetInversionError
        If ``detector_offset`` is given and the boresight that places it
        cannot be solved, as for a module near the zenith. A
        :class:`~fyst_trajectories.exceptions.PointingError` subclass.
    ElevationBoundsError
        If the elevation exceeds telescope limits.

    Warns
    -----
    PointingWarning
        If the field center is inside the Sun exclusion zone at
        ``start_time`` (and, with ``lsa_window``, at the resolved start),
        and separately if the swept pass enters it (see Notes); neither
        refuses the plan. Also when the solved pass is longer than half of
        ``max_search_hours`` or the ``lsa_window`` span is longer than 6 h,
        when :class:`~fyst_trajectories.patterns.ConstantElScanConfig`
        finds the speed or acceleration unusually large or the turnaround
        overshoot wider than the throw, and when azimuth motion at a high
        elevation cuts the on-sky azimuth speed.
    VelocityLimitWarning, AccelerationLimitWarning
        The built trajectory exceeds an axis velocity or acceleration limit
        (:func:`~fyst_trajectories.trajectory_utils.validate_trajectory_dynamics`).

    Notes
    -----
    Sun safety is screened in two stages. The field-center pre-flight is
    a cheap instant check that runs before the search, at ``start_time``,
    and again at the resolved ``obs_start`` when ``lsa_window`` delays the
    observation (the LSA branch can push it hours later, during which the
    Sun moves ~15 deg/hour). Once the pass is resolved, the planner then
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
    >>> print(block.summary)
    Constant-El scan: 5.00 x 6.70 deg field at RA=53.117, Dec=-27.808
      Elevation: 50.00 deg, Az range: [97.51, 114.17] deg (throw: 16.66 deg)
      Velocity: 0.500 deg/s, Acceleration: 1.000 deg/s^2
      Rising pass: 2026-03-15 17:20:43 to 2026-03-15 17:48:09
      Scans: 48, Duration: 1646.5s (27.4min), Trajectory points: 16466

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
    # Checked here, not only by the config built at the end: the crossing
    # search, the leg quantiser, the branch placement and the Sun sweep
    # compute with these first.
    elevation = _require_finite(elevation, "elevation")
    velocity = _require_positive(velocity, "velocity")
    az_accel = _require_positive(az_accel, "az_accel")
    timestep = _require_positive(timestep, "timestep")
    # A padding wider than a full turn is meaningless; one below it but past
    # the azimuth range reaches the bounds check as an AzimuthBoundsError.
    az_padding = _require_non_negative(az_padding, "az_padding", maximum=AZ_PADDING_MAX_DEG)
    if lsa_window is not None and rising is not None:
        raise ValueError(
            "rising is not accepted with lsa_window: the sidereal window fixes both "
            "the timing and the azimuth range, so there is no crossing half to choose. "
            "Drop rising, or drop lsa_window to plan an elevation crossing."
        )

    start_time = _coerce_start_time(start_time)

    # Refused before the Sun screen and the search, which an elevation past
    # 90 deg would otherwise fail first.
    el_limits = site.telescope_limits.elevation
    if not el_limits.is_in_range(elevation):
        raise ElevationBoundsError(
            actual_min=elevation,
            actual_max=elevation,
            limit_min=el_limits.min,
            limit_max=el_limits.max,
        )

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

    # The elevation-crossing window is quantised into whole azimuth legs so
    # the trajectory length matches n_scans exactly, rather than using the raw
    # crossing duration which may differ. The range is placed for the sweep
    # of those legs, and that sweep is built.
    az_min, az_max, n_scans, actual_duration = _compute_ce_az_range(
        field,
        angle,
        coords_obj,
        obs_start,
        obs_end,
        az_padding,
        velocity=velocity,
        az_accel=az_accel,
        duration=duration,
        timestep=timestep,
        boresight_az_offset=_boresight_az_offset(elevation, detector_offset, site),
    )
    az_throw = az_max - az_min

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
        "start_time_iso": obs_start.utc.iso,
        "end_time_iso": obs_end.utc.iso,
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
        f"{obs_start.utc.iso[:19]} to {obs_end.utc.iso[:19]}\n"
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


def plan_constant_el_window_scan(
    field: FieldRegion | None = None,
    *,
    az_window: tuple[float, float] | list[float],
    elevation: float,
    velocity: float,
    duration: float,
    site: Site,
    start_time: str | Time,
    az_accel: float = 1.0,
    timestep: float = 0.1,
    detector_offset: "InstrumentOffset | None" = None,
    az_padding: float = 0.0,
    atmosphere: AtmosphericConditions | None = None,
    sun_safe: "SunSafePredicate | None" = None,
) -> ScanBlock[ConstantElComputedParams]:
    """Plan a constant-elevation scan over a fixed azimuth window.

    The telescope sweeps ``az_window`` back and forth at one ``elevation``
    for ``duration`` seconds while the sky drifts through it, from
    ``start_time`` exactly. The inputs have the shape of a constant-elevation
    azimuth-range row of a fystplan observing unit (``az_min``, ``az_max``,
    ``elevation`` and a duration, given in hours there and in seconds here).
    :func:`plan_constant_el_scan` instead derives its azimuth range and its
    timing from a field's elevation crossing.

    The function:

    1. Widens ``az_window`` by ``az_padding`` on each side.
    2. Rounds ``duration`` to a whole number of azimuth legs with the
       turnarounds between them, as :func:`plan_constant_el_scan` rounds its
       pass, so the two quantise identically at equal throw, speed and
       acceleration.
    3. Places the window on a 360 deg branch by the rule
       :func:`plan_constant_el_scan` applies to its range (step 3 there).
    4. Builds the trajectory from ``start_time``, with no search: a later
       start sweeps the same window later.
    5. Screens the built trajectory against the Sun and, when ``field`` is
       given, checks that the field drifts through during the scan (see
       Notes). Both only warn.

    Parameters
    ----------
    field : FieldRegion or None, optional
        The field the window is meant to see drift through. It never shapes
        the commanded trajectory and is read only by the drift checks (see
        Notes). Default ``None`` skips them, for a target that no single
        RA/Dec box describes (a band around the whole RA circle, say).
    az_window : tuple of (az_lo, az_hi)
        Azimuth window in degrees, ``az_hi > az_lo``. A list is accepted, so
        a value that has round-tripped through JSON or ECSV can be passed
        back. The pair is the sweep itself, not a modular range: a window
        across azimuth 0/360 is written unwrapped, as ``(355.0, 370.0)`` or
        ``(-5.0, 10.0)``.
    elevation : float
        Fixed elevation for the scan in degrees.
    velocity : float
        Azimuth scan speed in azimuth coordinate degrees/second (not
        on-sky). The on-sky speed is ``velocity * cos(elevation)``. Must be
        positive.
    duration : float
        Requested sweep length in seconds. It is rounded to whole legs, and
        ``ScanBlock.duration`` is the rounded value. Must be positive.
    site : Site
        Telescope site configuration.
    start_time : str or Time
        Time of the trajectory's first sample, taken literally. A start
        after the field has crossed the window is not refused.
    az_accel : float, optional
        Azimuth acceleration in azimuth coordinate degrees/second^2 (mount
        frame, not on-sky). Default is 1.0. Must be positive.
    timestep : float, optional
        Time between trajectory points in seconds. Default is 0.1.
    detector_offset : InstrumentOffset or None, optional
        If provided, adjust the trajectory so this detector, rather than the
        boresight, sweeps the window.
    az_padding : float, optional
        Extra azimuth in degrees on each side of ``az_window``. Default is
        0.0, which commands the window as given. Must lie in ``[0, 360]``.
    atmosphere : AtmosphericConditions or None, optional
        Refraction model under which the drift checks time the field's
        crossings of ``elevation``; the commanded ``elevation`` itself is never
        refracted. ``None`` (default) times them in vacuum, the frame a scan
        sent to the telescope is commanded in, since refraction is applied
        downstream at execution time.
    sun_safe : SunSafePredicate or None, optional
        Sun-safety predicate implementing the
        :class:`~fyst_trajectories.sun_protocols.SunSafePredicate` contract,
        used by the screen of the built trajectory. ``None`` (default) keeps
        the built-in scalar exclusion-radius check, which screens every
        sample; an injected predicate is consulted instead, on about 600
        evenly spaced samples, both ends included, so the directional
        sun-avoidance model (see
        :func:`~fyst_trajectories.sun_models.make_sun_safe`) is honored
        end-to-end. Warn-only.

    Returns
    -------
    ScanBlock
        Planned observation containing trajectory, config, and computed
        parameters: ``az_start`` and ``az_stop`` (the padded window as
        commanded, on its chosen branch), ``az_throw``, ``n_scans``,
        ``start_time_iso`` (the literal start), ``end_time_iso``
        (``start_time`` plus the requested ``duration``) and ``duration``
        (the rounded trajectory length).

    Raises
    ------
    ValueError
        If ``az_window`` is not a pair of finite real numbers with
        ``az_hi > az_lo``, if the window widened by ``az_padding`` on each
        side is wider than 360 deg, if ``elevation`` is not a finite real
        number, if ``velocity``, ``duration`` or ``az_accel`` is not a
        positive one (a ``bool``, an array of one or more dimensions, text
        and ``None`` are not real numbers here), if ``timestep`` is not
        positive, if ``az_padding`` is not a real number in ``[0, 360]``, if
        ``start_time`` holds more than one instant, or if the rounded sweep
        lasts less than two samples at ``timestep``.
    TypeError
        If ``start_time`` is neither an ISO string nor an astropy ``Time``.
    AzimuthBoundsError
        If the built trajectory exceeds the telescope azimuth limits. The
        window is first moved to a branch where the trajectory's samples
        fit, when one does (step 3).
    OffsetInversionError
        If ``detector_offset`` is given and the boresight that places it
        cannot be solved, as for a module near the zenith. A
        :class:`~fyst_trajectories.exceptions.PointingError` subclass.
    ElevationBoundsError
        If the elevation exceeds telescope limits.

    Warns
    -----
    PointingWarning
        If the built trajectory enters the Sun exclusion zone (naming the
        closest approach, or with an injected ``sun_safe`` the earliest
        unsafe sample), if ``duration`` is longer than 6 h, and if a
        ``field`` is given and the drift checks find that it does not cross
        ``elevation`` during the scan (see Notes); none refuses the plan.
        Also when :class:`~fyst_trajectories.patterns.ConstantElScanConfig`
        finds the speed or acceleration unusually large or the turnaround
        overshoot wider than the throw, and when azimuth motion at a high
        elevation cuts the on-sky azimuth speed.
    VelocityLimitWarning, AccelerationLimitWarning
        The built trajectory exceeds an axis velocity or acceleration limit
        (:func:`~fyst_trajectories.trajectory_utils.validate_trajectory_dynamics`).

    Notes
    -----
    The window is fixed in the horizon frame while the Sun moves about
    15 deg per hour, so a block that starts clear of the Sun can end inside
    the exclusion zone. The screen therefore runs on the built trajectory,
    turnaround overshoot included, over the whole block.

    The drift checks read the field's two RA edges, at its central Dec,
    once, at ``start_time``, with no crossing search. The window's side
    comes from its midpoint: east of the meridian (azimuth 0 to 180 deg) the
    field rises through ``elevation``, west of it the field sets. Each edge
    crosses ``elevation`` on that side once per sidereal day, at the hour
    angle its declination fixes, so the field's pass runs from the leading
    edge's crossing to the trailing edge's. A scan that starts inside a
    pass, or lasts until the next one begins, is silent; one that lies
    wholly between two passes sweeps sky the field has left or has not yet
    reached, and a warning names the nearer pass and when its edge crosses.
    A field whose central Dec never crosses ``elevation`` draws its own
    warning. The checks are skipped when ``field`` is ``None``, when the
    commanded window reaches the meridian (contains a multiple of 180 deg,
    so it has no single side), and for the edges of a field that spans the
    whole RA circle at its central Dec. The checks test timing only, not
    whether the field crosses ``elevation`` inside the window's azimuths,
    and they judge the culminations from the site latitude and the field's
    catalogue Dec, so they are heuristic.

    Examples
    --------
    >>> from astropy.time import Time
    >>> from fyst_trajectories import get_fyst_site
    >>> from fyst_trajectories.planning import FieldRegion, plan_constant_el_window_scan
    >>> site = get_fyst_site()
    >>> field = FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7)
    >>> block = plan_constant_el_window_scan(
    ...     field,
    ...     az_window=(101.0, 111.6),
    ...     elevation=50.0,
    ...     velocity=0.5,
    ...     duration=1680.0,
    ...     site=site,
    ...     start_time=Time("2026-03-15T17:20:00", scale="utc"),
    ... )
    >>> print(block.summary)
    Constant-El window scan: 5.00 x 6.70 deg field at RA=53.117, Dec=-27.808
      Elevation: 50.00 deg, Az window: [101.00, 111.60] deg (throw: 10.60 deg)
      Velocity: 0.500 deg/s, Acceleration: 1.000 deg/s^2
      Requested: 2026-03-15 17:20:00 to 2026-03-15 17:48:00
      Scans: 76, Duration: 1686.2s (28.1min), Trajectory points: 16863
    """
    # Every argument is checked before anything is computed with it, so a
    # malformed request is a ValueError naming the argument.
    try:
        az_lo, az_hi = az_window
    except (TypeError, ValueError):
        raise ValueError(
            f"az_window must be a pair (az_lo, az_hi) in degrees, got {az_window!r}"
        ) from None
    az_lo = _require_finite(az_lo, "az_window[0]")
    az_hi = _require_finite(az_hi, "az_window[1]")
    if az_hi <= az_lo:
        # Read as a modular range, an inverted pair would command a sweep of
        # nearly a full turn, which fits FYST's 540 deg azimuth range.
        raise ValueError(
            f"az_window must have az_hi > az_lo, got ({az_lo}, {az_hi}). The pair is the "
            "commanded sweep, not a modular range: write a window across azimuth 0/360 "
            "unwrapped, as (355, 370) or (-5, 10)."
        )
    elevation = _require_finite(elevation, "elevation")
    velocity = _require_positive(velocity, "velocity")
    duration = _require_positive(duration, "duration")
    az_accel = _require_positive(az_accel, "az_accel")
    timestep = _require_positive(timestep, "timestep")
    az_padding = _require_non_negative(az_padding, "az_padding", maximum=AZ_PADDING_MAX_DEG)
    sweep_width = az_hi - az_lo + 2.0 * az_padding
    if sweep_width > 360.0:
        raise ValueError(
            f"az_window ({az_lo}, {az_hi}) widened by az_padding={az_padding} on each side "
            f"spans {sweep_width} deg, more than a full turn."
        )

    start_time = _coerce_start_time(start_time)

    if duration > _CE_LONG_BLOCK_HOURS * 3600.0:
        warnings.warn(
            f"Window-scan duration {duration / 3600.0:.1f}h is unusually long for a single "
            f"CE block. Check duration={duration} s.",
            PointingWarning,
            stacklevel=2,
        )

    az_min, az_max, n_scans, actual_duration = _place_ce_sweep(
        az_lo - az_padding,
        az_hi + az_padding,
        velocity=velocity,
        az_accel=az_accel,
        duration=duration,
        timestep=timestep,
        boresight_az_offset=_boresight_az_offset(elevation, detector_offset, site),
        limits=site.telescope_limits.azimuth,
    )
    az_throw = az_max - az_min

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
        start_time=start_time,
        atmosphere=atmosphere,
        detector_offset=detector_offset,
    )

    side = _window_side(az_min, az_max)
    if field is not None and side is not None:
        _check_window_field_drift(
            field,
            side=side,
            elevation=elevation,
            coords_obj=Coordinates(site, atmosphere=atmosphere),
            start=start_time,
            duration=duration,
        )

    _check_trajectory_sun_safety(
        site=site,
        trajectory=trajectory,
        scan_label=(
            f"constant-elevation window scan over az [{az_min:.2f}, {az_max:.2f}] "
            f"at el={elevation:.2f}"
        ),
        sun_safe=sun_safe,
    )

    end_time = start_time + TimeDelta(duration * u.s)
    computed_params: ConstantElComputedParams = {
        "az_start": az_min,
        "az_stop": az_max,
        "az_throw": az_throw,
        "n_scans": n_scans,
        "start_time_iso": start_time.utc.iso,
        "end_time_iso": end_time.utc.iso,
        "duration": actual_duration,
    }
    validate_computed_params(computed_params, "constant_el")

    if field is None:
        field_line = "no field region"
    else:
        field_line = (
            f"{field.width:.2f} x {field.height:.2f} deg field "
            f"at RA={field.ra_center:.3f}, Dec={field.dec_center:.3f}"
        )
    summary = (
        f"Constant-El window scan: {field_line}\n"
        f"  Elevation: {elevation:.2f} deg, "
        f"Az window: [{az_min:.2f}, {az_max:.2f}] deg "
        f"(throw: {az_throw:.2f} deg)\n"
        f"  Velocity: {velocity:.3f} deg/s, Acceleration: {az_accel:.3f} deg/s^2\n"
        f"  Requested: {start_time.utc.iso[:19]} to {end_time.utc.iso[:19]}\n"
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
