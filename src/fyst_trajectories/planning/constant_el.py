"""Constant-elevation scan planner (public via :mod:`fyst_trajectories.planning`)."""

import warnings
from typing import TYPE_CHECKING, overload

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
    _DEFAULT_MAX_SEARCH_HOURS,
    _DEFAULT_STEP_SECONDS,
    _boresight_az_offset,
    _check_window_field_drift,
    _compute_ce_az_range,
    _compute_ce_duration,
    _compute_ce_duration_from_lsa,
    _lsa_window_rising_side,
    _place_ce_sweep,
    _window_side,
)
from ._helpers import _build_altaz_trajectory, _coerce_start_time
from ._sun_safety import _check_field_sun_safety, _check_trajectory_sun_safety
from ._types import ConstantElComputedParams, FieldRegion, ScanBlock, validate_computed_params

if TYPE_CHECKING:
    from ..offsets import InstrumentOffset
    from ..sun_protocols import SunSafePredicate


@overload
def plan_constant_el_scan(
    field: FieldRegion,
    *,
    elevation: float,
    velocity: float,
    site: Site,
    start_time: str | Time,
    az_window: None = None,
    duration: None = None,
    rising: bool | None = None,
    angle: float | None = None,
    lsa_window: tuple[float, float] | list[float] | None = None,
    max_search_hours: float | None = None,
    step_seconds: float | None = None,
    az_accel: float = 1.0,
    timestep: float = 0.1,
    detector_offset: "InstrumentOffset | None" = None,
    az_padding: float = 0.0,
    atmosphere: AtmosphericConditions | None = None,
    sun_safe: "SunSafePredicate | None" = None,
) -> ScanBlock[ConstantElComputedParams]: ...


@overload
def plan_constant_el_scan(
    field: FieldRegion | None = None,
    *,
    elevation: float,
    velocity: float,
    site: Site,
    start_time: str | Time,
    az_window: tuple[float, float] | list[float],
    duration: float,
    rising: None = None,
    angle: None = None,
    lsa_window: None = None,
    max_search_hours: None = None,
    step_seconds: None = None,
    az_accel: float = 1.0,
    timestep: float = 0.1,
    detector_offset: "InstrumentOffset | None" = None,
    az_padding: float = 0.0,
    atmosphere: AtmosphericConditions | None = None,
    sun_safe: "SunSafePredicate | None" = None,
) -> ScanBlock[ConstantElComputedParams]: ...


def plan_constant_el_scan(
    field: FieldRegion | None = None,
    *,
    elevation: float,
    velocity: float,
    site: Site,
    start_time: str | Time,
    az_window: tuple[float, float] | list[float] | None = None,
    duration: float | None = None,
    rising: bool | None = None,
    angle: float | None = None,
    lsa_window: tuple[float, float] | list[float] | None = None,
    max_search_hours: float | None = None,
    step_seconds: float | None = None,
    az_accel: float = 1.0,
    timestep: float = 0.1,
    detector_offset: "InstrumentOffset | None" = None,
    az_padding: float = 0.0,
    atmosphere: AtmosphericConditions | None = None,
    sun_safe: "SunSafePredicate | None" = None,
) -> ScanBlock[ConstantElComputedParams]:
    """Plan a constant-elevation scan, across a field or over an azimuth window.

    The telescope holds one ``elevation`` and sweeps back and forth in
    azimuth while the sky drifts through. The planner takes one of two
    forms:

    - The field form, given a ``field``: it works out the azimuth range and
      the timing from the field's crossing of ``elevation``, or the timing
      from ``lsa_window``.
    - The window form, given ``az_window`` with a ``duration``: it sweeps
      the window it is given for that long, from ``start_time``, with no
      search. The inputs have the shape of a constant-elevation
      azimuth-range row of a fystplan observing unit (``az_min``,
      ``az_max``, ``elevation`` and a duration, given in hours there and in
      seconds here).

    ``start_time`` means a different instant in each. In the field form it
    is the anchor of a forward search: the scan starts at the field's
    crossing, or at the opening of ``lsa_window``, at or after it. In the
    window form it is the time of the trajectory's first sample, taken
    literally.

    ``rising``, ``angle``, ``lsa_window``, ``max_search_hours`` and
    ``step_seconds`` belong to the field form: each defaults to ``None``,
    meaning not given, and any other value is refused beside
    ``az_window``. ``duration`` belongs to the window form and is refused
    without it. ``elevation``, ``velocity``, ``site``, ``start_time``,
    ``az_accel``, ``timestep``, ``detector_offset``, ``az_padding``,
    ``atmosphere`` and ``sun_safe`` are common to both.

    **The field form.** The azimuth range is where the field's border meets
    the scan elevation: the azimuths at which its lowest and highest
    declinations cross ``elevation`` on the pass's side of the meridian,
    the quantity a constant-elevation ``azimuth_range`` row of a fystplan
    observing unit tabulates. So it depends on the field's declination
    extent, not on its RA extent or on the length of the pass. The planner:

    1. Finds when the RA edges of the (optionally rotated) field cross
       the target elevation (determines start/end time and duration);
       with ``lsa_window`` the window sets them instead. The edges are the
       field's extreme RA (for a rotated field, that of the box's corners)
       placed at its central Dec, so a field across RA 0/360, or wider
       than 180 deg of RA, is timed by its own extent. Parts of the field
       at other declinations cross the elevation earlier or later than
       those edges, rotated or not, so the pass is not the interval in
       which the box is at the elevation: a corner that crosses before the
       pass opens or after it closes is not scanned.
    2. Computes the azimuth range on the pass's side of the meridian: east
       (azimuth 0 to 180 deg) for the rising crossing, west (180 to 360 deg)
       for the setting one, and with ``lsa_window`` the side the field
       centre is on at the middle of the window. The range runs between
       the extreme azimuths at which points of the box's border cross
       ``elevation``; a point of apparent declination ``d`` crosses
       elevation ``e`` at site latitude ``lat`` at the azimuth ``A`` with
       ``cos A = (sin d - sin e sin lat) / (cos e cos lat)``, ``A`` east of
       the meridian and ``360 - A`` west of it. The crossing azimuth moves
       monotonically with declination, so the range holds the azimuths at
       which the field is at ``elevation`` over its whole crossing, inside
       the timed pass or not. The border is read at evenly spaced points
       along each edge, not only at the corners, because an edge of
       constant catalogue Dec is not one of constant apparent Dec
       (precession, nutation, aberration), and on an edge many degrees long
       in RA the apparent Dec is extreme between its corners.
       ``az_padding`` widens the range on each side.
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

    **The window form.** The planner:

    1. Widens ``az_window`` by ``az_padding`` on each side.
    2. Rounds ``duration`` to a whole number of azimuth legs with the
       turnarounds between them, as the field form rounds its pass, so the
       two quantise identically at equal throw, speed and acceleration.
    3. Places the window on a 360 deg branch by the rule the field form
       applies to its range (step 3 above).
    4. Builds the trajectory from ``start_time``, with no search: a later
       start sweeps the same window later.
    5. Screens the built trajectory against the Sun and, when ``field`` is
       given, checks that the field drifts through during the scan (see
       Notes). Both only warn.

    Parameters
    ----------
    field : FieldRegion or None, optional
        Rectangular field specification (center RA/Dec, width, height).
        The field form plans across it and requires it. In the window form
        it is optional, never shapes the commanded trajectory and is read
        only by the drift checks (see Notes); ``None`` (default) skips
        them, for a target that no single RA/Dec box describes (a band
        around the whole RA circle, say).
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
        In the field form, the approximate start time for the search
        window: the function searches up to ``max_search_hours`` forward
        from this time to find when the field edges cross the target
        elevation. In the window form, the time of the trajectory's first
        sample, taken literally; a start after the field has crossed the
        window is not refused.
    az_window : tuple of (az_lo, az_hi) or None, optional
        The window form's azimuth window in degrees, ``az_hi > az_lo``. A
        list is accepted, so a value that has round-tripped through JSON or
        ECSV can be passed back. The pair is the sweep itself, not a
        modular range: a window across azimuth 0/360 is written unwrapped,
        as ``(355.0, 370.0)`` or ``(-5.0, 10.0)``. Default ``None`` selects
        the field form.
    duration : float or None, optional
        The window form's requested sweep length in seconds, required with
        ``az_window`` and refused without it. It is rounded to whole legs,
        and ``ScanBlock.duration`` is the rounded value. Must be positive.
    rising : bool or None, optional
        Field form only. Which elevation crossing to observe: ``True`` the
        rising one, ``False`` the setting one. Default ``None`` selects the
        rising crossing. Not accepted together with ``lsa_window``, which
        derives the timing from sidereal angle instead and has no crossing
        to choose.
    angle : float or None, optional
        Field form only. Rotation angle of the field region in degrees.
        Default ``None``, which is no rotation.
    lsa_window : tuple of (min_lsa, max_lsa), optional
        Field form only. Local Sidereal Angle window in degrees. When
        supplied, the scan ``start_time`` and ``duration`` are derived from
        this LSA window rather than from RA-edge elevation crossings: the
        planner finds the first time at or after ``start_time`` at which
        Local Sidereal Time (in degrees) increases through ``min_lsa``, and
        the scan spans ``(max_lsa - min_lsa) mod 360 / 15`` hours of
        UTC, about 0.3 percent longer than the sidereal window it names.
        ``ScanBlock.duration`` is that span quantised to whole azimuth
        legs, so it is a little shorter again.
        Wrap-around windows where ``max_lsa < min_lsa`` are supported
        (e.g. ``(310.0, 10.0)`` is a 60 deg / (15 deg/h) = 4 hour scan crossing the
        LSA = 0/360 boundary). Both endpoints must lie in ``[0, 360)``
        and must not be equal. The window fixes the timing, and the azimuth
        range is taken on the side of the meridian the field centre is on
        at the middle of the window (east gives the rising range, west the
        setting one), so the window fixes both and ``rising`` has nothing
        left to choose: passing it alongside is rejected.
        ``max_search_hours`` and ``step_seconds`` still set the search's
        horizon and step. Default is ``None``, which times the scan from the
        elevation crossings.
    max_search_hours : float or None, optional
        Field form only. Maximum time to search forward in hours for
        elevation crossings. Default ``None`` searches 12.0 hours.
    step_seconds : float or None, optional
        Field form only. Time step in seconds for the elevation crossing
        search. Default ``None`` steps 30.0 seconds.
    az_accel : float, optional
        Azimuth acceleration in azimuth coordinate degrees/second^2
        (mount frame, not on-sky). Default is 1.0. Must be positive.
    timestep : float, optional
        Time between trajectory points in seconds. Default is 0.1.
    detector_offset : InstrumentOffset or None, optional
        If provided, adjust the trajectory so this detector, rather than the
        boresight, sweeps the azimuth range or the window.
    az_padding : float, optional
        Extra azimuth in degrees on each side of the computed range or of
        ``az_window``. Default is 0.0, which sweeps the range as computed or
        commands the window as given. Must lie in ``[0, 360]``.
    atmosphere : AtmosphericConditions or None, optional
        Refraction model; the commanded ``elevation`` itself is never
        refracted. In the field form it applies to the elevation-crossing
        search and the azimuth range: with one, the field is at
        ``elevation`` when its refracted elevation is, so the pass is timed
        by the refracted positions and the range's border points cross at
        the vacuum elevation that appears at ``elevation``. In the window
        form the drift checks time the field's crossings of ``elevation``
        under it.
        ``None`` (default) works in vacuum, which is what a scan sent to the
        telescope must use, since refraction is applied downstream at
        execution time. Pass one only for planning or simulation output.
    sun_safe : SunSafePredicate or None, optional
        Sun-safety predicate implementing the
        :class:`~fyst_trajectories.sun_protocols.SunSafePredicate` contract.
        ``None`` (default) keeps the built-in scalar exclusion-radius check;
        an injected predicate is consulted instead, so the directional
        sun-avoidance model (see
        :func:`~fyst_trajectories.sun_models.make_sun_safe`) is honored
        end-to-end. It is used by the screen of the built trajectory in both
        forms, where the built-in check screens every sample and an injected
        predicate is consulted on about 600 evenly spaced samples, both ends
        included, and in the field form also by the field-center
        pre-flight(s) described in the Notes. Warn-only.

    Returns
    -------
    ScanBlock
        Planned observation containing trajectory, config, and computed
        parameters (``az_start``, ``az_stop``, ``az_throw``, ``n_scans``,
        ``start_time_iso``, ``end_time_iso``, ``duration``). In the field
        form the times are the solved pass's opening and close. In the
        window form ``az_start`` and ``az_stop`` are the padded window as
        commanded, on its chosen branch, ``start_time_iso`` is the literal
        start and ``end_time_iso`` is ``start_time`` plus the requested
        ``duration``. In both, ``duration`` is the trajectory's length,
        rounded to whole legs.

    Raises
    ------
    ValueError
        If ``az_window`` is given together with any of ``rising``,
        ``angle``, ``lsa_window``, ``max_search_hours`` or ``step_seconds``
        that is not ``None``, or without ``duration``; if ``duration`` is
        given without
        ``az_window``; or if neither ``field`` nor ``az_window`` is given.
        If ``elevation`` is not a finite real number, if ``velocity``,
        ``az_accel`` or (window form) ``duration`` is not a positive one (a
        ``bool``, an array of one or more dimensions, text and ``None`` are
        not real numbers here), if ``timestep``, ``max_search_hours`` or
        ``step_seconds`` is not positive, if ``az_padding`` is not a real
        number in ``[0, 360]``, or if ``start_time`` holds more than one
        instant. In the field form, if ``rising`` and ``lsa_window`` are
        supplied together, or if ``lsa_window`` is supplied and is not a
        pair of finite real numbers, has equal endpoints or has values
        outside ``[0, 360)``. In the window form, if
        ``az_window`` is not a pair of finite real numbers with
        ``az_hi > az_lo``, or if the window widened by ``az_padding`` on each
        side is wider than 360 deg. In both forms, if the rounded sweep lasts
        less than two samples at ``timestep``.
    TypeError
        If ``start_time`` is neither an ISO string nor an astropy ``Time``.
    PointingError
        Field form only. If the elevation crossings cannot be found within
        the search window or the two field edges cross on different passes,
        if the field lies within 0.57 deg of a celestial pole or a corner of
        it lies past one, if, without ``lsa_window``, it spans a whole turn
        of RA or more at its central Dec, or if ``lsa_window`` is supplied
        and LST never increases through ``min_lsa`` within
        ``max_search_hours`` of ``start_time`` or no part of the field's
        border ever reaches ``elevation``, so the field has no azimuth
        range. An anchor at a pass's opening or inside the pass is refused,
        so a pass cannot be planned again from its own ``start_time_iso``.
    AzimuthBoundsError
        If the built trajectory exceeds the telescope azimuth limits. The
        range or the window is first moved to a branch where the
        trajectory's samples fit, when one does (field form step 3).
    OffsetInversionError
        If ``detector_offset`` is given and the boresight that places it
        cannot be solved, as for a module near the zenith. A
        :class:`~fyst_trajectories.exceptions.PointingError` subclass.
    ElevationBoundsError
        If the elevation lies outside the telescope's elevation limits.

    Warns
    -----
    PointingWarning
        In both forms, if the built trajectory enters the Sun exclusion zone
        (naming the closest approach, or with an injected ``sun_safe`` the
        earliest unsafe sample; see Notes). In the field form, if the field
        center is inside the zone at ``start_time`` (and, with
        ``lsa_window``, at the resolved start), and when the solved pass is
        longer than half of ``max_search_hours`` or the ``lsa_window`` span
        is longer than 6 h. In the window form, if ``duration`` is longer
        than 6 h, and if a ``field`` is given and the drift checks find that
        it does not cross ``elevation`` during the scan (see Notes). None of
        these refuses the plan. In both forms, when
        :class:`~fyst_trajectories.patterns.ConstantElScanConfig` finds the
        speed or acceleration unusually large or the turnaround overshoot
        wider than the throw, and when azimuth motion at a high elevation
        cuts the on-sky azimuth speed.
    VelocityLimitWarning, AccelerationLimitWarning
        The built trajectory exceeds an axis velocity or acceleration limit
        (:func:`~fyst_trajectories.trajectory_utils.validate_trajectory_dynamics`).

    Notes
    -----
    In both forms the built trajectory is screened against the Sun over the
    whole block: every sample with the built-in scalar check, and about 600
    evenly spaced samples plus the last with an injected ``sun_safe``. The
    trajectory is the boresight's, turnaround overshoot included, so with a
    ``detector_offset`` the screen judges where the telescope points, not
    the detector's ``elevation``, and a multi-hour block that starts clear
    of the Sun and ends inside the exclusion zone is caught. In the field
    form a field-center pre-flight runs first, a cheap instant check before
    the search, at ``start_time``, and again at the resolved ``obs_start``
    when ``lsa_window`` delays the observation (the LSA branch can push it
    hours later, during which the Sun moves ~15 deg/hour). Both warn and
    never block.

    In the field form, a part of the field whose declination never reaches
    ``elevation`` (culminates short of it) is never at that elevation: its
    crossing is placed on the meridian, at azimuth 0 or 360 deg in the
    north and 180 deg in the south, and the range ends there. Near that
    limit the crossing azimuth changes steeply with declination, since
    ``A`` moves as ``1 / sin A`` per unit of ``cos A``, so for a field whose
    lowest or highest declination only grazes ``elevation`` that end of the
    range is sensitive to the field's declination extent and to
    ``elevation``.

    In the window form, the drift checks read the field's two RA edges, at
    its central Dec, once, at ``start_time``, with no crossing search. The
    window's side comes from its midpoint: east of the meridian (azimuth 0
    to 180 deg) the field rises through ``elevation``, west of it the field
    sets. Each edge crosses ``elevation`` on that side once per sidereal
    day, at the hour angle its declination fixes, so the field's pass runs
    from the leading edge's crossing to the trailing edge's. A scan that
    starts inside a pass, or lasts until the next one begins, is silent;
    one that lies wholly between two passes sweeps sky the field has left
    or has not yet reached, and a warning names the nearer pass and when
    its edge crosses. A field whose central Dec never crosses ``elevation``
    draws its own warning. The checks are skipped when ``field`` is
    ``None``, when the commanded window reaches the meridian (contains a
    multiple of 180 deg, so it has no single side), and for the edges of a
    field that spans the whole RA circle at its central Dec. The checks
    test timing only, not whether the field crosses ``elevation`` inside
    the window's azimuths, and they judge the culminations from the site
    latitude and the field's catalogue Dec, so they are heuristic.

    Examples
    --------
    The field form, timed by the elevation crossing (the default):

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
      Elevation: 50.00 deg, Az range: [100.45, 112.11] deg (throw: 11.66 deg)
      Velocity: 0.500 deg/s, Acceleration: 1.000 deg/s^2
      Rising pass: 2026-03-15 17:20:43 to 2026-03-15 17:48:09
      Scans: 68, Duration: 1653.0s (27.6min), Trajectory points: 16531

    The field form, timed by an LSA window (Deep56-style, a 4-hour
    wrap-around window):

    >>> deep56 = FieldRegion(ra_center=0.0, dec_center=-2.0, width=60.0, height=14.0)
    >>> block = plan_constant_el_scan(
    ...     field=deep56,
    ...     elevation=50.0,
    ...     velocity=0.5,
    ...     site=site,
    ...     start_time=Time("2026-09-15T00:00:00", scale="utc"),
    ...     lsa_window=(310.0, 10.0),
    ... )

    The window form, sweeping a given window from a literal start:

    >>> block = plan_constant_el_scan(
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
    # Each form first refuses the other form's arguments, before anything is
    # computed.
    window_form = az_window is not None
    if not window_form:
        if field is None:
            raise ValueError(
                "plan_constant_el_scan needs a field or an az_window: a field to plan its "
                "crossing of the elevation, or an az_window with a duration to sweep that "
                "window from start_time."
            )
        if duration is not None:
            raise ValueError(
                "duration is not accepted without az_window: it belongs to the window form, "
                "while the field form takes its length from the field's crossing or from "
                "lsa_window. Drop duration, or give az_window to sweep a fixed window."
            )

        if angle is None:
            angle = 0.0
        if max_search_hours is None:
            max_search_hours = _DEFAULT_MAX_SEARCH_HOURS
        if step_seconds is None:
            step_seconds = _DEFAULT_STEP_SECONDS
        # Checked here, not only by the config built at the end: the crossing
        # search, the leg quantiser and the branch placement compute with these
        # first.
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
        _check_field_sun_safety(
            field.ra_center, field.dec_center, start_time, site, sun_safe=sun_safe
        )

        coords_obj = Coordinates(site, atmosphere=atmosphere)

        if lsa_window is not None:
            obs_start, obs_end, search_duration = _compute_ce_duration_from_lsa(
                lsa_window,
                coords_obj,
                start_time,
                max_search_hours=max_search_hours,
                step_seconds=step_seconds,
            )
            _check_field_sun_safety(
                field.ra_center, field.dec_center, obs_start, site, sun_safe=sun_safe
            )
            rising_side = _lsa_window_rising_side(
                field.ra_center, field.dec_center, obs_start, obs_end, coords_obj
            )
        else:
            obs_start, obs_end, search_duration = _compute_ce_duration(
                field,
                angle,
                elevation,
                coords_obj,
                start_time,
                True if rising is None else rising,
                max_search_hours=max_search_hours,
                step_seconds=step_seconds,
            )
            rising_side = True if rising is None else rising

        # The elevation-crossing window is quantised into whole azimuth legs so
        # the trajectory length matches n_scans exactly, rather than using the raw
        # crossing duration which may differ. The range is placed for the sweep
        # of those legs, and that sweep is built.
        az_min, az_max, n_scans, actual_duration = _compute_ce_az_range(
            field,
            angle,
            elevation,
            rising_side,
            coords_obj,
            obs_start,
            az_padding,
            velocity=velocity,
            az_accel=az_accel,
            duration=search_duration,
            timestep=timestep,
            boresight_az_offset=_boresight_az_offset(elevation, detector_offset, site),
        )
        az_throw = az_max - az_min
    else:
        # In signature order, so of several the first in the signature is named.
        field_form_only = (
            ("rising", rising),
            ("angle", angle),
            ("lsa_window", lsa_window),
            ("max_search_hours", max_search_hours),
            ("step_seconds", step_seconds),
        )
        for name, value in field_form_only:
            if value is not None:
                raise ValueError(
                    f"{name} is not accepted with az_window: it belongs to the field form, "
                    "which works out the azimuth range and the timing from the field, while "
                    "az_window is swept as given from start_time. Drop "
                    f"{name}, or drop az_window and duration to plan the field's crossing."
                )

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
        # A missing duration is refused here, by name, in its turn.
        duration = _require_positive(duration, "duration")  # type: ignore[arg-type]
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
        obs_start = start_time
        obs_end = start_time + TimeDelta(duration * u.s)

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

    if window_form:
        assert duration is not None
        side = _window_side(az_min, az_max)
        if field is not None and side is not None:
            _check_window_field_drift(
                field,
                side=side,
                elevation=elevation,
                coords_obj=Coordinates(site, atmosphere=atmosphere),
                start=obs_start,
                duration=duration,
            )
        scan_label = (
            f"constant-elevation window scan over az [{az_min:.2f}, {az_max:.2f}] "
            f"at el={elevation:.2f}"
        )
    else:
        assert field is not None
        scan_label = (
            f"constant-elevation scan at RA={field.ra_center:.3f}, Dec={field.dec_center:.3f}"
        )

    # A block can run for hours while the Sun moves about 15 deg per hour, so
    # every sample of the built boresight trajectory is screened, not its opening.
    _check_trajectory_sun_safety(
        site=site, trajectory=trajectory, scan_label=scan_label, sun_safe=sun_safe
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

    if field is None:
        field_line = "no field region"
    else:
        field_line = (
            f"{field.width:.2f} x {field.height:.2f} deg field "
            f"at RA={field.ra_center:.3f}, Dec={field.dec_center:.3f}"
        )
    if window_form:
        heading = f"Constant-El window scan: {field_line}"
        range_label = "Az window"
        # The window form names the span it was asked for.
        pass_label = "Requested"
    else:
        heading = f"Constant-El scan: {field_line}"
        range_label = "Az range"
        # Name the pass by what actually chose it: the crossing half on the
        # elevation-crossing path, the sidereal window on the LSA path (where
        # no half is chosen and claiming one would be a false report).
        if lsa_window is not None:
            pass_label = "LSA-window pass"
        else:
            pass_label = "Rising pass" if rising is None or rising else "Setting pass"
    summary = (
        f"{heading}\n"
        f"  Elevation: {elevation:.2f} deg, "
        f"{range_label}: [{az_min:.2f}, {az_max:.2f}] deg "
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
