"""Geometry helpers used by the constant-elevation scan planners.

Public to the package-internal planners; not part of the
:mod:`fyst_trajectories.planning` public API.

Several of these functions have in-repo consumers outside this
subpackage, the offline scheduler and its patch model. The scheduler
gates its constant-elevation emission on the planner's own solves,
:func:`_compute_ce_duration` (the forward crossing search) or, for an
LSA window, :func:`_compute_ce_duration_from_lsa` with
:func:`_lsa_window_rising_side`, and reads the azimuths a pass sweeps
from :func:`_ce_field_az_range`, the same solves the planner runs at
reconstruction, so that what is scheduled is what can be rebuilt. The
scheduler and the patch model also apply the planner's own rules where
they time a window scan or check a patch (:func:`_quantize_ce_duration`,
:func:`_field_ra_edges`, :func:`_lsa_window_endpoints`,
:func:`_window_side`). That coupling is deliberate and the
functions stay private, because their return values are internals of
the planner rather than a contract: they answer "when does this field
cross this elevation next" and "at which azimuths", with the
assumptions the planner brings to those questions.
"""

import math
import warnings

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

from .._validation import _is_finite, _require_positive
from ..coordinates import Coordinates, _threshold_crossings
from ..exceptions import OffsetInversionError, PointingError, PointingWarning
from ..offsets import InstrumentOffset, _solve_boresight
from ..patterns.constant_el import _ce_az_reach
from ..patterns.turnarounds import swept_az_envelope, turnaround_duration_sec
from ..site import AxisLimits, Site
from ._types import FieldRegion

# A constant-elevation block longer than this draws an "unusually long"
# advisory where its length is given rather than solved (an LSA window, a
# window scan's requested duration): sustained constant-elevation pointing
# beyond about 6 h is unusual for a single block, whatever the search horizon.
_CE_LONG_BLOCK_HOURS = 6.0

# The field form's crossing search when ``max_search_hours`` / ``step_seconds``
# are not given; the offline scheduler's corridor gate searches the same way.
_DEFAULT_MAX_SEARCH_HOURS = 12.0
_DEFAULT_STEP_SECONDS = 30.0

# The hour angle of a fixed RA/Dec grows by one turn per sidereal day.
_SIDEREAL_RATE_DEG_S = 360.0 / 86164.0905

# A field whose RA span at its central Dec reaches a whole turn, to rounding,
# covers every RA there and so has no leading and trailing RA edges.
_WHOLE_TURN_RA_SPAN_DEG = 360.0 - 1e-9

# Points per edge at which the constant-elevation range reads a box's border,
# corners included. 180 points keep the samples of an edge spanning a whole
# turn of RA about 2 deg of RA apart, and a read at 20000 points per edge
# moves the range by less than 1e-4 deg.
_CE_BORDER_SAMPLES_PER_EDGE = 180


def _field_region_corners(
    ra_center: float,
    dec_center: float,
    width: float,
    height: float,
    angle_deg: float,
) -> list[tuple[float, float]]:
    """Compute RA/Dec corners of a rotated rectangular field region.

    Uses a flat-sky approximation to rotate corners around the field center.

    Parameters
    ----------
    ra_center : float
        Right Ascension of the field center in degrees.
    dec_center : float
        Declination of the field center in degrees.
    width : float
        Angular width of the field in degrees (before rotation). This is the
        physical angular extent, not the RA span; the ``cos(dec)`` division
        into an RA span is applied here.
    height : float
        Dec extent of the field in degrees (before rotation).
    angle_deg : float
        Rotation angle in degrees.

    Returns
    -------
    list of (ra, dec) tuples
        The four corners of the rotated rectangle.
    """
    hw, hh = width / 2.0, height / 2.0
    corners_local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    cos_dec = math.cos(math.radians(dec_center))
    # The flat-sky rotation that follows is only a sensible approximation
    # while cos(dec) is non-trivial. ``cos(dec) < 0.01`` corresponds to
    # ``|dec| > 89.43 deg``; beyond that the cylindrical projection breaks
    # down well before the cos_dec=0 singularity. At FYST's latitude the
    # south celestial pole stands about 23 deg up, above the 20 deg
    # elevation floor, so the guard is reachable: a well-formed field there
    # is refused with a PointingError.
    if abs(cos_dec) < 0.01:
        raise PointingError("FieldRegion too close to celestial pole (|dec| > 89.43 deg)")
    corners = []
    for dx, dy in corners_local:
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        corners.append((ra_center + rx / cos_dec, dec_center + ry))
    # A corner past a pole has no RA/Dec, so the field is refused as the
    # near-pole guard above refuses one.
    if any(abs(dec) > 90.0 for _, dec in corners):
        raise PointingError(
            f"FieldRegion corners pass a celestial pole (Dec={dec_center:.3f}, "
            f"height={height:.3f}, angle={angle_deg:.1f} deg)"
        )
    return corners


def _field_region_border(
    corners: list[tuple[float, float]], samples_per_edge: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the border of a field's box at evenly spaced points along each edge.

    The box is the flat-sky parallelogram :func:`_field_region_corners`
    returns, whose corners go round it, so each edge is the straight line
    between two adjacent corners in their unwrapped RA and Dec. Each edge is
    sampled at ``samples_per_edge`` points from its first corner up to, not
    including, the next, so every corner appears once.

    Parameters
    ----------
    corners : list of (ra, dec) tuples
        The four corners in degrees, in order round the box.
    samples_per_edge : int
        Points per edge, its first corner included.

    Returns
    -------
    ra, dec : ndarray
        The ``4 * samples_per_edge`` border points in degrees, RA unwrapped
        as the corners give it.
    """
    start = np.array(corners, dtype=float)
    end = np.roll(start, -1, axis=0)
    fraction = np.arange(samples_per_edge) / samples_per_edge
    points = start[:, None, :] + fraction[None, :, None] * (end - start)[:, None, :]
    return points[..., 0].ravel(), points[..., 1].ravel()


def _find_elevation_crossing(
    el_array: np.ndarray,
    search_times: Time,
    target_el: float,
    rising: bool,
) -> Time | None:
    """Find the first rising or setting crossing of a target elevation.

    Parameters
    ----------
    el_array : ndarray
        Elevation values at each search time.
    search_times : Time
        Array of search times.
    target_el : float
        Target elevation in degrees.
    rising : bool
        If True, find the rising crossing; if False, the setting crossing.

    Returns
    -------
    Time or None
        Time of crossing, or None if no crossing found.
    """
    crossings = _threshold_crossings(el_array, search_times, target_el, rising=rising)
    return crossings[0] if crossings else None


def _field_ra_edges(field: FieldRegion, angle: float) -> tuple[float, float]:
    """Return a field's leading and trailing RA edges at its central Dec, in degrees.

    The edges are the extreme RAs of the (rotated) field's corners
    (:func:`_field_region_corners`), taken as offsets from
    ``field.ra_center`` and never reduced modulo 360: reduced, the corners
    of a field across RA 0/360 that spans more than 180 deg would bound its
    complement.

    Raises
    ------
    PointingError
        If the field lies within 0.57 deg of a celestial pole or a corner of
        it lies past one, or if it spans a whole turn of RA or more at its
        central Dec, where it has no leading and trailing edges.
    """
    corners = _field_region_corners(
        field.ra_center, field.dec_center, field.width, field.height, angle
    )
    ra_min = min(ra for ra, _ in corners)
    ra_max = max(ra for ra, _ in corners)
    if ra_max - ra_min >= _WHOLE_TURN_RA_SPAN_DEG:
        raise PointingError(
            f"The field spans {ra_max - ra_min:.2f} deg of RA at its central Dec "
            f"(Dec={field.dec_center:.3f}), a whole turn or more, so it has no leading and "
            f"trailing RA edges to time a constant-elevation pass by."
        )
    return ra_min, ra_max


def _compute_ce_duration(
    field: FieldRegion,
    angle: float,
    elevation: float,
    coords_obj: Coordinates,
    base_search_time: Time,
    rising: bool,
    max_search_hours: float = _DEFAULT_MAX_SEARCH_HOURS,
    step_seconds: float = _DEFAULT_STEP_SECONDS,
) -> tuple[Time, Time, float]:
    """Compute when RA edges of a field cross the target elevation.

    Searches forward from ``base_search_time`` to find when the leading
    and trailing RA edges (at the field's central Dec) pass through the
    target elevation.

    The edges are the extreme corners of the (rotated) field, taken as
    offsets from ``field.ra_center`` and never reduced modulo 360, so a
    field across RA = 0/360, or wider than 180 deg of RA, keeps its own
    extent. A field spanning a whole turn of RA or more at its central Dec
    has no such edges and is refused.

    Parameters
    ----------
    field : FieldRegion
        Rectangular field specification.
    angle : float
        Field rotation angle in degrees.
    elevation : float
        Target elevation in degrees.
    coords_obj : Coordinates
        Coordinate transformer for the site.
    base_search_time : Time
        Start of the time search window.
    rising : bool
        If True, find the rising crossing; if False, the setting crossing.
    max_search_hours : float, optional
        Maximum time to search forward in hours. Default is 12.0.
    step_seconds : float, optional
        Time step for the search in seconds. Default is 30.0.

    Returns
    -------
    start_time : Time
        When the first RA edge crosses the target elevation.
    end_time : Time
        When the last RA edge crosses the target elevation.
    duration_seconds : float
        Duration in seconds.

    Raises
    ------
    ValueError
        If ``max_search_hours`` or ``step_seconds`` is not positive.
    PointingError
        If elevation crossings cannot be found in the search window, if the
        two field edges cross on different passes, if the field lies within
        0.57 deg of a celestial pole or a corner of it lies past one, or if
        it spans a whole turn of RA or more at its central Dec.

    Warns
    -----
    PointingWarning
        If the pass lasts longer than half of ``max_search_hours``.

    Notes
    -----
    This signature is load-bearing outside the planner: the offline
    simulator gates constant-elevation emission on the same solve, so the
    pass a schedule is built around and the pass a consumer reconstructs
    are one. See the module docstring for why that coupling is deliberate
    and the function stays private.
    """
    max_search_hours = _require_positive(max_search_hours, "max_search_hours")
    step_seconds = _require_positive(step_seconds, "step_seconds")

    ra_min, ra_max = _field_ra_edges(field, angle)

    dt_sec = np.arange(0, max_search_hours * 3600, step_seconds)
    search_times = base_search_time + TimeDelta(dt_sec * u.s)

    _, el_min_arr = coords_obj.radec_to_altaz(
        np.full(len(search_times), ra_min),
        np.full(len(search_times), field.dec_center),
        search_times,
    )
    _, el_max_arr = coords_obj.radec_to_altaz(
        np.full(len(search_times), ra_max),
        np.full(len(search_times), field.dec_center),
        search_times,
    )

    t_start = _find_elevation_crossing(el_min_arr, search_times, elevation, rising)
    t_end = _find_elevation_crossing(el_max_arr, search_times, elevation, rising)

    # The trailing edge crosses the RA span at the sidereal rate after the
    # leading one, so a pass longer than the horizon is never found whole,
    # wherever the search starts; both refusals below say so when it is.
    pass_hours = (ra_max - ra_min) / _SIDEREAL_RATE_DEG_S / 3600.0
    longer_than_search = (
        f" The field spans {ra_max - ra_min:.2f} deg of RA, so its pass lasts about "
        f"{pass_hours:.1f} h, longer than max_search_hours={max_search_hours}."
        if pass_hours > max_search_hours
        else ""
    )

    if t_start is None or t_end is None:
        # The two edges share the central Dec, so a trailing crossing still ahead
        # with the leading edge already past the elevation (the side the
        # crossing search could not start from) places the anchor inside the pass.
        leading_past = el_min_arr[0] >= elevation if rising else el_min_arr[0] < elevation
        inside_pass = (
            f" The anchor {base_search_time.iso} falls inside a pass: the field's leading "
            f"edge crossed el={elevation} before it, so the next crossing is about a "
            f"sidereal day later; anchor the plan before the pass opens."
            if t_start is None and t_end is not None and leading_past
            else ""
        )
        raise PointingError(
            f"Could not find elevation crossing for field edges at el={elevation} "
            f"(rising={rising}) within {max_search_hours} hours of "
            f"{base_search_time.iso}.{inside_pass}{longer_than_search}"
        )

    if t_start > t_end:
        # The leading RA edge crosses first on every pass, so a later leading
        # crossing means the two edges were found on different passes: the
        # anchor fell inside a pass and the horizon reached the next day.
        raise PointingError(
            f"Field edges cross el={elevation} on different passes (rising={rising}): "
            f"{base_search_time.iso} falls inside a pass; anchor the search before "
            f"the pass opens.{longer_than_search}"
        )

    duration_seconds = (t_end - t_start).to_value(u.s)

    if duration_seconds > max_search_hours * 3600 * 0.5:
        warnings.warn(
            f"Computed observation duration {duration_seconds / 3600:.1f}h is unusually long. "
            f"Check field geometry and search parameters.",
            PointingWarning,
            stacklevel=3,
        )

    return t_start, t_end, duration_seconds


def _vacuum_crossing_elevation(
    coords_obj: Coordinates, elevation: float, at: Time
) -> tuple[Coordinates, float]:
    """Return a vacuum transformer and the vacuum elevation that appears at ``elevation``.

    Refraction depends on elevation alone, so a position appears at
    ``elevation`` exactly when its vacuum elevation is the one returned:
    that of the point the refracted transform places at azimuth 0,
    ``elevation``, at ``at``, converted back in vacuum. Without an
    atmosphere ``coords_obj`` and ``elevation`` are returned as given.

    Parameters
    ----------
    coords_obj : Coordinates
        Coordinate transformer for the site, refracted or not.
    elevation : float
        The elevation at which positions are seen, in degrees.
    at : Time
        The instant the conversion is made at.

    Returns
    -------
    vacuum : Coordinates
        A vacuum transformer for the same site (``coords_obj`` itself when
        it has no atmosphere).
    crossing_el : float
        The vacuum elevation in degrees that appears at ``elevation``.
    """
    if coords_obj.atmosphere.pressure_hpa <= 0.0:
        return coords_obj, elevation
    ra_seen, dec_seen = coords_obj.altaz_to_radec(0.0, elevation, at)
    vacuum = Coordinates(coords_obj.site)
    _, crossing_el = vacuum.radec_to_altaz(ra_seen, dec_seen, at)
    return vacuum, crossing_el


def _ce_field_az_range(
    field: FieldRegion,
    angle: float,
    elevation: float,
    rising: bool,
    coords_obj: Coordinates,
    at: Time,
    padding: float,
) -> tuple[float, float]:
    """Return the azimuth range in which a field's border meets a constant elevation.

    On one side of the meridian, east (azimuth 0 to 180 deg) for the rising
    crossing and west (180 to 360 deg) for the setting one, the range runs
    from the smallest to the largest azimuth at which points of the border
    of the field's box (:func:`_field_region_corners`, :func:`_field_region_border`)
    cross ``elevation``. A point of apparent declination ``d`` crosses
    elevation ``e`` at site latitude ``lat`` at the azimuth ``A`` given by

        cos A = (sin d - sin e sin lat) / (cos e cos lat),

    with ``A`` in ``[0, 180]`` east of the meridian and ``360 - A`` west of
    it. ``A`` depends on the declination alone and moves monotonically with
    it on each side, so the range runs between the crossings of the box's
    lowest and highest apparent declinations, and holds every azimuth at
    which the box is at ``elevation`` on that side over the field's whole
    crossing. It depends on the field's declination extent, not on its RA
    extent, and not on time beyond the apparent places.

    Those extremes lie on the border: the apparent declination differs from
    the catalogue one by a smooth shift of a few tenths of a degree at most
    (precession since J2000, nutation, aberration), so it has no stationary
    point inside the box. They need not lie at corners, since an edge of
    constant catalogue declination is not one of constant apparent
    declination: the shift varies along it (precession by about
    ``0.15 cos(RA)`` deg in 2026), and on an edge spanning tens of degrees of
    RA it bows past both corners. So each edge is sampled at
    ``_CE_BORDER_SAMPLES_PER_EDGE`` points.

    The ``sin d`` of each border point is read from the site's own
    transform: the points are converted to azimuth and elevation once, at
    ``at``, in vacuum, and ``sin d = sin lat sin el + cos lat cos el cos az``.
    With an atmosphere on ``coords_obj`` a point is at ``elevation`` when its
    refracted elevation is, so ``e`` is the vacuum elevation that appears at
    ``elevation`` (:func:`_vacuum_crossing_elevation`).

    The cosine is clipped to ``[-1, 1]``, so a point whose declination never
    reaches ``elevation`` (culminates short of it, or for a circumpolar
    field stays above it) lands on the meridian azimuth of its side, 0 or
    360 deg in the north and 180 deg in the south: the range then runs to
    the meridian and stops there. Near that limit ``A`` changes steeply with
    declination, so for a field whose extreme declination only grazes
    ``elevation`` that end of the range is sensitive to the declination and
    to ``elevation``. The range is not placed on an azimuth branch;
    :func:`_compute_ce_az_range` places it for the sweep the planner builds.

    Parameters
    ----------
    field : FieldRegion
        Rectangular field specification.
    angle : float
        Field rotation angle in degrees.
    elevation : float
        The scan elevation in degrees.
    rising : bool
        ``True`` for the range east of the meridian, ``False`` for the range
        west of it.
    coords_obj : Coordinates
        Coordinate transformer for the site, refracted or not.
    at : Time
        The instant the border's apparent places are read at.
    padding : float
        Extra padding in degrees on each side.

    Returns
    -------
    az_min, az_max : float
        Azimuth range in degrees, ``0 <= az_min < az_max <= 180`` east of
        the meridian or ``180 <= az_min < az_max <= 360`` west of it,
        widened by ``padding`` on each side, so it may lie outside
        ``[0, 360]`` by the padding and may overrun an azimuth limit.

    Raises
    ------
    PointingError
        If every border point lands on the same azimuth, so the range is
        empty: no part of the field's border ever reaches ``elevation``; or
        if the field lies within 0.57 deg of a celestial pole or a corner of
        it lies past one (:func:`_field_region_corners`).
    """
    corners = _field_region_corners(
        field.ra_center, field.dec_center, field.width, field.height, angle
    )
    ra, dec = _field_region_border(corners, _CE_BORDER_SAMPLES_PER_EDGE)
    vacuum, crossing_el = _vacuum_crossing_elevation(coords_obj, elevation, at)
    az, el = vacuum.radec_to_altaz(ra, dec, at)
    az = np.radians(np.asarray(az, dtype=float))
    el = np.radians(np.asarray(el, dtype=float))
    phi = math.radians(coords_obj.site.latitude)
    e = math.radians(crossing_el)
    sin_d = math.sin(phi) * np.sin(el) + math.cos(phi) * np.cos(el) * np.cos(az)
    cos_a = (sin_d - math.sin(e) * math.sin(phi)) / (math.cos(e) * math.cos(phi))
    crossing = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
    if not rising:
        crossing = 360.0 - crossing
    az_min, az_max = float(crossing.min()), float(crossing.max())
    if not az_max > az_min:
        raise PointingError(
            f"The field never crosses el={elevation:.2f} deg: no part of its border reaches "
            f"that elevation (Dec={field.dec_center:.3f}, height={field.height:.3f}, "
            f"angle={angle:.1f} deg), so it has no azimuth range there."
        )
    return az_min - padding, az_max + padding


def _compute_ce_az_range(
    field: FieldRegion,
    angle: float,
    elevation: float,
    rising: bool,
    coords_obj: Coordinates,
    at: Time,
    padding: float,
    *,
    velocity: float,
    az_accel: float,
    duration: float,
    timestep: float,
    boresight_az_offset: float = 0.0,
) -> tuple[float, float, int, float]:
    """Compute the azimuth range a constant-elevation pass sweeps, on its branch, and its legs.

    The range in which the field's border meets ``elevation`` on the
    ``rising`` side (:func:`_ce_field_az_range`), on the branch
    :func:`_ce_branch_shift` chooses for the sweep the planner builds
    across it: the whole legs ``duration`` rounds to
    (:func:`_quantize_ce_duration`), sampled at ``timestep``, and the
    boresight's, the same moved by ``boresight_az_offset``. So a setting
    pass just west of north, whose turnarounds at a high scan speed would
    end above 360 deg, is placed just below 0 deg instead. The legs are
    returned with the range, so the planner builds the sweep the branch was
    chosen for.

    Parameters
    ----------
    field, angle, elevation, rising, coords_obj, at, padding
        As for :func:`_ce_field_az_range`.
    velocity : float
        Azimuth scan speed in deg/s (mount frame).
    az_accel : float
        Azimuth acceleration in deg/s^2 (mount frame).
    duration : float
        The pass length in seconds, before it is rounded to whole legs.
    timestep : float
        Time between the trajectory's samples in seconds.
    boresight_az_offset : float, optional
        The boresight's azimuth minus the swept range's, in degrees
        (:func:`_boresight_az_offset`): nonzero when a detector offset
        places a module, not the boresight, on the field. Default 0.0.

    Returns
    -------
    az_min, az_max : float
        Azimuth range in degrees, on the 360 deg branch
        :func:`_ce_branch_shift` chooses; a range whose samples fit no
        branch is returned as :func:`_ce_field_az_range` gives it, for the
        trajectory's bounds validation to refuse.
    n_scans : int
        Number of azimuth legs, at least 1.
    built_duration : float
        The sweep's length in seconds: ``n_scans`` legs and the turnarounds
        between them.

    Raises
    ------
    PointingError
        As :func:`_ce_field_az_range` raises it: for a field that never
        reaches ``elevation``, or one at or past a celestial pole.
    """
    az_min, az_max = _ce_field_az_range(field, angle, elevation, rising, coords_obj, at, padding)
    # Three paths land the sweep on an out-of-range branch: padding pushing a
    # range that ends at or near 360 deg past the limit (a setting range that
    # reaches the northern meridian ends there), the turnaround overshoot
    # doing the same at speed, and a detector offset moving the boresight past
    # it. All are representation choices, not infeasibility.
    return _place_ce_sweep(
        az_min,
        az_max,
        velocity=velocity,
        az_accel=az_accel,
        duration=duration,
        timestep=timestep,
        boresight_az_offset=boresight_az_offset,
        limits=coords_obj.site.telescope_limits.azimuth,
    )


def _place_ce_sweep(
    az_min: float,
    az_max: float,
    *,
    velocity: float,
    az_accel: float,
    duration: float,
    timestep: float,
    boresight_az_offset: float,
    limits: AxisLimits,
) -> tuple[float, float, int, float]:
    """Round a sweep of ``(az_min, az_max)`` to whole legs and place it on its branch.

    The legs are counted first (:func:`_quantize_ce_duration`), since they
    decide which turnarounds the sweep makes, and the range is then moved by
    the whole turn :func:`_ce_branch_shift` returns for that sweep. A moved
    range's throw can differ from the given one in its last bits, so the same
    legs are timed again on the moved range, the one the pattern sweeps: the
    leg count stays the one the branch was chosen for, and planning the
    returned range again, for the returned duration, gives the same sweep.
    Both forms of ``plan_constant_el_scan`` place their sweep here.

    Parameters
    ----------
    az_min, az_max, velocity, az_accel, timestep, boresight_az_offset, limits
        As for :func:`_ce_branch_shift`.
    duration : float
        The requested length in seconds, before it is rounded to whole legs.

    Returns
    -------
    az_min, az_max : float
        The range on its branch, in degrees.
    n_scans : int
        Number of azimuth legs, at least 1.
    built_duration : float
        The sweep's length in seconds.
    """
    n_scans, built_duration = _quantize_ce_duration(
        az_throw=az_max - az_min,
        velocity=velocity,
        duration=duration,
        az_accel=az_accel,
    )
    shift = _ce_branch_shift(
        az_min,
        az_max,
        n_scans=n_scans,
        duration=built_duration,
        timestep=timestep,
        velocity=velocity,
        az_accel=az_accel,
        boresight_az_offset=boresight_az_offset,
        limits=limits,
    )
    az_min, az_max = az_min + shift, az_max + shift
    if shift != 0.0:
        # The same legs, timed on the moved throw: asked for exactly the first
        # count's whole legs, the quantiser cannot round to another count.
        n_scans, built_duration = _quantize_ce_duration(
            az_throw=az_max - az_min,
            velocity=velocity,
            duration=built_duration,
            az_accel=az_accel,
        )
    return az_min, az_max, n_scans, built_duration


def _boresight_az_offset(
    elevation: float, detector_offset: InstrumentOffset | None, site: Site
) -> float:
    """Return the boresight's azimuth minus the detector's on a constant-elevation sweep.

    With a ``detector_offset`` both forms of ``plan_constant_el_scan``
    command the detector along the sweep, and the trajectory builder moves the boresight
    so that the detector lands there
    (:func:`~fyst_trajectories.offsets.apply_detector_offset`). At a fixed
    detector elevation the focal-plane rotation is fixed too, so the
    boresight's azimuth differs from the detector's by one constant, the
    value returned, and its elevation by another.

    Where the solve does not converge, as near the zenith, the builder's
    own solve fails at every sample of the sweep on any branch, since the
    elevation alone decides it. So 0.0 is returned and the branch is left
    to the commanded path, as without an offset, and the builder's refusal
    is the one it would give without this term.

    Parameters
    ----------
    elevation : float
        The detector's elevation in degrees, the sweep's commanded one.
    detector_offset : InstrumentOffset or None
        The detector's offset from the boresight; ``None`` returns 0.0.
    site : Site
        Telescope site (provides the Nasmyth sign).

    Returns
    -------
    float
        Degrees to add to the detector's azimuth to get the boresight's;
        0.0 without an offset or where the boresight cannot be solved.
    """
    if detector_offset is None:
        return 0.0
    try:
        bore_az, _ = _solve_boresight(0.0, elevation, detector_offset, site=site)
    except OffsetInversionError:
        return 0.0
    return float(bore_az)


def _ce_branch_shift(
    az_min: float,
    az_max: float,
    *,
    n_scans: int,
    duration: float,
    timestep: float,
    velocity: float,
    az_accel: float,
    boresight_az_offset: float,
    limits: AxisLimits,
) -> float:
    """Return the whole turn that places a constant-elevation sweep where the mount can run it.

    Both forms of ``plan_constant_el_scan`` command ``n_scans`` legs across
    ``(az_min, az_max)``, the first from ``az_min`` up, for ``duration``
    seconds sampled at ``timestep``. Each turnaround between legs overshoots
    the edge it turns at by ``5 v**2 / (8 a)``
    (:func:`~fyst_trajectories.patterns.turnarounds.swept_az_envelope`), so
    one leg sweeps the range alone, two legs reach the overshoot past
    ``az_max`` as well, and three or more reach both overshoots. The
    boresight, which the mount follows, sweeps the same azimuths moved by
    ``boresight_az_offset`` (:func:`_boresight_az_offset`).

    The pattern checks the commanded samples (:func:`_ce_az_reach`) against
    the azimuth limits, and the trajectory builder the boresight's, so the
    sweep can be built on a branch where both lie within the limits. The
    mount runs the path between the samples too, and a sample can stop
    short of a turnaround's apex by up to ``3 a dt**2 / 16`` at a step
    ``dt``, so a branch that also holds the boresight's whole path, every
    turnaround to its apex, is better. The sweep keeps the branch it is
    given unless the one a turn away is better by these two tests, and a
    sweep that no branch can build is left where it is, for the
    trajectory's bounds validation to refuse. Each branch is judged at the
    azimuths the sweep would be placed at, with the pattern's own
    arithmetic, so the judgement rounds as the build does.

    Parameters
    ----------
    az_min, az_max : float
        The swept range in degrees, ``az_min < az_max``.
    n_scans : int
        Number of azimuth legs (:func:`_quantize_ce_duration`), at least 1.
    duration : float
        The sweep's length in seconds: ``n_scans`` legs and the turnarounds
        between them.
    timestep : float
        Time between the trajectory's samples in seconds.
    velocity : float
        Azimuth scan speed in deg/s (mount frame).
    az_accel : float
        Azimuth acceleration in deg/s^2 (mount frame).
    boresight_az_offset : float
        The boresight's azimuth minus the swept range's, in degrees.
    limits : AxisLimits
        The site's azimuth limits.

    Returns
    -------
    float
        ``-360.0``, ``0.0`` or ``360.0``, to add to ``az_min`` and ``az_max``.
    """
    offset = boresight_az_offset

    def inside(*azimuths: float) -> bool:
        return all(limits.is_in_range(az) for az in azimuths)

    def commanded_path(shift: float) -> tuple[float, float]:
        # The legs and the turnarounds they make, each to its apex, bounded as
        # the pattern bounds its samples, so every sample lies on this path.
        lo, hi = az_min + shift, az_max + shift
        env_lo, env_hi = swept_az_envelope(lo, hi, velocity, az_accel)
        return (env_lo if n_scans >= 3 else lo), (env_hi if n_scans >= 2 else hi)

    def grade(shift: float) -> int:
        # 2: the samples and the boresight's whole path fit; 1: only the
        # samples, the commanded and the boresight's; 0: not the samples.
        lo, hi = _ce_az_reach(
            az_min + shift, az_max + shift, velocity, az_accel, duration, timestep
        )
        if not inside(lo, hi, lo + offset, hi + offset):
            return 0
        path_lo, path_hi = commanded_path(shift)
        return 2 if inside(path_lo + offset, path_hi + offset) else 1

    path_lo, path_hi = commanded_path(0.0)
    if inside(path_lo, path_hi, path_lo + offset, path_hi + offset):
        return 0.0  # every sample lies on the paths, so the branch grades 2 unsampled
    here = grade(0.0)
    if here == 2:
        return 0.0
    # A path overruns one limit here, so only the branch a turn away from it
    # can do better.
    other = -360.0 if max(path_hi, path_hi + offset) > limits.max else 360.0
    return other if grade(other) > here else 0.0


def _lsa_window_endpoints(lsa_window: "tuple[float, float] | list[float]") -> tuple[float, float]:
    """Return an LSA window's ``(min_lsa, max_lsa)`` in degrees, refusing a malformed one.

    The window is a pair, a tuple, a list (ECSV/JSON serialisation turns
    a tuple into a list) or a NumPy array, of real numbers in ``[0, 360)`` with unequal
    endpoints; ``max_lsa < min_lsa`` wraps through LSA = 0/360.

    Parameters
    ----------
    lsa_window : tuple or list of (min_lsa, max_lsa)
        Local Sidereal Angle window in degrees.

    Returns
    -------
    tuple of float
        ``(min_lsa, max_lsa)``.

    Raises
    ------
    ValueError
        If ``lsa_window`` is not a pair, an endpoint is not a real number
        (text, a ``bool``, ``None``), an endpoint is not finite or lies
        outside ``[0, 360)``, or the endpoints are equal.
    """
    if not isinstance(lsa_window, (tuple, list, np.ndarray)) or len(lsa_window) != 2:
        raise ValueError(f"lsa_window must be a pair (min_lsa, max_lsa), got {lsa_window!r}")
    finite = [_is_finite(value, f"lsa_window[{i}]") for i, value in enumerate(lsa_window)]
    min_lsa, max_lsa = float(lsa_window[0]), float(lsa_window[1])
    if not all(finite) or not (0.0 <= min_lsa < 360.0) or not (0.0 <= max_lsa < 360.0):
        raise ValueError(f"lsa_window endpoints must lie in [0, 360); got ({min_lsa}, {max_lsa})")
    if min_lsa == max_lsa:
        raise ValueError(
            f"lsa_window has equal endpoints ({min_lsa}); refusing zero-duration window"
        )
    return min_lsa, max_lsa


def _lsa_window_rising_side(
    ra_center: float, dec_center: float, t_open: Time, t_close: Time, coords_obj: Coordinates
) -> bool:
    """Return the side of the meridian an LSA-window pass is scanned on, ``True`` for rising.

    The window chooses no crossing, so the range is taken on the side of the
    meridian the field centre is on at the window's middle: east, an azimuth
    below 180 deg, is the rising side.

    Parameters
    ----------
    ra_center, dec_center : float
        The field centre in degrees.
    t_open, t_close : Time
        The window's start and end.
    coords_obj : Coordinates
        Coordinate transformer for the site.

    Returns
    -------
    bool
        ``True`` for the rising (east) side, ``False`` for the setting side.
    """
    t_mid = t_open + (t_close - t_open) / 2.0
    centre_az, _ = coords_obj.radec_to_altaz(ra_center, dec_center, t_mid)
    return bool(centre_az % 360.0 < 180.0)


def _compute_ce_duration_from_lsa(
    lsa_window: "tuple[float, float] | list[float]",
    coords_obj: Coordinates,
    base_search_time: Time,
    max_search_hours: float = _DEFAULT_MAX_SEARCH_HOURS,
    step_seconds: float = _DEFAULT_STEP_SECONDS,
) -> tuple[Time, Time, float]:
    """Compute scan start/end from a Local Sidereal Angle window.

    Searches forward from ``base_search_time`` for the first time at which
    Local Sidereal Time (in degrees) crosses ``lsa_window[0]`` in the
    increasing direction. The end time is fixed at ``t_start +
    (max_lsa - min_lsa) mod 360 / 15`` hours of UTC, about 0.3 percent
    longer than the sidereal window it names.

    Wrap-around windows (``max_lsa < min_lsa``) are handled explicitly:
    the duration is computed modulo 360 deg, so ``(310, 10)`` is a
    60 deg at 15 deg/h = 4 hour scan crossing the LST = 0/360 boundary.

    Parameters
    ----------
    lsa_window : tuple or list of (min_lsa, max_lsa)
        Local Sidereal Angle window in degrees. Both endpoints must lie
        in ``[0, 360)`` and must not be equal. ``max_lsa < min_lsa``
        means the window wraps through LSA = 0/360. Accepts a list as
        well as a tuple so callers can pass a value that has
        round-tripped through ECSV/JSON serialisation (which converts
        tuples to lists).
    coords_obj : Coordinates
        Coordinate transformer for the site (used for ``get_lst`` and
        the underlying ``EarthLocation``).
    base_search_time : Time
        Earliest time at which to start searching.
    max_search_hours : float, optional
        Maximum search horizon in hours. Default is 12.0. Must be
        positive.
    step_seconds : float, optional
        Time step for the search in seconds. Default is 30.0.

    Returns
    -------
    start_time : Time
        First time at or after ``base_search_time`` at which LST passes
        through ``min_lsa`` in the increasing direction.
    end_time : Time
        ``start_time`` plus the LSA-derived duration.
    duration_seconds : float
        Duration of the window in seconds.

    Raises
    ------
    ValueError
        If ``lsa_window`` is not a pair of real numbers, ``min_lsa ==
        max_lsa`` (zero-duration window), either endpoint is outside
        ``[0, 360)`` (:func:`_lsa_window_endpoints`), or
        ``max_search_hours`` or ``step_seconds`` is not positive.
    PointingError
        If no increasing crossing of ``min_lsa`` is found within
        ``max_search_hours`` of ``base_search_time``.

    Warns
    -----
    PointingWarning
        If the window lasts longer than 6 h (``_CE_LONG_BLOCK_HOURS``).

    Notes
    -----
    Uses ``sidereal_time('apparent')`` (via
    :meth:`Coordinates.get_lst`), consistent with the rest of the
    fyst-trajectories pipeline. Tools that use ``sidereal_time('mean')``
    instead differ by at most the equation of the equinoxes (~20 arcsec
    ~ 1.3 s of time), well below operationally relevant precision for
    LSA-window scheduling.
    """
    min_lsa, max_lsa = _lsa_window_endpoints(lsa_window)
    max_search_hours = _require_positive(max_search_hours, "max_search_hours")
    # 0 makes the arange below divide by zero; a negative step makes it
    # empty, and the crossing search then reports "no crossing found"
    # over a horizon it never sampled.
    step_seconds = _require_positive(step_seconds, "step_seconds")

    duration_deg = (max_lsa - min_lsa) % 360.0
    duration_hours = duration_deg / 15.0
    duration_seconds = duration_hours * 3600.0

    dt_sec = np.arange(0.0, max_search_hours * 3600.0, step_seconds)
    search_times = base_search_time + TimeDelta(dt_sec * u.s)
    lsa = np.asarray(coords_obj.get_lst(search_times))

    # Unwrap LSA into a monotonic series so straddle detection works
    # uniformly when ``min_lsa`` sits at or near the LST = 0/360
    # boundary. With wrapped samples (e.g. 359.9 -> 0.1) and ``min_lsa
    # = 0``, the naive product test ``(lsa - 0) * (lsa_next - 0)`` is
    # positive at the wrap edge, producing a silent miss. ``np.unwrap``
    # turns 359.9 -> 360.1 -> ..., and LST is monotonically increasing in
    # time, so the unwrapped series stays monotonic.
    lsa_unwrapped = np.unwrap(np.deg2rad(lsa)) * 180.0 / np.pi
    # The smallest ``target = min_lsa + 360*k`` that lies at or after
    # the first sample is the only candidate crossing within the
    # horizon (LST advances ~15 deg/h; even a 24-hour search covers only
    # one full wrap, so the *first* increasing crossing of ``min_lsa``
    # is uniquely determined by this offset).
    k = math.ceil((lsa_unwrapped[0] - min_lsa) / 360.0)
    target = min_lsa + 360.0 * k
    if target > lsa_unwrapped[-1]:
        raise PointingError(
            f"lsa_window min_lsa={min_lsa} deg not reached in increasing direction within "
            f"{max_search_hours} h of {base_search_time.iso}. "
            f"LSA swept {lsa[0]:.1f} deg -> {lsa[-1]:.1f} deg."
        )

    # Find the [idx, idx+1] bracket that contains ``target``.
    idx = int(np.searchsorted(lsa_unwrapped, target, side="right")) - 1
    # Clamp into [0, len-2]: a target equal to the last sample gives
    # ``idx == len - 1`` (then ``frac`` is 1), and the lower bound only guards
    # rounding in ``k``; ``target`` equal to the first sample gives ``idx == 0``.
    idx = max(0, min(idx, len(lsa_unwrapped) - 2))
    lsa_left = lsa_unwrapped[idx]
    lsa_right = lsa_unwrapped[idx + 1]
    # ``lsa_right > lsa_left`` is guaranteed by monotonicity of the
    # unwrapped LST series (denom > 0 strictly).
    frac = (target - lsa_left) / (lsa_right - lsa_left)
    t_start = search_times[idx] + TimeDelta(frac * step_seconds * u.s)
    t_end = t_start + TimeDelta(duration_seconds * u.s)

    # Absolute threshold: LSA-window duration is independent of the
    # search horizon, so the warning should reflect operational scale
    # rather than couple to ``max_search_hours``.
    if duration_hours > _CE_LONG_BLOCK_HOURS:
        warnings.warn(
            f"LSA-derived duration {duration_hours:.1f}h is unusually long for a single "
            f"CE block. Check lsa_window={lsa_window}.",
            PointingWarning,
            stacklevel=3,
        )

    return t_start, t_end, duration_seconds


def _quantize_ce_duration(
    *,
    az_throw: float,
    velocity: float,
    duration: float,
    az_accel: float,
) -> tuple[int, float]:
    """Quantise a constant-elevation window into whole azimuth legs.

    Rounds the requested ``duration`` to an integer number of azimuth legs
    (each sweeping ``az_throw`` at ``velocity``) and returns the leg count
    together with the exact duration the trajectory builder produces for that
    count. Shared by both forms of ``plan_constant_el_scan`` and the
    source-CES planner so all three quantise a CE window identically.

    Parameters
    ----------
    az_throw : float
        Azimuth sweep of a single leg in degrees.
    velocity : float
        Azimuth scan speed in degrees/second.
    duration : float
        Requested window duration in seconds; rounded to whole legs.
    az_accel : float
        Azimuth acceleration in degrees/second^2 (sets the turnaround time).

    Returns
    -------
    n_scans : int
        Number of azimuth legs (at least 1).
    actual_duration : float
        Duration in seconds for ``n_scans`` legs including turnarounds.

    Raises
    ------
    ValueError
        If ``az_throw`` is not positive. A zero throw is a scan that sweeps
        nothing, and its leg count would be made entirely of turnarounds.
    """
    if not (az_throw > 0.0):
        raise ValueError(
            f"az_throw must be positive, got {az_throw}; a constant-elevation "
            "scan with no azimuth sweep covers nothing."
        )
    t_cruise = az_throw / velocity
    t_turnaround = turnaround_duration_sec(velocity, az_accel)
    # ``n_scans`` cruises with ``n_scans - 1`` inter-leg turnarounds (the
    # trailing turnaround of the final leg is unused), so the window holds
    # ``(duration + t_turnaround) / (t_cruise + t_turnaround)`` legs. Counting
    # legs by the cruise time alone overshoots the window whenever the
    # turnaround is not negligible against a leg (fast, short legs).
    # ``floor(x + 0.5)`` rather than ``round``, which rounds a tie to the even
    # count (2.5 legs down, 3.5 up).
    legs = (duration + t_turnaround) / (t_cruise + t_turnaround)
    n_scans = max(1, int(math.floor(legs + 0.5)))
    actual_duration = n_scans * t_cruise + (n_scans - 1) * t_turnaround
    return n_scans, actual_duration


def _window_side(az_min: float, az_max: float) -> str | None:
    """Return which crossing a fixed azimuth window watches: rising or setting.

    East of the meridian (azimuth between 0 and 180 deg) a field rises
    through a constant elevation, and west of it (180 to 360 deg) it sets.
    The side is read from the window's midpoint. A window that reaches a
    meridian azimuth (a multiple of 180 deg, endpoints included) has no
    single side.

    Parameters
    ----------
    az_min, az_max : float
        The window in degrees, ``az_min < az_max``, on any branch.

    Returns
    -------
    str or None
        ``"rising"``, ``"setting"``, or ``None`` for a window that reaches
        the meridian.
    """
    if math.floor(az_max / 180.0) * 180.0 >= az_min:
        return None
    return "rising" if (0.5 * (az_min + az_max)) % 360.0 < 180.0 else "setting"


def _check_window_field_drift(
    field: FieldRegion,
    *,
    side: str,
    elevation: float,
    coords_obj: Coordinates,
    start: Time,
    duration: float,
) -> None:
    """Warn when a fixed-window scan's timing misses the field it names.

    Heuristic and warn-only, with no crossing search. The field is taken as
    its two RA edges at its central Dec (the edges :func:`_compute_ce_duration`
    times a pass by, here for the unrotated field), read once, at ``start``.
    Each edge's hour angle ``H`` and apparent declination ``d`` are read off
    its azimuth and elevation, ``H = atan2(-sin(az) cos(el), cos(lat) sin(el)
    - sin(lat) cos(el) cos(az))`` and ``sin(d) = sin(lat) sin(el) + cos(lat)
    cos(el) cos(az)``, and ``d`` fixes the hour angle at which the edge
    crosses ``elevation`` on ``side``. The hour angle grows at the sidereal
    rate, so the field's pass runs from the leading edge's crossing to the
    trailing edge's once per sidereal day. A scan that starts inside a pass,
    or lasts until the next one begins, is silent; one that lies wholly
    between two passes draws one warning, naming the nearer of them: the
    pass it starts after, or the one it ends before.

    With an atmosphere on ``coords_obj`` the edges are read in vacuum and
    judged at the vacuum elevation the atmosphere refracts to ``elevation``
    (:func:`_vacuum_crossing_elevation`); refraction depends on elevation
    alone, so the crossings are those of the refracted positions, timed as
    in vacuum. The culminations are judged from the site latitude
    and the field's catalogue Dec (precession since J2000 ignored, which moves
    a declination by at most 0.2 deg before 2035), so an ``elevation`` that
    close to a culmination can be misjudged. A field spanning the whole RA
    circle at its central Dec has no edges, and only the culmination check
    runs.

    Parameters
    ----------
    field : FieldRegion
        The field the scan is meant to see drift through the window.
    side : str
        ``"rising"`` or ``"setting"``, from :func:`_window_side`.
    elevation : float
        Commanded elevation in degrees.
    coords_obj : Coordinates
        Coordinate transformer for the site.
    start : Time
        The scan's first sample.
    duration : float
        Requested duration in seconds.

    Warns
    -----
    PointingWarning
        If the field's central Dec never crosses ``elevation``, or if the
        scan lies wholly between two of the field's passes on ``side``. At
        most one is raised, attributed to the caller of the planner.
    """
    dec = field.dec_center
    lat = coords_obj.site.latitude
    upper = 90.0 - abs(dec - lat)
    lower = abs(dec + lat) - 90.0
    if not lower < elevation < upper:
        warnings.warn(
            f"The field never crosses el={elevation:.2f} deg: its central declination "
            f"(Dec={dec:.3f}) culminates at el={upper:.2f} and el={lower:.2f} deg, so the "
            f"window scan does not see it drift through.",
            PointingWarning,
            stacklevel=3,
        )
        return

    ra_span = field.width / math.cos(math.radians(dec))
    # A span of a whole turn, to rounding, leaves no edges to read.
    if ra_span >= _WHOLE_TURN_RA_SPAN_DEG:
        return

    vacuum, crossing_el = _vacuum_crossing_elevation(coords_obj, elevation, start)

    half_span = 0.5 * ra_span
    ra_lead = field.ra_center - half_span
    ra_trail = field.ra_center + half_span
    az, el = vacuum.radec_to_altaz(np.array([ra_trail, ra_lead]), np.full(2, dec), start)
    az = np.radians(np.asarray(az, dtype=float))
    el = np.radians(np.asarray(el, dtype=float))
    phi = math.radians(lat)
    hour_angle = np.degrees(
        np.arctan2(
            -np.sin(az) * np.cos(el),
            math.cos(phi) * np.sin(el) - math.sin(phi) * np.cos(el) * np.cos(az),
        )
    )
    sin_d = math.sin(phi) * np.sin(el) + math.cos(phi) * np.cos(el) * np.cos(az)
    cos_crossing = (math.sin(math.radians(crossing_el)) - math.sin(phi) * sin_d) / (
        math.cos(phi) * np.sqrt(1.0 - sin_d**2)
    )
    # The clamp places an edge whose apparent Dec culminates just short of the
    # crossing elevation, where the catalogue Dec reaches it, at the culmination.
    crossing = np.degrees(np.arccos(np.clip(cos_crossing, -1.0, 1.0)))
    if side == "rising":
        crossing = -crossing
    # Sidereal degrees since each edge last crossed ``elevation`` on ``side``:
    # inside a pass the leading edge has crossed more recently than the trailing one.
    since_trail, since_lead = (hour_angle - crossing) % 360.0
    if since_lead < since_trail:
        return

    until_next = 360.0 - since_lead
    sweep = duration * _SIDEREAL_RATE_DEG_S
    if sweep >= until_next:
        return

    if since_trail <= until_next - sweep:
        crossed_at = start - TimeDelta(since_trail / _SIDEREAL_RATE_DEG_S * u.s)
        warnings.warn(
            f"The window scan starts after the field has drifted through: its trailing RA "
            f"edge (RA={ra_trail % 360.0:.3f}, Dec={dec:.3f}) has already crossed "
            f"el={elevation:.2f} deg on the {side} side, at {crossed_at.utc.iso}.",
            PointingWarning,
            stacklevel=3,
        )
    else:
        end = start + TimeDelta(duration * u.s)
        arrives_at = start + TimeDelta(until_next / _SIDEREAL_RATE_DEG_S * u.s)
        warnings.warn(
            f"The window scan ends before the field arrives: its leading RA edge "
            f"(RA={ra_lead % 360.0:.3f}, Dec={dec:.3f}) has not crossed "
            f"el={elevation:.2f} deg on the {side} side by the requested end, {end.utc.iso}; "
            f"it crosses at {arrives_at.utc.iso}.",
            PointingWarning,
            stacklevel=3,
        )
