"""Pure helpers used by scheduler phases.

Dependency-free (no scheduler state); consumed by
:mod:`.phases` and the :func:`generate_timeline` wrapper.
"""

import math

import numpy as np
from astropy.time import Time, TimeDelta

from ...coordinates import Coordinates
from ...patterns.utils import normalize_azimuth
from ...planning import FieldRegion

# Private planner reuse, on purpose: the scheduler must gate constant-elevation
# emission on exactly the crossing solve that plan_constant_el_scan runs at
# reconstruction (planning = execution), so it calls the planner's own solver
# rather than approximating it.
from ...planning._ce_geometry import _compute_ce_duration
from ...site import Site
from ..constraints import Constraint, ElevationConstraint, SunAvoidanceConstraint
from ..models import ObservingPatch, OverheadModel

__all__ = [
    "_ce_crossing_corridor",
    "_ce_visit_plan",
    "_compute_az_range",
    "_compute_scan_duration",
    "_default_constraints",
    "_evaluate_patch",
    "_normalize_az",
    "_time_until_set",
]

#: Cached corridor hits are only trusted for anchors at least this many
#: seconds before the pass's opening crossing: the planner's 30 s-step
#: crossing search needs the leading edge still below the target elevation
#: at the anchor, so boundary anchors re-solve instead of trusting the cache.
_CE_ANCHOR_MARGIN_SEC = 60.0

#: Minimum scheduler time between re-solves after a corridor miss. A miss
#: costs a full 12-hour forward search; without a rate limit an infeasible
#: patch would re-run it every selection tick for the rest of the night.
_CE_MISS_RESOLVE_SEC = 600.0

#: Slew allowance folded into the CE readiness lead: a visit may start this
#: many seconds (plus one scheduler tick) before the pass's opening crossing
#: so the pre-scan slew never pushes the anchor past the opening.
_CE_READY_SLEW_ALLOWANCE_SEC = 180.0


def _normalize_az(az: float, site: Site) -> float:
    """Normalize a scalar azimuth into the site's cable-wrap window.

    Wraps :func:`fyst_trajectories.patterns.utils.normalize_azimuth` (which
    operates on arrays) for the scalar azimuths the scheduler carries.
    Raw astropy azimuths are in ``[0, 360)``; the slew-time and boresight
    math must compare them in the telescope's ``[az_min, az_max]`` window
    or a north-straddling pair inflates the slew distance ~17x and flips
    the boresight ~180 deg.

    Parameters
    ----------
    az : float
        Azimuth in degrees (typically raw astropy ``[0, 360)``).
    site : Site
        Site providing the azimuth limits.

    Returns
    -------
    float
        Azimuth shifted into ``[az_min, az_max]``.
    """
    return float(normalize_azimuth(np.array([az], dtype=float), site)[0])


def _default_constraints(site: Site) -> list[Constraint]:
    """Create default constraints from site configuration."""
    constraints: list[Constraint] = [
        ElevationConstraint(
            el_min=site.telescope_limits.elevation.min,
            el_max=site.telescope_limits.elevation.max,
        ),
    ]
    if site.sun_avoidance.enabled:
        constraints.append(SunAvoidanceConstraint(min_angle=site.sun_avoidance.exclusion_radius))
    return constraints


def _evaluate_patch(
    patch: ObservingPatch,
    time: Time,
    az: float,
    el: float,
    coords: Coordinates,
    constraints: list[Constraint],
) -> float:
    """Evaluate a patch against all constraints.

    Returns the product of all constraint scores. A zero from any
    constraint immediately returns 0.0 (short-circuit).
    """
    score = 1.0
    for constraint in constraints:
        s = constraint.score(patch, time, az, el, coords)
        if s == 0.0:
            return 0.0
        score *= s
    return score


def _ce_crossing_corridor(
    patch: ObservingPatch,
    elevation: float,
    rising: bool,
    start_time: Time,
    coords: Coordinates,
    cache: dict,
) -> tuple[Time, Time] | None:
    """Return the next plannable CE crossing pass ``(t_open, t_close)``, or None.

    Feasibility mirrors the constant-elevation planner exactly: the pass is
    plannable from ``start_time`` iff the planner's own crossing solver
    (:func:`~fyst_trajectories.planning._ce_geometry._compute_ce_duration`,
    the one :func:`~fyst_trajectories.planning.plan_constant_el_scan` runs at
    reconstruction) finds both RA-edge crossings forward of it. Once the
    leading edge is above the target elevation, the forward search cannot
    find its crossing again within the planner's 12-hour horizon, so a block
    anchored there is unreconstructable - the condition this gate exists to
    prevent (2026-07-15 repro).

    Solves are memoized in ``cache`` per ``(patch.name, elevation,
    rising)``: a hit is trusted while the anchor precedes the pass opening
    by :data:`_CE_ANCHOR_MARGIN_SEC`; a miss is re-solved at most every
    :data:`_CE_MISS_RESOLVE_SEC` of scheduler time. Elevation is part of
    the key because a patch without a pinned ``elevation`` is gated at its
    instantaneous fallback elevation, which varies between calls; patch
    names are assumed unique (an ``ObservingPatch`` documented
    precondition).

    Parameters
    ----------
    patch : ObservingPatch
        Constant-elevation patch supplying the field geometry.
    elevation : float
        Target scan elevation in degrees.
    rising : bool
        Which crossing half to solve for.
    start_time : Time
        Prospective anchor (the planner searches forward from here).
    coords : Coordinates
        Site coordinate transformer.
    cache : dict
        The per-run memo (``SchedulerContext.ce_corridors``).

    Returns
    -------
    tuple of (Time, Time) or None
        ``(t_open, t_close)`` of the pass (first and last RA-edge crossing),
        or None when no pass is plannable from ``start_time``.
    """
    key = (patch.name, float(elevation), bool(rising))
    hit = cache.get(key)
    if hit is not None:
        if hit[0] == "ok":
            _, t_open, t_close = hit
            if start_time.unix <= t_open.unix - _CE_ANCHOR_MARGIN_SEC:
                return t_open, t_close
            # The anchor has reached the pass opening; fall through and
            # re-solve (typically a miss until the other half's pass).
        else:  # ("miss", solved_from)
            if (start_time - hit[1]).sec < _CE_MISS_RESOLVE_SEC:
                return None

    field = FieldRegion(
        ra_center=patch.ra_center,
        dec_center=patch.dec_center,
        width=patch.width,
        height=patch.height,
    )
    try:
        t_open, t_close, _ = _compute_ce_duration(field, 0.0, elevation, coords, start_time, rising)
    except ValueError:
        cache[key] = ("miss", start_time)
        return None
    cache[key] = ("ok", t_open, t_close)
    return t_open, t_close


def _ce_visit_plan(
    patch: ObservingPatch,
    elevation: float,
    start_time: Time,
    end_time: Time,
    coords: Coordinates,
    cache: dict,
    ready_lead: float,
) -> tuple[bool, Time, Time] | None:
    """Choose the crossing half and pass for a CE visit starting at ``start_time``.

    An explicit ``scan_params["rising"]`` request pins the half (None is
    returned when that half has no plannable pass). Without a request both
    halves are tried and the earlier-opening plannable pass wins, so a patch
    whose rising pass has already begun falls over to its setting pass
    instead of being emitted unreconstructable.

    Two window conditions apply on top of plannability:

    - readiness: the pass must open within ``ready_lead`` seconds of
      ``start_time``. A CE drift scan only observes the field while its
      edges cross the scan elevation, so starting the visit hours early
      would book science blocks that point at empty sky (and, before
      2026-07-16, inflated the science accounting by the full wait).
      The patch simply stays unselected (idle, cals, or other patches)
      until the pass is imminent.
    - the opening crossing must precede ``end_time``: a pass that opens
      after the schedule window can never start inside it.

    Returns
    -------
    tuple of (bool, Time, Time) or None
        ``(rising, t_open, t_close)`` for the chosen pass, or None when
        neither half has a plannable pass that is ready and opens within
        the window.
    """
    requested = patch.scan_params.get("rising")
    halves = (bool(requested),) if requested is not None else (True, False)
    best: tuple[bool, Time, Time] | None = None
    for rising in halves:
        window = _ce_crossing_corridor(patch, elevation, rising, start_time, coords, cache)
        if window is None:
            continue
        t_open, t_close = window
        if t_open.unix > end_time.unix:
            continue
        if t_open.unix - start_time.unix > ready_lead:
            continue  # plannable, but the pass is not imminent yet
        if best is None or t_open.unix < best[1].unix:
            best = (rising, t_open, t_close)
    return best


def _time_until_set(
    ra: float,
    dec: float,
    start_time: Time,
    max_duration: float,
    coords: Coordinates,
    min_elevation: float,
    step_seconds: float = 300.0,
) -> float:
    """Compute how long a source stays above *min_elevation* from *start_time*.

    Samples elevation at ``step_seconds`` intervals up to *max_duration*
    using a single vectorised ``radec_to_altaz`` call, then linearly
    interpolates to find the crossing time.

    Returns *max_duration* if the source never drops below the limit
    (circumpolar or long transit).
    """
    n_steps = max(2, int(max_duration / step_seconds) + 1)
    dt = np.linspace(0.0, max_duration, n_steps)
    times = start_time + TimeDelta(dt, format="sec")

    _, el_arr = coords.radec_to_altaz(
        np.full(n_steps, ra),
        np.full(n_steps, dec),
        times,
    )

    below = np.where(el_arr < min_elevation)[0]
    if len(below) == 0:
        return max_duration

    idx = below[0]
    if idx == 0:
        return 0.0

    # Linear interpolation for sub-step precision at the crossing.
    el_prev = float(el_arr[idx - 1])
    el_curr = float(el_arr[idx])
    denom = el_prev - el_curr
    if abs(denom) < 1e-12:
        frac = 0.5
    else:
        frac = (el_prev - min_elevation) / denom
    return float(dt[idx - 1] + frac * (dt[idx] - dt[idx - 1]))


def _time_until_sun_unsafe(
    ra: float,
    dec: float,
    start_time: Time,
    max_duration: float,
    coords: Coordinates,
    min_sun_angle: float,
    step_seconds: float = 60.0,
) -> float:
    """Compute how long a source stays sun-safe from *start_time*.

    Samples target-Sun angular separation at ``step_seconds`` intervals up
    to *max_duration*, then linearly interpolates to the crossing where the
    separation first drops below *min_sun_angle*. Mirrors
    :func:`_time_until_set` (which checks elevation) so the pong/daisy
    duration clip can trim a scan that drifts into the exclusion radius
    mid-scan, the same sun-safety guarantee the constant_el branch gets
    from :func:`get_observable_windows`.

    Returns *max_duration* if the source never enters the exclusion zone.
    """
    n_steps = max(2, int(max_duration / step_seconds) + 1)
    dt = np.linspace(0.0, max_duration, n_steps)
    times = start_time + TimeDelta(dt, format="sec")

    az_arr, el_arr = coords.radec_to_altaz(
        np.full(n_steps, ra),
        np.full(n_steps, dec),
        times,
    )
    sun_az_arr, sun_el_arr = coords.get_sun_altaz(times)
    sep = coords.angular_separation(az_arr, el_arr, sun_az_arr, sun_el_arr)

    unsafe = np.where(sep < min_sun_angle)[0]
    if len(unsafe) == 0:
        return max_duration

    idx = unsafe[0]
    if idx == 0:
        return 0.0

    # Linear interpolation for sub-step precision at the crossing.
    sep_prev = float(sep[idx - 1])
    sep_curr = float(sep[idx])
    denom = sep_prev - sep_curr
    if abs(denom) < 1e-12:
        frac = 0.5
    else:
        frac = (sep_prev - min_sun_angle) / denom
    return float(dt[idx - 1] + frac * (dt[idx] - dt[idx - 1]))


def _compute_scan_duration(
    patch: ObservingPatch,
    start_time: Time,
    end_time: Time,
    site: Site,
    coords: Coordinates,
    overhead: OverheadModel,
    center_el: float = 50.0,
    ce_cache: dict | None = None,
    ce_ready_lead: float = 480.0,
) -> float:
    """Compute how long we can observe this patch.

    For constant-elevation scans, the visit runs until the chosen crossing
    pass closes (:func:`_ce_visit_plan`): a CE drift scan is only plannable
    while its RA-edge crossings lie ahead of the anchor, so the old
    field-center-above-elevation window (which stays open long after the
    pass, all the way to transit and beyond) over-emitted blocks the
    planner could never reconstruct. For pong/daisy, we start with the
    max scan duration (or remaining schedule time) and then clip to the
    observability window so the source never drops below the telescope
    elevation limit mid-scan.

    Parameters
    ----------
    center_el : float
        Computed elevation of the patch center at the current time.
        Used as fallback when patch.elevation is None.
    ce_cache : dict, optional
        Per-run corridor memo (``SchedulerContext.ce_corridors``) for the
        constant-elevation branch. Default None uses a throwaway dict.
    ce_ready_lead : float, optional
        Readiness lead (seconds) passed to :func:`_ce_visit_plan`.
        Default 480.0 for direct/helper callers; the scheduler always
        passes ``ctx.time_step + _CE_READY_SLEW_ALLOWANCE_SEC``.
    """
    remaining = (end_time - start_time).sec

    if patch.scan_type == "constant_el":
        el = patch.elevation if patch.elevation is not None else center_el
        plan = _ce_visit_plan(
            patch,
            el,
            start_time,
            end_time,
            coords,
            {} if ce_cache is None else ce_cache,
            ce_ready_lead,
        )
        if plan is None:
            return 0.0
        _, _, t_close = plan
        corridor_dur = min((t_close - start_time).sec, remaining)
        # Preserve the sun-safety clip the old window check provided: trim
        # the visit before the field center drifts inside the exclusion
        # radius (the planner re-validates the trajectory at rebuild).
        if site.sun_avoidance.enabled:
            sun_safe_dur = _time_until_sun_unsafe(
                patch.ra_center,
                patch.dec_center,
                start_time,
                corridor_dur,
                coords,
                site.sun_avoidance.exclusion_radius,
            )
            corridor_dur = min(corridor_dur, sun_safe_dur)
        return corridor_dur
    else:
        max_dur = min(overhead.max_scan_duration, remaining)
        el_min = site.telescope_limits.elevation.min
        observable_dur = _time_until_set(
            patch.ra_center,
            patch.dec_center,
            start_time,
            max_dur,
            coords,
            el_min,
        )
        # Clip to the sun-safe sub-window too, mirroring the constant_el branch
        # (which gets this from get_observable_windows). Without it a pong/daisy
        # scan that is sun-safe at start but drifts into the exclusion radius
        # mid-scan would not be trimmed.
        if site.sun_avoidance.enabled:
            sun_safe_dur = _time_until_sun_unsafe(
                patch.ra_center,
                patch.dec_center,
                start_time,
                max_dur,
                coords,
                site.sun_avoidance.exclusion_radius,
            )
            observable_dur = min(observable_dur, sun_safe_dur)
        return min(max_dur, observable_dur)


def _compute_az_range(
    patch: ObservingPatch, center_az: float, center_el: float, site: Site
) -> tuple[float, float]:
    """Compute azimuth range for a scan.

    Uses explicit overrides from scan_params if provided, otherwise
    estimates from the field width and elevation. Both endpoints are
    normalized into the site's cable-wrap window so downstream slew-time
    and boresight math operate in a single consistent azimuth frame.

    Parameters
    ----------
    patch : ObservingPatch
        The observing patch.
    center_az : float
        Center azimuth in degrees.
    center_el : float
        Center elevation in degrees (used when patch.elevation is None).
    site : Site
        Site providing the azimuth limits for normalization.
    """
    params = patch.scan_params

    if "az_min" in params and "az_max" in params:
        return _normalize_az(float(params["az_min"]), site), _normalize_az(
            float(params["az_max"]), site
        )

    elevation = patch.elevation if patch.elevation is not None else center_el
    el_rad = math.radians(elevation)
    cos_el = max(math.cos(el_rad), 0.1)
    half_throw = patch.width / (2.0 * cos_el)

    return (
        _normalize_az(center_az - half_throw, site),
        _normalize_az(center_az + half_throw, site),
    )
