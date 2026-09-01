"""Pure helpers used by scheduler phases.

Dependency-free (no scheduler state); consumed by
:mod:`.phases` and :mod:`.state`.
"""

import math
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from ...dispatch import SunSafePredicate

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


def _normalize_az(az: float, site: Site, ref: float | None = None) -> float:
    """Normalize a scalar azimuth into the site's cable-wrap window.

    Wraps :func:`fyst_trajectories.patterns.utils.normalize_azimuth` (which
    operates on arrays) for the scalar azimuths the scheduler carries.
    Raw astropy azimuths are in ``[0, 360)``; the slew-time and boresight
    math must compare them in the telescope's ``[az_min, az_max]`` window
    or a north-straddling pair inflates the slew distance ~17x and flips
    the boresight ~180 deg.

    With ``ref`` given, the in-limits 360-degree representative nearest
    ``ref`` is returned, so the scheduler models the mount's direct
    cable-wrap move from its current position. Without ``ref`` each
    scalar independently takes the representative nearest the window
    centre, which relocates the wrap seam rather than removing it; pass
    ``ref`` whenever a coherent frame with another azimuth is required.

    Parameters
    ----------
    az : float
        Azimuth in degrees (typically raw astropy ``[0, 360)``).
    site : Site
        Site providing the azimuth limits.
    ref : float or None, optional
        Reference azimuth (already in the cable-wrap window) selecting
        among the in-limits representatives. Default None.

    Returns
    -------
    float
        Azimuth shifted into ``[az_min, az_max]``.
    """
    base = float(normalize_azimuth(np.array([az], dtype=float), site)[0])
    if ref is None:
        return base
    limits = site.telescope_limits.azimuth
    best = base
    for cand in (base - 360.0, base + 360.0):
        if limits.min <= cand <= limits.max and abs(cand - ref) < abs(best - ref):
            best = cand
    return best


def _default_constraints(
    site: Site, sun_safe: "SunSafePredicate | None" = None
) -> list[Constraint]:
    """Create default constraints from site configuration.

    With ``sun_safe`` (a :class:`~fyst_trajectories.dispatch.SunSafePredicate`,
    e.g. from :func:`~fyst_trajectories.sun_models.make_sun_safe`) the Sun
    constraint runs that model; otherwise it runs the site's scalar
    exclusion radius. Either way it is only added when the site has Sun
    avoidance enabled.
    """
    constraints: list[Constraint] = [
        ElevationConstraint(
            el_min=site.telescope_limits.elevation.min,
            el_max=site.telescope_limits.elevation.max,
        ),
    ]
    if site.sun_avoidance.enabled:
        if sun_safe is not None:
            constraints.append(SunAvoidanceConstraint(sun_safe=sun_safe))
        else:
            constraints.append(
                SunAvoidanceConstraint(min_angle=site.sun_avoidance.exclusion_radius)
            )
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
    prevent.

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
      would book science blocks that point at empty sky and inflate the
      science accounting by the full wait. The patch simply stays
      unselected (idle, cals, or other patches) until the pass is
      imminent.
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
        # The "is None or" short circuit guarantees best is a tuple on the
        # right-hand side; pylint's inference cannot narrow the union here.
        if best is None or t_open.unix < best[1].unix:  # pylint: disable=unsubscriptable-object
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
    sun_safe: "SunSafePredicate | None" = None,
) -> float:
    """Compute how long a source stays sun-safe from *start_time*.

    Samples the track at ``step_seconds`` intervals up to *max_duration*
    and locates where it first stops being sun-safe. In scalar mode
    (``sun_safe=None``) safety is separation strictly greater than
    *min_sun_angle* (``<=`` at the boundary is unsafe, matching
    ``is_sun_safe``) and the crossing is linearly interpolated on the
    separation. With an injected
    :class:`~fyst_trajectories.dispatch.SunSafePredicate` the model's
    verdicts decide, and the crossing is refined by VERDICT BISECTION
    inside the bracketing samples, returning the last verified-safe time
    (conservative by construction). Bisection is used instead of margin
    interpolation because a directional model's threshold is a staircase
    over discrete table levels: a level step inside the bracket makes an
    interpolated crossing anti-conservative by up to a full step.
    Mirrors :func:`_time_until_set` (which checks elevation) so the
    pong/daisy duration clip can trim a scan that drifts into the
    exclusion zone mid-scan; the constant_el branch applies this same
    clip after its corridor solve (:func:`_ce_visit_plan`).

    Returns *max_duration* if the source never becomes unsafe.
    """
    n_steps = max(2, int(max_duration / step_seconds) + 1)
    dt = np.linspace(0.0, max_duration, n_steps)
    times = start_time + TimeDelta(dt, format="sec")

    az_arr, el_arr = coords.radec_to_altaz(
        np.full(n_steps, ra),
        np.full(n_steps, dec),
        times,
    )

    if sun_safe is not None:
        has_batch = hasattr(sun_safe, "batch")
        if has_batch:
            safe = np.atleast_1d(np.asarray(sun_safe.batch(az_arr, el_arr, times), dtype=bool))
            if safe.shape != (n_steps,):
                # A scalar/short result would silently broadcast one verdict
                # over the whole grid; fail loudly instead.
                raise ValueError(
                    f"sun_safe.batch returned shape {safe.shape}, expected "
                    f"({n_steps},) verdicts for the duration grid"
                )
        else:
            safe = np.array(
                [
                    bool(sun_safe(float(az_arr[i]), float(el_arr[i]), times[i]))
                    for i in range(n_steps)
                ],
                dtype=bool,
            )
        unsafe = np.flatnonzero(~safe)
        if unsafe.size == 0:
            return max_duration
        idx = int(unsafe[0])
        if idx == 0:
            return 0.0

        def _verdict_at(offset_s: float) -> bool:
            t_probe = start_time + TimeDelta(offset_s, format="sec")
            az_p, el_p = coords.radec_to_altaz(ra, dec, t_probe)
            if has_batch:
                return bool(np.atleast_1d(sun_safe.batch([float(az_p)], [float(el_p)], t_probe))[0])
            return bool(sun_safe(float(az_p), float(el_p), t_probe))

        # Verdict bisection between the last-safe and first-unsafe samples:
        # exact for any model shape (10 iterations resolve a 60 s step to
        # ~0.06 s) and conservative (returns the last VERIFIED-safe time).
        lo, hi = float(dt[idx - 1]), float(dt[idx])
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            if _verdict_at(mid):
                lo = mid
            else:
                hi = mid
        return lo

    sun_az_arr, sun_el_arr = coords.get_sun_altaz(times)
    sep = np.asarray(coords.angular_separation(az_arr, el_arr, sun_az_arr, sun_el_arr), dtype=float)

    # `<=`: a sample exactly at the radius is unsafe, matching is_sun_safe.
    unsafe = np.flatnonzero(sep <= min_sun_angle)
    if unsafe.size == 0:
        return max_duration

    idx = int(unsafe[0])
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
    sun_safe: "SunSafePredicate | None" = None,
) -> float:
    """Compute how long we can observe this patch.

    For constant-elevation scans, the visit runs until the chosen crossing
    pass closes (:func:`_ce_visit_plan`): a CE drift scan is only plannable
    while its RA-edge crossings lie ahead of the anchor. A plain
    field-center-above-elevation window is not a substitute; it stays open
    long after the pass, all the way to transit and beyond, and emits
    blocks the planner could never reconstruct. For pong/daisy, we start with the
    max scan duration (or remaining schedule time) and then clip to the
    observability window so the source never drops below the telescope
    elevation limit mid-scan.

    Parameters
    ----------
    patch : ObservingPatch
        The candidate patch.
    start_time : Time
        Scan start under consideration.
    end_time : Time
        End of the schedule window; the duration never extends past it.
    site : Site
        Telescope site configuration.
    coords : Coordinates
        Coordinate transform bound to ``site``.
    overhead : OverheadModel
        Supplies ``max_scan_duration`` for the pong/daisy branch.
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
    sun_safe : SunSafePredicate, optional
        Injected sun-safety model for the mid-scan drift clip. Default
        ``None`` keeps the site's scalar exclusion radius.

    Returns
    -------
    float
        Observable duration in seconds; ``0.0`` when no pass is
        plannable from ``start_time``.
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
        # Sun-safety clip on top of the corridor: trim the visit before the
        # field center drifts inside the exclusion radius (the planner
        # re-validates the trajectory at rebuild).
        if site.sun_avoidance.enabled:
            sun_safe_dur = _time_until_sun_unsafe(
                patch.ra_center,
                patch.dec_center,
                start_time,
                corridor_dur,
                coords,
                site.sun_avoidance.exclusion_radius,
                sun_safe=sun_safe,
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
        # Clip to the sun-safe sub-window too, mirroring the constant_el
        # branch's post-corridor clip. Without it a pong/daisy scan that is
        # sun-safe at start but drifts into the exclusion radius mid-scan
        # would not be trimmed.
        if site.sun_avoidance.enabled:
            sun_safe_dur = _time_until_sun_unsafe(
                patch.ra_center,
                patch.dec_center,
                start_time,
                max_dur,
                coords,
                site.sun_avoidance.exclusion_radius,
                sun_safe=sun_safe,
            )
            observable_dur = min(observable_dur, sun_safe_dur)
        return min(max_dur, observable_dur)


def _compute_az_range(
    patch: ObservingPatch, center_az: float, center_el: float, site: Site
) -> tuple[float, float]:
    """Compute azimuth range for a scan.

    Uses explicit overrides from scan_params if provided, otherwise
    estimates from the field width and elevation. The endpoints are
    normalized **jointly**: the pair is placed as one contiguous range
    in the site's cable-wrap window (a per-endpoint normalization would
    tear a range straddling the window seam into an unordered pair).
    When the range around ``center_az`` pokes past an azimuth limit,
    both endpoints shift together by 360 degrees onto the in-limits
    branch, matching how the planner places a constant-elevation range.

    Parameters
    ----------
    patch : ObservingPatch
        The observing patch.
    center_az : float
        Center azimuth in degrees, already normalized into the
        cable-wrap window.
    center_el : float
        Center elevation in degrees (used when patch.elevation is None).
    site : Site
        Site providing the azimuth limits for normalization.

    Returns
    -------
    tuple of float
        ``(az_start, az_end)`` in degrees with ``az_start <= az_end``.
        An explicit ``(az_min, az_max)`` pair is read as the ascending
        modular range from ``az_min``, so a pair inverted by noise
        reads as a near-full-circle sweep. Placement inside the limits
        is only possible when the range fits: a range wider than the
        cable-wrap window, or one overhanging both ends, is returned
        unshifted and may exceed the limits.
    """
    params = patch.scan_params
    limits = site.telescope_limits.azimuth

    if "az_min" in params and "az_max" in params:
        lo = _normalize_az(float(params["az_min"]), site)
        # Ascending representative of az_max from lo, so lo <= hi always.
        hi = lo + (float(params["az_max"]) - lo) % 360.0
    else:
        elevation = patch.elevation if patch.elevation is not None else center_el
        el_rad = math.radians(elevation)
        cos_el = max(math.cos(el_rad), 0.1)
        half_throw = patch.width / (2.0 * cos_el)
        lo = center_az - half_throw
        hi = center_az + half_throw

    if hi > limits.max and lo - 360.0 >= limits.min:
        lo -= 360.0
        hi -= 360.0
    elif lo < limits.min and hi + 360.0 <= limits.max:
        lo += 360.0
        hi += 360.0
    return lo, hi
