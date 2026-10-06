"""Pure helpers used by scheduler phases.

Dependency-free (no scheduler state); consumed by
:mod:`.phases` and :mod:`.state`.
"""

import math
from typing import TYPE_CHECKING

import numpy as np
from astropy.time import Time, TimeDelta

from ...coordinates import Coordinates
from ...exceptions import PointingError
from ...patterns import PongScanConfig, compute_pong_period

# Private pattern reuse, on purpose: a pong or daisy visit is sized by where
# the pattern the rebuild runs actually goes, so the bounds come from the
# pattern modules that generate it rather than from a copy of their geometry.
from ...patterns.daisy import _daisy_reach
from ...patterns.pong import _pong_peak_offsets
from ...patterns.turnarounds import swept_az_envelope
from ...patterns.utils import sky_offsets_to_altaz
from ...planning import FieldRegion

# Private planner reuse, on purpose: the scheduler must gate constant-elevation
# emission on exactly the crossing solve that plan_constant_el_scan runs at
# reconstruction (planning = execution), so it calls the planner's own solver
# rather than approximating it. The corridor helper is reused for the same
# reason: it is where the swept azimuth range comes from at reconstruction.
from ...planning._ce_geometry import _compute_ce_az_range, _compute_ce_duration
from ...site import Site
from ...sun_protocols import _sun_verdicts
from ..constraints import Constraint, ElevationConstraint, SunAvoidanceConstraint
from ..models import ObservingPatch, OverheadModel
from ..simulation import _DAISY_REBUILD_DEFAULTS, _PONG_REBUILD_DEFAULTS
from ..utils import _normalize_az

if TYPE_CHECKING:
    from ...sun_protocols import SunSafePredicate

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

#: Largest azimuth spacing (deg) between the columns an injected Sun model is
#: asked about when the Sun clip covers an azimuth span. The scalar radius
#: needs no grid (its nearest point is found analytically); a model whose zone
#: is not a cone is sampled on this grid, edges and centre always included.
_SPAN_AZ_STEP_DEG = 2.0


def _default_constraints(
    site: Site, sun_safe: "SunSafePredicate | None" = None
) -> list[Constraint]:
    """Create default constraints from site configuration.

    With ``sun_safe`` (a :class:`~fyst_trajectories.sun_protocols.SunSafePredicate`,
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
    the key because the solve depends on it and the caller passes it apart
    from the patch. In a context from ``SchedulerContext.build``, which
    refuses a constant-elevation patch without a pinned ``elevation``,
    every call for a patch passes that elevation; a hand-built context
    (whose phases gate an unpinned patch at the field centre's elevation
    at selection) or a direct call can solve one name at other elevations.
    Patch names are unique within a schedule, which
    ``SchedulerContext.build`` rejects a duplicate of.

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
    except PointingError:
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


def _ce_swept_az_envelope(
    patch: ObservingPatch,
    t_open: Time,
    t_close: Time,
    coords: Coordinates,
) -> tuple[float, float]:
    """Return the azimuth envelope a constant-elevation pass sweeps.

    The planner's own corridor for the crossing ``[t_open, t_close]`` (the
    azimuth extent of the field's corners over the whole pass, plus
    padding), widened by one turnaround overshoot per side. That is the
    range the mount occupies, and it is what the rebuilt trajectory
    sweeps, so a phase that has to reason about where the telescope goes
    before any trajectory exists can ask for it here.

    It is much wider than the instantaneous field width
    :func:`_compute_az_range` estimates, because the field drifts across
    the corridor for the whole pass; on an eight-hour single-patch night
    the two differ by tens of degrees, not by the fraction of a degree a
    turnaround overshoot adds.

    The ``az_padding`` and ``az_accel`` defaults read here are the ones
    the reconstruction path applies, so this estimate and the rebuild
    describe one geometry.

    Parameters
    ----------
    patch : ObservingPatch
        Constant-elevation patch supplying the field geometry, the scan
        velocity, and the optional ``az_padding`` / ``az_accel``
        overrides.
    t_open, t_close : Time
        First and last RA-edge crossing of the pass, as returned by
        :func:`_ce_visit_plan`.
    coords : Coordinates
        Site coordinate transformer.

    Returns
    -------
    tuple of float
        ``(env_min, env_max)`` in degrees, ordered. The pair may lie
        outside ``[0, 360)`` when the pass straddles north, the
        unwrapped representation the planner also returns.
    """
    params = patch.scan_params
    field = FieldRegion(
        ra_center=patch.ra_center,
        dec_center=patch.dec_center,
        width=patch.width,
        height=patch.height,
    )
    az_lo, az_hi = _compute_ce_az_range(
        field, 0.0, coords, t_open, t_close, float(params.get("az_padding", 2.0))
    )
    return swept_az_envelope(az_lo, az_hi, patch.velocity, float(params.get("az_accel", 1.0)))


def _first_crossing(
    dt: np.ndarray, values: np.ndarray, threshold: float, bad: np.ndarray, max_duration: float
) -> float:
    """Return when ``values`` first crosses ``threshold`` into the ``bad`` samples.

    Linearly interpolated between the last good and the first bad sample;
    ``max_duration`` when no sample is bad and ``0.0`` when the first one is.
    """
    hits = np.flatnonzero(bad)
    if hits.size == 0:
        return max_duration
    idx = int(hits[0])
    if idx == 0:
        return 0.0
    prev = float(values[idx - 1])
    curr = float(values[idx])
    denom = prev - curr
    frac = 0.5 if abs(denom) < 1e-12 else (prev - threshold) / denom
    return float(dt[idx - 1] + frac * (dt[idx] - dt[idx - 1]))


def _pattern_el_bounds(
    patch: ObservingPatch, times: Time, coords: Coordinates
) -> tuple[np.ndarray, np.ndarray]:
    """Return the lowest and highest elevation a patch's pattern reaches at each time.

    The pattern is the pong or daisy that
    :func:`~fyst_trajectories.overhead.schedule_to_trajectories` rebuilds a
    block of the patch with, tracking the field centre. A daisy stays
    within :func:`~fyst_trajectories.patterns.daisy._daisy_reach` of the
    centre, which is more than its ``radius`` (by at least
    ``turn_radius``), so its elevation stays within that distance of the
    centre's. A pong stays inside its box
    (:func:`~fyst_trajectories.patterns.pong._pong_peak_offsets`, turned by
    the pattern's ``angle``), and the lowest point of the box is one of its
    four corners. No interior point can be lowest, since above the horizon
    elevation has no local minimum, and along each edge the elevation can
    turn only where the edge runs level, and there it peaks: well above the
    horizon an edge of the box curves less than the circle of constant
    elevation it touches. The corners are placed on the sky and transformed
    exactly as the planner places the pattern. The highest point can lie
    along an edge, so it is bounded by the circle about the centre through
    the corners. The bounds hold whatever the pattern's phase, which is not
    known before the slew; a pong passes near a given corner only once a
    period, so its bound can sit up to a period's drift below the
    trajectory it stands for.

    Parameters
    ----------
    patch : ObservingPatch
        A pong or daisy patch.
    times : Time
        One-dimensional array of times.
    coords : Coordinates
        Coordinate transform bound to the site.

    Returns
    -------
    tuple of numpy.ndarray
        ``(lowest, highest)`` elevation in degrees at each time.
    """
    n_t = len(times)
    if patch.scan_type == "pong":
        config = _pong_config(patch)
        half_x, half_y = _pong_peak_offsets(config)
        reach = math.hypot(half_x, half_y)
        angle = math.radians(config.angle)
        corner_x = half_x * np.array([1.0, 1.0, -1.0, -1.0])
        corner_y = half_y * np.array([1.0, -1.0, 1.0, -1.0])
        # The centre, then the corners, turned as the pattern turns its offsets.
        x = np.concatenate(([0.0], corner_x * math.cos(angle) - corner_y * math.sin(angle)))
        y = np.concatenate(([0.0], corner_x * math.sin(angle) + corner_y * math.cos(angle)))
        _, el = sky_offsets_to_altaz(
            np.tile(x, n_t),
            np.tile(y, n_t),
            patch.ra_center,
            patch.dec_center,
            times[np.repeat(np.arange(n_t), x.size)],
            coords,
        )
        el = np.asarray(el, dtype=float).reshape(n_t, x.size)
        centre = el[:, 0]
        lowest = el[:, 1:].min(axis=1)
    else:
        settings = {**_DAISY_REBUILD_DEFAULTS, **patch.scan_params}
        reach = _daisy_reach(
            settings["radius"], settings["turn_radius"], patch.velocity, settings["timestep"]
        )
        _, centre = coords.radec_to_altaz(
            np.full(n_t, patch.ra_center), np.full(n_t, patch.dec_center), times
        )
        centre = np.asarray(centre, dtype=float)
        lowest = centre - reach
    return lowest, np.minimum(centre + reach, 90.0)


def _time_inside_el_limits(
    patch: ObservingPatch,
    start_time: Time,
    max_duration: float,
    coords: Coordinates,
    lead: float = 0.0,
    step_seconds: float = 300.0,
) -> float:
    """Compute how long a pong or daisy pattern stays inside the elevation limits.

    The whole pattern (:func:`_pattern_el_bounds`), not only the field
    centre, must stay within the site's elevation limits, outside which the
    planner refuses a trajectory, from ``start_time + lead`` on:
    ``lead`` is the boundary retune booked before the subscan, during which
    the telescope holds the field centre and no pattern runs. The bounds
    are sampled every ``step_seconds`` from then up to *max_duration* in
    one vectorised transform, and the first sample outside is refined by
    bisection between it and the last sample inside, returning the last
    time verified inside (conservative by construction, to within 0.3 s
    for the default step).

    Returns the window from *start_time*, ``lead`` included: *max_duration*
    when the pattern stays inside throughout, and ``0.0`` when it is not
    inside at ``start_time + lead`` (a field whose pattern edge has not yet
    risen above the lower limit, or has already set) or ``lead`` leaves no
    time.
    """
    limits = coords.site.telescope_limits.elevation
    if lead >= max_duration:
        return 0.0

    def inside(offsets: np.ndarray) -> np.ndarray:
        lowest, highest = _pattern_el_bounds(
            patch, start_time + TimeDelta(offsets, format="sec"), coords
        )
        return (lowest >= limits.min) & (highest <= limits.max)

    n_steps = max(2, int((max_duration - lead) / step_seconds) + 1)
    dt = np.linspace(lead, max_duration, n_steps)
    outside = np.flatnonzero(~inside(dt))
    if outside.size == 0:
        return max_duration
    idx = int(outside[0])
    if idx == 0:
        return 0.0
    # Bisection between the last sample inside and the first outside (ten
    # steps resolve a 300 s step to 0.3 s), returning the last time verified
    # inside: an interpolated crossing could land a sample past the limit.
    lo, hi = float(dt[idx - 1]), float(dt[idx])
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        if bool(inside(np.array([mid]))[0]):
            lo = mid
        else:
            hi = mid
    return lo


def _nearest_az_in_span(az: np.ndarray, az_span: tuple[float, float]) -> np.ndarray:
    """Return the azimuth of ``az_span`` circularly nearest each of ``az``.

    ``az`` itself where it lies inside the span (modulo 360), otherwise the
    nearer edge. A span 360 deg or wider contains every azimuth.
    """
    lo, hi = az_span
    width = hi - lo
    az = np.asarray(az, dtype=float)
    if width >= 360.0:
        return az
    past_lo = np.mod(az - lo, 360.0)
    past_hi = past_lo - width  # > 0 outside the span: how far beyond hi
    before_lo = 360.0 - past_lo  # how far short of lo, going the other way
    edge = np.where(past_hi <= before_lo, hi, lo)
    return np.where(past_lo <= width, az, edge)


def _span_columns(az_span: tuple[float, float]) -> np.ndarray:
    """Return the azimuth columns an injected Sun model is asked about.

    An odd count, so both edges and the centre are always columns, spaced
    at most :data:`_SPAN_AZ_STEP_DEG` apart and wrapped into ``[0, 360)``.
    """
    lo, hi = az_span
    width = min(hi - lo, 360.0)
    n_az = max(3, 2 * math.ceil(width / (2.0 * _SPAN_AZ_STEP_DEG)) + 1)
    return np.mod(lo + np.linspace(0.0, width, n_az), 360.0)


def _span_verdicts(
    sun_safe: "SunSafePredicate",
    columns: np.ndarray,
    el: np.ndarray,
    times: Time,
) -> np.ndarray:
    """Return, per time sample, whether every azimuth column is Sun-safe.

    ``el`` and ``times`` are same-length 1-D; the model is queried once on
    the flattened ``(time, column)`` grid.
    """
    n_t, n_az = el.size, columns.size
    az_flat = np.tile(columns, n_t)
    el_flat = np.repeat(el, n_az)
    t_flat = times[np.repeat(np.arange(n_t), n_az)]
    safe = _sun_verdicts(sun_safe, az_flat, el_flat, t_flat, what="azimuth-span grid")
    return safe.reshape(n_t, n_az).all(axis=1)


def _time_until_sun_unsafe(
    ra: float,
    dec: float,
    start_time: Time,
    max_duration: float,
    coords: Coordinates,
    min_sun_angle: float,
    step_seconds: float = 60.0,
    sun_safe: "SunSafePredicate | None" = None,
    fixed_el: float | None = None,
    *,
    az_span: tuple[float, float] | None = None,
) -> float:
    """Compute how long a source stays sun-safe from *start_time*.

    With ``fixed_el`` the track is followed in azimuth but held at that
    elevation, which is the pose a patch with a pinned ``elevation``
    actually commands: the selection phase scores it there, the slew
    drives there and every emitted block records it. Without it the
    field's own instantaneous elevation is used.

    With ``az_span`` the azimuth is not the field's but every azimuth of
    the span ``(az_lo, az_hi)``, ``az_lo <= az_hi`` in degrees (either may
    lie outside ``[0, 360)``; the span is taken modulo 360), and a sample
    is unsafe when any of them is. That is the question a
    constant-elevation visit poses, since the mount sweeps its whole
    corridor on every leg. In scalar mode the check is exact: at a fixed
    elevation the separation from the Sun,
    ``cos(sep) = sin(el) sin(el_sun) + cos(el) cos(el_sun) cos(az - az_sun)``,
    never decreases as the azimuth moves away from the Sun's (both
    elevation cosines are non-negative; at the zenith every azimuth is one
    point), so its minimum over the span sits at the Sun's azimuth when the
    span contains it and at the nearer edge otherwise. An injected model's
    zone need not be a cone, so it is asked about a grid of azimuth columns
    at most :data:`_SPAN_AZ_STEP_DEG` apart, both edges and the centre
    included.

    Samples the track at ``step_seconds`` intervals up to *max_duration*
    and locates where it first stops being sun-safe. In scalar mode
    (``sun_safe=None``) safety is separation strictly greater than
    *min_sun_angle* (``<=`` at the boundary is unsafe, matching
    ``is_sun_safe``) and the crossing is linearly interpolated on the
    separation. With an injected
    :class:`~fyst_trajectories.sun_protocols.SunSafePredicate` the model's
    verdicts decide, and the crossing is refined by VERDICT BISECTION
    inside the bracketing samples, returning the last verified-safe time
    (conservative by construction). Bisection is used instead of margin
    interpolation because a directional model's threshold is a staircase
    over discrete table levels: a level step inside the bracket makes an
    interpolated crossing anti-conservative by up to a full step.
    Mirrors :func:`_time_inside_el_limits` (which checks elevation) so the
    pong/daisy duration clip can trim a scan that drifts into the
    exclusion zone mid-scan, which happens with a pinned elevation or a
    directional Sun model; the constant_el branch applies this same
    clip after its corridor solve (:func:`_ce_visit_plan`), over the
    swept azimuth envelope (:func:`_ce_swept_az_envelope`) as ``az_span``.

    Returns *max_duration* if the source never becomes unsafe.
    """
    if az_span is not None:
        az_lo, az_hi = (float(a) for a in az_span)
        if not (math.isfinite(az_lo) and math.isfinite(az_hi) and az_lo <= az_hi):
            raise ValueError(f"az_span must be finite and ordered (lo <= hi), got {az_span}")
        az_span = (az_lo, az_hi)
    n_steps = max(2, int(max_duration / step_seconds) + 1)
    dt = np.linspace(0.0, max_duration, n_steps)
    times = start_time + TimeDelta(dt, format="sec")

    az_arr, el_arr = coords.radec_to_altaz(
        np.full(n_steps, ra),
        np.full(n_steps, dec),
        times,
    )
    if fixed_el is not None:
        el_arr = np.full(n_steps, float(fixed_el))

    if sun_safe is not None:
        columns = None if az_span is None else _span_columns(az_span)
        if columns is not None:
            safe = _span_verdicts(sun_safe, columns, el_arr, times)
        else:
            safe = _sun_verdicts(sun_safe, az_arr, el_arr, times, what="duration grid")
        unsafe = np.flatnonzero(~safe)
        if unsafe.size == 0:
            return max_duration
        idx = int(unsafe[0])
        if idx == 0:
            return 0.0

        def _verdict_at(offset_s: float) -> bool:
            t_probe = start_time + TimeDelta(offset_s, format="sec")
            az_p, el_p = coords.radec_to_altaz(ra, dec, t_probe)
            if fixed_el is not None:
                el_p = float(fixed_el)
            if columns is not None:
                t_one = start_time + TimeDelta([offset_s], format="sec")
                return bool(_span_verdicts(sun_safe, columns, np.array([el_p]), t_one)[0])
            verdict = _sun_verdicts(
                sun_safe, float(az_p), float(el_p), t_probe, what="duration grid"
            )
            return bool(verdict[0])

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
    if az_span is not None:
        # The span's point nearest the Sun in azimuth is its closest point.
        az_arr = _nearest_az_in_span(sun_az_arr, az_span)
    sep = np.asarray(coords.angular_separation(az_arr, el_arr, sun_az_arr, sun_el_arr), dtype=float)

    # `<=`: a sample exactly at the radius is unsafe, matching is_sun_safe.
    return _first_crossing(dt, sep, min_sun_angle, sep <= min_sun_angle, max_duration)


def _pong_config(patch: ObservingPatch) -> PongScanConfig:
    """Return the pattern a pong patch's blocks are rebuilt with.

    The pattern is the one
    :func:`~fyst_trajectories.overhead.schedule_to_trajectories` rebuilds
    a block of the patch with: the patch's geometry and velocity, and
    each pattern setting from its ``scan_params`` or, when absent, from
    the rebuild's defaults.
    """
    settings = {**_PONG_REBUILD_DEFAULTS, **patch.scan_params}
    return PongScanConfig(
        timestep=settings["timestep"],
        width=patch.width,
        height=patch.height,
        spacing=settings["spacing"],
        velocity=patch.velocity,
        num_terms=settings["num_terms"],
        angle=settings["angle"],
    )


def _pong_period(patch: ObservingPatch) -> float:
    """Return one period of a pong patch's pattern (:func:`_pong_config`), in seconds.

    A pong subscan is a whole number of these periods.
    """
    return compute_pong_period(_pong_config(patch))[0]


def _min_subscan_duration(patch: ObservingPatch, overhead: OverheadModel) -> float:
    """Return the shortest science subscan a patch can be booked for, in seconds.

    ``overhead.min_scan_duration``, or for a pong one pattern period when
    that is longer, since a pong subscan is a whole number of periods.
    """
    if patch.scan_type == "pong":
        return max(overhead.min_scan_duration, _pong_period(patch))
    return overhead.min_scan_duration


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
    *,
    retune_lead: float = 0.0,
) -> float:
    """Compute how long we can observe this patch.

    For constant-elevation scans, the visit runs until the chosen crossing
    pass closes (:func:`_ce_visit_plan`): a CE drift scan is only plannable
    while its RA-edge crossings lie ahead of the anchor. A plain
    field-center-above-elevation window is not a substitute; it stays open
    long after the pass, all the way to transit and beyond, and emits
    blocks the planner could never reconstruct. For pong/daisy, we start with the
    max scan duration (or remaining schedule time) and then clip it to the
    time the whole pattern stays inside the telescope elevation limits
    (:func:`_time_inside_el_limits`), so no part of the scan the planner
    builds leaves them; the window is ``0.0`` while the pattern's edge is
    below the lower limit when its subscan would start, as it is for a
    rising field before the whole pattern has cleared it. A pong visit
    fills this window with whole pattern periods after its boundary retune,
    so the selection phase admits a pong or daisy patch only while the
    window holds its shortest subscan and any retune that is due.

    Both Sun clips ask about the elevation the visit commands: a patch
    that pins ``elevation`` is clipped at that elevation, the one the
    selection phase scored and every emitted block records, while the
    field's own track still decides when its pattern leaves the elevation
    limits. The constant-elevation clip evaluates the whole swept azimuth
    corridor at that elevation (:func:`_ce_swept_az_envelope`, the planner's corridor
    plus the turnaround overshoot), since the mount crosses all of it on
    every leg. The pong and daisy clip follows the field centre.

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
    retune_lead : float, optional
        Pong and daisy only: seconds of boundary retune booked at
        ``start_time`` before the subscan, during which no pattern runs,
        so the pattern has to be inside the elevation limits only from
        ``start_time + retune_lead``. Default 0.0.

    Returns
    -------
    float
        Observable duration in seconds from ``start_time``; ``0.0`` when
        no pass is plannable from ``start_time`` or the pattern is not
        inside the elevation limits when its subscan would start.
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
        _, t_open, t_close = plan
        corridor_dur = min((t_close - start_time).sec, remaining)
        # Sun-safety clip on top of the corridor: trim the visit before any
        # azimuth the mount sweeps at the scan elevation comes inside the
        # exclusion zone. The field centre alone is not enough: every leg
        # crosses the whole corridor, tens of degrees wider than the field.
        if site.sun_avoidance.enabled:
            sun_safe_dur = _time_until_sun_unsafe(
                patch.ra_center,
                patch.dec_center,
                start_time,
                corridor_dur,
                coords,
                site.sun_avoidance.exclusion_radius,
                sun_safe=sun_safe,
                fixed_el=el,
                az_span=_ce_swept_az_envelope(patch, t_open, t_close, coords),
            )
            corridor_dur = min(corridor_dur, sun_safe_dur)
        return corridor_dur
    else:
        max_dur = min(overhead.max_scan_duration, remaining)
        observable_dur = _time_inside_el_limits(
            patch, start_time, max_dur, coords, lead=retune_lead
        )
        # Clip to the sun-safe sub-window too, mirroring the constant_el
        # branch's post-corridor clip. Without it a pong/daisy scan that is
        # sun-safe at start but drifts into the exclusion radius mid-scan
        # (reachable with a pinned elevation or a directional Sun model)
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
                fixed_el=patch.elevation,
            )
            observable_dur = min(observable_dur, sun_safe_dur)
        return min(max_dur, observable_dur)


def _compute_az_range(
    patch: ObservingPatch, center_az: float, center_el: float, site: Site
) -> tuple[float, float]:
    """Estimate the azimuth range a scan occupies at one instant.

    Uses explicit overrides from scan_params if provided, otherwise
    extends the field's projected half width,
    ``patch.width / (2 cos el)``, to each side of ``center_az`` (the
    cosine is floored at 0.1, capping the half throw at five field widths
    near the zenith). That is
    a scalar estimate at the tick time, not the envelope a built
    trajectory sweeps: a drifting constant-elevation pass crosses a
    corridor far wider than the instantaneous field width
    (:func:`_ce_swept_az_envelope` is that corridor). Use it only where
    nothing is built yet, to choose a slew target and to place the scan
    on a cable-wrap branch; an emitted block records the envelope of its
    own trajectory.

    The endpoints are
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
