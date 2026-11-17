"""Pure helpers used by scheduler phases.

Dependency-free (no scheduler state); consumed by
:mod:`.phases` and :mod:`.state`.
"""

import inspect
import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
from astropy.time import Time, TimeDelta

from ...coordinates import Coordinates
from ...exceptions import OffsetInversionError, PointingError, PointingWarning
from ...patterns import PongScanConfig, compute_pong_period

# Private pattern reuse, on purpose: a pong or daisy visit is sized by where
# the pattern the rebuild runs actually goes, so the bounds come from the
# pattern modules that generate it rather than from a copy of their geometry.
from ...patterns.daisy import _daisy_reach
from ...patterns.pong import _pong_peak_offsets
from ...patterns.turnarounds import swept_az_envelope, turnaround_duration_sec
from ...patterns.utils import sky_offsets_to_altaz, validate_sample_count
from ...planning import FieldRegion, plan_constant_el_scan

# Private planner reuse, on purpose: constant-elevation emission is gated on
# the crossing solve plan_constant_el_scan runs at reconstruction (planning =
# execution), the swept range comes from the planner's own range helper, and
# a window scan's legs are counted by the planner's own quantiser.
from ...planning._ce_geometry import (
    _DEFAULT_MAX_SEARCH_HOURS,
    _ce_field_az_range,
    _compute_ce_duration,
    _compute_ce_duration_from_lsa,
    _field_ra_edges,
    _lsa_window_rising_side,
    _quantize_ce_duration,
    _window_side,
)
from ...site import Site
from ...sun_protocols import ZonedSunSafePredicate, _sun_verdicts
from ..constraints import (
    Constraint,
    ElevationConstraint,
    SunAvoidanceConstraint,
    _boresight_pose,
)
from ..models import ObservingPatch, OverheadModel
from ..simulation import _DAISY_REBUILD_DEFAULTS, _PONG_REBUILD_DEFAULTS

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...offsets import InstrumentOffset
    from ...planning import ScanBlock
    from ...sun_protocols import SunSafePredicate

#: The rate a field's hour angle grows, deg/s: one turn per sidereal day
#: (86164.0905 s).
_SIDEREAL_RATE_DEG_S = 360.0 / 86164.0905

#: Cached corridor hits are only trusted for anchors at least this many
#: seconds before the pass's opening crossing: the planner's 30 s-step
#: crossing search needs the leading edge still below the target elevation
#: at the anchor, so boundary anchors re-solve instead of trusting the cache.
_CE_ANCHOR_MARGIN_SEC = 60.0

#: Minimum scheduler time between re-solves after a corridor miss. A miss
#: costs a full 12-hour forward search; without a rate limit an infeasible
#: patch would re-run it every selection tick for the rest of the night.
_CE_MISS_RESOLVE_SEC = 600.0

#: ``plan_constant_el_scan``'s parameters, read for its ``az_padding`` and
#: ``az_accel`` defaults, so the swept-envelope estimate follows the planner
#: the rebuild runs rather than a copy of its defaults.
_CE_PLANNER_PARAMETERS = inspect.signature(plan_constant_el_scan).parameters

#: Slew allowance folded into the CE readiness lead: a visit may start this
#: many seconds, plus one scheduler tick and the retune due when its pass opens,
#: before the pass's opening crossing, leaving room for the pre-scan slew.
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


def _ce_window(patch: ObservingPatch) -> tuple[float, float] | None:
    """Return a constant-elevation patch's azimuth window ``(az_min, az_max)``, or None.

    The window is given by ``scan_params["az_min"]`` and ``["az_max"]``,
    both or neither (:class:`~fyst_trajectories.overhead.ObservingPatch`
    refuses one alone); a visit to such a patch sweeps that window.
    """
    if patch.scan_type != "constant_el" or "az_min" not in patch.scan_params:
        return None
    return float(patch.scan_params["az_min"]), float(patch.scan_params["az_max"])


def _ce_patch_rising(patch: ObservingPatch) -> bool | None:
    """Return the crossing a constant-elevation patch asks for, or None for either.

    ``scan_params["rising"]`` when given; else, for a patch with an azimuth
    window (:func:`_ce_window`), the side of the meridian the window lies
    on, ``True`` east of it, where a field rises through the elevation
    (:func:`~fyst_trajectories.planning._ce_geometry._window_side`; a
    window that reaches a meridian azimuth carries ``rising``, which
    :class:`~fyst_trajectories.overhead.ObservingPatch` checks against the
    window's side); else ``None``, and the visit takes whichever crossing
    opens first. The crossing gate reads a patch's side only here; beside
    an ``lsa_window`` the window sets the side and this is not read.
    """
    requested = patch.scan_params.get("rising")
    if requested is not None:
        return bool(requested)
    window = _ce_window(patch)
    if window is None:
        return None
    return _window_side(*window) == "rising"


def _window_scan_settings(patch: ObservingPatch) -> tuple[float, float, float, float]:
    """Return a window patch's padded window edges, ``az_accel`` and ``timestep``.

    The edges are ``az_min - az_padding`` and ``az_max + az_padding``, as
    :func:`~fyst_trajectories.planning.plan_constant_el_scan` pads them;
    an option the patch does not give takes the planner's default.
    """
    window = _ce_window(patch)
    assert window is not None  # called for a window patch only
    params = patch.scan_params
    pad = float(params.get("az_padding", _CE_PLANNER_PARAMETERS["az_padding"].default))
    accel = float(params.get("az_accel", _CE_PLANNER_PARAMETERS["az_accel"].default))
    step = float(params.get("timestep", _CE_PLANNER_PARAMETERS["timestep"].default))
    return window[0] - pad, window[1] + pad, accel, step


def _window_scan_seconds(patch: ObservingPatch, legs: int) -> float:
    """Return how long a window scan of ``legs`` whole legs lasts, its last sample's time.

    ``legs`` cruises across the padded window and ``legs - 1`` turnarounds,
    as the planner's quantiser times them: asked for this duration, the
    planner builds exactly ``legs`` legs.
    """
    lo, hi, accel, _ = _window_scan_settings(patch)
    throw = hi - lo
    t_cruise = throw / patch.velocity
    t_turn = turnaround_duration_sec(patch.velocity, accel)
    _, built = _quantize_ce_duration(
        az_throw=throw,
        velocity=patch.velocity,
        duration=legs * t_cruise + (legs - 1) * t_turn,
        az_accel=accel,
    )
    return built


def _window_block_seconds(patch: ObservingPatch, legs: int) -> float:
    """Return how long the block of a window scan of ``legs`` legs lasts.

    The block starts at the scan's first sample and ends one sample interval
    after its last, the interval of the planner's own sample grid
    (``duration / (n - 1)`` for ``n`` samples). ``inf`` for a scan too short
    to sample, which the planner refuses.
    """
    duration = _window_scan_seconds(patch, legs)
    try:
        n_points = validate_sample_count(duration, _window_scan_settings(patch)[3])
    except ValueError:
        return math.inf
    return duration + duration / (n_points - 1)


def _window_legs(patch: ObservingPatch, seconds: float) -> int:
    """Return the most whole legs of a window patch whose block lasts ``seconds`` or less.

    0 when not even one leg fits. A block of exactly ``seconds`` fits, so a
    budget equal to the block of ``n`` legs holds ``n`` and one a float step
    shorter holds ``n - 1``.

    The first estimate counts legs and turnarounds alone and is never short:
    the block of ``n`` legs lasts one sample interval longer than ``n`` legs
    and ``n - 1`` turnarounds, so it is lowered until the block fits.
    """
    lo, hi, accel, _ = _window_scan_settings(patch)
    t_cruise = (hi - lo) / patch.velocity
    t_turn = turnaround_duration_sec(patch.velocity, accel)
    legs = max(0, int(math.floor((seconds + t_turn) / (t_cruise + t_turn))))
    while legs >= 1 and _window_block_seconds(patch, legs) > seconds:
        legs -= 1
    return legs


def _window_shortest_legs(patch: ObservingPatch, overhead: OverheadModel) -> int:
    """Return the fewest whole legs whose block lasts ``overhead.min_scan_duration`` or more."""
    legs = max(1, _window_legs(patch, overhead.min_scan_duration))
    if _window_block_seconds(patch, legs) < overhead.min_scan_duration:
        legs += 1
    return legs


def _window_scan(
    patch: ObservingPatch, start: Time, legs: int, site: Site, cache: dict
) -> "ScanBlock | None":
    """Return the window scan of ``legs`` whole legs a window patch sweeps from ``start``.

    The window form of
    :func:`~fyst_trajectories.planning.plan_constant_el_scan` called with
    the patch's field, window, elevation (the detector's), speed, offset and
    options, for the duration :func:`_window_scan_seconds` gives, from
    ``start`` taken literally; the subscan block records it with
    :meth:`~fyst_trajectories.overhead.TimelineBlock.from_window_scan`.
    ``None`` when the planner refuses it. The planner's advisories describe
    the patch the caller configured, so they are not raised here; a
    consumer who rebuilds the finished timeline raises them there. The
    result, ``None`` included, is kept in ``cache`` (the context's
    ``ce_scan_starts`` memo) per patch, start and leg count.
    """
    key = ("window", patch.name, start.jd1, start.jd2, legs)
    if key in cache:
        return cache[key]
    window = _ce_window(patch)
    assert window is not None and patch.elevation is not None  # a window patch
    options = {
        key: patch.scan_params[key]
        for key in ("az_accel", "timestep", "az_padding")
        if key in patch.scan_params
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PointingWarning)
            scan: ScanBlock | None = plan_constant_el_scan(
                FieldRegion(
                    ra_center=patch.ra_center,
                    dec_center=patch.dec_center,
                    width=patch.width,
                    height=patch.height,
                ),
                az_window=window,
                elevation=patch.elevation,
                velocity=patch.velocity,
                duration=_window_scan_seconds(patch, legs),
                site=site,
                start_time=start,
                detector_offset=patch.detector_offset,
                **options,
            )
    except (ValueError, KeyError, TypeError):
        scan = None
    cache[key] = scan
    return scan


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
    reconstruction, called through :func:`_ce_pass_solve`) finds both
    RA-edge crossings forward of it. Once the
    leading edge is above the target elevation, the forward search cannot
    find its crossing again within the planner's 12-hour horizon, so a block
    anchored there is unreconstructable - the condition this gate exists to
    prevent.

    A patch whose ``scan_params`` carry an ``lsa_window`` is timed by that
    window, as the planner times it: the pass is the planner's window
    (:func:`~fyst_trajectories.planning._ce_geometry._compute_ce_duration_from_lsa`,
    searched forward from ``start_time``), ``rising`` is not read, and the
    pass is plannable only while the field's border meets ``elevation`` on
    the side the planner places its range on
    (:func:`~fyst_trajectories.planning._ce_geometry._lsa_window_rising_side`).

    Solves are memoized in ``cache`` per ``(patch.name, elevation,
    rising)``, or ``(patch.name, elevation, "lsa")`` for an ``lsa_window``
    patch: a hit is trusted while the anchor precedes the pass opening
    by :data:`_CE_ANCHOR_MARGIN_SEC`; a miss is re-solved at most every
    :data:`_CE_MISS_RESOLVE_SEC` of scheduler time. Elevation is part of
    the key because the solve depends on it and the caller passes it apart
    from the patch. In a context from ``SchedulerContext.build``, which
    refuses a constant-elevation patch without an ``elevation``,
    every call for a patch passes that elevation; a hand-built context
    (whose phases gate a patch without one at the field centre's elevation
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
    lsa_window = patch.scan_params.get("lsa_window")
    key = (patch.name, float(elevation), "lsa" if lsa_window is not None else bool(rising))
    hit = cache.get(key)
    if hit is not None:
        if hit[0] == "ok":
            _, t_open, t_close = hit
            if start_time.unix <= t_open.unix - _CE_ANCHOR_MARGIN_SEC:
                return t_open, t_close
            # The anchor is within _CE_ANCHOR_MARGIN_SEC of the pass opening,
            # or past it: re-solve.
        else:  # ("miss", solved_from)
            if (start_time - hit[1]).sec < _CE_MISS_RESOLVE_SEC:
                return None

    try:
        t_open, t_close = _ce_pass_solve(patch, elevation, rising, start_time, coords)
    except PointingError:
        cache[key] = ("miss", start_time)
        return None
    cache[key] = ("ok", t_open, t_close)
    return t_open, t_close


def _ce_pass_solve(
    patch: ObservingPatch,
    elevation: float,
    rising: bool,
    start_time: Time,
    coords: Coordinates,
    max_search_hours: float = _DEFAULT_MAX_SEARCH_HOURS,
) -> tuple[Time, Time]:
    """Return the planner's own solve of a constant-elevation patch's next pass.

    The pass ``(t_open, t_close)``
    :func:`~fyst_trajectories.planning.plan_constant_el_scan` would scan
    from ``start_time``, searching ``max_search_hours`` forward (the
    planner's own horizon by default): the first and last RA-edge crossing
    of ``elevation`` on the ``rising`` side
    (:func:`~fyst_trajectories.planning._ce_geometry._compute_ce_duration`);
    or, for a patch whose ``scan_params`` carry an ``lsa_window``, the
    window's next opening and close
    (:func:`~fyst_trajectories.planning._ce_geometry._compute_ce_duration_from_lsa`),
    ``rising`` not read, provided the field's border meets ``elevation``
    on the side the planner places the window's range on
    (:func:`~fyst_trajectories.planning._ce_geometry._lsa_window_rising_side`,
    :func:`~fyst_trajectories.planning._ce_geometry._ce_field_az_range`).
    The solve's advisories describe the pass a block records and are not
    raised here; the timeline's rebuild raises them.

    Raises
    ------
    PointingError
        Where the planner refuses the pass from ``start_time``.
    """
    field = FieldRegion(
        ra_center=patch.ra_center,
        dec_center=patch.dec_center,
        width=patch.width,
        height=patch.height,
    )
    lsa_window = patch.scan_params.get("lsa_window")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PointingWarning)
        if lsa_window is None:
            t_open, t_close, _ = _compute_ce_duration(
                field, 0.0, elevation, coords, start_time, rising, max_search_hours
            )
            return t_open, t_close
        t_open, t_close, _ = _compute_ce_duration_from_lsa(
            lsa_window, coords, start_time, max_search_hours
        )
        side = _lsa_window_rising_side(patch.ra_center, patch.dec_center, t_open, t_close, coords)
        # The planner refuses a window whose side the border never reaches.
        _ce_field_az_range(field, 0.0, elevation, side, coords, t_open, 0.0)
    return t_open, t_close


def _ce_visit_plan(
    patch: ObservingPatch,
    elevation: float,
    start_time: Time,
    end_time: Time,
    coords: Coordinates,
    cache: dict,
    ready_lead: "float | Callable[[Time], float]",
) -> tuple[bool, Time, Time] | None:
    """Choose the crossing half and pass for a CE visit starting at ``start_time``.

    The half the patch asks for (:func:`_ce_patch_rising`: an explicit
    ``scan_params["rising"]``, or the side of the meridian an azimuth window
    lies on) pins the half (None is
    returned when that half has no plannable pass). Without a request both
    halves are tried and the earlier-opening plannable pass wins, so a patch
    whose rising pass has already begun falls over to its setting pass
    instead of being emitted unreconstructable. A ``scan_params["lsa_window"]``
    fixes both the pass and its half (:func:`_ce_crossing_corridor`,
    :func:`~fyst_trajectories.planning._ce_geometry._lsa_window_rising_side`),
    as it does for the planner, and a ``rising``
    beside it is not read.

    Two window conditions apply on top of plannability:

    - readiness: the pass must open within ``ready_lead`` seconds of
      ``start_time``; a callable ``ready_lead`` is given the pass's opening
      and returns the lead for that pass. A CE drift scan only observes
      the field while its edges cross the scan elevation, so starting the
      visit hours early would book science blocks that point at empty sky
      and inflate the science accounting by the full wait. The patch
      simply stays unselected (idle, cals, or other patches) until the pass
      is imminent.
    - the opening crossing must precede ``end_time``: a pass that opens
      after the schedule window can never start inside it.

    Returns
    -------
    tuple of (bool, Time, Time) or None
        ``(rising, t_open, t_close)`` for the chosen pass, or None when
        neither half has a plannable pass that is ready and opens within
        the window.
    """
    lsa = patch.scan_params.get("lsa_window") is not None
    requested = _ce_patch_rising(patch)
    halves = (requested,) if requested is not None and not lsa else (True, False)
    best: tuple[bool, Time, Time] | None = None
    for rising in halves[:1] if lsa else halves:
        window = _ce_crossing_corridor(patch, elevation, rising, start_time, coords, cache)
        if window is None:
            continue
        t_open, t_close = window
        if lsa:
            rising = _lsa_window_rising_side(
                patch.ra_center, patch.dec_center, t_open, t_close, coords
            )
        if t_open.unix > end_time.unix:
            continue
        lead = ready_lead(t_open) if callable(ready_lead) else ready_lead
        if t_open.unix - start_time.unix > lead:
            continue  # plannable, but the pass is not imminent yet
        # The "is None or" short circuit guarantees best is a tuple on the
        # right-hand side; pylint's inference cannot narrow the union here.
        if best is None or t_open.unix < best[1].unix:  # pylint: disable=unsubscriptable-object
            best = (rising, t_open, t_close)
    return best


def _ce_swept_az_envelope(
    patch: ObservingPatch,
    elevation: float,
    rising: bool,
    coords: Coordinates,
    at: Time,
) -> tuple[float, float]:
    """Return the azimuth envelope a constant-elevation pass sweeps.

    The planner's own range for the patch at ``elevation`` on the
    ``rising`` side of the meridian (where the field's border meets that
    elevation over its whole crossing, plus padding:
    :func:`~fyst_trajectories.planning._ce_geometry._ce_field_az_range`),
    widened by one turnaround overshoot per side. That holds the range the
    mount occupies, whatever turnarounds the rebuilt trajectory makes, so a
    phase that has to reason about where the telescope goes before any
    trajectory exists can ask for it here. It is not placed on the branch
    the planner sweeps: the planner chooses that for the sweep it builds,
    while every consumer here reads the envelope modulo 360 deg or re-wraps
    it to the slew's azimuth.

    It is not the instantaneous field width :func:`_compute_az_range`
    estimates, ``patch.width / cos(el)`` about the field centre's azimuth:
    it depends on the field's declination extent rather than its width,
    and holds the whole crossing.

    The ``az_padding`` and ``az_accel`` defaults read here are the ones
    the reconstruction path applies, so this estimate and the rebuild
    describe one geometry.

    The envelope is the boresight's. With a ``detector_offset`` the
    planner places the detector's sweep and the boresight follows it at
    one azimuth shift, the one the boresight that holds the detector at
    ``elevation`` takes (:func:`_ce_scan_pose`), so the detector's range
    is moved by that shift.

    A patch with an azimuth window (:func:`_ce_window`) sweeps that window,
    padded by ``az_padding``, whatever the field does, so its envelope is
    the padded window widened by the overshoot and moved with the
    boresight, on the turn the window is written on; ``rising`` and ``at``
    are not read.

    Parameters
    ----------
    patch : ObservingPatch
        Constant-elevation patch supplying the field geometry, the scan
        velocity, the optional ``az_padding`` / ``az_accel`` overrides and
        the ``detector_offset``.
    elevation : float
        The scan elevation in degrees: the detector's, the patch's
        ``elevation``.
    rising : bool
        The crossing half, as returned by :func:`_ce_visit_plan`: ``True``
        for the range east of the meridian, ``False`` for the range west
        of it.
    coords : Coordinates
        Site coordinate transformer.
    at : Time
        The instant the planner reads the border's apparent places at: the
        pass's opening crossing, as returned by :func:`_ce_visit_plan`.

    Returns
    -------
    tuple of float
        ``(env_min, env_max)`` in degrees, ordered, before the planner
        places the sweep on a branch. The pair may lie outside
        ``[0, 360)``, and past an azimuth limit, where the padding or an
        overshoot carries an edge across north (a range that reaches the
        northern meridian ends at 0 or 360 deg).
    """
    params = patch.scan_params
    if _ce_window(patch) is not None:
        az_lo, az_hi, az_accel, _ = _window_scan_settings(patch)
    else:
        field = FieldRegion(
            ra_center=patch.ra_center,
            dec_center=patch.dec_center,
            width=patch.width,
            height=patch.height,
        )
        az_padding = params.get("az_padding", _CE_PLANNER_PARAMETERS["az_padding"].default)
        az_accel = params.get("az_accel", _CE_PLANNER_PARAMETERS["az_accel"].default)
        az_lo, az_hi = _ce_field_az_range(
            field, 0.0, elevation, rising, coords, at, float(az_padding)
        )
    env_lo, env_hi = swept_az_envelope(az_lo, az_hi, patch.velocity, float(az_accel))
    if patch.detector_offset is None:
        return env_lo, env_hi
    shift, _ = _boresight_pose(0.0, elevation, patch.detector_offset, coords.site)
    return env_lo + float(shift), env_hi + float(shift)


def _ce_scan_pose(patch: ObservingPatch, site: Site) -> tuple[float, float]:
    """Return the azimuth shift and the elevation of a constant-elevation patch's boresight.

    The planner sweeps the detector at the patch's ``elevation``; the
    focal-plane rotation follows the elevation alone, so the boresight
    that holds the detector there scans at one elevation, returned
    second, and stays one azimuth shift, returned first, from the
    detector's azimuth. ``(0.0, patch.elevation)`` without a
    ``detector_offset``. :meth:`SchedulerContext.build` refuses a patch
    whose boresight cannot be solved there, so this does not raise for a
    scheduled patch.
    """
    assert patch.elevation is not None  # a constant-elevation patch always has one
    if patch.detector_offset is None:
        return 0.0, patch.elevation
    shift, elevation = _boresight_pose(0.0, patch.elevation, patch.detector_offset, site)
    return float(shift), float(elevation)


def _boresight_elevations(
    el: np.ndarray, offset: "InstrumentOffset | None", site: Site
) -> np.ndarray:
    """Return the boresight's elevation for each detector elevation in ``el``.

    The focal-plane rotation follows the elevation alone, so the boresight
    that holds the detector at ``el`` has one elevation whatever the
    azimuth, solved here at azimuth 0. For a Prime-Cam offset that
    elevation increases with the detector's. ``el`` itself without an
    offset; an elevation of 90 or more is returned as it is, and one whose
    boresight cannot be solved is ``nan``.
    """
    if offset is None:
        return el
    el = np.asarray(el, dtype=float)
    out = el.copy()
    solvable = el < 90.0
    if not np.any(solvable):
        return out
    try:
        _, bore = _boresight_pose(np.zeros(int(solvable.sum())), el[solvable], offset, site)
        out[solvable] = np.asarray(bore, dtype=float)
    except OffsetInversionError:
        for i in np.flatnonzero(solvable):
            try:
                out[i] = float(_boresight_pose(0.0, float(el[i]), offset, site)[1])
            except OffsetInversionError:
                out[i] = np.nan
    return out


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
    the corners, ``centre + reach``, which is not clamped at 90: at 90 or
    more the circle holds the zenith. The bounds hold whatever the
    pattern's phase, which is not known before the slew; a pong passes near
    a given corner only once a period, so its bound can sit up to a
    period's drift below the trajectory it stands for.

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
        ``(lowest, highest)`` in degrees at each time: the lowest elevation
        of the pattern and the top of the circle that bounds it, above 90
        when that circle holds the zenith.
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
    return lowest, centre + reach


def _transit_offset(patch: ObservingPatch, start_time: Time, coords: Coordinates) -> float:
    """Return the seconds from ``start_time`` to the field centre's next transit.

    The centre's hour angle at ``start_time`` is read off its azimuth and
    elevation, ``H = atan2(-sin(az) cos(el), cos(lat) sin(el) - sin(lat)
    cos(el) cos(az))``, so the transit is where the centre's elevation, and
    with it the top of the circle that bounds the pattern, peaks in the
    frame the bounds are computed in; the hour angle grows at the sidereal
    rate.
    """
    az, el = coords.radec_to_altaz(patch.ra_center, patch.dec_center, start_time)
    az, el = math.radians(float(az)), math.radians(float(el))
    lat = math.radians(coords.site.latitude)
    hour_angle = math.degrees(
        math.atan2(
            -math.sin(az) * math.cos(el),
            math.cos(lat) * math.sin(el) - math.sin(lat) * math.cos(el) * math.cos(az),
        )
    )
    return (-hour_angle % 360.0) / _SIDEREAL_RATE_DEG_S


#: How far forward, in hours, :func:`_ce_pass_refusal` looks for a pass. An
#: ``lsa_window`` opens within a sidereal day; a field's pass, searched from
#: the middle of the gap between two passes, opens within half a sidereal
#: day and closes within the planner's own horizon after that, if at all.
_PASS_LOOKAHEAD_HOURS = 24.0


def _ce_pass_refusal(patch: ObservingPatch, coords: Coordinates, start_time: Time) -> str | None:
    """Return why a constant-elevation patch can never have a pass, or None.

    The question is put to the solve the crossing gate makes
    (:func:`_ce_pass_solve`, through :func:`_ce_crossing_corridor`), so the
    refusal and the gate cannot disagree about a pass. A patch has a pass
    when that solve finds one from some instant of the sidereal day from
    ``start_time`` on a side it can be scanned on: the side its
    ``scan_params`` ask for (:func:`_ce_patch_rising`: ``rising``, or the
    side of the meridian its azimuth window lies on), or either side when
    it names none; an ``lsa_window`` patch is timed by its window, which
    opens once a sidereal day and chooses its own side.

    For a field-form or azimuth-window patch the field first needs leading
    and trailing RA edges
    (:func:`~fyst_trajectories.planning._ce_geometry._field_ra_edges`, the
    planner's own rule: not a whole turn of RA at its central Dec, not
    within 0.57 deg of a celestial pole). Then each side is solved twice.
    The first solve looks :data:`_PASS_LOOKAHEAD_HOURS` forward from the
    middle of the gap between two passes on that side, where it cannot
    fall inside a pass: the leading edge, at RA ``ra_lead`` and the
    field's central Dec ``d``, crosses elevation ``e`` at site latitude
    ``lat`` at the hour angle ``-h`` rising and ``+h`` setting, with

        cos h = (sin e - sin lat sin d) / (cos lat cos d)

    (clipped to ``[-1, 1]``), and the trailing edge crosses ``s``, the
    field's RA span, later, so the gap is centred where the leading edge's
    hour angle is ``-/+h - (360 - s) / 2``, with almost six hours or more
    to either side of that instant for any pass shorter than the planner's
    horizon. It finds the next pass whenever both edges cross
    ``elevation`` there. The
    second solve is the gate's own call, with the planner's horizon, from
    :data:`_CE_ANCHOR_MARGIN_SEC` before that pass opens, so a pass longer
    than the planner's horizon, which the planner never finds whole, is
    refused too. An ``lsa_window`` patch is solved the same two ways from
    ``start_time``.

    Returns
    -------
    str or None
        The reason, starting with the patch's name and ending with the
        planner's own message for the last side tried; ``None`` when the
        patch has a pass.
    """
    assert patch.elevation is not None  # SchedulerContext.build refuses one without
    lsa = patch.scan_params.get("lsa_window") is not None
    field = FieldRegion(
        ra_center=patch.ra_center,
        dec_center=patch.dec_center,
        width=patch.width,
        height=patch.height,
    )
    ra_lead = span = half_crossing = 0.0
    if not lsa:
        try:
            ra_lead, ra_trail = _field_ra_edges(field, 0.0)
        except PointingError as exc:
            return f"{patch.name}: {exc}"
        span = ra_trail - ra_lead
        lat, dec = math.radians(coords.site.latitude), math.radians(patch.dec_center)
        cos_h = (math.sin(math.radians(patch.elevation)) - math.sin(lat) * math.sin(dec)) / (
            math.cos(lat) * math.cos(dec)
        )
        half_crossing = math.degrees(math.acos(min(1.0, max(-1.0, cos_h))))
    requested = None if lsa else _ce_patch_rising(patch)
    sides = (True,) if lsa else ((requested,) if requested is not None else (True, False))
    why = ""
    for rising in sides:
        anchor = start_time
        if not lsa:
            # The leading edge's hour angle grows at the sidereal rate.
            hour_angle = float(coords.get_hour_angle(ra_lead, start_time))
            target = (-half_crossing if rising else half_crossing) - 0.5 * (360.0 - span)
            wait = ((target - hour_angle) % 360.0) / _SIDEREAL_RATE_DEG_S
            anchor = start_time + TimeDelta(wait, format="sec")
        try:
            t_open, _ = _ce_pass_solve(
                patch, patch.elevation, rising, anchor, coords, _PASS_LOOKAHEAD_HOURS
            )
            _ce_pass_solve(
                patch,
                patch.elevation,
                rising,
                t_open - TimeDelta(_CE_ANCHOR_MARGIN_SEC, format="sec"),
                coords,
            )
        except PointingError as exc:
            why = str(exc)
            continue
        return None
    if lsa:
        where = "in its lsa_window"
    elif len(sides) == 2:
        where = "on the rising or the setting side"
    else:
        where = f"on the {'rising' if sides[0] else 'setting'} side"
    return (
        f"{patch.name}: the planner finds no pass at the scan elevation "
        f"{patch.elevation:.2f} deg {where} over a sidereal day from {start_time.iso} "
        f"(field centre at Dec {patch.dec_center:.2f}); {why}"
    )


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
    the telescope holds the field centre and no pattern runs. The circle
    that bounds the pattern must also exclude the zenith,
    ``centre + reach < 90`` with no margin: a track passing round the
    zenith winds its azimuth through whole turns, past the azimuth limits,
    which the planner refuses too. Under a ``detector_offset`` both the
    detector's pattern and the boresight's must hold all this: the planner
    checks the pattern it generates, the detector's, then moves the
    boresight to hold the detector on it and checks the boresight's. The
    boresight that holds the detector at an elevation has one elevation
    whatever the azimuth, increasing with the detector's
    (:func:`_boresight_elevations`), so the detector's bounds map to the
    boresight's exactly; ``nan`` where the boresight cannot be solved,
    which no limit holds. The bounds
    are sampled every ``step_seconds`` from then up to *max_duration* in
    one vectorised transform, together with the field centre's transit
    (:func:`_transit_offset`) when it falls in that span: the circle peaks
    there, so it cannot hold the zenith between two samples unseen. The
    first sample outside is refined by bisection between it and the last
    sample inside, returning the last time verified inside (conservative
    by construction, to within 0.3 s for the default step).

    Returns the window from *start_time*, ``lead`` included: *max_duration*
    when the pattern stays inside throughout, and ``0.0`` when it is not
    inside at ``start_time + lead`` (a field whose pattern edge has not yet
    risen above the lower limit, or has already set, or whose circle holds
    the zenith) or ``lead`` leaves no time.
    """
    limits = coords.site.telescope_limits.elevation
    if lead >= max_duration:
        return 0.0

    def inside(offsets: np.ndarray) -> np.ndarray:
        lowest, highest = _pattern_el_bounds(
            patch, start_time + TimeDelta(offsets, format="sec"), coords
        )
        held = (lowest >= limits.min) & (highest <= limits.max) & (highest < 90.0)
        if patch.detector_offset is None:
            return held
        lowest = _boresight_elevations(lowest, patch.detector_offset, coords.site)
        highest = _boresight_elevations(highest, patch.detector_offset, coords.site)
        return held & (lowest >= limits.min) & (highest <= limits.max) & (highest < 90.0)

    n_steps = max(2, int((max_duration - lead) / step_seconds) + 1)
    dt = np.linspace(lead, max_duration, n_steps)
    transit = _transit_offset(patch, start_time, coords)
    if lead <= transit <= max_duration:
        dt = np.union1d(dt, [transit])
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
    reach: float = 0.0,
    detector_offset: "InstrumentOffset | None" = None,
) -> float:
    """Compute how long a source stays sun-safe from *start_time*.

    With ``fixed_el`` the track is followed in azimuth but held at that
    elevation, which is the pose a constant-elevation patch commands: the
    selection phase scores it there, the slew drives there and every
    emitted block records it. Without it the field's own instantaneous
    elevation is used.

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
    exclusion zone mid-scan as the field moves; the constant_el branch
    applies this same clip after its corridor solve
    (:func:`_ce_visit_plan`), over the swept azimuth envelope
    (:func:`_ce_swept_az_envelope`) as ``az_span``.

    With ``reach`` (degrees, not combined with ``az_span``) the question is
    whether every point within ``reach`` of the track is safe, the pong and
    daisy clip's question about its whole pattern. In scalar mode a sample
    is unsafe when the track's separation is at most ``min_sun_angle +
    reach``. A model with a ``threshold``
    (:class:`~fyst_trajectories.sun_protocols.ZonedSunSafePredicate`) is
    held to its threshold at the track plus ``reach``; any other model is
    asked about the track and eight points on the circle of radius
    ``reach`` about it, 45 deg apart in bearing (:func:`_reach_verdicts`).

    With ``detector_offset`` (not combined with ``fixed_el`` or
    ``az_span``, whose callers pass the boresight's pose themselves) the
    track is the boresight's: at each sample, the boresight that puts that
    detector on the source. A pattern about the source is then a pattern of
    the detector, and the boresight's pattern strays farther from the
    boresight's track than ``reach``; the reach is widened by
    :func:`_offset_reach` to a bound on that. A sample whose boresight
    cannot be solved ends the window there.

    Returns *max_duration* if the source never becomes unsafe.

    Raises
    ------
    ValueError
        If ``az_span`` is combined with a nonzero ``reach``,
        ``detector_offset`` with ``az_span`` or ``fixed_el``, or
        ``az_span`` is not finite and ordered.
    """
    if az_span is not None and reach != 0.0:
        raise ValueError("az_span and reach cannot be combined")
    if detector_offset is not None and (az_span is not None or fixed_el is not None):
        raise ValueError("detector_offset cannot be combined with az_span or fixed_el")
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
    if detector_offset is not None:
        bore_az, bore_el = _boresight_track(az_arr, el_arr, detector_offset, coords.site)
        unsolved = np.flatnonzero(~np.isfinite(bore_el))
        if unsolved.size:
            # The window ends at the last sample before one whose boresight
            # cannot be solved.
            n_steps = int(unsolved[0])
            if n_steps < 2:
                return 0.0
            max_duration = float(dt[n_steps - 1])
            dt, times = dt[:n_steps], times[:n_steps]
            az_arr, el_arr = az_arr[:n_steps], el_arr[:n_steps]
            bore_az, bore_el = bore_az[:n_steps], bore_el[:n_steps]
        reach = _offset_reach(reach, np.asarray(el_arr, dtype=float), detector_offset, coords.site)
        az_arr, el_arr = bore_az, bore_el

    if sun_safe is not None:
        columns = None if az_span is None else _span_columns(az_span)
        if columns is not None:
            safe = _span_verdicts(sun_safe, columns, el_arr, times)
        elif reach != 0.0:
            safe = _reach_verdicts(sun_safe, az_arr, el_arr, times, reach, coords)
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
            if detector_offset is not None:
                try:
                    az_p, el_p = _boresight_pose(
                        float(az_p), float(el_p), detector_offset, coords.site
                    )
                except OffsetInversionError:
                    return False
            if columns is not None:
                t_one = start_time + TimeDelta([offset_s], format="sec")
                return bool(_span_verdicts(sun_safe, columns, np.array([el_p]), t_one)[0])
            if reach != 0.0:
                t_one = start_time + TimeDelta([offset_s], format="sec")
                az_one, el_one = np.array([float(az_p)]), np.array([float(el_p)])
                return bool(_reach_verdicts(sun_safe, az_one, el_one, t_one, reach, coords)[0])
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
    threshold = min_sun_angle + reach
    return _first_crossing(dt, sep, threshold, sep <= threshold, max_duration)


def _boresight_track(
    az: np.ndarray, el: np.ndarray, offset: "InstrumentOffset", site: Site
) -> tuple[np.ndarray, np.ndarray]:
    """Return the boresight poses that put ``offset``'s detector at each ``(az, el)``.

    Solved together where every sample solves, else sample by sample, with
    ``nan`` where the boresight cannot be solved.
    """
    az = np.asarray(az, dtype=float)
    el = np.asarray(el, dtype=float)
    try:
        bore_az, bore_el = _boresight_pose(az, el, offset, site)
        return np.asarray(bore_az, dtype=float), np.asarray(bore_el, dtype=float)
    except OffsetInversionError:
        out_az, out_el = np.full(az.shape, np.nan), np.full(el.shape, np.nan)
        for i in range(az.size):
            try:
                pose = _boresight_pose(float(az[i]), float(el[i]), offset, site)
            except OffsetInversionError:
                continue
            out_az[i], out_el[i] = float(pose[0]), float(pose[1])
        return out_az, out_el


def _offset_reach(
    reach: float, centre_el: np.ndarray, offset: "InstrumentOffset", site: Site
) -> float:
    """Return how far a detector's pattern puts the boresight from the boresight's centre.

    A pattern keeps the detector within ``reach`` of the field centre
    ``c``; the boresight that holds the detector at a point ``p`` of it is
    ``b``, and the boresight that holds it at ``c`` is ``B``. The rotation
    of the sphere that carries ``c`` to ``p`` along their great circle
    moves no point by more than ``|cp| <= reach``, and carries ``B`` to
    ``B'``, a point ``rho`` (the offset's radius) from ``p``, the way the
    direction from ``c`` to ``B`` is carried along that great circle. ``b``
    is also ``rho`` from ``p``, at an angle ``D`` from ``B'`` about ``p``,
    so::

        |bB| <= |bB'| + |B'B| <= 2 asin(sin(rho) sin(D / 2)) + reach.

    ``D`` is the turn of the detector-to-boresight direction at ``p``
    against its carried copy. Three things turn it: the focal-plane
    rotation, which follows the boresight's elevation, by the change of
    that elevation between ``B`` and ``b``, at most
    ``max(F(c + reach) - F(c), F(c) - F(c - reach))`` with ``F`` the
    boresight's elevation for a detector elevation, which increases with
    it; and the azimuth-elevation frame, which turns against a carried
    direction by ``sin(el) d(az)`` along a path, at most ``tan(e_max)``
    per unit length at elevations up to ``e_max``, along the great circle
    from ``c`` to ``p`` (length up to ``reach``) and along the two offset
    arcs, from ``b`` to ``p`` and from ``B`` to ``c`` (length ``rho``
    each). Every point of those arcs lies within ``rho`` of ``p`` or
    ``c``, so ``e_max = max(c + reach, F(c + reach)) + rho`` holds them.
    Hence ``D <= dF + (reach + 2 rho) tan(e_max)``, capped at a half turn.
    Where ``e_max`` reaches 90 deg, or the boresight cannot be solved at
    ``c + reach`` or ``c - reach``, the bound falls back to the triangle
    inequality, ``reach + 2 rho``, which also caps it everywhere.

    Returns the largest bound over the centre's elevations ``centre_el``,
    in degrees.
    """
    rho = math.radians(math.hypot(offset.dx, offset.dy) / 60.0)
    cap = reach + 2.0 * math.degrees(rho)
    c = np.asarray(centre_el, dtype=float)
    f_c = _boresight_elevations(c, offset, site)
    f_hi = _boresight_elevations(c + reach, offset, site)
    f_lo = _boresight_elevations(c - reach, offset, site)
    d_rot = np.radians(np.maximum(f_hi - f_c, f_c - f_lo))
    e_max = np.maximum(c + reach, f_hi) + math.degrees(rho)
    with np.errstate(invalid="ignore"):
        turn = np.minimum(
            d_rot + (math.radians(reach) + 2.0 * rho) * np.tan(np.radians(e_max)), math.pi
        )
        bound = reach + np.degrees(2.0 * np.arcsin(math.sin(rho) * np.sin(0.5 * turn)))
    usable = (e_max < 90.0) & np.isfinite(bound) & (c + reach < 90.0)
    return float(np.max(np.where(usable, np.minimum(bound, cap), cap)))


def _reach_verdicts(
    sun_safe: "SunSafePredicate",
    az: np.ndarray,
    el: np.ndarray,
    times: Time,
    reach: float,
    coords: Coordinates,
) -> np.ndarray:
    """Return, per sample, whether every point within ``reach`` deg of the pose is Sun-safe.

    ``az``, ``el`` and ``times`` are same-length 1-D. A model with a
    ``threshold`` is safe where the pose's Sun separation exceeds its
    threshold there plus ``reach``. Any other model is asked once, on the
    flattened grid of the pose and the eight points ``reach`` from it at
    bearings ``b = 0, 45, ..., 315`` deg,
    ``el' = asin(sin(el) cos(r) + cos(el) sin(r) cos(b))`` and
    ``az' = az + atan2(sin(b) sin(r) cos(el), cos(r) - sin(el) sin(el'))``.
    """
    if isinstance(sun_safe, ZonedSunSafePredicate):
        sun_az, sun_el = coords.get_sun_altaz(times)
        sep = np.asarray(coords.angular_separation(az, el, sun_az, sun_el), dtype=float)
        required = np.asarray(sun_safe.threshold(az, el, times), dtype=float)
        return sep > required + reach
    bearing = np.radians(np.arange(8) * 45.0)
    r = math.radians(reach)
    el_r = np.radians(el)[:, None]
    el_pts = np.arcsin(np.sin(el_r) * math.cos(r) + np.cos(el_r) * math.sin(r) * np.cos(bearing))
    az_pts = az[:, None] + np.degrees(
        np.arctan2(
            np.sin(bearing) * math.sin(r) * np.cos(el_r),
            math.cos(r) - np.sin(el_r) * np.sin(el_pts),
        )
    )
    grid_az = np.column_stack([az, az_pts % 360.0])
    grid_el = np.column_stack([el, np.degrees(el_pts)])
    n_t, n_pts = grid_az.shape
    t_flat = times[np.repeat(np.arange(n_t), n_pts)]
    safe = _sun_verdicts(
        sun_safe, grid_az.ravel(), grid_el.ravel(), t_flat, what="pattern reach grid"
    )
    return safe.reshape(n_t, n_pts).all(axis=1)


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


def _pattern_reach(patch: ObservingPatch) -> float:
    """Return how far a pong's or daisy's pattern reaches from the field centre, in degrees.

    A pong stays inside its box (:func:`_pong_config`,
    :func:`~fyst_trajectories.patterns.pong._pong_peak_offsets`), whose
    corners lie the half-diagonal ``hypot(half_x, half_y)`` from the centre
    whatever the pattern's ``angle``; a daisy within
    :func:`~fyst_trajectories.patterns.daisy._daisy_reach` of it.
    """
    if patch.scan_type == "pong":
        return math.hypot(*_pong_peak_offsets(_pong_config(patch)))
    settings = {**_DAISY_REBUILD_DEFAULTS, **patch.scan_params}
    return _daisy_reach(
        settings["radius"], settings["turn_radius"], patch.velocity, settings["timestep"]
    )


def _min_subscan_duration(patch: ObservingPatch, overhead: OverheadModel) -> float:
    """Return the shortest science subscan a patch can be booked for, in seconds.

    ``overhead.min_scan_duration``, or for a pong one pattern period when
    that is longer, since a pong subscan is a whole number of periods. A
    constant-elevation patch with an azimuth window books whole legs, so its
    shortest subscan is the block of the fewest legs that lasts
    ``overhead.min_scan_duration`` or more (:func:`_window_shortest_legs`).
    """
    if patch.scan_type == "pong":
        return max(overhead.min_scan_duration, _pong_period(patch))
    if _ce_window(patch) is not None:
        return _window_block_seconds(patch, _window_shortest_legs(patch, overhead))
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
    sun_safe: "SunSafePredicate | None" = None,
    *,
    ce_ready_lead: "float | Callable[[Time], float]",
    retune_lead: float = 0.0,
    ce_plan: tuple[bool, Time, Time] | None = None,
) -> float:
    """Return how long this patch can be observed from ``start_time``, in seconds.

    For constant-elevation scans, the visit runs until the chosen crossing
    pass closes (:func:`_ce_visit_plan`, or the ``t_close`` of
    ``ce_plan``): a CE drift scan is only plannable
    while its RA-edge crossings lie ahead of the anchor. A plain
    field-center-above-elevation window is not a substitute; it stays open
    long after the pass, all the way to transit and beyond, and emits
    blocks the planner could never reconstruct. For pong/daisy, the window starts at the
    max scan duration (or remaining schedule time) and is then clipped to the
    time the whole pattern stays inside the telescope elevation limits
    (:func:`_time_inside_el_limits`), so no part of the scan the planner
    builds leaves them; the window is ``0.0`` while the pattern's edge is
    below the lower limit when its subscan would start, as it is for a
    rising field before the whole pattern has cleared it. A pong visit
    fills this window with whole pattern periods after its boundary retune,
    so the selection phase admits a pong or daisy patch only while the
    window holds its shortest subscan and any retune that is due.

    Both Sun clips ask about the pose the visit commands, the boresight's.
    The constant-elevation clip evaluates the whole swept azimuth corridor
    at the elevation the boresight scans at (:func:`_ce_swept_az_envelope`,
    the planner's corridor plus the turnaround overshoot, moved with the
    boresight; :func:`_ce_scan_pose`), since the mount crosses all of it on
    every leg. The pong and daisy clip follows the boresight that puts the
    patch's detector on the field centre and keeps every point within the
    pattern's reach of it (:func:`_pattern_reach`, widened by
    :func:`_offset_reach` under a ``detector_offset``) out of the zone, and
    the elevation limits judge the boresight's pattern
    (:func:`_pattern_el_bounds`).

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
        Computed elevation of the patch center at the current time. Read
        by the constant-elevation branch only, in place of a missing
        ``patch.elevation``, which ``SchedulerContext.build`` refuses.
    ce_cache : dict, optional
        Per-run corridor memo (``SchedulerContext.ce_corridors``) for the
        constant-elevation branch. Default None uses a throwaway dict.
    sun_safe : SunSafePredicate, optional
        Injected sun-safety model for the mid-scan drift clip. Default
        ``None`` keeps the site's scalar exclusion radius.
    ce_ready_lead : float or callable
        Readiness lead (seconds) passed to :func:`_ce_visit_plan` by the
        constant-elevation branch, or a function of a pass's opening
        returning it: the scheduler passes one tick, the slew allowance
        :data:`_CE_READY_SLEW_ALLOWANCE_SEC` and the retune due when the
        pass opens. The other branches do not read it.
    retune_lead : float, optional
        Pong and daisy only: seconds of boundary retune booked at
        ``start_time`` before the subscan, during which no pattern runs,
        so the pattern has to be inside the elevation limits only from
        ``start_time + retune_lead``. Default 0.0.
    ce_plan : tuple of (bool, Time, Time), optional
        Constant-elevation only: the crossing half and the pass
        ``(rising, t_open, t_close)`` the visit was admitted on, used in
        place of asking :func:`_ce_visit_plan` from ``start_time``; the
        budget runs to ``t_close``, which :class:`ScienceScanPhase` sets
        to the end of the planner's pass when that is later. Default None
        asks.

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
        plan = ce_plan
        if plan is None:
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
        rising, t_open, t_close = plan
        corridor_dur = min((t_close - start_time).sec, remaining)
        # Sun-safety clip on top of the corridor: trim the visit before any
        # azimuth the mount sweeps at the scan elevation comes inside the
        # exclusion zone. The field centre alone is not enough: every leg
        # crosses the whole range the pass sweeps.
        if site.sun_avoidance.enabled:
            sun_safe_dur = _time_until_sun_unsafe(
                patch.ra_center,
                patch.dec_center,
                start_time,
                corridor_dur,
                coords,
                site.sun_avoidance.exclusion_radius,
                sun_safe=sun_safe,
                fixed_el=el if patch.elevation is None else _ce_scan_pose(patch, site)[1],
                az_span=_ce_swept_az_envelope(patch, el, rising, coords, t_open),
            )
            corridor_dur = min(corridor_dur, sun_safe_dur)
        return corridor_dur
    else:
        max_dur = min(overhead.max_scan_duration, remaining)
        observable_dur = _time_inside_el_limits(
            patch, start_time, max_dur, coords, lead=retune_lead
        )
        # Sun clip, as in the constant_el branch: a scan safe at its start
        # can drift into the zone as the field moves. Every point the
        # pattern reaches has to stay out, not only its centre.
        if site.sun_avoidance.enabled:
            sun_safe_dur = _time_until_sun_unsafe(
                patch.ra_center,
                patch.dec_center,
                start_time,
                max_dur,
                coords,
                site.sun_avoidance.exclusion_radius,
                sun_safe=sun_safe,
                reach=_pattern_reach(patch),
                detector_offset=patch.detector_offset,
            )
            observable_dur = min(observable_dur, sun_safe_dur)
        return min(max_dur, observable_dur)


def _compute_az_range(
    patch: ObservingPatch, center_az: float, center_el: float, site: Site
) -> tuple[float, float]:
    """Estimate the azimuth range a scan occupies at one instant.

    Extends the field's projected half width,
    ``patch.width / (2 cos el)``, to each side of ``center_az`` (the
    cosine is floored at 0.1, capping the half throw at five field widths
    near the zenith). That is
    a scalar estimate at the tick time, not the envelope a built
    trajectory sweeps: a constant-elevation pass sweeps the range in which
    the field's border meets the scan elevation over its whole crossing
    (:func:`_ce_swept_az_envelope` is that range, with its turnarounds),
    which depends on the field's declination extent and not on its width.
    It is used only where nothing is built yet: for the slew's fallback
    target when the scan cannot be planned from the visit start; as the
    range the slew's cable wrap is judged on for a pong or daisy (which
    scans about its centre) and for a constant-elevation pass that cannot
    be planned from the tick; and as the range of the provisional block a
    subscan is built from. The slew ends at the scan's first sample, a
    constant-elevation pass's wrap is judged on the range it sweeps (an
    azimuth window's included), and an emitted block records the envelope
    of its own trajectory.

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
        Center elevation in degrees, used when ``patch.elevation`` is
        None, as it always is for a pong or daisy patch.
    site : Site
        Site providing the azimuth limits for normalization.

    Returns
    -------
    tuple of float
        ``(az_start, az_end)`` in degrees with ``az_start <= az_end``.
        Placement inside the limits
        is only possible when the range fits: a range wider than the
        cable-wrap window, or one overhanging both ends, is returned
        unshifted and may exceed the limits.
    """
    limits = site.telescope_limits.azimuth

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
