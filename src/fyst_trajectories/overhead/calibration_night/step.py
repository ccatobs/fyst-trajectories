"""The four pure step functions of a calibration night.

``list_candidates`` says which bodies could be visited now and why the
others cannot; ``plan_visit`` plans one visit as a value that is either
feasible (its blocks and transition) or infeasible (its reason), never
raising for infeasibility; ``commit_visit`` folds a plan into the state;
``advance_idle`` moves time forward at the current pose. A driver loops
over them, and a person at an interactive session calls them directly.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.time import Time, TimeDelta

from ...exceptions import PointingError, PointingWarning
from ...planning import ScanBlock, plan_source_ces_passes
from ...planning.footprints import inflate_footprint, resolve_footprint
from ...trajectory_utils import get_absolute_times
from .._moves import EscapeMove, plan_escape_move
from ..models import CalibrationType, TimelineBlock, validate_scan_params
from ..transitions import DeferralReason, Transition, plan_transition
from ..utils import estimate_slew_time
from .helpers import (
    json_native,
    module_crossings,
    science_fraction,
    sweep_sun_safe,
    validate_geometry_record,
)
from .policy import ScanOverrides
from .state import NightContext, NightState
from .tables import ElevationBin, table_for

__all__ = [
    "Candidate",
    "VisitPlan",
    "advance_idle",
    "commit_visit",
    "list_candidates",
    "plan_visit",
]

# Slack added when a visit is re-anchored after the slew and tuning overran
# the first solve, so the retune never overlaps the pass start.
_ANCHOR_SLACK_SEC = 60.0
# One re-anchor, then the visit is judged on what it has.
_MAX_REANCHOR = 1
# The reasons that drop a body for the night; every other reason defers.
_DROP_REASONS = frozenset({DeferralReason.NO_WRAP, DeferralReason.LIMITS})


@dataclass(frozen=True)
class Candidate:
    """A body as seen at one instant of the night.

    Parameters
    ----------
    body : str
        Body name, lower case.
    el_bore_estimate : float
        The body's elevation in degrees at the estimated pass start (now
        plus the slew to it plus the reserved tuning).
    az_throw : float or None
        The throw the table gives at that elevation, or ``None`` below
        the table's coverage.
    bin : ElevationBin or None
        The table bin at that elevation; ``None`` above the top bin,
        where the throw is extrapolated, or below the coverage.
    sun_ok : bool
        Whether the estimated retune pose is clear of the Sun.
    moon_separation : float or None
        Separation from the Moon in degrees when the policy checks it.
    time_left_in_band : float
        Seconds until the body's elevation drops below the floor, on the
        idle-tick grid and capped at the end of the night.
    reason : DeferralReason or None
        Why the body cannot be visited now; ``None`` when it can.
    """

    body: str
    el_bore_estimate: float
    az_throw: float | None
    bin: ElevationBin | None
    sun_ok: bool
    moon_separation: float | None
    time_left_in_band: float
    reason: DeferralReason | None

    @property
    def available(self) -> bool:
        """Whether the body can be visited now."""
        return self.reason is None


@dataclass(frozen=True)
class VisitPlan:
    """The outcome of planning one visit.

    Parameters
    ----------
    body : str
        The body the visit was planned for.
    feasible : bool
        Whether the visit can be committed.
    reason : DeferralReason or None
        Why not, when it cannot.
    transition : Transition or None
        The slew to the first pass, when one was planned.
    blocks : tuple of TimelineBlock
        The blocks the visit adds (slew, detector operations, passes);
        empty when infeasible.
    warnings : tuple of str
        Advisories raised while planning: dynamics limits, a partial or
        unapplied dwell, a non-centred footprint, the kernel's refusal.
    passes : tuple of ScanBlock
        The planned passes, for inspection or plotting.
    """

    body: str
    feasible: bool
    reason: DeferralReason | None
    transition: Transition | None
    blocks: tuple[TimelineBlock, ...]
    warnings: tuple[str, ...]
    passes: tuple[ScanBlock, ...] = ()

    @property
    def summary(self) -> str:
        """One line describing the plan."""
        if not self.feasible:
            return f"{self.body}: infeasible ({self.reason})"
        t0 = self.blocks[0].t_start.iso[:19]
        t1 = self.blocks[-1].t_stop.iso[:19]
        path = self.transition.path if self.transition is not None else "none"
        return (
            f"{self.body}: {len(self.passes)} pass(es), {len(self.blocks)} blocks, "
            f"{t0} to {t1}, transition {path}"
        )

    @property
    def end_time(self) -> Time | None:
        """When the visit's last block ends, or ``None`` when infeasible."""
        return self.blocks[-1].t_stop if self.blocks else None


def _skydip_due(state: NightState, ctx: NightContext) -> bool:
    last = state.cal_state.last_skydip
    cadence = ctx.calibration_policy.skydip_cadence
    return last is None or (state.t - last).to_value("s") >= cadence


def _tuning_seconds(state: NightState, ctx: NightContext) -> float:
    """Seconds of detector operations reserved before the next pass."""
    total = 0.0
    tuning = ctx.policy.tuning
    if tuning.find_detectors_at_start and state.cal_state.last_retune is None:
        total += tuning.find_detectors_duration
    if _skydip_due(state, ctx):
        total += ctx.overhead_model.get_calibration_duration(CalibrationType.SKYDIP)
    if tuning.retune_before_each_block:
        total += ctx.overhead_model.get_calibration_duration(CalibrationType.RETUNE)
    return total


def _nearest_wrap(sky_az: float, reference_az: float) -> float:
    """Return the 360 deg image of ``sky_az`` nearest ``reference_az`` (a slew estimate aid)."""
    k = round((reference_az - sky_az) / 360.0)
    return sky_az + 360.0 * k


def _time_left_in_band(ctx: NightContext, body: str, t: Time) -> float:
    """Seconds until the body drops below the floor, on the tick grid; one ephemeris call."""
    step = ctx.policy.time_step
    remaining = (ctx.end_time - t).to_value("s")
    if remaining <= 0.0:
        return 0.0
    offsets = np.arange(step, remaining + step, step)
    offsets = np.minimum(offsets, remaining)
    probes = t + TimeDelta(offsets, format="sec")
    _, el = ctx.coords.get_body_altaz(body, probes)
    below = np.flatnonzero(np.asarray(el) < ctx.policy.el_min)
    if below.size == 0:
        return float(remaining)
    return float(offsets[below[0]])


def list_candidates(state: NightState, ctx: NightContext) -> tuple[Candidate, ...]:
    """Assess every target at the current time, in the caller's order.

    Each body is placed at the estimated pass start (now plus the slew to
    it plus the reserved tuning), its table throw is looked up at that
    elevation, the Sun is checked at the estimated retune pose, and the
    Moon is checked when the policy asks. Deferred bodies carry their
    reason until their retry time; dropped bodies carry it for the night.

    Parameters
    ----------
    state : NightState
        The night so far.
    ctx : NightContext
        The resolved inputs.

    Returns
    -------
    tuple of Candidate
        One entry per target, available or not.
    """
    tuning = _tuning_seconds(state, ctx)
    out: list[Candidate] = []
    for body in ctx.targets:
        az_now, el_now = ctx.body_altaz(body, state.t)
        slew = estimate_slew_time(
            state.az, state.el, _nearest_wrap(az_now, state.az), el_now, ctx.site
        )
        t_est = state.t + TimeDelta(slew + tuning, format="sec")
        az_est, el_est = ctx.body_altaz(body, t_est)
        table = table_for(ctx.tables, body)
        el_bin = table.for_elevation(el_est)
        throw = table.az_throw_at(el_est) if el_est >= table.el_range[0] else None
        sun_ok = bool(ctx.sun_safe(az_est, el_est, t_est))
        moon_sep: float | None = None
        if ctx.policy.moon_min_separation is not None:
            moon_az, moon_el = ctx.body_altaz("moon", t_est)
            moon_sep = float(ctx.coords.angular_separation(az_est, el_est, moon_az, moon_el))

        reason: DeferralReason | None = None
        if body in state.dropped:
            reason = state.dropped[body]
        elif body in state.deferred and (state.t - state.deferred[body][0]).to_value("s") < 0.0:
            reason = state.deferred[body][1]
        elif el_est < ctx.policy.el_min:
            reason = DeferralReason.BELOW_BAND
        elif el_est > ctx.site.telescope_limits.elevation.max:
            reason = DeferralReason.ABOVE_BAND
        elif not sun_ok:
            reason = DeferralReason.SUN_POINT
        elif moon_sep is not None and moon_sep < ctx.policy.moon_min_separation:
            reason = DeferralReason.MOON
        out.append(
            Candidate(
                body=body,
                el_bore_estimate=float(el_est),
                az_throw=None if throw is None else float(throw),
                bin=el_bin,
                sun_ok=sun_ok,
                moon_separation=moon_sep,
                time_left_in_band=_time_left_in_band(ctx, body, state.t),
                reason=reason,
            )
        )
    return tuple(out)


def _resolve_geometry(
    ctx: NightContext, body: str, el_est: float, overrides: ScanOverrides
) -> tuple[dict[str, float], float | None]:
    """Resolve the requested geometry: overrides, then the table, then the policy.

    Returns the requested record and the dwell to apply (``None`` means
    solve the crossing).
    """
    policy = ctx.policy
    table = table_for(ctx.tables, body)
    requested: dict[str, float] = {
        "az_speed": overrides.az_speed if overrides.az_speed is not None else policy.az_speed,
        "az_accel": overrides.az_accel if overrides.az_accel is not None else policy.az_accel,
    }
    if overrides.az_throw is not None:
        requested["az_throw"] = overrides.az_throw
    elif el_est >= table.el_range[0]:
        requested["az_throw"] = table.az_throw_at(el_est)
    dwell: float | None = None
    if overrides.dwell is not None:
        dwell = overrides.dwell
    elif policy.use_table_dwell:
        el_bin = table.for_elevation(el_est)
        if el_bin is not None:
            dwell = el_bin.dwell_reference
    if dwell is not None:
        requested["dwell"] = dwell
    if policy.footprint_margin > 0.0:
        requested["footprint_margin"] = policy.footprint_margin
    return requested, dwell


def _infeasible(body: str, reason: DeferralReason, warns: list[str]) -> VisitPlan:
    return VisitPlan(
        body=body, feasible=False, reason=reason, transition=None, blocks=(), warnings=tuple(warns)
    )


def _plan_passes(
    ctx: NightContext,
    body: str,
    anchor: Time,
    footprint: Any,
    requested: dict[str, float],
    dwell: float | None,
    warns: list[str],
) -> list[ScanBlock]:
    """Run the kernel for one visit, capturing its advisories into ``warns``."""
    kwargs: dict[str, Any] = {
        "body": body,
        "footprint": footprint,
        "start_time": anchor,
        "n_passes": ctx.policy.n_passes,
        "site": ctx.site,
        "sun_safe": ctx.sun_safe,
        "az_speed": requested["az_speed"],
        "az_accel": requested["az_accel"],
    }
    if "az_throw" in requested:
        kwargs["az_throw"] = requested["az_throw"]
    if dwell is not None:
        kwargs["dwell"] = dwell
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PointingWarning)
        passes = plan_source_ces_passes(**kwargs)
    for w in caught:
        if issubclass(w.category, PointingWarning):
            warns.append(str(w.message))
        else:
            # Only the planner's own advisories belong on the visit
            # record. Anything else the kernel raised (a coordinate
            # library's, numpy's) is still the caller's to see, so it is
            # re-emitted rather than swallowed by the capture.
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    return list(passes)


def plan_visit(
    state: NightState,
    ctx: NightContext,
    body: str,
    overrides: ScanOverrides | None = None,
) -> VisitPlan:
    """Plan one visit of ``body`` from the current state.

    Resolves the scan geometry (overrides, then the table, then the
    policy), anchors the passes at now plus the slew and the reserved
    detector operations, plans them with the source-CES kernel on the
    margined footprint, keeps the whole passes that end before the night
    does, checks the solved boresight elevation against the band and the
    solved crossing against the policy's cap, plans the slew to the first
    pass in a wrap that holds the pass start clear for the length of that
    pass, re-anchors once if the slew plus tuning
    overruns the pass start, gates the visit on the Sun (the retune pose,
    both ends of every pass, the whole planned trajectory, fail closed),
    and assembles the blocks: the slew, the detector operations at the
    target pose, then one calibration block per pass carrying a relative
    dispatch dict and the geometry records. A state whose pose the Sun
    zone has overtaken is moved out first
    (:func:`~fyst_trajectories.overhead.plan_escape`): the escape leads
    the blocks as a slew named ``sun_escape`` and everything after it is
    planned from the escape pose and arrival time.

    Parameters
    ----------
    state : NightState
        The night so far; time comes from here, never from a clock.
    ctx : NightContext
        The resolved inputs.
    body : str
        The body to visit, one of the context's targets.
    overrides : ScanOverrides, optional
        Per-visit geometry; a set value wins over the table and the
        policy.

    Returns
    -------
    VisitPlan
        Feasible with blocks and a transition, or infeasible with a
        reason: ``UNPLANNABLE`` when the kernel refuses the anchor,
        ``BELOW_BAND`` or ``ABOVE_BAND`` when the solved boresight
        elevation leaves the band, ``CROSSING_TOO_SLOW`` when the solved
        crossing exceeds the cap, ``WINDOW_CLOSED`` when no whole pass
        fits before the night ends or the tuning cannot finish before
        the pass, ``SUN_POINT`` when the retune pose or the pass is inside
        the Sun zone, ``NO_ESCAPE`` when the zone holds the telescope
        where it sits, or the transition's own cause.

    Raises
    ------
    ValueError
        If ``body`` is not one of the context's targets, or an argument
        combination is invalid.
    """
    body = body.lower()
    if body not in ctx.targets:
        raise ValueError(f"{body!r} is not one of the night's targets {list(ctx.targets)}")
    overrides = ScanOverrides() if overrides is None else overrides
    policy = ctx.policy
    warns: list[str] = []

    if policy.footprint.lower() not in ("c", "center"):
        warns.append(
            f"footprint {policy.footprint!r} is planned for simulation; the execution "
            "layer accepts centred footprints only"
        )
    footprint = inflate_footprint(resolve_footprint(policy.footprint), policy.footprint_margin)

    # A pose the Sun zone has overtaken has no commandable transition;
    # move out first and plan the visit from where the escape ends.
    escape = plan_escape_move(
        state.az,
        state.el,
        state.t,
        ctx.site,
        sun_safe=ctx.sun_safe,
        slew_safe=ctx.slew_safe,
        settle_time=ctx.overhead_model.settle_time,
        el_floor=policy.el_min,
        remaining=(ctx.end_time - state.t).to_value("s"),
        scan_index=state.scan_counter,
        cache=ctx.escape_cache,
    )
    if escape.label is not None:
        return _infeasible(body, escape.label, warns)
    origin = state
    if not escape.clear:
        assert escape.transition is not None
        tr = escape.transition
        origin = state.advanced(t=tr.arrival, az=tr.az_to, el=tr.el_to)

    tuning = _tuning_seconds(state, ctx)
    az_now, el_now = ctx.body_altaz(body, origin.t)
    slew_est = estimate_slew_time(
        origin.az, origin.el, _nearest_wrap(az_now, origin.az), el_now, ctx.site
    )
    anchor = origin.t + TimeDelta(slew_est + tuning, format="sec")
    _, el_est = ctx.body_altaz(body, anchor)
    requested, dwell = _resolve_geometry(ctx, body, el_est, overrides)
    if dwell is not None and policy.n_passes > 1:
        warns.append("dwell applies to a single pass; a multi-pass visit scans the crossing")
        dwell = None
        requested.pop("dwell", None)

    transition: Transition | None = None
    passes: list[ScanBlock] = []
    for attempt in range(_MAX_REANCHOR + 1):
        try:
            passes = _plan_passes(ctx, body, anchor, footprint, requested, dwell, warns)
        except PointingError as exc:
            # PointingError subclasses ValueError, so it is caught first.
            warns.append(str(exc))
            return _infeasible(body, DeferralReason.UNPLANNABLE, warns)
        except ValueError as exc:
            if dwell is None or "dwell must not exceed" not in str(exc):
                raise
            warns.append(
                f"the requested dwell of {dwell:.0f} s exceeds this crossing; scanning the "
                "full crossing instead"
            )
            dwell = None
            requested.pop("dwell", None)
            try:
                passes = _plan_passes(ctx, body, anchor, footprint, requested, None, warns)
            except PointingError as exc2:
                warns.append(str(exc2))
                return _infeasible(body, DeferralReason.UNPLANNABLE, warns)

        passes = [
            p
            for p in passes
            if (Time(p.computed_params["t1_iso"], scale="utc") - ctx.end_time).to_value("s") <= 0.0
        ]
        if not passes:
            return _infeasible(body, DeferralReason.WINDOW_CLOSED, warns)
        cp = passes[0].computed_params
        el_bore = float(cp["el_bore"])
        if el_bore < policy.el_min:
            return _infeasible(body, DeferralReason.BELOW_BAND, warns)
        if el_bore > ctx.site.telescope_limits.elevation.max:
            return _infeasible(body, DeferralReason.ABOVE_BAND, warns)
        if float(cp["crossing_seconds"]) > policy.max_pass_seconds:
            return _infeasible(body, DeferralReason.CROSSING_TOO_SLOW, warns)

        traj = passes[0].trajectory
        transition = plan_transition(
            origin.az,
            origin.el,
            float(traj.az[0]),
            el_bore,
            origin.t,
            ctx.site,
            sun_safe=ctx.sun_safe,
            slew_safe=ctx.slew_safe,
            goal_az_span=(float(traj.az.min()), float(traj.az.max())),
            settle_time=ctx.overhead_model.settle_time,
            allow_detour=policy.allow_detour,
            # The pass has to stay clear for its whole length, so the wrap
            # choice holds the start pose that long. It is the cheap gate,
            # three instants at one azimuth; the full trajectory sweep
            # below is what actually certifies the pass.
            hold=float(passes[0].duration),
        )
        if not transition.safe:
            return _infeasible(body, transition.cause, warns)
        ready = transition.arrival + TimeDelta(tuning, format="sec")
        if (ready - traj.start_time).to_value("s") <= 0.0 or attempt == _MAX_REANCHOR:
            break
        anchor = ready + TimeDelta(_ANCHOR_SLACK_SEC, format="sec")

    assert transition is not None
    ready = transition.arrival + TimeDelta(tuning, format="sec")
    if (ready - passes[0].trajectory.start_time).to_value("s") > 0.0:
        return _infeasible(body, DeferralReason.WINDOW_CLOSED, warns)

    # The Sun gate: the retune pose, both ends of every pass, the whole
    # trajectory in the chosen wrap, fail closed.
    wrap_shift = transition.az_to - float(passes[0].trajectory.az[0])
    if not ctx.sun_safe(transition.az_to, el_bore, transition.arrival):
        return _infeasible(body, DeferralReason.SUN_POINT, warns)
    for p in passes:
        times = get_absolute_times(p.trajectory)
        az = np.asarray(p.trajectory.az) + wrap_shift
        el = np.asarray(p.trajectory.el)
        ends_ok = bool(ctx.sun_safe(float(az[0]), float(el[0]), times[0])) and bool(
            ctx.sun_safe(float(az[-1]), float(el[-1]), times[-1])
        )
        if not ends_ok or not sweep_sun_safe(ctx.sun_safe, az, el, times):
            return _infeasible(body, DeferralReason.SUN_POINT, warns)

    blocks = _assemble_blocks(
        state, ctx, body, transition, passes, wrap_shift, requested, escape=escape
    )
    return VisitPlan(
        body=body,
        feasible=True,
        reason=None,
        transition=transition,
        blocks=tuple(blocks),
        warnings=tuple(warns),
        passes=tuple(passes),
    )


def _parked(
    ctx: NightContext,
    cal_type: CalibrationType,
    t: Time,
    duration: float,
    az: float,
    el: float,
    scan_index: int,
    **extra: Any,
) -> TimelineBlock:
    return TimelineBlock.calibration(
        cal_type,
        t_start=t,
        duration=duration,
        az=az,
        el=el,
        site=ctx.site,
        scan_index=scan_index,
        extra_metadata=extra or None,
    )


def _assemble_blocks(
    state: NightState,
    ctx: NightContext,
    body: str,
    transition: Transition,
    passes: list[ScanBlock],
    wrap_shift: float,
    requested: dict[str, float],
    *,
    escape: EscapeMove,
) -> list[TimelineBlock]:
    """Build the slew, the detector operations at the target pose, then one block per pass."""
    model = ctx.overhead_model
    tuning = ctx.policy.tuning
    blocks: list[TimelineBlock] = []
    t = state.t
    az_from = state.az
    scan_index = state.scan_counter
    el_bore = float(passes[0].computed_params["el_bore"])
    pose_az = transition.az_to
    pose_el = el_bore

    if escape.block is not None:
        assert escape.transition is not None
        blocks.append(escape.block)
        t = escape.transition.arrival
        az_from = escape.transition.az_to

    if transition.duration > 0.0:
        blocks.append(
            TimelineBlock.slew(
                t_start=t,
                duration=transition.duration,
                az_start=az_from,
                az_end=pose_az,
                el=el_bore,
                site=ctx.site,
                scan_index=scan_index,
                patch_name=f"slew_to_{body}",
            )
        )
        t = transition.arrival

    operations: list[tuple[CalibrationType, float, dict[str, Any]]] = []
    if tuning.find_detectors_at_start and state.cal_state.last_retune is None:
        operations.append(
            (
                CalibrationType.RETUNE,
                tuning.find_detectors_duration,
                {"operation": "find_detectors"},
            )
        )
    if _skydip_due(state, ctx):
        operations.append(
            (CalibrationType.SKYDIP, model.get_calibration_duration(CalibrationType.SKYDIP), {})
        )
    if tuning.retune_before_each_block:
        operations.append(
            (CalibrationType.RETUNE, model.get_calibration_duration(CalibrationType.RETUNE), {})
        )
    for cal_type, duration, extra in operations:
        if duration <= 0.0:
            continue
        blocks.append(_parked(ctx, cal_type, t, duration, pose_az, el_bore, scan_index, **extra))
        t = blocks[-1].t_stop

    esc = escape.transition
    transition_record = {
        "wrap": float(transition.az_to),
        "cause": str(transition.cause),
        "path": transition.path,
        "duration": float(transition.duration),
        "detour_via": None if transition.detour_via is None else list(transition.detour_via),
        "escape_via": None if esc is None else [float(esc.az_to), float(esc.el_to)],
    }
    for position, p in enumerate(passes):
        t0_scan = p.trajectory.start_time
        gap = (t0_scan - t).to_value("s")
        if gap > 0.5:
            # The gap idle records the pose it starts from, which after
            # the first pass is where that pass left the telescope, not
            # the pose the visit began at: the passes step in elevation
            # and each ends at its own final azimuth.
            blocks.append(
                TimelineBlock.idle(
                    t_start=t,
                    duration=gap,
                    az=pose_az,
                    el=pose_el,
                    site=ctx.site,
                    scan_index=scan_index,
                    reason=str(DeferralReason.WAITING_FOR_PASS),
                )
            )
        blocks.append(
            _pass_block(
                ctx, body, p, wrap_shift, requested, transition_record, position, scan_index
            )
        )
        t = blocks[-1].t_stop
        pose_az = blocks[-1].end_pose_az
        pose_el = blocks[-1].elevation
        scan_index += 1
    return blocks


def _pass_block(
    ctx: NightContext,
    body: str,
    p: ScanBlock,
    wrap_shift: float,
    requested: dict[str, float],
    transition_record: dict[str, Any],
    position: int,
    scan_index: int,
) -> TimelineBlock:
    """One calibration block per pass: a relative dispatch dict plus the records."""
    cp = p.computed_params
    pp = p.trajectory.metadata.pattern_params
    policy = ctx.policy
    az = np.asarray(p.trajectory.az) + wrap_shift
    env_lo, env_hi = float(az.min()), float(az.max())
    # The drag stops on whichever leg endpoint the last turnaround left
    # it on, so the end pose is a separate quantity from the envelope.
    az_final = float(az[-1])
    el_bore = float(cp["el_bore"])

    scan_params: dict[str, Any] = {
        "body": body,
        "footprint": policy.footprint,
        "el_bore": el_bore,
        "mode": str(cp["mode"]),
        # The dict repeats the request, and the planner requests no boresight
        # rotation. None is not the 0.0 the kernel resolves it to: an
        # execution layer may accept only an uncommanded rotator.
        "boresight_rot": None,
        "timestep": float(p.config.timestep),
        "eta_offset_deg": float(pp.get("pass_eta_offset_deg", 0.0)),
        "pass_index": int(pp.get("pass_index", position)),
        "n_passes": int(pp.get("n_passes", policy.n_passes)),
        "az_speed": float(requested["az_speed"]),
        "az_accel": float(requested["az_accel"]),
    }
    if "az_throw" in requested:
        scan_params["az_throw"] = float(requested["az_throw"])
    if "dwell" in requested:
        scan_params["dwell"] = float(requested["dwell"])
    if policy.footprint_margin > 0.0:
        scan_params["footprint_margin"] = float(policy.footprint_margin)
    validate_scan_params(scan_params, "source_ces")

    applied = {
        "az_speed": float(cp["az_speed"]),
        # Echoed from the request: the solver reports no applied
        # acceleration, so this one key carries no information the
        # request does not already hold.
        "az_accel": float(requested["az_accel"]),
        "az_throw": float(cp["az_throw"]),
        "dwell": float(cp["duration"]),
    }
    if policy.footprint_margin > 0.0:
        applied["footprint_margin"] = float(policy.footprint_margin)
    solved = {
        "az_throw": float(cp["az_throw"]),
        "crossing_seconds": float(cp["crossing_seconds"]),
    }
    extra = {
        "requested": validate_geometry_record(requested, "requested"),
        "applied": validate_geometry_record(applied, "applied"),
        "solved": validate_geometry_record(solved, "solved"),
        "science_fraction": science_fraction(p.trajectory),
        "n_legs": int(cp["n_scans"]),
        "module_crossings": module_crossings(p, ctx.site),
        "transition": transition_record,
    }
    return TimelineBlock.calibration(
        CalibrationType.PLANET_CAL,
        t_start=p.trajectory.start_time,
        duration=float(p.duration),
        az=env_lo,
        el=el_bore,
        site=ctx.site,
        scan_index=scan_index,
        target=body,
        az_end=env_hi,
        az_final=az_final,
        scan_params=json_native(scan_params),
        t0_scan=str(p.trajectory.start_time.iso),
        rising=str(cp["mode"]) == "rising",
        extra_metadata=json_native(extra),
    )


def commit_visit(state: NightState, plan: VisitPlan, *, retry_after: float = 0.0) -> NightState:
    """Fold a plan into the state.

    A feasible plan appends its blocks, moves time to their end and the
    pose to the last block's end, marks the detector operations as done,
    clears the body's deferral and advances the scripted cursor. An
    infeasible plan records a deferral until ``retry_after`` seconds from
    now or, for ``NO_WRAP`` and ``LIMITS``, a drop for the night, and
    leaves time unchanged.

    Parameters
    ----------
    state : NightState
        The night so far.
    plan : VisitPlan
        The plan to commit.
    retry_after : float, optional
        Seconds before a deferred body is a candidate again. Default 0.

    Returns
    -------
    NightState
        The new state.
    """
    if not plan.feasible:
        assert plan.reason is not None
        if plan.reason in _DROP_REASONS:
            dropped = dict(state.dropped)
            dropped[plan.body] = plan.reason
            return state.advanced(dropped=dropped)
        deferred = dict(state.deferred)
        deferred[plan.body] = (state.t + TimeDelta(retry_after, format="sec"), plan.reason)
        return state.advanced(deferred=deferred)

    cal_state = state.cal_state
    for block in plan.blocks:
        if block.block_type == "calibration":
            cal_type = block.metadata.get("cal_type")
            if cal_type in ("retune", "skydip", "planet_cal"):
                cal_state = cal_state.update(cal_type, block.t_stop)
    last = plan.blocks[-1]
    deferred = {k: v for k, v in state.deferred.items() if k != plan.body}
    return state.advanced(
        t=last.t_stop,
        az=float(last.end_pose_az),
        el=float(last.elevation),
        blocks=state.blocks + plan.blocks,
        cal_state=cal_state,
        deferred=deferred,
        script_index=state.script_index + 1,
        script_waiting_since=None,
        scan_counter=state.scan_counter + len(plan.passes),
    )


def advance_idle(
    state: NightState, ctx: NightContext, seconds: float, reason: DeferralReason
) -> NightState:
    """Emit an idle block at the current pose and move time forward.

    Parameters
    ----------
    state : NightState
        The night so far.
    ctx : NightContext
        The resolved inputs (for the site and the end of the night).
    seconds : float
        How long to idle; clipped to the end of the night.
    reason : DeferralReason
        Why, recorded on the block.

    Returns
    -------
    NightState
        The new state, unchanged when no time remains. A pose the Sun
        zone has overtaken is moved out first
        (:func:`~fyst_trajectories.overhead.plan_escape`, a slew named
        ``sun_escape``) and the idle, shortened by the move, parks at the
        escape pose. When no escape exists, or too little of the night is
        left to make the one that does, the telescope stays inside the
        zone and the idle is labelled ``NO_ESCAPE`` instead of
        ``reason``, so the record says so.
    """
    remaining = (ctx.end_time - state.t).to_value("s")
    seconds = min(float(seconds), remaining)
    if seconds <= 0.0:
        return state
    escape = plan_escape_move(
        state.az,
        state.el,
        state.t,
        ctx.site,
        sun_safe=ctx.sun_safe,
        slew_safe=ctx.slew_safe,
        settle_time=ctx.overhead_model.settle_time,
        el_floor=ctx.policy.el_min,
        remaining=remaining,
        scan_index=state.scan_counter,
        cache=ctx.escape_cache,
    )
    if not escape.clear:
        if escape.label is not None:
            reason = escape.label
        else:
            block = escape.block
            tr = escape.transition
            assert block is not None and tr is not None
            state = state.advanced(
                t=block.t_stop, az=tr.az_to, el=tr.el_to, blocks=state.blocks + (block,)
            )
            seconds = min(seconds - tr.duration, (ctx.end_time - state.t).to_value("s"))
            if seconds <= 0.0:
                return state
    block = TimelineBlock.idle(
        t_start=state.t,
        duration=seconds,
        az=state.az,
        el=state.el,
        site=ctx.site,
        scan_index=state.scan_counter,
        reason=str(reason),
    )
    return state.advanced(t=block.t_stop, blocks=state.blocks + (block,))
