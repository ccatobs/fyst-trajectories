"""Scheduler phase classes.

Each phase is a pure transformer: it receives a
:class:`SchedulerState` plus a :class:`SchedulerContext`, decides what
(if anything) to emit, and returns a :class:`PhaseResult` holding the
emitted blocks and the evolved state.

Phases do not mutate global state. The :class:`Scheduler` orchestrator
(see :mod:`scheduler.scheduler`) is responsible for composing phase
outputs into the final block list.
"""

from __future__ import annotations

import abc
import dataclasses
import itertools
import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy.time import Time, TimeDelta

from ...exceptions import OffsetInversionError, PointingError, PointingWarning
from ...planning import FieldRegion, plan_source_ces_passes
from ...sun_models import make_sun_safe
from ...trajectory_utils import get_absolute_times
from .._execution_screen import _execution_refusal
from .._moves import plan_escape_move, sweep_sun_safe
from ..constraints import _boresight_pose
from ..exceptions import BlockNotReconstructableError
from ..models import (
    BlockType,
    CalibrationSpec,
    CalibrationType,
    ObservingPatch,
    OverheadModel,
    TimelineBlock,
)
from ..schemas import validate_scan_params
from ..simulation import (
    _generate_trajectory_for_block,
    _slice_to_block_window,
    _window_block_slice,
)
from ..transitions import DeferralReason, _pose_holds, plan_transition
from ..utils import _dispatch_footprint, _normalize_az, _search_start_record, _stored_instant
from .helpers import (
    _CE_READY_SLEW_ALLOWANCE_SEC,
    _ce_scan_pose,
    _ce_swept_az_envelope,
    _ce_visit_plan,
    _ce_window,
    _compute_az_range,
    _compute_scan_duration,
    _evaluate_patch,
    _min_subscan_duration,
    _pong_period,
    _window_block_seconds,
    _window_legs,
    _window_scan,
    _window_shortest_legs,
)
from .state import SchedulerState

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...planning import ScanBlock
    from ...site import Site
    from ...sun_protocols import SunSafePredicate
    from ..schemas import SourceCESScanParams
    from ..transitions import Transition
    from .state import SchedulerContext

__all__ = [
    "CalibrationPhase",
    "PatchSelectionPhase",
    "Phase",
    "PhaseResult",
    "ScienceScanPhase",
    "SlewPhase",
]


@dataclass
class PhaseResult:
    """Output of a single phase invocation.

    Attributes
    ----------
    state : SchedulerState
        The evolved scheduler state after this phase. If the phase was
        a no-op, this is identical to the input state.
    blocks : list[TimelineBlock]
        Timeline blocks emitted by this phase (possibly empty).
    selection : ObservingPatch | None
        Selected patch for subsequent phases (set by
        :class:`PatchSelectionPhase`). ``None`` means "no patch
        observable, skip downstream science phases".
    best_az : float | None
        Azimuth of ``selection`` at ``state.current_time``, in degrees
        (under a ``detector_offset``, of the boresight that holds the
        detector on the field centre, at the patch's ``elevation`` for a
        constant-elevation patch),
        placed in the telescope's cable-wrap window on the 360-degree
        representative nearest ``state.current_az``. A replacement
        selection phase must preserve that placement:
        :class:`SlewPhase` measures direct mount travel from the
        current pose, which is only meaningful when both azimuths
        share one coherent frame. Consumed by :class:`SlewPhase` and
        :class:`ScienceScanPhase`.
    best_el : float | None
        Elevation of ``selection``'s field centre at ``state.current_time``,
        in degrees (under a ``detector_offset``, of the boresight that holds
        the detector there). :class:`SlewPhase` replaces it, for a pong or
        daisy, with the elevation the slew ends at: that of the same pose
        where the first subscan starts, or this one when that pose cannot be
        solved.
    ce_plan : tuple of (Time, bool, Time, Time) or None
        For a constant-elevation selection, the pass the selection gate
        admitted it on, ``(anchor, rising, t_open, t_close)``: the instant
        the crossing was solved from (the selection tick), the crossing
        half and the pass's opening and closing crossings. Set by
        :class:`PatchSelectionPhase`, carried forward by
        :class:`SlewPhase`, and read by both downstream phases, which
        stamp ``anchor`` on every subscan as ``metadata["t0_scan"]``.
        ``None`` for any other scan type; a downstream phase handed a
        constant-elevation selection without it solves the pass from its
        own ``state.current_time``.
    skip_to_next_iter : bool
        If True, the scheduler should restart its outer loop. Set with
        every idle tick and Sun escape a phase emits: by
        :class:`CalibrationPhase` when a parked pose escapes, by
        :class:`PatchSelectionPhase` when no patch is left, by
        :class:`SlewPhase` when it escapes or idles instead of slewing to the
        selection, and by
        :class:`ScienceScanPhase` when its visit books nothing.
    stop : bool
        If True, the scheduler should break out of its outer loop
        (used by :class:`SlewPhase` when the slew would extend past
        ``ctx.end_time``, and by any phase whose Sun escape would not
        finish inside the window).
    """

    state: SchedulerState
    blocks: list[TimelineBlock] = field(default_factory=list)
    selection: ObservingPatch | None = None
    best_az: float | None = None
    best_el: float | None = None
    ce_plan: tuple[Time, bool, Time, Time] | None = None
    skip_to_next_iter: bool = False
    stop: bool = False


def _unpack_selection(
    selection: PhaseResult | None, phase_name: str
) -> tuple[ObservingPatch, float, float]:
    """Extract ``(patch, best_az, best_el)`` from a preceding phase's result.

    Downstream phases (:class:`SlewPhase`, :class:`ScienceScanPhase`)
    require a populated :class:`PhaseResult` from
    :class:`PatchSelectionPhase`. Raises :class:`ValueError` if the
    result is missing or unpopulated.
    """
    if selection is None or selection.selection is None:
        raise ValueError(f"{phase_name} requires a PatchSelectionPhase result")
    if selection.best_az is None or selection.best_el is None:
        # PatchSelectionPhase sets these with ``selection``; checked rather
        # than asserted so the invariant survives ``python -O``.
        raise RuntimeError(
            f"{phase_name} received a PhaseResult with selection populated but "
            "best_az/best_el unset; this indicates a PatchSelectionPhase bug."
        )
    return selection.selection, selection.best_az, selection.best_el


def _selected_ce_plan(
    selection: PhaseResult | None,
    elevation: float,
    state: SchedulerState,
    ctx: SchedulerContext,
) -> tuple[Time, bool, Time, Time] | None:
    """Return the pass a constant-elevation selection was admitted on, or None.

    The selection's own ``ce_plan`` when it carries one; otherwise the
    pass the gate admits from ``state.current_time``, which is then the
    anchor. ``None`` for any other scan type, or when no pass is admitted.
    """
    patch = None if selection is None else selection.selection
    if selection is None or patch is None or patch.scan_type != "constant_el":
        return None
    if selection.ce_plan is not None:
        return selection.ce_plan
    plan = _ce_visit_plan(
        patch,
        elevation,
        state.current_time,
        ctx.end_time,
        ctx.coords,
        ctx.ce_corridors,
        _ce_ready_lead(state, ctx),
    )
    return None if plan is None else (state.current_time, *plan)


def _escape_if_overtaken(
    state: SchedulerState, ctx: SchedulerContext, *, hold: float = 0.0
) -> PhaseResult | None:
    """Move a telescope the Sun zone has overtaken, before it idles or slews.

    A pose that is inside the zone now (under the context's model, or
    the site's scalar radius) has no commandable transition, since every
    path from it starts unsafe; the antenna's own avoidance system moves
    such a telescope, and the simulator models that move here with
    :func:`~fyst_trajectories.overhead._moves.plan_escape_move`, the
    decision both offline loops share. With ``hold``, the seconds the
    caller is about to hold the pose, the pose counts as overtaken when
    the zone reaches it at any instant of the hold, and the escape's end
    pose holds that long after its arrival. Returns ``None`` when the pose
    is safe. Otherwise emits the escape as a SLEW block named
    ``sun_escape``, moves the state to its pose and restarts the loop
    (the next tick re-selects from a safe pose), or, when the zone holds
    the telescope, one idle tick labelled ``no_escape``. An escape that
    would end past the window stops the loop; the remaining sliver is
    filled by the loop's own tail idle.

    Called from several places in one tick (the loop top, the
    calibration phase, the idle emitter and the slew phase), each of
    which must be covered: the clock can advance inside a tick, and each
    asks for its own hold, so a later call is not always the same
    question. The context's ``escapes`` memo makes the repeats free
    rather than dropping a call site.
    """
    move = plan_escape_move(
        state.current_az,
        state.current_el,
        state.current_time,
        ctx.site,
        sun_safe=ctx.sun_safe,
        slew_safe=ctx.slew_safe,
        settle_time=ctx.overhead_model.settle_time,
        el_floor=ctx.el_floor,
        remaining=(ctx.end_time - state.current_time).sec,
        scan_index=state.scan_counter,
        cache=ctx.escapes,
        hold=hold,
    )
    if move.clear:
        return None
    if not move.fits:
        return PhaseResult(state=state, blocks=[], stop=True)
    if move.block is None:
        return _idle_tick(state, ctx, reason=str(move.label))
    assert move.transition is not None
    new_state = state.advanced(
        current_time=move.transition.arrival,
        current_az=move.transition.az_to,
        current_el=move.transition.el_to,
    )
    return PhaseResult(state=new_state, blocks=[move.block], skip_to_next_iter=True)


def _emit_idle_tick(
    state: SchedulerState, ctx: SchedulerContext, *, reason: str | None = None
) -> PhaseResult:
    """Escape an overtaken pose, else emit one idle tick and restart the loop.

    The tick is ``ctx.time_step`` clipped to the window's end; the state
    advances by it and nothing else changes, so the next iteration
    re-selects from scratch. ``reason`` is stored on the block when
    given. A telescope the Sun zone overtakes during the tick never idles
    in place: :func:`_escape_if_overtaken`, asked with the tick as its
    hold, moves it first, or labels the tick ``no_escape`` when it cannot.
    """
    advance = min(ctx.time_step, (ctx.end_time - state.current_time).sec)
    escaped = _escape_if_overtaken(state, ctx, hold=advance)
    if escaped is not None:
        return escaped
    return _idle_tick(state, ctx, reason=reason)


def _holds(ctx: SchedulerContext, az: float, el: float, start: Time, hold: float) -> bool:
    """Whether a pose stays outside the Sun zone from ``start`` for ``hold`` seconds.

    The zone is the context's model, or the site's scalar radius; the pose
    is asked every 30 s of the hold and at its end
    (:func:`~fyst_trajectories.overhead.transitions._pose_holds`).
    """
    sun_safe = ctx.sun_safe if ctx.sun_safe is not None else make_sun_safe("scalar", site=ctx.site)
    return _pose_holds(sun_safe, az, el, start, hold)


def _idle_tick(
    state: SchedulerState, ctx: SchedulerContext, *, reason: str | None = None
) -> PhaseResult:
    """Emit the idle tick itself, with no Sun check."""
    advance = min(ctx.time_step, (ctx.end_time - state.current_time).sec)
    idle_block = TimelineBlock.idle(
        t_start=state.current_time,
        duration=advance,
        az=state.current_az,
        el=state.current_el,
        scan_index=state.scan_counter,
        reason=reason,
    )
    new_state = state.advanced(
        current_time=state.current_time + TimeDelta(advance, format="sec"),
    )
    return PhaseResult(state=new_state, blocks=[idle_block], skip_to_next_iter=True)


class Phase(abc.ABC):
    """Abstract base class for scheduler phases.

    A phase transforms :class:`SchedulerState` in response to the
    current :class:`SchedulerContext`, optionally emitting blocks.
    Subclasses override :meth:`run`.
    """

    @abc.abstractmethod
    def run(
        self,
        state: SchedulerState,
        ctx: SchedulerContext,
        *,
        selection: PhaseResult | None = None,
    ) -> PhaseResult:
        """Execute the phase and return its result.

        Parameters
        ----------
        state : SchedulerState
            Current scheduler state.
        ctx : SchedulerContext
            Scheduling context.
        selection : PhaseResult, optional
            The selection phase's result, for the two phases that act on
            a chosen patch (:class:`SlewPhase`,
            :class:`ScienceScanPhase`); they raise without it. The
            phases that choose rather than follow ignore it.

        Returns
        -------
        PhaseResult
            Emitted blocks plus the evolved state.
        """


# Trajectory sample spacing (seconds) used when a planet calibration is
# planned as a source-CES pass sequence. Matches the plan_source_ces
# default and is recorded in the block's scan_params so a consumer can
# rebuild the pass with the same sampling.
_SOURCE_CES_TIMESTEP_SEC = 0.1


def _planet_cal_pass_block(
    pass_block: ScanBlock,
    *,
    cal_spec: CalibrationSpec,
    footprint: str,
    t_start: Time,
    scan_index: int,
    subscan_index: int,
    az_shift: float,
    search_start: Time,
) -> tuple[TimelineBlock, Time, float, float]:
    """Build one planet-calibration CALIBRATION block from a source-CES pass.

    Returns ``(block, t_stop, az_final, el_bore)``. ``t_stop`` is one
    sample interval after the pass's last sample (:func:`_pass_bounds`),
    where the block ends. ``az_final`` / ``el_bore`` give the pass end pose
    the caller carries forward as the scheduler's current position. The
    sequence is one scan, so the passes share ``scan_index`` and are told
    apart by ``subscan_index``, their position in it.

    The block's ``t_start`` is supplied by the caller: the pass's first
    sample, or the previous block's stop when the pass opens within
    0.01 s of it, so the block holds the pass's samples and nothing
    before them. The planner's window start is recorded in
    ``metadata["t0_scan"]``. The azimuth bounds are the executed envelope:
    the min/max over the pass trajectory's azimuth samples (drift
    included), read from the planned trajectory itself rather than
    re-derived from the scalar parameters. The drag ends on whichever leg
    endpoint its last turnaround left it on, generally neither bound, so
    that azimuth is recorded separately as the block's ``az_final``. The
    recorded ``boresight_rot`` is the request, ``None``: the scheduler
    requests no rotation, and the planner reads ``None`` as ``0.0``.

    ``az_shift`` is the multiple of 360 degrees the acquisition slew's
    wrap applies to the planner's azimuths; the envelope and ``az_final``
    are recorded in that wrap, the one the telescope executes, while
    ``scan_params`` keep the planner's own solution, so a rebuild returns
    the pass in the planner's wrap. ``search_start`` is the anchor the
    planner searched from, recorded beside ``scan_params`` so a rebuild
    repeats that search.
    """
    cp = pass_block.computed_params
    pp = pass_block.trajectory.metadata.pattern_params

    t0_iso = str(cp["t0_iso"])
    t1_iso = str(cp["t1_iso"])
    _, t_stop = _pass_bounds(pass_block)
    el_bore = float(cp["el_bore"])
    mode = str(cp["mode"])

    env_lo = float(np.min(pass_block.trajectory.az)) + az_shift
    env_hi = float(np.max(pass_block.trajectory.az)) + az_shift
    az_final = float(pass_block.trajectory.az[-1]) + az_shift

    scan_params: SourceCESScanParams = {
        "body": str(cal_spec.target),
        # "c" for the on-axis bay whichever spelling the policy used, a ring
        # bay by its IM name, as a calibration night's pass dict names it.
        "footprint": _dispatch_footprint(footprint),
        "el_bore": el_bore,
        "mode": mode,
        "window": [t0_iso, t1_iso],
        "boresight_rot": None,
        "timestep": _SOURCE_CES_TIMESTEP_SEC,
        "eta_offset_deg": float(pp["pass_eta_offset_deg"]),
        "pass_index": int(pp["pass_index"]),
        "n_passes": int(pp["n_passes"]),
    }
    validate_scan_params(scan_params, "source_ces")

    block = TimelineBlock.calibration(
        cal_type=cal_spec.name,
        t_start=t_start,
        duration=(t_stop - t_start).sec,
        az=env_lo,
        el=el_bore,
        scan_index=scan_index,
        target=cal_spec.target,
        az_end=env_hi,
        az_final=az_final,
        subscan_index=subscan_index,
        scan_params=scan_params,
        t0_scan=t0_iso,
        rising=(mode == "rising"),
        extra_metadata={"search_start": _search_start_record(search_start)},
    )
    return block, t_stop, az_final, el_bore


def _pass_bounds(pass_block: ScanBlock) -> tuple[Time, Time]:
    """Return the span a planned pass's block covers, ``(first, stop)``.

    ``first`` is the pass's first sample and ``stop`` one sample interval
    after its last, the half-open span ``[first, stop)`` that holds every
    sample of the pass.
    """
    trajectory = pass_block.trajectory
    step = float(pass_block.config.timestep)
    first = trajectory.start_time + TimeDelta(float(trajectory.times[0]), format="sec")
    stop = trajectory.start_time + TimeDelta(float(trajectory.times[-1]) + step, format="sec")
    return first, stop


def _plan_planet_cal_acquisition(
    state: SchedulerState,
    ctx: SchedulerContext,
    planner_kwargs: dict[str, Any],
) -> tuple[list[ScanBlock], Transition, Time] | None:
    """Plan a planet calibration's passes and the slew to the first one.

    Returns ``(kept, transition, anchor)``: the passes whose blocks
    (:func:`_pass_bounds`) end by ``ctx.end_time``, less any pass whose
    first sample falls before the previous kept pass's block ends; the
    commandable slew to the first one, which arrives no later than that
    pass's first sample; and the anchor the passes were planned from.
    Returns ``None`` when the planner raises
    :class:`~fyst_trajectories.exceptions.PointingError`, no pass fits, the
    slew is refused, or it still arrives late after the second plan.
    ``planner_kwargs`` are the arguments of
    :func:`~fyst_trajectories.planning.plan_source_ces_passes` except
    ``start_time``.

    The planner's advisories are not raised, as a science subscan's build
    raises none (:func:`_record_executed_azimuth`): one about a pass that
    is emitted is raised again by a rebuild of the timeline, which plans
    the pass again from its block, and the one naming passes that overlap
    in time is acted on here, by that drop.
    """
    # Plan at the clock; when the slew arrives after the first pass would
    # start, plan once more anchored at the arrival. The planner starts a
    # pass at or after its anchor, and the arrival already includes the
    # settle time, so the second plan needs no slack.
    anchor = state.current_time
    for _ in range(2):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PointingWarning)
                passes = plan_source_ces_passes(start_time=anchor, **planner_kwargs)
        except PointingError:
            return None

        # End of night: keep only the passes whose blocks end by the window's
        # close (n_passes stays the requested total, so a truncation shows).
        bounds = [_pass_bounds(b) for b in passes]
        fitting = [
            (b, span)
            for b, span in zip(passes, bounds, strict=True)
            if span[1].unix <= ctx.end_time.unix
        ]
        # A pass whose first sample falls before the previous kept pass's
        # block ends (an el_step well below the footprint's elevation
        # extent) would overlap it. It is dropped, leaving a gap in
        # pass_index.
        kept: list[ScanBlock] = []
        starts: list[Time] = []
        previous_stop: Time | None = None
        for b, (opens, stop) in fitting:
            if previous_stop is not None and opens.unix < previous_stop.unix:
                continue
            kept.append(b)
            starts.append(opens)
            previous_stop = stop
        if not kept:
            return None

        first = kept[0]
        first_sample = starts[0]
        transition = plan_transition(
            state.current_az,
            state.current_el,
            float(first.trajectory.az[0]),
            float(first.computed_params["el_bore"]),
            state.current_time,
            ctx.site,
            sun_safe=ctx.sun_safe,
            slew_safe=ctx.slew_safe,
            goal_az_span=(
                min(float(np.min(b.trajectory.az)) for b in kept),
                max(float(np.max(b.trajectory.az)) for b in kept),
            ),
            settle_time=ctx.overhead_model.settle_time,
            hold=float(first.duration),
        )
        if not transition.safe:
            return None
        if (transition.arrival - first_sample).sec <= 0.0:
            return kept, transition, anchor
        anchor = transition.arrival
    return None


def _emit_planet_cal_passes(
    state: SchedulerState,
    ctx: SchedulerContext,
    cal_spec: CalibrationSpec,
) -> tuple[bool, list[TimelineBlock], SchedulerState]:
    """Plan a planet calibration as a multi-pass source-CES sequence.

    Returns ``(emitted, blocks, new_state)``. The target is the first entry
    of ``policy.planet_targets`` above ``policy.planet_min_elevation``
    whose own position is clear of the Sun now (the context's ``sun_safe``
    model, or the site's scalar radius), so a planet inside the zone is
    passed over for the next one. The passes are planned anchored at
    ``state.current_time``, and the slew to the first pass is planned with
    :func:`~fyst_trajectories.overhead.plan_transition` from the current
    pose, as :class:`SlewPhase` plans a science slew: the encoder wrap must
    hold every pass's azimuths, the direct path must clear the Sun zone
    and the first pass's start pose must stay clear for that pass's
    duration after arrival; no detour is attempted. When the slew arrives
    after the first pass would start, the passes are planned once more
    anchored at the arrival. Every pass, placed in the slew's wrap, is
    then swept against the Sun model with
    :func:`~fyst_trajectories.overhead._moves.sweep_sun_safe`, and each
    pass block is put to the booking screen
    (:func:`~fyst_trajectories.overhead._execution_screen._execution_refusal`).

    ``emitted`` is ``False`` when no target is clear, the sequence is
    infeasible (any :class:`~fyst_trajectories.exceptions.PointingError`),
    no pass finishes before ``ctx.end_time``, the slew is refused or still
    arrives late after the second plan, any pass sample is inside the
    Sun zone, the pose held while waiting for a pass is inside it at any
    instant of the wait, or the booking screen refuses a pass; in that case ``blocks`` is empty
    and ``new_state`` is the unchanged input state. The caller then neither emits nor marks the
    cadence, so the calibration stays due and is retried on the next
    scheduler iteration, exactly like a planet cal with no visible planet.

    On success the blocks tile ``[state.current_time, stop]`` with no
    gaps, ``stop`` being one sample interval after the last pass's last
    sample: a SLEW block named ``slew_to_<planet>`` when the move takes
    more than 1 second, then for each pass an IDLE block labelled
    ``waiting_for_pass`` when its first sample comes more than 0.01 s after
    the previous block ends, and its CALIBRATION block, which spans the
    pass's samples (:func:`_pass_bounds`). A wait holds the pose the slew
    or the previous pass left the telescope at; no move is booked between
    passes. ``new_state`` advances ``current_time`` to ``stop`` and
    ``current_az`` / ``current_el`` to the last pass's end pose, and the
    planet-cal cadence is marked at the (pre-slew) ``state.current_time``,
    matching the parked path.
    The whole sequence is one scan: the blocks share
    ``state.scan_counter`` and the passes carry their position in the
    sequence as ``subscan_index``, so each pass has its own identity.
    """
    policy = ctx.calibration_policy
    sun_safe = ctx.sun_safe if ctx.sun_safe is not None else make_sun_safe("scalar", site=ctx.site)

    # The first listed planet that is up and clear of the Sun now. One
    # point query per target here spares planning passes the sweep below
    # would refuse, on every iteration the calibration stays due.
    target = None
    for name in policy.planet_targets:
        body_az, body_el = ctx.coords.get_body_altaz(name, state.current_time)
        if float(body_el) > policy.planet_min_elevation and sun_safe(
            float(body_az), float(body_el), state.current_time
        ):
            target = name
            break
    if target is None:
        return False, [], state
    cal_spec = dataclasses.replace(cal_spec, target=target)

    kwargs = dict(
        body=target,
        footprint=policy.planet_cal_footprint,
        n_passes=policy.planet_cal_passes,
        site=ctx.site,
        timestep=_SOURCE_CES_TIMESTEP_SEC,
        # The injected sun model reaches the planet-cal planner too (None
        # keeps the planner's scalar default).
        sun_safe=ctx.sun_safe,
    )
    if policy.planet_cal_el_step is not None:
        kwargs["el_step"] = policy.planet_cal_el_step

    acquisition = _plan_planet_cal_acquisition(state, ctx, kwargs)
    if acquisition is None:
        return False, [], state
    kept, transition, anchor = acquisition

    # The transition's hold checks the first pass's start pose every 30 s;
    # the sweep certifies every sample of every pass, in the executed wrap,
    # and each wait is checked where it is booked.
    shift = transition.az_shift
    for b in kept:
        az = np.asarray(b.trajectory.az) + shift
        if not sweep_sun_safe(sun_safe, az, b.trajectory.el, get_absolute_times(b.trajectory)):
            return False, [], state

    blocks: list[TimelineBlock] = []
    t_start = state.current_time
    end_az = state.current_az
    end_el = state.current_el
    if transition.duration > 1.0:
        blocks.append(
            TimelineBlock.slew(
                t_start=state.current_time,
                duration=transition.duration,
                az_start=state.current_az,
                az_end=transition.az_to,
                el=transition.el_to,
                scan_index=state.scan_counter,
                patch_name=f"slew_to_{target}",
            )
        )
        t_start = transition.arrival
        end_az = transition.az_to
        end_el = transition.el_to
    for position, pass_block in enumerate(kept):
        # Until the pass opens the telescope holds where the slew or the
        # previous pass left it; no move is booked between passes.
        first_sample, _ = _pass_bounds(pass_block)
        wait = (first_sample - t_start).sec
        if wait > _SAMPLE_SPAN_TOLERANCE_SEC:
            if not _pose_holds(sun_safe, end_az, end_el, t_start, wait):
                return False, [], state
            blocks.append(
                TimelineBlock.idle(
                    t_start=t_start,
                    duration=wait,
                    az=end_az,
                    el=end_el,
                    scan_index=state.scan_counter,
                    reason=str(DeferralReason.WAITING_FOR_PASS),
                )
            )
            t_start = first_sample
        block, t_stop, end_az, end_el = _planet_cal_pass_block(
            pass_block,
            cal_spec=cal_spec,
            footprint=policy.planet_cal_footprint,
            t_start=t_start,
            scan_index=state.scan_counter,
            subscan_index=position,
            az_shift=shift,
            search_start=anchor,
        )
        refusal = _execution_refusal(
            pass_block.trajectory,
            "source_ces",
            block.metadata["scan_params"],
            ctx.site,
            ctx.sun_safe,
        )
        if refusal is not None:
            return False, [], state
        blocks.append(block)
        t_start = t_stop

    new_state = state.advanced(
        cal_state=state.cal_state.update(cal_spec.name, state.current_time),
        current_time=t_start,
        current_az=end_az,
        current_el=end_el,
    )
    return True, blocks, new_state


def _due_retune_spec(state: SchedulerState, ctx: SchedulerContext) -> CalibrationSpec | None:
    """Return the retune's spec if one is due at the current time, else None.

    With ``retune_cadence=0.0`` the tracker always reports one due; a
    nonzero cadence reports one only once elapsed. Peeking the spec
    separately from emitting lets :class:`ScienceScanPhase` verify a
    minimum-duration subscan still fits after the retune BEFORE booking
    it, so a visit never ends on a dangling retune.
    """
    needed = state.cal_state.needs_calibration(
        state.current_time,
        ctx.calibration_policy,
        ctx.overhead_model,
        coords=ctx.coords,
    )
    for spec in needed:
        if spec.name == CalibrationType.RETUNE:
            return spec
    return None


def _first_retune_spec(
    state: SchedulerState, ctx: SchedulerContext, visit_start: Time, opens: Time | None
) -> CalibrationSpec | None:
    """Return the retune a visit books before its first subscan, or None.

    It is the retune due at the instant the first subscan would start
    without one: the later of ``visit_start`` and ``opens``, the opening of
    the constant-elevation pass the visit scans (``None`` for a pong or
    daisy). A retune that falls due while a constant-elevation visit waits
    for its pass is booked when the visit starts, before the wait. With
    ``retune_cadence=0.0`` one is always due.
    """
    at = visit_start if opens is None or opens.unix <= visit_start.unix else opens
    return _due_retune_spec(state.advanced(current_time=at), ctx)


def _ce_ready_lead(state: SchedulerState, ctx: SchedulerContext) -> Callable[[Time], float]:
    """Return the readiness lead of a constant-elevation visit starting now, in seconds.

    The lead is a function of the opening ``t_open`` of the pass the visit
    would scan: one scheduler tick, the slew allowance and the retune the
    visit books before its first subscan
    (:func:`_first_retune_spec` from ``state.current_time`` for a pass
    opening at ``t_open``). A visit admitted at the earliest tick therefore
    has time to slew and retune before its pass opens, and waits for the
    opening rather than starting its scan after it, provided its slew
    takes no longer than the allowance.
    """

    def lead(t_open: Time) -> float:
        due = _first_retune_spec(state, ctx, state.current_time, t_open)
        return ctx.time_step + _CE_READY_SLEW_ALLOWANCE_SEC + (0.0 if due is None else due.duration)

    return lead


def _shortest_subscan_fits(
    patch: ObservingPatch, state: SchedulerState, ctx: SchedulerContext, center_el: float
) -> bool:
    """Return whether a pong or daisy visit starting at ``state.current_time`` can scan.

    The visit books the retune due at that time and then its shortest
    subscan (``min_scan_duration``, or one pattern period for a pong when
    that is longer), and both have to fit the budget
    :class:`ScienceScanPhase` gives it from that time: the time the whole
    pattern stays inside the elevation limits after the retune, clipped by
    the Sun and the end of the schedule. :class:`PatchSelectionPhase` asks
    at the tick, and :class:`SlewPhase` again at the slew's planned
    arrival, when the visit actually starts.
    """
    due = _due_retune_spec(state, ctx)
    retune_lead = 0.0 if due is None else due.duration
    budget = _compute_scan_duration(
        patch,
        state.current_time,
        ctx.end_time,
        ctx.site,
        ctx.coords,
        ctx.overhead_model,
        center_el,
        sun_safe=ctx.sun_safe,
        ce_ready_lead=_ce_ready_lead(state, ctx),
        retune_lead=retune_lead,
    )
    return budget >= retune_lead + _min_subscan_duration(patch, ctx.overhead_model)


def _emit_retune_block(
    state: SchedulerState,
    spec: CalibrationSpec,
    az_start: float,
    az_end: float,
    el: float,
) -> tuple[SchedulerState, TimelineBlock]:
    """Emit one retune block at the current time and advance past it."""
    block = TimelineBlock.retune(
        t_start=state.current_time,
        duration=spec.duration,
        az_start=az_start,
        az_end=az_end,
        el=el,
        scan_index=state.scan_counter,
    )
    state = state.advanced(
        cal_state=state.cal_state.update("retune", state.current_time),
        current_time=state.current_time + TimeDelta(spec.duration, format="sec"),
    )
    return state, block


def _scan_coupled_retune(
    spec: CalibrationSpec, state: SchedulerState, ctx: SchedulerContext
) -> bool:
    """Whether ``spec`` is a scan-coupled retune that :class:`CalibrationPhase` leaves alone.

    Scan-coupled retunes (cadence 0) belong to :class:`ScienceScanPhase`'s
    subscan boundaries, not the per-tick path; only the startup burst
    (``last_retune`` is None) fires one there, so the night begins tuned.
    """
    return (
        spec.name == CalibrationType.RETUNE
        and ctx.calibration_policy.retune_cadence == 0.0
        and state.cal_state.last_retune is not None
    )


def _loop_top_hold(state: SchedulerState, ctx: SchedulerContext) -> float:
    """Return how long the loop expects to hold the pose it starts an iteration at, in seconds.

    The duration of the first calibration :class:`CalibrationPhase` would
    park now, else one scheduler tick; either clipped to the window's end.
    The loop's escape asks with it, so an escape lands where the next
    block can be held, rather than where the phase that holds it would
    have to move the telescope again.
    """
    remaining = (ctx.end_time - state.current_time).sec
    needed = state.cal_state.needs_calibration(
        state.current_time, ctx.calibration_policy, ctx.overhead_model, coords=ctx.coords
    )
    for spec in needed:
        if _scan_coupled_retune(spec, state, ctx):
            continue
        if spec.name == CalibrationType.PLANET_CAL and ctx.calibration_policy.planet_cal_scan:
            continue
        return min(spec.duration, remaining)
    return min(ctx.time_step, remaining)


class CalibrationPhase(Phase):
    """Emit any calibration blocks whose cadence has elapsed.

    Queries the context's calibration policy and the state's
    :class:`~fyst_trajectories.overhead.CalibrationState` to determine
    which calibrations are due at ``state.current_time``. Emits each due
    calibration as a CALIBRATION block, updates the cadence tracker, and
    advances ``current_time`` past each block.

    Exception: a scan-coupled retune (``retune_cadence == 0.0``) is NOT
    emitted here after the startup burst. This phase runs on every
    outer-loop iteration, including idle ticks, and retuning a parked
    telescope every tick serves nothing; scan-coupled retunes fire in
    :class:`ScienceScanPhase` before each subscan instead (after the slew,
    and before the wait for a constant-elevation pass that opens later).

    Clamps each cal block's duration against the remaining schedule
    window so no block extends past ``ctx.end_time``. Stops early if
    the schedule window is exhausted partway through the burst.

    A calibration is parked only where the pose stays outside the Sun
    zone for the whole calibration. When the zone reaches it sooner the
    burst ends there: the telescope is moved out (a SLEW block named
    ``sun_escape``) or, when it cannot be, one idle tick labelled
    ``no_escape`` is booked, and the result asks the loop to restart (or to
    stop, when the escape would end past the window), with the calibration
    still due.

    When ``CalibrationPolicy.planet_cal_scan`` is set, a due
    ``planet_cal`` is instead planned as a multi-pass source-CES
    sequence on the first listed planet that is up and clear of the Sun:
    a SLEW block named ``slew_to_<planet>``, planned with
    :func:`~fyst_trajectories.overhead.plan_transition` as a science slew
    is, then for each pass an idle ``waiting_for_pass`` block until its
    first sample, when it opens later, and a CALIBRATION block spanning its
    samples (:func:`_emit_planet_cal_passes`). A sequence that cannot be
    planned or reached, that has a sample or a wait inside the Sun zone,
    or that the booking screen refuses is skipped and left due, so the
    calibration retries on a later iteration.
    """

    def run(
        self,
        state: SchedulerState,
        ctx: SchedulerContext,
        *,
        selection: PhaseResult | None = None,
    ) -> PhaseResult:
        """Emit any due calibration blocks and advance state."""
        blocks: list[TimelineBlock] = []

        needed_cals = state.cal_state.needs_calibration(
            state.current_time,
            ctx.calibration_policy,
            ctx.overhead_model,
            coords=ctx.coords,
        )
        for cal_spec in needed_cals:
            if state.current_time.unix >= ctx.end_time.unix:
                break

            if _scan_coupled_retune(cal_spec, state, ctx):
                continue

            if (
                cal_spec.name == CalibrationType.PLANET_CAL
                and ctx.calibration_policy.planet_cal_scan
            ):
                emitted, cal_blocks, new_state = _emit_planet_cal_passes(state, ctx, cal_spec)
                if emitted:
                    blocks.extend(cal_blocks)
                    state = new_state
                continue

            cal_duration = min(cal_spec.duration, (ctx.end_time - state.current_time).sec)
            # A pose the zone reaches during the calibration ends the burst:
            # the telescope is moved out first, and the calibration, still
            # due, is parked on a later iteration where the escape left it.
            escaped = _escape_if_overtaken(state, ctx, hold=cal_duration)
            if escaped is not None:
                return dataclasses.replace(escaped, blocks=blocks + escaped.blocks)
            cal_block = TimelineBlock.calibration(
                cal_type=cal_spec.name,
                t_start=state.current_time,
                duration=cal_duration,
                az=state.current_az,
                el=state.current_el,
                scan_index=state.scan_counter,
                target=cal_spec.target,
            )
            blocks.append(cal_block)
            state = state.advanced(
                cal_state=state.cal_state.update(cal_spec.name, state.current_time),
                current_time=state.current_time + TimeDelta(cal_duration, format="sec"),
            )

        return PhaseResult(state=state, blocks=blocks)


class PatchSelectionPhase(Phase):
    """Evaluate all patches against constraints; pick the best.

    For each patch in ``ctx.patches``, computes its instantaneous
    (az, el) at ``state.current_time``, scores it against
    ``ctx.constraints``, and multiplies by ``patch.weight /
    patch.priority``. A constant-elevation patch is scored at the azimuth
    the field centre is at now and the patch's ``elevation``, the elevation
    the rest of the visit is judged at (:class:`SlewPhase` slews there and
    every emitted block records it). Under a ``detector_offset`` every pose
    is the boresight's that holds the detector there (for a
    constant-elevation patch at the boresight's elevation for the patch's
    ``elevation``), and a patch whose boresight pose cannot be solved now
    is passed over.

    A patch with a positive score is still passed over when:

    - it is a constant-elevation patch and the planner's own crossing
      search finds no pass at its ``elevation`` that is plannable from now
      and opens before the window ends and within the readiness lead of now:
      one tick, a slew allowance and the retune due when that pass opens.
      A patch that asks for one side (``scan_params["rising"]``, or the
      side of the meridian its azimuth window lies on) is solved on that
      side only, and an ``lsa_window`` sets the pass and its side
      (:func:`~fyst_trajectories.overhead.scheduler.helpers._ce_visit_plan`);
    - it is a pong or daisy patch and the retune due now plus its
      shortest subscan (``min_scan_duration``, or one pattern period for
      a pong when that is longer) exceeds the budget
      :class:`ScienceScanPhase` would give it from now.

    That budget runs from now until the first of: ``max_scan_duration``
    elapsing; the end of the window; any point within the pattern's reach
    of the field centre's track entering the Sun zone; and, from
    the end of the due retune, the pattern's lowest point falling below
    the lower elevation limit or its bounding circle passing the upper
    limit or reaching the zenith. A pattern outside the limits, or
    holding the zenith, when the retune ends gets no budget. The pattern
    follows the field's own track: a pong's lowest point is a corner of
    its box, and its bounding circle is the circle about the field centre
    through the corners; a daisy's bounding circle is the circle of its
    reach about the centre, which also gives its lowest point, and the
    reach exceeds its ``radius`` by at least its turn radius. Under a
    ``detector_offset`` the Sun clip follows the boresight's track, with
    the reach widened for the offset, and both the detector's pattern and
    the boresight's must stay inside the limits and below the zenith.
    :class:`SlewPhase` asks the same question again at the slew's
    arrival.

    The highest-scoring patch left wins. If none is left, emits an IDLE
    block advancing by ``ctx.time_step`` and sets
    ``skip_to_next_iter=True``: the outer scheduler should skip the
    slew/science phases for this iteration. A pose the Sun zone has
    overtaken is moved out first instead of idling in place (see
    :class:`SlewPhase`).

    Otherwise, returns no blocks but populates ``selection``,
    ``best_az``, ``best_el`` in the result so downstream phases can
    consume them. ``best_az`` is placed in the telescope's cable-wrap
    window on the representative nearest the current pose, so the
    slew and the scan-range math downstream operate in one coherent frame.
    """

    def run(
        self,
        state: SchedulerState,
        ctx: SchedulerContext,
        *,
        selection: PhaseResult | None = None,
    ) -> PhaseResult:
        """Select the highest-scoring patch or emit an idle block."""
        best_patch: ObservingPatch | None = None
        best_score = 0.0
        best_az = 0.0
        best_el = 0.0
        best_plan: tuple[Time, bool, Time, Time] | None = None
        ce_lead = _ce_ready_lead(state, ctx)

        for patch in ctx.patches:
            az, el = ctx.coords.radec_to_altaz(
                patch.ra_center, patch.dec_center, state.current_time
            )
            check_az, check_el = az, patch.elevation if patch.elevation is not None else el
            # Every pose is the boresight's: the one that puts the patch's
            # detector on the field centre (a constant-elevation patch's at
            # its elevation, which is also the azimuth handed on). A pose the
            # boresight cannot take now passes the patch over.
            if patch.detector_offset is not None:
                try:
                    check_az, check_el = _boresight_pose(
                        float(az), float(check_el), patch.detector_offset, ctx.site
                    )
                    _, centre_el = _boresight_pose(
                        float(az), float(el), patch.detector_offset, ctx.site
                    )
                    az, el = check_az, centre_el
                except OffsetInversionError:
                    continue
            score = _evaluate_patch(
                patch, state.current_time, check_az, check_el, ctx.coords, ctx.constraints
            )
            score *= patch.weight / patch.priority

            # A constant-elevation patch is selectable only while a crossing
            # pass is still plannable from now (the forward solve
            # plan_constant_el_scan runs at reconstruction); without the gate
            # it scores until transit, long after the pass opens, and the
            # blocks emitted there cannot be rebuilt. The gate also holds a
            # patch to the side it asks for (scan_params["rising"], or the
            # side of the meridian its azimuth window lies on): only that
            # side's pass is solved, and it must open within the lead.
            plan = None
            if score > 0.0 and patch.scan_type == "constant_el":
                gate_el = patch.elevation if patch.elevation is not None else el
                plan = _ce_visit_plan(
                    patch,
                    gate_el,
                    state.current_time,
                    ctx.end_time,
                    ctx.coords,
                    ctx.ce_corridors,
                    ce_lead,
                )
                if plan is None:
                    score = 0.0

            # A pong or daisy patch is selectable only while its shortest
            # subscan, plus a due boundary retune, fits the time its whole
            # pattern stays inside the elevation limits, clipped by the Sun
            # and the window; otherwise it would be slewed to and idled on.
            if (
                score > 0.0
                and patch.scan_type in ("pong", "daisy")
                and not _shortest_subscan_fits(patch, state, ctx, el)
            ):
                score = 0.0

            if score > best_score:
                best_score = score
                best_patch = patch
                best_az = az
                best_el = el
                best_plan = None if plan is None else (state.current_time, *plan)

        if best_patch is None or best_score == 0.0:
            return _emit_idle_tick(state, ctx)

        # Astropy's azimuth is in [0, 360); the wrap image nearest the mount
        # keeps a north-straddling slew short.
        return PhaseResult(
            state=state,
            blocks=[],
            selection=best_patch,
            best_az=_normalize_az(best_az, ctx.site, ref=state.current_az),
            best_el=best_el,
            ce_plan=best_plan,
        )


#: The most transitions :class:`SlewPhase` plans for one science visit. Each
#: plan after the first starts the visit when the previous slew arrived, so the
#: lateness shrinks from round to round, apart from the hundredths of a second
#: a constant-elevation target's step to another sample can add; at the cap the
#: last transition is booked with whatever lateness remains.
_SLEW_MAX_PLANS = 10

#: A slew arriving at most this many seconds after the visit start its target
#: was read for counts as on time: well under one trajectory sample, so the
#: subscan still starts on the sample the slew ends at.
_SLEW_ARRIVAL_TOLERANCE_SEC = 0.01

#: How far apart two instants must be, in seconds, for the scheduler to act on
#: the gap: a pass that opens later than this after the block before it (a
#: retune, a slew or the previous pass) is waited for with an idle block, and a
#: pass that ends later than this after its closing crossing extends the visit's
#: budget to its end. The timeline's own block-time tolerance
#: (:meth:`~fyst_trajectories.overhead.ObservingTimeline.validate`).
_SAMPLE_SPAN_TOLERANCE_SEC = 0.01

#: How far short of a whole number of pong periods a subscan's time may fall
#: and still hold that many, in seconds: the noise of a difference of two
#: ``Time`` instants, far below one trajectory sample.
_PERIOD_FIT_TOLERANCE_SEC = 1e-6


def _probe_build(block: TimelineBlock, site: Site) -> ScanBlock | None:
    """Build a science block's trajectory through the reconstruction path, or None.

    ``None`` stands for any refusal the reconstruction path itself would
    log and skip. The build is a probe, so its advisories are not raised:
    they describe the patch geometry the caller already configured, and a
    consumer who rebuilds the finished timeline raises them there.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PointingWarning)
            return _generate_trajectory_for_block(block, site)
    except (ValueError, KeyError, TypeError):
        return None


def _probe_key(block: TimelineBlock) -> tuple[Any, ...]:
    """Return what a science block's build depends on: its window, elevation and metadata."""
    return (
        block.t_start.jd1,
        block.t_start.jd2,
        block.t_stop.jd1,
        block.t_stop.jd2,
        block.elevation,
        repr(sorted(block.metadata.items())),
    )


def _screened_build(
    block: TimelineBlock,
    site: Site,
    sun_safe: SunSafePredicate | None,
    probes: dict[tuple[Any, ...], ScanBlock] | None,
) -> ScanBlock | DeferralReason:
    """Build a science block's trajectory and put it to the booking screen, or say why not.

    Returns the build (:func:`_probe_build`) when the booking screen passes
    it, ``UNPLANNABLE`` when the reconstruction path refuses the block, and
    the reason :func:`~fyst_trajectories.overhead._execution_screen._execution_refusal`
    gives when the screen refuses the scan. The checks run under the build's
    advisory filter. ``probes``, when given, holds builds that passed, by
    :func:`_probe_key`: a build found there is taken out and returned, and
    one that passes here is put in.
    """
    key = _probe_key(block)
    if probes is not None and key in probes:
        return probes.pop(key)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PointingWarning)
        scan_block = _probe_build(block, site)
        if scan_block is None:
            return DeferralReason.UNPLANNABLE
        refusal = _execution_refusal(
            scan_block.trajectory, block.scan_type, block.metadata["scan_params"], site, sun_safe
        )
    if refusal is not None:
        return refusal
    if probes is not None:
        probes[key] = scan_block
    return scan_block


def _provisional_ce_scan(
    patch: ObservingPatch,
    *,
    elevation: float,
    rising: bool,
    t0_scan: str,
    scan_start: Time,
    ctx: SchedulerContext,
) -> ScanBlock | None:
    """Return the scan a constant-elevation visit's subscans are cut from, or None.

    The trajectory of a science block from ``scan_start`` (the end of the
    retune the visit books first) to the end of the schedule, stamped with
    the visit's anchor ``t0_scan`` and crossing half ``rising``, built
    through the reconstruction path as the booked blocks are rebuilt
    (:func:`_probe_build`). The rebuild plans the whole pass from the
    anchor and keeps the samples inside the block, so this is the pass
    from ``scan_start`` until it ends or the schedule does. Its first
    sample is where the slew ends (:func:`_scan_start_pose`) and, when the
    pass opens after ``scan_start``, where the first subscan starts; its
    last sample is where the visit's last subscan ends
    (:func:`_ce_pass_bounds`).

    One build per distinct ``scan_start``: :class:`SlewPhase` asks once per
    round of its slew solve, each round moving the visit's start, and
    :class:`ScienceScanPhase` reuses the build of the last round. The
    result, ``None`` for a refusal included, is kept in the context's
    ``ce_scan_starts`` memo.
    """
    key = (
        patch.name,
        float(elevation),
        bool(rising),
        t0_scan,
        scan_start.jd1,
        scan_start.jd2,
    )
    if key in ctx.ce_scan_starts:
        return ctx.ce_scan_starts[key]
    block = TimelineBlock.science(
        patch=patch,
        t_start=scan_start,
        duration=max((ctx.end_time - scan_start).sec, 1.0),
        az_start=0.0,
        az_end=0.0,
        el=elevation,
        scan_index=0,
        subscan_index=0,
        rising=rising,
        t0_scan=t0_scan,
    )
    scan_block = _probe_build(block, ctx.site)
    ctx.ce_scan_starts[key] = scan_block
    return scan_block


def _ce_pass_bounds(scan_block: ScanBlock, scan_start: Time) -> tuple[Time | None, Time]:
    """Return when a constant-elevation visit's pass opens after ``scan_start``, and ends.

    ``scan_block`` is the visit's scan from ``scan_start``
    (:func:`_provisional_ce_scan`). The first value is the pass's first
    sample when the pass opens more than :data:`_SAMPLE_SPAN_TOLERANCE_SEC`
    after ``scan_start``, and ``None`` when it is open by then. The planner
    starts a field-form pass at its opening crossing, recorded as
    ``computed_params["start_time_iso"]``, so a pass that opened before
    ``scan_start`` is told apart from one that opens within a sample
    interval after it. The second value is one sample interval after the
    scan's last sample: where the visit's last block ends. The planner
    times a pass as a whole number of legs, so that can be before or after
    the pass's closing crossing; when the schedule ends first it is one
    interval after the last sample before the schedule's end.
    """
    traj = scan_block.trajectory
    assert traj.start_time is not None  # the field form always dates its scan
    times = traj.times
    step = float(times[1] - times[0])
    first = traj.start_time + TimeDelta(float(times[0]), format="sec")
    end = traj.start_time + TimeDelta(float(times[-1]) + step, format="sec")
    opening = Time(str(scan_block.computed_params["start_time_iso"]), scale="utc")
    if (opening - scan_start).sec > _SAMPLE_SPAN_TOLERANCE_SEC:
        return first, end
    return None, end


def _window_open_at(t_open: Time, scan_start: Time) -> Time | None:
    """Return when a window visit's first subscan waits for, or None to start at ``scan_start``.

    The field's crossing of the patch elevation, ``t_open``, when it comes
    more than :data:`_SAMPLE_SPAN_TOLERANCE_SEC` after ``scan_start``.
    """
    return t_open if (t_open - scan_start).sec > _SAMPLE_SPAN_TOLERANCE_SEC else None


def _scan_start_pose(
    patch: ObservingPatch,
    visit_start: Time,
    *,
    elevation: float,
    wrap_ref: float,
    ce_plan: tuple[Time, bool, Time, Time] | None,
    state: SchedulerState,
    ctx: SchedulerContext,
) -> tuple[float, float] | None:
    """Return the first sample of a visit's first subscan, or None.

    :class:`ScienceScanPhase` starts the visit at ``visit_start``, books
    the retune :func:`_first_retune_spec` gives for it, and starts the
    first subscan after it. That subscan's first sample is the pose the
    slew has to end at:

    - for a pong or daisy, the boresight that puts the patch's detector on
      the field centre at the subscan's start, where the pattern starts,
      in the scheduler's own transform, on the wrap nearest ``wrap_ref``
      (``elevation`` is not read);
    - for a constant-elevation patch, the first sample at or after the
      retune's end of the pass the visit was admitted on (``ce_plan``,
      see :attr:`PhaseResult.ce_plan`), planned at ``elevation``, the
      detector's, built by :func:`_provisional_ce_scan` from the plan's
      anchor as :class:`ScienceScanPhase` records it in
      ``metadata["t0_scan"]``: the pass's first sample when the pass opens
      after the retune, where the visit then waits for it. Its azimuth is
      placed on the wrap the booked block records its range in, and its
      elevation is the boresight's (:func:`_ce_scan_pose`);
    - for a constant-elevation patch with an azimuth window, the first
      sample of the window scan its first subscan sweeps
      (:func:`~fyst_trajectories.overhead.scheduler.helpers._window_scan`,
      at its shortest), from where that subscan starts
      (:func:`_first_subscan_start`), placed and raised the same way.

    ``None`` when there is no constant-elevation plan, or its pass or
    window scan cannot be built from the subscan's start or holds too few
    samples after it, or the boresight cannot be solved for a pong or
    daisy's centre.
    """
    due = _first_retune_spec(state, ctx, visit_start, None if ce_plan is None else ce_plan[2])
    start = visit_start if due is None else visit_start + TimeDelta(due.duration, format="sec")
    if patch.scan_type != "constant_el":
        az, el = ctx.coords.radec_to_altaz(patch.ra_center, patch.dec_center, start)
        try:
            az, el = _boresight_pose(float(az), float(el), patch.detector_offset, ctx.site)
        except OffsetInversionError:
            return None
        az_wrapped = _normalize_az(float(az), ctx.site, ref=wrap_ref)
        return az_wrapped, float(el)

    if ce_plan is None:
        return None
    if _ce_window(patch) is not None:
        first = _first_subscan_start(
            patch, visit_start, elevation=elevation, ce_plan=ce_plan, state=state, ctx=ctx
        )
        scan_block = _window_scan(
            patch,
            first,
            _window_shortest_legs(patch, ctx.overhead_model),
            ctx.site,
            ctx.ce_scan_starts,
        )
    else:
        anchor, rising, _, _ = ce_plan
        scan_block = _provisional_ce_scan(
            patch,
            elevation=elevation,
            rising=rising,
            t0_scan=anchor.isot,
            scan_start=start,
            ctx=ctx,
        )
    if scan_block is None:
        return None
    traj = scan_block.trajectory
    mid = 0.5 * (float(np.min(traj.az)) + float(np.max(traj.az)))
    shift = 360.0 * round((wrap_ref - mid) / 360.0)
    return float(traj.az[0]) + shift, _ce_scan_pose(patch, ctx.site)[1]


def _first_subscan_start(
    patch: ObservingPatch,
    visit_start: Time,
    *,
    elevation: float,
    ce_plan: tuple[Time, bool, Time, Time] | None,
    state: SchedulerState,
    ctx: SchedulerContext,
) -> Time:
    """Return when a visit starting at ``visit_start`` starts its first subscan.

    After the retune :func:`_first_retune_spec` gives for it, and, for a
    constant-elevation pass that opens later, at the pass's first sample
    (:func:`_ce_pass_bounds` of :func:`_provisional_ce_scan`), as
    :class:`ScienceScanPhase` books it. A patch with an azimuth window is
    scanned from the field's crossing of its elevation (the pass's opening
    ``t_open``) when that is later. Until then the telescope holds the
    pose the slew ends at.
    """
    due = _first_retune_spec(state, ctx, visit_start, None if ce_plan is None else ce_plan[2])
    start = visit_start if due is None else visit_start + TimeDelta(due.duration, format="sec")
    if patch.scan_type != "constant_el" or ce_plan is None:
        return start
    anchor, rising, t_open, _ = ce_plan
    if _ce_window(patch) is not None:
        open_at = _window_open_at(t_open, start)
        return start if open_at is None else open_at
    scan_block = _provisional_ce_scan(
        patch, elevation=elevation, rising=rising, t0_scan=anchor.isot, scan_start=start, ctx=ctx
    )
    if scan_block is None:
        return start
    open_at, _ = _ce_pass_bounds(scan_block, start)
    return start if open_at is None else open_at


def _first_subscan_refusal(
    patch: ObservingPatch,
    visit_start: Time,
    *,
    elevation: float,
    best_az: float,
    best_el: float,
    wrap_ref: float,
    ce_plan: tuple[Time, bool, Time, Time] | None,
    state: SchedulerState,
    ctx: SchedulerContext,
) -> DeferralReason | None:
    """Return why a visit's shortest first subscan cannot be booked, or None.

    The subscan is the first one :class:`ScienceScanPhase` books for a
    visit starting at ``visit_start``, after the retune
    :func:`_first_retune_spec` gives, at its shortest
    (:func:`_min_subscan_duration`): the last length that phase tries
    (:func:`_plan_science_subscan`), so a visit this probe passes books at
    least this subscan. A pong or daisy subscan is built and screened as
    that phase builds it, and a build that passes is kept in
    ``ctx.science_probes``, emptied first, for that phase to take. A
    constant-elevation subscan is cut from the visit's scan from the
    retune's end (:func:`_provisional_ce_scan`), from where the science
    phase starts it (:func:`_first_subscan_start`), and put to
    :func:`~fyst_trajectories.overhead._execution_screen._execution_refusal`;
    for a patch with an azimuth window it is the window scan of the fewest
    legs that subscan may hold, planned from there.
    For a pong or daisy every refusal of that build is returned,
    ``UNPLANNABLE`` when the planner refuses the subscan: it is the science
    phase's last attempt, built from the same start. Where the planner
    cannot build a constant-elevation visit's scan, or its first window
    scan, ``None`` is returned and the science phase reports the refusal.
    """
    ctx.science_probes.clear()
    due = _first_retune_spec(state, ctx, visit_start, None if ce_plan is None else ce_plan[2])
    start = visit_start if due is None else visit_start + TimeDelta(due.duration, format="sec")
    shortest = _min_subscan_duration(patch, ctx.overhead_model)
    if patch.scan_type != "constant_el":
        ha = ctx.coords.get_hour_angle(patch.ra_center, visit_start)
        az_lo, az_hi = _compute_az_range(patch, best_az, best_el, ctx.site)
        planned = _plan_science_subscan(
            patch,
            t_start=start,
            available=shortest,
            shortest=shortest,
            period=_pong_period(patch) if patch.scan_type == "pong" else None,
            az_start=az_lo,
            az_end=az_hi,
            el=elevation,
            site=ctx.site,
            scan_index=state.scan_counter,
            subscan_index=0,
            rising=patch.scan_params.get("rising", ha < 0.0),
            t0_scan=None,
            wrap_ref=wrap_ref,
            sun_safe=ctx.sun_safe,
            probes=ctx.science_probes,
        )
        return planned if isinstance(planned, DeferralReason) else None
    if ce_plan is None:
        return None
    if _ce_window(patch) is not None:
        first = _first_subscan_start(
            patch, visit_start, elevation=elevation, ce_plan=ce_plan, state=state, ctx=ctx
        )
        window_scan = _window_scan(
            patch,
            first,
            _window_shortest_legs(patch, ctx.overhead_model),
            ctx.site,
            ctx.ce_scan_starts,
        )
        if window_scan is None:
            return None
        return _execution_refusal(window_scan.trajectory, "constant_el", {}, ctx.site, ctx.sun_safe)
    anchor, rising, _, _ = ce_plan
    scan_block = _provisional_ce_scan(
        patch,
        elevation=elevation,
        rising=rising,
        t0_scan=anchor.isot,
        scan_start=start,
        ctx=ctx,
    )
    if scan_block is None:
        return None
    # The subscan the science phase books last when longer ones are refused:
    # from the retune's end, or from the pass's first sample when the pass
    # opens later (:func:`_first_subscan_start`), for the shortest subscan.
    open_at, _ = _ce_pass_bounds(scan_block, start)
    head = TimelineBlock.science(
        patch=patch,
        t_start=start if open_at is None else open_at,
        duration=shortest,
        az_start=0.0,
        az_end=0.0,
        el=elevation,
        scan_index=state.scan_counter,
        subscan_index=0,
        rising=rising,
        t0_scan=anchor.isot,
    )
    try:
        head_scan = _slice_to_block_window(scan_block, head)
    except BlockNotReconstructableError:
        return None
    return _execution_refusal(
        head_scan.trajectory, "constant_el", patch.scan_params, ctx.site, ctx.sun_safe
    )


class SlewPhase(Phase):
    """Plan the slew to the selected patch and emit its block.

    The move from ``(state.current_az, state.current_el)`` to the science
    pose is planned with
    :func:`~fyst_trajectories.overhead.plan_transition`: the encoder wrap
    is chosen so the whole azimuth envelope the visit will occupy stays
    inside the azimuth limits, the
    direct path is clear of the Sun zone (the context's ``sun_safe``
    model, or the site's scalar radius) and the pose stays clear of it for
    one scheduler tick after arrival (asked every 30 s and at the tick's
    end), and the duration is the kinematic
    estimate plus the overhead model's settle time. Holding the goal for a
    tick is what keeps the loop from slewing to a pose the next tick would
    escape from.

    The slew ends where the visit's first subscan starts: its first
    sample, after the retune the visit books before it
    (:func:`_first_retune_spec`). For a pong or
    daisy that is the field centre at the subscan's start; for a
    constant-elevation patch it is the first sample of the pass the visit
    scans, and for one with an azimuth window the first sample of its first
    window scan. Under a ``detector_offset`` every target is the
    boresight's pose that holds the detector there, so a
    constant-elevation target is at the boresight's elevation for the
    patch's ``elevation`` (:func:`_ce_scan_pose`). The target is read for a visit
    starting at the tick; when
    the slew planned to it arrives later, the visit starts at that arrival
    and the slew is planned again to the target read there, until the
    slew arrives in time, or until ten plans have been made, when the last
    is booked late. A slew in time waits at its target until the visit
    starts, and the wait is part of the SLEW block's duration. When the scan cannot be
    planned from the visit start, the slew ends at ``selection.best_az`` (at
    the elevation a constant-elevation target has, else at
    ``selection.best_el``), or at the range
    midpoint when the scan's azimuth range lives on a different cable-wrap
    branch than the field centre (a range branch-shifted into the limits),
    so the unwind onto the scan's branch is modelled rather than silently
    skipped.

    A SLEW block is emitted when the booked duration exceeds 1 second;
    ``current_time`` then advances by its duration and ``current_az`` /
    ``current_el`` move to the slew target, so blocks emitted before the
    next science scan (idles, calibrations) carry the post-slew pose. A
    move under the threshold emits nothing and leaves the state untouched.
    When the transition lands on a wrap other than the one the selection
    placed (the near path blocked, the far one clear), ``best_az`` is
    carried forward shifted onto that wrap so the science range downstream
    is derived in the same frame.

    The span the wrap must hold is the azimuth envelope the visit
    executes. For a constant-elevation patch whose pass is plannable
    from the tick it is the range the pass sweeps (for a patch with an
    azimuth window, that window as the planner pads it, with its
    turnarounds), which can lie far from the width estimate about the
    field's centre; for a pong or daisy, which executes about its centre,
    and for a constant-elevation pass that cannot be planned from the
    tick, it is that width estimate. A constant-elevation pass follows
    whichever wrap holds its sweep.

    A refused transition (every wrap Sun-blocked, no clear direct path,
    the range fitting no wrap, the elevation outside the limits), in any
    round of the solve, never raises: the patch is treated as unselectable for this tick, an IDLE
    block carrying the refusal as ``metadata["reason"]`` is emitted, and
    the loop restarts, so the same patch is re-tried once the Sun has
    moved. No detour is attempted.

    The visit starts once the slew and any hold end, so a pong or daisy
    patch is slewed to only if its shortest subscan, after the retune due
    at that start, still fits the budget :class:`ScienceScanPhase` will give it
    then (the selection phase asks the same question at the tick, before
    the slew's length is known). When it no longer fits, because the
    pattern reaches an elevation limit, the Sun clip closes or the
    schedule ends within the slew, the tick idles at the unmoved pose
    with ``metadata["reason"]`` set to ``unplannable``. Nor is a visit
    slewed to when the booking screen refuses its shortest first subscan
    (:func:`_first_subscan_refusal`), for a sample inside the Sun zone (the
    site's exclusion radius, which PCS enforces, and ``sun_safe`` when
    given), an axis past a Go TCS limit, which PCS refuses, or two samples
    closer than the Go TCS minimum sample interval, or, for a pong or
    daisy, when the planner refuses that subscan: the tick idles at the
    unmoved pose with ``metadata["reason"]`` set to ``sun_scan``,
    ``axis_dynamics`` or ``unplannable``. Nor
    when the pose the slew ends at would be inside the Sun zone at any
    instant before the first subscan starts (the retune booked first and
    any wait for a constant-elevation pass, :func:`_first_subscan_start`):
    the tick idles with ``sun_point``.

    A telescope the Sun zone has overtaken while it sat still is moved
    out first (:func:`~fyst_trajectories.overhead.plan_escape`, emitted
    as a SLEW block named ``sun_escape``) and the loop restarts, so a
    transition is never planned from inside the zone; when no escape
    exists the tick idles with ``no_escape``.

    Requires the previous :class:`PatchSelectionPhase` result in
    ``selection``; if the slew would extend past ``ctx.end_time``,
    sets ``stop=True`` on the returned :class:`PhaseResult` so the
    outer loop terminates cleanly.
    """

    def run(
        self,
        state: SchedulerState,
        ctx: SchedulerContext,
        *,
        selection: PhaseResult | None = None,
    ) -> PhaseResult:
        """Plan the slew and emit its block, or idle on a refused transition."""
        best_patch, best_az, best_el = _unpack_selection(selection, "SlewPhase")

        escaped = _escape_if_overtaken(state, ctx)
        if escaped is not None:
            return escaped

        # Slewing to the field centre's instantaneous elevation would leave
        # the leg into a constant-elevation scan unmodelled. Its pass is
        # planned at the patch's elevation, the detector's (plan_el), and
        # the boresight scans at slew_el.
        plan_el = best_patch.elevation if best_patch.elevation is not None else best_el
        slew_el = (
            best_el if best_patch.elevation is None else _ce_scan_pose(best_patch, ctx.site)[1]
        )

        # The fallback target, for a visit whose scan cannot be planned, and
        # the branch the swept envelope is placed on: the field centre, or
        # the midpoint of the tick's scalar range when a branch shift into
        # the limits puts that range a full turn from the centre, so the
        # unwind onto the scan's branch is a modelled move.
        az_lo_sci, az_hi_sci = _compute_az_range(best_patch, best_az, best_el, ctx.site)
        mid_sci = 0.5 * (az_lo_sci + az_hi_sci)
        slew_az = best_az if abs(mid_sci - best_az) <= 180.0 else mid_sci

        ce_plan = _selected_ce_plan(selection, plan_el, state, ctx)
        swept = self._visit_az_envelope(best_patch, plan_el, slew_az, ce_plan, ctx)
        env_lo, env_hi = (az_lo_sci, az_hi_sci) if swept is None else swept
        # A constant-elevation pass executes its swept envelope, which can sit
        # far from the scalar estimate about the field centre, so the wrap must
        # hold that envelope; the visit also takes its wrap reference from it
        # and follows any wrap. A pong or daisy executes about its centre and
        # keeps the scalar range's reference.
        follows_envelope = swept is not None
        wrap_ref = 0.5 * (env_lo + env_hi) if follows_envelope else mid_sci

        def plan(goal_az: float, goal_el: float) -> Transition:
            # The wrap must hold the slew target and the whole azimuth
            # envelope the visit occupies.
            return plan_transition(
                state.current_az,
                state.current_el,
                goal_az,
                goal_el,
                state.current_time,
                ctx.site,
                sun_safe=ctx.sun_safe,
                slew_safe=ctx.slew_safe,
                goal_az_span=(min(env_lo, goal_az), max(env_hi, goal_az)),
                settle_time=ctx.overhead_model.settle_time,
                hold=ctx.time_step,
            )

        def pose(visit_start: Time) -> tuple[float, float] | None:
            return _scan_start_pose(
                best_patch,
                visit_start,
                elevation=plan_el,
                wrap_ref=wrap_ref,
                ce_plan=ce_plan,
                state=state,
                ctx=ctx,
            )

        transition, visit_start, (slew_az, slew_el) = self._slew_onto_scan_start(
            plan, (slew_az, slew_el), pose, state.current_time
        )
        if not transition.safe:
            return _emit_idle_tick(state, ctx, reason=str(transition.cause))

        # A wrap other than the placed one moves the science range too. An
        # envelope-placed pass follows it: the transition's span already
        # held that envelope on the new wrap. Otherwise the range is
        # re-derived from the shifted centre, and a range that cannot follow
        # makes the tick unselectable as well.
        shift = transition.az_shift
        if shift != 0.0 and follows_envelope:
            best_az += shift
        elif shift != 0.0:
            shifted = _compute_az_range(best_patch, best_az + shift, best_el, ctx.site)
            if not np.allclose(shifted, (az_lo_sci + shift, az_hi_sci + shift), atol=1e-6):
                return _emit_idle_tick(state, ctx, reason=str(DeferralReason.NO_WRAP))
            best_az += shift

        moves = transition.duration > 1.0
        if moves and visit_start.unix >= ctx.end_time.unix:
            return PhaseResult(
                state=state,
                blocks=[],
                selection=best_patch,
                best_az=best_az,
                best_el=best_el,
                ce_plan=ce_plan,
                stop=True,
            )

        # A pong or daisy whose shortest subscan no longer fits when the
        # visit starts is not slewed to: the selection gate asked at the
        # tick, before the slew's length was known, and a setting window
        # can close within the slew.
        if best_patch.scan_type in ("pong", "daisy") and not _shortest_subscan_fits(
            best_patch, state.advanced(current_time=visit_start), ctx, best_el
        ):
            return _emit_idle_tick(state, ctx, reason=str(DeferralReason.UNPLANNABLE))

        # Nor is a visit whose shortest first subscan the booking screen
        # refuses, or, for a pong or daisy, the planner.
        refusal = _first_subscan_refusal(
            best_patch,
            visit_start,
            elevation=plan_el if best_patch.scan_type == "constant_el" else slew_el,
            best_az=best_az,
            best_el=best_el,
            wrap_ref=transition.az_to if moves else state.current_az,
            ce_plan=ce_plan,
            state=state,
            ctx=ctx,
        )
        if refusal is not None:
            return _emit_idle_tick(state, ctx, reason=str(refusal))

        # Nor is a visit whose first subscan would wait, through its retune
        # and any wait for its pass, at a pose the Sun zone reaches first.
        held_az = transition.az_to if moves else state.current_az
        first_start = _first_subscan_start(
            best_patch, visit_start, elevation=plan_el, ce_plan=ce_plan, state=state, ctx=ctx
        )
        held = (first_start - visit_start).sec
        if held > 0.0 and not _holds(ctx, held_az, slew_el, visit_start, held):
            return _emit_idle_tick(state, ctx, reason=str(DeferralReason.SUN_POINT))

        blocks: list[TimelineBlock] = []
        if moves:
            slew_block = TimelineBlock.slew(
                t_start=state.current_time,
                duration=transition.duration,
                az_start=state.current_az,
                az_end=transition.az_to,
                el=transition.el_to,
                scan_index=state.scan_counter,
                patch_name=f"slew_to_{best_patch.name}",
            )
            blocks.append(slew_block)
            state = state.advanced(
                current_time=visit_start,
                current_az=transition.az_to,
                current_el=transition.el_to,
            )

        # A pong or daisy visit is stamped at the elevation its pattern
        # starts at, the one the slew ends at.
        if best_patch.scan_type != "constant_el":
            best_el = slew_el
        return PhaseResult(
            state=state,
            blocks=blocks,
            selection=best_patch,
            best_az=best_az,
            best_el=best_el,
            ce_plan=ce_plan,
        )

    @staticmethod
    def _slew_onto_scan_start(
        plan: Callable[[float, float], Transition],
        fallback: tuple[float, float],
        pose: Callable[[Time], tuple[float, float] | None],
        tick: Time,
    ) -> tuple[Transition, Time, tuple[float, float]]:
        """Plan the slew to the first sample of the visit's scan.

        The visit starts at the tick. The slew is planned to the first
        sample of a visit starting then (``pose``); when it arrives in
        time, to :data:`_SLEW_ARRIVAL_TOLERANCE_SEC`, it is booked to end
        when the visit starts, the move plus a hold. When it arrives later,
        the visit starts at that arrival and the slew is planned again to
        the target read there. The slew's length changes far less than the
        time it absorbs, so the lateness shrinks from round to round (a
        constant-elevation target moves in whole samples, so one round can
        add a few hundredths of a second); after
        :data:`_SLEW_MAX_PLANS` plans the last one is booked, late. A move
        of 1 second or less counts as arriving at the tick: in the first
        round it books nothing, and in a later round the block holds the
        pose until the visit starts.

        Returns the transition to book, when the visit starts and the
        target the slew ends at. On time, the target is the one read for
        the returned start and the transition ends there. At the cap, the
        last transition is returned as planned, the visit starts at its
        arrival and the target is the one that transition was planned to,
        read for the previous start. A refused transition, in any round,
        is returned as it is with the tick as the start, for the caller to
        idle on. When the scan cannot be planned from a visit start
        (``pose`` returns None), in any round, the transition to
        ``fallback`` is returned with its arrival as the start, so such a
        visit is slewed to as it would be with no scan to aim for.
        """
        visit_start = tick
        for _ in range(_SLEW_MAX_PLANS):
            target = pose(visit_start)
            if target is None:
                transition = plan(*fallback)
                arrival = transition.arrival if transition.duration > 1.0 else tick
                return transition, arrival, fallback
            transition = plan(*target)
            if not transition.safe:
                return transition, tick, target
            arrival = transition.arrival if transition.duration > 1.0 else tick
            if (arrival - visit_start).sec <= _SLEW_ARRIVAL_TOLERANCE_SEC:
                if visit_start > tick:
                    hold_to = (visit_start - tick).sec
                    transition = dataclasses.replace(transition, duration=hold_to)
                return transition, visit_start, target
            visit_start = arrival
        return transition, visit_start, target

    @staticmethod
    def _visit_az_envelope(
        best_patch: ObservingPatch,
        plan_el: float,
        slew_az: float,
        ce_plan: tuple[Time, bool, Time, Time] | None,
        ctx: SchedulerContext,
    ) -> tuple[float, float] | None:
        """Return the azimuth envelope a constant-elevation pass sweeps, or None.

        The pass is planned at ``plan_el``, the patch's elevation, and the
        envelope is the boresight's (:func:`_ce_swept_az_envelope`).

        The wrap the slew lands on has to hold the whole scan, so the
        span handed to the wrap chooser must be where the mount goes and
        not the science window, the tick's estimate of the field's width
        about its centre, which is not the range a constant-elevation pass
        sweeps: where the field's border meets the scan elevation over its
        whole crossing. That range is available before anything is built,
        from the crossing half and the pass opening of the pass the visit
        was admitted on (``ce_plan``, see :attr:`PhaseResult.ce_plan`).

        ``None`` for every other scan type and when there is no plan:
        nothing is built yet, so the caller's scalar estimate is all there
        is. The returned pair is placed on
        ``slew_az``'s cable-wrap branch: the corridor comes on the field's
        own branch, not the one the planner places the sweep on, and the
        caller joins it and the slew's target into one contiguous span.
        """
        if best_patch.scan_type != "constant_el" or ce_plan is None:
            return None
        _, rising, t_open, _ = ce_plan
        env_lo, env_hi = _ce_swept_az_envelope(best_patch, plan_el, rising, ctx.coords, t_open)
        turns = round((slew_az - 0.5 * (env_lo + env_hi)) / 360.0)
        return env_lo + 360.0 * turns, env_hi + 360.0 * turns


def _record_executed_azimuth(
    block: TimelineBlock,
    site: Site,
    wrap_ref: float,
    *,
    follows_field: bool = False,
    sun_safe: SunSafePredicate | None = None,
    probes: dict[tuple[Any, ...], ScanBlock] | None = None,
) -> TimelineBlock | DeferralReason:
    """Re-record a science subscan's azimuths from its own trajectory.

    A subscan is emitted with the tick's scalar estimate of its azimuth
    range (:func:`~fyst_trajectories.overhead.scheduler.helpers._compute_az_range`),
    which is an estimate of the field's width about its centre and not
    where the mount goes: a constant-elevation pass sweeps the range in
    which the field's border meets the scan elevation, and its
    turnarounds. The subscan's own trajectory is therefore built here, through
    the same reconstruction path a consumer of the finished timeline
    uses, and its azimuth envelope and last sample replace the estimate,
    so a recorded block and a rebuilt one cover the same sky. The envelope
    is recorded on the turn the telescope is on: the one that puts the
    subscan's first sample nearest ``wrap_ref``, the telescope's azimuth
    where the subscan starts (where the slew ended, or where the previous
    subscan stopped), as a planet-calibration pass's is recorded in its
    slew's wrap; ``scan_params`` keep the planner's own solution, so a
    rebuild returns the trajectory in the planner's wrap.

    Returns why not, a :class:`~fyst_trajectories.overhead.DeferralReason`,
    when the subscan is not to be emitted (:func:`_screened_build`):
    ``UNPLANNABLE`` when the trajectory cannot be built (any refusal the
    reconstruction path itself would log and skip) or has two samples
    closer than the Go TCS minimum sample interval, ``SUN_SCAN`` for a
    sample inside the site's exclusion radius or ``sun_safe``'s zone, or
    ``AXIS_DYNAMICS`` for an axis past a Go TCS limit (the last three from
    the booking screen,
    :func:`~fyst_trajectories.overhead._execution_screen._execution_refusal`).
    ``probes`` is the build memo :func:`_screened_build` reads and fills. The block is returned
    unchanged when the trajectory holds no samples.
    ``az_final`` stays ``None`` when the sweep does end on the envelope's
    upper bound, leaving that bound as the recorded pose.

    With ``follows_field`` (a pong or daisy, whose pattern follows the
    field away from the elevation the block is stamped at) the last
    sample's elevation is recorded as ``el_final`` too, unless
    it equals the stamped elevation. The same build supplies it, so the
    planner is called no more often.

    The build is a probe, so its advisories are suppressed: they describe
    the patch geometry the caller already configured, and a consumer who
    rebuilds the finished timeline raises them there.
    """
    scan_block = _screened_build(block, site, sun_safe, probes)
    if isinstance(scan_block, DeferralReason):
        return scan_block
    az = scan_block.trajectory.az
    if az.size == 0:
        return block
    shift = 360.0 * round((wrap_ref - float(az[0])) / 360.0)
    env_lo, env_hi = float(np.min(az)) + shift, float(np.max(az)) + shift
    az_final = float(az[-1]) + shift
    el_final = float(scan_block.trajectory.el[-1]) if follows_field else block.elevation
    return dataclasses.replace(
        block,
        az_start=env_lo,
        az_end=env_hi,
        az_final=None if az_final == env_hi else az_final,
        el_final=None if el_final == block.elevation else el_final,
    )


def _subscan_lengths(available: float, shortest: float) -> list[float]:
    """Return the lengths a daisy or constant-elevation subscan is tried at, longest first.

    The whole ``available`` budget, then half of the previous length while
    that is longer than ``shortest``, then ``shortest`` itself. What the
    booking screen refuses in such a subscan and not in its first
    ``shortest`` seconds lies later in the scan (a daisy nearing the
    zenith, the Sun reaching a sample), so a shorter scan from the same
    start leaves it out. Halving
    finds a length within a factor of two of the longest that runs in at
    most ``log2(available / shortest) + 2`` builds, and the last attempt is
    the shortest subscan, the one :func:`_first_subscan_refusal` screened
    before the slew was booked. A subscan the first attempt books costs one
    build, as before.
    """
    if available <= shortest:
        return [available]
    lengths = [available]
    while lengths[-1] / 2.0 > shortest:
        lengths.append(lengths[-1] / 2.0)
    lengths.append(shortest)
    return lengths


def _plan_science_subscan(
    patch: ObservingPatch,
    *,
    t_start: Time,
    available: float,
    shortest: float,
    period: float | None,
    az_start: float,
    az_end: float,
    el: float,
    site: Site,
    scan_index: int,
    subscan_index: int,
    rising: bool,
    t0_scan: str | None,
    wrap_ref: float,
    sun_safe: SunSafePredicate | None = None,
    probes: dict[tuple[Any, ...], ScanBlock] | None = None,
) -> TimelineBlock | DeferralReason:
    """Build one science subscan at ``t_start`` and probe it with the planner.

    A constant-elevation or daisy subscan lasts ``available`` seconds; when
    the planner refuses it, or the booking screen does
    (:func:`_record_executed_azimuth`: the Sun, the Go TCS axis limits and
    sample interval), it is retried at the shorter lengths
    :func:`_subscan_lengths` gives, down to ``shortest``, the patch's
    shortest subscan. A pong subscan (``period`` given) holds the most whole
    pattern periods that fit ``available`` (to within
    :data:`_PERIOD_FIT_TOLERANCE_SEC`), capped by the patch's own
    ``scan_params["n_cycles"]``, and records the count as the block's
    ``n_cycles``; when it is refused it is retried with one period fewer,
    down to one.

    Returns the block, with the azimuths its own trajectory records (see
    :func:`_record_executed_azimuth`) on the turn nearest ``wrap_ref``,
    the telescope's azimuth where the subscan starts, and, for a pong or
    daisy, the elevation it ends at; or, when every
    attempt is refused, the last attempt's reason, ``UNPLANNABLE`` when
    there is no attempt. ``probes`` is the build memo
    :func:`_screened_build` reads and fills.
    """
    follows_field = patch.scan_type != "constant_el"
    refusal = DeferralReason.UNPLANNABLE
    attempts: list[tuple[float, int | None]]
    if period is None:
        attempts = [(length, None) for length in _subscan_lengths(available, shortest)]
    else:
        n_max = math.floor((available + _PERIOD_FIT_TOLERANCE_SEC) / period)
        cap = patch.scan_params.get("n_cycles")
        if cap is not None:
            n_max = min(n_max, int(cap))
        attempts = [(n * period, n) for n in range(n_max, 0, -1)]

    for duration, n_cycles in attempts:
        block = TimelineBlock.science(
            patch=patch,
            t_start=t_start,
            duration=duration,
            az_start=az_start,
            az_end=az_end,
            el=el,
            scan_index=scan_index,
            subscan_index=subscan_index,
            rising=rising,
            t0_scan=t0_scan,
        )
        if n_cycles is not None:
            metadata = dict(block.metadata)
            metadata["scan_params"] = {**block.metadata["scan_params"], "n_cycles": n_cycles}
            block = dataclasses.replace(block, metadata=metadata)
        recorded = _record_executed_azimuth(
            block, site, wrap_ref, follows_field=follows_field, sun_safe=sun_safe, probes=probes
        )
        if not isinstance(recorded, DeferralReason):
            return recorded
        refusal = recorded
    return refusal


def _plan_window_subscan(
    patch: ObservingPatch,
    *,
    t_start: Time,
    available: float,
    scan_index: int,
    subscan_index: int,
    rising: bool,
    t0_scan: str | None,
    wrap_ref: float,
    ctx: SchedulerContext,
) -> TimelineBlock | DeferralReason:
    """Plan one subscan of a patch with an azimuth window: whole window scans from ``t_start``.

    The subscan is the window scan of the most whole legs whose block
    (first sample to one sample interval after the last) lasts
    ``available`` seconds or less
    (:func:`~fyst_trajectories.overhead.scheduler.helpers._window_legs`),
    planned from ``t_start``
    (:func:`~fyst_trajectories.overhead.scheduler.helpers._window_scan`).
    When the planner or the booking screen
    (:func:`~fyst_trajectories.overhead._execution_screen._execution_refusal`)
    refuses it, it is retried with the legs of the shorter lengths
    :func:`_subscan_lengths` gives, down to the fewest legs a subscan may
    hold (:func:`~fyst_trajectories.overhead.scheduler.helpers._window_shortest_legs`).
    The caller gives an ``available`` of at least that shortest block, so
    every attempt's block fits it as estimated; the built block can run
    past the estimate by the round-off of the planner's sample grid, up to
    a few 1e-12 s, and is booked as built.

    The block is written by
    :meth:`~fyst_trajectories.overhead.TimelineBlock.from_window_scan`, so
    it records the scan and rebuilds to it sample for sample, and then ends
    one sample interval after the scan's last sample. Its azimuth envelope
    and end pose are recorded on the turn that puts its first sample
    nearest ``wrap_ref``, the telescope's azimuth where the subscan starts;
    ``rising`` is the crossing the visit watches and ``t0_scan`` the
    visit's anchor.

    Returns the block, or when every attempt is refused the last
    attempt's reason.
    """
    shortest_legs = _window_shortest_legs(patch, ctx.overhead_model)
    counts: list[int] = []
    for length in _subscan_lengths(available, _window_block_seconds(patch, shortest_legs)):
        legs = max(_window_legs(patch, length), shortest_legs)
        if legs not in counts:
            counts.append(legs)

    field_region = FieldRegion(
        ra_center=patch.ra_center,
        dec_center=patch.dec_center,
        width=patch.width,
        height=patch.height,
    )
    refusal: DeferralReason = DeferralReason.UNPLANNABLE
    for legs in counts:
        scan_block = _window_scan(patch, t_start, legs, ctx.site, ctx.ce_scan_starts)
        if scan_block is None:
            refusal = DeferralReason.UNPLANNABLE
            continue
        traj = scan_block.trajectory
        span = float(traj.times[-1]) + float(traj.times[1] - traj.times[0])
        screened = _execution_refusal(traj, "constant_el", {}, ctx.site, ctx.sun_safe)
        if screened is not None:
            refusal = screened
            continue
        try:
            block = TimelineBlock.from_window_scan(
                scan_block,
                site=ctx.site,
                patch_name=patch.name,
                scan_index=scan_index,
                subscan_index=subscan_index,
                field=field_region,
                detector_offset=patch.detector_offset,
            )
        except ValueError:
            refusal = DeferralReason.UNPLANNABLE
            continue
        shift = 360.0 * round((wrap_ref - float(traj.az[0])) / 360.0)
        env_hi = float(np.max(traj.az)) + shift
        az_final = float(traj.az[-1]) + shift
        metadata = dict(block.metadata)
        if t0_scan is not None:
            metadata["t0_scan"] = t0_scan
        return dataclasses.replace(
            block,
            t_stop=block.t_start + TimeDelta(span, format="sec"),
            az_start=float(np.min(traj.az)) + shift,
            az_end=env_hi,
            az_final=None if az_final == env_hi else az_final,
            rising=rising,
            metadata=metadata,
        )
    return refusal


def _window_stretch_legs(
    patch: ObservingPatch,
    *,
    t_start: Time,
    budgets: list[float],
    slice_seconds: float,
    overhead: OverheadModel,
    retune_due_at: Callable[[Time], bool],
) -> tuple[int, int]:
    """Return how many blocks a stretch of a window visit spans, and its scan's legs.

    A stretch starts at ``t_start`` and is one window scan. ``budgets[k - 1]``
    is the time a stretch of ``k`` blocks may take. It spans one more block
    while no retune is due where its scan would end (``retune_due_at``,
    asked at the end of the scan's block, first sample to one interval after
    the last), the scan grows by at least one leg, and the block the cut at
    ``k * slice_seconds`` leaves last lasts ``overhead.min_scan_duration``
    or more. One block is the single subscan :func:`_plan_window_subscan`
    plans.
    """
    shortest_legs = _window_shortest_legs(patch, overhead)
    shortest = overhead.min_scan_duration
    count = 1
    legs = max(_window_legs(patch, budgets[0]), shortest_legs)
    while count < len(budgets):
        end = t_start + TimeDelta(_window_block_seconds(patch, legs), format="sec")
        if retune_due_at(end):
            break
        more = _window_legs(patch, budgets[count])
        if more <= legs or _window_block_seconds(patch, more) - count * slice_seconds < shortest:
            break
        count, legs = count + 1, more
    return count, legs


def _plan_window_stretch(
    patch: ObservingPatch,
    *,
    t_start: Time,
    legs: int,
    count: int,
    slice_seconds: float,
    scan_index: int,
    subscan_index: int,
    rising: bool,
    t0_scan: str | None,
    wrap_ref: float,
    ctx: SchedulerContext,
) -> list[TimelineBlock] | None:
    """Plan one window scan of ``legs`` legs from ``t_start`` as ``count`` consecutive blocks.

    The cuts fall every ``slice_seconds`` from ``t_start``, mid-leg
    included, and the last block ends one sample interval after the scan's
    last sample. Each block is written by
    :meth:`~fyst_trajectories.overhead.TimelineBlock.from_window_scan` with
    its ``span``, so it rebuilds to its own stretch of the scan, sample for
    sample, and the booking screen
    (:func:`~fyst_trajectories.overhead._execution_screen._execution_refusal`)
    screens what each block rebuilds to. The blocks are recorded on the turn
    that puts the scan's first sample nearest ``wrap_ref``, the telescope's
    azimuth where the stretch starts.

    Returns the blocks, or ``None`` when the planner refuses the scan, the
    screen refuses a block, or a block cannot be written.
    """
    scan_block = _window_scan(patch, t_start, legs, ctx.site, ctx.ce_scan_starts)
    if scan_block is None:
        return None
    traj = scan_block.trajectory
    span = float(traj.times[-1]) + float(traj.times[1] - traj.times[0])
    edges = [t_start + TimeDelta(j * slice_seconds, format="sec") for j in range(count)]
    edges.append(t_start + TimeDelta(span, format="sec"))
    field_region = FieldRegion(
        ra_center=patch.ra_center,
        dec_center=patch.dec_center,
        width=patch.width,
        height=patch.height,
    )
    shift = 360.0 * round((wrap_ref - float(traj.az[0])) / 360.0)
    # The samples each block holds are counted from the scan's start as the
    # rebuild reads it, to the millisecond.
    stored = dataclasses.replace(
        scan_block,
        trajectory=dataclasses.replace(traj, start_time=_stored_instant(traj.start_time)),
    )
    blocks: list[TimelineBlock] = []
    for j, (start, stop) in enumerate(itertools.pairwise(edges)):
        try:
            block = TimelineBlock.from_window_scan(
                scan_block,
                site=ctx.site,
                patch_name=patch.name,
                scan_index=scan_index,
                subscan_index=subscan_index + j,
                field=field_region,
                detector_offset=patch.detector_offset,
                span=(start, stop),
            )
        except ValueError:
            return None
        held = _window_block_slice(stored, block)
        if _execution_refusal(held.trajectory, "constant_el", {}, ctx.site, ctx.sun_safe):
            return None
        metadata = dict(block.metadata)
        if t0_scan is not None:
            metadata["t0_scan"] = t0_scan
        az_final = block.end_pose_az + shift
        blocks.append(
            dataclasses.replace(
                block,
                az_start=block.az_start + shift,
                az_end=block.az_end + shift,
                az_final=None if az_final == block.az_end + shift else az_final,
                rising=rising,
                metadata=metadata,
            )
        )
    return blocks


class ScienceScanPhase(Phase):
    """Emit science subscans for the selected patch, interleaving retunes.

    Splits the remaining observable duration into subscans capped by
    ``ctx.overhead_model.max_scan_duration``, injecting a retune
    calibration block between subscans whenever the cadence tracker
    says one is due; a window below the minimum scan duration is
    skipped. The observable duration is a wall-clock budget the whole
    visit shares: the boundary retunes run inside it, so the last
    subscan is clipped rather than pushing the visit past the end of a
    constant-elevation pass, the time a pong or daisy pattern first reaches an elevation
    limit, or the Sun-safe window (for a constant-elevation patch, the
    window over its whole swept azimuth corridor; for pong and daisy,
    over every point within the pattern's reach of the field centre's
    track). For constant-elevation patches the
    rising flag comes from the crossing pass the visit was admitted on
    (:attr:`PhaseResult.ce_plan`), and the instant the selection gate
    solved that pass from is stamped on every subscan as
    ``metadata["t0_scan"]`` so each one reconstructs from the same
    anchor; other scan types take the patch's ``scan_params["rising"]``,
    else the hour-angle sign.

    A constant-elevation visit's first subscan starts at the first sample
    of its pass. The retune the visit books first is the one due where
    that subscan would otherwise start (:func:`_first_retune_spec`), so a
    retune falling due while the visit waits is booked before the wait.
    When the pass opens after that retune, the time between is booked as
    an idle block with ``metadata["reason"]`` set to ``waiting_for_pass``,
    at the pose the slew left, and comes out of the visit's budget before
    the subscans are cut from it; a pass already open is scanned from the
    retune's end. The visit's budget, and its Sun clip, run to the end of
    the pass the planner returns from the anchor, which the planner's
    whole number of legs can put after the closing crossing, and the
    visit's last block ends one sample interval after the pass's last
    sample, unless the schedule's end or the Sun comes first. A patch with
    an azimuth window is scanned instead in whole window scans from the
    field's crossing of its elevation (:meth:`_emit_subscans_with_retunes`),
    each ending within the visit's budget. Every
    subscan of the visit, and the
    boundary retune before the first, is stamped at the science
    elevation, the boresight's: for a constant-elevation patch the
    elevation the boresight scans at to hold the detector at the patch's
    ``elevation`` (the patch's own without a ``detector_offset``), and for
    a pong or daisy the elevation of the boresight that puts the detector
    on the field centre where the first subscan starts, the elevation
    :class:`SlewPhase` ended the slew at. A pong or daisy
    follows the field away from that elevation, so its subscan also
    records the elevation its trajectory ends at (``el_final``).

    A pong subscan holds a whole number of pattern periods, the most that
    fit the budget after its boundary retune (capped by the patch's own
    ``scan_params["n_cycles"]``), and records the count as
    ``scan_params["n_cycles"]``, which
    :func:`~fyst_trajectories.overhead.schedule_to_trajectories` forwards
    to :func:`~fyst_trajectories.planning.plan_pong_scan`; a rebuilt pong
    block therefore runs exactly its own length.

    Every subscan records the azimuth envelope of its own trajectory and
    the pose that trajectory ends at, on the cable-wrap turn the telescope
    is on where the subscan starts: where the slew ended, or where the
    previous subscan stopped. The tick's width estimate about the field's
    centre only places the visit when nothing is built yet. A boundary
    retune records the range of the subscan it precedes, as geometry for
    the ECSV round trip rather than an executed envelope.

    Each subscan is planned before its boundary retune is booked. A
    subscan the planner refuses, or the booking screen refuses
    (:func:`~fyst_trajectories.overhead._execution_screen._execution_refusal`:
    a sample inside the Sun zone, the site's exclusion radius or the
    context's ``sun_safe`` model, or an axis past a Go TCS limit, which PCS
    refuses too, or two samples closer than the Go TCS minimum sample
    interval), is first retried shorter: a pong with one period fewer,
    down to one, a daisy or constant-elevation subscan at the lengths
    :func:`_subscan_lengths` gives, down to the shortest subscan, the one
    :class:`SlewPhase` screened before it booked the slew. A subscan refused
    at every length is not emitted, nor is its retune, and the visit ends
    there. A visit that emits nothing idles one tick, with
    ``metadata["reason"]`` set to ``unplannable`` when the planner refused
    its first subscan, the screen refused it for its sample interval or a pong
    period no longer fits after the slew, to ``sun_scan`` or
    ``axis_dynamics`` when the screen refused it for the Sun or an axis,
    and to ``sun_point`` when the pose held before it would be inside the
    Sun zone.
    """

    def run(
        self,
        state: SchedulerState,
        ctx: SchedulerContext,
        *,
        selection: PhaseResult | None = None,
    ) -> PhaseResult:
        """Emit science subscans + inter-subscan retune blocks."""
        best_patch, best_az, best_el = _unpack_selection(selection, "ScienceScanPhase")

        gate_el = best_patch.elevation if best_patch.elevation is not None else best_el
        # The pass the visit was admitted on; a constant-elevation visit with
        # none left has no time to scan.
        ce_plan = _selected_ce_plan(selection, gate_el, state, ctx)
        # A pong or daisy pattern has to be inside the elevation limits only
        # once its subscan starts, after the boundary retune booked first.
        due = _first_retune_spec(
            state, ctx, state.current_time, None if ce_plan is None else ce_plan[2]
        )
        retune_lead = 0.0 if due is None else due.duration

        open_at = None
        pass_end = None
        if ce_plan is not None:
            anchor, rising, t_open, t_close = ce_plan
            t0_scan = anchor.isot
            sci_start = state.current_time + TimeDelta(retune_lead, format="sec")
            if _ce_window(best_patch) is not None:
                # A window visit is scanned from the field's crossing of its
                # elevation, in whole window scans that end inside its budget.
                open_at = _window_open_at(t_open, sci_start)
            else:
                provisional = _provisional_ce_scan(
                    best_patch,
                    elevation=gate_el,
                    rising=rising,
                    t0_scan=t0_scan,
                    scan_start=sci_start,
                    ctx=ctx,
                )
                if provisional is not None:
                    # The visit's last block ends one sample after the pass's
                    # last sample, and a pass that opens after the retune is
                    # waited for: the subscans share what is left from its
                    # opening.
                    open_at, pass_end = _ce_pass_bounds(provisional, sci_start)
                    # A pass the planner rounds up to whole legs ends after its
                    # closing crossing; the budget and the Sun clip run to its
                    # end.
                    if (pass_end - t_close).sec > _SAMPLE_SPAN_TOLERANCE_SEC:
                        t_close = pass_end
            ce_plan = (anchor, rising, t_open, t_close)
        else:
            ha = ctx.coords.get_hour_angle(best_patch.ra_center, state.current_time)
            rising = best_patch.scan_params.get("rising", ha < 0.0)
            t0_scan = None

        if best_patch.scan_type == "constant_el" and ce_plan is None:
            scan_duration = 0.0
        else:
            scan_duration = _compute_scan_duration(
                best_patch,
                state.current_time,
                ctx.end_time,
                ctx.site,
                ctx.coords,
                ctx.overhead_model,
                best_el,
                ce_cache=ctx.ce_corridors,
                ce_ready_lead=_ce_ready_lead(state, ctx),
                sun_safe=ctx.sun_safe,
                retune_lead=retune_lead,
                ce_plan=None if ce_plan is None else ce_plan[1:],
            )

        if scan_duration < ctx.overhead_model.min_scan_duration:
            return _emit_idle_tick(state, ctx)

        # The observable duration is a WALL-CLOCK budget, not a science-only
        # one: it ends when the crossing pass ends, the pattern reaches an
        # elevation limit or the Sun catches it. Boundary retunes run inside
        # it, so the visit is capped at this deadline.
        deadline = min(
            ctx.end_time,
            state.current_time + TimeDelta(scan_duration, format="sec"),
        )
        if pass_end is not None:
            deadline = min(deadline, pass_end)
        if open_at is not None:
            scan_duration -= (open_at - sci_start).sec

        n_subscans = max(1, math.ceil(scan_duration / ctx.overhead_model.max_scan_duration))
        subscan_duration = scan_duration / n_subscans

        az_start_sci, az_end_sci = _compute_az_range(best_patch, best_az, best_el, ctx.site)

        state, sub_blocks, refusal = self._emit_subscans_with_retunes(
            state=state,
            ctx=ctx,
            best_patch=best_patch,
            best_el=best_el,
            n_subscans=n_subscans,
            subscan_duration=subscan_duration,
            rising=rising,
            az_start_sci=az_start_sci,
            az_end_sci=az_end_sci,
            t0_scan=t0_scan,
            deadline=deadline,
            open_at=open_at,
            first_retune=due,
        )

        if not sub_blocks:
            # No block fit (a boundary retune plus a minimum subscan, or a
            # pong period, no longer fits), or the first subscan was refused
            # (the planner, the booking screen, or the Sun zone reaching the
            # pose held before it): idle one tick, labelled with the refusal,
            # so the loop advances and the timeline stays tiled.
            return _emit_idle_tick(state, ctx, reason=refusal)

        # The visit leaves the mount wherever its last sweep stopped, which
        # is generally neither envelope bound, and, for a pong or daisy,
        # below or above the elevation the visit is stamped at; the next
        # slew is priced, wrap-chosen and Sun-checked from that pose. A
        # retune is only booked when the subscan after it fits, so a
        # non-empty visit always ends on a science block.
        science_blocks = [b for b in sub_blocks if b.block_type == BlockType.SCIENCE]
        state = state.advanced(
            current_az=science_blocks[-1].end_pose_az,
            current_el=science_blocks[-1].end_pose_el,
            scan_counter=state.scan_counter + 1,
        )

        return PhaseResult(state=state, blocks=sub_blocks)

    @staticmethod
    def _emit_subscans_with_retunes(
        *,
        state: SchedulerState,
        ctx: SchedulerContext,
        best_patch: ObservingPatch,
        best_el: float,
        n_subscans: int,
        subscan_duration: float,
        rising: bool,
        az_start_sci: float,
        az_end_sci: float,
        deadline: Time,
        first_retune: CalibrationSpec | None,
        t0_scan: str | None = None,
        open_at: Time | None = None,
    ) -> tuple[SchedulerState, list[TimelineBlock], str | None]:
        """Emit ``n_subscans`` science blocks with retunes at scan boundaries.

        Retune rule: **before every subscan**, a retune is booked when one
        is due (always, under the cadence-0 "every scan boundary"
        convention), and ``current_time`` advances past it before the
        subscan starts. Before the first subscan that retune is
        ``first_retune``, the visit's own decision
        (:func:`_first_retune_spec`); before each later one, the cadence
        tracker is asked at that boundary. A boundary retune is booked
        only together with the subscan after it: a minimum-duration subscan
        must still fit after it before ``deadline``, and that subscan, built
        at its post-retune start, must be one the planner plans and the
        booking screen passes. So a
        visit can never end on a dangling retune or push one past the window it
        was scored against.

        A patch with an azimuth window is scanned in stretches, each one
        window scan of whole legs from the stretch's start. A stretch runs
        over the subscans up to the first boundary at which a retune falls
        due (:func:`_window_stretch_legs`) and is cut into blocks every
        ``subscan_duration``, mid-leg included (:func:`_plan_window_stretch`),
        so blocks with no retune between them are consecutive slices of one
        scan and the path runs on from one into the next. A stretch of one
        subscan is one whole scan (:func:`_plan_window_subscan`): every
        subscan but the last takes the most whole legs that fit the slice,
        the last the most that fit what the visit has left, up to
        ``max_scan_duration``; a stretch whose blocks the planner or the
        booking screen refuses is planned as that one subscan instead. A
        new scan starts at the window's low end moving up, and only after a
        retune: between the two the telescope goes from one scan's last
        sample to the next scan's first, inside the retune, and no move is
        booked for it. A boundary with no retune due that a stretch does not
        cover ends the visit.

        For a pong the minimum subscan is one pattern period (or
        ``min_scan_duration``, whichever is longer), and a subscan holds
        the most whole periods that fit what the deadline leaves after its
        retune, up to the slice ``subscan_duration``. A subscan refused at
        every length :func:`_plan_science_subscan` tries (a pong with one
        period fewer down to one, a daisy or constant-elevation subscan
        down to the shortest) is not emitted, nor is its retune, and the
        visit ends there.

        ``open_at``, when given, is the first sample of a
        constant-elevation pass that opens after the first retune: the
        first subscan starts there, and the time between the retune's end
        (or the visit's start, with no retune due) and ``open_at`` is
        booked as an idle block labelled ``waiting_for_pass`` at the pose
        the slew left. Every boundary retune records the pose the telescope
        holds through it as its end pose (``az_final``): the slew's end
        before the first subscan, the previous subscan's end after that.

        ``deadline`` is the end of the visit's observable window (the
        schedule end, the end of the crossing pass, for a field-form pass
        one sample interval after its last sample, the point where the
        pattern reaches an elevation limit or the Sun catches it, whichever
        comes first). Every block emitted
        here, retunes included, fits inside it: the retunes
        share the budget with the science rather than extending it, so the
        final subscan is clipped to whatever the retunes left.

        A retune that becomes due during the final subscan is not injected
        here; the next visit's leading boundary (or, for nonzero cadences,
        :class:`CalibrationPhase`) picks it up.

        The retune and the wait before a subscan hold the telescope still,
        so they are booked only when the pose they hold stays outside the
        Sun zone (the context's model, or the site's scalar radius) until
        the subscan starts; otherwise the visit ends before them, as at a
        refused subscan, and the loop plans again from where the telescope
        is.

        Returns
        -------
        state : SchedulerState
            The state advanced past every emitted block.
        blocks : list of TimelineBlock
            The emitted retune, wait and science blocks, possibly empty.
        refusal : str or None
            When nothing was emitted: ``"unplannable"`` when the planner
            refused the first subscan, the booking screen refused its
            sample interval or a pong period no longer fits the window,
            ``"sun_scan"`` or ``"axis_dynamics"`` when the booking screen
            refuses it for the Sun or an axis, ``"sun_point"`` when the pose held before it
            is inside the Sun zone during the hold; ``None`` otherwise.
        """
        blocks: list[TimelineBlock] = []
        unplannable = str(DeferralReason.UNPLANNABLE)

        period = _pong_period(best_patch) if best_patch.scan_type == "pong" else None
        min_subscan = _min_subscan_duration(best_patch, ctx.overhead_model)

        if subscan_duration < min_subscan:
            # Splitting produced sub-minimum slices (for a pong, slices
            # shorter than one period); nothing can be emitted.
            return state, blocks, None if period is None else unplannable

        # Every subscan is stamped at the science elevation, where the slew
        # drove (the boresight's), and so is the retune before the first; a
        # retune after a subscan holds the elevation it ended at, which a
        # pong or daisy records as its ``el_final``.
        sci_el = best_el if best_patch.elevation is None else _ce_scan_pose(best_patch, ctx.site)[1]
        pose_el = sci_el
        # Each subscan is recorded on the turn the telescope is on when it
        # starts: where the slew left it, then where the previous one stopped.
        pose_az = state.current_az

        window = _ce_window(best_patch) is not None
        sub_idx = 0
        while sub_idx < n_subscans:
            retune_spec = first_retune if sub_idx == 0 else _due_retune_spec(state, ctx)
            if window and sub_idx > 0 and retune_spec is None:
                # A window scan runs on to where a retune falls due, so a
                # boundary with none due ends the visit: a new scan from the
                # window's low end is never started without a retune before it.
                break
            retune_dur = retune_spec.duration if retune_spec is not None else 0.0
            sci_start = state.current_time + TimeDelta(retune_dur, format="sec")
            # The first subscan of a visit whose pass opens after the retune
            # starts at the pass's first sample; the time between is a wait.
            scan_start = open_at if sub_idx == 0 and open_at is not None else sci_start
            wait = (scan_start - sci_start).sec
            remaining = (deadline - scan_start).sec
            # The boundary retune, any wait and a minimum-duration subscan
            # must all fit inside the window; otherwise the visit ends here.
            if remaining < min_subscan:
                if not blocks and period is not None:
                    return state, blocks, unplannable
                break

            # The retune and the wait hold the pose the slew or the previous
            # subscan left; one the Sun zone reaches before the subscan starts
            # ends the visit here, before anything of this subscan is booked.
            held = (scan_start - state.current_time).sec
            if held > 0.0 and not _holds(ctx, pose_az, pose_el, state.current_time, held):
                if not blocks:
                    return state, blocks, str(DeferralReason.SUN_POINT)
                break

            # Plan the subscan at its start before booking the retune, so a
            # refused subscan leaves neither block behind.
            planned: TimelineBlock | list[TimelineBlock] | DeferralReason
            if window:
                # A stretch of k subscans may take k slices; one that holds
                # the last subscan takes what the visit has left, up to the
                # longest subscan for that one.
                budgets = [
                    min(
                        j * subscan_duration
                        + (
                            ctx.overhead_model.max_scan_duration
                            if sub_idx + j == n_subscans - 1
                            else subscan_duration
                        ),
                        remaining,
                    )
                    for j in range(n_subscans - sub_idx)
                ]
                # The cadence as it stands once this subscan's retune is booked.
                tuned = (
                    state
                    if retune_spec is None
                    else state.advanced(
                        cal_state=state.cal_state.update("retune", state.current_time)
                    )
                )

                def retune_due_at(t: Time, tuned: SchedulerState = tuned) -> bool:
                    return _due_retune_spec(tuned.advanced(current_time=t), ctx) is not None

                count, legs = _window_stretch_legs(
                    best_patch,
                    t_start=scan_start,
                    budgets=budgets,
                    slice_seconds=subscan_duration,
                    overhead=ctx.overhead_model,
                    retune_due_at=retune_due_at,
                )
                stretch = (
                    _plan_window_stretch(
                        best_patch,
                        t_start=scan_start,
                        legs=legs,
                        count=count,
                        slice_seconds=subscan_duration,
                        scan_index=state.scan_counter,
                        subscan_index=sub_idx,
                        rising=rising,
                        t0_scan=t0_scan,
                        wrap_ref=pose_az,
                        ctx=ctx,
                    )
                    if count > 1
                    else None
                )
                planned = (
                    stretch
                    if stretch is not None
                    else _plan_window_subscan(
                        best_patch,
                        t_start=scan_start,
                        available=budgets[0],
                        scan_index=state.scan_counter,
                        subscan_index=sub_idx,
                        rising=rising,
                        t0_scan=t0_scan,
                        wrap_ref=pose_az,
                        ctx=ctx,
                    )
                )
            else:
                planned = _plan_science_subscan(
                    best_patch,
                    t_start=scan_start,
                    available=min(subscan_duration, remaining),
                    shortest=min_subscan,
                    period=period,
                    az_start=az_start_sci,
                    az_end=az_end_sci,
                    el=sci_el,
                    site=ctx.site,
                    scan_index=state.scan_counter,
                    subscan_index=sub_idx,
                    rising=rising,
                    t0_scan=t0_scan,
                    wrap_ref=pose_az,
                    sun_safe=ctx.sun_safe,
                    probes=ctx.science_probes,
                )
            if isinstance(planned, DeferralReason):
                return state, blocks, None if blocks else str(planned)
            science = planned if isinstance(planned, list) else [planned]

            if retune_spec is not None:
                # The retune records the range of the subscan after it, at
                # the elevation the telescope holds through it, and ends at
                # the pose it holds: where the slew or the previous subscan
                # left it.
                state, retune_block = _emit_retune_block(
                    state, retune_spec, science[0].az_start, science[0].az_end, pose_el
                )
                if retune_block.end_pose_az != pose_az:
                    retune_block = dataclasses.replace(retune_block, az_final=pose_az)
                blocks.append(retune_block)
            if wait > 0.0:
                blocks.append(
                    TimelineBlock.idle(
                        t_start=state.current_time,
                        duration=wait,
                        az=pose_az,
                        el=pose_el,
                        scan_index=state.scan_counter,
                        reason=str(DeferralReason.WAITING_FOR_PASS),
                    )
                )
            blocks.extend(science)
            pose_az, pose_el = science[-1].end_pose_az, science[-1].end_pose_el
            state = state.advanced(current_time=science[-1].t_stop)
            sub_idx += len(science)

        return state, blocks, None
