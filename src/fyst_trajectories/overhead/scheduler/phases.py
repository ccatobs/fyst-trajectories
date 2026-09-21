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
import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from astropy.time import Time, TimeDelta

from ...exceptions import PointingError, PointingWarning
from ...planning import plan_source_ces_passes
from .._moves import plan_escape_move
from ..models import (
    BlockType,
    CalibrationSpec,
    CalibrationType,
    ObservingPatch,
    TimelineBlock,
    validate_scan_params,
)
from ..simulation import _generate_trajectory_for_block
from ..transitions import DeferralReason, plan_transition
from ..utils import compute_nasmyth_rotation
from .helpers import (
    _CE_READY_SLEW_ALLOWANCE_SEC,
    _ce_swept_az_envelope,
    _ce_visit_plan,
    _compute_az_range,
    _compute_scan_duration,
    _evaluate_patch,
    _normalize_az,
)
from .state import SchedulerState

if TYPE_CHECKING:
    from ...planning import ScanBlock
    from ...site import Site
    from ..models import SourceCESScanParams
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
        Azimuth of ``selection`` at ``state.current_time``, in degrees,
        placed in the telescope's cable-wrap window on the 360-degree
        representative nearest ``state.current_az``. A replacement
        selection phase must preserve that placement:
        :class:`SlewPhase` measures direct mount travel from the
        current pose, which is only meaningful when both azimuths
        share one coherent frame. Consumed by :class:`SlewPhase` and
        :class:`ScienceScanPhase`.
    best_el : float | None
        Instantaneous elevation of ``selection`` at ``state.current_time``,
        in degrees.
    skip_to_next_iter : bool
        If True, the scheduler should restart its outer loop (used by
        :class:`PatchSelectionPhase` when it emits an idle block, by
        :class:`SlewPhase` when the transition is refused, and by later
        phases when the scan duration falls below the minimum).
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
        # Defensive: PatchSelectionPhase guarantees these are populated whenever
        # ``selection`` is populated. We re-check here so the invariant survives
        # ``python -O`` (which strips ``assert``).
        raise RuntimeError(
            f"{phase_name} received a PhaseResult with selection populated but "
            "best_az/best_el unset; this indicates a PatchSelectionPhase bug."
        )
    return selection.selection, selection.best_az, selection.best_el


def _escape_if_overtaken(state: SchedulerState, ctx: SchedulerContext) -> PhaseResult | None:
    """Move a telescope the Sun zone has overtaken, before it idles or slews.

    A pose that is inside the zone now (under the context's model, or
    the site's scalar radius) has no commandable transition, since every
    path from it starts unsafe; the antenna's own avoidance system moves
    such a telescope, and the simulator models that move here with
    :func:`~fyst_trajectories.overhead._moves.plan_escape_move`, the
    decision both offline loops share. Returns ``None`` when the pose is
    safe. Otherwise emits the escape as a SLEW block named
    ``sun_escape``, moves the state to its pose and restarts the loop
    (the next tick re-selects from a safe pose), or, when the zone holds
    the telescope, one idle tick labelled ``no_escape``. An escape that
    would end past the window stops the loop; the remaining sliver is
    filled by the loop's own tail idle.

    Called from three places in one tick (the loop top, the idle emitter
    and the slew phase), each of which must be covered: the clock can
    advance inside a tick, so a later call is not always the same
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
    given. A telescope the Sun zone has overtaken never idles in place:
    :func:`_escape_if_overtaken` moves it first, or labels the tick
    ``no_escape`` when it cannot.
    """
    escaped = _escape_if_overtaken(state, ctx)
    if escaped is not None:
        return escaped
    return _idle_tick(state, ctx, reason=reason)


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
        site=ctx.site,
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
    site: Site,
) -> tuple[TimelineBlock, Time, float, float]:
    """Build one planet-calibration CALIBRATION block from a source-CES pass.

    Returns ``(block, t_stop, az_final, el_bore)``. ``t_stop`` is the
    pass's ``t1``; the caller chains it as the next block's ``t_start`` so
    the blocks tile with no gaps. ``az_final`` / ``el_bore`` give the pass
    end pose the caller carries forward as the scheduler's current
    position. The sequence is one scan, so the passes share ``scan_index``
    and are told apart by ``subscan_index``, their position in it.

    The block's ``t_start`` is supplied by the caller (the scheduler clock
    for the first pass, the previous pass's ``t1`` afterwards), so the
    inter-pass repointing gap and the anchor-to-scan lead fold into the
    block as acquisition time. The true scan start is recorded in
    ``metadata["t0_scan"]``. The azimuth bounds are the honest executed
    envelope: the min/max over the pass trajectory's azimuth samples
    (drift included), read from the planned trajectory itself rather
    than re-derived from the scalar parameters. The drag ends on
    whichever leg endpoint its last turnaround left it on, generally
    neither bound, so that azimuth is recorded separately as the block's
    ``az_final``.
    """
    cp = pass_block.computed_params
    pp = pass_block.trajectory.metadata.pattern_params

    t0_iso = str(cp["t0_iso"])
    t1_iso = str(cp["t1_iso"])
    t_stop = Time(t1_iso, scale="utc")
    el_bore = float(cp["el_bore"])
    mode = str(cp["mode"])

    env_lo = float(np.min(pass_block.trajectory.az))
    env_hi = float(np.max(pass_block.trajectory.az))
    az_final = float(pass_block.trajectory.az[-1])

    scan_params: SourceCESScanParams = {
        "body": str(cal_spec.target),
        "footprint": footprint,
        "el_bore": el_bore,
        "mode": mode,
        "window": [t0_iso, t1_iso],
        "boresight_rot": float(cp["boresight_rot"]),
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
        site=site,
        scan_index=scan_index,
        target=cal_spec.target,
        az_end=env_hi,
        az_final=az_final,
        subscan_index=subscan_index,
        scan_params=scan_params,
        t0_scan=t0_iso,
        rising=(mode == "rising"),
    )
    return block, t_stop, az_final, el_bore


def _emit_planet_cal_passes(
    state: SchedulerState,
    ctx: SchedulerContext,
    cal_spec: CalibrationSpec,
) -> tuple[bool, list[TimelineBlock], SchedulerState]:
    """Plan a planet calibration as a multi-pass source-CES sequence.

    Returns ``(emitted, blocks, new_state)``. ``emitted`` is ``False`` when
    the sequence is infeasible (any
    :class:`~fyst_trajectories.exceptions.PointingError`) or when no pass finishes
    before ``ctx.end_time``; in that case ``blocks`` is empty and
    ``new_state`` is the unchanged input state. The caller then neither
    emits nor marks the cadence, so the calibration stays due and is
    retried on the next scheduler iteration, exactly like a planet cal with
    no visible planet.

    On success the blocks tile ``[state.current_time, last_pass_t1]`` with
    no gaps, ``new_state`` advances ``current_time`` to the last pass's
    ``t1`` and ``current_az`` / ``current_el`` to its end pose, and the
    planet-cal cadence is marked at the (pre-scan) ``state.current_time``,
    matching the parked path. The whole sequence is one scan: the blocks
    share ``state.scan_counter`` and carry their position in the sequence
    as ``subscan_index``, so each pass has its own identity.
    """
    policy = ctx.calibration_policy

    kwargs = dict(
        body=cal_spec.target,
        footprint=policy.planet_cal_footprint,
        n_passes=policy.planet_cal_passes,
        start_time=state.current_time,
        site=ctx.site,
        timestep=_SOURCE_CES_TIMESTEP_SEC,
        # The injected sun model reaches the planet-cal planner too (None
        # keeps the planner's scalar default).
        sun_safe=ctx.sun_safe,
    )
    if policy.planet_cal_el_step is not None:
        kwargs["el_step"] = policy.planet_cal_el_step

    try:
        passes = plan_source_ces_passes(**kwargs)
    except PointingError:
        return False, [], state

    # End-of-night: keep only whole passes that finish before the window
    # closes. n_passes in the recorded scan_params stays the requested total
    # so a truncated sequence is visible. Both ends have to clear the window:
    # ``t1_iso`` is what the emitted block's ``t_stop`` is set to, while the
    # trajectory runs for the leg-quantised duration from ``t0_iso``, and
    # quantisation moves the two apart in either direction by up to half a
    # leg plus turnaround.
    kept = [
        b
        for b in passes
        if max(
            Time(str(b.computed_params["t1_iso"]), scale="utc").unix,
            Time(str(b.computed_params["t0_iso"]), scale="utc").unix + float(b.duration),
        )
        <= ctx.end_time.unix
    ]
    if not kept:
        return False, [], state

    blocks: list[TimelineBlock] = []
    t_start = state.current_time
    end_az = state.current_az
    end_el = state.current_el
    for position, pass_block in enumerate(kept):
        block, t_stop, end_az, end_el = _planet_cal_pass_block(
            pass_block,
            cal_spec=cal_spec,
            footprint=policy.planet_cal_footprint,
            t_start=t_start,
            scan_index=state.scan_counter,
            subscan_index=position,
            site=ctx.site,
        )
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


def _emit_retune_block(
    state: SchedulerState,
    ctx: SchedulerContext,
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
        site=ctx.site,
        scan_index=state.scan_counter,
    )
    state = state.advanced(
        cal_state=state.cal_state.update("retune", state.current_time),
        current_time=state.current_time + TimeDelta(spec.duration, format="sec"),
    )
    return state, block


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
    :class:`ScienceScanPhase` immediately before each subscan instead.

    Clamps each cal block's duration against the remaining schedule
    window so no block extends past ``ctx.end_time``. Stops early if
    the schedule window is exhausted partway through the burst.

    When ``CalibrationPolicy.planet_cal_scan`` is set, a due
    ``planet_cal`` is instead planned as a multi-pass source-CES
    sequence anchored at the scheduler clock (one CALIBRATION block per
    pass). An infeasible sequence is skipped and left due, so the
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

            # Scan-coupled retunes (cadence 0) belong to ScienceScanPhase's
            # subscan boundaries, not this per-tick path; only the startup
            # burst (last_retune is None) fires one here so the night
            # begins tuned.
            if (
                cal_spec.name == CalibrationType.RETUNE
                and ctx.calibration_policy.retune_cadence == 0.0
                and state.cal_state.last_retune is not None
            ):
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
            cal_block = TimelineBlock.calibration(
                cal_type=cal_spec.name,
                t_start=state.current_time,
                duration=cal_duration,
                az=state.current_az,
                el=state.current_el,
                site=ctx.site,
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
    patch.priority``. The highest-scoring observable patch wins.

    A patch that pins ``elevation`` is scored at the azimuth the field
    is at now and the elevation the scan will command, which is the
    pose the rest of the visit is judged at (:class:`SlewPhase` slews
    there and every emitted block records it). The field's own
    elevation still decides when it sets.

    If no patch scores above zero, emits an IDLE block advancing by
    ``ctx.time_step`` and sets ``skip_to_next_iter=True``: the outer
    scheduler should skip the slew/science phases for this iteration.
    A pose the Sun zone has overtaken is moved out first instead of
    idling in place (see :class:`SlewPhase`).

    Otherwise, returns no blocks but populates ``selection``,
    ``best_az``, ``best_el`` in the result so downstream phases can
    consume them. ``best_az`` is placed in the telescope's cable-wrap
    window on the representative nearest the current pose, so the
    slew and boresight math downstream operate in one coherent frame.
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

        for patch in ctx.patches:
            az, el = ctx.coords.radec_to_altaz(
                patch.ra_center, patch.dec_center, state.current_time
            )
            check_el = patch.elevation if patch.elevation is not None else el
            score = _evaluate_patch(
                patch, state.current_time, az, check_el, ctx.coords, ctx.constraints
            )
            score *= patch.weight / patch.priority

            # Honor a requested elevation crossing: a patch whose
            # scan_params pins "rising" is only selectable while the sky
            # side matches (hour angle < 0 for rising, > 0 for setting).
            # Without this a setting request would be scheduled at a
            # rising-side time and the planner's geometry/timing would
            # decohere from the request.
            if score > 0.0 and "rising" in patch.scan_params:
                ha = ctx.coords.get_hour_angle(patch.ra_center, state.current_time)
                if bool(patch.scan_params["rising"]) != (ha < 0.0):
                    score = 0.0

            # A constant-elevation patch is selectable only while a crossing
            # pass is still plannable from now (the same forward solve
            # plan_constant_el_scan runs at reconstruction). Without this
            # gate the hour-angle default keeps the patch scoring until
            # transit, hours after the pass's opening crossing, and every
            # block emitted there is unreconstructable (on a single-patch
            # night that can lose most of the CE blocks).
            if score > 0.0 and patch.scan_type == "constant_el":
                gate_el = patch.elevation if patch.elevation is not None else el
                plan = _ce_visit_plan(
                    patch,
                    gate_el,
                    state.current_time,
                    ctx.end_time,
                    ctx.coords,
                    ctx.ce_corridors,
                    ctx.time_step + _CE_READY_SLEW_ALLOWANCE_SEC,
                )
                if plan is None:
                    score = 0.0

            if score > best_score:
                best_score = score
                best_patch = patch
                best_az = az
                best_el = el

        if best_patch is None or best_score == 0.0:
            return _emit_idle_tick(state, ctx)

        # Normalize the winning azimuth into the telescope's cable-wrap window
        # before it flows to SlewPhase / ScienceScanPhase. Raw astropy azimuth
        # is in [0, 360); leaving it unnormalized inflates a north-straddling
        # slew distance and flips the slew boresight angle by ~180 deg.
        return PhaseResult(
            state=state,
            blocks=[],
            selection=best_patch,
            best_az=_normalize_az(best_az, ctx.site, ref=state.current_az),
            best_el=best_el,
        )


class SlewPhase(Phase):
    """Plan the slew to the selected patch and emit its block.

    The move from ``(state.current_az, state.current_el)`` to the science
    pose is planned with
    :func:`~fyst_trajectories.overhead.plan_transition`: the encoder wrap
    is chosen so the whole azimuth envelope the visit will occupy stays
    inside the azimuth limits (:meth:`_visit_az_envelope`), the
    direct path is clear of the Sun zone (the context's ``sun_safe``
    model, or the site's scalar radius) and the pose still clears it one
    scheduler tick after arrival, and the duration is the kinematic
    estimate plus the overhead model's settle time. Holding the goal for a
    tick is what keeps the loop from slewing to a pose the next tick would
    escape from. The science pose is
    ``selection.best_az`` at ``patch.elevation`` when the patch pins one,
    else ``selection.best_el``; when the scan's azimuth range lives on a
    different cable-wrap branch than the field centre (an explicit
    ``az_min``/``az_max`` window across the wrap, or a range
    branch-shifted into the limits), the slew targets the range midpoint
    instead, so the unwind onto the scan's branch is modelled rather than
    silently skipped.

    A SLEW block is emitted when the planned move exceeds 1 second;
    ``current_time`` then advances by its duration and ``current_az`` /
    ``current_el`` move to the slew target, so blocks emitted before the
    next science scan (idles, calibrations) carry the post-slew pose. A
    move under the threshold emits nothing and leaves the state untouched.
    When the transition lands on a wrap other than the one the selection
    placed (the near path blocked, the far one clear), ``best_az`` is
    carried forward shifted onto that wrap so the science range downstream
    is derived in the same frame.

    A refused transition (every wrap Sun-blocked, no clear direct path,
    the range fitting no wrap, the elevation outside the limits) never
    raises: the patch is treated as unselectable for this tick, an IDLE
    block carrying the refusal as ``metadata["reason"]`` is emitted, and
    the loop restarts, so the same patch is re-tried once the Sun has
    moved. No detour is attempted. A patch whose scan range is pinned to
    one placement (an explicit window) is refused the same way when only
    another wrap is reachable, since the science range cannot follow.

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
        # the leg into a pinned-elevation scan unmodelled.
        slew_el = best_patch.elevation if best_patch.elevation is not None else best_el

        # The same deterministic range ScienceScanPhase computes. When it
        # sits a full turn from the field centre (explicit wrap-crossing
        # window, or a branch shift into the limits), slew to the range
        # midpoint so the unwind onto the scan's branch is a modelled
        # move; otherwise keep the field centre target unchanged.
        az_lo_sci, az_hi_sci = _compute_az_range(best_patch, best_az, best_el, ctx.site)
        mid_sci = 0.5 * (az_lo_sci + az_hi_sci)
        slew_az = best_az if abs(mid_sci - best_az) <= 180.0 else mid_sci

        env_lo, env_hi = self._visit_az_envelope(
            best_patch, slew_el, az_lo_sci, az_hi_sci, slew_az, state, ctx
        )

        # The wrap must hold the slew target and the whole azimuth
        # envelope the visit occupies.
        transition = plan_transition(
            state.current_az,
            state.current_el,
            slew_az,
            slew_el,
            state.current_time,
            ctx.site,
            sun_safe=ctx.sun_safe,
            slew_safe=ctx.slew_safe,
            goal_az_span=(min(env_lo, slew_az), max(env_hi, slew_az)),
            settle_time=ctx.overhead_model.settle_time,
            hold=ctx.time_step,
        )
        if not transition.safe:
            return _emit_idle_tick(state, ctx, reason=str(transition.cause))

        # A wrap other than the placed one moves the science range too;
        # the range is re-derived from the shifted centre, and a range
        # that cannot follow (an explicit window pins its placement)
        # makes the tick unselectable as well.
        shift = transition.az_shift
        if shift != 0.0:
            shifted = _compute_az_range(best_patch, best_az + shift, best_el, ctx.site)
            if not np.allclose(shifted, (az_lo_sci + shift, az_hi_sci + shift), atol=1e-6):
                return _emit_idle_tick(state, ctx, reason=str(DeferralReason.NO_WRAP))
            best_az += shift

        blocks: list[TimelineBlock] = []
        if transition.duration > 1.0:
            slew_end = state.current_time + TimeDelta(transition.duration, format="sec")
            if slew_end.unix >= ctx.end_time.unix:
                return PhaseResult(
                    state=state,
                    blocks=[],
                    selection=best_patch,
                    best_az=best_az,
                    best_el=best_el,
                    stop=True,
                )
            slew_block = TimelineBlock.slew(
                t_start=state.current_time,
                duration=transition.duration,
                az_start=state.current_az,
                az_end=transition.az_to,
                el=transition.el_to,
                site=ctx.site,
                scan_index=state.scan_counter,
                patch_name=f"slew_to_{best_patch.name}",
            )
            blocks.append(slew_block)
            state = state.advanced(
                current_time=slew_end,
                current_az=transition.az_to,
                current_el=transition.el_to,
            )

        return PhaseResult(
            state=state,
            blocks=blocks,
            selection=best_patch,
            best_az=best_az,
            best_el=best_el,
        )

    @staticmethod
    def _visit_az_envelope(
        best_patch: ObservingPatch,
        slew_el: float,
        az_lo_sci: float,
        az_hi_sci: float,
        slew_az: float,
        state: SchedulerState,
        ctx: SchedulerContext,
    ) -> tuple[float, float]:
        """Return the azimuth envelope the visit will occupy, for the wrap check.

        The wrap the slew lands on has to hold the whole scan, so the
        span handed to the wrap chooser must be where the mount goes and
        not the science window, which for a drifting constant-elevation
        pass is tens of degrees narrower than the corridor the pass
        sweeps. That corridor is available before anything is built,
        from the pass the selection tick already solved and memoized.

        Falls back to the science window for every other scan type and
        whenever the pass is no longer plannable from here: nothing is
        built yet, so a scalar estimate is all there is. The returned
        pair is placed on ``slew_az``'s cable-wrap branch, since the
        planner picks its own branch for the corridor and the two are
        combined into one contiguous span by the caller.
        """
        if best_patch.scan_type != "constant_el":
            return az_lo_sci, az_hi_sci
        plan = _ce_visit_plan(
            best_patch,
            slew_el,
            state.current_time,
            ctx.end_time,
            ctx.coords,
            ctx.ce_corridors,
            ctx.time_step + _CE_READY_SLEW_ALLOWANCE_SEC,
        )
        if plan is None:
            return az_lo_sci, az_hi_sci
        env_lo, env_hi = _ce_swept_az_envelope(best_patch, plan[1], plan[2], ctx.coords)
        turns = round((slew_az - 0.5 * (env_lo + env_hi)) / 360.0)
        return env_lo + 360.0 * turns, env_hi + 360.0 * turns


def _record_executed_azimuth(block: TimelineBlock, site: Site) -> TimelineBlock:
    """Re-record a science subscan's azimuths from its own trajectory.

    A subscan is emitted with the tick's scalar estimate of its azimuth
    range (:func:`~fyst_trajectories.overhead.scheduler.helpers._compute_az_range`),
    which is the instantaneous field width and not where the mount goes:
    a drifting constant-elevation pass sweeps a corridor tens of degrees
    wider. The subscan's own trajectory is therefore built here, through
    the same reconstruction path a consumer of the finished timeline
    uses, and its azimuth envelope and last sample replace the estimate,
    so a recorded block and a rebuilt one cannot disagree. This is the
    same rule the swept calibration blocks follow.

    The block is returned unchanged when the trajectory cannot be built
    (the same best-effort contract the reconstruction path itself keeps)
    or when it holds no samples. ``az_final`` stays ``None`` when the
    sweep does end on the envelope's upper bound, leaving that bound as
    the recorded pose.

    The build is a probe, so its advisories are suppressed: they describe
    the patch geometry the caller already configured, and a consumer who
    rebuilds the finished timeline raises them there.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PointingWarning)
            scan_block = _generate_trajectory_for_block(block, site)
    except (ValueError, KeyError, TypeError):
        return block
    az = scan_block.trajectory.az
    if az.size == 0:
        return block
    env_lo, env_hi = float(np.min(az)), float(np.max(az))
    az_final = float(az[-1])
    return dataclasses.replace(
        block,
        az_start=env_lo,
        az_end=env_hi,
        az_final=None if az_final == env_hi else az_final,
        # The factory evaluates the boresight angle at the midpoint of the
        # recorded range, so the widened range moves it too.
        boresight_angle=compute_nasmyth_rotation(0.5 * (env_lo + env_hi), block.elevation, site),
    )


class ScienceScanPhase(Phase):
    """Emit science subscans for the selected patch, interleaving retunes.

    Splits the remaining observable duration into subscans capped by
    ``ctx.overhead_model.max_scan_duration``, injecting a retune
    calibration block between subscans whenever the cadence tracker
    says one is due; a window below the minimum scan duration is
    skipped. The observable duration is a wall-clock budget the whole
    visit shares: the boundary retunes run inside it, so the last
    subscan is clipped rather than pushing the visit past the pass
    close, the field's set time or the Sun-safe window it was scored
    against. For constant-elevation patches the rising flag comes from
    the crossing pass chosen by ``_ce_visit_plan``, and the visit's own
    start time is stamped on every subscan as ``metadata["t0_scan"]`` so
    each one reconstructs from the same anchor; other scan types fall
    back to the hour-angle sign. Every block of the visit,
    subscans and boundary retunes alike, is stamped at the science
    elevation: the patch's pinned ``elevation`` when it has one,
    otherwise the field centre's elevation at selection time.

    Every subscan records the azimuth envelope of its own trajectory and
    the pose that trajectory ends at (:func:`_record_executed_azimuth`).
    The tick's scalar range estimate only places the visit and labels the
    boundary retunes: a retune sweeps nothing, and the range it inherits
    is geometry for the ECSV round trip rather than an executed envelope.
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

        scan_duration = _compute_scan_duration(
            best_patch,
            state.current_time,
            ctx.end_time,
            ctx.site,
            ctx.coords,
            ctx.overhead_model,
            best_el,
            ce_cache=ctx.ce_corridors,
            ce_ready_lead=ctx.time_step + _CE_READY_SLEW_ALLOWANCE_SEC,
            sun_safe=ctx.sun_safe,
        )

        ce_plan = None
        if best_patch.scan_type == "constant_el":
            gate_el = best_patch.elevation if best_patch.elevation is not None else best_el
            # Cache-hit re-fetch of the pass _compute_scan_duration just
            # solved with identical arguments, to obtain the chosen half
            # and the visit anchor for stamping.
            ce_plan = _ce_visit_plan(
                best_patch,
                gate_el,
                state.current_time,
                ctx.end_time,
                ctx.coords,
                ctx.ce_corridors,
                ctx.time_step + _CE_READY_SLEW_ALLOWANCE_SEC,
            )

        if scan_duration < ctx.overhead_model.min_scan_duration:
            advance = min(ctx.time_step, (ctx.end_time - state.current_time).sec)
            new_state = state.advanced(
                current_time=state.current_time + TimeDelta(advance, format="sec"),
            )
            return PhaseResult(state=new_state, blocks=[], skip_to_next_iter=True)

        n_subscans = max(1, math.ceil(scan_duration / ctx.overhead_model.max_scan_duration))
        subscan_duration = scan_duration / n_subscans

        # The observable duration is a WALL-CLOCK budget, not a science-only
        # one: it ends when the crossing pass closes, when the field sets,
        # or when the Sun catches it. Boundary retunes run inside it, so the
        # visit is capped at this deadline rather than letting each retune
        # push the remaining subscans past the window they were scored
        # against.
        deadline = min(
            ctx.end_time,
            state.current_time + TimeDelta(scan_duration, format="sec"),
        )

        if ce_plan is not None:
            rising = ce_plan[0]
            t0_scan = state.current_time.isot
        else:
            ha = ctx.coords.get_hour_angle(best_patch.ra_center, state.current_time)
            rising = best_patch.scan_params.get("rising", ha < 0.0)
            t0_scan = None

        az_start_sci, az_end_sci = _compute_az_range(best_patch, best_az, best_el, ctx.site)

        state, sub_blocks = self._emit_subscans_with_retunes(
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
        )

        if not sub_blocks:
            # The room checks ended the visit before any block fit (for
            # example a sliver window where the boundary retune plus a
            # minimum-duration subscan no longer fit together); tick
            # forward like the too-short-scan path so the outer loop
            # cannot spin in place.
            advance = min(ctx.time_step, (ctx.end_time - state.current_time).sec)
            new_state = state.advanced(
                current_time=state.current_time + TimeDelta(advance, format="sec"),
            )
            return PhaseResult(state=new_state, blocks=[], skip_to_next_iter=True)

        # The visit leaves the mount wherever its last sweep stopped, which
        # is generally neither envelope bound; the next slew is priced,
        # wrap-chosen and Sun-checked from that pose. A retune is only
        # booked when the subscan after it fits, so a non-empty visit
        # always ends on a science block.
        science_blocks = [b for b in sub_blocks if b.block_type == BlockType.SCIENCE]
        state = state.advanced(
            current_az=science_blocks[-1].end_pose_az,
            current_el=best_el if best_patch.elevation is None else best_patch.elevation,
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
        t0_scan: str | None = None,
    ) -> tuple[SchedulerState, list[TimelineBlock]]:
        """Emit ``n_subscans`` science blocks with retunes at scan boundaries.

        Retune rule: **before every subscan**, query the cadence tracker;
        if a retune is due (always, under the cadence-0 "every scan
        boundary" convention), emit it and advance ``current_time`` before
        the subscan starts. A boundary retune is booked only when a
        minimum-duration subscan still fits after it before ``deadline``,
        so a visit can never end on a dangling retune or push one past the
        window it was scored against.

        ``deadline`` is the end of the visit's observable window (the
        schedule end, or the point where the crossing pass closes, the
        field sets or the Sun catches it, whichever comes first). Every
        block emitted here, retunes included, fits inside it: the retunes
        share the budget with the science rather than extending it, so the
        final subscan is clipped to whatever the retunes left.

        A retune that becomes due during the final subscan is not injected
        here; the next visit's leading boundary (or, for nonzero cadences,
        :class:`CalibrationPhase`) picks it up.
        """
        blocks: list[TimelineBlock] = []

        if subscan_duration < ctx.overhead_model.min_scan_duration:
            # Splitting produced sub-minimum slices; nothing can be emitted.
            return state, blocks

        # The whole visit sits at the science elevation (the slew already
        # drove there), so boundary retunes are stamped at it too.
        sci_el = best_el if best_patch.elevation is None else best_patch.elevation

        for sub_idx in range(n_subscans):
            remaining = (deadline - state.current_time).sec
            retune_spec = _due_retune_spec(state, ctx)
            retune_dur = retune_spec.duration if retune_spec is not None else 0.0
            # The boundary retune plus a minimum-duration subscan must both
            # fit inside the window; otherwise the visit ends here.
            if remaining < retune_dur + ctx.overhead_model.min_scan_duration:
                break
            if retune_spec is not None:
                state, retune_block = _emit_retune_block(
                    state, ctx, retune_spec, az_start_sci, az_end_sci, sci_el
                )
                blocks.append(retune_block)

            actual_duration = min(subscan_duration, (deadline - state.current_time).sec)

            science_block = TimelineBlock.science(
                patch=best_patch,
                t_start=state.current_time,
                duration=actual_duration,
                az_start=az_start_sci,
                az_end=az_end_sci,
                el=sci_el,
                site=ctx.site,
                scan_index=state.scan_counter,
                subscan_index=sub_idx,
                rising=rising,
                t0_scan=t0_scan,
            )
            science_block = _record_executed_azimuth(science_block, ctx.site)
            blocks.append(science_block)
            state = state.advanced(
                current_time=state.current_time + TimeDelta(actual_duration, format="sec"),
            )

        return state, blocks
