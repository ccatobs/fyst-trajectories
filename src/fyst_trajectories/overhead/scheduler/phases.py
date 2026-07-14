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

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from astropy.time import Time, TimeDelta

from ...exceptions import PointingError
from ...planning import plan_source_ces_passes
from ..models import (
    CalibrationSpec,
    CalibrationType,
    ObservingPatch,
    TimelineBlock,
    validate_scan_params,
)
from ..utils import estimate_slew_time
from .helpers import _compute_az_range, _compute_scan_duration, _evaluate_patch, _normalize_az
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
        observable — skip downstream science phases".
    best_az : float | None
        Instantaneous azimuth of ``selection`` at ``state.current_time``,
        in degrees. Consumed by :class:`SlewPhase` and
        :class:`ScienceScanPhase`.
    best_el : float | None
        Instantaneous elevation of ``selection`` at ``state.current_time``,
        in degrees.
    skip_to_next_iter : bool
        If True, the scheduler should restart its outer loop (used by
        :class:`PatchSelectionPhase` when it emits an idle block, and by
        later phases when the scan duration falls below the minimum).
    stop : bool
        If True, the scheduler should break out of its outer loop
        (used by :class:`SlewPhase` when the slew would extend past
        ``ctx.end_time``).
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


class Phase:
    """Abstract base class for scheduler phases.

    A phase transforms :class:`SchedulerState` in response to the
    current :class:`SchedulerContext`, optionally emitting blocks.
    Subclasses override :meth:`run`.
    """

    def run(self, state: SchedulerState, ctx: SchedulerContext) -> PhaseResult:
        """Execute the phase and return its result.

        Parameters
        ----------
        state : SchedulerState
            Current scheduler state.
        ctx : SchedulerContext
            Read-only scheduling context.

        Returns
        -------
        PhaseResult
            Emitted blocks plus the evolved state.
        """
        raise NotImplementedError


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
    site: Site,
) -> tuple[TimelineBlock, Time, float, float]:
    """Build one planet-calibration CALIBRATION block from a source-CES pass.

    Returns ``(block, t_stop, az_end, el_bore)``. ``t_stop`` is the pass's
    ``t1``; the caller chains it as the next block's ``t_start`` so the
    blocks tile with no gaps. ``az_end`` / ``el_bore`` give the pass end
    pose the caller carries forward as the scheduler's current position.

    The block's ``t_start`` is supplied by the caller (the scheduler clock
    for the first pass, the previous pass's ``t1`` afterwards), so the
    inter-pass repointing gap and the anchor-to-scan lead fold into the
    block as acquisition time. The true scan start is recorded in
    ``metadata["t0_scan"]``. The azimuth bounds are the honest executed
    envelope: the min/max over the pass trajectory's azimuth samples
    (drift included), read from the planned trajectory itself rather
    than re-derived from the scalar parameters.
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
        scan_params=scan_params,
        t0_scan=t0_iso,
        rising=(mode == "rising"),
    )
    return block, t_stop, env_hi, el_bore


def _emit_planet_cal_passes(
    state: SchedulerState,
    ctx: SchedulerContext,
    cal_spec: CalibrationSpec,
) -> tuple[bool, list[TimelineBlock], SchedulerState]:
    """Plan a planet calibration as a multi-pass source-CES sequence.

    Returns ``(emitted, blocks, new_state)``. ``emitted`` is ``False`` when
    the sequence is infeasible (any
    :class:`~fyst_trajectories.PointingError`) or when no pass finishes
    before ``ctx.end_time``; in that case ``blocks`` is empty and
    ``new_state`` is the unchanged input state. The caller then neither
    emits nor marks the cadence, so the calibration stays due and is
    retried on the next scheduler iteration, exactly like a planet cal with
    no visible planet.

    On success the blocks tile ``[state.current_time, last_pass_t1]`` with
    no gaps, ``new_state`` advances ``current_time`` to the last pass's
    ``t1`` and ``current_az`` / ``current_el`` to its end pose, and the
    planet-cal cadence is marked at the (pre-scan) ``state.current_time``,
    matching the parked path.
    """
    policy = ctx.calibration_policy

    kwargs = dict(
        body=cal_spec.target,
        footprint=policy.planet_cal_footprint,
        n_passes=policy.planet_cal_passes,
        start_time=state.current_time,
        site=ctx.site,
        timestep=_SOURCE_CES_TIMESTEP_SEC,
    )
    if policy.planet_cal_el_step is not None:
        kwargs["el_step"] = policy.planet_cal_el_step

    try:
        passes = plan_source_ces_passes(**kwargs)
    except PointingError:
        return False, [], state

    # End-of-night: keep only whole passes that finish before the window
    # closes. n_passes in the recorded scan_params stays the requested total
    # so a truncated sequence is visible.
    kept = [
        b
        for b in passes
        if Time(str(b.computed_params["t1_iso"]), scale="utc").unix <= ctx.end_time.unix
    ]
    if not kept:
        return False, [], state

    blocks: list[TimelineBlock] = []
    t_start = state.current_time
    end_az = state.current_az
    end_el = state.current_el
    for pass_block in kept:
        block, t_stop, end_az, end_el = _planet_cal_pass_block(
            pass_block,
            cal_spec=cal_spec,
            footprint=policy.planet_cal_footprint,
            t_start=t_start,
            scan_index=state.scan_counter,
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


class CalibrationPhase(Phase):
    """Emit any calibration blocks whose cadence has elapsed.

    Queries the context's calibration policy and the state's
    :class:`CalibrationState` to determine which calibrations are due
    at ``state.current_time``. Emits each due calibration as a
    CALIBRATION block, updates the cadence tracker, and advances
    ``current_time`` past each block.

    Clamps each cal block's duration against the remaining schedule
    window so no block extends past ``ctx.end_time``. Stops early if
    the schedule window is exhausted partway through the burst.

    When ``CalibrationPolicy.planet_cal_scan`` is set, a due
    ``planet_cal`` is instead planned as a multi-pass source-CES
    sequence anchored at the scheduler clock (one CALIBRATION block per
    pass). An infeasible sequence is skipped and left due, so the
    calibration retries on a later iteration.
    """

    def run(self, state: SchedulerState, ctx: SchedulerContext) -> PhaseResult:
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

    If no patch scores above zero, emits an IDLE block advancing by
    ``ctx.time_step`` and sets ``skip_to_next_iter=True`` — the outer
    scheduler should skip the slew/science phases for this iteration.

    Otherwise, returns no blocks but populates ``selection``,
    ``best_az``, ``best_el`` in the result so downstream phases can
    consume them.
    """

    def run(self, state: SchedulerState, ctx: SchedulerContext) -> PhaseResult:
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

            if score > best_score:
                best_score = score
                best_patch = patch
                best_az = az
                best_el = el

        if best_patch is None or best_score == 0.0:
            advance = min(ctx.time_step, (ctx.end_time - state.current_time).sec)
            idle_block = TimelineBlock.idle(
                t_start=state.current_time,
                duration=advance,
                az=state.current_az,
                el=state.current_el,
                site=ctx.site,
                scan_index=state.scan_counter,
            )
            new_state = state.advanced(
                current_time=state.current_time + TimeDelta(advance, format="sec"),
            )
            return PhaseResult(
                state=new_state,
                blocks=[idle_block],
                skip_to_next_iter=True,
            )

        # Normalize the winning azimuth into the telescope's cable-wrap window
        # before it flows to SlewPhase / ScienceScanPhase. Raw astropy azimuth
        # is in [0, 360); leaving it unnormalized inflates a north-straddling
        # slew distance and flips the slew boresight angle by ~180 deg.
        return PhaseResult(
            state=state,
            blocks=[],
            selection=best_patch,
            best_az=_normalize_az(best_az, ctx.site),
            best_el=best_el,
        )


class SlewPhase(Phase):
    """Emit a slew block if the telescope needs to move.

    Uses :func:`estimate_slew_time` to compute the move time from
    ``(state.current_az, state.current_el)`` to ``(selection.best_az,
    selection.best_el)``, adds the overhead model's settle time, and
    emits a SLEW block if the total exceeds 1 second. Advances
    ``current_time`` by the slew duration.

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
        """Emit a slew block, if needed, carrying selection forward."""
        best_patch, best_az, best_el = _unpack_selection(selection, "SlewPhase")

        slew_time = estimate_slew_time(
            state.current_az, state.current_el, best_az, best_el, ctx.site
        )
        slew_time += ctx.overhead_model.settle_time

        blocks: list[TimelineBlock] = []
        if slew_time > 1.0:
            slew_end = state.current_time + TimeDelta(slew_time, format="sec")
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
                duration=slew_time,
                az_start=state.current_az,
                az_end=best_az,
                el=best_el,
                site=ctx.site,
                scan_index=state.scan_counter,
                patch_name=f"slew_to_{best_patch.name}",
            )
            blocks.append(slew_block)
            state = state.advanced(current_time=slew_end)

        return PhaseResult(
            state=state,
            blocks=blocks,
            selection=best_patch,
            best_az=best_az,
            best_el=best_el,
        )


class ScienceScanPhase(Phase):
    """Emit science subscans for the selected patch, interleaving retunes.

    1. Compute the remaining observable scan duration via
       :func:`_compute_scan_duration`.
    2. If the duration falls below the minimum scan duration, skip the
       scan: advance ``current_time`` by ``ctx.time_step`` and set
       ``skip_to_next_iter=True``.
    3. Otherwise split the scan into ``n_subscans`` (capped by
       ``ctx.overhead_model.max_scan_duration``), compute the rising
       flag from the hour angle, and compute the ordered-bounds
       azimuth range ``(az_start, az_end)``.
    4. Emit subscans back-to-back via
       :meth:`_emit_subscans_with_retunes`, which injects a retune
       calibration block between subscans whenever the cadence tracker
       says a retune is due.
    5. Advance ``current_az`` / ``current_el`` to the scan's final
       pose and increment ``scan_counter``.
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
        )

        if scan_duration < ctx.overhead_model.min_scan_duration:
            advance = min(ctx.time_step, (ctx.end_time - state.current_time).sec)
            new_state = state.advanced(
                current_time=state.current_time + TimeDelta(advance, format="sec"),
            )
            return PhaseResult(state=new_state, blocks=[], skip_to_next_iter=True)

        n_subscans = max(1, math.ceil(scan_duration / ctx.overhead_model.max_scan_duration))
        subscan_duration = scan_duration / n_subscans

        ha = ctx.coords.get_hour_angle(best_patch.ra_center, state.current_time)
        rising = best_patch.scan_params.get("rising", ha < 0.0)

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
        )

        state = state.advanced(
            current_az=az_end_sci,
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
    ) -> tuple[SchedulerState, list[TimelineBlock]]:
        """Emit ``n_subscans`` science blocks with retunes between them.

        Retune interleaving rule: after every subscan **except the last**,
        query the cadence tracker for a retune; if one is due, emit a
        CALIBRATION retune block and advance ``current_time`` by the
        retune duration before starting the next subscan.

        Note that retunes are only checked **between** subscans. A retune
        that becomes due during the final subscan is not injected here;
        it will be handled by the :class:`CalibrationPhase` on the next
        outer-loop iteration.

        Breaks early if the schedule window is exhausted mid-scan.
        """
        blocks: list[TimelineBlock] = []

        for sub_idx in range(n_subscans):
            if state.current_time.unix >= ctx.end_time.unix:
                break

            actual_duration = min(subscan_duration, (ctx.end_time - state.current_time).sec)
            if actual_duration < ctx.overhead_model.min_scan_duration:
                break

            sci_el = best_el if best_patch.elevation is None else best_patch.elevation
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
            )
            blocks.append(science_block)
            state = state.advanced(
                current_time=state.current_time + TimeDelta(actual_duration, format="sec"),
            )

            if sub_idx < n_subscans - 1:
                retune_needed = state.cal_state.needs_calibration(
                    state.current_time,
                    ctx.calibration_policy,
                    ctx.overhead_model,
                    coords=ctx.coords,
                )
                retune_specs = [c for c in retune_needed if c.name == "retune"]
                if retune_specs:
                    retune_dur = retune_specs[0].duration
                    retune_block = TimelineBlock.retune(
                        t_start=state.current_time,
                        duration=retune_dur,
                        az_start=az_start_sci,
                        az_end=az_end_sci,
                        el=best_el,
                        site=ctx.site,
                        scan_index=state.scan_counter,
                    )
                    blocks.append(retune_block)
                    state = state.advanced(
                        cal_state=state.cal_state.update("retune", state.current_time),
                        current_time=state.current_time + TimeDelta(retune_dur, format="sec"),
                    )

        return state, blocks
