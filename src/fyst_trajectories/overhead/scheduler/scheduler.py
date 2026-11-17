"""The :class:`Scheduler` class.

Runs the scheduler loop, terminating when the schedule window is
exhausted or remaining time falls below the minimum scan duration.
"""

from __future__ import annotations

from .._moves import plan_escape_move
from ..calibration_state import CalibrationState
from ..models import ObservingTimeline, TimelineBlock
from ..transitions import _TAIL_TOLERANCE_SEC, DeferralReason
from .phases import (
    CalibrationPhase,
    PatchSelectionPhase,
    ScienceScanPhase,
    SlewPhase,
    _escape_if_overtaken,
    _loop_top_hold,
)
from .state import SchedulerContext, SchedulerState

__all__ = ["Scheduler"]


class Scheduler:
    """Orchestrates scheduler phases to build a timeline.

    The scheduler holds a :class:`SchedulerContext` and runs
    the four phases in sequence per iteration:

    0. A pose the Sun zone overtakes before the first calibration due now
       (or, with none due, one scheduler tick) would end is moved out first
       (:func:`~fyst_trajectories.overhead.plan_escape`, a SLEW block
       named ``sun_escape``), so no phase starts inside the zone; each
       phase that holds the pose, a calibration or an idle, then asks
       again for the time it holds it, so none parks inside the zone.
    1. :class:`CalibrationPhase`: emit any due calibration blocks.
    2. :class:`PatchSelectionPhase`: pick the best observable patch,
       or emit an IDLE block if none are observable.
    3. :class:`SlewPhase`: emit a slew block if the telescope needs
       to move.
    4. :class:`ScienceScanPhase`: emit science subscans with
       interleaved retunes.

    Whatever the loop exits on, the stretch between the last block and
    the end of the window is filled with one idle labelled
    ``window_closed``, so the timeline tiles its declared extent and the
    science, calibration, slew and idle totals add up to it. When the Sun
    zone would reach the pose before the window closes, the telescope is
    moved out first and the idle is held where the escape ends; when it
    cannot be moved in time the idle is labelled ``no_escape``.

    ``Scheduler.run()`` returns the completed
    :class:`~fyst_trajectories.overhead.ObservingTimeline`. Downstream
    code should call :func:`~fyst_trajectories.overhead.generate_timeline`
    as the public entry point; direct ``Scheduler`` use is for callers
    that assemble the :class:`SchedulerContext` themselves. The phase
    sequence is fixed: ``run()`` instantiates the four phases and there
    is no hook for a different list. Callers needing another loop (a
    body queue, non-greedy selection, multi-night stitching) compose the
    phases or the timeline model directly, as the calibration-night
    planner does.

    Parameters
    ----------
    context : SchedulerContext
        The scheduling context (patches, site, coords, models,
        constraints, time window); phases read it and write only its four
        memos, ``ce_corridors``, ``escapes``, ``ce_scan_starts`` and
        ``science_probes``.

    Notes
    -----
    The loop is **greedy and single-pass**: calibrations check at the
    top of each iteration but cannot interrupt or trim a science scan
    in progress. A calibration that becomes due mid-scan is deferred
    to the next iteration, and an end-of-window calibration may be
    skipped entirely if a patch change pre-empts it. Critical
    calibrations (e.g. an opening pointing scan) should be checked
    against the returned timeline rather than assumed; in operations
    they are typically inserted by hand at the boundaries.
    """

    def __init__(self, context: SchedulerContext) -> None:
        self.context = context

    def run(self) -> ObservingTimeline:
        """Run the scheduler and return the completed timeline."""
        ctx = self.context
        state = SchedulerState.initial(start_time=ctx.start_time, cal_state=CalibrationState())
        blocks: list[TimelineBlock] = []

        cal_phase = CalibrationPhase()
        selection_phase = PatchSelectionPhase()
        slew_phase = SlewPhase()
        science_phase = ScienceScanPhase()

        while state.current_time.unix < ctx.end_time.unix:
            remaining = (ctx.end_time - state.current_time).sec
            if remaining < ctx.overhead_model.min_scan_duration:
                break

            # A pose the Sun zone overtakes before the next block it would
            # hold ends is moved out before any phase parks a calibration or
            # idles there.
            escaped = _escape_if_overtaken(state, ctx, hold=_loop_top_hold(state, ctx))
            if escaped is not None:
                blocks.extend(escaped.blocks)
                state = escaped.state
                if escaped.stop:
                    break
                continue

            cal_result = cal_phase.run(state, ctx)
            blocks.extend(cal_result.blocks)
            state = cal_result.state
            if cal_result.stop:
                break
            if cal_result.skip_to_next_iter:
                continue

            if state.current_time.unix >= ctx.end_time.unix:
                break

            selection_result = selection_phase.run(state, ctx)
            blocks.extend(selection_result.blocks)
            state = selection_result.state
            if selection_result.stop:
                # The idle path's escape would end past the window.
                break
            if selection_result.skip_to_next_iter:
                continue

            slew_result = slew_phase.run(state, ctx, selection=selection_result)
            blocks.extend(slew_result.blocks)
            state = slew_result.state
            if slew_result.stop:
                break
            if slew_result.skip_to_next_iter:
                continue

            science_result = science_phase.run(state, ctx, selection=slew_result)
            blocks.extend(science_result.blocks)
            state = science_result.state

        # Whatever the loop exited on, the window is the timeline's
        # declared extent, so the stretch after the last block belongs to
        # a block too: without this the four time totals do not add up to
        # total_time. The tail is under a minimum scan duration in the
        # ordinary case and one refused move in the others.
        tail = (ctx.end_time - state.current_time).sec
        reason = DeferralReason.WINDOW_CLOSED
        if tail > _TAIL_TOLERANCE_SEC:
            # The tail holds the pose too: one the zone reaches before the
            # window closes is moved out first and the rest is held where the
            # escape ends, or, when it cannot be moved in time, the tail
            # records that the zone holds it.
            move = plan_escape_move(
                state.current_az,
                state.current_el,
                state.current_time,
                ctx.site,
                sun_safe=ctx.sun_safe,
                slew_safe=ctx.slew_safe,
                settle_time=ctx.overhead_model.settle_time,
                el_floor=ctx.el_floor,
                remaining=tail,
                scan_index=state.scan_counter,
                cache=ctx.escapes,
                hold=tail,
            )
            if move.block is not None:
                assert move.transition is not None
                blocks.append(move.block)
                state = state.advanced(
                    current_time=move.transition.arrival,
                    current_az=move.transition.az_to,
                    current_el=move.transition.el_to,
                )
                tail = (ctx.end_time - state.current_time).sec
            elif move.label is not None:
                reason = move.label
        if tail > _TAIL_TOLERANCE_SEC:
            blocks.append(
                TimelineBlock.idle(
                    t_start=state.current_time,
                    duration=tail,
                    az=state.current_az,
                    el=state.current_el,
                    scan_index=state.scan_counter,
                    reason=str(reason),
                )
            )

        return ObservingTimeline(
            blocks=blocks,
            site=ctx.site,
            start_time=ctx.start_time,
            end_time=ctx.end_time,
            overhead_model=ctx.overhead_model,
            calibration_policy=ctx.calibration_policy,
            metadata={
                "n_patches": len(ctx.patches),
                "time_step": ctx.time_step,
            },
        )
