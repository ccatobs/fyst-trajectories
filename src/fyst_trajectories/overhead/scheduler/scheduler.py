"""The :class:`Scheduler` class.

Runs the scheduler loop, terminating when the schedule window is
exhausted or remaining time falls below the minimum scan duration.
"""

from __future__ import annotations

from ..calibration_state import CalibrationState
from ..models import ObservingTimeline, TimelineBlock
from ..transitions import DeferralReason
from .phases import (
    CalibrationPhase,
    PatchSelectionPhase,
    ScienceScanPhase,
    SlewPhase,
    _escape_if_overtaken,
)
from .state import SchedulerContext, SchedulerState

__all__ = ["Scheduler"]

# Shortest tail worth a block of its own, in seconds. Below this the
# remainder is the float noise of a window boundary, not a gap.
_TAIL_TOLERANCE_SEC = 0.01


class Scheduler:
    """Orchestrates scheduler phases to build a timeline.

    The scheduler holds a :class:`SchedulerContext` and runs
    the four phases in sequence per iteration:

    0. A pose the Sun zone has overtaken is moved out first
       (:func:`~fyst_trajectories.overhead.plan_escape`, a SLEW block
       named ``sun_escape``), so no phase parks inside the zone.
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
    science, calibration, slew and idle totals add up to it.

    ``Scheduler.run()`` returns the completed
    :class:`~fyst_trajectories.overhead.ObservingTimeline`. Downstream
    code should call :func:`~fyst_trajectories.overhead.generate_timeline`
    as the public entry point; direct ``Scheduler`` use is for callers
    that assemble the :class:`SchedulerContext` themselves. The phase
    sequence is fixed: ``run()`` instantiates the four phases and there
    is no hook for a different list. Callers needing another loop (a
    body queue, lookahead, multi-night stitching) compose the phases or
    the timeline model directly, as the calibration-night planner does.

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

    Parameters
    ----------
    context : SchedulerContext
        The scheduling context (patches, site, coords, models,
        constraints, time window); phases read it and write only its
        crossing-pass and escape memos.
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

            # A pose the Sun zone has overtaken is moved out before any
            # phase parks a calibration or idles there.
            escaped = _escape_if_overtaken(state, ctx)
            if escaped is not None:
                blocks.extend(escaped.blocks)
                state = escaped.state
                if escaped.stop:
                    break
                continue

            cal_result = cal_phase.run(state, ctx)
            blocks.extend(cal_result.blocks)
            state = cal_result.state

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
            if science_result.skip_to_next_iter:
                continue

        # Whatever the loop exited on, the window is the timeline's
        # declared extent, so the stretch after the last block belongs to
        # a block too: without this the four time totals do not add up to
        # total_time. The tail is under a minimum scan duration in the
        # ordinary case and one refused move in the others.
        tail = (ctx.end_time - state.current_time).sec
        if tail > _TAIL_TOLERANCE_SEC:
            blocks.append(
                TimelineBlock.idle(
                    t_start=state.current_time,
                    duration=tail,
                    az=state.current_az,
                    el=state.current_el,
                    site=ctx.site,
                    scan_index=state.scan_counter,
                    reason=str(DeferralReason.WINDOW_CLOSED),
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
