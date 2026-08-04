"""The :class:`Scheduler` class.

Runs the scheduler loop, terminating when the schedule window is
exhausted or remaining time falls below the minimum scan duration.
"""

from __future__ import annotations

from ..models import ObservingTimeline, TimelineBlock
from ..overhead import CalibrationState
from .phases import (
    CalibrationPhase,
    PatchSelectionPhase,
    ScienceScanPhase,
    SlewPhase,
)
from .state import SchedulerContext, SchedulerState

__all__ = ["Scheduler"]


class Scheduler:
    """Orchestrates scheduler phases to build a timeline.

    The scheduler holds a read-only :class:`SchedulerContext` and runs
    the four phases in sequence per iteration:

    1. :class:`CalibrationPhase`: emit any due calibration blocks.
    2. :class:`PatchSelectionPhase`: pick the best observable patch,
       or emit an IDLE block if none are observable.
    3. :class:`SlewPhase`: emit a slew block if the telescope needs
       to move.
    4. :class:`ScienceScanPhase`: emit science subscans with
       interleaved retunes.

    ``Scheduler.run()`` returns the completed
    :class:`~fyst_trajectories.overhead.ObservingTimeline`. Downstream
    code should call :func:`~fyst_trajectories.overhead.generate_timeline`
    as the public entry point; direct ``Scheduler`` use is for advanced
    extensions (custom phase lists, lookahead, multi-night stitching).

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
        The read-only scheduling context (patches, site, coords,
        models, constraints, time window).
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

            cal_result = cal_phase.run(state, ctx)
            blocks.extend(cal_result.blocks)
            state = cal_result.state

            if state.current_time.unix >= ctx.end_time.unix:
                break

            selection_result = selection_phase.run(state, ctx)
            blocks.extend(selection_result.blocks)
            state = selection_result.state
            if selection_result.skip_to_next_iter:
                continue

            slew_result = slew_phase.run(state, ctx, selection=selection_result)
            blocks.extend(slew_result.blocks)
            state = slew_result.state
            if slew_result.stop:
                break

            science_result = science_phase.run(state, ctx, selection=slew_result)
            blocks.extend(science_result.blocks)
            state = science_result.state
            if science_result.skip_to_next_iter:
                continue

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
