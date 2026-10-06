"""Context and state builders shared by the scheduler phase tests."""

from astropy.time import Time

from fyst_trajectories import get_fyst_site
from fyst_trajectories.overhead import CalibrationPolicy, CalibrationState, OverheadModel
from fyst_trajectories.overhead.scheduler import SchedulerContext, SchedulerState


def _make_ctx(
    patches,
    *,
    start_time="2026-06-15T02:00:00",
    end_time="2026-06-15T10:00:00",
    overhead_model=None,
    calibration_policy=None,
    time_step=300.0,
    sun_safe=None,
    constraints=None,
):
    """Build a context with sensible defaults for phase-level tests."""
    return SchedulerContext.build(
        patches=patches,
        site=get_fyst_site(),
        start_time=Time(start_time, scale="utc"),
        end_time=Time(end_time, scale="utc"),
        overhead_model=overhead_model or OverheadModel(),
        calibration_policy=calibration_policy or CalibrationPolicy(),
        constraints=constraints,
        time_step=time_step,
        sun_safe=sun_safe,
    )


def _initial_state(ctx):
    return SchedulerState.initial(start_time=ctx.start_time, cal_state=CalibrationState())
