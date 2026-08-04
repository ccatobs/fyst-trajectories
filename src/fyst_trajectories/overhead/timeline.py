"""Public :func:`generate_timeline` entry point."""

from typing import TYPE_CHECKING

from astropy.time import Time

from ..site import Site
from .constraints import Constraint

if TYPE_CHECKING:
    from ..dispatch import SunSafePredicate
from .models import (
    CalibrationPolicy,
    ObservingPatch,
    ObservingTimeline,
    OverheadModel,
)
from .scheduler import Scheduler, SchedulerContext

__all__ = [
    "generate_timeline",
]


def generate_timeline(
    patches: list[ObservingPatch],
    site: Site,
    start_time: Time | str,
    end_time: Time | str,
    overhead_model: OverheadModel | None = None,
    calibration_policy: CalibrationPolicy | None = None,
    constraints: list[Constraint] | None = None,
    time_step: float = 300.0,
    sun_safe: "SunSafePredicate | None" = None,
) -> ObservingTimeline:
    """Generate an observing timeline.

    At each time step, evaluates all patches, selects the highest-scoring
    one, schedules a science scan, and advances. Calibration operations
    are injected between scans when cadence thresholds are exceeded.

    Parameters
    ----------
    patches : list of ObservingPatch
        Sky regions to observe.
    site : Site
        Observatory site configuration.
    start_time : Time or str
        Timeline start time (UTC). Strings are auto-parsed.
    end_time : Time or str
        Timeline end time (UTC).
    overhead_model : OverheadModel or None
        Overhead timing parameters. Uses defaults if None.
    calibration_policy : CalibrationPolicy or None
        Calibration cadence policy. Uses defaults if None.
    constraints : list of Constraint or None
        Scheduling constraints. If None, uses default elevation + sun
        avoidance constraints from the site configuration.
    time_step : float
        Scheduler tick in seconds: how far the clock advances when no
        target is available, and (plus a slew allowance) the look-ahead
        used to decide a constant-elevation pass is imminent enough to
        start.
    sun_safe : SunSafePredicate, optional
        Injected sun-safety model
        (:class:`~fyst_trajectories.dispatch.SunSafePredicate`, e.g. from
        :func:`~fyst_trajectories.sun_models.make_sun_safe`) driving the
        default Sun constraint, the mid-scan sun-drift duration clips, and
        the scan-mode planet-calibration planner
        (``plan_source_ces_passes``). Default ``None`` keeps the site's
        scalar exclusion radius.
        Only consulted while the site has Sun avoidance enabled. When an
        explicit ``constraints`` list is supplied it is used as-is;
        ``sun_safe`` then affects the duration clips only.

    Returns
    -------
    ObservingTimeline
        Complete observing timeline with science, calibration,
        slew, and idle blocks.
    """
    if isinstance(start_time, str):
        start_time = Time(start_time, scale="utc")
    if isinstance(end_time, str):
        end_time = Time(end_time, scale="utc")

    ctx = SchedulerContext.build(
        patches=patches,
        site=site,
        start_time=start_time,
        end_time=end_time,
        overhead_model=overhead_model,
        calibration_policy=calibration_policy,
        constraints=constraints,
        time_step=time_step,
        sun_safe=sun_safe,
    )
    return Scheduler(ctx).run()
