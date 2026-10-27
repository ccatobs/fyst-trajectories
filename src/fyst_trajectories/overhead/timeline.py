"""Public :func:`generate_timeline` entry point."""

from typing import TYPE_CHECKING

from astropy.time import Time

from ..site import Site
from .constraints import Constraint
from .models import (
    CalibrationPolicy,
    ObservingPatch,
    ObservingTimeline,
    OverheadModel,
)
from .scheduler import Scheduler, SchedulerContext

if TYPE_CHECKING:
    from ..sun_protocols import SunSafePredicate

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
        Timeline start time, held in UTC: a string is read as UTC, and any
        ``Time`` is held as a UTC ``Time`` without a location and with
        astropy's default ``precision`` and ``out_subfmt`` (one in another
        scale is converted to UTC).
    end_time : Time or str
        Timeline end time, held in UTC like ``start_time``.
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
        (:class:`~fyst_trajectories.sun_protocols.SunSafePredicate`, e.g. from
        :func:`~fyst_trajectories.sun_models.make_sun_safe`) driving the
        default Sun constraint, the mid-scan sun-drift duration clips, the
        slew gate and the escape move, and the scan-mode planet
        calibrations (the planet choice, the planner
        ``plan_source_ces_passes``, the slew to the first pass and the
        sweep of every pass). Default ``None`` keeps the site's scalar
        exclusion radius.
        Only consulted while the site has Sun avoidance enabled. When an
        explicit ``constraints`` list is supplied it is used as-is, so
        ``sun_safe`` no longer sets the patch-selection constraint; it
        still drives everything else listed here.

    Returns
    -------
    ObservingTimeline
        Complete observing timeline with science, calibration,
        slew, and idle blocks.

    Raises
    ------
    ValueError
        If ``time_step`` is not positive, two patches share a name, a
        constant-elevation patch has no pinned ``elevation``, or a pong
        patch's pattern period exceeds ``max_scan_duration`` less, at
        ``retune_cadence=0``, the retune booked before every subscan.

    See Also
    --------
    fyst_trajectories.overhead.plan_calibration_night : one night of
        solar-system calibration passes planned back to back over a body
        queue; shares this simulator's block model and outputs.

    Examples
    --------
    Generate a one-night timeline:

    >>> from fyst_trajectories import get_fyst_site
    >>> from fyst_trajectories.overhead import (
    ...     ObservingPatch,
    ...     compute_budget,
    ...     generate_timeline,
    ... )
    >>> site = get_fyst_site()
    >>> patches = [
    ...     ObservingPatch(
    ...         name="Deep56",
    ...         ra_center=24.0,
    ...         dec_center=-32.0,
    ...         width=40.0,
    ...         height=10.0,
    ...         scan_type="constant_el",
    ...         velocity=1.0,
    ...         elevation=50.0,
    ...     ),
    ... ]
    >>> timeline = generate_timeline(
    ...     patches=patches,
    ...     site=site,
    ...     start_time="2026-06-15T00:00:00",
    ...     end_time="2026-06-15T12:00:00",
    ... )
    >>> stats = compute_budget(timeline)
    >>> print(f"Efficiency: {stats['efficiency']:.1%}")
    Efficiency: 23.7%
    """
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
