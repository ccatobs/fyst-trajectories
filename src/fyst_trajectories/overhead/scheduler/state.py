"""Scheduler state and context dataclasses.

The :class:`SchedulerState` is an immutable snapshot of scheduler
progress that is evolved between phases via
:func:`dataclasses.replace`. :class:`SchedulerContext` bundles the
read-only configuration (site, patches, overhead/calibration policy,
constraints, time window) that every phase reads but never mutates.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from astropy.time import Time

from .helpers import _default_constraints

if TYPE_CHECKING:
    from ...coordinates import Coordinates
    from ...dispatch import SunSafePredicate
    from ...site import Site
    from ..constraints import Constraint
    from ..models import CalibrationPolicy, ObservingPatch, OverheadModel
    from ..overhead import CalibrationState

__all__ = ["SchedulerContext", "SchedulerState"]


@dataclass(frozen=True)
class SchedulerState:
    """Immutable scheduler state; evolved via :func:`dataclasses.replace`.

    Attributes
    ----------
    current_time : Time
        UTC timestamp of the scheduler's current position.
    current_az : float
        Telescope azimuth (deg) at ``current_time``.
    current_el : float
        Telescope elevation (deg) at ``current_time``.
    cal_state : CalibrationState
        Cadence-tracking state for each calibration type. Immutable;
        replaced whenever a calibration fires.
    scan_counter : int
        Monotonically increasing counter used as ``scan_index`` on
        emitted blocks.
    """

    current_time: Time
    current_az: float
    current_el: float
    cal_state: CalibrationState
    scan_counter: int

    @classmethod
    def initial(cls, start_time: Time, cal_state: CalibrationState) -> SchedulerState:
        """Build the scheduler's initial state.

        The ``(current_az=180.0, current_el=50.0)`` initialization is a
        bare bootstrap: the telescope is assumed to start roughly pointed
        at the southern horizon at a mid-sky elevation. These values are
        the recorded pose until the first emitted slew or science scan
        replaces them, so any calibration or idle block emitted before
        that point is stamped at this position. The bootstrap azimuth
        also seeds the cable-wrap frame: the first selected target is
        placed on the representative nearest it, and each later target
        relative to the pose before it.
        """
        return cls(
            current_time=start_time,
            current_az=180.0,
            current_el=50.0,
            cal_state=cal_state,
            scan_counter=0,
        )

    def advanced(self, **changes) -> SchedulerState:
        """Return a copy of this state with ``changes`` applied."""
        return replace(self, **changes)


@dataclass(frozen=True)
class SchedulerContext:
    """Read-only scheduling context passed to every phase.

    Holds all configuration that remains constant across the entire
    timeline: patches, site, coordinate transform, overhead/calibration
    models, constraint list, time window, and idle time step.

    Attributes
    ----------
    patches : list of ObservingPatch
        The candidate sky regions the scheduler selects among.
    site : Site
        Telescope site configuration.
    coords : Coordinates
        Coordinate transform bound to ``site``.
    overhead_model : OverheadModel
        Per-activity durations and scan split thresholds.
    calibration_policy : CalibrationPolicy
        Calibration cadences.
    constraints : list of Constraint
        Patch-selection constraints, scored per candidate each tick.
    start_time, end_time : Time
        The timeline window.
    time_step : float
        Idle-tick step in seconds.
    """

    patches: list[ObservingPatch]
    site: Site
    coords: Coordinates
    overhead_model: OverheadModel
    calibration_policy: CalibrationPolicy
    constraints: list[Constraint]
    start_time: Time
    end_time: Time
    time_step: float
    #: Injected sun-safety model (:class:`~fyst_trajectories.dispatch.SunSafePredicate`,
    #: e.g. from :func:`~fyst_trajectories.sun_models.make_sun_safe`) driving
    #: the mid-scan duration clips; ``None`` keeps the scalar site radius.
    #: The Sun *constraint* is bound at construction time (see ``build``).
    sun_safe: SunSafePredicate | None = None
    #: Per-run memo of constant-elevation crossing-pass solves, keyed
    #: ``(patch_name, elevation, rising)`` (name, float, bool) with values
    #: ``("ok", t_open, t_close)`` or ``("miss", solved_from)``. Written
    #: only by the scheduler helper ``helpers._ce_crossing_corridor``; a
    #: cache, not state.
    ce_corridors: dict = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        patches: list[ObservingPatch],
        site: Site,
        start_time: Time,
        end_time: Time,
        overhead_model: OverheadModel | None = None,
        calibration_policy: CalibrationPolicy | None = None,
        constraints: list[Constraint] | None = None,
        time_step: float = 300.0,
        sun_safe: SunSafePredicate | None = None,
    ) -> SchedulerContext:
        """Assemble a context, filling in default overhead/policy/constraints.

        ``sun_safe`` reaches both consumers: the default constraint set
        (when ``constraints`` is None) and the scan-duration clips. A
        caller supplying an explicit ``constraints`` list owns its Sun
        constraint; ``sun_safe`` then affects the duration clips only.
        """
        from ...coordinates import Coordinates
        from ..models import CalibrationPolicy, OverheadModel

        if overhead_model is None:
            overhead_model = OverheadModel()
        if calibration_policy is None:
            calibration_policy = CalibrationPolicy()
        if constraints is None:
            constraints = _default_constraints(site, sun_safe=sun_safe)
        return cls(
            patches=patches,
            site=site,
            coords=Coordinates(site),
            overhead_model=overhead_model,
            calibration_policy=calibration_policy,
            constraints=constraints,
            start_time=start_time,
            end_time=end_time,
            time_step=time_step,
            sun_safe=sun_safe,
        )
