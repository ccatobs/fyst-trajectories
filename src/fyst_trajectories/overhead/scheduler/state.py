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
        at the southern horizon at a mid-sky elevation. The first slew
        block computed from this position replaces these values; they
        only influence the *first* slew-time estimate, not subsequent
        scheduling decisions.
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
    #: Per-run memo of constant-elevation crossing-pass solves, keyed
    #: ``(patch_name, rising)`` with values ``("ok", t_open, t_close)`` or
    #: ``("miss", solved_from)``. Written only by the scheduler helpers
    #: (:func:`.helpers._ce_crossing_corridor`); a cache, not state.
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
    ) -> SchedulerContext:
        """Assemble a context, filling in default overhead/policy/constraints."""
        from ...coordinates import Coordinates
        from ..models import CalibrationPolicy, OverheadModel

        if overhead_model is None:
            overhead_model = OverheadModel()
        if calibration_policy is None:
            calibration_policy = CalibrationPolicy()
        if constraints is None:
            constraints = _default_constraints(site)
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
        )
