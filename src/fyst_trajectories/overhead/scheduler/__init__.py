"""Scheduler subpackage behind ``overhead.timeline``.

Public API: :func:`fyst_trajectories.overhead.generate_timeline` stays
the sole entry point for downstream consumers. The classes in this
subpackage (:class:`Scheduler`, phase classes, state dataclasses) are
exposed for advanced users who want to extend scheduling behavior
(priority-weighted scheduling, multi-night stitching, lookahead). The
subpackage's own private helpers stay in ``scheduler.helpers``; a caller
that needs one imports it from there, so what this module exports is the
extension surface and nothing else.
"""

from .phases import (
    CalibrationPhase,
    PatchSelectionPhase,
    Phase,
    PhaseResult,
    ScienceScanPhase,
    SlewPhase,
)
from .scheduler import Scheduler
from .state import SchedulerContext, SchedulerState

__all__ = [
    "CalibrationPhase",
    "PatchSelectionPhase",
    "Phase",
    "PhaseResult",
    "ScienceScanPhase",
    "Scheduler",
    "SchedulerContext",
    "SchedulerState",
    "SlewPhase",
]
