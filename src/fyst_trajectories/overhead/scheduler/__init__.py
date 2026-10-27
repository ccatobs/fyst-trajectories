"""Scheduler subpackage behind ``overhead.timeline``.

Public API: :func:`fyst_trajectories.overhead.generate_timeline` stays
the sole entry point for downstream consumers. The classes in this
subpackage (:class:`Scheduler`, phase classes, state dataclasses) are
exposed for callers who assemble a :class:`SchedulerContext` themselves
or compose the phases into a different loop; that is the whole surface
this module exports, and the private helpers in ``scheduler.helpers``
are not part of it.
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
