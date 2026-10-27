"""Overhead modeling and observing timeline generation for FYST/Prime-Cam.

This subpackage provides overhead budget modeling, calibration cadence tracking,
and timeline generation for the FYST telescope, on top of the library's
coordinate transforms, site configuration, and trajectory generation.

The key entry point is :func:`generate_timeline`, which takes a list of
observing patches and produces a complete timeline with calibration injection.

.. note::

   The retune blocks the scheduler emits on the :class:`CalibrationPolicy`
   cadence (between subscans / iterations) are independent of the in-scan
   retune samples
   that :func:`fyst_trajectories.retune.inject_retune` injects on a single
   :class:`~fyst_trajectories.trajectory.Trajectory`. Both sets of retune
   knobs belong to the instrument team (see :doc:`/overhead_integration`),
   but they time different operations and are not kept in sync. Nothing in
   this subpackage calls
   :func:`~fyst_trajectories.retune.inject_retune`: a block rebuilt by
   :func:`~fyst_trajectories.overhead.schedule_to_trajectories` carries no
   retune samples.
"""

from .calibration_night import (
    BOOTSTRAP_POSE,
    DEFAULT_SCAN_TABLES,
    BodySummary,
    CalibrationNightMetadata,
    CalibrationNightPolicy,
    Candidate,
    ElevationBin,
    IdleRecord,
    NightContext,
    NightRecord,
    NightState,
    NightSummary,
    OperationRecord,
    PassRecord,
    ScanOverrides,
    ScanParameterTable,
    ScriptedSelection,
    SelectionRule,
    SlewRecord,
    TuningPolicy,
    VisitPlan,
    VisitPlanner,
    VisitRecord,
    advance_idle,
    commit_visit,
    dispatch_records,
    dispatch_sheet,
    list_candidates,
    load_scan_tables,
    plan_calibration_night,
    plan_visit,
    read_calibration_night_metadata,
    select_priority,
    summarize_calibration_night,
)
from .calibration_state import CalibrationState
from .constraints import (
    Constraint,
    ElevationConstraint,
    MinDurationConstraint,
    MoonAvoidanceConstraint,
    SunAvoidanceConstraint,
)
from .exceptions import BlockNotReconstructableError, ScanParamsSchemaError
from .io import read_timeline, write_timeline
from .models import (
    BlockType,
    CalibrationPolicy,
    CalibrationSpec,
    CalibrationType,
    ObservingPatch,
    ObservingTimeline,
    OverheadModel,
    TimelineBlock,
)
from .schemas import (
    CalibrationBlockMetadata,
    CEScanParams,
    DaisyScanParams,
    EmptyBlockMetadata,
    PongScanParams,
    ScanGeometryRecord,
    ScanParamsDict,
    ScienceBlockMetadata,
    SourceCESScanParams,
    TimelineBlockMetadata,
    TransitionRecord,
    validate_scan_params,
)
from .simulation import (
    BudgetStats,
    CalibrationBudget,
    PatchBudget,
    accumulate_hitmaps,
    compute_budget,
    schedule_to_trajectories,
)
from .timeline import generate_timeline
from .transitions import DeferralReason, Transition, plan_escape, plan_transition

__all__ = [
    "BOOTSTRAP_POSE",
    "BlockNotReconstructableError",
    "BlockType",
    "BodySummary",
    "BudgetStats",
    "CEScanParams",
    "CalibrationBlockMetadata",
    "CalibrationBudget",
    "CalibrationNightMetadata",
    "CalibrationNightPolicy",
    "CalibrationPolicy",
    "CalibrationSpec",
    "CalibrationState",
    "CalibrationType",
    "Candidate",
    "Constraint",
    "DEFAULT_SCAN_TABLES",
    "DaisyScanParams",
    "DeferralReason",
    "ElevationBin",
    "ElevationConstraint",
    "EmptyBlockMetadata",
    "IdleRecord",
    "MinDurationConstraint",
    "MoonAvoidanceConstraint",
    "NightContext",
    "NightRecord",
    "NightState",
    "NightSummary",
    "ObservingPatch",
    "ObservingTimeline",
    "OperationRecord",
    "OverheadModel",
    "PassRecord",
    "PatchBudget",
    "PongScanParams",
    "ScanGeometryRecord",
    "ScanOverrides",
    "ScanParameterTable",
    "ScanParamsDict",
    "ScanParamsSchemaError",
    "ScienceBlockMetadata",
    "ScriptedSelection",
    "SelectionRule",
    "SlewRecord",
    "SourceCESScanParams",
    "SunAvoidanceConstraint",
    "TimelineBlock",
    "TimelineBlockMetadata",
    "Transition",
    "TransitionRecord",
    "TuningPolicy",
    "VisitPlan",
    "VisitPlanner",
    "VisitRecord",
    "accumulate_hitmaps",
    "advance_idle",
    "commit_visit",
    "compute_budget",
    "dispatch_records",
    "dispatch_sheet",
    "generate_timeline",
    "list_candidates",
    "load_scan_tables",
    "plan_calibration_night",
    "plan_escape",
    "plan_transition",
    "plan_visit",
    "read_calibration_night_metadata",
    "read_timeline",
    "schedule_to_trajectories",
    "select_priority",
    "summarize_calibration_night",
    "validate_scan_params",
    "write_timeline",
]
