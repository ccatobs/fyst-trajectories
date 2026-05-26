"""Planning subpackage: translate astronomer inputs into trajectories.

This subpackage provides astronomer-friendly planning functions that
return :class:`ScanBlock` objects.
"""

from ._types import (
    ArrayFootprint,
    ComputedParams,
    ConstantElComputedParams,
    DaisyComputedParams,
    FieldRegion,
    PongComputedParams,
    ScanBlock,
    SourceCESComputedParams,
    validate_computed_params,
)
from .constant_el import plan_constant_el_scan
from .daisy import plan_daisy_scan
from .pong import plan_pong_rotation_sequence, plan_pong_scan
from .source_ces import compute_source_ces_params, plan_source_ces

__all__ = [
    "ArrayFootprint",
    "ComputedParams",
    "ConstantElComputedParams",
    "DaisyComputedParams",
    "FieldRegion",
    "PongComputedParams",
    "ScanBlock",
    "SourceCESComputedParams",
    "compute_source_ces_params",
    "plan_constant_el_scan",
    "plan_daisy_scan",
    "plan_pong_rotation_sequence",
    "plan_pong_scan",
    "plan_source_ces",
    "validate_computed_params",
]
