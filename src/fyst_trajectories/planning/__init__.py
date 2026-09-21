"""Planning subpackage: translate astronomer inputs into trajectories.

This subpackage provides astronomer-friendly planning functions that
return :class:`ScanBlock` objects.

The three transforms that build and adjust the footprint a source-tracking
scan is solved against (:func:`resolve_footprint`, :func:`inflate_footprint`,
:func:`offset_footprint_eta`) live here rather than in the root package,
because they are inputs to the planners. Anything rebuilding a recorded scan
must re-apply them in the order those functions document: inflate first,
then shift in eta.
"""

from ._types import (
    ArrayFootprint,
    ComputedParams,
    ConstantElComputedParams,
    DaisyAltAzComputedParams,
    DaisyComputedParams,
    FieldRegion,
    PongAltAzComputedParams,
    PongComputedParams,
    ScanBlock,
    SourceCESComputedParams,
    validate_computed_params,
)
from .constant_el import plan_constant_el_scan
from .daisy import plan_daisy_scan
from .daisy_altaz import plan_daisy_altaz_scan
from .footprints import inflate_footprint, offset_footprint_eta, resolve_footprint
from .pong import plan_pong_rotation_sequence, plan_pong_scan
from .pong_altaz import plan_pong_altaz_scan
from .source_ces import (
    compute_source_ces_params,
    plan_source_ces,
    plan_source_ces_passes,
    source_ces_focal_plane_track,
)

__all__ = [
    "ArrayFootprint",
    "ComputedParams",
    "ConstantElComputedParams",
    "DaisyAltAzComputedParams",
    "DaisyComputedParams",
    "FieldRegion",
    "PongAltAzComputedParams",
    "PongComputedParams",
    "ScanBlock",
    "SourceCESComputedParams",
    "compute_source_ces_params",
    "inflate_footprint",
    "offset_footprint_eta",
    "plan_constant_el_scan",
    "plan_daisy_altaz_scan",
    "plan_daisy_scan",
    "plan_pong_altaz_scan",
    "plan_pong_rotation_sequence",
    "plan_pong_scan",
    "plan_source_ces",
    "plan_source_ces_passes",
    "resolve_footprint",
    "source_ces_focal_plane_track",
    "validate_computed_params",
]
