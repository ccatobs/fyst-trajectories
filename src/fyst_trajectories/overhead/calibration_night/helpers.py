"""Pure helpers for the calibration-night planner.

Nothing here holds state or touches a clock: the solar gate, the per-module
coverage of a pass, the science duty cycle, and the JSON coercions that keep
every recorded metadata value a builtin. The Sun sweep over a planned
trajectory, :func:`sweep_sun_safe`, is shared with the survey scheduler and
lives in the simulator's move module; it is imported here for the planner.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from astropy.time import Time, TimeDelta

from ...coordinates import Coordinates
from ...observability import SUN_RISE_SET_ALTITUDE_DEG
from ...offsets import InstrumentOffset
from ...planning import ScanBlock, source_ces_focal_plane_track
from ...primecam import MODULE_FOV_RADIUS_DEG, PRIMECAM_MODULES
from ...site import Site
from ...trajectory import Trajectory
from .._moves import sweep_sun_safe
from ..schemas import ScanGeometryRecord

__all__ = [
    "json_native",
    "module_crossings",
    "science_fraction",
    "sweep_sun_safe",
    "usable_interval",
    "validate_geometry_record",
]


def usable_interval(
    site: Site,
    start: Time,
    end: Time,
    step_seconds: float,
    *,
    max_sun_altitude: float = SUN_RISE_SET_ALTITUDE_DEG,
) -> tuple[Time, Time] | None:
    """Clip a requested window to the longest stretch with the Sun down.

    Samples the Sun's altitude on a grid of ``step_seconds`` across the
    request and returns the longest contiguous run below
    ``max_sun_altitude``, resolved to the grid. The default threshold is
    the almanac sunrise and sunset altitude.

    Parameters
    ----------
    site : Site
        Observing site.
    start, end : Time
        The requested window (scalar UTC times, ``start < end``).
    step_seconds : float
        Grid spacing in seconds.
    max_sun_altitude : float, optional
        Sun altitude in degrees below which the sky counts as night.

    Returns
    -------
    tuple of Time or None
        The usable ``(start, end)``, or ``None`` when the Sun is above the
        threshold at every grid point.

    Raises
    ------
    ValueError
        If ``end`` is not after ``start`` or ``step_seconds`` is not
        positive.
    """
    if step_seconds <= 0.0:
        raise ValueError(f"step_seconds must be positive, got {step_seconds}")
    span = (end - start).to_value("s")
    if span <= 0.0:
        raise ValueError("end must be after start")
    offsets = np.arange(0.0, span + 0.5 * step_seconds, step_seconds)
    offsets[-1] = min(offsets[-1], span)
    grid = start + TimeDelta(offsets, format="sec")
    _, sun_alt = Coordinates(site).get_sun_altaz(grid)
    night = np.asarray(sun_alt) < max_sun_altitude
    if not night.any():
        return None
    # Longest run of consecutive night samples.
    best_len, best_start, run_start = 0, 0, None
    for i, is_night in enumerate(list(night) + [False]):
        if is_night and run_start is None:
            run_start = i
        elif not is_night and run_start is not None:
            if i - run_start > best_len:
                best_len, best_start = i - run_start, run_start
            run_start = None
    i0, i1 = best_start, best_start + best_len - 1
    return grid[i0], grid[i1]


def science_fraction(trajectory: Trajectory) -> float:
    """Fraction of a trajectory's samples flagged as science."""
    mask = trajectory.science_mask
    return float(np.mean(mask)) if mask.size else 0.0


def module_crossings(
    block: ScanBlock,
    site: Site,
    *,
    modules: Mapping[str, InstrumentOffset] | None = None,
    fov_radius_deg: float = MODULE_FOV_RADIUS_DEG,
) -> dict[str, float]:
    """Fraction of a pass during which the source sat inside each module.

    Traces the source through the focal plane with
    :func:`~fyst_trajectories.planning.source_ces_focal_plane_track` and,
    for each module, reports the fraction of trajectory samples whose
    source position lies within ``fov_radius_deg`` of the module centre.
    Alias keys that name one offset are reported once under the first key.

    Parameters
    ----------
    block : ScanBlock
        A source-CES pass.
    site : Site
        Observing site.
    modules : mapping of str to InstrumentOffset, optional
        Modules to test. Default
        :data:`~fyst_trajectories.primecam.PRIMECAM_MODULES`.
    fov_radius_deg : float, optional
        Module field-of-view radius in degrees.

    Returns
    -------
    dict of str to float
        Coverage fraction per module name, in module order.
    """
    modules = PRIMECAM_MODULES if modules is None else modules
    xi, eta = source_ces_focal_plane_track(block, site=site)
    seen: list[InstrumentOffset] = []
    out: dict[str, float] = {}
    for name, offset in modules.items():
        if any(offset is s for s in seen):
            continue
        seen.append(offset)
        inside = np.hypot(xi - offset.dx_deg, eta - offset.dy_deg) <= fov_radius_deg
        out[name] = float(np.mean(inside)) if inside.size else 0.0
    return out


def json_native(value: Any) -> Any:
    """Coerce a value (recursively) to JSON builtins.

    NumPy scalars become Python floats or ints, tuples become lists,
    mappings are rebuilt with string keys. Anything else is returned
    unchanged.
    """
    if isinstance(value, dict):
        return {str(k): json_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


def validate_geometry_record(record: Mapping[str, Any], name: str) -> dict[str, float]:
    """Check a geometry record's keys against :class:`ScanGeometryRecord` and coerce it.

    Raises
    ------
    ValueError
        If a key is not part of the record vocabulary or a value is not
        a finite number.
    """
    allowed = ScanGeometryRecord.__optional_keys__
    unknown = set(record) - set(allowed)
    if unknown:
        raise ValueError(f"{name} record has unknown keys {sorted(unknown)}")
    out: dict[str, float] = {}
    for key, value in record.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name}[{key!r}] must be a number, got {value!r}") from None
        if not np.isfinite(number):
            raise ValueError(f"{name}[{key!r}] must be finite, got {value!r}")
        out[key] = number
    return out
