"""The source-CES kernel: the params-only computation behind both single-pass planners.

``_compute_source_ces_core`` is the one computation ``compute_source_ces_params``
and ``plan_source_ces`` share, so the scalars a scheduler prices at emit time are
the ones the dispatched trajectory is built from. It runs as a sequence of stage
functions, in this order: input validation; sampling the source and selecting its
monotonic elevation arc through ``el_bore``; recovering the boresight azimuth and
projecting the footprint cover; the coverage guard, the crossing window and its
``dwell`` narrowing; the drift solve; the swept azimuth window; the leg speed; the
Sun screen and the peak-speed advisory; and quantisation and assembly.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta
from scipy import interpolate
from scipy.optimize import minimize

from ..._validation import _require_non_negative, _require_positive
from ...coordinates import Coordinates
from ...exceptions import (
    DwellExceedsCrossingError,
    ElevationBoundsError,
    PointingError,
    PointingWarning,
    TargetNotObservableError,
)
from ...offsets import (
    InstrumentOffset,
    boresight_to_detector,
    compute_focal_plane_rotation,
    detector_to_boresight,
)
from ...site import AtmosphericConditions, AxisLimits, Site
from .._ce_geometry import _quantize_ce_duration
from .._sun_safety import (
    _SUN_SAFETY_ARC_N_SAMPLES,
    _check_arc_sun_safety,
    _swept_arc_samples,
)
from .._types import ArrayFootprint, SourceCESComputedParams
from ..footprints import resolve_footprint
from ._source import _SourceSpec

if TYPE_CHECKING:
    # Annotation-only import: the predicate is invoked structurally, so only
    # the type hint needs the symbol.
    from ...sun_protocols import SunSafePredicate

# Default search-window horizon when only ``night`` is supplied. 24 h
# covers a full diurnal rotation; sources reachable from FYST always
# have a rising and setting pass within this window (or never reach the
# requested elevation, which we catch separately).
_DEFAULT_SEARCH_HORIZON_HOURS = 24.0

# Floor on the per-leg azimuth velocity used to seed the underlying
# ConstantEl pattern when no ``az_speed`` is given. It is the usual
# operating point rather than an edge guard: ``az_throw / duration_window``
# falls below it whenever the window is longer than 20 s per degree of
# throw, so the default sweeps a module-scale pass at this speed in several
# legs rather than in one. The floor does not affect the solved drift.
_MIN_PER_LEG_VELOCITY_DEG_S = 0.05

# Default azimuth padding added to each side of the solved footprint
# crossing. Named so the planners can tell an explicit padding from the
# default: an explicit ``az_throw`` replaces the padded throw outright,
# and combining it with an explicit padding is rejected.
_DEFAULT_AZ_PADDING_DEG = 0.5

# ``stacklevel`` of the kernel's advisories, counted from the stage that issues
# one: the stage, the kernel, ``_resolve_and_compute``, the public entry point,
# and the frame that called the entry point, to which the warning is attributed.
_ADVISORY_STACKLEVEL = 5


@dataclass(frozen=True)
class _SourceCESCore:
    """Internal carrier for the params-only phase of source-CES planning.

    Holds the completed :class:`SourceCESComputedParams` plus the intermediate
    state the trajectory builder in ``plan_source_ces`` needs.
    Strictly private; no public API guarantees on this class.
    """

    computed: SourceCESComputedParams
    # Intermediate state for the trajectory builder.
    actual_duration: float
    t0: Time
    n_scans: int
    mode: Literal["rising", "setting"]
    # The boresight elevation as the kernel received it: the caller's value,
    # or the one a ``start_time`` anchor derived.
    el_bore: float
    velocity: float
    az_stop: float
    source_label: str
    fp: ArrayFootprint
    # Source coords at ``t_at_el_bore`` for the trajectory metadata.
    src_ra_at_el_bore: float
    src_dec_at_el_bore: float


@dataclass(frozen=True)
class _SourceArc:
    """The source's monotonic elevation arc through ``el_bore``, as the later stages read it."""

    t_search_start: Time
    mode: Literal["rising", "setting"]
    # The arc's elevations in time order, and the arc's samples sorted by elevation.
    el_slice: np.ndarray
    el_sorted: np.ndarray
    t_sorted: np.ndarray
    az_sorted: np.ndarray
    # Linear ``t(el)`` over the sorted arc, in seconds from ``t_search_start``.
    t_of_el: interpolate.interp1d
    # The source where and when it crosses ``el_bore``.
    src_az_at_el_bore: float
    src_ra_at_el_bore: float
    src_dec_at_el_bore: float


@dataclass(frozen=True)
class _CrossingWindow:
    """The scanned source window, in seconds from the search start and as times."""

    t0_sec: float
    t1_sec: float
    # The full footprint crossing, before any ``dwell`` narrowing.
    crossing_seconds: float
    t0: Time
    t1: Time


def _enumerate_monotonic_arcs(el_src: np.ndarray) -> list[tuple[int, int]]:
    """Enumerate monotonic arcs in an elevation trace.

    Returns a list of ``(i_start, i_end_inclusive)`` index pairs marking
    every maximal monotonic sub-arc of ``el_src``. Window endpoints are
    always treated as arc boundaries so an arc that begins or ends
    mid-rise/fall is still captured.

    Plateaus (consecutive samples with identical elevation) are absorbed
    into the adjacent monotonic run rather than treated as separate
    extrema; this is a numerical robustness measure. Astronomical
    altitude traces are smooth, so true plateaus only occur as
    sampling-coincidence artefacts.

    Parameters
    ----------
    el_src : np.ndarray
        1-D elevation samples (degrees).

    Returns
    -------
    list of (int, int)
        Each tuple is an ``(i_start, i_end_inclusive)`` pair. Always
        contains at least one entry when ``el_src.size >= 2``.
    """
    n = el_src.size
    if n < 2:
        return []
    de = np.diff(el_src)
    # Treat zero diffs as continuing the prior direction so that a
    # plateau does not split a monotonic run. The first non-zero diff
    # seeds the direction.
    extrema: list[int] = [0]
    prev_sign = 0
    for i, d in enumerate(de):
        s = 1 if d > 0 else (-1 if d < 0 else 0)
        if s == 0:
            continue
        if prev_sign != 0 and s != prev_sign:
            # Sign change at index i+0 (i.e. el_src[i] is the extremum).
            extrema.append(i)
        prev_sign = s
    extrema.append(n - 1)
    # Deduplicate while preserving order (window endpoints may coincide
    # with an internal extremum if the trace happens to peak at the
    # boundary).
    seen: set[int] = set()
    deduped: list[int] = []
    for idx in extrema:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    return [(deduped[i], deduped[i + 1]) for i in range(len(deduped) - 1)]


def _project_cover_to_altaz(
    footprint: ArrayFootprint,
    az_bore: float,
    el_bore: float,
    focal_plane_rotation_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a footprint's cover polygon to on-sky (az, el)."""
    n = footprint.cover_xi_deg.size
    az_cover = np.empty(n)
    el_cover = np.empty(n)
    for i, (xi_deg, eta_deg) in enumerate(zip(footprint.cover_xi_deg, footprint.cover_eta_deg)):
        # InstrumentOffset takes arcmin; convert from degrees.
        vertex_offset = InstrumentOffset(dx=xi_deg * 60.0, dy=eta_deg * 60.0)
        az_v, el_v = boresight_to_detector(
            az=az_bore,
            el=el_bore,
            offset=vertex_offset,
            focal_plane_rotation=focal_plane_rotation_deg,
        )
        az_cover[i] = az_v
        el_cover[i] = el_v
    return az_cover, el_cover


def _select_source_arc(
    *,
    el_src: np.ndarray,
    el_bore: float,
    mode: Literal["rising", "setting"] | None,
    source_label: str,
    t_search_start: Time,
    el_limits: AxisLimits,
) -> tuple[int, int, Literal["rising", "setting"]]:
    """Pick the monotonic elevation arc whose range covers ``el_bore``.

    The window can span multiple local extrema (e.g. a 24 h window on a
    planet near a culmination contains both a max and an anti-culmination
    minimum). Naively picking ``(argmin, argmax)`` selects the dominant
    extrema regardless of order in time, producing an empty or reversed
    slice when the global min happens *after* the global max within the
    window. We instead enumerate all monotonic arcs between consecutive
    local extrema (with window endpoints inserted as "virtual" extrema so
    an arc that begins or ends mid-rise/fall is still captured) and pick
    the first arc whose elevation range covers ``el_bore``.

    When ``mode`` is ``None`` it is auto-detected from the longest covering
    arc; otherwise arcs are filtered by that direction. Returns the
    half-open sample-index slice ``(i_beg, i_end)`` and the resolved mode.

    Raises
    ------
    TargetNotObservableError
        If no arc covers ``el_bore`` (or, with an explicit ``mode``, no arc
        of that direction exists or covers it), or if the covering arc spans
        fewer than two samples.
    """
    arcs = _enumerate_monotonic_arcs(el_src)

    if mode is None:
        # Auto-detect: choose the mode of the longest arc that contains
        # ``el_bore``. If no arc covers ``el_bore`` we raise immediately
        # with the global el span (no silent fall-back to the longest
        # arc; that would defer the failure to the el-slice guard below with a confusing
        # error message).
        candidate_arcs = [
            arc
            for arc in arcs
            if min(el_src[arc[0]], el_src[arc[1]]) <= el_bore <= max(el_src[arc[0]], el_src[arc[1]])
        ]
        if not candidate_arcs:
            raise TargetNotObservableError(
                target=source_label,
                time_info=str(t_search_start.iso),
                bounds_error=ElevationBoundsError(
                    actual_min=float(np.min(el_src)),
                    actual_max=float(np.max(el_src)),
                    limit_min=el_bore,
                    limit_max=el_bore,
                ),
            )
        chosen_arc = max(candidate_arcs, key=lambda arc: arc[1] - arc[0])
        i_beg, i_end_inclusive = chosen_arc
        mode = "rising" if el_src[i_end_inclusive] > el_src[i_beg] else "setting"
    else:
        # Filter arcs by direction, then take the first (chronologically)
        # whose el-range covers ``el_bore``. If none does, raise
        # ``TargetNotObservableError`` reporting the best-available el span
        # across all directional arcs (no silent fall-back).
        if mode == "rising":
            directional = [arc for arc in arcs if el_src[arc[1]] > el_src[arc[0]]]
        else:
            directional = [arc for arc in arcs if el_src[arc[1]] < el_src[arc[0]]]
        if not directional:
            raise TargetNotObservableError(
                target=source_label,
                time_info=str(t_search_start.iso),
                bounds_error=ElevationBoundsError(
                    actual_min=float(np.min(el_src)),
                    actual_max=float(np.max(el_src)),
                    limit_min=el_limits.min,
                    limit_max=el_limits.max,
                ),
            )
        covering = [
            arc
            for arc in directional
            if min(el_src[arc[0]], el_src[arc[1]]) <= el_bore <= max(el_src[arc[0]], el_src[arc[1]])
        ]
        if not covering:
            # Report the best-available el span (across all directional
            # arcs) so the caller can adjust ``el_bore`` or extend the
            # window.
            best_max = max(float(max(el_src[arc[0]], el_src[arc[1]])) for arc in directional)
            best_min = min(float(min(el_src[arc[0]], el_src[arc[1]])) for arc in directional)
            raise TargetNotObservableError(
                target=source_label,
                time_info=str(t_search_start.iso),
                bounds_error=ElevationBoundsError(
                    actual_min=best_min,
                    actual_max=best_max,
                    limit_min=el_bore,
                    limit_max=el_bore,
                ),
            )
        i_beg, i_end_inclusive = covering[0]

    i_end = i_end_inclusive + 1  # half-open slice end
    if i_end - i_beg < 2:
        raise TargetNotObservableError(
            target=source_label,
            time_info=str(t_search_start.iso),
            bounds_error=ElevationBoundsError(
                actual_min=float(np.min(el_src)),
                actual_max=float(np.max(el_src)),
                limit_min=el_limits.min,
                limit_max=el_limits.max,
            ),
        )

    return i_beg, i_end, mode


def _validate_kernel_inputs(
    *,
    source: _SourceSpec,
    footprint: InstrumentOffset | str | Sequence[InstrumentOffset] | ArrayFootprint,
    el_bore: float,
    window: tuple[Time, Time] | None,
    night: Time | None,
    mode: Literal["rising", "setting"] | None,
    site: Site,
    sampling_step_seconds: float,
    az_accel: float,
    az_padding: float,
    az_speed: float | None,
    az_throw: float | None,
    dwell: float | None,
) -> None:
    """Refuse a malformed kernel request before any computation, in a fixed order.

    Raises
    ------
    ValueError
        On a non-positive step, acceleration, speed, throw or dwell, a negative
        ``az_padding``, an ``az_throw`` combined with an explicit ``az_padding``,
        an ``ArrayFootprint`` whose cover is a single point with ``az_padding``
        0 and no ``az_throw``, a ``dwell`` shorter than ``sampling_step_seconds``,
        a missing or conflicting source or window, a missing or unknown
        ``mode``, an array-valued time or a reversed window.
    ElevationBoundsError
        When ``el_bore`` lies outside ``site.telescope_limits.elevation``.
    """
    _require_positive(sampling_step_seconds, "sampling_step_seconds")
    _require_positive(az_accel, "az_accel")
    _require_non_negative(az_padding, "az_padding")
    if az_speed is not None:
        _require_positive(az_speed, "az_speed")
    if az_throw is not None:
        _require_positive(az_throw, "az_throw")
        if az_padding != _DEFAULT_AZ_PADDING_DEG:
            raise ValueError(
                "az_throw replaces the padded throw, so it cannot be combined with an "
                f"explicit az_padding (got az_throw={az_throw}, az_padding={az_padding})"
            )
    elif (
        az_padding == 0.0
        and isinstance(footprint, ArrayFootprint)
        and np.ptp(footprint.cover_xi_deg) == 0.0
        and np.ptp(footprint.cover_eta_deg) == 0.0
    ):
        # A cover with no extent is crossed at one azimuth at every site and
        # time, so without padding its swept window can never have a width.
        raise ValueError(
            "footprint cover is a single point, so with az_padding=0 and no az_throw the "
            "swept azimuth window has no width; give a positive az_padding, or an az_throw "
            "in place of az_padding"
        )
    if dwell is not None:
        _require_positive(dwell, "dwell")
        if dwell < sampling_step_seconds:
            # The window the dwell narrows is resampled on the
            # ``sampling_step_seconds`` grid, so a shorter dwell cannot be
            # honoured; refuse rather than silently widen it.
            raise ValueError(
                f"dwell must be at least sampling_step_seconds: got dwell={dwell} s with "
                f"sampling_step_seconds={sampling_step_seconds} s (lower "
                "sampling_step_seconds to plan a shorter pass)"
            )

    has_body = source.body is not None
    has_radec = source.ra is not None or source.dec is not None
    if has_body and has_radec:
        raise ValueError("specify either 'body' or 'ra'/'dec', not both")
    if not has_body:
        if source.ra is None or source.dec is None:
            raise ValueError("must specify 'body' or both 'ra' and 'dec'")
        if (source.pm_ra != 0.0 or source.pm_dec != 0.0) and source.ref_epoch is None:
            raise ValueError(
                "ref_epoch is required when pm_ra or pm_dec is non-zero "
                "(proper motion needs a reference epoch to propagate from)"
            )

    has_window = window is not None
    has_night = night is not None
    if has_window and has_night:
        raise ValueError("specify either 'window' or 'night', not both")
    if not has_window and not has_night:
        raise ValueError("must specify either 'window' or 'night'+'mode'")
    if has_night and mode is None:
        raise ValueError("'mode' is required when using 'night'")
    if mode is not None and mode not in ("rising", "setting"):
        raise ValueError(f"mode must be 'rising' or 'setting', got {mode!r}")
    # Each of these names one instant. An array-valued Time would fail far
    # downstream, either as a numpy broadcast error or, worse, as a
    # target-visibility refusal naming a whole grid of times.
    if has_night and not night.isscalar:  # type: ignore[union-attr]
        raise ValueError(f"night must be a single instant, got a Time of shape {night.shape}")  # type: ignore[union-attr]
    if has_window:
        for label, edge in zip(("window start", "window end"), window):  # type: ignore[arg-type]
            if not edge.isscalar:
                raise ValueError(
                    f"{label} must be a single instant, got a Time of shape {edge.shape}"
                )
        # A reversed or empty window makes the search grid below empty, and
        # the first reduction over it raised numpy's zero-size message from
        # inside an error constructor, naming neither argument.
        if (window[1] - window[0]).sec <= 0.0:  # type: ignore[index]
            raise ValueError(
                f"window end ({window[1].iso}) must be later than window start "  # type: ignore[index]
                f"({window[0].iso})."  # type: ignore[index]
            )

    el_limits = site.telescope_limits.elevation
    if not (el_limits.min <= el_bore <= el_limits.max):
        raise ElevationBoundsError(
            actual_min=el_bore,
            actual_max=el_bore,
            limit_min=el_limits.min,
            limit_max=el_limits.max,
        )


def _solve_source_arc(
    *,
    source: _SourceSpec,
    coords: Coordinates,
    site: Site,
    el_bore: float,
    window: tuple[Time, Time] | None,
    night: Time | None,
    mode: Literal["rising", "setting"] | None,
    sampling_step_seconds: float,
    source_label: str,
) -> _SourceArc:
    """Sample the source over the search window and select its arc through ``el_bore``.

    The search window is ``window``, or the default horizon
    (``_DEFAULT_SEARCH_HORIZON_HOURS``) from ``night``. The arc is the
    monotonic one :func:`_select_source_arc` picks, which also resolves an
    omitted ``mode``.

    Raises
    ------
    TargetNotObservableError
        When no arc of the window covers ``el_bore`` (see
        :func:`_select_source_arc`).
    """
    if window is not None:
        t_search_start, t_search_end = window
        horizon_seconds = (t_search_end - t_search_start).to_value(u.s)
    else:
        assert night is not None  # narrow for type-checker; guaranteed by the input stage
        t_search_start = night
        horizon_seconds = _DEFAULT_SEARCH_HORIZON_HOURS * 3600.0

    dt_sec = np.arange(0.0, horizon_seconds, sampling_step_seconds)
    search_times = t_search_start + TimeDelta(dt_sec * u.s)

    az_src, el_src = source.sample_altaz(coords, search_times)

    # Select the monotonic elevation arc that covers el_bore (the window may
    # span multiple extrema) and resolve mode when it was omitted.
    i_beg, i_end, mode = _select_source_arc(
        el_src=el_src,
        el_bore=el_bore,
        mode=mode,
        source_label=source_label,
        t_search_start=t_search_start,
        el_limits=site.telescope_limits.elevation,
    )

    t_slice = dt_sec[i_beg:i_end]
    az_slice = np.unwrap(np.deg2rad(az_src[i_beg:i_end]))
    az_slice = np.rad2deg(az_slice)
    el_slice = el_src[i_beg:i_end]

    # interp1d wants sorted x. For rising slice el is increasing; for
    # setting slice el is decreasing; sort by el value either way.
    sort_idx = np.argsort(el_slice)
    el_sorted = el_slice[sort_idx]
    t_sorted = t_slice[sort_idx]
    az_sorted = az_slice[sort_idx]
    # Linear interpolation for ``t(el)``: cubic is liable to overshoot
    # near the arc apex where ``dEl/dt -> 0`` and would silently
    # extrapolate out-of-range queries. ``_select_source_arc`` returns a
    # monotonic arc whose endpoints bracket ``el_bore``, so ``el_bore`` is
    # inside ``[el_slice.min(), el_slice.max()]``; the assertion below pins
    # that invariant for any future code change.
    t_of_el = interpolate.interp1d(
        el_sorted,
        t_sorted,
        kind="linear",
        fill_value="extrapolate",  # type: ignore[arg-type]
        assume_sorted=True,
    )
    az_of_el = interpolate.interp1d(
        el_sorted,
        az_sorted,
        kind="linear",
        fill_value="extrapolate",  # type: ignore[arg-type]
        assume_sorted=True,
    )

    assert el_sorted[0] <= el_bore <= el_sorted[-1], (
        "the el-slice guard must guarantee el_bore is in the sorted el slice; got "
        f"el_bore={el_bore} not in [{el_sorted[0]}, {el_sorted[-1]}]"
    )

    src_az_at_el_bore = float(az_of_el(el_bore))
    t_at_el_bore_sec = float(t_of_el(el_bore))
    t_at_el_bore = t_search_start + TimeDelta(t_at_el_bore_sec * u.s)

    # Source RA/Dec at el_bore (recorded as trajectory center metadata).
    src_ra_at_el_bore, src_dec_at_el_bore = source.radec_at(coords, t_at_el_bore)
    return _SourceArc(
        t_search_start=t_search_start,
        mode=mode,
        el_slice=el_slice,
        el_sorted=el_sorted,
        t_sorted=t_sorted,
        az_sorted=az_sorted,
        t_of_el=t_of_el,
        src_az_at_el_bore=src_az_at_el_bore,
        src_ra_at_el_bore=src_ra_at_el_bore,
        src_dec_at_el_bore=src_dec_at_el_bore,
    )


def _project_footprint(
    *,
    fp: ArrayFootprint,
    site: Site,
    el_bore: float,
    boresight_rot_deg: float,
    src_az_at_el_bore: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Put the footprint centre on the source and project the cover to the sky.

    Returns ``(az_bore, az_cover, el_cover)``: the boresight azimuth at
    ``el_bore`` that places the footprint centre on the source, and the
    cover polygon's on-sky azimuths and elevations at that boresight.

    Raises
    ------
    OffsetInversionError
        When the boresight inverse for an off-centre footprint cannot be
        solved at ``el_bore``.
    """
    # az_bore recovery. For a centred footprint (PrimeCam full-array,
    # PRIMECAM_CENTER) the boresight az IS the source az at el_bore.
    # For an off-centre footprint (single module), back out the
    # boresight via the spherical inverse so the array centre lands on
    # the source.
    if abs(fp.center_xi_deg) < 1e-9 and abs(fp.center_eta_deg) < 1e-9:
        az_bore = src_az_at_el_bore
    else:
        center_offset = InstrumentOffset(dx=fp.center_xi_deg * 60.0, dy=fp.center_eta_deg * 60.0)
        # Mechanical focal-plane rotation at el_bore (horizon-frame
        # projection, the parallactic angle is a horizon-to-celestial
        # quantity and does not enter).
        # ``compute_focal_plane_rotation`` only reads
        # ``offset.instrument_rotation``, it ignores ``dx``/``dy``, so
        # the offset is interchangeable for the rotation computation.
        # Use a zero-offset stub here for clarity (``instrument_rotation``
        # defaults to 0). ``center_offset`` is retained for the
        # subsequent ``detector_to_boresight`` call which DOES consume
        # the offset geometry.
        fp_rot_at_bore = float(
            compute_focal_plane_rotation(
                el=el_bore,
                site=site,
                offset=InstrumentOffset(dx=0.0, dy=0.0),
            )
            + boresight_rot_deg
        )
        az_bore_f, _ = detector_to_boresight(
            det_az=src_az_at_el_bore,
            det_el=el_bore,
            offset=center_offset,
            focal_plane_rotation=fp_rot_at_bore,
        )
        az_bore = float(az_bore_f)

    # Mechanical field rotation for the cover projection (horizon-frame,
    # as in the az_bore recovery above). Uses a zero-offset
    # InstrumentOffset (no per-module instrument_rotation; the cover
    # vertices already carry their own focal-plane positions).
    cover_field_rot = float(
        compute_focal_plane_rotation(
            el=el_bore,
            site=site,
            offset=InstrumentOffset(dx=0.0, dy=0.0),
        )
        + boresight_rot_deg
    )
    az_cover, el_cover = _project_cover_to_altaz(fp, az_bore, el_bore, cover_field_rot)
    return az_bore, az_cover, el_cover


def _crossing_window(
    *,
    arc: _SourceArc,
    el_cover: np.ndarray,
    el_bore: float,
    allow_partial: bool,
    dwell: float | None,
    source_label: str,
) -> _CrossingWindow:
    """Guard the footprint coverage, solve the crossing window and narrow it to ``dwell``.

    Raises
    ------
    TargetNotObservableError
        When ``allow_partial`` is false and the arc's elevation span does not
        cover the projected footprint.
    DwellExceedsCrossingError
        When ``dwell`` is longer than the solved crossing.

    Warns
    -----
    PointingWarning
        A partial cover under ``allow_partial``, and a ``dwell`` shorter than
        the crossing.
    """
    t_search_start = arc.t_search_start
    el_slice = arc.el_slice
    t_of_el = arc.t_of_el
    mode = arc.mode

    el_cover_min = float(el_cover.min())
    el_cover_max = float(el_cover.max())
    el_src_min = float(el_slice.min())
    el_src_max = float(el_slice.max())

    if not allow_partial:
        if el_cover_max > el_src_max or el_cover_min < el_src_min:
            raise TargetNotObservableError(
                target=source_label,
                time_info=str(t_search_start.iso),
                bounds_error=ElevationBoundsError(
                    actual_min=el_src_min,
                    actual_max=el_src_max,
                    limit_min=el_cover_min,
                    limit_max=el_cover_max,
                ),
            )
    else:
        if el_cover_max > el_src_max or el_cover_min < el_src_min:
            warnings.warn(
                f"Source {source_label} elevation span "
                f"[{el_src_min:.2f}, {el_src_max:.2f}] does not cover the "
                f"footprint extent [{el_cover_min:.2f}, {el_cover_max:.2f}] "
                f"at el_bore={el_bore:.2f}; proceeding with partial scan.",
                PointingWarning,
                stacklevel=_ADVISORY_STACKLEVEL,
            )

    el_lo = max(el_cover_min, el_src_min)
    el_hi = min(el_cover_max, el_src_max)
    if mode == "rising":
        t0_sec = float(t_of_el(el_lo))
        t1_sec = float(t_of_el(el_hi))
    else:
        t0_sec = float(t_of_el(el_hi))
        t1_sec = float(t_of_el(el_lo))
    if t1_sec < t0_sec:
        t0_sec, t1_sec = t1_sec, t0_sec
    # The full footprint crossing, before any dwell narrowing; reported as
    # ``crossing_seconds`` so a narrowed pass still records what it cut from.
    crossing_seconds = t1_sec - t0_sec
    if dwell is not None:
        # The one site where the dwell acts: narrow the window symmetrically
        # about the crossing midpoint. Everything downstream (drift anchor,
        # arc Sun check, leg speed, quantisation, t0_iso/t1_iso) then
        # describes the narrowed window, i.e. what is actually scanned.
        if dwell > crossing_seconds:
            raise DwellExceedsCrossingError(
                f"dwell must not exceed the solved footprint crossing: got dwell={dwell:.1f} s "
                f"but {source_label} crosses the footprint at el_bore={el_bore:.2f} deg in "
                f"{crossing_seconds:.1f} s (every extra second has the source outside the "
                "footprint; use more passes or a larger footprint margin for more time on "
                "source)",
                dwell=dwell,
                crossing_seconds=crossing_seconds,
            )
        if dwell < crossing_seconds:
            warnings.warn(
                f"dwell={dwell:.1f} s is shorter than the {crossing_seconds:.1f} s footprint "
                f"crossing of {source_label} at el_bore={el_bore:.2f} deg; planning a partial "
                "pass centred on the crossing midpoint.",
                PointingWarning,
                stacklevel=_ADVISORY_STACKLEVEL,
            )
        t_mid_sec = 0.5 * (t0_sec + t1_sec)
        t0_sec = t_mid_sec - 0.5 * dwell
        t1_sec = t_mid_sec + 0.5 * dwell
    t0 = t_search_start + TimeDelta(t0_sec * u.s)
    t1 = t_search_start + TimeDelta(t1_sec * u.s)
    return _CrossingWindow(
        t0_sec=t0_sec,
        t1_sec=t1_sec,
        crossing_seconds=crossing_seconds,
        t0=t0,
        t1=t1,
    )


def _solve_drift(
    *,
    arc: _SourceArc,
    az_cover: np.ndarray,
    el_cover: np.ndarray,
    az_bore: float,
    t0_sec: float,
    v_az: float | None,
    el_bore: float,
    source_label: str,
) -> tuple[float, float, float]:
    """Solve the azimuth drift rate that keeps the source's footprint crossing narrowest.

    Returns ``(v_az, az_start, crossing_throw)``: the drift rate (the
    Nelder-Mead solution, its fallback, or the given ``v_az``), and the low
    edge and width of the azimuth window the source crosses at that rate.

    Raises
    ------
    PointingError
        When the source crosses no footprint vertex at ``el_bore``, or the
        optimisation fails and no median source azimuth speed is usable.

    Warns
    -----
    PointingWarning
        When the optimisation does not converge and the median source azimuth
        speed is used instead.
    """
    el_sorted = arc.el_sorted
    t_sorted = arc.t_sorted
    az_sorted = arc.az_sorted

    def _throw_objective(v_az_candidate: float) -> tuple[float, float]:
        """Return (az_start, throw) for a candidate drift rate."""
        az_residual = az_sorted - v_az_candidate * (t_sorted - t0_sec)
        az_resid_of_el = interpolate.interp1d(
            el_sorted,
            az_residual,
            kind="linear",
            fill_value="extrapolate",  # type: ignore[arg-type]
            assume_sorted=True,
        )
        distances = []
        for av, ev in zip(az_cover, el_cover):
            if not (el_sorted[0] <= ev <= el_sorted[-1]):
                continue
            distances.append(float(az_resid_of_el(ev)) - av)
        if not distances:
            raise PointingError(
                f"Source {source_label} never crosses any footprint vertex at el_bore={el_bore:.2f}"
            )
        distances = np.asarray(distances)
        az_lo_local = distances.min() + az_bore
        az_hi_local = distances.max() + az_bore
        return az_lo_local, az_hi_local - az_lo_local

    if v_az is None:
        res = minimize(
            lambda x: _throw_objective(float(x[0]))[1],
            x0=np.array([0.0]),
            method="Nelder-Mead",
            options={"xatol": 1e-5, "fatol": 1e-4, "maxiter": 200},
        )
        if res.success:
            v_az_solved = float(res.x[0])
        else:
            warnings.warn(
                f"v_az optimisation did not converge for {source_label}; "
                f"falling back to median source az speed.",
                PointingWarning,
                stacklevel=_ADVISORY_STACKLEVEL,
            )
            dt = np.diff(t_sorted)
            dt = np.where(dt == 0, np.nan, dt)
            v_az_solved = float(np.nanmedian(np.diff(az_sorted) / dt))
            if not np.isfinite(v_az_solved):
                raise PointingError(
                    f"v_az optimisation failed and no usable median az speed "
                    f"available for {source_label}."
                )
    else:
        v_az_solved = float(v_az)

    az_start, crossing_throw = _throw_objective(v_az_solved)
    return v_az_solved, az_start, crossing_throw


def _sweep_window(
    *,
    az_start: float,
    crossing_throw: float,
    az_padding: float,
    az_throw: float | None,
    az_branch: float | None,
    el_bore: float,
    source_label: str,
) -> tuple[float, float, float]:
    """Pad the solved crossing into the swept azimuth window and apply its overrides.

    Returns ``(az_start, throw, az_stop)`` of the window the scan sweeps:
    the crossing padded by ``az_padding`` on each side, or replaced by
    ``az_throw`` about the same centre, then re-expressed in the
    ``az_branch`` wrap branch.

    Raises
    ------
    PointingError
        When the window has no width: the source crosses the cover at a single
        azimuth (one vertex of a partial cover in reach, coincident vertices, or
        vertices an explicit ``v_az`` lines up), so the solved crossing has no
        azimuth extent, and neither ``az_padding`` nor ``az_throw`` widens it.

    Warns
    -----
    PointingWarning
        When ``az_throw`` is narrower than the footprint crossing.
    """
    az_start -= az_padding
    throw = crossing_throw + 2 * az_padding
    if az_throw is not None:
        # Replace the padded throw, keeping the solved window's centre so
        # the source still crosses the middle of the sweep.
        if az_throw < crossing_throw:
            warnings.warn(
                f"az_throw={az_throw:.3f} deg is narrower than the {crossing_throw:.3f} deg "
                f"footprint crossing of {source_label} at el_bore={el_bore:.2f} deg; the "
                "source leaves the swept window during the pass.",
                PointingWarning,
                stacklevel=_ADVISORY_STACKLEVEL,
            )
        centre = az_start + 0.5 * throw
        throw = float(az_throw)
        az_start = centre - 0.5 * throw
    elif throw <= 0.0:
        raise PointingError(
            f"Source {source_label} crosses the footprint cover at a single azimuth at a "
            f"boresight elevation of {el_bore:.2f} deg, so with az_padding=0 and no az_throw "
            "the swept azimuth window has no width; give a positive az_padding, or an "
            "az_throw in place of az_padding"
        )
    az_stop = az_start + throw
    # Invariant: every override of the swept window sits above this line, so
    # the arc Sun check of the screen stage sees the final swept envelope.

    if az_branch is not None:
        # Re-express az_start in the requested wrap branch. The shift is a
        # multiple of 360 deg, so the commanded sweep points at the same sky
        # (azimuth is periodic) and source coverage is preserved; an
        # az_branch that pushes the swept window past the ACU az limits
        # surfaces downstream as an AzimuthBoundsError, not as silent loss.
        az_start = (az_start - (az_branch - 180.0)) % 360.0 + (az_branch - 180.0)
        az_stop = az_start + throw
    return az_start, throw, az_stop


def _leg_speed(
    *,
    t0_sec: float,
    t1_sec: float,
    throw: float,
    sampling_step_seconds: float,
    az_speed: float | None,
) -> tuple[float, float]:
    """Return ``(duration_window, velocity)``: the scanned window's length and the leg speed."""
    # Per-leg required speed comes from the underlying ConstantEl
    # pattern; the additional drift adds a small constant offset. Derived
    # ahead of the screen because the Sun sweep needs it to widen the
    # science window into the commanded envelope.
    duration_window = max(t1_sec - t0_sec, sampling_step_seconds)
    if az_speed is not None:
        nominal_velocity = float(az_speed)
    else:
        # Choose the per-leg velocity so the source-coverage window fits one
        # leg, floored at _MIN_PER_LEG_VELOCITY_DEG_S (which binds for most
        # module-scale passes; see its comment). The CE pattern below may
        # adjust n_scans, but the per-leg velocity stays the same.
        nominal_velocity = max(throw / duration_window, _MIN_PER_LEG_VELOCITY_DEG_S)
    return duration_window, nominal_velocity


def _screen_pass(
    *,
    coords: Coordinates,
    site: Site,
    arc: _SourceArc,
    crossing: _CrossingWindow,
    v_az_solved: float,
    az_start: float,
    throw: float,
    el_bore: float,
    nominal_velocity: float,
    az_accel: float,
    sun_safe: SunSafePredicate | None,
    source_label: str,
) -> None:
    """Screen the swept pass against the Sun, then its peak azimuth speed; warns only.

    Warns
    -----
    PointingWarning
        When the commanded azimuth envelope enters the Sun avoidance zone, and
        when the leg speed plus the drift exceeds the site's azimuth velocity
        limit.
    """
    t_search_start = arc.t_search_start
    t0_sec = crossing.t0_sec
    t1_sec = crossing.t1_sec

    # Sweep the commanded azimuth envelope, not the science window: the
    # turnarounds overshoot each science edge, so the mount goes further
    # than [az_start, az_stop] on both sides. A one- or two-leg pass has
    # fewer than two turnarounds and is therefore screened slightly wider
    # than it is driven; over-warning is the safe direction for a
    # warn-only check.
    arc_times_sec_base = np.linspace(t0_sec, t1_sec, _SUN_SAFETY_ARC_N_SAMPLES)
    drift = v_az_solved * (arc_times_sec_base - t0_sec)
    arc_az, arc_el, arc_times = _swept_arc_samples(
        az_min=az_start + drift,
        az_throw=throw,
        el_deg=el_bore,
        times=t_search_start + TimeDelta(arc_times_sec_base * u.s),
        az_speed=nominal_velocity,
        az_accel=az_accel,
    )
    _check_arc_sun_safety(
        coords,
        site,
        arc_az,
        arc_el,
        arc_times,
        f"source-CES on {source_label}",
        sun_safe=sun_safe,
        # One frame more than the stage's own advisories: the check warns itself.
        stacklevel=_ADVISORY_STACKLEVEL + 1,
    )

    az_vel_limit = site.telescope_limits.azimuth.max_velocity
    peak_required = nominal_velocity + abs(v_az_solved)
    if peak_required > az_vel_limit:
        warnings.warn(
            f"Required peak azimuth speed {peak_required:.3f} deg/s for "
            f"source-CES on {source_label} exceeds site limit "
            f"{az_vel_limit:.3f} deg/s.",
            PointingWarning,
            stacklevel=_ADVISORY_STACKLEVEL,
        )


def _assemble(
    *,
    arc: _SourceArc,
    crossing: _CrossingWindow,
    az_start: float,
    throw: float,
    az_stop: float,
    v_az_solved: float,
    nominal_velocity: float,
    duration_window: float,
    az_accel: float,
    el_bore: float,
    boresight_rot_deg: float,
    source_label: str,
    fp: ArrayFootprint,
) -> _SourceCESCore:
    """Quantise the pass to whole legs and assemble the computed parameters and builder state."""
    t0 = crossing.t0
    t1 = crossing.t1
    crossing_seconds = crossing.crossing_seconds
    mode = arc.mode
    src_ra_at_el_bore = arc.src_ra_at_el_bore
    src_dec_at_el_bore = arc.src_dec_at_el_bore

    # Quantise duration the same way the trajectory builder does so the returned
    # ``actual_duration`` matches what the trajectory builder will
    # produce. Mirrors the n_scans/duration quantisation in plan_constant_el_scan.
    n_scans, actual_duration = _quantize_ce_duration(
        az_throw=throw,
        velocity=nominal_velocity,
        duration=duration_window,
        az_accel=az_accel,
    )

    computed: SourceCESComputedParams = {
        "az_start": float(az_start),
        "az_throw": float(throw),
        "az_speed": float(nominal_velocity),
        "v_az": float(v_az_solved),
        "el_bore": float(el_bore),
        "boresight_rot": float(boresight_rot_deg),
        # ``.utc`` before ``.iso``: the recorded strings carry no scale, and
        # every reader parses them as UTC (which the TypedDict documents), so
        # the conversion has to happen here rather than at the reader.
        "t0_iso": str(t0.utc.iso),
        "t1_iso": str(t1.utc.iso),
        "duration": float(actual_duration),
        "crossing_seconds": float(crossing_seconds),
        "mode": mode,
        "n_scans": int(n_scans),
    }
    # Direct self-check against the TypedDict's required keys.
    # ``source_ces`` is intentionally not registered in
    # :data:`fyst_trajectories.planning._types._SCAN_TYPE_TO_KEYS`;
    # see the note there for the planning<->overhead boundary rationale.
    _missing = SourceCESComputedParams.__required_keys__ - computed.keys()
    if _missing:
        raise KeyError(f"source_ces computed_params missing required keys: {sorted(_missing)}")

    return _SourceCESCore(
        computed=computed,
        actual_duration=float(actual_duration),
        t0=t0,
        n_scans=int(n_scans),
        mode=mode,
        el_bore=el_bore,
        velocity=float(nominal_velocity),
        az_stop=float(az_stop),
        source_label=source_label,
        fp=fp,
        src_ra_at_el_bore=float(src_ra_at_el_bore),
        src_dec_at_el_bore=float(src_dec_at_el_bore),
    )


def _compute_source_ces_core(
    *,
    # --- Source ---
    source: _SourceSpec,
    # --- Footprint ---
    footprint: InstrumentOffset | str | Sequence[InstrumentOffset] | ArrayFootprint,
    # --- Geometry ---
    el_bore: float,
    boresight_rot: float | None = None,
    # --- Time window ---
    window: tuple[Time, Time] | None = None,
    night: Time | None = None,
    mode: Literal["rising", "setting"] | None = None,
    # --- Site / atmosphere ---
    site: Site,
    atmosphere: AtmosphericConditions | None = None,
    # --- Algorithm ---
    sampling_step_seconds: float = 30.0,
    az_accel: float = 1.0,
    az_padding: float = _DEFAULT_AZ_PADDING_DEG,
    az_branch: float | None = None,
    allow_partial: bool = False,
    v_az: float | None = None,
    sun_safe: SunSafePredicate | None = None,
    # --- Scan-geometry overrides ---
    az_speed: float | None = None,
    az_throw: float | None = None,
    dwell: float | None = None,
    # --- Screen switch ---
    screen: bool = True,
) -> _SourceCESCore:
    """Run the params-only phase of source-CES planning, returning scalars + builder state.

    Shared compute kernel for :func:`plan_source_ces` and
    :func:`compute_source_ces_params`. Validates inputs, resolves the
    footprint, samples the source arc, picks the monotonic slice, recovers
    ``az_bore``, projects the cover, derives ``(t0, t1)`` (narrowed to
    ``dwell`` when given), solves ``v_az``, applies the ``az_throw``
    override, runs the sun-safety arc check, and computes the
    peak-velocity sanity warning from ``az_speed`` or the derived leg
    speed. Does NOT build the per-sample trajectory.

    Each step is a stage function, called in that order. ``screen=False``
    skips the Sun sweep and the peak-speed advisory and changes nothing
    else: both only warn. The ``start_time`` anchor's probe sets it, since
    it reads only the pass start.
    """
    _validate_kernel_inputs(
        source=source,
        footprint=footprint,
        el_bore=el_bore,
        window=window,
        night=night,
        mode=mode,
        site=site,
        sampling_step_seconds=sampling_step_seconds,
        az_accel=az_accel,
        az_padding=az_padding,
        az_speed=az_speed,
        az_throw=az_throw,
        dwell=dwell,
    )

    boresight_rot_deg = 0.0 if boresight_rot is None else float(boresight_rot)
    source_label = source.label

    fp = resolve_footprint(footprint)
    coords = Coordinates(site, atmosphere=atmosphere)

    arc = _solve_source_arc(
        source=source,
        coords=coords,
        site=site,
        el_bore=el_bore,
        window=window,
        night=night,
        mode=mode,
        sampling_step_seconds=sampling_step_seconds,
        source_label=source_label,
    )
    az_bore, az_cover, el_cover = _project_footprint(
        fp=fp,
        site=site,
        el_bore=el_bore,
        boresight_rot_deg=boresight_rot_deg,
        src_az_at_el_bore=arc.src_az_at_el_bore,
    )
    crossing = _crossing_window(
        arc=arc,
        el_cover=el_cover,
        el_bore=el_bore,
        allow_partial=allow_partial,
        dwell=dwell,
        source_label=source_label,
    )
    v_az_solved, az_start, crossing_throw = _solve_drift(
        arc=arc,
        az_cover=az_cover,
        el_cover=el_cover,
        az_bore=az_bore,
        t0_sec=crossing.t0_sec,
        v_az=v_az,
        el_bore=el_bore,
        source_label=source_label,
    )
    az_start, throw, az_stop = _sweep_window(
        az_start=az_start,
        crossing_throw=crossing_throw,
        az_padding=az_padding,
        az_throw=az_throw,
        az_branch=az_branch,
        el_bore=el_bore,
        source_label=source_label,
    )
    duration_window, nominal_velocity = _leg_speed(
        t0_sec=crossing.t0_sec,
        t1_sec=crossing.t1_sec,
        throw=throw,
        sampling_step_seconds=sampling_step_seconds,
        az_speed=az_speed,
    )
    if screen:
        _screen_pass(
            coords=coords,
            site=site,
            arc=arc,
            crossing=crossing,
            v_az_solved=v_az_solved,
            az_start=az_start,
            throw=throw,
            el_bore=el_bore,
            nominal_velocity=nominal_velocity,
            az_accel=az_accel,
            sun_safe=sun_safe,
            source_label=source_label,
        )
    return _assemble(
        arc=arc,
        crossing=crossing,
        az_start=az_start,
        throw=throw,
        az_stop=az_stop,
        v_az_solved=v_az_solved,
        nominal_velocity=nominal_velocity,
        duration_window=duration_window,
        az_accel=az_accel,
        el_bore=el_bore,
        boresight_rot_deg=boresight_rot_deg,
        source_label=source_label,
        fp=fp,
    )
