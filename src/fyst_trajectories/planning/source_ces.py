"""Source-tracking constant-elevation scan planner.

Plans a constant-elevation scan that drags a moving source (planet or
sidereal point) across the focal-plane footprint of an instrument
array. Mirrors :func:`schedlib.source.make_source_ces` from Simons
Observatory's scheduler (https://github.com/simonsobs/scheduler).

The function is the source-tracking sibling of
:func:`plan_constant_el_scan`: where the latter aims at a fixed RA/Dec
rectangle and lets the source's natural sidereal motion fill the time
axis, this planner aims at a single moving source and solves for an
*additional* azimuth-drift rate ``v_az`` so the source sweeps across
the *entire* footprint at fixed boresight elevation. The output is a
``ScanBlock`` whose ``trajectory`` is a constant-elevation scan with
the solved drift baked into the azimuth track.

``plan_source_ces`` is planner-only; at dispatch time the PCS
``source_scan`` task consumes it. The params-only sibling
:func:`compute_source_ces_params` is the emit-time entry point for a
scheduler.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta
from scipy import interpolate
from scipy.optimize import minimize

from ..coordinates import Coordinates
from ..exceptions import (
    AzimuthBoundsError,
    ElevationBoundsError,
    PointingError,
    PointingWarning,
    TargetNotObservableError,
)
from ..offsets import (
    InstrumentOffset,
    boresight_to_detector,
    compute_focal_plane_rotation,
    detector_to_boresight,
)
from ..patterns.base import TrajectoryMetadata
from ..patterns.configs import ConstantElScanConfig
from ..primecam import MODULE_FOV_RADIUS_DEG, get_primecam_offset
from ..site import AtmosphericConditions, Site
from ..trajectory_utils import validate_trajectory_bounds
from ._helpers import _build_altaz_trajectory
from ._types import (
    ArrayFootprint,
    ScanBlock,
    SourceCESComputedParams,
)

if TYPE_CHECKING:
    # Annotation-only import to avoid an import cycle: ``dispatch`` imports
    # ``coordinates``/``site``/``exceptions`` at runtime. The predicate is
    # invoked structurally, so only the type hint needs the symbol.
    from ..dispatch import SunSafePredicate

__all__ = ["compute_source_ces_params", "plan_source_ces"]


# Number of polygon vertices used when constructing a circular cover
# from a single InstrumentOffset or list of offsets. SO's
# ``make_circular_cover`` uses the same value.
_CIRCULAR_COVER_N_VERTICES = 50

# Default search-window horizon when only ``night`` is supplied. 24 h
# covers a full diurnal rotation; sources reachable from FYST always
# have a rising and setting pass within this window (or never reach the
# requested elevation, which we catch separately).
_DEFAULT_SEARCH_HORIZON_HOURS = 24.0

# Floor on the per-leg azimuth velocity used to seed the underlying
# ConstantEl pattern when ``az_throw / duration_window`` would otherwise
# produce a meaninglessly small value (e.g. a source pass long enough
# that the natural rate is sub-mdeg/s). Keeps the leg/turnaround
# quantisation well-conditioned without affecting the solved drift.
_MIN_PER_LEG_VELOCITY_DEG_S = 0.05

# Number of samples drawn along the planned arc for the sun-safety
# sweep (see ``_check_arc_sun_safety``). 60 samples over a typical 10-minute arc gives ~10 s
# resolution, finer than the sun's apparent motion (~15"/s) and the
# array's footprint extent.
_SUN_SAFETY_ARC_N_SAMPLES = 60


@dataclass(frozen=True)
class _SourceCESCore:
    """Internal carrier for the params-only phase of source-CES planning.

    Holds the completed :class:`SourceCESComputedParams` plus the intermediate
    state the trajectory builder in ``plan_source_ces`` needs.
    Strictly private — no public API guarantees on this class.
    """

    computed: SourceCESComputedParams
    # Intermediate state for the trajectory builder.
    actual_duration: float
    t0: Time
    nominal_velocity: float
    n_scans: int
    mode: str
    velocity: float
    az_stop: float
    source_label: str
    fp: ArrayFootprint
    # Source coords at ``t_at_el_bore`` for the trajectory metadata.
    src_ra_at_el_bore: float
    src_dec_at_el_bore: float


def _resolve_footprint(
    footprint: InstrumentOffset | str | Sequence[InstrumentOffset] | ArrayFootprint,
) -> ArrayFootprint:
    """Normalise a user-supplied footprint into an :class:`ArrayFootprint`.

    Accepts:

    * ``ArrayFootprint`` — returned unchanged.
    * ``InstrumentOffset`` — built as a ``_CIRCULAR_COVER_N_VERTICES``-vertex
      circle of radius :data:`MODULE_FOV_RADIUS_DEG` around the offset.
    * ``str`` — resolved via :func:`get_primecam_offset` then treated as
      the single-``InstrumentOffset`` case.
    * Sequence of ``InstrumentOffset`` — each treated as a circle, the
      cover is the concatenation of per-module vertex lists, and the
      aggregate center is the arithmetic mean of per-module ``(dx, dy)``.
    """
    if isinstance(footprint, ArrayFootprint):
        return footprint

    if isinstance(footprint, str):
        footprint = get_primecam_offset(footprint)

    if isinstance(footprint, InstrumentOffset):
        return _module_circular_cover(footprint)

    if isinstance(footprint, Sequence):
        offsets = list(footprint)
        if not offsets:
            raise ValueError("footprint sequence cannot be empty")
        if not all(isinstance(o, InstrumentOffset) for o in offsets):
            raise TypeError("footprint sequence must contain only InstrumentOffset instances")
        per_module = [_module_circular_cover(o) for o in offsets]
        cover_xi = np.concatenate([f.cover_xi_deg for f in per_module])
        cover_eta = np.concatenate([f.cover_eta_deg for f in per_module])
        center_xi = float(np.mean([f.center_xi_deg for f in per_module]))
        center_eta = float(np.mean([f.center_eta_deg for f in per_module]))
        return ArrayFootprint(
            center_xi_deg=center_xi,
            center_eta_deg=center_eta,
            cover_xi_deg=cover_xi,
            cover_eta_deg=cover_eta,
        )

    raise TypeError(
        f"footprint must be InstrumentOffset, str, sequence of InstrumentOffset, "
        f"or ArrayFootprint; got {type(footprint).__name__}"
    )


def _module_circular_cover(offset: InstrumentOffset) -> ArrayFootprint:
    """Build a circular cover polygon for a single module offset."""
    theta = np.linspace(0.0, 2.0 * np.pi, _CIRCULAR_COVER_N_VERTICES, endpoint=False)
    cover_xi = offset.dx_deg + MODULE_FOV_RADIUS_DEG * np.cos(theta)
    cover_eta = offset.dy_deg + MODULE_FOV_RADIUS_DEG * np.sin(theta)
    return ArrayFootprint(
        center_xi_deg=offset.dx_deg,
        center_eta_deg=offset.dy_deg,
        cover_xi_deg=cover_xi,
        cover_eta_deg=cover_eta,
    )


def _enumerate_monotonic_arcs(el_src: np.ndarray) -> list[tuple[int, int]]:
    """Enumerate monotonic arcs in an elevation trace.

    Returns a list of ``(i_start, i_end_inclusive)`` index pairs marking
    every maximal monotonic sub-arc of ``el_src``. Window endpoints are
    always treated as arc boundaries so an arc that begins or ends
    mid-rise/fall is still captured.

    Plateaus (consecutive samples with identical elevation) are absorbed
    into the adjacent monotonic run rather than treated as separate
    extrema; this is a numerical robustness measure — astronomical
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


def _describe_source(
    body: str | None,
    ra: float | None,
    dec: float | None,
) -> str:
    """Human-readable source label for error messages and summaries."""
    if body is not None:
        return body.capitalize()
    return f"RA={ra:.3f}, Dec={dec:.3f}"


def _sample_source_altaz(
    coords: Coordinates,
    times: Time,
    *,
    body: str | None,
    ra: float | None,
    dec: float | None,
    pm_ra: float,
    pm_dec: float,
    ref_epoch: Time | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a source's (az, el) at the given times."""
    if body is not None:
        return coords.get_body_altaz(body, times)
    if pm_ra != 0.0 or pm_dec != 0.0:
        # radec_to_altaz_with_pm is scalar-time only; loop. For the typical
        # 30 s x 24 h = 2880 samples this is ~0.5 s in practice, acceptable
        # for a planning call and avoids forcing a vectorised PM variant.
        # If it becomes a bottleneck, vectorise in Coordinates.
        n = len(times)
        az = np.empty(n)
        el = np.empty(n)
        for i, t in enumerate(times):
            az[i], el[i] = coords.radec_to_altaz_with_pm(
                ra=ra,
                dec=dec,
                pm_ra=pm_ra,
                pm_dec=pm_dec,
                ref_epoch=ref_epoch,
                obstime=t,
            )
        return az, el
    return coords.radec_to_altaz(ra, dec, times)


def _source_radec_at(
    coords: Coordinates,
    obstime: Time,
    *,
    body: str | None,
    ra: float | None,
    dec: float | None,
) -> tuple[float, float]:
    """Return source RA/Dec at a single instant (for trajectory metadata)."""
    if body is not None:
        return coords.get_body_radec(body, obstime)
    return float(ra), float(dec)


def _project_cover_to_altaz(
    footprint: ArrayFootprint,
    az_bore: float,
    el_bore: float,
    field_rotation_deg: float,
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
            field_rotation=field_rotation_deg,
        )
        az_cover[i] = az_v
        el_cover[i] = el_v
    return az_cover, el_cover


def _check_arc_sun_safety(
    coords: Coordinates,
    site: Site,
    az_arr: np.ndarray,
    el_arr: np.ndarray,
    times: Time,
    source_label: str,
    sun_safe: SunSafePredicate | None = None,
) -> None:
    """Coarse sun-safety check along an arc; warns only.

    Computes sun separation at every sample and emits a single warning
    naming the closest approach if any point falls inside the exclusion
    radius. Boresight-level (one point per time), which is appropriate
    for a CES whose footprint extent is small relative to the
    exclusion radius.

    When ``sun_safe`` is ``None`` (default) the built-in vectorised
    scalar-radius check runs unchanged. When a predicate is injected it
    is consulted per-sample ``(az_i, el_i, time_i)`` instead, so the
    directional sun-avoidance model (future shared library) is honored;
    the warn-only semantics are preserved either way.
    """
    if not site.sun_avoidance.enabled:
        return

    if sun_safe is None:
        sun_az_arr, sun_el_arr = coords.get_sun_altaz(times)
        # Vectorised haversine on the sphere (in degrees) so we don't have
        # to call angular_separation in a loop of 60.
        az_rad = np.deg2rad(az_arr)
        el_rad = np.deg2rad(el_arr)
        sun_az_rad = np.deg2rad(np.asarray(sun_az_arr, dtype=float))
        sun_el_rad = np.deg2rad(np.asarray(sun_el_arr, dtype=float))
        cos_sep = np.sin(el_rad) * np.sin(sun_el_rad) + np.cos(el_rad) * np.cos(
            sun_el_rad
        ) * np.cos(az_rad - sun_az_rad)
        seps_deg = np.rad2deg(np.arccos(np.clip(cos_sep, -1.0, 1.0)))

        excl = site.sun_avoidance.exclusion_radius
        inside = seps_deg <= excl
        if not np.any(inside):
            return
        closest = int(np.argmin(seps_deg))
        # ``Time.__getitem__`` returns ``Time``; pyright's stubs sometimes
        # narrow it to ``Time | None`` because the dunder is generic. Coerce
        # via ``str()`` and silence the spurious optional-access warning.
        closest_iso = str(times[closest].iso)  # type: ignore[union-attr]
        warnings.warn(
            f"EXCLUSION ZONE: planned source-CES on {source_label} passes "
            f"{seps_deg[closest]:.1f} deg from the Sun at "
            f"{closest_iso} (exclusion radius {excl} deg).",
            PointingWarning,
            stacklevel=3,
        )
        return

    # Injected directional model: consult it per-sample. ``False`` marks an
    # unsafe (inside-the-zone) sample. Warn once, naming the first unsafe
    # sample, mirroring the scalar branch's single-warning semantics.
    unsafe_idx = [
        i
        for i in range(len(az_arr))
        if not sun_safe(float(az_arr[i]), float(el_arr[i]), times[i])  # type: ignore[index]
    ]
    if not unsafe_idx:
        return
    first = unsafe_idx[0]
    first_iso = str(times[first].iso)  # type: ignore[union-attr]
    warnings.warn(
        f"EXCLUSION ZONE: planned source-CES on {source_label} enters the Sun "
        f"avoidance zone at (az={float(az_arr[first]):.1f} deg, "
        f"el={float(el_arr[first]):.1f} deg) at {first_iso}.",
        PointingWarning,
        stacklevel=3,
    )


def _compute_source_ces_core(
    *,
    # --- Source ---
    body: str | None = None,
    ra: float | None = None,
    dec: float | None = None,
    pm_ra: float = 0.0,
    pm_dec: float = 0.0,
    ref_epoch: Time | None = None,
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
    az_padding: float = 0.5,
    az_branch: float | None = None,
    allow_partial: bool = False,
    v_az: float | None = None,
    sun_safe: SunSafePredicate | None = None,
) -> _SourceCESCore:
    """Run the params-only phase of source-CES planning, returning scalars + builder state.

    Shared compute kernel for :func:`plan_source_ces` and
    :func:`compute_source_ces_params`. Validates inputs, resolves the
    footprint, samples the source arc, picks the monotonic slice, recovers
    ``az_bore``, projects the cover, derives ``(t0, t1)``, solves ``v_az``,
    runs the sun-safety arc check, and computes the peak-velocity sanity
    warning. Does NOT build the per-sample trajectory.
    """
    has_body = body is not None
    has_radec = ra is not None or dec is not None
    if has_body and has_radec:
        raise ValueError("specify either 'body' or 'ra'/'dec', not both")
    if not has_body:
        if ra is None or dec is None:
            raise ValueError("must specify 'body' or both 'ra' and 'dec'")
        if (pm_ra != 0.0 or pm_dec != 0.0) and ref_epoch is None:
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

    el_limits = site.telescope_limits.elevation
    if not (el_limits.min <= el_bore <= el_limits.max):
        raise ElevationBoundsError(
            actual_min=el_bore,
            actual_max=el_bore,
            limit_min=el_limits.min,
            limit_max=el_limits.max,
        )

    boresight_rot_deg = 0.0 if boresight_rot is None else float(boresight_rot)
    source_label = _describe_source(body, ra, dec)

    fp = _resolve_footprint(footprint)
    coords = Coordinates(site, atmosphere=atmosphere)

    if has_window:
        assert window is not None  # narrow for type-checker; guaranteed by has_window
        t_search_start, t_search_end = window
        horizon_seconds = (t_search_end - t_search_start).to_value(u.s)
    else:
        assert night is not None  # narrow for type-checker; guaranteed by validation above
        t_search_start = night
        horizon_seconds = _DEFAULT_SEARCH_HORIZON_HOURS * 3600.0

    dt_sec = np.arange(0.0, horizon_seconds, sampling_step_seconds)
    search_times = t_search_start + TimeDelta(dt_sec * u.s)

    az_src, el_src = _sample_source_altaz(
        coords,
        search_times,
        body=body,
        ra=ra,
        dec=dec,
        pm_ra=pm_ra,
        pm_dec=pm_dec,
        ref_epoch=ref_epoch,
    )

    # The window can span multiple local extrema (e.g. a 24 h window on a
    # planet near a culmination contains both a max and an anti-culmination
    # minimum). Naively picking ``(argmin, argmax)`` selects the dominant
    # extrema regardless of order in time, producing an empty or reversed
    # slice when the global min happens *after* the global max within the
    # window. We instead enumerate all monotonic arcs between consecutive
    # local extrema (with window endpoints inserted as "virtual" extrema
    # so an arc that begins or ends mid-rise/fall is still captured) and
    # pick the first arc whose elevation range covers ``el_bore``.
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

    t_slice = dt_sec[i_beg:i_end]
    az_slice = np.unwrap(np.deg2rad(az_src[i_beg:i_end]))
    az_slice = np.rad2deg(az_slice)
    el_slice = el_src[i_beg:i_end]

    if not (el_slice.min() <= el_bore <= el_slice.max()):
        raise TargetNotObservableError(
            target=source_label,
            time_info=str(t_search_start.iso),
            bounds_error=ElevationBoundsError(
                actual_min=float(el_slice.min()),
                actual_max=float(el_slice.max()),
                limit_min=el_bore,
                limit_max=el_bore,
            ),
        )

    # interp1d wants sorted x. For rising slice el is increasing; for
    # setting slice el is decreasing; sort by el value either way.
    sort_idx = np.argsort(el_slice)
    el_sorted = el_slice[sort_idx]
    t_sorted = t_slice[sort_idx]
    az_sorted = az_slice[sort_idx]
    # Linear interpolation for ``t(el)``: cubic is liable to overshoot
    # near the arc apex where ``dEl/dt -> 0`` and would silently
    # extrapolate out-of-range queries. The guard above already
    # ensures ``el_bore`` is inside ``[el_slice.min(), el_slice.max()]``;
    # the assertion below pins that invariant for any future code change.
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
    src_ra_at_el_bore, src_dec_at_el_bore = _source_radec_at(
        coords,
        t_at_el_bore,
        body=body,
        ra=ra,
        dec=dec,
    )

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
        try:
            az_bore_f, _ = detector_to_boresight(
                det_az=src_az_at_el_bore,
                det_el=el_bore,
                offset=center_offset,
                field_rotation=fp_rot_at_bore,
            )
            az_bore = float(az_bore_f)
        except RuntimeError as exc:
            # Fall back to az = source az if the inverse fails (zenith
            # singularity); warn so the caller knows.
            warnings.warn(
                f"detector_to_boresight failed for off-centre footprint "
                f"({exc}); falling back to source azimuth as boresight.",
                PointingWarning,
                stacklevel=2,
            )
            az_bore = src_az_at_el_bore

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
                stacklevel=2,
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
    t0 = t_search_start + TimeDelta(t0_sec * u.s)
    t1 = t_search_start + TimeDelta(t1_sec * u.s)

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
                stacklevel=2,
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

    az_start, az_throw = _throw_objective(v_az_solved)
    az_start -= az_padding
    az_throw += 2 * az_padding
    az_stop = az_start + az_throw

    if az_branch is not None:
        # Re-express az_start in the requested wrap branch. The shift is a
        # multiple of 360 deg, so the commanded sweep points at the same sky
        # (azimuth is periodic) and source coverage is preserved; an
        # az_branch that pushes the swept window past the ACU az limits
        # surfaces downstream as an AzimuthBoundsError, not as silent loss.
        az_start = (az_start - (az_branch - 180.0)) % 360.0 + (az_branch - 180.0)
        az_stop = az_start + az_throw

    # Probe three azimuth positions per time sample (the low edge, the
    # midpoint, and the high edge of the sweep) so that a scan whose
    # midpoint clears the exclusion radius but whose +/-throw/2 edges do
    # not is still caught. Sun motion within a single sweep is
    # negligible (~15"/s << az_throw), so reusing the same time at all
    # three az positions is sound.
    arc_n = _SUN_SAFETY_ARC_N_SAMPLES
    arc_times_sec_base = np.linspace(t0_sec, t1_sec, arc_n)
    drift = v_az_solved * (arc_times_sec_base - t0_sec)
    arc_az_lo = az_start + drift
    arc_az_mid = arc_az_lo + az_throw / 2.0
    arc_az_hi = arc_az_lo + az_throw
    arc_az = np.concatenate([arc_az_lo, arc_az_mid, arc_az_hi])
    arc_el = np.full(arc_n * 3, el_bore)
    arc_times_sec = np.tile(arc_times_sec_base, 3)
    arc_times = t_search_start + TimeDelta(arc_times_sec * u.s)
    _check_arc_sun_safety(coords, site, arc_az, arc_el, arc_times, source_label, sun_safe=sun_safe)

    az_vel_limit = site.telescope_limits.azimuth.max_velocity
    # Per-leg required speed comes from the underlying ConstantEl
    # pattern; the additional drift adds a small constant offset.
    duration_window = max(t1_sec - t0_sec, sampling_step_seconds)
    # Tentatively choose the per-leg velocity so the source-coverage
    # window fits one full cycle (down + up). The CE pattern below
    # may adjust n_scans, but the per-leg velocity stays the same.
    nominal_velocity = max(az_throw / duration_window, _MIN_PER_LEG_VELOCITY_DEG_S)
    peak_required = nominal_velocity + abs(v_az_solved)
    if peak_required > az_vel_limit:
        warnings.warn(
            f"Required peak azimuth speed {peak_required:.3f} deg/s for "
            f"source-CES on {source_label} exceeds site limit "
            f"{az_vel_limit:.3f} deg/s.",
            PointingWarning,
            stacklevel=2,
        )

    # Quantise duration the same way the trajectory builder does so the returned
    # ``actual_duration`` matches what the trajectory builder will
    # produce. Mirrors the n_scans/duration quantisation in plan_constant_el_scan.
    velocity = nominal_velocity
    scan_leg_time = az_throw / velocity
    n_scans = max(1, round(duration_window / scan_leg_time))
    t_turnaround = 2.0 * velocity / az_accel
    t_cruise = az_throw / velocity
    actual_duration = n_scans * t_cruise + max(0, n_scans - 1) * t_turnaround

    computed: SourceCESComputedParams = {
        "az_start": float(az_start),
        "az_throw": float(az_throw),
        "v_az": float(v_az_solved),
        "el_bore": float(el_bore),
        "boresight_rot": float(boresight_rot_deg),
        "t0_iso": str(t0.iso),
        "t1_iso": str(t1.iso),
        "duration": float(actual_duration),
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
        nominal_velocity=float(nominal_velocity),
        n_scans=int(n_scans),
        mode=mode,
        velocity=float(velocity),
        az_stop=float(az_stop),
        source_label=source_label,
        fp=fp,
        src_ra_at_el_bore=float(src_ra_at_el_bore),
        src_dec_at_el_bore=float(src_dec_at_el_bore),
    )


def compute_source_ces_params(
    *,
    # --- Source ---
    body: str | None = None,
    ra: float | None = None,
    dec: float | None = None,
    pm_ra: float = 0.0,
    pm_dec: float = 0.0,
    ref_epoch: Time | None = None,
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
    az_padding: float = 0.5,
    az_branch: float | None = None,
    allow_partial: bool = False,
    v_az: float | None = None,
    sun_safe: SunSafePredicate | None = None,
) -> SourceCESComputedParams:
    """Compute source-CES scalar parameters without building the trajectory.

    Params-only sibling of :func:`plan_source_ces`. Returns just the
    :class:`SourceCESComputedParams` dict (az_start, az_throw, v_az,
    el_bore, boresight_rot, t0_iso, t1_iso, duration, mode, n_scans),
    skipping the per-sample trajectory generation. This is the emit-time
    entry point: a scheduler can price many candidate scans cheaply
    (feasibility, duration, azimuth throw) from the scalars alone and
    discard the trajectory, which the execution layer generates once at
    dispatch.

    All keyword arguments are identical to :func:`plan_source_ces`
    except that ``timestep`` is omitted — only the trajectory builder
    consumes it. See :func:`plan_source_ces` for full parameter
    documentation.

    Parameters
    ----------
    body : str, optional
        Solar-system body name; mutually exclusive with ``ra``/``dec``.
    ra, dec : float, optional
        Sidereal source position in degrees.
    pm_ra, pm_dec : float, optional
        Proper motion in mas/yr.
    ref_epoch : Time, optional
        Reference epoch for ``ra``/``dec``.
    footprint : InstrumentOffset, str, sequence of InstrumentOffset, or ArrayFootprint
        On-sky cover that the source must traverse.
    el_bore : float
        Fixed boresight elevation in degrees.
    boresight_rot : float, optional
        Mechanical boresight rotation in degrees.
    window : (Time, Time), optional
        Explicit search window.
    night : Time, optional
        Start of the search window (use with ``mode``).
    mode : {"rising", "setting"}, optional
        Direction of the source arc.
    site : Site
        Telescope site.
    atmosphere : AtmosphericConditions, optional
        Refraction model (default vacuum).
    sampling_step_seconds : float, optional
        Coarse time step for source sampling. Default 30.0.
    az_accel : float, optional
        Azimuth acceleration in deg/s^2. Default 1.0.
    az_padding : float, optional
        Extra azimuth padding on each side, in degrees. Default 0.5.
    az_branch : float, optional
        Centre of azimuth wrap branch.
    allow_partial : bool, optional
        If ``True``, downgrade footprint-not-fully-covered to a warning.
    v_az : float, optional
        Override the solved azimuth drift rate (deg/s).
    sun_safe : SunSafePredicate, optional
        Sun-safety predicate implementing the
        :class:`~fyst_trajectories.dispatch.SunSafePredicate` contract. ``None``
        (default) keeps the built-in scalar exclusion-radius arc check; an
        injected predicate is consulted per-sample along the planned arc
        instead, so the directional sun-avoidance model (future shared library)
        is honored. Warn-only either way.

    Returns
    -------
    SourceCESComputedParams
        Scalar parameters describing the planned source-CES — the same
        dict that ``plan_source_ces(...).computed_params`` returns.

    Raises
    ------
    ValueError
        On incompatible argument combinations.
    TargetNotObservableError
        When the source never reaches ``el_bore`` in the search window.
    AzimuthBoundsError
        When the envelope ``[az_start - az_padding, az_start + az_throw
        + az_padding]`` extended by the drift across ``(t1 - t0)``
        exceeds ``site.telescope_limits.azimuth``. This is a cheap
        pre-build check; :func:`plan_source_ces` runs a stricter
        per-sample check via ``validate_trajectory_bounds``.
    PointingError
        When the Nelder-Mead optimisation fails and no fallback ``v_az``
        can be derived from the source's median az speed.

    Warns
    -----
    PointingWarning
        Same warnings as :func:`plan_source_ces` (sun-avoidance,
        peak-velocity sanity).

    Notes
    -----
    See :func:`plan_source_ces` for the same computation plus per-sample
    trajectory generation. The params-only path avoids ~370 KB of
    trajectory arrays and ~10-20 ms of vectorised compute on a typical
    15-minute Jupiter scan at ``timestep=0.1`` — a meaningful saving
    when an upstream scheduler emits dozens of source_ces blocks per
    tactical pass and discards the trajectory.

    Examples
    --------
    Compute source-CES scalars for a Jupiter rising scan on PrimeCam's
    centre module, without building the trajectory:

    >>> from astropy.time import Time
    >>> from fyst_trajectories import get_fyst_site
    >>> from fyst_trajectories.planning import compute_source_ces_params
    >>> params = compute_source_ces_params(  # doctest: +SKIP
    ...     body="jupiter",
    ...     footprint="c",
    ...     el_bore=35.0,
    ...     night=Time("2026-03-15T00:00:00", scale="utc"),
    ...     mode="rising",
    ...     site=get_fyst_site(),
    ... )
    >>> # params is a SourceCESComputedParams (TypedDict / plain dict).
    """
    core = _compute_source_ces_core(
        body=body,
        ra=ra,
        dec=dec,
        pm_ra=pm_ra,
        pm_dec=pm_dec,
        ref_epoch=ref_epoch,
        footprint=footprint,
        el_bore=el_bore,
        boresight_rot=boresight_rot,
        window=window,
        night=night,
        mode=mode,
        site=site,
        atmosphere=atmosphere,
        sampling_step_seconds=sampling_step_seconds,
        az_accel=az_accel,
        az_padding=az_padding,
        az_branch=az_branch,
        allow_partial=allow_partial,
        v_az=v_az,
        sun_safe=sun_safe,
    )

    # Envelope-only az bounds check. The padded sweep [az_start, az_stop]
    # plus the linear drift across the source pass duration gives the
    # extreme az values the executed trajectory will hit, without
    # building per-sample arrays. ``plan_source_ces`` runs the stricter
    # per-sample check via ``validate_trajectory_bounds`` after building.
    az_limits = site.telescope_limits.azimuth
    cp = core.computed
    pass_duration = max(core.actual_duration, 0.0)
    drift_total = cp["v_az"] * pass_duration
    env_lo = min(cp["az_start"], cp["az_start"] + cp["az_throw"])
    env_hi = max(cp["az_start"], cp["az_start"] + cp["az_throw"])
    # The executed trajectory applies ``az + v_az*times`` (see
    # ``plan_source_ces``), so the linear drift shifts the track in a
    # single direction (the sign of ``v_az``): later samples move toward
    # ``+drift_total``. Widen only on that side so the envelope matches
    # the trajectory ``plan_source_ces`` actually builds and validates.
    env_lo += min(0.0, drift_total)
    env_hi += max(0.0, drift_total)
    if env_lo < az_limits.min or env_hi > az_limits.max:
        raise AzimuthBoundsError(
            actual_min=float(env_lo),
            actual_max=float(env_hi),
            limit_min=az_limits.min,
            limit_max=az_limits.max,
        )

    return core.computed


def plan_source_ces(
    *,
    # --- Source ---
    body: str | None = None,
    ra: float | None = None,
    dec: float | None = None,
    pm_ra: float = 0.0,
    pm_dec: float = 0.0,
    ref_epoch: Time | None = None,
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
    timestep: float = 0.1,
    sampling_step_seconds: float = 30.0,
    az_accel: float = 1.0,
    az_padding: float = 0.5,
    az_branch: float | None = None,
    allow_partial: bool = False,
    v_az: float | None = None,
    sun_safe: SunSafePredicate | None = None,
) -> ScanBlock:
    """Plan a constant-elevation scan that drags a moving source across an array footprint.

    Source-tracking variant of :func:`plan_constant_el_scan`. Where
    ``plan_constant_el_scan`` aims at a fixed RA/Dec field rectangle
    and lets the source's natural sidereal motion fill the time axis,
    this planner aims at a single moving source (planet or sidereal
    point) and solves for an *additional* azimuth-drift rate ``v_az``
    so the source sweeps across the *entire* focal-plane footprint of
    an instrument array while the boresight stays at a fixed elevation
    ``el_bore``. It is the fyst-trajectories analogue of
    ``schedlib.source.make_source_ces`` in Simons Observatory's
    scheduler.

    Parameters
    ----------
    body : str, optional
        Solar-system body name (one of
        :data:`fyst_trajectories.SOLAR_SYSTEM_BODIES`). Mutually
        exclusive with ``ra``/``dec``.
    ra, dec : float, optional
        Sidereal source position in degrees. Mutually exclusive with
        ``body``.
    pm_ra, pm_dec : float, optional
        Proper motion in mas/yr (RA includes the cos(dec) factor, Gaia
        convention). Ignored when ``body`` is given. Default 0.0.
    ref_epoch : Time, optional
        Reference epoch for ``ra``/``dec``. Required when proper motion
        is non-zero; ignored otherwise.
    footprint : InstrumentOffset, str, sequence of InstrumentOffset, or ArrayFootprint
        Specification of the on-sky cover that the source must traverse.
        Accepted forms:

        * **InstrumentOffset** — a single offset (e.g. one PrimeCam
          module). Built as a 50-vertex circle around ``(dx, dy)``
          with radius :data:`~fyst_trajectories.primecam.MODULE_FOV_RADIUS_DEG`.
        * **str** — a named PrimeCam module ("c", "i1", …); resolved
          via :func:`~fyst_trajectories.get_primecam_offset`.
        * **sequence of InstrumentOffset** — one entry per module;
          footprint is the union of per-module circles; the aggregate
          center is the arithmetic mean of per-module ``(dx, dy)``.
        * **ArrayFootprint** — explicit (center, cover) representation;
          mirrors the ``array_info`` dict that SO ``make_source_ces``
          consumes.
    el_bore : float
        Fixed boresight elevation in degrees. Must lie within
        ``site.telescope_limits.elevation``.
    boresight_rot : float, optional
        Mechanical boresight rotation in degrees, added to the focal-
        plane rotation when projecting the cover. ``None`` (default)
        is treated as ``0.0`` for footprint geometry but signals to the
        downstream consumer that the boresight rotator is not
        commanded — matches SO ``make_source_ces``'s "do not rotate
        the cover" semantics. Pass an explicit ``0.0`` if you want a
        commanded zero rotation.
    window : (Time, Time), optional
        Explicit ``(t_start, t_end)`` search window. Mutually exclusive
        with ``night``/``mode``.
    night : Time, optional
        Start of the search window. Used with ``mode`` to pick the
        first rising or setting pass of the source within the next
        24 h.
    mode : {"rising", "setting"}, optional
        Which monotonic arc of the source to use. Required when
        ``night`` is given. With ``window``, omitting ``mode``
        auto-detects: the planner picks the longest monotonic arc
        inside the window whose elevation range covers ``el_bore`` and
        sets ``mode`` to ``"rising"`` or ``"setting"`` based on its
        slope. Pass an explicit ``mode`` to override.
    site : Site
        Telescope site.
    atmosphere : AtmosphericConditions, optional
        Refraction model passed to the underlying :class:`Coordinates`.
        Default ``None`` (vacuum) — matches the rest of the planning
        subpackage. The FYST ACU applies refraction at execution.
    timestep : float, optional
        Time between trajectory samples in seconds. Default 0.1.
    sampling_step_seconds : float, optional
        Coarse time step used when sampling the source's az(t)/el(t)
        curve for crossing detection and ``v_az`` optimisation.
        Default 30.0 (matches SO).
    az_accel : float, optional
        Azimuth acceleration for the executed scan, deg/s². Default
        1.0 (FYST conservative).
    az_padding : float, optional
        Extra azimuth padding on each side of the solved
        ``[az_start, az_start + az_throw]`` interval, in degrees.
        Default 0.5.
    az_branch : float, optional
        Centre of the azimuth wrap branch. If given, ``az_start`` is
        re-expressed in ``[az_branch - 180, az_branch + 180)``.
        Default ``None`` (no rewrap).
    allow_partial : bool, optional
        Behaviour when the source's elevation span does not cover the
        full footprint at ``el_bore`` (i.e. some cover vertices fall
        outside the source's elevation range). Default ``False``
        raises :class:`~fyst_trajectories.TargetNotObservableError`.
        Pass ``True`` to clip ``(t0, t1)`` to the overlap and emit a
        :class:`~fyst_trajectories.PointingWarning` instead — useful
        for sources observed near the limit of their accessible arc.
    v_az : float, optional
        Override the solved azimuth drift rate (deg/s) instead of
        running the Nelder-Mead optimisation. Useful for repeatable
        observations and cross-checks.
    sun_safe : SunSafePredicate, optional
        Sun-safety predicate implementing the
        :class:`~fyst_trajectories.dispatch.SunSafePredicate` contract,
        ``(az_deg, el_deg, time) -> bool`` returning ``True`` when the
        position is clear of the Sun. ``None`` (default) keeps the built-in
        scalar exclusion-radius check along the planned arc; an injected
        predicate is consulted per-sample instead, so the directional
        sun-avoidance model (future shared library) is honored end-to-end.
        Warn-only either way. See
        :class:`~fyst_trajectories.dispatch.SunSafePredicate`.

    Returns
    -------
    ScanBlock
        Planned observation. ``trajectory`` is a constant-elevation
        scan with the solved drift baked in. ``config`` is a
        :class:`~fyst_trajectories.ConstantElScanConfig`.
        ``computed_params`` is a :class:`SourceCESComputedParams`.

    Raises
    ------
    ValueError
        On incompatible argument combinations.
    TargetNotObservableError
        When the source never reaches ``el_bore`` in the search
        window, or when ``allow_partial=False`` and the source's
        elevation span doesn't cover the footprint at ``el_bore``.
    PointingError
        When the Nelder-Mead optimisation fails and no fallback
        ``v_az`` can be derived from the source's median az speed.

    Warns
    -----
    PointingWarning
        - Source passes within the site sun-avoidance exclusion radius
          at any sample along the planned arc.
        - Required azimuth speed exceeds the site's azimuth velocity
          limit (the scan may still execute if the per-sample speed
          comes in under the limit after padding).

    Notes
    -----
    The algorithm mirrors ``schedlib.source.make_source_ces`` (Simons
    Observatory) using astropy + numpy in place of ``so3g.proj``
    quaternions.

    The cover-polygon projection and the off-centre boresight recovery
    rotate the footprint by the mechanical focal-plane rotation,
    ``nasmyth_sign * el_bore + boresight_rot`` (a horizon-frame
    projection). SO ``make_source_ces`` projects with a static rotation
    only (the LAT corotator holds the array fixed in az/el); the two
    conventions are reconciled by
    ``boresight_rot_fyst = boresight_rot_SO - nasmyth_sign * el_bore``.

    If ``az_branch`` produces an az interval outside
    ``site.telescope_limits.azimuth``, the post-build
    :func:`~fyst_trajectories.trajectory_utils.validate_trajectory_bounds`
    raises :class:`~fyst_trajectories.AzimuthBoundsError`. For FYST
    (limits −180° to 360°), ``az_branch`` values near −180° can
    produce out-of-range scans even when geometrically valid.

    ``plan_source_ces`` is planner-only: it is consumed at dispatch time
    by the PCS ``source_scan`` task, not by the in-tree
    :mod:`fyst_trajectories.overhead` simulator. See
    ``docs/planning.rst`` "Source CES" for the wider conventions
    discussion.

    Examples
    --------
    Plan a Jupiter rising CES on PrimeCam's centre module:

    >>> from astropy.time import Time
    >>> from fyst_trajectories import get_fyst_site
    >>> from fyst_trajectories.planning import plan_source_ces
    >>> block = plan_source_ces(  # doctest: +SKIP
    ...     body="jupiter",
    ...     footprint="c",
    ...     el_bore=35.0,
    ...     night=Time("2026-03-15T00:00:00", scale="utc"),
    ...     mode="rising",
    ...     site=get_fyst_site(),
    ... )
    """
    core = _compute_source_ces_core(
        body=body,
        ra=ra,
        dec=dec,
        pm_ra=pm_ra,
        pm_dec=pm_dec,
        ref_epoch=ref_epoch,
        footprint=footprint,
        el_bore=el_bore,
        boresight_rot=boresight_rot,
        window=window,
        night=night,
        mode=mode,
        site=site,
        atmosphere=atmosphere,
        sampling_step_seconds=sampling_step_seconds,
        az_accel=az_accel,
        az_padding=az_padding,
        az_branch=az_branch,
        allow_partial=allow_partial,
        v_az=v_az,
        sun_safe=sun_safe,
    )

    computed = core.computed
    az_start = computed["az_start"]
    az_throw = computed["az_throw"]
    v_az_solved = computed["v_az"]
    boresight_rot_deg = computed["boresight_rot"]
    actual_duration = core.actual_duration
    velocity = core.velocity
    n_scans = core.n_scans
    az_stop = core.az_stop
    t0 = core.t0
    source_label = core.source_label
    fp = core.fp
    mode_resolved = core.mode

    config = ConstantElScanConfig(
        timestep=timestep,
        az_start=az_start,
        az_stop=az_stop,
        elevation=el_bore,
        az_speed=velocity,
        az_accel=az_accel,
    )

    base_traj = _build_altaz_trajectory(
        site=site,
        config=config,
        duration=actual_duration,
        start_time=t0,
        atmosphere=atmosphere,
        detector_offset=None,
    )

    # Add the linear az drift on top of the per-leg azimuth track.
    drifted_az = base_traj.az + v_az_solved * base_traj.times
    drifted_az_vel = base_traj.az_vel + v_az_solved
    # Replace the underlying ConstantEl metadata with source-CES metadata
    # so downstream consumers can see the source RA/Dec.
    # ``src_ra_at_el_bore``/``src_dec_at_el_bore`` are the source
    # coordinates at ``t_at_el_bore``.
    source_metadata = TrajectoryMetadata(
        pattern_type="source_ces",
        pattern_params={
            "el_bore": float(el_bore),
            "boresight_rot": float(boresight_rot_deg),
            "v_az": float(v_az_solved),
            "az_start": float(az_start),
            "az_stop": float(az_stop),
            "mode": mode_resolved,
            "n_scans": int(n_scans),
        },
        center_ra=float(core.src_ra_at_el_bore),
        center_dec=float(core.src_dec_at_el_bore),
        target_name=source_label,
    )
    trajectory = dataclasses.replace(
        base_traj,
        az=drifted_az,
        az_vel=drifted_az_vel,
        metadata=source_metadata,
    )
    # Post-drift bounds check. validate_trajectory_bounds raises
    # AzimuthBoundsError / ElevationBoundsError on violation.
    validate_trajectory_bounds(site, trajectory.az, trajectory.el)

    summary = (
        f"Source-CES on {source_label} ({mode_resolved}) at el_bore={el_bore:.2f} deg\n"
        f"  Footprint: {fp.cover_xi_deg.size} cover vertices, "
        f"center=({fp.center_xi_deg:.3f}, {fp.center_eta_deg:.3f}) deg "
        f"(xi, eta)\n"
        f"  Az range: [{az_start:.2f}, {az_stop:.2f}] deg "
        f"(throw {az_throw:.2f} deg)\n"
        f"  Drift v_az={v_az_solved:+.5f} deg/s, "
        f"per-leg az_speed={velocity:.3f} deg/s, "
        f"az_accel={az_accel:.2f} deg/s^2\n"
        f"  Source pass: {computed['t0_iso'][:19]} to {computed['t1_iso'][:19]}\n"
        f"  Scans: {n_scans}, Duration: {actual_duration:.1f}s "
        f"({actual_duration / 60:.1f}min), "
        f"Trajectory points: {trajectory.n_points}"
    )

    return ScanBlock(
        trajectory=trajectory,
        config=config,
        duration=actual_duration,
        computed_params=computed,
        summary=summary,
    )
