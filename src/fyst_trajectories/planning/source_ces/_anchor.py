"""Approximate-start-time ("anchor") resolution for the source-CES planner.

These helpers derive ``el_bore``, ``mode``, and a forward search ``window``
from an approximate anchor time; the classic ``night``/``window`` forms never
touch them.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

from ..._validation import AZ_PADDING_MAX_DEG, _require_non_negative, _require_positive
from ...coordinates import Coordinates
from ...exceptions import (
    ElevationBoundsError,
    PointingError,
    PointingWarning,
    TargetNotObservableError,
)
from ...offsets import InstrumentOffset
from ...site import AtmosphericConditions, Site
from .._helpers import _coerce_start_time
from .._types import ArrayFootprint
from ..footprints import resolve_footprint
from ._kernel import _DEFAULT_SEARCH_HORIZON_HOURS, _compute_source_ces_core, _project_footprint
from ._source import _SourceSpec

# --- Approximate-start-time ("anchor") resolution constants ---------------
# These support the ``start_time`` keyword on the source-CES entry points,
# which lets a caller ask for "a pass starting about now" without solving
# el_bore themselves. They only affect the anchored code path; the classic
# night+mode and explicit-window forms never touch them.

# Time separation used to probe the source's elevation slope at the anchor.
# The slope sign picks the rising/setting mode when it is omitted, and its
# magnitude is checked against the near-transit drift guard below. 60 s is
# long enough for the finite difference to clear ephemeris rounding yet
# short enough that the local slope is still representative.
_ANCHOR_SLOPE_DT_SEC = 60.0

# Elevation buffer added on top of the footprint's angular radius from the
# boresight when choosing the probe el_bore. The projected elevation extent
# of any cover vertex is bounded by its angular distance from the boresight,
# so a probe el_bore this far beyond the anchor elevation puts the whole
# projected cover beyond the anchor elevation (no clip at the window's
# opening edge) while keeping the probe close enough that the geometry it
# measures nearly matches the final solve (the focal-plane rotation turns an
# off-centre cover between the two, which _derive_anchored_el_bore checks for).
# Near culmination the cover's far side can still lie beyond the source's arc,
# which the probe's allow_partial absorbs; a source that turns before it
# reaches the probe at all is left to _derive_anchored_el_bore's fallback.
_ANCHOR_PROBE_BUFFER_DEG = 0.1

# Deliberate elevation lead applied to the derived el_bore so the resolved
# pass start lands just after the anchor rather than exactly on it. The
# search window opens at the anchor, so the pass can never begin earlier;
# without this lead the footprint's near-anchor cover edge sits on the
# window boundary, where the elevation-crossing slice is prone to clip. At
# typical source drift rates this places the start within about a minute of
# the anchor. A cover that leads the boresight along the source's motion
# starts later; see _derive_anchored_el_bore.
_ANCHOR_START_LEAD_DEG = 0.05

# Fixed-point steps of the re-solve that moves the derived el_bore when the
# cover's near edge, projected there, lies behind the source at the anchor
# (see _derive_anchored_el_bore). The edge moves with el_bore at unit rate,
# give or take what the focal-plane rotation does to an off-centre cover (a
# few hundredths of a degree per degree for a ring module), so each step
# multiplies the residual by that excess and three leave the edge well
# within 1e-3 deg of its target, a small part of the lead: from the
# hundredths of a degree the rotation moves the edge between the probe and
# the derived el_bore, and from the 2.4 deg a ring module's cover can reach
# behind the boresight when the probe is out of reach (a few 1e-5 deg left).
# That holds for a source that culminates well below the zenith. Near the
# zenith the rotation turns the cover faster as el_bore moves, the excess
# grows and three steps leave more; in every case measured the edge still
# fell short of its target rather than past it, so it stays ahead of the
# source by most of the lead.
_ANCHOR_EDGE_STEPS = 3

# Minimum source elevation drift rate (deg/s) for deriving el_bore from an
# anchor. The binding requirement is timing: the derivation leads the anchor
# by _ANCHOR_START_LEAD_DEG of elevation, which the source crosses in
# lead / drift seconds, so honouring the documented "resolved start lands
# within about a minute of the anchor" contract needs
# drift >= _ANCHOR_START_LEAD_DEG / 60 s ~ 8.3e-4 deg/s. This supersedes the
# looser conditioning floor: near transit d(el)/dt -> 0 and the t(el)
# inversion that places the pass start becomes ill-conditioned; a single
# Prime-Cam module cover spans ~2 * MODULE_FOV_RADIUS_DEG = 1.3 deg in eta, so
# keeping a module-c pass under ~1 h (3600 s) needs
# drift >= 1.3 / 3600 ~ 3.6e-4 deg/s. Below the threshold the derivation
# refuses and asks the caller to anchor away from transit (or pass el_bore
# explicitly).
_MIN_ANCHOR_EL_DRIFT_DEG_S = _ANCHOR_START_LEAD_DEG / 60.0


def _source_el_at(
    coords: Coordinates,
    obstime: Time,
    *,
    source: _SourceSpec,
) -> float:
    """Sample the source elevation (degrees) at a single instant.

    Wraps :meth:`_SourceSpec.sample_altaz` with a one-element time array so
    the body / ra-dec / proper-motion dispatch stays identical to the search,
    then returns the scalar elevation.
    """
    arr = obstime + TimeDelta(np.array([0.0]) * u.s)
    _, el = source.sample_altaz(coords, arr)
    return float(np.asarray(el, dtype=float)[0])


def _probe_anchor_slope(
    coords: Coordinates,
    anchor: Time,
    *,
    source: _SourceSpec,
) -> tuple[float, float]:
    """Return ``(el_at_anchor_deg, el_slope_deg_per_s)`` for the source at ``anchor``.

    Samples the source elevation at ``anchor`` and
    ``anchor + _ANCHOR_SLOPE_DT_SEC`` through the same dispatch the planner
    uses, so the slope sign (which picks the rising/setting mode) and its
    magnitude (checked against the near-transit guard) stay consistent with
    the elevation-crossing search.
    """
    probe_times = anchor + TimeDelta(np.array([0.0, _ANCHOR_SLOPE_DT_SEC]) * u.s)
    _, el = source.sample_altaz(coords, probe_times)
    el = np.asarray(el, dtype=float)
    el_at_anchor = float(el[0])
    slope = (float(el[1]) - el_at_anchor) / _ANCHOR_SLOPE_DT_SEC
    return el_at_anchor, slope


def _anchored_el_limits_error(
    *,
    anchor: Time,
    el_at_anchor: float,
    source: _SourceSpec,
    site: Site,
) -> TargetNotObservableError:
    """Build the anchored-derivation elevation-limits error.

    Used when deriving ``el_bore`` from an anchor would place the boresight
    outside the telescope elevation limits (e.g. the source sits below the
    elevation floor at the anchor). The message is anchor-relative, naming
    the source's elevation at the anchor and the telescope limits, rather
    than an internally derived boresight elevation the caller never supplied.
    """
    label = source.label
    el_limits = site.telescope_limits.elevation
    return TargetNotObservableError(
        target=label,
        time_info=str(anchor.iso),
        bounds_error=ElevationBoundsError(
            actual_min=el_at_anchor,
            actual_max=el_at_anchor,
            limit_min=el_limits.min,
            limit_max=el_limits.max,
        ),
        message=(
            f"{label} is at elevation {el_at_anchor:.2f} deg at the anchor "
            f"{anchor.iso}: the derived boresight elevation would fall outside "
            f"the telescope elevation limits (floor {el_limits.min} deg, "
            f"ceiling {el_limits.max} deg). Anchor while the source pass fits "
            f"inside the elevation range, or pass el_bore explicitly."
        ),
    )


def _near_cover_edge(
    fp: ArrayFootprint,
    el_bore: float,
    boresight_rot: float | None,
    site: Site,
    drift_sign: float,
) -> float:
    """Return the elevation of the cover edge the source meets first, projected at ``el_bore``.

    The lowest projected vertex for a rising source (``drift_sign`` +1), the
    highest for a setting one, taken from the kernel's own footprint stage.
    A vertex's elevation does not depend on the boresight azimuth, so the
    projection is made at azimuth 0.
    """
    _, _, el_cover = _project_footprint(
        fp=fp,
        site=site,
        el_bore=el_bore,
        boresight_rot_deg=0.0 if boresight_rot is None else float(boresight_rot),
        src_az_at_el_bore=0.0,
    )
    return float(el_cover.min()) if drift_sign > 0.0 else float(el_cover.max())


def _derive_anchored_el_bore(
    *,
    anchor: Time,
    el_at_anchor: float,
    slope: float,
    mode: Literal["rising", "setting"],
    coords: Coordinates,
    fp: ArrayFootprint,
    source: _SourceSpec,
    boresight_rot: float | None,
    site: Site,
    atmosphere: AtmosphericConditions | None,
    sampling_step_seconds: float,
    az_accel: float,
    az_padding: float | None,
    az_branch: float | None,
) -> float:
    """Derive the ``el_bore`` whose pass starts approximately at ``anchor``.

    A source-tracking CES starts when the source reaches the cover's near
    edge, its lowest projected vertex for a rising source and its highest
    for a setting one, so ``el_bore = el(anchor)`` would centre the pass on
    the anchor and begin it earlier. This runs one probe of the params-only
    kernel at a lifted ``el_bore`` (past the anchor elevation by the cover's
    full reach from the boresight, so that the source meets the cover only
    after the anchor at any focal-plane rotation), reads the source
    elevation at the probe's resolved start, and shifts ``el_bore`` so that
    crossing lands on the anchor. A small :data:`_ANCHOR_START_LEAD_DEG`
    lead is added in the drift direction so the resolved start settles just
    after the anchor, clear of the search-window boundary. One probe, no
    iteration of the kernel. The anchor places the full footprint crossing;
    a ``dwell`` narrows the pass about that crossing's midpoint afterwards,
    so a dwell-limited pass starts half the cut after the anchor (placing
    the narrowed start on the anchor would open the crossing before the
    search window and trip the cover-vs-arc guard). The caller's
    ``allow_partial`` is not read here: it is the real solve's tolerance for
    a source that turns before it crosses the whole cover, and the checks
    below put the near edge ahead of the source whether or not it is set,
    so a derived pass is never clipped at the anchor.

    When the source moves in the ``mode`` direction at the anchor, the arc
    the search needs starts there, and the real solve refuses unless the
    boresight elevation and the near edge both lie ahead of the source at
    the anchor. (An explicit ``mode`` against the source's motion is met on
    an arc that starts at the source's next turn, which the anchor does not
    bound, and the probe's value stands.) The direction is read from
    ``slope``, the elevation rate :func:`_resolve_anchor_prefix` measured at
    the anchor, the same probe that resolves an omitted ``mode``, so a
    derived mode always moves along. So two checks follow the probe:

    * A cover that leads the boresight along the source's motion (above it
      while the source rises, below it while it sets) by more than the lead
      puts the derived ``el_bore`` behind the source, which then meets it
      only on the next day's arc. ``el_bore`` is then anchored on the source
      instead, the lead ahead of its elevation at the anchor: the source
      crosses it shortly after the anchor, and the pass starts when the
      source reaches the cover, the cover's elevation lead divided by the
      source's elevation rate later (minutes for a ring module).
    * The probe measures the near edge at its own ``el_bore``, but the
      focal-plane rotation turns an off-centre cover as ``el_bore``
      changes, so at the derived value the edge can lie behind the source
      at the anchor, where the pass would open before the window does.
      ``el_bore`` is then re-solved, by fixed-point steps on the kernel's
      own cover projection, to put the edge the lead ahead of the source.

    Where neither applies, as always for the centre module (a circle about
    the boresight projects nearly the same at every ``el_bore``), the probe's
    value is returned unchanged.

    The full reach suits every rotation but is more lift than most covers
    need, and near its culmination (its lower culmination, setting) a source
    moving in the ``mode`` direction can turn before it reaches the probe's
    elevation or the lifted cover, which the probe refuses. The source's
    elevation at the anchor plus the lead then stands in for the probe's
    value, and the second check's fixed-point steps lift it by as much as
    the cover, projected there, reaches behind the source: the lift this
    cover needs at this rotation, up to the full reach for a cover that
    trails the boresight and only the lead for one that leads it. The real
    solve then plans the pass where the source, before it turns, reaches
    both the derived ``el_bore`` and the whole cover. Where it turns inside
    the cover the pass is refused, or clipped under ``allow_partial``.
    Where it turns short of ``el_bore`` (a cover trailing the boresight can
    fit below the turn while its boresight does not) the pass is refused
    either way.

    The probe sweeps with the kernel's default padding, never the caller's
    ``az_padding``, which it does not need. So that padding is checked here,
    with the two arguments the kernel checks ahead of it and in the same
    order, before the probe runs: a malformed one is the kernel's
    ``ValueError`` wherever the probe would refuse the anchor.

    Raises :class:`TargetNotObservableError` (anchor-relative message) when
    the probe or the derived ``el_bore`` falls outside the telescope
    elevation limits, e.g. the source is below the elevation floor at the
    anchor; the probe's :class:`ElevationBoundsError` is preserved as the
    ``__cause__``. The probe's ``OffsetInversionError``, raised when the
    off-centre boresight inverse fails at its elevation, propagates
    unchanged, and so do its other refusals for a source that does not move
    in the ``mode`` direction at the anchor: a
    :class:`TargetNotObservableError` when the source never reaches the
    probe's elevation, and a ``PointingError`` when it reaches none of the
    lifted cover's vertices. The checks' own projection of the cover raises
    an ``OffsetInversionError`` too where the off-centre inverse fails at an
    ``el_bore`` they try, as it can for a source that culminates within
    about a degree of the zenith.
    """
    sampling_step_seconds = _require_positive(sampling_step_seconds, "sampling_step_seconds")
    az_accel = _require_positive(az_accel, "az_accel")
    if az_padding is not None:
        _require_non_negative(az_padding, "az_padding", maximum=AZ_PADDING_MAX_DEG)
    drift_sign = 1.0 if mode == "rising" else -1.0
    target = el_at_anchor + drift_sign * _ANCHOR_START_LEAD_DEG
    # Whether the source moves in the mode direction at the anchor, which the
    # two checks below and the fallback for a probe out of reach require.
    along = (slope >= 0.0) == (drift_sign > 0.0)
    # The projected elevation deviation of any cover vertex is bounded by its
    # angular distance from the boresight (focal-plane origin), so lifting the
    # probe el_bore by more than that distance puts the whole cover beyond the
    # anchor elevation regardless of the focal-plane rotation, and the crossing
    # begins after the window opens. Near culmination the source can turn
    # back before it crosses all of the cover, which allow_partial tolerates,
    # or before it reaches the probe at all, which the fallback below handles.
    r_bore = float(np.hypot(fp.cover_xi_deg, fp.cover_eta_deg).max())
    margin = r_bore + _ANCHOR_PROBE_BUFFER_DEG
    el_bore_probe = el_at_anchor + drift_sign * margin

    horizon = TimeDelta(_DEFAULT_SEARCH_HORIZON_HOURS * 3600.0 * u.s)
    probe_window = (anchor, anchor + horizon)
    try:
        with warnings.catch_warnings():
            # The probe is an internal measurement; v_az=0 skips the optimiser
            # and screen=False the Sun sweep and the peak-speed advisory (t0
            # depends on none of them), and allow_partial keeps it robust. It
            # takes the kernel's default padding, not the caller's, since t0
            # does not depend on that either: near culmination the source can
            # meet the partial cover at a single vertex, and with no padding
            # the kernel refuses that sweep for having no width. The
            # authoritative warnings come from the real solve, so silence the
            # probe's own, which allow_partial can still raise. The filter is
            # process-global while installed, so a PointingWarning another
            # thread raises meanwhile is lost too; it is narrowed to that
            # category so no other warning is.
            warnings.simplefilter("ignore", PointingWarning)
            probe = _compute_source_ces_core(
                source=source,
                footprint=fp,
                el_bore=el_bore_probe,
                boresight_rot=boresight_rot,
                window=probe_window,
                night=None,
                mode=mode,
                site=site,
                atmosphere=atmosphere,
                sampling_step_seconds=sampling_step_seconds,
                az_accel=az_accel,
                az_branch=az_branch,
                allow_partial=True,
                v_az=0.0,
                screen=False,
            )
    except ElevationBoundsError as err:
        # The kernel only raises a bare ElevationBoundsError for an el_bore
        # outside the telescope limits, and here that el_bore is the internal
        # probe value; report the failure in the caller's terms instead.
        raise _anchored_el_limits_error(
            anchor=anchor, el_at_anchor=el_at_anchor, source=source, site=site
        ) from err
    except PointingError as err:
        # Only a TargetNotObservableError (the source never reaches the probe's
        # el_bore) or the kernel's plain PointingError (it reaches no vertex of
        # the lifted cover) says the source turns within the cover's full reach;
        # any other refusal, an OffsetInversionError among them, propagates.
        # For a source moving in the mode direction, the source's elevation
        # plus the lead stands in for the probe's value, and the checks below
        # lift it by as much as the cover, projected there, reaches behind the
        # source.
        out_of_reach = isinstance(err, TargetNotObservableError) or type(err) is PointingError
        if not (out_of_reach and along):
            raise
        derived = target
    else:
        el_at_probe_t0 = _source_el_at(coords, probe.t0, source=source)
        derived = (
            el_bore_probe + (el_at_anchor - el_at_probe_t0) + drift_sign * _ANCHOR_START_LEAD_DEG
        )
    # The two checks of the docstring, for a source moving in the mode
    # direction. Both are strict, so a value that puts the boresight elevation
    # and the near edge at or ahead of the source is returned bit for bit.
    edge = _near_cover_edge(fp, derived, boresight_rot, site, drift_sign)
    bore_behind = drift_sign * (derived - el_at_anchor) < 0.0
    edge_behind = drift_sign * (edge - el_at_anchor) < 0.0
    if (bore_behind or edge_behind) and along:
        if bore_behind:
            # The cover leads the boresight: anchor on the source.
            derived = target
            edge = _near_cover_edge(fp, derived, boresight_rot, site, drift_sign)
        if drift_sign * (edge - el_at_anchor) < 0.0:
            # The near edge lies behind the source: the focal-plane rotation
            # turned it there between the probe's el_bore and this one, or,
            # with the probe out of reach, the cover reaches behind the
            # boresight.
            for _ in range(_ANCHOR_EDGE_STEPS):
                derived += target - edge
                edge = _near_cover_edge(fp, derived, boresight_rot, site, drift_sign)
    # The probe sits farther from the anchor elevation than the derived value
    # in most geometries, but not in all of them; pre-check the derived
    # el_bore too so an out-of-limits result surfaces with the same
    # anchor-relative message rather than the kernel's raw bounds error.
    el_limits = site.telescope_limits.elevation
    if not (el_limits.min <= derived <= el_limits.max):
        raise _anchored_el_limits_error(
            anchor=anchor, el_at_anchor=el_at_anchor, source=source, site=site
        )
    return derived


def _anchor_drift_guard(
    slope: float,
    *,
    anchor: Time,
    el_at_anchor: float,
    source: _SourceSpec,
    el_limits_min: float,
    el_limits_max: float,
) -> None:
    """Reject anchors too near transit to derive ``el_bore`` from.

    Raises :class:`TargetNotObservableError` when the source elevation drift
    rate at the anchor is below :data:`_MIN_ANCHOR_EL_DRIFT_DEG_S`.
    """
    if abs(slope) >= _MIN_ANCHOR_EL_DRIFT_DEG_S:
        return
    label = source.label
    raise TargetNotObservableError(
        target=label,
        time_info=str(anchor.iso),
        bounds_error=ElevationBoundsError(
            actual_min=el_at_anchor,
            actual_max=el_at_anchor,
            limit_min=el_limits_min,
            limit_max=el_limits_max,
        ),
        message=(
            f"{label} is too near transit at {anchor.iso} to derive el_bore from "
            f"start_time: the source elevation drift rate is {abs(slope):.2e} deg/s, "
            f"below the {_MIN_ANCHOR_EL_DRIFT_DEG_S:.1e} deg/s minimum. Anchor away "
            f"from transit, or pass el_bore explicitly."
        ),
    )


def _resolve_anchor_prefix(
    *,
    start_time: Time | str,
    el_bore: float | None,
    mode: Literal["rising", "setting"] | None,
    night: Time | None,
    window: tuple[Time, Time] | None,
    source: _SourceSpec,
    site: Site,
    atmosphere: AtmosphericConditions | None,
) -> tuple[Time, Coordinates, Literal["rising", "setting"], float, float]:
    """Shared validation and slope probe for the anchored entry points.

    Common front half of every ``start_time`` resolution: validates that
    ``start_time`` is not combined with ``night`` or ``window``, parses the
    anchor, and, when ``el_bore`` or ``mode`` is omitted, probes the source
    elevation slope at the anchor. The slope sign resolves an omitted
    ``mode``; the near-transit drift guard runs when ``el_bore`` must be
    derived. Returns ``(anchor, coords, mode, el_at_anchor, slope)``;
    ``el_at_anchor`` and ``slope`` (deg/s) are meaningful whenever
    ``el_bore`` is ``None`` (the probe is guaranteed to have run in that
    case).
    """
    if night is not None:
        raise ValueError("specify either 'start_time' or 'night', not both")
    if window is not None:
        raise ValueError("specify either 'start_time' or 'window', not both")

    anchor = _coerce_start_time(start_time)
    coords = Coordinates(site, atmosphere=atmosphere)

    resolved_mode = mode
    el_at_anchor = 0.0
    slope = 0.0
    if el_bore is None or mode is None:
        el_at_anchor, slope = _probe_anchor_slope(coords, anchor, source=source)
        if resolved_mode is None:
            resolved_mode = "rising" if slope >= 0.0 else "setting"
        if el_bore is None:
            el_limits = site.telescope_limits.elevation
            _anchor_drift_guard(
                slope,
                anchor=anchor,
                el_at_anchor=el_at_anchor,
                source=source,
                el_limits_min=el_limits.min,
                el_limits_max=el_limits.max,
            )

    assert resolved_mode is not None  # mode was given, or set from the probe slope above
    return anchor, coords, resolved_mode, el_at_anchor, slope


def _resolve_start_time_anchor(
    *,
    start_time: Time | str,
    el_bore: float | None,
    mode: Literal["rising", "setting"] | None,
    night: Time | None,
    window: tuple[Time, Time] | None,
    footprint: InstrumentOffset | str | Sequence[InstrumentOffset] | ArrayFootprint,
    source: _SourceSpec,
    boresight_rot: float | None,
    site: Site,
    atmosphere: AtmosphericConditions | None,
    sampling_step_seconds: float,
    az_accel: float,
    az_padding: float | None,
    az_branch: float | None,
) -> tuple[float, tuple[Time, Time], Literal["rising", "setting"]]:
    """Resolve an approximate ``start_time`` anchor into ``(el_bore, window, mode)``.

    :func:`_resolve_anchor_prefix` handles the shared validation, mode
    resolution, and near-transit guard; this then derives ``el_bore`` so the
    pass starts approximately at the anchor (when it was omitted) and returns
    a forward search ``window`` opening at the anchor, which the caller
    hands to the kernel as an explicit window.
    """
    anchor, coords, resolved_mode, el_at_anchor, slope = _resolve_anchor_prefix(
        start_time=start_time,
        el_bore=el_bore,
        mode=mode,
        night=night,
        window=window,
        source=source,
        site=site,
        atmosphere=atmosphere,
    )

    horizon = TimeDelta(_DEFAULT_SEARCH_HORIZON_HOURS * 3600.0 * u.s)
    resolved_window = (anchor, anchor + horizon)

    if el_bore is None:
        el_bore = _derive_anchored_el_bore(
            anchor=anchor,
            el_at_anchor=el_at_anchor,
            slope=slope,
            mode=resolved_mode,
            coords=coords,
            fp=resolve_footprint(footprint),
            source=source,
            boresight_rot=boresight_rot,
            site=site,
            atmosphere=atmosphere,
            sampling_step_seconds=sampling_step_seconds,
            az_accel=az_accel,
            az_padding=az_padding,
            az_branch=az_branch,
        )

    return float(el_bore), resolved_window, resolved_mode
