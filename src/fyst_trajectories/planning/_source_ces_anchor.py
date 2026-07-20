"""Approximate-start-time ("anchor") resolution for the source-CES planner.

These helpers derive ``el_bore``, ``mode``, and a forward search ``window``
from an approximate anchor time; the classic ``night``/``window`` forms never
touch them.

The module forms a deliberate two-way reference with :mod:`source_ces`: the
anchor helpers call back into the core (``_compute_source_ces_core``,
``_sample_source_altaz``, ``_describe_source``, ``_resolve_footprint``, and the
``_DEFAULT_SEARCH_HORIZON_HOURS`` default) while ``source_ces`` imports three
entry points from here. The cycle is broken by importing the ``source_ces``
module object (``from . import source_ces``) rather than its names: the module
binds mid-import and each attribute is resolved at call time, by which point
``source_ces`` is fully initialised.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

from ..coordinates import Coordinates
from ..exceptions import ElevationBoundsError, TargetNotObservableError
from ..offsets import InstrumentOffset
from ..site import AtmosphericConditions, Site
from . import source_ces
from ._helpers import _coerce_start_time
from ._types import ArrayFootprint

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
# so a probe el_bore this far beyond the anchor elevation keeps the whole
# projected cover inside the reachable arc (no window-edge clip) while
# keeping the probe close enough that the geometry it measures matches the
# final solve.
_ANCHOR_PROBE_BUFFER_DEG = 0.1

# Deliberate elevation lead applied to the derived el_bore so the resolved
# pass start lands just after the anchor rather than exactly on it. The
# search window opens at the anchor, so the pass can never begin earlier;
# without this lead the footprint's near-anchor cover edge sits on the
# window boundary, where the elevation-crossing slice is prone to clip. At
# typical source drift rates this places the start within about a minute of
# the anchor.
_ANCHOR_START_LEAD_DEG = 0.05

# Minimum source elevation drift rate (deg/s) for deriving el_bore from an
# anchor. The binding requirement is timing: the derivation leads the anchor
# by _ANCHOR_START_LEAD_DEG of elevation, which the source crosses in
# lead / drift seconds, so honouring the documented "resolved start lands
# within about a minute of the anchor" contract needs
# drift >= _ANCHOR_START_LEAD_DEG / 60 s ~ 8.3e-4 deg/s. This supersedes the
# looser conditioning floor: near transit d(el)/dt -> 0 and the t(el)
# inversion that places the pass start becomes ill-conditioned; a single
# PrimeCam module cover spans ~2 * MODULE_FOV_RADIUS_DEG = 1.3 deg in eta, so
# keeping a module-c pass under ~1 h (3600 s) needs
# drift >= 1.3 / 3600 ~ 3.6e-4 deg/s. Below the threshold the derivation
# refuses and asks the caller to anchor away from transit (or pass el_bore
# explicitly).
_MIN_ANCHOR_EL_DRIFT_DEG_S = _ANCHOR_START_LEAD_DEG / 60.0


def _source_el_at(
    coords: Coordinates,
    obstime: Time,
    *,
    body: str | None,
    ra: float | None,
    dec: float | None,
    pm_ra: float,
    pm_dec: float,
    ref_epoch: Time | None,
) -> float:
    """Sample the source elevation (degrees) at a single instant.

    Wraps :func:`_sample_source_altaz` with a one-element time array so the
    body / ra-dec / proper-motion dispatch stays identical to the search,
    then returns the scalar elevation.
    """
    arr = obstime + TimeDelta(np.array([0.0]) * u.s)
    _, el = source_ces._sample_source_altaz(
        coords, arr, body=body, ra=ra, dec=dec, pm_ra=pm_ra, pm_dec=pm_dec, ref_epoch=ref_epoch
    )
    return float(np.asarray(el, dtype=float)[0])


def _probe_anchor_slope(
    coords: Coordinates,
    anchor: Time,
    *,
    body: str | None,
    ra: float | None,
    dec: float | None,
    pm_ra: float,
    pm_dec: float,
    ref_epoch: Time | None,
) -> tuple[float, float]:
    """Return ``(el_at_anchor_deg, el_slope_deg_per_s)`` for the source at ``anchor``.

    Samples the source elevation at ``anchor`` and
    ``anchor + _ANCHOR_SLOPE_DT_SEC`` through the same dispatch the planner
    uses, so the slope sign (which picks the rising/setting mode) and its
    magnitude (checked against the near-transit guard) stay consistent with
    the elevation-crossing search.
    """
    probe_times = anchor + TimeDelta(np.array([0.0, _ANCHOR_SLOPE_DT_SEC]) * u.s)
    _, el = source_ces._sample_source_altaz(
        coords,
        probe_times,
        body=body,
        ra=ra,
        dec=dec,
        pm_ra=pm_ra,
        pm_dec=pm_dec,
        ref_epoch=ref_epoch,
    )
    el = np.asarray(el, dtype=float)
    el_at_anchor = float(el[0])
    slope = (float(el[1]) - el_at_anchor) / _ANCHOR_SLOPE_DT_SEC
    return el_at_anchor, slope


def _anchored_el_limits_error(
    *,
    anchor: Time,
    el_at_anchor: float,
    body: str | None,
    ra: float | None,
    dec: float | None,
    site: Site,
) -> TargetNotObservableError:
    """Build the anchored-derivation elevation-limits error.

    Used when deriving ``el_bore`` from an anchor would place the boresight
    outside the telescope elevation limits (e.g. the source sits below the
    elevation floor at the anchor). The message is anchor-relative, naming
    the source's elevation at the anchor and the telescope limits, rather
    than an internally derived boresight elevation the caller never supplied.
    """
    label = source_ces._describe_source(body, ra, dec)
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


def _derive_anchored_el_bore(
    *,
    anchor: Time,
    el_at_anchor: float,
    mode: Literal["rising", "setting"],
    coords: Coordinates,
    fp: ArrayFootprint,
    body: str | None,
    ra: float | None,
    dec: float | None,
    pm_ra: float,
    pm_dec: float,
    ref_epoch: Time | None,
    boresight_rot: float | None,
    site: Site,
    atmosphere: AtmosphericConditions | None,
    sampling_step_seconds: float,
    az_accel: float,
    az_padding: float,
    az_branch: float | None,
) -> float:
    """Derive the ``el_bore`` whose pass starts approximately at ``anchor``.

    A source-tracking CES starts (rising) or ends (setting) when the source
    crosses the footprint's lower/upper projected elevation edge, so
    ``el_bore = el(anchor)`` would centre the pass on the anchor and begin it
    earlier. This runs one probe of the params-only kernel at a lifted
    ``el_bore`` (far enough from the anchor that the whole cover is reachable),
    reads the source elevation at the probe's resolved start, and shifts
    ``el_bore`` so that crossing lands on the anchor. A small
    :data:`_ANCHOR_START_LEAD_DEG` lead is added in the drift direction so the
    resolved start settles just after the anchor, clear of the search-window
    boundary. One probe, no iteration.

    Raises :class:`TargetNotObservableError` (anchor-relative message) when
    the probe or the derived ``el_bore`` falls outside the telescope
    elevation limits, e.g. the source is below the elevation floor at the
    anchor; the probe's :class:`ElevationBoundsError` is preserved as the
    ``__cause__``.
    """
    drift_sign = 1.0 if mode == "rising" else -1.0
    # The projected elevation deviation of any cover vertex is bounded by its
    # angular distance from the boresight (focal-plane origin), so lifting the
    # probe el_bore by more than that distance keeps the whole cover inside
    # the reachable arc regardless of the field rotation.
    r_bore = float(np.hypot(fp.cover_xi_deg, fp.cover_eta_deg).max())
    margin = r_bore + _ANCHOR_PROBE_BUFFER_DEG
    el_bore_probe = el_at_anchor + drift_sign * margin

    horizon = TimeDelta(source_ces._DEFAULT_SEARCH_HORIZON_HOURS * 3600.0 * u.s)
    probe_window = (anchor, anchor + horizon)
    try:
        with warnings.catch_warnings():
            # The probe is an internal measurement; v_az=0 skips the optimiser
            # (t0 does not depend on it) and allow_partial keeps it robust. The
            # authoritative warnings come from the real solve, so silence these.
            warnings.simplefilter("ignore")
            probe = source_ces._compute_source_ces_core(
                body=body,
                ra=ra,
                dec=dec,
                pm_ra=pm_ra,
                pm_dec=pm_dec,
                ref_epoch=ref_epoch,
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
                az_padding=az_padding,
                az_branch=az_branch,
                allow_partial=True,
                v_az=0.0,
                sun_safe=None,
            )
    except ElevationBoundsError as err:
        # The kernel only raises a bare ElevationBoundsError for an el_bore
        # outside the telescope limits, and here that el_bore is the internal
        # probe value; report the failure in the caller's terms instead.
        raise _anchored_el_limits_error(
            anchor=anchor, el_at_anchor=el_at_anchor, body=body, ra=ra, dec=dec, site=site
        ) from err
    el_at_probe_t0 = _source_el_at(
        coords, probe.t0, body=body, ra=ra, dec=dec, pm_ra=pm_ra, pm_dec=pm_dec, ref_epoch=ref_epoch
    )
    derived = el_bore_probe + (el_at_anchor - el_at_probe_t0) + drift_sign * _ANCHOR_START_LEAD_DEG
    # The probe sits farther from the anchor elevation than the derived value
    # in most geometries, but not in all of them; pre-check the derived
    # el_bore too so an out-of-limits result surfaces with the same
    # anchor-relative message rather than the kernel's raw bounds error.
    el_limits = site.telescope_limits.elevation
    if not (el_limits.min <= derived <= el_limits.max):
        raise _anchored_el_limits_error(
            anchor=anchor, el_at_anchor=el_at_anchor, body=body, ra=ra, dec=dec, site=site
        )
    return derived


def _anchor_drift_guard(
    slope: float,
    *,
    anchor: Time,
    el_at_anchor: float,
    body: str | None,
    ra: float | None,
    dec: float | None,
    el_limits_min: float,
    el_limits_max: float,
) -> None:
    """Reject anchors too near transit to derive ``el_bore`` from.

    Raises :class:`TargetNotObservableError` when the source elevation drift
    rate at the anchor is below :data:`_MIN_ANCHOR_EL_DRIFT_DEG_S`.
    """
    if abs(slope) >= _MIN_ANCHOR_EL_DRIFT_DEG_S:
        return
    label = source_ces._describe_source(body, ra, dec)
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
    body: str | None,
    ra: float | None,
    dec: float | None,
    pm_ra: float,
    pm_dec: float,
    ref_epoch: Time | None,
    site: Site,
    atmosphere: AtmosphericConditions | None,
) -> tuple[Time, Coordinates, Literal["rising", "setting"], float]:
    """Shared validation and slope probe for the anchored entry points.

    Common front half of every ``start_time`` resolution: validates that
    ``start_time`` is not combined with ``night`` or ``window``, parses the
    anchor, and, when ``el_bore`` or ``mode`` is omitted, probes the source
    elevation slope at the anchor. The slope sign resolves an omitted
    ``mode``; the near-transit drift guard runs when ``el_bore`` must be
    derived. Returns ``(anchor, coords, mode, el_at_anchor)``;
    ``el_at_anchor`` is meaningful whenever ``el_bore`` is ``None`` (the
    probe is guaranteed to have run in that case).
    """
    if night is not None:
        raise ValueError("specify either 'start_time' or 'night', not both")
    if window is not None:
        raise ValueError("specify either 'start_time' or 'window', not both")

    anchor = _coerce_start_time(start_time)
    coords = Coordinates(site, atmosphere=atmosphere)

    resolved_mode = mode
    el_at_anchor = 0.0
    if el_bore is None or mode is None:
        el_at_anchor, slope = _probe_anchor_slope(
            coords,
            anchor,
            body=body,
            ra=ra,
            dec=dec,
            pm_ra=pm_ra,
            pm_dec=pm_dec,
            ref_epoch=ref_epoch,
        )
        if resolved_mode is None:
            resolved_mode = "rising" if slope >= 0.0 else "setting"
        if el_bore is None:
            el_limits = site.telescope_limits.elevation
            _anchor_drift_guard(
                slope,
                anchor=anchor,
                el_at_anchor=el_at_anchor,
                body=body,
                ra=ra,
                dec=dec,
                el_limits_min=el_limits.min,
                el_limits_max=el_limits.max,
            )

    assert resolved_mode is not None  # mode was given, or set from the probe slope above
    return anchor, coords, resolved_mode, el_at_anchor


def _resolve_start_time_anchor(
    *,
    start_time: Time | str,
    el_bore: float | None,
    mode: Literal["rising", "setting"] | None,
    night: Time | None,
    window: tuple[Time, Time] | None,
    footprint: InstrumentOffset | str | Sequence[InstrumentOffset] | ArrayFootprint,
    body: str | None,
    ra: float | None,
    dec: float | None,
    pm_ra: float,
    pm_dec: float,
    ref_epoch: Time | None,
    boresight_rot: float | None,
    site: Site,
    atmosphere: AtmosphericConditions | None,
    sampling_step_seconds: float,
    az_accel: float,
    az_padding: float,
    az_branch: float | None,
) -> tuple[float, tuple[Time, Time], Literal["rising", "setting"]]:
    """Resolve an approximate ``start_time`` anchor into ``(el_bore, window, mode)``.

    Thin resolution layer that sits in front of the classic search machinery.
    :func:`_resolve_anchor_prefix` handles the shared validation, mode
    resolution, and near-transit guard; this then derives ``el_bore`` so the
    pass starts approximately at the anchor (when it was omitted) and returns
    a forward search ``window`` opening at the anchor. The caller runs the
    existing kernel with the returned values, so the classic paths are
    untouched.
    """
    anchor, coords, resolved_mode, el_at_anchor = _resolve_anchor_prefix(
        start_time=start_time,
        el_bore=el_bore,
        mode=mode,
        night=night,
        window=window,
        body=body,
        ra=ra,
        dec=dec,
        pm_ra=pm_ra,
        pm_dec=pm_dec,
        ref_epoch=ref_epoch,
        site=site,
        atmosphere=atmosphere,
    )

    horizon = TimeDelta(source_ces._DEFAULT_SEARCH_HORIZON_HOURS * 3600.0 * u.s)
    resolved_window = (anchor, anchor + horizon)

    if el_bore is None:
        el_bore = _derive_anchored_el_bore(
            anchor=anchor,
            el_at_anchor=el_at_anchor,
            mode=resolved_mode,
            coords=coords,
            fp=source_ces._resolve_footprint(footprint),
            body=body,
            ra=ra,
            dec=dec,
            pm_ra=pm_ra,
            pm_dec=pm_dec,
            ref_epoch=ref_epoch,
            boresight_rot=boresight_rot,
            site=site,
            atmosphere=atmosphere,
            sampling_step_seconds=sampling_step_seconds,
            az_accel=az_accel,
            az_padding=az_padding,
            az_branch=az_branch,
        )

    return float(el_bore), resolved_window, resolved_mode
