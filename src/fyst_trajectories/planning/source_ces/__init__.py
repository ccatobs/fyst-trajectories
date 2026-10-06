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

``plan_source_ces`` builds the trajectory a control system dispatches;
the params-only sibling :func:`compute_source_ces_params` is the
emit-time entry point for a scheduler that only needs the numbers.

Both reach one kernel, run in stages (``_kernel``), through one path that
first resolves a ``start_time`` anchor (``_anchor``); ``_source`` describes the
source, ``passes`` holds the helpers of :func:`plan_source_ces_passes` and
``track`` the focal-plane track of a planned pass.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from astropy import units as u
from astropy.time import Time, TimeDelta

from ..._validation import _require_positive
from ...exceptions import AzimuthBoundsError, PointingWarning
from ...offsets import InstrumentOffset
from ...patterns.configs import ConstantElScanConfig
from ...patterns.turnarounds import swept_az_envelope
from ...site import AtmosphericConditions, Site
from ...trajectory import TrajectoryMetadata
from ...trajectory_utils import validate_trajectory_bounds, validate_trajectory_dynamics
from .._helpers import _build_altaz_trajectory
from .._types import ArrayFootprint, ScanBlock, SourceCESComputedParams
from ..footprints import offset_footprint_eta, resolve_footprint
from ._anchor import _derive_anchored_el_bore, _resolve_anchor_prefix, _resolve_start_time_anchor
from ._kernel import (
    _DEFAULT_AZ_PADDING_DEG,
    _DEFAULT_SEARCH_HORIZON_HOURS,
    _compute_source_ces_core,
    _SourceCESCore,
)
from ._source import _SourceSpec
from .passes import _resolve_pass_offsets, _tag_pass_block
from .track import source_ces_focal_plane_track

if TYPE_CHECKING:
    # Annotation-only import: the predicate is invoked structurally, so only
    # the type hint needs the symbol.
    from ...sun_protocols import SunSafePredicate

__all__ = [
    "compute_source_ces_params",
    "plan_source_ces",
    "plan_source_ces_passes",
    "source_ces_focal_plane_track",
]


def _resolve_and_compute(
    *,
    source: _SourceSpec,
    footprint: InstrumentOffset | str | Sequence[InstrumentOffset] | ArrayFootprint,
    el_bore: float | None,
    boresight_rot: float | None,
    window: tuple[Time, Time] | None,
    night: Time | None,
    start_time: Time | str | None,
    mode: Literal["rising", "setting"] | None,
    site: Site,
    atmosphere: AtmosphericConditions | None,
    sampling_step_seconds: float,
    az_accel: float,
    az_padding: float,
    az_branch: float | None,
    allow_partial: bool,
    v_az: float | None,
    sun_safe: SunSafePredicate | None,
    az_speed: float | None,
    az_throw: float | None,
    dwell: float | None,
) -> _SourceCESCore:
    """Resolve a ``start_time`` anchor, then run the kernel: the one single-pass path.

    :func:`compute_source_ces_params` and :func:`plan_source_ces` both reach
    the kernel through here, so the scalars a scheduler prices at emit time
    are the ones the dispatched trajectory is built from. With ``start_time``
    the anchor derives the forward search window, and ``el_bore`` and
    ``mode`` when they are omitted; without it ``el_bore`` is required. The
    parameters are the entry points' own, with the six source keywords in
    ``source``.

    Raises
    ------
    ValueError
        When neither ``el_bore`` nor ``start_time`` is given; otherwise what
        the anchor resolution and the kernel raise.
    """
    if start_time is not None:
        el_bore, window, mode = _resolve_start_time_anchor(
            start_time=start_time,
            el_bore=el_bore,
            mode=mode,
            night=night,
            window=window,
            footprint=footprint,
            source=source,
            boresight_rot=boresight_rot,
            site=site,
            atmosphere=atmosphere,
            sampling_step_seconds=sampling_step_seconds,
            az_accel=az_accel,
            az_padding=az_padding,
            az_branch=az_branch,
        )
    elif el_bore is None:
        raise ValueError("el_bore is required unless 'start_time' is given")

    return _compute_source_ces_core(
        source=source,
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
        az_speed=az_speed,
        az_throw=az_throw,
        dwell=dwell,
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
    el_bore: float | None = None,
    boresight_rot: float | None = None,
    # --- Time window ---
    window: tuple[Time, Time] | None = None,
    night: Time | None = None,
    start_time: Time | str | None = None,
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
) -> SourceCESComputedParams:
    """Compute source-CES scalar parameters without building the trajectory.

    Params-only sibling of :func:`plan_source_ces`. Returns just the
    :class:`SourceCESComputedParams` dict, skipping the per-sample
    trajectory generation. This is the emit-time
    entry point: a scheduler can price many candidate scans cheaply
    (feasibility, duration, azimuth throw) from the scalars alone and
    discard the trajectory, which the execution layer generates once at
    dispatch.

    All keyword arguments are identical to :func:`plan_source_ces`
    except that ``timestep`` is omitted - only the trajectory builder
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
    el_bore : float, optional
        Fixed boresight elevation in degrees. Required unless
        ``start_time`` is given, in which case it is derived so the pass
        starts near the anchor (pass it explicitly to instead force a
        forward search from the anchor for that elevation).
    boresight_rot : float, optional
        Mechanical boresight rotation in degrees.
    window : (Time, Time), optional
        Explicit search window.
    night : Time, optional
        Start of the search window (use with ``mode``).
    start_time : Time or str, optional
        Approximate anchor: plan the pass to begin near this time.
        Mutually exclusive with ``night`` and ``window``. When given,
        ``el_bore`` and ``mode`` are derived if omitted. See
        :func:`plan_source_ces` for the full semantics.
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
        Extra azimuth padding on each side, in degrees. Default 0.5. Must be non-negative.
    az_branch : float, optional
        Centre of azimuth wrap branch.
    allow_partial : bool, optional
        If ``True``, downgrade footprint-not-fully-covered to a warning.
    v_az : float, optional
        Override the solved azimuth drift rate (deg/s; a mount-frame
        azimuth coordinate rate, not an on-sky speed).
    sun_safe : SunSafePredicate, optional
        Sun-safety predicate implementing the
        :class:`~fyst_trajectories.sun_protocols.SunSafePredicate` contract. ``None``
        (default) keeps the built-in scalar exclusion-radius arc check; an
        injected predicate is consulted per-sample along the planned arc
        instead, so the directional sun-avoidance model
        (see :func:`~fyst_trajectories.sun_models.make_sun_safe`) is honored.
        Warn-only either way.
    az_speed : float, optional
        Per-leg azimuth speed of the sweep in deg/s. ``None`` (default)
        derives it so a single azimuth leg spans the scanned window,
        floored at a slow drag; an explicit value makes the
        sweep a fast drag at that speed. Distinct from ``v_az``: ``v_az``
        is the slow drift that keeps the window on the source, ``az_speed``
        is how fast the telescope crosses the window within it. Recorded
        as ``computed_params["az_speed"]`` either way.
    az_throw : float, optional
        Width of the swept azimuth window in degrees. ``None`` (default)
        uses the solved footprint crossing plus ``az_padding`` on each
        side; an explicit value replaces that padded throw, re-centred on
        the solved window, and cannot be combined with an explicit
        ``az_padding``. A value narrower than the footprint crossing warns
        (the source leaves the window during the pass).
    dwell : float, optional
        Time on source in seconds. ``None`` (default) scans the whole
        footprint crossing; an explicit value narrows the solved pass
        symmetrically about the crossing midpoint, so ``t0_iso`` and
        ``t1_iso`` report the narrowed source window and
        ``computed_params["crossing_seconds"]`` keeps the full crossing.
        ``duration`` is that window quantised to whole azimuth legs, so
        it agrees with ``t1_iso - t0_iso`` only to within half a leg plus
        turnaround (about 26 s on the slow-drag default, a few seconds at
        a fast drag). A value longer than the crossing is rejected (every
        extra second has the source outside the footprint), one shorter
        than ``sampling_step_seconds`` is rejected as unresolvable, and a
        shorter one in between warns as a partial pass. With
        ``start_time`` anchoring the anchor places the full crossing, so
        the narrowed pass starts half the cut later.

    Returns
    -------
    SourceCESComputedParams
        Scalar parameters describing the planned source-CES - the same
        dict that ``plan_source_ces(...).computed_params`` returns.

    Raises
    ------
    ValueError
        On incompatible argument combinations, a non-positive step, speed,
        acceleration, throw or dwell, a negative ``az_padding``, a reversed
        window, a ``dwell`` shorter than ``sampling_step_seconds``, or a
        footprint whose cover is a single point with ``az_padding=0`` and
        no ``az_throw`` (a sweep with no width at any time).
    DwellExceedsCrossingError
        When ``dwell`` is longer than the solved footprint crossing. A
        :class:`~fyst_trajectories.exceptions.PointingError` subclass that
        carries ``dwell`` and ``crossing_seconds``.
    TargetNotObservableError
        When the source never reaches ``el_bore`` in the search window, when
        ``allow_partial=False`` and the source's elevation span does not
        cover the footprint at ``el_bore``, or when a ``start_time`` anchor
        is too near transit, or places the derived ``el_bore`` outside the
        elevation limits.
    ElevationBoundsError
        When ``el_bore`` lies outside ``site.telescope_limits.elevation``.
    KeyError, TypeError
        From :func:`~fyst_trajectories.planning.resolve_footprint`: a string
        that names no module, or a footprint of an unsupported type.
    AzimuthBoundsError
        When the commanded envelope exceeds ``site.telescope_limits.azimuth``.
        That envelope is the returned ``[az_start, az_start + az_throw]`` (the
        solved window, with any padding already applied) widened on each side
        by the turnaround overshoot
        (:func:`~fyst_trajectories.patterns.turnarounds.swept_az_envelope`)
        and extended by ``v_az * duration`` on the side the drift runs to.
        This is a cheap, conservative pre-build check;
        :func:`plan_source_ces` checks every sample of the built trajectory
        via ``validate_trajectory_bounds`` instead, and near a limit can
        accept a pass this check refuses.
    PointingError
        When the source reaches no vertex of the footprint's cover: with
        ``allow_partial=True``, or when ``el_bore`` is derived from a
        ``start_time`` anchor too near culmination for the source to climb
        onto the footprint. When it crosses the cover at a single azimuth
        (one vertex of a partial cover in reach, coincident vertices, or
        vertices an explicit ``v_az`` lines up) with ``az_padding=0`` and no
        ``az_throw``, so the swept window has no width. When the Nelder-Mead
        optimisation fails and no fallback
        ``v_az`` can be derived from the source's median az speed.
    OffsetInversionError
        When the boresight inverse for an off-centre footprint cannot be
        solved at ``el_bore``. A :class:`~fyst_trajectories.exceptions.PointingError` subclass.

    Warns
    -----
    PointingWarning
        - The planned arc passes within the site sun-avoidance exclusion
          radius at any sample, or an injected ``sun_safe`` model reports
          one unsafe. The screened arc is the commanded azimuth envelope,
          so it includes the turnaround overshoot beyond each science edge.
        - The leg speed plus ``abs(v_az)`` exceeds the site's azimuth
          velocity limit.
        - The ``v_az`` optimisation did not converge and a median source
          azimuth speed was used instead.
        - ``allow_partial=True`` and the source's elevation span does not
          cover the footprint (the pass is clipped to the overlap).
        - ``dwell`` is shorter than the footprint crossing (a partial pass).
        - ``az_throw`` is narrower than the footprint crossing (the source
          leaves the swept window during the pass).

        No trajectory is built, so the scan-config and trajectory-dynamics
        advisories of :func:`plan_source_ces` do not arise.

    Notes
    -----
    See :func:`plan_source_ces` for the same computation plus per-sample
    trajectory generation. The params-only path avoids allocating
    ~370 KB of trajectory arrays on a typical 15-minute Jupiter scan at
    ``timestep=0.1``. The array build itself is cheap, so the saving is
    allocation and memory rather than compute - which is what matters
    when an upstream scheduler emits dozens of source_ces blocks per
    tactical pass and discards the trajectory.

    Examples
    --------
    Compute source-CES scalars for a Jupiter rising scan on PrimeCam's
    centre module, without building the trajectory:

    >>> from astropy.time import Time
    >>> from fyst_trajectories import get_fyst_site
    >>> from fyst_trajectories.planning import compute_source_ces_params
    >>> params = compute_source_ces_params(
    ...     body="jupiter",
    ...     footprint="c",
    ...     el_bore=35.0,
    ...     night=Time("2026-03-15T00:00:00", scale="utc"),
    ...     mode="rising",
    ...     site=get_fyst_site(),
    ... )
    >>> # params is a SourceCESComputedParams (TypedDict / plain dict).
    """
    core = _resolve_and_compute(
        source=_SourceSpec(
            body=body, ra=ra, dec=dec, pm_ra=pm_ra, pm_dec=pm_dec, ref_epoch=ref_epoch
        ),
        footprint=footprint,
        el_bore=el_bore,
        boresight_rot=boresight_rot,
        window=window,
        night=night,
        start_time=start_time,
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
        az_speed=az_speed,
        az_throw=az_throw,
        dwell=dwell,
    )

    # Envelope-only az bounds check. The commanded sweep (the padded
    # window [az_start, az_stop] widened by the turnaround overshoot on
    # each side) plus the linear drift across the source pass duration
    # gives the extreme az values the executed trajectory will hit,
    # without building per-sample arrays. ``plan_source_ces`` checks every
    # sample of the built trajectory via ``validate_trajectory_bounds``
    # instead. The envelope is the conservative one: a one- or two-leg pass
    # has fewer than two turnarounds, so near a limit this check can refuse
    # a pass the per-sample check would accept.
    az_limits = site.telescope_limits.azimuth
    cp = core.computed
    pass_duration = core.actual_duration
    drift_total = cp["v_az"] * pass_duration
    env_lo, env_hi = swept_az_envelope(
        min(cp["az_start"], cp["az_start"] + cp["az_throw"]),
        max(cp["az_start"], cp["az_start"] + cp["az_throw"]),
        cp["az_speed"],
        az_accel,
    )
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
    el_bore: float | None = None,
    boresight_rot: float | None = None,
    # --- Time window ---
    window: tuple[Time, Time] | None = None,
    night: Time | None = None,
    start_time: Time | str | None = None,
    mode: Literal["rising", "setting"] | None = None,
    # --- Site / atmosphere ---
    site: Site,
    atmosphere: AtmosphericConditions | None = None,
    # --- Algorithm ---
    timestep: float = 0.1,
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
) -> ScanBlock[SourceCESComputedParams]:
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
        :data:`fyst_trajectories.coordinates.SOLAR_SYSTEM_BODIES`). Mutually
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

        * **InstrumentOffset** - a single offset (e.g. one PrimeCam
          module). Built as a 50-vertex circle around ``(dx, dy)``
          with radius :data:`~fyst_trajectories.primecam.MODULE_FOV_RADIUS_DEG`.
        * **str** - a named PrimeCam module ("c", "i1", ...); resolved
          via :func:`~fyst_trajectories.primecam.get_primecam_offset`.
        * **sequence of InstrumentOffset** - one entry per module;
          footprint is the union of per-module circles; the aggregate
          center is the arithmetic mean of per-module ``(dx, dy)``.
        * **ArrayFootprint** - explicit (center, cover) representation;
          mirrors the ``array_info`` dict that SO ``make_source_ces``
          consumes.
    el_bore : float, optional
        Fixed boresight elevation in degrees. Must lie within
        ``site.telescope_limits.elevation``. Required for the classic
        ``night``/``window`` forms. Optional when ``start_time`` is
        given: if omitted it is derived so the pass starts near the
        anchor; if supplied it forces a forward search from the anchor
        for that elevation.
    boresight_rot : float, optional
        Mechanical boresight rotation in degrees, added to the
        focal-plane rotation when projecting the cover. ``None``
        (default) is treated as ``0.0``, and ``computed_params`` records
        ``0.0`` for both. An execution layer may accept only ``None``, so
        a dict sent to one carries the request, not the recorded value.
    window : (Time, Time), optional
        Explicit ``(t_start, t_end)`` search window. Mutually exclusive
        with ``night``/``mode``.
    night : Time, optional
        Start of the search window. Used with ``mode`` to pick the
        first rising or setting pass of the source within the next
        24 h.
    start_time : Time or str, optional
        Approximate anchor: plan the pass to begin near this time,
        mirroring ``plan_constant_el_scan``'s ``start_time`` (an
        approximate search anchor, not a literal start). Mutually
        exclusive with ``night`` and ``window`` (``ValueError`` if
        combined). The search runs forward from the anchor over the
        default 24 h horizon. When ``el_bore`` is omitted it is derived
        so the resolved start typically lands within about a minute
        after the anchor (the window opens at the anchor, so the pass
        cannot begin earlier). When ``mode`` is omitted it is taken
        from the sign of the source's elevation slope at the anchor.
        Anchors within a small drift rate of transit are rejected with
        :class:`~fyst_trajectories.exceptions.TargetNotObservableError`; anchor
        away from transit or pass ``el_bore`` explicitly.
    mode : {"rising", "setting"}, optional
        Which monotonic arc of the source to use. Required when
        ``night`` is given. With ``window``, omitting ``mode``
        auto-detects: the planner picks the longest monotonic arc
        inside the window whose elevation range covers ``el_bore`` and
        sets ``mode`` to ``"rising"`` or ``"setting"`` based on its
        slope. With ``start_time``, omitting ``mode`` takes it from the
        elevation slope sign at the anchor. Pass an explicit ``mode``
        to override.
    site : Site
        Telescope site.
    atmosphere : AtmosphericConditions, optional
        Refraction model passed to the underlying
        :class:`~fyst_trajectories.coordinates.Coordinates`.
        Default ``None`` (vacuum) - matches the rest of the planning
        subpackage. Refraction is applied downstream at execution time
        (by exactly one of the Go TCS or the ACU), so vacuum is correct.
    timestep : float, optional
        Time between trajectory samples in seconds. Default 0.1.
    sampling_step_seconds : float, optional
        Coarse time step used when sampling the source's az(t)/el(t)
        curve for crossing detection and ``v_az`` optimisation.
        Default 30.0 (matches SO).
    az_accel : float, optional
        Azimuth acceleration for the executed scan, deg/s^2. Default
        1.0 (FYST conservative).
    az_padding : float, optional
        Extra azimuth padding on each side of the solved
        ``[az_start, az_start + az_throw]`` interval, in degrees.
        Default 0.5. Must be non-negative.
    az_branch : float, optional
        Centre of the azimuth wrap branch. If given, ``az_start`` is
        re-expressed in ``[az_branch - 180, az_branch + 180)``.
        Default ``None`` (no rewrap).
    allow_partial : bool, optional
        Behaviour when the source's elevation span does not cover the
        full footprint at ``el_bore`` (i.e. some cover vertices fall
        outside the source's elevation range). Default ``False``
        raises :class:`~fyst_trajectories.exceptions.TargetNotObservableError`.
        Pass ``True`` to clip ``(t0, t1)`` to the overlap and emit a
        :class:`~fyst_trajectories.exceptions.PointingWarning` instead.
    v_az : float, optional
        Override the solved azimuth drift rate (deg/s) instead of
        running the Nelder-Mead optimisation.
    sun_safe : SunSafePredicate, optional
        Sun-safety predicate implementing the
        :class:`~fyst_trajectories.sun_protocols.SunSafePredicate` contract,
        ``(az_deg, el_deg, time) -> bool`` returning ``True`` when the
        position is clear of the Sun. ``None`` (default) keeps the built-in
        scalar exclusion-radius check along the planned arc; an injected
        predicate is consulted per-sample instead, so the directional
        sun-avoidance model (see :func:`~fyst_trajectories.sun_models.make_sun_safe`)
        is honored end-to-end. Warn-only either way. See
        :class:`~fyst_trajectories.sun_protocols.SunSafePredicate`.
    az_speed : float, optional
        Per-leg azimuth speed of the sweep in deg/s. ``None`` (default)
        derives it so a single azimuth leg spans the scanned window,
        floored at a slow drag; an explicit value makes the
        sweep a fast drag at that speed. Distinct from ``v_az``: ``v_az``
        is the slow drift that keeps the window on the source, ``az_speed``
        is how fast the telescope crosses the window within it. Recorded
        as ``computed_params["az_speed"]`` either way.
    az_throw : float, optional
        Width of the swept azimuth window in degrees. ``None`` (default)
        uses the solved footprint crossing plus ``az_padding`` on each
        side; an explicit value replaces that padded throw, re-centred on
        the solved window, and cannot be combined with an explicit
        ``az_padding``. A value narrower than the footprint crossing warns
        (the source leaves the window during the pass).
    dwell : float, optional
        Time on source in seconds. ``None`` (default) scans the whole
        footprint crossing; an explicit value narrows the solved pass
        symmetrically about the crossing midpoint, so ``t0_iso`` and
        ``t1_iso`` report the narrowed source window and
        ``computed_params["crossing_seconds"]`` keeps the full crossing.
        ``duration`` is that window quantised to whole azimuth legs, so
        it agrees with ``t1_iso - t0_iso`` only to within half a leg plus
        turnaround (about 26 s on the slow-drag default, a few seconds at
        a fast drag). A value longer than the crossing is rejected (every
        extra second has the source outside the footprint), one shorter
        than ``sampling_step_seconds`` is rejected as unresolvable, and a
        shorter one in between warns as a partial pass. With
        ``start_time`` anchoring the anchor places the full crossing, so
        the narrowed pass starts half the cut later.

    Returns
    -------
    ScanBlock
        Planned observation. ``trajectory`` is a constant-elevation
        scan with the solved drift baked in. ``config`` is a
        :class:`~fyst_trajectories.patterns.ConstantElScanConfig`.
        ``computed_params`` is a :class:`SourceCESComputedParams`.
        ``trajectory.metadata.pattern_params["body"]`` records the source:
        the lower-case body name, or ``None`` for an RA/Dec source
        (``target_name`` is a display label).

    Raises
    ------
    ValueError
        On incompatible argument combinations, a non-positive step, speed,
        acceleration, throw or dwell, a negative ``az_padding``, a reversed
        window, a ``dwell`` shorter than ``sampling_step_seconds``, or a
        footprint whose cover is a single point with ``az_padding=0`` and
        no ``az_throw`` (a sweep with no width at any time).
    DwellExceedsCrossingError
        When ``dwell`` is longer than the solved footprint crossing. A
        :class:`~fyst_trajectories.exceptions.PointingError` subclass that
        carries ``dwell`` and ``crossing_seconds``.
    TargetNotObservableError
        When the source never reaches ``el_bore`` in the search
        window, when ``allow_partial=False`` and the source's
        elevation span doesn't cover the footprint at ``el_bore``, or
        when a ``start_time`` anchor is too near transit, or places the
        derived ``el_bore`` outside the elevation limits.
    PointingError
        When the source reaches no vertex of the footprint's cover: with
        ``allow_partial=True``, or when ``el_bore`` is derived from a
        ``start_time`` anchor too near culmination for the source to climb
        onto the footprint. When it crosses the cover at a single azimuth
        (one vertex of a partial cover in reach, coincident vertices, or
        vertices an explicit ``v_az`` lines up) with ``az_padding=0`` and no
        ``az_throw``, so the swept window has no width. When the Nelder-Mead
        optimisation fails and no fallback
        ``v_az`` can be derived from the source's median az speed.
    AzimuthBoundsError, ElevationBoundsError
        When the built trajectory leaves the telescope envelope (the
        post-build ``validate_trajectory_bounds``, or ``el_bore``
        outside the elevation limits).
    KeyError, TypeError
        From :func:`~fyst_trajectories.planning.resolve_footprint`: a string
        that names no module, or a footprint of an unsupported type.
    OffsetInversionError
        When the boresight inverse for an off-centre footprint cannot be
        solved at ``el_bore``. A :class:`~fyst_trajectories.exceptions.PointingError` subclass.

    Warns
    -----
    PointingWarning
        - The planned arc passes within the site sun-avoidance exclusion
          radius at any sample, or an injected ``sun_safe`` model reports
          one unsafe. The screened arc is the commanded azimuth envelope,
          so it includes the turnaround overshoot beyond each science edge.
        - The leg speed plus ``abs(v_az)`` exceeds the site's azimuth
          velocity limit.
        - The ``v_az`` optimisation did not converge and a median source
          azimuth speed was used instead.
        - ``allow_partial=True`` and the source's elevation span does not
          cover the footprint (the pass is clipped to the overlap).
        - ``dwell`` is shorter than the footprint crossing (a partial pass).
        - ``az_throw`` is narrower than the footprint crossing (the source
          leaves the swept window during the pass).
        - The scan's :class:`~fyst_trajectories.patterns.ConstantElScanConfig`
          warns: an unusually large leg speed or ``az_accel``, or a
          turnaround overshoot wider than the throw.
        - :func:`~fyst_trajectories.trajectory_utils.validate_trajectory_dynamics`
          on the returned trajectory reports the on-sky azimuth compression
          of an ``el_bore`` above about 60 deg, or too few samples for its
          acceleration check.
    VelocityLimitWarning, AccelerationLimitWarning
        The drifted trajectory exceeds an axis velocity or acceleration
        limit (:func:`~fyst_trajectories.trajectory_utils.validate_trajectory_dynamics`
        on the returned trajectory; the quintic turnaround peaks at 1.5
        times ``az_accel``, so an ``az_accel`` above two thirds of the
        site's azimuth acceleration limit puts that peak over the limit).

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
    raises :class:`~fyst_trajectories.exceptions.AzimuthBoundsError`. For FYST
    (limits -180 to 360 deg), ``az_branch`` values near -180 deg can
    produce out-of-range scans even when geometrically valid.

    See :doc:`/planning` ("Source CES") for the wider conventions
    discussion.

    Examples
    --------
    Plan a Jupiter rising CES on PrimeCam's centre module:

    >>> from astropy.time import Time
    >>> from fyst_trajectories import get_fyst_site
    >>> from fyst_trajectories.planning import plan_source_ces
    >>> block = plan_source_ces(
    ...     body="jupiter",
    ...     footprint="c",
    ...     el_bore=35.0,
    ...     night=Time("2026-03-15T00:00:00", scale="utc"),
    ...     mode="rising",
    ...     site=get_fyst_site(),
    ... )

    Or anchor the pass to begin near an approximate ``start_time`` and let
    the planner derive ``el_bore`` and ``mode`` (rising, here) for you:

    >>> block = plan_source_ces(
    ...     body="jupiter",
    ...     footprint="c",
    ...     start_time=Time("2026-03-15T21:41:00", scale="utc"),
    ...     site=get_fyst_site(),
    ... )
    """
    core = _resolve_and_compute(
        source=_SourceSpec(
            body=body, ra=ra, dec=dec, pm_ra=pm_ra, pm_dec=pm_dec, ref_epoch=ref_epoch
        ),
        footprint=footprint,
        el_bore=el_bore,
        boresight_rot=boresight_rot,
        window=window,
        night=night,
        start_time=start_time,
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
        az_speed=az_speed,
        az_throw=az_throw,
        dwell=dwell,
    )

    computed = core.computed
    el_bore = core.el_bore
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

    # The dynamics check runs below on the drifted trajectory, the one
    # actually returned, rather than on this undrifted base.
    base_traj = _build_altaz_trajectory(
        site=site,
        config=config,
        duration=actual_duration,
        start_time=t0,
        atmosphere=atmosphere,
        detector_offset=None,
        validate_dynamics=False,
    )

    drifted_az = base_traj.az + v_az_solved * base_traj.times
    drifted_az_vel = base_traj.az_vel + v_az_solved
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
            "body": None if body is None else body.lower(),
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
    # Post-drift dynamics check (advisory): the drift adds v_az to every
    # leg velocity, so the limits are judged on the returned trajectory.
    validate_trajectory_dynamics(site, trajectory.az, trajectory.el, trajectory.times)

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


def plan_source_ces_passes(
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
    el_bore: float | None = None,
    boresight_rot: float | None = None,
    # --- Pass controls ---
    n_passes: int | None = None,
    step: float | None = None,
    eta_offsets: Sequence[float] | None = None,
    el_step: float | None = None,
    # --- Time window ---
    window: tuple[Time, Time] | None = None,
    night: Time | None = None,
    start_time: Time | str | None = None,
    mode: Literal["rising", "setting"] | None = None,
    # --- Site / atmosphere ---
    site: Site,
    atmosphere: AtmosphericConditions | None = None,
    # --- Algorithm ---
    timestep: float = 0.1,
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
) -> list[ScanBlock[SourceCESComputedParams]]:
    """Plan a sequence of source-CES passes for full focal-plane coverage.

    A single :func:`plan_source_ces` drags a moving source across the
    array footprint at one fixed boresight elevation, so the source only
    paints a sparse raster along one band of the focal plane. This
    wrapper builds a sequence of drift passes with the source stepped
    through different rows of the array: it returns ``list[ScanBlock]``,
    one per pass, each an ordinary :func:`plan_source_ces` block, ordered
    in time.

    Two knobs are stepped between passes, and they are independent:

    * **Coverage** is moved by offsetting the *footprint* along the
      focal-plane eta (elevation) axis. This is the correct knob: it
      slides the source's track to a different row 1:1. Stepping
      ``el_bore`` alone does *not* move the coverage, because a
      source-tracking CES re-centres on the source at every boresight
      elevation, so each ``el_bore`` reproduces the same focal-plane
      band.
    * **Timing** is set by stepping ``el_bore``. A rising source crosses
      a lower boresight elevation earlier and a higher one later, so
      stepping ``el_bore`` by ``el_step`` sequences the passes in time.
      ``el_step`` defaults to the footprint eta extent, which keeps the
      per-pass source windows from overlapping (the source must climb
      past the whole footprint height between passes).

    Because these two knobs are decoupled, the sequence both densifies
    coverage across the footprint (fine eta offsets) and stays
    non-overlapping in time (an ``el_step`` of order the footprint
    extent). Reducing ``el_step`` below the footprint extent will overlap
    the passes in time, and a
    :class:`~fyst_trajectories.exceptions.PointingWarning` is emitted when
    the returned pass windows overlap.

    Parameters
    ----------
    body : str, optional
        Solar-system body name; mutually exclusive with ``ra``/``dec``.
    ra, dec : float, optional
        Sidereal source position in degrees.
    pm_ra, pm_dec : float, optional
        Proper motion in mas/yr. Default 0.0.
    ref_epoch : Time, optional
        Reference epoch for ``ra``/``dec``; required with non-zero proper
        motion.
    footprint : InstrumentOffset, str, sequence of InstrumentOffset, or ArrayFootprint
        Base array footprint the source must traverse. Each pass shifts a
        copy of this footprint along the eta axis. See
        :func:`plan_source_ces` for the accepted forms.
    el_bore : float, optional
        Central boresight elevation in degrees. The passes step
        symmetrically around this value in ``el_step`` increments; for an
        odd ``n_passes`` the middle pass uses ``el_bore`` itself, while an
        even count straddles it. Required for the classic
        ``night``/``window`` forms. When ``start_time`` is given and this
        is omitted, the central elevation is derived so the first pass in
        time starts near the anchor.
    boresight_rot : float, optional
        Mechanical boresight rotation in degrees, forwarded to every pass.
    n_passes : int, optional
        Number of passes. Mutually exclusive with ``eta_offsets``; one of
        the two is required. Must be at least 1.
    step : float, optional
        Eta spacing in degrees between adjacent pass centers. Only valid
        with ``n_passes``. Defaults to ``footprint_eta_extent / n_passes``,
        which spreads the pass centers evenly across the footprint eta
        extent.
    eta_offsets : sequence of float, optional
        Explicit focal-plane eta offsets in degrees, one per pass. The
        source is dragged through the row at each offset. Mutually
        exclusive with ``n_passes``/``step``. Sorted internally, so the
        input order does not matter.
    el_step : float, optional
        Boresight-elevation spacing in degrees between consecutive passes.
        Defaults to the footprint eta extent, chosen so the source climbs
        past the full footprint height between passes and the source
        windows do not overlap. Must be positive.
    window : (Time, Time), optional
        Explicit search window, forwarded to each pass. Mutually
        exclusive with ``night``.
    night : Time, optional
        Start of the 24 h search window, forwarded to each pass. Used with
        ``mode``.
    start_time : Time or str, optional
        Approximate anchor for the first pass in time. Mutually exclusive
        with ``night`` and ``window``. When given, ``el_bore`` and
        ``mode`` are derived if omitted (see :func:`plan_source_ces`); the
        first chronological pass then starts near the anchor and the rest
        follow at the usual ``el_step`` spacing.
    mode : {"rising", "setting"}, optional
        Direction of the source arc, forwarded to each pass. With
        ``start_time`` and no ``mode``, taken from the elevation slope
        sign at the anchor.
    site : Site
        Telescope site.
    atmosphere : AtmosphericConditions, optional
        Refraction model, forwarded to each pass. Default vacuum.
    timestep : float, optional
        Trajectory sample spacing in seconds. Default 0.1.
    sampling_step_seconds : float, optional
        Coarse source-sampling step in seconds. Default 30.0.
    az_accel : float, optional
        Azimuth acceleration in deg/s^2. Default 1.0.
    az_padding : float, optional
        Extra azimuth padding per side in degrees. Default 0.5. Must be non-negative.
    az_branch : float, optional
        Centre of the azimuth wrap branch.
    allow_partial : bool, optional
        Forwarded to each pass. Default ``False``.
    v_az : float, optional
        Override the solved azimuth drift rate for every pass (deg/s).
    sun_safe : SunSafePredicate, optional
        Sun-safety predicate forwarded to each pass.
    az_speed : float, optional
        Per-leg azimuth speed in deg/s, forwarded to each pass (see
        :func:`plan_source_ces`).
    az_throw : float, optional
        Swept azimuth window in degrees, forwarded to each pass (see
        :func:`plan_source_ces`).
    dwell : float, optional
        Time on source in seconds, forwarded to the pass (see
        :func:`plan_source_ces`). Accepted only for a single pass: a
        multi-pass sequence with a narrowed window is rejected.

    Returns
    -------
    list of ScanBlock
        One block per pass, ordered by start time. Each block is an
        ordinary :func:`plan_source_ces` result (same guarantees and
        ``computed_params`` schema) with per-pass metadata added to
        ``trajectory.metadata.pattern_params`` (``pass_index``,
        ``n_passes``, ``pass_eta_offset_deg``, ``pass_el_bore_deg``) and
        to the summary header.

    Raises
    ------
    ValueError
        If neither or both of ``n_passes`` and ``eta_offsets`` are given,
        if ``n_passes < 1``, if ``step``/``el_step`` are non-positive, if
        ``step`` is passed without ``n_passes``, if ``dwell`` is given for
        more than one pass, or as :func:`plan_source_ces` raises it.
    TargetNotObservableError
        If a pass steps the footprint to a row the source never reaches
        within the search window (propagated from :func:`plan_source_ces`),
        or for a ``start_time`` anchor as :func:`plan_source_ces` raises it.
    ElevationBoundsError
        If a stepped ``el_bore`` falls outside the telescope elevation
        limits.
    AzimuthBoundsError
        Propagated unchanged from a pass's :func:`plan_source_ces` call.
    PointingError
        As :func:`plan_source_ces` raises it, from the ``start_time``
        anchor's derivation of ``el_bore`` or from a pass.

    Warns
    -----
    PointingWarning
        If consecutive pass windows overlap (``el_step`` too small for
        the source's drift), plus any warning propagated from the
        per-pass :func:`plan_source_ces` calls.

    See Also
    --------
    plan_source_ces : Plan a single source-CES pass.

    Examples
    --------
    Three Jupiter-rising passes tiling PrimeCam's centre module:

    >>> from astropy.time import Time
    >>> from fyst_trajectories import get_fyst_site
    >>> from fyst_trajectories.planning import plan_source_ces_passes
    >>> blocks = plan_source_ces_passes(
    ...     body="jupiter",
    ...     footprint="c",
    ...     el_bore=35.0,
    ...     n_passes=3,
    ...     night=Time("2026-03-15T00:00:00", scale="utc"),
    ...     mode="rising",
    ...     site=get_fyst_site(),
    ... )
    >>> [b.trajectory.metadata.pattern_params["pass_eta_offset_deg"] for b in blocks]
    [-0.432..., 0.0, 0.432...]
    """
    base_fp = resolve_footprint(footprint)
    eta_extent = float(base_fp.cover_eta_deg.max() - base_fp.cover_eta_deg.min())

    offsets = _resolve_pass_offsets(
        n_passes=n_passes,
        eta_offsets=eta_offsets,
        step=step,
        footprint_eta_extent=eta_extent,
    )
    n = len(offsets)
    if dwell is not None and n > 1:
        raise ValueError(
            "dwell narrows one pass about its crossing midpoint and is accepted only for a "
            f"single pass; got dwell={dwell} with {n} passes"
        )

    if el_step is None:
        el_step_val = eta_extent
    else:
        el_step_val = float(el_step)
        _require_positive(el_step_val, "el_step")

    if start_time is not None:
        # Resolve the anchor into (central el_bore, window, mode). The anchor
        # applies to the first pass chronologically; the central el_bore the
        # grid below expects is offset from that first pass by half the total
        # el_step span. The per-pass planning then runs unchanged on the
        # classic window path.
        source = _SourceSpec(
            body=body, ra=ra, dec=dec, pm_ra=pm_ra, pm_dec=pm_dec, ref_epoch=ref_epoch
        )
        anchor, coords, resolved_mode, el_at_anchor = _resolve_anchor_prefix(
            start_time=start_time,
            el_bore=el_bore,
            mode=mode,
            night=night,
            window=window,
            source=source,
            site=site,
            atmosphere=atmosphere,
        )
        if el_bore is None:
            drift_sign = 1.0 if resolved_mode == "rising" else -1.0
            # The first pass in time uses the lowest boresight elevation for a
            # rising source (offsets[0]) and the highest for a setting one
            # (offsets[-1]); derive that pass's el_bore, then step out to the
            # central value.
            first_eta = offsets[0] if resolved_mode == "rising" else offsets[-1]
            first_fp = offset_footprint_eta(base_fp, first_eta)
            first_el_bore = _derive_anchored_el_bore(
                anchor=anchor,
                el_at_anchor=el_at_anchor,
                mode=resolved_mode,
                coords=coords,
                fp=first_fp,
                source=source,
                boresight_rot=boresight_rot,
                site=site,
                atmosphere=atmosphere,
                sampling_step_seconds=sampling_step_seconds,
                az_accel=az_accel,
                az_padding=az_padding,
                az_branch=az_branch,
            )
            el_bore = first_el_bore + drift_sign * el_step_val * (n - 1) / 2.0
        horizon = TimeDelta(_DEFAULT_SEARCH_HORIZON_HOURS * 3600.0 * u.s)
        window = (anchor, anchor + horizon)
        mode = resolved_mode
    elif el_bore is None:
        raise ValueError("el_bore is required unless 'start_time' is given")

    # Shared per-pass keyword arguments (everything the passes have in
    # common). ``footprint`` and ``el_bore`` are overridden per pass.
    common = dict(
        body=body,
        ra=ra,
        dec=dec,
        pm_ra=pm_ra,
        pm_dec=pm_dec,
        ref_epoch=ref_epoch,
        boresight_rot=boresight_rot,
        window=window,
        night=night,
        mode=mode,
        site=site,
        atmosphere=atmosphere,
        timestep=timestep,
        sampling_step_seconds=sampling_step_seconds,
        az_accel=az_accel,
        az_padding=az_padding,
        az_branch=az_branch,
        allow_partial=allow_partial,
        v_az=v_az,
        sun_safe=sun_safe,
        az_speed=az_speed,
        az_throw=az_throw,
        dwell=dwell,
    )

    # Pair the lowest coverage row with the lowest boresight elevation so
    # the footprint's sky elevation (``el_bore + eta_offset``) is
    # monotonic across passes; that keeps the source windows sequential
    # (and, at the default ``el_step``, non-overlapping) for both rising
    # and setting sources.
    planned: list[ScanBlock[SourceCESComputedParams]] = []
    for k, eta in enumerate(offsets):
        el_bore_k = el_bore + el_step_val * (k - (n - 1) / 2.0)
        fp_k = offset_footprint_eta(base_fp, eta)
        block = plan_source_ces(footprint=fp_k, el_bore=el_bore_k, **common)
        planned.append(block)

    # Order the blocks by start time (setting sources cross higher
    # elevations first, so coverage order and time order are reversed).
    order = sorted(
        range(n),
        key=lambda i: Time(planned[i].computed_params["t0_iso"], scale="utc").unix,
    )
    tagged: list[ScanBlock[SourceCESComputedParams]] = []
    for pass_index, i in enumerate(order):
        block = planned[i]
        tagged.append(
            _tag_pass_block(
                block,
                pass_index=pass_index,
                n_passes=n,
                eta_offset_deg=offsets[i],
                el_bore_deg=float(block.computed_params["el_bore"]),
            )
        )

    # A consumer sequencing the blocks back to back would double-book the
    # mount if adjacent pass windows overlap (possible when ``el_step`` is
    # reduced below the footprint eta extent), so surface it.
    n_overlaps = sum(
        1
        for a, b in zip(tagged, tagged[1:])
        if Time(b.computed_params["t0_iso"], scale="utc").unix
        < Time(a.computed_params["t1_iso"], scale="utc").unix
    )
    if n_overlaps:
        warnings.warn(
            f"{n_overlaps} adjacent source-CES pass window(s) overlap in time; "
            "increase el_step or schedule the passes with the overlap in mind",
            PointingWarning,
            stacklevel=2,
        )
    return tagged
