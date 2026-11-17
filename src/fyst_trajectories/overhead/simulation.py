"""Timeline simulation: trajectory generation and hitmap accumulation.

Bridges the timeline (sequence of blocks) with fyst-trajectories' planning
functions to generate actual trajectories and accumulate coverage maps.
"""

import dataclasses
import logging
import warnings
from collections.abc import Mapping
from types import MappingProxyType
from typing import TypedDict, cast

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

from .._validation import _require_finite
from ..coordinates import Coordinates
from ..exceptions import PointingWarning
from ..planning import (
    FieldRegion,
    ScanBlock,
    plan_constant_el_scan,
    plan_daisy_scan,
    plan_pong_scan,
    plan_source_ces,
)

# A rebuilt constant-elevation slice is screened against the Sun with the
# check the AltAz planners run on their built blocks.
from ..planning._sun_safety import _check_trajectory_sun_safety
from ..planning.footprints import inflate_footprint, offset_footprint_eta, resolve_footprint
from ..site import Site
from ..trajectory import Trajectory
from .exceptions import BlockNotReconstructableError, ScanParamsSchemaError
from .models import BlockType, ObservingTimeline, TimelineBlock, _offset_from_record
from .schemas import (
    CEScanParams,
    DaisyScanParams,
    PongScanParams,
    TimelineBlockMetadata,
    WindowScanRecord,
    validate_scan_params,
)
from .utils import _search_start_time, _stored_instant

__all__ = [
    "BudgetStats",
    "CalibrationBudget",
    "PatchBudget",
    "accumulate_hitmaps",
    "compute_budget",
    "schedule_to_trajectories",
]

logger = logging.getLogger(__name__)

# For a pass block without ``search_start``: seconds added to each side of
# the recorded pass before re-solving it, since the stored [t0, t1] sits on
# the crossing edges where the planner's cover-vs-arc guard trips. The
# widened window rebuilds the pass only when it holds the source's whole
# crossing of the footprint and of the boresight elevation, which a
# dwell-narrowed pass or an off-centre module can leave outside it. A block
# that records ``search_start`` repeats the planner's own search instead.
_SOURCE_CES_WINDOW_BUFFER_SEC = 300.0

# Kernel-override keys of SourceCESScanParams, forwarded to plan_source_ces
# only when recorded, so the kernel's defaults hold otherwise. The rebuild
# test derives the set from the TypedDict, so a key added there and not here
# fails it.
_SOURCE_CES_OVERRIDE_KEYS = ("az_accel", "az_padding", "v_az", "az_speed", "az_throw", "dwell")

# The pattern settings a pong rebuild uses for a key the patch does not
# record: the spacing plan_pong_scan requires, and that planner's own
# defaults for the rest. The offline scheduler sizes a pong subscan from the
# period of the same pattern, so a block and its rebuilt trajectory hold the
# same whole number of periods.
_PONG_REBUILD_DEFAULTS: Mapping[str, float] = MappingProxyType(
    {"spacing": 0.1, "num_terms": 4, "timestep": 0.1, "angle": 0.0}
)

# The pattern settings a daisy rebuild uses for a key the patch does not
# record: the shape plan_daisy_scan requires, and that planner's own default
# timestep. The offline scheduler bounds a daisy visit by the reach of the
# same pattern, so the trajectory a block rebuilds stays inside the
# elevation limits it was booked against.
_DAISY_REBUILD_DEFAULTS: Mapping[str, float] = MappingProxyType(
    {
        "radius": 1.0,
        "turn_radius": 0.5,
        "avoidance_radius": 0.1,
        "start_acceleration": 0.5,
        "timestep": 0.1,
    }
)

# How far a window block's stop may fall short of its scan's last sample and
# still hold it. ObservingTimeline.validate allows 0.01 s between blocks; a
# window block that stops within that of the last sample (which sits exactly
# at the scan's duration) keeps it. This compares a block's stop with its
# scan's end, a question about the block; which side of an edge a sample
# falls on is _EDGE_TOLERANCE_SEC's.
_BLOCK_TIME_TOLERANCE_SEC = 0.01

# How far before a block's stored edge a sample still counts as on it. The
# rebuild reads a block's start and stop as a file stores them, each rounded
# to the millisecond, so each stored edge lies within 0.5 ms of the instant
# the scheduler placed. A sample the scheduler placed on the start, measured
# from the stored start, therefore lies within 0.5 ms of it; one placed on
# the stop, measured from a trajectory planned from the stored start, within
# 0.5 ms (the start's rounding) plus 0.5 ms (the stop's) of the stored stop.
# Twice that 1 ms covers the float noise of the instants, and it is far below
# any sample interval (the Go TCS accepts none under 50 ms). A sample belongs
# to a block when it lies at or after the stored start less this, and before
# the stored stop less this: a sample on the start belongs to the block, one
# on the stop to the block after it, unless it is the scan's own last sample.
_EDGE_TOLERANCE_SEC = 2e-3

# The four keys a window block carries when its scan was planned with a field.
_FIELD_KEYS = ("ra_center", "dec_center", "width", "height")

# Keys the source-CES rebuild reads unconditionally.
_SOURCE_CES_REQUIRED_KEYS = (
    "body",
    "footprint",
    "el_bore",
    "mode",
    "boresight_rot",
    "timestep",
    "eta_offset_deg",
)


def schedule_to_trajectories(
    timeline: ObservingTimeline,
    science_only: bool = True,
) -> list[tuple[TimelineBlock, ScanBlock]]:
    """Generate trajectories for timeline blocks.

    Reconstructs planning parameters from each block's metadata and calls
    the appropriate ``plan_*`` function, returning one
    ``(TimelineBlock, ScanBlock)`` pair per reconstructed block. A science
    block that records ``metadata["detector_offset"]`` is planned with that
    offset as recorded, whatever the Prime-Cam bay table holds when it is
    rebuilt, and a constant-elevation one at its recorded
    ``metadata["detector_elevation"]``, the detector's elevation.

    Parameters
    ----------
    timeline : ObservingTimeline
        Input timeline.
    science_only : bool
        Which blocks to reconstruct.

        * ``True`` (default): science blocks only.
        * ``False``: science blocks **plus** calibration blocks whose
          metadata carries a ``scan_params`` dict, the source-CES
          planet-calibration passes of
          :func:`~fyst_trajectories.overhead.plan_calibration_night` and of
          ``CalibrationPolicy.planet_cal_scan``. Each is rebuilt with
          :func:`~fyst_trajectories.planning.plan_source_ces` from its
          recorded parameters by repeating the planner's search from the
          block's ``metadata["search_start"]``, so the rebuilt pass is the
          planned one, sample for sample; a rebuilt pass that does not
          overlap its block is refused, as one rebuilt from a later block's
          ``search_start``, or from one a day off, is. A pass block without
          ``search_start`` is re-solved instead in a window of its recorded
          pass widened by 300 s on each side. That lands within about 0.1 s
          of the planned start, but skips the pass when the window misses the
          source's crossing of the boresight elevation (an off-centre module)
          or part of its crossing of the footprint (a dwell that cuts about
          300 s or more from it).

        Calibration blocks with **no** ``scan_params`` (parked planet cals,
        retunes, pointing, focus, skydip) are placeholders with no
        trajectory to rebuild and are skipped silently, not logged as
        failures. Slew and idle blocks are always skipped.

    Returns
    -------
    list of (TimelineBlock, ScanBlock)
        Pairs of timeline blocks and their generated trajectories. A
        science trajectory covers only its own block's
        ``[t_start, t_stop)`` window, plus the rebuilt scan's own last
        sample where that lies on ``t_stop``, as it does for a pong block
        of whole periods (a block that records a window scan follows the
        rule below): the subscans of one
        constant-elevation visit are consecutive slices of a single
        crossing solve (anchored at the shared ``metadata["t0_scan"]``),
        not one full pass each, so summing samples over the pairs counts
        each visit once. ``ScanBlock.duration`` is the
        slice; ``computed_params`` and ``summary`` still describe the
        full solved pass. Calibration passes are one block per pass
        and are returned whole.

        A block that records a window scan (``metadata["window_scan"]``,
        written by :meth:`TimelineBlock.from_window_scan`) is rebuilt with
        the window form of
        :func:`~fyst_trajectories.planning.plan_constant_el_scan` from the
        scan's start: ``metadata["scan_start"]`` on a block that holds a
        later stretch of the scan, else the block's own start (after an
        ECSV round trip, the file's millisecond form of either). The block
        gets the scan's samples in ``[t_start, t_stop)``, like any other
        science block, and also the scan's last sample, which sits at the
        scan's end, when it stops no earlier than that sample less 0.01 s.
        So the blocks the scheduler cuts one scan into rebuild to
        consecutive slices of it, and a block as long as its scan to the
        whole scan. :meth:`TimelineBlock.from_window_scan` writes only a
        block that rebuilds to the samples of the scan it holds, sample for
        sample.

    Warns
    -----
    PointingWarning
        The planners' advisories for the rebuilt blocks. A
        constant-elevation block's Sun advisory describes the samples the
        block returns: they are screened at every sample against the
        site's exclusion radius, and the planner's own Sun checks (of the
        field centre at the visit's start and of the whole crossing pass,
        or of the whole window scan) are not raised.

    Notes
    -----
    Blocks that attempt reconstruction but fail are logged at ``WARNING``
    and skipped, so one bad block does not abort the whole timeline; the
    line names a pass block's ``search_start``. This subpackage's own
    refusals are
    :class:`~fyst_trajectories.overhead.ScanParamsSchemaError` (recorded
    metadata that does not match the rebuild's schema) and
    :class:`~fyst_trajectories.overhead.BlockNotReconstructableError` (a
    block whose window no longer overlaps the re-solved scan); a plain
    ``ValueError``, ``KeyError`` or ``TypeError`` from the planner, for
    example a source no longer reachable at the recorded geometry, is
    caught the same way.
    """
    site = timeline.site
    blocks = timeline.science_blocks if science_only else timeline.blocks
    results = []

    for sblock in blocks:
        block_type = sblock.block_type
        if block_type == BlockType.SCIENCE:
            reconstructable = True
        elif block_type == BlockType.CALIBRATION:
            # Only source-CES planet-cal passes record replayable scan_params.
            # Parked cals, retunes, pointing/focus/skydip carry none - skip
            # them silently (placeholders, not reconstruction failures).
            reconstructable = "scan_params" in sblock.metadata
        else:
            # Slew / idle blocks carry no trajectory.
            reconstructable = False
        if not reconstructable:
            continue

        try:
            scan_block = _generate_trajectory_for_block(sblock, site)
            results.append((sblock, scan_block))
        except (ValueError, KeyError, TypeError) as exc:
            # Best-effort: the subpackage's own refusals are ValueErrors, and
            # malformed metadata surfaces as a plain ValueError (a zero
            # velocity) or as KeyError/TypeError (a truncated dict), so each
            # skips the block with a log line. A pass block's line names the
            # search_start its rebuild repeated.
            meta = sblock.metadata
            record = f" (search_start={meta['search_start']!r})" if "search_start" in meta else ""
            logger.warning(
                "Failed to generate trajectory for block '%s' at %s%s: %s",
                sblock.patch_name,
                sblock.t_start.iso,
                record,
                exc,
            )

    return results


def _generate_trajectory_for_block(
    sblock: TimelineBlock,
    site: Site,
) -> ScanBlock:
    """Generate a trajectory for a single timeline block.

    Parameters
    ----------
    sblock : TimelineBlock
        Timeline block with metadata containing scan parameters.
    site : Site
        Observatory site.

    Returns
    -------
    ScanBlock
        Generated scan block. Science trajectories are sliced to the
        block's own ``[t_start, t_stop)`` window by
        :func:`_slice_to_block_window`, the block's times read as a
        timeline file stores them (to the millisecond), so a block rebuilds
        to the same arrays from memory and from a file; ``duration`` and
        ``trajectory.start_time`` describe the slice while
        ``computed_params`` and ``summary`` describe the full solved
        pass. A window block holds the samples of its scan in that window
        and, when its stop is no earlier than the scan's last sample less
        ``_BLOCK_TIME_TOLERANCE_SEC``, that last sample too
        (:func:`_window_block_slice`).

    Warns
    -----
    PointingWarning
        The planner's advisories for the rebuilt block. A
        constant-elevation block, of either form, is screened against the
        Sun over its own samples, in place of the planner's own Sun checks,
        which can name an instant outside the block.

    Raises
    ------
    ScanParamsSchemaError
        If ``sblock.metadata`` is missing any of the required geometry
        keys (``ra_center``, ``dec_center``, ``width``, ``height``,
        ``velocity``), or names a scan type that cannot be rebuilt, or
        ``scan_params`` carries a key not declared for ``sblock.scan_type``,
        or a calibration pass's ``search_start`` is not two finite numbers,
        or a window block's record does not match
        :class:`~fyst_trajectories.overhead.WindowScanRecord` (see
        :func:`_rebuild_window_scan`), or the recorded ``detector_offset``
        or ``detector_elevation`` does not match
        :class:`~fyst_trajectories.overhead.ScienceBlockMetadata` (see
        :func:`_recorded_detector_elevation`).
    BlockNotReconstructableError
        If the re-solved scan does not overlap the block's time window:
        fewer than two of its samples fall inside ``[t_start, t_stop)``.

    Notes
    -----
    Calibration blocks are dispatched to :func:`_rebuild_calibration_pass`,
    which rebuilds the recorded source-CES planet-calibration pass; the pass
    is returned whole once it is found to overlap its block. A science block
    that records ``window_scan`` is dispatched to
    :func:`_rebuild_window_scan`, which plans it again with the window form
    of :func:`~fyst_trajectories.planning.plan_constant_el_scan`. Other
    science blocks take the ``ra_center``/``dec_center`` geometry path
    below, planned with the recorded ``detector_offset`` when there is one,
    and a constant-elevation block then at its ``detector_elevation``.
    """
    meta = sblock.metadata

    if sblock.block_type == BlockType.CALIBRATION:
        return _rebuild_calibration_pass(sblock, site)

    # The block's times as a file stores them, so a block held in memory and
    # the same block read back from a file are rebuilt from the same instants.
    t_start, t_stop = _stored_window(sblock)

    if "window_scan" in meta:
        return _rebuild_window_scan(sblock, site)

    required = ("ra_center", "dec_center", "width", "height", "velocity")
    missing = [k for k in required if k not in meta]
    if missing:
        raise ScanParamsSchemaError(
            f"TimelineBlock metadata missing required keys {missing}. "
            f"Ensure the timeline was generated with per-block scan geometry, "
            f"or provide the metadata explicitly."
        )
    ra_center = meta["ra_center"]
    dec_center = meta["dec_center"]
    width = meta["width"]
    height = meta["height"]
    velocity = meta["velocity"]
    scan_params = meta.get("scan_params", {})

    # The ``cast(...)`` below is a no-op at runtime; without this check a
    # mistyped key (``"radiu"`` on a daisy) would fall through to the
    # ``.get()`` default silently.
    try:
        validate_scan_params(scan_params, sblock.scan_type)
    except KeyError as exc:
        raise ScanParamsSchemaError(exc.args[0]) from None

    field = FieldRegion(
        ra_center=ra_center,
        dec_center=dec_center,
        width=width,
        height=height,
    )
    offset = _offset_from_record(meta.get("detector_offset"))
    detector_el = _recorded_detector_elevation(sblock, offset is not None)

    if sblock.scan_type == "constant_el":
        ce_params = cast(CEScanParams, scan_params)
        # CE subscans are slices of one crossing, re-solved from the visit
        # anchor (metadata["t0_scan"]) the corridor gate solved it from; a
        # subscan's own t_start may lie past the opening, where the forward
        # search cannot find the pass. A block without the key falls
        # back to t_start and rebuilds only when that precedes the opening.
        anchor = Time(meta["t0_scan"], scale="utc") if "t0_scan" in meta else t_start
        # A sidereal window fixes the timing and the range, so the planner
        # refuses ``rising`` beside it; the block's rising flag is not
        # forwarded then.
        lsa_window = ce_params.get("lsa_window")
        # Forward only what the patch recorded; the planner's defaults hold
        # for the rest.
        ce_overrides = {
            key: ce_params[key]
            for key in ("az_accel", "timestep", "az_padding")
            if key in ce_params
        }
        # The planner's Sun checks describe the field centre at the anchor
        # and the whole crossing pass, while this block is one slice of
        # that pass; they are set aside and the slice is screened over its
        # own window instead, so the advisory names an instant inside it.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="EXCLUSION ZONE: ", category=PointingWarning)
            scan_block = plan_constant_el_scan(
                field=field,
                elevation=sblock.elevation if detector_el is None else detector_el,
                velocity=velocity,
                site=site,
                start_time=anchor,
                rising=None if lsa_window is not None else sblock.rising,
                lsa_window=lsa_window,
                detector_offset=offset,
                **ce_overrides,
            )
        scan_block = _slice_to_block_window(scan_block, sblock)
        _check_trajectory_sun_safety(
            site=site,
            trajectory=scan_block.trajectory,
            scan_label=(
                f"constant-elevation scan at RA={field.ra_center:.3f}, Dec={field.dec_center:.3f}"
            ),
        )
        return scan_block
    elif sblock.scan_type == "pong":
        pong_params = cast(PongScanParams, scan_params)
        # The pattern the scheduler sized the block by: the recorded keys
        # over _PONG_REBUILD_DEFAULTS. A block written by the scheduler also
        # records the whole number of periods it holds as ``n_cycles``.
        scan_block = plan_pong_scan(
            field=field,
            velocity=velocity,
            site=site,
            start_time=t_start,
            detector_offset=offset,
            **{**_PONG_REBUILD_DEFAULTS, **pong_params},
        )
    elif sblock.scan_type == "daisy":
        daisy_params = cast(DaisyScanParams, scan_params)
        # The pattern the scheduler bounded the block by: the recorded keys
        # over _DAISY_REBUILD_DEFAULTS.
        scan_block = plan_daisy_scan(
            ra=ra_center,
            dec=dec_center,
            velocity=velocity,
            site=site,
            start_time=t_start,
            duration=float((t_stop - t_start).sec),
            detector_offset=offset,
            **{**_DAISY_REBUILD_DEFAULTS, **daisy_params},
        )
    else:
        raise ScanParamsSchemaError(f"Unknown scan type: {sblock.scan_type}")

    return _slice_to_block_window(scan_block, sblock)


def _recorded_detector_elevation(sblock: TimelineBlock, has_offset: bool) -> float | None:
    """Return the detector elevation a science block records, or ``None`` when it records none.

    Raises
    ------
    ScanParamsSchemaError
        If a constant-elevation block records ``detector_offset`` without
        ``detector_elevation`` or the other way round, ``detector_elevation``
        is not a finite real number, or a block of another scan type records
        ``detector_elevation``.
    """
    meta = sblock.metadata
    if "detector_elevation" not in meta:
        if has_offset and sblock.scan_type == "constant_el":
            raise ScanParamsSchemaError(
                "a constant-elevation block that records detector_offset must record "
                "detector_elevation, the elevation the detector scans at"
            )
        return None
    if sblock.scan_type != "constant_el":
        raise ScanParamsSchemaError(
            "detector_elevation is recorded only on a constant-elevation block, "
            f"not on a {sblock.scan_type!r} block"
        )
    if not has_offset:
        raise ScanParamsSchemaError(
            "detector_elevation is recorded only beside detector_offset, and this block "
            "records no offset"
        )
    value = meta["detector_elevation"]
    try:
        return _require_finite(value, "detector_elevation")
    except (TypeError, ValueError):
        raise ScanParamsSchemaError(
            f"detector_elevation must be a finite real number, got {value!r}"
        ) from None


def _rebuild_calibration_pass(sblock: TimelineBlock, site: Site) -> ScanBlock:
    """Rebuild a planet-calibration pass whole and require it to overlap its block.

    Raises
    ------
    ScanParamsSchemaError
        As :func:`_generate_source_ces_trajectory` raises it.
    BlockNotReconstructableError
        If fewer than two samples of the rebuilt pass fall inside the block's
        ``[t_start, t_stop)`` window.
    """
    meta = sblock.metadata
    t_start, t_stop = _stored_window(sblock)
    # The fallback window serves only a pass block with no recorded search
    # start whose dict holds no absolute window: it re-solves around its
    # own bounds, from t0_scan to t_stop.
    pass_start = Time(meta["t0_scan"], scale="utc") if "t0_scan" in meta else t_start
    scan_block = _generate_source_ces_trajectory(meta, site, fallback_window=(pass_start, t_stop))
    # The pass must overlap its block: a search_start that is not the
    # block's own repeats a search that can find another crossing (the
    # next day's, for a record minutes late).
    traj = scan_block.trajectory
    inside = int(np.count_nonzero(_block_window_mask(traj, sblock)))
    if inside < 2:
        source = (
            f"search_start {meta['search_start']!r}"
            if "search_start" in meta
            else "its recorded pass"
        )
        raise BlockNotReconstructableError(
            f"the pass rebuilt from {source} runs [{traj.start_time.isot} + "
            f"{float(traj.times[0]):.1f}s .. {float(traj.times[-1]):.1f}s] and has {inside} "
            f"sample(s) in its block window [{sblock.t_start.isot}, {sblock.t_stop.isot})"
        )
    return scan_block


def _rebuild_window_scan(sblock: TimelineBlock, site: Site) -> ScanBlock:
    """Rebuild a science block that records a constant-elevation window scan.

    Plans the recorded scan again with the window form of
    :func:`~fyst_trajectories.planning.plan_constant_el_scan` from the
    scan's start: ``metadata["scan_start"]`` when the block records it (a
    block that is a later slice of the scan), else the block's own start,
    read as a file stores it. The block gets the samples
    :func:`_window_block_slice` assigns it. The planner's own Sun advisory
    is set aside and the returned samples are screened instead, so the
    advisory names an instant inside the block.

    Raises
    ------
    ScanParamsSchemaError
        If the block's ``scan_type`` is not ``"constant_el"``; the record is
        not a mapping with exactly the keys of
        :class:`~fyst_trajectories.overhead.WindowScanRecord`; the block
        records no ``velocity``; its ``scan_params`` is not empty; it carries
        some but not all of the four field keys; the recorded
        ``detector_offset`` is neither ``None`` nor exactly the four fields
        of an :class:`~fyst_trajectories.offsets.InstrumentOffset`; or a
        recorded ``scan_start`` is not an ISO time string.
    BlockNotReconstructableError
        If fewer than two samples of the scan fall to the block.
    """
    meta = sblock.metadata
    record = meta["window_scan"]  # type: ignore[typeddict-item]
    if sblock.scan_type != "constant_el":
        raise ScanParamsSchemaError(
            f"a block recording window_scan must have scan_type 'constant_el', "
            f"got {sblock.scan_type!r}"
        )
    if not isinstance(record, Mapping):
        raise ScanParamsSchemaError(
            f"window_scan must be a mapping of {sorted(WindowScanRecord.__required_keys__)}, "
            f"got {type(record).__name__}"
        )
    missing = sorted(WindowScanRecord.__required_keys__ - set(record))
    unknown = sorted(set(record) - WindowScanRecord.__required_keys__)
    if missing or unknown:
        raise ScanParamsSchemaError(
            f"window_scan has missing keys {missing} and unknown keys {unknown}; "
            f"expected exactly {sorted(WindowScanRecord.__required_keys__)}"
        )
    if "velocity" not in meta:
        raise ScanParamsSchemaError(
            "a window block must record its azimuth rate as metadata key 'velocity'"
        )
    if meta.get("scan_params"):
        raise ScanParamsSchemaError(
            "a window block's scan_params must be empty: window_scan holds the scan's inputs, "
            f"got scan_params keys {sorted(meta['scan_params'])}"
        )
    present = [key for key in _FIELD_KEYS if key in meta]
    if present and len(present) != len(_FIELD_KEYS):
        absent = [key for key in _FIELD_KEYS if key not in meta]
        raise ScanParamsSchemaError(
            f"a window block records all four field keys or none; missing {absent}"
        )
    offset = _offset_from_record(record["detector_offset"])
    field = (
        FieldRegion(
            ra_center=meta["ra_center"],
            dec_center=meta["dec_center"],
            width=meta["width"],
            height=meta["height"],
        )
        if present
        else None
    )

    scan_start = _recorded_scan_start(sblock)
    # The planner's Sun check is set aside, as on the field form above, and
    # the samples the block returns are screened once below.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="EXCLUSION ZONE: ", category=PointingWarning)
        scan_block = plan_constant_el_scan(
            field,
            az_window=record["az_window"],
            elevation=record["elevation"],
            velocity=meta["velocity"],
            duration=record["duration"],
            site=site,
            start_time=scan_start,
            az_accel=record["az_accel"],
            timestep=record["timestep"],
            detector_offset=offset,
        )
    scan_block = _window_block_slice(scan_block, sblock)
    cp = scan_block.computed_params
    _check_trajectory_sun_safety(
        site=site,
        trajectory=scan_block.trajectory,
        scan_label=(
            f"constant-elevation window scan over az [{cp['az_start']:.2f}, "
            f"{cp['az_stop']:.2f}] at el={record['elevation']:.2f}"
        ),
    )
    return scan_block


def _window_scan_replays(sblock: TimelineBlock, scan_block: ScanBlock, site: Site) -> bool:
    """Whether rebuilding ``sblock`` returns ``scan_block``'s trajectory, sample for sample.

    A probe: the scan's advisories were raised when it was planned, so the
    rebuild's are not raised again. Exceptions from the rebuild propagate.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PointingWarning)
        rebuilt = _rebuild_window_scan(sblock, site)
    return all(
        np.array_equal(getattr(rebuilt.trajectory, name), getattr(scan_block.trajectory, name))
        for name in ("times", "az", "el", "az_vel", "el_vel", "scan_flag")
    )


def _stored_window(sblock: TimelineBlock) -> tuple[Time, Time]:
    """Return a block's start and stop as a timeline file stores them."""
    return _stored_instant(sblock.t_start), _stored_instant(sblock.t_stop)


def _recorded_scan_start(sblock: TimelineBlock) -> Time:
    """Return the start of the window scan a block holds a slice of.

    It is ``metadata["scan_start"]`` when the block records it, the start of
    a scan that began before the block, else the block's own start; either
    is read as a file stores it.

    Raises
    ------
    ScanParamsSchemaError
        If a recorded ``scan_start`` is not an ISO time string.
    """
    recorded = sblock.metadata.get("scan_start")
    if recorded is None:
        return _stored_instant(sblock.t_start)
    if not isinstance(recorded, str):
        raise ScanParamsSchemaError(
            f"scan_start must be an ISO time string, got {type(recorded).__name__}"
        )
    try:
        return _stored_instant(Time(recorded, format="isot", scale="utc"))
    except ValueError as err:
        raise ScanParamsSchemaError(
            f"scan_start must be an ISO time string, got {recorded!r}"
        ) from err


def _window_block_slice(scan_block: ScanBlock, sblock: TimelineBlock) -> ScanBlock:
    """Return the samples of a window scan that belong to one of its blocks.

    A window block holds the samples of its scan in its half-open
    ``[t_start, t_stop)`` as a file stores them (:func:`_block_window_mask`)
    and, when its stop is no earlier than the scan's last sample less
    ``_BLOCK_TIME_TOLERANCE_SEC``, that last sample as well, which no later
    block holds. So a block as long as its scan, or the last of several
    blocks that each hold a slice of one scan, keeps the scan's end, and a
    block that ends earlier keeps what lies inside it. A block that holds
    every sample gets ``scan_block`` back unchanged; any other gets the
    slice from :func:`_slice_to_block_window`.

    Raises
    ------
    BlockNotReconstructableError
        If fewer than two samples fall to the block.
    """
    traj = scan_block.trajectory
    if traj.start_time is None:
        return scan_block
    mask = _block_window_mask(traj, sblock)
    t_start, t_stop = _stored_window(sblock)
    last = float(traj.times[-1])
    if (t_stop - traj.start_time).sec >= last - _BLOCK_TIME_TOLERANCE_SEC:
        mask[-1] = last >= (t_start - traj.start_time).sec - _EDGE_TOLERANCE_SEC
    return _slice_to_block_window(scan_block, sblock, mask=mask)


def _block_window_mask(traj: Trajectory, sblock: TimelineBlock) -> np.ndarray:
    """Return which samples of ``traj`` fall inside the block's half-open ``[t_start, t_stop)``.

    The edges are the block's stored ones, each moved back by
    ``_EDGE_TOLERANCE_SEC``, so a sample on the start belongs to the block
    and a sample on the stop does not, whichever way the file rounded them:
    it is the first sample of the next block of the same scan. The scan's
    own last sample on the stop is the exception, since no later block
    holds it: a pong of whole periods stops on its last sample.
    """
    t_start, t_stop = _stored_window(sblock)
    times = traj.times
    rel_start = (t_start - traj.start_time).sec - _EDGE_TOLERANCE_SEC
    rel_stop = (t_stop - traj.start_time).sec
    inside = (times >= rel_start) & (times < rel_stop - _EDGE_TOLERANCE_SEC)
    if times.size and abs(times[-1] - rel_stop) <= _EDGE_TOLERANCE_SEC:
        inside[-1] = bool(times[-1] >= rel_start)
    return inside


def _slice_to_block_window(
    scan_block: ScanBlock, sblock: TimelineBlock, *, mask: np.ndarray | None = None
) -> ScanBlock:
    """Slice a rebuilt science trajectory to its block's own time window.

    A CE subscan re-solves its visit's whole crossing pass, so it comes
    back longer than the block it was rebuilt for; unsliced, a consumer
    summing samples would multiply-count the visit. Only the samples
    inside ``[t_start, t_stop)`` are kept, the edges as a file stores them,
    and the trajectory's own last sample when it lies on the stop
    (:func:`_block_window_mask`); the half-open window keeps back-to-back
    subscans from double-counting their shared boundary sample, which
    belongs to the later block. ``times`` are re-zeroed and
    ``start_time`` advanced so the slice keeps the ``times[0] == 0``
    convention, and
    ``retune_events`` are dropped (planners emit none; events from a
    pre-slice origin would be misplaced on the slice). The block's
    ``computed_params`` and ``summary`` still describe the full
    solved pass.

    A pong block the offline scheduler writes records the whole number
    of pattern periods it holds as ``n_cycles``, so its rebuild covers
    exactly the block, its last sample, on the block's stop, included. A
    pong block that records no ``n_cycles`` rebuilds one period, sliced to
    its window: a shorter block keeps the head of the pattern, a longer
    one only that period.

    A trajectory already inside the window (the daisy branch, which
    plans with the block duration) is returned unchanged.

    ``mask``, when given, names the samples to keep in place of the
    block window's own (:func:`_window_block_slice` passes one).

    Raises
    ------
    BlockNotReconstructableError
        If fewer than two trajectory samples fall inside the block
        window (the re-solved scan no longer overlaps this block).
        ``schedule_to_trajectories`` logs and skips such blocks like
        any other reconstruction failure.
    """
    traj = scan_block.trajectory
    if traj.start_time is None:
        return scan_block
    t = traj.times
    if mask is None:
        mask = _block_window_mask(traj, sblock)
    if bool(mask.all()):
        return scan_block
    idx = np.nonzero(mask)[0]
    if idx.size < 2:
        raise BlockNotReconstructableError(
            f"block window [{sblock.t_start.isot}, {sblock.t_stop.isot}) contains "
            f"{idx.size} trajectory sample(s); the re-solved scan "
            f"[{traj.start_time.isot} + {float(t[0]):.1f}s .. {float(t[-1]):.1f}s] "
            f"no longer overlaps this block"
        )
    new_traj = dataclasses.replace(
        traj,
        times=t[idx] - t[idx[0]],
        az=traj.az[idx],
        el=traj.el[idx],
        az_vel=traj.az_vel[idx],
        el_vel=traj.el_vel[idx],
        scan_flag=None if traj.scan_flag is None else traj.scan_flag[idx],
        start_time=traj.start_time + TimeDelta(float(t[idx[0]]), format="sec"),
        retune_events=(),
    )
    return dataclasses.replace(
        scan_block,
        trajectory=new_traj,
        duration=float(new_traj.times[-1]),
    )


def _generate_source_ces_trajectory(
    meta: TimelineBlockMetadata,
    site: Site,
    *,
    fallback_window: tuple[Time, Time] | None = None,
) -> ScanBlock:
    """Rebuild a source-CES planet-calibration pass from its recorded params.

    Consumes the :class:`~fyst_trajectories.overhead.SourceCESScanParams`
    stored under ``meta["scan_params"]`` when a planet calibration was planned
    as a source-CES pass sequence (``CalibrationPolicy.planet_cal_scan`` or a
    calibration night) and replays it through
    :func:`~fyst_trajectories.planning.plan_source_ces`.

    When the block records ``meta["search_start"]``, the UTC instant the
    planner's search for the pass began at, the rebuild repeats that search: the
    planner's anchored form with the recorded ``el_bore`` and ``mode``
    searches the 24 h from ``search_start``, the window
    :func:`~fyst_trajectories.planning.plan_source_ces_passes` solved every
    pass of the visit in. The same window gives the same source samples, the
    same arc and the same azimuth unwrap, so the rebuilt
    :class:`~fyst_trajectories.planning.ScanBlock` is the planned pass, sample
    for sample. A block written without it re-solves inside the recorded pass
    widened by ``_SOURCE_CES_WINDOW_BUFFER_SEC`` on each side, which lands
    within about 0.1 s of the planned start when it rebuilds the pass at all.

    Parameters
    ----------
    meta : TimelineBlockMetadata
        Calibration-block metadata carrying ``scan_params``.
    site : Site
        Observatory site.
    fallback_window : tuple of Time, optional
        For a block written without ``search_start`` whose ``scan_params``
        carry no ``window`` (a relative dispatch dict): the pass bounds,
        normally the block's ``t0_scan`` and ``t_stop``. The re-solve
        searches from ``_SOURCE_CES_WINDOW_BUFFER_SEC`` before the first to
        as long after the second. Unused when the block records
        ``search_start``.

    Returns
    -------
    ScanBlock
        The rebuilt source-CES pass.

    Raises
    ------
    ScanParamsSchemaError
        If ``scan_params`` carries a key not declared for ``source_ces`` or
        lacks one the rebuild reads, ``search_start`` is not two finite
        numbers, or ``footprint`` names no Prime-Cam module.
    BlockNotReconstructableError
        If the block records no ``search_start``, ``scan_params`` carries no
        ``window`` and no ``fallback_window`` was given.
    ValueError, TypeError
        Propagated from :func:`~fyst_trajectories.planning.plan_source_ces` when the
        recorded geometry can no longer be solved.
    """
    params = meta["scan_params"]
    try:
        validate_scan_params(params, "source_ces")
    except KeyError as exc:
        raise ScanParamsSchemaError(exc.args[0]) from None
    missing = [key for key in _SOURCE_CES_REQUIRED_KEYS if key not in params]
    if missing:
        raise ScanParamsSchemaError(f"source_ces scan_params missing required keys {missing}")

    search: dict[str, Time | tuple[Time, Time]]
    if "search_start" in meta:
        # Repeat the planner's search. With el_bore and mode given, the
        # anchored form searches (start, start + 24 h), the window the
        # planner passed to every pass, built by the same expression.
        search = {"start_time": _search_start_time(meta["search_start"])}
    else:
        if "window" in params:
            t0 = Time(params["window"][0], scale="utc")
            t1 = Time(params["window"][1], scale="utc")
        elif fallback_window is not None:
            t0, t1 = fallback_window
        else:
            raise BlockNotReconstructableError(
                "source_ces scan_params carry no window, the block records no search_start, "
                "and no fallback_window was given"
            )
        # Widen the recorded pass before re-solving; see
        # _SOURCE_CES_WINDOW_BUFFER_SEC.
        buffer = TimeDelta(_SOURCE_CES_WINDOW_BUFFER_SEC, format="sec")
        search = {"window": (t0 - buffer, t1 + buffer)}

    # The margin and the eta offset are load-bearing geometry, not
    # provenance: the margin widens the crossing the pass was solved on, and
    # each pass drags the source through a different focal-plane row, so the
    # rebuild inflates then shifts the base footprint as the planner did.
    try:
        base_footprint = resolve_footprint(params["footprint"])
    except KeyError as exc:
        raise ScanParamsSchemaError(exc.args[0]) from None
    base_footprint = inflate_footprint(base_footprint, params.get("footprint_margin", 0.0))
    pass_footprint = offset_footprint_eta(base_footprint, params["eta_offset_deg"])

    # Kernel overrides ride along only when they were recorded; see
    # _SOURCE_CES_OVERRIDE_KEYS.
    overrides = {key: params[key] for key in _SOURCE_CES_OVERRIDE_KEYS if key in params}

    return plan_source_ces(
        body=params["body"],
        footprint=pass_footprint,
        el_bore=params["el_bore"],
        boresight_rot=params["boresight_rot"],
        timestep=params["timestep"],
        mode=params["mode"],
        site=site,
        **search,
        **overrides,
    )


def accumulate_hitmaps(
    trajectory_pairs: list[tuple[TimelineBlock, ScanBlock]],
    site: Site,
    nside: int = 256,
) -> np.ndarray:
    """Accumulate a boresight-level HEALPix hitmap from trajectories.

    For each trajectory, converts every 10th sample from az/el to
    RA/Dec, bins the result into HEALPix pixels, and sums across all
    trajectories. Flagged trajectories contribute science samples only.

    This is a simplified boresight-level hitmap (boresight samples only,
    not per detector). For detector-level hitmaps, project each detector
    offset through
    :func:`~fyst_trajectories.offsets.boresight_to_detector` before
    binning, or feed the schedule to a dedicated mapping simulator. A
    block that records a ``detector_offset`` rebuilds to the boresight that
    holds that detector on the field, so its hits here are the
    boresight's; the detector's are its recorded offset projected the same
    way.

    Parameters
    ----------
    trajectory_pairs : list of (TimelineBlock, ScanBlock)
        Output of ``schedule_to_trajectories()``.
    site : Site
        Observatory site.
    nside : int
        HEALPix resolution parameter (default: 256).

    Returns
    -------
    numpy.ndarray
        HEALPix map of hit counts per pixel.

    Raises
    ------
    ImportError
        If ``healpy`` is not installed.

    Notes
    -----
    The az/el to RA/Dec inverse is taken in vacuum, whatever atmosphere
    the trajectory was generated under. A trajectory built with
    ``AtmosphericConditions.for_fyst()`` therefore bins slightly off:
    arcseconds near the zenith, about 1.4 arcmin at the 20 degree
    elevation floor, more below 10 degrees. That is sub-pixel down to
    the floor at ``nside=512`` (7 arcmin pixels); a low-elevation,
    high-resolution map is where it shows.
    """
    try:
        import healpy as hp  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError(
            "healpy is required for hitmap accumulation. "
            "Install it with: pip install 'fyst-trajectories[overhead]'"
        ) from None

    npix = hp.nside2npix(nside)
    hitmap = np.zeros(npix, dtype=np.float64)
    # Vacuum inverse; see Notes for the bias this leaves on a trajectory
    # built with refraction. Closing it would need the trajectory's own
    # atmosphere, which Trajectory does not carry.
    coords = Coordinates(site)

    for sblock, scan_block in trajectory_pairs:
        traj = scan_block.trajectory
        if traj.start_time is None:
            logger.warning(
                "Trajectory for '%s' has no start_time, skipping hitmap",
                sblock.patch_name,
            )
            continue

        stride = 10
        indices = np.arange(0, len(traj.times), stride)
        if traj.scan_flag is not None:
            mask = traj.science_mask[indices]
            indices = indices[mask]

        if len(indices) == 0:
            continue

        times = traj.start_time + TimeDelta((traj.times[indices] - traj.times[0]) * u.s)
        ra_arr, dec_arr = coords.altaz_to_radec(traj.az[indices], traj.el[indices], times)
        theta = np.radians(90.0 - dec_arr)
        phi = np.radians(ra_arr)
        pixels = hp.ang2pix(nside, theta, phi)
        np.add.at(hitmap, pixels, 1.0)

    return hitmap


class PatchBudget(TypedDict):
    """One patch's share of a timeline's science time.

    Attributes
    ----------
    science_time : float
        Seconds of science on this patch.
    n_scans : int
        Science blocks emitted for it, counting every subscan.
    n_unique_scans : int
        Distinct ``scan_index`` values among those blocks, so a visit cut
        into subscans counts once.
    """

    science_time: float
    n_scans: int
    n_unique_scans: int


class CalibrationBudget(TypedDict):
    """One calibration type's share of a timeline.

    Attributes
    ----------
    count : int
        Blocks of this calibration type.
    total_time : float
        Seconds spent on them.
    """

    count: int
    total_time: float


class BudgetStats(TypedDict):
    """The summary :func:`compute_budget` returns.

    Attributes
    ----------
    total_time : float
        Seconds the timeline spans.
    science_time, calibration_time, slew_time, idle_time : float
        Seconds in each block type; they tile ``total_time``.
    efficiency : float
        ``science_time / total_time``.
    n_science_scans : int
        Science blocks in the timeline.
    n_calibration_blocks : int
        Calibration blocks in the timeline.
    per_patch : dict of str to PatchBudget
        Science time and block counts, keyed by patch name.
    calibration_breakdown : dict of str to CalibrationBudget
        Block counts and time, keyed by calibration type.
    """

    total_time: float
    science_time: float
    calibration_time: float
    slew_time: float
    idle_time: float
    efficiency: float
    n_science_scans: int
    n_calibration_blocks: int
    per_patch: dict[str, PatchBudget]
    calibration_breakdown: dict[str, CalibrationBudget]


def compute_budget(timeline: ObservingTimeline) -> BudgetStats:
    """Compute summary statistics for a timeline.

    Parameters
    ----------
    timeline : ObservingTimeline
        Input timeline.

    Returns
    -------
    BudgetStats
        Timeline totals, efficiency, per-patch science time and the
        calibration breakdown; see :class:`BudgetStats` for the field
        contract.
    """
    per_patch: dict[str, PatchBudget] = {}
    scan_indices: dict[str, set[int]] = {}
    for block in timeline.science_blocks:
        patch = per_patch.setdefault(
            block.patch_name, {"science_time": 0.0, "n_scans": 0, "n_unique_scans": 0}
        )
        patch["science_time"] += block.duration
        patch["n_scans"] += 1
        scan_indices.setdefault(block.patch_name, set()).add(block.scan_index)
    for name, indices in scan_indices.items():
        per_patch[name]["n_unique_scans"] = len(indices)

    calibration_breakdown: dict[str, CalibrationBudget] = {}
    for block in timeline.calibration_blocks:
        cal = calibration_breakdown.setdefault(block.scan_type, {"count": 0, "total_time": 0.0})
        cal["count"] += 1
        cal["total_time"] += block.duration

    return {
        "total_time": timeline.total_time,
        "science_time": timeline.total_science_time,
        "calibration_time": timeline.total_calibration_time,
        "slew_time": timeline.total_slew_time,
        "idle_time": timeline.total_idle_time,
        "efficiency": timeline.efficiency,
        "n_science_scans": timeline.n_science_scans,
        "n_calibration_blocks": len(timeline.calibration_blocks),
        "per_patch": per_patch,
        "calibration_breakdown": calibration_breakdown,
    }
