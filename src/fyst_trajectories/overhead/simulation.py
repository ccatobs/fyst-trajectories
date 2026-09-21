"""Timeline simulation: trajectory generation and hitmap accumulation.

Bridges the timeline (sequence of blocks) with fyst-trajectories' planning
functions to generate actual trajectories and accumulate coverage maps.
"""

import dataclasses
import logging
from typing import TypedDict, cast

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

from ..coordinates import Coordinates
from ..planning import (
    FieldRegion,
    ScanBlock,
    plan_constant_el_scan,
    plan_daisy_scan,
    plan_pong_scan,
    plan_source_ces,
)
from ..planning.footprints import inflate_footprint, offset_footprint_eta, resolve_footprint
from ..site import Site
from .exceptions import BlockNotReconstructableError, ScanParamsSchemaError
from .models import (
    BlockType,
    CEScanParams,
    DaisyScanParams,
    ObservingTimeline,
    PongScanParams,
    TimelineBlock,
    TimelineBlockMetadata,
    validate_scan_params,
)

__all__ = [
    "BudgetStats",
    "CalibrationBudget",
    "PatchBudget",
    "accumulate_hitmaps",
    "compute_budget",
    "schedule_to_trajectories",
]

logger = logging.getLogger(__name__)

# Symmetric time buffer (seconds) added to each side of a recorded
# source-CES pass window before re-solving. The exact stored [t0, t1] sits
# on the elevation-crossing edges where the planner's cover-vs-arc guard is
# most sensitive, so widening the search window keeps the re-solve clear of
# those edges while still landing on the same crossing (the rebuilt t0/t1
# match the recorded window to within a coarse sampling step).
_SOURCE_CES_WINDOW_BUFFER_SEC = 300.0

# Kernel-override keys of SourceCESScanParams. Each is forwarded to
# plan_source_ces verbatim when the recorded params carry it and omitted
# otherwise, so the kernel's own defaults stay authoritative for passes
# planned without overrides. The rebuild test derives the expected set from
# the TypedDict, so a key added there without being forwarded here fails
# that test rather than being dropped silently.
_SOURCE_CES_OVERRIDE_KEYS = ("az_accel", "az_padding", "v_az", "az_speed", "az_throw", "dwell")


def schedule_to_trajectories(
    timeline: ObservingTimeline,
    science_only: bool = True,
) -> list[tuple[TimelineBlock, ScanBlock]]:
    """Generate trajectories for timeline blocks.

    Reconstructs planning parameters from each block's metadata and calls
    the appropriate ``plan_*`` function, returning one
    ``(TimelineBlock, ScanBlock)`` pair per reconstructed block.

    Parameters
    ----------
    timeline : ObservingTimeline
        Input timeline.
    science_only : bool
        Which blocks to reconstruct.

        * ``True`` (default): science blocks only.
        * ``False``: science blocks **plus** calibration blocks whose
          metadata carries a ``scan_params`` dict. Today those are the
          source-CES planet-calibration passes emitted when
          ``CalibrationPolicy.planet_cal_scan`` is set; each is rebuilt with
          :func:`~fyst_trajectories.planning.plan_source_ces` from its recorded
          parameters.

        Calibration blocks with **no** ``scan_params`` (parked planet cals,
        retunes, pointing, focus, skydip) are placeholders with no
        trajectory to rebuild and are skipped silently, not logged as
        failures. Slew and idle blocks are always skipped.

    Returns
    -------
    list of (TimelineBlock, ScanBlock)
        Pairs of timeline blocks and their generated trajectories. A
        science trajectory covers only its own block's
        ``[t_start, t_stop)`` window: the subscans of one
        constant-elevation visit are consecutive slices of a single
        crossing solve (anchored at the shared ``metadata["t0_scan"]``),
        not one full pass each, so summing samples over the pairs counts
        each visit once. ``ScanBlock.duration`` is the
        slice; ``computed_params`` and ``summary`` still describe the
        full solved pass. Calibration passes are one block per pass
        and are returned whole.

    Notes
    -----
    Blocks that attempt reconstruction but fail are logged at ``WARNING``
    and skipped, so one bad block does not abort the whole timeline. This
    subpackage's own refusals are
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
            # ``ScanParamsSchemaError`` and ``BlockNotReconstructableError``
            # are this subpackage's own refusals and are both ``ValueError``
            # subclasses. The rest of the tuple is the net for metadata that
            # is malformed rather than merely incomplete: a recorded
            # velocity of zero surfaces as a plain ``ValueError`` from config
            # validation, and a truncated dict as ``KeyError``/``TypeError``
            # from astropy. A rebuild is best-effort, so every one of those
            # skips the block with a log line instead of failing the batch.
            logger.warning(
                "Failed to generate trajectory for block '%s' at %s: %s",
                sblock.patch_name,
                sblock.t_start.iso,
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
        :func:`_slice_to_block_window`, so ``duration`` and
        ``trajectory.start_time`` describe the slice while
        ``computed_params`` and ``summary`` describe the full solved
        pass.

    Raises
    ------
    ScanParamsSchemaError
        If ``sblock.metadata`` is missing any of the required geometry
        keys (``ra_center``, ``dec_center``, ``width``, ``height``,
        ``velocity``), or names a scan type that cannot be rebuilt.
    BlockNotReconstructableError
        If the re-solved scan no longer overlaps the block's time window.
    KeyError
        If ``scan_params`` contains keys not allowed for
        ``sblock.scan_type`` (see
        :func:`~fyst_trajectories.overhead.validate_scan_params`).

    Notes
    -----
    Calibration blocks are dispatched to :func:`_generate_source_ces_trajectory`,
    which rebuilds the recorded source-CES planet-calibration pass. Science
    blocks take the ``ra_center``/``dec_center`` geometry path below.
    """
    meta = sblock.metadata

    if sblock.block_type == BlockType.CALIBRATION:
        # A pass recorded without an absolute window (a relative dispatch
        # dict) re-solves inside its own block bounds, anchored at t0_scan.
        pass_start = Time(meta["t0_scan"], scale="utc") if "t0_scan" in meta else sblock.t_start
        return _generate_source_ces_trajectory(
            meta, site, fallback_window=(pass_start, sblock.t_stop)
        )

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

    # Validate scan_params shape before dispatch. The ``cast(...)`` below
    # is a no-op at runtime; without this check, typos like ``"radiu"``
    # in a Daisy patch or a ``"spacing"`` key on a CE patch would fall
    # through to the ``.get()`` default silently.
    validate_scan_params(scan_params, sblock.scan_type)

    field = FieldRegion(
        ra_center=ra_center,
        dec_center=dec_center,
        width=width,
        height=height,
    )

    if sblock.scan_type == "constant_el":
        ce_params = cast(CEScanParams, scan_params)
        # CE subscans are slices of one physical crossing scan; the visit
        # anchor recorded at emission (metadata["t0_scan"]) is the anchor
        # the scheduler's corridor gate guaranteed the crossing solve
        # succeeds from. A subscan's own t_start may lie past the opening
        # crossing (where the forward search can no longer find it), so
        # blocks without the key (timelines written before ``t0_scan`` was
        # recorded) fall back to t_start and reconstruct only when that
        # anchor happens to precede the pass opening.
        anchor = Time(meta["t0_scan"], scale="utc") if "t0_scan" in meta else sblock.t_start
        # A sidereal window fixes the timing and the azimuth range on its
        # own, so the planner refuses ``rising`` beside it; the block still
        # records a rising flag (every science block does) and it is simply
        # not what placed this pass.
        lsa_window = ce_params.get("lsa_window")
        scan_block = plan_constant_el_scan(
            field=field,
            elevation=sblock.elevation,
            velocity=velocity,
            site=site,
            start_time=anchor,
            rising=None if lsa_window is not None else sblock.rising,
            az_accel=ce_params.get("az_accel", 1.0),
            timestep=ce_params.get("timestep", 0.1),
            az_padding=ce_params.get("az_padding", 2.0),
            lsa_window=lsa_window,
        )
    elif sblock.scan_type == "pong":
        pong_params = cast(PongScanParams, scan_params)
        scan_block = plan_pong_scan(
            field=field,
            velocity=velocity,
            spacing=pong_params.get("spacing", 0.1),
            num_terms=pong_params.get("num_terms", 4),
            site=site,
            start_time=sblock.t_start,
            timestep=pong_params.get("timestep", 0.1),
            angle=pong_params.get("angle", 0.0),
            n_cycles=pong_params.get("n_cycles", 1),
        )
    elif sblock.scan_type == "daisy":
        daisy_params = cast(DaisyScanParams, scan_params)
        scan_block = plan_daisy_scan(
            ra=ra_center,
            dec=dec_center,
            radius=daisy_params.get("radius", 1.0),
            velocity=velocity,
            turn_radius=daisy_params.get("turn_radius", 0.5),
            avoidance_radius=daisy_params.get("avoidance_radius", 0.1),
            start_acceleration=daisy_params.get("start_acceleration", 0.5),
            site=site,
            start_time=sblock.t_start,
            timestep=daisy_params.get("timestep", 0.1),
            duration=sblock.duration,
        )
    else:
        raise ScanParamsSchemaError(f"Unknown scan type: {sblock.scan_type}")

    return _slice_to_block_window(scan_block, sblock)


def _slice_to_block_window(scan_block: ScanBlock, sblock: TimelineBlock) -> ScanBlock:
    """Slice a rebuilt science trajectory to its block's own time window.

    A CE subscan re-solves its visit's whole crossing pass and a pong
    rebuild covers whole pattern periods, so both come back longer
    than the block they were rebuilt for; unsliced, a consumer
    summing samples would multiply-count the visit. Only the samples
    inside ``[t_start, t_stop)`` are kept; the half-open window keeps
    back-to-back subscans from double-counting their shared boundary
    sample. ``times`` are re-zeroed and ``start_time`` advanced so
    the slice keeps the ``times[0] == 0`` convention, and
    ``retune_events`` are dropped (planners emit none; events from a
    pre-slice origin would be misplaced on the slice). The block's
    ``computed_params`` and ``summary`` still describe the full
    solved pass. Because each pong subscan restarts its pattern from
    its own block start, only the first block-duration of the pong
    pattern is ever simulated; coverage consumers see the pattern
    head, not its tail.

    A trajectory already inside the window (the daisy branch, which
    plans with the block duration) is returned unchanged.

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
    rel_start = (sblock.t_start - traj.start_time).sec
    rel_stop = (sblock.t_stop - traj.start_time).sec
    mask = (t >= rel_start) & (t < rel_stop)
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
    as a source-CES pass sequence
    (``CalibrationPolicy.planet_cal_scan``) and replays it through
    :func:`~fyst_trajectories.planning.plan_source_ces`, returning the same
    :class:`~fyst_trajectories.planning.ScanBlock` the pass produced at emit
    time (to within the coarse-sampling re-solve tolerance).

    Parameters
    ----------
    meta : TimelineBlockMetadata
        Calibration-block metadata carrying ``scan_params``.
    site : Site
        Observatory site.
    fallback_window : tuple of Time, optional
        The pass bounds to re-solve inside when ``scan_params`` carries no
        ``window`` (a relative dispatch dict): normally the block's
        ``t0_scan`` and ``t_stop``.

    Returns
    -------
    ScanBlock
        The rebuilt source-CES pass.

    Raises
    ------
    KeyError
        If ``scan_params`` is absent or carries keys not allowed for
        ``source_ces`` (see
        :func:`~fyst_trajectories.overhead.validate_scan_params`).
    BlockNotReconstructableError
        If ``scan_params`` carries no ``window`` and no
        ``fallback_window`` was given.
    ValueError, TypeError
        Propagated from :func:`~fyst_trajectories.planning.plan_source_ces` when the
        recorded geometry can no longer be solved.
    """
    params = meta["scan_params"]
    validate_scan_params(params, "source_ces")

    if "window" in params:
        t0 = Time(params["window"][0], scale="utc")
        t1 = Time(params["window"][1], scale="utc")
    elif fallback_window is not None:
        t0, t1 = fallback_window
    else:
        raise BlockNotReconstructableError(
            "source_ces scan_params carry no window and no fallback_window was given"
        )
    buffer = TimeDelta(_SOURCE_CES_WINDOW_BUFFER_SEC, format="sec")

    # The margin and the eta offset are load-bearing geometry, not
    # provenance: the margin widens the crossing the pass was solved on, and
    # each pass drags the source through a different focal-plane row, so the
    # rebuilt pass must use the base footprint inflated then shifted exactly
    # as it was planned. Resolve the base tag, inflate, then shift.
    base_footprint = resolve_footprint(params["footprint"])
    base_footprint = inflate_footprint(base_footprint, params.get("footprint_margin", 0.0))
    pass_footprint = offset_footprint_eta(base_footprint, params["eta_offset_deg"])

    # Kernel overrides ride along only when they were recorded; see
    # _SOURCE_CES_OVERRIDE_KEYS.
    overrides = {key: params[key] for key in _SOURCE_CES_OVERRIDE_KEYS if key in params}

    # Widen the window before re-solving; see _SOURCE_CES_WINDOW_BUFFER_SEC.
    return plan_source_ces(
        body=params["body"],
        footprint=pass_footprint,
        el_bore=params["el_bore"],
        boresight_rot=params["boresight_rot"],
        timestep=params["timestep"],
        window=(t0 - buffer, t1 + buffer),
        mode=params["mode"],
        site=site,
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
    binning, or feed the schedule to a dedicated mapping simulator.

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
        import healpy as hp
    except ImportError:
        raise ImportError(
            "healpy is required for hitmap accumulation. Install with: pip install healpy"
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

        times = traj.start_time + TimeDelta(traj.times * u.s)

        stride = 10
        indices = np.arange(0, len(traj.times), stride)
        if traj.scan_flag is not None:
            mask = traj.science_mask[indices]
            indices = indices[mask]

        if len(indices) == 0:
            continue

        ra_arr, dec_arr = coords.altaz_to_radec(traj.az[indices], traj.el[indices], times[indices])
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
    stats: dict = {
        "total_time": timeline.total_time,
        "science_time": timeline.total_science_time,
        "calibration_time": timeline.total_calibration_time,
        "slew_time": timeline.total_slew_time,
        "idle_time": timeline.total_idle_time,
        "efficiency": timeline.efficiency,
        "n_science_scans": timeline.n_science_scans,
        "n_calibration_blocks": len(timeline.calibration_blocks),
    }

    patch_stats = {}
    for block in timeline.science_blocks:
        name = block.patch_name
        if name not in patch_stats:
            patch_stats[name] = {
                "science_time": 0.0,
                "n_scans": 0,
                "scan_indices": set(),
            }
        patch_stats[name]["science_time"] += block.duration
        patch_stats[name]["n_scans"] += 1
        patch_stats[name]["scan_indices"].add(block.scan_index)

    for name in patch_stats:
        patch_stats[name]["n_unique_scans"] = len(patch_stats[name]["scan_indices"])
        del patch_stats[name]["scan_indices"]

    stats["per_patch"] = patch_stats

    cal_stats = {}
    for block in timeline.calibration_blocks:
        cal_type = block.scan_type
        if cal_type not in cal_stats:
            cal_stats[cal_type] = {"count": 0, "total_time": 0.0}
        cal_stats[cal_type]["count"] += 1
        cal_stats[cal_type]["total_time"] += block.duration

    stats["calibration_breakdown"] = cal_stats

    return cast(BudgetStats, stats)
