"""Timeline simulation: trajectory generation and hitmap accumulation.

Bridges the timeline (sequence of blocks) with fyst-trajectories' planning
functions to generate actual trajectories and accumulate coverage maps.
"""

import logging
from typing import cast

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
from ..planning.source_ces import _offset_footprint_eta, _resolve_footprint
from ..site import Site
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
          :func:`~fyst_trajectories.plan_source_ces` from its recorded
          parameters.

        Calibration blocks with **no** ``scan_params`` (parked planet cals,
        retunes, pointing, focus, skydip) are placeholders with no
        trajectory to rebuild and are skipped silently, not logged as
        failures. Slew and idle blocks are always skipped.

    Returns
    -------
    list of (TimelineBlock, ScanBlock)
        Pairs of timeline blocks and their generated trajectories.

    Notes
    -----
    Blocks that attempt reconstruction but fail (a ``ValueError``,
    ``KeyError``, or ``TypeError`` from the planner, e.g. a source no longer
    reachable at the recorded geometry) are logged at ``WARNING`` and
    skipped so one bad block does not abort the whole timeline.
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
        Generated scan block with trajectory.

    Raises
    ------
    ValueError
        If ``sblock.metadata`` is missing any of the required geometry
        keys (``ra_center``, ``dec_center``, ``width``, ``height``,
        ``velocity``).
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
        return _generate_source_ces_trajectory(meta, site)

    required = ("ra_center", "dec_center", "width", "height", "velocity")
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(
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
        return plan_constant_el_scan(
            field=field,
            elevation=sblock.elevation,
            velocity=velocity,
            site=site,
            start_time=sblock.t_start,
            rising=sblock.rising,
            az_accel=ce_params.get("az_accel", 1.0),
            timestep=ce_params.get("timestep", 0.1),
            az_padding=ce_params.get("az_padding", 2.0),
            lsa_window=ce_params.get("lsa_window"),
        )
    elif sblock.scan_type == "pong":
        pong_params = cast(PongScanParams, scan_params)
        return plan_pong_scan(
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
        return plan_daisy_scan(
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
        raise ValueError(f"Unknown scan type: {sblock.scan_type}")


def _generate_source_ces_trajectory(
    meta: TimelineBlockMetadata,
    site: Site,
) -> ScanBlock:
    """Rebuild a source-CES planet-calibration pass from its recorded params.

    Consumes the :class:`~fyst_trajectories.overhead.SourceCESScanParams`
    stored under ``meta["scan_params"]`` when a planet calibration was planned
    as a source-CES pass sequence
    (``CalibrationPolicy.planet_cal_scan``) and replays it through
    :func:`~fyst_trajectories.plan_source_ces`, returning the same
    :class:`~fyst_trajectories.planning.ScanBlock` the pass produced at emit
    time (to within the coarse-sampling re-solve tolerance).

    Parameters
    ----------
    meta : TimelineBlockMetadata
        Calibration-block metadata carrying ``scan_params``.
    site : Site
        Observatory site.

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
    ValueError, TypeError
        Propagated from :func:`~fyst_trajectories.plan_source_ces` when the
        recorded geometry can no longer be solved.
    """
    params = meta["scan_params"]
    validate_scan_params(params, "source_ces")

    t0 = Time(params["window"][0], scale="utc")
    t1 = Time(params["window"][1], scale="utc")
    buffer = TimeDelta(_SOURCE_CES_WINDOW_BUFFER_SEC, format="sec")

    # The eta offset is load-bearing geometry, not provenance: each pass drags
    # the source through a different focal-plane row, so the rebuilt pass must
    # use the base footprint shifted by the recorded offset. Resolve the base
    # tag then shift it, exactly as plan_source_ces_passes does.
    base_footprint = _resolve_footprint(params["footprint"])
    pass_footprint = _offset_footprint_eta(base_footprint, params["eta_offset_deg"])

    # Widen the stored [t0, t1] symmetrically before re-solving: the exact
    # window edges trip the planner's cover-vs-arc guard, and the buffer keeps
    # the search off those edges while landing on the same elevation crossing.
    return plan_source_ces(
        body=params["body"],
        footprint=pass_footprint,
        el_bore=params["el_bore"],
        boresight_rot=params["boresight_rot"],
        timestep=params["timestep"],
        window=(t0 - buffer, t1 + buffer),
        mode=params["mode"],
        site=site,
    )


def accumulate_hitmaps(
    trajectory_pairs: list[tuple[TimelineBlock, ScanBlock]],
    site: Site,
    nside: int = 256,
) -> np.ndarray:
    """Accumulate a boresight-level HEALPix hitmap from trajectories.

    For each trajectory, converts az/el to RA/Dec at each timestep,
    bins into HEALPix pixels, and sums across all trajectories.

    This is a simplified boresight-level hitmap (one sample per timestep,
    not per detector). For full detector-level hitmaps, use
    primecam_camera_mapping_simulations with the schedule output.

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
    np.ndarray
        HEALPix map of hit counts per pixel.

    Raises
    ------
    ImportError
        If ``healpy`` is not installed.
    """
    try:
        import healpy as hp
    except ImportError:
        raise ImportError(
            "healpy is required for hitmap accumulation. Install with: pip install healpy"
        ) from None

    npix = hp.nside2npix(nside)
    hitmap = np.zeros(npix, dtype=np.float64)
    # Vacuum az/el -> RA/Dec is correct *only when the trajectory was
    # itself generated with vacuum coordinates*. Pattern generators
    # (patterns/{daisy,pong,sidereal,planet}) accept a user-supplied
    # ``atmosphere=`` and forward it to ``Coordinates``; if the
    # trajectory was built with ``AtmosphericConditions.for_fyst()`` the
    # az/el samples here are refracted and inverting them with vacuum
    # introduces a small systematic (~arcseconds at zenith, up to ~1' at
    # the horizon). For a healpix nside=512 hitmap (7' pixels) the error
    # stays sub-pixel except very near the horizon, so the practical
    # impact is bounded; the asymmetry remains for low-elevation, high-
    # nside maps.
    # TODO: thread the trajectory's atmospheric conditions through
    # ``Trajectory`` metadata so the inverse uses the same refraction
    # model.
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


def compute_budget(timeline: ObservingTimeline) -> dict:
    """Compute summary statistics for a timeline.

    Parameters
    ----------
    timeline : ObservingTimeline
        Input timeline.

    Returns
    -------
    dict
        Summary statistics including efficiency, time breakdowns,
        and per-patch information.
    """
    stats = {
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

    return stats
