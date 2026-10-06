"""The source's track through the focal plane over a planned source-CES pass."""

from __future__ import annotations

import numpy as np

from ...coordinates import Coordinates
from ...offsets import InstrumentOffset, compute_focal_plane_rotation, sky_to_focal_plane
from ...site import AtmosphericConditions, Site
from ...trajectory_utils import get_absolute_times
from .._types import ScanBlock, SourceCESComputedParams


def source_ces_focal_plane_track(
    block: ScanBlock[SourceCESComputedParams],
    *,
    site: Site,
    atmosphere: AtmosphericConditions | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace the source through the focal plane over a planned source-CES pass.

    For every sample of the pass trajectory, place the source relative to
    the boresight in the focal-plane frame the pass was planned in: the
    mechanical rotation at the pass elevation plus the recorded boresight
    rotation, the same angle the planner projected the footprint with. The
    result is what a plot of the pass shows and what a per-module coverage
    calculation integrates.

    Parameters
    ----------
    block : ScanBlock
        A block returned by :func:`plan_source_ces` or
        :func:`plan_source_ces_passes`.
    site : Site
        Telescope site (Nasmyth sign and location).
    atmosphere : AtmosphericConditions, optional
        Refraction model for the source ephemeris. Default ``None``, the
        vacuum frame the pass was planned in.

    Returns
    -------
    xi_deg : np.ndarray
        Cross-elevation focal-plane coordinate of the source in degrees,
        one per trajectory sample.
    eta_deg : np.ndarray
        Elevation focal-plane coordinate of the source in degrees.

    Raises
    ------
    ValueError
        If ``block`` was not produced by the source-CES planners (its
        metadata lacks the ``source_ces`` pattern type or the ``body`` key).

    Notes
    -----
    The source is the one recorded in ``pattern_params["body"]``, not the
    ``target_name`` display label, so relabelling a block leaves its track
    unchanged.

    A body is placed with the site ephemeris at each sample time. A fixed
    source is placed at the RA/Dec recorded on the block, which is the
    catalogue position the pass was planned from: the whole epoch
    propagation is neglected, not just the drift during the pass. The block
    records no proper motion or reference epoch to propagate with, so a
    caller that needs the propagated position has to supply it. The error is
    the accumulated proper motion between the catalogue epoch and the
    observation, which for a fast mover (10 arcsec/yr, 26 years) reaches
    about 4.3 arcmin, a ninth of the 0.65 deg module field radius; for an
    ordinary calibrator it is negligible.
    """
    metadata = block.trajectory.metadata
    if (
        metadata is None
        or metadata.pattern_type != "source_ces"
        or "body" not in metadata.pattern_params
    ):
        raise ValueError("block must come from plan_source_ces or plan_source_ces_passes")
    params = metadata.pattern_params
    trajectory = block.trajectory
    times = get_absolute_times(trajectory)
    coords = Coordinates(site, atmosphere=atmosphere)

    if params["body"] is None:
        src_az, src_el = coords.radec_to_altaz(metadata.center_ra, metadata.center_dec, times)
    else:
        src_az, src_el = coords.get_body_altaz(params["body"], times)

    rotation = compute_focal_plane_rotation(
        el=trajectory.el,
        site=site,
        offset=InstrumentOffset(dx=0.0, dy=0.0),
    ) + float(params["boresight_rot"])
    xi, eta = sky_to_focal_plane(
        trajectory.az, trajectory.el, np.asarray(src_az), np.asarray(src_el), rotation
    )
    return np.asarray(xi, dtype=float), np.asarray(eta, dtype=float)
