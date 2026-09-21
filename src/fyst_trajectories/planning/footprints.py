"""Caller-side footprint transforms shared by the planners and their consumers.

A source-tracking scan is solved against an
:class:`~fyst_trajectories.planning.ArrayFootprint`: the focal-plane
region the source has to be dragged across. The three functions here
build that region and adjust it, and they run *outside* the planning
kernel, in the caller, before the scan is planned.

That matters because anything rebuilding a recorded scan has to
reproduce the footprint the scan was solved against, transform for
transform and in the same order:

1. :func:`resolve_footprint` normalises whatever the caller named into
   an ``ArrayFootprint``.
2. :func:`inflate_footprint` grows it by the planning margin.
3. :func:`offset_footprint_eta` slides it to the pass's own eta step.

Inflate before the eta shift: the inflation is measured from the
footprint centre, so growing an already-shifted footprint pushes its
vertices away from the shifted centre and yields a different region.
A rebuild that applies the same recorded ``footprint_margin`` and
``eta_offset_deg`` in this order reproduces the solved crossing
exactly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..offsets import InstrumentOffset
from ..patterns.configs import _require_non_negative
from ..primecam import MODULE_FOV_RADIUS_DEG, get_primecam_offset
from ._types import ArrayFootprint

__all__ = ["inflate_footprint", "offset_footprint_eta", "resolve_footprint"]

# Vertices per module circle in a resolved cover polygon. Matches the
# 50-vertex count of the Simons Observatory scheduler's
# ``make_circular_cover``. That helper puts its vertices on a circle about
# 1.2 percent larger than the field radius (a 1 percent pad, plus the
# factor that circumscribes the polygon about it) while these sit on the
# radius itself, so the two covers are not vertex-for-vertex identical.
_CIRCULAR_COVER_N_VERTICES = 50


def resolve_footprint(
    footprint: InstrumentOffset | str | Sequence[InstrumentOffset] | ArrayFootprint,
) -> ArrayFootprint:
    """Normalise a user-supplied footprint into an :class:`ArrayFootprint`.

    Accepts:

    * ``ArrayFootprint``: returned unchanged.
    * ``InstrumentOffset``: built as a ``_CIRCULAR_COVER_N_VERTICES``-vertex
      circle of radius
      :data:`~fyst_trajectories.primecam.MODULE_FOV_RADIUS_DEG` around the
      offset.
    * ``str``: resolved via
      :func:`~fyst_trajectories.primecam.get_primecam_offset` then treated
      as the single-``InstrumentOffset`` case.
    * Sequence of ``InstrumentOffset``: each treated as a circle, the
      cover is the concatenation of per-module vertex lists, and the
      aggregate center is the arithmetic mean of per-module ``(dx, dy)``.

    Parameters
    ----------
    footprint : InstrumentOffset or str or sequence of InstrumentOffset or ArrayFootprint
        The footprint in any of the four accepted shapes.

    Returns
    -------
    ArrayFootprint
        The normalised footprint.

    Raises
    ------
    ValueError
        If a footprint sequence is empty.
    TypeError
        If the footprint is not one of the accepted shapes, or a
        sequence holds something other than ``InstrumentOffset``.
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


def offset_footprint_eta(fp: ArrayFootprint, d_eta_deg: float) -> ArrayFootprint:
    """Return a copy of ``fp`` shifted by ``d_eta_deg`` along the eta axis.

    The eta (elevation-direction) shift moves both the footprint center
    and every cover vertex, so the whole array footprint slides along the
    focal-plane elevation axis while keeping its cross-elevation (xi)
    geometry unchanged. It is how a multi-pass sequence steps its passes
    across the array, and it is applied last, after
    :func:`inflate_footprint`.

    Parameters
    ----------
    fp : ArrayFootprint
        Base footprint to shift.
    d_eta_deg : float
        Eta offset in degrees (positive = toward increasing elevation).

    Returns
    -------
    ArrayFootprint
        A new footprint with ``center_eta_deg`` and ``cover_eta_deg``
        shifted by ``d_eta_deg``.
    """
    return ArrayFootprint(
        center_xi_deg=fp.center_xi_deg,
        center_eta_deg=fp.center_eta_deg + d_eta_deg,
        cover_xi_deg=fp.cover_xi_deg.copy(),
        cover_eta_deg=fp.cover_eta_deg + d_eta_deg,
    )


def inflate_footprint(fp: ArrayFootprint, margin_deg: float) -> ArrayFootprint:
    """Return a copy of ``fp`` whose cover extends ``margin_deg`` farther from the centre.

    Every cover vertex moves directly away from the footprint centre by
    ``margin_deg``; the centre itself does not move. For the circular
    single-module covers the planners build this is exactly a larger
    circle, so the solved crossing lengthens as if the module were
    ``margin_deg`` wider in every direction. For a multi-module cover it
    grows the extent in every direction from the array centre by the
    margin, an approximation that is exact along each radial line.

    Applied caller-side before planning and before
    :func:`offset_footprint_eta`, and re-applied on rebuild from the
    recorded ``footprint_margin``; the kernel itself is unchanged. A zero
    margin returns an equal copy; a vertex sitting exactly at the centre
    has no outward direction and stays where it is.

    Parameters
    ----------
    fp : ArrayFootprint
        Base footprint to inflate.
    margin_deg : float
        On-sky margin in degrees added on every side. Must be
        non-negative.

    Returns
    -------
    ArrayFootprint
        A new footprint with the same centre and the pushed-out cover.

    Raises
    ------
    ValueError
        If ``margin_deg`` is negative or not finite.
    """
    _require_non_negative(margin_deg, "margin_deg")
    dx = fp.cover_xi_deg - fp.center_xi_deg
    dy = fp.cover_eta_deg - fp.center_eta_deg
    radius = np.hypot(dx, dy)
    scale = np.divide(radius + margin_deg, radius, out=np.ones_like(radius), where=radius > 0.0)
    return ArrayFootprint(
        center_xi_deg=fp.center_xi_deg,
        center_eta_deg=fp.center_eta_deg,
        cover_xi_deg=fp.center_xi_deg + dx * scale,
        cover_eta_deg=fp.center_eta_deg + dy * scale,
    )
