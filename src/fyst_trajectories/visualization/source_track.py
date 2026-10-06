"""The source's track through the focal plane over one source-CES pass.

Requires ``matplotlib`` (install via ``pip install fyst-trajectories[plotting]``),
lazy-imported inside the function so importing this module never loads it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..offsets import InstrumentOffset
from ..planning import ScanBlock, source_ces_focal_plane_track
from ..primecam import MODULE_FOV_RADIUS_DEG, PRIMECAM_MODULES
from ..site import Site, get_fyst_site
from ._common import FOOTPRINT_COLOR, _unique_offsets

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_source_track"]


def plot_source_track(
    block: ScanBlock,
    *,
    site: Site | None = None,
    modules: Mapping[str, InstrumentOffset] | None = None,
    fov_radius_deg: float = MODULE_FOV_RADIUS_DEG,
    labels: bool = True,
    title: str | None = None,
    ax: Axes | None = None,
    show: bool = True,
) -> Figure:
    """Plot where the source travels across the focal plane during a pass.

    Draws the module layout in the focal-plane frame (cross-elevation
    ``xi`` against elevation ``eta``, in degrees, to scale) and over it the
    source's track from :func:`~fyst_trajectories.planning.source_ces_focal_plane_track`:
    the sweep legs carry the source back and forth across the swept window
    while the drift moves it through the array, so the track is a sawtooth
    whose envelope shows which modules the source crossed and for how much
    of the pass. The start of the pass is marked with a circle and the end
    with a cross.

    The module positions come from the offsets registry, so the layout
    reflects whatever module-to-position assignment the registry holds; the
    title states the Nasmyth sign and the boresight rotation the track was
    projected with.

    Parameters
    ----------
    block : ScanBlock
        A pass from :func:`~fyst_trajectories.planning.plan_source_ces` or
        :func:`~fyst_trajectories.planning.plan_source_ces_passes`.
    site : Site, optional
        Observing site (Nasmyth sign and ephemeris location). Defaults to
        :func:`~fyst_trajectories.site.get_fyst_site`.
    modules : mapping of str to InstrumentOffset, optional
        Modules to draw. Default
        :data:`~fyst_trajectories.primecam.PRIMECAM_MODULES` (alias keys
        pointing at the same offset are drawn once).
    fov_radius_deg : float, optional
        Per-module field-of-view radius in degrees. Default
        :data:`~fyst_trajectories.primecam.MODULE_FOV_RADIUS_DEG`.
    labels : bool, optional
        Label each module circle with its offset name. Default True.
    title : str, optional
        Axes title. Default is an auto-generated summary naming the
        source, the pass elevation and direction, and the projection
        assumptions.
    ax : matplotlib.axes.Axes, optional
        Draw into this axes instead of creating a new figure. When given,
        ``show`` is ignored and no layout call is made on the caller's
        figure.
    show : bool, optional
        Call ``plt.show()`` after rendering (only when the function
        created the figure). Default True.

    Returns
    -------
    Figure
        The figure containing the track (``ax.get_figure()`` when ``ax``
        was supplied).

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If ``block`` is not a source-CES pass, ``fov_radius_deg`` is not
        positive, or ``modules`` is empty.
    """
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        from matplotlib.patches import Circle  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_source_track(). "
            "Install it with: pip install fyst-trajectories[plotting]"
        ) from None

    if not np.isfinite(fov_radius_deg) or fov_radius_deg <= 0:
        raise ValueError(f"fov_radius_deg must be a finite value > 0, got {fov_radius_deg}")
    site = get_fyst_site() if site is None else site
    modules = PRIMECAM_MODULES if modules is None else modules

    unique = _unique_offsets(modules)

    xi, eta = source_ces_focal_plane_track(block, site=site)
    metadata = block.trajectory.metadata
    params = metadata.pattern_params

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
    else:
        fig = ax.get_figure()

    for offset in unique:
        dx, dy = offset.dx_deg, offset.dy_deg
        ax.add_patch(
            Circle(
                (dx, dy),
                fov_radius_deg,
                facecolor=FOOTPRINT_COLOR,
                alpha=0.18,
                edgecolor=FOOTPRINT_COLOR,
                lw=1.2,
            )
        )
        if labels:
            name = (offset.name or "").removeprefix("PrimeCam-").lower() or "?"
            ax.annotate(
                name,
                xy=(dx, dy),
                xytext=(0, -14),  # below the centre, clear of the boresight marker
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=9,
                alpha=0.8,
            )

    ax.plot(xi, eta, color="#d62728", lw=0.8, alpha=0.9, label="source track")
    ax.plot(xi[0], eta[0], "o", color="#d62728", ms=7, mfc="white", mew=1.5, label="pass start")
    ax.plot(xi[-1], eta[-1], "x", color="#d62728", ms=8, mew=1.8, label="pass end")
    ax.plot(0.0, 0.0, "+", color="black", ms=10, mew=1.5, zorder=5)

    module_extent = max(np.hypot(o.dx_deg, o.dy_deg) for o in unique) + fov_radius_deg
    track_extent = float(max(np.abs(xi).max(), np.abs(eta).max()))
    limit = 1.1 * max(module_extent, track_extent)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.grid(ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlabel("Focal-plane xi (cross-elevation) [deg]", fontsize=10)
    ax.set_ylabel("Focal-plane eta (elevation) [deg]", fontsize=10)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    if title is None:
        el_bore = float(params["el_bore"])
        rot = float(params["boresight_rot"])
        title = (
            f"{metadata.target_name} source-CES pass, {params['mode']} at "
            f"el_bore = {el_bore:.1f} deg\n"
            f"Nasmyth sign {site.nasmyth_sign:+d}, boresight rotation {rot:+.1f} deg, "
            "registry layout"
        )
    ax.set_title(title, fontsize=10.5)

    if own_fig:
        fig.tight_layout()
        if show:
            plt.show()
    return fig
