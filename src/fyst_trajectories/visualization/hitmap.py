"""Hit-density coverage maps in equatorial coordinates.

Bins each detector module's sky track into a 2D RA/Dec histogram,
convolved by default with the module FOV disk, and draws one panel per
module (:func:`plot_hit_map`).

Everything is computed on a plain RA x Dec grid with no ``cos(dec)``
weighting: reported areas are **coordinate areas** (RA extent times Dec
extent), not solid angle, and the disk kernel is circular in coordinate
space, so on sky it is stretched in RA by ``1/cos(dec)`` (about 15% at
dec -30, about 100% at dec -60). Treat the numbers as relative
diagnostics rather than sky areas.

These functions require ``matplotlib`` (install via
``pip install fyst-trajectories[plotting]``).

Examples
--------
Plot the coverage of two PrimeCam modules, each averaged over its
0.65 deg field-of-view radius:

>>> from fyst_trajectories.primecam import get_primecam_offset
>>> from fyst_trajectories.visualization import plot_hit_map
>>> modules = {
...     "module i1": get_primecam_offset("i1"),
...     "module i6": get_primecam_offset("i6"),
... }
>>> fig = plot_hit_map(trajectory, modules=modules, show=True)

Plot the raw detector-centre tracks instead:

>>> fig = plot_hit_map(
...     trajectory,
...     modules=modules,
...     fov_radius_deg=None,
...     show=True,
... )
"""

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ..primecam import MODULE_FOV_RADIUS_DEG, PRIMECAM_MODULES
from ..site import Site, get_fyst_site
from ._common import _unique_keyed_offsets

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ..offsets import InstrumentOffset
    from ..trajectory import Trajectory

__all__ = ["plot_hit_map"]

#: Candidate RA tick spacings, in minutes of RA, smallest first.
_RA_TICK_STEPS_MIN = (1, 2, 5, 10, 15, 30, 60, 120)

#: Candidate RA tick spacings, in seconds of RA, for an axis that holds fewer
#: than two whole minutes of RA, smallest first.
_RA_TICK_STEPS_SEC = (1, 2, 3, 5, 10, 15, 20, 30)

#: Most RA ticks a hit-map panel labelled in minutes draws.
_RA_MAX_TICKS = 5

#: Most RA ticks a hit-map panel labelled to the second draws (longer labels).
_RA_MAX_SECOND_TICKS = 3


def _make_disk_kernel(radius_bins: float) -> np.ndarray:
    """Create a 2D circular disk kernel for convolution.

    Parameters
    ----------
    radius_bins : float
        Radius of the disk in bin units.

    Returns
    -------
    ndarray
        2D array, constant inside the disk and 0.0 outside,
        normalized so the sum equals 1.0.
    """
    r_int = int(np.ceil(radius_bins))
    y, x = np.ogrid[-r_int : r_int + 1, -r_int : r_int + 1]
    mask = (x**2 + y**2) <= radius_bins**2
    kernel = mask.astype(float)
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return kernel


def _format_ra_hm(deg: float, _pos: Any) -> str:
    """Format RA in degrees as hours and minutes."""
    total_minutes = round((deg % 360.0) * 4.0) % 1440
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}$^{{h}}${minutes:02d}$^{{m}}$"


def _format_ra_hms(deg: float, _pos: Any) -> str:
    """Format RA in degrees as hours, minutes and seconds."""
    total_seconds = round((deg % 360.0) * 240.0) % 86400
    hours, rest = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}$^{{h}}${minutes:02d}$^{{m}}${seconds:02d}$^{{s}}$"


def _ra_ticks_inside(lo: float, hi: float, step_deg: float) -> int:
    """Return how many multiples of ``step_deg`` lie in ``[lo, hi]``."""
    return math.floor(hi / step_deg) - math.ceil(lo / step_deg) + 1


def _ra_tick_step_deg(lo_deg: float, hi_deg: float) -> float:
    """Return the RA tick spacing, in degrees, for an axis from ``lo_deg`` to ``hi_deg``.

    An axis that holds two or more whole minutes of RA is ticked on the
    smallest of 1, 2, 5, 10, 15, 30, 60 and 120 minutes that puts at most
    five ticks inside it (120 minutes when none does); five ticks keep
    two-digit-hour labels clear of each other on a 6 in panel. A narrower
    axis is ticked on the smallest of 1, 2, 3, 5, 10, 15, 20 and 30 seconds
    that puts at most three ticks inside it, labelled to the second (see
    ``_format_ra_hms``), since those labels are longer. Every axis of at
    least two seconds of RA gets at least two ticks, and the labels never
    repeat.
    """
    lo, hi = sorted((lo_deg, hi_deg))
    if _ra_ticks_inside(lo, hi, 0.25) < 2:
        for step_sec in _RA_TICK_STEPS_SEC:
            if _ra_ticks_inside(lo, hi, step_sec / 240.0) <= _RA_MAX_SECOND_TICKS:
                return step_sec / 240.0
    for step_min in _RA_TICK_STEPS_MIN:
        if _ra_ticks_inside(lo, hi, step_min / 4.0) <= _RA_MAX_TICKS:
            return step_min / 4.0
    return _RA_TICK_STEPS_MIN[-1] / 4.0


def _format_dec_deg(deg: float, _pos: Any) -> str:
    """Format Dec in degrees with degree symbol."""
    return f"{deg:g}$^\\circ$"


def plot_hit_map(
    trajectory: "Trajectory",
    *,
    site: Site | None = None,
    modules: "Mapping[str, InstrumentOffset] | None" = None,
    fov_radius_deg: float | None = MODULE_FOV_RADIUS_DEG,
    bin_size: float = 0.02,
    smooth_sigma: float | None = None,
    footprint_threshold: float = 0.1,
    stats_threshold: float = 0.5,
    cmap: str = "viridis",
    title: str | None = None,
    axes: "Sequence[Axes] | None" = None,
    show: bool = True,
) -> "Figure":
    """Plot hit-density maps in RA/Dec, one panel per detector module.

    For each module, computes the detector's sky track by applying its
    offset to the boresight trajectory, converts Az/El to RA/Dec, and bins
    the track into a 2D histogram.

    By default the histogram is convolved with a disk kernel of the
    module's field-of-view radius (circular in RA/Dec coordinate space, so
    stretched in RA on sky away from the equator), producing filled
    coverage maps. Two statistics are drawn above each panel, at its
    right, clear of the map: areas in square coordinate degrees on a
    plain RA x Dec grid, with no ``cos(dec)`` weighting.

    With ``fov_radius_deg=None`` the raw detector-centre track is plotted
    instead. Statistics are then reported as fractional coverage ratios.

    Parameters
    ----------
    trajectory : Trajectory
        Boresight trajectory with ``start_time`` set (needed for
        Az/El -> RA/Dec conversion).
    site : Site, optional
        Observing site (Nasmyth sign and location). Defaults to
        :func:`~fyst_trajectories.site.get_fyst_site`.
    modules : mapping of str to InstrumentOffset, optional
        Modules to draw, one panel each. Default
        :data:`~fyst_trajectories.primecam.PRIMECAM_MODULES` (alias keys
        pointing at the same offset are drawn once). Each panel is titled
        with the first key of its offset, so a caller can label panels
        freely (the other plot functions label modules with the offset's
        name instead); use ``InstrumentOffset(dx=0, dy=0)`` for the
        boresight.
    fov_radius_deg : float or None, optional
        Per-module on-sky FOV radius in degrees: the histogram is
        convolved with a disk kernel of this radius in coordinate degrees,
        approximating coverage from the full module. Default
        :data:`~fyst_trajectories.primecam.MODULE_FOV_RADIUS_DEG` (0.65).
        ``None`` draws the raw detector-centre track, with no convolution.
    bin_size : float, optional
        Histogram bin size in degrees for both RA and Dec. Default 0.02.
    smooth_sigma : float or None, optional
        If not None, apply Gaussian smoothing with this sigma (in bins)
        after any module FOV convolution. Default is None (no smoothing).
    footprint_threshold : float, optional
        Fraction of max hit count to define the footprint boundary
        contour. Default 0.1 (10% of peak). The contour is drawn only on a
        filled map (``fov_radius_deg`` or ``smooth_sigma`` set).
    stats_threshold : float, optional
        Fraction of max for the area efficiency statistic. Default 0.5.
    cmap : str, optional
        Matplotlib colormap name. Default "viridis".
    title : str, optional
        Title text. On a figure this function creates it is drawn as the
        figure suptitle; with caller-supplied ``axes`` it replaces the
        first panel's title (the caller's suptitle is never touched).
        Default None, no title beyond the panel titles.
    axes : sequence of matplotlib.axes.Axes, optional
        Draw into these axes (one per distinct module, in mapping order)
        instead of creating a new figure. When given, ``show`` is ignored
        and no layout call is made on the caller's figure; each panel's
        colour bar takes its space from that panel's axes. Without it the
        function creates a figure of up to four 6 x 5 in panels per row.
    show : bool, optional
        Call ``plt.show()`` after rendering (only when the function
        created the figure). Default True.

    Returns
    -------
    Figure
        The figure containing the panels (``axes[0].get_figure()`` when
        ``axes`` was supplied).

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If ``modules`` is empty, ``bin_size`` is not positive,
        ``fov_radius_deg`` is neither None nor a finite positive value,
        ``axes`` does not hold one axes per distinct module, or the
        trajectory has no ``start_time`` set.
    """
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        from matplotlib import ticker  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_hit_map(). "
            "Install it with: pip install fyst-trajectories[plotting]"
        ) from None

    panels = _unique_keyed_offsets(PRIMECAM_MODULES if modules is None else modules)
    if not bin_size > 0.0:
        raise ValueError(f"bin_size must be positive, got {bin_size}")
    if fov_radius_deg is not None and not (np.isfinite(fov_radius_deg) and fov_radius_deg > 0.0):
        raise ValueError(f"fov_radius_deg must be a finite value > 0, got {fov_radius_deg}")
    if axes is not None and len(axes) != len(panels):
        raise ValueError(f"axes has {len(axes)} entries but {len(panels)} modules are drawn")
    if trajectory.start_time is None:
        raise ValueError("Trajectory must have start_time for RA/Dec conversion")
    site = get_fyst_site() if site is None else site

    from scipy.ndimage import gaussian_filter  # pylint: disable=import-outside-toplevel
    from scipy.signal import fftconvolve  # pylint: disable=import-outside-toplevel

    from ..coordinates import Coordinates  # pylint: disable=import-outside-toplevel
    from ..offsets import boresight_to_detector, compute_focal_plane_rotation
    from ..trajectory_utils import get_absolute_times

    coords = Coordinates(site)
    abs_times = get_absolute_times(trajectory)

    coverage_mode = fov_radius_deg is not None
    pad_deg = fov_radius_deg if fov_radius_deg is not None else 0.0
    bin_area_deg2 = bin_size * bin_size

    own_fig = axes is None
    if axes is None:
        n_panels = len(panels)
        ncols = min(n_panels, 4)
        nrows = -(-n_panels // ncols)
        fig, grid = plt.subplots(
            nrows,
            ncols,
            figsize=(6 * ncols, 5 * nrows),
            squeeze=False,
        )
        flat = list(grid.ravel())
        for spare in flat[n_panels:]:
            spare.set_visible(False)
        axes = flat[:n_panels]
    else:
        axes = list(axes)
        fig = axes[0].get_figure()

    for ax, (label, offset) in zip(axes, panels):
        # Horizon-frame projection: mechanical rotation; the celestial
        # rotation enters in the az/el -> RA/Dec conversion below.
        fr = compute_focal_plane_rotation(
            trajectory.el,
            site=site,
            offset=offset,
        )

        det_az, det_el = boresight_to_detector(
            trajectory.az,
            trajectory.el,
            offset,
            fr,
        )

        ra, dec = coords.altaz_to_radec(det_az, det_el, obstime=abs_times)

        ra = np.asarray(ra, dtype=float)
        dec = np.asarray(dec, dtype=float)
        if ra.max() - ra.min() > 180:
            ra = (ra + 180) % 360 - 180

        ra_bins = np.arange(
            ra.min() - bin_size - pad_deg,
            ra.max() + 2 * bin_size + pad_deg,
            bin_size,
        )
        dec_bins = np.arange(
            dec.min() - bin_size - pad_deg,
            dec.max() + 2 * bin_size + pad_deg,
            bin_size,
        )
        hist, ra_edges, dec_edges = np.histogram2d(
            ra,
            dec,
            bins=[ra_bins, dec_bins],
        )

        if fov_radius_deg is not None:
            radius_bins = fov_radius_deg / bin_size
            kernel = _make_disk_kernel(radius_bins)
            hist = fftconvolve(hist, kernel, mode="same")
            np.maximum(hist, 0.0, out=hist)
            hist[hist < 1e-10] = 0.0

        if smooth_sigma is not None:
            hist = gaussian_filter(hist, sigma=smooth_sigma)

        ra_centers = 0.5 * (ra_edges[:-1] + ra_edges[1:])
        dec_centers = 0.5 * (dec_edges[:-1] + dec_edges[1:])
        im = ax.pcolormesh(
            ra_centers,
            dec_centers,
            hist.T,
            cmap=cmap,
            shading="auto",
        )
        cbar_label = "Hits per bin, mean over the module field" if coverage_mode else "Hits"
        fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)

        # A contour of a raw track outlines every sample; draw it on filled maps only.
        if hist.max() > 0 and (coverage_mode or smooth_sigma is not None):
            threshold_val = footprint_threshold * hist.max()
            ax.contour(
                ra_centers,
                dec_centers,
                hist.T,
                levels=[threshold_val],
                colors="blue",
                linewidths=1.5,
            )

        thresh_pct = int(stats_threshold * 100)
        if coverage_mode:
            footprint_area = np.count_nonzero(hist) * bin_area_deg2
            if hist.max() > 0:
                well_covered_area = (
                    np.count_nonzero(hist > stats_threshold * hist.max()) * bin_area_deg2
                )
            else:
                well_covered_area = 0.0
            stats_text = (
                f"$A_{{footprint}}$ = {footprint_area:.1f} coord deg$^2$\n"
                f"$A_{{>{thresh_pct}\\%max}}$ = {well_covered_area:.1f} coord deg$^2$"
            )
        else:
            nonzero_bins = int(np.count_nonzero(hist))
            total_bins = hist.size
            if hist.max() > 0:
                above_thresh = int(
                    np.count_nonzero(
                        hist > stats_threshold * hist.max(),
                    )
                )
                n_ratio = nonzero_bins / total_bins if total_bins > 0 else 0.0
                a_ratio = above_thresh / nonzero_bins if nonzero_bins > 0 else 0.0
            else:
                n_ratio = 0.0
                a_ratio = 0.0
            stats_text = (
                f"$N_{{footprint}}/N_{{total}}$ = {n_ratio:.2f}\n"
                f"$A_{{>{thresh_pct}\\%max}}/A_{{footprint}}$ = {a_ratio:.2f}"
            )

        ax.set_xlabel("Right Ascension")
        ax.set_ylabel("Declination")
        ax.set_title(label, fontweight="bold", loc="left")
        # Above the map rather than over it, so that no cell is hidden.
        ax.set_title(stats_text, fontsize="small", loc="right")
        ax.invert_xaxis()

        ra_step_deg = _ra_tick_step_deg(*ax.get_xlim())
        ax.xaxis.set_major_locator(ticker.MultipleLocator(ra_step_deg))
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(_format_ra_hms if ra_step_deg < 0.25 else _format_ra_hm)
        )
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_format_dec_deg))

    if own_fig:
        if title is not None:
            fig.suptitle(title)
        fig.tight_layout()
        if show:
            plt.show()
    elif title is not None:
        axes[0].set_title(title, fontweight="bold", loc="left")

    return fig
