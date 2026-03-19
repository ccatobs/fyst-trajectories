"""Plotting utilities for trajectory and coverage analysis.

This module provides visualization functions for analyzing telescope
scan coverage, including hit-density maps in equatorial coordinates
for multiple detector modules. Primarily meant for developer testing,
should not be considered a polished user-facing API.

Two modes of operation:

- **Detector-center track** (default, ``module_fov=None``): Plots the
  raw point-sample track of each detector center. Useful for verifying
  scan geometry and trajectory correctness. Reports fractional coverage
  statistics (N_footprint / N_total).

- **Module coverage** (``module_fov`` set): Convolves the track with a
  circular disk kernel representing the module's field of view. Useful
  for observation planning — shows approximate sky coverage. Reports
  absolute area statistics in square degrees.

These functions require ``matplotlib`` (install via
``pip install fyst-trajectories[plotting]``). Gaussian smoothing and
module footprint convolution require ``scipy``.

Examples
--------
Plot detector-center tracks for two PrimeCam modules:

>>> from fyst_trajectories import get_fyst_site
>>> from fyst_trajectories.primecam import get_primecam_offset
>>> from fyst_trajectories.plotting import plot_hit_map
>>> site = get_fyst_site()
>>> offsets = [
...     (get_primecam_offset("i1"), "f280"),
...     (get_primecam_offset("i6"), "f350"),
... ]
>>> fig = plot_hit_map(trajectory, offsets, site, show=True)

Plot with module footprint convolution for realistic coverage:

>>> fig = plot_hit_map(
...     trajectory,
...     offsets,
...     site,
...     module_fov=1.1,
...     show=True,
... )
"""

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .offsets import InstrumentOffset
    from .site import Site
    from .trajectory import Trajectory


def _make_disk_kernel(radius_bins: float) -> np.ndarray:
    """Create a 2D circular disk kernel for convolution.

    Parameters
    ----------
    radius_bins : float
        Radius of the disk in bin units.

    Returns
    -------
    ndarray
        2D array with 1.0 inside the disk and 0.0 outside,
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
    """Format RA in degrees as hour-angle notation."""
    deg = deg % 360
    h = deg / 15.0
    hours = int(h)
    minutes = abs(h - hours) * 60
    return f"{hours}$^{{h}}${minutes:02.0f}$^{{m}}$"


def _format_dec_deg(deg: float, _pos: Any) -> str:
    """Format Dec in degrees with degree symbol."""
    return f"{deg:.0f}$^\\circ$"


def plot_hit_map(
    trajectory: "Trajectory",
    offsets: list[tuple["InstrumentOffset", str]],
    site: "Site",
    *,
    bin_size: float = 0.02,
    module_fov: float | None = None,
    smooth_sigma: float | None = None,
    footprint_threshold: float = 0.1,
    stats_threshold: float = 0.5,
    cmap: str = "viridis",
    show: bool = True,
) -> Any:
    """Plot hit-density maps in RA/Dec for multiple detector modules.

    For each (offset, label) pair, computes the detector's sky track by
    applying the offset to the boresight trajectory, converts Az/El to
    RA/Dec, and bins into a 2D histogram.

    When ``module_fov`` is set, the histogram is convolved with a
    circular disk kernel representing the module's field of view,
    producing filled coverage maps suitable for observation planning.
    Statistics are reported as absolute areas in square degrees.

    When ``module_fov`` is None (default), the raw detector-center
    track is plotted. Statistics are reported as fractional coverage
    ratios, useful for verifying scan geometry.

    Parameters
    ----------
    trajectory : Trajectory
        Boresight trajectory with ``start_time`` set (needed for
        Az/El -> RA/Dec conversion).
    offsets : list of (InstrumentOffset, str)
        List of (offset, label) pairs. Each offset produces one panel.
        Use ``InstrumentOffset(dx=0, dy=0)`` for boresight.
    site : Site
        Telescope site configuration.
    bin_size : float, optional
        Histogram bin size in degrees for both RA and Dec. Default 0.02.
    module_fov : float or None, optional
        Module field-of-view diameter in degrees. If set, the histogram
        is convolved with a circular disk kernel of this diameter,
        approximating coverage from the full module. PrimeCam modules
        are approximately 1.1 degrees. Requires scipy. Default is None.
    smooth_sigma : float or None, optional
        If not None, apply Gaussian smoothing with this sigma (in bins)
        after any module FOV convolution. Requires scipy. Default is
        None (no smoothing).
    footprint_threshold : float, optional
        Fraction of max hit count to define the footprint boundary
        contour. Default 0.1 (10% of peak).
    stats_threshold : float, optional
        Fraction of max for the area efficiency statistic. Default 0.5.
    cmap : str, optional
        Matplotlib colormap name. Default "viridis".
    show : bool, optional
        Whether to call ``plt.show()``. Default True.

    Returns
    -------
    Figure
        The matplotlib figure with one panel per offset.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If trajectory has no ``start_time`` set.
    """
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        from matplotlib import ticker  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_hit_map(). "
            "Install it with: pip install fyst-trajectories[plotting]"
        ) from None

    if trajectory.start_time is None:
        raise ValueError("Trajectory must have start_time for RA/Dec conversion")

    # Deferred imports to avoid circular dependencies at module load time
    # pylint: disable=import-outside-toplevel
    from .coordinates import Coordinates
    from .offsets import boresight_to_detector, compute_focal_plane_rotation
    from .trajectory_utils import get_absolute_times
    # pylint: enable=import-outside-toplevel

    coords = Coordinates(site, atmosphere=None)
    abs_times = get_absolute_times(trajectory)

    # Compute parallactic angle if celestial context is available
    if trajectory.center_ra is not None and trajectory.center_dec is not None:
        ra_arr = np.full(len(trajectory.times), trajectory.center_ra)
        dec_arr = np.full(len(trajectory.times), trajectory.center_dec)
        pa = coords.get_parallactic_angle(ra_arr, dec_arr, obstime=abs_times)
    else:
        pa = np.zeros(len(trajectory.times))

    coverage_mode = module_fov is not None
    pad_deg = (module_fov / 2.0) if coverage_mode else 0.0
    bin_area_deg2 = bin_size * bin_size

    n_panels = len(offsets)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(6 * n_panels, 5),
        squeeze=False,
    )
    axes = axes[0]

    for ax, (offset, label) in zip(axes, offsets):
        # Compute field rotation for this offset
        fr = compute_focal_plane_rotation(
            trajectory.el,
            site,
            offset,
            parallactic_angle=pa,
        )

        # Compute detector Az/El track
        det_az, det_el = boresight_to_detector(
            trajectory.az,
            trajectory.el,
            offset,
            fr,
        )

        # Convert to RA/Dec
        ra, dec = coords.altaz_to_radec(det_az, det_el, obstime=abs_times)

        # Handle RA wrapping around 0/360
        ra = np.asarray(ra, dtype=float)
        dec = np.asarray(dec, dtype=float)
        if ra.max() - ra.min() > 180:
            ra = (ra + 180) % 360 - 180

        # 2D histogram with padding for FOV convolution
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

        # Module FOV convolution (disk kernel)
        if coverage_mode:
            try:
                from scipy.signal import fftconvolve  # pylint: disable=import-outside-toplevel
            except ImportError:
                raise ImportError(
                    "scipy is required for module_fov convolution. "
                    "Install it with: pip install fyst-trajectories[plotting]"
                ) from None
            radius_bins = (module_fov / 2.0) / bin_size
            kernel = _make_disk_kernel(radius_bins)
            hist = fftconvolve(hist, kernel, mode="same")
            # Clean FFT artifacts: clip negatives and zero out noise floor
            np.maximum(hist, 0.0, out=hist)
            hist[hist < 1e-10] = 0.0

        # Optional Gaussian smoothing
        if smooth_sigma is not None:
            try:
                from scipy.ndimage import gaussian_filter  # pylint: disable=import-outside-toplevel
            except ImportError:
                raise ImportError(
                    "scipy is required for smooth_sigma. "
                    "Install it with: pip install fyst-trajectories[plotting]"
                ) from None
            hist = gaussian_filter(hist, sigma=smooth_sigma)

        # Plot
        ra_centers = 0.5 * (ra_edges[:-1] + ra_edges[1:])
        dec_centers = 0.5 * (dec_edges[:-1] + dec_edges[1:])
        im = ax.pcolormesh(
            ra_centers,
            dec_centers,
            hist.T,
            cmap=cmap,
            shading="auto",
        )
        fig.colorbar(im, ax=ax, label="Hits", shrink=0.8)

        # Footprint contour
        if hist.max() > 0:
            threshold_val = footprint_threshold * hist.max()
            ax.contour(
                ra_centers,
                dec_centers,
                hist.T,
                levels=[threshold_val],
                colors="blue",
                linewidths=1.5,
            )

        # Statistics overlay
        thresh_pct = int(stats_threshold * 100)
        if coverage_mode:
            # Coverage mode: report absolute area in deg^2
            footprint_area = np.count_nonzero(hist) * bin_area_deg2
            if hist.max() > 0:
                well_covered_area = (
                    np.count_nonzero(hist > stats_threshold * hist.max()) * bin_area_deg2
                )
            else:
                well_covered_area = 0.0
            stats_text = (
                f"$A_{{footprint}}$ = {footprint_area:.1f} deg$^2$\n"
                f"$A_{{>{thresh_pct}\\%max}}$ = {well_covered_area:.1f} deg$^2$"
            )
        else:
            # Track mode: report fractional ratios
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
                f"$A_{{>{thresh_pct}\\%max}}/A_{{total}}$ = {a_ratio:.2f}"
            )

        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        ax.set_xlabel("pos.eq.ra")
        ax.set_ylabel("pos.eq.dec")
        ax.set_title(label, fontweight="bold", loc="left")
        ax.invert_xaxis()

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_format_ra_hm))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_format_dec_deg))

    fig.tight_layout()

    if show:
        plt.show()

    return fig
