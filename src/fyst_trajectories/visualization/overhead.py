"""Night-level visualization for overhead timelines.

Renders :class:`~fyst_trajectories.overhead.ObservingTimeline` objects (from
:func:`~fyst_trajectories.overhead.generate_timeline` or
:func:`~fyst_trajectories.overhead.read_timeline`) as matplotlib figures:

- :func:`plot_timeline_gantt`: one lane per science patch and calibration
  type (plus slew and idle), one bar per timeline block.
- :func:`plot_sky_coverage`: the science scans' az/el samples converted to
  RA/Dec through the site, one point track per scan.

These functions require ``matplotlib`` (install via
``pip install fyst-trajectories[plotting]``). They never call
``matplotlib.use()``; backend selection is the caller's business.

Pass ``ax=`` to compose a panel into an existing figure: the function then
draws into that axes and returns ``ax.get_figure()``, and neither
``plt.show()`` nor ``fig.tight_layout()`` is invoked (both apply only to
figures the function itself creates, so a caller's half-built composition
is never popped or re-laid-out).

Examples
--------
Render a recorded night from its ECSV timeline:

>>> from fyst_trajectories.overhead import read_timeline
>>> from fyst_trajectories.visualization import (
...     plot_sky_coverage,
...     plot_timeline_gantt,
... )
>>> timeline = read_timeline("timeline.ecsv")  # doctest: +SKIP
>>> fig = plot_timeline_gantt(timeline, show=False)  # doctest: +SKIP
>>> fig.savefig("night_gantt.png", dpi=140, bbox_inches="tight")  # doctest: +SKIP
>>> fig = plot_sky_coverage(timeline, show=False)  # doctest: +SKIP
>>> fig.savefig("sky_coverage.png", dpi=140, bbox_inches="tight")  # doctest: +SKIP
"""

import math
from typing import TYPE_CHECKING

import numpy as np

from ..coordinates import Coordinates
from ..trajectory_utils import get_absolute_times

# The simulator tier (fyst_trajectories.overhead) is imported inside the
# functions that draw its timelines, so importing this subpackage for the
# library-tier plots never loads it (tests/test_tier_boundary.py).
if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ..overhead.models import ObservingTimeline, TimelineBlock
    from ..planning import ScanBlock

__all__ = [
    "plot_sky_coverage",
    "plot_timeline_gantt",
]

#: Colors for science-patch lanes and sky tracks, cycled in patch
#: first-appearance order.
SCIENCE_COLORS = ["#1f77b4", "#2ca02c", "#17becf", "#393b79", "#8c564b"]

#: Colors for calibration lanes, keyed by the block's ``scan_type``.
CAL_COLORS = {
    "retune": "#d62728",
    "pointing_cal": "#ff7f0e",
    "focus": "#e377c2",
    "skydip": "#bcbd22",
    "planet_cal": "#9467bd",
    "beam_map": "#7f7f7f",
}

SLEW_COLOR = "#555555"
IDLE_COLOR = "#cccccc"

#: Fallback colors for unknown calibration types / patches missing a color.
_UNKNOWN_CAL_COLOR = "#637939"
_UNKNOWN_PATCH_COLOR = "#333333"

#: Default figure sizes (inches) when the function creates its own figure.
GANTT_FIGSIZE = (15.0, 6.2)
SKY_FIGSIZE = (9.0, 5.9)


def _lanes_for(timeline: "ObservingTimeline") -> tuple[list, dict[str, str]]:
    """Group blocks into ordered gantt lanes and derive the patch color map.

    Lane order: science patches in first-appearance order, calibration types
    in :class:`~fyst_trajectories.overhead.CalibrationType` member order
    (unknown types appended in first-appearance order), then slew and idle
    (always present, possibly empty).

    Parameters
    ----------
    timeline : ObservingTimeline
        The timeline whose blocks are grouped.

    Returns
    -------
    lanes : list of (str, str, str, list[TimelineBlock])
        Ordered ``(label, color, kind, blocks)`` lane tuples, where ``kind``
        is one of ``"science"``, ``"calibration"``, ``"slew"``, ``"idle"``.
    patch_color : dict of str to str
        Science patch name to lane/track color.
    """
    from ..overhead.models import BlockType, CalibrationType

    cal_order = [c.value for c in CalibrationType]

    sci_order: list[str] = []
    cal_seen: list[str] = []
    groups: dict[tuple[str, str], list] = {}
    for block in timeline.blocks:
        if block.block_type == BlockType.SCIENCE:
            key = ("science", block.patch_name)
            if block.patch_name not in sci_order:
                sci_order.append(block.patch_name)
        elif block.block_type == BlockType.CALIBRATION:
            key = ("calibration", block.scan_type)
            if block.scan_type not in cal_seen:
                cal_seen.append(block.scan_type)
        elif block.block_type == BlockType.SLEW:
            key = ("slew", "")
        else:
            key = ("idle", "")
        groups.setdefault(key, []).append(block)

    cal_types = [c for c in cal_order if c in cal_seen]
    cal_types += [c for c in cal_seen if c not in cal_order]
    patch_color = {p: SCIENCE_COLORS[i % len(SCIENCE_COLORS)] for i, p in enumerate(sci_order)}

    lanes = [(p, patch_color[p], "science", groups[("science", p)]) for p in sci_order]
    lanes += [
        (c, CAL_COLORS.get(c, _UNKNOWN_CAL_COLOR), "calibration", groups[("calibration", c)])
        for c in cal_types
    ]
    lanes.append(("slew", SLEW_COLOR, "slew", groups.get(("slew", ""), [])))
    lanes.append(("idle", IDLE_COLOR, "idle", groups.get(("idle", ""), [])))
    return lanes, patch_color


def _wrap_ra(ra_deg: np.ndarray) -> np.ndarray:
    """Wrap RA to [-180, 180) degrees so fields near RA 0 plot contiguously."""
    return ((np.asarray(ra_deg, dtype=float) + 180.0) % 360.0) - 180.0


def _radec_track(
    scan_block: "ScanBlock", coords: Coordinates, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a rebuilt scan's az/el samples to a wrapped RA/Dec track.

    Parameters
    ----------
    scan_block : ScanBlock
        A rebuilt scan whose ``trajectory`` carries ``start_time``.
    coords : Coordinates
        The site's coordinate transformer (vacuum by default, matching the
        vacuum trajectories the overhead layer plans).
    stride : int
        Subsample stride over the trajectory samples.

    Returns
    -------
    ra, dec : ndarray
        Wrapped RA and Dec in degrees.
    """
    trajectory = scan_block.trajectory
    if trajectory.start_time is None:
        raise ValueError(
            "plot_sky_coverage: a scan trajectory has no start_time; "
            "absolute times are required for the az/el to RA/Dec conversion."
        )
    s = slice(None, None, stride)
    obstime = get_absolute_times(trajectory)[s]
    ra, dec = coords.altaz_to_radec(trajectory.az[s], trajectory.el[s], obstime)
    return _wrap_ra(ra), np.asarray(dec, dtype=float)


def plot_timeline_gantt(
    timeline: "ObservingTimeline",
    *,
    title: str | None = None,
    ax: "Axes | None" = None,
    show: bool = True,
) -> "Figure":
    """Plot an observing night as a gantt chart, one lane per block group.

    Science patches get one lane each (first-appearance order), calibration
    types one lane each (canonical order: retune, pointing_cal, focus,
    skydip, planet_cal, beam_map, then unknowns), followed by slew and idle
    lanes. The x-axis is UTC hours counted from the start date's midnight;
    tick labels wrap modulo 24 so a night crossing UTC midnight reads
    23:00, 00:00, 01:00.

    The lane band geometry is a stable contract for callers that annotate
    the returned axes: y-ticks sit at lane centers (``y + 0.5``) labelled
    with the lane name, bars span ``y + 0.1`` to ``y + 0.9``, and the first
    lane is drawn on top.

    The legend hangs below the axes, beneath the time-axis label, at a
    fixed distance in points, so it stays clear of the label at any figure
    height.

    Parameters
    ----------
    timeline : ObservingTimeline
        The timeline to render. Must contain at least one block.
    title : str, optional
        Axes title. Default is an auto-generated summary on two lines: the
        night (site name and start date) on the first, the numbers on the
        second (duration, block counts, efficiency from
        :func:`~fyst_trajectories.overhead.compute_budget`); a night with
        no science and at least one planet-calibration pass is summarised
        as a calibration night instead (passes, detector operations, the
        fraction of the night on source).
    ax : matplotlib.axes.Axes, optional
        Draw into this axes instead of creating a new figure. When given,
        ``show`` is ignored and no layout call is made on the caller's
        figure, so the caller's layout must leave room below the axes for
        the legend (``fig.tight_layout()`` counts it).
    show : bool, optional
        Call ``plt.show()`` after rendering (only when the function created
        the figure). Default True.

    Returns
    -------
    Figure
        The figure containing the gantt (``ax.get_figure()`` when ``ax``
        was supplied).

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If the timeline has no blocks.
    """
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        from matplotlib.font_manager import (  # pylint: disable=import-outside-toplevel
            FontProperties,
        )
        from matplotlib.patches import Patch  # pylint: disable=import-outside-toplevel
        from matplotlib.transforms import offset_copy  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_timeline_gantt(). "
            "Install it with: pip install fyst-trajectories[plotting]"
        ) from None

    if not timeline.blocks:
        raise ValueError("plot_timeline_gantt: timeline has no blocks to plot.")

    lanes, _ = _lanes_for(timeline)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=GANTT_FIGSIZE)
    else:
        fig = ax.get_figure()

    day0 = math.floor(timeline.start_time.mjd)

    def utc_hours(t):
        return (t.mjd - day0) * 24.0

    n = len(lanes)
    for row, (_label, color, _kind, blocks) in enumerate(lanes):
        y = n - 1 - row  # first lane on top
        if blocks:
            bars = [(utc_hours(b.t_start), b.duration / 3600.0) for b in blocks]
            ax.broken_barh(bars, (y + 0.1, 0.8), facecolors=color, edgecolor="black", linewidth=0.4)

    ax.set_yticks([n - 1 - row + 0.5 for row in range(n)])
    ax.set_yticklabels([label for label, _, _, _ in lanes], fontsize=10)
    ax.set_ylim(-0.1, n + 0.1)

    x0, x1 = utc_hours(timeline.start_time), utc_hours(timeline.end_time)
    # Whole hours for a full night, quarter or half hours for a short one. The
    # limits are set after the ticks: set_xticks widens the view to its ticks.
    step = next((s for s in (0.25, 0.5, 1.0, 2.0) if (x1 - x0) / s <= 12.0), 4.0)
    # The 1e-6 h slack absorbs the MJD round trip, so a whole-hour start keeps its tick.
    ticks = np.arange(math.ceil(x0 / step - 1e-6) * step, x1 + 1e-6, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(h // 1) % 24:02d}:{round(h % 1 * 60):02d}" for h in ticks])
    ax.set_xlim(x0, x1)
    night_label = timeline.start_time.iso[:10]
    xlabel_pt = 11.0
    ax.set_xlabel(f"Time (UTC) on {night_label}", fontsize=xlabel_pt)
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.set_axisbelow(True)

    if title is None:
        from ..overhead.simulation import compute_budget

        budget = compute_budget(timeline)
        hours = (timeline.end_time - timeline.start_time).sec / 3600.0
        site_name = timeline.site.name.strip()
        passes = [b for b in timeline.blocks if b.scan_type == "planet_cal"]
        if budget["n_science_scans"] == 0 and passes:
            # A calibration night: its yield is time on source, not science.
            on_source = sum(b.duration for b in passes) / max(timeline.total_time, 1e-9)
            head = f"{site_name} calibration night" if site_name else "Calibration night"
            title = (
                f"{head} {night_label}\n"
                f"{hours:.1f} h, "
                f"{len(passes)} passes, "
                f"{budget['n_calibration_blocks'] - len(passes)} detector operations, "
                f"on source {on_source * 100:.1f}%"
            )
        else:
            head = f"{site_name} observing night" if site_name else "Observing night"
            title = (
                f"{head} {night_label}\n"
                f"{hours:.1f} h, "
                f"{budget['n_science_scans']} science scans, "
                f"{budget['n_calibration_blocks']} calibrations, "
                f"efficiency {budget['efficiency'] * 100:.1f}%"
            )
    ax.set_title(title, fontsize=13)

    legend_labels = {"science": "science: {}", "calibration": "cal: {}"}
    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor="black",
            label=legend_labels.get(kind, "{}").format(label),
        )
        for label, color, kind, _blocks in lanes
    ]
    # The tick labels and the time-axis label have fixed heights in points, so
    # the legend is anchored that many points below the axes rather than at a
    # fraction of the axes height, which lands on the label in a short axes.
    rc = plt.rcParams
    tick_label_pt = FontProperties(size=rc["xtick.labelsize"]).get_size_in_points()
    below_pt = (
        max(rc["xtick.major.size"], 0.0)
        + rc["xtick.major.pad"]
        + 1.2 * tick_label_pt
        + rc["axes.labelpad"]
        + 1.2 * xlabel_pt
    )
    ax.legend(
        handles=legend_handles,
        ncol=4,
        fontsize=8.5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        bbox_transform=offset_copy(ax.transAxes, fig=fig, y=-below_pt, units="points"),
        frameon=False,
    )

    if own_fig:
        fig.tight_layout()
        if show:
            plt.show()
    return fig


def plot_sky_coverage(
    timeline: "ObservingTimeline",
    *,
    pairs: "list[tuple[TimelineBlock, ScanBlock]] | None" = None,
    stride: int = 12,
    title: str | None = None,
    legend_loc: str = "upper right",
    ax: "Axes | None" = None,
    show: bool = True,
) -> "Figure":
    """Plot the science scans' sky coverage in RA/Dec.

    Each rebuilt science scan's az/el samples are converted to RA/Dec
    through the timeline's site (vacuum transform, matching the vacuum
    trajectories the overhead layer plans) and drawn as a point track
    colored by science patch. RA is wrapped to [-180, 180) degrees (kept in
    [0, 360) when the scans straddle RA 180) and the x-axis is inverted (RA
    increases to the left); the aspect is equal.

    Parameters
    ----------
    timeline : ObservingTimeline
        The timeline whose science scans are rendered.
    pairs : list of (TimelineBlock, ScanBlock), optional
        Pre-rebuilt scan pairs (for example from
        :func:`~fyst_trajectories.overhead.schedule_to_trajectories`) to
        plot instead of rebuilding internally; pass these to avoid a second
        reconstruction when the caller already has them. Default None
        rebuilds the science blocks internally (warnings from the
        reconstruction propagate to the caller).
    stride : int, optional
        Subsample stride over each trajectory's samples before the RA/Dec
        conversion. Default 12.
    title : str, optional
        Axes title. Default is an auto-generated summary.
    legend_loc : str, optional
        Legend location. Default ``"upper right"``.
    ax : matplotlib.axes.Axes, optional
        Draw into this axes instead of creating a new figure. When given,
        ``show`` is ignored and no layout call is made on the caller's
        figure.
    show : bool, optional
        Call ``plt.show()`` after rendering (only when the function created
        the figure). Default True.

    Returns
    -------
    Figure
        The figure containing the sky plot (``ax.get_figure()`` when ``ax``
        was supplied).

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If ``stride < 1``, if there are no science trajectories to plot, or
        if a trajectory carries no ``start_time``.
    """
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_sky_coverage(). "
            "Install it with: pip install fyst-trajectories[plotting]"
        ) from None

    if stride < 1:
        raise ValueError(f"plot_sky_coverage: stride must be >= 1, got {stride}.")
    if pairs is None:
        from ..overhead.simulation import schedule_to_trajectories

        pairs = schedule_to_trajectories(timeline)
    if not pairs:
        raise ValueError(
            "plot_sky_coverage: no science trajectories to plot (the pairs list is "
            "empty; schedule_to_trajectories skips blocks it cannot reconstruct)."
        )

    _, patch_color = _lanes_for(timeline)
    coords = Coordinates(timeline.site)

    # Convert every track BEFORE creating the figure so a bad trajectory
    # (for example one without start_time) cannot leak a half-built figure.
    tracks = [
        (block.patch_name, _radec_track(scan_block, coords, stride)) for block, scan_block in pairs
    ]

    # Wrapping about 0 keeps fields near RA 0 contiguous but splits scans that
    # straddle RA 180 across the whole axis; keep [0, 360) for such a night.
    all_ra = np.concatenate([ra for _, (ra, _) in tracks])
    straddles_180 = np.ptp(all_ra % 360.0) <= 180.0 < np.ptp(all_ra)
    if straddles_180:
        tracks = [(name, (ra % 360.0, dec)) for name, (ra, dec) in tracks]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=SKY_FIGSIZE)
    else:
        fig = ax.get_figure()

    seen: dict[str, int] = {}
    for patch_name, (ra, dec) in tracks:
        color = patch_color.get(patch_name, _UNKNOWN_PATCH_COLOR)
        seen[patch_name] = seen.get(patch_name, 0) + 1
        ax.plot(ra, dec, ".", ms=1.6, color=color, alpha=0.5)

    ra_units = "[deg]" if straddles_180 else "[deg, wrapped about 0]"
    ax.set_xlabel(f"Right Ascension  {ra_units}", fontsize=11)
    ax.set_ylabel("Declination  [deg]", fontsize=11)
    ax.set_aspect("equal")
    if not ax.xaxis_inverted():
        ax.invert_xaxis()  # RA increases to the left
    ax.grid(ls=":", alpha=0.4)

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=7,
            color=patch_color.get(patch, _UNKNOWN_PATCH_COLOR),
            label=f"{patch}  ({count} scans)",
        )
        for patch, count in seen.items()
    ]
    ax.legend(handles=handles, fontsize=9.5, loc=legend_loc, markerscale=1.0)

    if title is None:
        title = (
            f"Science-scan sky coverage (az/el -> RA/Dec via {timeline.site.name} site)\n"
            f"observing night {timeline.start_time.iso[:10]}, equal aspect"
        )
    ax.set_title(title, fontsize=12.5)

    if own_fig:
        fig.tight_layout()
        if show:
            plt.show()
    return fig
