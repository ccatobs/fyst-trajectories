"""Regenerate the documentation figures in ``docs/figures/``.

Run from the repository root with the ``dev`` extra installed::

    python docs/make_figures.py                      # every figure
    python docs/make_figures.py --only pong_scan     # one figure
    python docs/make_figures.py --out-dir /tmp/figs  # somewhere else

Each figure calls the library's own plotting functions on the example the
page beside it already runs, so the picture and the text cannot disagree
about the inputs. The output is deterministic for a given matplotlib and
FreeType: fixed figure sizes and resolution, pinned style settings, no
timestamp or software tag in the PNG metadata. The sky-view figure needs the
shared sun-avoidance library for its CAD panel and fails loudly without it.

This file is a maintainer script, not part of the package: Sphinx reads only
``.rst`` sources, so it is never rendered as a page, and nothing under
``src/`` imports it.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from astropy.time import Time  # noqa: E402

from fyst_trajectories import PRIMECAM_MODULES, InstrumentOffset, get_fyst_site  # noqa: E402
from fyst_trajectories.planning import (  # noqa: E402
    FieldRegion,
    plan_constant_el_scan,
    plan_daisy_scan,
    plan_pong_scan,
    plan_source_ces,
)
from fyst_trajectories.primecam import get_primecam_offset  # noqa: E402
from fyst_trajectories.visualization import (  # noqa: E402
    plot_array_footprint,
    plot_hit_map,
    plot_sky_view,
    plot_source_track,
    plot_timeline_gantt,
    plot_trajectory,
)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"

#: Figure width in inches. The Read the Docs content column is about 700 to
#: 800 CSS px wide, so a figure drawn 8 in wide and shown at full column width
#: keeps its type close to the point size the plotting code asks for.
PAGE_WIDTH = 8.0

#: One resolution for every figure (1600 px across a full-width figure, so
#: the page stays sharp on a high-density display).
DPI = 200

#: Style settings that would otherwise follow a user's matplotlibrc.
RC_PARAMS = {
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "axes.titlesize": 11.0,
    "axes.labelsize": 10.0,
    "lines.linewidth": 1.0,
    "image.cmap": "viridis",
}

#: PNG metadata: drop the ``Software`` tag so a matplotlib upgrade alone
#: does not rewrite every committed file.
PNG_METADATA = {"Software": None}


@dataclass(frozen=True)
class FigureSpec:
    """One registered figure.

    Attributes
    ----------
    name : str
        Figure name; the PNG is ``<name>.png``.
    size_in : tuple of float
        Figure size in inches, width then height.
    build : callable
        Builder taking ``size_in`` and returning the drawn figure at that size.
    needs_sun_avoidance : bool
        Whether the builder needs the shared sun-avoidance library.
    """

    name: str
    size_in: tuple[float, float]
    build: Callable[[tuple[float, float]], plt.Figure]
    needs_sun_avoidance: bool = False

    @property
    def filename(self) -> str:
        """The PNG file name under ``docs/figures/``."""
        return f"{self.name}.png"

    @property
    def size_px(self) -> tuple[int, int]:
        """The PNG's pixel size, width then height."""
        return round(self.size_in[0] * DPI), round(self.size_in[1] * DPI)


def _fit(fig: plt.Figure, size_in: tuple[float, float]) -> plt.Figure:
    """Resize a figure the plotting function sized itself and lay it out again.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure drawn at the plotting function's own size.
    size_in : tuple of float
        Target size in inches, width then height.

    Returns
    -------
    matplotlib.figure.Figure
        The same figure, resized.
    """
    fig.set_size_inches(*size_in)
    fig.tight_layout()
    return fig


def pong_scan(size_in: tuple[float, float]) -> plt.Figure:
    """Draw one Pong period over a 2 x 2 deg field, in az/el and on sky.

    Placed on ``planning.rst``, Quick Start. The three trajectory panels sit
    above the boresight's hit map, lightly smoothed so that one period reads
    as coverage rather than as a speckle of single hits.
    """
    site = get_fyst_site()
    field = FieldRegion(ra_center=180.0, dec_center=-30.0, width=2.0, height=2.0)
    block = plan_pong_scan(
        field=field,
        velocity=0.4,
        spacing=0.1,
        num_terms=4,
        site=site,
        start_time=Time("2026-03-15T01:00:00", scale="utc"),
        timestep=0.1,
    )
    fig = plt.figure(figsize=size_in, layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.0, 1.4])
    plot_trajectory(block.trajectory, axes=top.subplots(1, 3))
    plot_hit_map(
        block.trajectory,
        modules={"boresight on sky": InstrumentOffset(dx=0.0, dy=0.0)},
        site=site,
        fov_radius_deg=None,
        bin_size=0.02,
        smooth_sigma=1.5,
        # Below the deepest dip inside the field (about 9 % of the peak), so the footprint
        # contour does not ring the gaps one period leaves as stray dots.
        footprint_threshold=0.05,
        axes=[bottom.subplots(1, 1)],
    )
    return fig


def daisy_scan(size_in: tuple[float, float]) -> plt.Figure:
    """Draw five minutes of daisy on the Crab, on sky.

    Placed on ``planning.rst``, Planning a Daisy Scan.
    """
    site = get_fyst_site()
    block = plan_daisy_scan(
        ra=83.633,
        dec=22.014,
        radius=0.5,
        velocity=0.3,
        turn_radius=0.2,
        avoidance_radius=0.0,
        start_acceleration=0.5,
        site=site,
        start_time=Time("2026-01-15T02:00:00", scale="utc"),
        timestep=0.1,
        duration=300.0,
    )
    fig = plot_hit_map(
        block.trajectory,
        modules={"boresight": InstrumentOffset(dx=0.0, dy=0.0)},
        site=site,
        fov_radius_deg=None,
        bin_size=0.01,
        smooth_sigma=1.0,
        show=False,
    )
    return _fit(fig, size_in)


def constant_el_scan(size_in: tuple[float, float]) -> plt.Figure:
    """Draw the rising constant-elevation crossing, on sky.

    Placed on ``planning.rst``, Planning a Constant-Elevation Scan.
    """
    site = get_fyst_site()
    field = FieldRegion(ra_center=0.0, dec_center=-2.0, width=10.0, height=6.0)
    block = plan_constant_el_scan(
        field=field,
        elevation=45.0,
        velocity=0.5,
        site=site,
        start_time="2026-09-15T00:00:00",
        rising=True,
    )
    fig = plot_hit_map(
        block.trajectory,
        modules={"centre module, 1.3 deg field of view": get_primecam_offset("c")},
        site=site,
        bin_size=0.05,
        show=False,
    )
    return _fit(fig, size_in)


def source_ces_track(size_in: tuple[float, float]) -> plt.Figure:
    """Draw Jupiter rising across all seven modules.

    Placed on ``planning.rst``, Planning a Source CES.
    """
    site = get_fyst_site()
    modules = [PRIMECAM_MODULES[k] for k in ("c", "i1", "i2", "i3", "i4", "i5", "i6")]
    block = plan_source_ces(
        body="jupiter",
        footprint=modules,
        el_bore=35.0,
        night=Time("2026-03-15T00:00:00", scale="utc"),
        mode="rising",
        site=site,
    )
    fig, ax = plt.subplots(figsize=size_in)
    plot_source_track(block, site=site, ax=ax)
    fig.tight_layout()
    return fig


def primecam_footprint(size_in: tuple[float, float]) -> plt.Figure:
    """Draw the array on sky at el 30 and el 70.

    Placed on ``instrument_offsets.rst``, beside the mechanical rotation. The
    titles take the rotation's sign from the site's Nasmyth port, so the
    figure follows the port when it is regenerated.
    """
    nasmyth_sign = get_fyst_site().nasmyth_sign
    fig, axes = plt.subplots(1, 2, figsize=size_in)
    for ax, el in zip(axes, (30.0, 70.0), strict=True):
        rotation = nasmyth_sign * el
        title = f"el = {el:.0f} deg: Nasmyth rotation {rotation:+.0f} deg"
        plot_array_footprint(el=el, title=title, ax=ax)
    fig.tight_layout()
    return fig


def sky_view_sun_zone(size_in: tuple[float, float]) -> plt.Figure:
    """Draw the scalar Sun circle beside the CAD zone.

    Placed on ``sun_avoidance.rst``, Seeing the zone.
    """
    fig, axes = plt.subplots(1, 2, figsize=size_in, subplot_kw={"projection": "polar"})
    when = Time("2026-11-15T18:00:00", scale="utc")
    step = 0.5  # finer than the 1.5 deg default: sharper CAD zone corners at page width
    plot_sky_view(
        when, boresight="moon", grid_step_deg=step, title="Default scalar policy", ax=axes[0]
    )
    plot_sky_view(
        when,
        sun_model="cad",
        boresight="moon",
        grid_step_deg=step,
        title='sun_model="cad"',
        ax=axes[1],
    )
    # At page width the legend would cover a quarter of each chart; redraw the
    # same entries below it.
    for ax in axes:
        legend = ax.get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        ax.legend(
            legend.legend_handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=2,
            fontsize=8,
            columnspacing=1.0,
            handlelength=1.5,
            frameon=False,
        )
    fig.tight_layout()
    return fig


def night_gantt(size_in: tuple[float, float]) -> plt.Figure:
    """Draw the 8-hour, two-patch simulated night.

    Placed on ``overhead_quickstart.rst``, Basic Usage.
    """
    from fyst_trajectories.overhead import ObservingPatch, generate_timeline

    site = get_fyst_site()
    patches = [
        ObservingPatch(
            name="Deep56",
            ra_center=24.0,
            dec_center=-32.0,
            width=40.0,
            height=10.0,
            scan_type="constant_el",
            velocity=1.0,
            elevation=50.0,
        ),
        ObservingPatch(
            name="Wide01",
            ra_center=180.0,
            dec_center=-30.0,
            width=4.0,
            height=4.0,
            scan_type="pong",
            velocity=0.5,
        ),
    ]
    timeline = generate_timeline(
        patches=patches,
        site=site,
        start_time="2026-06-15T02:00:00",
        end_time="2026-06-15T10:00:00",
    )
    title = "The two-patch night of 2026-06-15 (UTC)"
    fig = plot_timeline_gantt(timeline, title=title, show=False)
    return _fit(fig, size_in)


def calibration_night(size_in: tuple[float, float]) -> plt.Figure:
    """Draw 45 minutes of calibration passes on Saturn and Uranus.

    Placed on ``overhead_calibration_night.rst``, Quick Start.
    """
    from fyst_trajectories.overhead import plan_calibration_night

    site = get_fyst_site()
    timeline = plan_calibration_night(
        ["saturn", "uranus"],
        site,
        "2026-09-11T06:30:00",
        "2026-09-11T07:15:00",
    )
    fig = plot_timeline_gantt(
        timeline, title="The calibration night of 2026-09-11 (UTC)", show=False
    )
    return _fit(fig, size_in)


#: Every figure the docs reference, in page order.
REGISTRY: tuple[FigureSpec, ...] = (
    FigureSpec("pong_scan", (PAGE_WIDTH, 6.4), pong_scan),
    FigureSpec("daisy_scan", (5.6, 4.8), daisy_scan),
    FigureSpec("constant_el_scan", (PAGE_WIDTH, 5.2), constant_el_scan),
    FigureSpec("source_ces_track", (5.6, 5.6), source_ces_track),
    FigureSpec("primecam_footprint", (PAGE_WIDTH, 4.2), primecam_footprint),
    FigureSpec(
        "sky_view_sun_zone",
        (PAGE_WIDTH, 5.0),
        sky_view_sun_zone,
        needs_sun_avoidance=True,
    ),
    FigureSpec("night_gantt", (PAGE_WIDTH, 4.8), night_gantt),
    FigureSpec("calibration_night", (PAGE_WIDTH, 4.8), calibration_night),
)


def render(spec: FigureSpec, out_dir: Path) -> Path:
    """Build one figure and write it at its registered size.

    Parameters
    ----------
    spec : FigureSpec
        The registered figure to build.
    out_dir : pathlib.Path
        Existing directory to write the PNG into.

    Returns
    -------
    pathlib.Path
        The written file, ``out_dir / spec.filename``.
    """
    with plt.rc_context(RC_PARAMS):
        fig = spec.build(spec.size_in)
        path = out_dir / spec.filename
        fig.savefig(path, dpi=DPI, metadata=PNG_METADATA)
        plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    """Run the command line.

    Parameters
    ----------
    argv : list of str, optional
        Arguments without the program name; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        The process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        action="append",
        choices=[spec.name for spec in REGISTRY],
        help="build only this figure (repeatable)",
    )
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = [spec for spec in REGISTRY if not args.only or spec.name in args.only]
    for spec in selected:
        start = time.perf_counter()
        path = render(spec, args.out_dir)
        elapsed = time.perf_counter() - start
        print(f"{path.name:28s} {elapsed:6.1f} s  {path.stat().st_size / 1024:7.1f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
