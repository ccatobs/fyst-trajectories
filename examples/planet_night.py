"""Plan one night of planet calibration passes (offline planning).

This runnable example plans a commissioning night with
:func:`~fyst_trajectories.overhead.plan_calibration_night` at its default
policy, prints the summary and the dispatch sheet, writes the night
as a TOAST-compatible ECSV timeline, and, when matplotlib is installed,
saves the timeline gantt and the focal-plane track of the first pass. The
planned night is not a schedule a control system executes on its own; the
dispatch sheet is what a person dispatches from.

Run it from the repository root::

    python examples/planet_night.py
    python examples/planet_night.py 2026-09-10T22:00:00 2026-09-11T10:30:00 output_dir

With no arguments the example plans a two-hour window, which takes about
a minute; the full night from dusk to dawn takes a few minutes.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

from fyst_trajectories import PointingWarning, get_fyst_site
from fyst_trajectories.overhead import (
    dispatch_sheet,
    plan_calibration_night,
    schedule_to_trajectories,
    summarize_calibration_night,
    write_timeline,
)

TARGETS = ("jupiter", "saturn", "neptune", "uranus")
DEFAULT_START = "2026-09-11T06:00:00"
DEFAULT_END = "2026-09-11T08:00:00"


def save_figures(timeline, site, out_dir: Path) -> list[Path]:
    """Save the gantt and the first pass's track when matplotlib is available.

    Parameters
    ----------
    timeline : ObservingTimeline
        The planned night.
    site : Site
        The observing site.
    out_dir : Path
        Directory to write the figures into.

    Returns
    -------
    list of Path
        The figures written: the timeline gantt, plus the track of the
        first pass that rebuilds, when the night holds one. Empty when
        matplotlib is not installed.
    """
    try:
        from fyst_trajectories.visualization import plot_source_track, plot_timeline_gantt
    except ImportError:
        return []
    written: list[Path] = []
    try:
        fig = plot_timeline_gantt(timeline, show=False)
    except ImportError:
        return []
    gantt = out_dir / "planet_night_gantt.png"
    fig.savefig(gantt, dpi=120)
    written.append(gantt)
    # The rebuild re-runs the kernel, whose advisories the summary already
    # lists; keep the example's output to the planner's own report.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PointingWarning)
        pairs = schedule_to_trajectories(timeline, science_only=False)
    if pairs:
        fig = plot_source_track(pairs[0][1], site=site, show=False)
        track = out_dir / "planet_night_first_pass.png"
        fig.savefig(track, dpi=120)
        written.append(track)
    return written


def main(argv: list[str] | None = None) -> int:
    """Plan the night, print the summary and the sheet, write the artifacts.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments: optional ``start end`` ISO UTC times and an
        optional output directory (a temporary directory by default).

    Returns
    -------
    int
        Process exit status (0 on success).
    """
    argv = sys.argv[1:] if argv is None else argv
    start = argv[0] if len(argv) > 0 else DEFAULT_START
    end = argv[1] if len(argv) > 1 else DEFAULT_END
    out_dir = Path(argv[2]) if len(argv) > 2 else None

    site = get_fyst_site()
    timeline = plan_calibration_night(list(TARGETS), site, start, end)

    print(summarize_calibration_night(timeline))
    print()
    print(dispatch_sheet(timeline))

    with tempfile.TemporaryDirectory() as tmpdir:
        target_dir = out_dir if out_dir is not None else Path(tmpdir)
        target_dir.mkdir(parents=True, exist_ok=True)
        ecsv_path = target_dir / "planet_night.ecsv"
        write_timeline(timeline, ecsv_path)
        figures = save_figures(timeline, site, target_dir)
        print(f"Wrote {ecsv_path}" + (f" and {len(figures)} figure(s)" if figures else ""))
    passes = sum(1 for b in timeline.blocks if b.scan_type == "planet_cal")
    print(f"Night: {len(timeline.blocks)} blocks, {passes} passes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
