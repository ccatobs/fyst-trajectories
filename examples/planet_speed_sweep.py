"""Sweep the sweep speed on one planet, pass after pass (offline planning).

This runnable example scripts five consecutive passes on one body at five
azimuth sweep speeds with
:class:`~fyst_trajectories.overhead.ScriptedSelection`. Elevation still
drifts between the trials as the body moves, and with it the throw solved
from the footprint, so each pass records the elevation and throw it
actually ran at. It prints the applied geometry of each pass with its
science duty cycle and pass length.

Run it from the repository root::

    python examples/planet_speed_sweep.py
    python examples/planet_speed_sweep.py saturn 2026-09-11T06:30:00 2026-09-11T08:00:00
"""

from __future__ import annotations

import sys

from fyst_trajectories import get_fyst_site
from fyst_trajectories.overhead import (
    ScanOverrides,
    ScriptedSelection,
    plan_calibration_night,
    summarize_calibration_night,
)

SPEEDS = (0.5, 0.75, 1.0, 1.25, 1.5)
DEFAULT_BODY = "saturn"
DEFAULT_START = "2026-09-11T06:30:00"
DEFAULT_END = "2026-09-11T08:00:00"


def main(argv: list[str] | None = None) -> int:
    """Plan the sweep and print one line per pass.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments: an optional body name and optional
        ``start end`` ISO UTC times.

    Returns
    -------
    int
        Process exit status (0 on success).
    """
    argv = sys.argv[1:] if argv is None else argv
    body = argv[0] if len(argv) > 0 else DEFAULT_BODY
    start = argv[1] if len(argv) > 1 else DEFAULT_START
    end = argv[2] if len(argv) > 2 else DEFAULT_END

    sweep = ScriptedSelection([(body, ScanOverrides(az_speed=v)) for v in SPEEDS])
    timeline = plan_calibration_night([body], get_fyst_site(), start, end, selection=sweep)

    passes = [b for b in timeline.blocks if b.scan_type == "planet_cal"]
    print(
        f"{'start (UTC)':20s} {'deg/s':>6s} {'throw':>6s} {'legs':>5s} {'duty':>5s} {'len (s)':>7s}"
    )
    for block in passes:
        meta = block.metadata
        print(
            f"{block.t_start.iso[:19]:20s} "
            f"{meta['applied']['az_speed']:6.2f} "
            f"{meta['applied']['az_throw']:6.2f} "
            f"{meta['n_legs']:5d} "
            f"{meta['science_fraction']:5.0%} "
            f"{block.duration:7.1f}"
        )
    summary = summarize_calibration_night(timeline)
    print(f"Sweep: {len(passes)} of {len(SPEEDS)} passes planned; {len(summary.unplaced)} unplaced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
