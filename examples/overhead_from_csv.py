"""Source-list CSV to observing-timeline example (offline simulation).

This runnable example demonstrates the OFFLINE observing-night simulator
path at library scope: it reads a small source-list CSV, builds
:class:`~fyst_trajectories.overhead.ObservingPatch` objects, generates a
complete timeline with :func:`~fyst_trajectories.overhead.generate_timeline`,
round-trips it through TOAST-compatible ECSV, and prints a short summary.

The timeline produced here is a planning artifact for survey-design and
efficiency studies, not a schedule that drives a live observing night.

The CSV schema is ``name,RA,DEC,width,height,priority,velocity,scan_type``,
with ``RA`` in sexagesimal hour-angle and ``DEC`` in sexagesimal degrees
(parsed with :class:`astropy.coordinates.SkyCoord`), ``width`` / ``height``
in degrees, and ``scan_type`` one of ``constant_el``, ``pong``, or ``daisy``.

Run it from the repository root::

    python examples/overhead_from_csv.py
    python examples/overhead_from_csv.py path/to/sourcelist.csv
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord

from fyst_trajectories import get_fyst_site
from fyst_trajectories.overhead import (
    CalibrationPolicy,
    ObservingPatch,
    OverheadModel,
    compute_budget,
    generate_timeline,
    read_timeline,
    write_timeline,
)

# 8-hour southern-sky window; the bundled sample places several patches
# near the meridian during this span.
START_TIME = "2026-06-15T02:00:00"
END_TIME = "2026-06-15T10:00:00"

# Default elevation (degrees) for constant-elevation patches, which require
# a fixed elevation the CSV schema does not carry.
DEFAULT_CE_ELEVATION = 50.0

DEFAULT_CSV = Path(__file__).resolve().parent / "sample_sourcelist.csv"


def load_patches(csv_path: Path) -> list[ObservingPatch]:
    """Parse a source-list CSV into :class:`ObservingPatch` objects.

    Parameters
    ----------
    csv_path : Path
        Path to a CSV with columns ``name``, ``RA``, ``DEC``, ``width``,
        ``height``, ``priority``, ``velocity``, ``scan_type``. ``RA`` is
        sexagesimal hour-angle; ``DEC`` is sexagesimal degrees.

    Returns
    -------
    list of ObservingPatch
        One patch per CSV row, in file order.
    """
    patches: list[ObservingPatch] = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            coord = SkyCoord(row["RA"], row["DEC"], unit=(u.hourangle, u.deg))
            scan_type = row["scan_type"].strip()
            # Constant-elevation scans need a fixed elevation; pong/daisy
            # track RA/Dec and leave it unset.
            elevation = DEFAULT_CE_ELEVATION if scan_type == "constant_el" else None
            patches.append(
                ObservingPatch(
                    name=row["name"].strip(),
                    ra_center=float(coord.ra.deg),
                    dec_center=float(coord.dec.deg),
                    width=float(row["width"]),
                    height=float(row["height"]),
                    scan_type=scan_type,
                    velocity=float(row["velocity"]),
                    priority=float(row["priority"]),
                    elevation=elevation,
                )
            )
    return patches


def main(argv: list[str] | None = None) -> int:
    """Build a timeline from a source-list CSV and print a summary.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments. A single optional positional argument
        names the source-list CSV; it defaults to the bundled sample.

    Returns
    -------
    int
        Process exit status (0 on success).
    """
    argv = sys.argv[1:] if argv is None else argv
    csv_path = Path(argv[0]) if argv else DEFAULT_CSV

    patches = load_patches(csv_path)

    # Fully specify the overhead model and calibration policy so this example
    # is invariant to future changes in the library defaults.
    overhead_model = OverheadModel(
        retune_duration=5.0,
        pointing_cal_duration=180.0,
        focus_duration=300.0,
        skydip_duration=300.0,
        planet_cal_duration=600.0,
        beam_map_duration=600.0,
        settle_time=5.0,
        min_scan_duration=60.0,
        max_scan_duration=3600.0,
    )
    calibration_policy = CalibrationPolicy(
        retune_cadence=0.0,
        pointing_cadence=3600.0,
        focus_cadence=7200.0,
        skydip_cadence=10800.0,
        planet_cal_cadence=43200.0,
        beam_map_cadence=None,
        planet_targets=("jupiter", "saturn", "mars", "uranus", "neptune"),
        planet_min_elevation=20.0,
    )

    timeline = generate_timeline(
        patches=patches,
        site=get_fyst_site(),
        start_time=START_TIME,
        end_time=END_TIME,
        overhead_model=overhead_model,
        calibration_policy=calibration_policy,
    )

    # Round-trip through TOAST-compatible ECSV to show the I/O path.
    with tempfile.TemporaryDirectory() as tmpdir:
        ecsv_path = Path(tmpdir) / "timeline.ecsv"
        write_timeline(timeline, ecsv_path)
        loaded = read_timeline(ecsv_path)

    stats = compute_budget(loaded)
    print(f"Loaded {len(patches)} patches from {csv_path}")
    print(
        f"Timeline: {len(loaded)} blocks, "
        f"{stats['n_science_scans']} science scans, "
        f"efficiency {stats['efficiency']:.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
