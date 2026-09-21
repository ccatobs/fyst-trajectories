"""Run the bundled runnable examples end to end.

Each example under ``examples/`` is executed as a subprocess (the way a
reader would run it) and checked for a clean exit and its expected summary
output. Running out-of-process keeps the example's ``__main__`` path honest
and isolates its imports from the test session.

The one thing the child is not allowed to inherit from the reader is Earth
orientation: ``conftest.py`` pins the suite to the vendored IERS table, and a
fresh interpreter starts with none of that, so an unpinned launch would reach
for the network (or fail on a cold machine without one) and compute with
whatever EOP the runner happened to fetch. The pin is passed in through a
``sitecustomize`` shim on the child's import path; the examples themselves are
untouched and a reader still runs them exactly as documented.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

# Every bundled example runs the simulator tier.
pytestmark = pytest.mark.offline

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "overhead_from_csv.py"
SAMPLE_CSV = REPO_ROOT / "examples" / "sample_sourcelist.csv"
IERS_TABLE = Path(__file__).resolve().parent / "data" / "finals2000A.all"

# Applied by the child interpreter at startup, and inert unless the launcher
# asks for it: a sitecustomize on PYTHONPATH shadows any the environment
# provides, so it must do nothing to anything else.
_SITECUSTOMIZE = '''"""Apply the test suite's vendored IERS pin in a subprocess."""

import os

from astropy.utils import iers

_table = os.environ.get("FYST_IERS_TABLE")
if _table:
    iers.conf.auto_download = False
    iers.earth_orientation_table.set(iers.IERS_A.open(_table))
    iers.conf.iers_degraded_accuracy = "warn"
'''


@pytest.fixture(scope="module")
def example_env(tmp_path_factory):
    """Build the child environment that carries the suite's IERS pin."""
    directory = tmp_path_factory.mktemp("iers_pin")
    (directory / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
    env = dict(os.environ)
    env["FYST_IERS_TABLE"] = str(IERS_TABLE)
    inherited = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{directory}{os.pathsep}{inherited}" if inherited else str(directory)
    return env


def test_example_child_uses_the_vendored_iers_table(example_env, tmp_path):
    """A launched example computes on the vendored table, with downloads off.

    The probe reads the child's active Earth-orientation table rather than
    trusting that it worked: before the pin was passed in, the child started
    with ``auto_download`` on and astropy's own bundled table, whose coverage
    depends on the machine.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from astropy.utils import iers\n"
        "print('auto_download', iers.conf.auto_download)\n"
        "print('table', iers.earth_orientation_table.get().meta['data_path'])\n",
        encoding="utf-8",
    )
    cold_cache = tmp_path / "cold"  # no downloaded table to fall back on
    cold_cache.mkdir()
    env = dict(example_env)
    env["XDG_CACHE_HOME"] = str(cold_cache)
    result = subprocess.run([sys.executable, str(probe)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "auto_download False" in result.stdout
    assert f"table {IERS_TABLE}" in result.stdout


def test_overhead_from_csv_example_runs(example_env):
    """The CSV-to-timeline example exits 0 and reports a non-empty timeline.

    Runs the example in a fresh interpreter (the way a reader would), then
    checks the exit status and the summary line. It builds a full 8-hour
    timeline but completes in a few seconds, so it stays in the fast suite.
    """
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), str(SAMPLE_CSV)],
        cwd=REPO_ROOT,
        env=example_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    match = re.search(r"Timeline:\s*(\d+)\s*blocks", result.stdout)
    assert match is not None, f"summary line missing from output:\n{result.stdout}"
    assert int(match.group(1)) > 0, f"expected a non-empty timeline:\n{result.stdout}"

    # The sample's constant-elevation row exists to exercise that path, and a
    # constant-elevation pass is only placeable in the minutes before its
    # crossing opens; the row's priority and the window are tuned so it is.
    per_patch = re.search(r"GC_CE (\d+)", result.stdout)
    assert per_patch is not None, f"per-patch line missing from output:\n{result.stdout}"
    assert int(per_patch.group(1)) > 0, (
        f"the constant-elevation row scheduled nothing:\n{result.stdout}"
    )


PLANET_NIGHT = REPO_ROOT / "examples" / "planet_night.py"
SPEED_SWEEP = REPO_ROOT / "examples" / "planet_speed_sweep.py"


def test_planet_night_example_runs(example_env, tmp_path):
    """The planet-night example plans a short window, writes the ECSV and figures, exits 0."""
    result = subprocess.run(
        [
            sys.executable,
            str(PLANET_NIGHT),
            "2026-09-11T06:30:00",
            "2026-09-11T07:15:00",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        env=example_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    match = re.search(r"Night:\s*(\d+)\s*blocks,\s*(\d+)\s*passes", result.stdout)
    assert match is not None, f"summary line missing from output:\n{result.stdout}"
    assert int(match.group(2)) >= 1
    assert "source_scan on saturn" in result.stdout
    assert (tmp_path / "planet_night.ecsv").exists()
    pytest.importorskip("matplotlib")
    assert (tmp_path / "planet_night_gantt.png").exists()
    assert (tmp_path / "planet_night_first_pass.png").exists()


def test_planet_speed_sweep_example_runs(example_env):
    """The speed-sweep example scripts consecutive passes on one body, exits 0."""
    result = subprocess.run(
        [
            sys.executable,
            str(SPEED_SWEEP),
            "saturn",
            "2026-09-11T06:30:00",
            "2026-09-11T07:15:00",
        ],
        cwd=REPO_ROOT,
        env=example_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    match = re.search(r"Sweep:\s*(\d+)\s*of\s*5\s*passes planned", result.stdout)
    assert match is not None, f"summary line missing from output:\n{result.stdout}"
    assert int(match.group(1)) >= 2
    rows = [ln for ln in result.stdout.splitlines() if ln.startswith("2026-09-11 ")]
    speeds = [float(ln.split()[2]) for ln in rows]
    assert speeds == sorted(speeds) and speeds[0] == 0.5
