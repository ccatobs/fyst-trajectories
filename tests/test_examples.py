"""Run the bundled runnable examples end to end.

Each example under ``examples/`` is executed as a subprocess (the way a
reader would run it) and checked for a clean exit and its expected summary
output. Running out-of-process keeps the example's ``__main__`` path honest
and isolates its imports from the test session.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "overhead_from_csv.py"
SAMPLE_CSV = REPO_ROOT / "examples" / "sample_sourcelist.csv"


def test_overhead_from_csv_example_runs():
    """The CSV-to-timeline example exits 0 and reports a non-empty timeline.

    Runs the example in a fresh interpreter (the way a reader would), then
    checks the exit status and the summary line. It builds a full 8-hour
    timeline but completes in a few seconds, so it stays in the fast suite.
    """
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), str(SAMPLE_CSV)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    match = re.search(r"Timeline:\s*(\d+)\s*blocks", result.stdout)
    assert match is not None, f"summary line missing from output:\n{result.stdout}"
    assert int(match.group(1)) > 0, f"expected a non-empty timeline:\n{result.stdout}"
