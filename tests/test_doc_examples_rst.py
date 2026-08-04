"""Execute every code example under ``docs/`` (top level and ``api/``) so snippets cannot drift.

The docs' Python snippets are reStructuredText literal blocks (a paragraph ending in
``::`` followed by an indented block) and ``.. code-block:: python`` directives. Sphinx
*renders* these but never *runs* them, and the hand-typed copies in
``test_doc_examples.py`` (which assert specific outcomes for the load-bearing examples)
can silently drift from the prose. This guard extracts the actual ``.rst`` text and
executes each block, per page, cumulatively in a single namespace (a reader following
the page top to bottom), inside a temporary working directory.

A few pages assume the reader already holds an object (a built ``trajectory`` / ``traj``,
a ``timeline``) or a file (``retunes.csv``); the guard seeds those rather than cluttering
the docs with boilerplate setup. A small set of blocks are illustrative by design
(abbreviated ``...`` placeholders, agent-internal pseudo-code, a live HTTP ``POST``, and a
literal CSV schema) and are skipped with a recorded reason, see :data:`_SKIP`.

Strictness is "executes without error": a block passes if it runs to completion with no
exception. ``PointingWarning`` advisories (e.g. acceleration-limit notes from the compact
example configs) are expected and silenced. Outcome-value assertions live in
``test_doc_examples.py``, which this guard complements rather than replaces.
"""

import contextlib
import io
import os
import re
import textwrap
import warnings
from pathlib import Path

import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories import get_fyst_site
from fyst_trajectories.patterns import ConstantElScanConfig, TrajectoryBuilder

# Force a headless matplotlib backend before any doc block imports it (plotting pages).
os.environ.setdefault("MPLBACKEND", "Agg")

DOCS = Path(__file__).resolve().parents[1] / "docs"


def _collect_indented(lines, start, base_indent):
    """Collect the indented body of a block, stopping at the first dedent."""
    out = []
    i = start
    while i < len(lines):
        ln = lines[i]
        if ln.strip() == "":
            out.append("")
            i += 1
            continue
        if (len(ln) - len(ln.lstrip())) <= base_indent:
            break
        out.append(ln)
        i += 1
    while out and out[-1] == "":
        out.pop()
    return out, i


def _extract_blocks(text):
    """Extract ``(lineno, kind, code)`` for every literal / code-block in an rst page."""
    lines = text.split("\n")
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        m = re.match(r"^(\s*)\.\.\s+(?:code-block|code|sourcecode)::\s*python\s*$", line)
        if m:
            base = len(m.group(1))
            j = i + 1
            # skip directive options (":linenos:" etc.) and blank lines
            while j < n and (lines[j].strip() == "" or re.match(r"^\s*:\w[\w-]*:", lines[j])):
                j += 1
            body, end = _collect_indented(lines, j, base)
            blocks.append((i + 1, "code-block", textwrap.dedent("\n".join(body))))
            i = end
            continue
        stripped = line.rstrip()
        if stripped.endswith("::") and not stripped.lstrip().startswith(".."):
            para_indent = len(line) - len(line.lstrip())
            j = i + 1
            while j < n and lines[j].strip() == "":
                j += 1
            if j < n and (len(lines[j]) - len(lines[j].lstrip())) > para_indent:
                body, end = _collect_indented(lines, j, para_indent)
                blocks.append((j + 1, "literal", textwrap.dedent("\n".join(body))))
                i = end
                continue
        i += 1
    return blocks


def _is_python(code):
    """Return True if ``code`` compiles as Python (filters out CSV / shell / output)."""
    try:
        compile(code, "<doc>", "exec")
        return True
    except SyntaxError:
        return False


_VISUALIZATION_IMPORT_RE = re.compile(
    r"from\s+fyst_trajectories\.visualization\s+import\s+([^\n]+)"
)


def _visualization_symbols(text):
    """Names imported from ``fyst_trajectories.visualization`` anywhere on a page.

    Lets the matplotlib gate act per block: without the ``plotting`` extra, a page's
    plotting blocks are skipped while its non-plotting blocks still run, instead of
    skipping the whole page just because it references the visualization subpackage.
    """
    names = set()
    for match in _VISUALIZATION_IMPORT_RE.finditer(text):
        for part in match.group(1).replace("(", "").replace(")", "").split(","):
            name = part.strip().split(" as ")[-1].strip()
            if name:
                names.add(name)
    return names


# Blocks that are illustrative by design and must not be executed. Keyed by page;
# each entry matches a substring of the block's first non-blank line and records why.
_SKIP = {
    "planning.rst": [
        (
            "block = plan_pong_scan(...)",
            "abbreviated `...` placeholder; the full call is shown in Quick Start",
        ),
        (
            "az = x_offset / cos(radians(el_center)) + az_center",
            "illustrative horizon-frame mapping formula, not runnable Python",
        ),
    ],
    "retune_events.rst": [
        ("t_start_s,duration_s,module_index", "literal CSV schema shown as a block, not Python"),
    ],
    "trajectory_examples.rst": [
        ("# Inside ACUAgent", "agent-internal pseudo-code (undefined params/self)"),
        ("import requests", "live HTTP POST to a local TCS server"),
    ],
    "api/exceptions.rst": [
        (
            "import warnings",
            "abbreviated plan_constant_el_scan(...) placeholder; illustrates the "
            "PointingWarning-catch pattern",
        ),
    ],
    "api/visualization.rst": [
        (
            "from fyst_trajectories.overhead import read_timeline",
            "loads a user-provided ECSV timeline and renders it; illustrative I/O + plotting",
        ),
    ],
}


def _minimal_timeline(site):
    """Build a one-block :class:`ObservingTimeline` for pages that assume a ``timeline``."""
    from fyst_trajectories.overhead.models import (
        CalibrationPolicy,
        ObservingTimeline,
        OverheadModel,
        TimelineBlock,
    )

    t0 = Time("2026-06-15T02:00:00", scale="utc")
    block = TimelineBlock(
        t_start=t0,
        t_stop=t0 + TimeDelta(300, format="sec"),
        block_type="science",
        patch_name="field",
        az_start=120.0,
        az_end=180.0,
        elevation=45.0,
        scan_index=0,
        scan_type="pong",
        metadata={},
    )
    return ObservingTimeline(
        blocks=[block],
        site=site,
        start_time=t0,
        end_time=t0 + TimeDelta(3600, format="sec"),
        overhead_model=OverheadModel(),
        calibration_policy=CalibrationPolicy(),
    )


@pytest.fixture(scope="session")
def _doc_seed_trajectory():
    """Build a >= 700 s trajectory that the retune / export pages assume already exists.

    Read-only across pages (``inject_retune`` and ``to_path_format`` do not mutate it),
    so it is built once per session.
    """
    site = get_fyst_site()
    return (
        TrajectoryBuilder(site)
        .with_config(
            ConstantElScanConfig(
                timestep=0.1,
                az_start=120.0,
                az_stop=180.0,
                elevation=45.0,
                az_speed=1.0,
                az_accel=0.5,
            )
        )
        .duration(700.0)
        .starting_at(Time("2026-03-15T04:00:00", scale="utc"))
        .build()
    )


def _seed_namespace(tmp_path, trajectory, site):
    """Objects/files the docs assume the reader already has, in the temp cwd."""
    (tmp_path / "retunes.csv").write_text(
        "t_start_s,duration_s,module_index\n30.0,5.0,0\n300.0,5.0,0\n600.0,8.0,0\n"
    )
    return {
        "site": site,
        "trajectory": trajectory,
        "traj": trajectory,
        "timeline": _minimal_timeline(site),
    }


_RST_PAGES = sorted(DOCS.glob("*.rst")) + sorted((DOCS / "api").glob("*.rst"))


@pytest.mark.parametrize("rst", _RST_PAGES, ids=lambda p: p.relative_to(DOCS).as_posix())
def test_doc_page_examples_run(rst, tmp_path, monkeypatch, _doc_seed_trajectory):
    """Every Python code block on the page executes without error.

    Blocks run cumulatively in one namespace inside a temp working directory, seeded
    with the objects the page assumes the reader already has. Non-Python blocks and the
    by-design-illustrative blocks in :data:`_SKIP` are not executed.
    """
    monkeypatch.chdir(tmp_path)
    text = rst.read_text(encoding="utf-8")
    key = rst.relative_to(DOCS).as_posix()
    # The optional shared sun-avoidance library gates its examples the way
    # Without the shared sun-avoidance library, only the blocks that name
    # the library-backed "cad" model are skipped; every doc page keeps its
    # library-dependent blocks self-contained so no whole-page skip exists.
    try:
        import sun_avoidance  # noqa: F401

        have_sun_avoidance = True
    except ImportError:
        have_sun_avoidance = False
    # matplotlib gates only the plotting blocks. Without the ``plotting`` extra we skip
    # blocks that touch matplotlib or the plot functions per block, so the page's other
    # blocks still run.
    try:
        import matplotlib

        matplotlib.use("Agg")
        have_matplotlib = True
    except ImportError:
        have_matplotlib = False
    visualization_symbols = _visualization_symbols(text)
    ns = _seed_namespace(tmp_path, _doc_seed_trajectory, get_fyst_site())
    skips = _SKIP.get(key, [])
    for lineno, kind, code in _extract_blocks(text):
        if not code.strip() or not _is_python(code):
            continue
        first = next((ln for ln in code.splitlines() if ln.strip()), "")
        if any(sub in first for sub, _reason in skips):
            continue
        if not have_matplotlib and (
            "matplotlib" in code
            or any(re.search(rf"\b{re.escape(name)}\b", code) for name in visualization_symbols)
        ):
            continue
        if not have_sun_avoidance and ('"cad"' in code or "'cad'" in code):
            continue
        try:
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("ignore")
                exec(compile(code, f"{key}:{lineno}", "exec"), ns)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"{key}:{lineno} [{kind}] raised {type(exc).__name__}: {exc}\n"
                f"--- block ---\n{code}\n--- end ---"
            )
