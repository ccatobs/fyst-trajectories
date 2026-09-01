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
the docs with boilerplate setup.

Four guard mechanisms, each closing a class of drift that reached publication:

1. **Warnings are asserted, not suppressed.** Each block's emitted
   :class:`~fyst_trajectories.exceptions.PointingWarning`-family categories must equal
   the set declared for it in :data:`_EXPECT_WARNINGS` (default: none). A doc example
   that trips a limit advisory the page never mentions fails. Warnings from other
   libraries (matplotlib, astropy) are outside the assertion.
2. **Stdout is captured and checked.** A ``# comment`` stating a print's expected
   output (inline on the ``print(...)`` line, or alone on the next line) must appear in
   the block's captured stdout, so a wrong-but-running example cannot pass.
3. **Skipped blocks stay honest.** Blocks in :data:`_SKIP` are not executed (they are
   illustrative by design) but must still parse, their ``fyst_trajectories`` imports
   must resolve, and their calls to those symbols must bind against the real
   signatures (``test_skip_blocks_still_parse_and_bind``).
4. **Non-Python blocks are allow-listed.** A block that does not compile must match an
   entry in :data:`_NON_PYTHON` with a recorded reason, so a Python block that stops
   compiling cannot vanish silently into the shell-block bucket.

Outcome-value assertions live in ``test_doc_examples.py``, which this guard complements
rather than replaces.
"""

import ast
import contextlib
import inspect
import io
import os
import re
import textwrap
import warnings
from pathlib import Path

import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories import get_fyst_site
from fyst_trajectories.exceptions import PointingWarning
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
# Every entry must still parse and bind (see test_skip_blocks_still_parse_and_bind),
# and must match at least one block on its page (see test_registries_match_blocks).
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
        (
            "t_start_s,duration_s,module_index",
            "literal CSV schema; compiles as a bare tuple expression but is not Python",
        ),
    ],
    "trajectory_examples.rst": [
        ("import dataclasses", "live HTTP POST to a local TCS server"),
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

# Non-Python blocks (shell commands, ASCII diagrams, literal schemas) that the
# extractor picks up. A non-compiling block that matches no entry here fails its
# page test, so a Python block that stops compiling cannot vanish silently.
_NON_PYTHON = {
    "installation.rst": [
        ("pip install", "shell: install commands"),
        ("git clone", "shell: source checkout"),
        ("pytest tests/", "shell: test invocation"),
        ("ruff check", "shell: lint invocation"),
    ],
    "overhead_integration.rst": [
        ("OFFLINE SIM LANE", "ASCII lane diagram"),
    ],
    "sun_avoidance.rst": [
        ("pip install", "shell: pinned sun-avoidance install"),
    ],
    "trajectory_examples.rst": [
        ("scheduling layer --[scan_params]-->", "ASCII dispatch-flow diagram"),
    ],
    "api/visualization.rst": [
        ("pip install", "shell: plotting extra install"),
    ],
}

# Library warnings each block is EXPECTED to emit, as a set of
# PointingWarning-subclass names, keyed by page and matched on the block's first
# non-blank line. Blocks not listed must emit no PointingWarning-family warning at
# all. Currently empty: every published example runs advisory-clean, and new
# examples should stay that way (shrink the example rather than declaring the
# advisory, so pending limit decisions cannot silently change the docs' meaning).
_EXPECT_WARNINGS: dict = {}


def _expected_warnings_for(key, first_line):
    for sub, names in _EXPECT_WARNINGS.get(key, []):
        if sub in first_line:
            return set(names)
    return set()


_PRINT_LINE_RE = re.compile(r"^\s*print\(")


def _comment_text(line):
    """Return the text of a ``#`` comment on ``line`` that is outside any string."""
    in_s = in_d = False
    i = 0
    while i < len(line):
        c = line[i]
        if c == "'" and not in_d:
            in_s = not in_s
        elif c == '"' and not in_s:
            in_d = not in_d
        elif c == "#" and not in_s and not in_d:
            return line[i + 1 :].strip()
        i += 1
    return None


def _normalize_ws(text):
    """Collapse whitespace runs so column-aligned comments match single-space prints."""
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_output(text):
    """Distinguish an output claim from a prose description next to a print.

    A claim contains a digit, starts like a Python value (quote, bracket, brace,
    paren, minus), or is a single bare token (``icrs``); multi-word digit-free
    prose (``one-line human-readable verdict``) is not asserted.
    """
    if any(ch.isdigit() for ch in text):
        return True
    if text[0] in "'\"[({-":
        return True
    return " " not in text


def _stdout_expectations(code):
    """Expected-output substrings stated as comments next to ``print(...)`` calls.

    Two shapes are recognised: an inline comment on the ``print`` line, and one
    or more full-line comments directly below it (no blank line between, so a
    loop's per-iteration outputs can be listed). Surrounding quotes are
    stripped so ``# "altaz"`` matches the unquoted printed value; whitespace is
    normalised so column-aligned comments match; an expectation abbreviated
    with ``...`` asserts only the text before the ellipsis.
    """
    expectations = []
    lines = code.splitlines()
    for i, line in enumerate(lines):
        if not _PRINT_LINE_RE.match(line):
            continue
        candidates = []
        inline = _comment_text(line)
        if inline:
            candidates.append(inline)
        j = i + 1
        while j < len(lines) and lines[j].strip().startswith("#"):
            candidates.append(lines[j].strip().lstrip("#").strip())
            j += 1
        for text in candidates:
            if not text or text == "..." or not _looks_like_output(text):
                continue
            if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
                text = text[1:-1]
            if "..." in text:
                text = text.split("...", 1)[0].rstrip(" ,")
                if len(text) < 3:
                    continue
            expectations.append(_normalize_ws(text))
    return expectations


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
    so it is built once per session. Advisories from this scaffolding config (a 60 deg
    azimuth throw) are test-internal, not published examples, and are suppressed here.
    """
    site = get_fyst_site()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PointingWarning)
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
    """Every Python code block on the page executes cleanly, with checked output.

    Blocks run cumulatively in one namespace inside a temp working directory, seeded
    with the objects the page assumes the reader already has. Each block must raise
    nothing, emit exactly its declared set of PointingWarning-family warnings
    (default: none), and print any output its comments promise. Non-Python blocks
    must be allow-listed in :data:`_NON_PYTHON`; the by-design-illustrative blocks
    in :data:`_SKIP` are not executed but are parse- and bind-checked separately.
    """
    monkeypatch.chdir(tmp_path)
    text = rst.read_text(encoding="utf-8")
    key = rst.relative_to(DOCS).as_posix()
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
    non_python = _NON_PYTHON.get(key, [])
    for lineno, kind, code in _extract_blocks(text):
        if not code.strip():
            continue
        first = next((ln for ln in code.splitlines() if ln.strip()), "")
        if not _is_python(code):
            if not any(sub in first for sub, _reason in non_python):
                pytest.fail(
                    f"{key}:{lineno} [{kind}] does not compile as Python and is not "
                    f"allow-listed in _NON_PYTHON. If it is meant to be Python, fix it; "
                    f"if not, add an entry with a reason.\n--- block ---\n{code}\n--- end ---"
                )
            continue
        if any(sub in first for sub, _reason in skips):
            continue
        if not have_matplotlib and (
            "matplotlib" in code
            or any(re.search(rf"\b{re.escape(name)}\b", code) for name in visualization_symbols)
        ):
            continue
        if not have_sun_avoidance and ('"cad"' in code or "'cad'" in code):
            continue
        buf = io.StringIO()
        try:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                with contextlib.redirect_stdout(buf):
                    exec(compile(code, f"{key}:{lineno}", "exec"), ns)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"{key}:{lineno} [{kind}] raised {type(exc).__name__}: {exc}\n"
                f"--- block ---\n{code}\n--- end ---"
            )
        emitted = {type(w.message).__name__ for w in rec if isinstance(w.message, PointingWarning)}
        expected = _expected_warnings_for(key, first)
        if emitted != expected:
            pytest.fail(
                f"{key}:{lineno} [{kind}] emitted library warnings {sorted(emitted)}, "
                f"expected {sorted(expected)}. A doc example must not trip an advisory "
                f"its page never mentions; shrink the example (do not quote limit "
                f"values into prose) or declare the warning in _EXPECT_WARNINGS.\n"
                + "\n".join(
                    f"  {type(w.message).__name__}: {w.message}"
                    for w in rec
                    if isinstance(w.message, PointingWarning)
                )
            )
        out = _normalize_ws(buf.getvalue())
        for exp in _stdout_expectations(code):
            if exp not in out:
                pytest.fail(
                    f"{key}:{lineno} [{kind}] promises output {exp!r} in a comment, "
                    f"but the block printed:\n{buf.getvalue()}\n--- block ---\n{code}\n--- end ---"
                )


def _iter_registry_blocks(registry):
    """Yield (key, sub, reason, matching blocks) for every registry entry."""
    for key, entries in registry.items():
        text = (DOCS / key).read_text(encoding="utf-8")
        blocks = _extract_blocks(text)
        for sub, reason in entries:
            matches = [
                (lineno, code)
                for lineno, _kind, code in blocks
                if code.strip() and sub in next((ln for ln in code.splitlines() if ln.strip()), "")
            ]
            yield key, sub, reason, matches


def test_registries_match_blocks():
    """Every _SKIP / _NON_PYTHON / _EXPECT_WARNINGS entry matches a real block.

    A stale entry means the block it covered was edited or removed; the registry
    must follow, or an unrelated new block could silently inherit the exemption.
    """
    for key, sub, _reason, matches in _iter_registry_blocks(_SKIP):
        assert matches, f"_SKIP entry {sub!r} matches no block on {key}"
    for key, sub, _reason, matches in _iter_registry_blocks(_NON_PYTHON):
        assert matches, f"_NON_PYTHON entry {sub!r} matches no block on {key}"
    for key, entries in _EXPECT_WARNINGS.items():
        text = (DOCS / key).read_text(encoding="utf-8")
        blocks = _extract_blocks(text)
        for sub, _names in entries:
            assert any(
                sub in next((ln for ln in code.splitlines() if ln.strip()), "")
                for _lineno, _kind, code in blocks
                if code.strip()
            ), f"_EXPECT_WARNINGS entry {sub!r} matches no block on {key}"


def _fyst_imports(tree):
    """Map local name -> object for ``fyst_trajectories`` imports in ``tree``.

    Returns ``None`` for an import whose module needs an optional dependency that
    is not installed (the caller skips those).
    """
    import importlib

    resolved = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (node.module == "fyst_trajectories" or node.module.startswith("fyst_trajectories."))
        ):
            try:
                mod = importlib.import_module(node.module)
            except ImportError:
                return None
            for alias in node.names:
                assert hasattr(mod, alias.name), f"{node.module} has no attribute {alias.name!r}"
                resolved[alias.asname or alias.name] = getattr(mod, alias.name)
    return resolved


def test_skip_blocks_still_parse_and_bind():
    """Skipped blocks must parse, resolve their library imports, and bind their calls.

    _SKIP blocks never execute, so this is the check that keeps them from rotting:
    every ``from fyst_trajectories... import name`` must resolve, and every call to
    one of those names must bind against its real signature (a removed or newly
    required parameter fails here). Calls abbreviated with a literal ``...``
    argument are resolution-checked only.
    """
    for key, sub, _reason, matches in _iter_registry_blocks(_SKIP):
        assert matches, f"_SKIP entry {sub!r} matches no block on {key}"
        for lineno, code in matches:
            tree = ast.parse(code)
            resolved = _fyst_imports(tree)
            if resolved is None:
                continue  # optional dependency missing; nothing to bind against
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                fn = resolved.get(node.func.id)
                if fn is None or not callable(fn):
                    continue
                if any(isinstance(a, ast.Constant) and a.value is Ellipsis for a in node.args):
                    continue  # explicit `...` abbreviation: resolution check only
                if any(isinstance(a, ast.Starred) for a in node.args):
                    continue
                pos = [object()] * len(node.args)
                kw = {k.arg: None for k in node.keywords if k.arg is not None}
                try:
                    inspect.signature(fn).bind(*pos, **kw)
                except TypeError as exc:
                    pytest.fail(
                        f"{key}:{lineno}: skipped block calls {node.func.id}() in a way "
                        f"that no longer binds: {exc}"
                    )
