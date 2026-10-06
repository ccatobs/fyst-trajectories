"""The two-tier import boundary: the library tier never imports the simulator tier.

The tier definition lives in ``_tiers`` and is shared with the ``offline``
marker rule, so the guard and the marker cannot drift apart. The library tier
is everything outside ``fyst_trajectories.overhead`` and the timeline figures
in ``fyst_trajectories.visualization.overhead``: site, coordinates, trajectory,
patterns, offsets, planning, dispatch, sun models, observability, and the
plots that draw them. The dependency runs one way, downward: the simulator
imports the library, never the reverse, and the timeline figures load the
simulator only inside the functions that draw a timeline, so a consumer
importing the library-tier plots pays nothing for the simulator.

Two guards enforce it. A static scan reads every import in every library-tier
module, function-local, ``TYPE_CHECKING`` and string-argument ones included,
and resolves relative imports to absolute names. A fresh-interpreter import of
the root package, each library-tier subpackage and the visualization
subpackage asserts that nothing under ``fyst_trajectories.overhead`` was
loaded.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from _tiers import (
    OVERHEAD,
    PACKAGE,
    SRC,
    TIMELINE_PLOTS,
    VISUALIZATION,
    _module_name,
    _package_of,
    has_prefix,
    imported_names,
    is_simulator_tier,
)

# What a consumer of the library tier may import without loading the
# simulator. The visualization subpackage is included deliberately: its
# library-tier plots share the subpackage with the timeline plots.
LIBRARY_TIER_IMPORTS = (
    PACKAGE,
    f"{PACKAGE}.patterns",
    f"{PACKAGE}.planning",
    f"{PACKAGE}.dispatch",
    f"{PACKAGE}.sun_models",
    f"{PACKAGE}.observability",
    VISUALIZATION,
)


def _library_tier_modules() -> list[Path]:
    """Every source file outside the simulator tier."""
    return sorted(path for path in SRC.rglob("*.py") if not is_simulator_tier(_module_name(path)))


def _forbidden_prefixes(module: str) -> tuple[str, ...]:
    """Import prefixes a library-tier module may not name, by where it lives.

    Every library-tier module is barred from the simulator package. Outside
    the visualization subpackage the bar widens to the whole of
    ``visualization``: the library core draws nothing. Inside it, the
    subpackage ``__init__`` re-exports every plot the package publishes, the
    timeline figures included, and that import loads no simulator code because
    those figures defer their own simulator imports into the drawing
    functions; the fresh-interpreter guard below is what keeps that true.
    """
    if module == VISUALIZATION:
        return (OVERHEAD,)
    if module.startswith(f"{VISUALIZATION}."):
        return (OVERHEAD, TIMELINE_PLOTS)
    return (OVERHEAD, VISUALIZATION)


def _offending_imports(module: str, package: str, source: str) -> list[str]:
    """Names a library-tier module imports that cross the boundary."""
    forbidden = _forbidden_prefixes(module)
    return sorted({n for n in imported_names(source, package=package) if has_prefix(n, forbidden)})


@pytest.mark.parametrize("path", _library_tier_modules(), ids=lambda p: _module_name(p))
def test_library_tier_module_imports_nothing_from_the_simulator(path: Path):
    """No import in a library-tier module names the simulator tier."""
    module = _module_name(path)
    offending = _offending_imports(
        module, _package_of(module, path), path.read_text(encoding="utf-8")
    )
    assert not offending, f"{module} imports the simulator tier: {offending}"


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: _module_name(p))
def test_only_the_visualization_subpackage_imports_matplotlib(path: Path):
    """No module outside ``visualization`` names matplotlib, lazily or not."""
    module = _module_name(path)
    if has_prefix(module, (VISUALIZATION,)):
        return
    names = imported_names(path.read_text(encoding="utf-8"), package=_package_of(module, path))
    offending = sorted(n for n in names if has_prefix(n, ("matplotlib",)))
    assert not offending, f"{module} imports matplotlib: {offending}"


def test_scan_sees_every_module_and_resolves_relative_imports():
    """The scan covers the tree and its resolver reads a relative import correctly."""
    modules = {_module_name(p) for p in _library_tier_modules()}
    assert PACKAGE in modules
    assert f"{PACKAGE}.planning.source_ces" in modules
    assert not any(is_simulator_tier(m) for m in modules)
    source_ces = SRC / "planning" / "source_ces" / "__init__.py"
    names = imported_names(
        source_ces.read_text(encoding="utf-8"), package=f"{PACKAGE}.planning.source_ces"
    )
    assert f"{PACKAGE}.exceptions" in names


def test_the_scan_covers_the_library_tier_plot_modules():
    """The plot modules are scanned; only the timeline figures are exempt.

    The subpackage is mixed: its timeline figures are simulator tier and every
    other plot is library tier. Excluding the whole directory would leave six
    library-tier modules unscanned, and an ordinary function-local simulator
    import in one of them would then pass both guards.
    """
    modules = {_module_name(p) for p in _library_tier_modules()}
    assert VISUALIZATION in modules
    assert f"{VISUALIZATION}.visibility" in modules
    assert f"{VISUALIZATION}.sky_view" in modules
    assert TIMELINE_PLOTS not in modules


def test_a_function_local_simulator_import_in_a_plot_module_is_caught():
    """The codebase's own lazy-import idiom is flagged in a library-tier plot module."""
    source = (
        "def draw(timeline):\n"
        "    from ..overhead.models import ObservingTimeline\n"
        "    return isinstance(timeline, ObservingTimeline)\n"
    )
    offending = _offending_imports(f"{VISUALIZATION}.visibility", VISUALIZATION, source)
    assert offending == [f"{OVERHEAD}.models", f"{OVERHEAD}.models.ObservingTimeline"]


def test_a_string_argument_import_is_caught():
    """``import_module`` / ``__import__`` with a literal name are read like a statement."""
    source = (
        "import importlib\n"
        "\n"
        "def late():\n"
        f"    importlib.import_module({OVERHEAD + '.models'!r})\n"
        f"    __import__({OVERHEAD + '.timeline'!r})\n"
    )
    offending = _offending_imports(f"{PACKAGE}.observability", PACKAGE, source)
    assert offending == [f"{OVERHEAD}.models", f"{OVERHEAD}.timeline"]


def test_the_plot_subpackage_may_re_export_its_timeline_figures():
    """The subpackage ``__init__`` re-exports the timeline plots; that is not a crossing."""
    source = "from .overhead import plot_sky_coverage\nfrom .visibility import plot_visibility\n"
    assert _offending_imports(VISUALIZATION, VISUALIZATION, source) == []
    # A sibling plot module importing the timeline figures is a crossing.
    assert _offending_imports(f"{VISUALIZATION}.visibility", VISUALIZATION, source) == [
        TIMELINE_PLOTS,
        f"{TIMELINE_PLOTS}.plot_sky_coverage",
    ]


def test_library_tier_imports_load_nothing_from_the_simulator():
    """Importing the library tier in a fresh interpreter leaves overhead unloaded."""
    code = "\n".join(
        [
            "import importlib, sys",
            f"for name in {LIBRARY_TIER_IMPORTS!r}:",
            "    importlib.import_module(name)",
            "    leaked = sorted(",
            "        m for m in sys.modules",
            f"        if m == '{OVERHEAD}' or m.startswith('{OVERHEAD}.')",
            "    )",
            "    if leaked:",
            "        print(name, leaked)",
            "        sys.exit(1)",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, (
        f"a library-tier import loaded the simulator tier\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
