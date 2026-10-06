"""Private names one part of the package imports from another, against a reviewed list.

A unit is either the set of root modules (``site``, ``coordinates``,
``offsets`` and their siblings) or one top-level subpackage (``patterns``,
``planning``, ``overhead``, ``visualization``), read from the source tree. A
name is private when any component after the package name starts with an
underscore. Importing another unit's private name couples the two below the
published surface, so every such import is listed in :data:`ALLOWED` with the
reason it is allowed. The test fails on a crossing the list does not hold and
on an entry no import matches any more, so the list grows and shrinks only by
a conscious edit.

The root's shared internal layer is exempt: a private module at the package
root (``fyst_trajectories._validation``) and the Sun-verdict helper of the
seam module (``sun_protocols._sun_verdicts``), through which the consumers of
an injected Sun model ask for a grid's verdicts.

The scan reads import statements through ``_tiers.imported_names``, so
function-local, ``TYPE_CHECKING`` and string-argument imports count, and it
keeps only the most specific name of each import (``a._m`` is implied by
``a._m._f``). An attribute access on an imported module, ``module._name``, is
beyond it; no module of the package does that.
"""

from __future__ import annotations

import pytest
from _tiers import PACKAGE, SRC, _module_name, _package_of, imported_names

#: The top-level subpackages; each is a unit of its own.
SUBPACKAGES = frozenset(p.name for p in SRC.iterdir() if (p / "__init__.py").is_file())

#: Private names of public root modules that any unit may import.
SHARED_INTERNALS = frozenset({f"{PACKAGE}.sun_protocols._sun_verdicts"})

_PLANNER_SOLVE = "planning = execution: the simulator gates a visit on the planner's own solve"
_PATTERN_REACH = "planning = execution: the simulator bounds a visit by the reach of the pattern"
_FOOTPRINT = "the plot draws the focal-plane footprint with the offsets kernel"
_GRID = "the plot draws the target track on the observability grid it reports on"

#: Every private name one unit imports from another, and why it may, as
#: ``(importer, name)`` relative to the package.
_ALLOWED: dict[tuple[str, str], str] = {
    ("overhead.scheduler.helpers", "patterns.daisy._daisy_reach"): _PATTERN_REACH,
    ("overhead.scheduler.helpers", "patterns.pong._pong_peak_offsets"): _PATTERN_REACH,
    ("overhead.scheduler.helpers", "planning._ce_geometry._compute_ce_az_range"): _PLANNER_SOLVE,
    ("overhead.scheduler.helpers", "planning._ce_geometry._compute_ce_duration"): _PLANNER_SOLVE,
    ("planning._ce_geometry", "coordinates._threshold_crossings"): (
        "the crossing solve shares the rise and set search's threshold interpolation"
    ),
    ("visualization.sky_view", "observability._target_altaz_grid"): _GRID,
    ("visualization.sky_view", "offsets._offset_forward"): _FOOTPRINT,
    ("visualization.sky_view", "offsets._rotate_offset"): _FOOTPRINT,
    ("visualization.visibility", "coordinates._build_time_grid"): (
        "the plot samples on the time grid the observability report uses"
    ),
    ("visualization.visibility", "observability._target_altaz_grid"): _GRID,
    ("visualization.visibility", "offsets._rotate_offset"): _FOOTPRINT,
}

ALLOWED = {
    (f"{PACKAGE}.{importer}", f"{PACKAGE}.{name}"): reason
    for (importer, name), reason in _ALLOWED.items()
}


def _unit(name: str) -> str:
    """Return the unit of a dotted name: its top-level subpackage, else the root."""
    parts = name.split(".")
    if len(parts) > 1 and parts[1] in SUBPACKAGES:
        return f"{PACKAGE}.{parts[1]}"
    return PACKAGE


def _is_private(component: str) -> bool:
    """Report whether one dotted-name component is private (dunders are not)."""
    return component.startswith("_") and not component.endswith("__")


def crossings(module: str, package: str, source: str) -> set[tuple[str, str]]:
    """Private names ``module`` imports from another unit, as ``(module, name)`` pairs."""
    names = {n for n in imported_names(source, package=package) if n.startswith(f"{PACKAGE}.")}
    leaves = {n for n in names if not any(m.startswith(f"{n}.") for m in names)}
    found = set()
    for name in leaves:
        parts = name.split(".")
        if not any(_is_private(c) for c in parts[1:]):
            continue
        if _is_private(parts[1]) and parts[1] not in SUBPACKAGES:
            continue  # a private module at the package root
        if name in SHARED_INTERNALS:
            continue
        if _unit(name) != _unit(module):
            found.add((module, name))
    return found


def _observed() -> set[tuple[str, str]]:
    """Every cross-unit private import in the package source."""
    found: set[tuple[str, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        module = _module_name(path)
        found |= crossings(module, _package_of(module, path), path.read_text(encoding="utf-8"))
    return found


def test_cross_unit_private_imports_match_the_allowed_list():
    """The private names crossing a unit boundary are exactly the listed ones."""
    observed = _observed()
    new = sorted(observed - ALLOWED.keys())
    stale = sorted(ALLOWED.keys() - observed)
    assert not new and not stale, f"new crossings: {new}\nstale entries: {stale}"


@pytest.mark.parametrize(
    ("module", "source", "expected"),
    [
        pytest.param(
            f"{PACKAGE}.overhead.utils",
            "from ..planning._helpers import _x\n",
            {(f"{PACKAGE}.overhead.utils", f"{PACKAGE}.planning._helpers._x")},
            id="relative-cross-unit-private",
        ),
        pytest.param(
            f"{PACKAGE}.overhead.utils",
            "def f():\n    from ..planning._types import ScanBlock\n",
            {(f"{PACKAGE}.overhead.utils", f"{PACKAGE}.planning._types.ScanBlock")},
            id="function-local-public-name-of-a-private-module",
        ),
        pytest.param(
            f"{PACKAGE}.overhead.scheduler.phases",
            "from .._moves import plan_escape_move\n",
            set(),
            id="same-unit-private-module",
        ),
        pytest.param(
            f"{PACKAGE}.planning.pong",
            "from .._validation import _require_positive\n",
            set(),
            id="root-private-module",
        ),
        pytest.param(
            f"{PACKAGE}.planning._sun_safety",
            "from ..sun_protocols import _sun_verdicts\n",
            set(),
            id="shared-sun-verdict-helper",
        ),
        pytest.param(
            f"{PACKAGE}.planning.pong",
            "from ..coordinates import Coordinates\n",
            set(),
            id="public-import",
        ),
    ],
)
def test_the_scan_reads_a_crossing(module, source, expected):
    """The resolver flags exactly the cross-unit private imports."""
    package = module.rpartition(".")[0]
    assert crossings(module, package, source) == expected
