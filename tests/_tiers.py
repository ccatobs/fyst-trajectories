"""The two-tier boundary, defined once for every test-side rule that reads it.

The package is split in two tiers (see the two-tier note on the documentation
index). The *library tier* is everything a control system or a scheduler
imports; the *simulator tier* is the offline night simulator
``fyst_trajectories.overhead`` plus the timeline figures in
``fyst_trajectories.visualization.overhead``, the simulator's only module
outside that subpackage. The dependency runs one way, downward: the
simulator imports the library, never the reverse.

Three test-side rules follow from that one definition, and all three read it
from here so they cannot drift apart:

* ``conftest.py`` marks a test module ``offline`` when the module itself, or a
  ``conftest.py`` above it, imports the simulator tier;
* the documentation guards mark their simulator-tier pages and modules at the
  parametrization site, using :func:`is_simulator_tier` to decide;
* ``test_tier_boundary.py`` scans every library-tier source module for an
  import that crosses the boundary.

The scan reads imports rather than paths because the rule it enforces is about
imports: a test that never names the simulator does not exercise it, wherever
it happens to live. The source readers here (:func:`imported_names`,
:func:`_module_name`, :func:`_package_of`) also serve
``test_private_imports.py``, which checks the private names one part of the
package imports from another.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = "fyst_trajectories"
OVERHEAD = f"{PACKAGE}.overhead"
VISUALIZATION = f"{PACKAGE}.visualization"
TIMELINE_PLOTS = f"{VISUALIZATION}.overhead"

#: The simulator tier: the offline night simulator and the figures that draw
#: its timelines. Everything else in the package is library tier.
SIMULATOR_TIER = (OVERHEAD, TIMELINE_PLOTS)

_TESTS_ROOT = Path(__file__).resolve().parent

#: The package's source directory, whose modules the import scans read.
SRC = _TESTS_ROOT.parent / "src" / PACKAGE


def _module_name(path: Path) -> str:
    """Dotted module name of a source file under the package root."""
    parts = path.relative_to(SRC).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((PACKAGE, *parts))


def _package_of(module: str, path: Path) -> str:
    """Package a source file's relative imports resolve against."""
    return module if path.name == "__init__.py" else module.rpartition(".")[0]


def has_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    """Report whether a dotted name is one of ``prefixes`` or lives under one."""
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def is_simulator_tier(name: str) -> bool:
    """Report whether a dotted module name belongs to the simulator tier."""
    return has_prefix(name, SIMULATOR_TIER)


def _dynamic_import_name(node: ast.Call) -> str | None:
    """Read the module name out of a string-argument dynamic import call."""
    func = node.func
    named_import = (isinstance(func, ast.Name) and func.id in {"__import__", "import_module"}) or (
        isinstance(func, ast.Attribute) and func.attr == "import_module"
    )
    if not named_import or not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def imported_names(source: str, *, package: str | None = None) -> list[str]:
    """List every module name a Python source imports, anywhere in its body.

    The walk covers the whole tree, so function-local and ``TYPE_CHECKING``
    imports count, and it collects the two string-argument forms that no
    import statement records: ``importlib.import_module("...")`` and
    ``__import__("...")``. A name built by string arithmetic is beyond a
    static scan; nothing in the package uses one.

    Parameters
    ----------
    source : str
        Python source text.
    package : str, optional
        Package the file belongs to, used to resolve relative imports. Without
        it relative imports are ignored (test modules have none).

    Returns
    -------
    list of str
        Absolute module names, plus ``module.name`` for each name a
        ``from`` import binds, in source order.
    """
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level and package is None:
                continue
            if node.level:
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                base = f"{base}.{node.module}" if node.module else base
            else:
                base = node.module or ""
            names.append(base)
            names.extend(f"{base}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call):
            dynamic = _dynamic_import_name(node)
            if dynamic is not None:
                names.append(dynamic)
    return names


def file_imports_simulator_tier(path: Path) -> bool:
    """Report whether a Python file imports the simulator tier anywhere."""
    return any(is_simulator_tier(name) for name in imported_names(path.read_text(encoding="utf-8")))


def module_is_simulator_tier(path: Path) -> bool:
    """Report whether a test module exercises the simulator tier.

    True when the module imports it, or when a ``conftest.py`` above it does:
    a file that only takes a simulator-backed fixture never names the
    simulator itself.
    """
    if file_imports_simulator_tier(path):
        return True
    directory = path.parent
    while True:
        conftest = directory / "conftest.py"
        if conftest.exists() and file_imports_simulator_tier(conftest):
            return True
        if directory == _TESTS_ROOT or directory == directory.parent:
            return False
        directory = directory.parent
