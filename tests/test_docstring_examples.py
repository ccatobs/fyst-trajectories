"""Run every ``>>>`` docstring example in the package, so published Examples cannot drift.

Autodoc renders every NumPy ``Examples`` block and ``sphinx.ext.viewcode`` serves the
annotated source, so docstring examples are published output and have to be executed
somewhere. Each module's docstrings run under :mod:`doctest` with
``ELLIPSIS`` and ``NORMALIZE_WHITESPACE`` enabled.

Like ``test_doc_examples_rst.py``, the runner seeds the ambient objects the examples
assume the reader already has (``site``, ``coords``, and a built ``trajectory``), so the
docstrings stay uncluttered; everything else an example needs, it must define itself.
Examples that are illustrative by design (file paths, plotting output, optional
ephemeris kernels) carry an explicit ``# doctest: +SKIP``, which Sphinx strips from the
rendered page. Docstrings whose examples select the shared-library sun models
(``make_sun_safe("cad")`` / ``"cone"``) are skipped when the optional ``sun_avoidance``
package is not installed, the same conditional ``test_doc_examples_rst.py`` applies to
the docs pages; they still execute wherever the library is present.
"""

import ast
import doctest
import importlib
import inspect
import io
import os
import warnings
from pathlib import Path

import pytest
from _sun_stubs import HAVE_SUN_AVOIDANCE, needs_sun_avoidance
from _tiers import PACKAGE, is_simulator_tier
from astropy.time import Time

import fyst_trajectories
from fyst_trajectories import Coordinates, get_fyst_site
from fyst_trajectories.exceptions import PointingWarning
from fyst_trajectories.patterns import ConstantElScanConfig, TrajectoryBuilder

# Plotting examples run headless.
os.environ.setdefault("MPLBACKEND", "Agg")

_FLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE

SRC = Path(__file__).resolve().parents[1] / "src" / PACKAGE


def _needs_sun_avoidance(test: doctest.DocTest) -> bool:
    """Whether a doctest's examples select a shared-library sun model."""
    return needs_sun_avoidance("".join(example.source for example in test.examples))


def _example_imports(source):
    """Resolve the package names an example imports, asserting each one exists."""
    resolved = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(PACKAGE):
            module = importlib.import_module(node.module)
            for alias in node.names:
                assert hasattr(module, alias.name), f"{node.module} has no {alias.name!r}"
                resolved[alias.asname or alias.name] = getattr(module, alias.name)
    return resolved


def _call_target(func, namespace):
    """Resolve ``name(...)``, or ``Class.method(...)`` for a class or static method."""
    if isinstance(func, ast.Name):
        return namespace.get(func.id)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        owner = namespace.get(func.value.id)
        if inspect.isclass(owner):
            target = getattr(owner, func.attr, None)
            static = isinstance(inspect.getattr_static(owner, func.attr, None), staticmethod)
            if inspect.ismethod(target) or static:
                return target
    return None


def _module_names():
    """Every module in the package, read off the source tree.

    Walking the files rather than the import system matters for the tier
    split: ``pkgutil.walk_packages`` imports each package it descends into,
    which would load the simulator tier during collection of the library-tier
    job.
    """
    names = []
    for path in SRC.rglob("*.py"):
        parts = path.relative_to(SRC).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join((PACKAGE, *parts)))
    return sorted(names)


def _module_param(name: str):
    """Parametrize a module, marking the simulator tier ``offline``."""
    marks = [pytest.mark.offline] if is_simulator_tier(name) else []
    return pytest.param(name, id=name, marks=marks)


MODULE_NAMES = [_module_param(name) for name in _module_names()]


@pytest.fixture(scope="session")
def _doctest_globs():
    """Build the ambient namespace docstring examples assume: site, coords, a trajectory."""
    site = get_fyst_site()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PointingWarning)
        trajectory = (
            TrajectoryBuilder(site)
            .with_config(
                ConstantElScanConfig(
                    timestep=0.1,
                    az_start=120.0,
                    az_stop=145.0,
                    elevation=45.0,
                    az_speed=1.0,
                    az_accel=0.5,
                )
            )
            .duration(700.0)
            .starting_at(Time("2026-03-15T04:00:00", scale="utc"))
            .build()
        )
    return {
        "site": site,
        "coords": Coordinates(site),
        "trajectory": trajectory,
        "traj": trajectory,
    }


#: PointingWarning-family advisories a module's docstring examples may emit.
#: A published example should stay advisory-clean, the rule the rst guard
#: applies to the pages. The one entry is the example that trips an
#: advisory today; it leaves the registry when its example is shrunk.
_EXPECT_DOCTEST_WARNINGS: dict[str, set[str]] = {
    "fyst_trajectories.planning.daisy": {"AccelerationLimitWarning", "PointingWarning"},
}


@pytest.mark.parametrize("name", MODULE_NAMES)
def test_module_docstring_examples(name, _doctest_globs, tmp_path, monkeypatch):
    """Every doctest in the module passes and trips no undeclared advisory."""
    if name.startswith(f"{PACKAGE}.visualization"):
        pytest.importorskip("matplotlib")
    monkeypatch.chdir(tmp_path)
    mod = importlib.import_module(name)
    report = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runner = doctest.DocTestRunner(optionflags=_FLAGS)
        finder = doctest.DocTestFinder()
        for test in finder.find(mod, mod.__name__, extraglobs=dict(_doctest_globs)):
            if not HAVE_SUN_AVOIDANCE and _needs_sun_avoidance(test):
                continue
            runner.run(test, out=report.write)
    results = runner.summarize(verbose=False)
    if results.failed:
        pytest.fail(
            f"{results.failed} of {results.attempted} docstring example(s) in "
            f"{name} failed:\n{report.getvalue()}"
        )
    emitted = {type(w.message).__name__ for w in caught if isinstance(w.message, PointingWarning)}
    expected = _EXPECT_DOCTEST_WARNINGS.get(name, set())
    assert emitted == expected, (
        f"{name}: docstring examples emitted {sorted(emitted)}, expected {sorted(expected)}; "
        "shrink the example or declare the advisory in _EXPECT_DOCTEST_WARNINGS"
    )


@pytest.mark.parametrize("name", MODULE_NAMES)
def test_skipped_docstring_examples_still_parse_and_bind(name, _doctest_globs):
    """A ``# doctest: +SKIP`` example must still parse and bind its calls.

    Skipped examples never execute, so nothing else keeps them from rotting,
    and they are published: autodoc renders the Examples block and Sphinx
    strips the directive. The rst guard closes the same hole for its own skip
    list. Each call to a resolvable package symbol is bound against the real
    signature, so a removed or newly required parameter fails here; a call
    abbreviated with a literal ``...`` argument is resolution-checked only.
    """
    if name.startswith(f"{PACKAGE}.visualization"):
        pytest.importorskip("matplotlib")
    mod = importlib.import_module(name)
    for test in doctest.DocTestFinder().find(mod, mod.__name__):
        sources = [ex.source for ex in test.examples if ex.options.get(doctest.SKIP)]
        if not sources:
            continue
        namespace = {**vars(fyst_trajectories), **vars(mod), **_doctest_globs}
        namespace.update(_example_imports("".join(ex.source for ex in test.examples)))
        tree = ast.parse("".join(sources))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = _call_target(node.func, namespace)
            if func is None or not callable(func):
                continue
            if any(isinstance(a, ast.Constant) and a.value is Ellipsis for a in node.args):
                continue  # explicit `...` abbreviation: resolution check only
            if any(isinstance(a, ast.Starred) for a in node.args):
                continue
            positional = [object()] * len(node.args)
            keywords = {kw.arg: None for kw in node.keywords if kw.arg is not None}
            try:
                inspect.signature(func).bind(*positional, **keywords)
            except TypeError as exc:
                pytest.fail(
                    f"{test.name}: a skipped example calls {ast.unparse(node.func)}() "
                    f"in a way that no longer binds: {exc}"
                )
