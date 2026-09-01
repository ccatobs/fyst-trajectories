"""Run every ``>>>`` docstring example in the package, so published Examples cannot drift.

Autodoc renders every NumPy ``Examples`` block and ``sphinx.ext.viewcode`` serves the
annotated source, so docstring examples are published output; before this guard none of
them ever executed. Each module's docstrings run under :mod:`doctest` with
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

import doctest
import importlib
import io
import os
import pkgutil
import warnings

import pytest
from astropy.time import Time

import fyst_trajectories
from fyst_trajectories import Coordinates, get_fyst_site
from fyst_trajectories.exceptions import PointingWarning
from fyst_trajectories.patterns import ConstantElScanConfig, TrajectoryBuilder

# Plotting examples run headless.
os.environ.setdefault("MPLBACKEND", "Agg")

_FLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE

try:
    import sun_avoidance  # noqa: F401

    HAVE_SUN_AVOIDANCE = True
except ImportError:
    HAVE_SUN_AVOIDANCE = False


def _needs_sun_avoidance(test: doctest.DocTest) -> bool:
    """Whether a doctest's examples select a shared-library sun model."""
    source = "".join(example.source for example in test.examples)
    return any(marker in source for marker in ('"cad"', "'cad'", '"cone"', "'cone'"))


def _module_names():
    names = ["fyst_trajectories"]
    for info in pkgutil.walk_packages(fyst_trajectories.__path__, "fyst_trajectories."):
        names.append(info.name)
    return sorted(names)


MODULE_NAMES = _module_names()


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


@pytest.mark.parametrize("name", MODULE_NAMES)
def test_module_docstring_examples(name, _doctest_globs, tmp_path, monkeypatch):
    """Every doctest in the module passes (with the seeded ambient namespace)."""
    if name.startswith("fyst_trajectories.visualization"):
        pytest.importorskip("matplotlib")
    monkeypatch.chdir(tmp_path)
    mod = importlib.import_module(name)
    report = io.StringIO()
    with warnings.catch_warnings():
        # Doctests assert printed output, not advisories; warning hygiene for the
        # executed doc surface is owned by test_doc_examples_rst.py.
        warnings.simplefilter("ignore")
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
