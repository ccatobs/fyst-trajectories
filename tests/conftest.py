"""Pytest fixtures for fyst-trajectories tests.

Notes
-----
Hard-coded observation times in tests (e.g. ``Time("2026-03-15T04:00:00")``)
should remain within the range of the vendored IERS table at
``tests/data/finals2000A.all`` (currently covering through ~2027). When the test
epochs approach that limit, re-cut the snapshot (see ``tests/data/README.md``).
"""

import os
from collections import defaultdict
from pathlib import Path

import _tiers
import pytest
from astropy.utils import iers
from hypothesis import HealthCheck, settings

from fyst_trajectories import Coordinates, get_fyst_site

# Hypothesis's too_slow health check times data generation by the wall clock, so
# on a loaded machine (parallel test runs, a busy CI runner) it fails whatever the
# code under test does. The suite's profile suppresses that check and nothing
# else: filter_too_much and data_too_large still flag a strategy that wastes its
# draws. It is loaded here, before any test module is imported, because a
# @settings decorator inherits from the profile active when it runs. A test that
# passes its own suppress_health_check list replaces this one, so the list must
# repeat HealthCheck.too_slow.
settings.register_profile("fyst", suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("fyst")

# Pin astropy Earth-orientation handling to a vendored IERS table at import time,
# before any test is collected or run. On a cold CI runner there is no cached
# finals2000A.all, so stock astropy either downloads it from datacenter.iers.org
# (the network stall that timed out CI) or falls back to a bundled table that
# does not cover the 2026 test epochs, raising IERSRangeError on every transform.
# Loading the vendored finals2000A.all and disabling auto-download keeps the suite
# offline, deterministic, and full-accuracy: identical EOP locally and in CI.
# Re-cut the snapshot when the test epochs approach its range.
_IERS_A_FILE = Path(__file__).parent / "data" / "finals2000A.all"
iers.conf.auto_download = False
if _IERS_A_FILE.exists():
    iers.earth_orientation_table.set(iers.IERS_A.open(str(_IERS_A_FILE)))
# A few tests deliberately probe far-future epochs (e.g. the 2029-2035 precession
# checks) that lie past any real IERS table. Degrade gracefully to UT1-UTC=0 there
# instead of raising, and keep that deterministic regardless of test order.
iers.conf.iers_degraded_accuracy = "warn"

# A child interpreter a test launches (the bundled examples run as subprocesses)
# inherits none of the pin above: it would start with auto-download on and
# reach for the network, for the EOP table or for a leap-second update. The
# pin is passed in through a sitecustomize shim on the child's import path,
# inert unless FYST_IERS_TABLE is set, so a reader still runs the examples
# exactly as documented.
_EXAMPLE_SITECUSTOMIZE = '''"""Apply the test suite's vendored IERS pin in a subprocess."""

import os

from astropy.utils import iers

_table = os.environ.get("FYST_IERS_TABLE")
if _table:
    iers.conf.auto_download = False
    iers.earth_orientation_table.set(iers.IERS_A.open(_table))
    iers.conf.iers_degraded_accuracy = "warn"
'''


@pytest.fixture(scope="session")
def example_env(tmp_path_factory):
    """Environment for a child interpreter that carries this suite's IERS pin."""
    directory = tmp_path_factory.mktemp("iers_pin")
    (directory / "sitecustomize.py").write_text(_EXAMPLE_SITECUSTOMIZE, encoding="utf-8")
    env = dict(os.environ)
    env["FYST_IERS_TABLE"] = str(_IERS_A_FILE)
    inherited = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{directory}{os.pathsep}{inherited}" if inherited else str(directory)
    return env


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )
    parser.addoption(
        "--fail-on-skip",
        action="store_true",
        default=False,
        help="End with a failing exit status if any selected test or module was skipped",
    )


class _FailOnSkip:
    """Turn a passing run that skipped anything into a failing one, and name the skips.

    For a run whose purpose is one comparison, where a skip means the comparison
    did not happen. An expected failure (xfail) is not a skip.
    """

    def __init__(self):
        self.skipped = []

    def _record(self, report):
        if report.skipped and not hasattr(report, "wasxfail"):
            self.skipped.append(report.nodeid)

    def pytest_collectreport(self, report):
        self._record(report)

    def pytest_runtest_logreport(self, report):
        self._record(report)

    def pytest_sessionfinish(self, session):
        if self.skipped and session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter):
        if not self.skipped:
            return
        terminalreporter.section("skipped under --fail-on-skip", red=True)
        for nodeid in self.skipped:
            terminalreporter.write_line(nodeid)


def pytest_configure(config):
    if config.getoption("--fail-on-skip"):
        config.pluginmanager.register(_FailOnSkip(), "fail-on-skip")


def _mark_simulator_tier(items):
    """Apply the ``offline`` marker to every item that exercises the simulator.

    A test module is classified by the modules it imports (``_tiers``, the one
    boundary definition), not by the directory it sits in. A module that
    already marks items itself owns its classification: the documentation
    guards split their parameters page by page, and a few files mark the
    single test in them that reaches into the simulator, so a blanket
    module-level mark would move their library-tier tests to the wrong job.
    """
    by_module = defaultdict(list)
    for item in items:
        by_module[Path(str(item.fspath))].append(item)
    for path, module_items in by_module.items():
        if any(item.get_closest_marker("offline") for item in module_items):
            continue
        if not _tiers.module_is_simulator_tier(path):
            continue
        for item in module_items:
            item.add_marker(pytest.mark.offline)


def pytest_collection_modifyitems(config, items):
    _mark_simulator_tier(items)
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def site():
    """Provide a default FYST site for testing."""
    return get_fyst_site()


@pytest.fixture
def coordinates(site):
    """Provide a Coordinates instance for testing."""
    return Coordinates(site)
