"""Pytest fixtures for fyst-trajectories tests.

Notes
-----
Hard-coded observation times in tests (e.g. ``Time("2026-03-15T04:00:00")``)
should remain within the range of the vendored IERS table at
``tests/data/finals2000A.all`` (currently covering through ~2027). When the test
epochs approach that limit, re-cut the snapshot (see ``tests/data/README.md``).
"""

from pathlib import Path

import pytest
from astropy.utils import iers

from fyst_trajectories import Coordinates, get_fyst_site

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


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )


# The ``slow`` marker is registered in pyproject.toml under
# ``[tool.pytest.ini_options].markers``; no ``pytest_configure`` hook
# needed here.


def pytest_collection_modifyitems(config, items):
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
