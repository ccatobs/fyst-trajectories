"""Tests for ElevationConstraint default bounds.

Verifies that :class:`ElevationConstraint` constructed without arguments
matches the FYST telescope elevation limits (``FYST_EL_MIN = 20``,
``FYST_EL_MAX = 90``) rather than the historical more-restrictive
defaults of ``el_min=30.0, el_max=70.0``.
"""

import pytest
from astropy.time import Time

from fyst_trajectories import Coordinates, get_fyst_site
from fyst_trajectories.overhead.constraints import ElevationConstraint
from fyst_trajectories.overhead.models import ObservingPatch
from fyst_trajectories.site import FYST_EL_MAX, FYST_EL_MIN


@pytest.fixture
def patch():
    """Return a minimal ObservingPatch for constraint scoring."""
    return ObservingPatch(
        name="test",
        ra_center=180.0,
        dec_center=-30.0,
        width=4.0,
        height=4.0,
        scan_type="pong",
        velocity=0.5,
    )


@pytest.fixture
def coords():
    """Return a Coordinates instance for the default FYST site."""
    return Coordinates(get_fyst_site())


@pytest.fixture
def time():
    """Return a fixed observation time for deterministic tests."""
    return Time("2026-06-15T04:00:00", scale="utc")


def test_default_el_max_matches_fyst_el_max():
    """Default ``el_max`` should equal ``FYST_EL_MAX`` (90 deg)."""
    c = ElevationConstraint()
    assert c.el_max == FYST_EL_MAX
    assert c.el_max == 90.0


def test_default_el_min_matches_fyst_el_min():
    """Default ``el_min`` should equal ``FYST_EL_MIN`` (20 deg)."""
    c = ElevationConstraint()
    assert c.el_min == FYST_EL_MIN
    assert c.el_min == 20.0


def test_default_allows_elevation_89_deg(patch, time, coords):
    """An elevation of 89 deg must score as valid under the new defaults.

    Under the previous default (``el_max=70.0``) this case was scored
    infeasible (0.0).  The new default ``el_max=90.0`` must make it
    valid (1.0).
    """
    c = ElevationConstraint()
    assert c.score(patch, time, 180.0, 89.0, coords) == 1.0


def test_default_rejects_elevation_below_fyst_min(patch, time, coords):
    """An elevation of 10 deg must still score infeasible under defaults."""
    c = ElevationConstraint()
    assert c.score(patch, time, 180.0, 10.0, coords) == 0.0


def test_default_rejects_elevation_above_fyst_max(patch, time, coords):
    """An elevation of 95 deg must still score infeasible under defaults."""
    c = ElevationConstraint()
    assert c.score(patch, time, 180.0, 95.0, coords) == 0.0
