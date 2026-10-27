"""Shared constants, the block builder and the pass comparison of the source-CES planner tests."""

from __future__ import annotations

import numpy as np
from astropy.time import Time

from fyst_trajectories import PRIMECAM_MODULES, plan_source_ces

# Constants used across multiple tests. These dates and elevations were
# picked from a sweep over 2026 to give well-behaved Jupiter/sidereal
# arcs at FYST.
_JUPITER_NIGHT = Time("2026-03-15T00:00:00", scale="utc")
_FULL_PRIMECAM_MODULES = [PRIMECAM_MODULES[k] for k in ("c", "i1", "i2", "i3", "i4", "i5", "i6")]

# A Jupiter rising anchor on the test night (el ~ 32.6 deg, climbing). Reuses
# the ~21:41 UTC rising window the pass tests lean on.
_JUPITER_RISING_ANCHOR = Time("2026-03-15T21:41:00", scale="utc")

# Paddings the planners refuse as malformed, by test id: a NaN, text, an
# integer past the float range, and paddings wider than a full turn (9e307
# among them, whose doubled throw would overflow). None is the default.
_MALFORMED_AZ_PADDINGS = {
    "nan": float("nan"),
    "text": "0.5",
    "9e307": 9e307,
    "int-1e400": 10**400,
    "past-a-turn": 361.0,
}


def _full_primecam_block(site, **overrides):
    """Build a full-PrimeCam Jupiter-rising CES block (test convenience)."""
    kwargs = dict(
        body="jupiter",
        footprint=_FULL_PRIMECAM_MODULES,
        el_bore=35.0,
        night=_JUPITER_NIGHT,
        mode="rising",
        site=site,
    )
    kwargs.update(overrides)
    return plan_source_ces(**kwargs)


def _assert_same_pass(got, want):
    """Assert two blocks are the same pass to the bit.

    Every sample, the start instant, the duration and every solved value.
    """
    for name in ("times", "az", "el", "az_vel", "el_vel", "scan_flag"):
        assert np.array_equal(getattr(got.trajectory, name), getattr(want.trajectory, name)), name
    a, b = got.trajectory.start_time, want.trajectory.start_time
    assert (a.jd1, a.jd2) == (b.jd1, b.jd2)
    assert got.duration == want.duration
    assert got.computed_params == want.computed_params
