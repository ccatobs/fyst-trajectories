"""Shared constants and the block builder of the source-CES planner tests."""

from __future__ import annotations

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
