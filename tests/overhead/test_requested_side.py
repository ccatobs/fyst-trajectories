"""A constant-elevation patch that asks for one side is held to it by the crossing gate alone.

A patch asks for one side of the meridian with ``scan_params["rising"]`` or
with an azimuth window on that side. The crossing gate
(``_ce_visit_plan``) solves that side's pass only, so it refuses a pass
that is spent (the anchor at or inside it, where the planner finds none
within its horizon) and one not yet due (opening later than the readiness
lead), and admits the pass when it is imminent, wherever the field centre
then is. A setting pass opens before the centre transits when the field
culminates not far above the scan elevation; such a patch is booked.
"""

import dataclasses

import pytest
from astropy.time import Time

from fyst_trajectories import Coordinates, get_fyst_site
from fyst_trajectories.overhead import ObservingPatch, generate_timeline
from fyst_trajectories.overhead.scheduler.helpers import _ce_visit_plan

SITE = get_fyst_site()
COORDS = Coordinates(SITE)

#: A field culminating near el 63.9, scanned at el 62: its setting pass opens
#: at 06:36:31 on 2026-09-15, while the centre is still east of the meridian.
_NORTH = ObservingPatch(
    name="North",
    ra_center=30.0,
    dec_center=3.0,
    width=30.0,
    height=6.0,
    scan_type="constant_el",
    velocity=0.5,
    elevation=62.0,
)
_START = Time("2026-09-15T04:30:00", scale="utc")
_END = Time("2026-09-15T09:00:00", scale="utc")
_OPENS = Time("2026-09-15T06:36:31", scale="utc")

#: A readiness lead as the scheduler's: one tick and a slew allowance.
_LEAD = 480.0


def _setting(patch_params):
    return dataclasses.replace(_NORTH, scan_params=patch_params)


@pytest.mark.parametrize(
    "scan_params",
    [{"rising": False}, {"az_min": 330.0, "az_max": 350.0}],
    ids=["rising_false", "window"],
)
def test_the_gate_holds_a_requested_side(scan_params):
    """Not yet due, imminent, then spent: the gate decides, on the setting side only."""
    patch = _setting(scan_params)
    early = _ce_visit_plan(patch, 62.0, _START, _END, COORDS, {}, _LEAD)
    assert early is None  # the setting pass opens two hours later; the rising one is not asked

    imminent = Time("2026-09-15T06:33:00", scale="utc")
    assert COORDS.get_hour_angle(_NORTH.ra_center, imminent) < 0.0  # the centre is still east
    plan = _ce_visit_plan(patch, 62.0, imminent, _END, COORDS, {}, _LEAD)
    assert plan is not None
    rising, t_open, _ = plan
    assert rising is False
    assert abs((t_open - _OPENS).sec) < 1.0

    spent = Time("2026-09-15T07:00:00", scale="utc")
    assert _ce_visit_plan(patch, 62.0, spent, _END, COORDS, {}, _LEAD) is None


@pytest.mark.parametrize(
    ("scan_params", "science"),
    [({"rising": False}, 6608.5), ({"az_min": 330.0, "az_max": 350.0}, 6598.3)],
    ids=["rising_false", "window"],
)
def test_a_setting_pass_that_opens_before_the_transit_is_booked(scan_params, science):
    """The night books the setting pass from its opening, on the side asked for."""
    timeline = generate_timeline(
        patches=[_setting(scan_params)], site=SITE, start_time=_START, end_time=_END
    )
    blocks = timeline.science_blocks
    assert blocks[0].t_start.isot.startswith("2026-09-15T06:36:31")
    assert all(b.rising is False for b in blocks)
    assert sum(b.duration for b in blocks) == pytest.approx(science, abs=0.05)
