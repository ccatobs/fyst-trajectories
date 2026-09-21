"""Regression: every scheduled CE science block must reconstruct.

The scheduler gates CE emission on the planner's own crossing solve
(``_ce_visit_plan``), falls over to the setting half once the rising pass
is spent, and stamps each subscan with the visit anchor
(``metadata["t0_scan"]``) so every slice re-solves from the anchor the
gate guaranteed. Without that gate a constant-elevation patch scheduled
across its transit emits rising-half blocks for hours past the crossing
pass's opening, and ``schedule_to_trajectories`` can rebuild only the
blocks anchored before it (5 of 8 on this night).
"""

import warnings

import pytest
from astropy.time import Time

from fyst_trajectories import Coordinates, get_fyst_site
from fyst_trajectories.overhead import (
    BlockType,
    ObservingPatch,
    generate_timeline,
    read_timeline,
    schedule_to_trajectories,
    write_timeline,
)
from fyst_trajectories.overhead.scheduler.helpers import _ce_crossing_corridor

_AUGUST_PATCH = ObservingPatch(
    name="AugustCES",
    ra_center=330.0,
    dec_center=-50.0,
    width=30.0,
    height=8.0,
    scan_type="constant_el",
    velocity=1.0,
    elevation=50.0,
)


@pytest.fixture(scope="module")
def august_ce_timeline():
    """Build the night this file pins: one CE patch observed across its transit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return generate_timeline(
            patches=[_AUGUST_PATCH],
            site=get_fyst_site(),
            start_time="2026-08-01T23:00:00",
            end_time="2026-08-02T07:00:00",
        )


def _science(timeline):
    return [b for b in timeline.blocks if b.block_type == BlockType.SCIENCE]


def test_every_scheduled_ce_block_reconstructs(august_ce_timeline):
    science = _science(august_ce_timeline)
    assert science, "expected science blocks on the failing night"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pairs = schedule_to_trajectories(august_ce_timeline)
    assert len(pairs) == len(science), (
        f"{len(science) - len(pairs)} of {len(science)} scheduled science "
        f"blocks were not reconstructable"
    )


def test_ce_blocks_carry_visit_anchor(august_ce_timeline):
    """Every CE subscan stamps t0_scan, at or before its own start.

    The 5 ms tolerance absorbs the ISO-millisecond rounding of the stamp.
    Each visit's FIRST science block must additionally start within one
    boundary retune of its anchor (the leading scan-coupled retune is the
    only thing between the anchor and the first subscan).
    """
    from fyst_trajectories.overhead import OverheadModel

    retune_dur = OverheadModel().retune_duration
    science = _science(august_ce_timeline)
    assert science, "expected science blocks on the CE corridor night"
    by_anchor: dict[str, list] = {}
    for block in science:
        assert "t0_scan" in block.metadata
        anchor = Time(block.metadata["t0_scan"], scale="utc")
        assert anchor.unix <= block.t_start.unix + 5e-3
        by_anchor.setdefault(block.metadata["t0_scan"], []).append(block)
    for anchor_iso, group in by_anchor.items():
        anchor = Time(anchor_iso, scale="utc")
        first_start = min(b.t_start.unix for b in group)
        assert first_start - anchor.unix <= retune_dur + 5e-3


def test_default_half_falls_over_to_setting(august_ce_timeline):
    """Once the rising pass is spent, emission continues on the setting half."""
    science = _science(august_ce_timeline)
    halves = {bool(b.rising) for b in science}
    assert halves == {True, False}, f"expected both halves on this night, got {halves}"
    # The rising visit must not outlive its pass: every rising block starts
    # before the first setting block (the fall-over point).
    first_setting = min(b.t_start.unix for b in science if not b.rising)
    assert all(b.t_start.unix < first_setting for b in science if b.rising)


def test_ce_visits_end_inside_their_corridor(august_ce_timeline):
    """A visit's boundary retunes share its corridor, they do not extend it.

    The observable duration a CE visit is scored on ends when the pass
    closes. Booking the retunes between subscans on top of that budget
    would overrun the close by one retune per subscan (300 s each), and
    those trailing blocks point at a field whose RA edges have already
    crossed the scan elevation. The visit's blocks share one wall-clock
    budget instead, so the last subscan is clipped.

    The 1 s tolerance covers the ISO-millisecond stamp of the anchor and
    the 30 s-step crossing search re-run here from that anchor.
    """
    coords = Coordinates(get_fyst_site())
    by_anchor: dict[str, list] = {}
    for block in _science(august_ce_timeline):
        by_anchor.setdefault(block.metadata["t0_scan"], []).append(block)
    assert len(by_anchor) >= 2, "expected more than one CE visit on this night"

    for anchor_iso, group in by_anchor.items():
        anchor = Time(anchor_iso, scale="utc")
        window = _ce_crossing_corridor(
            _AUGUST_PATCH, 50.0, bool(group[0].rising), anchor, coords, {}
        )
        assert window is not None, f"the visit anchored at {anchor_iso} has no plannable pass"
        _, t_close = window
        visit_end = max(b.t_stop.unix for b in group)
        overrun = visit_end - t_close.unix
        assert overrun <= 1.0, (
            f"the visit anchored at {anchor_iso} runs {overrun:.0f} s past its corridor close"
        )


def test_reconstructs_after_ecsv_round_trip(august_ce_timeline, tmp_path):
    """t0_scan survives ECSV and the rebuilt count still matches."""
    path = tmp_path / "august_ce.ecsv"
    write_timeline(august_ce_timeline, str(path))
    loaded = read_timeline(str(path))
    science = _science(loaded)
    assert all("t0_scan" in b.metadata for b in science)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pairs = schedule_to_trajectories(loaded)
    assert len(pairs) == len(science)
