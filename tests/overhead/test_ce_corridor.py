"""Regression: every scheduled CE science block must reconstruct.

The 2026-07-15 repro: a constant-elevation patch scheduled across its
transit kept emitting rising-half blocks until transit, hours after the
crossing pass's opening, and ``schedule_to_trajectories`` could rebuild
only the blocks anchored before the opening (5 of 8 lost). The scheduler
now gates CE emission on the planner's own crossing solve
(``_ce_visit_plan``), falls over to the setting half once the rising pass
is spent, and stamps each subscan with the visit anchor
(``metadata["t0_scan"]``) so every slice re-solves from the anchor the
gate guaranteed.
"""

import warnings

import pytest
from astropy.time import Time

from fyst_trajectories import get_fyst_site
from fyst_trajectories.overhead import (
    BlockType,
    ObservingPatch,
    generate_timeline,
    read_timeline,
    schedule_to_trajectories,
    write_timeline,
)


@pytest.fixture(scope="module")
def august_ce_timeline():
    """Build the exact repro night: one CE patch observed across its transit."""
    patch = ObservingPatch(
        name="AugustCES",
        ra_center=330.0,
        dec_center=-50.0,
        width=30.0,
        height=8.0,
        scan_type="constant_el",
        velocity=1.0,
        elevation=50.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return generate_timeline(
            patches=[patch],
            site=get_fyst_site(),
            start_time="2026-08-01T23:00:00",
            end_time="2026-08-02T07:00:00",
        )


def _science(timeline):
    return [b for b in timeline.blocks if b.block_type == BlockType.SCIENCE]


def test_every_scheduled_ce_block_reconstructs(august_ce_timeline):
    science = _science(august_ce_timeline)
    assert science, "expected science blocks on the repro night"
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
    by_anchor: dict[str, list] = {}
    for block in _science(august_ce_timeline):
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
