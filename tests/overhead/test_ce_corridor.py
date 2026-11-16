"""Regression: every scheduled CE science block must reconstruct.

The scheduler gates CE emission on the planner's own crossing solve
(``_ce_visit_plan``), falls over to the setting half once the rising pass
is spent, and stamps each subscan with the visit anchor
(``metadata["t0_scan"]``) so every slice re-solves from the anchor the
gate guaranteed. Without that gate a constant-elevation patch scheduled
across its transit emits rising-half blocks for hours past the crossing
pass's opening, and ``schedule_to_trajectories`` can rebuild only the
blocks anchored before it. The gate reads a ``PointingError`` from the solve
as "no plannable pass" and lets any other error through, and leaves the
solve's advisories to the rebuild.
"""

import warnings
from dataclasses import replace

import numpy as np
import pytest
from _scheduler_helpers import _initial_state, _make_ctx
from astropy.time import Time, TimeDelta

from fyst_trajectories import Coordinates, get_fyst_site
from fyst_trajectories.exceptions import PointingError, PointingWarning
from fyst_trajectories.overhead import (
    BlockType,
    CalibrationPolicy,
    ObservingPatch,
    OverheadModel,
    generate_timeline,
    read_timeline,
    schedule_to_trajectories,
    write_timeline,
)
from fyst_trajectories.overhead.scheduler import helpers as _helpers
from fyst_trajectories.overhead.scheduler import phases as _phases
from fyst_trajectories.overhead.scheduler.helpers import (
    _ce_crossing_corridor,
    _ce_swept_az_envelope,
)
from fyst_trajectories.planning import FieldRegion, plan_constant_el_scan
from fyst_trajectories.trajectory_utils import get_absolute_times

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
    return generate_timeline(
        patches=[_AUGUST_PATCH],
        site=get_fyst_site(),
        start_time="2026-08-01T23:00:00",
        end_time="2026-08-02T07:00:00",
    )


def _science(timeline):
    return [b for b in timeline.blocks if b.block_type == BlockType.SCIENCE]


def _short_retune_overhead(retune_duration):
    return OverheadModel(
        retune_duration=retune_duration,
        pointing_cal_duration=0.0,
        focus_duration=0.0,
        skydip_duration=0.0,
        planet_cal_duration=0.0,
        beam_map_duration=0.0,
    )


@pytest.fixture(scope="module", params=[0.0, 60.0], ids=["zero", "short-retune"])
def ce_wait_night(request):
    return request.param, generate_timeline(
        patches=[_AUGUST_PATCH],
        site=get_fyst_site(),
        start_time="2026-08-01T23:00:00",
        end_time="2026-08-02T07:00:00",
        overhead_model=_short_retune_overhead(request.param),
        time_step=30.0,
    )


def test_ce_wait_night_is_tiled_and_every_science_block_reconstructs(ce_wait_night):
    _, timeline = ce_wait_night
    for left, right in zip(timeline.blocks, timeline.blocks[1:]):
        assert abs((left.t_stop - right.t_start).sec) < 1e-3
    pairs = schedule_to_trajectories(timeline)
    assert len(pairs) == len(_science(timeline)) > 0
    for block, rebuilt in pairs:
        assert rebuilt.trajectory.times.size > 0
        if block.subscan_index == 0:
            first_sample = get_absolute_times(rebuilt.trajectory)[0]
            assert 0.0 <= (first_sample - block.t_start).sec < 0.1


def test_ce_wait_precedes_retune_and_science_and_preserves_anchor(ce_wait_night):
    retune_duration, timeline = ce_wait_night
    coords = Coordinates(get_fyst_site())
    visits: dict[int, list] = {}
    for block in _science(timeline):
        visits.setdefault(block.scan_index, []).append(block)
    assert len(visits) >= 2
    assert any(len(group) > 1 for group in visits.values())
    for group in visits.values():
        first = group[0]
        assert first.subscan_index == 0
        anchor_iso = first.metadata["t0_scan"]
        assert all(b.metadata["t0_scan"] == anchor_iso for b in group)
        anchor = Time(anchor_iso, scale="utc")
        t_open, t_close = _ce_crossing_corridor(
            _AUGUST_PATCH, 50.0, bool(first.rising), anchor, coords, {}
        )
        wait_duration = (t_open - anchor).sec - retune_duration
        assert wait_duration > 0.0
        index = timeline.blocks.index(first)
        if retune_duration:
            retune = timeline.blocks[index - 1]
            assert retune.block_type == BlockType.CALIBRATION
            assert retune.scan_type == "retune"
            assert retune.duration == pytest.approx(retune_duration)
            assert abs((retune.t_stop - first.t_start).sec) < 1e-3
            index -= 1
        idle = timeline.blocks[index - 1]
        assert idle.block_type == BlockType.IDLE
        assert idle.metadata["reason"] == "waiting_for_pass"
        assert idle.scan_index == first.scan_index
        assert idle.duration == pytest.approx(wait_duration, abs=5e-3)
        assert abs((idle.t_start - anchor).sec) < 1e-3
        assert abs((idle.t_stop - timeline.blocks[index].t_start).sec) < 1e-3
        previous = timeline.blocks[index - 2]
        assert idle.az_start == idle.az_end == previous.end_pose_az
        assert idle.elevation == previous.elevation
        assert abs((first.t_start - t_open).sec) < 5e-3
        assert (first.t_start - anchor).sec == pytest.approx(
            idle.duration + retune_duration, abs=1e-3
        )
        assert (group[-1].t_stop - t_close).sec <= 0.01


def test_ce_default_overhead_needs_no_pass_wait():
    timeline = generate_timeline(
        patches=[_AUGUST_PATCH],
        site=get_fyst_site(),
        start_time="2026-08-01T23:00:00",
        end_time="2026-08-02T07:00:00",
        time_step=30.0,
    )
    assert _science(timeline)
    for index, block in enumerate(timeline.blocks):
        if block.block_type != BlockType.SCIENCE or block.subscan_index != 0:
            continue
        retune = timeline.blocks[index - 1]
        assert retune.scan_type == "retune"
        assert timeline.blocks[index - 2].metadata.get("reason") != "waiting_for_pass"


@pytest.fixture
def ce_wait_phase():
    coords = Coordinates(get_fyst_site())
    cache = {}
    t_open, _ = _ce_crossing_corridor(
        _AUGUST_PATCH, 50.0, True, Time("2026-08-01T23:00:00"), coords, cache
    )
    start = t_open - TimeDelta(180.0, format="sec")
    ctx = _make_ctx(
        patches=[_AUGUST_PATCH],
        start_time=start,
        end_time=t_open + TimeDelta(600.0, format="sec"),
        overhead_model=_short_retune_overhead(60.0),
        time_step=30.0,
    )
    ctx.ce_corridors.update(cache)
    state = _initial_state(ctx).advanced(current_az=100.0, current_el=50.0)
    selection = _phases.PhaseResult(
        state=state, selection=_AUGUST_PATCH, best_az=100.0, best_el=50.0
    )
    return state, ctx, selection


@pytest.mark.parametrize("budget", [179.0, 239.0, None], ids=["too-short", "retune", "refused"])
def test_ce_wait_is_discarded_when_no_subscan_fits(ce_wait_phase, monkeypatch, budget):
    state, ctx, selection = ce_wait_phase
    if budget is None:
        monkeypatch.setattr(_phases, "_plan_science_subscan", lambda *args, **kwargs: None)
    else:
        # The 120 s wait leaves less than a minimum scan, or retune + scan.
        monkeypatch.setattr(_phases, "_compute_scan_duration", lambda *args, **kwargs: budget)
    result = _phases.ScienceScanPhase().run(state, ctx, selection=selection)
    assert result.skip_to_next_iter
    assert len(result.blocks) == 1
    idle = result.blocks[0]
    assert idle.block_type == BlockType.IDLE
    assert idle.metadata.get("reason") == ("unplannable" if budget is None else None)
    assert abs((idle.t_start - state.current_time).sec) < 1e-6
    assert idle.duration == pytest.approx(ctx.time_step)
    assert result.state.current_time == idle.t_stop
    assert result.state.cal_state == state.cal_state
    assert result.state.scan_counter == state.scan_counter
    assert (result.state.current_az, result.state.current_el) == (100.0, 50.0)


def test_ce_wait_keeps_first_retune_due_check_at_entry(ce_wait_phase):
    state, ctx, selection = ce_wait_phase
    ctx = replace(ctx, calibration_policy=CalibrationPolicy(retune_cadence=90.0))
    state = state.advanced(cal_state=state.cal_state.update("retune", state.current_time))
    result = _phases.ScienceScanPhase().run(state, ctx, selection=selection)
    assert [b.block_type for b in result.blocks] == [BlockType.IDLE, BlockType.SCIENCE]
    assert result.blocks[0].duration == pytest.approx(180.0)
    assert result.state.cal_state == state.cal_state


def test_ce_wait_is_removed_before_subscan_split(ce_wait_phase):
    state, ctx, selection = ce_wait_phase
    ctx = replace(ctx, overhead_model=replace(ctx.overhead_model, max_scan_duration=700.0))
    result = _phases.ScienceScanPhase().run(state, ctx, selection=selection)
    assert [b.block_type for b in result.blocks] == [
        BlockType.IDLE,
        BlockType.CALIBRATION,
        BlockType.SCIENCE,
    ]
    assert [b.duration for b in result.blocks] == pytest.approx([120.0, 60.0, 600.0])
    assert abs((result.blocks[-1].t_stop - ctx.end_time).sec) < 1e-6


def test_ce_wait_checks_sun_escape_at_entry(ce_wait_phase, monkeypatch):
    state, ctx, selection = ce_wait_phase
    escape = _phases.PhaseResult(state=state, stop=True)

    def check_escape(actual_state, actual_ctx):
        assert actual_state is state
        assert actual_ctx is ctx
        return escape

    monkeypatch.setattr(_phases, "_escape_if_overtaken", check_escape)
    assert _phases.ScienceScanPhase().run(state, ctx, selection=selection) is escape


def test_every_scheduled_ce_block_reconstructs(august_ce_timeline):
    science = _science(august_ce_timeline)
    assert science, "expected science blocks on this night"
    pairs = schedule_to_trajectories(august_ce_timeline)
    assert len(pairs) == len(science), (
        f"{len(science) - len(pairs)} of {len(science)} scheduled science "
        f"blocks were not reconstructable"
    )


def test_ce_blocks_carry_visit_anchor(august_ce_timeline):
    """Every CE subscan stamps t0_scan, at or before its own start.

    The 5 ms tolerance absorbs the ISO-millisecond rounding of the stamp.
    Each visit's first science block follows its boundary retune and any
    wait that retune cannot cover. This fixture uses the native 300 s
    scheduling step, whose readiness lead can exceed the default retune.
    """
    retune_dur = OverheadModel().retune_duration
    coords = Coordinates(get_fyst_site())
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
        t_open, _ = _ce_crossing_corridor(
            _AUGUST_PATCH, 50.0, bool(group[0].rising), anchor, coords, {}
        )
        expected_lead = max(retune_dur, (t_open - anchor).sec)
        assert first_start - anchor.unix == pytest.approx(expected_lead, abs=0.01)


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

    The 10 ms tolerance covers the ISO-millisecond stamp of the anchor the
    crossing search is re-run from here.
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
        assert overrun <= 0.01, (
            f"the visit anchored at {anchor_iso} runs {overrun:.3f} s past its corridor close"
        )


def test_reconstructs_after_ecsv_round_trip(august_ce_timeline, tmp_path):
    """t0_scan survives ECSV and the rebuilt count still matches."""
    path = tmp_path / "august_ce.ecsv"
    write_timeline(august_ce_timeline, str(path))
    loaded = read_timeline(str(path))
    science = _science(loaded)
    assert all("t0_scan" in b.metadata for b in science)
    pairs = schedule_to_trajectories(loaded)
    assert len(pairs) == len(science)


def test_a_night_given_in_tt_is_the_utc_night_held_in_utc(august_ce_timeline, tmp_path):
    """The window given as TT ``Time`` objects plans the UTC night, held and recorded in UTC.

    Every block falls at the UTC night's instants, each CE subscan records the
    UTC night's ``t0_scan``, the file keeps the instants, and every subscan
    rebuilds as the UTC night's does.
    """
    night = generate_timeline(
        patches=[_AUGUST_PATCH],
        site=get_fyst_site(),
        start_time=Time("2026-08-01T23:00:00", scale="utc").tt,
        end_time=Time("2026-08-02T07:00:00", scale="utc").tt,
    )
    assert (night.start_time.scale, night.end_time.scale) == ("utc", "utc")
    assert len(night.blocks) == len(august_ce_timeline.blocks)
    for block, utc_block in zip(night.blocks, august_ce_timeline.blocks):
        assert (block.t_start.scale, block.t_stop.scale) == ("utc", "utc")
        assert abs((block.t_start - utc_block.t_start).to_value("s")) < 1e-6
        assert block.metadata.get("t0_scan") == utc_block.metadata.get("t0_scan")

    path = tmp_path / "august_ce_tt.ecsv"
    write_timeline(night, str(path))
    loaded = read_timeline(str(path))
    for block, back in zip(night.blocks, loaded.blocks, strict=True):
        assert abs((back.t_start - block.t_start).to_value("s")) <= 5e-4 + 1e-9
    pairs = schedule_to_trajectories(night)
    utc_pairs = schedule_to_trajectories(august_ce_timeline)
    assert len(schedule_to_trajectories(loaded)) == len(_science(loaded))
    assert len(pairs) == len(utc_pairs) == len(_science(night))
    for (_, rebuilt), (_, utc_rebuilt) in zip(pairs, utc_pairs):
        assert np.array_equal(rebuilt.trajectory.times, utc_rebuilt.trajectory.times)
        assert np.array_equal(rebuilt.trajectory.az, utc_rebuilt.trajectory.az)


def test_a_patch_beside_a_celestial_pole_is_skipped_not_raised():
    """The crossing solve refuses a field within 0.57 deg of a pole; the night goes on.

    ``ObservingPatch`` accepts such a declination, and the solve's flat-sky
    corner projection cannot plan it, so the refusal is a ``PointingError``
    the corridor gate reads as "no plannable pass".
    """
    polar = ObservingPatch(
        name="polar",
        ra_center=100.0,
        dec_center=-89.6,
        width=0.5,
        height=0.2,
        scan_type="constant_el",
        velocity=1.0,
        elevation=30.0,
    )
    timeline = generate_timeline(
        patches=[polar],
        site=get_fyst_site(),
        start_time="2026-04-01T00:00:00",
        end_time="2026-04-01T01:00:00",
    )
    assert timeline.blocks
    assert not _science(timeline)


def test_the_corridor_is_the_envelope_of_the_pass_the_planner_builds_at_the_limit():
    """At an azimuth limit the corridor is the rebuilt pass's envelope, a turn apart.

    At 3 deg/s this field's setting pass sweeps (345.14, 354.78) with a
    5.625 deg turnaround overshoot on each side, past the 360 deg limit, so
    the planner sweeps it one turn down. The corridor the scheduler slews to
    and clips against the Sun is the envelope of that trajectory on the
    field's own branch: the slew re-wraps it to the mount's azimuth, and the
    Sun clip reads it modulo 360 deg.
    """
    site = get_fyst_site()
    coords = Coordinates(site)
    start = Time("2026-11-15T18:00:00", scale="utc")
    patch = ObservingPatch(
        name="NorthSetting",
        ra_center=60.0,
        dec_center=26.25,
        width=2.0,
        height=2.0,
        scan_type="constant_el",
        velocity=3.0,
        elevation=40.0,
    )
    window = _ce_crossing_corridor(patch, 40.0, False, start, coords, {})
    assert window is not None
    lo, hi = _ce_swept_az_envelope(patch, *window, coords)

    block = plan_constant_el_scan(
        field=FieldRegion(ra_center=60.0, dec_center=26.25, width=2.0, height=2.0),
        elevation=40.0,
        velocity=3.0,
        site=site,
        start_time=start,
        rising=False,
    )
    az = block.trajectory.az
    assert lo - 360.0 == pytest.approx(az.min(), abs=0.01)
    assert hi - 360.0 == pytest.approx(az.max(), abs=0.01)
    lim = site.telescope_limits.azimuth
    assert lim.min <= az.min()
    assert az.max() <= lim.max


def test_a_pass_advisory_from_the_gate_is_raised_by_the_rebuild_not_the_night():
    """The crossing solve's advisory on a long pass is left to the rebuild.

    A 110 deg wide field crosses el 50 deg over 8.4 h, which the solve
    flags as unusually long. The gate runs that solve while the timeline is
    generated, and the advisory describes the pass the science block
    records, so it is raised where a science subscan's are: by the rebuild
    of the finished timeline, which runs the same solve.
    """
    wide = ObservingPatch(
        name="Wide",
        ra_center=0.0,
        dec_center=-30.0,
        width=110.0,
        height=10.0,
        scan_type="constant_el",
        velocity=1.0,
        elevation=50.0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PointingWarning)
        timeline = generate_timeline(
            patches=[wide],
            site=get_fyst_site(),
            start_time="2026-06-15T03:00:00",
            end_time="2026-06-15T04:00:00",
        )
    assert [str(w.message) for w in caught if issubclass(w.category, PointingWarning)] == []
    assert len(_science(timeline)) == 1

    with pytest.warns(PointingWarning, match=r"Computed observation duration 8\.4h is unusually"):
        pairs = schedule_to_trajectories(timeline)
    assert len(pairs) == 1


class TestCorridorSolveRefusals:
    """The corridor gate caches an infeasible solve as a miss and lets a bug through."""

    _START = Time("2026-08-01T23:00:00", scale="utc")

    def _corridor(self, monkeypatch, error):
        def _refuse(*args, **kwargs):
            raise error

        monkeypatch.setattr(_helpers, "_compute_ce_duration", _refuse)
        cache: dict = {}
        coords = Coordinates(get_fyst_site())
        window = _ce_crossing_corridor(_AUGUST_PATCH, 50.0, True, self._START, coords, cache)
        return window, cache

    def test_pointing_error_is_cached_as_a_miss(self, monkeypatch):
        window, cache = self._corridor(monkeypatch, PointingError("no crossing"))
        assert window is None
        assert cache == {("AugustCES", 50.0, True): ("miss", self._START)}

    def test_plain_value_error_propagates(self, monkeypatch):
        """A malformed argument is a caller's bug, not an unplannable pass."""
        with pytest.raises(ValueError, match="step_seconds"):
            self._corridor(monkeypatch, ValueError("step_seconds must be positive"))
