"""Tests for opt-in source-CES multi-pass planet calibrations.

When ``CalibrationPolicy.planet_cal_scan`` is set, a due ``planet_cal`` is
planned as a multi-pass source-CES sequence on the first listed planet that
is up and clear of the Sun, reached by a SLEW block planned with
``plan_transition`` and followed by one CALIBRATION block per pass, instead
of a single fixed-duration parked block. These tests cover the emit path,
the acquisition slew, the failure/truncation semantics, and the ECSV
round-trip of the recorded pass parameters.

Jupiter rises over Cerro Chajnantor across roughly 20:00-23:30 UTC on
2026-03-15 (the date the planning tests use), so the calibration anchors
below sit inside that rising arc.
"""

import dataclasses
import logging
import warnings

import numpy as np
import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories import get_fyst_site
from fyst_trajectories.exceptions import TargetNotObservableError
from fyst_trajectories.overhead import (
    BlockType,
    CalibrationPolicy,
    CalibrationState,
    ObservingPatch,
    ObservingTimeline,
    OverheadModel,
    accumulate_hitmaps,
    compute_budget,
    generate_timeline,
    plan_transition,
    read_timeline,
    schedule_to_trajectories,
    validate_scan_params,
    write_timeline,
)
from fyst_trajectories.overhead.scheduler import (
    CalibrationPhase,
    SchedulerContext,
    SchedulerState,
    phases,
)
from fyst_trajectories.planning import plan_source_ces
from fyst_trajectories.planning.footprints import offset_footprint_eta, resolve_footprint
from fyst_trajectories.sun_models import make_sun_safe
from fyst_trajectories.trajectory_utils import get_absolute_times

# Anchor inside Jupiter's rising arc where a 3-pass sequence is feasible.
_ANCHOR = "2026-03-15T20:45:00"
_END = "2026-03-15T23:30:00"

_SCAN_PARAM_KEYS = {
    "body",
    "footprint",
    "el_bore",
    "mode",
    "window",
    "boresight_rot",
    "timestep",
    "eta_offset_deg",
    "pass_index",
    "n_passes",
}


def _scan_policy(**overrides):
    """Policy with only ``planet_cal`` due so the passes are isolated.

    Every non-planet cadence is set well above the test window so, paired
    with a :func:`_only_planet_cal_due` state, ``needs_calibration`` returns
    ``planet_cal`` alone.
    """
    params = dict(
        retune_cadence=1.0e9,
        pointing_cadence=1.0e9,
        focus_cadence=1.0e9,
        skydip_cadence=1.0e9,
        planet_cal_cadence=1.0e9,
        planet_cal_scan=True,
        planet_cal_passes=3,
        planet_targets=("jupiter",),
        planet_min_elevation=15.0,
    )
    params.update(overrides)
    return CalibrationPolicy(**params)


def _only_planet_cal_due(anchor):
    """CalibrationState with every ``last_*`` recent except ``planet_cal``."""
    return CalibrationState(
        last_retune=anchor,
        last_pointing_cal=anchor,
        last_focus=anchor,
        last_skydip=anchor,
        last_planet_cal=None,
    )


def _ctx(policy, *, start=_ANCHOR, end=_END, **kwargs):
    return SchedulerContext.build(
        patches=[],
        site=get_fyst_site(),
        start_time=Time(start, scale="utc"),
        end_time=Time(end, scale="utc"),
        calibration_policy=policy,
        **kwargs,
    )


def _run_at(iso, az, el, targets, *, hours=4.0, **ctx_kwargs):
    """Run CalibrationPhase with only ``planet_cal`` due at one time and pose.

    Returns ``(anchor, ctx, result)``; the window runs ``hours`` past the
    anchor.
    """
    anchor = Time(iso, scale="utc")
    end = anchor + TimeDelta(hours * 3600.0, format="sec")
    ctx = _ctx(_scan_policy(planet_targets=targets), start=anchor.isot, end=end.isot, **ctx_kwargs)
    state = SchedulerState(
        current_time=anchor,
        current_az=az,
        current_el=el,
        cal_state=_only_planet_cal_due(anchor),
        scan_counter=0,
    )
    return anchor, ctx, CalibrationPhase().run(state, ctx)


# A field observable across the Jupiter-rising window whose science blocks
# reconstruct cleanly, so a flag-on timeline carries both science blocks and
# source-CES planet-cal passes (needed to exercise science + calibration
# reconstruction together in schedule_to_trajectories).
_RECON_PATCH = ObservingPatch(
    name="ReconField",
    ra_center=140.0,
    dec_center=-23.0,
    width=4.0,
    height=4.0,
    scan_type="pong",
    velocity=0.5,
)


@pytest.fixture(scope="module")
def flag_on_timeline():
    """Flag-on timeline carrying both science blocks and three planet-cal passes."""
    return generate_timeline(
        patches=[_RECON_PATCH],
        site=get_fyst_site(),
        start_time=_ANCHOR,
        end_time=_END,
        overhead_model=OverheadModel(),
        calibration_policy=CalibrationPolicy(
            planet_cal_cadence=43200.0,
            planet_cal_scan=True,
            planet_cal_passes=3,
            planet_targets=("jupiter",),
            planet_min_elevation=15.0,
        ),
    )


@pytest.fixture(scope="module")
def flag_on_recon(flag_on_timeline):
    """Reconstruction of the flag-on timeline, computed once for the module.

    Returns ``(site, science_only_pairs, science_plus_cal_pairs)``.
    """
    site = flag_on_timeline.site
    sci_only = schedule_to_trajectories(flag_on_timeline, science_only=True)
    with_cals = schedule_to_trajectories(flag_on_timeline, science_only=False)
    return site, sci_only, with_cals


def _reference_pass(block, site):
    """Independently plan the reference source-CES pass for a recorded block.

    Planned the way ``plan_source_ces_passes`` plans each pass: the recorded
    parameters (eta-shifted footprint, boresight elevation, mode) searched
    over the 24 h that follow the block's ``search_start``. A reconstruction
    that skipped the eta shift, or searched any other window, would not
    match it.
    """
    scan_params = block.metadata["scan_params"]
    base = resolve_footprint(scan_params["footprint"])
    fp = offset_footprint_eta(base, scan_params["eta_offset_deg"])
    jd1, jd2 = block.metadata["search_start"]
    start = Time(jd1, jd2, format="jd", scale="utc")
    return plan_source_ces(
        body=scan_params["body"],
        footprint=fp,
        el_bore=scan_params["el_bore"],
        boresight_rot=scan_params["boresight_rot"],
        timestep=scan_params["timestep"],
        window=(start, start + TimeDelta(24.0 * 3600.0, format="sec")),
        mode=scan_params["mode"],
        site=site,
    )


def _assert_reconstruction_faithful(block, scan_block, site):
    """Assert a rebuilt cal pass is the planned one: its window and a reference plan.

    Checks the two contracts the reconstruction must honour: the re-solved
    ``t0_iso`` / ``t1_iso`` are the recorded window, and the trajectory
    equals a reference pass planned independently from the same recorded
    parameters in the planner's own search window.
    """
    sp = block.metadata["scan_params"]
    # The rebuild repeats the planner's search, so it solves the same window.
    assert scan_block.computed_params["t0_iso"] == sp["window"][0]
    assert scan_block.computed_params["t1_iso"] == sp["window"][1]

    ref = _reference_pass(block, site)
    # Same planner, same recorded inputs: the rebuild is exact.
    assert np.array_equal(scan_block.trajectory.az, ref.trajectory.az)
    assert np.array_equal(scan_block.trajectory.el, ref.trajectory.el)


def _passes(blocks):
    """Keep the planet-cal pass blocks of a result, without the acquisition slew."""
    return [b for b in blocks if b.block_type == BlockType.CALIBRATION]


def _assert_acquisition(blocks, anchor, ctx, az, el):
    """Assert the slew from the anchor and the gap-free tiling after it.

    The first block is the SLEW from ``(az, el)`` at ``anchor``, as long as
    ``plan_transition`` makes the move to the first pass's start pose; the
    first pass starts at its arrival and every later block at the previous
    block's stop.
    """
    slew, passes = blocks[0], _passes(blocks)
    assert slew.block_type == BlockType.SLEW
    assert slew.patch_name == "slew_to_jupiter"
    assert slew.t_start.unix == anchor.unix
    assert all(b.block_type == BlockType.CALIBRATION for b in blocks[1:])

    refs = [_reference_pass(b, ctx.site) for b in passes]
    expected = plan_transition(
        az,
        el,
        float(refs[0].trajectory.az[0]),
        passes[0].elevation,
        anchor,
        ctx.site,
        goal_az_span=(
            min(float(np.min(r.trajectory.az)) for r in refs),
            max(float(np.max(r.trajectory.az)) for r in refs),
        ),
        settle_time=ctx.overhead_model.settle_time,
        hold=float(refs[0].duration),
    )
    assert expected.safe
    # The references are the planned passes, so this is the slew the scheduler
    # planned; only the block's own times round its duration.
    assert slew.duration == pytest.approx(expected.duration, abs=1e-6)
    assert slew.az_end == expected.az_to

    # Tiling holds to 1e-6 s: each t_stop is constructed as the next t_start, so
    # unix-second float round-off is the only gap (same tolerance elsewhere in this file).
    for a, b in zip(blocks, blocks[1:]):
        assert abs(a.t_stop.unix - b.t_start.unix) < 1e-6


@pytest.fixture(scope="module")
def rising_run():
    """Plan the 3-pass rising sequence at the anchor once for the module."""
    anchor = Time(_ANCHOR, scale="utc")
    ctx = _ctx(_scan_policy(planet_cal_passes=3))
    state = SchedulerState(
        current_time=anchor,
        current_az=180.0,
        current_el=50.0,
        cal_state=_only_planet_cal_due(anchor),
        scan_counter=0,
    )
    return anchor, ctx, CalibrationPhase().run(state, ctx)


class TestPlanetCalScanEmit:
    """The multi-pass source-CES emit path and its recorded parameters."""

    def test_emits_one_block_per_pass_anchored_and_contiguous(self, rising_run):
        anchor, ctx, result = rising_run
        blocks = result.blocks

        # The acquisition slew, then one CALIBRATION block per requested pass.
        assert len(blocks) == 4
        passes = _passes(blocks)
        assert len(passes) == 3
        assert all(b.scan_type == "planet_cal" for b in passes)

        # The slew starts exactly at the pre-cal clock and lasts as long as
        # plan_transition says; the first pass starts at its arrival and the
        # blocks tile with no gaps (each t_stop is the next t_start).
        _assert_acquisition(blocks, anchor, ctx, 180.0, 50.0)

        # State advances to the last block's stop and marks the cadence at
        # the pre-slew clock.
        assert abs(result.state.current_time.unix - blocks[-1].t_stop.unix) < 1e-6
        assert result.state.cal_state.last_planet_cal is not None
        assert result.state.cal_state.last_planet_cal.unix == anchor.unix

    def test_each_pass_has_its_own_identity(self):
        """The sequence is one scan whose passes are numbered as subscans.

        ``scan_index`` and ``subscan_index`` are the canonical
        observation / sub-observation columns of the emitted table, so
        three passes all writing ``(0, 0)`` would leave an external
        consumer unable to tell them apart.
        """
        anchor = Time(_ANCHOR, scale="utc")
        ctx = _ctx(_scan_policy(planet_cal_passes=3))
        state = SchedulerState(
            current_time=anchor,
            current_az=180.0,
            current_el=50.0,
            cal_state=_only_planet_cal_due(anchor),
            scan_counter=4,
        )

        blocks = CalibrationPhase().run(state, ctx).blocks

        assert blocks[0].block_type == BlockType.SLEW
        assert blocks[0].scan_index == 4
        assert [(b.scan_index, b.subscan_index) for b in _passes(blocks)] == [
            (4, 0),
            (4, 1),
            (4, 2),
        ]

    def test_the_end_pose_is_where_the_drag_stopped(self, rising_run):
        """The pose carried forward is the trajectory's last azimuth.

        The block's ``az_start`` / ``az_end`` are the pass's azimuth
        envelope; the drag stops on whichever leg endpoint its last
        turnaround left it on, generally neither bound. Carrying the
        envelope maximum forward would price and Sun-check the next move
        from a pose the telescope is not at.
        """
        _, _, result = rising_run
        last = result.blocks[-1]

        assert last.az_final is not None
        assert last.az_start <= last.az_final < last.az_end - 1.0
        assert last.end_pose_az == last.az_final
        assert result.state.current_az == pytest.approx(last.az_final)

    def test_scan_params_recorded_and_valid(self, rising_run):
        blocks = _passes(rising_run[2].blocks)

        for idx, block in enumerate(blocks):
            meta = block.metadata
            assert set(meta) == {"cal_type", "target", "t0_scan", "scan_params", "search_start"}
            sp = meta["scan_params"]
            # Full replay-grade parameter set, and it validates.
            assert set(sp) == _SCAN_PARAM_KEYS
            validate_scan_params(sp, "source_ces")
            assert sp["body"] == "jupiter"
            assert sp["footprint"] == "c"
            assert sp["mode"] == "rising"
            assert sp["n_passes"] == 3
            assert sp["pass_index"] == idx
            # The recorded window is the pass extent [t0, t1]; its start is
            # the scan start.
            assert len(sp["window"]) == 2
            assert sp["window"][0] == meta["t0_scan"]
            # el_bore matches the block elevation and is the pass value.
            assert sp["el_bore"] == block.elevation
            # The true scan start is at or after the block start (acquisition
            # and inter-pass repointing fold into the block).
            assert Time(meta["t0_scan"]).unix >= block.t_start.unix - 1e-6

        # One search planned the sequence, from an anchor before its first pass;
        # every pass records it, beside the dict rather than in it.
        (record,) = {tuple(b.metadata["search_start"]) for b in blocks}
        jd1, jd2 = record
        start = Time(jd1, jd2, format="jd", scale="utc")
        assert start.unix <= Time(blocks[0].metadata["t0_scan"], scale="utc").unix

        # The three passes tile module-c in eta on a [-1, 0, +1] x (extent/3) grid,
        # where extent is the footprint eta span (1.2974 deg), so the step is ~0.4325 deg.
        # The span follows the per-module FOV radius and the pass count follows
        # planet_cal_passes, both awaiting instrument-team values: these pins
        # move with those decisions.
        eta_offsets = [b.metadata["scan_params"]["eta_offset_deg"] for b in blocks]
        # abs=1e-3 deg: milli-degree tolerance on the geometric eta offsets.
        assert eta_offsets == pytest.approx([-0.4325, 0.0, 0.4325], abs=1e-3)

    def test_a_centred_spelling_is_recorded_as_c(self, rising_run):
        """A policy naming the centre module ``IM0`` records what a ``"c"`` policy does.

        The recorded ``footprint`` is the module's canonical name, so a
        consumer comparing it as a string sees one spelling per module.
        """
        anchor, _, c_run = rising_run
        ctx = _ctx(_scan_policy(planet_cal_passes=3, planet_cal_footprint="IM0"))
        state = SchedulerState(
            current_time=anchor,
            current_az=180.0,
            current_el=50.0,
            cal_state=_only_planet_cal_due(anchor),
            scan_counter=0,
        )
        im0_run = CalibrationPhase().run(state, ctx)
        got = [b.metadata["scan_params"] for b in _passes(im0_run.blocks)]
        want = [b.metadata["scan_params"] for b in _passes(c_run.blocks)]
        assert len(got) == 3
        assert got == want
        assert {sp["footprint"] for sp in got} == {"c"}

    def test_el_bore_steps_monotonically_with_mode(self, rising_run):
        blocks = _passes(rising_run[2].blocks)

        # Jupiter is rising, so passes are ordered by increasing el_bore.
        assert all(b.metadata["scan_params"]["mode"] == "rising" for b in blocks)
        el_bores = [b.elevation for b in blocks]
        assert el_bores == sorted(el_bores)
        assert all(a < b for a, b in zip(el_bores, el_bores[1:]))
        # Each pass steps el_bore by the full footprint eta extent (1.2974 deg),
        # so consecutive passes tile the source in elevation with no overlap or gap.
        # The extent follows the per-module FOV radius, an instrument-team
        # value still pending; the pin moves with it.
        deltas = [b - a for a, b in zip(el_bores, el_bores[1:])]
        # abs=1e-3 deg: milli-degree tolerance on the geometric el_bore steps.
        assert deltas == pytest.approx([1.2974, 1.2974], abs=1e-3)

    def test_setting_planet_steps_el_bore_down(self):
        """A setting anchor produces a descending, contiguous setting sequence.

        Jupiter sets over roughly 00:00-04:00 UTC on the same UTC date (the
        tail of the previous night), so an anchor at 01:30 sits inside a
        setting arc: the passes step the boresight elevation strictly
        downward and every block observes the setting side.
        """
        anchor = Time("2026-03-15T01:30:00", scale="utc")
        ctx = _ctx(
            _scan_policy(planet_cal_passes=3),
            start="2026-03-15T01:30:00",
            end="2026-03-15T03:30:00",
        )
        state = SchedulerState(
            current_time=anchor,
            current_az=180.0,
            current_el=50.0,
            cal_state=_only_planet_cal_due(anchor),
            scan_counter=0,
        )

        blocks = CalibrationPhase().run(state, ctx).blocks
        passes = _passes(blocks)

        assert len(passes) == 3
        # Every pass runs on the setting arc, and the blocks say so.
        assert all(b.metadata["scan_params"]["mode"] == "setting" for b in passes)
        assert all(b.rising is False for b in passes)
        # A setting source crosses higher elevations first: strictly
        # decreasing el_bore across the sequence.
        el_bores = [b.elevation for b in passes]
        assert all(a > b for a, b in zip(el_bores, el_bores[1:]))
        # The slew from the anchor, then contiguous tiling, same as the
        # rising path.
        _assert_acquisition(blocks, anchor, ctx, 180.0, 50.0)

    def test_total_time_conserved_and_budget_counts_planet_cal(self, rising_run):
        anchor, ctx, result = rising_run
        blocks = result.blocks
        timeline = ObservingTimeline(
            blocks=blocks,
            site=ctx.site,
            start_time=anchor,
            end_time=blocks[-1].t_stop,
            overhead_model=ctx.overhead_model,
            calibration_policy=ctx.calibration_policy,
        )

        # Blocks, the acquisition slew included, tile [anchor, last t_stop]
        # with no holes.
        span = (blocks[-1].t_stop - anchor).sec
        block_total = sum(b.duration for b in blocks)
        # 1e-3 s: durations sum to the span to within milli-second round-off (seconds).
        assert abs(block_total - span) < 1e-3
        assert timeline.validate() == []

        breakdown = compute_budget(timeline)["calibration_breakdown"]
        assert breakdown["planet_cal"]["count"] == 3


class TestPlanetCalScanIntegration:
    """The flag drives through the public ``generate_timeline`` entry point."""

    def test_generate_timeline_flag_on(self):
        site = get_fyst_site()
        patches = [
            ObservingPatch(
                name="Deep56",
                ra_center=24.0,
                dec_center=-32.0,
                width=40.0,
                height=10.0,
                scan_type="constant_el",
                velocity=1.0,
                elevation=50.0,
            )
        ]
        policy = CalibrationPolicy(
            planet_cal_cadence=43200.0,
            planet_cal_scan=True,
            planet_cal_passes=3,
            planet_targets=("jupiter",),
            planet_min_elevation=15.0,
        )
        timeline = generate_timeline(
            patches=patches,
            site=site,
            start_time=_ANCHOR,
            end_time=_END,
            overhead_model=OverheadModel(),
            calibration_policy=policy,
        )

        planet_blocks = [b for b in timeline.blocks if b.scan_type == "planet_cal"]
        # The opening burst fires one planet cal as a 3-pass sequence.
        assert len(planet_blocks) == 3
        for b in planet_blocks:
            assert "scan_params" in b.metadata
            validate_scan_params(b.metadata["scan_params"], "source_ces")

        # Standard conservation + validation invariants still hold.
        block_total = sum(b.duration for b in timeline.blocks)
        assert abs(block_total - timeline.total_time) < 1e-3
        assert timeline.validate() == []
        assert compute_budget(timeline)["calibration_breakdown"]["planet_cal"]["count"] == 3


class TestPlanetCalScanFlagOff:
    """With the flag off the planet cal stays a single parked block."""

    def test_default_policy_emits_single_parked_planet_cal(self):
        anchor = Time(_ANCHOR, scale="utc")
        # planet_cal_scan defaults to False.
        policy = _scan_policy(planet_cal_scan=False)
        ctx = _ctx(policy)
        state = SchedulerState(
            current_time=anchor,
            current_az=175.0,
            current_el=48.0,
            cal_state=_only_planet_cal_due(anchor),
            scan_counter=0,
        )

        blocks = CalibrationPhase().run(state, ctx).blocks

        assert len(blocks) == 1
        block = blocks[0]
        assert block.scan_type == "planet_cal"
        # Parked at the current pose with the two-key metadata.
        assert block.az_start == block.az_end == 175.0
        assert block.elevation == 48.0
        assert set(block.metadata) == {"cal_type", "target"}
        assert block.metadata["cal_type"] == "planet_cal"
        assert block.metadata["target"] == "jupiter"
        assert abs(block.duration - OverheadModel().planet_cal_duration) < 1e-6


class TestPlanetCalScanDeferOnFailure:
    """An infeasible sequence is skipped and left due, other cals still run."""

    def test_planning_failure_defers_without_marking(self, monkeypatch):
        def _raise(**kwargs):
            raise TargetNotObservableError(
                target="jupiter",
                time_info=_ANCHOR,
                bounds_error=None,
                message="forced infeasible for test",
            )

        monkeypatch.setattr(
            "fyst_trajectories.overhead.scheduler.phases.plan_source_ces_passes",
            _raise,
        )

        anchor = Time(_ANCHOR, scale="utc")
        policy = _scan_policy(skydip_cadence=1.0e9)
        ctx = _ctx(policy)
        # skydip AND planet_cal both due; retune/pointing/focus recent.
        state = SchedulerState(
            current_time=anchor,
            current_az=180.0,
            current_el=50.0,
            cal_state=CalibrationState(
                last_retune=anchor,
                last_pointing_cal=anchor,
                last_focus=anchor,
                last_skydip=None,
                last_planet_cal=None,
            ),
            scan_counter=0,
        )

        result = CalibrationPhase().run(state, ctx)

        # The in-place skydip still emits; no planet_cal block appears.
        scan_types = [b.scan_type for b in result.blocks]
        assert "skydip" in scan_types
        assert "planet_cal" not in scan_types
        # planet_cal was not marked, so it stays due for a later iteration.
        assert result.state.cal_state.last_planet_cal is None
        assert result.state.cal_state.last_skydip is not None


def _refuse_every_path(current_az, current_el, goal_az, goal_el, time):
    """Refuse every slew, as a path-level model."""
    return False


class TestPlanetCalScanAcquisition:
    """The slew to the first pass is planned, and a Sun-blocked planet is passed over."""

    def test_no_clear_target_defers(self):
        """Mars at dawn sits inside the Sun zone, so the calibration stays due."""
        anchor, _, result = _run_at("2026-03-01T11:08:48", -78.69, 20.0, ("mars",))

        assert result.blocks == []
        assert result.state.cal_state.last_planet_cal is None
        assert result.state.current_time.unix == anchor.unix

    def test_a_planet_inside_the_zone_falls_back_to_the_next_target(self):
        """Venus is up but inside the Sun zone; Jupiter is up and clear."""
        _, _, result = _run_at("2026-03-15T21:00:00", 180.0, 50.0, ("venus", "jupiter"))

        assert result.blocks
        assert result.blocks[0].block_type == BlockType.SLEW
        assert result.blocks[0].patch_name == "slew_to_jupiter"
        passes = result.blocks[1:]
        assert passes
        assert all(b.block_type == BlockType.CALIBRATION for b in passes)
        assert all(b.metadata["target"] == "jupiter" for b in passes)
        assert all(b.metadata["scan_params"]["body"] == "jupiter" for b in passes)

    def test_the_acquisition_takes_the_near_wrap(self):
        """From az 21.83 the near image of the rising Jupiter pass is below 0.

        The planner records the pass at about az 311.7; the slew there runs
        the long way round, while its image one turn down is reached in well
        under half the time.
        """
        _, _, result = _run_at("2026-03-01T04:04:20", 21.83, 61.35, ("jupiter",))

        slew = result.blocks[0]
        assert slew.block_type == BlockType.SLEW
        assert slew.patch_name == "slew_to_jupiter"
        # abs=0.5 deg: the re-anchored first sample sits near -48.6.
        assert slew.az_end == pytest.approx(-48.6, abs=0.5)
        passes = result.blocks[1:]
        assert passes
        for block in passes:
            assert block.az_start < 0.0
            assert block.az_end < 0.0
            assert block.az_final is not None and block.az_final < 0.0
        assert result.state.current_az < 0.0

    def test_a_refused_transition_defers(self):
        anchor, _, result = _run_at(
            _ANCHOR, 180.0, 50.0, ("jupiter",), slew_safe=_refuse_every_path
        )

        assert result.blocks == []
        assert result.state.cal_state.last_planet_cal is None
        assert result.state.current_time.unix == anchor.unix

    def test_every_emitted_pass_sweeps_clear(self, rising_run):
        """Each pass, rebuilt and placed in its recorded wrap, is clear of the Sun.

        Three states: rising Jupiter at night, Mars at dawn and Venus by day.
        Only the first has a Sun-clear planet, so only its passes may be
        emitted.
        """
        site = get_fyst_site()
        sun = make_sun_safe("scalar", site=site)
        results = [
            (_ANCHOR, rising_run[2]),
            ("2026-03-01T11:08:48", _run_at("2026-03-01T11:08:48", -78.69, 20.0, ("mars",))[2]),
            ("2026-03-15T13:18:00", _run_at("2026-03-15T13:18:00", 180.0, 50.0, ("venus",))[2]),
        ]
        swept = 0
        for iso, result in results:
            for block in result.blocks:
                if block.block_type != BlockType.CALIBRATION:
                    continue
                trajectory = _reference_pass(block, site).trajectory
                turns = (block.az_start - float(np.min(trajectory.az))) / 360.0
                # The reference is the planned pass, so only float rounding
                # separates the recorded envelope from a whole number of turns.
                assert turns == pytest.approx(round(turns), abs=1e-9)
                verdicts = sun.batch(
                    np.asarray(trajectory.az) + 360.0 * round(turns),
                    np.asarray(trajectory.el),
                    get_absolute_times(trajectory),
                )
                assert bool(np.all(verdicts)), (iso, block.metadata["target"])
                swept += 1
        assert swept >= 1


class TestPlanetCalScanTruncation:
    """End-of-night keeps only the passes that finish before the window closes."""

    def test_truncates_to_fitting_prefix(self):
        anchor = Time("2026-03-15T20:30:00", scale="utc")
        # Only the first pass (finishing ~20:37) fits before this end time.
        ctx = _ctx(
            _scan_policy(planet_cal_passes=3),
            start="2026-03-15T20:30:00",
            end="2026-03-15T20:40:00",
        )
        state = SchedulerState(
            current_time=anchor,
            current_az=180.0,
            current_el=50.0,
            cal_state=_only_planet_cal_due(anchor),
            scan_counter=0,
        )

        blocks = CalibrationPhase().run(state, ctx).blocks
        passes = _passes(blocks)

        assert blocks[0].block_type == BlockType.SLEW
        assert len(passes) == 1
        # The requested total is preserved so truncation is visible.
        assert passes[0].metadata["scan_params"]["n_passes"] == 3
        assert passes[0].metadata["scan_params"]["pass_index"] == 0
        assert passes[0].t_stop.unix <= ctx.end_time.unix

    def test_a_pass_quantised_short_of_its_window_is_still_dropped(self):
        # Leg quantisation moves the trajectory end and the recorded window
        # close apart in either direction. This window closes 3.75 s before
        # the third pass's ``t1_iso`` while the pass's own quantised
        # trajectory ends before the close, so a filter that looks only at
        # the trajectory end keeps a block whose ``t_stop`` lies past the
        # window, breaking the tiling invariant ``validate`` checks.
        anchor = Time("2026-03-15T20:30:00", scale="utc")
        ctx = _ctx(
            _scan_policy(planet_cal_passes=3),
            start="2026-03-15T20:30:00",
            end="2026-03-15T20:56:45",
        )
        state = SchedulerState(
            current_time=anchor,
            current_az=180.0,
            current_el=50.0,
            cal_state=_only_planet_cal_due(anchor),
            scan_counter=0,
        )

        result = CalibrationPhase().run(state, ctx)

        assert result.blocks
        for block in result.blocks:
            assert block.t_stop.unix <= ctx.end_time.unix
        assert result.state.current_time.unix <= ctx.end_time.unix


class TestPlanetCalScanECSVRoundTrip:
    """The recorded pass parameters survive a TOAST-ECSV write/read."""

    def test_scan_params_and_t0_scan_round_trip(self, rising_run, tmp_path):
        anchor, ctx, result = rising_run
        blocks = result.blocks
        timeline = ObservingTimeline(
            blocks=blocks,
            site=ctx.site,
            start_time=anchor,
            end_time=blocks[-1].t_stop,
            overhead_model=ctx.overhead_model,
            calibration_policy=ctx.calibration_policy,
        )

        path = tmp_path / "planet_cal_scan_rt.ecsv"
        write_timeline(timeline, path)
        loaded = read_timeline(path)

        cal_blocks = _passes(loaded.blocks)
        assert len(cal_blocks) == 3
        for original, restored in zip(_passes(blocks), cal_blocks):
            assert restored.metadata["t0_scan"] == original.metadata["t0_scan"]
            assert restored.metadata["scan_params"] == original.metadata["scan_params"]
            # The search start is two floats, which JSON keeps exactly.
            assert restored.metadata["search_start"] == original.metadata["search_start"]
            # Restored params still validate.
            validate_scan_params(restored.metadata["scan_params"], "source_ces")


class TestScheduleToTrajectoriesScienceOnly:
    """Default True returns science pairs only; False adds nothing without cal params."""

    def test_flag_on_timeline_default_returns_science_only(self, flag_on_recon, flag_on_timeline):
        """Default ``science_only=True`` returns science pairs and no calibration."""
        _site, sci_only, _with_cals = flag_on_recon

        assert sci_only, "expected the flag-on timeline to carry science blocks"
        assert all(b.block_type == BlockType.SCIENCE for b, _ in sci_only)
        # Every science block reconstructs; calibration passes are excluded.
        assert len(sci_only) == len(flag_on_timeline.science_blocks)
        # The planet-cal passes exist in the timeline but are not returned here.
        assert any(b.scan_type == "planet_cal" for b in flag_on_timeline.blocks)

    def test_default_policy_skips_calibrations_silently(self, caplog):
        """``science_only=False`` skips parked cals/retunes with no logs; science unchanged."""
        site = get_fyst_site()
        patch = ObservingPatch(
            name="pong_field",
            ra_center=180.0,
            dec_center=-30.0,
            width=4.0,
            height=4.0,
            scan_type="pong",
            velocity=0.5,
        )
        # Default policy: parked planet cal + retunes, none carrying scan_params.
        timeline = generate_timeline(
            patches=[patch],
            site=site,
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T06:00:00",
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )
        # There really are calibration blocks with no scan_params, so the
        # science_only=False path exercises the silent-skip branch.
        assert any(
            b.block_type == BlockType.CALIBRATION and "scan_params" not in b.metadata
            for b in timeline.blocks
        )

        sci_true = schedule_to_trajectories(timeline, science_only=True)
        with caplog.at_level(logging.WARNING, logger="fyst_trajectories.overhead.simulation"):
            all_false = schedule_to_trajectories(timeline, science_only=False)

        # The parked/retune/idle blocks were skipped silently: no reconstruction
        # failure was logged for them.
        sim_records = [
            r for r in caplog.records if r.name == "fyst_trajectories.overhead.simulation"
        ]
        assert sim_records == []

        # With no reconstructable calibration blocks, False adds nothing: the
        # science pairs are identical to the science_only=True result.
        assert len(all_false) == len(sci_true)
        assert all(b.block_type == BlockType.SCIENCE for b, _ in all_false)
        for (b_false, sb_false), (b_true, sb_true) in zip(all_false, sci_true):
            assert b_false is b_true
            assert np.array_equal(sb_false.trajectory.az, sb_true.trajectory.az)
            assert np.array_equal(sb_false.trajectory.el, sb_true.trajectory.el)


class TestPlanetCalScanReconstruction:
    """``schedule_to_trajectories(science_only=False)`` rebuilds source-CES passes."""

    def test_reconstructs_science_plus_one_pair_per_pass(self, flag_on_recon, flag_on_timeline):
        site, sci_only, with_cals = flag_on_recon

        cal_pairs = [(b, sb) for b, sb in with_cals if b.block_type == BlockType.CALIBRATION]
        sci_pairs = [(b, sb) for b, sb in with_cals if b.block_type == BlockType.SCIENCE]
        planet_blocks = [b for b in flag_on_timeline.blocks if b.scan_type == "planet_cal"]

        # One reconstructed pair per planet-cal pass, plus the science pairs.
        assert len(planet_blocks) == 3
        assert len(cal_pairs) == len(planet_blocks)
        assert all(b.scan_type == "planet_cal" for b, _ in cal_pairs)

        # The science subset is exactly the science_only=True result.
        assert len(sci_pairs) == len(sci_only)
        for (b_f, sb_f), (b_t, sb_t) in zip(sci_pairs, sci_only):
            assert b_f is b_t
            assert np.array_equal(sb_f.trajectory.az, sb_t.trajectory.az)

        # Each rebuilt pass solves its recorded window and matches a reference
        # plan built from the same shifted footprint in the planner's search window.
        for block, scan_block in cal_pairs:
            _assert_reconstruction_faithful(block, scan_block, site)

    def test_full_chain_ecsv_roundtrip_feeds_reconstruction(self, flag_on_timeline, tmp_path):
        path = tmp_path / "flag_on_rt.ecsv"
        write_timeline(flag_on_timeline, path)
        loaded = read_timeline(path)

        pairs = schedule_to_trajectories(loaded, science_only=False)
        cal_pairs = [(b, sb) for b, sb in pairs if b.block_type == BlockType.CALIBRATION]
        assert len(cal_pairs) == 3
        # The scan_params survive the ECSV JSON round-trip and still rebuild the
        # same geometry, matched against a reference planned from the loaded
        # parameters.
        for block, scan_block in cal_pairs:
            _assert_reconstruction_faithful(block, scan_block, loaded.site)

    def test_truncated_pass_reconstructs(self):
        """A truncated emission (``n_passes`` > emitted) still rebuilds per block."""
        anchor = Time("2026-03-15T20:30:00", scale="utc")
        ctx = _ctx(
            _scan_policy(planet_cal_passes=3),
            start="2026-03-15T20:30:00",
            end="2026-03-15T20:40:00",
        )
        state = SchedulerState(
            current_time=anchor,
            current_az=180.0,
            current_el=50.0,
            cal_state=_only_planet_cal_due(anchor),
            scan_counter=0,
        )
        blocks = CalibrationPhase().run(state, ctx).blocks
        passes = _passes(blocks)
        assert len(passes) == 1
        assert passes[0].metadata["scan_params"]["n_passes"] == 3

        timeline = ObservingTimeline(
            blocks=blocks,
            site=ctx.site,
            start_time=anchor,
            end_time=blocks[-1].t_stop,
            overhead_model=ctx.overhead_model,
            calibration_policy=ctx.calibration_policy,
        )
        pairs = schedule_to_trajectories(timeline, science_only=False)
        assert len(pairs) == 1
        block, scan_block = pairs[0]
        assert block.block_type == BlockType.CALIBRATION
        _assert_reconstruction_faithful(block, scan_block, ctx.site)

    def test_setting_pass_reconstructs(self):
        """A setting-direction pass rebuilds onto its recorded window and geometry."""
        anchor = Time("2026-03-15T01:30:00", scale="utc")
        ctx = _ctx(
            _scan_policy(planet_cal_passes=1),
            start="2026-03-15T01:30:00",
            end="2026-03-15T03:30:00",
        )
        state = SchedulerState(
            current_time=anchor,
            current_az=180.0,
            current_el=50.0,
            cal_state=_only_planet_cal_due(anchor),
            scan_counter=0,
        )
        blocks = CalibrationPhase().run(state, ctx).blocks
        passes = _passes(blocks)
        assert len(passes) == 1
        assert passes[0].metadata["scan_params"]["mode"] == "setting"

        timeline = ObservingTimeline(
            blocks=blocks,
            site=ctx.site,
            start_time=anchor,
            end_time=blocks[-1].t_stop,
            overhead_model=ctx.overhead_model,
            calibration_policy=ctx.calibration_policy,
        )
        pairs = schedule_to_trajectories(timeline, science_only=False)
        assert len(pairs) == 1
        block, scan_block = pairs[0]
        assert block.block_type == BlockType.CALIBRATION
        assert scan_block.computed_params["mode"] == "setting"
        _assert_reconstruction_faithful(block, scan_block, ctx.site)

    def test_an_off_centre_pass_reconstructs_from_its_search_start(self):
        """On ``i6`` Jupiter reaches the boresight elevation only after the pass ends.

        The kernel places the module on the source from that crossing, which
        the planner's 24 h search holds and the recorded pass widened by 300 s
        does not: the block without ``search_start`` is refused, while the
        block as written rebuilds as planned.
        """
        anchor = Time(_ANCHOR, scale="utc")
        ctx = _ctx(_scan_policy(planet_cal_passes=1, planet_cal_footprint="i6"))
        state = SchedulerState(
            current_time=anchor,
            current_az=180.0,
            current_el=50.0,
            cal_state=_only_planet_cal_due(anchor),
            scan_counter=0,
        )
        blocks = CalibrationPhase().run(state, ctx).blocks
        timeline = ObservingTimeline(
            blocks=blocks,
            site=ctx.site,
            start_time=anchor,
            end_time=blocks[-1].t_stop,
            overhead_model=ctx.overhead_model,
            calibration_policy=ctx.calibration_policy,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pairs = schedule_to_trajectories(timeline, science_only=False)
            keyless = dataclasses.replace(
                timeline,
                blocks=[
                    dataclasses.replace(
                        b,
                        metadata={k: v for k, v in b.metadata.items() if k != "search_start"},
                    )
                    for b in blocks
                ],
            )
            refused = schedule_to_trajectories(keyless, science_only=False)
        ((block, scan_block),) = pairs
        assert block.metadata["scan_params"]["footprint"] == "i6"
        _assert_reconstruction_faithful(block, scan_block, ctx.site)
        assert refused == []

    def test_a_calibration_planned_from_a_tt_clock_is_the_utc_one(self, monkeypatch, tmp_path):
        """A state and a window given in TT plan the calibration the UTC ones do, in UTC.

        The scheduler holds its clock in UTC, so the passes are planned from a
        UTC instant: every block is held and recorded in UTC at the UTC run's
        instants, ``search_start`` is the instant the planner searched from,
        and every pass rebuilds bit for bit, from memory and from ECSV.
        """
        planned = []
        real = phases.plan_source_ces_passes

        def recording(*args, **kwargs):
            passes = real(*args, **kwargs)
            planned.extend((kwargs["start_time"], p) for p in passes)
            return passes

        monkeypatch.setattr(phases, "plan_source_ces_passes", recording)
        _, _, utc_result = _run_at(_ANCHOR, 180.0, 50.0, ("jupiter",))
        planned.clear()
        anchor = Time(_ANCHOR, scale="utc").tt
        ctx = SchedulerContext.build(
            patches=[],
            site=get_fyst_site(),
            start_time=anchor,
            end_time=anchor + TimeDelta(4 * 3600.0, format="sec"),
            calibration_policy=_scan_policy(),
        )
        state = SchedulerState(
            current_time=anchor,
            current_az=180.0,
            current_el=50.0,
            cal_state=_only_planet_cal_due(anchor),
            scan_counter=0,
        )
        blocks = CalibrationPhase().run(state, ctx).blocks
        assert len(blocks) == len(utc_result.blocks) == 4
        for block, utc_block in zip(blocks, utc_result.blocks):
            assert (block.t_start.scale, block.t_stop.scale) == ("utc", "utc")
            assert abs((block.t_start - utc_block.t_start).to_value("s")) < 1e-6
            assert abs((block.t_stop - utc_block.t_stop).to_value("s")) < 1e-6
            assert block.metadata.get("t0_scan") == utc_block.metadata.get("t0_scan")

        timeline = ObservingTimeline(
            blocks=blocks,
            site=ctx.site,
            start_time=ctx.start_time,
            end_time=blocks[-1].t_stop,
            overhead_model=ctx.overhead_model,
            calibration_policy=ctx.calibration_policy,
        )
        path = tmp_path / "tt_clock.ecsv"
        write_timeline(timeline, path)
        for source in (timeline, read_timeline(path)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pairs = schedule_to_trajectories(source, science_only=False)
            assert len(pairs) == 3
            for block, rebuilt in pairs:
                el_bore = block.metadata["scan_params"]["el_bore"]
                # The last solve wins: a late slew re-plans at its arrival.
                solved = [(a, q) for a, q in planned if q.computed_params["el_bore"] == el_bore]
                start, p = solved[-1]
                assert (start.scale, start.location) == ("utc", None)
                assert block.metadata["search_start"] == [start.jd1, start.jd2]
                for name in ("times", "az", "el", "az_vel", "el_vel", "scan_flag"):
                    assert np.array_equal(
                        getattr(rebuilt.trajectory, name), getattr(p.trajectory, name)
                    )
                got, want = rebuilt.trajectory.start_time, p.trajectory.start_time
                assert (got.scale, got.jd1, got.jd2) == (want.scale, want.jd1, want.jd2)
                assert rebuilt.computed_params == p.computed_params

    def test_hitmap_gains_hits_from_reconstructed_cals(self, flag_on_recon):
        """Feeding science+cal pairs into accumulate_hitmaps adds exactly the cal hits."""
        pytest.importorskip("healpy")
        site, sci_only, with_cals = flag_on_recon

        def expected_hits(pairs):
            # accumulate_hitmaps bins every 10th science-flagged sample once.
            return sum(int(np.count_nonzero(sb.trajectory.science_mask[::10])) for _, sb in pairs)

        hm_science = accumulate_hitmaps(sci_only, site, nside=16)
        hm_all = accumulate_hitmaps(with_cals, site, nside=16)
        assert hm_science.sum() == expected_hits(sci_only)
        assert hm_all.sum() == expected_hits(with_cals)
        assert hm_all.sum() > hm_science.sum()
