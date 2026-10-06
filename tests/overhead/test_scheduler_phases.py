"""Phase-level unit tests for the offline scheduler.

Each scheduler phase (``CalibrationPhase``, ``PatchSelectionPhase``,
``SlewPhase``, ``ScienceScanPhase``) is exercised through its public
API directly, asserting the state and block invariants that phase owns
without driving the whole scheduler loop. A composition test checks that
the loop reproduces ``generate_timeline``.
"""

import warnings

import pytest
from _scheduler_helpers import _initial_state, _make_ctx
from astropy.time import Time, TimeDelta

from fyst_trajectories import get_fyst_site
from fyst_trajectories.exceptions import PointingError
from fyst_trajectories.overhead import (
    CalibrationPolicy,
    ObservingPatch,
    OverheadModel,
)
from fyst_trajectories.overhead.scheduler import (
    CalibrationPhase,
    PatchSelectionPhase,
    PhaseResult,
    Scheduler,
    SchedulerContext,
    ScienceScanPhase,
    SlewPhase,
    phases,
)
from fyst_trajectories.overhead.scheduler.helpers import _compute_az_range
from fyst_trajectories.overhead.utils import _normalize_az
from fyst_trajectories.patterns import PongScanConfig, compute_pong_period


def _deep56_ce_patch(name="deep56"):
    """Construct the Deep56 constant-elevation patch used across phase tests."""
    return ObservingPatch(
        name=name,
        ra_center=24.0,
        dec_center=-32.0,
        width=40.0,
        height=10.0,
        scan_type="constant_el",
        velocity=1.0,
        elevation=50.0,
    )


def _pong_patch(name="Wide01", width=4.0, height=4.0, **kwargs):
    """Build a pong patch that is up from 02:00 UTC on the fixture night."""
    return ObservingPatch(
        name=name,
        ra_center=180.0,
        dec_center=-30.0,
        width=width,
        height=height,
        scan_type="pong",
        velocity=0.5,
        **kwargs,
    )


def _pong_period(patch):
    """One period of ``patch``'s pattern at the rebuild's default spacing."""
    config = PongScanConfig(
        timestep=0.1,
        width=patch.width,
        height=patch.height,
        spacing=0.1,
        velocity=patch.velocity,
        num_terms=4,
        angle=0.0,
    )
    return compute_pong_period(config)[0]


def _kinds(blocks):
    """Name each block: its calibration type, or its block type."""
    return [
        str(b.scan_type) if str(b.block_type) == "calibration" else str(b.block_type)
        for b in blocks
    ]


def _ce_ready_ctx(patch, **ctx_kwargs):
    """Build a context anchored one tick before the patch's rising pass opens.

    The corridor gate only selects a CE patch while its crossing pass is
    imminent, so phase-mechanics tests anchor the schedule window just
    before the pass opening (located via the scheduler's own corridor
    solver) instead of running from the fixture night's start, where the
    patch is hours from plannable.
    """
    from fyst_trajectories.coordinates import Coordinates
    from fyst_trajectories.overhead.scheduler.helpers import _ce_crossing_corridor

    coords = Coordinates(get_fyst_site())
    t_open, _ = _ce_crossing_corridor(
        patch, patch.elevation, True, Time("2026-06-15T02:00:00", scale="utc"), coords, {}
    )
    start = t_open - TimeDelta(300.0, format="sec")
    return _make_ctx(patches=[patch], start_time=start.isot, **ctx_kwargs)


class TestCalibrationPhase:
    """Calibration phase emits a block when a cadence has elapsed."""

    def test_startup_emits_multiple_cal_blocks(self):
        """With CalibrationState.last_* all None, all cadences fire at t=0."""
        ctx = _make_ctx(patches=[])
        state = _initial_state(ctx)

        result = CalibrationPhase().run(state, ctx)

        assert isinstance(result, PhaseResult)
        # At startup every due cadence fires, in the canonical order; no
        # planet is up yet at 02:00 UTC, so planet_cal is not among them.
        assert [str(b.scan_type) for b in result.blocks] == [
            "retune",
            "pointing_cal",
            "focus",
            "skydip",
        ]
        # The state advances past every block.
        assert result.state.current_time.unix > state.current_time.unix
        # The cal state has updated, at least retune is no longer None.
        assert result.state.cal_state.last_retune is not None

    def test_idle_ticks_do_not_retune(self):
        """A cadence-0 retune is scan-coupled: an all-idle night retunes once.

        Without the scan-coupled rule the per-tick CalibrationPhase would
        consume the always-due cadence-0 retune on every idle tick, booking
        a 300 s retune on every 300 s tick while the telescope sits parked.
        Only the startup burst may fire one outside a scan boundary.
        """
        from fyst_trajectories.overhead import BlockType, generate_timeline

        unreachable = ObservingPatch(
            name="never_up",
            ra_center=150.0,
            dec_center=80.0,  # never rises from FYST
            width=4.0,
            height=4.0,
            scan_type="pong",
            velocity=0.5,
        )
        timeline = generate_timeline(
            patches=[unreachable],
            site=get_fyst_site(),
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T04:00:00",
        )
        retunes = [
            b
            for b in timeline.blocks
            if b.block_type == BlockType.CALIBRATION and b.scan_type == "retune"
        ]
        assert len(retunes) == 1  # the startup burst only

    def test_noop_when_no_cals_due(self):
        """Immediately after firing cals, re-running emits nothing."""
        # Use a finite retune cadence so retune doesn't fire on every
        # invocation (the default ``retune_cadence=0.0`` means "always").
        policy = CalibrationPolicy(
            retune_cadence=3600.0,
            pointing_cadence=3600.0,
            focus_cadence=7200.0,
            skydip_cadence=10800.0,
            planet_cal_cadence=43200.0,
        )
        ctx = _make_ctx(patches=[], calibration_policy=policy)
        state = _initial_state(ctx)

        first = CalibrationPhase().run(state, ctx)
        second = CalibrationPhase().run(first.state, ctx)

        assert second.blocks == []
        assert second.state.current_time.unix == first.state.current_time.unix


class TestPatchSelectionPhase:
    """Patch selection chooses the best observable patch or emits idle."""

    def test_no_patches_emits_idle(self):
        """With zero patches, the phase emits an IDLE block and skips."""
        ctx = _make_ctx(patches=[])
        state = _initial_state(ctx)

        result = PatchSelectionPhase().run(state, ctx)

        assert len(result.blocks) == 1
        assert str(result.blocks[0].block_type) == "idle"
        assert result.selection is None
        assert result.skip_to_next_iter is True
        # Time advanced by time_step (or end-time distance, whichever smaller).
        assert result.state.current_time.unix > state.current_time.unix

    def test_patch_below_elevation_emits_idle(self):
        """A patch that never rises yields an idle block, not a selection."""
        unreachable = ObservingPatch(
            name="never_up",
            ra_center=150.0,
            dec_center=80.0,  # Never rises from FYST (lat ~ -23): max el ~ -13 deg.
            width=4.0,
            height=4.0,
            scan_type="pong",
            velocity=0.5,
        )
        ctx = _make_ctx(
            patches=[unreachable],
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T04:00:00",
        )
        state = _initial_state(ctx)

        result = PatchSelectionPhase().run(state, ctx)

        # The patch can never be observable, so the phase MUST emit idle and skip.
        assert result.selection is None
        assert len(result.blocks) == 1
        assert str(result.blocks[0].block_type) == "idle"
        assert result.skip_to_next_iter is True

    def test_observable_patch_selected(self):
        """A well-placed patch is selected with best_az/best_el populated."""
        ce_patch = _deep56_ce_patch()
        # Anchor just before the patch's crossing pass so the corridor
        # gate deems it selectable (no calibration pre-step needed).
        ctx = _ce_ready_ctx(ce_patch)
        state = _initial_state(ctx)

        result = PatchSelectionPhase().run(state, ctx)

        assert result.selection is not None
        assert result.selection.name == "deep56"
        assert result.best_az is not None
        assert result.best_el is not None
        assert result.skip_to_next_iter is False
        # No blocks emitted, the selection result is consumed by next phase.
        assert result.blocks == []


class TestSlewPhase:
    """Slew phase emits a block when the telescope needs to move."""

    def test_requires_selection(self):
        ctx = _make_ctx(patches=[])
        state = _initial_state(ctx)

        with pytest.raises(ValueError, match="PatchSelectionPhase"):
            SlewPhase().run(state, ctx)

    def test_small_slew_is_skipped(self):
        """When slew+settle <= 1s, no block is emitted."""
        ce_patch = _deep56_ce_patch()
        overhead = OverheadModel(settle_time=0.0)
        ctx = _ce_ready_ctx(ce_patch, overhead_model=overhead)
        state = _initial_state(ctx)
        selection = PatchSelectionPhase().run(state, ctx)
        # Simulate a state where the telescope is already at the science
        # pose: the slew targets the pinned patch elevation, not the
        # field centre's instantaneous elevation.
        assert selection.best_az is not None
        assert selection.best_el is not None
        at_patch = state.advanced(current_az=selection.best_az, current_el=ce_patch.elevation)

        result = SlewPhase().run(at_patch, ctx, selection=selection)

        # slew_time was < 1s, no block emitted; state unchanged.
        assert result.blocks == []
        assert result.state.current_time.unix == at_patch.current_time.unix

    def test_large_slew_emits_block(self):
        """A large az change yields a SLEW block advancing current_time."""
        ce_patch = _deep56_ce_patch()
        ctx = _ce_ready_ctx(ce_patch)
        state = _initial_state(ctx)
        selection = PatchSelectionPhase().run(state, ctx)

        result = SlewPhase().run(state, ctx, selection=selection)

        # Slew from state's (180, 50) to deep56's science pose (~115 az at
        # the pinned el=50): a large ~65 deg az move.
        assert len(result.blocks) == 1
        block = result.blocks[0]
        assert str(block.block_type) == "slew"
        assert block.az_start == state.current_az
        assert block.az_end == selection.best_az
        assert block.elevation == ce_patch.elevation
        # The slew establishes the pose downstream blocks are stamped with.
        assert result.state.current_az == selection.best_az
        assert result.state.current_el == ce_patch.elevation
        # Pin the dominant az move against the ~65 deg expectation stated above.
        assert abs(block.az_end - block.az_start) == pytest.approx(65.15, abs=0.01)
        # Time advanced by the slew duration (~29 s for this move).
        assert result.state.current_time.unix > state.current_time.unix
        assert block.duration == pytest.approx(28.72, abs=0.01)


class TestScienceScanPhase:
    """Science scan phase emits subscans with interleaved retunes."""

    def test_requires_selection(self):
        ctx = _make_ctx(patches=[])
        state = _initial_state(ctx)

        with pytest.raises(ValueError, match="PatchSelectionPhase"):
            ScienceScanPhase().run(state, ctx)

    def test_emits_one_or_more_science_blocks(self):
        ce_patch = _deep56_ce_patch()
        ctx = _ce_ready_ctx(ce_patch)
        state = _initial_state(ctx)
        selection = PatchSelectionPhase().run(state, ctx)
        slew = SlewPhase().run(state, ctx, selection=selection)

        result = ScienceScanPhase().run(slew.state, ctx, selection=slew)

        science_blocks = [b for b in result.blocks if str(b.block_type) == "science"]
        assert len(science_blocks) >= 1
        # Scan counter must have advanced exactly once, regardless of subscans.
        assert result.state.scan_counter == slew.state.scan_counter + 1

    def test_a_swept_subscan_records_the_pose_it_ends_at(self):
        """A CE sweep stops on a leg endpoint, not on its envelope bound.

        The recorded ``az_final`` is read from the subscan's own
        trajectory, so it must equal what a consumer rebuilding the block
        gets, and the state the next slew is priced from must be that
        pose rather than the envelope maximum.
        """
        from fyst_trajectories.overhead.simulation import _generate_trajectory_for_block

        ce_patch = _deep56_ce_patch()
        ctx = _ce_ready_ctx(ce_patch)
        state = _initial_state(ctx)
        selection = PatchSelectionPhase().run(state, ctx)
        slew = SlewPhase().run(state, ctx, selection=selection)

        result = ScienceScanPhase().run(slew.state, ctx, selection=slew)

        science_blocks = [b for b in result.blocks if str(b.block_type) == "science"]
        assert science_blocks
        for block in science_blocks:
            rebuilt = _generate_trajectory_for_block(block, ctx.site)
            true_end_az = float(rebuilt.trajectory.az[-1])
            assert block.az_final == pytest.approx(true_end_az)
            assert block.end_pose_az == pytest.approx(true_end_az)
            # The envelope bound in ``az_end`` is a different pose.
            assert abs(block.az_end - true_end_az) > 1.0
        assert result.state.current_az == pytest.approx(science_blocks[-1].end_pose_az)

    def test_sliver_window_emits_nothing_not_a_dangling_retune(self):
        """A window too small for retune + minimum subscan emits NO blocks.

        The boundary retune is booked only when a minimum-duration subscan
        still fits after it; a sliver visit must end empty rather than on
        a dangling retune (or a retune spilling past end_time).
        """
        ce_patch = _deep56_ce_patch()
        overhead = OverheadModel()
        start = Time("2026-06-15T02:00:00", scale="utc")
        window = overhead.min_scan_duration + overhead.retune_duration - 1.0
        ctx = _make_ctx(
            patches=[ce_patch],
            start_time=start.isot,
            end_time=(start + TimeDelta(window, format="sec")).isot,
            overhead_model=overhead,
        )
        state = _initial_state(ctx)  # fresh cal state: cadence-0 retune is due

        state, blocks, refusal = ScienceScanPhase._emit_subscans_with_retunes(
            state=state,
            ctx=ctx,
            best_patch=ce_patch,
            best_el=50.0,
            n_subscans=1,
            subscan_duration=window,
            rising=True,
            az_start_sci=100.0,
            az_end_sci=140.0,
            deadline=ctx.end_time,
            t0_scan=None,
        )
        assert blocks == []
        assert refusal is None

    def test_boundary_retune_plus_min_subscan_fit(self):
        """With one retune-width more room, the visit emits retune + science.

        The visit is anchored where Deep56's pass is plannable, since a
        subscan the planner refuses is not emitted.
        """
        ce_patch = _deep56_ce_patch()
        overhead = OverheadModel()
        ctx = _ce_ready_ctx(ce_patch, overhead_model=overhead)
        window = overhead.min_scan_duration + overhead.retune_duration + 1.0
        deadline = ctx.start_time + TimeDelta(window, format="sec")
        state = _initial_state(ctx)

        state, blocks, refusal = ScienceScanPhase._emit_subscans_with_retunes(
            state=state,
            ctx=ctx,
            best_patch=ce_patch,
            best_el=50.0,
            n_subscans=1,
            subscan_duration=window,
            rising=True,
            az_start_sci=100.0,
            az_end_sci=140.0,
            deadline=deadline,
            t0_scan=ctx.start_time.isot,
        )
        assert refusal is None
        assert _kinds(blocks) == ["retune", "science"]
        science = blocks[1]
        assert science.duration >= overhead.min_scan_duration
        assert science.t_stop.unix <= deadline.unix + 1e-6

    def test_long_scan_splits_into_subscans(self):
        """When scan_duration > max_scan_duration, emit multiple subscans."""
        ce_patch = _deep56_ce_patch()
        # Force small max_scan_duration so splitting is guaranteed.
        overhead = OverheadModel(max_scan_duration=1200.0)
        ctx = _ce_ready_ctx(ce_patch, overhead_model=overhead)
        state = _initial_state(ctx)
        selection = PatchSelectionPhase().run(state, ctx)
        slew = SlewPhase().run(state, ctx, selection=selection)

        result = ScienceScanPhase().run(slew.state, ctx, selection=slew)

        science_blocks = [b for b in result.blocks if str(b.block_type) == "science"]
        # The pass spans hours against a 1200 s subscan cap, so expect 2+.
        assert len(science_blocks) >= 2
        # Subscan indices should be sequential.
        sub_indices = [b.subscan_index for b in science_blocks]
        assert sub_indices == list(range(len(sub_indices)))


class TestWholePeriodPong:
    """A pong subscan holds whole pattern periods and is planned before it is booked.

    The rebuild runs ``n_cycles`` periods of the pattern, so a block sized
    from the scheduler's budget alone counted science its trajectory never
    ran, or stopped mid-pattern. A subscan is probed before its boundary
    retune is booked, so a refusal leaves neither block behind.
    """

    @staticmethod
    def _visit(ctx):
        """Select, slew and scan once from a fresh state; return (slew, science)."""
        state = _initial_state(ctx)
        selection = PatchSelectionPhase().run(state, ctx)
        slew = SlewPhase().run(state, ctx, selection=selection)
        return slew, ScienceScanPhase().run(slew.state, ctx, selection=slew)

    def test_a_subscan_is_a_whole_number_of_periods(self):
        """The block lasts ``n_cycles`` periods and rebuilds to its own length."""
        from fyst_trajectories.overhead.simulation import _generate_trajectory_for_block

        patch = _pong_patch()
        ctx = _make_ctx(patches=[patch])
        _, result = self._visit(ctx)

        assert _kinds(result.blocks) == ["retune", "science"]
        retune, science = result.blocks
        assert science.t_start.unix == pytest.approx(retune.t_stop.unix, abs=1e-6)
        n_cycles = science.metadata["scan_params"]["n_cycles"]
        # The 3600 s budget less the 300 s retune holds four 696 s periods.
        assert isinstance(n_cycles, int)
        assert n_cycles == 4
        assert science.duration == pytest.approx(n_cycles * _pong_period(patch), abs=1e-6)
        rebuilt = _generate_trajectory_for_block(science, ctx.site).trajectory
        assert abs(float(rebuilt.times[-1]) - science.duration) <= 0.1  # one timestep

    def test_a_patch_n_cycles_caps_the_count(self):
        patch = _pong_patch(scan_params={"n_cycles": 2})
        ctx = _make_ctx(patches=[patch])
        _, result = self._visit(ctx)

        assert _kinds(result.blocks) == ["retune", "science"]
        science = result.blocks[1]
        assert science.metadata["scan_params"]["n_cycles"] == 2
        assert science.duration == pytest.approx(2 * _pong_period(patch), abs=1e-6)

    @pytest.mark.parametrize("scan_type", ["pong", "constant_el"])
    def test_a_refused_subscan_books_neither_block(self, monkeypatch, scan_type):
        """A subscan the planner refuses leaves no retune; the tick idles."""

        def refuse(block, site):
            raise PointingError("refused for the test")

        monkeypatch.setattr(phases, "_generate_trajectory_for_block", refuse)
        if scan_type == "pong":
            ctx = _make_ctx(patches=[_pong_patch()])
        else:
            ctx = _ce_ready_ctx(_deep56_ce_patch())
        slew, result = self._visit(ctx)

        assert _kinds(result.blocks) == ["idle"]
        assert result.blocks[0].metadata["reason"] == "unplannable"
        # The boundary retune was never booked: the cadence tracker is untouched.
        assert result.state.cal_state.last_retune is None
        assert result.state.scan_counter == slew.state.scan_counter
        assert result.blocks[0].t_start.unix == pytest.approx(slew.state.current_time.unix)

    def test_a_refused_pong_retries_with_fewer_periods(self, monkeypatch):
        """A refusal at ``n`` periods emits the largest plannable count below it."""
        original = phases._generate_trajectory_for_block
        asked = []

        def refuse_above_two(block, site):
            n_cycles = block.metadata["scan_params"].get("n_cycles", 1)
            asked.append(n_cycles)
            if n_cycles > 2:
                raise PointingError("too long for the test")
            return original(block, site)

        monkeypatch.setattr(phases, "_generate_trajectory_for_block", refuse_above_two)
        patch = _pong_patch()
        ctx = _make_ctx(patches=[patch])
        _, result = self._visit(ctx)

        assert asked == [4, 3, 2]
        assert _kinds(result.blocks) == ["retune", "science"]
        science = result.blocks[1]
        assert science.metadata["scan_params"]["n_cycles"] == 2
        assert science.duration == pytest.approx(2 * _pong_period(patch), abs=1e-6)

    def test_the_gate_passes_over_a_pong_whose_period_no_longer_fits(self):
        """A pong is selectable only while one period plus the due retune fits."""
        long_pong = _pong_patch("Long", width=8.0, height=8.0, priority=1.0)
        short_pong = _pong_patch("Short", priority=10.0)
        assert _pong_period(long_pong) == pytest.approx(2644.8)

        roomy = _make_ctx(patches=[long_pong, short_pong])
        result = PatchSelectionPhase().run(_initial_state(roomy), roomy)
        assert result.selection is not None
        assert result.selection.name == "Long"

        # 2700 s left: the 2644.8 s period plus the due 300 s retune does not fit.
        tight = _make_ctx(patches=[long_pong, short_pong], end_time="2026-06-15T02:45:00")
        result = PatchSelectionPhase().run(_initial_state(tight), tight)
        assert result.selection is not None
        assert result.selection.name == "Short"

        # Alone, the long pong is not selected and the tick idles in place.
        alone = _make_ctx(patches=[long_pong], end_time="2026-06-15T02:45:00")
        result = PatchSelectionPhase().run(_initial_state(alone), alone)
        assert result.selection is None
        assert _kinds(result.blocks) == ["idle"]


class TestSchedulerComposition:
    """``Scheduler(ctx).run()`` matches ``generate_timeline`` and tiles its window."""

    def test_scheduler_matches_generate_timeline(self):
        from fyst_trajectories.overhead import generate_timeline

        ce_patch = _deep56_ce_patch()
        site = get_fyst_site()
        # A window that reaches selection, the slew and science (the pass
        # opens ~08:02), not only calibrations and idle ticks.
        start = "2026-06-15T07:30:00"
        end = "2026-06-15T09:30:00"

        ctx = SchedulerContext.build(
            patches=[ce_patch],
            site=site,
            start_time=Time(start, scale="utc"),
            end_time=Time(end, scale="utc"),
        )
        direct = Scheduler(ctx).run()
        wrapped = generate_timeline(
            patches=[ce_patch],
            site=site,
            start_time=start,
            end_time=end,
        )

        # Block counts identical; t_start times identical.
        assert len(direct.blocks) == len(wrapped.blocks)
        for a, b in zip(direct.blocks, wrapped.blocks, strict=True):
            assert a.block_type == b.block_type
            assert abs(a.t_start.unix - b.t_start.unix) < 1e-6
            assert abs(a.t_stop.unix - b.t_stop.unix) < 1e-6

    def test_the_window_tail_belongs_to_a_block(self):
        """The schedule tiles its declared window, tail included.

        The loop stops as soon as less than a minimum scan duration is
        left, so a window whose length is not a whole number of ticks
        would otherwise end with a stretch belonging to no block, and the
        four time totals would not add up to ``total_time``. This end time
        leaves a 57 s remainder, the largest found by sweeping the end of
        a single-patch night in 37 s steps.
        """
        from fyst_trajectories.overhead import BlockType, generate_timeline

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            timeline = generate_timeline(
                patches=[_deep56_ce_patch()],
                site=get_fyst_site(),
                start_time="2026-06-15T02:00:00",
                end_time="2026-06-15T06:12:57",
            )

        last = timeline.blocks[-1]
        assert last.t_stop.unix == pytest.approx(timeline.end_time.unix, abs=0.01)
        assert last.block_type == BlockType.IDLE
        assert last.metadata["reason"] == "window_closed"
        assert last.duration == pytest.approx(57.0, abs=0.01)
        accounted = (
            timeline.total_science_time
            + timeline.total_calibration_time
            + timeline.total_slew_time
            + timeline.total_idle_time
        )
        assert accounted == pytest.approx(timeline.total_time, abs=0.01)


class TestRisingSetting:
    """A CE patch's ``scan_params['rising']`` request is honored end to end.

    The test field (RA=40, Dec=-32) transits near zenith at FYST. At
    el=50 this window contains only the SETTING pass (open ~15:46 UTC):
    the rising pass's opening crossing precedes the window start, so the
    planner cannot solve it from any in-window anchor. Under the corridor
    gate the no-request default therefore lands on the setting pass, and
    an explicit rising request is refused outright (without the gate, an
    hour-angle default emits "rising" blocks here that
    ``schedule_to_trajectories`` could never reconstruct).
    """

    # Window brackets both crossings of the el=50 transit of this field.
    _START = "2026-06-15T10:00:00"
    _END = "2026-06-15T17:00:00"
    _RA = 40.0

    def _field_patch(self, scan_params=None):
        return ObservingPatch(
            name="transit_field",
            ra_center=self._RA,
            dec_center=-32.0,
            width=20.0,
            height=10.0,
            scan_type="constant_el",
            velocity=1.0,
            elevation=50.0,
            scan_params=scan_params or {},
        )

    def _first_science(self, patch):
        from fyst_trajectories.overhead import BlockType, generate_timeline

        timeline = generate_timeline(
            patches=[patch],
            site=get_fyst_site(),
            start_time=self._START,
            end_time=self._END,
        )
        science = [b for b in timeline.blocks if b.block_type == BlockType.SCIENCE]
        assert science, "expected at least one science block"
        return science[0]

    def test_setting_request_lands_on_setting_side(self):
        """``rising=False`` moves the block to the setting (west) crossing."""
        from fyst_trajectories.coordinates import Coordinates

        block = self._first_science(self._field_patch({"rising": False}))
        coords = Coordinates(get_fyst_site())

        # The block must carry the requested flag.
        assert block.rising is False
        # Selection must have waited for the setting side: hour angle > 0
        # (west of the meridian) at the block start, per the planner's own
        # HA convention.
        ha = float(coords.get_hour_angle(self._RA, block.t_start))
        assert ha > 0.0, f"expected setting-side (HA>0), got HA={ha:.1f}"
        # The az sweep must sit in the western (setting) half of the
        # transit. Its center is west of the meridian: az > 180 for a
        # source that transits to the south at FYST's southern latitude.
        az_center = 0.5 * (block.az_start + block.az_end)
        assert az_center > 180.0, f"expected western az center, got {az_center:.1f}"

    def test_no_rising_key_picks_the_plannable_pass(self):
        """Absent the key, selection lands on the only plannable pass.

        For this window that is the setting pass, and the default path is
        bit-for-bit identical to an explicit ``rising=False`` request.
        """
        from fyst_trajectories.coordinates import Coordinates

        default_block = self._first_science(self._field_patch())
        setting_block = self._first_science(self._field_patch({"rising": False}))
        coords = Coordinates(get_fyst_site())

        assert default_block.rising is False
        ha = float(coords.get_hour_angle(self._RA, default_block.t_start))
        assert ha > 0.0, f"expected setting-side (HA>0), got HA={ha:.1f}"
        assert abs(default_block.t_start.unix - setting_block.t_start.unix) < 1e-6

    def test_unplannable_rising_request_is_refused(self):
        """``rising=True`` past its pass emits NO science blocks.

        The rising pass's opening crossing precedes this window, so no
        in-window anchor can reconstruct a rising scan; the corridor gate
        refuses selection instead of emitting dead-air blocks.
        """
        from fyst_trajectories.overhead import BlockType, generate_timeline

        timeline = generate_timeline(
            patches=[self._field_patch({"rising": True})],
            site=get_fyst_site(),
            start_time=self._START,
            end_time=self._END,
        )
        science = [b for b in timeline.blocks if b.block_type == BlockType.SCIENCE]
        assert science == []


class TestCoherentAzimuthFrames:
    """The scheduler's azimuth frame stays coherent from slew to scan."""

    def test_normalize_az_ref_picks_nearest_representative(self):
        site = get_fyst_site()
        # Without ref: the window-centre representative (seam at az 270).
        assert _normalize_az(280.0, site) == pytest.approx(-80.0)
        # With ref, the in-limits representative nearest the pose wins.
        assert _normalize_az(280.0, site, ref=260.0) == pytest.approx(280.0)
        assert _normalize_az(280.0, site, ref=-150.0) == pytest.approx(-80.0)
        # A representative outside the limits is never chosen: 365 is out
        # of [-180, 360], so a pose at 355 still gets 5.
        assert _normalize_az(5.0, site, ref=355.0) == pytest.approx(5.0)

    def test_compute_az_range_explicit_window_contiguous(self):
        """A wrap-crossing az_min/az_max window stays one contiguous range."""
        site = get_fyst_site()
        patch = ObservingPatch(
            name="win",
            ra_center=150.0,
            dec_center=2.2,
            width=4.0,
            height=4.0,
            scan_type="constant_el",
            velocity=0.5,
            elevation=45.0,
            scan_params={"az_min": 350.0, "az_max": 15.0},
        )
        lo, hi = _compute_az_range(patch, 316.8, 45.0, site)
        assert lo == pytest.approx(-10.0)
        assert hi == pytest.approx(15.0)

    def test_compute_az_range_width_branch_shift(self):
        """A range poking past az_max shifts both endpoints together."""
        site = get_fyst_site()
        patch = ObservingPatch(
            name="edge",
            ra_center=150.0,
            dec_center=2.2,
            width=10.0,
            height=4.0,
            scan_type="pong",
            velocity=0.5,
        )
        lo, hi = _compute_az_range(patch, 355.0, 45.0, site)
        assert lo == pytest.approx(-12.071, abs=1e-3)
        assert hi == pytest.approx(2.071, abs=1e-3)
        assert lo <= hi

    def test_slew_follows_scan_branch(self):
        """An explicit wrap-crossing window pulls the slew onto its branch.

        Targeting the field centre (316.8) while the scan sits at
        (-10, 15) would be a ~327 deg cable-wrap unwind modelled as zero
        time, invisible to validate() because the pose tracker follows
        the recorded blocks.
        """
        ce = ObservingPatch(
            name="win",
            ra_center=150.0,
            dec_center=2.2,
            width=4.0,
            height=4.0,
            scan_type="constant_el",
            velocity=0.5,
            elevation=45.0,
            scan_params={"az_min": 350.0, "az_max": 15.0},
        )
        ctx = _make_ctx(patches=[ce])
        state = _initial_state(ctx)
        sel = PhaseResult(state=state, blocks=[], selection=ce, best_az=316.8, best_el=44.0)

        result = SlewPhase().run(state, ctx, selection=sel)

        block = result.blocks[0]
        # The slew targets the scan range's midpoint on its branch, and
        # the pose follows it.
        assert block.az_end == pytest.approx(2.5)
        assert result.state.current_az == pytest.approx(2.5)
        lo, hi = _compute_az_range(ce, sel.best_az, sel.best_el, ctx.site)
        assert abs(0.5 * (lo + hi) - result.state.current_az) <= 180.0

    def test_slew_wrap_check_holds_the_swept_corridor(self, monkeypatch):
        """The wrap must hold the corridor the pass sweeps, not the window.

        The science window is an estimate of the field's width at the tick
        azimuth. A drifting constant-elevation pass crosses a corridor tens
        of degrees wider, so a wrap admitted against the narrow range can
        be one the scan does not fit inside.
        """
        from fyst_trajectories.overhead.scheduler import phases

        ce = _deep56_ce_patch()
        ctx = _ce_ready_ctx(ce)
        state = _initial_state(ctx)
        az, el = ctx.coords.radec_to_altaz(ce.ra_center, ce.dec_center, state.current_time)
        sel = PhaseResult(
            state=state,
            blocks=[],
            selection=ce,
            best_az=_normalize_az(az, ctx.site, ref=state.current_az),
            best_el=el,
        )

        spans = []
        original = phases.plan_transition

        def recording(*args, **kwargs):
            spans.append(kwargs["goal_az_span"])
            return original(*args, **kwargs)

        monkeypatch.setattr(phases, "plan_transition", recording)

        SlewPhase().run(state, ctx, selection=sel)

        (span,) = spans
        win_lo, win_hi = _compute_az_range(ce, sel.best_az, sel.best_el, ctx.site)
        assert span[0] == pytest.approx(102.33, abs=0.01)
        assert span[1] == pytest.approx(215.62, abs=0.01)
        assert span[1] - span[0] > (win_hi - win_lo) + 40.0

    def test_slew_keeps_field_centre_on_same_branch(self):
        """Without a branch split the slew target is the field centre, exactly."""
        ce = _deep56_ce_patch()
        ctx = _make_ctx(patches=[ce])
        state = _initial_state(ctx)
        sel = PhaseResult(state=state, blocks=[], selection=ce, best_az=114.83, best_el=29.05)

        result = SlewPhase().run(state, ctx, selection=sel)

        assert result.blocks[0].az_end == 114.83
        assert result.state.current_az == 114.83


class TestRetuneElevation:
    """Boundary retunes are stamped at the science elevation."""

    def test_retune_uses_pinned_elevation(self):
        ce_patch = _deep56_ce_patch()  # pins elevation=50.0
        overhead = OverheadModel()
        # Anchored where Deep56's pass is plannable, so the subscan is emitted.
        ctx = _ce_ready_ctx(ce_patch, overhead_model=overhead)
        window = overhead.min_scan_duration + overhead.retune_duration + 1.0
        state = _initial_state(ctx)

        # best_el deliberately differs from the pinned elevation so the
        # test discriminates: a retune stamped at best_el would fail here.
        state, blocks, _ = ScienceScanPhase._emit_subscans_with_retunes(
            state=state,
            ctx=ctx,
            best_patch=ce_patch,
            best_el=29.0,
            n_subscans=1,
            subscan_duration=window,
            rising=True,
            az_start_sci=100.0,
            az_end_sci=140.0,
            deadline=ctx.start_time + TimeDelta(window, format="sec"),
            t0_scan=ctx.start_time.isot,
        )
        retunes = [b for b in blocks if str(b.scan_type) == "retune"]
        science = [b for b in blocks if str(b.block_type) == "science"]
        assert retunes and science
        assert retunes[0].elevation == ce_patch.elevation
        assert science[0].elevation == ce_patch.elevation


class TestPatchNameUniqueness:
    """Two patches may not share a name within one schedule.

    Patch names key the constant-elevation corridor memo and label every
    emitted block. ``ObservingPatch`` states the uniqueness precondition;
    unenforced, a duplicate name would silently make two patches read each
    other's crossing solve.
    """

    def test_duplicate_names_are_refused(self):
        """The context refuses to build and names the repeated value."""
        patches = [_deep56_ce_patch(), _deep56_ce_patch()]
        with pytest.raises(ValueError, match="Patch names must be unique.*deep56"):
            _make_ctx(patches)

    def test_generate_timeline_refuses_too(self):
        """The public entry point carries the same refusal."""
        from fyst_trajectories.overhead import generate_timeline

        with pytest.raises(ValueError, match="Patch names must be unique"):
            generate_timeline(
                patches=[_deep56_ce_patch(), _deep56_ce_patch()],
                site=get_fyst_site(),
                start_time="2026-06-15T02:00:00",
                end_time="2026-06-15T04:00:00",
            )

    def test_distinct_names_still_build(self):
        """Two patches differing only in name are accepted."""
        ctx = _make_ctx([_deep56_ce_patch("a"), _deep56_ce_patch("b")])
        assert [p.name for p in ctx.patches] == ["a", "b"]


class TestUnpinnedConstantElevation:
    """A constant-elevation patch must pin its elevation to be scheduled.

    The crossing-pass gate solves at the patch's pinned elevation; an
    unpinned patch would fall back to the field centre's current elevation,
    which its leading edge has already crossed, so it would never be selected.
    """

    def test_build_refuses_an_unpinned_patch(self):
        """The context refuses to build and names the patch."""
        patch = ObservingPatch(
            name="floating",
            ra_center=24.0,
            dec_center=-32.0,
            width=40.0,
            height=10.0,
            scan_type="constant_el",
            velocity=1.0,
        )
        with pytest.raises(ValueError, match="pinned elevation.*floating"):
            _make_ctx([patch])
