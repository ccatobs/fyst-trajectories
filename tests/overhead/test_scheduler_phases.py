"""Phase-level unit tests for the scheduler refactor.

Each scheduler phase (``CalibrationPhase``, ``PatchSelectionPhase``,
``SlewPhase``, ``ScienceScanPhase``) is independently testable now
that ``timeline.py`` has been decomposed. These tests exercise the
public phase API directly and assert state/block invariants that
would previously have required running the full scheduler.
"""

import pytest
from astropy.time import Time

from fyst_trajectories import get_fyst_site
from fyst_trajectories.overhead import (
    CalibrationPolicy,
    CalibrationState,
    ObservingPatch,
    OverheadModel,
)
from fyst_trajectories.overhead.scheduler import (
    CalibrationPhase,
    PatchSelectionPhase,
    PhaseResult,
    Scheduler,
    SchedulerContext,
    SchedulerState,
    ScienceScanPhase,
    SlewPhase,
)


def _make_ctx(
    patches,
    *,
    start_time="2026-06-15T02:00:00",
    end_time="2026-06-15T10:00:00",
    overhead_model=None,
    calibration_policy=None,
    time_step=300.0,
):
    """Build a context with sensible defaults for phase-level tests."""
    return SchedulerContext.build(
        patches=patches,
        site=get_fyst_site(),
        start_time=Time(start_time, scale="utc"),
        end_time=Time(end_time, scale="utc"),
        overhead_model=overhead_model or OverheadModel(),
        calibration_policy=calibration_policy or CalibrationPolicy(),
        time_step=time_step,
    )


def _initial_state(ctx):
    return SchedulerState.initial(start_time=ctx.start_time, cal_state=CalibrationState())


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


class TestCalibrationPhase:
    """Calibration phase emits a block when a cadence has elapsed."""

    def test_startup_emits_multiple_cal_blocks(self):
        """With CalibrationState.last_* all None, all cadences fire at t=0."""
        ctx = _make_ctx(patches=[])
        state = _initial_state(ctx)

        result = CalibrationPhase().run(state, ctx)

        assert isinstance(result, PhaseResult)
        # At startup, every due cadence fires: retune, pointing_cal,
        # focus, skydip (and planet_cal when a planet is visible).
        assert len(result.blocks) >= 4
        # The state advances past every block.
        assert result.state.current_time.unix > state.current_time.unix
        # The cal state has updated — at least retune is no longer None.
        assert result.state.cal_state.last_retune is not None

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
        # Previously the whole assertion sat behind `if result.selection is None`,
        # which silently passed whenever the geometry put the patch above horizon.
        assert result.selection is None
        assert len(result.blocks) == 1
        assert str(result.blocks[0].block_type) == "idle"
        assert result.skip_to_next_iter is True

    def test_observable_patch_selected(self):
        """A well-placed patch is selected with best_az/best_el populated."""
        ce_patch = _deep56_ce_patch()
        ctx = _make_ctx(patches=[ce_patch])
        # Advance past the initial calibration burst so the selection runs
        # against an already-observable sky.
        state = _initial_state(ctx)
        state = CalibrationPhase().run(state, ctx).state

        result = PatchSelectionPhase().run(state, ctx)

        assert result.selection is not None
        assert result.selection.name == "deep56"
        assert result.best_az is not None
        assert result.best_el is not None
        assert result.skip_to_next_iter is False
        # No blocks emitted — the selection result is consumed by next phase.
        assert result.blocks == []


class TestSlewPhase:
    """Slew phase emits a block when the telescope needs to move."""

    def test_requires_selection(self):
        """Passing no selection raises."""
        ctx = _make_ctx(patches=[])
        state = _initial_state(ctx)

        with pytest.raises(ValueError, match="PatchSelectionPhase"):
            SlewPhase().run(state, ctx)

    def test_small_slew_is_skipped(self):
        """When slew+settle <= 1s, no block is emitted."""
        ce_patch = _deep56_ce_patch()
        overhead = OverheadModel(settle_time=0.0)
        ctx = _make_ctx(patches=[ce_patch], overhead_model=overhead)
        state = _initial_state(ctx)
        state = CalibrationPhase().run(state, ctx).state
        selection = PatchSelectionPhase().run(state, ctx)
        # Simulate a state where the telescope is already at the patch.
        assert selection.best_az is not None
        assert selection.best_el is not None
        at_patch = state.advanced(current_az=selection.best_az, current_el=selection.best_el)

        result = SlewPhase().run(at_patch, ctx, selection=selection)

        # slew_time was < 1s, no block emitted; state unchanged.
        assert result.blocks == []
        assert result.state.current_time.unix == at_patch.current_time.unix

    def test_large_slew_emits_block(self):
        """A large az change yields a SLEW block advancing current_time."""
        ce_patch = _deep56_ce_patch()
        ctx = _make_ctx(patches=[ce_patch])
        state = _initial_state(ctx)
        state = CalibrationPhase().run(state, ctx).state
        selection = PatchSelectionPhase().run(state, ctx)

        result = SlewPhase().run(state, ctx, selection=selection)

        # Slew from state's (180, 50) to deep56's az/el is typically ~30+ deg
        assert len(result.blocks) == 1
        block = result.blocks[0]
        assert str(block.block_type) == "slew"
        assert block.az_start == state.current_az
        assert block.az_end == selection.best_az
        # Time advanced by the slew duration.
        assert result.state.current_time.unix > state.current_time.unix


class TestScienceScanPhase:
    """Science scan phase emits subscans with interleaved retunes."""

    def test_requires_selection(self):
        """Passing no selection raises."""
        ctx = _make_ctx(patches=[])
        state = _initial_state(ctx)

        with pytest.raises(ValueError, match="PatchSelectionPhase"):
            ScienceScanPhase().run(state, ctx)

    def test_emits_one_or_more_science_blocks(self):
        """A healthy CE patch yields at least one science block."""
        ce_patch = _deep56_ce_patch()
        ctx = _make_ctx(patches=[ce_patch])
        state = _initial_state(ctx)
        state = CalibrationPhase().run(state, ctx).state
        selection = PatchSelectionPhase().run(state, ctx)
        slew = SlewPhase().run(state, ctx, selection=selection)

        result = ScienceScanPhase().run(slew.state, ctx, selection=slew)

        science_blocks = [b for b in result.blocks if str(b.block_type) == "science"]
        assert len(science_blocks) >= 1
        # Scan counter must have advanced exactly once, regardless of subscans.
        assert result.state.scan_counter == slew.state.scan_counter + 1

    def test_long_scan_splits_into_subscans(self):
        """When scan_duration > max_scan_duration, emit multiple subscans."""
        ce_patch = _deep56_ce_patch()
        # Force small max_scan_duration so splitting is guaranteed.
        overhead = OverheadModel(max_scan_duration=1200.0)
        ctx = _make_ctx(
            patches=[ce_patch],
            overhead_model=overhead,
        )
        state = _initial_state(ctx)
        state = CalibrationPhase().run(state, ctx).state
        selection = PatchSelectionPhase().run(state, ctx)
        slew = SlewPhase().run(state, ctx, selection=selection)

        result = ScienceScanPhase().run(slew.state, ctx, selection=slew)

        science_blocks = [b for b in result.blocks if str(b.block_type) == "science"]
        # With a 4-6h observable window and 1200s subscan max, expect 2+.
        assert len(science_blocks) >= 2
        # Subscan indices should be sequential.
        sub_indices = [b.subscan_index for b in science_blocks]
        assert sub_indices == list(range(len(sub_indices)))


class TestSchedulerComposition:
    """The Scheduler class should produce a valid timeline."""

    def test_scheduler_matches_generate_timeline(self):
        """Direct Scheduler(ctx).run() yields the same output as generate_timeline."""
        from fyst_trajectories.overhead import generate_timeline

        ce_patch = _deep56_ce_patch()
        site = get_fyst_site()
        start = "2026-06-15T02:00:00"
        end = "2026-06-15T06:00:00"

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
