"""Phase-level unit tests for the scheduler refactor.

Each scheduler phase (``CalibrationPhase``, ``PatchSelectionPhase``,
``SlewPhase``, ``ScienceScanPhase``) is independently testable now
that ``timeline.py`` has been decomposed. These tests exercise the
public phase API directly and assert state/block invariants that
would previously have required running the full scheduler.
"""

import pytest
from astropy.time import Time, TimeDelta

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


def _ce_ready_ctx(patch, **ctx_kwargs):
    """Build a context anchored one tick before the patch's rising pass opens.

    The corridor gate (2026-07-16) only selects a CE patch while its
    crossing pass is imminent, so phase-mechanics tests anchor the schedule
    window just before the pass opening (located via the scheduler's own
    corridor solver) instead of running from the fixture night's start,
    where the patch is hours from plannable.
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
        # At startup, every due cadence fires: retune, pointing_cal,
        # focus, skydip (and planet_cal when a planet is visible).
        assert len(result.blocks) >= 4
        # The state advances past every block.
        assert result.state.current_time.unix > state.current_time.unix
        # The cal state has updated, at least retune is no longer None.
        assert result.state.cal_state.last_retune is not None

    def test_idle_ticks_do_not_retune(self):
        """A cadence-0 retune is scan-coupled: an all-idle night retunes once.

        Before 2026-07-16 the per-tick CalibrationPhase consumed the
        always-due cadence-0 retune on every idle tick, booking a 5 s
        retune every 300 s while the telescope sat parked. Only the
        startup burst may fire one outside a scan boundary.
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
        # Previously the whole assertion sat behind `if result.selection is None`,
        # which silently passed whenever the geometry put the patch above horizon.
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
        """Passing no selection raises."""
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
        ctx = _ce_ready_ctx(ce_patch)
        state = _initial_state(ctx)
        selection = PatchSelectionPhase().run(state, ctx)

        result = SlewPhase().run(state, ctx, selection=selection)

        # Slew from state's (180, 50) to deep56's ~(115, 29): a large ~65 deg az move.
        assert len(result.blocks) == 1
        block = result.blocks[0]
        assert str(block.block_type) == "slew"
        assert block.az_start == state.current_az
        assert block.az_end == selection.best_az
        # Pin the dominant az move against the ~65 deg expectation stated above
        # (tolerance covers minor ephemeris drift in the CE corridor anchor).
        assert abs(block.az_end - block.az_start) == pytest.approx(65.1, abs=1.5)
        # Time advanced by the slew duration (~29 s for this move).
        assert result.state.current_time.unix > state.current_time.unix
        assert block.duration == pytest.approx(28.7, abs=1.0)


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
        ctx = _ce_ready_ctx(ce_patch)
        state = _initial_state(ctx)
        selection = PatchSelectionPhase().run(state, ctx)
        slew = SlewPhase().run(state, ctx, selection=selection)

        result = ScienceScanPhase().run(slew.state, ctx, selection=slew)

        science_blocks = [b for b in result.blocks if str(b.block_type) == "science"]
        assert len(science_blocks) >= 1
        # Scan counter must have advanced exactly once, regardless of subscans.
        assert result.state.scan_counter == slew.state.scan_counter + 1

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

        state, blocks = ScienceScanPhase._emit_subscans_with_retunes(
            state=state,
            ctx=ctx,
            best_patch=ce_patch,
            best_el=50.0,
            n_subscans=1,
            subscan_duration=window,
            rising=True,
            az_start_sci=100.0,
            az_end_sci=140.0,
            t0_scan=None,
        )
        assert blocks == []

    def test_boundary_retune_plus_min_subscan_fit(self):
        """With one retune-width more room, the visit emits retune + science."""
        ce_patch = _deep56_ce_patch()
        overhead = OverheadModel()
        start = Time("2026-06-15T02:00:00", scale="utc")
        window = overhead.min_scan_duration + overhead.retune_duration + 1.0
        ctx = _make_ctx(
            patches=[ce_patch],
            start_time=start.isot,
            end_time=(start + TimeDelta(window, format="sec")).isot,
            overhead_model=overhead,
        )
        state = _initial_state(ctx)

        state, blocks = ScienceScanPhase._emit_subscans_with_retunes(
            state=state,
            ctx=ctx,
            best_patch=ce_patch,
            best_el=50.0,
            n_subscans=1,
            subscan_duration=window,
            rising=True,
            az_start_sci=100.0,
            az_end_sci=140.0,
            t0_scan=None,
        )
        kinds = [
            str(b.scan_type) if str(b.block_type) == "calibration" else "science" for b in blocks
        ]
        assert kinds == ["retune", "science"]
        science = blocks[1]
        assert science.duration >= overhead.min_scan_duration
        assert science.t_stop.unix <= ctx.end_time.unix + 1e-6

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


class TestRisingSetting:
    """A CE patch's ``scan_params['rising']`` request is honored end to end.

    The test field (RA=40, Dec=-32) transits near zenith at FYST. At
    el=50 this window contains only the SETTING pass (open ~15:46 UTC):
    the rising pass's opening crossing precedes the window start, so the
    planner cannot solve it from any in-window anchor. Under the
    2026-07-16 corridor gate the no-request default therefore lands on
    the setting pass, and an explicit rising request is refused outright
    (before the gate, the hour-angle default emitted "rising" blocks
    here that ``schedule_to_trajectories`` could never reconstruct).
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
        (Before the corridor gate the hour-angle default stayed on the
        rising half until transit, emitting blocks whose crossing the
        planner could no longer solve.)
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
