"""Phase-level unit tests for the offline scheduler.

Each scheduler phase (``CalibrationPhase``, ``PatchSelectionPhase``,
``SlewPhase``, ``ScienceScanPhase``) is exercised through its public
API directly, asserting the state and block invariants that phase owns
without driving the whole scheduler loop. A composition test checks that
the loop reproduces ``generate_timeline``.
"""

import warnings

import pytest
from _sun_stubs import pose_blocker
from astropy.time import Time, TimeDelta

from fyst_trajectories import get_fyst_site
from fyst_trajectories.overhead import (
    CalibrationPolicy,
    CalibrationState,
    ElevationConstraint,
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
from fyst_trajectories.overhead.scheduler.helpers import (
    _compute_az_range,
    _normalize_az,
)


def _make_ctx(
    patches,
    *,
    start_time="2026-06-15T02:00:00",
    end_time="2026-06-15T10:00:00",
    overhead_model=None,
    calibration_policy=None,
    time_step=300.0,
    sun_safe=None,
    constraints=None,
):
    """Build a context with sensible defaults for phase-level tests."""
    return SchedulerContext.build(
        patches=patches,
        site=get_fyst_site(),
        start_time=Time(start_time, scale="utc"),
        end_time=Time(end_time, scale="utc"),
        overhead_model=overhead_model or OverheadModel(),
        calibration_policy=calibration_policy or CalibrationPolicy(),
        constraints=constraints,
        time_step=time_step,
        sun_safe=sun_safe,
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
        # At startup, every due cadence fires: retune, pointing_cal,
        # focus, skydip (and planet_cal when a planet is visible).
        assert len(result.blocks) >= 4
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
        # Pin the dominant az move against the ~65 deg expectation stated above
        # (tolerance covers minor ephemeris drift in the CE corridor anchor).
        assert abs(block.az_end - block.az_start) == pytest.approx(65.1, abs=1.5)
        # Time advanced by the slew duration (~29 s for this move).
        assert result.state.current_time.unix > state.current_time.unix
        assert block.duration == pytest.approx(28.7, abs=1.0)


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
            deadline=ctx.end_time,
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
            deadline=ctx.end_time,
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
    """``Scheduler(ctx).run()`` matches ``generate_timeline`` and tiles its window."""

    def test_scheduler_matches_generate_timeline(self):
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
        start = Time("2026-06-15T02:00:00", scale="utc")
        window = overhead.min_scan_duration + overhead.retune_duration + 1.0
        ctx = _make_ctx(
            patches=[ce_patch],
            start_time=start.isot,
            end_time=(start + TimeDelta(window, format="sec")).isot,
            overhead_model=overhead,
        )
        state = _initial_state(ctx)

        # best_el deliberately differs from the pinned elevation so the
        # test discriminates: a retune stamped at best_el would fail here.
        state, blocks = ScienceScanPhase._emit_subscans_with_retunes(
            state=state,
            ctx=ctx,
            best_patch=ce_patch,
            best_el=29.0,
            n_subscans=1,
            subscan_duration=window,
            rising=True,
            az_start_sci=100.0,
            az_end_sci=140.0,
            deadline=ctx.end_time,
            t0_scan=None,
        )
        retunes = [b for b in blocks if str(b.scan_type) == "retune"]
        science = [b for b in blocks if str(b.block_type) == "science"]
        assert retunes and science
        assert retunes[0].elevation == ce_patch.elevation
        assert science[0].elevation == ce_patch.elevation


def _sky_band_blocker(lo: float, hi: float, until: Time | None = None):
    """Build a point predicate that is unsafe inside a sky-azimuth band.

    The band is open, in degrees on the sky (any encoder wrap maps into
    it); with ``until`` the band clears at that time, so a refusal can be
    retried against a Sun that has moved on.
    """

    def predicate(az, el, t):
        if until is not None and t >= until:
            return True
        return not (lo < float(az) % 360.0 < hi)

    return predicate


class TestSlewTransition:
    """The slew is Sun-checked; a refusal idles the tick instead of raising."""

    def _pong_patch(self, **overrides):
        params = dict(
            name="field",
            ra_center=24.0,
            dec_center=-32.0,
            width=10.0,
            height=10.0,
            scan_type="pong",
            velocity=1.0,
        )
        params.update(overrides)
        return ObservingPatch(**params)

    def test_blocked_path_idles_with_the_reason(self):
        """A blocked direct path emits one idle tick, labelled, at the unmoved pose."""
        patch = self._pong_patch()
        # Sky azimuth 115 has a single encoder image, so the blocked band
        # between the bootstrap pose (180) and the goal leaves no wrap.
        ctx = _make_ctx(patches=[patch], sun_safe=_sky_band_blocker(140.0, 160.0))
        state = _initial_state(ctx)
        sel = PhaseResult(state=state, blocks=[], selection=patch, best_az=115.0, best_el=50.0)

        result = SlewPhase().run(state, ctx, selection=sel)

        assert result.skip_to_next_iter is True
        assert result.stop is False
        assert result.selection is None
        (block,) = result.blocks
        assert str(block.block_type) == "idle"
        assert block.metadata["reason"] == "sun_path"
        assert block.duration == pytest.approx(ctx.time_step)
        assert block.az_start == state.current_az and block.elevation == state.current_el
        assert result.state.current_az == state.current_az
        assert result.state.current_time.unix == pytest.approx(
            state.current_time.unix + ctx.time_step
        )

    def test_sun_blocked_goal_idles_as_sun_point(self):
        """A goal inside the zone in every wrap is refused as ``sun_point``."""
        patch = self._pong_patch()
        ctx = _make_ctx(patches=[patch], sun_safe=_sky_band_blocker(100.0, 130.0))
        state = _initial_state(ctx)
        sel = PhaseResult(state=state, blocks=[], selection=patch, best_az=115.0, best_el=50.0)

        result = SlewPhase().run(state, ctx, selection=sel)

        assert result.blocks[0].metadata["reason"] == "sun_point"
        assert result.skip_to_next_iter is True

    def test_selection_idle_carries_no_reason(self):
        """The nothing-observable idle keeps its bare metadata."""
        ctx = _make_ctx(patches=[])
        state = _initial_state(ctx)

        result = PatchSelectionPhase().run(state, ctx)

        assert result.blocks[0].metadata == {}

    def test_far_wrap_carries_the_science_frame(self):
        """When only the far wrap has a clear path, best_az follows it.

        Sky azimuth 200 has encoder images 200 and -160. The band blocks
        the short path from 180 to 200 but not the long way round, so the
        transition lands on -160 and the science range downstream must be
        derived on that wrap, not a full turn away from the telescope.
        """
        patch = self._pong_patch()
        ctx = _make_ctx(patches=[patch], sun_safe=_sky_band_blocker(185.0, 195.0))
        state = _initial_state(ctx)
        sel = PhaseResult(state=state, blocks=[], selection=patch, best_az=200.0, best_el=50.0)

        result = SlewPhase().run(state, ctx, selection=sel)

        (block,) = result.blocks
        assert str(block.block_type) == "slew"
        assert block.az_end == pytest.approx(-160.0)
        assert result.state.current_az == pytest.approx(-160.0)
        assert result.best_az == pytest.approx(-160.0)
        lo, hi = _compute_az_range(patch, result.best_az, result.best_el, ctx.site)
        assert lo <= result.state.current_az <= hi
        # The long way round is priced as such.
        assert block.duration > 100.0

    def test_explicit_window_cannot_follow_the_far_wrap(self):
        """An explicit window pins the range, so a far-wrap-only slew idles."""
        patch = self._pong_patch(scan_params={"az_min": 200.0, "az_max": 220.0})
        ctx = _make_ctx(patches=[patch], sun_safe=_sky_band_blocker(185.0, 195.0))
        state = _initial_state(ctx)
        sel = PhaseResult(state=state, blocks=[], selection=patch, best_az=210.0, best_el=50.0)

        result = SlewPhase().run(state, ctx, selection=sel)

        assert result.skip_to_next_iter is True
        assert result.blocks[0].metadata["reason"] == "no_wrap"
        assert result.state.current_az == state.current_az

    def test_scheduler_retries_once_the_path_clears(self):
        """The loop idles on a refusal and slews once the Sun has moved.

        The band between the bootstrap pose and the field clears at a
        fixed time; every refusal must precede it, each labelled and at
        the unmoved pose, and the first slew and the science must follow.
        """
        from fyst_trajectories.overhead import generate_timeline

        patch = self._pong_patch()
        clear_at = Time("2026-06-15T08:35:00", scale="utc")
        timeline = generate_timeline(
            patches=[patch],
            site=get_fyst_site(),
            start_time="2026-06-15T07:50:00",
            end_time="2026-06-15T09:30:00",
            sun_safe=_sky_band_blocker(140.0, 160.0, until=clear_at),
        )

        refused = [b for b in timeline.blocks if b.metadata.get("reason") == "sun_path"]
        assert len(refused) >= 2
        assert all(b.t_start.unix < clear_at.unix for b in refused)
        assert all(b.az_start == 180.0 and str(b.block_type) == "idle" for b in refused)
        slews = [b for b in timeline.blocks if str(b.block_type) == "slew"]
        science = [b for b in timeline.blocks if str(b.block_type) == "science"]
        assert slews and science
        assert slews[0].t_start.unix >= clear_at.unix
        assert science[0].t_start.unix > slews[0].t_start.unix
        assert timeline.validate() == []


class TestSunEscape:
    """A pose the zone has overtaken is moved out before the telescope idles or slews."""

    def _pong_patch(self):
        return ObservingPatch(
            name="field",
            ra_center=24.0,
            dec_center=-32.0,
            width=10.0,
            height=10.0,
            scan_type="pong",
            velocity=1.0,
        )

    def test_overtaken_pose_escapes_before_the_slew(self):
        """The bootstrap pose is inside the zone: the slew phase escapes and restarts."""
        patch = self._pong_patch()
        ctx = _make_ctx(patches=[patch], sun_safe=pose_blocker(180.0, 50.0))
        state = _initial_state(ctx)
        sel = PhaseResult(state=state, blocks=[], selection=patch, best_az=115.0, best_el=50.0)

        result = SlewPhase().run(state, ctx, selection=sel)

        (block,) = result.blocks
        assert str(block.block_type) == "slew" and block.patch_name == "sun_escape"
        assert block.az_start == 180.0
        assert result.skip_to_next_iter is True and result.selection is None
        assert (result.state.current_az, result.state.current_el) == (block.az_end, block.elevation)
        moved_az = abs(result.state.current_az - 180.0) >= 5.0
        moved_el = abs(result.state.current_el - 50.0) >= 5.0
        assert moved_az or moved_el
        arrival = state.current_time.unix + block.duration
        assert result.state.current_time.unix == pytest.approx(arrival)

    def test_idle_path_escapes_instead_of_parking_in_the_zone(self):
        """With nothing observable, the selection idle still moves an overtaken pose."""
        ctx = _make_ctx(patches=[], sun_safe=pose_blocker(180.0, 50.0))
        state = _initial_state(ctx)

        result = PatchSelectionPhase().run(state, ctx)

        (block,) = result.blocks
        assert block.patch_name == "sun_escape"
        assert result.skip_to_next_iter is True
        # The next tick idles at the escape pose, with no reason label.
        again = PatchSelectionPhase().run(result.state, ctx)
        assert str(again.blocks[0].block_type) == "idle"
        assert again.blocks[0].az_start == result.state.current_az
        assert again.blocks[0].metadata == {}

    def test_escape_searches_down_to_the_schedule_elevation_floor(self):
        """The escape stops at the observing floor, not at the mount limit.

        Only sky at or below 35 deg is clear here, and the schedule's
        elevation constraint floors it at 40, so there is nowhere to go
        and the tick idles labelled ``no_escape``. Searching down to the
        mount limit (20 deg at FYST) instead would park the telescope
        5 deg below the sky the selection phase is willing to use.
        """
        ctx = _make_ctx(
            patches=[],
            sun_safe=lambda az, el, t: float(el) <= 35.0,
            constraints=[ElevationConstraint(el_min=40.0, el_max=90.0)],
        )
        assert ctx.el_floor == 40.0
        assert ctx.slew_safe is not None
        state = _initial_state(ctx)

        result = PatchSelectionPhase().run(state, ctx)

        (block,) = result.blocks
        assert str(block.block_type) == "idle"
        assert block.metadata["reason"] == "no_escape"
        assert (result.state.current_az, result.state.current_el) == (
            state.current_az,
            state.current_el,
        )

    def test_trapped_pose_idles_with_no_escape(self):
        ctx = _make_ctx(patches=[], sun_safe=lambda az, el, t: False)
        state = _initial_state(ctx)

        result = PatchSelectionPhase().run(state, ctx)

        (block,) = result.blocks
        assert str(block.block_type) == "idle"
        assert block.metadata["reason"] == "no_escape"
        assert result.state.current_az == state.current_az

    def test_escape_past_the_window_end_stops_the_loop(self):
        patch = self._pong_patch()
        ctx = _make_ctx(
            patches=[patch],
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T02:00:10",
            sun_safe=pose_blocker(180.0, 50.0),
        )
        state = _initial_state(ctx)
        sel = PhaseResult(state=state, blocks=[], selection=patch, best_az=115.0, best_el=50.0)

        result = SlewPhase().run(state, ctx, selection=sel)

        assert result.stop is True and result.blocks == []

    def test_escape_past_the_window_end_on_the_idle_path_stops_cleanly(self):
        """The idle path's escape can end past the window; the loop must stop, not raise.

        The pose is safe when the loop-top check runs, the opening
        calibrations then consume the window down to a sliver, and the
        selection idle finds the pose overtaken with too little time for
        the escape. That result carries ``stop`` and no selection, which
        the loop must not hand on to the slew phase.
        """
        from fyst_trajectories.overhead import generate_timeline

        start = Time("2026-06-15T02:00:00", scale="utc")
        overtaken_at = start + TimeDelta(1000.0, format="sec")

        def sun_safe(az, el, t):
            if t < overtaken_at:
                return True
            return not (abs(float(az) - 180.0) < 5.0 and abs(float(el) - 50.0) < 5.0)

        # The four opening calibrations take 1080 s; 20 s then remain,
        # less than the escape's 35 s.
        timeline = generate_timeline(
            patches=[],
            site=get_fyst_site(),
            start_time=start.isot,
            end_time=(start + TimeDelta(1100.0, format="sec")).isot,
            sun_safe=sun_safe,
        )

        assert timeline.blocks
        assert all(b.t_stop.unix <= timeline.end_time.unix + 1e-6 for b in timeline.blocks)
        assert not any(b.patch_name == "sun_escape" for b in timeline.blocks)
        assert timeline.validate() == []

    def test_slew_is_refused_when_the_goal_will_not_hold_for_a_tick(self):
        """A pose the next tick would escape from is not slewed to at all.

        Without the hold the wrap choice sees only the slew instant, so
        the loop would command a pose, find it overtaken one tick later
        and escape again: three round trips and no science. The slew
        phase asks for one scheduler tick of dwell instead, and refuses.
        """
        patch = self._pong_patch()
        start = Time("2026-06-15T02:00:00", scale="utc")
        closes_at = start + TimeDelta(120.0, format="sec")

        def sun_safe(az, el, t):
            # The parked pose stays clear; the science pose closes after
            # two minutes, well inside the 300 s tick.
            if abs(float(az) - 180.0) < 5.0 and abs(float(el) - 50.0) < 5.0:
                return True
            return bool(t < closes_at)

        ctx = _make_ctx(patches=[patch], sun_safe=sun_safe, time_step=300.0)
        state = _initial_state(ctx)
        sel = PhaseResult(state=state, blocks=[], selection=patch, best_az=115.0, best_el=50.0)

        result = SlewPhase().run(state, ctx, selection=sel)

        (block,) = result.blocks
        assert str(block.block_type) == "idle"
        assert block.metadata["reason"] == "sun_point"
        assert result.skip_to_next_iter is True
        assert (result.state.current_az, result.state.current_el) == (180.0, 50.0)

    def test_one_pose_and_time_is_searched_once(self, monkeypatch):
        """The three call sites per tick share one search, and none is dropped.

        The loop top, the idle emitter and the slew phase all have to ask,
        because the calibration phase can advance the clock inside a tick;
        the context's memo makes the repeats free instead. On this night
        every call is a duplicate of the loop top's, so the search count
        must equal the number of distinct (pose, time) questions.
        """
        from fyst_trajectories.overhead import _moves, generate_timeline

        seen = []
        original = _moves.plan_escape

        def counting(az, el, t, site, **kwargs):
            seen.append((round(float(az), 6), round(float(el), 6), round(t.unix, 6)))
            return original(az, el, t, site, **kwargs)

        monkeypatch.setattr(_moves, "plan_escape", counting)

        timeline = generate_timeline(
            patches=[self._pong_patch()],
            site=get_fyst_site(),
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T04:00:00",
        )

        assert timeline.blocks
        assert seen  # the check runs at all
        assert len(seen) == len(set(seen))

    def test_night_escapes_once_then_observes(self):
        """The loop escapes the overtaken bootstrap pose and the night proceeds normally."""
        from fyst_trajectories.overhead import generate_timeline

        patch = self._pong_patch()
        clear_at = Time("2026-06-15T08:35:00", scale="utc")
        timeline = generate_timeline(
            patches=[patch],
            site=get_fyst_site(),
            start_time="2026-06-15T07:50:00",
            end_time="2026-06-15T09:30:00",
            sun_safe=pose_blocker(180.0, 50.0, until=clear_at),
        )

        escapes = [b for b in timeline.blocks if b.patch_name == "sun_escape"]
        assert len(escapes) == 1
        assert escapes[0].t_start.unix == timeline.start_time.unix
        assert not any(b.metadata.get("reason") for b in timeline.blocks)
        kinds = [str(b.block_type) for b in timeline.blocks]
        assert "science" in kinds
        assert timeline.validate() == []


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
