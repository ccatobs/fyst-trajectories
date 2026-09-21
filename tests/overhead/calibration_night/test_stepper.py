"""Tests for the four step functions and the driver loop.

The loop is exercised through the ``visit_planner`` seam with prepared
plans, so no test here solves a pass except the one real-visit class,
which plans a single Saturn visit on the 2026-09-11 night.
"""

import json
import warnings

import pytest
from _sun_stubs import pose_blocker
from astropy.time import Time, TimeDelta

from fyst_trajectories import get_fyst_site
from fyst_trajectories.overhead import (
    BOOTSTRAP_POSE,
    CalibrationNightPolicy,
    CalibrationType,
    DeferralReason,
    ElevationBin,
    NightContext,
    NightState,
    ObservingTimeline,
    ScanOverrides,
    ScanParameterTable,
    ScriptedSelection,
    TimelineBlock,
    VisitPlan,
    advance_idle,
    commit_visit,
    list_candidates,
    plan_calibration_night,
    plan_visit,
    read_calibration_night_metadata,
    read_timeline,
    schedule_to_trajectories,
    validate_scan_params,
    write_timeline,
)
from fyst_trajectories.overhead.calibration_night.tables import table_for

NIGHT_START = "2026-09-11T06:30:00"
NIGHT_END = "2026-09-11T08:00:00"


def _parked_visit(state, ctx, body, overrides=None, *, duration=600.0):
    """Build a feasible plan with one parked pass block at the current pose (no sky)."""
    block = TimelineBlock.calibration(
        CalibrationType.PLANET_CAL,
        t_start=state.t,
        duration=duration,
        az=state.az,
        el=state.el,
        site=ctx.site,
        scan_index=state.scan_counter,
        target=body,
    )
    return VisitPlan(
        body=body,
        feasible=True,
        reason=None,
        transition=None,
        blocks=(block,),
        warnings=(),
        passes=(),
    )


def _refusal(reason):
    def planner(state, ctx, body, overrides=None):
        return VisitPlan(
            body=body, feasible=False, reason=reason, transition=None, blocks=(), warnings=()
        )

    return planner


class _RefuseOnce:
    """Refuse the first call for a body with ``reason``, then plan parked visits."""

    def __init__(self, reason):
        self.reason = reason
        self.refused = set()

    def __call__(self, state, ctx, body, overrides=None):
        if body not in self.refused:
            self.refused.add(body)
            return _refusal(self.reason)(state, ctx, body, overrides)
        return _parked_visit(state, ctx, body, overrides)


@pytest.fixture(scope="module")
def site():
    return get_fyst_site()


class TestDriverThroughTheSeam:
    """The loop over the four steps, with prepared plans."""

    def test_feasible_plans_fill_the_night(self, site):
        ctx = NightContext.build(
            ["saturn"], site, NIGHT_START, NIGHT_END, visit_planner=_parked_visit
        )
        timeline = _run(ctx)
        passes = [b for b in timeline.blocks if b.scan_type == "planet_cal"]
        assert len(passes) == 9  # the 5400 s window divides exactly into 600 s parked passes
        assert all(b.metadata["target"] == "saturn" for b in passes)
        assert timeline.validate() == []
        meta = read_calibration_night_metadata(timeline)
        assert meta["deferrals"] == [] and meta["drops"] == []
        assert meta["start_pose"] == list(BOOTSTRAP_POSE)

    def test_sun_refusal_defers_then_retries(self, site):
        policy = CalibrationNightPolicy(retry_after_seconds=300.0, time_step=300.0)
        ctx = NightContext.build(
            ["saturn"],
            site,
            NIGHT_START,
            NIGHT_END,
            policy=policy,
            visit_planner=_RefuseOnce(DeferralReason.SUN_POINT),
        )
        timeline = _run(ctx)
        first, second = timeline.blocks[:2]
        assert first.scan_type == "idle" and first.metadata["reason"] == "nothing_available"
        assert first.duration == pytest.approx(300.0)
        assert second.scan_type == "planet_cal"
        meta = read_calibration_night_metadata(timeline)
        assert meta["deferrals"] == [
            {"body": "saturn", "at": first.t_start.iso, "reason": "sun_point"}
        ]

    def test_geometry_refusal_drops_for_the_night(self, site):
        ctx = NightContext.build(
            ["saturn"], site, NIGHT_START, NIGHT_END, visit_planner=_refusal(DeferralReason.NO_WRAP)
        )
        timeline = _run(ctx)
        assert {b.scan_type for b in timeline.blocks} == {"idle"}
        meta = read_calibration_night_metadata(timeline)
        assert [d["reason"] for d in meta["drops"]] == ["no_wrap"]
        state = NightState.initial(ctx.start_time)
        state = commit_visit(state, _refusal(DeferralReason.NO_WRAP)(state, ctx, "saturn"))
        assert list_candidates(state, ctx)[0].reason is DeferralReason.NO_WRAP

    def test_scripted_entry_waits_then_is_set_aside(self, site):
        policy = CalibrationNightPolicy(max_wait_seconds=600.0, time_step=300.0)
        ctx = NightContext.build(
            ["saturn", "jupiter"],
            site,
            NIGHT_START,
            NIGHT_END,
            policy=policy,
            visit_planner=_parked_visit,
        )
        rule = ScriptedSelection(["jupiter", "saturn"])
        timeline = _run(ctx, rule)
        kinds = [b.scan_type for b in timeline.blocks[:3]]
        assert kinds[:2] == ["idle", "idle"]
        assert timeline.blocks[0].metadata["reason"] == "script_waiting"
        assert kinds[2] == "planet_cal" and timeline.blocks[2].metadata["target"] == "saturn"
        meta = read_calibration_night_metadata(timeline)
        assert meta["unplaced"] == [{"body": "jupiter", "overrides": {}}]
        assert meta["selection"] == "ScriptedSelection"

    def test_selection_must_return_a_candidate(self, site):
        ctx = NightContext.build(
            ["saturn"], site, NIGHT_START, NIGHT_END, visit_planner=_parked_visit
        )

        def rogue(candidates, state):
            return "mars", ScanOverrides()

        with pytest.raises(ValueError, match="not a candidate"):
            _run(ctx, rogue)

    def test_two_runs_are_identical(self, site):
        ctx = NightContext.build(
            ["saturn"], site, NIGHT_START, NIGHT_END, visit_planner=_parked_visit
        )
        a, b = _run(ctx), _run(ctx)
        assert [(x.t_start.iso, x.t_stop.iso, x.az_start, x.metadata) for x in a.blocks] == [
            (x.t_start.iso, x.t_stop.iso, x.az_start, x.metadata) for x in b.blocks
        ]
        assert a.metadata == b.metadata

    def test_sun_never_sets_gives_an_empty_night(self, site):
        timeline = plan_calibration_night(
            ["saturn"], site, "2026-09-11T14:00:00", "2026-09-11T18:00:00"
        )
        assert timeline.blocks == []
        assert read_calibration_night_metadata(timeline)["usable_interval"] is None


def _run(ctx, rule=None):
    """Drive the loop on a prepared context, the way plan_calibration_night does."""
    from fyst_trajectories.overhead.calibration_night.night import _run_night

    return _run_night(ctx, rule)


class TestContextInputs:
    """What NightContext.build accepts, and what it refuses."""

    def test_a_repeated_target_is_refused(self, site):
        """A body named twice is an error, not two bodies.

        The night revisits its targets under the selection rule, so a
        duplicate buys nothing; it would cost the tick an extra ephemeris
        solve per copy and make the night's own summary report the body's
        passes and minutes once per copy.
        """
        with pytest.raises(ValueError, match="must not repeat"):
            NightContext.build(["saturn", "Saturn"], site, NIGHT_START, NIGHT_END)

    def test_a_body_with_no_ephemeris_is_refused_at_build(self, site):
        """The name is checked here, not at the first tick.

        A name the ephemeris cannot resolve would otherwise reach the
        coordinate layer from inside the loop, after the whole context
        has been built and the night has started, because the shared
        ``default`` table matches any name.
        """
        with pytest.raises(ValueError, match="no ephemeris"):
            NightContext.build(["not_a_planet"], site, NIGHT_START, NIGHT_END)

    def test_a_capitalised_table_key_is_found(self, site):
        """A caller's table keyed ``"Uranus"`` is the one Uranus is planned on.

        Lookups lower-case the body name, so a mapping stored verbatim
        would never match a capitalised key: it would fall through to the
        shared ``"default"`` table, planning a different azimuth throw
        with no warning anywhere in the night.
        """
        uranus = ScanParameterTable([ElevationBin(30.0, 40.0, 9.99, 900.0)])
        shared = ScanParameterTable([ElevationBin(30.0, 40.0, 1.11, 600.0)])
        ctx = NightContext.build(
            ["uranus"],
            site,
            NIGHT_START,
            NIGHT_END,
            tables={"Uranus": uranus, "default": shared},
        )
        assert table_for(ctx.tables, "uranus") is uranus


class TestStateBookkeeping:
    """commit_visit and advance_idle in isolation."""

    def test_commit_feasible_updates_pose_time_and_cadences(self, site):
        ctx = NightContext.build(["saturn"], site, NIGHT_START, NIGHT_END)
        state = NightState.initial(ctx.start_time, (100.0, 40.0))
        retune = TimelineBlock.calibration(
            CalibrationType.RETUNE, state.t, 300.0, 120.0, 45.0, site, 0
        )
        pass_block = TimelineBlock.calibration(
            CalibrationType.PLANET_CAL,
            retune.t_stop,
            600.0,
            118.0,
            45.0,
            site,
            0,
            target="saturn",
            az_end=122.0,
        )
        plan = VisitPlan("saturn", True, None, None, (retune, pass_block), (), ())
        deferred_before = state.advanced(deferred={"saturn": (state.t, DeferralReason.SUN_POINT)})
        after = commit_visit(deferred_before, plan)
        assert after.t == pass_block.t_stop
        assert (after.az, after.el) == (122.0, 45.0)
        assert after.cal_state.last_retune == retune.t_stop
        assert after.cal_state.last_planet_cal == pass_block.t_stop
        assert "saturn" not in after.deferred
        assert after.script_index == 1

    def test_commit_infeasible_records_retry_time(self, site):
        ctx = NightContext.build(["saturn"], site, NIGHT_START, NIGHT_END)
        state = NightState.initial(ctx.start_time)
        plan = VisitPlan("saturn", False, DeferralReason.CROSSING_TOO_SLOW, None, (), ())
        after = commit_visit(state, plan, retry_after=450.0)
        retry_at, reason = after.deferred["saturn"]
        assert (retry_at - state.t).to_value("s") == pytest.approx(450.0)
        assert reason is DeferralReason.CROSSING_TOO_SLOW
        assert after.t == state.t

    def test_advance_idle_clips_to_the_end_and_labels_the_block(self, site):
        ctx = NightContext.build(["saturn"], site, NIGHT_START, NIGHT_END)
        state = NightState.initial(ctx.end_time - TimeDelta(100.0, format="sec"))
        after = advance_idle(state, ctx, 300.0, DeferralReason.NOTHING_AVAILABLE)
        assert after.blocks[-1].duration == pytest.approx(100.0)
        assert after.blocks[-1].metadata["reason"] == "nothing_available"
        assert advance_idle(after, ctx, 300.0, DeferralReason.NOTHING_AVAILABLE) is after


@pytest.fixture(scope="module")
def saturn_visit(site):
    """One real Saturn visit from the bootstrap pose at the start of the night."""
    ctx = NightContext.build(["saturn", "jupiter", "uranus"], site, NIGHT_START, NIGHT_END)
    state = NightState.initial(ctx.start_time)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = plan_visit(state, ctx, "saturn")
    return ctx, state, plan


class TestRealVisit:
    """One planned visit: block structure, metadata, rebuild."""

    def test_block_structure(self, saturn_visit):
        ctx, state, plan = saturn_visit
        assert plan.feasible, plan.summary
        kinds = [b.scan_type for b in plan.blocks]
        assert kinds[0] == "slew"
        assert kinds[-1] == "planet_cal"
        operations = [b.metadata.get("operation", b.scan_type) for b in plan.blocks[1:-1]]
        assert operations[:3] == ["find_detectors", "skydip", "retune"]
        # Every detector operation sits at the target pose, after the slew.
        for block in plan.blocks[1:-1]:
            if block.scan_type != "idle":
                assert block.az_start == pytest.approx(plan.transition.az_to)
                assert block.elevation == pytest.approx(plan.blocks[-1].elevation)
        assert plan.transition.path == "direct"
        assert plan.blocks[0].az_start == state.az

    def test_pass_metadata_is_relative_and_round_trippable(self, saturn_visit):
        ctx, _, plan = saturn_visit
        block = plan.blocks[-1]
        meta = block.metadata
        assert {
            "cal_type",
            "target",
            "t0_scan",
            "scan_params",
            "requested",
            "applied",
            "solved",
            "science_fraction",
            "n_legs",
            "module_crossings",
            "transition",
        } <= set(meta)
        params = meta["scan_params"]
        assert "window" not in params
        assert params["body"] == "saturn" and params["mode"] in ("rising", "setting")
        assert params["az_speed"] == 1.5 and params["az_accel"] == 1.5
        validate_scan_params(params, "source_ces")
        json.dumps(meta)
        assert meta["requested"]["az_throw"] == pytest.approx(params["az_throw"])
        assert meta["solved"]["crossing_seconds"] > 0.0
        assert 0.3 < meta["science_fraction"] < 0.8
        assert meta["transition"]["path"] == "direct"
        assert meta["module_crossings"]["c"] > 0.0
        assert meta["t0_scan"] == block.t_start.iso

    def test_pass_dict_requests_no_boresight_rotation(self, saturn_visit):
        """The dispatch dict repeats the request, and the planner requests no rotation.

        The kernel resolves an unset rotation to 0.0, but an execution layer
        that accepts only an uncommanded rotator tells 0.0 from None, so the
        dict carries the request rather than the resolved value. The rebuild
        test below proves a None still reconstructs the pass.
        """
        _, _, plan = saturn_visit
        params = plan.blocks[-1].metadata["scan_params"]
        assert "boresight_rot" in params
        assert params["boresight_rot"] is None

    def test_commit_moves_the_pose_to_the_pass_end(self, saturn_visit):
        """The pose handed forward is where the drag stopped, not the envelope.

        A source-CES drag ends on whichever leg endpoint its last
        turnaround left it on, generally neither azimuth bound, so
        ``az_end`` (the envelope maximum) is the wrong pose to price and
        Sun-check the next move from. The block records the real one in
        ``az_final`` and ``commit_visit`` reads it through
        ``end_pose_az``.
        """
        ctx, state, plan = saturn_visit
        after = commit_visit(state, plan)
        last = plan.blocks[-1]
        wrap_shift = plan.transition.az_to - float(plan.passes[0].trajectory.az[0])
        true_end_az = float(plan.passes[-1].trajectory.az[-1]) + wrap_shift
        assert last.az_final == pytest.approx(true_end_az)
        assert last.end_pose_az == last.az_final
        # The envelope maximum is several degrees away, so the two poses
        # are genuinely different and this check discriminates.
        assert last.az_end > last.az_final + 1.0
        assert (after.az, after.el) == (last.az_final, last.elevation)
        assert after.scan_counter == 1
        assert after.cal_state.last_retune is not None

    def test_pass_rebuilds_from_its_relative_dict(self, saturn_visit):
        ctx, state, plan = saturn_visit
        timeline = ObservingTimeline(
            blocks=list(plan.blocks),
            site=ctx.site,
            start_time=state.t,
            end_time=plan.blocks[-1].t_stop,
            overhead_model=ctx.overhead_model,
            calibration_policy=ctx.calibration_policy,
        )
        assert timeline.validate() == []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pairs = schedule_to_trajectories(timeline, science_only=False)
        assert len(pairs) == 1
        block, rebuilt = pairs[0]
        assert abs((rebuilt.trajectory.start_time - block.t_start).to_value("s")) < 2.0
        assert rebuilt.computed_params["az_throw"] == pytest.approx(
            block.metadata["applied"]["az_throw"], abs=1e-6
        )
        assert rebuilt.computed_params["az_speed"] == 1.5

    def test_unknown_body_raises(self, saturn_visit):
        ctx, state, _ = saturn_visit
        with pytest.raises(ValueError, match="not one of the night's targets"):
            plan_visit(state, ctx, "mars")

    def test_body_below_the_band_is_infeasible_not_raised(self, saturn_visit):
        ctx, state, _ = saturn_visit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plan = plan_visit(state, ctx, "jupiter")
        assert not plan.feasible
        assert plan.reason in (DeferralReason.UNPLANNABLE, DeferralReason.BELOW_BAND)
        assert plan.blocks == ()

    def test_overrides_win_over_policy_and_table(self, saturn_visit):
        ctx, state, _ = saturn_visit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plan = plan_visit(state, ctx, "saturn", ScanOverrides(az_speed=0.8, az_throw=3.0))
        assert plan.feasible, plan.summary
        meta = plan.blocks[-1].metadata
        assert meta["requested"]["az_speed"] == 0.8 and meta["requested"]["az_throw"] == 3.0
        assert meta["applied"]["az_speed"] == 0.8
        assert meta["scan_params"]["az_throw"] == 3.0

    def test_table_dwell_longer_than_the_crossing_is_reported_not_applied(self, site):
        policy = CalibrationNightPolicy(use_table_dwell=True)
        ctx = NightContext.build(["uranus"], site, NIGHT_START, NIGHT_END, policy=policy)
        state = NightState.initial(ctx.start_time)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plan = plan_visit(state, ctx, "uranus")
        assert plan.feasible, plan.summary
        assert any("exceeds this crossing" in w for w in plan.warnings)
        assert "dwell" not in plan.blocks[-1].metadata["scan_params"]


class TestMultiPassVisit:
    """A three-pass visit: the poses the inter-pass idles record.

    The passes step in elevation by design, and each drag ends at its own
    azimuth, so every gap idle is checked against the pass trajectories
    themselves rather than against ``validate()``: the validator follows
    the recorded blocks, so a pose wrong in both the pass block and the
    idle would pass it.
    """

    @pytest.fixture(scope="class")
    def three_pass_visit(self, site):
        policy = CalibrationNightPolicy(n_passes=3)
        ctx = NightContext.build(
            ["saturn"], site, NIGHT_START, "2026-09-11T09:30:00", policy=policy
        )
        state = NightState.initial(ctx.start_time)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plan = plan_visit(state, ctx, "saturn")
        return ctx, state, plan

    def test_pass_blocks_record_their_own_end_pose(self, three_pass_visit):
        _, _, plan = three_pass_visit
        assert plan.feasible, plan.summary
        assert len(plan.passes) == 3
        wrap_shift = plan.transition.az_to - float(plan.passes[0].trajectory.az[0])
        pass_blocks = [b for b in plan.blocks if b.scan_type == "planet_cal"]
        assert len(pass_blocks) == 3
        for block, p in zip(pass_blocks, plan.passes):
            assert block.az_final == pytest.approx(float(p.trajectory.az[-1]) + wrap_shift)
            assert block.elevation == pytest.approx(float(p.computed_params["el_bore"]))
        # The elevations really do step, so the pose checks below are not
        # vacuous, and the drag never stops at the envelope maximum.
        el_bores = [b.elevation for b in pass_blocks]
        assert len({round(e, 6) for e in el_bores}) == 3
        assert all(b.az_end > b.az_final + 1.0 for b in pass_blocks)

    def test_gap_idles_park_at_the_previous_pass_pose(self, three_pass_visit):
        ctx, state, plan = three_pass_visit
        idles = [(i, b) for i, b in enumerate(plan.blocks) if b.scan_type == "idle"]
        assert idles
        after_a_pass = 0
        for i, idle in idles:
            previous = plan.blocks[i - 1]
            assert idle.az_start == pytest.approx(previous.end_pose_az, abs=1e-9)
            assert idle.elevation == pytest.approx(previous.elevation, abs=1e-9)
            if previous.scan_type == "planet_cal":
                after_a_pass += 1
        assert after_a_pass >= 1

    def test_timeline_validates_and_round_trips_the_end_pose(self, three_pass_visit, tmp_path):
        ctx, state, plan = three_pass_visit
        timeline = ObservingTimeline(
            blocks=list(plan.blocks),
            site=ctx.site,
            start_time=state.t,
            end_time=plan.blocks[-1].t_stop,
            overhead_model=ctx.overhead_model,
            calibration_policy=ctx.calibration_policy,
        )
        assert timeline.validate() == []
        path = tmp_path / "three_pass.ecsv"
        write_timeline(timeline, path)
        reloaded = read_timeline(path)
        assert reloaded.validate() == []
        for before, after in zip(timeline.blocks, reloaded.blocks):
            if before.az_final is None:
                assert after.az_final is None
            else:
                assert after.az_final == pytest.approx(before.az_final)


class TestCandidates:
    """list_candidates at one instant of the night (ephemeris only)."""

    def test_reasons_and_estimates(self, site):
        ctx = NightContext.build(["jupiter", "saturn", "uranus"], site, NIGHT_START, NIGHT_END)
        state = NightState.initial(Time(NIGHT_START, scale="utc"), (100.0, 40.0))
        by_body = {c.body: c for c in list_candidates(state, ctx)}
        assert by_body["jupiter"].reason is DeferralReason.BELOW_BAND
        assert by_body["jupiter"].az_throw is None
        assert by_body["saturn"].available and by_body["saturn"].el_bore_estimate > 55.0
        assert by_body["saturn"].bin is None  # above the shared table, throw extrapolated
        assert by_body["saturn"].az_throw > 3.11
        assert by_body["uranus"].available and by_body["uranus"].bin is not None
        assert 0.0 < by_body["uranus"].time_left_in_band <= 5400.0

    def test_moon_check_is_opt_in(self, site):
        policy = CalibrationNightPolicy(moon_min_separation=10.0)
        ctx = NightContext.build(["saturn"], site, NIGHT_START, NIGHT_END, policy=policy)
        state = NightState.initial(Time(NIGHT_START, scale="utc"))
        (candidate,) = list_candidates(state, ctx)
        assert candidate.moon_separation is not None and candidate.moon_separation > 0.0


class TestSunEscape:
    """A pose the zone has overtaken is moved out before the planner idles or slews."""

    def test_advance_idle_escapes_then_idles_at_the_new_pose(self, site):
        ctx = NightContext.build(
            ["saturn"], site, NIGHT_START, NIGHT_END, sun_safe=pose_blocker(*BOOTSTRAP_POSE)
        )
        state = NightState.initial(ctx.start_time)

        after = advance_idle(state, ctx, 600.0, DeferralReason.NOTHING_AVAILABLE)

        escape, idle = after.blocks
        assert escape.scan_type == "slew" and escape.patch_name == "sun_escape"
        assert escape.az_start == BOOTSTRAP_POSE[0]
        assert idle.scan_type == "idle" and idle.metadata["reason"] == "nothing_available"
        assert (idle.az_start, idle.elevation) == (escape.az_end, escape.elevation)
        assert escape.duration + idle.duration == pytest.approx(600.0)
        assert (after.az, after.el) == (escape.az_end, escape.elevation)
        assert after.t.unix == pytest.approx(ctx.start_time.unix + 600.0)

    def test_advance_idle_labels_a_trapped_pose(self, site):
        ctx = NightContext.build(
            ["saturn"], site, NIGHT_START, NIGHT_END, sun_safe=lambda az, el, t: False
        )
        state = NightState.initial(ctx.start_time)

        after = advance_idle(state, ctx, 600.0, DeferralReason.NOTHING_AVAILABLE)

        (idle,) = after.blocks
        assert idle.metadata["reason"] == "no_escape"
        assert (after.az, after.el) == BOOTSTRAP_POSE

    def test_advance_idle_labels_an_escape_that_does_not_fit(self, site):
        """Too little night left for the move: the idle says so, it does not lie.

        The escape exists and is safe, but it is longer than what is left
        of the window, so the telescope stays inside the zone.
        """
        ctx = NightContext.build(
            ["saturn"], site, NIGHT_START, NIGHT_END, sun_safe=pose_blocker(*BOOTSTRAP_POSE)
        )
        state = NightState.initial(ctx.end_time - TimeDelta(1.0, format="sec"))

        after = advance_idle(state, ctx, 600.0, DeferralReason.NOTHING_AVAILABLE)

        (idle,) = after.blocks
        assert idle.scan_type == "idle"
        assert idle.metadata["reason"] == "no_escape"
        assert (after.az, after.el) == BOOTSTRAP_POSE
        assert idle.duration == pytest.approx(1.0)

    def test_plan_visit_escapes_first_and_records_it(self, site):
        ctx = NightContext.build(
            ["saturn"], site, NIGHT_START, NIGHT_END, sun_safe=pose_blocker(*BOOTSTRAP_POSE)
        )
        state = NightState.initial(ctx.start_time)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plan = plan_visit(state, ctx, "saturn")

        assert plan.feasible, plan.summary
        escape, slew = plan.blocks[:2]
        assert escape.patch_name == "sun_escape" and escape.az_start == state.az
        assert slew.patch_name == "slew_to_saturn"
        assert slew.az_start == escape.az_end
        assert slew.t_start.unix == pytest.approx(escape.t_stop.unix)
        record = plan.blocks[-1].metadata["transition"]
        assert record["escape_via"] == [escape.az_end, escape.elevation]
        assert record["path"] == "direct"

    def test_plan_visit_from_a_trapped_pose_is_infeasible_not_raised(self, site):
        ctx = NightContext.build(
            ["saturn"], site, NIGHT_START, NIGHT_END, sun_safe=lambda az, el, t: False
        )
        state = NightState.initial(ctx.start_time)

        plan = plan_visit(state, ctx, "saturn")

        assert not plan.feasible
        assert plan.reason is DeferralReason.NO_ESCAPE

    def test_one_state_is_searched_once_across_the_candidate_bodies(self, site, monkeypatch):
        """Every body offered in one tick asks the same question; it is solved once.

        An infeasible visit leaves the state where it was, so the driver
        walks on to the next body from the identical pose and time, and
        the idle after them asks a third time. Each search is a Sun
        ephemeris solve, so the context memoizes them.
        """
        from fyst_trajectories.overhead import _moves

        seen = []
        original = _moves.plan_escape

        def counting(az, el, t, site_, **kwargs):
            seen.append((round(float(az), 6), round(float(el), 6), round(t.unix, 6)))
            return original(az, el, t, site_, **kwargs)

        monkeypatch.setattr(_moves, "plan_escape", counting)

        ctx = NightContext.build(
            ["saturn", "uranus"],
            site,
            NIGHT_START,
            NIGHT_END,
            sun_safe=lambda az, el, t: False,
        )
        state = NightState.initial(ctx.start_time)

        assert plan_visit(state, ctx, "saturn").reason is DeferralReason.NO_ESCAPE
        assert plan_visit(state, ctx, "uranus").reason is DeferralReason.NO_ESCAPE
        advance_idle(state, ctx, 600.0, DeferralReason.NOTHING_AVAILABLE)

        assert len(seen) == 1


class TestSweepSunSafeFailsClosed:
    """An empty trajectory is not "safe"; it is nothing to screen.

    The dispatch-time wrap gate already refuses an empty time array with an
    explicit comment about the vacuous ``all(...)``. The sweep's own
    docstring says it fails closed, so it must not answer ``True`` on the
    batch path (numpy's empty ``.all()``) nor raise ``IndexError`` on the
    per-sample path.
    """

    @staticmethod
    def _empty():
        import numpy as np
        from astropy.time import Time

        return np.array([]), np.array([]), Time([], format="jd", scale="utc")

    def test_empty_trajectory_is_unsafe_with_a_batch_model(self):
        import numpy as np

        from fyst_trajectories.overhead.calibration_night.helpers import sweep_sun_safe

        class _AlwaysClear:
            def __call__(self, az, el, t):
                return True

            def batch(self, az, el, t):
                return np.ones(np.shape(az), dtype=bool)

        assert sweep_sun_safe(_AlwaysClear(), *self._empty()) is False

    def test_empty_trajectory_is_unsafe_with_a_bare_predicate(self):
        from fyst_trajectories.overhead.calibration_night.helpers import sweep_sun_safe

        assert sweep_sun_safe(lambda az, el, t: True, *self._empty()) is False
