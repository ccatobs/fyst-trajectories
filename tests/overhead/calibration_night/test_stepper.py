"""Tests for the four step functions and the driver loop.

The loop is exercised through the ``visit_planner`` seam with prepared
plans; the visit classes plan single visits on the 2026-09-11 night, and
the resume tests compare a state replayed from a planned night with the
states the driver produced while planning it.
"""

import dataclasses
import json
import logging
import math
import re
import warnings

import numpy as np
import pytest
from _sun_stubs import fake_sun_model, pose_blocker
from astropy.time import Time, TimeDelta

from fyst_trajectories import MODULE_FOV_RADIUS_DEG, PRIMECAM_MODULES, get_fyst_site
from fyst_trajectories.overhead import (
    BOOTSTRAP_POSE,
    BlockNotReconstructableError,
    CalibrationNightPolicy,
    CalibrationType,
    DeferralReason,
    ElevationBin,
    NightContext,
    NightState,
    ObservingTimeline,
    ScanOverrides,
    ScanParameterTable,
    ScanParamsSchemaError,
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
from fyst_trajectories.overhead.calibration_night import step
from fyst_trajectories.overhead.calibration_night.tables import table_for
from fyst_trajectories.overhead.simulation import _generate_trajectory_for_block

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
        set_aside_at = timeline.blocks[2].t_start.iso
        assert meta["unplaced"] == [{"body": "jupiter", "at": set_aside_at, "overrides": {}}]
        assert meta["selection"] == "ScriptedSelection"

    def test_selection_must_return_a_candidate(self, site):
        ctx = NightContext.build(
            ["saturn"], site, NIGHT_START, NIGHT_END, visit_planner=_parked_visit
        )

        def rogue(candidates, state):
            return "mars", ScanOverrides()

        with pytest.raises(ValueError, match="not a candidate"):
            _run(ctx, rogue)

    def test_selection_must_return_an_available_candidate(self, site):
        """Offering a body is not making it available: one below the band is refused."""
        ctx = NightContext.build(
            ["jupiter", "saturn"],
            site,
            "2026-09-10T23:30:00",
            "2026-09-11T01:00:00",
            visit_planner=_parked_visit,
        )
        assert not any(
            c.available for c in list_candidates(NightState.initial(ctx.start_time), ctx)
        )

        def first_offered(candidates, state):
            return candidates[0].body, ScanOverrides()

        with pytest.raises(ValueError, match="not a candidate available now"):
            _run(ctx, first_offered)

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


class TestPolicyFootprint:
    """The footprint tag is resolved when the policy is built, as a body name is by the context."""

    @pytest.mark.parametrize("tag", ["c,i1", "zzz", "IM1", "all", "", " c"])
    def test_a_tag_naming_no_module_is_refused_when_the_policy_is_built(self, tag):
        """The tag fails here, not at the first visit's footprint lookup.

        A tag no Prime-Cam module answers to would otherwise construct,
        pass ``NightContext.build`` and stop the night with a ``KeyError``
        from inside ``plan_visit``. The message names the tag and every
        module name the policy takes.
        """
        prefix = f"footprint: Unknown PrimeCam module '{tag}'. Available: "
        with pytest.raises(ValueError, match=re.escape(prefix)) as caught:
            CalibrationNightPolicy(footprint=tag)
        available = str(caught.value).split("Available: ", 1)[1].split(", ")
        assert {*PRIMECAM_MODULES, "im0"} <= set(available)

    @pytest.mark.parametrize("tag", [None, ["c", "i1"], 0])
    def test_a_footprint_that_is_not_a_string_is_refused(self, tag):
        """A malformed argument is a ``ValueError`` saying what is expected."""
        expected = f"footprint must be a str naming one Prime-Cam module, got {tag!r}"
        with pytest.raises(ValueError, match=re.escape(expected)):
            CalibrationNightPolicy(footprint=tag)

    @pytest.mark.parametrize("tag", ["c", "C", "center", "IM0", "i1", "I6"])
    def test_a_module_name_in_any_case_is_accepted(self, tag):
        assert CalibrationNightPolicy(footprint=tag).footprint == tag


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
        assert after.cal_state.last_retune == retune.t_start
        assert after.cal_state.last_planet_cal == pass_block.t_start
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


def _visit_timeline(ctx, state, blocks):
    """Return a timeline holding the blocks of one visit planned from ``state``."""
    blocks = list(blocks)
    return ObservingTimeline(
        blocks=blocks,
        site=ctx.site,
        start_time=state.t,
        end_time=blocks[-1].t_stop,
        overhead_model=ctx.overhead_model,
        calibration_policy=ctx.calibration_policy,
    )


def _without_search_start(blocks):
    """Return the blocks as a timeline written without ``search_start`` holds them."""
    return [
        dataclasses.replace(
            b, metadata={k: v for k, v in b.metadata.items() if k != "search_start"}
        )
        for b in blocks
    ]


def _assert_same_pass(rebuilt, planned):
    """Every sample, the start instant, the duration and every solved value, to the bit."""
    for name in ("times", "az", "el", "az_vel", "el_vel", "scan_flag"):
        assert np.array_equal(getattr(rebuilt.trajectory, name), getattr(planned.trajectory, name))
    got, want = rebuilt.trajectory.start_time, planned.trajectory.start_time
    assert (got.jd1, got.jd2) == (want.jd1, want.jd2)
    assert rebuilt.duration == planned.duration
    assert rebuilt.computed_params == planned.computed_params


def _rebuild_visit(ctx, state, plan, blocks=None):
    """Rebuild a visit's pass blocks, returning ``(block, rebuilt)`` pairs."""
    timeline = _visit_timeline(ctx, state, plan.blocks if blocks is None else blocks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return schedule_to_trajectories(timeline, science_only=False)


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
            "search_start",
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
        # The search start rides beside the dict, never in it: the dict is what
        # the execution layer and the dispatch sheet read.
        assert "window" not in params and "search_start" not in params
        assert params["body"] == "saturn" and params["mode"] in ("rising", "setting")
        # The policy's 1.5 deg/s and 1.0 deg/s^2 are commissioning values the
        # instrument team requested, not ratified limits; this pin moves with them.
        assert params["az_speed"] == 1.5 and params["az_accel"] == 1.0
        validate_scan_params(params, "source_ces")
        assert json.loads(json.dumps(meta)) == meta
        # The throw is solved from the footprint, so the dict asks for no
        # padding instead of naming a throw.
        assert "az_throw" not in params and params["az_padding"] == 0.0
        assert meta["applied"]["az_throw"] == meta["solved"]["az_throw"]
        assert meta["solved"]["crossing_seconds"] > 0.0
        assert 0.3 < meta["science_fraction"] < 0.8
        assert meta["transition"]["path"] == "direct"
        assert meta["module_crossings"]["c"] > 0.0
        assert meta["t0_scan"] == block.t_start.iso

    def test_pass_dict_requests_no_boresight_rotation(self, saturn_visit):
        """The dispatch dict repeats the request, and the planner requests no rotation.

        The kernel resolves an unset rotation to 0.0, but an execution layer
        that accepts only None tells 0.0 from None, so the dict carries the
        request rather than the resolved value. The rebuild test below proves
        a None still reconstructs the pass.
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
        """The rebuild repeats the kernel's search, so it is the planned pass to the bit."""
        ctx, state, plan = saturn_visit
        timeline = _visit_timeline(ctx, state, plan.blocks)
        assert timeline.validate() == []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pairs = schedule_to_trajectories(timeline, science_only=False)
        assert len(pairs) == 1
        block, rebuilt = pairs[0]
        _assert_same_pass(rebuilt, plan.passes[0])
        assert rebuilt.computed_params["az_throw"] == block.metadata["applied"]["az_throw"]
        assert rebuilt.computed_params["az_speed"] == 1.5

    def test_a_pass_without_search_start_rebuilds_around_its_bounds(self, saturn_visit):
        """A block without ``search_start`` re-solves inside its pass widened by 300 s.

        That rebuild lands near the planned pass, not on it: the kernel solves
        the throw again on another sampling grid, to its 1e-4 deg objective
        tolerance (5.5e-6 deg here, up to 1.15e-4 deg over a full night).
        """
        ctx, state, plan = saturn_visit
        timeline = _visit_timeline(ctx, state, _without_search_start(plan.blocks))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ((block, rebuilt),) = schedule_to_trajectories(timeline, science_only=False)
        assert abs((rebuilt.trajectory.start_time - block.t_start).to_value("s")) < 0.1
        assert rebuilt.computed_params["az_throw"] == pytest.approx(
            block.metadata["applied"]["az_throw"], abs=5e-4
        )
        assert not np.array_equal(rebuilt.trajectory.az, plan.passes[0].trajectory.az)

    def test_the_centre_module_by_its_im_label_is_not_warned(self, site):
        """``IM0`` names the centre module, so the visit records no off-centre advisory.

        The execution layer's centred-only check compares the dict's
        ``footprint`` as a string and accepts ``"c"`` and ``"center"``
        only, so the dict names the module ``"c"``, not ``"IM0"``.
        """
        policy = CalibrationNightPolicy(footprint="IM0")
        ctx = NightContext.build(["saturn"], site, NIGHT_START, NIGHT_END, policy=policy)
        state = NightState.initial(ctx.start_time)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plan = plan_visit(state, ctx, "saturn")
        assert plan.feasible, plan.summary
        assert not any("centred footprints only" in w for w in plan.warnings)
        assert plan.blocks[-1].metadata["scan_params"]["footprint"] == "c"

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


# The default Saturn visit of ``saturn_visit`` as the planner recorded it
# before the solved throw became the default (table throw, az_accel 1.5).
# Recorded by running that version, not derived.
_TABLE_THROW_VISIT = {
    "scan_params": {
        "body": "saturn",
        "footprint": "c",
        "el_bore": 61.480398064863735,
        "mode": "setting",
        "boresight_rot": None,
        "timestep": 0.1,
        "eta_offset_deg": 0.0,
        "pass_index": 0,
        "n_passes": 1,
        "az_speed": 1.5,
        "az_accel": 1.5,
        "az_throw": 4.502110539438211,
    },
    "requested": {"az_speed": 1.5, "az_accel": 1.5, "az_throw": 4.502110539438211},
    "applied": {
        "az_speed": 1.5,
        "az_accel": 1.5,
        "az_throw": 4.502110539438211,
        "dwell": 738.2082398912368,
    },
    "solved": {"az_throw": 4.502110539438211, "crossing_seconds": 736.0076316649648},
}


def _assert_same_record(got, want):
    """Keys and non-float values exactly, floats to a relative 1e-9."""
    assert set(got) == set(want)
    for key, value in want.items():
        if isinstance(value, float):
            assert got[key] == pytest.approx(value, rel=1e-9), key
        else:
            assert got[key] == value, key


class TestPassThrow:
    """Which azimuth throw a pass sweeps, and what its dispatch dict says about it.

    By default the kernel solves the throw from the footprint, with no
    padding; ``use_table_throw`` sweeps the table's width instead, and a
    per-visit ``az_throw`` override wins under either setting.
    """

    @pytest.mark.parametrize("margin", [0.0, 0.4])
    def test_the_default_sweeps_the_footprint_throw(self, site, monkeypatch, margin):
        """The solved throw is the footprint's azimuth extent at the pass elevation."""
        calls = _record_kernel_calls(monkeypatch)
        plan = _saturn_visit(site, policy=CalibrationNightPolicy(footprint_margin=margin))
        assert plan.feasible, plan.summary
        (call,) = calls
        assert call["az_padding"] == 0.0 and "az_throw" not in call
        meta = plan.blocks[-1].metadata
        params = meta["scan_params"]
        assert params["az_padding"] == 0.0 and "az_throw" not in params
        assert "az_throw" not in meta["requested"]
        validate_scan_params(params, "source_ces")
        extent = 2.0 * (MODULE_FOV_RADIUS_DEG + margin) / math.cos(math.radians(params["el_bore"]))
        assert meta["solved"]["az_throw"] == pytest.approx(extent, abs=0.01)
        assert meta["applied"]["az_throw"] == meta["solved"]["az_throw"]

    @pytest.mark.parametrize("body", ["saturn", "uranus"])
    def test_the_switch_requests_the_table_throw(self, site, monkeypatch, body):
        """Saturn sits above the table's top bin (extrapolated), Uranus inside it."""
        calls = _record_kernel_calls(monkeypatch)
        ctx = NightContext.build(
            ["saturn", "uranus"],
            site,
            NIGHT_START,
            NIGHT_END,
            policy=CalibrationNightPolicy(use_table_throw=True),
        )
        plan = plan_visit(NightState.initial(ctx.start_time), ctx, body)
        assert plan.feasible, plan.summary
        call = calls[0]
        table = table_for(ctx.tables, body)
        _, el_est = ctx.body_altaz(body, call["start_time"])
        assert el_est >= table.el_range[0]
        assert call["az_throw"] == table.az_throw_at(el_est)
        assert "az_padding" not in call
        meta = plan.blocks[-1].metadata
        assert meta["scan_params"]["az_throw"] == call["az_throw"]
        assert meta["requested"]["az_throw"] == call["az_throw"]
        assert "az_padding" not in meta["scan_params"]

    def test_below_the_table_the_switch_requests_no_throw(self, site, monkeypatch):
        """Below the table's floor the kernel's own padding applies, as before the switch."""
        calls = _record_kernel_calls(monkeypatch)
        high_floor = ScanParameterTable(
            (ElevationBin(35.0, 50.0, az_throw=3.0, dwell_reference=600.0),)
        )
        ctx = NightContext.build(
            ["saturn", "uranus"],
            site,
            NIGHT_START,
            NIGHT_END,
            policy=CalibrationNightPolicy(use_table_throw=True),
            tables={"default": high_floor},
        )
        plan = plan_visit(NightState.initial(ctx.start_time), ctx, "uranus")
        assert plan.feasible, plan.summary
        call = calls[0]
        _, el_est = ctx.body_altaz("uranus", call["start_time"])
        assert el_est < 35.0
        assert "az_throw" not in call and "az_padding" not in call
        params = plan.blocks[-1].metadata["scan_params"]
        assert "az_throw" not in params and "az_padding" not in params

    @pytest.mark.parametrize("use_table_throw", [False, True])
    def test_an_override_throw_wins_under_either_setting(self, site, monkeypatch, use_table_throw):
        calls = _record_kernel_calls(monkeypatch)
        plan = _saturn_visit(
            site,
            ScanOverrides(az_throw=3.0),
            policy=CalibrationNightPolicy(use_table_throw=use_table_throw),
        )
        assert plan.feasible, plan.summary
        assert calls[0]["az_throw"] == 3.0 and "az_padding" not in calls[0]
        meta = plan.blocks[-1].metadata
        assert meta["scan_params"]["az_throw"] == 3.0 and "az_padding" not in meta["scan_params"]
        assert meta["requested"]["az_throw"] == 3.0

    def test_a_rebuild_re_solves_the_footprint_throw(self, saturn_visit):
        """The dict carries no throw, so the rebuild solves it again on the planner's grid."""
        ctx, state, plan = saturn_visit
        timeline = _visit_timeline(ctx, state, plan.blocks)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ((block, rebuilt),) = schedule_to_trajectories(timeline, science_only=False)
        params = block.metadata["scan_params"]
        assert "az_throw" not in params and params["az_padding"] == 0.0
        # The rebuild repeats the planner's search, so the solve sees the same
        # samples and lands on the recorded throw exactly.
        assert rebuilt.computed_params["az_throw"] == block.metadata["solved"]["az_throw"]

    def test_the_switch_restores_the_previous_visit(self, site):
        """With the table throw and the old acceleration, the visit is the one recorded before."""
        ctx = NightContext.build(
            ["saturn", "jupiter", "uranus"],
            site,
            NIGHT_START,
            NIGHT_END,
            policy=CalibrationNightPolicy(use_table_throw=True, az_accel=1.5),
        )
        plan = plan_visit(NightState.initial(ctx.start_time), ctx, "saturn")
        assert plan.feasible, plan.summary
        meta = plan.blocks[-1].metadata
        for record, want in _TABLE_THROW_VISIT.items():
            _assert_same_record(meta[record], want)


@pytest.fixture(scope="class")
def three_pass_visit(site):
    policy = CalibrationNightPolicy(n_passes=3)
    ctx = NightContext.build(["saturn"], site, NIGHT_START, "2026-09-11T09:30:00", policy=policy)
    state = NightState.initial(ctx.start_time)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = plan_visit(state, ctx, "saturn")
    return ctx, state, plan


def _waits(plan):
    """Each wait before a later pass: its start, the pose it records, its end, the next start.

    A wait runs from the end of one pass block to the start of the next;
    the poses are in the visit's wrap, as the Sun gate sees them.
    """
    wrap_shift = plan.transition.az_to - float(plan.passes[0].trajectory.az[0])
    blocks = [b for b in plan.blocks if b.scan_type == "planet_cal"]
    waits = []
    for before, after, p in zip(blocks, blocks[1:], plan.passes[1:]):
        start = (float(p.trajectory.az[0]) + wrap_shift, float(p.trajectory.el[0]))
        waits.append((before.t_stop, (before.end_pose_az, before.elevation), after.t_start, start))
    return waits


def _strictly_inside(t, lo, hi):
    """Whether ``t`` lies inside ``(lo, hi)`` by more than a millisecond at each end."""
    return lo.unix + 1e-3 < t.unix < hi.unix - 1e-3


def _replan(ctx, state, predicate):
    """Re-plan the three-pass visit with a plain point predicate as the Sun model."""
    blocked = NightContext.build(
        ["saturn"],
        ctx.site,
        NIGHT_START,
        "2026-09-11T09:30:00",
        policy=ctx.policy,
        sun_safe=fake_sun_model(predicate, batch=False),
    )
    return plan_visit(state, blocked, "saturn")


class TestMultiPassVisit:
    """A three-pass visit: the poses the inter-pass idles record, and their Sun check.

    The passes step in elevation by design, and each drag ends at its own
    azimuth, so every gap idle is checked against the pass trajectories
    themselves rather than against ``validate()``: the validator follows
    the recorded blocks, so a pose wrong in both the pass block and the
    idle would pass it.
    """

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

    def test_every_pass_rebuilds_as_planned_through_ecsv(self, three_pass_visit, tmp_path):
        """Every pass of the visit was searched from one anchor, and each rebuilds to the bit.

        The later passes start well after the anchor, so a rebuild that opened
        its search at its own pass would sample another grid; the recorded
        start, read back from ECSV unchanged, repeats the visit's.
        """
        ctx, state, plan = three_pass_visit
        blocks = [b for b in plan.blocks if b.scan_type == "planet_cal"]
        records = {tuple(b.metadata["search_start"]) for b in blocks}
        assert len(records) == 1
        reloaded = _through_ecsv(_visit_timeline(ctx, state, plan.blocks), tmp_path)
        assert [
            b.metadata["search_start"] for b in reloaded.blocks if "search_start" in b.metadata
        ] == [b.metadata["search_start"] for b in blocks]
        for timeline in (_visit_timeline(ctx, state, plan.blocks), reloaded):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pairs = schedule_to_trajectories(timeline, science_only=False)
            assert len(pairs) == len(plan.passes) == 3
            for (_, rebuilt), planned in zip(pairs, plan.passes):
                _assert_same_pass(rebuilt, planned)

    def test_a_blocked_step_between_passes_is_refused(self, three_pass_visit):
        """The step from one pass's end to the next pass's start is swept as a slew.

        The blocker sits on the midpoint of the second step, and is unsafe
        only inside the second wait; its radius is under half the elevation
        step, so no pass and neither pose the telescope waits at meets it.
        The first wait is clear, so both waits are asked about.
        """
        ctx, state, plan = three_pass_visit
        waits = _waits(plan)
        t_end, end, t0, start = waits[1]
        radius = 0.4
        assert abs(end[1] - start[1]) / 2.0 > radius
        blocker = pose_blocker((end[0] + start[0]) / 2.0, (end[1] + start[1]) / 2.0, radius)
        asked = []

        def predicate(az, el, t):
            asked.append(t.unix)
            return not _strictly_inside(t, t_end, t0) or blocker(az, el, t)

        refused = _replan(ctx, state, predicate)
        assert refused.reason is DeferralReason.SUN_PATH
        assert refused.blocks == () and refused.transition is None
        for lo, _, hi, _ in waits:
            assert any(lo.unix + 1e-3 < u < hi.unix - 1e-3 for u in asked)

    def test_a_step_blocked_late_in_the_wait_is_refused(self, three_pass_visit):
        """The step is swept leaving at either end of the wait, as it can be made at any time.

        The blocker sits on the midpoint of the first step and is unsafe only
        in the last 30 s of the first wait, so a step swept leaving when the
        previous pass ends never meets it.
        """
        ctx, state, plan = three_pass_visit
        t_end, end, t0, start = _waits(plan)[0]
        blocker = pose_blocker((end[0] + start[0]) / 2.0, (end[1] + start[1]) / 2.0, 0.4)
        late = t0.unix - 30.0

        def predicate(az, el, t):
            return not (late < t.unix < t0.unix - 1e-3) or blocker(az, el, t)

        refused = _replan(ctx, state, predicate)
        assert refused.reason is DeferralReason.SUN_PATH

    @pytest.mark.parametrize("pose", ["previous_end", "next_start"])
    def test_a_blocked_wait_before_a_later_pass_is_refused(self, three_pass_visit, pose):
        """Both poses the telescope can wait at are held clear for the whole wait.

        The idle records the pose the previous pass left the telescope at,
        and the step to the next pass's start can be made at any time in the
        wait. The blocker is unsafe only inside the first wait, so the passes
        themselves, which end and start at these poses, never meet it.
        """
        ctx, state, plan = three_pass_visit
        t_end, end, t0, start = _waits(plan)[0]
        blocker = pose_blocker(*(end if pose == "previous_end" else start), radius=0.1)

        def predicate(az, el, t):
            return not _strictly_inside(t, t_end, t0) or blocker(az, el, t)

        refused = _replan(ctx, state, predicate)
        assert refused.reason is DeferralReason.SUN_POINT
        assert refused.blocks == () and refused.transition is None

    def test_the_wrap_holds_every_pass_near_an_azimuth_limit(self, site):
        """The wrap is chosen for the whole visit, not for its first pass.

        The three passes drift about 11 deg in azimuth. With the lower
        azimuth limit at -40 deg and the mount parked 5 deg inside it, the
        image nearest the mount holds the first pass but not the third, so
        only a wrap chosen for every pass keeps the visit inside the limits.
        """
        import dataclasses

        import numpy as np

        azimuth = dataclasses.replace(site.telescope_limits.azimuth, min=-40.0)
        limits = dataclasses.replace(site.telescope_limits, azimuth=azimuth)
        near_limit_site = dataclasses.replace(site, telescope_limits=limits)
        ctx = NightContext.build(
            ["saturn"],
            near_limit_site,
            NIGHT_START,
            "2026-09-11T09:30:00",
            policy=CalibrationNightPolicy(n_passes=3),
        )
        state = NightState.initial(ctx.start_time, (-35.0, 50.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plan = plan_visit(state, ctx, "saturn")

        assert plan.feasible, plan.summary
        assert len(plan.passes) == 3
        wrap_shift = plan.transition.az_to - float(plan.passes[0].trajectory.az[0])
        for p in plan.passes:
            az = np.asarray(p.trajectory.az) + wrap_shift
            assert azimuth.min <= az.min() and az.max() <= azimuth.max


def _kinds(warns):
    """Name each visit warning: the kernel's advisories by kind, the visit's own in full.

    The kernel's dynamics and Sun-zone advisories carry solved numbers, so
    they are named by their opening words; the order and the count are
    what show that every kernel run's advisories reached the record.
    """
    kinds = []
    for text in warns:
        if text.startswith("High elevation reduces on-sky azimuth speed"):
            kinds.append("speed")
        elif text.startswith("Trajectory azimuth acceleration"):
            kinds.append("accel")
        elif text.startswith("EXCLUSION ZONE"):
            kinds.append("sun_zone")
        else:
            kinds.append(text)
    return kinds


_KERNEL = ["speed"]
_DWELL_FALLBACK = (
    "the requested dwell of 5000 s exceeds this crossing; scanning the full crossing instead"
)


def _saturn_visit(site, overrides=None, **context):
    """Plan one Saturn visit from the bootstrap pose at the start of the night."""
    ctx = NightContext.build(["saturn", "uranus"], site, NIGHT_START, NIGHT_END, **context)
    return plan_visit(NightState.initial(ctx.start_time), ctx, "saturn", overrides)


def _record_kernel_calls(monkeypatch):
    """Record the keyword arguments of every kernel call the visit makes."""
    calls = []
    real = step.plan_source_ces_passes

    def recording(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(step, "plan_source_ces_passes", recording)
    return calls


def _record_transitions(monkeypatch, delays=()):
    """Record every transition the visit plans, adding ``delays[i]`` seconds to the i-th.

    A delay moves the arrival past the instant the transition's own Sun
    gate checked, which is how a test makes the slew late.
    """
    planned = []
    real = step.plan_transition

    def recording(*args, **kwargs):
        transition = real(*args, **kwargs)
        planned.append(transition)
        extra = delays[min(len(planned), len(delays)) - 1] if delays else 0.0
        if transition.safe and extra:
            return dataclasses.replace(transition, duration=transition.duration + extra)
        return transition

    monkeypatch.setattr(step, "plan_transition", recording)
    return planned


def _record_sweeps(monkeypatch):
    """Record the verdict of every pass sweep the Sun gate runs."""
    verdicts = []
    real = step.sweep_sun_safe

    def recording(*args, **kwargs):
        verdicts.append(real(*args, **kwargs))
        return verdicts[-1]

    monkeypatch.setattr(step, "sweep_sun_safe", recording)
    return verdicts


class TestPlanVisitRefusals:
    """Every refusal branch of plan_visit, driven through plan_visit itself.

    Each test asserts the reason and the warnings, so a split of the
    function that loses a branch's request state (the anchor, the dwell,
    the boresight elevation across the re-anchor) or a kernel run's
    advisories fails here. The visits are on 2026-09-11 from 06:30 UTC,
    where Saturn's default visit is feasible: the slew lands at 06:30:58,
    the detector operations end at 06:45:58 and the pass starts 30 s later.
    """

    def test_below_band(self, site):
        plan = _saturn_visit(site, policy=CalibrationNightPolicy(el_min=66.0))
        assert plan.reason is DeferralReason.BELOW_BAND
        assert _kinds(plan.warnings) == _KERNEL
        assert plan.blocks == () and plan.transition is None

    def test_above_band(self, site, monkeypatch):
        """The gate refuses a solved elevation above the limit.

        The kernel refuses that elevation itself (next test), so the gate
        is reached only when the two disagree; the stubbed kernel raises
        the solved elevation past the limit after a real solve.
        """
        ceiling = site.telescope_limits.elevation.max
        real = step.plan_source_ces_passes

        def too_high(**kwargs):
            first, *rest = real(**kwargs)
            params = {**first.computed_params, "el_bore": ceiling + 1.0}
            return [dataclasses.replace(first, computed_params=params), *rest]

        monkeypatch.setattr(step, "plan_source_ces_passes", too_high)
        plan = _saturn_visit(site)
        assert plan.reason is DeferralReason.ABOVE_BAND
        assert _kinds(plan.warnings) == _KERNEL

    def test_a_ceiling_below_the_body_is_unplannable(self, site):
        """A lowered elevation limit is refused by the kernel before the gate."""
        lim = site.telescope_limits
        low = dataclasses.replace(
            site,
            telescope_limits=dataclasses.replace(
                lim, elevation=dataclasses.replace(lim.elevation, max=55.0)
            ),
        )
        plan = _saturn_visit(low)
        assert plan.reason is DeferralReason.UNPLANNABLE
        (refusal,) = plan.warnings
        assert "would fall outside the telescope elevation limits" in refusal
        assert "ceiling 55.0 deg" in refusal

    def test_a_refused_re_solve_after_the_dwell_fallback_is_unplannable(self, site, monkeypatch):
        """The full-crossing re-solve after a dwell the crossing cannot hold is refused.

        The dwell is refused before any trajectory is built, so the full
        crossing is the first one the azimuth limits judge: with the limit
        at 338 deg, inside Saturn's crossing, the re-solve is refused by the
        trajectory bounds check.
        """
        lim = site.telescope_limits
        clipped = dataclasses.replace(
            site,
            telescope_limits=dataclasses.replace(
                lim, azimuth=dataclasses.replace(lim.azimuth, max=338.0)
            ),
        )
        calls = _record_kernel_calls(monkeypatch)
        planned = _record_transitions(monkeypatch)
        plan = _saturn_visit(clipped, ScanOverrides(dwell=5000.0))
        assert plan.reason is DeferralReason.UNPLANNABLE
        fallback, refusal = plan.warnings
        assert fallback == _DWELL_FALLBACK
        assert "exceeds limits" in refusal
        assert [call.get("dwell") for call in calls] == [5000.0, None]
        assert planned == [] and plan.transition is None and plan.blocks == ()

    def test_crossing_too_slow(self, site):
        plan = _saturn_visit(site, policy=CalibrationNightPolicy(max_pass_seconds=10.0))
        assert plan.reason is DeferralReason.CROSSING_TOO_SLOW
        assert _kinds(plan.warnings) == _KERNEL

    def test_no_pass_ending_before_the_night_ends_closes_the_window(self, site, monkeypatch):
        """From 07:45 Saturn's next crossing ends after the night does, at 08:00.

        The pass gate refuses the emptied list before any slew is planned,
        which tells this refusal from the late slew's.
        """
        ctx = NightContext.build(["saturn", "uranus"], site, NIGHT_START, NIGHT_END)
        real = step.plan_source_ces_passes
        solved = []

        def recording(**kwargs):
            passes = real(**kwargs)
            solved.append(passes)
            return passes

        monkeypatch.setattr(step, "plan_source_ces_passes", recording)
        planned = _record_transitions(monkeypatch)
        state = NightState.initial(Time("2026-09-11T07:45:00", scale="utc"))
        plan = plan_visit(state, ctx, "saturn")
        assert plan.reason is DeferralReason.WINDOW_CLOSED
        assert _kinds(plan.warnings) == []
        (passes,) = solved
        assert passes and all(
            Time(p.computed_params["t1_iso"], scale="utc") > ctx.end_time for p in passes
        )
        assert planned == [] and plan.transition is None and plan.blocks == ()

    def test_the_transition_refusal_is_the_reason(self, site):
        plan = _saturn_visit(site, slew_safe=lambda *_: False)
        assert plan.reason is DeferralReason.SUN_PATH
        assert _kinds(plan.warnings) == _KERNEL
        assert plan.transition is None

    def test_sun_gate_refuses_the_retune_pose(self, site, monkeypatch):
        """The retune pose is judged at the arrival the visit will use.

        The transition's own gate checks the pose at the arrival it
        computed; a slew 20 s later than that (under the 30 s of slack
        before the pass, so no re-anchor) puts the arrival inside a Sun
        window opening just after the computed arrival, which the
        transition never saw, and only the visit's gate sees it.
        """
        planned = _record_transitions(monkeypatch, delays=(20.0,))
        sweeps = _record_sweeps(monkeypatch)

        def after_the_planned_arrival(az, el, t):
            if not planned:
                return True
            late = (t - planned[-1].arrival).to_value("s")
            return not 0.0 < late < 60.0

        sun_safe = fake_sun_model(after_the_planned_arrival, batch=False)
        plan = _saturn_visit(site, sun_safe=sun_safe, slew_safe=lambda *_: True)
        assert plan.reason is DeferralReason.SUN_POINT
        assert _kinds(plan.warnings) == _KERNEL
        assert len(planned) == 1 and planned[0].safe
        assert sweeps == []  # refused before any pass was swept

    def test_sun_gate_refuses_a_pass_sample(self, site, monkeypatch):
        """A pass that runs into the zone after the slew and the retune pose cleared it.

        The stub's zone covers the sky above 50 deg from 06:50, which the
        slew (06:30:58), the retune pose and the transition's hold
        (to 06:43:16) all precede; the pass (from 06:46:28) runs into it.
        """
        import numpy as np

        cut = Time(NIGHT_START, scale="utc") + TimeDelta(1200.0, format="sec")

        def late_and_high(az, el, t):
            return np.logical_not((np.asarray(Time(t).unix) > cut.unix) & (np.asarray(el) > 50.0))

        planned = _record_transitions(monkeypatch)
        sweeps = _record_sweeps(monkeypatch)
        sun_safe = fake_sun_model(late_and_high, batch=True)
        plan = _saturn_visit(site, sun_safe=sun_safe, slew_safe=lambda *_: True)
        assert plan.reason is DeferralReason.SUN_POINT
        # The kernel's own arc check sees the zone too; it only advises.
        assert _kinds(plan.warnings) == ["sun_zone", *_KERNEL]
        assert len(planned) == 1 and planned[0].safe
        assert sweeps == [False]

    def test_a_late_slew_past_the_reanchor_closes_the_window(self, site, monkeypatch):
        """The one re-anchor is spent and the slew is still late.

        The first slew is 40 s late, past the 30 s of slack, so the visit
        re-anchors; the second is 2000 s late, past the new anchor too.
        """
        planned = _record_transitions(monkeypatch, delays=(40.0, 2000.0))
        kernel = _record_kernel_calls(monkeypatch)
        plan = _saturn_visit(site)
        assert plan.reason is DeferralReason.WINDOW_CLOSED
        assert _kinds(plan.warnings) == _KERNEL + _KERNEL
        assert len(planned) == 2 and all(t.safe for t in planned)
        assert len(kernel) == 2
        assert kernel[1]["start_time"] > kernel[0]["start_time"]

    def test_the_dwell_fallback_holds_across_the_reanchor(self, site, monkeypatch):
        """A dwell the crossing cannot hold is dropped for the re-anchored solve too."""
        planned = _record_transitions(monkeypatch, delays=(40.0, 2000.0))
        kernel = _record_kernel_calls(monkeypatch)
        plan = _saturn_visit(site, ScanOverrides(dwell=5000.0))
        assert plan.reason is DeferralReason.WINDOW_CLOSED
        assert _kinds(plan.warnings) == [_DWELL_FALLBACK, *_KERNEL, *_KERNEL]
        assert len(planned) == 2
        assert [call.get("dwell") for call in kernel] == [5000.0, None, None]

    def test_dwell_on_a_multi_pass_visit_is_dropped(self, site, monkeypatch):
        kernel = _record_kernel_calls(monkeypatch)
        plan = _saturn_visit(
            site, ScanOverrides(dwell=60.0), policy=CalibrationNightPolicy(n_passes=2)
        )
        assert plan.feasible, plan.summary
        assert len(plan.passes) == 2
        assert _kinds(plan.warnings) == [
            "dwell applies to a single pass; a multi-pass visit scans the crossing",
            *_KERNEL,
            *_KERNEL,
        ]
        assert len(kernel) == 1 and "dwell" not in kernel[0]
        for block in plan.blocks:
            if block.scan_type == "planet_cal":
                assert "dwell" not in block.metadata["scan_params"]
                assert "dwell" not in block.metadata["requested"]


def test_a_visit_whose_anchor_probe_grazes_the_module_is_deferred_as_under_the_table(site):
    """The solved-throw night defers the body exactly as the table-throw night does.

    The night starts so that its first Neptune visit on i2, on 2026-09-11, is
    anchored at 04:35:49.303, 2.9 deg below Neptune's culmination and in the
    middle of the band of anchors where the probe that derives the boresight
    elevation lifts the module so high that Neptune reaches only one of its
    cover vertices. The probe reads only its start, so the padding the solved
    throw plans with (none) has no say, and the visit takes the real solve's
    refusal: the derived boresight elevation lies below Neptune's elevation at
    the anchor, so the forward search takes the next day's arc, which the 24 h
    window cuts off below the module's top edge.
    """

    def night(use_table_throw):
        policy = CalibrationNightPolicy(footprint="i2", use_table_throw=use_table_throw)
        timeline = plan_calibration_night(
            ["neptune"],
            site,
            "2026-09-11T04:19:59.511",
            "2026-09-11T04:25:59.511",
            policy=policy,
        )
        return read_calibration_night_metadata(timeline)

    solved, table = night(False), night(True)
    assert solved["deferrals"] == [
        {"at": "2026-09-11 04:19:59.511", "body": "neptune", "reason": "unplannable"}
    ]
    assert solved["deferrals"] == table["deferrals"]
    assert solved["warnings"] == table["warnings"]
    # The refusal names the anchor, which pins the visit to the band's middle.
    refusal = solved["warnings"][-1]["message"]
    assert "Neptune is not fully observable at 2026-09-11 04:35:49.303" in refusal


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
        default_ctx = NightContext.build(["saturn"], site, NIGHT_START, NIGHT_END)
        (default_candidate,) = list_candidates(state, default_ctx)
        assert default_candidate.moon_separation is None

    def test_the_slew_estimate_wraps_inside_the_limits(self, site):
        """A move across the north seam is priced as the unwind the mount makes.

        From a mount at 350 deg to a body at 20 deg the nearest image is 380
        deg, outside the azimuth window, so the estimate must use 20 deg.
        """
        from fyst_trajectories.overhead.utils import _normalize_az

        assert _normalize_az(20.0, site, ref=350.0) == 20.0
        assert _normalize_az(20.0, site, ref=-170.0) == 20.0
        assert _normalize_az(300.0, site, ref=-10.0) == -60.0


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


def _planned(site, overrides=None, **context):
    """Plan one Saturn visit at the start of the night, returning the context and state too."""
    ctx = NightContext.build(["saturn", "uranus"], site, NIGHT_START, NIGHT_END, **context)
    state = NightState.initial(ctx.start_time)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = plan_visit(state, ctx, "saturn", overrides)
    assert plan.feasible, plan.summary
    return ctx, state, plan


def _only_pass(plan):
    (block,) = [b for b in plan.blocks if b.scan_type == "planet_cal"]
    return block


class TestPassRebuildRepeatsTheSearch:
    """Each pass block records where the kernel's search began, and rebuilds from it to the bit.

    The paths that move the anchor are driven through ``plan_visit``: a
    re-anchor after a late slew with the dwell fallback before it, and an
    escape before the visit. The last two tests are passes that a block
    without ``search_start``, re-solved inside its pass widened by 300 s,
    cannot rebuild.
    """

    def test_a_re_anchored_visit_records_the_anchor_it_kept(self, site, monkeypatch):
        """The 5000 s dwell is dropped, then the 40 s late slew re-anchors the visit."""
        _record_transitions(monkeypatch, delays=(40.0,))
        kernel = _record_kernel_calls(monkeypatch)
        ctx, state, plan = _planned(site, ScanOverrides(dwell=5000.0))
        assert [call.get("dwell") for call in kernel] == [5000.0, None, None]
        first, kept = kernel[1]["start_time"], kernel[2]["start_time"]
        assert kept > first
        block = _only_pass(plan)
        assert block.metadata["search_start"] == [kept.utc.jd1, kept.utc.jd2]
        assert "dwell" not in block.metadata["scan_params"]
        ((_, rebuilt),) = _rebuild_visit(ctx, state, plan)
        _assert_same_pass(rebuilt, plan.passes[0])

    def test_an_escape_first_records_the_anchor_after_it(self, site, monkeypatch):
        """The visit is planned from where the escape ends; the block records that anchor."""
        kernel = _record_kernel_calls(monkeypatch)
        ctx, state, plan = _planned(site, sun_safe=pose_blocker(*BOOTSTRAP_POSE))
        escape = plan.blocks[0]
        assert escape.patch_name == "sun_escape"
        (anchor,) = [call["start_time"] for call in kernel]
        assert anchor > escape.t_stop
        assert _only_pass(plan).metadata["search_start"] == [anchor.utc.jd1, anchor.utc.jd2]
        ((_, rebuilt),) = _rebuild_visit(ctx, state, plan)
        _assert_same_pass(rebuilt, plan.passes[0])

    def test_a_dwell_cutting_more_than_300_s_rebuilds(self, site):
        """A 60 s dwell on Saturn's 736 s crossing starts the pass 338 s into it.

        Re-solved inside the pass widened by 300 s, the search opens after the
        source has crossed the top of the footprint and the rebuild is
        refused; the planner's own search opened before the crossing.
        """
        ctx, state, plan = _planned(site, ScanOverrides(dwell=60.0))
        block = _only_pass(plan)
        cut = 0.5 * (block.metadata["solved"]["crossing_seconds"] - 60.0)
        assert cut > 300.0
        ((_, rebuilt),) = _rebuild_visit(ctx, state, plan)
        _assert_same_pass(rebuilt, plan.passes[0])
        assert _rebuild_visit(ctx, state, plan, _without_search_start(plan.blocks)) == []

    def test_an_off_centre_pass_crossing_its_elevation_after_it_rebuilds(self, site):
        """On ``i3`` Saturn reaches the boresight elevation 21 min after the pass starts.

        The kernel places the module on the source from that crossing, so a
        search that ends 300 s after the 12 min pass has nothing to place it
        from and the rebuild is refused.
        """
        ctx, state, plan = _planned(site, policy=CalibrationNightPolicy(footprint="i3"))
        ((_, rebuilt),) = _rebuild_visit(ctx, state, plan)
        _assert_same_pass(rebuilt, plan.passes[0])
        assert _rebuild_visit(ctx, state, plan, _without_search_start(plan.blocks)) == []


def _with_search_start(plan, record):
    """Return the visit's blocks with the pass block's ``search_start`` replaced by ``record``."""
    return [
        dataclasses.replace(b, metadata={**b.metadata, "search_start": record})
        if b.scan_type == "planet_cal"
        else b
        for b in plan.blocks
    ]


def _later_record(plan, seconds):
    """Return the pass block's ``search_start``, moved ``seconds`` later."""
    later = Time(*_only_pass(plan).metadata["search_start"], format="jd", scale="utc")
    later += TimeDelta(seconds, format="sec")
    return [later.jd1, later.jd2]


class TestPassRebuildRefusesAnotherRecord:
    """A rebuilt pass that does not overlap its block is refused, as is a record of another form.

    A record of an instant after the pass began repeats a search that finds the next day's
    crossing, which does not overlap the block, and a record of any other
    form names no instant. The rebuild refuses both with an error that names
    the key, and the best-effort ``schedule_to_trajectories`` logs and skips
    the block, as it does every other block it cannot rebuild.
    """

    @pytest.mark.parametrize("seconds", [600.0, 6 * 3600.0], ids=["10 min", "6 h"])
    def test_a_record_of_a_later_search_is_refused(self, saturn_visit, seconds):
        """The search finds the next day's crossing, which does not overlap the block."""
        ctx, _, plan = saturn_visit
        (block,) = [
            b
            for b in _with_search_start(plan, _later_record(plan, seconds))
            if b.scan_type == "planet_cal"
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(BlockNotReconstructableError, match="search_start"):
                _generate_trajectory_for_block(block, ctx.site)

    def test_a_record_of_another_form_is_a_schema_error(self, saturn_visit):
        """``[True, False]`` holds two numbers to Python, but it names no instant.

        Read as a Julian date it would put the search in the year -4713.
        """
        ctx, _, plan = saturn_visit
        (block,) = [
            b for b in _with_search_start(plan, [True, False]) if b.scan_type == "planet_cal"
        ]
        with pytest.raises(ScanParamsSchemaError, match="search_start"):
            _generate_trajectory_for_block(block, ctx.site)

    @pytest.mark.parametrize("record", ["later", [True, False]], ids=["later search", "booleans"])
    def test_the_best_effort_rebuild_logs_and_skips_it(self, saturn_visit, caplog, record):
        ctx, state, plan = saturn_visit
        if record == "later":
            record = _later_record(plan, 600.0)
        with caplog.at_level(logging.WARNING, logger="fyst_trajectories.overhead.simulation"):
            assert _rebuild_visit(ctx, state, plan, _with_search_start(plan, record)) == []
        (logged,) = caplog.records
        assert "search_start" in logged.getMessage()


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


# A resume compares times to 1 ms: the timeline stores block times and the
# payload stores record times as ISO strings with millisecond precision.
_RESUME_TOL_SEC = 1e-3
_SCRIPT = ["jupiter", "jupiter", "saturn", "saturn"]


def _same_time(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs((a - b).to_value("s")) <= _RESUME_TOL_SEC


def _through_ecsv(timeline, tmp_path):
    path = tmp_path / "night.ecsv"
    write_timeline(timeline, path)
    return read_timeline(path)


def _driver_state_at(states, t):
    """Return the last state the driver produced at ``t``."""
    return [s for s in states if _same_time(s.t, t)][-1]


def _assert_same_state(resumed, driver):
    """Every field of a resumed state equals the driver's, times to 1 ms."""
    assert _same_time(resumed.t, driver.t)
    assert (resumed.az, resumed.el) == pytest.approx((driver.az, driver.el), abs=1e-9)
    assert [b.scan_type for b in resumed.blocks] == [b.scan_type for b in driver.blocks]
    for got, want in zip(resumed.blocks, driver.blocks):
        assert _same_time(got.t_start, want.t_start) and _same_time(got.t_stop, want.t_stop)
    for f in dataclasses.fields(driver.cal_state):
        got, want = getattr(resumed.cal_state, f.name), getattr(driver.cal_state, f.name)
        assert _same_time(got, want), f.name
    assert set(resumed.deferred) == set(driver.deferred)
    for body, (retry, reason) in driver.deferred.items():
        assert _same_time(resumed.deferred[body][0], retry)
        assert resumed.deferred[body][1] is reason
    assert dict(resumed.dropped) == dict(driver.dropped)
    assert resumed.script_index == driver.script_index
    assert _same_time(resumed.script_waiting_since, driver.script_waiting_since)
    assert resumed.unplaced == driver.unplaced
    assert resumed.scan_counter == driver.scan_counter


@pytest.fixture(scope="module")
def scripted_night(site):
    """Plan a scripted night with real plans, capturing every state the driver produced.

    Jupiter stays below the band, so both Jupiter entries wait out
    ``max_wait_seconds`` and are set aside before the Saturn entries run.
    """
    from fyst_trajectories.overhead.calibration_night import night

    policy = CalibrationNightPolicy(max_wait_seconds=600.0, time_step=300.0)
    ctx = NightContext.build(
        ["saturn", "jupiter"], site, "2026-09-11T06:30:00", "2026-09-11T07:30:00", policy=policy
    )
    rule = ScriptedSelection(_SCRIPT)
    states = []

    def capture(function):
        def wrapper(*args, **kwargs):
            state = function(*args, **kwargs)
            states.append(state)
            return state

        return wrapper

    with pytest.MonkeyPatch.context() as mp:
        for name in ("commit_visit", "advance_idle", "_idle"):
            mp.setattr(night, name, capture(getattr(night, name)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            timeline = night._run_night(ctx, rule)
    set_aside = []
    for state in states:
        if len(state.unplaced) > len(set_aside):
            set_aside.append(state.t)
    return timeline, tuple(states), tuple(set_aside), rule


class TestResumeFromTimeline:
    """NightContext.from_timeline and NightState.from_timeline against the driver."""

    @pytest.mark.parametrize("source", ["memory", "ecsv"])
    def test_scripted_position_matches_the_driver(self, scripted_night, source, tmp_path):
        """Resumed at each set-aside instant and while the second entry waits.

        The wait timer dates from the first block after the last script
        event, not from the start of the trailing run of waiting idles,
        which spans both Jupiter entries.
        """
        timeline, states, set_aside, rule = scripted_night
        assert [t.iso for t in set_aside] == [
            "2026-09-11 06:40:00.000",
            "2026-09-11 06:50:00.000",
        ]
        meta = read_calibration_night_metadata(timeline)
        assert [entry["at"] for entry in meta["unplaced"]] == [t.iso for t in set_aside]
        if source == "ecsv":
            timeline = _through_ecsv(timeline, tmp_path)
        waiting = Time("2026-09-11T06:45:00", scale="utc")
        assert _same_time(_driver_state_at(states, waiting).script_waiting_since, set_aside[0])
        for t in (*set_aside, waiting):
            resumed = NightState.from_timeline(timeline, t, selection=rule)
            _assert_same_state(resumed, _driver_state_at(states, t))
        for state in states:
            resumed = NightState.from_timeline(timeline, state.t, selection=rule)
            _assert_same_state(resumed, _driver_state_at(states, state.t))
        # Inside the first waiting idle of the second entry, which has not ended.
        mid_wait = Time("2026-09-11T06:42:00", scale="utc")
        resumed = NightState.from_timeline(timeline, mid_wait, selection=rule)
        assert _same_time(resumed.script_waiting_since, set_aside[0])

    def test_short_night_round_trip(self, short_night, tmp_path):
        """The state after the first visit, and the next visit planned from it."""
        site, timeline = short_night
        ctx = NightContext.build(
            ["saturn", "uranus"], site, "2026-09-11T06:30:00", "2026-09-11T07:30:00"
        )
        initial = NightState.initial(ctx.start_time)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first = commit_visit(
                initial,
                plan_visit(initial, ctx, "saturn"),
                retry_after=ctx.policy.retry_after_seconds,
            )
            expected = plan_visit(first, ctx, "saturn")
        assert first.script_index == 1 and first.scan_counter == 1
        for source in (timeline, _through_ecsv(timeline, tmp_path)):
            ctx_r = NightContext.from_timeline(source)
            assert ctx_r.targets == ctx.targets
            assert ctx_r.policy == ctx.policy
            assert dict(ctx_r.tables) == dict(ctx.tables)
            assert ctx_r.overhead_model == ctx.overhead_model
            assert ctx_r.calibration_policy == ctx.calibration_policy
            assert ctx_r.site == ctx.site
            for got, want in (
                (ctx_r.start_time, ctx.start_time),
                (ctx_r.end_time, ctx.end_time),
                (ctx_r.requested_start, ctx.requested_start),
                (ctx_r.requested_end, ctx.requested_end),
            ):
                assert _same_time(got, want)
            resumed = NightState.from_timeline(source, first.t)
            _assert_same_state(resumed, first)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plan = plan_visit(resumed, ctx_r, "saturn")
            assert (plan.feasible, plan.reason) == (expected.feasible, expected.reason)
            assert [b.scan_type for b in plan.blocks] == [b.scan_type for b in expected.blocks]
            for got, want in zip(plan.blocks, expected.blocks):
                assert _same_time(got.t_start, want.t_start)
                assert _same_time(got.t_stop, want.t_stop)

    def test_a_timeline_without_search_start_resumes_and_rebuilds(self, short_night, tmp_path):
        """A night whose pass blocks carry no ``search_start`` reads, resumes and rebuilds.

        Its passes are re-solved inside the recorded pass widened by 300 s,
        which for these centre-module passes lands within about 0.1 s of the
        planned start (0.007 to 0.014 s here).
        """
        _, timeline = short_night
        keyless = dataclasses.replace(timeline, blocks=_without_search_start(timeline.blocks))
        keyless = _through_ecsv(keyless, tmp_path)
        assert not any("search_start" in b.metadata for b in keyless.blocks)
        ctx, ctx_keyless = NightContext.from_timeline(timeline), NightContext.from_timeline(keyless)
        assert (ctx_keyless.targets, ctx_keyless.policy) == (ctx.targets, ctx.policy)
        first_pass = next(b for b in timeline.blocks if b.scan_type == "planet_cal")
        _assert_same_state(
            NightState.from_timeline(keyless, first_pass.t_stop),
            NightState.from_timeline(timeline, first_pass.t_stop),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pairs = schedule_to_trajectories(keyless, science_only=False)
        assert len(pairs) == 3
        for block, rebuilt in pairs:
            assert abs((rebuilt.trajectory.start_time - block.t_start).to_value("s")) < 0.1

    def test_commit_and_replay_share_the_fold(self, short_night):
        """Both stamp each retune, skydip and pass at the block's start."""
        from fyst_trajectories.overhead import CalibrationState
        from fyst_trajectories.overhead.calibration_night.state import _fold_calibrations

        _, timeline = short_night
        blocks = tuple(timeline.blocks)
        folded = _fold_calibrations(CalibrationState(), blocks)

        def last_start(cal_type):
            return [b.t_start for b in blocks if b.metadata.get("cal_type") == cal_type][-1]

        assert folded.last_retune == last_start(CalibrationType.RETUNE)
        assert folded.last_skydip == last_start(CalibrationType.SKYDIP)
        assert folded.last_planet_cal == last_start(CalibrationType.PLANET_CAL)
        plan = VisitPlan("saturn", True, None, None, blocks, (), ())
        assert commit_visit(NightState.initial(timeline.start_time), plan).cal_state == folded
        assert NightState.from_timeline(timeline, blocks[-1].t_stop).cal_state == folded

    def test_records_at_the_resume_time_count(self, short_night):
        """Two deferrals share the instant a pass ended; a resume at that instant has both."""
        _, timeline = short_night
        meta = read_calibration_night_metadata(timeline)
        last_at = meta["deferrals"][-1]["at"]
        tied = {d["body"]: d["reason"] for d in meta["deferrals"] if d["at"] == last_at}
        assert sorted(tied) == ["saturn", "uranus"]
        t = Time(last_at, scale="utc")
        assert any(_same_time(b.t_stop, t) for b in timeline.blocks if b.scan_type == "planet_cal")
        state = NightState.from_timeline(timeline, t)
        retry_after = meta["policy"]["retry_after_seconds"]
        for body, reason in tied.items():
            retry, got = state.deferred[body]
            assert got is DeferralReason(reason)
            assert (retry - t).to_value("s") == pytest.approx(retry_after)

    def test_a_different_sun_model_is_refused(self, short_night):
        from fyst_trajectories.sun_models import make_sun_safe

        _, timeline = short_night
        wider = get_fyst_site(sun_exclusion_radius=50.0, sun_warning_radius=55.0)
        with pytest.raises(ValueError, match="sun_safe"):
            NightContext.from_timeline(timeline, sun_safe=make_sun_safe("scalar", site=wider))

    def test_a_scripted_night_needs_its_rule(self, scripted_night):
        timeline, _, set_aside, _ = scripted_night
        with pytest.raises(ValueError, match="ScriptedSelection"):
            NightState.from_timeline(timeline, set_aside[0])

    def test_a_time_before_the_usable_start_is_refused(self, short_night):
        _, timeline = short_night
        with pytest.raises(ValueError, match="before the night's usable start"):
            NightState.from_timeline(timeline, timeline.start_time - TimeDelta(60.0, format="sec"))

    def test_a_different_usable_interval_is_refused(self, short_night):
        from fyst_trajectories.overhead.calibration_night.policy import (
            encode_calibration_night_metadata,
        )

        _, timeline = short_night
        meta = read_calibration_night_metadata(timeline)
        start, end = meta["usable_interval"]
        shifted = (Time(start, scale="utc") + TimeDelta(1.0, format="sec")).iso
        tampered = dataclasses.replace(
            timeline,
            metadata=encode_calibration_night_metadata({**meta, "usable_interval": [shifted, end]}),
        )
        with pytest.raises(ValueError, match="usable interval"):
            NightContext.from_timeline(tampered)

    def test_a_record_without_the_throw_switch_resumes_on_the_table(self, short_night, monkeypatch):
        """A payload written before ``use_table_throw`` existed swept the table's throw."""
        from fyst_trajectories.overhead.calibration_night.policy import (
            encode_calibration_night_metadata,
        )

        _, timeline = short_night
        meta = read_calibration_night_metadata(timeline)
        policy = {k: v for k, v in meta["policy"].items() if k != "use_table_throw"}
        older = dataclasses.replace(
            timeline, metadata=encode_calibration_night_metadata({**meta, "policy": policy})
        )
        ctx = NightContext.from_timeline(older)
        assert ctx.policy.use_table_throw is True
        first_pass = next(b for b in older.blocks if b.scan_type == "planet_cal")
        state = NightState.from_timeline(older, first_pass.t_stop)
        calls = _record_kernel_calls(monkeypatch)
        plan_visit(state, ctx, "saturn")
        _, el_est = ctx.body_altaz("saturn", calls[0]["start_time"])
        assert calls[0]["az_throw"] == table_for(ctx.tables, "saturn").az_throw_at(el_est)
        assert "az_padding" not in calls[0]

    @pytest.mark.parametrize("use_table_throw", [False, True])
    def test_a_record_with_the_throw_switch_keeps_its_value(self, site, use_table_throw):
        ctx = NightContext.build(
            ["saturn"],
            site,
            NIGHT_START,
            NIGHT_END,
            policy=CalibrationNightPolicy(use_table_throw=use_table_throw),
            visit_planner=_parked_visit,
        )
        timeline = _run(ctx)
        assert read_calibration_night_metadata(timeline)["policy"]["use_table_throw"] is (
            use_table_throw
        )
        assert NightContext.from_timeline(timeline).policy.use_table_throw is use_table_throw

    @pytest.mark.parametrize("tag", ["IM0", "I3"])
    def test_a_night_on_any_module_name_reads_back(self, site, tag, tmp_path):
        """The tag is recorded as given, and the record rebuilds the same policy."""
        policy = CalibrationNightPolicy(footprint=tag)
        ctx = NightContext.build(
            ["saturn"], site, NIGHT_START, NIGHT_END, policy=policy, visit_planner=_parked_visit
        )
        timeline = _run(ctx)
        assert read_calibration_night_metadata(timeline)["policy"]["footprint"] == tag
        for source in (timeline, _through_ecsv(timeline, tmp_path)):
            assert NightContext.from_timeline(source).policy == policy

    def test_a_record_naming_no_module_is_refused(self, site):
        """A recorded footprint no module answers to is refused on reading, not at a visit."""
        from fyst_trajectories.overhead.calibration_night.policy import (
            encode_calibration_night_metadata,
        )

        ctx = NightContext.build(
            ["saturn"], site, NIGHT_START, NIGHT_END, visit_planner=_parked_visit
        )
        timeline = _run(ctx)
        meta = read_calibration_night_metadata(timeline)
        policy = {**meta["policy"], "footprint": "c,i1"}
        tampered = dataclasses.replace(
            timeline, metadata=encode_calibration_night_metadata({**meta, "policy": policy})
        )
        refusal = re.escape("footprint: Unknown PrimeCam module 'c,i1'")
        with pytest.raises(ValueError, match=refusal):
            NightContext.from_timeline(tampered)
        with pytest.raises(ValueError, match=refusal):
            NightState.from_timeline(tampered, timeline.blocks[0].t_stop)

    def test_axis_limits_the_file_drops_are_refused(self, site, tmp_path):
        """A night planned on lowered limits reads back with the FYST limits."""
        limits = site.telescope_limits
        lowered = dataclasses.replace(
            site,
            telescope_limits=dataclasses.replace(
                limits, elevation=dataclasses.replace(limits.elevation, max=70.0)
            ),
        )
        ctx = NightContext.build(
            ["saturn"], lowered, NIGHT_START, NIGHT_END, visit_planner=_parked_visit
        )
        timeline = _run(ctx)
        assert NightContext.from_timeline(timeline).site == lowered
        back = _through_ecsv(timeline, tmp_path)
        assert back.site.telescope_limits.elevation.max == 90.0
        with pytest.raises(ValueError, match="axis limits"):
            NightContext.from_timeline(back)
        assert NightContext.from_timeline(back, site=lowered).site == lowered
