"""The slew onto a science visit ends where the visit's scan begins.

A science visit is booked as a slew, a boundary retune when one is due, a
wait for a constant-elevation pass that opens after the retune, and the
first science subscan. The slew ends at the first sample of that subscan's
trajectory, in azimuth and in elevation, so the timeline records the
telescope at a pose it holds and the move onto the scan is priced in full.
For a pong or daisy that sample is the field centre at the scan's start,
which the pattern starts on; for a constant-elevation pass it is the planned
sweep's position at the scan's start. Each night below is generated once,
and every visit on it is checked against its own rebuilt trajectory.
"""

import dataclasses
import itertools
import math
import warnings

import pytest
from _scheduler_helpers import _initial_state, _make_ctx
from astropy.time import Time, TimeDelta

from fyst_trajectories import (
    InstrumentOffset,
    get_absolute_times,
    get_fyst_site,
    get_primecam_offset,
)
from fyst_trajectories.coordinates import Coordinates
from fyst_trajectories.dispatch import estimate_slew_time
from fyst_trajectories.exceptions import PointingError, PointingWarning
from fyst_trajectories.offsets import _solve_boresight
from fyst_trajectories.overhead import (
    BlockType,
    CalibrationPolicy,
    DeferralReason,
    ObservingPatch,
    OverheadModel,
    generate_timeline,
    schedule_to_trajectories,
)
from fyst_trajectories.overhead.scheduler import (
    PatchSelectionPhase,
    Scheduler,
    ScienceScanPhase,
    SlewPhase,
    phases,
)
from fyst_trajectories.overhead.transitions import Transition

from . import _nights

#: How far the slew's end may sit from the first sample, in degrees.
_POSE_TOL_DEG = 0.02

#: How late a slew may arrive and still count as on time, in seconds: the
#: offline scheduler's arrival tolerance, well under one trajectory sample.
_ARRIVAL_TOL_SEC = 0.01

#: The price of moving the pose tolerance, in seconds (about 1/3 s per deg in
#: cruise, steeper on a short move).
_PRICE_TOL_SEC = 0.05


def _ce(name, ra, dec, width, height, elevation):
    return ObservingPatch(
        name=name,
        ra_center=ra,
        dec_center=dec,
        width=width,
        height=height,
        scan_type="constant_el",
        velocity=1.0,
        elevation=elevation,
    )


def _pattern(name, ra, dec, size, scan_type):
    return ObservingPatch(
        name=name,
        ra_center=ra,
        dec_center=dec,
        width=size,
        height=size,
        scan_type=scan_type,
        velocity=0.5,
    )


_DEEP56_TEST = _ce("Deep56", 24.0, -32.0, 40.0, 10.0, 50.0)
_DEEP56_SEEDED = _ce("Deep56", 15.0, -2.0, 60.0 * math.cos(math.radians(-2.0)), 14.0, 50.0)

#: The nights shared with the other simulator tests (``_nights.NIGHTS``):
#: the 8-hour regression night, the quick-start page's night, both halves of
#: one transit, the pong and daisy fields of the five-patch day, three
#: nights whose patches carry a detector offset, where the slew ends at the
#: boresight's first sample, and the two nights of a patch with an azimuth
#: window, where it ends at the first sample of the first window scan.
_SHARED = (
    "regression",
    "quickstart",
    "august",
    "pong_and_daisy",
    "ce_offset",
    "lsa_window_offset",
    "pong_and_daisy_offset",
    "window_rising",
    "window_setting_day",
)

#: Each night: patches, window and the keyword arguments of generate_timeline.
_NIGHTS = {
    **{name: _nights.NIGHTS[name] for name in _SHARED},
    # docs/overhead_timeline.rst: one field observed rising and setting
    "timeline_page": (
        [_ce("CMB", 0.0, -2.0, 10.0, 6.0, 45.0)],
        "2026-09-15T00:00:00",
        "2026-09-15T12:00:00",
        {
            "overhead_model": OverheadModel(max_scan_duration=1800.0),
            "calibration_policy": CalibrationPolicy(
                retune_cadence=0.0,
                pointing_cadence=1800.0,
                focus_cadence=3600.0,
                skydip_cadence=7200.0,
                planet_cal_cadence=43200.0,
            ),
        },
    ),
    # The seeded Deep56 field, rising and setting, with a boundary retune
    # and a wait for the pass between the slew and the block ...
    "deep56_seeded": (
        [_DEEP56_SEEDED],
        "2026-11-14T18:00:00",
        "2026-11-15T10:00:00",
        {},
    ),
    # ... and with time-based retunes, where only the wait sits between them.
    "deep56_seeded_hourly_retune": (
        [_DEEP56_SEEDED],
        "2026-11-14T18:00:00",
        "2026-11-15T10:00:00",
        {"calibration_policy": CalibrationPolicy(retune_cadence=3600.0)},
    ),
}


_WAITING = str(DeferralReason.WAITING_FOR_PASS)


def _leads_into_the_scan(block):
    """Whether ``block`` is a boundary retune or a wait for a pass, booked after the slew."""
    if block.block_type == BlockType.CALIBRATION:
        return block.scan_type == "retune"
    return block.block_type == BlockType.IDLE and block.metadata.get("reason") == _WAITING


@pytest.fixture(scope="module", params=sorted(_NIGHTS))
def night(request):
    """One night of ``_NIGHTS``, its rebuilt science blocks and its visits.

    Returns ``(timeline, first_samples, visits)``: ``first_samples`` maps a
    science block's ``id`` to its rebuilt trajectory, and each visit is
    ``(before, slew, between, science)``, the block before the slew, the
    slew onto the visit, the blocks between it and the visit's first
    science block (a boundary retune, a wait for the pass), and that block.
    """
    if request.param in _SHARED:
        timeline = _nights.night(request.param)
        pairs = [
            (block, scan)
            for block, scan in _nights.rebuilt(request.param)
            if block.block_type == BlockType.SCIENCE
        ]
    else:
        patches, start, end, kwargs = _NIGHTS[request.param]
        timeline = generate_timeline(
            patches=patches, site=get_fyst_site(), start_time=start, end_time=end, **kwargs
        )
        pairs = schedule_to_trajectories(timeline)
    rebuilt = {id(block): scan.trajectory for block, scan in pairs}

    blocks = list(timeline.blocks)
    visits = []
    for i, block in enumerate(blocks):
        if block.block_type != BlockType.SCIENCE or block.subscan_index != 0:
            continue
        j = i - 1
        while _leads_into_the_scan(blocks[j]):
            j -= 1
        visits.append((blocks[j - 1], blocks[j], blocks[j + 1 : i], block))
    return request.param, timeline, rebuilt, pairs, visits


def _on_turn_of(az, ref):
    """Return ``az`` on the turn nearest ``ref``."""
    return az + 360.0 * round((ref - az) / 360.0)


def test_every_visit_is_slewed_onto(night):
    """Each visit's first science block follows a slew onto its own patch."""
    name, _, _, _, visits = night
    assert visits, f"{name}: no science visit"
    for _, slew, _, science in visits:
        assert slew.block_type == BlockType.SLEW
        assert slew.patch_name == f"slew_to_{science.patch_name}"


def test_the_slew_ends_at_the_first_sample_of_the_scan(night):
    """The slew's end pose is the first sample of the visit's first science block.

    Compared on the slew's wrap: the block records its range in the wrap
    the slew chose, while the rebuilt trajectory is in the planner's.
    """
    name, _, rebuilt, _, visits = night
    for _, slew, _, science in visits:
        traj = rebuilt[id(science)]
        az0 = _on_turn_of(float(traj.az[0]), slew.az_end)
        el0 = float(traj.el[0])
        where = f"{name}: {science.patch_name} at {science.t_start.isot}"
        assert abs(slew.az_end - az0) <= _POSE_TOL_DEG, where
        assert abs(slew.elevation - el0) <= _POSE_TOL_DEG, where
        assert science.az_start - 1e-9 <= slew.az_end <= science.az_end + 1e-9, where


def test_the_slew_is_priced_for_the_move_to_the_scan(night):
    """The slew's duration is the price of the move to the scan's start, plus a short hold.

    Priced with the library's estimate from the pose before the slew to
    the first sample of the visit's scan, plus the settle time. A slew that
    arrives before its visit starts waits there inside the same block, so
    the duration may exceed that price, by the hold. The hold is what the
    solve's previous slew took beyond its last one (both leave at the
    tick), so it is at most the time to slew between their two targets
    (one move can always be split in two), and both are samples of the
    scan, inside the range its first subscan records: the hold is held
    under the time to cross that range. It is never short of the price by
    more than the solve's arrival tolerance (0.01 s), plus the pricing of
    the pose tolerance (``_PRICE_TOL_SEC``).
    """
    name, timeline, rebuilt, _, visits = night
    settle = OverheadModel().settle_time
    for before, slew, _, science in visits:
        assert slew.az_start == pytest.approx(before.end_pose_az, abs=1e-9)
        traj = rebuilt[id(science)]
        where = f"{name}: {science.patch_name} at {science.t_start.isot}"
        az0 = _on_turn_of(float(traj.az[0]), slew.az_end)
        el0 = float(traj.el[0])
        priced = estimate_slew_time(slew.az_start, before.end_pose_el, az0, el0, site=timeline.site)
        hold = slew.duration - (priced + settle)
        crossing = estimate_slew_time(
            science.az_start, el0, science.az_end, el0, site=timeline.site
        )
        assert hold >= -(_ARRIVAL_TOL_SEC + _PRICE_TOL_SEC), where
        assert hold <= crossing + _PRICE_TOL_SEC, where


def test_the_visit_starts_where_the_slew_ended(night):
    """Between the slew and the scan sit at most a boundary retune and then a wait.

    Both hold the pose the slew ended at, the wait (``waiting_for_pass``)
    only on a constant-elevation visit whose pass opens after the retune,
    and the blocks run on from the slew to the scan without a gap.
    """
    name, _, _, _, visits = night
    for _, slew, between, science in visits:
        kinds = ["retune" if b.block_type == BlockType.CALIBRATION else "wait" for b in between]
        assert kinds in ([], ["retune"], ["wait"], ["retune", "wait"]), name
        if "wait" in kinds:
            assert science.scan_type == "constant_el", name
        chain = [slew, *between, science]
        for before, after in itertools.pairwise(chain):
            assert after.t_start.unix == pytest.approx(before.t_stop.unix, abs=1e-6), name
        for b in between:
            assert b.elevation == pytest.approx(slew.elevation, abs=1e-9), name
            assert b.end_pose_el == pytest.approx(slew.elevation, abs=1e-9), name
            # Before a wait every block hands on the slew's pose.
            if "wait" in kinds:
                assert b.end_pose_az == pytest.approx(slew.az_end, abs=1e-9), name


def test_a_boundary_retune_records_the_range_of_the_scan_after_it(night):
    """A retune booked before a subscan carries that subscan's own azimuth range.

    The subscan follows the retune directly, or after the wait for its pass.
    """
    name, timeline, _, _, _ = night
    blocks = list(timeline.blocks)
    boundary = []
    for i, b in enumerate(blocks[:-1]):
        if b.scan_type != "retune":
            continue
        nxt = blocks[i + 1]
        if _leads_into_the_scan(nxt) and nxt.block_type == BlockType.IDLE and i + 2 < len(blocks):
            nxt = blocks[i + 2]
        if nxt.block_type == BlockType.SCIENCE:
            boundary.append((b, nxt))
    if name != "deep56_seeded_hourly_retune":
        assert boundary, f"{name}: no boundary retune"
    for retune, science in boundary:
        assert (retune.az_start, retune.az_end) == (science.az_start, science.az_end), name


def test_the_night_validates_tiles_and_rebuilds(night):
    """The night validates clean, tiles its window and every science block rebuilds."""
    _, timeline, _, pairs, _ = night
    assert timeline.validate() == []
    total = sum(b.duration for b in timeline.blocks)
    assert total == pytest.approx(timeline.total_time, abs=0.01)
    assert len(pairs) == len(timeline.science_blocks)


def test_pong_and_daisy_start_at_the_field_centre_at_their_start(night):
    """A rebuilt pong or daisy starts on the field centre at its own start time.

    The slew onto such a visit targets the field centre at the scan's
    start without building the pattern, which rests on this. A pong's
    first sample is the centre itself. A daisy's is one integrator step
    from rest away from it (the start acceleration times the step
    squared, along the pattern's x axis): measured at most 3.1e-5 deg in
    azimuth at el 78 and 2.2e-5 deg in elevation on the pong and daisy
    day, so it is held to 1e-4 deg. With a detector offset the centre is
    where the detector starts, and the first sample is the boresight that
    puts it there, solved as the planners solve it.
    """
    _, timeline, _, pairs, _ = night
    coords = Coordinates(timeline.site)
    for block, scan in pairs:
        if block.scan_type not in ("pong", "daisy"):
            continue
        tol = 1e-6 if block.scan_type == "pong" else 1e-4
        traj = scan.trajectory
        az, el = coords.radec_to_altaz(
            block.metadata["ra_center"], block.metadata["dec_center"], traj.start_time
        )
        offset = block.metadata.get("detector_offset")
        if offset is not None:
            az, el = _solve_boresight(
                float(az), float(el), InstrumentOffset(**offset), site=timeline.site
            )
        assert abs(_on_turn_of(float(az), float(traj.az[0])) - float(traj.az[0])) <= tol
        assert abs(float(el) - float(traj.el[0])) <= tol


def test_a_subscan_records_where_it_ends_in_both_axes(night):
    """Every science block's end pose is its rebuilt trajectory's last sample.

    A pong or daisy is stamped at the field centre's elevation where its
    visit starts and follows the field, so it records the last sample's
    elevation as ``el_final``, and the block after it starts there: a
    retune before the next subscan, a calibration or an idle is stamped at
    that elevation, and a slew leaves from its azimuth. A
    constant-elevation block records none, so its end pose is the
    elevation it is stamped at.
    """
    name, timeline, rebuilt, _, _ = night
    blocks = list(timeline.blocks)
    off_stamp = 0
    for i, block in enumerate(blocks):
        if block.block_type != BlockType.SCIENCE:
            continue
        where = f"{name}: {block.patch_name} at {block.t_start.isot}"
        if block.scan_type not in ("pong", "daisy"):
            assert block.el_final is None, where
            assert block.end_pose_el == block.elevation, where
            continue
        off_stamp += 1
        traj = rebuilt[id(block)]
        az_last = _on_turn_of(float(traj.az[-1]), block.end_pose_az)
        assert block.end_pose_az == pytest.approx(az_last, abs=1e-9), where
        assert block.end_pose_el == pytest.approx(float(traj.el[-1]), abs=1e-9), where
        if i + 1 == len(blocks):
            continue
        after = blocks[i + 1]
        if after.block_type == BlockType.SLEW:
            assert after.az_start == pytest.approx(block.end_pose_az, abs=1e-9), where
        else:
            # A retune before the next subscan, a calibration or an idle:
            # each holds the pose the scan left.
            assert after.block_type != BlockType.SCIENCE, where
            assert after.elevation == pytest.approx(block.end_pose_el, abs=1e-9), where
    if name in ("pong_and_daisy", "quickstart"):
        assert off_stamp, f"{name}: no pong or daisy block"


#: The pong and daisy day as the scheduler books it, re-run under pytest:
#: no Sun escape (each visit's end pose is the one its scan ends at, clear
#: of the Sun), the science seconds and the block count.
_PONG_AND_DAISY_DAY = {"escapes": 0, "science_s": 34375.29, "blocks": 55}


@pytest.mark.parametrize("night", ["pong_and_daisy"], indirect=True)
def test_the_pong_and_daisy_day_escapes_from_no_pose_it_never_held(night):
    """The day books no Sun escape: each visit leaves the telescope where its scan ended.

    The daisy F3 visit from 13:03:49 starts at el 78.43 and its scan ends at
    el 65.79, where the visit leaves the telescope, 57.0 deg from the Sun
    and outside the 45 deg zone; the afternoon's later F3 visits end clear
    of it the same way.
    """
    _, timeline, _, _, _ = night
    escapes = [b for b in timeline.blocks if b.patch_name == "sun_escape"]
    science = sum(b.duration for b in timeline.science_blocks)
    assert len(escapes) == _PONG_AND_DAISY_DAY["escapes"]
    assert science == pytest.approx(_PONG_AND_DAISY_DAY["science_s"], abs=0.05)
    assert len(timeline.blocks) == _PONG_AND_DAISY_DAY["blocks"]


@pytest.mark.parametrize("scan_type", ["pong", "daisy"])
def test_the_scheduler_leaves_a_pattern_visit_at_its_last_sample(scan_type):
    """After a pong or daisy visit the scheduler's pose is the scan's last sample.

    The field sets through the hour the visit lasts, so the pattern ends
    degrees below the elevation the visit is stamped at; the next move,
    calibration and idle start from where the scan ended.
    """
    from fyst_trajectories.overhead.simulation import _generate_trajectory_for_block

    patch = _pattern("Wide01", 180.0, -30.0, 4.0, scan_type)
    ctx = _make_ctx(patches=[patch])
    state = _initial_state(ctx)
    selection = PatchSelectionPhase().run(state, ctx)
    slew = SlewPhase().run(state, ctx, selection=selection)
    result = ScienceScanPhase().run(slew.state, ctx, selection=slew)

    science = [b for b in result.blocks if b.block_type == BlockType.SCIENCE]
    assert science
    last = science[-1]
    traj = _generate_trajectory_for_block(last, ctx.site).trajectory
    az_last = _on_turn_of(float(traj.az[-1]), last.end_pose_az)
    el_last = float(traj.el[-1])
    assert abs(el_last - last.elevation) > 1.0
    assert last.el_final == pytest.approx(el_last, abs=1e-9)
    assert (last.end_pose_az, last.end_pose_el) == pytest.approx((az_last, el_last), abs=1e-9)
    assert (result.state.current_az, result.state.current_el) == pytest.approx(
        (az_last, el_last), abs=1e-9
    )


def test_a_constant_elevation_visit_records_no_end_elevation():
    """A constant-elevation pass ends at the elevation it is stamped at."""
    patch = _DEEP56_TEST
    ctx = _ce_ready_ctx(patch)
    state = _initial_state(ctx)
    selection = PatchSelectionPhase().run(state, ctx)
    slew = SlewPhase().run(state, ctx, selection=selection)
    result = ScienceScanPhase().run(slew.state, ctx, selection=slew)

    science = [b for b in result.blocks if b.block_type == BlockType.SCIENCE]
    assert science
    assert all(b.el_final is None for b in science)
    assert all(b.end_pose_el == patch.elevation for b in science)
    assert result.state.current_el == patch.elevation


def _ce_ready_ctx(patch, lead=300.0):
    """Return a context from ``lead`` s before the patch's rising pass to an hour after it."""
    from astropy.time import Time

    from fyst_trajectories.overhead.scheduler.helpers import _ce_crossing_corridor

    coords = Coordinates(get_fyst_site())
    t_open, _ = _ce_crossing_corridor(
        patch, patch.elevation, True, Time("2026-06-15T02:00:00", scale="utc"), coords, {}
    )
    start = t_open - TimeDelta(lead, format="sec")
    end = t_open + TimeDelta(3600.0, format="sec")
    return _make_ctx(patches=[patch], start_time=start.isot, end_time=end.isot)


class TestFallback:
    """A visit whose scan cannot be planned at slew time slews and idles as before."""

    def test_a_refused_first_subscan_keeps_the_field_centre_target(self, monkeypatch):
        """The probe refuses: the slew targets the field centre at the pinned elevation."""

        def refuse(block, site):
            raise PointingError("refused for the test")

        monkeypatch.setattr(phases, "_generate_trajectory_for_block", refuse)
        ctx = _ce_ready_ctx(_DEEP56_TEST)
        state = _initial_state(ctx)
        selection = PatchSelectionPhase().run(state, ctx)
        slew = SlewPhase().run(state, ctx, selection=selection)

        (block,) = slew.blocks
        assert block.az_end == selection.best_az
        assert block.elevation == _DEEP56_TEST.elevation
        science = ScienceScanPhase().run(slew.state, ctx, selection=slew)
        assert [str(b.block_type) for b in science.blocks] == ["idle"]
        assert science.blocks[0].metadata["reason"] == str(DeferralReason.UNPLANNABLE)

    def test_a_refused_first_subscan_with_an_offset_targets_the_boresight(self, monkeypatch):
        """With ``IM6`` the fallback target is the boresight that holds the detector there.

        The detector looks at the field centre's azimuth at the patch's
        elevation; the slew ends at the boresight the planners' solve gives
        for that pose, 1.66 deg higher.
        """

        def refuse(block, site):
            raise PointingError("refused for the test")

        monkeypatch.setattr(phases, "_generate_trajectory_for_block", refuse)
        patch = dataclasses.replace(_DEEP56_TEST, detector_offset=get_primecam_offset("IM6"))
        ctx = _ce_ready_ctx(patch)
        state = _initial_state(ctx)
        selection = PatchSelectionPhase().run(state, ctx)
        slew = SlewPhase().run(state, ctx, selection=selection)

        (block,) = slew.blocks
        centre_az, _ = ctx.coords.radec_to_altaz(patch.ra_center, patch.dec_center, ctx.start_time)
        bore_az, bore_el = _solve_boresight(
            float(centre_az), patch.elevation, patch.detector_offset, site=ctx.site
        )
        assert block.elevation == pytest.approx(float(bore_el), abs=1e-9)
        assert block.elevation - patch.elevation > 1.6
        turn = 360.0 * round((block.az_end - float(bore_az)) / 360.0)
        assert block.az_end == pytest.approx(float(bore_az) + turn, abs=1e-9)

    def test_a_night_whose_scans_are_refused_still_tiles(self, monkeypatch):
        """Every first subscan refused: the window is still tiled and validates."""

        def refuse(block, site):
            raise PointingError("refused for the test")

        monkeypatch.setattr(phases, "_generate_trajectory_for_block", refuse)
        # Long enough a lead for the startup calibrations to finish before
        # the pass is selectable.
        ctx = _ce_ready_ctx(_DEEP56_TEST, lead=1800.0)
        timeline = Scheduler(ctx).run()

        assert not timeline.science_blocks
        assert any(
            b.metadata.get("reason") == str(DeferralReason.UNPLANNABLE) for b in timeline.blocks
        )
        assert timeline.validate() == []
        total = sum(b.duration for b in timeline.blocks)
        assert total == pytest.approx(timeline.total_time, abs=0.01)

    @pytest.mark.parametrize("refused_call", [0, 1])
    def test_a_refused_transition_idles_with_its_reason(self, monkeypatch, refused_call):
        """A refusal of the first plan, or of a later one, idles the tick.

        The first transition is planned to the scan's first sample for a
        visit starting at the tick, the next to the one for a visit starting
        at that slew's arrival; a refusal of either leaves the telescope
        where it is for one tick, labelled with the refusal.
        """
        original = phases.plan_transition
        calls = []

        def refuse_one(*args, **kwargs):
            calls.append(args)
            planned = original(*args, **kwargs)
            if len(calls) - 1 != refused_call:
                return planned
            return Transition(
                az_from=planned.az_from,
                el_from=planned.el_from,
                az_to=args[2],
                el_to=args[3],
                t_start=planned.t_start,
                duration=0.0,
                cause=DeferralReason.SUN_PATH,
            )

        monkeypatch.setattr(phases, "plan_transition", refuse_one)
        ctx = _ce_ready_ctx(_DEEP56_TEST)
        state = _initial_state(ctx)
        selection = PatchSelectionPhase().run(state, ctx)
        result = SlewPhase().run(state, ctx, selection=selection)

        assert len(calls) == refused_call + 1
        assert [str(b.block_type) for b in result.blocks] == ["idle"]
        idle = result.blocks[0]
        assert idle.metadata["reason"] == str(DeferralReason.SUN_PATH)
        assert (idle.az_start, idle.elevation) == (state.current_az, state.current_el)
        assert result.skip_to_next_iter is True


@pytest.mark.parametrize("scan_type", ["pong", "daisy"])
def test_a_pattern_whose_first_subscan_the_planner_refuses_is_not_slewed_to(monkeypatch, scan_type):
    """The planner refuses a pong's or daisy's shortest first subscan: the tick idles unmoved.

    That subscan is the last one the science phase tries, built from the same
    start, so a visit whose probe the planner refuses books nothing: the slew
    phase idles the tick at the unmoved pose with ``unplannable`` and books
    no slew.
    """
    patch = _pattern("Wide01", 180.0, -30.0, 4.0, scan_type)
    ctx = _make_ctx(patches=[patch])
    state = _initial_state(ctx)
    selection = PatchSelectionPhase().run(state, ctx)
    assert selection.selection is patch
    # The premise: with a plannable scan the visit is slewed to.
    slewed = SlewPhase().run(state, ctx, selection=selection)
    assert [b.patch_name for b in slewed.blocks] == ["slew_to_Wide01"]

    def refuse(block, site):
        raise PointingError("refused for the test")

    monkeypatch.setattr(phases, "_generate_trajectory_for_block", refuse)
    result = SlewPhase().run(state, ctx, selection=selection)

    assert [str(b.block_type) for b in result.blocks] == ["idle"]
    idle = result.blocks[0]
    assert idle.metadata["reason"] == str(DeferralReason.UNPLANNABLE)
    assert (idle.az_start, idle.elevation) == (state.current_az, state.current_el)
    assert (result.state.current_az, result.state.current_el) == (
        state.current_az,
        state.current_el,
    )
    assert result.skip_to_next_iter is True


def test_a_trial_start_reads_the_selected_pass_and_asks_the_gate_nothing(monkeypatch):
    """Reading the first sample for a trial visit start uses the pass the selection admitted.

    The slew reads the scan's first sample for visit starts that may never
    be booked. Each read builds the pass the selection gate admitted, from
    the anchor it solved it from, so a trial start past the pass opening
    neither re-solves the pass nor writes to the corridor memo, where a
    miss recorded there would be trusted for the next 600 s of scheduler
    time and the tick that selected the patch would lose its pass.
    """
    ctx = _ce_ready_ctx(_DEEP56_TEST)
    state = _initial_state(ctx)
    selection = PatchSelectionPhase().run(state, ctx)
    assert selection.selection is _DEEP56_TEST
    anchor, rising, t_open, _ = selection.ce_plan
    assert anchor == state.current_time
    assert rising is True
    before = dict(ctx.ce_corridors)

    def no_gate(*args, **kwargs):
        raise AssertionError("the gate was asked again")

    monkeypatch.setattr(phases, "_ce_visit_plan", no_gate)
    pose = phases._scan_start_pose(
        _DEEP56_TEST,
        t_open + TimeDelta(120.0, format="sec"),
        elevation=_DEEP56_TEST.elevation,
        wrap_ref=selection.best_az,
        ce_plan=selection.ce_plan,
        state=state,
        ctx=ctx,
    )

    assert pose is not None
    assert ctx.ce_corridors == before


#: A constant-elevation field 100 deg wide whose pass at el 45 on 2026-03-20
#: sweeps az 256-272 (on the turn nearest the parked pose at az 180), far
#: from the azimuth of the field's centre at the tick (az 63.4).
_WIDE = _ce("W100", 200.0, -20.0, 100.0, 10.0, 45.0)


def _wide_night(start, end, monkeypatch):
    """Run the wide field over the window; return the night, its rebuilds and its slews.

    The slews are given as the last transition planned at each start time,
    the one a slew is booked from.
    """
    plan = phases.plan_transition
    booked = {}

    def recording(*args, **kwargs):
        transition = plan(*args, **kwargs)
        booked[transition.t_start.isot] = transition
        return transition

    monkeypatch.setattr(phases, "plan_transition", recording)
    timeline = generate_timeline(
        patches=[_WIDE], site=get_fyst_site(), start_time=start, end_time=end
    )
    with warnings.catch_warnings():
        # The field is 100 deg wide, so its crossing lasts hours; the
        # rebuild's long-duration advisory describes the field.
        warnings.filterwarnings(
            "ignore", message="Computed observation duration", category=PointingWarning
        )
        pairs = schedule_to_trajectories(timeline)
    return timeline, {id(block): scan.trajectory for block, scan in pairs}, booked


class TestWideConstantElevationField:
    """A pass far from its field's centre is slewed to on the wrap that holds its sweep.

    The width estimate about the field's centre is not where such a pass
    scans, so the wrap the slew takes is chosen on the pass's swept
    envelope, and each block records its range on the telescope's own
    wrap. The base figures are those of a scheduler that slewed to the
    field's centre: on the short window one 45.87 s slew and 974.13 s of
    science.
    """

    def _visits(self, timeline, rebuilt, booked):
        """Check every visit; return the wrap shifts of the transitions their slews took."""
        blocks = list(timeline.blocks)
        shifts = []
        for i, science in enumerate(blocks):
            if science.block_type != BlockType.SCIENCE or science.subscan_index != 0:
                continue
            j = i - 1
            while _leads_into_the_scan(blocks[j]):
                j -= 1
            slew = blocks[j]
            assert slew.block_type == BlockType.SLEW
            traj = rebuilt[id(science)]
            az0 = _on_turn_of(float(traj.az[0]), slew.az_end)
            assert abs(slew.az_end - az0) <= _POSE_TOL_DEG
            assert science.az_start - 1e-9 <= slew.az_end <= science.az_end + 1e-9
            # The rebuild is placed on the turn the block records.
            assert float(traj.az.min()) == pytest.approx(science.az_start, abs=1e-6)
            for retune in blocks[j + 1 : i]:
                if retune.block_type == BlockType.CALIBRATION:
                    assert (retune.az_start, retune.az_end) == (science.az_start, science.az_end)
            shifts.append(booked[slew.t_start.isot].az_shift)
        return shifts

    def test_the_visit_is_booked_on_the_wrap_its_sweep_fits(self, monkeypatch):
        """The short window books the visit; only the slew's change moves its science time."""
        timeline, rebuilt, booked = _wide_night(
            "2026-03-20T05:20:00", "2026-03-20T06:00:00", monkeypatch
        )
        assert timeline.validate() == []
        assert len(timeline.science_blocks) == 1
        # The pass's first sample is read a turn below the one that holds
        # its sweep nearest the parked pose, so the slew takes the turn up.
        assert self._visits(timeline, rebuilt, booked) == [360.0]
        # The window's end cuts the visit, so its science time moves by
        # what its slew (move and hold) gained on the base's.
        assert timeline.total_science_time - 974.13 == pytest.approx(
            45.87 - timeline.total_slew_time, abs=0.01
        )

    @pytest.mark.slow
    def test_the_whole_day_keeps_its_visits(self, monkeypatch):
        """Over the day no visit idles for want of a wrap and both visits scan from their openings.

        The day keeps its 217 blocks. Each of its two visits slews, retunes and waits at its pass's
        first sample, so its science starts when its pass opens (05:42:16.3
        and 23:11:23.2), and the day holds 26309.45 s of science.
        """
        timeline, rebuilt, booked = _wide_night(
            "2026-03-20T00:00:00", "2026-03-21T00:00:00", monkeypatch
        )
        assert timeline.validate() == []
        assert any(shift != 0.0 for shift in self._visits(timeline, rebuilt, booked))
        assert not any(b.metadata.get("reason") == "no_wrap" for b in timeline.blocks)
        assert len(timeline.blocks) == 217
        firsts = [b for b in timeline.science_blocks if b.subscan_index == 0 and id(b) in rebuilt]
        assert len(firsts) == 2
        for science in firsts:
            first = get_absolute_times(rebuilt[id(science)])[0]
            assert abs((first - science.t_start).sec) <= 1e-3
        assert timeline.total_science_time == pytest.approx(26309.45, abs=0.02)


def _transition(az, el, tick, duration, cause):
    return Transition(
        az_from=180.0,
        el_from=50.0,
        az_to=az,
        el_to=el,
        t_start=tick,
        duration=duration,
        cause=cause,
    )


class TestSlewOntoScanStart:
    """The solve on synthetic plans and poses: arrive, hold, start on time."""

    TICK = Time("2026-06-15T08:00:00", scale="utc")
    FALLBACK = (99.0, 40.0)

    def _since(self, t):
        return (t - self.TICK).sec

    def _solve(self, pose, duration_of, refuse_at=None):
        """Run the solve; ``duration_of(az)`` prices a move, ``refuse_at`` is a plan to refuse."""
        calls = []

        def plan(az, el):
            calls.append((az, el))
            cause = DeferralReason.SUN_PATH if len(calls) == refuse_at else DeferralReason.OK
            return _transition(az, el, self.TICK, duration_of(az), cause)

        result = SlewPhase._slew_onto_scan_start(plan, self.FALLBACK, pose, self.TICK)
        return result, calls

    def test_on_time_in_the_first_round_starts_at_the_tick(self):
        """A move of 1 s or less counts as arriving at the tick: nothing is held."""
        (transition, start, target), calls = self._solve(lambda v: (10.0, 50.0), lambda az: 1.0)
        assert calls == [(10.0, 50.0)]
        assert start == self.TICK
        assert transition.duration == 1.0
        assert target == (10.0, 50.0)

    def test_a_move_just_over_a_second_starts_the_visit_at_its_arrival(self):
        """A 1.5 s move is late at the tick, so the visit starts when it arrives."""
        (transition, start, target), calls = self._solve(lambda v: (10.0, 50.0), lambda az: 1.5)
        assert len(calls) == 2
        assert self._since(start) == pytest.approx(1.5)
        assert transition.duration == pytest.approx(1.5)

    def test_late_then_early_books_the_move_and_the_hold(self):
        """The second slew arrives early and waits; the block ends at the visit start."""

        def pose(v):
            return (10.0, 50.0) if v == self.TICK else (5.0, 50.0)

        (transition, start, target), calls = self._solve(pose, lambda az: 2.0 * az)
        assert calls == [(10.0, 50.0), (5.0, 50.0)]
        # The first slew (20 s) sets the visit start; the second (10 s) holds 10 s.
        assert self._since(start) == pytest.approx(20.0)
        assert transition.duration == pytest.approx(self._since(start))
        assert transition.az_to == 5.0
        assert target == (5.0, 50.0)

    def test_a_short_move_in_a_later_round_holds_until_the_visit_starts(self):
        """Under 1 s in the second round: the block is the hold, not nothing.

        The first slew (30 s) starts the visit 30 s after the tick; the
        second, to a target 0.15 deg away, is a 0.3 s move, which counts as
        arriving at the tick and so in time, and the block booked holds the
        pose until the visit starts.
        """

        def pose(v):
            return (15.0, 50.0) if v == self.TICK else (0.15, 50.0)

        (transition, start, target), calls = self._solve(pose, lambda az: 2.0 * az)
        assert calls == [(15.0, 50.0), (0.15, 50.0)]
        assert self._since(start) == pytest.approx(30.0)
        assert transition.duration == pytest.approx(30.0)
        assert transition.az_to == 0.15
        assert target == (0.15, 50.0)

    def test_a_drifting_target_converges_to_the_arrival_tolerance(self):
        """The booked slew ends at the visit start, at the target read for that start."""

        def pose(v):
            return (10.0 + 0.1 * self._since(v), 50.0)

        (transition, start, target), calls = self._solve(pose, lambda az: 2.0 * az)
        assert 2 < len(calls) < phases._SLEW_MAX_PLANS
        assert transition.arrival.unix == pytest.approx(start.unix, abs=1e-6)
        assert target == pose(start)
        # The move the slew was priced at lies within the tolerance of its booked end.
        assert 2.0 * target[0] - self._since(start) <= _ARRIVAL_TOL_SEC

    @pytest.mark.parametrize("unplannable_round", [1, 2])
    def test_an_unplannable_scan_slews_to_the_fallback(self, unplannable_round):
        """The scan cannot be planned in the first round or a later one: the fallback is booked."""

        def pose(v):
            return None if unplannable_round == 1 or v > self.TICK else (10.0, 50.0)

        (transition, start, target), calls = self._solve(pose, lambda az: 2.0 * az)
        assert len(calls) == unplannable_round
        assert calls[-1] == self.FALLBACK
        assert target == self.FALLBACK
        assert self._since(start) == pytest.approx(198.0)
        assert transition.duration == pytest.approx(198.0)

    def test_a_refusal_in_a_later_round_is_returned_at_the_tick(self):
        """The second plan is refused: it is returned for the caller to idle the tick on."""

        def pose(v):
            return (10.0 + 0.1 * self._since(v), 50.0)

        (transition, start, target), calls = self._solve(pose, lambda az: 2.0 * az, refuse_at=2)
        assert len(calls) == 2
        assert transition.cause is DeferralReason.SUN_PATH
        assert start == self.TICK
        assert target == calls[-1]

    def test_the_cap_books_the_last_plan_late(self):
        """A target that runs away is planned at most the cap; the last plan is booked as it is.

        The visit then starts when that slew arrives, and the target returned
        is the one the slew was planned to: read for the previous visit start.
        """

        def pose(v):
            return (10.0 + self._since(v), 50.0)

        (transition, start, target), calls = self._solve(pose, lambda az: 2.0 * az)
        assert len(calls) == phases._SLEW_MAX_PLANS
        assert target == calls[-1]
        assert transition.az_to == target[0]
        assert transition.duration == pytest.approx(2.0 * target[0])
        assert start == transition.arrival
        assert target != pose(start)
