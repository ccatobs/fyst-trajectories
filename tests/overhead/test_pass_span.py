"""Every simulated block spans the samples it records, and holds the pose it was left at.

A science block starts at the first sample of the scan it records and stops
within one sample interval of its last; the rebuild cuts a
constant-elevation block to its own window, so the last block of a visit is
also held against the pass the planner returns from the visit's anchor, uncut:
it ends one sample interval after that pass's last sample, unless the night
ends first. A constant-elevation visit is booked as the slew onto its scan,
the retune due where its first subscan would start (the later of the slew's
end and the pass opening), an idle labelled ``waiting_for_pass`` when the pass
opens after the retune, and then science from the pass's first sample. A
visit admitted by the readiness lead alone, one scheduler tick, a slew
allowance and that retune, starts its scan at its pass's first sample.

A planet-calibration pass block holds its pass's samples the same way, and
the time before each pass, after the slew onto the planet or after the
previous pass, is an idle labelled ``waiting_for_pass`` at the pose the
telescope was left at. Every block that holds a pose, an idle or a
calibration that does not scan, holds the one the block before it ended at.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import warnings

import numpy as np
import pytest
from _scheduler_helpers import _initial_state, _make_ctx
from astropy.time import Time, TimeDelta

from fyst_trajectories import InstrumentOffset, get_absolute_times, get_fyst_site
from fyst_trajectories.exceptions import PointingWarning
from fyst_trajectories.overhead import (
    BlockType,
    CalibrationPolicy,
    DeferralReason,
    ObservingPatch,
    OverheadModel,
    generate_timeline,
    schedule_to_trajectories,
)
from fyst_trajectories.overhead.scheduler import SchedulerContext, phases
from fyst_trajectories.overhead.scheduler.helpers import _CE_READY_SLEW_ALLOWANCE_SEC
from fyst_trajectories.overhead.simulation import _EDGE_TOLERANCE_SEC
from fyst_trajectories.overhead.utils import _stored_instant
from fyst_trajectories.patterns.turnarounds import turnaround_duration_sec
from fyst_trajectories.planning import FieldRegion, plan_constant_el_scan

from ._nights import (
    CES,
    DEEP56,
    NIGHTS,
    REGRESSION_POLICY,
    WIDE_SLOW_CES,
    ZERO_OVERHEAD,
    night,
    rebuilt,
    rebuilt_from_file,
)

#: The timeline's block-time tolerance, in seconds.
_TOL = 0.01

_WAITING = str(DeferralReason.WAITING_FOR_PASS)

#: Nights of this module alone: patches, window and the keyword arguments of
#: generate_timeline.
_LOCAL_NIGHTS = {
    # Deep56 with a retune due every 600 s and a 60 s tick.
    "retune_every_600_s": (
        [DEEP56],
        "2026-06-15T02:00:00",
        "2026-06-15T10:00:00",
        {
            "time_step": 60.0,
            "calibration_policy": dataclasses.replace(REGRESSION_POLICY, retune_cadence=600.0),
        },
    ),
    # Deep56 with a retune due every 7000 s from the night's start, at
    # 07:58:20, after the slew onto the visit and before its pass opens.
    "retune_due_in_the_wait": (
        [DEEP56],
        "2026-06-15T06:01:40",
        "2026-06-15T09:30:00",
        {"time_step": 300.0, "calibration_policy": CalibrationPolicy(retune_cadence=7000.0)},
    ),
    # The August field 20 deg wide and swept at 0.5 deg/s, whose pass ends
    # after its closing crossing.
    "narrow_slow_ce": (
        [dataclasses.replace(CES, name="NarrowSlowCES", width=20.0, velocity=0.5)],
        "2026-08-02T01:45:00",
        "2026-08-02T04:45:00",
        {},
    ),
}


@functools.cache
def _timeline(name):
    """Return the named night, from the shared nights or this module's own."""
    if name in NIGHTS:
        return night(name)
    patches, start, end, kwargs = _LOCAL_NIGHTS[name]
    return generate_timeline(
        patches=patches, site=get_fyst_site(), start_time=start, end_time=end, **kwargs
    )


@functools.cache
def _rebuilt(name):
    """Return the named night's rebuilt blocks, calibration passes included."""
    if name in NIGHTS:
        return rebuilt(name)
    return tuple(schedule_to_trajectories(_timeline(name), science_only=False))


def _step(trajectory) -> float:
    """Return a trajectory's sample interval, in seconds."""
    return float(trajectory.times[1] - trajectory.times[0])


def _ce_visits(timeline):
    """Return each constant-elevation visit as ``(lead, science)``.

    ``science`` is the visit's first science block and ``lead`` the blocks
    from the slew onto the visit up to it.
    """
    blocks = list(timeline.blocks)
    visits = []
    for i, block in enumerate(blocks):
        if block.block_type != BlockType.SCIENCE or block.scan_type != "constant_el":
            continue
        if block.subscan_index != 0:
            continue
        j = i - 1
        while blocks[j].block_type not in (BlockType.SLEW, BlockType.SCIENCE):
            j -= 1
        assert blocks[j].block_type == BlockType.SLEW, f"no slew onto {block.t_start.isot}"
        visits.append((blocks[j:i], block))
    return visits


def _ce_visit_blocks(timeline):
    """Return the science blocks of each constant-elevation visit, in order."""
    visits: dict[int, list] = {}
    for block in timeline.science_blocks:
        if block.scan_type == "constant_el":
            visits.setdefault(block.scan_index, []).append(block)
    return list(visits.values())


def _admitted_by_the_lead_alone(timeline, lead, opens):
    """Whether the readiness lead, and nothing before the slew, set when the visit starts.

    It does when the block before the slew is an ordinary idle tick, since
    the scheduler then had the chance to admit the visit one tick earlier,
    or when a calibration ends at the slew's start with the slew allowance
    and the visit's retune still left before the pass opens at ``opens``.
    """
    blocks = list(timeline.blocks)
    before = blocks[blocks.index(lead[0]) - 1]
    if before.block_type == BlockType.IDLE:
        return "reason" not in before.metadata
    if before.block_type != BlockType.CALIBRATION:
        return False
    retune = sum(b.duration for b in lead if b.block_type == BlockType.CALIBRATION)
    room = (opens - lead[0].t_start).sec
    return room >= _CE_READY_SLEW_ALLOWANCE_SEC + retune - _TOL


def _first_sample(name, block):
    """Return the rebuilt first sample of ``block`` on the named night, and the scan."""
    for rebuilt_block, scan in _rebuilt(name):
        if rebuilt_block is block:
            return get_absolute_times(scan.trajectory)[0], scan
    raise AssertionError(f"{block.t_start.isot} did not rebuild")


def _planned_pass(block):
    """Return the sample times and interval of a constant-elevation block's whole pass.

    The pass is planned with :func:`~fyst_trajectories.planning.plan_constant_el_scan`
    from the block's recorded anchor ``t0_scan``, with the block's field,
    speed, crossing half (or recorded ``lsa_window``, which sets the half
    itself) and recorded planner options, for the detector the block
    records (``detector_offset``) at the elevation the detector scans at
    (``detector_elevation``, or the block's own ``elevation`` with no
    offset), and is not cut to the block's window. A block that records a
    window scan (``window_scan``) is a subscan of a visit timed by this
    pass, the field's crossing of the record's elevation on the block's
    side, while it sweeps its window; the pass is planned from the record's
    elevation and offset.
    """
    meta = block.metadata
    params = meta["scan_params"]
    lsa_window = params.get("lsa_window")
    offset = meta.get("detector_offset")
    record = meta.get("window_scan")
    elevation = meta.get("detector_elevation", block.elevation)
    if record is not None:
        elevation, offset = record["elevation"], record["detector_offset"]
    with warnings.catch_warnings():
        # The planner screens the field centre at the anchor and the whole
        # crossing for the Sun; that advisory describes the pass, not this check.
        warnings.filterwarnings("ignore", message="EXCLUSION ZONE: ", category=PointingWarning)
        scan = plan_constant_el_scan(
            field=FieldRegion(
                ra_center=meta["ra_center"],
                dec_center=meta["dec_center"],
                width=meta["width"],
                height=meta["height"],
            ),
            elevation=elevation,
            velocity=meta["velocity"],
            site=get_fyst_site(),
            start_time=Time(meta["t0_scan"], scale="utc"),
            rising=None if lsa_window is not None else block.rising,
            lsa_window=lsa_window,
            detector_offset=None if offset is None else InstrumentOffset(**offset),
            **{key: params[key] for key in ("az_accel", "timestep", "az_padding") if key in params},
        )
    return get_absolute_times(scan.trajectory), _step(scan.trajectory)


@pytest.mark.parametrize("source", ["memory", "file"])
@pytest.mark.parametrize("name", sorted(NIGHTS))
def test_every_science_block_spans_its_samples(name, source):
    """A science block starts at its first sample and stops within one sample interval of its last.

    Both ends are held to one sample interval of the block's own scan plus
    the timeline's 0.01 s tolerance, measured from the block's times as a
    file stores them, which is how the rebuild reads them; a sample on an
    edge may lie up to the rebuild's edge tolerance on either side of the
    stored edge. A constant-elevation subscan is cut on the samples of its
    pass, its half-open window keeping the sample at its start and leaving
    the one at its stop to the next block; a pong of whole periods keeps its
    own last sample, which sits on its stop. The rebuild cuts each block to its own window,
    so this does not see a visit that stops before its pass does; the next
    test does.
    """
    timeline = night(name)
    checked = 0
    edge = _EDGE_TOLERANCE_SEC
    for block, scan in rebuilt(name) if source == "memory" else rebuilt_from_file(name):
        if block.block_type != BlockType.SCIENCE:
            continue
        checked += 1
        times = get_absolute_times(scan.trajectory)
        step = _step(scan.trajectory)
        head = (times[0] - _stored_instant(block.t_start)).sec
        tail = (_stored_instant(block.t_stop) - times[-1]).sec
        where = f"{name}: {block.patch_name} {block.t_start.isot}-{block.t_stop.isot}"
        assert -edge <= head <= step + _TOL, f"{where}: first sample {head:.3f} s after the start"
        assert -edge <= tail <= step + _TOL, f"{where}: last sample {tail:.3f} s before the stop"
    assert checked == len(timeline.science_blocks)


@pytest.mark.parametrize("name", sorted(NIGHTS))
def test_every_block_rebuilds_the_same_from_memory_and_from_a_file(name):
    """A block read back from ECSV rebuilds to the very arrays it rebuilds to in memory.

    The rebuild reads a block's times as the file stores them, to the
    millisecond, so the two paths give it the same inputs.
    """
    memory, from_file = rebuilt(name), rebuilt_from_file(name)
    assert len(memory) == len(from_file)
    for (block, scan), (read, scan_read) in zip(memory, from_file, strict=True):
        where = f"{name}: {block.patch_name} {block.t_start.isot}"
        assert read.t_start.isot == block.t_start.isot, where
        want, got = scan.trajectory, scan_read.trajectory
        assert (got.start_time.jd1, got.start_time.jd2) == (
            want.start_time.jd1,
            want.start_time.jd2,
        ), where
        for attr in ("times", "az", "el", "scan_flag"):
            assert np.array_equal(getattr(got, attr), getattr(want, attr)), (where, attr)


#: The constant-elevation visits whose pass the night's end cuts, by night,
#: as their patch names in order; every other visit ends with its pass.
_CUT_BY_THE_NIGHTS_END = {
    "august": ["AugustCES"],
    "ce_offset": ["Deep56IM6"],
    "default_overhead": ["CES"],
    "planet_scan": ["Deep56"],
    "quickstart": ["Deep56"],
    "regression": ["Deep56"],
    "time_based_retune": ["Deep56"],
    "window_rising": ["Deep56WindowIM6"],
    "window_setting_day": ["Deep56Window"],
    "window_cut": ["Deep56Window"],
    "zero_overhead": ["CES"],
    "retune_every_600_s": ["Deep56"],
    "retune_due_in_the_wait": ["Deep56"],
}


@pytest.mark.parametrize("name", [*sorted(NIGHTS), *_LOCAL_NIGHTS])
def test_a_constant_elevation_visit_ends_one_interval_after_its_pass(name):
    """A visit's last block stops one sample interval after its pass's last sample.

    The pass is the planner's own from the visit's anchor, not cut to any
    block (:func:`_planned_pass`). The planner times a pass as a whole number
    of legs, so it can end before or after the closing crossing, by up to
    half a leg; the passes of the two slowly swept fields end after it. A
    visit the night's end cuts stops at the night's end instead; which
    visits those are is listed per night. No visit here is cut by the Sun.

    A visit to a patch with an azimuth window holds whole window scans, so
    it stops no later than that end and less than one more leg and
    turnaround of its window, plus one sample interval, before it.
    """
    timeline = _timeline(name)
    cut = []
    for visit in _ce_visit_blocks(timeline):
        times, step = _planned_pass(visit[0])
        pass_stop = times[-1] + TimeDelta(step, format="sec")
        expected = pass_stop
        if pass_stop.unix > timeline.end_time.unix + _TOL:
            cut.append(visit[0].patch_name)
            expected = timeline.end_time
        last = visit[-1]
        where = f"{name}: {last.patch_name} visit at {visit[0].t_start.isot}"
        early = (expected - last.t_stop).sec
        record = last.metadata.get("window_scan")
        if record is not None:
            velocity = last.metadata["velocity"]
            lo, hi = record["az_window"]
            leg = (hi - lo) / velocity + turnaround_duration_sec(velocity, record["az_accel"])
            assert -_TOL <= early < leg + record["timestep"] + _TOL, (
                f"{where}: stops {early:.3f} s before {expected.isot}, one leg is {leg:.3f} s"
            )
            continue
        assert abs(early) <= _TOL, f"{where}: stops {-early:.3f} s from {expected.isot}"
    assert cut == _CUT_BY_THE_NIGHTS_END.get(name, [])


def test_the_reported_visit_waits_for_its_pass_where_the_slew_ended():
    """With every calibration duration zero, the wait for the pass is an idle, not science.

    The visit is admitted at a 30 s tick 207.9 s before its pass opens and
    the slew takes 22.0 s, so the telescope waits 185.9 s at the slew's end,
    the pass's first sample, and the science block starts when the pass
    opens. No retune is booked, since it lasts no time.
    """
    timeline = night("zero_overhead")
    ((lead, science),) = _ce_visits(timeline)
    slew, wait = lead
    assert wait.block_type == BlockType.IDLE
    assert wait.metadata["reason"] == _WAITING
    assert wait.duration == pytest.approx(185.9, abs=0.05)
    assert (wait.az_start, wait.elevation) == (slew.az_end, slew.elevation)
    first, scan = _first_sample("zero_overhead", science)
    assert 0.0 <= (first - science.t_start).sec <= _step(scan.trajectory) + _TOL
    assert timeline.validate() == []


def test_a_waiting_visit_is_a_slew_a_retune_a_wait_and_the_scan():
    """The retune is booked after the slew, then the wait, both at the pose the slew left.

    The retune hands the wait the pose it holds (its ``az_final``), so the
    idle continues it, and the visit's anchor ``t0_scan`` stays before the
    pass opens, where the planner can still find the pass.
    """
    timeline = night("regression")
    ((lead, science),) = _ce_visits(timeline)
    assert [b.block_type for b in lead] == [
        BlockType.SLEW,
        BlockType.CALIBRATION,
        BlockType.IDLE,
    ]
    slew, retune, wait = lead
    assert retune.scan_type == "retune"
    assert wait.metadata["reason"] == _WAITING
    assert (retune.end_pose_az, retune.end_pose_el) == (slew.az_end, slew.elevation)
    assert (wait.az_start, wait.elevation) == (slew.az_end, slew.elevation)
    for before, after in zip((slew, retune, wait), (retune, wait, science), strict=True):
        assert after.t_start.unix == pytest.approx(before.t_stop.unix, abs=1e-6)
    assert Time(science.metadata["t0_scan"], scale="utc") < science.t_start
    assert timeline.validate() == []


#: The nights holding at least one constant-elevation visit the readiness
#: lead alone admitted.
_LEAD_ALONE = {
    "august",
    "ce_offset",
    "lsa_window",
    "lsa_window_offset",
    "narrow_slow_ce",
    "planet_scan",
    "quickstart",
    "regression",
    "time_based_retune",
    "wide_slow_ce",
    "window_rising",
    "window_cut",
    "zero_overhead",
    "retune_every_600_s",
    "retune_due_in_the_wait",
}


@pytest.mark.parametrize("name", [*sorted(NIGHTS), *_LOCAL_NIGHTS])
def test_a_visit_admitted_by_the_lead_alone_scans_from_its_pass_opening(name):
    """A visit waiting on the readiness lead alone scans from its pass's first sample.

    The lead holds the slew and the retune the visit books before its first
    subscan, the one due when the pass opens, so a visit admitted by it
    (:func:`_admitted_by_the_lead_alone`) has its first science sample at
    the first sample of the pass the planner returns from the visit's
    anchor, to one sample interval plus 0.01 s. A visit that something else
    held too close to or past its opening (the startup calibrations,
    another visit) is not this case and is not checked.
    """
    timeline = _timeline(name)
    checked = 0
    for lead, science in _ce_visits(timeline):
        times, step = _planned_pass(science)
        if not _admitted_by_the_lead_alone(timeline, lead, times[0]):
            continue
        checked += 1
        first, _ = _first_sample(name, science)
        late = (first - times[0]).sec
        where = f"{name}: {science.patch_name} at {science.t_start.isot}"
        assert -_TOL <= late <= step + _TOL, f"{where}: first sample {late:.3f} s into the pass"
    assert bool(checked) == (name in _LEAD_ALONE)


def test_a_retune_due_while_the_visit_waits_is_booked_before_the_wait():
    """A retune due by the pass opening, though not when the slew ends, comes before the wait.

    The retune falls due 7000 s after the night's first, at 07:58:20, after
    the slew onto Deep56 and before its pass opens; the visit books it when
    the slew ends, then waits, and scans from the pass's first sample.
    """
    name = "retune_due_in_the_wait"
    timeline = _timeline(name)
    ((lead, science),) = _ce_visits(timeline)
    assert [b.block_type for b in lead] == [BlockType.SLEW, BlockType.CALIBRATION, BlockType.IDLE]
    slew, retune, wait = lead
    assert retune.scan_type == "retune"
    assert wait.metadata["reason"] == _WAITING
    times, step = _planned_pass(science)
    falls_due = timeline.start_time + TimeDelta(7000.0, format="sec")
    assert slew.t_stop < falls_due < times[0]
    first, _ = _first_sample(name, science)
    assert -_TOL <= (first - times[0]).sec <= step + _TOL
    assert abs((science.t_start - wait.t_stop).sec) <= 1e-6
    assert timeline.validate() == []


def test_the_first_retune_is_the_one_due_where_the_first_subscan_would_start():
    """A visit books the retune due at the later of its start and its pass opening.

    The last retune is at ``t0`` and the cadence is the interval from ``t0``
    to a pass opening 600 s later, as the cadence tracker measures it, so
    the retune falls due exactly at the opening: a visit starting before it
    books one, and an opening a millisecond earlier does not. A visit
    starting at or after the opening asks at its own start, as a pong or
    daisy, which has no opening, does. With a cadence of zero a retune is
    always due.
    """
    t0 = Time("2026-06-15T02:00:00", scale="utc")

    def at(seconds):
        return t0 + TimeDelta(seconds, format="sec")

    opens = at(600.0)
    cadence = (opens - t0).sec
    ctx = _make_ctx([], calibration_policy=CalibrationPolicy(retune_cadence=cadence))
    state = _initial_state(ctx)
    state = state.advanced(cal_state=state.cal_state.update("retune", t0))
    first = phases._first_retune_spec
    assert first(state, ctx, at(200.0), opens).name == "retune"
    assert first(state, ctx, at(200.0), at(599.999)) is None
    assert first(state, ctx, opens, opens) is not None
    assert first(state, ctx, at(599.999), opens) is not None
    assert first(state, ctx, at(600.0), at(300.0)) is not None
    assert first(state, ctx, at(599.999), at(300.0)) is None
    assert first(state, ctx, at(200.0), None) is None
    assert first(state, ctx, at(600.0), None) is not None

    always = _make_ctx([], calibration_policy=CalibrationPolicy(retune_cadence=0.0))
    assert first(state, always, at(200.0), None) is not None
    assert first(state, always, at(200.0), opens) is not None


def _planned(patch, anchor):
    """Plan ``patch``'s pass from ``anchor`` with its recorded options, uncut."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="EXCLUSION ZONE: ", category=PointingWarning)
        return plan_constant_el_scan(
            field=FieldRegion(
                ra_center=patch.ra_center,
                dec_center=patch.dec_center,
                width=patch.width,
                height=patch.height,
            ),
            elevation=patch.elevation,
            velocity=patch.velocity,
            site=get_fyst_site(),
            start_time=anchor,
            rising=True,
            **patch.scan_params,
        )


def test_a_wait_is_booked_only_when_the_pass_opens_more_than_the_tolerance_later():
    """A pass opening 0.0099 s after the scan could start is scanned from there; 0.0101 s waits."""
    scan = _planned(WIDE_SLOW_CES, Time("2026-08-02T01:23:00", scale="utc"))
    opening = Time(str(scan.computed_params["start_time_iso"]), scale="utc")
    first = get_absolute_times(scan.trajectory)[0]
    for before, waits in ((0.0099, False), (0.0101, True)):
        scan_start = opening - TimeDelta(before, format="sec")
        open_at, _ = phases._ce_pass_bounds(scan, scan_start)
        if waits:
            assert open_at is not None and abs((open_at - first).sec) <= 1e-6
        else:
            assert open_at is None


@pytest.mark.parametrize("before_the_end", [0.0, 1e-3])
def test_a_pass_ending_one_interval_before_the_nights_end_keeps_its_last_sample(before_the_end):
    """A night ending one interval after a pass's last sample keeps that sample.

    The scan a visit is cut from runs to the night's end, and the visit's
    end is one interval after the last sample that scan holds: with the
    night ending exactly one interval after the pass's last sample, or a
    millisecond sooner, that is the pass's last sample.
    """
    anchor = Time("2026-08-02T01:23:00", scale="utc")
    scan = _planned(WIDE_SLOW_CES, anchor)
    times = get_absolute_times(scan.trajectory)
    step = _step(scan.trajectory)
    stop = times[-1] + TimeDelta(step - before_the_end, format="sec")
    ctx = SchedulerContext.build(
        patches=[WIDE_SLOW_CES],
        site=get_fyst_site(),
        start_time=Time("2026-08-02T01:00:00", scale="utc"),
        end_time=stop,
        overhead_model=OverheadModel(),
        calibration_policy=CalibrationPolicy(),
        time_step=300.0,
    )
    provisional = phases._provisional_ce_scan(
        WIDE_SLOW_CES,
        elevation=WIDE_SLOW_CES.elevation,
        rising=True,
        t0_scan=anchor.isot,
        scan_start=times[0],
        ctx=ctx,
    )
    sliced = get_absolute_times(provisional.trajectory)
    _, end = phases._ce_pass_bounds(provisional, times[0])
    assert abs((sliced[-1] - times[-1]).sec) <= 1e-6
    assert abs((end - (times[-1] + TimeDelta(step, format="sec"))).sec) <= 1e-6


def test_a_visit_whose_pass_is_open_after_the_retune_books_no_wait():
    """A pass already open when the retune ends is scanned from the retune's end.

    The startup calibrations hold this visit back until after its pass has
    opened, so nothing is waited for: the science block follows the retune
    directly.
    """
    timeline = night("default_overhead")
    ((lead, science),) = _ce_visits(timeline)
    assert [b.block_type for b in lead] == [BlockType.SLEW, BlockType.CALIBRATION]
    retune = lead[1]
    assert science.t_start.unix == pytest.approx(retune.t_stop.unix, abs=1e-6)
    _, scan = _first_sample("default_overhead", science)
    assert Time(scan.computed_params["start_time_iso"], scale="utc") < science.t_start


def test_the_first_subscan_after_a_wait_is_sized_from_where_it_starts():
    """The wait comes out of the visit's budget before the subscans are cut from it.

    Here the pass leaves 9602 s after its opening against a 9700 s
    ``max_scan_duration``, so the visit is one subscan from the opening;
    counted with the 185.9 s wait it would be two of 4894 s, the first of
    them 185.9 s short of the 4800 s ``min_scan_duration`` once it starts
    at the opening.
    """
    model = dataclasses.replace(ZERO_OVERHEAD, max_scan_duration=9700.0, min_scan_duration=4800.0)
    timeline = generate_timeline(
        patches=[CES],
        site=get_fyst_site(),
        start_time="2026-08-02T01:25:00",
        end_time="2026-08-02T04:25:00",
        overhead_model=model,
        time_step=30.0,
    )
    ((lead, science),) = _ce_visits(timeline)
    assert lead[-1].metadata["reason"] == _WAITING
    assert len(timeline.science_blocks) == 1
    assert science.duration >= model.min_scan_duration
    ((block, scan),) = schedule_to_trajectories(timeline)
    head = (get_absolute_times(scan.trajectory)[0] - block.t_start).sec
    assert 0.0 <= head <= _step(scan.trajectory) + _TOL
    assert timeline.validate() == []


def test_a_subscan_a_hair_short_of_whole_pong_periods_holds_them():
    """A subscan's time a fraction of a microsecond short of one period holds that period.

    The time a subscan is given is a difference of two instants, which can
    fall short of a whole number of periods by float noise alone: 2044.7999999 s
    against this pong's period of 2044.8000000000004 s is one period.
    """
    site = get_fyst_site()
    patch = ObservingPatch(
        name="P",
        ra_center=24.0,
        dec_center=-32.0,
        width=10.0,
        height=10.0,
        scan_type="pong",
        velocity=1.0,
    )
    period = phases._pong_period(patch)
    assert 0.0 < period - 2044.7999999 < 1e-6
    start = Time("2026-06-15T08:00:00", scale="utc")
    block = phases._plan_science_subscan(
        patch,
        t_start=start,
        available=2044.7999999,
        shortest=period,
        period=period,
        az_start=100.0,
        az_end=120.0,
        el=50.0,
        site=site,
        scan_index=0,
        subscan_index=0,
        rising=False,
        t0_scan=None,
        wrap_ref=110.0,
    )
    assert block is not None
    assert block.metadata["scan_params"]["n_cycles"] == 1
    assert block.duration == pytest.approx(period, abs=1e-9)


def _planet_passes(name):
    """Return the named night's planet-calibration pass blocks with their rebuilt scans."""
    return [
        (block, scan)
        for block, scan in rebuilt(name)
        if block.block_type == BlockType.CALIBRATION and block.scan_type == "planet_cal"
    ]


def test_every_planet_pass_block_spans_its_samples():
    """A planet-calibration pass block runs from its first sample to one interval past its last.

    The block holds the pass's samples and nothing else, so its start is the
    pass's first sample and its stop one sample interval after the last,
    each to the timeline's 0.01 s tolerance.
    """
    passes = _planet_passes("planet_scan")
    assert len(passes) >= 6
    for block, scan in passes:
        times = get_absolute_times(scan.trajectory)
        step = _step(scan.trajectory)
        where = f"pass {block.subscan_index} at {block.t_start.isot}"
        assert abs((times[0] - block.t_start).sec) <= _TOL, where
        assert abs((block.t_stop - times[-1]).sec - step) <= _TOL, where


def test_the_time_before_a_planet_pass_is_a_wait_where_the_telescope_is():
    """Before each pass the telescope waits, idle, at the pose it was left at.

    After the slew onto the planet the wait holds the slew's end pose, and
    between passes the previous pass's end pose: no move is booked between
    passes, and no pass block starts before the block it follows ends.
    """
    timeline = night("planet_scan")
    blocks = list(timeline.blocks)
    waits = 0
    for i, block in enumerate(blocks):
        if block.scan_type != "planet_cal" or "scan_params" not in block.metadata:
            continue
        before = blocks[i - 1]
        assert before.t_stop.unix == pytest.approx(block.t_start.unix, abs=1e-6)
        if before.block_type == BlockType.IDLE:
            waits += 1
            assert before.metadata["reason"] == _WAITING
            holder = blocks[i - 2]
            assert holder.block_type == BlockType.SLEW or holder.scan_type == "planet_cal"
            assert (before.az_start, before.elevation) == (holder.end_pose_az, holder.end_pose_el)
        else:
            assert before.block_type == BlockType.SLEW or before.scan_type == "planet_cal"
    assert waits >= 1
    assert timeline.validate() == []


#: The nights that book no idle and no calibration that does not scan.
_HOLDS_NOTHING = {"zero_settle"}


@pytest.mark.parametrize("name", sorted(NIGHTS))
def test_every_held_pose_is_where_the_block_before_left_the_telescope(name):
    """An idle, and a calibration that does not scan, ends at the pose the block before it left.

    Such a block holds the telescope still, so its end pose, in azimuth and
    elevation, is the end pose of the block before it: a retune before a
    visit's first subscan holds the slew's end pose, and one between two
    subscans the pose the earlier subscan ended at. The moves between two
    subscans of one visit and between two passes of a planet calibration
    are not booked, so the first sample of a science block or a pass is not
    held to this.
    """
    held = 0
    for before, block in itertools.pairwise(night(name).blocks):
        scans = "scan_params" in block.metadata
        if block.block_type == BlockType.IDLE or (
            block.block_type == BlockType.CALIBRATION and not scans
        ):
            held += 1
            where = f"{name}: {block.block_type} {block.scan_type} at {block.t_start.isot}"
            assert (block.end_pose_az, block.end_pose_el) == (
                before.end_pose_az,
                before.end_pose_el,
            ), where
    assert bool(held) == (name not in _HOLDS_NOTHING)


def test_a_planet_pass_records_no_boresight_rotation():
    """Every pass records the rotation it requested, none, and rebuilds from that record."""
    timeline = night("planet_scan")
    recorded = [
        b.metadata["scan_params"]
        for b in timeline.calibration_blocks
        if b.scan_type == "planet_cal" and "scan_params" in b.metadata
    ]
    assert recorded
    assert {sp["boresight_rot"] for sp in recorded} == {None}
    assert len(_planet_passes("planet_scan")) == len(recorded)


@pytest.mark.parametrize("name", ["regression", "quickstart", "august", "zero_overhead"])
def test_every_constant_elevation_block_is_anchored_before_its_pass(name):
    """A block's anchor ``t0_scan`` precedes the pass the planner returns from it.

    The planner finds a pass forward of its anchor and refuses an anchor at
    or inside the pass, so every block of a visit records an anchor before
    the first sample of the pass planned from it, and no block's own start
    is its anchor.
    """
    blocks = [b for b in night(name).science_blocks if b.scan_type == "constant_el"]
    assert blocks
    for block in blocks:
        anchor = Time(block.metadata["t0_scan"], scale="utc")
        where = f"{name}: {block.patch_name} at {block.t_start.isot}"
        assert abs((block.t_start - anchor).sec) > _TOL, where
        times, _ = _planned_pass(block)
        assert anchor < times[0], where
