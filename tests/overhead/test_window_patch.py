"""A constant-elevation patch with an azimuth window is scanned over that window.

A constant-elevation :class:`~fyst_trajectories.overhead.ObservingPatch`
whose ``scan_params`` give ``az_min`` and ``az_max`` is scanned over that
azimuth window: the subscans up to the next retune are one whole window scan
of :func:`~fyst_trajectories.planning.plan_constant_el_scan` (its window
form) from their first start, sized to whole legs inside their budget and
cut into blocks of at most the longest subscan. Each block records that scan
(``metadata["window_scan"]``), so a rebuild returns its own samples of it,
sample for sample. The visit is timed by the field's crossing of the patch
elevation on the window's side of the meridian.

These tests pin, on the shared window nights of ``_nights.py`` (a rising
window with the bay ``IM6``, written a whole turn below the azimuth limit,
and a setting window by day):

- every science block records its window scan and rebuilds to the scan the
  planner returns for the patch's own numbers, from memory and from a file;
- the samples outside the turnarounds lie inside the window plus its
  padding, each leg reaching both ends, a range other than the field's own;
- every subscan is whole legs, no longer than the longest subscan, and the
  visit opens no earlier than the field's pass and ends inside it;
- with a ring bay the record holds the detector's window and elevation and
  the block the boresight's pose;
- blocks with no retune between them join into the one scan the planner
  returns for their start and total length, every one of them screened;
- the count of whole legs at its ties, on a grid of throws, speeds,
  accelerations, timesteps and paddings;
- the refusals of a malformed window when the patch is made.
"""

import dataclasses
import itertools
import math
import warnings

import numpy as np
import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories import (
    SCAN_FLAG_SCIENCE,
    InstrumentOffset,
    boresight_to_detector,
    compute_focal_plane_rotation,
    get_absolute_times,
    get_fyst_site,
    get_primecam_offset,
)
from fyst_trajectories.exceptions import PointingWarning
from fyst_trajectories.overhead import (
    BlockType,
    CalibrationPolicy,
    DeferralReason,
    ObservingPatch,
    OverheadModel,
    generate_timeline,
    read_timeline,
    schedule_to_trajectories,
    write_timeline,
)
from fyst_trajectories.overhead.scheduler import helpers, phases
from fyst_trajectories.overhead.scheduler.state import SchedulerContext
from fyst_trajectories.overhead.utils import _stored_instant
from fyst_trajectories.patterns.turnarounds import turnaround_duration_sec
from fyst_trajectories.planning import FieldRegion, plan_constant_el_scan

from ._nights import (
    DEEP56,
    DEEP56_WINDOW_RISING,
    DEEP56_WINDOW_SETTING,
    NIGHTS,
    night,
    rebuilt,
    rebuilt_from_file,
)

SITE = get_fyst_site()

ARRAYS = ("times", "az", "el", "az_vel", "el_vel", "scan_flag")

#: The window nights and the patch each scans.
WINDOW_NIGHTS = {
    "window_rising": DEEP56_WINDOW_RISING,
    "window_setting_day": DEEP56_WINDOW_SETTING,
}

#: The timeline's block-time tolerance, in seconds.
_TOL = 0.01


def _field(patch):
    return FieldRegion(
        ra_center=patch.ra_center,
        dec_center=patch.dec_center,
        width=patch.width,
        height=patch.height,
    )


def _side(patch):
    """Return the crossing the window watches: east of the meridian is the rising one."""
    mid = 0.5 * (patch.scan_params["az_min"] + patch.scan_params["az_max"])
    return mid % 360.0 < 180.0


def _window_blocks(name):
    return [b for b in night(name).science_blocks if b.scan_type == "constant_el"]


def _visits(name):
    visits: dict[int, list] = {}
    for block in _window_blocks(name):
        visits.setdefault(block.scan_index, []).append(block)
    return list(visits.values())


def _field_pass(patch, name):
    """Return the field's own pass on the window's side, planned from the night's start.

    Its first and last instants (``start_time_iso``, ``end_time_iso``) are
    the field's crossings of the patch elevation, and its range is where the
    field's border meets that elevation.
    """
    _, start, _, _ = NIGHTS[name]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="EXCLUSION ZONE: ", category=PointingWarning)
        return plan_constant_el_scan(
            _field(patch),
            elevation=patch.elevation,
            velocity=patch.velocity,
            site=SITE,
            start_time=Time(start, scale="utc"),
            rising=_side(patch),
        )


def _leg_seconds(patch):
    """One leg and one turnaround of the patch's window scan, in seconds."""
    params = patch.scan_params
    pad = params.get("az_padding", 0.0)
    throw = (params["az_max"] + pad) - (params["az_min"] - pad)
    accel = params.get("az_accel", 1.0)
    return throw / patch.velocity, turnaround_duration_sec(patch.velocity, accel)


def _planned_window_scan(patch, start, duration, *, with_field=True):
    """Return the window scan the planner returns for the patch's own numbers.

    The field never shapes the scan; without it the planner judges no drift
    of the field through the window.
    """
    params = patch.scan_params
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="EXCLUSION ZONE: ", category=PointingWarning)
        return plan_constant_el_scan(
            _field(patch) if with_field else None,
            az_window=(params["az_min"], params["az_max"]),
            elevation=patch.elevation,
            velocity=patch.velocity,
            duration=duration,
            site=SITE,
            start_time=start,
            **{key: params[key] for key in ("az_accel", "timestep", "az_padding") if key in params},
            detector_offset=patch.detector_offset,
        )


@pytest.mark.parametrize("name", sorted(WINDOW_NIGHTS))
def test_every_science_block_records_its_window_scan(name):
    """Each block records the scan of its window, and the record is the patch's own numbers."""
    patch = WINDOW_NIGHTS[name]
    blocks = _window_blocks(name)
    assert blocks, name
    params = patch.scan_params
    pad = params.get("az_padding", 0.0)
    offset = None if patch.detector_offset is None else dataclasses.asdict(patch.detector_offset)
    for block in blocks:
        record = block.metadata["window_scan"]
        assert block.metadata["scan_params"] == {}
        lo, hi = record["az_window"]
        turns = round((lo - (params["az_min"] - pad)) / 360.0)
        assert lo == pytest.approx(params["az_min"] - pad + 360.0 * turns, abs=1e-9)
        assert hi == pytest.approx(params["az_max"] + pad + 360.0 * turns, abs=1e-9)
        assert record["elevation"] == patch.elevation
        assert record["detector_offset"] == offset
        assert block.metadata["velocity"] == patch.velocity


@pytest.mark.parametrize("source", ["memory", "file"])
@pytest.mark.parametrize("name", sorted(WINDOW_NIGHTS))
def test_every_block_rebuilds_to_the_window_scan_of_the_patch(name, source):
    """A block rebuilds to the scan the planner returns for the patch, from the block's start.

    The expectation is the window form of the planner called with the
    patch's field, window, elevation, speed and offset and the block's
    recorded duration, from the block's start as a file stores it; the
    duration is a whole number of legs.
    """
    patch = WINDOW_NIGHTS[name]
    t_cruise, t_turn = _leg_seconds(patch)
    pairs = rebuilt(name) if source == "memory" else rebuilt_from_file(name)
    checked = 0
    for block, scan in pairs:
        if block.block_type != BlockType.SCIENCE:
            continue
        checked += 1
        duration = block.metadata["window_scan"]["duration"]
        legs = (duration + t_turn) / (t_cruise + t_turn)
        assert legs == pytest.approx(round(legs), abs=1e-9), (block.t_start.isot, legs)
        want = _planned_window_scan(patch, _stored_instant(block.t_start), duration)
        assert want.duration == duration
        for attr in ARRAYS:
            assert np.array_equal(getattr(scan.trajectory, attr), getattr(want.trajectory, attr)), (
                block.t_start.isot,
                attr,
            )
    assert checked == len(_window_blocks(name))


def _detector_azimuth(patch, trajectory):
    if patch.detector_offset is None:
        return np.asarray(trajectory.az)
    rotation = compute_focal_plane_rotation(trajectory.el, site=SITE, offset=patch.detector_offset)
    az, _ = boresight_to_detector(trajectory.az, trajectory.el, patch.detector_offset, rotation)
    return np.asarray(az)


@pytest.mark.parametrize("name", sorted(WINDOW_NIGHTS))
def test_the_window_is_what_is_swept(name):
    """Outside the turnarounds the detector stays inside the window and reaches both ends.

    The expectation is the patch's window plus its padding, on the turn
    the samples sit on. The field's own pass on the window's side would
    sweep another range (asserted, so this cannot pass on a night where the
    two coincide).
    """
    patch = WINDOW_NIGHTS[name]
    params = patch.scan_params
    pad = params.get("az_padding", 0.0)
    lo, hi = params["az_min"] - pad, params["az_max"] + pad
    cp = _field_pass(patch, name).computed_params
    field_lo, field_hi = cp["az_start"], cp["az_stop"]
    turn = 360.0 * round((field_lo - lo) / 360.0)
    assert abs(field_lo - (lo + turn)) > 1.0 and abs(field_hi - (hi + turn)) > 1.0
    for block, scan in rebuilt(name):
        if block.block_type != BlockType.SCIENCE:
            continue
        az = _detector_azimuth(patch, scan.trajectory)
        science = scan.trajectory.scan_flag == SCAN_FLAG_SCIENCE
        shift = 360.0 * round((float(az[science].min()) - lo) / 360.0)
        az = az - shift
        reach = patch.velocity * float(scan.trajectory.times[1] - scan.trajectory.times[0])
        assert float(az[science].min()) >= lo - 1e-6, block.t_start.isot
        assert float(az[science].max()) <= hi + 1e-6, block.t_start.isot
        # Each run of science samples is one leg, from one end to the other.
        edges = np.flatnonzero(np.diff(science.astype(np.int8)))
        bounds = np.concatenate(([0], edges + 1, [science.size]))
        legs = 0
        for start, stop in itertools.pairwise(bounds):
            if not science[start]:
                continue
            legs += 1
            leg = az[start:stop]
            assert float(leg.min()) <= lo + reach + 1e-6, (block.t_start.isot, legs)
            assert float(leg.max()) >= hi - reach - 1e-6, (block.t_start.isot, legs)
        assert legs >= 1


@pytest.mark.parametrize("name", sorted(WINDOW_NIGHTS))
def test_every_subscan_fits_its_budget_and_the_visit_its_pass(name):
    """Whole legs, no longer than the longest subscan, inside the field's pass.

    The visit opens no earlier than the field's crossing pass on the
    window's side opens (the planner's ``start_time_iso`` for the field) and
    ends by the time it closes (``end_time_iso``) or the night ends, within
    one leg of it, since a subscan holds whole legs.
    """
    patch = WINDOW_NIGHTS[name]
    _, _, end, _ = NIGHTS[name]
    longest = OverheadModel().max_scan_duration
    t_cruise, t_turn = _leg_seconds(patch)
    cp = _field_pass(patch, name).computed_params
    opens = Time(cp["start_time_iso"], scale="utc")
    deadline = min(Time(cp["end_time_iso"], scale="utc"), Time(end, scale="utc"))
    visits = _visits(name)
    assert visits
    for visit in visits:
        assert visit[0].t_start >= opens - TimeDelta(1e-3, format="sec"), visit[0].t_start.isot
        for block in visit:
            assert block.duration <= longest + 1e-9, block.t_start.isot
        stop = visit[-1].t_stop
        assert stop <= deadline + TimeDelta(_TOL, format="sec")
        assert (deadline - stop).sec <= t_cruise + t_turn + 1.0, (stop.isot, deadline.isot)


def test_a_ring_bay_records_the_detector_and_the_block_the_boresight():
    """With IM6 the record holds the detector's window and elevation, the block the boresight's.

    The planner moves the window (-252, -242) a whole turn up, onto
    (108, 118), and sweeps the detector there at el 50; the boresight scans
    1.7 deg higher, so the block's elevation is the rebuilt trajectory's.
    """
    blocks = _window_blocks("window_rising")
    assert blocks
    for (block, scan), _ in zip(
        [(b, s) for b, s in rebuilt("window_rising") if b.block_type == BlockType.SCIENCE],
        blocks,
        strict=True,
    ):
        record = block.metadata["window_scan"]
        assert record["az_window"] == [108.0, 118.0]
        assert record["elevation"] == 50.0
        assert record["detector_offset"] == dataclasses.asdict(get_primecam_offset("IM6"))
        assert block.elevation == float(scan.trajectory.el[0])
        assert block.elevation - 50.0 > 1.0


def test_the_first_sample_of_a_window_visit_is_where_the_slew_ends():
    """The slew onto a window visit ends at its first window scan's first sample."""
    for name in WINDOW_NIGHTS:
        blocks = list(night(name).blocks)
        pairs = {id(b): s for b, s in rebuilt(name)}
        for visit in _visits(name):
            first = visit[0]
            index = blocks.index(first)
            slew = next(b for b in reversed(blocks[:index]) if b.block_type == BlockType.SLEW)
            trajectory = pairs[id(first)].trajectory
            az0 = float(trajectory.az[0])
            assert slew.az_end == pytest.approx(
                az0 + 360.0 * round((slew.az_end - az0) / 360.0), abs=1e-6
            )
            assert slew.elevation == pytest.approx(float(trajectory.el[0]), abs=1e-6)
            assert get_absolute_times(trajectory)[0] >= slew.t_stop - TimeDelta(1e-6, format="sec")


def _window_night(patch):
    _, start, end, _ = NIGHTS["window_setting_day"]
    return generate_timeline(patches=[patch], site=SITE, start_time=start, end_time=end)


def test_a_window_visit_the_axes_cannot_run_is_never_slewed_to():
    """A window scan over the hardware's turnaround limit idles at the unmoved pose.

    With ``az_accel`` 4.5 deg/s^2 the quintic turnaround peaks at 6.75,
    over the 6.0 the hardware takes, so the screen before the slew refuses
    the visit's shortest first subscan: the night books no slew and no
    science, and idles ``axis_dynamics``.
    """
    patch = dataclasses.replace(
        DEEP56_WINDOW_SETTING,
        scan_params={**DEEP56_WINDOW_SETTING.scan_params, "az_accel": 4.5},
    )
    blocks = list(_window_night(patch).blocks)
    assert not [b for b in blocks if b.block_type in (BlockType.SLEW, BlockType.SCIENCE)]
    assert "axis_dynamics" in {b.metadata.get("reason") for b in blocks}


def test_a_refused_window_subscan_is_retried_with_fewer_legs(monkeypatch):
    """A subscan the booking screen refuses is planned again shorter, down to the fewest legs.

    The screen is made to refuse every window scan longer than the shortest
    subscan; the shortest passes, so the visit is slewed to, and every
    science block holds the fewest legs a subscan may hold.
    """
    patch = DEEP56_WINDOW_SETTING
    shortest = helpers._window_scan_seconds(
        patch, helpers._window_shortest_legs(patch, OverheadModel())
    )
    real = phases._execution_refusal
    refused = []

    def screen(trajectory, scan_type, scan_params, site, sun_safe):
        if float(trajectory.times[-1]) > shortest + 1e-6:
            refused.append(float(trajectory.times[-1]))
            return DeferralReason.AXIS_DYNAMICS
        return real(trajectory, scan_type, scan_params, site, sun_safe)

    monkeypatch.setattr(phases, "_execution_refusal", screen)
    science = [b for b in _window_night(patch).blocks if b.block_type == BlockType.SCIENCE]
    assert science
    assert refused
    for block in science:
        assert block.metadata["window_scan"]["duration"] == pytest.approx(shortest, abs=1e-9)


def test_every_block_of_a_stretch_is_screened(monkeypatch):
    """The booking screen sees each block's own samples; one it refuses books no stretch.

    With a retune every two hours and subscans of at most 1377 s, the visit
    would be one scan cut into blocks. The screen is made to refuse every
    sample run that does not begin at the window's low end, which only a
    later block of a stretch does; so the visit is booked as one whole scan,
    and since no retune is due after it, ends there: every science block
    starts its own scan at the window's low end, and none records a scan
    start of its own.
    """
    patch = DEEP56_WINDOW_SETTING
    low = patch.scan_params["az_min"]
    real = phases._execution_refusal
    refused = []

    def screen(trajectory, scan_type, scan_params, site, sun_safe):
        if float(trajectory.az[0]) != low:
            refused.append(float(trajectory.az[0]))
            return DeferralReason.AXIS_DYNAMICS
        return real(trajectory, scan_type, scan_params, site, sun_safe)

    monkeypatch.setattr(phases, "_execution_refusal", screen)
    _, start, end, kwargs = NIGHTS["window_cut"]
    timeline = generate_timeline(
        patches=[patch], site=SITE, start_time=start, end_time=end, **kwargs
    )
    science = [b for b in timeline.blocks if b.block_type == BlockType.SCIENCE]
    assert science
    assert refused
    kinds = [b.block_type for b in timeline.blocks]
    assert (BlockType.SCIENCE, BlockType.SCIENCE) not in set(itertools.pairwise(kinds))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="EXCLUSION ZONE: ", category=PointingWarning)
        pairs = schedule_to_trajectories(timeline)
    for block, scan in pairs:
        assert "scan_start" not in block.metadata
        assert float(scan.trajectory.az[0]) == low


def _stretches(timeline, pairs):
    """Group the rebuilt science blocks into runs booked back to back, no block between."""
    scans = {id(block): scan for block, scan in pairs}
    runs: list[list] = []
    previous = None
    for block in timeline.blocks:
        if block.block_type != BlockType.SCIENCE:
            previous = None
            continue
        if previous is None:
            runs.append([])
        runs[-1].append((block, scans[id(block)]))
        previous = block
    return runs


@pytest.mark.parametrize("source", ["memory", "file"])
@pytest.mark.parametrize("longest", [1000.0, 1377.0])
def test_blocks_with_no_retune_between_them_are_one_window_scan(longest, source, tmp_path):
    """Science blocks booked back to back rebuild to one window scan, sample for sample.

    With a retune every two hours and subscans of at most ``longest``
    seconds, the visit is cut with no retune between its first blocks. The
    samples of such a run, joined, are the window scan the planner builds
    directly from the run's start for the time from its first sample to its
    last, from memory and from a written and re-read file.
    """
    patch = DEEP56_WINDOW_SETTING
    _, start, end, _ = NIGHTS["window_setting_day"]
    timeline = generate_timeline(
        patches=[patch],
        site=SITE,
        start_time=start,
        end_time=end,
        overhead_model=OverheadModel(max_scan_duration=longest),
        calibration_policy=CalibrationPolicy(retune_cadence=7200.0),
    )
    if source == "file":
        path = tmp_path / "night.ecsv"
        write_timeline(timeline, path)
        timeline = read_timeline(path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="EXCLUSION ZONE: ", category=PointingWarning)
        pairs = schedule_to_trajectories(timeline)
    runs = [run for run in _stretches(timeline, pairs) if len(run) > 1]
    assert runs
    for run in runs:
        head = run[0][1].trajectory
        starts = [scan.trajectory.start_time for _, scan in run]
        last = run[-1][1].trajectory
        span = (starts[-1] - starts[0]).sec + float(last.times[-1])
        planned = _planned_window_scan(patch, head.start_time, span, with_field=False).trajectory
        for name in ARRAYS[1:]:
            joined = np.concatenate([getattr(scan.trajectory, name) for _, scan in run])
            assert np.array_equal(joined, getattr(planned, name)), (run[0][0].t_start.isot, name)
        absolute = np.concatenate(
            [
                (t0 - starts[0]).sec + scan.trajectory.times
                for t0, (_, scan) in zip(starts, run, strict=True)
            ]
        )
        np.testing.assert_allclose(absolute, planned.times, rtol=0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# The count of whole legs at its ties
# ---------------------------------------------------------------------------

#: Throws, speeds, accelerations, timesteps and paddings: each row is one
#: window scan, chosen so the legs and the samples fall on uneven numbers.
_GRID = [
    (2.0, 0.5, 1.0, 0.1, 0.0),
    (10.0, 0.5, 1.0, 0.1, 0.0),
    (10.0, 0.3, 0.5, 0.05, 0.75),
    (37.3, 1.1, 0.9, 0.37, 0.0),
    (7.7, 0.9, 0.7, 0.2, 0.25),
    (23.45, 0.45, 0.95, 0.13, 1.5),
    (0.6, 0.25, 0.4, 0.1, 0.0),
    (55.0, 2.0, 0.8, 0.05, 2.0),
]


def _grid_patch(throw, velocity, accel, timestep, padding):
    return dataclasses.replace(
        DEEP56,
        name="GridWindow",
        velocity=velocity,
        scan_params={
            "az_min": 108.0,
            "az_max": 108.0 + throw,
            "az_accel": accel,
            "timestep": timestep,
            "az_padding": padding,
        },
    )


def _block_seconds(patch, legs):
    """Return the block ``legs`` whole legs fill: the planner's last sample plus one interval.

    The planner is asked for exactly ``legs`` legs and turnarounds and must
    build that count (so a tie falls where the planner falls).
    """
    t_cruise, t_turn = _leg_seconds(patch)
    duration = legs * t_cruise + (legs - 1) * t_turn
    scan = _planned_window_scan(
        patch, Time("2026-06-15T08:00:00", scale="utc"), duration, with_field=False
    )
    assert scan.computed_params["n_scans"] == legs
    times = scan.trajectory.times
    return float(times[-1]) + float(times[1] - times[0])


@pytest.mark.parametrize("row", _GRID)
def test_the_whole_legs_of_a_budget_at_its_ties(row):
    """A budget exactly on whole legs holds them; a float step less holds one leg fewer.

    The block of ``n`` legs lasts the planner's built duration plus one
    sample interval. A budget equal to it, or one float step longer, holds
    ``n`` legs, and one float step shorter holds ``n - 1``: the built scan is
    never longer than its budget and never one leg fewer than fits. The tie
    falls on the side the planner falls on when asked for that duration: it
    builds exactly ``n`` legs.
    """
    patch = _grid_patch(*row)
    for legs in (1, 2, 7):
        exact = _block_seconds(patch, legs)
        assert helpers._window_legs(patch, exact) == legs
        assert helpers._window_legs(patch, math.nextafter(exact, math.inf)) == legs
        assert helpers._window_legs(patch, math.nextafter(exact, -math.inf)) == legs - 1
        between = 0.5 * (exact + _block_seconds(patch, legs + 1))
        assert helpers._window_legs(patch, between) == legs


def test_a_budget_on_the_estimated_block_books_its_legs_as_built():
    """The built scan can run past its estimated block by round-off, and is booked whole.

    A window written below the azimuth limit is planned on the turn above
    it, where the planner's grid comes out 1.1e-13 s longer than the block
    estimated from the window as written. A subscan given exactly that
    estimate still books the one leg it holds, ending one sample interval
    after the scan's last sample.
    """
    patch = ObservingPatch(
        name="Low",
        ra_center=24.0,
        dec_center=-32.0,
        width=40.0,
        height=10.0,
        scan_type="constant_el",
        velocity=0.3,
        elevation=50.0,
        scan_params={"az_min": -200.0, "az_max": -58.15, "timestep": 0.1, "rising": True},
    )
    start = Time("2026-06-15T06:00:00", scale="utc")
    ctx = SchedulerContext.build(
        [patch], get_fyst_site(), start, start + TimeDelta(7200.0, format="sec")
    )
    estimate = helpers._window_block_seconds(patch, 1)
    assert helpers._window_shortest_legs(patch, ctx.overhead_model) == 1
    scan = helpers._window_scan(patch, start, 1, ctx.site, {})
    times = scan.trajectory.times
    built = float(times[-1]) + float(times[1] - times[0])
    assert 0.0 < built - estimate < 1e-9  # the premise: the grid runs past the estimate
    block = phases._plan_window_subscan(
        patch,
        t_start=start,
        available=estimate,
        scan_index=0,
        subscan_index=0,
        rising=True,
        t0_scan=None,
        wrap_ref=0.0,
        ctx=ctx,
    )
    assert not isinstance(block, DeferralReason)
    assert block.metadata["window_scan"]["duration"] == pytest.approx(float(times[-1]))
    assert (block.t_stop - block.t_start).sec == pytest.approx(built, abs=1e-6)


# ---------------------------------------------------------------------------
# The refusals of a malformed window
# ---------------------------------------------------------------------------


def _window(**scan_params):
    return dataclasses.replace(DEEP56, name="W", scan_params=scan_params)


@pytest.mark.parametrize(
    ("scan_params", "match"),
    [
        ({"az_min": 100.0}, "az_max"),
        ({"az_max": 120.0}, "az_min"),
        ({"az_min": 120.0, "az_max": 100.0}, "az_min must be less than az_max"),
        ({"az_min": 110.0, "az_max": 110.0}, "az_min must be less than az_max"),
        ({"az_min": math.nan, "az_max": 120.0}, "az_min"),
        ({"az_min": 100.0, "az_max": math.inf}, "az_max"),
        ({"az_min": "100", "az_max": 120.0}, "az_min"),
        ({"az_min": 100.0, "az_max": 120.0, "lsa_window": (330.0, 350.0)}, "lsa_window"),
        ({"az_min": 170.0, "az_max": 190.0}, "meridian"),
        ({"az_min": -10.0, "az_max": 15.0}, "meridian"),
        ({"az_min": 100.0, "az_max": 180.0}, "meridian"),
        ({"az_min": 100.0, "az_max": 120.0, "rising": False}, "rising"),
        ({"az_min": 200.0, "az_max": 220.0, "rising": True}, "rising"),
    ],
)
def test_a_malformed_window_is_refused_when_the_patch_is_made(scan_params, match):
    with pytest.raises(ValueError, match=match):
        _window(**scan_params)


@pytest.mark.parametrize(
    "scan_params",
    [
        {"az_min": 100.0, "az_max": 120.0},
        {"az_min": 100.0, "az_max": 120.0, "rising": True},
        {"az_min": 200.0, "az_max": 220.0, "rising": False},
        {"az_min": 170.0, "az_max": 190.0, "rising": True},
        {"az_min": -10.0, "az_max": 15.0, "rising": False},
        {"az_min": -252.0, "az_max": -242.0},
    ],
)
def test_a_window_with_its_side_is_accepted(scan_params):
    patch = _window(**scan_params)
    assert patch.scan_params == scan_params


def test_the_window_side_is_the_side_the_scheduler_watches():
    """One helper reads a patch's side: ``rising`` when given, else the window's."""
    assert helpers._ce_patch_rising(_window(az_min=100.0, az_max=120.0)) is True
    assert helpers._ce_patch_rising(_window(az_min=-252.0, az_max=-242.0)) is True
    assert helpers._ce_patch_rising(_window(az_min=242.0, az_max=252.0)) is False
    assert helpers._ce_patch_rising(_window(az_min=170.0, az_max=190.0, rising=False)) is False
    assert helpers._ce_patch_rising(_window(rising=True)) is True
    assert helpers._ce_patch_rising(_window()) is None


def test_a_pong_or_daisy_window_is_an_unknown_key():
    for scan_type in ("pong", "daisy"):
        with pytest.raises(ValueError, match=r"unknown keys \['az_max', 'az_min'\]"):
            ObservingPatch(
                name="P",
                ra_center=24.0,
                dec_center=-32.0,
                width=4.0,
                height=4.0,
                scan_type=scan_type,
                velocity=0.5,
                scan_params={"az_min": 100.0, "az_max": 120.0},
            )


def test_the_offset_is_recorded_by_value():
    """A window block's offset is the patch's, field for field."""
    (block, *_) = _window_blocks("window_rising")
    assert InstrumentOffset(**block.metadata["window_scan"]["detector_offset"]) == (
        DEEP56_WINDOW_RISING.detector_offset
    )
