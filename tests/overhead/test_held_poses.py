"""No pose the telescope holds still is inside the Sun zone at any instant of the block.

A block holds the telescope still when it is an idle (whatever its reason,
the wait for a pass and the stretch a schedule ends with included) or a
calibration that does not scan (a retune, a pointing or focus calibration, a
skydip, a planet calibration booked as a placeholder). For every such block
of the shared nights, of two short survey windows and of three calibration
nights, the separation from the Sun of the pose the block records is
computed here, from the Sun's position, on a 1 s grid from the block's start
plus its end, and it must exceed the site's exclusion radius at every
instant (at the radius is inside, as :func:`~fyst_trajectories.is_sun_safe`
has it). The one exception is an idle labelled ``no_escape``, the record
that the zone holds the telescope where it sits.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools

import numpy as np
import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories import Coordinates, get_fyst_site
from fyst_trajectories.overhead import (
    BlockType,
    CalibrationPolicy,
    DeferralReason,
    ObservingPatch,
    ObservingTimeline,
    OverheadModel,
    TimelineBlock,
    generate_timeline,
    plan_calibration_night,
)
from fyst_trajectories.sun_models import make_sun_safe

from ._nights import DEEP56, NIGHTS, night, short_calibration_night

#: Spacing of the instants each held block is checked at, in seconds.
_GRID_SEC = 1.0

_NO_ESCAPE = str(DeferralReason.NO_ESCAPE)

#: Calibration nights: targets, window and start pose (``None`` for the
#: planner's default). ``short_night`` is the planner's short night;
#: ``twilight`` runs from the evening twilight, the Sun a few degrees below
#: the horizon; in ``dawn`` Jupiter is not up, so the night idles from a
#: pose low in the east while the Sun rises towards it.
_CALIBRATION_NIGHTS = {
    "short_night": (["saturn", "uranus"], "2026-09-11T06:30:00", "2026-09-11T07:30:00", None),
    "twilight": (["saturn", "neptune"], "2026-11-15T23:10:00", "2026-11-16T00:10:00", None),
    "dawn": (["jupiter"], "2027-03-20T09:09:00", "2027-03-20T10:39:00", (90.0, 35.0)),
}

#: Short survey windows of this guard's own. In ``never_rises`` the one
#: patch never clears the lower elevation limit, so the telescope idles
#: through a daytime window while the Sun moves through the poses it holds.
#: ``tail`` is the first five and a half minutes of the pong and daisy
#: daytime night with a minimum scan of 300 s, so the schedule ends on a
#: tail held at the pose its first move out of the zone reached.
_SURVEY_WINDOWS = ("never_rises", "tail")

#: The nights that book no idle and no calibration that does not scan.
_HOLDS_NOTHING = {"zero_settle"}


def _is_held(block) -> bool:
    """Whether ``block`` holds the telescope still and is held to the zone."""
    if block.block_type == BlockType.IDLE:
        return block.metadata.get("reason") != _NO_ESCAPE
    return block.block_type == BlockType.CALIBRATION and "scan_params" not in block.metadata


#: The separations of each timeline already read, keyed by ``id`` and holding
#: the timeline, so an entry is never matched by a later object of that id.
_SEPARATIONS: dict[int, tuple[ObservingTimeline, list]] = {}


def _held_separations(timeline):
    """Return ``(block, instants, separations, Sun elevations)`` for every held block.

    The instants are the block's start, every :data:`_GRID_SEC` after it and
    its end; the pose is the one the block records. The Sun is solved once
    per timeline.
    """
    entry = _SEPARATIONS.get(id(timeline))
    if entry is not None and entry[0] is timeline:
        return entry[1]
    out = _solve_held_separations(timeline)
    _SEPARATIONS[id(timeline)] = (timeline, out)
    return out


def _solve_held_separations(timeline):
    held = [b for b in timeline.blocks if _is_held(b)]
    if not held:
        return []
    coords = Coordinates(timeline.site)
    grids = [np.append(np.arange(0.0, b.duration, _GRID_SEC), b.duration) for b in held]
    unix = np.concatenate([b.t_start.unix + g for b, g in zip(held, grids, strict=True)])
    times = Time(unix, format="unix", scale="utc")
    sun_az, sun_el = coords.get_sun_altaz(times)
    az = np.concatenate([np.full(g.size, b.end_pose_az) for b, g in zip(held, grids, strict=True)])
    el = np.concatenate([np.full(g.size, b.end_pose_el) for b, g in zip(held, grids, strict=True)])
    sep = np.asarray(coords.angular_separation(az, el, sun_az, sun_el))
    sun_el = np.asarray(sun_el)
    out = []
    edge = 0
    for block, grid in zip(held, grids, strict=True):
        span = slice(edge, edge + grid.size)
        out.append((block, times[span], sep[span], sun_el[span]))
        edge += grid.size
    return out


def _inside(timeline):
    """Return ``(scan type, block start, first instant inside, separation)`` per pose inside."""
    radius = timeline.site.sun_avoidance.exclusion_radius
    found = []
    for block, times, sep, _ in _held_separations(timeline):
        bad = sep <= radius
        if bad.any():
            k = int(np.argmax(bad))
            found.append((block.scan_type, block.t_start.isot, times[k].isot, float(sep[k])))
    return found


@functools.cache
def _calibration_night(name):
    if name == "short_night":
        return short_calibration_night()
    targets, start, end, start_pose = _CALIBRATION_NIGHTS[name]
    return plan_calibration_night(targets, get_fyst_site(), start, end, start_pose=start_pose)


def _never_rises(sun_safe=None, end_time="2026-12-15T18:00:00"):
    """Plan the ``never_rises`` window, under ``sun_safe`` when given, to ``end_time``."""
    north = ObservingPatch(
        name="North",
        ra_center=0.0,
        dec_center=75.0,
        width=2.0,
        height=2.0,
        scan_type="pong",
        velocity=0.5,
    )
    return generate_timeline(
        patches=[north],
        site=get_fyst_site(),
        start_time="2026-12-15T15:00:00",
        end_time=end_time,
        sun_safe=sun_safe,
    )


@functools.cache
def _survey_window(name):
    if name == "never_rises":
        return _never_rises()
    patches, start, _, kwargs = NIGHTS["pong_and_daisy_daytime"]
    return generate_timeline(
        patches=patches,
        site=get_fyst_site(),
        start_time=start,
        end_time="2026-12-15T16:05:30",
        overhead_model=OverheadModel(min_scan_duration=300.0),
        **kwargs,
    )


@pytest.mark.parametrize("name", sorted(NIGHTS))
def test_no_held_pose_of_a_shared_night_is_inside_the_sun_zone(name):
    assert _inside(night(name)) == []


@pytest.mark.parametrize("name", sorted(_CALIBRATION_NIGHTS))
def test_no_held_pose_of_a_calibration_night_is_inside_the_sun_zone(name):
    assert _inside(_calibration_night(name)) == []


@pytest.mark.parametrize("name", _SURVEY_WINDOWS)
def test_no_held_pose_of_a_short_survey_window_is_inside_the_sun_zone(name):
    assert _inside(_survey_window(name)) == []


def _is_escape(block) -> bool:
    return block.block_type == BlockType.SLEW and block.patch_name == "sun_escape"


def test_a_move_out_of_the_zone_lands_where_the_next_block_is_held():
    """No move out of the zone is followed directly by another.

    The scheduler asks for the move with the time it next holds the pose
    (the first calibration due, else one tick), so the calibration or idle
    after the move is held where the move ends.
    """
    timelines = {name: night(name) for name in sorted(NIGHTS)}
    timelines.update((name, _survey_window(name)) for name in _SURVEY_WINDOWS)
    in_a_row = {
        name: [
            b.t_start.isot
            for a, b in itertools.pairwise(tl.blocks)
            if _is_escape(a) and _is_escape(b)
        ]
        for name, tl in timelines.items()
    }
    assert in_a_row == dict.fromkeys(timelines, [])
    assert any(_is_escape(b) for b in night("pong_and_daisy_daytime").blocks)


def _daylight(timeline):
    return [
        block for block, _, _, sun_el in _held_separations(timeline) if float(np.min(sun_el)) > 0.0
    ]


def test_the_guard_reads_held_poses_and_some_in_daylight():
    """The guard is not vacuous: the nights hold poses, and the daytime ones by day."""
    for name in sorted(set(NIGHTS) - _HOLDS_NOTHING):
        assert _held_separations(night(name)), name
    for name in _CALIBRATION_NIGHTS:
        assert _held_separations(_calibration_night(name)), name
    assert len(_daylight(night("pong_and_daisy_daytime"))) >= 5
    idles = [
        block
        for block in _daylight(_survey_window("never_rises"))
        if block.block_type == BlockType.IDLE
    ]
    assert len(idles) >= 20
    last = _survey_window("tail").blocks[-1]
    assert last.block_type == BlockType.IDLE
    assert last.metadata["reason"] == str(DeferralReason.WINDOW_CLOSED)
    assert last.duration > 200.0


def test_the_guard_counts_a_pose_at_the_radius_as_inside():
    """A pose whose separation is the site's radius, to the last bit, is reported."""
    site = get_fyst_site()
    coords = Coordinates(site)
    t = Time("2026-12-15T16:00:00", scale="utc")
    sun_az, sun_el = coords.get_sun_altaz(t)
    az, el = float(sun_az), float(sun_el) - 45.0
    sep = float(coords.angular_separation(az, el, sun_az, sun_el))
    zone = dataclasses.replace(site.sun_avoidance, exclusion_radius=sep, warning_radius=sep + 5.0)
    at_radius = dataclasses.replace(site, sun_avoidance=zone)
    block = TimelineBlock.idle(t, 0.0, az, el, 0)
    probe = ObservingTimeline(
        blocks=[block],
        site=at_radius,
        start_time=t,
        end_time=t + TimeDelta(1.0, format="sec"),
        overhead_model=OverheadModel(),
        calibration_policy=CalibrationPolicy(),
    )
    assert _inside(probe) == [("idle", t.isot, t.isot, sep)]


def _separation(az1, el1, az2, el2):
    """Great-circle separation of two horizontal positions, in degrees, elementwise."""
    a1, e1, a2, e2 = (np.radians(np.asarray(x, dtype=float)) for x in (az1, el1, az2, el2))
    cos_sep = np.sin(e1) * np.sin(e2) + np.cos(e1) * np.cos(e2) * np.cos(a1 - a2)
    return np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0)))


class _ClosedDisc:
    """The site's scalar Sun zone, and a small disc of sky closed for a stretch of time.

    A Sun model like any other to the schedulers (a point verdict and a
    ``batch`` of them). The Sun moves slowly enough that a pose clear at
    both ends of a hold of a few minutes is clear between them; the disc is
    closed between the ends, so it finds a hold whose instants between
    the ends are not asked.
    """

    def __init__(self, az, el, start, stop, radius=0.05):
        self._scalar = make_sun_safe("scalar", site=get_fyst_site())
        self.az, self.el, self.start, self.stop, self.radius = az, el, start, stop, radius

    def covers(self, az, el, unix):
        """Whether the disc holds ``(az, el)`` at ``unix`` (seconds), elementwise."""
        az, el, unix = np.broadcast_arrays(*(np.atleast_1d(x) for x in (az, el, unix)))
        inside = _separation(az, el, self.az, self.el) < self.radius
        return inside & (unix >= self.start) & (unix <= self.stop)

    def batch(self, az, el, times):
        return self._scalar.batch(az, el, times) & ~self.covers(az, el, times.unix)

    def __call__(self, az, el, time):
        return bool(self.batch(az, el, time)[0])


def _held_in(timeline, disc):
    """Return ``(scan type, block start)`` of every held block the disc reaches."""
    found = []
    for block in timeline.blocks:
        if not _is_held(block):
            continue
        grid = np.append(np.arange(0.0, block.duration, _GRID_SEC), block.duration)
        if disc.covers(block.end_pose_az, block.end_pose_el, block.t_start.unix + grid).any():
            found.append((block.scan_type, block.t_start.isot))
    return found


def _slews_given_up(timeline):
    """Return the start of every slew to a patch that is given up on arrival.

    A slew is given up when the block after it is a move out of the zone
    or an idle the Sun zone refused (``sun_point``) at its end pose.
    """
    sun_point = str(DeferralReason.SUN_POINT)
    return [
        a.t_start.isot
        for a, b in itertools.pairwise(timeline.blocks)
        if a.block_type == BlockType.SLEW
        and a.patch_name.startswith("slew_to_")
        and (_is_escape(b) or b.metadata.get("reason") == sun_point)
    ]


#: Deep56 on the night its rising pass opens near 08:02, in subscans of at
#: most 600 s with a retune before each.
_SHORT_SUBSCANS = (
    [DEEP56],
    "2026-06-15T07:00:00",
    "2026-06-15T09:30:00",
    {"overhead_model": OverheadModel(max_scan_duration=600.0)},
)


def _survey_night(spec, sun_safe=None):
    patches, start, end, kwargs = spec
    return generate_timeline(
        patches=patches,
        site=get_fyst_site(),
        start_time=start,
        end_time=end,
        sun_safe=sun_safe,
        **kwargs,
    )


@functools.cache
def _short_subscans_open():
    """Return the :data:`_SHORT_SUBSCANS` night under the site's own Sun model, planned once."""
    return _survey_night(_SHORT_SUBSCANS)


def _first(blocks, predicate):
    return next(b for b in blocks if predicate(b))


class TestAPoseHeldBetweenItsEnds:
    """A pose held at both ends of a hold is held between them too.

    For every place a scheduler holds a pose before what follows it, a
    disc of sky is closed between the two ends of the hold the open night
    books there, so only a check of the instants between them sees it.
    The night planned with the disc closed holds no pose inside it, and
    the open night shows the disc lies where a held block sits.
    """

    def test_the_wait_before_a_constant_elevation_pass_is_not_slewed_to(self):
        """The slew's arrival pose has to hold until the pass's first sample."""
        night_open = _short_subscans_open()
        slew = _first(night_open.blocks, lambda b: b.patch_name == "slew_to_Deep56")
        science = _first(night_open.blocks, lambda b: b.block_type == BlockType.SCIENCE)
        start, stop = slew.t_stop.unix + 330.0, science.t_start.unix - 30.0
        assert stop - start > 60.0
        disc = _ClosedDisc(slew.end_pose_az, slew.end_pose_el, start, stop)
        assert _held_in(night_open, disc)

        night_closed = _survey_night(_SHORT_SUBSCANS, sun_safe=disc)

        assert _held_in(night_closed, disc) == []
        assert _slews_given_up(night_closed) == []
        assert any(b.block_type == BlockType.SCIENCE for b in night_closed.blocks)

    def test_an_idle_after_a_calibration_is_not_held_where_the_zone_reaches(self):
        """The idle tick after a calibration asks about its pose for the whole tick.

        The move out of the zone at the top of the loop asks only for the
        calibration due then, so the tick after it is the idle's own.
        """
        night_open = _survey_window("never_rises")
        idle = next(
            b
            for a, b in itertools.pairwise(night_open.blocks)
            if a.block_type == BlockType.CALIBRATION and b.block_type == BlockType.IDLE
        )
        start, stop = idle.t_start.unix + 30.0, idle.t_stop.unix - 30.0
        assert stop - start > 60.0
        disc = _ClosedDisc(idle.end_pose_az, idle.end_pose_el, start, stop)
        assert _held_in(night_open, disc)

        night_closed = _never_rises(sun_safe=disc)

        assert _held_in(night_closed, disc) == []

    def test_a_calibration_after_another_is_not_held_where_the_zone_reaches(self):
        """Each calibration of a burst asks about its pose for its own length.

        The move out of the zone at the top of the loop asks only for the
        first calibration due, so the second is the calibration's own.
        """
        night_open = _survey_window("never_rises")
        second = next(
            b
            for a, b in itertools.pairwise(night_open.blocks)
            if a.block_type == BlockType.CALIBRATION and b.block_type == BlockType.CALIBRATION
        )
        start, stop = second.t_start.unix + 30.0, second.t_stop.unix - 30.0
        assert stop - start > 60.0
        disc = _ClosedDisc(second.end_pose_az, second.end_pose_el, start, stop)
        assert _held_in(night_open, disc)

        night_closed = _never_rises(sun_safe=disc)

        assert _held_in(night_closed, disc) == []

    def test_the_schedules_tail_is_not_held_where_the_zone_reaches(self):
        """The stretch after the last block asks about its pose until the window closes.

        The window ends 55 s after an idle tick, under the minimum scan
        duration, so the tail follows the tick and nothing before it asks
        about the tail's instants.
        """
        ticks = [
            b
            for b in _survey_window("never_rises").blocks
            if b.block_type == BlockType.IDLE and b.duration > 299.0
        ]
        end_time = (ticks[-4].t_stop + TimeDelta(55.0, format="sec")).isot
        night_open = _never_rises(end_time=end_time)
        tail = night_open.blocks[-1]
        assert tail.metadata["reason"] == str(DeferralReason.WINDOW_CLOSED)
        assert night_open.blocks[-2].duration > 299.0
        start, stop = tail.t_start.unix + 10.0, tail.t_stop.unix - 10.0
        assert stop - start > 30.0
        disc = _ClosedDisc(tail.end_pose_az, tail.end_pose_el, start, stop)
        assert _held_in(night_open, disc)

        night_closed = _never_rises(sun_safe=disc, end_time=end_time)

        assert _held_in(night_closed, disc) == []

    def test_a_retune_between_subscans_is_not_held_where_the_zone_reaches(self):
        night_open = _short_subscans_open()
        first = _first(night_open.blocks, lambda b: b.block_type == BlockType.SCIENCE)
        nxt = night_open.blocks[night_open.blocks.index(first) + 1]
        assert nxt.scan_type == "retune"
        stop = first.t_stop.unix
        disc = _ClosedDisc(first.end_pose_az, first.end_pose_el, stop + 60.0, stop + 240.0)
        assert _held_in(night_open, disc)

        night_closed = _survey_night(_SHORT_SUBSCANS, sun_safe=disc)

        assert _held_in(night_closed, disc) == []

    def test_a_wait_between_planet_calibration_passes_is_not_held_where_the_zone_reaches(self):
        night_open = night("planet_scan")
        wait = next(
            b
            for a, b in itertools.pairwise(night_open.blocks)
            if a.scan_type == "planet_cal"
            and b.metadata.get("reason") == str(DeferralReason.WAITING_FOR_PASS)
        )
        start, stop = wait.t_start.unix + 20.0, wait.t_stop.unix - 20.0
        assert stop - start > 30.0
        disc = _ClosedDisc(wait.end_pose_az, wait.end_pose_el, start, stop)
        assert _held_in(night_open, disc)

        night_closed = _survey_night(NIGHTS["planet_scan"], sun_safe=disc)

        assert _held_in(night_closed, disc) == []
        assert any(b.scan_type == "planet_cal" for b in night_closed.blocks)

    def test_a_calibration_visit_holds_its_arrival_pose_until_its_first_pass(self):
        """The detector operations and the wait hold the slew's end pose."""
        targets, start, end, _ = _CALIBRATION_NIGHTS["short_night"]
        night_open = _calibration_night("short_night")
        slew = _first(night_open.blocks, lambda b: b.block_type == BlockType.SLEW)
        first_pass = _first(night_open.blocks, lambda b: b.scan_type == "planet_cal")
        # The slew is planned holding its end pose for the first pass's
        # length; the disc is closed after that and before the pass.
        closes = slew.t_stop.unix + first_pass.duration + 30.0
        opens = first_pass.t_start.unix - 30.0
        assert opens - closes > 60.0
        disc = _ClosedDisc(slew.end_pose_az, slew.end_pose_el, closes, opens)
        assert _held_in(night_open, disc)

        night_closed = plan_calibration_night(targets, get_fyst_site(), start, end, sun_safe=disc)

        assert _held_in(night_closed, disc) == []
        assert any(b.scan_type == "planet_cal" for b in night_closed.blocks)
