"""A patch with a detector offset is scheduled for the boresight that puts the detector on it.

The field of an :class:`~fyst_trajectories.overhead.ObservingPatch`, and a
constant-elevation patch's ``elevation``, are where its ``detector_offset``
looks. The offline scheduler scores, limit-checks, Sun-checks, slews to,
holds and records the boresight pose that places that detector there,
solved as the planners solve it, and each science block records the offset
(and, for constant elevation, the detector's elevation) so it rebuilds to
that boresight scan.

The checks here are computed from the rebuilt trajectories, the library's
forward transform :func:`~fyst_trajectories.boresight_to_detector` and the
Sun's position, not from the numbers the scheduler books:

- a constant-elevation block records the elevation its boresight scans at,
  and the detector scans at the patch's elevation;
- the detector of every rebuilt block is on the field, sample for sample:
  its track is the scan of the same block planned with no offset;
- the pass times of the offset nights are those of the nights with none;
- a constant-elevation elevation that puts the boresight outside the
  elevation limits is refused when the context is built;
- the elevation bounds, the Sun clips and the booking screen judge the
  boresight's samples, each shown with a case in both directions;
- the selection scores, the forward check asks about, the slew's wrap holds
  and the held-pose check asks about the boresight's pose;
- the written file holds the boresight's pose, and an on-axis offset gives
  the night no offset gives.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import warnings
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table
from astropy.time import Time, TimeDelta

from fyst_trajectories import (
    PRIMECAM_IM0,
    Coordinates,
    InstrumentOffset,
    boresight_to_detector,
    compute_focal_plane_rotation,
    get_absolute_times,
    get_fyst_site,
    get_primecam_offset,
)
from fyst_trajectories.exceptions import AccelerationLimitWarning, PointingWarning
from fyst_trajectories.offsets import _solve_boresight
from fyst_trajectories.overhead import (
    BlockType,
    CalibrationState,
    DeferralReason,
    MinDurationConstraint,
    ObservingPatch,
    OverheadModel,
    TimelineBlock,
    generate_timeline,
    read_timeline,
    write_timeline,
)
from fyst_trajectories.overhead import simulation as sim
from fyst_trajectories.overhead._execution_screen import _execution_refusal
from fyst_trajectories.overhead.scheduler import SchedulerContext, SchedulerState
from fyst_trajectories.overhead.scheduler.helpers import (
    _boresight_elevations,
    _ce_visit_plan,
    _compute_scan_duration,
    _offset_reach,
    _pattern_el_bounds,
    _pattern_reach,
    _pong_period,
    _time_inside_el_limits,
    _time_until_sun_unsafe,
)
from fyst_trajectories.overhead.scheduler.phases import _screened_build
from fyst_trajectories.planning import FieldRegion, plan_constant_el_scan
from fyst_trajectories.site import AxisLimits, TelescopeLimits

from ._nights import (
    DEEP56,
    DEEP56_IM6,
    F1,
    NIGHTS,
    OFFSET_NIGHTS,
    _rebuild,
    night,
    rebuilt,
)

SITE = get_fyst_site()
COORDS = Coordinates(SITE)

#: How far the detector of a rebuilt block may sit from where the same block
#: planned with no offset puts it, in arcseconds: the boresight solve settles
#: to 1e-12 deg, so this holds far more than it needs.
_ON_FIELD_ARCSEC = 1e-3

#: How far inside a pass's analytic azimuth envelope its rebuilt samples may
#: stop, in degrees: a turnaround's apex falls between two samples.
_ENVELOPE_SAMPLING_DEG = 1e-4


def _science(name):
    """Return the named night's science blocks with their rebuilt scans."""
    return [(b, s) for b, s in rebuilt(name) if b.block_type == BlockType.SCIENCE]


def _offset_of(block) -> InstrumentOffset:
    return InstrumentOffset(**block.metadata["detector_offset"])


def _detector_track(trajectory, offset):
    """Apply ``offset`` to a boresight trajectory with the library's forward transform."""
    rotation = compute_focal_plane_rotation(trajectory.el, site=SITE, offset=offset)
    return boresight_to_detector(trajectory.az, trajectory.el, offset, rotation)


def _without_offset(block):
    """Return ``block`` as it would be recorded for the boresight on the same field."""
    meta = {
        k: v
        for k, v in block.metadata.items()
        if k not in ("detector_offset", "detector_elevation")
    }
    elevation = block.metadata.get("detector_elevation", block.elevation)
    return dataclasses.replace(block, metadata=meta, elevation=elevation)


def _quiet_rebuild(block):
    """Rebuild one block, its scan's advisories set aside by message as :func:`rebuilt` does."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="High elevation reduces on-sky azimuth speed",
            category=PointingWarning,
        )
        warnings.filterwarnings(
            "ignore", message="Trajectory azimuth acceleration", category=AccelerationLimitWarning
        )
        return sim._generate_trajectory_for_block(block, SITE).trajectory


def _wrapped(delta):
    return (np.asarray(delta, dtype=float) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# The constant-elevation night
# ---------------------------------------------------------------------------


def test_a_constant_elevation_block_records_the_boresight_elevation():
    """Each block's ``elevation`` is its rebuilt scan's; the detector's is the patch's.

    The bay ``IM6`` sits below and to the left of the boresight at el 50, so
    the boresight scans about 1.66 deg higher than the patch's elevation.
    """
    blocks = _science("ce_offset")
    assert len(blocks) >= 2
    for block, scan in blocks:
        traj = scan.trajectory
        where = f"{block.t_start.isot}"
        assert block.metadata["detector_elevation"] == DEEP56_IM6.elevation, where
        assert block.metadata["detector_offset"] == dataclasses.asdict(
            get_primecam_offset("IM6")
        ), where
        assert abs(block.elevation - float(traj.el[0])) <= 1e-9, where
        assert np.all(traj.el == traj.el[0]), where
        assert 1.6 < block.elevation - DEEP56_IM6.elevation < 1.7, where


@pytest.mark.parametrize("name", sorted(OFFSET_NIGHTS))
def test_an_offset_night_validates(name):
    assert night(name).validate() == []


@pytest.mark.parametrize("name", sorted(OFFSET_NIGHTS))
def test_the_pass_times_are_the_nights_without_an_offset(name):
    """A constant-elevation pass is timed by the detector's crossing, so the offset moves no pass.

    Every constant-elevation science block of the offset night spans the
    samples of the pass the night with no offset books, from the same
    anchor: the same start, stop and anchor. A pong or daisy is not held to
    this, since the boresight's pattern is not the field's.
    """
    with_offset = [b for b, _ in _science(name) if b.scan_type == "constant_el"]
    without = [b for b, _ in _science(OFFSET_NIGHTS[name]) if b.scan_type == "constant_el"]
    if not without:
        assert not with_offset
        return
    assert with_offset
    assert [(b.t_start.isot, b.t_stop.isot, b.metadata["t0_scan"]) for b in with_offset] == [
        (b.t_start.isot, b.t_stop.isot, b.metadata["t0_scan"]) for b in without
    ]


@pytest.mark.parametrize("name", ["ce_offset", "lsa_window_offset"])
def test_the_slew_is_planned_and_held_for_the_boresight_scan(name, monkeypatch):
    """The span the slew's wrap must hold is the boresight's sweep; the pose held is its elevation.

    The night is generated again with the envelope and the held-pose check
    recorded. Every envelope a visit's slew is planned with is the azimuth
    range the rebuilt science blocks sweep, on the same turn, and every pose
    held before a subscan is asked at the boresight's elevation, the one the
    rebuilt scan runs at.
    """
    from fyst_trajectories.overhead.scheduler import phases

    envelopes, held = [], []
    real_envelope, real_holds = phases.SlewPhase._visit_az_envelope, phases._holds

    def envelope(*args, **kwargs):
        out = real_envelope(*args, **kwargs)
        envelopes.append(out)
        return out

    def holds(ctx, az, el, start, hold):
        held.append(el)
        return real_holds(ctx, az, el, start, hold)

    monkeypatch.setattr(phases.SlewPhase, "_visit_az_envelope", staticmethod(envelope))
    monkeypatch.setattr(phases, "_holds", holds)
    patches, start, end, kwargs = NIGHTS[name]
    timeline = generate_timeline(
        patches=patches, site=SITE, start_time=start, end_time=end, **kwargs
    )
    scans = [s.trajectory for b, s in _rebuild(timeline) if b.block_type == BlockType.SCIENCE]
    az = np.concatenate([t.az for t in scans])
    assert [e for e in envelopes if e is not None]
    for lo, hi in (e for e in envelopes if e is not None):
        turn = 360.0 * round((0.5 * (lo + hi) - float(np.mean(az))) / 360.0)
        assert lo - 1e-9 <= float(np.min(az)) + turn < lo + _ENVELOPE_SAMPLING_DEG
        assert hi - _ENVELOPE_SAMPLING_DEG < float(np.max(az)) + turn <= hi + 1e-9
    assert held
    assert set(held) == {float(scans[0].el[0])}


@pytest.mark.parametrize("name", sorted(OFFSET_NIGHTS))
def test_the_detector_scans_the_field(name):
    """The forward transform of each rebuilt block puts the detector on the field.

    The boresight trajectory of every science block is carried to the
    detector with :func:`~fyst_trajectories.boresight_to_detector`, at the
    rotation of the boresight's own elevation, and compared with the same
    block planned with no offset (for constant elevation, at the detector's
    elevation): sample for sample, in azimuth on the sky and in elevation.
    """
    worst = 0.0
    for block, scan in _science(name):
        offset = _offset_of(block)
        det_az, det_el = _detector_track(scan.trajectory, offset)
        plain = _quiet_rebuild(_without_offset(block))
        assert np.array_equal(plain.times, scan.trajectory.times)
        d_az = _wrapped(det_az - plain.az) * np.cos(np.radians(plain.el))
        d_el = det_el - plain.el
        worst = max(worst, float(np.max(np.hypot(d_az, d_el))) * 3600.0)
        if block.scan_type == "constant_el":
            assert np.max(np.abs(det_el - block.metadata["detector_elevation"])) * 3600.0 < (
                _ON_FIELD_ARCSEC
            )
    assert worst < _ON_FIELD_ARCSEC


# ---------------------------------------------------------------------------
# The context refuses a boresight outside the elevation limits
# ---------------------------------------------------------------------------


def _ce_patch(elevation, offset):
    return dataclasses.replace(
        DEEP56, name="Deep56Bay", elevation=elevation, detector_offset=offset
    )


def _build(patches, site=SITE):
    return SchedulerContext.build(
        patches=patches,
        site=site,
        start_time=Time("2026-06-15T07:30:00", scale="utc"),
        end_time=Time("2026-06-15T08:30:00", scale="utc"),
    )


def test_a_boresight_under_the_floor_is_refused_at_build():
    """``IM3`` at el 21 needs the boresight at el 19.25, under the 20 deg floor."""
    with pytest.raises(ValueError, match=r"Deep56Bay.*IM3.*21\.000.*19\.2"):
        _build([_ce_patch(21.0, get_primecam_offset("IM3"))])
    _build([_ce_patch(23.0, get_primecam_offset("IM3"))])


def _boresight_el(detector_el, offset):
    """Return the elevation of the boresight that holds the detector at ``detector_el``.

    The focal-plane rotation turns with the elevation alone, so the solve
    gives the same elevation at every azimuth.
    """
    return float(_solve_boresight(0.0, detector_el, offset, site=SITE)[1])


def _with_floor(floor):
    limits = SITE.telescope_limits
    elevation = dataclasses.replace(limits.elevation, min=floor)
    return dataclasses.replace(
        SITE, telescope_limits=TelescopeLimits(azimuth=limits.azimuth, elevation=elevation)
    )


def test_a_boresight_exactly_at_the_floor_is_inside():
    """The lower limit is inside, as the limits have it; a float step above the boresight is not."""
    offset = get_primecam_offset("IM3")
    bore = _boresight_el(30.0, offset)
    assert isinstance(SITE.telescope_limits.elevation, AxisLimits)
    _build([_ce_patch(30.0, offset)], site=_with_floor(bore))
    with pytest.raises(ValueError, match="Deep56Bay"):
        _build([_ce_patch(30.0, offset)], site=_with_floor(math.nextafter(bore, math.inf)))


# ---------------------------------------------------------------------------
# The pattern bounds, the Sun clips and the screen judge the boresight
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("patch_name", ["F1", "F3"])
def test_the_elevation_bounds_hold_the_boresight_pattern(patch_name):
    """Every rebuilt boresight sample lies between the mapped bounds, and above the floor.

    The detector's pattern bounds, mapped to the boresight's elevation,
    bound the boresight's samples. Over a pong period the pattern passes
    near its lowest corner, so the lower bound lies within a hundredth of a
    degree of the lowest sample.
    """
    patches = {p.name: p for p in NIGHTS["pong_and_daisy_offset"][0]}
    patch = patches[patch_name]
    blocks = [(b, s) for b, s in _science("pong_and_daisy_offset") if b.patch_name == patch_name]
    assert blocks
    floor = SITE.telescope_limits.elevation.min
    for block, scan in blocks:
        traj = scan.trajectory
        lowest, highest = (
            _boresight_elevations(bound, patch.detector_offset, SITE)
            for bound in _pattern_el_bounds(patch, get_absolute_times(traj), COORDS)
        )
        assert np.all(traj.el >= floor)
        assert np.all(traj.el >= lowest - 1e-9), block.t_start.isot
        assert np.all(traj.el <= highest + 1e-9), block.t_start.isot
        if patch.scan_type == "pong":
            assert float(np.min(traj.el - lowest)) < 0.01


def _inside_window(bay, start):
    patch = dataclasses.replace(
        F1, detector_offset=None if bay is None else get_primecam_offset(bay)
    )
    return _time_inside_el_limits(patch, start, 7200.0, COORDS)


def test_a_setting_pattern_leaves_when_either_pattern_leaves_the_limits():
    """The window ends where the detector's pattern or the boresight's first reaches the floor.

    F1 sets after 12:00 UTC. ``IM3`` puts the boresight about 1.75 deg below
    the detector there, so its window ends before the window with no offset,
    by about the time the field takes to sink that far; ``IM6`` puts the
    boresight above the detector, so the detector's own pattern ends it, as
    with no offset. At each window's end the bound of the pattern that ended
    it sits at the floor, computed here from the pattern bounds and the solve.
    """
    start = Time("2026-12-15T12:00:00", scale="utc")
    floor = SITE.telescope_limits.elevation.min
    windows = {bay: _inside_window(bay, start) for bay in (None, "IM3", "IM6")}
    assert 0.0 < windows["IM3"] < windows[None] - 300.0
    assert windows["IM6"] == windows[None]
    for bay, window in windows.items():
        end = start + TimeDelta([window], format="sec")
        lowest = _pattern_el_bounds(F1, end, COORDS)[0]
        if bay == "IM3":
            lowest = _boresight_elevations(lowest, get_primecam_offset(bay), SITE)
        assert floor <= float(lowest[0]) < floor + 0.01, (bay, float(lowest[0]))


def test_the_widened_reach_bounds_a_boresight_pattern_the_plain_reach_does_not():
    """A pong with ``IM3`` strays past its reach about the boresight's centre, inside the widening.

    One period of a 10 deg pong at Dec -60 (el 21 to 37): the farthest
    boresight sample from the boresight that holds the detector on the field
    centre, at the same instant, lies more than 0.1 deg past the pattern's
    reach and within the widened bound.
    """
    offset = get_primecam_offset("IM3")
    patch = ObservingPatch(
        name="P",
        ra_center=60.0,
        dec_center=-60.0,
        width=10.0,
        height=10.0,
        scan_type="pong",
        velocity=0.5,
        detector_offset=offset,
    )
    start = Time("2026-06-15T10:00:00", scale="utc")
    traj = _quiet_rebuild(
        TimelineBlock.science(patch, start, _pong_period(patch), 0.0, 0.0, 0.0, 0)
    )
    times = get_absolute_times(traj)
    n = traj.az.size
    c_az, c_el = COORDS.radec_to_altaz(
        np.full(n, patch.ra_center), np.full(n, patch.dec_center), times
    )
    b_az, b_el = _solve_boresight(np.asarray(c_az), np.asarray(c_el), offset, site=SITE)
    far = float(np.max(COORDS.angular_separation(traj.az, traj.el, b_az, b_el)))
    reach = _pattern_reach(patch)
    assert reach + 0.1 < far <= _offset_reach(reach, np.asarray(c_el), offset, SITE)


def test_the_pattern_sun_clip_widens_the_reach_for_the_offset():
    """The clip keeps the boresight's centre the widened reach out, not the plain reach.

    The radius sits halfway between the two: the boresight's centre is more
    than ``radius + reach`` from the Sun and less than ``radius`` plus the
    widened reach, so the visit is clipped.
    """
    offset = get_primecam_offset("IM3")
    reach = _pattern_reach(F1)
    _, bore = _centre_seps(offset)
    instants = _CLIP_TIME + TimeDelta([0.0, 60.0], format="sec")
    _, centre_el = COORDS.radec_to_altaz(
        np.full(2, F1.ra_center), np.full(2, F1.dec_center), instants
    )
    widening = _offset_reach(reach, np.asarray(centre_el), offset, SITE) - reach
    assert widening > 0.2
    radius = bore - reach - 0.5 * widening
    window = _time_until_sun_unsafe(
        F1.ra_center,
        F1.dec_center,
        _CLIP_TIME,
        60.0,
        COORDS,
        radius,
        reach=reach,
        detector_offset=offset,
    )
    assert window == 0.0


def test_the_pattern_sun_clip_of_a_visit_follows_the_boresight():
    """The duration of a pong visit is clipped where its boresight's pattern meets the Sun zone.

    At a radius between the boresight's and the field centre's clearance the
    visit with ``IM6`` gets no time, the one without an offset does.
    """
    offset = get_primecam_offset("IM6")
    centre, bore = _centre_seps(offset)
    radius = 0.5 * (centre + bore) - _pattern_reach(F1)
    site = get_fyst_site(sun_exclusion_radius=radius, sun_warning_radius=radius + 5.0)
    windows = {
        bay: _compute_scan_duration(
            dataclasses.replace(F1, detector_offset=offset if bay else None),
            _CLIP_TIME,
            _CLIP_TIME + TimeDelta(3600, format="sec"),
            site,
            Coordinates(site),
            OverheadModel(),
            ce_ready_lead=0.0,
        )
        for bay in (False, True)
    }
    assert windows[True] == 0.0 < windows[False]


def _sep(az, el, times):
    sun_az, sun_el = COORDS.get_sun_altaz(times)
    return np.asarray(COORDS.angular_separation(az, el, sun_az, sun_el), dtype=float)


#: The pong's field at 12:00 UTC, el 38.6 and the Sun up at el 29.4; with
#: ``IM6`` the boresight that holds the field centre sits 1.70 deg nearer
#: the Sun than the centre, with ``IM3`` 1.72 deg farther.
_CLIP_TIME = Time("2026-12-15T12:00:00", scale="utc")


def _centre_seps(offset):
    """Return the Sun separations of the field centre and of the boresight that holds it."""
    az, el = COORDS.radec_to_altaz(F1.ra_center, F1.dec_center, _CLIP_TIME)
    bore_az, bore_el = _solve_boresight(float(az), float(el), offset, site=SITE)
    return float(_sep(az, el, _CLIP_TIME)), float(_sep(bore_az, bore_el, _CLIP_TIME))


@pytest.mark.parametrize(("bay", "clips"), [("IM6", True), ("IM3", False)])
def test_the_pattern_sun_clip_follows_the_boresight(bay, clips):
    """A radius the field centre's pattern clears and the boresight's does not clips, and back.

    The radius is set from the separations: with ``IM6`` the boresight
    centre sits at the radius plus the pattern's reach, inside the clip,
    while the field centre is clear of it; with ``IM3`` the field centre
    sits there and the boresight is farther out by more than the clip's
    allowance for the turning offset.
    """
    offset = get_primecam_offset(bay)
    reach = _pattern_reach(F1)
    centre, bore = _centre_seps(offset)
    if clips:
        assert centre - bore > 1.0
        radius = bore - reach + 1e-3
    else:
        assert bore - centre > 1.0
        radius = centre - reach + 1e-3
    for detector_offset, expect_clip in ((offset, clips), (None, not clips)):
        window = _time_until_sun_unsafe(
            F1.ra_center,
            F1.dec_center,
            _CLIP_TIME,
            60.0,
            COORDS,
            radius,
            reach=reach,
            detector_offset=detector_offset,
        )
        assert (window == 0.0) == expect_clip, (detector_offset, window)


#: A daytime constant-elevation field: its setting pass at el 40 opens at
#: 15:38:18 UTC on 2026-12-15, its sweep then about 61 deg from the Sun.
_CE_DAY = ObservingPatch(
    name="CEDay",
    ra_center=200.0,
    dec_center=-20.0,
    width=6.0,
    height=6.0,
    scan_type="constant_el",
    velocity=0.5,
    elevation=40.0,
)


def _ce_case(offset):
    """Return the patch, its visit start, its admitted pass, its rebuilt scan and the detector's.

    The pass is rebuilt whole, from the visit start to after it closes, so
    the scan's azimuth envelope is the sweep's.
    """
    patch = dataclasses.replace(_CE_DAY, detector_offset=offset)
    start = Time("2026-12-15T15:30:00", scale="utc")
    end = start + TimeDelta(7200, format="sec")
    plan = _ce_visit_plan(patch, patch.elevation, start, end, COORDS, {}, 600.0)
    assert plan is not None
    rising, _, t_close = plan
    block = TimelineBlock.science(
        patch,
        start,
        float((t_close - start).sec) + 60.0,
        0.0,
        0.0,
        0.0,
        0,
        rising=rising,
        t0_scan=start.isot,
    )
    traj = _quiet_rebuild(block)
    det_az, _ = _detector_track(traj, offset)
    return patch, start, plan, traj, det_az


def _min_span_sep(az, el, when):
    """Return the least Sun separation over every azimuth from ``az.min()`` to ``az.max()``."""
    grid = np.linspace(float(np.min(az)), float(np.max(az)), 4001)
    times = when + TimeDelta(np.zeros(grid.size), format="sec")
    return float(np.min(_sep(grid, np.full(grid.size, el), times)))


@pytest.mark.parametrize("bay", ["IM6", "IM3"])
def test_the_constant_elevation_sun_clip_follows_the_boresight(bay):
    """The clip asks about the boresight's sweep at the boresight's elevation.

    The radius is set between the least Sun separation over the boresight's
    sweep and over the detector's: the clip closes the visit at its start
    when the boresight's is the nearer, and keeps it open when the
    detector's is.
    """
    offset = get_primecam_offset(bay)
    patch, start, plan, traj, det_az = _ce_case(offset)
    bore = _min_span_sep(traj.az, float(traj.el[0]), start)
    det = _min_span_sep(det_az, patch.elevation, start)
    assert abs(bore - det) > 0.5
    radius = 0.5 * (bore + det)
    site = get_fyst_site(sun_exclusion_radius=radius, sun_warning_radius=radius + 5.0)
    window = _compute_scan_duration(
        patch,
        start,
        start + TimeDelta(7200, format="sec"),
        site,
        Coordinates(site),
        OverheadModel(),
        ce_ready_lead=600.0,
        ce_plan=plan,
    )
    assert (window == 0.0) == (bore < det), (bore, det, window)


def _screen_case(bay):
    """Return a daytime pong block with ``bay``, its boresight scan and the detector track."""
    patch = dataclasses.replace(F1, detector_offset=get_primecam_offset(bay))
    block = TimelineBlock.science(patch, _CLIP_TIME, _pong_period(patch), 0.0, 0.0, 0.0, 0)
    traj = _quiet_rebuild(block)
    det_az, det_el = _detector_track(traj, patch.detector_offset)
    times = get_absolute_times(traj)
    return (
        block,
        float(np.min(_sep(traj.az, traj.el, times))),
        float(np.min(_sep(det_az, det_el, times))),
    )


@pytest.mark.parametrize("bay", ["IM3", "IM6"])
def test_the_booking_screen_judges_the_boresight_samples(bay):
    """A radius between the boresight's and the detector's least separations: refused or booked.

    With ``IM6`` the boresight comes nearer the Sun than the detector does,
    and the screen refuses the block ``sun_scan``; with ``IM3`` the detector
    comes nearer, and the screen books it.
    """
    block, bore, det = _screen_case(bay)
    assert abs(bore - det) > 0.5
    radius = 0.5 * (bore + det)
    site = get_fyst_site(sun_exclusion_radius=radius, sun_warning_radius=radius + 5.0)
    built = _screened_build(block, site, None, None)
    traj = _quiet_rebuild(block)
    refusal = _execution_refusal(traj, "pong", block.metadata["scan_params"], site, None)
    if bore < det:
        assert built is DeferralReason.SUN_SCAN
        assert refusal is DeferralReason.SUN_SCAN
    else:
        assert not isinstance(built, DeferralReason)
        assert refusal is None
    assert (bay == "IM6") == (bore < det)


# ---------------------------------------------------------------------------
# The selection and the forward check ask about the boresight
# ---------------------------------------------------------------------------

#: One patch of each scan type, each with a bay that moves the boresight.
_SELECTION_PATCHES = [
    DEEP56_IM6,
    dataclasses.replace(F1, detector_offset=get_primecam_offset("IM3")),
    dataclasses.replace(
        F1, name="F1daisy", scan_type="daisy", detector_offset=get_primecam_offset("IM5")
    ),
]


def _field_bore(patch, when):
    """Return the field centre and the boresight that holds the detector there (CE: at its el)."""
    az, el = COORDS.radec_to_altaz(patch.ra_center, patch.dec_center, when)
    el = float(el) if patch.elevation is None else patch.elevation
    bore_az, bore_el = _solve_boresight(float(az), el, patch.detector_offset, site=SITE)
    return (float(az), el), (float(bore_az), float(bore_el))


def test_the_selection_scores_the_boresight_pose(monkeypatch):
    """Each patch is scored at its boresight pose, and the pose handed on is the boresight's."""
    from fyst_trajectories.overhead.scheduler import phases

    asked = {}
    real = phases._evaluate_patch

    def recording(patch, time, az, el, coords, constraints):
        asked[patch.name] = (float(az), float(el))
        return real(patch, time, az, el, coords, constraints)

    monkeypatch.setattr(phases, "_evaluate_patch", recording)
    ctx = SchedulerContext.build(
        patches=_SELECTION_PATCHES,
        site=SITE,
        start_time=Time("2026-12-15T08:30:00", scale="utc"),
        end_time=Time("2026-12-15T11:00:00", scale="utc"),
    )
    state = SchedulerState.initial(start_time=ctx.start_time, cal_state=CalibrationState())
    result = phases.PatchSelectionPhase().run(state, ctx)
    for patch in _SELECTION_PATCHES:
        field, bore = _field_bore(patch, ctx.start_time)
        assert asked[patch.name] == pytest.approx(bore, abs=1e-9), patch.name
        assert float(COORDS.angular_separation(*field, *bore)) > 1.0, patch.name
    assert result.selection is not None
    chosen = result.selection
    if chosen.scan_type != "constant_el":
        _, bore = _field_bore(chosen, ctx.start_time)
        assert result.best_el == pytest.approx(bore[1], abs=1e-9)


@pytest.mark.parametrize(
    ("patch", "at"),
    list(zip(_SELECTION_PATCHES, ["02:00", "08:30", "08:30"], strict=True)),
    ids=lambda v: getattr(v, "name", v),
)
def test_the_forward_check_asks_about_the_boresight(patch, at):
    """The forward Sun check of :class:`MinDurationConstraint` is asked at the boresight pose.

    Each field is up at the time given (UTC, 2026-12-15).
    """
    asked = []

    def sun_safe(az, el, when):
        asked.append((az, el))
        return True

    minimum = 600.0
    when = Time(f"2026-12-15T{at}:00", scale="utc")
    assert MinDurationConstraint(minimum, sun_safe=sun_safe).score(patch, when, 0.0, 0.0, COORDS)
    _, bore = _field_bore(patch, when + TimeDelta(minimum, format="sec"))
    assert asked == [pytest.approx(bore, abs=1e-9)]


def test_the_forward_check_needs_both_the_field_and_the_boresight_above_the_floor():
    """A pong whose field is above the floor in ten minutes but whose boresight is not scores 0.

    The field sets through el 20.77 at 13:21 UTC, where ``IM3`` puts the
    boresight at 19.02; the same field and time score 1 with ``IM6``, which
    raises the boresight to 22.54, and with no offset.
    """
    when = Time("2026-12-15T13:11:00", scale="utc")
    minimum = 600.0
    field_el = float(
        COORDS.radec_to_altaz(F1.ra_center, F1.dec_center, when + TimeDelta(minimum, format="sec"))[
            1
        ]
    )
    assert 20.0 < field_el < 21.5
    check = MinDurationConstraint(minimum)
    scores = {
        bay: check.score(
            dataclasses.replace(
                F1, detector_offset=None if bay is None else get_primecam_offset(bay)
            ),
            when,
            0.0,
            0.0,
            COORDS,
        )
        for bay in (None, "IM3", "IM6")
    }
    assert scores == {None: 1.0, "IM3": 0.0, "IM6": 1.0}


# ---------------------------------------------------------------------------
# The file a collaborator reads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(OFFSET_NIGHTS))
def test_the_file_holds_the_boresight_pose(name, tmp_path):
    """The written rows carry the boresight's elevation and azimuths, and the offset by value.

    A science row's ``el`` is its rebuilt scan's first elevation, its
    ``azmin`` and ``azmax`` the rebuilt scan's azimuth envelope on the
    turn the row records, and its ``boresight_angle`` the roll of that
    elevation. Read back, each block records the offset field for field.
    """
    timeline = night(name)
    path = Path(tmp_path) / f"{name}.ecsv"
    write_timeline(timeline, path)
    table = Table.read(path, format="ascii.ecsv")
    read = read_timeline(path)
    assert len(table) == len(timeline.blocks) == len(read.blocks)
    pairs = {id(b): s for b, s in _science(name)}
    checked = 0
    for row, block, back in zip(table, timeline.blocks, read.blocks, strict=True):
        if block.block_type != BlockType.SCIENCE:
            continue
        checked += 1
        traj = pairs[id(block)].trajectory
        el = float(row["el"])
        assert abs(el - block.elevation) <= 1e-9
        assert float(row["boresight_angle"]) == pytest.approx(-SITE.nasmyth_sign * el, abs=1e-9)
        if block.scan_type == "constant_el":
            assert abs(el - float(traj.el[0])) <= 1e-9
            detector_el = block.metadata["detector_elevation"]
            assert abs(el - _boresight_el(detector_el, _offset_of(block))) <= 1e-9
            assert abs(el - detector_el) > 0.1
        else:
            assert abs(el - float(traj.el[0])) <= 0.01
        turn = 360.0 * round((float(row["azmin"]) - float(np.min(traj.az))) / 360.0)
        assert float(row["azmin"]) == pytest.approx(float(np.min(traj.az)) + turn, abs=1e-6)
        assert float(row["azmax"]) == pytest.approx(float(np.max(traj.az)) + turn, abs=1e-6)
        assert back.metadata["detector_offset"] == block.metadata["detector_offset"]
        assert back.metadata.get("detector_elevation") == block.metadata.get("detector_elevation")
    assert checked


def test_a_window_patch_with_an_offset_rebuilds_to_the_planner():
    """Each block of the sidereal-window night is the planner's own pass, cut to the block."""
    for block, scan in _science("lsa_window_offset"):
        meta = block.metadata
        planned = plan_constant_el_scan(
            field=FieldRegion(meta["ra_center"], meta["dec_center"], meta["width"], meta["height"]),
            elevation=meta["detector_elevation"],
            velocity=meta["velocity"],
            site=SITE,
            start_time=Time(meta["t0_scan"], scale="utc"),
            lsa_window=meta["scan_params"]["lsa_window"],
            detector_offset=_offset_of(block),
        ).trajectory
        times = get_absolute_times(planned)
        inside = (times >= block.t_start - TimeDelta(1e-3, format="sec")) & (times < block.t_stop)
        assert np.array_equal(planned.az[inside], scan.trajectory.az)
        assert np.array_equal(planned.el[inside], scan.trajectory.el)


# ---------------------------------------------------------------------------
# An on-axis offset changes nothing
# ---------------------------------------------------------------------------


@functools.cache
def _zero_offset_night():
    patches, start, end, kwargs = NIGHTS["zero_settle"]
    zero = [dataclasses.replace(p, detector_offset=PRIMECAM_IM0) for p in patches]
    return generate_timeline(patches=zero, site=SITE, start_time=start, end_time=end, **kwargs)


def test_an_on_axis_offset_gives_the_night_none_gives(tmp_path):
    plain, zero = night("zero_settle"), _zero_offset_night()
    write_timeline(plain, tmp_path / "plain.ecsv")
    write_timeline(zero, tmp_path / "zero.ecsv")
    assert (tmp_path / "plain.ecsv").read_bytes() == (tmp_path / "zero.ecsv").read_bytes()
    pairs = _rebuild(zero)
    want = rebuilt("zero_settle")
    assert len(pairs) == len(want)
    for (_, got), (_, expected) in zip(pairs, want, strict=True):
        for attr in ("times", "az", "el"):
            assert np.array_equal(getattr(got.trajectory, attr), getattr(expected.trajectory, attr))


def test_the_scheduler_does_not_read_the_silent_azimuth_shift():
    """The scheduler solves each pose; it never reads the shift that is 0.0 near the zenith."""
    import fyst_trajectories.overhead.constraints as constraints
    import fyst_trajectories.overhead.scheduler as scheduler

    sources = [Path(constraints.__file__)] + sorted(Path(scheduler.__file__).parent.glob("*.py"))
    for source in sources:
        assert "_boresight_az_offset" not in source.read_text(encoding="utf-8"), source


def test_a_patch_with_an_offset_is_scheduled():
    """The context takes a patch with an offset of every scan type."""
    patches = [
        dataclasses.replace(DEEP56_IM6),
        dataclasses.replace(F1, detector_offset=get_primecam_offset("IM2")),
        ObservingPatch(
            name="daisy",
            ra_center=200.0,
            dec_center=-20.0,
            width=6.0,
            height=6.0,
            scan_type="daisy",
            velocity=0.5,
            detector_offset=get_primecam_offset("IM4"),
        ),
    ]
    ctx = _build(patches)
    assert [p.detector_offset for p in ctx.patches] == [p.detector_offset for p in patches]
