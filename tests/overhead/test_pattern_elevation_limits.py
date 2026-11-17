"""A pong or daisy visit keeps its whole scan pattern inside the elevation limits.

The offline scheduler builds every science subscan through the rebuild path
before it books it, and idles with ``reason="unplannable"`` on a refusal.
These tests pin that a patch whose pattern edge, rather than its field
centre, is under the lower limit is not selected, slewed to and refused, and
that the subscan it books is one the planner builds:

- ``_time_inside_el_limits``: how long the whole pattern stays inside the
  limits from when its subscan starts, checked against trajectories the
  planner builds.
- ``_pattern_el_bounds``: every sample of a built pattern lies between the
  bounds, and a pong comes within a hundredth of a degree of its lowest
  corner's, whatever the pattern's ``angle``.
- end to end, through ``generate_timeline``: a daisy near setting and a pong
  near rising on 2026-06-15, the two cases of the sims source list's night,
  where a selection by the field centre slews to the patch and idles
  ``unplannable`` tick after tick.
- the slew's arrival: a setting pattern, a Sun clip or the schedule end that
  closes the window within the slew leaves the telescope where it is, with
  the tick's idle labelled ``unplannable``.
- the zenith: a pong whose box holds the zenith near transit, where its
  track winds its azimuth round the zenith and the planner refuses it, is
  not scanned while the circle about its centre through its corners holds
  the zenith, and the night idles in place rather than slewing to it.
"""

import logging
from itertools import pairwise

import numpy as np
import pytest
from _scheduler_helpers import _initial_state, _make_ctx
from _sun_stubs import fake_sun_model
from astropy.time import Time, TimeDelta

from fyst_trajectories.overhead import (
    CalibrationPolicy,
    ObservingPatch,
    OverheadModel,
    generate_timeline,
    schedule_to_trajectories,
)
from fyst_trajectories.overhead.scheduler import PatchSelectionPhase, SlewPhase
from fyst_trajectories.overhead.scheduler.helpers import (
    _min_subscan_duration,
    _pattern_el_bounds,
    _pong_config,
    _time_inside_el_limits,
    _transit_offset,
)
from fyst_trajectories.patterns import TrajectoryBuilder, compute_pong_period
from fyst_trajectories.patterns.pong import _pong_peak_offsets
from fyst_trajectories.patterns.utils import sky_offsets_to_altaz
from fyst_trajectories.planning import plan_daisy_scan

# M83 with the sims source list's daisy radius. The rebuild's 0.5 deg turn
# radius carries the petals 1.08 deg from the centre, well past 0.3 deg.
_M83 = ObservingPatch(
    name="M83",
    ra_center=204.25,
    dec_center=-29.865,
    width=0.5,
    height=0.5,
    scan_type="daisy",
    velocity=0.3,
    scan_params={"radius": 0.3},
)

# ELAIS-S1 as the sims run it: a 4 x 3 deg pong at 0.5 deg spacing, five terms.
_ELAIS = ObservingPatch(
    name="ELAIS-S1",
    ra_center=8.7667,
    dec_center=-43.58,
    width=4.0,
    height=3.0,
    scan_type="pong",
    velocity=0.5,
    scan_params={"spacing": 0.5, "num_terms": 5},
)

# RA 60, Dec -30 is at el 29 and setting at 19:30 UTC on 2026-06-15; its
# centre reaches the 20 deg limit 2579 s later.
_SETTING_TIME = Time("2026-06-15T19:30:00", scale="utc")


def _setting_patch(scan_type):
    """Build a 4 x 4 deg pong or a 0.3 deg daisy at RA 60, Dec -30."""
    if scan_type == "pong":
        return ObservingPatch(
            name="setting",
            ra_center=60.0,
            dec_center=-30.0,
            width=4.0,
            height=4.0,
            scan_type="pong",
            velocity=0.5,
        )
    return ObservingPatch(
        name="setting",
        ra_center=60.0,
        dec_center=-30.0,
        width=1.0,
        height=1.0,
        scan_type="daisy",
        velocity=0.3,
        scan_params={"radius": 0.3},
    )


def _build(patch, start, duration, site):
    """Build the trajectory the rebuild would run for ``duration`` seconds."""
    if patch.scan_type == "pong":
        builder = TrajectoryBuilder(site).at(ra=patch.ra_center, dec=patch.dec_center)
        return (
            builder.with_config(_pong_config(patch)).duration(duration).starting_at(start).build()
        )
    return plan_daisy_scan(
        patch.ra_center,
        patch.dec_center,
        radius=0.3,
        velocity=patch.velocity,
        turn_radius=0.5,
        avoidance_radius=0.1,
        start_acceleration=0.5,
        site=site,
        start_time=start,
        duration=duration,
    ).trajectory


class TestPatternElevationBounds:
    """The bounds hold every sample the planner builds; a pong's lower one is reached."""

    @pytest.mark.parametrize(
        ("ra", "dec", "when", "angle"),
        [
            (60.0, -30.0, "2026-06-15T19:30:00", 0.0),  # setting, el ~29
            (60.0, -30.0, "2026-06-15T19:30:00", 30.0),
            (8.7667, -43.58, "2026-06-15T06:11:00", -60.0),  # rising, el ~22
            (180.0, -30.0, "2026-06-15T02:00:00", 75.0),  # high, el ~70
        ],
    )
    def test_the_lowest_point_of_a_box_is_a_corner(self, ra, dec, when, angle, coordinates):
        """A 10 x 10 deg box sampled every 0.25 deg: no point of it lies below its corners."""
        patch = ObservingPatch(
            name="box",
            ra_center=ra,
            dec_center=dec,
            width=10.0,
            height=10.0,
            scan_type="pong",
            velocity=0.5,
            scan_params={"angle": angle},
        )
        half_x, half_y = _pong_peak_offsets(_pong_config(patch))
        gx, gy = np.meshgrid(np.linspace(-half_x, half_x, 41), np.linspace(-half_y, half_y, 41))
        turn = np.radians(angle)
        x = gx.ravel() * np.cos(turn) - gy.ravel() * np.sin(turn)
        y = gx.ravel() * np.sin(turn) + gy.ravel() * np.cos(turn)
        t = Time(when, scale="utc")
        _, el = sky_offsets_to_altaz(
            x, y, ra, dec, t + TimeDelta(np.zeros(x.size), format="sec"), coordinates
        )

        lowest, _ = _pattern_el_bounds(patch, t + TimeDelta([0.0], format="sec"), coordinates)
        # The grid holds the four corners, so its minimum is the corners' when
        # no other point is lower.
        assert float(np.min(el)) == pytest.approx(float(lowest[0]), abs=1e-9)

    @pytest.mark.parametrize("angle", [0.0, 30.0, -60.0, 75.0])
    def test_a_turned_pong_stays_between_its_bounds(self, angle, coordinates, site):
        patch = ObservingPatch(
            name="turned",
            ra_center=60.0,
            dec_center=-30.0,
            width=2.0,
            height=1.0,
            scan_type="pong",
            velocity=0.3,
            scan_params={"angle": angle},
        )
        config = _pong_config(patch)
        period, _, _ = compute_pong_period(config)
        builder = TrajectoryBuilder(site).at(ra=60.0, dec=-30.0).with_config(config)
        trajectory = builder.duration(period).starting_at(_SETTING_TIME).build()

        times = _SETTING_TIME + TimeDelta(trajectory.times, format="sec")
        lowest, highest = _pattern_el_bounds(patch, times, coordinates)
        assert np.all(trajectory.el <= highest)
        # Over one period the pattern passes close to every corner of its box,
        # the lowest included; the corners turned the other way would leave a
        # gap or a sample below them for these angles.
        margin = float(np.min(trajectory.el - lowest))
        assert 0.0 <= margin < 0.01

    def test_a_daisy_stays_within_its_reach_of_the_centre(self, coordinates, site):
        patch = _setting_patch("daisy")
        trajectory = _build(patch, _SETTING_TIME, 120.0, site)
        times = _SETTING_TIME + TimeDelta(trajectory.times, format="sec")
        lowest, highest = _pattern_el_bounds(patch, times, coordinates)
        assert np.all(trajectory.el >= lowest)
        assert np.all(trajectory.el <= highest)


class TestTimeInsideElevationLimits:
    """The window ends when any part of the pattern reaches a limit, not its centre."""

    def test_a_field_near_transit_gets_the_whole_window(self, coordinates):
        patch = ObservingPatch(
            name="high",
            ra_center=180.0,
            dec_center=-30.0,
            width=4.0,
            height=4.0,
            scan_type="pong",
            velocity=0.5,
        )
        # RA 180, Dec -30 is at el ~83 at 23:00 UTC, near its transit.
        start = Time("2026-06-15T23:00:00", scale="utc")
        assert _time_inside_el_limits(patch, start, 3600.0, coordinates) == 3600.0

    def test_a_set_field_gets_nothing(self, coordinates):
        # RA 60, Dec -30 is far below the horizon at 03:00 UTC.
        start = Time("2026-06-15T03:00:00", scale="utc")
        _, el = coordinates.radec_to_altaz(60.0, -30.0, start)
        assert el < 20.0
        assert _time_inside_el_limits(_setting_patch("pong"), start, 3600.0, coordinates) == 0.0

    @pytest.mark.parametrize("scan_type", ["pong", "daisy"])
    def test_a_setting_pattern_leaves_before_its_centre_sets(self, scan_type, coordinates, site):
        patch = _setting_patch(scan_type)
        window = _time_inside_el_limits(patch, _SETTING_TIME, 7200.0, coordinates)

        # The centre reaches 20 deg at 2579 s; the pattern's lowest point
        # (a corner of the pong's box, the daisy's reach below the centre)
        # reaches it minutes earlier.
        assert 0.0 < window < 2579.0 - 240.0
        lowest, _ = _pattern_el_bounds(
            patch, _SETTING_TIME + TimeDelta([window, window + 1.0], format="sec"), coordinates
        )
        # The bisection lands on the limit from inside, to within its 0.3 s.
        assert 20.0 <= lowest[0] < 20.01
        assert lowest[1] < 20.0
        # A scan that long builds: the planner keeps it inside the limits.
        trajectory = _build(patch, _SETTING_TIME, window, site)
        assert float(np.min(trajectory.el)) >= 20.0

    def test_a_rising_pattern_counts_from_its_scan_start(self, coordinates):
        """With a retune booked first, the pattern only has to be inside once it starts."""
        start = Time("2026-06-15T06:11:00", scale="utc")
        lowest, _ = _pattern_el_bounds(
            _ELAIS, start + TimeDelta([0.0, 300.0], format="sec"), coordinates
        )
        _, centre = coordinates.radec_to_altaz(_ELAIS.ra_center, _ELAIS.dec_center, start)
        # The centre is up, the box's lowest corner is not, until 300 s later.
        assert centre > 22.0
        assert lowest[0] < 20.0 <= lowest[1]

        assert _time_inside_el_limits(_ELAIS, start, 3600.0, coordinates) == 0.0
        assert _time_inside_el_limits(_ELAIS, start, 3600.0, coordinates, lead=300.0) == 3600.0


class TestPatternEdgeNearTheLimit:
    """Selection and the duration budget follow the whole pattern, end to end.

    Selected by its field centre, the setting daisy is slewed to and found
    unplannable on every tick of its last half hour, and the rising pong on
    the tick before its pattern edge clears the limit. Followed by its
    pattern, each night books science that rebuilds, and no slew to the
    patch ends in an idle tick.
    """

    @pytest.mark.parametrize(
        ("patch", "start", "end"),
        [
            pytest.param(_M83, "2026-06-15T05:00:00", "2026-06-15T06:00:00", id="daisy-setting"),
            pytest.param(_ELAIS, "2026-06-15T05:40:00", "2026-06-15T06:30:00", id="pong-rising"),
        ],
    )
    def test_the_visit_is_booked_and_rebuilds(self, patch, start, end, site, caplog):
        timeline = generate_timeline(patches=[patch], site=site, start_time=start, end_time=end)

        assert timeline.science_blocks, "expected the night to scan the patch"
        assert not [b for b in timeline.blocks if b.metadata.get("reason") == "unplannable"]
        slews = [
            i
            for i, b in enumerate(timeline.blocks)
            if str(b.block_type) == "slew" and b.patch_name == f"slew_to_{patch.name}"
        ]
        assert slews
        for i in slews:
            assert str(timeline.blocks[i + 1].block_type) != "idle"

        with caplog.at_level(logging.WARNING, logger="fyst_trajectories.overhead.simulation"):
            pairs = schedule_to_trajectories(timeline)
        assert not caplog.records
        assert len(pairs) == len(timeline.science_blocks)
        for _, scan in pairs:
            assert float(np.min(scan.trajectory.el)) >= site.telescope_limits.elevation.min


class TestTheVisitStartsOnArrival:
    """A pong or daisy is slewed to only if its shortest subscan still fits on arrival.

    The selection gate asks at the tick, before the slew's length is known;
    a window that closes within the slew would otherwise cost a slew and
    then an idle tick.
    """

    @pytest.mark.parametrize("scan_type", ["pong", "daisy"])
    @pytest.mark.parametrize(("margin", "slews"), [(5.0, False), (60.0, True)])
    def test_a_setting_window_is_checked_at_arrival(
        self, scan_type, margin, slews, site, coordinates
    ):
        """First selected with ``margin`` seconds to spare at the tick, against a ~30 s slew."""
        patch = _setting_patch(scan_type)
        overhead = OverheadModel()
        need = overhead.retune_duration + _min_subscan_duration(patch, overhead)
        # The setting pattern leaves the limits at window_end, whenever its
        # subscan starts; the first tick after the startup calibrations falls
        # ``margin`` seconds before it is too late from the tick itself.
        window = _time_inside_el_limits(patch, _SETTING_TIME, 7200.0, coordinates, lead=300.0)
        tick = _SETTING_TIME + TimeDelta(window - need - margin, format="sec")
        # With no planet targets the startup planet calibration runs without
        # a visibility check, so the startup burst always lasts this long.
        burst = (
            overhead.retune_duration
            + overhead.pointing_cal_duration
            + overhead.focus_duration
            + overhead.skydip_duration
            + overhead.planet_cal_duration
        )
        start = tick - TimeDelta(burst, format="sec")
        timeline = generate_timeline(
            patches=[patch],
            site=site,
            start_time=start,
            end_time=start + TimeDelta(3600.0, format="sec"),
            overhead_model=overhead,
            calibration_policy=CalibrationPolicy(planet_targets=()),
        )

        blocks = timeline.blocks
        first = next(b for b in blocks if str(b.block_type) != "calibration")
        assert abs((first.t_start - tick).sec) < 1e-3
        for before, after in pairwise(blocks):
            if str(before.block_type) == "slew":
                assert str(after.block_type) != "idle"
        if slews:
            assert first.patch_name == f"slew_to_{patch.name}"
            assert timeline.science_blocks
        else:
            assert str(first.block_type) == "idle"
            assert first.metadata["reason"] == "unplannable"
            assert not any(str(b.block_type) == "slew" for b in blocks)
            assert not timeline.science_blocks

    @pytest.mark.parametrize("closes", ["sun", "schedule"])
    def test_a_window_closing_within_the_slew_leaves_the_telescope_parked(self, closes):
        """The Sun clip or the schedule end leaves one second to spare at the tick."""
        patch = ObservingPatch(
            name="Wide01",
            ra_center=180.0,
            dec_center=-30.0,
            width=4.0,
            height=4.0,
            scan_type="pong",
            velocity=0.5,
        )
        start = Time("2026-06-15T02:00:00", scale="utc")
        overhead = OverheadModel()
        need = overhead.retune_duration + _min_subscan_duration(patch, overhead)
        closing = start + TimeDelta(need + 1.0, format="sec")
        if closes == "sun":
            # Every pose turns unsafe at ``closing``: the Sun clip ends the budget.
            model = fake_sun_model(lambda az, el, t: np.asarray(t.unix) < closing.unix)
            ctx = _make_ctx(patches=[patch], sun_safe=model)
        else:
            ctx = _make_ctx(patches=[patch], end_time=closing.isot)
        state = _initial_state(ctx)

        selection = PatchSelectionPhase().run(state, ctx)
        assert selection.selection is patch  # the retune and one period fit from the tick
        result = SlewPhase().run(state, ctx, selection=selection)

        assert [str(b.block_type) for b in result.blocks] == ["idle"]
        idle = result.blocks[0]
        assert idle.metadata["reason"] == "unplannable"
        assert idle.t_start.unix == pytest.approx(start.unix)
        assert (result.state.current_az, result.state.current_el) == (
            state.current_az,
            state.current_el,
        )


def _near_zenith_patch(dec):
    """Build a 4 x 4 deg pong at RA 140, which transits at about 03:16 UTC on 2026-03-01."""
    return ObservingPatch(
        name=f"F140{dec:+.0f}",
        ra_center=140.0,
        dec_center=dec,
        width=4.0,
        height=4.0,
        scan_type="pong",
        velocity=0.5,
    )


# At Dec -23 the centre transits just short of the zenith, well inside the
# circle through the box's corners.
_THROUGH_ZENITH = _near_zenith_patch(-23.0)
_TRANSIT = Time("2026-03-01T03:16:00", scale="utc")


class TestZenithKeepOut:
    """A pattern whose circle holds the zenith is not scanned, and is not slewed to.

    A pong's track passing round the zenith winds its azimuth through whole
    turns, past the azimuth limits, so the planner refuses it; the elevation
    limits cannot see that. The circle about the field centre through the
    pong's corners, or at the daisy's reach, has to exclude the zenith, with
    no margin.
    """

    def test_a_box_holding_the_zenith_gets_nothing(self, coordinates, site):
        lowest, highest = _pattern_el_bounds(
            _THROUGH_ZENITH, _TRANSIT + TimeDelta([0.0], format="sec"), coordinates
        )
        assert lowest[0] > site.telescope_limits.elevation.min + 60.0
        assert highest[0] > 90.0
        assert _time_inside_el_limits(_THROUGH_ZENITH, _TRANSIT, 3600.0, coordinates) == 0.0

    def test_the_window_closes_where_the_circle_reaches_the_zenith(self, coordinates):
        start = Time("2026-03-01T02:30:00", scale="utc")
        window = _time_inside_el_limits(_THROUGH_ZENITH, start, 3600.0, coordinates)
        assert 0.0 < window < 3600.0
        _, highest = _pattern_el_bounds(
            _THROUGH_ZENITH, start + TimeDelta([window, window + 1.0], format="sec"), coordinates
        )
        # The window ends at the last instant verified clear; the circle reaches
        # the zenith within a second after it.
        assert highest[0] < 90.0 <= highest[1]

    def test_a_circle_short_of_the_zenith_keeps_its_window(self, coordinates):
        """At Dec -26 the circle tops out 0.3 deg short of the zenith and scans through transit."""
        patch = _near_zenith_patch(-26.0)
        start = _TRANSIT - TimeDelta(1800.0, format="sec")
        offsets = np.linspace(0.0, 3600.0, 121)
        _, highest = _pattern_el_bounds(
            patch, start + TimeDelta(offsets, format="sec"), coordinates
        )
        assert 89.6 < float(np.max(highest)) < 90.0
        assert _time_inside_el_limits(patch, start, 3600.0, coordinates) == 3600.0

    def test_a_hold_between_two_samples_ends_the_window_before_it(self, coordinates):
        """At Dec -25.64 the circle holds the zenith for 228 s, less than the 300 s step.

        From a start whose samples fall 36 s before the hold and 36 s after
        it, the window still ends where the circle reaches the zenith,
        because the centre's transit, where the circle peaks, is sampled too.
        """
        patch = _near_zenith_patch(-25.64)
        fine = np.arange(-600.0, 600.0, 1.0)
        _, highest = _pattern_el_bounds(
            patch, _TRANSIT + TimeDelta(fine, format="sec"), coordinates
        )
        held = fine[highest >= 90.0]
        assert held[-1] - held[0] + 1.0 == 228.0
        start = _TRANSIT + TimeDelta(held[0] - 1236.0, format="sec")
        steps = np.linspace(0.0, 3600.0, 13)
        _, sampled = _pattern_el_bounds(patch, start + TimeDelta(steps, format="sec"), coordinates)
        assert np.all(sampled < 90.0)

        window = _time_inside_el_limits(patch, start, 3600.0, coordinates)
        assert 0.0 < window < 1236.0
        _, highest = _pattern_el_bounds(
            patch, start + TimeDelta([window, window + 1.0], format="sec"), coordinates
        )
        assert highest[0] < 90.0 <= highest[1]

    def test_the_sampled_transit_is_where_the_circle_peaks(self, coordinates):
        """The centre's transit lands within 1 s of the circle's top, found on a 0.01 s grid.

        The grid spans 5 s either side of the transit, so a transit more
        than a second off finds the top more than a second from it, at the
        true peak or at the grid's edge.
        """
        patch = _near_zenith_patch(-25.64)
        start = _TRANSIT - TimeDelta(1800.0, format="sec")
        transit = _transit_offset(patch, start, coordinates)
        fine = transit + np.arange(-5.0, 5.0, 0.01)
        _, highest = _pattern_el_bounds(patch, start + TimeDelta(fine, format="sec"), coordinates)
        assert abs(fine[np.argmax(highest)] - transit) < 1.0

    def test_a_daisy_hold_between_two_samples_ends_the_window_before_it(self, coordinates):
        """At Dec -23.9 a daisy's circle holds the zenith for 183 s, less than the 300 s step.

        From a start whose samples fall 58 s before the hold and 59 s after
        it, the window still ends where the circle reaches the zenith,
        because the centre's transit, where the circle peaks, is sampled too.
        """
        patch = ObservingPatch(
            name="daisy",
            ra_center=140.0,
            dec_center=-23.9,
            width=1.0,
            height=1.0,
            scan_type="daisy",
            velocity=0.3,
            scan_params={"radius": 0.3},
        )
        fine = np.arange(-600.0, 600.0, 1.0)
        _, highest = _pattern_el_bounds(
            patch, _TRANSIT + TimeDelta(fine, format="sec"), coordinates
        )
        held = fine[highest >= 90.0]
        assert held[-1] - held[0] + 1.0 == 183.0
        start = _TRANSIT + TimeDelta(held[0] - 1258.0, format="sec")
        steps = np.linspace(0.0, 3600.0, 13)
        _, sampled = _pattern_el_bounds(patch, start + TimeDelta(steps, format="sec"), coordinates)
        assert np.all(sampled < 90.0)

        window = _time_inside_el_limits(patch, start, 3600.0, coordinates)
        assert 0.0 < window < 1258.0
        _, highest = _pattern_el_bounds(
            patch, start + TimeDelta([window, window + 1.0], format="sec"), coordinates
        )
        assert highest[0] < 90.0 <= highest[1]

    def test_a_daisy_short_of_the_zenith_keeps_its_window(self, coordinates):
        """The same field as a daisy: its circle tops out at 88.3 deg and scans through transit."""
        patch = ObservingPatch(
            name="daisy",
            ra_center=140.0,
            dec_center=-25.64,
            width=1.0,
            height=1.0,
            scan_type="daisy",
            velocity=0.3,
            scan_params={"radius": 0.3},
        )
        start = _TRANSIT - TimeDelta(1800.0, format="sec")
        offsets = np.linspace(0.0, 3600.0, 3601)
        _, highest = _pattern_el_bounds(
            patch, start + TimeDelta(offsets, format="sec"), coordinates
        )
        assert 88.0 < float(np.max(highest)) < 88.5
        assert _time_inside_el_limits(patch, start, 3600.0, coordinates) == 3600.0

    def test_the_night_idles_in_place_while_the_circle_holds_the_zenith(self, site, coordinates):
        """No slew to the patch ends in an idle tick, and every scan keeps the zenith out.

        A bound clamped at 90 deg would scan through the approach to transit
        and then slew to the patch only to idle ``unplannable``, the planner
        refusing the subscan's azimuth.
        """
        timeline = generate_timeline(
            patches=[_THROUGH_ZENITH],
            site=site,
            start_time="2026-03-01T02:00:00",
            end_time="2026-03-01T04:30:00",
        )
        blocks = timeline.blocks

        assert timeline.science_blocks
        assert not [b for b in blocks if b.metadata.get("reason") == "unplannable"]
        for before, after in pairwise(blocks):
            if str(before.block_type) == "slew":
                assert str(after.block_type) != "idle"
            if str(after.block_type) == "idle":
                assert after.az_start == after.az_end == pytest.approx(before.end_pose_az)
                assert after.elevation == pytest.approx(before.end_pose_el)
        for block in timeline.science_blocks:
            offsets = np.linspace(0.0, block.duration, 31)
            _, highest = _pattern_el_bounds(
                _THROUGH_ZENITH, block.t_start + TimeDelta(offsets, format="sec"), coordinates
            )
            assert np.all(highest < 90.0), block.t_start.isot

        # While the circle holds the zenith the telescope waits where it is.
        offsets = np.arange(0.0, 9000.0, 30.0)
        _, highest = _pattern_el_bounds(
            _THROUGH_ZENITH, timeline.start_time + TimeDelta(offsets, format="sec"), coordinates
        )
        held = timeline.start_time + TimeDelta(offsets[highest >= 90.0], format="sec")
        assert any(str(b.block_type) == "idle" and held[0] <= b.t_start <= held[-1] for b in blocks)
