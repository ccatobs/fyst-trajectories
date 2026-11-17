"""What each planner's returned fields say, held against the trajectory it built.

A planner returns a trajectory and, beside it, fields that describe it: a
duration, a window, a centre, the parameters it was planned with. Each test
here reads one such field and measures the same quantity on the samples
themselves, or follows a planned block through the hand-off a control
system makes with it.
"""

import math

import numpy as np
import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories import (
    SCAN_FLAG_SCIENCE,
    boresight_to_detector,
    choose_encoder_solution,
    compute_focal_plane_rotation,
    estimate_slew_time,
    get_fyst_site,
    get_primecam_offset,
    rewrap_trajectory_azimuth,
    sky_to_focal_plane,
    to_path_format,
    to_path_payload,
)
from fyst_trajectories.exceptions import TargetNotObservableError
from fyst_trajectories.offsets import apply_detector_offset
from fyst_trajectories.patterns import PongScanConfig
from fyst_trajectories.patterns.pong import PongScanPattern, compute_pong_period
from fyst_trajectories.planning import (
    FieldRegion,
    plan_constant_el_scan,
    plan_daisy_altaz_scan,
    plan_daisy_scan,
    plan_pong_altaz_scan,
    plan_pong_rotation_scans,
    plan_pong_scan,
    plan_source_ces,
)
from fyst_trajectories.trajectory import Trajectory

_NIGHT = Time("2026-03-15T00:00:00", scale="utc")
_EVENING = Time("2026-03-15T01:00:00", scale="utc")
#: A pong within the site's advisory speed and acceleration limits.
_PONG_FIELD = FieldRegion(ra_center=180.0, dec_center=-30.0, width=1.0, height=1.0)
_PONG = {"velocity": 0.3, "spacing": 0.2}
_ECDFS = FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7)


def _short_blocks(site):
    """One short block from each planner, keyed by a readable name."""
    daisy = dict(
        radius=0.5,
        velocity=0.3,
        turn_radius=0.2,
        avoidance_radius=0.0,
        start_acceleration=0.5,
        duration=60.0,
        site=site,
        start_time=_EVENING,
    )
    return {
        "pong": lambda: plan_pong_scan(_PONG_FIELD, **_PONG, site=site, start_time=_EVENING),
        "pong rotation": lambda: plan_pong_rotation_scans(
            _PONG_FIELD, n_rotations=2, **_PONG, site=site, start_time=_EVENING
        )[1],
        "pong altaz": lambda: plan_pong_altaz_scan(
            120.0, 50.0, width=1.0, height=1.0, **_PONG, site=site, start_time=_EVENING
        ),
        "daisy": lambda: plan_daisy_scan(180.0, -30.0, **daisy),
        "daisy altaz": lambda: plan_daisy_altaz_scan(120.0, 50.0, **daisy),
        "constant-el field": lambda: plan_constant_el_scan(
            FieldRegion(ra_center=53.117, dec_center=-27.808, width=1.0, height=1.0),
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=Time("2026-03-15T17:00:00", scale="utc"),
        ),
        "constant-el window": lambda: plan_constant_el_scan(
            az_window=(100.0, 103.0),
            elevation=45.0,
            velocity=0.5,
            duration=60.0,
            site=site,
            start_time=_EVENING,
        ),
        "source-ces": lambda: plan_source_ces(
            body="jupiter",
            footprint="IM0",
            el_bore=35.0,
            night=_NIGHT,
            mode="rising",
            site=site,
        ),
    }


@pytest.mark.parametrize(
    "planner",
    [
        "pong",
        "pong rotation",
        "pong altaz",
        "daisy",
        "daisy altaz",
        "constant-el field",
        "constant-el window",
        "source-ces",
    ],
)
def test_a_block_reports_the_duration_it_built(site, planner):
    """A block's duration fields are the length of the trajectory it built.

    A pong's parameters give its period and cycle count instead of a duration.
    """
    block = _short_blocks(site)[planner]()

    assert block.duration == block.trajectory.duration
    if "duration" in block.computed_params:
        assert block.computed_params["duration"] == block.trajectory.duration
    if planner.startswith("daisy"):
        # 60 s requested at a 0.1 s step builds 59.9 s.
        assert f"Duration: {block.trajectory.duration:.1f}s" in block.summary


class TestPathSampleInterval:
    """The Go TCS sample-interval check reads the trajectory's own spacing."""

    @pytest.mark.parametrize(("timestep", "accepted"), [(0.0500001, False), (0.051, True)])
    def test_a_requested_step_within_rounding_of_the_limit_is_refused(
        self, site, timestep, accepted
    ):
        """A 120 s window scan snaps 0.0500001 s to 0.05 s, which falls a hair below the limit."""
        block = plan_constant_el_scan(
            az_window=(100.0, 110.0),
            elevation=45.0,
            velocity=0.5,
            duration=120.0,
            timestep=timestep,
            site=site,
            start_time=_EVENING,
        )

        if accepted:
            assert len(to_path_format(block.trajectory)) == block.trajectory.n_points
        else:
            with pytest.raises(ValueError, match="below the Go TCS /path minimum"):
                to_path_format(block.trajectory)


def _trapezoid_seconds(distance, max_velocity, max_acceleration):
    """Travel time of a rest-to-rest move under a speed and an acceleration limit."""
    if distance <= 0.0:
        return 0.0
    if distance >= max_velocity**2 / max_acceleration:
        return distance / max_velocity + max_velocity / max_acceleration
    return 2.0 * math.sqrt(distance / max_acceleration)


class TestDispatchHandOff:
    """A planned block followed through the wrap choice, the re-wrap and the payload."""

    @pytest.fixture(scope="class")
    def blocks(self):
        site = get_fyst_site()
        return {
            # Written below -180 deg, so the planner moves it up a turn.
            "moved-up window": plan_constant_el_scan(
                az_window=(-200.0, -58.15),
                elevation=45.0,
                velocity=1.0,
                duration=141.85,
                site=site,
                start_time=_EVENING,
            ),
            "pong": plan_pong_scan(_PONG_FIELD, **_PONG, site=site, start_time=_EVENING),
        }

    @pytest.mark.parametrize("block_name", ["moved-up window", "pong"])
    @pytest.mark.parametrize("current_az", [-170.0, 0.0, 170.0, 350.0])
    def test_the_payload_starts_where_the_slew_ends(self, blocks, block_name, current_az):
        """The first payload point is the chosen encoder azimuth, bit for bit.

        The wrap is chosen for the block's whole span, the trajectory is moved
        by the whole turn the choice names, and the slew there is priced as a
        rest-to-rest move on each axis.
        """
        site = get_fyst_site()
        no_sun = get_fyst_site(sun_avoidance_enabled=False)
        trajectory = blocks[block_name].trajectory
        span = (float(trajectory.az.min()), float(trajectory.az.max()))

        solution = choose_encoder_solution(
            current_az,
            45.0,
            float(trajectory.az[0]),
            float(trajectory.el[0]),
            trajectory.start_time,
            site=no_sun,
            goal_az_span=span,
        )
        enc_az, enc_el = solution
        moved = rewrap_trajectory_azimuth(trajectory, solution.az_shift)
        payload = to_path_payload(moved)

        assert solution.az_shift % 360.0 == 0.0
        assert enc_az == float(trajectory.az[0]) + solution.az_shift
        assert payload["points"][0][1] == enc_az
        limits = site.telescope_limits.azimuth
        assert limits.min <= moved.az.min() and moved.az.max() <= limits.max

        expected = max(
            _trapezoid_seconds(
                abs(enc_az - current_az), limits.max_velocity, limits.max_acceleration
            ),
            _trapezoid_seconds(
                abs(enc_el - 45.0),
                site.telescope_limits.elevation.max_velocity,
                site.telescope_limits.elevation.max_acceleration,
            ),
        )
        assert estimate_slew_time(current_az, 45.0, enc_az, enc_el, site=site) == pytest.approx(
            expected, abs=1e-9
        )


def _detector_track(trajectory, offset, site):
    """Where the detector at ``offset`` points along a boresight trajectory."""
    rotation = compute_focal_plane_rotation(trajectory.el, site=site, offset=offset)
    az, el = boresight_to_detector(trajectory.az, trajectory.el, offset, rotation)
    return np.asarray(az), np.asarray(el)


class TestFieldsUnderADetectorOffset:
    """With a detector offset, the window, elevation and centre fields are the detector's."""

    def test_a_constant_elevation_window_and_elevation_are_the_detectors(self, site):
        offset = get_primecam_offset("IM6")
        block = plan_constant_el_scan(
            _ECDFS,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=Time("2026-03-15T17:00:00", scale="utc"),
            detector_offset=offset,
        )
        trajectory = block.trajectory
        det_az, det_el = _detector_track(trajectory, offset, site)
        science = trajectory.scan_flag == SCAN_FLAG_SCIENCE

        az_start = block.computed_params["az_start"]
        az_stop = block.computed_params["az_stop"]
        # A science sample falls within one step of a window edge, never past it.
        step = 0.5 * block.config.timestep
        assert az_start - 1e-9 <= det_az[science].min() <= az_start + step
        assert az_stop - step <= det_az[science].max() <= az_stop + 1e-9
        assert det_el == pytest.approx(block.config.elevation, abs=1e-9)
        # The boresight's own window and elevation are well away from these.
        assert trajectory.az[science].max() < az_stop - 0.5
        assert trajectory.el.mean() - block.config.elevation > 1.0

    def test_an_altaz_pong_centre_is_the_detectors(self, site):
        """The detector traces the pong planned about the reported centre without an offset."""
        offset = get_primecam_offset("IM2")
        kwargs = dict(width=1.0, height=1.0, **_PONG, site=site, start_time=_EVENING)
        block = plan_pong_altaz_scan(120.0, 50.0, detector_offset=offset, **kwargs)
        bare = plan_pong_altaz_scan(
            block.computed_params["az_center"], block.computed_params["el_center"], **kwargs
        )
        det_az, det_el = _detector_track(block.trajectory, offset, site)

        assert det_az == pytest.approx(bare.trajectory.az, abs=1e-7)
        assert det_el == pytest.approx(bare.trajectory.el, abs=1e-7)
        # The boresight's own pattern sits a degree and more away.
        assert np.abs(block.trajectory.az - bare.trajectory.az).min() > 1.0


def test_a_source_pass_window_is_the_undrifted_one(site):
    """``az_start`` is the window at ``t0_iso``; the commanded path drifts from it by ``v_az``."""
    block = plan_source_ces(
        body="jupiter", footprint="IM0", el_bore=35.0, night=_NIGHT, mode="rising", site=site
    )
    cp = block.computed_params
    trajectory = block.trajectory
    science = trajectory.scan_flag == SCAN_FLAG_SCIENCE
    elapsed = trajectory.times - trajectory.times[0]
    undrifted = trajectory.az - cp["v_az"] * elapsed

    assert undrifted[science].min() == pytest.approx(cp["az_start"], abs=1e-9)
    assert undrifted[science].max() == pytest.approx(cp["az_start"] + cp["az_throw"], abs=1e-9)
    assert cp["v_az"] != 0.0
    assert abs(trajectory.az[science].min() - cp["az_start"]) <= abs(cp["v_az"]) * block.duration


class TestSourcePassAnchors:
    """A rising source pass is found from an anchor before the source enters the cover."""

    _ANCHOR = Time("2026-03-15T21:41:00", scale="utc")

    @pytest.fixture(scope="class")
    def planned(self):
        return plan_source_ces(
            body="jupiter", footprint="IM0", start_time=self._ANCHOR, site=get_fyst_site()
        )

    def _replan(self, planned, anchor):
        cp = planned.computed_params
        return plan_source_ces(
            body="jupiter",
            footprint="IM0",
            start_time=anchor,
            el_bore=cp["el_bore"],
            mode=cp["mode"],
            site=get_fyst_site(),
        )

    def test_ten_seconds_before_its_start_returns_the_pass(self, planned):
        t0 = Time(planned.computed_params["t0_iso"], scale="utc")

        again = self._replan(planned, t0 - TimeDelta(10.0, format="sec"))

        assert again.computed_params["t0_iso"] == planned.computed_params["t0_iso"]
        assert again.computed_params["n_scans"] == planned.computed_params["n_scans"]

    def test_its_own_start_is_refused(self, planned):
        """At ``t0_iso`` the source is a few milliseconds inside the cover already."""
        t0 = Time(planned.computed_params["t0_iso"], scale="utc")

        with pytest.raises(TargetNotObservableError):
            self._replan(planned, t0)


@pytest.mark.parametrize("module", ["IM1", "IM2", "IM6"])
@pytest.mark.parametrize("elevation", [25.0, 45.0, 70.0, 85.0])
def test_a_detector_offset_puts_the_target_at_the_offset(site, module, elevation):
    """Seen from the solved boresight, the target sits at the module's focal-plane offset.

    ``sky_to_focal_plane`` projects the target into the focal plane of the
    boresight independently of the solve, so it checks the frame, which a
    forward-then-inverse round trip cannot.
    """
    offset = get_primecam_offset(module)
    target = Trajectory(
        times=np.arange(5.0),
        az=np.full(5, 100.0),
        el=np.full(5, elevation),
        az_vel=np.zeros(5),
        el_vel=np.zeros(5),
        start_time=_EVENING,
    )

    boresight = apply_detector_offset(target, offset, site=site)
    rotation = compute_focal_plane_rotation(boresight.el, site=site, offset=offset)
    xi, eta = sky_to_focal_plane(boresight.az, boresight.el, target.az, target.el, rotation)

    error_arcsec = np.hypot(np.asarray(xi) - offset.dx_deg, np.asarray(eta) - offset.dy_deg)
    assert float(error_arcsec.max()) * 3600.0 < 1e-3


def test_a_pong_reaches_past_its_box_by_the_vertex_rounding():
    """A 2 x 2 deg pong at 0.1 deg spacing reaches 1.074 deg in height, past its 1.0 deg half."""
    config = PongScanConfig(
        timestep=0.1, width=2.0, height=2.0, spacing=0.1, velocity=0.4, num_terms=4, angle=0.0
    )
    period, _, _ = compute_pong_period(config)

    _, x, y = PongScanPattern(180.0, -30.0, config).generate_offsets(period)

    assert float(np.abs(y).max()) == pytest.approx(1.074, abs=5e-4)
    assert float(np.abs(x).max()) > 1.0
