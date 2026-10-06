"""Tests for plan_pong_altaz_scan."""

import math
import re
import warnings

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories import get_fyst_site
from fyst_trajectories.exceptions import PointingWarning, TrajectoryBoundsError
from fyst_trajectories.offsets import InstrumentOffset
from fyst_trajectories.patterns.configs import PongAltAzScanConfig
from fyst_trajectories.planning import (
    PongAltAzComputedParams,
    ScanBlock,
    plan_pong_altaz_scan,
)
from fyst_trajectories.trajectory_utils import get_absolute_times

# The parity test compares against the ``scanning`` package (scan_patterns).
# It is optional; gate on its availability with the same HAS_* + skipif
# precedent used elsewhere in the suite (e.g. test_offsets_projection.py).
try:
    from scanning import Pong as _ScanningPong  # noqa: F401

    HAS_SCANNING = True
except ImportError:
    HAS_SCANNING = False


@pytest.fixture
def start_time():
    """Provide a standard start time."""
    return Time("2026-03-15T04:00:00", scale="utc")


# A morning instant with the Sun low in the east (az 86.94, el 12.33 deg) and
# climbing about 15 deg per hour, and a night instant with it far below.
_DAY = Time("2026-03-15T11:30:00", scale="utc")
_NIGHT = Time("2026-03-15T04:00:00", scale="utc")

_BLOCK_MESSAGE = "EXCLUSION ZONE: planned AltAz Pong scan"
_SPEED_ADVISORY = (
    "ignore:High elevation reduces on-sky azimuth speed:"
    "fyst_trajectories.exceptions.PointingWarning"
)


def _reported_closest_approach(message):
    """Return the separation and time the block warning names."""
    match = re.search(r"passes ([0-9.]+) deg from the Sun at (\S+ \S+) \(", message)
    assert match, message
    return float(match.group(1)), Time(match.group(2), scale="utc")


def _per_sample_closest_approach(coordinates, trajectory):
    """Closest approach of every sample, with the Sun solved at every sample."""
    times = get_absolute_times(trajectory)
    sun_az, sun_el = coordinates.get_sun_altaz(times)
    seps = np.asarray(
        coordinates.angular_separation(trajectory.az, trajectory.el, sun_az, sun_el), dtype=float
    )
    closest = int(np.argmin(seps))
    return float(seps[closest]), times[closest], np.asarray(sun_el, dtype=float)


class TestPlanPongAltAzScan:
    """Block shape, the computed-params schema, the period, and the input guards."""

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_basic_plan(self, site, start_time):
        """Returns a ScanBlock with a pong_altaz config and trajectory."""
        block = plan_pong_altaz_scan(
            az_center=120.0,
            el_center=60.0,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            site=site,
            start_time=start_time,
        )

        assert isinstance(block, ScanBlock)
        assert isinstance(block.config, PongAltAzScanConfig)
        assert block.duration > 0
        assert block.trajectory.n_points > 0
        assert block.trajectory.pattern_type == "pong_altaz"
        assert "AltAz Pong scan" in block.summary

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_computed_params_schema_validates(self, site, start_time):
        block = plan_pong_altaz_scan(
            az_center=120.0,
            el_center=60.0,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            site=site,
            start_time=start_time,
        )

        params = block.computed_params
        # Exactly the PongAltAzComputedParams keys, no more, no less.
        assert set(params) == set(PongAltAzComputedParams.__required_keys__)
        assert params["az_center"] == 120.0
        assert params["el_center"] == 60.0
        assert params["n_cycles"] == 1

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_known_square_field_period(self, site, start_time):
        """A 2x2 deg, 0.1 deg-spacing pong has the hand-derived period.

        Same geometry as
        ``tests/patterns/test_pong.py::TestComputePongPeriod::test_known_square_field_period``:
        x_numvert=15, y_numvert=16, period = 4*15*16*0.1/0.5 = 192.0 s. The
        AltAz mapping does not change the period (it is an on-sky-geometry
        quantity).
        """
        block = plan_pong_altaz_scan(
            az_center=120.0,
            el_center=60.0,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            site=site,
            start_time=start_time,
        )
        assert block.computed_params["x_numvert"] == 15
        assert block.computed_params["y_numvert"] == 16
        assert block.computed_params["period"] == pytest.approx(192.0)
        assert block.duration == pytest.approx(192.0)

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_duration_scales_with_n_cycles(self, site, start_time):
        block1 = plan_pong_altaz_scan(
            az_center=120.0,
            el_center=60.0,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            site=site,
            start_time=start_time,
            n_cycles=1,
        )
        block3 = plan_pong_altaz_scan(
            az_center=120.0,
            el_center=60.0,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            site=site,
            start_time=start_time,
            n_cycles=3,
        )
        assert block3.duration == pytest.approx(3.0 * block1.duration)
        assert block3.computed_params["period"] == pytest.approx(block1.computed_params["period"])

    def test_invalid_n_cycles_raises(self, site, start_time):
        with pytest.raises(ValueError, match="n_cycles must be at least 1"):
            plan_pong_altaz_scan(
                az_center=120.0,
                el_center=60.0,
                width=2.0,
                height=2.0,
                spacing=0.1,
                velocity=0.5,
                site=site,
                start_time=start_time,
                n_cycles=0,
            )

    def test_el_center_above_range_raises_config_message(self, site, start_time):
        """el_center > 90 raises the config's message, not an astropy latitude error."""
        with pytest.raises(ValueError, match="el_center"):
            plan_pong_altaz_scan(
                az_center=120.0,
                el_center=95.0,
                width=2.0,
                height=2.0,
                spacing=0.1,
                velocity=0.5,
                site=site,
                start_time=start_time,
            )

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_accepts_iso_start_time(self, site):
        """A start_time string is accepted, like the other planners."""
        block = plan_pong_altaz_scan(
            az_center=120.0,
            el_center=60.0,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            site=site,
            start_time="2026-03-15T04:00:00",
        )
        assert block.trajectory.n_points > 0

    def test_bounds_error_for_high_el_center(self, site, start_time):
        """A high el_center drives the elevation extent past the 90 deg limit.

        With el_center=85 and a 12 deg on-sky height the upper elevation
        extent reaches ~90.8 deg (the realized Pong y-extent is a little
        under the nominal height, so 12 rather than exactly 10 is needed to
        clear 90), so bounds validation in the AltAz builder path must raise.
        """
        with (
            pytest.warns(PointingWarning, match="Azimuth-coordinate velocity"),
            pytest.raises(TrajectoryBoundsError),
        ):
            plan_pong_altaz_scan(
                az_center=120.0,
                el_center=85.0,
                width=10.0,
                height=12.0,
                spacing=0.2,
                velocity=0.5,
                site=site,
                start_time=start_time,
            )

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_detector_offset_changes_trajectory(self, site, start_time):
        block_no_offset = plan_pong_altaz_scan(
            az_center=120.0,
            el_center=60.0,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            site=site,
            start_time=start_time,
        )
        offset = InstrumentOffset(dx=5.0, dy=3.0, name="TestDet")
        block_with_offset = plan_pong_altaz_scan(
            az_center=120.0,
            el_center=60.0,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            site=site,
            start_time=start_time,
            detector_offset=offset,
        )
        assert not np.allclose(block_no_offset.trajectory.az, block_with_offset.trajectory.az)


class TestPlanPongAltAzSunSafety:
    """The centre pre-flight and the block screen against the Sun (warn-only)."""

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_sun_safety_is_warn_only(self, site, coordinates):
        """A sun-adjacent center warns and still returns a valid ScanBlock.

        The planner judges the horizon-frame center at the start time and
        then screens the built block, so aiming the center straight at the
        Sun's az/el trips both warnings. Pick a start time where the Sun is
        well above the horizon so the pattern still builds.
        """
        obstime = Time("2026-03-15T17:00:00", scale="utc")
        sun_az, sun_alt = coordinates.get_sun_altaz(obstime)
        assert 20.0 < sun_alt < 80.0, f"Sun elevation {sun_alt:.1f} out of the test band"

        with (
            pytest.warns(PointingWarning, match="EXCLUSION ZONE: Field center"),
            pytest.warns(PointingWarning, match=_BLOCK_MESSAGE),
        ):
            block = plan_pong_altaz_scan(
                az_center=float(sun_az),
                el_center=float(sun_alt),
                width=1.0,
                height=1.0,
                spacing=0.1,
                velocity=0.5,
                site=site,
                start_time=obstime,
            )
        assert block.trajectory.n_points > 0

    @pytest.mark.filterwarnings(
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_injected_predicate_drives_the_verdict(self, site):
        """An injected predicate replaces the scalar radius, in both directions.

        The seam is what lets the directional sun-avoidance model reach the
        AltAz planners; without a test on each planner a refactor could drop
        the keyword silently.
        """
        obstime = Time("2026-03-15T02:00:00", scale="utc")
        kwargs = dict(
            az_center=90.0,
            el_center=45.0,
            width=1.0,
            height=1.0,
            spacing=0.1,
            velocity=0.5,
            site=site,
            start_time=obstime,
        )

        # Precondition: the scalar default is silent for this night-time centre.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan_pong_altaz_scan(**kwargs)
        assert not [w for w in caught if "EXCLUSION ZONE" in str(w.message)]

        with (
            pytest.warns(PointingWarning, match="EXCLUSION ZONE: Field center"),
            pytest.warns(PointingWarning, match=_BLOCK_MESSAGE),
        ):
            plan_pong_altaz_scan(**kwargs, sun_safe=lambda az, el, t: False)

    @pytest.mark.filterwarnings(
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_injected_predicate_receives_the_converted_center(self, site):
        """The predicate is consulted at the pattern centre's own (az, el, time)."""
        obstime = Time("2026-03-15T02:00:00", scale="utc")
        seen: list[tuple[float, float]] = []

        def spy(az, el, t):
            seen.append((float(az), float(el)))
            return True

        plan_pong_altaz_scan(
            az_center=90.0,
            el_center=45.0,
            width=1.0,
            height=1.0,
            spacing=0.1,
            velocity=0.5,
            site=site,
            start_time=obstime,
            sun_safe=spy,
        )
        assert seen, "sun_safe predicate was never consulted"
        # The centre pre-flight is the first consultation; the block screen's
        # probes of the built trajectory follow it.
        assert seen[0][0] == pytest.approx(90.0, abs=1e-6)
        assert seen[0][1] == pytest.approx(45.0, abs=1e-6)

    def test_permissive_predicate_overrides_the_scalar_radius(self, site, coordinates):
        """A permissive predicate suppresses the warning a sun-adjacent centre earns."""
        obstime = Time("2026-03-15T17:00:00", scale="utc")
        sun_az, sun_alt = coordinates.get_sun_altaz(obstime)
        assert 20.0 < sun_alt < 80.0, f"Sun elevation {sun_alt:.1f} out of the test band"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan_pong_altaz_scan(
                az_center=float(sun_az),
                el_center=float(sun_alt),
                width=1.0,
                height=1.0,
                spacing=0.1,
                velocity=0.5,
                site=site,
                start_time=obstime,
                sun_safe=lambda az, el, t: True,
            )
        assert not [w for w in caught if "EXCLUSION ZONE" in str(w.message)]

    @pytest.mark.filterwarnings(_SPEED_ADVISORY)
    def test_block_screen_catches_the_sun_closing_in(self, site, coordinates):
        """A centre clear at the start warns once the Sun closes on the fixed block.

        The Sun climbs about 15 deg per hour towards a horizon-frame centre
        48 deg above it, so the hour-long block ends well inside the 45 deg
        zone while the start-time centre check stays silent. The warning is
        attributed to the planner's caller.
        """
        sun_az, sun_el = (float(v) for v in coordinates.get_sun_altaz(_DAY))
        with pytest.warns(PointingWarning, match=_BLOCK_MESSAGE) as record:
            block = plan_pong_altaz_scan(
                az_center=sun_az,
                el_center=sun_el + 48.0,
                width=1.0,
                height=1.0,
                spacing=0.1,
                velocity=0.2,
                site=site,
                start_time=_DAY,
                n_cycles=25,
            )
        assert block.duration == pytest.approx(3600.0)
        assert not [w for w in record if "Field center" in str(w.message)]
        (warning,) = [w for w in record if _BLOCK_MESSAGE in str(w.message)]
        assert warning.filename == __file__
        separation, _ = _reported_closest_approach(str(warning.message))
        assert separation < 45.0

    @pytest.mark.filterwarnings(_SPEED_ADVISORY)
    def test_block_screen_catches_the_pattern_extent(self, site, coordinates):
        """A centre outside the zone warns when the pattern's edge reaches inside."""
        sun_az, sun_el = (float(v) for v in coordinates.get_sun_altaz(_DAY))
        with pytest.warns(PointingWarning, match=_BLOCK_MESSAGE) as record:
            plan_pong_altaz_scan(
                az_center=sun_az,
                el_center=sun_el + 46.5,
                width=4.0,
                height=4.0,
                spacing=0.2,
                velocity=0.5,
                site=site,
                start_time=_DAY,
            )
        assert not [w for w in record if "Field center" in str(w.message)]
        (warning,) = [w for w in record if _BLOCK_MESSAGE in str(w.message)]
        assert warning.filename == __file__

    def test_night_block_and_disabled_avoidance_are_silent(self, site, coordinates):
        """The screen is silent at night and when Sun avoidance is disabled."""
        sun_az, sun_el = (float(v) for v in coordinates.get_sun_altaz(_DAY))
        kwargs = dict(
            az_center=sun_az,
            el_center=sun_el + 46.5,
            width=4.0,
            height=4.0,
            spacing=0.2,
            velocity=0.5,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan_pong_altaz_scan(**kwargs, site=site, start_time=_NIGHT)
            plan_pong_altaz_scan(
                **kwargs, site=get_fyst_site(sun_avoidance_enabled=False), start_time=_DAY
            )
        assert not [w for w in caught if "EXCLUSION ZONE" in str(w.message)]

    @pytest.mark.filterwarnings(_SPEED_ADVISORY)
    def test_block_warning_names_the_exact_closest_approach(self, site, coordinates):
        """The built-in screen reports the closest sample, not a sampled estimate."""
        sun_az, sun_el = (float(v) for v in coordinates.get_sun_altaz(_DAY))
        with pytest.warns(PointingWarning, match=_BLOCK_MESSAGE) as record:
            block = plan_pong_altaz_scan(
                az_center=sun_az,
                el_center=sun_el + 46.5,
                width=4.0,
                height=4.0,
                spacing=0.2,
                velocity=0.5,
                site=site,
                start_time=_DAY,
            )
        (warning,) = [w for w in record if _BLOCK_MESSAGE in str(w.message)]
        separation, when = _reported_closest_approach(str(warning.message))
        true_sep, true_when, _ = _per_sample_closest_approach(coordinates, block.trajectory)
        # The message rounds to 0.1 deg.
        assert separation == pytest.approx(true_sep, abs=0.05)
        assert abs((when - true_when).sec) < 0.5

    def test_block_screen_through_a_zenith_transit(self, site, coordinates):
        """The interpolated Sun holds while it transits within 0.5 deg of the zenith.

        At the December solstice the Sun crosses the meridian about 0.45 deg
        south of FYST's zenith near 16:29 UTC, sweeping through about 135 deg
        of azimuth in ten minutes; the block's reported closest approach still
        matches the per-sample ephemeris.
        """
        start = Time("2026-12-21T16:25:00", scale="utc")
        with pytest.warns(PointingWarning, match=_BLOCK_MESSAGE) as record:
            block = plan_pong_altaz_scan(
                az_center=180.0,
                el_center=44.0,
                width=2.0,
                height=2.0,
                spacing=0.2,
                velocity=0.5,
                site=site,
                start_time=start,
                timestep=0.5,
                n_cycles=3,
            )
        true_sep, true_when, sun_el = _per_sample_closest_approach(coordinates, block.trajectory)
        assert sun_el.max() > 89.4
        (warning,) = [w for w in record if _BLOCK_MESSAGE in str(w.message)]
        separation, when = _reported_closest_approach(str(warning.message))
        assert separation == pytest.approx(true_sep, abs=0.05)
        assert abs((when - true_when).sec) < 1.0

    @pytest.mark.filterwarnings(_SPEED_ADVISORY)
    @pytest.mark.parametrize("with_batch", [False, True], ids=["predicate", "batch"])
    def test_injected_model_sees_at_most_602_times(self, site, coordinates, with_batch):
        """An injected model is asked about the centre and at most 601 probes.

        The probes stride the built trajectory with both ends included, so the
        last sample's time is among them, whether the model answers per call
        or through ``batch``.
        """
        seen_jd: list[float] = []

        class Spy:
            def __call__(self, az, el, t):
                seen_jd.append(float(t.jd))
                return True

        class BatchSpy(Spy):
            def batch(self, az, el, times):
                seen_jd.extend(np.atleast_1d(times.jd).tolist())
                return np.ones(np.size(az), dtype=bool)

        sun_az, sun_el = (float(v) for v in coordinates.get_sun_altaz(_DAY))
        block = plan_pong_altaz_scan(
            az_center=sun_az,
            el_center=sun_el + 46.5,
            width=4.0,
            height=4.0,
            spacing=0.2,
            velocity=0.5,
            site=site,
            start_time=_NIGHT,
            sun_safe=BatchSpy() if with_batch else Spy(),
        )
        assert block.trajectory.n_points > 602
        assert len(seen_jd) <= 602
        last_jd = float(get_absolute_times(block.trajectory)[-1].jd)
        assert np.any(np.isclose(seen_jd, last_jd, rtol=0.0, atol=1e-8))


@pytest.mark.slow
@pytest.mark.skipif(not HAS_SCANNING, reason="requires the scanning (scan_patterns) package")
class TestLegacyMappingParity:
    """End-to-end parity with the legacy scanning.Pong + horizon mapping.

    scanning.Pong delegates its offset generation to this library's
    PongScanPattern, so this validates that plan_pong_altaz_scan reproduces
    the legacy horizon-frame mapping (x/cos(el0)+az0, y+el0) and the
    surrounding plumbing end to end, not the offset math itself.
    """

    @pytest.mark.filterwarnings(
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_trajectory_matches_legacy_mapping(self, site):
        from scanning import Pong

        az_center = 130.0
        el_center = 55.0
        width = 2.0
        height = 2.0
        spacing = 0.1
        velocity = 0.5
        num_terms = 4
        angle = 0.0
        timestep = 0.1

        block = plan_pong_altaz_scan(
            az_center=az_center,
            el_center=el_center,
            width=width,
            height=height,
            spacing=spacing,
            velocity=velocity,
            site=site,
            start_time=Time("2026-03-15T04:00:00", scale="utc"),
            num_terms=num_terms,
            angle=angle,
            timestep=timestep,
        )
        traj = block.trajectory

        # Build scanning.Pong with matching parameters and apply the
        # horizon-frame mapping inline:
        #   coscorr = cos(radians(el_center))
        #   az = x/coscorr + az_center ; el = y + el_center
        pong = Pong(
            num_term=num_terms,
            width=width,
            height=height,
            spacing=spacing,
            velocity=velocity,
            angle=angle,
            sample_interval=timestep,
            max_scan_duration=block.duration,
        )
        x_off = pong.x_coord.value
        y_off = pong.y_coord.value

        coscorr = math.cos(math.radians(el_center))
        legacy_az = x_off / coscorr + az_center
        legacy_el = y_off + el_center

        # Guard: the two paths must sample the same number of points.
        assert len(legacy_az) == traj.n_points

        np.testing.assert_allclose(traj.az, legacy_az, atol=1e-6)
        np.testing.assert_allclose(traj.el, legacy_el, atol=1e-6)
