"""Tests for plan_daisy_altaz_scan."""

import math
import re
import warnings

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories import get_fyst_site
from fyst_trajectories.exceptions import PointingWarning, TrajectoryBoundsError
from fyst_trajectories.offsets import InstrumentOffset
from fyst_trajectories.patterns.configs import DaisyAltAzScanConfig
from fyst_trajectories.planning import (
    DaisyAltAzComputedParams,
    ScanBlock,
    plan_daisy_altaz_scan,
)
from fyst_trajectories.trajectory_utils import get_absolute_times

# The parity test compares against the ``scanning`` package (scan_patterns).
# It is optional; gate on its availability with the same HAS_* + skipif
# precedent used elsewhere in the suite (e.g. test_planning_pong_altaz.py).
try:
    from scanning import Daisy as _ScanningDaisy  # noqa: F401

    HAS_SCANNING = True
except ImportError:
    HAS_SCANNING = False


@pytest.fixture
def start_time():
    """Provide a standard start time."""
    return Time("2026-03-15T04:00:00", scale="utc")


def _plan(site, start_time, **overrides):
    """Call plan_daisy_altaz_scan with sensible test defaults."""
    params = dict(
        az_center=120.0,
        el_center=60.0,
        radius=0.5,
        velocity=0.3,
        turn_radius=0.2,
        avoidance_radius=0.0,
        start_acceleration=0.5,
        site=site,
        start_time=start_time,
        timestep=0.1,
        duration=100.0,
    )
    params.update(overrides)
    return plan_daisy_altaz_scan(**params)


# A morning instant with the Sun low in the east (az 86.94, el 12.33 deg) and
# climbing about 15 deg per hour, and a night instant with it far below.
_DAY = Time("2026-03-15T11:30:00", scale="utc")
_NIGHT = Time("2026-03-15T04:00:00", scale="utc")

_BLOCK_MESSAGE = "EXCLUSION ZONE: planned AltAz Daisy scan"
_SPEED_ADVISORY = (
    "ignore:High elevation reduces on-sky azimuth speed:"
    "fyst_trajectories.exceptions.PointingWarning"
)


def _extent_overrides(coordinates):
    """Return a Daisy whose centre is 46.5 deg from the Sun at ``_DAY``, petals reaching in."""
    sun_az, sun_el = (float(v) for v in coordinates.get_sun_altaz(_DAY))
    return dict(
        az_center=sun_az,
        el_center=sun_el + 46.5,
        radius=2.0,
        velocity=0.5,
        turn_radius=0.5,
        duration=120.0,
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


class TestPlanDaisyAltAzScan:
    """Block shape, the computed-params schema, and the el_center / bounds guards."""

    @pytest.mark.filterwarnings(_SPEED_ADVISORY)
    def test_basic_plan(self, site, start_time):
        """Returns a ScanBlock with a daisy_altaz config and trajectory."""
        block = _plan(site, start_time)

        assert isinstance(block, ScanBlock)
        assert isinstance(block.config, DaisyAltAzScanConfig)
        assert block.duration > 0
        assert block.trajectory.n_points > 0
        assert block.trajectory.pattern_type == "daisy_altaz"
        assert "AltAz Daisy scan" in block.summary

    @pytest.mark.filterwarnings(_SPEED_ADVISORY)
    def test_computed_params_schema_validates(self, site, start_time):
        block = _plan(site, start_time)

        params = block.computed_params
        # Exactly the DaisyAltAzComputedParams keys, no more, no less.
        assert set(params) == set(DaisyAltAzComputedParams.__required_keys__)
        assert params["az_center"] == 120.0
        assert params["el_center"] == 60.0
        assert params["duration"] == pytest.approx(100.0)

    @pytest.mark.filterwarnings(_SPEED_ADVISORY)
    def test_duration_honored(self, site, start_time):
        block = _plan(site, start_time, duration=250.0)
        assert block.duration == pytest.approx(250.0)
        assert block.computed_params["duration"] == pytest.approx(250.0)

    @pytest.mark.filterwarnings(_SPEED_ADVISORY)
    def test_accepts_iso_start_time(self, site):
        """A start_time string is accepted, like the other planners."""
        block = _plan(site, "2026-03-15T04:00:00")
        assert block.trajectory.n_points > 0

    def test_el_center_above_range_raises_config_message(self, site, start_time):
        """el_center > 90 raises the config's message, not an astropy latitude error."""
        with pytest.raises(ValueError, match="el_center"):
            _plan(site, start_time, el_center=95.0)

    def test_bounds_error_when_elevation_exceeds_limit(self, site, start_time):
        """A high el_center plus a large radius drives el past the 90 deg limit.

        With el_center=87 and radius=6 the realized Daisy y-extent reaches
        ~5.7 deg, so the upper elevation extent hits ~92.7 deg and bounds
        validation in the AltAz builder path must raise.
        """
        with (
            pytest.warns(PointingWarning, match="Azimuth-coordinate velocity"),
            pytest.raises(TrajectoryBoundsError),
        ):
            _plan(
                site,
                start_time,
                el_center=87.0,
                radius=6.0,
                turn_radius=0.5,
                duration=400.0,
            )

    @pytest.mark.filterwarnings(_SPEED_ADVISORY)
    def test_detector_offset_changes_trajectory(self, site, start_time):
        block_no_offset = _plan(site, start_time)
        offset = InstrumentOffset(dx=5.0, dy=3.0, name="TestDet")
        block_with_offset = _plan(site, start_time, detector_offset=offset)
        assert not np.allclose(block_no_offset.trajectory.az, block_with_offset.trajectory.az)

    @pytest.mark.filterwarnings(_SPEED_ADVISORY)
    def test_default_timestep_equals_explicit(self, site, start_time):
        """Omitting ``timestep`` plans exactly what ``timestep=0.1`` plans."""
        common = dict(
            az_center=120.0,
            el_center=60.0,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            site=site,
            start_time=start_time,
            duration=100.0,
        )
        defaulted = plan_daisy_altaz_scan(**common)
        explicit = plan_daisy_altaz_scan(**common, timestep=0.1)

        assert defaulted.config == explicit.config
        for name in ("times", "az", "el", "az_vel", "el_vel"):
            np.testing.assert_array_equal(
                getattr(defaulted.trajectory, name), getattr(explicit.trajectory, name)
            )


class TestPlanDaisyAltAzSunSafety:
    """The centre pre-flight and the block screen against the Sun (warn-only)."""

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
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
            block = _plan(
                site,
                obstime,
                az_center=float(sun_az),
                el_center=float(sun_alt),
            )
        assert block.trajectory.n_points > 0

    @pytest.mark.filterwarnings(_SPEED_ADVISORY)
    def test_injected_predicate_drives_the_verdict(self, site):
        """An injected predicate replaces the scalar radius, in both directions.

        The seam is what lets the directional sun-avoidance model reach the
        AltAz planners; without a test on each planner a refactor could drop
        the keyword silently.
        """
        obstime = Time("2026-03-15T02:00:00", scale="utc")

        # Precondition: the scalar default is silent for this night-time centre.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _plan(site, obstime)
        assert not [w for w in caught if "EXCLUSION ZONE" in str(w.message)]

        with (
            pytest.warns(PointingWarning, match="EXCLUSION ZONE: Field center"),
            pytest.warns(PointingWarning, match=_BLOCK_MESSAGE),
        ):
            _plan(site, obstime, sun_safe=lambda az, el, t: False)

    @pytest.mark.filterwarnings(_SPEED_ADVISORY)
    def test_injected_predicate_receives_the_converted_center(self, site):
        """The predicate is consulted at the pattern centre's own (az, el, time)."""
        obstime = Time("2026-03-15T02:00:00", scale="utc")
        seen: list[tuple[float, float]] = []

        def spy(az, el, t):
            seen.append((float(az), float(el)))
            return True

        _plan(site, obstime, sun_safe=spy)
        assert seen, "sun_safe predicate was never consulted"
        # The centre pre-flight is the first consultation; the block screen's
        # probes of the built trajectory follow it.
        assert seen[0][0] == pytest.approx(120.0, abs=1e-6)
        assert seen[0][1] == pytest.approx(60.0, abs=1e-6)

    def test_permissive_predicate_overrides_the_scalar_radius(self, site, coordinates):
        """A permissive predicate suppresses the warning a sun-adjacent centre earns."""
        obstime = Time("2026-03-15T17:00:00", scale="utc")
        sun_az, sun_alt = coordinates.get_sun_altaz(obstime)
        assert 20.0 < sun_alt < 80.0, f"Sun elevation {sun_alt:.1f} out of the test band"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _plan(
                site,
                obstime,
                az_center=float(sun_az),
                el_center=float(sun_alt),
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
            _plan(
                site,
                _DAY,
                az_center=sun_az,
                el_center=sun_el + 48.0,
                duration=3600.0,
            )
        assert not [w for w in record if "Field center" in str(w.message)]
        (warning,) = [w for w in record if _BLOCK_MESSAGE in str(w.message)]
        assert warning.filename == __file__
        separation, _ = _reported_closest_approach(str(warning.message))
        assert separation < 45.0

    @pytest.mark.filterwarnings(_SPEED_ADVISORY)
    def test_block_screen_catches_the_pattern_extent(self, site, coordinates):
        """A centre outside the zone warns when the petals reach inside."""
        with pytest.warns(PointingWarning, match=_BLOCK_MESSAGE) as record:
            _plan(site, _DAY, **_extent_overrides(coordinates))
        assert not [w for w in record if "Field center" in str(w.message)]
        (warning,) = [w for w in record if _BLOCK_MESSAGE in str(w.message)]
        assert warning.filename == __file__

    def test_night_block_and_disabled_avoidance_are_silent(self, site, coordinates):
        """The screen is silent at night and when Sun avoidance is disabled."""
        overrides = _extent_overrides(coordinates)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _plan(site, _NIGHT, **overrides)
            _plan(get_fyst_site(sun_avoidance_enabled=False), _DAY, **overrides)
        assert not [w for w in caught if "EXCLUSION ZONE" in str(w.message)]

    @pytest.mark.filterwarnings(_SPEED_ADVISORY)
    def test_block_warning_names_the_exact_closest_approach(self, site, coordinates):
        """The built-in screen reports the closest sample, not a sampled estimate."""
        with pytest.warns(PointingWarning, match=_BLOCK_MESSAGE) as record:
            block = _plan(site, _DAY, **_extent_overrides(coordinates))
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
            block = _plan(
                site,
                start,
                az_center=180.0,
                el_center=44.0,
                radius=1.0,
                velocity=0.5,
                turn_radius=0.3,
                duration=600.0,
                timestep=0.5,
            )
        true_sep, true_when, sun_el = _per_sample_closest_approach(coordinates, block.trajectory)
        assert sun_el.max() > 89.4
        (warning,) = [w for w in record if _BLOCK_MESSAGE in str(w.message)]
        separation, when = _reported_closest_approach(str(warning.message))
        assert separation == pytest.approx(true_sep, abs=0.05)
        assert abs((when - true_when).sec) < 1.0

    @pytest.mark.filterwarnings(_SPEED_ADVISORY)
    @pytest.mark.parametrize("with_batch", [False, True], ids=["predicate", "batch"])
    def test_injected_model_sees_at_most_602_times(self, site, with_batch):
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

        block = _plan(
            site,
            _NIGHT,
            duration=300.0,
            sun_safe=BatchSpy() if with_batch else Spy(),
        )
        assert block.trajectory.n_points > 602
        assert len(seen_jd) <= 602
        last_jd = float(get_absolute_times(block.trajectory)[-1].jd)
        assert np.any(np.isclose(seen_jd, last_jd, rtol=0.0, atol=1e-8))


@pytest.mark.slow
@pytest.mark.skipif(not HAS_SCANNING, reason="requires the scanning (scan_patterns) package")
class TestLegacyMappingParity:
    """End-to-end parity with the legacy scanning.Daisy + horizon mapping.

    scanning.Daisy delegates its offset generation to this library's
    DaisyScanPattern, so this validates that plan_daisy_altaz_scan reproduces
    the legacy horizon-frame mapping (x/cos(el0)+az0, y+el0) and the
    surrounding plumbing end to end, not the offset math itself.
    """

    def test_trajectory_matches_legacy_mapping(self, site):
        from scanning import Daisy

        az_center = 130.0
        el_center = 55.0
        radius = 0.5
        velocity = 0.3
        turn_radius = 0.2
        avoidance_radius = 0.0
        start_acceleration = 0.5
        y_offset = 0.0
        timestep = 0.1
        duration = 100.0

        block = plan_daisy_altaz_scan(
            az_center=az_center,
            el_center=el_center,
            radius=radius,
            velocity=velocity,
            turn_radius=turn_radius,
            avoidance_radius=avoidance_radius,
            start_acceleration=start_acceleration,
            site=site,
            start_time=Time("2026-03-15T04:00:00", scale="utc"),
            timestep=timestep,
            duration=duration,
            y_offset=y_offset,
        )
        traj = block.trajectory

        # Build scanning.Daisy with matching parameters and apply the
        # horizon-frame mapping inline:
        #   coscorr = cos(radians(el_center))
        #   az = x/coscorr + az_center ; el = y + el_center
        daisy = Daisy(
            velocity=velocity,
            start_acc=start_acceleration,
            R0=radius,
            Rt=turn_radius,
            Ra=avoidance_radius,
            T=duration,
            sample_interval=timestep,
            y_offset=y_offset,
        )
        x_off = daisy.x_coord.value
        y_off = daisy.y_coord.value

        coscorr = math.cos(math.radians(el_center))
        legacy_az = x_off / coscorr + az_center
        legacy_el = y_off + el_center

        # Guard: the two paths must sample the same number of points.
        assert len(legacy_az) == traj.n_points

        # AltAz azimuth is used as provided (no normalization), so a direct
        # comparison is expected.
        np.testing.assert_allclose(traj.az, legacy_az, atol=1e-6)
        np.testing.assert_allclose(traj.el, legacy_el, atol=1e-6)
