"""Tests for sun avoidance integration.

Tests cover:
- validate_sun_avoidance() for safe, excluded, and warning-zone trajectories
  (exclusion zone emits PointingWarning, never blocks trajectory generation)
- Sun avoidance disabled via config
- Sun avoidance skipped when trajectory.start_time is None
- Planning pre-flight check (_check_field_sun_safety)
- Subsampling behaviour (many points, few sun computations)
"""

import warnings
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time, TimeDelta

from fyst_trajectories import get_fyst_site
from fyst_trajectories.coordinates import Coordinates
from fyst_trajectories.exceptions import (
    PointingWarning,
)
from fyst_trajectories.planning import (
    FieldRegion,
    plan_constant_el_scan,
    plan_daisy_scan,
    plan_pong_scan,
)
from fyst_trajectories.planning._sun_safety import _check_field_sun_safety
from fyst_trajectories.site import SunAvoidanceConfig
from fyst_trajectories.trajectory import Trajectory
from fyst_trajectories.trajectory_utils import validate_sun_avoidance, validate_trajectory

# Pre-flight check: verify the sun is high enough at the test obstime for
# meaningful sun-avoidance tests.  This avoids silent skips inside tests
# that could mask regressions.
_TEST_OBSTIME = Time("2026-06-15T16:00:00", scale="utc")
_site_for_check = get_fyst_site()
_coords_for_check = Coordinates(_site_for_check)
_sun_az_check, _sun_alt_check = _coords_for_check.get_sun_altaz(_TEST_OBSTIME)
assert _sun_alt_check >= 20.0, (
    f"Sun altitude at test obstime {_TEST_OBSTIME.iso} is {_sun_alt_check:.1f} deg, "
    f"which is below 20 deg. Choose a different test time when the sun is higher."
)
del _site_for_check, _coords_for_check, _sun_az_check, _sun_alt_check


@pytest.fixture
def site():
    """Provide a default FYST site for testing."""
    return get_fyst_site()


@pytest.fixture
def coords(site):
    """Provide a Coordinates instance for testing."""
    return Coordinates(site)


@pytest.fixture
def obstime():
    """Provide a fixed observation time for reproducible tests."""
    return _TEST_OBSTIME


@pytest.fixture
def sun_ra_dec(coords, obstime):
    """Compute the ICRS RA/Dec that maps to the sun's AltAz position.

    We use altaz_to_radec on the sun's AltAz rather than
    get_body_radec("sun") because the sun's ICRS coordinates differ
    from its apparent position due to annual aberration and parallax.
    The planning pre-flight check works in AltAz, so we need RA/Dec
    that will map to the sun's actual AltAz location.
    """
    sun_az, sun_alt = coords.get_sun_altaz(obstime)
    return coords.altaz_to_radec(sun_az, sun_alt, obstime)


@pytest.fixture
def sun_az_el(coords, obstime):
    """Compute the sun's Az/El at the test obstime."""
    return coords.get_sun_altaz(obstime)


# ---------------------------------------------------------------------------
# validate_sun_avoidance
# ---------------------------------------------------------------------------


class TestValidateSunAvoidance:
    """Test validate_sun_avoidance() function."""

    def test_safe_trajectory_passes(self, site, coords, obstime):
        """Trajectory far from the sun passes without error or warning."""
        sun_az, _sun_alt = coords.get_sun_altaz(obstime)
        safe_az = (sun_az + 180.0) % 360.0
        safe_el = 45.0

        n = 100
        abs_times = obstime + TimeDelta(np.arange(n) * u.s)

        az = np.full(n, safe_az)
        el = np.full(n, safe_el)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_sun_avoidance(site, az, el, abs_times, coords=coords)
            sun_warnings = [x for x in w if issubclass(x.category, PointingWarning)]
            assert len(sun_warnings) == 0

    def test_trajectory_at_sun_warns(self, site, coords, obstime):
        """Trajectory pointing at the sun emits EXCLUSION ZONE warning."""
        sun_az, sun_alt = coords.get_sun_altaz(obstime)

        n = 50
        abs_times = obstime + TimeDelta(np.arange(n) * u.s)

        az = np.full(n, sun_az)
        el = np.full(n, sun_alt)

        with pytest.warns(PointingWarning, match="EXCLUSION ZONE"):
            validate_sun_avoidance(site, az, el, abs_times, coords=coords)

    def test_warning_zone_emits_warning(self, site, coords, obstime):
        """Trajectory in the warning zone emits PointingWarning."""
        sun_az, sun_alt = coords.get_sun_altaz(obstime)

        # Wide gap between exclusion and warning so we can land in between
        custom_sun = SunAvoidanceConfig(
            enabled=True,
            exclusion_radius=10.0,
            warning_radius=60.0,
        )
        custom_site = replace(site, sun_avoidance=custom_sun)

        offset_az = sun_az + 30.0
        offset_el = sun_alt

        n = 20
        abs_times = obstime + TimeDelta(np.arange(n) * u.s)

        az = np.full(n, offset_az)
        el = np.full(n, offset_el)

        sep = coords.angular_separation(offset_az, offset_el, sun_az, sun_alt)

        if sep < custom_sun.exclusion_radius or sep > custom_sun.warning_radius:
            pytest.skip(
                f"Separation {sep:.1f} not in warning zone "
                f"[{custom_sun.exclusion_radius}, {custom_sun.warning_radius}]"
            )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_sun_avoidance(custom_site, az, el, abs_times, coords=coords)
            sun_warnings = [x for x in w if issubclass(x.category, PointingWarning)]
            assert len(sun_warnings) >= 1
            assert "Sun" in str(sun_warnings[0].message)

    def test_disabled_skips_check(self, site, coords, obstime):
        """Sun avoidance check is skipped when disabled."""
        sun_az, sun_alt = coords.get_sun_altaz(obstime)

        disabled_sun = SunAvoidanceConfig(
            enabled=False,
            exclusion_radius=45.0,
            warning_radius=50.0,
        )
        disabled_site = replace(site, sun_avoidance=disabled_sun)

        n = 10
        abs_times = obstime + TimeDelta(np.arange(n) * u.s)

        az = np.full(n, sun_az)
        el = np.full(n, sun_alt)

        # Should NOT raise even though pointing at the sun
        validate_sun_avoidance(disabled_site, az, el, abs_times, coords=coords)

    def test_subsampling_behavior(self, site, coords, obstime):
        """Trajectory with many points uses sparse sun position computation."""
        sun_az, _sun_alt = coords.get_sun_altaz(obstime)
        safe_az = (sun_az + 180.0) % 360.0

        # 100k points over 10 seconds, should subsample heavily
        n = 100_000
        abs_times = obstime + TimeDelta(np.linspace(0, 10, n) * u.s)

        az = np.full(n, safe_az)
        el = np.full(n, 45.0)

        call_count = [0]
        original_get_sun = coords.get_sun_altaz

        def counting_get_sun(t):
            call_count[0] += 1
            return original_get_sun(t)

        with patch.object(coords, "get_sun_altaz", side_effect=counting_get_sun):
            validate_sun_avoidance(site, az, el, abs_times, coords=coords)

        # Over 10 seconds with 60s interval, should call get_sun_altaz once
        # (or a handful of times). Certainly not 100,000.
        assert call_count[0] < 100
        assert call_count[0] >= 1


# ---------------------------------------------------------------------------
# validate_trajectory integration
# ---------------------------------------------------------------------------


class TestValidateTrajectoryWithSun:
    """Test that validate_trajectory() includes sun check."""

    def test_skips_sun_when_no_start_time(self, site):
        """Sun check is skipped when trajectory.start_time is None."""
        traj = Trajectory(
            times=np.array([0.0, 1.0, 2.0]),
            az=np.array([100.0, 100.0, 100.0]),
            el=np.full(3, 45.0),
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
            start_time=None,
        )
        validate_trajectory(traj, site)

    def test_skips_sun_when_check_sun_false(self, site, coords, obstime):
        """Sun check is skipped when check_sun=False."""
        sun_az, sun_alt = coords.get_sun_altaz(obstime)

        limits = site.telescope_limits
        az_val = np.clip(sun_az, limits.azimuth.min, limits.azimuth.max)
        el_val = np.clip(max(sun_alt, 25.0), limits.elevation.min, limits.elevation.max)
        traj = Trajectory(
            times=np.array([0.0, 1.0, 2.0]),
            az=np.full(3, az_val),
            el=np.full(3, el_val),
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
            start_time=obstime,
        )

        validate_trajectory(traj, site, check_sun=False)


# ---------------------------------------------------------------------------
# Planning pre-flight check
# ---------------------------------------------------------------------------


class TestCheckFieldSunSafety:
    """Test _check_field_sun_safety() pre-flight check."""

    def test_safe_field_passes(self, site, coords, obstime, sun_ra_dec):
        """Field far from the sun passes the pre-flight check."""
        sun_ra, _sun_dec = sun_ra_dec
        safe_ra = (sun_ra + 90.0) % 360.0
        safe_dec = -30.0

        _check_field_sun_safety(safe_ra, safe_dec, obstime, site)

    def test_field_at_sun_warns(self, site, coords, obstime, sun_ra_dec):
        """Field at the sun's position emits EXCLUSION ZONE warning."""
        sun_ra, sun_dec = sun_ra_dec

        with pytest.warns(PointingWarning, match="EXCLUSION ZONE"):
            _check_field_sun_safety(sun_ra, sun_dec, obstime, site)

    def test_disabled_skips(self, site, obstime, sun_ra_dec):
        """Pre-flight check is skipped when sun avoidance is disabled."""
        sun_ra, sun_dec = sun_ra_dec
        disabled_sun = SunAvoidanceConfig(
            enabled=False,
            exclusion_radius=45.0,
            warning_radius=50.0,
        )
        disabled_site = replace(site, sun_avoidance=disabled_sun)

        _check_field_sun_safety(sun_ra, sun_dec, obstime, disabled_site)


# ---------------------------------------------------------------------------
# Planning functions integration
# ---------------------------------------------------------------------------


class TestPlanningIntegration:
    """Test that planning functions invoke the sun pre-flight check."""

    def test_plan_pong_scan_warns_sun_field(self, site, coords, obstime, sun_ra_dec):
        """plan_pong_scan emits EXCLUSION ZONE warning for a field at the sun."""
        sun_ra, sun_dec = sun_ra_dec
        field = FieldRegion(
            ra_center=sun_ra,
            dec_center=sun_dec,
            width=2.0,
            height=2.0,
        )

        with pytest.warns(PointingWarning, match="EXCLUSION ZONE"):
            plan_pong_scan(
                field=field,
                velocity=0.5,
                spacing=0.1,
                num_terms=4,
                site=site,
                start_time=obstime,
                timestep=0.1,
            )

    def test_plan_daisy_scan_warns_sun_field(self, site, coords, obstime, sun_ra_dec):
        """plan_daisy_scan emits EXCLUSION ZONE warning for a source at the sun."""
        sun_ra, sun_dec = sun_ra_dec

        with pytest.warns(PointingWarning, match="EXCLUSION ZONE"):
            plan_daisy_scan(
                ra=sun_ra,
                dec=sun_dec,
                radius=0.5,
                velocity=0.3,
                turn_radius=0.2,
                avoidance_radius=0.0,
                start_acceleration=0.5,
                site=site,
                start_time=obstime,
                timestep=0.1,
                duration=60.0,
            )

    def test_plan_pong_scan_passes_safe_field(self, site):
        """plan_pong_scan succeeds for a field far from the sun."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        field = FieldRegion(
            ra_center=180.0,
            dec_center=-30.0,
            width=2.0,
            height=2.0,
        )

        block = plan_pong_scan(
            field=field,
            velocity=0.5,
            spacing=0.1,
            num_terms=4,
            site=site,
            start_time=start_time,
            timestep=0.1,
        )
        assert block.trajectory.n_points > 0


# ---------------------------------------------------------------------------
# Injectable sun_safe predicate (A3 seam): directional model picked up
# end-to-end at planning time, default path unchanged.
# ---------------------------------------------------------------------------

# A field far from the Sun at _TEST_OBSTIME, the scalar default never warns
# here, so any warning a predicate produces proves the predicate (not the
# 45 deg scalar) drove the verdict. RA = anti-solar, Dec well south.
_SAFE_FIELD = FieldRegion(ra_center=0.0, dec_center=-30.0, width=2.0, height=2.0)


def _block_everything(az, el, t):
    """SunSafePredicate that reports every position unsafe (inside the zone)."""
    return False


def _allow_everything(az, el, t):
    """SunSafePredicate that reports every position clear of the Sun."""
    return True


class TestCheckFieldSunSafetyInjectedPredicate:
    """``_check_field_sun_safety`` honors an injected ``sun_safe`` predicate."""

    def test_injected_predicate_flags_otherwise_safe_field(self, site, coords, obstime):
        """A predicate returning False warns even when the scalar check would not.

        The field is placed far from the Sun (anti-solar RA), so the built-in
        scalar exclusion check passes silently. Injecting a predicate that
        returns False must still raise the EXCLUSION ZONE warning, proving
        the directional model's verdict is what is consulted.
        """
        sun_ra, _ = coords.altaz_to_radec(*coords.get_sun_altaz(obstime), obstime)
        safe_ra = (sun_ra + 180.0) % 360.0

        # Precondition: the default (scalar) check is silent for this field.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_field_sun_safety(safe_ra, -30.0, obstime, site)
            assert not [x for x in w if "Field center" in str(x.message)]

        with pytest.warns(PointingWarning, match="EXCLUSION ZONE"):
            _check_field_sun_safety(safe_ra, -30.0, obstime, site, sun_safe=_block_everything)

    def test_injected_predicate_receives_field_altaz(self, site, coords, obstime):
        """The predicate is consulted with the field center's (az, el, time)."""
        seen = []

        def spy(az, el, t):
            seen.append((float(az), float(el)))
            return True

        az_expected, el_expected = coords.radec_to_altaz(120.0, -40.0, obstime)
        _check_field_sun_safety(120.0, -40.0, obstime, site, sun_safe=spy)

        assert len(seen) == 1, "predicate should be consulted exactly once"
        az_seen, el_seen = seen[0]
        assert az_seen == pytest.approx(float(az_expected), abs=1e-6)
        assert el_seen == pytest.approx(float(el_expected), abs=1e-6)

    def test_injected_allow_predicate_overrides_unsafe_field(self, site, obstime, sun_ra_dec):
        """A permissive predicate suppresses the warning for a field AT the Sun.

        Pointing the field at the Sun trips the scalar check, but an injected
        predicate that returns True must override it (warn-only seam, predicate
        owns the verdict).
        """
        sun_ra, sun_dec = sun_ra_dec

        # Precondition: the scalar default DOES warn for a field at the Sun.
        with pytest.warns(PointingWarning, match="EXCLUSION ZONE"):
            _check_field_sun_safety(sun_ra, sun_dec, obstime, site)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_field_sun_safety(sun_ra, sun_dec, obstime, site, sun_safe=_allow_everything)
            assert not [x for x in w if "Field center" in str(x.message)]

    def test_default_none_unchanged(self, site, obstime, sun_ra_dec):
        """sun_safe=None reproduces the built-in scalar behavior exactly."""
        sun_ra, sun_dec = sun_ra_dec
        safe_ra = (sun_ra + 90.0) % 360.0

        # Safe field: silent with and without an explicit None.
        for kw in ({}, {"sun_safe": None}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _check_field_sun_safety(safe_ra, -30.0, obstime, site, **kw)
                assert not [x for x in w if "Field center" in str(x.message)]

        # Field at the Sun: warns with and without an explicit None.
        for kw in ({}, {"sun_safe": None}):
            with pytest.warns(PointingWarning, match="EXCLUSION ZONE"):
                _check_field_sun_safety(sun_ra, sun_dec, obstime, site, **kw)


class TestPlanningEntryPointsInjectedPredicate:
    """The public planners thread ``sun_safe`` to the field pre-flight check."""

    def test_plan_pong_scan_honors_injected_predicate(self, site, obstime):
        """plan_pong_scan warns for an otherwise-safe field when sun_safe=False."""
        # Default path is silent for this safe field.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plan_pong_scan(
                field=_SAFE_FIELD,
                velocity=0.5,
                spacing=0.1,
                num_terms=4,
                site=site,
                start_time=obstime,
                timestep=0.5,
            )
            assert not [x for x in w if "Field center" in str(x.message)]

        with pytest.warns(PointingWarning, match="EXCLUSION ZONE"):
            plan_pong_scan(
                field=_SAFE_FIELD,
                velocity=0.5,
                spacing=0.1,
                num_terms=4,
                site=site,
                start_time=obstime,
                timestep=0.5,
                sun_safe=_block_everything,
            )

    def test_plan_daisy_scan_honors_injected_predicate(self, site, obstime, sun_ra_dec):
        """Thread sun_safe so a permissive predicate suppresses a source-at-Sun warning."""
        sun_ra, sun_dec = sun_ra_dec

        # Precondition: default scalar check warns at the Sun.
        with pytest.warns(PointingWarning, match="EXCLUSION ZONE"):
            plan_daisy_scan(
                ra=sun_ra,
                dec=sun_dec,
                radius=0.5,
                velocity=0.3,
                turn_radius=0.2,
                avoidance_radius=0.0,
                start_acceleration=0.5,
                site=site,
                start_time=obstime,
                timestep=0.5,
                duration=60.0,
            )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plan_daisy_scan(
                ra=sun_ra,
                dec=sun_dec,
                radius=0.5,
                velocity=0.3,
                turn_radius=0.2,
                avoidance_radius=0.0,
                start_acceleration=0.5,
                site=site,
                start_time=obstime,
                timestep=0.5,
                duration=60.0,
                sun_safe=_allow_everything,
            )
            assert not [x for x in w if "Field center" in str(x.message)]

    def test_plan_constant_el_scan_honors_injected_predicate(self, site):
        """plan_constant_el_scan threads sun_safe to its field pre-flight check."""
        field = FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7)
        ce_time = Time("2026-03-15T17:00:00", scale="utc")

        # Default path is silent (the E-CDF-S field is far from the Sun here).
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plan_constant_el_scan(
                field=field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=ce_time,
                rising=True,
            )
            assert not [x for x in w if "Field center" in str(x.message)]

        with pytest.warns(PointingWarning, match="EXCLUSION ZONE"):
            plan_constant_el_scan(
                field=field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=ce_time,
                rising=True,
                sun_safe=_block_everything,
            )


class TestValidateSunAvoidanceInjectedPredicate:
    """``validate_sun_avoidance`` / ``validate_trajectory`` honor ``sun_safe``."""

    def _safe_arrays(self, coords, obstime, n=120):
        """Build an n-point trajectory far from the Sun (anti-solar az, el=45)."""
        sun_az, _ = coords.get_sun_altaz(obstime)
        safe_az = (sun_az + 180.0) % 360.0
        abs_times = obstime + TimeDelta(np.arange(n) * u.s)
        return np.full(n, safe_az), np.full(n, 45.0), abs_times

    def test_injected_predicate_flags_otherwise_safe_trajectory(self, site, coords, obstime):
        """A False predicate warns on a trajectory the scalar check passes."""
        az, el, abs_times = self._safe_arrays(coords, obstime)

        # Default path: silent.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_sun_avoidance(site, az, el, abs_times, coords=coords)
            assert not [x for x in w if issubclass(x.category, PointingWarning)]

        with pytest.warns(PointingWarning, match="EXCLUSION ZONE"):
            validate_sun_avoidance(
                site, az, el, abs_times, coords=coords, sun_safe=_block_everything
            )

    def test_injected_predicate_consulted_per_subsample(self, site, coords, obstime):
        """The predicate sees (az, el, time) triples drawn from the trajectory."""
        az, el, abs_times = self._safe_arrays(coords, obstime, n=240)
        seen = []

        def spy(a, e, t):
            seen.append((float(a), float(e)))
            return True

        validate_sun_avoidance(site, az, el, abs_times, coords=coords, sun_safe=spy)

        assert seen, "predicate was never consulted"
        # Subsampling is preserved: far fewer calls than trajectory points.
        assert len(seen) < 240
        assert all(e == pytest.approx(45.0) for _, e in seen)
        assert all(a == pytest.approx(float(az[0])) for a, _ in seen)

    def test_allow_predicate_overrides_trajectory_at_sun(self, site, coords, obstime):
        """A permissive predicate suppresses the warning for a trajectory at the Sun."""
        sun_az, sun_alt = coords.get_sun_altaz(obstime)
        n = 50
        abs_times = obstime + TimeDelta(np.arange(n) * u.s)
        az = np.full(n, float(sun_az))
        el = np.full(n, float(sun_alt))

        # Precondition: scalar default warns at the Sun.
        with pytest.warns(PointingWarning, match="EXCLUSION ZONE"):
            validate_sun_avoidance(site, az, el, abs_times, coords=coords)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_sun_avoidance(
                site, az, el, abs_times, coords=coords, sun_safe=_allow_everything
            )
            assert not [x for x in w if issubclass(x.category, PointingWarning)]

    def test_validate_trajectory_threads_sun_safe(self, site, coords, obstime):
        """validate_trajectory forwards sun_safe to the sun-avoidance check."""
        az, el, _ = self._safe_arrays(coords, obstime, n=120)
        traj = Trajectory(
            times=np.arange(120, dtype=float),
            az=az,
            el=el,
            az_vel=np.zeros(120),
            el_vel=np.zeros(120),
            start_time=obstime,
        )

        # Default path: no EXCLUSION ZONE warning for this safe trajectory.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_trajectory(traj, site)
            assert not [x for x in w if "EXCLUSION ZONE" in str(x.message)]

        with pytest.warns(PointingWarning, match="EXCLUSION ZONE"):
            validate_trajectory(traj, site, sun_safe=_block_everything)

    def test_validate_trajectory_default_none_unchanged(self, site, coords, obstime):
        """validate_trajectory with sun_safe=None matches the scalar behavior."""
        sun_az, sun_alt = coords.get_sun_altaz(obstime)
        n = 50
        traj = Trajectory(
            times=np.arange(n, dtype=float),
            az=np.full(n, float(sun_az)),
            el=np.full(n, float(sun_alt)),
            az_vel=np.zeros(n),
            el_vel=np.zeros(n),
            start_time=obstime,
        )
        # Trajectory pointed at the Sun warns under the scalar default,
        # with or without an explicit None.
        with pytest.warns(PointingWarning, match="EXCLUSION ZONE"):
            validate_trajectory(traj, site, sun_safe=None)
