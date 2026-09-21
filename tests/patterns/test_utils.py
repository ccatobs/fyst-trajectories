"""Tests for pattern utility functions."""

import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time, TimeDelta

from fyst_trajectories import (
    Coordinates,
    Trajectory,
    choose_encoder_solution,
    get_fyst_site,
)
from fyst_trajectories.exceptions import (
    AccelerationLimitWarning,
    AzimuthBoundsError,
    ElevationBoundsError,
    PointingError,
    PointingWarning,
    TrajectoryBoundsError,
    VelocityLimitWarning,
)
from fyst_trajectories.patterns.utils import (
    compute_velocities,
    generate_time_array,
    normalize_azimuth,
    rewrap_trajectory_azimuth,
    sky_offsets_to_altaz,
)
from fyst_trajectories.trajectory_utils import (
    validate_trajectory_bounds,
    validate_trajectory_dynamics,
)


class TestComputeVelocities:
    """Gradient velocities, with ``is_angular`` unwrapping azimuth across the 0/360 seam."""

    def test_basic_velocity(self):
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        positions = np.array([100.0, 101.0, 102.0, 103.0, 104.0])

        velocities = compute_velocities(positions, times, is_angular=False)

        # Constant velocity of 1 deg/s
        np.testing.assert_allclose(velocities, 1.0, rtol=1e-10)

    def test_zero_velocity(self):
        times = np.array([0.0, 1.0, 2.0, 3.0])
        positions = np.array([45.0, 45.0, 45.0, 45.0])

        velocities = compute_velocities(positions, times, is_angular=False)

        np.testing.assert_allclose(velocities, 0.0, atol=1e-10)

    def test_negative_velocity(self):
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        positions = np.array([200.0, 198.0, 196.0, 194.0, 192.0])

        velocities = compute_velocities(positions, times, is_angular=False)

        # Constant velocity of -2 deg/s
        np.testing.assert_allclose(velocities, -2.0, rtol=1e-10)

    def test_varying_timesteps(self):
        times = np.array([0.0, 0.5, 2.0, 2.5, 4.0])
        # Linear motion: position = 10 + 2*t
        positions = 10.0 + 2.0 * times

        velocities = compute_velocities(positions, times, is_angular=False)

        # Velocity should be constant at 2 deg/s
        np.testing.assert_allclose(velocities, 2.0, rtol=1e-10)

    def test_azimuth_wraparound_positive(self):
        """Wrapping from 359 to 1 gives about +2 deg/s, not -358 deg/s."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        # Azimuth increasing at 2 deg/s through the 360/0 boundary.
        az = np.array([358.0, 0.0, 2.0, 4.0])

        velocities = compute_velocities(az, times, is_angular=True)

        # All velocities should be close to +2 deg/s
        np.testing.assert_allclose(velocities, 2.0, atol=0.1)

    def test_azimuth_wraparound_negative(self):
        times = np.array([0.0, 1.0, 2.0, 3.0])
        # Azimuth decreasing through the 0/360 boundary
        # Moving at about -2 deg/s: 4 -> 2 -> 0 -> 358
        az = np.array([4.0, 2.0, 0.0, 358.0])

        velocities = compute_velocities(az, times, is_angular=True)

        # All velocities should be close to -2 deg/s
        np.testing.assert_allclose(velocities, -2.0, atol=0.1)

    def test_azimuth_multiple_wraps(self):
        times = np.linspace(0, 4, 9)  # 0, 0.5, 1, ..., 4
        # Azimuth increasing at 100 deg/s, wrapping multiple times
        # Start at 350, after 4 seconds should be at 350 + 400 = 750 = 30 (mod 360)
        # Intermediate: 350, 400(40), 450(90), ..., 700(340), 750(30)
        az = np.array([350.0, 40.0, 90.0, 140.0, 190.0, 240.0, 290.0, 340.0, 30.0])

        velocities = compute_velocities(az, times, is_angular=True)

        # Velocity should be close to 100 deg/s
        np.testing.assert_allclose(velocities, 100.0, atol=1.0)

    def test_no_angular_flag_does_not_unwrap(self):
        """Without ``is_angular=True`` the 0/360 seam is not unwrapped."""
        times = np.array([0.0, 1.0, 2.0])
        az = np.array([358.0, 0.0, 2.0])

        velocities = compute_velocities(az, times, is_angular=False)

        # Middle point sees 358->0->2, gradient computes (2-358)/2 = -178
        # ``is_angular=True`` is what makes this case come out as +2 deg/s
        assert velocities[1] < -100  # Large negative, not +2

    def test_elevation_no_wrap(self):
        """Elevation never wraps, so it needs no unwrapping."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        el = np.array([30.0, 35.0, 40.0, 45.0])

        # is_angular=False is appropriate for elevation
        velocities = compute_velocities(el, times, is_angular=False)

        np.testing.assert_allclose(velocities, 5.0, rtol=1e-10)

    def test_small_array(self):
        times = np.array([0.0, 1.0])
        positions = np.array([0.0, 10.0])

        velocities = compute_velocities(positions, times, is_angular=False)

        np.testing.assert_allclose(velocities, 10.0, rtol=1e-10)

    def test_azimuth_near_boundary_no_wrap(self):
        times = np.array([0.0, 1.0, 2.0, 3.0])
        # Values near 0 but not crossing the boundary
        az = np.array([5.0, 10.0, 15.0, 20.0])

        velocities = compute_velocities(az, times, is_angular=True)

        # Should compute correct velocity of 5 deg/s
        np.testing.assert_allclose(velocities, 5.0, rtol=1e-10)

    def test_azimuth_near_360_no_wrap(self):
        times = np.array([0.0, 1.0, 2.0, 3.0])
        # Values near 360 but decreasing (not crossing boundary)
        az = np.array([355.0, 350.0, 345.0, 340.0])

        velocities = compute_velocities(az, times, is_angular=True)

        # Should compute correct velocity of -5 deg/s
        np.testing.assert_allclose(velocities, -5.0, rtol=1e-10)


class TestGenerateTimeArray:
    """Sample-count rounding, and the two-point floor when the timestep exceeds duration."""

    def test_basic_case(self):
        times = generate_time_array(10.0, 1.0)
        assert times[0] == 0.0
        assert times[-1] == 10.0
        assert len(times) == 11  # 0, 1, 2, ..., 10

    def test_non_integer_division(self):
        times = generate_time_array(10.0, 3.0)
        # round(10/3) = 3, so 3+1 = 4 points
        assert times[0] == 0.0
        assert times[-1] == 10.0
        assert len(times) == 4

    def test_timestep_greater_than_duration(self):
        times = generate_time_array(0.5, 1.0)
        # Minimum 2 points enforced: start and end
        assert times[0] == 0.0
        assert times[-1] == pytest.approx(0.5)
        assert len(times) == 2

    def test_timestep_equals_duration(self):
        times = generate_time_array(5.0, 5.0)
        assert times[0] == 0.0
        assert times[-1] == 5.0
        assert len(times) == 2  # Start and end


class TestSkyOffsetsToAltaz:
    """Offset-to-horizon conversion, checked against direct RA/Dec transforms."""

    def test_zero_offsets(self):
        """Zero offsets land on the centre position."""
        site = get_fyst_site()
        coords = Coordinates(site)
        obstime = Time("2026-03-15T04:00:00", scale="utc")

        x_offsets = np.array([0.0])
        y_offsets = np.array([0.0])

        az, el = sky_offsets_to_altaz(
            x_offsets,
            y_offsets,
            180.0,
            -30.0,
            obstime,
            coords,
        )

        # Should match direct radec_to_altaz of the center
        az_ref, el_ref = coords.radec_to_altaz(180.0, -30.0, obstime)
        np.testing.assert_allclose(az, az_ref, atol=0.001)
        np.testing.assert_allclose(el, el_ref, atol=0.001)

    def test_small_offset_matches_direct_radec(self):
        site = get_fyst_site()
        coords = Coordinates(site)
        obstime = Time("2026-03-15T04:00:00", scale="utc")

        # A 1-degree x offset at dec=0 should shift RA by ~1 degree
        x_offsets = np.array([1.0])
        y_offsets = np.array([0.0])

        az_0, el_0 = sky_offsets_to_altaz(
            x_offsets,
            y_offsets,
            180.0,
            0.0,
            obstime,
            coords,
        )

        # Reference: direct shift (spherical_offsets_by at dec=0 gives RA+1)
        az_ref, el_ref = coords.radec_to_altaz(181.0, 0.0, obstime)
        np.testing.assert_allclose(az_0, az_ref, atol=0.01)
        np.testing.assert_allclose(el_0, el_ref, atol=0.01)

    def test_array_inputs(self):
        site = get_fyst_site()
        coords = Coordinates(site)
        obstime = Time("2026-03-15T04:00:00", scale="utc")
        x_offsets = np.array([0.0, 0.1, -0.1, 0.0])
        y_offsets = np.array([0.0, 0.0, 0.0, 0.1])
        obstimes = obstime + TimeDelta(np.arange(4) * 0.1 * u.s)

        az, el = sky_offsets_to_altaz(
            x_offsets,
            y_offsets,
            180.0,
            -30.0,
            obstimes,
            coords,
        )

        assert len(az) == 4
        assert len(el) == 4
        assert np.all(np.isfinite(az))
        assert np.all(np.isfinite(el))


class TestValidateTrajectoryBounds:
    """Per-axis bounds refusals, their structured attributes, and inclusive limit values."""

    def test_within_limits(self):
        site = get_fyst_site()
        az = np.array([100.0, 150.0, 200.0])
        el = np.array([45.0, 50.0, 55.0])
        validate_trajectory_bounds(site, az, el)  # Should not raise

    def test_az_below_limit(self):
        site = get_fyst_site()
        az = np.array([-300.0, 0.0, 100.0])  # -300 < az_min (-180)
        el = np.array([45.0, 45.0, 45.0])

        with pytest.raises(AzimuthBoundsError, match="azimuth") as exc_info:
            validate_trajectory_bounds(site, az, el)

        err = exc_info.value
        assert err.axis == "azimuth"
        assert err.actual_min == -300.0
        assert err.limit_min == site.telescope_limits.azimuth.min

    def test_az_above_limit(self):
        site = get_fyst_site()
        az = np.array([0.0, 100.0, 400.0])  # 400 > az_max (360)
        el = np.array([45.0, 45.0, 45.0])

        with pytest.raises(AzimuthBoundsError, match="azimuth") as exc_info:
            validate_trajectory_bounds(site, az, el)

        err = exc_info.value
        assert err.axis == "azimuth"
        assert err.actual_max == 400.0
        assert err.limit_max == site.telescope_limits.azimuth.max

    def test_el_below_limit(self):
        site = get_fyst_site()
        az = np.array([100.0, 150.0, 200.0])
        el = np.array([10.0, 45.0, 50.0])  # 10 < el_min (20)

        with pytest.raises(ElevationBoundsError, match="elevation") as exc_info:
            validate_trajectory_bounds(site, az, el)

        err = exc_info.value
        assert err.axis == "elevation"
        assert err.actual_min == 10.0
        assert err.limit_min == site.telescope_limits.elevation.min

    def test_el_above_limit(self):
        site = get_fyst_site()
        az = np.array([100.0, 150.0, 200.0])
        el = np.array([45.0, 50.0, 95.0])  # 95 > el_max (90)

        with pytest.raises(ElevationBoundsError, match="elevation") as exc_info:
            validate_trajectory_bounds(site, az, el)

        err = exc_info.value
        assert err.axis == "elevation"
        assert err.actual_max == 95.0
        assert err.limit_max == site.telescope_limits.elevation.max

    def test_bounds_errors_subclass_valueerror(self):
        """The bounds exceptions subclass ``ValueError``."""
        site = get_fyst_site()
        az = np.array([-300.0, 0.0, 100.0])
        el = np.array([45.0, 45.0, 45.0])

        with pytest.raises(ValueError, match="azimuth"):
            validate_trajectory_bounds(site, az, el)

    def test_structured_data_on_bounds_error(self):
        site = get_fyst_site()
        az = np.array([100.0, 150.0, 200.0])
        el = np.array([10.0, 45.0, 50.0])

        with pytest.raises(TrajectoryBoundsError) as exc_info:
            validate_trajectory_bounds(site, az, el)

        err = exc_info.value
        assert hasattr(err, "axis")
        assert hasattr(err, "actual_min")
        assert hasattr(err, "actual_max")
        assert hasattr(err, "limit_min")
        assert hasattr(err, "limit_max")

    def test_boundary_values(self):
        """The exact limit values are accepted."""
        site = get_fyst_site()
        limits = site.telescope_limits
        az = np.array([limits.azimuth.min, 0.0, limits.azimuth.max])
        el = np.array([limits.elevation.min, 45.0, limits.elevation.max])
        validate_trajectory_bounds(site, az, el)  # Should not raise


class TestValidateTrajectoryDynamics:
    """Velocity and acceleration breaches warn by category; short inputs warn and skip."""

    def test_no_warning_within_limits(self):
        site = get_fyst_site()
        times = np.linspace(0, 10, 100)
        # Slow scan: 0.1 deg/s az velocity
        az = 100.0 + 0.1 * times
        el = np.full_like(times, 45.0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_trajectory_dynamics(site, az, el, times)
            dyn_warnings = [
                x
                for x in w
                if "velocity" in str(x.message).lower() or "acceleration" in str(x.message).lower()
            ]
            assert len(dyn_warnings) == 0

    def test_warns_on_high_velocity(self):
        site = get_fyst_site()
        times = np.linspace(0, 10, 100)
        # Very fast scan: 10 deg/s (exceeds 3 deg/s limit)
        az = 100.0 + 10.0 * times
        el = np.full_like(times, 45.0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_trajectory_dynamics(site, az, el, times)
            vel_warnings = [x for x in w if "velocity" in str(x.message).lower()]
            assert len(vel_warnings) >= 1

    def test_warns_on_high_acceleration(self):
        site = get_fyst_site()
        times = np.linspace(0, 10, 1000)
        # Quadratic motion with high acceleration: a = 5 deg/s^2
        az = 100.0 + 0.5 * 5.0 * times**2
        el = np.full_like(times, 45.0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_trajectory_dynamics(site, az, el, times)
            accel_warnings = [x for x in w if "acceleration" in str(x.message).lower()]
            assert len(accel_warnings) >= 1

    def test_high_velocity_warning_has_velocity_limit_category(self):
        """The velocity breach warning is a ``VelocityLimitWarning`` (category, not text).

        Dispatch-time gates escalate on this category, so a velocity breach
        must carry it regardless of message wording.
        """
        site = get_fyst_site()
        times = np.linspace(0, 10, 100)
        az = 100.0 + 10.0 * times  # 10 deg/s, exceeds the 3 deg/s limit
        el = np.full_like(times, 45.0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_trajectory_dynamics(site, az, el, times)
        assert any(issubclass(x.category, VelocityLimitWarning) for x in w)

    def test_high_acceleration_warning_has_acceleration_limit_category(self):
        """The accel breach warning is an ``AccelerationLimitWarning`` (category, not text)."""
        site = get_fyst_site()
        times = np.linspace(0, 10, 1000)
        az = 100.0 + 0.5 * 5.0 * times**2  # a = 5 deg/s^2, exceeds the limit
        el = np.full_like(times, 45.0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_trajectory_dynamics(site, az, el, times)
        assert any(issubclass(x.category, AccelerationLimitWarning) for x in w)

    def test_single_point_trajectory_warns_skipped(self):
        """Single-point trajectory warns that dynamics validation is skipped entirely."""
        site = get_fyst_site()
        times = np.array([0.0])
        az = np.array([100.0])
        el = np.array([45.0])

        with pytest.warns(PointingWarning, match="fewer than 2 points"):
            validate_trajectory_dynamics(site, az, el, times)

    def test_short_trajectory_warns_skipped(self):
        """Two-point trajectory warns that acceleration validation is skipped."""
        site = get_fyst_site()
        times = np.array([0.0, 1.0])
        az = np.array([100.0, 200.0])
        el = np.array([45.0, 45.0])

        with pytest.warns(PointingWarning, match="only 2 points"):
            validate_trajectory_dynamics(site, az, el, times)

    def test_three_point_trajectory_warns_skipped(self):
        """Three-point trajectory warns that acceleration validation is skipped."""
        site = get_fyst_site()
        times = np.array([0.0, 1.0, 2.0])
        az = np.array([100.0, 200.0, 250.0])
        el = np.array([45.0, 45.0, 45.0])

        with pytest.warns(PointingWarning, match="only 3 points"):
            validate_trajectory_dynamics(site, az, el, times)


class TestNormalizeAzimuth:
    """Trajectories crossing the 0/360 seam come back continuous and in range.

    The function takes azimuth values from astropy's [0, 360] convention
    and normalizes them into the telescope's allowed range (e.g., [-180, 360])
    by unwrapping discontinuities and shifting by multiples of 360 degrees.
    """

    def test_basic_shift_from_astropy_range(self):
        """Values in [0, 360] are shifted into [-180, 360].

        Astropy returns az in [0, 360]. The unwrapped track 340..370 is shifted
        by -360, the whole turn that puts its midpoint (-5) closest to the
        center of the telescope range.
        """
        site = get_fyst_site()
        # Trajectory around az=350, which is equivalent to az=-10
        az = np.array([340.0, 345.0, 350.0, 355.0, 0.0, 5.0, 10.0])
        result = normalize_azimuth(az, site)

        np.testing.assert_allclose(result, [-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0], atol=1e-10)
        assert result.min() >= site.telescope_limits.azimuth.min
        assert result.max() <= site.telescope_limits.azimuth.max

    def test_no_shift_needed_when_already_in_range(self):
        """Test that values already in [-180, 360] are not shifted."""
        site = get_fyst_site()
        # Trajectory already in the telescope's range
        az = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        result = normalize_azimuth(az, site)

        np.testing.assert_allclose(result, az, atol=1e-10)

    def test_unwrap_removes_discontinuity(self):
        """Test that azimuth discontinuity at 0/360 boundary is unwrapped.

        A trajectory crossing from 350 to 10 should be continuous
        (350, 360, 370) not jump (350, 10).
        """
        site = get_fyst_site()
        # Simulate azimuth crossing 360/0 boundary: 350, 355, 0, 5, 10
        az = np.array([350.0, 355.0, 0.0, 5.0, 10.0])
        result = normalize_azimuth(az, site)

        # After normalization, the trajectory should be continuous
        diffs = np.diff(result)
        # All steps should be ~5 degrees (no 355-degree jumps)
        assert np.all(np.abs(diffs) < 10.0)

    def test_trajectory_near_zero_azimuth(self):
        """Test trajectory straddling north (az=0).

        A trajectory going through north should produce continuous values
        near zero azimuth.
        """
        site = get_fyst_site()
        # Tracking through north: 355, 358, 1, 4
        az = np.array([355.0, 358.0, 1.0, 4.0])
        result = normalize_azimuth(az, site)

        # Should be continuous and within range
        diffs = np.diff(result)
        assert np.all(np.abs(diffs) < 10.0)
        assert result.min() >= site.telescope_limits.azimuth.min
        assert result.max() <= site.telescope_limits.azimuth.max

    def test_trajectory_centered_around_180(self):
        """Test trajectory centered around az=180 stays near 180."""
        site = get_fyst_site()
        az = np.array([170.0, 175.0, 180.0, 185.0, 190.0])
        result = normalize_azimuth(az, site)

        # Values around 180 are already in range, should stay there
        np.testing.assert_allclose(result, az, atol=1e-10)

    def test_single_point(self):
        site = get_fyst_site()
        az = np.array([350.0])
        result = normalize_azimuth(az, site)

        # Single point at 350 should be shifted to -10
        np.testing.assert_allclose(result, np.array([-10.0]), atol=1e-10)

    def test_shift_is_multiple_of_360(self):
        site = get_fyst_site()
        az = np.array([340.0, 345.0, 350.0, 355.0, 0.0, 5.0])
        result = normalize_azimuth(az, site)

        # The unwrapped version should differ from the result by a multiple of 360
        az_unwrapped = np.unwrap(az, period=360.0)
        shift = result[0] - az_unwrapped[0]
        assert shift % 360.0 == pytest.approx(0.0, abs=1e-10)

    def test_multiple_boundary_crossings(self):
        """Repeated 0/360 crossings unwrap to a single monotonic track.

        This can happen with a long sidereal track where the object
        crosses north repeatedly (unlikely in practice, but testing robustness).
        """
        site = get_fyst_site()
        # Simulate multiple crossings: going from 350 to 370 (=10) to 380 (=20)
        # In astropy [0,360] this looks like: 350, 355, 0, 5, 10, 15, 20
        az = np.array([350.0, 355.0, 0.0, 5.0, 10.0, 15.0, 20.0])
        result = normalize_azimuth(az, site)

        # Should be continuous monotonically increasing
        diffs = np.diff(result)
        assert np.all(diffs > 0)
        assert np.all(np.abs(diffs) < 10.0)

    def test_wide_trajectory_exceeds_range(self):
        """Test that a trajectory spanning > 540 degrees remains out of range.

        The telescope range is [-180, 360] = 540 degrees total. A trajectory
        wider than 540 degrees cannot fit, and normalize_azimuth does not
        fail, it just places the midpoint as close to center as possible.
        The subsequent validate_trajectory_bounds will catch the violation.
        """
        site = get_fyst_site()
        # A trajectory spanning 600 degrees (wider than 540 degree range)
        az = np.linspace(0, 600, 100)
        # This is already unwrapped (no discontinuities), so unwrap is a no-op
        with pytest.warns(PointingWarning, match="No 360-degree shift"):
            result = normalize_azimuth(az, site)

        # The span should be preserved (600 degrees)
        span = result.max() - result.min()
        assert span == pytest.approx(600.0, abs=0.1)

    def test_negative_azimuth_input(self):
        """Negative input, outside astropy's [0, 360] convention, passes through unchanged."""
        site = get_fyst_site()
        # Values already negative and within range
        az = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        result = normalize_azimuth(az, site)

        # These should stay the same (already centered near 0)
        np.testing.assert_allclose(result, az, atol=1e-10)

    def test_consistent_with_validate_trajectory_bounds(self):
        site = get_fyst_site()
        az = np.array([170.0, 175.0, 180.0, 185.0, 190.0])
        el = np.full(5, 45.0)

        az_normalized = normalize_azimuth(az, site)

        # Should pass bounds check without raising
        validate_trajectory_bounds(site, az_normalized, el)

    def test_trajectory_well_inside_range(self):
        """A trajectory comfortably inside [-180, 360] is left where it is."""
        site = get_fyst_site()
        # 260-270 needs no 360-degree shift to fit the telescope range.
        az = np.array([260.0, 263.0, 265.0, 267.0, 270.0])
        result = normalize_azimuth(az, site)

        np.testing.assert_allclose(result, az, atol=1e-10)
        assert result.max() <= site.telescope_limits.azimuth.max
        assert result.min() >= site.telescope_limits.azimuth.min


class TestRewrapTrajectoryAzimuth:
    """The trajectory side of the azimuth-wrap contract.

    ``normalize_azimuth`` fixes a trajectory's wrap frame when the pattern is
    built and ``choose_encoder_solution`` may pick a different one just before
    the slew; ``rewrap_trajectory_azimuth`` applies that decision to the
    trajectory.
    """

    @staticmethod
    def _trajectory() -> Trajectory:
        """Build a small constant-elevation trajectory to re-wrap."""
        times = np.linspace(0.0, 4.0, 5)
        az = np.array([200.0, 201.0, 202.0, 203.0, 204.0])
        el = np.full(5, 45.0)
        return Trajectory(
            times=times,
            az=az,
            el=el,
            az_vel=np.gradient(az, times),
            el_vel=np.zeros(5),
            coordsys="altaz",
        )

    def test_shifts_every_azimuth_sample(self):
        """A whole-turn shift moves the azimuth track rigidly."""
        traj = self._trajectory()
        shifted = rewrap_trajectory_azimuth(traj, -360.0)
        assert np.allclose(shifted.az, traj.az - 360.0)

    def test_leaves_everything_else_alone(self):
        """Times, elevation and both velocity tracks are untouched."""
        traj = self._trajectory()
        shifted = rewrap_trajectory_azimuth(traj, 360.0)
        assert np.array_equal(shifted.times, traj.times)
        assert np.array_equal(shifted.el, traj.el)
        assert np.array_equal(shifted.az_vel, traj.az_vel)
        assert np.array_equal(shifted.el_vel, traj.el_vel)

    def test_zero_shift_returns_the_same_object(self):
        """The no-op case allocates nothing."""
        traj = self._trajectory()
        assert rewrap_trajectory_azimuth(traj, 0.0) is traj

    def test_does_not_mutate_the_input(self):
        """The source trajectory keeps its own azimuth."""
        traj = self._trajectory()
        before = traj.az.copy()
        rewrap_trajectory_azimuth(traj, -360.0)
        assert np.array_equal(traj.az, before)

    @pytest.mark.parametrize("bad", [1.0, 359.0, -180.0, float("nan"), float("inf")])
    def test_refuses_a_shift_that_is_not_whole_turns(self, bad):
        """Anything but a multiple of 360 would move the trajectory to other sky."""
        with pytest.raises(PointingError, match="whole multiple of 360"):
            rewrap_trajectory_azimuth(self._trajectory(), bad)

    def test_applies_the_encoder_solution_shift(self):
        """The helper consumes ``EncoderSolution.az_shift`` directly."""
        site = get_fyst_site(sun_avoidance_enabled=False)
        traj = self._trajectory()
        solution = choose_encoder_solution(
            -170.0,
            45.0,
            float(traj.az[0]),
            45.0,
            Time("2026-03-15T12:00:00", scale="utc"),
            site,
            goal_az_span=(float(traj.az.min()), float(traj.az.max())),
        )
        commanded = rewrap_trajectory_azimuth(traj, solution.az_shift)
        assert float(commanded.az[0]) == pytest.approx(solution.az)
