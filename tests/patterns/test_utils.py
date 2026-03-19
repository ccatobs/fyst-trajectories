"""Tests for pattern utility functions."""

import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time, TimeDelta

from fyst_trajectories import Coordinates, get_fyst_site
from fyst_trajectories.exceptions import (
    AzimuthBoundsError,
    ElevationBoundsError,
    PointingWarning,
    TrajectoryBoundsError,
)
from fyst_trajectories.patterns.utils import (
    compute_velocities,
    generate_time_array,
    normalize_azimuth,
    sky_offsets_to_altaz,
)
from fyst_trajectories.trajectory_utils import (
    validate_trajectory_bounds,
    validate_trajectory_dynamics,
)


class TestComputeVelocities:
    """Tests for compute_velocities function."""

    def test_basic_velocity(self):
        """Test basic velocity computation with constant velocity."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        positions = np.array([100.0, 101.0, 102.0, 103.0, 104.0])

        velocities = compute_velocities(positions, times, is_angular=False)

        # Constant velocity of 1 deg/s
        np.testing.assert_allclose(velocities, 1.0, rtol=1e-10)

    def test_zero_velocity(self):
        """Test with constant position (zero velocity)."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        positions = np.array([45.0, 45.0, 45.0, 45.0])

        velocities = compute_velocities(positions, times, is_angular=False)

        np.testing.assert_allclose(velocities, 0.0, atol=1e-10)

    def test_negative_velocity(self):
        """Test with decreasing positions."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        positions = np.array([200.0, 198.0, 196.0, 194.0, 192.0])

        velocities = compute_velocities(positions, times, is_angular=False)

        # Constant velocity of -2 deg/s
        np.testing.assert_allclose(velocities, -2.0, rtol=1e-10)

    def test_varying_timesteps(self):
        """Test with non-uniform time steps."""
        times = np.array([0.0, 0.5, 2.0, 2.5, 4.0])
        # Linear motion: position = 10 + 2*t
        positions = 10.0 + 2.0 * times

        velocities = compute_velocities(positions, times, is_angular=False)

        # Velocity should be constant at 2 deg/s
        np.testing.assert_allclose(velocities, 2.0, rtol=1e-10)

    def test_azimuth_wraparound_positive(self):
        """Test azimuth wrap-around from 359 to small positive values.

        This is the critical bug fix test: when azimuth wraps from 359 to 1,
        the true velocity is about +2 deg/s, not -358 deg/s.
        """
        times = np.array([0.0, 1.0, 2.0, 3.0])
        # Azimuth increasing through the 360/0 boundary
        # Moving at about 2 deg/s: 358 -> 359 -> 0/360 -> 1 -> 2
        # Simulating: 358, 0 (=360), 2
        az = np.array([358.0, 0.0, 2.0, 4.0])

        # Without is_angular=True, this would compute ~-358 deg/s
        # With is_angular=True, it should compute ~+2 deg/s
        velocities = compute_velocities(az, times, is_angular=True)

        # All velocities should be close to +2 deg/s
        np.testing.assert_allclose(velocities, 2.0, atol=0.1)

    def test_azimuth_wraparound_negative(self):
        """Test azimuth wrap-around in the negative direction.

        Going from small positive to near 360 (decreasing azimuth).
        """
        times = np.array([0.0, 1.0, 2.0, 3.0])
        # Azimuth decreasing through the 0/360 boundary
        # Moving at about -2 deg/s: 4 -> 2 -> 0 -> 358
        az = np.array([4.0, 2.0, 0.0, 358.0])

        velocities = compute_velocities(az, times, is_angular=True)

        # All velocities should be close to -2 deg/s
        np.testing.assert_allclose(velocities, -2.0, atol=0.1)

    def test_azimuth_multiple_wraps(self):
        """Test multiple wrap-arounds in sequence."""
        # Spinning around multiple times
        times = np.linspace(0, 4, 9)  # 0, 0.5, 1, ..., 4
        # Azimuth increasing at 100 deg/s, wrapping multiple times
        # Start at 350, after 4 seconds should be at 350 + 400 = 750 = 30 (mod 360)
        # Intermediate: 350, 400(40), 450(90), ..., 700(340), 750(30)
        az = np.array([350.0, 40.0, 90.0, 140.0, 190.0, 240.0, 290.0, 340.0, 30.0])

        velocities = compute_velocities(az, times, is_angular=True)

        # Velocity should be close to 100 deg/s
        np.testing.assert_allclose(velocities, 100.0, atol=1.0)

    def test_no_angular_flag_preserves_old_behavior(self):
        """Test that without is_angular=True, behavior is unchanged.

        This tests backward compatibility - existing code that doesn't
        pass is_angular=True should get the same (potentially wrong)
        results as before.
        """
        times = np.array([0.0, 1.0, 2.0])
        az = np.array([358.0, 0.0, 2.0])

        # Without is_angular, the function computes the "wrong" velocity
        velocities = compute_velocities(az, times, is_angular=False)

        # Middle point sees 358->0->2, gradient computes (2-358)/2 = -178
        # This is the "wrong" behavior that is_angular=True fixes
        assert velocities[1] < -100  # Large negative, not +2

    def test_elevation_no_wrap(self):
        """Test that elevation (non-angular) doesn't need unwrapping.

        Elevation never wraps since it's bounded to [0, 90] or similar.
        """
        times = np.array([0.0, 1.0, 2.0, 3.0])
        el = np.array([30.0, 35.0, 40.0, 45.0])

        # is_angular=False is appropriate for elevation
        velocities = compute_velocities(el, times, is_angular=False)

        np.testing.assert_allclose(velocities, 5.0, rtol=1e-10)

    def test_small_array(self):
        """Test with minimal array size."""
        times = np.array([0.0, 1.0])
        positions = np.array([0.0, 10.0])

        velocities = compute_velocities(positions, times, is_angular=False)

        np.testing.assert_allclose(velocities, 10.0, rtol=1e-10)

    def test_azimuth_near_boundary_no_wrap(self):
        """Test azimuth values near 0/360 that don't actually wrap."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        # Values near 0 but not crossing the boundary
        az = np.array([5.0, 10.0, 15.0, 20.0])

        velocities = compute_velocities(az, times, is_angular=True)

        # Should compute correct velocity of 5 deg/s
        np.testing.assert_allclose(velocities, 5.0, rtol=1e-10)

    def test_azimuth_near_360_no_wrap(self):
        """Test azimuth values near 360 that don't wrap."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        # Values near 360 but decreasing (not crossing boundary)
        az = np.array([355.0, 350.0, 345.0, 340.0])

        velocities = compute_velocities(az, times, is_angular=True)

        # Should compute correct velocity of -5 deg/s
        np.testing.assert_allclose(velocities, -5.0, rtol=1e-10)


class TestGenerateTimeArray:
    """Tests for generate_time_array function."""

    def test_basic_case(self):
        """Test basic time array generation."""
        times = generate_time_array(10.0, 1.0)
        assert times[0] == 0.0
        assert times[-1] == 10.0
        assert len(times) == 11  # 0, 1, 2, ..., 10

    def test_non_integer_division(self):
        """Test when duration is not evenly divisible by timestep."""
        times = generate_time_array(10.0, 3.0)
        # round(10/3) = 3, so 3+1 = 4 points
        assert times[0] == 0.0
        assert times[-1] == 10.0
        assert len(times) == 4

    def test_timestep_greater_than_duration(self):
        """Test when timestep exceeds duration -- minimum 2 points."""
        times = generate_time_array(0.5, 1.0)
        # Minimum 2 points enforced: start and end
        assert times[0] == 0.0
        assert times[-1] == pytest.approx(0.5)
        assert len(times) == 2

    def test_timestep_equals_duration(self):
        """Test when timestep equals duration."""
        times = generate_time_array(5.0, 5.0)
        assert times[0] == 0.0
        assert times[-1] == 5.0
        assert len(times) == 2  # Start and end


class TestSkyOffsetsToAltaz:
    """Tests for sky_offsets_to_altaz function."""

    def test_zero_offsets(self):
        """Test that zero offsets produce center position."""
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
        """Test that a small offset matches direct RA/Dec conversion."""
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
        """Test with array of offsets."""
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
    """Tests for validate_trajectory_bounds function."""

    def test_within_limits(self):
        """Test that valid trajectories pass without error."""
        site = get_fyst_site()
        az = np.array([100.0, 150.0, 200.0])
        el = np.array([45.0, 50.0, 55.0])
        validate_trajectory_bounds(site, az, el)  # Should not raise

    def test_az_below_limit(self):
        """Test that azimuth below minimum raises AzimuthBoundsError."""
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
        """Test that azimuth above maximum raises AzimuthBoundsError."""
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
        """Test that elevation below minimum raises ElevationBoundsError."""
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
        """Test that elevation above maximum raises ElevationBoundsError."""
        site = get_fyst_site()
        az = np.array([100.0, 150.0, 200.0])
        el = np.array([45.0, 50.0, 95.0])  # 95 > el_max (90)

        with pytest.raises(ElevationBoundsError, match="elevation") as exc_info:
            validate_trajectory_bounds(site, az, el)

        err = exc_info.value
        assert err.axis == "elevation"
        assert err.actual_max == 95.0
        assert err.limit_max == site.telescope_limits.elevation.max

    def test_backward_compatible_with_valueerror(self):
        """Test that custom exceptions are still caught by ValueError."""
        site = get_fyst_site()
        az = np.array([-300.0, 0.0, 100.0])
        el = np.array([45.0, 45.0, 45.0])

        with pytest.raises(ValueError, match="azimuth"):
            validate_trajectory_bounds(site, az, el)

    def test_structured_data_on_bounds_error(self):
        """Test that TrajectoryBoundsError has structured attributes."""
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
        """Test that exact boundary values are accepted."""
        site = get_fyst_site()
        limits = site.telescope_limits
        az = np.array([limits.azimuth.min, 0.0, limits.azimuth.max])
        el = np.array([limits.elevation.min, 45.0, limits.elevation.max])
        validate_trajectory_bounds(site, az, el)  # Should not raise


class TestValidateTrajectoryDynamics:
    """Tests for validate_trajectory_dynamics function."""

    def test_no_warning_within_limits(self):
        """Test that no warning is issued when within limits."""
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
        """Test warning when velocity exceeds limits."""
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
        """Test warning when acceleration exceeds limits."""
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
    """Tests for normalize_azimuth function.

    The function takes azimuth values from astropy's [0, 360] convention
    and normalizes them into the telescope's allowed range (e.g., [-180, 360])
    by unwrapping discontinuities and shifting by multiples of 360 degrees.
    """

    def test_basic_shift_from_astropy_range(self):
        """Test that values in [0, 360] are shifted into [-180, 360].

        Astropy returns az in [0, 360]. A trajectory centered around 350
        should be shifted by -360 to center around -10, which fits in [-180, 360].
        """
        site = get_fyst_site()
        # Trajectory around az=350, which is equivalent to az=-10
        az = np.array([340.0, 345.0, 350.0, 355.0, 0.0, 5.0, 10.0])
        result = normalize_azimuth(az, site)

        # After unwrap + shift, values should be near -10 (centered)
        # The midpoint of the unwrapped trajectory should be near 0
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
        """Test normalization of a single-point array."""
        site = get_fyst_site()
        az = np.array([350.0])
        result = normalize_azimuth(az, site)

        # Single point at 350 should be shifted to -10
        np.testing.assert_allclose(result, np.array([-10.0]), atol=1e-10)

    def test_shift_is_multiple_of_360(self):
        """Test that the shift applied is always a multiple of 360 degrees."""
        site = get_fyst_site()
        az = np.array([340.0, 345.0, 350.0, 355.0, 0.0, 5.0])
        result = normalize_azimuth(az, site)

        # The unwrapped version should differ from the result by a multiple of 360
        az_unwrapped = np.unwrap(az, period=360.0)
        shift = result[0] - az_unwrapped[0]
        assert shift % 360.0 == pytest.approx(0.0, abs=1e-10)

    def test_multiple_boundary_crossings(self):
        """Test trajectory that crosses the 0/360 boundary multiple times.

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
        result = normalize_azimuth(az, site)

        # The span should be preserved (600 degrees)
        span = result.max() - result.min()
        assert span == pytest.approx(600.0, abs=0.1)

    def test_negative_azimuth_input(self):
        """Test that negative azimuth input is handled correctly.

        While astropy normally returns [0, 360], the function should handle
        pre-processed negative values gracefully.
        """
        site = get_fyst_site()
        # Values already negative and within range
        az = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        result = normalize_azimuth(az, site)

        # These should stay the same (already centered near 0)
        np.testing.assert_allclose(result, az, atol=1e-10)

    def test_consistent_with_validate_trajectory_bounds(self):
        """Test that normalized azimuth passes bounds validation.

        A trajectory centered on az=180 (well within range) should pass
        validation after normalization.
        """
        site = get_fyst_site()
        az = np.array([170.0, 175.0, 180.0, 185.0, 190.0])
        el = np.full(5, 45.0)

        az_normalized = normalize_azimuth(az, site)

        # Should pass bounds check without raising
        validate_trajectory_bounds(site, az_normalized, el)

    def test_trajectory_near_negative_limit(self):
        """Test trajectory near the negative azimuth limit.

        A trajectory around az=90 in astropy convention (=90 in telescope)
        or az=270 (near the positive limit) should be handled correctly.
        """
        site = get_fyst_site()
        # Trajectory near az=265 (close to +270 limit)
        az = np.array([260.0, 263.0, 265.0, 267.0, 270.0])
        result = normalize_azimuth(az, site)

        # Should stay in range
        assert result.max() <= site.telescope_limits.azimuth.max
        assert result.min() >= site.telescope_limits.azimuth.min
