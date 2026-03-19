"""Tests for the custom exception hierarchy.

Verifies:
- Exception inheritance (PointingError -> ValueError backward compat)
- Structured data on all exception types
- Pattern-level error wrapping (TargetNotObservableError)
- AltAz pattern direct bounds errors
- Trajectory.validate() bounds errors
"""

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories.exceptions import (
    AzimuthBoundsError,
    ElevationBoundsError,
    PointingError,
    TargetNotObservableError,
    TrajectoryBoundsError,
)
from fyst_trajectories.patterns import (
    ConstantElScanConfig,
    ConstantElScanPattern,
    DaisyScanConfig,
    DaisyScanPattern,
    LinearMotionConfig,
    LinearMotionPattern,
    PongScanConfig,
    PongScanPattern,
    TrajectoryBuilder,
)
from fyst_trajectories.trajectory import Trajectory


class TestExceptionStructuredData:
    """Test that exceptions carry correct structured data."""

    def test_target_not_observable_attributes(self):
        """Test TargetNotObservableError has all structured attributes."""
        bounds = ElevationBoundsError(
            actual_min=-10.0,
            actual_max=15.0,
            limit_min=20.0,
            limit_max=90.0,
        )
        exc = TargetNotObservableError(
            target="RA=180.000 Dec=80.000",
            time_info="2026-03-15T04:00:00.000",
            bounds_error=bounds,
        )
        assert exc.target == "RA=180.000 Dec=80.000"
        assert exc.time_info == "2026-03-15T04:00:00.000"
        assert exc.bounds_error is bounds
        assert exc.bounds_error.axis == "elevation"
        assert exc.bounds_error.actual_min == -10.0

    def test_azimuth_bounds_error_message(self):
        """Test AzimuthBoundsError has a meaningful message."""
        exc = AzimuthBoundsError(
            actual_min=-300.0,
            actual_max=200.0,
            limit_min=-270.0,
            limit_max=270.0,
        )
        msg = str(exc)
        assert "azimuth" in msg
        assert "-300.00" in msg
        assert "200.00" in msg
        assert "-270" in msg
        assert "270" in msg

    def test_elevation_bounds_error_message(self):
        """Test ElevationBoundsError has a meaningful message."""
        exc = ElevationBoundsError(
            actual_min=10.0,
            actual_max=85.0,
            limit_min=20.0,
            limit_max=90.0,
        )
        msg = str(exc)
        assert "elevation" in msg
        assert "10.00" in msg

    def test_target_not_observable_message(self):
        """Test TargetNotObservableError has a meaningful message."""
        bounds = ElevationBoundsError(
            actual_min=-10.0,
            actual_max=15.0,
            limit_min=20.0,
            limit_max=90.0,
        )
        exc = TargetNotObservableError(
            target="Mars",
            time_info="2026-03-15T04:00:00",
            bounds_error=bounds,
        )
        msg = str(exc)
        assert "Mars" in msg
        assert "2026-03-15T04:00:00" in msg
        assert "elevation" in msg

    def test_target_not_observable_wraps_original(self):
        """Test that TargetNotObservableError preserves the original bounds error."""
        bounds = AzimuthBoundsError(
            actual_min=-280.0,
            actual_max=100.0,
            limit_min=-270.0,
            limit_max=270.0,
        )
        exc = TargetNotObservableError(
            target="RA=350.0 Dec=-30.0",
            time_info="2026-06-15",
            bounds_error=bounds,
        )
        # The wrapped error is accessible and has the right type
        assert isinstance(exc.bounds_error, AzimuthBoundsError)
        assert isinstance(exc.bounds_error, TrajectoryBoundsError)
        assert exc.bounds_error.axis == "azimuth"
        assert exc.bounds_error.actual_min == -280.0


class TestPongRaisesTargetNotObservable:
    """Test that PongScanPattern raises TargetNotObservableError for unobservable targets."""

    def test_pong_unobservable_high_dec(self, site):
        """Test that Pong raises TargetNotObservableError for target below horizon."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = PongScanConfig(
            timestep=0.1,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        )
        # Dec=+80 never visible from FYST (lat -22.96)
        pattern = PongScanPattern(ra=180.0, dec=80.0, config=config)

        with pytest.raises(TargetNotObservableError) as exc_info:
            pattern.generate(site, duration=300.0, start_time=start_time)

        exc = exc_info.value
        assert "RA=180.000" in exc.target
        assert "Dec=80.000" in exc.target
        assert exc.time_info == start_time.iso
        assert isinstance(exc.bounds_error, TrajectoryBoundsError)

    def test_pong_unobservable_is_catchable_as_valueerror(self, site):
        """Test backward compatibility: TargetNotObservableError caught as ValueError."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = PongScanConfig(
            timestep=0.1,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=80.0, config=config)

        with pytest.raises(ValueError):
            pattern.generate(site, duration=300.0, start_time=start_time)


class TestDaisyRaisesTargetNotObservable:
    """Test that DaisyScanPattern raises TargetNotObservableError for unobservable targets."""

    def test_daisy_unobservable_high_dec(self, site):
        """Test that Daisy raises TargetNotObservableError for target below horizon."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.0,
        )
        # Dec=+80 never visible from FYST (lat -22.96)
        pattern = DaisyScanPattern(ra=180.0, dec=80.0, config=config)

        with pytest.raises(TargetNotObservableError) as exc_info:
            pattern.generate(site, duration=300.0, start_time=start_time)

        exc = exc_info.value
        assert "RA=180.000" in exc.target
        assert "Dec=80.000" in exc.target
        assert isinstance(exc.bounds_error, TrajectoryBoundsError)


class TestConstantElRaisesBoundsError:
    """Test that ConstantElScanPattern raises bounds errors directly."""

    def test_elevation_below_limit(self, site):
        """Test ElevationBoundsError for elevation below minimum."""
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=120.0,
            az_stop=180.0,
            elevation=15.0,
            az_speed=1.0,
            az_accel=0.5,
            n_scans=4,
        )
        pattern = ConstantElScanPattern(config)

        with pytest.raises(ElevationBoundsError) as exc_info:
            pattern.generate(site, duration=120.0, start_time=None)

        exc = exc_info.value
        assert exc.axis == "elevation"
        assert exc.actual_min == 15.0
        assert exc.actual_max == 15.0
        assert exc.limit_min == site.telescope_limits.elevation.min

    def test_azimuth_out_of_range(self, site):
        """Test AzimuthBoundsError for azimuth exceeding limits."""
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=-280.0,
            az_stop=-260.0,
            elevation=45.0,
            az_speed=1.0,
            az_accel=0.5,
            n_scans=4,
        )
        pattern = ConstantElScanPattern(config)

        with pytest.raises(AzimuthBoundsError) as exc_info:
            pattern.generate(site, duration=120.0, start_time=None)

        exc = exc_info.value
        assert exc.axis == "azimuth"
        assert exc.actual_min < site.telescope_limits.azimuth.min

    def test_constant_el_not_target_not_observable(self, site):
        """Test that ConstantEl raises bounds error directly, not TargetNotObservable."""
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=120.0,
            az_stop=180.0,
            elevation=15.0,
            az_speed=1.0,
            az_accel=0.5,
            n_scans=4,
        )
        pattern = ConstantElScanPattern(config)

        with pytest.raises(ElevationBoundsError):
            pattern.generate(site, duration=120.0, start_time=None)

        # Should NOT be TargetNotObservableError
        try:
            pattern.generate(site, duration=120.0, start_time=None)
        except TargetNotObservableError:
            pytest.fail("ConstantElScanPattern should not raise TargetNotObservableError")
        except ElevationBoundsError:
            pass  # Expected


class TestLinearRaisesBoundsError:
    """Test that LinearMotionPattern raises bounds errors directly."""

    def test_elevation_exceeds_limit(self, site):
        """Test ElevationBoundsError when linear motion goes above max elevation."""
        config = LinearMotionConfig(
            timestep=0.1,
            az_start=100.0,
            el_start=85.0,
            az_velocity=0.0,
            el_velocity=1.0,
        )
        pattern = LinearMotionPattern(config)

        start_time = Time("2026-03-15T04:00:00", scale="utc")
        with pytest.raises(ElevationBoundsError) as exc_info:
            pattern.generate(site, duration=60.0, start_time=start_time)

        exc = exc_info.value
        assert exc.axis == "elevation"
        assert exc.actual_max > site.telescope_limits.elevation.max


class TestBuilderRaisesExceptions:
    """Test that TrajectoryBuilder propagates exceptions correctly."""

    def test_builder_propagates_target_not_observable(self, site):
        """Test that builder propagates TargetNotObservableError from celestial patterns."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")

        with pytest.raises(TargetNotObservableError):
            TrajectoryBuilder(site).at(
                ra=180.0,
                dec=80.0,
            ).with_config(
                PongScanConfig(
                    timestep=0.1,
                    width=2.0,
                    height=2.0,
                    spacing=0.1,
                    velocity=0.5,
                    num_terms=4,
                    angle=0.0,
                )
            ).duration(300.0).starting_at(start_time).build()

    def test_builder_propagates_elevation_bounds_error(self, site):
        """Test that builder propagates ElevationBoundsError from AltAz patterns."""
        with pytest.raises(ElevationBoundsError):
            TrajectoryBuilder(site).with_config(
                ConstantElScanConfig(
                    timestep=0.1,
                    az_start=120.0,
                    az_stop=180.0,
                    elevation=15.0,
                    az_speed=1.0,
                    az_accel=0.5,
                    n_scans=4,
                )
            ).duration(120.0).build()


class TestTrajectoryValidateExceptions:
    """Test that Trajectory.validate() raises the correct exceptions."""

    def test_validate_azimuth_out_of_bounds(self, site):
        """Test that validate raises AzimuthBoundsError for out-of-range azimuth."""
        traj = Trajectory(
            times=np.array([0.0, 1.0, 2.0]),
            az=np.array([100.0, 110.0, 400.0]),
            el=np.full(3, 45.0),
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
        )
        with pytest.raises(AzimuthBoundsError) as exc_info:
            traj.validate(site)

        exc = exc_info.value
        assert exc.axis == "azimuth"
        assert exc.actual_max == 400.0

    def test_validate_elevation_out_of_bounds(self, site):
        """Test that validate raises ElevationBoundsError for out-of-range elevation."""
        traj = Trajectory(
            times=np.array([0.0, 1.0, 2.0]),
            az=np.array([100.0, 110.0, 120.0]),
            el=np.array([10.0, 45.0, 50.0]),
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
        )
        with pytest.raises(ElevationBoundsError) as exc_info:
            traj.validate(site)

        exc = exc_info.value
        assert exc.axis == "elevation"
        assert exc.actual_min == 10.0

    def test_validate_catchable_as_pointing_error(self, site):
        """Test that validate errors are catchable as PointingError."""
        traj = Trajectory(
            times=np.array([0.0, 1.0, 2.0]),
            az=np.array([100.0, 110.0, 400.0]),
            el=np.full(3, 45.0),
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
        )
        with pytest.raises(PointingError):
            traj.validate(site)


class TestExceptionChaining:
    """Test that exception chaining suppresses noisy tracebacks."""

    def test_target_not_observable_suppresses_chained_traceback(self, site):
        """Test that TargetNotObservableError suppresses the chained traceback.

        The ``raise ... from None`` pattern suppresses the inner
        TrajectoryBoundsError traceback to keep error output clean.
        The original error is still accessible via the ``bounds_error``
        attribute for programmatic inspection.
        """
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = PongScanConfig(
            timestep=0.1,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=80.0, config=config)

        with pytest.raises(TargetNotObservableError) as exc_info:
            pattern.generate(site, duration=300.0, start_time=start_time)

        exc = exc_info.value
        # __cause__ is None because we use "raise ... from None" to suppress
        # the chained traceback for cleaner error output
        assert exc.__cause__ is None
        # __suppress_context__ is True when "from None" is used
        assert exc.__suppress_context__ is True
        # The original bounds error is still available via the attribute
        assert isinstance(exc.bounds_error, TrajectoryBoundsError)
