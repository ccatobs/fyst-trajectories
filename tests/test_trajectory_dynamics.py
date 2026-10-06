"""Tests for trajectory dynamics - velocity/derivative consistency.

Pattern velocities against their positions (continuity, acceleration
sentinels, the integral invariant); the bounds and dynamics validators (limits,
structured errors, warning categories, short inputs) and their input guards;
the Sun validator's time-type guard and the high-elevation azimuth advisory.
"""

import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time, TimeDelta

from fyst_trajectories import (
    InstrumentOffset,
    get_fyst_site,
)
from fyst_trajectories.exceptions import (
    AccelerationLimitWarning,
    AzimuthBoundsError,
    ElevationBoundsError,
    PointingWarning,
    VelocityLimitWarning,
)
from fyst_trajectories.patterns import (
    ConstantElScanConfig,
    ConstantElScanPattern,
    DaisyScanConfig,
    DaisyScanPattern,
    LinearMotionConfig,
    LinearMotionPattern,
    PongAltAzScanConfig,
    PongAltAzScanPattern,
    PongScanConfig,
    PongScanPattern,
    SiderealTrackConfig,
    SiderealTrackPattern,
    TrajectoryBuilder,
)
from fyst_trajectories.patterns.utils import compute_velocities
from fyst_trajectories.planning import (
    FieldRegion,
    plan_constant_el_scan,
    plan_daisy_scan,
    plan_pong_scan,
)
from fyst_trajectories.primecam import get_primecam_offset
from fyst_trajectories.trajectory import Trajectory
from fyst_trajectories.trajectory_utils import (
    validate_sun_avoidance,
    validate_trajectory,
    validate_trajectory_bounds,
    validate_trajectory_dynamics,
)


@pytest.fixture
def start_time():
    """Return standard start time for trajectory dynamics tests."""
    return Time("2026-03-15T04:00:00", scale="utc")


class TestVelocityMatchesDerivative:
    """Analytic-velocity patterns (linear, constant-el) store velocities matching positions."""

    def _compute_numerical_velocity(
        self, positions: np.ndarray, times: np.ndarray, is_angular: bool = False
    ) -> np.ndarray:
        """Compute numerical velocity using gradient, optionally handling wrap."""
        return compute_velocities(positions, times, is_angular=is_angular)

    def _assert_velocities_match(
        self,
        trajectory,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ):
        """Assert that trajectory velocities match numerical derivatives."""
        az_vel_computed = self._compute_numerical_velocity(
            trajectory.az, trajectory.times, is_angular=True
        )
        el_vel_computed = self._compute_numerical_velocity(
            trajectory.el, trajectory.times, is_angular=False
        )

        np.testing.assert_allclose(
            trajectory.az_vel,
            az_vel_computed,
            rtol=rtol,
            atol=atol,
            err_msg="Azimuth velocity does not match numerical derivative",
        )
        np.testing.assert_allclose(
            trajectory.el_vel,
            el_vel_computed,
            rtol=rtol,
            atol=atol,
            err_msg="Elevation velocity does not match numerical derivative",
        )

    def test_linear_motion_velocity_consistency(self, site, start_time):
        config = LinearMotionConfig(
            timestep=0.1,
            az_start=100.0,
            el_start=45.0,
            az_velocity=0.5,
            el_velocity=0.1,
        )
        pattern = LinearMotionPattern(config)
        trajectory = pattern.generate(site, duration=30.0, start_time=start_time)

        self._assert_velocities_match(trajectory)

    def test_constant_el_scan_velocity_consistency(self, site, start_time):
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=100.0,
            az_stop=150.0,
            elevation=45.0,
            az_speed=2.0,
            az_accel=1.0,
        )
        pattern = ConstantElScanPattern(config)
        trajectory = pattern.generate(site, duration=120.0, start_time=start_time)

        np.testing.assert_allclose(
            trajectory.el_vel,
            np.zeros_like(trajectory.el_vel),
            atol=1e-10,
            err_msg="Elevation velocity should be zero for constant_el scan",
        )

        assert np.abs(trajectory.az_vel).max() <= config.az_speed + 0.01, (
            "Azimuth velocity exceeds configured speed"
        )

        dt = np.diff(trajectory.times)
        daz = np.diff(trajectory.az)
        avg_vel = (trajectory.az_vel[:-1] + trajectory.az_vel[1:]) / 2
        expected_daz = avg_vel * dt

        errors = np.abs(daz - expected_daz)
        tolerance = 0.01

        fraction_within_tolerance = np.sum(errors < tolerance) / len(errors)
        assert fraction_within_tolerance > 0.99, (
            f"Only {fraction_within_tolerance * 100:.1f}% of points within tolerance. "
            f"Expected >99%. Max error: {errors.max():.4f} deg"
        )


class TestVelocityContinuity:
    """Consecutive-sample velocity jumps stay under a per-pattern multiple of the step."""

    def _max_velocity_jump(self, velocities: np.ndarray) -> float:
        """Compute maximum velocity jump between consecutive points."""
        return np.abs(np.diff(velocities)).max()

    def test_pong_velocity_continuity(self, site, start_time):
        config = PongScanConfig(
            timestep=0.1,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)
        trajectory = pattern.generate(site, duration=120.0, start_time=start_time)

        dt = trajectory.times[1] - trajectory.times[0]
        max_az_jump = self._max_velocity_jump(trajectory.az_vel)
        max_el_jump = self._max_velocity_jump(trajectory.el_vel)

        max_reasonable_jump = 5.0 * dt
        assert max_az_jump < max_reasonable_jump, (
            f"Azimuth velocity jump {max_az_jump:.4f} deg/s exceeds reasonable limit"
        )
        assert max_el_jump < max_reasonable_jump, (
            f"Elevation velocity jump {max_el_jump:.4f} deg/s exceeds reasonable limit"
        )

    def test_constant_el_velocity_continuity(self, site, start_time):
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=100.0,
            az_stop=150.0,
            elevation=45.0,
            az_speed=2.0,
            az_accel=1.0,
        )
        pattern = ConstantElScanPattern(config)
        trajectory = pattern.generate(site, duration=200.0, start_time=start_time)

        dt = trajectory.times[1] - trajectory.times[0]
        max_az_jump = self._max_velocity_jump(trajectory.az_vel)

        # The standard turnaround peaks at 1.5 x the configured az_accel.
        max_reasonable_jump = 1.5 * config.az_accel * dt
        assert max_az_jump < max_reasonable_jump, (
            f"Azimuth velocity jump {max_az_jump:.4f} exceeds the turnaround peak"
        )

    def test_sidereal_velocity_continuity(self, site, start_time):
        config = SiderealTrackConfig(timestep=0.1)
        pattern = SiderealTrackPattern(ra=180.0, dec=-30.0, config=config)
        trajectory = pattern.generate(site, duration=300.0, start_time=start_time)

        dt = trajectory.times[1] - trajectory.times[0]
        max_az_jump = self._max_velocity_jump(trajectory.az_vel)
        max_el_jump = self._max_velocity_jump(trajectory.el_vel)

        max_reasonable_jump = 2.0 * dt
        assert max_az_jump < max_reasonable_jump
        assert max_el_jump < max_reasonable_jump

    def test_daisy_velocity_continuity(self, site, start_time):
        config = DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.1,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.0,
        )
        pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)
        trajectory = pattern.generate(site, duration=120.0, start_time=start_time)

        dt = trajectory.times[1] - trajectory.times[0]
        max_az_jump = self._max_velocity_jump(trajectory.az_vel)
        max_el_jump = self._max_velocity_jump(trajectory.el_vel)

        max_reasonable_jump = 5.0 * dt
        assert max_az_jump < max_reasonable_jump
        assert max_el_jump < max_reasonable_jump


class TestAccelerationBounds:
    """Accelerations carry no numerical spikes: a loose 5 deg/s^2 sentinel.

    This is not an axis-limit check. The site's own acceleration limits
    (1.5 deg/s^2 in azimuth, 0.75 in elevation) are what
    ``validate_trajectory_dynamics`` compares a trajectory against, as advisory
    warnings, and the pong and daisy geometries used here exceed the azimuth
    one. Two cases are bounded exactly instead: linear motion at zero, and
    constant-el at the 1.5x turnaround peak its configured acceleration implies.
    """

    MAX_AZIMUTH_ACCEL = 5.0
    MAX_ELEVATION_ACCEL = 5.0

    def _compute_acceleration(self, velocities: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Compute acceleration from velocity array."""
        return np.gradient(velocities, times)

    def _assert_accel_bounds(self, trajectory, az_limit: float = None, el_limit: float = None):
        """Assert that accelerations are within limits."""
        az_limit = az_limit or self.MAX_AZIMUTH_ACCEL
        el_limit = el_limit or self.MAX_ELEVATION_ACCEL

        az_accel = self._compute_acceleration(trajectory.az_vel, trajectory.times)
        el_accel = self._compute_acceleration(trajectory.el_vel, trajectory.times)

        max_az_accel = np.abs(az_accel).max()
        max_el_accel = np.abs(el_accel).max()

        assert max_az_accel <= az_limit * 1.1, (
            f"Max azimuth acceleration {max_az_accel:.2f} deg/s^2 exceeds limit {az_limit} deg/s^2"
        )
        assert max_el_accel <= el_limit * 1.1, (
            f"Max elevation acceleration {max_el_accel:.2f} deg/s^2 "
            f"exceeds limit {el_limit} deg/s^2"
        )

    def test_constant_el_respects_configured_accel(self, site, start_time):
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=100.0,
            az_stop=150.0,
            elevation=45.0,
            az_speed=2.0,
            az_accel=1.0,
        )
        pattern = ConstantElScanPattern(config)
        trajectory = pattern.generate(site, duration=200.0, start_time=start_time)

        # The standard turnaround peaks at 1.5 x the configured az_accel.
        self._assert_accel_bounds(trajectory, az_limit=1.5 * config.az_accel, el_limit=0.1)

    def test_pong_reasonable_acceleration(self, site, start_time):
        config = PongScanConfig(
            timestep=0.1,
            width=1.0,
            height=1.0,
            spacing=0.1,
            velocity=0.3,
            num_terms=4,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)
        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        self._assert_accel_bounds(trajectory)

    def test_daisy_reasonable_acceleration(self, site, start_time):
        config = DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.1,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.0,
        )
        pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)
        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        self._assert_accel_bounds(trajectory)

    def test_linear_zero_acceleration(self, site, start_time):
        config = LinearMotionConfig(
            timestep=0.1,
            az_start=100.0,
            el_start=45.0,
            az_velocity=0.5,
            el_velocity=0.1,
        )
        pattern = LinearMotionPattern(config)
        trajectory = pattern.generate(site, duration=30.0, start_time=start_time)

        az_accel = self._compute_acceleration(trajectory.az_vel, trajectory.times)
        el_accel = self._compute_acceleration(trajectory.el_vel, trajectory.times)

        np.testing.assert_allclose(az_accel, np.zeros_like(az_accel), atol=1e-10)
        np.testing.assert_allclose(el_accel, np.zeros_like(el_accel), atol=1e-10)


class TestPositionContinuity:
    """Position steps stay inside the velocity budget; constant-el and linear hold exactly."""

    def test_pong_position_continuity(self, site, start_time):
        config = PongScanConfig(
            timestep=0.1,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)
        trajectory = pattern.generate(site, duration=120.0, start_time=start_time)

        dt = trajectory.times[1] - trajectory.times[0]
        max_vel = max(np.abs(trajectory.az_vel).max(), np.abs(trajectory.el_vel).max())

        az_diffs = np.abs(np.diff(trajectory.az))
        el_diffs = np.abs(np.diff(trajectory.el))

        expected_max_diff = max_vel * dt * 2.0
        assert az_diffs.max() < expected_max_diff, (
            f"Azimuth position jump {az_diffs.max():.4f} exceeds expected"
        )
        assert el_diffs.max() < expected_max_diff, (
            f"Elevation position jump {el_diffs.max():.4f} exceeds expected"
        )

    def test_constant_el_elevation_is_constant(self, site, start_time):
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=100.0,
            az_stop=150.0,
            elevation=45.0,
            az_speed=2.0,
            az_accel=0.5,
        )
        pattern = ConstantElScanPattern(config)
        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        np.testing.assert_allclose(
            trajectory.el,
            np.full_like(trajectory.el, 45.0),
            atol=1e-10,
        )

    def test_linear_position_is_linear(self, site, start_time):
        config = LinearMotionConfig(
            timestep=0.1,
            az_start=100.0,
            el_start=45.0,
            az_velocity=0.5,
            el_velocity=0.1,
        )
        pattern = LinearMotionPattern(config)
        trajectory = pattern.generate(site, duration=30.0, start_time=start_time)

        expected_az = 100.0 + 0.5 * trajectory.times
        expected_el = 45.0 + 0.1 * trajectory.times

        np.testing.assert_allclose(trajectory.az, expected_az, atol=1e-10)
        np.testing.assert_allclose(trajectory.el, expected_el, atol=1e-10)

    def test_sidereal_no_jumps(self, site, start_time):
        config = SiderealTrackConfig(timestep=0.1)
        pattern = SiderealTrackPattern(ra=180.0, dec=-30.0, config=config)
        trajectory = pattern.generate(site, duration=300.0, start_time=start_time)

        dt = trajectory.times[1] - trajectory.times[0]

        az_diffs = np.abs(np.diff(trajectory.az))
        el_diffs = np.abs(np.diff(trajectory.el))

        max_expected_diff = 0.1 * dt * 2
        assert az_diffs.max() < max_expected_diff
        assert el_diffs.max() < max_expected_diff


class TestAzimuthWrapAround:
    """A constant-elevation scan through azimuth 0 keeps bounded, continuous velocities."""

    def test_trajectory_crossing_north(self, site):
        start_time = Time("2026-03-15T04:00:00", scale="utc")

        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=-10.0,
            az_stop=10.0,
            elevation=45.0,
            az_speed=1.0,
            az_accel=0.5,
        )
        pattern = ConstantElScanPattern(config)
        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        assert np.abs(trajectory.az_vel).max() < 2.0, (
            "Velocity should be bounded, not showing wrap artifacts"
        )

        dt = trajectory.times[1] - trajectory.times[0]
        dv = np.diff(trajectory.az_vel)
        max_dv = np.abs(dv).max()
        assert max_dv < 2.0 * dt, f"Velocity jump {max_dv} too large at wrap"


class TestBuilderDynamics:
    """The builder keeps velocities consistent with positions under a detector offset."""

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory azimuth acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_builder_with_offset_velocity_consistency(self, site):
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        offset = InstrumentOffset(dx=30.0, dy=30.0)

        trajectory = (
            TrajectoryBuilder(site)
            .at(ra=180.0, dec=-30.0)
            .with_config(
                PongScanConfig(
                    timestep=0.1,
                    width=1.0,
                    height=1.0,
                    spacing=0.1,
                    velocity=0.3,
                    num_terms=4,
                    angle=0.0,
                )
            )
            .for_detector(offset)
            .duration(60.0)
            .starting_at(start_time)
            .build()
        )

        dt = np.diff(trajectory.times)
        avg_az_vel = (trajectory.az_vel[:-1] + trajectory.az_vel[1:]) / 2
        avg_el_vel = (trajectory.el_vel[:-1] + trajectory.el_vel[1:]) / 2

        daz = np.diff(np.unwrap(trajectory.az, period=360.0))
        del_ = np.diff(trajectory.el)

        expected_daz = avg_az_vel * dt
        expected_del = avg_el_vel * dt

        az_errors = np.abs(daz - expected_daz)
        el_errors = np.abs(del_ - expected_del)

        az_tol = np.maximum(0.02, 0.1 * np.abs(expected_daz))
        el_tol = np.maximum(0.02, 0.1 * np.abs(expected_del))

        az_within_tol = np.sum(az_errors < az_tol) / len(az_errors)
        el_within_tol = np.sum(el_errors < el_tol) / len(el_errors)

        assert az_within_tol > 0.95, (
            f"Only {az_within_tol * 100:.1f}% of azimuth points within tolerance"
        )
        assert el_within_tol > 0.95, (
            f"Only {el_within_tol * 100:.1f}% of elevation points within tolerance"
        )


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

        with (
            pytest.warns(VelocityLimitWarning),
            pytest.warns(PointingWarning, match="only 2 points") as record,
        ):
            validate_trajectory_dynamics(site, az, el, times)
        # The velocity check still runs on a short trajectory: 100 deg/s is far
        # over the azimuth limit.
        assert any(issubclass(w.category, VelocityLimitWarning) for w in record)

    def test_three_point_trajectory_warns_skipped(self):
        """Three-point trajectory warns that acceleration validation is skipped."""
        site = get_fyst_site()
        times = np.array([0.0, 1.0, 2.0])
        az = np.array([100.0, 200.0, 250.0])
        el = np.array([45.0, 45.0, 45.0])

        with (
            pytest.warns(VelocityLimitWarning),
            pytest.warns(PointingWarning, match="only 3 points") as record,
        ):
            validate_trajectory_dynamics(site, az, el, times)
        # The velocity check still runs on a short trajectory: 100 deg/s is far
        # over the azimuth limit.
        assert any(issubclass(w.category, VelocityLimitWarning) for w in record)


def _uniform_motion(site, axis, factor):
    """Return ``(az, el, times)`` moving one axis at ``factor`` times its velocity limit."""
    limits = site.telescope_limits
    times = np.linspace(0.0, 10.0, 101)
    az = np.full_like(times, 100.0)
    el = np.full_like(times, 30.0)
    if axis == "azimuth":
        az = az + factor * limits.azimuth.max_velocity * times
    else:
        el = el + factor * limits.elevation.max_velocity * times
    return az, el, times


class TestVelocityLimitTolerance:
    """A velocity peak at its limit does not warn from rounding; one above it does."""

    def test_constant_el_scan_at_velocity_ceiling_does_not_warn(self):
        """A constant-elevation scan planned at the azimuth velocity limit validates clean.

        The position-derived peak lands on the limit only to within rounding,
        and a dispatcher that escalates ``VelocityLimitWarning`` would refuse
        the scan if rounding counted as a breach.
        """
        site = get_fyst_site()
        limit = site.telescope_limits.azimuth.max_velocity
        field = FieldRegion(ra_center=0.0, dec_center=-2.0, width=10.0, height=6.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            block = plan_constant_el_scan(
                field=field,
                elevation=45.0,
                velocity=limit,
                site=site,
                start_time="2026-09-15T00:00:00",
            )
            trajectory = block.trajectory
            validate_trajectory_dynamics(site, trajectory.az, trajectory.el, trajectory.times)
        derived = np.gradient(np.unwrap(trajectory.az, period=360.0), trajectory.times)
        assert abs(float(np.max(np.abs(derived))) - limit) <= 1e-9 * limit
        velocity = [str(x.message) for x in w if issubclass(x.category, VelocityLimitWarning)]
        assert velocity == []

    @pytest.mark.parametrize("axis", ["azimuth", "elevation"])
    def test_rounding_above_limit_does_not_warn(self, axis):
        """A peak above the limit by one part in 1e12 is rounding, not a breach."""
        site = get_fyst_site()
        az, el, times = _uniform_motion(site, axis, 1.0 + 1e-12)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_trajectory_dynamics(site, az, el, times)
        velocity = [str(x.message) for x in w if issubclass(x.category, VelocityLimitWarning)]
        assert velocity == []

    @pytest.mark.parametrize("axis", ["azimuth", "elevation"])
    def test_one_part_in_a_million_above_limit_warns(self, axis):
        """The tolerance cannot hide a real violation: 1e-6 above the limit warns."""
        site = get_fyst_site()
        az, el, times = _uniform_motion(site, axis, 1.0 + 1e-6)
        with pytest.warns(VelocityLimitWarning, match=f"{axis} velocity"):
            validate_trajectory_dynamics(site, az, el, times)


def _column_trajectory(az_vel, el_vel, az_rate=1.0):
    """Return a 20 s trajectory moving in azimuth at ``az_rate`` with constant velocity columns."""
    times = np.arange(0.0, 20.0, 0.1)
    return Trajectory(
        times=times,
        az=100.0 + az_rate * times,
        el=np.full_like(times, 50.0),
        az_vel=np.full_like(times, az_vel),
        el_vel=np.full_like(times, el_vel),
    )


def _velocity_messages(trajectory, site):
    """Return the ``VelocityLimitWarning`` messages ``validate_trajectory`` emits."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_trajectory(trajectory, site, check_sun=False)
    return [str(x.message) for x in w if issubclass(x.category, VelocityLimitWarning)]


class TestValidateTrajectoryVelocityColumns:
    """``validate_trajectory`` checks the commanded ``az_vel``/``el_vel`` columns.

    The columns are what ``to_path_format`` uploads, so a column over a limit
    warns even when the positions move within it.
    """

    @pytest.mark.parametrize("column", [5.0, -5.0])
    def test_azimuth_column_over_limit_warns(self, site, column):
        """An azimuth column of 5 deg/s warns although the positions move at 1 deg/s."""
        limit = site.telescope_limits.azimuth.max_velocity
        assert abs(column) > limit
        expected = (
            f"Commanded azimuth velocity column ({abs(column):.2f} deg/s) "
            f"exceeds limit ({limit:.2f} deg/s)."
        )
        assert _velocity_messages(_column_trajectory(column, 0.0), site) == [expected]

    def test_elevation_column_over_limit_warns(self, site):
        """An elevation column of 2 deg/s warns although the elevation is fixed."""
        limit = site.telescope_limits.elevation.max_velocity
        assert limit < 2.0
        with pytest.warns(VelocityLimitWarning, match="Commanded elevation velocity column"):
            validate_trajectory(_column_trajectory(1.0, 2.0), site, check_sun=False)

    @pytest.mark.parametrize("factor", [1.0, 1.0 + 1e-12])
    def test_column_at_limit_does_not_warn(self, site, factor):
        """A column exactly at its limit, or above it only by rounding, is no breach."""
        limits = site.telescope_limits
        trajectory = _column_trajectory(
            factor * limits.azimuth.max_velocity, factor * limits.elevation.max_velocity
        )
        assert _velocity_messages(trajectory, site) == []

    def test_over_limit_in_both_quantities_warns_once_for_each(self, site):
        """Positions and column both over the limit give one warning for each."""
        limit = site.telescope_limits.azimuth.max_velocity
        messages = _velocity_messages(_column_trajectory(3.5, 0.0, az_rate=3.5), site)
        assert messages == [
            f"Trajectory azimuth velocity (3.50 deg/s) exceeds limit ({limit:.2f} deg/s).",
            f"Commanded azimuth velocity column (3.50 deg/s) exceeds limit ({limit:.2f} deg/s).",
        ]

    @pytest.mark.parametrize("pattern", ["pong", "daisy"])
    def test_library_pattern_under_limits_emits_no_velocity_warning(self, site, pattern):
        """A planned pong or daisy whose velocities are within the limits stays silent."""
        if pattern == "pong":
            field = FieldRegion(ra_center=180.0, dec_center=-30.0, width=2.0, height=2.0)
            block = plan_pong_scan(
                field, velocity=0.3, spacing=0.1, site=site, start_time="2026-06-15T04:00:00"
            )
        else:
            block = plan_daisy_scan(
                180.0,
                -30.0,
                radius=1.0,
                velocity=0.5,
                turn_radius=0.5,
                avoidance_radius=0.1,
                start_acceleration=0.5,
                site=site,
                start_time="2026-06-15T04:00:00",
                duration=300.0,
            )
        assert _velocity_messages(block.trajectory, site) == []

    @pytest.mark.parametrize("offset", [None, get_primecam_offset("i1")], ids=["boresight", "i1"])
    def test_constant_el_scan_at_velocity_ceiling_gains_no_warning(self, site, offset):
        """A constant-elevation scan at the azimuth limit, with or without an offset, is clean.

        Its column is analytic and sits on the limit to within rounding, also
        after a detector offset.
        """
        field = FieldRegion(ra_center=0.0, dec_center=-2.0, width=10.0, height=6.0)
        block = plan_constant_el_scan(
            field,
            elevation=45.0,
            velocity=site.telescope_limits.azimuth.max_velocity,
            site=site,
            start_time="2026-09-15T00:00:00",
            detector_offset=offset,
        )
        assert _velocity_messages(block.trajectory, site) == []


class TestValidatorInputGuards:
    """Validators reject non-finite / non-monotonic input, not silently pass.

    Without an explicit guard a NaN slips through every comparison (``NaN >
    limit`` is ``False``), even the raising bounds gate, and duplicate timestamps
    make ``np.gradient`` divide by zero. Both validators raise instead.
    """

    def test_bounds_rejects_nan(self, site):
        az = np.array([100.0, np.nan, 120.0])
        el = np.array([45.0, 50.0, 55.0])
        with pytest.raises(ValueError, match="[Nn]on-finite"):
            validate_trajectory_bounds(site, az, el)

    def test_bounds_rejects_inf(self, site):
        az = np.array([100.0, 110.0, 120.0])
        el = np.array([45.0, np.inf, 55.0])
        with pytest.raises(ValueError, match="[Nn]on-finite"):
            validate_trajectory_bounds(site, az, el)

    def test_dynamics_rejects_nan_az(self, site):
        times = np.arange(6) * 0.1
        az = np.full(6, 120.0)
        az[3] = np.nan
        el = np.full(6, 45.0)
        with pytest.raises(ValueError, match="[Nn]on-finite"):
            validate_trajectory_dynamics(site, az, el, times)

    def test_dynamics_rejects_inf_times(self, site):
        times = np.arange(6) * 0.1
        times[4] = np.inf
        az = np.full(6, 120.0)
        el = np.full(6, 45.0)
        with pytest.raises(ValueError, match="[Nn]on-finite"):
            validate_trajectory_dynamics(site, az, el, times)

    def test_dynamics_rejects_duplicate_times(self, site):
        times = np.array([0.0, 0.1, 0.1, 0.3, 0.4])  # not strictly increasing
        az = np.full(5, 120.0)
        el = np.full(5, 45.0)
        with pytest.raises(ValueError, match="increasing"):
            validate_trajectory_dynamics(site, az, el, times)

    def test_dynamics_rejects_decreasing_times(self, site):
        times = np.array([0.0, 0.1, 0.05, 0.3, 0.4])  # step goes backwards
        az = np.full(5, 120.0)
        el = np.full(5, 45.0)
        with pytest.raises(ValueError, match="increasing"):
            validate_trajectory_dynamics(site, az, el, times)


class TestIntegralVelocityEqualsPosition:
    """The cumulative integral of velocity recovers position.

    A velocity computed on a different time grid from its position breaks this
    invariant, and so does a stretched grid that scales every velocity. The
    assertion is ``cumtrapz(az_vel, times) ~ az - az[0]`` (and the same for
    el), on a daisy, a sidereal track, a builder-produced pong and a source-CES block. A ~1%
    grid-scale velocity bias surfaces here as a ~1% relative miss, well above
    the ~0.1% trapezoid/gradient round-trip floor.
    """

    @staticmethod
    def _assert_integral_recovers_position(traj):
        from scipy.integrate import cumulative_trapezoid

        for pos, vel, name in [
            (traj.az, traj.az_vel, "az"),
            (traj.el, traj.el_vel, "el"),
        ]:
            recon = cumulative_trapezoid(vel, traj.times, initial=0.0)
            err = float(np.max(np.abs(recon - (pos - pos[0]))))
            excursion = float(pos.max() - pos.min())
            tol = max(3e-3 * excursion, 1e-3)  # catch ~1% grid bias; pass ~0.1% floor
            assert err < tol, (
                f"{name}: integral(velocity) misses position by {err:.3e} deg "
                f"(excursion {excursion:.3f} deg, tol {tol:.3e})"
            )

    def test_daisy_integral_recovers_position(self, site, start_time):
        cfg = DaisyScanConfig(
            timestep=0.05,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.0,
        )
        traj = DaisyScanPattern(ra=180.0, dec=-30.0, config=cfg).generate(
            site, duration=60.0, start_time=start_time
        )
        self._assert_integral_recovers_position(traj)

    def test_sidereal_integral_recovers_position(self, site, start_time):
        traj = SiderealTrackPattern(
            ra=180.0, dec=-30.0, config=SiderealTrackConfig(timestep=0.1)
        ).generate(site, duration=60.0, start_time=start_time)
        self._assert_integral_recovers_position(traj)

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_pong_integral_recovers_position(self, site, start_time):
        traj = (
            TrajectoryBuilder(site)
            .at(ra=180.0, dec=-30.0)
            .with_config(
                PongScanConfig(
                    timestep=0.05,
                    width=1.0,
                    height=1.0,
                    spacing=0.1,
                    velocity=0.5,
                    num_terms=4,
                    angle=0.0,
                )
            )
            .duration(60.0)
            .starting_at(start_time)
            .build()
        )
        self._assert_integral_recovers_position(traj)

    def test_source_ces_integral_recovers_position(self, site):
        from fyst_trajectories.planning import plan_source_ces
        from fyst_trajectories.primecam import PRIMECAM_MODULES

        modules = [PRIMECAM_MODULES[k] for k in ("c", "i1", "i2", "i3", "i4", "i5", "i6")]
        block = plan_source_ces(
            body="jupiter",
            footprint=modules,
            el_bore=35.0,
            night=Time("2026-03-15T00:00:00", scale="utc"),
            mode="rising",
            site=site,
        )
        self._assert_integral_recovers_position(block.trajectory)


class TestOnSkyAzimuthAdvisory:
    """The advisory judges one sample against its own on-sky rate.

    Comparing the largest coordinate azimuth rate anywhere in the trajectory
    with the largest on-sky rate anywhere would compare two samples that are
    generally not the same one, and a trajectory that moves fast where it is
    high and slower where it is low would have its high-elevation pinch masked
    by the low samples' healthy on-sky rate.
    """

    @staticmethod
    def _two_speed_trajectory():
        """One trajectory that moves fast where it is high and slower where it is low.

        The first half runs at 1.0 deg/s of coordinate azimuth at el 75, where
        cos(el) is 0.259, so its on-sky rate is 0.259 deg/s. The second half
        runs at 0.6 deg/s at el 20, where cos(el) is 0.940 and the on-sky rate
        is 0.564 deg/s. Comparing the global maxima of the two quantities (1.0
        against 0.564) would stay silent; the samples at el 75, whose on-sky
        rate is a quarter of their coordinate rate, are reported.
        """
        times = np.arange(0.0, 40.0, 0.1)
        n = len(times)
        half = n // 2
        el = np.concatenate([np.full(half, 75.0), np.full(n - half, 20.0)])
        rate = np.concatenate([np.full(half, 1.0), np.full(n - half, 0.6)])
        az = 180.0 + np.cumsum(rate) * 0.1
        return az, el, times

    @pytest.mark.filterwarnings(
        "ignore:Trajectory elevation velocity:fyst_trajectories.exceptions.VelocityLimitWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_high_elevation_pinch_is_reported(self, site):
        """The fast, high-elevation half trips the advisory."""
        az, el, times = self._two_speed_trajectory()
        with pytest.warns(PointingWarning, match="reduces on-sky azimuth speed"):
            validate_trajectory_dynamics(site, az, el, times)

    def test_message_quotes_the_judged_sample(self, site):
        """The message quotes the judged sample's own numbers, at el 75 deg."""
        az, el, times = self._two_speed_trajectory()
        with pytest.warns(PointingWarning) as record:
            validate_trajectory_dynamics(site, az, el, times)
        # The samples at el 75 move at 1.0 deg/s; cos(75 deg) = 0.259, so they
        # cover 0.26 deg/s of sky. cos(20 deg) = 0.940 must not appear.
        assert "to 0.26 deg/s" in _advisory_message(record)
        assert "(coordinate: 1.00 deg/s, cos(el)=0.259)" in _advisory_message(record)

    def test_uniformly_low_trajectory_is_not_flagged(self, site):
        """A scan that never reaches high elevation still passes silently."""
        times = np.arange(0.0, 40.0, 0.1)
        az = 180.0 + times * 0.5
        el = np.full(len(times), 30.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", PointingWarning)
            validate_trajectory_dynamics(site, az, el, times)


_N_GRID = 1000

#: Time grids for the high-elevation advisory: uniform in floating point (the
#: 0.125 s steps), uniform only to rounding (0.1 s steps, from zero or from a
#: large offset), and deliberately non-uniform.
_ADVISORY_GRIDS = {
    "0.1s": np.arange(_N_GRID) * 0.1,
    "0.125s": np.arange(_N_GRID) * 0.125,
    "0.1s-linspace": np.linspace(0.0, 99.9, _N_GRID),
    "0.1s-from-5000s": 5000.0 + np.arange(_N_GRID) * 0.1,
    "0.1s-from-1.7e9s": 1.7e9 + np.arange(_N_GRID) * 0.1,
    "random-0.05-to-0.15s": np.cumsum(np.random.default_rng(20261003).uniform(0.05, 0.15, _N_GRID)),
    "alternating-0.01-and-1s": np.cumsum(np.tile([0.01, 1.0], _N_GRID // 2)),
}


def _advisory_message(record):
    """Return the high-elevation advisory's message from a ``pytest.warns`` record."""
    return next(str(w.message) for w in record if "reduces on-sky azimuth speed" in str(w.message))


def _fixed_az_ramp(times, az0, el_top):
    """Return ``(az, el)``: azimuth fixed at ``az0``, elevation rising from 30 deg to ``el_top``.

    The elevation rises linearly in time, so its rate stays under the site
    limit on every grid in ``_ADVISORY_GRIDS``.
    """
    frac = (times - times[0]) / (times[-1] - times[0])
    return np.full(times.size, az0), 30.0 + (el_top - 30.0) * frac


def _constant_rate_sweep(rel, rising):
    """Return ``(az, el)``: azimuth at 0.5 deg/s, elevation linear in time between 40 and 70 deg.

    ``rel`` is the time since the first sample; the elevation rises from 40 to
    70 deg, or falls from 70 to 40 deg, over it.
    """
    frac = rel / rel[-1]
    el = 40.0 + 30.0 * (frac if rising else 1.0 - frac)
    return 100.0 + 0.5 * rel, el


#: The advisory of a 0.5 deg/s sweep judged at el 70 deg: cos(70 deg) = 0.342,
#: so the on-sky speed is 0.5 * 0.342 = 0.17 deg/s.
_SWEEP_AT_70_DEG = (
    "to 0.17 deg/s at the highest sample moving at half the top azimuth speed or more "
    "(coordinate: 0.50 deg/s, cos(el)=0.342)"
)


class TestJudgedSample:
    """The highest sample that moves at half the top azimuth speed or more is judged.

    The threshold is far from any rounding scale, so the verdict and the quoted
    numbers follow the trajectory: whichever time grid, time origin or storage
    precision carries it, a 0.5 deg/s sweep from 40 to 70 deg is judged at its
    top, el 70 deg. Taking the single fastest sample instead would let rounding
    choose among samples whose computed speeds differ by rounding alone: the
    same sweep would warn on 0.1 s steps and pass silently on 0.125 s steps.
    """

    @pytest.mark.parametrize("timestep", [0.1, 0.125, 0.2, 0.25, 0.3])
    def test_linear_motion_sweep_warns_on_every_timestep(self, site, timestep):
        """``LinearMotionConfig`` at 0.5 deg/s in both axes is judged at its top, el 70 deg."""
        builder = (
            TrajectoryBuilder(site)
            .with_config(
                LinearMotionConfig(
                    timestep=timestep,
                    az_start=100.0,
                    el_start=40.0,
                    az_velocity=0.5,
                    el_velocity=0.5,
                )
            )
            .duration(60.0)
        )
        with pytest.warns(PointingWarning, match="reduces on-sky azimuth speed") as record:
            builder.build()
        assert _SWEEP_AT_70_DEG in _advisory_message(record)

    @pytest.mark.parametrize("rising", [True, False], ids=["rising", "falling"])
    @pytest.mark.parametrize("grid", list(_ADVISORY_GRIDS))
    def test_constant_rate_sweep_warns_on_every_grid(self, site, grid, rising):
        """A hand-built constant-rate sweep is judged at its highest sample on any grid."""
        times = _ADVISORY_GRIDS[grid]
        az, el = _constant_rate_sweep(times - times[0], rising)
        with pytest.warns(PointingWarning, match="reduces on-sky azimuth speed") as record:
            validate_trajectory_dynamics(site, az, el, times)
        assert _SWEEP_AT_70_DEG in _advisory_message(record)

    @pytest.mark.parametrize("storage", ["exact-times-at-1.7e9s", "float32"])
    def test_constant_rate_sweep_survives_its_storage(self, site, storage):
        """Positions from unrounded times at 1.7e9 s, or float32 storage, leave one message.

        Near 1.7e9 s a stored 0.1 s step carries about 2.4e-7 s of rounding, so
        the computed speeds of a constant rate spread by a few parts in a
        million; float32 storage spreads them by about one part in ten
        thousand. Neither comes near half the top speed.
        """
        rel = np.arange(_N_GRID) * 0.1
        az, el = _constant_rate_sweep(rel, rising=True)
        if storage == "float32":
            az, el, times = (a.astype(np.float32) for a in (az, el, rel))
        else:
            times = 1.7e9 + rel
        # Precondition: the storage spreads the computed speeds.
        assert np.ptp(np.abs(np.gradient(np.asarray(az, float), np.asarray(times, float)))) > 1e-7
        with pytest.warns(PointingWarning, match="reduces on-sky azimuth speed") as record:
            validate_trajectory_dynamics(site, az, el, times)
        assert _SWEEP_AT_70_DEG in _advisory_message(record)

    @pytest.mark.parametrize(("high_rate", "silent"), [(0.4, True), (0.6, False)])
    def test_half_the_top_speed_decides_which_samples_are_judged(self, site, high_rate, silent):
        """A segment at el 75 deg is judged only when it moves at half the top speed or more.

        The low segment runs at 1.0 deg/s at el 45; the high one at 0.4 or
        0.6 deg/s, on either side of half the top speed. The step in elevation
        between them also trips the elevation limits, which are not the subject.
        """
        times = np.arange(0.0, 40.0, 0.1)
        half = times.size // 2
        el = np.concatenate([np.full(half, 45.0), np.full(times.size - half, 75.0)])
        rate = np.concatenate([np.full(half, 1.0), np.full(times.size - half, high_rate)])
        az = 180.0 + np.cumsum(rate) * 0.1
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            validate_trajectory_dynamics(site, az, el, times)
        advisories = [str(w.message) for w in record if "on-sky azimuth speed" in str(w.message)]
        if silent:
            assert advisories == []
            return
        # cos(75 deg) = 0.259 and 0.6 * 0.259 = 0.16 deg/s on sky.
        assert len(advisories) == 1
        assert "to 0.16 deg/s" in advisories[0]
        assert "(coordinate: 0.60 deg/s, cos(el)=0.259)" in advisories[0]

    def test_on_equal_elevation_the_faster_sample_is_quoted(self, site):
        """At a single elevation the fastest of the judged samples is quoted."""
        times = np.arange(0.0, 40.0, 0.1)
        half = times.size // 2
        rate = np.concatenate([np.full(half, 0.5), np.full(times.size - half, 0.3)])
        az = 180.0 + np.cumsum(rate) * 0.1
        el = np.full(times.size, 65.0)
        with pytest.warns(PointingWarning, match="reduces on-sky azimuth speed") as record:
            validate_trajectory_dynamics(site, az, el, times)
        # cos(65 deg) = 0.423 and 0.5 * 0.423 = 0.21 deg/s on sky.
        assert "to 0.21 deg/s" in _advisory_message(record)
        assert "(coordinate: 0.50 deg/s, cos(el)=0.423)" in _advisory_message(record)

    @pytest.mark.parametrize(
        ("el_low", "storage", "quoted"),
        [
            (59.9996, "float64", "(coordinate: 1.00 deg/s, cos(el)=0.500)"),
            (59.999499999, "float64", "(coordinate: 0.60 deg/s, cos(el)=0.500)"),
            (59.999499999, "float32", "(coordinate: 1.00 deg/s, cos(el)=0.500)"),
        ],
        ids=["faster-inside-band", "faster-just-outside", "faster-rounded-into-band-by-float32"],
    )
    def test_the_quoting_band_never_decides_the_verdict(self, site, el_low, storage, quoted):
        """A faster sample just under 60 deg changes only which numbers are quoted.

        A segment at 0.6 deg/s at el 60.0005 deg is followed by one at 1.0 deg/s
        just under 60 deg. The highest sample decides, so all three warn. The
        faster sample is quoted when it lies within 0.001 deg of the highest: at
        59.9996 deg, or at 59.999499999 deg once float32 storage rounds it into
        that band (in float64 it lies 0.001000001 deg below). The step in speed
        also trips the acceleration limit, which is not the subject.
        """
        rel = np.arange(400) * 0.1
        first = rel < 20.0
        az = np.where(first, 100.0 + 0.6 * rel, 112.0 + 1.0 * (rel - 20.0))
        el = np.where(first, 60.0005, el_low)
        if storage == "float32":
            el = el.astype(np.float32)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            validate_trajectory_dynamics(site, az, el, rel)
        advisories = [str(w.message) for w in record if "on-sky azimuth speed" in str(w.message)]
        assert len(advisories) == 1
        assert quoted in advisories[0]

    def test_altaz_pong_message_does_not_depend_on_time_origin_or_precision(self, site):
        """The same AltAz pong timed from 0, 5000 s or 1.7e9 s, or stored in float32, reads once.

        At 0.125 s steps the stamps from 1.7e9 s are exact, so all four carry
        the same motion and give the same message.
        """
        pattern = PongAltAzScanPattern(
            PongAltAzScanConfig(
                az_center=120.0,
                el_center=59.0,
                width=2.0,
                height=2.0,
                spacing=0.1,
                velocity=0.37,
                timestep=0.125,
            )
        )
        trajectory = pattern.generate(site, duration=300.0)
        az, el, times = trajectory.az, trajectory.el, trajectory.times
        messages = set()
        for a, e, t in (
            (az, el, times),
            (az, el, 5000.0 + times),
            (az, el, 1.7e9 + times),
            (az.astype(np.float32), el.astype(np.float32), times),
        ):
            with pytest.warns(PointingWarning, match="reduces on-sky azimuth speed") as record:
                validate_trajectory_dynamics(site, a, e, t)
            messages.add(_advisory_message(record))
        assert len(messages) == 1
        # The legs top out at el 60.07 deg; the fastest sample within 0.001 deg of
        # that top moves at 0.58 deg/s, 0.29 deg/s on sky.
        assert "to 0.29 deg/s" in next(iter(messages))
        assert "(coordinate: 0.58 deg/s, cos(el)=0.499)" in next(iter(messages))

    def test_a_speed_that_overflows_is_left_out(self, site):
        """A first time step of 5e-324 s overflows one speed; the finite samples are judged.

        The validator returns, and the samples that move at 1 deg/s at el 70
        deg give the advisory.
        """
        times = np.array([0.0, 5e-324, 1.0, 2.0, 3.0])
        az = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        el = np.full(times.size, 70.0)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            validate_trajectory_dynamics(site, az, el, times)
        advisories = [str(w.message) for w in record if issubclass(w.category, PointingWarning)]
        assert len(advisories) == 1
        assert "(coordinate: 1.00 deg/s, cos(el)=0.342)" in advisories[0]

    def test_an_infinite_time_step_does_not_crash(self, site):
        """A time step that overflows to infinity leaves the validator returning normally."""
        times = np.array([-1.7e308, 1.7e308, 1.75e308, 1.79e308])
        az = np.array([10.0, 11.0, 12.0, 13.0])
        el = np.full(times.size, 70.0)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            validate_trajectory_dynamics(site, az, el, times)


class TestNoAzimuthMotion:
    """A trajectory whose azimuth spans less than 0.001 deg gets no high-elevation advisory.

    Without azimuth motion there is no azimuth rate for high elevation to
    compress. Whether the azimuth moves is read from the span of the positions,
    against a floor far above what rounding leaves in a fixed pointing, so the
    verdict does not depend on the time grid: on 0.1 s steps the computed speed
    of a constant azimuth is rounding noise rather than zero.
    """

    def test_tenth_second_steps_give_a_constant_a_nonzero_gradient(self):
        """Precondition of the grid cases: on 0.1 s steps the gradient of a constant is noise."""
        times = _ADVISORY_GRIDS["0.1s"]
        assert np.abs(np.gradient(np.full(times.size, 123.4), times)).max() > 0.0

    @pytest.mark.parametrize("az0", [123.4, 359.99, 360.0])
    @pytest.mark.parametrize("grid", list(_ADVISORY_GRIDS))
    def test_fixed_azimuth_ramp_is_silent_on_every_grid(self, site, grid, az0):
        """A fixed azimuth rising from 30 to 80 deg passes silently on any grid."""
        times = _ADVISORY_GRIDS[grid]
        az, el = _fixed_az_ramp(times, az0, el_top=80.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", PointingWarning)
            validate_trajectory_dynamics(site, az, el, times)

    @pytest.mark.parametrize(
        "case", ["noise-1e-13", "noise-1e-12", "step-1e-9", "round-trip", "float32-midpoint"]
    )
    def test_rounding_level_motion_does_not_move(self, site, case):
        """What rounding leaves in a fixed pointing at el 75 deg passes silently.

        The cases: a constant with noise of 1e-13 or 1e-12 deg; one step of
        1e-9 deg mid-trajectory; an AltAz to ICRS to AltAz round trip of a
        fixed pointing; and float32 storage of a pointing that sits on a
        float32 rounding midpoint near 360 deg, which leaves one float32 step,
        3.05e-5 deg.
        """
        times = np.arange(120) * 1.0
        el = np.full(times.size, 75.0)
        rng = np.random.default_rng(20261004)
        if case == "noise-1e-13":
            az = 200.0 + rng.normal(0.0, 1e-13, times.size)
        elif case == "noise-1e-12":
            az = 200.0 + rng.normal(0.0, 1e-12, times.size)
        elif case == "step-1e-9":
            az = np.full(times.size, 200.0)
            az[times.size // 2 :] += 1e-9
        elif case == "round-trip":
            frame = AltAz(
                obstime=Time("2026-06-01T04:00:00", scale="utc") + times * u.s,
                location=site.location,
            )
            fixed = SkyCoord(az=np.full(times.size, 200.0) * u.deg, alt=el * u.deg, frame=frame)
            az = fixed.icrs.transform_to(frame).az.deg
        else:
            lo = np.float32(359.7)
            mid = (float(lo) + float(np.nextafter(lo, np.float32(np.inf)))) / 2.0
            az = (mid + np.tile([1e-11, -1e-11], times.size // 2)).astype(np.float32)
            assert np.ptp(az.astype(float)) == pytest.approx(3.05e-5, rel=1e-3)
        # Precondition: the azimuth does move, by rounding.
        assert 0.0 < np.ptp(np.asarray(az, dtype=float)) < 1e-4
        with warnings.catch_warnings():
            warnings.simplefilter("error", PointingWarning)
            validate_trajectory_dynamics(site, az, el, times)

    @pytest.mark.parametrize(("span", "silent"), [(5e-4, True), (2e-3, False)])
    def test_the_travel_floor_separates_still_from_moving(self, site, span, silent):
        """A drift spanning 5e-4 deg does not move; one spanning 2e-3 deg does, and is judged."""
        rel = np.arange(_N_GRID) * 0.1
        az = 180.0 + span * rel / rel[-1]
        el = np.full(rel.size, 75.0)
        if silent:
            with warnings.catch_warnings():
                warnings.simplefilter("error", PointingWarning)
                validate_trajectory_dynamics(site, az, el, rel)
            return
        with pytest.warns(PointingWarning, match="reduces on-sky azimuth speed") as record:
            validate_trajectory_dynamics(site, az, el, rel)
        assert "cos(el)=0.259" in _advisory_message(record)

    @pytest.mark.parametrize("timestep", [0.1, 0.125, 0.3])
    def test_builder_elevation_ramp_at_fixed_azimuth_is_silent(self, site, timestep):
        """``LinearMotionConfig`` with no azimuth velocity, rising to 70 deg, passes silently."""
        builder = (
            TrajectoryBuilder(site)
            .with_config(
                LinearMotionConfig(
                    timestep=timestep,
                    az_start=180.0,
                    el_start=40.0,
                    az_velocity=0.0,
                    el_velocity=0.5,
                )
            )
            .duration(60.0)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", PointingWarning)
            builder.build()

    def test_validate_trajectory_passes_a_fixed_azimuth_ramp(self, site):
        """``validate_trajectory`` passes a hand-built fixed-azimuth ramp to 80 deg silently."""
        times = _ADVISORY_GRIDS["0.1s"]
        az, el = _fixed_az_ramp(times, 200.0, el_top=80.0)
        trajectory = Trajectory(
            times=times,
            az=az,
            el=el,
            az_vel=np.zeros(times.size),
            el_vel=np.gradient(el, times),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", PointingWarning)
            validate_trajectory(trajectory, site, check_sun=False)


class TestSunAvoidanceTimesType:
    """``validate_sun_avoidance`` takes its times as an astropy ``Time`` only.

    A plain numeric array would reach the Sun ephemeris, or an injected
    predicate outside its contract (whose ``time`` is a ``Time``), and die
    several frames from the caller; it is refused up front, naming the
    argument.
    """

    def test_float_times_without_a_predicate_is_refused(self, site):
        """A numeric time array raises a TypeError naming the argument."""
        az = np.array([180.0, 181.0, 182.0])
        el = np.array([45.0, 45.0, 45.0])
        with pytest.raises(TypeError, match="times must be an astropy Time"):
            validate_sun_avoidance(site, az, el, np.array([0.0, 1.0, 2.0]))

    def test_float_times_with_a_predicate_is_refused(self, site):
        """An injected predicate does not make a numeric time array acceptable."""
        az = np.array([180.0, 181.0, 182.0])
        el = np.array([45.0, 45.0, 45.0])
        with pytest.raises(TypeError, match="times must be an astropy Time"):
            validate_sun_avoidance(
                site,
                az,
                el,
                np.array([0.0, 1.0, 2.0]),
                sun_safe=lambda a, e, t: True,
            )

    def test_time_array_still_works(self, site):
        """The documented ``Time`` form is unaffected."""
        az = np.array([180.0, 181.0, 182.0])
        el = np.array([45.0, 45.0, 45.0])
        step = TimeDelta(1.0, format="sec")
        times = Time("2026-06-15T04:00:00", scale="utc") + np.arange(3) * step
        validate_sun_avoidance(site, az, el, times)
