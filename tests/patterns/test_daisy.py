"""Tests for DaisyScanPattern."""

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories import Coordinates
from fyst_trajectories.patterns import DaisyScanConfig, DaisyScanPattern


class TestDaisyScanPattern:
    """Tests for Daisy scan pattern."""

    def test_basic_daisy_scan(self, site):
        """Test generating a basic Daisy scan pattern."""
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
        pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=120.0, start_time=start_time)

        assert trajectory.n_points > 0
        assert trajectory.duration == pytest.approx(120.0, abs=0.2)
        assert trajectory.start_time == start_time
        assert trajectory.pattern_type == "daisy"
        assert trajectory.center_ra == 180.0
        assert trajectory.center_dec == -30.0

    def test_daisy_crosses_center(self, site):
        """Test that Daisy pattern crosses near the center."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.2,
            turn_radius=0.15,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.0,
        )
        pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=300.0, start_time=start_time)

        coords = Coordinates(site)
        center_az, center_el = coords.radec_to_altaz(180.0, -30.0, obstime=start_time)

        # On-sky offset frame: the az component must be scaled by cos(el) or the
        # metric over-weights azimuth at this declination. The rosette passes
        # essentially through the centre, so the closest approach is well under
        # a turn radius, far tighter than the old 0.5 deg bound.
        dx = (trajectory.az - center_az) * np.cos(np.radians(trajectory.el))
        dy = trajectory.el - center_el
        min_distance = np.hypot(dx, dy).min()

        assert min_distance < 0.1

    @pytest.mark.slow
    def test_daisy_constant_velocity(self, site):
        """Test that Daisy pattern maintains approximately constant velocity."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.2,
            turn_radius=0.15,
            avoidance_radius=0.0,
            start_acceleration=1.0,
            y_offset=0.0,
        )
        pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=120.0, start_time=start_time)

        total_vel = np.sqrt(trajectory.az_vel**2 + trajectory.el_vel**2)

        ramp_time = config.velocity / config.start_acceleration
        ramp_samples = int(ramp_time / (trajectory.times[1] - trajectory.times[0])) + 5

        steady_state_vel = total_vel[ramp_samples:]

        vel_std = np.std(steady_state_vel)
        vel_mean = np.mean(steady_state_vel)

        assert vel_std / vel_mean < 0.5

    def test_daisy_with_offset(self, site):
        """Test Daisy pattern with y_offset."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config_no_offset = DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.2,
            turn_radius=0.15,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.0,
        )
        config_with_offset = DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.2,
            turn_radius=0.15,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.2,
        )

        pattern_no_offset = DaisyScanPattern(ra=180.0, dec=-30.0, config=config_no_offset)
        pattern_with_offset = DaisyScanPattern(ra=180.0, dec=-30.0, config=config_with_offset)

        traj_no_offset = pattern_no_offset.generate(site, duration=60.0, start_time=start_time)
        traj_with_offset = pattern_with_offset.generate(site, duration=60.0, start_time=start_time)

        assert not np.allclose(traj_no_offset.az, traj_with_offset.az)

    def test_daisy_metadata_stored(self, site):
        """Test that Daisy pattern stores metadata correctly."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.1,
            start_acceleration=0.5,
            y_offset=0.05,
        )
        pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        assert trajectory.pattern_params is not None
        params = trajectory.pattern_params
        assert params["radius"] == 0.5
        assert params["velocity"] == 0.3
        assert params["turn_radius"] == 0.2
        assert params["avoidance_radius"] == 0.1
        assert params["start_acceleration"] == 0.5
        assert params["y_offset"] == 0.05

    def test_daisy_small_radius(self, site):
        """Test Daisy pattern with small radius."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = DaisyScanConfig(
            timestep=0.1,
            radius=0.1,
            velocity=0.1,
            turn_radius=0.05,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.0,
        )
        pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        assert trajectory.n_points > 0
        assert np.all(np.isfinite(trajectory.az))
        assert np.all(np.isfinite(trajectory.el))

    def test_daisy_finite_positions(self, site):
        """Test that Daisy pattern produces finite positions."""
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
        pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=120.0, start_time=start_time)

        assert np.all(np.isfinite(trajectory.az))
        assert np.all(np.isfinite(trajectory.el))
        assert np.all(np.isfinite(trajectory.az_vel))
        assert np.all(np.isfinite(trajectory.el_vel))


class TestDaisyScanFlags:
    """Daisy emits scan_flag, with the start ramp-up flagged as non-science."""

    def test_scan_flag_populated(self, site):
        """``Trajectory.scan_flag`` is no longer ``None`` for Daisy patterns."""
        from fyst_trajectories.trajectory import SCAN_FLAG_SCIENCE, SCAN_FLAG_TURNAROUND

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
        pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=120.0, start_time=start_time)

        assert trajectory.scan_flag is not None
        assert trajectory.scan_flag.shape == trajectory.times.shape
        # The trajectory must contain both flag values
        assert np.any(trajectory.scan_flag == SCAN_FLAG_SCIENCE)
        assert np.any(trajectory.scan_flag == SCAN_FLAG_TURNAROUND)

    def test_initial_ramp_up_flagged_as_turnaround(self, site):
        """The first samples (during start_acceleration ramp-up) are non-science."""
        from fyst_trajectories.trajectory import SCAN_FLAG_TURNAROUND

        start_time = Time("2026-03-15T04:00:00", scale="utc")
        # A slow start_acceleration relative to velocity makes the ramp
        # take several timesteps so the test is not razor-thin.
        config = DaisyScanConfig(
            timestep=0.05,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.3,
            y_offset=0.0,
        )
        pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)
        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        # First few samples must be flagged as non-science.
        assert trajectory.scan_flag[0] == SCAN_FLAG_TURNAROUND


class TestDaisyAvoidanceRadius:
    """A non-zero avoidance_radius holds the rosette out of a central keep-out."""

    def test_avoidance_radius_keeps_out_center(self):
        # Measured in the offset frame (free of tracking drift); y_offset starts
        # the path outside the keep-out so the start is not trivially at centre.
        common = dict(
            timestep=0.1,
            radius=1.0,
            velocity=0.3,
            turn_radius=0.2,
            start_acceleration=0.5,
            y_offset=0.5,
        )
        keepout = DaisyScanPattern(
            ra=180.0, dec=-30.0, config=DaisyScanConfig(avoidance_radius=0.3, **common)
        )
        no_keepout = DaisyScanPattern(
            ra=180.0, dec=-30.0, config=DaisyScanConfig(avoidance_radius=0.0, **common)
        )

        _, xk, yk = keepout.generate_offsets(600.0)
        _, x0, y0 = no_keepout.generate_offsets(600.0)

        # With the keep-out the closest approach to centre is ~avoidance_radius;
        # without it the path passes essentially through the centre.
        assert np.hypot(xk, yk).min() >= 0.3 - 0.02
        assert np.hypot(x0, y0).min() < 0.05


class TestDaisyTimeGrid:
    """The reported time grid matches the integrator grid (no linspace stretch).

    Samples are exactly ``config.timestep`` apart; a ``linspace`` re-labelling would
    stretch the axis (~1% at a 10 s scan), biasing every derived velocity. Mirrors
    ``test_constant_el.py``'s position/velocity-consistency guard.
    """

    def _config(self):
        return DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.0,
        )

    def test_time_grid_is_uniform_at_timestep(self):
        config = self._config()
        pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)
        times, _, _ = pattern.generate_offsets(duration=10.0)

        dt = np.diff(times)
        assert np.allclose(dt, config.timestep, rtol=0, atol=1e-9), (
            f"time grid not uniform at timestep: diff range "
            f"[{dt.min():.6f}, {dt.max():.6f}] vs timestep {config.timestep}"
        )

    def test_cruise_speed_recovers_velocity(self):
        config = self._config()
        pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)
        times, x, y = pattern.generate_offsets(duration=10.0)

        # Inter-sample (segment) speed ds/dt: directly probes whether the time
        # grid recovers the integrator's speed, without the second-derivative
        # discretization error that np.gradient adds on a curving path. A
        # stretched time axis scales every segment speed by the stretch factor.
        seg_speed = np.hypot(np.diff(x), np.diff(y)) / np.diff(times)
        # Cruise = the petal arcs held at the target speed; the start ramp and
        # center turnarounds are slower, so threshold near ``velocity``.
        cruise = seg_speed[seg_speed >= 0.95 * config.velocity]
        assert cruise.size > 10
        rel = abs(cruise.mean() - config.velocity) / config.velocity
        assert rel < 0.001, (
            f"cruise speed {cruise.mean():.5f} deg/s vs configured {config.velocity} "
            f"(relative error {rel:.4%}); time grid is stretching velocities"
        )
