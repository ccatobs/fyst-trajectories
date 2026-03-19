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

        distances = np.sqrt((trajectory.az - center_az) ** 2 + (trajectory.el - center_el) ** 2)
        min_distance = distances.min()

        assert min_distance < 0.5

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
