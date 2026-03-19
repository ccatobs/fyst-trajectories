"""Tests for LinearMotionPattern."""

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories.exceptions import ElevationBoundsError
from fyst_trajectories.patterns import LinearMotionConfig, LinearMotionPattern

# Fixed start time for deterministic tests.
_START_TIME = Time("2026-03-15T04:00:00", scale="utc")


class TestLinearMotionPattern:
    """Tests for linear motion pattern."""

    def test_basic_linear_motion(self, site):
        """Test generating a basic linear motion trajectory."""
        start_time = _START_TIME
        config = LinearMotionConfig(
            timestep=0.1,
            az_start=100.0,
            el_start=45.0,
            az_velocity=0.5,
            el_velocity=0.1,
        )
        pattern = LinearMotionPattern(config)

        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        assert trajectory.n_points > 0
        assert trajectory.duration == pytest.approx(60.0, abs=0.2)
        assert trajectory.start_time == start_time
        assert trajectory.pattern_type == "linear"

    def test_linear_motion_positions(self, site):
        """Test that positions follow linear motion."""
        config = LinearMotionConfig(
            timestep=0.1,
            az_start=100.0,
            el_start=45.0,
            az_velocity=1.0,
            el_velocity=0.5,
        )
        pattern = LinearMotionPattern(config)

        trajectory = pattern.generate(site, duration=10.0, start_time=_START_TIME)

        assert trajectory.az[0] == pytest.approx(100.0, abs=0.01)
        assert trajectory.el[0] == pytest.approx(45.0, abs=0.01)

        assert trajectory.az[-1] == pytest.approx(110.0, abs=0.2)
        assert trajectory.el[-1] == pytest.approx(50.0, abs=0.2)

    def test_linear_motion_constant_velocity(self, site):
        """Test that velocities are constant throughout trajectory."""
        config = LinearMotionConfig(
            timestep=0.1,
            az_start=100.0,
            el_start=45.0,
            az_velocity=0.5,
            el_velocity=0.1,
        )
        pattern = LinearMotionPattern(config)

        trajectory = pattern.generate(site, duration=30.0, start_time=_START_TIME)

        np.testing.assert_array_almost_equal(
            trajectory.az_vel, np.full_like(trajectory.az_vel, 0.5), decimal=5
        )
        np.testing.assert_array_almost_equal(
            trajectory.el_vel, np.full_like(trajectory.el_vel, 0.1), decimal=5
        )

    def test_linear_motion_zero_velocity(self, site):
        """Test linear motion with zero velocity (stationary)."""
        config = LinearMotionConfig(
            timestep=0.1,
            az_start=120.0,
            el_start=50.0,
            az_velocity=0.0,
            el_velocity=0.0,
        )
        pattern = LinearMotionPattern(config)

        trajectory = pattern.generate(site, duration=10.0, start_time=_START_TIME)

        np.testing.assert_array_almost_equal(
            trajectory.az, np.full_like(trajectory.az, 120.0), decimal=5
        )
        np.testing.assert_array_almost_equal(
            trajectory.el, np.full_like(trajectory.el, 50.0), decimal=5
        )

    def test_linear_motion_negative_velocity(self, site):
        """Test linear motion with negative velocity."""
        config = LinearMotionConfig(
            timestep=0.1,
            az_start=150.0,
            el_start=60.0,
            az_velocity=-0.5,
            el_velocity=-0.1,
        )
        pattern = LinearMotionPattern(config)

        trajectory = pattern.generate(site, duration=20.0, start_time=_START_TIME)

        assert trajectory.az[-1] < trajectory.az[0]
        assert trajectory.el[-1] < trajectory.el[0]
        assert np.all(trajectory.az_vel < 0)
        assert np.all(trajectory.el_vel < 0)

    def test_linear_motion_custom_timestep(self, site):
        """Test linear motion with custom timestep."""
        config_fine = LinearMotionConfig(
            timestep=0.1,
            az_start=100.0,
            el_start=45.0,
            az_velocity=0.5,
            el_velocity=0.1,
        )
        config_coarse = LinearMotionConfig(
            timestep=1.0,
            az_start=100.0,
            el_start=45.0,
            az_velocity=0.5,
            el_velocity=0.1,
        )

        pattern_fine = LinearMotionPattern(config_fine)
        pattern_coarse = LinearMotionPattern(config_coarse)

        traj_fine = pattern_fine.generate(site, duration=10.0, start_time=_START_TIME)
        traj_coarse = pattern_coarse.generate(site, duration=10.0, start_time=_START_TIME)

        assert traj_fine.n_points > traj_coarse.n_points
        assert traj_fine.n_points > traj_coarse.n_points * 5

    def test_linear_motion_metadata_stored(self, site):
        """Test that linear motion stores pattern parameters correctly."""
        config = LinearMotionConfig(
            timestep=0.1,
            az_start=100.0,
            el_start=45.0,
            az_velocity=0.5,
            el_velocity=0.1,
        )
        pattern = LinearMotionPattern(config)

        trajectory = pattern.generate(site, duration=30.0, start_time=_START_TIME)

        assert trajectory.pattern_params is not None
        params = trajectory.pattern_params
        assert params["az_start"] == 100.0
        assert params["el_start"] == 45.0
        assert params["az_velocity"] == 0.5
        assert params["el_velocity"] == 0.1

    def test_linear_motion_validates_bounds(self, site):
        """Test that linear motion validates trajectory against telescope limits."""
        config = LinearMotionConfig(
            timestep=0.1,
            az_start=100.0,
            el_start=85.0,
            az_velocity=0.0,
            el_velocity=1.0,
        )
        pattern = LinearMotionPattern(config)

        with pytest.raises(ElevationBoundsError, match="elevation"):
            pattern.generate(site, duration=60.0, start_time=_START_TIME)

    def test_linear_motion_el_only(self, site):
        """Test linear motion in elevation only."""
        config = LinearMotionConfig(
            timestep=0.1,
            az_start=150.0,
            el_start=40.0,
            az_velocity=0.0,
            el_velocity=0.2,
        )
        pattern = LinearMotionPattern(config)

        trajectory = pattern.generate(site, duration=30.0, start_time=_START_TIME)

        np.testing.assert_array_almost_equal(
            trajectory.az, np.full_like(trajectory.az, 150.0), decimal=5
        )
        assert trajectory.el[-1] > trajectory.el[0]

    def test_linear_motion_az_only(self, site):
        """Test linear motion in azimuth only."""
        config = LinearMotionConfig(
            timestep=0.1,
            az_start=100.0,
            el_start=50.0,
            az_velocity=0.3,
            el_velocity=0.0,
        )
        pattern = LinearMotionPattern(config)

        trajectory = pattern.generate(site, duration=30.0, start_time=_START_TIME)

        np.testing.assert_array_almost_equal(
            trajectory.el, np.full_like(trajectory.el, 50.0), decimal=5
        )
        assert trajectory.az[-1] > trajectory.az[0]

    def test_linear_motion_start_time_optional(self, site):
        """Test that start_time=None produces a valid trajectory."""
        config = LinearMotionConfig(
            timestep=0.1,
            az_start=100.0,
            el_start=45.0,
            az_velocity=0.5,
            el_velocity=0.1,
        )
        pattern = LinearMotionPattern(config)

        trajectory = pattern.generate(site, duration=10.0, start_time=None)
        assert trajectory.start_time is None
        assert trajectory.n_points > 0
        assert trajectory.az[0] == pytest.approx(100.0, abs=0.01)
