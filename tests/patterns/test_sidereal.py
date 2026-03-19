"""Tests for SiderealTrackPattern."""

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories.patterns import SiderealTrackConfig, SiderealTrackPattern


class TestSiderealTrackPattern:
    """Tests for sidereal tracking pattern."""

    def test_basic_track(self, site):
        """Test generating a sidereal tracking trajectory."""
        start_time = Time("2026-01-15T02:00:00", scale="utc")
        config = SiderealTrackConfig(timestep=0.1)
        pattern = SiderealTrackPattern(ra=83.633, dec=22.014, config=config)

        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        assert trajectory.n_points > 0
        assert trajectory.start_time == start_time
        assert trajectory.pattern_type == "sidereal"
        assert trajectory.center_ra == 83.633
        assert trajectory.center_dec == 22.014

    @pytest.mark.slow
    def test_track_changes_with_time(self, site):
        """Test that Az/El changes during tracking (Earth rotation)."""
        start_time = Time("2026-10-15T03:00:00", scale="utc")
        config = SiderealTrackConfig(timestep=0.1)
        pattern = SiderealTrackPattern(ra=0.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=600.0, start_time=start_time)

        az_change = trajectory.az[-1] - trajectory.az[0]
        assert abs(az_change) > 1.0

    def test_metadata(self, site):
        """Test that metadata is correctly populated."""
        config = SiderealTrackConfig(timestep=0.1)
        pattern = SiderealTrackPattern(ra=83.633, dec=22.014, config=config)

        metadata = pattern.get_metadata()

        assert metadata.pattern_type == "sidereal"
        assert metadata.pattern_params == {}
        assert metadata.center_ra == 83.633
        assert metadata.center_dec == 22.014

    def test_with_config(self, site):
        """Test creating pattern with explicit config."""
        start_time = Time("2026-10-15T03:00:00", scale="utc")
        config = SiderealTrackConfig(timestep=0.5)
        pattern = SiderealTrackPattern(ra=0.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=10.0, start_time=start_time)

        # With 0.5s timestep over 10s, should have ~20 points
        assert trajectory.n_points == pytest.approx(20, abs=2)

    def test_finite_positions(self, site):
        """Test that positions are finite."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = SiderealTrackConfig(timestep=0.1)
        pattern = SiderealTrackPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        assert np.all(np.isfinite(trajectory.az))
        assert np.all(np.isfinite(trajectory.el))
        assert np.all(np.isfinite(trajectory.az_vel))
        assert np.all(np.isfinite(trajectory.el_vel))

    def test_none_start_time_raises(self, site):
        """Test that generate raises ValueError when start_time is None."""
        config = SiderealTrackConfig(timestep=0.1)
        pattern = SiderealTrackPattern(ra=180.0, dec=-30.0, config=config)

        with pytest.raises(ValueError, match="start_time is required"):
            pattern.generate(site, duration=60.0, start_time=None)
