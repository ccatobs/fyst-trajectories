"""Tests for SiderealTrackPattern."""

import pytest
from astropy.time import Time

from fyst_trajectories.patterns import SiderealTrackConfig, SiderealTrackPattern


class TestSiderealTrackPattern:
    """Fixed RA/Dec tracking: az/el drift, metadata, sample density, required start_time."""

    def test_basic_track(self, site):
        start_time = Time("2026-01-15T02:00:00", scale="utc")
        config = SiderealTrackConfig(timestep=0.1)
        pattern = SiderealTrackPattern(ra=83.633, dec=22.014, config=config)

        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        assert trajectory.start_time == start_time
        assert trajectory.pattern_type == "sidereal"
        assert trajectory.center_ra == 83.633
        assert trajectory.center_dec == 22.014

    def test_track_changes_with_time(self, site):
        """Az/El changes during tracking as the Earth rotates."""
        start_time = Time("2026-10-15T03:00:00", scale="utc")
        config = SiderealTrackConfig(timestep=0.1)
        pattern = SiderealTrackPattern(ra=0.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=600.0, start_time=start_time)

        az_change = trajectory.az[-1] - trajectory.az[0]
        assert abs(az_change) > 1.0

    def test_metadata(self, site):
        config = SiderealTrackConfig(timestep=0.1)
        pattern = SiderealTrackPattern(ra=83.633, dec=22.014, config=config)

        metadata = pattern.get_metadata()

        assert metadata.pattern_type == "sidereal"
        assert metadata.pattern_params == {}
        assert metadata.center_ra == 83.633
        assert metadata.center_dec == 22.014

    def test_with_config(self, site):
        start_time = Time("2026-10-15T03:00:00", scale="utc")
        config = SiderealTrackConfig(timestep=0.5)
        pattern = SiderealTrackPattern(ra=0.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=10.0, start_time=start_time)

        # round(10 / 0.5) + 1 samples, both endpoints included
        assert trajectory.n_points == 21

    def test_none_start_time_raises(self, site):
        config = SiderealTrackConfig(timestep=0.1)
        pattern = SiderealTrackPattern(ra=180.0, dec=-30.0, config=config)

        with pytest.raises(ValueError, match="start_time is required"):
            pattern.generate(site, duration=60.0, start_time=None)
