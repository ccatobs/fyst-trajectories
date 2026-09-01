"""Tests for PlanetTrackPattern."""

import inspect
import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time, TimeDelta

from fyst_trajectories import Coordinates
from fyst_trajectories.exceptions import PointingWarning
from fyst_trajectories.offsets import InstrumentOffset, apply_detector_offset
from fyst_trajectories.patterns import PlanetTrackConfig, PlanetTrackPattern


class TestPlanetTrackPattern:
    """Tests for planet tracking pattern."""

    def test_track_mars(self, site):
        """Test tracking Mars."""
        start_time = Time("2026-01-15T14:00:00", scale="utc")
        config = PlanetTrackConfig(timestep=0.1, body="mars")
        pattern = PlanetTrackPattern(config=config)

        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        assert trajectory.n_points > 0
        assert trajectory.start_time == start_time
        assert trajectory.pattern_type == "planet"
        assert trajectory.pattern_params is not None
        assert trajectory.pattern_params["body"] == "mars"

    @pytest.mark.slow
    def test_planet_track_has_motion(self, site):
        """Test that planet position changes over time.

        The Moon moves approximately 0.5 degrees per minute in the sky,
        so in 1 hour there should significant motion. However, the distribution
        between azimuth and elevation depends on where the Moon is in the sky.
        So just verify there is meaningful motion in at least one axis.
        """
        # Use a fixed time when Moon is observable from FYST
        start_time = Time("2026-01-15T10:00:00", scale="utc")
        config = PlanetTrackConfig(timestep=0.1, body="moon")
        pattern = PlanetTrackPattern(config=config)

        trajectory = pattern.generate(site, duration=3600.0, start_time=start_time)

        az_range = trajectory.az.max() - trajectory.az.min()
        el_range = trajectory.el.max() - trajectory.el.min()

        # The Moon should move significantly in at least one coordinate over 1 hour
        # Combined motion should be well over 1 degree
        total_motion = np.sqrt(az_range**2 + el_range**2)
        assert total_motion > 1.0, (
            f"Expected significant motion, got az_range={az_range:.2f}, "
            f"el_range={el_range:.2f}, total={total_motion:.2f}"
        )

    def test_metadata(self):
        """Test that metadata is correctly populated."""
        config = PlanetTrackConfig(timestep=0.1, body="jupiter")
        pattern = PlanetTrackPattern(config=config)

        metadata = pattern.get_metadata()

        assert metadata.pattern_type == "planet"
        assert metadata.pattern_params["body"] == "jupiter"
        assert metadata.target_name == "jupiter"

    def test_does_not_accept_ra_dec(self):
        """Test that PlanetTrackPattern does not accept ra/dec parameters."""
        sig = inspect.signature(PlanetTrackPattern.__init__)
        param_names = list(sig.parameters.keys())
        assert "ra" not in param_names
        assert "dec" not in param_names

    def test_finite_positions(self, site):
        """Test that positions are finite."""
        start_time = Time("2026-01-15T14:00:00", scale="utc")
        config = PlanetTrackConfig(timestep=0.1, body="venus")
        pattern = PlanetTrackPattern(config=config)

        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        assert np.all(np.isfinite(trajectory.az))
        assert np.all(np.isfinite(trajectory.el))
        assert np.all(np.isfinite(trajectory.az_vel))
        assert np.all(np.isfinite(trajectory.el_vel))

    def test_metadata_has_radec_after_generate(self, site):
        """Test that generated trajectory metadata includes planet RA/Dec."""
        start_time = Time("2026-01-15T14:00:00", scale="utc")
        config = PlanetTrackConfig(timestep=0.1, body="mars")
        pattern = PlanetTrackPattern(config=config)

        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        assert trajectory.center_ra is not None
        assert trajectory.center_dec is not None
        # RA must be in [0, 360), Dec in [-90, 90]
        assert 0.0 <= trajectory.center_ra < 360.0
        assert -90.0 <= trajectory.center_dec <= 90.0

    def test_get_metadata_without_args_has_no_radec(self):
        """Test that get_metadata() without args returns None RA/Dec."""
        config = PlanetTrackConfig(timestep=0.1, body="jupiter")
        pattern = PlanetTrackPattern(config=config)

        metadata = pattern.get_metadata()

        assert metadata.center_ra is None
        assert metadata.center_dec is None

    def test_apply_detector_offset_no_warning(self, site):
        """Test that apply_detector_offset on a planet trajectory does not warn.

        apply_detector_offset is a horizon-frame projection using the
        mechanical rotation only (pa-in-horizon-frame fix), so it needs no
        celestial metadata and must never emit a PointingWarning.
        """
        start_time = Time("2026-01-15T14:00:00", scale="utc")
        config = PlanetTrackConfig(timestep=0.1, body="mars")
        pattern = PlanetTrackPattern(config=config)

        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)
        offset = InstrumentOffset(dx=5.0, dy=3.0, name="test-detector")

        with warnings.catch_warnings():
            warnings.simplefilter("error", PointingWarning)
            adjusted = apply_detector_offset(trajectory, offset, site)

        assert adjusted.n_points == trajectory.n_points

    def test_planet_track_center_is_apparent(self, site):
        """The track's metadata center RA/Dec is the planet's apparent position.

        ``center_ra``/``center_dec`` feed celestial-frame consumers (map
        orientation via ``get_field_rotation``, ECSV provenance). The
        barycentric ``get_body_radec`` bug skewed this (up to ~17 deg
        for Mars), so guard that the reported center round-trips back to Mars'
        Az/El at the track midpoint.
        """
        start_time = Time("2026-06-15T14:00:00", scale="utc")
        duration = 60.0
        config = PlanetTrackConfig(timestep=0.1, body="mars")
        pattern = PlanetTrackPattern(config=config)

        trajectory = pattern.generate(site, duration=duration, start_time=start_time)

        midpoint_time = start_time + TimeDelta(duration / 2.0 * u.s)
        coords = Coordinates(site)
        az_c, el_c = coords.radec_to_altaz(
            trajectory.center_ra, trajectory.center_dec, midpoint_time
        )
        az_m, el_m = coords.get_body_altaz("mars", midpoint_time)
        sep_arcsec = np.hypot((az_c - az_m) * np.cos(np.deg2rad(el_m)), el_c - el_m) * 3600.0
        assert sep_arcsec < 1.0
