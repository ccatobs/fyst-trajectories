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
    """Body tracking: metadata, the apparent centre RA/Dec, and no ra/dec constructor args."""

    def test_track_mars(self, site):
        start_time = Time("2026-01-15T14:00:00", scale="utc")
        config = PlanetTrackConfig(timestep=0.1, body="mars")
        pattern = PlanetTrackPattern(config=config)

        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        assert trajectory.start_time == start_time
        assert trajectory.pattern_type == "planet"
        assert trajectory.pattern_params["body"] == "mars"

    def test_planet_track_has_motion(self, site):
        """An hour of Moon tracking produces more than 10 degrees of total motion.

        Apparent Az/El motion is dominated by Earth's rotation, of order 15 degrees
        per hour, not by the Moon's own ~0.5 deg/hour drift against the stars. How
        that motion splits between azimuth and elevation depends on where the Moon
        is, so assert only that the combined motion exceeds 10 degrees (13 degrees at
        this epoch). A 10 s timestep samples the hour as finely as the check needs.
        """
        # Use a fixed time when Moon is observable from FYST
        start_time = Time("2026-01-15T10:00:00", scale="utc")
        config = PlanetTrackConfig(timestep=10.0, body="moon")
        pattern = PlanetTrackPattern(config=config)

        trajectory = pattern.generate(site, duration=3600.0, start_time=start_time)

        az_range = trajectory.az.max() - trajectory.az.min()
        el_range = trajectory.el.max() - trajectory.el.min()

        total_motion = np.sqrt(az_range**2 + el_range**2)
        assert total_motion > 10.0, (
            f"Expected significant motion, got az_range={az_range:.2f}, "
            f"el_range={el_range:.2f}, total={total_motion:.2f}"
        )

    def test_metadata(self):
        config = PlanetTrackConfig(timestep=0.1, body="jupiter")
        pattern = PlanetTrackPattern(config=config)

        metadata = pattern.get_metadata()

        assert metadata.pattern_type == "planet"
        assert metadata.pattern_params["body"] == "jupiter"
        assert metadata.target_name == "jupiter"

    def test_does_not_accept_ra_dec(self):
        sig = inspect.signature(PlanetTrackPattern.__init__)
        param_names = list(sig.parameters.keys())
        assert "ra" not in param_names
        assert "dec" not in param_names

    def test_get_metadata_without_args_has_no_radec(self):
        config = PlanetTrackConfig(timestep=0.1, body="jupiter")
        pattern = PlanetTrackPattern(config=config)

        metadata = pattern.get_metadata()

        assert metadata.center_ra is None
        assert metadata.center_dec is None

    def test_apply_detector_offset_no_warning(self, site):
        """``apply_detector_offset`` on a planet trajectory does not warn.

        apply_detector_offset is a horizon-frame projection using the
        mechanical rotation only, not the parallactic angle, so it needs no
        celestial metadata and must never emit a PointingWarning.
        """
        start_time = Time("2026-01-15T14:00:00", scale="utc")
        config = PlanetTrackConfig(timestep=0.1, body="mars")
        pattern = PlanetTrackPattern(config=config)

        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)
        offset = InstrumentOffset(dx=5.0, dy=3.0, name="test-detector")

        with warnings.catch_warnings():
            warnings.simplefilter("error", PointingWarning)
            adjusted = apply_detector_offset(trajectory, offset, site=site)

        assert adjusted.n_points == trajectory.n_points

    def test_planet_track_center_is_apparent(self, site):
        """The track's metadata center RA/Dec is the planet's apparent position.

        ``center_ra``/``center_dec`` feed celestial-frame consumers (map
        orientation via ``get_field_rotation``, ECSV provenance). A barycentric
        (solar-system-barycentre-relative) direction differs from the apparent
        place of a nearby body by degrees to tens of degrees, about 24 deg for
        Mars at this epoch, so the round trip is a real apparent-place check and
        not a tautology: guard that the reported center round-trips back to Mars'
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
        # The stored RA is also normalised to [0, 360).
        assert 0.0 <= trajectory.center_ra < 360.0


class TestPlanetTrackBodyCasing:
    """The config stores ``body`` lower-cased, like ``SatelliteTrackConfig``."""

    def test_config_lowercases_body(self):
        """A mixed-case body is stored lower-case and equals the lower-case config."""
        lower = PlanetTrackConfig(timestep=1.0, body="jupiter")
        for spelling in ("Jupiter", "JUPITER", "jUpItEr"):
            config = PlanetTrackConfig(timestep=1.0, body=spelling)
            assert config.body == "jupiter"
            assert config == lower
            assert hash(config) == hash(lower)
            assert repr(config) == "PlanetTrackConfig(timestep=1.0, body='jupiter')"

    def test_trajectory_metadata_is_lowercase(self, site):
        """The built trajectory records the lower-case body; its arrays do not change."""
        start_time = Time("2026-03-15T00:00:00", scale="utc")
        mixed = PlanetTrackPattern(PlanetTrackConfig(timestep=1.0, body="Jupiter"))
        lower = PlanetTrackPattern(PlanetTrackConfig(timestep=1.0, body="jupiter"))

        trajectory = mixed.generate(site, duration=60.0, start_time=start_time)
        reference = lower.generate(site, duration=60.0, start_time=start_time)

        assert trajectory.pattern_params["body"] == "jupiter"
        assert trajectory.metadata.target_name == "jupiter"
        np.testing.assert_array_equal(trajectory.az, reference.az)
        np.testing.assert_array_equal(trajectory.el, reference.el)

    def test_unknown_body_message_echoes_input(self):
        """The refusal names the body as the caller spelled it."""
        with pytest.raises(ValueError, match="Unknown body 'Pluto'"):
            PlanetTrackConfig(timestep=1.0, body="Pluto")
