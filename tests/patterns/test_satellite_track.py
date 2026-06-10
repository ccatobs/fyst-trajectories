"""Tests for SatelliteTrackPattern (Titan point-tracking).

The fixture kernel ``tests/data/titan_excerpt.bsp`` is a self-contained
NAIF ``sat441`` excerpt covering 2026-06-01 .. 2026-10-01 (see
``tests/data/README.md``). All epochs here sit inside that window and
inside the IERS prediction window. Titan transits around midday UTC at
FYST on 2026-06-15, so the "Titan is up" epochs use the ~08:00-16:00 UTC
window and the below-horizon epoch uses 00:00 UTC.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time, TimeDelta

from fyst_trajectories import (
    FYST_ELEVATION,
    FYST_LATITUDE,
    FYST_LONGITUDE,
    Coordinates,
)
from fyst_trajectories.exceptions import TargetNotObservableError
from fyst_trajectories.patterns import (
    SatelliteTrackConfig,
    SatelliteTrackPattern,
    TrajectoryBuilder,
)
from fyst_trajectories.patterns.utils import normalize_azimuth

# tests/patterns/ -> parents[1] is tests/ -> data/titan_excerpt.bsp
TITAN_KERNEL = str((Path(__file__).parents[1] / "data" / "titan_excerpt.bsp").resolve())

# Epoch where Titan is comfortably up (el ~ 53 deg) at FYST, inside the
# excerpt + IERS windows.
TITAN_UP_TIME = Time("2026-06-15T10:00:00", scale="utc")

# Epoch where Titan is far below the horizon (el ~ -70 deg) at FYST.
TITAN_BELOW_TIME = Time("2026-06-15T00:00:00", scale="utc")


class TestSatelliteTrackPattern:
    """Tests for Titan point-tracking."""

    def test_titan_track_follows_titan(self, site):
        """Every sample equals get_body_altaz('titan', t_i) exactly.

        The pattern is a thin reuse of PlanetTrackPattern.generate, so each
        (az, el) must match the resolver call for a matching Coordinates with
        the same kernel -- the same call, byte-for-byte.
        """
        config = SatelliteTrackConfig(timestep=1.0, body="titan", satellite_kernel=TITAN_KERNEL)
        pattern = SatelliteTrackPattern(config=config)

        duration = 120.0
        trajectory = pattern.generate(site, duration=duration, start_time=TITAN_UP_TIME)

        coords = Coordinates(site, satellite_kernel=TITAN_KERNEL)
        n_points = int(round(duration / config.timestep)) + 1
        times = np.linspace(0, duration, n_points)
        obstimes = TITAN_UP_TIME + TimeDelta(times * u.s)
        az_ref, el_ref = coords.get_body_altaz("titan", obstimes)
        az_ref = normalize_azimuth(az_ref, site)

        np.testing.assert_array_equal(trajectory.az, az_ref)
        np.testing.assert_array_equal(trajectory.el, el_ref)

    def test_titan_track_velocities_finite_and_smooth(self, site):
        """Velocities are finite, slow, and continuous (no nonphysical jumps).

        Parity with the planet pattern: Titan's apparent motion is smooth, so
        per-axis velocities are finite, bounded well under any slew limit, and
        free of step discontinuities. The continuity (per-step velocity change)
        assertion gives the "smooth" claim teeth -- a glitch far below the
        velocity ceiling would still be caught.
        """
        config = SatelliteTrackConfig(timestep=1.0, body="titan", satellite_kernel=TITAN_KERNEL)
        pattern = SatelliteTrackPattern(config=config)

        trajectory = pattern.generate(site, duration=120.0, start_time=TITAN_UP_TIME)

        assert np.all(np.isfinite(trajectory.az_vel))
        assert np.all(np.isfinite(trajectory.el_vel))
        assert np.all(np.abs(trajectory.az_vel) < 1.0)
        assert np.all(np.abs(trajectory.el_vel) < 1.0)
        # Smoothness: the step-to-step velocity change stays tiny for a
        # sidereally-tracked body (a 1e-2 deg/s step bound flags any nonphysical
        # glitch while staying orders of magnitude above the real values).
        assert np.all(np.abs(np.diff(trajectory.az_vel)) < 1e-2)
        assert np.all(np.abs(np.diff(trajectory.el_vel)) < 1e-2)

    def test_titan_track_az_normalized(self, site):
        """All azimuths lie within the telescope range [-180, 360].

        The normalize_azimuth gotcha: celestial/planet/satellite patterns map
        astropy's [0, 360) into the telescope's [-180, 360].
        """
        config = SatelliteTrackConfig(timestep=1.0, body="titan", satellite_kernel=TITAN_KERNEL)
        pattern = SatelliteTrackPattern(config=config)

        trajectory = pattern.generate(site, duration=120.0, start_time=TITAN_UP_TIME)

        assert np.all(trajectory.az >= -180.0)
        assert np.all(trajectory.az <= 360.0)

    def test_titan_track_bounds_respected(self, site):
        """A time when Titan is below el_min raises TargetNotObservableError.

        At 2026-06-15T00:00 UTC Titan sits at el ~ -70 deg at FYST, deep below
        the FYST_EL_MIN limit, so bounds validation must reject the track.
        """
        config = SatelliteTrackConfig(timestep=1.0, body="titan", satellite_kernel=TITAN_KERNEL)
        pattern = SatelliteTrackPattern(config=config)

        with pytest.raises(TargetNotObservableError) as exc_info:
            pattern.generate(site, duration=60.0, start_time=TITAN_BELOW_TIME)

        assert exc_info.value.bounds_error is not None

    def test_titan_track_center_is_apparent(self, site):
        """The metadata center RA/Dec is Titan's apparent position.

        center_ra/center_dec feed celestial-frame consumers (map orientation
        via get_field_rotation, ECSV provenance). Push
        the STORED center through the INDEPENDENT radec_to_altaz transform and
        compare to Titan's Az/El at the midpoint (mirrors the planet
        test_planet_track_center_is_apparent). This is a real apparent-place check,
        not a tautology: a barycentric get_body_radec regression would blow the
        < 1 arcsec bound by degrees.
        """
        config = SatelliteTrackConfig(timestep=1.0, body="titan", satellite_kernel=TITAN_KERNEL)
        pattern = SatelliteTrackPattern(config=config)

        duration = 120.0
        trajectory = pattern.generate(site, duration=duration, start_time=TITAN_UP_TIME)

        midpoint_time = TITAN_UP_TIME + TimeDelta(duration / 2.0 * u.s)
        coords = Coordinates(site, satellite_kernel=TITAN_KERNEL)
        az_c, el_c = coords.radec_to_altaz(
            trajectory.center_ra, trajectory.center_dec, midpoint_time
        )
        az_m, el_m = coords.get_body_altaz("titan", midpoint_time)
        sep_arcsec = np.hypot((az_c - az_m) * np.cos(np.deg2rad(el_m)), el_c - el_m) * 3600.0
        assert sep_arcsec < 1.0

    def test_titan_track_requires_start_time(self, site):
        """start_time=None raises ValueError (like PlanetTrackPattern)."""
        config = SatelliteTrackConfig(timestep=1.0, body="titan", satellite_kernel=TITAN_KERNEL)
        pattern = SatelliteTrackPattern(config=config)

        with pytest.raises(ValueError, match="start_time is required"):
            pattern.generate(site, duration=60.0, start_time=None)

    def test_titan_track_metadata(self):
        """Metadata reports the satellite pattern type and target."""
        config = SatelliteTrackConfig(timestep=1.0, body="titan", satellite_kernel=TITAN_KERNEL)
        pattern = SatelliteTrackPattern(config=config)

        metadata = pattern.get_metadata()

        assert pattern.name == "satellite"
        assert metadata.pattern_type == "satellite"
        assert metadata.target_name == "titan"
        assert metadata.pattern_params["body"] == "titan"

    def test_titan_track_env_var(self, site, monkeypatch):
        """With FYST_SATELLITE_KERNEL set and satellite_kernel=None, it builds.

        Operational path: the kernel reaches Coordinates via the environment
        rather than the config field.
        """
        monkeypatch.setenv("FYST_SATELLITE_KERNEL", TITAN_KERNEL)
        config = SatelliteTrackConfig(timestep=1.0, body="titan", satellite_kernel=None)
        pattern = SatelliteTrackPattern(config=config)

        trajectory = pattern.generate(site, duration=60.0, start_time=TITAN_UP_TIME)

        assert trajectory.n_points > 0
        assert np.all(np.isfinite(trajectory.az))
        assert np.all(np.isfinite(trajectory.el))

    def test_titan_track_requires_kernel(self, site, monkeypatch):
        """No kernel and no env var raises a clear ValueError through generate()."""
        monkeypatch.delenv("FYST_SATELLITE_KERNEL", raising=False)
        config = SatelliteTrackConfig(timestep=1.0, body="titan", satellite_kernel=None)
        pattern = SatelliteTrackPattern(config=config)
        with pytest.raises(ValueError, match="requires a JPL satellite SPK kernel"):
            pattern.generate(site, duration=60.0, start_time=TITAN_UP_TIME)

    @pytest.mark.slow
    def test_titan_track_centroid_drift_under_beam(self, site):
        """Each sample is within 1.5 arcsec of an independent skyfield Titan.

        1.5 arcsec is ~1/10 of the ~15 arcsec Prime-Cam beam. The excerpt lacks
        the Jupiter barycentre, so skyfield's default relativistic deflectors
        would crash; deflection cancels on both sides, so deflectors=() is a
        like-for-like comparison.
        """
        pytest.importorskip("skyfield")
        from skyfield.api import load, wgs84
        from skyfield.iokit import load_file

        config = SatelliteTrackConfig(timestep=10.0, body="titan", satellite_kernel=TITAN_KERNEL)
        pattern = SatelliteTrackPattern(config=config)

        duration = 120.0
        trajectory = pattern.generate(site, duration=duration, start_time=TITAN_UP_TIME)

        ts = load.timescale()
        eph = load_file(TITAN_KERNEL)
        observer = eph[399] + wgs84.latlon(
            FYST_LATITUDE, FYST_LONGITUDE, elevation_m=FYST_ELEVATION
        )
        titan = eph[606]

        n_points = int(round(duration / config.timestep)) + 1
        times = np.linspace(0, duration, n_points)
        obstimes = TITAN_UP_TIME + TimeDelta(times * u.s)

        for i, obstime in enumerate(obstimes):
            sf_time = ts.from_astropy(obstime)
            apparent = observer.at(sf_time).observe(titan).apparent(deflectors=())
            alt, az, _ = apparent.altaz()
            d_az = (az.degrees - trajectory.az[i] % 360.0 + 180.0) % 360.0 - 180.0
            sep_arcsec = (
                np.hypot(d_az * np.cos(np.deg2rad(alt.degrees)), alt.degrees - trajectory.el[i])
                * 3600.0
            )
            assert sep_arcsec < 1.5, f"sample {i}: {sep_arcsec:.3f} arcsec from skyfield Titan"


class TestSatelliteTrackConfig:
    """Tests for SatelliteTrackConfig validation."""

    @pytest.mark.parametrize("body", ["mars", "jupiter", "pluto"])
    def test_satellite_config_rejects_non_satellite(self, body):
        """Planets and unknown names are not satellites and are rejected."""
        with pytest.raises(ValueError, match="Unknown satellite"):
            SatelliteTrackConfig(timestep=1.0, body=body)

    def test_satellite_config_lowercases_body(self):
        """Mixed-case satellite names are lower-cased and accepted."""
        config = SatelliteTrackConfig(timestep=1.0, body="TITAN")
        assert config.body == "titan"


class TestSatelliteTrackBuilder:
    """Tests for builder integration."""

    def test_builder_satellite_round_trip(self, site):
        """TrajectoryBuilder infers SatelliteTrackPattern from the config."""
        config = SatelliteTrackConfig(timestep=1.0, body="titan", satellite_kernel=TITAN_KERNEL)

        trajectory = (
            TrajectoryBuilder(site)
            .with_config(config)
            .duration(60.0)
            .starting_at(TITAN_UP_TIME)
            .build()
        )

        assert trajectory.pattern_type == "satellite"
        assert trajectory.pattern_params["body"] == "titan"
        assert trajectory.metadata.target_name == "titan"
        assert trajectory.n_points > 0


class TestSatelliteTrackPublicAPI:
    """Tests that the new symbols are additively exported."""

    def test_new_symbols_importable_and_in_all(self):
        """The three new public symbols import from the top level and are in __all__."""
        import fyst_trajectories as ft

        for name in ("SatelliteTrackConfig", "SatelliteTrackPattern", "SATELLITE_BODIES"):
            assert hasattr(ft, name), f"{name} not importable from fyst_trajectories"
            assert name in ft.__all__, f"{name} missing from __all__"

    def test_satellite_bodies_contains_titan(self):
        """SATELLITE_BODIES is the public tuple of resolvable satellites."""
        from fyst_trajectories import SATELLITE_BODIES

        assert "titan" in SATELLITE_BODIES


def test_no_pointing_warning_on_apply_detector_offset(site):
    """Detector offset on a Titan track must not warn.

    Parity with the planet pattern: apply_detector_offset is a horizon-frame
    projection using the mechanical rotation only (pa-in-horizon-frame fix),
    so it needs no celestial metadata and must never emit a PointingWarning.
    """
    from fyst_trajectories.exceptions import PointingWarning
    from fyst_trajectories.offsets import InstrumentOffset, apply_detector_offset

    config = SatelliteTrackConfig(timestep=1.0, body="titan", satellite_kernel=TITAN_KERNEL)
    pattern = SatelliteTrackPattern(config=config)
    trajectory = pattern.generate(site, duration=60.0, start_time=TITAN_UP_TIME)
    offset = InstrumentOffset(dx=5.0, dy=3.0, name="test-detector")

    with warnings.catch_warnings():
        warnings.simplefilter("error", PointingWarning)
        adjusted = apply_detector_offset(trajectory, offset, site)

    assert adjusted.n_points == trajectory.n_points
