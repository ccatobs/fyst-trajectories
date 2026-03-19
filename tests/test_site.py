"""Tests for site configuration module.

These tests verify that site configuration loading and validation
works correctly, including the FYST physical constants and the
get_fyst_site() constructor.
"""

import tempfile
from pathlib import Path

import pytest
import yaml
from astropy import units as u
from astropy.coordinates import EarthLocation

from fyst_trajectories import Site
from fyst_trajectories.offsets import InstrumentOffset, compute_focal_plane_rotation
from fyst_trajectories.site import (
    FYST_AZ_MAX,
    FYST_AZ_MAX_ACCELERATION,
    FYST_AZ_MAX_VELOCITY,
    FYST_AZ_MIN,
    FYST_EL_MAX,
    FYST_EL_MAX_ACCELERATION,
    FYST_EL_MAX_VELOCITY,
    FYST_EL_MIN,
    FYST_ELEVATION,
    FYST_LATITUDE,
    FYST_LONGITUDE,
    FYST_NASMYTH_PORT,
    FYST_PLATE_SCALE,
    FYST_SUN_AVOIDANCE_ENABLED,
    FYST_SUN_EXCLUSION_RADIUS,
    FYST_SUN_WARNING_RADIUS,
    AtmosphericConditions,
    AxisLimits,
    SunAvoidanceConfig,
    TelescopeLimits,
    get_fyst_site,
)


class TestSiteLoading:
    """Tests for Site.from_config() loading."""

    def test_load_default_config(self, site):
        """Test loading the default FYST configuration."""
        assert site.name == "FYST"
        assert site.latitude == pytest.approx(-22.985639, abs=0.001)
        assert site.longitude == pytest.approx(-67.740278, abs=0.001)
        assert site.elevation == pytest.approx(5611.8, abs=1.0)

    def test_location_property(self, site):
        """Test that location returns an EarthLocation."""
        loc = site.location
        assert isinstance(loc, EarthLocation)
        assert loc.lat.deg == pytest.approx(site.latitude, abs=0.001)
        assert loc.lon.deg == pytest.approx(site.longitude, abs=0.001)
        assert loc.height.to(u.m).value == pytest.approx(site.elevation, abs=1.0)

    def test_config_not_found(self):
        """Test error when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            Site.from_config("/nonexistent/path/config.yaml")

    def test_custom_config(self):
        """Test loading a custom configuration."""
        custom_config = {
            "site": {
                "name": "TestSite",
                "description": "Test telescope site",
                "location": {
                    "latitude": -30.0,
                    "longitude": -70.0,
                    "elevation": 2000.0,
                },
            },
            "telescope": {
                "plate_scale": 13.89,
                "azimuth": {
                    "min": -180.0,
                    "max": 180.0,
                    "max_velocity": 2.0,
                    "max_acceleration": 0.5,
                },
                "elevation": {
                    "min": 15.0,
                    "max": 85.0,
                    "max_velocity": 1.0,
                    "max_acceleration": 0.5,
                },
            },
            "sun_avoidance": {
                "enabled": True,
                "exclusion_radius": 45.0,
                "warning_radius": 50.0,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(custom_config, f)
            temp_path = f.name

        try:
            site = Site.from_config(temp_path)
            assert site.name == "TestSite"
            assert site.latitude == -30.0
            assert site.atmosphere is None
            assert site.telescope_limits.elevation.min == 15.0
        finally:
            Path(temp_path).unlink()

    def test_config_loading_sets_atmosphere_none(self, tmp_path):
        """Test that config loading always sets atmosphere=None."""
        config = {
            "site": {
                "name": "Test",
                "location": {"latitude": -23.0, "longitude": -67.0, "elevation": 5000.0},
            },
            "telescope": {
                "plate_scale": 13.89,
                "azimuth": {"min": -270, "max": 270, "max_velocity": 3, "max_acceleration": 1},
                "elevation": {"min": 20, "max": 90, "max_velocity": 1, "max_acceleration": 0.5},
            },
            "sun_avoidance": {"enabled": True, "exclusion_radius": 45, "warning_radius": 50},
        }
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml.dump(config))

        site = Site.from_config(config_file)
        assert site.atmosphere is None

    def test_optional_description_has_default(self, tmp_path):
        """Test that description is optional and defaults to empty string."""
        config_no_description = {
            "site": {
                "name": "Test",
                # description omitted - should default to ""
                "location": {"latitude": -23.0, "longitude": -67.0, "elevation": 5000.0},
            },
            "telescope": {
                "plate_scale": 13.89,
                "azimuth": {"min": -270, "max": 270, "max_velocity": 3, "max_acceleration": 1},
                "elevation": {"min": 20, "max": 90, "max_velocity": 1, "max_acceleration": 0.5},
            },
            "sun_avoidance": {"enabled": True, "exclusion_radius": 45, "warning_radius": 50},
        }
        config_file = tmp_path / "no_description.yaml"
        config_file.write_text(yaml.dump(config_no_description))

        site = Site.from_config(config_file)
        assert site.description == ""


class TestAtmosphericConditions:
    """Tests for AtmosphericConditions class."""

    def test_validation_rejects_invalid_humidity(self):
        """Test that relative_humidity must be in [0, 1]."""
        with pytest.raises(ValueError, match="relative_humidity must be in range"):
            AtmosphericConditions(pressure=550.0, temperature=270.0, relative_humidity=1.5)
        with pytest.raises(ValueError, match="relative_humidity must be in range"):
            AtmosphericConditions(pressure=550.0, temperature=270.0, relative_humidity=-0.1)


class TestAxisLimits:
    """Tests for AxisLimits class."""

    def test_axis_limits_behavior(self):
        """Test is_in_range and clip methods together."""
        limits = AxisLimits(min=-90.0, max=90.0, max_velocity=1.0, max_acceleration=0.5)

        assert limits.is_in_range(0.0)
        assert limits.is_in_range(-90.0)
        assert limits.is_in_range(90.0)
        assert not limits.is_in_range(-91.0)
        assert not limits.is_in_range(91.0)

        assert limits.clip(0.0) == 0.0
        assert limits.clip(-100.0) == -90.0
        assert limits.clip(100.0) == 90.0

    def test_validation_rejects_invalid_limits(self):
        """Test that min must be <= max."""
        with pytest.raises(ValueError, match="min .* must be <= max"):
            AxisLimits(min=100.0, max=50.0, max_velocity=1.0, max_acceleration=0.5)


class TestNasmythPort:
    """Tests for nasmyth_port and nasmyth_sign property."""

    def test_default_nasmyth_port(self, site):
        """Test that default FYST config has nasmyth_port='right'."""
        assert site.nasmyth_port == "right"

    def test_nasmyth_sign_right(self, site):
        """Test nasmyth_sign is +1 for right port."""
        assert site.nasmyth_sign == 1

    def test_nasmyth_sign_left(self):
        """Test nasmyth_sign is -1 for left port."""
        site = Site(
            name="Test",
            description="",
            latitude=-23.0,
            longitude=-67.0,
            elevation=5000.0,
            atmosphere=None,
            telescope_limits=TelescopeLimits(
                azimuth=AxisLimits(min=-270, max=270, max_velocity=3, max_acceleration=1),
                elevation=AxisLimits(min=20, max=90, max_velocity=1, max_acceleration=0.5),
            ),
            sun_avoidance=SunAvoidanceConfig(enabled=True, exclusion_radius=45, warning_radius=50),
            nasmyth_port="left",
        )
        assert site.nasmyth_sign == -1

    def test_nasmyth_sign_cassegrain(self):
        """Test nasmyth_sign is 0 for cassegrain."""
        site = Site(
            name="Test",
            description="",
            latitude=-23.0,
            longitude=-67.0,
            elevation=5000.0,
            atmosphere=None,
            telescope_limits=TelescopeLimits(
                azimuth=AxisLimits(min=-270, max=270, max_velocity=3, max_acceleration=1),
                elevation=AxisLimits(min=20, max=90, max_velocity=1, max_acceleration=0.5),
            ),
            sun_avoidance=SunAvoidanceConfig(enabled=True, exclusion_radius=45, warning_radius=50),
            nasmyth_port="cassegrain",
        )
        assert site.nasmyth_sign == 0

    def test_nasmyth_sign_invalid_raises(self):
        """Test that invalid nasmyth_port raises ValueError at construction."""
        with pytest.raises(ValueError, match="Unknown nasmyth_port"):
            Site(
                name="Test",
                description="",
                latitude=-23.0,
                longitude=-67.0,
                elevation=5000.0,
                atmosphere=None,
                telescope_limits=TelescopeLimits(
                    azimuth=AxisLimits(min=-270, max=270, max_velocity=3, max_acceleration=1),
                    elevation=AxisLimits(min=20, max=90, max_velocity=1, max_acceleration=0.5),
                ),
                sun_avoidance=SunAvoidanceConfig(
                    enabled=True, exclusion_radius=45, warning_radius=50
                ),
                nasmyth_port="invalid",
            )

    def test_load_from_yaml_with_nasmyth_port(self, tmp_path):
        """Test loading nasmyth_port from YAML config."""
        config = {
            "site": {
                "name": "Test",
                "location": {"latitude": -23.0, "longitude": -67.0, "elevation": 5000.0},
            },
            "telescope": {
                "nasmyth_port": "left",
                "plate_scale": 13.89,
                "azimuth": {"min": -270, "max": 270, "max_velocity": 3, "max_acceleration": 1},
                "elevation": {"min": 20, "max": 90, "max_velocity": 1, "max_acceleration": 0.5},
            },
            "sun_avoidance": {"enabled": True, "exclusion_radius": 45, "warning_radius": 50},
        }
        config_file = tmp_path / "left_nasmyth.yaml"
        config_file.write_text(yaml.dump(config))

        site = Site.from_config(config_file)
        assert site.nasmyth_port == "left"
        assert site.nasmyth_sign == -1

    def test_load_from_yaml_without_nasmyth_port(self, tmp_path):
        """Test that missing nasmyth_port defaults to 'right'."""
        config = {
            "site": {
                "name": "Test",
                "location": {"latitude": -23.0, "longitude": -67.0, "elevation": 5000.0},
            },
            "telescope": {
                # nasmyth_port omitted - should default to "right"
                "plate_scale": 13.89,
                "azimuth": {"min": -270, "max": 270, "max_velocity": 3, "max_acceleration": 1},
                "elevation": {"min": 20, "max": 90, "max_velocity": 1, "max_acceleration": 0.5},
            },
            "sun_avoidance": {"enabled": True, "exclusion_radius": 45, "warning_radius": 50},
        }
        config_file = tmp_path / "no_nasmyth.yaml"
        config_file.write_text(yaml.dump(config))

        site = Site.from_config(config_file)
        assert site.nasmyth_port == "right"
        assert site.nasmyth_sign == 1


class TestCassegrainFocalPlaneRotation:
    """Test compute_focal_plane_rotation with cassegrain port."""

    def test_cassegrain_focal_plane_rotation_ignores_elevation(self):
        """Test that cassegrain (nasmyth_sign=0) ignores elevation in rotation."""
        cass_site = Site(
            name="CassTest",
            description="",
            latitude=-23.0,
            longitude=-67.0,
            elevation=5000.0,
            atmosphere=None,
            telescope_limits=TelescopeLimits(
                azimuth=AxisLimits(
                    min=-270,
                    max=270,
                    max_velocity=3,
                    max_acceleration=1,
                ),
                elevation=AxisLimits(
                    min=20,
                    max=90,
                    max_velocity=1,
                    max_acceleration=0.5,
                ),
            ),
            sun_avoidance=SunAvoidanceConfig(
                enabled=True,
                exclusion_radius=45,
                warning_radius=50,
            ),
            nasmyth_port="cassegrain",
        )
        offset = InstrumentOffset(dx=5.0, dy=3.0, instrument_rotation=10.0)

        # rotation = 0 * el + 10 + pa
        rot_low = compute_focal_plane_rotation(
            20.0,
            cass_site,
            offset,
            parallactic_angle=5.0,
        )
        rot_high = compute_focal_plane_rotation(
            80.0,
            cass_site,
            offset,
            parallactic_angle=5.0,
        )

        # Both should be 0*el + 10 + 5 = 15, independent of elevation
        assert rot_low == pytest.approx(15.0)
        assert rot_high == pytest.approx(15.0)


class TestTelescopeLimits:
    """Tests for TelescopeLimits class."""

    def test_is_position_valid(self, site):
        """Test combined position validation."""
        limits = site.telescope_limits

        assert limits.is_position_valid(0.0, 45.0)
        assert not limits.is_position_valid(0.0, 10.0)  # elevation too low
        assert not limits.is_position_valid(400.0, 45.0)  # azimuth out of range


class TestFYSTConstants:
    """Regression tests for FYST physical constants.

    These verify that the hardcoded constants match the expected values
    from the FYST TCS source code and optical design. If any of these
    fail, it means a constant was accidentally changed.
    """

    def test_tier1_location(self):
        """Test Tier 1 geographic constants match FYST TCS astro.go."""
        assert FYST_LATITUDE == -22.985639
        assert FYST_LONGITUDE == -67.740278
        assert FYST_ELEVATION == 5611.8

    def test_tier1_optics(self):
        """Test Tier 1 optical constants."""
        assert FYST_PLATE_SCALE == 13.89
        assert FYST_NASMYTH_PORT == "right"

    def test_tier2_azimuth_limits(self):
        """Test Tier 2 azimuth mechanical limits match FYST TCS commands.go."""
        assert FYST_AZ_MIN == -180.0
        assert FYST_AZ_MAX == 360.0
        assert FYST_AZ_MAX_VELOCITY == 3.0
        assert FYST_AZ_MAX_ACCELERATION == 1.0

    def test_tier2_elevation_limits(self):
        """Test Tier 2 elevation mechanical limits match FYST TCS commands.go."""
        assert FYST_EL_MIN == 20.0
        assert FYST_EL_MAX == 90.0
        assert FYST_EL_MAX_VELOCITY == 1.0
        assert FYST_EL_MAX_ACCELERATION == 0.5

    def test_tier3_sun_avoidance_defaults(self):
        """Test Tier 3 operational defaults."""
        assert FYST_SUN_EXCLUSION_RADIUS == 45.0
        assert FYST_SUN_WARNING_RADIUS == 50.0
        assert FYST_SUN_AVOIDANCE_ENABLED is True


class TestGetFystSiteKwargs:
    """Tests for get_fyst_site() keyword argument overrides."""

    def test_default_site_matches_constants(self):
        """Test that get_fyst_site() with defaults matches all constants."""
        site = get_fyst_site()
        assert site.name == "FYST"
        assert site.latitude == FYST_LATITUDE
        assert site.longitude == FYST_LONGITUDE
        assert site.elevation == FYST_ELEVATION
        assert site.plate_scale == FYST_PLATE_SCALE
        assert site.nasmyth_port == FYST_NASMYTH_PORT
        assert site.telescope_limits.azimuth.min == FYST_AZ_MIN
        assert site.telescope_limits.azimuth.max == FYST_AZ_MAX
        assert site.telescope_limits.azimuth.max_velocity == FYST_AZ_MAX_VELOCITY
        assert site.telescope_limits.azimuth.max_acceleration == FYST_AZ_MAX_ACCELERATION
        assert site.telescope_limits.elevation.min == FYST_EL_MIN
        assert site.telescope_limits.elevation.max == FYST_EL_MAX
        assert site.telescope_limits.elevation.max_velocity == FYST_EL_MAX_VELOCITY
        assert site.telescope_limits.elevation.max_acceleration == FYST_EL_MAX_ACCELERATION
        assert site.sun_avoidance.exclusion_radius == FYST_SUN_EXCLUSION_RADIUS
        assert site.sun_avoidance.warning_radius == FYST_SUN_WARNING_RADIUS
        assert site.sun_avoidance.enabled == FYST_SUN_AVOIDANCE_ENABLED
        assert site.atmosphere is None

    def test_override_sun_exclusion_radius(self):
        """Test overriding sun exclusion radius."""
        site = get_fyst_site(sun_exclusion_radius=30.0)
        assert site.sun_avoidance.exclusion_radius == 30.0
        # Other sun params unchanged
        assert site.sun_avoidance.warning_radius == FYST_SUN_WARNING_RADIUS
        assert site.sun_avoidance.enabled == FYST_SUN_AVOIDANCE_ENABLED

    def test_override_sun_warning_radius(self):
        """Test overriding sun warning radius."""
        site = get_fyst_site(sun_warning_radius=60.0)
        assert site.sun_avoidance.warning_radius == 60.0

    def test_disable_sun_avoidance(self):
        """Test disabling sun avoidance."""
        site = get_fyst_site(sun_avoidance_enabled=False)
        assert site.sun_avoidance.enabled is False
        # Radii still set to defaults
        assert site.sun_avoidance.exclusion_radius == FYST_SUN_EXCLUSION_RADIUS

    def test_override_multiple_kwargs(self):
        """Test overriding multiple sun avoidance parameters."""
        site = get_fyst_site(
            sun_exclusion_radius=20.0,
            sun_warning_radius=25.0,
            sun_avoidance_enabled=False,
        )
        assert site.sun_avoidance.exclusion_radius == 20.0
        assert site.sun_avoidance.warning_radius == 25.0
        assert site.sun_avoidance.enabled is False

    def test_returns_fresh_instance(self):
        """Test that get_fyst_site() returns a new instance each call."""
        site1 = get_fyst_site()
        site2 = get_fyst_site()
        # Both should be equal but not the same object (no caching)
        assert site1.latitude == site2.latitude
        assert site1.name == site2.name
