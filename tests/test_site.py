"""Tests for site configuration module.

These tests verify that site configuration loading and validation
works correctly, including the FYST physical constants and the
get_fyst_site() constructor.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy import units as u
from astropy.coordinates import EarthLocation

from fyst_trajectories import Site
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
    """Tests for the shipped FYST site values and the ``Site.from_config()`` loader."""

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
            assert site.telescope_limits.elevation.min == 15.0
        finally:
            Path(temp_path).unlink()

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

    @pytest.mark.parametrize("case", ["empty-file", "scalar-section", "null-nasmyth-port"])
    def test_malformed_config_raises_value_error(self, tmp_path, case):
        """A document of the wrong shape is reported as the invalid config it is."""
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
        config_file = tmp_path / f"{case}.yaml"
        if case == "empty-file":
            config_file.write_text("")
        else:
            if case == "scalar-section":
                config["site"] = 3
            else:
                config["telescope"]["nasmyth_port"] = None
            config_file.write_text(yaml.dump(config))

        with pytest.raises(ValueError, match=f"Config '{case}.yaml' is malformed"):
            Site.from_config(config_file)

    def test_zero_velocity_config_raises_value_error(self, tmp_path):
        """A ``max_velocity: 0`` typo is refused rather than pricing every slew as free."""
        config = {
            "site": {
                "name": "Test",
                "location": {"latitude": -23.0, "longitude": -67.0, "elevation": 5000.0},
            },
            "telescope": {
                "plate_scale": 13.89,
                "azimuth": {"min": -270, "max": 270, "max_velocity": 0, "max_acceleration": 1},
                "elevation": {"min": 20, "max": 90, "max_velocity": 1, "max_acceleration": 0.5},
            },
            "sun_avoidance": {"enabled": True, "exclusion_radius": 45, "warning_radius": 50},
        }
        config_file = tmp_path / "zero_velocity.yaml"
        config_file.write_text(yaml.dump(config))

        with pytest.raises(ValueError, match="max_velocity must be a finite, positive number"):
            Site.from_config(config_file)


class TestAtmosphericConditions:
    """Tests for AtmosphericConditions class."""

    def test_validation_rejects_invalid_humidity(self):
        """Test that relative_humidity must be in [0, 1]."""
        with pytest.raises(ValueError, match="relative_humidity must be in range"):
            AtmosphericConditions(pressure=550.0, temperature=270.0, relative_humidity=1.5)
        with pytest.raises(ValueError, match="relative_humidity must be in range"):
            AtmosphericConditions(pressure=550.0, temperature=270.0, relative_humidity=-0.1)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            pytest.param(dict(temperature=-5.0), "temperature must be >= 0 K", id="celsius"),
            pytest.param(dict(temperature=float("nan")), "temperature must be >= 0 K", id="nan-t"),
            pytest.param(dict(pressure=float("nan")), "pressure must be >= 0 hPa", id="nan-p"),
        ],
    )
    def test_validation_rejects_negative_or_nan_pressure_and_temperature(self, kwargs, match):
        """A Celsius value passed as Kelvin, or a NaN, is refused rather than refracted."""
        values = dict(pressure=550.0, temperature=270.0, relative_humidity=0.2)
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            AtmosphericConditions(**values)

    def test_no_refraction_zero_kelvin_still_builds(self):
        """The vacuum synonym's 0 K temperature stays legal."""
        assert AtmosphericConditions.no_refraction().temperature == 0.0

    def test_for_fyst_defaults(self):
        """``for_fyst()`` returns a Cerro-Chajnantor profile with submm wavelength."""
        atmo = AtmosphericConditions.for_fyst()
        assert atmo.pressure == pytest.approx(500.0)
        assert atmo.temperature == pytest.approx(265.0)
        assert atmo.relative_humidity == pytest.approx(0.10)
        # 200 um forces astropy's radio-IR refraction model.
        assert atmo.obswl == pytest.approx(200.0)
        assert atmo.obswl_quantity is not None

    def test_for_fyst_overrides(self):
        """``for_fyst()`` accepts overrides for current weather."""
        atmo = AtmosphericConditions.for_fyst(
            pressure=520.0, temperature=270.0, relative_humidity=0.2, obswl=350.0
        )
        assert atmo.pressure == pytest.approx(520.0)
        assert atmo.temperature == pytest.approx(270.0)
        assert atmo.relative_humidity == pytest.approx(0.20)
        assert atmo.obswl == pytest.approx(350.0)


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

    @pytest.mark.parametrize("field", ["min", "max"])
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_bound(self, field, value):
        """A NaN bound disables every range comparison; an infinite one breaks the arithmetic."""
        kwargs = {"min": -90.0, "max": 90.0, "max_velocity": 1.0, "max_acceleration": 0.5}
        kwargs[field] = value
        with pytest.raises(ValueError, match=f"^{field} must be a finite number"):
            AxisLimits(**kwargs)

    @pytest.mark.parametrize("field", ["max_velocity", "max_acceleration"])
    @pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_rate_that_is_not_finite_and_positive(self, field, value):
        """A zero or negative rate priced every slew as free; a NaN one priced it as NaN."""
        kwargs = {"min": -90.0, "max": 90.0, "max_velocity": 1.0, "max_acceleration": 0.5}
        kwargs[field] = value
        with pytest.raises(ValueError, match=f"^{field} must be a finite, positive number"):
            AxisLimits(**kwargs)

    def test_clip_returns_a_builtin_float(self):
        """The result is a plain float, not a numpy scalar leaking into caller arithmetic."""
        limits = AxisLimits(min=-90.0, max=90.0, max_velocity=1.0, max_acceleration=0.5)
        clipped = limits.clip(100.0)
        assert clipped == 90.0
        assert type(clipped) is float
        assert not isinstance(clipped, np.floating)


class TestSunAvoidanceConfig:
    """SunAvoidanceConfig enforces 0 <= exclusion_radius < warning_radius when enabled."""

    def test_valid_config_constructs(self):
        cfg = SunAvoidanceConfig(enabled=True, exclusion_radius=45.0, warning_radius=50.0)
        assert cfg.exclusion_radius == 45.0
        assert cfg.warning_radius == 50.0

    def test_rejects_warning_below_exclusion(self):
        # Inverted radii make the warning band [exclusion, warning) empty.
        with pytest.raises(ValueError, match="warning_radius"):
            SunAvoidanceConfig(enabled=True, exclusion_radius=50.0, warning_radius=40.0)

    def test_rejects_negative_exclusion(self):
        # Negative exclusion_radius silently disables avoidance while enabled=True.
        with pytest.raises(ValueError, match="exclusion_radius"):
            SunAvoidanceConfig(enabled=True, exclusion_radius=-5.0, warning_radius=50.0)

    def test_equal_radii_rejected(self):
        # warning == exclusion leaves an empty warning band (every
        # warning-worthy pointing is already excluded), so warning_radius
        # must be strictly greater than exclusion_radius.
        with pytest.raises(ValueError, match="strictly greater"):
            SunAvoidanceConfig(enabled=True, exclusion_radius=45.0, warning_radius=45.0)

    def test_disabled_config_skips_validation(self):
        # When disabled the radii are inert; misordered/negative values are allowed.
        cfg = SunAvoidanceConfig(enabled=False, exclusion_radius=-5.0, warning_radius=0.0)
        assert cfg.enabled is False

    @pytest.mark.parametrize("field", ["exclusion_radius", "warning_radius"])
    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_rejects_non_finite_radius_when_enabled(self, field, value):
        # Every ``separation <= radius`` test is False against a NaN radius,
        # so such a config passed every Sun warning and observability check.
        kwargs = {"exclusion_radius": 45.0, "warning_radius": 50.0}
        kwargs[field] = value
        with pytest.raises(ValueError, match=f"^{field} must be a finite number"):
            SunAvoidanceConfig(enabled=True, **kwargs)

    def test_disabled_config_accepts_non_finite_radii(self):
        # The finite check sits on the enabled branch, like the other radius checks.
        cfg = SunAvoidanceConfig(
            enabled=False, exclusion_radius=float("nan"), warning_radius=float("inf")
        )
        assert cfg.enabled is False


class TestNasmythPort:
    """Tests for nasmyth_port and nasmyth_sign property."""

    def test_nasmyth_sign_right(self, site):
        """Test nasmyth_sign is +1 for right port."""
        assert site.nasmyth_sign == 1

    def test_nasmyth_sign_cassegrain(self):
        """Test nasmyth_sign is 0 for cassegrain."""
        site = Site(
            name="Test",
            description="",
            latitude=-23.0,
            longitude=-67.0,
            elevation=5000.0,
            telescope_limits=TelescopeLimits(
                azimuth=AxisLimits(min=-270, max=270, max_velocity=3, max_acceleration=1),
                elevation=AxisLimits(min=20, max=90, max_velocity=1, max_acceleration=0.5),
            ),
            sun_avoidance=SunAvoidanceConfig(enabled=True, exclusion_radius=45, warning_radius=50),
            nasmyth_port="cassegrain",
        )
        assert site.nasmyth_sign == 0

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

    These pin the hardcoded constants. The Tier 1 geographic values trace to
    the FYST TCS source code; the Tier 2 kinematic values either match the
    TCS values or sit below them, as each test states. The Nasmyth port, the
    plate scale, the velocity and acceleration limits and the Sun radii are
    commissioning defaults listed under "Pending instrument verification" in
    ``docs/index.rst``: their pins move when the FYST team confirms or
    changes them. Any other failure means a constant was changed by accident.
    """

    def test_tier1_location(self):
        """Test Tier 1 geographic constants match FYST TCS astro.go."""
        assert FYST_LATITUDE == -22.985639
        assert FYST_LONGITUDE == -67.740278
        assert FYST_ELEVATION == 5611.8

    def test_tier1_optics(self):
        """Test Tier 1 optical constants (both pending instrument verification)."""
        assert FYST_PLATE_SCALE == 13.89
        assert FYST_NASMYTH_PORT == "right"

    def test_tier2_azimuth_limits(self):
        """Test Tier 2 azimuth limits.

        The range and velocity match FYST TCS commands.go; the acceleration
        is the conservative operational limit (TCS hardware: 6.0 deg/s^2).
        """
        assert FYST_AZ_MIN == -180.0
        assert FYST_AZ_MAX == 360.0
        assert FYST_AZ_MAX_VELOCITY == 3.0
        assert FYST_AZ_MAX_ACCELERATION == 1.5

    def test_tier2_elevation_limits(self):
        """Test Tier 2 elevation limits.

        All four are conservative operational limits, not TCS values: the
        TCS accepts el from -90 to 180 deg with hardware maxima of 1.5 deg/s
        and 1.5 deg/s^2.
        """
        assert FYST_EL_MIN == 20.0
        assert FYST_EL_MAX == 90.0
        assert FYST_EL_MAX_VELOCITY == 1.0
        assert FYST_EL_MAX_ACCELERATION == 0.75

    def test_tier3_sun_avoidance_defaults(self):
        """Test Tier 3 operational defaults.

        45 is the Prime-Cam observing-policy baseline, the circle the survey
        planner schedules against; 50 keeps 5 deg of warning margin above it.
        The stricter directional CAD zone is opt-in via make_sun_safe("cad").
        """
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

    def test_override_sun_exclusion_radius(self):
        """Test overriding sun exclusion radius."""
        site = get_fyst_site(sun_exclusion_radius=30.0)
        assert site.sun_avoidance.exclusion_radius == 30.0
        # Other sun params unchanged
        assert site.sun_avoidance.warning_radius == FYST_SUN_WARNING_RADIUS
        assert site.sun_avoidance.enabled == FYST_SUN_AVOIDANCE_ENABLED

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

    def test_rejects_nan_sun_exclusion_radius(self):
        """A NaN radius is refused at construction instead of passing every Sun check."""
        with pytest.raises(ValueError, match="^exclusion_radius must be a finite number"):
            get_fyst_site(sun_exclusion_radius=float("nan"))

    def test_returns_fresh_instance(self):
        """Each call builds a new ``Site``; nothing memoises it.

        Equality alone cannot fail on the no-caching claim, so identity is
        what this asserts.
        """
        site1 = get_fyst_site()
        site2 = get_fyst_site()
        assert site1 is not site2
        assert site1.latitude == site2.latitude
