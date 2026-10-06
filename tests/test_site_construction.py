"""Construction-time guards on :class:`~fyst_trajectories.site.Site`.

The behavioural companion to ``test_site.py``, which covers the value
semantics of the site record and its YAML loader. These tests pin what a
``Site`` refuses to be built from, and the canonicalisation it performs while
building: every geometry field is checked at construction, so a latitude of
999 is refused here rather than failing several frames later inside astropy,
and an accepted port string is stored lower-cased, so ``site.nasmyth_port``
and the value the sign lookup uses can never differ in case.
"""

import pytest

from fyst_trajectories.site import (
    AxisLimits,
    Site,
    SunAvoidanceConfig,
    TelescopeLimits,
)


def _make_site(**overrides) -> Site:
    """Build a minimal site, overriding the named fields."""
    kwargs = {
        "name": "Test",
        "description": "",
        "latitude": -22.985639,
        "longitude": -67.740278,
        "elevation": 5611.8,
        "telescope_limits": TelescopeLimits(
            azimuth=AxisLimits(min=-180.0, max=360.0, max_velocity=3.0, max_acceleration=1.5),
            elevation=AxisLimits(min=20.0, max=89.0, max_velocity=1.0, max_acceleration=0.75),
        ),
        "sun_avoidance": SunAvoidanceConfig(
            enabled=True, exclusion_radius=45.0, warning_radius=50.0
        ),
    }
    kwargs.update(overrides)
    return Site(**kwargs)


class TestGeographyValidation:
    """Latitude, longitude and elevation are checked where they are set."""

    @pytest.mark.parametrize("latitude", [999.0, -90.5, 90.5, float("nan")])
    def test_out_of_range_latitude_is_refused(self, latitude):
        """A latitude outside [-90, 90] raises at construction, not at ``location``."""
        with pytest.raises(ValueError, match="latitude must lie in"):
            _make_site(latitude=latitude)

    @pytest.mark.parametrize("longitude", [float("nan"), float("inf")])
    def test_non_finite_longitude_is_refused(self, longitude):
        with pytest.raises(ValueError, match="longitude must be a finite number"):
            _make_site(longitude=longitude)

    @pytest.mark.parametrize("elevation", [float("nan"), float("-inf")])
    def test_non_finite_elevation_is_refused(self, elevation):
        with pytest.raises(ValueError, match="elevation must be a finite number"):
            _make_site(elevation=elevation)

    def test_valid_extremes_still_build(self):
        """The poles are legal latitudes and a below-sea-level site is legal."""
        assert _make_site(latitude=90.0).latitude == 90.0
        assert _make_site(latitude=-90.0).latitude == -90.0
        assert _make_site(elevation=-400.0).elevation == -400.0


class TestPlateScaleValidation:
    """``plate_scale`` is a non-negative arcsec/mm scale."""

    @pytest.mark.parametrize("plate_scale", [-1.0, float("nan"), float("inf")])
    def test_negative_or_non_finite_plate_scale_is_refused(self, plate_scale):
        with pytest.raises(ValueError, match="plate_scale must be a finite, non-negative"):
            _make_site(plate_scale=plate_scale)

    def test_zero_plate_scale_still_builds(self):
        """Zero is the documented default: every focal-plane position maps to zero offset."""
        assert _make_site().plate_scale == 0.0

    def test_the_yaml_loader_is_deliberately_stricter_than_the_constructor(self, tmp_path):
        """A config that declares the key has declared geometry, so zero is a typo there.

        The two constructors disagree about zero on purpose, and the loader's
        message has to say which one accepts it.
        """
        import yaml

        config = {
            "site": {
                "name": "custom",
                "location": {"latitude": -22.0, "longitude": -67.0, "elevation": 5000.0},
            },
            "telescope": {
                "plate_scale": 0.0,
                "azimuth": {"min": -180, "max": 360, "max_velocity": 3, "max_acceleration": 1.5},
                "elevation": {"min": 20, "max": 90, "max_velocity": 1.5, "max_acceleration": 0.75},
            },
            "sun_avoidance": {"enabled": True, "exclusion_radius": 45, "warning_radius": 50},
        }
        path = tmp_path / "custom.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")

        with pytest.raises(ValueError, match=r"plate_scale must be positive[\s\S]*construct Site"):
            Site.from_config(path)


class TestNasmythPortCanonicalisation:
    """The stored port spelling is the one the sign lookup uses."""

    @pytest.mark.parametrize("given", ["LEFT", "Left", "left"])
    def test_port_is_stored_lower_cased(self, given):
        site = _make_site(nasmyth_port=given)
        assert site.nasmyth_port == "left"
        assert site.nasmyth_sign == -1

    def test_unknown_port_still_names_the_value_as_given(self):
        with pytest.raises(ValueError, match="Unknown nasmyth_port 'Middle'"):
            _make_site(nasmyth_port="Middle")
