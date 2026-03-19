"""Tests for TrajectoryBuilder."""

import pytest
from astropy.time import Time

from fyst_trajectories.patterns import (
    ConstantElScanConfig,
    DaisyScanConfig,
    PlanetTrackConfig,
    PongScanConfig,
    ScanConfig,
    SiderealTrackConfig,
    TrajectoryBuilder,
)

# Reusable config instances for tests
_PONG_CONFIG = PongScanConfig(
    timestep=0.1, width=1.0, height=1.0, spacing=0.1, velocity=0.5, num_terms=4, angle=0.0
)
_CONST_EL_CONFIG = ConstantElScanConfig(
    timestep=0.1,
    az_start=100.0,
    az_stop=150.0,
    elevation=45.0,
    az_speed=1.0,
    az_accel=0.5,
    n_scans=1,
)
_DAISY_CONFIG = DaisyScanConfig(
    timestep=0.1,
    radius=0.3,
    velocity=0.2,
    turn_radius=0.1,
    avoidance_radius=0.0,
    start_acceleration=0.5,
    y_offset=0.0,
)
_SIDEREAL_CONFIG = SiderealTrackConfig(timestep=0.1)
_PLANET_CONFIG = PlanetTrackConfig(timestep=0.1, body="mars")


class TestTrajectoryBuilder:
    """Tests for TrajectoryBuilder fluent API."""

    def test_builder_basic_pong(self, site):
        """Test building a basic Pong trajectory."""
        # Use a fixed start time and position that will be well above horizon
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        trajectory = (
            TrajectoryBuilder(site)
            .at(ra=180.0, dec=-30.0)
            .with_config(_PONG_CONFIG)
            .duration(60.0)
            .starting_at(start_time)
            .build()
        )

        assert trajectory.n_points > 0
        assert trajectory.pattern_type == "pong"
        assert trajectory.center_ra == 180.0
        assert trajectory.center_dec == -30.0

    def test_builder_constant_el(self, site):
        """Test building a constant elevation scan."""
        trajectory = TrajectoryBuilder(site).with_config(_CONST_EL_CONFIG).duration(30.0).build()

        assert trajectory.n_points > 0
        assert trajectory.pattern_type == "constant_el"

    def test_builder_with_start_time(self, site):
        """Test building trajectory with explicit start time."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")

        trajectory = (
            TrajectoryBuilder(site)
            .at(ra=180.0, dec=-30.0)
            .with_config(_PONG_CONFIG)
            .duration(60.0)
            .starting_at(start_time)
            .build()
        )

        assert trajectory.start_time == start_time

    def test_builder_with_string_start_time(self, site):
        """Test building trajectory with ISO string start time."""
        trajectory = (
            TrajectoryBuilder(site)
            .at(ra=180.0, dec=-30.0)
            .with_config(_PONG_CONFIG)
            .duration(60.0)
            .starting_at("2026-03-15T04:00:00")
            .build()
        )

        assert trajectory.start_time is not None

    def test_builder_missing_config_raises(self, site):
        """Test that build raises if config not set."""
        builder = TrajectoryBuilder(site).duration(60.0)

        with pytest.raises(ValueError, match="Pattern not set"):
            builder.build()

    def test_builder_missing_duration_raises(self, site):
        """Test that build raises if duration not set."""
        builder = TrajectoryBuilder(site).at(ra=180.0, dec=-30.0).with_config(_PONG_CONFIG)

        with pytest.raises(ValueError, match="Duration not set"):
            builder.build()

    def test_builder_invalid_config_raises(self, site):
        """Test that invalid config type raises."""

        # Create a custom config class not in CONFIG_TO_PATTERN
        class UnknownConfig(ScanConfig):
            pass

        with pytest.raises(ValueError, match="Unknown config type"):
            TrajectoryBuilder(site).with_config(UnknownConfig(timestep=0.1))

    def test_builder_negative_duration_raises(self, site):
        """Test that negative duration raises."""
        with pytest.raises(ValueError, match="Duration must be positive"):
            TrajectoryBuilder(site).duration(-10.0)

    def test_builder_zero_duration_raises(self, site):
        """Test that zero duration raises."""
        with pytest.raises(ValueError, match="Duration must be positive"):
            TrajectoryBuilder(site).duration(0.0)

    def test_builder_daisy(self, site):
        """Test building a Daisy scan."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        trajectory = (
            TrajectoryBuilder(site)
            .at(ra=180.0, dec=-30.0)
            .with_config(_DAISY_CONFIG)
            .duration(60.0)
            .starting_at(start_time)
            .build()
        )

        assert trajectory.n_points > 0
        assert trajectory.pattern_type == "daisy"

    def test_builder_sidereal(self, site):
        """Test building a Sidereal track with config."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        trajectory = (
            TrajectoryBuilder(site)
            .at(ra=180.0, dec=-30.0)
            .with_config(_SIDEREAL_CONFIG)
            .duration(60.0)
            .starting_at(start_time)
            .build()
        )

        assert trajectory.n_points > 0
        assert trajectory.pattern_type == "sidereal"

    def test_builder_planet(self, site):
        """Test building a planet track via the builder (no ra/dec needed)."""
        start_time = Time("2026-03-15T12:00:00", scale="utc")
        trajectory = (
            TrajectoryBuilder(site)
            .with_config(_PLANET_CONFIG)
            .duration(60.0)
            .starting_at(start_time)
            .build()
        )

        assert trajectory.n_points > 0
        assert trajectory.pattern_type == "planet"

    def test_builder_planet_ignores_at(self, site):
        """Test that .at() coordinates emit a warning for planet tracking."""
        start_time = Time("2026-03-15T12:00:00", scale="utc")
        with pytest.warns(UserWarning, match="ra/dec values are ignored"):
            trajectory = (
                TrajectoryBuilder(site)
                .at(ra=999.0, dec=999.0)  # Should warn and be ignored for planet
                .with_config(_PLANET_CONFIG)
                .duration(60.0)
                .starting_at(start_time)
                .build()
            )

        assert trajectory.n_points > 0
        assert trajectory.pattern_type == "planet"

    def test_builder_missing_at_for_celestial_raises(self, site):
        """Test that build raises if .at() not called for celestial pattern."""
        builder = TrajectoryBuilder(site).with_config(_PONG_CONFIG).duration(60.0)

        with pytest.raises(ValueError, match="requires sky coordinates"):
            builder.build()

    def test_builder_missing_starting_at_for_celestial_raises(self, site):
        """Test that build raises if .starting_at() not called for celestial pattern."""
        builder = (
            TrajectoryBuilder(site).at(ra=180.0, dec=-30.0).with_config(_PONG_CONFIG).duration(60.0)
        )

        with pytest.raises(ValueError, match="requires a start time"):
            builder.build()

    def test_builder_missing_starting_at_for_daisy_raises(self, site):
        """Test that build raises if .starting_at() not called for Daisy pattern."""
        builder = (
            TrajectoryBuilder(site)
            .at(ra=180.0, dec=-30.0)
            .with_config(_DAISY_CONFIG)
            .duration(60.0)
        )

        with pytest.raises(ValueError, match="requires a start time"):
            builder.build()

    def test_builder_missing_starting_at_for_sidereal_raises(self, site):
        """Test that build raises if .starting_at() not called for Sidereal pattern."""
        builder = (
            TrajectoryBuilder(site)
            .at(ra=180.0, dec=-30.0)
            .with_config(_SIDEREAL_CONFIG)
            .duration(60.0)
        )

        with pytest.raises(ValueError, match="requires a start time"):
            builder.build()

    def test_builder_missing_starting_at_for_planet_raises(self, site):
        """Test that build raises if .starting_at() not called for Planet pattern."""
        builder = TrajectoryBuilder(site).with_config(_PLANET_CONFIG).duration(60.0)

        with pytest.raises(ValueError, match="requires a start time"):
            builder.build()

    def test_builder_constant_el_without_starting_at(self, site):
        """Test that ConstantEl builds without .starting_at() (AltAz pattern)."""
        trajectory = TrajectoryBuilder(site).with_config(_CONST_EL_CONFIG).duration(30.0).build()

        assert trajectory.n_points > 0
        assert trajectory.pattern_type == "constant_el"
