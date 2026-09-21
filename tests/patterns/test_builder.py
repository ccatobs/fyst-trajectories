"""Tests for TrajectoryBuilder."""

import pytest
from astropy.time import Time

from fyst_trajectories.exceptions import AzimuthBoundsError, ElevationBoundsError
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
    """The fluent chain per pattern type, and the refusals when a required step is missing."""

    def test_builder_basic_pong(self, site):
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

    def test_builder_with_start_time(self, site):
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
        builder = TrajectoryBuilder(site).duration(60.0)

        with pytest.raises(ValueError, match="Pattern not set"):
            builder.build()

    def test_builder_missing_duration_raises(self, site):
        builder = TrajectoryBuilder(site).at(ra=180.0, dec=-30.0).with_config(_PONG_CONFIG)

        with pytest.raises(ValueError, match="Duration not set"):
            builder.build()

    def test_builder_invalid_config_raises(self, site):
        # Create a custom config class not in CONFIG_TO_PATTERN
        class UnknownConfig(ScanConfig):
            pass

        with pytest.raises(ValueError, match="Unknown config type"):
            TrajectoryBuilder(site).with_config(UnknownConfig(timestep=0.1))

    def test_builder_negative_duration_raises(self, site):
        with pytest.raises(ValueError, match="Duration must be positive"):
            TrajectoryBuilder(site).duration(-10.0)

    def test_builder_zero_duration_raises(self, site):
        with pytest.raises(ValueError, match="Duration must be positive"):
            TrajectoryBuilder(site).duration(0.0)

    def test_builder_daisy(self, site):
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
        """A planet track builds with no ra/dec supplied."""
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
        """``.at()`` coordinates warn and are ignored for planet tracking."""
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
        builder = TrajectoryBuilder(site).with_config(_PONG_CONFIG).duration(60.0)

        with pytest.raises(ValueError, match="requires sky coordinates"):
            builder.build()

    def test_builder_missing_starting_at_for_celestial_raises(self, site):
        builder = (
            TrajectoryBuilder(site).at(ra=180.0, dec=-30.0).with_config(_PONG_CONFIG).duration(60.0)
        )

        with pytest.raises(ValueError, match="requires a start time"):
            builder.build()

    def test_builder_missing_starting_at_for_daisy_raises(self, site):
        builder = (
            TrajectoryBuilder(site)
            .at(ra=180.0, dec=-30.0)
            .with_config(_DAISY_CONFIG)
            .duration(60.0)
        )

        with pytest.raises(ValueError, match="requires a start time"):
            builder.build()

    def test_builder_missing_starting_at_for_sidereal_raises(self, site):
        builder = (
            TrajectoryBuilder(site)
            .at(ra=180.0, dec=-30.0)
            .with_config(_SIDEREAL_CONFIG)
            .duration(60.0)
        )

        with pytest.raises(ValueError, match="requires a start time"):
            builder.build()

    def test_builder_missing_starting_at_for_planet_raises(self, site):
        builder = TrajectoryBuilder(site).with_config(_PLANET_CONFIG).duration(60.0)

        with pytest.raises(ValueError, match="requires a start time"):
            builder.build()

    def test_builder_constant_el_without_starting_at(self, site):
        """ConstantEl is an AltAz pattern, so it builds without ``.starting_at()``."""
        trajectory = TrajectoryBuilder(site).with_config(_CONST_EL_CONFIG).duration(30.0).build()

        assert trajectory.n_points > 0
        assert trajectory.pattern_type == "constant_el"


class TestBuilderBoundsValidation:
    """Builder must re-validate trajectory bounds after generation.

    Individual patterns already validate their own bounds, so the
    builder's call is defence in depth. These tests neutralise the
    per-pattern bounds check and verify that the builder still refuses
    to emit an out-of-bounds trajectory.
    """

    def test_build_raises_when_azimuth_exceeds_limits(self, site, monkeypatch):
        """Builder must raise if a trajectory slips past the pattern's bounds check."""
        # Configure a constant-el scan whose azimuth range exceeds the
        # FYST azimuth limit (az_max=360). Ordinarily the pattern's own
        # validate_trajectory_bounds call would raise, so we neutralise
        # it inside the pattern module only. The builder imports its
        # own reference to validate_trajectory_bounds from
        # trajectory_utils, which is unaffected.
        from fyst_trajectories.patterns import constant_el as ce_module

        monkeypatch.setattr(ce_module, "validate_trajectory_bounds", lambda *a, **k: None)

        bad_config = ConstantElScanConfig(
            timestep=0.1,
            az_start=355.0,
            az_stop=400.0,  # > FYST_AZ_MAX (360)
            elevation=45.0,
            az_speed=1.0,
            az_accel=0.5,
        )

        builder = TrajectoryBuilder(site).with_config(bad_config).duration(30.0)

        with pytest.raises(AzimuthBoundsError):
            builder.build()

    def test_build_raises_when_elevation_exceeds_limits(self, site, monkeypatch):
        """Builder must raise if the final trajectory elevation is out of range."""
        from fyst_trajectories.patterns import constant_el as ce_module

        monkeypatch.setattr(ce_module, "validate_trajectory_bounds", lambda *a, **k: None)

        # Elevation 95 degrees is above FYST_EL_MAX (90 deg).
        bad_config = ConstantElScanConfig(
            timestep=0.1,
            az_start=100.0,
            az_stop=150.0,
            elevation=95.0,
            az_speed=1.0,
            az_accel=0.5,
        )

        builder = TrajectoryBuilder(site).with_config(bad_config).duration(30.0)

        with pytest.raises(ElevationBoundsError):
            builder.build()

    def test_build_succeeds_for_in_bounds_trajectory(self, site):
        """Valid in-bounds configs build without raising."""
        # Sanity check: the bounds re-validation must not refuse the happy
        # path. Uses the reusable in-bounds config from the top of this
        # module.
        trajectory = TrajectoryBuilder(site).with_config(_CONST_EL_CONFIG).duration(30.0).build()
        assert trajectory.n_points > 0


class TestBuildValidateDynamicsOptOut:
    """``build(validate_dynamics=False)`` skips only the dynamics advisory."""

    # az_accel 1.5 makes the quintic turnaround peak at 2.25 deg/s^2, over
    # the 1.5 deg/s^2 site limit, so the default build warns.
    _HOT_CONFIG = ConstantElScanConfig(
        timestep=0.1,
        az_start=100.0,
        az_stop=102.0,
        elevation=45.0,
        az_speed=1.5,
        az_accel=1.5,
    )

    def test_default_build_warns_on_the_turnaround_peak(self, site):
        from fyst_trajectories.exceptions import AccelerationLimitWarning

        with pytest.warns(AccelerationLimitWarning):
            TrajectoryBuilder(site).with_config(self._HOT_CONFIG).duration(30.0).build()

    def test_opt_out_is_silent(self, site):
        import warnings

        from fyst_trajectories.exceptions import PointingWarning

        builder = TrajectoryBuilder(site).with_config(self._HOT_CONFIG).duration(30.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", PointingWarning)
            trajectory = builder.build(validate_dynamics=False)
        assert trajectory.n_points > 0

    def test_opt_out_keeps_the_bounds_check(self, site, monkeypatch):
        """The position-bounds check is not part of the opt-out."""
        # Same construction as test_build_raises_when_azimuth_exceeds_limits:
        # neutralise the pattern's own bounds check so only the builder's
        # defence-in-depth call can refuse the out-of-range trajectory.
        from fyst_trajectories.patterns import constant_el as ce_module

        monkeypatch.setattr(ce_module, "validate_trajectory_bounds", lambda *a, **k: None)
        bad_config = ConstantElScanConfig(
            timestep=0.1,
            az_start=355.0,
            az_stop=400.0,
            elevation=45.0,
            az_speed=1.0,
            az_accel=0.5,
        )
        builder = TrajectoryBuilder(site).with_config(bad_config).duration(30.0)

        with pytest.raises(AzimuthBoundsError):
            builder.build(validate_dynamics=False)
