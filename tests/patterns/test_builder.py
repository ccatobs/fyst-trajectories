"""Tests for TrajectoryBuilder."""

import pytest
from astropy.time import Time

from fyst_trajectories.exceptions import (
    AzimuthBoundsError,
    ElevationBoundsError,
    PointingWarning,
)
from fyst_trajectories.offsets import InstrumentOffset
from fyst_trajectories.patterns import (
    ConstantElScanConfig,
    DaisyScanConfig,
    PlanetTrackConfig,
    PongScanConfig,
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

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
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

        assert trajectory.pattern_type == "pong"
        assert trajectory.center_ra == 180.0
        assert trajectory.center_dec == -30.0

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
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

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_builder_with_string_start_time(self, site):
        trajectory = (
            TrajectoryBuilder(site)
            .at(ra=180.0, dec=-30.0)
            .with_config(_PONG_CONFIG)
            .duration(60.0)
            .starting_at("2026-03-15T04:00:00")
            .build()
        )

        assert trajectory.start_time == Time("2026-03-15T04:00:00", scale="utc")

    def test_builder_missing_config_raises(self, site):
        builder = TrajectoryBuilder(site).duration(60.0)

        with pytest.raises(ValueError, match="Pattern not set"):
            builder.build()

    def test_builder_missing_duration_raises(self, site):
        builder = TrajectoryBuilder(site).at(ra=180.0, dec=-30.0).with_config(_PONG_CONFIG)

        with pytest.raises(ValueError, match="Duration not set"):
            builder.build()

    def test_builder_negative_duration_raises(self, site):
        with pytest.raises(ValueError, match="Duration must be positive"):
            TrajectoryBuilder(site).duration(-10.0)

    def test_builder_zero_duration_raises(self, site):
        with pytest.raises(ValueError, match="Duration must be positive"):
            TrajectoryBuilder(site).duration(0.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_builder_non_finite_duration_raises(self, site, bad):
        """A NaN or infinite duration is refused where it is set, not later in ``round``."""
        with pytest.raises(ValueError, match="Duration must be positive"):
            TrajectoryBuilder(site).duration(bad)

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory azimuth acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
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

        assert trajectory.pattern_type == "daisy"

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
    )
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

        assert trajectory.pattern_type == "planet"

    def test_builder_planet_ignores_at(self, site):
        """``.at()`` coordinates warn and are ignored for planet tracking."""
        start_time = Time("2026-03-15T12:00:00", scale="utc")
        with pytest.warns(PointingWarning, match="ra/dec values are ignored"):
            trajectory = (
                TrajectoryBuilder(site)
                .at(ra=999.0, dec=999.0)  # Should warn and be ignored for planet
                .with_config(_PLANET_CONFIG)
                .duration(60.0)
                .starting_at(start_time)
                .build()
            )
        reference = (
            TrajectoryBuilder(site)
            .with_config(_PLANET_CONFIG)
            .duration(60.0)
            .starting_at(start_time)
            .build()
        )

        assert trajectory.pattern_type == "planet"
        assert (trajectory.az == reference.az).all()
        assert (trajectory.el == reference.el).all()

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

        assert trajectory.pattern_type == "constant_el"


class TestBuilderBoundsValidation:
    """Builder must re-validate trajectory bounds after generation.

    Individual patterns already validate their own bounds, so the
    builder's call is defence in depth: it is the only check that sees a
    detector offset. The first two tests neutralise the per-pattern bounds
    check; the third drives the offset case directly.
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

    def test_build_raises_when_the_detector_offset_leaves_the_limits(self, site):
        """The case the re-validation exists for: an offset moves the boresight out.

        A constant-elevation scan at 21 deg is inside the limits; a detector
        90 arcmin along the focal-plane y axis (the elevation direction at zero
        field rotation) puts the boresight near 19.6 deg, below the 20 deg
        floor. No pattern-level check sees the offset trajectory.
        """
        low_config = ConstantElScanConfig(
            timestep=0.1,
            az_start=100.0,
            az_stop=110.0,
            elevation=21.0,
            az_speed=0.5,
            az_accel=0.5,
        )
        builder = (
            TrajectoryBuilder(site)
            .with_config(low_config)
            .for_detector(InstrumentOffset(dx=0.0, dy=90.0, name="low"))
            .duration(30.0)
        )

        with pytest.raises(ElevationBoundsError):
            builder.build()


class TestBuildValidateDynamicsOptOut:
    """``build(validate_dynamics=False)`` skips only the dynamics advisory."""

    # az_accel 1.5 makes the quintic turnaround peak at 2.25 deg/s^2, over
    # the 1.5 deg/s^2 site limit, so the default build warns. That limit is an
    # operational placeholder pending the FYST team's ratification; this config
    # moves with it.
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
