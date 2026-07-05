"""Tests for PongAltAzScanPattern and PongAltAzScanConfig."""

import math

import numpy as np
import pytest

from fyst_trajectories.exceptions import PointingWarning
from fyst_trajectories.patterns import (
    PongAltAzScanConfig,
    PongAltAzScanPattern,
    PongScanConfig,
    PongScanPattern,
    TrajectoryBuilder,
    get_pattern,
)


def _base_config(**overrides):
    """Build a PongAltAzScanConfig with sensible test defaults."""
    params = dict(
        az_center=120.0,
        el_center=60.0,
        width=2.0,
        height=2.0,
        spacing=0.1,
        velocity=0.5,
    )
    params.update(overrides)
    return PongAltAzScanConfig(**params)


class TestPongAltAzScanConfig:
    """Validation and defaults for PongAltAzScanConfig."""

    def test_defaults_match_celestial_pong(self):
        """num_terms, angle, timestep default to the celestial Pong values."""
        config = _base_config()
        assert config.num_terms == 4
        assert config.angle == 0.0
        assert config.timestep == 0.1

    def test_frozen(self):
        """Config is immutable after creation."""
        config = _base_config()
        with pytest.raises((AttributeError, TypeError)):
            config.az_center = 200.0

    @pytest.mark.parametrize("field", ["width", "height", "spacing", "velocity"])
    def test_nonpositive_geometry_raises(self, field):
        """Non-positive width/height/spacing/velocity raises ValueError."""
        with pytest.raises(ValueError, match=f"{field} must be positive"):
            _base_config(**{field: 0.0})

    def test_num_terms_below_one_raises(self):
        """num_terms < 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_terms must be at least 1"):
            _base_config(num_terms=0)

    def test_timestep_nonpositive_raises(self):
        """Non-positive timestep raises ValueError (from the base config)."""
        with pytest.raises(ValueError, match="timestep must be positive"):
            _base_config(timestep=0.0)

    @pytest.mark.parametrize("el_center", [0.0, 90.0, -10.0, 95.0])
    def test_el_center_out_of_range_raises(self, el_center):
        """el_center outside (0, 90) raises so cos(el_center) stays nonzero."""
        with pytest.raises(ValueError, match="el_center must be in"):
            _base_config(el_center=el_center)

    def test_large_width_warns(self):
        """An unusually large width emits a PointingWarning."""
        with pytest.warns(PointingWarning, match="Scan width"):
            _base_config(width=40.0)

    def test_azimuth_coordinate_velocity_warns(self):
        """A center elevation that inflates the az-coordinate speed warns.

        At el_center=88 deg, cos is ~0.035, so a modest on-sky velocity maps
        to a very large azimuth-coordinate speed, which should trip the
        azimuth-coordinate velocity advisory even though the on-sky velocity
        itself is well under the threshold.
        """
        with pytest.warns(PointingWarning, match="Azimuth-coordinate velocity"):
            _base_config(el_center=88.0, velocity=1.0)


class TestPongAltAzScanPattern:
    """Trajectory generation and the horizon-frame mapping."""

    def test_basic_generation(self, site):
        """Pattern generates a finite altaz trajectory with the right name."""
        pattern = PongAltAzScanPattern(_base_config())
        trajectory = pattern.generate(site, duration=60.0)

        assert trajectory.n_points > 0
        assert trajectory.pattern_type == "pong_altaz"
        assert trajectory.coordsys == "altaz"
        assert np.all(np.isfinite(trajectory.az))
        assert np.all(np.isfinite(trajectory.el))

    def test_start_time_not_required(self, site):
        """AltAz pattern builds without a start_time."""
        pattern = PongAltAzScanPattern(_base_config())
        trajectory = pattern.generate(site, duration=60.0, start_time=None)
        assert trajectory.start_time is None

    def test_center_placement(self, site):
        """The trajectory is centered on (az_center, el_center)."""
        config = _base_config(az_center=130.0, el_center=55.0)
        pattern = PongAltAzScanPattern(config)
        trajectory = pattern.generate(site, duration=200.0)

        az_mid = 0.5 * (trajectory.az.min() + trajectory.az.max())
        el_mid = 0.5 * (trajectory.el.min() + trajectory.el.max())
        assert az_mid == pytest.approx(130.0, abs=0.05)
        assert el_mid == pytest.approx(55.0, abs=0.05)

    def test_mapping_matches_hand_formula(self, site):
        """Az/el extents follow az = x/cos(el0)+az0, el = y+el0 exactly.

        The pattern must reproduce the pinned legacy mapping. With
        el_center=60 deg the azimuth coordinate is stretched by
        1/cos(60 deg) = 2, so the azimuth extent equals the offset x-extent
        times 2, and the elevation extent equals the offset y-extent. Both
        are checked against the offsets pulled from the reused celestial
        Pong, to float tolerance.
        """
        config = _base_config(el_center=60.0)
        pattern = PongAltAzScanPattern(config)
        duration = 200.0
        trajectory = pattern.generate(site, duration=duration)

        # Offsets from the exact same machinery the pattern reuses.
        offset_pong = PongScanPattern(
            ra=0.0,
            dec=0.0,
            config=PongScanConfig(
                timestep=config.timestep,
                width=config.width,
                height=config.height,
                spacing=config.spacing,
                velocity=config.velocity,
                num_terms=config.num_terms,
                angle=config.angle,
            ),
        )
        _, x_off, y_off = offset_pong.generate_offsets(duration)

        cos60 = math.cos(math.radians(60.0))
        assert cos60 == pytest.approx(0.5)

        # Azimuth extent = x-offset extent / cos(el_center); elevation
        # extent = y-offset extent (no stretch).
        expected_az_extent = np.ptp(x_off) / cos60
        expected_el_extent = np.ptp(y_off)
        assert np.ptp(trajectory.az) == pytest.approx(expected_az_extent, abs=1e-6)
        assert np.ptp(trajectory.el) == pytest.approx(expected_el_extent, abs=1e-6)

        # The azimuth stretch factor is exactly 1/cos(60 deg) = 2, applied to
        # the x-offset extent (not the y-offset: the Pong x and y extents
        # differ because the vertex counts differ).
        assert np.ptp(trajectory.az) == pytest.approx(2.0 * np.ptp(x_off), rel=1e-6)

    def test_pointwise_mapping(self, site):
        """Every sample obeys the mapping against the reused offsets."""
        config = _base_config(el_center=45.0, az_center=100.0)
        pattern = PongAltAzScanPattern(config)
        duration = 100.0
        trajectory = pattern.generate(site, duration=duration)

        offset_pong = PongScanPattern(
            ra=0.0,
            dec=0.0,
            config=PongScanConfig(
                timestep=config.timestep,
                width=config.width,
                height=config.height,
                spacing=config.spacing,
                velocity=config.velocity,
                num_terms=config.num_terms,
                angle=config.angle,
            ),
        )
        _, x_off, y_off = offset_pong.generate_offsets(duration)

        cos45 = math.cos(math.radians(45.0))
        expected_az = x_off / cos45 + 100.0
        expected_el = y_off + 45.0

        np.testing.assert_allclose(trajectory.az, expected_az, atol=1e-6)
        np.testing.assert_allclose(trajectory.el, expected_el, atol=1e-6)

    def test_scan_flags_present(self, site):
        """Trajectory carries both SCIENCE and TURNAROUND flags."""
        pattern = PongAltAzScanPattern(_base_config())
        trajectory = pattern.generate(site, duration=200.0)

        assert trajectory.scan_flag is not None
        assert np.any(trajectory.scan_flag == 1)  # SCAN_FLAG_SCIENCE
        assert np.any(trajectory.scan_flag == 2)  # SCAN_FLAG_TURNAROUND
        # Science samples exist and dominate.
        science_frac = (trajectory.scan_flag == 1).sum() / len(trajectory.scan_flag)
        assert science_frac > 0.7

    def test_metadata_stored(self, site):
        """Pattern metadata records the center and Lissajous params."""
        config = _base_config(az_center=130.0, el_center=55.0, angle=30.0)
        pattern = PongAltAzScanPattern(config)
        trajectory = pattern.generate(site, duration=60.0)

        params = trajectory.pattern_params
        assert params["az_center"] == 130.0
        assert params["el_center"] == 55.0
        assert params["width"] == 2.0
        assert params["height"] == 2.0
        assert params["angle"] == 30.0
        assert "period" in params
        assert "x_numvert" in params
        assert "y_numvert" in params

    def test_angle_changes_trajectory(self, site):
        """A non-zero rotation angle changes the trajectory."""
        traj_no_rot = PongAltAzScanPattern(_base_config(angle=0.0)).generate(site, duration=60.0)
        traj_rot = PongAltAzScanPattern(_base_config(angle=45.0)).generate(site, duration=60.0)
        assert not np.allclose(traj_no_rot.az, traj_rot.az)


class TestRegistryAndBuilderIntegration:
    """The pattern is discoverable via the registry and the builder."""

    def test_registered_under_name(self):
        """get_pattern('pong_altaz') returns the pattern class."""
        assert get_pattern("pong_altaz") is PongAltAzScanPattern

    def test_builder_infers_pattern_from_config(self, site):
        """TrajectoryBuilder builds the pattern from the config type alone."""
        trajectory = TrajectoryBuilder(site).with_config(_base_config()).duration(60.0).build()
        assert trajectory.pattern_type == "pong_altaz"
        assert trajectory.coordsys == "altaz"
