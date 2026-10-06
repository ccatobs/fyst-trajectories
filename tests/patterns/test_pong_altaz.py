"""Tests for PongAltAzScanPattern and PongAltAzScanConfig."""

import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from fyst_trajectories.exceptions import PointingWarning
from fyst_trajectories.patterns import (
    PongAltAzScanConfig,
    PongAltAzScanPattern,
    PongScanConfig,
    PongScanPattern,
    TrajectoryBuilder,
    compute_pong_period,
    get_pattern,
)
from fyst_trajectories.patterns.utils import _SCIENCE_SPEED_FRACTION
from fyst_trajectories.planning import plan_pong_altaz_scan


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

    def test_documented_defaults(self):
        """num_terms, angle and timestep carry the documented defaults.

        The celestial ``PongScanConfig`` has no defaults of its own; these
        are the values it is conventionally built with.
        """
        config = _base_config()
        assert config.num_terms == 4
        assert config.angle == 0.0
        assert config.timestep == 0.1

    def test_frozen(self):
        config = _base_config()
        with pytest.raises(FrozenInstanceError):
            config.az_center = 200.0

    @pytest.mark.parametrize("field", ["width", "height", "spacing", "velocity"])
    def test_nonpositive_geometry_raises(self, field):
        with pytest.raises(ValueError, match=f"{field} must be positive"):
            _base_config(**{field: 0.0})

    def test_num_terms_below_one_raises(self):
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

        assert trajectory.pattern_type == "pong_altaz"

    def test_start_time_not_required(self, site):
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

    @pytest.mark.parametrize("num_terms", [1, 4, 7])
    @pytest.mark.parametrize("angle", [0.0, -30.0, 17.5])
    @pytest.mark.parametrize(("az_center", "el_center"), [(100.0, 45.0), (120.0, 60.0)])
    def test_pointwise_mapping(self, site, az_center, el_center, angle, num_terms):
        """Every sample obeys az = x / cos(el0) + az0, el = y + el0 against the reused offsets.

        The comparison is exact: the AltAz pattern must reproduce the celestial
        Pong's offsets, flags and period bit for bit, whatever the rotation and
        the number of Fourier terms.
        """
        config = _base_config(
            el_center=el_center, az_center=az_center, angle=angle, num_terms=num_terms
        )
        pattern = PongAltAzScanPattern(config)
        duration = 100.0
        trajectory = pattern.generate(site, duration=duration)

        celestial_config = PongScanConfig(
            timestep=config.timestep,
            width=config.width,
            height=config.height,
            spacing=config.spacing,
            velocity=config.velocity,
            num_terms=config.num_terms,
            angle=config.angle,
        )
        offset_pong = PongScanPattern(ra=0.0, dec=0.0, config=celestial_config)
        times, x_off, y_off = offset_pong.generate_offsets(duration)

        cos_el = math.cos(math.radians(el_center))
        expected_az = x_off / cos_el + az_center
        expected_el = y_off + el_center
        speed = np.sqrt(np.gradient(x_off, times) ** 2 + np.gradient(y_off, times) ** 2)
        expected_flag = np.full(len(times), 2, dtype=np.int8)  # SCAN_FLAG_TURNAROUND
        expected_flag[speed >= _SCIENCE_SPEED_FRACTION * config.velocity] = 1

        np.testing.assert_array_equal(trajectory.az, expected_az)
        np.testing.assert_array_equal(trajectory.el, expected_el)
        np.testing.assert_array_equal(trajectory.scan_flag, expected_flag)
        assert compute_pong_period(config) == compute_pong_period(celestial_config)

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
        traj_no_rot = PongAltAzScanPattern(_base_config(angle=0.0)).generate(site, duration=60.0)
        traj_rot = PongAltAzScanPattern(_base_config(angle=45.0)).generate(site, duration=60.0)
        assert not np.allclose(traj_no_rot.az, traj_rot.az)


class TestRegistryAndBuilderIntegration:
    """The pattern is discoverable via the registry and the builder."""

    def test_registered_under_name(self):
        assert get_pattern("pong_altaz") is PongAltAzScanPattern

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_builder_infers_pattern_from_config(self, site):
        """TrajectoryBuilder builds the pattern from the config type alone."""
        trajectory = TrajectoryBuilder(site).with_config(_base_config()).duration(60.0).build()
        assert trajectory.pattern_type == "pong_altaz"


def _config_advisories(recwarn):
    """Return the configuration advisories ("... is unusually large") recorded so far."""
    return [
        w
        for w in recwarn.list
        if issubclass(w.category, PointingWarning) and "unusually large" in str(w.message)
    ]


class TestAdvisoriesAreNotRepeated:
    """A configuration advisory is emitted once, where the configuration is built.

    The pattern hands its own configuration to the celestial Pong, so building
    and inspecting a pattern re-runs no validation, and the planner builds one
    configuration for both the trajectory and the period.
    """

    def test_pattern_route_emits_no_advisory(self, site, recwarn):
        config = _base_config(velocity=12.0)
        recwarn.clear()
        pattern = PongAltAzScanPattern(config)
        pattern.generate(site, duration=20.0)
        pattern.generate(site, duration=20.0)
        pattern.get_metadata()
        advisories = _config_advisories(recwarn)
        assert advisories == [], [str(w.message) for w in advisories]

    def test_planner_emits_each_advisory_once(self, site, recwarn):
        recwarn.clear()
        plan_pong_altaz_scan(
            az_center=120.0,
            el_center=60.0,
            width=31.0,
            height=2.0,
            spacing=0.5,
            velocity=0.5,
            site=site,
            start_time="2026-03-15T01:00:00",
        )
        advisories = _config_advisories(recwarn)
        messages = [str(w.message) for w in advisories]
        assert len(messages) == 1, messages
        assert messages[0].startswith("Scan width 31.0 deg is unusually large")
        assert (
            Path(advisories[0].filename)
            .as_posix()
            .endswith("fyst_trajectories/planning/pong_altaz.py")
        )
