"""Tests for DaisyAltAzScanPattern and DaisyAltAzScanConfig."""

import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from fyst_trajectories.exceptions import ElevationBoundsError, PointingWarning
from fyst_trajectories.patterns import (
    DaisyAltAzScanConfig,
    DaisyAltAzScanPattern,
    DaisyScanConfig,
    DaisyScanPattern,
    TrajectoryBuilder,
    get_pattern,
)
from fyst_trajectories.patterns.utils import _SCIENCE_SPEED_FRACTION
from fyst_trajectories.planning import plan_daisy_altaz_scan


def _base_config(**overrides):
    """Build a DaisyAltAzScanConfig with sensible test defaults."""
    params = dict(
        az_center=120.0,
        el_center=60.0,
        radius=0.5,
        velocity=0.3,
        turn_radius=0.2,
        avoidance_radius=0.0,
        start_acceleration=0.5,
    )
    params.update(overrides)
    return DaisyAltAzScanConfig(**params)


class TestDaisyAltAzScanConfig:
    """Validation and defaults for DaisyAltAzScanConfig."""

    def test_defaults(self):
        """y_offset and timestep default to the documented values."""
        config = _base_config()
        assert config.y_offset == 0.0
        assert config.timestep == 0.1

    def test_frozen(self):
        config = _base_config()
        with pytest.raises(FrozenInstanceError):
            config.az_center = 200.0

    @pytest.mark.parametrize("field", ["radius", "velocity", "turn_radius"])
    def test_nonpositive_geometry_raises(self, field):
        with pytest.raises(ValueError, match=f"{field} must be positive"):
            _base_config(**{field: 0.0})

    def test_negative_avoidance_radius_raises(self):
        with pytest.raises(ValueError, match="avoidance_radius must be non-negative"):
            _base_config(avoidance_radius=-0.1)

    def test_nonpositive_start_acceleration_raises(self):
        with pytest.raises(ValueError, match="start_acceleration must be positive"):
            _base_config(start_acceleration=0.0)

    def test_timestep_nonpositive_raises(self):
        """Non-positive timestep raises ValueError (from the base config)."""
        with pytest.raises(ValueError, match="timestep must be positive"):
            _base_config(timestep=0.0)

    @pytest.mark.parametrize("el_center", [0.0, 90.0, -10.0, 95.0])
    def test_el_center_out_of_range_raises(self, el_center):
        """el_center outside (0, 90) raises so cos(el_center) stays nonzero."""
        with pytest.raises(ValueError, match="el_center must be in"):
            _base_config(el_center=el_center)

    def test_large_radius_warns(self):
        with pytest.warns(PointingWarning, match="Daisy radius"):
            _base_config(radius=20.0)

    def test_azimuth_coordinate_velocity_warns(self):
        """A center elevation that inflates the az-coordinate speed warns.

        At el_center=88 deg, cos is ~0.035, so a modest on-sky velocity maps
        to a very large azimuth-coordinate speed, which should trip the
        azimuth-coordinate velocity advisory even though the on-sky velocity
        itself is well under the threshold.
        """
        with pytest.warns(PointingWarning, match="Azimuth-coordinate velocity"):
            _base_config(el_center=88.0, velocity=1.0)


class TestDaisyAltAzScanPattern:
    """Trajectory generation and the horizon-frame mapping."""

    def test_basic_generation(self, site):
        """Pattern generates a finite altaz trajectory with the right name."""
        pattern = DaisyAltAzScanPattern(_base_config())
        trajectory = pattern.generate(site, duration=100.0)

        assert trajectory.pattern_type == "daisy_altaz"

    def test_start_time_not_required(self, site):
        pattern = DaisyAltAzScanPattern(_base_config())
        trajectory = pattern.generate(site, duration=100.0, start_time=None)
        assert trajectory.start_time is None

    def test_center_placement(self, site):
        """The trajectory is centered on (az_center, el_center)."""
        config = _base_config(az_center=130.0, el_center=55.0)
        pattern = DaisyAltAzScanPattern(config)
        trajectory = pattern.generate(site, duration=200.0)

        az_mid = 0.5 * (trajectory.az.min() + trajectory.az.max())
        el_mid = 0.5 * (trajectory.el.min() + trajectory.el.max())
        assert az_mid == pytest.approx(130.0, abs=0.05)
        assert el_mid == pytest.approx(55.0, abs=0.05)

    @pytest.mark.parametrize("avoidance_radius", [0.0, 0.1])
    @pytest.mark.parametrize("y_offset", [0.0, 0.3])
    @pytest.mark.parametrize(("az_center", "el_center"), [(100.0, 45.0), (120.0, 60.0)])
    def test_pointwise_mapping(self, site, az_center, el_center, y_offset, avoidance_radius):
        """Every sample obeys az = x / cos(el0) + az0, el = y + el0 against the reused offsets.

        The comparison is exact: the AltAz pattern must reproduce the celestial
        Daisy's offsets and flags bit for bit, with or without a petal offset
        and an avoidance radius.
        """
        config = _base_config(
            el_center=el_center,
            az_center=az_center,
            y_offset=y_offset,
            avoidance_radius=avoidance_radius,
        )
        pattern = DaisyAltAzScanPattern(config)
        duration = 100.0
        trajectory = pattern.generate(site, duration=duration)

        offset_daisy = DaisyScanPattern(
            ra=0.0,
            dec=0.0,
            config=DaisyScanConfig(
                timestep=config.timestep,
                radius=config.radius,
                velocity=config.velocity,
                turn_radius=config.turn_radius,
                avoidance_radius=config.avoidance_radius,
                start_acceleration=config.start_acceleration,
                y_offset=config.y_offset,
            ),
        )
        times, x_off, y_off = offset_daisy.generate_offsets(duration)

        cos_el = math.cos(math.radians(el_center))
        expected_az = x_off / cos_el + az_center
        expected_el = y_off + el_center
        speed = np.sqrt(np.gradient(x_off, times) ** 2 + np.gradient(y_off, times) ** 2)
        expected_flag = np.full(len(times), 2, dtype=np.int8)  # SCAN_FLAG_TURNAROUND
        expected_flag[speed >= _SCIENCE_SPEED_FRACTION * config.velocity] = 1

        # az is used as provided by the AltAz mapping (no normalization), so a
        # direct comparison is expected here.
        np.testing.assert_array_equal(trajectory.az, expected_az)
        np.testing.assert_array_equal(trajectory.el, expected_el)
        np.testing.assert_array_equal(trajectory.scan_flag, expected_flag)

    def test_scan_flags_match_celestial_daisy(self, site):
        """Flags mirror the celestial Daisy: SCIENCE dominates, ramp is not.

        The Daisy has no turnaround concept. The only non-science phase is
        the initial start_acceleration ramp, flagged from the offset-frame
        speed exactly as the celestial Daisy does, so the flag meaning is
        identical to plan_daisy_scan's and independent of the cos(el) stretch.
        The flags are reproduced here from the reused offsets using the
        celestial Daisy's own science-speed threshold to prove the AltAz
        mapping does not change which samples are science.
        """
        config = _base_config(start_acceleration=0.1)
        pattern = DaisyAltAzScanPattern(config)
        duration = 200.0
        trajectory = pattern.generate(site, duration=duration)

        assert trajectory.scan_flag is not None
        assert np.any(trajectory.scan_flag == 1)  # SCAN_FLAG_SCIENCE
        # A slow ramp guarantees some sub-threshold (non-science) samples.
        assert np.any(trajectory.scan_flag == 2)  # SCAN_FLAG_TURNAROUND

        # Recompute the expected flags straight from the reused offsets with
        # the celestial Daisy's threshold; the mapping must leave them intact.
        offset_daisy = DaisyScanPattern(
            ra=0.0,
            dec=0.0,
            config=DaisyScanConfig(
                timestep=config.timestep,
                radius=config.radius,
                velocity=config.velocity,
                turn_radius=config.turn_radius,
                avoidance_radius=config.avoidance_radius,
                start_acceleration=config.start_acceleration,
                y_offset=config.y_offset,
            ),
        )
        times, x_off, y_off = offset_daisy.generate_offsets(duration)
        speed = np.hypot(np.gradient(x_off, times), np.gradient(y_off, times))
        expected = np.full(len(times), 2, dtype=np.int8)  # SCAN_FLAG_TURNAROUND
        expected[speed >= _SCIENCE_SPEED_FRACTION * config.velocity] = 1
        np.testing.assert_array_equal(trajectory.scan_flag, expected)

    def test_metadata_stored(self, site):
        """Pattern metadata records the center and Daisy params."""
        config = _base_config(az_center=130.0, el_center=55.0, radius=0.7)
        pattern = DaisyAltAzScanPattern(config)
        trajectory = pattern.generate(site, duration=100.0)

        params = trajectory.pattern_params
        assert params["az_center"] == 130.0
        assert params["el_center"] == 55.0
        assert params["radius"] == 0.7
        assert params["velocity"] == 0.3
        assert params["turn_radius"] == 0.2
        assert params["start_acceleration"] == 0.5
        assert params["y_offset"] == 0.0

    def test_bounds_error_when_elevation_exceeds_limit(self, site):
        """A high el_center plus a large radius drives el past the 90 deg limit.

        With el_center=87 and radius=6 the realized Daisy y-extent reaches
        ~5.7 deg, so the upper elevation extent hits ~92.7 deg and bounds
        validation must raise. (The radius is chosen so the realized petal
        extent, not merely R0, clears 90.)
        """
        with pytest.warns(PointingWarning, match="Azimuth-coordinate velocity"):
            config = _base_config(
                el_center=87.0,
                radius=6.0,
                turn_radius=0.5,
            )
        pattern = DaisyAltAzScanPattern(config)
        with pytest.raises(ElevationBoundsError, match=r"92\.7"):
            pattern.generate(site, duration=400.0)


class TestRegistryAndBuilderIntegration:
    """The pattern is discoverable via the registry and the builder."""

    def test_registered_under_name(self):
        assert get_pattern("daisy_altaz") is DaisyAltAzScanPattern

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning"
    )
    def test_builder_infers_pattern_from_config(self, site):
        """TrajectoryBuilder builds the pattern from the config type alone."""
        trajectory = TrajectoryBuilder(site).with_config(_base_config()).duration(100.0).build()
        assert trajectory.pattern_type == "daisy_altaz"


def _config_advisories(recwarn):
    """Return the configuration advisories ("... is unusually large") recorded so far."""
    return [
        w
        for w in recwarn.list
        if issubclass(w.category, PointingWarning) and "unusually large" in str(w.message)
    ]


class TestAdvisoriesAreNotRepeated:
    """A configuration advisory is emitted once, where the configuration is built.

    The pattern hands its own configuration to the celestial Daisy, so building
    and inspecting a pattern re-runs no validation, and the planner builds one
    configuration for the trajectory.
    """

    def test_pattern_route_emits_no_advisory(self, site, recwarn):
        config = _base_config(velocity=12.0)
        recwarn.clear()
        pattern = DaisyAltAzScanPattern(config)
        pattern.generate(site, duration=20.0)
        pattern.generate(site, duration=20.0)
        pattern.get_metadata()
        advisories = _config_advisories(recwarn)
        assert advisories == [], [str(w.message) for w in advisories]

    def test_planner_emits_each_advisory_once(self, site, recwarn):
        recwarn.clear()
        plan_daisy_altaz_scan(
            az_center=120.0,
            el_center=60.0,
            radius=0.5,
            velocity=6.0,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            site=site,
            start_time="2026-03-15T01:00:00",
            timestep=0.1,
            duration=60.0,
        )
        advisories = _config_advisories(recwarn)
        messages = [str(w.message) for w in advisories]
        assert len(messages) == 2, messages
        assert len(set(messages)) == 2, messages
        assert any(m.startswith("Scan velocity 6.0 deg/s") for m in messages), messages
        assert any(m.startswith("Azimuth-coordinate velocity") for m in messages), messages
        for w in advisories:
            assert (
                Path(w.filename).as_posix().endswith("fyst_trajectories/planning/daisy_altaz.py")
            ), w.filename
