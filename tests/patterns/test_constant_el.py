"""Tests for ConstantElScanPattern."""

import numpy as np
import pytest
from astropy.time import Time
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fyst_trajectories.patterns import ConstantElScanConfig, ConstantElScanPattern
from fyst_trajectories.trajectory import SCAN_FLAG_SCIENCE, SCAN_FLAG_TURNAROUND


class TestConstantElScanPattern:
    """Tests for constant elevation scan pattern."""

    def test_basic_scan(self, site):
        """Test generating a basic constant elevation scan."""
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=120.0,
            az_stop=180.0,
            elevation=45.0,
            az_speed=1.0,
            az_accel=0.5,
            n_scans=1,
        )
        pattern = ConstantElScanPattern(config)

        trajectory = pattern.generate(site, duration=60.0, start_time=None)

        assert trajectory.n_points > 0
        assert trajectory.duration == pytest.approx(60.0, abs=0.2)
        assert trajectory.pattern_type == "constant_el"

    def test_constant_elevation(self, site):
        """Test that elevation stays constant throughout scan."""
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=100.0,
            az_stop=150.0,
            elevation=50.0,
            az_speed=2.0,
            az_accel=0.5,
            n_scans=1,
        )
        pattern = ConstantElScanPattern(config)

        trajectory = pattern.generate(site, duration=30.0, start_time=None)

        np.testing.assert_array_almost_equal(
            trajectory.el, np.full_like(trajectory.el, 50.0), decimal=5
        )

    def test_azimuth_range(self, site):
        """Test that azimuth extends beyond science bounds."""
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=100.0,
            az_stop=150.0,
            elevation=45.0,
            az_speed=1.0,
            az_accel=0.5,
            n_scans=1,
        )
        pattern = ConstantElScanPattern(config)
        d_half_turn = 5 * config.az_speed**2 / (8 * config.az_accel)

        trajectory = pattern.generate(site, duration=120.0, start_time=None)

        # With overscan, positions extend beyond science bounds by d_half_turn
        assert trajectory.az.min() >= 100.0 - d_half_turn - 0.05
        assert trajectory.az.max() <= 150.0 + d_half_turn + 0.05
        # But should actually reach the overscan zones
        assert trajectory.az.min() < 100.0
        assert trajectory.az.max() > 150.0

    def test_velocity_bounds(self, site):
        """Test that velocities don't exceed limits."""
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=100.0,
            az_stop=150.0,
            elevation=45.0,
            az_speed=1.0,
            az_accel=0.5,
            n_scans=1,
        )
        pattern = ConstantElScanPattern(config)

        trajectory = pattern.generate(site, duration=60.0, start_time=None)

        np.testing.assert_array_almost_equal(
            trajectory.el_vel, np.zeros_like(trajectory.el_vel), decimal=5
        )

        # Velocity should be within 5% of configured speed
        assert np.abs(trajectory.az_vel).max() <= 1.0 * 1.05

    def test_scan_direction_reverse(self, site):
        """Test scan in reverse direction (az_start > az_stop)."""
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=180.0,
            az_stop=120.0,
            elevation=45.0,
            az_speed=1.0,
            az_accel=0.5,
            n_scans=1,
        )
        pattern = ConstantElScanPattern(config)
        d_half_turn = 5 * config.az_speed**2 / (8 * config.az_accel)

        trajectory = pattern.generate(site, duration=60.0, start_time=None)

        assert trajectory.n_points > 0
        # With overscan, positions extend beyond science bounds
        assert trajectory.az.min() >= 120.0 - d_half_turn - 0.05
        assert trajectory.az.max() <= 180.0 + d_half_turn + 0.05

    def test_metadata(self, site):
        """Test that metadata is correctly populated."""
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=100.0,
            az_stop=150.0,
            elevation=45.0,
            az_speed=1.0,
            az_accel=0.5,
            n_scans=2,
        )
        pattern = ConstantElScanPattern(config)

        metadata = pattern.get_metadata()

        assert metadata.pattern_type == "constant_el"
        assert metadata.pattern_params["az_start"] == 100.0
        assert metadata.pattern_params["az_stop"] == 150.0
        assert metadata.pattern_params["elevation"] == 45.0
        assert metadata.pattern_params["az_speed"] == 1.0
        assert metadata.pattern_params["n_scans"] == 2


class TestTurnaroundBehavior:
    """Tests for smooth turnaround behavior."""

    def test_velocity_is_continuous(self, site):
        """Test that velocity changes smoothly."""
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=100.0,
            az_stop=150.0,
            elevation=45.0,
            az_speed=2.0,
            az_accel=1.0,
            n_scans=1,
        )
        pattern = ConstantElScanPattern(config)

        trajectory = pattern.generate(site, duration=200.0, start_time=None)

        dv = np.diff(trajectory.az_vel)
        dt = np.diff(trajectory.times)
        acceleration = dv / dt

        # Quintic turnaround has peak acceleration of 1.5x the average
        assert np.abs(acceleration).max() <= 1.5 * 1.1

    def test_velocity_passes_through_zero(self, site):
        """Test that velocity passes through zero at turnarounds."""
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=100.0,
            az_stop=120.0,
            elevation=45.0,
            az_speed=2.0,
            az_accel=1.0,
            n_scans=1,
        )
        pattern = ConstantElScanPattern(config)

        trajectory = pattern.generate(site, duration=100.0, start_time=None)

        sign_changes = np.where(np.diff(np.sign(trajectory.az_vel)))[0]

        assert len(sign_changes) > 0

        for idx in sign_changes:
            v_near_turnaround = min(abs(trajectory.az_vel[idx]), abs(trajectory.az_vel[idx + 1]))
            assert v_near_turnaround < 0.5

    def test_cruise_velocity_profile(self, site):
        """Test that cruise segments reach full speed."""
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=100.0,
            az_stop=150.0,
            elevation=45.0,
            az_speed=2.0,
            az_accel=1.0,
            n_scans=1,
        )
        pattern = ConstantElScanPattern(config)

        trajectory = pattern.generate(site, duration=60.0, start_time=None)

        at_cruise = np.abs(np.abs(trajectory.az_vel) - 2.0) < 0.1
        assert np.sum(at_cruise) > 0

        assert np.abs(trajectory.az_vel).max() <= 2.01

    def test_small_throw_reaches_cruise(self, site):
        """Even small throws reach full cruise speed with overscan."""
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=100.0,
            az_stop=102.0,
            elevation=45.0,
            az_speed=2.0,
            az_accel=1.0,
            n_scans=1,
        )
        pattern = ConstantElScanPattern(config)

        trajectory = pattern.generate(site, duration=30.0, start_time=None)

        # With overscan, the cruise covers the full 2-degree science region
        at_cruise = np.abs(np.abs(trajectory.az_vel) - 2.0) < 0.1
        assert np.sum(at_cruise) > 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_short_scan(self, site):
        """Test generating a very short scan."""
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=100.0,
            az_stop=101.0,
            elevation=45.0,
            az_speed=1.0,
            az_accel=0.5,
            n_scans=1,
        )
        pattern = ConstantElScanPattern(config)

        trajectory = pattern.generate(site, duration=10.0, start_time=None)
        assert trajectory.n_points > 0

    def test_very_slow_scan(self, site):
        """Test generating a very slow scan."""
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=100.0,
            az_stop=150.0,
            elevation=45.0,
            az_speed=0.1,
            az_accel=0.5,
            n_scans=1,
        )
        pattern = ConstantElScanPattern(config)

        trajectory = pattern.generate(site, duration=60.0, start_time=None)
        assert trajectory.n_points > 0

    def test_with_start_time(self, site):
        """Test generating scan with explicit start time."""
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=100.0,
            az_stop=150.0,
            elevation=45.0,
            az_speed=1.0,
            az_accel=0.5,
            n_scans=1,
        )
        pattern = ConstantElScanPattern(config)
        start_time = Time("2026-03-15T04:00:00", scale="utc")

        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        assert trajectory.start_time == start_time


class TestConstantElPropertyBased:
    """Property-based tests for constant elevation scan pattern."""

    @given(
        az_start=st.floats(min_value=-170.0, max_value=170.0),
        az_throw=st.floats(min_value=1.0, max_value=80.0),
        elevation=st.floats(min_value=25.0, max_value=85.0),
        az_speed=st.floats(min_value=0.1, max_value=3.0),
        az_accel=st.floats(min_value=0.1, max_value=2.0),
        duration=st.floats(min_value=5.0, max_value=300.0),
    )
    @settings(
        max_examples=50,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_invariants(self, site, az_start, az_throw, elevation, az_speed, az_accel, duration):
        """Test invariants hold for random valid parameters.

        Invariants checked:
        - All positions within motion range (science + overscan, within tolerance)
        - All velocities <= az_speed (within tolerance)
        - Trajectory starts at the overscan edge (az_start - d_half_turn for
          start_increasing, or az_start + d_half_turn otherwise)
        - Duration matches requested duration
        """
        az_stop = az_start + az_throw

        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=az_start,
            az_stop=az_stop,
            elevation=elevation,
            az_speed=az_speed,
            az_accel=az_accel,
            n_scans=1,
        )
        pattern = ConstantElScanPattern(config)
        trajectory = pattern.generate(site, duration=duration, start_time=None)

        az_min = min(az_start, az_stop)
        az_max = max(az_start, az_stop)
        d_half_turn = 5 * az_speed**2 / (8 * az_accel)

        # All positions within motion range (science + overscan, with tolerance)
        assert trajectory.az.min() >= az_min - d_half_turn - 0.05, (
            f"Position {trajectory.az.min():.4f} below motion bound {az_min - d_half_turn}"
        )
        assert trajectory.az.max() <= az_max + d_half_turn + 0.05, (
            f"Position {trajectory.az.max():.4f} above motion bound {az_max + d_half_turn}"
        )

        # All velocities bounded by configured speed (5% tolerance)
        assert np.abs(trajectory.az_vel).max() <= az_speed * 1.05, (
            f"Velocity {np.abs(trajectory.az_vel).max():.4f} exceeds speed {az_speed}"
        )

        # Trajectory starts at the overscan edge
        expected_start = az_start - d_half_turn if az_start < az_stop else az_start + d_half_turn
        assert trajectory.az[0] == pytest.approx(expected_start, abs=0.05), (
            f"Start position {trajectory.az[0]:.4f} != {expected_start}"
        )

        # Duration matches requested
        assert trajectory.duration == pytest.approx(duration, abs=0.5)


class TestScanFlags:
    """Tests for turnaround flagging in constant elevation scans."""

    def _make_trajectory(self, site, az_start=100.0, az_stop=150.0, az_speed=2.0, az_accel=1.0):
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=az_start,
            az_stop=az_stop,
            elevation=45.0,
            az_speed=az_speed,
            az_accel=az_accel,
            n_scans=1,
        )
        pattern = ConstantElScanPattern(config)
        return pattern.generate(site, duration=120.0, start_time=None)

    def test_ce_trajectory_has_scan_flags(self, site):
        """CE trajectory should have scan_flag with only values 1 and 2."""
        trajectory = self._make_trajectory(site)

        assert trajectory.scan_flag is not None
        assert len(trajectory.scan_flag) == trajectory.n_points
        unique_flags = set(np.unique(trajectory.scan_flag))
        assert unique_flags <= {SCAN_FLAG_SCIENCE, SCAN_FLAG_TURNAROUND}

    def test_ce_science_mask(self, site):
        """science_mask should be True only for constant-velocity samples."""
        trajectory = self._make_trajectory(site)
        mask = trajectory.science_mask

        assert mask.dtype == bool
        assert mask.shape == (trajectory.n_points,)
        # Science mask matches scan_flag == 1
        np.testing.assert_array_equal(mask, trajectory.scan_flag == SCAN_FLAG_SCIENCE)
        # There should be both science and turnaround samples
        assert mask.sum() > 0
        assert (~mask).sum() > 0

    def test_ce_turnaround_outside_science(self, site):
        """With overscan, turnaround samples should be outside science bounds."""
        trajectory = self._make_trajectory(site)

        az_min = 100.0  # science region boundaries from _make_trajectory
        az_max = 150.0

        # All samples outside the science region should be turnaround
        outside_science = (trajectory.az < az_min) | (trajectory.az > az_max)
        if outside_science.sum() > 0:
            assert np.all(trajectory.scan_flag[outside_science] == SCAN_FLAG_TURNAROUND)

        # All science-flagged samples should be within science bounds
        science_mask = trajectory.scan_flag == SCAN_FLAG_SCIENCE
        if science_mask.sum() > 0:
            assert trajectory.az[science_mask].min() >= az_min - 0.01
            assert trajectory.az[science_mask].max() <= az_max + 0.01

    def test_overscan_science_at_cruise_velocity(self, site):
        """With overscan, all samples in the science region should be at cruise velocity."""
        trajectory = self._make_trajectory(site)
        az_speed = 2.0  # from _make_trajectory

        science_mask = trajectory.scan_flag == SCAN_FLAG_SCIENCE
        assert science_mask.sum() > 0

        # All science samples should be at cruise speed (within 5% tolerance)
        science_speeds = np.abs(trajectory.az_vel[science_mask])
        assert np.all(science_speeds > az_speed * 0.95)

    def test_overscan_extends_beyond_science(self, site):
        """With overscan, the trajectory should extend beyond the science region."""
        trajectory = self._make_trajectory(site)

        # Science region is [100, 150] from _make_trajectory defaults
        assert trajectory.az.min() < 100.0
        assert trajectory.az.max() > 150.0
