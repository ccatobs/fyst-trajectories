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
        """Test that azimuth stays within scan bounds."""
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

        trajectory = pattern.generate(site, duration=120.0, start_time=None)

        assert trajectory.az.min() >= 100.0 - 0.05
        assert trajectory.az.max() <= 150.0 + 0.05

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

        trajectory = pattern.generate(site, duration=60.0, start_time=None)

        assert trajectory.n_points > 0
        assert trajectory.az.min() >= 120.0 - 0.05
        assert trajectory.az.max() <= 180.0 + 0.05

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

        assert np.abs(acceleration).max() <= 1.1

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

    def test_trapezoidal_velocity_profile(self, site):
        """Test that velocity profile is trapezoidal."""
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

    def test_small_throw_triangular_profile(self, site):
        """Test that small throws use triangular velocity profile."""
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

        assert np.abs(trajectory.az_vel).max() < 2.0
        assert np.abs(trajectory.az_vel).max() < 1.5


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
        - All positions within [az_start, az_stop] bounds (within tolerance)
        - All velocities <= az_speed (within tolerance)
        - Trajectory starts at az_start
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

        # All positions within bounds (with 0.05 deg tolerance)
        assert trajectory.az.min() >= az_min - 0.05, (
            f"Position {trajectory.az.min():.4f} below bound {az_min}"
        )
        assert trajectory.az.max() <= az_max + 0.05, (
            f"Position {trajectory.az.max():.4f} above bound {az_max}"
        )

        # All velocities bounded by configured speed (5% tolerance)
        assert np.abs(trajectory.az_vel).max() <= az_speed * 1.05, (
            f"Velocity {np.abs(trajectory.az_vel).max():.4f} exceeds speed {az_speed}"
        )

        # Trajectory starts at az_start
        assert trajectory.az[0] == pytest.approx(az_start, abs=0.05), (
            f"Start position {trajectory.az[0]:.4f} != {az_start}"
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

    def test_ce_turnaround_at_edges(self, site):
        """Turnaround flags should appear near azimuth extremes."""
        trajectory = self._make_trajectory(site)

        az_min = trajectory.az.min()
        az_max = trajectory.az.max()
        az_range = az_max - az_min

        # Samples near the edges (within 10% of range) should be turnaround
        near_min = trajectory.az < az_min + 0.1 * az_range
        near_max = trajectory.az > az_max - 0.1 * az_range
        near_edges = near_min | near_max

        # At least some edge samples should be flagged as turnaround
        turnaround_at_edges = near_edges & (trajectory.scan_flag == SCAN_FLAG_TURNAROUND)
        assert turnaround_at_edges.sum() > 0

    def test_triangular_profile_all_turnaround(self, site):
        """Small throws with triangular profile should be all turnaround."""
        trajectory = self._make_trajectory(
            site, az_start=100.0, az_stop=102.0, az_speed=2.0, az_accel=1.0
        )

        assert trajectory.scan_flag is not None
        assert np.all(trajectory.scan_flag == SCAN_FLAG_TURNAROUND)
