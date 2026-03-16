"""Tests for PongScanPattern."""

import math

import numpy as np
import pytest
from astropy.time import Time

from fyst_pointing.patterns import PongScanConfig, PongScanPattern


class TestPongScanPattern:
    """Tests for Pong scan pattern."""

    def test_basic_pong_scan(self, site):
        """Test generating a basic Pong scan pattern."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = PongScanConfig(
            timestep=0.1,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        assert trajectory.n_points > 0
        assert trajectory.duration == pytest.approx(60.0, abs=0.2)
        assert trajectory.start_time == start_time
        assert trajectory.pattern_type == "pong"
        assert trajectory.center_ra == 180.0
        assert trajectory.center_dec == -30.0
        # Verify new coordsys field is set
        assert trajectory.coordsys == "altaz"
        # Verify input_frame is set in metadata
        assert trajectory.metadata.input_frame == "icrs"

    def test_pong_covers_expected_region(self, site):
        """Test that Pong pattern covers approximately the expected region."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = PongScanConfig(
            timestep=0.1,
            width=1.0,
            height=1.0,
            spacing=0.1,
            velocity=0.3,
            num_terms=4,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=300.0, start_time=start_time)

        az_range = trajectory.az.max() - trajectory.az.min()
        el_range = trajectory.el.max() - trajectory.el.min()

        assert az_range > 0.5
        assert el_range > 0.5

    def test_pong_smooth_velocities(self, site):
        """Test that Pong pattern has smooth velocities."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = PongScanConfig(
            timestep=0.1,
            width=1.0,
            height=1.0,
            spacing=0.1,
            velocity=0.3,
            num_terms=4,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        assert np.abs(trajectory.az_vel).max() < 2.0
        assert np.abs(trajectory.el_vel).max() < 2.0

        dt = trajectory.times[1] - trajectory.times[0]
        az_accel = np.diff(trajectory.az_vel) / dt
        el_accel = np.diff(trajectory.el_vel) / dt

        assert np.abs(az_accel).max() < 10.0
        assert np.abs(el_accel).max() < 10.0

    def test_pong_with_rotation(self, site):
        """Test Pong pattern with non-zero rotation angle."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config_no_rot = PongScanConfig(
            timestep=0.1,
            width=1.0,
            height=1.0,
            spacing=0.1,
            velocity=0.3,
            num_terms=4,
            angle=0.0,
        )
        config_with_rot = PongScanConfig(
            timestep=0.1,
            width=1.0,
            height=1.0,
            spacing=0.1,
            velocity=0.3,
            num_terms=4,
            angle=45.0,
        )

        pattern_no_rot = PongScanPattern(ra=180.0, dec=-30.0, config=config_no_rot)
        pattern_with_rot = PongScanPattern(ra=180.0, dec=-30.0, config=config_with_rot)

        traj_no_rot = pattern_no_rot.generate(site, duration=60.0, start_time=start_time)
        traj_with_rot = pattern_with_rot.generate(site, duration=60.0, start_time=start_time)

        assert not np.allclose(traj_no_rot.az, traj_with_rot.az)

    def test_pong_metadata_stored(self, site):
        """Test that Pong pattern stores metadata correctly."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = PongScanConfig(
            timestep=0.1,
            width=2.0,
            height=1.5,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=30.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        assert trajectory.pattern_params is not None
        params = trajectory.pattern_params
        assert params["width"] == 2.0
        assert params["height"] == 1.5
        assert params["spacing"] == 0.1
        assert params["velocity"] == 0.5
        assert params["num_terms"] == 4
        assert params["angle"] == 30.0
        assert "period" in params
        assert "x_numvert" in params
        assert "y_numvert" in params

    def test_pong_narrow_pattern(self, site):
        """Test Pong pattern with very different width and height."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = PongScanConfig(
            timestep=0.1,
            width=3.0,
            height=0.5,
            spacing=0.1,
            velocity=0.3,
            num_terms=4,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=120.0, start_time=start_time)

        assert trajectory.n_points > 0
        assert np.all(np.isfinite(trajectory.az))
        assert np.all(np.isfinite(trajectory.el))


class TestPongVertexComputation:
    """Tests for Pong vertex computation algorithm."""

    def test_vertices_are_coprime(self):
        """Test that computed vertex counts are coprime."""
        config = PongScanConfig(
            timestep=0.1,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)

        x_numvert, y_numvert, _, _ = pattern._compute_vertices()

        assert math.gcd(x_numvert, y_numvert) == 1

    def test_vertices_have_opposite_parity(self):
        """Test that vertex counts have opposite parity."""
        config = PongScanConfig(
            timestep=0.1,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)

        x_numvert, y_numvert, _, _ = pattern._compute_vertices()

        assert (x_numvert % 2) != (y_numvert % 2)


class TestPongScanFlags:
    """Tests for scan flag behavior on Pong trajectories."""

    def test_pong_trajectory_no_flags(self, site):
        """Pong trajectory should have scan_flag=None (continuous pattern)."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = PongScanConfig(
            timestep=0.1,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)
        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        assert trajectory.scan_flag is None
        # science_mask should be all True when scan_flag is None
        assert np.all(trajectory.science_mask)
