"""Tests for generate_offsets() on PongScanPattern and DaisyScanPattern."""

import numpy as np
import pytest
from astropy.time import Time

from fyst_pointing.patterns import (
    DaisyScanConfig,
    DaisyScanPattern,
    PongScanConfig,
    PongScanPattern,
)


class TestPongGenerateOffsets:
    """Tests for PongScanPattern.generate_offsets()."""

    @pytest.fixture
    def pong_pattern(self):
        """Create a standard Pong pattern for testing."""
        config = PongScanConfig(
            timestep=0.1,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        )
        return PongScanPattern(ra=180.0, dec=-30.0, config=config)

    def test_returns_three_equal_length_arrays(self, pong_pattern):
        """Test that generate_offsets returns 3 arrays of equal length."""
        times, x_off, y_off = pong_pattern.generate_offsets(duration=60.0)

        assert isinstance(times, np.ndarray)
        assert isinstance(x_off, np.ndarray)
        assert isinstance(y_off, np.ndarray)
        assert len(times) == len(x_off) == len(y_off)
        assert len(times) > 0

    def test_times_span_duration(self, pong_pattern):
        """Test that times array spans the requested duration."""
        duration = 60.0
        times, _, _ = pong_pattern.generate_offsets(duration=duration)

        assert times[0] == pytest.approx(0.0)
        assert times[-1] == pytest.approx(duration)

    def test_offsets_in_reasonable_degree_range(self, pong_pattern):
        """Test that offsets are in degrees with reasonable magnitude."""
        _, x_off, y_off = pong_pattern.generate_offsets(duration=60.0)

        # For a 2x2 degree scan, offsets should be within a few degrees
        assert np.abs(x_off).max() < 5.0
        assert np.abs(y_off).max() < 5.0
        # But they should actually cover some area
        assert np.abs(x_off).max() > 0.1
        assert np.abs(y_off).max() > 0.1

    def test_all_values_finite(self, pong_pattern):
        """Test that all returned values are finite."""
        times, x_off, y_off = pong_pattern.generate_offsets(duration=60.0)

        assert np.all(np.isfinite(times))
        assert np.all(np.isfinite(x_off))
        assert np.all(np.isfinite(y_off))

    def test_generate_uses_generate_offsets(self, pong_pattern, site):
        """Test that generate() produces results consistent with generate_offsets()."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        duration = 60.0

        times, x_off, y_off = pong_pattern.generate_offsets(duration=duration)
        trajectory = pong_pattern.generate(site, duration=duration, start_time=start_time)

        # Times arrays should match
        np.testing.assert_array_equal(trajectory.times, times)
        # Same number of points
        assert trajectory.n_points == len(times)

    def test_generate_still_works(self, pong_pattern, site):
        """Regression test: generate() still produces valid trajectories."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        trajectory = pong_pattern.generate(site, duration=60.0, start_time=start_time)

        assert trajectory.n_points > 0
        assert trajectory.duration == pytest.approx(60.0, abs=0.2)
        assert trajectory.pattern_type == "pong"
        assert trajectory.coordsys == "altaz"
        assert np.all(np.isfinite(trajectory.az))
        assert np.all(np.isfinite(trajectory.el))

    def test_generate_offsets_negative_duration_raises(self, pong_pattern):
        """Test that a negative duration raises ValueError."""
        with pytest.raises(ValueError, match="duration must be positive"):
            pong_pattern.generate_offsets(-1.0)

    def test_generate_offsets_zero_duration_raises(self, pong_pattern):
        """Test that zero duration raises ValueError."""
        with pytest.raises(ValueError, match="duration must be positive"):
            pong_pattern.generate_offsets(0.0)

    def test_rotation_applied(self):
        """Test that rotation angle affects offsets."""
        config_0 = PongScanConfig(
            timestep=0.1,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        )
        config_45 = PongScanConfig(
            timestep=0.1,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=45.0,
        )
        p0 = PongScanPattern(ra=180.0, dec=-30.0, config=config_0)
        p45 = PongScanPattern(ra=180.0, dec=-30.0, config=config_45)

        _, x0, y0 = p0.generate_offsets(duration=60.0)
        _, x45, y45 = p45.generate_offsets(duration=60.0)

        assert not np.allclose(x0, x45)


class TestDaisyGenerateOffsets:
    """Tests for DaisyScanPattern.generate_offsets()."""

    @pytest.fixture
    def daisy_pattern(self):
        """Create a standard Daisy pattern for testing."""
        config = DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.0,
        )
        return DaisyScanPattern(ra=180.0, dec=-30.0, config=config)

    def test_returns_three_equal_length_arrays(self, daisy_pattern):
        """Test that generate_offsets returns 3 arrays of equal length."""
        times, x_off, y_off = daisy_pattern.generate_offsets(duration=60.0)

        assert isinstance(times, np.ndarray)
        assert isinstance(x_off, np.ndarray)
        assert isinstance(y_off, np.ndarray)
        assert len(times) == len(x_off) == len(y_off)
        assert len(times) > 0

    def test_times_span_duration(self, daisy_pattern):
        """Test that times array spans the requested duration."""
        duration = 60.0
        times, _, _ = daisy_pattern.generate_offsets(duration=duration)

        assert times[0] == pytest.approx(0.0)
        assert times[-1] == pytest.approx(duration)

    def test_offsets_in_reasonable_degree_range(self, daisy_pattern):
        """Test that offsets are in degrees with reasonable magnitude."""
        _, x_off, y_off = daisy_pattern.generate_offsets(duration=120.0)

        # For a 0.5 degree radius daisy, offsets should stay within a few degrees
        assert np.abs(x_off).max() < 5.0
        assert np.abs(y_off).max() < 5.0

    def test_all_values_finite(self, daisy_pattern):
        """Test that all returned values are finite."""
        times, x_off, y_off = daisy_pattern.generate_offsets(duration=60.0)

        assert np.all(np.isfinite(times))
        assert np.all(np.isfinite(x_off))
        assert np.all(np.isfinite(y_off))

    def test_generate_uses_generate_offsets(self, daisy_pattern, site):
        """Test that generate() produces results consistent with generate_offsets()."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        duration = 60.0

        times, x_off, y_off = daisy_pattern.generate_offsets(duration=duration)
        trajectory = daisy_pattern.generate(site, duration=duration, start_time=start_time)

        # Times arrays should match
        np.testing.assert_array_equal(trajectory.times, times)
        # Same number of points
        assert trajectory.n_points == len(times)

    def test_generate_still_works(self, daisy_pattern, site):
        """Regression test: generate() still produces valid trajectories."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        trajectory = daisy_pattern.generate(site, duration=60.0, start_time=start_time)

        assert trajectory.n_points > 0
        assert trajectory.duration == pytest.approx(60.0, abs=0.2)
        assert trajectory.pattern_type == "daisy"
        assert trajectory.coordsys == "altaz"
        assert np.all(np.isfinite(trajectory.az))
        assert np.all(np.isfinite(trajectory.el))

    def test_generate_offsets_negative_duration_raises(self, daisy_pattern):
        """Test that a negative duration raises ValueError."""
        with pytest.raises(ValueError, match="duration must be positive"):
            daisy_pattern.generate_offsets(-1.0)

    def test_generate_offsets_zero_duration_raises(self, daisy_pattern):
        """Test that zero duration raises ValueError."""
        with pytest.raises(ValueError, match="duration must be positive"):
            daisy_pattern.generate_offsets(0.0)

    def test_y_offset_affects_offsets(self):
        """Test that y_offset parameter affects the generated offsets."""
        config_0 = DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.0,
        )
        config_offset = DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.2,
        )
        p0 = DaisyScanPattern(ra=180.0, dec=-30.0, config=config_0)
        p_off = DaisyScanPattern(ra=180.0, dec=-30.0, config=config_offset)

        _, x0, y0 = p0.generate_offsets(duration=60.0)
        _, x_off, y_off = p_off.generate_offsets(duration=60.0)

        assert not np.allclose(y0, y_off)
