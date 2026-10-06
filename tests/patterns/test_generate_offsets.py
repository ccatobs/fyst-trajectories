"""Tests for generate_offsets() on PongScanPattern and DaisyScanPattern."""

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories.patterns import (
    DaisyScanConfig,
    DaisyScanPattern,
    PongScanConfig,
    PongScanPattern,
)


class TestPongGenerateOffsets:
    """The Pong offset-frame API: shapes, time span, magnitudes, and duration refusals."""

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
        times, x_off, y_off = pong_pattern.generate_offsets(duration=60.0)

        assert isinstance(times, np.ndarray)
        assert isinstance(x_off, np.ndarray)
        assert isinstance(y_off, np.ndarray)
        assert len(times) == len(x_off) == len(y_off)
        assert len(times) > 0

    def test_times_span_duration(self, pong_pattern):
        duration = 60.0
        times, _, _ = pong_pattern.generate_offsets(duration=duration)

        assert times[0] == pytest.approx(0.0)
        assert times[-1] == pytest.approx(duration)

    def test_offsets_in_reasonable_degree_range(self, pong_pattern):
        _, x_off, y_off = pong_pattern.generate_offsets(duration=60.0)

        # For a 2x2 degree scan, offsets should be within a few degrees
        assert np.abs(x_off).max() < 5.0
        assert np.abs(y_off).max() < 5.0
        # But they should actually cover some area
        assert np.abs(x_off).max() > 0.1
        assert np.abs(y_off).max() > 0.1

    def test_generate_uses_generate_offsets(self, pong_pattern, site):
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        duration = 60.0

        times, x_off, y_off = pong_pattern.generate_offsets(duration=duration)
        trajectory = pong_pattern.generate(site, duration=duration, start_time=start_time)

        np.testing.assert_array_equal(trajectory.times, times)
        assert trajectory.n_points == len(times)

    def test_generate_offsets_negative_duration_raises(self, pong_pattern):
        with pytest.raises(ValueError, match="fewer than 2 samples"):
            pong_pattern.generate_offsets(-1.0)

    def test_generate_offsets_zero_duration_raises(self, pong_pattern):
        with pytest.raises(ValueError, match="fewer than 2 samples"):
            pong_pattern.generate_offsets(0.0)

    def test_rotation_applied(self):
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

        # ``angle`` rotates the unrotated path counter-clockwise, x toward y.
        cos_a, sin_a = np.cos(np.radians(45.0)), np.sin(np.radians(45.0))
        np.testing.assert_allclose(x45, x0 * cos_a - y0 * sin_a, atol=1e-12)
        np.testing.assert_allclose(y45, x0 * sin_a + y0 * cos_a, atol=1e-12)


class TestDaisyGenerateOffsets:
    """The Daisy offset-frame API: shapes, the uniform time grid, and duration refusals."""

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
        times, x_off, y_off = daisy_pattern.generate_offsets(duration=60.0)

        assert isinstance(times, np.ndarray)
        assert isinstance(x_off, np.ndarray)
        assert isinstance(y_off, np.ndarray)
        assert len(times) == len(x_off) == len(y_off)
        assert len(times) > 0

    def test_times_span_duration(self, daisy_pattern):
        """Times start at 0 and step uniformly at config.timestep up to ~duration.

        The Daisy integrator samples at fixed ``timestep`` intervals, so the
        last sample lands at ``duration - timestep`` (not exactly ``duration``),
        and the grid must be uniform (no ``linspace`` stretch).
        """
        duration = 60.0
        timestep = daisy_pattern.config.timestep
        times, _, _ = daisy_pattern.generate_offsets(duration=duration)

        assert times[0] == pytest.approx(0.0)
        assert times[-1] == pytest.approx(duration - timestep)
        assert np.allclose(np.diff(times), timestep, rtol=0, atol=1e-9)

    def test_offsets_in_reasonable_degree_range(self, daisy_pattern):
        _, x_off, y_off = daisy_pattern.generate_offsets(duration=120.0)

        # The rosette stays inside the radius plus the turn-radius overshoot, and
        # must actually cover some area.
        envelope = daisy_pattern.config.radius + 2.0 * daisy_pattern.config.turn_radius
        assert 0.1 < np.abs(x_off).max() < envelope
        assert 0.1 < np.abs(y_off).max() < envelope

    def test_generate_uses_generate_offsets(self, daisy_pattern, site):
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        duration = 60.0

        times, x_off, y_off = daisy_pattern.generate_offsets(duration=duration)
        trajectory = daisy_pattern.generate(site, duration=duration, start_time=start_time)

        np.testing.assert_array_equal(trajectory.times, times)
        assert trajectory.n_points == len(times)

    def test_generate_offsets_negative_duration_raises(self, daisy_pattern):
        with pytest.raises(ValueError, match="fewer than 2 samples"):
            daisy_pattern.generate_offsets(-1.0)

    def test_generate_offsets_zero_duration_raises(self, daisy_pattern):
        with pytest.raises(ValueError, match="fewer than 2 samples"):
            daisy_pattern.generate_offsets(0.0)

    def test_y_offset_affects_offsets(self):
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

        # ``y_offset`` is where the rosette starts on the y axis.
        assert y0[0] == pytest.approx(0.0)
        assert y_off[0] == pytest.approx(0.2)
        assert not np.allclose(y0, y_off)
