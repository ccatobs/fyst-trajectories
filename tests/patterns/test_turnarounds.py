"""Tests for quintic turnaround profile."""

import numpy as np
import pytest

from fyst_trajectories.patterns.turnarounds import quintic_turnaround


class TestQuinticTurnaround:
    """Tests for the quintic polynomial turnaround."""

    @pytest.fixture
    def params(self):
        """Return standard test parameters."""
        return {"v": 2.0, "T": 4.0}

    def test_boundary_positions(self, params):
        """Position is zero at t=0 and t=T."""
        t = np.array([0.0, params["T"]])
        pos, _ = quintic_turnaround(t, params["v"], params["T"])
        np.testing.assert_allclose(pos[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(pos[-1], 0.0, atol=1e-12)

    def test_boundary_velocities(self, params):
        """Velocity is +v at t=0 and -v at t=T."""
        t = np.array([0.0, params["T"]])
        _, vel = quintic_turnaround(t, params["v"], params["T"])
        np.testing.assert_allclose(vel[0], params["v"], atol=1e-12)
        np.testing.assert_allclose(vel[-1], -params["v"], atol=1e-12)

    def test_zero_acceleration_at_boundaries(self, params):
        """Acceleration (numerical derivative of velocity) is zero at boundaries."""
        T = params["T"]
        dt = 1e-7
        t_start = np.array([0.0, dt])
        t_end = np.array([T - dt, T])

        _, vel_start = quintic_turnaround(t_start, params["v"], T)
        _, vel_end = quintic_turnaround(t_end, params["v"], T)

        accel_start = (vel_start[1] - vel_start[0]) / dt
        accel_end = (vel_end[1] - vel_end[0]) / dt

        np.testing.assert_allclose(accel_start, 0.0, atol=1e-4)
        np.testing.assert_allclose(accel_end, 0.0, atol=1e-4)

    def test_peak_displacement(self, params):
        """Peak displacement at t=T/2 equals 5*v*T/16."""
        T = params["T"]
        v = params["v"]
        t = np.array([T / 2.0])
        pos, _ = quintic_turnaround(t, v, T)
        expected = 5.0 * v * T / 16.0
        np.testing.assert_allclose(pos[0], expected, rtol=1e-12)

    def test_velocity_zero_at_midpoint(self, params):
        """Velocity passes through zero at t=T/2."""
        t = np.array([params["T"] / 2.0])
        _, vel = quintic_turnaround(t, params["v"], params["T"])
        np.testing.assert_allclose(vel[0], 0.0, atol=1e-12)

    def test_position_symmetry(self, params):
        """Position is symmetric: p(t) = p(T-t)."""
        T = params["T"]
        t = np.linspace(0, T, 101)
        pos, _ = quintic_turnaround(t, params["v"], T)
        pos_rev = pos[::-1]
        np.testing.assert_allclose(pos, pos_rev, atol=1e-12)

    def test_velocity_antisymmetry(self, params):
        """Velocity is antisymmetric: v(t) = -v(T-t)."""
        T = params["T"]
        t = np.linspace(0, T, 101)
        _, vel = quintic_turnaround(t, params["v"], T)
        vel_rev = vel[::-1]
        np.testing.assert_allclose(vel, -vel_rev, atol=1e-12)

    def test_position_nonnegative(self, params):
        """Position stays non-negative throughout turnaround."""
        T = params["T"]
        t = np.linspace(0, T, 1001)
        pos, _ = quintic_turnaround(t, params["v"], T)
        assert np.all(pos >= -1e-12)

    def test_peak_acceleration(self, params):
        """Peak acceleration is 1.5 * a_avg = 1.5 * (2*v/T)."""
        T = params["T"]
        v = params["v"]
        t = np.linspace(0, T, 10001)
        _, vel = quintic_turnaround(t, v, T)
        dt = t[1] - t[0]
        accel = np.diff(vel) / dt
        a_avg = 2.0 * v / T
        np.testing.assert_allclose(np.max(np.abs(accel)), 1.5 * a_avg, rtol=1e-3)
