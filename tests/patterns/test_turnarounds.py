"""Tests for quintic turnaround profile."""

import numpy as np
import pytest

from fyst_trajectories.patterns.turnarounds import (
    quintic_turnaround,
    turnaround_duration_sec,
)
from fyst_trajectories.planning._ce_geometry import _quantize_ce_duration


class TestQuinticTurnaround:
    """Boundary conditions, symmetry, and the 5*v*T/16 peak displacement of the quintic."""

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


class TestTurnaroundDurationIsShared:
    """One definition of ``T = 2 * v / a``, used by the generator and the quantiser."""

    @pytest.mark.parametrize(
        "az_speed,az_accel",
        [(1.5, 1.0), (1.0, 0.75), (3.0, 1.5), (0.25, 1.0)],
    )
    def test_the_quantiser_counts_the_same_turnaround_the_generator_emits(self, az_speed, az_accel):
        """A leg count derived from a different turnaround overruns its window.

        The quantiser's leg arithmetic and the generated trajectory have to
        price a reversal identically; deriving the count from the cruise time
        alone quantises a 300 s window to 848 s for a 2.44 deg leg at
        1.5 deg/s and 1.0 deg/s^2.
        """
        expected = turnaround_duration_sec(az_speed, az_accel)
        assert expected == pytest.approx(2.0 * az_speed / az_accel)

        # One leg plus one turnaround, exactly: the quantiser must return two
        # legs and a duration built from the same turnaround.
        az_throw = 10.0
        t_cruise = az_throw / az_speed
        n_scans, actual = _quantize_ce_duration(
            az_throw=az_throw,
            velocity=az_speed,
            duration=2.0 * t_cruise + expected,
            az_accel=az_accel,
        )
        assert n_scans == 2
        assert actual == pytest.approx(2.0 * t_cruise + expected)
