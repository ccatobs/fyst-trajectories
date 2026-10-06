"""Tests for DaisyScanPattern."""

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories import Coordinates
from fyst_trajectories.patterns import DaisyScanConfig, DaisyScanPattern
from fyst_trajectories.patterns.daisy import _DAISY_INTERNAL_TIMESTEP, _daisy_reach


class TestDaisyScanPattern:
    """Rosette generation: centre crossing, steady cruise speed, y_offset, small radius."""

    def test_basic_daisy_scan(self, site):
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.0,
        )
        pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=120.0, start_time=start_time)

        assert trajectory.duration == pytest.approx(120.0, abs=0.2)
        assert trajectory.start_time == start_time
        assert trajectory.pattern_type == "daisy"
        assert trajectory.center_ra == 180.0
        assert trajectory.center_dec == -30.0

    def test_daisy_crosses_center(self, site):
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.2,
            turn_radius=0.15,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.0,
        )
        pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=300.0, start_time=start_time)

        coords = Coordinates(site)
        center_az, center_el = coords.radec_to_altaz(180.0, -30.0, obstime=start_time)

        # On-sky offset frame: the az component must be scaled by cos(el) or the
        # metric over-weights azimuth at this declination. The rosette passes
        # essentially through the centre, so the closest approach is well under
        # a turn radius.
        dx = (trajectory.az - center_az) * np.cos(np.radians(trajectory.el))
        dy = trajectory.el - center_el
        min_distance = np.hypot(dx, dy).min()

        assert min_distance < 0.1

    def test_daisy_horizon_rate_stays_in_a_band(self, site):
        """Horizon-frame speed stays in a bounded band during the cruise.

        ``az_vel`` is a mount-frame rate inflated by ``1 / cos(el)``, so it is not
        constant even when the offset-frame speed is; this pins only that it does
        not wander. The offset-frame speed itself is pinned by
        ``TestDaisyTimeGrid::test_cruise_speed_recovers_velocity``.
        """
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.2,
            turn_radius=0.15,
            avoidance_radius=0.0,
            start_acceleration=1.0,
            y_offset=0.0,
        )
        pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=120.0, start_time=start_time)

        total_vel = np.sqrt(trajectory.az_vel**2 + trajectory.el_vel**2)

        ramp_time = config.velocity / config.start_acceleration
        ramp_samples = int(ramp_time / (trajectory.times[1] - trajectory.times[0])) + 5

        steady_state_vel = total_vel[ramp_samples:]

        vel_std = np.std(steady_state_vel)
        vel_mean = np.mean(steady_state_vel)

        assert vel_std / vel_mean < 0.5

    def test_daisy_with_offset(self, site):
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config_no_offset = DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.2,
            turn_radius=0.15,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.0,
        )
        config_with_offset = DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.2,
            turn_radius=0.15,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.2,
        )

        pattern_no_offset = DaisyScanPattern(ra=180.0, dec=-30.0, config=config_no_offset)
        pattern_with_offset = DaisyScanPattern(ra=180.0, dec=-30.0, config=config_with_offset)

        traj_no_offset = pattern_no_offset.generate(site, duration=60.0, start_time=start_time)
        traj_with_offset = pattern_with_offset.generate(site, duration=60.0, start_time=start_time)

        assert not np.allclose(traj_no_offset.az, traj_with_offset.az)

    def test_daisy_metadata_stored(self, site):
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.1,
            start_acceleration=0.5,
            y_offset=0.05,
        )
        pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        params = trajectory.pattern_params
        assert params["radius"] == 0.5
        assert params["velocity"] == 0.3
        assert params["turn_radius"] == 0.2
        assert params["avoidance_radius"] == 0.1
        assert params["start_acceleration"] == 0.5
        assert params["y_offset"] == 0.05

    def test_daisy_small_radius(self, site):
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = DaisyScanConfig(
            timestep=0.1,
            radius=0.1,
            velocity=0.1,
            turn_radius=0.05,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.0,
        )
        pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)

        pattern.generate(site, duration=60.0, start_time=start_time)

        # The small rosette stays inside radius + 2 * turn_radius (0.2 deg) and
        # still covers area (measured reach 0.16 deg).
        _, x_off, y_off = pattern.generate_offsets(60.0)
        reach = np.hypot(x_off, y_off).max()
        assert 0.05 < reach < config.radius + 2.0 * config.turn_radius


class TestDaisyScanFlags:
    """Daisy emits scan_flag, with the start ramp-up flagged as non-science."""

    def test_scan_flag_populated(self, site):
        """Daisy patterns populate ``Trajectory.scan_flag``."""
        from fyst_trajectories.trajectory import SCAN_FLAG_SCIENCE, SCAN_FLAG_TURNAROUND

        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.0,
        )
        pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=120.0, start_time=start_time)

        assert trajectory.scan_flag is not None
        assert trajectory.scan_flag.shape == trajectory.times.shape
        # The trajectory must contain both flag values
        assert np.any(trajectory.scan_flag == SCAN_FLAG_SCIENCE)
        assert np.any(trajectory.scan_flag == SCAN_FLAG_TURNAROUND)

    def test_initial_ramp_up_flagged_as_turnaround(self, site):
        """The first sample (during start_acceleration ramp-up) is non-science."""
        from fyst_trajectories.trajectory import SCAN_FLAG_TURNAROUND

        start_time = Time("2026-03-15T04:00:00", scale="utc")
        # A slow start_acceleration relative to velocity makes the ramp
        # take several timesteps so the test is not razor-thin.
        config = DaisyScanConfig(
            timestep=0.05,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.3,
            y_offset=0.0,
        )
        pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)
        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        assert trajectory.scan_flag[0] == SCAN_FLAG_TURNAROUND


class TestDaisyAvoidanceRadius:
    """A non-zero avoidance_radius holds the rosette out of a central keep-out."""

    def test_avoidance_radius_keeps_out_center(self):
        # Measured in the offset frame (free of tracking drift); y_offset starts
        # the path outside the keep-out so the start is not trivially at centre.
        common = dict(
            timestep=0.1,
            radius=1.0,
            velocity=0.3,
            turn_radius=0.2,
            start_acceleration=0.5,
            y_offset=0.5,
        )
        keepout = DaisyScanPattern(
            ra=180.0, dec=-30.0, config=DaisyScanConfig(avoidance_radius=0.3, **common)
        )
        no_keepout = DaisyScanPattern(
            ra=180.0, dec=-30.0, config=DaisyScanConfig(avoidance_radius=0.0, **common)
        )

        _, xk, yk = keepout.generate_offsets(600.0)
        _, x0, y0 = no_keepout.generate_offsets(600.0)

        # With the keep-out the closest approach to centre is ~avoidance_radius;
        # without it the path passes essentially through the centre.
        assert np.hypot(xk, yk).min() >= 0.3 - 0.02
        assert np.hypot(x0, y0).min() < 0.05


class TestDaisyTimeGrid:
    """The reported time grid matches the integrator grid (no linspace stretch).

    Samples are exactly ``config.timestep`` apart; a ``linspace`` re-labelling would
    stretch the axis (~1% at a 10 s scan), biasing every derived velocity. Mirrors
    ``test_constant_el.py``'s position/velocity-consistency guard.
    """

    def _config(self):
        return DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.0,
        )

    def test_time_grid_is_uniform_at_timestep(self):
        config = self._config()
        pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)
        times, _, _ = pattern.generate_offsets(duration=10.0)

        dt = np.diff(times)
        assert np.allclose(dt, config.timestep, rtol=0, atol=1e-9), (
            f"time grid not uniform at timestep: diff range "
            f"[{dt.min():.6f}, {dt.max():.6f}] vs timestep {config.timestep}"
        )

    def test_cruise_speed_recovers_velocity(self):
        config = self._config()
        pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)
        times, x, y = pattern.generate_offsets(duration=10.0)

        # Inter-sample (segment) speed ds/dt: directly probes whether the time
        # grid recovers the integrator's speed, without the second-derivative
        # discretization error that np.gradient adds on a curving path. A
        # stretched time axis scales every segment speed by the stretch factor.
        seg_speed = np.hypot(np.diff(x), np.diff(y)) / np.diff(times)
        # Cruise = every petal arc after the start ramp; the ramp is the only
        # sub-speed phase, so threshold near ``velocity``.
        cruise = seg_speed[seg_speed >= 0.95 * config.velocity]
        assert cruise.size > 10
        rel = abs(cruise.mean() - config.velocity) / config.velocity
        assert rel < 0.001, (
            f"cruise speed {cruise.mean():.5f} deg/s vs configured {config.velocity} "
            f"(relative error {rel:.4%}); time grid is stretching velocities"
        )


class TestDaisyReach:
    """``_daisy_reach`` bounds how far the petals go, which is past ``radius``.

    A petal turns on a circle of ``turn_radius`` after it crosses ``radius``,
    so it reaches ``hypot(radius, turn_radius) + turn_radius`` from the centre
    (1.08 deg for a 0.3 deg radius and the 0.5 deg turn radius the offline
    rebuild defaults to). The offline scheduler bounds a daisy visit by this
    reach, so an offset past it would let a booked scan leave the limits.
    """

    @pytest.mark.parametrize(
        "radius, turn_radius, avoidance_radius, velocity, timestep, y_offset",
        [
            (0.3, 0.5, 0.1, 0.3, 0.1, 0.0),  # turn radius larger than the radius
            (0.5, 0.2, 0.0, 0.3, 0.1, 0.0),
            (1.0, 0.5, 0.1, 1.0, 0.1, 0.0),  # fast: the last step before a turn is long
            (2.0, 0.5, 0.3, 1.0, 0.002, 0.0),  # output finer than the integrator's ceiling
            (0.5, 0.2, 0.1, 0.3, 0.1, 0.8),  # starts outside the radius: the start is farthest
        ],
    )
    def test_bounds_the_pattern_and_the_first_petal_reaches_it(
        self, radius, turn_radius, avoidance_radius, velocity, timestep, y_offset
    ):
        config = DaisyScanConfig(
            timestep=timestep,
            radius=radius,
            velocity=velocity,
            turn_radius=turn_radius,
            avoidance_radius=avoidance_radius,
            start_acceleration=0.5,
            y_offset=y_offset,
        )
        _, x_off, y_off = DaisyScanPattern(ra=0.0, dec=0.0, config=config).generate_offsets(120.0)
        farthest = float(np.hypot(x_off, y_off).max())

        reach = _daisy_reach(radius, turn_radius, velocity, timestep, y_offset)
        assert farthest <= reach
        # The first petal comes within two integrator steps (the one the
        # bound adds and the last before the turn, each at most
        # velocity / 150 s) of the bound; 0.01 deg covers the output grid
        # sampling past the farthest point.
        step = velocity * min(timestep, _DAISY_INTERNAL_TIMESTEP)
        assert farthest >= reach - 2.0 * step - 0.01
        assert reach >= radius + turn_radius

    @pytest.mark.parametrize(
        "radius, turn_radius, avoidance_radius, velocity, duration",
        [
            (0.05, 0.005, 0.0, 3.0, 120.0),  # one step is four turn radii: the turn overshoots
            (0.05, 0.2, 1.0, 0.05, 600.0),  # avoidance radius wider than the whole pattern
            (0.01, 0.03, 0.1, 0.05, 600.0),
        ],
    )
    def test_bounds_the_degenerate_regimes(
        self, radius, turn_radius, avoidance_radius, velocity, duration
    ):
        """The integrator's own discretisation, every step sampled, stays inside.

        Without the step the bound adds for the integrator, these exceed
        ``hypot(radius + step, turn_radius) + turn_radius`` by 0.0065, 2e-6
        and 4.9e-6 deg.
        """
        config = DaisyScanConfig(
            timestep=_DAISY_INTERNAL_TIMESTEP,
            radius=radius,
            velocity=velocity,
            turn_radius=turn_radius,
            avoidance_radius=avoidance_radius,
            start_acceleration=0.5,
            y_offset=0.0,
        )
        pattern = DaisyScanPattern(ra=0.0, dec=0.0, config=config)
        _, x_off, y_off = pattern.generate_offsets(duration)

        reach = _daisy_reach(radius, turn_radius, velocity, _DAISY_INTERNAL_TIMESTEP)
        assert float(np.hypot(x_off, y_off).max()) <= reach
