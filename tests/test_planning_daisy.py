"""Tests for plan_daisy_scan."""

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories import Coordinates
from fyst_trajectories.exceptions import TargetNotObservableError
from fyst_trajectories.patterns.configs import DaisyScanConfig
from fyst_trajectories.planning import ScanBlock, plan_daisy_scan


@pytest.fixture
def start_time():
    """Provide a standard start time when the target is observable."""
    return Time("2026-03-15T04:00:00", scale="utc")


class TestPlanDaisyScan:
    """Tests for plan_daisy_scan."""

    def test_basic_plan(self, site, start_time):
        """plan_daisy_scan returns a ScanBlock with daisy config."""
        block = plan_daisy_scan(
            ra=180.0,
            dec=-30.0,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            site=site,
            start_time=start_time,
            timestep=0.1,
            duration=60.0,
        )

        assert isinstance(block, ScanBlock)
        assert isinstance(block.config, DaisyScanConfig)
        assert block.duration == pytest.approx(60.0)
        assert block.trajectory.n_points > 0
        assert "Daisy scan" in block.summary

        # It's a daisy rosette, not a degenerate straight line. In the on-sky
        # offset frame about the tracked centre the path crosses near the centre
        # and spans out toward the radius in BOTH axes.
        coords = Coordinates(site)
        c_az, c_el = coords.radec_to_altaz(180.0, -30.0, obstime=start_time)
        traj = block.trajectory
        dx = (traj.az - c_az) * np.cos(np.radians(traj.el))
        dy = traj.el - c_el
        assert np.hypot(dx, dy).min() < 0.15  # crosses near centre
        assert np.ptp(dx) > 0.5  # spans 2-D, not collinear
        assert np.ptp(dy) > 0.5

    def test_trajectory_has_valid_bounds(self, site, start_time):
        """Generated trajectory must stay within telescope elevation limits."""
        block = plan_daisy_scan(
            ra=180.0,
            dec=-30.0,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            site=site,
            start_time=start_time,
            timestep=0.1,
            duration=60.0,
        )

        traj = block.trajectory
        limits = site.telescope_limits
        assert traj.el.min() >= limits.elevation.min
        assert traj.el.max() <= limits.elevation.max
        assert traj.az.min() >= limits.azimuth.min
        assert traj.az.max() <= limits.azimuth.max

    def test_unobservable_target_raises(self, site, start_time):
        """Test that an unobservable target raises TargetNotObservableError."""
        with pytest.raises(TargetNotObservableError):
            plan_daisy_scan(
                ra=180.0,
                dec=80.0,  # Not visible from FYST
                radius=0.5,
                velocity=0.3,
                turn_radius=0.2,
                avoidance_radius=0.0,
                start_acceleration=0.5,
                site=site,
                start_time=start_time,
                timestep=0.1,
                duration=60.0,
            )
