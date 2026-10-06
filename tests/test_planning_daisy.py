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
    """Block shape, the rosette's two-axis spread, and the refusal."""

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory azimuth acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
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

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory azimuth acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_default_timestep_equals_explicit(self, site, start_time):
        """Omitting ``timestep`` plans exactly what ``timestep=0.1`` plans."""
        common = dict(
            ra=180.0,
            dec=-30.0,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            site=site,
            start_time=start_time,
            duration=60.0,
        )
        defaulted = plan_daisy_scan(**common)
        explicit = plan_daisy_scan(**common, timestep=0.1)

        assert defaulted.config == explicit.config
        for name in ("times", "az", "el", "az_vel", "el_vel"):
            np.testing.assert_array_equal(
                getattr(defaulted.trajectory, name), getattr(explicit.trajectory, name)
            )

    def test_unobservable_target_raises(self, site, start_time):
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
