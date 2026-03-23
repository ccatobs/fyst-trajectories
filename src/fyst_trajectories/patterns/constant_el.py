"""Constant elevation scan pattern.

Scans back and forth in azimuth at a fixed elevation using a
quintic polynomial velocity profile with smooth turnarounds.
"""

import warnings

import numpy as np
from astropy.time import Time

from ..exceptions import PointingWarning
from ..site import AtmosphericConditions, Site
from ..trajectory import SCAN_FLAG_SCIENCE, SCAN_FLAG_TURNAROUND, Trajectory
from ..trajectory_utils import validate_trajectory_bounds
from .base import AltAzPattern, TrajectoryMetadata
from .configs import ConstantElScanConfig
from .registry import register_pattern
from .turnarounds import quintic_turnaround


@register_pattern("constant_el")
class ConstantElScanPattern(AltAzPattern):
    """Constant elevation scan pattern.

    Generates a trajectory that scans back and forth in azimuth at
    a fixed elevation, using a quintic polynomial velocity profile
    for smooth turnarounds.

    The quintic turnaround has zero acceleration at the cruise/turn
    boundaries, providing C2 continuity. The peak acceleration is
    1.5x the average acceleration (``az_accel``), and turnaround
    overshoot is 25% larger than the linear (trapezoidal) profile.

    Note: The first scan leg begins at full cruise speed. Initial
    ramp-up from rest is not modeled.

    Parameters
    ----------
    config : ConstantElScanConfig
        Scan configuration parameters.

    Attributes
    ----------
    config : ConstantElScanConfig
        The configuration for this pattern.

    Examples
    --------
    >>> from fyst_trajectories.patterns import ConstantElScanPattern, ConstantElScanConfig
    >>> config = ConstantElScanConfig(az_start=120.0, az_stop=180.0, elevation=45.0)
    >>> pattern = ConstantElScanPattern(config)
    >>> trajectory = pattern.generate(site, duration=60.0)
    """

    def __init__(self, config: ConstantElScanConfig):
        self.config = config

    @property
    def name(self) -> str:
        """Return pattern identifier."""
        return "constant_el"

    def generate(
        self,
        site: Site,
        duration: float,
        start_time: Time | None,
        atmosphere: AtmosphericConditions | None = None,
    ) -> Trajectory:
        """Generate the scan trajectory.

        Parameters
        ----------
        site : Site
            Telescope site configuration.
        duration : float
            Total duration in seconds.
        start_time : Time or None
            Start time for the trajectory.
        atmosphere : AtmosphericConditions or None, optional
            Not used by this AltAz pattern (no coordinate transforms).
            Accepted for interface compatibility.

        Returns
        -------
        Trajectory
            The generated trajectory.

        Raises
        ------
        AzimuthBoundsError
            If the scan azimuth range exceeds telescope limits.
        ElevationBoundsError
            If the scan elevation exceeds telescope limits.
        """
        az_speed = self.config.az_speed
        az_accel = self.config.az_accel
        timestep = self.config.timestep

        az_min = min(self.config.az_start, self.config.az_stop)
        az_max = max(self.config.az_start, self.config.az_stop)

        start_increasing = self.config.az_start < self.config.az_stop

        n_points = int(round(duration / timestep)) + 1
        times = np.linspace(0, duration, n_points)

        el = np.full(n_points, self.config.elevation)
        el_vel = np.zeros(n_points)

        az, az_vel, scan_flag = self._compute_scan_positions(
            times=times,
            az_min=az_min,
            az_max=az_max,
            az_speed=az_speed,
            az_accel=az_accel,
            start_increasing=start_increasing,
        )

        validate_trajectory_bounds(site, az, el)

        return Trajectory(
            times=times,
            az=az,
            el=el,
            az_vel=az_vel,
            el_vel=el_vel,
            start_time=start_time,
            metadata=self.get_metadata(),
            coordsys="altaz",
            scan_flag=scan_flag,
        )

    def get_metadata(self) -> TrajectoryMetadata:
        """Get pattern metadata.

        Returns
        -------
        TrajectoryMetadata
            Metadata including pattern type and parameters.
        """
        return TrajectoryMetadata(
            pattern_type=self.name,
            pattern_params={
                "az_start": self.config.az_start,
                "az_stop": self.config.az_stop,
                "elevation": self.config.elevation,
                "az_speed": self.config.az_speed,
                "az_accel": self.config.az_accel,
                "n_scans": self.config.n_scans,
            },
        )

    def _compute_scan_positions(
        self,
        times: np.ndarray,
        az_min: float,
        az_max: float,
        az_speed: float,
        az_accel: float,
        start_increasing: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute positions, velocities, and scan flags for a back-and-forth scan.

        Uses a quintic polynomial turnaround profile. Turnarounds
        happen outside the science region so that every sample within
        ``[az_min, az_max]`` is at constant cruise velocity.

        Parameters
        ----------
        times : np.ndarray
            Time array in seconds from start.
        az_min : float
            Minimum azimuth bound (science region) in degrees.
        az_max : float
            Maximum azimuth bound (science region) in degrees.
        az_speed : float
            Scan speed in azimuth coordinate degrees/second
            (not on-sky).
        az_accel : float
            Average acceleration in azimuth coordinate degrees/second^2
            (not on-sky). Peak acceleration is 1.5x this value.
        start_increasing : bool
            If True, start scanning toward increasing azimuth.

        Returns
        -------
        positions : np.ndarray
            Azimuth positions in degrees.
        velocities : np.ndarray
            Azimuth velocities in degrees/second.
        scan_flag : np.ndarray
            Per-sample scan flag (1 = science, 2 = turnaround).
        """
        az_throw = az_max - az_min
        # Quintic turnaround: T = 2*v/a_avg, d = 5*v^2/(8*a_avg)
        t_turnaround = 2.0 * az_speed / az_accel
        d_half_turn = 5.0 * az_speed**2 / (8.0 * az_accel)

        # Cruise covers the full science region, turnarounds are outside
        d_cruise = az_throw
        motion_min = az_min - d_half_turn
        motion_max = az_max + d_half_turn

        dir_fwd = 1.0 if start_increasing else -1.0
        dir_rev = -dir_fwd
        pos_fwd = motion_min if start_increasing else motion_max
        pos_rev = motion_max if start_increasing else motion_min

        scan_flag = np.empty(len(times), dtype=np.int8)

        # Quintic velocity profile: cruise + single turnaround phase
        t_cruise_time = d_cruise / az_speed
        t_half_cycle = t_cruise_time + t_turnaround
        cycle_time = 2.0 * t_half_cycle

        t_in_cycle = times % cycle_time

        in_first_half = t_in_cycle < t_half_cycle
        t_in_half = np.where(in_first_half, t_in_cycle, t_in_cycle - t_half_cycle)
        direction = np.where(in_first_half, dir_fwd, dir_rev)
        start_pos = np.where(in_first_half, pos_fwd, pos_rev)

        # Two phases per half-cycle: cruise, then quintic turnaround
        in_cruise = t_in_half < t_cruise_time

        # Cruise phase
        vel_cruise = direction * az_speed
        pos_cruise = start_pos + direction * az_speed * t_in_half

        # Quintic turnaround phase
        t_turn = t_in_half - t_cruise_time
        turn_offset, turn_vel = quintic_turnaround(t_turn, az_speed, t_turnaround)
        pos_turn = start_pos + direction * (d_cruise + turn_offset)
        vel_turn = direction * turn_vel

        velocities = np.where(in_cruise, vel_cruise, vel_turn)
        positions = np.where(in_cruise, pos_cruise, pos_turn)

        # Cruise samples within science region are science,
        # everything else (including cruise samples in overscan) is turnaround
        scan_flag[:] = SCAN_FLAG_TURNAROUND
        in_science = in_cruise & (positions >= az_min) & (positions <= az_max)
        scan_flag[in_science] = SCAN_FLAG_SCIENCE

        # Positions intentionally exceed [az_min, az_max].
        # Clip only floating-point noise beyond the expected motion range.
        pos_min = positions.min()
        pos_max = positions.max()
        overshoot = max(motion_min - pos_min, pos_max - motion_max)
        if overshoot > 0.01:
            warnings.warn(
                f"Scan positions exceed motion range by {overshoot:.4f} deg "
                f"(positions [{pos_min:.4f}, {pos_max:.4f}], "
                f"motion range [{motion_min:.4f}, {motion_max:.4f}]). "
                "Check scan parameters for consistency.",
                category=PointingWarning,
                stacklevel=2,
            )
        positions = np.clip(positions, motion_min, motion_max)

        return positions, velocities, scan_flag
