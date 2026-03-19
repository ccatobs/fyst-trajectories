"""Constant elevation scan pattern.

Scans back and forth in azimuth at a fixed elevation using a
trapezoidal velocity profile with smooth turnarounds.
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


@register_pattern("constant_el")
class ConstantElScanPattern(AltAzPattern):
    """Constant elevation scan pattern.

    Generates a trajectory that scans back and forth in azimuth at
    a fixed elevation, using a trapezoidal velocity profile with
    smooth acceleration-limited turnarounds.

    The velocity profile ensures physically realistic motion that
    respects the telescope's acceleration limits. For small throws
    where full cruise speed cannot be reached, a triangular velocity
    profile is used instead.

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

        Uses a trapezoidal velocity profile with smooth turnarounds.
        All points are computed simultaneously using NumPy vectorized operations.

        Parameters
        ----------
        times : np.ndarray
            Time array in seconds from start.
        az_min : float
            Minimum azimuth bound in degrees.
        az_max : float
            Maximum azimuth bound in degrees.
        az_speed : float
            Scan speed in azimuth coordinate degrees/second
            (not on-sky).
        az_accel : float
            Acceleration in azimuth coordinate degrees/second^2
            (not on-sky).
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
        t_half_turn = az_speed / az_accel
        d_half_turn = (az_speed**2) / (2 * az_accel)
        d_cruise = az_throw - 2 * d_half_turn

        positions = np.empty_like(times)
        velocities = np.empty_like(times)

        dir_fwd = 1.0 if start_increasing else -1.0
        dir_rev = -dir_fwd
        pos_fwd = az_min if start_increasing else az_max
        pos_rev = az_max if start_increasing else az_min

        scan_flag = np.empty(len(times), dtype=np.int8)

        if d_cruise < 0:
            # Triangular velocity profile — no cruise phase, all turnaround
            v_peak = np.sqrt(az_accel * az_throw)
            t_half_turn_actual = v_peak / az_accel
            t_half_cycle = 2 * t_half_turn_actual
            cycle_time = 2 * t_half_cycle

            t_in_cycle = times % cycle_time

            # Determine which half of the cycle we're in
            in_first_half = t_in_cycle < t_half_cycle
            t_in_half = np.where(in_first_half, t_in_cycle, t_in_cycle - t_half_cycle)
            direction = np.where(in_first_half, dir_fwd, dir_rev)
            start_pos = np.where(in_first_half, pos_fwd, pos_rev)

            # Accelerating phase vs decelerating phase
            in_accel = t_in_half < t_half_turn_actual
            t_decel = t_in_half - t_half_turn_actual

            # Accelerating phase
            vel_accel = direction * az_accel * t_in_half
            disp_accel = 0.5 * az_accel * t_in_half**2

            # Decelerating phase
            d_accel_phase = 0.5 * az_accel * t_half_turn_actual**2
            d_decel_phase = v_peak * t_decel - 0.5 * az_accel * t_decel**2
            vel_decel = direction * (v_peak - az_accel * t_decel)
            disp_decel = d_accel_phase + d_decel_phase

            velocities = np.where(in_accel, vel_accel, vel_decel)
            displacement = np.where(in_accel, disp_accel, disp_decel)
            positions = start_pos + direction * displacement

            # Triangular profile never reaches cruise speed — all turnaround
            scan_flag[:] = SCAN_FLAG_TURNAROUND

        else:
            # Trapezoidal velocity profile
            t_turnaround = 2 * t_half_turn
            t_cruise_time = d_cruise / az_speed
            t_half_cycle = t_cruise_time + t_turnaround
            cycle_time = 2 * t_half_cycle

            t_in_cycle = times % cycle_time

            in_first_half = t_in_cycle < t_half_cycle
            t_in_half = np.where(in_first_half, t_in_cycle, t_in_cycle - t_half_cycle)
            direction = np.where(in_first_half, dir_fwd, dir_rev)
            start_pos = np.where(in_first_half, pos_fwd, pos_rev)

            # Three phases: cruise, decelerate, accelerate (reverse)
            in_cruise = t_in_half < t_cruise_time
            in_decel = (~in_cruise) & (t_in_half < t_cruise_time + t_half_turn)
            # Remaining points are in accelerate-reverse phase (handled by else in np.where)

            # Cruise phase
            vel_cruise = direction * az_speed
            pos_cruise = start_pos + direction * az_speed * t_in_half

            # Decelerate phase: position = start_pos + direction * (d_cruise + d_d)
            # d_d is the distance traveled since deceleration started
            t_d = t_in_half - t_cruise_time
            vel_decel = direction * (az_speed - az_accel * t_d)
            d_d = az_speed * t_d - 0.5 * az_accel * t_d**2
            pos_decel = start_pos + direction * (d_cruise + d_d)

            # Accelerate reverse phase: anchored at the turning point
            # turning_point = start_pos + direction * (d_cruise + d_half_turn)
            t_a = t_in_half - t_cruise_time - t_half_turn
            vel_accel_rev = -direction * az_accel * t_a
            d_a = 0.5 * az_accel * t_a**2
            turning_point = start_pos + direction * (d_cruise + d_half_turn)
            pos_accel_rev = turning_point - direction * d_a

            velocities = np.where(
                in_cruise, vel_cruise, np.where(in_decel, vel_decel, vel_accel_rev)
            )
            positions = np.where(
                in_cruise, pos_cruise, np.where(in_decel, pos_decel, pos_accel_rev)
            )

            # Cruise = science, decel + accel-reverse = turnaround
            scan_flag[:] = SCAN_FLAG_TURNAROUND
            scan_flag[in_cruise] = SCAN_FLAG_SCIENCE

        # Diagnostic: warn if positions exceed bounds by more than floating-point noise
        pos_min = positions.min()
        pos_max = positions.max()
        overshoot = max(az_min - pos_min, pos_max - az_max)
        if overshoot > 0.01:
            warnings.warn(
                f"Scan positions exceed bounds by {overshoot:.4f} deg "
                f"(positions [{pos_min:.4f}, {pos_max:.4f}], "
                f"bounds [{az_min:.4f}, {az_max:.4f}]). "
                "Check scan parameters for consistency.",
                category=PointingWarning,
                stacklevel=2,
            )
        # Clip only floating-point noise (< 0.01 deg)
        positions = np.clip(positions, az_min, az_max)

        return positions, velocities, scan_flag
