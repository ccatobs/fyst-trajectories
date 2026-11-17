"""Constant elevation scan pattern.

Scans back and forth in azimuth at a fixed elevation using a
smooth polynomial turnaround profile.
"""

import numpy as np
from astropy.time import Time

from ..site import AtmosphericConditions, Site
from ..trajectory import SCAN_FLAG_SCIENCE, SCAN_FLAG_TURNAROUND, Trajectory, TrajectoryMetadata
from ..trajectory_utils import validate_trajectory_bounds
from .base import AltAzPattern
from .configs import ConstantElScanConfig
from .registry import register_pattern
from .turnarounds import quintic_turnaround, swept_az_envelope, turnaround_duration_sec
from .utils import validate_sample_count

#: How far, in degrees, a sample may lie outside the science range and still be
#: science: far below any pointing scale, far above the rounding that leaves a
#: sample on a science edge to either side of it.
_SCIENCE_EDGE_TOLERANCE_DEG: float = 1e-9


@register_pattern("constant_el", config=ConstantElScanConfig)
class ConstantElScanPattern(AltAzPattern):
    """Constant elevation scan pattern.

    Generates a trajectory that scans back and forth in azimuth at
    a fixed elevation, using the smooth polynomial turnaround of
    :func:`~fyst_trajectories.patterns.turnarounds.quintic_turnaround`.

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
    >>> config = ConstantElScanConfig(
    ...     timestep=0.1,
    ...     az_start=120.0,
    ...     az_stop=180.0,
    ...     elevation=45.0,
    ...     az_speed=0.5,
    ...     az_accel=1.0,
    ... )
    >>> pattern = ConstantElScanPattern(config)
    >>> trajectory = pattern.generate(site, duration=60.0)
    """

    def __init__(self, config: ConstantElScanConfig):
        self.config = config

    def generate(
        self,
        site: Site,
        duration: float,
        start_time: Time | None = None,
        atmosphere: AtmosphericConditions | None = None,
    ) -> Trajectory:
        """Generate the scan trajectory.

        Parameters
        ----------
        site : Site
            Telescope site configuration.
        duration : float
            Total duration in seconds.
        start_time : Time or None, optional
            Start time for the trajectory. Optional; defaults to ``None``
            (acceptable for AltAz patterns where absolute time is not needed).
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
        ValueError
            If ``duration`` yields fewer than two samples at the config
            timestep.
        ElevationBoundsError
            If the scan elevation exceeds telescope limits.
        """
        az_speed = self.config.az_speed
        az_accel = self.config.az_accel
        timestep = self.config.timestep

        az_min = min(self.config.az_start, self.config.az_stop)
        az_max = max(self.config.az_start, self.config.az_stop)

        start_increasing = self.config.az_start < self.config.az_stop

        n_points = validate_sample_count(duration, timestep)
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

        validate_trajectory_bounds(az, el, site=site)

        return Trajectory(
            times=times,
            az=az,
            el=el,
            az_vel=az_vel,
            el_vel=el_vel,
            start_time=start_time,
            metadata=self.get_metadata(),
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
            },
        )

    @staticmethod
    def _compute_scan_positions(
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
        ``[az_min, az_max]`` is at constant cruise velocity, and a sample
        is science when it lies in that range, to within
        ``_SCIENCE_EDGE_TOLERANCE_DEG``, as the sample where a cruise meets
        a turnaround does. The one exception is a science sample whose
        velocity is opposite to the science sample before it: the reversal
        fell between the two (a turnaround no longer than a timestep), and
        that sample is flagged as turnaround so that each leg is its own run
        of science samples.

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
        t_turnaround = turnaround_duration_sec(az_speed, az_accel)

        # The cruise covers exactly the science region [az_min, az_max] at constant
        # speed; the quintic turnaround lives in the overscan zone, overshooting the
        # science edge by the quintic peak displacement before returning to it.
        # ``swept_az_envelope`` is the shared definition of that wider range, so
        # every envelope consumer (axis limits, Sun screens) sees the same numbers
        # this generator produces; every science sample is at cruise velocity.
        d_cruise = az_throw
        motion_min, motion_max = swept_az_envelope(az_min, az_max, az_speed, az_accel)

        dir_fwd = 1.0 if start_increasing else -1.0
        dir_rev = -dir_fwd
        # Forward cruise starts at the science edge it is moving away from. The
        # reverse cruise then starts where the forward turnaround returns to (the
        # opposite science edge), so the two half-cycles form a single continuous
        # closed loop: forward cruise az_min->az_max (increasing), top turnaround
        # returning to az_max, reverse cruise az_max->az_min, bottom turnaround
        # returning to az_min. Position is C0/C1 continuous across every seam and
        # consistent with ``velocities`` everywhere. Starting the reverse cruise at
        # the far overscan edge instead leaves a ~2*d_half_turn position jump at each
        # half-cycle boundary while velocity stays smooth, so the seam must be the
        # science edge the turnaround returns to.
        pos_fwd = az_min if start_increasing else az_max
        pos_rev = az_max if start_increasing else az_min

        scan_flag = np.empty(len(times), dtype=np.int8)

        t_cruise_time = d_cruise / az_speed
        t_half_cycle = t_cruise_time + t_turnaround
        cycle_time = 2.0 * t_half_cycle

        t_in_cycle = times % cycle_time

        in_first_half = t_in_cycle < t_half_cycle
        t_in_half = np.where(in_first_half, t_in_cycle, t_in_cycle - t_half_cycle)
        direction = np.where(in_first_half, dir_fwd, dir_rev)
        start_pos = np.where(in_first_half, pos_fwd, pos_rev)

        in_cruise = t_in_half < t_cruise_time

        vel_cruise = direction * az_speed
        pos_cruise = start_pos + direction * az_speed * t_in_half

        t_turn = t_in_half - t_cruise_time
        turn_offset, turn_vel = quintic_turnaround(t_turn, az_speed, t_turnaround)
        pos_turn = start_pos + direction * (d_cruise + turn_offset)
        vel_turn = direction * turn_vel

        velocities = np.where(in_cruise, vel_cruise, vel_turn)
        positions = np.where(in_cruise, pos_cruise, pos_turn)

        # Samples fall where a cruise meets a turnaround by construction: a block
        # quantised to whole legs ends on its last leg's edge, and every seam falls
        # on a sample when the legs and turnarounds last whole timesteps. Rounding
        # leaves such a sample a few ulps to either side of the edge, so the range is
        # widened by _SCIENCE_EDGE_TOLERANCE_DEG and science samples are clipped onto
        # it, which keeps the last bits of the inputs from picking the flag. A zero
        # throw has no cruise, so none of its samples is science.
        in_science = (
            (az_throw > 0.0)
            & (positions >= az_min - _SCIENCE_EDGE_TOLERANCE_DEG)
            & (positions <= az_max + _SCIENCE_EDGE_TOLERANCE_DEG)
        )
        scan_flag[:] = SCAN_FLAG_TURNAROUND
        scan_flag[in_science] = SCAN_FLAG_SCIENCE
        # A turnaround no longer than a timestep can fall between two samples, so
        # both lie in the range and the two legs would read as one run. Where two
        # consecutive science samples move in opposite directions, the later one is
        # flagged as turnaround, which ends the leg there.
        reversal = in_science[:-1] & in_science[1:] & (velocities[:-1] * velocities[1:] < 0.0)
        scan_flag[1:][reversal] = SCAN_FLAG_TURNAROUND

        # The turnaround's peak displacement equals turnaround_overshoot_deg,
        # so every sample already lies inside the envelope; the clip only
        # absorbs floating-point rounding at the turnaround apex.
        positions = np.where(
            in_science,
            np.clip(positions, az_min, az_max),
            np.clip(positions, motion_min, motion_max),
        )

        return positions, velocities, scan_flag


def _ce_az_reach(
    az_min: float,
    az_max: float,
    az_speed: float,
    az_accel: float,
    duration: float,
    timestep: float,
) -> tuple[float, float]:
    """Return the lowest and highest azimuth a constant-elevation sweep commands.

    The sweep is the one :meth:`ConstantElScanPattern.generate` makes with
    ``az_start=az_min`` and ``az_stop=az_max``, the first leg running up,
    for ``duration`` seconds at ``timestep``, and the extremes are taken over
    the same samples. So they hold only the turnarounds the sweep makes (one
    leg makes none, two only the one past ``az_max``), and each turnaround
    only as far as its samples reach it, which can stop short of the apex
    :func:`~fyst_trajectories.patterns.turnarounds.swept_az_envelope` gives.

    Parameters
    ----------
    az_min, az_max : float
        The swept range in degrees, ``az_min < az_max``.
    az_speed : float
        Azimuth scan speed in deg/s (mount frame).
    az_accel : float
        Azimuth acceleration in deg/s^2 (mount frame).
    duration : float
        Sweep length in seconds.
    timestep : float
        Time between samples in seconds.

    Returns
    -------
    lo, hi : float
        The lowest and highest commanded azimuth in degrees.

    Raises
    ------
    ValueError
        If ``duration`` yields fewer than two samples at ``timestep``
        (:func:`~fyst_trajectories.patterns.utils.validate_sample_count`).
    """
    times = np.linspace(0, duration, validate_sample_count(duration, timestep))
    # pylint: disable-next=protected-access
    positions, _, _ = ConstantElScanPattern._compute_scan_positions(
        times=times,
        az_min=az_min,
        az_max=az_max,
        az_speed=az_speed,
        az_accel=az_accel,
        start_increasing=True,
    )
    return float(positions.min()), float(positions.max())
