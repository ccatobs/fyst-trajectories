"""Linear motion pattern.

Simple constant velocity motion in Az/El space.
"""

import numpy as np
from astropy.time import Time

from ..site import AtmosphericConditions, Site
from ..trajectory import Trajectory
from ..trajectory_utils import validate_trajectory_bounds
from .base import AltAzPattern, TrajectoryMetadata
from .configs import LinearMotionConfig
from .registry import register_pattern


@register_pattern("linear")
class LinearMotionPattern(AltAzPattern):
    """Linear motion pattern with constant velocity.

    Generates a trajectory that moves in a straight line in Az/El space
    at constant velocity. Useful for simple scans or testing.

    Parameters
    ----------
    config : LinearMotionConfig
        Motion configuration parameters.

    Attributes
    ----------
    config : LinearMotionConfig
        The configuration for this pattern.

    Examples
    --------
    >>> from fyst_trajectories.patterns import LinearMotionPattern, LinearMotionConfig
    >>> config = LinearMotionConfig(az_start=100.0, el_start=45.0, az_velocity=0.5, el_velocity=0.1)
    >>> pattern = LinearMotionPattern(config)
    >>> trajectory = pattern.generate(site, duration=60.0)
    """

    def __init__(self, config: LinearMotionConfig):
        self.config = config

    @property
    def name(self) -> str:
        """Return pattern identifier."""
        return "linear"

    def generate(
        self,
        site: Site,
        duration: float,
        start_time: Time | None,
        atmosphere: AtmosphericConditions | None = None,
    ) -> Trajectory:
        """Generate the linear motion trajectory.

        Parameters
        ----------
        site : Site
            Telescope site configuration.
        duration : float
            Total duration in seconds.
        start_time : Time or None, optional
            Start time for the trajectory. Optional; if None, the
            trajectory will have ``start_time=None`` (acceptable for
            AltAz patterns where absolute time is not needed).
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
            If the trajectory azimuth exceeds telescope limits.
        ElevationBoundsError
            If the trajectory elevation exceeds telescope limits.
        """
        timestep = self.config.timestep

        n_points = int(round(duration / timestep)) + 1
        times = np.linspace(0, duration, n_points)

        az = self.config.az_start + self.config.az_velocity * times
        el = self.config.el_start + self.config.el_velocity * times

        az_vel = np.full(n_points, self.config.az_velocity)
        el_vel = np.full(n_points, self.config.el_velocity)

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
                "el_start": self.config.el_start,
                "az_velocity": self.config.az_velocity,
                "el_velocity": self.config.el_velocity,
            },
        )
