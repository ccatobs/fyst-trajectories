"""Planet tracking pattern."""

import dataclasses
from typing import ClassVar

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

from ..coordinates import Coordinates
from ..site import AtmosphericConditions, Site
from ..trajectory import Trajectory
from ..trajectory_utils import validate_trajectory_bounds
from .base import AltAzPattern, TrajectoryMetadata
from .configs import PlanetTrackConfig
from .registry import register_pattern
from .utils import compute_velocities, normalize_azimuth, wrap_bounds_error


@register_pattern("planet", config=PlanetTrackConfig)
class PlanetTrackPattern(AltAzPattern):
    """Planet tracking pattern.

    Extends :class:`AltAzPattern` because it works directly with body
    ephemeris rather than fixed RA/Dec coordinates. Unlike most
    AltAzPattern subclasses, ``start_time`` is *required* here (for
    ephemeris lookup) and ``generate()`` raises ``ValueError`` if it is
    None.

    Parameters
    ----------
    config : PlanetTrackConfig
        Tracking configuration with body name.

    Attributes
    ----------
    config : PlanetTrackConfig
        The configuration for this pattern.

    Examples
    --------
    >>> from astropy.time import Time
    >>> from fyst_trajectories.patterns import PlanetTrackPattern, PlanetTrackConfig
    >>> config = PlanetTrackConfig(timestep=0.1, body="mars")
    >>> pattern = PlanetTrackPattern(config=config)
    >>> start_time = Time("2026-03-15T12:00:00", scale="utc")
    >>> trajectory = pattern.generate(site, duration=300.0, start_time=start_time)
    """

    requires_start_time: ClassVar[bool] = True

    def __init__(
        self,
        config: PlanetTrackConfig,
    ):
        self.config = config

    @property
    def name(self) -> str:
        return "planet"

    def generate(
        self,
        site: Site,
        duration: float,
        start_time: Time | None,
        atmosphere: AtmosphericConditions | None = None,
    ) -> Trajectory:
        """Generate the tracking trajectory.

        Parameters
        ----------
        site : Site
            Telescope site configuration.
        duration : float
            Duration to track in seconds.
        start_time : Time
            Start time for tracking. Required for ephemeris lookup.
        atmosphere : AtmosphericConditions or None, optional
            Atmospheric conditions for refraction correction.
            If None, no refraction is applied.

        Returns
        -------
        Trajectory
            The generated tracking trajectory.

        Raises
        ------
        ValueError
            If ``start_time`` is None.
        TargetNotObservableError
            If the body is below the horizon or outside telescope limits
            at the requested time. The exception has ``target``,
            ``time_info``, and ``bounds_error`` attributes.
        """
        if start_time is None:
            raise ValueError(
                "start_time is required for PlanetTrackPattern (ephemeris lookup). "
                "Provide an astropy Time object."
            )

        timestep = self.config.timestep
        coords = Coordinates(site, atmosphere=atmosphere)

        n_points = int(round(duration / timestep)) + 1
        times = np.linspace(0, duration, n_points)
        obstimes = start_time + TimeDelta(times * u.s)

        az, el = coords.get_body_altaz(self.config.body, obstimes)
        az = normalize_azimuth(az, site)

        az_vel = compute_velocities(az, times, is_angular=True)
        el_vel = compute_velocities(el, times, is_angular=False)

        with wrap_bounds_error(self.config.body.capitalize(), start_time.iso):
            validate_trajectory_bounds(site, az, el)

        # Compute planet RA/Dec at the trajectory midpoint so that
        # apply_detector_offset() can compute the parallactic angle.
        # This is the same single-center approximation used by celestial
        # patterns (Pong, Daisy, Sidereal).
        midpoint_time = start_time + TimeDelta(duration / 2.0 * u.s)
        mid_ra, mid_dec = coords.get_body_radec(self.config.body, midpoint_time)

        # Attach midpoint RA/Dec directly to the (frozen) metadata via replace.
        metadata = dataclasses.replace(self.get_metadata(), center_ra=mid_ra, center_dec=mid_dec)

        return Trajectory(
            times=times,
            az=az,
            el=el,
            az_vel=az_vel,
            el_vel=el_vel,
            start_time=start_time,
            metadata=metadata,
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
            pattern_params={"body": self.config.body},
            target_name=self.config.body,
        )
