"""Sidereal tracking pattern."""

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

from ..coordinates import Coordinates
from ..site import AtmosphericConditions, Site
from ..trajectory import Trajectory
from ..trajectory_utils import validate_trajectory_bounds
from .base import CelestialPattern, TrajectoryMetadata
from .configs import SiderealTrackConfig
from .registry import register_pattern
from .utils import compute_velocities, normalize_azimuth, wrap_bounds_error


@register_pattern("sidereal", config=SiderealTrackConfig)
class SiderealTrackPattern(CelestialPattern):
    """Sidereal tracking pattern.

    Parameters
    ----------
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    config : SiderealTrackConfig
        Tracking configuration parameters.

    Attributes
    ----------
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    config : SiderealTrackConfig
        The configuration for this pattern.

    Examples
    --------
    >>> from astropy.time import Time
    >>> from fyst_trajectories.patterns import SiderealTrackPattern, SiderealTrackConfig
    >>> start_time = Time("2026-01-15T02:00:00", scale="utc")
    >>> config = SiderealTrackConfig(timestep=0.1)
    >>> pattern = SiderealTrackPattern(ra=83.633, dec=22.014, config=config)
    >>> trajectory = pattern.generate(site, duration=300.0, start_time=start_time)
    """

    def __init__(
        self,
        ra: float,
        dec: float,
        config: SiderealTrackConfig,
    ):
        super().__init__(ra, dec)
        self.config = config

    @property
    def name(self) -> str:
        return "sidereal"

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
            Start time for tracking. Required for coordinate
            transforms (RA/Dec to AltAz).
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
            If the target is below the horizon or outside telescope
            limits at the requested time.
        """
        if start_time is None:
            raise ValueError(
                "start_time is required for SiderealTrackPattern (celestial pattern). "
                "Provide an astropy Time object."
            )

        timestep = self.config.timestep
        coords = Coordinates(site, atmosphere=atmosphere)

        n_points = int(round(duration / timestep)) + 1
        times = np.linspace(0, duration, n_points)

        obstimes = start_time + TimeDelta(times * u.s)

        az, el = coords.radec_to_altaz(self.ra, self.dec, obstimes)
        az = normalize_azimuth(az, site)

        az_vel = compute_velocities(az, times, is_angular=True)
        el_vel = compute_velocities(el, times, is_angular=False)

        with wrap_bounds_error(f"RA={self.ra:.3f} Dec={self.dec:.3f}", start_time.iso):
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
            pattern_params={},
            center_ra=self.ra,
            center_dec=self.dec,
            input_frame="icrs",
        )
