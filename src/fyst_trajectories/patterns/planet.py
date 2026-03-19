"""Planet tracking pattern.

Tracks a solar system body as it moves across the sky.
"""

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

from ..coordinates import Coordinates
from ..exceptions import TargetNotObservableError, TrajectoryBoundsError
from ..site import AtmosphericConditions, Site
from ..trajectory import Trajectory
from ..trajectory_utils import validate_trajectory_bounds
from .base import AltAzPattern, TrajectoryMetadata
from .configs import PlanetTrackConfig
from .registry import register_pattern
from .utils import compute_velocities, normalize_azimuth


@register_pattern("planet")
class PlanetTrackPattern(AltAzPattern):
    """Planet tracking pattern.

    Generates a trajectory that tracks a solar system body as it
    moves across the sky. The body position is computed from ephemeris
    at each time step, so no RA/Dec center is needed.

    This pattern extends AltAzPattern because it works directly with
    body ephemeris rather than fixed RA/Dec coordinates. Unlike most
    AltAzPattern subclasses, ``start_time`` is *required* here (for
    ephemeris lookup) and ``generate()`` raises ``ValueError`` if it
    is None. This is an intentional deviation from the AltAzPattern
    contract where ``start_time`` is typically optional.

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

    def __init__(
        self,
        config: PlanetTrackConfig,
    ):
        self.config = config

    @property
    def name(self) -> str:
        """Return pattern identifier."""
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

        try:
            validate_trajectory_bounds(site, az, el)
        except TrajectoryBoundsError as exc:
            raise TargetNotObservableError(
                target=self.config.body.capitalize(),
                time_info=start_time.iso,
                bounds_error=exc,
            ) from None

        # Compute planet RA/Dec at the trajectory midpoint so that
        # apply_detector_offset() can compute the parallactic angle.
        # This is the same single-center approximation used by celestial
        # patterns (Pong, Daisy, Sidereal).
        midpoint_time = start_time + TimeDelta(duration / 2.0 * u.s)
        mid_ra, mid_dec = coords.get_body_radec(self.config.body, midpoint_time)

        return Trajectory(
            times=times,
            az=az,
            el=el,
            az_vel=az_vel,
            el_vel=el_vel,
            start_time=start_time,
            metadata=self.get_metadata(center_ra=mid_ra, center_dec=mid_dec),
            coordsys="altaz",
        )

    def get_metadata(
        self,
        center_ra: float | None = None,
        center_dec: float | None = None,
    ) -> TrajectoryMetadata:
        """Get pattern metadata.

        Parameters
        ----------
        center_ra : float or None, optional
            Right Ascension of the body at trajectory midpoint, in degrees.
            Populated by ``generate()`` for parallactic angle computation.
        center_dec : float or None, optional
            Declination of the body at trajectory midpoint, in degrees.

        Returns
        -------
        TrajectoryMetadata
            Metadata including pattern type and parameters.
        """
        return TrajectoryMetadata(
            pattern_type=self.name,
            pattern_params={"body": self.config.body},
            target_name=self.config.body,
            center_ra=center_ra,
            center_dec=center_dec,
        )
