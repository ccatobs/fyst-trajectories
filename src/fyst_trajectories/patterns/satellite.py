"""Planetary-satellite tracking pattern."""

from .configs import SatelliteTrackConfig
from .planet import PlanetTrackPattern
from .registry import register_pattern


@register_pattern("satellite", config=SatelliteTrackConfig)
class SatelliteTrackPattern(PlanetTrackPattern):
    """Planetary-satellite tracking pattern.

    A thin subclass of :class:`PlanetTrackPattern` that follows a
    planetary satellite (e.g. Titan) as it moves across the sky, used as
    an unresolved submillimetre flux calibrator. All trajectory
    generation is inherited from :class:`PlanetTrackPattern`: the only
    difference is that the body is resolved from a JPL satellite SPK
    kernel (configured via ``SatelliteTrackConfig.satellite_kernel``
    or the ``FYST_SATELLITE_KERNEL`` environment variable) rather than
    astropy's builtin ephemeris. As with planet tracking, ``start_time``
    is required and az normalisation, velocities, and bounds checking
    behave identically.

    Parameters
    ----------
    config : SatelliteTrackConfig
        Tracking configuration with the satellite body name and an
        optional kernel path.

    Attributes
    ----------
    config : SatelliteTrackConfig
        The configuration for this pattern.

    Examples
    --------
    >>> from astropy.time import Time
    >>> from fyst_trajectories.patterns import SatelliteTrackPattern, SatelliteTrackConfig
    >>> config = SatelliteTrackConfig(timestep=0.1, body="titan", satellite_kernel="titan.bsp")
    >>> pattern = SatelliteTrackPattern(config=config)
    >>> start_time = Time("2026-06-15T04:00:00", scale="utc")
    >>> trajectory = pattern.generate(site, duration=300.0, start_time=start_time)
    """

    @property
    def name(self) -> str:
        return "satellite"
