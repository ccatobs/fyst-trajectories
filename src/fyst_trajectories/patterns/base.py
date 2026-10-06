"""Base classes for scan patterns.

This module defines:

- ``ScanPattern``: the interface that all patterns implement
- ``CelestialPattern``: base for RA/Dec centered patterns
- ``AltAzPattern``: base for native AltAz patterns
"""

from abc import ABC, abstractmethod
from typing import ClassVar, Protocol, runtime_checkable

from astropy.time import Time

from ..site import AtmosphericConditions, Site
from ..trajectory import Trajectory, TrajectoryMetadata


@runtime_checkable
class ScanPattern(Protocol):
    """Interface that all scan patterns implement.

    Any class that provides these members can be used as a scan pattern:

    - ``name``: unique identifier for the pattern type, a class attribute
    - ``requires_start_time``: whether ``generate`` needs a start time,
      a class attribute
    - ``generate`` method: creates a Trajectory for the pattern
    - ``get_metadata`` method: returns TrajectoryMetadata for the pattern
    """

    name: ClassVar[str]
    """Unique identifier for this pattern type (e.g. "pong", "daisy").

    :func:`~fyst_trajectories.patterns.register_pattern` sets it on the
    class from the registered name.
    """

    requires_start_time: ClassVar[bool]
    """Whether ``generate`` needs a start time.

    :class:`~fyst_trajectories.patterns.TrajectoryBuilder` refuses to build
    a pattern that needs one before ``.starting_at()`` is called.
    """

    def generate(
        self,
        site: Site,
        duration: float,
        start_time: Time | None,
        atmosphere: AtmosphericConditions | None = None,
    ) -> Trajectory:
        """Generate a trajectory for this pattern.

        Parameters
        ----------
        site : Site
            Telescope site configuration.
        duration : float
            Total duration in seconds.
        start_time : Time or None
            Start time for the trajectory.
        atmosphere : AtmosphericConditions or None, optional
            Atmospheric conditions for refraction correction.
            If None, no refraction is applied.

        Returns
        -------
        Trajectory
            The generated trajectory.
        """

    def get_metadata(self) -> TrajectoryMetadata:
        """Get metadata describing this pattern configuration.

        Returns
        -------
        TrajectoryMetadata
            Metadata including pattern type and parameters.
        """


class _PatternBase(ABC):
    """Abstract base shared by :class:`CelestialPattern` and :class:`AltAzPattern`.

    ``name`` is set on each concrete class by ``register_pattern``.
    """

    name: ClassVar[str]
    requires_start_time: ClassVar[bool] = False

    @abstractmethod
    def generate(
        self,
        site: Site,
        duration: float,
        start_time: Time | None,
        atmosphere: AtmosphericConditions | None = None,
    ) -> Trajectory:
        """Generate the trajectory.

        Parameters
        ----------
        site : Site
            Telescope site configuration.
        duration : float
            Total duration in seconds.
        start_time : Time or None
            Start time for the trajectory.
        atmosphere : AtmosphericConditions or None, optional
            Atmospheric conditions for refraction correction.
            If None, no refraction is applied.

        Returns
        -------
        Trajectory
            The generated trajectory.
        """

    @abstractmethod
    def get_metadata(self) -> TrajectoryMetadata:
        """Get pattern metadata.

        Returns
        -------
        TrajectoryMetadata
            Metadata including pattern type and parameters.
        """


class CelestialPattern(_PatternBase):
    """Base class for patterns centered on celestial coordinates.

    These patterns are defined relative to an RA/Dec center point
    and converted to AltAz during generation based on the observation
    time and site location.

    Examples: Pong, Daisy, SiderealTrack

    Parameters
    ----------
    ra : float
        Right Ascension of pattern center in degrees.
    dec : float
        Declination of pattern center in degrees.

    Attributes
    ----------
    ra : float
        Right Ascension of pattern center in degrees.
    dec : float
        Declination of pattern center in degrees.
    requires_start_time : bool
        Always True for celestial patterns (coordinate transforms need time).
    """

    requires_start_time: ClassVar[bool] = True

    def __init__(self, ra: float, dec: float):
        self.ra = ra
        self.dec = dec


class AltAzPattern(_PatternBase):
    """Base class for patterns defined in AltAz coordinates.

    These patterns take no RA/Dec center. Most are defined directly in
    the telescope's native az/el frame and need no coordinate transform;
    the two trackers below resolve a body's ephemeris to az/el instead.

    Subclasses: ``ConstantElScanPattern``, ``LinearMotionPattern``,
    ``PongAltAzScanPattern``, ``DaisyAltAzScanPattern``, and the
    trackers ``PlanetTrackPattern`` / ``SatelliteTrackPattern`` (which
    override ``requires_start_time`` for their ephemerides).

    Attributes
    ----------
    requires_start_time : bool
        False by default for AltAz patterns. Override to True for
        patterns that need start_time (e.g., PlanetTrackPattern).
    """
