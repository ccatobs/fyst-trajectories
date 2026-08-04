"""Base classes for scan patterns.

This module defines:

- ``ScanPattern``: the interface that all patterns implement
- ``CelestialPattern``: base for RA/Dec centered patterns
- ``AltAzPattern``: base for native AltAz patterns
- ``TrajectoryMetadata``: optional trajectory metadata container
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from astropy.time import Time

from ..site import AtmosphericConditions, Site
from ..trajectory import Trajectory


@dataclass(frozen=True)
class TrajectoryMetadata:
    """Metadata about how a trajectory was generated.

    Attached to a :class:`~fyst_trajectories.trajectory.Trajectory` when you need to preserve the
    pattern type and the parameters it was built from.

    Parameters
    ----------
    pattern_type : str
        Name of the pattern that generated this trajectory.
    pattern_params : dict
        Parameters used to generate the pattern.
    center_ra : float, optional
        Right Ascension of pattern center in degrees.
    center_dec : float, optional
        Declination of pattern center in degrees.
    target_name : str, optional
        Name of the target (e.g., "M42", "mars").
    input_frame : str, optional
        The input coordinate frame used for the pattern center
        (e.g., "icrs", "galactic"). Default is None.
    epoch : str, optional
        The epoch/equinox if relevant (e.g., "J2000"). Primarily
        used when the input coordinates have an associated epoch.
        Default is None.
    """

    pattern_type: str
    pattern_params: dict[str, Any] = field(default_factory=dict)
    center_ra: float | None = None
    center_dec: float | None = None
    target_name: str | None = None
    input_frame: str | None = None
    epoch: str | None = None


@runtime_checkable
class ScanPattern(Protocol):
    """Interface that all scan patterns implement.

    Any class that implements these methods and properties can be used
    as a scan pattern:

    - ``name`` property: unique identifier for the pattern type
    - ``generate`` method: creates a Trajectory for the pattern
    - ``get_metadata`` method: returns TrajectoryMetadata for the pattern
    """

    @property
    def name(self) -> str:
        """Unique identifier for this pattern type.

        Returns
        -------
        str
            Pattern name (e.g., "pong", "daisy", "constant_el").
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


class CelestialPattern(ABC):
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

    @property
    @abstractmethod
    def name(self) -> str:
        """Pattern identifier.

        Returns
        -------
        str
            Pattern name.
        """

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


class AltAzPattern(ABC):
    """Base class for patterns defined in AltAz coordinates.

    These patterns work directly in the telescope's native coordinate
    system without requiring coordinate transformations.

    Examples: ConstantElScan, LinearMotion

    Attributes
    ----------
    requires_start_time : bool
        False by default for AltAz patterns. Override to True for
        patterns that need start_time (e.g., PlanetTrackPattern).
    """

    requires_start_time: ClassVar[bool] = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Pattern identifier.

        Returns
        -------
        str
            Pattern name.
        """

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
