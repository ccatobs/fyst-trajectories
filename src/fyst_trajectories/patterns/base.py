"""Base classes and protocols for scan patterns.

This module defines:
- ScanPattern: Protocol that all patterns implement
- CelestialPattern: Base for RA/Dec centered patterns
- AltAzPattern: Base for native AltAz patterns
- TrajectoryMetadata: Optional trajectory metadata container
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from astropy.time import Time

from ..site import AtmosphericConditions, Site
from ..trajectory import Trajectory


@dataclass(frozen=True)
class TrajectoryMetadata:
    """Metadata about how a trajectory was generated.

    This is stored separately from the core Trajectory data to keep
    the Trajectory class lean. Attach this when you need to preserve
    information about the pattern configuration.

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
    """Protocol defining the interface for all scan patterns.

    Any class that implements these methods/properties can be used
    as a scan pattern, enabling duck typing while still providing
    type checking support.

    All scan patterns must implement:
    - name property: unique identifier for the pattern type
    - generate method: creates a Trajectory for the pattern
    - get_metadata method: returns TrajectoryMetadata for the pattern
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
    """

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
    """

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
