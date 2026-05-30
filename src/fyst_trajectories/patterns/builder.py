"""Fluent builder for trajectory construction.

The TrajectoryBuilder provides a fluent API for constructing
trajectories, allowing incremental configuration with validation.

The pattern type is automatically inferred from the config class,
so there is no need to explicitly call `.pattern()`.

Examples
--------
>>> from astropy.time import Time
>>> from fyst_trajectories import get_fyst_site
>>> from fyst_trajectories.patterns import TrajectoryBuilder, PongScanConfig
>>>
>>> site = get_fyst_site()
>>> start_time = Time("2026-03-15T04:00:00", scale="utc")
>>> trajectory = (
...     TrajectoryBuilder(site)
...     .at(ra=180.0, dec=-30.0)
...     .with_config(
...         PongScanConfig(
...             timestep=0.1,
...             width=2.0,
...             height=2.0,
...             spacing=0.1,
...             velocity=0.5,
...             num_terms=4,
...             angle=0.0,
...         )
...     )
...     .duration(300.0)
...     .starting_at(start_time)
...     .build()
... )
"""

import warnings
from typing import TYPE_CHECKING

from astropy.time import Time

from ..offsets import apply_detector_offset
from ..site import AtmosphericConditions, Site
from ..trajectory import Trajectory
from ..trajectory_utils import validate_trajectory_bounds, validate_trajectory_dynamics
from .base import AltAzPattern, CelestialPattern
from .configs import ScanConfig
from .registry import get_pattern, get_pattern_for_config

if TYPE_CHECKING:
    from ..offsets import InstrumentOffset


class TrajectoryBuilder:
    """Fluent builder for trajectory construction.

    Provides a chainable API for building trajectories step by step,
    with validation at build time. The pattern type is automatically
    inferred from the config class passed to `with_config()`.

    Celestial patterns (Pong, Daisy, Sidereal) and planet tracking
    require ``.starting_at()`` to be called before ``.build()``.
    AltAz patterns (ConstantEl, Linear) do not require a start time.

    Parameters
    ----------
    site : Site
        Telescope site configuration.

    Attributes
    ----------
    site : Site
        The telescope site configuration.

    Examples
    --------
    Build a Pong scan trajectory:

    >>> from astropy.time import Time
    >>> from fyst_trajectories import get_fyst_site
    >>> from fyst_trajectories.patterns import TrajectoryBuilder, PongScanConfig
    >>>
    >>> site = get_fyst_site()
    >>> start_time = Time("2026-03-15T04:00:00", scale="utc")
    >>> trajectory = (
    ...     TrajectoryBuilder(site)
    ...     .at(ra=180.0, dec=-30.0)
    ...     .with_config(
    ...         PongScanConfig(
    ...             timestep=0.1,
    ...             width=2.0,
    ...             height=2.0,
    ...             spacing=0.1,
    ...             velocity=0.5,
    ...             num_terms=4,
    ...             angle=0.0,
    ...         )
    ...     )
    ...     .duration(300.0)
    ...     .starting_at(start_time)
    ...     .build()
    ... )

    Build a constant elevation scan:

    >>> from fyst_trajectories.patterns import ConstantElScanConfig
    >>> trajectory = (
    ...     TrajectoryBuilder(site)
    ...     .with_config(
    ...         ConstantElScanConfig(
    ...             timestep=0.1,
    ...             az_start=120.0,
    ...             az_stop=180.0,
    ...             elevation=45.0,
    ...             az_speed=1.0,
    ...             az_accel=0.5,
    ...         )
    ...     )
    ...     .duration(60.0)
    ...     .build()
    ... )
    """

    def __init__(self, site: Site):
        self._site = site
        self._pattern_name: str | None = None
        self._ra: float | None = None
        self._dec: float | None = None
        self._config: ScanConfig | None = None
        self._duration: float | None = None
        self._start_time: Time | None = None
        self._detector_offset: InstrumentOffset | None = None
        self._atmosphere: AtmosphericConditions | None = None

    def at(self, ra: float, dec: float) -> "TrajectoryBuilder":
        """Set the celestial center coordinates.

        Used for celestial patterns like Pong, Daisy, and Sidereal.

        Parameters
        ----------
        ra : float
            Right Ascension in degrees.
        dec : float
            Declination in degrees.

        Returns
        -------
        TrajectoryBuilder
            Self for chaining.
        """
        self._ra = ra
        self._dec = dec
        return self

    def with_config(self, config: ScanConfig) -> "TrajectoryBuilder":
        """Set the pattern configuration and infer pattern type.

        The pattern type is automatically inferred from the config class.
        For example, passing a ``PongScanConfig`` will set the pattern
        to "pong".

        Parameters
        ----------
        config : ScanConfig
            Pattern-specific configuration.

        Returns
        -------
        TrajectoryBuilder
            Self for chaining.

        Raises
        ------
        ValueError
            If the config type is not recognized.
        """
        self._config = config

        try:
            self._pattern_name = get_pattern_for_config(type(config))
        except KeyError as exc:
            raise ValueError(str(exc)) from None

        return self

    def duration(self, seconds: float) -> "TrajectoryBuilder":
        """Set the trajectory duration.

        Parameters
        ----------
        seconds : float
            Duration in seconds.

        Returns
        -------
        TrajectoryBuilder
            Self for chaining.

        Raises
        ------
        ValueError
            If duration is not positive.
        """
        if seconds <= 0:
            raise ValueError(f"Duration must be positive, got {seconds}")
        self._duration = seconds
        return self

    def starting_at(self, time: Time | str) -> "TrajectoryBuilder":
        """Set the start time.

        Parameters
        ----------
        time : Time or str
            Start time as astropy Time or ISO string.

        Returns
        -------
        TrajectoryBuilder
            Self for chaining.
        """
        if isinstance(time, str):
            time = Time(time, scale="utc")
        self._start_time = time
        return self

    def with_atmosphere(self, atmosphere: AtmosphericConditions) -> "TrajectoryBuilder":
        """Set atmospheric conditions for refraction correction.

        Without this call, coordinate transforms use no refraction
        (pressure=0). Call this to enable atmospheric refraction in
        celestial-to-horizontal transforms.

        Parameters
        ----------
        atmosphere : AtmosphericConditions
            Atmospheric conditions to use for refraction correction.

        Returns
        -------
        TrajectoryBuilder
            Self for chaining.
        """
        self._atmosphere = atmosphere
        return self

    def for_detector(
        self,
        offset: "InstrumentOffset | None",
    ) -> "TrajectoryBuilder":
        """Adjust trajectory so specified detector is centered on target.

        When generating trajectories, the boresight positions will be
        computed such that the detector with the given offset observes
        the target coordinates instead of the boresight. Uses spherical
        trigonometry for accurate offset projection at any offset size.

        This is useful when you want an off-axis instrument or detector
        module to track a celestial target, rather than the telescope
        boresight. Passing ``None`` is a no-op (boresight tracking),
        which allows callers to unconditionally call this method without
        checking whether an offset was resolved.

        Parameters
        ----------
        offset : InstrumentOffset or None
            The offset of the detector from the boresight.
            If ``None``, the trajectory targets the boresight (no-op).

        Returns
        -------
        TrajectoryBuilder
            Self for chaining.

        Examples
        --------
        >>> from astropy.time import Time
        >>> from fyst_trajectories import InstrumentOffset
        >>> from fyst_trajectories.patterns import TrajectoryBuilder, PongScanConfig
        >>>
        >>> offset = InstrumentOffset(dx=5.0, dy=3.0, name="Mod2")
        >>> start_time = Time("2026-03-15T04:00:00", scale="utc")
        >>> trajectory = (
        ...     TrajectoryBuilder(site)
        ...     .at(ra=180.0, dec=-30.0)
        ...     .with_config(
        ...         PongScanConfig(
        ...             timestep=0.1,
        ...             width=1.0,
        ...             height=1.0,
        ...             spacing=0.1,
        ...             velocity=0.5,
        ...             num_terms=4,
        ...             angle=0.0,
        ...         )
        ...     )
        ...     .for_detector(offset)
        ...     .duration(60.0)
        ...     .starting_at(start_time)
        ...     .build()
        ... )
        """
        self._detector_offset = offset
        return self

    @staticmethod
    def _needs_start_time(pattern_cls: type) -> bool:
        """Check if a pattern class requires a start time.

        Uses the ``requires_start_time`` ClassVar declared on base classes.
        CelestialPattern defaults to True, AltAzPattern defaults to False,
        and PlanetTrackPattern overrides to True.

        Parameters
        ----------
        pattern_cls : type
            The pattern class to check.

        Returns
        -------
        bool
            True if the pattern requires a start time.
        """
        return getattr(pattern_cls, "requires_start_time", False)

    def build(self) -> Trajectory:
        """Build the trajectory.

        Validates all required parameters are set, instantiates the
        pattern, generates the trajectory, and attaches metadata.

        If a detector offset was specified via for_detector(), the
        trajectory positions are adjusted so the detector observes
        the target instead of the boresight.

        After generating the trajectory (and applying any detector
        offset), ``validate_trajectory_dynamics()`` is called
        automatically to warn if velocity or acceleration limits
        are exceeded, and ``validate_trajectory_bounds()`` is called
        as a defence-in-depth check that the final trajectory is
        within telescope position limits.  Pattern modules already
        validate their own bounds; this extra call catches any
        out-of-bounds positions introduced by detector offsets or
        future changes to pattern generators.

        Returns
        -------
        Trajectory
            The generated trajectory with metadata attached.

        Raises
        ------
        ValueError
            If required parameters are missing (config, duration,
            coordinates for celestial patterns, or start time for
            time-dependent patterns).
        TargetNotObservableError
            If the target is not observable at the requested time.
        TrajectoryBoundsError
            If the trajectory exceeds telescope limits.
        """
        if self._pattern_name is None:
            raise ValueError("Pattern not set. Call .with_config() first.")
        if self._duration is None:
            raise ValueError("Duration not set. Call .duration() first.")

        pattern_cls = get_pattern(self._pattern_name)

        kwargs = {}
        # Only pass ra/dec to patterns that accept them (CelestialPattern subclasses).
        # AltAzPattern subclasses (e.g., PlanetTrackPattern) don't use ra/dec.
        if issubclass(pattern_cls, CelestialPattern):
            if self._ra is None or self._dec is None:
                raise ValueError(
                    f"{pattern_cls.__name__} requires sky coordinates. "
                    "Call .at(ra, dec) before .build()."
                )
            kwargs["ra"] = self._ra
            kwargs["dec"] = self._dec
        elif issubclass(pattern_cls, AltAzPattern):
            if self._ra is not None or self._dec is not None:
                warnings.warn(
                    f"ra/dec values are ignored for {pattern_cls.__name__} (AltAz pattern)",
                    stacklevel=2,
                )

        # Validate start_time for patterns that need it
        if self._needs_start_time(pattern_cls) and self._start_time is None:
            raise ValueError(
                f"{pattern_cls.__name__} requires a start time for coordinate "
                "transforms. Call .starting_at(start_time) before .build()."
            )

        if self._config is not None:
            kwargs["config"] = self._config

        pattern = pattern_cls(**kwargs)

        trajectory = pattern.generate(
            site=self._site,
            duration=self._duration,
            start_time=self._start_time,
            atmosphere=self._atmosphere,
        )

        if self._detector_offset is not None:
            trajectory = apply_detector_offset(
                trajectory,
                self._detector_offset,
                self._site,
            )

        # Validate dynamics after detector offset so the check covers
        # the actual trajectory the telescope will execute.
        validate_trajectory_dynamics(self._site, trajectory.az, trajectory.el, trajectory.times)

        # Defence in depth: patterns already validate their own bounds,
        # but applying a detector offset can push points past telescope
        # limits.  Re-validate the final trajectory here so the builder
        # refuses to hand off an infeasible path to the caller.
        validate_trajectory_bounds(self._site, trajectory.az, trajectory.el)

        return trajectory
