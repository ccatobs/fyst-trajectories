"""Trajectory container for telescope scan patterns.

This module provides the Trajectory class which holds time-stamped
position and velocity setpoints for both azimuth and elevation axes,
suitable for upload to the telescope ACU.

The trajectory data is intentionally minimal; metadata about pattern
type, generation parameters, and input coordinates can be attached
via the optional ``metadata`` attribute.

All utility functions (validate, export, format, plot) are free
functions in :mod:`fyst_trajectories.trajectory_utils`. They are the
sole API for operating on a :class:`Trajectory` so the container itself
stays dependency-free.

Examples
--------
Create a trajectory manually:

>>> import numpy as np
>>> times = np.array([0, 1, 2, 3, 4])
>>> az = np.array([100, 101, 102, 101, 100])
>>> el = np.full(5, 45.0)
>>> az_vel = np.array([1, 1, 0, -1, -1])
>>> el_vel = np.zeros(5)
>>> traj = Trajectory(times=times, az=az, el=el, az_vel=az_vel, el_vel=el_vel)

Use with pattern generators:

>>> from astropy.time import Time
>>> from fyst_trajectories.patterns import TrajectoryBuilder, PongScanConfig
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

Print trajectory summary:

>>> from fyst_trajectories import print_trajectory
>>> print_trajectory(trajectory, head=3, tail=2)
"""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy.time import Time

if TYPE_CHECKING:
    from .patterns.base import TrajectoryMetadata

SCAN_FLAG_UNCLASSIFIED: int = 0
SCAN_FLAG_SCIENCE: int = 1
SCAN_FLAG_TURNAROUND: int = 2
SCAN_FLAG_RETUNE: int = 3


@dataclass(frozen=True)
class RetuneEvent:
    """A single retune event in trajectory-relative seconds.

    Parameters
    ----------
    t_start : float
        Seconds from the trajectory start (i.e. ``trajectory.times[0]``).
        Must be finite and non-negative. Events past ``trajectory.times[-1]``
        are skipped with a :class:`~fyst_trajectories.exceptions.PointingWarning`.
    duration : float
        Wall-clock duration of the event in seconds. Must be positive.
        Events that would extend past the trajectory end are clipped to
        the trajectory end (matching the uniform-cadence path's behaviour).

    Notes
    -----
    The minimal two-field shape is deliberate. See
    ``docs/reviews/methodology_audit.md`` and Q-9..Q-13 in
    ``docs/reviews/fyst_team_questions.md`` for the research basis.
    Per-module staggering is the caller's composition: invoke
    :func:`~fyst_trajectories.trajectory_utils.inject_retune` once per
    module with a different event list rather than embedding module
    identity in the event.
    """

    t_start: float
    duration: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.t_start):
            raise ValueError(f"t_start must be finite, got {self.t_start}")
        if self.t_start < 0:
            raise ValueError(f"t_start must be non-negative, got {self.t_start}")
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")


@dataclass
class Trajectory:
    """Container for a telescope trajectory.

    Holds time-stamped position and velocity setpoints for both
    azimuth and elevation axes, suitable for upload to the ACU.

    Parameters
    ----------
    times : np.ndarray
        Timestamps in seconds from start.
    az : np.ndarray
        Azimuth positions in degrees.
    el : np.ndarray
        Elevation positions in degrees.
    az_vel : np.ndarray
        Azimuth velocities in degrees/second.
    el_vel : np.ndarray
        Elevation velocities in degrees/second.
    start_time : Time, optional
        Absolute start time for the trajectory.
    metadata : TrajectoryMetadata, optional
        Optional metadata about pattern generation.
    coordsys : str, optional
        Coordinate system of the trajectory points. Typically "altaz" for
        generated trajectories since pattern classes output Az/El coordinates.
        Default is None, but patterns should set this to "altaz".
    epoch : str, optional
        The epoch/equinox if relevant (e.g., "J2000"). Primarily used for
        documentation when the trajectory was derived from celestial coordinates.
        Default is None.
    scan_flag : np.ndarray or None, optional
        Per-sample scan phase flag. Values follow the SO ACU convention:
        0 = unclassified, 1 = constant-velocity science sweep,
        2 = turnaround. None means no flagging info is available.
    retune_events : tuple of RetuneEvent, optional
        Event-level provenance for the :data:`SCAN_FLAG_RETUNE` entries in
        ``scan_flag``. Populated by
        :func:`~fyst_trajectories.trajectory_utils.inject_retune` for both
        the uniform-cadence and explicit event-list code paths. Empty tuple
        (the default) means no retune events have been injected. See
        :class:`RetuneEvent`.

    Attributes
    ----------
    duration : float
        Total duration of trajectory in seconds.
    n_points : int
        Number of trajectory points.
    pattern_type : str or None
        Pattern type from metadata, if available.
    pattern_params : dict or None
        Pattern parameters from metadata, if available.
    center_ra : float or None
        Center RA from metadata, if available.
    center_dec : float or None
        Center Dec from metadata, if available.
    """

    times: np.ndarray
    az: np.ndarray
    el: np.ndarray
    az_vel: np.ndarray
    el_vel: np.ndarray
    start_time: Time | None = None
    metadata: "TrajectoryMetadata | None" = field(default=None, repr=False)
    coordsys: str | None = None
    epoch: str | None = None
    scan_flag: np.ndarray | None = None
    retune_events: tuple[RetuneEvent, ...] = ()

    def __post_init__(self) -> None:
        n = len(self.times)
        if n < 1:
            raise ValueError("Trajectory requires at least 1 time point")
        for name, arr in [
            ("az", self.az),
            ("el", self.el),
            ("az_vel", self.az_vel),
            ("el_vel", self.el_vel),
        ]:
            if len(arr) != n:
                raise ValueError(
                    f"Array length mismatch: times has {n} elements but {name} has {len(arr)}"
                )
        if self.scan_flag is not None:
            if len(self.scan_flag) != n:
                raise ValueError(
                    f"Array length mismatch: times has {n} elements "
                    f"but scan_flag has {len(self.scan_flag)}"
                )
            # Coerce scan_flag to int8; downstream code indexes with the
            # SCAN_FLAG_* constants, which are int, and the ECSV writer
            # expects a fixed dtype. Assignment on the non-frozen dataclass
            # is safe.
            if self.scan_flag.dtype != np.int8:
                self.scan_flag = np.asarray(self.scan_flag, dtype=np.int8)
        for name, arr in [
            ("times", self.times),
            ("az", self.az),
            ("el", self.el),
            ("az_vel", self.az_vel),
            ("el_vel", self.el_vel),
        ]:
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Non-finite values (NaN or Inf) detected in '{name}' array")

    @property
    def duration(self) -> float:
        """Total duration of trajectory in seconds."""
        return float(self.times[-1] - self.times[0])

    @property
    def n_points(self) -> int:
        """Number of trajectory points."""
        return len(self.times)

    @property
    def pattern_type(self) -> str | None:
        """Pattern type from metadata, if available."""
        return self.metadata.pattern_type if self.metadata else None

    @property
    def pattern_params(self) -> dict[str, Any] | None:
        """Pattern parameters from metadata, if available."""
        return self.metadata.pattern_params if self.metadata else None

    @property
    def center_ra(self) -> float | None:
        """Center RA from metadata, if available."""
        return self.metadata.center_ra if self.metadata else None

    @property
    def center_dec(self) -> float | None:
        """Center Dec from metadata, if available."""
        return self.metadata.center_dec if self.metadata else None

    @property
    def science_mask(self) -> np.ndarray:
        """Boolean mask that is True for science-quality samples.

        Returns True where ``scan_flag == SCAN_FLAG_SCIENCE``, or all True
        if ``scan_flag`` is None (no flagging information available).

        Returns
        -------
        np.ndarray
            Boolean array with the same length as ``times``.
        """
        if self.scan_flag is None:
            return np.ones(self.n_points, dtype=bool)
        return self.scan_flag == SCAN_FLAG_SCIENCE

    @property
    def az_accel(self) -> np.ndarray:
        """Azimuth acceleration in degrees/second^2.

        Computed as the numerical gradient of azimuth velocity with
        respect to time using ``np.gradient``.

        Returns
        -------
        np.ndarray
            Azimuth acceleration at each trajectory point.
        """
        return np.gradient(self.az_vel, self.times)

    @property
    def el_accel(self) -> np.ndarray:
        """Elevation acceleration in degrees/second^2.

        Computed as the numerical gradient of elevation velocity with
        respect to time using ``np.gradient``.

        Returns
        -------
        np.ndarray
            Elevation acceleration at each trajectory point.
        """
        return np.gradient(self.el_vel, self.times)

    @property
    def az_jerk(self) -> np.ndarray:
        """Azimuth jerk in degrees/second^3.

        Computed as the numerical gradient of azimuth acceleration with
        respect to time using ``np.gradient``.

        Returns
        -------
        np.ndarray
            Azimuth jerk at each trajectory point.
        """
        return np.gradient(self.az_accel, self.times)

    @property
    def el_jerk(self) -> np.ndarray:
        """Elevation jerk in degrees/second^3.

        Computed as the numerical gradient of elevation acceleration with
        respect to time using ``np.gradient``.

        Returns
        -------
        np.ndarray
            Elevation jerk at each trajectory point.
        """
        return np.gradient(self.el_accel, self.times)

    def __repr__(self) -> str:
        pattern_info = f", pattern={self.pattern_type}" if self.pattern_type else ""
        return (
            f"Trajectory(n_points={self.n_points}, "
            f"duration={self.duration:.1f}s, "
            f"az=[{self.az.min():.1f}, {self.az.max():.1f}]deg, "
            f"el=[{self.el.min():.1f}, {self.el.max():.1f}]deg"
            f"{pattern_info})"
        )
