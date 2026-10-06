"""Trajectory container for telescope scan patterns.

This module provides the Trajectory class which holds time-stamped
position and velocity setpoints for both azimuth and elevation axes,
suitable for upload to the telescope ACU.

The trajectory data is intentionally minimal; metadata about pattern
type, generation parameters, and input coordinates can be attached
via the optional ``metadata`` attribute.

Utility functions (validate, export, format) are free functions in
:mod:`fyst_trajectories.trajectory_utils`; plotting lives in
:mod:`fyst_trajectories.visualization`. Keeping them out of the container
leaves it free of intra-package imports, apart from a private read-only
mapping type that itself imports nothing.

Examples
--------
Create a trajectory manually:

>>> import numpy as np
>>> from fyst_trajectories import get_fyst_site, Trajectory
>>> times = np.array([0, 1, 2, 3, 4])
>>> az = np.array([100, 101, 102, 101, 100])
>>> el = np.full(5, 45.0)
>>> az_vel = np.array([1, 1, 0, -1, -1])
>>> el_vel = np.zeros(5)
>>> traj = Trajectory(times=times, az=az, el=el, az_vel=az_vel, el_vel=el_vel)

Use with pattern generators:

>>> from astropy.time import Time
>>> site = get_fyst_site()
>>> from fyst_trajectories.patterns import TrajectoryBuilder, PongScanConfig
>>> start_time = Time("2026-03-15T01:00:00", scale="utc")
>>> trajectory = (
...     TrajectoryBuilder(site)
...     .at(ra=180.0, dec=-30.0)
...     .with_config(
...         PongScanConfig(
...             timestep=0.1,
...             width=2.0,
...             height=2.0,
...             spacing=0.1,
...             velocity=0.4,
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
>>> print_trajectory(traj, head=3, tail=2)
Trajectory(n_points=5, duration=4.0s, az=[100.0, 102.0]deg, el=[45.0, 45.0]deg)
<BLANKLINE>
   t (s)          az          el      az_vel      el_vel
--------------------------------------------------------
    0.00    100.0000     45.0000      1.0000      0.0000
    1.00    101.0000     45.0000      1.0000      0.0000
    2.00    102.0000     45.0000      0.0000      0.0000
    3.00    101.0000     45.0000     -1.0000      0.0000
    4.00    100.0000     45.0000     -1.0000      0.0000
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from astropy.time import Time

from ._readonly import ReadOnlyDict

SCAN_FLAG_UNCLASSIFIED: int = 0
SCAN_FLAG_SCIENCE: int = 1
SCAN_FLAG_TURNAROUND: int = 2
SCAN_FLAG_RETUNE: int = 3


def _time_derivative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Return the derivative of ``values`` with respect to ``times``.

    ``np.gradient`` needs at least two samples; a single sample has no time
    step, so its derivative is returned as zeros.
    """
    if len(times) < 2:
        return np.zeros(len(times))
    return np.gradient(values, times)


@dataclass(frozen=True)
class RetuneEvent:
    """A single retune event in trajectory-relative seconds.

    Parameters
    ----------
    t_start : float
        Seconds from the trajectory start (i.e. ``trajectory.times[0]``).
        Must be finite and non-negative. Events starting at or after the
        trajectory end (``trajectory.times[-1]``) are skipped with a
        :class:`~fyst_trajectories.exceptions.PointingWarning`.
    duration : float
        Wall-clock duration of the event in seconds. Must be positive.
        Events that would extend past the trajectory end are clipped to
        the trajectory end (matching the uniform-cadence path's behaviour).

    Raises
    ------
    ValueError
        If ``t_start`` is non-finite or negative, or ``duration`` is
        non-finite or not positive.

    Notes
    -----
    The minimal two-field shape is deliberate: no public KID-camera
    retune log has been published, so the dataclass captures only the
    fields every event must have. Per-module staggering is the caller's
    composition: invoke
    :func:`~fyst_trajectories.retune.inject_retune` once per
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


@dataclass(frozen=True)
class TrajectoryMetadata:
    """Metadata about how a trajectory was generated.

    Attached to a :class:`~fyst_trajectories.trajectory.Trajectory` when you need to preserve the
    pattern type and the parameters it was built from.

    Parameters
    ----------
    pattern_type : str
        Name of the pattern that generated this trajectory.
    pattern_params : Mapping
        Parameters used to generate the pattern, stored as a read-only copy.
        It is still a ``dict``, but item assignment, ``update``, ``pop`` and
        the other mutating methods raise ``TypeError``, since a trajectory and
        every copy derived from it share one metadata object. Derive edited
        metadata with :func:`dataclasses.replace`, and convert with
        ``dict(...)`` before editing a copy or dumping it to YAML.
    center_ra : float, optional
        Right Ascension of pattern center in degrees.
    center_dec : float, optional
        Declination of pattern center in degrees.
    target_name : str, optional
        Name of the target (e.g., "M42", "mars").
    input_frame : str, optional
        The input coordinate frame used for the pattern center:
        ``"icrs"`` for celestial patterns, ``None`` for AltAz patterns.
        Default is None.
    """

    pattern_type: str
    pattern_params: Mapping[str, Any] = field(default_factory=dict, hash=False)
    center_ra: float | None = None
    center_dec: float | None = None
    target_name: str | None = None
    input_frame: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pattern_params", ReadOnlyDict(self.pattern_params))


@dataclass(frozen=True, eq=False)
class Trajectory:
    """Container for a telescope trajectory.

    Holds time-stamped position and velocity setpoints for both
    azimuth and elevation axes, suitable for upload to the ACU.

    Parameters
    ----------
    times : np.ndarray or sequence of float
        Timestamps in seconds; must be strictly increasing. They need not
        start at 0, since the absolute-time and ``/path`` exports are taken
        relative to ``times[0]``.
    az : np.ndarray or sequence of float
        Azimuth positions in degrees.
    el : np.ndarray or sequence of float
        Elevation positions in degrees.
    az_vel : np.ndarray or sequence of float
        Azimuth velocities in degrees/second.
    el_vel : np.ndarray or sequence of float
        Elevation velocities in degrees/second.
    start_time : Time, optional
        Absolute start time for the trajectory.
    metadata : TrajectoryMetadata, optional
        Optional metadata about pattern generation.
    scan_flag : np.ndarray or sequence of int or None, optional
        Per-sample scan phase flag: 0 = unclassified, 1 = constant-velocity
        science sweep, 2 = turnaround, 3 = retune
        pause (``SCAN_FLAG_RETUNE``, written by
        :func:`~fyst_trajectories.retune.inject_retune`). None
        means no flagging info is available. Stored as a one-dimensional
        int8 array.
    retune_events : tuple of RetuneEvent, optional
        Event-level provenance for the ``SCAN_FLAG_RETUNE`` entries in
        ``scan_flag``. Populated by
        :func:`~fyst_trajectories.retune.inject_retune` for both
        the uniform-cadence and explicit event-list code paths. Empty tuple
        (the default) means no retune events have been injected. See
        :class:`RetuneEvent`.

    Raises
    ------
    ValueError
        If any of ``times``/``az``/``el``/``az_vel``/``el_vel`` or
        ``scan_flag`` is not one-dimensional, ``times`` is empty, any
        array's length (including ``scan_flag``'s) differs from ``times``,
        any of ``times``/``az``/``el``/``az_vel``/``el_vel`` contains a
        non-finite value, or ``times`` is not strictly increasing.

    Notes
    -----
    ``times``, ``az``, ``el``, ``az_vel`` and ``el_vel`` are stored as
    one-dimensional float64 arrays. A float64 array is stored as given,
    without a copy; any other input (a list, an integer array) is converted
    once at construction. An astropy ``Quantity`` is stored as its bare
    numeric value in its own unit, so convert it to degrees or seconds first.

    Instances are immutable: the dataclass is ``frozen=True``, so rebinding a
    field (for example ``trajectory.az = new_array``) raises
    :class:`dataclasses.FrozenInstanceError`. Derive a modified copy with
    :func:`dataclasses.replace` instead. Freezing prevents field rebinding
    only; the contents of the stored NumPy arrays are not made read-only.

    Trajectories compare and hash by identity: ``==`` is ``True`` only for
    the same object, so a :func:`dataclasses.replace` copy is unequal to its
    original, and a trajectory can be a set member or a dictionary key.
    Compare two trajectories field by field with :func:`numpy.array_equal`.
    """

    times: np.ndarray
    az: np.ndarray
    el: np.ndarray
    az_vel: np.ndarray
    el_vel: np.ndarray
    start_time: Time | None = None
    metadata: TrajectoryMetadata | None = field(default=None, repr=False)
    scan_flag: np.ndarray | None = None
    retune_events: tuple[RetuneEvent, ...] = ()

    def __post_init__(self) -> None:
        # Store the five arrays as one-dimensional float64. A float64 array
        # passes through np.asarray uncopied; anything else (a list, an
        # integer array) is converted once, so arithmetic on the stored
        # arrays never wraps and repr can call .min()/.max().
        for name in ("times", "az", "el", "az_vel", "el_vel"):
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError(f"'{name}' must be one-dimensional, got shape {arr.shape}")
            object.__setattr__(self, name, arr)
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
            flag = np.asarray(self.scan_flag)
            if flag.ndim != 1:
                raise ValueError(f"'scan_flag' must be one-dimensional, got shape {flag.shape}")
            if len(flag) != n:
                raise ValueError(
                    f"Array length mismatch: times has {n} elements but scan_flag has {len(flag)}"
                )
            # Coerce scan_flag to int8; downstream indexes with the int-valued
            # SCAN_FLAG_* constants and the pattern generators all produce int8,
            # so the field has one dtype everywhere. object.__setattr__ performs
            # the one-time canonicalisation under frozen=True (the sibling value
            # types coerce the same way).
            if flag.dtype != np.int8:
                flag = np.asarray(flag, dtype=np.int8)
            object.__setattr__(self, "scan_flag", flag)
        for name, arr in [
            ("times", self.times),
            ("az", self.az),
            ("el", self.el),
            ("az_vel", self.az_vel),
            ("el_vel", self.el_vel),
        ]:
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Non-finite values (NaN or Inf) detected in '{name}' array")
        if n > 1:
            # Every derived quantity divides by a time step: the acceleration
            # and jerk properties, and the dynamics validator. A repeated or
            # out-of-order sample turns those into inf or nan silently, so
            # reject it at construction where the caller can see it.
            steps = np.diff(self.times)
            if not np.all(steps > 0):
                first = int(np.argmax(steps <= 0))
                raise ValueError(
                    "Trajectory times must be strictly increasing; "
                    f"times[{first + 1}]={float(self.times[first + 1])!r} does not exceed "
                    f"times[{first}]={float(self.times[first])!r}"
                )

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
    def pattern_params(self) -> Mapping[str, Any] | None:
        """Pattern parameters from metadata, if available (read-only)."""
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
            Azimuth acceleration at each trajectory point; zeros for a
            single-sample trajectory.
        """
        return _time_derivative(self.az_vel, self.times)

    @property
    def el_accel(self) -> np.ndarray:
        """Elevation acceleration in degrees/second^2.

        Computed as the numerical gradient of elevation velocity with
        respect to time using ``np.gradient``.

        Returns
        -------
        np.ndarray
            Elevation acceleration at each trajectory point; zeros for a
            single-sample trajectory.
        """
        return _time_derivative(self.el_vel, self.times)

    @property
    def az_jerk(self) -> np.ndarray:
        """Azimuth jerk in degrees/second^3.

        Computed as the numerical gradient of azimuth acceleration with
        respect to time using ``np.gradient``.

        Returns
        -------
        np.ndarray
            Azimuth jerk at each trajectory point; zeros for a
            single-sample trajectory.
        """
        return _time_derivative(self.az_accel, self.times)

    @property
    def el_jerk(self) -> np.ndarray:
        """Elevation jerk in degrees/second^3.

        Computed as the numerical gradient of elevation acceleration with
        respect to time using ``np.gradient``.

        Returns
        -------
        np.ndarray
            Elevation jerk at each trajectory point; zeros for a
            single-sample trajectory.
        """
        return _time_derivative(self.el_accel, self.times)

    def __repr__(self) -> str:
        pattern_info = f", pattern={self.pattern_type}" if self.pattern_type else ""
        return (
            f"Trajectory(n_points={self.n_points}, "
            f"duration={self.duration:.1f}s, "
            f"az=[{self.az.min():.1f}, {self.az.max():.1f}]deg, "
            f"el=[{self.el.min():.1f}, {self.el.max():.1f}]deg"
            f"{pattern_info})"
        )
