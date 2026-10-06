"""Custom exceptions for the fyst-trajectories library.

This module defines a hierarchy of exceptions for graceful error handling
when trajectories exceed telescope limits or targets are not observable.

Warning Hierarchy
-----------------
::

    PointingWarning(UserWarning)
        VelocityLimitWarning
        AccelerationLimitWarning

Exception Hierarchy
-------------------
::

    PointingError (ValueError)
        TrajectoryBoundsError
            AzimuthBoundsError
            ElevationBoundsError
        TargetNotObservableError
        EncoderSolutionError
        OffsetInversionError
        DwellExceedsCrossingError

The offline simulator adds two more ``PointingError`` subclasses,
``ScanParamsSchemaError`` and ``BlockNotReconstructableError``, in
``fyst_trajectories.overhead.exceptions``; they live in that tier and are
not re-exported here.

One rule decides which of the two an error is. A ``PointingError``
means a well-formed request that cannot be satisfied for this site,
target and time: a target outside the telescope limits, no elevation
crossing within the search window, a dwell longer than the solved
crossing. A plain ``ValueError`` means a malformed request: a
non-positive timestep, a duplicate pattern name, an unrecognised
config. The offline simulator's ``ScanParamsSchemaError`` and
``BlockNotReconstructableError`` report a recorded block that cannot be
rebuilt.

``PointingError`` inherits from ``ValueError``, and every exception
below it inherits from ``PointingError``, so all of them can be caught
as ``ValueError``. The reason is live rather than historical: a control
system's ``except ValueError`` around a dispatch, and the offline
simulator's best-effort rebuild path, both rely on it. A handler that
treats the two differently catches ``PointingError`` first, to tell an
infeasibility from a malformed argument.

The bounds family and :class:`EncoderSolutionError` overlap at one
condition, deliberately. ``TrajectoryBoundsError`` and its two
subclasses report a *trajectory or position* that violates a limit;
``EncoderSolutionError`` reports that *no commandable encoder solution
exists*. ``cause="goal_elevation"`` is where the two coincide, because
the goal elevation is refused before any azimuth wrap is considered, so
the caller learns there is no solution rather than which sample was out
of range.

Every exception in the hierarchy survives :mod:`pickle` and
:mod:`copy`. The structured subclasses take more than the single
message argument :class:`BaseException` reconstructs from, so each one
defines ``__reduce__``; a consumer that marshals errors across a
process boundary gets the attributes back, not a ``TypeError``.
"""

import functools
from collections.abc import Sequence
from typing import Literal

#: The stages at which :func:`~fyst_trajectories.dispatch.choose_encoder_solution`
#: can refuse a goal; the vocabulary of ``EncoderSolutionError.cause``.
EncoderSolutionCause = Literal[
    "goal_elevation", "no_image", "span_unreachable", "sun_blocked", "path_blocked"
]


class PointingWarning(UserWarning):
    """Base warning class for fyst-trajectories.

    Allows users to filter fyst-trajectories warnings specifically::

        import warnings
        from fyst_trajectories.exceptions import PointingWarning

        warnings.filterwarnings("ignore", category=PointingWarning)
    """


class VelocityLimitWarning(PointingWarning):
    """A trajectory's velocity exceeds a configured axis limit.

    Subclass of :class:`PointingWarning`, so ``except PointingWarning``
    handlers still catch it. Filter on this category (``issubclass`` or
    ``warnings.filterwarnings(category=...)``) instead of matching the
    message text.
    """


class AccelerationLimitWarning(PointingWarning):
    """A trajectory's acceleration exceeds a configured axis limit.

    Subclass of :class:`PointingWarning`, the acceleration counterpart of
    :class:`VelocityLimitWarning`.
    """


class PointingError(ValueError):
    """A well-formed request that cannot be satisfied for this site, target and time.

    The base class of the library's own error types. A malformed request
    raises plain ``ValueError`` instead. ``PointingError`` inherits from
    ``ValueError`` (see the module docstring for why), so
    ``except ValueError`` is the handler that catches both, and a handler
    that treats them differently catches ``PointingError`` first.

    Examples
    --------
    Narrow to ``except PointingError`` to catch a request that cannot be
    satisfied and let plain ``ValueError`` argument checks through:

    >>> from fyst_trajectories import validate_trajectory
    >>> from fyst_trajectories.exceptions import PointingError
    >>> try:
    ...     validate_trajectory(trajectory, site)
    ... except PointingError as exc:
    ...     print(f"Pointing error: {exc}")
    """


class TrajectoryBoundsError(PointingError):
    """Raised when a trajectory exceeds telescope position limits.

    Parameters
    ----------
    axis : str
        The axis that exceeded limits ("azimuth" or "elevation").
    actual_min : float
        Minimum value in the trajectory (degrees).
    actual_max : float
        Maximum value in the trajectory (degrees).
    limit_min : float
        Allowed minimum (degrees).
    limit_max : float
        Allowed maximum (degrees).

    Examples
    --------
    Catch and inspect a bounds error:

    >>> from fyst_trajectories import validate_trajectory
    >>> from fyst_trajectories.exceptions import TrajectoryBoundsError
    >>> try:
    ...     validate_trajectory(trajectory, site)
    ... except TrajectoryBoundsError as exc:
    ...     print(f"Axis: {exc.axis}")
    ...     print(f"Actual: [{exc.actual_min:.2f}, {exc.actual_max:.2f}]")
    ...     print(f"Limits: [{exc.limit_min}, {exc.limit_max}]")
    """

    def __init__(
        self,
        axis: str,
        actual_min: float,
        actual_max: float,
        limit_min: float,
        limit_max: float,
    ):
        self.axis = axis
        self.actual_min = actual_min
        self.actual_max = actual_max
        self.limit_min = limit_min
        self.limit_max = limit_max
        message = (
            f"Trajectory {axis} [{actual_min:.2f}, {actual_max:.2f}] "
            f"exceeds limits [{limit_min}, {limit_max}]. "
            f"Check that the target is observable at the requested time, "
            f"or adjust scan parameters to stay within telescope limits."
        )
        super().__init__(message)

    def __reduce__(self):
        """Return the pickle/copy reconstruction of this error."""
        return (
            type(self),
            (self.axis, self.actual_min, self.actual_max, self.limit_min, self.limit_max),
            self.__dict__,
        )


class AzimuthBoundsError(TrajectoryBoundsError):
    """Raised when trajectory azimuth exceeds telescope limits.

    Parameters
    ----------
    actual_min : float
        Minimum azimuth in the trajectory (degrees).
    actual_max : float
        Maximum azimuth in the trajectory (degrees).
    limit_min : float
        Allowed minimum azimuth (degrees).
    limit_max : float
        Allowed maximum azimuth (degrees).

    Examples
    --------
    >>> import numpy as np
    >>> from fyst_trajectories.exceptions import AzimuthBoundsError
    >>> from fyst_trajectories.trajectory_utils import validate_trajectory_bounds
    >>> az = np.array([100.0, 400.0])  # 400 exceeds the +360 encoder limit
    >>> el = np.array([45.0, 45.0])
    >>> try:
    ...     validate_trajectory_bounds(site, az, el)
    ... except AzimuthBoundsError as exc:
    ...     print(
    ...         f"Az [{exc.actual_min:.1f}, {exc.actual_max:.1f}] "
    ...         f"exceeds [{exc.limit_min}, {exc.limit_max}]"
    ...     )
    Az [100.0, 400.0] exceeds [-180.0, 360.0]
    """

    def __init__(
        self,
        actual_min: float,
        actual_max: float,
        limit_min: float,
        limit_max: float,
    ):
        super().__init__("azimuth", actual_min, actual_max, limit_min, limit_max)

    def __reduce__(self):
        """Return the pickle/copy reconstruction of this error."""
        return (
            type(self),
            (self.actual_min, self.actual_max, self.limit_min, self.limit_max),
            self.__dict__,
        )


class ElevationBoundsError(TrajectoryBoundsError):
    """Raised when trajectory elevation exceeds telescope limits.

    Parameters
    ----------
    actual_min : float
        Minimum elevation in the trajectory (degrees).
    actual_max : float
        Maximum elevation in the trajectory (degrees).
    limit_min : float
        Allowed minimum elevation (degrees).
    limit_max : float
        Allowed maximum elevation (degrees).

    Examples
    --------
    >>> import numpy as np
    >>> from fyst_trajectories.exceptions import ElevationBoundsError
    >>> from fyst_trajectories.trajectory_utils import validate_trajectory_bounds
    >>> az = np.array([100.0, 100.0])
    >>> el = np.array([10.0, 45.0])  # 10 is below the 20 deg elevation floor
    >>> try:
    ...     validate_trajectory_bounds(site, az, el)
    ... except ElevationBoundsError as exc:
    ...     print(
    ...         f"El [{exc.actual_min:.1f}, {exc.actual_max:.1f}] "
    ...         f"exceeds [{exc.limit_min}, {exc.limit_max}]"
    ...     )
    El [10.0, 45.0] exceeds [20.0, 90.0]
    """

    def __init__(
        self,
        actual_min: float,
        actual_max: float,
        limit_min: float,
        limit_max: float,
    ):
        super().__init__("elevation", actual_min, actual_max, limit_min, limit_max)

    def __reduce__(self):
        """Return the pickle/copy reconstruction of this error."""
        return (
            type(self),
            (self.actual_min, self.actual_max, self.limit_min, self.limit_max),
            self.__dict__,
        )


class TargetNotObservableError(PointingError):
    """Raised when a target is not observable at the requested time.

    This is a higher-level error that wraps a ``TrajectoryBoundsError``
    with context about which target was being observed.

    Parameters
    ----------
    target : str
        Name or description of the target (e.g., "mars", "RA=180.0 Dec=-30.0").
    time_info : str
        Human-readable time description (e.g., "2026-06-15T04:00:00").
    bounds_error : TrajectoryBoundsError
        The underlying bounds error with structured limit data.
    message : str, optional
        Override for the human-readable message. When ``None`` (default) the
        message is composed from ``target``, ``time_info``, and
        ``bounds_error``. Supply an explicit string when the
        infeasibility is not a plain limit overshoot (e.g. a rate below a
        minimum) and the composed "exceeds limits" wording would misdescribe
        it; ``bounds_error`` is still stored for structured access.

    Examples
    --------
    Catch and inspect an unobservable target error:

    >>> from astropy.time import Time
    >>> from fyst_trajectories.exceptions import TargetNotObservableError
    >>> from fyst_trajectories.patterns import SiderealTrackConfig, TrajectoryBuilder
    >>> t = Time("2026-03-15T23:00:00", scale="utc")  # target below the elevation floor
    >>> try:
    ...     trajectory = (
    ...         TrajectoryBuilder(site)
    ...         .at(ra=180.0, dec=-30.0)
    ...         .with_config(SiderealTrackConfig(timestep=0.1))
    ...         .duration(300.0)
    ...         .starting_at(t)
    ...         .build()
    ...     )
    ... except TargetNotObservableError as exc:
    ...     print(f"Target: {exc.target}")
    ...     print(f"Axis: {exc.bounds_error.axis}")
    Target: RA=180.000 Dec=-30.000
    Axis: elevation
    """

    def __init__(
        self,
        target: str,
        time_info: str,
        bounds_error: TrajectoryBoundsError,
        message: str | None = None,
    ):
        self.target = target
        self.time_info = time_info
        self.bounds_error = bounds_error
        if message is None:
            message = (
                f"{target} is not fully observable at {time_info}. "
                f"The trajectory {bounds_error.axis} "
                f"[{bounds_error.actual_min:.2f}, {bounds_error.actual_max:.2f}] "
                f"exceeds limits [{bounds_error.limit_min}, {bounds_error.limit_max}]. "
                f"Try a different observation time or shorter duration."
            )
        super().__init__(message)

    def __reduce__(self):
        """Return the pickle/copy reconstruction of this error."""
        return (
            type(self),
            (self.target, self.time_info, self.bounds_error, str(self)),
            self.__dict__,
        )


class EncoderSolutionError(PointingError):
    """Raised when no encoder position can be commanded for a goal.

    Raised by :func:`~fyst_trajectories.dispatch.choose_encoder_solution`
    when the goal cannot be reached at all, in any azimuth wrap, or only
    through the Sun. The ``cause`` names which stage refused, so a caller
    can decide between deferring the goal and dropping it without matching
    on the message text.

    Parameters
    ----------
    cause : {"goal_elevation", "no_image", "span_unreachable", \
"sun_blocked", "path_blocked"}
        Which stage refused. ``"goal_elevation"``: the goal elevation is
        outside the telescope limits. ``"no_image"``: no 360 degree image
        of the goal azimuth lands within the azimuth limits.
        ``"span_unreachable"``: the goal itself has an in-range image but
        the requested azimuth span fits no wrap. ``"sun_blocked"``: every
        in-range wrap is inside the Sun avoidance zone at some requested
        time. ``"path_blocked"``: every point-safe wrap has a direct slew
        path through the Sun avoidance zone.
    message : str
        Human-readable description, composed at the raise site from the
        values that refused.
    goal_az, goal_el : float
        The requested sky azimuth and elevation in degrees.
    current_az, current_el : float, optional
        The encoder position the slew would have started from, when the
        stage that refused depends on it (``"path_blocked"``).
    candidates : sequence of float, optional
        Encoder azimuths that survived the previous stages, in degrees:
        the in-range wraps for ``"sun_blocked"``, the point-safe wraps for
        ``"path_blocked"``, empty otherwise. A caller planning a detour
        starts from these.
    time_iso : str, optional
        ISO UTC time (or time range) at which the refusing check was
        evaluated, when one applies.

    Examples
    --------
    Catch a refused goal and branch on its cause:

    >>> from astropy.time import Time
    >>> from fyst_trajectories import get_fyst_site
    >>> from fyst_trajectories.dispatch import choose_encoder_solution
    >>> from fyst_trajectories.exceptions import EncoderSolutionError
    >>> site = get_fyst_site(sun_avoidance_enabled=False)
    >>> t = Time("2026-03-15T12:00:00", scale="utc")
    >>> try:
    ...     choose_encoder_solution(190.0, 45.0, 200.0, 10.0, t, site)
    ... except EncoderSolutionError as exc:
    ...     print(exc.cause, exc.goal_el)
    goal_elevation 10.0
    """

    def __init__(
        self,
        cause: "EncoderSolutionCause",
        message: str,
        *,
        goal_az: float,
        goal_el: float,
        current_az: float | None = None,
        current_el: float | None = None,
        candidates: "Sequence[float]" = (),
        time_iso: str | None = None,
    ):
        self.cause = cause
        self.goal_az = goal_az
        self.goal_el = goal_el
        self.current_az = current_az
        self.current_el = current_el
        self.candidates = tuple(float(az) for az in candidates)
        self.time_iso = time_iso
        super().__init__(message)

    def __reduce__(self):
        """Return the pickle/copy reconstruction of this error.

        The structured fields are keyword-only, so the reconstruction
        callable is a :func:`functools.partial` that binds them.
        """
        return (
            functools.partial(
                type(self),
                goal_az=self.goal_az,
                goal_el=self.goal_el,
                current_az=self.current_az,
                current_el=self.current_el,
                candidates=self.candidates,
                time_iso=self.time_iso,
            ),
            (self.cause, str(self)),
            self.__dict__,
        )


class OffsetInversionError(PointingError):
    """Raised when a detector position cannot be inverted to a boresight.

    The focal-plane inverse
    (:func:`~fyst_trajectories.offsets.detector_to_boresight` and the
    trajectory-level :func:`~fyst_trajectories.offsets.apply_detector_offset`)
    refuses in two situations: the requested detector position is within
    the pole guard, where azimuth is degenerate and the residual check
    cannot validate the answer, or the iterative refinement does not
    converge. Both are geometric infeasibility, so they belong in the
    :class:`PointingError` hierarchy the callers' ``Raises`` sections
    advertise.

    Parameters
    ----------
    message : str
        Human-readable description, composed at the raise site.
    indices : sequence of int, optional
        Positions of the offending samples in the input arrays, for an
        array-valued call. Empty for a scalar call and when the refusal
        is not sample-specific.

    Examples
    --------
    >>> from fyst_trajectories.exceptions import OffsetInversionError
    >>> exc = OffsetInversionError("degenerate at the pole", indices=[3, 4])
    >>> exc.indices
    (3, 4)
    """

    def __init__(self, message: str, *, indices: "Sequence[int]" = ()):
        self.indices = tuple(int(i) for i in indices)
        super().__init__(message)

    def __reduce__(self):
        """Return the pickle/copy reconstruction of this error."""
        return (functools.partial(type(self), indices=self.indices), (str(self),), self.__dict__)


class DwellExceedsCrossingError(PointingError):
    """Raised when a requested dwell is longer than the solved footprint crossing.

    Raised by :func:`~fyst_trajectories.planning.compute_source_ces_params`,
    :func:`~fyst_trajectories.planning.plan_source_ces` and a single-pass
    :func:`~fyst_trajectories.planning.plan_source_ces_passes`. Whether a
    dwell fits depends on the crossing solved for this site, source, time
    and ``el_bore``, so the refusal is a :class:`PointingError`. It carries
    both durations, so a caller can fall back to the full crossing without
    reading the message.

    Parameters
    ----------
    message : str
        Human-readable description, composed at the raise site.
    dwell : float
        The requested time on source in seconds.
    crossing_seconds : float
        The solved footprint crossing in seconds.

    Examples
    --------
    >>> from fyst_trajectories.exceptions import DwellExceedsCrossingError
    >>> exc = DwellExceedsCrossingError(
    ...     "dwell must not exceed the solved footprint crossing",
    ...     dwell=900.0,
    ...     crossing_seconds=581.35,
    ... )
    >>> exc.crossing_seconds
    581.35
    """

    def __init__(self, message: str, *, dwell: float, crossing_seconds: float):
        self.dwell = float(dwell)
        self.crossing_seconds = float(crossing_seconds)
        super().__init__(message)

    def __reduce__(self):
        """Return the pickle/copy reconstruction of this error."""
        return (
            functools.partial(type(self), dwell=self.dwell, crossing_seconds=self.crossing_seconds),
            (str(self),),
            self.__dict__,
        )
