"""Custom exceptions for the fyst-trajectories library.

This module defines a hierarchy of exceptions for graceful error handling
when trajectories exceed telescope limits or targets are not observable.

Warning Hierarchy
-----------------
::

    PointingWarning(UserWarning)

Exception Hierarchy
-------------------
::

    PointingError (ValueError)
        TrajectoryBoundsError
            AzimuthBoundsError
            ElevationBoundsError
        TargetNotObservableError

All exceptions inherit from both ``PointingError`` and ``ValueError``
for backward compatibility with code that catches ``ValueError``.
"""


class PointingWarning(UserWarning):
    """Base warning class for fyst-trajectories.

    Allows users to filter fyst-trajectories warnings specifically::

        import warnings
        from fyst_trajectories.exceptions import PointingWarning

        warnings.filterwarnings("ignore", category=PointingWarning)
    """


class PointingError(ValueError):
    """Base exception for all fyst-trajectories errors.

    Inherits from ``ValueError`` for backward compatibility with
    code that catches ``ValueError``.

    Examples
    --------
    Catch any pointing-related error:

    >>> from fyst_trajectories.exceptions import PointingError
    >>> try:
    ...     trajectory.validate(site)
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

    Attributes
    ----------
    axis : str
        The axis that exceeded limits.
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

    >>> from fyst_trajectories.exceptions import TrajectoryBoundsError
    >>> try:
    ...     trajectory.validate(site)
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
    >>> from fyst_trajectories.exceptions import AzimuthBoundsError
    >>> try:
    ...     validate_trajectory_bounds(site, az, el)
    ... except AzimuthBoundsError as exc:
    ...     print(
    ...         f"Az [{exc.actual_min:.1f}, {exc.actual_max:.1f}] "
    ...         f"exceeds [{exc.limit_min}, {exc.limit_max}]"
    ...     )
    """

    def __init__(
        self,
        actual_min: float,
        actual_max: float,
        limit_min: float,
        limit_max: float,
    ):
        super().__init__("azimuth", actual_min, actual_max, limit_min, limit_max)


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
    >>> from fyst_trajectories.exceptions import ElevationBoundsError
    >>> try:
    ...     validate_trajectory_bounds(site, az, el)
    ... except ElevationBoundsError as exc:
    ...     print(
    ...         f"El [{exc.actual_min:.1f}, {exc.actual_max:.1f}] "
    ...         f"exceeds [{exc.limit_min}, {exc.limit_max}]"
    ...     )
    """

    def __init__(
        self,
        actual_min: float,
        actual_max: float,
        limit_min: float,
        limit_max: float,
    ):
        super().__init__("elevation", actual_min, actual_max, limit_min, limit_max)


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

    Attributes
    ----------
    target : str
        Name or description of the target.
    time_info : str
        Human-readable time description.
    bounds_error : TrajectoryBoundsError
        The underlying bounds error with structured limit data.

    Examples
    --------
    Catch and inspect an unobservable target error:

    >>> from fyst_trajectories.exceptions import TargetNotObservableError
    >>> try:
    ...     trajectory = pattern.generate(site, duration=300.0, start_time=t)
    ... except TargetNotObservableError as exc:
    ...     print(f"Target: {exc.target}")
    ...     print(f"Time: {exc.time_info}")
    ...     print(f"Axis: {exc.bounds_error.axis}")
    ...     print(f"Range: [{exc.bounds_error.actual_min:.1f}, {exc.bounds_error.actual_max:.1f}]")
    """

    def __init__(
        self,
        target: str,
        time_info: str,
        bounds_error: TrajectoryBoundsError,
    ):
        self.target = target
        self.time_info = time_info
        self.bounds_error = bounds_error
        message = (
            f"{target} is not fully observable at {time_info}. "
            f"The trajectory {bounds_error.axis} "
            f"[{bounds_error.actual_min:.2f}, {bounds_error.actual_max:.2f}] "
            f"exceeds limits [{bounds_error.limit_min}, {bounds_error.limit_max}]. "
            f"Try a different observation time or shorter duration."
        )
        super().__init__(message)
