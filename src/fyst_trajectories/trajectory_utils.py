"""Trajectory utility functions.

Free functions for validating, exporting, and formatting
Trajectory objects. These are the primary API; the
:class:`~fyst_trajectories.trajectory.Trajectory` container itself
exposes no methods that delegate here.
"""

import sys
import warnings
from typing import TextIO, TypedDict

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

from .coordinates import Coordinates
from .exceptions import (
    AccelerationLimitWarning,
    AzimuthBoundsError,
    ElevationBoundsError,
    PointingWarning,
    VelocityLimitWarning,
)
from .site import Site
from .sun_protocols import SunSafePredicate, _sun_verdicts
from .trajectory import (
    SCAN_FLAG_SCIENCE,
    SCAN_FLAG_UNCLASSIFIED,
    Trajectory,
)


def validate_trajectory_bounds(
    site: Site,
    az: np.ndarray,
    el: np.ndarray,
) -> None:
    """Validate that all trajectory points are within telescope limits.

    Parameters
    ----------
    site : Site
        Telescope site configuration containing telescope_limits.
    az : np.ndarray
        Azimuth positions in degrees.
    el : np.ndarray
        Elevation positions in degrees.

    Raises
    ------
    ValueError
        If any az/el value is non-finite (NaN or Inf).
    AzimuthBoundsError
        If any point exceeds telescope azimuth limits.
    ElevationBoundsError
        If any point exceeds telescope elevation limits.

    Examples
    --------
    >>> from fyst_trajectories import get_fyst_site
    >>> site = get_fyst_site()
    >>> az = np.array([100, 150, 200])
    >>> el = np.array([45, 50, 55])
    >>> validate_trajectory_bounds(site, az, el)  # Passes if within limits
    """
    limits = site.telescope_limits

    # Guard non-finite input: NaN/Inf would slip through every comparison
    # below (``NaN > limit`` is ``False``), silently passing the raising gate.
    if not (np.all(np.isfinite(az)) and np.all(np.isfinite(el))):
        raise ValueError("Non-finite values (NaN or Inf) detected in trajectory az/el arrays")

    az_min, az_max = float(az.min()), float(az.max())
    if az_min < limits.azimuth.min or az_max > limits.azimuth.max:
        raise AzimuthBoundsError(
            actual_min=az_min,
            actual_max=az_max,
            limit_min=limits.azimuth.min,
            limit_max=limits.azimuth.max,
        )

    el_min, el_max = float(el.min()), float(el.max())
    if el_min < limits.elevation.min or el_max > limits.elevation.max:
        raise ElevationBoundsError(
            actual_min=el_min,
            actual_max=el_max,
            limit_min=limits.elevation.min,
            limit_max=limits.elevation.max,
        )


#: Relative tolerance of the velocity limit comparison: far below any
#: physical margin, far above the rounding of a numerical derivative.
_VELOCITY_LIMIT_RTOL: float = 1e-9

#: Angular resolution, in degrees, of the high-elevation advisory: a
#: trajectory whose azimuth travels less (the span of its unwrapped azimuths)
#: does not move in azimuth, and the numbers quoted are those of the fastest
#: sample within it of the highest one judged. It is far above what rounding
#: and float32 storage leave in an angle (one float32 step near 360 deg is
#: 3.05e-5 deg; double precision leaves far less) and far below any scan's extent.
_ADVISORY_ANGLE_RESOLUTION_DEG: float = 1e-3


def _exceeds_velocity_limit(peak: float, limit: float) -> bool:
    """Return whether a velocity peak exceeds its limit by more than rounding.

    A scan planned exactly at a velocity limit can return a peak a few parts
    in 1e14 above it; that is not a breach. A peak counts as one only when it
    exceeds ``limit`` by more than ``_VELOCITY_LIMIT_RTOL`` of the limit.
    """
    return peak > limit * (1.0 + _VELOCITY_LIMIT_RTOL)


def validate_trajectory_dynamics(
    site: Site,
    az: np.ndarray,
    el: np.ndarray,
    times: np.ndarray,
) -> None:
    """Check that trajectory velocities and accelerations are within limits.

    Computes numerical derivatives of position to estimate velocity and
    acceleration, then warns if they exceed the telescope's configured
    limits.

    Limit violations are advisory (warnings) because exceeding a dynamics
    limit does not make a trajectory unexecutable; the telescope simply
    tracks slower than requested at those points. Malformed input, however,
    raises ``ValueError``: non-finite (NaN/Inf) ``az``/``el``/``times`` would
    otherwise pass silently (``NaN > limit`` is ``False``), and non-monotonic
    timestamps make the numerical derivative divide by zero.

    The velocity peak of each axis is compared with its limit at a relative
    tolerance of 1e-9, so a scan planned exactly at a velocity limit does not
    warn when the derivative's rounding puts its peak slightly above the limit.
    Accelerations are compared with their limits exactly.

    Only velocity and acceleration are checked. Third-derivative (jerk)
    limiting is the ACU motion profiler's responsibility; this library does
    not validate jerk and ``AxisLimits`` carries no ``max_jerk`` field, even
    though the Go TCS defines hardware jerk limits (az 12, el 6 deg/s^3).

    Parameters
    ----------
    site : Site
        Telescope site configuration containing telescope_limits with
        max_velocity and max_acceleration for each axis.
    az : np.ndarray
        Azimuth positions in degrees.
    el : np.ndarray
        Elevation positions in degrees.
    times : np.ndarray
        Timestamps in seconds.

    Raises
    ------
    ValueError
        If any ``az``/``el``/``times`` value is non-finite (NaN or Inf), or
        if the timestamps are not strictly increasing.

    Warns
    -----
    VelocityLimitWarning
        If any axis velocity exceeds its configured limit by more than the
        relative tolerance of 1e-9.
    AccelerationLimitWarning
        If any axis acceleration exceeds its configured limit.
    PointingWarning
        If the trajectory has too few points for meaningful validation, or if
        high elevation compresses on-sky azimuth motion: of the samples that
        move at half the trajectory's top azimuth speed or more, the highest
        has cos(el) below 0.5 (elevation above about 60 deg), so its on-sky
        azimuth speed is under half its coordinate rate. The numbers quoted
        are those of the fastest sample within 0.001 deg of that highest one.
        A trajectory whose azimuth spans less than 0.001 deg (whole turns
        aside) does not move in azimuth and gets no such advisory.
    """
    if not (np.all(np.isfinite(az)) and np.all(np.isfinite(el)) and np.all(np.isfinite(times))):
        raise ValueError("Non-finite values (NaN or Inf) detected in trajectory az/el/times arrays")

    if len(times) < 2:
        warnings.warn(
            "Trajectory has fewer than 2 points, skipping dynamics validation.",
            PointingWarning,
            stacklevel=2,
        )
        return

    if np.any(np.diff(times) <= 0):
        raise ValueError(
            "Trajectory times must be strictly increasing; found duplicate or "
            "non-monotonic timestamps"
        )

    limits = site.telescope_limits

    az_unwrapped = np.unwrap(az, period=360.0)
    az_vel = np.gradient(az_unwrapped, times)
    el_vel = np.gradient(el, times)

    max_az_vel = np.abs(az_vel).max()
    max_el_vel = np.abs(el_vel).max()

    if _exceeds_velocity_limit(max_az_vel, limits.azimuth.max_velocity):
        warnings.warn(
            f"Trajectory azimuth velocity ({max_az_vel:.2f} deg/s) exceeds "
            f"limit ({limits.azimuth.max_velocity:.2f} deg/s).",
            VelocityLimitWarning,
            stacklevel=2,
        )

    if _exceeds_velocity_limit(max_el_vel, limits.elevation.max_velocity):
        warnings.warn(
            f"Trajectory elevation velocity ({max_el_vel:.2f} deg/s) exceeds "
            f"limit ({limits.elevation.max_velocity:.2f} deg/s).",
            VelocityLimitWarning,
            stacklevel=2,
        )

    # Advisory: at high elevation an azimuth rate carries the beam across less
    # sky, cos(el) times the rate. Of the samples that move at half the top
    # azimuth speed or more, the highest (the smallest cos(el)) decides the
    # verdict; the numbers quoted are those of the fastest sample, then the
    # first, within _ADVISORY_ANGLE_RESOLUTION_DEG of it in elevation, a band
    # that chooses only what is quoted. Half the top speed is far from any
    # rounding scale, so the verdict follows the trajectory, not the rounding of
    # the computed speeds, the time origin or the precision of the input, short
    # of a sample or a travel that sits to within rounding exactly on a threshold
    # (half the top speed, 60 deg, the resolution). A trajectory whose azimuth
    # travels less than the resolution has no azimuth motion to judge, and a
    # speed that overflows (after a time step near the smallest double) is left
    # out.
    cos_el = np.cos(np.radians(el))
    speed = np.abs(az_vel)
    finite = np.isfinite(speed)
    top = speed[finite].max() if finite.any() else 0.0
    moves = np.ptp(az_unwrapped) >= _ADVISORY_ANGLE_RESOLUTION_DEG
    if cos_el.min() > 0 and moves and top > 0:
        fast = np.flatnonzero(finite & (speed >= 0.5 * top))
        if cos_el[fast].min() < 0.5:
            height = np.abs(np.asarray(el, dtype=float))[fast]
            near = fast[height >= height.max() - _ADVISORY_ANGLE_RESOLUTION_DEG]
            i = int(near[np.argmax(speed[near])])
            warnings.warn(
                f"High elevation reduces on-sky azimuth speed to "
                f"{speed[i] * cos_el[i]:.2f} deg/s at the highest sample moving at half "
                f"the top azimuth speed or more (coordinate: {speed[i]:.2f} deg/s, "
                f"cos(el)={cos_el[i]:.3f}). Verify scan design is appropriate.",
                PointingWarning,
                stacklevel=2,
            )

    if len(times) < 4:
        warnings.warn(
            f"Trajectory has only {len(times)} points. Acceleration estimates "
            "require at least 4 points; skipping acceleration validation.",
            PointingWarning,
            stacklevel=2,
        )
        return

    az_accel = np.gradient(az_vel, times)
    el_accel = np.gradient(el_vel, times)

    max_az_accel = np.abs(az_accel).max()
    max_el_accel = np.abs(el_accel).max()

    if max_az_accel > limits.azimuth.max_acceleration:
        warnings.warn(
            f"Trajectory azimuth acceleration ({max_az_accel:.2f} deg/s^2) exceeds "
            f"limit ({limits.azimuth.max_acceleration:.2f} deg/s^2).",
            AccelerationLimitWarning,
            stacklevel=2,
        )

    if max_el_accel > limits.elevation.max_acceleration:
        warnings.warn(
            f"Trajectory elevation acceleration ({max_el_accel:.2f} deg/s^2) exceeds "
            f"limit ({limits.elevation.max_acceleration:.2f} deg/s^2).",
            AccelerationLimitWarning,
            stacklevel=2,
        )


def validate_trajectory(
    trajectory: Trajectory,
    site: Site,
    check_sun: bool = True,
    sun_safe: "SunSafePredicate | None" = None,
) -> None:
    """Validate trajectory against telescope limits.

    Checks position bounds (raises on violation),
    velocity/acceleration limits (warns on violation), and optionally
    sun avoidance constraints (warns on violation).

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory to validate.
    site : Site
        Telescope site with axis limits.
    check_sun : bool, optional
        Whether to check sun avoidance constraints. Default True.
        Sun checking requires ``trajectory.start_time`` to be set;
        if it is None the sun check is skipped silently.
    sun_safe : SunSafePredicate, optional
        Sun-safety predicate implementing the
        :class:`~fyst_trajectories.sun_protocols.SunSafePredicate` contract,
        forwarded to :func:`validate_sun_avoidance`. ``None`` (default)
        keeps the built-in scalar exclusion/warning-radius check; an
        injected predicate is consulted on the subsamples instead, so the
        directional sun-avoidance model
        (see :func:`~fyst_trajectories.sun_models.make_sun_safe`) is honored
        end-to-end. Advisory either way. Has no effect when ``check_sun`` is
        ``False`` or ``trajectory.start_time`` is ``None``.

    Raises
    ------
    AzimuthBoundsError
        If azimuth positions are outside telescope movement range.
    ElevationBoundsError
        If elevation positions are outside telescope movement range.
    ValueError
        If any ``az``/``el``/``times`` value is non-finite, the timestamps
        are not strictly increasing, or an injected ``sun_safe.batch``
        returns a result of the wrong shape.

    Warns
    -----
    VelocityLimitWarning, AccelerationLimitWarning
        If any velocity or acceleration exceeds its configured limit, as
        implied by the positions or, for velocity, as commanded by the
        ``az_vel``/``el_vel`` columns that :func:`to_path_format` uploads.
    PointingWarning
        If the trajectory has too few points for full dynamics validation, if
        high elevation compresses on-sky azimuth motion (see
        :func:`validate_trajectory_dynamics`), or if any subsampled trajectory
        point is within the sun exclusion or warning radius (the sun check
        samples roughly every 60 seconds; see :func:`validate_sun_avoidance`
        for the gap this leaves).
    """
    validate_trajectory_bounds(site, trajectory.az, trajectory.el)
    validate_trajectory_dynamics(site, trajectory.az, trajectory.el, trajectory.times)
    # The velocity columns are what to_path_format uploads; they can disagree
    # with the velocities the positions imply, so check them on their own.
    limits = site.telescope_limits
    for axis, column, limit in (
        ("azimuth", trajectory.az_vel, limits.azimuth.max_velocity),
        ("elevation", trajectory.el_vel, limits.elevation.max_velocity),
    ):
        peak = float(np.abs(column).max())
        if _exceeds_velocity_limit(peak, limit):
            warnings.warn(
                f"Commanded {axis} velocity column ({peak:.2f} deg/s) exceeds "
                f"limit ({limit:.2f} deg/s).",
                VelocityLimitWarning,
                stacklevel=2,
            )
    if check_sun and trajectory.start_time is not None:
        abs_times = get_absolute_times(trajectory)
        validate_sun_avoidance(site, trajectory.az, trajectory.el, abs_times, sun_safe=sun_safe)


def get_absolute_times(trajectory: Trajectory) -> Time:
    """Get absolute timestamps for the trajectory.

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory with a start_time set.

    Returns
    -------
    Time
        Astropy Time array with absolute timestamps.

    Raises
    ------
    ValueError
        If start_time is not set.
    """
    if trajectory.start_time is None:
        raise ValueError("start_time not set; cannot compute absolute times")
    # ``times`` are seconds relative to the trajectory start; take them
    # relative to ``times[0]`` so the first sample maps to ``start_time``
    # even when the trajectory clock does not begin at 0.
    relative = trajectory.times - trajectory.times[0]
    return trajectory.start_time + TimeDelta(relative * u.s)


def validate_sun_avoidance(
    site: Site,
    az: np.ndarray,
    el: np.ndarray,
    times: Time,
    coords: Coordinates | None = None,
    sun_safe: "SunSafePredicate | None" = None,
) -> None:
    """Check sun avoidance constraints, emitting warnings for violations.

    .. warning::

       This check is **advisory only**.  A Sun violation emits a Python
       warning; it never blocks trajectory generation or raises an
       exception.  Nothing downstream of this library is guaranteed to
       enforce sun avoidance; do not rely on a later stage to reject an
       unsafe trajectory.

    The check is also **subsampled, not exhaustive**: the trajectory is
    sampled approximately every 60 seconds and each sampled position is
    compared against the Sun at that same sample time.  Points strictly
    between samples are never checked, and at the 3 deg/s azimuth maximum
    a trajectory can sweep up to 180 degrees of azimuth between samples,
    far more than the 5 degree exclusion-to-warning band.  A fast scan can
    therefore cross the exclusion zone between samples without a warning.

    Parameters
    ----------
    site : Site
        Site configuration with sun avoidance settings.
    az : np.ndarray
        Azimuth array in degrees.
    el : np.ndarray
        Elevation array in degrees.
    times : Time
        Absolute times for each trajectory point, as an astropy
        :class:`~astropy.time.Time` (for example
        :func:`get_absolute_times` of the trajectory); the Sun ephemeris and
        the :class:`~fyst_trajectories.sun_protocols.SunSafePredicate`
        contract both take a ``Time``.
    coords : Coordinates, optional
        Pre-constructed Coordinates instance. Created internally if
        not provided.
    sun_safe : SunSafePredicate, optional
        Sun-safety predicate implementing the
        :class:`~fyst_trajectories.sun_protocols.SunSafePredicate` contract,
        ``(az_deg, el_deg, time) -> bool`` returning ``True`` when the
        position is clear of the Sun. ``None`` (default) keeps the built-in
        scalar exclusion/warning-radius check (vectorised, subsampled
        separation computation). When a predicate is injected it is
        consulted on the subsamples instead, in one call when it exposes the
        ``batch`` extension and per subsample ``(az_i, el_i, time_i)``
        otherwise, so the directional sun-avoidance model
        (see :func:`~fyst_trajectories.sun_models.make_sun_safe`) is honored,
        using the same ~60 s subsampling for parity and performance. The
        injected model owns its own boundary, so only
        an "EXCLUSION ZONE" warning is emitted (no separate warning-radius
        band). Advisory either way. The whole check, injected predicate
        included, is skipped when ``site.sun_avoidance.enabled`` is
        ``False``; enable it on the ``Site`` even when the scalar radii are
        not the model you intend to use. See
        :class:`~fyst_trajectories.sun_protocols.SunSafePredicate`.

    Warns
    -----
    PointingWarning
        If any subsampled trajectory point is within the exclusion radius
        ("EXCLUSION ZONE") or the warning radius ("WARNING ZONE")
        of the Sun. With an injected ``sun_safe`` predicate, an
        "EXCLUSION ZONE" warning is emitted for any subsample the
        predicate reports unsafe. Points between subsamples are not
        checked.

    Raises
    ------
    TypeError
        If ``times`` is not a :class:`~astropy.time.Time` and the site's Sun
        avoidance is enabled (a disabled site returns before any check).
    ValueError
        If an injected ``sun_safe.batch`` returns a result of the wrong
        shape.
    """
    if not site.sun_avoidance.enabled:
        return

    if not isinstance(times, Time):
        # The Sun ephemeris and every injected predicate take a real Time;
        # a numeric array would fail several frames from the caller.
        raise TypeError(
            "times must be an astropy Time (for example get_absolute_times(trajectory)), "
            f"got {type(times).__name__}."
        )

    if coords is None:
        coords = Coordinates(site)

    n_points = len(az)
    if n_points == 0:
        return

    total_seconds = (times[-1] - times[0]).to_value(u.s)

    subsample_interval = 60.0  # seconds
    if total_seconds <= 0:
        step = n_points
    else:
        step = max(1, int(subsample_interval * n_points / total_seconds))

    sample_indices = np.arange(0, n_points, step)
    if sample_indices[-1] != n_points - 1:
        sample_indices = np.append(sample_indices, n_points - 1)

    sample_times = times[sample_indices]
    sample_az = az[sample_indices]
    sample_el = el[sample_indices]

    if sun_safe is not None:
        # Injected directional model: consult it on the subsamples. ``False``
        # marks an unsafe (inside-the-zone) sample. Warn once, naming the
        # first unsafe subsample, mirroring the scalar branch's single
        # closest-approach warning. The ~60 s subsampling above is preserved
        # so injection does not regress performance.
        verdicts = _sun_verdicts(
            sun_safe, sample_az, sample_el, sample_times, what="trajectory subsample"
        )
        unsafe_idx = np.flatnonzero(~verdicts)
        if unsafe_idx.size == 0:
            return
        first = int(unsafe_idx[0])
        first_time_str = sample_times[first].iso
        warnings.warn(
            f"EXCLUSION ZONE: Trajectory at (az={float(sample_az[first]):.1f} deg, "
            f"el={float(sample_el[first]):.1f} deg) is inside the Sun avoidance "
            f"zone at {first_time_str}. This violates the configured Sun "
            f"avoidance policy; nothing downstream is guaranteed to reject it.",
            PointingWarning,
            stacklevel=2,
        )
        return

    sun_az, sun_alt = coords.get_sun_altaz(sample_times)
    sun_az = np.atleast_1d(sun_az)
    sun_alt = np.atleast_1d(sun_alt)

    separations = np.atleast_1d(coords.angular_separation(sample_az, sample_el, sun_az, sun_alt))

    min_idx = int(np.argmin(separations))
    min_sep = float(separations[min_idx])

    exclusion = site.sun_avoidance.exclusion_radius
    warning = site.sun_avoidance.warning_radius

    closest_time_str = sample_times[min_idx].iso

    # Use ``<=`` so a separation exactly at the exclusion radius counts as
    # unsafe, matching ``Coordinates.is_sun_safe`` / ``is_position_observable``
    # and the planning sun checks (the conservative ``sep <= radius`` convention).
    if min_sep <= exclusion:
        warnings.warn(
            f"EXCLUSION ZONE: Trajectory passes {min_sep:.1f} deg from the Sun "
            f"(exclusion radius: {exclusion} deg) at {closest_time_str}. "
            f"This violates the configured Sun avoidance policy; nothing "
            f"downstream is guaranteed to reject it.",
            PointingWarning,
            stacklevel=2,
        )
    elif min_sep < warning:
        warnings.warn(
            f"WARNING ZONE: Trajectory passes {min_sep:.1f} deg from the Sun "
            f"(warning radius: {warning} deg) at {closest_time_str}.",
            PointingWarning,
            stacklevel=2,
        )


def to_arrays(
    trajectory: Trajectory,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Export trajectory as simple arrays for ACU upload.

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory to export.

    Returns
    -------
    times : np.ndarray
        Timestamps in seconds on the trajectory's own clock (a copy of
        ``trajectory.times``); subtract ``times[0]`` for seconds from the
        first sample, as :func:`to_path_format` does.
    az : np.ndarray
        Azimuth positions in degrees.
    el : np.ndarray
        Elevation positions in degrees.
    """
    return trajectory.times.copy(), trajectory.az.copy(), trajectory.el.copy()


#: Minimum spacing between consecutive ``/path`` samples accepted by Go TCS.
#: The ``/path`` receiver hard-rejects any pair closer than 50 ms
#: (ACU ICD 2.0 section 8.9.3).
GO_TCS_MIN_SAMPLE_INTERVAL_SEC: float = 0.05


def to_path_format(trajectory: Trajectory) -> list[list[float]]:
    """Convert trajectory to the ``points`` list for the Go TCS ``/path`` endpoint.

    Converts the trajectory arrays into the ``points`` list the Go TCS
    ``/path`` endpoint expects: a list of ``[t, az, el, az_vel, el_vel]``
    rows, where ``t`` is **relative** seconds from the trajectory start.

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory to convert.

    Returns
    -------
    list
        List of ``[t, az, el, az_vel, el_vel]`` rows. This is only the
        ``points`` array; the full request body also needs ``start_time``
        and ``coordsys`` (see Notes).

    Raises
    ------
    ValueError
        If any consecutive pair of samples is separated by less than
        ``GO_TCS_MIN_SAMPLE_INTERVAL_SEC`` (50 ms). The Go TCS ``/path``
        receiver hard-rejects such a body with HTTP 400 (ACU ICD 2.0 section 8.9.3),
        so the check is enforced here at the serialization boundary rather
        than failing at POST time. A time grid built at exactly 0.05 s
        (``timestep=0.05``) usually carries float round-off just below it,
        which this check and Go TCS both refuse, so choose a timestep above
        0.05 s.

    Notes
    -----
    The Go TCS ``/path`` receiver requires three keys and rejects unknown ones
    (``DisallowUnknownFields``); its one optional key, a ``tags`` object, is not
    emitted here. The body is
    ``{"start_time": <abs Unix s>, "coordsys": "Horizon", "points": [...]}``:

    - ``coordsys`` is **required**; use ``"Horizon"`` for trajectory upload
      (see :func:`to_path_payload` for why). Omitting it or adding a key other
      than ``tags`` yields HTTP 400.
    - ``points`` times are **relative** seconds; ``start_time`` is **absolute**
      Unix seconds (``trajectory.start_time.unix``). Do not conflate the two.

    Examples
    --------
    >>> points = to_path_format(trajectory)
    >>> data = {
    ...     "start_time": trajectory.start_time.unix,
    ...     "coordsys": "Horizon",
    ...     "points": points,
    ... }
    """
    times = np.asarray(trajectory.times)
    # Emit times relative to ``times[0]`` so the first sample lands exactly on
    # ``start_time``, matching ``get_absolute_times``; a trajectory whose clock
    # does not begin at 0 would otherwise be commanded ``times[0]`` late.
    relative = times - times[0]
    if relative.size >= 2:
        # Check the values Go TCS will difference, not the raw clock: the two
        # can disagree in the last bit.
        min_dt = float(np.diff(relative).min())
        if min_dt < GO_TCS_MIN_SAMPLE_INTERVAL_SEC:
            raise ValueError(
                f"Trajectory sample interval {min_dt!r} s is below the Go TCS "
                f"/path minimum of {GO_TCS_MIN_SAMPLE_INTERVAL_SEC} s (ACU ICD 2.0 "
                "section 8.9.3); the upload would be rejected with HTTP 400. A grid "
                "built at exactly 0.05 s carries float round-off below it; increase "
                "the pattern timestep."
            )
    return np.column_stack(
        [
            relative,
            trajectory.az,
            trajectory.el,
            trajectory.az_vel,
            trajectory.el_vel,
        ]
    ).tolist()


class PathPayload(TypedDict):
    """The Go TCS ``/path`` request body that :func:`to_path_payload` returns.

    A plain ``dict`` at runtime, ready to JSON-serialize and POST; the class
    names its three keys and their types for a static type checker.
    """

    start_time: float
    """Absolute Unix seconds of the first sample."""

    coordsys: str
    """Coordinate system of the points, ``"Horizon"`` or ``"ICRS"``."""

    points: list[list[float]]
    """The rows of :func:`to_path_format`, one ``[time, az, el, az_vel, el_vel]``
    per sample, with ``time`` in seconds relative to ``start_time``."""


def to_path_payload(trajectory: Trajectory, coordsys: str = "Horizon") -> PathPayload:
    """Assemble the full Go TCS ``/path`` request body for a trajectory.

    Wraps :func:`to_path_format` with the two scalar keys the Go TCS
    ``/path`` endpoint also requires, returning the three-key dict
    ``{"start_time", "coordsys", "points"}``. Prefer this over assembling the
    body by hand around :func:`to_path_format`: the Go TCS receiver sets
    ``DisallowUnknownFields`` and switches on a required ``coordsys``, so a
    body that omits ``coordsys``, or adds any key other than the optional
    ``tags`` object, is rejected with HTTP 400.

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory to serialize. Must have ``start_time`` set.
    coordsys : str, optional
        Coordinate system for the points. Must be ``"Horizon"`` (default)
        or ``"ICRS"``. Use ``"Horizon"`` for trajectory upload: the rows are
        nominal (vacuum) az/el per P-INCM-ICD-0003-A Eq.(1), carrying no
        refraction, SPEM, or non-repeatable pointing terms. Those are applied
        downstream at execution time: refraction by exactly one of the Go TCS
        or the ACU (section 6), the pointing model in the ACU with any extra
        terms added by the Go TCS (section 6), the tiltmeter term in the ACU
        (section 7). ICRS ``/path`` velocities are unimplemented in Go TCS.

    Returns
    -------
    PathPayload
        ``{"start_time": <abs Unix s>, "coordsys": coordsys, "points": [...]}``,
        ready to JSON-serialize and POST to Go TCS ``/path``. ``start_time``
        is absolute Unix seconds; ``points`` times are relative seconds.

    Raises
    ------
    ValueError
        If ``coordsys`` is not ``"Horizon"`` or ``"ICRS"``, if
        ``trajectory.start_time`` is not set, or, via :func:`to_path_format`,
        if any consecutive sample interval is below
        ``GO_TCS_MIN_SAMPLE_INTERVAL_SEC`` (50 ms).

    Warns
    -----
    PointingWarning
        If ``coordsys="ICRS"``: Go TCS does not implement ICRS ``/path``
        velocities, so such a body is unsafe for scanning.

    Notes
    -----
    Go TCS also gates ``/path`` (and ``/azimuth-scan`` and ``/track``) on
    a **minimum dispatch lead**: a body whose ``start_time`` is less than
    about 10 seconds in the future is rejected with HTTP 400. This
    function does not pre-check that, because the remaining lead keeps
    shrinking between serialization and POST; the dispatching layer owns
    the floor (the typed scan tasks push a late scheduled start out to now
    plus a 10 s dispatch buffer before planning).
    """
    if coordsys not in ("Horizon", "ICRS"):
        raise ValueError(f"coordsys must be 'Horizon' or 'ICRS', got {coordsys!r}")
    if trajectory.start_time is None:
        raise ValueError("start_time not set; cannot build /path payload")
    if coordsys == "ICRS":
        warnings.warn(
            "coordsys='ICRS' but Go TCS does not implement ICRS /path velocities; "
            "use 'Horizon' for scanning trajectories.",
            PointingWarning,
            stacklevel=2,
        )
    return {
        "start_time": float(trajectory.start_time.unix),
        "coordsys": coordsys,
        "points": to_path_format(trajectory),
    }


#: Number of points at the start of each new constant-velocity science leg
#: that carry ``group_flag = 1`` in :func:`to_trackpoint_format`. Matches the
#: Simons Observatory ACU driver convention of front-loading the ProgramTrack
#: stack at a new leg so the following points are uploaded promptly.
TRACKPOINT_NEW_LEG_GROUP_SIZE: int = 4


def to_trackpoint_format(trajectory: Trajectory) -> list[dict]:
    """Convert a trajectory to ACU ProgramTrack ``TrackPoint`` rows.

    Lowers a :class:`~fyst_trajectories.trajectory.Trajectory` to the
    per-point representation the Vertex ACU's ProgramTrack stack consumes,
    for a consumer that drives the ACU **directly** (e.g. the socs ACU
    agent's ``UploadPtStack`` path) rather than through Go TCS. Use
    :func:`to_path_payload` for the Go TCS ``/path`` route; use this for
    direct-ACU upload.

    Each row is a dict keyed exactly as the socs ACU agent's ``TrackPoint``, so a consumer
    can wrap it directly (``TrackPoint(**row)``): ``timestamp`` (absolute Unix
    seconds), ``az``, ``el`` (deg), ``az_vel``, ``el_vel`` (deg/s), ``az_flag``,
    ``el_flag`` (int), and ``group_flag`` (int).

    The ``SCAN_FLAG_SCIENCE`` /
    ``SCAN_FLAG_*`` values are mapped to the ACU ``az_flag`` convention:

    - interior point of a science sweep leg -> ``az_flag = 1``
    - final point of a science sweep leg -> ``az_flag = 2``
    - turnaround / retune / unclassified / unflagged -> ``az_flag = 0``

    ``group_flag = 1`` is set on the first
    :data:`TRACKPOINT_NEW_LEG_GROUP_SIZE` points of each new science leg
    (front-loading the stack, matching the SO ACU driver). ``el_flag`` is
    always 0: the leg flags emitted here describe the azimuth sweep structure,
    and the socs ACU driver likewise leaves ``el_flag`` 0 on its azimuth scans.
    Batching the rows for upload is left to the consumer.

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory to convert. Must have ``start_time`` set (per-point
        timestamps are absolute Unix seconds derived from it). If
        ``scan_flag`` is ``None`` every point is treated as unclassified
        (``az_flag = 0``).

    Returns
    -------
    list of dict
        One dict per sample, in time order, with the eight ``TrackPoint``
        keys described above.

    Raises
    ------
    ValueError
        If ``trajectory.start_time`` is not set.

    See Also
    --------
    to_path_payload : the Go TCS ``/path`` body (relative point times plus a
        single absolute ``start_time``); use that for the Go-TCS route, this
        for direct-ACU upload.
    """
    if trajectory.start_time is None:
        raise ValueError("start_time not set; cannot build absolute TrackPoint timestamps")

    t0 = float(trajectory.start_time.unix)
    # Relative to ``times[0]`` so the first sample lands exactly on
    # ``start_time``, matching ``get_absolute_times`` and ``to_path_format``.
    times = trajectory.times - trajectory.times[0]
    az = trajectory.az
    el = trajectory.el
    az_vel = trajectory.az_vel
    el_vel = trajectory.el_vel
    scan_flag = trajectory.scan_flag
    n = len(times)

    rows: list[dict] = []
    group_countdown = 0
    for i in range(n):
        sf = int(scan_flag[i]) if scan_flag is not None else SCAN_FLAG_UNCLASSIFIED

        if sf == SCAN_FLAG_SCIENCE:
            is_last_science = i == n - 1 or int(scan_flag[i + 1]) != SCAN_FLAG_SCIENCE
            az_flag = 2 if is_last_science else 1
        else:
            az_flag = 0

        # Front-load the ProgramTrack stack at the start of each science leg.
        if sf == SCAN_FLAG_SCIENCE:
            if i == 0 or int(scan_flag[i - 1]) != SCAN_FLAG_SCIENCE:
                group_countdown = TRACKPOINT_NEW_LEG_GROUP_SIZE
        else:
            # The countdown belongs to the leg that opened it. A science leg
            # shorter than the group size would otherwise carry the flag on
            # past its own end, marking the following turnaround samples as
            # the start of a new leg.
            group_countdown = 0
        group_flag = 1 if group_countdown > 0 else 0
        if group_countdown > 0:
            group_countdown -= 1

        rows.append(
            {
                "timestamp": t0 + float(times[i]),
                "az": float(az[i]),
                "el": float(el[i]),
                "az_vel": float(az_vel[i]),
                "el_vel": float(el_vel[i]),
                "az_flag": az_flag,
                "el_flag": 0,
                "group_flag": group_flag,
            }
        )
    return rows


def _format_trajectory(
    trajectory: Trajectory,
    head: int | None = 5,
    tail: int | None = 5,
) -> str:
    """Format trajectory as a table string."""
    lines = [repr(trajectory), ""]
    n = trajectory.n_points
    head_n = min(head or 0, n)
    tail_n = min(tail or 0, n)

    if (head_n + tail_n) >= n:
        indices: list[int | None] = list(range(n))
    else:
        indices = list(range(head_n))
        if head_n > 0 and tail_n > 0:
            indices.append(None)
        indices.extend(range(n - tail_n, n))

    has_abs = trajectory.start_time is not None
    abs_times = get_absolute_times(trajectory) if has_abs else None

    if has_abs:
        hdr = f"{'t (s)':>8}  {'UTC':^23}  {'az':>10}  {'el':>10}  {'az_vel':>10}  {'el_vel':>10}"
    else:
        hdr = f"{'t (s)':>8}  {'az':>10}  {'el':>10}  {'az_vel':>10}  {'el_vel':>10}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for i in indices:
        if i is None:
            lines.append(
                "..."
                if not has_abs
                else f"{'...':>8}  {'':^23}  {'...':>10}  {'...':>10}  {'...':>10}  {'...':>10}"
            )
        else:
            row = f"{trajectory.times[i]:8.2f}  "
            if has_abs:
                row += f"{abs_times[i].iso[:23]:^23}  "
            row += f"{trajectory.az[i]:10.4f}  {trajectory.el[i]:10.4f}  "
            row += f"{trajectory.az_vel[i]:10.4f}  {trajectory.el_vel[i]:10.4f}"
            lines.append(row)

    return "\n".join(lines)


def print_trajectory(
    trajectory: Trajectory,
    head: int | None = 5,
    tail: int | None = 5,
    file: TextIO | None = None,
) -> None:
    """Print a formatted table of trajectory points.

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory to print.
    head : int or None, optional
        Number of points from the beginning. Default is 5.
    tail : int or None, optional
        Number of points from the end. Default is 5.
    file : TextIO or None, optional
        Output stream. Default is sys.stdout.

    Examples
    --------
    >>> from fyst_trajectories import print_trajectory
    >>> print_trajectory(trajectory)
    Trajectory(n_points=...

    Print only the first 10 points:

    >>> print_trajectory(trajectory, head=10, tail=None)
    Trajectory(n_points=...
    """
    print(_format_trajectory(trajectory, head=head, tail=tail), file=file or sys.stdout)
