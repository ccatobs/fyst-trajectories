"""Trajectory utility functions.

Free functions for validating, exporting, formatting, and plotting
Trajectory objects. These are the primary API; the
:class:`~fyst_trajectories.trajectory.Trajectory` container itself
exposes no methods that delegate here.
"""

import dataclasses
import math
import sys
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, TextIO

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
from .trajectory import (
    SCAN_FLAG_RETUNE,
    SCAN_FLAG_SCIENCE,
    SCAN_FLAG_TURNAROUND,
    SCAN_FLAG_UNCLASSIFIED,
    RetuneEvent,
    Trajectory,
)

# ``RetuneEvent`` is re-exported here for backward compatibility with
# callers that import it from ``fyst_trajectories.trajectory_utils``. The
# canonical definition now lives in :mod:`fyst_trajectories.trajectory`
# alongside the :data:`SCAN_FLAG_*` constants because the class is a
# structural piece of :class:`Trajectory`.

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    # Annotation-only import to avoid an import cycle: ``dispatch`` imports
    # ``coordinates``/``site``/``exceptions`` at runtime, all of which this
    # module also imports. The predicate is invoked structurally, so only the
    # type hint needs the symbol.
    from .dispatch import SunSafePredicate


# Default wall-clock duration of a single retune event (seconds).
# Used as the default for both the inter-scan retunes scheduled by
# :class:`~fyst_trajectories.overhead.CalibrationPolicy` (via
# :attr:`~fyst_trajectories.overhead.OverheadModel.retune_duration`)
# and the in-scan retunes injected by :func:`inject_retune` for
# per-module staggering. The two code paths are independent (different
# layers, different retune semantics) but share the same nominal
# wall-time; exposing the constant here keeps them in sync if the
# instrument team re-baselines the value.
DEFAULT_RETUNE_DURATION_SEC: float = 5.0


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
    PointingWarning
        If velocity or acceleration exceeds configured limits, or if
        the trajectory has too few points for meaningful validation.
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

    if len(times) < 4:
        warnings.warn(
            f"Trajectory has only {len(times)} points. Acceleration estimates "
            "require at least 4 points; skipping acceleration validation.",
            PointingWarning,
            stacklevel=2,
        )
        return

    limits = site.telescope_limits

    az_unwrapped = np.unwrap(az, period=360.0)
    az_vel = np.gradient(az_unwrapped, times)
    el_vel = np.gradient(el, times)

    max_az_vel = np.abs(az_vel).max()
    max_el_vel = np.abs(el_vel).max()

    if max_az_vel > limits.azimuth.max_velocity:
        warnings.warn(
            f"Trajectory azimuth velocity ({max_az_vel:.2f} deg/s) exceeds "
            f"limit ({limits.azimuth.max_velocity:.2f} deg/s).",
            VelocityLimitWarning,
            stacklevel=2,
        )

    if max_el_vel > limits.elevation.max_velocity:
        warnings.warn(
            f"Trajectory elevation velocity ({max_el_vel:.2f} deg/s) exceeds "
            f"limit ({limits.elevation.max_velocity:.2f} deg/s).",
            VelocityLimitWarning,
            stacklevel=2,
        )

    # Advisory: check if cos(el) scaling makes coordinate velocity misleading.
    # At high elevation, small on-sky motions require large az coordinate rates.
    cos_el = np.cos(np.radians(el))
    min_cos_el = cos_el.min()
    if min_cos_el > 0 and max_az_vel > 0:
        on_sky_az_vel = np.abs(az_vel) * cos_el
        max_on_sky = on_sky_az_vel.max()
        # Warn when coordinate velocity exceeds on-sky by >2x (el > ~60 deg)
        if max_az_vel > 2.0 * max_on_sky:
            warnings.warn(
                f"High elevation reduces on-sky azimuth speed to "
                f"{max_on_sky:.2f} deg/s (coordinate: {max_az_vel:.2f} deg/s, "
                f"min cos(el)={min_cos_el:.3f}). Verify scan design is appropriate.",
                PointingWarning,
                stacklevel=2,
            )
    elif min_cos_el > 0 and min_cos_el < 0.5:
        # Fixed-azimuth (or near-zero az motion) high-elevation trajectories
        # never trip the cos(el) advisory above because ``max_az_vel == 0``
        # short-circuits the on-sky comparison. Surface the on-sky pinch as
        # an independent observation so operators don't miss the high-el
        # implication for a sidereal-track or zero-throw scan.
        warnings.warn(
            f"Trajectory reaches high elevation (min cos(el)={min_cos_el:.3f}, "
            f"el > ~{np.degrees(np.arccos(min_cos_el)):.0f} deg). On-sky azimuth "
            "resolution is reduced even though coordinate-frame az motion is "
            "small or zero.",
            PointingWarning,
            stacklevel=2,
        )

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
        :class:`~fyst_trajectories.dispatch.SunSafePredicate` contract,
        forwarded to :func:`validate_sun_avoidance`. ``None`` (default)
        keeps the built-in scalar exclusion/warning-radius check; an
        injected predicate is consulted per-subsample instead, so the
        directional sun-avoidance model (future shared library) is honored
        end-to-end. Advisory either way. Has no effect when ``check_sun`` is
        ``False`` or ``trajectory.start_time`` is ``None``.

    Raises
    ------
    AzimuthBoundsError
        If azimuth positions are outside telescope movement range.
    ElevationBoundsError
        If elevation positions are outside telescope movement range.

    Warns
    -----
    PointingWarning
        If any velocities or accelerations exceed limits, or if any
        trajectory point is within the sun exclusion or warning radius.
    """
    validate_trajectory_bounds(site, trajectory.az, trajectory.el)
    validate_trajectory_dynamics(site, trajectory.az, trajectory.el, trajectory.times)
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
    relative = trajectory.times
    if len(relative):
        relative = relative - relative[0]
    return trajectory.start_time + TimeDelta(relative * u.s)


def validate_sun_avoidance(
    site: Site,
    az: np.ndarray,
    el: np.ndarray,
    times: Time | np.ndarray,
    coords: Coordinates | None = None,
    sun_safe: "SunSafePredicate | None" = None,
) -> None:
    """Check sun avoidance constraints, emitting warnings for violations.

    .. warning::

       This check is **advisory only**.  It emits Python warnings but
       never blocks trajectory generation or raises exceptions.  Telescope
       control systems **must** enforce their own hard sun-avoidance limits
       independently of this function.

    The sun moves slowly (~0.25 deg/min), so this function subsamples
    the trajectory to check approximately every 60 seconds rather than
    every point.  For each subsampled time the sun position is computed
    once and compared against all trajectory points in that time window.

    This function never blocks trajectory generation. Violations are
    reported as warnings so that downstream consumers can decide how
    to handle them.

    Parameters
    ----------
    site : Site
        Site configuration with sun avoidance settings.
    az : np.ndarray
        Azimuth array in degrees.
    el : np.ndarray
        Elevation array in degrees.
    times : Time or np.ndarray
        Absolute times for each trajectory point.
    coords : Coordinates, optional
        Pre-constructed Coordinates instance. Created internally if
        not provided.
    sun_safe : SunSafePredicate, optional
        Sun-safety predicate implementing the
        :class:`~fyst_trajectories.dispatch.SunSafePredicate` contract,
        ``(az_deg, el_deg, time) -> bool`` returning ``True`` when the
        position is clear of the Sun. ``None`` (default) keeps the built-in
        scalar exclusion/warning-radius check (the existing vectorised
        subsampled separation computation, unchanged). When a predicate is
        injected it is consulted per-subsample ``(az_i, el_i, time_i)``
        instead, so the directional sun-avoidance model (future shared
        library) is honored, using the same ~60 s subsampling for parity
        and performance. The injected model owns its own boundary, so only
        an "EXCLUSION ZONE" warning is emitted (no separate warning-radius
        band). Advisory either way. See
        :class:`~fyst_trajectories.dispatch.SunSafePredicate`.

    Warns
    -----
    PointingWarning
        If any trajectory point is within the exclusion radius
        ("EXCLUSION ZONE") or the warning radius ("WARNING ZONE")
        of the Sun. With an injected ``sun_safe`` predicate, an
        "EXCLUSION ZONE" warning is emitted for any subsample the
        predicate reports unsafe.
    """
    if not site.sun_avoidance.enabled:
        return

    if coords is None:
        coords = Coordinates(site)

    n_points = len(az)
    if n_points == 0:
        return

    if isinstance(times, Time):
        total_seconds = (times[-1] - times[0]).to_value(u.s)
    else:
        total_seconds = float(times[-1] - times[0])

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
        # Injected directional model: consult it per-subsample. ``False``
        # marks an unsafe (inside-the-zone) sample. Warn once, naming the
        # first unsafe subsample, mirroring the scalar branch's single
        # closest-approach warning. The ~60 s subsampling above is preserved
        # so injection does not regress performance.
        unsafe_idx = [
            i
            for i in range(len(sample_az))
            if not sun_safe(float(sample_az[i]), float(sample_el[i]), sample_times[i])
        ]
        if not unsafe_idx:
            return
        first = unsafe_idx[0]
        if isinstance(sample_times, Time):
            first_time_str = sample_times[first].iso
        else:
            first_time_str = str(sample_times[first])
        warnings.warn(
            f"EXCLUSION ZONE: Trajectory at (az={float(sample_az[first]):.1f}°, "
            f"el={float(sample_el[first]):.1f}°) is inside the Sun avoidance "
            f"zone at {first_time_str}. The telescope hardware may refuse this "
            f"trajectory.",
            PointingWarning,
            stacklevel=2,
        )
        return

    sun_az, sun_alt = coords.get_sun_altaz(sample_times)
    sun_az = np.atleast_1d(sun_az)
    sun_alt = np.atleast_1d(sun_alt)

    az1 = np.deg2rad(sample_az)
    el1 = np.deg2rad(sample_el)
    az2 = np.deg2rad(sun_az)
    el2 = np.deg2rad(sun_alt)

    daz = az1 - az2
    cos_el2 = np.cos(el2)
    sin_el2 = np.sin(el2)
    cos_el1 = np.cos(el1)
    sin_el1 = np.sin(el1)

    num = np.sqrt(
        (cos_el2 * np.sin(daz)) ** 2 + (cos_el1 * sin_el2 - sin_el1 * cos_el2 * np.cos(daz)) ** 2
    )
    den = sin_el1 * sin_el2 + cos_el1 * cos_el2 * np.cos(daz)
    separations = np.rad2deg(np.arctan2(num, den))

    min_idx = int(np.argmin(separations))
    min_sep = float(separations[min_idx])

    exclusion = site.sun_avoidance.exclusion_radius
    warning = site.sun_avoidance.warning_radius

    if isinstance(sample_times, Time):
        closest_time_str = sample_times[min_idx].iso
    else:
        closest_time_str = str(sample_times[min_idx])

    # Use ``<=`` so a separation exactly at the exclusion radius counts as
    # unsafe, matching ``Coordinates.is_sun_safe`` / ``is_position_observable``
    # and the planning sun checks (the conservative ``sep <= radius`` convention).
    if min_sep <= exclusion:
        warnings.warn(
            f"EXCLUSION ZONE: Trajectory passes {min_sep:.1f}\u00b0 from the Sun "
            f"(exclusion radius: {exclusion}\u00b0) at {closest_time_str}. "
            f"The telescope hardware may refuse this trajectory.",
            PointingWarning,
            stacklevel=2,
        )
    elif min_sep < warning:
        warnings.warn(
            f"WARNING ZONE: Trajectory passes {min_sep:.1f}\u00b0 from the Sun "
            f"(warning radius: {warning}\u00b0) at {closest_time_str}.",
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
        Timestamps in seconds from start.
    az : np.ndarray
        Azimuth positions in degrees.
    el : np.ndarray
        Elevation positions in degrees.
    """
    return trajectory.times.copy(), trajectory.az.copy(), trajectory.el.copy()


#: Minimum spacing between consecutive ``/path`` samples accepted by Go TCS.
#: ``pathCmd.Check()`` hard-rejects any pair closer than 50 ms (ACU ICD 2.0
#: §8.9.3; ``telescope-control-system/commands.go:248-254``).
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
        receiver hard-rejects such a body with HTTP 400 (ACU ICD 2.0 §8.9.3),
        so the check is enforced here at the serialization boundary rather
        than failing at POST time.

    Notes
    -----
    The Go TCS ``/path`` receiver requires a three-key JSON body and sets
    ``DisallowUnknownFields``, so the body must be exactly
    ``{"start_time": <abs Unix s>, "coordsys": "Horizon", "points": [...]}``:

    - ``coordsys`` is **required** and must be ``"Horizon"``; these are
      nominal (vacuum) az/el rows per P-INCM-ICD-0003-A Eq.(1), carrying no
      refraction, SPEM, or non-repeatable pointing terms (those are applied
      downstream at OCS and/or the ACU), and ICRS ``/path`` velocities are
      unimplemented in Go TCS. Omitting ``coordsys`` or adding any extra key
      yields HTTP 400.
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
    if times.size >= 2:
        min_dt = float(np.diff(times).min())
        if min_dt < GO_TCS_MIN_SAMPLE_INTERVAL_SEC:
            raise ValueError(
                f"Trajectory sample interval {min_dt:.4g} s is below the Go TCS "
                f"/path minimum of {GO_TCS_MIN_SAMPLE_INTERVAL_SEC} s (ACU ICD 2.0 "
                "§8.9.3); the upload would be rejected with HTTP 400. Increase the "
                "pattern timestep."
            )
    return np.column_stack(
        [
            trajectory.times,
            trajectory.az,
            trajectory.el,
            trajectory.az_vel,
            trajectory.el_vel,
        ]
    ).tolist()


def to_path_payload(trajectory: Trajectory, coordsys: str = "Horizon") -> dict:
    """Assemble the full Go TCS ``/path`` request body for a trajectory.

    Wraps :func:`to_path_format` with the two scalar keys the Go TCS
    ``/path`` endpoint also requires, returning the exact three-key dict
    ``{"start_time", "coordsys", "points"}``. Prefer this over assembling the
    body by hand around :func:`to_path_format`: the Go TCS receiver sets
    ``DisallowUnknownFields`` and switches on a required ``coordsys``, so a
    body that omits ``coordsys`` (or adds an extra key) is rejected with
    HTTP 400.

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory to serialize. Must have ``start_time`` set.
    coordsys : str, optional
        Coordinate system for the points. Must be ``"Horizon"`` (default)
        or ``"ICRS"``. Use ``"Horizon"`` for trajectory upload: the rows are
        nominal (vacuum) az/el per P-INCM-ICD-0003-A Eq.(1), carrying no
        refraction, SPEM, or non-repeatable pointing terms (those are applied
        downstream at OCS and/or the ACU), and ICRS ``/path`` velocities are
        unimplemented in Go TCS.

    Returns
    -------
    dict
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
    the same rows the Go TCS ``/path`` endpoint builds internally, but ready
    for a consumer that drives the ACU **directly** (e.g. a PCS agent's
    ``UploadPtStack`` path) rather than through Go TCS. Use
    :func:`to_path_payload` for the Go TCS ``/path`` route; use this for
    direct-ACU upload.

    Each row is a dict keyed exactly as a PCS ``TrackPoint``, so a consumer
    can wrap it directly (``TrackPoint(**row)``): ``timestamp`` (absolute Unix
    seconds), ``az``, ``el`` (deg), ``az_vel``, ``el_vel`` (deg/s), ``az_flag``,
    ``el_flag`` (int), and ``group_flag`` (int).

    The :data:`~fyst_trajectories.trajectory.SCAN_FLAG_SCIENCE` /
    ``SCAN_FLAG_*`` values are mapped to the ACU ``az_flag`` convention:

    - interior point of a science sweep leg -> ``az_flag = 1``
    - final point of a science sweep leg -> ``az_flag = 2``
    - turnaround / retune / unclassified / unflagged -> ``az_flag = 0``

    ``group_flag = 1`` is set on the first
    :data:`TRACKPOINT_NEW_LEG_GROUP_SIZE` points of each new science leg
    (front-loading the stack, matching the SO ACU driver). ``el_flag`` is
    always 0; FYST scans hold or el-nod within a leg, so the ACU profiler
    keys off azimuth. Batching the rows for upload is left to the consumer.

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
    times = trajectory.times
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
        if sf == SCAN_FLAG_SCIENCE and (i == 0 or int(scan_flag[i - 1]) != SCAN_FLAG_SCIENCE):
            group_countdown = TRACKPOINT_NEW_LEG_GROUP_SIZE
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


def plot_trajectory(trajectory: Trajectory, show: bool) -> "Figure":
    """Plot trajectory az/el vs time and sky track.

    Creates a 3-panel figure showing azimuth vs time, elevation vs time,
    and azimuth vs elevation (sky track).

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory to plot.
    show : bool
        Whether to call plt.show() after creating the figure.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_trajectory(). "
            "Install it with: pip install fyst-trajectories[plotting]"
        ) from None

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(trajectory.times, trajectory.az)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Azimuth (deg)")
    axes[0].set_title("Az vs Time")

    axes[1].plot(trajectory.times, trajectory.el)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Elevation (deg)")
    axes[1].set_title("El vs Time")

    axes[2].plot(trajectory.az, trajectory.el)
    axes[2].set_xlabel("Azimuth (deg)")
    axes[2].set_ylabel("Elevation (deg)")
    axes[2].set_title("Sky Track")
    axes[2].set_aspect("equal")

    fig.tight_layout()

    if show:
        plt.show()

    return fig


# Tolerance for treating consecutive events as non-overlapping. A gap
# smaller than this (in seconds) between the end of one event and the
# start of the next is treated as a floating-point artefact and allowed.
# A positive epsilon makes the overlap check stricter: two events are
# accepted as non-overlapping only when the gap between them exceeds
# ``EPS`` seconds beyond the event boundary (a gap smaller than EPS is
# treated as a touching/adjacent boundary, not as overlap).
_EVENT_OVERLAP_EPS: float = 1e-9


def _zero_velocity_guard(trajectory: Trajectory, prefer_turnarounds: bool) -> bool:
    """Return possibly-adjusted ``prefer_turnarounds`` flag.

    Shared guard used by both the uniform-cadence and event-list code
    paths. If ``prefer_turnarounds`` is True but the trajectory has
    identically zero az/el velocities, warn and fall back to time-based
    placement (``prefer_turnarounds=False``). Otherwise returns the flag
    unchanged.
    """
    if not prefer_turnarounds:
        return prefer_turnarounds
    if (
        trajectory.az_vel is not None
        and trajectory.el_vel is not None
        and np.all(trajectory.az_vel == 0.0)
        and np.all(trajectory.el_vel == 0.0)
    ):
        warnings.warn(
            "inject_retune called with prefer_turnarounds=True but the "
            "trajectory has all-zero velocities; turnaround detection "
            "requires real velocities. Falling back to time-based retune "
            "placement.",
            PointingWarning,
            stacklevel=3,
        )
        return False
    return prefer_turnarounds


def _collect_turnaround_starts(scan_flag: np.ndarray, times: np.ndarray) -> list[float]:
    """Return the start times of each contiguous turnaround region.

    A turnaround start is the first sample in a run of consecutive
    ``SCAN_FLAG_TURNAROUND`` samples. Returned times are trajectory-
    absolute (i.e. they come directly from ``times``, including any
    ``times[0]`` offset).
    """
    turnaround_starts: list[float] = []
    is_turnaround = scan_flag == SCAN_FLAG_TURNAROUND
    for i in range(len(is_turnaround)):
        if is_turnaround[i] and (i == 0 or not is_turnaround[i - 1]):
            turnaround_starts.append(float(times[i]))
    return turnaround_starts


def _snap_to_turnaround(
    due_time: float,
    turnaround_starts: Sequence[float],
    turnaround_window: float,
) -> float:
    """Return the nearest turnaround start within ``turnaround_window``.

    If no turnaround sits within the window, returns ``due_time``
    unchanged.
    """
    best_ta = None
    best_dist = turnaround_window + 1.0
    for ta_start in turnaround_starts:
        dist = abs(ta_start - due_time)
        if dist <= turnaround_window and dist < best_dist:
            best_ta = ta_start
            best_dist = dist
    if best_ta is not None:
        return best_ta
    return due_time


def _inject_retune_uniform(
    trajectory: Trajectory,
    retune_interval: float,
    retune_duration: float,
    prefer_turnarounds: bool,
    turnaround_window: float,
    module_index: int,
    n_modules: int,
) -> Trajectory:
    """Uniform-cadence retune injection (extracted helper).

    Flag-array generation is byte-identical to the pre-refactor
    ``inject_retune`` body: same variable names, same comments, same
    warning semantics. Callers are responsible for having already
    validated scalar inputs and run the zero-velocity guard.

    As a side-effect the helper also records every retune it scheduled as
    a :class:`~fyst_trajectories.trajectory.RetuneEvent` on the returned
    trajectory's ``retune_events`` field, so introspection and ECSV
    round-trip work uniformly regardless of which code path populated
    ``scan_flag``.
    """
    times = trajectory.times

    if trajectory.scan_flag is None:
        scan_flag = np.full(len(times), SCAN_FLAG_SCIENCE, dtype=np.int8)
    else:
        scan_flag = trajectory.scan_flag.copy()

    duration = float(times[-1] - times[0])
    if duration < retune_interval:
        return dataclasses.replace(trajectory, scan_flag=scan_flag, retune_events=())

    turnaround_starts: list[float] = []
    if prefer_turnarounds:
        turnaround_starts = _collect_turnaround_starts(scan_flag, times)

    # For staggered retune, offset the first retune time by a fraction
    # of the retune interval so different modules retune at different times.
    # ``next_due_anchor`` is the synthetic anchor whose increments by
    # ``retune_interval`` produce the next ``due_time``; it is *not* the
    # wall-clock time of the most recent retune (those two coincide only
    # when ``prefer_turnarounds`` does not snap).
    stagger_offset = module_index * retune_interval / n_modules
    next_due_anchor = float(times[0]) + stagger_offset

    generated_events: list[RetuneEvent] = []
    t0 = float(times[0])

    while True:
        due_time = next_due_anchor + retune_interval
        if due_time > float(times[-1]):
            break

        retune_start = due_time
        if prefer_turnarounds and turnaround_starts:
            retune_start = _snap_to_turnaround(due_time, turnaround_starts, turnaround_window)

        retune_end = retune_start + retune_duration

        mask = (times >= retune_start) & (times < retune_end) & (scan_flag == SCAN_FLAG_SCIENCE)
        scan_flag[mask] = SCAN_FLAG_RETUNE

        # Record the event using trajectory-relative t_start (subtracting
        # ``times[0]``) so the provenance matches the event-list path's
        # convention. ``retune_duration`` is used verbatim — any clipping
        # at ``times[-1]`` is an application detail that the scan_flag
        # array already captures.
        generated_events.append(RetuneEvent(t_start=retune_start - t0, duration=retune_duration))

        # Use max(retune_end, due_time) to prevent backward drift when
        # prefer_turnarounds snaps to a turnaround before the due time.
        next_due_anchor = max(retune_end, due_time)

    return dataclasses.replace(
        trajectory, scan_flag=scan_flag, retune_events=tuple(generated_events)
    )


def _inject_retune_events(
    trajectory: Trajectory,
    events: Sequence[RetuneEvent],
    prefer_turnarounds: bool,
    turnaround_window: float,
) -> Trajectory:
    """Event-list retune injection.

    Validates, sorts, clips, and applies a caller-supplied list of
    :class:`RetuneEvent` instances, setting both ``scan_flag`` (the
    per-sample array) and ``retune_events`` (the event-level provenance)
    on the returned trajectory. ``trajectory.metadata`` is left verbatim:
    pattern metadata and retune provenance are distinct concerns and
    live at different fields on :class:`Trajectory`.

    See :func:`inject_retune` for the user-facing contract.
    """
    sorted_events = tuple(sorted(events, key=lambda e: e.t_start))

    # Overlap check.
    for i in range(1, len(sorted_events)):
        a = sorted_events[i - 1]
        b = sorted_events[i]
        if a.t_start + a.duration > b.t_start + _EVENT_OVERLAP_EPS:
            raise ValueError(
                f"Overlapping retune events at sorted indices {i - 1} "
                f"(t_start={a.t_start}, duration={a.duration}) and {i} "
                f"(t_start={b.t_start}). Events must not overlap when "
                "passed to a single inject_retune call; call inject_retune "
                "once per module with its own event list to stagger."
            )

    times = trajectory.times
    t0 = float(times[0])
    t_end = float(times[-1])

    if trajectory.scan_flag is None:
        scan_flag = np.full(len(times), SCAN_FLAG_SCIENCE, dtype=np.int8)
    else:
        scan_flag = trajectory.scan_flag.copy()

    # Partition into in-bounds / out-of-bounds and warn once. Indices are
    # measured in the *sorted* list, not the caller's input order — we
    # surface that in the warning message so callers who pass an unsorted
    # list can still locate the offending events.
    in_bounds: list[RetuneEvent] = []
    skipped: list[tuple[int, float]] = []
    for idx, event in enumerate(sorted_events):
        # Events are trajectory-relative: add t0 to compare against the
        # raw ``times`` domain.
        if event.t_start + t0 >= t_end:
            skipped.append((idx, event.t_start))
        else:
            in_bounds.append(event)

    if skipped:
        skipped_str = ", ".join(f"sorted_index={idx} at t_start={t:.1f}" for idx, t in skipped)
        warnings.warn(
            "inject_retune: skipping retune events past trajectory end: "
            f"{skipped_str}. Indices refer to the sorted event list, not "
            f"the caller-supplied input order. Trajectory spans "
            f"[0, {t_end - t0}] seconds in trajectory-relative time.",
            PointingWarning,
            stacklevel=3,
        )

    turnaround_starts: list[float] = []
    if prefer_turnarounds:
        turnaround_starts = _collect_turnaround_starts(scan_flag, times)

    for event in in_bounds:
        # Trajectory-relative -> absolute-in-times-domain offset. The
        # uniform path works in the raw ``times`` domain (it seeds
        # ``next_due_anchor`` with ``float(times[0])``), so we match
        # that convention here by adding ``t0`` to the caller-supplied
        # relative time.
        start = event.t_start + t0
        if prefer_turnarounds and turnaround_starts:
            start = _snap_to_turnaround(start, turnaround_starts, turnaround_window)
        end = min(start + event.duration, t_end)
        # When the event is clipped to the trajectory end, include the final
        # sample (``times < end`` would drop the science sample at exactly
        # ``t_end``); otherwise keep the half-open upper bound.
        upper = (times <= end) if end >= t_end else (times < end)
        mask = (times >= start) & upper & (scan_flag == SCAN_FLAG_SCIENCE)
        scan_flag[mask] = SCAN_FLAG_RETUNE

    return dataclasses.replace(trajectory, scan_flag=scan_flag, retune_events=sorted_events)


def inject_retune(
    trajectory: Trajectory,
    retune_interval: float = 300.0,
    retune_duration: float = DEFAULT_RETUNE_DURATION_SEC,
    prefer_turnarounds: bool = False,
    turnaround_window: float = 5.0,
    module_index: int = 0,
    n_modules: int = 1,
    *,
    retune_events: Sequence[RetuneEvent] | None = None,
) -> Trajectory:
    """Inject retune flags into a trajectory.

    Two modes are supported. In **uniform-cadence mode** (the default,
    when ``retune_events`` is ``None``), retunes are scheduled every
    ``retune_interval`` seconds; optional per-module staggering is
    controlled by ``module_index`` and ``n_modules``. In **event-list
    mode** (when ``retune_events`` is supplied), the caller provides an
    explicit sequence of :class:`RetuneEvent` instances; the uniform-
    cadence / per-module-stagger kwargs are not used.

    The uniform-cadence behaviour is unchanged from earlier releases.
    Walks forward through the trajectory timeline and places retune
    events every ``retune_interval`` seconds. If ``prefer_turnarounds``
    is True and a turnaround region exists within ``turnaround_window``
    seconds of the due time, the retune is snapped to start at the
    turnaround (zero additional dead time). Otherwise the retune is
    placed at the time-based position.

    The default is ``prefer_turnarounds=False`` (time-based placement),
    which produces uniform coverage. Set to True to snap retunes to nearby
    turnarounds, which saves ~0.04% science time but concentrates gaps at
    turnaround positions, creating persistent coverage non-uniformity.

    Only samples with ``SCAN_FLAG_SCIENCE`` are overwritten with
    ``SCAN_FLAG_RETUNE``; turnaround flags are never modified.

    **Per-module staggered retune** (UNCONFIRMED, needs FYST team
    verification): Prime-Cam has 7 independent readout modules. If modules
    can retune independently, setting ``n_modules > 1`` offsets the first
    retune by ``module_index * retune_interval / n_modules``, so only one
    module is retuning at any given time. This reduces effective overhead
    from ~16% to ~2.4% for 7 modules. Set ``n_modules=1`` (the default)
    to disable staggering and retune all modules simultaneously. Per-
    module staggering in **event-list mode** is handled by composition:
    call ``inject_retune`` once per module with its own event list.

    .. note::

       ``retune_interval``, ``retune_duration``, and ``n_modules`` are
       instrument-team inputs, not astronomer-tunable knobs. Default
       values are commissioning-era placeholders; obtain actual values
       from the Prime-Cam instrument team.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory with scan_flag array.
    retune_interval : float
        Seconds between retune events (from last retune or start).
        Default 300 s (5 min). Ignored when ``retune_events`` is
        supplied.
    retune_duration : float
        Duration in seconds of each retune event. Ignored when
        ``retune_events`` is supplied.
    prefer_turnarounds : bool
        If True, snap retunes to nearby turnarounds when possible.
        Default is False (time-based placement for uniform coverage).
        Applies to both modes.
    turnaround_window : float
        Maximum seconds from due time to search for a turnaround start.
        Applies to both modes.
    module_index : int
        Index of this module (0-based) for staggered retune scheduling.
        Default is 0. Only meaningful when ``n_modules > 1``. Must be
        0 in event-list mode (the caller handles per-module staggering
        by composition).
    n_modules : int
        Total number of independent modules. Default is 1 (no staggering,
        all modules retune simultaneously, current behavior). Set to 7
        for Prime-Cam staggered retune. Must be 1 in event-list mode.
    retune_events : sequence of RetuneEvent, optional, keyword-only
        If supplied, enables event-list mode. Each event's ``t_start``
        is measured in seconds from the trajectory start
        (``trajectory.times[0]``). Events are validated, sorted, and
        applied in chronological order. The validated, sorted tuple is
        set on the returned trajectory's
        :attr:`~fyst_trajectories.trajectory.Trajectory.retune_events`
        field. The uniform-cadence path also populates this field with
        the events it generated, so introspection and ECSV round-trip
        work uniformly regardless of which mode produced the retunes.

    Returns
    -------
    Trajectory
        New trajectory with retune samples flagged.

    Raises
    ------
    ValueError
        If ``module_index`` is negative or >= ``n_modules``, or if
        ``n_modules`` is less than 1 (uniform-cadence mode). If
        ``retune_events`` is supplied with ``module_index != 0`` or
        ``n_modules != 1`` (per-module composition is the caller's
        responsibility in event-list mode). If events overlap.

    Warns
    -----
    PointingWarning
        If ``retune_events`` is supplied together with a non-default
        ``retune_interval`` or ``retune_duration`` (the scalar kwarg is
        ignored in event-list mode). If any event has ``t_start`` past
        the trajectory end (those events are dropped with a single
        summary warning naming the affected sorted indices). If
        ``prefer_turnarounds=True`` but the trajectory has identically
        zero velocities (falls back to time-based placement).

    Examples
    --------
    Uniform cadence, matching existing behaviour::

        result = inject_retune(traj, retune_interval=300.0, retune_duration=5.0)

    Explicit event list (Monte Carlo, log replay, etc.)::

        from fyst_trajectories import RetuneEvent

        events = [
            RetuneEvent(t_start=30.0, duration=5.0),
            RetuneEvent(t_start=200.0, duration=8.0),
        ]
        result = inject_retune(traj, retune_events=events)
        assert result.retune_events == tuple(events)
    """
    if retune_events is not None:
        # Mutual-exclusion: per-module staggering is not a scalar concept
        # in event-list mode; callers compose by calling once per module.
        if module_index != 0 or n_modules != 1:
            raise ValueError(
                f"inject_retune: retune_events is mutually exclusive with "
                f"per-module staggering (got module_index={module_index}, "
                f"n_modules={n_modules}). Call inject_retune once per "
                "module with its own event list to stagger."
            )
        if retune_interval != 300.0:
            warnings.warn(
                "inject_retune: retune_interval is ignored when "
                f"retune_events is supplied (got retune_interval={retune_interval}).",
                PointingWarning,
                stacklevel=2,
            )
        if retune_duration != DEFAULT_RETUNE_DURATION_SEC:
            warnings.warn(
                "inject_retune: retune_duration is ignored when "
                f"retune_events is supplied (got retune_duration={retune_duration}).",
                PointingWarning,
                stacklevel=2,
            )

        prefer_turnarounds = _zero_velocity_guard(trajectory, prefer_turnarounds)
        return _inject_retune_events(
            trajectory,
            retune_events,
            prefer_turnarounds=prefer_turnarounds,
            turnaround_window=turnaround_window,
        )

    if retune_interval <= 0:
        raise ValueError(f"retune_interval must be positive, got {retune_interval}")
    if retune_duration <= 0:
        raise ValueError(f"retune_duration must be positive, got {retune_duration}")
    if n_modules < 1:
        raise ValueError(f"n_modules must be >= 1, got {n_modules}")
    if module_index < 0 or module_index >= n_modules:
        raise ValueError(f"module_index must be in [0, {n_modules}), got {module_index}")

    # N-6 defensive guard: turnaround detection relies on real velocities
    # (see inject_retune body below, which classifies turnarounds via
    # ``SCAN_FLAG_TURNAROUND`` samples derived from the trajectory's
    # velocity profile). A trajectory with identically zero az/el
    # velocities, which the primecam wrapper currently supplies, has no
    # detectable turnarounds, so snapping would silently collapse to
    # time-based placement anyway. Warn and fall back explicitly so the
    # caller is not misled.
    prefer_turnarounds = _zero_velocity_guard(trajectory, prefer_turnarounds)

    return _inject_retune_uniform(
        trajectory,
        retune_interval=retune_interval,
        retune_duration=retune_duration,
        prefer_turnarounds=prefer_turnarounds,
        turnaround_window=turnaround_window,
        module_index=module_index,
        n_modules=n_modules,
    )


def sample_retune_events(
    duration: float,
    *,
    interval_sampler: Callable[[np.random.Generator], float],
    duration_sampler: Callable[[np.random.Generator], float],
    rng: np.random.Generator,
    t_start: float = 0.0,
) -> list[RetuneEvent]:
    """Draw a retune event list from caller-supplied samplers.

    No canonical distribution is baked in because no public KID-camera
    retune log has been published.

    Walks forward from ``t_start``, alternating draws from
    ``interval_sampler`` (gap until the next retune) and
    ``duration_sampler`` (duration of that retune). Stops when the next
    drawn interval would push ``t_start`` past ``duration``; the
    partially-drawn event is discarded, not truncated, so every
    returned event has exactly the duration the sampler produced and
    the returned list is guaranteed non-overlapping.

    Parameters
    ----------
    duration : float
        Trajectory window to fill, in seconds. Must be finite and
        non-negative.
    interval_sampler : callable
        ``(rng) -> float``: draws the gap between consecutive retunes
        (or between ``t_start`` and the first retune). Must return a
        positive, finite value; negative or non-finite draws raise
        :class:`ValueError`.
    duration_sampler : callable
        ``(rng) -> float``: draws the duration of the next retune.
        Must return a positive, finite value.
    rng : np.random.Generator
        Seeded generator for reproducibility. Caller owns seed policy.
    t_start : float
        Starting offset in seconds from the trajectory origin. Default 0.

    Returns
    -------
    list of RetuneEvent
        Events in chronological order, guaranteed non-overlapping.

    Raises
    ------
    ValueError
        If ``duration`` is negative or non-finite. If ``t_start`` is
        negative or non-finite. If either sampler returns a
        non-positive or non-finite value.

    Notes
    -----
    The walk consumes one extra ``interval_sampler`` draw past the last
    emitted event to evaluate the termination condition — callers who
    seed for an exact draw count should account for this.

    Examples
    --------
    >>> import numpy as np
    >>> from fyst_trajectories import sample_retune_events
    >>> rng = np.random.default_rng(seed=42)
    >>> events = sample_retune_events(
    ...     duration=600.0,
    ...     interval_sampler=lambda r: r.uniform(60.0, 120.0),
    ...     duration_sampler=lambda r: r.uniform(3.0, 8.0),
    ...     rng=rng,
    ... )
    """
    if not math.isfinite(duration) or duration < 0:
        raise ValueError(f"duration must be finite and non-negative, got {duration}")
    if not math.isfinite(t_start) or t_start < 0:
        raise ValueError(f"t_start must be finite and non-negative, got {t_start}")

    events: list[RetuneEvent] = []
    current = t_start
    while True:
        gap = interval_sampler(rng)
        if not math.isfinite(gap) or gap <= 0:
            raise ValueError(f"interval_sampler returned non-positive or non-finite value: {gap}")
        event_start = current + gap
        if event_start >= duration:
            break
        dur = duration_sampler(rng)
        if not math.isfinite(dur) or dur <= 0:
            raise ValueError(f"duration_sampler returned non-positive or non-finite value: {dur}")
        events.append(RetuneEvent(t_start=event_start, duration=dur))
        current = event_start + dur
    return events


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

    Print only the first 10 points:

    >>> print_trajectory(trajectory, head=10, tail=None)
    """
    print(_format_trajectory(trajectory, head=head, tail=tail), file=file or sys.stdout)
