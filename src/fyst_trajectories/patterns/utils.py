"""Utility functions for scan patterns.

Shared helper functions used by multiple pattern implementations.
Trajectory validation functions live in :mod:`fyst_trajectories.trajectory_utils`.
"""

import dataclasses
import math
import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta

from ..coordinates import Coordinates
from ..exceptions import (
    PointingWarning,
    TargetNotObservableError,
    TrajectoryBoundsError,
)
from ..site import AtmosphericConditions, Site
from ..trajectory import SCAN_FLAG_SCIENCE, SCAN_FLAG_TURNAROUND, Trajectory, TrajectoryMetadata
from ..trajectory_utils import validate_trajectory_bounds

__all__ = [
    "compute_velocities",
    "normalize_azimuth",
    "rewrap_trajectory_azimuth",
    "sky_offsets_to_altaz",
    "validate_sample_count",
    "wrap_bounds_error",
]

_SCIENCE_SPEED_FRACTION: float = 0.8
"""Fraction of nominal velocity below which samples are flagged as turnaround.

Samples with offset-frame speed below ``_SCIENCE_SPEED_FRACTION * velocity``
are classified as SCAN_FLAG_TURNAROUND.

The 0.8 threshold is empirical and sits above the typical dip in the
*diagonal* speed at the vertices of a Fourier-truncated triangle wave:
at a vertex one axis passes through its turning point with zero
single-axis velocity, so the diagonal speed falls to the other axis's
speed alone, typically near ``1/sqrt(2)`` (about 71%) of the nominal
``velocity``. Increasing ``num_terms`` sharpens the corners, which
generally shortens the flagged turnaround regions and raises the
science fraction, though not strictly monotonically; the threshold
itself does not need retuning with ``num_terms``.

The Daisy flags with the same fraction: its initial ramp-up (set by
``start_acceleration``) falls below it and is classified as non-science.

No published cross-facility standard exists for this exact fraction, but
speed-based turnaround detection is the common practice. SO's sotodlib
offers both an azimuth-percentile and a scan-speed criterion in
``tod_ops.flags.get_turnaround_flags`` and defaults to the scan-speed one;
JCMT's SMURF flags turnaround data by slew speed (the FLAGSLOW cleaning
parameter, in arcsec/s). The criterion here is speed-based too, expressed
as a fraction of the nominal scan velocity and tuned for FYST/Prime-Cam
scan dynamics.
"""


def validate_sample_count(duration: float, timestep: float) -> int:
    """Validate that a duration yields at least two samples and return the count.

    Every pattern samples on the grid ``n_points = round(duration / timestep)
    + 1``. A duration that is zero, negative, or shorter than half a
    ``timestep`` collapses to a single sample, which then either fails
    opaquely in ``np.gradient`` (an unhelpful ``IndexError``) or silently
    produces a one-point trajectory for patterns that set velocities
    directly. This guard rejects those degenerate cases up front with a
    clear error so no pattern emits a sub-two-sample trajectory.

    Parameters
    ----------
    duration : float
        Total scan duration in seconds.
    timestep : float
        Output sampling step in seconds.

    Returns
    -------
    int
        The number of samples (``>= 2``).

    Raises
    ------
    ValueError
        If ``timestep`` is not finite and positive, if ``duration`` is not
        finite, or if the duration yields fewer than two samples.
    """
    if not math.isfinite(timestep) or timestep <= 0:
        raise ValueError(f"timestep must be positive, got {timestep}s")
    if not math.isfinite(duration):
        raise ValueError(f"duration must be finite, got {duration}s")
    n_points = int(round(duration / timestep)) + 1
    if n_points < 2:
        raise ValueError(
            f"duration {duration}s yields fewer than 2 samples at timestep "
            f"{timestep}s; a scan shorter than one sample is degenerate."
        )
    return n_points


@contextmanager
def wrap_bounds_error(target: str, time_info: str) -> Iterator[None]:
    """Convert a bounds error into :class:`~fyst_trajectories.exceptions.TargetNotObservableError`.

    Many pattern generators need to wrap a call to
    :func:`~fyst_trajectories.trajectory_utils.validate_trajectory_bounds`
    with identical boilerplate that re-raises the
    :class:`~fyst_trajectories.exceptions.TrajectoryBoundsError` as a
    target-not-observable error, suppressing the chained traceback.

    Parameters
    ----------
    target : str
        Human-readable target identifier (e.g. ``"RA=180.0 Dec=-30.0"``
        or ``"Jupiter"``).
    time_info : str
        Human-readable time (typically ``start_time.iso``).

    Yields
    ------
    None

    Raises
    ------
    TargetNotObservableError
        If the wrapped block raises :class:`~fyst_trajectories.exceptions.TrajectoryBoundsError`.

    Examples
    --------
    Wrap a pattern's bounds check in its ``generate()`` method::

        with wrap_bounds_error(f"RA={self.ra:.3f} Dec={self.dec:.3f}", start_time.iso):
            validate_trajectory_bounds(site, az, el)
    """
    try:
        yield
    except TrajectoryBoundsError as exc:
        raise TargetNotObservableError(
            target=target,
            time_info=time_info,
            bounds_error=exc,
        ) from None


def normalize_azimuth(
    az: np.ndarray,
    site: Site,
) -> np.ndarray:
    """Normalize azimuth values into the telescope's allowed range.

    Astropy returns azimuth in [0, 360], but telescopes with cable wrap
    typically operate in a range like [-180, 360]. This function unwraps
    the azimuth to remove discontinuities, then shifts by multiples of
    360 degrees to fit within the telescope's azimuth limits.

    If the unwrapped trajectory span exceeds the telescope's azimuth
    range, a :class:`~fyst_trajectories.exceptions.PointingWarning` is
    emitted because no 360-degree shift can make it fit.

    Parameters
    ----------
    az : np.ndarray
        Azimuth positions in degrees (e.g., from astropy [0, 360]).
    site : Site
        Telescope site configuration containing azimuth limits.

    Returns
    -------
    np.ndarray
        Azimuth values shifted into the telescope's range.

    Warns
    -----
    PointingWarning
        If the trajectory's azimuth span exceeds the telescope's
        azimuth range, meaning no shift can make it fit, or if the
        shifted azimuth range still leaves the telescope limits.
    """
    limits = site.telescope_limits

    az_unwrapped = np.unwrap(az, period=360.0)

    az_span = float(az_unwrapped.max() - az_unwrapped.min())
    telescope_range = limits.azimuth.max - limits.azimuth.min
    if az_span > telescope_range:
        warnings.warn(
            f"Trajectory azimuth span ({az_span:.1f} deg) exceeds the "
            f"telescope azimuth range ({telescope_range:.1f} deg = "
            f"[{limits.azimuth.min}, {limits.azimuth.max}]). "
            f"No 360-degree shift can make this trajectory fit within limits. "
            f"validate_trajectory_bounds will report the violation.",
            PointingWarning,
            stacklevel=2,
        )

    az_mid = (az_unwrapped.min() + az_unwrapped.max()) / 2.0
    range_center = (limits.azimuth.min + limits.azimuth.max) / 2.0
    shift = round((range_center - az_mid) / 360.0) * 360.0

    shifted = az_unwrapped + shift

    shifted_min = float(shifted.min())
    shifted_max = float(shifted.max())
    if shifted_min < limits.azimuth.min or shifted_max > limits.azimuth.max:
        warnings.warn(
            f"Shifted azimuth [{shifted_min:.1f}, {shifted_max:.1f}] exceeds "
            f"telescope limits [{limits.azimuth.min}, {limits.azimuth.max}]. "
            f"validate_trajectory_bounds will report the violation.",
            PointingWarning,
            stacklevel=2,
        )

    return shifted


def rewrap_trajectory_azimuth(trajectory: "Trajectory", az_shift: float) -> "Trajectory":
    """Move a whole trajectory onto another azimuth wrap.

    ``normalize_azimuth`` fixes a trajectory's cable-wrap frame when the
    pattern is built, and the encoder choice made just before the slew
    (:func:`~fyst_trajectories.dispatch.choose_encoder_solution`) may land on a
    different one. This applies that decision to the trajectory: the azimuth
    samples move by ``az_shift`` and everything else, velocities included, is
    unchanged, because a rigid shift by whole turns is the same sky path.

    Parameters
    ----------
    trajectory : Trajectory
        The commanded trajectory, in the azimuth frame the encoder choice was
        made against.
    az_shift : float
        Azimuth shift in degrees, a whole multiple of 360. Take it from
        :attr:`~fyst_trajectories.dispatch.EncoderSolution.az_shift`.

    Returns
    -------
    Trajectory
        A copy whose azimuth is ``trajectory.az + az_shift``. The input is
        returned unchanged when the shift is zero.

    Raises
    ------
    ValueError
        If ``az_shift`` is not a whole multiple of 360 degrees. Any other
        shift would move the trajectory to a different patch of sky, which is
        never what a wrap change means.

    Notes
    -----
    Telescope limits are not re-checked here. A shift from
    :func:`~fyst_trajectories.dispatch.choose_encoder_solution` called with
    ``goal_az_span`` set to the trajectory's own azimuth span leaves the path
    in range by construction; for a shift chosen without the span, or taken
    from anywhere else, call
    :func:`~fyst_trajectories.trajectory_utils.validate_trajectory_bounds`.

    Examples
    --------
    >>> import numpy as np
    >>> from fyst_trajectories import rewrap_trajectory_azimuth
    >>> shifted = rewrap_trajectory_azimuth(trajectory, -360.0)
    >>> bool(np.allclose(shifted.az, trajectory.az - 360.0))
    True
    """
    if not math.isfinite(az_shift) or abs(az_shift - round(az_shift / 360.0) * 360.0) > 1e-9:
        raise ValueError(
            f"az_shift must be a whole multiple of 360 degrees, got {az_shift}; "
            "an azimuth wrap change moves the trajectory by whole turns only."
        )
    if az_shift == 0.0:
        return trajectory
    return dataclasses.replace(trajectory, az=trajectory.az + az_shift)


def compute_velocities(
    positions: np.ndarray,
    times: np.ndarray,
    is_angular: bool,
) -> np.ndarray:
    """Compute velocities from positions using numerical differentiation.

    Uses numpy.gradient for numerical differentiation, which handles
    edge points correctly.

    Parameters
    ----------
    positions : np.ndarray
        Position values in degrees (e.g., azimuth or elevation).
        Angular values must be in degrees because the unwrap uses a
        360-degree period.
    times : np.ndarray
        Timestamps in seconds.
    is_angular : bool
        If True, unwrap positions assuming 360-degree periodicity before
        computing gradient. Use for azimuth to handle wrap-around correctly
        (e.g., 359 -> 1 degree transitions).

    Returns
    -------
    np.ndarray
        Velocities computed using numpy.gradient, in units of
        positions per second.

    Examples
    --------
    >>> import numpy as np
    >>> times = np.array([0, 1, 2, 3, 4])
    >>> az = np.array([100, 101, 102, 103, 104])
    >>> compute_velocities(az, times, is_angular=False)
    array([1., 1., 1., 1., 1.])

    Handle azimuth wrap-around:

    >>> times = np.array([0, 1, 2])
    >>> az = np.array([358, 359, 1])  # Wraps from 359 to 1
    >>> compute_velocities(az, times, is_angular=True)  # not -178.5 across the wrap
    array([1. , 1.5, 2. ])
    """
    if is_angular:
        positions = np.unwrap(positions, period=360.0)
    return np.gradient(positions, times)


def sky_offsets_to_altaz(
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
    center_ra: float,
    center_dec: float,
    obstimes: Time,
    coords: "Coordinates",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert sky pattern offsets to Az/El positions.

    Transforms offsets in a sky-aligned coordinate system (where x is
    along RA and y is along Dec) to telescope Az/El coordinates using
    ``SkyCoord.spherical_offsets_by`` for proper great-circle offsets
    on the celestial sphere.

    Parameters
    ----------
    x_offsets : np.ndarray
        X offsets (along RA direction) in degrees.
    y_offsets : np.ndarray
        Y offsets (along Dec direction) in degrees.
    center_ra : float
        Right Ascension of pattern center in degrees.
    center_dec : float
        Declination of pattern center in degrees.
    obstimes : Time
        Observation times for each point.
    coords : Coordinates
        Coordinates converter instance.

    Returns
    -------
    az : np.ndarray
        Azimuth values in degrees.
    el : np.ndarray
        Elevation values in degrees.
    """
    center = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg, frame="icrs")
    offset_coords = center.spherical_offsets_by(x_offsets * u.deg, y_offsets * u.deg)

    az, el = coords.radec_to_altaz(offset_coords.ra.deg, offset_coords.dec.deg, obstimes)

    return az, el


def _flag_by_offset_speed(
    times: np.ndarray,
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
    velocity: float,
    threshold: float,
) -> np.ndarray:
    """Flag samples as science when their offset-frame speed is fast enough.

    A sample is science at or above ``threshold * velocity`` and
    turnaround below it. Measuring speed in the tangent plane keeps the
    criterion independent of elevation and of the ``1 / cos(el)`` azimuth
    stretch, so every offset-driven pattern flags with the same meaning.

    Parameters
    ----------
    times : np.ndarray
        Sample times in seconds (at least two).
    x_offsets, y_offsets : np.ndarray
        Tangent-plane offsets in degrees.
    velocity : float
        Nominal scan speed in degrees/second.
    threshold : float
        Fraction of ``velocity`` at or above which a sample is science.

    Returns
    -------
    np.ndarray
        ``int8`` scan flags.
    """
    x_vel = np.gradient(x_offsets, times)
    y_vel = np.gradient(y_offsets, times)
    speed = np.sqrt(x_vel**2 + y_vel**2)
    scan_flag = np.full(len(times), SCAN_FLAG_TURNAROUND, dtype=np.int8)
    scan_flag[speed >= threshold * velocity] = SCAN_FLAG_SCIENCE
    return scan_flag


def _celestial_offsets_to_trajectory(
    site: Site,
    ra: float,
    dec: float,
    times: np.ndarray,
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
    scan_flag: np.ndarray,
    start_time: Time,
    atmosphere: AtmosphericConditions | None,
    metadata: TrajectoryMetadata,
) -> Trajectory:
    """Map tangent-plane offsets about an RA/Dec center to a trajectory.

    Shared by the celestial Pong and Daisy: each sample's offset is placed
    on the sky about (``ra``, ``dec``), converted to Az/El at its own time,
    normalized onto the telescope's azimuth wrap and bounds-checked.

    Parameters
    ----------
    site : Site
        Telescope site configuration.
    ra, dec : float
        Pattern center in degrees.
    times : np.ndarray
        Sample times in seconds from ``start_time``.
    x_offsets, y_offsets : np.ndarray
        Tangent-plane offsets in degrees.
    scan_flag : np.ndarray
        Per-sample scan flags.
    start_time : Time
        Time of the first sample.
    atmosphere : AtmosphericConditions or None
        Atmospheric conditions for refraction correction; None applies no
        refraction.
    metadata : TrajectoryMetadata
        Metadata to attach to the trajectory.

    Returns
    -------
    Trajectory
        The generated trajectory.

    Raises
    ------
    TargetNotObservableError
        If the path leaves the telescope limits.

    Warns
    -----
    PointingWarning
        If no whole-turn shift places the azimuth track inside the
        telescope's azimuth range.
    """
    coords = Coordinates(site, atmosphere=atmosphere)

    obstimes = start_time + TimeDelta(times * u.s)

    az, el = sky_offsets_to_altaz(
        x_offsets,
        y_offsets,
        ra,
        dec,
        obstimes,
        coords,
    )
    az = normalize_azimuth(az, site)

    az_vel = compute_velocities(az, times, is_angular=True)
    el_vel = compute_velocities(el, times, is_angular=False)

    with wrap_bounds_error(f"RA={ra:.3f} Dec={dec:.3f}", start_time.iso):
        validate_trajectory_bounds(site, az, el)

    return Trajectory(
        times=times,
        az=az,
        el=el,
        az_vel=az_vel,
        el_vel=el_vel,
        start_time=start_time,
        metadata=metadata,
        scan_flag=scan_flag,
    )


def _altaz_offsets_to_trajectory(
    site: Site,
    az_center: float,
    el_center: float,
    times: np.ndarray,
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
    scan_flag: np.ndarray,
    start_time: Time | None,
    metadata: TrajectoryMetadata,
) -> Trajectory:
    """Map tangent-plane offsets about a fixed Az/El center to a trajectory.

    Shared by the AltAz Pong and Daisy: a static horizon-frame projection,
    ``az = x / cos(el_center) + az_center`` and ``el = y + el_center``.

    Parameters
    ----------
    site : Site
        Telescope site configuration.
    az_center, el_center : float
        Pattern center in degrees.
    times : np.ndarray
        Sample times in seconds.
    x_offsets, y_offsets : np.ndarray
        Tangent-plane offsets in degrees.
    scan_flag : np.ndarray
        Per-sample scan flags.
    start_time : Time or None
        Attached to the trajectory when given.
    metadata : TrajectoryMetadata
        Metadata to attach to the trajectory.

    Returns
    -------
    Trajectory
        The generated trajectory.

    Raises
    ------
    TrajectoryBoundsError
        If the trajectory exceeds telescope limits.
    """
    cos_el = math.cos(math.radians(el_center))
    az = x_offsets / cos_el + az_center
    el = y_offsets + el_center

    az_vel = compute_velocities(az, times, is_angular=True)
    el_vel = compute_velocities(el, times, is_angular=False)

    validate_trajectory_bounds(site, az, el)

    return Trajectory(
        times=times,
        az=az,
        el=el,
        az_vel=az_vel,
        el_vel=el_vel,
        start_time=start_time,
        metadata=metadata,
        scan_flag=scan_flag,
    )
