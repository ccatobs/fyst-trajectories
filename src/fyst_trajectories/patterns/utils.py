"""Utility functions for scan patterns.

Shared helper functions used by multiple pattern implementations.
Trajectory validation functions live in :mod:`fyst_trajectories.trajectory_utils`.
"""

import warnings
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from ..exceptions import PointingWarning
from ..site import Site

if TYPE_CHECKING:
    from ..coordinates import Coordinates

__all__ = [
    "compute_velocities",
    "generate_time_array",
    "normalize_azimuth",
    "sky_offsets_to_altaz",
]


def generate_time_array(duration: float, timestep: float) -> np.ndarray:
    """Generate evenly-spaced time array with exact endpoint.

    Parameters
    ----------
    duration : float
        Total duration in seconds.
    timestep : float
        Time step in seconds.

    Returns
    -------
    np.ndarray
        Time array from 0 to duration (inclusive), with at least 2 points.
    """
    n_points = max(2, int(round(duration / timestep)) + 1)
    return np.linspace(0, duration, n_points)


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
        azimuth range, meaning no shift can make it fit.
    """
    limits = site.telescope_limits

    # Unwrap to remove 360-degree discontinuities, preserving continuity
    az_unwrapped = np.unwrap(az, period=360.0)

    # Check if trajectory span exceeds the telescope range
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

    # Find the shift (multiple of 360) that places the trajectory midpoint
    # closest to the center of the allowed range
    az_mid = (az_unwrapped.min() + az_unwrapped.max()) / 2.0
    range_center = (limits.azimuth.min + limits.azimuth.max) / 2.0
    shift = round((range_center - az_mid) / 360.0) * 360.0

    return az_unwrapped + shift


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
    >>> times = np.array([0, 1, 2, 3, 4])
    >>> az = np.array([100, 101, 102, 103, 104])
    >>> az_vel = compute_velocities(az, times, is_angular=False)
    >>> # Returns array of 1.0 (constant velocity of 1 deg/s)

    Handle azimuth wrap-around:

    >>> times = np.array([0, 1, 2])
    >>> az = np.array([358, 359, 1])  # Wraps from 359 to 1
    >>> az_vel = compute_velocities(az, times, is_angular=True)
    >>> # Returns velocities near 1.5 deg/s (correct), not -178.5 deg/s (wrong)
    """
    if is_angular:
        # Unwrap to handle 359->1 degree transitions correctly
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
