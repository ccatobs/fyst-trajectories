"""Scheduling utility functions.

Provides slew time estimation, observable window computation, transit time
calculation, and maximum elevation lookup, small utilities that fill gaps
in the fyst-trajectories API needed by the scheduler.
"""

import math
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time, TimeDelta

from ..coordinates import Coordinates
from ..site import Site

if TYPE_CHECKING:
    from ..dispatch import SunSafePredicate

__all__ = [
    "circular_mean_deg",
    "compute_nasmyth_rotation",
    "estimate_slew_time",
    "get_max_elevation",
    "get_observable_windows",
    "get_transit_time",
]


def circular_mean_deg(a: float, b: float) -> float:
    """Circular mean of two azimuths in degrees.

    Averages two angles on the circle via ``atan2`` of the mean sine and
    cosine, so a north-straddling pair (e.g. 355 and 5) returns the true
    midpoint (0) rather than the arithmetic mean (180, ~180 deg wrong).

    Parameters
    ----------
    a, b : float
        Azimuths in degrees.

    Returns
    -------
    float
        Circular-mean azimuth in degrees, in ``[-180, 180]``.
    """
    a_rad = math.radians(a)
    b_rad = math.radians(b)
    mean_sin = (math.sin(a_rad) + math.sin(b_rad)) / 2.0
    mean_cos = (math.cos(a_rad) + math.cos(b_rad)) / 2.0
    return math.degrees(math.atan2(mean_sin, mean_cos))


def compute_nasmyth_rotation(az: float, el: float, site: Site) -> float:
    """Compute Nasmyth boresight rotation from AltAz coordinates.

    Returns ``site.nasmyth_sign * el + parallactic_angle`` where the
    parallactic angle is derived from azimuth, elevation, and site
    latitude (the AltAz form of the parallactic angle). This is the
    overhead-timeline equivalent of
    :meth:`fyst_trajectories.coordinates.Coordinates.get_field_rotation`,
    which needs RA/Dec.

    :meth:`~fyst_trajectories.coordinates.Coordinates.get_parallactic_angle`
    derives the parallactic angle from the same transformed (vacuum) Az/El,
    so this function and that method return the same value to machine
    precision.

    Parameters
    ----------
    az : float
        Azimuth in degrees.
    el : float
        Elevation in degrees.
    site : Site
        Site configuration providing ``latitude`` and ``nasmyth_sign``.

    Returns
    -------
    float
        Nasmyth boresight rotation in degrees.
    """
    az_rad = math.radians(az)
    el_rad = math.radians(el)
    lat_rad = math.radians(site.latitude)

    sin_az = math.sin(az_rad)
    cos_az = math.cos(az_rad)
    sin_el = math.sin(el_rad)
    cos_el = math.cos(el_rad)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)

    # Parallactic angle from AltAz (IAU convention).
    numerator = -sin_az * cos_lat
    denominator = sin_lat * cos_el - cos_lat * sin_el * cos_az
    pa = math.degrees(math.atan2(numerator, denominator))

    return site.nasmyth_sign * el + pa


def estimate_slew_time(
    az1: float,
    el1: float,
    az2: float,
    el2: float,
    site: Site,
) -> float:
    """Estimate telescope slew time between two positions.

    Uses a trapezoidal motion profile (accelerate to max velocity, cruise,
    decelerate). Returns the maximum of azimuth and elevation slew times
    since axes move simultaneously.

    Azimuth distance is the direct path ``abs(az2 - az1)`` when both
    positions are within the telescope's azimuth range, respecting the
    cable wrap constraint. The telescope cannot take a shorter modular
    path if it would require passing through the cable wrap boundary.

    .. note::

       Both ``az1`` and ``az2`` must be **pre-normalised** into the same
       cable-wrap window (typically the telescope's
       ``[az_min, az_max] = [-180, 360]`` range). Mixing raw astropy
       ``[0, 360]`` azimuth with telescope-normalised ``[-180, 360]``
       azimuth produces a 360°-off slew estimate; the function does not
       enforce or check this. The overhead scheduler normalises azimuth
       before every call; standalone callers must do the same.

    Parameters
    ----------
    az1, el1 : float
        Starting azimuth and elevation in degrees.
    az2, el2 : float
        Ending azimuth and elevation in degrees.
    site : Site
        Observatory site with telescope limits.

    Returns
    -------
    float
        Estimated slew time in seconds (including settle_time=0 here;
        the caller adds settle_time from OverheadModel if desired).
    """
    az_limits = site.telescope_limits.azimuth
    el_limits = site.telescope_limits.elevation

    # Direct path respecting cable wrap limits.  Both positions should
    # already be within [az_min, az_max]; the direct distance is the
    # actual motor travel without wrapping around 360 deg.
    az_dist = abs(az2 - az1)
    az_time = _axis_slew_time(az_dist, az_limits.max_velocity, az_limits.max_acceleration)
    el_time = _axis_slew_time(abs(el2 - el1), el_limits.max_velocity, el_limits.max_acceleration)
    return max(az_time, el_time)


def _axis_slew_time(distance: float, max_vel: float, max_accel: float) -> float:
    """Compute slew time for a single axis with trapezoidal profile.

    Same profile as ``sun_models._axis_slew_duration`` (deliberate
    duplication: this subpackage sits above ``sun_models`` in the import
    graph).

    Parameters
    ----------
    distance : float
        Angular distance in degrees.
    max_vel : float
        Maximum velocity in deg/s.
    max_accel : float
        Maximum acceleration in deg/s^2.

    Returns
    -------
    float
        Slew time in seconds.
    """
    if distance <= 0 or max_vel <= 0 or max_accel <= 0:
        return 0.0

    t_accel = max_vel / max_accel
    d_accel = max_vel * t_accel

    if distance <= d_accel:
        return 2.0 * math.sqrt(distance / max_accel)
    else:
        t_cruise = (distance - d_accel) / max_vel
        return 2.0 * t_accel + t_cruise


def get_observable_windows(
    ra: float,
    dec: float,
    start_time: Time,
    end_time: Time,
    site: Site,
    min_elevation: float = 30.0,
    check_sun: bool = True,
    sun_safe: "SunSafePredicate | None" = None,
) -> list[tuple[Time, Time]]:
    """Find all time windows when a target is observable.

    Composes ``Coordinates.get_rise_set_times()`` with sun avoidance
    checks to find all continuous observable windows within the given
    time range.

    Parameters
    ----------
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    start_time : Time
        Start of search window (UTC).
    end_time : Time
        End of search window (UTC).
    site : Site
        Observatory site configuration.
    min_elevation : float
        Minimum elevation in degrees (default: 30).
    check_sun : bool
        Whether to check sun avoidance (default: True).
    sun_safe : SunSafePredicate, optional
        Injected sun-safety model
        (:class:`~fyst_trajectories.dispatch.SunSafePredicate`, e.g. from
        :func:`~fyst_trajectories.sun_models.make_sun_safe`) used for the
        sun sub-window filtering instead of the site's scalar exclusion
        radius. Default ``None``. Only consulted when ``check_sun`` is
        true and the site has Sun avoidance enabled.

    Returns
    -------
    list of (Time, Time)
        List of (window_start, window_end) pairs.

    Raises
    ------
    ValueError
        If ``sun_safe`` is given while ``check_sun`` is False (the model
        would be silently ignored).
    """
    if sun_safe is not None and not check_sun:
        raise ValueError(
            "sun_safe was given but check_sun is False; enable check_sun or drop the model."
        )
    coords = Coordinates(site)
    windows = []
    search_start = start_time
    total_hours = (end_time - start_time).sec / 3600.0

    while search_start.unix < end_time.unix:
        remaining_hours = (end_time - search_start).sec / 3600.0
        if remaining_hours < 0.01:
            break

        rise, set_time = coords.get_rise_set_times(
            ra,
            dec,
            search_start,
            horizon=min_elevation,
            max_search_hours=min(remaining_hours, total_hours),
            step_hours=0.1,
        )

        if rise is None and set_time is None:
            # Check if source is currently above min_elevation
            az, el = coords.radec_to_altaz(ra, dec, search_start)
            if el > min_elevation:
                # Source is up. Search for when it sets below min_elevation.
                set_time = _find_set_time(ra, dec, search_start, end_time, coords, min_elevation)
                rise = search_start
                if set_time is None:
                    # Truly circumpolar: observable for the full remaining window
                    set_time = end_time
            else:
                break
        elif rise is not None and set_time is None:
            # Source rises but doesn't set within search window
            set_time = end_time
        elif rise is None and set_time is not None:
            # Source is setting, use search_start as rise
            rise = search_start

        # Clip to search range
        if rise.unix < start_time.unix:
            rise = start_time
        if set_time.unix > end_time.unix:
            set_time = end_time

        if set_time.unix <= rise.unix:
            search_start = set_time + TimeDelta(60, format="sec")
            continue

        if check_sun and site.sun_avoidance.enabled:
            # Sub-divide the window, removing sun-unsafe intervals
            safe_windows = _filter_sun_unsafe(
                ra,
                dec,
                rise,
                set_time,
                coords,
                site.sun_avoidance.exclusion_radius,
                sun_safe=sun_safe,
            )
            windows.extend(safe_windows)
        else:
            windows.append((rise, set_time))

        # Search for next window after this set
        search_start = set_time + TimeDelta(60, format="sec")

    return windows


def _filter_sun_unsafe(
    ra: float,
    dec: float,
    window_start: Time,
    window_end: Time,
    coords: Coordinates,
    min_sun_angle: float,
    step_minutes: float = 5.0,
    sun_safe: "SunSafePredicate | None" = None,
) -> list[tuple[Time, Time]]:
    """Filter out sun-unsafe portions of an observable window.

    Parameters
    ----------
    ra, dec : float
        Target coordinates in degrees.
    window_start, window_end : Time
        Observable window bounds.
    coords : Coordinates
        Coordinate transformer.
    min_sun_angle : float
        Minimum angular separation from Sun in degrees (scalar mode).
    step_minutes : float
        Time resolution for sun safety sampling.
    sun_safe : SunSafePredicate, optional
        Injected model whose verdicts replace the scalar separation test.

    Returns
    -------
    list of (Time, Time)
        Safe sub-windows.
    """
    duration_minutes = (window_end - window_start).sec / 60.0
    n_steps = max(2, int(duration_minutes / step_minutes) + 1)
    times = window_start + TimeDelta(
        np.linspace(0, (window_end - window_start).sec, n_steps), format="sec"
    )

    az_arr, el_arr = coords.radec_to_altaz(ra, dec, times)
    if sun_safe is not None:
        if hasattr(sun_safe, "batch"):
            safe_mask = np.atleast_1d(np.asarray(sun_safe.batch(az_arr, el_arr, times), dtype=bool))
            if safe_mask.shape != (n_steps,):
                # A scalar/short result would silently broadcast one verdict
                # over the whole window; fail loudly instead.
                raise ValueError(
                    f"sun_safe.batch returned shape {safe_mask.shape}, expected "
                    f"({n_steps},) verdicts for the window grid"
                )
        else:
            safe_mask = np.array(
                [
                    bool(sun_safe(float(az_arr[i]), float(el_arr[i]), times[i]))
                    for i in range(n_steps)
                ],
                dtype=bool,
            )
    else:
        sun_az_arr, sun_alt_arr = coords.get_sun_altaz(times)
        c_target = SkyCoord(az=az_arr * u.deg, alt=el_arr * u.deg, frame="altaz")
        c_sun = SkyCoord(az=sun_az_arr * u.deg, alt=sun_alt_arr * u.deg, frame="altaz")
        sep = c_target.separation(c_sun).deg
        safe_mask = sep > min_sun_angle

    windows = []
    in_safe = False
    safe_start = None

    for i in range(n_steps):
        if safe_mask[i] and not in_safe:
            safe_start = times[i]
            in_safe = True
        elif not safe_mask[i] and in_safe:
            windows.append((safe_start, times[i]))
            in_safe = False

    if in_safe:
        windows.append((safe_start, times[-1]))

    return windows


def _find_set_time(
    ra: float,
    dec: float,
    start_time: Time,
    end_time: Time,
    coords: Coordinates,
    min_elevation: float,
    step_hours: float = 0.1,
) -> Time | None:
    """Find when a source sets below min_elevation within a time range.

    Used when the source is currently above min_elevation and
    ``get_rise_set_times`` returned ``(None, None)`` because no rise was
    found; the source may still set within the window.

    Parameters
    ----------
    ra, dec : float
        Target coordinates in degrees.
    start_time, end_time : Time
        Search range.
    coords : Coordinates
        Coordinate transformer.
    min_elevation : float
        Horizon altitude in degrees.
    step_hours : float
        Time step for sampling in hours.

    Returns
    -------
    Time or None
        Set time, or None if the source stays above min_elevation for
        the entire range (circumpolar).
    """
    remaining_hours = (end_time - start_time).sec / 3600.0
    n_steps = max(2, int(remaining_hours / step_hours) + 1)
    dt = np.linspace(0, (end_time - start_time).sec, n_steps)
    times = start_time + TimeDelta(dt, format="sec")

    source = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    altaz_frame = AltAz(
        obstime=times,
        location=coords.location,
        pressure=0 * u.hPa,
    )
    altitudes = source.transform_to(altaz_frame).alt.to_value(u.deg)

    set_indices = np.where((altitudes[:-1] >= min_elevation) & (altitudes[1:] < min_elevation))[0]
    if len(set_indices) == 0:
        return None

    i_set = set_indices[0]
    denom = altitudes[i_set + 1] - altitudes[i_set]
    frac = 0.0 if abs(denom) < 1e-12 else (min_elevation - altitudes[i_set]) / denom
    return times[i_set] + frac * (times[i_set + 1] - times[i_set])


def get_transit_time(
    ra: float,
    dec: float,
    start_time: Time,
    site: Site,
    max_search_hours: float = 24.0,
) -> Time | None:
    """Find the next transit (meridian crossing) for a source.

    Transit occurs when the hour angle is zero (source on the meridian),
    corresponding to maximum elevation for non-circumpolar sources.

    Parameters
    ----------
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    start_time : Time
        Start of search window (UTC).
    site : Site
        Observatory site configuration.
    max_search_hours : float
        Maximum forward search window in hours.

    Returns
    -------
    Time or None
        Transit time, or None if not found within search window.
    """
    coords = Coordinates(site)
    n_steps = int(max_search_hours * 60) + 1
    dt = np.arange(n_steps) * 60.0
    times = start_time + TimeDelta(dt, format="sec")

    ha_values = coords.get_hour_angle(ra, times)

    for i in range(len(ha_values) - 1):
        if ha_values[i] <= 0 < ha_values[i + 1]:
            frac = -ha_values[i] / (ha_values[i + 1] - ha_values[i])
            transit_dt = dt[i] + frac * 60.0
            return start_time + TimeDelta(transit_dt, format="sec")

    return None


def get_max_elevation(
    ra: float,
    dec: float,
    site: Site,
) -> float:
    """Compute the maximum elevation a source reaches at this site.

    For a source at declination ``dec`` observed from latitude ``lat``,
    the maximum elevation is ``90 - |lat - dec|`` degrees.

    Parameters
    ----------
    ra : float
        Right Ascension in degrees (unused, included for API consistency).
    dec : float
        Declination in degrees.
    site : Site
        Observatory site configuration.

    Returns
    -------
    float
        Maximum elevation in degrees. At most 90 (reached when ``dec``
        equals the site latitude); can be negative for sources that never
        rise at this site.
    """
    return 90.0 - abs(site.latitude - dec)
