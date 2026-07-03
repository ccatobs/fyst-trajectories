"""Field-geometry helpers used by the constant-elevation scan planner.

Public to the package-internal planner; not part of the
:mod:`fyst_trajectories.planning` public API.
"""

import math
import warnings

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

from ..coordinates import Coordinates
from ..exceptions import PointingError, PointingWarning
from ._types import FieldRegion


def _field_region_corners(
    ra_center: float,
    dec_center: float,
    width: float,
    height: float,
    angle_deg: float,
) -> list[tuple[float, float]]:
    """Compute RA/Dec corners of a rotated rectangular field region.

    Uses a flat-sky approximation to rotate corners around the field center.

    Parameters
    ----------
    ra_center : float
        Right Ascension of the field center in degrees.
    dec_center : float
        Declination of the field center in degrees.
    width : float
        RA extent of the field in degrees (before rotation).
    height : float
        Dec extent of the field in degrees (before rotation).
    angle_deg : float
        Rotation angle in degrees.

    Returns
    -------
    list of (ra, dec) tuples
        The four corners of the rotated rectangle.
    """
    hw, hh = width / 2.0, height / 2.0
    corners_local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    cos_dec = math.cos(math.radians(dec_center))
    # The flat-sky rotation that follows is only a sensible approximation
    # while cos(dec) is non-trivial. ``cos(dec) < 0.01`` corresponds to
    # ``|dec| > 89.43 deg``; beyond that the cylindrical projection breaks
    # down well before the cos_dec=0 singularity. FYST's lat=-23 deg puts
    # such declinations below the elevation cut anyway, so the check is
    # a defensive boundary, not an operationally tight one.
    if abs(cos_dec) < 0.01:
        raise ValueError("FieldRegion too close to celestial pole (|dec| > 89.43 deg)")
    corners = []
    for dx, dy in corners_local:
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        corners.append((ra_center + rx / cos_dec, dec_center + ry))
    return corners


def _find_elevation_crossing(
    el_array: np.ndarray,
    search_times: Time,
    target_el: float,
    rising: bool,
    step_seconds: float,
) -> Time | None:
    """Find the first rising or setting crossing of a target elevation.

    Parameters
    ----------
    el_array : ndarray
        Elevation values at each search time.
    search_times : Time
        Array of search times.
    target_el : float
        Target elevation in degrees.
    rising : bool
        If True, find the rising crossing; if False, the setting crossing.
    step_seconds : float
        Time step between search times in seconds.

    Returns
    -------
    Time or None
        Time of crossing, or None if no crossing found.
    """
    above = el_array >= target_el
    diff = np.diff(above.astype(int))
    if rising:
        crossings = np.where(diff == 1)[0]
    else:
        crossings = np.where(diff == -1)[0]
    if len(crossings) == 0:
        return None
    idx = crossings[0]
    denom = el_array[idx + 1] - el_array[idx]
    frac = 0.5 if abs(denom) < 1e-12 else (target_el - el_array[idx]) / denom
    return search_times[idx] + TimeDelta(frac * step_seconds * u.s)


def _compute_ce_duration(
    field: FieldRegion,
    angle: float,
    elevation: float,
    coords_obj: Coordinates,
    base_search_time: Time,
    rising: bool,
    max_search_hours: float = 12.0,
    step_seconds: float = 30.0,
) -> tuple[Time, Time, float]:
    """Compute when RA edges of a field cross the target elevation.

    Searches forward from ``base_search_time`` to find when the leading
    and trailing RA edges (at the field's central Dec) pass through the
    target elevation.

    Handles the RA = 0/360 wrap by detecting when the corner span exceeds
    180° and re-centering the values around ``field.ra_center`` so the
    leading and trailing RA edges are correctly identified for fields
    near RA = 0.

    Parameters
    ----------
    field : FieldRegion
        Rectangular field specification.
    angle : float
        Field rotation angle in degrees.
    elevation : float
        Target elevation in degrees.
    coords_obj : Coordinates
        Coordinate transformer for the site.
    base_search_time : Time
        Start of the time search window.
    rising : bool
        If True, find the rising crossing; if False, the setting crossing.
    max_search_hours : float, optional
        Maximum time to search forward in hours. Default is 12.0.
    step_seconds : float, optional
        Time step for the search in seconds. Default is 30.0.

    Returns
    -------
    start_time : Time
        When the first RA edge crosses the target elevation.
    end_time : Time
        When the last RA edge crosses the target elevation.
    duration_seconds : float
        Duration in seconds.

    Raises
    ------
    ValueError
        If elevation crossings cannot be found in the search window.
    """
    corners = _field_region_corners(
        field.ra_center, field.dec_center, field.width, field.height, angle
    )
    ra_vals = [c[0] % 360.0 for c in corners]

    # Detect RA wrap-around: if the naive span exceeds 180 degrees, the
    # field straddles the RA=0/360 boundary.
    naive_span = max(ra_vals) - min(ra_vals)
    if naive_span > 180.0:
        # Shift values that are below the center into [center-180, center+180]
        ra_center_mod = field.ra_center % 360.0
        ra_vals = [((r - ra_center_mod + 180.0) % 360.0 - 180.0 + ra_center_mod) for r in ra_vals]
    ra_min = min(ra_vals)
    ra_max = max(ra_vals)

    dt_sec = np.arange(0, max_search_hours * 3600, step_seconds)
    search_times = base_search_time + TimeDelta(dt_sec * u.s)

    _, el_min_arr = coords_obj.radec_to_altaz(
        np.full(len(search_times), ra_min),
        np.full(len(search_times), field.dec_center),
        search_times,
    )
    _, el_max_arr = coords_obj.radec_to_altaz(
        np.full(len(search_times), ra_max),
        np.full(len(search_times), field.dec_center),
        search_times,
    )

    t_start = _find_elevation_crossing(el_min_arr, search_times, elevation, rising, step_seconds)
    t_end = _find_elevation_crossing(el_max_arr, search_times, elevation, rising, step_seconds)

    if t_start is None or t_end is None:
        raise ValueError(
            f"Could not find elevation crossing for field edges at el={elevation} "
            f"(rising={rising}) within {max_search_hours} hours of {base_search_time.iso}"
        )

    if t_start > t_end:
        t_start, t_end = t_end, t_start

    duration_seconds = (t_end - t_start).to_value(u.s)

    if duration_seconds > max_search_hours * 3600 * 0.5:
        warnings.warn(
            f"Computed observation duration {duration_seconds / 3600:.1f}h is unusually long. "
            f"Check field geometry and search parameters.",
            PointingWarning,
            stacklevel=2,
        )

    return t_start, t_end, duration_seconds


def _compute_ce_az_range(
    field: FieldRegion,
    angle: float,
    coords_obj: Coordinates,
    obs_start: Time,
    obs_end: Time,
    padding: float,
) -> tuple[float, float]:
    """Compute azimuth range needed to cover a field at given elevation.

    Evaluates the azimuth of all four rotated corners and the field center
    at three times (start, midpoint, end) and returns the encompassing range
    with padding. Using three times captures the temporal variation in
    azimuth coverage as the field transits.

    Handles the azimuth = 0/360 discontinuity for sources transiting through
    north (plausible at FYST's −23° latitude for sources with dec ≳ +20°):
    when the naive max−min span exceeds 180°, the samples are unwrapped
    around the median azimuth so the returned range is contiguous. The
    padded interval is then shifted by a whole turn onto a branch inside
    the telescope azimuth limits when one fits, so a setting pass (west
    of north) emits the same near-zero branch as a rising pass (east of
    north).

    Parameters
    ----------
    field : FieldRegion
        Rectangular field specification.
    angle : float
        Field rotation angle in degrees.
    coords_obj : Coordinates
        Coordinate transformer for the site.
    obs_start : Time
        Start time of the observation.
    obs_end : Time
        End time of the observation.
    padding : float
        Extra padding in degrees on each side.

    Returns
    -------
    az_min, az_max : float
        Azimuth range in degrees. May lie outside ``[0, 360)`` when the
        field straddles north (e.g. ``(-5.0, 12.0)`` rather than
        ``(355.0, 12.0)``); callers and consumers handle the unwrapped
        representation directly. The interval is placed on a 360° branch
        within the telescope azimuth limits whenever such a branch
        exists; an interval too wide for any branch is returned as-is
        for downstream bounds validation to refuse.
    """
    corners = _field_region_corners(
        field.ra_center, field.dec_center, field.width, field.height, angle
    )

    obs_mid = obs_start + (obs_end - obs_start) / 2.0
    eval_times = [obs_start, obs_mid, obs_end]

    all_azimuths: list[float] = []
    points = list(corners) + [(field.ra_center, field.dec_center)]
    for t in eval_times:
        for ra_c, dec_c in points:
            az_c, _ = coords_obj.radec_to_altaz(ra_c, dec_c, t)
            all_azimuths.append(az_c)

    az_arr = np.asarray(all_azimuths)
    # Unwrap if the samples straddle the 0/360 discontinuity. Re-centre
    # around the median so the result is a single contiguous interval.
    if az_arr.max() - az_arr.min() > 180.0:
        median = float(np.median(az_arr))
        az_arr = ((az_arr - median + 180.0) % 360.0) - 180.0 + median

    az_min = float(az_arr.min()) - padding
    az_max = float(az_arr.max()) + padding

    # Two paths land the interval on an out-of-range branch: the median re-centre
    # of a west-heavy straddle window, and padding pushing a near-360 window past
    # the limit. Both are representation choices, not infeasibility; an interval
    # that fits no branch is left for downstream bounds validation to refuse.
    lim = coords_obj.site.telescope_limits.azimuth
    if az_max > lim.max and lim.is_in_range(az_min - 360.0) and lim.is_in_range(az_max - 360.0):
        az_min -= 360.0
        az_max -= 360.0
    elif az_min < lim.min and lim.is_in_range(az_min + 360.0) and lim.is_in_range(az_max + 360.0):
        az_min += 360.0
        az_max += 360.0

    return az_min, az_max


def _compute_ce_duration_from_lsa(
    lsa_window: "tuple[float, float] | list[float]",
    coords_obj: Coordinates,
    base_search_time: Time,
    max_search_hours: float = 12.0,
    step_seconds: float = 30.0,
) -> tuple[Time, Time, float]:
    """Compute scan start/end from a Local Sidereal Angle window.

    Searches forward from ``base_search_time`` for the first time at which
    Local Sidereal Time (in degrees) crosses ``lsa_window[0]`` in the
    increasing direction. The end time is fixed at ``t_start +
    (max_lsa - min_lsa) mod 360 / 15`` hours, so the scan spans exactly
    the requested LSA window.

    Wrap-around windows (``max_lsa < min_lsa``) are handled explicitly:
    the duration is computed modulo 360°, so ``(310, 10)`` is a
    60°/15 = 4 hour scan crossing the LST = 0/360 boundary.

    Mirrors the legacy ``get_dur`` LSA branch in
    ``primecam_scan_patterns.py`` but parametrises the search anchor
    (the legacy hardcoded ``2022-09-21``) and raises structured errors
    rather than calling ``exit(1)``. The wrap-aware straddle detection
    here improves on the legacy implementation, which silently failed
    when ``min_lsa`` sat on or near the LST = 0/360 boundary.

    Parameters
    ----------
    lsa_window : tuple or list of (min_lsa, max_lsa)
        Local Sidereal Angle window in degrees. Both endpoints must lie
        in ``[0, 360)`` and must not be equal. ``max_lsa < min_lsa``
        means the window wraps through LSA = 0/360. Accepts a list as
        well as a tuple so callers can pass a value that has
        round-tripped through ECSV/JSON serialisation (which converts
        tuples to lists).
    coords_obj : Coordinates
        Coordinate transformer for the site (used for ``get_lst`` and
        the underlying ``EarthLocation``).
    base_search_time : Time
        Earliest time at which to start searching.
    max_search_hours : float, optional
        Maximum search horizon in hours. Default is 12.0. Must be
        positive.
    step_seconds : float, optional
        Time step for the search in seconds. Default is 30.0.

    Returns
    -------
    start_time : Time
        First time at or after ``base_search_time`` at which LST passes
        through ``min_lsa`` in the increasing direction.
    end_time : Time
        ``start_time`` plus the LSA-derived duration.
    duration_seconds : float
        Duration of the window in seconds.

    Raises
    ------
    ValueError
        If ``min_lsa == max_lsa`` (zero-duration window), either
        endpoint is outside ``[0, 360)``, or ``max_search_hours`` is
        not positive.
    PointingError
        If no increasing crossing of ``min_lsa`` is found within
        ``max_search_hours`` of ``base_search_time``.

    Notes
    -----
    Uses ``sidereal_time('apparent')`` (via
    :meth:`Coordinates.get_lst`), consistent with the rest of the
    fyst-trajectories pipeline. The legacy ``get_dur`` in
    ``primecam_scan_patterns.py`` uses ``sidereal_time('mean')``; the
    two differ by at most the equation of the equinoxes (~20 arcsec
    ≈ 1.3 s of time), well below operationally relevant precision for
    LSA-window scheduling.
    """
    min_lsa, max_lsa = float(lsa_window[0]), float(lsa_window[1])
    if not (0.0 <= min_lsa < 360.0) or not (0.0 <= max_lsa < 360.0):
        raise ValueError(f"lsa_window endpoints must lie in [0, 360); got ({min_lsa}, {max_lsa})")
    if min_lsa == max_lsa:
        raise ValueError(
            f"lsa_window has equal endpoints ({min_lsa}); refusing zero-duration window"
        )
    if max_search_hours <= 0:
        raise ValueError(
            f"max_search_hours must be positive, got {max_search_hours}; "
            f"cannot search for LSA crossings in a non-positive horizon"
        )

    duration_deg = (max_lsa - min_lsa) % 360.0
    duration_hours = duration_deg / 15.0
    duration_seconds = duration_hours * 3600.0

    dt_sec = np.arange(0.0, max_search_hours * 3600.0, step_seconds)
    search_times = base_search_time + TimeDelta(dt_sec * u.s)
    lsa = np.asarray(coords_obj.get_lst(search_times))

    # Unwrap LSA into a monotonic series so straddle detection works
    # uniformly when ``min_lsa`` sits at or near the LST = 0/360
    # boundary. With wrapped samples (e.g. 359.9 -> 0.1) and ``min_lsa
    # = 0``, the naive product test ``(lsa - 0) * (lsa_next - 0)`` is
    # positive at the wrap edge, producing a silent miss. ``np.unwrap``
    # turns 359.9 -> 360.1 -> ..., and LST is monotonically increasing in
    # time, so the unwrapped series stays monotonic.
    lsa_unwrapped = np.unwrap(np.deg2rad(lsa)) * 180.0 / np.pi
    # The smallest ``target = min_lsa + 360*k`` that lies at or after
    # the first sample is the only candidate crossing within the
    # horizon (LST advances ~15 deg/h; even a 24-hour search covers only
    # one full wrap, so the *first* increasing crossing of ``min_lsa``
    # is uniquely determined by this offset).
    k = math.ceil((lsa_unwrapped[0] - min_lsa) / 360.0)
    target = min_lsa + 360.0 * k
    if target > lsa_unwrapped[-1]:
        raise PointingError(
            f"lsa_window min_lsa={min_lsa} deg not reached in increasing direction within "
            f"{max_search_hours} h of {base_search_time.iso}. "
            f"LSA swept {lsa[0]:.1f} deg -> {lsa[-1]:.1f} deg."
        )

    # Find the [idx, idx+1] bracket that contains ``target``.
    idx = int(np.searchsorted(lsa_unwrapped, target, side="right")) - 1
    # Clamp into [0, len-2] for safety; ``target <= lsa_unwrapped[-1]``
    # guarantees ``idx+1`` is in range, but ``target == lsa_unwrapped[0]``
    # would yield ``idx == -1``.
    idx = max(0, min(idx, len(lsa_unwrapped) - 2))
    lsa_left = lsa_unwrapped[idx]
    lsa_right = lsa_unwrapped[idx + 1]
    # ``lsa_right > lsa_left`` is guaranteed by monotonicity of the
    # unwrapped LST series (denom > 0 strictly).
    frac = (target - lsa_left) / (lsa_right - lsa_left)
    t_start = search_times[idx] + TimeDelta(frac * step_seconds * u.s)
    t_end = t_start + TimeDelta(duration_seconds * u.s)

    # Absolute threshold: LSA-window duration is independent of the
    # search horizon, so the warning should reflect operational scale
    # (sustained ~6 h pointing is unusually long for a single CE block)
    # rather than couple to ``max_search_hours``.
    if duration_hours > 6.0:
        warnings.warn(
            f"LSA-derived duration {duration_hours:.1f}h is unusually long for a single "
            f"CE block. Check lsa_window={lsa_window}.",
            PointingWarning,
            stacklevel=2,
        )

    return t_start, t_end, duration_seconds
