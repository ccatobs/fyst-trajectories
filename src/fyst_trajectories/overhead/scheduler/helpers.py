"""Pure helpers used by scheduler phases.

Dependency-free (no scheduler state); consumed by
:mod:`.phases` and the :func:`generate_timeline` wrapper.
"""

import math

import numpy as np
from astropy.time import Time, TimeDelta

from ...coordinates import Coordinates
from ...patterns.utils import normalize_azimuth
from ...site import Site
from ..constraints import Constraint, ElevationConstraint, SunAvoidanceConstraint
from ..models import ObservingPatch, OverheadModel
from ..utils import get_observable_windows

__all__ = [
    "_compute_az_range",
    "_compute_scan_duration",
    "_default_constraints",
    "_evaluate_patch",
    "_normalize_az",
    "_time_until_set",
]


def _normalize_az(az: float, site: Site) -> float:
    """Normalize a scalar azimuth into the site's cable-wrap window.

    Wraps :func:`fyst_trajectories.patterns.utils.normalize_azimuth` (which
    operates on arrays) for the scalar azimuths the scheduler carries.
    Raw astropy azimuths are in ``[0, 360)``; the slew-time and boresight
    math must compare them in the telescope's ``[az_min, az_max]`` window
    or a north-straddling pair inflates the slew distance ~17x and flips
    the boresight ~180 deg.

    Parameters
    ----------
    az : float
        Azimuth in degrees (typically raw astropy ``[0, 360)``).
    site : Site
        Site providing the azimuth limits.

    Returns
    -------
    float
        Azimuth shifted into ``[az_min, az_max]``.
    """
    return float(normalize_azimuth(np.array([az], dtype=float), site)[0])


def _default_constraints(site: Site) -> list[Constraint]:
    """Create default constraints from site configuration."""
    constraints: list[Constraint] = [
        ElevationConstraint(
            el_min=site.telescope_limits.elevation.min,
            el_max=site.telescope_limits.elevation.max,
        ),
    ]
    if site.sun_avoidance.enabled:
        constraints.append(SunAvoidanceConstraint(min_angle=site.sun_avoidance.exclusion_radius))
    return constraints


def _evaluate_patch(
    patch: ObservingPatch,
    time: Time,
    az: float,
    el: float,
    coords: Coordinates,
    constraints: list[Constraint],
) -> float:
    """Evaluate a patch against all constraints.

    Returns the product of all constraint scores. A zero from any
    constraint immediately returns 0.0 (short-circuit).
    """
    score = 1.0
    for constraint in constraints:
        s = constraint.score(patch, time, az, el, coords)
        if s == 0.0:
            return 0.0
        score *= s
    return score


def _time_until_set(
    ra: float,
    dec: float,
    start_time: Time,
    max_duration: float,
    coords: Coordinates,
    min_elevation: float,
    step_seconds: float = 300.0,
) -> float:
    """Compute how long a source stays above *min_elevation* from *start_time*.

    Samples elevation at ``step_seconds`` intervals up to *max_duration*
    using a single vectorised ``radec_to_altaz`` call, then linearly
    interpolates to find the crossing time.

    Returns *max_duration* if the source never drops below the limit
    (circumpolar or long transit).
    """
    n_steps = max(2, int(max_duration / step_seconds) + 1)
    dt = np.linspace(0.0, max_duration, n_steps)
    times = start_time + TimeDelta(dt, format="sec")

    _, el_arr = coords.radec_to_altaz(
        np.full(n_steps, ra),
        np.full(n_steps, dec),
        times,
    )

    below = np.where(el_arr < min_elevation)[0]
    if len(below) == 0:
        return max_duration

    idx = below[0]
    if idx == 0:
        return 0.0

    # Linear interpolation for sub-step precision at the crossing.
    el_prev = float(el_arr[idx - 1])
    el_curr = float(el_arr[idx])
    denom = el_prev - el_curr
    if abs(denom) < 1e-12:
        frac = 0.5
    else:
        frac = (el_prev - min_elevation) / denom
    return float(dt[idx - 1] + frac * (dt[idx] - dt[idx - 1]))


def _time_until_sun_unsafe(
    ra: float,
    dec: float,
    start_time: Time,
    max_duration: float,
    coords: Coordinates,
    min_sun_angle: float,
    step_seconds: float = 60.0,
) -> float:
    """Compute how long a source stays sun-safe from *start_time*.

    Samples target-Sun angular separation at ``step_seconds`` intervals up
    to *max_duration*, then linearly interpolates to the crossing where the
    separation first drops below *min_sun_angle*. Mirrors
    :func:`_time_until_set` (which checks elevation) so the pong/daisy
    duration clip can trim a scan that drifts into the exclusion radius
    mid-scan -- the same sun-safety guarantee the constant_el branch gets
    from :func:`get_observable_windows`.

    Returns *max_duration* if the source never enters the exclusion zone.
    """
    n_steps = max(2, int(max_duration / step_seconds) + 1)
    dt = np.linspace(0.0, max_duration, n_steps)
    times = start_time + TimeDelta(dt, format="sec")

    az_arr, el_arr = coords.radec_to_altaz(
        np.full(n_steps, ra),
        np.full(n_steps, dec),
        times,
    )
    sun_az_arr, sun_el_arr = coords.get_sun_altaz(times)
    sep = coords.angular_separation(az_arr, el_arr, sun_az_arr, sun_el_arr)

    unsafe = np.where(sep < min_sun_angle)[0]
    if len(unsafe) == 0:
        return max_duration

    idx = unsafe[0]
    if idx == 0:
        return 0.0

    # Linear interpolation for sub-step precision at the crossing.
    sep_prev = float(sep[idx - 1])
    sep_curr = float(sep[idx])
    denom = sep_prev - sep_curr
    if abs(denom) < 1e-12:
        frac = 0.5
    else:
        frac = (sep_prev - min_sun_angle) / denom
    return float(dt[idx - 1] + frac * (dt[idx] - dt[idx - 1]))


def _compute_scan_duration(
    patch: ObservingPatch,
    start_time: Time,
    end_time: Time,
    site: Site,
    coords: Coordinates,
    overhead: OverheadModel,
    center_el: float = 50.0,
) -> float:
    """Compute how long we can observe this patch.

    For constant-elevation scans, the duration is determined by how long
    the field crosses the elevation.  For pong/daisy, we start with the
    max scan duration (or remaining schedule time) and then clip to the
    observability window so the source never drops below the telescope
    elevation limit mid-scan.

    Parameters
    ----------
    center_el : float
        Computed elevation of the patch center at the current time.
        Used as fallback when patch.elevation is None.
    """
    remaining = (end_time - start_time).sec

    if patch.scan_type == "constant_el":
        el = patch.elevation if patch.elevation is not None else center_el
        windows = get_observable_windows(
            patch.ra_center,
            patch.dec_center,
            start_time,
            end_time,
            site,
            min_elevation=el - 1.0,  # Slightly below target elevation
            check_sun=site.sun_avoidance.enabled,
        )
        if not windows:
            return 0.0
        for w_start, w_end in windows:
            if w_end.unix > start_time.unix:
                window_dur = (w_end - max(w_start, start_time, key=lambda t: t.unix)).sec
                return min(window_dur, remaining)
        return 0.0
    else:
        max_dur = min(overhead.max_scan_duration, remaining)
        el_min = site.telescope_limits.elevation.min
        observable_dur = _time_until_set(
            patch.ra_center,
            patch.dec_center,
            start_time,
            max_dur,
            coords,
            el_min,
        )
        # Clip to the sun-safe sub-window too, mirroring the constant_el branch
        # (which gets this from get_observable_windows). Without it a pong/daisy
        # scan that is sun-safe at start but drifts into the exclusion radius
        # mid-scan would not be trimmed.
        if site.sun_avoidance.enabled:
            sun_safe_dur = _time_until_sun_unsafe(
                patch.ra_center,
                patch.dec_center,
                start_time,
                max_dur,
                coords,
                site.sun_avoidance.exclusion_radius,
            )
            observable_dur = min(observable_dur, sun_safe_dur)
        return min(max_dur, observable_dur)


def _compute_az_range(
    patch: ObservingPatch, center_az: float, center_el: float, site: Site
) -> tuple[float, float]:
    """Compute azimuth range for a scan.

    Uses explicit overrides from scan_params if provided, otherwise
    estimates from the field width and elevation. Both endpoints are
    normalized into the site's cable-wrap window so downstream slew-time
    and boresight math operate in a single consistent azimuth frame.

    Parameters
    ----------
    patch : ObservingPatch
        The observing patch.
    center_az : float
        Center azimuth in degrees.
    center_el : float
        Center elevation in degrees (used when patch.elevation is None).
    site : Site
        Site providing the azimuth limits for normalization.
    """
    params = patch.scan_params

    if "az_min" in params and "az_max" in params:
        return _normalize_az(float(params["az_min"]), site), _normalize_az(
            float(params["az_max"]), site
        )

    elevation = patch.elevation if patch.elevation is not None else center_el
    el_rad = math.radians(elevation)
    cos_el = max(math.cos(el_rad), 0.1)
    half_throw = patch.width / (2.0 * cos_el)

    return (
        _normalize_az(center_az - half_throw, site),
        _normalize_az(center_az + half_throw, site),
    )
