"""Planning module for translating astronomer inputs into pattern configurations.

This module provides a high-level planning layer that converts astronomer-style
inputs (field regions, elevation constraints, velocities) into the existing
pattern configurations and trajectories.

The main entry points are:

- :func:`plan_pong_scan` -- Plan a Pong scan over a rectangular field region.
- :func:`plan_constant_el_scan` -- Plan a CE scan with auto-computed timing and azimuth.
- :func:`plan_daisy_scan` -- Plan a Daisy scan for point-source observations.

Examples
--------
Plan a Pong scan:

>>> from astropy.time import Time
>>> from fyst_pointing import get_fyst_site
>>> from fyst_pointing.planning import FieldRegion, plan_pong_scan
>>> site = get_fyst_site()
>>> field = FieldRegion(ra_center=180.0, dec_center=-30.0, width=2.0, height=2.0)
>>> block = plan_pong_scan(
...     field=field,
...     velocity=0.5,
...     spacing=0.1,
...     num_terms=4,
...     site=site,
...     start_time=Time("2026-03-15T04:00:00", scale="utc"),
...     timestep=0.1,
... )
>>> print(block.summary)
"""

import dataclasses
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

from .coordinates import Coordinates
from .exceptions import PointingWarning
from .offsets import apply_detector_offset

# Planning functions import pattern classes directly rather than going through
# the registry or TrajectoryBuilder.  This is intentional: planning computes
# derived parameters (period, azimuth throw, n_scans, etc.) from the field
# geometry and passes them to pattern constructors, which requires access to
# the concrete classes.  The registry is designed for user-facing "name to
# class" lookup. Planning already knows which pattern it needs and benefits
# from type safety and IDE support that direct imports provide.
from .patterns.configs import ConstantElScanConfig, DaisyScanConfig, PongScanConfig, ScanConfig
from .patterns.constant_el import ConstantElScanPattern
from .patterns.daisy import DaisyScanPattern
from .patterns.pong import PongScanPattern
from .site import AtmosphericConditions, Site
from .trajectory import Trajectory

if TYPE_CHECKING:
    from .offsets import InstrumentOffset

__all__ = [
    "FieldRegion",
    "ScanBlock",
    "plan_constant_el_scan",
    "plan_daisy_scan",
    "plan_pong_scan",
]


@dataclass(frozen=True)
class FieldRegion:
    """Astronomer's specification of a rectangular field on the sky.

    Parameters
    ----------
    ra_center : float
        Right Ascension of the field center in degrees.
    dec_center : float
        Declination of the field center in degrees.
    width : float
        Angular width of the field in degrees (cross-scan direction).
        This is the physical angular extent, not the RA span. The
        cos(dec) projection is applied internally when computing
        RA boundaries. Must be positive.
    height : float
        Angular height of the field in degrees (Dec extent). Must be
        positive.

    Raises
    ------
    ValueError
        If width or height is not positive.

    Examples
    --------
    >>> field = FieldRegion(ra_center=180.0, dec_center=-30.0, width=2.0, height=2.0)
    """

    ra_center: float
    dec_center: float
    width: float
    height: float

    def __post_init__(self) -> None:
        """Validate field region parameters."""
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")

    @property
    def dec_min(self) -> float:
        """Minimum declination of the field in degrees."""
        return self.dec_center - self.height / 2.0

    @property
    def dec_max(self) -> float:
        """Maximum declination of the field in degrees."""
        return self.dec_center + self.height / 2.0


@dataclass(frozen=True)
class ScanBlock:
    """Complete observation specification produced by a planning function.

    Contains the generated trajectory, the pattern configuration used, and
    computed parameters that help the astronomer understand the observation.

    Parameters
    ----------
    trajectory : Trajectory
        The generated trajectory ready for telescope upload.
        Note: ``Trajectory`` is a mutable dataclass, so while ``ScanBlock``
        is frozen (``block.trajectory = x`` raises ``AttributeError``),
        the trajectory's internal arrays can still be modified in place
        (e.g., ``block.trajectory.metadata = y``). Treat the trajectory
        as read-only after planning.
    config : ScanConfig
        The pattern configuration used to generate the trajectory.
    duration : float
        Observation duration in seconds.
    computed_params : dict
        Dictionary of computed parameters (e.g., period, azimuth throw).
        Keys are parameter names; values are numeric (float or int).
    summary : str
        Human-readable summary of the planned observation.

    Examples
    --------
    >>> block = plan_pong_scan(...)
    >>> print(block.summary)
    >>> print(f"Duration: {block.duration:.1f}s")
    >>> print(f"Points: {block.trajectory.n_points}")
    """

    trajectory: Trajectory
    config: ScanConfig
    duration: float
    computed_params: dict[str, float | int] = dataclasses.field(default_factory=dict)
    summary: str = ""


def _check_field_sun_safety(
    ra: float,
    dec: float,
    start_time: Time,
    site: Site,
) -> None:
    """Quick pre-flight check that a field center is not near the sun.

    This is a lightweight check that warns before expensive trajectory
    generation. It never blocks trajectory generation. Violations are
    reported as warnings.

    Parameters
    ----------
    ra : float
        Right Ascension of the field center in degrees.
    dec : float
        Declination of the field center in degrees.
    start_time : Time
        Observation start time.
    site : Site
        Site configuration with sun avoidance settings.

    Warns
    -----
    PointingWarning
        If the field center is within the sun exclusion radius.
    """
    if not site.sun_avoidance.enabled:
        return
    coords = Coordinates(site, atmosphere=None)
    az, el = coords.radec_to_altaz(ra, dec, start_time)
    sun_az, sun_alt = coords.get_sun_altaz(start_time)
    sep = coords.angular_separation(az, el, sun_az, sun_alt)
    if sep <= site.sun_avoidance.exclusion_radius:
        warnings.warn(
            f"EXCLUSION ZONE: Field center passes {sep:.1f}\u00b0 from the Sun "
            f"(exclusion radius: {site.sun_avoidance.exclusion_radius}\u00b0) "
            f"at {start_time.iso}. The telescope hardware may refuse this trajectory.",
            PointingWarning,
            stacklevel=2,
        )


def plan_pong_scan(
    field: FieldRegion,
    velocity: float,
    spacing: float,
    num_terms: int,
    site: Site,
    start_time: Time,
    timestep: float,
    angle: float = 0.0,
    n_cycles: int = 1,
    detector_offset: "InstrumentOffset | None" = None,
    atmosphere: AtmosphericConditions | None = None,
) -> ScanBlock:
    """Plan a Pong scan over a rectangular field region.

    Converts astronomer-friendly field specifications into a PongScanConfig
    and generates the trajectory. By default the duration is set to complete
    ``n_cycles`` full periods of the Pong pattern.

    Parameters
    ----------
    field : FieldRegion
        Rectangular field specification (center RA/Dec, width, height).
    velocity : float
        Scan velocity in sky-offset degrees/second. This is the
        speed in the tangent plane, not azimuth coordinate
        velocity. Must be positive.
    spacing : float
        Line spacing in degrees. Must be positive.
    num_terms : int
        Number of Fourier terms for smooth turnarounds. Must be >= 1.
    site : Site
        Telescope site configuration.
    start_time : Time
        Observation start time (required for celestial patterns).
    timestep : float
        Time between trajectory points in seconds. Must be positive.
    angle : float, optional
        Rotation angle of the scan pattern in degrees. Default is 0.0
        (no rotation).
    n_cycles : int, optional
        Number of full pattern cycles to observe. Default is 1.
    detector_offset : InstrumentOffset or None, optional
        If provided, adjust the trajectory so this detector tracks the
        target instead of the boresight.
    atmosphere : AtmosphericConditions or None, optional
        Atmospheric conditions for refraction correction. If None,
        no refraction is applied.

    Returns
    -------
    ScanBlock
        Planned observation containing trajectory, config, and computed
        parameters (period, x_numvert, y_numvert).

    Raises
    ------
    ValueError
        If n_cycles is less than 1.
    TargetNotObservableError
        If the target is not observable at the requested time.
    TrajectoryBoundsError
        If the trajectory exceeds telescope limits.

    Examples
    --------
    >>> from astropy.time import Time
    >>> from fyst_pointing import get_fyst_site
    >>> from fyst_pointing.planning import FieldRegion, plan_pong_scan
    >>> site = get_fyst_site()
    >>> field = FieldRegion(ra_center=180.0, dec_center=-30.0, width=2.0, height=2.0)
    >>> block = plan_pong_scan(
    ...     field=field,
    ...     velocity=0.5,
    ...     spacing=0.1,
    ...     num_terms=4,
    ...     site=site,
    ...     start_time=Time("2026-03-15T04:00:00", scale="utc"),
    ...     timestep=0.1,
    ... )
    """
    if n_cycles < 1:
        raise ValueError(f"n_cycles must be at least 1, got {n_cycles}")

    _check_field_sun_safety(field.ra_center, field.dec_center, start_time, site)

    config = PongScanConfig(
        timestep=timestep,
        width=field.width,
        height=field.height,
        spacing=spacing,
        velocity=velocity,
        num_terms=num_terms,
        angle=angle,
    )

    pattern = PongScanPattern(ra=field.ra_center, dec=field.dec_center, config=config)
    metadata = pattern.get_metadata()
    period = metadata.pattern_params["period"]
    x_numvert = metadata.pattern_params["x_numvert"]
    y_numvert = metadata.pattern_params["y_numvert"]

    duration = period * n_cycles

    trajectory = pattern.generate(
        site=site, duration=duration, start_time=start_time, atmosphere=atmosphere
    )

    if detector_offset is not None:
        trajectory = apply_detector_offset(trajectory, detector_offset, site)

    computed_params = {
        "period": period,
        "x_numvert": x_numvert,
        "y_numvert": y_numvert,
        "n_cycles": n_cycles,
    }

    summary = (
        f"Pong scan: {field.width:.2f} x {field.height:.2f} deg field "
        f"at RA={field.ra_center:.3f}, Dec={field.dec_center:.3f}\n"
        f"  Velocity: {velocity:.3f} deg/s, Spacing: {spacing:.3f} deg, "
        f"Fourier terms: {num_terms}\n"
        f"  Period: {period:.1f}s, Cycles: {n_cycles}, "
        f"Duration: {duration:.1f}s\n"
        f"  Vertices: {x_numvert} x {y_numvert}, "
        f"Trajectory points: {trajectory.n_points}"
    )

    return ScanBlock(
        trajectory=trajectory,
        config=config,
        duration=duration,
        computed_params=computed_params,
        summary=summary,
    )


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
    if abs(cos_dec) < 1e-10:
        raise ValueError("FieldRegion too close to celestial pole (|dec| > 89.9999)")
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
    field: "FieldRegion",
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

    .. note::

        This function uses ``min(ra_vals)`` and ``max(ra_vals)`` to find
        the RA edges.  For fields spanning the RA = 0/360 boundary, these
        extremes are incorrect (e.g. a field at RA = 355 with width = 20
        has corners at 345 and 5, but ``min``/``max`` returns 5 and 345
        instead of the intended wrapping range).  The
        ``MAX_REASONABLE_SCAN_WIDTH_DEG`` guard in ``ConstantElScanConfig``
        limits practical impact, but fields near RA = 0 should be checked
        manually.

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
    ra_vals = [c[0] for c in corners]
    ra_min = min(ra_vals)
    ra_max = max(ra_vals)

    dt_sec = np.arange(0, max_search_hours * 3600, step_seconds)
    search_times = base_search_time + TimeDelta(dt_sec * u.s)

    # Find when (RA_min, Dec_center) crosses the target elevation
    _, el_min_arr = coords_obj.radec_to_altaz(
        np.full(len(search_times), ra_min),
        np.full(len(search_times), field.dec_center),
        search_times,
    )
    # Find when (RA_max, Dec_center) crosses the target elevation
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

    # Ensure start < end
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
    field: "FieldRegion",
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

    .. note::

        Like ``_compute_ce_duration``, this does not handle fields that
        straddle the azimuth 0/360 discontinuity (e.g. corners at 350°
        and 10°). Such fields would produce an incorrect ~340° throw
        instead of ~20°. This is acceptable for FYST's typical targets
        but should be addressed if planning near-north transit scans.

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
        Azimuth range in degrees.
    """
    corners = _field_region_corners(
        field.ra_center, field.dec_center, field.width, field.height, angle
    )

    obs_mid = obs_start + (obs_end - obs_start) / 2.0
    eval_times = [obs_start, obs_mid, obs_end]

    all_azimuths = []
    points = list(corners) + [(field.ra_center, field.dec_center)]
    for t in eval_times:
        for ra_c, dec_c in points:
            az_c, _ = coords_obj.radec_to_altaz(ra_c, dec_c, t)
            all_azimuths.append(az_c)

    return min(all_azimuths) - padding, max(all_azimuths) + padding


def plan_constant_el_scan(
    field: FieldRegion,
    elevation: float,
    velocity: float,
    site: Site,
    start_time: str | Time,
    rising: bool = True,
    angle: float = 0.0,
    az_accel: float = 1.0,
    timestep: float = 0.1,
    detector_offset: "InstrumentOffset | None" = None,
    az_padding: float = 2.0,
    atmosphere: AtmosphericConditions | None = None,
    max_search_hours: float = 12.0,
    step_seconds: float = 30.0,
) -> ScanBlock:
    """Plan a constant-elevation scan that covers a FieldRegion.

    Auto-computes the azimuth range and observation duration from the
    field geometry, matching the algorithm used by the FYST scan strategy
    planning tools.

    The function:

    1. Finds when the RA edges of the (optionally rotated) field cross
       the target elevation (determines start/end time and duration).
    2. Computes the azimuth range that covers the entire field at that
       elevation at the midpoint of the observation.
    3. Computes n_scans from the duration and single-leg sweep time.
    4. Builds and returns a ScanBlock.

    Parameters
    ----------
    field : FieldRegion
        Rectangular field specification (center RA/Dec, width, height).
    elevation : float
        Fixed elevation for the scan in degrees.
    velocity : float
        Azimuth scan speed in azimuth coordinate degrees/second
        (not on-sky). The on-sky speed is
        ``velocity * cos(elevation)``. This is the value sent
        directly to the Vertex ACU. Must be positive.
    site : Site
        Telescope site configuration.
    start_time : str or Time
        Approximate start time for the search window. The function
        searches up to ``max_search_hours`` forward from this time to
        find when the field edges cross the target elevation.
    rising : bool, optional
        If True (default), use the rising crossing; if False, the
        setting crossing.
    angle : float, optional
        Rotation angle of the field region in degrees. Default is 0.0.
    az_accel : float, optional
        Azimuth acceleration in degrees/second^2. Default is 1.0.
    timestep : float, optional
        Time between trajectory points in seconds. Default is 0.1.
    detector_offset : InstrumentOffset or None, optional
        If provided, adjust the trajectory for this detector offset.
    az_padding : float, optional
        Extra azimuth padding in degrees on each side of the computed
        range. Default is 2.0.
    atmosphere : AtmosphericConditions or None, optional
        Atmospheric conditions for refraction correction. If None,
        no refraction is applied.
    max_search_hours : float, optional
        Maximum time to search forward in hours for elevation crossings.
        Default is 12.0.
    step_seconds : float, optional
        Time step in seconds for the elevation crossing search.
        Default is 30.0.

    Returns
    -------
    ScanBlock
        Planned observation containing trajectory, config, and computed
        parameters (az_start, az_stop, az_throw, start_time, end_time,
        duration).

    Raises
    ------
    ValueError
        If the elevation crossings cannot be found within the search
        window.
    AzimuthBoundsError
        If the computed azimuth range exceeds telescope limits.
    ElevationBoundsError
        If the elevation exceeds telescope limits.

    Examples
    --------
    >>> from astropy.time import Time
    >>> from fyst_pointing import get_fyst_site
    >>> from fyst_pointing.planning import FieldRegion, plan_constant_el_scan
    >>> site = get_fyst_site()
    >>> field = FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7)
    >>> block = plan_constant_el_scan(
    ...     field=field,
    ...     elevation=50.0,
    ...     velocity=0.5,
    ...     site=site,
    ...     start_time=Time("2026-03-15T17:00:00", scale="utc"),
    ...     rising=True,
    ...     angle=170.0,
    ... )
    """
    if isinstance(start_time, str):
        start_time = Time(start_time, scale="utc")

    _check_field_sun_safety(field.ra_center, field.dec_center, start_time, site)

    coords_obj = Coordinates(site, atmosphere=atmosphere)

    # Find when field RA edges cross the target elevation
    obs_start, obs_end, duration = _compute_ce_duration(
        field,
        angle,
        elevation,
        coords_obj,
        start_time,
        rising,
        max_search_hours=max_search_hours,
        step_seconds=step_seconds,
    )

    # Compute azimuth range across the full observation window
    az_min, az_max = _compute_ce_az_range(field, angle, coords_obj, obs_start, obs_end, az_padding)

    # Compute n_scans from duration and sweep time
    az_throw = az_max - az_min
    scan_leg_time = az_throw / velocity
    n_scans = max(1, round(duration / scan_leg_time))

    # Build config, pattern, and trajectory
    config = ConstantElScanConfig(
        timestep=timestep,
        az_start=az_min,
        az_stop=az_max,
        elevation=elevation,
        az_speed=velocity,
        az_accel=az_accel,
        n_scans=n_scans,
    )

    pattern = ConstantElScanPattern(config=config)
    trajectory = pattern.generate(
        site=site, duration=duration, start_time=obs_start, atmosphere=atmosphere
    )

    if detector_offset is not None:
        trajectory = apply_detector_offset(trajectory, detector_offset, site)

    computed_params = {
        "az_start": az_min,
        "az_stop": az_max,
        "az_throw": az_throw,
        "n_scans": n_scans,
        "start_time_iso": obs_start.iso,
        "end_time_iso": obs_end.iso,
        "duration": duration,
    }

    summary = (
        f"Constant-El scan: {field.width:.2f} x {field.height:.2f} deg field "
        f"at RA={field.ra_center:.3f}, Dec={field.dec_center:.3f}\n"
        f"  Elevation: {elevation:.2f} deg, "
        f"Az range: [{az_min:.2f}, {az_max:.2f}] deg "
        f"(throw: {az_throw:.2f} deg)\n"
        f"  Velocity: {velocity:.3f} deg/s, Acceleration: {az_accel:.3f} deg/s^2\n"
        f"  {'Rising' if rising else 'Setting'} pass: "
        f"{obs_start.iso[:19]} to {obs_end.iso[:19]}\n"
        f"  Scans: {n_scans}, Duration: {duration:.1f}s ({duration / 60:.1f}min), "
        f"Trajectory points: {trajectory.n_points}"
    )

    return ScanBlock(
        trajectory=trajectory,
        config=config,
        duration=duration,
        computed_params=computed_params,
        summary=summary,
    )


def plan_daisy_scan(
    ra: float,
    dec: float,
    radius: float,
    velocity: float,
    turn_radius: float,
    avoidance_radius: float,
    start_acceleration: float,
    site: Site,
    start_time: Time,
    timestep: float,
    duration: float,
    y_offset: float = 0.0,
    detector_offset: "InstrumentOffset | None" = None,
    atmosphere: AtmosphericConditions | None = None,
) -> ScanBlock:
    """Plan a Daisy scan for point-source observations.

    Configures a Daisy (constant-velocity petal) scan centered on a single
    RA/Dec position. Unlike Pong scans which cover rectangular fields,
    Daisy scans repeatedly cross the central source for optimal point-source
    sensitivity.

    Parameters
    ----------
    ra : float
        Right Ascension of the source in degrees.
    dec : float
        Declination of the source in degrees.
    radius : float
        Characteristic radius R0 in degrees. Must be positive.
    velocity : float
        Scan velocity in sky-offset degrees/second. This is the
        speed in the tangent plane, not azimuth coordinate
        velocity. Must be positive.
    turn_radius : float
        Radius of curvature for turns in degrees. Must be positive.
    avoidance_radius : float
        Radius to avoid near center in degrees. Must be non-negative.
    start_acceleration : float
        Ramp-up acceleration in degrees/second^2. Must be positive.
    site : Site
        Telescope site configuration.
    start_time : Time
        Observation start time (required for celestial patterns).
    timestep : float
        Time between trajectory points in seconds. Must be positive.
    duration : float
        Observation duration in seconds. Must be positive.
    y_offset : float, optional
        Initial y offset in degrees. Default is 0.0 (start at center).
    detector_offset : InstrumentOffset or None, optional
        If provided, adjust the trajectory for this detector offset.
    atmosphere : AtmosphericConditions or None, optional
        Atmospheric conditions for refraction correction. If None,
        no refraction is applied.

    Returns
    -------
    ScanBlock
        Planned observation containing trajectory, config, and computed
        parameters.

    Raises
    ------
    TargetNotObservableError
        If the target is not observable at the requested time.
    TrajectoryBoundsError
        If the trajectory exceeds telescope limits.

    Examples
    --------
    >>> from astropy.time import Time
    >>> from fyst_pointing import get_fyst_site
    >>> from fyst_pointing.planning import plan_daisy_scan
    >>> site = get_fyst_site()
    >>> block = plan_daisy_scan(
    ...     ra=180.0,
    ...     dec=-30.0,
    ...     radius=0.5,
    ...     velocity=0.3,
    ...     turn_radius=0.2,
    ...     avoidance_radius=0.0,
    ...     start_acceleration=0.5,
    ...     site=site,
    ...     start_time=Time("2026-03-15T04:00:00", scale="utc"),
    ...     timestep=0.1,
    ...     duration=300.0,
    ... )
    """
    _check_field_sun_safety(ra, dec, start_time, site)

    config = DaisyScanConfig(
        timestep=timestep,
        radius=radius,
        velocity=velocity,
        turn_radius=turn_radius,
        avoidance_radius=avoidance_radius,
        start_acceleration=start_acceleration,
        y_offset=y_offset,
    )

    pattern = DaisyScanPattern(ra=ra, dec=dec, config=config)
    trajectory = pattern.generate(
        site=site,
        duration=duration,
        start_time=start_time,
        atmosphere=atmosphere,
    )

    if detector_offset is not None:
        trajectory = apply_detector_offset(trajectory, detector_offset, site)

    computed_params = {
        "duration": duration,
    }

    summary = (
        f"Daisy scan: radius={radius:.3f} deg at RA={ra:.3f}, Dec={dec:.3f}\n"
        f"  Velocity: {velocity:.3f} deg/s, Turn radius: {turn_radius:.3f} deg\n"
        f"  Avoidance radius: {avoidance_radius:.3f} deg, "
        f"Start acceleration: {start_acceleration:.3f} deg/s^2\n"
        f"  Duration: {duration:.1f}s, "
        f"Trajectory points: {trajectory.n_points}"
    )

    return ScanBlock(
        trajectory=trajectory,
        config=config,
        duration=duration,
        computed_params=computed_params,
        summary=summary,
    )
