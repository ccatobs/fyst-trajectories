"""AltAz Pong scan planner (public via :mod:`fyst_trajectories.planning`)."""

from typing import TYPE_CHECKING

from astropy.time import Time

from ..patterns.configs import PongAltAzScanConfig, PongScanConfig
from ..patterns.pong import compute_pong_period
from ..site import AtmosphericConditions, Site
from ._helpers import _build_altaz_trajectory, _coerce_start_time
from ._sun_safety import _check_altaz_center_sun_safety
from ._types import PongAltAzComputedParams, ScanBlock, validate_computed_params

if TYPE_CHECKING:
    from ..dispatch import SunSafePredicate
    from ..offsets import InstrumentOffset


def plan_pong_altaz_scan(
    az_center: float,
    el_center: float,
    width: float,
    height: float,
    spacing: float,
    velocity: float,
    site: Site,
    start_time: str | Time,
    num_terms: int = 4,
    angle: float = 0.0,
    timestep: float = 0.1,
    n_cycles: int = 1,
    detector_offset: "InstrumentOffset | None" = None,
    atmosphere: AtmosphericConditions | None = None,
    sun_safe: "SunSafePredicate | None" = None,
) -> ScanBlock:
    """Plan a Curvy-Pong scan about a fixed AltAz center.

    Generates the same on-sky Pong pattern as :func:`plan_pong_scan`, but
    executes it about a fixed horizon-frame center (``az_center``,
    ``el_center``) with no sky tracking. The on-sky tangent-plane offsets
    are mapped into telescope coordinates by::

        az = x_offset / cos(radians(el_center)) + az_center
        el = y_offset + el_center

    so ``width``, ``height``, ``spacing``, and ``velocity`` are
    tangent-plane (on-sky) quantities, identical in meaning to the
    :func:`plan_pong_scan` arguments of the same name. The azimuth
    coordinate is stretched by ``1 / cos(el_center)``: the azimuth-coordinate
    extent is ``width / cos(el_center)`` and the azimuth-coordinate speed
    exceeds the on-sky ``velocity`` by the same factor. By default the
    duration completes ``n_cycles`` full periods of the Pong pattern.

    Parameters
    ----------
    az_center : float
        Azimuth of the pattern center in degrees.
    el_center : float
        Elevation of the pattern center in degrees. Must be in ``(0, 90)``.
    width : float
        On-sky width of the scan region in degrees (cross-elevation extent
        before the ``1 / cos(el_center)`` azimuth stretch). Must be positive.
    height : float
        On-sky height of the scan region in degrees (elevation extent).
        Must be positive.
    spacing : float
        On-sky line spacing in degrees. Must be positive.
    velocity : float
        Mean diagonal on-sky scan speed in degrees/second (tangent-plane,
        not azimuth-coordinate velocity). Must be positive.
    site : Site
        Telescope site configuration.
    start_time : str or Time
        Observation start time. Accepts an ISO string or
        ``astropy.time.Time``. Used to anchor the trajectory timestamp and
        to convert the center to RA/Dec for the sun-safety pre-flight check.
    num_terms : int, optional
        Number of Fourier terms for smooth turnarounds. Default is 4.
        Must be >= 1.
    angle : float, optional
        Rotation angle of the on-sky pattern in degrees, applied in the
        tangent plane before the horizon-frame mapping. Default is 0.0.
    timestep : float, optional
        Time between trajectory points in seconds. Default is 0.1. Must be
        positive.
    n_cycles : int, optional
        Number of full pattern cycles to observe. Default is 1.
    detector_offset : InstrumentOffset or None, optional
        If provided, adjust the trajectory so this detector tracks the
        center instead of the boresight.
    atmosphere : AtmosphericConditions or None, optional
        Not used for the AltAz geometry (no coordinate transforms), accepted
        for parity with the other planners. Default is None.
    sun_safe : SunSafePredicate or None, optional
        Sun-safety predicate implementing the
        :class:`~fyst_trajectories.dispatch.SunSafePredicate` contract,
        forwarded to the center pre-flight check. ``None`` (default) keeps
        the built-in scalar exclusion-radius check; an injected predicate is
        consulted instead, so the directional sun-avoidance model (future
        shared library) is honored end-to-end. Warn-only.

    Returns
    -------
    ScanBlock
        Planned observation containing trajectory, config, and computed
        parameters (period, x_numvert, y_numvert, n_cycles, az_center,
        el_center).

    Raises
    ------
    ValueError
        If n_cycles is less than 1, or if any config field is invalid
        (non-positive width/height/spacing/velocity, num_terms < 1, or
        el_center outside ``(0, 90)``).
    TrajectoryBoundsError
        If the trajectory exceeds telescope limits.

    Examples
    --------
    >>> from astropy.time import Time
    >>> from fyst_trajectories import get_fyst_site
    >>> from fyst_trajectories.planning import plan_pong_altaz_scan
    >>> site = get_fyst_site()
    >>> block = plan_pong_altaz_scan(
    ...     az_center=120.0,
    ...     el_center=60.0,
    ...     width=2.0,
    ...     height=2.0,
    ...     spacing=0.1,
    ...     velocity=0.5,
    ...     site=site,
    ...     start_time=Time("2026-03-15T04:00:00", scale="utc"),
    ... )
    """
    if n_cycles < 1:
        raise ValueError(f"n_cycles must be at least 1, got {n_cycles}")

    start_time = _coerce_start_time(start_time)

    # Building the config first validates el_center, so a bad value raises the
    # config's clear message instead of an astropy latitude error below.
    config = PongAltAzScanConfig(
        az_center=az_center,
        el_center=el_center,
        width=width,
        height=height,
        spacing=spacing,
        velocity=velocity,
        num_terms=num_terms,
        angle=angle,
        timestep=timestep,
    )

    _check_altaz_center_sun_safety(
        site=site,
        az_center=az_center,
        el_center=el_center,
        start_time=start_time,
        sun_safe=sun_safe,
    )

    # The period depends only on the on-sky geometry, which is shared with the
    # celestial Pong, so reuse ``compute_pong_period`` via the equivalent
    # PongScanConfig rather than duplicating the Lissajous period math.
    period, x_numvert, y_numvert = compute_pong_period(
        PongScanConfig(
            timestep=timestep,
            width=width,
            height=height,
            spacing=spacing,
            velocity=velocity,
            num_terms=num_terms,
            angle=angle,
        )
    )

    duration = period * n_cycles

    trajectory = _build_altaz_trajectory(
        site=site,
        config=config,
        duration=duration,
        start_time=start_time,
        atmosphere=atmosphere,
        detector_offset=detector_offset,
    )

    computed_params: PongAltAzComputedParams = {
        "period": period,
        "x_numvert": x_numvert,
        "y_numvert": y_numvert,
        "n_cycles": n_cycles,
        "az_center": az_center,
        "el_center": el_center,
    }
    validate_computed_params(computed_params, "pong_altaz")

    summary = (
        f"AltAz Pong scan: {width:.2f} x {height:.2f} deg on-sky field "
        f"at az={az_center:.3f}, el={el_center:.3f}\n"
        f"  Velocity: {velocity:.3f} deg/s (on-sky), Spacing: {spacing:.3f} deg, "
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
