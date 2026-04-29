"""Pong scan planner (public via :mod:`fyst_trajectories.planning`)."""

import dataclasses
from typing import TYPE_CHECKING

from astropy.time import Time

from ..patterns.configs import PongScanConfig
from ..patterns.pong import compute_pong_period
from ..site import AtmosphericConditions, Site
from ._helpers import _build_celestial_trajectory
from ._sun_safety import _check_field_sun_safety
from ._types import FieldRegion, PongComputedParams, ScanBlock, validate_computed_params

if TYPE_CHECKING:
    from ..offsets import InstrumentOffset


def plan_pong_scan(
    field: FieldRegion,
    velocity: float,
    spacing: float,
    num_terms: int,
    site: Site,
    start_time: str | Time,
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
    start_time : str or Time
        Observation start time (required for celestial patterns).
        Accepts an ISO string or ``astropy.time.Time``.
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
    >>> from fyst_trajectories import get_fyst_site
    >>> from fyst_trajectories.planning import FieldRegion, plan_pong_scan
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

    if isinstance(start_time, str):
        start_time = Time(start_time, scale="utc")

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

    period, x_numvert, y_numvert = compute_pong_period(config)

    duration = period * n_cycles

    trajectory = _build_celestial_trajectory(
        site=site,
        ra=field.ra_center,
        dec=field.dec_center,
        config=config,
        duration=duration,
        start_time=start_time,
        atmosphere=atmosphere,
        detector_offset=detector_offset,
    )

    computed_params: PongComputedParams = {
        "period": period,
        "x_numvert": x_numvert,
        "y_numvert": y_numvert,
        "n_cycles": n_cycles,
    }
    validate_computed_params(computed_params, "pong")

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


def plan_pong_rotation_sequence(
    config: PongScanConfig,
    n_rotations: int,
) -> list[PongScanConfig]:
    """Generate a sequence of evenly spaced Pong rotations.

    Returns ``n_rotations`` copies of ``config`` with ``angle_i = i *
    180 / n_rotations`` (the Pong pattern is invariant under 180°
    rotation). Pass each returned config through :func:`plan_pong_scan`
    in turn.

    Parameters
    ----------
    config : PongScanConfig
        Base Pong configuration. Its ``angle`` field is ignored; the
        returned configs override it with the rotation sequence.
    n_rotations : int
        Number of rotations to emit. Must be at least 1.

    Returns
    -------
    list of PongScanConfig
        ``n_rotations`` configs, each a copy of ``config`` with
        ``angle`` overridden.

    Raises
    ------
    ValueError
        If ``n_rotations`` is less than 1.

    Examples
    --------
    >>> from fyst_trajectories import PongScanConfig
    >>> from fyst_trajectories.planning import plan_pong_rotation_sequence
    >>> base = PongScanConfig(
    ...     timestep=0.1,
    ...     width=2.0,
    ...     height=2.0,
    ...     spacing=0.1,
    ...     velocity=0.5,
    ...     num_terms=4,
    ...     angle=0.0,
    ... )
    >>> configs = plan_pong_rotation_sequence(base, n_rotations=4)
    >>> [c.angle for c in configs]
    [0.0, 45.0, 90.0, 135.0]
    """
    if n_rotations < 1:
        raise ValueError(f"n_rotations must be at least 1, got {n_rotations}")

    step = 180.0 / n_rotations
    return [dataclasses.replace(config, angle=i * step) for i in range(n_rotations)]
