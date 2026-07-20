"""Shared helpers for the ``plan_*_scan`` entry points.

These are internal utilities — the public entry points live in
:mod:`fyst_trajectories.planning` (via ``pong.py``, ``daisy.py``, and
``constant_el.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astropy.time import Time

from ..patterns.builder import TrajectoryBuilder
from ..patterns.configs import ScanConfig
from ..site import AtmosphericConditions, Site
from ..trajectory import Trajectory

if TYPE_CHECKING:
    from ..offsets import InstrumentOffset


def _coerce_start_time(start_time: str | Time) -> Time:
    """Coerce a ``start_time`` argument to an :class:`~astropy.time.Time`.

    The ``plan_*_scan`` entry points accept ``start_time`` as either an ISO
    string or a :class:`~astropy.time.Time`. A bare string is parsed in the
    UTC scale; an existing ``Time`` is returned unchanged.
    """
    if isinstance(start_time, str):
        return Time(start_time, scale="utc")
    return start_time


def _build_trajectory_with_options(
    *,
    builder: TrajectoryBuilder,
    atmosphere: AtmosphericConditions | None,
    detector_offset: InstrumentOffset | None,
) -> Trajectory:
    """Finish configuring a :class:`TrajectoryBuilder` and call ``.build()``.

    The three ``plan_*_scan`` entry points share an identical tail:
    apply an optional atmosphere, apply an optional detector offset,
    then build. Centralising that here keeps the call sites down to
    one line and ensures the option-application order stays in sync
    across planners.

    Parameters
    ----------
    builder : TrajectoryBuilder
        A partially configured builder. Callers are responsible for
        setting the pattern config, duration, and (for celestial
        patterns) ra/dec + start_time before invoking this helper.
    atmosphere : AtmosphericConditions or None
        If not ``None``, attached via
        :meth:`TrajectoryBuilder.with_atmosphere` for refraction
        correction.
    detector_offset : InstrumentOffset or None
        If not ``None``, attached via
        :meth:`TrajectoryBuilder.for_detector` so the offset detector
        tracks the target instead of the boresight.

    Returns
    -------
    Trajectory
        The built trajectory, including any detector-offset adjustment.
    """
    if atmosphere is not None:
        builder = builder.with_atmosphere(atmosphere)
    if detector_offset is not None:
        builder = builder.for_detector(detector_offset)
    return builder.build()


def _build_celestial_trajectory(
    *,
    site: Site,
    ra: float,
    dec: float,
    config: ScanConfig,
    duration: float,
    start_time: Time,
    atmosphere: AtmosphericConditions | None,
    detector_offset: InstrumentOffset | None,
) -> Trajectory:
    """Build a celestial-pattern trajectory (Pong, Daisy, etc.).

    Wraps the fluent ``TrajectoryBuilder`` chain used by
    :func:`plan_pong_scan` and :func:`plan_daisy_scan`.

    Parameters
    ----------
    site : Site
        Telescope site configuration.
    ra, dec : float
        Celestial center in degrees.
    config : ScanConfig
        Pattern configuration. The pattern class is inferred from the
        config type by :meth:`TrajectoryBuilder.with_config`.
    duration : float
        Trajectory duration in seconds.
    start_time : Time
        Observation start time (required for celestial patterns).
    atmosphere, detector_offset
        See :func:`_build_trajectory_with_options`.

    Returns
    -------
    Trajectory
        The built trajectory.
    """
    builder = (
        TrajectoryBuilder(site)
        .at(ra=ra, dec=dec)
        .with_config(config)
        .duration(duration)
        .starting_at(start_time)
    )
    return _build_trajectory_with_options(
        builder=builder,
        atmosphere=atmosphere,
        detector_offset=detector_offset,
    )


def _build_altaz_trajectory(
    *,
    site: Site,
    config: ScanConfig,
    duration: float,
    start_time: Time,
    atmosphere: AtmosphericConditions | None,
    detector_offset: InstrumentOffset | None,
) -> Trajectory:
    """Build an altaz-pattern trajectory (ConstantEl, Linear).

    Wraps the fluent ``TrajectoryBuilder`` chain used by
    :func:`plan_constant_el_scan`. Unlike celestial patterns, altaz
    patterns do not need an ``.at(ra, dec)`` call, but they still want
    a ``start_time`` so the generated trajectory carries a proper UTC
    timestamp.

    Parameters
    ----------
    site : Site
        Telescope site configuration.
    config : ScanConfig
        Pattern configuration.
    duration : float
        Trajectory duration in seconds.
    start_time : Time
        Observation start time.
    atmosphere, detector_offset
        See :func:`_build_trajectory_with_options`.

    Returns
    -------
    Trajectory
        The built trajectory.
    """
    builder = TrajectoryBuilder(site).with_config(config).duration(duration).starting_at(start_time)
    return _build_trajectory_with_options(
        builder=builder,
        atmosphere=atmosphere,
        detector_offset=detector_offset,
    )
