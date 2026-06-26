"""Dispatch-time helpers for commanding the telescope.

These functions run at *dispatch* (command) time in the execution layer (for
example inside a PCS scan task, just before it slews to a scan's start point),
not at planning time. They turn a goal sky position into a concrete encoder
command, choosing among the telescope's redundant azimuth-wrap solutions so the
commanded slew is sun-safe.

The sun-safety test is injected via the ``sun_safe`` predicate so the directional
sun-avoidance model (a separate shared library; future work) can be plugged in
without changing call sites. The default binding is the scalar exclusion check
:meth:`fyst_trajectories.coordinates.Coordinates.is_sun_safe`.

Why this lives here and not in the scheduler: CCAT's schedule can be overridden
by the instrument at runtime, so the telescope's current position can differ from
what was planned. The azimuth-wrap / encoder choice must therefore be made at the
moment of the slew, from the *current* encoder az/el (read from the live position
broadcast), which is a dispatch-time concern.
"""

import math
from typing import Protocol, runtime_checkable

from astropy.time import Time

from .coordinates import Coordinates
from .exceptions import PointingError
from .site import Site


@runtime_checkable
class SunSafePredicate(Protocol):
    """The build-to contract for FYST's pluggable sun-avoidance model.

    This is **the** interface a sun-avoidance check must implement to be injected
    into :func:`choose_encoder_solution` via its ``sun_safe`` parameter. It is a
    structural :class:`typing.Protocol`, so any callable with the matching
    signature satisfies it; no base class or registration is required.

    The current default binding is the scalar exclusion check
    :meth:`fyst_trajectories.coordinates.Coordinates.is_sun_safe` (a single
    isotropic radius). The intended future implementer is the shared FYST
    directional sun-avoidance library (a 50-90 deg direction-dependent CAD
    table, plus the non-trapping / "pocket" escapability logic): it implements
    *this* signature and is dropped in through the ``sun_safe`` seam with **no
    change to any call site**. The scalar default and the directional model are
    interchangeable precisely because both honour this contract.

    The query is instantaneous, a single ``(az, el, time)`` point. Dwell /
    exit-window ("how soon does the Sun enter this wrap") logic is *not* part of
    this contract; it belongs to the future directional model's internal state,
    not its per-point verdict.
    """

    def __call__(self, az_deg: float, el_deg: float, time: Time) -> bool:
        """Return whether an encoder position is clear of the Sun.

        Parameters
        ----------
        az_deg : float
            Encoder azimuth in degrees (telescope range, not astropy
            ``[0, 360)`` sky range).
        el_deg : float
            Encoder elevation in degrees.
        time : Time
            Time at which to locate the Sun.

        Returns
        -------
        bool
            ``True`` when the position is clear of the Sun (safe to command),
            ``False`` when it is inside the avoidance zone.
        """
        ...


def choose_encoder_solution(
    current_az: float,
    current_el: float,
    goal_az: float,
    goal_el: float,
    obstime: Time,
    site: Site,
    *,
    sun_safe: SunSafePredicate | None = None,
) -> tuple[float, float]:
    """Choose a sun-safe encoder ``(az, el)`` to slew to for a goal sky position.

    The telescope azimuth axis travels more than one full turn
    (``site.telescope_limits.azimuth`` spans more than 360 deg), so a single sky
    azimuth has up to two valid encoder representations 360 deg apart. This
    function enumerates the in-range encoder solutions for ``goal_az`` and returns
    one that is sun-safe, preferring the smallest slew from the current encoder
    azimuth.

    This is a *dispatch-time* helper: call it with the telescope's current encoder
    position (from the live position broadcast) just before commanding the slew to
    a scan's start point, so the wrap choice reflects where the dish actually is
    rather than where the schedule assumed it would be.

    Parameters
    ----------
    current_az : float
        Current encoder azimuth in degrees (e.g. from the ACU position
        broadcast). This is an encoder value in the telescope range, not the
        astropy ``[0, 360)`` sky range.
    current_el : float
        Current encoder elevation in degrees. Part of the current-position
        contract; reserved for the elevation-aware (over-the-top / non-trapping)
        selection described in Notes. Does not affect the present minimum-slew
        choice (every candidate shares ``goal_el``).
    goal_az : float
        Target sky azimuth in degrees (e.g. the first sample of a scan
        trajectory). May be given in any range; its 360 deg images are enumerated
        against the telescope azimuth limits.
    goal_el : float
        Target sky elevation in degrees.
    obstime : Time
        Time the slew/scan begins, used to locate the Sun.
    site : Site
        Telescope site, providing the azimuth/elevation limits and (for the
        default ``sun_safe``) the sun-avoidance configuration.
    sun_safe : SunSafePredicate, optional
        Sun-safety predicate implementing the :class:`SunSafePredicate` contract,
        ``(az_deg, el_deg, time) -> bool`` returning ``True`` when the encoder
        position is clear of the Sun. Defaults to
        :meth:`~fyst_trajectories.coordinates.Coordinates.is_sun_safe` (a scalar
        exclusion radius). This is the seam for the directional sun-avoidance
        model (future shared library): pass that model's predicate here and the
        call sites do not change. See :class:`SunSafePredicate` for the contract.

    Returns
    -------
    tuple of float
        ``(encoder_az, encoder_el)`` in degrees, ready to command (e.g. via the
        PCS ``go_to`` task). ``encoder_el`` equals ``goal_el``.

    Raises
    ------
    PointingError
        If ``goal_el`` is outside the elevation limits, if no 360 deg image of
        ``goal_az`` lands within the azimuth limits, or if every in-range azimuth
        wrap is sun-blocked at ``obstime``.

    Notes
    -----
    **Selection is minimum-slew, not yet "non-trapping".** Among the sun-safe,
    in-range candidates this returns the one closest to ``current_az`` (smallest
    azimuth travel), tie-broken toward the larger margin to the azimuth travel
    limits. This matches the Simons Observatory scheduler's minimise-angular-
    deviation objective. The "non-trapping / pocket" refinement, choosing a wrap
    you can always escape from to the next target over the asymmetric directional
    avoidance map, is future work that belongs in the shared sun-avoidance
    library; it plugs in here through ``sun_safe`` (and a richer selection step)
    without changing this function's call sites.

    **Over-the-top (el > 90) is not enumerated.** FYST caps elevation at 90 deg
    (``FYST_EL_MAX``) and over-the-top pointing is forbidden during the day, so
    the third (el > 90, az + 180) encoder solution is intentionally omitted. When
    that solution is admitted, ``current_el`` will inform the choice.

    **The default sun test is instantaneous.** ``Coordinates.is_sun_safe`` checks the
    angular separation at ``obstime`` only; it has no notion of how soon the Sun
    enters a wrap (dwell / exit-window). That ``min_sun_time``-style logic belongs in
    the future directional / non-trapping model supplied via ``sun_safe``.

    Examples
    --------
    With sun avoidance disabled the choice is purely geometric (nearest wrap):

    >>> from astropy.time import Time
    >>> from fyst_trajectories import get_fyst_site
    >>> from fyst_trajectories.dispatch import choose_encoder_solution
    >>> site = get_fyst_site(sun_avoidance_enabled=False)
    >>> t = Time("2026-03-15T12:00:00", scale="utc")
    >>> # Sky az 200 deg has encoder images 200 and -160 in [-180, 360];
    >>> # from current az 190 the nearer wrap is 200.
    >>> choose_encoder_solution(190.0, 45.0, 200.0, 45.0, t, site)
    (200.0, 45.0)
    """
    el_limits = site.telescope_limits.elevation
    az_limits = site.telescope_limits.azimuth

    if not el_limits.is_in_range(goal_el):
        raise PointingError(
            f"Goal elevation {goal_el:.3f} deg is outside the telescope elevation "
            f"limits [{el_limits.min}, {el_limits.max}]."
        )

    if sun_safe is None:
        sun_safe = Coordinates(site).is_sun_safe

    # Enumerate the 360 deg azimuth images of ``goal_az`` that fall within the
    # encoder range. The k window is padded by one on each side and then filtered
    # by ``is_in_range`` so floating-point error at a boundary cannot drop a valid
    # image. For FYST's [-180, 360] range this yields one or two candidates.
    k_lo = math.floor((az_limits.min - goal_az) / 360.0) - 1
    k_hi = math.ceil((az_limits.max - goal_az) / 360.0) + 1
    candidates = [
        goal_az + 360.0 * k
        for k in range(k_lo, k_hi + 1)
        if az_limits.is_in_range(goal_az + 360.0 * k)
    ]

    if not candidates:
        raise PointingError(
            f"No encoder azimuth in range [{az_limits.min}, {az_limits.max}] "
            f"represents sky azimuth {goal_az:.3f} deg."
        )

    safe = [az for az in candidates if sun_safe(az, goal_el, obstime)]
    if not safe:
        raise PointingError(
            f"No sun-safe azimuth wrap for sky position "
            f"(az={goal_az:.3f}, el={goal_el:.3f}) deg at {obstime.iso}: every "
            f"in-range wrap {[round(c, 3) for c in candidates]} is inside the Sun "
            f"exclusion zone."
        )

    # Minimum-slew selection (see Notes); tie-break toward the larger margin to the
    # azimuth travel limits as a coarse nod to escapability.
    def _limit_margin(az: float) -> float:
        return min(az - az_limits.min, az_limits.max - az)

    encoder_az = min(safe, key=lambda az: (abs(az - current_az), -_limit_margin(az)))
    return encoder_az, goal_el
