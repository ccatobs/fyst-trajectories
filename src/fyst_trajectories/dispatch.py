"""Dispatch-time helpers for commanding the telescope.

These functions run at *dispatch* (command) time in the execution layer (for
example inside a PCS scan task, just before it slews to a scan's start point),
not at planning time. They turn a goal sky position into a concrete encoder
command, choosing among the telescope's redundant azimuth-wrap solutions so the
commanded slew is sun-safe.

The sun-safety test is injected via the ``sun_safe`` predicate so the directional
sun-avoidance model (bound in :mod:`fyst_trajectories.sun_models`) plugs in
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

    The default binding is the scalar exclusion check
    :meth:`fyst_trajectories.coordinates.Coordinates.is_sun_safe` (one isotropic
    radius). The directional alternative, ``make_sun_safe("cad")``
    (:mod:`fyst_trajectories.sun_models`, FYST's own 50-90 deg
    direction-dependent zone from the shared sun-avoidance library), implements
    this same signature, so swapping models changes no call site.

    The query is instantaneous, a single ``(az, el, time)`` point. A caller may
    query it at several instants to cover a dwell window (see
    :func:`choose_encoder_solution`'s array-valued ``obstime``); implementations
    stay single-instant. Dwell / exit-window ("how soon does the Sun enter this
    wrap") logic is *not* part of this contract; it belongs to the directional
    model's internal state, not its per-point verdict.

    **Optional batch extension.** An implementation MAY additionally expose
    ``batch(az_deg, el_deg, times) -> numpy.ndarray[bool]`` (vectorized
    verdicts over broadcastable inputs).
    :func:`~fyst_trajectories.observability.check_observability` uses
    ``batch`` when present and falls back to per-sample scalar calls
    otherwise; the visibility renderer's ``sun_model`` parameter *requires*
    the ``batch``/``threshold`` extensions; dispatch always uses the scalar
    call. The predicates built by
    :func:`~fyst_trajectories.sun_models.make_sun_safe` implement all of
    it.

    This contract judges a POINT. Whether the *slew path* to that point
    stays clear is the separate :class:`SlewSafePredicate` contract.
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


@runtime_checkable
class SlewSafePredicate(Protocol):
    """The path-level sibling of :class:`SunSafePredicate`.

    Judges whether the telescope's DIRECT slew from the current encoder
    position to a goal encoder position stays clear of the Sun for the
    whole motion, with the Sun advanced along the path. A correct
    single-point predicate is necessarily invariant under ``az -> az +
    360`` (same sky direction), so it can never *choose* an azimuth wrap;
    the paths to two wraps differ, which is exactly what this contract
    evaluates. Built by
    :func:`~fyst_trajectories.sun_models.make_slew_safe`, which sweeps a
    point model along the trapezoidal kinematic path using the FYST axis
    velocity/acceleration limits.

    Consumed by :func:`choose_encoder_solution`'s optional ``slew_safe``
    parameter to rank admissible wraps by path safety. When no direct path is
    clear the dispatch layer raises rather than auto-rerouting; a caller wanting
    a two-leg reroute plans it explicitly via
    :func:`~fyst_trajectories.sun_models.find_sun_safe_detour`.
    """

    def __call__(
        self,
        current_az: float,
        current_el: float,
        goal_az: float,
        goal_el: float,
        time: Time,
    ) -> bool:
        """Return whether the direct slew is clear of the Sun throughout.

        Parameters
        ----------
        current_az, current_el : float
            Current encoder position in degrees.
        goal_az, goal_el : float
            Goal encoder position in degrees (``goal_az`` in the
            telescope encoder range; the path is the literal encoder
            travel, no wrapping).
        time : Time
            Slew start time (scalar); implementations advance the Sun
            along the path from here.

        Returns
        -------
        bool
            ``True`` when the whole path is clear of the Sun.
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
    slew_safe: SlewSafePredicate | None = None,
    goal_az_span: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Choose a sun-safe encoder ``(az, el)`` to slew to for a commanded trajectory.

    The telescope azimuth axis travels more than one full turn
    (``site.telescope_limits.azimuth`` spans more than 360 deg), so a single sky
    azimuth has up to two valid encoder representations 360 deg apart. This
    function enumerates the in-range encoder wraps and returns one that is
    sun-safe, preferring the smallest slew from the current encoder azimuth.

    When ``goal_az_span`` is supplied, wrap admissibility is judged against the
    whole commanded trajectory's azimuth span rather than the goal point alone:
    the caller shifts the entire trajectory by the chosen 360 deg multiple, so a
    wrap is admissible only if both span endpoints stay within the azimuth limits
    after that shift. This keeps the nearest wrap from being chosen when it would
    push a north-crossing scan's span outside the limits even though the other
    wrap fits.

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
        against the telescope azimuth limits. Must lie within ``goal_az_span``
        when that is supplied.
    goal_el : float
        Target sky elevation in degrees.
    obstime : Time
        The time or times at which the commanded position must be clear of the
        Sun (for example every sample of a pre-scan dwell). Scalar or
        array-valued astropy :class:`~astropy.time.Time`; a wrap is sun-safe only
        if the predicate holds at every element.
    site : Site
        Telescope site, providing the azimuth/elevation limits and (for the
        default ``sun_safe``) the sun-avoidance configuration.
    sun_safe : SunSafePredicate, optional
        Sun-safety predicate implementing the :class:`SunSafePredicate` contract,
        ``(az_deg, el_deg, time) -> bool`` returning ``True`` when the encoder
        position is clear of the Sun. Defaults to
        :meth:`~fyst_trajectories.coordinates.Coordinates.is_sun_safe` (a scalar
        exclusion radius). This is the seam for the directional sun-avoidance
        model (e.g. :func:`~fyst_trajectories.sun_models.make_sun_safe`): pass
        that model's predicate here and the call sites do not change. See
        :class:`SunSafePredicate` for the contract.
    slew_safe : SlewSafePredicate, optional
        Path-level sun-safety check (e.g. from
        :func:`~fyst_trajectories.sun_models.make_slew_safe`). When given,
        the point-safe wraps are additionally required to have a clear
        DIRECT slew path from ``(current_az, current_el)``, evaluated at
        the first ``obstime`` element (the slew happens now; the dwell
        samples cover the goal). The minimum-slew choice then runs over the
        path-safe candidates, making ``current_az``/``current_el``
        load-bearing for safety, not only for distance. If no wrap has a
        clear direct path, :class:`~fyst_trajectories.exceptions.PointingError`
        is raised (dispatch rejects rather than auto-rerouting); a caller may
        then plan a two-leg detour via
        :func:`~fyst_trajectories.sun_models.find_sun_safe_detour`.
    goal_az_span : tuple of float, optional
        ``(span_min, span_max)``, the minimum and maximum azimuth of the full
        commanded trajectory in the SAME wrap frame as ``goal_az`` (e.g.
        ``(traj.az.min(), traj.az.max())`` where ``goal_az == traj.az[0]``).
        ``goal_az`` need not equal either endpoint but must lie within the span.
        A wrap ``goal_az + 360 k`` is admissible only if both shifted endpoints
        stay within the azimuth limits, matching the whole-trajectory shift the
        caller applies after this function returns. When ``None`` (default),
        admissibility reduces to the goal point alone.

    Returns
    -------
    tuple of float
        ``(encoder_az, encoder_el)`` in degrees, ready to command (e.g. via the
        PCS ``go_to`` task). ``encoder_el`` equals ``goal_el``.

    Raises
    ------
    ValueError
        If ``goal_az_span`` is given with ``span_min > span_max`` or with
        ``goal_az`` outside ``[span_min, span_max]`` by more than a small
        tolerance.
    PointingError
        If ``goal_el`` is outside the elevation limits, if no 360 deg image of
        ``goal_az`` lands within the azimuth limits, if no wrap keeps the whole
        span within the azimuth limits, if every admissible azimuth wrap is
        sun-blocked at some element of ``obstime``, or (with ``slew_safe``)
        if no point-safe wrap has a clear direct slew path.

    Notes
    -----
    **Selection is minimum-slew.** Among the sun-safe, in-range candidates this
    returns the one closest to ``current_az`` (smallest azimuth travel),
    tie-broken toward the larger margin to the azimuth travel limits, measured
    against the shifted span endpoints. The selection does not yet prefer a wrap
    the telescope can always escape from to the next target ("non-trapping"
    selection over the directional avoidance map); that refinement plugs in
    through ``sun_safe`` and a richer selection step. The Simons Observatory
    scheduler handles the azimuth branch differently: ``make_source_ces`` takes
    it as a caller-declared preference (``az_branch``) and wraps into it, rather
    than searching over candidates.

    **Over-the-top (el > 90) is not enumerated.** The library caps elevation at
    ``FYST_EL_MAX`` (90 deg; Prime-Cam does not point over the top), so the third
    (el > 90, az + 180) encoder solution is intentionally omitted. If that
    solution is ever admitted, ``current_el`` will inform the choice.

    **The default sun test is instantaneous.** ``Coordinates.is_sun_safe`` checks
    the angular separation at one instant; it has no notion of how soon the Sun
    enters a wrap (dwell / exit-window). Passing several ``obstime`` elements
    covers a dwell only by sampling; that ``min_sun_time``-style logic belongs in
    a richer non-trapping model supplied via ``sun_safe``.

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

    # A bare goal is the degenerate point span (goal_az, goal_az); every check
    # below runs on the resolved endpoints.
    span_lo, span_hi = goal_az_span if goal_az_span is not None else (goal_az, goal_az)
    # The tolerance absorbs float round-off from the caller building the span
    # (e.g. traj.az.min()/max() vs traj.az[0]).
    tol = 1e-6
    if span_lo > span_hi:
        raise ValueError(f"goal_az_span min ({span_lo}) must be <= max ({span_hi}).")
    if not (span_lo - tol <= goal_az <= span_hi + tol):
        raise ValueError(f"goal_az {goal_az} must lie within goal_az_span [{span_lo}, {span_hi}].")

    # Enumerate the 360 deg images of ``goal_az`` whose whole span lands in the
    # encoder range. The k window brackets the span endpoints, padded by one on
    # each side and filtered by ``is_in_range`` so floating-point error at a
    # boundary cannot drop a valid image.
    k_lo = math.floor((az_limits.min - span_hi) / 360.0) - 1
    k_hi = math.ceil((az_limits.max - span_lo) / 360.0) + 1

    admissible = []
    goal_image_in_range = False
    for k in range(k_lo, k_hi + 1):
        shift = 360.0 * k
        if az_limits.is_in_range(goal_az + shift):
            goal_image_in_range = True
        if az_limits.is_in_range(span_lo + shift) and az_limits.is_in_range(span_hi + shift):
            admissible.append((goal_az + shift, shift))

    if not admissible:
        # Distinguish a span that fits no wrap from a goal with no in-range image
        # at all (for a point span the two coincide and the goal error fires).
        if goal_image_in_range:
            span_width = span_hi - span_lo
            raise PointingError(
                f"trajectory azimuth span [{span_lo:.3f}, {span_hi:.3f}] deg "
                f"(width {span_width:.3f}) does not fit within the telescope "
                f"azimuth limits [{az_limits.min}, {az_limits.max}] in any wrap "
                f"of sky azimuth {goal_az:.3f}."
            )
        raise PointingError(
            f"No encoder azimuth in range [{az_limits.min}, {az_limits.max}] "
            f"represents sky azimuth {goal_az:.3f} deg."
        )

    check_times = [obstime] if obstime.isscalar else list(obstime)
    safe = [
        (az, shift)
        for az, shift in admissible
        if all(sun_safe(az, goal_el, t) for t in check_times)
    ]
    if not safe:
        when = (
            check_times[0].iso
            if len(check_times) == 1
            else f"{check_times[0].iso} through {check_times[-1].iso}"
        )
        raise PointingError(
            f"No sun-safe azimuth wrap for sky position "
            f"(az={goal_az:.3f}, el={goal_el:.3f}) deg at {when}: every "
            f"in-range wrap {[round(az, 3) for az, _ in admissible]} is inside the "
            f"Sun exclusion zone."
        )

    if slew_safe is not None:
        # Path admissibility: the slew starts NOW, so evaluate at the first
        # obstime element; the remaining elements cover the goal dwell and
        # were already required point-safe above.
        t_slew = check_times[0]
        path_safe = [
            (az, shift)
            for az, shift in safe
            if slew_safe(current_az, current_el, az, goal_el, t_slew)
        ]
        if not path_safe:
            raise PointingError(
                f"Every point-safe azimuth wrap {[round(az, 3) for az, _ in safe]} for "
                f"sky position (az={goal_az:.3f}, el={goal_el:.3f}) deg has a direct "
                f"slew path from (az={current_az:.3f}, el={current_el:.3f}) that "
                f"crosses the Sun avoidance zone at {t_slew.iso}. No direct slew is "
                "safe; a two-leg detour may be available "
                "(fyst_trajectories.sun_models.find_sun_safe_detour)."
            )
        safe = path_safe

    # Minimum-slew selection (see Notes); the limit-margin tie-break is a coarse
    # nod to escapability, measured against the shifted span endpoints.
    def _limit_margin(shift: float) -> float:
        return min(span_lo + shift - az_limits.min, az_limits.max - (span_hi + shift))

    encoder_az, _ = min(safe, key=lambda c: (abs(c[0] - current_az), -_limit_margin(c[1])))
    return encoder_az, goal_el
