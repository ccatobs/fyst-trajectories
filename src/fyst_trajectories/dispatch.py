"""Dispatch-time helpers for commanding the telescope.

These functions run at *dispatch* (command) time in the execution layer (for
example inside a typed scan task, just before it slews to a scan's start
point), not at planning time. They turn a goal sky position into a concrete encoder
command, choosing among the telescope's redundant azimuth-wrap solutions so the
commanded slew is sun-safe, and estimate how long that slew takes.

The sun-safety test is injected via the ``sun_safe`` predicate so the directional
sun-avoidance model (bound in :mod:`fyst_trajectories.sun_models`) plugs in
without changing call sites; its contracts are in
:mod:`fyst_trajectories.sun_protocols`. The default binding is the scalar model
``make_sun_safe("scalar")`` builds from the site's exclusion radius, whose
verdicts are those of :meth:`fyst_trajectories.coordinates.Coordinates.is_sun_safe`.

Why this lives here and not in the scheduler: CCAT's schedule can be overridden
by the instrument at runtime, so the telescope's current position can differ from
what was planned. The azimuth-wrap / encoder choice must therefore be made at the
moment of the slew, from the *current* encoder az/el (read from the live position
broadcast), which is a dispatch-time concern.
"""

import math

import numpy as np
from astropy.time import Time

from .exceptions import EncoderSolutionError
from .site import Site
from .sun_models import _axis_slew_duration, make_sun_safe
from .sun_protocols import SlewSafePredicate, SunSafePredicate, _sun_verdicts


class EncoderSolution(tuple):
    """The encoder pose :func:`choose_encoder_solution` chose, with its wrap.

    A two-element tuple of ``(az, el)`` in degrees, also readable as the
    properties of those names, so ``az, el = choose_encoder_solution(...)``
    unpacks it. It additionally carries the azimuth wrap the pose belongs to.

    The value is immutable. Equality and hashing are those of the
    ``(az, el)`` tuple and ignore ``az_shift``, which depends on the frame
    the caller's goal azimuth was given in rather than on the pose, so a
    solution compares equal to the bare pair; compare
    ``(tuple(solution), solution.az_shift)`` when the shift matters.

    Examples
    --------
    >>> solution = EncoderSolution(-160.0, 45.0, -360.0)
    >>> az, el = solution
    >>> az, solution.az_shift
    (-160.0, -360.0)
    """

    az_shift: float
    """The multiple of 360 degrees that carries the caller's goal azimuth frame onto
    the chosen wrap. Add it to every azimuth of the commanded trajectory before
    sending it, which is what
    :func:`~fyst_trajectories.patterns.rewrap_trajectory_azimuth` does; ``0.0``
    when the trajectory is already in the chosen wrap."""

    def __new__(cls, az: float, el: float, az_shift: float) -> "EncoderSolution":
        """Build the pose tuple and attach the wrap shift."""
        solution = super().__new__(cls, (float(az), float(el)))
        # The one assignment; __setattr__ refuses every later one.
        object.__setattr__(solution, "az_shift", float(az_shift))
        return solution

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"EncoderSolution is immutable; cannot set {name!r}")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"EncoderSolution is immutable; cannot delete {name!r}")

    def __getnewargs__(self) -> tuple[float, float, float]:
        """Return the arguments that rebuild this value (``copy``/``pickle``)."""
        return (self[0], self[1], self.az_shift)

    @property
    def az(self) -> float:
        """Encoder azimuth in degrees."""
        return self[0]

    @property
    def el(self) -> float:
        """Encoder elevation in degrees."""
        return self[1]

    def __repr__(self) -> str:
        return f"EncoderSolution(az={self[0]!r}, el={self[1]!r}, az_shift={self.az_shift!r})"


def _wraps_sun_safe(
    sun_safe: SunSafePredicate,
    az_candidates: np.ndarray,
    goal_el: float,
    check_times: Time,
) -> np.ndarray:
    """Return, per candidate azimuth, whether every dwell time is sun-safe.

    The Sun's position does not depend on the azimuth wrap, so a model that
    exposes the vectorised ``batch`` extension is asked for the whole
    ``(wrap, time)`` grid in one call and solves the ephemeris once. A bare
    predicate has no such entry point and falls back to per-pair calls.

    Parameters
    ----------
    sun_safe : SunSafePredicate
        The point model to consult.
    az_candidates : np.ndarray
        Candidate encoder azimuths in degrees.
    goal_el : float
        Commanded elevation in degrees, the same for every candidate.
    check_times : Time
        Non-empty array of times the position must be clear at.

    Returns
    -------
    np.ndarray
        Boolean array, one entry per candidate azimuth.

    Raises
    ------
    ValueError
        If ``sun_safe.batch`` returns a result of the wrong shape.
    """
    n_az = az_candidates.size
    n_t = len(check_times)
    az_grid = np.repeat(az_candidates, n_t)
    el_grid = np.full(n_az * n_t, float(goal_el))
    time_grid = check_times[np.tile(np.arange(n_t), n_az)]
    verdict = _sun_verdicts(sun_safe, az_grid, el_grid, time_grid, what="wrap and dwell grid")
    return verdict.reshape(n_az, n_t).all(axis=1)


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
) -> EncoderSolution:
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
        selection described in Notes. Does not affect the minimum-slew choice
        itself (every candidate shares ``goal_el``), but it does select the
        path evaluated when ``slew_safe`` is supplied, so it decides whether a
        wrap survives the path gate at all.
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
        Sun-safety predicate implementing the
        :class:`~fyst_trajectories.sun_protocols.SunSafePredicate` contract,
        ``(az_deg, el_deg, time) -> bool`` returning ``True`` when the encoder
        position is clear of the Sun. Defaults to the scalar model
        ``make_sun_safe("scalar", site=site)`` builds (the site's exclusion
        radius, with the verdicts of
        :meth:`~fyst_trajectories.coordinates.Coordinates.is_sun_safe`). This is
        the seam for the directional sun-avoidance model (e.g.
        :func:`~fyst_trajectories.sun_models.make_sun_safe`): pass that model's
        predicate here and the call sites do not change. See
        :class:`~fyst_trajectories.sun_protocols.SunSafePredicate` for the
        contract. A model exposing the optional vectorised ``batch`` extension
        (:class:`~fyst_trajectories.sun_protocols.BatchSunSafePredicate`) is
        consulted once for the whole ``(wrap, time)`` grid rather than per
        pair, which is what keeps a long dwell grid off the dispatch critical
        path; the default does expose it.
    slew_safe : SlewSafePredicate, optional
        Path-level sun-safety check (e.g. from
        :func:`~fyst_trajectories.sun_models.make_slew_safe`). When given,
        the point-safe wraps are additionally required to have a clear
        DIRECT slew path from ``(current_az, current_el)``, evaluated at
        the first ``obstime`` element (the slew happens now; the dwell
        samples cover the goal). The minimum-slew choice then runs over the
        path-safe candidates, making ``current_az``/``current_el``
        load-bearing for safety, not only for distance. If no wrap has a
        clear direct path, :class:`~fyst_trajectories.exceptions.EncoderSolutionError`
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
    EncoderSolution
        ``(encoder_az, encoder_el)`` in degrees, ready to command (e.g. via the
        execution layer's go-to task). ``encoder_el`` equals ``goal_el``. The
        value is a two-element tuple, so ``az, el = choose_encoder_solution(...)``
        works; its ``az_shift`` attribute additionally carries the multiple of
        360 degrees that maps the caller's goal azimuth frame onto the chosen
        wrap, which the whole commanded trajectory has to be shifted by (see
        :func:`~fyst_trajectories.patterns.rewrap_trajectory_azimuth`).

    Raises
    ------
    ValueError
        If ``current_az``, ``current_el``, ``goal_az``, ``goal_el`` or an
        endpoint of ``goal_az_span`` is NaN or infinite (a lost position
        read is refused rather than turned into a wrap choice), if
        ``goal_az_span`` is given with ``span_min > span_max`` or with
        ``goal_az`` outside ``[span_min, span_max]`` by more than a small
        tolerance, if ``obstime`` is an empty ``Time`` array (the sun gate
        fails closed rather than passing every wrap vacuously), if the default
        Sun test is given a ``goal_el`` outside [-90, 90] deg (possible only on
        a custom ``Site`` whose elevation limits extend past it, Sun avoidance
        enabled or not), or if ``sun_safe.batch`` returns a result of the
        wrong shape.
    EncoderSolutionError
        A :class:`~fyst_trajectories.exceptions.PointingError` whose
        ``cause`` names the refusing stage: ``"goal_elevation"`` if
        ``goal_el`` is outside the elevation limits, ``"no_image"`` if no
        360 deg image of ``goal_az`` lands within the azimuth limits,
        ``"span_unreachable"`` if no wrap keeps the whole span within the
        azimuth limits, ``"sun_blocked"`` if every admissible azimuth wrap
        is sun-blocked at some element of ``obstime``, or (with
        ``slew_safe``) ``"path_blocked"`` if no point-safe wrap has a clear
        direct slew path. The last two carry the surviving wraps as
        ``candidates``.

    Notes
    -----
    **Selection is minimum-slew.** Among the sun-safe, in-range candidates this
    returns the one closest to ``current_az`` (smallest azimuth travel),
    tie-broken toward the larger margin to the azimuth travel limits, measured
    against the shifted span endpoints.

    **Over-the-top (el > 90) is not enumerated.** The library caps elevation at
    ``FYST_EL_MAX`` (90 deg; Prime-Cam does not point over the top), so the third
    (el > 90, az + 180) encoder solution is intentionally omitted, and the default
    Sun test refuses a goal elevation above 90 deg.

    **The default sun test is instantaneous.** The scalar model checks the
    angular separation at one instant; it has no notion of how soon the Sun
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
    EncoderSolution(az=200.0, el=45.0, az_shift=0.0)
    >>> az, el = choose_encoder_solution(190.0, 45.0, 200.0, 45.0, t, site)
    >>> az, el
    (200.0, 45.0)
    """
    # Refuse a NaN or infinite pose or span endpoint here, naming the argument:
    # past this point a NaN fails every comparison silently, and an infinity can
    # overflow the wrap enumeration or make every slew distance equal.
    for name, value in (
        ("current_az", current_az),
        ("current_el", current_el),
        ("goal_az", goal_az),
        ("goal_el", goal_el),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
    if goal_az_span is not None and not all(math.isfinite(v) for v in goal_az_span):
        raise ValueError(f"goal_az_span must be finite, got {goal_az_span}")

    el_limits = site.telescope_limits.elevation
    az_limits = site.telescope_limits.azimuth

    if not el_limits.is_in_range(goal_el):
        raise EncoderSolutionError(
            "goal_elevation",
            f"Goal elevation {goal_el:.3f} deg is outside the telescope elevation "
            f"limits [{el_limits.min}, {el_limits.max}].",
            goal_az=goal_az,
            goal_el=goal_el,
        )

    if sun_safe is None:
        sun_safe = make_sun_safe("scalar", site=site)

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
            raise EncoderSolutionError(
                "span_unreachable",
                f"trajectory azimuth span [{span_lo:.3f}, {span_hi:.3f}] deg "
                f"(width {span_width:.3f}) does not fit within the telescope "
                f"azimuth limits [{az_limits.min}, {az_limits.max}] in any wrap "
                f"of sky azimuth {goal_az:.3f}.",
                goal_az=goal_az,
                goal_el=goal_el,
            )
        raise EncoderSolutionError(
            "no_image",
            f"No encoder azimuth in range [{az_limits.min}, {az_limits.max}] "
            f"represents sky azimuth {goal_az:.3f} deg.",
            goal_az=goal_az,
            goal_el=goal_el,
        )

    check_times = obstime.reshape(1) if obstime.isscalar else obstime
    if len(check_times) == 0:
        # Fail closed: an empty time array would make the all(...) sun gate
        # below vacuously pass every wrap, silently skipping the check.
        raise ValueError(
            "obstime is an empty Time array; the sun-safety gate cannot be "
            "evaluated. Pass at least one time."
        )
    verdicts = _wraps_sun_safe(
        sun_safe, np.array([az for az, _ in admissible], dtype=float), goal_el, check_times
    )
    safe = [candidate for candidate, ok in zip(admissible, verdicts) if ok]
    if not safe:
        when = (
            check_times[0].iso
            if len(check_times) == 1
            else f"{check_times[0].iso} through {check_times[-1].iso}"
        )
        raise EncoderSolutionError(
            "sun_blocked",
            f"No sun-safe azimuth wrap for sky position "
            f"(az={goal_az:.3f}, el={goal_el:.3f}) deg at {when}: every "
            f"in-range wrap {[round(az, 3) for az, _ in admissible]} is inside the "
            f"Sun exclusion zone.",
            goal_az=goal_az,
            goal_el=goal_el,
            candidates=[az for az, _ in admissible],
            time_iso=when,
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
            raise EncoderSolutionError(
                "path_blocked",
                f"Every point-safe azimuth wrap {[round(az, 3) for az, _ in safe]} for "
                f"sky position (az={goal_az:.3f}, el={goal_el:.3f}) deg has a direct "
                f"slew path from (az={current_az:.3f}, el={current_el:.3f}) that "
                f"crosses the Sun avoidance zone at {t_slew.iso}. No direct slew is "
                "safe; a two-leg detour may be available "
                "(fyst_trajectories.sun_models.find_sun_safe_detour).",
                goal_az=goal_az,
                goal_el=goal_el,
                current_az=current_az,
                current_el=current_el,
                candidates=[az for az, _ in safe],
                time_iso=t_slew.iso,
            )
        safe = path_safe

    # Minimum-slew selection (see Notes); the limit-margin tie-break is a coarse
    # nod to escapability, measured against the shifted span endpoints.
    def _limit_margin(shift: float) -> float:
        return min(span_lo + shift - az_limits.min, az_limits.max - (span_hi + shift))

    encoder_az, az_shift = min(safe, key=lambda c: (abs(c[0] - current_az), -_limit_margin(c[1])))
    return EncoderSolution(encoder_az, goal_el, az_shift)


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

       ``az1`` and ``az2`` must be expressed in one **coherent**
       cable-wrap frame, not merely both inside the telescope's
       ``[az_min, az_max] = [-180, 360]`` window: that window is 540
       degrees wide, so two in-window values can denote the same sky
       azimuth a full turn apart, and this function then reports a
       phantom unwind; mixing raw astropy ``[0, 360]`` azimuth with
       telescope-normalised azimuth has the same effect. Encoder
       azimuths form a coherent frame: the current encoder azimuth and
       the azimuth :func:`choose_encoder_solution` returns for a slew
       from it can be passed as they are. Otherwise place ``az2`` on the
       in-limits 360-degree representative nearest ``az1``. The function
       does not enforce or check this.

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
        Estimated slew time in seconds. No settle time is included; the
        caller adds its own.
    """
    az_limits = site.telescope_limits.azimuth
    el_limits = site.telescope_limits.elevation

    # Direct path respecting cable wrap limits.  Both positions should
    # already be within [az_min, az_max]; the direct distance is the
    # actual motor travel without wrapping around 360 deg.
    az_dist = abs(az2 - az1)
    az_time = _axis_slew_duration(az_dist, az_limits.max_velocity, az_limits.max_acceleration)
    el_time = _axis_slew_duration(
        abs(el2 - el1), el_limits.max_velocity, el_limits.max_acceleration
    )
    return max(az_time, el_time)
