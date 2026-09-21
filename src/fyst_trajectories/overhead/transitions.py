"""Pairwise slew transitions between observing poses.

A transition answers one question for an offline planner: can the
telescope go from where it is to where the next observation starts,
in which azimuth wrap, how long will it take, and if not, why not.
:func:`plan_transition` composes the dispatch-time wrap choice
(:func:`~fyst_trajectories.dispatch.choose_encoder_solution`), the
path-level Sun sweep (:func:`~fyst_trajectories.sun_models.make_slew_safe`)
and the kinematic slew estimate (:func:`~fyst_trajectories.overhead.estimate_slew_time`)
into a single value, :class:`Transition`, whose ``cause`` is a
:class:`DeferralReason`. A blocked transition is returned, never raised,
so a sequencer can defer or drop the target and move on.

The duration always comes from the kinematic estimate for the path that
is finally chosen, plus the settle time; the safety model adds no
penalty of its own. It still changes the duration indirectly, because it
decides which azimuth wrap survives and whether a two-leg detour is
taken.

:func:`plan_escape` is the companion for a telescope the Sun zone has
overtaken while it sat still: it finds a move out of the zone, under the
relaxed rule an escape needs (the path may start unsafe but never goes
deeper into the zone than it started, and it ends safe), so a planner
never leaves an idle telescope inside the zone.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, get_args

import numpy as np
from astropy.time import Time, TimeDelta

from ..coordinates import Coordinates
from ..dispatch import choose_encoder_solution
from ..exceptions import EncoderSolutionCause, EncoderSolutionError
from ..sun_models import find_sun_safe_detour, make_slew_safe, make_sun_safe
from .utils import estimate_slew_time

if TYPE_CHECKING:
    from ..dispatch import EncoderSolution, SlewSafePredicate, SunSafePredicate
    from ..site import Site

__all__ = [
    "DeferralReason",
    "Transition",
    "plan_escape",
    "plan_transition",
]

#: Elevation step of the escape candidate grid, in degrees.
_ESCAPE_EL_STEP_DEG = 5.0
#: How much deeper into the zone than its start an escape path may go, in
#: degrees, absorbing the sweep's sampling and the Sun's own motion. Depth
#: is measured against the model's own threshold when it exposes one and
#: against the raw Sun separation otherwise, so for a directional zone the
#: tolerance absorbs a step of the threshold table as well.
_ESCAPE_APPROACH_TOLERANCE_DEG = 0.5


class DeferralReason(str, enum.Enum):
    """Why a target was deferred, dropped or left idle, and why a slew was refused.

    One vocabulary serves three uses: a transition's ``cause``
    (``OK``, ``SUN_POINT``, ``SUN_PATH``, ``NO_WRAP``, ``LIMITS``, and
    ``NO_ESCAPE`` from :func:`plan_escape`), a planner's reason for
    deferring or dropping a target (those plus
    ``BELOW_BAND``, ``ABOVE_BAND``, ``CROSSING_TOO_SLOW``, ``MOON``,
    ``UNPLANNABLE``, ``WINDOW_CLOSED``), and the label on an idle stretch
    (``NOTHING_AVAILABLE``, ``SCRIPT_WAITING``, ``WAITING_FOR_PASS``,
    ``WINDOW_CLOSED`` on the stretch a schedule ends with, and
    ``NO_ESCAPE`` when the Sun zone holds the telescope where it sits).
    ``NO_WRAP`` and ``LIMITS`` are geometry and drop a target for the
    night; every other refusal passes with time and defers it. Members
    are strings, so they serialise as their values.
    """

    OK = "ok"
    BELOW_BAND = "below_band"
    ABOVE_BAND = "above_band"
    CROSSING_TOO_SLOW = "crossing_too_slow"
    SUN_POINT = "sun_point"
    SUN_PATH = "sun_path"
    MOON = "moon"
    NO_WRAP = "no_wrap"
    LIMITS = "limits"
    UNPLANNABLE = "unplannable"
    WINDOW_CLOSED = "window_closed"
    NOTHING_AVAILABLE = "nothing_available"
    SCRIPT_WAITING = "script_waiting"
    WAITING_FOR_PASS = "waiting_for_pass"
    NO_ESCAPE = "no_escape"

    def __str__(self) -> str:
        return self.value


# The five stages at which choose_encoder_solution refuses, mapped onto the
# transition vocabulary. Geometry refusals drop a target for the night; the
# Sun refusals defer it. A test asserts the mapping covers every cause.
_CAUSE_TO_REASON: dict[str, DeferralReason] = {
    "goal_elevation": DeferralReason.LIMITS,
    "no_image": DeferralReason.NO_WRAP,
    "span_unreachable": DeferralReason.NO_WRAP,
    "sun_blocked": DeferralReason.SUN_POINT,
    "path_blocked": DeferralReason.SUN_PATH,
}
assert set(_CAUSE_TO_REASON) == set(get_args(EncoderSolutionCause))

# The encoder azimuth window is 540 degrees wide, so two azimuths in one
# coherent frame are never farther apart than that. A larger gap means the
# caller mixed frames (a raw sky azimuth against an encoder azimuth, or a
# pose from another wrap convention), which estimate_slew_time cannot detect.
_FRAME_COHERENCE_LIMIT_DEG = 540.0


@dataclass(frozen=True)
class Transition:
    """One slew between two poses, with its verdict.

    Parameters
    ----------
    az_from, el_from : float
        The starting encoder position in degrees.
    az_to, el_to : float
        The commanded encoder position in degrees, in the chosen azimuth
        wrap. For a blocked transition these are the requested sky
        azimuth and elevation, since no wrap was chosen; for the
        ``NO_ESCAPE`` transition of :func:`plan_escape`, which has no
        requested target, they are the current pose the zone holds the
        telescope at.
    t_start : Time
        When the slew starts.
    duration : float
        Slew time in seconds in the chosen wrap plus the settle time,
        from the kinematic estimate; ``0.0`` for a blocked transition.
    cause : DeferralReason
        ``OK`` for a commandable transition, otherwise why it was refused
        (``SUN_POINT``, ``SUN_PATH``, ``NO_WRAP`` or ``LIMITS``; ``NO_ESCAPE``
        from :func:`plan_escape` when the zone holds the telescope).
    detour_via : tuple of float, optional
        The intermediate ``(az, el)`` of a two-leg detour in degrees, when
        the direct path was blocked and a detour was found. ``None`` for a
        direct or blocked transition.
    az_shift : float, optional
        The multiple of 360 degrees taking the requested goal azimuth onto
        ``az_to``, carried from the encoder solution that chose the wrap.
        A caller commanding a trajectory into this pose applies the same
        shift to it (see
        :func:`~fyst_trajectories.patterns.rewrap_trajectory_azimuth`).
        ``0.0`` for a blocked transition and for an escape, neither of
        which has a requested goal to shift from.
    """

    az_from: float
    el_from: float
    az_to: float
    el_to: float
    t_start: Time
    duration: float
    cause: DeferralReason
    detour_via: tuple[float, float] | None = None
    az_shift: float = 0.0

    @property
    def safe(self) -> bool:
        """Whether the transition can be commanded (``cause`` is ``OK``)."""
        return self.cause is DeferralReason.OK

    @property
    def path(self) -> str:
        """``"direct"``, ``"detour"`` or ``"blocked"``, derived from the fields."""
        if not self.safe:
            return "blocked"
        return "detour" if self.detour_via is not None else "direct"

    @property
    def arrival(self) -> Time:
        """``t_start`` plus ``duration``."""
        return self.t_start + TimeDelta(self.duration, format="sec")


def _default_slew_safe(sun_safe: SunSafePredicate, site: Site) -> SlewSafePredicate:
    """Sweep ``sun_safe`` along the direct path under the site's own axis limits.

    Built from ``site.telescope_limits`` rather than the library-wide
    constants so the verdict and :func:`estimate_slew_time`, which reads
    the same limits, describe one telescope.
    """
    limits = site.telescope_limits
    return make_slew_safe(
        sun_safe,
        az_speed=limits.azimuth.max_velocity,
        az_accel=limits.azimuth.max_acceleration,
        el_speed=limits.elevation.max_velocity,
        el_accel=limits.elevation.max_acceleration,
    )


def plan_transition(
    current_az: float,
    current_el: float,
    goal_az: float,
    goal_el: float,
    time: Time,
    site: Site,
    *,
    sun_safe: SunSafePredicate | None = None,
    slew_safe: SlewSafePredicate | None = None,
    goal_az_span: tuple[float, float] | None = None,
    settle_time: float = 0.0,
    allow_detour: bool = False,
    hold: float = 0.0,
) -> Transition:
    """Plan the slew from the current pose to a goal, in a Sun-safe wrap.

    Chooses the encoder azimuth wrap with
    :func:`~fyst_trajectories.dispatch.choose_encoder_solution` (point-safe
    at ``time``, direct path clear under ``slew_safe``, the whole
    ``goal_az_span`` inside the azimuth limits), then prices the slew with
    :func:`~fyst_trajectories.overhead.estimate_slew_time` in that wrap.
    A refused goal becomes a :class:`Transition` whose ``cause`` names the
    refusal; nothing is raised for infeasibility.

    With ``hold`` the goal must also stay point-safe for that long after
    arrival, which is the question a planner needs answered before it
    commands a pose it will sit at: without it a slew is accepted on an
    instant's verdict and abandoned minutes later, and the schedule
    records a move that bought nothing.

    Parameters
    ----------
    current_az, current_el : float
        Current encoder position in degrees, in the telescope's azimuth
        frame.
    goal_az, goal_el : float
        Goal sky azimuth and elevation in degrees (any wrap; the encoder
        image is chosen here).
    time : Time
        Scalar slew start time.
    site : Site
        Telescope site; supplies the axis limits for the wrap choice, the
        kinematic estimate and the default safety models.
    sun_safe : SunSafePredicate, optional
        Point-level Sun predicate for the wrap choice. ``None`` (default)
        builds the scalar model from the site's avoidance radii, so a site
        with avoidance disabled accepts every pose.
    slew_safe : SlewSafePredicate, optional
        Path-level predicate for the direct slew. ``None`` (default) sweeps
        ``sun_safe`` along the path under the site's own axis limits
        (:func:`~fyst_trajectories.sun_models.make_slew_safe`). A detour
        search needs a predicate built that way, since it must expose
        ``evaluate``.
    goal_az_span : tuple of float, optional
        Azimuth span the observation will sweep, ``(min, max)`` in the same
        frame as ``goal_az``; every wrap must keep the whole span within
        the azimuth limits.
    settle_time : float, optional
        Seconds added to the kinematic estimate after arrival. Default
        ``0.0``; a planner passes its overhead model's settle time.
    allow_detour : bool, optional
        When the direct path is blocked, try a two-leg detour with
        :func:`~fyst_trajectories.sun_models.find_sun_safe_detour` from each
        point-safe wrap, nearest first. Default ``False``: a blocked path
        is reported as ``SUN_PATH`` and the caller decides.
    hold : float, optional
        Seconds the goal must stay point-safe after arrival. Default
        ``0.0``, the arrival instant alone. A positive value re-runs the
        wrap choice over ``(time, arrival, arrival + hold)``, using the
        all-times gate
        :func:`~fyst_trajectories.dispatch.choose_encoder_solution` already
        applies to a dwell grid, and a goal that does not survive it is
        refused as ``SUN_POINT``. The window is anchored on the arrival
        the instantaneous choice priced, so the hold is a sampled gate,
        not a proof.

    Returns
    -------
    Transition
        The planned slew. ``cause`` is ``OK`` and ``duration`` is the
        estimate plus ``settle_time`` when commandable; otherwise
        ``cause`` is ``LIMITS`` (goal elevation outside the limits),
        ``NO_WRAP`` (no wrap fits the goal or its span), ``SUN_POINT``
        (every wrap is inside the Sun zone) or ``SUN_PATH`` (every
        point-safe wrap has a blocked direct path and no detour was
        allowed or found), with ``duration`` ``0.0``.

    Raises
    ------
    ValueError
        If ``time`` is not scalar, ``settle_time`` or ``hold`` is negative,
        the span is malformed, the chosen encoder azimuth lies more than
        540 degrees from ``current_az`` (the two are not in one coherent
        frame), or a detour is requested with a ``slew_safe`` that cannot
        evaluate a path.

    Examples
    --------
    With Sun avoidance disabled the nearer wrap of sky azimuth 200 from
    encoder azimuth 190 is 200, ten degrees away:

    >>> from astropy.time import Time
    >>> from fyst_trajectories import get_fyst_site
    >>> from fyst_trajectories.overhead import plan_transition
    >>> site = get_fyst_site(sun_avoidance_enabled=False)
    >>> t = Time("2026-03-15T12:00:00", scale="utc")
    >>> tr = plan_transition(190.0, 45.0, 200.0, 45.0, t, site, settle_time=5.0)
    >>> tr.path, tr.az_to, round(tr.duration, 1)
    ('direct', 200.0, 10.3)
    """
    if not time.isscalar:
        raise ValueError("plan_transition takes a scalar start time")
    if settle_time < 0.0:
        raise ValueError(f"settle_time must be non-negative, got {settle_time}")
    if hold < 0.0:
        raise ValueError(f"hold must be non-negative, got {hold}")
    if sun_safe is None:
        sun_safe = make_sun_safe("scalar", site=site)
    if slew_safe is None:
        slew_safe = _default_slew_safe(sun_safe, site)

    def _solve(obstime: Time) -> EncoderSolution:
        return choose_encoder_solution(
            current_az,
            current_el,
            goal_az,
            goal_el,
            obstime,
            site,
            sun_safe=sun_safe,
            slew_safe=slew_safe,
            goal_az_span=goal_az_span,
        )

    try:
        solution = _solve(time)
        az_to, el_to = solution
        if hold > 0.0:
            # The dwell window needs the arrival, and the arrival needs a
            # wrap, so the instantaneous solve above prices the leg and this
            # one re-runs the same gate over it. The path check is unchanged:
            # the slew still starts at the grid's first element.
            duration = estimate_slew_time(current_az, current_el, az_to, el_to, site) + settle_time
            grid = time + TimeDelta([0.0, duration, duration + hold], format="sec")
            solution = _solve(grid)
            az_to, el_to = solution
    except EncoderSolutionError as exc:
        cause = _CAUSE_TO_REASON[exc.cause]
        if cause is DeferralReason.SUN_PATH and allow_detour:
            detour = _plan_detour(
                current_az,
                current_el,
                exc.candidates,
                goal_az,
                goal_el,
                time,
                site,
                slew_safe,
                settle_time,
            )
            if detour is not None:
                return detour
        return Transition(
            az_from=current_az,
            el_from=current_el,
            az_to=goal_az,
            el_to=goal_el,
            t_start=time,
            duration=0.0,
            cause=cause,
        )

    if abs(az_to - current_az) > _FRAME_COHERENCE_LIMIT_DEG:
        raise ValueError(
            f"current_az={current_az:.3f} and the chosen encoder azimuth {az_to:.3f} are "
            f"more than {_FRAME_COHERENCE_LIMIT_DEG:.0f} deg apart, so they are not in one "
            "coherent azimuth frame; pass the current pose in the telescope's encoder frame."
        )
    duration = estimate_slew_time(current_az, current_el, az_to, el_to, site) + settle_time
    return Transition(
        az_from=current_az,
        el_from=current_el,
        az_to=az_to,
        el_to=el_to,
        t_start=time,
        duration=duration,
        cause=DeferralReason.OK,
        az_shift=solution.az_shift,
    )


def _plan_detour(
    current_az: float,
    current_el: float,
    candidates: tuple[float, ...],
    goal_az: float,
    goal_el: float,
    time: Time,
    site: Site,
    slew_safe: SlewSafePredicate,
    settle_time: float,
) -> Transition | None:
    """Try a two-leg detour to each point-safe wrap, nearest first."""
    for az_to in sorted(candidates, key=lambda az: abs(az - current_az)):
        via = find_sun_safe_detour(
            current_az, current_el, az_to, goal_el, time, slew_safe, site=site
        )
        if via is None:
            continue
        az_mid, el_mid = via
        duration = (
            estimate_slew_time(current_az, current_el, az_mid, el_mid, site)
            + estimate_slew_time(az_mid, el_mid, az_to, goal_el, site)
            + settle_time
        )
        return Transition(
            az_from=current_az,
            el_from=current_el,
            az_to=az_to,
            el_to=goal_el,
            t_start=time,
            duration=duration,
            cause=DeferralReason.OK,
            detour_via=(float(az_mid), float(el_mid)),
            az_shift=az_to - goal_az,
        )
    return None


def plan_escape(
    current_az: float,
    current_el: float,
    time: Time,
    site: Site,
    *,
    sun_safe: SunSafePredicate | None = None,
    slew_safe: SlewSafePredicate | None = None,
    settle_time: float = 0.0,
    el_floor: float | None = None,
) -> Transition | None:
    """Plan the move out of the Sun zone for a pose the zone has overtaken.

    A telescope that sits still while the Sun approaches ends up inside
    the zone with no commandable transition, since every path from it
    starts unsafe. The mount's own sun-avoidance system performs this
    move at the telescope; an offline planner models it so an idle
    telescope is never left inside the zone.

    Candidate azimuths are the current one, the ends of the azimuth
    window, a half-turn grid across it, and every encoder image of the
    anti-solar azimuth
    (the direction that maximises the separation at a given elevation,
    which a directional zone reaching 90 degrees can require). They are
    tried at the current elevation first and then stepping down toward
    ``el_floor`` in 5 degree steps (more sky opens away from the zenith);
    a floor at or above the current elevation leaves the current
    elevation as the only candidate, since the search never climbs.
    The first elevation with any admissible path wins, and among its
    candidates the pose with the largest safety margin on arrival, ties
    going to the shorter move. A path is admissible under the rule an
    escape needs: it may start unsafe, it never goes more than half a
    degree deeper into the zone than it started, and its end pose is
    safe at arrival.

    Depth and margin are measured against the point model's own
    requirement when it exposes the optional ``threshold`` extension of
    the :class:`~fyst_trajectories.dispatch.SunSafePredicate` contract
    (separation minus the required separation), and against the raw Sun
    separation otherwise. The two agree for a model whose requirement is
    one radius; they differ for a directional zone, where a route can
    hold its separation while crossing into a sector that demands tens of
    degrees more, and the raw rule would rate that route unchanged.

    Parameters
    ----------
    current_az, current_el : float
        Current encoder position in degrees.
    time : Time
        Scalar time the move starts.
    site : Site
        Telescope site; supplies the axis limits, the kinematic estimate
        and the default safety models.
    sun_safe : SunSafePredicate, optional
        Point-level Sun predicate; ``None`` (default) builds the scalar
        model from the site's avoidance radii.
    slew_safe : SlewSafePredicate, optional
        Path model exposing ``evaluate`` (the site-built default does;
        see :func:`~fyst_trajectories.sun_models.make_slew_safe`). Its
        sampled path supplies the separations the admissibility rule
        reads. Only consulted for a pose that is actually inside the
        zone, so a bare predicate still answers the safe-pose case.
    settle_time : float, optional
        Seconds added to the kinematic estimate after arrival. Default
        ``0.0``.
    el_floor : float, optional
        Lowest elevation the escape may descend to, in degrees. Default
        the site's elevation limit. A floor at or above the current
        elevation does not raise the telescope: the search keeps the
        current elevation, since a floor bounds the descent rather than
        commanding a climb.

    Returns
    -------
    Transition or None
        ``None`` when the current pose is safe (nothing to escape from).
        Otherwise the escape as a direct ``OK`` transition, or, when no
        candidate path is admissible (the zone holds the telescope), a
        transition whose ``cause`` is ``NO_ESCAPE`` with ``duration``
        ``0.0`` and the current pose as its target.

    Raises
    ------
    ValueError
        If ``time`` is not scalar, ``settle_time`` is negative, ``el_floor``
        is outside the elevation limits, or ``slew_safe`` cannot
        ``evaluate`` a path.
    """
    if not time.isscalar:
        raise ValueError("plan_escape takes a scalar start time")
    if settle_time < 0.0:
        raise ValueError(f"settle_time must be non-negative, got {settle_time}")
    limits = site.telescope_limits
    floor = limits.elevation.min if el_floor is None else float(el_floor)
    if not limits.elevation.is_in_range(floor):
        raise ValueError(
            f"el_floor {floor} is outside the elevation limits "
            f"[{limits.elevation.min}, {limits.elevation.max}]"
        )
    if sun_safe is None:
        sun_safe = make_sun_safe("scalar", site=site)
    # The pose question is answered before the path model is touched, so a
    # caller holding a bare predicate can still ask it (only an escape
    # itself needs the sampled path).
    if sun_safe(current_az, current_el, time):
        return None
    if slew_safe is None:
        slew_safe = _default_slew_safe(sun_safe, site)
    if not hasattr(slew_safe, "evaluate"):
        raise ValueError(
            "slew_safe must expose evaluate() (build it with make_slew_safe) so the "
            "escape rule can read the path's Sun separations."
        )

    coords = Coordinates(site)
    sun_az, sun_el = coords.get_sun_altaz(time)
    start_separation = float(coords.angular_separation(current_az, current_el, sun_az, sun_el))
    # A model exposing ``threshold`` states how much separation it wants at
    # each pose, so depth is measured against that requirement; without it
    # the raw separation is the only ordering available. Subtracting zero
    # in the second case keeps one code path.
    threshold = getattr(sun_safe, "threshold", None)
    start_margin = start_separation
    if threshold is not None:
        start_margin -= float(np.atleast_1d(threshold(current_az, current_el, time))[0])

    az_limits = limits.azimuth
    # Half-turn steps across the azimuth window, anchored on its lower
    # limit so the grid never empties for a window narrower than a half
    # turn, plus the window's own ends and the current pose.
    n_half = int(np.floor((az_limits.max - az_limits.min) / 180.0))
    az_candidates = [float(current_az), az_limits.min, az_limits.max]
    az_candidates += [az_limits.min + 180.0 * k for k in range(n_half + 1)]
    anti_solar = (float(sun_az) + 180.0) % 360.0
    turns = range(int(np.floor(az_limits.min / 360.0)) - 1, int(np.ceil(az_limits.max / 360.0)) + 1)
    az_candidates += [anti_solar + 360.0 * k for k in turns]
    az_candidates = sorted({az for az in az_candidates if az_limits.is_in_range(az)})
    # A floor above the current elevation bounds nothing: the search
    # descends, so it starts where the telescope is.
    el_start = min(float(current_el), limits.elevation.max)
    n_steps = max(0, int(np.floor((el_start - floor) / _ESCAPE_EL_STEP_DEG)))
    el_candidates = [el_start - _ESCAPE_EL_STEP_DEG * i for i in range(n_steps + 1)]
    if el_candidates[-1] > floor:
        el_candidates.append(floor)

    for el_to in el_candidates:
        best: tuple[tuple[float, float], Transition] | None = None
        for az_to in az_candidates:
            if az_to == float(current_az) and el_to == float(current_el):
                continue
            _, az_path, el_path, times = slew_safe.evaluate(
                current_az, current_el, az_to, el_to, time
            )
            path_sun_az, path_sun_el = coords.get_sun_altaz(times)
            margins = np.atleast_1d(
                coords.angular_separation(az_path, el_path, path_sun_az, path_sun_el)
            )
            if threshold is not None:
                margins = margins - np.atleast_1d(threshold(az_path, el_path, times))
            if float(margins.min()) < start_margin - _ESCAPE_APPROACH_TOLERANCE_DEG:
                continue
            if not sun_safe(az_to, el_to, times[-1]):
                continue
            duration = estimate_slew_time(current_az, current_el, az_to, el_to, site) + settle_time
            key = (float(margins[-1]), -duration)
            if best is None or key > best[0]:
                best = (
                    key,
                    Transition(
                        az_from=float(current_az),
                        el_from=float(current_el),
                        az_to=az_to,
                        el_to=el_to,
                        t_start=time,
                        duration=duration,
                        cause=DeferralReason.OK,
                    ),
                )
        if best is not None:
            return best[1]
    return Transition(
        az_from=float(current_az),
        el_from=float(current_el),
        az_to=float(current_az),
        el_to=float(current_el),
        t_start=time,
        duration=0.0,
        cause=DeferralReason.NO_ESCAPE,
    )
