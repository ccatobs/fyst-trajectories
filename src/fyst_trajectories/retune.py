"""In-scan retune injection.

Stamps the in-scan retune gaps of the detector readout into a
:class:`~fyst_trajectories.trajectory.Trajectory`'s ``scan_flag``, at a
uniform cadence or from an explicit list of
:class:`~fyst_trajectories.trajectory.RetuneEvent` instances, and draws
such event lists from caller-supplied samplers.
"""

import dataclasses
import math
import warnings
from collections.abc import Callable, Sequence

import numpy as np

from .exceptions import PointingWarning
from .trajectory import (
    SCAN_FLAG_RETUNE,
    SCAN_FLAG_SCIENCE,
    SCAN_FLAG_TURNAROUND,
    RetuneEvent,
    Trajectory,
)

#: Default wall-clock duration of a single in-scan retune gap (seconds):
#: the per-module tone correction that :func:`inject_retune` stamps into
#: a trajectory after each ``retune_interval`` of observing. This is a
#: different operation from the whole-array retune reserved between scan blocks
#: (:class:`~fyst_trajectories.overhead.OverheadModel`'s ``retune_duration``,
#: minutes of probe-tone placement plus a target sweep). The two values
#: are independent and are deliberately not kept in sync; re-baseline
#: each with its own instrument-team measurement.
DEFAULT_RETUNE_DURATION_SEC: float = 5.0
#: Default seconds between in-scan retunes in :func:`inject_retune`.
_DEFAULT_RETUNE_INTERVAL_SEC: float = 300.0


# Tolerance for treating consecutive events as non-overlapping. A
# positive epsilon makes the overlap check more permissive, not
# stricter: an overlap of up to this many seconds is read as the
# floating-point residue of two events that touch, and only a longer
# one is refused.
_EVENT_OVERLAP_EPS: float = 1e-9


def _zero_velocity_guard(trajectory: Trajectory, prefer_turnarounds: bool) -> bool:
    """Return possibly-adjusted ``prefer_turnarounds`` flag.

    Shared guard used by both the uniform-cadence and event-list code
    paths. If ``prefer_turnarounds`` is True but the trajectory has
    identically zero az/el velocities, warn and fall back to time-based
    placement (``prefer_turnarounds=False``). Otherwise returns the flag
    unchanged.
    """
    if not prefer_turnarounds:
        return prefer_turnarounds
    if np.all(trajectory.az_vel == 0.0) and np.all(trajectory.el_vel == 0.0):
        warnings.warn(
            "inject_retune called with prefer_turnarounds=True but the "
            "trajectory has all-zero velocities; turnaround detection "
            "requires real velocities. Falling back to time-based retune "
            "placement.",
            PointingWarning,
            stacklevel=3,
        )
        return False
    return prefer_turnarounds


def _collect_turnaround_starts(scan_flag: np.ndarray, times: np.ndarray) -> list[float]:
    """Return the start times of each contiguous turnaround region.

    A turnaround start is the first sample in a run of consecutive
    ``SCAN_FLAG_TURNAROUND`` samples. Returned times are trajectory-
    absolute (i.e. they come directly from ``times``, including any
    ``times[0]`` offset).
    """
    is_turnaround = np.asarray(scan_flag) == SCAN_FLAG_TURNAROUND
    opens = is_turnaround & ~np.concatenate(([False], is_turnaround[:-1]))
    return [float(t) for t in np.asarray(times)[opens]]


def _snap_to_turnaround(
    due_time: float,
    turnaround_starts: Sequence[float],
    turnaround_window: float,
) -> float:
    """Return the nearest turnaround start within ``turnaround_window``.

    If no turnaround sits within the window, returns ``due_time``
    unchanged.
    """
    best_ta = None
    best_dist = turnaround_window + 1.0
    for ta_start in turnaround_starts:
        dist = abs(ta_start - due_time)
        if dist <= turnaround_window and dist < best_dist:
            best_ta = ta_start
            best_dist = dist
    if best_ta is not None:
        return best_ta
    return due_time


def _inject_retune_uniform(
    trajectory: Trajectory,
    retune_interval: float,
    retune_duration: float,
    prefer_turnarounds: bool,
    turnaround_window: float,
    module_index: int,
    n_modules: int,
) -> Trajectory:
    """Uniform-cadence retune injection.

    Callers are responsible for having already validated scalar inputs and
    run the zero-velocity guard.

    As a side-effect the helper also records every retune it scheduled as
    a :class:`~fyst_trajectories.trajectory.RetuneEvent` on the returned
    trajectory's ``retune_events`` field.
    """
    times = trajectory.times

    if trajectory.scan_flag is None:
        scan_flag = np.full(len(times), SCAN_FLAG_SCIENCE, dtype=np.int8)
    else:
        scan_flag = trajectory.scan_flag.copy()

    duration = float(times[-1] - times[0])
    if duration < retune_interval:
        return dataclasses.replace(trajectory, scan_flag=scan_flag, retune_events=())

    turnaround_starts: list[float] = []
    if prefer_turnarounds:
        turnaround_starts = _collect_turnaround_starts(scan_flag, times)

    # For staggered retune, offset the first retune time by a fraction
    # of the retune interval so different modules retune at different times.
    # ``next_due_anchor`` is the synthetic anchor whose increments by
    # ``retune_interval`` produce the next ``due_time``; it is *not* the
    # wall-clock time of the most recent retune (those two coincide only
    # when ``prefer_turnarounds`` does not snap).
    stagger_offset = module_index * retune_interval / n_modules
    next_due_anchor = float(times[0]) + stagger_offset

    generated_events: list[RetuneEvent] = []
    t0 = float(times[0])

    while True:
        due_time = next_due_anchor + retune_interval
        if due_time > float(times[-1]):
            break

        retune_start = due_time
        if prefer_turnarounds and turnaround_starts:
            retune_start = _snap_to_turnaround(due_time, turnaround_starts, turnaround_window)

        retune_end = retune_start + retune_duration

        mask = (times >= retune_start) & (times < retune_end) & (scan_flag == SCAN_FLAG_SCIENCE)
        scan_flag[mask] = SCAN_FLAG_RETUNE

        # Record the event using trajectory-relative t_start (subtracting
        # ``times[0]``) so the provenance matches the event-list path's
        # convention. ``retune_duration`` is used verbatim; any clipping
        # at ``times[-1]`` is an application detail that the scan_flag
        # array already captures.
        generated_events.append(RetuneEvent(t_start=retune_start - t0, duration=retune_duration))

        # Use max(retune_end, due_time) to prevent backward drift when
        # prefer_turnarounds snaps to a turnaround before the due time.
        next_due_anchor = max(retune_end, due_time)

    return dataclasses.replace(
        trajectory, scan_flag=scan_flag, retune_events=tuple(generated_events)
    )


def _inject_retune_events(
    trajectory: Trajectory,
    events: Sequence[RetuneEvent],
    prefer_turnarounds: bool,
    turnaround_window: float,
) -> Trajectory:
    """Event-list retune injection.

    Validates, sorts, clips, and applies a caller-supplied list of
    :class:`~fyst_trajectories.trajectory.RetuneEvent` instances, setting both ``scan_flag`` (the
    per-sample array) and ``retune_events`` (the event-level provenance)
    on the returned trajectory. ``trajectory.metadata`` is left verbatim:
    pattern metadata and retune provenance are distinct concerns and
    live at different fields on :class:`Trajectory`.

    Records the caller's own sorted request; see :func:`inject_retune` for
    what each mode records and for the user-facing contract.
    """
    for idx, event in enumerate(events):
        if not isinstance(event, RetuneEvent):
            raise TypeError(
                f"retune_events[{idx}] is a {type(event).__name__}, not a RetuneEvent; "
                "pass RetuneEvent(t_start=..., duration=...) instances."
            )

    sorted_events = tuple(sorted(events, key=lambda e: e.t_start))

    for i in range(1, len(sorted_events)):
        a = sorted_events[i - 1]
        b = sorted_events[i]
        if a.t_start + a.duration > b.t_start + _EVENT_OVERLAP_EPS:
            raise ValueError(
                f"Overlapping retune events at sorted indices {i - 1} "
                f"(t_start={a.t_start}, duration={a.duration}) and {i} "
                f"(t_start={b.t_start}). Events must not overlap when "
                "passed to a single inject_retune call; call inject_retune "
                "once per module with its own event list to stagger."
            )

    times = trajectory.times
    t0 = float(times[0])
    t_end = float(times[-1])

    if trajectory.scan_flag is None:
        scan_flag = np.full(len(times), SCAN_FLAG_SCIENCE, dtype=np.int8)
    else:
        scan_flag = trajectory.scan_flag.copy()

    # Partition into in-bounds / out-of-bounds and warn once. Indices are
    # measured in the *sorted* list, not the caller's input order, and that
    # is surfaced in the warning message so callers who pass an unsorted
    # list can still locate the offending events.
    in_bounds: list[RetuneEvent] = []
    skipped: list[tuple[int, float]] = []
    for idx, event in enumerate(sorted_events):
        # Events are trajectory-relative: add t0 to compare against the
        # raw ``times`` domain.
        if event.t_start + t0 >= t_end:
            skipped.append((idx, event.t_start))
        else:
            in_bounds.append(event)

    if skipped:
        skipped_str = ", ".join(f"sorted_index={idx} at t_start={t:.1f}" for idx, t in skipped)
        warnings.warn(
            "inject_retune: skipping retune events at or past trajectory end: "
            f"{skipped_str}. Indices refer to the sorted event list, not "
            f"the caller-supplied input order. Trajectory spans "
            f"[0, {t_end - t0}] seconds in trajectory-relative time.",
            PointingWarning,
            stacklevel=3,
        )

    turnaround_starts: list[float] = []
    if prefer_turnarounds:
        turnaround_starts = _collect_turnaround_starts(scan_flag, times)

    # Trajectory-relative -> absolute-in-times-domain offset. The uniform
    # path works in the raw ``times`` domain (it seeds ``next_due_anchor``
    # with ``float(times[0])``), so we match that convention here by adding
    # ``t0`` to the caller-supplied relative time.
    starts = [event.t_start + t0 for event in in_bounds]
    if prefer_turnarounds and turnaround_starts:
        starts = [_snap_to_turnaround(s, turnaround_starts, turnaround_window) for s in starts]
        # Snapping can pull two events that did not overlap as requested onto
        # the same turnaround, where the second would paint samples the first
        # already claimed and vanish. The request-level check above runs
        # before snapping and cannot see it, so re-check what will be applied.
        applied = sorted(zip(starts, (e.duration for e in in_bounds)))
        for (s_a, d_a), (s_b, _) in zip(applied, applied[1:]):
            if s_a + d_a > s_b + _EVENT_OVERLAP_EPS:
                raise ValueError(
                    f"Retune events overlap after snapping to turnarounds: an event "
                    f"placed at t_start={s_a - t0} (duration={d_a}) runs into one placed "
                    f"at t_start={s_b - t0}. Widen turnaround_window, space the events "
                    "further apart, or pass prefer_turnarounds=False."
                )

    for event, start in zip(in_bounds, starts):
        end = min(start + event.duration, t_end)
        # When the event is clipped to the trajectory end, include the final
        # sample (``times < end`` would drop the science sample at exactly
        # ``t_end``); otherwise keep the half-open upper bound.
        upper = (times <= end) if end >= t_end else (times < end)
        mask = (times >= start) & upper & (scan_flag == SCAN_FLAG_SCIENCE)
        scan_flag[mask] = SCAN_FLAG_RETUNE

    return dataclasses.replace(trajectory, scan_flag=scan_flag, retune_events=sorted_events)


def inject_retune(
    trajectory: Trajectory,
    retune_interval: float = _DEFAULT_RETUNE_INTERVAL_SEC,
    retune_duration: float = DEFAULT_RETUNE_DURATION_SEC,
    prefer_turnarounds: bool = False,
    turnaround_window: float = 5.0,
    module_index: int = 0,
    n_modules: int = 1,
    *,
    retune_events: Sequence[RetuneEvent] | None = None,
) -> Trajectory:
    """Inject retune flags into a trajectory.

    Two modes are supported. In **uniform-cadence mode** (the default, when
    ``retune_events`` is ``None``), each retune is scheduled
    ``retune_interval`` seconds after the previous one ends unless turnaround
    snapping (below) applies; optional per-module staggering is controlled by
    ``module_index`` and ``n_modules``. In **event-list mode** (when
    ``retune_events`` is supplied), the caller provides an explicit sequence
    of :class:`~fyst_trajectories.trajectory.RetuneEvent` instances; the
    uniform-cadence / per-module-stagger kwargs are not used.

    Uniform-cadence mode walks forward through the trajectory timeline: the
    first retune is due ``retune_interval`` seconds after the trajectory
    start and each later one ``retune_interval`` seconds after the previous
    one ends, so unsnapped retunes start ``retune_interval + retune_duration``
    apart (the first three at 300, 605 and 910 s with the defaults). If
    ``prefer_turnarounds`` is True and a turnaround region exists within
    ``turnaround_window`` seconds of the due time, the retune is snapped to
    start at the turnaround (zero additional dead time), and the next one is
    due ``retune_interval`` seconds after the later of the snapped retune's
    end and its due time. Otherwise the retune is placed at the time-based
    position.

    The default is ``prefer_turnarounds=False`` (time-based placement),
    which produces uniform coverage. Set to True to snap retunes to nearby
    turnarounds, which saves only a sliver of science time (~0.04% in the
    configurations measured) but concentrates gaps at turnaround
    positions, creating persistent coverage non-uniformity.

    Only samples with ``SCAN_FLAG_SCIENCE`` are overwritten with
    ``SCAN_FLAG_RETUNE``; turnaround flags are never modified.

    **Per-module staggered retune** (UNCONFIRMED, needs FYST team
    verification): Prime-Cam has 7 independent readout modules. If modules
    can retune independently, setting ``n_modules > 1`` offsets the first
    retune by ``module_index * retune_interval / n_modules``, so only one
    module is retuning at a time, as long as ``retune_duration`` is shorter
    than ``retune_interval / n_modules`` (it is, at the defaults; the
    library enforces only ``retune_duration < retune_interval``). Each
    module still pays ``retune_duration / (retune_interval + retune_duration)``
    of its own time (about 1.6% at the defaults, 5 s in every 305 s) whether
    or not retunes are staggered; what staggering changes is where the loss
    lands. Simultaneous retunes leave whole-array gaps in coverage, while a
    7-way stagger under that condition keeps at least six modules on sky
    throughout, provided the non-retuning modules keep observing through
    each module's retune (an instrument-team premise this library does not
    model). Set ``n_modules=1`` (the default) to disable staggering and retune
    all modules simultaneously. Per-module staggering in **event-list mode** is
    handled by composition: call ``inject_retune`` once per module with its own
    event list.

    .. note::

       ``retune_interval``, ``retune_duration``, and ``n_modules`` are
       instrument-team inputs, not astronomer-tunable knobs. Default
       values are commissioning-era placeholders; obtain actual values
       from the Prime-Cam instrument team.

    Parameters
    ----------
    trajectory : Trajectory
        Input trajectory with scan_flag array.
    retune_interval : float, optional
        Seconds from the end of one retune to the time the next is due
        (from the trajectory start, plus any stagger offset, for the
        first; a snapped retune counts from the later of its end and its
        due time). Default 300 s (5 min). Ignored when ``retune_events``
        is supplied.
    retune_duration : float, optional
        Duration in seconds of each retune event. Default
        :data:`DEFAULT_RETUNE_DURATION_SEC` (5 s). Must be shorter than
        ``retune_interval``. Ignored when ``retune_events`` is supplied.
    prefer_turnarounds : bool, optional
        If True, snap retunes to nearby turnarounds when possible.
        Default is False (time-based placement for uniform coverage).
        Applies to both modes.
    turnaround_window : float, optional
        Maximum seconds from due time to search for a turnaround start.
        Default 5 s. Applies to both modes.
    module_index : int, optional
        Index of this module (0-based) for staggered retune scheduling.
        Default is 0. Only meaningful when ``n_modules > 1``. Must be
        0 in event-list mode (the caller handles per-module staggering
        by composition).
    n_modules : int, optional
        Total number of independent modules. Default is 1 (no staggering,
        all modules retune simultaneously). Set to 7
        for Prime-Cam staggered retune. Must be 1 in event-list mode.
    retune_events : sequence of RetuneEvent, optional, keyword-only
        If supplied, enables event-list mode. Each event's ``t_start``
        is measured in seconds from the trajectory start
        (``trajectory.times[0]``). Events are validated, sorted, and
        applied in chronological order. The validated, sorted tuple is
        set on the returned trajectory's
        :attr:`~fyst_trajectories.trajectory.Trajectory.retune_events`
        field. The uniform-cadence path also populates this field with
        the events it generated, so introspection works the same
        regardless of which mode produced the retunes.
        The two modes record different things, deliberately: event-list
        mode records the request (an event dropped for starting at or
        after the trajectory end stays in the tuple, one clipped at the
        end keeps its full ``duration``, and a ``prefer_turnarounds``
        snap does not move a recorded ``t_start``), while uniform-cadence
        mode has no request and records the placements it made.
        ``scan_flag`` says what was applied in both.

    Returns
    -------
    Trajectory
        New trajectory with retune samples flagged.

    Raises
    ------
    ValueError
        If ``turnaround_window`` is negative or not finite, in either mode
        and whether or not ``prefer_turnarounds`` is set. If
        ``retune_interval`` or ``retune_duration`` is not a finite value
        above zero, in uniform-cadence mode. If ``retune_duration`` is not
        shorter than ``retune_interval``, in uniform-cadence mode: under
        time-based placement such a gap, at least as long as the observing
        interval between gaps, would flag
        ``retune_duration / (retune_interval + retune_duration)``, at least
        half, of a long trajectory's science samples, so such a pair almost
        certainly means the two arguments are swapped. If
        ``module_index`` is negative or >= ``n_modules``, or if
        ``n_modules`` is less than 1 (uniform-cadence mode). If
        ``retune_events`` is supplied with ``module_index != 0`` or
        ``n_modules != 1`` (per-module composition is the caller's
        responsibility in event-list mode). If events overlap, either as
        supplied or after ``prefer_turnarounds`` has snapped them.
    TypeError
        If an element of ``retune_events`` is not a
        :class:`~fyst_trajectories.trajectory.RetuneEvent`.

    Warns
    -----
    PointingWarning
        If ``retune_events`` is supplied together with a non-default
        ``retune_interval`` or ``retune_duration`` (the scalar kwarg is
        ignored in event-list mode). If any event has ``t_start`` at or
        after the trajectory end (those events are dropped with a single
        summary warning naming the affected sorted indices). If
        ``prefer_turnarounds=True`` but the trajectory has identically
        zero velocities (falls back to time-based placement).

    Examples
    --------
    Uniform cadence::

        result = inject_retune(traj, retune_interval=300.0, retune_duration=5.0)

    Explicit event list (Monte Carlo, log replay, etc.)::

        from fyst_trajectories import RetuneEvent

        events = [
            RetuneEvent(t_start=30.0, duration=5.0),
            RetuneEvent(t_start=200.0, duration=8.0),
        ]
        result = inject_retune(traj, retune_events=events)
        assert result.retune_events == tuple(events)
    """
    # The window applies in both modes, so it is checked before the split.
    if not math.isfinite(turnaround_window) or turnaround_window < 0:
        raise ValueError(
            f"turnaround_window must be finite and non-negative, got {turnaround_window}"
        )

    if retune_events is not None:
        # Mutual-exclusion: per-module staggering is not a scalar concept
        # in event-list mode; callers compose by calling once per module.
        if module_index != 0 or n_modules != 1:
            raise ValueError(
                f"inject_retune: retune_events is mutually exclusive with "
                f"per-module staggering (got module_index={module_index}, "
                f"n_modules={n_modules}). Call inject_retune once per "
                "module with its own event list to stagger."
            )
        if retune_interval != _DEFAULT_RETUNE_INTERVAL_SEC:
            warnings.warn(
                "inject_retune: retune_interval is ignored when "
                f"retune_events is supplied (got retune_interval={retune_interval}).",
                PointingWarning,
                stacklevel=2,
            )
        if retune_duration != DEFAULT_RETUNE_DURATION_SEC:
            warnings.warn(
                "inject_retune: retune_duration is ignored when "
                f"retune_events is supplied (got retune_duration={retune_duration}).",
                PointingWarning,
                stacklevel=2,
            )

        prefer_turnarounds = _zero_velocity_guard(trajectory, prefer_turnarounds)
        return _inject_retune_events(
            trajectory,
            retune_events,
            prefer_turnarounds=prefer_turnarounds,
            turnaround_window=turnaround_window,
        )

    if not math.isfinite(retune_interval) or retune_interval <= 0:
        raise ValueError(f"retune_interval must be finite and positive, got {retune_interval}")
    if not math.isfinite(retune_duration) or retune_duration <= 0:
        raise ValueError(f"retune_duration must be finite and positive, got {retune_duration}")
    if retune_duration >= retune_interval:
        raise ValueError(
            f"retune_duration must be shorter than retune_interval, "
            f"got {retune_duration} >= {retune_interval}"
        )
    if n_modules < 1:
        raise ValueError(f"n_modules must be >= 1, got {n_modules}")
    if module_index < 0 or module_index >= n_modules:
        raise ValueError(f"module_index must be in [0, {n_modules}), got {module_index}")

    # Defensive guard: turnaround detection relies on real velocities (the
    # uniform-cadence helper classifies turnarounds from
    # ``SCAN_FLAG_TURNAROUND`` samples derived from the trajectory's velocity
    # profile). A trajectory with identically zero az/el velocities has no
    # detectable turnarounds, so snapping would silently collapse to
    # time-based placement anyway. Warn and fall back explicitly so the
    # caller is not misled.
    prefer_turnarounds = _zero_velocity_guard(trajectory, prefer_turnarounds)

    return _inject_retune_uniform(
        trajectory,
        retune_interval=retune_interval,
        retune_duration=retune_duration,
        prefer_turnarounds=prefer_turnarounds,
        turnaround_window=turnaround_window,
        module_index=module_index,
        n_modules=n_modules,
    )


def sample_retune_events(
    duration: float,
    *,
    interval_sampler: Callable[[np.random.Generator], float],
    duration_sampler: Callable[[np.random.Generator], float],
    rng: np.random.Generator,
    t_start: float = 0.0,
) -> list[RetuneEvent]:
    """Draw a retune event list from caller-supplied samplers.

    No canonical distribution is baked in because no public KID-camera
    retune log has been published.

    Walks forward from ``t_start``, alternating draws from
    ``interval_sampler`` (gap until the next retune) and
    ``duration_sampler`` (duration of that retune). Stops when the next
    drawn interval would push ``t_start`` past ``duration``; the
    partially-drawn event is discarded, not truncated, so every
    returned event has exactly the duration the sampler produced and
    the returned list is guaranteed non-overlapping.

    Parameters
    ----------
    duration : float
        Trajectory window to fill, in seconds. Must be finite and
        non-negative.
    interval_sampler : callable
        ``(rng) -> float``: draws the gap between consecutive retunes
        (or between ``t_start`` and the first retune). Must return a
        positive, finite value; negative or non-finite draws raise
        :class:`ValueError`.
    duration_sampler : callable
        ``(rng) -> float``: draws the duration of the next retune.
        Must return a positive, finite value.
    rng : np.random.Generator
        Seeded generator for reproducibility. Caller owns seed policy.
    t_start : float
        Starting offset in seconds from the trajectory origin. Default 0.

    Returns
    -------
    list of RetuneEvent
        Events in chronological order, guaranteed non-overlapping.

    Raises
    ------
    ValueError
        If ``duration`` is negative or non-finite. If ``t_start`` is
        negative or non-finite. If either sampler returns a
        non-positive or non-finite value.

    Notes
    -----
    The walk consumes one extra ``interval_sampler`` draw past the last
    emitted event to evaluate the termination condition, so callers who
    seed for an exact draw count should account for this.

    Examples
    --------
    >>> import numpy as np
    >>> from fyst_trajectories import sample_retune_events
    >>> rng = np.random.default_rng(seed=42)
    >>> events = sample_retune_events(
    ...     duration=600.0,
    ...     interval_sampler=lambda r: r.uniform(60.0, 120.0),
    ...     duration_sampler=lambda r: r.uniform(3.0, 8.0),
    ...     rng=rng,
    ... )
    """
    if not math.isfinite(duration) or duration < 0:
        raise ValueError(f"duration must be finite and non-negative, got {duration}")
    if not math.isfinite(t_start) or t_start < 0:
        raise ValueError(f"t_start must be finite and non-negative, got {t_start}")

    events: list[RetuneEvent] = []
    current = t_start
    while True:
        gap = interval_sampler(rng)
        if not math.isfinite(gap) or gap <= 0:
            raise ValueError(f"interval_sampler returned non-positive or non-finite value: {gap}")
        event_start = current + gap
        if event_start >= duration:
            break
        dur = duration_sampler(rng)
        if not math.isfinite(dur) or dur <= 0:
            raise ValueError(f"duration_sampler returned non-positive or non-finite value: {dur}")
        events.append(RetuneEvent(t_start=event_start, duration=dur))
        current = event_start + dur
    return events
