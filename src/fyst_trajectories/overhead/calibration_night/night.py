"""The night driver: loop the four step functions from dusk to dawn."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from astropy.time import Time

from ..models import CalibrationPolicy, ObservingTimeline, OverheadModel
from ..transitions import DeferralReason
from .policy import (
    CalibrationNightMetadata,
    CalibrationNightPolicy,
    encode_calibration_night_metadata,
    tables_as_record,
)
from .selection import ScriptedSelection, SelectionRule, select_priority
from .state import NightContext, NightState, _describe, _rule_name
from .step import advance_idle, commit_visit, list_candidates
from .tables import ScanParameterTable

if TYPE_CHECKING:
    from ...site import Site
    from ...sun_protocols import SlewSafePredicate, SunSafePredicate

__all__ = ["plan_calibration_night"]

# The loop advances time on every idle tick and records a deferral or a drop
# on every infeasible visit, so it always terminates; this cap only turns a
# programming error into an exception instead of a hang.
_MAX_ITERATIONS_PER_TICK = 64


def plan_calibration_night(
    targets: Sequence[str],
    site: Site,
    start_time: Time | str,
    end_time: Time | str,
    *,
    policy: CalibrationNightPolicy | None = None,
    tables: Mapping[str, ScanParameterTable] | None = None,
    selection: SelectionRule | None = None,
    overhead_model: OverheadModel | None = None,
    calibration_policy: CalibrationPolicy | None = None,
    sun_safe: SunSafePredicate | None = None,
    slew_safe: SlewSafePredicate | None = None,
    start_pose: tuple[float, float] | None = None,
) -> ObservingTimeline:
    """Plan one night of solar-system calibration passes back to back.

    Walks the night from the solar-gated start to its end: at each step
    the targets are assessed (:func:`list_candidates`), the selection rule
    picks one, its visit is planned (:func:`plan_visit`) and committed
    (:func:`commit_visit`); when nothing is selected the night idles one
    tick (:func:`advance_idle`) with the reason recorded on the idle
    block. The result is an ordinary
    :class:`~fyst_trajectories.overhead.ObservingTimeline` whose
    calibration blocks carry relative dispatch dicts and the geometry
    records, with the night's inputs and every deferral recorded in its
    metadata; :func:`summarize_calibration_night` and
    :func:`dispatch_sheet` read it back. Time comes from the state, never
    from a clock, so the same inputs plan the same night every time.

    Parameters
    ----------
    targets : sequence of str
        Bodies in priority order (for example ``["jupiter", "saturn"]``).
        Names are matched in lower case and each body may appear only
        once; the night revisits its targets under the selection rule, so
        a repeated name is rejected rather than visited twice.
    site : Site
        Observing site.
    start_time, end_time : Time or str
        The requested window, held in UTC: an ISO string is read as UTC,
        and any ``Time`` is held as a UTC ``Time`` without a location and
        with astropy's default ``precision`` and ``out_subfmt`` (one in
        another scale is converted to UTC). The planned interval is clipped
        to when the Sun is down inside it.
    policy : CalibrationNightPolicy, optional
        The night's policy. Default ``CalibrationNightPolicy()``.
    tables : mapping of str to ScanParameterTable, optional
        Scan-parameter tables keyed by body with ``"default"`` for the
        shared one. Keys are matched in lower case, so ``"Uranus"`` and
        ``"uranus"`` name the same table. Default
        :data:`DEFAULT_SCAN_TABLES`.
    selection : SelectionRule, optional
        Which body to visit next. Default :func:`select_priority`.
    overhead_model : OverheadModel, optional
        Durations of the detector operations and the settle time.
    calibration_policy : CalibrationPolicy, optional
        Cadences; the skydip cadence is read from it.
    sun_safe : SunSafePredicate, optional
        Point-level Sun predicate; default the scalar site model.
    slew_safe : SlewSafePredicate, optional
        Path-level Sun predicate; default ``sun_safe`` swept along the
        slew path under the site's axis limits.
    start_pose : tuple of float, optional
        The ``(az, el)`` the telescope starts from; default the offline
        scheduler's bootstrap pose.

    Returns
    -------
    ObservingTimeline
        The planned night. When the Sun never sets inside the request the
        timeline is empty and its metadata records ``usable_interval`` as
        ``None``.

    Raises
    ------
    ValueError
        If ``targets`` is empty, names a body twice or names one with no
        ephemeris, the window is not ordered, a body has no table, or the
        selection rule returns a body that is not an available candidate.

    See Also
    --------
    fyst_trajectories.overhead.generate_timeline : the survey-night
        simulator over science patches; this planner shares its block
        model and outputs but walks a body queue instead of a cadence
        loop.
    """
    ctx = NightContext.build(
        targets,
        site,
        start_time,
        end_time,
        policy=policy,
        tables=tables,
        overhead_model=overhead_model,
        calibration_policy=calibration_policy,
        sun_safe=sun_safe,
        slew_safe=slew_safe,
    )
    return _run_night(ctx, selection, start_pose)


def _run_night(
    ctx: NightContext,
    selection: SelectionRule | None = None,
    start_pose: tuple[float, float] | None = None,
) -> ObservingTimeline:
    """Loop the four steps over a prepared context (the seam the loop tests use)."""
    rule: SelectionRule = select_priority if selection is None else selection
    initial = NightState.initial(ctx.start_time, start_pose)
    state = initial
    pol = ctx.policy
    deferrals: list[dict[str, str]] = []
    drops: list[dict[str, str]] = []
    advisories: list[dict[str, str]] = []
    set_aside_at: list[str] = []

    while ctx.usable and (ctx.end_time - state.t).to_value("s") > pol.min_pass_seconds:
        for _ in range(_MAX_ITERATIONS_PER_TICK):
            candidates = list_candidates(state, ctx)
            choice = rule(candidates, state)
            if choice is None:
                n_unplaced = len(state.unplaced)
                state = _idle(state, ctx, rule)
                if len(state.unplaced) > n_unplaced:
                    set_aside_at.append(state.t.iso)
                break
            body, overrides = choice
            body = body.lower()
            if body not in {c.body for c in candidates if c.available}:
                raise ValueError(
                    f"selection returned {body!r}, which is not a candidate available now"
                )
            plan = ctx.visit_planner(state, ctx, body, overrides)
            advisories.extend(
                {"body": body, "at": state.t.iso, "message": message} for message in plan.warnings
            )
            state = commit_visit(state, plan, retry_after=pol.retry_after_seconds)
            if plan.feasible:
                break
            record = {"body": body, "at": state.t.iso, "reason": str(plan.reason)}
            (drops if body in state.dropped else deferrals).append(record)
        else:
            raise RuntimeError("the night planner made no progress in one tick")

    if ctx.usable:
        state = advance_idle(
            state, ctx, (ctx.end_time - state.t).to_value("s"), DeferralReason.WINDOW_CLOSED
        )

    meta: CalibrationNightMetadata = {
        "targets": list(ctx.targets),
        "policy": pol.as_record(),
        "tables": tables_as_record(dict(ctx.tables)),
        "selection": _rule_name(rule),
        "sun_safe": _describe(ctx.sun_safe),
        "slew_safe": _describe(ctx.slew_safe),
        "requested_window": [ctx.requested_start.iso, ctx.requested_end.iso],
        "usable_interval": [ctx.start_time.iso, ctx.end_time.iso] if ctx.usable else None,
        "start_pose": [initial.az, initial.el],
        "deferrals": deferrals,
        "drops": drops,
        "unplaced": [
            {"body": body, "at": at, "overrides": overrides.as_record()}
            for (body, overrides), at in zip(state.unplaced, set_aside_at, strict=True)
        ],
        "warnings": advisories,
        "telescope_limits": dataclasses.asdict(ctx.site.telescope_limits),
    }
    return ObservingTimeline(
        blocks=list(state.blocks),
        site=ctx.site,
        start_time=ctx.start_time,
        end_time=ctx.end_time,
        overhead_model=ctx.overhead_model,
        calibration_policy=ctx.calibration_policy,
        metadata=encode_calibration_night_metadata(meta),
    )


def _idle(state: NightState, ctx: NightContext, rule: SelectionRule) -> NightState:
    """One idle tick, with the scripted rule's waiting bookkeeping."""
    pol = ctx.policy
    reason = DeferralReason.NOTHING_AVAILABLE
    if isinstance(rule, ScriptedSelection) and rule.current(state) is not None:
        reason = DeferralReason.SCRIPT_WAITING
        since = state.script_waiting_since or state.t
        waited = (state.t - since).to_value("s")
        # Time arithmetic carries sub-millisecond noise; a wait is judged to the millisecond.
        if waited + 1e-3 >= pol.max_wait_seconds:
            entry = rule.current(state)
            assert entry is not None
            state = state.advanced(
                unplaced=state.unplaced + (entry,),
                script_index=state.script_index + 1,
                script_waiting_since=None,
            )
            return state
        state = state.advanced(script_waiting_since=since)
    return advance_idle(state, ctx, pol.time_step, reason)
