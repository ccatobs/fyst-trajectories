"""Read a planned night back: the summary value, the dispatch sheet and the dispatch records.

All three are derived from the timeline's blocks and its decoded metadata
and never re-plan anything. The sheet and the records render every time
and duration as the ECSV file stores the block times, so each is identical
from a timeline read back from ECSV and from the timeline in memory.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple, TypedDict, cast

from astropy.time import Time

from ..models import BlockType, ObservingTimeline, TimelineBlock
from ..schemas import SourceCESScanParams
from ..transitions import DeferralReason
from ..utils import _search_start_time, _utc_instant
from .policy import (
    CalibrationNightMetadata,
    _policy_from_record,
    read_calibration_night_metadata,
)
from .state import _opens_visit

__all__ = [
    "BodySummary",
    "IdleRecord",
    "NightRecord",
    "NightSummary",
    "OperationRecord",
    "PassRecord",
    "SlewRecord",
    "VisitRecord",
    "dispatch_records",
    "dispatch_sheet",
    "summarize_calibration_night",
]


def _in_minutes(seconds: float) -> float:
    return seconds / 60.0


def _is_pass(block: TimelineBlock) -> bool:
    return (
        block.block_type == BlockType.CALIBRATION and block.metadata.get("cal_type") == "planet_cal"
    )


def _is_waiting(block: TimelineBlock) -> bool:
    """Return True for an idle block that waits for a pass's crossing to open."""
    return block.block_type == BlockType.IDLE and block.metadata.get("reason") == str(
        DeferralReason.WAITING_FOR_PASS
    )


def _stored(t: Time) -> Time:
    """Return ``t`` as :func:`write_timeline` stores it, a UTC ISO string to the millisecond."""
    return Time(_utc_instant(t).iso, scale="utc")


class _Span(NamedTuple):
    """A block with its start and its duration in seconds as the file stores its times."""

    block: TimelineBlock
    start: Time
    seconds: float


def _in_stored_order(timeline: ObservingTimeline) -> list[_Span]:
    """Return the night's blocks in time order, each with its stored start and duration.

    The sheet and the records render from these, never from the times held
    in memory, so a night read back from ECSV renders the same. Blocks are
    ordered by their stored starts, ties keeping the timeline's own order,
    which the file keeps too.
    """
    spans = []
    for block in timeline.blocks:
        start = _stored(block.t_start)
        spans.append(_Span(block, start, float((_stored(block.t_stop) - start).sec)))
    return sorted(spans, key=lambda span: span.start.unix)


def _pass_slack(previous: _Span | None) -> float:
    """Seconds of slack the plan left in front of a pass.

    That is the duration of the idle block immediately before it that
    waits for the crossing to open, or 0 when the pass follows something
    else. Before a visit's first pass the wait runs from the end of the
    detector operations to the pass start, never shorter than the dispatch
    lead: the anchor's distance past the operations, about the lead, plus
    the kernel's own offset from the anchor. Before a later pass of a
    visit it is the wait for the next elevation step.
    """
    if previous is not None and _is_waiting(previous.block):
        return previous.seconds
    return 0.0


def _operation(block: TimelineBlock) -> str:
    """Name the detector operation a calibration block stands for."""
    return str(block.metadata.get("operation") or block.metadata.get("cal_type") or block.scan_type)


@dataclass(frozen=True)
class BodySummary:
    """One body's totals over the night.

    Parameters
    ----------
    body : str
        Body name.
    visits : int
        Number of visits (a visit is one or more consecutive passes).
    passes : int
        Number of passes.
    minutes_on_source : float
        Total pass time in minutes.
    mean_duty_cycle : float
        Mean science fraction over the passes.
    requested, applied, solved : tuple of dict
        The geometry records of each pass, in time order.
    module_crossings : dict of str to float
        Mean coverage fraction per module over the passes.
    """

    body: str
    visits: int
    passes: int
    minutes_on_source: float
    mean_duty_cycle: float
    requested: tuple[dict[str, float], ...]
    applied: tuple[dict[str, float], ...]
    solved: tuple[dict[str, float], ...]
    module_crossings: dict[str, float]


@dataclass(frozen=True)
class NightSummary:
    """The totals of a planned night; ``str()`` renders a text report.

    Parameters
    ----------
    requested_window : tuple of str
        The requested ``(start, end)`` as ISO UTC.
    usable_interval : tuple of str or None
        The solar-gated interval planned, or ``None``.
    targets : tuple of str
        The bodies in priority order.
    bodies : tuple of BodySummary
        Per-body totals, in target order, bodies with no passes included.
    tuning_minutes : float
        Minutes reserved for detector operations.
    slew_minutes : float
        Minutes spent slewing.
    idle_minutes : float
        Minutes idle.
    idle_reasons : dict of str to float
        Idle minutes by reason label.
    deferrals : tuple of dict
        Every deferral, as ``{"body", "at", "reason"}``.
    drops : tuple of dict
        Every drop for the night.
    unplaced : tuple of dict
        Scripted entries that timed out, as ``{"body", "at", "overrides"}``,
        ``at`` being when the entry was set aside.
    warnings : tuple of str
        Advisories the planner recorded, one per distinct message.
    """

    requested_window: tuple[str, str]
    usable_interval: tuple[str, str] | None
    targets: tuple[str, ...]
    bodies: tuple[BodySummary, ...]
    tuning_minutes: float
    slew_minutes: float
    idle_minutes: float
    idle_reasons: dict[str, float] = field(default_factory=dict)
    deferrals: tuple[dict[str, str], ...] = ()
    drops: tuple[dict[str, str], ...] = ()
    unplaced: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()

    def __str__(self) -> str:
        start, end = self.requested_window
        lines = [f"Calibration night {start[:16]} to {end[:16]} UTC"]
        if self.usable_interval is None:
            lines.append("  usable interval: none (the Sun never set inside the request)")
        else:
            lines.append(
                f"  usable interval: {self.usable_interval[0][:16]} to "
                f"{self.usable_interval[1][:16]} UTC"
            )
        lines.append(
            f"  tuning {self.tuning_minutes:.1f} min, slew {self.slew_minutes:.1f} min, "
            f"idle {self.idle_minutes:.1f} min"
        )
        for reason, minutes in sorted(self.idle_reasons.items()):
            lines.append(f"    idle ({reason}): {minutes:.1f} min")
        for b in self.bodies:
            if b.passes == 0:
                lines.append(f"  {b.body}: no passes")
                continue
            lines.append(
                f"  {b.body}: {b.visits} visit(s), {b.passes} pass(es), "
                f"{b.minutes_on_source:.1f} min on source, duty cycle {b.mean_duty_cycle:.0%}"
            )
            for record_name, records in (
                ("requested", b.requested),
                ("applied", b.applied),
                ("solved", b.solved),
            ):
                lines.append(f"    {record_name}: {_render_records(records)}")
            crossed = {k: v for k, v in b.module_crossings.items() if v > 0.0}
            if crossed:
                lines.append(
                    "    modules crossed: "
                    + ", ".join(f"{k} {v:.0%}" for k, v in sorted(crossed.items()))
                )
        for record in self.deferrals:
            lines.append(f"  deferred {record['body']} at {record['at'][:16]}: {record['reason']}")
        for record in self.drops:
            lines.append(f"  dropped {record['body']} at {record['at'][:16]}: {record['reason']}")
        for record in self.unplaced:
            lines.append(f"  unplaced script entry: {record['body']} {record['overrides']}")
        for message in self.warnings:
            lines.append(f"  warning: {message}")
        return "\n".join(lines)


def _render_records(records: tuple[dict[str, float], ...]) -> str:
    if not records:
        return "none"
    first = records[0]
    text = ", ".join(f"{k}={v:.3g}" for k, v in sorted(first.items()))
    if len(records) > 1:
        text += f" (+{len(records) - 1} more)"
    return text


def summarize_calibration_night(timeline: ObservingTimeline) -> NightSummary:
    """Summarise a planned night from its blocks and metadata.

    Parameters
    ----------
    timeline : ObservingTimeline
        A timeline from :func:`plan_calibration_night`, in memory or read
        back from ECSV.

    Returns
    -------
    NightSummary
        The totals; ``str()`` renders them.

    Raises
    ------
    KeyError
        If the timeline carries no calibration-night metadata.
    ValueError
        If the payload's schema version is not supported.
    """
    meta: CalibrationNightMetadata = read_calibration_night_metadata(timeline)
    blocks = sorted(timeline.blocks, key=lambda b: b.t_start.unix)

    tuning = slew = idle = 0.0
    idle_reasons: dict[str, float] = defaultdict(float)
    per_body: dict[str, list[TimelineBlock]] = defaultdict(list)
    visits: dict[str, int] = defaultdict(int)
    previous_pass_body: str | None = None
    for block in blocks:
        if block.block_type == BlockType.SLEW:
            slew += block.duration
            previous_pass_body = None
        elif block.block_type == BlockType.IDLE:
            idle += block.duration
            idle_reasons[str(block.metadata.get("reason", "unlabelled"))] += _in_minutes(
                block.duration
            )
        elif _is_pass(block):
            body = str(block.metadata.get("target"))
            per_body[body].append(block)
            if previous_pass_body != body or _opens_visit(block):
                visits[body] += 1
            previous_pass_body = body
        elif block.block_type == BlockType.CALIBRATION:
            tuning += block.duration
            previous_pass_body = None

    bodies = []
    for body in meta["targets"]:
        passes = per_body.get(body, [])
        crossings: dict[str, list[float]] = defaultdict(list)
        for block in passes:
            for name, fraction in block.metadata.get("module_crossings", {}).items():
                crossings[name].append(float(fraction))
        duty = [float(b.metadata.get("science_fraction", 0.0)) for b in passes]
        bodies.append(
            BodySummary(
                body=body,
                visits=visits.get(body, 0),
                passes=len(passes),
                minutes_on_source=_in_minutes(sum(b.duration for b in passes)),
                mean_duty_cycle=sum(duty) / len(duty) if duty else 0.0,
                requested=tuple(dict(b.metadata.get("requested", {})) for b in passes),
                applied=tuple(dict(b.metadata.get("applied", {})) for b in passes),
                solved=tuple(dict(b.metadata.get("solved", {})) for b in passes),
                module_crossings={
                    name: sum(values) / len(values) for name, values in sorted(crossings.items())
                },
            )
        )

    usable = meta["usable_interval"]
    return NightSummary(
        requested_window=(meta["requested_window"][0], meta["requested_window"][1]),
        usable_interval=None if usable is None else (usable[0], usable[1]),
        targets=tuple(meta["targets"]),
        bodies=tuple(bodies),
        tuning_minutes=_in_minutes(tuning),
        slew_minutes=_in_minutes(slew),
        idle_minutes=_in_minutes(idle),
        idle_reasons=dict(idle_reasons),
        deferrals=tuple(meta["deferrals"]),
        drops=tuple(meta["drops"]),
        unplaced=tuple(meta["unplaced"]),
        warnings=tuple(dict.fromkeys(record["message"] for record in meta.get("warnings", []))),
    )


# The pass-dict keys that shape a pass but that the execution layer's
# source-scan task, at the revision this library is checked against, does
# not read: it drops them without a message and plans the pass from its own
# defaults. The sheet's note and the records' ``unforwarded_keys`` name the
# ones a pass carries. The simulator cannot import from tests, so this is a
# copy; tests/test_ac_schema_contract.py holds it equal to the keys that test
# classifies as awaiting forwarding against the vendored snapshot of that
# task's keys.
_UNFORWARDED_KEYS = frozenset(
    {"az_speed", "az_throw", "dwell", "eta_offset_deg", "footprint_margin"}
)


def _unforwarded_keys(params: Mapping[str, Any]) -> list[str]:
    """Return the keys of a pass dict that the execution layer drops, sorted."""
    return [key for key in sorted(params) if key in _UNFORWARDED_KEYS]


# The execution layer's rule for a pass: the sheet's header prints it and
# dispatch_records records it; dispatch_sheet's docstring paraphrases it, so
# change the two together.
_LATENESS_RULE = (
    "PCS builds a pass from its dict, searching from scheduled_t0_unix when dispatched at "
    "least SCAN_DISPATCH_BUFFER_SEC before that instant (the planned pass, its unforwarded "
    "keys aside) and from its own floor when later (the same crossing to within tens of "
    "milliseconds, or a refusal once the source has entered the footprint); it refuses a pass "
    "start more than MAX_DISPATCH_DELAY_SEC out and never re-plans one, so skip a missed pass "
    "and keep the next visit's times"
)


def _scheduled_t0_unix(span: _Span) -> float:
    """Return the start of the planner's search for a pass, as Unix seconds.

    ``metadata["search_start"]``, the visit's anchor, which every pass of
    the visit records: the execution layer searches from it, or from its
    own floor when that is later, and builds a visit's first pass as
    planned (its unforwarded keys aside) when dispatched at least
    ``SCAN_DISPATCH_BUFFER_SEC`` before it. A block written without it
    gives its start as the file stores it. On a day with a leap second
    astropy's Unix time spreads the day's 86401 s over 86400, so the start
    held in memory and the millisecond the file holds can differ in the
    third decimal; reading the stored one gives the same value from memory
    and from ECSV.
    """
    record = span.block.metadata.get("search_start")
    if record is None:
        return float(span.start.unix)
    return float(_search_start_time(record).unix)


def dispatch_sheet(timeline: ObservingTimeline) -> str:
    """Render a planned night as numbered rows a person can dispatch from.

    One row per block in time order. A pass row carries the literal
    relative ``scan_params`` dict to hand to the execution layer's
    source-scan task with the ``scheduled_t0_unix`` it takes beside it,
    the scan mode and boresight elevation, the planned encoder wrap, the
    slack the plan left in front of the pass, and a blank column for the
    actual elapsed time. Detector operations are rows of their own. The
    ``scheduled_t0_unix`` is the instant the planner's search for the pass
    began, not the pass's start: the source-scan task floors its start to
    ``SCAN_DISPATCH_BUFFER_SEC`` after the dispatch and searches for the
    pass from the later of that floor and ``scheduled_t0_unix``, so a pass
    dispatched at least the buffer before that instant is built as planned
    (its unforwarded keys aside), and one dispatched later is built on the
    same crossing to within tens of milliseconds, or refused once the
    source has entered the footprint; a pass whose start is more than
    ``MAX_DISPATCH_DELAY_SEC`` away is refused too, and a pass is never
    re-planned by the task. The header states that rule and the dispatch
    lead the night's policy reserved before every pass; the slack on a
    pass row is the whole wait before it, never under the lead. A pass row
    whose dict carries keys that shape the pass but that the source-scan
    task, at the revision this library is checked against, drops without a
    message (``az_speed``, ``az_throw``, ``dwell``, ``eta_offset_deg`` and
    ``footprint_margin``) ends with a note naming the ones it carries, for
    the person dispatching to confirm the execution layer forwards them;
    the planner writes ``az_speed`` and ``eta_offset_deg`` into every pass
    dict, so every pass row it plans carries the note. Rendering only:
    nothing is re-planned, and every time and duration is rendered as the
    ECSV file stores the block times, so a sheet printed from a timeline
    read back from ECSV is byte-identical to one printed from memory.

    Parameters
    ----------
    timeline : ObservingTimeline
        A timeline from :func:`plan_calibration_night`.

    Returns
    -------
    str
        The sheet, one row per line.

    Raises
    ------
    KeyError
        If the timeline carries no calibration-night metadata.
    ValueError
        If the payload's schema version is not supported, or the recorded
        policy holds a key this version of the library does not know (named
        in the message) or is one the planner refuses.
    ScanParamsSchemaError
        If a pass block's ``search_start`` is not two finite numbers.
    """
    meta = read_calibration_night_metadata(timeline)
    lead = _policy_from_record(meta["policy"]).dispatch_lead_seconds
    lines = [
        f"# calibration night, targets {', '.join(meta['targets'])}; "
        f"Sun policy {meta['sun_safe']}; times UTC",
        f"# lateness rule: {_LATENESS_RULE}",
        f"# dispatch lead: {lead:.0f} s reserved before every pass; the slack on a pass row "
        "is its whole wait, the elevation step included before a later pass of a visit",
        "#   row  start                 duration  action",
    ]
    previous: _Span | None = None
    for n, span in enumerate(_in_stored_order(timeline), start=1):
        block = span.block
        start = span.start.iso[:19]
        duration = f"{span.seconds / 60.0:7.1f} min"
        if _is_pass(block):
            record = _pass_record(span, previous)
            params = dict(record["scan_params"])
            lines.append(
                f"{n:6d}  {start}  {duration}  source_scan on {params.get('body')}: "
                f"mode {record['mode']}, el_bore {float(record['el_bore'] or 0.0):.2f} deg, "
                f"visit wrap az {record['wrap_az']:.1f} deg, "
                f"legs {record['n_legs']}, "
                f"duty {float(record['science_fraction'] or 0.0):.0%}; "
                f"slack {_pass_slack(previous):.0f} s; elapsed ______"
            )
            lines.append(
                f"        scan_params={json.dumps(params, sort_keys=True)}  "
                f"scheduled_t0_unix={record['scheduled_t0_unix']:.3f}"
            )
            unforwarded = record["unforwarded_keys"]
            if unforwarded:
                lines.append(
                    "        note: confirm the execution layer forwards " + ", ".join(unforwarded)
                )
        elif block.block_type == BlockType.CALIBRATION:
            lines.append(
                f"{n:6d}  {start}  {duration}  {_operation(block)} at az {block.az_start:.1f}, "
                f"el {block.elevation:.1f} deg (operator action); elapsed ______"
            )
        elif block.block_type == BlockType.SLEW:
            lines.append(
                f"{n:6d}  {start}  {duration}  slew az {block.az_start:.1f} to "
                f"{block.az_end:.1f} deg, el {block.elevation:.1f} deg"
            )
        else:
            lines.append(
                f"{n:6d}  {start}  {duration}  idle ({block.metadata.get('reason', 'unlabelled')})"
            )
        previous = span
    return "\n".join(lines) + "\n"


# The shape of the records below. A key may be added under the same
# version; renaming, removing or redefining a key needs a new one.
_RECORD_SCHEMA_VERSION = 1

_UNFORWARDED_KEYS_NOTE = (
    "confirm the execution layer forwards the keys a pass lists under unforwarded_keys"
)


class SlewRecord(TypedDict):
    """One slew of a :class:`VisitRecord`.

    Attributes
    ----------
    name : str
        The slew block's name: ``"slew_to_<body>"`` for the slew to a
        visit's pose, ``"sun_escape"`` for a move out of the Sun zone.
    t_start_utc : str
        Start as UTC ``"YYYY-MM-DD HH:MM:SS"``, the fraction of a second
        dropped.
    t_start_unix : float
        Start in Unix seconds, to the millisecond.
    duration_s : float
        Duration in seconds, to the millisecond.
    az_from : float
        Encoder azimuth the slew leaves from, in degrees.
    az_to : float
        Encoder azimuth the slew arrives at, in degrees.
    el_to : float
        Elevation the slew arrives at, in degrees.
    """

    name: str
    t_start_utc: str
    t_start_unix: float
    duration_s: float
    az_from: float
    az_to: float
    el_to: float


class OperationRecord(TypedDict):
    """One detector operation of a :class:`VisitRecord`, at the visit's pose.

    Attributes
    ----------
    operation : str
        The operation: the block's ``metadata["operation"]`` when it names
        one (``"find_detectors"``), else its calibration type (``"skydip"``,
        ``"retune"``).
    t_start_utc : str
        Start as UTC ``"YYYY-MM-DD HH:MM:SS"``, the fraction of a second
        dropped.
    t_start_unix : float
        Start in Unix seconds, to the millisecond.
    duration_s : float
        Duration in seconds, to the millisecond.
    az : float
        Encoder azimuth of the pose, in degrees.
    el : float
        Elevation of the pose, in degrees.
    """

    operation: str
    t_start_utc: str
    t_start_unix: float
    duration_s: float
    az: float
    el: float


class PassRecord(TypedDict):
    """One source-CES pass of a :class:`VisitRecord`.

    Attributes
    ----------
    t_start_utc : str
        Pass start as UTC ``"YYYY-MM-DD HH:MM:SS"``, the fraction of a
        second dropped.
    t_start_unix : float
        Pass start in Unix seconds, to the millisecond: the execution layer
        refuses the pass once its search floor passes this instant.
    scheduled_t0_unix : float
        The instant the planner's search for the pass began, in Unix
        seconds to the millisecond. It is not the pass start: it is the
        value the execution layer's source-scan task takes as the start of
        its own search. Dispatched at least ``SCAN_DISPATCH_BUFFER_SEC``
        before it, the task builds a visit's first pass as planned, to
        within microseconds; a later pass of a multi-pass visit records the
        visit's anchor, so the task searches from its own floor and finds
        the same crossing to within tens of milliseconds. Both assume the
        task forwards the keys under ``unforwarded_keys``: without a
        multi-pass visit's ``eta_offset_deg`` a pass moves by minutes or is
        refused. A block recorded without ``search_start`` gives its start
        instead.
    duration_s : float
        Pass duration in seconds, to the millisecond.
    slack_s : float
        Duration in seconds, to the millisecond, of the
        ``waiting_for_pass`` idle immediately before the pass, or 0.0 when
        the pass follows anything else.
    mode : str or None
        ``"rising"`` or ``"setting"``, from ``scan_params``.
    el_bore : float or None
        Boresight elevation in degrees, from ``scan_params``.
    wrap_az : float
        Encoder azimuth the pass was planned in, in degrees: the ``wrap``
        of the block's ``transition`` record (where the visit's slew
        arrived), else the block's start azimuth.
    n_legs : int or None
        Number of azimuth legs.
    science_fraction : float or None
        Fraction of the pass's samples flagged as science.
    scan_params : SourceCESScanParams
        A plain copy of the pass block's relative dispatch dict, which the
        source-scan task takes with ``scheduled_t0_unix``.
    unforwarded_keys : list of str
        The keys of ``scan_params``, sorted, that the source-scan task, at
        the revision this library is checked against, does not forward:
        it drops them without a message and plans the pass from its own
        defaults. The candidates are ``az_speed``, ``az_throw``, ``dwell``,
        ``eta_offset_deg`` and ``footprint_margin``, and the dispatch
        sheet's note under the pass names the same keys.
    """

    t_start_utc: str
    t_start_unix: float
    scheduled_t0_unix: float
    duration_s: float
    slack_s: float
    mode: str | None
    el_bore: float | None
    wrap_az: float
    n_legs: int | None
    science_fraction: float | None
    scan_params: SourceCESScanParams
    unforwarded_keys: list[str]


class VisitRecord(TypedDict):
    """One visit of a :class:`NightRecord`.

    A visit is the slews and detector operations that lead up to a pass,
    that pass and, for a multi-pass visit, the passes that follow it. A
    move with no pass behind it (an escape slew made while the night
    idles) forms a record of its own, with no body and no passes, since
    the telescope still makes the move.

    Attributes
    ----------
    body : str or None
        The body of the visit's passes, from the first pass's
        ``scan_params``; ``None`` on a record without passes.
    slews : list of SlewRecord
        The slews, in time order.
    detector_operations : list of OperationRecord
        The detector operations, in execution order.
    passes : list of PassRecord
        The passes, in time order; empty on a record of moves alone.
    """

    body: str | None
    slews: list[SlewRecord]
    detector_operations: list[OperationRecord]
    passes: list[PassRecord]


class IdleRecord(TypedDict):
    """One idle period of a :class:`NightRecord`.

    Attributes
    ----------
    t_start_utc : str
        Start as UTC ``"YYYY-MM-DD HH:MM:SS"``, the fraction of a second
        dropped.
    t_start_unix : float
        Start in Unix seconds, to the millisecond.
    duration_s : float
        Duration in seconds, to the millisecond.
    reason : str
        The idle block's ``reason`` label (a :class:`DeferralReason` value,
        such as ``"nothing_available"``), or ``"unlabelled"`` when it
        carries none.
    """

    t_start_utc: str
    t_start_unix: float
    duration_s: float
    reason: str


class NightRecord(TypedDict):
    """A planned calibration night as :func:`dispatch_records` returns it.

    Every value is a str, float, int, bool, ``None``, list or dict, so
    ``json.dumps(record)`` serialises the whole record.

    Attributes
    ----------
    schema_version : int
        The shape of the record: 1.
    source : str or None
        The ``source`` the caller passed, such as the name of the file the
        night was read from; ``None`` when none was passed.
    targets : list of str
        The bodies the night was planned for, in priority order.
    sun_policy : str
        Description of the point-level Sun predicate the night was planned
        with.
    usable_interval_utc : list of str or None
        The solar-gated ``[start, end]`` that was planned, as ISO UTC
        strings, or ``None`` when the Sun never set inside the requested
        window.
    lateness_rule : str
        The execution layer's lateness rule, as the dispatch sheet's header
        states it.
    dispatch_lead_s : float
        Seconds the night's policy reserved before every pass, its
        ``dispatch_lead_seconds``; 0.0 for a night recorded before the
        policy had that field.
    unforwarded_keys_note : str
        What to confirm about the keys each pass lists under
        ``unforwarded_keys``.
    night_start_utc : str or None
        Start of the first block as UTC ``"YYYY-MM-DD HH:MM:SS"``, the
        fraction of a second dropped; ``None`` for a night without blocks.
    night_end_utc : str or None
        End of the last block, in the same form; ``None`` for a night
        without blocks.
    visits : list of VisitRecord
        The visits, in time order.
    idle : list of IdleRecord
        The idle periods in time order, but for the ``waiting_for_pass``
        idle immediately before a pass, which is that pass's ``slack_s``.
    """

    schema_version: int
    source: str | None
    targets: list[str]
    sun_policy: str
    usable_interval_utc: list[str] | None
    lateness_rule: str
    dispatch_lead_s: float
    unforwarded_keys_note: str
    night_start_utc: str | None
    night_end_utc: str | None
    visits: list[VisitRecord]
    idle: list[IdleRecord]


def _when(span: _Span) -> tuple[str, float, float]:
    """Return a block's start, as UTC text and as Unix seconds, and its duration in seconds."""
    return span.start.iso[:19], round(float(span.start.unix), 3), round(span.seconds, 3)


def _slew_record(span: _Span) -> SlewRecord:
    block = span.block
    t_start_utc, t_start_unix, duration_s = _when(span)
    return {
        "name": block.patch_name,
        "t_start_utc": t_start_utc,
        "t_start_unix": t_start_unix,
        "duration_s": duration_s,
        "az_from": float(block.az_start),
        "az_to": float(block.az_end),
        "el_to": float(block.elevation),
    }


def _operation_record(span: _Span) -> OperationRecord:
    block = span.block
    t_start_utc, t_start_unix, duration_s = _when(span)
    return {
        "operation": _operation(block),
        "t_start_utc": t_start_utc,
        "t_start_unix": t_start_unix,
        "duration_s": duration_s,
        "az": float(block.az_start),
        "el": float(block.elevation),
    }


def _pass_record(span: _Span, previous: _Span | None) -> PassRecord:
    block = span.block
    params: dict[str, Any] = dict(block.metadata.get("scan_params", {}))
    transition = block.metadata.get("transition", {})
    t_start_utc, t_start_unix, duration_s = _when(span)
    return {
        "t_start_utc": t_start_utc,
        "t_start_unix": t_start_unix,
        "scheduled_t0_unix": round(_scheduled_t0_unix(span), 3),
        "duration_s": duration_s,
        "slack_s": round(_pass_slack(previous), 3),
        "mode": params.get("mode"),
        "el_bore": params.get("el_bore"),
        "wrap_az": float(transition.get("wrap", block.az_start)),
        "n_legs": block.metadata.get("n_legs"),
        "science_fraction": block.metadata.get("science_fraction"),
        "scan_params": cast(SourceCESScanParams, params),
        "unforwarded_keys": _unforwarded_keys(params),
    }


def _idle_record(span: _Span) -> IdleRecord:
    t_start_utc, t_start_unix, duration_s = _when(span)
    return {
        "t_start_utc": t_start_utc,
        "t_start_unix": t_start_unix,
        "duration_s": duration_s,
        "reason": str(span.block.metadata.get("reason", "unlabelled")),
    }


def _new_visit() -> VisitRecord:
    return {"body": None, "slews": [], "detector_operations": [], "passes": []}


def _has_moves(visit: VisitRecord) -> bool:
    return bool(visit["slews"] or visit["detector_operations"])


def dispatch_records(timeline: ObservingTimeline, *, source: str | None = None) -> NightRecord:
    """Return a planned night as one record per visit, the data to dispatch it from.

    The content of :func:`dispatch_sheet` as data. Each visit is one
    :class:`VisitRecord`: its slews, the detector operations reserved at
    the target pose in execution order, and each pass with the literal
    relative ``scan_params`` dict to hand to the execution layer's
    source-scan task, the ``scheduled_t0_unix`` the task takes beside it
    and the keys the sheet's note names, as the sheet prints them for that
    pass. The idle periods are listed apart with their reasons, except the
    ``waiting_for_pass`` idle immediately before a pass, which is that
    pass's ``slack_s``; ``lateness_rule`` is the rule the sheet's header
    states.

    The record is the hand-off contract: the keys listed on
    :class:`NightRecord` and on the record types it holds are stable,
    further keys may be added, and ``schema_version`` (1) names the shape.
    Every value is a JSON type, so ``json.dumps(record)`` is the file;
    ``examples/planet_night_visits.py`` writes it, indented, for a night
    read back from its ECSV. Nothing is re-planned: every value is read
    from the blocks and the night's metadata, and every time from the
    block times as :func:`write_timeline` stores them, UTC to the
    millisecond, so the record of a night in memory is the record of the
    same night read back from ECSV.

    Parameters
    ----------
    timeline : ObservingTimeline
        A night from :func:`plan_calibration_night`, in memory or read back
        from ECSV.
    source : str or None, optional
        Recorded as the record's ``source``, such as the name of the file
        the night was read from.

    Returns
    -------
    NightRecord
        The night's record.

    Raises
    ------
    KeyError
        If the timeline carries no calibration-night metadata.
    ValueError
        If the payload's schema version is not supported, or the recorded
        policy holds a key this version of the library does not know (named
        in the message) or is one the planner refuses.
    ScanParamsSchemaError
        If a pass block's ``search_start`` is not two finite numbers.
    """
    meta = read_calibration_night_metadata(timeline)
    lead = _policy_from_record(meta["policy"]).dispatch_lead_seconds
    spans = _in_stored_order(timeline)
    visits: list[VisitRecord] = []
    idle: list[IdleRecord] = []
    current = _new_visit()  # the visit being filled; open once it holds a pass
    pending = _new_visit()  # moves and operations seen since the last pass
    for n, span in enumerate(spans):
        block = span.block
        if block.block_type == BlockType.SLEW:
            pending["slews"].append(_slew_record(span))
        elif _is_pass(block):
            params = block.metadata.get("scan_params", {})
            if current["passes"] and int(params.get("pass_index", 0)) > 0:
                # A later pass of the open visit. The planner puts only the
                # waiting idle between passes, so this merge is a guard for
                # a timeline built by hand.
                current["slews"].extend(pending["slews"])
                current["detector_operations"].extend(pending["detector_operations"])
            else:
                if current["passes"]:
                    visits.append(current)
                current = pending
                current["body"] = params.get("body")
            current["passes"].append(_pass_record(span, spans[n - 1] if n else None))
            pending = _new_visit()
        elif block.block_type == BlockType.CALIBRATION:
            pending["detector_operations"].append(_operation_record(span))
        elif _is_waiting(block) and n + 1 < len(spans) and _is_pass(spans[n + 1].block):
            pass  # the pass that follows records this wait as its slack
        else:
            idle.append(_idle_record(span))
            if _has_moves(pending):
                # Moves with no pass behind them: close the open visit first
                # so the records stay in time order.
                if current["passes"]:
                    visits.append(current)
                visits.append(pending)
                current = _new_visit()
                pending = _new_visit()
    if current["passes"]:
        visits.append(current)
    if _has_moves(pending):
        visits.append(pending)
    return {
        "schema_version": _RECORD_SCHEMA_VERSION,
        "source": source,
        "targets": list(meta["targets"]),
        "sun_policy": meta["sun_safe"],
        "usable_interval_utc": meta["usable_interval"],
        "lateness_rule": _LATENESS_RULE,
        "dispatch_lead_s": float(lead),
        "unforwarded_keys_note": _UNFORWARDED_KEYS_NOTE,
        "night_start_utc": spans[0].start.iso[:19] if spans else None,
        "night_end_utc": _stored(spans[-1].block.t_stop).iso[:19] if spans else None,
        "visits": visits,
        "idle": idle,
    }
