"""Read a planned night back: the summary value and the dispatch sheet.

Both are derived from the timeline's blocks and its decoded metadata and
never re-plan anything, so a sheet printed from a timeline read back from
ECSV is identical to one printed from the timeline in memory.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..models import BlockType, ObservingTimeline, TimelineBlock
from ..transitions import DeferralReason
from .policy import CalibrationNightMetadata, read_calibration_night_metadata

__all__ = [
    "BodySummary",
    "NightSummary",
    "dispatch_sheet",
    "summarize_calibration_night",
]


def _minutes(seconds: float) -> float:
    return seconds / 60.0


def _is_pass(block: TimelineBlock) -> bool:
    return (
        block.block_type == BlockType.CALIBRATION and block.metadata.get("cal_type") == "planet_cal"
    )


def _pass_slack(previous: TimelineBlock | None) -> float:
    """Seconds of slack the plan left in front of a pass.

    That is the duration of the idle block immediately before it that
    waits for the crossing to open, or 0 when the pass follows something
    else.
    """
    if (
        previous is not None
        and previous.block_type == BlockType.IDLE
        and previous.metadata.get("reason") == str(DeferralReason.WAITING_FOR_PASS)
    ):
        return previous.duration
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
            idle_reasons[str(block.metadata.get("reason", "unlabelled"))] += _minutes(
                block.duration
            )
        elif _is_pass(block):
            body = str(block.metadata.get("target"))
            per_body[body].append(block)
            if previous_pass_body != body:
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
                minutes_on_source=_minutes(sum(b.duration for b in passes)),
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
        tuning_minutes=_minutes(tuning),
        slew_minutes=_minutes(slew),
        idle_minutes=_minutes(idle),
        idle_reasons=dict(idle_reasons),
        deferrals=tuple(meta["deferrals"]),
        drops=tuple(meta["drops"]),
        unplaced=tuple(meta["unplaced"]),
        warnings=tuple(dict.fromkeys(record["message"] for record in meta.get("warnings", []))),
    )


# The pass-dict keys that shape a pass but that the execution layer's
# source-scan task, at the revision this library is checked against, does
# not read: it drops them without a message and plans the pass from its own
# defaults. The simulator cannot import from tests, so this is a copy;
# tests/test_ac_schema_contract.py holds it equal to the keys that test
# classifies as awaiting forwarding against the vendored snapshot of that
# task's keys.
_UNFORWARDED_KEYS = frozenset(
    {"az_speed", "az_throw", "dwell", "eta_offset_deg", "footprint_margin"}
)


def dispatch_sheet(timeline: ObservingTimeline) -> str:
    """Render a planned night as numbered rows a person can dispatch from.

    One row per block in time order. A pass row carries the literal
    relative ``scan_params`` dict to hand to the execution layer's
    source-scan task, the scan mode and boresight elevation, the planned
    encoder wrap, the slack the plan left in front of the pass, and
    a blank column for the actual elapsed time. Detector operations are
    rows of their own. A pass row whose dict carries keys that shape the
    pass but that the source-scan task, at the revision this library is
    checked against, drops without a message (``az_speed``, ``az_throw``,
    ``dwell``, ``eta_offset_deg`` and ``footprint_margin``) ends with a
    note naming the ones it carries, for the person dispatching to confirm
    the execution layer forwards them; the planner writes ``az_speed`` and
    ``eta_offset_deg`` into every pass dict, so every pass row it plans
    carries the note. Rendering only: nothing is re-planned, so a sheet
    printed from a timeline read back from ECSV is byte-identical to one
    printed from memory.

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
        If the payload's schema version is not supported.
    """
    meta = read_calibration_night_metadata(timeline)
    blocks = sorted(timeline.blocks, key=lambda b: b.t_start.unix)
    lines = [
        f"# calibration night, targets {', '.join(meta['targets'])}; "
        f"Sun policy {meta['sun_safe']}; times UTC",
        "# lateness rule: the slack on a pass row is what the plan left in front of it; "
        "once its start has gone by, re-plan the next block from now",
        "#   row  start                 duration  action",
    ]
    previous: TimelineBlock | None = None
    for n, block in enumerate(blocks, start=1):
        start = block.t_start.iso[:19]
        duration = f"{block.duration / 60.0:7.1f} min"
        if _is_pass(block):
            params = dict(block.metadata.get("scan_params", {}))
            transition = block.metadata.get("transition", {})
            lines.append(
                f"{n:6d}  {start}  {duration}  source_scan on {params.get('body')}: "
                f"mode {params.get('mode')}, el_bore {float(params.get('el_bore', 0.0)):.2f} deg, "
                f"visit wrap az {float(transition.get('wrap', block.az_start)):.1f} deg, "
                f"legs {block.metadata.get('n_legs')}, "
                f"duty {float(block.metadata.get('science_fraction', 0.0)):.0%}; "
                f"slack {_pass_slack(previous):.0f} s; elapsed ______"
            )
            lines.append(
                f"        scan_params={json.dumps(params, sort_keys=True)}  "
                f"scheduled_t0_unix={block.t_start.unix:.3f}"
            )
            unforwarded = [key for key in sorted(params) if key in _UNFORWARDED_KEYS]
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
        previous = block
    return "\n".join(lines) + "\n"
