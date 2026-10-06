"""The immutable state and the resolved context of a calibration night.

:class:`NightState` is what the step functions transform; it mirrors the
offline scheduler's state record (initial pose, ``advanced(**changes)``)
without sharing it, and carries the body queue the scheduler lacks: the
deferrals with their retry times, the drops, the scripted-selection
cursor and the entries it set aside. :class:`NightContext` holds every
resolved input so a step never reads a default at run time. Both rebuild
from a planned night's timeline (``from_timeline``), so a night can be
resumed after the session that planned it is gone.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from astropy.time import Time, TimeDelta

from ...coordinates import SATELLITE_BODIES, SOLAR_SYSTEM_BODIES, Coordinates
from ...site import Site
from ...sun_models import make_sun_safe
from ..calibration_state import CalibrationState
from ..models import (
    BlockType,
    CalibrationPolicy,
    CalibrationType,
    ObservingTimeline,
    OverheadModel,
    TimelineBlock,
)
from ..scheduler.state import BOOTSTRAP_POSE
from ..transitions import DeferralReason, _default_slew_safe
from ..utils import _utc_instant
from .helpers import usable_interval
from .policy import (
    CalibrationNightPolicy,
    ScanOverrides,
    _policy_from_record,
    read_calibration_night_metadata,
    tables_from_record,
)
from .selection import ScriptedSelection, SelectionRule, select_priority
from .tables import DEFAULT_SCAN_TABLES, ScanParameterTable, table_for

if TYPE_CHECKING:
    from ...sun_protocols import SlewSafePredicate, SunSafePredicate
    from .step import VisitPlan

__all__ = [
    "BOOTSTRAP_POSE",
    "NightContext",
    "NightState",
    "VisitPlanner",
]

# The detector operations and passes a committed visit stamps on the
# calibration record.
_FOLDED_CALIBRATIONS = (
    CalibrationType.RETUNE,
    CalibrationType.SKYDIP,
    CalibrationType.PLANET_CAL,
)
# A resume compares times to the millisecond: the ECSV stores block times
# and the night's record stores its times as ISO strings at that precision.
_RESUME_TOL_SEC = 1e-3


def _fold_calibrations(
    cal_state: CalibrationState, blocks: Iterable[TimelineBlock]
) -> CalibrationState:
    """Stamp each retune, skydip and planet-cal block on ``cal_state`` at its start."""
    for block in blocks:
        if block.block_type == BlockType.CALIBRATION:
            cal_type = block.metadata.get("cal_type")
            if cal_type in _FOLDED_CALIBRATIONS:
                cal_state = cal_state.update(cal_type, block.t_start)
    return cal_state


def _describe(predicate: Any) -> str:
    """Return the description a night records for a Sun predicate."""
    return str(getattr(predicate, "describe", type(predicate).__name__))


def _rule_name(rule: Any) -> str:
    """Return the name a night records for its selection rule."""
    return str(getattr(rule, "__name__", type(rule).__name__))


def _by(at: Time, t: Time) -> bool:
    """Whether ``at`` is at or before ``t``, to the millisecond."""
    return (at - t).to_value("s") <= _RESUME_TOL_SEC


def _is_pass(block: TimelineBlock) -> bool:
    return (
        block.block_type == BlockType.CALIBRATION
        and block.metadata.get("cal_type") == CalibrationType.PLANET_CAL
    )


def _opens_visit(block: TimelineBlock) -> bool:
    """Whether a pass block is the first of its visit (``pass_index`` 0, or no dispatch dict)."""
    scan_params = block.metadata.get("scan_params")
    return not scan_params or int(scan_params.get("pass_index", 0)) == 0


@runtime_checkable
class VisitPlanner(Protocol):
    """The seam through which a night plans one visit.

    The default is :func:`~fyst_trajectories.overhead.plan_visit`; a test
    or an interactive session can substitute a planner that returns a
    prepared :class:`~fyst_trajectories.overhead.VisitPlan` without any
    sky computation.
    """

    def __call__(
        self,
        state: NightState,
        ctx: NightContext,
        body: str,
        overrides: ScanOverrides | None = None,
    ) -> VisitPlan:
        """Plan the visit of ``body`` from ``state``."""
        ...


@dataclass(frozen=True)
class NightState:
    """Everything a step needs to know about the night so far.

    Parameters
    ----------
    t : Time
        The current time, held as a UTC ``Time`` without a location and
        with astropy's default ``precision`` and ``out_subfmt`` (one in
        another scale is converted to UTC).
    az, el : float
        The telescope pose in degrees, azimuth in the encoder frame.
    blocks : tuple of TimelineBlock
        The blocks emitted so far, in time order.
    cal_state : CalibrationState
        When each detector operation last ran.
    deferred : mapping of str to (Time, DeferralReason)
        Bodies waiting for a retry time, with the reason they deferred.
    dropped : mapping of str to DeferralReason
        Bodies dropped for the night, with the reason.
    script_index : int
        Cursor of a scripted selection.
    script_waiting_since : Time or None
        When the current scripted entry started waiting for its body.
    unplaced : tuple of (str, ScanOverrides)
        Scripted entries set aside after waiting too long.
    scan_counter : int
        Scan counter carried on the blocks.
    """

    t: Time
    az: float
    el: float
    blocks: tuple[TimelineBlock, ...] = ()
    cal_state: CalibrationState = field(default_factory=CalibrationState)
    deferred: Mapping[str, tuple[Time, DeferralReason]] = field(default_factory=dict)
    dropped: Mapping[str, DeferralReason] = field(default_factory=dict)
    script_index: int = 0
    script_waiting_since: Time | None = None
    unplaced: tuple[tuple[str, ScanOverrides], ...] = ()
    scan_counter: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", _utc_instant(self.t))

    @classmethod
    def initial(cls, start_time: Time, start_pose: tuple[float, float] | None = None) -> NightState:
        """Build the state at the start of a night, at ``start_pose`` or the bootstrap pose."""
        az, el = BOOTSTRAP_POSE if start_pose is None else start_pose
        return cls(t=start_time, az=float(az), el=float(el))

    @classmethod
    def from_timeline(
        cls,
        timeline: ObservingTimeline,
        t: Time | str,
        *,
        selection: SelectionRule | None = None,
    ) -> NightState:
        """Rebuild the state of a planned night at ``t`` from its timeline.

        Every block that ended by ``t`` is taken as executed as recorded:
        the pose is where the last of them left the telescope (the night's
        start pose when none has), the detector operations and passes are
        stamped on the calibration record at each block's start, as
        :func:`~fyst_trajectories.overhead.commit_visit` stamps them, and
        the scan counter counts the pass blocks. The deferrals and drops
        recorded by ``t`` are restored, a deferral being cleared by a later
        pass of its body; a record made at ``t`` itself counts, so the
        state is the latest the night held at that instant. The scripted
        cursor is the number of visits done plus the scripted entries set
        aside by ``t``, whatever the rule. Under a
        :class:`~fyst_trajectories.overhead.ScriptedSelection` with an
        entry still to run, the wait timer starts at the first block that
        starts at or after the last script event (the last entry set aside,
        the end of the last pass, or the start of the night) and before
        ``t``, whether or not that block has ended.

        The replay assumes the night ran as planned up to ``t``, and that
        each pass is one ``planet_cal`` block, the first of a visit
        carrying ``pass_index`` 0 in its ``scan_params``, as
        :func:`~fyst_trajectories.overhead.plan_visit` emits them; a pass
        block without ``scan_params`` counts as a visit of its own. A visit
        planner whose plans carry no ``passes`` therefore resumes with a
        scan counter that counts their pass blocks.

        Parameters
        ----------
        timeline : ObservingTimeline
            A night planned by
            :func:`~fyst_trajectories.overhead.plan_calibration_night`, in
            memory or read back from ECSV.
        t : Time or str
            The time to resume at, held in UTC (an ISO string is read as
            UTC). Times are compared to the millisecond, the precision the
            ECSV stores.
        selection : SelectionRule, optional
            The rule the night was planned with; default
            :func:`~fyst_trajectories.overhead.select_priority`. Its name
            must match the recorded one, and a scripted rule must be the
            night's own script for its cursor to be meaningful.

        Returns
        -------
        NightState
            The state at ``t``.

        Raises
        ------
        KeyError
            If the timeline carries no calibration-night record.
        ValueError
            If the record's schema version is not supported, the night has
            no usable interval, ``t`` is before its usable start,
            ``selection`` is not the rule the night recorded, the recorded
            policy is one
            :class:`~fyst_trajectories.overhead.CalibrationNightPolicy`
            refuses, or an entry set aside carries no time (a record written
            before the time was recorded).

        Notes
        -----
        The timeline does not say whether a block after the last script
        event was spent waiting for the current scripted entry or in a
        visit, so the wait timer is dated from the first such block either
        way, and in two cases the resumed state holds a timer the driver
        did not. When a visit
        began straight after that event and ``t`` falls inside it, the
        resumed ``script_waiting_since`` is the start of the visit's first
        block, while the driver held no timer there, having chosen the
        entry at once. When a visit has several passes and ``t`` falls
        after the end of its first pass and before the end of its last,
        the visit is counted as done and the end of its last finished pass
        is the last script event, so the next entry's timer is the start of
        the block after that pass, while the driver had not yet reached
        that entry. If the current entry then turns out to be unavailable,
        the resumed night counts its wait from that start rather than from
        ``t``, and so may set the entry aside sooner than a night that
        started waiting at ``t``.
        """
        meta = read_calibration_night_metadata(timeline)
        t = _utc_instant(t)
        if meta["usable_interval"] is None:
            raise ValueError("the night has no usable interval to resume")
        usable_start = Time(meta["usable_interval"][0], scale="utc")
        if (t - usable_start).to_value("s") < -_RESUME_TOL_SEC:
            raise ValueError(f"t {t.iso} is before the night's usable start {usable_start.iso}")
        rule: SelectionRule = select_priority if selection is None else selection
        if _rule_name(rule) != meta["selection"]:
            raise ValueError(
                f"the night was planned with the selection rule {meta['selection']!r}, "
                f"not {_rule_name(rule)!r}; pass the night's own rule"
            )
        retry_after = _policy_from_record(meta["policy"]).retry_after_seconds

        done = tuple(b for b in timeline.blocks if _by(b.t_stop, t))
        if done:
            az, el = float(done[-1].end_pose_az), float(done[-1].elevation)
        else:
            az, el = (float(v) for v in meta["start_pose"])
        passes = [b for b in done if _is_pass(b)]

        deferred_at: dict[str, tuple[Time, DeferralReason]] = {}
        for record in meta["deferrals"]:
            at = Time(record["at"], scale="utc")
            if _by(at, t):
                deferred_at[record["body"]] = (at, DeferralReason(record["reason"]))
        deferred = {
            body: (at + TimeDelta(retry_after, format="sec"), reason)
            for body, (at, reason) in deferred_at.items()
            if not any(
                b.metadata.get("target") == body
                and (b.t_start - at).to_value("s") >= -_RESUME_TOL_SEC
                for b in passes
            )
        }
        dropped = {
            record["body"]: DeferralReason(record["reason"])
            for record in meta["drops"]
            if _by(Time(record["at"], scale="utc"), t)
        }

        aside: list[tuple[str, ScanOverrides]] = []
        aside_at: list[Time] = []
        for record in meta["unplaced"]:
            if "at" not in record:
                raise ValueError(
                    "the night's record does not say when its scripted entries were set "
                    "aside, so the script position cannot be rebuilt"
                )
            at = Time(record["at"], scale="utc")
            if _by(at, t):
                aside.append((record["body"], ScanOverrides(**record["overrides"])))
                aside_at.append(at)
        script_index = sum(1 for b in passes if _opens_visit(b)) + len(aside)

        waiting_since: Time | None = None
        if isinstance(rule, ScriptedSelection) and script_index < len(rule.entries):
            events = [usable_start, *aside_at[-1:], *(b.t_stop for b in passes[-1:])]
            last_event = max(events, key=lambda e: (e - usable_start).to_value("s"))
            waiting_since = next(
                (
                    b.t_start
                    for b in timeline.blocks
                    if (b.t_start - last_event).to_value("s") >= -_RESUME_TOL_SEC
                    and (b.t_start - t).to_value("s") < -_RESUME_TOL_SEC
                ),
                None,
            )

        return cls(
            t=t,
            az=az,
            el=el,
            blocks=done,
            cal_state=_fold_calibrations(CalibrationState(), done),
            deferred=deferred,
            dropped=dropped,
            script_index=script_index,
            script_waiting_since=waiting_since,
            unplaced=tuple(aside),
            scan_counter=len(passes),
        )

    def advanced(self, **changes: Any) -> NightState:
        """Return a copy with the given fields replaced."""
        return replace(self, **changes)

    @property
    def elapsed_blocks(self) -> int:
        """Number of blocks emitted so far."""
        return len(self.blocks)


@dataclass(frozen=True)
class NightContext:
    """The resolved inputs of a night; build it with :meth:`build`.

    Parameters
    ----------
    site : Site
        Observing site.
    coords : Coordinates
        Coordinate transformer for the site (vacuum frame).
    targets : tuple of str
        Bodies in the caller's priority order, lower case.
    policy : CalibrationNightPolicy
        The night's policy.
    tables : mapping of str to ScanParameterTable
        Scan-parameter tables, resolved (never ``None``).
    overhead_model : OverheadModel
        Durations of the detector operations and the settle time.
    calibration_policy : CalibrationPolicy
        Cadences (the skydip cadence is read from here).
    sun_safe : SunSafePredicate
        Point-level Sun predicate.
    slew_safe : SlewSafePredicate
        Path-level Sun predicate, built from the site's axis limits by
        default.
    start_time, end_time : Time
        The usable interval (solar-gated) the night is planned inside, in
        UTC.
    requested_start, requested_end : Time
        The interval the caller asked for, in UTC.
    visit_planner : VisitPlanner
        The seam that plans one visit.
    ephem_cache : dict
        A cache of body positions keyed by ``(body, jd)``; not state.
    escape_cache : dict
        A cache of escape searches keyed by pose, time, elevation floor
        and settle time (see :func:`~fyst_trajectories.overhead.plan_escape`). The driver asks
        the same question once per candidate body per tick and each search
        costs a Sun ephemeris solve; not state.
    """

    site: Site
    coords: Coordinates
    targets: tuple[str, ...]
    policy: CalibrationNightPolicy
    tables: Mapping[str, ScanParameterTable]
    overhead_model: OverheadModel
    calibration_policy: CalibrationPolicy
    sun_safe: SunSafePredicate
    slew_safe: SlewSafePredicate
    start_time: Time
    end_time: Time
    requested_start: Time
    requested_end: Time
    visit_planner: VisitPlanner
    ephem_cache: dict = field(default_factory=dict, compare=False, repr=False)
    escape_cache: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def build(
        cls,
        targets: Sequence[str],
        site: Site,
        start_time: Time | str,
        end_time: Time | str,
        *,
        policy: CalibrationNightPolicy | None = None,
        tables: Mapping[str, ScanParameterTable] | None = None,
        overhead_model: OverheadModel | None = None,
        calibration_policy: CalibrationPolicy | None = None,
        sun_safe: SunSafePredicate | None = None,
        slew_safe: SlewSafePredicate | None = None,
        visit_planner: VisitPlanner | None = None,
    ) -> NightContext:
        """Resolve every input to a concrete object and gate the window on the Sun.

        Parameters
        ----------
        targets : sequence of str
            Bodies in priority order; each must have a table, either under
            its own name or ``"default"``. Names are compared in lower
            case, and a body may appear only once: a night visits each
            target repeatedly under the selection rule, so a repeated
            name buys nothing and would double-count the body in
            :func:`~fyst_trajectories.overhead.summarize_calibration_night`.
        site : Site
            Observing site.
        start_time, end_time : Time or str
            The requested window, held in UTC: an ISO string is read as
            UTC, and any ``Time`` is held as a UTC ``Time`` without a
            location and with astropy's default ``precision`` and
            ``out_subfmt`` (one in another scale is converted to UTC).
        policy : CalibrationNightPolicy, optional
            Default ``CalibrationNightPolicy()``.
        tables : mapping of str to ScanParameterTable, optional
            Default :data:`~fyst_trajectories.overhead.DEFAULT_SCAN_TABLES`.
        overhead_model : OverheadModel, optional
            Default ``OverheadModel()``.
        calibration_policy : CalibrationPolicy, optional
            Default ``CalibrationPolicy()``.
        sun_safe : SunSafePredicate, optional
            Default: the scalar Sun model built from the site's radii.
        slew_safe : SlewSafePredicate, optional
            Default: ``sun_safe`` swept along the direct slew path under
            the site's axis limits.
        visit_planner : VisitPlanner, optional
            Default :func:`~fyst_trajectories.overhead.plan_visit`.

        Returns
        -------
        NightContext
            The resolved context. When the Sun never sets inside the
            request, ``end_time`` equals ``start_time`` and the night is
            empty.

        Raises
        ------
        ValueError
            If ``targets`` is empty, names a body twice or names one with
            no ephemeris, the window is not ordered, or a body has no
            table.
        """
        from .step import plan_visit

        if not targets:
            raise ValueError("targets must name at least one body")
        names = tuple(str(t).lower() for t in targets)
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(f"targets must not repeat a body; {repeated} appear more than once")
        # Refuse a name the ephemeris cannot resolve here rather than at
        # the first tick, where it would surface from the coordinate layer
        # after the whole context is built.
        known = set(SOLAR_SYSTEM_BODIES) | set(SATELLITE_BODIES)
        unknown = sorted(set(names) - known)
        if unknown:
            raise ValueError(
                f"targets name bodies with no ephemeris: {unknown}; "
                f"supported bodies are {sorted(known)}"
            )
        t_start = _utc_instant(start_time)
        t_end = _utc_instant(end_time)
        if (t_end - t_start).to_value("s") <= 0.0:
            raise ValueError("end_time must be after start_time")
        policy = CalibrationNightPolicy() if policy is None else policy
        # Body names are matched in lower case throughout, so a table
        # mapping keyed "Uranus" must not silently fall through to the
        # shared "default" table.
        supplied = DEFAULT_SCAN_TABLES if tables is None else tables
        resolved_tables = {str(k).lower(): v for k, v in supplied.items()}
        for body in names:
            try:
                table_for(resolved_tables, body)
            except KeyError as exc:
                raise ValueError(str(exc)) from None
        overhead_model = OverheadModel() if overhead_model is None else overhead_model
        calibration_policy = (
            CalibrationPolicy() if calibration_policy is None else calibration_policy
        )
        if sun_safe is None:
            sun_safe = make_sun_safe("scalar", site=site)
        if slew_safe is None:
            slew_safe = _default_slew_safe(sun_safe, site)
        usable = usable_interval(site, t_start, t_end, policy.time_step)
        if usable is None:
            usable_start, usable_end = t_start, t_start
        else:
            usable_start, usable_end = usable
        return cls(
            site=site,
            coords=Coordinates(site),
            targets=names,
            policy=policy,
            tables=resolved_tables,
            overhead_model=overhead_model,
            calibration_policy=calibration_policy,
            sun_safe=sun_safe,
            slew_safe=slew_safe,
            start_time=usable_start,
            end_time=usable_end,
            requested_start=t_start,
            requested_end=t_end,
            visit_planner=plan_visit if visit_planner is None else visit_planner,
        )

    @classmethod
    def from_timeline(
        cls,
        timeline: ObservingTimeline,
        *,
        site: Site | None = None,
        sun_safe: SunSafePredicate | None = None,
        slew_safe: SlewSafePredicate | None = None,
        visit_planner: VisitPlanner | None = None,
    ) -> NightContext:
        """Rebuild the context a planned night was resolved from.

        The targets, policy, scan tables and requested window come from the
        night's record, the overhead model and calibration policy from the
        timeline, and the context is resolved as :meth:`build` resolves it,
        then checked against the record. A record written before the
        policy had ``use_table_throw`` is read with it True, since that
        night swept the table's throw.

        Parameters
        ----------
        timeline : ObservingTimeline
            A night planned by
            :func:`~fyst_trajectories.overhead.plan_calibration_night`, in
            memory or read back from ECSV.
        site : Site, optional
            Default ``timeline.site``. A timeline read back from ECSV
            carries the FYST axis limits whatever the night was planned on,
            so a night planned on other limits needs its site passed in.
        sun_safe : SunSafePredicate, optional
            Default as in :meth:`build`. The record keeps only the
            predicate's description, so a night planned on another model,
            the directional CAD zone for example, needs it passed in.
        slew_safe : SlewSafePredicate, optional
            Default as in :meth:`build`.
        visit_planner : VisitPlanner, optional
            Default :func:`~fyst_trajectories.overhead.plan_visit`.

        Returns
        -------
        NightContext
            The rebuilt context.

        Raises
        ------
        KeyError
            If the timeline carries no calibration-night record.
        ValueError
            If the record's schema version is not supported, the site's
            axis limits differ from those the record holds, the recorded
            policy is one
            :class:`~fyst_trajectories.overhead.CalibrationNightPolicy`
            refuses, a Sun predicate's description differs from the
            recorded one, or the usable interval resolved here differs from
            the recorded one by more than a millisecond.
        """
        meta = read_calibration_night_metadata(timeline)
        site = timeline.site if site is None else site
        # A record written before the limits were recorded skips this check.
        recorded_limits = meta.get("telescope_limits")
        limits = dataclasses.asdict(site.telescope_limits)
        if recorded_limits is not None and limits != recorded_limits:
            raise ValueError(
                f"the site's axis limits {limits} differ from those the night was planned "
                f"on, {recorded_limits}; pass the night's site"
            )
        requested_start, requested_end = meta["requested_window"]
        ctx = cls.build(
            meta["targets"],
            site,
            Time(requested_start, scale="utc"),
            Time(requested_end, scale="utc"),
            policy=_policy_from_record(meta["policy"]),
            tables=tables_from_record(meta["tables"]),
            overhead_model=timeline.overhead_model,
            calibration_policy=timeline.calibration_policy,
            sun_safe=sun_safe,
            slew_safe=slew_safe,
            visit_planner=visit_planner,
        )
        for name, predicate in (("sun_safe", ctx.sun_safe), ("slew_safe", ctx.slew_safe)):
            if _describe(predicate) != meta[name]:
                raise ValueError(
                    f"{name} is {_describe(predicate)!r}, but the night was planned with "
                    f"{meta[name]!r}; pass the night's predicate"
                )
        recorded = meta["usable_interval"]
        resolved = (ctx.start_time, ctx.end_time) if ctx.usable else None
        if (recorded is None) != (resolved is None) or (
            recorded is not None
            and resolved is not None
            and any(
                abs((Time(r, scale="utc") - x).to_value("s")) > _RESUME_TOL_SEC
                for r, x in zip(recorded, resolved)
            )
        ):
            shown = None if resolved is None else [x.iso for x in resolved]
            raise ValueError(
                f"the usable interval resolved here, {shown}, differs from the recorded "
                f"one, {recorded}"
            )
        return ctx

    def body_altaz(self, body: str, t: Time) -> tuple[float, float]:
        """Return the body's ``(az, el)`` in degrees at ``t``, cached per instant."""
        key = (body, round(float(t.jd), 9))
        hit = self.ephem_cache.get(key)
        if hit is None:
            az, el = self.coords.get_body_altaz(body, t)
            hit = (float(az), float(el))
            self.ephem_cache[key] = hit
        return hit

    @property
    def usable(self) -> bool:
        """Whether the request left any night to plan."""
        return (self.end_time - self.start_time).to_value("s") > 0.0
