"""The immutable state and the resolved context of a calibration night.

:class:`NightState` is what the step functions transform; it mirrors the
offline scheduler's state record (initial pose, ``advanced(**changes)``)
without sharing it, and carries the body queue the scheduler lacks: the
deferrals with their retry times, the drops, the scripted-selection
cursor and the entries it set aside. :class:`NightContext` holds every
resolved input so a step never reads a default at run time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from astropy.time import Time

from ...coordinates import SATELLITE_BODIES, SOLAR_SYSTEM_BODIES, Coordinates
from ...site import Site
from ...sun_models import make_sun_safe
from ..calibration_state import CalibrationState
from ..models import CalibrationPolicy, OverheadModel, TimelineBlock
from ..transitions import DeferralReason, _default_slew_safe
from .helpers import usable_interval
from .policy import CalibrationNightPolicy, ScanOverrides
from .tables import DEFAULT_SCAN_TABLES, ScanParameterTable, table_for

if TYPE_CHECKING:
    from ...dispatch import SlewSafePredicate, SunSafePredicate
    from .step import VisitPlan

__all__ = [
    "BOOTSTRAP_POSE",
    "NightContext",
    "NightState",
    "VisitPlanner",
]

#: The pose a night starts from when none is given: the offline scheduler's
#: bootstrap, roughly the southern horizon at mid elevation.
BOOTSTRAP_POSE: tuple[float, float] = (180.0, 50.0)


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
        The current time.
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

    @classmethod
    def initial(cls, start_time: Time, start_pose: tuple[float, float] | None = None) -> NightState:
        """Build the state at the start of a night, at ``start_pose`` or the bootstrap pose."""
        az, el = BOOTSTRAP_POSE if start_pose is None else start_pose
        return cls(t=start_time, az=float(az), el=float(el))

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
        The usable interval (solar-gated) the night is planned inside.
    requested_start, requested_end : Time
        The interval the caller asked for.
    visit_planner : VisitPlanner
        The seam that plans one visit.
    ephem_cache : dict
        A cache of body positions keyed by ``(body, jd)``; not state.
    escape_cache : dict
        A cache of escape searches keyed by pose, time, elevation floor
        and settle time (see ``_moves.plan_escape_move``). The driver asks
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
            The requested window (ISO UTC strings accepted).
        policy, tables, overhead_model, calibration_policy : optional
            Defaults: :class:`CalibrationNightPolicy()`,
            :data:`~fyst_trajectories.overhead.DEFAULT_SCAN_TABLES`,
            :class:`~fyst_trajectories.overhead.OverheadModel()`,
            :class:`~fyst_trajectories.overhead.CalibrationPolicy()`.
        sun_safe, slew_safe : optional
            Defaults: the scalar Sun model built from the site's radii,
            and its sweep along the direct slew path under the site's
            axis limits.
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
        # the first tick, where it surfaced from the coordinate layer
        # after the whole context had been built.
        known = set(SOLAR_SYSTEM_BODIES) | set(SATELLITE_BODIES)
        unknown = sorted(set(names) - known)
        if unknown:
            raise ValueError(
                f"targets name bodies with no ephemeris: {unknown}; "
                f"supported bodies are {sorted(known)}"
            )
        t_start = Time(start_time, scale="utc") if isinstance(start_time, str) else start_time
        t_end = Time(end_time, scale="utc") if isinstance(end_time, str) else end_time
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
