"""Scheduler state and context dataclasses.

The :class:`SchedulerState` is an immutable snapshot of scheduler
progress that is evolved between phases via
:func:`dataclasses.replace`. :class:`SchedulerContext` bundles the
configuration (site, patches, overhead/calibration policy, constraints,
time window) that every phase reads, with four carve-outs: the ``ce_corridors``,
``escapes``, ``ce_scan_starts`` and ``science_probes`` memos, which are written
as solves and builds are made and are documented on the fields themselves.
Nothing else in the context changes once it is built.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from astropy.time import Time

from ...coordinates import Coordinates
from ...exceptions import OffsetInversionError
from ...sun_models import make_sun_safe
from ...trajectory_utils import GO_TCS_MIN_SAMPLE_INTERVAL_SEC
from ..constraints import ElevationConstraint
from ..models import CalibrationPolicy, OverheadModel
from ..transitions import _default_slew_safe
from ..utils import _utc_instant
from .helpers import _ce_pass_refusal, _ce_scan_pose, _default_constraints, _pong_period

if TYPE_CHECKING:
    from ...site import Site
    from ...sun_protocols import SlewSafePredicate, SunSafePredicate
    from ..calibration_state import CalibrationState
    from ..constraints import Constraint
    from ..models import ObservingPatch

__all__ = ["SchedulerContext", "SchedulerState"]

#: The pose a schedule starts from when none is given: roughly the southern
#: horizon at mid elevation. The calibration-night planner starts there too.
BOOTSTRAP_POSE: tuple[float, float] = (180.0, 50.0)


@dataclass(frozen=True)
class SchedulerState:
    """Immutable scheduler state; evolved via :func:`dataclasses.replace`.

    Attributes
    ----------
    current_time : Time
        UTC timestamp of the scheduler's current position, held as a UTC
        ``Time`` without a location and with astropy's default
        ``precision`` and ``out_subfmt`` (one in another scale is converted
        to UTC).
    current_az : float
        Telescope azimuth (deg) at ``current_time``.
    current_el : float
        Telescope elevation (deg) at ``current_time``.
    cal_state : CalibrationState
        Cadence-tracking state for each calibration type. Immutable;
        replaced whenever a calibration fires.
    scan_counter : int
        Monotonically increasing counter used as ``scan_index`` on
        emitted blocks.
    """

    current_time: Time
    current_az: float
    current_el: float
    cal_state: CalibrationState
    scan_counter: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "current_time", _utc_instant(self.current_time))

    @classmethod
    def initial(cls, start_time: Time, cal_state: CalibrationState) -> SchedulerState:
        """Build the scheduler's initial state.

        The ``(current_az=180.0, current_el=50.0)`` initialization is a
        bare bootstrap: the telescope is assumed to start roughly pointed
        at the southern horizon at a mid-sky elevation. These values are
        the recorded pose until the first emitted slew or science scan
        replaces them, so any calibration or idle block emitted before
        that point is stamped at this position. The bootstrap azimuth
        also seeds the cable-wrap frame: the first selected target is
        placed on the representative nearest it, and each later target
        relative to the pose before it.
        """
        return cls(
            current_time=start_time,
            current_az=BOOTSTRAP_POSE[0],
            current_el=BOOTSTRAP_POSE[1],
            cal_state=cal_state,
            scan_counter=0,
        )

    def advanced(self, **changes) -> SchedulerState:
        """Return a copy of this state with ``changes`` applied."""
        return replace(self, **changes)


def _ce_boresight_outside(patch: ObservingPatch, site: Site) -> str | None:
    """Say why a constant-elevation patch's boresight cannot scan, or return None.

    The planner sweeps the patch's detector at its ``elevation``; the
    boresight that holds it there scans at one elevation, which has to lie
    within the site's elevation limits, the lower limit included.
    """
    offset = patch.detector_offset
    assert offset is not None and patch.elevation is not None
    name = offset.name if offset.name else "unnamed"
    where = (
        f"{patch.name} with the offset {name} (dx={offset.dx} arcmin, dy={offset.dy} arcmin) "
        f"at the detector's el {patch.elevation:.3f}"
    )
    try:
        _, elevation = _ce_scan_pose(patch, site)
    except OffsetInversionError:
        return f"{where}: no boresight places the detector there"
    limits = site.telescope_limits.elevation
    if limits.min <= elevation <= limits.max:
        return None
    return (
        f"{where} needs the boresight at el {elevation:.3f}, outside [{limits.min}, {limits.max}]"
    )


@dataclass(frozen=True)
class SchedulerContext:
    """Scheduling context passed to every phase.

    Holds all configuration that remains constant across the entire
    timeline: patches, site, coordinate transform, overhead/calibration
    models, constraint list, time window, and idle time step. A phase
    reads it and does not change it, except for the ``ce_corridors``,
    ``escapes``, ``ce_scan_starts`` and ``science_probes`` memos it fills
    as crossing passes and escapes are solved and scans are built; those
    fields are within-run caches of results the inputs already determine,
    so two runs of the same context still plan the same night.

    Attributes
    ----------
    patches : list of ObservingPatch
        The candidate sky regions the scheduler selects among.
    site : Site
        Telescope site configuration.
    coords : Coordinates
        Coordinate transform bound to ``site``.
    overhead_model : OverheadModel
        Per-activity durations and scan split thresholds.
    calibration_policy : CalibrationPolicy
        Calibration cadences.
    constraints : list of Constraint
        Patch-selection constraints, scored per candidate each tick.
    start_time, end_time : Time
        The timeline window: :meth:`build` holds each as a UTC ``Time``
        without a location and with astropy's default ``precision`` and
        ``out_subfmt`` (one in another scale is converted to UTC).
    time_step : float
        Idle-tick step in seconds.
    """

    patches: list[ObservingPatch]
    site: Site
    coords: Coordinates
    overhead_model: OverheadModel
    calibration_policy: CalibrationPolicy
    constraints: list[Constraint]
    start_time: Time
    end_time: Time
    time_step: float
    #: Injected sun-safety model (:class:`~fyst_trajectories.sun_protocols.SunSafePredicate`,
    #: e.g. from :func:`~fyst_trajectories.sun_models.make_sun_safe`) driving
    #: the duration clips, the slew and escape checks and the planet-calibration
    #: planner; ``None`` keeps the scalar site radius. The Sun *constraint* is
    #: bound at construction time (see ``build``).
    sun_safe: SunSafePredicate | None = None
    #: Path-level Sun model (:class:`~fyst_trajectories.sun_protocols.SlewSafePredicate`)
    #: swept along a slew. Built once in ``build`` from ``sun_safe`` and the
    #: site's axis limits, so the transition and escape planners are handed
    #: one model instead of rebuilding it on every call.
    slew_safe: SlewSafePredicate | None = None
    #: Per-run memo of constant-elevation crossing-pass solves, keyed
    #: ``(patch_name, elevation, rising)`` (name, float, bool), or
    #: ``(patch_name, elevation, "lsa")`` for an ``lsa_window`` patch, with
    #: values ``("ok", t_open, t_close)`` or ``("miss", solved_from)``.
    #: Written only by the scheduler's crossing-pass solve; a cache, not state.
    ce_corridors: dict = field(default_factory=dict)
    #: Per-run memo of escape searches, keyed by pose, time, elevation
    #: floor, settle time and hold (see :func:`~fyst_trajectories.overhead.plan_escape`).
    #: The loop asks about one pose and time from several places per tick
    #: and each search costs a Sun ephemeris solve; a cache, not state.
    escapes: dict = field(default_factory=dict)
    #: Per-run memo of the scans constant-elevation visits are cut from, keyed
    #: by patch name, elevation, crossing half, the visit's anchor (the ISO
    #: ``t0_scan``) and the instant the scan starts, as its ``(jd1, jd2)``
    #: pair. A value is the trajectory a science block from that instant to
    #: the schedule's end rebuilds to, or ``None`` when the planner refuses
    #: it. The slew phase builds one per round of its slew solve, each round
    #: moving the start, and reads its first sample for the slew's target;
    #: the science phase reuses the last round's for the visit's first
    #: subscan and last block. A patch with an azimuth window keeps its
    #: window scans here too, keyed by patch name, start and leg count. A
    #: cache, not state.
    ce_scan_starts: dict = field(default_factory=dict)
    #: Per-visit memo of pong and daisy subscan builds the booking screen
    #: passed, keyed by the block's window, elevation and metadata. The slew phase empties
    #: it, then builds and screens the visit's shortest first subscan before
    #: booking the slew; the science phase takes that build out when it plans
    #: the same subscan. A cache, not state.
    science_probes: dict = field(default_factory=dict)

    @property
    def el_floor(self) -> float:
        """Lowest elevation the schedule uses, in degrees.

        The tightest ``el_min`` among the elevation constraints, or the
        site's own elevation limit when none is configured. An escape
        searches down to this floor rather than to the mount limit, so it
        never parks the telescope below the sky the selection phase is
        willing to observe.
        """
        floors = [c.el_min for c in self.constraints if isinstance(c, ElevationConstraint)]
        return max(floors) if floors else self.site.telescope_limits.elevation.min

    @classmethod
    def build(
        cls,
        patches: list[ObservingPatch],
        site: Site,
        start_time: Time,
        end_time: Time,
        overhead_model: OverheadModel | None = None,
        calibration_policy: CalibrationPolicy | None = None,
        constraints: list[Constraint] | None = None,
        time_step: float = 300.0,
        sun_safe: SunSafePredicate | None = None,
        slew_safe: SlewSafePredicate | None = None,
    ) -> SchedulerContext:
        """Assemble a context, filling in default overhead/policy/constraints.

        ``sun_safe`` is the point-level Sun model the whole run uses: the
        default constraint set (when ``constraints`` is None), the
        scan-duration clips, the slew and escape checks, the
        planet-calibration planner, and the path model built from it. A
        caller supplying an explicit ``constraints`` list owns its Sun
        constraint; ``sun_safe`` still drives everything else. ``slew_safe``
        defaults to that point model swept along the direct path under the
        site's axis limits, built once here rather than per call.

        Parameters
        ----------
        patches : list of ObservingPatch
            The candidate sky regions; names must be unique.
        site : Site
            Telescope site configuration.
        start_time, end_time : Time
            The timeline window, held in UTC.
        overhead_model : OverheadModel, optional
            Default ``OverheadModel()``.
        calibration_policy : CalibrationPolicy, optional
            Default ``CalibrationPolicy()``.
        constraints : list of Constraint, optional
            Default: the site's elevation limits and, when the site has Sun
            avoidance enabled, a Sun constraint running ``sun_safe`` or the
            site's scalar radius.
        time_step : float, optional
            Idle-tick step in seconds, positive. Default 300.
        sun_safe : SunSafePredicate, optional
            The point-level Sun model the run uses (see above). Default
            ``None``, the site's scalar exclusion radius.
        slew_safe : SlewSafePredicate, optional
            The path-level Sun model. Default ``None`` builds it from
            ``sun_safe`` (see above).

        Raises
        ------
        ValueError
            If ``time_step`` is not positive, two patches share a name, a
            patch's ``scan_params["timestep"]`` is under the Go TCS
            ``/path`` minimum sample interval
            (``GO_TCS_MIN_SAMPLE_INTERVAL_SEC``, 0.05 s), a
            constant-elevation patch has no ``elevation`` or one at
            which the boresight that holds its ``detector_offset`` there
            lies outside the site's elevation limits (the lower limit
            itself is inside) or cannot be solved, a constant-elevation
            patch can never have a pass (the planner's own solve finds no
            whole pass for it from any instant of the sidereal day from
            ``start_time``, on the side it asks for or on either side when
            it names none, or in its ``lsa_window``: a field with no
            leading and trailing RA edges, spanning a whole turn of RA or
            reaching a celestial pole, one whose edges never cross the
            patch's ``elevation``, or a pass longer than the planner's
            12 h search), or a pong patch's
            pattern period exceeds what one subscan can hold. Names
            identify a patch in the corridor
            memo and in every emitted block, so they have to be unique across
            one schedule. A pong subscan is a whole number of periods inside
            ``max_scan_duration``, which at ``retune_cadence=0`` also holds
            the retune booked before every subscan, so a longer period could
            never be scanned.
        """
        if not time_step > 0:
            raise ValueError(f"time_step must be positive, got {time_step}")

        # Patch names key the constant-elevation corridor memo, so two
        # patches sharing one name would read each other's crossing solve and
        # observe the wrong field. ``ObservingPatch`` states the uniqueness
        # precondition; this is where the schedule can actually check it.
        counts = Counter(p.name for p in patches)
        duplicates = sorted(name for name, n in counts.items() if n > 1)
        if duplicates:
            raise ValueError(
                f"Patch names must be unique within a schedule; repeated: {duplicates}."
            )
        # The Go TCS /path receiver refuses samples closer than its minimum
        # interval at any time of night, so such a patch could never be run.
        too_fine = [
            f"{p.name} (timestep {p.scan_params['timestep']} s)"
            for p in patches
            if p.scan_params.get("timestep") is not None
            and float(p.scan_params["timestep"]) < GO_TCS_MIN_SAMPLE_INTERVAL_SEC
        ]
        if too_fine:
            raise ValueError(
                "patches whose sample interval is under the Go TCS /path minimum of "
                f"{GO_TCS_MIN_SAMPLE_INTERVAL_SEC} s: {'; '.join(too_fine)}"
            )
        unpinned = sorted(
            p.name for p in patches if p.scan_type == "constant_el" and p.elevation is None
        )
        if unpinned:
            raise ValueError(
                "constant-elevation patches need a pinned elevation in the offline "
                "scheduler, whose crossing-pass gate solves at that elevation; "
                f"unpinned: {unpinned}"
            )
        outside = [
            why
            for p in patches
            if p.scan_type == "constant_el" and p.detector_offset is not None
            for why in [_ce_boresight_outside(p, site)]
            if why is not None
        ]
        if outside:
            raise ValueError(
                "constant-elevation patches whose detector offset puts the boresight outside "
                f"the elevation limits: {'; '.join(outside)}"
            )
        coords = Coordinates(site)
        start_time = _utc_instant(start_time)
        passless = [
            why
            for p in patches
            if p.scan_type == "constant_el"
            for why in [_ce_pass_refusal(p, coords, start_time)]
            if why is not None
        ]
        if passless:
            raise ValueError(
                f"constant-elevation patches that can never have a pass: {'; '.join(passless)}"
            )

        if overhead_model is None:
            overhead_model = OverheadModel()
        if calibration_policy is None:
            calibration_policy = CalibrationPolicy()
        # A pong subscan is a whole number of pattern periods inside
        # max_scan_duration; at cadence 0 the retune booked before every
        # subscan shares that budget.
        boundary_retune = (
            overhead_model.retune_duration if calibration_policy.retune_cadence == 0.0 else 0.0
        )
        room = overhead_model.max_scan_duration - boundary_retune
        periods = {p.name: _pong_period(p) for p in patches if p.scan_type == "pong"}
        too_long = sorted(
            f"{name} ({period:.1f} s)" for name, period in periods.items() if period > room
        )
        if too_long:
            why = (
                "max_scan_duration less the retune a zero retune_cadence books before every subscan"
                if boundary_retune
                else "max_scan_duration"
            )
            raise ValueError(
                f"pong patches need a pattern period that fits the {room:.1f} s one subscan "
                f"can hold ({why}); too long: {too_long}"
            )
        if constraints is None:
            constraints = _default_constraints(site, sun_safe=sun_safe)
        if slew_safe is None:
            point = make_sun_safe("scalar", site=site) if sun_safe is None else sun_safe
            slew_safe = _default_slew_safe(point, site)
        return cls(
            patches=patches,
            site=site,
            coords=coords,
            overhead_model=overhead_model,
            calibration_policy=calibration_policy,
            constraints=constraints,
            start_time=start_time,
            end_time=_utc_instant(end_time),
            time_step=time_step,
            sun_safe=sun_safe,
            slew_safe=slew_safe,
        )
