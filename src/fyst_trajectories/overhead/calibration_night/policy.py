"""Policies, per-visit overrides and the night's metadata payload.

The policy objects hold what the instrument and operations teams decide
for a commissioning night: the observing floor, the sweep speed and
acceleration, the footprint, how long a solved crossing may take, the
idle tick and the retry interval. Per-visit :class:`ScanOverrides` win
over the policy and the tables. :class:`CalibrationNightMetadata` is the
record a planned night stores in the timeline header so it can be read
back and re-planned from.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from typing import Any, TypedDict

from ...trajectory_utils import GO_TCS_AZ_MAX_ACCELERATION, GO_TCS_AZ_MAX_VELOCITY
from .._execution_screen import _QUINTIC_PEAK_ACCEL_FACTOR
from ..models import ObservingTimeline
from ..transitions import DeferralReason
from ..utils import _require_module_tag
from .tables import ElevationBin, ScanParameterTable

__all__ = [
    "CALNIGHT_SCHEMA_VERSION",
    "CalibrationNightMetadata",
    "CalibrationNightPolicy",
    "DeferralReason",
    "ScanOverrides",
    "TuningPolicy",
    "encode_calibration_night_metadata",
    "read_calibration_night_metadata",
    "tables_as_record",
    "tables_from_record",
]

#: Version of the header payload written by :func:`encode_calibration_night_metadata`.
CALNIGHT_SCHEMA_VERSION = 1

_META_VERSION_KEY = "calnight_schema_version"
_META_PAYLOAD_KEY = "calnight_json"


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


def _non_negative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _within_axis_ceilings(az_speed: float | None, az_accel: float | None) -> None:
    """Refuse a sweep speed or acceleration no pass can be commanded with.

    A pass sweeps at ``az_speed`` and turns with a quintic profile whose
    peak is ``_QUINTIC_PEAK_ACCEL_FACTOR`` times ``az_accel``; the Go TCS
    command layer refuses a path past ``GO_TCS_AZ_MAX_VELOCITY`` or
    ``GO_TCS_AZ_MAX_ACCELERATION`` (strictly greater, as the booking
    screen judges), so such a value would be refused at every pass.

    Parameters
    ----------
    az_speed : float or None
        Sweep speed in deg/s; ``None`` is not checked.
    az_accel : float or None
        Turnaround acceleration in deg/s^2; ``None`` is not checked.

    Raises
    ------
    ValueError
        If ``az_speed`` exceeds ``GO_TCS_AZ_MAX_VELOCITY`` or
        ``_QUINTIC_PEAK_ACCEL_FACTOR`` times ``az_accel`` exceeds
        ``GO_TCS_AZ_MAX_ACCELERATION``.
    """
    if az_speed is not None and az_speed > GO_TCS_AZ_MAX_VELOCITY:
        raise ValueError(
            f"az_speed {az_speed} deg/s exceeds the Go TCS azimuth velocity limit "
            f"of {GO_TCS_AZ_MAX_VELOCITY} deg/s"
        )
    if az_accel is not None and _QUINTIC_PEAK_ACCEL_FACTOR * az_accel > GO_TCS_AZ_MAX_ACCELERATION:
        raise ValueError(
            f"az_accel {az_accel} deg/s^2 gives a turnaround peak of "
            f"{_QUINTIC_PEAK_ACCEL_FACTOR * az_accel} deg/s^2 "
            f"({_QUINTIC_PEAK_ACCEL_FACTOR} x az_accel), which exceeds the Go TCS azimuth "
            f"acceleration limit of {GO_TCS_AZ_MAX_ACCELERATION} deg/s^2"
        )


@dataclass(frozen=True)
class ScanOverrides:
    """Per-visit scan-geometry overrides; a set value wins over table and policy.

    Parameters
    ----------
    az_speed : float, optional
        Per-leg azimuth speed in deg/s.
    az_accel : float, optional
        Azimuth acceleration in deg/s^2.
    az_throw : float, optional
        Swept azimuth window in degrees.
    dwell : float, optional
        Time on source in seconds.

    Raises
    ------
    ValueError
        If a set value is not a positive, finite number, or ``az_speed``
        or ``az_accel`` is one no pass can be commanded with (as
        :class:`CalibrationNightPolicy` refuses it).
    """

    az_speed: float | None = None
    az_accel: float | None = None
    az_throw: float | None = None
    dwell: float | None = None

    def __post_init__(self) -> None:
        for name in ("az_speed", "az_accel", "az_throw", "dwell"):
            value = getattr(self, name)
            if value is not None:
                _positive(name, value)
        _within_axis_ceilings(self.az_speed, self.az_accel)

    def as_record(self) -> dict[str, float]:
        """Return the set overrides as a plain dict of JSON builtins."""
        return {
            name: float(value)
            for name, value in dataclasses.asdict(self).items()
            if value is not None
        }


@dataclass(frozen=True)
class TuningPolicy:
    """Detector operations reserved around the scans.

    The retune and skydip durations come from
    :class:`~fyst_trajectories.overhead.OverheadModel` and the skydip
    cadence from :class:`~fyst_trajectories.overhead.CalibrationPolicy`;
    this policy decides where the operations go, and how long the
    detector-finding one takes.

    Parameters
    ----------
    retune_before_each_block : bool, optional
        Reserve a retune before every scan block, at the target pose after
        the slew. Default True.
    find_detectors_at_start : bool, optional
        Reserve one detector-finding operation at the start of the night,
        before the first retune. Default True.
    find_detectors_duration : float, optional
        Duration of that operation in seconds. A placeholder pending an
        instrument-team value; default 300.
    skydip_after_retune : bool, optional
        Reserve a skydip immediately after the retune before each scan
        block, at the same pose, so the dispatch lead follows that skydip;
        a retune of zero duration is not scheduled, and then neither is
        this skydip. It is in addition to the skydip reserved before the
        retune whenever the cadence says one is due, and like any skydip it
        restarts the cadence. Needs ``retune_before_each_block``. Default
        False.

    Raises
    ------
    ValueError
        If ``find_detectors_duration`` is negative or not finite, or
        ``skydip_after_retune`` is set without ``retune_before_each_block``.
    """

    retune_before_each_block: bool = True
    find_detectors_at_start: bool = True
    find_detectors_duration: float = 300.0
    skydip_after_retune: bool = False

    def __post_init__(self) -> None:
        _non_negative("find_detectors_duration", self.find_detectors_duration)
        if self.skydip_after_retune and not self.retune_before_each_block:
            raise ValueError(
                "skydip_after_retune needs retune_before_each_block: the skydip follows the retune"
            )


@dataclass(frozen=True)
class CalibrationNightPolicy:
    """What a commissioning night may do and how its scans are shaped.

    Parameters
    ----------
    el_min : float, optional
        Observing floor in degrees for the pass's boresight elevation.
        Default 30.
    max_pass_seconds : float, optional
        A solved footprint crossing longer than this is refused as too
        slow for now (the body is retried later). Default 1200.
    az_speed : float, optional
        Per-leg azimuth speed in deg/s. Default 1.5, a commissioning value
        rather than a ratified limit.
    az_accel : float, optional
        Nominal azimuth acceleration in deg/s^2. Default 1.0, a
        commissioning value rather than a ratified limit; the quintic
        turnaround peaks at 1.5 times this value, 1.5 deg/s^2 at the
        default.
    footprint : str, optional
        The Prime-Cam module bay the passes are planned on, by any name
        :func:`~fyst_trajectories.primecam.get_primecam_offset` takes
        (``"IM0"`` .. ``"IM6"`` in any case, or ``"c"`` or ``"center"`` for
        ``IM0``). A pass's dispatch dict names the on-axis bay ``"c"``,
        whichever spelling the policy used, and a ring bay by its ``IM``
        name (``"IM1"`` .. ``"IM6"``), since the execution layer compares
        the name as a string; the night's record keeps the name as given.
        Default ``"IM0"``; a name of any other bay is accepted for
        simulation and a warning recorded (in
        ``VisitPlan.warnings`` and the night's metadata), since the
        execution layer accepts centred footprints only.
    footprint_margin : float, optional
        On-sky margin in degrees added on every side of the footprint
        before the crossing is solved. Default 0.
    n_passes : int, optional
        Passes per visit. Default 1.
    use_table_dwell : bool, optional
        Apply the table's reference dwell instead of the solved crossing.
        Default False; a reference dwell longer than the solved crossing
        is not applied (the full crossing is scanned and a warning
        recorded).
    use_table_throw : bool, optional
        Sweep the table's azimuth throw instead of the throw solved from
        the footprint. Default False: each pass sweeps the footprint's
        azimuth extent at its elevation (the margined module's width over
        the cosine of the elevation), with no padding. When True, the
        table's throw is swept at and above the table's lowest bin
        (extrapolated above its top bin) and, below it, the kernel's
        default padded throw.
    dispatch_lead_seconds : float, optional
        Seconds reserved between the end of the detector operations and
        the start of each pass. The planner's search for the pass, whose
        start the dispatch sheet prints as ``scheduled_t0_unix``, begins
        the lead after the operations' end as estimated before the slew is
        planned (the estimate prices the slew to where the body will be at
        the anchor and adds the settle), so it sits the lead after the
        operations to within about a second for a single pass of the
        centre module (a multi-pass visit or a ring module, planned for
        simulation, up to a few seconds), while the pass itself, which
        starts the kernel's own offset after the search start, never
        starts sooner than the lead after them. The
        execution layer floors a dispatch's start to
        ``SCAN_DISPATCH_BUFFER_SEC`` after the dispatch (10 s in PCS at the
        revision this library is checked against) and searches for the
        pass from the later of that floor and ``scheduled_t0_unix``: a
        later floor finds the same crossing to within some tens of
        milliseconds, until it passes the pass start, where the task
        refuses the pass. Default 20: PCS's 10 s floor plus about ten
        seconds for the dispatch path (the completion of the operation
        before the pass, the OCS round trips and the script's own
        latency). A later pass of a multi-pass visit is kept only when it
        starts at least the step to it plus the lead after the previous
        kept pass ends. 0 reserves nothing.
    min_pass_seconds : float, optional
        The night ends when less than this remains. Default 60.
    retry_after_seconds : float, optional
        How long a deferred body waits before it is a candidate again.
        Default 300.
    max_wait_seconds : float, optional
        How long a scripted entry waits for its body before it is set
        aside as unplaced. Default 3600.
    time_step : float, optional
        Idle tick in seconds. Default 300.
    moon_min_separation : float, optional
        Minimum separation from the Moon in degrees; ``None`` (default)
        disables the check.
    allow_detour : bool, optional
        Let a Sun-blocked direct slew be replaced by a two-leg detour.
        Default False.
    tuning : TuningPolicy, optional
        Where the detector operations go.

    Raises
    ------
    ValueError
        If ``el_min`` is outside [0, 90), a duration, ``az_speed`` or
        ``az_accel`` is not a positive, finite number, ``az_speed``
        exceeds the Go TCS azimuth velocity limit
        (:data:`~fyst_trajectories.trajectory_utils.GO_TCS_AZ_MAX_VELOCITY`,
        3 deg/s), 1.5 times ``az_accel`` (the turnaround's peak) exceeds
        its azimuth acceleration limit
        (:data:`~fyst_trajectories.trajectory_utils.GO_TCS_AZ_MAX_ACCELERATION`,
        6 deg/s^2), ``footprint_margin``, ``dispatch_lead_seconds`` or
        ``moon_min_separation`` is negative or not finite, ``n_passes``
        is below 1, or ``footprint`` is not a
        str naming one Prime-Cam module (for an unknown name the message
        lists the names it takes).
    """

    el_min: float = 30.0
    max_pass_seconds: float = 1200.0
    az_speed: float = 1.5
    az_accel: float = 1.0
    footprint: str = "IM0"
    footprint_margin: float = 0.0
    n_passes: int = 1
    use_table_dwell: bool = False
    use_table_throw: bool = False
    dispatch_lead_seconds: float = 20.0
    min_pass_seconds: float = 60.0
    retry_after_seconds: float = 300.0
    max_wait_seconds: float = 3600.0
    time_step: float = 300.0
    moon_min_separation: float | None = None
    allow_detour: bool = False
    tuning: TuningPolicy = field(default_factory=TuningPolicy)

    def __post_init__(self) -> None:
        if not math.isfinite(self.el_min) or not 0.0 <= self.el_min < 90.0:
            raise ValueError(f"el_min must be within [0, 90) degrees, got {self.el_min}")
        for name in (
            "max_pass_seconds",
            "az_speed",
            "az_accel",
            "min_pass_seconds",
            "retry_after_seconds",
            "max_wait_seconds",
            "time_step",
        ):
            _positive(name, getattr(self, name))
        _within_axis_ceilings(self.az_speed, self.az_accel)
        _non_negative("footprint_margin", self.footprint_margin)
        _non_negative("dispatch_lead_seconds", self.dispatch_lead_seconds)
        if self.n_passes < 1:
            raise ValueError(f"n_passes must be at least 1, got {self.n_passes}")
        if self.moon_min_separation is not None:
            _non_negative("moon_min_separation", self.moon_min_separation)
        # Checked here, as CalibrationPolicy checks its planet_cal_footprint,
        # so a tag no module answers to is refused when the policy is built
        # rather than by the first visit's footprint lookup.
        _require_module_tag("footprint", self.footprint)

    def as_record(self) -> dict[str, Any]:
        """Return the policy as a plain dict of JSON builtins (nested for ``tuning``)."""
        record = dataclasses.asdict(self)
        return json.loads(json.dumps(record))


def _policy_from_record(record: dict[str, Any]) -> CalibrationNightPolicy:
    """Rebuild a policy from :meth:`CalibrationNightPolicy.as_record` output.

    A record without ``use_table_throw`` was written before the switch
    existed, when every pass swept the table's throw, so it is read as
    ``use_table_throw=True``; one without ``dispatch_lead_seconds`` was
    written before the lead existed, when nothing was reserved in front
    of a pass, so it is read as ``dispatch_lead_seconds=0.0``; and a
    tuning record without ``skydip_after_retune`` was written before that
    option existed, when no skydip followed the retune, so it is read as
    ``skydip_after_retune=False``.

    Raises
    ------
    ValueError
        If the record, or its ``tuning`` record, holds a key the policy
        does not have (a key from a later version of this library), named
        in the message, or the policy it describes is one
        :class:`CalibrationNightPolicy` refuses.
    """
    fields = dict(record)
    tuning = dict(fields["tuning"])
    unknown = sorted(set(fields) - {f.name for f in dataclasses.fields(CalibrationNightPolicy)})
    unknown += [
        f"tuning.{key}"
        for key in sorted(set(tuning) - {f.name for f in dataclasses.fields(TuningPolicy)})
    ]
    if unknown:
        raise ValueError(
            "the recorded policy holds keys this version of the library does not know: "
            + ", ".join(unknown)
        )
    fields.setdefault("use_table_throw", True)
    fields.setdefault("dispatch_lead_seconds", 0.0)
    fields["tuning"] = TuningPolicy(**{"skydip_after_retune": False, **tuning})
    return CalibrationNightPolicy(**fields)


class CalibrationNightMetadata(TypedDict):
    """What a planned night records in the timeline header.

    Attributes
    ----------
    targets : list of str
        The bodies the night was planned for, in the caller's order.
    policy : dict
        The :class:`CalibrationNightPolicy` as a plain dict.
    tables : dict
        The scan-parameter tables in force, keyed by body, each a list of
        bin dicts.
    selection : str
        Name of the selection rule.
    sun_safe : str
        Description of the point-level Sun predicate.
    slew_safe : str
        Description of the path-level Sun predicate.
    requested_window : list of str
        The requested ``[start, end]`` as ISO UTC.
    usable_interval : list of str or None
        The solar-gated ``[start, end]`` actually planned, or ``None``
        when the Sun never set inside the request.
    start_pose : list of float
        The ``[az, el]`` the night started from.
    deferrals : list of dict
        Every deferral, as ``{"body", "at", "reason"}``.
    drops : list of dict
        Every drop for the night, as ``{"body", "at", "reason"}``.
    unplaced : list of dict
        Scripted entries that timed out, as ``{"body", "at", "overrides"}``,
        ``at`` being when the entry was set aside.
    warnings : list of dict
        Advisories the planner recorded while planning visits, as
        ``{"body", "at", "message"}``.
    telescope_limits : dict
        The site's axis limits, as nested dicts of the
        :class:`~fyst_trajectories.site.TelescopeLimits` fields, which
        :meth:`~fyst_trajectories.overhead.NightContext.from_timeline` holds
        the site to.
    """

    targets: list[str]
    policy: dict[str, Any]
    tables: dict[str, list[dict[str, float]]]
    selection: str
    sun_safe: str
    slew_safe: str
    requested_window: list[str]
    usable_interval: list[str] | None
    start_pose: list[float]
    deferrals: list[dict[str, str]]
    drops: list[dict[str, str]]
    unplaced: list[dict[str, Any]]
    warnings: list[dict[str, str]]
    telescope_limits: dict[str, dict[str, float]]


def tables_as_record(tables: dict[str, ScanParameterTable]) -> dict[str, list[dict[str, float]]]:
    """Tables as JSON builtins, one list of bin dicts per body."""
    return {
        body: [dataclasses.asdict(b) for b in table.bins] for body, table in sorted(tables.items())
    }


def tables_from_record(record: dict[str, list[dict[str, float]]]) -> dict[str, ScanParameterTable]:
    """Rebuild tables from :func:`tables_as_record` output."""
    return {
        body: ScanParameterTable(tuple(ElevationBin(**b) for b in bins))
        for body, bins in record.items()
    }


def encode_calibration_night_metadata(meta: CalibrationNightMetadata) -> dict[str, Any]:
    """Encode the night's record as the two namespaced header keys.

    The keys are ``calnight_schema_version`` (an int) and ``calnight_json``
    (one JSON string with sorted keys), so nothing in the record can
    collide with the timeline's own header fields and the payload survives
    the ECSV round trip unchanged.
    """
    return {
        _META_VERSION_KEY: CALNIGHT_SCHEMA_VERSION,
        _META_PAYLOAD_KEY: json.dumps(meta, sort_keys=True),
    }


def read_calibration_night_metadata(timeline: ObservingTimeline) -> CalibrationNightMetadata:
    """Decode the night's record from a timeline's metadata.

    Parameters
    ----------
    timeline : ObservingTimeline
        A timeline produced by :func:`plan_calibration_night`, in memory
        or read back from ECSV.

    Returns
    -------
    CalibrationNightMetadata
        The decoded record.

    Raises
    ------
    KeyError
        If the timeline carries no calibration-night payload.
    ValueError
        If the payload's schema version is not supported.
    """
    meta = timeline.metadata
    if _META_PAYLOAD_KEY not in meta:
        raise KeyError("timeline carries no calibration-night metadata")
    version = int(meta.get(_META_VERSION_KEY, 0))
    if version != CALNIGHT_SCHEMA_VERSION:
        raise ValueError(
            f"calibration-night metadata schema version {version} is not supported "
            f"(this library reads version {CALNIGHT_SCHEMA_VERSION})"
        )
    return json.loads(meta[_META_PAYLOAD_KEY])
