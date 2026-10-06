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
        If a set value is not a positive, finite number.
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

    Raises
    ------
    ValueError
        If ``find_detectors_duration`` is negative or not finite.
    """

    retune_before_each_block: bool = True
    find_detectors_at_start: bool = True
    find_detectors_duration: float = 300.0

    def __post_init__(self) -> None:
        _non_negative("find_detectors_duration", self.find_detectors_duration)


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
        The Prime-Cam module the passes are planned on, by one name as
        :func:`~fyst_trajectories.primecam.get_primecam_offset` takes it
        (``"c"`` or its aliases ``"center"`` and ``"IM0"``, or ``"i1"`` ..
        ``"i6"``, in any case). A pass's dispatch dict names the module by
        its canonical name, ``"c"`` or ``"i1"`` .. ``"i6"``, since the
        execution layer compares the name as a string; the night's record
        keeps the name as given. Default ``"c"``; a tag naming any other
        module is accepted for simulation and a warning recorded (in
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
        ``az_accel`` is not a positive, finite number,
        ``footprint_margin`` or ``moon_min_separation`` is negative or
        not finite, ``n_passes`` is below 1, or ``footprint`` is not a
        str naming one Prime-Cam module (for an unknown name the message
        lists the names it takes).
    """

    el_min: float = 30.0
    max_pass_seconds: float = 1200.0
    az_speed: float = 1.5
    az_accel: float = 1.0
    footprint: str = "c"
    footprint_margin: float = 0.0
    n_passes: int = 1
    use_table_dwell: bool = False
    use_table_throw: bool = False
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
        _non_negative("footprint_margin", self.footprint_margin)
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
    ``use_table_throw=True``.
    """
    fields = dict(record)
    fields.setdefault("use_table_throw", True)
    fields["tuning"] = TuningPolicy(**fields["tuning"])
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
        :class:`~fyst_trajectories.site.TelescopeLimits` fields; the ECSV
        header does not carry them.
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
