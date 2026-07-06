"""Data model for observation scheduling.

Observation patches, calibration specifications, timeline blocks, overhead
models, calibration policies, and complete timelines.
"""

import dataclasses
import enum
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict

from astropy.time import Time, TimeDelta

if TYPE_CHECKING:
    from ..planning import FieldRegion
    from ..site import Site

__all__ = [
    "BlockType",
    "CEScanParams",
    "CalibrationBlockMetadata",
    "CalibrationPolicy",
    "CalibrationSpec",
    "CalibrationType",
    "DaisyScanParams",
    "EmptyBlockMetadata",
    "ObservingPatch",
    "ObservingTimeline",
    "OverheadModel",
    "PongScanParams",
    "ScanParamsDict",
    "ScienceBlockMetadata",
    "TimelineBlock",
    "TimelineBlockMetadata",
    "validate_scan_params",
]


class CEScanParams(TypedDict, total=False):
    """Optional scan_params for a constant-elevation :class:`ObservingPatch`.

    All keys are optional — any combination may be supplied to override
    defaults computed from the patch geometry.

    Attributes
    ----------
    az_min : float
        Explicit lower azimuth bound in degrees.
    az_max : float
        Explicit upper azimuth bound in degrees.
    az_accel : float
        Azimuth acceleration in deg/s^2.
    az_padding : float
        Extra azimuth padding on each side in degrees.
    timestep : float
        Trajectory time step in seconds.
    lsa_window : tuple or list of (min_lsa, max_lsa)
        Local Sidereal Angle window in degrees. When supplied, the
        constant-elevation planner derives ``start_time`` / ``duration``
        from the LSA window instead of from RA-edge elevation crossings.
        Declared as ``tuple | list`` because ECSV round-trip serialises
        through JSON, which converts tuples to lists — a value freshly
        constructed in Python is typically a tuple, but a value
        deserialised from a stored timeline is a list. The CE planner
        accepts both via ``float(lsa_window[0])`` indexing. See
        :func:`~fyst_trajectories.planning.plan_constant_el_scan`
        for the full semantics (wrap-around handling, ``rising``
        interaction, search horizon).
    rising : bool
        Which elevation crossing to observe: ``True`` for the rising
        (east-of-meridian) half of the field's transit, ``False`` for
        the setting (west-of-meridian) half. When omitted, the scheduler
        picks the crossing from the hour angle at selection time. When
        supplied, the patch is only selectable while the sky side matches
        this request, and the value is forwarded to the planner's
        ``rising`` argument so the emitted trajectory covers the
        requested half.
    """

    az_min: float
    az_max: float
    az_accel: float
    az_padding: float
    timestep: float
    lsa_window: tuple[float, float] | list[float]
    rising: bool


class PongScanParams(TypedDict, total=False):
    """Optional scan_params for a Pong :class:`ObservingPatch`.

    All keys are optional — any combination may be supplied to override
    defaults used by :func:`~fyst_trajectories.planning.plan_pong_scan`.

    Attributes
    ----------
    spacing : float
        Line spacing in degrees.
    num_terms : int
        Number of Fourier terms for smooth turnarounds.
    timestep : float
        Trajectory time step in seconds.
    angle : float
        Rotation angle of the scan pattern in degrees.
    n_cycles : int
        Number of full pattern cycles.
    """

    spacing: float
    num_terms: int
    timestep: float
    angle: float
    n_cycles: int


class DaisyScanParams(TypedDict, total=False):
    """Optional scan_params for a Daisy :class:`ObservingPatch`.

    All keys are optional — any combination may be supplied to override
    defaults used by :func:`~fyst_trajectories.planning.plan_daisy_scan`.

    Attributes
    ----------
    radius : float
        Characteristic radius R0 in degrees.
    turn_radius : float
        Radius of curvature for turns in degrees.
    avoidance_radius : float
        Radius to avoid near center in degrees.
    start_acceleration : float
        Ramp-up acceleration in deg/s^2.
    timestep : float
        Trajectory time step in seconds.
    """

    radius: float
    turn_radius: float
    avoidance_radius: float
    start_acceleration: float
    timestep: float


# Umbrella alias used by :attr:`ObservingPatch.scan_params`. Which
# concrete TypedDict applies depends on the patch's ``scan_type``.
ScanParamsDict = CEScanParams | PongScanParams | DaisyScanParams


# Allowed keys per scan type, derived from each TypedDict's
# ``__optional_keys__`` so the table cannot drift from the declared
# schemas (each TypedDict is ``total=False`` with no required members).
_SCAN_TYPE_TO_SCAN_PARAM_KEYS: dict[str, frozenset[str]] = {
    "constant_el": CEScanParams.__optional_keys__,
    "pong": PongScanParams.__optional_keys__,
    "daisy": DaisyScanParams.__optional_keys__,
}


def validate_scan_params(params: Mapping[str, object], scan_type: str) -> None:
    """Validate that a ``scan_params`` dict matches its declared scan type.

    Catches typos and scan-type/parameter mismatches (e.g. a ``"radiu"``
    typo or a ``"spacing"`` key on a constant-el scan). Call this before
    consuming ``scan_params`` from ECSV round-trips or manually
    constructed timelines.

    Parameters
    ----------
    params : mapping of str to object
        The candidate ``scan_params`` dict.
    scan_type : str
        One of ``"constant_el"``, ``"pong"``, or ``"daisy"``. Must
        match the scan type of the enclosing block.

    Raises
    ------
    KeyError
        If ``scan_type`` is unknown, or if ``params`` contains any key
        not declared by the matching TypedDict.
    """
    if scan_type not in _SCAN_TYPE_TO_SCAN_PARAM_KEYS:
        raise KeyError(
            f"Unknown scan_type {scan_type!r}; expected one of "
            f"{sorted(_SCAN_TYPE_TO_SCAN_PARAM_KEYS)}"
        )
    allowed = _SCAN_TYPE_TO_SCAN_PARAM_KEYS[scan_type]
    unknown = set(params) - allowed
    if unknown:
        raise KeyError(
            f"{scan_type} scan_params has unknown keys {sorted(unknown)}; "
            f"allowed keys for this scan type are {sorted(allowed)}"
        )


class ScienceBlockMetadata(TypedDict, total=False):
    """Metadata attached to a science :class:`TimelineBlock`.

    All keys are optional at the type level, but science blocks emitted
    by :func:`generate_timeline` populate all six keys so
    :func:`schedule_to_trajectories` can reconstruct the trajectory
    after an ECSV round-trip.

    Attributes
    ----------
    ra_center : float
        Right Ascension of the patch center in degrees.
    dec_center : float
        Declination of the patch center in degrees.
    width : float
        Angular width of the field in degrees.
    height : float
        Angular height of the field in degrees.
    velocity : float
        Scan velocity in deg/s.
    scan_params : ScanParamsDict
        Scan-type-specific parameters (see :class:`ScanParamsDict`).
    """

    ra_center: float
    dec_center: float
    width: float
    height: float
    velocity: float
    scan_params: ScanParamsDict


class CalibrationBlockMetadata(TypedDict, total=False):
    """Metadata attached to a calibration :class:`TimelineBlock`.

    Attributes
    ----------
    cal_type : str
        Calibration operation name (e.g. ``"retune"``, ``"planet_cal"``).
    target : str or None, optional
        Target identifier (e.g. ``"jupiter"`` for a planet calibration);
        None for in-place operations.
    """

    cal_type: str
    target: str | None


class EmptyBlockMetadata(TypedDict, total=False):
    """Metadata for slew and idle :class:`TimelineBlock` entries.

    Slew and idle blocks carry no scan-specific payload.
    """


# Exhaustive union of metadata shapes a :class:`TimelineBlock` may carry.
# Every :class:`BlockType` maps to exactly one variant:
#   * ``BlockType.SCIENCE``     -> :class:`ScienceBlockMetadata`
#   * ``BlockType.CALIBRATION`` -> :class:`CalibrationBlockMetadata`
#   * ``BlockType.SLEW`` / ``IDLE`` -> :class:`EmptyBlockMetadata`
TimelineBlockMetadata = ScienceBlockMetadata | CalibrationBlockMetadata | EmptyBlockMetadata


class BlockType(str, enum.Enum):
    """Type identifier for a :class:`TimelineBlock`.

    Members compare equal to their string values, so either the enum
    member (``BlockType.SCIENCE``) or the plain string (``"science"``)
    can be used interchangeably.
    """

    SCIENCE = "science"
    CALIBRATION = "calibration"
    SLEW = "slew"
    IDLE = "idle"

    def __str__(self) -> str:
        return self.value


class CalibrationType(str, enum.Enum):
    """Calibration operation types.

    Members compare equal to their string values, so either the enum
    member (``CalibrationType.RETUNE``) or the plain string
    (``"retune"``) can be used interchangeably.
    """

    RETUNE = "retune"
    POINTING_CAL = "pointing_cal"
    FOCUS = "focus"
    SKYDIP = "skydip"
    PLANET_CAL = "planet_cal"
    BEAM_MAP = "beam_map"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def coerce(cls, value: "CalibrationType | str") -> "CalibrationType":
        """Return ``value`` as a :class:`CalibrationType`, accepting strings.

        Raises :class:`ValueError` with a message listing valid names when
        ``value`` is a string that does not match any member.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"Unknown calibration type {value!r}, expected one of {[e.value for e in cls]}"
            ) from None

    @property
    def duration_field(self) -> str:
        """Name of the :class:`OverheadModel` attribute holding this type's duration.

        For example ``CalibrationType.FOCUS.duration_field == "focus_duration"``.
        Every :class:`CalibrationType` member has its own duration field.
        """
        return _CAL_TYPE_DURATION_FIELD[self]

    @property
    def state_field(self) -> str:
        """Name of the :class:`CalibrationState` attribute holding this type's last-run time.

        For example ``CalibrationType.RETUNE.state_field == "last_retune"``.
        Every :class:`CalibrationType` member has its own state field.

        .. versionchanged:: Unreleased
            Return type tightened from ``str | None`` to ``str`` once
            ``BEAM_MAP`` was promoted to a first-class calibration with its
            own ``last_beam_map`` state field. The ``None`` branch was
            previously unreachable for every member except ``BEAM_MAP``.
        """
        return _CAL_TYPE_STATE_FIELD[self]


# Private lookup tables keyed on :class:`CalibrationType`. Centralising
# these mappings here keeps :meth:`OverheadModel.get_calibration_duration`
# and :meth:`CalibrationState.update` trivially in sync; adding a new
# calibration type is a single-location change. See the docstrings on
# :attr:`CalibrationType.duration_field` and
# :attr:`CalibrationType.state_field` for the public API.
_CAL_TYPE_DURATION_FIELD: dict[CalibrationType, str] = {
    CalibrationType.RETUNE: "retune_duration",
    CalibrationType.POINTING_CAL: "pointing_cal_duration",
    CalibrationType.FOCUS: "focus_duration",
    CalibrationType.SKYDIP: "skydip_duration",
    CalibrationType.PLANET_CAL: "planet_cal_duration",
    CalibrationType.BEAM_MAP: "beam_map_duration",
}

_CAL_TYPE_STATE_FIELD: dict[CalibrationType, str] = {
    CalibrationType.RETUNE: "last_retune",
    CalibrationType.POINTING_CAL: "last_pointing_cal",
    CalibrationType.FOCUS: "last_focus",
    CalibrationType.SKYDIP: "last_skydip",
    CalibrationType.PLANET_CAL: "last_planet_cal",
    CalibrationType.BEAM_MAP: "last_beam_map",
}


@dataclass(frozen=True)
class ObservingPatch:
    """A sky region to observe.

    Parameters
    ----------
    name : str
        Unique identifier for this patch.
    ra_center : float
        Right Ascension of field center in degrees.
    dec_center : float
        Declination of field center in degrees.
    width : float
        Angular width of the field in degrees.
    height : float
        Angular height of the field in degrees.
    scan_type : str
        Scan pattern type: ``"constant_el"``, ``"pong"``, or ``"daisy"``.
    velocity : float
        Scan velocity in deg/s.
    priority : float
        Scheduling priority. Lower values = higher priority.
    weight : float
        Scheduling weight for patch equalization.
    elevation : float or None
        Fixed elevation for CE scans (degrees). None for auto-compute.
    scan_params : ScanParamsDict
        Additional scan-type-specific parameters. The concrete schema
        depends on ``scan_type``: :class:`CEScanParams` for
        ``"constant_el"``, :class:`PongScanParams` for ``"pong"``, or
        :class:`DaisyScanParams` for ``"daisy"``.
    """

    name: str
    ra_center: float
    dec_center: float
    width: float
    height: float
    scan_type: str
    velocity: float
    priority: float = 1.0
    weight: float = 1.0
    elevation: float | None = None
    # Runtime is a plain ``dict``; the TypedDict union is advisory for
    # static checkers. mypy can't match ``dict`` to any union member.
    scan_params: ScanParamsDict = field(default_factory=dict)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")
        if self.scan_type not in ("constant_el", "pong", "daisy"):
            raise ValueError(
                f"scan_type must be 'constant_el', 'pong', or 'daisy', got '{self.scan_type}'"
            )
        if self.velocity <= 0:
            raise ValueError(f"velocity must be positive, got {self.velocity}")
        if self.priority <= 0:
            raise ValueError(f"priority must be positive, got {self.priority}")
        if self.weight < 0:
            raise ValueError(f"weight must be non-negative, got {self.weight}")

    @classmethod
    def from_field_region(
        cls,
        field: "FieldRegion",
        name: str,
        scan_type: str,
        velocity: float,
        **kwargs: Any,
    ) -> "ObservingPatch":
        """Create an ObservingPatch from an existing FieldRegion.

        Parameters
        ----------
        field : FieldRegion
            Field region from ``fyst_trajectories.planning``.
        name : str
            Unique patch identifier.
        scan_type : str
            Scan pattern type.
        velocity : float
            Scan velocity in deg/s.
        **kwargs
            Additional keyword arguments passed to ``ObservingPatch``.

        Returns
        -------
        ObservingPatch
        """
        return cls(
            name=name,
            ra_center=field.ra_center,
            dec_center=field.dec_center,
            width=field.width,
            height=field.height,
            scan_type=scan_type,
            velocity=velocity,
            **kwargs,
        )

    @property
    def dec_min(self) -> float:
        """Minimum declination of the field in degrees."""
        return self.dec_center - self.height / 2.0

    @property
    def dec_max(self) -> float:
        """Maximum declination of the field in degrees."""
        return self.dec_center + self.height / 2.0


@dataclass(frozen=True)
class CalibrationSpec:
    """Specification for a single calibration operation.

    Parameters
    ----------
    name : str or CalibrationType
        Calibration type. Accepts string values for backward
        compatibility; they are coerced to ``CalibrationType``.
    duration : float
        Expected duration in seconds.
    target : str or None
        Planet name for planet calibrations, None for in-place operations.
    elevation : float or None
        Required elevation (e.g., for skydips), or None.
    """

    name: CalibrationType | str
    duration: float
    target: str | None = None
    elevation: float | None = None

    def __post_init__(self) -> None:
        # object.__setattr__ bypasses frozen=True to coerce str -> CalibrationType.
        if not isinstance(self.name, CalibrationType):
            object.__setattr__(self, "name", CalibrationType.coerce(self.name))
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")


@dataclass(frozen=True)
class TimelineBlock:
    """A single time-bounded entry in a timeline.

    Represents either a science observation, calibration operation,
    telescope slew, or idle period.

    Parameters
    ----------
    t_start : Time
        UTC start time.
    t_stop : Time
        UTC stop time.
    block_type : BlockType or str
        Block kind. Accepts string values for backward compatibility;
        they are coerced to :class:`BlockType`.
    patch_name : str
        Patch name (science) or calibration type name.
    az_start : float
        First azimuth endpoint in degrees. For science and calibration
        blocks this is the lower bound (``az_start <= az_end``); for
        slew blocks it is the initial azimuth ("from"), which may
        exceed ``az_end`` when slewing westward.
    az_end : float
        Second azimuth endpoint in degrees. For science and calibration
        blocks this is the upper bound; for slew blocks it is the
        target azimuth ("to").
    elevation : float
        Elevation in degrees.
    scan_index : int
        Parent scan counter.
    subscan_index : int
        Sub-scan index within a split observation (0 if unsplit).
    rising : bool
        Whether this is a rising-side observation.
    scan_type : str
        Scan pattern or calibration type identifier.
    boresight_angle : float
        Nasmyth/boresight rotation angle in degrees (``nasmyth_sign *
        elevation + parallactic_angle``). ``0.0`` means unset — I/O
        routines will recompute it from az/el as needed.
    metadata : ScienceBlockMetadata or CalibrationBlockMetadata or EmptyBlockMetadata
        Additional per-block metadata. Science blocks populate a
        :class:`ScienceBlockMetadata` shape (ra/dec center, width,
        height, velocity, scan_params), calibration blocks use
        :class:`CalibrationBlockMetadata` (cal_type, optional target),
        and slew/idle blocks default to an empty :class:`EmptyBlockMetadata`.
        For science blocks, the geometry keys must be populated for
        :func:`schedule_to_trajectories` to reconstruct trajectories
        after an ECSV round-trip.

    Notes
    -----
    The Python attribute names ``az_start``/``az_end`` deliberately
    carry no ordering implication (the discriminator is ``block_type``).
    TOAST canonical ECSV columns are still written/read as
    ``azmin``/``azmax`` for compatibility with external consumers.
    """

    t_start: Time
    t_stop: Time
    block_type: BlockType | str
    patch_name: str
    az_start: float
    az_end: float
    elevation: float
    scan_index: int
    subscan_index: int = 0
    rising: bool = True
    scan_type: str = ""
    boresight_angle: float = 0.0
    metadata: TimelineBlockMetadata = field(
        default_factory=dict,  # type: ignore[assignment]
    )

    def __post_init__(self) -> None:
        # object.__setattr__ bypasses frozen=True to coerce str -> BlockType.
        if not isinstance(self.block_type, BlockType):
            try:
                object.__setattr__(self, "block_type", BlockType(self.block_type))
            except ValueError:
                valid = [bt.value for bt in BlockType]
                raise ValueError(
                    f"block_type must be one of {valid}, got {self.block_type!r}"
                ) from None
        if self.t_stop.unix < self.t_start.unix:
            raise ValueError(f"t_stop ({self.t_stop.iso}) must be >= t_start ({self.t_start.iso})")
        # NOTE: az_start/az_end are NOT ordered-checked here. Science and
        # calibration blocks have az_start <= az_end, but SLEW blocks use
        # the fields as "from" (az_start) and "to" (az_end), and the from
        # position can exceed the to position when the telescope slews
        # westward. The block_type discriminator captures the semantic
        # difference; enforcing ordering here would break the scheduler.

    @property
    def duration(self) -> float:
        """Block duration in seconds."""
        return (self.t_stop - self.t_start).sec

    @classmethod
    def calibration(
        cls,
        cal_type: "CalibrationType | str",
        t_start: Time,
        duration: float,
        az: float,
        el: float,
        site: "Site",
        scan_index: int,
        *,
        target: str | None = None,
    ) -> "TimelineBlock":
        """Construct a CALIBRATION block in place at a single azimuth.

        Factory for the common case where the telescope is parked
        (``az_start == az_end == az``) while a calibration operation runs.
        The boresight angle is computed from the site and the az/el pose
        via :func:`~fyst_trajectories.overhead.utils.compute_nasmyth_rotation`.
        For retune calibrations emitted *between* subscans, use
        :meth:`retune` instead — that variant carries the parent scan's
        azimuth range so ECSV round-trips preserve the subscan geometry.

        Parameters
        ----------
        cal_type : CalibrationType or str
            Calibration type (e.g. ``"pointing_cal"``). Coerced to
            :class:`CalibrationType` for the ``scan_type`` field and used
            verbatim as ``patch_name``.
        t_start : Time
            UTC start time.
        duration : float
            Block duration in seconds.
        az, el : float
            Parked telescope pose during the calibration, in degrees.
        site : Site
            Observatory site (supplies ``nasmyth_sign`` and latitude
            needed by ``compute_nasmyth_rotation``).
        scan_index : int
            Parent scan counter.
        target : str or None, optional
            Calibration target (e.g. ``"jupiter"`` for a planet cal);
            stored in ``metadata["target"]``.

        Returns
        -------
        TimelineBlock
            A CALIBRATION block with ``az_start == az_end == az``.
        """
        from .utils import compute_nasmyth_rotation

        cal_name = str(CalibrationType.coerce(cal_type))
        meta: CalibrationBlockMetadata = {"cal_type": cal_name, "target": target}
        return cls(
            t_start=t_start,
            t_stop=t_start + TimeDelta(duration, format="sec"),
            block_type=BlockType.CALIBRATION,
            patch_name=cal_name,
            az_start=az,
            az_end=az,
            elevation=el,
            scan_index=scan_index,
            scan_type=cal_name,
            boresight_angle=compute_nasmyth_rotation(az, el, site),
            metadata=meta,
        )

    @classmethod
    def retune(
        cls,
        t_start: Time,
        duration: float,
        az_start: float,
        az_end: float,
        el: float,
        site: "Site",
        scan_index: int,
    ) -> "TimelineBlock":
        """Construct a CALIBRATION retune block spanning a scan's azimuth range.

        Factory for retunes emitted between subscans of a science scan.
        Unlike :meth:`calibration` (which parks at a single azimuth),
        this variant carries the parent scan's ``(az_start, az_end)``
        so the ECSV round-trip preserves the subscan geometry. The
        boresight angle is evaluated at the midpoint of the az range.

        Parameters
        ----------
        t_start : Time
            UTC start time.
        duration : float
            Retune duration in seconds.
        az_start, az_end : float
            Azimuth bounds inherited from the parent science scan, in
            degrees. Typically ``az_start <= az_end``.
        el : float
            Elevation in degrees.
        site : Site
            Observatory site.
        scan_index : int
            Parent scan counter.

        Returns
        -------
        TimelineBlock
            A CALIBRATION block with ``scan_type="retune"``.
        """
        from .utils import compute_nasmyth_rotation

        return cls(
            t_start=t_start,
            t_stop=t_start + TimeDelta(duration, format="sec"),
            block_type=BlockType.CALIBRATION,
            patch_name="retune",
            az_start=az_start,
            az_end=az_end,
            elevation=el,
            scan_index=scan_index,
            scan_type="retune",
            boresight_angle=compute_nasmyth_rotation(0.5 * (az_start + az_end), el, site),
        )

    @classmethod
    def idle(
        cls,
        t_start: Time,
        duration: float,
        az: float,
        el: float,
        site: "Site",
        scan_index: int,
    ) -> "TimelineBlock":
        """Construct an IDLE block advancing wall-clock time at a parked pose.

        Emitted by the scheduler when no patch scores above zero — the
        telescope stays at ``(az, el)`` and the timeline advances by
        ``duration`` seconds.

        Parameters
        ----------
        t_start : Time
            UTC start time.
        duration : float
            Idle interval in seconds.
        az, el : float
            Parked telescope pose, in degrees.
        site : Site
            Observatory site.
        scan_index : int
            Scan counter carried forward (no increment for idle).

        Returns
        -------
        TimelineBlock
            An IDLE block with ``patch_name="no_target"`` and
            ``scan_type="idle"``.
        """
        from .utils import compute_nasmyth_rotation

        empty_meta: EmptyBlockMetadata = {}
        return cls(
            t_start=t_start,
            t_stop=t_start + TimeDelta(duration, format="sec"),
            block_type=BlockType.IDLE,
            patch_name="no_target",
            az_start=az,
            az_end=az,
            elevation=el,
            scan_index=scan_index,
            scan_type="idle",
            boresight_angle=compute_nasmyth_rotation(az, el, site),
            metadata=empty_meta,
        )

    @classmethod
    def slew(
        cls,
        t_start: Time,
        duration: float,
        az_start: float,
        az_end: float,
        el: float,
        site: "Site",
        scan_index: int,
        *,
        patch_name: str,
    ) -> "TimelineBlock":
        """Construct a SLEW block moving from ``az_start`` to ``az_end``.

        The boresight angle is evaluated at the midpoint of the slew's
        azimuth range. Callers are expected to compute the slew duration
        (including any settle time) themselves and pass it in via
        ``duration``.

        Parameters
        ----------
        t_start : Time
            UTC start time (beginning of the slew).
        duration : float
            Total slew time in seconds (move + settle).
        az_start, az_end : float
            Initial ("from") and target ("to") azimuths in degrees. Not
            ordering-checked — westward slews may have
            ``az_start > az_end``.
        el : float
            Target elevation in degrees.
        site : Site
            Observatory site.
        scan_index : int
            Parent scan counter.
        patch_name : str
            Descriptive name (typically ``f"slew_to_{patch}"``) written
            to ECSV.

        Returns
        -------
        TimelineBlock
            A SLEW block with ``scan_type="slew"``.
        """
        from .utils import circular_mean_deg, compute_nasmyth_rotation

        empty_meta: EmptyBlockMetadata = {}
        return cls(
            t_start=t_start,
            t_stop=t_start + TimeDelta(duration, format="sec"),
            block_type=BlockType.SLEW,
            patch_name=patch_name,
            az_start=az_start,
            az_end=az_end,
            elevation=el,
            scan_index=scan_index,
            scan_type="slew",
            # Use a circular mean (atan2 of mean sin/cos) for the mid-azimuth so a
            # north-straddling slew (e.g. 355 -> 5) yields the true midpoint (0)
            # rather than the arithmetic mean (180, ~180 deg wrong). az_start and
            # az_end may be on opposite sides of the north wrap when one position
            # is normalized to a negative azimuth and the other is not.
            boresight_angle=compute_nasmyth_rotation(circular_mean_deg(az_start, az_end), el, site),
            metadata=empty_meta,
        )

    @classmethod
    def science(
        cls,
        patch: "ObservingPatch",
        t_start: Time,
        duration: float,
        az_start: float,
        az_end: float,
        el: float,
        site: "Site",
        scan_index: int,
        *,
        subscan_index: int = 0,
        rising: bool = True,
    ) -> "TimelineBlock":
        """Construct a SCIENCE block for a subscan of ``patch``.

        Metadata is populated from the patch (ra/dec center, width,
        height, velocity, scan_params) so the ECSV round-trip can
        reconstruct the trajectory via :func:`schedule_to_trajectories`.
        The boresight angle is evaluated at the midpoint of the scan's
        azimuth range.

        Parameters
        ----------
        patch : ObservingPatch
            Patch being observed. Supplies ``name``, ``scan_type``, and
            the science metadata keys written to ECSV.
        t_start : Time
            UTC start time of this subscan.
        duration : float
            Subscan duration in seconds.
        az_start, az_end : float
            Ordered azimuth bounds (``az_start <= az_end``) in degrees.
        el : float
            Elevation in degrees.
        site : Site
            Observatory site.
        scan_index : int
            Parent scan counter.
        subscan_index : int, optional
            0-based index within a split scan. Default 0.
        rising : bool, optional
            Whether this is a rising-side observation. Default True.

        Returns
        -------
        TimelineBlock
            A SCIENCE block with ``patch_name=patch.name`` and
            ``scan_type=patch.scan_type``.
        """
        from .utils import compute_nasmyth_rotation

        meta: ScienceBlockMetadata = {
            "velocity": patch.velocity,
            "scan_params": patch.scan_params,
            "ra_center": patch.ra_center,
            "dec_center": patch.dec_center,
            "width": patch.width,
            "height": patch.height,
        }
        return cls(
            t_start=t_start,
            t_stop=t_start + TimeDelta(duration, format="sec"),
            block_type=BlockType.SCIENCE,
            patch_name=patch.name,
            az_start=az_start,
            az_end=az_end,
            elevation=el,
            scan_index=scan_index,
            subscan_index=subscan_index,
            rising=rising,
            scan_type=patch.scan_type,
            boresight_angle=compute_nasmyth_rotation(0.5 * (az_start + az_end), el, site),
            metadata=meta,
        )


@dataclass(frozen=True)
class OverheadModel:
    """Timing parameters for non-science activities.

    Default values are commissioning-era placeholders that should be
    confirmed by the instrument team.

    Parameters
    ----------
    retune_duration : float
        KID probe tone reset duration in seconds.
    pointing_cal_duration : float
        Pointing correction scan duration in seconds.
    focus_duration : float
        Focus check duration in seconds.
    skydip_duration : float
        Sky dip / elevation nod duration in seconds.
    planet_cal_duration : float
        Planet calibration scan duration in seconds.
    beam_map_duration : float
        Beam-map scan duration in seconds. Defaults to the same value
        as ``planet_cal_duration`` since beam maps typically run on the
        same planet targets.
    settle_time : float
        Post-slew settling time in seconds.
    min_scan_duration : float
        Minimum useful science scan duration in seconds.
    max_scan_duration : float
        Maximum scan duration before forced split in seconds.
    """

    retune_duration: float = 5.0
    pointing_cal_duration: float = 180.0
    focus_duration: float = 300.0
    skydip_duration: float = 300.0
    planet_cal_duration: float = 600.0
    beam_map_duration: float = 600.0
    settle_time: float = 5.0
    min_scan_duration: float = 60.0
    max_scan_duration: float = 3600.0

    def __post_init__(self) -> None:
        for fld in dataclasses.fields(self):
            val = getattr(self, fld.name)
            if val < 0:
                raise ValueError(f"{fld.name} must be non-negative, got {val}")
        # ``min_scan_duration > 0`` is a tighter contract than ``>= 0``: every
        # downstream phase tests scan candidates against this floor, so a
        # zero minimum would let a one-sample scan emit a sub-second
        # ``TimelineBlock``. Settle/calibration durations may legitimately be
        # zero (e.g. fixture runs) so we only tighten the scan-duration knob.
        if self.min_scan_duration <= 0:
            raise ValueError(f"min_scan_duration must be positive, got {self.min_scan_duration}")
        if self.min_scan_duration >= self.max_scan_duration:
            raise ValueError(
                f"min_scan_duration ({self.min_scan_duration}) must be less than "
                f"max_scan_duration ({self.max_scan_duration})"
            )

    def get_calibration_duration(self, cal_type: CalibrationType | str) -> float:
        """Get duration for a calibration type.

        Parameters
        ----------
        cal_type : CalibrationType or str
            Calibration type name.

        Returns
        -------
        float
            Duration in seconds.
        """
        cal_type = CalibrationType.coerce(cal_type)
        return float(getattr(self, cal_type.duration_field))


@dataclass(frozen=True)
class CalibrationPolicy:
    """Cadences for calibration operations.

    A cadence of 0 means "perform after every science scan". Cadences
    are in seconds. Default values are commissioning-era placeholders
    that should be confirmed by the instrument team.

    Parameters
    ----------
    retune_cadence : float
        Seconds between KID retunes. 0 = every scan boundary.
    pointing_cadence : float
        Seconds between pointing corrections. Default ``3600.0`` (1 h).
        A value of ``1800.0`` may be appropriate for commissioning.
    focus_cadence : float
        Seconds between focus checks.
    skydip_cadence : float
        Seconds between sky dips.
    planet_cal_cadence : float
        Seconds between planet calibrations.
    beam_map_cadence : float or None
        Seconds between beam-map scans. ``None`` (the default) disables
        automatic beam-map scheduling — beam maps are then injected by
        hand only. Set a non-None value to have the scheduler treat
        beam mapping like the other cadenced calibrations. Beam maps
        target the same planets as ``planet_cal``.
    planet_targets : tuple of str
        Planet names to use for calibration (e.g., ``["jupiter", "saturn"]``).
    planet_min_elevation : float
        Minimum altitude in degrees for a planet to be considered visible
        for calibration. Default is 20.0 degrees.
    """

    retune_cadence: float = 0.0
    pointing_cadence: float = 3600.0
    focus_cadence: float = 7200.0
    skydip_cadence: float = 10800.0
    planet_cal_cadence: float = 43200.0
    beam_map_cadence: float | None = None
    planet_targets: tuple[str, ...] = ("jupiter", "saturn", "mars", "uranus", "neptune")
    planet_min_elevation: float = 20.0

    def __post_init__(self) -> None:
        for fld in (
            "retune_cadence",
            "pointing_cadence",
            "focus_cadence",
            "skydip_cadence",
            "planet_cal_cadence",
        ):
            val = getattr(self, fld)
            if val < 0:
                raise ValueError(f"{fld} must be non-negative, got {val}")
        if self.beam_map_cadence is not None and self.beam_map_cadence < 0:
            raise ValueError(
                f"beam_map_cadence must be non-negative or None, got {self.beam_map_cadence}"
            )


# ObservingTimeline is intentionally non-frozen: the scheduler builds it
# incrementally by appending to ``blocks``.  Once returned to the caller it
# should be treated as read-only (analogous to ``Trajectory`` in trajectory.py).
@dataclass
class ObservingTimeline:
    """A complete observation timeline.

    Contains an ordered sequence of timeline blocks (science, calibration,
    slew, idle) along with the configuration used to generate them.

    Parameters
    ----------
    blocks : list of TimelineBlock
        Time-ordered sequence of timeline entries.
    site : Site
        Observatory site configuration.
    start_time : Time
        Timeline start time (UTC).
    end_time : Time
        Timeline end time (UTC).
    overhead_model : OverheadModel
        Overhead timing parameters used.
    calibration_policy : CalibrationPolicy
        Calibration cadence policy used.
    metadata : dict
        Generation parameters, version info, etc.
    """

    blocks: list[TimelineBlock]
    site: "Site"
    start_time: Time
    end_time: Time
    overhead_model: OverheadModel
    calibration_policy: CalibrationPolicy
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def science_blocks(self) -> list[TimelineBlock]:
        """All science observation blocks."""
        return [b for b in self.blocks if b.block_type == BlockType.SCIENCE]

    @property
    def calibration_blocks(self) -> list[TimelineBlock]:
        """All calibration blocks."""
        return [b for b in self.blocks if b.block_type == BlockType.CALIBRATION]

    @property
    def total_science_time(self) -> float:
        """Total science observation time in seconds."""
        return sum(b.duration for b in self.science_blocks)

    @property
    def total_calibration_time(self) -> float:
        """Total calibration time in seconds."""
        return sum(b.duration for b in self.calibration_blocks)

    @property
    def total_slew_time(self) -> float:
        """Total slew time in seconds."""
        return sum(b.duration for b in self.blocks if b.block_type == BlockType.SLEW)

    @property
    def total_idle_time(self) -> float:
        """Total idle time in seconds."""
        return sum(b.duration for b in self.blocks if b.block_type == BlockType.IDLE)

    @property
    def total_time(self) -> float:
        """Total timeline span in seconds."""
        return (self.end_time - self.start_time).sec

    @property
    def efficiency(self) -> float:
        """Science time as a fraction of total time."""
        total = self.total_time
        if total <= 0:
            return 0.0
        return self.total_science_time / total

    @property
    def n_science_scans(self) -> int:
        """Number of science scan blocks."""
        return len(self.science_blocks)

    def __len__(self) -> int:
        """Return the number of blocks in the timeline."""
        return len(self.blocks)

    def __iter__(self) -> Iterator[TimelineBlock]:
        """Iterate over timeline blocks."""
        return iter(self.blocks)

    def __str__(self) -> str:
        """Human-readable timeline summary."""
        sci_h = self.total_science_time / 3600.0
        cal_h = self.total_calibration_time / 3600.0
        slew_h = self.total_slew_time / 3600.0
        idle_h = self.total_idle_time / 3600.0

        lines = [
            f"ObservingTimeline: {self.start_time.iso} to {self.end_time.iso}",
            f"  Science:     {sci_h:5.1f}h ({self.efficiency:5.1%}), {self.n_science_scans} blocks",
            f"  Calibration: {cal_h:5.1f}h, {len(self.calibration_blocks)} blocks",
            f"  Slew:        {slew_h:5.1f}h",
            f"  Idle:        {idle_h:5.1f}h",
        ]

        patch_times: dict[str, float] = {}
        patch_counts: dict[str, int] = {}
        for b in self.science_blocks:
            patch_times[b.patch_name] = patch_times.get(b.patch_name, 0.0) + b.duration
            patch_counts[b.patch_name] = patch_counts.get(b.patch_name, 0) + 1

        if patch_times:
            parts = [
                f"{name} ({t / 3600:.1f}h, {patch_counts[name]} blks)"
                for name, t in sorted(patch_times.items(), key=lambda x: -x[1])
            ]
            lines.append(f"  Patches:     {', '.join(parts)}")

        return "\n".join(lines)

    def validate(self) -> list[str]:
        """Check timeline for common issues.

        Returns
        -------
        list of str
            Warning messages for any issues found. Empty if clean.
        """
        warnings_list = []
        sorted_blocks = sorted(self.blocks, key=lambda b: b.t_start.unix)

        # Check for overlaps
        for i in range(len(sorted_blocks) - 1):
            if sorted_blocks[i].t_stop.unix > sorted_blocks[i + 1].t_start.unix + 0.01:
                warnings_list.append(
                    f"Overlap: '{sorted_blocks[i].patch_name}' ends at "
                    f"{sorted_blocks[i].t_stop.iso} but "
                    f"'{sorted_blocks[i + 1].patch_name}' starts at "
                    f"{sorted_blocks[i + 1].t_start.iso}"
                )

        # Check time range
        for b in sorted_blocks:
            if b.t_start.unix < self.start_time.unix - 0.01:
                warnings_list.append(
                    f"Block '{b.patch_name}' starts before timeline: {b.t_start.iso}"
                )
            if b.t_stop.unix > self.end_time.unix + 0.01:
                warnings_list.append(f"Block '{b.patch_name}' ends after timeline: {b.t_stop.iso}")

        return warnings_list
