"""Data model for observation scheduling.

Observation patches, calibration specifications, timeline blocks, overhead
models, calibration policies, and complete timelines.
"""

import dataclasses
import enum
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from astropy.time import Time, TimeDelta

from .._readonly import ReadOnlyDict
from .._validation import _require_finite
from ..offsets import InstrumentOffset
from ..patterns import ConstantElScanConfig
from ..planning._ce_geometry import _lsa_window_endpoints, _window_side
from .exceptions import ScanParamsSchemaError
from .schemas import (
    CalibrationBlockMetadata,
    EmptyBlockMetadata,
    ScanParamsDict,
    ScienceBlockMetadata,
    SourceCESScanParams,
    TimelineBlockMetadata,
    WindowScanRecord,
    validate_scan_params,
)
from .utils import _require_module_tag, _stored_instant, _stored_isot, _utc_instant

if TYPE_CHECKING:
    from ..planning import FieldRegion, ScanBlock
    from ..site import Site

__all__ = [
    "BlockType",
    "CalibrationPolicy",
    "CalibrationSpec",
    "CalibrationType",
    "ObservingPatch",
    "ObservingTimeline",
    "OverheadModel",
    "TimelineBlock",
]


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

        Raises
        ------
        ValueError
            If ``value`` is a string that matches no member; the message
            lists the valid names.
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
        """
        return _CAL_TYPE_STATE_FIELD[self]


# Private lookup tables keyed on :class:`CalibrationType`, read by
# :meth:`OverheadModel.get_calibration_duration` and
# :meth:`CalibrationState.update`; a new calibration type needs an entry in both.
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

_OFFSET_FIELDS = frozenset(f.name for f in dataclasses.fields(InstrumentOffset))


def _offset_record(offset: InstrumentOffset | None) -> dict[str, Any] | None:
    """Return ``offset`` as a block records it: ``None``, or its fields as a dict.

    The dict is :func:`dataclasses.asdict` of the offset: its ``name``,
    ``dx`` and ``dy`` in arcmin, and ``instrument_rotation`` in degrees.

    Raises
    ------
    ValueError
        If ``offset.name`` is neither a str nor ``None``, since a block's
        metadata is written to a file as JSON.
    """
    if offset is None:
        return None
    if not isinstance(offset.name, (str, type(None))):
        raise ValueError(
            "detector_offset.name must be a str or None, since the block records the "
            f"offset as JSON; got {offset.name!r}"
        )
    return dataclasses.asdict(offset)


def _offset_from_record(record: Any) -> InstrumentOffset | None:
    """Return the offset a block recorded with :func:`_offset_record`.

    Raises
    ------
    ScanParamsSchemaError
        If ``record`` is neither ``None`` nor a mapping of exactly the four
        fields of an :class:`~fyst_trajectories.offsets.InstrumentOffset`,
        with a str or ``None`` name and finite numbers for the rest.
    """
    if record is None:
        return None
    if (
        isinstance(record, Mapping)
        and set(record) == _OFFSET_FIELDS
        and isinstance(record["name"], (str, type(None)))
    ):
        try:
            return InstrumentOffset(**record)
        except (TypeError, ValueError):
            pass
    raise ScanParamsSchemaError(
        "detector_offset must be None or the fields dx, dy, name and "
        f"instrument_rotation of an InstrumentOffset, got {record!r}"
    )


@dataclass(frozen=True)
class ObservingPatch:
    """A sky region to observe.

    Parameters
    ----------
    name : str
        Identifier for this patch, unique within a schedule. It labels every
        block the patch produces and keys the scheduler's crossing-pass memo,
        so ``generate_timeline`` rejects a repeated name.
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
        Scan velocity in deg/s, forwarded verbatim to the pattern: an
        on-sky (tangent-plane) speed for ``pong`` and ``daisy``, a
        mount-frame azimuth coordinate rate for ``constant_el``.
    priority : float
        Divides the patch's selection score: the scheduler ranks patches by
        their constraint score times ``weight / priority``, so a lower value
        is more urgent.
    weight : float
        Multiplies the patch's selection score (see ``priority``).
    elevation : float or None
        Fixed scan elevation in degrees, for a constant-elevation patch.
        The offline scheduler refuses a constant-elevation patch without
        one; ``None`` is accepted there for direct block construction. A
        pong or daisy patch takes ``None`` only: its pattern is scanned
        where the field is on the sky, at the elevation the field has.
    scan_params : ScanParamsDict
        Additional scan-type-specific parameters. The concrete schema
        depends on ``scan_type``: :class:`CEScanParams` for
        ``"constant_el"``, :class:`PongScanParams` for ``"pong"``, or
        :class:`DaisyScanParams` for ``"daisy"``. Stored as a read-only
        copy: it is still a ``dict``, but item assignment, ``update``,
        ``pop`` and the other mutating methods raise ``TypeError``. Derive
        an edited patch with :func:`dataclasses.replace`, and convert with
        ``dict(...)`` before editing a copy or dumping it to YAML.
    detector_offset : InstrumentOffset or None
        Focal-plane offset of the detector the patch is observed with, as
        :func:`~fyst_trajectories.get_primecam_offset` returns it for a
        Prime-Cam bay; ``None`` (the default) is the boresight. An offset
        with ``dx`` and ``dy`` both zero is stored as ``None``, whatever its
        ``name`` and ``instrument_rotation``. On a constant-elevation patch
        that carries one, ``elevation`` is the detector's elevation. Every
        science block :meth:`TimelineBlock.science` writes for the patch
        records the offset by value, and
        :func:`~fyst_trajectories.overhead.schedule_to_trajectories` plans
        the block's rebuild with it.
        :func:`~fyst_trajectories.overhead.generate_timeline` schedules the
        patch for the boresight that holds the detector on the field, and
        the blocks it writes carry the boresight's azimuth and elevation.

    Raises
    ------
    TypeError
        If ``detector_offset`` is neither an
        :class:`~fyst_trajectories.offsets.InstrumentOffset` nor ``None``.
    ValueError
        If ``width``, ``height``, ``velocity``, or ``priority`` is not
        positive, ``weight`` is negative, ``scan_type`` is not one of
        ``"constant_el"`` / ``"pong"`` / ``"daisy"``, a pong or daisy
        patch is given an ``elevation``, ``scan_params``
        carries a key that scan type does not declare, its ``lsa_window``
        is not a pair of real numbers in ``[0, 360)`` with unequal
        endpoints, its azimuth window (``az_min`` and ``az_max``) gives one
        key without the other, a value that is not a finite real number,
        ``az_min`` not less than ``az_max``, or stands beside
        ``lsa_window``, a window that reaches a meridian azimuth comes
        without ``rising``, a ``rising`` names the side of the meridian the
        window does not lie on, or
        ``detector_offset.name`` is neither a str nor ``None`` (a block
        records the offset as JSON).
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
    # Runtime is a read-only ``dict`` subclass; the TypedDict union is
    # advisory for static checkers. mypy can't match ``dict`` to any union
    # member. Left out of the generated hash, since a ``dict`` is unhashable.
    scan_params: ScanParamsDict = field(  # type: ignore[assignment]
        default_factory=dict, hash=False
    )
    detector_offset: InstrumentOffset | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scan_params", ReadOnlyDict(self.scan_params))
        offset = self.detector_offset
        if offset is not None:
            if not isinstance(offset, InstrumentOffset):
                raise TypeError(
                    "detector_offset must be an InstrumentOffset or None, "
                    f"got {type(offset).__name__}"
                )
            # Refuses a name the blocks of this patch could not record.
            _offset_record(offset)
            if offset.dx == 0.0 and offset.dy == 0.0:
                object.__setattr__(self, "detector_offset", None)
        if not self.width > 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if not self.height > 0:
            raise ValueError(f"height must be positive, got {self.height}")
        # Only the science scan types the simulator emits directly.
        # ``source_ces`` is deliberately rejected: it reaches a timeline only
        # as planet-calibration passes, validated by
        # ``_SCAN_TYPE_TO_SCAN_PARAM_KEYS``. Do not equalize the two.
        if self.scan_type not in ("constant_el", "pong", "daisy"):
            raise ValueError(
                f"scan_type must be 'constant_el', 'pong', or 'daisy', got '{self.scan_type}'"
            )
        if self.scan_type != "constant_el" and self.elevation is not None:
            raise ValueError(
                f"patch {self.name!r} is a {self.scan_type} patch with elevation "
                f"{self.elevation}: a {self.scan_type} pattern is scanned where the field is "
                "on the sky, so its elevation cannot be set; leave elevation as None"
            )
        if not self.velocity > 0:
            raise ValueError(f"velocity must be positive, got {self.velocity}")
        if not self.priority > 0:
            raise ValueError(f"priority must be positive, got {self.priority}")
        if not self.weight >= 0:
            raise ValueError(f"weight must be non-negative, got {self.weight}")
        # Refuse a mistyped key here, where it is made, rather than at
        # reconstruction, where it would skip every block of the patch.
        try:
            validate_scan_params(self.scan_params, self.scan_type)
        except KeyError as exc:
            raise ValueError(exc.args[0]) from None
        lsa_window = self.scan_params.get("lsa_window")
        if lsa_window is not None:
            try:
                _lsa_window_endpoints(lsa_window)
            except ValueError as exc:
                raise ValueError(
                    f"patch {self.name!r}: scan_params['lsa_window'] is refused: {exc}"
                ) from None
        self._check_azimuth_window()

    def _check_azimuth_window(self) -> None:
        """Refuse an azimuth window a visit could not sweep, or one whose side is unclear.

        The window is ``scan_params["az_min"]`` and ``["az_max"]`` (only a
        constant-elevation patch declares them): both or neither, real
        finite numbers with ``az_min < az_max``, not beside ``lsa_window``.
        The window watches the crossing on its side of the meridian; one
        that reaches a meridian azimuth needs ``rising``, and a ``rising``
        that names the other side is refused.
        """
        params = self.scan_params
        given = [key for key in ("az_min", "az_max") if key in params]
        if not given:
            return
        if len(given) == 1:
            other = "az_max" if given[0] == "az_min" else "az_min"
            raise ValueError(
                f"patch {self.name!r}: scan_params gives {given[0]} without {other}; an "
                "azimuth window needs both"
            )
        az_min = _require_finite(params["az_min"], "scan_params['az_min']")
        az_max = _require_finite(params["az_max"], "scan_params['az_max']")
        if not az_min < az_max:
            raise ValueError(
                f"patch {self.name!r}: az_min must be less than az_max, got ({az_min}, "
                f"{az_max}). The pair is the window the scan sweeps, from az_min up to az_max: "
                "write a window across azimuth 0/360 unwrapped, as (355, 370) or (-5, 10)."
            )
        if params.get("lsa_window") is not None:
            raise ValueError(
                f"patch {self.name!r}: an azimuth window (az_min, az_max) and lsa_window "
                "cannot both be given: lsa_window times the pass by the sidereal angle and "
                "places its azimuth range itself, while the window is swept as given."
            )
        side = _window_side(az_min, az_max)
        rising = params.get("rising")
        if side is None and rising is None:
            raise ValueError(
                f"patch {self.name!r}: the azimuth window ({az_min}, {az_max}) reaches the "
                "meridian (an azimuth that is a multiple of 180 deg), so it lies on neither "
                "side where the field rises or sets through the elevation; give "
                "scan_params['rising'] to name the crossing the visit watches."
            )
        if side is not None and rising is not None and bool(rising) != (side == "rising"):
            where = "east" if side == "rising" else "west"
            raise ValueError(
                f"patch {self.name!r}: scan_params['rising'] is {rising}, but the azimuth "
                f"window ({az_min}, {az_max}) lies {where} of the meridian, where the field "
                f"{'rises' if side == 'rising' else 'sets'} through the elevation; drop "
                "rising, or give the window on the other side."
            )

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
            Scan velocity in deg/s, forwarded verbatim to the pattern:
            an on-sky (tangent-plane) speed for ``pong`` and ``daisy``,
            a mount-frame azimuth coordinate rate for ``constant_el``.
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
        Calibration type; a string value is coerced to
        ``CalibrationType``.
    duration : float
        Expected duration in seconds.
    target : str or None
        Planet name for planet calibrations, None for in-place operations.

    Raises
    ------
    ValueError
        If ``name`` names no calibration type or ``duration`` is negative.
    """

    name: CalibrationType | str
    duration: float
    target: str | None = None

    def __post_init__(self) -> None:
        # object.__setattr__ bypasses frozen=True to coerce str -> CalibrationType.
        if not isinstance(self.name, CalibrationType):
            object.__setattr__(self, "name", CalibrationType.coerce(self.name))
        if not self.duration >= 0:
            raise ValueError(f"duration must be non-negative, got {self.duration}")


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
        Block kind; a string value is coerced to :class:`BlockType`.
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

        On a swept block, science or calibration, the pair is the
        azimuth *envelope* the block executes, read from the block's own
        trajectory, so it includes the drift a pass crosses and the
        turnaround overshoot past each science edge. A parked
        calibration has ``az_start == az_end``; a retune emitted between
        subscans instead carries the range of the subscan after it as
        geometry for the ECSV round trip.
    elevation : float
        Elevation in degrees.
    scan_index : int
        Parent scan counter.
    subscan_index : int
        Sub-scan index within a split observation (0 if unsplit). A
        multi-pass calibration emitted as one scan numbers its passes
        here. The pair ``(scan_index, subscan_index)`` does not identify a
        block. In a timeline from
        :func:`~fyst_trajectories.overhead.generate_timeline` it is unique
        among the science blocks, and separately among the calibration
        passes, which count their scans on a counter of their own, so a
        pass can share a science block's pair; the idle, slew and parked
        calibration blocks share the pair of a scan block. A block is told
        apart by its times.
    rising : bool
        Whether this is a rising-side observation.
    scan_type : str
        Scan pattern or calibration type identifier.
    metadata : ScienceBlockMetadata or CalibrationBlockMetadata or EmptyBlockMetadata
        Additional per-block metadata. Science blocks populate a
        :class:`ScienceBlockMetadata` shape (ra/dec center, width,
        height, velocity, scan_params), calibration blocks use
        :class:`CalibrationBlockMetadata` (cal_type, optional target),
        and slew/idle blocks default to an empty :class:`EmptyBlockMetadata`.
        :func:`schedule_to_trajectories` reconstructs a science block's
        trajectory from these keys, after an ECSV round-trip too: from the
        geometry keys for a block written from a patch, or from the
        ``window_scan`` record for a block written by
        :meth:`from_window_scan`, which carries the four field keys only
        when its scan was planned with a field.
    az_final : float or None
        Azimuth in degrees the telescope is left at when the block ends,
        recorded only when it differs from ``az_end``. A sweep stops at
        whichever leg endpoint the last turnaround left it on, generally
        neither bound of its envelope, so the pose the next move starts
        from is carried in this separate field. ``None`` (the default)
        means the block ends at ``az_end``: parked calibrations, idles and
        slews always do, and so does a swept block whose trajectory could
        not be built. A retune the offline scheduler emits between
        subscans records here the azimuth the telescope holds through it,
        where the slew or the previous subscan left it. Read it through
        :attr:`end_pose_az` rather than branching at every call site.
    el_final : float or None
        Elevation in degrees the telescope is left at when the block ends,
        recorded only when it differs from ``elevation``. The offline
        scheduler records it on a pong or daisy subscan: such a block is
        stamped at the field centre's elevation where it starts (the
        boresight's that holds the detector there, for a patch with a
        ``detector_offset``), and its pattern follows the field, so it
        ends at the elevation of its trajectory's last sample. ``None``
        (the default) means the block ends at ``elevation``, as every other
        block does. Read it through :attr:`end_pose_el`.

    Raises
    ------
    ValueError
        If ``block_type`` names no :class:`BlockType` member or ``t_stop``
        is not at or after ``t_start`` (a NaN duration included).

    Notes
    -----
    The Python attribute names ``az_start``/``az_end`` deliberately
    carry no ordering implication (the discriminator is ``block_type``).
    The ECSV file names the two columns ``azmin``/``azmax``, TOAST's names.
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
    metadata: TimelineBlockMetadata = field(
        default_factory=dict,  # type: ignore[assignment]
        hash=False,
    )
    az_final: float | None = None
    el_final: float | None = None

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
        if not self.t_stop.unix >= self.t_start.unix:
            raise ValueError(f"t_stop ({self.t_stop.iso}) must be >= t_start ({self.t_start.iso})")
        # az_start/az_end are not order-checked here: a SLEW block holds
        # "from" and "to", and "from" exceeds "to" on a westward slew.
        # ObservingTimeline.validate checks the order per block type.

    @property
    def duration(self) -> float:
        """Block duration in seconds."""
        return (self.t_stop - self.t_start).sec

    @property
    def end_pose_az(self) -> float:
        """Azimuth in degrees the telescope is left at when the block ends.

        ``az_final`` when the block records one, otherwise ``az_end``.
        This is the azimuth the next move starts from, and it is what
        :meth:`ObservingTimeline.validate` follows: ``az_start`` and
        ``az_end`` are the block's azimuth envelope, and a swept block
        ends at neither bound. The ``az_end`` fallback is exact for a
        parked, idle or slew block, which ends where its recorded bound
        says; for a swept block it is an approximation that applies only
        when the trajectory the pose is read from could not be built.
        """
        return self.az_end if self.az_final is None else self.az_final

    @property
    def end_pose_el(self) -> float:
        """Elevation in degrees the telescope is left at when the block ends.

        ``el_final`` when the block records one, otherwise ``elevation``.
        With :attr:`end_pose_az` this is the pose the next move starts
        from, and the pose :meth:`ObservingTimeline.validate` follows.
        """
        return self.elevation if self.el_final is None else self.el_final

    @classmethod
    def calibration(
        cls,
        cal_type: "CalibrationType | str",
        t_start: Time,
        duration: float,
        az: float,
        el: float,
        scan_index: int,
        *,
        target: str | None = None,
        az_end: float | None = None,
        az_final: float | None = None,
        subscan_index: int = 0,
        scan_params: "SourceCESScanParams | None" = None,
        t0_scan: str | None = None,
        rising: bool = True,
        extra_metadata: "Mapping[str, Any] | None" = None,
    ) -> "TimelineBlock":
        """Construct a CALIBRATION block at a single azimuth or across a range.

        Factory for the common case where the telescope is parked
        (``az_start == az_end == az``) while a calibration operation runs.
        For retune calibrations emitted *between* subscans, use
        :meth:`retune` instead; that variant carries the azimuth range of
        the subscan after it so ECSV round-trips preserve the subscan
        geometry.

        The optional ``az_end`` / ``az_final`` / ``scan_params`` / ``t0_scan``
        / ``rising`` keywords support a planet calibration planned as a
        source-CES pass: pass ``az_end`` to record the swept azimuth
        envelope, ``az_final`` to record where the sweep actually stops, and
        ``scan_params`` / ``t0_scan`` to record the pass geometry. All five
        default to the parked single pose.

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
            Telescope pose during the calibration, in degrees. ``az`` is
            the lower azimuth bound (equal to the upper bound when parked).
        scan_index : int
            Parent scan counter.
        target : str or None, optional
            Calibration target (e.g. ``"jupiter"`` for a planet cal);
            stored in ``metadata["target"]``.
        az_end : float or None, optional
            Upper azimuth bound of a swept calibration. ``None`` (default)
            parks at ``az`` (``az_start == az_end == az``).
        az_final : float or None, optional
            Azimuth the sweep ends at, when it differs from ``az_end``.
            ``None`` (default) means the block ends at ``az_end``, which
            is correct for a parked calibration. Supply the planned
            trajectory's last azimuth for a swept pass, so the following
            move is priced and Sun-checked from the real pose.
        subscan_index : int, optional
            Position of this pass within a multi-pass sequence emitted
            under one ``scan_index``. Default ``0``, the single-block
            case.
        scan_params : SourceCESScanParams or None, optional
            Per-pass source-CES parameters; stored in
            ``metadata["scan_params"]`` when not ``None``.
        t0_scan : str or None, optional
            ISO UTC time the scan geometry begins; stored in
            ``metadata["t0_scan"]`` when not ``None``.
        rising : bool, optional
            Whether the calibration observes the rising side. Default
            ``True`` (matches the parked default).
        extra_metadata : mapping, optional
            Further :class:`CalibrationBlockMetadata` keys (for example the
            ``requested`` / ``applied`` / ``solved`` geometry records of a
            calibration-night pass), merged after the standard keys. Must
            not repeat ``cal_type``, ``target``, ``scan_params`` or
            ``t0_scan``, and every key must be one the metadata schema
            declares, so a mistyped field is a refusal rather than a
            silently recorded stranger.

        Returns
        -------
        TimelineBlock
            A CALIBRATION block. Parked (``az_end is None``) blocks have
            ``az_start == az_end == az``.

        Raises
        ------
        ValueError
            If ``extra_metadata`` repeats one of the standard keys, or
            carries a key the metadata schema does not declare.
        """
        cal_name = str(CalibrationType.coerce(cal_type))
        meta: CalibrationBlockMetadata = {"cal_type": cal_name, "target": target}
        if t0_scan is not None:
            meta["t0_scan"] = t0_scan
        if scan_params is not None:
            meta["scan_params"] = scan_params
        if extra_metadata:
            clash = set(extra_metadata) & {"cal_type", "target", "scan_params", "t0_scan"}
            if clash:
                raise ValueError(
                    f"extra_metadata must not repeat the standard keys {sorted(clash)}"
                )
            known = (
                CalibrationBlockMetadata.__required_keys__
                | CalibrationBlockMetadata.__optional_keys__
            )
            unknown = sorted(set(extra_metadata) - known)
            if unknown:
                raise ValueError(
                    f"extra_metadata keys {unknown} are not CalibrationBlockMetadata fields; "
                    f"accepted keys are {sorted(known)}"
                )
            meta.update(extra_metadata)  # type: ignore[typeddict-item]
        az_hi = az if az_end is None else az_end
        return cls(
            t_start=t_start,
            t_stop=t_start + TimeDelta(duration, format="sec"),
            block_type=BlockType.CALIBRATION,
            patch_name=cal_name,
            az_start=az,
            az_end=az_hi,
            elevation=el,
            scan_index=scan_index,
            subscan_index=subscan_index,
            rising=rising,
            scan_type=cal_name,
            metadata=meta,
            az_final=None if az_final is None else float(az_final),
        )

    @classmethod
    def retune(
        cls,
        t_start: Time,
        duration: float,
        az_start: float,
        az_end: float,
        el: float,
        scan_index: int,
    ) -> "TimelineBlock":
        """Construct a CALIBRATION retune block spanning a scan's azimuth range.

        Factory for retunes emitted between subscans of a science scan.
        Unlike :meth:`calibration` (which parks at a single azimuth),
        this variant carries a subscan's ``(az_start, az_end)``
        so the ECSV round-trip preserves the subscan geometry.
        A retune sweeps nothing, so that range is geometry rather than
        an executed envelope: the offline scheduler supplies the range of
        the subscan the retune precedes, and sets ``az_final`` to the
        azimuth the telescope holds through it.

        Parameters
        ----------
        t_start : Time
            UTC start time.
        duration : float
            Retune duration in seconds.
        az_start, az_end : float
            Azimuth bounds of the science subscan the retune belongs to,
            in degrees. Typically ``az_start <= az_end``.
        el : float
            Elevation in degrees.
        scan_index : int
            Parent scan counter.

        Returns
        -------
        TimelineBlock
            A CALIBRATION block with ``scan_type="retune"``.
        """
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
        )

    @classmethod
    def idle(
        cls,
        t_start: Time,
        duration: float,
        az: float,
        el: float,
        scan_index: int,
        *,
        reason: str | None = None,
    ) -> "TimelineBlock":
        """Construct an IDLE block advancing wall-clock time at a parked pose.

        Emitted whenever no scan was placed: the telescope stays at
        ``(az, el)`` and the timeline advances by ``duration`` seconds.
        No patch scoring above zero is the commonest cause and the one
        that carries no ``reason``; a refused slew, a subscan that cannot
        be planned or has two samples closer than the Go TCS ``/path``
        minimum interval (``unplannable``), a subscan with a sample inside
        the Sun zone (``sun_scan``: the site's exclusion radius, which PCS
        enforces, and the night's injected Sun model when one is given), a
        subscan PCS refuses for an axis speed or acceleration
        (``axis_dynamics``), the wait for a pass to open
        (``waiting_for_pass``), a pose the Sun zone holds, and the stretch
        after the last block all label themselves.

        Parameters
        ----------
        t_start : Time
            UTC start time.
        duration : float
            Idle interval in seconds.
        az, el : float
            Parked telescope pose, in degrees.
        scan_index : int
            Scan counter carried forward (no increment for idle).
        reason : str, optional
            Why the telescope idled; stored as ``metadata["reason"]`` when
            given.

        Returns
        -------
        TimelineBlock
            An IDLE block with ``patch_name="no_target"`` and
            ``scan_type="idle"``.
        """
        empty_meta: EmptyBlockMetadata = {}
        if reason is not None:
            empty_meta["reason"] = reason
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
        scan_index: int,
        *,
        patch_name: str,
    ) -> "TimelineBlock":
        """Construct a SLEW block moving from ``az_start`` to ``az_end``.

        Callers are expected to compute the slew duration (including any
        settle time) themselves and pass it in via ``duration``.

        Parameters
        ----------
        t_start : Time
            UTC start time (beginning of the slew).
        duration : float
            Total slew time in seconds (move + settle).
        az_start, az_end : float
            Initial ("from") and target ("to") azimuths in degrees, in
            one coherent cable-wrap frame (their difference is the move
            the mount actually makes). Not ordering-checked; westward
            slews may have ``az_start > az_end``.
        el : float
            Target elevation in degrees.
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
        scan_index: int,
        *,
        subscan_index: int = 0,
        rising: bool = True,
        t0_scan: str | None = None,
        az_final: float | None = None,
        el_final: float | None = None,
    ) -> "TimelineBlock":
        """Construct a SCIENCE block for a subscan of ``patch``.

        Metadata is populated from the patch (ra/dec center, width,
        height, velocity, scan_params) so the ECSV round-trip can
        reconstruct the trajectory via :func:`schedule_to_trajectories`.
        For a patch that carries a ``detector_offset`` the block also
        records ``metadata["detector_offset"]``, the offset's fields, and,
        on a constant-elevation patch, ``metadata["detector_elevation"]``,
        the patch's ``elevation``: the elevation the detector scans at,
        which the rebuild plans at. Neither key is written for a patch
        without an offset.

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
            The subscan's executed azimuth envelope; the offline
            scheduler reads it from the subscan's own trajectory.
        el : float
            The block's elevation in degrees, the boresight's: for a
            constant-elevation patch with a ``detector_offset`` it differs
            from the patch's ``elevation``, which is the detector's.
        scan_index : int
            Parent scan counter.
        subscan_index : int, optional
            0-based index within a split scan. Default 0.
        rising : bool, optional
            Whether this is a rising-side observation. Default True.
        t0_scan : str or None, optional
            ISO timestamp of the visit's planner anchor, stored as
            ``metadata["t0_scan"]`` when not ``None``. Constant-elevation
            subscans record their visit anchor here so
            :func:`~fyst_trajectories.overhead.schedule_to_trajectories`
            re-solves the crossing from the anchor the scheduler gated on,
            not from the subscan's own (possibly post-crossing) start.
        az_final : float or None, optional
            Azimuth the subscan's sweep ends at, when it differs from the
            ``az_end`` envelope bound. Default ``None``.
        el_final : float or None, optional
            Elevation the subscan ends at, when it differs from ``el``.
            Default ``None``.

        Returns
        -------
        TimelineBlock
            A SCIENCE block with ``patch_name=patch.name`` and
            ``scan_type=patch.scan_type``.

        Raises
        ------
        ValueError
            If ``patch`` is a constant-elevation patch with a
            ``detector_offset`` and no ``elevation``.
        """
        meta: ScienceBlockMetadata = {
            "velocity": patch.velocity,
            # Copied into a plain dict: block metadata stays mutable, like
            # the metadata the ECSV reader loads, while the patch's own
            # mapping is read-only.
            "scan_params": dict(patch.scan_params),
            "ra_center": patch.ra_center,
            "dec_center": patch.dec_center,
            "width": patch.width,
            "height": patch.height,
        }
        if t0_scan is not None:
            meta["t0_scan"] = t0_scan
        offset_record = _offset_record(patch.detector_offset)
        if offset_record is not None:
            meta["detector_offset"] = offset_record
            if patch.scan_type == "constant_el":
                if patch.elevation is None:
                    raise ValueError(
                        f"patch {patch.name!r} carries a detector_offset and no elevation: a "
                        "constant-elevation block with an offset needs a pinned elevation, "
                        "the detector's, which its rebuild plans at"
                    )
                meta["detector_elevation"] = float(patch.elevation)
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
            metadata=meta,
            az_final=az_final,
            el_final=el_final,
        )

    @classmethod
    def from_window_scan(
        cls,
        scan_block: "ScanBlock",
        *,
        site: "Site",
        patch_name: str,
        scan_index: int,
        subscan_index: int = 0,
        field: "FieldRegion | None" = None,
        detector_offset: "InstrumentOffset | None" = None,
        span: "tuple[Time | str, Time | str] | None" = None,
    ) -> "TimelineBlock":
        """Construct a SCIENCE block that records one constant-elevation window scan.

        ``scan_block`` is what the window form of
        :func:`~fyst_trajectories.planning.plan_constant_el_scan` (given
        ``az_window`` and ``duration``) returned. The block records that
        planner's inputs under
        ``metadata["window_scan"]`` (a :class:`WindowScanRecord`), so
        :func:`~fyst_trajectories.overhead.schedule_to_trajectories` rebuilds
        it with the same planner, from memory and after an ECSV round trip.
        The azimuth rate is stored as ``metadata["velocity"]``, and the field,
        when one is given, as the four field keys; ``scan_params`` is empty.

        ``field`` and ``detector_offset`` must be the ones given to the
        planner: the scan block does not hold them. A whole scan from the
        field form of the same planner is accepted too when it replays, and
        is recorded, and rebuilt, as the window scan it swept.

        Without ``span`` the block holds the whole scan: ``t_start`` is the
        trajectory's start (held as a UTC instant) and ``t_stop`` the start
        plus the scan's duration. With ``span=(t_start, t_stop)`` the block
        is that stretch of the scan, so one scan can be booked as several
        consecutive blocks: it holds the scan's samples in ``[t_start,
        t_stop)``, and the scan's last sample too when ``t_stop`` is no
        earlier than that sample less 0.01 s. A block that starts after the
        scan's first sample records the scan's start as
        ``metadata["scan_start"]``, the time its rebuild plans the scan
        from. ``az_start`` and ``az_end`` are the boresight's azimuth
        envelope over the samples the block holds, turnaround overshoot
        included, as on every swept block, ``az_final`` the azimuth of its
        last sample when that is not ``az_end``, and ``elevation`` the
        boresight's; with a ``detector_offset`` these differ from the window
        and elevation the record holds, which are the detector's.
        ``rising`` is whether the commanded window's midpoint lies east of
        the meridian.

        Not recorded: the planner's ``atmosphere`` and ``sun_safe`` (a
        rebuild runs the planner's warn-only checks with their defaults and
        screens the Sun with the site's scalar exclusion radius),
        ``az_padding`` (already inside the recorded window) and the duration
        first requested. The record holds the built duration, so a rebuilt
        block's ``computed_params["end_time_iso"]`` is the start plus the
        built duration, and the planner's long-block and field-drift
        advisories are judged on it.

        The block is rebuilt once before it is returned, as
        :func:`~fyst_trajectories.overhead.schedule_to_trajectories` rebuilds
        it, and refused unless that returns ``scan_block``'s trajectory, or
        with ``span`` the samples of it the block holds, sample for sample.
        That costs one more call of the planner, about the time the planner
        itself took. The ``field`` never shapes the trajectory, so this does
        not check it.

        Parameters
        ----------
        scan_block : ScanBlock
            The block the window form of
            :func:`~fyst_trajectories.planning.plan_constant_el_scan`
            returned, whole.
        site : Site
            The site the scan was planned for.
        patch_name : str
            Name written to the block and to the ECSV ``name`` column.
        scan_index : int
            Parent scan counter.
        subscan_index : int, optional
            Sub-scan index. Default 0.
        field : FieldRegion or None, optional
            The field given to the planner, if any. Default ``None``.
        detector_offset : InstrumentOffset or None, optional
            The detector offset given to the planner, if any. Default
            ``None``.
        span : tuple of (Time or str, Time or str), optional
            The block's start and stop when it holds a stretch of the scan
            rather than all of it; a string is read as UTC. Default
            ``None``, the whole scan.

        Returns
        -------
        TimelineBlock
            A SCIENCE block with ``scan_type="constant_el"``.

        Raises
        ------
        ValueError
            If ``scan_block`` is not a constant-elevation scan, its
            trajectory has no ``start_time``, it is a slice of a longer scan
            (its ``duration`` differs from ``computed_params["duration"]``),
            ``span`` does not stop after it starts or starts before the
            scan, ``detector_offset.name`` is neither a str nor ``None``, or
            the block written for it does not rebuild to it: a modified
            trajectory, or a ``detector_offset`` or planner other than the
            ones used. A ``span`` that holds fewer than two samples of the
            scan raises
            :class:`~fyst_trajectories.overhead.BlockNotReconstructableError`,
            a ``ValueError``.

        Examples
        --------
        >>> from astropy.time import Time
        >>> from fyst_trajectories.overhead import TimelineBlock
        >>> from fyst_trajectories.planning import plan_constant_el_scan
        >>> scan = plan_constant_el_scan(
        ...     az_window=(59.7, 67.0),
        ...     elevation=43.0,
        ...     velocity=0.6,
        ...     duration=1200.0,
        ...     site=site,
        ...     start_time=Time("2026-11-15T08:00:00", scale="utc"),
        ... )
        >>> block = TimelineBlock.from_window_scan(
        ...     scan, site=site, patch_name="COSMOS", scan_index=0
        ... )
        >>> block.metadata["window_scan"]["az_window"]
        [59.7, 67.0]
        >>> print(block.t_stop.iso, sorted(block.metadata))
        2026-11-15 08:20:01.800 ['scan_params', 'velocity', 'window_scan']
        """
        config = scan_block.config
        traj = scan_block.trajectory
        if not isinstance(config, ConstantElScanConfig):
            raise ValueError(
                "from_window_scan records a window scan of plan_constant_el_scan, whose config "
                f"is a ConstantElScanConfig; got {type(config).__name__}"
            )
        if traj.start_time is None:
            raise ValueError(
                "the scan block's trajectory has no start_time; pass the block "
                "plan_constant_el_scan returned for an az_window"
            )
        if scan_block.duration != scan_block.computed_params["duration"]:
            # The head of a longer scan can pass the rebuild below in memory
            # and still lose its last sample once a file rounds the block's stop.
            raise ValueError(
                f"the scan block lasts {scan_block.duration} s of a "
                f"{scan_block.computed_params['duration']} s scan: a slice cannot be replayed "
                "from its start; pass the whole block the planner returned"
            )
        offset_record = _offset_record(detector_offset)
        scan_start = _utc_instant(traj.start_time)
        if span is None:
            t_start = scan_start
            t_stop = t_start + TimeDelta(float(scan_block.duration), format="sec")
        else:
            t_start, t_stop = (_utc_instant(t) for t in span)
            if t_stop <= t_start:
                raise ValueError(
                    f"span must stop after it starts, got [{t_start.isot}, {t_stop.isot})"
                )
            if _stored_instant(t_start) < _stored_instant(scan_start):
                raise ValueError(
                    f"span starts at {t_start.isot}, before the scan's start {scan_start.isot}"
                )

        meta: ScienceBlockMetadata = {"velocity": float(config.az_speed), "scan_params": {}}
        if field is not None:
            meta["ra_center"] = float(field.ra_center)
            meta["dec_center"] = float(field.dec_center)
            meta["width"] = float(field.width)
            meta["height"] = float(field.height)
        record: WindowScanRecord = {
            "az_window": [float(config.az_start), float(config.az_stop)],
            "elevation": float(config.elevation),
            "duration": float(scan_block.duration),
            "az_accel": float(config.az_accel),
            "timestep": float(config.timestep),
            "detector_offset": offset_record,
        }
        meta["window_scan"] = record
        if _stored_isot(t_start) != _stored_isot(scan_start):
            meta["scan_start"] = _stored_isot(scan_start)
        block = cls(
            t_start=t_start,
            t_stop=t_stop,
            block_type=BlockType.SCIENCE,
            patch_name=patch_name,
            az_start=0.0,
            az_end=0.0,
            elevation=float(traj.el[0]),
            scan_index=scan_index,
            subscan_index=subscan_index,
            rising=(0.5 * (config.az_start + config.az_stop)) % 360.0 < 180.0,
            scan_type="constant_el",
            metadata=meta,
        )
        # Imported here: simulation imports this module.
        # pylint: disable-next=import-outside-toplevel,cyclic-import
        from .simulation import _window_block_slice, _window_scan_replays

        held = scan_block
        if span is not None:
            # The samples the block holds, counted from the scan's start as
            # the rebuild reads it.
            stored = dataclasses.replace(traj, start_time=_stored_instant(scan_start))
            held = _window_block_slice(dataclasses.replace(scan_block, trajectory=stored), block)
        az = held.trajectory.az
        az_end = float(az.max())
        az_last = float(az[-1])
        block = dataclasses.replace(
            block,
            az_start=float(az.min()),
            az_end=az_end,
            elevation=float(held.trajectory.el[0]),
            az_final=None if az_last == az_end else az_last,
        )

        if not _window_scan_replays(block, held, site):
            raise ValueError(
                "the block written for this scan does not rebuild to it: planning what it "
                "records again does not return the trajectory given. Pass the whole block "
                "plan_constant_el_scan returned, unmodified, with the detector_offset it "
                "was planned with."
            )
        return block


@dataclass(frozen=True)
class OverheadModel:
    """Timing parameters for non-science activities.

    Default values are commissioning-era placeholders. Ownership is
    per-field: ``retune_duration`` is an instrument-team input (KID
    readout wall-time); the remaining durations belong to the
    operations / commissioning team.

    Parameters
    ----------
    retune_duration : float
        Whole-array detector retune reserved between scan blocks, in
        seconds: probe-tone placement followed by a target sweep across
        every module. The default is the instrument team's commissioning
        estimate, pending on-sky timing. This is a different
        operation from the in-scan tone-correction gap that
        :func:`~fyst_trajectories.retune.inject_retune` stamps
        into a trajectory (``DEFAULT_RETUNE_DURATION_SEC``, a few
        seconds); the two values are independent and are not kept in
        sync.
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
        Longest science subscan in seconds. A longer constant-elevation
        pass is split into subscans; a pong or daisy visit is capped at
        it, and a pong subscan holds the most whole pattern periods that
        fit after its boundary retune.

    Raises
    ------
    ValueError
        If a duration is negative or NaN, ``min_scan_duration`` is not
        positive, or ``min_scan_duration`` is not below
        ``max_scan_duration``.
    """

    retune_duration: float = 300.0
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
            if not val >= 0:
                raise ValueError(f"{fld.name} must be non-negative, got {val}")
        # Strictly positive: every phase tests scan candidates against this
        # floor, and a zero floor would admit a sub-second science block.
        # Settle and calibration durations may be zero.
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

    A cadence of 0 keeps that calibration permanently due: retune then
    fires before every science subscan, ahead of any wait for its pass
    to open (plus once at startup), and never on an idle tick, while
    every other calibration type fires on each scheduler iteration, idle
    ticks included.
    Cadences are in seconds. Default values are commissioning-era
    placeholders. Ownership is per-field: ``retune_cadence`` is an
    instrument-team input; the remaining cadences belong to the
    operations / commissioning team.

    Parameters
    ----------
    retune_cadence : float
        Seconds between KID retunes. 0 = scan-coupled: a retune fires
        before every science subscan, ahead of any wait for its pass to
        open (and once at startup), never on idle ticks. Placeholder
        pending Prime-Cam confirmation.
    pointing_cadence : float
        Seconds between pointing corrections. Default ``3600.0`` (1 h).
    focus_cadence : float
        Seconds between focus checks.
    skydip_cadence : float
        Seconds between sky dips.
    planet_cal_cadence : float
        Seconds between planet calibrations.
    beam_map_cadence : float or None
        Seconds between beam-map scans. ``None`` (the default) disables
        automatic beam-map scheduling; beam maps are then injected by
        hand only. Set a non-None value to have the scheduler treat
        beam mapping like the other cadenced calibrations. Beam maps
        target the same planets as ``planet_cal``.
    planet_targets : tuple of str
        Planet names to use for calibration (e.g. ``("jupiter", "saturn")``).
        Must not be empty when ``planet_cal_scan`` is set.
    planet_min_elevation : float
        Minimum altitude in degrees for a planet to be considered visible
        for calibration. Default is 20.0 degrees.
    planet_cal_scan : bool
        When ``True``, plan each planet calibration as a real multi-pass
        source-CES sequence (via
        :func:`~fyst_trajectories.planning.plan_source_ces_passes`),
        reached by a Sun-checked slew planned with
        :func:`~fyst_trajectories.overhead.plan_transition`, instead of a
        single fixed-duration parked block. A listed planet inside the Sun
        zone is passed over for the next one up. Default ``False``
        (parked). Instrument/operations-team placeholder.
    planet_cal_passes : int
        Number of source-CES passes per planet calibration when
        ``planet_cal_scan`` is set. Must be at least 1. Default 3.
        Instrument/operations-team placeholder.
    planet_cal_el_step : float or None
        Boresight-elevation spacing in degrees between consecutive passes.
        ``None`` (default) uses the planner default (the footprint eta
        extent). Must be positive when given. Values sufficiently below
        the footprint's elevation extent can make adjacent pass windows
        overlap in time; a pass that starts before the previous emitted
        pass ends is not emitted, which leaves a gap in the recorded
        ``pass_index``.
        Instrument/operations-team placeholder.
    planet_cal_footprint : str
        Prime-Cam module bay defining the base footprint the passes tile:
        any name ``get_primecam_offset`` (in :doc:`/api/offsets`) takes,
        ``"IM0"`` .. ``"IM6"`` in any case, or ``"c"`` or ``"center"`` for
        ``IM0``. An unknown name, or a value that is not a str, raises
        :class:`ValueError` at construction. The passes' ``scan_params``
        name the on-axis bay ``"c"``, whichever spelling the policy used,
        and a ring bay by its ``IM`` name (``"IM1"`` .. ``"IM6"``).
        Default ``"IM0"``. Instrument/operations-team placeholder.

    Raises
    ------
    ValueError
        If a cadence is negative, ``planet_targets`` is a str rather than a
        sequence of names, ``planet_cal_passes`` is below 1,
        ``planet_min_elevation`` is outside [-90, 90] degrees,
        ``planet_cal_el_step`` is not positive, ``planet_cal_footprint`` is
        not a str naming a known module, or ``planet_cal_scan`` is set with
        no ``planet_targets``.
    """

    retune_cadence: float = 0.0
    pointing_cadence: float = 3600.0
    focus_cadence: float = 7200.0
    skydip_cadence: float = 10800.0
    planet_cal_cadence: float = 43200.0
    beam_map_cadence: float | None = None
    planet_targets: tuple[str, ...] = ("jupiter", "saturn", "mars", "uranus", "neptune")
    planet_min_elevation: float = 20.0
    planet_cal_scan: bool = False
    planet_cal_passes: int = 3
    planet_cal_el_step: float | None = None
    planet_cal_footprint: str = "IM0"

    def __post_init__(self) -> None:
        if isinstance(self.planet_targets, str):
            raise ValueError(
                "planet_targets must be a sequence of planet names, got the str "
                f"{self.planet_targets!r}; wrap a single name as ({self.planet_targets!r},)"
            )
        object.__setattr__(self, "planet_targets", tuple(self.planet_targets))
        for fld in (
            "retune_cadence",
            "pointing_cadence",
            "focus_cadence",
            "skydip_cadence",
            "planet_cal_cadence",
        ):
            val = getattr(self, fld)
            if not val >= 0:
                raise ValueError(f"{fld} must be non-negative, got {val}")
        if self.beam_map_cadence is not None and not self.beam_map_cadence >= 0:
            raise ValueError(
                f"beam_map_cadence must be non-negative or None, got {self.beam_map_cadence}"
            )
        if not -90.0 <= self.planet_min_elevation <= 90.0:
            raise ValueError(
                "planet_min_elevation must be within [-90, 90] degrees, got "
                f"{self.planet_min_elevation}"
            )
        if not self.planet_cal_passes >= 1:
            raise ValueError(f"planet_cal_passes must be at least 1, got {self.planet_cal_passes}")
        if self.planet_cal_el_step is not None and not self.planet_cal_el_step > 0:
            raise ValueError(
                f"planet_cal_el_step must be positive when given, got {self.planet_cal_el_step}"
            )
        if self.planet_cal_scan and not self.planet_targets:
            raise ValueError("planet_cal_scan=True needs at least one entry in planet_targets")
        # Checked even with planet_cal_scan off, so a bad tag fails here rather
        # than inside generate_timeline at the first planet calibration.
        _require_module_tag("planet_cal_footprint", self.planet_cal_footprint)


# ObservingTimeline is intentionally non-frozen: the scheduler builds it
# incrementally by appending to ``blocks``. Once returned to the caller it
# should be treated as read-only.
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
        Free-form header fields: the scheduler records ``n_patches`` and
        ``time_step``, the calibration-night planner its record
        (``calnight_schema_version``, ``calnight_json``), and a timeline
        read by :func:`read_timeline` holds the header keys it does not
        consume.
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

        Checks, in order: block overlap in time, gaps between
        consecutive blocks, blocks outside the timeline window,
        azimuth ordering on science and calibration blocks (slew
        blocks are from/to pairs and exempt), and pose
        continuity: each slew must start from the azimuth, and each
        idle must park at the azimuth and elevation, that the
        previous pose-setting block established. Science,
        calibration, and slew blocks hand their
        :attr:`~fyst_trajectories.overhead.TimelineBlock.end_pose_az`
        and :attr:`~fyst_trajectories.overhead.TimelineBlock.end_pose_el`
        forward; an idle leaves the pose unchanged.
        Only the slew's azimuth is checked, because a slew block
        records its target elevation and carries no starting
        elevation to compare. The tracker starts at the first
        pose-setting block (nothing is checked against the
        scheduler's bootstrap pose), and it follows the recorded
        blocks themselves, so an error consistent across a whole
        timeline is not detectable from the inside.

        A schedule is expected to tile: every emitter here fills the
        time between blocks with an idle, so an unaccounted stretch
        means time that is neither science, calibration, slew nor idle,
        and the four totals then do not add up to ``total_time``. The
        gap check uses the same 0.01 s tolerance as the overlap check,
        which also covers the millisecond rounding of an ECSV round
        trip.

        Returns
        -------
        list of str
            Warning messages for any issues found. Empty if clean.
        """
        warnings_list = []
        sorted_blocks = sorted(self.blocks, key=lambda b: b.t_start.unix)
        pose_tol = 0.1  # deg; poses are copied through state, so drift is float noise

        for i in range(len(sorted_blocks) - 1):
            gap = sorted_blocks[i + 1].t_start.unix - sorted_blocks[i].t_stop.unix
            if gap < -0.01:
                warnings_list.append(
                    f"Overlap: '{sorted_blocks[i].patch_name}' ends at "
                    f"{sorted_blocks[i].t_stop.iso} but "
                    f"'{sorted_blocks[i + 1].patch_name}' starts at "
                    f"{sorted_blocks[i + 1].t_start.iso}"
                )
            elif gap > 0.01:
                warnings_list.append(
                    f"Gap of {gap:.3f} s: '{sorted_blocks[i].patch_name}' ends at "
                    f"{sorted_blocks[i].t_stop.iso} but "
                    f"'{sorted_blocks[i + 1].patch_name}' starts at "
                    f"{sorted_blocks[i + 1].t_start.iso}"
                )

        for b in sorted_blocks:
            if b.t_start.unix < self.start_time.unix - 0.01:
                warnings_list.append(
                    f"Block '{b.patch_name}' starts before timeline: {b.t_start.iso}"
                )
            if b.t_stop.unix > self.end_time.unix + 0.01:
                warnings_list.append(f"Block '{b.patch_name}' ends after timeline: {b.t_stop.iso}")

        for b in sorted_blocks:
            if b.block_type in (BlockType.SCIENCE, BlockType.CALIBRATION):
                if b.az_start > b.az_end + 0.01:
                    warnings_list.append(
                        f"Unordered azimuth bounds on {b.block_type} block "
                        f"'{b.patch_name}' at {b.t_start.iso}: az_start "
                        f"{b.az_start:.3f} > az_end {b.az_end:.3f}"
                    )

        pose_az: float | None = None
        pose_el: float | None = None
        for b in sorted_blocks:
            if b.block_type == BlockType.SLEW:
                if pose_az is not None and abs(b.az_start - pose_az) > pose_tol:
                    warnings_list.append(
                        f"Slew '{b.patch_name}' at {b.t_start.iso} starts at az "
                        f"{b.az_start:.3f} but the previous block ended at az "
                        f"{pose_az:.3f}"
                    )
                pose_az, pose_el = b.end_pose_az, b.end_pose_el
            elif b.block_type == BlockType.IDLE:
                if pose_az is not None and (
                    abs(b.az_start - pose_az) > pose_tol or abs(b.elevation - pose_el) > pose_tol
                ):
                    warnings_list.append(
                        f"Idle at {b.t_start.iso} parked at az {b.az_start:.3f} / el "
                        f"{b.elevation:.2f} but the previous block ended at az "
                        f"{pose_az:.3f} / el {pose_el:.2f}"
                    )
                # Parked: the pose carries through unchanged.
            else:
                # Science and calibration blocks hand their end pose
                # forward; see ``az_final`` and ``el_final``.
                pose_az, pose_el = b.end_pose_az, b.end_pose_el

        return warnings_list
