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
from ..coordinates import Coordinates
from .schemas import (
    CalibrationBlockMetadata,
    EmptyBlockMetadata,
    ScanParamsDict,
    ScienceBlockMetadata,
    SourceCESScanParams,
    TimelineBlockMetadata,
    validate_scan_params,
)
from .utils import _require_module_tag

if TYPE_CHECKING:
    from ..planning import FieldRegion
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
        """
        return _CAL_TYPE_STATE_FIELD[self]


# Private lookup tables keyed on :class:`CalibrationType`. Keeping both
# mappings here puts a new calibration type's duration and state entries
# side by side; :meth:`OverheadModel.get_calibration_duration` and
# :meth:`CalibrationState.update` read them. See the docstrings on
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
        Scheduling priority. Lower values = higher priority.
    weight : float
        Scheduling weight for patch equalization.
    elevation : float or None
        Fixed elevation for CE scans (degrees). Required when the patch is
        scheduled by the offline scheduler; None is accepted only for direct
        block construction.
    scan_params : ScanParamsDict
        Additional scan-type-specific parameters. The concrete schema
        depends on ``scan_type``: :class:`CEScanParams` for
        ``"constant_el"``, :class:`PongScanParams` for ``"pong"``, or
        :class:`DaisyScanParams` for ``"daisy"``. Stored as a read-only
        copy: it is still a ``dict``, but item assignment, ``update``,
        ``pop`` and the other mutating methods raise ``TypeError``. Derive
        an edited patch with :func:`dataclasses.replace`, and convert with
        ``dict(...)`` before editing a copy or dumping it to YAML.

    Raises
    ------
    ValueError
        If ``width``, ``height``, ``velocity``, or ``priority`` is not
        positive, ``weight`` is negative, ``scan_type`` is not one of
        ``"constant_el"`` / ``"pong"`` / ``"daisy"``, or ``scan_params``
        carries a key that scan type does not declare.
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "scan_params", ReadOnlyDict(self.scan_params))
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
    """

    name: CalibrationType | str
    duration: float
    target: str | None = None

    def __post_init__(self) -> None:
        # object.__setattr__ bypasses frozen=True to coerce str -> CalibrationType.
        if not isinstance(self.name, CalibrationType):
            object.__setattr__(self, "name", CalibrationType.coerce(self.name))
        if not self.duration > 0:
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
        subscans instead carries its parent scan's estimated range as
        geometry for the ECSV round trip.
    elevation : float
        Elevation in degrees.
    scan_index : int
        Parent scan counter.
    subscan_index : int
        Sub-scan index within a split observation (0 if unsplit). A
        multi-pass calibration emitted as one scan numbers its passes
        here, so ``(scan_index, subscan_index)`` identifies every block
        of a timeline uniquely.
    rising : bool
        Whether this is a rising-side observation.
    scan_type : str
        Scan pattern or calibration type identifier.
    boresight_angle : float
        Celestial-frame field rotation of the focal plane in degrees
        (``nasmyth_sign * elevation + parallactic_angle``), exported as
        the TOAST ``boresight_angle`` column. A derived orientation
        quantity, not a commandable rotator angle (FYST has no
        instrument rotator). ``0.0`` means unset; I/O routines will
        recompute it from az/el as needed.
    metadata : ScienceBlockMetadata or CalibrationBlockMetadata or EmptyBlockMetadata
        Additional per-block metadata. Science blocks populate a
        :class:`ScienceBlockMetadata` shape (ra/dec center, width,
        height, velocity, scan_params), calibration blocks use
        :class:`CalibrationBlockMetadata` (cal_type, optional target),
        and slew/idle blocks default to an empty :class:`EmptyBlockMetadata`.
        For science blocks, the geometry keys must be populated for
        :func:`schedule_to_trajectories` to reconstruct trajectories
        after an ECSV round-trip.
    az_final : float or None
        Azimuth in degrees the telescope is left at when the block ends,
        recorded only when it differs from ``az_end``. A sweep stops at
        whichever leg endpoint the last turnaround left it on, generally
        neither bound of its envelope, so the pose the next move starts
        from is carried in this separate field. ``None`` (the default)
        means the block ends at ``az_end``: parked calibrations, idles and
        slews always do, and so does a swept block whose trajectory could
        not be built. Read it through :attr:`end_pose_az` rather than
        branching at every call site.

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
    az_final: float | None = None

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
        # ObservingTimeline.validate applies the ordering check per block
        # type after the fact, where the discriminator is available.

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
        The boresight angle is computed from the site and the az/el pose
        via
        :meth:`~fyst_trajectories.coordinates.Coordinates.get_field_rotation_from_altaz`.
        For retune calibrations emitted *between* subscans, use
        :meth:`retune` instead; that variant carries the parent scan's
        azimuth range so ECSV round-trips preserve the subscan geometry.

        The optional ``az_end`` / ``az_final`` / ``scan_params`` / ``t0_scan``
        / ``rising`` keywords support a planet calibration planned as a
        source-CES pass: pass ``az_end`` to record the swept azimuth
        envelope, ``az_final`` to record where the sweep actually stops, and
        ``scan_params`` / ``t0_scan`` to record the pass geometry. All five
        default to the parked single-pose behavior, so a call that omits
        them is unchanged.

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
        site : Site
            Observatory site (supplies ``nasmyth_sign`` and latitude
            for the boresight angle).
        scan_index : int
            Parent scan counter.
        target : str or None, optional
            Calibration target (e.g. ``"jupiter"`` for a planet cal);
            stored in ``metadata["target"]``.
        az_end : float or None, optional
            Upper azimuth bound of a swept calibration. ``None`` (default)
            parks at ``az`` (``az_start == az_end == az``). When given, the
            boresight angle is evaluated at the az-range midpoint.
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
            boresight_angle=Coordinates(site).get_field_rotation_from_altaz(0.5 * (az + az_hi), el),
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
        site: "Site",
        scan_index: int,
    ) -> "TimelineBlock":
        """Construct a CALIBRATION retune block spanning a scan's azimuth range.

        Factory for retunes emitted between subscans of a science scan.
        Unlike :meth:`calibration` (which parks at a single azimuth),
        this variant carries the parent scan's ``(az_start, az_end)``
        so the ECSV round-trip preserves the subscan geometry. The
        boresight angle is evaluated at the midpoint of the az range.
        A retune sweeps nothing, so that range is geometry rather than
        an executed envelope: the caller supplies the estimate it placed
        the visit with, not the envelope the neighbouring subscans record.

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
            boresight_angle=Coordinates(site).get_field_rotation_from_altaz(
                0.5 * (az_start + az_end), el
            ),
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
        *,
        reason: str | None = None,
    ) -> "TimelineBlock":
        """Construct an IDLE block advancing wall-clock time at a parked pose.

        Emitted whenever no scan was placed: the telescope stays at
        ``(az, el)`` and the timeline advances by ``duration`` seconds.
        No patch scoring above zero is the commonest cause and the one
        that carries no ``reason``; a refused slew, a pose the Sun zone
        holds, and the stretch after the last block all label themselves.

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
            boresight_angle=Coordinates(site).get_field_rotation_from_altaz(az, el),
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

        The boresight angle is evaluated at the arithmetic mid-travel
        azimuth, ``(az_start + az_end) / 2``. Callers are expected to
        compute the slew duration (including any settle time)
        themselves and pass it in via ``duration``.

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
            # The arithmetic mean is the true mid-travel azimuth of a
            # coherent pair; a circular mean would pick the short modular
            # arc the mount does not travel.
            boresight_angle=Coordinates(site).get_field_rotation_from_altaz(
                0.5 * (az_start + az_end), el
            ),
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
        t0_scan: str | None = None,
        az_final: float | None = None,
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
            The subscan's executed azimuth envelope; the offline
            scheduler reads it from the subscan's own trajectory.
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

        Returns
        -------
        TimelineBlock
            A SCIENCE block with ``patch_name=patch.name`` and
            ``scan_type=patch.scan_type``.
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
            boresight_angle=Coordinates(site).get_field_rotation_from_altaz(
                0.5 * (az_start + az_end), el
            ),
            metadata=meta,
            az_final=az_final,
        )


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

    A cadence of 0 keeps that calibration permanently due: retune then
    fires immediately before every science subscan (plus once at
    startup) and never on an idle tick, while every other calibration
    type fires on each scheduler iteration, idle ticks included.
    Cadences are in seconds. Default values are commissioning-era
    placeholders. Ownership is per-field: ``retune_cadence`` is an
    instrument-team input; the remaining cadences belong to the
    operations / commissioning team.

    Parameters
    ----------
    retune_cadence : float
        Seconds between KID retunes. 0 = scan-coupled: a retune fires
        immediately before every science subscan (and once at startup),
        never on idle ticks. Placeholder pending Prime-Cam confirmation.
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
        extent). Must be positive when given. Values smaller than the
        footprint's elevation extent make adjacent pass windows overlap
        in time, and the planner emits a
        :class:`~fyst_trajectories.exceptions.PointingWarning` when they do.
        Instrument/operations-team placeholder.
    planet_cal_footprint : str
        Prime-Cam module tag defining the base footprint the passes tile
        (e.g. ``"c"``). Must be a known module tag (see
        ``get_primecam_offset`` in :doc:`/api/offsets`); an unknown tag, or
        a value that is not a str, raises :class:`ValueError` at
        construction. The passes' ``scan_params`` name the module by its
        canonical name (``"c"`` for any spelling of the centre module).
        Default ``"c"``. Instrument/operations-team placeholder.

    Raises
    ------
    ValueError
        If a cadence is negative, ``planet_cal_passes`` is below 1,
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
    planet_cal_footprint: str = "c"

    def __post_init__(self) -> None:
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
        if self.planet_cal_passes < 1:
            raise ValueError(f"planet_cal_passes must be at least 1, got {self.planet_cal_passes}")
        if self.planet_cal_el_step is not None and not self.planet_cal_el_step > 0:
            raise ValueError(
                f"planet_cal_el_step must be positive when given, got {self.planet_cal_el_step}"
            )
        if self.planet_cal_scan and not self.planet_targets:
            raise ValueError("planet_cal_scan=True needs at least one entry in planet_targets")
        # Validate the footprint tag at construction (regardless of
        # planet_cal_scan) so a bad tag fails fast here instead of escaping
        # as a KeyError from the first planet-cal emission and aborting
        # generate_timeline.
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

        Checks, in order: block overlap in time, gaps between
        consecutive blocks, blocks outside the timeline window,
        azimuth ordering on science and calibration blocks (slew
        blocks are from/to pairs and exempt), and pose
        continuity: each slew must start from the azimuth, and each
        idle must park at the azimuth and elevation, that the
        previous pose-setting block established. Science,
        calibration, and slew blocks hand their
        :attr:`~fyst_trajectories.overhead.TimelineBlock.end_pose_az`
        and ``elevation`` forward; an idle leaves the pose unchanged.
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
                pose_az, pose_el = b.end_pose_az, b.elevation
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
                # forward; see ``az_final``.
                pose_az, pose_el = b.end_pose_az, b.elevation

        return warnings_list
