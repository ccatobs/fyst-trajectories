"""Scan-parameter and block-metadata schemas for the offline simulator.

The ``scan_params`` shapes an observing patch or a source-CES calibration pass
carries, the metadata shapes of each timeline block type, and
``validate_scan_params``, which checks a ``scan_params`` dict against its scan
type.
"""

from collections.abc import Mapping
from typing import Any, TypedDict

__all__ = [
    "CEScanParams",
    "CalibrationBlockMetadata",
    "DaisyScanParams",
    "EmptyBlockMetadata",
    "PongScanParams",
    "ScanGeometryRecord",
    "ScanParamsDict",
    "ScienceBlockMetadata",
    "SourceCESScanParams",
    "TimelineBlockMetadata",
    "TransitionRecord",
    "WindowScanRecord",
    "validate_scan_params",
]


class CEScanParams(TypedDict, total=False):
    """Optional scan_params for a constant-elevation :class:`ObservingPatch`.

    All keys are optional. ``az_accel``, ``az_padding`` and ``timestep``
    override the :func:`~fyst_trajectories.planning.plan_constant_el_scan`
    defaults; the other keys steer placement and timing as described below.

    Attributes
    ----------
    az_min : float
        Lower end of the azimuth window in degrees, given together with
        ``az_max`` (``az_min < az_max``; a window across azimuth 0/360 is
        written unwrapped, as (355, 370) or (-5, 10)). The subscans of a
        visit up to the next retune are consecutive stretches of one scan
        of the window form of
        :func:`~fyst_trajectories.planning.plan_constant_el_scan`, in whole
        legs, sweeping the window widened by ``az_padding`` on each side,
        and a new scan starts after each retune; the visit is timed by the
        field's crossing of the patch elevation on the window's side of the
        meridian. Not combined with ``lsa_window``; a window that reaches
        the meridian needs ``rising``.
    az_max : float
        Upper end of the azimuth window in degrees. See ``az_min``.
    az_accel : float
        Azimuth acceleration in deg/s^2.
    az_padding : float
        Extra azimuth padding on each side in degrees.
    timestep : float
        Trajectory time step in seconds. :func:`generate_timeline` refuses
        a patch whose ``timestep`` is under the Go TCS ``/path`` minimum
        sample interval of 0.05 s.
    lsa_window : tuple or list of (min_lsa, max_lsa)
        Local Sidereal Angle window in degrees. When supplied, the
        constant-elevation planner derives ``start_time`` / ``duration``
        from the LSA window instead of from RA-edge elevation crossings.
        Declared as ``tuple | list`` because the ECSV round trip
        serialises through JSON, which turns a tuple into a list; the
        constant-elevation planner accepts either. See
        :func:`~fyst_trajectories.planning.plan_constant_el_scan`
        for the full semantics (wrap-around handling, search horizon).
    rising : bool
        Which elevation crossing to observe: ``True`` for the rising
        (east-of-meridian) half of the field's transit, ``False`` for
        the setting (west-of-meridian) half. When omitted, the scheduler
        takes whichever half opens its next plannable crossing pass
        first. When supplied, only that half's pass is solved and booked,
        and on a patch without an azimuth window the value is forwarded to
        the planner's ``rising`` argument so the emitted trajectory covers
        the requested half. Beside an azimuth window it names the window's
        side (east of the meridian is the rising one) and must agree with
        it. Beside ``lsa_window`` it is not read: the
        sidereal window fixes both the timing and the side of the azimuth
        range, so the scheduler selects, books and rebuilds the patch as
        it would without ``rising``, and the planner, which refuses the
        pair, is called without it.
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

    All keys are optional. ``num_terms``, ``timestep`` and ``angle``
    override :func:`~fyst_trajectories.planning.plan_pong_scan` defaults;
    that planner requires ``spacing``, so an omitted ``spacing`` falls
    back to the value
    :func:`~fyst_trajectories.overhead.schedule_to_trajectories` rebuilds with.
    The offline scheduler sizes a pong subscan by the period of the same
    pattern and keeps the box that pattern fills, ``angle`` included,
    inside the elevation limits for the whole subscan.

    Attributes
    ----------
    spacing : float
        Line spacing in degrees.
    num_terms : int
        Number of Fourier terms for smooth turnarounds.
    timestep : float
        Trajectory time step in seconds. :func:`generate_timeline` refuses
        a patch whose ``timestep`` is under the Go TCS ``/path`` minimum
        sample interval of 0.05 s.
    angle : float
        Rotation angle of the scan pattern in degrees.
    n_cycles : int
        Number of full pattern periods. On a patch, the most one science
        subscan may hold; on a science block, the number it holds, which
        the offline scheduler records and
        :func:`~fyst_trajectories.overhead.schedule_to_trajectories`
        forwards to :func:`~fyst_trajectories.planning.plan_pong_scan`.
    """

    spacing: float
    num_terms: int
    timestep: float
    angle: float
    n_cycles: int


class DaisyScanParams(TypedDict, total=False):
    """Optional scan_params for a Daisy :class:`ObservingPatch`.

    All keys are optional here. ``timestep`` overrides the
    :func:`~fyst_trajectories.planning.plan_daisy_scan` default; that planner
    requires the other four, so an omitted one falls back to the value
    :func:`~fyst_trajectories.overhead.schedule_to_trajectories` rebuilds with.
    The offline scheduler keeps the same pattern inside the elevation limits
    for the whole subscan; its petals reach past ``radius`` from the centre
    by at least ``turn_radius``.

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
        Trajectory time step in seconds. :func:`generate_timeline` refuses
        a patch whose ``timestep`` is under the Go TCS ``/path`` minimum
        sample interval of 0.05 s.
    """

    radius: float
    turn_radius: float
    avoidance_radius: float
    start_acceleration: float
    timestep: float


class SourceCESScanParams(TypedDict, total=False):
    """Replay parameters for one source-CES pass of a planet calibration.

    Attached to a calibration :class:`TimelineBlock` under
    ``metadata["scan_params"]`` when a planet calibration is planned as a
    multi-pass source-CES sequence (``CalibrationPolicy.planet_cal_scan``).
    All keys are optional at the type level (``total=False``). The keys
    fall into three groups: the geometry keys (``body`` through
    ``eta_offset_deg``, plus ``footprint_margin`` when a margin was
    applied), which every emit path records so the pass can be rebuilt;
    the kernel-override keys (``az_accel``, ``az_padding``, ``v_az``,
    ``az_speed``, ``az_throw``, ``dwell``), recorded only when the
    planner was given a value and forwarded on rebuild only when present,
    so the kernel defaults stay authoritative for passes planned without
    them; and the provenance keys (``pass_index``, ``n_passes``), which
    describe the sequence and play no part in the rebuild.

    The fields together record one pass of
    :func:`~fyst_trajectories.planning.plan_source_ces_passes`: a single
    :func:`~fyst_trajectories.planning.plan_source_ces` call that drags
    the planet across a focal-plane row at a fixed boresight elevation.

    Attributes
    ----------
    body : str
        Solar-system body the pass tracked (e.g. ``"jupiter"``).
    footprint : str
        The base Prime-Cam module bay before the per-pass eta shift:
        ``"c"`` for the on-axis bay, whichever spelling (``"IM0"``,
        ``"c"``, ``"center"``) the policy named it by, and ``"IM1"`` ..
        ``"IM6"`` for a ring bay. ``eta_offset_deg`` records the shift
        applied to it.
    el_bore : float
        Fixed boresight elevation of this pass in degrees. Passes step
        this value monotonically so the source crosses each footprint row
        in sequence.
    mode : str
        Direction of the source arc, ``"rising"`` or ``"setting"``.
    window : list of str
        The pass extent as ``[t0_iso, t1_iso]``: the UTC times the source
        enters and exits the projected footprint. This is the boresight
        pass window, not the search horizon.
    boresight_rot : float or None
        Mechanical boresight rotation in degrees, the source-CES planner's
        ``boresight_rot`` argument. Neither the offline scheduler nor the
        calibration-night planner requests a rotation, and both record
        the request itself, ``None``, which the source-CES planner reads
        as ``0.0``.
    timestep : float
        Trajectory sample spacing in seconds used to build the pass.
    eta_offset_deg : float
        Focal-plane eta (elevation-axis) offset in degrees applied to the
        base footprint for this pass. This is a load-bearing geometry
        parameter, not provenance: rebuilding the pass trajectory requires
        shifting the base footprint by this offset (each offset selects a
        different focal-plane row, so the source-tracking track and its
        azimuth throw differ between passes even at the same ``el_bore``).
    az_accel : float
        Azimuth acceleration in deg/s^2 the pass was planned with. A
        kernel override: recorded when supplied, forwarded when present.
    az_padding : float
        Extra azimuth throw per side in degrees beyond the solved
        footprint crossing. A kernel override: recorded when supplied,
        forwarded when present.
    v_az : float
        Explicit azimuth drift rate in deg/s that replaced the solved
        drift. A kernel override: recorded when supplied, forwarded when
        present; absent means the rebuild re-solves the drift.
    az_speed : float
        Per-leg azimuth speed in deg/s the pass was planned with. A
        kernel override: recorded when supplied, forwarded when present;
        absent means the rebuild derives the slow-drag speed.
    az_throw : float
        Explicit swept azimuth window in degrees. A kernel override:
        recorded when supplied, forwarded when present; absent means the
        rebuild re-solves the padded footprint crossing. Never recorded
        beside ``az_padding``, which the planner refuses with it, so
        :func:`schedule_to_trajectories` logs and skips a record holding
        both.
    dwell : float
        Time on source in seconds that narrowed the pass about the
        crossing midpoint. A kernel override: recorded when supplied,
        forwarded when present; absent means the rebuild scans the whole
        crossing.
    footprint_margin : float
        On-sky margin in degrees the base footprint was inflated by on
        every side before planning. Load-bearing geometry like
        ``eta_offset_deg``, applied caller-side and re-applied on rebuild
        to the base module tag before the eta shift; absent means no
        inflation.
    pass_index : int
        0-based index of this pass within the sequence, ordered by start
        time.
    n_passes : int
        Total number of passes requested for the sequence. Unchanged by
        end-of-night truncation, so a truncated sequence still reports the
        full count.
    """

    body: str
    footprint: str
    el_bore: float
    mode: str
    window: list[str]
    boresight_rot: float | None
    timestep: float
    eta_offset_deg: float
    az_accel: float
    az_padding: float
    v_az: float
    az_speed: float
    az_throw: float
    dwell: float
    footprint_margin: float
    pass_index: int
    n_passes: int


# Umbrella alias used by :attr:`ObservingPatch.scan_params`. Which
# concrete TypedDict applies depends on the patch's ``scan_type``.
#
# Deliberately EXCLUDES ``SourceCESScanParams``: a patch never carries
# source-CES geometry. Do not equalize with the planning-side vocabularies.
ScanParamsDict = CEScanParams | PongScanParams | DaisyScanParams


# Allowed keys per scan type, derived from each TypedDict's
# ``__optional_keys__`` so the table cannot drift from the declared
# schemas (each TypedDict is ``total=False`` with no required members).
# ``source_ces`` is registered here so a planet calibration planned as a
# source-CES pass sequence can validate the parameters it records; the
# science planners never emit it (``ObservingPatch`` rejects the type).
# Do not equalize with the planning-side ``_SCAN_TYPE_TO_KEYS``, which
# excludes ``source_ces`` for its own reason.
_SCAN_TYPE_TO_SCAN_PARAM_KEYS: dict[str, frozenset[str]] = {
    "constant_el": CEScanParams.__optional_keys__,
    "pong": PongScanParams.__optional_keys__,
    "daisy": DaisyScanParams.__optional_keys__,
    "source_ces": SourceCESScanParams.__optional_keys__,
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
        One of ``"constant_el"``, ``"pong"``, ``"daisy"``, or
        ``"source_ces"``. The first three are the science scan types an
        :class:`ObservingPatch` carries; ``"source_ces"`` validates the
        :class:`SourceCESScanParams` recorded on planet-calibration
        blocks planned as source-CES pass sequences. Must match the
        scan type the enclosing block declares for its ``scan_params``.

    Raises
    ------
    KeyError
        If ``scan_type`` is not one of the four registered types, or if
        ``params`` contains any key not declared by the matching
        TypedDict.
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


class WindowScanRecord(TypedDict):
    """The inputs of one constant-elevation window scan, as a science block records them.

    A science :class:`TimelineBlock` written by
    :meth:`TimelineBlock.from_window_scan` carries one of these under
    ``metadata["window_scan"]``: what the window form of
    :func:`~fyst_trajectories.planning.plan_constant_el_scan` needs to plan
    the same scan again, so
    :func:`~fyst_trajectories.overhead.schedule_to_trajectories` rebuilds the
    block with that planner. The azimuth rate is the block's ``velocity`` key
    and the field, when the scan was planned with one, its ``ra_center``,
    ``dec_center``, ``width`` and ``height`` keys. Every key is required and
    every value is a JSON builtin.

    Attributes
    ----------
    az_window : list of float
        ``[az_lo, az_hi]`` in degrees, the window as commanded: the padding
        is already included and the pair sits on the 360 deg branch the
        planner chose.
    elevation : float
        Commanded elevation of the window in degrees: the detector's when
        ``detector_offset`` is recorded, while the block's own
        ``elevation`` is the boresight's.
    duration : float
        The scan's built length in seconds, a whole number of azimuth legs,
        not the duration first requested.
    az_accel : float
        Azimuth acceleration in deg/s^2.
    timestep : float
        Trajectory time step in seconds.
    detector_offset : dict or None
        ``None``, or the four fields of the
        :class:`~fyst_trajectories.offsets.InstrumentOffset` the scan was
        planned with (``dx`` and ``dy`` in arcmin, ``name``,
        ``instrument_rotation`` in degrees), kept by value so the block
        replays the boresight that was planned.
    """

    az_window: list[float]
    elevation: float
    duration: float
    az_accel: float
    timestep: float
    detector_offset: dict[str, Any] | None


class ScienceBlockMetadata(TypedDict, total=False):
    """Metadata attached to a science :class:`TimelineBlock`.

    All keys are optional at the type level, but science blocks emitted
    by :func:`generate_timeline` populate the six geometry/scan keys
    (``t0_scan`` is added only on constant-elevation subscans) so
    :func:`schedule_to_trajectories` can reconstruct the trajectory
    after an ECSV round-trip. A block written by
    :meth:`TimelineBlock.from_window_scan` carries ``velocity``, an empty
    ``scan_params``, ``window_scan``, the four field keys only when the
    scan was planned with a field, and ``scan_start`` only when the block
    starts after its scan does. Any other science block of a patch that
    carries a ``detector_offset`` also carries ``detector_offset`` and, on
    a constant-elevation block, ``detector_elevation``.

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
        Scan velocity in deg/s, forwarded verbatim to the pattern: an
        on-sky (tangent-plane) speed for ``pong`` and ``daisy``, a
        mount-frame azimuth coordinate rate for ``constant_el``.
    scan_params : ScanParamsDict
        Scan-type-specific parameters (see :data:`ScanParamsDict`).
    t0_scan : str, optional
        ISO UTC timestamp of the visit's planner anchor (constant-elevation
        subscans only): the time the scheduler gated the crossing solve
        on, used by :func:`schedule_to_trajectories` as the
        reconstruction anchor instead of the subscan's own start. A block
        that carries ``window_scan`` is rebuilt from that record and
        ``scan_start`` instead, and this key is not read.
    window_scan : WindowScanRecord, optional
        The inputs of a constant-elevation window scan (see
        :class:`WindowScanRecord`). A block that carries it is rebuilt with
        the window form of
        :func:`~fyst_trajectories.planning.plan_constant_el_scan`.
    scan_start : str, optional
        ISO UTC time, to the millisecond, of the start of the window scan a
        block holds a later stretch of, beside ``window_scan`` on such a
        block only: the rebuild plans the scan from it and keeps the
        block's own samples. A block that starts with its scan does not
        carry it.
    detector_offset : dict, optional
        The four fields of the
        :class:`~fyst_trajectories.offsets.InstrumentOffset` the block was
        planned with (``dx`` and ``dy`` in arcmin, ``name``,
        ``instrument_rotation`` in degrees), written by
        :meth:`TimelineBlock.science` for a patch that carries one and
        passed back to the planner on a rebuild, so the block replays the
        offset it was planned with. A block written by
        :meth:`TimelineBlock.from_window_scan` keeps its offset inside
        ``window_scan`` instead.
    detector_elevation : float, optional
        Elevation in degrees the detector scans at, on a constant-elevation
        block that records ``detector_offset`` (and only there, always
        beside it): the elevation the rebuild plans at, while the block's
        own ``elevation`` is the boresight's.
    """

    ra_center: float
    dec_center: float
    width: float
    height: float
    velocity: float
    scan_params: ScanParamsDict
    t0_scan: str
    window_scan: WindowScanRecord
    scan_start: str
    detector_offset: dict[str, Any]
    detector_elevation: float


class ScanGeometryRecord(TypedDict, total=False):
    """Scan geometry of one source-CES pass at one stage of planning.

    A calibration-night pass records three of these under
    ``metadata["requested"]``, ``metadata["applied"]`` and
    ``metadata["solved"]``: what the caller asked for (overrides, table,
    policy), what the planner applied after re-centring and quantisation,
    and what the solver found. Every value is a JSON builtin; absent keys
    were not set at that stage.

    Attributes
    ----------
    az_speed : float
        Per-leg azimuth speed in deg/s.
    az_accel : float
        Azimuth acceleration in deg/s^2.
    az_throw : float
        Swept azimuth window in degrees.
    dwell : float
        Time on source in seconds.
    footprint_margin : float
        On-sky inflation of the base footprint in degrees.
    crossing_seconds : float
        Full footprint crossing in seconds (solved stage only).
    """

    az_speed: float
    az_accel: float
    az_throw: float
    dwell: float
    footprint_margin: float
    crossing_seconds: float


class TransitionRecord(TypedDict):
    """The slew that preceded a calibration-night pass, as recorded on its block.

    Attributes
    ----------
    wrap : float
        Encoder azimuth the pass was planned in, in degrees.
    cause : str
        The transition's verdict (``"ok"`` on a recorded pass).
    path : str
        ``"direct"`` or ``"detour"``.
    duration : float
        Slew plus settle time in seconds.
    detour_via : list of float or None
        Intermediate ``[az, el]`` of a two-leg detour, else ``None``.
    escape_via : list of float or None
        The ``[az, el]`` the telescope first moved to when the Sun zone
        had overtaken its idle pose, else ``None``. The escape is its own
        slew block, named ``sun_escape``, ahead of the visit's slew.
    """

    wrap: float
    cause: str
    path: str
    duration: float
    detour_via: list[float] | None
    escape_via: list[float] | None


class CalibrationBlockMetadata(TypedDict, total=False):
    """Metadata attached to a calibration :class:`TimelineBlock`.

    Attributes
    ----------
    cal_type : str
        Calibration operation name (e.g. ``"retune"``, ``"planet_cal"``).
    target : str or None, optional
        Target identifier (e.g. ``"jupiter"`` for a planet calibration);
        None for in-place operations.
    scan_params : SourceCESScanParams, optional
        Per-pass source-CES parameters, present only when a planet
        calibration is planned as a source-CES pass sequence
        (``CalibrationPolicy.planet_cal_scan``). Absent for parked
        (fixed-duration) calibrations.
    t0_scan : str, optional
        ISO UTC time the pass begins, present alongside ``scan_params``:
        the pass's first sample, which is where the block starts, to the
        millisecond the string holds.
    search_start : list of float, optional
        Where the planner's search for this pass began, present alongside
        ``scan_params``: ``[jd1, jd2]``, the two parts of the UTC Julian
        date of the ``Time`` the search began at. Both planners hold every
        time as a UTC ``Time`` without a location and with astropy's
        default ``precision`` and ``out_subfmt`` (one given in another scale
        is converted to UTC), so
        ``Time(jd1, jd2, format="jd", scale="utc")`` restores that very
        ``Time``. The planner searched the 24 hours from it, and
        :func:`~fyst_trajectories.overhead.schedule_to_trajectories`
        repeats that search, so the pass rebuilds sample for sample as it
        was planned.
    operation : str, optional
        For a ``retune`` block that stands for another detector operation
        (``"find_detectors"``), the operation's name; absent on a plain
        retune.
    requested, applied, solved : ScanGeometryRecord, optional
        The pass geometry at the three planning stages, on
        calibration-night passes only.
    science_fraction : float, optional
        Fraction of the pass trajectory's samples flagged as science
        (legs, not turnarounds).
    n_legs : int, optional
        Number of azimuth legs in the pass.
    module_crossings : dict of str to float, optional
        Fraction of the pass during which the source sat inside each
        module's field of view, keyed by module name.
    transition : TransitionRecord, optional
        The slew that preceded the pass.
    """

    cal_type: str
    target: str | None
    scan_params: "SourceCESScanParams"
    t0_scan: str
    search_start: list[float]
    operation: str
    requested: ScanGeometryRecord
    applied: ScanGeometryRecord
    solved: ScanGeometryRecord
    science_fraction: float
    n_legs: int
    module_crossings: dict[str, float]
    transition: TransitionRecord


class EmptyBlockMetadata(TypedDict, total=False):
    """Metadata for slew and idle :class:`TimelineBlock` entries.

    Slew and idle blocks carry no scan-specific payload. An idle block
    emitted by a planner that knows why it waited may carry a ``reason``
    label.

    Attributes
    ----------
    reason : str, optional
        Why the telescope idled (a planner's deferral vocabulary).
    """

    reason: str


# Exhaustive union of metadata shapes a :class:`TimelineBlock` may carry.
# Every :class:`BlockType` maps to exactly one variant:
#   * ``BlockType.SCIENCE``     -> :class:`ScienceBlockMetadata`
#   * ``BlockType.CALIBRATION`` -> :class:`CalibrationBlockMetadata`
#   * ``BlockType.SLEW`` / ``IDLE`` -> :class:`EmptyBlockMetadata`
TimelineBlockMetadata = ScienceBlockMetadata | CalibrationBlockMetadata | EmptyBlockMetadata
