"""Scan-parameter and block-metadata schemas for the offline simulator.

The ``scan_params`` shapes an observing patch or a source-CES calibration pass
carries, the metadata shapes of each timeline block type, and
``validate_scan_params``, which checks a ``scan_params`` dict against its scan
type.
"""

from collections.abc import Mapping
from typing import TypedDict

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
        Explicit lower azimuth bound in degrees. The pair steers where the
        scan is placed - the slew target and the cable-wrap branch - and is
        never passed to
        :func:`~fyst_trajectories.planning.plan_constant_el_scan`, which
        derives its own azimuth range from the field geometry, so it does
        not bound the trajectory the scan executes.
    az_max : float
        Explicit upper azimuth bound in degrees. See ``az_min``.
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
        through JSON, which converts tuples to lists. A value freshly
        constructed in Python is typically a tuple, but a value
        deserialised from a stored timeline is a list. The CE planner
        accepts both via ``float(lsa_window[0])`` indexing. See
        :func:`~fyst_trajectories.planning.plan_constant_el_scan`
        for the full semantics (wrap-around handling, search horizon).
    rising : bool
        Which elevation crossing to observe: ``True`` for the rising
        (east-of-meridian) half of the field's transit, ``False`` for
        the setting (west-of-meridian) half. When omitted, the scheduler
        takes whichever half opens its next plannable crossing pass
        first. When supplied, the patch is only selectable while the sky
        side matches this request, and the value is forwarded to the
        planner's ``rising`` argument so the emitted trajectory covers
        the requested half. Not forwarded alongside ``lsa_window``: the
        sidereal window fixes both the timing and the azimuth range, and
        the planner rejects the pair.
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
        Trajectory time step in seconds.
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
        Trajectory time step in seconds.
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
        The base Prime-Cam module before the per-pass eta shift, by its
        canonical name: ``"c"`` or ``"i1"`` .. ``"i6"``, whichever spelling
        or alias (``"IM0"``, ``"center"``) the policy named it by.
        ``eta_offset_deg`` records the shift applied to it.
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
        calibration-night planner requests a rotation: the scheduler
        records the value the source-CES planner resolves an unset
        rotation to, ``0.0``, and a calibration-night pass records the
        request itself, ``None``. The source-CES planner reads ``None`` as
        ``0.0``, so both rebuild to the same pass.
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
        rebuild re-solves the padded footprint crossing.
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


class ScienceBlockMetadata(TypedDict, total=False):
    """Metadata attached to a science :class:`TimelineBlock`.

    All keys are optional at the type level, but science blocks emitted
    by :func:`generate_timeline` populate the six geometry/scan keys
    (``t0_scan`` is added only on constant-elevation subscans) so
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
        Scan velocity in deg/s, forwarded verbatim to the pattern: an
        on-sky (tangent-plane) speed for ``pong`` and ``daisy``, a
        mount-frame azimuth coordinate rate for ``constant_el``.
    scan_params : ScanParamsDict
        Scan-type-specific parameters (see :data:`ScanParamsDict`).
    t0_scan : str, optional
        ISO UTC timestamp of the visit's planner anchor (constant-elevation
        subscans only): the time the scheduler gated the crossing solve
        on, used by :func:`schedule_to_trajectories` as the
        reconstruction anchor instead of the subscan's own start.
    """

    ra_center: float
    dec_center: float
    width: float
    height: float
    velocity: float
    scan_params: ScanParamsDict
    t0_scan: str


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
        ISO UTC time the scan geometry actually begins, present alongside
        ``scan_params``. The block's ``t_start`` may precede this when
        inter-pass repointing and acquisition time are folded into the
        block.
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
