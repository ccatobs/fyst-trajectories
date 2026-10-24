"""TOAST-compatible ECSV timeline I/O.

Reads and writes observation timelines in TOAST v5 (ECSV) format
with FYST-specific extensions for calibration blocks.
"""

import dataclasses
import json
import math
import warnings
from datetime import datetime
from pathlib import Path

from astropy import units as u
from astropy.table import QTable, Table
from astropy.time import Time

from ..coordinates import Coordinates
from ..exceptions import PointingWarning
from ..site import (
    AxisLimits,
    Site,
    SunAvoidanceConfig,
    TelescopeLimits,
    get_fyst_site,
)
from ..trajectory import RetuneEvent
from .models import (
    CalibrationPolicy,
    ObservingTimeline,
    OverheadModel,
    TimelineBlock,
)
from .schemas import EmptyBlockMetadata, ScienceBlockMetadata, TimelineBlockMetadata
from .utils import _utc_instant

__all__ = [
    "read_timeline",
    "write_timeline",
]


_UTC_SUFFIXES = ("+00:00", "Z")


def _iso_utc(value: Time | str) -> str:
    """Return the millisecond UTC ISO string with an explicit ``+00:00`` offset.

    The digits are astropy's own millisecond rendering of the instant, the
    form the dispatch sheet is rendered from; the offset keeps consumers that
    parse with ``datetime.fromisoformat`` (TOAST's ``GroundSchedule``)
    independent of the host timezone.
    """
    return _utc_instant(value).isot + "+00:00"


def _read_time(value: str) -> Time:
    """Read a UTC ISO string with a ``+00:00`` / ``Z`` suffix or none.

    The UTC suffix is stripped so these strings take the same astropy parse
    as files written without an offset; any other offset is converted to UTC.
    """
    text = value.strip()
    for suffix in _UTC_SUFFIXES:
        if text.endswith(suffix):
            return Time(text[: -len(suffix)], scale="utc")
    try:
        return Time(text, scale="utc")
    except ValueError:
        return Time(datetime.fromisoformat(text), scale="utc")


# Metadata key used for per-block retune event provenance. When a block's
# ``metadata`` carries this key (value: a sequence of
# :class:`~fyst_trajectories.trajectory.RetuneEvent`), the ECSV writer
# encodes it inside the per-block JSON column as a list of
# ``[t_start, duration]`` float pairs. The reader decodes it back into a
# tuple of :class:`RetuneEvent`. This reuses the existing
# ``block_meta_json`` extra-payload channel (see the JSON encoding/decoding
# helpers below) rather than adding a sidecar table or new columns.
_RETUNE_EVENTS_META_KEY = "retune_events"

_LIMIT_META_KEYS = (
    "site_az_min",
    "site_az_max",
    "site_az_max_velocity",
    "site_az_max_acceleration",
    "site_el_min",
    "site_el_max",
    "site_el_max_velocity",
    "site_el_max_acceleration",
)


def _limit_field(key: str) -> tuple[str, str]:
    """Return the ``TelescopeLimits`` axis and the ``AxisLimits`` field a limit key holds.

    A key of :data:`_LIMIT_META_KEYS` is ``site_<az|el>_<field>``.
    """
    _, axis, field = key.split("_", 2)
    return {"az": "azimuth", "el": "elevation"}[axis], field


def _encode_retune_events_for_json(meta: dict) -> dict:
    """Return a shallow copy of ``meta`` with ``retune_events`` JSON-encoded.

    If ``meta`` does not carry the ``retune_events`` key, the input is
    returned unchanged. Otherwise the value (expected to be an iterable of
    :class:`RetuneEvent`) is converted to a list of ``[t_start, duration]``
    float pairs. Any other JSON-native value is passed through verbatim so
    callers can construct retune-events entries by hand if they prefer
    plain Python types.
    """
    if _RETUNE_EVENTS_META_KEY not in meta:
        return meta
    encoded: list[list[float]] = []
    for ev in meta[_RETUNE_EVENTS_META_KEY]:
        if isinstance(ev, RetuneEvent):
            encoded.append([float(ev.t_start), float(ev.duration)])
        else:
            # Allow pre-encoded list/tuple payloads to pass through. This
            # keeps the writer robust against callers who already encoded.
            encoded.append([float(ev[0]), float(ev[1])])
    out = dict(meta)
    out[_RETUNE_EVENTS_META_KEY] = encoded
    return out


def _decode_retune_events_from_json(extra: dict) -> dict:
    """Return ``extra`` with a JSON-encoded ``retune_events`` turned into a tuple.

    If the key is absent, the input is returned unchanged. Otherwise the
    list of ``[t_start, duration]`` pairs is converted into a tuple of
    :class:`RetuneEvent` instances so consumers see the same type that
    ``Trajectory.retune_events`` exposes.
    """
    if _RETUNE_EVENTS_META_KEY not in extra:
        return extra
    raw = extra[_RETUNE_EVENTS_META_KEY]
    decoded = tuple(RetuneEvent(t_start=float(item[0]), duration=float(item[1])) for item in raw)
    extra = dict(extra)
    extra[_RETUNE_EVENTS_META_KEY] = decoded
    return extra


def write_timeline(
    timeline: ObservingTimeline,
    path: str | Path,
) -> None:
    """Write a timeline to a TOAST-compatible ECSV file.

    The output uses TOAST canonical column names (``start_time``,
    ``stop_time`` as ISO strings, ``name``, ``azmin``, ``azmax``,
    ``el``, ``boresight_angle``, ``scan_index``, ``subscan_index``)
    plus FYST extension columns (``block_type``, ``scan_type``,
    ``rising``, ``az_final``, the science-block geometry columns
    ``ra_center``, ``dec_center``, ``width``, ``height``, ``velocity``,
    and the JSON payload columns ``scan_params_json`` and
    ``block_meta_json``).
    ``boresight_angle`` is the TOAST horizon-frame roll
    (``-nasmyth_sign * elevation``); ``field_rotation`` preserves the
    block's celestial-frame field rotation in degrees.
    For slew rows ``azmin`` / ``azmax`` carry the from / to azimuths of
    the move and may be unordered; they are true minimum / maximum
    bounds only for science and calibration rows. ``az_final`` holds the
    azimuth a swept block ends at when that differs from ``azmax``, and
    is ``nan`` (read back as ``None``) on every other block.
    Every time is written as a UTC ISO string to the millisecond with an
    explicit ``+00:00`` offset, the form ``read_timeline`` reads, whatever
    scale, ``precision`` or ``out_subfmt`` the timeline's ``Time`` objects
    carry. The offset keeps TOAST's ``datetime.fromisoformat`` parsing
    independent of the host timezone.
    The timeline window, site, overhead model, and calibration policy
    are stored in the table header metadata, so ``read_timeline``
    restores them.

    Parameters
    ----------
    timeline : ObservingTimeline
        Timeline to write.
    path : str or Path
        Output file path. Should end in ``.ecsv``.
    """
    path = Path(path)

    rows = []
    for block in timeline.blocks:
        # Prefer the stored boresight_angle on the block; fall back to
        # recomputing from az/el for timelines built without populating
        # the field (e.g. manually constructed TimelineBlocks).
        bangle = block.boresight_angle
        if bangle == 0.0:
            bangle = Coordinates(timeline.site).get_field_rotation_from_altaz(
                0.5 * (block.az_start + block.az_end),
                block.elevation,
            )

        meta = block.metadata
        meta_for_json = _encode_retune_events_for_json(dict(meta))
        rows.append(
            {
                "start_time": _iso_utc(block.t_start),
                "stop_time": _iso_utc(block.t_stop),
                "boresight_angle": -timeline.site.nasmyth_sign * block.elevation,
                "field_rotation": bangle,
                "name": block.patch_name,
                "azmin": block.az_start,
                "azmax": block.az_end,
                # NaN is the "no separate end pose" marker: an ECSV float
                # column has a value in every row, and the reader maps it
                # back to ``None``.
                "az_final": float("nan") if block.az_final is None else float(block.az_final),
                "el": block.elevation,
                "scan_index": block.scan_index,
                "subscan_index": block.subscan_index,
                "block_type": str(block.block_type),
                "scan_type": block.scan_type,
                "rising": block.rising,
                "ra_center": float(meta.get("ra_center", 0.0)),
                "dec_center": float(meta.get("dec_center", 0.0)),
                "width": float(meta.get("width", 0.0)),
                "height": float(meta.get("height", 0.0)),
                "velocity": float(meta.get("velocity", 0.0)),
                "scan_params_json": json.dumps(meta.get("scan_params", {})),
                "block_meta_json": json.dumps(
                    {
                        k: v
                        for k, v in meta_for_json.items()
                        if k
                        not in (
                            "ra_center",
                            "dec_center",
                            "width",
                            "height",
                            "velocity",
                            "scan_params",
                        )
                    }
                ),
            }
        )

    # ECSV cannot express a table with no rows and a typed header, so a
    # timeline with no blocks (a night the Sun never leaves, for
    # instance) writes one placeholder row. The header says so, and
    # ``read_timeline`` drops the row again, otherwise the file reads
    # back as a timeline holding one block dated 2000-01-01 that fails
    # the library's own ``validate()``.
    substituted = not rows
    if substituted:
        rows = [_empty_row()]

    table = Table(rows)

    # Attach angular units to the TOAST canonical columns so a fyst-written
    # ECSV is readable by TOAST's GroundSchedule v5 reader, whose
    # ``GroundScan.__init__`` calls ``az_min.to_value(u.degree)`` on them.
    # Setting ``.unit`` on a plain ``Table`` column is NOT sufficient: that
    # records ``unit: deg`` in the header but deserializes back to a bare
    # float, and ``.to_value`` then raises ``AttributeError``. The table is
    # promoted to ``QTable`` at write time (below) so these columns serialize
    # as Quantity mixins, which is what round-trips. ``read_timeline`` reads
    # with a plain ``Table`` and ``float(row[...])``, so it is unaffected.
    for _deg_col in ("azmin", "azmax", "az_final", "el", "boresight_angle", "field_rotation"):
        table[_deg_col].unit = u.deg

    # Persist the declared timeline window so total_time / efficiency survive
    # a round-trip even when the blocks do not reach the window edges (a padded
    # window would otherwise shrink to the block extents on read). Stored as ISO
    # strings in UTC, mirroring the block time columns. Files without these keys
    # fall back to the block-extent derivation in ``read_timeline``.
    table.meta["timeline_start_time"] = _iso_utc(timeline.start_time)
    table.meta["timeline_end_time"] = _iso_utc(timeline.end_time)
    table.meta["timeline_is_empty"] = substituted
    table.meta["site_name"] = timeline.site.name
    table.meta["site_description"] = timeline.site.description
    table.meta["telescope_name"] = timeline.site.name
    table.meta["site_lat"] = timeline.site.latitude * u.deg
    table.meta["site_lon"] = timeline.site.longitude * u.deg
    table.meta["site_alt"] = timeline.site.elevation * u.m
    # Persist the non-coordinate Site fields that ``_site_from_meta`` would
    # otherwise reset to FYST defaults. nasmyth_port matters most: without it a
    # custom ``"left"`` port reads back as ``"right"``, negating the mechanical
    # (elevation) term of the field rotation.
    table.meta["site_nasmyth_port"] = timeline.site.nasmyth_port
    table.meta["site_plate_scale"] = timeline.site.plate_scale
    table.meta["site_sun_enabled"] = timeline.site.sun_avoidance.enabled
    table.meta["site_sun_exclusion_radius"] = timeline.site.sun_avoidance.exclusion_radius
    table.meta["site_sun_warning_radius"] = timeline.site.sun_avoidance.warning_radius
    # The axis limits, flat like the fields above, so a site whose limits are
    # not FYST's reads back with its own.
    limits = timeline.site.telescope_limits
    for key in _LIMIT_META_KEYS:
        axis, field = _limit_field(key)
        table.meta[key] = float(getattr(getattr(limits, axis), field))
    # OverheadModel: persist ALL fields with overhead_ prefix.
    table.meta["overhead_retune_duration"] = timeline.overhead_model.retune_duration
    table.meta["overhead_pointing_cal_duration"] = timeline.overhead_model.pointing_cal_duration
    table.meta["overhead_focus_duration"] = timeline.overhead_model.focus_duration
    table.meta["overhead_skydip_duration"] = timeline.overhead_model.skydip_duration
    table.meta["overhead_planet_cal_duration"] = timeline.overhead_model.planet_cal_duration
    table.meta["overhead_beam_map_duration"] = timeline.overhead_model.beam_map_duration
    table.meta["overhead_settle_time"] = timeline.overhead_model.settle_time
    table.meta["overhead_min_scan_duration"] = timeline.overhead_model.min_scan_duration
    table.meta["overhead_max_scan_duration"] = timeline.overhead_model.max_scan_duration
    # CalibrationPolicy: persist ALL fields with calibration_ prefix.
    table.meta["calibration_retune_cadence"] = timeline.calibration_policy.retune_cadence
    table.meta["calibration_pointing_cadence"] = timeline.calibration_policy.pointing_cadence
    table.meta["calibration_focus_cadence"] = timeline.calibration_policy.focus_cadence
    table.meta["calibration_skydip_cadence"] = timeline.calibration_policy.skydip_cadence
    table.meta["calibration_planet_cal_cadence"] = timeline.calibration_policy.planet_cal_cadence
    # ``beam_map_cadence`` is ``float | None``; ECSV preserves ``None`` cleanly
    # in table metadata so we can store it directly without sentinel encoding.
    table.meta["calibration_beam_map_cadence"] = timeline.calibration_policy.beam_map_cadence
    table.meta["calibration_planet_targets"] = json.dumps(
        list(timeline.calibration_policy.planet_targets)
    )
    table.meta["calibration_planet_min_elevation"] = (
        timeline.calibration_policy.planet_min_elevation
    )
    table.meta["calibration_planet_cal_scan"] = timeline.calibration_policy.planet_cal_scan
    table.meta["calibration_planet_cal_passes"] = timeline.calibration_policy.planet_cal_passes
    # ``planet_cal_el_step`` is ``float | None``; ECSV preserves ``None``
    # in table metadata, so store it directly (mirrors beam_map_cadence).
    table.meta["calibration_planet_cal_el_step"] = timeline.calibration_policy.planet_cal_el_step
    table.meta["calibration_planet_cal_footprint"] = (
        timeline.calibration_policy.planet_cal_footprint
    )
    table.meta.update(timeline.metadata)

    QTable(table).write(str(path), format="ascii.ecsv", overwrite=True)


def read_timeline(path: str | Path) -> ObservingTimeline:
    """Read a timeline from a TOAST-compatible ECSV file.

    Handles both standard TOAST format (science blocks only, no
    ``block_type`` column) and FYST extended format with calibration
    blocks, scan metadata, and patch geometry columns.

    A file whose header marks the timeline empty reads back with no
    blocks: the single placeholder row such a file carries exists only
    because ECSV cannot hold a typed table with no rows.

    Block times are stored in UTC to millisecond precision, so a block's
    start and stop can shift by up to half a millisecond across a round
    trip, and an equality test on raw durations is not exact.
    The dispatch sheet and the dispatch records render every time and
    duration from that stored form, so the sheet is byte-identical and the
    records equal either way.
    A planet pass's ``search_start`` is two floats in
    the ``block_meta_json`` column and reads back exactly, so the pass
    rebuilds the same from the file as from memory.
    Times may carry a ``+00:00`` or ``Z`` suffix, or none (older files);
    any other offset is converted to UTC.

    Parameters
    ----------
    path : str or Path
        Input file path.

    Returns
    -------
    ObservingTimeline
        Loaded timeline.

    Warns
    -----
    PointingWarning
        When the header describes a non-FYST site but lacks the key of a
        site field the reader restores (a file written before the header
        held the axis limits lacks those eight): that field takes the FYST
        value.
    """
    path = Path(path)
    table = Table.read(str(path), format="ascii.ecsv")

    meta = table.meta
    site = _site_from_meta(meta)

    # Use the dataclass defaults as fall-backs so the I/O defaults can never
    # drift from the class defaults. Repeating a literal default here would
    # silently diverge the moment the dataclass default changes.
    overhead_defaults = OverheadModel()
    overhead = OverheadModel(
        retune_duration=meta.get("overhead_retune_duration", overhead_defaults.retune_duration),
        pointing_cal_duration=meta.get(
            "overhead_pointing_cal_duration", overhead_defaults.pointing_cal_duration
        ),
        focus_duration=meta.get("overhead_focus_duration", overhead_defaults.focus_duration),
        skydip_duration=meta.get("overhead_skydip_duration", overhead_defaults.skydip_duration),
        planet_cal_duration=meta.get(
            "overhead_planet_cal_duration", overhead_defaults.planet_cal_duration
        ),
        beam_map_duration=meta.get(
            "overhead_beam_map_duration", overhead_defaults.beam_map_duration
        ),
        settle_time=meta.get("overhead_settle_time", overhead_defaults.settle_time),
        min_scan_duration=meta.get(
            "overhead_min_scan_duration", overhead_defaults.min_scan_duration
        ),
        max_scan_duration=meta.get(
            "overhead_max_scan_duration", overhead_defaults.max_scan_duration
        ),
    )

    # planet_targets is stored as a JSON list of strings.
    _pt_raw = meta.get("calibration_planet_targets", None)
    cal_defaults = CalibrationPolicy()
    _planet_targets = (
        tuple(json.loads(_pt_raw))
        if _pt_raw is not None
        else cal_defaults.planet_targets  # class-level default
    )

    # ``beam_map_cadence`` defaults to ``None`` (manual-only); ECSV preserves
    # ``None`` so we can pass it through verbatim. Use a sentinel to distinguish
    # "missing meta key" from "explicit None" since both are valid.
    _MISSING = object()
    _bmc = meta.get("calibration_beam_map_cadence", _MISSING)
    beam_map_cadence = cal_defaults.beam_map_cadence if _bmc is _MISSING else _bmc

    # ``planet_cal_el_step`` is also ``float | None``; use the same sentinel
    # so a stored ``None`` is preserved and only a missing key falls back.
    _ces = meta.get("calibration_planet_cal_el_step", _MISSING)
    if _ces is _MISSING:
        planet_cal_el_step = cal_defaults.planet_cal_el_step
    else:
        planet_cal_el_step = None if _ces is None else float(_ces)

    cal_policy = CalibrationPolicy(
        retune_cadence=meta.get("calibration_retune_cadence", cal_defaults.retune_cadence),
        pointing_cadence=meta.get("calibration_pointing_cadence", cal_defaults.pointing_cadence),
        focus_cadence=meta.get("calibration_focus_cadence", cal_defaults.focus_cadence),
        skydip_cadence=meta.get("calibration_skydip_cadence", cal_defaults.skydip_cadence),
        planet_cal_cadence=meta.get(
            "calibration_planet_cal_cadence", cal_defaults.planet_cal_cadence
        ),
        beam_map_cadence=beam_map_cadence,
        planet_targets=_planet_targets,
        planet_min_elevation=meta.get(
            "calibration_planet_min_elevation", cal_defaults.planet_min_elevation
        ),
        planet_cal_scan=bool(meta.get("calibration_planet_cal_scan", cal_defaults.planet_cal_scan)),
        planet_cal_passes=int(
            meta.get("calibration_planet_cal_passes", cal_defaults.planet_cal_passes)
        ),
        planet_cal_el_step=planet_cal_el_step,
        planet_cal_footprint=str(
            meta.get("calibration_planet_cal_footprint", cal_defaults.planet_cal_footprint)
        ),
    )

    # Detect which optional FYST extension columns are present.
    # Standard TOAST files lack block_type, scan_type, rising, and
    # the patch-geometry columns; those are read as all-science timelines
    # with sensible defaults.
    has_block_type = "block_type" in table.colnames
    has_scan_type = "scan_type" in table.colnames
    has_rising = "rising" in table.colnames
    has_boresight = "boresight_angle" in table.colnames
    has_field_rotation = "field_rotation" in table.colnames
    has_metadata = "ra_center" in table.colnames
    has_az_final = "az_final" in table.colnames

    # A file the writer marked empty holds one placeholder row only
    # because ECSV needs one; the timeline it stands for has no blocks.
    rows = [] if bool(meta.get("timeline_is_empty", False)) else table

    blocks = []
    for row in rows:
        t_start = _read_time(str(row["start_time"]))
        t_stop = _read_time(str(row["stop_time"]))

        block_type = str(row["block_type"]) if has_block_type else "science"
        scan_type = str(row["scan_type"]) if has_scan_type else ""
        rising = bool(row["rising"]) if has_rising else (float(row["azmin"]) % 360 < 180)

        block_meta: TimelineBlockMetadata
        if has_metadata and block_type == "science":
            sci_meta: ScienceBlockMetadata = {
                "ra_center": float(row["ra_center"]),
                "dec_center": float(row["dec_center"]),
                "width": float(row["width"]),
                "height": float(row["height"]),
                "velocity": float(row["velocity"]),
                "scan_params": json.loads(str(row["scan_params_json"]))
                if "scan_params_json" in table.colnames
                else {},
            }
            block_meta = sci_meta
        else:
            # Slew/idle (and any non-science) blocks default to the
            # exhaustive union's empty variant. Calibration-specific keys
            # (``cal_type``, ``target``) are layered in below from
            # ``block_meta_json``.
            empty_meta: EmptyBlockMetadata = {}
            block_meta = empty_meta
            # A non-science block may still carry scan_params (a planet
            # calibration planned as a source-CES pass records its replay
            # parameters there). The writer always dumps scan_params to the
            # scan_params_json column, so restore it here when present and
            # non-empty; slew/idle/retune blocks (empty scan_params) stay
            # metadata-empty.
            if "scan_params_json" in table.colnames:
                sp = json.loads(str(row["scan_params_json"]))
                if sp:
                    block_meta["scan_params"] = sp  # type: ignore[typeddict-item]
        # Merge any extra per-block metadata stored in block_meta_json.
        # For calibration blocks this is where ``cal_type``/``target`` live.
        # The retune-events payload (if present) is decoded back into a
        # tuple of :class:`RetuneEvent` here, mirroring the encoding
        # performed by ``write_timeline``.
        if "block_meta_json" in table.colnames:
            extra = json.loads(str(row["block_meta_json"]))
            if extra:
                extra = _decode_retune_events_from_json(extra)
                # ``block_meta`` is runtime-``dict``; ``.update`` stays
                # legal across all union variants.
                block_meta.update(extra)  # type: ignore[typeddict-item]

        boresight = float(row["boresight_angle"]) if has_boresight else 0.0
        if has_field_rotation:
            boresight = float(row["field_rotation"])

        # A missing column and a NaN both mean "this block ends at azmax":
        # files written before the column existed read back unchanged.
        az_final: float | None = None
        if has_az_final:
            _raw_final = float(row["az_final"])
            if not math.isnan(_raw_final):
                az_final = _raw_final

        block = TimelineBlock(
            t_start=t_start,
            t_stop=t_stop,
            block_type=block_type,
            patch_name=str(row["name"]),
            az_start=float(row["azmin"]),
            az_end=float(row["azmax"]),
            elevation=float(row["el"]),
            scan_index=int(row["scan_index"]),
            subscan_index=int(row["subscan_index"]),
            rising=rising,
            scan_type=scan_type,
            boresight_angle=boresight,
            metadata=block_meta,
            az_final=az_final,
        )
        blocks.append(block)

    # Prefer the persisted timeline window; fall back to the block extents for
    # older files written before these keys existed (and for the empty case).
    _meta_start = meta.get("timeline_start_time")
    _meta_end = meta.get("timeline_end_time")
    if _meta_start is not None and _meta_end is not None:
        tl_start = _read_time(str(_meta_start))
        tl_end = _read_time(str(_meta_end))
    elif blocks:
        tl_start = min(b.t_start for b in blocks)
        tl_end = max(b.t_stop for b in blocks)
    else:
        tl_start = Time("2000-01-01T00:00:00", scale="utc")
        tl_end = tl_start

    return ObservingTimeline(
        blocks=blocks,
        site=site,
        start_time=tl_start,
        end_time=tl_end,
        overhead_model=overhead,
        calibration_policy=cal_policy,
        metadata={k: v for k, v in meta.items() if k not in _KNOWN_META_KEYS},
    )


def _limits_from_meta(meta: dict, default: TelescopeLimits) -> TelescopeLimits:
    """Rebuild the axis limits the header records, an absent key taking ``default``'s value."""
    fields: dict[str, dict[str, float]] = {"azimuth": {}, "elevation": {}}
    for key in _LIMIT_META_KEYS:
        axis, field = _limit_field(key)
        fields[axis][field] = float(meta.get(key, getattr(getattr(default, axis), field)))
    return TelescopeLimits(**{axis: AxisLimits(**values) for axis, values in fields.items()})


def _site_from_meta(meta: dict) -> Site:
    """Reconstruct a ``Site`` from ECSV table metadata.

    If ``site_lat``/``site_lon`` are present and match the FYST
    coordinates to 4 decimal places, the site is built from
    ``get_fyst_site()``, with the persisted axis limits,
    ``nasmyth_port``/``plate_scale`` and Sun settings applied on top; the
    stored altitude is not compared, so a file carrying FYST lat/lon with
    a different ``site_alt`` reads back with the FYST 5611.8 m value.
    Otherwise a custom ``Site`` is constructed from the metadata
    coordinates and the same persisted fields.
    A field whose key the header lacks takes the FYST value (the axis
    limits, in a file written before the header held them); for a
    non-FYST site a :class:`PointingWarning` names the missing keys.
    """
    fyst = get_fyst_site()
    lat = meta.get("site_lat")
    lon = meta.get("site_lon")
    alt = meta.get("site_alt")

    if lat is None or lon is None or alt is None:
        return fyst

    # Coordinates may be stored as bare floats (older files) or as Quantities
    # (deg/deg/m) once TOAST-compatible units were added.
    lat = float(lat.to_value(u.deg)) if isinstance(lat, u.Quantity) else float(lat)
    lon = float(lon.to_value(u.deg)) if isinstance(lon, u.Quantity) else float(lon)
    alt = float(alt.to_value(u.m)) if isinstance(alt, u.Quantity) else float(alt)

    if round(lat, 4) == round(fyst.latitude, 4) and round(lon, 4) == round(fyst.longitude, 4):
        # The FYST site, plus the per-run choices the writer persisted:
        # lowered limits, a "left" port or changed Sun settings survive the trip.
        fyst_site = get_fyst_site(
            sun_exclusion_radius=float(
                meta.get("site_sun_exclusion_radius", fyst.sun_avoidance.exclusion_radius)
            ),
            sun_warning_radius=float(
                meta.get("site_sun_warning_radius", fyst.sun_avoidance.warning_radius)
            ),
            sun_avoidance_enabled=bool(meta.get("site_sun_enabled", fyst.sun_avoidance.enabled)),
        )
        return dataclasses.replace(
            fyst_site,
            telescope_limits=_limits_from_meta(meta, fyst.telescope_limits),
            nasmyth_port=str(meta.get("site_nasmyth_port", fyst.nasmyth_port)),
            plate_scale=float(meta.get("site_plate_scale", fyst.plate_scale)),
        )

    # Non-FYST site: a field the header lacks takes the FYST value, which a
    # consumer recomputing pose or feasibility from the loaded site needs to know.
    restored = (
        *_LIMIT_META_KEYS,
        "site_nasmyth_port",
        "site_plate_scale",
        "site_sun_enabled",
        "site_sun_exclusion_radius",
        "site_sun_warning_radius",
    )
    missing = [key for key in restored if key not in meta]
    if missing:
        warnings.warn(
            f"Reconstructing a non-FYST Site from ECSV whose header lacks {', '.join(missing)}: "
            "the Site fields they hold take the FYST values.",
            PointingWarning,
            stacklevel=3,
        )
    return Site(
        name=str(meta.get("site_name", "custom")),
        description=str(meta.get("site_description", "")),
        latitude=lat,
        longitude=lon,
        elevation=alt,
        telescope_limits=_limits_from_meta(meta, fyst.telescope_limits),
        sun_avoidance=SunAvoidanceConfig(
            enabled=bool(meta.get("site_sun_enabled", fyst.sun_avoidance.enabled)),
            exclusion_radius=float(
                meta.get("site_sun_exclusion_radius", fyst.sun_avoidance.exclusion_radius)
            ),
            warning_radius=float(
                meta.get("site_sun_warning_radius", fyst.sun_avoidance.warning_radius)
            ),
        ),
        nasmyth_port=str(meta.get("site_nasmyth_port", fyst.nasmyth_port)),
        plate_scale=float(meta.get("site_plate_scale", fyst.plate_scale)),
    )


def _empty_row() -> dict:
    """Create an empty row for an empty timeline.

    Returns
    -------
    dict
        Row with TOAST-compatible column names and zero values.
    """
    t0 = Time("2000-01-01T00:00:00", scale="utc")
    return {
        "start_time": _iso_utc(t0),
        "stop_time": _iso_utc(t0),
        "boresight_angle": 0.0,
        "field_rotation": 0.0,
        "name": "",
        "azmin": 0.0,
        "azmax": 0.0,
        "az_final": float("nan"),
        "el": 0.0,
        "scan_index": 0,
        "subscan_index": 0,
        "block_type": "idle",
        "scan_type": "",
        "rising": True,
        "ra_center": 0.0,
        "dec_center": 0.0,
        "width": 0.0,
        "height": 0.0,
        "velocity": 0.0,
        "scan_params_json": "{}",
        "block_meta_json": "{}",
    }


_KNOWN_META_KEYS = frozenset(
    {
        "timeline_start_time",
        "timeline_end_time",
        "timeline_is_empty",
        "site_name",
        "site_description",
        "telescope_name",
        "site_lat",
        "site_lon",
        "site_alt",
        "site_nasmyth_port",
        "site_plate_scale",
        "site_sun_enabled",
        "site_sun_exclusion_radius",
        "site_sun_warning_radius",
        *_LIMIT_META_KEYS,
        # OverheadModel.
        "overhead_retune_duration",
        "overhead_pointing_cal_duration",
        "overhead_focus_duration",
        "overhead_skydip_duration",
        "overhead_planet_cal_duration",
        "overhead_beam_map_duration",
        "overhead_settle_time",
        "overhead_min_scan_duration",
        "overhead_max_scan_duration",
        # CalibrationPolicy.
        "calibration_retune_cadence",
        "calibration_pointing_cadence",
        "calibration_focus_cadence",
        "calibration_skydip_cadence",
        "calibration_planet_cal_cadence",
        "calibration_beam_map_cadence",
        "calibration_planet_targets",
        "calibration_planet_min_elevation",
        "calibration_planet_cal_scan",
        "calibration_planet_cal_passes",
        "calibration_planet_cal_el_step",
        "calibration_planet_cal_footprint",
    }
)
