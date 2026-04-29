"""TOAST-compatible ECSV timeline I/O.

Reads and writes observation timelines in TOAST v5 (ECSV) format
with FYST-specific extensions for calibration blocks.
"""

import json
from pathlib import Path

from astropy.table import Table
from astropy.time import Time

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
    EmptyBlockMetadata,
    ObservingTimeline,
    OverheadModel,
    ScienceBlockMetadata,
    TimelineBlock,
    TimelineBlockMetadata,
)
from .utils import compute_nasmyth_rotation

__all__ = [
    "read_timeline",
    "write_timeline",
]


# Metadata key used for per-block retune event provenance. When a block's
# ``metadata`` carries this key (value: a sequence of
# :class:`~fyst_trajectories.trajectory.RetuneEvent`), the ECSV writer
# encodes it inside the per-block JSON column as a list of
# ``[t_start, duration]`` float pairs. The reader decodes it back into a
# tuple of :class:`RetuneEvent`. This reuses the existing
# ``block_meta_json`` extra-payload channel — see the JSON encoding/decoding
# helpers below — rather than adding a sidecar table or new columns.
_RETUNE_EVENTS_META_KEY = "retune_events"


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
    ``rising``, plus metadata columns for science blocks:
    ``ra_center``, ``dec_center``, ``width``, ``height``, ``velocity``,
    ``scan_params_json``).

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
            bangle = compute_nasmyth_rotation(
                0.5 * (block.az_start + block.az_end),
                block.elevation,
                timeline.site,
            )

        meta = block.metadata
        # TOAST canonical column names azmin/azmax are preserved even though
        # the Python attributes are az_start/az_end. For slew blocks the
        # columns therefore carry the "from"/"to" azimuths directly and may
        # not satisfy azmin <= azmax — consumers that rely on ordered bounds
        # must filter on block_type first.
        # Encode any RetuneEvent payload into JSON-native shape before
        # dumping the extra-metadata column.
        meta_for_json = _encode_retune_events_for_json(dict(meta))
        rows.append(
            {
                "start_time": block.t_start.iso,
                "stop_time": block.t_stop.iso,
                "boresight_angle": bangle,
                "name": block.patch_name,
                "azmin": block.az_start,
                "azmax": block.az_end,
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

    if not rows:
        rows = [_empty_row()]

    table = Table(rows)

    table.meta["site_name"] = timeline.site.name
    table.meta["telescope_name"] = "FYST"
    table.meta["site_lat"] = timeline.site.latitude
    table.meta["site_lon"] = timeline.site.longitude
    table.meta["site_alt"] = timeline.site.elevation
    # OverheadModel — persist ALL fields with overhead_ prefix.
    table.meta["overhead_retune_duration"] = timeline.overhead_model.retune_duration
    table.meta["overhead_pointing_cal_duration"] = timeline.overhead_model.pointing_cal_duration
    table.meta["overhead_focus_duration"] = timeline.overhead_model.focus_duration
    table.meta["overhead_skydip_duration"] = timeline.overhead_model.skydip_duration
    table.meta["overhead_planet_cal_duration"] = timeline.overhead_model.planet_cal_duration
    table.meta["overhead_beam_map_duration"] = timeline.overhead_model.beam_map_duration
    table.meta["overhead_settle_time"] = timeline.overhead_model.settle_time
    table.meta["overhead_min_scan_duration"] = timeline.overhead_model.min_scan_duration
    table.meta["overhead_max_scan_duration"] = timeline.overhead_model.max_scan_duration
    # CalibrationPolicy — persist ALL fields with calibration_ prefix.
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
    table.meta.update(timeline.metadata)

    table.write(str(path), format="ascii.ecsv", overwrite=True)


def read_timeline(path: str | Path) -> ObservingTimeline:
    """Read a timeline from a TOAST-compatible ECSV file.

    Handles both standard TOAST format (science blocks only, no
    ``block_type`` column) and FYST extended format with calibration
    blocks, scan metadata, and patch geometry columns.

    Parameters
    ----------
    path : str or Path
        Input file path.

    Returns
    -------
    ObservingTimeline
        Loaded timeline.
    """
    path = Path(path)
    table = Table.read(str(path), format="ascii.ecsv")

    meta = table.meta
    site = _site_from_meta(meta)

    # Use the dataclass defaults as fall-backs so the I/O defaults can never
    # drift from the class defaults — this is the same pattern that surfaced
    # the BEAM_MAP regression and the pointing_cadence default mismatch.
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
    )

    # Detect which optional FYST extension columns are present.
    # Standard TOAST files lack block_type, scan_type, rising, and
    # the patch-geometry columns; those are read as all-science timelines
    # with sensible defaults.
    has_block_type = "block_type" in table.colnames
    has_scan_type = "scan_type" in table.colnames
    has_rising = "rising" in table.colnames
    has_boresight = "boresight_angle" in table.colnames
    has_metadata = "ra_center" in table.colnames

    blocks = []
    for row in table:
        t_start = Time(str(row["start_time"]), scale="utc")
        t_stop = Time(str(row["stop_time"]), scale="utc")

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
        )
        blocks.append(block)

    if blocks:
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


def _site_from_meta(meta: dict) -> Site:
    """Reconstruct a ``Site`` from ECSV table metadata.

    If ``site_lat``/``site_lon``/``site_alt`` are present and match the
    FYST coordinates to 4 decimal places, ``get_fyst_site()`` is used so
    the returned site has the full FYST default limits and atmosphere.
    Otherwise a custom ``Site`` is constructed using the metadata
    coordinates together with the default FYST telescope limits and
    sun avoidance settings (which are not currently persisted).
    """
    fyst = get_fyst_site()
    lat = meta.get("site_lat")
    lon = meta.get("site_lon")
    alt = meta.get("site_alt")

    if lat is None or lon is None or alt is None:
        return fyst

    lat = float(lat)
    lon = float(lon)
    alt = float(alt)

    if round(lat, 4) == round(fyst.latitude, 4) and round(lon, 4) == round(fyst.longitude, 4):
        return fyst

    # Non-FYST site: build a custom Site using the stored coordinates plus
    # the default FYST mechanical limits and sun-avoidance config (which
    # are not serialised separately in v0.3 ECSV files).
    return Site(
        name=str(meta.get("site_name", "custom")),
        description=str(meta.get("site_description", "")),
        latitude=lat,
        longitude=lon,
        elevation=alt,
        atmosphere=None,
        telescope_limits=TelescopeLimits(
            azimuth=AxisLimits(
                min=fyst.telescope_limits.azimuth.min,
                max=fyst.telescope_limits.azimuth.max,
                max_velocity=fyst.telescope_limits.azimuth.max_velocity,
                max_acceleration=fyst.telescope_limits.azimuth.max_acceleration,
            ),
            elevation=AxisLimits(
                min=fyst.telescope_limits.elevation.min,
                max=fyst.telescope_limits.elevation.max,
                max_velocity=fyst.telescope_limits.elevation.max_velocity,
                max_acceleration=fyst.telescope_limits.elevation.max_acceleration,
            ),
        ),
        sun_avoidance=SunAvoidanceConfig(
            enabled=fyst.sun_avoidance.enabled,
            exclusion_radius=fyst.sun_avoidance.exclusion_radius,
            warning_radius=fyst.sun_avoidance.warning_radius,
        ),
        nasmyth_port=fyst.nasmyth_port,
        plate_scale=fyst.plate_scale,
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
        "start_time": t0.iso,
        "stop_time": t0.iso,
        "boresight_angle": 0.0,
        "name": "",
        "azmin": 0.0,
        "azmax": 0.0,
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
        "site_name",
        "telescope_name",
        "site_lat",
        "site_lon",
        "site_alt",
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
    }
)
