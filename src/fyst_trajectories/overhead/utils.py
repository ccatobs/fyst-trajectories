"""Scheduling utility functions.

Provides what the scheduler and the calibration-night planner share: the
UTC instant both hold every time as, the scalar cable-wrap placement, the
check of a policy's footprint tag, and the canonical module name and the
search start a recorded pass carries.
"""

import math
from numbers import Real

import numpy as np
from astropy.time import Time

from ..patterns.utils import normalize_azimuth
from ..primecam import PRIMECAM_MODULES, get_primecam_offset
from ..site import Site
from .exceptions import ScanParamsSchemaError

__all__: list[str] = []


def _utc_instant(t: Time | str) -> Time:
    """Return ``t`` as the simulator holds an instant: a UTC ``Time`` with astropy's defaults.

    A string is read as UTC. A UTC ``Time`` without a location and with
    astropy's default ``precision`` (3) and ``out_subfmt`` (``"*"``) is
    returned unchanged; any other ``Time`` is rebuilt from its UTC Julian
    date, without a location and with those defaults. Both planners hold
    every time this way, so a night is planned on the UTC instants its times
    name, as exactly as astropy converts them (from TT, TAI, TDB and TCG
    exactly, from UT1 and TCB to within a few units in the last place);
    every time the simulator records is a UTC string to the millisecond, the
    form its readers parse; and a pass's ``search_start``, which keeps the
    UTC Julian date alone, restores the very instant the planner searched
    from (a location on the ``Time`` would enter its conversion to TDB, and
    so the planned pass).
    """
    if isinstance(t, str):
        return Time(t, scale="utc")
    if t.scale == "utc" and t.location is None and t.precision == 3 and t.out_subfmt == "*":
        return t
    utc = t.utc
    instant = Time(utc.jd1, utc.jd2, format="jd", scale="utc")
    instant.format = "isot"
    return instant


def _search_start_record(start: Time) -> list[float]:
    """Record where a pass's search began as its two-part UTC Julian date ``[jd1, jd2]``.

    The two parts are the ``jd1`` and ``jd2`` of the instant as
    :func:`_utc_instant` holds it, so :func:`_search_start_time` restores
    that instant exactly; an ISO string, even at nanosecond precision,
    generally does not.
    """
    instant = _utc_instant(start)
    return [float(instant.jd1), float(instant.jd2)]


def _search_start_time(record: object) -> Time:
    """Restore the instant :func:`_search_start_record` recorded, exactly.

    Raises
    ------
    ScanParamsSchemaError
        If ``record`` is not two finite numbers.
    """
    if not (
        isinstance(record, (list, tuple))
        and len(record) == 2
        and all(isinstance(x, Real) and not isinstance(x, bool) for x in record)
        and all(math.isfinite(x) for x in record)
    ):
        raise ScanParamsSchemaError(
            "search_start must be two finite numbers, the UTC Julian date [jd1, jd2] the "
            f"planner's search began at; got {record!r}"
        )
    start = Time(float(record[0]), float(record[1]), format="jd", scale="utc")
    start.format = "isot"
    return start


def _require_module_tag(field: str, tag: object) -> None:
    """Refuse a policy's footprint tag unless it names one Prime-Cam module.

    Parameters
    ----------
    field : str
        The policy field the tag came from, named in the message.
    tag : object
        The value to check.

    Raises
    ------
    ValueError
        If ``tag`` is not a str, or is a str that
        :func:`~fyst_trajectories.primecam.get_primecam_offset` does not
        resolve (the message then lists the names it takes).
    """
    if not isinstance(tag, str):
        raise ValueError(f"{field} must be a str naming one Prime-Cam module, got {tag!r}")
    try:
        get_primecam_offset(tag)
    except KeyError as exc:
        raise ValueError(f"{field}: {exc.args[0]}") from None


def _canonical_module_name(tag: str) -> str:
    """Return the canonical name of the Prime-Cam module ``tag`` names.

    That is the first key of
    :data:`~fyst_trajectories.primecam.PRIMECAM_MODULES` naming the same
    module, the name the per-module coverage records use: ``"c"`` for every
    spelling of the centre module (``"C"``, ``"center"``, ``"IM0"``) and
    ``"i1"`` .. ``"i6"`` for the ring. A recorded pass names its module this
    way because an execution layer may compare the name as a string.

    Raises
    ------
    KeyError
        If ``tag`` names no Prime-Cam module.
    """
    module = get_primecam_offset(tag)
    return next(name for name, offset in PRIMECAM_MODULES.items() if offset is module)


def _normalize_az(az: float, site: Site, ref: float | None = None) -> float:
    """Normalize a scalar azimuth into the site's cable-wrap window.

    Wraps :func:`fyst_trajectories.patterns.utils.normalize_azimuth` (which
    operates on arrays) for the scalar azimuths the scheduler and the
    calibration-night planner carry.
    Raw astropy azimuths are in ``[0, 360)``; the slew-time and boresight
    math must compare them in the telescope's ``[az_min, az_max]`` window
    or a north-straddling pair measures the long way round instead of the
    short one and flips the boresight by ~180 deg.

    With ``ref`` given, the in-limits 360-degree representative nearest
    ``ref`` is returned, so the scheduler models the mount's direct
    cable-wrap move from its current position. Without ``ref`` each
    scalar independently takes the representative nearest the window
    centre, which relocates the wrap seam rather than removing it; pass
    ``ref`` whenever a coherent frame with another azimuth is required.

    Parameters
    ----------
    az : float
        Azimuth in degrees (typically raw astropy ``[0, 360)``).
    site : Site
        Site providing the azimuth limits.
    ref : float or None, optional
        Reference azimuth (already in the cable-wrap window) selecting
        among the in-limits representatives. Default None.

    Returns
    -------
    float
        Azimuth shifted into ``[az_min, az_max]``. When two in-limits
        representatives lie equally far from ``ref`` (a half turn either
        way), the window-centre one is kept.

    Warns
    -----
    PointingWarning
        If no representative of the azimuth lies inside the window, which
        happens only for a window narrower than 360 deg; the window-centre
        representative, outside the limits, is then returned.
    """
    base = float(normalize_azimuth(np.array([az], dtype=float), site)[0])
    if ref is None:
        return base
    limits = site.telescope_limits.azimuth
    best = base
    for cand in (base - 360.0, base + 360.0):
        if limits.min <= cand <= limits.max and abs(cand - ref) < abs(best - ref):
            best = cand
    return best
