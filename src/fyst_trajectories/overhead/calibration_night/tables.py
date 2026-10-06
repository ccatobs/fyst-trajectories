"""Per-body scan-parameter tables binned by elevation.

A :class:`ScanParameterTable` holds, for one body or for the shared
default, a reference azimuth throw for each elevation bin together with a
reference dwell, the table's scan time. The planner applies each only
when the policy asks for it (``use_table_throw``, ``use_table_dwell``);
by default a pass sweeps the throw solved from the footprint and scans
the solved crossing. The shipped :data:`DEFAULT_SCAN_TABLES` are
instrument-team commissioning defaults pending on-sky testing;
:func:`load_scan_tables` reads a caller's own table file so the defaults
can be replaced without a release.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_SCAN_TABLES",
    "ElevationBin",
    "ScanParameterTable",
    "load_scan_tables",
    "table_for",
]

# Bins that meet within this many degrees are contiguous; the tables are
# authored by hand in whole degrees, so float noise is the only tolerance needed.
_CONTIGUITY_TOL_DEG = 1e-9


@dataclass(frozen=True)
class ElevationBin:
    """One elevation bin of a scan-parameter table.

    Parameters
    ----------
    el_lo : float
        Lower elevation bound in degrees, inclusive.
    el_hi : float
        Upper elevation bound in degrees, exclusive except on a table's
        top bin, where it is inclusive.
    az_throw : float
        Reference azimuth throw of the swept window in degrees for this
        bin (the table's "scan width"), swept only when a policy asks for
        it.
    dwell_reference : float
        Reference time on source in seconds (the table's "scan time"),
        applied only when a policy asks for it.

    Raises
    ------
    ValueError
        If the bounds are not ordered or a value is not positive.
    """

    el_lo: float
    el_hi: float
    az_throw: float
    dwell_reference: float

    def __post_init__(self) -> None:
        for name in ("el_lo", "el_hi", "az_throw", "dwell_reference"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if self.el_lo >= self.el_hi:
            raise ValueError(f"el_lo must be below el_hi, got {self.el_lo} >= {self.el_hi}")
        if self.az_throw <= 0.0:
            raise ValueError(f"az_throw must be positive, got {self.az_throw}")
        if self.dwell_reference <= 0.0:
            raise ValueError(f"dwell_reference must be positive, got {self.dwell_reference}")

    @property
    def el_centre(self) -> float:
        """Midpoint elevation of the bin in degrees."""
        return 0.5 * (self.el_lo + self.el_hi)


@dataclass(frozen=True)
class ScanParameterTable:
    """Scan parameters for one body, binned by boresight elevation.

    Parameters
    ----------
    bins : tuple of ElevationBin
        Contiguous, non-overlapping bins in ascending elevation.

    Raises
    ------
    ValueError
        If there are no bins, or the bins overlap or leave a gap.

    Examples
    --------
    >>> table = ScanParameterTable(
    ...     (
    ...         ElevationBin(30.0, 35.0, az_throw=2.44, dwell_reference=600.0),
    ...         ElevationBin(35.0, 40.0, az_throw=2.61, dwell_reference=600.0),
    ...     )
    ... )
    >>> table.for_elevation(37.0).az_throw
    2.61
    >>> table.el_range
    (30.0, 40.0)
    """

    bins: tuple[ElevationBin, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.bins, key=lambda b: b.el_lo))
        if not ordered:
            raise ValueError("bins must not be empty")
        for lower, upper in zip(ordered, ordered[1:]):
            if upper.el_lo < lower.el_hi - _CONTIGUITY_TOL_DEG:
                raise ValueError(
                    f"bins overlap: [{lower.el_lo}, {lower.el_hi}) and "
                    f"[{upper.el_lo}, {upper.el_hi})"
                )
            if upper.el_lo > lower.el_hi + _CONTIGUITY_TOL_DEG:
                raise ValueError(
                    f"bins leave a gap between {lower.el_hi} and {upper.el_lo} deg; "
                    "tables must be contiguous"
                )
        object.__setattr__(self, "bins", ordered)

    @property
    def el_range(self) -> tuple[float, float]:
        """The elevation coverage ``(el_lo, el_hi)`` of the table in degrees."""
        return (self.bins[0].el_lo, self.bins[-1].el_hi)

    def for_elevation(self, el: float) -> ElevationBin | None:
        """Return the bin covering ``el``, or ``None`` outside the table's coverage.

        The top bin includes its upper bound.
        """
        lo, hi = self.el_range
        if el < lo or el > hi:
            return None
        for b in self.bins:
            if b.el_lo <= el < b.el_hi:
                return b
        return self.bins[-1]

    def az_throw_at(self, el: float) -> float:
        """Azimuth throw in degrees at boresight elevation ``el``.

        Inside the coverage this is the bin's throw. Above the top bin the
        throw is extrapolated from the top bin at constant on-sky width:
        the top bin's throw times ``cos(top centre) / cos(el)``, so a
        window that tracked a fixed sky extent keeps tracking it higher up.

        Raises
        ------
        ValueError
            If ``el`` is below the table's coverage.
        """
        b = self.for_elevation(el)
        if b is not None:
            return b.az_throw
        lo, _ = self.el_range
        if el < lo:
            raise ValueError(
                f"elevation {el:.2f} deg is below the table's coverage starting at {lo:.2f} deg"
            )
        top = self.bins[-1]
        cos_el = math.cos(math.radians(el))
        if cos_el <= 0.0:
            raise ValueError(f"elevation {el:.2f} deg has no finite azimuth throw")
        return top.az_throw * math.cos(math.radians(top.el_centre)) / cos_el


def _minutes(m: float) -> float:
    return 60.0 * m


# Instrument-team commissioning defaults, pending on-sky testing.
# Jupiter, Saturn and Neptune share one table; Uranus, which is not observed
# above about 40 degrees at commissioning, has its own two bins. Throws are
# degrees of azimuth; the reference dwells are the table's scan times.
_SHARED_BINS = (
    ElevationBin(30.0, 35.0, az_throw=2.44, dwell_reference=_minutes(10)),
    ElevationBin(35.0, 40.0, az_throw=2.61, dwell_reference=_minutes(10)),
    ElevationBin(40.0, 45.0, az_throw=2.83, dwell_reference=_minutes(13)),
    ElevationBin(45.0, 50.0, az_throw=3.11, dwell_reference=_minutes(15)),
)
_URANUS_BINS = (
    ElevationBin(30.0, 35.0, az_throw=2.44, dwell_reference=_minutes(15)),
    ElevationBin(35.0, 40.0, az_throw=2.61, dwell_reference=_minutes(15)),
)

#: The shipped tables, keyed by body name with ``"default"`` for the shared one.
DEFAULT_SCAN_TABLES: Mapping[str, ScanParameterTable] = {
    "default": ScanParameterTable(_SHARED_BINS),
    "uranus": ScanParameterTable(_URANUS_BINS),
}


def table_for(tables: Mapping[str, ScanParameterTable], body: str) -> ScanParameterTable:
    """Return the table for ``body``, falling back to ``"default"``.

    The body name is lower-cased before the lookup, so ``tables`` must be
    keyed in lower case for a per-body entry to be found;
    :meth:`~fyst_trajectories.overhead.NightContext.build` and
    :func:`load_scan_tables` both lower-case the keys they resolve.

    Raises
    ------
    KeyError
        If neither the body nor a default table is present.
    """
    key = body.lower()
    if key in tables:
        return tables[key]
    if "default" in tables:
        return tables["default"]
    raise KeyError(f"no scan-parameter table for {body!r} and no 'default' table")


_HEADER_ALIASES = {
    "body": ("body", "target", "source"),
    "el_lo": ("el_lo", "elevation_lo", "el_min", "elev_lo"),
    "el_hi": ("el_hi", "elevation_hi", "el_max", "elev_hi"),
    "scan_time_min": ("scan_time_min", "scan time", "scan_time", "time", "scan time (min)"),
    "az_throw": ("az_throw", "scan width", "scan_width", "width", "scan width (deg az)"),
}


def _resolve_columns(fieldnames: list[str]) -> dict[str, str]:
    lowered = {name.strip().lower(): name for name in fieldnames}
    columns: dict[str, str] = {}
    for key, aliases in _HEADER_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                columns[key] = lowered[alias]
                break
        else:
            raise ValueError(
                f"scan-table CSV is missing a {key!r} column (accepted headings: "
                f"{', '.join(aliases)}); found {fieldnames}"
            )
    return columns


def _number(text: str, what: str) -> float:
    cleaned = text.strip().lower()
    for suffix in ("min", "minutes", "deg", "degrees"):
        cleaned = cleaned.removesuffix(suffix).strip()
    try:
        return float(cleaned)
    except ValueError:
        raise ValueError(f"cannot read {what} from {text!r}") from None


def load_scan_tables(csv_path: str | Path) -> dict[str, ScanParameterTable]:
    """Read scan-parameter tables from a CSV file, one row per elevation bin.

    The file needs a body column (``body``; ``"default"`` names the shared
    table), the bin bounds (``el_lo``, ``el_hi``), the scan time in minutes
    (``scan_time_min``, also accepted as ``Scan Time``) and the scan width
    in degrees of azimuth (``az_throw``, also accepted as
    ``Scan Width (deg az)``). Headings are case-insensitive; a trailing
    ``min`` or ``deg`` on a value is tolerated.

    Parameters
    ----------
    csv_path : str or Path
        The file to read.

    Returns
    -------
    dict of str to ScanParameterTable
        Tables keyed by lower-case body name, ready for
        :func:`plan_calibration_night`'s ``tables``.

    Raises
    ------
    ValueError
        If a required column is missing, a value does not parse, or a
        body's bins are not contiguous.
    """
    path = Path(csv_path)
    rows_by_body: dict[str, list[ElevationBin]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        columns = _resolve_columns(list(reader.fieldnames))
        for row in reader:
            body = row[columns["body"]].strip().lower()
            if not body:
                continue
            rows_by_body.setdefault(body, []).append(
                ElevationBin(
                    el_lo=_number(row[columns["el_lo"]], "el_lo"),
                    el_hi=_number(row[columns["el_hi"]], "el_hi"),
                    az_throw=_number(row[columns["az_throw"]], "az_throw"),
                    dwell_reference=_minutes(_number(row[columns["scan_time_min"]], "scan time")),
                )
            )
    if not rows_by_body:
        raise ValueError(f"{path} holds no table rows")
    return {body: ScanParameterTable(tuple(bins)) for body, bins in rows_by_body.items()}
