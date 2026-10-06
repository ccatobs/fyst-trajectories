"""Golden-output test of the three source-CES planners.

The cases below call ``compute_source_ces_params``, ``plan_source_ces`` and
``plan_source_ces_passes`` over the geometry the planners support (planets
rising and setting, a window with the mode left to the planner, off-centre and
bare-offset footprints, the seven-module array, fixed and proper-motion
sources, the scan-geometry overrides, anchored starts, pass sequences and three
refusals) and compare each outcome with ``tests/data/source_ces_golden.json``.
A refactor of the planners must leave every case passing without a re-cut.

What a case records
-------------------
For a built block: the computed parameters key by key, the duration, every
field of its ``ConstantElScanConfig``, the trajectory metadata, the summary
text, the start time, the ``scan_flag`` counts per value, and for each float
array of the trajectory and for the two arrays ``source_ces_focal_plane_track``
returns: the length, the first and last value, the minimum, the maximum, the
sum and the sum of squares. For a refusal: the exception type and message. For
every case: the warnings the call emitted, in order, each with its category,
its message and whether Python attributed it to the line of this module that
calls the entry point (``"caller"``) or to any other place (``"library"``), so
a ``stacklevel`` one frame short or one frame long is seen. The advisories are
issued with a ``stacklevel`` that points at the caller of the public entry
point, and nothing else in the suite guards that attribution.

How a run is compared with the fixture
--------------------------------------
Strings, integers, booleans, ``None``, counts, key sets, warning lists and
exceptions compare exactly. Scalar floats (the computed parameters, the
duration, the config and metadata floats) compare at a relative 1e-12 with no
absolute term. An array summary compares its length exactly and each of its
other six values within the larger of a relative 1e-10 and an absolute 1e-12
times the summary's natural scale: with ``A`` the larger magnitude of the
recorded minimum and maximum, the scale is ``A`` for the first, last, minimum
and maximum, ``n * A`` for the sum and ``n * A**2`` for the sum of squares.
The absolute term is there because the fixture is cut on one platform and
compared on others, and vectorised floating-point maths differs between them
in the last bits: a focal-plane track whose samples reach about 1 deg can sum
to well under 1 deg, so the sum cancels and its relative error is
ill-conditioned (a Windows record differed from Linux by 1.9e-10 relative in
such a sum, about 1e-10 deg over 3562 samples), while every scalar agreed to
3e-14. No looser tolerance is used anywhere.

Re-cutting the fixture
----------------------
Re-cut only in a change whose changelog entry says these numbers moved, never
to make a refactor pass. The re-cut runs under pytest, so that the vendored
IERS table ``tests/conftest.py`` pins applies to the record as it does to the
comparison (a bare script records the September 2026 Saturn cases from
whatever table the interpreter holds)::

    FYST_RECUT_GOLDEN=1 pytest tests/test_source_ces_golden.py --run-slow

The run records every case and writes the whole fixture, or fails and writes
nothing when any case is not run or not recorded (no ``--run-slow``, a ``-k``
selection, an error). The header records the commit, the package versions,
the platform and the last measured day of the IERS table; it is information
and is not compared.

The exact mode
--------------
For a refactor that must be bit-identical, set ``FYST_GOLDEN_EXACT`` to a file
path outside the repository and run the module with ``--run-slow``: besides
comparing, the run writes every recorded value at full precision plus the
SHA-256 of every raw array of every built block (and of the focal-plane
track). Run it before and after the change in the same environment and require
the two files to be byte-identical. The file is never committed and never
compared across machines, since the raw bytes depend on the CPU's vector
maths.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
import platform
import subprocess
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import astropy
import erfa
import numpy as np
import pytest
import scipy
from astropy.time import Time
from astropy.utils import iers

from fyst_trajectories import InstrumentOffset, get_fyst_site
from fyst_trajectories.planning import (
    compute_source_ces_params,
    plan_source_ces,
    plan_source_ces_passes,
    source_ces_focal_plane_track,
)
from fyst_trajectories.primecam import PRIMECAM_MODULES

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
_FIXTURE = _THIS_FILE.parent / "data" / "source_ces_golden.json"

_RECUT = os.environ.get("FYST_RECUT_GOLDEN") == "1"
_EXACT_PATH = os.environ.get("FYST_GOLDEN_EXACT") or None
# A re-cut or an exact run must record every case, so neither mode skips the
# slow cases: each case instead fails at once without the run-slow option.
_SWITCHED = _RECUT or _EXACT_PATH is not None

_SCALAR_REL = 1e-12
_SUMMARY_REL = 1e-10
_SUMMARY_ABS = 1e-12
_FLOAT_ARRAYS = ("times", "az", "el", "az_vel", "el_vel")

SITE = get_fyst_site()
_JUPITER_NIGHT = Time("2026-03-15T00:00:00", scale="utc")
_SATURN_WINDOW = (
    Time("2026-09-10T22:00:00", scale="utc"),
    Time("2026-09-11T10:00:00", scale="utc"),
)
_FULL_ARRAY = [PRIMECAM_MODULES[k] for k in ("c", "i1", "i2", "i3", "i4", "i5", "i6")]

_JUP = dict(body="jupiter", night=_JUPITER_NIGHT, site=SITE)
_SAT = dict(body="saturn", window=_SATURN_WINDOW, site=SITE)
_FIXED = dict(
    ra=180.0,
    dec=-30.0,
    window=(Time("2026-03-15T07:30:00", scale="utc"), Time("2026-03-15T10:00:00", scale="utc")),
    site=SITE,
)
_PM = dict(
    _FIXED,
    pm_ra=1000.0,
    pm_dec=500.0,
    ref_epoch=Time("2000-01-01T00:00:00", scale="utc"),
)
_JUP_ANCHOR = dict(
    body="jupiter",
    footprint="c",
    start_time=Time("2026-03-15T21:41:00", scale="utc"),
    site=SITE,
)
_SAT_PASSES_ANCHOR = dict(
    body="saturn",
    footprint="c",
    n_passes=2,
    start_time=Time("2026-09-11T02:15:00", scale="utc"),
    site=SITE,
    az_speed=1.5,
    az_accel=1.0,
)


class _Case(NamedTuple):
    """One golden case: its name, entry point, keyword arguments and tier."""

    name: str
    entry: str
    kwargs: dict[str, Any]
    slow: bool


_CASES = (
    _Case("jup_c_rising", "params", dict(_JUP, footprint="c", el_bore=35.0, mode="rising"), False),
    _Case("jup_c_setting", "params", dict(_JUP, footprint="c", el_bore=35.0, mode="setting"), True),
    _Case(
        "jup_c_rising_plan", "plan", dict(_JUP, footprint="c", el_bore=35.0, mode="rising"), True
    ),
    _Case(
        "jup_c_setting_plan", "plan", dict(_JUP, footprint="c", el_bore=35.0, mode="setting"), True
    ),
    _Case("sat_c_auto_mode", "plan", dict(_SAT, footprint="c", el_bore=32.5), False),
    _Case(
        "sat_i1_offcentre", "plan", dict(_SAT, footprint="i1", el_bore=40.0, mode="rising"), False
    ),
    _Case(
        "sat_offset_footprint",
        "plan",
        dict(_SAT, footprint=InstrumentOffset(dx=12.0, dy=-9.0), el_bore=40.0, mode="rising"),
        True,
    ),
    _Case(
        "jup_full_array",
        "plan",
        dict(_JUP, footprint=_FULL_ARRAY, el_bore=35.0, mode="rising"),
        True,
    ),
    _Case(
        "fixed_setting", "plan", dict(_FIXED, footprint="c", el_bore=40.0, mode="setting"), False
    ),
    _Case("pm_setting", "plan", dict(_PM, footprint="c", el_bore=40.0, mode="setting"), False),
    _Case(
        "sat_overrides",
        "plan",
        dict(
            _SAT,
            footprint="c",
            el_bore=32.5,
            mode="rising",
            az_speed=1.5,
            az_accel=1.5,
            az_throw=2.44,
        ),
        False,
    ),
    _Case(
        "sat_dwell",
        "plan",
        dict(_SAT, footprint="c", el_bore=32.5, mode="rising", dwell=200.0, az_padding=0.0),
        False,
    ),
    _Case(
        "sat_zero_padding",
        "plan",
        dict(_SAT, footprint="c", el_bore=32.5, mode="rising", az_padding=0.0),
        False,
    ),
    _Case(
        "sat_branch_rot_vaz",
        "plan",
        dict(
            _SAT,
            footprint="c",
            el_bore=32.5,
            mode="setting",
            az_branch=0.0,
            boresight_rot=10.0,
            v_az=0.001,
            sampling_step_seconds=20.0,
            timestep=0.2,
        ),
        False,
    ),
    _Case(
        "jup_partial",
        "plan",
        dict(_JUP, footprint=_FULL_ARRAY, el_bore=43.5, mode="rising", allow_partial=True),
        True,
    ),
    _Case("jup_anchor_plan", "plan", _JUP_ANCHOR, True),
    _Case("jup_anchor_params", "params", _JUP_ANCHOR, True),
    _Case(
        "jup_passes_3",
        "passes",
        dict(_JUP, footprint="c", el_bore=35.0, n_passes=3, mode="rising"),
        True,
    ),
    _Case("sat_passes_anchor", "passes", _SAT_PASSES_ANCHOR, True),
    _Case(
        "sat_passes_anchor_zero_padding",
        "passes",
        dict(_SAT_PASSES_ANCHOR, az_padding=0.0),
        False,
    ),
    # Refusals: the exception type and message are part of the contract.
    _Case(
        "err_not_reached", "params", dict(_JUP, footprint="c", el_bore=80.0, mode="rising"), True
    ),
    _Case("err_dwell_long", "params", dict(_SAT, footprint="c", el_bore=32.5, dwell=3000.0), False),
    _Case(
        "err_near_transit",
        "params",
        dict(
            body="jupiter",
            footprint="c",
            start_time=Time("2026-03-15T23:55:00", scale="utc"),
            site=SITE,
        ),
        False,
    ),
)
_CASE_NAMES = tuple(case.name for case in _CASES)


def _plain(value: Any) -> Any:
    """Convert a recorded value to JSON types, refusing a type it does not know."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    raise TypeError(f"the golden record has no conversion for {type(value).__name__}")


def _summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "first": float(array[0]),
        "last": float(array[-1]),
        "min": float(array.min()),
        "max": float(array.max()),
        "sum": float(array.sum()),
        "sumsq": float(np.square(array).sum()),
    }


def _sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256(array.tobytes()).hexdigest()
    return f"{array.dtype.str}{list(array.shape)}:{digest}"


def _block_record(block: Any) -> tuple[dict[str, Any], dict[str, str]]:
    traj = block.trajectory
    meta = traj.metadata
    xi, eta = source_ces_focal_plane_track(block, site=SITE)
    flags, counts = np.unique(traj.scan_flag, return_counts=True)
    record = {
        "computed": _plain(block.computed_params),
        "duration": _plain(block.duration),
        "config": _plain(dataclasses.asdict(block.config)),
        "metadata": _plain(
            {
                "pattern_type": meta.pattern_type,
                "pattern_params": meta.pattern_params,
                "center_ra": meta.center_ra,
                "center_dec": meta.center_dec,
                "target_name": meta.target_name,
            }
        ),
        "summary": block.summary,
        "start_time": traj.start_time.isot,
        "scan_flag": {str(int(flag)): int(count) for flag, count in zip(flags, counts)},
        "arrays": {name: _summary(getattr(traj, name)) for name in _FLOAT_ARRAYS},
        "track": {"xi": _summary(xi), "eta": _summary(eta)},
    }
    hashes = {name: _sha256(getattr(traj, name)) for name in (*_FLOAT_ARRAYS, "scan_flag")}
    hashes["track_xi"] = _sha256(xi)
    hashes["track_eta"] = _sha256(eta)
    return record, hashes


def _call(entry: Any, kwargs: dict[str, Any]) -> Any:
    return entry(**kwargs)


_CALL_LINE = _call.__code__.co_firstlineno + 1


def _attribution(filename: str, lineno: int) -> str:
    """Name where a warning was attributed: the entry call of this module, or the library."""
    at_call = Path(filename).resolve() == _THIS_FILE and lineno == _CALL_LINE
    return "caller" if at_call else "library"


def _run_case(case: _Case) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one case; return its record and the raw-array hashes of its blocks."""
    entry = {
        "params": compute_source_ces_params,
        "plan": plan_source_ces,
        "passes": plan_source_ces_passes,
    }[case.entry]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = _call(entry, case.kwargs)
        except Exception as exc:  # noqa: BLE001 - every refusal is recorded and compared
            result = exc
    record: dict[str, Any] = {
        "warnings": [
            {
                "category": item.category.__name__,
                "message": str(item.message),
                "attributed_to": _attribution(item.filename, item.lineno),
            }
            for item in caught
        ]
    }
    hashes: dict[str, Any] = {}
    if isinstance(result, Exception):
        record["error"] = {"type": type(result).__name__, "message": str(result)}
    elif case.entry == "params":
        record["computed"] = _plain(result)
    elif case.entry == "plan":
        record["block"], hashes["block"] = _block_record(result)
    else:
        pairs = [_block_record(block) for block in result]
        record["passes"] = [pair[0] for pair in pairs]
        hashes["passes"] = [pair[1] for pair in pairs]
    return record, hashes


def _compare_values(got: Any, want: Any, path: str, problems: list[str]) -> None:
    """Compare exactly, except a float, which compares at a relative 1e-12."""
    if isinstance(want, dict):
        if not isinstance(got, dict) or got.keys() != want.keys():
            problems.append(f"{path}: {got!r} has not the recorded keys {sorted(want)}")
            return
        for key in want:
            _compare_values(got[key], want[key], f"{path}.{key}", problems)
    elif isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            problems.append(f"{path}: {got!r} != recorded {want!r}")
            return
        for index, (got_item, want_item) in enumerate(zip(got, want)):
            _compare_values(got_item, want_item, f"{path}[{index}]", problems)
    elif type(want) is float:
        if type(got) is not float or got != pytest.approx(want, rel=_SCALAR_REL, abs=0.0):
            problems.append(f"{path}: {got!r} != recorded {want!r} (rel {_SCALAR_REL})")
    elif type(got) is not type(want) or got != want:
        problems.append(f"{path}: {got!r} != recorded {want!r}")


def _compare_summary(got: Any, want: dict[str, Any], path: str, problems: list[str]) -> None:
    """Compare an array summary: the length exactly, the rest against its scale."""
    if not isinstance(got, dict) or got.keys() != want.keys():
        problems.append(f"{path}: summary keys differ from the record")
        return
    if type(got["n"]) is not int or got["n"] != want["n"]:
        problems.append(f"{path}.n: {got['n']!r} != recorded {want['n']!r}")
        return
    n = want["n"]
    a = max(abs(want["min"]), abs(want["max"]))
    scales = {"first": a, "last": a, "min": a, "max": a, "sum": n * a, "sumsq": n * a * a}
    for key, scale in scales.items():
        tolerance = pytest.approx(want[key], rel=_SUMMARY_REL, abs=_SUMMARY_ABS * scale)
        if type(got[key]) is not float or got[key] != tolerance:
            problems.append(f"{path}.{key}: {got[key]!r} != recorded {tolerance}")


def _compare_block(got: Any, want: dict[str, Any], path: str, problems: list[str]) -> None:
    if not isinstance(got, dict) or got.keys() != want.keys():
        problems.append(f"{path}: block keys differ from the record")
        return
    for key in want:
        if key in ("arrays", "track"):
            if got[key].keys() != want[key].keys():
                problems.append(f"{path}.{key}: array names differ from the record")
                continue
            for name in want[key]:
                _compare_summary(got[key][name], want[key][name], f"{path}.{key}.{name}", problems)
        else:
            _compare_values(got[key], want[key], f"{path}.{key}", problems)


def _compare_case(got: dict[str, Any], want: dict[str, Any], name: str) -> list[str]:
    problems: list[str] = []
    if got.keys() != want.keys():
        return [f"{name}: outcome keys {sorted(got)} != recorded {sorted(want)}"]
    for key in want:
        if key == "block":
            _compare_block(got[key], want[key], f"{name}.block", problems)
        elif key == "passes":
            if len(got[key]) != len(want[key]):
                problems.append(f"{name}: {len(got[key])} passes != recorded {len(want[key])}")
                continue
            for index, (got_block, want_block) in enumerate(zip(got[key], want[key])):
                _compare_block(got_block, want_block, f"{name}.passes[{index}]", problems)
        else:
            _compare_values(got[key], want[key], f"{name}.{key}", problems)
    return problems


@functools.lru_cache(maxsize=1)
def _recorded_cases() -> dict[str, Any]:
    with _FIXTURE.open(encoding="utf-8") as handle:
        return json.load(handle)["cases"]


def _dump(payload: dict[str, Any], path: Path) -> None:
    """Write sorted, indented JSON with LF line endings and a final newline."""
    text = json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _header() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    table = iers.earth_orientation_table.get()
    measured = np.asarray(table["UT1Flag"]) != "P"
    return {
        "commit": commit,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "astropy": astropy.__version__,
        "pyerfa": erfa.__version__,
        "platform": platform.platform(),
        "iers_last_measured_mjd": float(np.asarray(table["MJD"])[measured].max()),
        "cases": list(_CASE_NAMES),
    }


class _GoldenRun:
    """The records of one module run, written out when every case is recorded."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.hashes: dict[str, dict[str, Any]] = {}

    def finish(self) -> None:
        if not _SWITCHED:
            return
        missing = [name for name in _CASE_NAMES if name not in self.records]
        if missing:
            pytest.fail(
                f"golden cases not recorded, nothing written: {', '.join(missing)}",
                pytrace=False,
            )
        cases = {name: self.records[name] for name in _CASE_NAMES}
        if _EXACT_PATH is not None:
            exact = {
                name: {"record": cases[name], "sha256": self.hashes[name]} for name in _CASE_NAMES
            }
            _dump({"cases": exact}, Path(_EXACT_PATH))
        if _RECUT:
            _dump({"header": _header(), "cases": cases}, _FIXTURE)


@pytest.fixture(scope="module")
def golden_run():
    if _EXACT_PATH is not None:
        exact = Path(_EXACT_PATH).resolve()
        if exact == _REPO_ROOT or _REPO_ROOT in exact.parents:
            pytest.fail(f"FYST_GOLDEN_EXACT must lie outside the repository: {exact}")
    run = _GoldenRun()
    yield run
    run.finish()


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            case, id=case.name, marks=[pytest.mark.slow] if case.slow and not _SWITCHED else []
        )
        for case in _CASES
    ],
)
def test_source_ces_golden(case, golden_run, request):
    if _SWITCHED and not request.config.getoption("--run-slow"):
        pytest.fail("a re-cut or an exact run records every case and needs --run-slow")
    record, hashes = _run_case(case)
    golden_run.records[case.name] = record
    golden_run.hashes[case.name] = hashes
    if _RECUT:
        return
    recorded = _recorded_cases()
    assert sorted(recorded) == sorted(_CASE_NAMES), "the fixture's cases are not the module's"
    problems = _compare_case(record, recorded[case.name], case.name)
    assert not problems, "\n".join(problems)
