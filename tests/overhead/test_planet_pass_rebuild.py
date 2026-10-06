"""Every planet-calibration pass rebuilds as the pass its planner planned: the slow sweep.

Both planners run through their public entry points while every pass
``plan_source_ces_passes`` returns is captured with the anchor it was given.
Each pass block must record that anchor as its ``search_start``, and
``schedule_to_trajectories(..., science_only=False)`` must return the captured
pass, sample for sample, from the timeline in memory and from its ECSV round
trip. The cases cover every module, one and seven passes, footprint margins of
0, 0.3 and 0.6 deg, the table's throw, per-visit overrides, a dwell that cuts
more than 300 s from its crossing, and the offline scheduler with each
``planet_cal_footprint``, and both planners with their times given in TT, TAI,
TDB, UT1 and as a UTC ``Time`` carrying a location. Most cases hold passes
that the same blocks without ``search_start``, re-solved inside the pass
widened by 300 s, cannot rebuild, and they check that they do, so none of them
passes vacuously.
"""

import dataclasses
import warnings

import numpy as np
import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories import get_fyst_site
from fyst_trajectories.overhead import (
    CalibrationNightPolicy,
    CalibrationPolicy,
    CalibrationState,
    ElevationBin,
    ObservingTimeline,
    ScanOverrides,
    ScanParameterTable,
    ScriptedSelection,
    plan_calibration_night,
    read_timeline,
    schedule_to_trajectories,
    write_timeline,
)
from fyst_trajectories.overhead.calibration_night import step
from fyst_trajectories.overhead.scheduler import (
    CalibrationPhase,
    SchedulerContext,
    SchedulerState,
    phases,
)

pytestmark = pytest.mark.slow

_ARRAYS = ("times", "az", "el", "az_vel", "el_vel", "scan_flag")


@pytest.fixture
def captured(monkeypatch):
    """Every ``(anchor, pass)`` either planner plans, in planning order."""
    planned = []
    for module in (step, phases):

        def recording(*args, _real=module.plan_source_ces_passes, **kwargs):
            passes = _real(*args, **kwargs)
            planned.extend((kwargs["start_time"], p) for p in passes)
            return passes

        monkeypatch.setattr(module, "plan_source_ces_passes", recording)
    return planned


def _planned_pass(block, captured):
    """Return the captured anchor and pass a block was written from (the last solve wins)."""
    t0 = Time(block.metadata["t0_scan"], scale="utc")
    el_bore = block.metadata["scan_params"]["el_bore"]
    matches = [
        (anchor, p)
        for anchor, p in captured
        if abs((p.trajectory.start_time - t0).to_value("s")) < 2e-3
        and p.computed_params["el_bore"] == el_bore
    ]
    assert matches, f"no planned pass for the block at {block.metadata['t0_scan']}"
    return matches[-1]


def _rebuild(timeline):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pairs = schedule_to_trajectories(timeline, science_only=False)
    return [(b, sb) for b, sb in pairs if b.scan_type == "planet_cal"]


def _assert_every_pass_rebuilds_as_planned(timeline, captured, tmp_path, old_rule_misses):
    passes = [b for b in timeline.blocks if b.scan_type == "planet_cal"]
    assert passes, "the case planned no pass"
    path = tmp_path / "timeline.ecsv"
    write_timeline(timeline, path)
    for source in (timeline, read_timeline(path)):
        rebuilt = _rebuild(source)
        assert len(rebuilt) == len(passes)
        for block, scan_block in rebuilt:
            anchor, planned = _planned_pass(block, captured)
            assert block.metadata["search_start"] == [anchor.utc.jd1, anchor.utc.jd2]
            for name in _ARRAYS:
                assert np.array_equal(
                    getattr(scan_block.trajectory, name), getattr(planned.trajectory, name)
                ), name
            got, want = scan_block.trajectory.start_time, planned.trajectory.start_time
            assert (got.jd1, got.jd2) == (want.jd1, want.jd2)
            assert scan_block.duration == planned.duration
            assert scan_block.computed_params == planned.computed_params
    keyless = dataclasses.replace(
        timeline,
        blocks=[
            dataclasses.replace(
                b, metadata={k: v for k, v in b.metadata.items() if k != "search_start"}
            )
            for b in timeline.blocks
        ],
    )
    missed = len(passes) - len(_rebuild(keyless))
    assert (missed > 0) == old_rule_misses, f"{missed} of {len(passes)} missed without the key"
    return passes


_PAGE_NIGHT = (["saturn", "uranus"], "2026-09-11T06:30:00", "2026-09-11T07:30:00")
_DWELL_300 = {"default": ScanParameterTable((ElevationBin(30.0, 80.0, 3.0, 300.0),))}

# (targets, start, end, planner keywords, the keys some pass dict must carry,
# whether the blocks without ``search_start`` miss a pass).
_NIGHTS = {
    "c": (*_PAGE_NIGHT, {}, (), False),
    "i1": (*_PAGE_NIGHT, {"policy": CalibrationNightPolicy(footprint="i1")}, (), True),
    "i3": (*_PAGE_NIGHT, {"policy": CalibrationNightPolicy(footprint="i3")}, (), True),
    "i6": (*_PAGE_NIGHT, {"policy": CalibrationNightPolicy(footprint="i6")}, (), True),
    # A later pass of the first visit crosses its boresight elevation more
    # than 300 s before it starts.
    "i1, 7 passes, margin 0.3": (
        ["saturn"],
        "2026-09-10T08:00:00",
        "2026-09-10T10:00:00",
        {"policy": CalibrationNightPolicy(footprint="i1", n_passes=7, footprint_margin=0.3)},
        ("footprint_margin",),
        True,
    ),
    "i6, 7 passes, margin 0.6": (
        ["saturn"],
        "2026-09-10T07:40:00",
        "2026-09-10T10:00:00",
        {"policy": CalibrationNightPolicy(footprint="i6", n_passes=7, footprint_margin=0.6)},
        ("footprint_margin",),
        True,
    ),
    "i2, table throw, margin 0.3": (
        *_PAGE_NIGHT,
        {
            "policy": CalibrationNightPolicy(
                footprint="i2", use_table_throw=True, footprint_margin=0.3
            )
        },
        ("az_throw", "footprint_margin"),
        False,
    ),
    # Three back-to-back Saturn visits, one per override.
    "i4, per-visit overrides": (
        ["saturn"],
        "2026-09-11T06:30:00",
        "2026-09-11T08:00:00",
        {
            "policy": CalibrationNightPolicy(footprint="i4"),
            "selection": ScriptedSelection(
                [
                    ("saturn", ScanOverrides(az_speed=1.0)),
                    ("saturn", ScanOverrides(az_throw=3.5)),
                    ("saturn", ScanOverrides(az_accel=0.8, dwell=240.0)),
                ]
            ),
        },
        ("az_throw", "dwell"),
        False,
    ),
    # The 300 s dwell cuts 399, 293 and 230 s from Jupiter's three crossings.
    "c, a 300 s dwell": (
        ["jupiter"],
        "2026-04-15T22:40:00",
        "2026-04-15T23:40:00",
        {"policy": CalibrationNightPolicy(use_table_dwell=True), "tables": _DWELL_300},
        ("dwell",),
        True,
    ),
}


@pytest.mark.parametrize("case", list(_NIGHTS))
def test_every_calibration_night_pass_rebuilds_as_planned(case, captured, tmp_path):
    targets, start, end, kwargs, keys, old_rule_misses = _NIGHTS[case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        timeline = plan_calibration_night(targets, get_fyst_site(), start, end, **kwargs)
    passes = _assert_every_pass_rebuilds_as_planned(timeline, captured, tmp_path, old_rule_misses)
    for key in keys:
        assert any(key in b.metadata["scan_params"] for b in passes), key


# One offline-scheduler planet calibration per footprint, at an anchor where it
# plans passes: (body, anchor, passes, whether the blocks without
# ``search_start`` miss a pass). On i1, i3, i4 and i6 the source crosses the
# boresight elevation of some pass more than 300 s after the pass ends.
_SCHEDULER = {
    "c": ("saturn", "2026-08-10T05:05:00", 1, False),
    "i1": ("jupiter", "2026-02-10T00:40:00", 3, True),
    "i2": ("saturn", "2026-06-10T15:10:00", 3, False),
    "i3": ("saturn", "2026-06-10T13:20:00", 3, True),
    "i4": ("uranus", "2026-10-10T09:20:00", 3, True),
    "i5": ("saturn", "2026-10-10T05:20:00", 3, False),
    "i6": ("neptune", "2026-06-10T10:00:00", 3, True),
}


def _scheduler_calibration(footprint, anchor):
    """Plan the ``_SCHEDULER`` calibration on ``footprint`` from ``anchor`` as a timeline."""
    body, _, n_passes, _ = _SCHEDULER[footprint]
    site = get_fyst_site()
    policy = CalibrationPolicy(
        retune_cadence=1.0e9,
        pointing_cadence=1.0e9,
        focus_cadence=1.0e9,
        skydip_cadence=1.0e9,
        planet_cal_cadence=1.0e9,
        planet_cal_scan=True,
        planet_cal_passes=n_passes,
        planet_targets=(body,),
        planet_min_elevation=15.0,
        planet_cal_footprint=footprint,
    )
    ctx = SchedulerContext.build(
        patches=[],
        site=site,
        start_time=anchor,
        end_time=anchor + TimeDelta(6 * 3600.0, format="sec"),
        calibration_policy=policy,
    )
    state = SchedulerState(
        current_time=anchor,
        current_az=180.0,
        current_el=50.0,
        cal_state=CalibrationState(
            last_retune=anchor,
            last_pointing_cal=anchor,
            last_focus=anchor,
            last_skydip=anchor,
            last_planet_cal=None,
        ),
        scan_counter=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        blocks = CalibrationPhase().run(state, ctx).blocks
    return ObservingTimeline(
        blocks=list(blocks),
        site=site,
        start_time=ctx.start_time,
        end_time=blocks[-1].t_stop,
        overhead_model=ctx.overhead_model,
        calibration_policy=ctx.calibration_policy,
    )


@pytest.mark.parametrize("footprint", list(_SCHEDULER))
def test_every_scheduler_pass_rebuilds_as_planned(footprint, captured, tmp_path):
    _, iso, n_passes, old_rule_misses = _SCHEDULER[footprint]
    timeline = _scheduler_calibration(footprint, Time(iso, scale="utc"))
    passes = _assert_every_pass_rebuilds_as_planned(timeline, captured, tmp_path, old_rule_misses)
    assert len(passes) == n_passes


_SCALES = ["tt", "tai", "tdb", "ut1", "utc+location"]


def _given_in(iso, scale):
    """Return the UTC instant ``iso`` as a ``Time`` in ``scale``.

    ``"utc+location"`` is a UTC ``Time`` carrying the site's location, which
    enters its conversion to TDB.
    """
    t = Time(iso, scale="utc")
    if scale == "utc+location":
        return Time(t, location=get_fyst_site().location)
    return getattr(t, scale)


@pytest.mark.parametrize("scale", _SCALES)
def test_a_night_given_in_another_scale_rebuilds_as_planned(scale, captured, tmp_path):
    """The ``i1`` page night with its window ends given in ``scale`` rebuilds bit for bit.

    The planner holds every time in UTC, so each pass block records the UTC
    instant its search began at and its UTC start as ``t0_scan``, which the
    capture matches on.
    """
    targets, start, end = _PAGE_NIGHT
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        timeline = plan_calibration_night(
            targets,
            get_fyst_site(),
            _given_in(start, scale),
            _given_in(end, scale),
            policy=CalibrationNightPolicy(footprint="i1"),
        )
    assert timeline.start_time.scale == "utc"
    _assert_every_pass_rebuilds_as_planned(timeline, captured, tmp_path, True)


@pytest.mark.parametrize("scale", _SCALES)
def test_a_scheduler_calibration_from_another_scale_rebuilds_as_planned(scale, captured, tmp_path):
    """The ``i6`` calibration planned from a state and window in ``scale`` rebuilds bit for bit."""
    _, iso, n_passes, old_rule_misses = _SCHEDULER["i6"]
    timeline = _scheduler_calibration("i6", _given_in(iso, scale))
    assert {b.t_start.scale for b in timeline.blocks} == {"utc"}
    passes = _assert_every_pass_rebuilds_as_planned(timeline, captured, tmp_path, old_rule_misses)
    assert len(passes) == n_passes
