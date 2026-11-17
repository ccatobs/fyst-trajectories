"""A science block planned for a detector off the boresight rebuilds with that detector's offset.

A block written by :meth:`TimelineBlock.science` for a patch that carries a
``detector_offset`` records the offset by value, and a constant-elevation
block records the detector's elevation beside it as ``detector_elevation``.
:func:`schedule_to_trajectories` passes both back to the planner, so each
block below rebuilds to the trajectory the planner returns when called
directly with the same inputs, sample for sample, from memory and after an
ECSV round trip, whatever the bay table holds when it is rebuilt.
"""

import dataclasses
import logging
import math

import numpy as np
import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories import get_fyst_site, get_primecam_offset, primecam
from fyst_trajectories.overhead import (
    CalibrationPolicy,
    ObservingPatch,
    ObservingTimeline,
    OverheadModel,
    ScanParamsSchemaError,
    TimelineBlock,
    read_timeline,
    schedule_to_trajectories,
    write_timeline,
)
from fyst_trajectories.overhead import simulation as sim
from fyst_trajectories.planning import (
    FieldRegion,
    plan_constant_el_scan,
    plan_daisy_scan,
    plan_pong_scan,
)

SITE = get_fyst_site()
IM3 = get_primecam_offset("IM3")

#: The detector's elevation for the constant-elevation block.
DETECTOR_EL = 50.0

#: The 40 x 10 deg field whose rising pass at el 50 opens at 08:02:25 UTC on
#: 2026-06-15, and the anchor before it the block records.
DEEP56 = FieldRegion(ra_center=24.0, dec_center=-32.0, width=40.0, height=10.0)
CE_ANCHOR = "2026-06-15T07:00:00.000"

#: A 4 x 4 deg field near el 43 at 02:30 UTC on 2026-06-15, for the pong and the daisy.
SMALL = FieldRegion(ra_center=180.0, dec_center=-30.0, width=4.0, height=4.0)
PATTERN_START = Time("2026-06-15T02:30:00.000", scale="utc")


def _patch(scan_type, field, **kwargs):
    return ObservingPatch(
        name=f"{scan_type}_IM3",
        ra_center=field.ra_center,
        dec_center=field.dec_center,
        width=field.width,
        height=field.height,
        scan_type=scan_type,
        velocity=1.0 if scan_type == "constant_el" else 0.5,
        detector_offset=IM3,
        **kwargs,
    )


def _ce_planned():
    """Return the pass the planner gives for IM3 at el 50 from the anchor."""
    return plan_constant_el_scan(
        field=DEEP56,
        elevation=DETECTOR_EL,
        velocity=1.0,
        site=SITE,
        start_time=Time(CE_ANCHOR, scale="utc"),
        rising=True,
        detector_offset=IM3,
    )


def _ce_block():
    """Return a 600 s block 1000 s into the pass, its edges about half a sample from any sample."""
    traj = _ce_planned().trajectory
    t_start = Time((traj.start_time + TimeDelta(1000.05, format="sec")).isot, scale="utc")
    return TimelineBlock.science(
        _patch("constant_el", DEEP56, elevation=DETECTOR_EL),
        t_start,
        600.0,
        float(traj.az.min()),
        float(traj.az.max()),
        float(traj.el[0]),
        0,
        rising=True,
        t0_scan=CE_ANCHOR,
    )


def _pong_block():
    """Return a block holding the head of one pong period, ending half a sample after a sample."""
    return TimelineBlock.science(
        _patch("pong", SMALL, scan_params={"n_cycles": 1}),
        PATTERN_START,
        650.05,
        170.0,
        190.0,
        40.0,
        0,
    )


def _daisy_block():
    return TimelineBlock.science(
        _patch("daisy", SMALL), PATTERN_START, 300.0, 170.0, 190.0, 40.0, 0
    )


def _planned_for(block):
    """Call the planner directly with the inputs the block was written from."""
    if block.scan_type == "constant_el":
        return _ce_planned()
    if block.scan_type == "pong":
        return plan_pong_scan(
            field=SMALL,
            velocity=0.5,
            site=SITE,
            start_time=PATTERN_START,
            spacing=0.1,
            num_terms=4,
            timestep=0.1,
            angle=0.0,
            n_cycles=1,
            detector_offset=IM3,
        )
    return plan_daisy_scan(
        ra=SMALL.ra_center,
        dec=SMALL.dec_center,
        velocity=0.5,
        site=SITE,
        start_time=block.t_start,
        duration=block.duration,
        radius=1.0,
        turn_radius=0.5,
        avoidance_radius=0.1,
        start_acceleration=0.5,
        timestep=0.1,
        detector_offset=IM3,
    )


_BLOCKS = {"constant_el": _ce_block, "pong": _pong_block, "daisy": _daisy_block}


def _through_file(block, tmp_path):
    timeline = ObservingTimeline(
        blocks=[block],
        site=SITE,
        start_time=block.t_start,
        end_time=block.t_stop,
        overhead_model=OverheadModel(),
        calibration_policy=CalibrationPolicy(),
    )
    path = tmp_path / "offset_block.ecsv"
    write_timeline(timeline, path)
    (loaded,) = read_timeline(path).blocks
    return loaded


def _assert_rebuilds_to_the_planned_samples(block, planned):
    """Assert the rebuild holds the planned samples inside the block's ``[t_start, t_stop)``."""
    traj = planned.trajectory
    rel_start = (block.t_start - traj.start_time).sec
    rel_stop = (block.t_stop - traj.start_time).sec
    (idx,) = np.nonzero((traj.times >= rel_start) & (traj.times < rel_stop))
    assert idx.size > 100

    rebuilt = sim._generate_trajectory_for_block(block, SITE).trajectory
    for name in ("az", "el", "az_vel", "el_vel", "scan_flag"):
        assert np.array_equal(getattr(rebuilt, name), getattr(traj, name)[idx]), name
    assert np.array_equal(rebuilt.times, traj.times[idx] - traj.times[idx[0]])
    first = traj.start_time + TimeDelta(float(traj.times[idx[0]]), format="sec")
    assert abs((rebuilt.start_time - first).sec) < 1e-9


@pytest.mark.parametrize("scan_type", sorted(_BLOCKS))
@pytest.mark.parametrize("source", ["memory", "file"])
def test_a_block_rebuilds_to_the_planned_samples(scan_type, source, tmp_path):
    block = _BLOCKS[scan_type]()
    if source == "file":
        block = _through_file(block, tmp_path)
    assert block.metadata["detector_offset"] == dataclasses.asdict(IM3)
    _assert_rebuilds_to_the_planned_samples(block, _planned_for(block))


def test_the_constant_elevation_block_holds_the_boresight_and_the_detector_elevations():
    """The premise of the case: the block's elevation is the boresight's, not the detector's."""
    block = _ce_block()
    assert block.metadata["detector_elevation"] == DETECTOR_EL
    assert abs(block.elevation - DETECTOR_EL) > 1.0


@pytest.mark.parametrize("scan_type", sorted(_BLOCKS))
def test_the_recorded_offset_wins_over_a_changed_bay_table(scan_type, monkeypatch):
    block = _BLOCKS[scan_type]()
    planned = _planned_for(block)
    moved = dataclasses.replace(IM3, dx=IM3.dx + 10.0)
    monkeypatch.setattr(
        primecam, "_MODULES_BY_LOWER_NAME", {**primecam._MODULES_BY_LOWER_NAME, "im3": moved}
    )
    assert get_primecam_offset("IM3") == moved
    _assert_rebuilds_to_the_planned_samples(block, planned)


def _with_meta(block, drop=(), **changes):
    meta = {k: v for k, v in block.metadata.items() if k not in drop}
    return dataclasses.replace(block, metadata={**meta, **changes})


_MALFORMED_OFFSETS = [
    {"dx": 1.0},
    "IM3",
    [1.0, 0.0],
    {**dataclasses.asdict(IM3), "name": 3},
    {**dataclasses.asdict(IM3), "dx": math.nan},
    {**dataclasses.asdict(IM3), "dx": "53.4"},
    {**dataclasses.asdict(IM3), "plate_scale": 13.89},
]


class TestMalformedRecords:
    """Each malformed record is a ``ScanParamsSchemaError``; the timeline rebuild skips it."""

    def test_an_offset_without_its_detector_elevation(self):
        block = _with_meta(_ce_block(), drop=("detector_elevation",))
        with pytest.raises(ScanParamsSchemaError, match="must record detector_elevation"):
            sim._generate_trajectory_for_block(block, SITE)

    @pytest.mark.parametrize("offset", ["absent", None])
    def test_a_detector_elevation_without_an_offset(self, offset):
        block = _ce_block()
        if offset == "absent":
            block = _with_meta(block, drop=("detector_offset",))
        else:
            block = _with_meta(block, detector_offset=offset)
        with pytest.raises(ScanParamsSchemaError, match="only beside detector_offset"):
            sim._generate_trajectory_for_block(block, SITE)

    @pytest.mark.parametrize("value", [math.nan, math.inf, "50", True, None, [50.0]])
    def test_a_detector_elevation_that_is_not_a_finite_number(self, value):
        block = _with_meta(_ce_block(), detector_elevation=value)
        with pytest.raises(ScanParamsSchemaError, match="detector_elevation must be a finite"):
            sim._generate_trajectory_for_block(block, SITE)

    @pytest.mark.parametrize("scan_type", ["pong", "daisy"])
    def test_a_detector_elevation_on_a_pong_or_daisy(self, scan_type):
        block = _with_meta(_BLOCKS[scan_type](), detector_elevation=DETECTOR_EL)
        with pytest.raises(ScanParamsSchemaError, match="only on a constant-elevation block"):
            sim._generate_trajectory_for_block(block, SITE)

    @pytest.mark.parametrize("scan_type", sorted(_BLOCKS))
    @pytest.mark.parametrize("record", _MALFORMED_OFFSETS)
    def test_a_malformed_offset(self, scan_type, record):
        block = _with_meta(_BLOCKS[scan_type](), detector_offset=record)
        with pytest.raises(ScanParamsSchemaError, match="detector_offset must be None or"):
            sim._generate_trajectory_for_block(block, SITE)

    def test_the_timeline_rebuild_logs_and_skips_them(self, caplog):
        good = _ce_block()
        bad = [
            _with_meta(good, drop=("detector_elevation",)),
            _with_meta(good, detector_offset={"dx": 1.0}),
            _with_meta(_pong_block(), detector_elevation=DETECTOR_EL),
        ]
        blocks = sorted([good, *bad], key=lambda b: b.t_start.unix)
        timeline = ObservingTimeline(
            blocks=blocks,
            site=SITE,
            start_time=blocks[0].t_start,
            end_time=max(b.t_stop for b in blocks),
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )
        with caplog.at_level(logging.WARNING, logger="fyst_trajectories.overhead.simulation"):
            pairs = schedule_to_trajectories(timeline)
        assert [block for block, _ in pairs] == [good]
        assert len(caplog.records) == len(bad)
        assert all("detector_" in record.getMessage() for record in caplog.records)
