"""Constant-elevation window scans written to a timeline and rebuilt from it.

A scan planned with the window form of ``plan_constant_el_scan`` (given
``az_window`` and ``duration``) is written to a science block with
``TimelineBlock.from_window_scan``, which records the planner's inputs under
``metadata["window_scan"]``. ``schedule_to_trajectories`` rebuilds such a
block with the same form of the planner, from memory and after an ECSV round
trip. These tests pin:

- the rebuilt trajectory equals the planned one sample for sample, for the
  centre module with a field, an off-axis module, a scan without a field, a
  padded window and a window the planner moved down or up a whole turn;
- the block's azimuths and elevation are the boresight's, and the module
  planned with an offset lands on the recorded window;
- a block without a field carries no field keys, before and after the file;
- malformed records are refused by name, and a block without a record still
  rebuilds through ``plan_constant_el_scan``;
- a whole block keeps the scan's last sample, a shorter one its head;
- the Sun advisory describes the samples the block returns, and a field's
  drift advisory is raised again by the rebuild;
- the writer rebuilds the block once and refuses a scan it does not
  reproduce: a slice, a modified trajectory, a wrong or missing offset, or
  another planner's scan; a whole field-form scan is written, and rebuilt,
  as the window scan it swept.
"""

import dataclasses
import json
import logging
import math
import re
import warnings

import numpy as np
import pytest
from astropy.table import Table
from astropy.time import Time, TimeDelta

from fyst_trajectories import (
    SCAN_FLAG_SCIENCE,
    Coordinates,
    InstrumentOffset,
    boresight_to_detector,
    compute_focal_plane_rotation,
    get_fyst_site,
    get_primecam_offset,
    inject_retune,
)
from fyst_trajectories.exceptions import PointingWarning
from fyst_trajectories.overhead import (
    BlockNotReconstructableError,
    CalibrationPolicy,
    ObservingPatch,
    ObservingTimeline,
    OverheadModel,
    ScanParamsSchemaError,
    TimelineBlock,
    WindowScanRecord,
    read_timeline,
    schedule_to_trajectories,
    write_timeline,
)
from fyst_trajectories.overhead import simulation as sim
from fyst_trajectories.planning import (
    FieldRegion,
    plan_constant_el_scan,
    plan_pong_scan,
    plan_source_ces,
)

SITE = get_fyst_site()

ARRAYS = ("times", "az", "el", "az_vel", "el_vel", "scan_flag")
FIELD_KEYS = ("ra_center", "dec_center", "width", "height")

# The COSMOS row of a constant-elevation unit: az 59.7 to 67.0 deg at el 43,
# 0.33 h, on-sky 0.5 deg/s, from when the field's leading edge reaches 43 deg.
COSMOS_FIELD = FieldRegion(
    ra_center=150.125, dec_center=2.1666665, width=4.996425402671095, height=4.166667
)
COSMOS_START = Time("2026-11-15T08:01:44", scale="utc")
COSMOS_ROW = {
    "az_window": (59.7, 67.0),
    "elevation": 43.0,
    "velocity": 0.5 / math.cos(math.radians(43.0)),
    "duration": 0.33 * 3600.0,
    "start_time": COSMOS_START,
}

# name: (planner keywords, field, module offset)
CASES = {
    "centre_with_field": (COSMOS_ROW, COSMOS_FIELD, None),
    "module_im6": (COSMOS_ROW, None, "IM6"),
    "no_field": (
        {
            "az_window": (26.0, 154.0),
            "elevation": 50.0,
            "velocity": 1.0,
            "duration": 1800.0,
            "start_time": Time("2026-11-15T04:00:00", scale="utc"),
        },
        None,
        None,
    ),
    "padded": ({**COSMOS_ROW, "az_padding": 0.5}, COSMOS_FIELD, None),
    # At 3 deg/s the turnaround overshoot past 359.9 overruns the limit, so
    # the planner moves the window down a whole turn, to (-60.0, -0.1).
    "moved_down_a_turn": (
        {
            "az_window": (300.0, 359.9),
            "elevation": 40.0,
            "velocity": 3.0,
            "duration": 600.0,
            "start_time": Time("2026-03-15T03:00:00", scale="utc"),
        },
        None,
        None,
    ),
    # 2026-03-15 at dawn: the Sun climbs towards a window fixed above it and
    # enters the exclusion zone during the hour.
    "dawn": (
        {
            "az_window": (80.0, 100.0),
            "elevation": 55.0,
            "velocity": 0.5,
            "duration": 3600.0,
            "timestep": 1.0,
            "start_time": Time("2026-03-15T11:00:00", scale="utc"),
        },
        None,
        None,
    ),
    "western": (
        {
            "az_window": (200.0, 230.0),
            "elevation": 50.0,
            "velocity": 1.0,
            "duration": 600.0,
            "start_time": Time("2026-11-15T04:00:00", scale="utc"),
        },
        None,
        None,
    ),
}
EXACT_CASES = ("centre_with_field", "module_im6", "no_field", "padded", "moved_down_a_turn")
SUN_MESSAGE = "EXCLUSION ZONE: planned constant-elevation window scan"
REFUSED = "does not rebuild"

# Windows the planner moves UP a whole turn (the first because its turnarounds
# overshoot -180 deg, the second because it is written below -180 deg), so the
# placed throw differs from the written one in its last bits. The first rounds
# every array differently on the two throws, the second its number of samples.
MOVED_UP_A_TURN = {
    "last_bits": {
        "az_window": (-178.04917435311467, -40.246759103332664),
        "elevation": 50.0,
        "velocity": 2.5,
        "timestep": 0.07,
        "duration": 461.06,
        "start_time": Time("2026-03-15T03:00:00", scale="utc"),
    },
    "sample_count": {
        "az_window": (-200.0, -58.15),
        "elevation": 45.0,
        "velocity": 1.0,
        "duration": 141.85,
        "start_time": Time("2026-03-15T03:00:00", scale="utc"),
    },
}
DRIFT_MESSAGE = "starts after the field has drifted through"


def _plan(name, **overrides):
    """Plan one case; the dawn case's own Sun advisory is set aside here."""
    kwargs, field, tag = CASES[name]
    offset = None if tag is None else get_primecam_offset(tag)
    kwargs = {"detector_offset": offset, **kwargs, **overrides}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=SUN_MESSAGE, category=PointingWarning)
        scan = plan_constant_el_scan(field, site=SITE, **kwargs)
    return scan, field, kwargs["detector_offset"]


@pytest.fixture(scope="module")
def planned():
    """Each case planned once, with the block written for it: ``(scan, block, field, offset)``."""
    cache = {}

    def get(name):
        if name not in cache:
            scan, field, offset = _plan(name)
            block = TimelineBlock.from_window_scan(
                scan,
                site=SITE,
                patch_name=name,
                scan_index=0,
                field=field,
                detector_offset=offset,
            )
            cache[name] = (scan, block, field, offset)
        return cache[name]

    return get


def _timeline(*blocks):
    return ObservingTimeline(
        blocks=list(blocks),
        site=SITE,
        start_time=min(b.t_start for b in blocks),
        end_time=max(b.t_stop for b in blocks),
        overhead_model=OverheadModel(),
        calibration_policy=CalibrationPolicy(),
    )


def _through_file(block, tmp_path):
    """Write a one-block timeline and read its block back."""
    path = tmp_path / "window.ecsv"
    write_timeline(_timeline(block), path)
    (read,) = read_timeline(path).blocks
    return read


def _rebuild(block):
    """Rebuild one block through the timeline entry point."""
    pairs = schedule_to_trajectories(_timeline(block))
    assert len(pairs) == 1
    return pairs[0][1]


def _assert_identical(rebuilt, scan):
    for name in ARRAYS:
        assert np.array_equal(getattr(rebuilt.trajectory, name), getattr(scan.trajectory, name)), (
            name
        )


class TestRoundTrip:
    """The rebuilt trajectory is the planned one, from memory and from a file."""

    @pytest.mark.parametrize("source", ["memory", "file"])
    @pytest.mark.parametrize("name", EXACT_CASES)
    def test_the_rebuild_is_the_planned_scan(self, planned, tmp_path, name, source):
        scan, block, _, _ = planned(name)
        if source == "file":
            block = _through_file(block, tmp_path)
        rebuilt = _rebuild(block)
        _assert_identical(rebuilt, scan)
        assert rebuilt.duration == scan.duration
        assert rebuilt.config == scan.config
        for key in ("n_scans", "az_start", "az_stop"):
            assert rebuilt.computed_params[key] == scan.computed_params[key]
        assert rebuilt.trajectory.start_time == scan.trajectory.start_time
        built_end = scan.trajectory.start_time + TimeDelta(scan.duration, format="sec")
        assert rebuilt.computed_params["end_time_iso"] == built_end.utc.iso

    def test_a_start_with_sub_millisecond_digits(self, tmp_path):
        """Memory and a file both rebuild from the start as a file stores it, to the millisecond."""
        start = Time("2026-11-15T08:01:44.0004", scale="utc")
        scan, field, _ = _plan("centre_with_field", start_time=start)
        block = TimelineBlock.from_window_scan(
            scan, site=SITE, patch_name="P", scan_index=0, field=field
        )
        from_memory = _rebuild(block)
        _assert_identical(from_memory, scan)
        from_file = _rebuild(_through_file(block, tmp_path))
        _assert_identical(from_file, scan)
        stored = Time("2026-11-15T08:01:44.000", scale="utc")
        assert from_memory.trajectory.start_time == stored
        assert from_file.trajectory.start_time == stored

    def test_a_tt_start_is_held_as_the_utc_instant(self):
        utc = CASES["no_field"][0]["start_time"]
        start = Time(utc.tt.jd1, utc.tt.jd2, format="jd", scale="tt", location=SITE.location)
        scan, _, _ = _plan("no_field", start_time=start)
        block = TimelineBlock.from_window_scan(scan, site=SITE, patch_name="P", scan_index=0)
        assert block.t_start.scale == "utc"
        assert block.t_start.location is None
        assert abs((block.t_start - start).sec) < 1e-9
        _assert_identical(_rebuild(block), scan)


class TestBlockGeometry:
    """The block holds the boresight's azimuths and elevation; the record holds the window."""

    def test_an_off_axis_block_records_the_boresight_and_its_module_sweeps_the_window(
        self, planned
    ):
        scan, block, _, offset = planned("module_im6")
        traj = scan.trajectory
        record = block.metadata["window_scan"]
        assert block.az_start == float(traj.az.min())
        assert block.az_end == float(traj.az.max())
        assert block.elevation == float(traj.el[0])
        assert block.az_final == float(traj.az[-1])
        assert (block.az_start, block.az_end) == pytest.approx((58.79, 66.67), abs=0.01)
        assert block.elevation == pytest.approx(44.723, abs=0.001)
        assert record["az_window"] == [59.7, 67.0]
        assert record["elevation"] == 43.0
        assert record["detector_offset"] == dataclasses.asdict(offset)

        rebuilt = _rebuild(block).trajectory
        rotation = compute_focal_plane_rotation(rebuilt.el, site=SITE, offset=offset)
        module_az, module_el = boresight_to_detector(rebuilt.az, rebuilt.el, offset, rotation)
        science = rebuilt.scan_flag == SCAN_FLAG_SCIENCE
        assert float(module_az[science].min()) == pytest.approx(59.7, abs=1e-6)
        assert float(module_az[science].max()) == pytest.approx(67.0, abs=1e-6)
        np.testing.assert_allclose(module_el, 43.0, rtol=0.0, atol=1e-6)

    def test_a_block_holds_the_whole_scan_from_its_start(self, planned):
        scan, block, _, _ = planned("centre_with_field")
        assert block.t_start == scan.trajectory.start_time
        assert block.duration == pytest.approx(scan.duration, abs=1e-6)
        assert block.scan_type == "constant_el"
        assert block.metadata["scan_params"] == {}
        assert block.metadata["velocity"] == COSMOS_ROW["velocity"]
        assert block.metadata["window_scan"]["duration"] == scan.duration
        assert set(block.metadata["window_scan"]) == WindowScanRecord.__required_keys__

    @pytest.mark.parametrize(
        ("name", "rising"),
        [
            ("centre_with_field", True),
            ("module_im6", True),
            ("no_field", True),
            ("moved_down_a_turn", False),
            ("western", False),
        ],
    )
    def test_rising_is_the_side_of_the_window_midpoint(self, planned, name, rising):
        assert planned(name)[1].rising is rising

    @pytest.mark.parametrize("name", ["centre_with_field", "module_im6"])
    def test_the_metadata_is_json(self, planned, name):
        meta = planned(name)[1].metadata
        assert json.loads(json.dumps(meta)) == meta


class TestNoField:
    """A scan planned without a field is written and read without one."""

    @pytest.mark.parametrize("name", ["no_field", "module_im6"])
    def test_no_field_keys_before_or_after_the_file(self, planned, tmp_path, name):
        block = planned(name)[1]
        assert not set(FIELD_KEYS) & set(block.metadata)
        assert not set(FIELD_KEYS) & set(_through_file(block, tmp_path).metadata)

    def test_the_file_holds_zeros_in_the_field_columns(self, planned, tmp_path):
        block = planned("no_field")[1]
        path = tmp_path / "window.ecsv"
        write_timeline(_timeline(block), path)
        (row,) = Table.read(str(path), format="ascii.ecsv")
        for key in FIELD_KEYS:
            assert float(row[key]) == 0.0
        assert float(row["azmin"]) == block.az_start
        assert float(row["azmax"]) == block.az_end
        assert float(row["el"]) == block.elevation

    @pytest.mark.parametrize("name", ["no_field", "centre_with_field", "module_im6"])
    def test_the_block_read_back_is_the_block_written(self, planned, tmp_path, name):
        block = planned(name)[1]
        read = _through_file(block, tmp_path)
        for item in dataclasses.fields(TimelineBlock):
            written, got = getattr(block, item.name), getattr(read, item.name)
            if item.name == "t_stop":
                # The file keeps times to the millisecond; the stop is the start
                # plus the built duration.
                assert abs((got - written).sec) < 0.0005
            else:
                assert got == written, item.name


def _replace_meta(block, **changes):
    """Return ``block`` with metadata keys replaced (``None`` removes a key)."""
    meta = json.loads(json.dumps(block.metadata))
    for key, value in changes.items():
        if value is None:
            meta.pop(key, None)
        else:
            meta[key] = value
    return dataclasses.replace(block, metadata=meta)


def _replace_record(block, **changes):
    record = dict(block.metadata["window_scan"])
    for key, value in changes.items():
        if value is None:
            record.pop(key)
        else:
            record[key] = value
    return _replace_meta(block, window_scan=record)


class TestMalformedRecords:
    """A malformed window block is refused by name and skipped by the timeline rebuild."""

    @pytest.mark.parametrize("key", sorted(WindowScanRecord.__required_keys__))
    def test_a_missing_record_key(self, planned, key):
        block = _replace_record(planned("centre_with_field")[1], **{key: None})
        with pytest.raises(ScanParamsSchemaError, match=rf"missing keys \['{key}'\]"):
            sim._generate_trajectory_for_block(block, SITE)

    def test_an_unknown_record_key(self, planned):
        block = _replace_record(planned("centre_with_field")[1], az_padding=0.5)
        with pytest.raises(ScanParamsSchemaError, match=r"unknown keys \['az_padding'\]"):
            sim._generate_trajectory_for_block(block, SITE)

    def test_a_record_that_is_not_a_mapping(self, planned):
        block = _replace_meta(planned("centre_with_field")[1], window_scan=[59.7, 67.0])
        with pytest.raises(ScanParamsSchemaError, match="window_scan must be a mapping"):
            sim._generate_trajectory_for_block(block, SITE)

    def test_a_missing_velocity(self, planned):
        block = _replace_meta(planned("centre_with_field")[1], velocity=None)
        with pytest.raises(ScanParamsSchemaError, match="'velocity'"):
            sim._generate_trajectory_for_block(block, SITE)

    def test_scan_params_that_are_not_empty(self, planned):
        block = _replace_meta(planned("centre_with_field")[1], scan_params={"az_accel": 1.0})
        with pytest.raises(ScanParamsSchemaError, match=r"scan_params must be empty.*az_accel"):
            sim._generate_trajectory_for_block(block, SITE)

    def test_a_partial_field(self, planned):
        block = _replace_meta(planned("centre_with_field")[1], height=None)
        with pytest.raises(ScanParamsSchemaError, match=r"missing \['height'\]"):
            sim._generate_trajectory_for_block(block, SITE)

    @pytest.mark.parametrize(
        "offset",
        [
            {"dx": 1.0},
            "IM6",
            {"dx": 1.0, "dy": 0.0},
            {"dx": 1.0, "dy": 0.0, "name": 6, "instrument_rotation": 0.0},
        ],
    )
    def test_a_malformed_detector_offset(self, planned, offset):
        block = _replace_record(planned("centre_with_field")[1], detector_offset=offset)
        with pytest.raises(ScanParamsSchemaError, match="detector_offset"):
            sim._generate_trajectory_for_block(block, SITE)

    def test_a_scan_type_other_than_constant_el(self, planned):
        block = dataclasses.replace(planned("centre_with_field")[1], scan_type="pong")
        with pytest.raises(ScanParamsSchemaError, match="scan_type 'constant_el'"):
            sim._generate_trajectory_for_block(block, SITE)

    def test_the_timeline_rebuild_logs_and_skips_it(self, planned, caplog):
        block = _replace_record(planned("centre_with_field")[1], az_padding=0.5)
        with caplog.at_level(logging.WARNING, logger="fyst_trajectories.overhead.simulation"):
            assert schedule_to_trajectories(_timeline(block)) == []
        assert any(
            record.name == "fyst_trajectories.overhead.simulation"
            and record.levelno == logging.WARNING
            and "az_padding" in record.getMessage()
            for record in caplog.records
        )


def test_a_block_without_a_record_rebuilds_through_plan_constant_el_scan(monkeypatch, tmp_path):
    """A file block with ``az_min``/``az_max`` in scan_params and no record keeps the field form.

    A constant-elevation block whose ``scan_params`` carry the pair and
    which has no ``window_scan`` record, as a file can hold one, is rebuilt
    by the field form, read back from the file: the pair (62.0, 64.5) is not
    the range the rebuild sweeps; ``plan_constant_el_scan`` sweeps its own,
    (59.84, 66.78), where the field's border meets el 43, so the trajectory
    runs more than 2 deg past each end of the pair before any turnaround.
    """
    planner = sim.plan_constant_el_scan
    windows = []

    def field_form_only(*args, **kwargs):
        windows.append(kwargs.get("az_window"))
        if kwargs.get("az_window") is not None:
            raise AssertionError("the window form must not be called")
        return planner(*args, **kwargs)

    monkeypatch.setattr(sim, "plan_constant_el_scan", field_form_only)
    patch = ObservingPatch(
        name="COSMOS_CE",
        ra_center=COSMOS_FIELD.ra_center,
        dec_center=COSMOS_FIELD.dec_center,
        width=COSMOS_FIELD.width,
        height=COSMOS_FIELD.height,
        scan_type="constant_el",
        velocity=COSMOS_ROW["velocity"],
        elevation=43.0,
        scan_params={"rising": True, "az_min": 62.0, "az_max": 64.5},
    )
    block = TimelineBlock.science(
        patch, COSMOS_START, COSMOS_ROW["duration"], 62.0, 64.5, 43.0, 0, rising=True
    )
    block = _through_file(block, tmp_path)
    assert "window_scan" not in block.metadata
    assert block.metadata["scan_params"]["az_min"] == 62.0
    scan = sim._generate_trajectory_for_block(block, SITE)
    cp = scan.computed_params
    assert (cp["az_start"], cp["az_stop"]) == pytest.approx((59.842, 66.778), abs=1e-3)
    assert float(scan.trajectory.az.min()) < 62.0 - 2.0
    assert float(scan.trajectory.az.max()) > 64.5 + 2.0
    assert windows == [None]


class TestBlockWindow:
    """A block that stops at its scan's end keeps the last sample; a shorter one keeps its head."""

    @pytest.mark.parametrize("source", ["memory", "file"])
    def test_a_whole_block_keeps_the_last_sample(self, planned, tmp_path, source):
        scan, block, _, _ = planned("no_field")
        if source == "file":
            block = _through_file(block, tmp_path)
        rebuilt = _rebuild(block)
        assert rebuilt.duration == scan.duration
        assert len(rebuilt.trajectory.times) == len(scan.trajectory.times)
        assert rebuilt.trajectory.times[-1] == scan.duration

    @pytest.mark.parametrize(("short_by", "dropped"), [(0.004, 0), (0.05, 1)])
    def test_a_block_a_few_milliseconds_short_is_still_whole(self, planned, short_by, dropped):
        """Within 0.01 s of the scan's length the last sample stays; 0.05 s short it goes."""
        scan, block, _, _ = planned("no_field")
        stop = block.t_start + TimeDelta(scan.duration - short_by, format="sec")
        rebuilt = _rebuild(dataclasses.replace(block, t_stop=stop))
        assert len(rebuilt.trajectory.times) == len(scan.trajectory.times) - dropped

    def test_a_block_cut_short_returns_the_head(self, planned):
        scan, block, _, _ = planned("no_field")
        short = dataclasses.replace(block, t_stop=block.t_start + TimeDelta(600.0, format="sec"))
        rebuilt = _rebuild(short)
        head = scan.trajectory.times < 600.0
        assert len(rebuilt.trajectory.times) == int(np.count_nonzero(head))
        assert np.array_equal(rebuilt.trajectory.az, scan.trajectory.az[head])
        assert rebuilt.trajectory.start_time == scan.trajectory.start_time

    def test_a_block_longer_than_its_scan_returns_the_scan_whole(self, planned):
        scan, block, _, _ = planned("no_field")
        longer = dataclasses.replace(block, t_stop=block.t_stop + TimeDelta(60.0, format="sec"))
        _assert_identical(_rebuild(longer), scan)


def _spans(scan, *cuts):
    """Return the spans that cut ``scan`` at ``cuts`` seconds from its start, the last open."""
    start = scan.trajectory.start_time
    edges = [start, *(start + TimeDelta(c, format="sec") for c in cuts)]
    edges.append(start + TimeDelta(scan.duration + 0.1, format="sec"))
    return list(zip(edges[:-1], edges[1:], strict=True))


def _span_blocks(scan, spans):
    return [
        TimelineBlock.from_window_scan(
            scan, site=SITE, patch_name="cut", scan_index=0, subscan_index=j, span=span
        )
        for j, span in enumerate(spans)
    ]


class TestSpans:
    """Blocks written with ``span`` are consecutive slices of one scan."""

    @pytest.mark.parametrize("source", ["memory", "file"])
    def test_the_slices_join_into_the_scan(self, planned, tmp_path, source):
        """Cut mid-leg at 333.333 and 777.7777 s, the three blocks rebuild to the scan."""
        scan = planned("no_field")[0]
        blocks = _span_blocks(scan, _spans(scan, 333.333, 777.7777))
        if source == "file":
            path = tmp_path / "spans.ecsv"
            write_timeline(_timeline(*blocks), path)
            blocks = list(read_timeline(path).blocks)
        rebuilt = [sim._generate_trajectory_for_block(b, SITE).trajectory for b in blocks]
        for name in ARRAYS[1:]:
            joined = np.concatenate([getattr(t, name) for t in rebuilt])
            assert np.array_equal(joined, getattr(scan.trajectory, name)), name
        start = scan.trajectory.start_time
        times = np.concatenate([(t.start_time - start).sec + t.times for t in rebuilt])
        np.testing.assert_allclose(times, scan.trajectory.times, rtol=0.0, atol=1e-6)

    def test_only_a_block_after_the_scan_start_records_it(self, planned):
        scan = planned("no_field")[0]
        blocks = _span_blocks(scan, _spans(scan, 333.333))
        assert "scan_start" not in blocks[0].metadata
        assert blocks[1].metadata["scan_start"] == scan.trajectory.start_time.isot
        json.dumps(blocks[1].metadata)

    def test_each_block_records_the_envelope_of_its_own_samples(self, planned):
        scan = planned("no_field")[0]
        for block in _span_blocks(scan, _spans(scan, 333.333)):
            traj = sim._generate_trajectory_for_block(block, SITE).trajectory
            assert (block.az_start, block.az_end) == (float(traj.az.min()), float(traj.az.max()))
            assert block.end_pose_az == float(traj.az[-1])

    def test_a_later_block_without_the_key_rebuilds_from_its_own_start(self, planned):
        """Without ``scan_start`` the scan is replanned from the block's start: its head."""
        scan = planned("no_field")[0]
        later = _span_blocks(scan, _spans(scan, 333.333))[1]
        kept = sim._generate_trajectory_for_block(later, SITE).trajectory
        bare = sim._generate_trajectory_for_block(_replace_meta(later, scan_start=None), SITE)
        assert not np.array_equal(bare.trajectory.az[: kept.az.size], kept.az)
        assert bare.trajectory.az[0] == scan.trajectory.az[0]

    @pytest.mark.parametrize("value", [1.0, "not a time"])
    def test_a_malformed_scan_start_is_refused(self, planned, value):
        scan = planned("no_field")[0]
        later = _replace_meta(_span_blocks(scan, _spans(scan, 333.333))[1], scan_start=value)
        with pytest.raises(ScanParamsSchemaError, match="scan_start must be an ISO time string"):
            sim._generate_trajectory_for_block(later, SITE)

    def test_a_span_that_does_not_stop_after_it_starts_is_refused(self, planned):
        scan = planned("no_field")[0]
        start = scan.trajectory.start_time
        with pytest.raises(ValueError, match="span must stop after it starts"):
            _span_blocks(scan, [(start, start)])

    def test_a_span_before_the_scan_is_refused(self, planned):
        scan = planned("no_field")[0]
        start = scan.trajectory.start_time
        early = start - TimeDelta(1.0, format="sec")
        with pytest.raises(ValueError, match="before the scan's start"):
            _span_blocks(scan, [(early, start + TimeDelta(60.0, format="sec"))])

    def test_a_span_with_one_sample_is_refused(self, planned):
        scan = planned("no_field")[0]
        start = scan.trajectory.start_time + TimeDelta(100.0, format="sec")
        with pytest.raises(BlockNotReconstructableError):
            _span_blocks(scan, [(start, start + TimeDelta(0.05, format="sec"))])


class TestSunScreen:
    """The rebuild's Sun advisory describes the samples the block returns."""

    def test_a_block_that_enters_the_zone_warns_once(self, planned):
        block = planned("dawn")[1]
        with pytest.warns(PointingWarning, match=re.escape(SUN_MESSAGE)) as record:
            _rebuild(block)
        assert len([w for w in record if issubclass(w.category, PointingWarning)]) == 1

    def test_a_head_that_ends_before_the_zone_is_silent(self, planned):
        scan, block, _, _ = planned("dawn")
        coords = Coordinates(SITE)
        head = scan.trajectory.times < 600.0
        times = scan.trajectory.start_time + TimeDelta(scan.trajectory.times[head], format="sec")
        sun_az, sun_el = coords.get_sun_altaz(times)
        separation = coords.angular_separation(
            scan.trajectory.az[head], scan.trajectory.el[head], sun_az, sun_el
        )
        assert float(np.min(separation)) > SITE.sun_avoidance.exclusion_radius
        short = dataclasses.replace(block, t_stop=block.t_start + TimeDelta(600.0, format="sec"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _rebuild(short)
        assert [w for w in caught if issubclass(w.category, PointingWarning)] == []


class TestFieldDrift:
    """The recorded field reaches the planner on rebuild, so its drift check runs again."""

    LATE = Time("2026-11-15T08:22:35", scale="utc")

    def test_a_late_scan_warns_when_planned_and_rebuilt_but_not_when_written(self):
        with pytest.warns(PointingWarning, match=DRIFT_MESSAGE):
            scan, field, _ = _plan("centre_with_field", start_time=self.LATE)
        # The writer's own rebuild is a probe and raises nothing.
        block = TimelineBlock.from_window_scan(
            scan, site=SITE, patch_name="P", scan_index=0, field=field
        )
        with pytest.warns(PointingWarning, match=DRIFT_MESSAGE):
            _assert_identical(_rebuild(block), scan)

    def test_the_same_scan_without_its_field_rebuilds_silently(self):
        scan = plan_constant_el_scan(site=SITE, **{**COSMOS_ROW, "start_time": self.LATE})
        block = TimelineBlock.from_window_scan(scan, site=SITE, patch_name="P", scan_index=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rebuilt = _rebuild(block)
        assert [w for w in caught if issubclass(w.category, PointingWarning)] == []
        _assert_identical(rebuilt, scan)


class TestWriter:
    """What ``TimelineBlock.from_window_scan`` refuses and what it writes."""

    def test_a_pong_scan_is_refused(self):
        field = FieldRegion(ra_center=150.125, dec_center=2.1666665, width=4.0, height=4.0)
        pong = plan_pong_scan(
            field=field,
            velocity=0.2,
            spacing=0.1,
            site=SITE,
            start_time=Time("2026-11-15T09:00:00", scale="utc"),
        )
        with pytest.raises(ValueError, match="ConstantElScanConfig"):
            TimelineBlock.from_window_scan(pong, site=SITE, patch_name="P", scan_index=0)

    def test_a_trajectory_without_a_start_is_refused(self, planned):
        scan = planned("no_field")[0]
        unanchored = dataclasses.replace(
            scan, trajectory=dataclasses.replace(scan.trajectory, start_time=None)
        )
        with pytest.raises(ValueError, match="start_time"):
            TimelineBlock.from_window_scan(unanchored, site=SITE, patch_name="P", scan_index=0)

    def test_a_slice_is_refused(self, planned):
        _, block, _, _ = planned("no_field")
        short = dataclasses.replace(block, t_stop=block.t_start + TimeDelta(600.0, format="sec"))
        with pytest.raises(ValueError, match="a slice cannot be replayed"):
            TimelineBlock.from_window_scan(_rebuild(short), site=SITE, patch_name="P", scan_index=0)

    @pytest.mark.parametrize(
        ("name", "tag"),
        [
            pytest.param("module_im6", None, id="offset-omitted"),
            pytest.param("centre_with_field", "IM6", id="offset-not-used"),
            pytest.param("module_im6", "IM1", id="another-offset"),
        ],
    )
    def test_an_offset_other_than_the_planned_one_is_refused(self, planned, name, tag):
        scan, _, field, _ = planned(name)
        offset = None if tag is None else get_primecam_offset(tag)
        with pytest.raises(ValueError, match=REFUSED):
            TimelineBlock.from_window_scan(
                scan, site=SITE, patch_name="P", scan_index=0, field=field, detector_offset=offset
            )

    def _modified(self, scan, how):
        traj = scan.trajectory
        if how == "retunes":
            traj = inject_retune(traj, retune_interval=300.0, retune_duration=5.0)
        elif how == "shifted":
            traj = dataclasses.replace(traj, az=traj.az + 0.5)
        else:
            half = len(traj.times) // 2
            traj = dataclasses.replace(
                traj, **{name: getattr(traj, name)[:half] for name in ARRAYS}
            )
        return dataclasses.replace(scan, trajectory=traj)

    @pytest.mark.parametrize("how", ["retunes", "shifted", "truncated"])
    def test_a_modified_trajectory_is_refused(self, planned, how):
        scan, _, field, _ = planned("centre_with_field")
        modified = self._modified(scan, how)
        with pytest.raises(ValueError, match=REFUSED):
            TimelineBlock.from_window_scan(
                modified, site=SITE, patch_name="P", scan_index=0, field=field
            )

    def test_a_source_ces_scan_is_refused(self):
        """Its config is a ConstantElScanConfig too, but its sweep drifts in azimuth."""
        scan = plan_source_ces(
            body="jupiter",
            footprint="c",
            start_time=Time("2026-03-15T21:30:00", scale="utc"),
            site=SITE,
        )
        with pytest.raises(ValueError, match=REFUSED):
            TimelineBlock.from_window_scan(scan, site=SITE, patch_name="P", scan_index=0)

    @pytest.mark.parametrize("source", ["memory", "file"])
    @pytest.mark.parametrize("name", sorted(MOVED_UP_A_TURN))
    def test_a_window_moved_up_a_turn_is_written_and_replays(self, tmp_path, name, source):
        """The block records the placed window and rebuilds to the scan sample for sample."""
        scan = plan_constant_el_scan(site=SITE, **MOVED_UP_A_TURN[name])
        assert scan.config.az_start == MOVED_UP_A_TURN[name]["az_window"][0] + 360.0
        block = TimelineBlock.from_window_scan(scan, site=SITE, patch_name="P", scan_index=0)
        assert block.metadata["window_scan"]["az_window"] == [
            scan.config.az_start,
            scan.config.az_stop,
        ]
        if source == "file":
            block = _through_file(block, tmp_path)
        rebuilt = _rebuild(block)
        _assert_identical(rebuilt, scan)
        assert rebuilt.duration == scan.duration

    def test_an_offset_name_that_is_not_text_is_refused(self):
        offset = InstrumentOffset(dx=0.0, dy=0.0, name=b"x")
        scan, field, _ = _plan("centre_with_field", detector_offset=offset)
        with pytest.raises(ValueError, match=r"detector_offset\.name"):
            TimelineBlock.from_window_scan(
                scan, site=SITE, patch_name="P", scan_index=0, field=field, detector_offset=offset
            )

    def test_a_zero_offset_is_accepted_and_replays(self):
        zero = InstrumentOffset(dx=0.0, dy=0.0)
        scan, field, _ = _plan("centre_with_field", detector_offset=zero)
        block = TimelineBlock.from_window_scan(
            scan, site=SITE, patch_name="P", scan_index=0, field=field, detector_offset=zero
        )
        assert block.metadata["window_scan"]["detector_offset"] == dataclasses.asdict(zero)
        _assert_identical(_rebuild(block), scan)

    @pytest.mark.parametrize("name", ["centre_with_field", "module_im6"])
    def test_a_rebuilt_scan_writes_the_same_block(self, planned, name):
        _, block, field, offset = planned(name)
        again = TimelineBlock.from_window_scan(
            _rebuild(block),
            site=SITE,
            patch_name=block.patch_name,
            scan_index=block.scan_index,
            field=field,
            detector_offset=offset,
        )
        assert again == block

    @pytest.mark.parametrize("source", ["memory", "file"])
    def test_a_field_form_scan_is_written_as_the_window_scan_it_swept(self, tmp_path, source):
        """A whole field-form scan replays, so it is recorded and rebuilt in the window form."""
        scan = plan_constant_el_scan(
            COSMOS_FIELD,
            elevation=43.0,
            velocity=COSMOS_ROW["velocity"],
            site=SITE,
            start_time=COSMOS_START - TimeDelta(1800.0, format="sec"),
        )
        assert scan.summary.startswith("Constant-El scan:")
        block = TimelineBlock.from_window_scan(
            scan, site=SITE, patch_name="P", scan_index=0, field=COSMOS_FIELD
        )
        record = block.metadata["window_scan"]
        assert record["az_window"] == [scan.config.az_start, scan.config.az_stop]
        assert record["elevation"] == 43.0
        if source == "file":
            block = _through_file(block, tmp_path)
        rebuilt = _rebuild(block)
        _assert_identical(rebuilt, scan)
        assert rebuilt.summary.startswith("Constant-El window scan:")

    def test_a_timeline_of_one_window_block_validates(self, planned):
        assert _timeline(planned("module_im6")[1]).validate() == []
