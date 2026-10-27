"""Tests for the night summary, the dispatch sheet and the metadata payload."""

import dataclasses
import json

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories import get_fyst_site
from fyst_trajectories.overhead import (
    BlockType,
    CalibrationNightPolicy,
    CalibrationPolicy,
    CalibrationType,
    NightContext,
    NightSummary,
    ObservingTimeline,
    OverheadModel,
    ScriptedSelection,
    TimelineBlock,
    VisitPlan,
    dispatch_records,
    dispatch_sheet,
    plan_calibration_night,
    read_calibration_night_metadata,
    read_timeline,
    schedule_to_trajectories,
    summarize_calibration_night,
    write_timeline,
)
from fyst_trajectories.overhead.calibration_night.night import _run_night
from fyst_trajectories.overhead.calibration_night.policy import (
    CALNIGHT_SCHEMA_VERSION,
    encode_calibration_night_metadata,
    tables_as_record,
    tables_from_record,
)
from fyst_trajectories.overhead.calibration_night.reporting import _LATENESS_RULE
from fyst_trajectories.overhead.calibration_night.tables import DEFAULT_SCAN_TABLES
from fyst_trajectories.overhead.utils import _search_start_time

# A pass dict of the keys the pinned execution layer's source-scan task reads,
# plus the two sequence keys it has no use for.
_BASE_PASS_DICT = {
    "body": "saturn",
    "footprint": "c",
    "el_bore": 50.0,
    "mode": "rising",
    "boresight_rot": None,
    "timestep": 0.1,
    "az_accel": 1.0,
    "pass_index": 0,
    "n_passes": 1,
}


def _night_of_parked_passes(scan_params):
    """Plan half an hour of parked Saturn passes, each carrying ``scan_params`` (no sky)."""

    def parked(state, ctx, body, overrides=None):
        block = TimelineBlock.calibration(
            CalibrationType.PLANET_CAL,
            t_start=state.t,
            duration=600.0,
            az=state.az,
            el=state.el,
            scan_index=state.scan_counter,
            target=body,
            scan_params=scan_params,
        )
        return VisitPlan(body, True, None, None, (block,), ())

    ctx = NightContext.build(
        ["saturn"],
        get_fyst_site(),
        "2026-09-11T06:30:00",
        "2026-09-11T07:00:00",
        visit_planner=parked,
    )
    return _run_night(ctx)


@pytest.fixture(scope="module")
def im0_night():
    """Plan the package's short night again, with the centre module named ``IM0``."""
    return plan_calibration_night(
        ["saturn", "uranus"],
        get_fyst_site(),
        "2026-09-11T06:30:00",
        "2026-09-11T07:30:00",
        policy=CalibrationNightPolicy(footprint="IM0"),
    )


class TestSummary:
    """The summary value and its text rendering."""

    def test_totals(self, short_night):
        _, timeline = short_night
        summary = summarize_calibration_night(timeline)
        assert isinstance(summary, NightSummary)
        assert summary.targets == ("saturn", "uranus")
        saturn, uranus = summary.bodies
        assert saturn.passes == 3 and saturn.visits == 3
        assert uranus.passes == 0
        assert 25.0 < saturn.minutes_on_source < 35.0
        assert 0.3 < saturn.mean_duty_cycle < 0.45
        assert saturn.module_crossings["c"] > 0.2
        assert summary.tuning_minutes == pytest.approx(25.0)
        assert summary.slew_minutes > 0.0 and summary.idle_minutes > 0.0
        assert set(summary.idle_reasons) <= {
            "nothing_available",
            "waiting_for_pass",
            "window_closed",
        }
        assert summary.deferrals == ()
        assert any("on-sky azimuth speed" in w for w in summary.warnings)

    def test_visits_are_counted_as_the_dispatch_record_counts_them(self, short_night):
        """A pass at ``pass_index`` 0 opens a visit even with no move before it."""
        _, timeline = short_night
        passes = [
            b
            for b in timeline.blocks
            if b.block_type == BlockType.CALIBRATION and b.metadata.get("cal_type") == "planet_cal"
        ]
        back_to_back = dataclasses.replace(timeline, blocks=passes)
        saturn = summarize_calibration_night(back_to_back).bodies[0]
        assert saturn.visits == len(dispatch_records(back_to_back)["visits"]) == 3

    def test_text_rendering(self, short_night):
        _, timeline = short_night
        text = str(summarize_calibration_night(timeline))
        assert text.startswith("Calibration night 2026-09-11 06:30 to 2026-09-11 07:30 UTC")
        assert "saturn: 3 visit(s), 3 pass(es)" in text
        assert "uranus: no passes" in text
        assert "requested: az_accel=1, az_speed=1.5 (+2 more)" in text
        assert "modules crossed: c" in text


class TestDispatchSheet:
    """Rendering only, byte-identical after an ECSV round trip."""

    def test_rows(self, short_night):
        _, timeline = short_night
        sheet = dispatch_sheet(timeline)
        lines = sheet.splitlines()
        assert lines[0].startswith("# calibration night, targets saturn, uranus")
        rows = [ln for ln in lines if ln[:6].strip().isdigit()]
        assert len(rows) == len(timeline.blocks)
        pass_rows = [ln for ln in rows if "source_scan on saturn" in ln]
        assert len(pass_rows) == 3
        dict_lines = [ln for ln in lines if ln.strip().startswith("scan_params=")]
        assert len(dict_lines) == 3
        params = json.loads(
            dict_lines[0].split("scan_params=", 1)[1].split("  scheduled_t0_unix")[0]
        )
        assert "window" not in params and params["body"] == "saturn"
        assert any("find_detectors at az" in ln and "operator action" in ln for ln in rows)

    def test_every_pass_row_names_the_keys_the_execution_layer_drops(self, short_night):
        """The planner writes ``az_speed`` and ``eta_offset_deg`` into every pass dict.

        The pinned execution layer's source-scan task reads neither, so
        the note under each dict names both, and nothing else on this
        default-policy night.
        """
        _, timeline = short_night
        lines = dispatch_sheet(timeline).splitlines()
        dicts = [i for i, ln in enumerate(lines) if ln.strip().startswith("scan_params=")]
        assert len(dicts) == 3
        notes = [ln for ln in lines if ln.strip().startswith("note:")]
        assert [lines[i + 1] for i in dicts] == notes
        assert set(notes) == {
            "        note: confirm the execution layer forwards az_speed, eta_offset_deg"
        }

    @pytest.mark.parametrize(
        "extra, named",
        [
            pytest.param(None, None, id="no-dict"),
            pytest.param({"az_padding": 0.0}, None, id="read-keys-only"),
            pytest.param({"eta_offset_deg": 0.25}, "eta_offset_deg", id="eta-offset"),
            pytest.param(
                {
                    "az_speed": 0.8,
                    "az_throw": 3.0,
                    "dwell": 300.0,
                    "eta_offset_deg": -0.4,
                    "footprint_margin": 0.2,
                },
                "az_speed, az_throw, dwell, eta_offset_deg, footprint_margin",
                id="every-dropped-key",
            ),
        ],
    )
    def test_the_note_names_exactly_the_dropped_keys(self, extra, named, tmp_path):
        """A key the task reads (``az_accel``, ``az_padding``) is never named, one it drops always.

        The note is a function of the dict's keys alone, so the sheet stays
        byte-identical after an ECSV round trip.
        """
        timeline = _night_of_parked_passes(None if extra is None else {**_BASE_PASS_DICT, **extra})
        sheet = dispatch_sheet(timeline)
        lines = sheet.splitlines()
        dicts = [i for i, ln in enumerate(lines) if ln.strip().startswith("scan_params=")]
        assert len(dicts) == 3
        notes = [ln for ln in lines if ln.strip().startswith("note:")]
        if named is None:
            assert notes == []
        else:
            assert [lines[i + 1] for i in dicts] == notes
            assert set(notes) == {f"        note: confirm the execution layer forwards {named}"}
        path = tmp_path / "night.ecsv"
        write_timeline(timeline, path)
        assert dispatch_sheet(read_timeline(path)) == sheet

    def test_slack_column_is_the_waiting_idle(self, short_night):
        _, timeline = short_night
        blocks = sorted(timeline.blocks, key=lambda b: b.t_start.unix)
        waits = [
            previous.duration
            for previous, block in zip(blocks, blocks[1:])
            if block.metadata.get("cal_type") == "planet_cal"
            and previous.metadata.get("reason") == "waiting_for_pass"
        ]
        pass_rows = [ln for ln in dispatch_sheet(timeline).splitlines() if "source_scan on" in ln]
        assert len(waits) == len(pass_rows) == 3
        for row, wait in zip(pass_rows, waits):
            assert f"slack {wait:.0f} s" in row

    def test_header_states_the_rule_and_the_lead(self, short_night):
        """The header carries the execution layer's rule and the lead the policy reserved."""
        _, timeline = short_night
        lines = dispatch_sheet(timeline).splitlines()
        assert lines[1] == f"# lateness rule: {_LATENESS_RULE}"
        lead = CalibrationNightPolicy().dispatch_lead_seconds
        assert lines[2].startswith(f"# dispatch lead: {lead:.0f} s reserved before every pass")
        assert lines[3] == "#   row  start                 duration  action"

    def test_scheduled_t0_is_where_the_search_began(self, short_night):
        """The dict line hands the execution layer the search start, not the pass start."""
        _, timeline = short_night
        blocks = sorted(timeline.blocks, key=lambda b: b.t_start.unix)
        passes = [b for b in blocks if b.metadata.get("cal_type") == "planet_cal"]
        lines = [ln for ln in dispatch_sheet(timeline).splitlines() if "scheduled_t0_unix=" in ln]
        assert len(lines) == len(passes) == 3
        for block, line in zip(passes, lines):
            printed = float(line.rsplit("scheduled_t0_unix=", 1)[1])
            anchor = _search_start_time(block.metadata["search_start"])
            assert printed == pytest.approx(anchor.unix, abs=1e-3)
            assert 0.0 < (block.t_start - anchor).to_value("s") < 60.0
        # A block written without the record (an older night) gives its start.
        first = passes[0]
        metadata = {k: v for k, v in first.metadata.items() if k != "search_start"}
        stripped = [
            dataclasses.replace(first, metadata=metadata) if b is first else b for b in blocks
        ]
        older = dataclasses.replace(timeline, blocks=stripped)
        line = next(ln for ln in dispatch_sheet(older).splitlines() if "scheduled_t0_unix=" in ln)
        printed = float(line.rsplit("scheduled_t0_unix=", 1)[1])
        assert printed == pytest.approx(first.t_start.unix, abs=1e-3)

    def test_identical_after_round_trip(self, short_night, tmp_path):
        _, timeline = short_night
        path = tmp_path / "night.ecsv"
        write_timeline(timeline, path)
        back = read_timeline(path)
        assert dispatch_sheet(back) == dispatch_sheet(timeline)
        assert str(summarize_calibration_night(back)) == str(summarize_calibration_night(timeline))

    @pytest.mark.parametrize("search_start", [True, False], ids=["search-start", "start-only"])
    def test_identical_after_round_trip_on_a_leap_second_day(
        self, leap_second_day_night, search_start, tmp_path
    ):
        """Every time and duration prints as the file stores it.

        Without ``search_start`` the sheet hands over the pass start, whose
        Unix time on 2016-12-31 differs in its third decimal between the
        instant in memory and the millisecond in the file. The slew sits on
        a rounding edge: 2.9996 s held in memory printed as 0.0 min, while
        the 3.000 s between the stored milliseconds prints as 0.1 min, from
        memory and from the file alike.
        """
        night = leap_second_day_night(search_start=search_start)
        start = night.blocks[-1].t_start
        assert round(float(start.unix), 3) != round(float(Time(start.iso, scale="utc").unix), 3)
        path = tmp_path / "night.ecsv"
        write_timeline(night, path)
        sheet = dispatch_sheet(night)
        assert dispatch_sheet(read_timeline(path)) == sheet
        slew = night.blocks[0]
        stored = Time(slew.t_stop.iso, scale="utc") - Time(slew.t_start.iso, scale="utc")
        slew_row = next(line for line in sheet.splitlines() if " slew az " in line)
        assert f"{stored.to_value('s') / 60.0:7.1f} min" in slew_row
        assert f"{slew.duration / 60.0:7.1f} min" not in slew_row, "the slew is off the edge"
        printed = float(sheet.rsplit("scheduled_t0_unix=", 1)[1].split()[0])
        if search_start:
            anchor = _search_start_time(night.blocks[-1].metadata["search_start"])
            assert printed == round(float(anchor.unix), 3)
        else:
            assert printed == round(float(Time(start.iso, scale="utc").unix), 3)

    def test_every_block_metadata_is_json(self, short_night):
        _, timeline = short_night
        for block in timeline.blocks:
            json.dumps(block.metadata)


class TestCentreModuleSpelling:
    """A night whose policy names the centre module ``IM0`` dispatches as a ``"c"`` night does.

    The execution layer's centred-only check compares a pass dict's
    ``footprint`` with ``"c"`` and ``"center"`` as strings, so the dict names
    the module by its canonical name whatever spelling the policy was given.
    """

    def test_every_dict_names_the_centre_module_c(self, im0_night):
        passes = [b for b in im0_night.blocks if b.scan_type == "planet_cal"]
        assert len(passes) == 3
        footprints = [b.metadata["scan_params"]["footprint"] for b in passes]
        assert all(fp in ("c", "center") for fp in footprints)
        assert set(footprints) == {"c"}
        # The night's own record keeps the spelling the caller gave.
        assert read_calibration_night_metadata(im0_night)["policy"]["footprint"] == "IM0"

    def test_the_sheet_is_the_c_night_sheet(self, short_night, im0_night, tmp_path):
        _, c_night = short_night
        sheet = dispatch_sheet(im0_night)
        assert sheet == dispatch_sheet(c_night)
        path = tmp_path / "night.ecsv"
        write_timeline(im0_night, path)
        assert dispatch_sheet(read_timeline(path)) == sheet

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning"
    )
    def test_the_passes_rebuild_as_on_the_c_night(self, short_night, im0_night):
        _, c_night = short_night
        want = schedule_to_trajectories(c_night, science_only=False)
        got = schedule_to_trajectories(im0_night, science_only=False)
        assert len(got) == len(want) == 3
        for (_, rebuilt), (_, reference) in zip(got, want):
            assert rebuilt.computed_params == reference.computed_params
            for name in ("times", "az", "el", "scan_flag"):
                assert np.array_equal(
                    getattr(rebuilt.trajectory, name), getattr(reference.trajectory, name)
                )


class TestMetadataPayload:
    """The namespaced header payload and its decoder."""

    def test_round_trip_through_ecsv(self, short_night, tmp_path):
        _, timeline = short_night
        original = read_calibration_night_metadata(timeline)
        path = tmp_path / "night.ecsv"
        write_timeline(timeline, path)
        assert read_calibration_night_metadata(read_timeline(path)) == original
        assert timeline.metadata["calnight_schema_version"] == CALNIGHT_SCHEMA_VERSION
        assert set(timeline.metadata) == {"calnight_schema_version", "calnight_json"}

    def test_two_write_cycles_do_not_accumulate(self, short_night, tmp_path):
        _, timeline = short_night
        first = tmp_path / "a.ecsv"
        second = tmp_path / "b.ecsv"
        write_timeline(timeline, first)
        write_timeline(read_timeline(first), second)
        assert read_timeline(second).metadata == timeline.metadata

    def test_missing_and_unsupported_payload(self, short_night):
        site, _ = short_night
        bare = ObservingTimeline(
            blocks=[],
            site=site,
            start_time=Time("2026-09-11T06:30:00", scale="utc"),
            end_time=Time("2026-09-11T06:30:00", scale="utc"),
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )
        with pytest.raises(KeyError, match="no calibration-night metadata"):
            read_calibration_night_metadata(bare)
        bare.metadata.update(
            {"calnight_schema_version": CALNIGHT_SCHEMA_VERSION + 1, "calnight_json": "{}"}
        )
        with pytest.raises(ValueError, match="schema version"):
            read_calibration_night_metadata(bare)

    def test_encode_is_sorted_json(self):
        payload = encode_calibration_night_metadata({"b": 1, "a": [2]})  # type: ignore[arg-type]
        assert payload["calnight_json"] == '{"a": [2], "b": 1}'

    def test_tables_record_round_trip(self):
        record = tables_as_record(dict(DEFAULT_SCAN_TABLES))
        assert tables_from_record(record) == dict(DEFAULT_SCAN_TABLES)
        json.dumps(record)

    def test_the_payload_records_the_axis_limits(self, short_night, tmp_path):
        """The payload keeps its own copy, the one a resume holds the site to."""
        site, timeline = short_night
        limits = dataclasses.asdict(site.telescope_limits)
        assert read_calibration_night_metadata(timeline)["telescope_limits"] == limits
        path = tmp_path / "night.ecsv"
        write_timeline(timeline, path)
        assert read_calibration_night_metadata(read_timeline(path))["telescope_limits"] == limits

    def test_a_set_aside_entry_records_when(self, short_night):
        """The summary lists the entry with the instant it was set aside."""
        site, _ = short_night

        def parked(state, ctx, body, overrides=None):
            block = TimelineBlock.calibration(
                CalibrationType.PLANET_CAL,
                t_start=state.t,
                duration=600.0,
                az=state.az,
                el=state.el,
                scan_index=state.scan_counter,
                target=body,
            )
            return VisitPlan(body, True, None, None, (block,), ())

        ctx = NightContext.build(
            ["saturn", "jupiter"],
            site,
            "2026-09-11T06:30:00",
            "2026-09-11T08:00:00",
            policy=CalibrationNightPolicy(max_wait_seconds=600.0, time_step=300.0),
            visit_planner=parked,
        )
        summary = summarize_calibration_night(
            _run_night(ctx, ScriptedSelection(["jupiter", "saturn"]))
        )
        assert summary.unplaced == (
            {"body": "jupiter", "at": "2026-09-11 06:40:00.000", "overrides": {}},
        )
        assert "  unplaced script entry: jupiter {}" in str(summary).splitlines()
