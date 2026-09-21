"""Tests for the night summary, the dispatch sheet and the metadata payload."""

import json

import pytest
from astropy.time import Time

from fyst_trajectories.overhead import (
    CalibrationPolicy,
    NightSummary,
    ObservingTimeline,
    OverheadModel,
    dispatch_sheet,
    read_calibration_night_metadata,
    read_timeline,
    summarize_calibration_night,
    write_timeline,
)
from fyst_trajectories.overhead.calibration_night.policy import (
    CALNIGHT_SCHEMA_VERSION,
    encode_calibration_night_metadata,
    tables_as_record,
    tables_from_record,
)
from fyst_trajectories.overhead.calibration_night.tables import DEFAULT_SCAN_TABLES


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
        assert 0.5 < saturn.mean_duty_cycle < 0.7
        assert saturn.module_crossings["c"] > 0.2
        assert summary.tuning_minutes == pytest.approx(25.0)
        assert summary.slew_minutes > 0.0 and summary.idle_minutes > 0.0
        assert set(summary.idle_reasons) <= {"nothing_available", "waiting_for_pass"}
        assert {d["reason"] for d in summary.deferrals} == {"window_closed"}
        assert any("acceleration" in w for w in summary.warnings)

    def test_text_rendering(self, short_night):
        _, timeline = short_night
        text = str(summarize_calibration_night(timeline))
        assert text.startswith("Calibration night 2026-09-11 06:30 to 2026-09-11 07:30 UTC")
        assert "saturn: 3 visit(s), 3 pass(es)" in text
        assert "uranus: no passes" in text
        assert "requested: az_accel=1.5, az_speed=1.5, az_throw=" in text
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
        assert any("confirm the execution layer forwards them" in ln for ln in lines)
        assert any("find_detectors at az" in ln and "operator action" in ln for ln in rows)

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

    def test_identical_after_round_trip(self, short_night, tmp_path):
        _, timeline = short_night
        path = tmp_path / "night.ecsv"
        write_timeline(timeline, path)
        back = read_timeline(path)
        assert dispatch_sheet(back) == dispatch_sheet(timeline)
        assert str(summarize_calibration_night(back)) == str(summarize_calibration_night(timeline))

    def test_every_block_metadata_is_json(self, short_night):
        _, timeline = short_night
        for block in timeline.blocks:
            json.dumps(block.metadata)


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
