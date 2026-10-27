"""``dispatch_records`` returns a planned night as one JSON-ready record per visit.

On the package's hour-long ``short_night`` the records repeat the dispatch
sheet's pass rows in order and are the same from the night in memory and
from its ECSV. The grouping branches that night never reaches (a second pass
joining its visit with a slew before it, escape slews with no pass behind
them, a waiting idle that no pass follows) are pinned on nights built by hand.
"""

import json
import re

import pytest
from _calibration_night_helpers import _with_policy
from astropy.time import Time, TimeDelta

from fyst_trajectories.overhead import (
    CalibrationNightPolicy,
    CalibrationPolicy,
    DeferralReason,
    IdleRecord,
    NightContext,
    NightRecord,
    ObservingTimeline,
    OperationRecord,
    OverheadModel,
    PassRecord,
    ScanParamsSchemaError,
    SlewRecord,
    TimelineBlock,
    VisitRecord,
    dispatch_records,
    dispatch_sheet,
    read_calibration_night_metadata,
    read_timeline,
    write_timeline,
)
from fyst_trajectories.overhead.calibration_night.reporting import _LATENESS_RULE

# The hand-off contract: every key of each record, in order.
NIGHT_KEYS = [
    "schema_version",
    "source",
    "targets",
    "sun_policy",
    "usable_interval_utc",
    "lateness_rule",
    "dispatch_lead_s",
    "unforwarded_keys_note",
    "night_start_utc",
    "night_end_utc",
    "visits",
    "idle",
]
VISIT_KEYS = ["body", "slews", "detector_operations", "passes"]
SLEW_KEYS = ["name", "t_start_utc", "t_start_unix", "duration_s", "az_from", "az_to", "el_to"]
OPERATION_KEYS = ["operation", "t_start_utc", "t_start_unix", "duration_s", "az", "el"]
PASS_KEYS = [
    "t_start_utc",
    "t_start_unix",
    "scheduled_t0_unix",
    "duration_s",
    "slack_s",
    "mode",
    "el_bore",
    "wrap_az",
    "n_legs",
    "science_fraction",
    "scan_params",
    "unforwarded_keys",
]
IDLE_KEYS = ["t_start_utc", "t_start_unix", "duration_s", "reason"]

_PASS_ROW = re.compile(r"source_scan on \w+: .*; slack (\d+) s; elapsed")
_PASS_PAIR = re.compile(r"^\s+scan_params=(\{.*\})  scheduled_t0_unix=([\d.]+)$")
_PASS_NOTE = re.compile(r"^\s+note: confirm the execution layer forwards (.+)$")

_WAITING = str(DeferralReason.WAITING_FOR_PASS)


@pytest.fixture(scope="module")
def ecsv_night(short_night, tmp_path_factory):
    """Return the short night read back from its ECSV."""
    _, timeline = short_night
    path = tmp_path_factory.mktemp("records") / "night.ecsv"
    write_timeline(timeline, path)
    return read_timeline(path)


def _sheet_passes(sheet):
    """Return the sheet's pass rows in order: (slack, params json, t0 text, note keys)."""
    lines = sheet.splitlines()
    rows = []
    for n, line in enumerate(lines):
        row = _PASS_ROW.search(line)
        if row is None:
            continue
        pair = _PASS_PAIR.match(lines[n + 1])
        assert pair is not None, lines[n + 1]
        note = _PASS_NOTE.match(lines[n + 2]) if n + 2 < len(lines) else None
        keys = note.group(1).split(", ") if note else []
        rows.append((int(row.group(1)), pair.group(1), pair.group(2), keys))
    return rows


def _assert_json_native(value, where="record"):
    """Every container is a dict with str keys or a list, every leaf a JSON builtin."""
    if type(value) is dict:
        for key, item in value.items():
            assert type(key) is str, where
            _assert_json_native(item, f"{where}[{key!r}]")
    elif type(value) is list:
        for i, item in enumerate(value):
            _assert_json_native(item, f"{where}[{i}]")
    else:
        assert type(value) in (str, float, int, bool, type(None)), (
            f"{where} is {type(value).__name__}"
        )


class TestShortNight:
    """A planned night, in memory and read back from ECSV."""

    def test_every_pass_matches_its_sheet_row(self, short_night):
        """Each pass record repeats what the sheet prints for it, in order."""
        _, timeline = short_night
        record = dispatch_records(timeline)
        passes = [p for visit in record["visits"] for p in visit["passes"]]
        rows = _sheet_passes(dispatch_sheet(timeline))
        assert len(rows) == 3
        for record_pass, (slack, params_json, t0_text, keys) in zip(passes, rows, strict=True):
            assert json.dumps(record_pass["scan_params"], sort_keys=True) == params_json
            assert f"{record_pass['scheduled_t0_unix']:.3f}" == t0_text
            assert f"{record_pass['slack_s']:.0f}" == str(slack)
            assert record_pass["unforwarded_keys"] == keys == ["az_speed", "eta_offset_deg"]

    def test_visits_group_the_operations_before_their_pass(self, short_night):
        """A visit with a pass names its body and finishes its operations before the pass."""
        _, timeline = short_night
        record = dispatch_records(timeline)
        with_passes = [visit for visit in record["visits"] if visit["passes"]]
        assert len(with_passes) == 3
        for visit in with_passes:
            assert visit["body"] == visit["passes"][0]["scan_params"]["body"]
            operations = [op["operation"] for op in visit["detector_operations"]]
            assert "retune" in operations
            last_operation = visit["detector_operations"][-1]
            first_pass = visit["passes"][0]
            assert (
                last_operation["t_start_unix"] + last_operation["duration_s"]
                <= first_pass["scheduled_t0_unix"]
            )
            assert (
                last_operation["t_start_unix"]
                + last_operation["duration_s"]
                + record["dispatch_lead_s"]
                <= first_pass["t_start_unix"]
            )
        first = with_passes[0]
        assert [op["operation"] for op in first["detector_operations"]][:2] == [
            "find_detectors",
            "skydip",
        ]
        assert record["targets"] == ["saturn", "uranus"]
        assert record["usable_interval_utc"] is not None
        assert record["lateness_rule"] == _LATENESS_RULE
        assert record["schema_version"] == 1
        assert record["source"] is None

    def test_the_record_is_the_same_from_ecsv(self, short_night, ecsv_night):
        """Every time comes from the block times as the file stores them."""
        _, timeline = short_night
        assert dispatch_records(ecsv_night) == dispatch_records(timeline)

    def test_json_dumps_is_the_whole_serialisation(self, short_night):
        _, timeline = short_night
        record = dispatch_records(timeline, source="night.ecsv")
        _assert_json_native(record)
        assert json.loads(json.dumps(record)) == record
        assert record["source"] == "night.ecsv"

    def test_the_record_carries_the_lead_as_the_planner_reads_it(self, short_night):
        """The lead is the recorded policy's; a record from before the field existed gives 0."""
        _, timeline = short_night
        record = dispatch_records(timeline)
        assert record["dispatch_lead_s"] == CalibrationNightPolicy().dispatch_lead_seconds == 20.0
        older = _with_policy(timeline, drop=("dispatch_lead_seconds",))
        assert "dispatch_lead_seconds" not in read_calibration_night_metadata(older)["policy"]
        assert dispatch_records(older)["dispatch_lead_s"] == 0.0

    def test_a_policy_the_planner_refuses_is_refused_as_the_sheet_refuses_it(self, short_night):
        _, timeline = short_night
        refused = _with_policy(timeline, dispatch_lead_seconds=-1.0)
        with pytest.raises(ValueError, match="dispatch_lead_seconds") as sheet_error:
            dispatch_sheet(refused)
        with pytest.raises(ValueError, match="dispatch_lead_seconds") as record_error:
            dispatch_records(refused)
        assert str(record_error.value) == str(sheet_error.value)

    def test_a_policy_key_the_library_does_not_know_is_refused_by_name(self, short_night):
        """An unknown policy or tuning key is refused by name; the three defaulted keys may go."""
        _, timeline = short_night
        tuning = read_calibration_night_metadata(timeline)["policy"]["tuning"]
        newer = _with_policy(timeline, later_key=1.0, tuning={**tuning, "later_tuning_key": True})
        for render in (dispatch_sheet, dispatch_records):
            with pytest.raises(ValueError, match="know: later_key, tuning.later_tuning_key$"):
                render(newer)
        older = _with_policy(
            timeline,
            drop=("use_table_throw", "dispatch_lead_seconds"),
            tuning={k: v for k, v in tuning.items() if k != "skydip_after_retune"},
        )
        policy = NightContext.from_timeline(older).policy
        assert (policy.use_table_throw, policy.dispatch_lead_seconds) == (True, 0.0)
        assert policy.tuning.skydip_after_retune is False


def _at(seconds, t0="2026-09-11T06:30:00"):
    return Time(t0, scale="utc") + TimeDelta(seconds, format="sec")


def _pass_block(site, start, body, index, n_passes, scan_index, **extra):
    params = {"body": body, "el_bore": 50.0, "mode": "rising", "az_speed": 1.5}
    params.update(pass_index=index, n_passes=n_passes)
    return TimelineBlock.calibration(
        "planet_cal",
        start,
        600.0,
        330.0,
        50.0,
        scan_index,
        target=body,
        az_end=338.0,
        scan_params=params,
        t0_scan=start.isot,
        extra_metadata={
            "transition": {"wrap": 334.0},
            "n_legs": 10,
            "science_fraction": 0.4,
            **extra,
        },
    )


def _night(short_night, blocks, end):
    """Return a timeline of hand-built blocks carrying the short night's metadata."""
    site, planned = short_night
    return ObservingTimeline(
        blocks,
        site,
        blocks[0].t_start,
        end,
        OverheadModel(),
        CalibrationPolicy(),
        metadata=dict(planned.metadata),
    )


def _hand_built_night(short_night):
    """Build a night the planner would not produce.

    A two-pass visit with a slew between its passes, and two escape slews.
    """
    site, _ = short_night
    blocks = [
        TimelineBlock.slew(_at(0), 60.0, 180.0, 334.0, 50.0, 0, patch_name="slew_to_saturn"),
        TimelineBlock.calibration(
            "retune",
            _at(60),
            300.0,
            334.0,
            50.0,
            0,
            extra_metadata={"operation": "find_detectors"},
        ),
        TimelineBlock.calibration("retune", _at(360), 300.0, 334.0, 50.0, 0),
        TimelineBlock.idle(_at(660), 30.0, 334.0, 50.0, 0, reason=_WAITING),
        _pass_block(site, _at(690), "saturn", 0, 2, 1),
        TimelineBlock.slew(_at(1290), 20.0, 338.0, 334.0, 50.0, 1, patch_name="slew_to_saturn"),
        TimelineBlock.idle(_at(1310), 100.0, 334.0, 50.0, 1, reason=_WAITING),
        _pass_block(site, _at(1410), "saturn", 1, 2, 1),
        TimelineBlock.idle(_at(2010), 300.0, 334.0, 50.0, 2, reason="nothing_available"),
        TimelineBlock.slew(_at(2310), 40.0, 334.0, 200.0, 40.0, 2, patch_name="sun_escape"),
        TimelineBlock.idle(_at(2350), 300.0, 200.0, 40.0, 2, reason="nothing_available"),
        TimelineBlock.slew(_at(2650), 60.0, 200.0, 100.0, 50.0, 3, patch_name="slew_to_uranus"),
        TimelineBlock.calibration("retune", _at(2710), 300.0, 100.0, 50.0, 3),
        _pass_block(site, _at(3010), "uranus", 0, 1, 3),
        TimelineBlock.slew(_at(3610), 40.0, 100.0, 250.0, 45.0, 4, patch_name="sun_escape"),
    ]
    return _night(short_night, blocks, _at(3650))


class TestHandBuiltNights:
    """The grouping rules, on nights the planner would not produce."""

    def test_a_second_pass_joins_its_visit_and_escapes_stand_alone(self, short_night):
        record = dispatch_records(_hand_built_night(short_night), source="hand")
        visits = record["visits"]
        assert [(v["body"], len(v["passes"])) for v in visits] == [
            ("saturn", 2),
            (None, 0),
            ("uranus", 1),
            (None, 0),
        ]
        saturn, escape, uranus, trailing = visits
        # The slew between the two passes joins the open visit, after its own slew.
        assert [(s["name"], s["t_start_utc"]) for s in saturn["slews"]] == [
            ("slew_to_saturn", "2026-09-11 06:30:00"),
            ("slew_to_saturn", "2026-09-11 06:51:30"),
        ]
        assert [op["operation"] for op in saturn["detector_operations"]] == [
            "find_detectors",
            "retune",
        ]
        assert [p["slack_s"] for p in saturn["passes"]] == [30.0, 100.0]
        assert [p["scan_params"]["pass_index"] for p in saturn["passes"]] == [0, 1]
        assert saturn["passes"][0]["unforwarded_keys"] == ["az_speed"]
        assert saturn["passes"][0]["wrap_az"] == 334.0
        # Without search_start the pass hands over its own start.
        first_saturn = saturn["passes"][0]
        assert (
            first_saturn["scheduled_t0_unix"]
            == first_saturn["t_start_unix"]
            == round(float(_at(690).unix), 3)
        )
        assert [s["name"] for s in escape["slews"]] == ["sun_escape"]
        assert escape["slews"][0]["el_to"] == 40.0
        assert escape["detector_operations"] == []
        assert uranus["passes"][0]["slack_s"] == 0.0
        assert [s["name"] for s in trailing["slews"]] == ["sun_escape"]
        assert [(i["t_start_utc"], i["duration_s"], i["reason"]) for i in record["idle"]] == [
            ("2026-09-11 07:03:30", 300.0, "nothing_available"),
            ("2026-09-11 07:09:10", 300.0, "nothing_available"),
        ]
        assert record["source"] == "hand"
        assert record["night_start_utc"] == "2026-09-11 06:30:00"
        assert record["night_end_utc"] == "2026-09-11 07:30:50"
        _assert_json_native(record)

    def test_every_record_has_its_keys_in_order(self, short_night):
        """The keys of each record are the contract's, in its order, as the types declare them."""
        record = dispatch_records(_hand_built_night(short_night))
        assert list(record) == NIGHT_KEYS
        for visit in record["visits"]:
            assert list(visit) == VISIT_KEYS
            for slew in visit["slews"]:
                assert list(slew) == SLEW_KEYS
            for operation in visit["detector_operations"]:
                assert list(operation) == OPERATION_KEYS
            for record_pass in visit["passes"]:
                assert list(record_pass) == PASS_KEYS
        for period in record["idle"]:
            assert list(period) == IDLE_KEYS
        for record_type, keys in [
            (NightRecord, NIGHT_KEYS),
            (VisitRecord, VISIT_KEYS),
            (SlewRecord, SLEW_KEYS),
            (OperationRecord, OPERATION_KEYS),
            (PassRecord, PASS_KEYS),
            (IdleRecord, IDLE_KEYS),
        ]:
            assert list(record_type.__annotations__) == keys, record_type.__name__

    def test_a_waiting_idle_no_pass_follows_is_listed(self, short_night):
        """It is listed with its reason, and the moves before it form their own record."""
        site, _ = short_night
        blocks = [
            TimelineBlock.slew(_at(0), 60.0, 180.0, 334.0, 50.0, 0, patch_name="slew_to_saturn"),
            TimelineBlock.calibration("retune", _at(60), 300.0, 334.0, 50.0, 0),
            TimelineBlock.idle(_at(360), 30.0, 334.0, 50.0, 0, reason=_WAITING),
            TimelineBlock.slew(_at(390), 60.0, 334.0, 100.0, 50.0, 1, patch_name="slew_to_uranus"),
            TimelineBlock.calibration("retune", _at(450), 300.0, 100.0, 50.0, 1),
            TimelineBlock.idle(_at(750), 30.0, 100.0, 50.0, 1, reason=_WAITING),
            _pass_block(site, _at(780), "uranus", 0, 1, 1),
            TimelineBlock.idle(_at(1380), 120.0, 338.0, 50.0, 2, reason=_WAITING),
        ]
        record = dispatch_records(_night(short_night, blocks, _at(1500)))
        assert [
            (v["body"], [s["name"] for s in v["slews"]], len(v["passes"])) for v in record["visits"]
        ] == [(None, ["slew_to_saturn"], 0), ("uranus", ["slew_to_uranus"], 1)]
        assert record["visits"][1]["passes"][0]["slack_s"] == 30.0
        assert [(i["t_start_utc"], i["duration_s"], i["reason"]) for i in record["idle"]] == [
            ("2026-09-11 06:36:00", 30.0, _WAITING),
            ("2026-09-11 06:53:00", 120.0, _WAITING),
        ]

    @pytest.mark.parametrize("search_start", [True, False], ids=["search-start", "start-only"])
    def test_the_record_is_the_same_from_ecsv_on_a_leap_second_day(
        self, leap_second_day_night, search_start, tmp_path
    ):
        """The record reads the stored times, so it does not change.

        On 2016-12-31 the Unix time of a start held in memory and of the
        millisecond the file stores differ in the third decimal, and the
        night's 2.9996 s slew is 3.000 s between the stored milliseconds, a
        duration on the sheet's rounding edge.
        """
        night = leap_second_day_night(search_start=search_start)
        start = night.blocks[-1].t_start
        assert round(float(start.unix), 3) != round(float(Time(start.iso, scale="utc").unix), 3)
        path = tmp_path / "night.ecsv"
        write_timeline(night, path)
        back = read_timeline(path)
        record = dispatch_records(night)
        assert dispatch_records(back) == record
        (visit,) = record["visits"]
        assert visit["slews"][0]["duration_s"] == 3.0
        (record_pass,) = visit["passes"]
        assert record_pass["slack_s"] == 240.001
        if not search_start:
            stored = Time(night.blocks[-1].t_start.iso, scale="utc")
            assert record_pass["scheduled_t0_unix"] == round(float(stored.unix), 3)

    def test_a_night_without_blocks(self, short_night):
        """No visits and no idle periods, and neither a start nor an end."""
        site, planned = short_night
        night = ObservingTimeline(
            [],
            site,
            _at(0),
            _at(0),
            OverheadModel(),
            CalibrationPolicy(),
            metadata=dict(planned.metadata),
        )
        record = dispatch_records(night)
        assert list(record) == NIGHT_KEYS
        assert record["visits"] == record["idle"] == []
        assert record["night_start_utc"] is record["night_end_utc"] is None
        _assert_json_native(record)

    def test_a_malformed_search_start_is_a_schema_error(self, short_night):
        site, _ = short_night
        blocks = [_pass_block(site, _at(0), "saturn", 0, 1, 0, search_start=[2461294.5])]
        night = _night(short_night, blocks, _at(600))
        with pytest.raises(ScanParamsSchemaError, match="search_start"):
            dispatch_records(night)
        with pytest.raises(ScanParamsSchemaError, match="search_start"):
            dispatch_sheet(night)
