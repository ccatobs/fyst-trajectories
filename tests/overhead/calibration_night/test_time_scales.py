"""A night given as ``Time`` objects of another form is the same night, held in UTC.

The shared one-hour night (``short_night``, planned on ISO strings read as
UTC) is planned again with its window ends given as ``Time`` objects naming
the same instants: in TT, and in UTC with a ``precision`` of 0 or an
``out_subfmt`` of ``"date"``, which change how a ``Time`` prints. The
planner holds every time as a UTC ``Time`` with astropy's defaults, so each
of these nights writes the UTC night's file byte for byte, records its
times, prints its sheet and summary from memory and from the file, resumes
as it does, and every pass rebuilds from its block, bit for bit, as it was
planned.
"""

import warnings

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories.overhead import (
    NightContext,
    NightState,
    dispatch_sheet,
    plan_calibration_night,
    read_calibration_night_metadata,
    read_timeline,
    schedule_to_trajectories,
    summarize_calibration_night,
    write_timeline,
)
from fyst_trajectories.overhead.calibration_night import step

_ARRAYS = ("times", "az", "el", "az_vel", "el_vel", "scan_flag")
# The ECSV stores times to the millisecond.
_ISO_ROUNDING_SEC = 5e-4 + 1e-9
_START, _END = "2026-09-11T06:30:00", "2026-09-11T07:30:00"
# The forms a window end is given in, each naming the instant of an ISO UTC string.
_FORMS = {
    "tt": lambda iso: Time(iso, scale="utc").tt,
    "utc precision 0": lambda iso: Time(iso, scale="utc", precision=0),
    "utc out_subfmt date": lambda iso: Time(iso, scale="utc", out_subfmt="date"),
}


@pytest.fixture(scope="module", params=list(_FORMS))
def given_night(request, short_night):
    """Plan the shared night on window ends of one form, keeping each pass its planner returned."""
    site, _ = short_night
    given = _FORMS[request.param]
    planned = []
    real = step.plan_source_ces_passes

    def recording(*args, **kwargs):
        passes = real(*args, **kwargs)
        planned.extend((kwargs["start_time"], p) for p in passes)
        return passes

    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        patch.setattr(step, "plan_source_ces_passes", recording)
        warnings.simplefilter("ignore")
        timeline = plan_calibration_night(["saturn", "uranus"], site, given(_START), given(_END))
    return timeline, planned


@pytest.fixture(scope="module")
def given_night_file(given_night, tmp_path_factory):
    """Write the night's ECSV file; return its path and the timeline read back from it."""
    path = tmp_path_factory.mktemp("given_night") / "night.ecsv"
    write_timeline(given_night[0], path)
    return path, read_timeline(path)


def _passes(timeline):
    return [b for b in timeline.blocks if "scan_params" in b.metadata]


def _seconds(a, b):
    return abs((a - b).to_value("s"))


def _planned_pass(block, planned):
    """Return the anchor and the pass a block was written from (the last solve wins)."""
    el_bore = block.metadata["scan_params"]["el_bore"]
    return [(a, p) for a, p in planned if p.computed_params["el_bore"] == el_bore][-1]


def test_every_time_is_held_in_utc_at_the_utc_nights_instants(given_night, short_night):
    timeline, _ = given_night
    _, utc_night = short_night
    for t in (timeline.start_time, timeline.end_time):
        assert (t.scale, t.location, t.precision, t.out_subfmt) == ("utc", None, 3, "*")
    assert len(timeline.blocks) == len(utc_night.blocks)
    for block, utc_block in zip(timeline.blocks, utc_night.blocks):
        assert (block.t_start.scale, block.t_stop.scale) == ("utc", "utc")
        assert _seconds(block.t_start, utc_block.t_start) < 1e-6
        assert _seconds(block.t_stop, utc_block.t_stop) < 1e-6


def test_the_file_is_the_utc_nights_byte_for_byte(given_night_file, short_night, tmp_path):
    _, utc_night = short_night
    utc_path = tmp_path / "utc_night.ecsv"
    write_timeline(utc_night, utc_path)
    path, _ = given_night_file
    assert path.read_bytes() == utc_path.read_bytes()


def test_the_recorded_times_are_the_utc_nights(given_night, short_night):
    """``t0_scan`` and every time in the night's record are the UTC night's strings."""
    timeline, _ = given_night
    _, utc_night = short_night
    assert read_calibration_night_metadata(timeline) == read_calibration_night_metadata(utc_night)
    for block, utc_block in zip(_passes(timeline), _passes(utc_night), strict=True):
        assert block.metadata["t0_scan"] == utc_block.metadata["t0_scan"]
        t0_scan = Time(block.metadata["t0_scan"], scale="utc")
        assert _seconds(t0_scan, block.t_start) <= _ISO_ROUNDING_SEC


def test_the_sheet_and_the_summary_are_the_utc_nights(given_night, given_night_file, short_night):
    timeline, _ = given_night
    _, from_file = given_night_file
    _, utc_night = short_night
    sheet = dispatch_sheet(utc_night)
    assert dispatch_sheet(timeline) == sheet
    assert dispatch_sheet(from_file) == sheet
    assert str(summarize_calibration_night(timeline)) == str(summarize_calibration_night(utc_night))


def test_the_file_holds_the_same_instants(given_night, given_night_file):
    timeline, _ = given_night
    _, from_file = given_night_file
    assert _seconds(from_file.start_time, timeline.start_time) <= _ISO_ROUNDING_SEC
    assert _seconds(from_file.end_time, timeline.end_time) <= _ISO_ROUNDING_SEC
    for block, back in zip(timeline.blocks, from_file.blocks, strict=True):
        assert _seconds(back.t_start, block.t_start) <= _ISO_ROUNDING_SEC
        assert _seconds(back.t_stop, block.t_stop) <= _ISO_ROUNDING_SEC


def test_it_resumes_from_its_file_as_the_utc_night_does(given_night_file, short_night):
    """At the end of the first pass, the resumed state is the UTC night's."""
    _, from_file = given_night_file
    _, utc_night = short_night
    ctx, utc_ctx = NightContext.from_timeline(from_file), NightContext.from_timeline(utc_night)
    assert (ctx.targets, ctx.policy) == (utc_ctx.targets, utc_ctx.policy)
    for name in ("start_time", "end_time", "requested_start", "requested_end"):
        assert _seconds(getattr(ctx, name), getattr(utc_ctx, name)) < 1e-3
    t = Time(_passes(utc_night)[0].t_stop.isot, scale="utc")
    state = NightState.from_timeline(from_file, t)
    utc_state = NightState.from_timeline(utc_night, t)
    assert len(state.blocks) == len(utc_state.blocks)
    assert (state.az, state.el, state.scan_counter) == (
        utc_state.az,
        utc_state.el,
        utc_state.scan_counter,
    )
    assert (state.deferred.keys(), state.dropped) == (utc_state.deferred.keys(), utc_state.dropped)


@pytest.mark.parametrize("source", ["memory", "file"])
def test_every_pass_rebuilds_as_planned(given_night, given_night_file, source):
    """The planner searched from a UTC instant, which ``search_start`` restores exactly."""
    timeline, planned = given_night
    rebuilt_from = timeline if source == "memory" else given_night_file[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pairs = schedule_to_trajectories(rebuilt_from, science_only=False)
    assert len(pairs) == 3
    for block, rebuilt in pairs:
        anchor, p = _planned_pass(block, planned)
        assert (anchor.scale, anchor.location) == ("utc", None)
        assert block.metadata["search_start"] == [anchor.jd1, anchor.jd2]
        for name in _ARRAYS:
            assert np.array_equal(getattr(rebuilt.trajectory, name), getattr(p.trajectory, name))
        got, want = rebuilt.trajectory.start_time, p.trajectory.start_time
        assert (got.scale, got.jd1, got.jd2) == (want.scale, want.jd1, want.jd2)
        assert rebuilt.duration == p.duration
        assert rebuilt.computed_params == p.computed_params
