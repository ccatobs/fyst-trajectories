"""Simulated nights shared by the simulator tests, each built once per session.

A night is generated the first time a test asks for it and kept for the rest
of the run, together with its rebuilt trajectories, so the tests that pin a
night and the guards that check every block of it pay for one build. The
nights are the inputs of the pages and tests that pin them:

* ``regression``: the 8-hour two-patch night of ``test_regression.py``;
* ``quickstart``: the night of ``docs/overhead_quickstart.rst``;
* ``august``: one constant-elevation patch over both halves of its transit
  (``test_ce_corridor.py``);
* ``pong_and_daisy``: the pong and daisy fields of the five-patch day;
* ``zero_overhead``: a constant-elevation patch with every calibration and
  retune duration zero and a 30 s tick, whose visit is admitted minutes
  before its pass opens;
* ``default_overhead``: the same patch and tick with the default overheads,
  whose startup calibrations hold the visit back past its pass's opening;
* ``planet_scan``: Deep56 with every planet calibration planned as a
  multi-pass source-CES sequence, due every two hours;
* ``time_based_retune``: Deep56 with a retune due every 6970 s and a 300 s
  tick, whose retune falls due between the tick that admits the visit and
  its pass opening;
* ``wide_slow_ce``: the August field 34 deg wide and swept at 0.5 deg/s,
  whose pass, a whole number of legs, ends after its closing crossing;
* ``zenith``: a pong field whose centre transits next to the zenith, where
  its azimuth speed exceeds what the Go TCS accepts;
* ``zenith_daisy``: the same field scanned as a daisy, whose full-length
  subscan reaches past the Go TCS azimuth speed before its transit while a
  shorter one does not;
* ``zero_settle``: the pong and daisy fields with every calibration,
  retune and settle duration zero and a 30 s tick;
* ``pong_and_daisy_daytime``: two hours of the pong and daisy day around
  local noon, when the Sun is high and moves into poses the telescope holds;
* ``lsa_window``: Deep56 scanned in the sidereal window (330, 350) deg;
* ``ce_offset``: Deep56 scanned with the bay ``IM6`` at el 50, so the
  boresight scans 1.66 deg higher;
* ``pong_and_daisy_daytime_offset``: the daytime night with the pong given
  ``IM3`` and the daisy ``IM5`` (the pong's field is below the horizon then,
  so the daisy is what it scans);
* ``lsa_window_offset``: the sidereal-window Deep56 scanned with ``IM2``;
* ``pong_and_daisy_offset``: the pong with ``IM3`` and the daisy with
  ``IM5`` over two and a half hours of the night, both fields up;
* ``window_rising``: Deep56 swept at 0.5 deg/s over the azimuth window
  (-252, -242) deg with the bay ``IM6``, a window below the azimuth limit
  that the planner sweeps a whole turn up, at (108, 118) deg, east of the
  meridian, inside the 104.9 to 120.6 deg the field's own rising pass would
  sweep;
* ``window_setting_day``: Deep56 swept at 0.5 deg/s over the window
  (242, 252) deg, west of the meridian, by day;
* ``window_cut``: the same window and day with a retune every two
  hours and subscans of at most 1377 s, so the visit is cut into blocks
  with no retune between them before its retune falls due.

Beside them, :func:`short_calibration_night` is the calibration-night
planner's hour of Saturn and Uranus.

The returned timelines and trajectories are shared: a test reads them and
never changes them.
"""

from __future__ import annotations

import dataclasses
import functools
import tempfile
import warnings
from pathlib import Path

from fyst_trajectories import get_fyst_site, get_primecam_offset
from fyst_trajectories.exceptions import AccelerationLimitWarning, PointingWarning
from fyst_trajectories.overhead import (
    CalibrationPolicy,
    ObservingPatch,
    ObservingTimeline,
    OverheadModel,
    generate_timeline,
    plan_calibration_night,
    read_timeline,
    schedule_to_trajectories,
    write_timeline,
)

#: The 40 x 10 deg constant-elevation field of the regression and quickstart
#: nights.
DEEP56 = ObservingPatch(
    name="Deep56",
    ra_center=24.0,
    dec_center=-32.0,
    width=40.0,
    height=10.0,
    scan_type="constant_el",
    velocity=1.0,
    elevation=50.0,
)

#: The constant-elevation field of the August nights.
AUGUST_CES = ObservingPatch(
    name="AugustCES",
    ra_center=330.0,
    dec_center=-50.0,
    width=30.0,
    height=8.0,
    scan_type="constant_el",
    velocity=1.0,
    elevation=50.0,
)

#: The same field as a patch that records its sampling and turnaround rate.
CES = ObservingPatch(
    name="CES",
    ra_center=330.0,
    dec_center=-50.0,
    width=30.0,
    height=8.0,
    scan_type="constant_el",
    velocity=1.0,
    elevation=50.0,
    scan_params={"timestep": 0.1, "az_accel": 1.0},
)

#: The same field 34 deg wide, swept at 0.5 deg/s.
WIDE_SLOW_CES = dataclasses.replace(CES, name="WideSlowCES", width=34.0, velocity=0.5)

#: Every calibration and retune duration zero; the slews keep their settle time.
ZERO_OVERHEAD = dataclasses.replace(
    OverheadModel(),
    retune_duration=0.0,
    pointing_cal_duration=0.0,
    focus_duration=0.0,
    skydip_duration=0.0,
    planet_cal_duration=0.0,
    beam_map_duration=0.0,
)

#: Every calibration, retune and settle duration zero.
ZERO_SETTLE = dataclasses.replace(ZERO_OVERHEAD, settle_time=0.0)

#: The calibration cadences of the regression night.
REGRESSION_POLICY = CalibrationPolicy(
    retune_cadence=0.0,
    pointing_cadence=3600.0,
    focus_cadence=7200.0,
    skydip_cadence=10800.0,
    planet_cal_cadence=43200.0,
)


#: Deep56 scanned in the sidereal window (330, 350) deg.
DEEP56_LSA = dataclasses.replace(
    DEEP56, name="Deep56LSA", scan_params={"lsa_window": (330.0, 350.0)}
)


def _pattern(name, ra, dec, size, scan_type, detector_offset=None):
    return ObservingPatch(
        name=name,
        ra_center=ra,
        dec_center=dec,
        width=size,
        height=size,
        scan_type=scan_type,
        velocity=0.5,
        detector_offset=detector_offset,
    )


#: The pong and the daisy of the pong and daisy day, each with no offset or a ring bay.
F1 = _pattern("F1", 150.0, 2.2, 4.0, "pong")
F3 = _pattern("F3", 200.0, -20.0, 6.0, "daisy")
F1_IM3 = _pattern("F1", 150.0, 2.2, 4.0, "pong", get_primecam_offset("IM3"))
F3_IM5 = _pattern("F3", 200.0, -20.0, 6.0, "daisy", get_primecam_offset("IM5"))

#: Deep56 and its sidereal-window form, each scanned with a ring bay.
DEEP56_IM6 = dataclasses.replace(
    DEEP56, name="Deep56IM6", detector_offset=get_primecam_offset("IM6")
)
DEEP56_LSA_IM2 = dataclasses.replace(
    DEEP56_LSA, name="Deep56LSAIM2", detector_offset=get_primecam_offset("IM2")
)

#: Deep56 swept at 0.5 deg/s over an azimuth window: rising, with the bay
#: IM6, written below the azimuth limit, a whole turn under where it is
#: swept; and setting.
DEEP56_WINDOW_RISING = dataclasses.replace(
    DEEP56,
    name="Deep56WindowIM6",
    velocity=0.5,
    detector_offset=get_primecam_offset("IM6"),
    scan_params={"az_min": -252.0, "az_max": -242.0},
)
DEEP56_WINDOW_SETTING = dataclasses.replace(
    DEEP56, name="Deep56Window", velocity=0.5, scan_params={"az_min": 242.0, "az_max": 252.0}
)


#: Each night: patches, window and the keyword arguments of generate_timeline.
NIGHTS = {
    "regression": (
        [DEEP56, _pattern("COSMOS", 150.0, 2.2, 4.0, "pong")],
        "2026-06-15T02:00:00",
        "2026-06-15T10:00:00",
        {"overhead_model": OverheadModel(), "calibration_policy": REGRESSION_POLICY},
    ),
    "quickstart": (
        [DEEP56, _pattern("Wide01", 180.0, -30.0, 4.0, "pong")],
        "2026-06-15T02:00:00",
        "2026-06-15T10:00:00",
        {},
    ),
    "august": ([AUGUST_CES], "2026-08-01T23:00:00", "2026-08-02T07:00:00", {}),
    "pong_and_daisy": (
        [F1, F3],
        "2026-12-15T05:00:00",
        "2026-12-15T17:00:00",
        {"calibration_policy": CalibrationPolicy(retune_cadence=0.0)},
    ),
    "zero_overhead": (
        [CES],
        "2026-08-02T01:25:00",
        "2026-08-02T04:25:00",
        {"overhead_model": ZERO_OVERHEAD, "time_step": 30.0},
    ),
    "default_overhead": (
        [CES],
        "2026-08-02T01:25:00",
        "2026-08-02T04:25:00",
        {"time_step": 30.0},
    ),
    "planet_scan": (
        [DEEP56],
        "2026-09-10T00:00:00",
        "2026-09-10T10:00:00",
        {"calibration_policy": CalibrationPolicy(planet_cal_scan=True, planet_cal_cadence=7200.0)},
    ),
    "time_based_retune": (
        [DEEP56],
        "2026-06-15T06:01:40",
        "2026-06-15T09:30:00",
        {"calibration_policy": CalibrationPolicy(retune_cadence=6970.0), "time_step": 300.0},
    ),
    "wide_slow_ce": ([WIDE_SLOW_CES], "2026-08-02T01:00:00", "2026-08-02T05:30:00", {}),
    "zenith": (
        [_pattern("ZenP", 140.0, -23.0, 4.0, "pong")],
        "2026-03-01T01:30:00",
        "2026-03-01T03:30:00",
        {},
    ),
    "zenith_daisy": (
        [_pattern("ZenD", 140.0, -23.0, 4.0, "daisy")],
        "2026-03-01T01:30:00",
        "2026-03-01T03:30:00",
        {},
    ),
    "zero_settle": (
        [F1, F3],
        "2026-12-15T09:30:00",
        "2026-12-15T11:00:00",
        {"overhead_model": ZERO_SETTLE, "time_step": 30.0},
    ),
    "pong_and_daisy_daytime": (
        [F1, F3],
        "2026-12-15T16:00:00",
        "2026-12-15T18:00:00",
        {"calibration_policy": CalibrationPolicy(retune_cadence=0.0)},
    ),
    "lsa_window": ([DEEP56_LSA], "2026-06-15T08:00:00", "2026-06-15T10:30:00", {}),
    "ce_offset": ([DEEP56_IM6], "2026-06-15T07:00:00", "2026-06-15T09:30:00", {}),
    "pong_and_daisy_daytime_offset": (
        [F1_IM3, F3_IM5],
        "2026-12-15T16:00:00",
        "2026-12-15T18:00:00",
        {"calibration_policy": CalibrationPolicy(retune_cadence=0.0)},
    ),
    "lsa_window_offset": ([DEEP56_LSA_IM2], "2026-06-15T08:00:00", "2026-06-15T10:30:00", {}),
    "pong_and_daisy_offset": (
        [F1_IM3, F3_IM5],
        "2026-12-15T08:30:00",
        "2026-12-15T11:00:00",
        {"calibration_policy": CalibrationPolicy(retune_cadence=0.0)},
    ),
    "window_rising": (
        [DEEP56_WINDOW_RISING],
        "2026-06-15T07:20:00",
        "2026-06-15T09:50:00",
        {},
    ),
    "window_setting_day": (
        [DEEP56_WINDOW_SETTING],
        "2026-06-15T13:20:00",
        "2026-06-15T15:50:00",
        {},
    ),
    "window_cut": (
        [DEEP56_WINDOW_SETTING],
        "2026-06-15T13:20:00",
        "2026-06-15T15:50:00",
        {
            "overhead_model": OverheadModel(max_scan_duration=1377.0),
            "calibration_policy": CalibrationPolicy(retune_cadence=7200.0),
        },
    ),
}

#: Each night with a detector offset, and the night with none that observes
#: the same fields over the same window.
OFFSET_NIGHTS = {
    "ce_offset": "ce_offset_without",
    "pong_and_daisy_daytime_offset": "pong_and_daisy_daytime",
    "lsa_window_offset": "lsa_window",
    "pong_and_daisy_offset": "pong_and_daisy_offset_without",
}

#: The nights with no offset that only an offset night is compared with:
#: patches, window and the keyword arguments of generate_timeline.
COMPARISON_NIGHTS = {
    "ce_offset_without": (
        [dataclasses.replace(DEEP56_IM6, detector_offset=None)],
        "2026-06-15T07:00:00",
        "2026-06-15T09:30:00",
        {},
    ),
    "pong_and_daisy_offset_without": (
        [F1, F3],
        "2026-12-15T08:30:00",
        "2026-12-15T11:00:00",
        {"calibration_policy": CalibrationPolicy(retune_cadence=0.0)},
    ),
}


@functools.cache
def night(name: str) -> ObservingTimeline:
    """Return the named night of :data:`NIGHTS` or :data:`COMPARISON_NIGHTS`, generated once."""
    patches, start, end, kwargs = NIGHTS[name] if name in NIGHTS else COMPARISON_NIGHTS[name]
    return generate_timeline(
        patches=patches, site=get_fyst_site(), start_time=start, end_time=end, **kwargs
    )


@functools.cache
def rebuilt(name: str) -> tuple:
    """Return the named night's rebuilt blocks, ``schedule_to_trajectories(science_only=False)``.

    The daisy near its transit on the pong and daisy day scans at el 78; its
    rebuild's high-elevation and azimuth-acceleration advisories describe
    the scan, so they are set aside here by message.
    """
    return _rebuild(night(name))


@functools.cache
def rebuilt_from_file(name: str) -> tuple:
    """Return the named night's blocks as :func:`rebuilt` does, after an ECSV round trip."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"{name}.ecsv"
        write_timeline(night(name), path)
        return _rebuild(read_timeline(path))


def _rebuild(timeline: ObservingTimeline) -> tuple:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="High elevation reduces on-sky azimuth speed",
            category=PointingWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Trajectory azimuth acceleration",
            category=AccelerationLimitWarning,
        )
        return tuple(schedule_to_trajectories(timeline, science_only=False))


@functools.cache
def short_calibration_night() -> ObservingTimeline:
    """Return the calibration-night planner's hour of Saturn and Uranus on 2026-09-11.

    Planned on the first call with the planner's defaults (three Saturn
    passes); the calibration-night tests and the held-pose guard share it.
    """
    return plan_calibration_night(
        ["saturn", "uranus"], get_fyst_site(), "2026-09-11T06:30:00", "2026-09-11T07:30:00"
    )
