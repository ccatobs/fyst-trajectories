"""Shared fixtures: a planned hour-long night for the reporting and regression tests.

Beside it, the same hour planned with a skydip after every retune, and a
one-pass night built by hand on a day that ended with a leap second, for the
read-back tests.
"""

import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories import get_fyst_site
from fyst_trajectories.overhead import (
    CalibrationNightPolicy,
    CalibrationPolicy,
    DeferralReason,
    ObservingTimeline,
    OverheadModel,
    TimelineBlock,
    TuningPolicy,
    plan_calibration_night,
)
from fyst_trajectories.overhead.utils import _search_start_record


@pytest.fixture(scope="package")
def short_night():
    """Plan a one-hour two-body night on 2026-09-11 (three Saturn passes)."""
    site = get_fyst_site()
    timeline = plan_calibration_night(
        ["saturn", "uranus"], site, "2026-09-11T06:30:00", "2026-09-11T07:30:00"
    )
    return site, timeline


@pytest.fixture(scope="package")
def short_night_with_dip():
    """Plan the same hour with a skydip after every retune (two Saturn passes)."""
    site = get_fyst_site()
    timeline = plan_calibration_night(
        ["saturn", "uranus"],
        site,
        "2026-09-11T06:30:00",
        "2026-09-11T07:30:00",
        policy=CalibrationNightPolicy(tuning=TuningPolicy(skydip_after_retune=True)),
    )
    return site, timeline


@pytest.fixture(scope="package")
def leap_second_day_night(short_night):
    """Return a factory for a one-pass night on 2016-12-31 whose times sit off the millisecond.

    That day ended with a leap second, and astropy's Unix time spreads its
    86401 s over 86400, so the Unix time of the pass start held in memory
    (06:30:00.001433) and of the millisecond the ECSV file stores differ in
    the third decimal (...799.731 against ...799.730). The slew lasts
    2.9996 s in memory, 0.0 min to one decimal, and the 3.000 s between the
    milliseconds the file stores is 0.1 min: a duration on a rounding edge.
    The factory takes whether the pass block records ``search_start``; the
    night carries the short night's metadata.
    """
    site, planned = short_night
    t0 = Time("2016-12-31T06:20:00", scale="utc")

    def at(seconds):
        return t0 + TimeDelta(seconds, format="sec")

    def build(*, search_start):
        start = at(600.001433)
        extra = {"transition": {"wrap": 334.0}, "n_legs": 10, "science_fraction": 0.4}
        if search_start:
            extra["search_start"] = _search_start_record(start - TimeDelta(18.0004, format="sec"))
        params = {
            "body": "saturn",
            "el_bore": 50.0,
            "mode": "rising",
            "az_speed": 1.5,
            "pass_index": 0,
            "n_passes": 1,
        }
        waiting = str(DeferralReason.WAITING_FOR_PASS)
        blocks = [
            TimelineBlock.slew(
                at(57.0004), 2.9996, 180.0, 334.0, 50.0, 0, patch_name="slew_to_saturn"
            ),
            TimelineBlock.calibration("retune", at(60), 300.0, 334.0, 50.0, 0),
            TimelineBlock.idle(
                at(360), (start - at(360)).to_value("s"), 334.0, 50.0, 0, reason=waiting
            ),
            TimelineBlock.calibration(
                "planet_cal",
                start,
                600.0,
                330.0,
                50.0,
                1,
                target="saturn",
                az_end=338.0,
                scan_params=params,
                t0_scan=start.isot,
                extra_metadata=extra,
            ),
        ]
        return ObservingTimeline(
            blocks,
            site,
            t0,
            start + TimeDelta(600.0, format="sec"),
            OverheadModel(),
            CalibrationPolicy(),
            metadata=dict(planned.metadata),
        )

    return build
