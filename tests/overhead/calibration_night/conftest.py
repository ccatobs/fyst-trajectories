"""Shared fixtures: one planned hour-long night for the reporting and regression tests."""

import warnings

import pytest

from fyst_trajectories import get_fyst_site
from fyst_trajectories.overhead import plan_calibration_night


@pytest.fixture(scope="package")
def short_night():
    """Plan a one-hour two-body night on 2026-09-11 (three Saturn passes)."""
    site = get_fyst_site()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        timeline = plan_calibration_night(
            ["saturn", "uranus"], site, "2026-09-11T06:30:00", "2026-09-11T07:30:00"
        )
    return site, timeline
