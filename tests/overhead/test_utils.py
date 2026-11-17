"""Tests for scheduling utilities."""

import dataclasses
import json
import math

import pytest
from astropy.time import Time

from fyst_trajectories.dispatch import estimate_slew_time
from fyst_trajectories.exceptions import PointingWarning
from fyst_trajectories.overhead import ScanParamsSchemaError
from fyst_trajectories.overhead.utils import (
    _dispatch_footprint,
    _is_on_axis_bay,
    _normalize_az,
    _search_start_record,
    _search_start_time,
    _utc_instant,
)
from fyst_trajectories.primecam import PRIMECAM_MODULES

# Off the millisecond grid, so a record rounded anywhere would show.
_INSTANT = Time("2026-09-11T06:46:28.371234567", scale="utc")
_OTHER_SCALES = ["tt", "tai", "tdb", "tcg", "tcb", "ut1", "utc+location"]


def _given_in(scale, site):
    """``_INSTANT`` as a ``Time`` in ``scale``; ``"utc+location"`` carries the site's location."""
    if scale == "utc+location":
        return Time(_INSTANT, location=site.location)
    return getattr(_INSTANT, scale)


class TestUtcInstant:
    """Both planners hold every time as a UTC ``Time`` without a location, with the defaults."""

    def test_a_utc_time_without_a_location_is_held_as_it_is(self):
        assert _utc_instant(_INSTANT) is _INSTANT

    @pytest.mark.parametrize(
        "attributes",
        [{"precision": 0}, {"precision": 6}, {"out_subfmt": "date"}],
        ids=["precision 0", "precision 6", "out_subfmt date"],
    )
    def test_a_utc_time_with_other_output_attributes_is_rebuilt_with_the_defaults(self, attributes):
        """The instant is kept to the bit; its strings are the millisecond ISO ones."""
        given = Time(_INSTANT, **attributes)
        held = _utc_instant(given)
        assert held is not given
        assert (held.jd1, held.jd2) == (given.jd1, given.jd2)
        assert (held.scale, held.location, held.precision, held.out_subfmt) == ("utc", None, 3, "*")
        assert held.iso == "2026-09-11 06:46:28.371"
        assert _utc_instant(held) is held

    def test_a_string_is_read_as_utc(self):
        held = _utc_instant("2026-09-11T06:46:28.371")
        assert (held.scale, held.location, held.isot) == ("utc", None, "2026-09-11T06:46:28.371")

    @pytest.mark.parametrize("scale", _OTHER_SCALES)
    def test_any_other_time_is_held_as_the_same_instant_in_utc(self, scale, site):
        """It is converted to UTC with its location dropped, and then held unchanged."""
        held = _utc_instant(_given_in(scale, site))
        assert (held.scale, held.location) == ("utc", None)
        assert abs((held - _INSTANT).to_value("s")) < 1e-9
        assert _utc_instant(held) is held


class TestSearchStartRecord:
    """A pass's ``search_start`` restores the instant the planner held, through JSON."""

    @pytest.mark.parametrize("scale", ["utc", *_OTHER_SCALES])
    def test_the_record_restores_the_held_instant_to_the_bit(self, scale, site):
        held = _utc_instant(_INSTANT if scale == "utc" else _given_in(scale, site))
        back = _search_start_time(json.loads(json.dumps(_search_start_record(held))))
        assert (back.scale, back.location) == ("utc", None)
        assert (back.jd1, back.jd2) == (held.jd1, held.jd2)

    @pytest.mark.parametrize(
        "record",
        [
            [True, False],
            [2461295.0],
            [2461295.0, -0.2, 0.0],
            [],
            [math.nan, -0.2],
            [2461295.0, math.inf],
            "2026-09-11T06:30:00",
            ["2461295.0", "-0.2"],
            None,
            {"jd1": 2461295.0, "jd2": -0.2},
        ],
        ids=[
            "booleans",
            "one number",
            "three numbers",
            "empty",
            "nan",
            "infinity",
            "iso string",
            "strings",
            "none",
            "dict",
        ],
    )
    def test_any_other_form_is_a_schema_error_naming_the_key(self, record):
        with pytest.raises(ScanParamsSchemaError, match="search_start"):
            _search_start_time(record)


_CENTRE_SPELLINGS = ["c", "C", "center", "Center", "CENTER", "IM0", "im0"]
_RING_BAYS = [name for name in PRIMECAM_MODULES if name != "IM0"]


class TestDispatchFootprint:
    """The one name a recorded pass dict gives each Prime-Cam bay."""

    @pytest.mark.parametrize("tag", _CENTRE_SPELLINGS)
    def test_every_spelling_of_the_centre_is_written_c(self, tag):
        assert _dispatch_footprint(tag) == "c"

    @pytest.mark.parametrize("bay", _RING_BAYS)
    def test_a_ring_bay_in_any_case_is_written_as_its_key(self, bay):
        for tag in (bay, bay.lower(), bay[0] + bay[1].lower() + bay[2:]):
            assert _dispatch_footprint(tag) == bay

    @pytest.mark.parametrize("tag", ["i1", "I6", "IM7", "zzz", "c,IM1", ""])
    def test_an_unknown_or_retired_name_raises(self, tag):
        with pytest.raises(KeyError, match="Unknown PrimeCam module"):
            _dispatch_footprint(tag)


class TestIsOnAxisBay:
    """Whether a footprint name is the on-axis bay ``IM0``."""

    @pytest.mark.parametrize("tag", _CENTRE_SPELLINGS)
    def test_every_spelling_of_the_centre_is_on_axis(self, tag):
        assert _is_on_axis_bay(tag)

    @pytest.mark.parametrize("bay", _RING_BAYS)
    def test_no_ring_bay_is_on_axis(self, bay):
        assert not _is_on_axis_bay(bay)
        assert not _is_on_axis_bay(bay.lower())

    @pytest.mark.parametrize("tag", ["i1", "I6", "IM7", "zzz", "c,IM1", ""])
    def test_an_unknown_or_retired_name_raises(self, tag):
        with pytest.raises(KeyError, match="Unknown PrimeCam module"):
            _is_on_axis_bay(tag)


class TestNormalizeAz:
    """The scalar cable-wrap placement the scheduler and the calibration-night planner share."""

    def test_a_half_turn_tie_keeps_the_window_centre_image(self, site):
        """From a mount at 0 deg, sky 180 has two in-limits images 180 deg away.

        The window-centre image is kept; both price the same move.
        """
        assert _normalize_az(180.0, site, ref=0.0) == 180.0
        assert estimate_slew_time(0.0, 50.0, 180.0, 50.0, site=site) == estimate_slew_time(
            0.0, 50.0, -180.0, 50.0, site=site
        )

    def test_an_azimuth_outside_a_narrow_window_keeps_the_window_centre_image(self, site):
        """On a window narrower than 360 deg an unreachable azimuth is not moved into it.

        Sky 272.5 has no image inside ``[0, 270]``: the helper returns the
        window-centre image, outside the limits, and warns.
        """
        azimuth = dataclasses.replace(site.telescope_limits.azimuth, min=0.0, max=270.0)
        limits = dataclasses.replace(site.telescope_limits, azimuth=azimuth)
        narrow = dataclasses.replace(site, telescope_limits=limits)
        with pytest.warns(PointingWarning, match="exceeds telescope limits"):
            assert _normalize_az(272.5, narrow, ref=0.0) == 272.5
