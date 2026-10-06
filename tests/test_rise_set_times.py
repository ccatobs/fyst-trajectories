"""Tests for rise/set time calculations.

Covers the happy path, circumpolar and never-visible sources, a custom horizon,
and the search-window and step-size parameters.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time, TimeDelta


class TestRiseSetTimes:
    """Tests for rise/set time calculations."""

    def test_basic_rise_set_calculation(self, coordinates):
        """Test basic rise/set time calculation returns sensible results."""
        # Use a source that definitely rises and sets from Chile
        # RA 6h (90 deg), Dec +20 (northern source, will rise and set)
        # From lat -23 this source has max elevation ~47 deg, so it
        # definitely rises and sets within 48 hours.
        ra = 90.0
        dec = 20.0
        obstime = Time("2026-06-15T12:00:00", scale="utc")

        rise, set_ = coordinates.get_rise_set_times(
            ra,
            dec,
            start_time=obstime,
            horizon=0.0,
            max_search_hours=48.0,
            step_hours=0.5,
        )

        # This source should rise and set from a southern site
        assert rise is not None, "Expected a rise time for RA=90, Dec=+20 from Chile"
        assert isinstance(rise, Time)
        assert rise > obstime  # Rise should be after start time
        assert isinstance(set_, Time)
        assert set_ > rise  # Set should be after rise

    def test_circumpolar_source_returns_none(self, coordinates):
        """Test that circumpolar sources return None for both times.

        From Chile (latitude ~-23), a source at dec -80 is circumpolar
        (always above the horizon).
        """
        ra = 180.0
        dec = -80.0  # Far south, circumpolar from Chile
        obstime = Time("2026-06-15T00:00:00", scale="utc")

        rise, set_ = coordinates.get_rise_set_times(
            ra,
            dec,
            start_time=obstime,
            horizon=0.0,
            max_search_hours=24.0,
            step_hours=0.5,
        )

        assert rise is None, "Circumpolar source should not have a rise time"
        assert set_ is None, "Circumpolar source should not have a set time"

    def test_circumpolar_source_no_rise_set(self, coordinates):
        """Circumpolar source (Dec=-70 at FYST) returns (None, None)."""
        ra, dec = 180.0, -70.0
        start_time = Time("2026-06-15T00:00:00", scale="utc")

        rise, set_ = coordinates.get_rise_set_times(
            ra,
            dec,
            start_time=start_time,
            horizon=0.0,
            max_search_hours=24.0,
            step_hours=0.1,
        )

        # Dec=-70 is circumpolar from FYST: upper culmination is near 90-|lat-dec| = 43 deg
        # and lower culmination is only ~3 deg above the horizon, so it never sets below
        # horizon=0.
        assert rise is None and set_ is None, (
            f"Expected (None, None) for circumpolar source, got rise={rise}, set={set_}"
        )

    def test_never_visible_source_returns_none(self, coordinates):
        """Test that sources never visible return None for both times.

        From Chile (latitude ~-23), a source at dec +80 (far north)
        never rises above the horizon.
        """
        ra = 0.0
        dec = 80.0  # Far north, never visible from Chile
        obstime = Time("2026-06-15T00:00:00", scale="utc")

        rise, set_ = coordinates.get_rise_set_times(
            ra,
            dec,
            start_time=obstime,
            horizon=0.0,
            max_search_hours=24.0,
            step_hours=0.5,
        )

        # Source at dec +80 from lat -23 has max elevation ~-23+90-80 = -13
        # so it should never rise above horizon
        assert rise is None, "Source at dec +80 should never rise from Chile"
        assert set_ is None, "Source at dec +80 should never set from Chile"

    def test_custom_horizon(self, coordinates):
        """Test that higher horizon produces shorter visible window."""
        ra = 90.0
        dec = 0.0  # Equatorial source
        obstime = Time("2026-06-15T12:00:00", scale="utc")

        rise_0, set_0 = coordinates.get_rise_set_times(
            ra,
            dec,
            start_time=obstime,
            horizon=0.0,
            max_search_hours=48.0,
            step_hours=0.5,
        )
        rise_20, set_20 = coordinates.get_rise_set_times(
            ra,
            dec,
            start_time=obstime,
            horizon=20.0,
            max_search_hours=48.0,
            step_hours=0.5,
        )

        # An equatorial source from Chile (lat -23) has max elevation ~67 deg,
        # so it should rise and set for both horizon=0 and horizon=20.
        assert rise_0 is not None, "Expected rise at horizon=0 for RA=90, Dec=0"
        assert set_0 is not None, "Expected set at horizon=0 for RA=90, Dec=0"
        assert rise_20 is not None, "Expected rise at horizon=20 for RA=90, Dec=0"
        assert set_20 is not None, "Expected set at horizon=20 for RA=90, Dec=0"

        window_0 = (set_0 - rise_0).to_value("hour")
        window_20 = (set_20 - rise_20).to_value("hour")
        assert window_20 < window_0, (
            f"Higher horizon should give shorter visible window: "
            f"{window_20:.2f}h >= {window_0:.2f}h"
        )

    def test_max_search_hours_parameter(self, coordinates):
        """Test that max_search_hours limits the search window."""
        # Choose a source at RA=270 (18h), Dec=+10. From Chile at this start time,
        # the source is below the horizon and rises about 23 hours later.
        ra = 270.0
        dec = 10.0
        obstime = Time("2026-06-15T00:00:00", scale="utc")

        # With a 1-hour search window, should NOT find rise (it's hours away)
        rise_short, _set_short = coordinates.get_rise_set_times(
            ra,
            dec,
            start_time=obstime,
            horizon=0.0,
            max_search_hours=1.0,
            step_hours=0.5,
        )

        # With a 48-hour window, should find the rise
        rise_long, _set_long = coordinates.get_rise_set_times(
            ra,
            dec,
            start_time=obstime,
            horizon=0.0,
            max_search_hours=48.0,
            step_hours=0.5,
        )

        # Short search should miss it, long search should find it
        assert rise_short is None, "Expected no rise in 1-hour window"
        assert rise_long is not None, "Expected rise in 48-hour window"

    def test_step_hours_parameter(self, coordinates):
        """Test that step_hours affects calculation precision."""
        # Use a well-behaved equatorial source that rises cleanly
        ra = 90.0
        dec = 0.0  # Equatorial source, rises and sets clearly from Chile
        obstime = Time("2026-06-15T00:00:00", scale="utc")

        # With different step sizes, results should be very similar
        rise_coarse, _ = coordinates.get_rise_set_times(
            ra,
            dec,
            start_time=obstime,
            horizon=0.0,
            max_search_hours=48.0,
            step_hours=1.0,
        )
        rise_fine, _ = coordinates.get_rise_set_times(
            ra,
            dec,
            start_time=obstime,
            horizon=0.0,
            max_search_hours=48.0,
            step_hours=0.1,
        )

        assert rise_coarse is not None, "Expected rise with coarse step"
        assert rise_fine is not None, "Expected rise with fine step"

        # Linear interpolation near the horizon: the 1 h and 0.1 h grids agree
        # to a fraction of a second at this declination (measured 0.25 s).
        diff_hours = abs((rise_fine - rise_coarse).to_value("hour"))
        assert diff_hours < 10.0 / 3600.0, (
            f"Coarse and fine results differ by {diff_hours * 3600:.2f} s, expected < 10 s"
        )

    def test_set_time_after_rise_time(self, coordinates):
        """Test that set time is always after rise time when both exist."""
        # Equatorial source from Chile, definitely rises and sets.
        ra = 90.0
        dec = 0.0
        obstime = Time("2026-06-15T12:00:00", scale="utc")

        rise, set_ = coordinates.get_rise_set_times(
            ra,
            dec,
            start_time=obstime,
            horizon=0.0,
            max_search_hours=48.0,
            step_hours=0.5,
        )

        assert rise is not None, "Expected a rise time for RA=90, Dec=0"
        assert set_ is not None, "Expected a set time for RA=90, Dec=0"
        assert set_ > rise, "Set time must be after rise time"

    def test_returned_times_sit_on_the_horizon(self, coordinates):
        """The source's elevation at the returned rise and set equals the horizon.

        The crossing is linearly interpolated between grid samples, so at a
        0.1 h step it lands within 1e-3 deg of the horizon.
        """
        ra, dec = 90.0, 0.0
        obstime = Time("2026-06-15T12:00:00", scale="utc")
        rise, set_ = coordinates.get_rise_set_times(
            ra, dec, start_time=obstime, horizon=20.0, max_search_hours=48.0, step_hours=0.1
        )
        for event in (rise, set_):
            _, el = coordinates.radec_to_altaz(ra, dec, event)
            assert float(el) == pytest.approx(20.0, abs=1e-3)


# The Crab at a 20 deg floor rises at about 02:49:58 UTC on 2026-11-15.
_CRAB_RA, _CRAB_DEC, _FLOOR = 83.633, 22.014, 20.0


def _fine_grid_rise(coordinates) -> Time:
    """Return the Crab's rise through the floor, interpolated on a 1 s grid."""
    dt = np.arange(0.0, 3600.0, 1.0)
    grid = Time("2026-11-15T02:00:00", scale="utc") + TimeDelta(dt, format="sec")
    _, el = coordinates.radec_to_altaz(
        np.full(dt.size, _CRAB_RA), np.full(dt.size, _CRAB_DEC), grid
    )
    i = int(np.flatnonzero((el[:-1] < _FLOOR) & (el[1:] >= _FLOOR))[0])
    return grid[i] + (_FLOOR - el[i]) / (el[i + 1] - el[i]) * (grid[i + 1] - grid[i])


class TestSearchSpan:
    """The search covers the whole ``[start_time, start_time + max_search_hours]`` span."""

    @pytest.mark.parametrize(
        ("lead_hours", "max_search_hours", "step_hours"),
        [
            (0.25, 0.3, 0.1),  # int(0.3 / 0.1) is 2: the last tenth of the span
            (1.03, 1.05, 0.1),  # a span that is not a whole number of steps
            (0.03, 0.05, 0.1),  # a span shorter than one step
        ],
    )
    def test_rise_in_the_last_partial_step_is_found(
        self, coordinates, lead_hours, max_search_hours, step_hours
    ):
        """A rise between the last whole step and the end of the span is found."""
        reference = _fine_grid_rise(coordinates)
        start = reference - TimeDelta(lead_hours * 3600.0, format="sec")
        rise, _ = coordinates.get_rise_set_times(
            _CRAB_RA,
            _CRAB_DEC,
            start_time=start,
            horizon=_FLOOR,
            max_search_hours=max_search_hours,
            step_hours=step_hours,
        )
        assert rise is not None
        assert abs((rise - reference).to_value("s")) < 1.0

    def test_set_selection_rule(self, coordinates, monkeypatch):
        """The set is the first one at or after the first rise.

        Synthetic altitudes on the 0.1 h grid of a 0.3 h search: the source
        sets in the first step, which precedes any rise and is ignored, then
        touches the horizon exactly at 0.2 h and falls again, so the rise and
        the set land on the same sample and the set equals the rise.
        """
        start = Time("2026-11-15T00:00:00", scale="utc")
        altitudes = np.array([5.0, -5.0, 0.0, -5.0])

        class _SyntheticSource:
            def __init__(self, *args, **kwargs):
                pass

            def transform_to(self, frame):
                hours = (frame.obstime - start).to_value("hour")
                index = np.rint(hours / 0.1).astype(int)
                return SimpleNamespace(alt=altitudes[index] * u.deg)

        monkeypatch.setattr("fyst_trajectories.coordinates.SkyCoord", _SyntheticSource)
        rise, set_ = coordinates.get_rise_set_times(
            0.0, 0.0, start_time=start, horizon=0.0, max_search_hours=0.3, step_hours=0.1
        )
        assert rise is not None
        assert (rise - start).to_value("s") == pytest.approx(720.0, abs=1e-6)
        assert set_ is not None
        assert set_ == rise
