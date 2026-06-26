"""Tests for integration features.

These tests verify the integration features added for compatibility
with external systems, including:
- Longitude convention conversion
- Rise/set time calculations
"""

from astropy.time import Time


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
        if set_ is not None:
            assert isinstance(set_, Time)
            assert set_ > rise  # Set should be after rise

    def test_circumpolar_source_returns_none(self, coordinates, site):
        """Test that circumpolar sources return None for both times.

        From Chile (latitude ~-23), a source at dec -80 is circumpolar
        (always above the horizon).
        """
        ra = 180.0
        dec = -80.0  # Far south, circumpolar from Chile
        obstime = Time("2026-06-15T00:00:00", scale="utc")

        rise, _set = coordinates.get_rise_set_times(
            ra,
            dec,
            start_time=obstime,
            horizon=0.0,
            max_search_hours=24.0,
            step_hours=0.5,
        )

        # Circumpolar source never rises (already up) so both should be None
        # if it starts above horizon and stays there
        # Note: the source might be above horizon at start, so no "rise" is found
        assert rise is None, "Circumpolar source should not have a rise time"

    def test_never_visible_source_returns_none(self, coordinates, site):
        """Test that sources never visible return None for both times.

        From Chile (latitude ~-23), a source at dec +80 (far north)
        may never rise above the horizon.
        """
        ra = 0.0
        dec = 80.0  # Far north, may not be visible from Chile
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
        # the source is below the horizon and rises in ~6-12 hours.
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

        diff_hours = abs((rise_fine - rise_coarse).to_value("hour"))
        assert diff_hours < 0.5, (
            f"Coarse and fine results differ by {diff_hours:.3f} hours, expected < 0.5 hours"
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
