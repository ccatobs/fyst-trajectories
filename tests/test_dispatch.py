"""Tests for dispatch-time encoder-choice helpers (``fyst_trajectories.dispatch``)."""

import dataclasses

import pytest
from astropy.time import Time

from fyst_trajectories import choose_encoder_solution, get_fyst_site
from fyst_trajectories.exceptions import PointingError

# Fixed time; only consulted by the sun predicate. The geometry tests disable
# sun avoidance, so they need no ephemeris/network, the default predicate
# short-circuits to True when avoidance is disabled.
OBSTIME = Time("2026-03-15T12:00:00", scale="utc")


class TestChooseEncoderSolution:
    """Wrap enumeration, the sun-safety seam, and minimum-slew selection."""

    def test_returns_goal_el_unchanged(self):
        """The returned encoder elevation equals the goal elevation."""
        site = get_fyst_site(sun_avoidance_enabled=False)
        _, el = choose_encoder_solution(190.0, 45.0, 200.0, 45.0, OBSTIME, site)
        assert el == 45.0

    def test_picks_nearest_wrap(self):
        """Sky az 200 has images {200, -160}; from 190 the nearest is 200."""
        site = get_fyst_site(sun_avoidance_enabled=False)
        az, _ = choose_encoder_solution(190.0, 45.0, 200.0, 45.0, OBSTIME, site)
        assert az == pytest.approx(200.0)

    def test_picks_nearest_wrap_from_other_side(self):
        """From current az -170 the nearer image of sky az 200 is -160."""
        site = get_fyst_site(sun_avoidance_enabled=False)
        az, _ = choose_encoder_solution(-170.0, 45.0, 200.0, 45.0, OBSTIME, site)
        assert az == pytest.approx(-160.0)

    def test_single_image_low_azimuth(self):
        """Sky az 10 has a single in-range encoder image (10 itself)."""
        site = get_fyst_site(sun_avoidance_enabled=False)
        az, _ = choose_encoder_solution(0.0, 45.0, 10.0, 45.0, OBSTIME, site)
        assert az == pytest.approx(10.0)

    def test_chosen_az_within_limits(self):
        """The returned encoder azimuth is within the telescope az limits."""
        site = get_fyst_site(sun_avoidance_enabled=False)
        az, _ = choose_encoder_solution(0.0, 45.0, 350.0, 45.0, OBSTIME, site)
        lim = site.telescope_limits.azimuth
        assert lim.min <= az <= lim.max

    def test_injected_sun_predicate_selects_safe_wrap(self):
        """When a wrap is sun-blocked, the other in-range wrap is chosen."""
        site = get_fyst_site()

        def block_nonnegative(az, el, t):
            return az < 0  # block every encoder az >= 0

        az, _ = choose_encoder_solution(
            190.0, 45.0, 200.0, 45.0, OBSTIME, site, sun_safe=block_nonnegative
        )
        assert az == pytest.approx(-160.0)

    def test_all_wraps_blocked_raises(self):
        """A fully sun-blocked target raises PointingError."""
        site = get_fyst_site()

        def block_all(az, el, t):
            return False

        with pytest.raises(PointingError, match="sun-safe"):
            choose_encoder_solution(190.0, 45.0, 200.0, 45.0, OBSTIME, site, sun_safe=block_all)

    def test_goal_elevation_below_minimum_raises(self):
        """A goal elevation below the elevation limit raises PointingError."""
        site = get_fyst_site(sun_avoidance_enabled=False)
        with pytest.raises(PointingError, match="elevation"):
            choose_encoder_solution(190.0, 45.0, 200.0, 5.0, OBSTIME, site)

    def test_goal_elevation_above_maximum_raises(self):
        """A goal elevation above the elevation limit raises PointingError."""
        site = get_fyst_site(sun_avoidance_enabled=False)
        with pytest.raises(PointingError, match="elevation"):
            choose_encoder_solution(190.0, 45.0, 200.0, 95.0, OBSTIME, site)

    def test_sun_predicate_receives_goal_elevation(self):
        """The injected predicate is consulted with the goal elevation."""
        site = get_fyst_site()
        seen = []

        def spy(az, el, t):
            seen.append((az, el))
            return True

        choose_encoder_solution(190.0, 45.0, 200.0, 45.0, OBSTIME, site, sun_safe=spy)
        assert seen, "sun_safe predicate was not consulted"
        assert all(el == 45.0 for _, el in seen)

    def test_docstring_example_result(self):
        """Regression guard mirroring the dispatch.py docstring example."""
        site = get_fyst_site(sun_avoidance_enabled=False)
        assert choose_encoder_solution(190.0, 45.0, 200.0, 45.0, OBSTIME, site) == (200.0, 45.0)

    def test_no_in_range_wrap_raises(self):
        """A sky azimuth with no encoder image in a narrow az range raises PointingError."""
        base = get_fyst_site(sun_avoidance_enabled=False)
        narrow_az = dataclasses.replace(base.telescope_limits.azimuth, min=0.0, max=10.0)
        limits = dataclasses.replace(base.telescope_limits, azimuth=narrow_az)
        site = dataclasses.replace(base, telescope_limits=limits)
        with pytest.raises(PointingError, match="No encoder azimuth in range"):
            choose_encoder_solution(5.0, 45.0, 200.0, 45.0, OBSTIME, site)
