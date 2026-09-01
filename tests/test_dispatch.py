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


class TestChooseEncoderSolutionSpan:
    """Span-aware wrap admissibility (``goal_az_span``).

    The caller shifts the whole trajectory by the chosen 360 deg multiple, so a
    wrap is admissible only if both span endpoints stay within the azimuth limits
    after that shift. FYST azimuth limits are [-180, 360].
    """

    def test_far_wrap_chosen_when_near_wrap_span_overflows(self):
        """The span-fitting far wrap is returned even though the near wrap is nearer.

        ``goal_az = 350`` has images {350, -10} in [-180, 360]. The span
        (340, 370) overflows the upper limit at the near wrap (370 > 360) but
        fits at the far wrap (shifted to (-20, 10)). From ``current_az = 355``
        the near wrap (350) is the minimum slew, so without the span the
        function returns 350; with the span it must return -10.
        """
        site = get_fyst_site(sun_avoidance_enabled=False)
        # Control: without the span, the nearer (out-of-span) wrap is returned.
        az_no_span, _ = choose_encoder_solution(355.0, 45.0, 350.0, 45.0, OBSTIME, site)
        assert az_no_span == pytest.approx(350.0)
        # With the span, the far wrap that keeps the whole span in range wins.
        az_span, _ = choose_encoder_solution(
            355.0, 45.0, 350.0, 45.0, OBSTIME, site, goal_az_span=(340.0, 370.0)
        )
        assert az_span == pytest.approx(-10.0)

    def test_span_fits_no_wrap_raises_distinct_error(self):
        """A span wider than the whole azimuth range raises a span-named PointingError.

        The [-180, 360] range is 540 deg wide; a 600 deg span cannot fit at any
        wrap. The error message names the span and its width, distinct from the
        no-image-in-range message.
        """
        site = get_fyst_site(sun_avoidance_enabled=False)
        with pytest.raises(PointingError, match=r"trajectory azimuth span .* does not fit"):
            choose_encoder_solution(
                0.0, 45.0, 300.0, 45.0, OBSTIME, site, goal_az_span=(0.0, 600.0)
            )

    def test_span_endpoint_exactly_at_limit_is_admissible(self):
        """A span whose endpoints sit exactly on both limits is admissible.

        ``goal_az = 0`` with span (-180, 360) has width exactly 540 deg, with
        endpoints on the lower and upper limits. The inclusive bound check
        admits the k = 0 wrap and returns 0.
        """
        site = get_fyst_site(sun_avoidance_enabled=False)
        az, _ = choose_encoder_solution(
            0.0, 45.0, 0.0, 45.0, OBSTIME, site, goal_az_span=(-180.0, 360.0)
        )
        assert az == pytest.approx(0.0)

    def test_none_span_returns_near_wrap_unchanged(self):
        """``goal_az_span=None`` selects by the goal point alone (nearest wrap).

        Same scenario as the far-wrap test but with no span: the nearest wrap
        (350) is returned, proving the span admissibility is gated on the
        parameter.
        """
        site = get_fyst_site(sun_avoidance_enabled=False)
        az, _ = choose_encoder_solution(355.0, 45.0, 350.0, 45.0, OBSTIME, site, goal_az_span=None)
        assert az == pytest.approx(350.0)

    def test_span_min_greater_than_max_raises_valueerror(self):
        """An inverted span (min > max) is rejected with ValueError."""
        site = get_fyst_site(sun_avoidance_enabled=False)
        with pytest.raises(ValueError, match="must be <= max"):
            choose_encoder_solution(0.0, 45.0, 10.0, 45.0, OBSTIME, site, goal_az_span=(20.0, 5.0))

    def test_goal_outside_span_raises_valueerror(self):
        """A goal azimuth outside its declared span is rejected with ValueError."""
        site = get_fyst_site(sun_avoidance_enabled=False)
        with pytest.raises(ValueError, match="must lie within"):
            choose_encoder_solution(
                0.0, 45.0, 100.0, 45.0, OBSTIME, site, goal_az_span=(10.0, 20.0)
            )

    def test_goal_at_span_endpoint_within_tolerance(self):
        """A goal at a span endpoint, or outside it by less than the tolerance, passes.

        The endpoint bounds are inclusive, and a small tolerance absorbs the
        float round-off of callers deriving span and goal from the same array.
        """
        site = get_fyst_site(sun_avoidance_enabled=False)
        # goal_az == span_min; a normal single-wrap case that must not raise.
        az, _ = choose_encoder_solution(
            10.0, 45.0, 10.0, 45.0, OBSTIME, site, goal_az_span=(10.0, 25.0)
        )
        assert az == pytest.approx(10.0)
        # A goal a hair below span_min (float round-off scale) is accepted too.
        az, _ = choose_encoder_solution(
            10.0, 45.0, 10.0 - 5e-7, 45.0, OBSTIME, site, goal_az_span=(10.0, 25.0)
        )
        assert az == pytest.approx(10.0)

    def test_span_tie_break_uses_shifted_span_margin(self):
        """With equal slews, the tie-break margin is measured on the shifted span.

        Sky az -80 has images {-80, 280} in [-180, 360], both admissible for
        span (-110, -75) and both exactly 180 deg from current az 100. The
        margin to the az limits measured on the shifted span endpoints is
        70 deg at -80 vs 75 deg at 280, so 280 must win the tie; a margin
        measured on the goal point instead (100 vs 80 deg), or no tie-break at
        all (first candidate), would return -80.
        """
        site = get_fyst_site(sun_avoidance_enabled=False)
        az, _ = choose_encoder_solution(
            100.0, 45.0, -80.0, 45.0, OBSTIME, site, goal_az_span=(-110.0, -75.0)
        )
        assert az == pytest.approx(280.0)

    def test_degenerate_span_equals_no_span(self):
        """``goal_az_span=None`` and a point span ``(goal, goal)`` are equivalent.

        Both forms must return the same result on a genuine two-wrap geometry:
        sky az 200 has images {200, -160} in [-180, 360], and from current az
        190 the nearer wrap is 200.
        """
        site = get_fyst_site(sun_avoidance_enabled=False)
        no_span = choose_encoder_solution(190.0, 45.0, 200.0, 45.0, OBSTIME, site)
        point_span = choose_encoder_solution(
            190.0, 45.0, 200.0, 45.0, OBSTIME, site, goal_az_span=(200.0, 200.0)
        )
        assert no_span == point_span


# Three ISO instants; az 120 has a single in-range encoder image at FYST, so with
# no alternate wrap a predicate that fails at any one instant leaves nothing safe.
_T0 = Time("2026-03-15T12:00:00", scale="utc")
_T1 = Time("2026-03-15T12:05:00", scale="utc")
_T2 = Time("2026-03-15T12:10:00", scale="utc")
_OBSTIME_ARRAY = Time([_T0.iso, _T1.iso, _T2.iso], scale="utc")


class TestChooseEncoderSolutionObstimeArray:
    """Array-valued ``obstime``: a wrap is sun-safe only if safe at EVERY instant."""

    def test_empty_obstime_array_fails_closed(self):
        """An empty obstime array raises instead of vacuously passing the sun gate.

        ``list()`` of an empty Time array is ``[]`` and ``all(...)`` over an
        empty iterable is True, so without the guard every wrap would skip
        the sun check silently (and the ``slew_safe`` path would IndexError).
        """
        site = get_fyst_site()

        def must_not_be_consulted(az, el, t):
            pytest.fail("sun_safe must not be consulted for an empty obstime")

        with pytest.raises(ValueError, match="empty Time array"):
            choose_encoder_solution(
                120.0,
                45.0,
                120.0,
                45.0,
                _OBSTIME_ARRAY[:0],
                site,
                sun_safe=must_not_be_consulted,
            )

    def test_array_obstime_queries_every_instant(self):
        """A predicate safe at all three instants keeps the single in-range wrap.

        The recording predicate is consulted once per ``obstime`` element (one
        candidate az), so all three instants are queried and the wrap is returned.
        """
        site = get_fyst_site()
        seen = []

        def spy(az, el, t):
            seen.append(float(t.unix))
            return True

        az, _ = choose_encoder_solution(
            120.0, 45.0, 120.0, 45.0, _OBSTIME_ARRAY, site, sun_safe=spy
        )
        assert az == pytest.approx(120.0)
        assert sorted(round(u, 3) for u in seen) == [
            round(float(_T0.unix), 3),
            round(float(_T1.unix), 3),
            round(float(_T2.unix), 3),
        ]

    def test_array_obstime_unsafe_at_one_instant_excludes_wrap(self):
        """Unsafe at one of three instants excludes the only wrap, raising.

        The predicate is safe at ``_T0`` and ``_T2`` but blocked at ``_T1``; az 120
        has no alternate in-range wrap, so nothing survives and the sun-blocked
        PointingError fires, naming the first-through-last instant range.
        """
        site = get_fyst_site()

        def unsafe_at_t1(az, el, t):
            return abs(float(t.unix) - float(_T1.unix)) > 1.0

        with pytest.raises(PointingError, match=r"sun-safe azimuth wrap.*through"):
            choose_encoder_solution(
                120.0, 45.0, 120.0, 45.0, _OBSTIME_ARRAY, site, sun_safe=unsafe_at_t1
            )

    def test_scalar_obstime_control_returns_wrap(self):
        """Scalar ``obstime`` control: the same geometry, single instant, succeeds.

        Confirms the array path is what excludes the wrap above, not the geometry:
        with a scalar ``obstime`` and a predicate blocked only at ``_T1`` the wrap
        at ``_T0`` is safe and returned.
        """
        site = get_fyst_site()

        def unsafe_at_t1(az, el, t):
            return abs(float(t.unix) - float(_T1.unix)) > 1.0

        az, _ = choose_encoder_solution(120.0, 45.0, 120.0, 45.0, _T0, site, sun_safe=unsafe_at_t1)
        assert az == pytest.approx(120.0)
