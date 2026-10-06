"""Tests for dispatch-time encoder-choice helpers (``fyst_trajectories.dispatch``)."""

import dataclasses

import pytest
from astropy.time import Time

from fyst_trajectories import choose_encoder_solution, get_fyst_site
from fyst_trajectories.dispatch import EncoderSolution, estimate_slew_time
from fyst_trajectories.exceptions import PointingError
from fyst_trajectories.sun_models import _axis_slew_duration

# Fixed time; only consulted by the sun predicate. The geometry tests disable
# sun avoidance, so they need no ephemeris/network, the default predicate
# short-circuits to True when avoidance is disabled.
OBSTIME = Time("2026-03-15T12:00:00", scale="utc")


class TestChooseEncoderSolution:
    """Wrap enumeration, the sun-safety seam, and minimum-slew selection."""

    def test_single_image_low_azimuth(self):
        """Sky az 10 has a single in-range encoder image (10 itself)."""
        site = get_fyst_site(sun_avoidance_enabled=False)
        az, _ = choose_encoder_solution(0.0, 45.0, 10.0, 45.0, OBSTIME, site)
        assert az == pytest.approx(10.0)

    def test_chosen_az_within_limits(self):
        """From current az 0 the nearer image of sky az 350 is -10, inside the limits."""
        site = get_fyst_site(sun_avoidance_enabled=False)
        az, _ = choose_encoder_solution(0.0, 45.0, 350.0, 45.0, OBSTIME, site)
        assert az == pytest.approx(-10.0)
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


class TestEncoderSolutionErrorCauses:
    """Each refusing stage raises the typed error with its cause and diagnostics."""

    @staticmethod
    def _catch(**kwargs):
        from fyst_trajectories.exceptions import EncoderSolutionError

        site = kwargs.pop("site", None) or get_fyst_site(sun_avoidance_enabled=False)
        args = kwargs.pop("args")
        with pytest.raises(EncoderSolutionError) as info:
            choose_encoder_solution(*args, site, **kwargs)
        exc = info.value
        assert isinstance(exc, PointingError)
        return exc

    def test_goal_elevation(self):
        exc = self._catch(args=(190.0, 45.0, 200.0, 10.0, OBSTIME))
        assert exc.cause == "goal_elevation"
        assert (exc.goal_az, exc.goal_el) == (200.0, 10.0)
        assert exc.candidates == ()
        assert exc.current_az is None

    def test_span_unreachable(self):
        exc = self._catch(args=(0.0, 45.0, 100.0, 45.0, OBSTIME), goal_az_span=(-200.0, 400.0))
        assert exc.cause == "span_unreachable"
        assert exc.candidates == ()

    def test_no_image(self):
        base = get_fyst_site(sun_avoidance_enabled=False)
        narrow = dataclasses.replace(base.telescope_limits.azimuth, min=0.0, max=90.0)
        site = dataclasses.replace(
            base, telescope_limits=dataclasses.replace(base.telescope_limits, azimuth=narrow)
        )
        exc = self._catch(args=(10.0, 45.0, 180.0, 45.0, OBSTIME), site=site)
        assert exc.cause == "no_image"

    def test_sun_blocked_carries_the_in_range_wraps(self):
        exc = self._catch(
            args=(190.0, 45.0, 200.0, 45.0, OBSTIME),
            site=get_fyst_site(),
            sun_safe=lambda az, el, t: False,
        )
        assert exc.cause == "sun_blocked"
        assert sorted(exc.candidates) == [-160.0, 200.0]
        assert exc.time_iso == OBSTIME.iso

    def test_path_blocked_carries_the_point_safe_wraps_and_the_start(self):
        exc = self._catch(
            args=(190.0, 45.0, 200.0, 45.0, OBSTIME),
            site=get_fyst_site(),
            sun_safe=lambda az, el, t: True,
            slew_safe=lambda a, b, c, d, t: False,
        )
        assert exc.cause == "path_blocked"
        assert sorted(exc.candidates) == [-160.0, 200.0]
        assert (exc.current_az, exc.current_el) == (190.0, 45.0)
        assert exc.time_iso == OBSTIME.iso


class TestSunGateBatching:
    """The Sun gate consults a batch-capable model once, not once per pair.

    The default predicate solves the Sun ephemeris on every call, so a long
    dwell grid has to be consulted once rather than once per (wrap, time)
    pair: a dispatcher has about 10 s before the scan must start.
    """

    def test_default_gate_is_the_scalar_model_and_agrees_with_is_sun_safe(self, monkeypatch):
        """The default Sun test is the scalar model, asked once per call through ``batch``.

        Over a grid of goals around the Sun its outcomes match the bare
        ``Coordinates.is_sun_safe`` consulted per (wrap, time) pair, refusals
        included.
        """
        import numpy as np
        from astropy import units as u

        from fyst_trajectories import Coordinates
        from fyst_trajectories.exceptions import EncoderSolutionError
        from fyst_trajectories.sun_models import _ScalarSunModel

        calls = []
        real_batch = _ScalarSunModel.batch

        def spy(self, az_deg, el_deg, times):
            calls.append(np.size(az_deg))
            return real_batch(self, az_deg, el_deg, times)

        monkeypatch.setattr(_ScalarSunModel, "batch", spy)

        site = get_fyst_site()
        bare = Coordinates(site).is_sun_safe
        grid = OBSTIME + np.array([0.0, 300.0, 600.0]) * u.s

        def outcome(goal_az, goal_el, sun_safe):
            try:
                return tuple(
                    choose_encoder_solution(
                        0.0, 45.0, goal_az, goal_el, grid, site, sun_safe=sun_safe
                    )
                )
            except EncoderSolutionError as exc:
                return exc.cause

        outcomes = []
        for goal_az in np.arange(0.0, 360.0, 30.0):
            for goal_el in (20.0, 50.0, 80.0):
                calls.clear()
                default = outcome(float(goal_az), goal_el, None)
                assert len(calls) == 1
                assert outcome(float(goal_az), goal_el, bare) == default
                outcomes.append(default)
        # The grid holds both answers, so the agreement is not vacuous.
        assert "sun_blocked" in outcomes
        assert any(isinstance(o, tuple) for o in outcomes)

    def test_goal_el_above_90_is_refused_even_with_avoidance_disabled(self):
        """A custom site whose elevation limit exceeds 90 deg still gets no over-the-top pose.

        The default Sun test checks its inputs whether or not the site's Sun
        avoidance is enabled, and an elevation above 90 deg is outside them.
        """
        import dataclasses

        site = get_fyst_site(sun_avoidance_enabled=False)
        limits = site.telescope_limits
        custom = dataclasses.replace(
            site,
            telescope_limits=dataclasses.replace(
                limits, elevation=dataclasses.replace(limits.elevation, max=180.0)
            ),
        )
        with pytest.raises(ValueError, match=r"el_deg must lie within \[-90, 90\]"):
            choose_encoder_solution(0.0, 45.0, 100.0, 100.0, OBSTIME, custom)

    def test_batch_capable_model_is_called_once(self):
        import numpy as np
        from astropy import units as u

        calls = {"batch": 0, "scalar": 0}

        class _Model:
            def __call__(self, az, el, t):
                calls["scalar"] += 1
                return True

            def batch(self, az, el, t):
                calls["batch"] += 1
                return np.ones(np.shape(az), dtype=bool)

        grid = OBSTIME + np.arange(50) * 1.0 * u.s
        choose_encoder_solution(190.0, 45.0, 200.0, 45.0, grid, get_fyst_site(), sun_safe=_Model())
        assert calls["batch"] == 1
        assert calls["scalar"] == 0

    def test_bare_predicate_still_works_per_pair(self):
        import numpy as np
        from astropy import units as u

        seen = []

        def predicate(az, el, t):
            seen.append((az, float(t.unix)))
            return True

        grid = OBSTIME + np.arange(4) * 1.0 * u.s
        choose_encoder_solution(190.0, 45.0, 200.0, 45.0, grid, get_fyst_site(), sun_safe=predicate)
        # Two in-range wraps of sky azimuth 200 in [-180, 360], four times.
        assert len(seen) == 8
        assert sorted({az for az, _ in seen}) == [-160.0, 200.0]

    def test_a_wrap_is_rejected_when_any_grid_time_is_unsafe(self):
        """The batch path keeps the all-times-must-pass semantics."""
        import numpy as np
        from astropy import units as u

        grid = OBSTIME + np.arange(3) * 1.0 * u.s

        class _BlocksOneWrapLate:
            def batch(self, az, el, t):
                az = np.asarray(az, dtype=float)
                unix = np.asarray(t.unix, dtype=float)
                # Wrap 200 is unsafe at the last sample only.
                return ~((az > 0.0) & (unix >= float(grid[-1].unix)))

            def __call__(self, az, el, t):  # pragma: no cover - batch is used
                raise AssertionError("batch should have been used")

        az, _ = choose_encoder_solution(
            190.0, 45.0, 200.0, 45.0, grid, get_fyst_site(), sun_safe=_BlocksOneWrapLate()
        )
        assert az == -160.0


class TestEncoderSolutionCarriesTheWrapShift:
    """The chosen wrap's 360-degree multiple is returned, not discarded.

    ``choose_encoder_solution`` computes the shift that maps the caller's goal
    azimuth frame onto the wrap it picked; dropping it on the way out would
    leave every consumer to rediscover it. The return value is a two-element
    tuple so the shift is additive: two-value unpacking is unchanged.
    """

    def test_unpacks_to_two_values(self):
        """``az, el = ...`` still works, and the tuple compares equal to the pair."""
        site = get_fyst_site(sun_avoidance_enabled=False)
        solution = choose_encoder_solution(190.0, 45.0, 200.0, 45.0, OBSTIME, site)
        az, el = solution
        assert (az, el) == (200.0, 45.0)
        assert solution == (200.0, 45.0)
        assert len(solution) == 2

    def test_named_access_matches_the_tuple(self):
        """``.az`` and ``.el`` name the two elements."""
        site = get_fyst_site(sun_avoidance_enabled=False)
        solution = choose_encoder_solution(190.0, 45.0, 200.0, 45.0, OBSTIME, site)
        assert (solution.az, solution.el) == (solution[0], solution[1])

    def test_shift_is_zero_when_the_goal_wrap_is_chosen(self):
        """Sky az 200 from current az 190 lands on the goal's own wrap."""
        site = get_fyst_site(sun_avoidance_enabled=False)
        solution = choose_encoder_solution(190.0, 45.0, 200.0, 45.0, OBSTIME, site)
        assert solution.az_shift == 0.0

    def test_shift_names_the_multiple_that_reaches_the_chosen_wrap(self):
        """From current az -170 the near wrap is -160, one turn below the goal."""
        site = get_fyst_site(sun_avoidance_enabled=False)
        solution = choose_encoder_solution(-170.0, 45.0, 200.0, 45.0, OBSTIME, site)
        assert solution.az == -160.0
        assert solution.az_shift == -360.0
        assert 200.0 + solution.az_shift == solution.az

    def test_shift_is_a_whole_number_of_turns(self):
        """Whatever the geometry, the shift is a multiple of 360."""
        site = get_fyst_site(sun_avoidance_enabled=False)
        for current, goal in ((0.0, 350.0), (350.0, 10.0), (190.0, 200.0)):
            shift = choose_encoder_solution(current, 45.0, goal, 45.0, OBSTIME, site).az_shift
            assert shift % 360.0 == 0.0

    def test_survives_copy_and_pickle(self):
        """The shift is carried through the marshalling a task boundary performs."""
        import copy
        import pickle

        site = get_fyst_site(sun_avoidance_enabled=False)
        solution = choose_encoder_solution(-170.0, 45.0, 200.0, 45.0, OBSTIME, site)
        for revived in (copy.copy(solution), pickle.loads(pickle.dumps(solution))):
            assert revived == solution
            assert revived.az_shift == solution.az_shift


class TestNonFiniteInputsAreRefused:
    """A NaN or infinite pose is refused with a ValueError naming the argument.

    A NaN ``current_az`` (a lost position read) makes every wrap distance NaN,
    so without the refusal ``min`` returns the first, most negative wrap; an
    infinite goal overflows the wrap enumeration. Neither is an infeasibility,
    so the refusal is not a ``PointingError``.
    """

    _GOOD = {"current_az": 190.0, "current_el": 45.0, "goal_az": 200.0, "goal_el": 45.0}

    @pytest.mark.parametrize("name", ["current_az", "current_el", "goal_az", "goal_el"])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_each_pose_argument(self, name, bad):
        site = get_fyst_site(sun_avoidance_enabled=False)
        kwargs = {**self._GOOD, name: bad}
        with pytest.raises(ValueError, match=f"{name} must be finite") as info:
            choose_encoder_solution(obstime=OBSTIME, site=site, **kwargs)
        assert not isinstance(info.value, PointingError)

    @pytest.mark.parametrize("span", [(float("nan"), 210.0), (190.0, float("inf"))])
    def test_span_endpoints(self, span):
        site = get_fyst_site(sun_avoidance_enabled=False)
        with pytest.raises(ValueError, match="goal_az_span must be finite") as info:
            choose_encoder_solution(190.0, 45.0, 200.0, 45.0, OBSTIME, site, goal_az_span=span)
        assert not isinstance(info.value, PointingError)

    def test_nan_current_el_is_not_reported_as_a_blocked_path(self):
        """With a path check, a NaN start elevation is not a Sun refusal."""
        site = get_fyst_site()

        def clear(az, el, t):
            return True

        def path_clear_when_finite(current_az, current_el, goal_az, goal_el, t):
            return all(v == v for v in (current_az, current_el, goal_az, goal_el))

        with pytest.raises(ValueError, match="current_el must be finite") as info:
            choose_encoder_solution(
                190.0,
                float("nan"),
                200.0,
                45.0,
                OBSTIME,
                site,
                sun_safe=clear,
                slew_safe=path_clear_when_finite,
            )
        assert not isinstance(info.value, PointingError)

    def test_finite_control_is_unchanged(self):
        site = get_fyst_site(sun_avoidance_enabled=False)
        solution = choose_encoder_solution(190.0, 45.0, 200.0, 45.0, OBSTIME, site)
        assert solution == (200.0, 45.0)
        assert solution.az_shift == 0.0


class TestEncoderSolutionIsImmutable:
    """``az_shift`` is set once; equality and hashing are those of the pose."""

    def test_assignment_deletion_and_new_attributes_raise(self):
        solution = EncoderSolution(-160.0, 45.0, -360.0)
        with pytest.raises(AttributeError):
            solution.az_shift = 0.0
        with pytest.raises(AttributeError):
            del solution.az_shift
        with pytest.raises(AttributeError):
            solution.other = 1
        assert solution.az_shift == -360.0
        assert not hasattr(solution, "other")

    def test_equality_and_hash_follow_the_pose(self):
        a = EncoderSolution(10.0, 45.0, 0.0)
        b = EncoderSolution(10.0, 45.0, 360.0)
        assert a == b == (10.0, 45.0)
        assert hash(a) == hash(b) == hash((10.0, 45.0))
        assert (tuple(a), a.az_shift) != (tuple(b), b.az_shift)

    def test_copies_revive_the_shift_and_stay_immutable(self):
        import copy
        import pickle

        solution = EncoderSolution(-160.0, 45.0, -360.0)
        revived_all = [copy.copy(solution), copy.deepcopy(solution)]
        revived_all += [
            pickle.loads(pickle.dumps(solution, protocol=p))
            for p in range(pickle.HIGHEST_PROTOCOL + 1)
        ]
        for revived in revived_all:
            assert revived == solution
            assert revived.az_shift == -360.0
            with pytest.raises(AttributeError):
                revived.az_shift = 0.0


class TestEstimateSlewTime:
    """Trapezoidal and triangular slew profiles, and the shared kinematic kernel.

    The pinned durations follow from the site's axis velocity and
    acceleration limits, which are pending instrument verification (see the
    table on the documentation index); they move when those limits do.
    """

    def test_zero_distance(self, site):
        t = estimate_slew_time(180.0, 50.0, 180.0, 50.0, site)
        assert t == 0.0

    def test_az_only(self, site):
        # 10 deg azimuth slew, trapezoidal profile (FYST az vel=3.0, accel=1.5):
        # t_accel=2, d_accel=6; distance 10 > 6, so
        # t = 2*t_accel + (10 - d_accel)/vel = 4 + 4/3 = 5.333 s.
        t = estimate_slew_time(180.0, 50.0, 190.0, 50.0, site)
        assert t == pytest.approx(5.333, abs=0.01)

    def test_el_only(self, site):
        # 10 deg elevation slew, trapezoidal (FYST el vel=1.0, accel=0.75):
        # t_accel=1.333, d_accel=1.333; distance 10 > 1.333, so
        # t = 2*t_accel + (10 - d_accel)/vel = 2.667 + 8.667 = 11.333 s.
        t = estimate_slew_time(180.0, 50.0, 180.0, 60.0, site)
        assert t == pytest.approx(11.333, abs=0.01)

    def test_el_slower_than_az(self, site):
        t_az = estimate_slew_time(180.0, 50.0, 190.0, 50.0, site)
        t_el = estimate_slew_time(180.0, 50.0, 180.0, 60.0, site)
        assert t_el > t_az

    def test_large_slew(self, site):
        t = estimate_slew_time(0.0, 30.0, 180.0, 70.0, site)
        assert t > 30.0

    def test_short_az_slew_is_triangular(self, site):
        # A 2 deg az slew never reaches cruise: d_accel = v^2/a = 6 deg > 2 deg, so
        # the triangular branch gives t = 2*sqrt(distance/a) = 2*sqrt(2/1.5) = 2.309 s.
        t = estimate_slew_time(180.0, 50.0, 182.0, 50.0, site)
        assert t == pytest.approx(2.309, abs=0.01)

    @pytest.mark.parametrize(
        "d_az,d_el",
        [(10.0, 0.0), (0.0, 10.0), (0.5, 0.0), (180.0, 40.0), (3.0, 1.5)],
    )
    def test_the_estimate_is_the_shared_profile(self, site, d_az, d_el):
        """The estimator and the Sun sweep price a slew with one profile.

        A second copy of the trapezoid here would let the duration a scan is
        priced with drift from the duration its path is sampled over.
        """
        az_limits = site.telescope_limits.azimuth
        el_limits = site.telescope_limits.elevation
        expected = max(
            _axis_slew_duration(d_az, az_limits.max_velocity, az_limits.max_acceleration),
            _axis_slew_duration(d_el, el_limits.max_velocity, el_limits.max_acceleration),
        )
        assert estimate_slew_time(0.0, 45.0, d_az, 45.0 + d_el, site) == pytest.approx(expected)
