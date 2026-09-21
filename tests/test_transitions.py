"""Tests for the pairwise slew transition primitive.

Every cause is reached through predicate fakes, so no test here touches
the Sun ephemeris; the one real-model test uses a site with avoidance
disabled.
"""

import dataclasses
import typing

import numpy as np
import pytest
from _sun_stubs import fake_sun_model
from astropy.time import Time, TimeDelta

from fyst_trajectories import Coordinates, get_fyst_site
from fyst_trajectories.exceptions import EncoderSolutionCause
from fyst_trajectories.overhead import (
    DeferralReason,
    Transition,
    estimate_slew_time,
    plan_escape,
    plan_transition,
)
from fyst_trajectories.overhead import transitions as transitions_module

T0 = Time("2026-03-15T12:00:00", scale="utc")


def _always(verdict: bool):
    """Build a point predicate with a fixed verdict."""
    return fake_sun_model(verdict, batch=False)


def _path_always(verdict: bool):
    """Build a path predicate with a fixed verdict."""

    def predicate(current_az, current_el, goal_az, goal_el, t):
        return verdict

    return predicate


class _HighPathsOnly:
    """A path predicate that clears any leg touching a high elevation.

    Blocks the direct low-elevation slew but clears a detour that climbs,
    and exposes ``evaluate`` so ``find_sun_safe_detour`` can chain legs.
    """

    def __init__(self, clear_above: float = 50.0):
        self.clear_above = clear_above

    def __call__(self, current_az, current_el, goal_az, goal_el, t):
        return max(current_el, goal_el) >= self.clear_above

    def evaluate(self, current_az, current_el, goal_az, goal_el, t):
        safe = self(current_az, current_el, goal_az, goal_el, t)
        times = t + TimeDelta(np.array([0.0, 30.0]), format="sec")
        return safe, np.array([current_az, goal_az]), np.array([current_el, goal_el]), times


@pytest.fixture
def quiet_site():
    """Provide the FYST site with Sun avoidance disabled (no ephemeris consulted)."""
    return get_fyst_site(sun_avoidance_enabled=False)


class TestDeferralReason:
    """The shared reason vocabulary and its mapping from the encoder causes."""

    def test_str_is_the_value(self):
        assert str(DeferralReason.SUN_PATH) == "sun_path"
        assert DeferralReason("no_wrap") is DeferralReason.NO_WRAP

    def test_cause_mapping_covers_every_encoder_cause(self):
        assert set(transitions_module._CAUSE_TO_REASON) == set(
            typing.get_args(EncoderSolutionCause)
        )
        assert set(transitions_module._CAUSE_TO_REASON.values()) == {
            DeferralReason.LIMITS,
            DeferralReason.NO_WRAP,
            DeferralReason.SUN_POINT,
            DeferralReason.SUN_PATH,
        }


class TestDirectTransition:
    """Commandable transitions: wrap choice, pricing, spans."""

    def test_nearer_wrap_and_priced_by_the_kinematic_estimate(self, site):
        tr = plan_transition(
            190.0,
            45.0,
            200.0,
            45.0,
            T0,
            site,
            sun_safe=_always(True),
            slew_safe=_path_always(True),
            settle_time=5.0,
        )
        assert tr.safe and tr.path == "direct" and tr.cause is DeferralReason.OK
        assert tr.az_to == 200.0 and tr.el_to == 45.0
        assert tr.duration == pytest.approx(
            estimate_slew_time(190.0, 45.0, 200.0, 45.0, site) + 5.0
        )
        assert (tr.arrival - T0).to_value("s") == pytest.approx(tr.duration)
        assert tr.detour_via is None

    def test_duration_is_independent_of_the_safety_model(self, site):
        kwargs = dict(sun_safe=_always(True), settle_time=0.0)
        a = plan_transition(0.0, 30.0, 90.0, 60.0, T0, site, slew_safe=_path_always(True), **kwargs)

        class _Slow:
            def __call__(self, *args):
                return True

        b = plan_transition(0.0, 30.0, 90.0, 60.0, T0, site, slew_safe=_Slow(), **kwargs)
        assert a.duration == b.duration

    def test_span_selects_the_wrap_that_fits(self, site):
        # Sky azimuth 350 has encoder images 350 and -10; a span reaching
        # 370 fits only the lower wrap [-10, 10] inside [-180, 360].
        tr = plan_transition(
            0.0,
            45.0,
            350.0,
            45.0,
            T0,
            site,
            sun_safe=_always(True),
            slew_safe=_path_always(True),
            goal_az_span=(330.0, 370.0),
        )
        assert tr.az_to == pytest.approx(-10.0)
        # The wrap decision is carried, not left to be rediscovered by
        # subtraction: the shift is what a caller applies to the whole
        # trajectory it is about to command.
        assert tr.az_shift == pytest.approx(-360.0)
        assert tr.az_to == pytest.approx(350.0 + tr.az_shift)

    def test_the_nearer_wrap_records_a_zero_shift(self, site):
        tr = plan_transition(
            190.0,
            45.0,
            200.0,
            45.0,
            T0,
            site,
            sun_safe=_always(True),
            slew_safe=_path_always(True),
        )
        assert tr.az_shift == 0.0

    def test_real_scalar_model_with_avoidance_disabled(self, quiet_site):
        tr = plan_transition(190.0, 45.0, 200.0, 45.0, T0, quiet_site, settle_time=5.0)
        assert tr.path == "direct"
        assert tr.duration == pytest.approx(10.3, abs=0.05)


class TestBlockedTransitions:
    """Every refusal cause comes back as a value, never a raise."""

    def test_goal_elevation_outside_limits_is_limits(self, site):
        tr = plan_transition(
            190.0, 45.0, 200.0, 10.0, T0, site, sun_safe=_always(True), slew_safe=_path_always(True)
        )
        assert not tr.safe
        assert tr.cause is DeferralReason.LIMITS
        assert tr.path == "blocked"
        assert tr.duration == 0.0
        assert tr.arrival == T0
        assert (tr.az_to, tr.el_to) == (200.0, 10.0)

    def test_span_wider_than_the_window_is_no_wrap(self, site):
        tr = plan_transition(
            0.0,
            45.0,
            100.0,
            45.0,
            T0,
            site,
            sun_safe=_always(True),
            slew_safe=_path_always(True),
            goal_az_span=(-200.0, 400.0),
        )
        assert tr.cause is DeferralReason.NO_WRAP

    def test_no_image_is_no_wrap(self, site):
        narrow = dataclasses.replace(site.telescope_limits.azimuth, min=0.0, max=90.0)
        limits = dataclasses.replace(site.telescope_limits, azimuth=narrow)
        narrow_site = dataclasses.replace(site, telescope_limits=limits)
        tr = plan_transition(
            10.0,
            45.0,
            180.0,
            45.0,
            T0,
            narrow_site,
            sun_safe=_always(True),
            slew_safe=_path_always(True),
        )
        assert tr.cause is DeferralReason.NO_WRAP

    def test_every_wrap_in_the_sun_is_sun_point(self, site):
        tr = plan_transition(
            190.0,
            45.0,
            200.0,
            45.0,
            T0,
            site,
            sun_safe=_always(False),
            slew_safe=_path_always(True),
        )
        assert tr.cause is DeferralReason.SUN_POINT

    def test_blocked_direct_path_is_sun_path(self, site):
        tr = plan_transition(
            190.0,
            45.0,
            200.0,
            45.0,
            T0,
            site,
            sun_safe=_always(True),
            slew_safe=_path_always(False),
        )
        assert tr.cause is DeferralReason.SUN_PATH
        assert tr.path == "blocked"

    def test_blocked_path_stays_blocked_without_detour_permission(self, site):
        tr = plan_transition(
            0.0, 30.0, 90.0, 30.0, T0, site, sun_safe=_always(True), slew_safe=_HighPathsOnly()
        )
        assert tr.cause is DeferralReason.SUN_PATH


class TestTransitionHold:
    """The goal must still be clear when the telescope gets there and stays."""

    def _safe_until(self, cut_seconds: float):
        """Build a point predicate that closes ``cut_seconds`` after ``T0``."""

        def predicate(az, el, t):
            return bool((t - T0).sec < cut_seconds)

        return predicate

    def test_a_goal_that_closes_during_the_hold_is_refused(self, site):
        """Safe now and on arrival, gone a tick later: the slew buys nothing."""
        kwargs = dict(sun_safe=self._safe_until(120.0), slew_safe=_path_always(True))
        arrival_only = plan_transition(190.0, 45.0, 200.0, 45.0, T0, site, **kwargs)
        assert arrival_only.safe and arrival_only.duration < 120.0

        held = plan_transition(190.0, 45.0, 200.0, 45.0, T0, site, hold=300.0, **kwargs)
        assert not held.safe
        assert held.cause is DeferralReason.SUN_POINT
        assert held.duration == 0.0

    def test_a_goal_that_survives_the_hold_is_unchanged(self, site):
        """A hold the goal outlives changes neither the wrap nor the price."""
        kwargs = dict(sun_safe=_always(True), slew_safe=_path_always(True), settle_time=5.0)
        plain = plan_transition(190.0, 45.0, 200.0, 45.0, T0, site, **kwargs)
        held = plan_transition(190.0, 45.0, 200.0, 45.0, T0, site, hold=300.0, **kwargs)
        assert (held.az_to, held.el_to) == (plain.az_to, plain.el_to)
        assert held.duration == pytest.approx(plain.duration)

    def test_the_hold_grid_covers_start_arrival_and_the_dwell(self, site):
        """Three instants are asked about: now, arrival, and arrival plus the hold."""
        seen = []

        class _Recorder:
            def __call__(self, az, el, t):
                seen.append(float((t - T0).sec))
                return True

        tr = plan_transition(
            190.0,
            45.0,
            200.0,
            45.0,
            T0,
            site,
            sun_safe=_Recorder(),
            slew_safe=_path_always(True),
            hold=300.0,
        )
        assert tr.safe
        offsets = sorted(set(round(s, 3) for s in seen))
        assert offsets[0] == 0.0
        assert offsets[-1] == pytest.approx(tr.duration + 300.0, abs=1e-3)
        assert any(abs(o - tr.duration) < 1e-3 for o in offsets)

    def test_a_negative_hold_raises(self, site):
        with pytest.raises(ValueError, match="hold"):
            plan_transition(190.0, 45.0, 200.0, 45.0, T0, site, hold=-1.0)


class TestDetour:
    """Two-leg detours when the direct path is blocked."""

    def test_detour_when_allowed(self, site):
        tr = plan_transition(
            0.0,
            30.0,
            90.0,
            30.0,
            T0,
            site,
            sun_safe=_always(True),
            slew_safe=_HighPathsOnly(),
            settle_time=5.0,
            allow_detour=True,
        )
        assert tr.safe and tr.path == "detour"
        assert tr.detour_via is not None
        az_mid, el_mid = tr.detour_via
        assert el_mid >= 50.0
        assert az_mid == pytest.approx(45.0)
        assert tr.az_to == pytest.approx(90.0)
        expected = (
            estimate_slew_time(0.0, 30.0, az_mid, el_mid, site)
            + estimate_slew_time(az_mid, el_mid, 90.0, 30.0, site)
            + 5.0
        )
        assert tr.duration == pytest.approx(expected)

    def test_no_detour_found_stays_blocked(self, site):
        tr = plan_transition(
            0.0,
            30.0,
            90.0,
            30.0,
            T0,
            site,
            sun_safe=_always(True),
            slew_safe=_HighPathsOnly(clear_above=1000.0),
            allow_detour=True,
        )
        assert tr.cause is DeferralReason.SUN_PATH

    def test_detour_needs_an_evaluating_predicate(self, site):
        with pytest.raises(ValueError, match="evaluate"):
            plan_transition(
                0.0,
                30.0,
                90.0,
                30.0,
                T0,
                site,
                sun_safe=_always(True),
                slew_safe=_path_always(False),
                allow_detour=True,
            )


class TestArgumentRules:
    """Bad arguments raise ValueError."""

    def test_non_scalar_time_raises(self, site):
        times = Time(["2026-03-15T12:00:00", "2026-03-15T12:05:00"], scale="utc")
        with pytest.raises(ValueError, match="scalar start time"):
            plan_transition(0.0, 45.0, 10.0, 45.0, times, site, sun_safe=_always(True))

    def test_negative_settle_raises(self, site):
        with pytest.raises(ValueError, match="settle_time must be non-negative"):
            plan_transition(
                0.0, 45.0, 10.0, 45.0, T0, site, sun_safe=_always(True), settle_time=-1.0
            )

    def test_malformed_span_stays_a_value_error(self, site):
        with pytest.raises(ValueError, match="must be <= max"):
            plan_transition(
                0.0, 45.0, 10.0, 45.0, T0, site, sun_safe=_always(True), goal_az_span=(20.0, 0.0)
            )

    def test_incoherent_frame_raises(self, site):
        with pytest.raises(ValueError, match="coherent azimuth frame"):
            plan_transition(
                -1000.0,
                45.0,
                200.0,
                45.0,
                T0,
                site,
                sun_safe=_always(True),
                slew_safe=_path_always(True),
            )


class TestDefaultSlewSafe:
    """The default path model reads the site limits; the value type is frozen."""

    def test_default_sweep_uses_the_site_limits(self, site):
        slow_az = dataclasses.replace(site.telescope_limits.azimuth, max_velocity=0.5)
        limits = dataclasses.replace(site.telescope_limits, azimuth=slow_az)
        slow_site = dataclasses.replace(site, telescope_limits=limits)
        model = transitions_module._default_slew_safe(_always(True), slow_site)
        assert model._az_speed == 0.5
        assert model._el_accel == site.telescope_limits.elevation.max_acceleration

    def test_transition_is_frozen(self):
        tr = Transition(0.0, 45.0, 10.0, 45.0, T0, 5.0, DeferralReason.OK)
        with pytest.raises(dataclasses.FrozenInstanceError):
            tr.duration = 1.0


def _unsafe_at(az0: float, el0: float, radius: float = 1.0):
    """Build a point predicate unsafe only within ``radius`` deg of one pose."""
    return fake_sun_model(
        lambda az, el, t: not (abs(float(az) - az0) < radius and abs(float(el) - el0) < radius),
        batch=False,
    )


def _unsafe_above(el_cap: float):
    """Build a point predicate unsafe at every azimuth above ``el_cap`` (a high-Sun cap)."""
    return fake_sun_model(lambda az, el, t: float(el) <= el_cap, batch=False)


#: Sky-azimuth sector of the directional fake that demands more separation,
#: chosen to straddle the anti-solar azimuth at ``T0`` (263.85 deg).
_HUNGRY_SECTOR = (223.85, 303.85)
#: Required separation inside and outside that sector, in degrees. Outside it
#: the anti-solar pose (110.79 deg away at T0) has the largest separation of
#: any candidate but the smallest margin, so the two rules disagree.
_HUNGRY_THRESHOLD_DEG = 109.0
_PLAIN_THRESHOLD_DEG = 75.0


def _required_separation(az: float) -> float:
    """Return the fake zone's required separation at one sky azimuth."""
    sky_az = float(az) % 360.0
    lo, hi = _HUNGRY_SECTOR
    return _HUNGRY_THRESHOLD_DEG if lo <= sky_az <= hi else _PLAIN_THRESHOLD_DEG


class _DirectionalFake:
    """A zone whose required separation depends on the azimuth sector.

    Stands in for the shipped directional model: it answers verdicts like
    any point predicate and additionally exposes ``threshold``, the
    optional extension ``plan_escape`` reads to measure depth against the
    model's own requirement rather than against raw Sun separation.
    """

    def __init__(self, coords):
        self._coords = coords

    def _separation(self, az, el, t):
        sun_az, sun_el = self._coords.get_sun_altaz(t)
        return np.atleast_1d(self._coords.angular_separation(az, el, sun_az, sun_el))

    def __call__(self, az, el, t):
        return bool(self._separation(az, el, t)[0] > _required_separation(az))

    def threshold(self, az, el, times):
        return np.array([_required_separation(a) for a in np.atleast_1d(az)], dtype=float)


class TestEscape:
    """plan_escape moves an overtaken pose out of the zone under the relaxed rule."""

    def test_safe_pose_needs_no_escape(self, site):
        assert plan_escape(180.0, 50.0, T0, site, sun_safe=_always(True)) is None

    def test_zone_everywhere_is_no_escape(self, site):
        tr = plan_escape(180.0, 50.0, T0, site, sun_safe=_always(False))
        assert tr is not None and not tr.safe
        assert tr.cause is DeferralReason.NO_ESCAPE
        assert (tr.az_to, tr.el_to, tr.duration) == (180.0, 50.0, 0.0)

    def test_escapes_to_the_anti_solar_azimuth_at_the_current_elevation(self, site):
        """Only the start pose is unsafe, so the current elevation is kept.

        Among that elevation's candidates the farthest on arrival, the
        anti-solar azimuth, wins.
        """
        coords = Coordinates(site)
        sun_az, sun_el = coords.get_sun_altaz(T0)
        start_sep = coords.angular_separation(180.0, 50.0, sun_az, sun_el)

        tr = plan_escape(180.0, 50.0, T0, site, sun_safe=_unsafe_at(180.0, 50.0), settle_time=5.0)

        assert tr is not None and tr.safe and tr.path == "direct"
        assert tr.el_to == 50.0
        assert tr.az_to == pytest.approx((sun_az + 180.0) % 360.0)
        end_sep = coords.angular_separation(tr.az_to, tr.el_to, sun_az, sun_el)
        assert end_sep > start_sep
        assert tr.duration == pytest.approx(
            estimate_slew_time(180.0, 50.0, tr.az_to, tr.el_to, site) + 5.0
        )
        assert tr.arrival.unix == pytest.approx(T0.unix + tr.duration)

    def test_high_sun_cap_steps_down_to_the_first_clear_elevation(self, site):
        """Every azimuth above 32 deg is inside the zone: the escape drops to 30."""
        tr = plan_escape(180.0, 50.0, T0, site, sun_safe=_unsafe_above(32.0))
        assert tr is not None and tr.safe
        assert tr.el_to == 30.0

    def test_elevation_floor_is_honoured(self, site):
        tr = plan_escape(180.0, 50.0, T0, site, sun_safe=_unsafe_above(32.0), el_floor=40.0)
        assert tr is not None and tr.cause is DeferralReason.NO_ESCAPE
        with pytest.raises(ValueError, match="el_floor"):
            plan_escape(180.0, 50.0, T0, site, sun_safe=_always(False), el_floor=5.0)

    def test_path_may_not_approach_the_sun(self, site):
        """A candidate whose path sweeps past the Sun is rejected even if its end is safe.

        With the Sun low in the east, the wrap a full turn away from 180 is
        as far from the Sun as the start, but its path passes through the
        Sun's azimuth; the escape must not pick it.
        """
        coords = Coordinates(site)
        sun_az, sun_el = coords.get_sun_altaz(T0)
        tr = plan_escape(180.0, 50.0, T0, site, sun_safe=_unsafe_at(180.0, 50.0))
        assert tr is not None and tr.safe
        assert tr.az_to != pytest.approx(-180.0)
        _, az_path, el_path, times = transitions_module._default_slew_safe(
            _always(True), site
        ).evaluate(180.0, 50.0, tr.az_to, tr.el_to, T0)
        path_sun_az, path_sun_el = coords.get_sun_altaz(times)
        seps = coords.angular_separation(az_path, el_path, path_sun_az, path_sun_el)
        assert seps.min() >= coords.angular_separation(180.0, 50.0, sun_az, sun_el) - 0.5

    def test_needs_an_evaluating_path_model(self, site):
        with pytest.raises(ValueError, match="evaluate"):
            plan_escape(
                180.0, 50.0, T0, site, sun_safe=_always(False), slew_safe=_path_always(True)
            )

    def test_a_safe_pose_answers_without_a_path_model(self, site):
        """Answer the overtaken question from the point model alone.

        Whether a pose is inside the zone is a point question, so a
        caller holding a path model that cannot sample a path still gets
        an answer when the answer is no.
        """
        assert (
            plan_escape(180.0, 50.0, T0, site, sun_safe=_always(True), slew_safe=_path_always(True))
            is None
        )

    def test_a_floor_above_the_current_elevation_does_not_force_a_climb(self, site):
        """``el_floor`` bounds the descent; it never becomes the only candidate.

        The floor exists so an escape does not park below the sky the
        caller will observe. Clamping the search start up to it instead
        skipped the current elevation, which is where an escape is
        cheapest and where the search is documented to begin.
        """
        tr = plan_escape(180.0, 50.0, T0, site, sun_safe=_unsafe_at(180.0, 50.0), el_floor=70.0)
        assert tr is not None and tr.safe
        assert tr.el_to == 50.0

    def test_an_azimuth_window_narrower_than_a_half_turn_still_has_candidates(self, site):
        """The half-turn grid is anchored on the window, so it never empties.

        A grid of multiples of 180 contains none for a window like
        ``[10, 100]``, which left only the current azimuth: the search
        could then move in elevation but never in azimuth, and since
        descending toward a low Sun closes on it, the telescope came out
        trapped. No such site exists today; this pins the generalisation.
        """
        narrow = dataclasses.replace(site.telescope_limits.azimuth, min=10.0, max=100.0)
        limits = dataclasses.replace(site.telescope_limits, azimuth=narrow)
        narrow_site = dataclasses.replace(site, telescope_limits=limits)

        tr = plan_escape(50.0, 50.0, T0, narrow_site, sun_safe=_unsafe_at(50.0, 50.0))

        assert tr is not None and tr.safe
        assert narrow.is_in_range(tr.az_to)
        # The move is in azimuth at the current elevation, away from the
        # Sun's azimuth (83.8 deg at T0), not a step down in elevation.
        assert tr.el_to == 50.0
        assert tr.az_to == 10.0

    def test_argument_rules(self, site):
        times = T0 + TimeDelta([0.0, 10.0], format="sec")
        with pytest.raises(ValueError, match="scalar"):
            plan_escape(180.0, 50.0, times, site, sun_safe=_always(False))
        with pytest.raises(ValueError, match="settle_time"):
            plan_escape(180.0, 50.0, T0, site, sun_safe=_always(False), settle_time=-1.0)

    def test_real_scalar_model_at_night_needs_no_escape(self):
        """The default model at a night-time pose returns None without a search."""
        site = get_fyst_site()
        night = Time("2026-06-15T04:00:00", scale="utc")
        assert plan_escape(180.0, 50.0, night, site) is None

    def test_a_directional_model_is_ranked_by_its_own_threshold(self, site):
        """A zone that demands more separation in one sector is not ranked by separation.

        The anti-solar pose is 110.8 deg from the Sun at ``T0``, the
        farthest any candidate gets, but it sits in the sector this fake
        model requires 109 deg in, so it clears the requirement by 1.8 deg;
        the southern pose is 79.2 deg away in a sector requiring 75 and
        clears by 4.2. Ranking by raw separation therefore parks the
        telescope at the pose closest to the model's own limit, which is
        what the directional zone the escape exists for actually does.
        """
        coords = Coordinates(site)
        directional = _DirectionalFake(coords)

        def separation_only(az, el, t):
            return directional(az, el, t)

        margin_ranked = plan_escape(90.0, 50.0, T0, site, sun_safe=directional)
        separation_ranked = plan_escape(90.0, 50.0, T0, site, sun_safe=separation_only)

        assert margin_ranked is not None and margin_ranked.safe
        assert separation_ranked is not None and separation_ranked.safe
        # Same verdicts, same candidates: only the ordering rule differs.
        assert separation_ranked.az_to == pytest.approx(263.85, abs=0.01)
        assert margin_ranked.az_to == 180.0
        assert margin_ranked.el_to == 50.0

        def margin(tr):
            sun_az, sun_el = coords.get_sun_altaz(T0)
            sep = float(coords.angular_separation(tr.az_to, tr.el_to, sun_az, sun_el))
            return sep - _required_separation(tr.az_to)

        assert margin(margin_ranked) > margin(separation_ranked)

    def test_a_constant_threshold_ranks_exactly_like_separation(self, site):
        """A model whose requirement is one radius picks the same pose either way.

        The scalar and cone models are constant-threshold, so the margin
        rule is the separation rule shifted by a constant; this pins that
        the extension changes nothing for them.
        """
        coords = Coordinates(site)

        class _Constant:
            def __call__(self, az, el, t):
                sun_az, sun_el = coords.get_sun_altaz(t)
                return bool(coords.angular_separation(az, el, sun_az, sun_el) > 75.0)

            def threshold(self, az, el, times):
                return np.full(np.shape(np.atleast_1d(az)), 75.0)

        def bare(az, el, t):
            return _Constant()(az, el, t)

        with_threshold = plan_escape(90.0, 50.0, T0, site, sun_safe=_Constant())
        without = plan_escape(90.0, 50.0, T0, site, sun_safe=bare)
        assert with_threshold is not None and without is not None
        assert (with_threshold.az_to, with_threshold.el_to) == (without.az_to, without.el_to)
