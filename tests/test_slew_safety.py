"""Tests for the path-level slew-safety seam (Phase 6).

The kinematic path sampler and the direct-slew evaluator run entirely
offline via the ``"scalar"`` point model; the one CAD-backed check skips
without the shared library. Geometry-engineered scenarios use an evening
time when the Sun sits mid-low in the west so through-Sun and around-Sun
azimuth sweeps discriminate cleanly.
"""

import time as _clock

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories import Coordinates, get_fyst_site
from fyst_trajectories.dispatch import SlewSafePredicate, choose_encoder_solution
from fyst_trajectories.exceptions import PointingError
from fyst_trajectories.overhead.utils import _axis_slew_time
from fyst_trajectories.sun_models import (
    _axis_positions,
    _axis_slew_duration,
    find_sun_safe_detour,
    make_slew_safe,
    make_sun_safe,
)

# Evening at FYST: the Sun mid-low (el ~31) in the west (az ~261).
T_EVENING = Time("2026-11-15T20:30:00", scale="utc")


@pytest.fixture(scope="module")
def sun_evening():
    coords = Coordinates(get_fyst_site())
    sun_az, sun_el = coords.get_sun_altaz(T_EVENING)
    assert 5.0 < sun_el < 35.0  # precondition: low Sun
    return float(sun_az), float(sun_el)


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "delta,vmax,amax",
    [
        (180.0, 3.0, 1.5),  # cruise-limited azimuth sweep
        (0.5, 3.0, 1.5),  # triangular (never reaches vmax)
        (-70.0, 1.0, 0.75),  # negative elevation-style move
        (3.0, 3.0, 1.5),  # exactly the accel+decel distance
    ],
)
def test_axis_profile_endpoints_and_duration(delta, vmax, amax):
    """Profile positions are exact at both ends and monotone.

    The duration matches the scheduler's trapezoidal slew-time formula
    exactly, and the per-sample speed never exceeds the axis limit.
    """
    duration = _axis_slew_duration(abs(delta), vmax, amax)
    assert duration == pytest.approx(_axis_slew_time(abs(delta), vmax, amax))
    t = np.linspace(0.0, duration * 1.2, 400)  # overshoot: must hold at delta
    pos = _axis_positions(delta, vmax, amax, t)
    assert pos[0] == pytest.approx(0.0, abs=1e-12)
    assert pos[-1] == pytest.approx(delta, abs=1e-9)
    signed = np.sign(delta) * np.diff(pos)
    assert np.all(signed >= -1e-9)  # monotone toward the goal
    # Speed limit respected between samples.
    dt = t[1] - t[0]
    assert np.max(np.abs(np.diff(pos))) <= vmax * dt * 1.001


def test_axis_profile_zero_move():
    t = np.linspace(0.0, 10.0, 11)
    assert np.all(_axis_positions(0.0, 3.0, 1.5, t) == 0.0)


# ---------------------------------------------------------------------------
# Direct-slew evaluation (scalar model, offline)
# ---------------------------------------------------------------------------


def test_slew_through_sun_blocked_around_sun_clear(sun_evening):
    """An endpoint-safe sweep THROUGH the Sun's azimuth is rejected.

    The sweep the other way around is clear; the point seam provably
    cannot make this call (both endpoints pass it).
    """
    sun_az, sun_el = sun_evening
    el = 25.0  # within telescope limits; ~|25 - sun_el| from the Sun at crossing
    slew_safe = make_slew_safe("scalar")
    start_az = sun_az - 80.0
    goal_az = sun_az + 80.0

    point = make_sun_safe("scalar")
    assert point(start_az, el, T_EVENING) and point(goal_az, el, T_EVENING)  # endpoints safe

    assert not slew_safe(start_az, el, goal_az, el, T_EVENING)  # crosses the Sun
    # Same sky goal approached from the other side (no Sun crossing).
    assert slew_safe(sun_az + 200.0, el, goal_az, el, T_EVENING)


def test_slew_evaluate_returns_path_and_arrival(sun_evening):
    sun_az, _ = sun_evening
    slew_safe = make_slew_safe("scalar")
    safe, az_path, el_path, times = slew_safe.evaluate(100.0, 30.0, 130.0, 60.0, T_EVENING)
    assert az_path[0] == pytest.approx(100.0) and az_path[-1] == pytest.approx(130.0)
    assert el_path[0] == pytest.approx(30.0) and el_path[-1] == pytest.approx(60.0)
    # Arrival respects the slower axis (el: 30 deg at 1.0 deg/s + accel).
    assert (times[-1] - times[0]).sec == pytest.approx(
        _axis_slew_duration(30.0, 1.0, 0.75), abs=1e-6
    )
    with pytest.raises(ValueError, match="scalar start time"):
        slew_safe.evaluate(0.0, 30.0, 10.0, 30.0, Time([T_EVENING.isot, T_EVENING.isot]))


def test_make_slew_safe_validation_and_protocol():
    predicate = make_slew_safe("scalar")
    assert isinstance(predicate, SlewSafePredicate)
    with pytest.raises(ValueError, match="az_speed"):
        make_slew_safe("scalar", az_speed=0.0)
    with pytest.raises(ValueError, match="step_seconds"):
        make_slew_safe("scalar", step_seconds=-1.0)
    # A prebuilt point model is accepted; combining it with model kwargs is not.
    point = make_sun_safe("scalar")
    assert make_slew_safe(point)(100.0, 45.0, 120.0, 45.0, T_EVENING)
    with pytest.raises(ValueError, match="already-built"):
        make_slew_safe(point, maxoffset=0.65)


def test_plain_scalar_predicate_swept_per_sample(sun_evening):
    """A batch-less SunSafePredicate is accepted and consulted per path sample."""
    sun_az, _ = sun_evening
    calls = []

    def plain_point(az, el, time):
        calls.append(1)
        # Unsafe within 20 deg of the Sun's azimuth at any elevation.
        return abs(((az - sun_az) + 180.0) % 360.0 - 180.0) > 20.0

    slew_safe = make_slew_safe(plain_point)
    assert not slew_safe(sun_az - 60.0, 30.0, sun_az + 60.0, 30.0, T_EVENING)
    assert calls  # the per-sample loop actually ran


def test_slew_perf_budget():
    """A worst-case 540-deg encoder sweep evaluates well under the budget."""
    slew_safe = make_slew_safe("scalar")
    slew_safe(0.0, 45.0, 10.0, 45.0, T_EVENING)  # warm astropy caches
    t0 = _clock.perf_counter()
    slew_safe(-180.0, 20.0, 360.0, 90.0, T_EVENING)
    elapsed = _clock.perf_counter() - t0
    # Measured ~40-80 ms locally (one vectorized ephemeris over ~190 samples);
    # the generous bound absorbs CI jitter without hiding a per-sample
    # regression (which would cost seconds).
    assert elapsed < 1.0


# ---------------------------------------------------------------------------
# choose_encoder_solution integration
# ---------------------------------------------------------------------------


def test_wrap_ranking_prefers_path_safe(sun_evening):
    """The nearer wrap crosses the Sun; with slew_safe the safe far wrap wins."""
    sun_az, _ = sun_evening
    site = get_fyst_site()
    goal_el = 25.0
    goal_sky_az = (sun_az + 90.0) % 360.0  # ~335: images at 335 and -25
    current_az = sun_az - 65.0  # ~180: nearer image (335) crosses the Sun

    without = choose_encoder_solution(current_az, goal_el, goal_sky_az, goal_el, T_EVENING, site)
    with_path = choose_encoder_solution(
        current_az,
        goal_el,
        goal_sky_az,
        goal_el,
        T_EVENING,
        site,
        slew_safe=make_slew_safe("scalar", site=site),
    )
    assert without == pytest.approx((goal_sky_az, goal_el))  # nearest wrap
    assert with_path == pytest.approx((goal_sky_az - 360.0, goal_el))  # path-safe wrap


def test_all_paths_blocked_raises_with_detour_hint(sun_evening):
    sun_az, _ = sun_evening
    goal_sky_az = (sun_az + 150.0) % 360.0  # far from the Sun: point-safe goal

    class _BlockAllPaths:
        def __call__(self, current_az, current_el, goal_az, goal_el, time):
            return False

    with pytest.raises(PointingError, match="may be available.*find_sun_safe_detour"):
        choose_encoder_solution(
            180.0,
            45.0,
            goal_sky_az,
            45.0,
            T_EVENING,
            get_fyst_site(),
            slew_safe=_BlockAllPaths(),
        )


def test_slew_safe_none_unchanged(sun_evening):
    sun_az, _ = sun_evening
    site = get_fyst_site()
    goal_sky_az = (sun_az + 150.0) % 360.0  # far from the Sun: point-safe goal
    base = choose_encoder_solution(190.0, 45.0, goal_sky_az, 45.0, T_EVENING, site)
    assert base == pytest.approx((goal_sky_az, 45.0))


# ---------------------------------------------------------------------------
# Detour search
# ---------------------------------------------------------------------------


def test_detour_climbs_over_a_low_sun_under_a_narrow_policy(sun_evening):
    """Under a narrow zone the detour climbs over the low Sun and verifies.

    FYST's own 45 deg radius admits no same-wrap vertical detour (see the
    None test below), so the working domain is exercised with a narrow
    15 deg policy: the direct traverse at el 25 crosses inside the zone,
    and the detour must raise the crossing elevation just enough to
    clear, staying commandable, with both legs verifying under
    arrival-time propagation.
    """
    sun_az, sun_el = sun_evening
    narrow_site = get_fyst_site(sun_exclusion_radius=15.0, sun_warning_radius=20.0)
    slew_safe = make_slew_safe("scalar", site=narrow_site)
    start_az, goal_az = sun_az - 80.0, sun_az + 80.0
    el = 25.0
    assert not slew_safe(start_az, el, goal_az, el, T_EVENING)  # direct is blocked

    detour = find_sun_safe_detour(start_az, el, goal_az, el, T_EVENING, slew_safe, site=narrow_site)
    assert detour is not None
    az_mid, el_mid = detour
    assert az_mid == pytest.approx((start_az + goal_az) / 2.0)
    limits = narrow_site.telescope_limits.elevation
    assert limits.min <= el_mid <= limits.max
    assert el_mid > sun_el  # climbs over the low Sun
    # Both legs verify under the same evaluator, arrival-time propagated.
    ok1, _, _, times1 = slew_safe.evaluate(start_az, el, az_mid, el_mid, T_EVENING)
    assert ok1
    assert slew_safe(az_mid, el_mid, goal_az, goal_el=el, time=times1[-1])


def test_detour_none_under_the_fyst_policy(sun_evening):
    """FYST's 45 deg zone admits NO same-wrap two-leg detour here => None.

    Going under the Sun needs elevations below the telescope floor, and
    going over fails kinematically: azimuth travels 3x faster than
    elevation, so the Sun's azimuth is crossed while the elevation axis
    has climbed only ~28 deg (separation ~22 deg, well inside 45). The
    honest answer is None; the other azimuth wrap
    (choose_encoder_solution's slew_safe ranking) or waiting is the real
    recourse. (The shared library's detour returns uncommandable negative
    elevations here instead.)
    """
    sun_az, _ = sun_evening
    slew_safe = make_slew_safe("scalar")
    start_az, goal_az = sun_az - 80.0, sun_az + 80.0
    assert not slew_safe(start_az, 25.0, goal_az, 25.0, T_EVENING)  # direct blocked
    assert find_sun_safe_detour(start_az, 25.0, goal_az, 25.0, T_EVENING, slew_safe) is None


def test_detour_requires_evaluate():
    class _Bare:
        def __call__(self, *args):
            return False

    with pytest.raises(ValueError, match="evaluate"):
        find_sun_safe_detour(0.0, 25.0, 100.0, 25.0, T_EVENING, _Bare())


def test_cad_slew_end_to_end():
    """The CAD point model drives the path evaluator (skips without the lib)."""
    pytest.importorskip("sun_avoidance", exc_type=ImportError)
    slew_safe = make_slew_safe("cad")
    assert isinstance(slew_safe, SlewSafePredicate)
    verdict = slew_safe(100.0, 45.0, 140.0, 45.0, T_EVENING)
    assert isinstance(verdict, bool)
