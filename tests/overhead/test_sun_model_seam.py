"""Tests for the overhead sun-model seam and the boundary unification.

Covers the sun-model seam end to end: every overhead sun verdict site
accepts an injected :class:`~fyst_trajectories.dispatch.SunSafePredicate`,
and the scalar-mode boundary is unified on "exactly at the radius is
UNSAFE" (matching ``Coordinates.is_sun_safe``), closing the old ``<`` /
``>`` divergence.
"""

import numpy as np
import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories import Coordinates, get_fyst_site
from fyst_trajectories.overhead import (
    MinDurationConstraint,
    ObservingPatch,
    SunAvoidanceConstraint,
    generate_timeline,
    get_observable_windows,
)
from fyst_trajectories.overhead.scheduler.helpers import (
    _default_constraints,
    _time_until_sun_unsafe,
)
from fyst_trajectories.overhead.utils import _filter_sun_unsafe
from fyst_trajectories.sun_models import make_sun_safe

T_DAY = Time("2026-11-15T16:00:00", scale="utc")  # Sun up at FYST


def _patch(name="p", ra=24.0, dec=-32.0, scan_type="pong"):
    return ObservingPatch(
        name=name,
        ra_center=ra,
        dec_center=dec,
        width=10.0,
        height=10.0,
        scan_type=scan_type,
        velocity=1.0,
        elevation=50.0,
    )


class _StubPredicate:
    """Scripted verdicts; counts batch/scalar consultations."""

    def __init__(self, verdict=True):
        self.verdict = verdict
        self.batch_calls = 0
        self.scalar_calls = 0

    def __call__(self, az, el, time):
        self.scalar_calls += 1
        return bool(self.verdict)

    def batch(self, az, el, times):
        self.batch_calls += 1
        n = np.shape(np.atleast_1d(az))
        return np.full(n, self.verdict, dtype=bool)


# ---------------------------------------------------------------------------
# Boundary unification (scalar mode): exactly at the radius is UNSAFE
# ---------------------------------------------------------------------------


def test_boundary_unified_at_exact_radius(coordinates):
    """is_sun_safe, the constraints, and the clip helpers agree at sep == radius."""
    coords = coordinates
    t = T_DAY
    az, el = 180.0, 45.0
    sun_az, sun_el = coords.get_sun_altaz(t)
    sep = coords.angular_separation(az, el, sun_az, sun_el)

    # A site whose exclusion radius equals this exact separation.
    site_eq = get_fyst_site(sun_exclusion_radius=sep, sun_warning_radius=sep + 5.0)
    assert not Coordinates(site_eq).is_sun_safe(az, el, t)  # the reference convention

    constraint = SunAvoidanceConstraint(min_angle=sep)
    assert constraint.score(_patch(), t, az, el, coords) == 0.0

    # _time_until_sun_unsafe: first sample exactly at the radius => 0 s left.
    ra, dec = coords.altaz_to_radec(az, el, t)
    assert (
        _time_until_sun_unsafe(ra, dec, t, 600.0, coords, min_sun_angle=sep, step_seconds=60.0)
        == 0.0
    )

    # _filter_sun_unsafe: the window's first sample is unsafe at the radius.
    windows = _filter_sun_unsafe(ra, dec, t, t + TimeDelta(600, format="sec"), coords, sep)
    assert all(abs((w0 - t).sec) > 1.0 for w0, _ in windows)


def test_min_duration_constraint_boundary(coordinates):
    """MinDurationConstraint's sun forward-check is unsafe at the exact radius."""
    coords = coordinates
    t = T_DAY
    patch = _patch()
    constraint = MinDurationConstraint(min_duration=60.0)
    future = t + TimeDelta(60.0, format="sec")
    f_az, f_el = coords.radec_to_altaz(patch.ra_center, patch.dec_center, future)
    sun_az, sun_el = coords.get_sun_altaz(future)
    sep_future = coords.angular_separation(float(f_az), float(f_el), sun_az, sun_el)

    site_eq = get_fyst_site(sun_exclusion_radius=sep_future, sun_warning_radius=sep_future + 5.0)
    coords_eq = Coordinates(site_eq)
    az, el = coords_eq.radec_to_altaz(patch.ra_center, patch.dec_center, t)
    assert constraint.score(patch, t, float(az), float(el), coords_eq) == 0.0


# ---------------------------------------------------------------------------
# SunAvoidanceConstraint modes
# ---------------------------------------------------------------------------


def test_sun_constraint_requires_exactly_one_mode():
    with pytest.raises(ValueError, match="exactly one"):
        SunAvoidanceConstraint()
    with pytest.raises(ValueError, match="exactly one"):
        SunAvoidanceConstraint(min_angle=45.0, sun_safe=_StubPredicate())
    with pytest.raises(ValueError, match="non-negative"):
        SunAvoidanceConstraint(min_angle=-1.0)


def test_sun_constraint_injected_model_drives_score(coordinates):
    allow = _StubPredicate(verdict=True)
    block = _StubPredicate(verdict=False)
    assert (
        SunAvoidanceConstraint(sun_safe=allow).score(_patch(), T_DAY, 180.0, 45.0, coordinates)
        == 1.0
    )
    assert (
        SunAvoidanceConstraint(sun_safe=block).score(_patch(), T_DAY, 180.0, 45.0, coordinates)
        == 0.0
    )
    assert allow.scalar_calls == 1 and block.scalar_calls == 1


def test_min_duration_injected_model(coordinates):
    block = _StubPredicate(verdict=False)
    constraint = MinDurationConstraint(min_duration=60.0, sun_safe=block)
    patch = _patch()
    az, el = coordinates.radec_to_altaz(patch.ra_center, patch.dec_center, T_DAY)
    # The elevation forward-check must pass for the sun branch to be reached.
    if float(el) >= coordinates.site.telescope_limits.elevation.min:
        assert constraint.score(patch, T_DAY, float(az), float(el), coordinates) == 0.0
        assert block.scalar_calls == 1


def test_default_constraints_bind_the_model():
    site = get_fyst_site()
    stub = _StubPredicate()
    constraints = _default_constraints(site, sun_safe=stub)
    sun_constraints = [c for c in constraints if isinstance(c, SunAvoidanceConstraint)]
    assert len(sun_constraints) == 1
    assert sun_constraints[0].sun_safe is stub
    # Without a model the scalar radius comes from the site, not a literal.
    scalar = _default_constraints(site)
    sun_scalar = [c for c in scalar if isinstance(c, SunAvoidanceConstraint)][0]
    assert sun_scalar.min_angle == site.sun_avoidance.exclusion_radius


# ---------------------------------------------------------------------------
# _time_until_sun_unsafe with an injected model
# ---------------------------------------------------------------------------


class _TimeGatedModel:
    """Verdicts flip at a scripted offset from a reference time.

    Shape-independent by construction: the verdict is a pure function of
    the sample time, so it stands in for ANY model - including the CAD
    staircase, whose threshold steps a whole table level inside a bracket
    and made margin interpolation anti-conservative (the reproducer that
    motivated verdict bisection: booked ~42-51 s of science past the
    model's own unsafe onset).
    """

    def __init__(self, t0, safe_before_sec):
        self._t0 = t0
        self._cut = safe_before_sec
        self.batch_calls = 0

    def __call__(self, az, el, time):
        return bool((time - self._t0).sec < self._cut)

    def batch(self, az, el, times):
        self.batch_calls += 1
        offsets = np.atleast_1d((times - self._t0).sec)
        return offsets < self._cut


def test_time_until_sun_unsafe_bisection_finds_the_true_crossing(coordinates):
    """The crossing is verdict-bisected to sub-step precision, conservatively.

    The scripted flip at 137.4 s sits mid-bracket (samples at 120/180 s);
    bisection must land just below it (last VERIFIED-safe time), never
    beyond it - regardless of how the model's internal threshold behaves.
    """
    coords = coordinates
    ra, dec = coords.altaz_to_radec(180.0, 45.0, T_DAY)
    model = _TimeGatedModel(T_DAY, 137.4)

    out = _time_until_sun_unsafe(
        ra, dec, T_DAY, 300.0, coords, 45.0, step_seconds=60.0, sun_safe=model
    )
    assert out == pytest.approx(137.4, abs=0.2)
    assert out < 137.4  # conservative: strictly before the flip
    assert model.batch_calls >= 1  # grid pass + bisection probes


def test_time_until_sun_unsafe_batch_shape_guard(coordinates):
    """A short batch result must raise, never broadcast over the grid."""
    coords = coordinates
    ra, dec = coords.altaz_to_radec(180.0, 45.0, T_DAY)

    class _Short:
        def __call__(self, az, el, time):
            return True

        def batch(self, az, el, times):
            return np.ones(1, dtype=bool)

    with pytest.raises(ValueError, match="sun_safe.batch"):
        _time_until_sun_unsafe(ra, dec, T_DAY, 300.0, coords, 45.0, sun_safe=_Short())


def test_time_until_sun_unsafe_scalar_fallback_without_batch(coordinates):
    """A plain SunSafePredicate (no batch attr) drives the grid AND bisection."""
    coords = coordinates
    ra, dec = coords.altaz_to_radec(180.0, 45.0, T_DAY)
    calls = []
    cut = 150.0

    def plain_predicate(az, el, time):
        calls.append(1)
        return (time - T_DAY).sec < cut

    out = _time_until_sun_unsafe(
        ra, dec, T_DAY, 300.0, coords, 45.0, step_seconds=60.0, sun_safe=plain_predicate
    )
    assert len(calls) == 6 + 10  # grid samples + bisection probes
    assert out == pytest.approx(cut, abs=0.2)
    assert out < cut  # conservative side of the flip


def test_time_until_sun_unsafe_all_safe_and_all_unsafe(coordinates):
    coords = coordinates
    ra, dec = coords.altaz_to_radec(180.0, 45.0, T_DAY)
    allow = _StubPredicate(verdict=True)
    block = _StubPredicate(verdict=False)
    assert _time_until_sun_unsafe(ra, dec, T_DAY, 300.0, coords, 45.0, sun_safe=allow) == 300.0
    assert _time_until_sun_unsafe(ra, dec, T_DAY, 300.0, coords, 45.0, sun_safe=block) == 0.0
    # The vectorized path is actually taken: one batch call, zero scalar calls.
    assert allow.batch_calls == 1 and allow.scalar_calls == 0
    assert block.batch_calls == 1 and block.scalar_calls == 0


# ---------------------------------------------------------------------------
# get_observable_windows with an injected model
# ---------------------------------------------------------------------------


def test_observable_windows_scalar_model_parity():
    """The 'scalar' model reproduces the built-in scalar filtering exactly."""
    site = get_fyst_site()
    t0 = Time("2026-11-15T00:00:00", scale="utc")
    t1 = Time("2026-11-16T00:00:00", scale="utc")
    default = get_observable_windows(24.0, -32.0, t0, t1, site)
    injected = get_observable_windows(
        24.0, -32.0, t0, t1, site, sun_safe=make_sun_safe("scalar", site=site)
    )
    assert len(default) == len(injected)
    for (a0, a1), (b0, b1) in zip(default, injected):
        assert abs((a0 - b0).sec) < 1e-6
        assert abs((a1 - b1).sec) < 1e-6


def test_observable_windows_model_with_check_sun_off_raises():
    site = get_fyst_site()
    t0 = Time("2026-11-15T00:00:00", scale="utc")
    with pytest.raises(ValueError, match="check_sun"):
        get_observable_windows(
            24.0,
            -32.0,
            t0,
            t0 + TimeDelta(3600, format="sec"),
            site,
            check_sun=False,
            sun_safe=make_sun_safe("scalar", site=site),
        )


def test_observable_windows_scalar_fallback_without_batch():
    """A plain predicate (no batch) drives the window filter per sample."""
    site = get_fyst_site()
    t0 = Time("2026-11-15T00:00:00", scale="utc")
    t1 = Time("2026-11-15T12:00:00", scale="utc")
    calls = []

    def plain_predicate(az, el, time):
        calls.append(1)
        return True

    injected = get_observable_windows(24.0, -32.0, t0, t1, site, sun_safe=plain_predicate)
    default = get_observable_windows(24.0, -32.0, t0, t1, site, check_sun=False)
    assert calls  # consulted
    assert len(injected) == len(default)  # all-safe model == no sun filtering


# ---------------------------------------------------------------------------
# End-to-end scheduler parity and the CAD-subset property
# ---------------------------------------------------------------------------


def _timeline_signature(timeline):
    return [
        (b.block_type, b.patch_name, round(b.t_start.unix, 3), round(b.t_stop.unix, 3))
        for b in timeline.blocks
    ]


def test_generate_timeline_scalar_model_parity():
    """Injecting the 'scalar' model reproduces the default night exactly."""
    site = get_fyst_site()
    patches = [_patch(name="Deep56", scan_type="constant_el")]
    kwargs = dict(
        patches=patches,
        site=site,
        start_time="2026-06-15T00:00:00",
        end_time="2026-06-15T12:00:00",
    )
    default = generate_timeline(**kwargs)
    injected = generate_timeline(**kwargs, sun_safe=make_sun_safe("scalar", site=site))
    assert _timeline_signature(default) == _timeline_signature(injected)


def test_generate_timeline_cad_night_is_subset_of_cone50():
    """CAD science time is a strict subset of the cone-50 night.

    The patch rides at Jupiter's Nov-15 sky position, ~87-88 deg from the
    Sun all day: outside every cone (45/50 pass it untouched) but inside
    the directional CAD zone whenever its clock angle enters the 88-90 deg
    sectors, so the CAD run must genuinely lose science time while every
    CAD block stays inside a cone-50 block.
    """
    pytest.importorskip("sun_avoidance", exc_type=ImportError)
    site = get_fyst_site()
    coords = Coordinates(site)
    mid = Time("2026-11-15T12:00:00", scale="utc")
    jup_ra, jup_dec = coords.get_body_radec("jupiter", mid)
    patches = [_patch(name="JupField", ra=float(jup_ra), dec=float(jup_dec), scan_type="pong")]
    kwargs = dict(
        patches=patches,
        site=site,
        start_time="2026-11-15T05:00:00",
        end_time="2026-11-15T16:00:00",
    )
    cone50 = generate_timeline(**kwargs, sun_safe=make_sun_safe("cone", radius=50.0, site=site))
    cad = generate_timeline(**kwargs, sun_safe=make_sun_safe("cad", site=site))

    def science_intervals(timeline):
        return [
            (b.t_start.unix, b.t_stop.unix)
            for b in timeline.blocks
            if getattr(b.block_type, "value", b.block_type) == "science"
        ]

    cone_iv = science_intervals(cone50)
    cad_iv = science_intervals(cad)
    cone_total = sum(k1 - k0 for k0, k1 in cone_iv)
    cad_total = sum(c1 - c0 for c0, c1 in cad_iv)

    assert cone_total > 0.0  # the field is observable at all under cone-50
    # Subset invariant, robust to block re-segmentation (idle/retune
    # boundaries shift between runs): every CAD-scheduled science sample
    # must itself be cone-50-safe, since the CAD zone contains the 50 cone.
    cone_model = make_sun_safe("cone", radius=50.0, site=site)
    for c0, c1 in cad_iv:
        ts = Time([c0, (c0 + c1) / 2.0, c1], format="unix", scale="utc")
        az, el = coords.radec_to_altaz(np.full(3, float(jup_ra)), np.full(3, float(jup_dec)), ts)
        assert cone_model.batch(az, el, ts).all(), "CAD scheduled inside the cone-50 zone"
    # STRICT: the directional zone must actually bite on this geometry; a
    # CAD path wired to cone-50 (or disconnected) fails here.
    assert cad_total < cone_total - 60.0
