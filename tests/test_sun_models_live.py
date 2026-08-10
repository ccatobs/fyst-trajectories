"""Live half of the sun-model parity harness (needs the shared library).

Skips entirely when ``sun_avoidance`` is not installed (the offline half in
``test_sun_models_fixture.py`` still runs). With it installed, this file is
the drift detector: it regenerates every recorded verdict/threshold from the
installed library through our adapter and requires bit-equality with the
checked-in fixture, plus the adapter behaviours that need real library calls
(protocol conformance, the el=90 nudge, wrap invariance, threshold
semantics, guards).

Re-cutting the fixture after a DELIBERATE re-pin::

    python tests/test_sun_models_live.py

(applies the same IERS pin as conftest so the record stays reproducible).
"""

import numpy as np
import pytest
from astropy.time import Time

sun_avoidance = pytest.importorskip("sun_avoidance", exc_type=ImportError)

from test_sun_models_fixture import FIXTURE, MODEL_KEYS  # noqa: E402

from fyst_trajectories import (  # noqa: E402
    Coordinates,
    get_fyst_site,
    sun_models,
)
from fyst_trajectories.dispatch import SunSafePredicate  # noqa: E402
from fyst_trajectories.exceptions import PointingWarning  # noqa: E402
from fyst_trajectories.sun_models import load_avoidance_data, make_sun_safe  # noqa: E402

T_REF = Time("2026-11-15T16:00:00", scale="utc")


def _model_set():
    """Build the recorded model configurations, in fixture order.

    The scalar row pins its radii explicitly (45/50) rather than reading the
    site defaults, so a future default bump does not force a re-cut of a
    fixture whose purpose is locking the LIBRARY's behaviour.
    """
    pinned_site = get_fyst_site(sun_exclusion_radius=45.0, sun_warning_radius=50.0)
    return {
        "scalar": make_sun_safe("scalar", site=pinned_site),
        "cone45": make_sun_safe("cone", radius=45.0),
        "cone50": make_sun_safe("cone", radius=50.0),
        "cad": make_sun_safe("cad"),
        "cad_msa0": make_sun_safe("cad", min_solar_altitude=0.0),
        "cad_island": make_sun_safe("cad", island_check=True),
    }


def _record(az, el, times_iso):
    """Verdicts + thresholds for every model over the fixture grid."""
    A, E = np.meshgrid(az, el)
    A, E = A.ravel(), E.ravel()
    out = {}
    for name, predicate in _model_set().items():
        verdicts, thresholds = [], []
        for iso in times_iso:
            t = Time(str(iso), scale="utc")
            verdicts.append(predicate.batch(A, E, t))
            thresholds.append(predicate.threshold(A, E, t))
        out[f"verdict_{name}"] = np.stack(verdicts)
        out[f"threshold_{name}"] = np.stack(thresholds)
    return out


def test_no_drift_against_fixture():
    """Regenerated verdicts/thresholds are bit-identical to the record.

    A failure here means the installed sun-avoidance revision (or our
    adapter) no longer produces the pinned verdicts: re-pin DELIBERATELY
    (update SUN_AVOIDANCE_PINNED_SHA / CAD_TABLE_SHA256, re-cut the fixture
    via ``python tests/test_sun_models_live.py``) rather than mixing
    revisions.
    """
    fx = np.load(FIXTURE)
    regenerated = _record(fx["az"], fx["el"], fx["times_iso"])
    for key in MODEL_KEYS:
        for kind in ("verdict", "threshold"):
            name = f"{kind}_{key}"
            assert np.array_equal(regenerated[name], fx[name]), (
                f"{name} drifted from the fixture recorded at "
                f"sun-avoidance {str(fx['pinned_sha'])[:12]}"
            )
    # The recorded CAD table itself matches the installed one.
    installed = load_avoidance_data("cad")
    assert np.array_equal(fx["cad_gamma"], np.asarray(installed.gamma, dtype=float))
    assert np.array_equal(fx["cad_delta"], np.asarray(installed.delta, dtype=float))


def test_satisfies_sun_safe_predicate_contract():
    for predicate in _model_set().values():
        assert isinstance(predicate, SunSafePredicate)
        verdict = predicate(180.0, 45.0, T_REF)
        assert isinstance(verdict, bool)
    # The scalar-call contract is one point; arrays must go through batch().
    grid = Time(["2026-11-15T16:00:00", "2026-11-15T16:10:00", "2026-11-15T16:20:00"], scale="utc")
    with pytest.raises(ValueError, match="batch"):
        make_sun_safe("cad")(180.0, 45.0, grid)


def test_el90_clamp_matches_stable_near_zenith_geometry():
    """el=90.0 evaluates like the numerically STABLE near-zenith geometry.

    The reference elevation 89.99 sits outside the clamp band and well clear
    of the library's degenerate final-ULPs-below-90 region (where the clock
    angle quantises to multiples of 90 and the threshold can collapse to the
    table floor). Thresholds at el=90 must match it exactly over a full
    azimuth sweep at several epochs; an infinitesimal-nudge implementation
    fails this with floor-collapsed thresholds and false-SAFE verdicts.
    """
    predicate = make_sun_safe("cad")
    az = np.arange(0.0, 360.0, 5.0)
    for iso in ("2026-11-15T10:30:00", "2026-11-15T16:00:00", "2026-12-21T12:00:00"):
        t = Time(iso, scale="utc")
        thr_90 = predicate.threshold(az, np.full_like(az, 90.0), t)
        thr_ref = predicate.threshold(az, np.full_like(az, 89.99), t)
        assert np.array_equal(thr_90, thr_ref)
        assert np.array_equal(
            predicate.batch(az, np.full_like(az, 90.0), t),
            predicate.batch(az, np.full_like(az, 89.99), t),
        )
        # And the thresholds are genuinely directional, not the collapsed floor.
        assert len(np.unique(thr_90)) > 1


def test_cone45_matches_is_sun_safe_live():
    """The adapter's cone-45 reproduces Coordinates.is_sun_safe point-for-point.

    The site is pinned at 45/50 (the fixture's convention) so this parity
    check is independent of the current site defaults.
    """
    site = get_fyst_site(sun_exclusion_radius=45.0, sun_warning_radius=50.0)
    coords = Coordinates(site)
    predicate = make_sun_safe("cone", radius=45.0, site=site)
    rng_az = np.linspace(-180.0, 359.0, 40)
    rng_el = np.linspace(20.0, 90.0, 5)
    for t in (T_REF, Time("2027-02-15T06:00:00", scale="utc")):
        for el in rng_el:
            ours = np.array([coords.is_sun_safe(a, el, t) for a in rng_az])
            theirs = predicate.batch(rng_az, np.full_like(rng_az, el), t)
            assert np.array_equal(ours, theirs)


def test_wrap_invariance():
    predicate = make_sun_safe("cad")
    az = np.arange(0.0, 360.0, 7.5)
    el = np.full_like(az, 45.0)
    base = predicate.batch(az, el, T_REF)
    assert np.array_equal(base, predicate.batch(az + 360.0, el, T_REF))
    assert np.array_equal(base, predicate.batch(az - 360.0, el, T_REF))


@pytest.mark.parametrize("kwargs", [{}, {"maxoffset": 0.65}, {"tracking_module": "primecam_f280"}])
def test_threshold_is_geometric_boundary(kwargs):
    """Assert verdict == (geometric separation > threshold), padding included."""
    site = get_fyst_site()
    coords = Coordinates(site)
    predicate = make_sun_safe("cad", site=site, **kwargs)
    az = np.arange(-180.0, 360.0, 12.5)
    el = np.linspace(20.0, 90.0, az.size)
    sun_az, sun_el = coords.get_sun_altaz(T_REF)
    separation = np.atleast_1d(coords.angular_separation(az, el, sun_az, sun_el))
    verdict = predicate.batch(az, el, T_REF)
    threshold = predicate.threshold(az, el, T_REF)
    assert np.array_equal(verdict, separation > threshold)


def test_night_exemption_semantics():
    """min_solar_altitude=0 waives the zone in twilight; the default never waives.

    The step only matters when the Sun is *just* below the horizon: in deep
    night even a pointing at the Sun's azimuth clears the 50 deg floor from
    the elevation floor up (the Sun is 40+ deg down), so the probe must use
    twilight geometry.
    """
    t_twilight = Time("2026-11-15T23:10:00", scale="utc")  # shortly after FYST sunset
    coords = Coordinates(get_fyst_site())
    sun_az, sun_el = coords.get_sun_altaz(t_twilight)
    assert -10.0 < sun_el < -0.8  # precondition: geometrically set, still shallow
    # A pointing at the Sun's azimuth at el 20 sits ~25 deg from the Sun,
    # well inside the CAD floor: the default (never waive) rejects it while
    # the library-default exemption accepts the whole sky.
    exempt = make_sun_safe("cad", min_solar_altitude=0.0)
    strict = make_sun_safe("cad")
    az = np.array([sun_az, sun_az + 90.0, sun_az + 180.0])
    el = np.array([20.0, 45.0, 70.0])
    assert exempt.batch(az, el, t_twilight).all()
    verdicts = strict.batch(az, el, t_twilight)
    assert not verdicts[0]  # at the Sun's azimuth: inside the zone
    assert not verdicts.all()


def test_unknown_tracking_module_warns():
    with pytest.warns(PointingWarning, match="ZERO"):
        make_sun_safe("cad", tracking_module="primecam_f850")
    # Known-padded and center names stay silent.
    make_sun_safe("cad", tracking_module="primecam_f280")
    make_sun_safe("cad")


def test_island_check_forbids_reachability_risk():
    """island_check=True rejects a zone-safe position on the Sun's far side.

    The sample is taken from the recorded fixture (cad safe, cad_island
    unsafe): a mid-elevation pointing above the below-horizon Sun's azimuth
    band that the library's forbidden-island line marks as at risk of
    becoming isolated. Locks that the flag actually reaches the library's
    island geometry rather than merely tightening thresholds.
    """
    t = Time("2026-11-15T00:00:00", scale="utc")
    point = dict(az_deg=-82.5, el_deg=30.0, times=t)
    assert make_sun_safe("cad").batch(**point)[0]
    assert not make_sun_safe("cad", island_check=True).batch(**point)[0]


def test_library_models_honor_disabled_site():
    """site.sun_avoidance.enabled=False disables EVERY model uniformly."""
    site = get_fyst_site(sun_avoidance_enabled=False)
    predicate = make_sun_safe("cad", site=site)
    az = np.array([0.0, 120.0, 240.0])
    el = np.array([20.0, 45.0, 90.0])
    assert predicate.batch(az, el, T_REF).all()
    assert np.all(predicate.threshold(az, el, T_REF) == 0.0)
    assert predicate.describe.endswith("(disabled)")


def test_guards_reject_bad_inputs():
    predicate = make_sun_safe("cad")
    with pytest.raises(ValueError, match="scalar or 1-D"):
        predicate.batch(np.zeros((2, 2)), np.zeros((2, 2)), T_REF)
    with pytest.raises(ValueError, match=r"\[-90, 90\]"):
        predicate.batch([120.0], [100.0], T_REF)  # over-the-top encoding
    with pytest.raises(ValueError, match="min_solar_altitude"):
        make_sun_safe("cad", min_solar_altitude=91.0)
    with pytest.raises(ValueError, match="radius"):
        make_sun_safe("cone", radius=200.0)


def test_scalar_time_sun_cache_populates():
    """Repeated scalar-time calls hit the per-predicate Sun cache."""
    predicate = make_sun_safe("cad")
    for _ in range(3):
        predicate(180.0, 45.0, T_REF)
        predicate(90.0, 60.0, T_REF)
    assert len(predicate._sun_cache) == 1  # one unique instant, computed once


def test_sun_cache_is_scale_invariant():
    """A TAI Time sharing a UTC Time's numeric jd must not hit its entry.

    ``Time(iso, "utc").jd == Time(iso, "tai").jd`` although the instants
    are 37 s apart; a jd-keyed cache would answer the TAI query with the
    UTC instant's Sun. The key is the TAI jd, so different instants get
    different entries and ONE instant expressed in two scales shares one.
    """
    t_utc = Time("2026-12-21T16:00:00", scale="utc")
    t_tai = Time("2026-12-21T16:00:00", scale="tai")
    assert t_utc.jd == t_tai.jd  # the collision this test guards against
    warmed = make_sun_safe("cad")
    warmed(180.0, 45.0, t_utc)  # warm with the UTC instant
    cold = make_sun_safe("cad")
    az = np.arange(-180.0, 360.0, 2.5)
    el = np.full_like(az, 30.0)
    assert np.array_equal(warmed.batch(az, el, t_tai), cold.batch(az, el, t_tai))
    assert len(warmed._sun_cache) == 2  # two distinct instants
    warmed(180.0, 45.0, t_utc.tai)  # same instant as t_utc, different scale
    assert len(warmed._sun_cache) == 2  # merged into the existing entry


def test_cad_sha_pin_enforced(monkeypatch):
    monkeypatch.setattr(sun_models, "CAD_TABLE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="re-pin"):
        load_avoidance_data("cad")


def test_load_validation_with_library():
    with pytest.raises(ValueError, match="radius"):
        load_avoidance_data("cad", radius=45.0)
    with pytest.raises(ValueError, match="finite radius"):
        load_avoidance_data("cone", radius=float("nan"))
    cone = load_avoidance_data("cone", radius=50.0)
    assert cone.deltaMin == cone.deltaMax == 50.0


if __name__ == "__main__":
    # Fixture re-cut path: same IERS pin as conftest, then record and write.
    from pathlib import Path

    from astropy.utils import iers

    data_dir = Path(__file__).parent / "data"
    iers.conf.auto_download = False
    iers.earth_orientation_table.set(iers.IERS_A.open(str(data_dir / "finals2000A.all")))
    iers.conf.iers_degraded_accuracy = "warn"

    from fyst_trajectories.sun_models import CAD_TABLE_SHA256, SUN_AVOIDANCE_PINNED_SHA

    az = np.arange(-180.0, 360.0, 7.5)
    el = np.arange(20.0, 90.1, 2.5)
    times_iso = [
        "2026-11-15T00:00:00",
        "2026-11-15T06:00:00",
        "2026-11-15T12:00:00",
        "2026-11-15T18:00:00",
        "2026-12-21T12:00:00",
        "2027-02-15T06:00:00",
        "2027-03-15T18:00:00",
        "2027-05-15T12:00:00",
    ]
    cad = load_avoidance_data("cad")
    payload = {
        "az": az,
        "el": el,
        "times_iso": np.array(times_iso),
        "pinned_sha": np.array(SUN_AVOIDANCE_PINNED_SHA),
        "cad_table_sha256": np.array(CAD_TABLE_SHA256),
        "cad_gamma": np.asarray(cad.gamma, dtype=float),
        "cad_delta": np.asarray(cad.delta, dtype=float),
        **_record(az, el, times_iso),
    }
    np.savez_compressed(FIXTURE, **payload)
    print(f"re-cut {FIXTURE}")
