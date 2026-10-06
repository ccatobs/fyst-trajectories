"""Tests for the observability (OBSERVE / AVOID) primitives.

Cases are made deterministic without hand-computed ephemeris by constructing
geometry from the same primitives under test: a near-zenith FIXED source is
placed on the meridian at ``ra = LST(t)``; Sun/avoid conditions are forced by
placing a FIXED target at a body's RA/Dec, or by an oversized AVOID zone.

``T_NIGHT`` is local midnight at FYST (Sun well below the horizon) and
``T_DAY`` is ~local noon (Sun up); both are stable year-to-year for the
chosen calendar date and within the IERS prediction window.
"""

import numpy as np
import pytest
from _sun_stubs import allow_everything, block_everything
from astropy.time import Time, TimeDelta

from fyst_trajectories import Coordinates, get_fyst_site
from fyst_trajectories.coordinates import _build_time_grid
from fyst_trajectories.observability import (
    FLUX_CALIBRATORS,
    AvoidZone,
    ReasonCode,
    Target,
    TargetKind,
    _all_windows,
    check_observability,
    resolve_target,
)

T_NIGHT = Time("2026-06-15T05:00:00", scale="utc")
T_DAY = Time("2026-06-15T16:30:00", scale="utc")


def _near_zenith_fixed(coords, t, name="zen"):
    """Return a FIXED source transiting near the zenith at time ``t`` (el ~ 85 deg)."""
    lst = coords.get_lst(t)
    return Target(name, TargetKind.FIXED, ra_deg=float(lst), dec_deg=coords.site.latitude + 5.0)


def test_instant_happy_path(coordinates):
    t = T_NIGHT
    _, sun_el = coordinates.get_sun_altaz(t)
    assert sun_el < 0  # precondition: night
    tgt = _near_zenith_fixed(coordinates, t)
    r = check_observability([tgt], t, site=coordinates.site)[0]
    assert r.observable is True
    assert r.reasons == ()
    assert r.windows is None
    assert r.total_observable_hours == 0.0  # windows not evaluated => 0.0, not an error
    assert r.sun_clear is True
    assert 80.0 < r.el_deg < 90.0
    assert r.position_approximate is False


def test_horizon_window(coordinates):
    t = T_NIGHT
    tgt = _near_zenith_fixed(coordinates, t)
    r = check_observability([tgt], t, site=coordinates.site, horizon_hours=24.0)[0]
    assert r.observable is True
    assert r.windows
    first = r.windows[0]
    assert first.duration_hours > 0.0
    # Observable now => the first window opens at t and is truncated at the horizon start.
    assert first.truncated_start is True
    assert abs((first.start - t).to_value("s")) < 1.0
    assert r.total_observable_hours >= first.duration_hours


def test_below_el_min(coordinates):
    t = T_NIGHT
    # dec = +80 deg is never visible from FYST (lat ~ -23 deg): always below the horizon.
    tgt = Target("far_north", TargetKind.FIXED, ra_deg=0.0, dec_deg=80.0)
    r = check_observability([tgt], t, site=coordinates.site)[0]
    assert r.observable is False
    assert ReasonCode.BELOW_EL_MIN in r.reasons
    assert r.el_deg < 20.0


def test_above_el_max(coordinates):
    t = T_NIGHT
    tgt = _near_zenith_fixed(coordinates, t)  # el ~ 85
    r = check_observability([tgt], t, site=coordinates.site, el_max=80.0)[0]
    assert r.observable is False
    assert ReasonCode.ABOVE_EL_MAX in r.reasons


def test_sun_too_close(coordinates):
    t = T_DAY
    sun_az, sun_el = coordinates.get_sun_altaz(t)
    assert sun_el > 20.0  # precondition: Sun well up
    # Place a FIXED source at the Sun's Az/El (inverted to RA/Dec) so it
    # coincides with the Sun under the same vacuum transform.
    sun_ra, sun_dec = coordinates.altaz_to_radec(sun_az, sun_el, t)
    tgt = Target("at_sun", TargetKind.FIXED, ra_deg=sun_ra, dec_deg=sun_dec)
    r = check_observability([tgt], t, site=coordinates.site)[0]
    assert r.sun_clear is False
    assert ReasonCode.SUN_TOO_CLOSE in r.reasons
    assert r.observable is False
    assert r.sun_separation_deg == pytest.approx(0.0, abs=1e-6)


def test_avoid_pass(coordinates):
    t = T_NIGHT
    jra, jdec = coordinates.get_body_radec("jupiter", t)
    tgt = Target("away", TargetKind.FIXED, ra_deg=(jra + 120.0) % 360.0, dec_deg=-jdec)
    r = check_observability([tgt], t, site=coordinates.site, avoid=[AvoidZone("jupiter", 3.0)])[0]
    assert len(r.avoid_separations) == 1
    assert r.avoid_separations[0].body == "jupiter"
    assert r.avoid_separations[0].clear is True
    assert r.avoid_separations[0].separation_deg > 3.0
    assert ReasonCode.AVOID_TOO_CLOSE not in r.reasons


def test_avoid_fail(coordinates):
    t = T_NIGHT
    jaz, jel = coordinates.get_body_altaz("jupiter", t)
    jra, jdec = coordinates.altaz_to_radec(jaz, jel, t)
    tgt = Target("at_jup", TargetKind.FIXED, ra_deg=jra, dec_deg=jdec)
    r = check_observability([tgt], t, site=coordinates.site, avoid=[AvoidZone("jupiter", 3.0)])[0]
    assert r.avoid_separations[0].clear is False
    assert r.avoid_separations[0].separation_deg == pytest.approx(0.0, abs=1e-3)
    assert ReasonCode.AVOID_TOO_CLOSE in r.reasons
    assert r.observable is False


def test_avoid_zone_at_exactly_its_radius_is_clear(coordinates):
    """A target exactly at an AvoidZone radius is CLEAR (``sep >= zone_deg``).

    Deliberately the opposite convention to the Sun's ``sep <= radius`` is
    unsafe: an AVOID zone is a caller-supplied minimum separation, so
    standing exactly on it satisfies the request. Constructing a position at
    an exact separation is float-fragile, so the boundary is pinned from the
    other side: measure the separation, then set the zone radius to that
    number.
    """
    t = T_NIGHT
    jaz, jel = coordinates.get_body_altaz("jupiter", t)
    tgt = Target("near_jup", TargetKind.FIXED, ra_deg=0.0, dec_deg=0.0)
    az, el = coordinates.radec_to_altaz(tgt.ra_deg, tgt.dec_deg, t)
    sep = float(coordinates.angular_separation(az, el, jaz, jel))

    at_radius = check_observability(
        [tgt], t, site=coordinates.site, avoid=[AvoidZone("jupiter", sep)]
    )[0]
    assert at_radius.avoid_separations[0].clear is True
    assert ReasonCode.AVOID_TOO_CLOSE not in at_radius.reasons

    # One ULP wider and the same target is inside, so the verdict really is
    # decided at the radius.
    wider = check_observability(
        [tgt], t, site=coordinates.site, avoid=[AvoidZone("jupiter", np.nextafter(sep, 1e9))]
    )[0]
    assert wider.avoid_separations[0].clear is False
    assert ReasonCode.AVOID_TOO_CLOSE in wider.reasons


def test_both_avoidance_kinds_reported_separately(coordinates):
    t = T_DAY
    sun_az, sun_el = coordinates.get_sun_altaz(t)
    sun_ra, sun_dec = coordinates.altaz_to_radec(sun_az, sun_el, t)
    tgt = Target("at_sun", TargetKind.FIXED, ra_deg=sun_ra, dec_deg=sun_dec)
    # A zone > 180 deg (the maximum possible separation) forces the AVOID branch
    # deterministically, independent of the Moon's phase/position.
    r = check_observability([tgt], t, site=coordinates.site, avoid=[AvoidZone("moon", 181.0)])[0]
    assert r.sun_clear is False
    assert ReasonCode.SUN_TOO_CLOSE in r.reasons
    assert ReasonCode.AVOID_TOO_CLOSE in r.reasons
    moon_seps = [s for s in r.avoid_separations if s.body == "moon"]
    assert len(moon_seps) == 1 and moon_seps[0].clear is False
    # Structural separation: the Sun is never an avoid_separations entry.
    assert all(s.body != "sun" for s in r.avoid_separations)


def test_self_exclusion(coordinates):
    t = T_NIGHT
    # Observing Jupiter while avoiding Jupiter: must not self-exclude.
    r = check_observability(
        ["jupiter"], t, site=coordinates.site, avoid=[AvoidZone("jupiter", 5.0)]
    )[0]
    assert r.avoid_separations == ()
    assert ReasonCode.AVOID_TOO_CLOSE not in r.reasons
    # The Moon must NOT inherit the offline scorer's point-at-it behaviour either.
    r2 = check_observability(["moon"], t, site=coordinates.site, avoid=[AvoidZone("moon", 5.0)])[0]
    assert r2.avoid_separations == ()
    assert ReasonCode.AVOID_TOO_CLOSE not in r2.reasons


def test_empty_avoid(coordinates):
    t = T_NIGHT
    for avoid in (None, []):
        r = check_observability(["mars"], t, site=coordinates.site, avoid=avoid)[0]
        assert r.avoid_separations == ()
        assert ReasonCode.AVOID_TOO_CLOSE not in r.reasons


def test_name_resolution_and_aliases():
    assert resolve_target("LUNA").name == "moon"
    assert resolve_target("Jupiter").name == "jupiter"
    assert resolve_target("titan").kind == TargetKind.SATELLITE
    with pytest.raises(ValueError):
        resolve_target("pluto")
    r = check_observability(["luna"], T_NIGHT, site=get_fyst_site())[0]
    assert r.name == "moon"


def test_flux_calibrator_catalog_is_read_only():
    """No caller can add or replace a calibrator for every other caller."""
    with pytest.raises(TypeError):
        FLUX_CALIBRATORS["ceres"] = Target("mars", TargetKind.BODY)
    with pytest.raises(TypeError):
        FLUX_CALIBRATORS["mars"] = Target("jupiter", TargetKind.BODY)


def test_fixed_target(coordinates):
    t = T_NIGHT
    lst = coordinates.get_lst(t)
    extra = {
        "src1": Target(
            "src1", TargetKind.FIXED, ra_deg=float(lst), dec_deg=coordinates.site.latitude + 5.0
        )
    }
    r = check_observability(
        ["src1"], t, site=coordinates.site, horizon_hours=24.0, extra_targets=extra
    )[0]
    assert r.name == "src1"
    assert r.target.kind == TargetKind.FIXED
    assert r.windows


def test_fixed_target_rejects_non_finite_ra():
    """A FIXED target with a NaN ra_deg is refused at construction."""
    with pytest.raises(ValueError, match="finite"):
        Target("nan_ra", TargetKind.FIXED, ra_deg=float("nan"), dec_deg=-30.0)


def test_avoid_zone_requires_radius():
    with pytest.raises(TypeError):
        AvoidZone("jupiter")  # missing required radius
    with pytest.raises(ValueError):
        AvoidZone("jupiter", -1.0)
    with pytest.raises(ValueError):
        AvoidZone.from_pair(("jupiter", ""))
    with pytest.raises(ValueError):
        AvoidZone.from_pair(("jupiter",))
    assert AvoidZone.from_pair(("jupiter", "3deg")).zone_deg == 3.0
    assert AvoidZone.from_pair(("moon", 5)).zone_deg == 5.0


def test_titan_saturn_proxy(coordinates):
    t = T_NIGHT
    r = check_observability(["titan"], t, site=coordinates.site)[0]
    assert r.name == "titan"
    assert r.target.kind == TargetKind.SATELLITE
    assert r.position_approximate is True
    sat_az, sat_el = coordinates.get_body_altaz("saturn", t)
    # The Titan proxy returns Saturn's position identically (same ephemeris call).
    assert r.az_deg == pytest.approx(sat_az, abs=0.0)
    assert r.el_deg == pytest.approx(sat_el, abs=0.0)


def test_order_and_count(coordinates):
    t = T_NIGHT
    names = ["mars", "jupiter", "uranus"]
    reports = check_observability(names, t, site=coordinates.site)
    assert [r.name for r in reports] == names
    assert len(reports) == 3


# Regression: SATELLITE self-exclusion keys on the resolved position body
def test_satellite_self_exclusion(coordinates):
    # Titan is proxied by Saturn, so AVOIDing Saturn must self-exclude (Titan IS
    # at Saturn's position), otherwise Titan is silently un-schedulable.
    r = check_observability(
        ["titan"], T_NIGHT, site=coordinates.site, avoid=[AvoidZone("saturn", 5.0)]
    )[0]
    assert r.avoid_separations == ()
    assert ReasonCode.AVOID_TOO_CLOSE not in r.reasons
    # A different AVOID body is still evaluated against Titan's (Saturn-proxy) position.
    r2 = check_observability(
        ["titan"], T_NIGHT, site=coordinates.site, avoid=[AvoidZone("jupiter", 5.0)]
    )[0]
    assert [s.body for s in r2.avoid_separations] == ["jupiter"]


# _all_windows returns EVERY contiguous run in time order (deterministic, no ephemeris)
def test_all_windows_returns_every_run():
    t0 = Time("2026-06-15T00:00:00", scale="utc")
    grid = t0 + TimeDelta(np.arange(7) * 600.0, format="sec")  # 7 samples, 10 min apart
    ok = np.array([True, True, False, False, True, True, True])
    windows = _all_windows(ok, grid)
    assert len(windows) == 2
    first, second = windows
    assert first.truncated_start is True  # first run starts at sample 0
    assert first.truncated_end is False  # first run ends before the grid end
    assert first.duration_hours == pytest.approx(10.0 / 60.0)  # samples 0..1 => 10 min
    assert second.truncated_start is False
    assert second.truncated_end is True  # second run abuts the grid end
    assert second.duration_hours == pytest.approx(20.0 / 60.0)  # samples 4..6 => 20 min
    assert second.start.mjd > first.end.mjd  # time order, disjoint
    assert _all_windows(np.zeros(7, dtype=bool), grid) == ()
    # A single-sample run is a zero-duration window, not a dropped one.
    lone = _all_windows(np.array([False, True, False, False, False, False, False]), grid)
    assert len(lone) == 1
    assert lone[0].duration_hours == 0.0
    # Quantization worst case: an interior window is short by up to TWO steps
    # (one per endpoint). True criterion just misses samples 1 and 5, so the
    # reported run 2..4 (20 min) understates the true ~40 min interval by
    # exactly 2 x 10 min. Locks the total_observable_hours docstring bound.
    interior = _all_windows(np.array([False, False, True, True, True, False, False]), grid)
    assert len(interior) == 1
    assert interior[0].truncated_start is False and interior[0].truncated_end is False
    assert interior[0].duration_hours == pytest.approx(20.0 / 60.0)  # true window ~40 min


# window_step_minutes must be positive when a horizon is requested
def test_window_step_must_be_positive(coordinates):
    tgt = _near_zenith_fixed(coordinates, T_NIGHT)
    with pytest.raises(ValueError):
        check_observability(
            [tgt], T_NIGHT, site=coordinates.site, horizon_hours=24.0, window_step_minutes=0.0
        )
    with pytest.raises(ValueError):
        check_observability(
            [tgt], T_NIGHT, site=coordinates.site, horizon_hours=24.0, window_step_minutes=-5.0
        )
    # Without a horizon the step is unused, so it does not raise.
    r = check_observability([tgt], T_NIGHT, site=coordinates.site, window_step_minutes=0.0)[0]
    assert r.windows is None


# el_min > el_max is a caller error
def test_el_min_gt_el_max_raises(coordinates):
    with pytest.raises(ValueError):
        check_observability(["mars"], T_NIGHT, site=coordinates.site, el_min=80.0, el_max=20.0)


def test_non_finite_el_limits_raise(coordinates):
    """A NaN elevation limit is refused, not read as an empty elevation check."""
    for limits in ({"el_min": float("nan")}, {"el_max": float("nan")}):
        with pytest.raises(ValueError, match="finite"):
            check_observability(["mars"], T_NIGHT, site=coordinates.site, **limits)


# The Sun is never an AvoidZone
def test_avoid_zone_rejects_sun():
    with pytest.raises(ValueError):
        AvoidZone("sun", 30.0)
    with pytest.raises(ValueError):
        AvoidZone("SUN", 30.0)


# Disabled Sun avoidance: sun_clear True, no SUN_TOO_CLOSE, separation still set
def test_sun_avoidance_disabled():
    site = get_fyst_site(sun_avoidance_enabled=False)
    coords = Coordinates(site)
    sun_az, sun_el = coords.get_sun_altaz(T_DAY)
    ra, dec = coords.altaz_to_radec(sun_az, sun_el, T_DAY)
    tgt = Target("at_sun", TargetKind.FIXED, ra_deg=ra, dec_deg=dec)
    r = check_observability([tgt], T_DAY, site=site)[0]
    assert r.sun_clear is True
    assert ReasonCode.SUN_TOO_CLOSE not in r.reasons
    assert r.sun_separation_deg == pytest.approx(0.0, abs=1e-6)  # still populated


# Empty target list
def test_empty_targets(coordinates):
    assert check_observability([], T_NIGHT, site=coordinates.site) == []


# Multiple distinct AVOID bodies each get an entry
def test_multiple_avoid_bodies(coordinates):
    r = check_observability(
        ["mars"],
        T_NIGHT,
        site=coordinates.site,
        avoid=[AvoidZone("jupiter", 3.0), AvoidZone("moon", 5.0)],
    )[0]
    assert sorted(s.body for s in r.avoid_separations) == ["jupiter", "moon"]


# An AVOID body outside SOLAR_SYSTEM_BODIES raises a clear error
def test_invalid_avoid_body_raises(coordinates):
    with pytest.raises(ValueError):
        check_observability(
            ["mars"], T_NIGHT, site=coordinates.site, avoid=[AvoidZone("pluto", 5.0)]
        )


# .summary text for both branches
def test_summary_text(coordinates):
    good = check_observability(
        [_near_zenith_fixed(coordinates, T_NIGHT)], T_NIGHT, site=coordinates.site
    )[0]
    assert "observable" in good.summary
    bad = check_observability(
        [Target("fn", TargetKind.FIXED, ra_deg=0.0, dec_deg=80.0)], T_NIGHT, site=coordinates.site
    )[0]
    assert "NOT observable" in bad.summary


# windows is EMPTY (not None) when a horizon was evaluated and none exists
def test_windows_empty_when_never_observable(coordinates):
    # dec=+80 deg never rises from FYST; with a horizon, _all_windows finds no run.
    tgt = Target("far_north", TargetKind.FIXED, ra_deg=0.0, dec_deg=80.0)
    r = check_observability([tgt], T_NIGHT, site=coordinates.site, horizon_hours=24.0)[0]
    assert r.observable is False
    assert r.windows == ()
    assert r.total_observable_hours == 0.0
    assert ReasonCode.BELOW_EL_MIN in r.reasons


# Titan proxy is exact, and observable when Saturn is up
def test_titan_proxy_when_saturn_up(coordinates):
    # Find an hour within 24h where Saturn clears el_min, deterministically.
    grid = T_NIGHT + TimeDelta(np.arange(0, 24 * 3600, 3600), format="sec")
    _, sat_el = coordinates.get_body_altaz("saturn", grid)
    el_min = coordinates.site.telescope_limits.elevation.min
    up = np.flatnonzero(np.asarray(sat_el) > el_min + 5.0)
    assert up.size, "Saturn never sufficiently up in the test window"
    t = grid[int(up[0])]
    r = check_observability(["titan"], t, site=coordinates.site)[0]
    sat_az, sat_el0 = coordinates.get_body_altaz("saturn", t)
    assert r.az_deg == pytest.approx(sat_az, abs=0.0)
    assert r.el_deg == pytest.approx(sat_el0, abs=0.0)
    assert r.position_approximate is True
    assert r.observable is True


# from_pair degree-symbol and whitespace/case normalization
def test_from_pair_unit_and_whitespace():
    assert AvoidZone.from_pair(("moon", "5\u00b0")).zone_deg == 5.0
    assert AvoidZone.from_pair(("JUPITER", " 3 DEG ")).zone_deg == 3.0
    assert AvoidZone.from_pair(("moon", "3.0")).zone_deg == 3.0


# AVOID body aliases resolve like targets ("luna" -> Moon)
def test_avoid_body_alias_resolves(coordinates):
    # "luna" must resolve to the Moon, identical to AvoidZone("moon", ...).
    r_luna = check_observability(
        ["mars"], T_NIGHT, site=coordinates.site, avoid=[AvoidZone("luna", 181.0)]
    )[0]
    r_moon = check_observability(
        ["mars"], T_NIGHT, site=coordinates.site, avoid=[AvoidZone("moon", 181.0)]
    )[0]
    # Same physical body => same separation; 181 deg zone forces AVOID_TOO_CLOSE.
    assert r_luna.avoid_separations[0].separation_deg == pytest.approx(
        r_moon.avoid_separations[0].separation_deg, abs=1e-9
    )
    assert ReasonCode.AVOID_TOO_CLOSE in r_luna.reasons


# AVOIDing a satellite resolves to its parent; self-excludes the parent target
def test_avoid_satellite_resolves_to_parent(coordinates):
    # AvoidZone("titan") -> Saturn; observing Saturn must self-exclude.
    r = check_observability(
        ["saturn"], T_NIGHT, site=coordinates.site, avoid=[AvoidZone("titan", 5.0)]
    )[0]
    assert r.avoid_separations == ()
    assert ReasonCode.AVOID_TOO_CLOSE not in r.reasons


# from_pair rejects non-numeric / bad-shape inputs with a clear ValueError
def test_from_pair_rejects_malformed():
    with pytest.raises(ValueError):
        AvoidZone.from_pair(("jupiter", "xy"))  # non-numeric
    with pytest.raises(ValueError):
        AvoidZone.from_pair(("jupiter", None))  # None zone
    with pytest.raises(ValueError):
        AvoidZone.from_pair(("a", "b", "c"))  # wrong length
    with pytest.raises(ValueError):
        AvoidZone.from_pair("xy")  # not a tuple/list pair


# Non-finite zone_deg is rejected at construction
def test_avoid_zone_rejects_non_finite():
    with pytest.raises(ValueError):
        AvoidZone("jupiter", float("nan"))
    with pytest.raises(ValueError):
        AvoidZone("jupiter", float("inf"))
    with pytest.raises(ValueError):
        AvoidZone.from_pair(("jupiter", "nan"))


# A non-divisor step keeps the window within [time, time+horizon]
def test_grid_within_horizon_nondivisor_step():
    t0 = Time("2026-06-15T00:00:00", scale="utc")
    grid = _build_time_grid(t0, horizon_hours=1.0, step_minutes=7.0)
    # Last sample clipped to exactly time + horizon; none past it.
    offs = (grid - t0).to_value("s")
    assert offs[-1] == pytest.approx(3600.0)
    assert np.all(offs <= 3600.0 + 1e-6)
    assert len(grid) >= 2


# A sub-step positive horizon still yields a real (n>=2) interval
def test_grid_substep_horizon_not_degenerate():
    t0 = Time("2026-06-15T00:00:00", scale="utc")
    grid = _build_time_grid(t0, horizon_hours=2.0 / 60.0, step_minutes=5.0)  # 2 min horizon
    assert len(grid) >= 2
    offs = (grid - t0).to_value("s")
    assert offs[-1] == pytest.approx(120.0)  # clipped to the 2-min horizon


# ---------------------------------------------------------------------------
# Injectable sun_safe predicate (injectable seam): a directional model drives the
# sun_clear / SUN_TOO_CLOSE verdict end-to-end, default path unchanged.
# ---------------------------------------------------------------------------


# An injected False predicate flips an otherwise-clear target to
# SUN_TOO_CLOSE while leaving the geometric sun_separation_deg untouched.
def test_injected_predicate_flips_sun_clear(coordinates):
    t = T_NIGHT
    _, sun_el = coordinates.get_sun_altaz(t)
    assert sun_el < 0  # precondition: Sun below horizon, scalar check trivially clear
    tgt = _near_zenith_fixed(coordinates, t)

    r_default = check_observability([tgt], t, site=coordinates.site)[0]
    assert r_default.sun_clear is True
    assert r_default.observable is True
    assert ReasonCode.SUN_TOO_CLOSE not in r_default.reasons

    r_blocked = check_observability([tgt], t, site=coordinates.site, sun_safe=block_everything)[0]
    assert r_blocked.sun_clear is False
    assert r_blocked.observable is False
    assert ReasonCode.SUN_TOO_CLOSE in r_blocked.reasons
    # The reported separation is the geometric Sun separation regardless of
    # the predicate; only the verdict changes.
    assert r_blocked.sun_separation_deg == pytest.approx(r_default.sun_separation_deg, abs=1e-6)


# The predicate is consulted with the target's own (az, el, time).
def test_injected_predicate_receives_target_altaz(coordinates):
    t = T_NIGHT
    tgt = _near_zenith_fixed(coordinates, t)
    seen = []

    def spy(az, el, tt):
        seen.append((float(az), float(el)))
        return True

    r = check_observability([tgt], t, site=coordinates.site, sun_safe=spy)[0]
    assert seen, "sun_safe predicate was never consulted"
    # Instant mode (horizon_hours=0) => single grid sample => one call.
    assert len(seen) == 1
    az_seen, el_seen = seen[0]
    assert az_seen == pytest.approx(r.az_deg, abs=1e-6)
    assert el_seen == pytest.approx(r.el_deg, abs=1e-6)


# The predicate drives the horizon-window computation too.
def test_injected_predicate_drives_window(coordinates):
    t = T_NIGHT
    tgt = _near_zenith_fixed(coordinates, t)

    # Default: a window exists over the horizon.
    r_default = check_observability([tgt], t, site=coordinates.site, horizon_hours=6.0)[0]
    assert r_default.windows

    # A predicate that blocks every sample leaves no observable window.
    r_blocked = check_observability(
        [tgt], t, site=coordinates.site, horizon_hours=6.0, sun_safe=block_everything
    )[0]
    assert r_blocked.windows == ()
    assert ReasonCode.SUN_TOO_CLOSE in r_blocked.reasons


# A permissive predicate clears a daytime target the scalar rejects.
def test_injected_allow_predicate_overrides_daytime(coordinates):
    t = T_DAY
    _, sun_el = coordinates.get_sun_altaz(t)
    assert sun_el > 0  # precondition: Sun up
    # A FIXED source AT the Sun's position: the scalar check rejects it.
    sun_az, sun_alt = coordinates.get_sun_altaz(t)
    sun_ra, sun_dec = coordinates.altaz_to_radec(sun_az, sun_alt, t)
    at_sun = Target("at_sun", TargetKind.FIXED, ra_deg=float(sun_ra), dec_deg=float(sun_dec))

    r_default = check_observability([at_sun], t, site=coordinates.site)[0]
    assert r_default.sun_clear is False
    assert ReasonCode.SUN_TOO_CLOSE in r_default.reasons

    r_allowed = check_observability([at_sun], t, site=coordinates.site, sun_safe=allow_everything)[
        0
    ]
    assert r_allowed.sun_clear is True
    assert ReasonCode.SUN_TOO_CLOSE not in r_allowed.reasons


# sun_safe=None reproduces the built-in scalar verdict exactly.
def test_injected_predicate_default_none_unchanged(coordinates):
    t = T_NIGHT
    tgt = _near_zenith_fixed(coordinates, t)
    r_implicit = check_observability([tgt], t, site=coordinates.site)[0]
    r_explicit_none = check_observability([tgt], t, site=coordinates.site, sun_safe=None)[0]
    assert r_explicit_none.sun_clear == r_implicit.sun_clear
    assert r_explicit_none.observable == r_implicit.observable
    assert r_explicit_none.reasons == r_implicit.reasons
    assert r_explicit_none.sun_separation_deg == pytest.approx(r_implicit.sun_separation_deg)


# A 24 h horizon catches BOTH daily passes of a transiting source (a
# single-window report would hide the second one).
def test_two_daily_passes_both_reported(coordinates):
    t = T_NIGHT
    tgt = _near_zenith_fixed(coordinates, t)  # transits at t; ~10 h above el_min=20
    r = check_observability([tgt], t, site=coordinates.site, horizon_hours=24.0)[0]
    assert r.windows is not None
    assert len(r.windows) == 2
    first, second = r.windows
    # Mid-pass at t: the first window is the tail of today's pass.
    assert first.truncated_start is True
    assert first.truncated_end is False
    # The second window is tomorrow's pass, cut off by the horizon end.
    assert second.truncated_start is False
    assert second.truncated_end is True
    assert second.start.mjd > first.end.mjd
    assert r.total_observable_hours == pytest.approx(first.duration_hours + second.duration_hours)


# A predicate exposing the optional `batch` extension is evaluated in
# ONE vectorized call; its verdicts flow through to reasons and windows.
def test_batch_predicate_used_vectorized(coordinates):
    t = T_NIGHT
    tgt = _near_zenith_fixed(coordinates, t)
    calls = {"batch": 0, "scalar": 0}

    class _AllowBatch:
        def __call__(self, az, el, time):
            calls["scalar"] += 1
            return True

        def batch(self, az, el, times):
            calls["batch"] += 1
            return np.ones(np.shape(np.atleast_1d(az)), dtype=bool)

    r = check_observability(
        [tgt], t, site=coordinates.site, horizon_hours=6.0, sun_safe=_AllowBatch()
    )[0]
    assert calls == {"batch": 1, "scalar": 0}
    assert r.sun_clear is True
    assert r.windows

    class _BlockBatch:
        def __call__(self, az, el, time):
            return False

        def batch(self, az, el, times):
            return np.zeros(np.shape(np.atleast_1d(az)), dtype=bool)

    r_blocked = check_observability(
        [tgt], t, site=coordinates.site, horizon_hours=6.0, sun_safe=_BlockBatch()
    )[0]
    assert ReasonCode.SUN_TOO_CLOSE in r_blocked.reasons
    assert r_blocked.windows == ()

    class _WrongShapeBatch:
        def __call__(self, az, el, time):
            return True

        def batch(self, az, el, times):
            return np.ones(3, dtype=bool)  # wrong length: must not broadcast

    with pytest.raises(ValueError, match="sun_safe.batch"):
        check_observability(
            [tgt], t, site=coordinates.site, horizon_hours=6.0, sun_safe=_WrongShapeBatch()
        )


class TestHorizonArgumentFiniteness:
    """``check_observability`` refuses a non-finite horizon instead of ignoring it.

    ``sun_events`` documents finiteness as necessary and checks it; without the
    same check here a NaN ``horizon_hours`` would fail both ``bool()`` guards
    and fall through to instant-only mode, returning a report with no windows
    and no explanation.
    """

    TIME = Time("2026-06-15T00:00:00", scale="utc")

    @pytest.mark.parametrize("horizon", [float("nan"), float("inf")])
    def test_non_finite_horizon_is_refused(self, horizon):
        """NaN and infinity raise, naming the argument."""
        with pytest.raises(ValueError, match="horizon_hours must be a finite value"):
            check_observability(["jupiter"], self.TIME, horizon_hours=horizon)

    def test_non_finite_step_is_refused(self):
        """A NaN step with a real horizon raises rather than building a bad grid."""
        with pytest.raises(ValueError, match="window_step_minutes must be a finite value"):
            check_observability(
                ["jupiter"], self.TIME, horizon_hours=6.0, window_step_minutes=float("nan")
            )

    def test_instant_only_mode_still_accepts_zero_and_none(self):
        """The documented instant-only spellings (0.0 and None) stay legal."""
        for horizon in (0.0, None):
            report = check_observability(["jupiter"], self.TIME, horizon_hours=horizon)[0]
            assert report.windows is None
