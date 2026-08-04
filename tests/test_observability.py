"""Tests for the observability (OBSERVE / EXCLUDE) primitives.

Cases are made deterministic without hand-computed ephemeris by constructing
geometry from the same primitives under test: a near-zenith FIXED source is
placed on the meridian at ``ra = LST(t)``; Sun/avoid conditions are forced by
placing a FIXED target at a body's RA/Dec, or by an oversized AVOID zone.

``T_NIGHT`` is local midnight at FYST (Sun well below the horizon) and
``T_DAY`` is ~local noon (Sun up); both are stable year-to-year for the
chosen calendar date and within the IERS prediction window.
"""

import subprocess
import sys

import numpy as np
import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories import Coordinates, get_fyst_site
from fyst_trajectories.observability import (
    AvoidZone,
    ReasonCode,
    SunEventKind,
    Target,
    TargetKind,
    _all_windows,
    _build_time_grid,
    _threshold_crossings,
    check_observability,
    resolve_target,
    sun_events,
)

T_NIGHT = Time("2026-06-15T05:00:00", scale="utc")
T_DAY = Time("2026-06-15T16:30:00", scale="utc")


def _near_zenith_fixed(coords, t, name="zen"):
    """Return a FIXED source transiting near the zenith at time ``t`` (el ~ 85 deg)."""
    lst = coords.get_lst(t)
    return Target(name, TargetKind.FIXED, ra_deg=float(lst), dec_deg=coords.site.latitude + 5.0)


# 1
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


# 2
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


# 3
def test_below_el_min(coordinates):
    t = T_NIGHT
    # dec = +80 deg is never visible from FYST (lat ~ -23 deg): always below the horizon.
    tgt = Target("far_north", TargetKind.FIXED, ra_deg=0.0, dec_deg=80.0)
    r = check_observability([tgt], t, site=coordinates.site)[0]
    assert r.observable is False
    assert ReasonCode.BELOW_EL_MIN in r.reasons
    assert r.el_deg < 20.0


# 4
def test_above_el_max(coordinates):
    t = T_NIGHT
    tgt = _near_zenith_fixed(coordinates, t)  # el ~ 85
    r = check_observability([tgt], t, site=coordinates.site, el_max=80.0)[0]
    assert r.observable is False
    assert ReasonCode.ABOVE_EL_MAX in r.reasons


# 5
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
    assert r.sun_separation_deg < 45.0


# 6
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


# 7
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


# 8
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


# 9
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


# 10
def test_empty_avoid(coordinates):
    t = T_NIGHT
    for avoid in (None, []):
        r = check_observability(["mars"], t, site=coordinates.site, avoid=avoid)[0]
        assert r.avoid_separations == ()
        assert ReasonCode.AVOID_TOO_CLOSE not in r.reasons


# 11
def test_name_resolution_and_aliases():
    assert resolve_target("LUNA").name == "moon"
    assert resolve_target("Jupiter").name == "jupiter"
    assert resolve_target("titan").kind == TargetKind.SATELLITE
    with pytest.raises(ValueError):
        resolve_target("pluto")
    r = check_observability(["luna"], T_NIGHT, site=get_fyst_site())[0]
    assert r.name == "moon"


# 12
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


# 13
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


# 14
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


# 15
def test_order_and_count(coordinates):
    t = T_NIGHT
    names = ["mars", "jupiter", "uranus"]
    reports = check_observability(names, t, site=coordinates.site)
    assert [r.name for r in reports] == names
    assert len(reports) == 3


# 16
def test_no_overhead_import_at_load():
    code = (
        "import sys; import fyst_trajectories.observability as o; "
        "assert 'fyst_trajectories.overhead' not in sys.modules, "
        "sorted(m for m in sys.modules if 'overhead' in m)"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


# 17 - regression: SATELLITE self-exclusion keys on the resolved position body
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


# 18 - _all_windows returns EVERY contiguous run in time order (deterministic, no ephemeris)
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


# 19 - window_step_minutes must be positive when a horizon is requested
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


# 20 - el_min > el_max is a caller error
def test_el_min_gt_el_max_raises(coordinates):
    with pytest.raises(ValueError):
        check_observability(["mars"], T_NIGHT, site=coordinates.site, el_min=80.0, el_max=20.0)


# 21 - the Sun is never an AvoidZone
def test_avoid_zone_rejects_sun():
    with pytest.raises(ValueError):
        AvoidZone("sun", 30.0)
    with pytest.raises(ValueError):
        AvoidZone("SUN", 30.0)


# 22 - disabled Sun avoidance: sun_clear True, no SUN_TOO_CLOSE, separation still set (T5)
def test_sun_avoidance_disabled():
    site = get_fyst_site(sun_avoidance_enabled=False)
    coords = Coordinates(site)
    sun_az, sun_el = coords.get_sun_altaz(T_DAY)
    ra, dec = coords.altaz_to_radec(sun_az, sun_el, T_DAY)
    tgt = Target("at_sun", TargetKind.FIXED, ra_deg=ra, dec_deg=dec)
    r = check_observability([tgt], T_DAY, site=site)[0]
    assert r.sun_clear is True
    assert ReasonCode.SUN_TOO_CLOSE not in r.reasons
    assert r.sun_separation_deg < 1.0  # still populated


# 23 - empty target list
def test_empty_targets(coordinates):
    assert check_observability([], T_NIGHT, site=coordinates.site) == []


# 24 - multiple distinct AVOID bodies each get an entry
def test_multiple_avoid_bodies(coordinates):
    r = check_observability(
        ["mars"],
        T_NIGHT,
        site=coordinates.site,
        avoid=[AvoidZone("jupiter", 3.0), AvoidZone("moon", 5.0)],
    )[0]
    assert sorted(s.body for s in r.avoid_separations) == ["jupiter", "moon"]


# 25 - an AVOID body outside SOLAR_SYSTEM_BODIES raises a clear error
def test_invalid_avoid_body_raises(coordinates):
    with pytest.raises(ValueError):
        check_observability(
            ["mars"], T_NIGHT, site=coordinates.site, avoid=[AvoidZone("pluto", 5.0)]
        )


# 26 - .summary text for both branches
def test_summary_text(coordinates):
    good = check_observability(
        [_near_zenith_fixed(coordinates, T_NIGHT)], T_NIGHT, site=coordinates.site
    )[0]
    assert "observable" in good.summary
    bad = check_observability(
        [Target("fn", TargetKind.FIXED, ra_deg=0.0, dec_deg=80.0)], T_NIGHT, site=coordinates.site
    )[0]
    assert "NOT observable" in bad.summary


# 27 - T6: windows is EMPTY (not None) when a horizon was evaluated and none exists
def test_windows_empty_when_never_observable(coordinates):
    # dec=+80 deg never rises from FYST; with a horizon, _all_windows finds no run.
    tgt = Target("far_north", TargetKind.FIXED, ra_deg=0.0, dec_deg=80.0)
    r = check_observability([tgt], T_NIGHT, site=coordinates.site, horizon_hours=24.0)[0]
    assert r.observable is False
    assert r.windows == ()
    assert r.total_observable_hours == 0.0
    assert ReasonCode.BELOW_EL_MIN in r.reasons


# 28 - T9: Titan proxy is exact, and observable when Saturn is up
def test_titan_proxy_when_saturn_up(coordinates):
    # Find an hour within 24h where Saturn clears el_min, deterministically.
    grid = T_NIGHT + TimeDelta(np.arange(0, 24 * 3600, 3600), format="sec")
    _, sat_el = coordinates.get_body_altaz("saturn", grid)
    el_min = coordinates.site.telescope_limits.elevation.min
    up = np.flatnonzero(np.asarray(sat_el) > el_min + 5.0)
    if up.size == 0:
        pytest.skip("Saturn never sufficiently up in the test window")
    t = grid[int(up[0])]
    r = check_observability(["titan"], t, site=coordinates.site)[0]
    sat_az, sat_el0 = coordinates.get_body_altaz("saturn", t)
    assert r.az_deg == pytest.approx(sat_az, abs=0.0)
    assert r.el_deg == pytest.approx(sat_el0, abs=0.0)
    assert r.position_approximate is True


# 29 - T10: from_pair degree-symbol and whitespace/case normalization
def test_from_pair_unit_and_whitespace():
    assert AvoidZone.from_pair(("moon", "5°")).zone_deg == 5.0
    assert AvoidZone.from_pair(("JUPITER", " 3 DEG ")).zone_deg == 3.0
    assert AvoidZone.from_pair(("moon", "3.0")).zone_deg == 3.0


# 30 - AVOID body aliases resolve like targets ("luna" -> Moon)
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


# 31 - AVOIDing a satellite resolves to its parent; self-excludes the parent target
def test_avoid_satellite_resolves_to_parent(coordinates):
    # AvoidZone("titan") -> Saturn; observing Saturn must self-exclude.
    r = check_observability(
        ["saturn"], T_NIGHT, site=coordinates.site, avoid=[AvoidZone("titan", 5.0)]
    )[0]
    assert r.avoid_separations == ()
    assert ReasonCode.AVOID_TOO_CLOSE not in r.reasons


# 32 - an unresolvable AVOID body raises a clear error up front
def test_avoid_unresolvable_body_raises(coordinates):
    with pytest.raises(ValueError):
        check_observability(
            ["mars"], T_NIGHT, site=coordinates.site, avoid=[AvoidZone("pluto", 5.0)]
        )


# 33 - F10: from_pair rejects non-numeric / bad-shape inputs with a clear ValueError
def test_from_pair_rejects_malformed():
    with pytest.raises(ValueError):
        AvoidZone.from_pair(("jupiter", "xy"))  # non-numeric
    with pytest.raises(ValueError):
        AvoidZone.from_pair(("jupiter", None))  # None zone
    with pytest.raises(ValueError):
        AvoidZone.from_pair(("a", "b", "c"))  # wrong length
    with pytest.raises(ValueError):
        AvoidZone.from_pair("xy")  # not a tuple/list pair


# 34 - F10: non-finite zone_deg is rejected at construction
def test_avoid_zone_rejects_non_finite():
    with pytest.raises(ValueError):
        AvoidZone("jupiter", float("nan"))
    with pytest.raises(ValueError):
        AvoidZone("jupiter", float("inf"))
    with pytest.raises(ValueError):
        AvoidZone.from_pair(("jupiter", "nan"))


# 35 - a non-divisor step keeps the window within [time, time+horizon]
def test_grid_within_horizon_nondivisor_step():
    t0 = Time("2026-06-15T00:00:00", scale="utc")
    grid = _build_time_grid(t0, horizon_hours=1.0, step_minutes=7.0)
    # Last sample clipped to exactly time + horizon; none past it.
    offs = (grid - t0).to_value("s")
    assert offs[-1] == pytest.approx(3600.0)
    assert np.all(offs <= 3600.0 + 1e-6)
    assert len(grid) >= 2


# 36 - a sub-step positive horizon still yields a real (n>=2) interval
def test_grid_substep_horizon_not_degenerate():
    t0 = Time("2026-06-15T00:00:00", scale="utc")
    grid = _build_time_grid(t0, horizon_hours=2.0 / 60.0, step_minutes=5.0)  # 2 min horizon
    assert len(grid) >= 2
    offs = (grid - t0).to_value("s")
    assert offs[-1] == pytest.approx(120.0)  # clipped to the 2-min horizon


# ---------------------------------------------------------------------------
# Injectable sun_safe predicate (A3 seam): a directional model drives the
# sun_clear / SUN_TOO_CLOSE verdict end-to-end, default path unchanged.
# ---------------------------------------------------------------------------


def _block_everything(az, el, t):
    """SunSafePredicate that reports every grid sample unsafe."""
    return False


def _allow_everything(az, el, t):
    """SunSafePredicate that reports every grid sample clear of the Sun."""
    return True


# 37 - A3: an injected False predicate flips an otherwise-clear target to
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

    r_blocked = check_observability([tgt], t, site=coordinates.site, sun_safe=_block_everything)[0]
    assert r_blocked.sun_clear is False
    assert r_blocked.observable is False
    assert ReasonCode.SUN_TOO_CLOSE in r_blocked.reasons
    # The reported separation is the geometric Sun separation regardless of
    # the predicate; only the verdict changes.
    assert r_blocked.sun_separation_deg == pytest.approx(r_default.sun_separation_deg, abs=1e-6)


# 38 - A3: the predicate is consulted with the target's own (az, el, time).
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


# 39 - A3: the predicate drives the horizon-window computation too.
def test_injected_predicate_drives_window(coordinates):
    t = T_NIGHT
    tgt = _near_zenith_fixed(coordinates, t)

    # Default: a window exists over the horizon.
    r_default = check_observability([tgt], t, site=coordinates.site, horizon_hours=6.0)[0]
    assert r_default.windows

    # A predicate that blocks every sample leaves no observable window.
    r_blocked = check_observability(
        [tgt], t, site=coordinates.site, horizon_hours=6.0, sun_safe=_block_everything
    )[0]
    assert r_blocked.windows == ()
    assert ReasonCode.SUN_TOO_CLOSE in r_blocked.reasons


# 40 - A3: a permissive predicate clears a daytime target the scalar rejects.
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

    r_allowed = check_observability([at_sun], t, site=coordinates.site, sun_safe=_allow_everything)[
        0
    ]
    assert r_allowed.sun_clear is True
    assert ReasonCode.SUN_TOO_CLOSE not in r_allowed.reasons


# 41 - A3: sun_safe=None reproduces the built-in scalar verdict exactly.
def test_injected_predicate_default_none_unchanged(coordinates):
    t = T_NIGHT
    tgt = _near_zenith_fixed(coordinates, t)
    r_implicit = check_observability([tgt], t, site=coordinates.site)[0]
    r_explicit_none = check_observability([tgt], t, site=coordinates.site, sun_safe=None)[0]
    assert r_explicit_none.sun_clear == r_implicit.sun_clear
    assert r_explicit_none.observable == r_implicit.observable
    assert r_explicit_none.reasons == r_implicit.reasons
    assert r_explicit_none.sun_separation_deg == pytest.approx(r_implicit.sun_separation_deg)


# 42 - a 24 h horizon catches BOTH daily passes of a transiting source (the
# pre-fix single-window report hid the second one).
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


# 43 - sun_events: one FYST day from local noon yields the full 8-event
# sequence, dusk side first, in strict time order with sane times.
def test_sun_events_full_day_sequence():
    t = Time("2026-11-15T16:00:00", scale="utc")  # ~13:00 Chile local
    events = sun_events(t)
    kinds = [e.kind for e in events]
    assert kinds == [
        SunEventKind.SUNSET,
        SunEventKind.CIVIL_DUSK,
        SunEventKind.NAUTICAL_DUSK,
        SunEventKind.ASTRONOMICAL_DUSK,
        SunEventKind.ASTRONOMICAL_DAWN,
        SunEventKind.NAUTICAL_DAWN,
        SunEventKind.CIVIL_DAWN,
        SunEventKind.SUNRISE,
    ]
    assert [e.rising for e in events] == [False] * 4 + [True] * 4
    mjds = [e.time.mjd for e in events]
    assert mjds == sorted(mjds)
    sunset = events[0]
    sunrise = events[-1]
    # Geometric (0 deg) crossings for this date are ~22:47 set / ~09:43 rise UTC
    # (cross-checked against get_rise_set_times); the -0.8333 deg almanac
    # threshold shifts set ~5 min later and rise ~5 min earlier.
    assert Time("2026-11-15T22:45:00", scale="utc") <= sunset.time
    assert sunset.time <= Time("2026-11-15T23:00:00", scale="utc")
    assert Time("2026-11-16T09:25:00", scale="utc") <= sunrise.time
    assert sunrise.time <= Time("2026-11-16T09:45:00", scale="utc")


# 44 - sun_events: the Sun's geometric altitude at each event time equals the
# event's threshold (locks the interpolation and the vacuum convention).
def test_sun_events_altitude_invariant(site):
    events = sun_events(Time("2026-11-15T16:00:00", scale="utc"), site=site)
    coords = Coordinates(site)  # vacuum, matching the implementation
    assert events
    for event in events:
        _, el = coords.get_sun_altaz(event.time)
        # Locks the "seconds level" interpolation claim: 1e-3 deg is ~0.3 s
        # of solar altitude motion at FYST twilight rates.
        assert el == pytest.approx(event.altitude_deg, abs=1e-3)


# 45 - sun_events: parameter validation (incl. the NaN/inf hole: NaN passes
# a bare `<= 0` and would silently return an empty tuple).
def test_sun_events_validation():
    t = Time("2026-11-15T16:00:00", scale="utc")
    for bad_horizon in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            sun_events(t, horizon_hours=bad_horizon)
    for bad_step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            sun_events(t, step_minutes=bad_step)


# 45b - a predicate exposing the optional `batch` extension is evaluated in
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


# 46 - _threshold_crossings: synthetic arrays lock the crossing partition,
# the interpolation, and the clipped-final-cell handling (no ephemeris).
def test_threshold_crossings_synthetic():
    t0 = Time("2026-06-15T00:00:00", scale="utc")
    grid4 = t0 + TimeDelta(np.arange(4) * 600.0, format="sec")

    # Plain interior crossing: linear interpolation between samples.
    up = _threshold_crossings(np.array([-2.0, -1.0, 0.5, 2.0]), grid4, 0.0, rising=True)
    assert len(up) == 1
    assert (up[0] - t0).to_value("s") == pytest.approx(600.0 + 600.0 * (1.0 / 1.5), abs=1e-6)

    # A value exactly AT the threshold on a grid sample: exactly one event,
    # landing exactly on that sample (frac = 1 in the preceding cell).
    grid3 = t0 + TimeDelta(np.arange(3) * 600.0, format="sec")
    exact = _threshold_crossings(np.array([-1.0, 0.0, 1.0]), grid3, 0.0, rising=True)
    assert len(exact) == 1
    assert abs((exact[0] - grid3[1]).to_value("s")) < 1e-9

    # Plateau at the threshold: still a single event, not one per sample.
    plateau = _threshold_crossings(np.array([-1.0, 0.0, 0.0, 1.0]), grid4, 0.0, rising=True)
    assert len(plateau) == 1

    # Tangential touch: landing exactly ON the threshold yields no events
    # (the >=/< partition), while dipping infinitesimally below yields a
    # set+rise pair. Pins the boundary semantics.
    assert _threshold_crossings(np.array([1.0, 0.0, 1.0]), grid3, 0.0, rising=True) == []
    assert _threshold_crossings(np.array([1.0, 0.0, 1.0]), grid3, 0.0, rising=False) == []
    dip = np.array([1.0, -1e-9, 1.0])
    assert len(_threshold_crossings(dip, grid3, 0.0, rising=False)) == 1
    assert len(_threshold_crossings(dip, grid3, 0.0, rising=True)) == 1

    # No crossings => empty.
    assert _threshold_crossings(np.array([1.0, 2.0, 3.0]), grid3, 0.0, rising=True) == []

    # Clipped final cell from _build_time_grid (horizon 9 min @ 4 min step =>
    # cells of 240/240/60 s): interpolation must use the actual 60 s spacing.
    grid_clip = _build_time_grid(t0, horizon_hours=0.15, step_minutes=4.0)
    assert (grid_clip[-1] - grid_clip[-2]).to_value("s") == pytest.approx(60.0)
    clipped = _threshold_crossings(np.array([-3.0, -2.0, -1.0, 1.0]), grid_clip, 0.0, rising=True)
    assert len(clipped) == 1
    assert (clipped[0] - t0).to_value("s") == pytest.approx(480.0 + 0.5 * 60.0, abs=1e-6)


# 47 - sun_events: a sub-day evening span returns only the dusk side.
def test_sun_events_subday_dusk_only():
    events = sun_events(Time("2026-11-15T21:00:00", scale="utc"), horizon_hours=4.0)
    assert [e.kind for e in events] == [
        SunEventKind.SUNSET,
        SunEventKind.CIVIL_DUSK,
        SunEventKind.NAUTICAL_DUSK,
        SunEventKind.ASTRONOMICAL_DUSK,
    ]
    assert all(e.rising is False for e in events)


# 48 - sun_events: a 48 h horizon returns two full days of events, sorted,
# in alternating dusk-block / dawn-block order.
def test_sun_events_two_days():
    events = sun_events(Time("2026-11-15T16:00:00", scale="utc"), horizon_hours=48.0)
    assert len(events) == 16
    mjds = [e.time.mjd for e in events]
    assert mjds == sorted(mjds)
    assert [e.rising for e in events] == [False] * 4 + [True] * 4 + [False] * 4 + [True] * 4
