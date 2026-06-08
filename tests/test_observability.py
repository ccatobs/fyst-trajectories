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
    Target,
    TargetKind,
    _build_time_grid,
    _first_window,
    check_observability,
    resolve_target,
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
    assert r.window is None
    assert r.sun_clear is True
    assert 80.0 < r.el_deg < 90.0
    assert r.position_approximate is False


# 2
def test_horizon_window(coordinates):
    t = T_NIGHT
    tgt = _near_zenith_fixed(coordinates, t)
    r = check_observability([tgt], t, site=coordinates.site, horizon_hours=24.0)[0]
    assert r.observable is True
    assert r.window is not None
    assert r.window.duration_hours > 0.0
    # Observable now => the window opens at t and is truncated at the horizon start.
    assert r.window.truncated_start is True
    assert abs((r.window.start - t).to_value("s")) < 1.0


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
    assert r.window is not None


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


# 17 -- F1 regression: SATELLITE self-exclusion keys on the resolved position body
def test_satellite_self_exclusion(coordinates):
    # Titan is proxied by Saturn, so AVOIDing Saturn must self-exclude (Titan IS
    # at Saturn's position) -- otherwise Titan is silently un-schedulable.
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


# 18 -- _first_window picks the FIRST contiguous run (deterministic, no ephemeris)
def test_first_window_picks_first_run():
    t0 = Time("2026-06-15T00:00:00", scale="utc")
    grid = t0 + TimeDelta(np.arange(7) * 600.0, format="sec")  # 7 samples, 10 min apart
    ok = np.array([True, True, False, False, True, True, True])
    w = _first_window(ok, grid)
    assert w is not None
    assert w.truncated_start is True  # run starts at sample 0
    assert w.truncated_end is False  # first run ends before the grid end
    assert w.duration_hours == pytest.approx(10.0 / 60.0)  # samples 0..1 => 10 min
    assert _first_window(np.zeros(7, dtype=bool), grid) is None


# 19 -- window_step_minutes must be positive when a horizon is requested (F2)
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
    assert r.window is None


# 20 -- el_min > el_max is a caller error (F8)
def test_el_min_gt_el_max_raises(coordinates):
    with pytest.raises(ValueError):
        check_observability(["mars"], T_NIGHT, site=coordinates.site, el_min=80.0, el_max=20.0)


# 21 -- the Sun is never an AvoidZone (F7)
def test_avoid_zone_rejects_sun():
    with pytest.raises(ValueError):
        AvoidZone("sun", 30.0)
    with pytest.raises(ValueError):
        AvoidZone("SUN", 30.0)


# 22 -- disabled Sun avoidance: sun_clear True, no SUN_TOO_CLOSE, separation still set (T5)
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


# 23 -- empty target list
def test_empty_targets(coordinates):
    assert check_observability([], T_NIGHT, site=coordinates.site) == []


# 24 -- multiple distinct AVOID bodies each get an entry
def test_multiple_avoid_bodies(coordinates):
    r = check_observability(
        ["mars"],
        T_NIGHT,
        site=coordinates.site,
        avoid=[AvoidZone("jupiter", 3.0), AvoidZone("moon", 5.0)],
    )[0]
    assert sorted(s.body for s in r.avoid_separations) == ["jupiter", "moon"]


# 25 -- an AVOID body outside SOLAR_SYSTEM_BODIES raises a clear error
def test_invalid_avoid_body_raises(coordinates):
    with pytest.raises(ValueError):
        check_observability(
            ["mars"], T_NIGHT, site=coordinates.site, avoid=[AvoidZone("pluto", 5.0)]
        )


# 26 -- .summary text for both branches
def test_summary_text(coordinates):
    good = check_observability(
        [_near_zenith_fixed(coordinates, T_NIGHT)], T_NIGHT, site=coordinates.site
    )[0]
    assert "observable" in good.summary
    bad = check_observability(
        [Target("fn", TargetKind.FIXED, ra_deg=0.0, dec_deg=80.0)], T_NIGHT, site=coordinates.site
    )[0]
    assert "NOT observable" in bad.summary


# 27 -- T6: window is None when a target is never observable over the horizon
def test_window_none_when_never_observable(coordinates):
    # dec=+80 deg never rises from FYST; with a horizon, _first_window finds no run.
    tgt = Target("far_north", TargetKind.FIXED, ra_deg=0.0, dec_deg=80.0)
    r = check_observability([tgt], T_NIGHT, site=coordinates.site, horizon_hours=24.0)[0]
    assert r.observable is False
    assert r.window is None
    assert ReasonCode.BELOW_EL_MIN in r.reasons


# 28 -- T9: Titan proxy is exact, and observable when Saturn is up
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


# 29 -- T10: from_pair degree-symbol and whitespace/case normalization
def test_from_pair_unit_and_whitespace():
    assert AvoidZone.from_pair(("moon", "5°")).zone_deg == 5.0
    assert AvoidZone.from_pair(("JUPITER", " 3 DEG ")).zone_deg == 3.0
    assert AvoidZone.from_pair(("moon", "3.0")).zone_deg == 3.0


# 30 -- F9: AVOID body aliases resolve like targets ("luna" -> Moon)
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


# 31 -- F9: AVOIDing a satellite resolves to its parent; self-excludes the parent target
def test_avoid_satellite_resolves_to_parent(coordinates):
    # AvoidZone("titan") -> Saturn; observing Saturn must self-exclude.
    r = check_observability(
        ["saturn"], T_NIGHT, site=coordinates.site, avoid=[AvoidZone("titan", 5.0)]
    )[0]
    assert r.avoid_separations == ()
    assert ReasonCode.AVOID_TOO_CLOSE not in r.reasons


# 32 -- F9: an unresolvable AVOID body raises a clear error up front
def test_avoid_unresolvable_body_raises(coordinates):
    with pytest.raises(ValueError):
        check_observability(
            ["mars"], T_NIGHT, site=coordinates.site, avoid=[AvoidZone("pluto", 5.0)]
        )


# 33 -- F10: from_pair rejects non-numeric / bad-shape inputs with a clear ValueError
def test_from_pair_rejects_malformed():
    with pytest.raises(ValueError):
        AvoidZone.from_pair(("jupiter", "xy"))  # non-numeric
    with pytest.raises(ValueError):
        AvoidZone.from_pair(("jupiter", None))  # None zone
    with pytest.raises(ValueError):
        AvoidZone.from_pair(("a", "b", "c"))  # wrong length
    with pytest.raises(ValueError):
        AvoidZone.from_pair("xy")  # not a tuple/list pair


# 34 -- F10: non-finite zone_deg is rejected at construction
def test_avoid_zone_rejects_non_finite():
    with pytest.raises(ValueError):
        AvoidZone("jupiter", float("nan"))
    with pytest.raises(ValueError):
        AvoidZone("jupiter", float("inf"))
    with pytest.raises(ValueError):
        AvoidZone.from_pair(("jupiter", "nan"))


# 35 -- F3: a non-divisor step keeps the window within [time, time+horizon]
def test_grid_within_horizon_nondivisor_step():
    t0 = Time("2026-06-15T00:00:00", scale="utc")
    grid = _build_time_grid(t0, horizon_hours=1.0, step_minutes=7.0)
    # Last sample clipped to exactly time + horizon; none past it.
    offs = (grid - t0).to_value("s")
    assert offs[-1] == pytest.approx(3600.0)
    assert np.all(offs <= 3600.0 + 1e-6)
    assert len(grid) >= 2


# 36 -- F3: a sub-step positive horizon still yields a real (n>=2) interval
def test_grid_substep_horizon_not_degenerate():
    t0 = Time("2026-06-15T00:00:00", scale="utc")
    grid = _build_time_grid(t0, horizon_hours=2.0 / 60.0, step_minutes=5.0)  # 2 min horizon
    assert len(grid) >= 2
    offs = (grid - t0).to_value("s")
    assert offs[-1] == pytest.approx(120.0)  # clipped to the 2-min horizon
