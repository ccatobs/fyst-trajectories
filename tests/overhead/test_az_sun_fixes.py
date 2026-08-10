"""Regression tests for overhead-scheduler azimuth + sun bugs.

The overhead subpackage is an OFFLINE observing-night simulator. These
tests pin three fixes:

1. Cable-wrap azimuth normalization. A north-straddling azimuth pair
   (e.g. 350 deg and 10 deg) must give the short-path slew (~20 deg, not
   ~340 deg) and a non-flipped boresight (circular mean, not the ~180
   deg-off arithmetic mean).
2. Sun-aware pong/daisy duration clip. A pong/daisy scan that is
   sun-unsafe (or drifts into the exclusion radius) must have its
   duration trimmed, mirroring the constant_el branch.
3. MinDurationConstraint sun forward-check. A target that stays above
   the elevation floor but is inside the Sun exclusion zone within
   ``min_duration`` must be rejected, matching the class docstring.
"""

import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories import Coordinates
from fyst_trajectories.overhead.constraints import MinDurationConstraint
from fyst_trajectories.overhead.models import ObservingPatch, OverheadModel, TimelineBlock
from fyst_trajectories.overhead.scheduler.helpers import (
    _compute_scan_duration,
    _normalize_az,
    _time_until_set,
    _time_until_sun_unsafe,
)
from fyst_trajectories.overhead.utils import (
    circular_mean_deg,
    compute_nasmyth_rotation,
    estimate_slew_time,
)

# A time/place where a target offset ~20 deg in RA from the Sun is well
# above the FYST horizon but inside the 50 deg Sun exclusion radius.
_SUN_TIME = Time("2026-06-15T17:30:00", scale="utc")


@pytest.fixture
def coords(site):
    return Coordinates(site)


@pytest.fixture
def sun_radec(coords):
    """Sun RA/Dec at the test time (for placing sunward targets)."""
    ra, dec = coords.get_body_radec("sun", _SUN_TIME)
    return float(ra), float(dec)


class TestNorthStraddleSlew:
    """Fix 1: a north-straddling azimuth pair takes the short slew path."""

    def test_short_path_slew_time(self, site):
        """Raw [0,360) azimuths 350 and 10 must slew ~20 deg, not ~340 deg."""
        na = _normalize_az(350.0, site)
        nb = _normalize_az(10.0, site)
        # Normalized into the [-180, 360] cable-wrap window, the pair is
        # contiguous (-10, 10), a 20 deg move.
        assert abs(nb - na) == pytest.approx(20.0, abs=1e-6)

        t_norm = estimate_slew_time(na, 50.0, nb, 50.0, site)
        # 20 deg az slew (FYST az vel=3, accel=1.5): d_accel=6 < 20, so
        # t = 2*t_accel + (20 - 6)/3 = 4 + 4.667 = 8.667 s.
        assert t_norm == pytest.approx(8.667, abs=0.05)

        # The un-normalized direct path would be abs(10 - 350) = 340 deg,
        # an order of magnitude longer, the bug this fix removes.
        t_raw = estimate_slew_time(350.0, 50.0, 10.0, 50.0, site)
        assert t_raw > 100.0
        assert t_norm < t_raw / 5.0

    def test_boresight_not_flipped(self, site):
        """A north-straddling slew block uses the circular-mean mid-azimuth."""
        # az_start/az_end on opposite sides of the north wrap (355, 5): the
        # arithmetic mean is 180 (~180 deg-wrong boresight); the circular
        # mean is 0 (correct).
        block = TimelineBlock.slew(
            t_start=_SUN_TIME,
            duration=10.0,
            az_start=355.0,
            az_end=5.0,
            el=50.0,
            site=site,
            scan_index=0,
            patch_name="slew_to_test",
        )
        expected = compute_nasmyth_rotation(circular_mean_deg(355.0, 5.0), 50.0, site)
        arithmetic = compute_nasmyth_rotation(0.5 * (355.0 + 5.0), 50.0, site)

        assert block.boresight_angle == pytest.approx(expected, abs=1e-9)
        # And it is genuinely different from the arithmetic-midpoint value.
        assert abs(block.boresight_angle - arithmetic) == pytest.approx(180.0, abs=1e-6)

    def test_circular_mean_handles_straddle(self):
        """circular_mean_deg of a north-straddling pair is the true midpoint."""
        assert circular_mean_deg(355.0, 5.0) == pytest.approx(0.0, abs=1e-9)
        assert circular_mean_deg(350.0, 10.0) == pytest.approx(0.0, abs=1e-9)
        # Non-straddling pair behaves like the ordinary mean.
        assert circular_mean_deg(100.0, 120.0) == pytest.approx(110.0, abs=1e-9)


class TestSunAwarePongDuration:
    """Fix 2: pong/daisy duration is clipped to the sun-safe sub-window."""

    def test_time_until_sun_unsafe_immediate(self, coords, sun_radec, site):
        """A target already inside the exclusion radius is unsafe from t=0."""
        sun_ra, sun_dec = sun_radec
        # ~20 deg in RA from the Sun -> ~18 deg separation, inside 50 deg.
        ra = (sun_ra + 20.0) % 360
        dur = _time_until_sun_unsafe(
            ra, sun_dec, _SUN_TIME, 3600.0, coords, site.sun_avoidance.exclusion_radius
        )
        assert dur == pytest.approx(0.0, abs=1.0)

    def test_time_until_sun_unsafe_far_target(self, coords, sun_radec, site):
        """A target far from the Sun stays safe for the whole window."""
        sun_ra, sun_dec = sun_radec
        ra = (sun_ra + 120.0) % 360
        dur = _time_until_sun_unsafe(
            ra, sun_dec, _SUN_TIME, 3600.0, coords, site.sun_avoidance.exclusion_radius
        )
        assert dur == pytest.approx(3600.0, abs=1.0)

    def test_pong_duration_clipped_by_sun(self, coords, sun_radec, site):
        """A sunward pong scan is clipped though it stays above the el floor.

        The target is inside the Sun exclusion radius but well above the
        elevation limit, so an elevation-only check returns the full
        max_scan_duration, while the sun-aware clip trims it to ~0.
        """
        sun_ra, sun_dec = sun_radec
        ra = (sun_ra + 20.0) % 360
        end_time = _SUN_TIME + TimeDelta(8 * 3600, format="sec")
        overhead = OverheadModel()
        patch = ObservingPatch(
            name="sunward",
            ra_center=ra,
            dec_center=sun_dec,
            width=2.0,
            height=2.0,
            scan_type="pong",
            velocity=0.5,
        )

        # Elevation-only window is the full max_scan_duration (target is up).
        el_only = _time_until_set(
            ra, sun_dec, _SUN_TIME, 3600.0, coords, site.telescope_limits.elevation.min
        )
        assert el_only == pytest.approx(3600.0, abs=1.0)

        # But the sun-aware duration is clipped to ~0.
        dur = _compute_scan_duration(patch, _SUN_TIME, end_time, site, coords, overhead, 43.0)
        assert dur < 60.0


class TestMinDurationConstraintSun:
    """Fix 3: MinDurationConstraint forward-checks Sun exclusion."""

    def test_rejects_target_entering_exclusion(self, coords, sun_radec):
        """Above the el floor but inside the exclusion zone -> score 0.0."""
        sun_ra, sun_dec = sun_radec
        ra = (sun_ra + 20.0) % 360
        patch = ObservingPatch(
            name="sunward",
            ra_center=ra,
            dec_center=sun_dec,
            width=2.0,
            height=2.0,
            scan_type="pong",
            velocity=0.5,
        )
        constraint = MinDurationConstraint(min_duration=600.0)
        # At time + 600s the target is still high (el ~44) but ~18 deg from
        # the Sun, inside the 45 deg exclusion. An elevation-only check would
        # score 1.0; the sun forward-check rejects it.
        assert constraint.score(patch, _SUN_TIME, 0.0, 0.0, coords) == 0.0

    def test_accepts_sun_safe_high_target(self, coords):
        """A sun-safe target above the el floor still scores 1.0."""
        # ra=45, dec=-40 at the test time: el ~40, ~73 deg from the Sun.
        patch = ObservingPatch(
            name="safe",
            ra_center=45.0,
            dec_center=-40.0,
            width=2.0,
            height=2.0,
            scan_type="pong",
            velocity=0.5,
        )
        constraint = MinDurationConstraint(min_duration=600.0)
        assert constraint.score(patch, _SUN_TIME, 0.0, 0.0, coords) == 1.0
