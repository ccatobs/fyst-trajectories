"""Tests for plan_constant_el_scan."""

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

from fyst_trajectories.coordinates import Coordinates
from fyst_trajectories.exceptions import PointingError, PointingWarning
from fyst_trajectories.patterns.configs import ConstantElScanConfig
from fyst_trajectories.planning import FieldRegion, ScanBlock, plan_constant_el_scan
from fyst_trajectories.planning._ce_geometry import (
    _compute_ce_az_range,
    _compute_ce_duration,
    _compute_ce_duration_from_lsa,
)


class TestPlanConstantElScan:
    """Tests for plan_constant_el_scan."""

    @pytest.fixture
    def ecdfs_field(self):
        """E-CDF-S field region used in the scan strategy script."""
        return FieldRegion(
            ra_center=53.117,
            dec_center=-27.808,
            width=5.0,
            height=6.7,
        )

    @pytest.fixture
    def search_time(self):
        """Provide a base search time for E-CDF-S CE scans."""
        return Time("2026-03-15T17:00:00", scale="utc")

    def test_basic_plan(self, site, ecdfs_field, search_time):
        """plan_constant_el_scan returns a ScanBlock with correct types."""
        block = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=True,
            angle=170.0,
        )

        assert isinstance(block, ScanBlock)
        assert isinstance(block.config, ConstantElScanConfig)
        assert block.duration > 0
        assert block.trajectory.n_points > 0
        assert "az_start" in block.computed_params
        assert "az_stop" in block.computed_params
        assert "az_throw" in block.computed_params
        assert "n_scans" in block.computed_params
        assert "Constant-El scan" in block.summary

    def test_elevation_in_trajectory(self, site, ecdfs_field, search_time):
        """Trajectory should be at the requested elevation."""
        block = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=True,
            angle=170.0,
        )

        assert np.allclose(block.trajectory.el, 50.0)

    def test_rising_vs_setting_different_times(self, site, ecdfs_field, search_time):
        """Rising and setting passes should have different start times."""
        rising = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=True,
            angle=170.0,
        )
        setting = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=False,
            angle=170.0,
        )

        assert rising.computed_params["start_time_iso"] != setting.computed_params["start_time_iso"]

    def test_string_start_time(self, site, ecdfs_field):
        """start_time as a string should work."""
        block = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time="2026-03-15T17:00:00",
            rising=True,
            angle=170.0,
        )

        assert block.duration > 0

    def test_azimuth_throw_positive(self, site, ecdfs_field, search_time):
        """Computed azimuth throw should be positive."""
        block = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=True,
            angle=170.0,
        )

        assert block.computed_params["az_throw"] > 0

    def test_unreachable_field_raises(self, site):
        """A field that never reaches the target elevation should raise."""
        # Dec = +70 is never reachable at el=50 from FYST
        field = FieldRegion(ra_center=180.0, dec_center=70.0, width=1.0, height=1.0)
        with pytest.raises(ValueError, match="Could not find elevation crossing"):
            plan_constant_el_scan(
                field=field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=Time("2026-03-15T00:00:00", scale="utc"),
                rising=True,
            )

    def test_science_mask_excludes_turnarounds(self, site, ecdfs_field, search_time):
        """CE scan trajectory should have turnaround samples excluded by science_mask."""
        block = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=True,
            angle=170.0,
        )

        traj = block.trajectory
        assert traj.scan_flag is not None
        science = traj.science_mask
        # science_mask should exclude some turnaround samples
        assert science.sum() < traj.n_points
        # But the majority should be science
        assert science.sum() > traj.n_points * 0.5


class TestCEGeometryWrapHandling:
    """Regression tests for the RA = 0 / az = 0/360 wrap handling.

    Both bugs were documented in ``_ce_geometry.py`` as known edge cases.
    The azimuth-wrap case is plausible at FYST's -23 deg latitude for
    sources that transit through north (dec >= +20 deg).
    """

    def test_az_range_handles_north_transit(self, site):
        """``_compute_ce_az_range`` returns a contiguous range for north-transiting sources.

        At FYST (lat = -22.99 deg), a source at RA = 0 deg, dec = +35 deg transits
        through north around 04:50 UTC on 2026-09-16, with corner azimuths
        straddling the 0/360 discontinuity. The naive ``min``/``max``
        computation reports a ~358 deg throw; the unwrapped result should
        be a few-degree throw matching the field width.
        """
        coords = Coordinates(site)
        field = FieldRegion(ra_center=0.0, dec_center=35.0, width=4.0, height=4.0)
        obs_start = Time("2026-09-16T04:30:00", scale="utc")
        obs_end = Time("2026-09-16T05:10:00", scale="utc")

        az_min, az_max = _compute_ce_az_range(
            field,
            angle=0.0,
            coords_obj=coords,
            obs_start=obs_start,
            obs_end=obs_end,
            padding=0.5,
        )

        throw = az_max - az_min
        # The field is 4 deg wide; the temporal sweep adds ~10 deg of azimuth
        # variation as it transits. A wrapped (broken) result would be
        # close to 358 deg.
        assert throw < 30.0, f"az_throw {throw:.2f} deg suggests az-wrap was not handled"
        assert throw > field.width

    def test_az_range_normal_field_unchanged(self, site):
        """``_compute_ce_az_range`` is unchanged for fields away from the discontinuity.

        A southern-hemisphere field that transits well away from north
        should never trigger the wrap-detection branch.
        """
        coords = Coordinates(site)
        # ECDFS-like field; pick an obs window when it's actually visible
        # so the per-time azimuth values are coherent.
        field = FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7)
        obs_start = Time("2026-03-15T08:30:00", scale="utc")
        obs_end = Time("2026-03-15T09:30:00", scale="utc")
        az_min, az_max = _compute_ce_az_range(
            field,
            angle=170.0,
            coords_obj=coords,
            obs_start=obs_start,
            obs_end=obs_end,
            padding=0.5,
        )
        # Throw should be modest and well inside [0, 360)
        assert 0.0 <= az_min < 360.0
        assert 0.0 <= az_max < 360.0
        assert (az_max - az_min) < 60.0

    def test_ra_wrap_handled_for_field_near_ra_zero(self, site):
        """``_compute_ce_duration`` correctly identifies edges for RA ~ 0 fields.

        A 3-deg-wide field centred at RA = 1 deg has corners at RA ~ -0.5 deg
        and ~ +2.5 deg. After ``% 360``, naive ``min``/``max`` would return
        the wrong leading/trailing edges. The wrap-detection branch
        should re-centre the values around the field centre.
        """
        coords = Coordinates(site)
        # Use a southern-hemisphere field that transits comfortably
        # above the requested elevation at FYST (lat = -23 deg).
        field = FieldRegion(ra_center=1.0, dec_center=-25.0, width=3.0, height=3.0)
        # Pick a search start time before the field rises through el=40
        # (transit happens ~03:30 UTC at this RA on this date).
        base_time = Time("2026-09-15T00:00:00", scale="utc")

        t_start, t_end, duration = _compute_ce_duration(
            field,
            angle=0.0,
            elevation=40.0,
            coords_obj=coords,
            base_search_time=base_time,
            rising=True,
        )

        # Without the wrap fix, ``min(ra_vals)``/``max(ra_vals)`` would be
        # 2.5 and 359.5; the leading edge would be searched at RA = 359.5
        # (which crosses el=40 hours later than the true RA = -0.5 edge)
        # and the reported duration would be a few-hour overestimate.
        # With the fix, the duration is the short interval between the
        # two true RA edges crossing el=40.
        assert duration > 0
        assert duration < 30 * 60  # 30 min; true value ~10 min for 3 deg width

    def test_compute_ce_duration_rejects_non_positive_step(self, site):
        """``_compute_ce_duration`` rejects a non-positive ``step_seconds``."""
        coords = Coordinates(site)
        field = FieldRegion(ra_center=1.0, dec_center=-25.0, width=3.0, height=3.0)
        base_time = Time("2026-09-15T00:00:00", scale="utc")
        with pytest.raises(ValueError, match="step_seconds must be positive"):
            _compute_ce_duration(
                field,
                angle=0.0,
                elevation=40.0,
                coords_obj=coords,
                base_search_time=base_time,
                rising=True,
                step_seconds=0,
            )

    def test_north_transit_planning_succeeds(self, site):
        """End-to-end: ``plan_constant_el_scan`` works for a north-transiting source.

        Without the az-wrap fix, ``_compute_ce_az_range`` returns a ~358 deg
        throw, which overflows the configured azimuth range and either
        crashes downstream validation or produces a nonsense scan.
        """
        field = FieldRegion(ra_center=0.0, dec_center=35.0, width=2.0, height=2.0)
        block = plan_constant_el_scan(
            field=field,
            elevation=30.0,
            velocity=0.5,
            site=site,
            start_time=Time("2026-09-16T04:30:00", scale="utc"),
            rising=False,
            angle=0.0,
        )
        # az_throw must be small (matches field width plus temporal sweep)
        assert block.computed_params["az_throw"] < 30.0

    def test_setting_north_crossing_plans_in_range(self, site):
        """A setting scan straddling north plans with azimuth inside telescope limits.

        At dec = +35 the field transits north at el ~ 32.0; at el = 31.9 the
        setting crossing sits just west of north while the field's RA span keeps
        the trailing corners east of north, so the window samples straddle 0/360
        with the majority on the west side. The planned trajectory must come out
        on the in-limits near-zero branch (as the rising pass does), not on the
        equivalent branch extending past the 360 deg azimuth limit.
        """
        field = FieldRegion(ra_center=180.0, dec_center=35.0, width=4.0, height=2.0)
        block = plan_constant_el_scan(
            field=field,
            elevation=31.9,
            velocity=0.5,
            site=site,
            start_time=Time("2026-09-15T16:13:00", scale="utc"),
            rising=False,
            angle=0.0,
        )
        lim = site.telescope_limits.azimuth
        assert block.trajectory.az.min() >= lim.min
        assert block.trajectory.az.max() <= lim.max
        assert block.computed_params["az_throw"] < 30.0

    def test_setting_north_crossing_az_range_within_limits(self, site):
        """``_compute_ce_az_range`` keeps a west-heavy straddle window within limits.

        Same regime as the end-to-end test above, exercised at the helper level:
        the setting window sits mostly west of north, and the returned interval
        must be the 360 deg branch that fits the telescope range, not the
        equivalent one extending past 360.
        """
        coords = Coordinates(site)
        field = FieldRegion(ra_center=180.0, dec_center=35.0, width=4.0, height=2.0)
        t_start, t_end, _ = _compute_ce_duration(
            field,
            angle=0.0,
            elevation=31.9,
            coords_obj=coords,
            base_search_time=Time("2026-09-15T16:13:00", scale="utc"),
            rising=False,
        )
        az_min, az_max = _compute_ce_az_range(
            field, angle=0.0, coords_obj=coords, obs_start=t_start, obs_end=t_end, padding=2.0
        )
        lim = site.telescope_limits.azimuth
        assert az_min >= lim.min
        assert az_max <= lim.max
        # The interval stays contiguous and field-sized across the crossing.
        assert 0.0 < az_max - az_min < 30.0

    def test_setting_padding_overflow_az_range_within_limits(self, site):
        """Azimuth padding alone must not push the returned interval past 360.

        A wide setting field at dec = +25, el = 40 samples azimuths up to
        ~359.3 deg without straddling north, but the 2 deg padding pushes the
        raw maximum to ~361.3 deg. The returned interval must sit on the branch
        that fits the telescope range.
        """
        coords = Coordinates(site)
        field = FieldRegion(ra_center=0.0, dec_center=25.0, width=12.0, height=12.0)
        t_start, t_end, _ = _compute_ce_duration(
            field,
            angle=0.0,
            elevation=40.0,
            coords_obj=coords,
            base_search_time=Time("2026-09-15T05:21:00", scale="utc"),
            rising=False,
        )
        az_min, az_max = _compute_ce_az_range(
            field, angle=0.0, coords_obj=coords, obs_start=t_start, obs_end=t_end, padding=2.0
        )
        lim = site.telescope_limits.azimuth
        assert az_min >= lim.min
        assert az_max <= lim.max
        assert 0.0 < az_max - az_min < 60.0

    def test_rising_north_crossing_control_on_near_zero_branch(self, site):
        """Rising-pass control: the already-in-range branch is not shifted.

        Same field, elevation, and anchor as the setting regression above but
        rising = True. The rising window (east of north) is already inside the
        telescope limits, so the branch placement must leave it on the near-zero
        branch; a spurious whole-turn shift would move the endpoints by 360 deg.
        """
        field = FieldRegion(ra_center=180.0, dec_center=35.0, width=4.0, height=2.0)
        block = plan_constant_el_scan(
            field=field,
            elevation=31.9,
            velocity=0.5,
            site=site,
            start_time=Time("2026-09-15T16:13:00", scale="utc"),
            rising=True,
            angle=0.0,
        )
        lim = site.telescope_limits.azimuth
        az = block.trajectory.az
        assert az.min() >= lim.min
        assert az.max() <= lim.max
        # Near-zero branch (loose tolerance; a whole-turn error is 360 deg off).
        cp = block.computed_params
        assert cp["az_start"] == pytest.approx(-1.12, abs=2.0)
        assert cp["az_stop"] == pytest.approx(12.49, abs=2.0)


class TestPlanConstantElLsaWindow:
    """Tests for the ``lsa_window`` kwarg on ``plan_constant_el_scan``.

    Covers no-wrap, wrap-around, equal-endpoint validation, not-found
    handling, the ``rising`` interaction, and backward compatibility of
    the default ``None``.
    """

    @pytest.fixture
    def deep56_field(self):
        """Deep56-like field centred near RA = 0, dec ~ -2 deg.

        The full Deep56 patch from sourcelist_CE.csv spans 60 deg in RA
        (23:00 -> 03:00 wraps) and 14 deg in Dec, but a 60 deg physical field
        would overflow the azimuth limits. The legacy LSA pipeline
        sweeps the patch piecewise, not as a single 60-deg-wide raster.
        For LSA-window unit tests we only need any FieldRegion that
        coexists with the legal azimuth range; the LSA branch is
        independent of field geometry.
        """
        return FieldRegion(ra_center=0.0, dec_center=-2.0, width=4.0, height=4.0)

    @pytest.fixture
    def deep56_search_time(self):
        """Search anchor for Deep56 LSA tests."""
        return Time("2026-09-15T00:00:00", scale="utc")

    def test_lsa_window_no_wrap(self, site, deep56_field, deep56_search_time):
        """A no-wrap window (22 deg, 82 deg) produces a 4-hour scan.

        Duration = (82 - 22) / 15 = 4 h. The start lies between
        ``start_time`` and ``start_time + max_search_hours``.
        """
        block = plan_constant_el_scan(
            field=deep56_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=deep56_search_time,
            rising=True,
            lsa_window=(22.0, 82.0),
        )

        assert block.duration > 0
        # Reported obs_end - obs_start (from computed_params duration)
        # tracks the LSA-derived 4-hour duration; the recomputed
        # n_scans-driven ``actual_duration`` is allowed to differ by
        # at most a leg/turnaround pair.
        obs_start = Time(block.computed_params["start_time_iso"], scale="utc")
        obs_end = Time(block.computed_params["end_time_iso"], scale="utc")
        lsa_duration_h = (obs_end - obs_start).to_value(u.hour)
        assert lsa_duration_h == pytest.approx(4.0, abs=1e-3)

        # Start lies inside the search horizon.
        assert obs_start >= deep56_search_time
        assert obs_start <= deep56_search_time + 12.0 * u.hour

    def test_lsa_window_with_wrap(self, site, deep56_field, deep56_search_time):
        """A wrap-around window (310 deg, 10 deg) produces a 4-hour scan."""
        block = plan_constant_el_scan(
            field=deep56_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=deep56_search_time,
            rising=True,
            lsa_window=(310.0, 10.0),
        )

        obs_start = Time(block.computed_params["start_time_iso"], scale="utc")
        obs_end = Time(block.computed_params["end_time_iso"], scale="utc")
        lsa_duration_h = (obs_end - obs_start).to_value(u.hour)
        # (10 - 310) mod 360 / 15 = 60 / 15 = 4 h.
        assert lsa_duration_h == pytest.approx(4.0, abs=1e-3)

        # LST at obs_start should be very close to 310 deg.
        coords = Coordinates(site)
        lst_at_start = coords.get_lst(obs_start)
        diff = (lst_at_start - 310.0 + 180.0) % 360.0 - 180.0
        assert abs(diff) < 0.05  # 0.05 deg ~ 12 s of sidereal time

    def test_lsa_window_deep56_pattern(self, site, deep56_field, deep56_search_time):
        """Exercise both Deep56 LSA configurations from sourcelist_CE.csv."""
        block_rising = plan_constant_el_scan(
            field=deep56_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=deep56_search_time,
            rising=True,
            lsa_window=(310.0, 10.0),
        )
        block_setting = plan_constant_el_scan(
            field=deep56_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=deep56_search_time,
            rising=False,
            lsa_window=(22.0, 82.0),
        )

        # Both windows are 4 hours wide.
        for block in (block_rising, block_setting):
            obs_start = Time(block.computed_params["start_time_iso"], scale="utc")
            obs_end = Time(block.computed_params["end_time_iso"], scale="utc")
            assert (obs_end - obs_start).to_value(u.hour) == pytest.approx(4.0, abs=1e-3)

        # The 22->82 setting window starts after the 310->10 rising window
        # (both anchored at the same start_time; 310 deg comes up first, then 22 deg).
        t_r = Time(block_rising.computed_params["start_time_iso"], scale="utc")
        t_s = Time(block_setting.computed_params["start_time_iso"], scale="utc")
        assert t_r < t_s

    def test_lsa_window_equal_endpoints_raises(self, site, deep56_field, deep56_search_time):
        """Equal endpoints produce a zero-duration window, refused."""
        with pytest.raises(ValueError, match="zero-duration"):
            plan_constant_el_scan(
                field=deep56_field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=deep56_search_time,
                lsa_window=(45.0, 45.0),
            )

    def test_lsa_window_out_of_range_raises(self, site, deep56_field, deep56_search_time):
        """Endpoints outside [0, 360) are rejected."""
        with pytest.raises(ValueError, match=r"\[0, 360\)"):
            plan_constant_el_scan(
                field=deep56_field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=deep56_search_time,
                lsa_window=(-10.0, 30.0),
            )

    def test_lsa_window_not_found_raises(self, site, deep56_field):
        """A tiny search horizon that never crosses min_lsa raises PointingError.

        Pick a start_time where LST is just past 100 deg and search only
        0.5 hours forward (~ 7.5 deg of LSA travel) for a target of
        100 deg, the increasing-direction crossing will already be in
        the past.
        """
        coords = Coordinates(site)
        # Find a time when LST = 100 deg + a small margin, so the next
        # 0.5 h won't include an increasing crossing of 100 deg.
        # Sample LST across a day; pick the first time LST > 100.5 deg
        # (so the next 0.5 h spans LSA ~108 deg-115 deg, never re-crossing 100 deg).
        anchor = Time("2026-09-15T00:00:00", scale="utc")
        dt = np.arange(0, 24 * 3600, 30.0)
        times = anchor + dt * u.s
        lsa = np.asarray(coords.get_lst(times))
        # First index where LSA is between 100.5 and 105 (just past 100 deg).
        idx_arr = np.flatnonzero((lsa > 100.5) & (lsa < 105.0))
        assert len(idx_arr) > 0
        start_time = times[idx_arr[0]]

        with pytest.raises(PointingError, match="not reached in increasing direction"):
            plan_constant_el_scan(
                field=deep56_field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=start_time,
                lsa_window=(100.0, 200.0),
                max_search_hours=0.5,
            )

    def test_lsa_window_independent_of_rising(self, site, deep56_field, deep56_search_time):
        """``rising`` does not affect LSA-derived start/end.

        Different ``rising`` values should produce identical
        ``obs_start`` / ``obs_end`` from the LSA branch (the flag only
        feeds the azimuth-range computation, which is allowed to
        change).
        """
        block_r = plan_constant_el_scan(
            field=deep56_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=deep56_search_time,
            rising=True,
            lsa_window=(22.0, 82.0),
        )
        block_s = plan_constant_el_scan(
            field=deep56_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=deep56_search_time,
            rising=False,
            lsa_window=(22.0, 82.0),
        )

        assert (
            block_r.computed_params["start_time_iso"] == block_s.computed_params["start_time_iso"]
        )
        assert block_r.computed_params["end_time_iso"] == block_s.computed_params["end_time_iso"]

    def test_lsa_window_none_preserves_legacy_behavior(self, site):
        """Default ``lsa_window=None`` is byte-identical to omitting the kwarg.

        Guards against accidental coupling between the new branch and
        the existing elevation-crossing path.
        """
        ecdfs_field = FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7)
        search_time = Time("2026-03-15T17:00:00", scale="utc")
        block_default = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=True,
            angle=170.0,
        )
        block_explicit_none = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=True,
            angle=170.0,
            lsa_window=None,
        )

        cp_default = block_default.computed_params
        cp_explicit = block_explicit_none.computed_params
        assert cp_default["start_time_iso"] == cp_explicit["start_time_iso"]
        assert cp_default["end_time_iso"] == cp_explicit["end_time_iso"]
        assert cp_default["az_start"] == cp_explicit["az_start"]
        assert cp_default["az_stop"] == cp_explicit["az_stop"]
        assert cp_default["n_scans"] == cp_explicit["n_scans"]
        assert cp_default["duration"] == cp_explicit["duration"]
        assert block_default.duration == block_explicit_none.duration
        # Trajectory arrays should match byte-for-byte.
        assert np.array_equal(block_default.trajectory.az, block_explicit_none.trajectory.az)
        assert np.array_equal(block_default.trajectory.el, block_explicit_none.trajectory.el)

    def test_lsa_window_at_lst_zero_crossing(self, site):
        """``min_lsa = 0`` works when LST is currently just below 360 deg.

        Regression for the wrap-edge straddle-detection bug: with a
        ``min_lsa = 0`` and consecutive samples ``(359.9, 0.1)``, the
        legacy ``(lsa - 0) * (lsa_next - 0) < 0`` product test fails
        because both factors are positive (the 0/360 boundary is not a
        true sign change in the wrapped-modulo representation). The
        unwrapped LSA-series fix locates the crossing correctly and the
        planner produces a valid 4-hour scan starting at LST ~ 0 deg.

        The field is positioned at RA = 60 deg so the LST = 0 crossing
        doesn't place the field at meridian (which would put the field
        near zenith for low |dec| and break the az-range computation).
        """
        # Find an anchor at which LST is currently ~358 deg so the first
        # increasing crossing of 0 deg (== 360 deg unwrapped) lies within a
        # short search horizon.
        coords = Coordinates(site)
        sample_anchor = Time("2026-09-15T00:00:00", scale="utc")
        dt = np.arange(0, 24 * 3600, 30.0)
        times = sample_anchor + dt * u.s
        lsa = np.asarray(coords.get_lst(times))
        # First index where LSA is in [358, 359.5], gives ~2-7 minutes
        # before the wrap crossing.
        idx_arr = np.flatnonzero((lsa > 358.0) & (lsa < 359.5))
        assert len(idx_arr) > 0
        start_time = times[idx_arr[0]]

        # Field at RA = 60 deg, dec = -25 deg: when LST crosses 0 deg the field
        # is at HA = -60 deg (rising side), well clear of meridian/zenith.
        field = FieldRegion(ra_center=60.0, dec_center=-25.0, width=4.0, height=4.0)

        block = plan_constant_el_scan(
            field=field,
            elevation=30.0,
            velocity=0.5,
            site=site,
            start_time=start_time,
            lsa_window=(0.0, 60.0),
        )

        obs_start = Time(block.computed_params["start_time_iso"], scale="utc")
        obs_end = Time(block.computed_params["end_time_iso"], scale="utc")
        lsa_duration_h = (obs_end - obs_start).to_value(u.hour)
        # (60 - 0) / 15 = 4 hours.
        assert lsa_duration_h == pytest.approx(4.0, abs=1e-3)

        # LST at obs_start should be very close to 0 deg (== 360 deg).
        lst_at_start = coords.get_lst(obs_start)
        diff = (lst_at_start - 0.0 + 180.0) % 360.0 - 180.0
        assert abs(diff) < 0.05  # 0.05 deg ~ 12 s of sidereal time

    def test_lsa_window_min_just_above_zero(self, site):
        """``min_lsa ~ 0.0001`` is in the wrap dead-zone the legacy test missed.

        With the legacy straddle test ``(lsa - 0.0001) * (lsa_next - 0.0001) < 0``,
        consecutive samples like ``(359.9, 0.1)`` produce a negative product
        only if both factors have opposite sign, but both are positive after
        the ``% 360`` wrap. The unwrap fix recognises the crossing.

        Uses the helper directly to keep the test focused on the
        wrap-detection geometry (the end-to-end planner is exercised by
        ``test_lsa_window_at_lst_zero_crossing``).
        """
        coords = Coordinates(site)
        sample_anchor = Time("2026-09-15T00:00:00", scale="utc")
        dt = np.arange(0, 24 * 3600, 30.0)
        times = sample_anchor + dt * u.s
        lsa = np.asarray(coords.get_lst(times))
        idx_arr = np.flatnonzero((lsa > 358.0) & (lsa < 359.5))
        assert len(idx_arr) > 0
        base_time = times[idx_arr[0]]

        t_start, t_end, duration = _compute_ce_duration_from_lsa(
            lsa_window=(0.0001, 60.0001),
            coords_obj=coords,
            base_search_time=base_time,
        )
        # 60 deg LSA window / 15 deg per hr = 4 h = 14400 s exactly; abs=1e-6 s absorbs round-off.
        assert duration == pytest.approx(4.0 * 3600.0, abs=1e-6)
        # The crossing of LST = 0.0001 deg lies just past the wrap.
        lst_at_start = coords.get_lst(t_start)
        diff = (lst_at_start - 0.0001 + 180.0) % 360.0 - 180.0
        assert abs(diff) < 0.05

    def test_lsa_window_json_roundtrip(self, site, deep56_field, deep56_search_time):
        """``lsa_window`` as a *list* (post-JSON shape) works through the dispatcher.

        ECSV serialisation in :func:`fyst_trajectories.overhead.io.write_timeline`
        passes ``scan_params`` through ``json.dumps``, which converts
        Python tuples to JSON arrays. The corresponding ``json.loads`` on
        read returns lists, so a round-tripped ``lsa_window`` is
        ``[310.0, 10.0]`` (list), not the tuple form. The simulation
        dispatcher must accept both and produce identical trajectories.
        """
        import json

        from fyst_trajectories.overhead.models import ObservingPatch, TimelineBlock
        from fyst_trajectories.overhead.simulation import _generate_trajectory_for_block

        scan_params_tuple = {"lsa_window": (310.0, 10.0)}
        # Simulate ECSV write -> read: tuple becomes list via JSON.
        scan_params_list = json.loads(json.dumps(scan_params_tuple))
        assert isinstance(scan_params_list["lsa_window"], list)

        patch_tuple = ObservingPatch(
            name="deep56_tuple",
            ra_center=deep56_field.ra_center,
            dec_center=deep56_field.dec_center,
            width=deep56_field.width,
            height=deep56_field.height,
            scan_type="constant_el",
            velocity=0.5,
            elevation=50.0,
            scan_params=scan_params_tuple,
        )
        patch_list = ObservingPatch(
            name="deep56_list",
            ra_center=deep56_field.ra_center,
            dec_center=deep56_field.dec_center,
            width=deep56_field.width,
            height=deep56_field.height,
            scan_type="constant_el",
            velocity=0.5,
            elevation=50.0,
            scan_params=scan_params_list,
        )

        # A 4-hour duration matches the LSA-derived window; az_start / az_end
        # are placeholders for the SCIENCE block factory (the LSA-window
        # branch recomputes both internally).
        t_start = deep56_search_time
        block_tuple = TimelineBlock.science(
            patch=patch_tuple,
            t_start=t_start,
            duration=4.0 * 3600.0,
            az_start=0.0,
            az_end=10.0,
            el=50.0,
            site=site,
            scan_index=0,
        )
        block_list = TimelineBlock.science(
            patch=patch_list,
            t_start=t_start,
            duration=4.0 * 3600.0,
            az_start=0.0,
            az_end=10.0,
            el=50.0,
            site=site,
            scan_index=0,
        )

        scan_tuple = _generate_trajectory_for_block(block_tuple, site)
        scan_list = _generate_trajectory_for_block(block_list, site)

        # Both should produce identical trajectories.
        assert (
            scan_tuple.computed_params["start_time_iso"]
            == (scan_list.computed_params["start_time_iso"])
        )
        assert (
            scan_tuple.computed_params["end_time_iso"]
            == (scan_list.computed_params["end_time_iso"])
        )
        assert scan_tuple.computed_params["duration"] == scan_list.computed_params["duration"]
        assert np.array_equal(scan_tuple.trajectory.az, scan_list.trajectory.az)

    def test_lsa_window_narrow_window(self, site, deep56_field, deep56_search_time):
        """A narrow 0.5 deg window produces a valid 2-minute scan."""
        block = plan_constant_el_scan(
            field=deep56_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=deep56_search_time,
            lsa_window=(50.0, 50.5),
        )
        # 0.5 deg / 15 = 0.0333 h = 120 s.
        obs_start = Time(block.computed_params["start_time_iso"], scale="utc")
        obs_end = Time(block.computed_params["end_time_iso"], scale="utc")
        assert (obs_end - obs_start).to_value(u.s) == pytest.approx(120.0, abs=1e-2)
        # Trajectory has positive duration and a non-empty az array.
        assert block.duration > 0
        assert block.trajectory.az.size > 0

    def test_lsa_window_invalid_max_search_hours(self, site, deep56_field, deep56_search_time):
        """Non-positive ``max_search_hours`` is rejected up front."""
        with pytest.raises(ValueError, match="max_search_hours"):
            plan_constant_el_scan(
                field=deep56_field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=deep56_search_time,
                lsa_window=(22.0, 82.0),
                max_search_hours=0.0,
            )
        with pytest.raises(ValueError, match="max_search_hours"):
            plan_constant_el_scan(
                field=deep56_field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=deep56_search_time,
                lsa_window=(22.0, 82.0),
                max_search_hours=-1.0,
            )

    def test_lsa_window_long_duration_warning(self, site, deep56_search_time):
        """Sustained > 6 h LSA windows emit a ``PointingWarning``.

        Exercises the helper directly: the warning is an
        LSA-duration-only signal, independent of field geometry, and a
        wide-enough planner call would also need its azimuth range to
        fit inside telescope limits (a separate concern).
        """
        coords = Coordinates(site)
        # 120 deg / 15 = 8 hours, past the 6 h threshold.
        with pytest.warns(PointingWarning, match="long"):
            _compute_ce_duration_from_lsa(
                lsa_window=(22.0, 142.0),
                coords_obj=coords,
                base_search_time=deep56_search_time,
                max_search_hours=24.0,
            )

    def test_lsa_window_sun_safety_recheck_invoked(
        self, site, deep56_field, deep56_search_time, monkeypatch
    ):
        """``_check_field_sun_safety`` is called twice in the LSA branch.

        The first call is at ``start_time`` (search anchor); the second
        is at the LSA-resolved ``obs_start``. The two times should be
        distinct in this scenario because the LSA window starts > 1 h
        after ``deep56_search_time``.
        """
        from fyst_trajectories.planning import constant_el as ce_module

        recorded: list[Time] = []
        original = ce_module._check_field_sun_safety

        def spy(ra, dec, t, s, sun_safe=None):
            recorded.append(t)
            return original(ra, dec, t, s, sun_safe=sun_safe)

        monkeypatch.setattr(ce_module, "_check_field_sun_safety", spy)

        plan_constant_el_scan(
            field=deep56_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=deep56_search_time,
            lsa_window=(22.0, 82.0),
        )

        assert len(recorded) == 2, f"expected 2 sun-safety calls, got {len(recorded)}"
        # The two calls should be at distinct times (start_time vs.
        # resolved obs_start, which is hours later for this window).
        dt_sec = (recorded[1] - recorded[0]).to_value(u.s)
        assert dt_sec > 60.0, (
            f"obs_start should differ from start_time by minutes-to-hours, got {dt_sec} s"
        )

    def test_lsa_window_sun_safety_single_call_in_legacy_path(self, site, monkeypatch):
        """Without ``lsa_window``, only one sun-safety call fires.

        Guards the targeted-fix property: re-checking sun safety is
        gated on the LSA branch (a regression that would also call the
        check at obs_start in the elevation-crossing path is the kind
        of surprise we want to catch early).
        """
        from fyst_trajectories.planning import constant_el as ce_module

        recorded: list[Time] = []
        original = ce_module._check_field_sun_safety

        def spy(ra, dec, t, s, sun_safe=None):
            recorded.append(t)
            return original(ra, dec, t, s, sun_safe=sun_safe)

        monkeypatch.setattr(ce_module, "_check_field_sun_safety", spy)

        ecdfs_field = FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7)
        plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=Time("2026-03-15T17:00:00", scale="utc"),
            rising=True,
            angle=170.0,
        )

        assert len(recorded) == 1

    def test_compute_ce_duration_from_lsa_zero_min_lsa(self, site):
        """Helper-level regression: ``min_lsa = 0`` is detectable.

        Direct unit test on the geometry helper, independent of the
        planner. Mirrors ``test_lsa_window_at_lst_zero_crossing`` but
        without the surrounding planner machinery.
        """
        coords = Coordinates(site)
        sample_anchor = Time("2026-09-15T00:00:00", scale="utc")
        dt = np.arange(0, 24 * 3600, 30.0)
        times = sample_anchor + dt * u.s
        lsa = np.asarray(coords.get_lst(times))
        idx_arr = np.flatnonzero((lsa > 358.0) & (lsa < 359.5))
        assert len(idx_arr) > 0
        base_time = times[idx_arr[0]]

        t_start, t_end, duration = _compute_ce_duration_from_lsa(
            lsa_window=(0.0, 60.0),
            coords_obj=coords,
            base_search_time=base_time,
        )
        # 60 deg LSA window / 15 deg per hr = 4 h = 14400 s exactly; abs=1e-6 s absorbs round-off.
        assert duration == pytest.approx(4.0 * 3600.0, abs=1e-6)
        lst_at_start = coords.get_lst(t_start)
        diff = (lst_at_start - 0.0 + 180.0) % 360.0 - 180.0
        assert abs(diff) < 0.05
