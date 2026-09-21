"""Tests for scheduling utilities."""

import pytest
from astropy.time import Time

from fyst_trajectories.coordinates import Coordinates
from fyst_trajectories.overhead.utils import (
    compute_nasmyth_rotation,
    estimate_slew_time,
    get_max_elevation,
    get_observable_windows,
    get_transit_time,
)
from fyst_trajectories.sun_models import _axis_slew_duration


class TestEstimateSlewTime:
    """Trapezoidal and triangular slew profiles, and the shared kinematic kernel."""

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

    def test_an_axis_that_cannot_move_costs_no_time(self, site):
        """A non-positive limit describes an immobile axis, on both sides of the seam."""
        import dataclasses

        frozen_az = dataclasses.replace(site.telescope_limits.azimuth, max_velocity=0.0)
        immobile = dataclasses.replace(
            site,
            telescope_limits=dataclasses.replace(site.telescope_limits, azimuth=frozen_az),
        )
        assert _axis_slew_duration(30.0, 0.0, 1.5) == 0.0
        assert estimate_slew_time(0.0, 45.0, 30.0, 45.0, immobile) == 0.0


class TestNasmythRotationSharesOneKernel:
    """One quantity, two entry points: RA/Dec and an already-transformed pose."""

    @pytest.mark.parametrize(
        "ra,dec",
        [(83.633, 22.014), (24.0, -32.0), (150.0, 2.2), (269.452, 4.693)],
    )
    def test_it_agrees_with_the_radec_entry_point(self, site, ra, dec):
        """Both add the Nasmyth term to the same parallactic angle, so they must match.

        Two copies of the formula would let the boresight angle a timeline
        block records drift from the field rotation the same pose reports.
        """
        coords = Coordinates(site)
        obstime = Time("2026-03-15T04:00:00", scale="utc")
        az, el = coords.radec_to_altaz(ra, dec, obstime)

        assert compute_nasmyth_rotation(az, el, site) == pytest.approx(
            coords.get_field_rotation(ra, dec, obstime), abs=1e-9
        )


class TestGetMaxElevation:
    """Transit elevation is 90 deg at the site latitude and falls with dec distance."""

    def test_overhead_source(self, site):
        max_el = get_max_elevation(0.0, site.latitude, site)
        assert abs(max_el - 90.0) < 0.01

    def test_low_source(self, site):
        max_el = get_max_elevation(0.0, 60.0, site)
        assert max_el < 10.0

    def test_moderate_source(self, site):
        max_el = get_max_elevation(0.0, -30.0, site)
        assert max_el > 80.0


class TestGetTransitTime:
    """The search returns a meridian crossing, or None if the window brackets none."""

    def test_finds_transit(self, site, start_time):
        """Verify transit is found and HA is near zero at that time."""
        transit = get_transit_time(180.0, -30.0, start_time, site)
        assert transit is not None, "Should find transit within 24 hours"
        from fyst_trajectories import Coordinates

        coords = Coordinates(site)
        ha = coords.get_hour_angle(180.0, transit)
        assert abs(ha) < 2.0  # HA near zero at transit

    def test_returns_none_if_not_found(self, site):
        # max_search_hours=0.001 yields a single sample (no interval to bracket a
        # meridian crossing), so the search is guaranteed to return None.
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        transit = get_transit_time(180.0, -30.0, t0, site, max_search_hours=0.001)
        assert transit is None


class TestGetObservableWindows:
    """Windows lie inside the search range: several, none, or one when circumpolar."""

    def test_finds_windows(self, site, start_time, end_time):
        windows = get_observable_windows(
            180.0,
            -30.0,
            start_time,
            end_time,
            site,
            min_elevation=30.0,
            check_sun=False,
        )
        assert isinstance(windows, list)
        assert len(windows) >= 1
        for rise, set_time in windows:
            assert start_time.unix <= rise.unix < set_time.unix <= end_time.unix

    def test_never_visible_source(self, site, start_time, end_time):
        windows = get_observable_windows(
            0.0,
            80.0,
            start_time,
            end_time,
            site,
            min_elevation=30.0,
            check_sun=False,
        )
        assert len(windows) == 0

    def test_circumpolar_source(self, site, start_time, end_time):
        """A genuinely circumpolar source yields one window over the full range.

        dec=-80 from FYST (lat ~ -23) never sets, it is circumpolar
        (dec < -(90 - |lat|) = -67). Its lower culmination sits near 13 deg, so
        with a 5 deg horizon it stays observable for the entire search window,
        exercising the "truly circumpolar" branch (``set_time = end_time``).
        """
        windows = get_observable_windows(
            0.0,
            -80.0,
            start_time,
            end_time,
            site,
            min_elevation=5.0,
            check_sun=False,
        )

        assert len(windows) == 1
        rise, set_time = windows[0]
        assert rise.unix == pytest.approx(start_time.unix, abs=1.0)
        assert set_time.unix == pytest.approx(end_time.unix, abs=1.0)
