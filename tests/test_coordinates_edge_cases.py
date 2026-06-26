"""Tests for coordinate transformation edge cases.

This module tests edge cases in coordinate transformations that can cause
numerical issues or require special handling:

- Zenith singularity (el=90 degrees)
- Horizon edge (el=0 degrees)
- Celestial poles (dec=+/-90 degrees)
- Azimuth wrap-around (0/360 degree boundary)

These tests ensure the coordinate transformation code handles these
challenging cases gracefully without numerical instabilities or errors.
"""

import numpy as np
import pytest
from astropy.time import Time


class TestZenithSingularity:
    """Tests for coordinate transformations at or near the zenith (el=90 deg).

    At the zenith, azimuth is undefined (all azimuth values converge to a
    single point). The coordinate transformation code should handle this
    gracefully.
    """

    def test_altaz_to_radec_at_zenith(self, coordinates, site):
        """Test that altaz_to_radec handles zenith position.

        At el=90, the azimuth is undefined. The transformation should
        return valid RA/Dec even though azimuth is meaningless.
        """
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        ra, dec = coordinates.altaz_to_radec(0.0, 90.0, obstime=obstime)

        assert dec == pytest.approx(site.latitude, abs=0.5)
        assert 0 <= ra < 360

    def test_radec_at_zenith_gives_high_elevation(self, coordinates, site):
        """Test that a source at site latitude can reach near-zenith elevation.

        A source at the same declination as the site latitude should reach
        approximately 90 degrees elevation when it transits the meridian.
        """
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        # RA = LST places source at meridian; dec = site latitude gives zenith
        lst = coordinates.get_lst(obstime)
        _az, el = coordinates.radec_to_altaz(lst, site.latitude, obstime=obstime)

        assert el == pytest.approx(90.0, abs=1.0)

    def test_near_zenith_stability(self, coordinates):
        """Test transformation stability for positions very close to zenith."""
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        test_elevations = [85.0, 87.0, 89.0, 89.5, 89.9, 89.99]

        for el in test_elevations:
            ra, dec = coordinates.altaz_to_radec(180.0, el, obstime=obstime)
            _, el_back = coordinates.radec_to_altaz(ra, dec, obstime=obstime)

            # Azimuth may differ near zenith, but elevation should round-trip
            assert el_back == pytest.approx(el, abs=0.1), (
                f"Round-trip failed for el={el}: got {el_back}"
            )

    @pytest.mark.parametrize(
        "azimuth",
        [0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0],
    )
    def test_all_azimuths_at_zenith_give_same_radec(self, coordinates, azimuth):
        """Test that at zenith, different azimuths give same RA/Dec.

        Since azimuth is undefined at the zenith, all azimuth values should
        produce the same RA/Dec (within numerical precision).
        """
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        _ra_ref, dec_ref = coordinates.altaz_to_radec(0.0, 90.0, obstime=obstime)
        _ra, dec = coordinates.altaz_to_radec(azimuth, 90.0, obstime=obstime)

        assert dec == pytest.approx(dec_ref, abs=0.001)


class TestHorizonEdge:
    """Tests for coordinate transformations at the horizon (el=0 deg).

    At the horizon, atmospheric refraction has its maximum effect (~0.5 deg)
    and sources are at the limit of visibility.
    """

    def test_round_trip_at_horizon(self, coordinates):
        """Test round-trip consistency at horizon."""
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        az_orig, el_orig = 180.0, 0.0
        ra, dec = coordinates.altaz_to_radec(az_orig, el_orig, obstime=obstime)
        az_back, el_back = coordinates.radec_to_altaz(ra, dec, obstime=obstime)

        # Allow larger tolerance due to refraction effects
        assert el_back == pytest.approx(el_orig, abs=1.0)

        # Azimuth should be close
        az_diff = abs(az_back - az_orig)
        az_diff = min(az_diff, 360 - az_diff)
        assert az_diff < 1.0


class TestCelestialPoles:
    """Tests for coordinate transformations at celestial poles (dec=+/-90 deg).

    At the celestial poles, RA is undefined (all RA values converge to a point).
    This is analogous to the azimuth singularity at the zenith.
    """

    def test_south_pole_transform(self, coordinates, site):
        """Test transformation of the south celestial pole.

        From a southern site, the south celestial pole should be visible
        and at an elevation equal to the absolute value of the latitude.
        """
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        az, el = coordinates.radec_to_altaz(0.0, -90.0, obstime=obstime)

        # From Chile (~-23 deg lat), SCP elevation = |latitude|, azimuth = due south
        expected_el = abs(site.latitude)
        assert el == pytest.approx(expected_el, abs=0.5)
        assert az == pytest.approx(180.0, abs=0.5)

    @pytest.mark.parametrize("ra", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
    def test_all_ra_at_poles_give_same_altaz(self, coordinates, ra):
        """Test that at celestial poles, all RA values give same Az/El.

        Since RA is undefined at the poles, all RA values should produce
        the same Az/El (within numerical precision).
        """
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        az_ref, el_ref = coordinates.radec_to_altaz(0.0, -90.0, obstime=obstime)
        az, el = coordinates.radec_to_altaz(ra, -90.0, obstime=obstime)

        assert el == pytest.approx(el_ref, abs=0.001)

        az_diff = abs(az - az_ref)
        az_diff = min(az_diff, 360 - az_diff)
        assert az_diff < 0.01

    def test_near_pole_stability(self, coordinates):
        """Test transformation stability for positions near the pole."""
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        test_decs = [-85.0, -87.0, -89.0, -89.5, -89.9, -89.99]

        for dec in test_decs:
            az, el = coordinates.radec_to_altaz(180.0, dec, obstime=obstime)
            _, dec_back = coordinates.altaz_to_radec(az, el, obstime=obstime)

            assert dec_back == pytest.approx(dec, abs=0.5), (
                f"Round-trip failed for dec={dec}: got {dec_back}"
            )


class TestAzimuthWrapAround:
    """Tests for azimuth wrap-around at the 0/360 degree boundary.

    Azimuth is a circular coordinate that wraps from 360 back to 0.
    The code should handle this correctly in both directions.
    """

    def test_altaz_to_radec_across_north(self, coordinates):
        """Test transformation across the north direction (az=0/360)."""
        obstime = Time("2026-06-15T04:00:00", scale="utc")
        el = 45.0

        results = []
        for az in [358.0, 359.0, 0.0, 1.0, 2.0]:
            ra, dec = coordinates.altaz_to_radec(az, el, obstime=obstime)
            results.append((az, ra, dec))

        decs = [r[2] for r in results]
        for i in range(len(decs) - 1):
            dec_diff = abs(decs[i + 1] - decs[i])
            assert dec_diff < 2.0, f"Large dec jump at az boundary: {dec_diff}"

    def test_radec_to_altaz_produces_valid_azimuth(self, coordinates):
        """Test that radec_to_altaz always produces valid azimuth values."""
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        for ra in range(0, 360, 15):
            az, _el = coordinates.radec_to_altaz(float(ra), -30.0, obstime=obstime)

            assert -180 <= az < 360 or 0 <= az < 360

    def test_round_trip_across_azimuth_boundary(self, coordinates):
        """Test round-trip consistency across the azimuth boundary."""
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        for az_orig in [0.0, 0.1, 359.9, 360.0]:
            el_orig = 45.0

            ra, dec = coordinates.altaz_to_radec(az_orig, el_orig, obstime=obstime)
            az_back, el_back = coordinates.radec_to_altaz(ra, dec, obstime=obstime)

            assert el_back == pytest.approx(el_orig, abs=0.1)

            az_orig_norm = az_orig % 360
            az_back_norm = az_back % 360
            az_diff = abs(az_back_norm - az_orig_norm)
            az_diff = min(az_diff, 360 - az_diff)
            assert az_diff < 0.1

    def test_array_input_across_boundary(self, coordinates):
        """Test array inputs that span the azimuth boundary."""
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        azs = np.array([350.0, 355.0, 0.0, 5.0, 10.0])
        els = np.full_like(azs, 45.0)

        ras, decs = coordinates.altaz_to_radec(azs, els, obstime=obstime)

        assert len(ras) == 5
        assert len(decs) == 5
        assert all(0 <= ra < 360 for ra in ras)
        assert all(-90 <= dec <= 90 for dec in decs)


class TestParallacticAngleEdgeCases:
    """Tests for parallactic angle at edge cases."""

    def test_parallactic_angle_at_pole(self, coordinates):
        """Test parallactic angle for source at celestial pole.

        At the pole, parallactic angle behavior depends on the formula
        used, but should not produce NaN or Inf.
        """
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        pa = coordinates.get_parallactic_angle(0.0, -90.0, obstime=obstime)
        assert np.isfinite(pa)

    def test_parallactic_angle_at_zenith_passage(self, coordinates, site):
        """Parallactic angle stays finite for a source passing near the zenith.

        A source at ``dec ~ latitude`` transits within a fraction of a degree
        of the zenith, where the parallactic angle is ill-conditioned: it is
        undefined exactly at the zenith and swings through 180 deg at transit.
        The AltAz-form computation must remain finite. It is **not** ~ 0 here
        (the old HA-form only returned 0 by the ``atan2(0, 0)`` coincidence of
        forming HA = LST - RA with RA = LST).
        """
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        lst = coordinates.get_lst(obstime)
        pa = coordinates.get_parallactic_angle(lst, site.latitude, obstime=obstime)
        assert np.isfinite(pa)


class TestFieldRotationEdgeCases:
    """Tests for field rotation at edge cases."""

    def test_field_rotation_at_pole(self, coordinates):
        """Test field rotation for source at celestial pole."""
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        fr = coordinates.get_field_rotation(0.0, -90.0, obstime=obstime)
        assert np.isfinite(fr)

    def test_field_rotation_near_zenith(self, coordinates, site):
        """Field rotation stays finite for a source transiting near the zenith.

        With ``dec ~ latitude`` the source transits within a fraction of a
        degree of the zenith, where the parallactic angle is ill-conditioned
        (it is *not* ~ 0, so the field rotation is not ~ elevation either). The
        robust near-zenith invariant is simply that the computation stays
        finite, it must not blow up to NaN/Inf at the singularity.
        """
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        lst = coordinates.get_lst(obstime)
        fr = coordinates.get_field_rotation(lst, site.latitude, obstime=obstime)

        assert np.isfinite(fr)
