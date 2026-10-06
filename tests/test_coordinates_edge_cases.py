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
    """Transforms stay well-behaved at and near el=90 deg.

    At the zenith azimuth is undefined (every azimuth converges on a single
    point), so elevation must still round-trip and the resulting Dec must not
    depend on which azimuth was fed in.
    """

    def test_altaz_to_radec_at_zenith(self, coordinates, site):
        """At el=90 the azimuth is meaningless, but the Dec is still the site latitude."""
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        ra, dec = coordinates.altaz_to_radec(0.0, 90.0, obstime=obstime)

        assert dec == pytest.approx(site.latitude, abs=0.5)
        assert 0 <= ra < 360

    def test_radec_at_zenith_gives_high_elevation(self, coordinates, site):
        """A source at ``dec = site latitude`` reaches ~90 deg as it transits."""
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        # RA = LST places source at meridian; dec = site latitude gives zenith
        lst = coordinates.get_lst(obstime)
        _az, el = coordinates.radec_to_altaz(lst, site.latitude, obstime=obstime)

        assert el == pytest.approx(90.0, abs=1.0)

    def test_near_zenith_stability(self, coordinates):
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        test_elevations = [85.0, 87.0, 89.0, 89.5, 89.9, 89.99]

        for el in test_elevations:
            ra, dec = coordinates.altaz_to_radec(180.0, el, obstime=obstime)
            _, el_back = coordinates.radec_to_altaz(ra, dec, obstime=obstime)

            # Vacuum round trip: closes far below a milliarcsecond even at 89.99 deg.
            assert el_back == pytest.approx(el, abs=1e-6), (
                f"Round-trip failed for el={el}: got {el_back}"
            )

    @pytest.mark.parametrize(
        "azimuth",
        [0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0],
    )
    def test_all_azimuths_at_zenith_give_same_radec(self, coordinates, azimuth):
        """Azimuth is undefined at the zenith, so every azimuth gives the same Dec."""
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        _ra_ref, dec_ref = coordinates.altaz_to_radec(0.0, 90.0, obstime=obstime)
        _ra, dec = coordinates.altaz_to_radec(azimuth, 90.0, obstime=obstime)

        assert dec == pytest.approx(dec_ref, abs=0.001)


class TestHorizonEdge:
    """Transforms round-trip at the horizon (el=0 deg).

    The horizon is where a refracted transform would bend the position most, but
    these coordinates are vacuum, so the round trip must close exactly.
    """

    def test_round_trip_at_horizon(self, coordinates):
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        az_orig, el_orig = 180.0, 0.0
        ra, dec = coordinates.altaz_to_radec(az_orig, el_orig, obstime=obstime)
        az_back, el_back = coordinates.radec_to_altaz(ra, dec, obstime=obstime)

        # Vacuum round trip; closes far below a milliarcsecond.
        assert el_back == pytest.approx(el_orig, abs=1e-6)

        # Azimuth should be close
        az_diff = abs(az_back - az_orig)
        az_diff = min(az_diff, 360 - az_diff)
        assert az_diff < 1e-6


class TestCelestialPoles:
    """The pole transform is RA-independent and stable approaching dec=-90.

    At the celestial poles, RA is undefined (all RA values converge to a point).
    This is analogous to the azimuth singularity at the zenith.
    """

    def test_south_pole_transform(self, coordinates, site):
        """From a southern site the south celestial pole sits due south at el = |lat|."""
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        az, el = coordinates.radec_to_altaz(0.0, -90.0, obstime=obstime)

        # From Chile (~-23 deg lat), SCP elevation = |latitude|, azimuth = due south
        expected_el = abs(site.latitude)
        assert el == pytest.approx(expected_el, abs=0.5)
        assert az == pytest.approx(180.0, abs=0.5)

    @pytest.mark.parametrize("ra", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
    def test_all_ra_at_poles_give_same_altaz(self, coordinates, ra):
        """RA is undefined at the poles, so every RA gives the same Az/El."""
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        az_ref, el_ref = coordinates.radec_to_altaz(0.0, -90.0, obstime=obstime)
        az, el = coordinates.radec_to_altaz(ra, -90.0, obstime=obstime)

        assert el == pytest.approx(el_ref, abs=0.001)

        az_diff = abs(az - az_ref)
        az_diff = min(az_diff, 360 - az_diff)
        assert az_diff < 0.01

    def test_near_pole_stability(self, coordinates):
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        test_decs = [-85.0, -87.0, -89.0, -89.5, -89.9, -89.99]

        for dec in test_decs:
            az, el = coordinates.radec_to_altaz(180.0, dec, obstime=obstime)
            _, dec_back = coordinates.altaz_to_radec(az, el, obstime=obstime)

            assert dec_back == pytest.approx(dec, abs=1e-6), (
                f"Round-trip failed for dec={dec}: got {dec_back}"
            )


class TestAzimuthWrapAround:
    """Transforms cross the 0/360 azimuth seam without a discontinuity.

    Azimuth is a circular coordinate that wraps from 360 back to 0, and the
    seam must be invisible to both directions, scalar and array alike.
    """

    def test_altaz_to_radec_across_north(self, coordinates):
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
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        for ra in range(0, 360, 15):
            az, _el = coordinates.radec_to_altaz(float(ra), -30.0, obstime=obstime)

            assert 0.0 <= az < 360.0

    def test_round_trip_across_azimuth_boundary(self, coordinates):
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        for az_orig in [0.0, 0.1, 359.9, 360.0]:
            el_orig = 45.0

            ra, dec = coordinates.altaz_to_radec(az_orig, el_orig, obstime=obstime)
            az_back, el_back = coordinates.radec_to_altaz(ra, dec, obstime=obstime)

            assert el_back == pytest.approx(el_orig, abs=1e-6)

            az_orig_norm = az_orig % 360
            az_back_norm = az_back % 360
            az_diff = abs(az_back_norm - az_orig_norm)
            az_diff = min(az_diff, 360 - az_diff)
            assert az_diff < 1e-6

    def test_array_input_across_boundary(self, coordinates):
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        azs = np.array([350.0, 355.0, 0.0, 5.0, 10.0])
        els = np.full_like(azs, 45.0)

        ras, decs = coordinates.altaz_to_radec(azs, els, obstime=obstime)

        assert len(ras) == 5
        assert len(decs) == 5
        assert all(0 <= ra < 360 for ra in ras)
        assert all(-90 <= dec <= 90 for dec in decs)
        for az, ra, dec in zip(azs, ras, decs):
            expected = coordinates.altaz_to_radec(float(az), 45.0, obstime=obstime)
            assert (ra, dec) == pytest.approx(expected, abs=1e-9)


class TestParallacticAngleEdgeCases:
    """The parallactic angle stays finite at the pole and through a zenith transit."""

    def test_parallactic_angle_at_pole(self, coordinates):
        """At the pole the PA value is formula-dependent, but must not be NaN or Inf."""
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        pa = coordinates.get_parallactic_angle(0.0, -90.0, obstime=obstime)
        assert np.isfinite(pa)

    def test_parallactic_angle_at_zenith_passage(self, coordinates, site):
        """Parallactic angle stays finite for a source passing near the zenith.

        A source at ``dec ~ latitude`` transits within a fraction of a degree
        of the zenith, where the parallactic angle is ill-conditioned: it is
        undefined exactly at the zenith and swings through 180 deg at transit.
        The AltAz-form computation must remain finite. It is **not** ~ 0 here
        (an HA-form would return 0 only through the ``atan2(0, 0)`` coincidence
        of forming HA = LST - RA with RA = LST).
        """
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        lst = coordinates.get_lst(obstime)
        pa = coordinates.get_parallactic_angle(lst, site.latitude, obstime=obstime)
        assert np.isfinite(pa)


class TestFieldRotationEdgeCases:
    """Field rotation stays finite at the pole and through a near-zenith transit."""

    def test_field_rotation_at_pole(self, coordinates):
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
