"""Tests for coordinate transformation module.

These tests verify coordinate transformations between celestial and
horizontal coordinate systems, including atmospheric refraction
corrections and solar system ephemeris calculations.
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time, TimeDelta

from fyst_trajectories import (
    FRAME_ALIASES,
    SOLAR_SYSTEM_BODIES,
    AtmosphericConditions,
    Coordinates,
    normalize_frame,
)


class TestRadecToAltaz:
    """Tests for RA/Dec to Az/El transformation."""

    def test_circumpolar_source(self, coordinates, site):
        """Test behavior for circumpolar sources.

        For the southern site, sources near the south celestial pole
        should always be above the horizon.
        """
        ra = 0.0
        dec = -85.0
        obstime = Time("2026-06-15T12:00:00", scale="utc")

        _az, el = coordinates.radec_to_altaz(ra, dec, obstime=obstime)
        # From Chile (latitude ~-23), this should be visible
        assert el > 0, "South polar source should be above horizon from Chile"

    def test_array_input(self, coordinates):
        """Test transformation of multiple coordinates at once."""
        ras = np.array([0, 90, 180, 270])
        decs = np.array([-30, -30, -30, -30])
        obstime = Time("2026-06-15T12:00:00", scale="utc")

        azs, els = coordinates.radec_to_altaz(ras, decs, obstime=obstime)

        assert len(azs) == 4
        assert len(els) == 4
        assert all(-90 <= el <= 90 for el in els)
        assert all(0 <= az < 360 or -180 <= az < 180 for az in azs)


class TestAltazToRadec:
    """Tests for Az/El to RA/Dec transformation."""

    def test_round_trip(self, coordinates):
        """Test round-trip consistency: RA/Dec -> Az/El -> RA/Dec."""
        original_ra = 150.0
        original_dec = -30.0
        obstime = Time("2026-06-15T12:00:00", scale="utc")

        az, el = coordinates.radec_to_altaz(original_ra, original_dec, obstime=obstime)
        recovered_ra, recovered_dec = coordinates.altaz_to_radec(az, el, obstime=obstime)

        # Small differences expected due to atmospheric refraction
        ra_diff = (recovered_ra - original_ra + 180) % 360 - 180

        assert ra_diff == pytest.approx(0, abs=0.02)
        assert recovered_dec == pytest.approx(original_dec, abs=0.02)

    def test_zenith_is_site_dec(self, coordinates, site):
        """Test that zenith maps to site latitude in declination."""
        obstime = Time("2026-03-20T06:00:00", scale="utc")

        _ra, dec = coordinates.altaz_to_radec(0, 90, obstime=obstime)

        # Small deviation expected due to atmospheric refraction
        assert dec == pytest.approx(site.latitude, abs=0.2)


class TestSolarSystemBodies:
    """Tests for solar system ephemeris calculations."""

    def test_get_body_altaz_array_time(self, coordinates):
        """Test getting a body's Az/El with array of times.

        Beyond shape, pin the vectorised path to the scalar path at the first
        sample and require continuous, non-zero motion across the 5-minute
        window — a broken array path returning constants, garbage, or a
        broadcast-misaligned result fails here.
        """
        obstime = Time("2026-03-15T04:30:00", scale="utc")
        times = obstime + TimeDelta(np.arange(5) * 60 * u.s)

        az, el = coordinates.get_body_altaz("mars", obstime=times)

        assert isinstance(az, np.ndarray)
        assert isinstance(el, np.ndarray)
        assert len(az) == 5
        assert len(el) == 5

        # Array path agrees with the scalar path at the shared first time.
        az0, el0 = coordinates.get_body_altaz("mars", obstime=obstime)
        assert az[0] == pytest.approx(az0, abs=1e-4)
        assert el[0] == pytest.approx(el0, abs=1e-4)

        # Physical sanity + genuine, smooth motion over the window.
        assert np.all(np.isfinite(az)) and np.all(np.isfinite(el))
        assert np.all((el >= -90.0) & (el <= 90.0))
        assert np.ptp(el) > 0.0  # the body actually moved in elevation
        assert np.all(np.abs(np.diff(el)) < 1.0)  # but smoothly (< 1 deg/min)

    def test_get_body_radec_array_time(self, coordinates):
        """Test getting a body's RA/Dec with array of times.

        Pin the vectorised path to the scalar path at the first sample and
        require physically valid RA/Dec across the window.
        """
        obstime = Time("2026-03-15T04:30:00", scale="utc")
        times = obstime + TimeDelta(np.arange(5) * 60 * u.s)

        ra, dec = coordinates.get_body_radec("mars", obstime=times)

        assert isinstance(ra, np.ndarray)
        assert isinstance(dec, np.ndarray)
        assert len(ra) == 5
        assert len(dec) == 5

        ra0, dec0 = coordinates.get_body_radec("mars", obstime=obstime)
        assert ra[0] == pytest.approx(ra0, abs=1e-4)
        assert dec[0] == pytest.approx(dec0, abs=1e-4)

        assert np.all(np.isfinite(ra)) and np.all(np.isfinite(dec))
        assert np.all((dec >= -90.0) & (dec <= 90.0))
        assert np.all((ra >= 0.0) & (ra < 360.0))

    def test_invalid_body_altaz(self, coordinates):
        """Test error for invalid body name in get_body_altaz."""
        obstime = Time("2026-06-15T12:00:00", scale="utc")
        with pytest.raises(ValueError, match="Unknown body"):
            coordinates.get_body_altaz("pluto", obstime=obstime)

    def test_invalid_body_radec(self, coordinates):
        """Test error for invalid body name in get_body_radec."""
        obstime = Time("2026-06-15T12:00:00", scale="utc")
        with pytest.raises(ValueError, match="Unknown body"):
            coordinates.get_body_radec("pluto", obstime=obstime)

    @pytest.mark.slow
    def test_all_bodies_work(self, coordinates):
        """Test that all supported bodies can be queried."""
        obstime = Time("2026-06-15T12:00:00", scale="utc")
        for body in SOLAR_SYSTEM_BODIES:
            az, el = coordinates.get_body_altaz(body, obstime=obstime)
            assert isinstance(az, float)
            assert isinstance(el, float)

            ra, dec = coordinates.get_body_radec(body, obstime=obstime)
            assert isinstance(ra, float)
            assert isinstance(dec, float)


class TestAngularSeparation:
    """Tests for angular separation calculation."""

    def test_known_separation(self, coordinates):
        """Test separation of positions 90 degrees apart and at same position."""
        sep_same = coordinates.angular_separation(100, 45, 100, 45)
        assert sep_same == pytest.approx(0, abs=0.001)

        sep_90 = coordinates.angular_separation(0, 0, 90, 0)
        assert sep_90 == pytest.approx(90, abs=0.1)


class TestSunSafety:
    """Tests for Sun avoidance checking."""

    def test_position_far_from_sun_is_safe(self, coordinates):
        """Test that a position far from the Sun is safe."""
        obstime = Time("2026-06-15T18:00:00", scale="utc")
        sun_az, sun_el = coordinates.get_sun_altaz(obstime=obstime)

        test_az = (sun_az + 180) % 360
        test_el = 45.0

        sep = coordinates.angular_separation(test_az, test_el, sun_az, sun_el)
        assert sep > 45.0, f"Test position not far enough from sun (sep={sep})"

        assert coordinates.is_sun_safe(test_az, test_el, obstime=obstime)

    def test_position_near_sun_is_unsafe(self, coordinates):
        """Test that a position near the Sun is unsafe."""
        obstime = Time("2026-06-15T18:00:00", scale="utc")
        sun_az, sun_el = coordinates.get_sun_altaz(obstime=obstime)

        test_az = sun_az + 10
        test_el = sun_el

        assert not coordinates.is_sun_safe(test_az, test_el, obstime=obstime)


class TestObservability:
    """Tests for position observability checking."""

    def test_valid_position(self, coordinates):
        """Test that a valid position is observable."""
        obstime = Time("2026-06-15T12:00:00", scale="utc")
        observable, reason = coordinates.is_position_observable(
            az=0, el=45, obstime=obstime, check_sun=False
        )
        assert observable
        assert reason == ""

    def test_elevation_too_low(self, coordinates, site):
        """Test that low elevation is rejected."""
        min_el = site.telescope_limits.elevation.min
        obstime = Time("2026-06-15T12:00:00", scale="utc")

        observable, reason = coordinates.is_position_observable(
            az=0, el=min_el - 5, obstime=obstime, check_sun=False
        )
        assert not observable
        assert "Elevation" in reason

    def test_elevation_too_high(self, coordinates, site):
        """Test that elevation above limit is rejected."""
        max_el = site.telescope_limits.elevation.max
        obstime = Time("2026-06-15T12:00:00", scale="utc")

        observable, reason = coordinates.is_position_observable(
            az=0, el=max_el + 5, obstime=obstime, check_sun=False
        )
        assert not observable
        assert "Elevation" in reason

    def test_azimuth_out_of_range(self, coordinates, site):
        """Test that azimuth out of range is rejected."""
        max_az = site.telescope_limits.azimuth.max
        obstime = Time("2026-06-15T12:00:00", scale="utc")

        observable, reason = coordinates.is_position_observable(
            az=max_az + 10, el=45, obstime=obstime, check_sun=False
        )
        assert not observable
        assert "Azimuth" in reason


class TestNormalizeFrame:
    """Tests for the normalize_frame() function."""

    def test_normalize_frame_valid(self):
        """Test that known frame aliases are correctly normalized."""
        assert normalize_frame("J2000") == "icrs"
        assert normalize_frame("FK5") == "fk5"
        assert normalize_frame("B1950") == "fk4"
        assert normalize_frame("HORIZON") == "altaz"

        assert normalize_frame("j2000") == "icrs"
        assert normalize_frame("fk5") == "fk5"
        assert normalize_frame("b1950") == "fk4"
        assert normalize_frame("horizon") == "altaz"

        # L-5: only spherical RA/Dec frames are aliased.
        expected_keys = {"J2000", "FK5", "B1950", "HORIZON"}
        assert set(FRAME_ALIASES.keys()) == expected_keys

    def test_galactic_ecliptic_not_aliased(self):
        """L-5: GALACTIC/ECLIPTIC are not aliased (they raise in the transforms).

        They were dropped from ``FRAME_ALIASES`` because
        ``radec_to_altaz``/``altaz_to_radec`` read ``ra``/``dec`` and would
        reject ``l``/``b`` (galactic) or ``lon``/``lat`` (ecliptic) frames.
        An unknown name still falls through to a plain lowercase.
        """
        assert "GALACTIC" not in FRAME_ALIASES
        assert "ECLIPTIC" not in FRAME_ALIASES
        # ECLIPTIC no longer maps to the astropy frame name; it just lowercases.
        assert normalize_frame("ECLIPTIC") == "ecliptic"
        assert normalize_frame("GALACTIC") == "galactic"  # lowercase fallback only

    def test_normalize_frame_invalid(self):
        """Test that unknown frames are lowercased for astropy compatibility."""
        assert normalize_frame("MyCustomFrame") == "mycustomframe"
        assert normalize_frame("geocentric") == "geocentric"
        assert normalize_frame("icrs") == "icrs"
        assert normalize_frame("ICRS") == "icrs"
        assert normalize_frame("altaz") == "altaz"


class TestGetLst:
    """Tests for the get_lst() method."""

    def test_lst_at_specific_time(self, coordinates, site):
        """Test LST at a known time returns valid value.

        At midnight UTC on the vernal equinox (March 20), the LST at
        longitude 0 is approximately 12h (180 deg). We test that LST
        is computed and is in valid range.
        """
        obstime = Time("2026-03-20T00:00:00", scale="utc")
        lst = coordinates.get_lst(obstime=obstime)

        assert 0 <= lst < 360
        assert isinstance(lst, float)

        # For FYST at longitude ~-67.8 degrees, LST differs from Greenwich
        # by about -67.8/15 = -4.5 hours. At Greenwich midnight on vernal
        # equinox, LST ~ 12h, so at FYST it should be ~12h - 4.5h = 7.5h = 112.5 deg
        # This is approximate due to precession and nutation
        # We just verify it's a reasonable value
        assert 50 < lst < 180  # Reasonable range for this time/location

    def test_lst_with_array_time(self, coordinates):
        """Test that get_lst handles array of times."""
        times = Time(["2026-01-01T00:00:00", "2026-01-01T06:00:00"], scale="utc")
        lst = coordinates.get_lst(obstime=times)

        assert isinstance(lst, np.ndarray)
        assert len(lst) == 2
        assert all(0 <= val < 360 for val in lst)

    def test_lst_increases_with_time(self, coordinates):
        """Test that LST increases with time (sidereal rate)."""
        t1 = Time("2026-06-15T00:00:00", scale="utc")
        t2 = Time("2026-06-15T06:00:00", scale="utc")

        lst1 = coordinates.get_lst(obstime=t1)
        lst2 = coordinates.get_lst(obstime=t2)

        # LST should increase by ~90 degrees in 6 hours (sidereal rate)
        # Account for wrapping at 360
        diff = (lst2 - lst1) % 360
        assert diff == pytest.approx(90, abs=2)  # Within 2 degrees


class TestGetHourAngle:
    """Tests for the get_hour_angle() method."""

    def test_hour_angle_is_lst_minus_ra(self, coordinates):
        """Test that HA = LST - RA (with normalization to [-180, 180])."""
        obstime = Time("2026-06-15T12:00:00", scale="utc")
        ra = 150.0

        lst = coordinates.get_lst(obstime=obstime)
        ha = coordinates.get_hour_angle(ra, obstime=obstime)

        expected = (lst - ra + 180) % 360 - 180
        assert ha == pytest.approx(expected, abs=0.001)

    def test_hour_angle_with_array_ra(self, coordinates):
        """Test that get_hour_angle handles array of RA values."""
        obstime = Time("2026-06-15T12:00:00", scale="utc")
        ras = np.array([0, 90, 180, 270])

        ha = coordinates.get_hour_angle(ras, obstime=obstime)

        assert isinstance(ha, np.ndarray)
        assert len(ha) == 4
        assert all(-180 <= h <= 180 for h in ha)

    def test_hour_angle_at_meridian(self, coordinates):
        """Test that HA=0 when RA equals LST (at meridian)."""
        obstime = Time("2026-06-15T12:00:00", scale="utc")
        lst = coordinates.get_lst(obstime=obstime)

        ha = coordinates.get_hour_angle(lst, obstime=obstime)
        assert ha == pytest.approx(0, abs=0.001)


class TestGetParallacticAngle:
    """Tests for the get_parallactic_angle() method."""

    def test_at_meridian_near_zero(self, coordinates, site):
        """Parallactic angle is near zero at the meridian for moderate dec.

        On the meridian (HA~0) the PA is ~0 (north is up). ``RA = LST`` is only
        an apparent-meridian proxy -- catalogue RA carries a small precession
        offset -- so a dec well away from the site latitude keeps the source
        clear of the ill-conditioned zenith where that offset is amplified.
        """
        obstime = Time("2026-06-15T12:00:00", scale="utc")
        lst = coordinates.get_lst(obstime=obstime)

        # RA = LST places the object near the meridian; dec well south of the
        # latitude (-22.99) so it transits at a moderate elevation (~53 deg).
        ra = lst
        dec = -60.0

        pa = coordinates.get_parallactic_angle(ra, dec, obstime=obstime)
        assert pa == pytest.approx(0, abs=1.0)

    def test_sign_east_west_of_meridian(self, coordinates, site):
        """Test that parallactic angle has correct signs east/west of meridian.

        For sources in the southern sky (from a southern site):
        - East of meridian (negative HA): parallactic angle should be positive
        - West of meridian (positive HA): parallactic angle should be negative
        """
        obstime = Time("2026-06-15T12:00:00", scale="utc")
        lst = coordinates.get_lst(obstime=obstime)

        dec = -30.0

        ra_east = (lst + 30) % 360  # HA = -30 (east of meridian)
        pa_east = coordinates.get_parallactic_angle(ra_east, dec, obstime=obstime)

        ra_west = (lst - 30) % 360  # HA = +30 (west of meridian)
        pa_west = coordinates.get_parallactic_angle(ra_west, dec, obstime=obstime)

        assert pa_east * pa_west < 0, "Parallactic angles should have opposite signs"

    def test_parallactic_angle_with_array_input(self, coordinates):
        """Test that get_parallactic_angle handles array inputs."""
        obstime = Time("2026-06-15T12:00:00", scale="utc")
        ras = np.array([0, 90, 180, 270])
        decs = np.array([-30, -30, -30, -30])

        pa = coordinates.get_parallactic_angle(ras, decs, obstime=obstime)

        assert isinstance(pa, np.ndarray)
        assert len(pa) == 4

    def test_parallactic_angle_matches_altaz_form(self, coordinates, site):
        """PA equals the IAU AltAz-form computed from the transformed Az/El.

        ``get_parallactic_angle`` derives the parallactic angle from the
        vacuum-transformed horizontal coordinates, not from ``HA = LST - RA``
        (which would mix the apparent-equinox LST with the catalogue RA -- the
        H-1 frame bias). This checks it equals the documented AltAz-form built
        from the same transform.
        """
        obstime = Time("2026-06-15T08:00:00", scale="utc")
        ra = 120.0
        dec = -40.0

        pa = coordinates.get_parallactic_angle(ra, dec, obstime=obstime)

        az, el = coordinates.radec_to_altaz(ra, dec, obstime=obstime)
        az_rad = np.deg2rad(az)
        el_rad = np.deg2rad(el)
        lat_rad = np.deg2rad(site.latitude)

        numerator = -np.sin(az_rad) * np.cos(lat_rad)
        denominator = np.sin(lat_rad) * np.cos(el_rad) - np.cos(lat_rad) * np.sin(el_rad) * np.cos(
            az_rad
        )
        expected = np.rad2deg(np.arctan2(numerator, denominator))

        assert pa == pytest.approx(expected, abs=1e-6)


class TestNoRefraction:
    """Tests for disabling atmospheric refraction via AtmosphericConditions.no_refraction()."""

    def test_no_refraction_creates_zero_pressure(self):
        """AtmosphericConditions.no_refraction() returns pressure=0."""
        atmo = AtmosphericConditions.no_refraction()
        assert atmo.pressure == 0.0
        assert atmo.temperature == 0.0
        assert atmo.relative_humidity == 0.0

    def test_refraction_changes_elevation(self, site):
        """Explicit atmosphere produces a measurably different elevation.

        Atmospheric refraction bends light upward, so the refracted
        elevation should be higher than the geometric (no-refraction)
        elevation for a source above the horizon.
        """
        obstime = Time("2026-03-15T04:00:00", scale="utc")
        ra, dec = 180.0, -60.0

        atmo = AtmosphericConditions(pressure=500.0, temperature=270.0, relative_humidity=0.2)
        coords_refr = Coordinates(site, atmosphere=atmo)
        coords_norefr = Coordinates(site)  # default: no refraction

        _, el_refr = coords_refr.radec_to_altaz(ra, dec, obstime=obstime)
        _, el_norefr = coords_norefr.radec_to_altaz(ra, dec, obstime=obstime)

        # Refraction lifts the apparent position
        assert el_refr > el_norefr
        # At moderate elevation (~50 deg), the difference is ~0.01-0.03 deg
        diff = el_refr - el_norefr
        assert 0.005 < diff < 0.1


class TestSunUsesTopocentricBody:
    """Tests for the ``get_sun`` → ``get_body("sun", ..., location=...)`` switch."""

    def test_sun_altaz_carries_topocentric_parallax(self, coordinates, site):
        """The Sun's altaz position differs from a geocentric calculation by ~few arcsec.

        ``astropy.coordinates.get_sun`` is geocentric; switching to
        ``get_body`` with ``location=self.location`` adds the topocentric
        parallax (~8.8 arcsec). The two should differ by less than the Sun's
        angular diameter (~0.5°) but by more than zero.
        """
        from astropy.coordinates import AltAz, get_sun

        obstime = Time("2026-03-15T16:00:00", scale="utc")
        # Topocentric (library default) — uses get_body with location
        az_topo, alt_topo = coordinates.get_sun_altaz(obstime)
        # Geocentric (via legacy get_sun)
        sun_geo = get_sun(obstime)
        altaz_frame = AltAz(obstime=obstime, location=site.location)
        geo = sun_geo.transform_to(altaz_frame)
        # Difference is small but nonzero (~arcsec scale)
        diff_alt = abs(alt_topo - float(geo.alt.deg))
        assert diff_alt > 0.0
        # Sun angular diameter ~0.53°; our shift is much smaller
        assert diff_alt < 0.5


class TestGetFieldRotation:
    """Tests for the get_field_rotation() method."""

    def test_field_rotation_equals_el_plus_pa(self, coordinates):
        """Test that field rotation equals elevation plus parallactic angle."""
        obstime = Time("2026-06-15T08:00:00", scale="utc")
        ra = 150.0
        dec = -35.0

        _, el = coordinates.radec_to_altaz(ra, dec, obstime=obstime)
        pa = coordinates.get_parallactic_angle(ra, dec, obstime=obstime)
        fr = coordinates.get_field_rotation(ra, dec, obstime=obstime)

        expected = el + pa
        assert fr == pytest.approx(expected, abs=0.001)

    def test_field_rotation_with_array_input(self, coordinates):
        """Test that get_field_rotation handles array inputs."""
        obstime = Time("2026-06-15T08:00:00", scale="utc")
        ras = np.array([100, 150, 200])
        decs = np.array([-30, -40, -50])

        fr = coordinates.get_field_rotation(ras, decs, obstime=obstime)

        assert isinstance(fr, np.ndarray)
        assert len(fr) == 3


class TestProperMotion:
    """Tests for the radec_to_altaz_with_pm() method."""

    def test_proper_motion_makes_difference(self, coordinates):
        """Test that proper motion produces different result than no motion."""
        ra = 269.452
        dec = 4.693
        pm_ra = -798.58  # Barnard's Star: large proper motion (mas/yr)
        pm_dec = 10328.12
        ref_epoch = Time("J2015.5")
        obstime = Time("2025-06-15T04:00:00", scale="utc")  # ~10 years after ref epoch

        az_pm, el_pm = coordinates.radec_to_altaz_with_pm(
            ra, dec, pm_ra, pm_dec, ref_epoch, obstime=obstime
        )

        az_static, el_static = coordinates.radec_to_altaz(ra, dec, obstime=obstime)

        # ~10"/yr over ~10 years = ~100" (~0.028 deg); at least one axis should differ
        diff_az = abs(az_pm - az_static)
        diff_el = abs(el_pm - el_static)
        assert diff_az > 0.01 or diff_el > 0.01

    def test_proper_motion_zero_gives_same_result(self, coordinates):
        """Test that zero proper motion gives same result as static transform."""
        ra = 180.0
        dec = -30.0
        pm_ra = 0.0
        pm_dec = 0.0
        ref_epoch = Time("J2000.0")
        obstime = Time("2026-06-15T04:00:00", scale="utc")

        az_pm, el_pm = coordinates.radec_to_altaz_with_pm(
            ra, dec, pm_ra, pm_dec, ref_epoch, obstime=obstime
        )

        az_static, el_static = coordinates.radec_to_altaz(ra, dec, obstime=obstime)

        assert az_pm == pytest.approx(az_static, abs=0.001)
        assert el_pm == pytest.approx(el_static, abs=0.001)

    def test_barnards_star_no_distance_workaround(self, coordinates):
        """Regression guard for the 1 Mpc dummy-distance ``apply_space_motion`` workaround.

        Barnard's Star has the largest known proper motion of any
        catalogued star (~10.4 arcsec/yr in declination). This test
        compares the no-distance code path (which uses the 1 Mpc
        workaround documented in astropy issues #10092 and #10296)
        against the same call with the real distance (1.83 pc).
        Agreement to better than the proper-motion accumulation over
        10 years confirms the workaround tracks the canonical path.

        If astropy ever gains a first-class no-distance code path, the
        numeric result here will change and force a deliberate review
        of the Coordinates.radec_to_altaz_with_pm implementation.
        """
        # Barnard's Star Gaia DR2 epoch (J2015.5) parameters
        ra = 269.452
        dec = 4.693
        pm_ra = -798.58
        pm_dec = 10328.12
        ref_epoch = Time("J2015.5")
        obstime = Time("2025-06-15T04:00:00", scale="utc")

        az_no_dist, el_no_dist = coordinates.radec_to_altaz_with_pm(
            ra, dec, pm_ra, pm_dec, ref_epoch, obstime=obstime
        )
        az_with_dist, el_with_dist = coordinates.radec_to_altaz_with_pm(
            ra, dec, pm_ra, pm_dec, ref_epoch, obstime=obstime, distance=1.83
        )

        # Both paths should agree to well below the proper-motion
        # accumulation (~0.03°). 0.005° is the tightest tolerance the
        # 1 Mpc workaround can reasonably hit; loosen if astropy or ERFA
        # changes their PM-propagation precision.
        assert az_no_dist == pytest.approx(az_with_dist, abs=0.005)
        assert el_no_dist == pytest.approx(el_with_dist, abs=0.005)


class TestObservingWavelength:
    """Tests for observing wavelength (obswl) support in refraction calculations."""

    def test_radio_refraction_differs_from_optical(self, site):
        """Radio refraction (obswl=200 µm) produces different results from optical.

        At moderate elevations the difference should be ~1-2 arcsec. The radio
        model refracts LESS than optical, so radio elevation should be slightly
        lower (closer to geometric) than optical elevation for the same source.
        """
        # Typical FYST conditions at 5612 m: ~500 hPa, ~270 K
        atmo_optical = AtmosphericConditions(
            pressure=500.0, temperature=270.0, relative_humidity=0.2
        )
        atmo_radio = AtmosphericConditions(
            pressure=500.0, temperature=270.0, relative_humidity=0.2, obswl=200.0
        )

        coords_optical = Coordinates(site, atmosphere=atmo_optical)
        coords_radio = Coordinates(site, atmosphere=atmo_radio)

        # Pick a source at moderate elevation (~30 deg) where refraction
        # difference is measurable but not extreme.
        obstime = Time("2026-06-15T04:00:00", scale="utc")
        ra, dec = 180.0, -30.0

        _az_opt, el_opt = coords_optical.radec_to_altaz(ra, dec, obstime=obstime)
        _az_rad, el_rad = coords_radio.radec_to_altaz(ra, dec, obstime=obstime)

        diff_arcsec = abs(el_opt - el_rad) * 3600.0

        # The difference should be nonzero (radio != optical refraction model)
        assert diff_arcsec > 0.1, f"Expected measurable difference, got {diff_arcsec:.3f} arcsec"
        # At moderate elevation the difference should be under ~5 arcsec
        assert diff_arcsec < 5.0, f"Difference unexpectedly large: {diff_arcsec:.3f} arcsec"

    def test_obswl_none_matches_default(self, site):
        """obswl=None should behave identically to the old code (no obswl kwarg)."""
        atmo_with_none = AtmosphericConditions(
            pressure=500.0, temperature=270.0, relative_humidity=0.2, obswl=None
        )
        atmo_without = AtmosphericConditions(
            pressure=500.0, temperature=270.0, relative_humidity=0.2
        )

        coords_with = Coordinates(site, atmosphere=atmo_with_none)
        coords_without = Coordinates(site, atmosphere=atmo_without)

        obstime = Time("2026-06-15T04:00:00", scale="utc")
        ra, dec = 83.633, 22.014

        az1, el1 = coords_with.radec_to_altaz(ra, dec, obstime=obstime)
        az2, el2 = coords_without.radec_to_altaz(ra, dec, obstime=obstime)

        assert az1 == pytest.approx(az2, abs=1e-12)
        assert el1 == pytest.approx(el2, abs=1e-12)

    def test_no_refraction_ignores_obswl(self, site):
        """no_refraction() should leave obswl=None (irrelevant when pressure=0)."""
        atmo = AtmosphericConditions.no_refraction()
        assert atmo.obswl is None


class TestSunBoundaryParity:
    """Section-6.8: sun predicates share the exclusion-radius boundary convention.

    ``Coordinates.is_sun_safe`` and ``trajectory_utils.validate_sun_avoidance``
    must agree on which side of ``exclusion_radius`` is unsafe. The validator
    previously used a strict ``<`` while ``is_sun_safe`` used ``>``, so they
    disagreed exactly at the boundary; both now use the conservative
    ``sep <= radius`` convention. This pins their agreement just inside and
    just outside the exclusion radius.
    """

    def test_is_sun_safe_and_validate_agree_across_boundary(self, site):
        import warnings as _warnings

        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astropy.time import TimeDelta

        from fyst_trajectories.coordinates import Coordinates
        from fyst_trajectories.trajectory_utils import validate_sun_avoidance

        coords = Coordinates(site)
        excl = site.sun_avoidance.exclusion_radius
        t = Time("2026-03-15T16:30:00", scale="utc")  # ~local noon at FYST: sun well up
        sun_az, sun_el = coords.get_sun_altaz(t)
        assert sun_el > 0.0

        # Offset along the meridian, in whichever direction keeps the target
        # elevation inside [20, 90] so the position is genuinely observable.
        pa = 180.0 if (sun_el + excl) > 88.0 else 0.0
        sun = SkyCoord(sun_az * u.deg, sun_el * u.deg, frame="altaz")

        for delta, expect_safe in [(-0.05, False), (0.05, True)]:
            tgt = sun.directional_offset_by(pa * u.deg, (excl + delta) * u.deg)
            az, el = float(tgt.az.deg), float(tgt.alt.deg)
            assert 20.0 <= el <= 90.0

            assert coords.is_sun_safe(az, el, t) == expect_safe

            times = t + TimeDelta(np.array([0.0, 1.0]), format="sec")
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                validate_sun_avoidance(site, np.array([az, az]), np.array([el, el]), times)
            flagged = any("EXCLUSION" in str(w.message) for w in caught)
            assert flagged == (not expect_safe)

        # Exactly at the exclusion radius: the constructed separation is
        # float-fragile to a few ULP, so we assert the two predicates land on
        # the *same* side together (the ``sep <= radius`` convention), not a
        # fixed safe/unsafe value.
        tgt = sun.directional_offset_by(pa * u.deg, excl * u.deg)
        az, el = float(tgt.az.deg), float(tgt.alt.deg)
        boundary_safe = coords.is_sun_safe(az, el, t)
        times = t + TimeDelta(np.array([0.0, 1.0]), format="sec")
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            validate_sun_avoidance(site, np.array([az, az]), np.array([el, el]), times)
        boundary_flagged = any("EXCLUSION" in str(w.message) for w in caught)
        assert boundary_safe == (not boundary_flagged)
