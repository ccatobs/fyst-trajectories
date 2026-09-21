"""Tests for coordinate transformation module.

These tests verify coordinate transformations between celestial and
horizontal coordinate systems, including atmospheric refraction
corrections and solar system ephemeris calculations.
"""

from pathlib import Path

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

try:
    from skyfield.api import load, load_file, wgs84

    SKYFIELD_AVAILABLE = True
except ImportError:  # pragma: no cover - skyfield is a dev-only extra
    SKYFIELD_AVAILABLE = False

DE421_KERNEL = str((Path(__file__).parent / "data" / "de421_excerpt.bsp").resolve())


@pytest.fixture(scope="module")
def skyfield_de421():
    """Load the vendored de421 excerpt + a timescale once per module.

    Uses ``load_file`` on the offline excerpt in ``tests/data/`` (the same
    vendored-kernel pattern as the Titan tests), so the slow oracle never
    downloads de421 from JPL in CI.
    """
    if not SKYFIELD_AVAILABLE:
        pytest.skip("Skyfield not installed. Install with: pip install skyfield")
    eph = load_file(DE421_KERNEL)
    ts = load.timescale()
    return eph, ts


class TestRadecToAltaz:
    """Forward transform: a circumpolar source stays up, and arrays go through."""

    def test_circumpolar_source(self, coordinates):
        """From a southern site, a source near the south celestial pole never sets."""
        ra = 0.0
        dec = -85.0
        obstime = Time("2026-06-15T12:00:00", scale="utc")

        _az, el = coordinates.radec_to_altaz(ra, dec, obstime=obstime)
        assert el > 0, "South polar source should be above horizon from Chile"

    def test_array_input(self, coordinates):
        ras = np.array([0, 90, 180, 270])
        decs = np.array([-30, -30, -30, -30])
        obstime = Time("2026-06-15T12:00:00", scale="utc")

        azs, els = coordinates.radec_to_altaz(ras, decs, obstime=obstime)

        assert len(azs) == 4
        assert len(els) == 4
        assert all(-90 <= el <= 90 for el in els)
        assert all(0.0 <= az < 360.0 for az in azs)


class TestAltazToRadec:
    """Inverse transform: the round trip closes and the zenith is the site latitude."""

    def test_round_trip(self, coordinates):
        """Round-trip consistency: RA/Dec -> Az/El -> RA/Dec."""
        original_ra = 150.0
        original_dec = -30.0
        obstime = Time("2026-06-15T12:00:00", scale="utc")

        az, el = coordinates.radec_to_altaz(original_ra, original_dec, obstime=obstime)
        recovered_ra, recovered_dec = coordinates.altaz_to_radec(az, el, obstime=obstime)

        # Vacuum transform: the round trip closes to well under an arcsec.
        ra_diff = (recovered_ra - original_ra + 180) % 360 - 180

        assert ra_diff == pytest.approx(0, abs=0.02)
        assert recovered_dec == pytest.approx(original_dec, abs=0.02)

    def test_zenith_is_site_dec(self, coordinates, site):
        obstime = Time("2026-03-20T06:00:00", scale="utc")

        _ra, dec = coordinates.altaz_to_radec(0, 90, obstime=obstime)

        # Precession/nutation shift the zenith's ICRS dec ~0.14 deg from the geodetic latitude.
        assert dec == pytest.approx(site.latitude, abs=0.2)


class TestSolarSystemBodies:
    """Body ephemerides: array paths, unknown-body refusals, apparent-place guards."""

    def test_get_body_altaz_array_time(self, coordinates):
        """The vectorised Az/El path agrees with the scalar path at the shared time.

        Beyond shape, require continuous, non-zero motion across the 5-minute
        window. A broken array path returning constants, garbage, or a
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
        """The vectorised RA/Dec path matches the scalar path and stays valid."""
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
        obstime = Time("2026-06-15T12:00:00", scale="utc")
        with pytest.raises(ValueError, match="Unknown body"):
            coordinates.get_body_altaz("pluto", obstime=obstime)

    def test_invalid_body_radec(self, coordinates):
        obstime = Time("2026-06-15T12:00:00", scale="utc")
        with pytest.raises(ValueError, match="Unknown body"):
            coordinates.get_body_radec("pluto", obstime=obstime)

    @pytest.mark.slow
    def test_all_bodies_work(self, coordinates):
        obstime = Time("2026-06-15T12:00:00", scale="utc")
        for body in SOLAR_SYSTEM_BODIES:
            az, el = coordinates.get_body_altaz(body, obstime=obstime)
            assert isinstance(az, float)
            assert isinstance(el, float)

            ra, dec = coordinates.get_body_radec(body, obstime=obstime)
            assert isinstance(ra, float)
            assert isinstance(dec, float)

    @pytest.mark.parametrize(
        "epoch",
        ["2026-06-15T16:30:00", "2026-03-15T04:30:00"],
    )
    @pytest.mark.parametrize("body", ["moon", "mars", "jupiter", "saturn", "neptune", "sun"])
    def test_get_body_radec_is_apparent_position(self, coordinates, body, epoch):
        """get_body_radec returns the apparent place, not the barycentric direction.

        The returned RA/Dec must round-trip back to the same Az/El that
        ``get_body_altaz`` reports. An ``.icrs`` implementation would return the
        barycentric (SSB->body) direction, off by 7k-613k arcsec here, so this
        is the regression guard for the barycentric bug.
        """
        t = Time(epoch, scale="utc")
        ra, dec = coordinates.get_body_radec(body, obstime=t)
        az_rt, el_rt = coordinates.radec_to_altaz(ra, dec, obstime=t)
        az_body, el_body = coordinates.get_body_altaz(body, obstime=t)
        sep_arcsec = (
            np.hypot((az_rt - az_body) * np.cos(np.deg2rad(el_body)), el_rt - el_body) * 3600.0
        )
        assert sep_arcsec < 1.0

    @pytest.mark.parametrize(
        "epoch",
        ["2026-06-15T16:30:00", "2026-03-15T04:30:00"],
    )
    @pytest.mark.parametrize("body", ["moon", "mars", "jupiter", "saturn", "neptune", "sun"])
    def test_get_body_radec_parallactic_angle_geometric(self, coordinates, body, epoch):
        """get_body_radec feeds get_parallactic_angle to the geometric truth.

        Compute the geometric parallactic angle directly from the body's
        apparent Az/El (IAU AltAz form) and require ``get_parallactic_angle``
        of the reported RA/Dec to match. The barycentric ``.icrs`` gives PA
        errors up to ~348 deg (Moon); the apparent place is exact.
        """
        t = Time(epoch, scale="utc")
        az_body, el_body = coordinates.get_body_altaz(body, obstime=t)
        az_r = np.deg2rad(az_body)
        el_r = np.deg2rad(el_body)
        lat_r = np.deg2rad(coordinates.site.latitude)
        num = -np.sin(az_r) * np.cos(lat_r)
        den = np.sin(lat_r) * np.cos(el_r) - np.cos(lat_r) * np.sin(el_r) * np.cos(az_r)
        pa_truth = np.rad2deg(np.arctan2(num, den))

        ra, dec = coordinates.get_body_radec(body, obstime=t)
        pa = coordinates.get_parallactic_angle(ra, dec, obstime=t)
        dpa = abs(((pa - pa_truth + 180.0) % 360.0) - 180.0)
        assert dpa < 1e-3

    def test_get_body_radec_not_antisolar(self, coordinates):
        """The Sun's RA/Dec is the apparent place, not the barycentric anti-solar point.

        At this epoch the apparent Sun is near (83.7, +23.3); an ``.icrs`` take
        returns the anti-solar (248.9, -20.9). Guards against regressing to the
        SSB direction.
        """
        t = Time("2026-06-15T16:30:00", scale="utc")
        ra, dec = coordinates.get_body_radec("sun", obstime=t)
        assert ra == pytest.approx(83.7, abs=1.0)
        assert dec == pytest.approx(23.3, abs=1.0)
        # ...and far from the barycentric anti-solar direction.
        sep_antisolar = np.rad2deg(
            np.arccos(
                np.clip(
                    np.sin(np.deg2rad(dec)) * np.sin(np.deg2rad(-20.9))
                    + np.cos(np.deg2rad(dec))
                    * np.cos(np.deg2rad(-20.9))
                    * np.cos(np.deg2rad(ra - 248.9)),
                    -1.0,
                    1.0,
                )
            )
        )
        assert sep_antisolar > 90.0

    @pytest.mark.slow
    @pytest.mark.parametrize("body", ["moon", "mars", "jupiter", "saturn", "neptune", "sun"])
    def test_get_body_radec_matches_skyfield(self, coordinates, body, skyfield_de421):
        """Cross-check the apparent RA/Dec against skyfield (independent oracle).

        Skyfield is a dev-only dependency, imported here (never in ``src/``).
        The ~30 arcsec tolerance absorbs the astropy-vs-skyfield ephemeris and
        aberration differences; it is far tighter than the barycentric error.
        """
        from astropy.coordinates import angular_separation

        eph, ts = skyfield_de421
        observer = eph["earth"] + wgs84.latlon(
            coordinates.site.latitude,
            coordinates.site.longitude,
            elevation_m=coordinates.site.elevation,
        )
        sf_names = {
            "sun": "sun",
            "moon": "moon",
            "mars": "mars",
            "jupiter": "jupiter barycenter",
            "saturn": "saturn barycenter",
            "neptune": "neptune barycenter",
        }
        t = Time("2026-06-15T16:30:00", scale="utc")
        ra, dec = coordinates.get_body_radec(body, obstime=t)
        astrometric = observer.at(ts.from_astropy(t)).observe(eph[sf_names[body]]).apparent()
        ra_sf, dec_sf, _ = astrometric.radec()
        sep_arcsec = (
            np.rad2deg(
                angular_separation(np.deg2rad(ra), np.deg2rad(dec), ra_sf.radians, dec_sf.radians)
            )
            * 3600.0
        )
        assert sep_arcsec < 30.0


class TestAngularSeparation:
    """Separation is zero at a repeated position and 90 deg for a quarter turn."""

    def test_known_separation(self, coordinates):
        """Test separation of positions 90 degrees apart and at same position."""
        sep_same = coordinates.angular_separation(100, 45, 100, 45)
        assert sep_same == pytest.approx(0, abs=0.001)

        sep_90 = coordinates.angular_separation(0, 0, 90, 0)
        assert sep_90 == pytest.approx(90, abs=0.1)


class TestSunSafety:
    """``is_sun_safe`` clears the anti-solar direction and refuses 10 deg from the Sun."""

    def test_position_far_from_sun_is_safe(self, coordinates):
        obstime = Time("2026-06-15T18:00:00", scale="utc")
        sun_az, sun_el = coordinates.get_sun_altaz(obstime=obstime)

        test_az = (sun_az + 180) % 360
        test_el = 45.0

        sep = coordinates.angular_separation(test_az, test_el, sun_az, sun_el)
        assert sep > 45.0, f"Test position not far enough from sun (sep={sep})"

        assert coordinates.is_sun_safe(test_az, test_el, obstime=obstime)

    def test_position_near_sun_is_unsafe(self, coordinates):
        obstime = Time("2026-06-15T18:00:00", scale="utc")
        sun_az, sun_el = coordinates.get_sun_altaz(obstime=obstime)

        test_az = sun_az + 10
        test_el = sun_el

        assert not coordinates.is_sun_safe(test_az, test_el, obstime=obstime)


class TestObservability:
    """``is_position_observable`` passes a legal pose and names the axis it refuses."""

    def test_valid_position(self, coordinates):
        obstime = Time("2026-06-15T12:00:00", scale="utc")
        observable, reason = coordinates.is_position_observable(
            az=0, el=45, obstime=obstime, check_sun=False
        )
        assert observable
        assert reason == ""

    def test_elevation_too_low(self, coordinates, site):
        min_el = site.telescope_limits.elevation.min
        obstime = Time("2026-06-15T12:00:00", scale="utc")

        observable, reason = coordinates.is_position_observable(
            az=0, el=min_el - 5, obstime=obstime, check_sun=False
        )
        assert not observable
        assert "Elevation" in reason

    def test_elevation_too_high(self, coordinates, site):
        max_el = site.telescope_limits.elevation.max
        obstime = Time("2026-06-15T12:00:00", scale="utc")

        observable, reason = coordinates.is_position_observable(
            az=0, el=max_el + 5, obstime=obstime, check_sun=False
        )
        assert not observable
        assert "Elevation" in reason

    def test_azimuth_out_of_range(self, coordinates, site):
        max_az = site.telescope_limits.azimuth.max
        obstime = Time("2026-06-15T12:00:00", scale="utc")

        observable, reason = coordinates.is_position_observable(
            az=max_az + 10, el=45, obstime=obstime, check_sun=False
        )
        assert not observable
        assert "Azimuth" in reason


class TestNormalizeFrame:
    """``normalize_frame`` maps the four documented aliases and lowercases the rest."""

    def test_normalize_frame_valid(self):
        assert normalize_frame("J2000") == "icrs"
        assert normalize_frame("FK5") == "fk5"
        assert normalize_frame("B1950") == "fk4"
        assert normalize_frame("HORIZON") == "altaz"

        assert normalize_frame("j2000") == "icrs"
        assert normalize_frame("fk5") == "fk5"
        assert normalize_frame("b1950") == "fk4"
        assert normalize_frame("horizon") == "altaz"

        # Only spherical RA/Dec frames are aliased.
        expected_keys = {"J2000", "FK5", "B1950", "HORIZON"}
        assert set(FRAME_ALIASES.keys()) == expected_keys

    def test_galactic_ecliptic_not_aliased(self):
        """GALACTIC/ECLIPTIC are not aliased (they raise in the transforms).

        They are absent from ``FRAME_ALIASES`` because
        ``radec_to_altaz``/``altaz_to_radec`` read ``ra``/``dec`` and would
        reject ``l``/``b`` (galactic) or ``lon``/``lat`` (ecliptic) frames.
        An unknown name still falls through to a plain lowercase.
        """
        assert "GALACTIC" not in FRAME_ALIASES
        assert "ECLIPTIC" not in FRAME_ALIASES
        # ECLIPTIC does not map to the astropy frame name; it just lowercases.
        assert normalize_frame("ECLIPTIC") == "ecliptic"
        assert normalize_frame("GALACTIC") == "galactic"  # lowercase fallback only

    def test_normalize_frame_invalid(self):
        """Unknown frames are lowercased for astropy compatibility."""
        assert normalize_frame("MyCustomFrame") == "mycustomframe"
        assert normalize_frame("geocentric") == "geocentric"
        assert normalize_frame("icrs") == "icrs"
        assert normalize_frame("ICRS") == "icrs"
        assert normalize_frame("altaz") == "altaz"


class TestGetLst:
    """LST is in range, vectorises, and advances ~90 deg in six hours."""

    def test_lst_at_specific_time(self, coordinates):
        """LST at a known time lands in the range the equinox estimate implies.

        At midnight UTC on the vernal equinox (March 20), the LST at
        longitude 0 is approximately 12h (180 deg).
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
        times = Time(["2026-01-01T00:00:00", "2026-01-01T06:00:00"], scale="utc")
        lst = coordinates.get_lst(obstime=times)

        assert isinstance(lst, np.ndarray)
        assert len(lst) == 2
        assert all(0 <= val < 360 for val in lst)

    def test_lst_increases_with_time(self, coordinates):
        t1 = Time("2026-06-15T00:00:00", scale="utc")
        t2 = Time("2026-06-15T06:00:00", scale="utc")

        lst1 = coordinates.get_lst(obstime=t1)
        lst2 = coordinates.get_lst(obstime=t2)

        # LST should increase by ~90 degrees in 6 hours (sidereal rate)
        # Account for wrapping at 360
        diff = (lst2 - lst1) % 360
        assert diff == pytest.approx(90, abs=2)  # Within 2 degrees


class TestGetHourAngle:
    """Hour angle: the LST-minus-RA identity, array input, and zero at the meridian."""

    def test_hour_angle_is_lst_minus_ra(self, coordinates):
        """``HA = LST - RA``, normalised to [-180, 180]."""
        obstime = Time("2026-06-15T12:00:00", scale="utc")
        ra = 150.0

        lst = coordinates.get_lst(obstime=obstime)
        ha = coordinates.get_hour_angle(ra, obstime=obstime)

        expected = (lst - ra + 180) % 360 - 180
        assert ha == pytest.approx(expected, abs=0.001)

    def test_hour_angle_with_array_ra(self, coordinates):
        obstime = Time("2026-06-15T12:00:00", scale="utc")
        ras = np.array([0, 90, 180, 270])

        ha = coordinates.get_hour_angle(ras, obstime=obstime)

        assert isinstance(ha, np.ndarray)
        assert len(ha) == 4
        assert all(-180 <= h <= 180 for h in ha)

    def test_hour_angle_at_meridian(self, coordinates):
        """HA is 0 when RA equals LST."""
        obstime = Time("2026-06-15T12:00:00", scale="utc")
        lst = coordinates.get_lst(obstime=obstime)

        ha = coordinates.get_hour_angle(lst, obstime=obstime)
        assert ha == pytest.approx(0, abs=0.001)


class TestGetParallacticAngle:
    """PA is near zero at the meridian, flips sign across it, and matches the AltAz form."""

    def test_at_meridian_near_zero(self, coordinates):
        """Parallactic angle is near zero at the meridian for moderate dec.

        On the meridian (HA~0) the PA is ~0 (north is up). ``RA = LST`` is only
        an apparent-meridian proxy (catalogue RA carries a small precession
        offset), so a dec well away from the site latitude keeps the source
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

    def test_sign_east_west_of_meridian(self, coordinates):
        """PA flips sign across the meridian, seen from a southern site.

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
        (which would mix the apparent-equinox LST with the catalogue RA, the
        frame bias). This checks it equals the documented AltAz-form built
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
    """``no_refraction()`` zeroes the atmosphere; an explicit one lifts the elevation."""

    def test_no_refraction_creates_zero_pressure(self):
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
        # At ~50 deg elevation with 500 hPa the lift is ~0.007 deg.
        diff = el_refr - el_norefr
        assert 0.005 < diff < 0.1


class TestSunUsesTheEphemerisBody:
    """The Sun comes from ``get_body("sun", ..., location=...)``, not geocentric ``get_sun``."""

    def test_sun_altaz_differs_from_the_geocentric_helper(self, coordinates, site):
        """get_body('sun', location=...) differs from a geocentric get_sun() by ~arcsec.

        The library uses get_body('sun', ..., location=...) rather than the geocentric
        helper astropy.coordinates.get_sun. Both carry a finite distance, so the AltAz
        transform applies the observer offset either way (the Sun's topocentric
        parallax is about 8.8 arcsec * cos(altitude), ~3.5 arcsec here); what remains
        is the ephemeris/algorithm difference between the two, a few milliarcseconds.
        This is a sanity range-check (nonzero, well under the Sun's ~0.5 deg
        diameter), not a pinned value.
        """
        from astropy.coordinates import AltAz, get_sun

        obstime = Time("2026-03-15T16:00:00", scale="utc")
        # Topocentric (library default), uses get_body with location
        az_topo, alt_topo = coordinates.get_sun_altaz(obstime)
        # Geocentric (via legacy get_sun)
        sun_geo = get_sun(obstime)
        altaz_frame = AltAz(obstime=obstime, location=site.location)
        geo = sun_geo.transform_to(altaz_frame)
        # Difference is small but nonzero
        diff_alt = abs(alt_topo - float(geo.alt.deg))
        assert diff_alt > 0.0
        # Sun angular diameter ~0.53 deg; our shift is much smaller
        assert diff_alt < 0.5


class TestGetFieldRotation:
    """Field rotation is elevation plus PA, vectorised, and independent of the atmosphere."""

    def test_field_rotation_with_array_input(self, coordinates):
        obstime = Time("2026-06-15T08:00:00", scale="utc")
        ras = np.array([100, 150, 200])
        decs = np.array([-30, -40, -50])

        fr = coordinates.get_field_rotation(ras, decs, obstime=obstime)

        assert isinstance(fr, np.ndarray)
        assert len(fr) == 3

    def test_field_rotation_atmosphere_invariant(self, coordinates):
        """Field rotation is vacuum/geometric regardless of instance atmosphere.

        The Nasmyth elevation term and the parallactic angle are both vacuum
        quantities; a refracted elevation would leak refraction into the
        mechanical term. A refracted instance must return an identical result.
        """
        obstime = Time("2026-06-15T08:00:00", scale="utc")
        ref = Coordinates(coordinates.site, atmosphere=AtmosphericConditions.for_fyst())
        for ra, dec in [(200.0, -30.0), (83.633, 22.014), (10.0, -60.0)]:
            assert coordinates.get_field_rotation(ra, dec, obstime=obstime) == pytest.approx(
                ref.get_field_rotation(ra, dec, obstime=obstime), abs=1e-6
            )


class TestProperMotion:
    """Proper motion moves a fast star, vanishes at zero, and survives the no-distance path."""

    def test_proper_motion_makes_difference(self, coordinates):
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
        # Barnard's Star, J2000 catalogue position and proper motion
        ra = 269.452
        dec = 4.693
        pm_ra = -798.58
        pm_dec = 10328.12
        ref_epoch = Time("J2000.0")
        obstime = Time("2025-06-15T04:00:00", scale="utc")

        az_no_dist, el_no_dist = coordinates.radec_to_altaz_with_pm(
            ra, dec, pm_ra, pm_dec, ref_epoch, obstime=obstime
        )
        az_with_dist, el_with_dist = coordinates.radec_to_altaz_with_pm(
            ra, dec, pm_ra, pm_dec, ref_epoch, obstime=obstime, distance=1.83
        )

        # Both paths should agree to well below the proper-motion
        # accumulation (~0.03 deg). 0.005 deg is the tightest tolerance the
        # 1 Mpc workaround can reasonably hit; loosen if astropy or ERFA
        # changes their PM-propagation precision.
        assert az_no_dist == pytest.approx(az_with_dist, abs=0.005)
        assert el_no_dist == pytest.approx(el_with_dist, abs=0.005)


class TestObservingWavelength:
    """``obswl`` selects the radio refraction model; ``None`` leaves the default in place."""

    def test_radio_refraction_differs_from_optical(self, site):
        """Radio refraction (obswl=200 um) produces different results from optical.

        At ~23 deg elevation the difference is about 2 arcsec. The radio
        model carries the wet term, so at these conditions it refracts
        slightly more than the optical one and the radio elevation sits
        marginally higher, not lower.
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

        # Pick a source at ~23 deg elevation, where the refraction difference is
        # measurable but not extreme.
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
        """``obswl=None`` matches omitting the kwarg entirely."""
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
    """Sun predicates share the exclusion-radius boundary convention.

    ``Coordinates.is_sun_safe`` and ``trajectory_utils.validate_sun_avoidance``
    must agree on which side of ``exclusion_radius`` is unsafe; both use the
    conservative ``sep <= radius`` convention. This pins their agreement just
    inside and just outside the exclusion radius.
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

        # Pick an offset direction that keeps the target elevation inside
        # [20, 90] at this radius: a 50 deg span from a mid-sky Sun leaves
        # the observable band along the meridian, so sideways offsets are
        # tried too (0.1 deg of guard band covers the +/-0.05 deltas).
        sun = SkyCoord(sun_az * u.deg, sun_el * u.deg, frame="altaz")
        for pa in (0.0, 180.0, 90.0, 270.0):
            probe = sun.directional_offset_by(pa * u.deg, excl * u.deg)
            if 20.1 <= float(probe.alt.deg) <= 89.9:
                break
        else:
            pytest.fail("no offset direction keeps the boundary probe observable")

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

    def test_exactly_at_the_exclusion_radius_is_unsafe(self):
        """A separation exactly equal to the radius is UNSAFE, for both models.

        The convention is ``sep <= radius`` -> unsafe, shared by
        ``Coordinates.is_sun_safe`` and the scalar model from
        ``make_sun_safe``. Constructing a position whose separation is
        exactly the radius is float-fragile, so this pins the boundary from
        the other side: measure the separation of a fixed position, then
        build a site whose exclusion radius IS that number.
        """
        from fyst_trajectories import get_fyst_site
        from fyst_trajectories.coordinates import Coordinates
        from fyst_trajectories.sun_models import make_sun_safe

        t = Time("2026-03-15T16:30:00", scale="utc")
        az, el = 120.0, 45.0
        probe = Coordinates(get_fyst_site())
        sun_az, sun_el = probe.get_sun_altaz(t)
        sep = float(probe.angular_separation(az, el, sun_az, sun_el))

        at_radius = get_fyst_site(sun_exclusion_radius=sep, sun_warning_radius=sep + 5.0)
        # ``bool(...)``: the predicate answers with a numpy bool here, and the verdict,
        # not the container type, is what this pins.
        assert bool(Coordinates(at_radius).is_sun_safe(az, el, t)) is False
        assert bool(make_sun_safe("scalar", site=at_radius)(az, el, t)) is False

        # One ULP more permissive and the same position is clear, so the
        # verdict really is decided at the radius and not by a wide margin.
        just_inside = np.nextafter(sep, 0.0)
        looser = get_fyst_site(
            sun_exclusion_radius=just_inside, sun_warning_radius=just_inside + 5.0
        )
        assert bool(Coordinates(looser).is_sun_safe(az, el, t)) is True


# Vendored Titan excerpt kernel (see tests/data/README.md).
TITAN_KERNEL = str((Path(__file__).parent / "data" / "titan_excerpt.bsp").resolve())

# Frozen JPL Horizons airless apparent Az/El for Titan from FYST, an independent
# gold-standard oracle. Regenerate together with the excerpt if the window moves
# (see tests/data/README.md).
_TITAN_HORIZONS_AZEL = [
    ("2026-06-15T04:00:00", 98.336359112, -26.454688715),
    ("2026-07-15T12:00:00", 307.061067414, 49.690328506),
    ("2026-08-20T06:00:00", 43.948773558, 55.309186938),
]


class TestTitanSatelliteResolver:
    """Satellite (Titan) resolution via a JPL kernel."""

    @pytest.fixture
    def titan_coords(self, site):
        """Return a vacuum Coordinates wired to the vendored Titan excerpt kernel."""
        return Coordinates(site, satellite_kernel=TITAN_KERNEL)

    def test_titan_not_in_solar_system_bodies(self):
        """Titan is a satellite, never silently the Saturn builtin/proxy."""
        assert "titan" not in SOLAR_SYSTEM_BODIES

    def test_titan_get_body_altaz_matches_horizons(self, titan_coords):
        """get_body_altaz('titan') matches JPL Horizons airless Az/El (<= 1 arcsec)."""
        for iso, h_az, h_el in _TITAN_HORIZONS_AZEL:
            t = Time(iso, scale="utc")
            az, el = titan_coords.get_body_altaz("titan", obstime=t)
            sep = titan_coords.angular_separation(az, el, h_az, h_el) * 3600.0
            assert sep < 1.0, f"{iso}: Titan {sep:.3f} arcsec from Horizons"

    def test_titan_get_body_radec_round_trips_to_altaz(self, titan_coords):
        """get_body_radec('titan') is the apparent place: round-trips to get_body_altaz."""
        for iso, _, _ in _TITAN_HORIZONS_AZEL:
            t = Time(iso, scale="utc")
            ra, dec = titan_coords.get_body_radec("titan", obstime=t)
            az_rt, el_rt = titan_coords.radec_to_altaz(ra, dec, obstime=t)
            az_b, el_b = titan_coords.get_body_altaz("titan", obstime=t)
            sep = titan_coords.angular_separation(az_rt, el_rt, az_b, el_b) * 3600.0
            assert sep < 1.0

    def test_titan_get_body_radec_is_apparent_not_barycentric(self, titan_coords):
        """Titan RA/Dec is the apparent place, far from the barycentric .icrs trap."""
        from astropy.coordinates import SkyCoord, get_body

        t = Time("2026-06-15T04:00:00", scale="utc")
        ra, dec = titan_coords.get_body_radec("titan", obstime=t)
        apparent = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
        barycentric = get_body(
            [(0, 6), (6, 606)], t, location=titan_coords.location, ephemeris=TITAN_KERNEL
        ).icrs
        # The barycentric `.icrs` trap lands ~4.5-6.1 deg off across the window; bracket it.
        assert 3.0 < apparent.separation(barycentric).deg < 8.0

    def test_titan_requires_kernel_clear_error(self, monkeypatch, site):
        monkeypatch.delenv("FYST_SATELLITE_KERNEL", raising=False)
        coords = Coordinates(site)
        t = Time("2026-06-15T04:00:00", scale="utc")
        with pytest.raises(ValueError, match="FYST_SATELLITE_KERNEL"):
            coords.get_body_altaz("titan", obstime=t)
        with pytest.raises(ValueError, match="FYST_SATELLITE_KERNEL"):
            coords.get_body_radec("titan", obstime=t)

    def test_titan_env_var_resolves(self, monkeypatch, site):
        """FYST_SATELLITE_KERNEL resolves Titan when no explicit kwarg is given."""
        monkeypatch.setenv("FYST_SATELLITE_KERNEL", TITAN_KERNEL)
        coords = Coordinates(site)  # no explicit satellite_kernel
        az, el = coords.get_body_altaz("titan", obstime=Time("2026-06-15T04:00:00", scale="utc"))
        assert np.isfinite(az) and np.isfinite(el)

    def test_titan_get_body_radec_matches_horizons(self, titan_coords):
        """get_body_radec('titan'), round-tripped through AltAz, matches frozen Horizons.

        An independent RA/Dec oracle (sibling to the Az/El Horizons check), via the
        library's own ``altaz_to_radec`` convention, not a raw skyfield ``.radec()``
        (which differs by ~7 arcsec on axis convention).
        """
        for iso, h_az, h_el in _TITAN_HORIZONS_AZEL:
            t = Time(iso, scale="utc")
            ra, dec = titan_coords.get_body_radec("titan", obstime=t)
            az, el = titan_coords.radec_to_altaz(ra, dec, obstime=t)
            sep = titan_coords.angular_separation(az, el, h_az, h_el) * 3600.0
            assert sep < 1.0, f"{iso}: Titan radec->altaz {sep:.3f} arcsec from Horizons"

    def test_titan_missing_kernel_file_raises(self, monkeypatch, site):
        """A configured-but-missing kernel raises FileNotFoundError with an absolute path."""
        import os

        monkeypatch.delenv("FYST_SATELLITE_KERNEL", raising=False)
        coords = Coordinates(site, satellite_kernel="no_such_titan.bsp")  # relative + missing
        t = Time("2026-06-15T04:00:00", scale="utc")
        with pytest.raises(FileNotFoundError) as exc:
            coords.get_body_altaz("titan", obstime=t)
        msg = str(exc.value)
        assert "no_such_titan.bsp" in msg
        # resolved to an ABSOLUTE path (the de###-regex / cwd-relative trap is dodged)
        assert os.path.isabs(msg.rsplit(": ", 1)[1])

    def test_titan_kwarg_overrides_env(self, monkeypatch, site):
        """The explicit satellite_kernel kwarg takes precedence over the env var (both ways)."""
        t = Time("2026-06-15T04:00:00", scale="utc")
        # good kwarg beats a bogus env
        monkeypatch.setenv("FYST_SATELLITE_KERNEL", "/bogus/env_kernel.bsp")
        az, el = Coordinates(site, satellite_kernel=TITAN_KERNEL).get_body_altaz("titan", obstime=t)
        assert np.isfinite(az) and np.isfinite(el)
        # a bogus kwarg overrides a good env (precedence is the kwarg, not "first valid")
        monkeypatch.setenv("FYST_SATELLITE_KERNEL", TITAN_KERNEL)
        with pytest.raises(FileNotFoundError):
            Coordinates(site, satellite_kernel="/bogus/kwarg_kernel.bsp").get_body_altaz(
                "titan", obstime=t
            )

    def test_titan_missing_jplephem_actionable_error(self, monkeypatch, site):
        """Kernel present but jplephem absent raises an actionable [ephemeris] error."""
        import importlib.util as _iu

        real_find_spec = _iu.find_spec
        monkeypatch.setattr(
            _iu,
            "find_spec",
            lambda name, *a, **k: None if name == "jplephem" else real_find_spec(name, *a, **k),
        )
        coords = Coordinates(site, satellite_kernel=TITAN_KERNEL)
        t = Time("2026-06-15T04:00:00", scale="utc")
        with pytest.raises(ModuleNotFoundError, match="ephemeris"):
            coords.get_body_altaz("titan", obstime=t)

    @pytest.mark.slow
    def test_titan_get_body_altaz_matches_skyfield(self, titan_coords, site):
        """get_body_altaz('titan') matches skyfield eph[606] (independent oracle)."""
        pytest.importorskip("skyfield")
        from skyfield.api import load, load_file, wgs84

        eph = load_file(TITAN_KERNEL)
        ts = load.timescale()
        observer = eph[399] + wgs84.latlon(
            site.latitude, site.longitude, elevation_m=site.elevation
        )
        for iso, _, _ in _TITAN_HORIZONS_AZEL:
            t = Time(iso, scale="utc")
            # deflectors=() disables relativistic light deflection. astropy's get_body
            # likewise omits gravitational deflection for solar-system bodies, so the
            # term cancels on both sides. The reason this is safe is *cancellation*,
            # NOT smallness (the solar deflection here is ~1-2.5 arcsec, not negligible).
            # The excerpt also lacks the Jupiter barycenter the default deflector set
            # needs, which would otherwise crash skyfield.
            app = observer.at(ts.from_astropy(t)).observe(eph[606]).apparent(deflectors=())
            alt, az, _ = app.altaz()
            a_az, a_el = titan_coords.get_body_altaz("titan", obstime=t)
            sep = titan_coords.angular_separation(a_az, a_el, az.degrees, alt.degrees) * 3600.0
            # Alt/Az (unlike RA/Dec) depends on Earth orientation, and skyfield and astropy
            # handle predicted IERS EOP slightly differently, so the agreement widens for
            # epochs past the last measured IERS row (~1 arcsec at the +8 week epoch here)
            # while astropy still matches gold-standard Horizons under 1 arcsec. 2 arcsec
            # absorbs that oracle EOP difference and stays ~5 orders of magnitude inside the
            # barycentric bug this guards (7k-613k arcsec); the RA/Dec skyfield oracle uses 30.
            assert sep < 2.0, f"{iso}: Titan {sep:.3f} arcsec from skyfield"


class TestFrameAliasesReachTheTransforms:
    """``normalize_frame`` is applied where a caller supplies a frame name.

    ``FRAME_ALIASES`` and :func:`normalize_frame` are the documented
    compatibility path for control-system frame spellings, and the transform
    methods run every caller-supplied ``frame`` through the function, so
    ``frame="J2000"`` resolves instead of reaching astropy as a raw string it
    does not know.
    """

    OBSTIME = Time("2026-03-15T04:00:00", scale="utc")
    RA, DEC = 83.633, 22.014

    @pytest.mark.parametrize("alias,astropy_name", sorted(FRAME_ALIASES.items()))
    def test_radec_to_altaz_accepts_every_spherical_alias(self, coordinates, alias, astropy_name):
        """Each RA/Dec alias transforms, and agrees with the astropy name it maps to."""
        if astropy_name == "altaz":
            # The horizontal alias resolves, but this entry point reads
            # ra/dec: it must say so by name rather than dying inside astropy.
            with pytest.raises(ValueError, match="horizontal frame"):
                coordinates.radec_to_altaz(self.RA, self.DEC, self.OBSTIME, frame=alias)
            return
        aliased = coordinates.radec_to_altaz(self.RA, self.DEC, self.OBSTIME, frame=alias)
        direct = coordinates.radec_to_altaz(self.RA, self.DEC, self.OBSTIME, frame=astropy_name)
        assert aliased == pytest.approx(direct)

    @pytest.mark.parametrize(
        "call",
        [
            lambda c, t: c.radec_to_altaz(83.633, 22.014, t, frame="HORIZON"),
            lambda c, t: c.altaz_to_radec(120.0, 55.0, t, frame="HORIZON"),
            lambda c, t: c.radec_to_altaz_with_pm(
                269.452, 4.693, -798.58, 10328.12, Time("J2000.0"), obstime=t, frame="HORIZON"
            ),
        ],
        ids=["radec_to_altaz", "altaz_to_radec", "radec_to_altaz_with_pm"],
    )
    def test_horizontal_frame_is_refused_by_name(self, coordinates, call):
        """Every RA/Dec entry point names the argument rather than failing inside astropy."""
        with pytest.raises(ValueError, match="HORIZON"):
            call(coordinates, self.OBSTIME)

    def test_upper_case_astropy_name_is_accepted(self, coordinates):
        """An upper-cased astropy name is lowered rather than refused."""
        upper = coordinates.radec_to_altaz(self.RA, self.DEC, self.OBSTIME, frame="ICRS")
        lower = coordinates.radec_to_altaz(self.RA, self.DEC, self.OBSTIME, frame="icrs")
        assert upper == pytest.approx(lower)

    def test_altaz_to_radec_accepts_an_alias(self, coordinates):
        az, el = 120.0, 55.0
        aliased = coordinates.altaz_to_radec(az, el, self.OBSTIME, frame="J2000")
        direct = coordinates.altaz_to_radec(az, el, self.OBSTIME, frame="icrs")
        assert aliased == pytest.approx(direct)

    def test_proper_motion_transform_accepts_an_alias(self, coordinates):
        kwargs = dict(
            ra=269.452,
            dec=4.693,
            pm_ra=-798.58,
            pm_dec=10328.12,
            ref_epoch=Time("J2000.0"),
            obstime=self.OBSTIME,
            distance=1.8,
        )
        aliased = coordinates.radec_to_altaz_with_pm(**kwargs, frame="J2000")
        direct = coordinates.radec_to_altaz_with_pm(**kwargs, frame="icrs")
        assert aliased == pytest.approx(direct)


class TestFieldRotationSharesOneVacuumTransform:
    """The elevation term and the parallactic angle come from one transform.

    ``get_field_rotation`` reads the elevation and the parallactic angle off a
    single vacuum ``AltAz`` position, so the composed value equals the two
    published pieces exactly. That equality is what this pins.
    """

    OBSTIME = Time("2026-03-15T04:00:00", scale="utc")

    @pytest.mark.parametrize("ra,dec", [(83.633, 22.014), (24.0, -32.0), (150.0, -60.0)])
    def test_field_rotation_equals_nasmyth_term_plus_parallactic_angle(self, coordinates, ra, dec):
        el = coordinates.radec_to_altaz(ra, dec, self.OBSTIME)[1]
        pa = coordinates.get_parallactic_angle(ra, dec, self.OBSTIME)
        expected = coordinates.site.nasmyth_sign * el + pa
        assert coordinates.get_field_rotation(ra, dec, self.OBSTIME) == pytest.approx(
            expected, abs=1e-9
        )

    def test_array_form_matches_the_scalar_form(self, coordinates):
        ra = np.array([83.633, 24.0, 150.0])
        dec = np.array([22.014, -32.0, -60.0])
        vector = coordinates.get_field_rotation(ra, dec, self.OBSTIME)
        scalars = [
            coordinates.get_field_rotation(float(r), float(d), self.OBSTIME)
            for r, d in zip(ra, dec)
        ]
        assert vector == pytest.approx(scalars, abs=1e-9)
