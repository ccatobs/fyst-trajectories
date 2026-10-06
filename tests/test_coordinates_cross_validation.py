"""Cross-validation tests comparing fyst-trajectories against Skyfield.

This module validates the coordinate transformations in fyst-trajectories by
comparing results against Skyfield, an independent Python library for
high-precision astronomy calculations.

Each test class sets its own assertion threshold. Catalogue-star positions and
sidereal time agree with Skyfield to about an arcsecond; solar-system bodies
agree to under twenty arcseconds, where ephemeris version, light-time and
aberration handling differ. The thresholds sit a few times above the measured
agreement, so an Earth-orientation update cannot turn a cross-check into a flake
while an arcsecond-scale frame error still fails.

Skyfield is chosen as the reference because it:
- Uses JPL DE ephemerides for solar system positions
- Has independent implementations of coordinate transformations
- Is widely used and well-tested in the astronomy community
"""

from pathlib import Path

import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories import SOLAR_SYSTEM_BODIES

try:
    from skyfield.api import S, Star, W, load, load_file, wgs84

    SKYFIELD_AVAILABLE = True
except ImportError:
    SKYFIELD_AVAILABLE = False

DE421_KERNEL = str((Path(__file__).parent / "data" / "de421_excerpt.bsp").resolve())

pytestmark = pytest.mark.skipif(
    not SKYFIELD_AVAILABLE, reason="Skyfield not installed. Install with: pip install skyfield"
)


@pytest.fixture(scope="module")
def skyfield_timescale():
    """Load the Skyfield timescale (shared across tests for efficiency)."""
    ts = load.timescale()
    return ts


@pytest.fixture(scope="module")
def skyfield_planets():
    """Load ephemeris data for planets.

    Uses the vendored de421 excerpt (tests/data/de421_excerpt.bsp) so the slow
    cross-validation never downloads de421 from JPL in CI.
    """
    eph = load_file(DE421_KERNEL)
    return eph


@pytest.fixture(scope="module")
def fyst_topos(skyfield_planets):
    """Create Skyfield geographic location for FYST site."""
    # FYST coordinates from TCS (astro.go)
    # Latitude: -22.985639 degrees (South)
    # Longitude: -67.740278 degrees (West)
    # Elevation: 5611.8 meters
    # Skyfield uses positive values with directional indicators
    fyst = wgs84.latlon(
        22.985639 * S,  # Positive value * S = south latitude
        67.740278 * W,  # Positive value * W = west longitude
        elevation_m=5611.8,
    )
    return fyst


class TestRadecToAltazCrossValidation:
    """RA/Dec to Az/El matches Skyfield at six sky positions and four epochs."""

    # Both sides are airless (vacuum Coordinates, airless skyfield altaz); the measured
    # disagreement is under 1 arcsec per axis. 5 arcsec is headroom for Earth-orientation
    # differences, well below the ~20 arcsec of an omitted aberration or nutation term.
    POSITION_TOLERANCE = 5.0 / 3600.0  # degrees

    @pytest.fixture
    def comparison_cases(self):
        """Test cases with RA, Dec, and observation time."""
        return [
            # (ra, dec, time_str, description)
            (83.633, 22.014, "2026-03-15T04:00:00", "Crab Nebula"),
            (180.0, -30.0, "2026-06-15T08:00:00", "Arbitrary southern sky"),
            (0.0, -45.0, "2026-09-15T02:00:00", "Near south celestial pole region"),
            (270.0, -60.0, "2026-12-15T06:00:00", "Deep southern sky"),
            (45.0, -20.0, "2026-01-15T10:00:00", "Moderate declination"),
            (315.0, -70.0, "2026-07-15T00:00:00", "Very southern declination"),
        ]

    def _skyfield_radec_to_altaz(
        self,
        ra: float,
        dec: float,
        time_str: str,
        skyfield_timescale,
        skyfield_planets,
        fyst_topos,
    ) -> tuple:
        """Compute Az/El using Skyfield for comparison.

        Parameters
        ----------
        ra : float
            Right ascension in degrees (ICRS).
        dec : float
            Declination in degrees (ICRS).
        time_str : str
            ISO format UTC time string.
        skyfield_timescale : skyfield.timelib.Timescale
            Skyfield timescale object.
        skyfield_planets : skyfield.jpllib.SpiceKernel
            Skyfield ephemeris object.
        fyst_topos : skyfield.toposlib.GeographicPosition
            Skyfield geographic position for FYST.

        Returns
        -------
        az : float
            Azimuth in degrees.
        alt : float
            Altitude in degrees.
        """
        ts = skyfield_timescale
        earth = skyfield_planets["earth"]

        dt = Time(time_str, scale="utc").datetime
        t = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

        star = Star(ra_hours=ra / 15.0, dec_degrees=dec)
        observer = earth + fyst_topos
        apparent = observer.at(t).observe(star).apparent()
        alt, az, _ = apparent.altaz()

        return az.degrees, alt.degrees

    @pytest.mark.slow
    def test_radec_to_altaz_agreement(
        self,
        coordinates,
        comparison_cases,
        skyfield_timescale,
        skyfield_planets,
        fyst_topos,
    ):
        for ra, dec, time_str, description in comparison_cases:
            obstime = Time(time_str, scale="utc")

            az_ccat, el_ccat = coordinates.radec_to_altaz(ra, dec, obstime=obstime)
            az_sf, el_sf = self._skyfield_radec_to_altaz(
                ra, dec, time_str, skyfield_timescale, skyfield_planets, fyst_topos
            )

            el_diff = abs(el_ccat - el_sf)
            assert el_diff < self.POSITION_TOLERANCE, (
                f"{description}: Elevation mismatch. "
                f"ccat={el_ccat:.6f}, skyfield={el_sf:.6f}, diff={el_diff:.6f} deg"
            )

            az_diff = abs(az_ccat - az_sf)
            az_diff = min(az_diff, 360 - az_diff)
            assert az_diff < self.POSITION_TOLERANCE, (
                f"{description}: Azimuth mismatch. "
                f"ccat={az_ccat:.6f}, skyfield={az_sf:.6f}, diff={az_diff:.6f} deg"
            )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "time_str",
        [
            "2026-01-01T00:00:00",
            "2026-04-01T06:00:00",
            "2026-07-01T12:00:00",
            "2026-10-01T18:00:00",
        ],
    )
    def test_radec_to_altaz_multiple_times(
        self,
        coordinates,
        skyfield_timescale,
        skyfield_planets,
        fyst_topos,
        time_str,
    ):
        """Varying the epoch exercises Earth-orientation and precession/nutation."""
        ra, dec = 83.633, 22.014  # Crab Nebula
        obstime = Time(time_str, scale="utc")

        az_ccat, el_ccat = coordinates.radec_to_altaz(ra, dec, obstime=obstime)
        az_sf, el_sf = self._skyfield_radec_to_altaz(
            ra, dec, time_str, skyfield_timescale, skyfield_planets, fyst_topos
        )

        el_diff = abs(el_ccat - el_sf)
        az_diff = abs(az_ccat - az_sf)
        az_diff = min(az_diff, 360 - az_diff)

        assert el_diff < self.POSITION_TOLERANCE, f"El diff at {time_str}: {el_diff}"
        assert az_diff < self.POSITION_TOLERANCE, f"Az diff at {time_str}: {az_diff}"


class TestSolarSystemCrossValidation:
    """Every supported solar-system body matches Skyfield's apparent Az/El."""

    # Bodies differ more: astropy's ephemeris against skyfield's DE421, light-time and
    # aberration handling. Measured: up to ~17 arcsec (Jupiter).
    POSITION_TOLERANCE = 60.0 / 3600.0  # degrees

    @pytest.mark.slow
    @pytest.mark.parametrize("body", SOLAR_SYSTEM_BODIES)
    def test_solar_system_body_positions(
        self,
        coordinates,
        body,
        skyfield_timescale,
        skyfield_planets,
        fyst_topos,
    ):
        """Bodies move fast, so light-time and aberration set the looser tolerance."""
        time_str = "2026-06-15T04:00:00"
        obstime = Time(time_str, scale="utc")

        az_ccat, el_ccat = coordinates.get_body_altaz(body, obstime=obstime)

        ts = skyfield_timescale
        earth = skyfield_planets["earth"]
        dt = obstime.datetime
        t = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

        observer = earth + fyst_topos

        if body == "sun":
            target = skyfield_planets["sun"]
        elif body == "moon":
            target = skyfield_planets["moon"]
        else:
            target = skyfield_planets[f"{body} barycenter"]

        apparent = observer.at(t).observe(target).apparent()
        alt_sf, az_sf, _ = apparent.altaz()

        el_diff = abs(el_ccat - alt_sf.degrees)
        az_diff = abs(az_ccat - az_sf.degrees)
        az_diff = min(az_diff, 360 - az_diff)

        assert el_diff < self.POSITION_TOLERANCE, (
            f"{body}: Elevation mismatch. "
            f"ccat={el_ccat:.4f}, skyfield={alt_sf.degrees:.4f}, diff={el_diff:.4f}"
        )
        assert az_diff < self.POSITION_TOLERANCE, (
            f"{body}: Azimuth mismatch. "
            f"ccat={az_ccat:.4f}, skyfield={az_sf.degrees:.4f}, diff={az_diff:.4f}"
        )


class TestLSTCrossValidation:
    """LST matches Skyfield's own sidereal time at five points around the year."""

    LST_TOLERANCE = 0.01  # degrees (~2.4 seconds of time)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "time_str",
        [
            "2026-01-01T00:00:00",
            "2026-03-20T12:00:00",  # Near vernal equinox
            "2026-06-21T12:00:00",  # Near summer solstice
            "2026-09-22T12:00:00",  # Near autumnal equinox
            "2026-12-21T12:00:00",  # Near winter solstice
        ],
    )
    def test_lst_agreement(
        self,
        coordinates,
        time_str,
        skyfield_timescale,
        fyst_topos,
    ):
        """LST underpins every transform, so this pins the underlying time handling."""
        obstime = Time(time_str, scale="utc")

        lst_ccat = coordinates.get_lst(obstime)

        ts = skyfield_timescale
        dt = obstime.datetime
        t = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

        lst_sf = fyst_topos.lst_hours_at(t)
        lst_sf_deg = lst_sf * 15.0

        diff = abs(lst_ccat - lst_sf_deg)
        diff = min(diff, 360 - diff)

        assert diff < self.LST_TOLERANCE, (
            f"LST mismatch at {time_str}: "
            f"ccat={lst_ccat:.4f}, skyfield={lst_sf_deg:.4f}, diff={diff:.4f}"
        )


class TestProperMotionCrossValidation:
    """``radec_to_altaz_with_pm`` matches Skyfield's ``Star`` propagation.

    Skyfield's ``Star()`` object handles proper motion natively, so it is an
    independent oracle for the two highest-proper-motion catalogue stars.
    """

    POSITION_TOLERANCE = 5.0 / 3600.0  # degrees; measured under 1 arcsec per axis

    @pytest.fixture
    def high_pm_stars(self):
        """High proper-motion test stars with J2000 catalog data.

        Returns list of (name, ra_deg, dec_deg, pmra_mas_yr, pmdec_mas_yr).
        pmra is mu_ra * cos(dec) (Gaia/Hipparcos convention).
        """
        return [
            ("Barnard's Star", 269.452, 4.693, -798.58, 10328.12),
            ("Proxima Centauri", 217.429, -62.680, -3781.74, 769.47),
        ]

    def _skyfield_altaz_with_pm(
        self,
        ra_deg,
        dec_deg,
        pmra_mas_yr,
        pmdec_mas_yr,
        time_str,
        skyfield_timescale,
        skyfield_planets,
        fyst_topos,
    ):
        """Compute Az/El via Skyfield for a star with proper motion."""
        ts = skyfield_timescale
        earth = skyfield_planets["earth"]

        dt = Time(time_str, scale="utc").datetime
        t = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

        star = Star(
            ra_hours=ra_deg / 15.0,
            dec_degrees=dec_deg,
            ra_mas_per_year=pmra_mas_yr,
            dec_mas_per_year=pmdec_mas_yr,
        )

        observer = earth + fyst_topos
        apparent = observer.at(t).observe(star).apparent()
        alt, az, _ = apparent.altaz()

        return az.degrees, alt.degrees

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "time_str",
        [
            "2026-03-15T04:00:00",
            "2026-06-15T08:00:00",
            "2026-10-01T02:00:00",
        ],
    )
    def test_proper_motion_agreement(
        self,
        coordinates,
        high_pm_stars,
        time_str,
        skyfield_timescale,
        skyfield_planets,
        fyst_topos,
    ):
        ref_epoch = Time("J2000.0")

        for name, ra, dec, pmra, pmdec in high_pm_stars:
            obstime = Time(time_str, scale="utc")

            az_ccat, el_ccat = coordinates.radec_to_altaz_with_pm(
                ra,
                dec,
                pmra,
                pmdec,
                ref_epoch,
                obstime=obstime,
            )
            az_sf, el_sf = self._skyfield_altaz_with_pm(
                ra,
                dec,
                pmra,
                pmdec,
                time_str,
                skyfield_timescale,
                skyfield_planets,
                fyst_topos,
            )

            # Skip comparison if the star is below the horizon in both
            if el_sf < -5 and el_ccat < -5:
                continue

            el_diff = abs(el_ccat - el_sf)
            assert el_diff < self.POSITION_TOLERANCE, (
                f"{name} at {time_str}: Elevation mismatch. "
                f"ccat={el_ccat:.6f}, skyfield={el_sf:.6f}, diff={el_diff:.6f} deg"
            )

            az_diff = abs(az_ccat - az_sf)
            az_diff = min(az_diff, 360 - az_diff)
            assert az_diff < self.POSITION_TOLERANCE, (
                f"{name} at {time_str}: Azimuth mismatch. "
                f"ccat={az_ccat:.6f}, skyfield={az_sf:.6f}, diff={az_diff:.6f} deg"
            )

    @pytest.mark.slow
    def test_array_obstime_agreement(
        self,
        coordinates,
        high_pm_stars,
        skyfield_timescale,
        skyfield_planets,
        fyst_topos,
    ):
        """One call over the three instants as an array matches Skyfield at each."""
        time_strs = ["2026-03-15T04:00:00", "2026-06-15T08:00:00", "2026-10-01T02:00:00"]
        obstimes = Time(time_strs, scale="utc")

        for name, ra, dec, pmra, pmdec in high_pm_stars:
            az_arr, el_arr = coordinates.radec_to_altaz_with_pm(
                ra, dec, pmra, pmdec, Time("J2000.0"), obstime=obstimes
            )
            for i, time_str in enumerate(time_strs):
                az_sf, el_sf = self._skyfield_altaz_with_pm(
                    ra,
                    dec,
                    pmra,
                    pmdec,
                    time_str,
                    skyfield_timescale,
                    skyfield_planets,
                    fyst_topos,
                )
                if el_sf < -5 and el_arr[i] < -5:
                    continue
                el_diff = abs(el_arr[i] - el_sf)
                assert el_diff < self.POSITION_TOLERANCE, (
                    f"{name} at {time_str}: elevation diff {el_diff:.6f} deg"
                )
                az_diff = abs(az_arr[i] - az_sf)
                az_diff = min(az_diff, 360 - az_diff)
                assert az_diff < self.POSITION_TOLERANCE, (
                    f"{name} at {time_str}: azimuth diff {az_diff:.6f} deg"
                )


class TestRiseSetCrossValidation:
    """Rise and set times match Skyfield's ``find_risings``/``find_settings``.

    fyst-trajectories uses linear interpolation on a coarse grid; Skyfield
    uses root-finding. Both run without refraction (pressure=0).
    """

    TIME_TOLERANCE_MINUTES = 2.0  # coarse grid vs root-finding

    @pytest.mark.slow
    def test_sirius_rise_set(
        self,
        coordinates,
        skyfield_timescale,
        skyfield_planets,
        fyst_topos,
    ):
        """Sirius (RA=101.29, Dec=-16.72) rises and sets normally at FYST latitude."""
        from skyfield.almanac import find_risings, find_settings

        ra, dec = 101.29, -16.72
        horizon = 0.0
        start_time = Time("2026-03-15T00:00:00", scale="utc")

        # fyst-trajectories rise/set (uses pressure=0 internally)
        rise_ccat, set_ccat = coordinates.get_rise_set_times(
            ra,
            dec,
            start_time=start_time,
            horizon=horizon,
            max_search_hours=36.0,
            step_hours=0.05,
        )
        ts = skyfield_timescale
        observer = skyfield_planets["earth"] + fyst_topos
        star = Star(ra_hours=ra / 15.0, dec_degrees=dec)
        t0 = ts.from_astropy(start_time)
        t1 = ts.from_astropy(start_time + TimeDelta(36 * 3600, format="sec"))
        rise_times_sf, _ = find_risings(observer, star, t0, t1, horizon_degrees=horizon)
        # The library reports the first set AFTER the rise, so search from there.
        set_times_sf, _ = find_settings(
            observer, star, rise_times_sf[0], t1, horizon_degrees=horizon
        )
        assert rise_ccat is not None and set_ccat is not None
        for label, ours, theirs in (
            ("rise", rise_ccat, rise_times_sf[0]),
            ("set", set_ccat, set_times_sf[0]),
        ):
            diff_minutes = abs(ours.tt.jd - theirs.tt) * 24 * 60
            assert diff_minutes < self.TIME_TOLERANCE_MINUTES, (
                f"{label} mismatch: ccat={ours.iso}, diff={diff_minutes:.2f} min"
            )


class TestRefractionIsolation:
    """The refraction delta matches Skyfield's, so transform differences cancel.

    Comparing the delta (with-atmosphere minus no-atmosphere) rather than
    absolute positions cancels systematic differences in the coordinate
    transforms, isolating the refraction model agreement.
    """

    # The two refraction models' deltas agree to ~0.0003 deg at ~49 deg elevation
    # (measured), against a ~0.007 deg delta.
    REFRACTION_DELTA_TOLERANCE = 0.001  # degrees

    @pytest.mark.slow
    def test_refraction_delta_agreement(
        self,
        site,
        skyfield_timescale,
        skyfield_planets,
        fyst_topos,
    ):
        """Refraction deltas match at moderate elevation (~50 deg).

        At ~50 deg elevation with ~500 hPa pressure (FYST altitude),
        refraction shifts apparent position by ~0.007 deg.
        """
        from fyst_trajectories import Coordinates
        from fyst_trajectories.site import AtmosphericConditions

        # Typical conditions for Cerro Chajnantor (~5612m)
        atmo = AtmosphericConditions(pressure=500.0, temperature=270.0, relative_humidity=0.2)

        coords_refracted = Coordinates(site, atmosphere=atmo)
        coords_vacuum = Coordinates(
            site,
            atmosphere=AtmosphericConditions.no_refraction(),
        )

        # RA=180, Dec=-30 at 02:00 UTC gives ~49 deg elevation from FYST
        ra, dec = 180.0, -30.0
        time_str = "2026-06-15T02:00:00"
        obstime = Time(time_str, scale="utc")

        _, el_refracted = coords_refracted.radec_to_altaz(
            ra,
            dec,
            obstime=obstime,
        )
        _, el_vacuum = coords_vacuum.radec_to_altaz(
            ra,
            dec,
            obstime=obstime,
        )
        delta_ccat = el_refracted - el_vacuum

        ts = skyfield_timescale
        earth = skyfield_planets["earth"]
        dt = obstime.datetime
        t = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

        star = Star(ra_hours=ra / 15.0, dec_degrees=dec)
        observer = earth + fyst_topos
        apparent = observer.at(t).observe(star).apparent()

        alt_sf_refracted, _, _ = apparent.altaz(
            temperature_C=atmo.temperature - 273.15,
            pressure_mbar=atmo.pressure,
        )
        alt_sf_vacuum, _, _ = apparent.altaz(
            temperature_C=0,
            pressure_mbar=0,
        )
        delta_sf = alt_sf_refracted.degrees - alt_sf_vacuum.degrees

        # Both deltas should be positive (refraction bends light upward)
        # and in a physically reasonable range for ~50 deg el, ~500 hPa
        assert 0.001 < delta_ccat < 0.05, f"ccat refraction delta out of range: {delta_ccat:.6f}"
        assert 0.001 < delta_sf < 0.05, f"Skyfield refraction delta out of range: {delta_sf:.6f}"

        delta_diff = abs(delta_ccat - delta_sf)
        assert delta_diff < self.REFRACTION_DELTA_TOLERANCE, (
            f"Refraction delta mismatch: ccat={delta_ccat:.6f}, "
            f"skyfield={delta_sf:.6f}, diff={delta_diff:.6f} deg"
        )
