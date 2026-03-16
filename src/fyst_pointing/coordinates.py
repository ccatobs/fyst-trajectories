"""Coordinate transformations for telescope pointing.

This module provides coordinate transformation utilities for converting
between celestial coordinates (RA/Dec) and horizontal coordinates (Az/El),
with support for atmospheric refraction corrections and solar system
ephemeris calculations.

The transformations use astropy's coordinate transformation framework
with IERS data for Earth orientation parameters.

Examples
--------
Basic coordinate transformation:

>>> from astropy.time import Time
>>> from fyst_pointing.coordinates import Coordinates
>>> from fyst_pointing.site import get_fyst_site
>>> coords = Coordinates(get_fyst_site())
>>> obstime = Time("2026-03-15T04:00:00", scale="utc")
>>> az, el = coords.radec_to_altaz(83.633, 22.014, obstime=obstime)  # Crab Nebula
>>> print(f"Az: {az:.2f}°, El: {el:.2f}°")

Get planet position:

>>> obstime = Time("2026-03-15T16:00:00", scale="utc")
>>> az, el = coords.get_body_altaz("mars", obstime)
>>> print(f"Mars at Az: {az:.2f}°, El: {el:.2f}°")
"""

import warnings
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord, get_body, get_sun
from astropy.time import Time, TimeDelta

from .site import AtmosphericConditions, Site

try:
    import erfa

    _erfa_warning_cls = erfa.ErfaWarning
except (ImportError, AttributeError):
    # ImportError: erfa not installed (should not happen with astropy, but
    # defensive).  AttributeError: older erfa versions that lack ErfaWarning.
    _erfa_warning_cls = UserWarning

# Supported solar system bodies for ephemeris
SOLAR_SYSTEM_BODIES = [
    "sun",
    "moon",
    "mercury",
    "venus",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
]

# Frame name aliases for KOSMA/OCS compatibility
# Maps common telescope control system names to astropy frame names.
FRAME_ALIASES: MappingProxyType[str, str] = MappingProxyType(
    {
        "J2000": "icrs",
        "FK5": "fk5",
        "B1950": "fk4",
        "GALACTIC": "galactic",
        "ECLIPTIC": "geocentrictrueecliptic",
        "HORIZON": "altaz",
    }
)


def normalize_frame(frame: str) -> str:
    """Convert KOSMA/OCS frame names to astropy equivalents.

    Handles common frame name aliases used in telescope control systems,
    converting them to the corresponding astropy coordinate frame names.
    Unknown frame names are lowercased for astropy compatibility.

    Parameters
    ----------
    frame : str
        Frame name, either a KOSMA/OCS alias or an astropy frame name.

    Returns
    -------
    str
        The astropy-compatible frame name (always lowercase).

    Examples
    --------
    >>> normalize_frame("J2000")
    'icrs'
    >>> normalize_frame("GALACTIC")
    'galactic'
    >>> normalize_frame("icrs")
    'icrs'
    >>> normalize_frame("ICRS")
    'icrs'
    """
    return FRAME_ALIASES.get(frame.upper(), frame.lower())


@dataclass(frozen=True)
class AltAzCoord:
    """Horizontal coordinate (Altitude-Azimuth).

    Parameters
    ----------
    az : float
        Azimuth in degrees (N=0, E=90).
    alt : float
        Altitude (elevation) in degrees above the horizon.
    obstime : Time, optional
        Observation time.
    """

    az: float
    alt: float
    obstime: Time | None = None

    @property
    def el(self) -> float:
        """Alias for altitude (elevation)."""
        return self.alt

    def __repr__(self) -> str:
        """Return string representation of the coordinate."""
        return f"AltAzCoord(az={self.az:.4f}°, alt={self.alt:.4f}°)"


class Coordinates:
    """Coordinate transformation engine for a telescope site.

    This class provides methods for converting between celestial and
    horizontal coordinate systems, accounting for atmospheric refraction
    and providing solar system ephemeris calculations.

    Parameters
    ----------
    site : Site
        Telescope site configuration containing location.
    atmosphere : AtmosphericConditions or None, optional
        Atmospheric conditions for refraction correction. If not provided,
        refraction is disabled (pressure=0). Construct an
        ``AtmosphericConditions`` instance to enable refraction.

    Examples
    --------
    >>> from fyst_pointing.coordinates import Coordinates
    >>> from fyst_pointing.site import get_fyst_site
    >>> coords = Coordinates(get_fyst_site())

    Transform a single position:

    >>> from astropy.time import Time
    >>> t = Time("2026-03-15T04:00:00", scale="utc")
    >>> az, el = coords.radec_to_altaz(180.0, -45.0, obstime=t)
    """

    def __init__(
        self,
        site: Site,
        atmosphere: AtmosphericConditions | None = None,
    ):
        self.site = site
        self.location = site.location
        if atmosphere is not None:
            self.atmosphere = atmosphere
        else:
            self.atmosphere = AtmosphericConditions.no_refraction()

    def _get_altaz_frame(self, obstime: Time) -> AltAz:
        """Get the AltAz frame for the site at a given time.

        Parameters
        ----------
        obstime : Time
            Observation time.

        Returns
        -------
        AltAz
            Astropy AltAz frame configured for the site. When the
            atmosphere has ``obswl > 100 µm``, astropy automatically
            uses the radio refraction model instead of optical.
        """
        kwargs = {
            "obstime": obstime,
            "location": self.location,
            "pressure": self.atmosphere.pressure_hpa,
            "temperature": self.atmosphere.temperature_degc,
            "relative_humidity": self.atmosphere.relative_humidity,
        }
        obswl = self.atmosphere.obswl_quantity
        if obswl is not None:
            kwargs["obswl"] = obswl
        return AltAz(**kwargs)

    def radec_to_altaz(
        self,
        ra: float | np.ndarray,
        dec: float | np.ndarray,
        obstime: Time,
        frame: str = "icrs",
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Convert RA/Dec to Az/El.

        Transforms celestial coordinates to horizontal coordinates,
        accounting for atmospheric refraction.

        Parameters
        ----------
        ra : float or array
            Right Ascension in degrees.
        dec : float or array
            Declination in degrees.
        obstime : Time
            Observation time.
        frame : str, optional
            Celestial reference frame. Default is "icrs" (J2000).

        Returns
        -------
        az : float or array
            Azimuth in degrees (N=0, E=90).
        alt : float or array
            Altitude (elevation) in degrees above the horizon.

        Examples
        --------
        >>> from astropy.time import Time
        >>> coords = Coordinates(site)
        >>> obstime = Time("2026-03-15T04:00:00", scale="utc")
        >>> az, el = coords.radec_to_altaz(83.633, 22.014, obstime)
        """
        sky_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame=frame)

        altaz_frame = self._get_altaz_frame(obstime)
        altaz = sky_coord.transform_to(altaz_frame)

        az = altaz.az.deg
        alt = altaz.alt.deg

        if np.isscalar(ra) and np.isscalar(dec) and obstime.isscalar:
            return float(az), float(alt)
        return az, alt

    def altaz_to_radec(
        self,
        az: float | np.ndarray,
        alt: float | np.ndarray,
        obstime: Time,
        frame: str = "icrs",
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Convert Az/El to RA/Dec.

        Transforms horizontal coordinates to celestial coordinates.

        Parameters
        ----------
        az : float or array
            Azimuth in degrees, measured from North through East.
        alt : float or array
            Altitude (elevation) in degrees above the horizon.
        obstime : Time
            Observation time.
        frame : str, optional
            Output celestial reference frame. Default is "icrs" (J2000).

        Returns
        -------
        ra : float or array
            Right Ascension in degrees.
        dec : float or array
            Declination in degrees.
        """
        altaz_frame = self._get_altaz_frame(obstime)
        altaz = SkyCoord(az=az * u.deg, alt=alt * u.deg, frame=altaz_frame)

        sky_coord = altaz.transform_to(frame)

        ra = sky_coord.ra.deg
        dec = sky_coord.dec.deg

        if np.isscalar(az) and np.isscalar(alt):
            return float(ra), float(dec)
        return ra, dec

    def get_body_altaz(
        self,
        body: str,
        obstime: Time,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Get the Az/El position of a solar system body.

        Parameters
        ----------
        body : str
            Name of the solar system body. Supported values:
            sun, moon, mercury, venus, mars, jupiter, saturn, uranus, neptune.
        obstime : Time
            Observation time. Can be a scalar Time or an array of Times.

        Returns
        -------
        az : float or array
            Azimuth in degrees.
        alt : float or array
            Altitude (elevation) in degrees.

        Raises
        ------
        ValueError
            If the body name is not recognized.

        Examples
        --------
        >>> from astropy.time import Time
        >>> obstime = Time("2026-03-15T16:00:00", scale="utc")
        >>> az, el = coords.get_body_altaz("mars", obstime)
        """
        body = body.lower()
        if body not in SOLAR_SYSTEM_BODIES:
            raise ValueError(f"Unknown body '{body}'. Supported bodies: {SOLAR_SYSTEM_BODIES}")

        if body == "sun":
            body_coord = get_sun(obstime)
        else:
            body_coord = get_body(body, obstime, location=self.location)

        altaz_frame = self._get_altaz_frame(obstime)
        altaz = body_coord.transform_to(altaz_frame)

        az = altaz.az.deg
        alt = altaz.alt.deg

        if obstime.isscalar:
            return float(az), float(alt)
        return az, alt

    def get_body_radec(
        self,
        body: str,
        obstime: Time,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Get the RA/Dec position of a solar system body.

        Parameters
        ----------
        body : str
            Name of the solar system body. Supported values:
            sun, moon, mercury, venus, mars, jupiter, saturn, uranus, neptune.
        obstime : Time
            Observation time. Can be a scalar Time or an array of Times.

        Returns
        -------
        ra : float or array
            Right Ascension in degrees.
        dec : float or array
            Declination in degrees.

        Raises
        ------
        ValueError
            If the body name is not recognized.

        Examples
        --------
        >>> from astropy.time import Time
        >>> obstime = Time("2026-03-15T00:00:00", scale="utc")
        >>> ra, dec = coords.get_body_radec("jupiter", obstime)
        """
        body = body.lower()
        if body not in SOLAR_SYSTEM_BODIES:
            raise ValueError(f"Unknown body '{body}'. Supported bodies: {SOLAR_SYSTEM_BODIES}")

        if body == "sun":
            body_coord = get_sun(obstime)
        else:
            body_coord = get_body(body, obstime, location=self.location)

        icrs = body_coord.icrs
        ra = icrs.ra.deg
        dec = icrs.dec.deg

        if obstime.isscalar:
            return float(ra), float(dec)
        return ra, dec

    def get_sun_altaz(self, obstime: Time) -> tuple[float, float]:
        """Get the Az/El position of the Sun.

        Convenience method for sun avoidance calculations.

        Parameters
        ----------
        obstime : Time
            Observation time.

        Returns
        -------
        az : float
            Sun azimuth in degrees.
        alt : float
            Sun altitude (elevation) in degrees.
        """
        return self.get_body_altaz("sun", obstime)

    def angular_separation(
        self,
        az1: float,
        alt1: float,
        az2: float,
        alt2: float,
    ) -> float:
        """Calculate angular separation between two Az/El positions.

        Parameters
        ----------
        az1, alt1 : float
            First position (azimuth, altitude) in degrees.
        az2, alt2 : float
            Second position (azimuth, altitude) in degrees.

        Returns
        -------
        float
            Angular separation in degrees.
        """
        c1 = SkyCoord(az=az1 * u.deg, alt=alt1 * u.deg, frame="altaz")
        c2 = SkyCoord(az=az2 * u.deg, alt=alt2 * u.deg, frame="altaz")
        return c1.separation(c2).deg

    def is_sun_safe(
        self,
        az: float,
        el: float,
        obstime: Time,
    ) -> bool:
        """Check if a position is safe from Sun exposure.

        Parameters
        ----------
        az : float
            Azimuth in degrees.
        el : float
            Elevation in degrees.
        obstime : Time
            Observation time.

        Returns
        -------
        bool
            True if position is outside the Sun exclusion radius.
        """
        if not self.site.sun_avoidance.enabled:
            return True

        sun_az, sun_alt = self.get_sun_altaz(obstime)
        separation = self.angular_separation(az, el, sun_az, sun_alt)

        return separation > self.site.sun_avoidance.exclusion_radius

    def is_position_observable(
        self,
        az: float,
        el: float,
        obstime: Time,
        check_sun: bool = True,
    ) -> tuple[bool, str]:
        """Check if a position is observable.

        Checks telescope limits and optionally sun avoidance.

        Parameters
        ----------
        az : float
            Azimuth in degrees.
        el : float
            Elevation in degrees.
        obstime : Time
            Observation time for sun check.
        check_sun : bool, optional
            Whether to check sun avoidance. Default True.

        Returns
        -------
        observable : bool
            True if position is observable.
        reason : str
            Empty string if observable, otherwise reason for rejection.
        """
        limits = self.site.telescope_limits

        if not limits.elevation.is_in_range(el):
            return (
                False,
                f"Elevation {el:.1f}° outside limits "
                f"[{limits.elevation.min}, {limits.elevation.max}]",
            )

        if not limits.azimuth.is_in_range(az):
            return (
                False,
                f"Azimuth {az:.1f}° outside limits [{limits.azimuth.min}, {limits.azimuth.max}]",
            )

        if check_sun and self.site.sun_avoidance.enabled:
            sun_az, sun_alt = self.get_sun_altaz(obstime)
            sep = self.angular_separation(az, el, sun_az, sun_alt)
            if sep <= self.site.sun_avoidance.exclusion_radius:
                return False, f"Position too close to Sun (separation: {sep:.1f}°)"

        return True, ""

    def get_rise_set_times(
        self,
        ra: float,
        dec: float,
        start_time: Time,
        horizon: float,
        max_search_hours: float,
        step_hours: float,
    ) -> tuple[Time | None, Time | None]:
        """Calculate rise and set times for a celestial target.

        Finds when a source at the given RA/Dec rises above and sets below
        the specified horizon altitude.

        Parameters
        ----------
        ra : float
            Right Ascension of the target in degrees.
        dec : float
            Declination of the target in degrees.
        start_time : Time
            Start time for the search.
        horizon : float
            Horizon altitude in degrees. Use 0.0 for geometric horizon
            or positive values (e.g., 20.0) for telescope elevation limits.
        max_search_hours : float
            Maximum time to search forward in hours.
        step_hours : float
            Time step for the search in hours.
            Smaller values give more precision but take longer.

        Returns
        -------
        rise_time : Time or None
            Time when the source next rises above the horizon.
            None if the source is circumpolar (always above) or never rises
            within the search window.
        set_time : Time or None
            Time when the source next sets below the horizon after rising.
            None if the source is circumpolar or never sets within the
            search window.

        Notes
        -----
        Returns (None, None) for circumpolar or never-visible sources.
        Finds the FIRST rise, then the FIRST set after that rise.

        Refraction is disabled (pressure=0). Calculated times
        may differ from observed rise/set by a few minutes (~0.5 deg
        refraction near horizon). Use a lower horizon value to compensate.

        Between coarse grid points, altitude may have local extrema that
        the grid misses, especially for sources with grazing passes near
        the horizon. Use a smaller step_hours for such cases.

        The crossing time is estimated via linear interpolation between
        adjacent grid points. Newton refinement is not used because each
        iteration would require a full astropy coordinate transform. The
        precision gain (~seconds) is not worth the cost for planning
        purposes. For higher precision, use a finer step_hours.

        Examples
        --------
        >>> from astropy.time import Time
        >>> coords = Coordinates(site)
        >>> start = Time("2026-03-15T00:00:00", scale="utc")
        >>> # Find when Orion rises and sets
        >>> rise, set_ = coords.get_rise_set_times(
        ...     83.633,
        ...     22.014,
        ...     start_time=start,
        ...     horizon=0.0,
        ...     max_search_hours=24.0,
        ...     step_hours=0.1,
        ... )
        >>> if rise is not None:
        ...     print(f"Rises at: {rise.iso}")
        ...     print(f"Sets at: {set_.iso}")
        ... else:
        ...     print("Source is circumpolar or never visible")

        Using telescope elevation limit as horizon:

        >>> rise, set_ = coords.get_rise_set_times(
        ...     ra=180.0,
        ...     dec=-30.0,
        ...     start_time=start,
        ...     horizon=20.0,  # Telescope minimum elevation
        ...     max_search_hours=24.0,
        ...     step_hours=0.1,
        ... )
        """
        source = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

        n_steps = int(max_search_hours / step_hours) + 1
        times = start_time + np.arange(n_steps) * TimeDelta(step_hours * u.hour)

        altaz_frame = AltAz(
            obstime=times,
            location=self.location,
            pressure=0 * u.hPa,
        )
        altitudes = source.transform_to(altaz_frame).alt.to_value(u.deg)

        rise_time = None
        set_time = None

        rise_indices = np.where((altitudes[:-1] < horizon) & (altitudes[1:] >= horizon))[0]

        if len(rise_indices) > 0:
            i_rise = rise_indices[0]
            denom = altitudes[i_rise + 1] - altitudes[i_rise]
            frac = 0.0 if abs(denom) < 1e-12 else (horizon - altitudes[i_rise]) / denom
            rise_time = times[i_rise] + frac * (times[i_rise + 1] - times[i_rise])

            set_indices = np.where((altitudes[:-1] >= horizon) & (altitudes[1:] < horizon))[0]

            after_rise = set_indices[set_indices >= i_rise]

            if len(after_rise) > 0:
                i_set = after_rise[0]
                denom = altitudes[i_set + 1] - altitudes[i_set]
                frac = 0.0 if abs(denom) < 1e-12 else (horizon - altitudes[i_set]) / denom
                set_time = times[i_set] + frac * (times[i_set + 1] - times[i_set])

        return rise_time, set_time

    def get_lst(self, obstime: Time) -> float | np.ndarray:
        """Get Local Sidereal Time at the site.

        Parameters
        ----------
        obstime : Time
            Observation time. Can be a scalar Time or an array of Times.

        Returns
        -------
        float or array
            Local Sidereal Time in degrees (0 to 360).

        Examples
        --------
        >>> from astropy.time import Time
        >>> coords = Coordinates(site)
        >>> lst = coords.get_lst(Time("2026-03-15T04:00:00", scale="utc"))
        """
        lst = obstime.sidereal_time("apparent", longitude=self.location.lon)
        lst_deg = lst.to_value(u.deg)

        if obstime.isscalar:
            return float(lst_deg)
        return lst_deg

    def get_hour_angle(
        self,
        ra: float | np.ndarray,
        obstime: Time,
    ) -> float | np.ndarray:
        """Calculate hour angle (HA = LST - RA).

        Parameters
        ----------
        ra : float or array
            Right Ascension in degrees.
        obstime : Time
            Observation time.

        Returns
        -------
        float or array
            Hour angle in degrees, normalized to -180 to 180.
            Positive values indicate the object is west of the meridian.

        Examples
        --------
        >>> from astropy.time import Time
        >>> coords = Coordinates(site)
        >>> ha = coords.get_hour_angle(83.633, Time("2026-03-15T04:00:00", scale="utc"))
        """
        lst = self.get_lst(obstime)
        ha = lst - ra

        ha = np.mod(ha + 180, 360) - 180

        if np.isscalar(ra) and obstime.isscalar:
            return float(ha)
        return ha

    def get_parallactic_angle(
        self,
        ra: float | np.ndarray,
        dec: float | np.ndarray,
        obstime: Time,
    ) -> float | np.ndarray:
        """Calculate the parallactic angle for a celestial position.

        Parameters
        ----------
        ra : float or array
            Right Ascension in degrees.
        dec : float or array
            Declination in degrees.
        obstime : Time
            Observation time.

        Returns
        -------
        float or array
            Parallactic angle in degrees.

        Notes
        -----
        The parallactic angle q is computed using:

        tan(q) = sin(H) / (cos(dec) * tan(lat) - sin(dec) * cos(H))

        where H is the hour angle and lat is the observatory latitude.
        Positive parallactic angle is measured from North through East,
        following the IAU convention.

        For objects at the zenith or very close to it, the parallactic
        angle is undefined (returns 0).

        Examples
        --------
        >>> from astropy.time import Time
        >>> coords = Coordinates(site)
        >>> obstime = Time("2026-03-15T04:00:00", scale="utc")
        >>> pa = coords.get_parallactic_angle(83.633, 22.014, obstime=obstime)
        >>> print(f"Parallactic angle: {pa:.2f}°")
        """
        ha = self.get_hour_angle(ra, obstime)

        ha_rad = np.deg2rad(ha)
        dec_rad = np.deg2rad(dec)
        lat_rad = np.deg2rad(self.site.latitude)

        sin_ha = np.sin(ha_rad)
        cos_ha = np.cos(ha_rad)
        sin_dec = np.sin(dec_rad)
        cos_dec = np.cos(dec_rad)
        tan_lat = np.tan(lat_rad)

        numerator = sin_ha
        denominator = cos_dec * tan_lat - sin_dec * cos_ha

        pa_rad = np.arctan2(numerator, denominator)
        pa_deg = np.rad2deg(pa_rad)

        if np.isscalar(ra) and np.isscalar(dec):
            return float(pa_deg)
        return pa_deg

    def get_field_rotation(
        self,
        ra: float | np.ndarray,
        dec: float | np.ndarray,
        obstime: Time,
    ) -> float | np.ndarray:
        """Calculate sky field rotation (nasmyth_sign * elevation + parallactic angle).

        Computes ``site.nasmyth_sign * elevation + parallactic_angle``
        with no instrument rotation. This is the sky rotation component
        only, using the Nasmyth port sign from the site configuration.

        For the full focal-plane rotation that also includes instrument
        rotation, use ``fyst_pointing.offsets.compute_focal_plane_rotation``
        instead.

        Parameters
        ----------
        ra : float or array
            Right Ascension in degrees.
        dec : float or array
            Declination in degrees.
        obstime : Time
            Observation time.

        Returns
        -------
        float or array
            Field rotation in degrees (nasmyth_sign * elevation + parallactic angle).

        Notes
        -----
        The field rotation rate is highest when the object transits
        near the zenith and lowest near the horizon.

        The Nasmyth sign is +1 for Right Nasmyth, -1 for Left Nasmyth,
        and 0 for Cassegrain (no elevation-dependent rotation).

        See Also
        --------
        fyst_pointing.offsets.compute_focal_plane_rotation :
            Full focal-plane rotation including Nasmyth sign and
            instrument rotation.

        Examples
        --------
        >>> from astropy.time import Time
        >>> coords = Coordinates(site)
        >>> fr = coords.get_field_rotation(83.633, 22.014, Time("2026-03-15T04:00:00", scale="utc"))
        """
        _, el = self.radec_to_altaz(ra, dec, obstime)

        pa = self.get_parallactic_angle(ra, dec, obstime)

        field_rotation = self.site.nasmyth_sign * el + pa

        if np.isscalar(ra) and np.isscalar(dec):
            return float(field_rotation)
        return field_rotation

    def radec_to_altaz_with_pm(
        self,
        ra: float,
        dec: float,
        pm_ra: float,
        pm_dec: float,
        ref_epoch: Time,
        obstime: Time,
        distance: float | None = None,
        radial_velocity: float | None = None,
        frame: str = "icrs",
    ) -> tuple[float, float]:
        """Convert RA/Dec to Az/El with proper motion correction.

        Propagates the position from the reference epoch to the observation
        time using the provided proper motion values before converting to
        horizontal coordinates.

        Parameters
        ----------
        ra : float
            Right Ascension in degrees at reference epoch.
        dec : float
            Declination in degrees at reference epoch.
        pm_ra : float
            Proper motion in RA (including cos(dec) factor) in mas/yr.
            This follows the Gaia convention (pmra = mu_ra * cos(dec)).
        pm_dec : float
            Proper motion in Dec in mas/yr.
        ref_epoch : Time
            Reference epoch for the catalog coordinates (e.g., J2000.0 or
            the Gaia observation epoch).
        obstime : Time
            Observation time to compute position for.
        distance : float, optional
            Distance in parsecs. If provided along with radial_velocity,
            enables full 3D space motion propagation. If None, only 2D proper
            motion on the sky is used.
        radial_velocity : float, optional
            Radial velocity in km/s (positive = receding). Used for full 3D
            space motion propagation when distance is also provided.
        frame : str, optional
            Input coordinate frame. Default is "icrs".

        Returns
        -------
        az : float
            Azimuth in degrees at observation time.
        el : float
            Elevation in degrees at observation time.

        Notes
        -----
        When distance is provided, the full space motion is computed using
        astropy's apply_space_motion() method. Without distance, an approximate
        2D propagation on the celestial sphere is used.

        Examples
        --------
        Track Barnard's Star (high proper motion):

        >>> from astropy.time import Time
        >>> # Barnard's Star coordinates at Gaia DR2 epoch
        >>> ra, dec = 269.452, 4.693  # degrees
        >>> pmra, pmdec = -798.58, 10328.12  # mas/yr
        >>> ref_epoch = Time("J2015.5")
        >>> obs_time = Time("2026-06-15T04:00:00")
        >>> az, el = coords.radec_to_altaz_with_pm(
        ...     ra, dec, pmra, pmdec, ref_epoch, obstime=obs_time, distance=1.8
        ... )
        """
        coord_kwargs = {
            "ra": ra * u.deg,
            "dec": dec * u.deg,
            "pm_ra_cosdec": pm_ra * u.mas / u.yr,
            "pm_dec": pm_dec * u.mas / u.yr,
            "frame": frame,
            "obstime": ref_epoch,
        }

        if distance is not None:
            coord_kwargs["distance"] = distance * u.pc
        if radial_velocity is not None:
            coord_kwargs["radial_velocity"] = radial_velocity * u.km / u.s

        coord = SkyCoord(**coord_kwargs)

        if distance is not None:
            coord_at_obs = coord.apply_space_motion(new_obstime=obstime)
        else:
            # Without a real distance, use a large dummy distance to leverage
            # astropy's spherical proper motion propagation. This avoids edge
            # cases near the poles that arise from the manual linear approach
            # (dividing by cos(dec) near dec=+/-90).
            # Suppress the ERFA "distance overridden" warning that may be
            # triggered by the unrealistically large dummy distance (1e6 pc).
            dummy_coord = SkyCoord(
                ra=ra * u.deg,
                dec=dec * u.deg,
                pm_ra_cosdec=pm_ra * u.mas / u.yr,
                pm_dec=pm_dec * u.mas / u.yr,
                distance=1e6 * u.pc,
                frame=frame,
                obstime=ref_epoch,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*distance overridden.*",
                    category=_erfa_warning_cls,
                )
                coord_at_obs = dummy_coord.apply_space_motion(new_obstime=obstime)

        return self.radec_to_altaz(
            float(coord_at_obs.ra.deg),
            float(coord_at_obs.dec.deg),
            obstime=obstime,
        )
