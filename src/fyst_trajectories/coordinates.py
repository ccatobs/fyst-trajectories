"""Coordinate transformations for telescope pointing.

This module provides coordinate transformation utilities for converting
between celestial coordinates (RA/Dec) and horizontal coordinates (Az/El),
with support for atmospheric refraction corrections and solar system
ephemeris calculations.

The transformations use astropy's coordinate transformation framework
with IERS data for Earth orientation parameters.

``Coordinates(site)`` defaults to vacuum (geometric) coordinates:
refraction is applied downstream at execution time, by exactly one of
the Go TCS or the ACU (ICD P-INCM-ICD-0003-A section 6), so vacuum
output is correct either way. For planning and
simulation (visibility calculations, observability checks, hitmap
simulations) where the output is NOT sent to the telescope, pass
``AtmosphericConditions.for_fyst()`` to apply submillimetre refraction.

Examples
--------
Trajectory generation (vacuum; refraction is applied downstream):

>>> from astropy.time import Time
>>> from fyst_trajectories.coordinates import Coordinates
>>> from fyst_trajectories.site import get_fyst_site
>>> coords = Coordinates(get_fyst_site())
>>> obstime = Time("2026-01-15T02:00:00", scale="utc")
>>> az, el = coords.radec_to_altaz(83.633, 22.014, obstime=obstime)  # Crab Nebula
>>> print(f"Az: {az:.2f} deg, El: {el:.2f} deg")
Az: 9.40 deg, El: 44.44 deg

Planning with refraction (visibility checks, not sent to ACU):

>>> from fyst_trajectories.site import AtmosphericConditions
>>> coords_plan = Coordinates(get_fyst_site(), atmosphere=AtmosphericConditions.for_fyst())
>>> az, el = coords_plan.radec_to_altaz(83.633, 22.014, obstime=obstime)
"""

import importlib.util
import os
import warnings
from types import MappingProxyType

import erfa
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord, get_body
from astropy.time import Time, TimeDelta

from .site import AtmosphericConditions, Site

# Supported solar system bodies for ephemeris
SOLAR_SYSTEM_BODIES = (
    "sun",
    "moon",
    "mercury",
    "venus",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
)
"""Solar-system bodies resolvable through astropy's built-in ephemeris.

Accepted by the body-tracking coordinate methods (for example
``Coordinates.get_body_altaz``); these require no external kernel. Planetary
satellites are addressed separately, see ``SATELLITE_BODIES``.
"""


# Known planetary-satellite NAIF kernel chains (SSB -> ... -> satellite). astropy's
# get_body has no name for a moon, so it is addressed by integer NAIF-ID chain,
# evaluated against a JPL *satellite* SPK kernel (not the builtin ephemeris).
# Extensible (e.g. the Galilean moons: "io": ((0, 5), (5, 501))).
_SATELLITE_NAIF_CHAINS: dict[str, tuple[tuple[int, int], ...]] = {
    "titan": ((0, 6), (6, 606)),  # SSB -> Saturn-system barycentre -> Titan
}

# Public names of the planetary satellites resolvable via a JPL satellite SPK
# kernel (parallel to ``SOLAR_SYSTEM_BODIES``). Unlike the builtin bodies these
# require a kernel (``satellite_kernel`` / ``FYST_SATELLITE_KERNEL``).
SATELLITE_BODIES = tuple(_SATELLITE_NAIF_CHAINS)
"""Public names of the planetary satellites resolvable via a JPL satellite SPK kernel.

Parallel to ``SOLAR_SYSTEM_BODIES`` but, unlike the built-in bodies, each name
requires a satellite kernel supplied through ``Coordinates(satellite_kernel=...)``
or the ``FYST_SATELLITE_KERNEL`` environment variable.
"""

# Environment variable holding the path to a JPL satellite SPK kernel. Read
# lazily, only when a satellite body is requested (never at import).
_SATELLITE_KERNEL_ENV = "FYST_SATELLITE_KERNEL"


def _resolve_satellite_kernel(explicit: str | None) -> str:
    """Resolve a JPL satellite SPK kernel to an absolute file path.

    Prefers ``explicit`` (the ``Coordinates(satellite_kernel=...)`` value), else
    the ``FYST_SATELLITE_KERNEL`` environment variable. The result is made
    **absolute** on purpose: astropy's ephemeris loader special-cases a ``de###``
    prefix (a regex tested before the on-disk check) and resolves relative paths
    against the process cwd, so a relative or ``de``-prefixed path would be
    silently mishandled.

    Parameters
    ----------
    explicit : str or None
        An explicit kernel path, or None to fall back to the environment.

    Returns
    -------
    str
        Absolute path to the kernel.

    Raises
    ------
    ValueError
        If no kernel is configured (message includes actionable guidance).
    FileNotFoundError
        If the configured path does not exist.
    ModuleNotFoundError
        If the optional ``jplephem`` dependency is not installed.
    """
    path = explicit or os.environ.get(_SATELLITE_KERNEL_ENV)
    if not path:
        raise ValueError(
            "Tracking a planetary satellite requires a JPL satellite SPK kernel, "
            f"which is not configured. Set the {_SATELLITE_KERNEL_ENV} environment "
            "variable (or pass satellite_kernel=... to Coordinates) to a .bsp path. "
            "Install the optional dependency with `pip install "
            "'fyst-trajectories[ephemeris]'` and build a small kernel with: "
            "`python -m jplephem excerpt --targets 3,399,10,6,606 <start> <end> "
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/sat441.bsp "
            "titan.bsp`."
        )
    abspath = os.path.abspath(path)
    if not os.path.isfile(abspath):
        raise FileNotFoundError(f"Satellite SPK kernel not found: {abspath}")
    # Loading a non-builtin SPK needs jplephem (astropy does not pull it in by
    # default). find_spec does not import the module, so this stays import-safe.
    if importlib.util.find_spec("jplephem") is None:
        raise ModuleNotFoundError(
            "Loading a satellite SPK kernel requires the optional 'jplephem' "
            "dependency. Install it with `pip install 'fyst-trajectories[ephemeris]'`."
        )
    return abspath


# Frame name aliases mapping telescope control-system names to astropy frame names.
# Note: ``"J2000"`` maps to ICRS, not FK5 J2000.0. The two frames differ at the
# tens-of-milliarcsecond level: the FK5 equinox sits -22.9 +/- 2.3 mas from the
# ICRS right-ascension origin, and the FK5 pole agrees with the ICRS pole only
# to within FK5's own +/-50 mas uncertainty (IERS Conventions 2010, TN36
# chapter 2). The ICRS was adopted by IAU 1997 Resolution B2,
# effective 1998 January 1. For sub-arcsecond catalogue work this matters; for
# telescope pointing it is well below the beam and is harmless.
#
# Only spherical RA/Dec frames are aliased. GALACTIC (``l``/``b``) and ECLIPTIC
# (``lon``/``lat``) are intentionally omitted: the transform methods read
# ``ra``/``dec`` attributes and would raise on them (see ``normalize_frame``).
FRAME_ALIASES: MappingProxyType[str, str] = MappingProxyType(
    {
        "J2000": "icrs",
        "FK5": "fk5",
        "B1950": "fk4",
        "HORIZON": "altaz",
    }
)
"""Frame-name aliases mapping telescope control-system names to astropy frame names.

Only spherical RA/Dec frames are aliased; ``"J2000"`` maps to ICRS (not FK5
J2000.0), which is harmless for telescope pointing. Consumed by ``normalize_frame``.
"""


def normalize_frame(frame: str) -> str:
    """Convert telescope control-system frame names to astropy equivalents.

    Handles common frame name aliases used in telescope control systems,
    converting them to the corresponding astropy coordinate frame names.
    Unknown frame names are lowercased for astropy compatibility.

    Parameters
    ----------
    frame : str
        Frame name, either a control-system alias or an astropy frame name.

    Returns
    -------
    str
        The astropy-compatible frame name (always lowercase).

    Notes
    -----
    Only spherical RA/Dec frames (``icrs``/``J2000``, ``fk5``/``FK5``,
    ``fk4``/``B1950``) are usable with :meth:`Coordinates.radec_to_altaz` and
    :meth:`Coordinates.altaz_to_radec`, which read ``ra``/``dec`` attributes.
    ``HORIZON`` is aliased so a control system's own spelling of the
    horizontal frame resolves, but those two methods refuse it by name for
    the same reason: azimuth and elevation are separate arguments there.
    ``GALACTIC`` and ``ECLIPTIC`` are deliberately not aliased: those frames
    use ``l``/``b`` and ``lon``/``lat`` and would raise in the transform
    methods. (An unknown name is still lowercased for astropy, so a caller can
    pass an astropy frame name directly at their own risk.)

    Examples
    --------
    >>> normalize_frame("J2000")
    'icrs'
    >>> normalize_frame("FK5")
    'fk5'
    >>> normalize_frame("icrs")
    'icrs'
    >>> normalize_frame("ICRS")
    'icrs'
    """
    return FRAME_ALIASES.get(frame.upper(), frame.lower())


def _radec_frame(frame: str) -> str:
    """Resolve a caller's frame name to an astropy RA/Dec frame name.

    Wraps :func:`normalize_frame` for the entry points that read
    ``ra``/``dec``, so a horizontal-frame name is refused by name rather
    than dying inside astropy on a missing attribute.

    Raises
    ------
    ValueError
        If the name resolves to the horizontal frame.
    """
    resolved = normalize_frame(frame)
    if resolved == "altaz":
        raise ValueError(
            f"frame={frame!r} names the horizontal frame, which has no RA/Dec to "
            f"read. Pass an RA/Dec frame name ('J2000', 'FK5', 'B1950' or an "
            f"astropy spelling); azimuth and elevation are separate arguments."
        )
    return resolved


def _parallactic_angle_from_altaz(
    az_rad: float | np.ndarray,
    el_rad: float | np.ndarray,
    latitude_deg: float,
) -> float | np.ndarray:
    """Compute the parallactic angle from a vacuum horizon position.

    The package's single definition of the IAU AltAz parallactic-angle
    formula. Shared rather than duplicated because two entry points
    publish the same quantity from different inputs:
    :meth:`Coordinates.get_field_rotation` takes RA/Dec and
    :meth:`Coordinates.get_field_rotation_from_altaz` takes an
    already-transformed horizon position, and both add the mechanical
    Nasmyth term to it.

    Parameters
    ----------
    az_rad : float or array
        Azimuth in radians.
    el_rad : float or array
        Elevation in radians.
    latitude_deg : float
        Site latitude in degrees.

    Returns
    -------
    float or array
        Parallactic angle in degrees, in the same shape as the inputs.
    """
    lat_rad = np.deg2rad(latitude_deg)

    sin_az = np.sin(az_rad)
    cos_az = np.cos(az_rad)
    sin_el = np.sin(el_rad)
    cos_el = np.cos(el_rad)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)

    numerator = -sin_az * cos_lat
    denominator = sin_lat * cos_el - cos_lat * sin_el * cos_az

    return np.rad2deg(np.arctan2(numerator, denominator))


def _build_time_grid(time: Time, horizon_hours: float, step_minutes: float) -> Time:
    """Build the sample grid. Length 1 (just ``time``) when ``horizon_hours <= 0``."""
    if horizon_hours and horizon_hours > 0:
        horizon_s = horizon_hours * 3600.0
        step_s = step_minutes * 60.0
        # Cover [time, time + horizon] with uniform step_s spacing. ceil so the
        # interval is fully covered; n >= 2 so a positive horizon is always a real
        # interval (never a degenerate length-1 grid). The final sample is clipped
        # to land exactly on time + horizon (no sample past the horizon), so the
        # last cell may be shorter than step_s. Endpoints therefore land on grid
        # samples spaced <= step_minutes apart.
        n = max(2, int(np.ceil(horizon_s / step_s)) + 1)
        offsets_s = np.minimum(np.arange(n) * step_s, horizon_s)
    else:
        offsets_s = np.zeros(1)
    return time + TimeDelta(offsets_s, format="sec")


def _threshold_crossings(
    values: np.ndarray, grid: Time, threshold: float, *, rising: bool
) -> list[Time]:
    """Linearly interpolated times where ``values`` crosses ``threshold``.

    ``rising=True`` finds upward crossings (``values[i] < threshold <=
    values[i+1]``); ``rising=False`` downward ones (``values[i] >= threshold >
    values[i+1]``). The two masks partition each grid cell, so one cell yields
    at most one crossing and a value exactly at the threshold is never counted
    twice. Interpolation uses the actual per-cell spacing, so a clipped final
    grid cell (see :func:`_build_time_grid`) is handled exactly.
    """
    below, above = values[:-1], values[1:]
    if rising:
        mask = (below < threshold) & (above >= threshold)
    else:
        mask = (below >= threshold) & (above < threshold)
    times: list[Time] = []
    for i in np.flatnonzero(mask):
        denom = values[i + 1] - values[i]
        frac = 0.0 if abs(denom) < 1e-12 else (threshold - values[i]) / denom
        times.append(grid[i] + frac * (grid[i + 1] - grid[i]))
    return times


class Coordinates:
    """Coordinate transformation engine for a telescope site.

    This class provides methods for converting between celestial and
    horizontal coordinate systems, with optional atmospheric refraction
    and solar system ephemeris calculations.

    The default (``atmosphere=None``) produces vacuum (geometric)
    coordinates. This is the correct default for trajectory generation:
    refraction is applied downstream at execution time, by exactly one
    of the Go TCS or the ACU (ICD P-INCM-ICD-0003-A section 6), so
    vacuum output is correct either way.
    Pass ``AtmosphericConditions.for_fyst()`` for planning and
    simulation where the output is not sent to the telescope.

    Parameters
    ----------
    site : Site
        Telescope site configuration containing location.
    atmosphere : AtmosphericConditions or None, optional
        Atmospheric conditions for refraction correction. If not
        provided, defaults to vacuum (pressure=0). Pass
        ``AtmosphericConditions.for_fyst()`` for planning/simulation,
        or ``AtmosphericConditions.no_refraction()`` as an explicit
        synonym for the vacuum default.
    satellite_kernel : str or None, optional
        Path to a JPL satellite SPK kernel (e.g. an excerpt of NAIF
        ``sat441``) used to resolve planetary-satellite bodies such as
        ``"titan"``. If ``None``, the ``FYST_SATELLITE_KERNEL`` environment
        variable is used. Only consulted when a satellite body is requested;
        builtin planets/Moon/Sun never need it.

    Examples
    --------
    Trajectory generation (vacuum; refraction is applied downstream):

    >>> from fyst_trajectories.coordinates import Coordinates
    >>> from fyst_trajectories.site import get_fyst_site
    >>> coords = Coordinates(get_fyst_site())

    Planning with refraction (not sent to ACU):

    >>> from fyst_trajectories.site import AtmosphericConditions
    >>> coords = Coordinates(get_fyst_site(), atmosphere=AtmosphericConditions.for_fyst())

    Transform a single position:

    >>> from astropy.time import Time
    >>> t = Time("2026-03-15T04:00:00", scale="utc")
    >>> az, el = coords.radec_to_altaz(180.0, -45.0, obstime=t)
    """

    def __init__(
        self,
        site: Site,
        atmosphere: AtmosphericConditions | None = None,
        *,
        satellite_kernel: str | None = None,
    ):
        self.site = site
        self.location = site.location
        if atmosphere is not None:
            self.atmosphere = atmosphere
        else:
            self.atmosphere = AtmosphericConditions.no_refraction()
        self._satellite_kernel = satellite_kernel

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
            atmosphere has ``obswl > 100`` (microns), astropy automatically
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
        applying atmospheric refraction only when this instance was
        constructed with an atmosphere (the default is vacuum).

        Parameters
        ----------
        ra : float or array
            Right Ascension in degrees.
        dec : float or array
            Declination in degrees.
        obstime : Time
            Observation time.
        frame : str, optional
            Celestial reference frame. Default is "icrs" (J2000). The name is
            passed through :func:`normalize_frame`, so the control-system
            spellings ``"J2000"``, ``"FK5"`` and ``"B1950"`` are accepted
            alongside the astropy names. ``"HORIZON"`` names the horizontal
            frame and is refused here, since this method reads ``ra``/``dec``.

        Returns
        -------
        az : float or array
            Azimuth in degrees (N=0, E=90).
        alt : float or array
            Altitude (elevation) in degrees above the horizon.

        Raises
        ------
        ValueError
            If ``frame`` names the horizontal frame (``"HORIZON"``).

        Examples
        --------
        >>> from astropy.time import Time
        >>> coords = Coordinates(site)
        >>> obstime = Time("2026-03-15T04:00:00", scale="utc")
        >>> az, el = coords.radec_to_altaz(83.633, 22.014, obstime)
        """
        sky_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame=_radec_frame(frame))

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
            Output celestial reference frame. Default is "icrs" (J2000). The
            name is passed through :func:`normalize_frame`, so the
            control-system spellings ``"J2000"``, ``"FK5"`` and ``"B1950"``
            are accepted alongside the astropy names. ``"HORIZON"`` names the
            horizontal frame and is refused here, since this method returns
            ``ra``/``dec``.

        Returns
        -------
        ra : float or array
            Right Ascension in degrees.
        dec : float or array
            Declination in degrees.

        Raises
        ------
        ValueError
            If ``frame`` names the horizontal frame (``"HORIZON"``).
        """
        altaz_frame = self._get_altaz_frame(obstime)
        altaz = SkyCoord(az=az * u.deg, alt=alt * u.deg, frame=altaz_frame)

        sky_coord = altaz.transform_to(_radec_frame(frame))

        ra = sky_coord.ra.deg
        dec = sky_coord.dec.deg

        if np.isscalar(az) and np.isscalar(alt) and obstime.isscalar:
            return float(ra), float(dec)
        return ra, dec

    def _resolve_body(self, body: str) -> tuple[str | list[tuple[int, int]], str | None]:
        """Map a body name to its ``get_body`` spec and ephemeris kwarg.

        Returns ``(name, None)`` for a builtin solar-system body (planets, Moon,
        Sun), or ``(NAIF integer chain, absolute kernel path)`` for a known
        satellite (resolved via ``satellite_kernel`` / ``FYST_SATELLITE_KERNEL``).

        Raises
        ------
        ValueError
            If ``body`` is neither a builtin body nor a known satellite.
        """
        if body in SOLAR_SYSTEM_BODIES:
            return body, None
        if body in _SATELLITE_NAIF_CHAINS:
            return (
                list(_SATELLITE_NAIF_CHAINS[body]),
                _resolve_satellite_kernel(self._satellite_kernel),
            )
        raise ValueError(
            f"Unknown body '{body}'. Supported bodies: {SOLAR_SYSTEM_BODIES}; "
            f"supported satellites (require a kernel): {sorted(_SATELLITE_NAIF_CHAINS)}"
        )

    def get_body_altaz(
        self,
        body: str,
        obstime: Time,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Get the Az/El position of a solar system body.

        Parameters
        ----------
        body : str
            Name of the body. Builtin values: sun, moon, mercury, venus, mars,
            jupiter, saturn, uranus, neptune. Known satellites (e.g. ``"titan"``)
            are also accepted when a satellite SPK kernel is configured (see the
            ``satellite_kernel`` argument / ``FYST_SATELLITE_KERNEL``).
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
            If the body name is not recognized, or a satellite is requested
            without a configured kernel.
        FileNotFoundError
            If a satellite is requested and the configured kernel path
            does not exist.
        ModuleNotFoundError
            If a satellite is requested and ``jplephem`` is not
            installed (the ``ephemeris`` extra).

        Examples
        --------
        >>> from astropy.time import Time
        >>> obstime = Time("2026-03-15T16:00:00", scale="utc")
        >>> az, el = coords.get_body_altaz("mars", obstime)
        """
        body = body.lower()
        body_spec, ephemeris = self._resolve_body(body)

        # Use get_body uniformly (not get_sun) so every body shares one
        # topocentric code path. The AltAz frame's own location is what makes
        # the apparent place site-topocentric: measured against the geocentric
        # direction that shift is 8.8 arcsec * cos(el) for the Sun and up to
        # ~1 deg for the Moon. Passing location= here moves only the
        # light-travel-time reference point to the site, which moves this AltAz
        # result by under 0.001 arcsec (Sun) and under 0.4 arcsec (Moon). The
        # visible get_sun()-vs-get_body() difference (~0.01 arcsec typical) is
        # an ephemeris/algorithm difference, not parallax.
        body_coord = get_body(body_spec, obstime, location=self.location, ephemeris=ephemeris)

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
            Name of the body. Builtin values: sun, moon, mercury, venus, mars,
            jupiter, saturn, uranus, neptune. Known satellites (e.g. ``"titan"``)
            are also accepted when a satellite SPK kernel is configured (see the
            ``satellite_kernel`` argument / ``FYST_SATELLITE_KERNEL``).
        obstime : Time
            Observation time. Can be a scalar Time or an array of Times.

        Returns
        -------
        ra : float or array
            Apparent topocentric Right Ascension in degrees (ICRS axes).
        dec : float or array
            Apparent topocentric Declination in degrees (ICRS axes).

        Raises
        ------
        ValueError
            If the body name is not recognized, or a satellite is requested
            without a configured kernel.
        FileNotFoundError
            If a satellite is requested and the configured kernel path
            does not exist.
        ModuleNotFoundError
            If a satellite is requested and ``jplephem`` is not
            installed (the ``ephemeris`` extra).

        Notes
        -----
        The returned RA/Dec is the *apparent* sky position seen from the site,
        consistent with :meth:`get_body_altaz` (it round-trips:
        ``radec_to_altaz(get_body_radec(body, t), t) == get_body_altaz(body, t)``
        to ~arcsec) and with :meth:`get_parallactic_angle`'s ``pressure=0``
        transform.

        Examples
        --------
        >>> from astropy.time import Time
        >>> obstime = Time("2026-03-15T00:00:00", scale="utc")
        >>> ra, dec = coords.get_body_radec("jupiter", obstime)
        """
        body = body.lower()
        body_spec, ephemeris = self._resolve_body(body)

        # get_body returns a GCRS position carrying the body's finite
        # (topocentric) distance. Taking ``.icrs`` reprojects that finite-distance
        # vector to the barycentric frame, yielding the SSB->body direction
        # (for the Moon, close to the anti-solar point; for the Sun, a direction
        # set by the planets' pull on the barycentre), NOT the apparent sky
        # position. Instead, project to the site's *vacuum* horizontal frame and
        # back to ICRS so the result is the apparent place, consistent with
        # get_body_altaz and with get_parallactic_angle's pressure=0 transform.
        # A vacuum frame (pressure=0) is used regardless of this instance's
        # atmosphere so the RA/Dec is the geometric apparent place.
        body_coord = get_body(body_spec, obstime, location=self.location, ephemeris=ephemeris)
        vacuum_altaz = AltAz(obstime=obstime, location=self.location, pressure=0 * u.hPa)
        altaz = body_coord.transform_to(vacuum_altaz)
        icrs = SkyCoord(az=altaz.az, alt=altaz.alt, frame=vacuum_altaz).transform_to("icrs")

        ra = icrs.ra.deg
        dec = icrs.dec.deg

        if obstime.isscalar:
            return float(ra), float(dec)
        return ra, dec

    def get_sun_altaz(self, obstime: Time) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Get the Az/El position of the Sun.

        Convenience method for sun avoidance calculations.

        Parameters
        ----------
        obstime : Time
            Observation time. Can be a scalar Time or an array of Times
            (forwarded to :meth:`get_body_altaz`).

        Returns
        -------
        az : float or array
            Sun azimuth in degrees.
        alt : float or array
            Sun altitude (elevation) in degrees.
        """
        return self.get_body_altaz("sun", obstime)

    def angular_separation(
        self,
        az1: float | np.ndarray,
        alt1: float | np.ndarray,
        az2: float | np.ndarray,
        alt2: float | np.ndarray,
    ) -> float | np.ndarray:
        """Calculate angular separation between two Az/El positions.

        Parameters
        ----------
        az1, alt1 : float or array
            First position (azimuth, altitude) in degrees.
        az2, alt2 : float or array
            Second position (azimuth, altitude) in degrees.

        Returns
        -------
        float or array
            Angular separation in degrees. Array inputs broadcast against
            each other and the result keeps the broadcast shape, which is
            how the observability grid and the Sun-zone renderers use it.
        """
        c1 = SkyCoord(az=az1 * u.deg, alt=alt1 * u.deg, frame="altaz")
        c2 = SkyCoord(az=az2 * u.deg, alt=alt2 * u.deg, frame="altaz")
        return c1.separation(c2).deg

    def is_sun_safe(
        self,
        az: float | np.ndarray,
        el: float | np.ndarray,
        obstime: Time,
    ) -> bool | np.ndarray:
        """Check if a position is safe from Sun exposure.

        Parameters
        ----------
        az : float or array
            Azimuth in degrees.
        el : float or array
            Elevation in degrees.
        obstime : Time
            Observation time.

        Returns
        -------
        bool or array of bool
            True if the Sun separation is strictly greater than the site's
            exclusion radius; a position exactly at the exclusion radius
            counts as unsafe. Returns True unconditionally when the site's
            sun avoidance is disabled.

        Notes
        -----
        The scalar form is the
        :class:`~fyst_trajectories.sun_protocols.SunSafePredicate` contract.
        The array form broadcasts the verdict to the shape of the inputs,
        except with avoidance disabled, where the answer is a plain ``True``
        whatever the input shape. The scalar model ``make_sun_safe("scalar")``
        builds, the default Sun test of
        :func:`~fyst_trajectories.dispatch.choose_encoder_solution`, gives the
        same verdicts and adds the vectorised ``batch`` extension.

        The Sun position is computed with this instance's atmosphere
        (vacuum by default), so pass ``az``/``el`` in the same frame.
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
                f"Elevation {el:.1f} deg outside limits "
                f"[{limits.elevation.min}, {limits.elevation.max}]",
            )

        if not limits.azimuth.is_in_range(az):
            return (
                False,
                f"Azimuth {az:.1f} deg outside limits [{limits.azimuth.min}, {limits.azimuth.max}]",
            )

        if check_sun and self.site.sun_avoidance.enabled:
            sun_az, sun_alt = self.get_sun_altaz(obstime)
            sep = self.angular_separation(az, el, sun_az, sun_alt)
            if sep <= self.site.sun_avoidance.exclusion_radius:
                return False, f"Position too close to Sun (separation: {sep:.1f} deg)"

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

        Raises
        ------
        ValueError
            If ``step_hours`` or ``max_search_hours`` is not a finite value > 0.

        Notes
        -----
        Returns (None, None) for circumpolar or never-visible sources.
        Finds the FIRST rise, then the FIRST set at or after that rise.

        The search covers ``[start_time, start_time + max_search_hours]``:
        the grid steps by ``step_hours`` and its last sample lands on
        ``start_time + max_search_hours``, so the last step may be shorter.

        Refraction is disabled (pressure=0). Calculated times
        may differ from observed rise/set by a few minutes (~0.5 deg
        refraction near horizon). Use a lower horizon value to compensate.

        Between coarse grid points, altitude may have local extrema that
        the grid misses, especially for sources with grazing passes near
        the horizon. Use a smaller step_hours for such cases.

        The crossing time is estimated by linear interpolation between
        adjacent grid points, so it resolves to a fraction of
        ``step_hours``.

        Examples
        --------
        >>> from astropy.time import Time
        >>> coords = Coordinates(site)  # refraction is disabled internally anyway
        >>> start = Time("2026-03-15T00:00:00", scale="utc")
        >>> # Find when the Crab Nebula rises and sets
        >>> rise, set_ = coords.get_rise_set_times(
        ...     83.633,
        ...     22.014,
        ...     start_time=start,
        ...     horizon=0.0,
        ...     max_search_hours=48.0,
        ...     step_hours=0.1,
        ... )
        >>> if rise is not None and set_ is not None:
        ...     print(f"Rises at: {rise.iso}")
        ...     print(f"Sets at: {set_.iso}")
        ... else:
        ...     print("Source is circumpolar, never visible, or does not set within window")
        Rises at: 2026-03-15 17:13:...
        Sets at: 2026-03-16 03:52:...

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

        if not (np.isfinite(step_hours) and step_hours > 0):
            raise ValueError(f"step_hours must be a finite value > 0, got {step_hours}")
        if not (np.isfinite(max_search_hours) and max_search_hours > 0):
            raise ValueError(f"max_search_hours must be a finite value > 0, got {max_search_hours}")
        times = _build_time_grid(start_time, max_search_hours, step_hours * 60.0)

        altaz_frame = AltAz(
            obstime=times,
            location=self.location,
            pressure=0 * u.hPa,
        )
        altitudes = source.transform_to(altaz_frame).alt.to_value(u.deg)

        rises = _threshold_crossings(altitudes, times, horizon, rising=True)
        if not rises:
            return None, None
        rise_time = rises[0]
        sets = _threshold_crossings(altitudes, times, horizon, rising=False)
        set_time = next((t for t in sets if t >= rise_time), None)
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

        Notes
        -----
        ``HA = LST - RA`` pairs the apparent-equinox local sidereal time
        (``sidereal_time("apparent")``) with the supplied RA. When that RA is a
        catalogue (ICRS/J2000) value, the result carries the precession of RA
        since J2000 (about 0.34 deg in 2026 on the celestial equator, growing
        about 0.014 deg/yr there; both vary with RA and declination): it is
        the hour angle relative to the *mean* catalogue position, not the
        apparent place. That is adequate for the coarse scheduling uses in this
        library (transit finding, rising/setting sign) but **not** for
        parallactic-angle or precise pointing work. Use
        :meth:`get_parallactic_angle`, which transforms to Az/El and is
        referenced to the apparent pole.

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
        The parallactic angle is derived from the *transformed* horizontal
        coordinates (Az, El), using the IAU North-through-East AltAz form

        tan(q) = (-sin(A) cos(phi)) / (sin(phi) cos(a) - cos(phi) sin(a) cos(A))

        where ``A`` is azimuth, ``a`` is elevation and ``phi`` is the site
        latitude. RA/Dec are transformed to Az/El first, so the full
        precession/nutation/aberration chain is folded into the geometry and
        the result is referenced to the **apparent** celestial pole. Computing
        the angle from ``HA = LST - RA`` instead would mix the apparent-equinox
        LST with the catalogue (ICRS/J2000) RA, leaving the RA precession since
        J2000 (see :meth:`get_hour_angle`) uncorrected. This is the same AltAz
        form used by
        :meth:`get_field_rotation_from_altaz`, so the two paths agree.

        A vacuum (zero-pressure) transform is used regardless of the
        atmosphere configured on this ``Coordinates`` instance, so the
        parallactic angle is the geometric sky-vs-mount rotation.

        Near the zenith the parallactic angle is ill-conditioned: it is
        undefined exactly at the zenith, and at transit it swings through
        180 deg at a rate set by the transit zenith distance (roughly 820 s
        per degree, so seconds only within a few hundredths of a degree
        of the zenith-crossing declination, over an hour 5 deg away).
        ``arctan2`` keeps the computation
        finite, but the result is **not** close to 0 there; downstream consumers that
        depend on PA continuity (e.g. focal-plane rotation rate) should be
        aware. FYST's lat = -22.99 deg puts sources with dec of about -18 to -28 deg in
        this regime, and the site's elevation limits (20 to 90 deg) do not keep a
        pointing away from the zenith.

        Examples
        --------
        >>> from astropy.time import Time
        >>> coords = Coordinates(site)
        >>> obstime = Time("2026-01-15T02:00:00", scale="utc")
        >>> pa = coords.get_parallactic_angle(83.633, 22.014, obstime=obstime)
        >>> print(f"Parallactic angle: {pa:.2f} deg")
        Parallactic angle: -170.66 deg
        """
        # Transform RA/Dec -> vacuum Az/El, then take the AltAz-form PA (see
        # Notes): this references the result to the apparent pole and keeps it
        # geometric, independent of this instance's atmosphere.
        altaz = self._vacuum_altaz(ra, dec, obstime)
        pa_deg = _parallactic_angle_from_altaz(altaz.az.rad, altaz.alt.rad, self.site.latitude)

        if np.isscalar(ra) and np.isscalar(dec) and obstime.isscalar:
            return float(pa_deg)
        return pa_deg

    def _vacuum_altaz(
        self,
        ra: float | np.ndarray,
        dec: float | np.ndarray,
        obstime: Time,
    ) -> SkyCoord:
        """Transform an ICRS position into the site's vacuum horizon frame.

        The zero-pressure frame is what the geometric quantities are defined
        in, so building it here keeps the parallactic angle and the field
        rotation on one convention and lets a caller that needs both pay for
        a single transform.

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
        SkyCoord
            The position in a zero-pressure ``AltAz`` frame at this site.
        """
        sky_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        altaz_frame = AltAz(obstime=obstime, location=self.location, pressure=0 * u.hPa)
        return sky_coord.transform_to(altaz_frame)

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

        For the focal plane's orientation relative to the horizon (az/el)
        axes, the rotation the az/el projections use, call
        :func:`~fyst_trajectories.offsets.compute_focal_plane_rotation`,
        which returns ``nasmyth_sign * elevation + instrument_rotation`` and
        takes the parallactic angle only as an optional argument.

        This is the RA/Dec entry point to one quantity that the package
        publishes under two names, both of which share a single
        parallactic-angle kernel and agree to machine precision:
        :meth:`get_field_rotation_from_altaz` returns the same sum from an
        already-transformed horizon position and is what stamps a timeline
        block's ``boresight_angle``. This method is the canonical one: it
        owns the vacuum transform the geometry is defined in.

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
        The Nasmyth sign is +1 for Right Nasmyth, -1 for Left Nasmyth,
        and 0 for Cassegrain (no elevation-dependent rotation).

        Like :meth:`get_parallactic_angle`, the elevation term is computed with
        a vacuum (zero-pressure) transform, so the field rotation is the
        geometric sky-vs-mount rotation regardless of the atmosphere configured
        on this instance.

        See Also
        --------
        :func:`~fyst_trajectories.offsets.compute_focal_plane_rotation` :
            Mechanical (horizon-frame) focal-plane rotation; the parallactic
            angle is an optional argument there.

        Examples
        --------
        >>> from astropy.time import Time
        >>> coords = Coordinates(site)
        >>> fr = coords.get_field_rotation(83.633, 22.014, Time("2026-03-15T04:00:00", scale="utc"))
        """
        # Use a vacuum (pressure=0) elevation so the mechanical Nasmyth term is
        # frame-consistent with the vacuum geometric parallactic angle: a
        # refracted el would leak the refraction bump into the mechanical term
        # while pa stays vacuum. Matches get_parallactic_angle's convention, so
        # the result is the geometric field rotation regardless of this
        # instance's atmosphere. Both terms come from one transform: the
        # elevation and the parallactic angle are read off the same position.
        altaz = self._vacuum_altaz(ra, dec, obstime)
        el = altaz.alt.deg
        pa = _parallactic_angle_from_altaz(altaz.az.rad, altaz.alt.rad, self.site.latitude)

        field_rotation = self.site.nasmyth_sign * el + pa

        if np.isscalar(ra) and np.isscalar(dec) and obstime.isscalar:
            return float(field_rotation)
        return field_rotation

    def get_field_rotation_from_altaz(
        self,
        az: float | np.ndarray,
        el: float | np.ndarray,
    ) -> float | np.ndarray:
        """Calculate sky field rotation from a horizon position.

        Returns ``site.nasmyth_sign * el + parallactic_angle`` for a pose
        that is already in azimuth and elevation: the same quantity
        :meth:`get_field_rotation` returns from RA/Dec, the sky orientation
        of the Nasmyth-mounted focal plane. Both methods take the
        parallactic angle from one shared kernel, so they agree to machine
        precision for the same pose. This is the value a timeline block
        records as its ``boresight_angle``.

        Pass the geometric (vacuum) pose, the one a ``Coordinates`` built
        without refraction returns. The method uses the pose as given, so
        the atmosphere configured on this instance has no effect; a
        refracted elevation moves the result, by more the lower the pose
        and the nearer it is to the celestial pole.

        Parameters
        ----------
        az : float or array
            Azimuth in degrees.
        el : float or array
            Elevation in degrees.

        Returns
        -------
        float or array
            Field rotation in degrees (nasmyth_sign * elevation + parallactic
            angle). A float when both inputs are scalars, otherwise an array
            of the broadcast shape.

        See Also
        --------
        :meth:`get_field_rotation` : The same quantity from RA/Dec and a time.
        :func:`~fyst_trajectories.offsets.compute_focal_plane_rotation` :
            Mechanical (horizon-frame) focal-plane rotation; the parallactic
            angle is an optional argument there.

        Examples
        --------
        >>> from astropy.time import Time
        >>> coords = Coordinates(site)
        >>> t = Time("2026-03-15T04:00:00", scale="utc")
        >>> az, el = coords.radec_to_altaz(83.633, 22.014, t)
        >>> fr = coords.get_field_rotation_from_altaz(az, el)
        >>> abs(fr - coords.get_field_rotation(83.633, 22.014, t)) < 1e-9
        True
        """
        pa = _parallactic_angle_from_altaz(np.radians(az), np.radians(el), self.site.latitude)
        field_rotation = self.site.nasmyth_sign * np.asarray(el, dtype=float) + pa

        if np.isscalar(az) and np.isscalar(el):
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
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
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
            Observation time to compute position for: a scalar, or an array
            of times, each propagated from ``ref_epoch`` in one call.
        distance : float, optional
            Distance in parsecs. If provided along with radial_velocity,
            enables full 3D space motion propagation. If None, only 2D proper
            motion on the sky is used.
        radial_velocity : float, optional
            Radial velocity in km/s (positive = receding). Used for full 3D
            space motion propagation when distance is also provided, and
            ignored without one.
        frame : str, optional
            Input coordinate frame. Default is "icrs". Passed through
            :func:`normalize_frame`; ``"HORIZON"`` and ``"B1950"`` are refused,
            since this method reads ``ra``/``dec`` and astropy cannot apply
            space motion in the ``fk4`` frame.

        Returns
        -------
        az : float or array
            Azimuth in degrees at each observation time; a float for a
            scalar ``obstime``.
        alt : float or array
            Altitude (elevation) in degrees at each observation time.

        Raises
        ------
        ValueError
            If ``frame`` names the horizontal frame or ``"B1950"`` (``fk4``).

        Examples
        --------
        Track Barnard's Star (high proper motion):

        >>> from astropy.time import Time
        >>> # Barnard's Star, J2000 catalogue position and proper motion
        >>> ra, dec = 269.452, 4.693  # degrees
        >>> pmra, pmdec = -798.58, 10328.12  # mas/yr
        >>> ref_epoch = Time("J2000.0")
        >>> obs_time = Time("2026-06-15T04:00:00")
        >>> az, el = coords.radec_to_altaz_with_pm(
        ...     ra, dec, pmra, pmdec, ref_epoch, obstime=obs_time
        ... )
        """
        frame = _radec_frame(frame)
        if frame == "fk4":
            raise ValueError(
                "frame='B1950' (fk4) is not supported here: astropy cannot apply "
                "space motion in a frame with its own obstime. Convert the catalogue "
                "position to ICRS or FK5 first."
            )
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

        # Without a distance, ERFA pmsafe propagates on the sphere and warns
        # "distance overridden" for every time; the filter silences that. It is
        # process-global for its duration, so a concurrent thread can lose an
        # unrelated ERFA warning for the length of one call.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*distance overridden.*",
                category=erfa.ErfaWarning,
            )
            coord_at_obs = SkyCoord(**coord_kwargs).apply_space_motion(new_obstime=obstime)

        return self.radec_to_altaz(
            coord_at_obs.ra.deg,
            coord_at_obs.dec.deg,
            obstime=obstime,
            frame=frame,
        )
