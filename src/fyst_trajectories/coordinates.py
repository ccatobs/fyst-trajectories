"""Coordinate transformations for telescope pointing.

This module provides coordinate transformation utilities for converting
between celestial coordinates (RA/Dec) and horizontal coordinates (Az/El),
with support for atmospheric refraction corrections and solar system
ephemeris calculations.

The transformations use astropy's coordinate transformation framework
with IERS data for Earth orientation parameters.

``Coordinates(site)`` defaults to vacuum (geometric) coordinates because
the FYST ACU applies atmospheric refraction downstream. For planning and
simulation (visibility calculations, observability checks, hitmap
simulations) where the output is NOT sent to the ACU, pass
``AtmosphericConditions.for_fyst()`` to apply submillimetre refraction.

Examples
--------
Trajectory generation (vacuum; the ACU applies refraction):

>>> from astropy.time import Time
>>> from fyst_trajectories.coordinates import Coordinates
>>> from fyst_trajectories.site import get_fyst_site
>>> coords = Coordinates(get_fyst_site())
>>> obstime = Time("2026-03-15T04:00:00", scale="utc")
>>> az, el = coords.radec_to_altaz(83.633, 22.014, obstime=obstime)  # Crab Nebula
>>> print(f"Az: {az:.2f}°, El: {el:.2f}°")

Planning with refraction (visibility checks, not sent to ACU):

>>> from fyst_trajectories.site import AtmosphericConditions
>>> coords_plan = Coordinates(get_fyst_site(), atmosphere=AtmosphericConditions.for_fyst())
>>> az, el = coords_plan.radec_to_altaz(83.633, 22.014, obstime=obstime)
"""

import importlib.util
import os
import warnings
from dataclasses import dataclass
from types import MappingProxyType

import erfa
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord, get_body
from astropy.time import Time, TimeDelta

from .site import AtmosphericConditions, Site

# ``erfa`` (PyPI: ``pyerfa``) ships ``ErfaWarning`` in every release reachable
# from any astropy>=5.0 install (the dependency floor in pyproject.toml), so
# the import and attribute lookup are unconditional. The previous defensive
# try/except fell back to ``UserWarning``, which would have silently demoted
# real ERFA messages if it ever fired, a worse signal than failing loudly.
_erfa_warning_cls = erfa.ErfaWarning

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


# Frame name aliases for KOSMA/OCS compatibility
# Maps common telescope control system names to astropy frame names.
# Note: ``"J2000"`` maps to ICRS, not FK5 J2000.0. The two frames differ
# by ~22 mas at the catalogue level (the IAU 1997 alignment of ICRS to
# FK5). For sub-arcsecond catalogue work this matters; for telescope
# pointing it is well below the beam and is harmless.
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
"""Frame-name aliases mapping KOSMA/OCS names to astropy frame names.

Only spherical RA/Dec frames are aliased; ``"J2000"`` maps to ICRS (not FK5
J2000.0), which is harmless for telescope pointing. Consumed by ``normalize_frame``.
"""


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

    Notes
    -----
    Only spherical RA/Dec frames (``icrs``/``J2000``, ``fk5``/``FK5``,
    ``fk4``/``B1950``) are usable with :meth:`Coordinates.radec_to_altaz` and
    :meth:`Coordinates.altaz_to_radec`, which read ``ra``/``dec`` attributes.
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
        return f"AltAzCoord(az={self.az:.4f}°, alt={self.alt:.4f}°)"


class Coordinates:
    """Coordinate transformation engine for a telescope site.

    This class provides methods for converting between celestial and
    horizontal coordinate systems, with optional atmospheric refraction
    and solar system ephemeris calculations.

    The default (``atmosphere=None``) produces vacuum (geometric)
    coordinates. This is the correct default for trajectory generation
    because the FYST ACU applies atmospheric refraction downstream.
    Pass ``AtmosphericConditions.for_fyst()`` for planning and
    simulation where the output is not sent to the ACU.

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
    Trajectory generation (vacuum; ACU applies refraction):

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

        Examples
        --------
        >>> from astropy.time import Time
        >>> obstime = Time("2026-03-15T16:00:00", scale="utc")
        >>> az, el = coords.get_body_altaz("mars", obstime)
        """
        body = body.lower()
        body_spec, ephemeris = self._resolve_body(body)

        # Use get_body uniformly (not get_sun) so every body shares one
        # topocentric code path. Passing location= is what makes the AltAz
        # apparent place site-topocentric; that parallax is physically
        # meaningful for finite-distance bodies (the Moon, ~1°) and negligible
        # for the Sun (~0.01″ on-sky). The visible get_sun()-vs-get_body()
        # difference (~arcsec) is an ephemeris/algorithm difference, not
        # parallax (the old "~8.8 arcsec" note was the horizontal-parallax
        # constant, not the on-sky shift).
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
        # (e.g. the anti-solar point for the Sun), NOT the apparent sky
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
            True if the Sun separation is strictly greater than the site's
            exclusion radius; a position exactly at the exclusion radius
            counts as unsafe. Returns True unconditionally when the site's
            sun avoidance is disabled.
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
        >>> coords = Coordinates(site, atmosphere=AtmosphericConditions.for_fyst())
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
        >>> if rise is not None and set_ is not None:
        ...     print(f"Rises at: {rise.iso}")
        ...     print(f"Sets at: {set_.iso}")
        ... else:
        ...     print("Source is circumpolar, never visible, or does not set within window")

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

        Notes
        -----
        ``HA = LST − RA`` pairs the apparent-equinox local sidereal time
        (``sidereal_time("apparent")``) with the supplied RA. When that RA is a
        catalogue (ICRS/J2000) value, the result carries the precession of RA
        since J2000 (dec-dependent, ~0.3° in 2026, growing ~0.018°/yr): it is
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

        tan(q) = (−sin(A) cos(φ)) / (sin(φ) cos(a) − cos(φ) sin(a) cos(A))

        where ``A`` is azimuth, ``a`` is elevation and ``φ`` is the site
        latitude. RA/Dec are transformed to Az/El first, so the full
        precession/nutation/aberration chain is folded into the geometry and
        the result is referenced to the **apparent** celestial pole. Computing
        the angle from ``HA = LST − RA`` instead would mix the apparent-equinox
        LST with the catalogue (ICRS/J2000) RA, leaving an uncorrected
        precession term (~0.3° in 2026, growing ~0.013-0.018°/yr depending on
        declination) in the parallactic angle. This is the same AltAz form used by
        ``overhead.utils.compute_nasmyth_rotation``, so the two paths agree.

        A vacuum (zero-pressure) transform is used regardless of the
        atmosphere configured on this ``Coordinates`` instance, so the
        parallactic angle is the geometric sky-vs-mount rotation.

        Near the zenith the parallactic angle is ill-conditioned: it is
        undefined exactly at the zenith and swings through 180° within a few
        seconds at transit for sources whose declination is close to the site
        latitude (``|dec − lat|`` small). ``arctan2`` keeps the computation
        finite, but the result is **not** ≈ 0 there; downstream consumers that
        depend on PA continuity (e.g. focal-plane rotation rate) should be
        aware. FYST's lat = −22.99° puts sources with dec ≈ −18° to −28° in
        this regime; the ``el_min = 20°`` constraint mitigates but does not
        eliminate the issue.

        Examples
        --------
        >>> from astropy.time import Time
        >>> coords = Coordinates(site)
        >>> obstime = Time("2026-03-15T04:00:00", scale="utc")
        >>> pa = coords.get_parallactic_angle(83.633, 22.014, obstime=obstime)
        >>> print(f"Parallactic angle: {pa:.2f}°")
        """
        # Transform RA/Dec -> vacuum Az/El, then take the AltAz-form PA (see
        # Notes): this references the result to the apparent pole and keeps it
        # geometric, independent of this instance's atmosphere.
        sky_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        altaz_frame = AltAz(obstime=obstime, location=self.location, pressure=0 * u.hPa)
        altaz = sky_coord.transform_to(altaz_frame)

        az_rad = altaz.az.rad
        el_rad = altaz.alt.rad
        lat_rad = np.deg2rad(self.site.latitude)

        sin_az = np.sin(az_rad)
        cos_az = np.cos(az_rad)
        sin_el = np.sin(el_rad)
        cos_el = np.cos(el_rad)
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)

        numerator = -sin_az * cos_lat
        denominator = sin_lat * cos_el - cos_lat * sin_el * cos_az

        pa_deg = np.rad2deg(np.arctan2(numerator, denominator))

        if np.isscalar(ra) and np.isscalar(dec) and obstime.isscalar:
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
        rotation, use ``fyst_trajectories.offsets.compute_focal_plane_rotation``
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

        Like :meth:`get_parallactic_angle`, the elevation term is computed with
        a vacuum (zero-pressure) transform, so the field rotation is the
        geometric sky-vs-mount rotation regardless of the atmosphere configured
        on this instance.

        See Also
        --------
        :func:`~fyst_trajectories.offsets.compute_focal_plane_rotation` :
            Full focal-plane rotation including Nasmyth sign and
            instrument rotation.

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
        # instance's atmosphere.
        sky_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        altaz_frame = AltAz(obstime=obstime, location=self.location, pressure=0 * u.hPa)
        el = sky_coord.transform_to(altaz_frame).alt.deg

        pa = self.get_parallactic_angle(ra, dec, obstime)

        field_rotation = self.site.nasmyth_sign * el + pa

        if np.isscalar(ra) and np.isscalar(dec) and obstime.isscalar:
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
            # Without a real distance, use a large dummy distance (1 Mpc)
            # to leverage astropy's spherical proper motion propagation.
            # This is the documented workaround for the no-distance case
            # — see astropy issues #10092 and #10296 — and avoids the
            # cos(dec) singularity of a naive linear approach at the
            # celestial poles. The Barnard's Star regression test in
            # tests/test_coordinates.py guards against future astropy
            # behaviour drift here; if astropy ever gains a first-class
            # no-distance code path, the test will catch the change.
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
