"""Site configuration for telescope locations.

This module provides classes and utilities for managing telescope site
configurations, including geographic location and telescope mechanical
limits.  FYST telescope parameters are defined as module-level constants
(``FYST_LATITUDE``, ``FYST_LONGITUDE``, etc.) and the convenience
function ``get_fyst_site()`` builds a ``Site`` from those constants
with no file I/O.

Atmospheric conditions are always provided by the user at runtime via
``AtmosphericConditions``, not from constants or config files.

For custom (non-FYST) sites, use ``Site.from_config()`` with a YAML
file or construct a ``Site`` directly.

Notes
-----
Pointing-model corrections (collimation, gravitational sag, encoder
offsets) and PWV/opacity modelling are intentionally out of scope.
Pointing-model corrections are applied at execution time by the FYST
Telescope Control System; PWV/opacity modelling lives in the
downstream calibration pipeline. This module describes the *site
geometry and mechanical limits* used by trajectory planning, nothing
more.

Examples
--------
Get the default FYST site:

>>> from fyst_trajectories.site import get_fyst_site
>>> site = get_fyst_site()
>>> print(site.latitude)
-22.985639

Load a custom (non-FYST) configuration from YAML:

>>> site = Site.from_config("/path/to/custom_config.yaml")
"""

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy import units as u
from astropy.coordinates import EarthLocation

# Tier 1: Truly fixed (geography / optical prescription)
# Source: FYST TCS (telescope-control-system/astro.go)
#   lat = -22d59m08.30s, lon = -67d44m25.00s, elev = 5611.8 m

FYST_LATITUDE: float = -22.985639
"""FYST latitude in degrees (South). Source: FYST TCS astro.go."""

FYST_LONGITUDE: float = -67.740278
"""FYST longitude in degrees (West). Source: FYST TCS astro.go."""

FYST_ELEVATION: float = 5611.8
"""FYST elevation in meters above sea level. Source: FYST TCS astro.go."""

FYST_PLATE_SCALE: float = 13.89
"""FYST plate scale in arcsec/mm. Source: optical design."""
# UNVERIFIED: see "Pending instrument verification" in docs/index.rst

FYST_NASMYTH_PORT: str = "right"
"""Nasmyth port side for instrument mounting ('right' = +1 sign)."""
# UNVERIFIED: see "Pending instrument verification" in docs/index.rst
# Wrong sign produces up to 2 * el degrees of focal-plane rotation error
# at every off-axis Prime-Cam module. Confirmation essential before
# first-light science.

# Tier 2: Mechanical limits (from FYST TCS commands.go:18-28). Two envelopes:
# the ENCODER/SLEW envelope (az [-180, 360], el [-90, 180]) is enforced by Go
# TCS and matches P-INCM-ICD-0003-A sections 2/5; the narrower OBSERVING
# envelope (el [20, 90]) is enforced by the planners and validate_trajectory.
# UNVERIFIED: see "Pending instrument verification" in docs/index.rst
# The velocity and acceleration limits below — the conservative operational
# values may need to be relaxed once commissioning ratifies the TCS
# hardware limits cited in commands.go.

FYST_AZ_MIN: float = -180.0
"""Minimum azimuth in degrees. Source: FYST TCS commands.go."""

FYST_AZ_MAX: float = 360.0
"""Maximum azimuth in degrees. Source: FYST TCS commands.go."""

FYST_AZ_MAX_VELOCITY: float = 3.0
"""Maximum azimuth velocity in degrees/second. Source: FYST TCS."""

FYST_AZ_MAX_ACCELERATION: float = 1.5
"""Maximum azimuth acceleration in degrees/second^2.

Conservative operational limit (TCS hardware limit: 6.0 deg/s^2).
The operational value sits at the quintic scan turnaround peak
(1.5 * az_accel, which equals 1.5 for the planner's default az_accel of
1.0 deg/s^2), so a default constant-elevation plan does not warn about the
acceleration of its own trajectory.
"""

FYST_EL_MIN: float = 20.0
"""Minimum elevation in degrees.

Conservative operational limit (planning-layer choice), not a TCS bound: the
Go TCS accepts el down to -90 deg (commands.go).
"""

FYST_EL_MAX: float = 90.0
"""Maximum elevation in degrees.

Conservative operational limit (planning-layer choice): Prime-Cam does not point
over the top. The Go TCS hardware bound is 180 deg / el>90 (commands.go).
"""

FYST_EL_MAX_VELOCITY: float = 1.0
"""Maximum elevation velocity in degrees/second.

Conservative operational limit (TCS hardware limit: 1.5 deg/s).
"""

FYST_EL_MAX_ACCELERATION: float = 0.75
"""Maximum elevation acceleration in degrees/second^2.

Conservative operational limit (TCS hardware limit: 1.5 deg/s^2).
Raised in tandem with the azimuth limit (1.5x the previous floor) as a
companion consistency adjustment; it clears the near-limit elevation band
for aggressive pong scans, while genuinely over-limit plans still warn.
"""

# Tier 3: Operational defaults (may change between observing seasons)

# FYST sun avoidance radii. The 45 deg exclusion is comparable to the
# Simons Observatory SAT's 41 deg at similar altitude and submillimetre
# wavelengths (Hoang et al. 2024, arXiv:2406.10905), supporting the
# magnitude. ALMA uses 15 deg (much more permissive) at lower altitude
# with different optical/thermal constraints.
# UNVERIFIED: see "Pending instrument verification" in docs/index.rst
FYST_SUN_EXCLUSION_RADIUS: float = 45.0
"""Sun exclusion radius in degrees."""

# UNVERIFIED: see "Pending instrument verification" in docs/index.rst
FYST_SUN_WARNING_RADIUS: float = 50.0
"""Sun warning radius in degrees."""

FYST_SUN_AVOIDANCE_ENABLED: bool = True
"""Whether sun avoidance is enabled by default."""


def _get_required(
    config_dict: dict,
    key: str,
    section: str,
    config_name: str = "config",
) -> Any:
    """Get a required value from config dict, raising error if missing.

    Parameters
    ----------
    config_dict : dict
        The configuration dictionary to search.
    key : str
        The key to look up in config_dict.
    section : str
        The section name for error message (e.g., "telescope.azimuth").
    config_name : str
        Name of config file for error message.

    Returns
    -------
    Any
        The config value.

    Raises
    ------
    ValueError
        If the key is missing from the config dict.
    """
    if key in config_dict:
        return config_dict[key]

    raise ValueError(
        f"Config '{config_name}' missing required key '{key}' in section '{section}'. "
        f"Please add '{key}' to your configuration file."
    )


@dataclass(frozen=True)
class AtmosphericConditions:
    """Atmospheric conditions at the observing site.

    These parameters are used for atmospheric refraction corrections.
    Should be constructed with current weather data and passed to
    ``Coordinates(site, atmosphere=...)``,
    ``TrajectoryBuilder.with_atmosphere()``, or planning functions.
    Atmosphere is never loaded from config files.

    Most callers should construct via the factory classmethods rather than
    the raw constructor: :meth:`for_fyst` for typical Cerro Chajnantor
    submm conditions (sets ``obswl=200 µm`` so astropy uses the radio
    refraction model), or :meth:`no_refraction` to explicitly disable
    refraction (vacuum coordinates, useful for cross-validation).

    Parameters
    ----------
    pressure : float
        Atmospheric pressure in hPa. Must be non-negative; ``0`` disables
        refraction (see :meth:`no_refraction`).
    temperature : float
        Temperature in Kelvin.
    relative_humidity : float
        Relative humidity as a fraction (0-1).
    obswl : float or None, optional
        Observing wavelength in microns. When ``> 100 µm``, astropy uses
        the radio refraction model instead of optical. The radio model is
        wavelength-independent, so any value above 100 µm (e.g. 200 µm)
        covers all FYST submillimeter bands. Default is ``None``, which
        preserves astropy's default optical refraction (1.0 µm). Must be
        positive when provided.

    Raises
    ------
    ValueError
        If ``relative_humidity`` is outside ``[0, 1]``, ``pressure`` is
        negative, or ``obswl`` is non-positive.

    See Also
    --------
    AtmosphericConditions.for_fyst :
        Factory for FYST-typical submm conditions (recommended default).
    AtmosphericConditions.no_refraction :
        Factory for vacuum (refraction-disabled) coordinates.
    """

    pressure: float
    temperature: float
    relative_humidity: float
    obswl: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.relative_humidity <= 1:
            raise ValueError(
                f"relative_humidity must be in range [0, 1], got {self.relative_humidity}"
            )
        # ``pressure == 0`` is load-bearing (it is how ``no_refraction`` disables
        # refraction), so only negative pressure is rejected. ``obswl <= 0`` is the
        # real silent-garbage path: ERFA's optical-dispersion term evaluates to a
        # finite but wrong refraction constant (~2x over-refraction) rather than
        # erroring; a negative pressure is harmless (ERFA clamps it to vacuum).
        if self.pressure < 0:
            raise ValueError(f"pressure must be >= 0 hPa, got {self.pressure}")
        if self.obswl is not None and self.obswl <= 0:
            raise ValueError(f"obswl must be > 0 microns, got {self.obswl}")

    @classmethod
    def no_refraction(cls) -> "AtmosphericConditions":
        """Create atmospheric conditions that disable refraction correction.

        Setting pressure to zero causes astropy's AltAz frame to skip
        atmospheric refraction, producing geometric (vacuum) coordinates.
        Useful for cross-validation against backends that don't model
        refraction, or for testing.

        Returns
        -------
        AtmosphericConditions
            Instance with pressure=0 (no refraction).

        Examples
        --------
        >>> from fyst_trajectories.site import AtmosphericConditions
        >>> atmo = AtmosphericConditions.no_refraction()
        >>> atmo.pressure
        0.0
        """
        return cls(pressure=0.0, temperature=0.0, relative_humidity=0.0)

    @classmethod
    def for_fyst(
        cls,
        pressure: float = 500.0,
        temperature: float = 265.0,
        relative_humidity: float = 0.10,
        obswl: float = 200.0,
    ) -> "AtmosphericConditions":
        """Create FYST-typical atmospheric conditions with submm refraction.

        Convenience factory that defaults to a "typical winter night on
        Cerro Chajnantor" weather profile and forces ``obswl=200 µm`` so
        astropy switches to its radio-IR refraction model. The radio
        model is wavelength-independent above ~100 µm; any value above
        that threshold covers all FYST submillimeter bands.
        Without this factory, callers who pass realistic
        pressure/temperature/humidity but forget ``obswl`` silently get
        astropy's optical (1 µm) refraction model.
        Pressure/temperature defaults follow Bustos et al. 2014 / Cortes
        et al. 2016 for the Chajnantor plateau and are appropriate
        starting values; pass current weather data when available.

        Parameters
        ----------
        pressure : float, optional
            Atmospheric pressure in hPa. Default 500.
        temperature : float, optional
            Temperature in Kelvin. Default 265.
        relative_humidity : float, optional
            Relative humidity as a fraction (0-1). Default 0.10.
        obswl : float, optional
            Observing wavelength in microns. Default 200 (selects the
            radio refraction model).

        Returns
        -------
        AtmosphericConditions
            FYST-typical atmospheric conditions.

        Examples
        --------
        >>> from fyst_trajectories.site import AtmosphericConditions
        >>> atmo = AtmosphericConditions.for_fyst()
        >>> atmo.obswl
        200.0
        """
        return cls(
            pressure=pressure,
            temperature=temperature,
            relative_humidity=relative_humidity,
            obswl=obswl,
        )

    @property
    def pressure_hpa(self) -> u.Quantity:
        """Pressure as an astropy Quantity in hPa."""
        return self.pressure * u.hPa

    @property
    def temperature_degc(self) -> u.Quantity:
        """Temperature as an astropy Quantity in Celsius (for AltAz frame)."""
        return (self.temperature - 273.15) * u.deg_C

    @property
    def obswl_quantity(self) -> u.Quantity | None:
        """Observing wavelength as an astropy Quantity in microns, or None."""
        if self.obswl is None:
            return None
        return self.obswl * u.micron


@dataclass(frozen=True)
class AxisLimits:
    """Motion limits for a telescope axis.

    Parameters
    ----------
    min : float
        Minimum position in degrees.
    max : float
        Maximum position in degrees.
    max_velocity : float
        Maximum velocity in degrees/second.
    max_acceleration : float
        Maximum acceleration in degrees/second^2.

    Raises
    ------
    ValueError
        If min > max.
    """

    min: float
    max: float
    max_velocity: float
    max_acceleration: float

    def __post_init__(self) -> None:
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")

    def is_in_range(self, position: float) -> bool:
        """Return True if *position* (degrees) is within [min, max]."""
        return self.min <= position <= self.max

    def clip(self, position: float) -> float:
        """Clip *position* (degrees) to the [min, max] range."""
        return np.clip(position, self.min, self.max)


@dataclass(frozen=True)
class TelescopeLimits:
    """Mechanical limits for the telescope.

    Parameters
    ----------
    azimuth : AxisLimits
        Azimuth axis limits.
    elevation : AxisLimits
        Elevation axis limits.
    """

    azimuth: AxisLimits
    elevation: AxisLimits

    def is_position_valid(self, az: float, el: float) -> bool:
        """Check if an az/el position is within telescope limits.

        Parameters
        ----------
        az : float
            Azimuth in degrees.
        el : float
            Elevation in degrees.

        Returns
        -------
        bool
            True if both az and el are within limits.
        """
        return self.azimuth.is_in_range(az) and self.elevation.is_in_range(el)


@dataclass(frozen=True)
class SunAvoidanceConfig:
    """Sun avoidance configuration.

    Parameters
    ----------
    enabled : bool
        Whether sun avoidance is enabled.
    exclusion_radius : float
        Radius around Sun to exclude, in degrees.
    warning_radius : float
        Radius around Sun to warn about, in degrees.

    Raises
    ------
    ValueError
        When ``enabled`` and either ``exclusion_radius < 0`` or
        ``warning_radius < exclusion_radius`` (which would leave an empty
        warning band, or silently disable avoidance entirely).
    """

    enabled: bool
    exclusion_radius: float
    warning_radius: float

    def __post_init__(self) -> None:
        # Only meaningful when avoidance is on; disabled configs may carry
        # inert placeholder radii. Mirrors AxisLimits.__post_init__.
        if not self.enabled:
            return
        if self.exclusion_radius < 0:
            raise ValueError(
                f"exclusion_radius ({self.exclusion_radius}) must be >= 0 when enabled"
            )
        if self.warning_radius < self.exclusion_radius:
            raise ValueError(
                f"warning_radius ({self.warning_radius}) must be >= exclusion_radius "
                f"({self.exclusion_radius}) when enabled"
            )


_NASMYTH_SIGNS: dict[str, int] = {"right": 1, "left": -1, "cassegrain": 0}


@dataclass(frozen=True)
class Site:
    """Telescope site configuration.

    This class encapsulates all site-specific configuration including
    geographic location, atmospheric conditions, telescope limits,
    and default operational parameters.

    Parameters
    ----------
    name : str
        Site name.
    description : str
        Site description.
    latitude : float
        Latitude in degrees (negative for South).
    longitude : float
        Longitude in degrees (negative for West).
    elevation : float
        Elevation above sea level in meters.
    atmosphere : AtmosphericConditions or None
        Atmospheric conditions for refraction corrections. Always
        ``None`` when constructed by ``get_fyst_site()`` or loaded from
        config; atmosphere is never read from files. Construct an
        ``AtmosphericConditions`` instance with current weather data
        and pass it to ``Coordinates``,
        ``TrajectoryBuilder.with_atmosphere()``, or planning functions.
    telescope_limits : TelescopeLimits
        Telescope mechanical limits.
    sun_avoidance : SunAvoidanceConfig
        Sun avoidance configuration.
    nasmyth_port : str, optional
        Which Nasmyth port instruments are mounted on. Determines the sign
        of the elevation component in focal-plane rotation. One of "right"
        (+1), "left" (-1), or "cassegrain" (0). Default is "right".
    plate_scale : float
        Telescope plate scale in arcsec/mm. Used to convert focal-plane
        positions (mm) to angular offsets (arcsec).

    Examples
    --------
    >>> from fyst_trajectories.site import get_fyst_site
    >>> site = get_fyst_site()
    >>> print(site.name)
    FYST
    >>> print(site.location)  # Returns astropy EarthLocation
    """

    name: str
    description: str
    latitude: float
    longitude: float
    elevation: float
    atmosphere: AtmosphericConditions | None
    telescope_limits: TelescopeLimits
    sun_avoidance: SunAvoidanceConfig
    nasmyth_port: str = "right"
    plate_scale: float = 0.0

    def __post_init__(self) -> None:
        port = self.nasmyth_port.lower()
        if port not in _NASMYTH_SIGNS:
            raise ValueError(
                f"Unknown nasmyth_port '{self.nasmyth_port}'. "
                f"Must be one of: {', '.join(_NASMYTH_SIGNS.keys())}"
            )

    @property
    def nasmyth_sign(self) -> int:
        """Sign convention for Nasmyth field rotation.

        Returns +1 for Right Nasmyth, -1 for Left Nasmyth, and 0 for
        Cassegrain (no Nasmyth rotation). The nasmyth_port value is
        validated at construction time, so this property always succeeds.

        Returns
        -------
        int
            +1, -1, or 0 depending on the Nasmyth port.
        """
        return _NASMYTH_SIGNS[self.nasmyth_port.lower()]

    @functools.cached_property
    def location(self) -> EarthLocation:
        """Get the site location as an astropy EarthLocation.

        Returns
        -------
        EarthLocation
            The geographic location of the site.
        """
        return EarthLocation(
            lat=self.latitude * u.deg,
            lon=self.longitude * u.deg,
            height=self.elevation * u.m,
        )

    @classmethod
    def from_config(cls, config_path: str | Path) -> "Site":
        """Load site configuration from a YAML file.

        For the default FYST telescope, prefer ``get_fyst_site()`` which
        constructs a ``Site`` from hardcoded constants with no file I/O.
        Use this method for custom (non-FYST) sites or testing with
        alternative configurations.

        Parameters
        ----------
        config_path : str or Path
            Path to configuration file.

        Returns
        -------
        Site
            Site configuration loaded from the file.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        ValueError
            If the configuration file is invalid.

        Examples
        --------
        >>> site = Site.from_config("/path/to/custom.yaml")
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return cls._from_dict(config, config_name=str(config_path.name))

    @classmethod
    def _from_dict(cls, config: dict, config_name: str = "config") -> "Site":
        """Create Site from configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary loaded from YAML.
        config_name : str
            Name of the config file (used in warning messages).

        Returns
        -------
        Site
            Site instance.
        """
        if "site" not in config:
            raise ValueError(f"Config '{config_name}' missing required section 'site'.")
        site_config = config["site"]

        if "telescope" not in config:
            raise ValueError(f"Config '{config_name}' missing required section 'telescope'.")
        telescope_config = config["telescope"]

        if "azimuth" not in telescope_config:
            raise ValueError(
                f"Config '{config_name}' missing required section 'telescope.azimuth'."
            )
        if "elevation" not in telescope_config:
            raise ValueError(
                f"Config '{config_name}' missing required section 'telescope.elevation'."
            )
        az_limits = telescope_config["azimuth"]
        el_limits = telescope_config["elevation"]

        telescope_limits = TelescopeLimits(
            azimuth=AxisLimits(
                min=_get_required(az_limits, "min", "telescope.azimuth", config_name),
                max=_get_required(az_limits, "max", "telescope.azimuth", config_name),
                max_velocity=_get_required(
                    az_limits, "max_velocity", "telescope.azimuth", config_name
                ),
                max_acceleration=_get_required(
                    az_limits, "max_acceleration", "telescope.azimuth", config_name
                ),
            ),
            elevation=AxisLimits(
                min=_get_required(el_limits, "min", "telescope.elevation", config_name),
                max=_get_required(el_limits, "max", "telescope.elevation", config_name),
                max_velocity=_get_required(
                    el_limits, "max_velocity", "telescope.elevation", config_name
                ),
                max_acceleration=_get_required(
                    el_limits, "max_acceleration", "telescope.elevation", config_name
                ),
            ),
        )

        if "sun_avoidance" not in config:
            raise ValueError(f"Config '{config_name}' missing required section 'sun_avoidance'.")
        sun_config = config["sun_avoidance"]
        sun_avoidance = SunAvoidanceConfig(
            enabled=_get_required(sun_config, "enabled", "sun_avoidance", config_name),
            exclusion_radius=_get_required(
                sun_config, "exclusion_radius", "sun_avoidance", config_name
            ),
            warning_radius=_get_required(
                sun_config, "warning_radius", "sun_avoidance", config_name
            ),
        )

        if "location" not in site_config:
            raise ValueError(f"Config '{config_name}' missing required section 'site.location'.")
        loc = site_config["location"]

        plate_scale = _get_required(telescope_config, "plate_scale", "telescope", config_name)
        if plate_scale <= 0:
            raise ValueError(
                f"Config '{config_name}': telescope.plate_scale must be positive, got {plate_scale}"
            )

        return cls(
            name=_get_required(site_config, "name", "site", config_name),
            description=site_config.get("description", ""),
            latitude=_get_required(loc, "latitude", "site.location", config_name),
            longitude=_get_required(loc, "longitude", "site.location", config_name),
            elevation=_get_required(loc, "elevation", "site.location", config_name),
            atmosphere=None,
            telescope_limits=telescope_limits,
            sun_avoidance=sun_avoidance,
            nasmyth_port=telescope_config.get("nasmyth_port", "right"),
            plate_scale=plate_scale,
        )


def get_fyst_site(
    *,
    sun_exclusion_radius: float = FYST_SUN_EXCLUSION_RADIUS,
    sun_warning_radius: float = FYST_SUN_WARNING_RADIUS,
    sun_avoidance_enabled: bool = FYST_SUN_AVOIDANCE_ENABLED,
) -> Site:
    """Get the default FYST site configuration.

    Constructs a ``Site`` from the FYST physical constants defined in
    this module. Tier 3 parameters (sun avoidance) can be overridden
    via keyword arguments; Tier 1 and Tier 2 parameters (location, optics,
    mechanical limits) are fixed constants. Construct a custom ``Site``
    directly for non-FYST telescopes or testing.

    Parameters
    ----------
    sun_exclusion_radius : float, optional
        Sun exclusion radius in degrees. Default: ``FYST_SUN_EXCLUSION_RADIUS``.
    sun_warning_radius : float, optional
        Sun warning radius in degrees. Default: ``FYST_SUN_WARNING_RADIUS``.
    sun_avoidance_enabled : bool, optional
        Whether sun avoidance is enabled. Default: ``FYST_SUN_AVOIDANCE_ENABLED``.

    Returns
    -------
    Site
        FYST site configuration.

    Examples
    --------
    >>> from fyst_trajectories.site import get_fyst_site
    >>> site = get_fyst_site()
    >>> print(site.latitude)
    -22.985639

    Override sun avoidance for testing:

    >>> site = get_fyst_site(sun_avoidance_enabled=False)
    >>> site.sun_avoidance.enabled
    False
    """
    return Site(
        name="FYST",
        description="Fred Young Submillimeter Telescope on Cerro Chajnantor",
        latitude=FYST_LATITUDE,
        longitude=FYST_LONGITUDE,
        elevation=FYST_ELEVATION,
        atmosphere=None,
        telescope_limits=TelescopeLimits(
            azimuth=AxisLimits(
                min=FYST_AZ_MIN,
                max=FYST_AZ_MAX,
                max_velocity=FYST_AZ_MAX_VELOCITY,
                max_acceleration=FYST_AZ_MAX_ACCELERATION,
            ),
            elevation=AxisLimits(
                min=FYST_EL_MIN,
                max=FYST_EL_MAX,
                max_velocity=FYST_EL_MAX_VELOCITY,
                max_acceleration=FYST_EL_MAX_ACCELERATION,
            ),
        ),
        sun_avoidance=SunAvoidanceConfig(
            enabled=sun_avoidance_enabled,
            exclusion_radius=sun_exclusion_radius,
            warning_radius=sun_warning_radius,
        ),
        nasmyth_port=FYST_NASMYTH_PORT,
        plate_scale=FYST_PLATE_SCALE,
    )
