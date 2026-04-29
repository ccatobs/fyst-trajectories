"""FYST Pointing Library.

A FYST-specific astronomy utilities library that wraps astropy with
telescope-specific defaults, conventions, and safety checks.

Instead of every project manually setting up astropy with FYST's
coordinates and re-implementing the same transformations, this library
provides a pre-configured, FYST-aware toolkit.

Examples
--------
Coordinate transformation (vacuum -- the ACU applies refraction):

>>> from astropy.time import Time
>>> from fyst_trajectories import Coordinates, get_fyst_site
>>> coords = Coordinates(get_fyst_site())
>>> obstime = Time("2026-01-15T02:00:00", scale="utc")
>>> az, el = coords.radec_to_altaz(83.633, 22.014, obstime=obstime)  # Crab Nebula

Generate a Pong scan using the builder:

>>> from astropy.time import Time
>>> from fyst_trajectories import get_fyst_site
>>> from fyst_trajectories.patterns import TrajectoryBuilder, PongScanConfig
>>> site = get_fyst_site()
>>> start_time = Time("2026-03-15T04:00:00", scale="utc")
>>> trajectory = (
...     TrajectoryBuilder(site)
...     .at(ra=180.0, dec=-30.0)
...     .with_config(
...         PongScanConfig(
...             timestep=0.1,
...             width=2.0,
...             height=2.0,
...             spacing=0.1,
...             velocity=0.5,
...             num_terms=4,
...             angle=0.0,
...         )
...     )
...     .duration(300.0)
...     .starting_at(start_time)
...     .build()
... )

Planning with refraction (visibility checks, not sent to ACU):

>>> from fyst_trajectories import AtmosphericConditions, Coordinates, get_fyst_site
>>> coords = Coordinates(get_fyst_site(), atmosphere=AtmosphericConditions.for_fyst())
>>> obstime = Time("2026-03-15T16:00:00", scale="utc")
>>> az, el = coords.get_body_altaz("mars", obstime)
"""

__version__ = "0.4.0"

from .coordinates import (
    FRAME_ALIASES,
    SOLAR_SYSTEM_BODIES,
    Coordinates,
    normalize_frame,
)
from .exceptions import (
    AzimuthBoundsError,
    ElevationBoundsError,
    PointingError,
    PointingWarning,
    TargetNotObservableError,
    TrajectoryBoundsError,
)
from .offsets import (
    InstrumentOffset,
    apply_detector_offset,
    boresight_to_detector,
    compute_focal_plane_rotation,
    detector_to_boresight,
)
from .patterns import (
    AltAzPattern,
    CelestialPattern,
    ConstantElScanConfig,
    ConstantElScanPattern,
    DaisyScanConfig,
    DaisyScanPattern,
    LinearMotionConfig,
    LinearMotionPattern,
    PlanetTrackConfig,
    PlanetTrackPattern,
    PongScanConfig,
    PongScanPattern,
    ScanConfig,
    ScanPattern,
    SiderealTrackConfig,
    SiderealTrackPattern,
    TrajectoryBuilder,
    TrajectoryMetadata,
    compute_pong_period,
    get_pattern,
    get_pattern_for_config,
    list_patterns,
    register_pattern,
)
from .planning import (
    ComputedParams,
    ConstantElComputedParams,
    DaisyComputedParams,
    FieldRegion,
    PongComputedParams,
    ScanBlock,
    plan_constant_el_scan,
    plan_daisy_scan,
    plan_pong_rotation_sequence,
    plan_pong_scan,
    validate_computed_params,
)
from .plotting import plot_hit_map
from .primecam import (
    PRIMECAM_CENTER,
    PRIMECAM_I1,
    PRIMECAM_I2,
    PRIMECAM_I3,
    PRIMECAM_I4,
    PRIMECAM_I5,
    PRIMECAM_I6,
    PRIMECAM_MODULES,
    get_primecam_offset,
    resolve_offset,
)
from .site import (
    FYST_AZ_MAX,
    FYST_AZ_MAX_ACCELERATION,
    FYST_AZ_MAX_VELOCITY,
    FYST_AZ_MIN,
    FYST_EL_MAX,
    FYST_EL_MAX_ACCELERATION,
    FYST_EL_MAX_VELOCITY,
    FYST_EL_MIN,
    FYST_ELEVATION,
    FYST_LATITUDE,
    FYST_LONGITUDE,
    FYST_NASMYTH_PORT,
    FYST_PLATE_SCALE,
    FYST_SUN_AVOIDANCE_ENABLED,
    FYST_SUN_EXCLUSION_RADIUS,
    FYST_SUN_WARNING_RADIUS,
    AtmosphericConditions,
    AxisLimits,
    Site,
    SunAvoidanceConfig,
    TelescopeLimits,
    get_fyst_site,
)
from .trajectory import (
    SCAN_FLAG_RETUNE,
    SCAN_FLAG_SCIENCE,
    SCAN_FLAG_TURNAROUND,
    SCAN_FLAG_UNCLASSIFIED,
    RetuneEvent,
    Trajectory,
)
from .trajectory_utils import (
    get_absolute_times,
    inject_retune,
    plot_trajectory,
    print_trajectory,
    sample_retune_events,
    to_arrays,
    to_path_format,
    validate_sun_avoidance,
    validate_trajectory,
    validate_trajectory_bounds,
    validate_trajectory_dynamics,
)

FYST_LOCATION = get_fyst_site().location
"""FYST location as astropy EarthLocation (convenience constant)."""

__all__ = [
    # Version
    "__version__",
    # Exceptions and warnings
    "PointingError",
    "PointingWarning",
    "TrajectoryBoundsError",
    "AzimuthBoundsError",
    "ElevationBoundsError",
    "TargetNotObservableError",
    # Site configuration
    "Site",
    "AtmosphericConditions",
    "AxisLimits",
    "TelescopeLimits",
    "SunAvoidanceConfig",
    "get_fyst_site",
    "FYST_LOCATION",
    # FYST physical constants
    "FYST_LATITUDE",
    "FYST_LONGITUDE",
    "FYST_ELEVATION",
    "FYST_PLATE_SCALE",
    "FYST_NASMYTH_PORT",
    "FYST_AZ_MIN",
    "FYST_AZ_MAX",
    "FYST_AZ_MAX_VELOCITY",
    "FYST_AZ_MAX_ACCELERATION",
    "FYST_EL_MIN",
    "FYST_EL_MAX",
    "FYST_EL_MAX_VELOCITY",
    "FYST_EL_MAX_ACCELERATION",
    "FYST_SUN_EXCLUSION_RADIUS",
    "FYST_SUN_WARNING_RADIUS",
    "FYST_SUN_AVOIDANCE_ENABLED",
    # Coordinates
    "Coordinates",
    "SOLAR_SYSTEM_BODIES",
    "FRAME_ALIASES",
    "normalize_frame",
    # Trajectory
    "Trajectory",
    "SCAN_FLAG_UNCLASSIFIED",
    "SCAN_FLAG_SCIENCE",
    "SCAN_FLAG_TURNAROUND",
    "SCAN_FLAG_RETUNE",
    "print_trajectory",
    "inject_retune",
    "RetuneEvent",
    "sample_retune_events",
    "validate_sun_avoidance",
    "validate_trajectory",
    "validate_trajectory_bounds",
    "validate_trajectory_dynamics",
    "get_absolute_times",
    "to_arrays",
    "to_path_format",
    "plot_trajectory",
    "plot_hit_map",
    # Pattern registry
    "register_pattern",
    "get_pattern",
    "get_pattern_for_config",
    "list_patterns",
    # Pattern base classes
    "ScanPattern",
    "CelestialPattern",
    "AltAzPattern",
    "TrajectoryMetadata",
    # Pattern configs
    "ScanConfig",
    "ConstantElScanConfig",
    "PongScanConfig",
    "DaisyScanConfig",
    "SiderealTrackConfig",
    "PlanetTrackConfig",
    "LinearMotionConfig",
    # Pattern classes
    "ConstantElScanPattern",
    "LinearMotionPattern",
    "SiderealTrackPattern",
    "PlanetTrackPattern",
    "PongScanPattern",
    "DaisyScanPattern",
    # Builder
    "TrajectoryBuilder",
    # Pattern helpers
    "compute_pong_period",
    # Planning
    "ComputedParams",
    "ConstantElComputedParams",
    "DaisyComputedParams",
    "FieldRegion",
    "PongComputedParams",
    "ScanBlock",
    "plan_pong_scan",
    "plan_pong_rotation_sequence",
    "plan_constant_el_scan",
    "plan_daisy_scan",
    "validate_computed_params",
    # Instrument offsets
    "InstrumentOffset",
    "boresight_to_detector",
    "detector_to_boresight",
    "apply_detector_offset",
    "compute_focal_plane_rotation",
    "get_primecam_offset",
    "resolve_offset",
    "PRIMECAM_CENTER",
    "PRIMECAM_I1",
    "PRIMECAM_I2",
    "PRIMECAM_I3",
    "PRIMECAM_I4",
    "PRIMECAM_I5",
    "PRIMECAM_I6",
    "PRIMECAM_MODULES",
]
