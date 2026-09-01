"""Scan pattern implementations for telescope trajectories.

This package provides pattern base classes, individual pattern
implementations, a name-based pattern registry, and the
:class:`TrajectoryBuilder` fluent API for assembling trajectories.

Examples
--------
Using the builder (recommended):

>>> from astropy.time import Time
>>> from fyst_trajectories import get_fyst_site
>>> from fyst_trajectories.patterns import TrajectoryBuilder, PongScanConfig
>>>
>>> site = get_fyst_site()
>>> start_time = Time("2026-03-15T01:00:00", scale="utc")
>>> trajectory = (
...     TrajectoryBuilder(site)
...     .at(ra=180.0, dec=-30.0)
...     .with_config(
...         PongScanConfig(
...             timestep=0.1,
...             width=2.0,
...             height=2.0,
...             spacing=0.1,
...             velocity=0.4,
...             num_terms=4,
...             angle=0.0,
...         )
...     )
...     .duration(300.0)
...     .starting_at(start_time)
...     .build()
... )

Using the registry directly:

>>> from fyst_trajectories.patterns import get_pattern, PongScanConfig
>>> PongPattern = get_pattern("pong")
>>> config = PongScanConfig(
...     timestep=0.1,
...     width=2.0,
...     height=2.0,
...     spacing=0.1,
...     velocity=0.4,
...     num_terms=4,
...     angle=0.0,
... )
>>> pattern = PongPattern(ra=180.0, dec=-30.0, config=config)
>>> trajectory = pattern.generate(site, duration=300.0, start_time=start_time)

Listing available patterns:

>>> from fyst_trajectories.patterns import list_patterns
>>> list_patterns()  # doctest: +NORMALIZE_WHITESPACE
['constant_el', 'daisy', 'daisy_altaz', 'linear', 'planet', 'pong',
 'pong_altaz', 'satellite', 'sidereal']
"""

# Import patterns to trigger registration (order matters)
from . import constant_el as constant_el  # noqa: F401  # pylint: disable=useless-import-alias
from . import daisy as daisy  # noqa: F401  # pylint: disable=useless-import-alias
from . import daisy_altaz as daisy_altaz  # noqa: F401  # pylint: disable=useless-import-alias
from . import linear as linear  # noqa: F401  # pylint: disable=useless-import-alias
from . import planet as planet  # noqa: F401  # pylint: disable=useless-import-alias
from . import pong as pong  # noqa: F401  # pylint: disable=useless-import-alias
from . import pong_altaz as pong_altaz  # noqa: F401  # pylint: disable=useless-import-alias
from . import satellite as satellite  # noqa: F401  # pylint: disable=useless-import-alias
from . import sidereal as sidereal  # noqa: F401  # pylint: disable=useless-import-alias
from .base import AltAzPattern, CelestialPattern, ScanPattern, TrajectoryMetadata
from .builder import TrajectoryBuilder
from .configs import (
    ConstantElScanConfig,
    DaisyAltAzScanConfig,
    DaisyScanConfig,
    LinearMotionConfig,
    PlanetTrackConfig,
    PongAltAzScanConfig,
    PongScanConfig,
    SatelliteTrackConfig,
    ScanConfig,
    SiderealTrackConfig,
)
from .constant_el import ConstantElScanPattern
from .daisy import DaisyScanPattern
from .daisy_altaz import DaisyAltAzScanPattern
from .linear import LinearMotionPattern
from .planet import PlanetTrackPattern
from .pong import PongScanPattern, compute_pong_period
from .pong_altaz import PongAltAzScanPattern
from .registry import (
    get_pattern,
    get_pattern_for_config,
    list_patterns,
    register_pattern,
)
from .satellite import SatelliteTrackPattern
from .sidereal import SiderealTrackPattern

__all__ = [
    # Registry
    "register_pattern",
    "get_pattern",
    "get_pattern_for_config",
    "list_patterns",
    # Base classes
    "ScanPattern",
    "CelestialPattern",
    "AltAzPattern",
    "TrajectoryMetadata",
    # Configs
    "ScanConfig",
    "ConstantElScanConfig",
    "PongScanConfig",
    "PongAltAzScanConfig",
    "DaisyScanConfig",
    "DaisyAltAzScanConfig",
    "SiderealTrackConfig",
    "PlanetTrackConfig",
    "SatelliteTrackConfig",
    "LinearMotionConfig",
    # Pattern classes
    "ConstantElScanPattern",
    "LinearMotionPattern",
    "SiderealTrackPattern",
    "PlanetTrackPattern",
    "SatelliteTrackPattern",
    "PongScanPattern",
    "PongAltAzScanPattern",
    "DaisyScanPattern",
    "DaisyAltAzScanPattern",
    # Builder
    "TrajectoryBuilder",
    # Helpers
    "compute_pong_period",
]
