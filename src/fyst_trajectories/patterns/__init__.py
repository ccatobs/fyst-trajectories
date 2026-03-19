"""Scan pattern implementations for telescope trajectories.

This package provides:
- Pattern base classes and protocols
- Registry for pattern discovery
- Individual pattern implementations
- TrajectoryBuilder for fluent construction

Examples
--------
Using the builder (recommended):

>>> from astropy.time import Time
>>> from fyst_trajectories import get_fyst_site
>>> from fyst_trajectories.patterns import TrajectoryBuilder, PongScanConfig
>>>
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

Using the registry directly:

>>> from fyst_trajectories.patterns import get_pattern, PongScanConfig
>>> PongPattern = get_pattern("pong")
>>> config = PongScanConfig(
...     timestep=0.1,
...     width=2.0,
...     height=2.0,
...     spacing=0.1,
...     velocity=0.5,
...     num_terms=4,
...     angle=0.0,
... )
>>> pattern = PongPattern(ra=180.0, dec=-30.0, config=config)
>>> trajectory = pattern.generate(site, duration=300.0, start_time=start_time)

Listing available patterns:

>>> from fyst_trajectories.patterns import list_patterns
>>> print(list_patterns())
['constant_el', 'daisy', 'linear', 'planet', 'pong', 'sidereal']
"""

# Registry - must be imported first before patterns
# Import patterns to trigger registration (order matters for dependencies)
from . import constant_el as constant_el  # noqa: F401  # pylint: disable=useless-import-alias
from . import daisy as daisy  # noqa: F401  # pylint: disable=useless-import-alias
from . import linear as linear  # noqa: F401  # pylint: disable=useless-import-alias
from . import planet as planet  # noqa: F401  # pylint: disable=useless-import-alias
from . import pong as pong  # noqa: F401  # pylint: disable=useless-import-alias
from . import sidereal as sidereal  # noqa: F401  # pylint: disable=useless-import-alias

# Base classes
from .base import AltAzPattern, CelestialPattern, ScanPattern, TrajectoryMetadata

# Builder - import last after all patterns are registered
from .builder import TrajectoryBuilder

# Configs - all pattern configuration classes
from .configs import (
    CONFIG_TO_PATTERN,
    ConstantElScanConfig,
    DaisyScanConfig,
    LinearMotionConfig,
    PlanetTrackConfig,
    PongScanConfig,
    ScanConfig,
    SiderealTrackConfig,
)

# Pattern classes (for direct use)
from .constant_el import ConstantElScanPattern
from .daisy import DaisyScanPattern
from .linear import LinearMotionPattern
from .planet import PlanetTrackPattern
from .pong import PongScanPattern
from .registry import get_pattern, list_patterns, register_pattern
from .sidereal import SiderealTrackPattern

__all__ = [
    # Registry
    "register_pattern",
    "get_pattern",
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
    "DaisyScanConfig",
    "SiderealTrackConfig",
    "PlanetTrackConfig",
    "LinearMotionConfig",
    "CONFIG_TO_PATTERN",
    # Pattern classes
    "ConstantElScanPattern",
    "LinearMotionPattern",
    "SiderealTrackPattern",
    "PlanetTrackPattern",
    "PongScanPattern",
    "DaisyScanPattern",
    # Builder
    "TrajectoryBuilder",
]
