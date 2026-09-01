"""Pattern registry for scan pattern discovery and instantiation.

The registry allows patterns to be:
1. Registered by name (and optional config class) using a decorator
2. Retrieved by name for instantiation
3. Looked up by config class via :func:`get_pattern_for_config`
4. Listed to show available patterns

Examples
--------
Register a pattern with its config class so that the builder can infer
the pattern type (shown with a new name; each name and config class may
only be registered once):

>>> @register_pattern("my_pattern", config=MyPatternConfig)  # doctest: +SKIP
... class MyPattern(CelestialPattern):
...     pass

Retrieve a pattern:

>>> from fyst_trajectories.patterns import PongScanConfig
>>> config = PongScanConfig(
...     timestep=0.1,
...     width=2.0,
...     height=2.0,
...     spacing=0.1,
...     velocity=0.4,
...     num_terms=4,
...     angle=0.0,
... )
>>> pattern_cls = get_pattern("pong")
>>> pattern = pattern_cls(ra=180.0, dec=-30.0, config=config)

List available patterns:

>>> print(list_patterns())  # doctest: +NORMALIZE_WHITESPACE
['constant_el', 'daisy', 'daisy_altaz', 'linear', 'planet', 'pong',
 'pong_altaz', 'satellite', 'sidereal']
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import ScanPattern
    from .configs import ScanConfig

# Scan-type vocabulary (one of six; see the "Scan-type vocabularies" section
# in docs/overhead_integration.rst). Surfaced by ``list_patterns``, this is the
# widest scan-name vocabulary: the 9 buildable scan patterns. ``source_ces`` is
# planner-only (``plan_source_ces``), NOT a registered pattern, so it never
# appears here; the narrower planning-side (``ComputedParams`` /
# ``_SCAN_TYPE_TO_KEYS``) and overhead-side (``ScanParamsDict`` /
# ``_SCAN_TYPE_TO_SCAN_PARAM_KEYS``) vocabularies are differently scoped.
_PATTERN_REGISTRY: dict[str, type["ScanPattern"]] = {}
_CONFIG_TO_PATTERN_NAME: dict[type, str] = {}


def register_pattern(name: str, *, config: type["ScanConfig"] | None = None):
    """Register a pattern class via decorator.

    Parameters
    ----------
    name : str
        Unique identifier for the pattern (e.g., "pong", "daisy").
    config : type, optional
        The :class:`ScanConfig` subclass that configures this pattern.
        When supplied, the decorator also adds a mapping from the config
        class to the pattern name so :class:`TrajectoryBuilder` can
        infer the pattern type from the config. Making this a decorator
        argument means adding a new pattern is a single-location change.

    Returns
    -------
    callable
        Decorator that registers the class and returns it unchanged.

    Raises
    ------
    ValueError
        If a pattern with the same name is already registered, or if
        the config class is already mapped to another pattern.

    Examples
    --------
    >>> @register_pattern("my_pattern", config=MyPatternConfig)  # doctest: +SKIP
    ... class MyPattern(CelestialPattern):
    ...     pass
    """

    def decorator(cls: type["ScanPattern"]) -> type["ScanPattern"]:
        if name in _PATTERN_REGISTRY:
            raise ValueError(
                f"Pattern '{name}' already registered by {_PATTERN_REGISTRY[name].__name__}"
            )
        _PATTERN_REGISTRY[name] = cls
        if config is not None:
            if config in _CONFIG_TO_PATTERN_NAME:
                existing = _CONFIG_TO_PATTERN_NAME[config]
                raise ValueError(f"Config {config.__name__} already mapped to pattern '{existing}'")
            _CONFIG_TO_PATTERN_NAME[config] = name
        return cls

    return decorator


def get_pattern(name: str) -> type["ScanPattern"]:
    """Get a pattern class by name.

    Parameters
    ----------
    name : str
        Pattern identifier.

    Returns
    -------
    Type[ScanPattern]
        The pattern class.

    Raises
    ------
    KeyError
        If no pattern with that name is registered.

    Examples
    --------
    >>> from fyst_trajectories.patterns import PongScanConfig
    >>> config = PongScanConfig(
    ...     timestep=0.1,
    ...     width=2.0,
    ...     height=2.0,
    ...     spacing=0.1,
    ...     velocity=0.4,
    ...     num_terms=4,
    ...     angle=0.0,
    ... )
    >>> PongPattern = get_pattern("pong")
    >>> pattern = PongPattern(ra=180.0, dec=-30.0, config=config)
    """
    if name not in _PATTERN_REGISTRY:
        available = ", ".join(sorted(_PATTERN_REGISTRY.keys()))
        raise KeyError(f"Unknown pattern '{name}'. Available: {available}")
    return _PATTERN_REGISTRY[name]


def get_pattern_for_config(config_cls: type) -> str:
    """Get the pattern name associated with a :class:`ScanConfig` subclass.

    Parameters
    ----------
    config_cls : type
        A :class:`ScanConfig` subclass.

    Returns
    -------
    str
        The registered pattern name.

    Raises
    ------
    KeyError
        If no pattern has registered that config class.
    """
    if config_cls not in _CONFIG_TO_PATTERN_NAME:
        available = ", ".join(sorted(c.__name__ for c in _CONFIG_TO_PATTERN_NAME))
        raise KeyError(f"Unknown config type: {config_cls.__name__}. Expected one of: {available}")
    return _CONFIG_TO_PATTERN_NAME[config_cls]


def list_patterns() -> list[str]:
    """List all registered pattern names.

    Returns
    -------
    list[str]
        Sorted list of pattern names.

    Examples
    --------
    >>> print(list_patterns())  # doctest: +NORMALIZE_WHITESPACE
    ['constant_el', 'daisy', 'daisy_altaz', 'linear', 'planet', 'pong',
     'pong_altaz', 'satellite', 'sidereal']
    """
    return sorted(_PATTERN_REGISTRY.keys())
