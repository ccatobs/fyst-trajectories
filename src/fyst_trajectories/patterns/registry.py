"""Pattern registry for scan pattern discovery and instantiation.

The registry allows patterns to be:
1. Registered by name using a decorator
2. Retrieved by name for instantiation
3. Listed to show available patterns

Examples
--------
Register a pattern:

>>> @register_pattern("pong")
... class PongPattern(CelestialPattern):
...     pass

Retrieve a pattern:

>>> pattern_cls = get_pattern("pong")
>>> pattern = pattern_cls(ra=180.0, dec=-30.0, config=config)

List available patterns:

>>> print(list_patterns())
['constant_el', 'daisy', 'linear', 'planet', 'pong', 'sidereal']
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import ScanPattern

_PATTERN_REGISTRY: dict[str, type["ScanPattern"]] = {}


def register_pattern(name: str):
    """Register a pattern class via decorator.

    Parameters
    ----------
    name : str
        Unique identifier for the pattern (e.g., "pong", "daisy").

    Returns
    -------
    callable
        Decorator that registers the class and returns it unchanged.

    Raises
    ------
    ValueError
        If a pattern with the same name is already registered.

    Examples
    --------
    >>> @register_pattern("my_pattern")
    ... class MyPattern(CelestialPattern):
    ...     pass
    """

    def decorator(cls: type["ScanPattern"]) -> type["ScanPattern"]:
        if name in _PATTERN_REGISTRY:
            raise ValueError(
                f"Pattern '{name}' already registered by {_PATTERN_REGISTRY[name].__name__}"
            )
        _PATTERN_REGISTRY[name] = cls
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
    >>> PongPattern = get_pattern("pong")
    >>> pattern = PongPattern(ra=180.0, dec=-30.0, config=config)
    """
    if name not in _PATTERN_REGISTRY:
        available = ", ".join(sorted(_PATTERN_REGISTRY.keys()))
        raise KeyError(f"Unknown pattern '{name}'. Available: {available}")
    return _PATTERN_REGISTRY[name]


def list_patterns() -> list[str]:
    """List all registered pattern names.

    Returns
    -------
    list[str]
        Sorted list of pattern names.

    Examples
    --------
    >>> print(list_patterns())
    ['constant_el', 'daisy', 'linear', 'planet', 'pong', 'sidereal']
    """
    return sorted(_PATTERN_REGISTRY.keys())
