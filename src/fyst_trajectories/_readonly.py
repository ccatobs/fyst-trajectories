"""A read-only ``dict`` for the parameter mappings that frozen value types carry.

The module imports nothing, so any module of the package can use it without
creating an import cycle.
"""


class ReadOnlyDict(dict):
    """A ``dict`` whose mutating methods raise ``TypeError``.

    It is a ``dict`` subclass rather than a :class:`types.MappingProxyType`,
    so ``pickle``, :func:`copy.copy`, :func:`copy.deepcopy`,
    :func:`dataclasses.asdict` and :func:`json.dumps` keep working on it and on
    the value types that hold it, and ``isinstance(value, dict)`` stays true.
    ``dict(mapping)`` returns an ordinary mutable copy.

    The guarantee is shallow: nested containers stay mutable, and calling
    ``dict.__setitem__`` directly bypasses it. It stops accidental edits of a
    mapping that several objects share.

    YAML dumpers represent only plain containers: ``yaml.safe_dump`` raises
    on this type, so convert it with ``dict(...)`` before dumping.
    """

    __slots__ = ()

    def _refusal(self) -> TypeError:
        return TypeError(f"{type(self).__name__} is read-only; copy it with dict() to edit")

    def __setitem__(self, key: object, value: object) -> None:
        raise self._refusal()

    def __delitem__(self, key: object) -> None:
        raise self._refusal()

    def __ior__(self, other: object) -> "ReadOnlyDict":
        raise self._refusal()

    def clear(self) -> None:
        """Refuse: the mapping is read-only."""
        raise self._refusal()

    def pop(self, *args: object) -> None:
        """Refuse: the mapping is read-only."""
        raise self._refusal()

    def popitem(self) -> None:
        """Refuse: the mapping is read-only."""
        raise self._refusal()

    def setdefault(self, *args: object) -> None:
        """Refuse: the mapping is read-only."""
        raise self._refusal()

    def update(self, *args: object, **kwargs: object) -> None:
        """Refuse: the mapping is read-only."""
        raise self._refusal()

    def __reduce__(self) -> tuple[type, tuple[dict]]:
        # The default reduction replays the items through __setitem__, which
        # raises, so pickle and copy rebuild from a plain dict instead.
        return (type(self), (dict(self),))
