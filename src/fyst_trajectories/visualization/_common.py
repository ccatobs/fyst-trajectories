"""Helpers shared by the library-tier plot modules; matplotlib loads only on call."""

from collections.abc import Iterable, Mapping

from ..offsets import InstrumentOffset

#: Color of the PrimeCam module outlines and the boresight marker.
FOOTPRINT_COLOR = "#1f77b4"


def _unique_offsets(modules: Mapping[str, InstrumentOffset]) -> list[InstrumentOffset]:
    """Return each distinct offset object once, in mapping order.

    Alias keys (``"c"``/``"center"``) reference one offset, drawn once.

    Raises
    ------
    ValueError
        If ``modules`` is empty.
    """
    return [offset for _, offset in _unique_keyed_offsets(modules)]


def _unique_keyed_offsets(
    modules: Mapping[str, InstrumentOffset],
) -> list[tuple[str, InstrumentOffset]]:
    """Return each distinct offset object once with its first key, in mapping order.

    Alias keys (``"c"``/``"center"``) reference one offset, returned once
    under the key that comes first (``"c"``).

    Raises
    ------
    ValueError
        If ``modules`` is empty.
    """
    unique: list[tuple[str, InstrumentOffset]] = []
    for key, offset in modules.items():
        if not any(offset is seen for _, seen in unique):
            unique.append((key, offset))
    if not unique:
        raise ValueError("modules must not be empty")
    return unique


def _palette(reserved: Iterable[str], fallback: str) -> list[str]:
    """Return the rc property-cycle colors minus ``reserved`` (normalized hex).

    A marker or curve drawn from this palette can never masquerade as one of
    the reserved semantic colors (the Sun, the zone overlays, the footprint).
    """
    import matplotlib.colors as mcolors  # pylint: disable=import-outside-toplevel
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    banned = {mcolors.to_hex(c) for c in reserved}
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return [c for c in cycle if mcolors.to_hex(c) not in banned] or [fallback]
