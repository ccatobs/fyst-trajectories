"""Helpers shared by the library-tier plot modules; matplotlib loads only on call."""

import importlib.util
from collections.abc import Iterable, Mapping
from datetime import tzinfo
from typing import TYPE_CHECKING

from ..offsets import InstrumentOffset

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: Color of the Prime-Cam module outlines and the boresight marker.
FOOTPRINT_COLOR = "#1f77b4"
#: Color of the Sun's own curve and marker (excluded from the target palettes).
SUN_COLOR = "#e6a817"
#: Overlay color for sky or samples inside the exclusion radius.
EXCLUSION_COLOR = "#d62728"
#: Overlay color for sky or samples inside the warning radius.
WARNING_COLOR = "#ff7f0e"


def _tz_label(tz: tzinfo | None) -> str:
    """Human-readable axis label for a timezone (``"UTC"`` when None)."""
    return str(tz) if tz is not None else "UTC"


def _require_matplotlib(caller: str) -> None:
    """Raise the install hint for ``caller`` when matplotlib is not installed."""
    if importlib.util.find_spec("matplotlib") is None:
        raise ImportError(
            f"matplotlib is required for {caller}(). "
            "Install it with: pip install 'fyst-trajectories[plotting]'"
        )


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


def _module_label(offset: InstrumentOffset) -> str:
    """Return the short label drawn beside a module circle (``"c"``, ``"i1"``)."""
    return (offset.name or "").removeprefix("PrimeCam-").lower() or "?"


def _draw_module_circles(
    ax: "Axes",
    placements: Iterable[tuple[InstrumentOffset, float, float]],
    fov_radius_deg: float,
    *,
    labels: bool,
    face_alpha: float,
    lw: float,
    label_alpha: float | None = None,
) -> None:
    """Draw each module's FOV circle at ``(dx, dy)`` and, optionally, its label below it."""
    from matplotlib.patches import Circle  # pylint: disable=import-outside-toplevel

    for offset, dx, dy in placements:
        ax.add_patch(
            Circle(
                (dx, dy),
                fov_radius_deg,
                facecolor=FOOTPRINT_COLOR,
                alpha=face_alpha,
                edgecolor=FOOTPRINT_COLOR,
                lw=lw,
            )
        )
        if labels:
            ax.annotate(
                _module_label(offset),
                xy=(dx, dy),
                xytext=(0, -14),  # below the centre, clear of the boresight marker
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=9,
                alpha=label_alpha,
            )
