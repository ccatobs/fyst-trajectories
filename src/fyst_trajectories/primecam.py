"""PrimeCam instrument configuration.

Pre-defined module offsets for the PrimeCam focal-plane instrument on FYST.
PrimeCam has one center module on the optical axis and six inner-ring modules
at 461.3 mm radius, spaced 60 degrees apart.

The module positions are converted from physical focal-plane coordinates (mm)
to angular offsets (arcminutes) using ``FYST_PLATE_SCALE`` from ``site.py``.

Examples
--------
Get a named module offset:

>>> from fyst_trajectories.primecam import get_primecam_offset
>>> offset = get_primecam_offset("i1")
>>> f"{offset.name}: dx={offset.dx:.1f}', dy={offset.dy:.1f}'"
"PrimeCam-I1: dx=0.0', dy=-106.8'"

List available modules:

>>> from fyst_trajectories.primecam import PRIMECAM_MODULES
>>> print(sorted(PRIMECAM_MODULES.keys()))
['c', 'center', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6']
"""

from collections.abc import Sequence

import numpy as np

from .offsets import InstrumentOffset
from .site import FYST_PLATE_SCALE

# Module positions use standard polar convention: x = r*cos(theta),
# y = r*sin(theta), where theta is measured counterclockwise from the
# +x axis. At zero field rotation, x is the cross-elevation direction
# and y is the elevation direction.
# UNVERIFIED: see "Pending instrument verification" in docs/index.rst
# (plate scale and inner ring radius). The on-sky angular offsets of every
# off-axis module scale linearly with both, so wrong values produce
# correlated astrometric biases across the inner-ring modules.

INNER_RING_RADIUS_MM = 461.3
"""Inner ring module distance from optical axis in millimeters."""


MODULE_FOV_RADIUS_DEG: float = 0.65
"""Per-module on-sky FOV radius in degrees.

Used by :func:`fyst_trajectories.planning.plan_source_ces` to build a circular
cover polygon when the caller passes a single ``InstrumentOffset``
(or a module name) instead of an explicit
:class:`~fyst_trajectories.planning.ArrayFootprint`.

The 0.65° value is the published Prime-Cam per-module field of view: each
module has up to a 1.3° **diameter** on sky (Vavagiakis et al. 2022,
"CCAT-prime: Design of the Mod-Cam receiver and 280 GHz MKID instrument
module", Proc. SPIE, arXiv:2208.05468), i.e. a 0.65° **radius**. This is an
upper bound across modules; the 850 GHz module's baseline optical design is
1.1° (arXiv:2208.10634), so 0.65° over-covers that module by design, pending an
as-built per-module measurement. The bare detector-wafer extent (~0.39°
*diameter* at the FYST plate scale) is only a lower bound; the optical FOV is
larger than the illuminated wafer, so 0.65° is the FOV figure to cover with,
not a padded-up wafer estimate.

Pass an explicit :class:`~fyst_trajectories.planning.ArrayFootprint`
to override.
"""


PRIMECAM_CENTER = InstrumentOffset(dx=0.0, dy=0.0, name="PrimeCam-Center")

# Inner ring modules: 6 positions at 60-degree intervals.
# theta is the angular position on the ring (counterclockwise from +x axis).

PRIMECAM_I1 = InstrumentOffset.from_focal_plane(
    x_mm=0.0,
    y_mm=-INNER_RING_RADIUS_MM,
    plate_scale=FYST_PLATE_SCALE,
    name="PrimeCam-I1",
)  # theta=-90 deg

PRIMECAM_I2 = InstrumentOffset.from_focal_plane(
    x_mm=INNER_RING_RADIUS_MM * np.cos(np.deg2rad(-30)),
    y_mm=INNER_RING_RADIUS_MM * np.sin(np.deg2rad(-30)),
    plate_scale=FYST_PLATE_SCALE,
    name="PrimeCam-I2",
)  # theta=-30 deg

PRIMECAM_I3 = InstrumentOffset.from_focal_plane(
    x_mm=INNER_RING_RADIUS_MM * np.cos(np.deg2rad(30)),
    y_mm=INNER_RING_RADIUS_MM * np.sin(np.deg2rad(30)),
    plate_scale=FYST_PLATE_SCALE,
    name="PrimeCam-I3",
)  # theta=30 deg

PRIMECAM_I4 = InstrumentOffset.from_focal_plane(
    x_mm=0.0,
    y_mm=INNER_RING_RADIUS_MM,
    plate_scale=FYST_PLATE_SCALE,
    name="PrimeCam-I4",
)  # theta=90 deg

PRIMECAM_I5 = InstrumentOffset.from_focal_plane(
    x_mm=INNER_RING_RADIUS_MM * np.cos(np.deg2rad(150)),
    y_mm=INNER_RING_RADIUS_MM * np.sin(np.deg2rad(150)),
    plate_scale=FYST_PLATE_SCALE,
    name="PrimeCam-I5",
)  # theta=150 deg

PRIMECAM_I6 = InstrumentOffset.from_focal_plane(
    x_mm=INNER_RING_RADIUS_MM * np.cos(np.deg2rad(-150)),
    y_mm=INNER_RING_RADIUS_MM * np.sin(np.deg2rad(-150)),
    plate_scale=FYST_PLATE_SCALE,
    name="PrimeCam-I6",
)  # theta=-150 deg

PRIMECAM_MODULES: dict[str, InstrumentOffset] = {
    "c": PRIMECAM_CENTER,
    "center": PRIMECAM_CENTER,
    "i1": PRIMECAM_I1,
    "i2": PRIMECAM_I2,
    "i3": PRIMECAM_I3,
    "i4": PRIMECAM_I4,
    "i5": PRIMECAM_I5,
    "i6": PRIMECAM_I6,
}
"""Dict mapping module names to InstrumentOffset instances."""


def get_primecam_offset(module_name: str) -> InstrumentOffset:
    """Get the offset for a PrimeCam module by name.

    Parameters
    ----------
    module_name : str
        Module name (e.g., "c", "center", "i1", "i2", ..., "i6").

    Returns
    -------
    InstrumentOffset
        The offset for the specified module.

    Raises
    ------
    KeyError
        If the module name is not recognized.

    Examples
    --------
    >>> offset = get_primecam_offset("i1")
    >>> f"{offset.dx:.1f}', {offset.dy:.1f}'"
    "0.0', -106.8'"
    """
    key = module_name.lower()
    if key not in PRIMECAM_MODULES:
        available = ", ".join(sorted(PRIMECAM_MODULES.keys()))
        raise KeyError(f"Unknown PrimeCam module '{module_name}'. Available: {available}")
    return PRIMECAM_MODULES[key]


def resolve_module_tag(tag: str | Sequence[str]) -> list[InstrumentOffset]:
    """Resolve an SO-style module tag into a list of module offsets.

    Expands a comma-separated tag of module names into the
    ``list[InstrumentOffset]`` accepted by
    :func:`fyst_trajectories.planning.plan_source_ces` /
    :func:`fyst_trajectories.planning.compute_source_ces_params` via their ``footprint``
    argument, which averages the module centres so the centroid of the selected
    modules lands on the source. Adds no geometry.

    Parameters
    ----------
    tag : str or sequence of str
        ``"i1,i2"`` (exact module names, comma-separated), a sequence of
        names (``["i1", "i2"]``), or ``"all"`` for every module
        (``c, i1..i6``). Case-insensitive; whitespace ignored. ``c`` and
        ``center`` are the same module and are de-duplicated.

    Returns
    -------
    list of InstrumentOffset
        One offset per distinct module, in input order.

    Raises
    ------
    KeyError
        If a token is not a recognised module (from :func:`get_primecam_offset`).
    ValueError
        If the tag resolves to no modules.
    TypeError
        If ``tag`` is not a str or sequence of str.

    Examples
    --------
    >>> [o.name for o in resolve_module_tag("i1,i2")]
    ['PrimeCam-I1', 'PrimeCam-I2']
    """
    if isinstance(tag, str):
        if tag.strip().lower() == "all":
            names = [k for k in PRIMECAM_MODULES if k != "center"]
        else:
            names = [n.strip() for n in tag.split(",") if n.strip()]
    elif isinstance(tag, Sequence) and not isinstance(tag, (bytes, bytearray)):
        names = [str(n).strip() for n in tag if str(n).strip()]
    else:
        raise TypeError(f"tag must be a str or sequence of str, got {type(tag).__name__}")

    offsets: list[InstrumentOffset] = []
    seen: set[str] = set()
    for name in names:
        offset = get_primecam_offset(name)
        if offset.name in seen:
            continue
        seen.add(offset.name)
        offsets.append(offset)

    if not offsets:
        raise ValueError(f"module tag {tag!r} resolved to no modules")
    return offsets


def resolve_offset(
    module: str | None = None,
    dx: float | None = None,
    dy: float | None = None,
    name: str = "custom",
) -> InstrumentOffset | None:
    """Resolve user input to an InstrumentOffset or None (boresight).

    Provides a single entry point for converting user-facing offset
    specifications into an InstrumentOffset. Handles three cases:

    1. Named PrimeCam module (e.g., "i1", "center") -> predefined offset
    2. Custom dx/dy values in arcminutes -> new InstrumentOffset
    3. Neither specified -> None (boresight pointing)

    Parameters
    ----------
    module : str, optional
        PrimeCam module name (e.g., "i1", "i3"). Looks up predefined offset.
    dx : float, optional
        Custom cross-elevation offset in arcminutes.
    dy : float, optional
        Custom elevation offset in arcminutes.
    name : str
        Label for custom offsets. Default "custom".

    Returns
    -------
    InstrumentOffset or None
        The resolved offset, or None for boresight.

    Raises
    ------
    ValueError
        If both `module` and `dx`/`dy` are specified.

    Examples
    --------
    Named module lookup:

    >>> offset = resolve_offset(module="i3")
    >>> f"{offset.dx:.1f}', {offset.dy:.1f}'"
    "92.5', 53.4'"

    Custom offset:

    >>> resolve_offset(dx=5.0, dy=3.0, name="I1-I6 midpoint")
    InstrumentOffset(dx=5.0', dy=3.0', name='I1-I6 midpoint')

    Boresight (no offset):

    >>> resolve_offset()
    """
    has_module = module is not None
    has_custom = dx is not None or dy is not None

    if has_module and has_custom:
        raise ValueError(
            "Cannot specify both 'module' and 'dx'/'dy'. "
            "Use module for PrimeCam offsets or dx/dy for custom offsets."
        )

    if has_module:
        return get_primecam_offset(module)

    if has_custom:
        return InstrumentOffset(
            dx=dx if dx is not None else 0.0,
            dy=dy if dy is not None else 0.0,
            name=name,
        )

    return None


def primecam_geometry_dict(
    radius_deg: float = MODULE_FOV_RADIUS_DEG,
    xi_offset_deg: float = 0.0,
    eta_offset_deg: float = 0.0,
) -> dict[str, dict[str, list[float] | float]]:
    """Build a scheduler geometry dict for the PrimeCam modules.

    Adapts :data:`PRIMECAM_MODULES` into the ``{name: {"center": [xi_deg,
    eta_deg], "radius": deg}}`` schema that a scheduler geometry model
    consumes. Each
    module center is the :class:`~fyst_trajectories.offsets.InstrumentOffset`
    cross-elevation/elevation offset (``xi = dx``, ``eta = dy``), converted from
    arcminutes to degrees.

    The duplicate ``"center"`` alias of ``"c"`` in :data:`PRIMECAM_MODULES` is
    dropped, so the result has one slot per physical module, ``"c"`` plus
    ``"i1"`` .. ``"i6"`` (seven entries). A duplicate would double the cover
    polygon when the consumer merges queried slots.

    Parameters
    ----------
    radius_deg : float, optional
        Per-module on-sky FOV radius in degrees, applied to every slot.
        Defaults to :data:`MODULE_FOV_RADIUS_DEG` (0.65).
    xi_offset_deg, eta_offset_deg : float, optional
        Global boresight offsets in degrees added to every module center.
        Default 0.0.

    Returns
    -------
    dict
        ``{name: {"center": [xi_deg, eta_deg], "radius": radius_deg}}`` with
        seven entries (``"c"``, ``"i1"`` .. ``"i6"``).

    Notes
    -----
    Centers and radius are in **degrees**. The
    ``xi``/``eta`` axes match
    :class:`~fyst_trajectories.offsets.InstrumentOffset` ``dx``/``dy``
    (cross-elevation / elevation).

    Examples
    --------
    >>> geom = primecam_geometry_dict()
    >>> sorted(geom)
    ['c', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6']
    >>> geom["c"]["center"]
    [0.0, 0.0]
    """
    return {
        name: {
            "center": [offset.dx_deg + xi_offset_deg, offset.dy_deg + eta_offset_deg],
            "radius": radius_deg,
        }
        for name, offset in PRIMECAM_MODULES.items()
        if name != "center"
    }
