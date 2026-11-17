"""Prime-Cam instrument configuration.

Pre-defined module offsets for the Prime-Cam focal-plane instrument on FYST.
Prime-Cam has seven module bays: ``IM0`` on the optical axis and ``IM1`` ..
``IM6`` on an inner ring of 461.3 mm radius, spaced 60 degrees apart.

The names ``IM0`` .. ``IM6`` are the Prime-Cam instrument team's
designations (Keller et al. 2026, "CCAT: Design and Characterization of the
350 GHz Instrument Module", arXiv:2608.05121, Fig. 1). They label bay
positions, not the instrument modules that occupy them. Lookup ignores case,
and ``c`` and ``center`` are accepted wherever a module name is, as
spellings of ``IM0``.

The bay angles below are seen looking out at the sky, with elevation up and
azimuth increasing to the right, and are measured counter-clockwise from
+cross-elevation (``+dx``, toward increasing azimuth) at boresight
elevation 0. Under the default rotation sign (``nasmyth_port="right"``, see
:attr:`~fyst_trajectories.site.Site.nasmyth_sign`) the focal plane turns
counter-clockwise on the sky by the boresight elevation, so a bay sits at
its angle plus the boresight elevation. ``IM1`` to ``IM6`` run clockwise in
this view.

=====  ===========
Bay    Angle (deg)
=====  ===========
IM0    on axis
IM1    180
IM2    120
IM3    60
IM4    0
IM5    -60
IM6    -120
=====  ===========

The positions follow the instrument team's optical model of the receiver;
an on-sky check is still to come (see the pending-verification list on the
documentation index).

The module positions are converted from physical coordinates (mm) on the
telescope focal surface, where the module bays sit, to angular offsets
(arcminutes) using ``FYST_PLATE_SCALE`` from ``site.py``, the plate scale of
that surface (a module's detector plane has its own, larger scale, which the
library does not carry).

The ring radius and the plate scale used to convert it are design values
awaiting as-built instrument confirmation; see the pending-verification list
on the documentation index.

Examples
--------
Get a named module offset:

>>> from fyst_trajectories.primecam import get_primecam_offset
>>> offset = get_primecam_offset("IM6")
>>> f"{offset.name}: dx={offset.dx:.1f}', dy={offset.dy:.1f}'"
"PrimeCam-IM6: dx=-53.4', dy=-92.5'"

List available modules:

>>> from fyst_trajectories.primecam import PRIMECAM_MODULES
>>> print(list(PRIMECAM_MODULES))
['IM0', 'IM1', 'IM2', 'IM3', 'IM4', 'IM5', 'IM6']
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType

import numpy as np

from ._validation import _require_finite, _require_positive
from .offsets import InstrumentOffset
from .site import FYST_PLATE_SCALE

# Bay positions on the telescope focal surface at boresight elevation 0, seen
# looking out at the sky: x is the cross-elevation direction (toward
# increasing azimuth), y the elevation direction, and a bay at angle theta
# (counter-clockwise from +x) sits at x = R*cos(theta), y = R*sin(theta).
# The coordinates are written from R, R/2 and R*sqrt(3)/2 rather than from
# cos and sin, so IM1 and IM4 lie exactly on the x axis and every mirror pair
# is an exact negation.
# Awaiting confirmation: see "Pending instrument verification" in
# docs/index.rst (plate scale, inner ring radius, and the on-sky check of the
# bay positions). The on-sky angular offsets of every off-axis bay scale
# linearly with the first two, so wrong values produce correlated astrometric
# biases across the inner ring. instrument_rotation=0 further assumes the
# cryostat is not clocked relative to the Nasmyth flange.

INNER_RING_RADIUS_MM = 461.3
"""Inner ring module distance from the optical axis in millimeters, on the
telescope focal surface."""

_RING_HALF_MM = INNER_RING_RADIUS_MM / 2.0
_RING_SQRT3_HALF_MM = INNER_RING_RADIUS_MM * np.sqrt(3.0) / 2.0


MODULE_FOV_RADIUS_DEG: float = 0.65
"""Per-module on-sky FOV radius in degrees.

Used by :func:`fyst_trajectories.planning.plan_source_ces` to build a circular
cover polygon when the caller passes a single ``InstrumentOffset``
(or a module name) instead of an explicit
:class:`~fyst_trajectories.planning.ArrayFootprint`.

The 0.65 deg value is the published Prime-Cam per-module field of view: each
module has up to a 1.3 deg **diameter** on sky (Vavagiakis et al. 2022,
"CCAT-prime: Design of the Mod-Cam receiver and 280 GHz MKID instrument
module", Proc. SPIE, arXiv:2208.05468), i.e. a 0.65 deg **radius**. This is an
upper bound across modules; one module's candidate design gives it a 1.1 deg
**diameter** (Chapman et al. 2022, "CCAT-prime: The 850 GHz camera for
Prime-Cam on FYST", Proc. SPIE, arXiv:2208.10634), so 0.65 deg over-covers it by
design, pending an as-built per-module measurement. The illuminated detector wafer is smaller
than the optical field, so 0.65 deg is the field-of-view figure to cover with,
not a padded-up wafer estimate.

Pass an explicit :class:`~fyst_trajectories.planning.ArrayFootprint`
to override.
"""


PRIMECAM_IM0 = InstrumentOffset.from_focal_plane(
    x_mm=0.0,
    y_mm=0.0,
    plate_scale=FYST_PLATE_SCALE,
    name="PrimeCam-IM0",
)  # on the optical axis

PRIMECAM_IM1 = InstrumentOffset.from_focal_plane(
    x_mm=-INNER_RING_RADIUS_MM,
    y_mm=0.0,
    plate_scale=FYST_PLATE_SCALE,
    name="PrimeCam-IM1",
)  # theta=180 deg

PRIMECAM_IM2 = InstrumentOffset.from_focal_plane(
    x_mm=-_RING_HALF_MM,
    y_mm=_RING_SQRT3_HALF_MM,
    plate_scale=FYST_PLATE_SCALE,
    name="PrimeCam-IM2",
)  # theta=120 deg

PRIMECAM_IM3 = InstrumentOffset.from_focal_plane(
    x_mm=_RING_HALF_MM,
    y_mm=_RING_SQRT3_HALF_MM,
    plate_scale=FYST_PLATE_SCALE,
    name="PrimeCam-IM3",
)  # theta=60 deg

PRIMECAM_IM4 = InstrumentOffset.from_focal_plane(
    x_mm=INNER_RING_RADIUS_MM,
    y_mm=0.0,
    plate_scale=FYST_PLATE_SCALE,
    name="PrimeCam-IM4",
)  # theta=0 deg

PRIMECAM_IM5 = InstrumentOffset.from_focal_plane(
    x_mm=_RING_HALF_MM,
    y_mm=-_RING_SQRT3_HALF_MM,
    plate_scale=FYST_PLATE_SCALE,
    name="PrimeCam-IM5",
)  # theta=-60 deg

PRIMECAM_IM6 = InstrumentOffset.from_focal_plane(
    x_mm=-_RING_HALF_MM,
    y_mm=-_RING_SQRT3_HALF_MM,
    plate_scale=FYST_PLATE_SCALE,
    name="PrimeCam-IM6",
)  # theta=-120 deg

PRIMECAM_MODULES: Mapping[str, InstrumentOffset] = MappingProxyType(
    {
        "IM0": PRIMECAM_IM0,
        "IM1": PRIMECAM_IM1,
        "IM2": PRIMECAM_IM2,
        "IM3": PRIMECAM_IM3,
        "IM4": PRIMECAM_IM4,
        "IM5": PRIMECAM_IM5,
        "IM6": PRIMECAM_IM6,
    }
)
"""Read-only mapping of the bay names ``IM0`` .. ``IM6`` to InstrumentOffset instances."""

# Every name get_primecam_offset takes, lower-cased: the seven bays and the two
# spellings of the on-axis bay.
_MODULES_BY_LOWER_NAME: dict[str, InstrumentOffset] = {
    **{name.lower(): offset for name, offset in PRIMECAM_MODULES.items()},
    "c": PRIMECAM_IM0,
    "center": PRIMECAM_IM0,
}

# i1..i6 name no position in this library (they are not bays); a lookup of one
# gets the note below.
_RETIRED_RING_LABELS = frozenset(f"i{n}" for n in range(1, 7))
_RETIRED_RING_LABEL_NOTE = (
    ". The labels i1..i6 are no longer defined: the ring is the bays IM1..IM6, "
    "at different positions, and i<n> is not IM<n>."
)


def get_primecam_offset(module_name: str) -> InstrumentOffset:
    """Get the offset for a Prime-Cam module bay by name.

    Parameters
    ----------
    module_name : str
        Bay name, ``"IM0"`` .. ``"IM6"``, in any case. ``"c"`` and
        ``"center"`` (in any case) name the on-axis bay ``IM0``.

    Returns
    -------
    InstrumentOffset
        The offset for the specified bay.

    Raises
    ------
    KeyError
        If the name is not recognized. The message lists the bay names.
    TypeError
        If ``module_name`` is not a str.

    Examples
    --------
    >>> offset = get_primecam_offset("IM6")
    >>> f"{offset.dx:.1f}', {offset.dy:.1f}'"
    "-53.4', -92.5'"

    >>> get_primecam_offset("c").name
    'PrimeCam-IM0'
    """
    if not isinstance(module_name, str):
        raise TypeError(f"module_name must be a str, got {type(module_name).__name__}")
    key = module_name.lower()
    offset = _MODULES_BY_LOWER_NAME.get(key)
    if offset is None:
        available = ", ".join(PRIMECAM_MODULES)
        message = f"Unknown PrimeCam module '{module_name}'. Available: {available}"
        if key in _RETIRED_RING_LABELS:
            message += _RETIRED_RING_LABEL_NOTE
        raise KeyError(message)
    return offset


def resolve_module_tag(tag: str | Sequence[str]) -> list[InstrumentOffset]:
    """Resolve a comma-separated module tag into a list of module offsets.

    Expands a comma-separated tag of module names into the
    ``list[InstrumentOffset]`` accepted by
    :func:`fyst_trajectories.planning.plan_source_ces` /
    :func:`fyst_trajectories.planning.compute_source_ces_params` via their ``footprint``
    argument, which averages the module centres so the centroid of the selected
    modules lands on the source. Adds no geometry.

    Parameters
    ----------
    tag : str or sequence of str
        ``"IM5,IM6"`` (bay names, comma-separated), a sequence of names
        (``["IM5", "IM6"]``), or ``"all"`` for every bay (``IM0`` ..
        ``IM6``). Case-insensitive; whitespace ignored. Names of the same
        bay (``IM0``, ``c`` and ``center``) are de-duplicated.

    Returns
    -------
    list of InstrumentOffset
        One offset per distinct bay, in input order.

    Raises
    ------
    KeyError
        If a token is not a recognised module (from :func:`get_primecam_offset`).
    ValueError
        If the tag resolves to no modules.
    TypeError
        If ``tag`` is not a str or a sequence, or is a sequence holding an
        element that is not a str.

    Examples
    --------
    >>> [o.name for o in resolve_module_tag("IM5,IM6")]
    ['PrimeCam-IM5', 'PrimeCam-IM6']
    """
    if isinstance(tag, str):
        if tag.strip().lower() == "all":
            names = list(PRIMECAM_MODULES)
        else:
            names = [n.strip() for n in tag.split(",") if n.strip()]
    elif isinstance(tag, Sequence) and not isinstance(tag, (bytes, bytearray)):
        for index, name in enumerate(tag):
            if not isinstance(name, str):
                raise TypeError(
                    "tag must be a str or sequence of str, got "
                    f"{type(name).__name__} at index {index} of {type(tag).__name__}"
                )
        names = [n.strip() for n in tag if n.strip()]
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

    1. Named Prime-Cam bay (e.g., "IM3", "center") -> predefined offset
    2. Custom dx/dy values in arcminutes -> new InstrumentOffset
    3. Neither specified -> None (boresight pointing)

    Parameters
    ----------
    module : str, optional
        Prime-Cam bay name (e.g., "IM3", "IM6"). Looks up predefined offset.
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
        If both ``module`` and ``dx``/``dy`` are specified, or if ``dx`` or
        ``dy`` is not finite.
    KeyError
        If ``module`` is not a recognised module name (from
        :func:`get_primecam_offset`).
    TypeError
        If ``module`` is neither None nor a str (from
        :func:`get_primecam_offset`).

    Examples
    --------
    Named module lookup:

    >>> offset = resolve_offset(module="IM3")
    >>> f"{offset.dx:.1f}', {offset.dy:.1f}'"
    "53.4', 92.5'"

    Custom offset:

    >>> resolve_offset(dx=5.0, dy=3.0)
    InstrumentOffset(dx=5.0', dy=3.0', name='custom')

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
    """Build a scheduler geometry dict for the Prime-Cam modules.

    Adapts :data:`PRIMECAM_MODULES` into the ``{name: {"center": [xi_deg,
    eta_deg], "radius": deg}}`` schema that a scheduler geometry model
    consumes. Each
    module center is the :class:`~fyst_trajectories.offsets.InstrumentOffset`
    cross-elevation/elevation offset (``xi = dx``, ``eta = dy``), converted from
    arcminutes to degrees. The result has one slot per bay, keyed ``"IM0"``
    .. ``"IM6"`` (seven entries).

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
        seven entries (``"IM0"`` .. ``"IM6"``).

    Raises
    ------
    ValueError
        If ``radius_deg`` is not a finite positive number, or an offset is
        not a finite number.

    Notes
    -----
    Centers and radius are in **degrees**. The
    ``xi``/``eta`` axes match
    :class:`~fyst_trajectories.offsets.InstrumentOffset` ``dx``/``dy``
    (cross-elevation / elevation).

    Examples
    --------
    >>> geom = primecam_geometry_dict()
    >>> list(geom)
    ['IM0', 'IM1', 'IM2', 'IM3', 'IM4', 'IM5', 'IM6']
    >>> geom["IM0"]["center"]
    [0.0, 0.0]
    """
    radius_deg = _require_positive(radius_deg, "radius_deg")
    xi_offset_deg = _require_finite(xi_offset_deg, "xi_offset_deg")
    eta_offset_deg = _require_finite(eta_offset_deg, "eta_offset_deg")
    return {
        name: {
            "center": [offset.dx_deg + xi_offset_deg, offset.dy_deg + eta_offset_deg],
            "radius": radius_deg,
        }
        for name, offset in PRIMECAM_MODULES.items()
    }
