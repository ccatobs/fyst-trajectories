"""PrimeCam instrument configuration.

Pre-defined module offsets for the PrimeCam focal-plane instrument on FYST.
PrimeCam has one center module on the optical axis and six inner-ring modules
at 461.3 mm radius, spaced 60 degrees apart.

The module positions are converted from physical focal-plane coordinates (mm)
to angular offsets (arcminutes) using ``FYST_PLATE_SCALE`` from ``site.py``.

Examples
--------
Get a named module offset:

>>> from fyst_pointing.primecam import get_primecam_offset
>>> offset = get_primecam_offset("i1")
>>> print(offset)
InstrumentOffset(dx=0.0', dy=-106.8', name='PrimeCam-I1')

List available modules:

>>> from fyst_pointing.primecam import PRIMECAM_MODULES
>>> print(sorted(PRIMECAM_MODULES.keys()))
['c', 'center', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6']
"""

import numpy as np

from .offsets import InstrumentOffset
from .site import FYST_PLATE_SCALE

# Module positions use standard polar convention: x = r*cos(theta),
# y = r*sin(theta), where theta is measured counterclockwise from the
# +x axis. At zero field rotation, x is the cross-elevation direction
# and y is the elevation direction.
#
# TODO: Actual values should be verified with the instrument team.

INNER_RING_RADIUS_MM = 461.3
"""Inner ring module distance from optical axis in millimeters."""


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
    >>> print(offset)
    InstrumentOffset(dx=0.0', dy=-106.8', name='PrimeCam-I1')
    """
    key = module_name.lower()
    if key not in PRIMECAM_MODULES:
        available = ", ".join(sorted(PRIMECAM_MODULES.keys()))
        raise KeyError(f"Unknown PrimeCam module '{module_name}'. Available: {available}")
    return PRIMECAM_MODULES[key]


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

    >>> resolve_offset(module="i3")
    InstrumentOffset(dx=92.5', dy=53.4', name='PrimeCam-I3')

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
        return InstrumentOffset(dx=dx or 0.0, dy=dy or 0.0, name=name)

    return None
