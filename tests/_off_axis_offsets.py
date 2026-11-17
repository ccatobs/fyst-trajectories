"""Off-axis detector offsets at fixed focal-plane positions, for the planner tests.

Each offset sits one Prime-Cam ring radius (``INNER_RING_RADIUS_MM``) from the
boresight at one of six focal-plane angles, -90, -30, 30, 90, 150 and -150 deg,
measured counter-clockwise from +cross-elevation. They are NOT Prime-Cam bays:
they are plain ``InstrumentOffset`` positions. The tests that use them pin how a
planner places a pass or a sweep for a cover at one particular position, and their
pinned values were tuned at these positions, so the positions must not move when
the Prime-Cam module layout does.
"""

from __future__ import annotations

import numpy as np

from fyst_trajectories import InstrumentOffset
from fyst_trajectories.primecam import INNER_RING_RADIUS_MM
from fyst_trajectories.site import FYST_PLATE_SCALE


def _on_ring(angle_deg: float) -> InstrumentOffset:
    """Build the offset one ring radius from the boresight at ``angle_deg``."""
    if angle_deg in (-90, 90):
        # On the elevation axis the cross-elevation coordinate is exactly zero.
        x_mm = 0.0
        y_mm = INNER_RING_RADIUS_MM if angle_deg > 0 else -INNER_RING_RADIUS_MM
    else:
        x_mm = INNER_RING_RADIUS_MM * np.cos(np.deg2rad(angle_deg))
        y_mm = INNER_RING_RADIUS_MM * np.sin(np.deg2rad(angle_deg))
    return InstrumentOffset.from_focal_plane(
        x_mm=x_mm,
        y_mm=y_mm,
        plate_scale=FYST_PLATE_SCALE,
        name=f"off-axis {angle_deg} deg",
    )


_OFF_AXIS_OFFSETS = {angle: _on_ring(angle) for angle in (-90, -30, 30, 90, 150, -150)}


def off_axis_offset(angle_deg: int) -> InstrumentOffset:
    """Return the off-axis offset at one of the six focal-plane angles.

    Parameters
    ----------
    angle_deg : int
        The focal-plane angle in degrees: -90, -30, 30, 90, 150 or -150.

    Returns
    -------
    InstrumentOffset
        The offset one ring radius from the boresight at that angle.

    Raises
    ------
    KeyError
        If ``angle_deg`` is not one of the six angles.
    """
    return _OFF_AXIS_OFFSETS[angle_deg]
