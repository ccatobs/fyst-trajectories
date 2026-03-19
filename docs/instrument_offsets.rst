Instrument Offsets
==================

Handle detector offsets from telescope boresight. When an off-axis detector
should track a target, the boresight must be offset in the opposite direction,
accounting for field rotation.

Quick Example
-------------

::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import PongScanConfig, TrajectoryBuilder
    from fyst_trajectories.primecam import get_primecam_offset

    site = get_fyst_site()
    start_time = Time("2026-03-15T04:00:00", scale="utc")

    # Use predefined PrimeCam offset
    offset = get_primecam_offset("i1")

    # Boresight adjusted so detector I1 tracks the target
    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=180.0, dec=-30.0)
        .with_config(PongScanConfig(
            timestep=0.1, width=1.0, height=1.0, spacing=0.1,
            velocity=0.5, num_terms=4, angle=0.0,
        ))
        .for_detector(offset)
        .duration(60.0)
        .starting_at(start_time)
        .build()
    )

Field Rotation Decomposition
-----------------------------

For alt-az mounted telescopes like FYST, the focal plane rotates as objects
are tracked. The total rotation applied to detector offsets is decomposed into:

**Mechanical rotation** (always available):

    ``nasmyth_sign * elevation + instrument_rotation``

- ``nasmyth_sign``: +1 for Right Nasmyth, -1 for Left Nasmyth, 0 for Cassegrain.
  Configured via ``Site.nasmyth_port`` (default: ``"right"``).
- ``instrument_rotation``: fixed rotation of the instrument relative to the
  Nasmyth flange, in degrees. Set on ``InstrumentOffset`` (default: 0.0).

**Parallactic angle** (requires celestial coordinates):

    Added when the trajectory has RA/Dec metadata (celestial patterns like
    Pong, Daisy, Sidereal). Not available for AltAz-only patterns
    (ConstantEl, Linear).

The full rotation is:

    ``rotation = nasmyth_sign * elevation + instrument_rotation + parallactic_angle``

``apply_detector_offset`` automatically selects the appropriate decomposition
based on whether the trajectory contains celestial coordinates.

The helper ``compute_focal_plane_rotation`` provides the same calculation
for use outside of trajectory adjustment::

    from fyst_trajectories import get_fyst_site, InstrumentOffset
    from fyst_trajectories.offsets import compute_focal_plane_rotation

    site = get_fyst_site()
    offset = InstrumentOffset(dx=5.0, dy=3.0, instrument_rotation=10.0)

    rotation = compute_focal_plane_rotation(
        el=45.0, site=site, offset=offset, parallactic_angle=20.0
    )
    # rotation = +1 * 45.0 + 10.0 + 20.0 = 75.0

Point Transformations
---------------------

Compute detector position from boresight::

    from fyst_trajectories import InstrumentOffset
    from fyst_trajectories.offsets import boresight_to_detector

    offset = InstrumentOffset(dx=5.0, dy=3.0)  # arcmin

    det_az, det_el = boresight_to_detector(
        az=180.0, el=45.0,
        offset=offset,
        field_rotation=30.0,  # degrees
    )

Compute boresight position to place detector on target::

    from fyst_trajectories.offsets import detector_to_boresight

    bore_az, bore_el = detector_to_boresight(
        det_az=180.0, det_el=45.0,
        offset=offset,
        field_rotation=30.0,
    )

Trajectory Adjustment
---------------------

Apply offset to entire trajectory with time-varying field rotation::

    from astropy.time import Time

    from fyst_trajectories import InstrumentOffset, get_fyst_site
    from fyst_trajectories.offsets import apply_detector_offset
    from fyst_trajectories.patterns import PongScanConfig, TrajectoryBuilder

    site = get_fyst_site()
    start_time = Time("2026-03-15T04:00:00", scale="utc")

    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=180.0, dec=-30.0)
        .with_config(PongScanConfig(
            timestep=0.1, width=1.0, height=1.0, spacing=0.1,
            velocity=0.5, num_terms=4, angle=0.0,
        ))
        .duration(60.0)
        .starting_at(start_time)
        .build()
    )

    offset = InstrumentOffset(dx=30.0, dy=0.0)
    adjusted = apply_detector_offset(trajectory, offset, site)

.. note::

   For AltAz trajectories (no RA/Dec), ``apply_detector_offset`` uses only
   the mechanical rotation and emits a warning. This is physically correct
   for focal-plane-to-AltAz conversion but does not include sky rotation.

PrimeCam Offsets
----------------

Predefined offsets for PrimeCam focal plane:

**Center**: ``get_primecam_offset("c")`` or ``PRIMECAM_CENTER`` - at boresight (0, 0)

**Inner Ring** (1.78 deg = 106.8 arcmin from center):

+------------+----------------+----------------+
| Name       | dx (arcmin)    | dy (arcmin)    |
+============+================+================+
| I1         | 0.0            | -106.8         |
+------------+----------------+----------------+
| I2         | 92.5           | -53.4          |
+------------+----------------+----------------+
| I3         | 92.5           | 53.4           |
+------------+----------------+----------------+
| I4         | 0.0            | 106.8          |
+------------+----------------+----------------+
| I5         | -92.5          | 53.4           |
+------------+----------------+----------------+
| I6         | -92.5          | -53.4          |
+------------+----------------+----------------+

.. note::

   These are illustrative values. We need to verify specific with the instrument team before
   production use!

**Access**::

    from fyst_trajectories.primecam import (
        PRIMECAM_I1,
        PRIMECAM_MODULES,
        get_primecam_offset,
        resolve_offset,
    )

    offset = get_primecam_offset("i1")  # Case-insensitive
    offset = PRIMECAM_I1                # Direct access

    for name, offset in PRIMECAM_MODULES.items():
        print(f"{name}: dx={offset.dx:.1f}', dy={offset.dy:.1f}'")

Resolving user input with ``resolve_offset``:

``resolve_offset`` is the preferred entry point when offset specification comes
from user input (CLI, config file, API request). It handles all three cases in
one call: named module, custom dx/dy, or boresight (``None``)::

    from fyst_trajectories.primecam import resolve_offset

    # Named module -> predefined InstrumentOffset
    offset = resolve_offset(module="i1")

    # Custom angular offset (arcmin)
    offset = resolve_offset(dx=10.0, dy=5.0, name="my-detector")

    # dx only; dy defaults to 0.0
    offset = resolve_offset(dx=10.0)

    # No arguments -> None (boresight pointing)
    offset = resolve_offset()

Custom Offsets
--------------

**From angular offsets (arcminutes)**::

    from fyst_trajectories import InstrumentOffset

    offset = InstrumentOffset(dx=10.0, dy=5.0, name="MyDetector")

    # Values are in arcminutes; properties provide degrees
    print(f"{offset.dx_deg:.4f} x {offset.dy_deg:.4f} degrees")

    # With instrument rotation (e.g., dewar rotated 15 degrees)
    offset = InstrumentOffset(
        dx=10.0, dy=5.0, name="RotatedDetector", instrument_rotation=15.0
    )

**From physical focal plane coordinates (millimeters)**::

    from fyst_trajectories import InstrumentOffset, get_fyst_site

    site = get_fyst_site()

    # Convert physical position to angular offset using plate scale
    offset = InstrumentOffset.from_focal_plane(
        x_mm=230.65,           # Cross-elevation position (mm)
        y_mm=399.5,            # Elevation position (mm)
        plate_scale=site.plate_scale,  # 13.89 arcsec/mm
        name="Module-A2",
    )
    print(f"Angular offset: {offset.dx:.1f}' x {offset.dy:.1f}'")

    # With instrument rotation (e.g., dewar at 15 degree angle)
    offset = InstrumentOffset.from_focal_plane(
        x_mm=100.0, y_mm=200.0,
        plate_scale=site.plate_scale,
        name="RotatedModule",
        instrument_rotation=15.0,
    )

The plate scale (13.89 arcsec/mm) is a property of the telescope optical
system (6m primary + reimaging optics), accessible via ``site.plate_scale``.
It converts detector positions in the focal plane to angular offsets on
the sky.

Offset Calculation
------------------

The library uses spherical trigonometry for all offset calculations, providing
sub-milliarcsecond precision for both small and large offsets::

    from fyst_trajectories import InstrumentOffset
    from fyst_trajectories.offsets import boresight_to_detector

    # Works for any offset size (small or large)
    offset = InstrumentOffset(dx=180.0, dy=90.0, name="OuterRing")  # 3 degrees

    det_az, det_el = boresight_to_detector(
        az=180.0, el=45.0,
        offset=offset,
        field_rotation=30.0,
    )
