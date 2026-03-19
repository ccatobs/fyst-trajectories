Instrument Offsets
==================

Utilities for handling detector offsets from telescope boresight.

.. automodule:: fyst_trajectories.offsets
   :members: InstrumentOffset, boresight_to_detector, detector_to_boresight, apply_detector_offset, compute_focal_plane_rotation
   :undoc-members:
   :no-index:

.. automodule:: fyst_trajectories.primecam
   :members: resolve_offset, get_primecam_offset, PRIMECAM_MODULES, INNER_RING_RADIUS_MM
   :undoc-members:
   :no-index:

Quick Example
-------------

::

    from astropy.time import Time

    from fyst_trajectories import InstrumentOffset, get_fyst_site
    from fyst_trajectories.offsets import boresight_to_detector
    from fyst_trajectories.patterns import PongScanConfig, TrajectoryBuilder
    from fyst_trajectories.primecam import get_primecam_offset

    # Custom offset (arcmin)
    offset = InstrumentOffset(dx=5.0, dy=3.0)

    # Compute detector position from boresight
    det_az, det_el = boresight_to_detector(
        az=180.0, el=45.0,
        offset=offset,
        field_rotation=0.0,
    )

    # Use predefined PrimeCam offset
    i1_offset = get_primecam_offset("i1")

    # Generate trajectory with detector offset
    # Use a specific time when target is observable
    start_time = Time("2026-03-15T04:00:00", scale="utc")

    trajectory = (
        TrajectoryBuilder(get_fyst_site())
        .at(ra=180.0, dec=-30.0)
        .with_config(PongScanConfig(
            timestep=0.1, width=1.0, height=1.0, spacing=0.1,
            velocity=0.5, num_terms=4, angle=0.0,
        ))
        .for_detector(i1_offset)
        .duration(60.0)
        .starting_at(start_time)
        .build()
    )

PrimeCam Modules
----------------

.. py:data:: fyst_trajectories.primecam.PRIMECAM_CENTER

   Center module (0, 0).

.. py:data:: fyst_trajectories.primecam.PRIMECAM_I1

   Inner ring module 1.

.. py:data:: fyst_trajectories.primecam.PRIMECAM_I2

   Inner ring module 2.

.. py:data:: fyst_trajectories.primecam.PRIMECAM_I3

   Inner ring module 3.

.. py:data:: fyst_trajectories.primecam.PRIMECAM_I4

   Inner ring module 4.

.. py:data:: fyst_trajectories.primecam.PRIMECAM_I5

   Inner ring module 5.

.. py:data:: fyst_trajectories.primecam.PRIMECAM_I6

   Inner ring module 6.

.. py:data:: fyst_trajectories.primecam.PRIMECAM_MODULES

   Dictionary mapping module names to InstrumentOffset objects.
   Keys: ``"c"``, ``"center"``, ``"i1"`` through ``"i6"``.

Offset Calculation
------------------

All offset calculations use spherical trigonometry, providing sub-milliarcsecond
precision for both small and large offsets.

See :doc:`../instrument_offsets` for detailed usage examples.
