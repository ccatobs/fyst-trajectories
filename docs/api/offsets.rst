Instrument Offsets
==================

.. automodule:: fyst_trajectories.offsets
   :members: InstrumentOffset, boresight_to_detector, detector_to_boresight, apply_detector_offset, compute_focal_plane_rotation
   :undoc-members:

.. automodule:: fyst_trajectories.primecam
   :members: resolve_offset, resolve_module_tag, get_primecam_offset, primecam_geometry_dict, PRIMECAM_MODULES, INNER_RING_RADIUS_MM
   :undoc-members:

Quick Example
-------------

::

    from fyst_trajectories import InstrumentOffset
    from fyst_trajectories.offsets import boresight_to_detector
    from fyst_trajectories.primecam import get_primecam_offset

    # Custom offset (arcmin)
    offset = InstrumentOffset(dx=5.0, dy=3.0)

    # Where a detector lands when the boresight is at (az, el)
    det_az, det_el = boresight_to_detector(
        az=180.0, el=45.0,
        offset=offset,
        field_rotation=0.0,
    )

    # Predefined PrimeCam module offset, ready for TrajectoryBuilder.for_detector()
    i1_offset = get_primecam_offset("i1")

PrimeCam Modules
----------------

.. py:data:: PRIMECAM_CENTER

   Center module (0, 0).

.. py:data:: PRIMECAM_I1

   Inner ring module 1.

.. py:data:: PRIMECAM_I2

   Inner ring module 2.

.. py:data:: PRIMECAM_I3

   Inner ring module 3.

.. py:data:: PRIMECAM_I4

   Inner ring module 4.

.. py:data:: PRIMECAM_I5

   Inner ring module 5.

.. py:data:: PRIMECAM_I6

   Inner ring module 6.

.. autodata:: fyst_trajectories.primecam.MODULE_FOV_RADIUS_DEG

See :doc:`../instrument_offsets` for the offset workflow and worked examples.
