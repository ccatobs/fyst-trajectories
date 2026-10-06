Offsets and Prime-Cam Modules
=============================

Focal-plane offset projection (boresight to detector and back) and the
Prime-Cam module constants.

.. automodule:: fyst_trajectories.offsets
   :members: InstrumentOffset, boresight_to_detector, detector_to_boresight, sky_to_focal_plane, apply_detector_offset, compute_focal_plane_rotation
   :undoc-members:

.. automodule:: fyst_trajectories.primecam
   :members: resolve_offset, resolve_module_tag, get_primecam_offset, primecam_geometry_dict,
      INNER_RING_RADIUS_MM
   :undoc-members:

.. autodata:: fyst_trajectories.primecam.PRIMECAM_MODULES
   :no-value:

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
        focal_plane_rotation=0.0,
    )

    # Predefined PrimeCam module offset, ready for TrajectoryBuilder.for_detector()
    i1_offset = get_primecam_offset("i1")

Prime-Cam Modules
-----------------

Module names label focal-plane positions (one on-axis, six on the inner
ring), not the instrument modules that occupy them. The six ring
positions differ only in clocking: ``i1`` sits at focal-plane angle -90°
and ``i1`` .. ``i6`` step counterclockwise in the focal-plane
(cross-elevation, elevation) frame, the orientation
:func:`~fyst_trajectories.visualization.plot_array_footprint` draws.
See :doc:`../instrument_offsets` for the offset table and for the naming
convention, which awaits as-built confirmation.

.. py:data:: PRIMECAM_CENTER
             PRIMECAM_I1
             PRIMECAM_I2
             PRIMECAM_I3
             PRIMECAM_I4
             PRIMECAM_I5
             PRIMECAM_I6

   The seven positions as :class:`~fyst_trajectories.offsets.InstrumentOffset`
   constants, the same objects
   :data:`~fyst_trajectories.primecam.PRIMECAM_MODULES` holds under ``"c"``
   and ``"i1"`` .. ``"i6"``; all seven are also exported from
   ``fyst_trajectories``.

.. autodata:: fyst_trajectories.primecam.MODULE_FOV_RADIUS_DEG
