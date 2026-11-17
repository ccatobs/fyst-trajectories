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
    im6_offset = get_primecam_offset("IM6")

Prime-Cam Modules
-----------------

Module names label focal-plane positions (one on-axis bay, ``IM0``, and
six on the inner ring), not the instrument modules that occupy them. The
six ring bays differ only in clocking: ``IM1`` sits at focal-plane angle
180° and ``IM2`` .. ``IM6`` step clockwise by 60° in the focal-plane
(cross-elevation, elevation) frame seen looking out at the sky, the
orientation :func:`~fyst_trajectories.visualization.plot_array_footprint`
draws. See :doc:`../instrument_offsets` for the offset table and for the
naming convention; the positions await an on-sky check.

.. py:data:: PRIMECAM_IM0
             PRIMECAM_IM1
             PRIMECAM_IM2
             PRIMECAM_IM3
             PRIMECAM_IM4
             PRIMECAM_IM5
             PRIMECAM_IM6

   The seven positions as :class:`~fyst_trajectories.offsets.InstrumentOffset`
   constants, the same objects
   :data:`~fyst_trajectories.primecam.PRIMECAM_MODULES` holds under
   ``"IM0"`` .. ``"IM6"``; all seven are also exported from
   ``fyst_trajectories``.

.. autodata:: fyst_trajectories.primecam.MODULE_FOV_RADIUS_DEG
