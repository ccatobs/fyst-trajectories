fyst-trajectories
==================

Trajectory generation for the FYST (Fred Young Submillimeter
Telescope).  Wraps astropy with FYST-specific site coordinates, telescope
limits, and scan pattern generators.

This release adds multi-pass source-CES sequences
(``plan_source_ces_passes``) that step the focal plane across a source in
elevation for complete calibration coverage, ``start_time`` anchoring on the
source-CES planners, opt-in planet calibrations run as real source-CES scans
in the overhead simulator (reconstructible from the saved timeline), and
night-level visualization figures. See :doc:`planning`.

Scope and boundaries
--------------------

This library generates planning-time trajectories and overhead
estimates. A few concerns deliberately live outside its scope:

- **Pointing-model corrections** are applied at execution time by the 
  Telescope Control System. They are not computed here.
- **PWV / atmospheric opacity** affects sky brightness and absolute
  flux calibration but does not affect trajectory geometry; opacity
  modelling lives downstream in the calibration pipeline / sky model.
- **Hard interlocks** are enforced downstream by the TCS at execution
  time. The library's own checks split two ways: elevation and azimuth
  position bounds *raise* (``ElevationBoundsError``,
  ``TargetNotObservableError``) and refuse to return an out-of-bounds
  trajectory, while dynamics (scan velocity, acceleration) and Sun
  proximity are advisory only, emitting ``PointingWarning`` without
  refusing to generate. Downstream consumers must still enforce the
  actual hardware limits.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   coordinate_systems
   trajectory_examples
   instrument_offsets
   planning
   retune_events
   api/index

.. toctree::
   :maxdepth: 2
   :caption: Overhead Modeling:

   overhead_quickstart
   overhead_integration
   overhead_timeline
   overhead_model
   overhead_io

Pending instrument verification
-------------------------------

The following parameters use commissioning-era defaults that should be
confirmed by the FYST instrument and operations teams before production
use.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Parameter
     - Default
     - Override
   * - Sun avoidance radius
     - 45° exclusion / 50° warning
     - ``get_fyst_site(sun_exclusion_radius=...)``
   * - Nasmyth port
     - ``"right"`` (+1 sign)
     - module constant ``site.FYST_NASMYTH_PORT`` (not a call-time option)
   * - Az/El velocity limits
     - 3.0 / 1.0 deg/s
     - ``get_fyst_site()`` kwargs
   * - Az/El acceleration limits
     - 1.5 / 0.75 deg/s²
     - ``get_fyst_site()`` kwargs
   * - Plate scale
     - 13.89 arcsec/mm
     - module constant ``site.FYST_PLATE_SCALE`` (not a call-time option)
   * - PrimeCam inner ring radius
     - 461.3 mm
     - module constant ``primecam.INNER_RING_RADIUS_MM`` (not a call-time option)
   * - Retune interval
     - 300 s
     - ``inject_retune(retune_interval=...)``
   * - Skydip cadence
     - 10 800 s (3 h)
     - ``CalibrationPolicy(skydip_cadence=...)``
   * - Per-module retune
     - Disabled (all modules retune together)
     - ``inject_retune(n_modules=7, module_index=...)``
   * - Per-module FOV radius (PrimeCam)
     - 0.65°
     - ``primecam.MODULE_FOV_RADIUS_DEG`` or pass an explicit
       ``ArrayFootprint`` to ``plan_source_ces``

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
