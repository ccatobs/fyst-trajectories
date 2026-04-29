fyst-trajectories
==================

Trajectory generation for the FYST (Fred Young Submillimeter
Telescope).  Wraps astropy with FYST-specific site coordinates, telescope
limits, and scan pattern generators.

Scope and boundaries
--------------------

This library generates planning-time trajectories and overhead
estimates. A few concerns deliberately live outside its scope:

- **Pointing-model corrections** are applied at execution time by the 
  Telescope Control System. They are not computed here.
- **PWV / atmospheric opacity** affects sky brightness and absolute
  flux calibration but does not affect trajectory geometry; opacity
  modelling lives downstream in the calibration pipeline / sky model.
- **Hard interlocks** (sun, elevation, scan-velocity limits) are the
  TCS's responsibility. This library emits ``PointingWarning`` when a
  trajectory approaches the configured envelopes but never refuses to
  generate; downstream consumers must enforce the actual hardware
  limits.

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
     - ``site.FYST_NASMYTH_PORT``
   * - Az/El velocity limits
     - 3.0 / 1.0 deg/s
     - ``get_fyst_site()`` kwargs
   * - Az/El acceleration limits
     - 1.0 / 0.5 deg/s²
     - ``get_fyst_site()`` kwargs
   * - Plate scale
     - 13.89 arcsec/mm
     - ``site.FYST_PLATE_SCALE``
   * - PrimeCam inner ring radius
     - 461.3 mm
     - ``primecam.INNER_RING_RADIUS_MM``
   * - Retune interval
     - 300 s
     - ``inject_retune(retune_interval=...)``
   * - Skydip cadence
     - 10 800 s (3 h)
     - ``CalibrationPolicy(skydip_cadence=...)``
   * - Per-module retune
     - Disabled (all modules retune together)
     - ``inject_retune(n_modules=7, module_index=...)``

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
