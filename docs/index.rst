fyst-trajectories
==================

Trajectory generation for the FYST (Fred Young Submillimeter
Telescope).  Wraps astropy with FYST-specific site coordinates, telescope
limits, and scan pattern generators.

What it does: nine scan patterns (pong, daisy, constant-elevation,
linear, sidereal, planet and satellite tracking, plus AltAz-frame pong
and daisy), focal-plane offsets for the PrimeCam modules, source-crossing
constant-elevation planning (single- and multi-pass), target-visibility
and Sun-almanac reporting, selectable sun-avoidance models (the site
scalar radii by default; FYST's directional CAD zone is opt-in today and
is expected to become the default in a future release), an offline
observing-night overhead simulator, and matplotlib figures for
trajectories, visibility, all-sky and coverage views, focal-plane
footprints, hit maps, and night timelines.

Start here:

- :doc:`quickstart` - the site, coordinate transforms, and a first
  trajectory.
- :doc:`planning` - turn a field or a source into a scan block.
- :doc:`sun_avoidance` - check a target against the Sun, choose an
  avoidance policy, and gate a slew at dispatch.
- :doc:`api/observability` - which calibrators are up, when, and why not.
- :doc:`api/visualization` - visibility curves, the instantaneous all-sky
  view, the focal-plane footprint, and night-level overhead figures.

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
  position bounds *raise* (``ElevationBoundsError``, ``AzimuthBoundsError``;
  the planners wrap them as ``TargetNotObservableError``) and refuse to
  return an out-of-bounds trajectory, while dynamics (scan velocity,
  acceleration) and in-scan Sun proximity are advisory only, emitting
  ``PointingWarning`` without refusing to generate. The dispatch-time
  gate is stricter:
  :func:`~fyst_trajectories.dispatch.choose_encoder_solution` raises
  when no sun-safe azimuth wrap is available (see :doc:`sun_avoidance`).
  Downstream consumers must still enforce the actual hardware limits.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   coordinate_systems
   trajectory_examples
   instrument_offsets
   planning
   sun_avoidance
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
   * - Nasmyth port
     - ``"right"`` (+1 sign)
     - module constant ``site.FYST_NASMYTH_PORT`` (not a call-time option)
   * - Az/El velocity limits
     - 3.0 / 1.0 deg/s
     - module constants ``site.FYST_AZ_MAX_VELOCITY`` /
       ``site.FYST_EL_MAX_VELOCITY`` (not a call-time option)
   * - Az/El acceleration limits
     - 1.5 / 0.75 deg/s²
     - module constants ``site.FYST_AZ_MAX_ACCELERATION`` /
       ``site.FYST_EL_MAX_ACCELERATION`` (not a call-time option)
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
