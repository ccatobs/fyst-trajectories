fyst-trajectories
==================

Trajectory generation for FYST, the Fred Young Submillimeter Telescope.
Wraps astropy with FYST-specific site coordinates, telescope limits, and
scan pattern generators. Written for CCAT/FYST collaborators: authors of
control and scheduling software, and anyone planning or simulating
Prime-Cam observations.

What it does: nine scan patterns (pong, daisy, constant-elevation,
linear, sidereal, planet and satellite tracking, plus AltAz-frame pong
and daisy) and focal-plane offsets for the Prime-Cam modules, with scan
planning, selectable sun-avoidance models, observability reporting and
matplotlib figures on top. The offline observing-night simulator and the
calibration-night planner are a separate tier (below).

Start here:

- :doc:`quickstart` - the site, coordinate transforms, and a first
  trajectory.
- :doc:`planning` - turn a field or a source into a scan block.
- :doc:`sun_avoidance` - check a target against the Sun, choose an
  avoidance policy, and gate a slew at dispatch.
- :doc:`api/observability` - which calibrators are up, when, and why not.
- :doc:`api/visualization` - visibility curves, the instantaneous all-sky
  view, the focal-plane footprint, and hit-density maps.
- :doc:`overhead_calibration_night` - one night of solar-system
  calibration passes, planned back to back.

Two tiers
---------

The package has a library tier and a simulator tier, and the dependency
between them runs one way.

- **Library tier** (``import fyst_trajectories``): the site, coordinate
  transforms, scan patterns and trajectories, focal-plane offsets, the
  scan planners, the dispatch-time encoder gate, the sun-avoidance
  models, the observability reports, and the plotting subpackage
  ``fyst_trajectories.visualization``. This is what a control system
  or a scheduler imports, and it never loads the simulator.
- **Simulator tier** (``import fyst_trajectories.overhead``): the offline
  observing-night simulator and its scheduler, the calibration-night
  planner, the timeline model with its ECSV format, and the figures that
  draw timelines. It imports the library; nothing in the library imports
  it.

Scope and boundaries
--------------------

This library generates trajectories, at planning time and at dispatch,
and the offline simulator's overhead estimates. A few concerns live
outside its scope:

- **Pointing-model corrections** are applied downstream at execution
  time, nominally in the ACU. They are not computed here.
- **PWV / atmospheric opacity** affects sky brightness and absolute
  flux calibration but does not affect trajectory geometry; opacity
  modelling lives downstream in the calibration pipeline / sky model.
- **Hard limits** live downstream: the telescope control system range-checks
  commanded position and velocity, and the Sun interlock is an ACU function,
  not a control-system one. The library's own checks are planning aids: the
  "Where the check runs" table in :doc:`sun_avoidance` says which stage raises
  and which only warns, and :doc:`api/exceptions` documents the classes.
  Nothing downstream is guaranteed to enforce this library's limits, so a
  consumer must enforce the ones it relies on.

.. toctree::
   :maxdepth: 2
   :caption: Library

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
   :caption: Offline simulator

   overhead_quickstart
   overhead_timeline
   overhead_model
   overhead_io
   overhead_calibration_night
   overhead_integration
   api/overhead_index

.. toctree::
   :maxdepth: 1
   :caption: Project

   changelog

.. _index-pending-verification:

Pending instrument verification
-------------------------------

The following parameters use commissioning-era defaults that should be
confirmed by the FYST instrument and operations teams before production
use. A row whose Override is a module constant has no call-time keyword.
``get_fyst_site()`` reads the ``site`` constants each time it builds a
site, but a value bound at import does not follow a later rebinding: the
Prime-Cam geometry is computed once, at import, and
:func:`~fyst_trajectories.sun_models.make_slew_safe` takes the axis
limits as keyword defaults (pass ``az_speed=`` and the others there).

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Parameter
     - Default
     - Override
   * - Nasmyth port
     - ``"right"`` (+1 sign)
     - module constant ``site.FYST_NASMYTH_PORT``
   * - Sun exclusion / warning radii
     - 45° / 50° (the Prime-Cam observing baseline, not yet formalised in
       an interface control document)
     - ``get_fyst_site(sun_exclusion_radius=, sun_warning_radius=)``, or
       select the directional model with
       :func:`~fyst_trajectories.sun_models.make_sun_safe`
   * - Az/El velocity limits
     - 3.0 / 1.0 deg/s
     - module constants ``site.FYST_AZ_MAX_VELOCITY`` /
       ``site.FYST_EL_MAX_VELOCITY``
   * - Az/El acceleration limits
     - 1.5 / 0.75 deg/s²
     - module constants ``site.FYST_AZ_MAX_ACCELERATION`` /
       ``site.FYST_EL_MAX_ACCELERATION``
   * - Plate scale
     - 13.89 arcsec/mm
     - module constant ``site.FYST_PLATE_SCALE``, read once at import by
       the Prime-Cam offsets; for another value build them with
       ``InstrumentOffset.from_focal_plane(x_mm, y_mm, plate_scale)``
   * - Prime-Cam inner-ring radius
     - 461.3 mm
     - module constant ``primecam.INNER_RING_RADIUS_MM``, read once at
       import; for another radius build the offsets with
       ``InstrumentOffset.from_focal_plane``
   * - Prime-Cam inner-ring ordering (clocking and parity)
     - ``i1`` at focal-plane angle -90°, ``i1`` .. ``i6`` counterclockwise
       in the focal-plane (cross-elevation, elevation) frame, the
       orientation
       :func:`~fyst_trajectories.visualization.plot_array_footprint`
       draws; the mapping to the instrument team's ``IM0`` .. ``IM6``
       labels is pending confirmation (see :doc:`instrument_offsets`)
     - module constants ``primecam.PRIMECAM_I1`` .. ``PRIMECAM_I6``, fixed
       at import; pass your own ``InstrumentOffset`` to ``.for_detector()``
       or ``detector_offset=`` instead
   * - Retune interval (in-scan)
     - 300 s
     - ``inject_retune(retune_interval=...)``
   * - Whole-array retune duration
     - 300 s
     - ``OverheadModel(retune_duration=...)``; distinct from the few-second
       in-scan gap ``inject_retune(retune_duration=...)`` stamps into a
       trajectory
   * - Skydip cadence
     - 10 800 s (3 h)
     - ``CalibrationPolicy(skydip_cadence=...)``
   * - Per-module retune
     - Disabled (all modules retune together)
     - ``inject_retune(n_modules=7, module_index=...)``
   * - Per-module FOV radius (Prime-Cam)
     - 0.65°
     - pass an explicit ``ArrayFootprint`` to ``plan_source_ces``;
       ``primecam.MODULE_FOV_RADIUS_DEG`` is bound at import, so rebinding
       it has no effect
   * - Calibration cadences (offline simulator)
     - pointing 3600 s, focus 7200 s, planet cal 43 200 s
     - ``CalibrationPolicy(pointing_cadence=, ...)``
   * - Planet-calibration scan geometry
     - parked block; 3 passes on ``c`` if ``planet_cal_scan=True``
     - ``CalibrationPolicy(planet_cal_passes=, planet_cal_footprint=, ...)``
   * - Calibration-night scan tables and scan azimuth speed / acceleration
     - ``DEFAULT_SCAN_TABLES["default"]``; 1.5 deg/s, 1.0 deg/s²
     - ``CalibrationNightPolicy(az_speed=, az_accel=)``,
       ``plan_calibration_night(tables=)``
   * - Constant-elevation azimuth range
     - projection of the whole field over the whole pass (not an
       elevation-band drift corridor)
     - none in the planner; build the scan from an explicit
       ``ConstantElScanConfig`` for another range
   * - Constant-elevation azimuth padding
     - 2.0° per side
     - ``plan_constant_el_scan(az_padding=)``

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
