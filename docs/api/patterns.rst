Patterns Package
================

Scan pattern implementations for telescope trajectory generation.

Overview
--------

``TrajectoryBuilder`` generates trajectories from config objects.
The pattern type is automatically inferred from the config class::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import PongScanConfig, TrajectoryBuilder

    start_time = Time("2026-03-15T04:00:00", scale="utc")

    trajectory = (
        TrajectoryBuilder(get_fyst_site())
        .at(ra=180.0, dec=-30.0)
        .with_config(PongScanConfig(
            timestep=0.1, width=2.0, height=2.0, spacing=0.1,
            velocity=0.5, num_terms=4, angle=0.0,
        ))
        .duration(300.0)
        .starting_at(start_time)
        .build()
    )

Available patterns: ``constant_el``, ``daisy``, ``daisy_altaz``, ``linear``, ``planet``, ``pong``, ``pong_altaz``, ``satellite``, ``sidereal``.

TrajectoryBuilder
-----------------

.. autoclass:: fyst_trajectories.patterns.TrajectoryBuilder
   :members:
   :undoc-members:

**Detector offset support**::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import PongScanConfig, TrajectoryBuilder
    from fyst_trajectories.primecam import get_primecam_offset

    site = get_fyst_site()
    start_time = Time("2026-03-15T04:00:00", scale="utc")

    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=180.0, dec=-30.0)
        .with_config(PongScanConfig(
            timestep=0.1, width=1.0, height=1.0, spacing=0.1,
            velocity=0.5, num_terms=4, angle=0.0,
        ))
        .for_detector(get_primecam_offset("i1"))
        .duration(60.0)
        .starting_at(start_time)
        .build()
    )

Base Classes
------------

.. autoclass:: fyst_trajectories.patterns.ScanPattern
   :members:

.. autoclass:: fyst_trajectories.patterns.CelestialPattern
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.patterns.AltAzPattern
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.patterns.TrajectoryMetadata
   :members:

Configuration Classes
---------------------

.. autoclass:: fyst_trajectories.patterns.ScanConfig
   :members:

.. autoclass:: fyst_trajectories.patterns.ConstantElScanConfig
   :members:
   :show-inheritance:

.. tip::

   For field-based observations, use
   :func:`~fyst_trajectories.planning.plan_constant_el_scan` instead of manually
   constructing ``ConstantElScanConfig``. It auto-computes the azimuth range,
   duration, and number of scans from a ``FieldRegion``.

.. autoclass:: fyst_trajectories.patterns.PongScanConfig
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.patterns.PongAltAzScanConfig
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.patterns.DaisyScanConfig
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.patterns.DaisyAltAzScanConfig
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.patterns.SiderealTrackConfig
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.patterns.PlanetTrackConfig
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.patterns.SatelliteTrackConfig
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.patterns.LinearMotionConfig
   :members:
   :show-inheritance:

Pattern Classes
---------------

.. autoclass:: fyst_trajectories.patterns.ConstantElScanPattern
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.patterns.PongScanPattern
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.patterns.PongAltAzScanPattern
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.patterns.DaisyScanPattern
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.patterns.DaisyAltAzScanPattern
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.patterns.SiderealTrackPattern
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.patterns.PlanetTrackPattern
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.patterns.SatelliteTrackPattern
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.patterns.LinearMotionPattern
   :members:
   :show-inheritance:

Pattern Selection
-----------------

+-------------------+------------------+-------------------+
| Pattern           | Base Class       | Key Config Params |
+===================+==================+===================+
| ``sidereal``      | CelestialPattern | ``ra``, ``dec``   |
+-------------------+------------------+-------------------+
| ``planet``        | AltAzPattern     | ``body``          |
+-------------------+------------------+-------------------+
| ``satellite``     | AltAzPattern     | ``body``          |
+-------------------+------------------+-------------------+
| ``pong``          | CelestialPattern | ``width``,        |
|                   |                  | ``height``,       |
|                   |                  | ``spacing``       |
+-------------------+------------------+-------------------+
| ``pong_altaz``    | AltAzPattern     | ``az_center``,    |
|                   |                  | ``el_center``,    |
|                   |                  | ``width``,        |
|                   |                  | ``height``        |
+-------------------+------------------+-------------------+
| ``daisy``         | CelestialPattern | ``radius``,       |
|                   |                  | ``velocity``      |
+-------------------+------------------+-------------------+
| ``daisy_altaz``   | AltAzPattern     | ``az_center``,    |
|                   |                  | ``el_center``,    |
|                   |                  | ``radius``,       |
|                   |                  | ``velocity``      |
+-------------------+------------------+-------------------+
| ``constant_el``   | AltAzPattern     | ``az_start``,     |
|                   |                  | ``az_stop``,      |
|                   |                  | ``elevation``     |
+-------------------+------------------+-------------------+
| ``linear``        | AltAzPattern     | ``az_velocity``,  |
|                   |                  | ``el_velocity``   |
+-------------------+------------------+-------------------+

Registry Functions (Advanced)
-----------------------------

For interactive discovery or dynamic scenarios where pattern names are
determined at runtime::

    from fyst_trajectories import list_patterns, get_pattern
    from fyst_trajectories.patterns import PongScanConfig, get_pattern_for_config

    # List available patterns
    print(list_patterns())

    # Get pattern class by name (useful for plugins or config-driven selection)
    PatternClass = get_pattern("pong")

    # Get the pattern NAME from a config class (used by TrajectoryBuilder)
    pattern_name = get_pattern_for_config(PongScanConfig)   # "pong"

.. autofunction:: fyst_trajectories.patterns.list_patterns

.. autofunction:: fyst_trajectories.patterns.get_pattern

.. autofunction:: fyst_trajectories.patterns.get_pattern_for_config

.. autofunction:: fyst_trajectories.patterns.register_pattern

Geometry Helpers
----------------

.. autofunction:: fyst_trajectories.patterns.compute_pong_period

Boundary-Error Handling
-----------------------

When a trajectory exceeds telescope limits, a
:class:`~fyst_trajectories.exceptions.TargetNotObservableError` is raised
identifying the target and start time. Custom pattern authors should wrap
their bounds check for consistent error messages::

    from fyst_trajectories.patterns.utils import wrap_bounds_error

.. autofunction:: fyst_trajectories.patterns.utils.wrap_bounds_error
