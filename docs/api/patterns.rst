Patterns Package
================

Scan pattern implementations for telescope trajectory generation.
``TrajectoryBuilder`` builds a trajectory from a config object, inferring
the pattern type from the config class::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import PongScanConfig, TrajectoryBuilder

    start_time = Time("2026-03-15T01:00:00", scale="utc")
    config = PongScanConfig(
        timestep=0.1, width=2.0, height=2.0, spacing=0.1,
        velocity=0.4, num_terms=4, angle=0.0,
    )

    trajectory = (
        TrajectoryBuilder(get_fyst_site())
        .at(ra=180.0, dec=-30.0)
        .with_config(config)
        .duration(300.0)
        .starting_at(start_time)
        .build()
    )

TrajectoryBuilder
-----------------

.. autoclass:: fyst_trajectories.patterns.TrajectoryBuilder
   :members:
   :undoc-members:

**Detector offset support** - the same builder, with the boresight
offset so module ``i1`` tracks the target::

    from fyst_trajectories.primecam import get_primecam_offset

    trajectory = (
        TrajectoryBuilder(get_fyst_site())
        .at(ra=180.0, dec=-30.0)
        .with_config(config)
        .for_detector(get_primecam_offset("i1"))
        .duration(60.0)
        .starting_at(start_time)
        .build()
    )

Base Classes
------------

The split is what the builder requires of you:
:class:`~fyst_trajectories.patterns.CelestialPattern` subclasses take a
sky center via ``.at(ra, dec)`` and need ``.starting_at()``;
:class:`~fyst_trajectories.patterns.AltAzPattern` subclasses skip
``.at()``, though the planet and satellite trackers still need
``.starting_at()`` for their ephemerides.

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

   For field-based observations, prefer
   :func:`~fyst_trajectories.planning.plan_constant_el_scan` over building a
   ``ConstantElScanConfig`` by hand; see :doc:`../planning`.

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

Each pattern is selected by its config class:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Pattern
     - Config class
   * - ``sidereal``
     - ``SiderealTrackConfig``
   * - ``planet``
     - ``PlanetTrackConfig``
   * - ``satellite``
     - ``SatelliteTrackConfig``
   * - ``pong``
     - ``PongScanConfig``
   * - ``pong_altaz``
     - ``PongAltAzScanConfig``
   * - ``daisy``
     - ``DaisyScanConfig``
   * - ``daisy_altaz``
     - ``DaisyAltAzScanConfig``
   * - ``constant_el``
     - ``ConstantElScanConfig``
   * - ``linear``
     - ``LinearMotionConfig``

Registry and Helpers
--------------------

For interactive discovery or config-driven selection at runtime
(:doc:`../trajectory_examples` builds a trajectory this way)::

    from fyst_trajectories import get_pattern
    from fyst_trajectories.patterns import PongScanConfig, get_pattern_for_config

    PatternClass = get_pattern("pong")                      # name -> class
    pattern_name = get_pattern_for_config(PongScanConfig)   # config -> name

.. autofunction:: fyst_trajectories.patterns.list_patterns

.. autofunction:: fyst_trajectories.patterns.get_pattern

.. autofunction:: fyst_trajectories.patterns.get_pattern_for_config

.. autofunction:: fyst_trajectories.patterns.register_pattern

.. autofunction:: fyst_trajectories.patterns.compute_pong_period

.. autofunction:: fyst_trajectories.patterns.rewrap_trajectory_azimuth

A trajectory that exceeds the telescope limits raises
:class:`~fyst_trajectories.exceptions.TargetNotObservableError`; pattern
authors get that message by wrapping their own bounds check.

.. autofunction:: fyst_trajectories.patterns.utils.wrap_bounds_error
