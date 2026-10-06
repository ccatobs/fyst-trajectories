Library API Reference
=====================

The library tier's module reference: the core stack first (site through
dispatch, bottom to top), then observability, the sun-avoidance models,
and visualization. The offline simulator has its own reference,
:doc:`overhead_index`.

The package root re-exports every public name of these modules except the
sun-avoidance models (``sun_models``), the seam extension protocols
(``sun_protocols``), the planning footprint transforms, the
pattern-authoring helpers documented from ``patterns.utils`` and
``patterns.turnarounds``, and the visualization functions, which are
imported from their own modules; the offline simulator is imported from
``fyst_trajectories.overhead``.

.. toctree::
   :maxdepth: 2

   site
   coordinates
   trajectory
   trajectory_utils
   retune
   exceptions
   patterns
   offsets
   planning
   dispatch
   observability
   sun_models
   visualization
