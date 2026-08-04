Overhead Timeline Generation
============================

Sequences science scans with calibration breaks and slew overhead
over a given observing window. See :doc:`../overhead_timeline` for
worked examples.

.. autofunction:: fyst_trajectories.overhead.generate_timeline

Constraints
-----------

.. note::

   Constraints are an extension API for the scheduling internals;
   :func:`~fyst_trajectories.overhead.generate_timeline` builds a
   default set of (elevation + sun) constraints from the supplied
   :class:`~fyst_trajectories.site.Site` envelopes. Most callers do not
   need to construct or pass these directly.

.. autoclass:: fyst_trajectories.overhead.Constraint
   :members:

.. autoclass:: fyst_trajectories.overhead.ElevationConstraint
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.overhead.SunAvoidanceConstraint
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.overhead.MoonAvoidanceConstraint
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.overhead.MinDurationConstraint
   :members:
   :show-inheritance:

Utilities
---------

.. autofunction:: fyst_trajectories.overhead.estimate_slew_time

.. autofunction:: fyst_trajectories.overhead.get_observable_windows

.. autofunction:: fyst_trajectories.overhead.get_transit_time

.. autofunction:: fyst_trajectories.overhead.get_max_elevation

.. autofunction:: fyst_trajectories.overhead.compute_nasmyth_rotation
