Overhead Simulation
===================

Rebuilds the trajectory behind each timeline block, accumulates HEALPix
coverage maps, and totals a timeline's time budget. For worked examples
see :doc:`../overhead_timeline`.

.. autofunction:: fyst_trajectories.overhead.schedule_to_trajectories

.. autofunction:: fyst_trajectories.overhead.accumulate_hitmaps

.. autofunction:: fyst_trajectories.overhead.compute_budget

Refusals
--------

Both are :class:`~fyst_trajectories.exceptions.PointingError` subclasses,
raised while rebuilding a recorded block and logged and skipped by
:func:`~fyst_trajectories.overhead.schedule_to_trajectories`.

.. autoclass:: fyst_trajectories.overhead.ScanParamsSchemaError
   :show-inheritance:

.. autoclass:: fyst_trajectories.overhead.BlockNotReconstructableError
   :show-inheritance:
