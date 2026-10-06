Planning Package
================

High-level planning functions that translate astronomer-friendly inputs
into pattern configurations and trajectories. Worked examples for every
scan type are in :doc:`../planning`.

.. automodule:: fyst_trajectories.planning
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: PongComputedParams, PongAltAzComputedParams,
                     ConstantElComputedParams,
                     DaisyComputedParams, DaisyAltAzComputedParams,
                     SourceCESComputedParams,
                     ComputedParams, validate_computed_params

Computed Parameter Schemas
--------------------------

Each planner function returns a :class:`ScanBlock` whose
``computed_params`` attribute follows a scan-type-specific schema.

.. autoclass:: fyst_trajectories.planning.PongComputedParams
   :members:

.. autoclass:: fyst_trajectories.planning.PongAltAzComputedParams
   :members:

.. autoclass:: fyst_trajectories.planning.ConstantElComputedParams
   :members:

.. autoclass:: fyst_trajectories.planning.DaisyComputedParams
   :members:

.. autoclass:: fyst_trajectories.planning.DaisyAltAzComputedParams
   :members:

.. autoclass:: fyst_trajectories.planning.SourceCESComputedParams
   :members:

.. py:data:: ComputedParams
   :value: PongComputedParams | PongAltAzComputedParams |
           ConstantElComputedParams | DaisyComputedParams |
           DaisyAltAzComputedParams | SourceCESComputedParams

   Umbrella union alias for the ``computed_params`` mapping carried on
   :class:`ScanBlock`. The concrete schema is the one the planner that
   built the block returns.

.. autofunction:: fyst_trajectories.planning.validate_computed_params

.. note::

   The overhead-side :func:`~fyst_trajectories.overhead.validate_scan_params`
   accepts ``"source_ces"`` (for planet-calibration passes recorded as
   ``SourceCESScanParams``), which ``validate_computed_params`` refuses; the
   two validators track deliberately different scan-type sets.
