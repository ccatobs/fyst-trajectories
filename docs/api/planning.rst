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

.. autofunction:: fyst_trajectories.planning.validate_computed_params

.. note::

   ``"source_ces"`` is intentionally not accepted by
   ``validate_computed_params``. :func:`plan_source_ces` is
   planner-only and self-validates against
   ``SourceCESComputedParams.__required_keys__`` directly.
   The overhead-side :func:`~fyst_trajectories.overhead.validate_scan_params`
   *does* accept ``"source_ces"`` (for planet-calibration passes recorded as
   ``SourceCESScanParams``); the two validators track deliberately different
   scan-type sets. See :ref:`scan-type-vocabularies`.
