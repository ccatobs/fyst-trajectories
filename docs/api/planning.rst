Planning Module
===============

High-level planning functions that translate astronomer-friendly inputs
into pattern configurations and trajectories.

.. automodule:: fyst_trajectories.planning
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: PongComputedParams, ConstantElComputedParams,
                     DaisyComputedParams, SourceCESComputedParams,
                     ComputedParams, validate_computed_params

Computed Parameter Schemas
--------------------------

Each planner function returns a :class:`ScanBlock` whose
``computed_params`` attribute follows a scan-type-specific schema.

.. autoclass:: fyst_trajectories.planning.PongComputedParams
   :members:

.. autoclass:: fyst_trajectories.planning.ConstantElComputedParams
   :members:

.. autoclass:: fyst_trajectories.planning.DaisyComputedParams
   :members:

.. autoclass:: fyst_trajectories.planning.SourceCESComputedParams
   :members:

.. autofunction:: fyst_trajectories.planning.validate_computed_params

.. note::

   ``"source_ces"`` is intentionally not accepted by
   ``validate_computed_params``. :func:`plan_source_ces` is
   planner-only (its consumer is the FYST policy in the ``pcam_gen_schedule`` schedlib fork,
   not the in-tree overhead simulator) and self-validates against
   :attr:`SourceCESComputedParams.__required_keys__` directly.
