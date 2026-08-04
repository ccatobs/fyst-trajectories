Overhead Scheduler Internals
============================

The :func:`~fyst_trajectories.overhead.generate_timeline` entry point
is a thin wrapper around the
:class:`~fyst_trajectories.overhead.scheduler.Scheduler` class, which
orchestrates a sequence of phase objects. This page documents the
phase API for advanced users who want to extend scheduling behavior
(e.g., priority-weighted scheduling, lookahead, multi-night
stitching).

.. note::

   The ``generate_timeline`` signature and return type are stable; the
   phase API below may evolve.

Scheduler
---------

.. autoclass:: fyst_trajectories.overhead.scheduler.Scheduler
   :members:

State Objects
-------------

.. autoclass:: fyst_trajectories.overhead.scheduler.SchedulerState
   :members:

.. autoclass:: fyst_trajectories.overhead.scheduler.SchedulerContext
   :members:

Phases
------

Each phase reads a
:class:`~fyst_trajectories.overhead.scheduler.SchedulerState` /
:class:`~fyst_trajectories.overhead.scheduler.SchedulerContext` pair
and returns a
:class:`~fyst_trajectories.overhead.scheduler.PhaseResult` with
emitted blocks and the updated state.

.. autoclass:: fyst_trajectories.overhead.scheduler.Phase
   :members:

.. autoclass:: fyst_trajectories.overhead.scheduler.PhaseResult
   :members:

.. autoclass:: fyst_trajectories.overhead.scheduler.CalibrationPhase
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.overhead.scheduler.PatchSelectionPhase
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.overhead.scheduler.SlewPhase
   :members:
   :show-inheritance:

.. autoclass:: fyst_trajectories.overhead.scheduler.ScienceScanPhase
   :members:
   :show-inheritance:
