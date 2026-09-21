Calibration Night Planner
=========================

One night of solar-system calibration passes planned back to back. See
:doc:`../overhead_calibration_night` for the walkthrough.

Entry Point
-----------

.. autofunction:: fyst_trajectories.overhead.plan_calibration_night

Step Functions
--------------

The driver loops over these four; call them directly to step through a
night by hand.

.. autofunction:: fyst_trajectories.overhead.list_candidates

.. autofunction:: fyst_trajectories.overhead.plan_visit

.. autofunction:: fyst_trajectories.overhead.commit_visit

.. autofunction:: fyst_trajectories.overhead.advance_idle

.. autoclass:: fyst_trajectories.overhead.Candidate
   :members:

.. autoclass:: fyst_trajectories.overhead.VisitPlan
   :members:

State and Context
-----------------

.. autoclass:: fyst_trajectories.overhead.NightState
   :members:

.. autoclass:: fyst_trajectories.overhead.NightContext
   :members:

.. autoclass:: fyst_trajectories.overhead.VisitPlanner
   :members:

.. py:data:: fyst_trajectories.overhead.BOOTSTRAP_POSE

   The ``(az, el)`` a night starts from when none is given, matching the
   pose the offline scheduler bootstraps from.

Policies and Tables
-------------------

.. autoclass:: fyst_trajectories.overhead.CalibrationNightPolicy
   :members:

.. autoclass:: fyst_trajectories.overhead.TuningPolicy
   :members:

.. autoclass:: fyst_trajectories.overhead.ScanOverrides
   :members:

.. autoclass:: fyst_trajectories.overhead.ScanParameterTable
   :members:

.. autoclass:: fyst_trajectories.overhead.ElevationBin
   :members:

.. py:data:: fyst_trajectories.overhead.DEFAULT_SCAN_TABLES

   The shipped scan-parameter tables, keyed by body name with ``"default"``
   for the shared one: instrument-team commissioning defaults, pending
   on-sky testing.

.. autofunction:: fyst_trajectories.overhead.load_scan_tables

Selection
---------

.. autoclass:: fyst_trajectories.overhead.SelectionRule
   :members:

.. autofunction:: fyst_trajectories.overhead.select_priority

.. autoclass:: fyst_trajectories.overhead.ScriptedSelection
   :members:

Transitions
-----------

.. autofunction:: fyst_trajectories.overhead.plan_transition

.. autofunction:: fyst_trajectories.overhead.plan_escape

.. autoclass:: fyst_trajectories.overhead.Transition
   :members:

.. autoclass:: fyst_trajectories.overhead.DeferralReason
   :members:
   :undoc-members:

Reading a Night Back
--------------------

.. autofunction:: fyst_trajectories.overhead.summarize_calibration_night

.. autoclass:: fyst_trajectories.overhead.NightSummary
   :members:

.. autoclass:: fyst_trajectories.overhead.BodySummary
   :members:

.. autofunction:: fyst_trajectories.overhead.dispatch_sheet

.. autofunction:: fyst_trajectories.overhead.read_calibration_night_metadata

.. autoclass:: fyst_trajectories.overhead.CalibrationNightMetadata
   :members:

Block Records
-------------

.. autoclass:: fyst_trajectories.overhead.ScanGeometryRecord
   :members:

.. autoclass:: fyst_trajectories.overhead.TransitionRecord
   :members:
