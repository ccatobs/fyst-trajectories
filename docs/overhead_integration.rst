Offline Simulation and Live Operations
======================================

The ``overhead`` subpackage is a **planning-time simulator**. It produces
a realistic minute-by-minute observing night for survey design; it never
drives a telescope. This page places that lane beside the live one and
says who owns each category of input.

FYST hosts two instrument pipelines, Prime-Cam on Simons
Observatory-derived control software and CHAI on the KOSMA stack. This
library serves the Prime-Cam lane, and the framing below is written for
it.

Where the Subpackage Fits
-------------------------

The subpackage feeds the offline lane; live operations run a separate
path. Its input is a list of :class:`~fyst_trajectories.overhead.ObservingPatch`
objects built in Python, by hand or from a source list such as the
bundled CSV example::

   OFFLINE SIM LANE (where this subpackage lives):
   ─────────────────────────────────────────────────

   ObservingPatch     generate_timeline()           write_timeline()
   objects       ──▶  ObservingTimeline      ──▶    schedule.ecsv
                                                          │
                                                          ▼
                                          coverage-simulation tooling
                                          (hitmaps, coverage and cadence studies)


   LIVE OPS LANE (Prime-Cam, what actually drives the telescope):
   ──────────────────────────────────────────────────────────────

   long-term schedule ──▶ observatory scheduling layer ──▶ ACU agent
                          (dispatches the typed scan            │
                           tasks; each calls                    ▼
                           plan_*_scan at dispatch)   telescope control system
                                                                │
                                                                ▼
                                                               ACU


fyst-trajectories sits *underneath* both lanes: the core library is
imported in both, the ``overhead`` subpackage only in the sim lane, where
the diagram's hitmap accumulation is
:func:`~fyst_trajectories.overhead.accumulate_hitmaps` (the ``overhead``
extra supplies its ``healpy`` dependency).

**Planning = execution, within fyst-trajectories.** The same
``plan_*_scan`` functions are called by ``overhead.generate_timeline``
(sim lane) and by the live typed scan tasks (ops lane), so the sim's
wall-clock prediction matches what the telescope executes when the same
parameters are dispatched. It is not a contract between the
overhead-emitted ECSV and live execution: the ECSV is a sim artifact, not
the schedule the telescope reads.

**Retunes are planning-side only.** Nothing on the live path reads the
sample-level retune flags: the ``/path`` payload carries positions and
velocities only, and
:func:`~fyst_trajectories.overhead.schedule_to_trajectories` never calls
:func:`~fyst_trajectories.trajectory_utils.inject_retune` when it rebuilds
a block.

Parameter Ownership
-------------------

Three categories of input shape the outputs. Each has a natural owner; a
user should not be guessing at values they do not control.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Input
     - Natural owner
     - Examples
   * - **Layer 1 (in-scan)**: detector timing
     - Prime-Cam / instrument team
     - ``retune_interval``, ``retune_duration``, ``n_modules``. These
       describe KID thermal drift and readout wall-time, not astronomy,
       and parameterize
       :func:`~fyst_trajectories.trajectory_utils.inject_retune` on a
       single trajectory rather than the timeline. The timeline's own
       ``OverheadModel.retune_duration`` has the same owner; see
       :doc:`overhead_model` for how the two differ.
   * - **Layer 2 (block-level)**: calibration cadences and activity durations
     - Operations / commissioning team
     - ``CalibrationPolicy`` cadences and ``OverheadModel`` durations,
       excepting the retune fields (``retune_cadence``,
       ``retune_duration``), which stay with Layer 1's owner. Reflect
       site atmosphere, telescope settling, and calibration strategy, not
       per-proposal knobs.
   * - **Per-proposal**: what to observe
     - Astronomer
     - ``ObservingPatch`` geometry, ``scan_type``, ``velocity``,
       ``elevation`` (for constant-el scans), time window.

:func:`~fyst_trajectories.overhead.generate_timeline` accepts bare
``OverheadModel()`` / ``CalibrationPolicy()`` defaults, but relying on those
hides physical assumptions and should be avoided outside of quick
exploratory scripts.

See also
--------

* :doc:`overhead_quickstart` - the minimal working example.
* :doc:`planning` - the ``plan_*_scan`` functions both lanes call.
* :ref:`planet-cal-source-ces` - planet calibrations as real
  source-CES passes, reconstructable from the timeline.
