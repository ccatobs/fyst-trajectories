Pipeline Integration
====================

The overhead subpackage is a **planning-time / simulation-only** tool. This
page explains where it sits in the FYST observing pipeline, who owns each
category of input, and what downstream integrations exist today.

FYST hosts two instrument pipelines: Prime-Cam (SO-derived) and CHAI
(KOSMA-derived). This library and the typed PCS scan tasks serve the
Prime-Cam lane; the framing below is written for that lane.

Three Scheduler Layers
----------------------

FYST's observing workflow separates into three conceptual layers. The
overhead subpackage is an offline simulator that mimics what the
tactical layer produces; it does not drive live observing.

.. list-table::
   :header-rows: 1
   :widths: 8 45 47

   * - Layer
     - Responsibility
     - Status
   * - A
     - **Survey/visibility scheduler.** Decide which observing units are
       observable across many nights and enforce
       cadence/grouping/PWV constraints.
     - An upstream survey planner selects observing units across nights
       and records the selection for the tactical layer.
   * - B
     - **Tactical scheduler / command emission.** Given a prioritized
       block list for one night, interleave science with calibrations
       and slews and emit executable commands.
     - **Live:** the observatory scheduling layer expands the selected
       observing units and dispatches the typed PCS scan tasks
       (``pong_scan``, ``daisy_scan``, ``constant_el_scan``,
       ``source_scan``), each of which calls the ``plan_*_scan``
       planners at dispatch time.
       **Offline simulation:** this subpackage,
       :func:`~fyst_trajectories.overhead.generate_timeline` produces a
       realistic minute-by-minute ECSV for survey-design studies and
       hitmap accumulation in ``primecam_camera_mapping_simulations``.
       It does *not* drive live observing.
   * - C
     - **Execution.** Run the night against the ACU.
     - The ``ccatobs/pcs`` ACU agent posts each trajectory to the
       telescope control system, which uploads the motion program to the
       Vertex ACU. The typed PCS scan tasks call
       :func:`~fyst_trajectories.plan_pong_scan` etc. and
       :func:`~fyst_trajectories.to_path_payload`.

Where the Subpackage Fits (current architecture)
------------------------------------------------

The overhead subpackage feeds the offline simulation lane. Live operations
run through a separate path::

   OFFLINE SIM LANE (where this subpackage lives):
   ─────────────────────────────────────────────────

   Patches            generate_timeline()           write_timeline()
   (in-process)  ──▶  ObservingTimeline      ──▶    schedule.ecsv
                                                          │
                                                          ▼
                                          primecam_camera_mapping_simulations
                                          (hitmap accumulation, coverage studies,
                                           cadence/efficiency comparisons)


   LIVE OPS LANE (Prime-Cam, what actually drives the telescope):
   ──────────────────────────────────────────────────────────────

   long-term schedule ──▶ observatory scheduling layer ──▶ PCS ACU agent
                          (dispatches the typed scan            │
                           tasks; each calls                    ▼
                           plan_*_scan at dispatch)   telescope control system
                                                                │
                                                                ▼
                                                               ACU


fyst-trajectories sits *underneath* both lanes. The core library (Site,
Coordinates, Patterns, Planning, Offsets, PrimeCam geometry) is imported
in both. The ``overhead`` subpackage itself is only used in the sim lane.

Planning = Execution invariant (still applies, narrower scope)
--------------------------------------------------------------

The invariant holds within fyst-trajectories: the same ``plan_*_scan``
functions are called by ``overhead.generate_timeline`` (sim lane) and by
the live PCS scan tasks (ops lane). This guarantees the sim's wall-clock
prediction matches what the telescope will actually execute when the same
parameters are dispatched. It is *not*, however, a contract
between the overhead-emitted ECSV and the live execution. The ECSV is a
sim artifact, not the schedule the telescope reads.

Retunes are planning-side only
------------------------------

:func:`~fyst_trajectories.inject_retune` is used to mark sample-level
retune flags for accurate sim hitmaps (which exclude retune-flagged
samples from coverage). It is not called at execution time;
:func:`~fyst_trajectories.overhead.schedule_to_trajectories` does not call
it. At real execution, retunes are triggered by the Prime-Cam detector
readout, which flags the data itself; the trajectory az/el is unaffected.

Source-CES is planner-only too
------------------------------

:func:`~fyst_trajectories.plan_source_ces` is planner-only here: it is
**not** a supported overhead scan type. The overhead simulator dispatches
on ``pong`` / ``constant_el`` / ``daisy`` only; planet calibrations are
handled as fixed-duration blocks (``OverheadModel.planet_cal_duration``)
without invoking any ``plan_*`` function. At dispatch time the live PCS
``source_scan`` task consumes ``plan_source_ces`` instead. See the
"Planning a Source CES" section in :doc:`planning` for details.

Parameter Ownership
-------------------

A timeline is driven by three categories of input. Each has a natural owner;
end users of the subpackage should not be guessing at values they do not
control.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Input
     - Natural owner
     - Examples
   * - **Layer 1 (in-scan)**: detector timing
     - Prime-Cam / instrument team
     - ``retune_interval``, ``retune_duration``, ``n_modules``. These
       describe KID thermal drift and readout wall-time, not astronomy.
   * - **Layer 2 (block-level)**: calibration cadences and activity durations
     - Operations / commissioning team
     - ``CalibrationPolicy`` cadences, ``OverheadModel`` durations. Reflect
       site atmosphere, telescope settling, and calibration strategy, not
       per-proposal knobs.
   * - **Per-proposal**: what to observe
     - Astronomer
     - ``ObservingPatch`` geometry, ``scan_type``, ``velocity``,
       ``elevation`` (for constant-el scans), time window.

In practice, the FYST team will publish canonical ``OverheadModel`` and
``CalibrationPolicy`` presets (commissioning vs. survey vs. deep-field) so
that proposal authors do not need to invent cadence numbers themselves.
:func:`~fyst_trajectories.overhead.generate_timeline` accepts bare
``OverheadModel()`` / ``CalibrationPolicy()`` defaults, but relying on those
hides physical assumptions and should be avoided outside of quick
exploratory scripts.

.. note::

   The overhead subpackage is a planning-time tool and should **not** be
   called from a live observing loop. Nothing live reads the ECSV it emits;
   it is an offline artifact. Coverage tooling regenerates motion arrays
   from the stored ``TimelineBlock`` metadata via
   :func:`~fyst_trajectories.overhead.schedule_to_trajectories` rather than
   re-running the scheduler.

Related Reading
---------------

* :doc:`overhead_quickstart` - minimal working example.
* :doc:`overhead_timeline` - ``generate_timeline`` walk-through with
  per-patch and per-calibration breakdowns.
* :doc:`overhead_model` - field-by-field reference for ``OverheadModel``
  and ``CalibrationPolicy``.
* :doc:`overhead_io` - ECSV column schema and TOAST compatibility notes.
