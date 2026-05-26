Pipeline Integration
====================

The overhead subpackage is a **planning-time / simulation-only** tool. This
page explains where it sits in the FYST observing pipeline, who owns each
category of input, and what downstream integrations exist today.

Three Scheduler Layers
----------------------

FYST's observing workflow separates into three conceptual layers. The
overhead subpackage is an offline simulator that mimics what a real
Layer-B scheduler would produce; the live execution path uses the SO
operations stack.

.. list-table::
   :header-rows: 1
   :widths: 8 45 47

   * - Layer
     - Responsibility
     - Status
   * - A
     - **Survey/visibility scheduler.** Decide which patches are
       observable across many nights and enforce
       cadence/grouping/PWV constraints.
     - ``fystplan`` (astroplan-based) in ``obs_implementation``. Writes
       ``ObsUnit`` selections to Redis via ops-db-api. May additionally
       generate a TOAST-style master schedule for ``schedlib`` ingestion.
   * - B
     - **Tactical scheduler / command emission.** Given a prioritized
       block list for one night, interleave science with calibrations
       and slews and emit executable commands.
     - **Live:** ``simonsobs/scheduler`` (schedlib) -- needs a new
       ``policies/fyst.py`` (not yet written) that subclasses
       ``TelPolicy`` and uses fyst-trajectories for FYST astronomy.
       Emits a Python script of ``sorunlib`` calls.
       **Offline simulation:** this subpackage --
       :func:`~fyst_trajectories.overhead.generate_timeline` produces a
       realistic minute-by-minute ECSV for survey-design studies and
       hitmap accumulation in ``primecam_camera_mapping_simulations``.
       It does *not* drive live observing.
   * - C
     - **Execution.** Run the night against the ACU.
     - Nextline (line-by-line Python interpreter) executes the
       schedlib-emitted script; sorunlib dispatches OCS RPCs;
       ``ccatobs/pcs`` ACU agent posts trajectories to the FYST Go TCS;
       Go TCS uploads ProgramTrack to the Vertex ACU. PCS tasks
       (``pong_scan``, ``daisy_scan``, ``constant_el_scan``) call
       :func:`~fyst_trajectories.plan_pong_scan` etc. and
       :func:`~fyst_trajectories.to_path_format`.

Where the Subpackage Fits (current architecture)
------------------------------------------------

The overhead subpackage feeds the offline simulation lane. Live operations
use a separate stack::

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

   Astronomer          fystplan or            schedlib              nextline + sorunlib
   target list   ──▶   upstream master ──▶    policies/fyst.py ──▶  (line-by-line
                       schedule generator     (not yet written;     interpreter +
                                              uses fyst-trajec-     OCS RPC
                                              tories for FYST       dispatch)
                                              astronomy)                  │
                                              │                           ▼
                                              ▼                    pcs ACU agent
                                              Python script               │
                                              of sorunlib calls           ▼
                                                                   Go TCS /path
                                                                          │
                                                                          ▼
                                                                   Vertex ACU


fyst-trajectories sits *underneath* both lanes -- the core library (Site,
Coordinates, Patterns, Planning, Offsets, PrimeCam geometry) is imported
in both. The ``overhead`` subpackage itself is only used in the sim lane.

Planning = Execution invariant (still applies, narrower scope)
--------------------------------------------------------------

The invariant holds within fyst-trajectories: the same ``plan_*_scan``
functions are called by ``overhead.generate_timeline`` (sim lane) and by
live PCS ACU-agent tasks (ops lane). This guarantees the sim's wall-clock
prediction matches what the telescope will actually execute when the same
parameters are submitted by ``schedlib``. It is *not*, however, a contract
between the overhead-emitted ECSV and the live execution -- the ECSV is a
sim artifact, not the schedule the telescope reads.

Retunes are planning-side only
------------------------------

:func:`~fyst_trajectories.inject_retune` is used to mark sample-level
retune flags for accurate sim hitmaps (which exclude retune-flagged
samples from coverage). It is not called at execution time --
:func:`~fyst_trajectories.overhead.schedule_to_trajectories` does not call
it. At real execution, retunes are triggered by the Prime-Cam SMuRF
readout (via ``sorunlib.smurf`` calls in the Nextline-executed script),
which flags the data itself; the trajectory az/el is unaffected.

Source-CES is planner-only too
------------------------------

:func:`~fyst_trajectories.plan_source_ces` is part of the planning
subpackage but is **not** a supported overhead scan type. The
overhead simulator dispatches on ``pong`` / ``constant_el`` / ``daisy``
only; planet calibrations are handled as fixed-duration blocks
(``OverheadModel.planet_cal_duration``) without invoking any
``plan_*`` function. ``plan_source_ces`` exists so a future
``schedlib/policies/fyst.py`` can call it directly. See the
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
       site atmosphere, telescope settling, and calibration strategy -- not
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
   called from a live observing loop. At execution time the orchestrator
   should read a pre-computed ECSV, then regenerate motion arrays from
   the stored ``ScanBlock`` metadata -- not re-run the scheduler
   mid-night.

Related Reading
---------------

* :doc:`overhead_quickstart` -- minimal working example.
* :doc:`overhead_timeline` -- ``generate_timeline`` walk-through with
  per-patch and per-calibration breakdowns.
* :doc:`overhead_model` -- field-by-field reference for ``OverheadModel``
  and ``CalibrationPolicy``.
* :doc:`overhead_io` -- ECSV column schema and TOAST compatibility notes.
