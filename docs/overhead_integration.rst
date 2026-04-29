Pipeline Integration
====================

The overhead subpackage is a **planning-time** tool. This page explains where
it sits in the FYST observing pipeline, who owns each category of input, and
what downstream integrations exist today versus what is still on the roadmap.

Three Scheduler Layers
----------------------

FYST's observing workflow separates into three conceptual layers. The
overhead subpackage is Layer B.

.. list-table::
   :header-rows: 1
   :widths: 8 45 47

   * - Layer
     - Responsibility
     - Status
   * - A
     - **Survey scheduler.** Decide which patches are observable in a given
       observing window and in what order, using visibility, airmass, and
       elevation constraints. The astronomer interacts here.
     - ``fystplan`` (astroplan-based) lives in ``obs_implementation`` and is
       actively used.
   * - B
     - **Timeline generator.** Given a prioritized patch list, interleave
       science scans with retunes, pointing/focus/skydip/planet calibrations,
       and slews to produce a minute-by-minute schedule.
     - :func:`~fyst_trajectories.overhead.generate_timeline` in this
       subpackage. Realistic wall-clock budgets for survey simulations.
       **Not yet wired to Layer A** programmatically -- today it accepts
       ``ObservingPatch`` lists directly.
   * - C
     - **Execution orchestrator.** At run time, step through a Layer-B
       timeline, call
       :func:`~fyst_trajectories.overhead.schedule_to_trajectories` (or the
       appropriate ``plan_*_scan`` function) to regenerate each scan's
       motion from the stored metadata, stream az/el/time arrays to the
       OCS ``/path`` endpoint, and request calibration activities from OCS
       in the right order.
     - **Not built.** At execution time the Observatory Control System (OCS)
       exposes the trajectory/path endpoints, but nothing today consumes an
       ECSV timeline and orchestrates an observing night from it.

Where the Subpackage Fits
-------------------------

The longer-term integration picture looks like this::

   Astronomer                Layer A                    Layer B                     Layer C                  Hardware
   (patches, time)     (fystplan: visibility)    (overhead: schedule)        (OCS: execution)        (ACU + detectors)
        │                     │                          │                           │                       │
        └──patches────────────▶                          │                           │                       │
                              └──prioritized list────────▶                           │                       │
                                                         │                           │                       │
                                                         │  generate_timeline()      │                       │
                                                         │  ObservingTimeline        │                       │
                                                         │                           │                       │
                                                         └──write_timeline()─────────▶  (schedule.ecsv)      │
                                                                                     │                       │
                                                                                     │  for each block:      │
                                                                                     │    schedule_to_trajectories │
                                                                                     │    inject_retune()    │
                                                                                     │    POST /path ────────▶
                                                                                     │    POST /pointing ────▶
                                                                                     │    POST /focus ───────▶

Layer B produces an ECSV schedule that Layer C would consume. The schedule
records which patch is observed when, which calibrations fire, and where
slews are inserted -- it does **not** contain motion arrays. Layer C
regenerates trajectories from each block's stored metadata at execution
time (via :func:`~fyst_trajectories.overhead.schedule_to_trajectories` or
a direct ``plan_*_scan`` call), so that the same planning code runs at
both planning and execution. This is the *Planning = Execution* invariant.

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
