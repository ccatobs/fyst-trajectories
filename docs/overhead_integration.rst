Pipeline Integration
====================

The overhead subpackage is a **planning-time / simulation-only** tool. This
page explains where it sits in the FYST observing pipeline, who owns each
category of input, and what downstream integrations exist today.

FYST hosts two instrument pipelines: Prime-Cam, whose control software is
derived from the Simons Observatory stack, and CHAI, on the KOSMA stack.
This library and the typed PCS scan tasks serve the Prime-Cam lane; the
framing below is written for that lane.

Three Scheduler Layers
----------------------

FYST's observing workflow separates into three conceptual layers.

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
     - Outside this library. An upstream survey planner records its
       selection for the tactical layer.
   * - B
     - **Tactical scheduler / command emission.** Given a prioritized
       block list for one night, interleave science with calibrations
       and slews and emit executable commands.
     - **Live:** the observatory scheduling layer expands the selected
       observing units and dispatches the typed PCS scan tasks
       (``pong_scan``, ``daisy_scan``, ``constant_el_scan``,
       ``source_scan``), each of which calls a fyst-trajectories planner
       at dispatch time.
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
       :func:`~fyst_trajectories.planning.plan_pong_scan` etc. and
       :func:`~fyst_trajectories.trajectory_utils.to_path_payload`.

Where the Subpackage Fits
-------------------------

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

Planning = Execution invariant
------------------------------

The invariant holds within fyst-trajectories: the same ``plan_*_scan``
functions are called by ``overhead.generate_timeline`` (sim lane) and by
the live PCS scan tasks (ops lane). This guarantees the sim's wall-clock
prediction matches what the telescope will actually execute when the same
parameters are dispatched. It is not a contract between the
overhead-emitted ECSV and live execution: the ECSV is a sim artifact, not
the schedule the telescope reads.

Retunes are planning-side only
------------------------------

:func:`~fyst_trajectories.trajectory_utils.inject_retune` is used to mark
sample-level retune flags for accurate sim hitmaps (which exclude
retune-flagged samples from coverage). It is not called at execution time;
:func:`~fyst_trajectories.overhead.schedule_to_trajectories` does not call
it. Nothing on the live path reads those flags: the ``/path`` payload posted
to the telescope control system is a list of ``[t, az, el, az_vel,
el_vel]`` rows and carries no retune information. Detector retunes at
execution are dispatched separately by the observatory scheduling layer;
the trajectory az/el is unaffected either way.

Planet calibrations and source-CES
----------------------------------

The overhead simulator emits **science** blocks on
``pong`` / ``constant_el`` / ``daisy`` only. Planet calibrations are
fixed-duration parked blocks by default
(``OverheadModel.planet_cal_duration``), with no scan geometry to
reconstruct.

Setting ``CalibrationPolicy.planet_cal_scan`` instead plans each planet
calibration as a real multi-pass source-CES sequence via
:func:`~fyst_trajectories.planning.plan_source_ces_passes`, anchored at the
scheduler clock: one calibration block per pass, each recording the full
source-CES parameters in ``metadata["scan_params"]`` and the true scan
start in ``metadata["t0_scan"]`` (see :ref:`planet-cal-source-ces` in
:doc:`overhead_model`).

Those recorded parameters make each pass reconstructable from the
timeline. :func:`~fyst_trajectories.overhead.schedule_to_trajectories`
returns science blocks only by default (``science_only=True``), but with
``science_only=False`` it rebuilds every source-CES planet-cal block from
its recorded parameters via
:func:`~fyst_trajectories.planning.plan_source_ces`::

    from fyst_trajectories.overhead import schedule_to_trajectories

    pairs = schedule_to_trajectories(timeline, science_only=False)

Calibration blocks with no ``scan_params`` (parked planet cals, retunes,
pointing/focus/skydip) carry no trajectory to rebuild and are skipped
silently. At dispatch time the live PCS ``source_scan`` task consumes
:func:`~fyst_trajectories.planning.plan_source_ces` directly. See the
"Planning a Source CES" section in :doc:`planning` for details.

.. _scan-type-vocabularies:

Scan-type vocabularies
----------------------

Six constructs across the planning and overhead subpackages enumerate scan
types at different granularities and for different purposes. Their
near-identical names invite the assumption that they should be equal; they
should not. Each is individually correct, and the table below is the map.

.. list-table::
   :header-rows: 1
   :widths: 32 8 12 48

   * - Construct
     - Members
     - ``source_ces``?
     - Role
   * - ``_PATTERN_REGISTRY`` (:func:`~fyst_trajectories.patterns.list_patterns`)
     - 9
     - no
     - Every buildable scan pattern. ``source_ces`` is planner-only
       (:func:`~fyst_trajectories.planning.plan_source_ces`), not a
       registered pattern.
   * - ``ComputedParams`` union
     - 6
     - yes
     - Static type of
       :attr:`ScanBlock.computed_params <fyst_trajectories.planning.ScanBlock.computed_params>`;
       one member per ``plan_*`` return schema, including
       ``SourceCESComputedParams``.
   * - ``_SCAN_TYPE_TO_KEYS``
     - 5
     - no
     - Call-site table for
       :func:`~fyst_trajectories.planning.validate_computed_params`;
       ``source_ces`` self-validates inside
       :func:`~fyst_trajectories.planning.plan_source_ces`.
   * - ``ObservingPatch`` scan-type guard
     - 3
     - no
     - The science scan types the offline simulator emits directly
       (``constant_el`` / ``pong`` / ``daisy``).
   * - ``ScanParamsDict`` union
     - 3
     - no
     - Static type of
       :attr:`ObservingPatch.scan_params <fyst_trajectories.overhead.ObservingPatch.scan_params>`;
       the three science schemas, excluding ``SourceCESScanParams``.
   * - ``_SCAN_TYPE_TO_SCAN_PARAM_KEYS``
     - 4
     - yes
     - Call-site table for
       :func:`~fyst_trajectories.overhead.validate_scan_params`; adds
       ``source_ces`` so planet-calibration passes recorded as
       ``SourceCESScanParams`` validate.

Each **union** (``ComputedParams``, ``ScanParamsDict``) tracks a
dataclass/TypedDict attribute annotation, the static shape of a
``computed_params`` or ``scan_params`` mapping. Each **table**
(``_SCAN_TYPE_TO_KEYS``, ``_SCAN_TYPE_TO_SCAN_PARAM_KEYS``) tracks the call
sites a runtime validator accepts. The two subpackages are mirror-inverted
about ``source_ces``: on the planning side the union is the superset of its
table (6 vs 5, the union adds ``source_ces``), while on the overhead side the
table is the superset of its union (4 vs 3, the table adds ``source_ces``).
The inversion is intentional and must not be equalized.

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

:func:`~fyst_trajectories.overhead.generate_timeline` accepts bare
``OverheadModel()`` / ``CalibrationPolicy()`` defaults, but relying on those
hides physical assumptions and should be avoided outside of quick
exploratory scripts.

.. note::

   Coverage tooling regenerates motion arrays from the stored
   ``TimelineBlock`` metadata via
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
