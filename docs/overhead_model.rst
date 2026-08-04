Overhead Model and Calibration Policy
======================================

Two configuration objects control overhead timing:
:class:`~fyst_trajectories.overhead.OverheadModel` for activity durations, and
:class:`~fyst_trajectories.overhead.CalibrationPolicy` for how often each calibration
is performed.

OverheadModel
-------------

Controls the duration of each non-science activity::

    from fyst_trajectories.overhead import OverheadModel

    model = OverheadModel(
        retune_duration=5.0,          # KID probe tone reset (s)
        pointing_cal_duration=180.0,  # pointing correction scan (s)
        focus_duration=300.0,         # focus check (s)
        skydip_duration=300.0,        # elevation nod (s)
        planet_cal_duration=600.0,    # planet calibration scan (s)
        beam_map_duration=600.0,      # beam-map scan (same default as planet cal)
        settle_time=5.0,              # post-slew settling (s)
        min_scan_duration=60.0,       # minimum useful science scan (s)
        max_scan_duration=3600.0,     # forced split threshold (s)
    )

``min_scan_duration`` prevents short, wasteful scans. ``max_scan_duration``
forces long observations to split into sub-scans with retune breaks.
``beam_map_duration`` defaults to the same value as
``planet_cal_duration`` because beam maps typically run on the same
planet targets, but can be tuned independently when the science goals
demand a different map size or velocity.

CalibrationPolicy
-----------------

Controls *when* each calibration type is triggered. Cadences are in seconds.
A cadence of 0 keeps that calibration permanently due: retune then fires
immediately before every science subscan (plus once at startup) and never on
an idle tick, while every other calibration type fires on each scheduler
iteration, idle ticks included. A cadence of ``None`` (valid only for
``beam_map_cadence``) disables automatic scheduling for that calibration type
entirely::

    from fyst_trajectories.overhead import CalibrationPolicy

    policy = CalibrationPolicy(
        retune_cadence=0.0,           # before every science subscan
        pointing_cadence=3600.0,      # every 1 hour
        focus_cadence=7200.0,         # every 2 hours
        skydip_cadence=10800.0,       # every 3 hours
        planet_cal_cadence=43200.0,   # every 12 hours
        beam_map_cadence=None,        # default: manual injection only
        planet_targets=("jupiter", "saturn", "mars", "uranus", "neptune"),
        planet_min_elevation=20.0,    # planet must be above this
        planet_cal_scan=False,        # False = parked; True = source-CES passes
        planet_cal_passes=3,          # passes per planet cal when scanning
        planet_cal_el_step=None,      # None = planner default (footprint extent)
        planet_cal_footprint="c",     # Prime-Cam module tag the passes tile
    )

Planet calibrations and beam maps are only scheduled when at least one
planet target in ``planet_targets`` is above ``planet_min_elevation``.

The ``planet_cal_scan`` / ``planet_cal_passes`` / ``planet_cal_el_step`` /
``planet_cal_footprint`` group controls how a planet calibration is
realised (see :ref:`planet-cal-source-ces` below). Like the cadences and
durations, these are commissioning-era placeholders for the
instrument/operations team to confirm.

Scheduling Beam Maps
~~~~~~~~~~~~~~~~~~~~

``BEAM_MAP`` is a :class:`~fyst_trajectories.overhead.CalibrationType`
with its own cadence (``CalibrationPolicy.beam_map_cadence``) and
duration (``OverheadModel.beam_map_duration``). It is off the automatic
schedule by default (``beam_map_cadence=None``); set a positive cadence
to opt in. Beam maps then use the same ``planet_targets`` machinery as
``planet_cal``.

**Example: 6-hour beam-map cadence**

.. code-block:: python

    # Beam map every 6 hours using the configured planet targets.
    policy = CalibrationPolicy(beam_map_cadence=21600.0)

.. _planet-cal-source-ces:

Planet Calibrations as Source-CES Scans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default a planet calibration is a single fixed-duration parked block
(``OverheadModel.planet_cal_duration``): the telescope holds its current
pose while the calibration runs, and no scan geometry is recorded.

Setting ``planet_cal_scan=True`` instead plans each planet calibration as
a multi-pass source-CES sequence via
:func:`~fyst_trajectories.planning.plan_source_ces_passes`, anchored at
the scheduler clock. The planet is dragged across the Prime-Cam focal
plane at a fixed boresight elevation, once per pass, with the passes
stepped in elevation so they run sequentially:

.. code-block:: python

    policy = CalibrationPolicy(
        planet_cal_scan=True,
        planet_cal_passes=3,          # three drift passes per calibration
        planet_cal_el_step=None,      # None = planner default spacing
        planet_cal_footprint="c",     # tile Prime-Cam module "c"
    )

Each pass becomes its own calibration block (``scan_type="planet_cal"``),
carrying the full source-CES parameters in
``metadata["scan_params"]`` (a
:class:`~fyst_trajectories.overhead.SourceCESScanParams`) and the true
scan start in ``metadata["t0_scan"]``. If the sequence is not feasible at
the anchor (the planet never reaches the required geometry in the search
window), the calibration is skipped and left due, so it is retried on a
later scheduler iteration, exactly like a planet cal with no visible
planet.

Rebuild the pass trajectories with
``schedule_to_trajectories(timeline, science_only=False)``; the default
``science_only=True`` returns science blocks only. An explicit
``planet_cal_el_step`` smaller than the footprint's elevation extent makes
adjacent pass windows overlap in time, and the planner emits a
:class:`~fyst_trajectories.exceptions.PointingWarning` when they do.
