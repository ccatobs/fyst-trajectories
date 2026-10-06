Overhead Model and Calibration Policy
======================================

Two configuration objects control overhead timing:
:class:`~fyst_trajectories.overhead.OverheadModel` for activity durations, and
:class:`~fyst_trajectories.overhead.CalibrationPolicy` for how often each
calibration is performed.

Activity Durations
------------------

Controls the duration of each non-science activity::

    from fyst_trajectories.overhead import OverheadModel

    model = OverheadModel(
        retune_duration=300.0,        # whole-array KID retune between blocks (s)
        pointing_cal_duration=180.0,  # pointing correction scan (s)
        focus_duration=300.0,         # focus check (s)
        skydip_duration=300.0,        # elevation nod (s)
        planet_cal_duration=600.0,    # planet calibration scan (s)
        beam_map_duration=600.0,      # beam-map scan (same default as planet cal)
        settle_time=5.0,              # post-slew settling (s)
        min_scan_duration=60.0,       # minimum useful science scan (s)
        max_scan_duration=3600.0,     # longest science subscan (s)
    )

``min_scan_duration`` prevents short, wasteful scans; ``max_scan_duration``
forces long observations to split into sub-scans, with a retune between
them whenever one is due (always, at the default ``retune_cadence=0.0``); and
``beam_map_duration`` starts equal to ``planet_cal_duration`` because beam
maps run on the same planet targets.

A pong sub-scan holds the most whole pattern periods that fit its visit's
budget, which is at most ``max_scan_duration`` less any retune booked before
the sub-scan. A pong patch whose period can never fit (at
``retune_cadence=0.0``, a period longer than ``max_scan_duration`` less
``retune_duration``) is refused by
:func:`~fyst_trajectories.overhead.generate_timeline`: shrink the field,
widen ``spacing``, raise ``velocity`` or raise ``max_scan_duration``.

``retune_duration`` reserves a whole-array detector retune between
scan blocks (probe-tone placement followed by a target sweep across
every module; the default is the instrument team's commissioning
estimate, pending on-sky timing). It is a different operation from the
few-second in-scan tone-correction gap that
:func:`~fyst_trajectories.retune.inject_retune` stamps into a
trajectory (see :doc:`retune_events`).

Calibration Cadences
--------------------

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
An empty ``planet_targets`` skips that check and schedules them with no
target, which only the parked form accepts: ``planet_cal_scan=True``
needs at least one target.

The ``planet_cal_scan`` / ``planet_cal_passes`` / ``planet_cal_el_step`` /
``planet_cal_footprint`` group controls how a planet calibration is
realised (see :ref:`planet-cal-source-ces` below). Like the cadences and
durations, these are commissioning-era placeholders, listed with the
other unconfirmed defaults in :doc:`the pending-verification table <index>`.

Beam maps run on the same ``planet_targets`` machinery as planet
calibrations once a cadence opts them in:

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
:func:`~fyst_trajectories.planning.plan_source_ces_passes`, on the first
entry of ``planet_targets`` that is above ``planet_min_elevation`` and
clear of the Sun, so a planet inside the Sun zone is passed over for the
next one. The slew to the first pass is planned with
:func:`~fyst_trajectories.overhead.plan_transition`, as a science slew is,
and recorded as a ``SLEW`` block named ``slew_to_<planet>``: its azimuth
wrap holds every pass, its path clears the Sun zone, and the passes start
after it arrives. The planet is dragged across the Prime-Cam focal
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
:class:`~fyst_trajectories.overhead.SourceCESScanParams`), the true
scan start in ``metadata["t0_scan"]`` and the instant the planner's search
for the pass began in ``metadata["search_start"]``. Unlike a
calibration-night pass (see :doc:`overhead_calibration_night`), the dict
records the absolute ``window`` of its pass, so it describes that pass
rather than serving as a dispatch dict. Each pass is swept against the Sun
model in the slew's azimuth wrap before it is emitted. If no listed planet
is up and clear of the Sun, the sequence is not feasible (the planet never
reaches the required geometry in the search window), the slew is refused,
or a pass would enter the Sun zone, the calibration is skipped and left
due, so it is retried on a later scheduler iteration, exactly like a
planet cal with no visible planet.

Rebuild the pass trajectories with
``schedule_to_trajectories(timeline, science_only=False)``, which repeats
the planner's search from each block's ``search_start`` and so returns
each pass as it was planned, sample for sample (a block without
``search_start`` is re-solved around its recorded ``window`` instead; see
:func:`~fyst_trajectories.overhead.schedule_to_trajectories`); the default
``science_only=True`` returns science blocks only. An explicit
``planet_cal_el_step`` smaller than the footprint's elevation extent makes
adjacent pass windows overlap in time, and the planner emits a
:class:`~fyst_trajectories.exceptions.PointingWarning` when they do.
