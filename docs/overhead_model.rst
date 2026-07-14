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
A cadence of 0 means "every scan boundary"; a cadence of ``None`` (only
valid for ``beam_map_cadence``) disables automatic scheduling for that
calibration type entirely::

    from fyst_trajectories.overhead import CalibrationPolicy

    policy = CalibrationPolicy(
        retune_cadence=0.0,           # every scan boundary
        pointing_cadence=3600.0,      # every 1 hour
        focus_cadence=7200.0,         # every 2 hours
        skydip_cadence=10800.0,       # every 3 hours
        planet_cal_cadence=43200.0,   # every 12 hours
        beam_map_cadence=None,        # default: manual injection only
        planet_targets=("jupiter", "saturn", "mars", "uranus", "neptune"),
        planet_min_elevation=20.0,    # planet must be above this
        planet_cal_scan=False,        # plan planet cals as source-CES passes
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

``BEAM_MAP`` is a first-class :class:`~fyst_trajectories.overhead.CalibrationType`
with its own cadence (``CalibrationPolicy.beam_map_cadence``) and
duration (``OverheadModel.beam_map_duration``). The default
``beam_map_cadence=None`` keeps beam maps off the automatic schedule
so existing operators are not surprised by extra calibration blocks
appearing in their timelines; setting it to a positive value opts the
schedule in to cadence-driven beam mapping using the same
``planet_targets`` machinery as ``planet_cal``.

**Example: 6-hour beam-map cadence**

.. code-block:: python

    # Beam map every 6 hours using the configured planet targets.
    policy = CalibrationPolicy(beam_map_cadence=21600.0)

Beam maps and planet calibrations share planet-target visibility checking
but have independent cadences and durations.

.. _planet-cal-source-ces:

Planet Calibrations as Source-CES Scans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default a planet calibration is a single fixed-duration parked block
(``OverheadModel.planet_cal_duration``): the telescope holds its current
pose while the calibration runs, and no scan geometry is recorded.

Setting ``planet_cal_scan=True`` instead plans each planet calibration as
a real multi-pass source-CES sequence via
:func:`~fyst_trajectories.plan_source_ces_passes`, anchored at the
scheduler clock. The planet is dragged across the Prime-Cam focal plane at
a fixed boresight elevation, once per pass, with the passes stepped in
elevation so they run sequentially:

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

``planet_cal_passes`` must be at least 1, and ``planet_cal_el_step`` (when
given) must be positive; ``None`` uses the planner default (the footprint
elevation extent).

Default values are commissioning-era placeholders that should be
confirmed by the instrument team.
