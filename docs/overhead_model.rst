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
    )

Planet calibrations and beam maps are only scheduled when at least one
planet target in ``planet_targets`` is above ``planet_min_elevation``.

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

Default values are commissioning-era placeholders that should be
confirmed by the instrument team.
