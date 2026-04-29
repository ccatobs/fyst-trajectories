Overhead Quickstart
===================

Basic Usage
-----------

Generate an 8-hour observing timeline with 2 patches.
``generate_timeline`` also works without the ``overhead_model`` /
``calibration_policy`` arguments and falls back to the defaults shown
here (the same defaults used to fill in any field absent from a stored
ECSV file's header metadata — see :doc:`overhead_io`).
::

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.overhead import (
        ObservingPatch,
        OverheadModel,
        CalibrationPolicy,
        generate_timeline,
        compute_budget,
    )

    site = get_fyst_site()

    patches = [
        ObservingPatch(
            name="Deep56",
            ra_center=24.0,
            dec_center=-32.0,
            width=40.0,
            height=10.0,
            scan_type="constant_el",
            velocity=1.0,
            elevation=50.0,
        ),
        ObservingPatch(
            name="Wide01",
            ra_center=180.0,
            dec_center=-30.0,
            width=20.0,
            height=10.0,
            scan_type="pong",
            velocity=0.5,
        ),
    ]

    # Duration of each non-science activity, in seconds.
    # These values set the per-event duration of calibrations and the
    # min/max science-scan split thresholds. Defaults shown explicitly.
    overhead_model = OverheadModel(
        retune_duration=5.0,           # KID probe-tone reset
        pointing_cal_duration=180.0,   # 3 min cross-scan on a bright quasar
        focus_duration=300.0,          # 5 min M2 focus check
        skydip_duration=300.0,         # 5 min elevation nod for opacity
        planet_cal_duration=600.0,     # 10 min planet calibration scan
        beam_map_duration=600.0,       # 10 min beam-map raster (separate from above)
        settle_time=5.0,               # post-slew mount settling
        min_scan_duration=60.0,        # reject useless short scans
        max_scan_duration=3600.0,      # force split past 1 h
    )

    # How often each calibration type fires, in seconds.
    # These cadences determine how many of each event end up in the timeline.
    # ``beam_map_cadence=None`` (the default) keeps beam maps off the
    # automatic schedule; set a positive value to opt in.
    calibration_policy = CalibrationPolicy(
        retune_cadence=0.0,            # 0 = between every science scan
        pointing_cadence=3600.0,       # every 1 hr
        focus_cadence=7200.0,          # every 2 hr
        skydip_cadence=10800.0,        # every 3 hr
        planet_cal_cadence=43200.0,    # 1-2x per night
        beam_map_cadence=None,         # None = manual injection only
        planet_targets=("jupiter", "saturn", "mars", "uranus", "neptune"),
        planet_min_elevation=20.0,     # planet must be above this to be used
    )

    timeline = generate_timeline(
        patches=patches,
        site=site,
        start_time="2026-06-15T02:00:00",
        end_time="2026-06-15T10:00:00",
        overhead_model=overhead_model,
        calibration_policy=calibration_policy,
    )

    print(f"{len(timeline)} blocks scheduled")

A detailed breakdown is available via ``compute_budget`` (next section).

Efficiency Statistics
---------------------

:func:`~fyst_trajectories.overhead.compute_budget` provides a summary::

    stats = compute_budget(timeline)
    print(f"Efficiency: {stats['efficiency']:.1%}")
    print(f"Science:     {stats['science_time'] / 3600:.1f}h")
    print(f"Calibration: {stats['calibration_time'] / 3600:.1f}h")
    print(f"Slew:        {stats['slew_time'] / 3600:.1f}h")
    print(f"Idle:        {stats['idle_time'] / 3600:.1f}h")

The returned dict also contains per-patch breakdowns and calibration
type counts in ``stats['per_patch']`` and ``stats['calibration_breakdown']``.

Saving a Timeline
------------------

Write to TOAST-compatible ECSV and read it back::

    from fyst_trajectories.overhead import write_timeline, read_timeline

    write_timeline(timeline, "my_timeline.ecsv")
    loaded = read_timeline("my_timeline.ecsv")
    print(f"Loaded {len(loaded)} blocks")

See :doc:`overhead_io` for format details and TOAST compatibility.
