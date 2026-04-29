Timeline Generation
===================

:func:`~fyst_trajectories.overhead.generate_timeline` sequences science scans
and calibration activities over an observing window. At each time step it
inserts any calibrations whose cadence has elapsed, picks the best-positioned
observable patch, schedules a science scan on it, and advances the clock.

ObservingPatch Setup
--------------------

Each sky region is defined as an :class:`~fyst_trajectories.overhead.ObservingPatch`::

    from fyst_trajectories.overhead import ObservingPatch

    # Constant-elevation scan at a fixed elevation
    deep_field = ObservingPatch(
        name="Deep56",
        ra_center=24.0,
        dec_center=-32.0,
        width=40.0,
        height=10.0,
        scan_type="constant_el",
        velocity=1.0,
        elevation=50.0,
    )

    # Pong scan (elevation computed from source position)
    wide_field = ObservingPatch(
        name="Wide01",
        ra_center=180.0,
        dec_center=-30.0,
        width=20.0,
        height=10.0,
        scan_type="pong",
        velocity=0.5,
        scan_params={"spacing": 0.1, "num_terms": 4},
    )

Supported ``scan_type`` values: ``"constant_el"``, ``"pong"``, ``"daisy"``.

**From an existing FieldRegion**::

    from fyst_trajectories.planning import FieldRegion

    field = FieldRegion(ra_center=0.0, dec_center=-2.0, width=60.0, height=14.0)
    patch = ObservingPatch.from_field_region(
        field, name="Stripe82", scan_type="constant_el", velocity=1.0, elevation=50.0,
    )

Custom CalibrationPolicy
------------------------

Override the default cadences and durations. See :doc:`overhead_model`
for all available fields.

::

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.overhead import (
        ObservingPatch,
        CalibrationPolicy,
        OverheadModel,
        generate_timeline,
    )

    site = get_fyst_site()

    # Aggressive calibration (commissioning override): pointing every
    # 30 min, focus every hour. Overrides the default pointing_cadence
    # of 3600 s.
    policy = CalibrationPolicy(
        retune_cadence=0.0,          # every scan boundary
        pointing_cadence=1800.0,     # 30 min (override of 3600 s default)
        focus_cadence=3600.0,        # 1 hour
        skydip_cadence=7200.0,       # 2 hours
        planet_cal_cadence=43200.0,  # 12 hours
    )

    # Faster retunes, shorter scans
    overhead = OverheadModel(
        retune_duration=3.0,
        max_scan_duration=1800.0,
    )

    patches = [
        ObservingPatch(
            name="CMB",
            ra_center=0.0,
            dec_center=-2.0,
            width=60.0,
            height=14.0,
            scan_type="constant_el",
            velocity=1.0,
            elevation=50.0,
        ),
    ]

    timeline = generate_timeline(
        patches=patches,
        site=site,
        start_time="2026-09-15T00:00:00",
        end_time="2026-09-15T12:00:00",
        overhead_model=overhead,
        calibration_policy=policy,
    )

    print(timeline)

Budget Output
-------------

:func:`~fyst_trajectories.overhead.compute_budget` returns a dict with time breakdowns::

    from fyst_trajectories.overhead import compute_budget

    stats = compute_budget(timeline)

    # Top-level stats
    print(f"Total time:   {stats['total_time'] / 3600:.1f}h")
    print(f"Efficiency:   {stats['efficiency']:.1%}")
    print(f"Science:      {stats['science_time'] / 3600:.1f}h")
    print(f"Calibration:  {stats['calibration_time'] / 3600:.1f}h")

    # Per-patch breakdown
    for name, pstats in stats['per_patch'].items():
        print(f"  {name}: {pstats['science_time'] / 3600:.1f}h, "
              f"{pstats['n_scans']} scans")

    # Calibration type breakdown
    for cal_type, cstats in stats['calibration_breakdown'].items():
        print(f"  {cal_type}: {cstats['count']}x, "
              f"{cstats['total_time']:.0f}s total")

Timeline Blocks
---------------

The returned :class:`~fyst_trajectories.overhead.ObservingTimeline` contains a list of
:class:`~fyst_trajectories.overhead.TimelineBlock` objects. Each block has a ``block_type``:

+-------------------+-----------------------------------------------+
| Block type        | Description                                   |
+===================+===============================================+
| ``"science"``     | Science observation of a patch                |
+-------------------+-----------------------------------------------+
| ``"calibration"`` | Retune, pointing, focus, skydip, or planet cal|
+-------------------+-----------------------------------------------+
| ``"slew"``        | Telescope slew between positions              |
+-------------------+-----------------------------------------------+
| ``"idle"``        | No observable target available                |
+-------------------+-----------------------------------------------+

Inspect individual blocks::

    for block in timeline:
        print(
            f"{block.t_start.iso} | {block.block_type:12s} | "
            f"{block.patch_name:15s} | {block.duration:7.0f}s"
        )

Regenerating Trajectories
-------------------------

A timeline stores scan geometry in ``block.metadata``, not motion arrays.
To rebuild az/el/time arrays for every science block (e.g. for coverage
simulation or to stream to OCS),
:func:`~fyst_trajectories.overhead.schedule_to_trajectories` iterates the
timeline, calls the appropriate ``plan_*_scan`` function for each science
block, and yields ``(TimelineBlock, ScanBlock)`` pairs. Blocks that are
no longer feasible at execution time (e.g. because of drifting
elevation) are logged and skipped rather than raising.

::

    from fyst_trajectories.overhead import schedule_to_trajectories

    results = schedule_to_trajectories(timeline)
    for timeline_block, scan_block in results:
        traj = scan_block.trajectory
        # feed traj.az, traj.el, traj.times into coverage or upload code

Validation
----------

A timeline can be checked for overlapping blocks or out-of-range entries::

    warnings = timeline.validate()
    if warnings:
        for w in warnings:
            print(f"WARNING: {w}")
    else:
        print("Timeline is clean")
