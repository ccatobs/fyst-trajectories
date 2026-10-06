Timeline Generation
===================

:func:`~fyst_trajectories.overhead.generate_timeline` sequences science scans
and calibration activities over an observing window. At each time step it
inserts any calibrations whose cadence has elapsed, picks the best-positioned
observable patch, plans a Sun-checked slew to it, schedules a science scan on
it, and advances the clock.

Besides a tick on which nothing is observable, four things make the clock
advance without a scan. A constant-elevation patch is only selectable while
its elevation-crossing pass is imminent, so expect idle blocks before such a
pass opens; a pong or daisy patch, only while its shortest subscan (for a
pong, one whole pattern period), and any retune due before it, fit before any
part of its scan pattern reaches an elevation limit or the Sun reaches the
field. The pattern reaches well past the field centre: a pong fills its box,
and a daisy's petals reach past ``radius`` by at least ``turn_radius``, so
near rising and setting such a patch is passed over while its centre is
still inside the limits. The pong's whole box is taken at every instant,
while the pattern passes near each corner once a period, so near setting a
pong can stop up to a period before its own trajectory would reach the
limit. A visit whose first science subscan cannot be planned books nothing,
and that tick idles: the planner refuses the subscan (a pong first retries
with fewer periods), or the subscan no longer fits by the time the slew
would arrive, and then the telescope does not slew. A slew whose direct
path would cross the Sun zone is refused, and that tick idles at the
unmoved pose and retries once the Sun has moved. A parked pose the zone has
overtaken is moved out of it before anything else runs that tick. See
:doc:`sun_avoidance` for the Sun policy these checks apply.

Defining Patches
----------------

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
        width=4.0,
        height=4.0,
        scan_type="pong",
        velocity=0.5,
        scan_params={"spacing": 0.1, "num_terms": 4},
    )

Supported ``scan_type`` values: ``"constant_el"``, ``"pong"``, ``"daisy"``.

``velocity`` is forwarded to the pattern unchanged, and its frame follows
``scan_type``: a tangent-plane (on-sky) speed in deg/s for ``"pong"`` and
``"daisy"``, and a mount-frame azimuth coordinate rate for ``"constant_el"``,
whose on-sky speed is ``velocity * cos(elevation)``.

``priority`` and ``weight`` tune the selection when several patches are
observable at once: each candidate's score is multiplied by
``weight / priority``, so lower ``priority`` values and higher
``weight`` values win. Both default to 1.0.

**From an existing FieldRegion**::

    from fyst_trajectories.planning import FieldRegion

    field = FieldRegion(ra_center=0.0, dec_center=-2.0, width=10.0, height=6.0)
    patch = ObservingPatch.from_field_region(
        field, name="Stripe82", scan_type="constant_el", velocity=1.0, elevation=45.0,
    )

Custom Cadences and Durations
-----------------------------

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

    # Aggressive commissioning cadences: pointing every 30 min, focus
    # every hour.
    policy = CalibrationPolicy(
        retune_cadence=0.0,          # before every science subscan
        pointing_cadence=1800.0,     # 30 min (override of 3600 s default)
        focus_cadence=3600.0,        # 1 hour
        skydip_cadence=7200.0,       # 2 hours
        planet_cal_cadence=43200.0,  # 12 hours
    )

    # Shorter science blocks, so calibrations interleave more often
    overhead = OverheadModel(max_scan_duration=1800.0)

    patches = [
        ObservingPatch(
            name="CMB",
            ra_center=0.0,
            dec_center=-2.0,
            width=10.0,
            height=6.0,
            scan_type="constant_el",
            velocity=1.0,
            elevation=45.0,
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

Time Budget
-----------

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
| ``"calibration"`` | Retune, pointing, focus, skydip, planet cal,  |
|                   | or beam map                                   |
+-------------------+-----------------------------------------------+
| ``"slew"``        | Telescope slew between positions, including   |
|                   | a move out of the Sun zone                    |
+-------------------+-----------------------------------------------+
| ``"idle"``        | No scan was placed this tick: nothing         |
|                   | observable or a pass not yet open (neither    |
|                   | carries a reason), a refused slew, a subscan  |
|                   | that cannot be built or no longer fits by     |
|                   | the slew's arrival (``unplannable``), a pose  |
|                   | the Sun zone holds, or the stretch after the  |
|                   | last block; the last four name themselves in  |
|                   | ``metadata["reason"]``                        |
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
simulation),
:func:`~fyst_trajectories.overhead.schedule_to_trajectories` walks the
timeline, calls the appropriate ``plan_*_scan`` function for each science
block, and returns a list of ``(TimelineBlock, ScanBlock)`` pairs. Its
reference entry says what each pair covers, which calibration blocks
``science_only=False`` adds, and which blocks are logged and skipped.

::

    from fyst_trajectories.overhead import schedule_to_trajectories

    results = schedule_to_trajectories(timeline)
    for timeline_block, scan_block in results:
        traj = scan_block.trajectory
        # feed traj.az, traj.el, traj.times into coverage code

Validation
----------

A timeline can be checked for common authoring defects: blocks that
overlap in time, gaps between consecutive blocks, blocks that fall
outside the timeline window, science or calibration blocks whose azimuth
bounds are unordered, and pose discontinuities (a slew whose start
azimuth, or an idle whose parked az/el, does not match where the previous
block left the telescope)::

    warnings = timeline.validate()
    if warnings:
        for w in warnings:
            print(f"WARNING: {w}")
    else:
        print("Timeline is clean")
