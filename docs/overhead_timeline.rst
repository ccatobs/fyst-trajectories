Timeline Generation
===================

:func:`~fyst_trajectories.overhead.generate_timeline` sequences science scans
and calibration activities over an observing window. At each time step it
inserts any calibrations whose cadence has elapsed, picks the best-positioned
observable patch, plans a Sun-checked slew to it, schedules a science scan on
it, and advances the clock.

Besides a tick on which nothing is observable, several things make the clock
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
limit. The pattern must also keep clear of the zenith: such a patch is
passed over while the circle about the field centre that bounds its
pattern (through a pong's corners, at a daisy's reach) holds the zenith,
since a track that passes round the zenith winds its azimuth through whole
turns, which the planner refuses. A visit whose first science subscan
cannot be booked books nothing, and that tick idles: the planner or the
booking screen refuses it at every length tried (see
:ref:`overhead-timeline-booking-screen`), or the subscan no longer fits by
the time the slew would arrive, and then the telescope does not slew. A
slew whose direct path would cross the Sun zone is refused, and that tick
idles at the unmoved pose and retries once the Sun has moved. Every pose
the telescope holds stays outside the Sun zone for the whole block (see
:ref:`overhead-timeline-held-poses`). See :doc:`sun_avoidance` for the Sun
policy these checks apply.

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
A constant-elevation patch needs an ``elevation``, the one its scan holds.
A pong or daisy patch takes none, and ``ObservingPatch`` refuses one: its
pattern is scanned where the field is on the sky.

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

Patches refused before scheduling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before it schedules anything, ``generate_timeline`` raises a ``ValueError``
that names every patch of the first of these kinds it finds:

* a patch whose ``scan_params["timestep"]`` is under 0.05 s,
  ``GO_TCS_MIN_SAMPLE_INTERVAL_SEC``, the shortest sample interval the
  Go TCS ``/path`` endpoint accepts;
* a constant-elevation patch without an ``elevation``, or with a
  ``detector_offset`` whose boresight would lie outside the elevation
  limits;
* a constant-elevation patch that can never have a pass. The scheduler asks
  the planner for the patch's pass over a sidereal day from the night's
  start, on the side the patch asks for, on either side when it names
  none, or inside its window for an ``lsa_window``. A field spanning a
  whole turn of RA, or reaching a celestial pole, has no pass to time, and
  a pass longer than the planner's 12 h search horizon is never found
  whole, so it is refused too;
* a pong whose pattern period can never fit a subscan (see
  :doc:`overhead_model`).

.. _overhead-timeline-module:

Observing with a Prime-Cam module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A patch observed with an off-axis Prime-Cam module carries its offset,
``detector_offset=get_primecam_offset("IM3")``. The scheduler then points
the boresight where the module sees the field: every pose it scores,
checks against the elevation limits and the Sun, slews to, holds or
records is the boresight's, so every block's azimuth range and elevation
are the boresight's. A constant-elevation patch's ``elevation`` is the
module's. A pong or daisy is booked only while the module's pattern and
the boresight's both stay inside the elevation limits, and a patch is not
observable while the boresight that would hold the module on the field
lies past the zenith. Every science block of the patch records the offset
as ``metadata["detector_offset"]`` and, on a constant-elevation patch, the
module's elevation as ``metadata["detector_elevation"]``, and
:func:`~fyst_trajectories.overhead.schedule_to_trajectories` rebuilds the
block with them. An offset whose ``dx`` and ``dy`` are both zero, such as
``IM0``'s, is stored as ``None``, and the patch is scheduled as one without
an offset.

.. _overhead-timeline-window-patch:

Sweeping an azimuth window
~~~~~~~~~~~~~~~~~~~~~~~~~~

A constant-elevation patch with ``scan_params["az_min"]`` and
``scan_params["az_max"]`` is swept over that window: each scan is the
window form of :func:`~fyst_trajectories.planning.plan_constant_el_scan`
(see :ref:`overhead-timeline-window-scan`) in whole legs, padded by
``az_padding``, and the visit is timed by the field's pass on the window's
side of the meridian. Write a window across azimuth 0°/360° unwrapped, as
(355, 370). ``ObservingPatch`` refuses one key without the other,
``az_min`` not below ``az_max``, the pair beside ``lsa_window``, a window
that reaches the meridian (an azimuth that is a multiple of 180°) without
``rising``, and a ``rising`` that names the side the window does not lie
on.

A visit longer than ``max_scan_duration`` is cut. Its subscans up to the
next retune are consecutive slices of one window scan, each at most
``max_scan_duration`` long and cut mid-leg where the limit falls, with no
retune between them; each block replays its own samples of that scan and,
when it starts after the scan, records the scan's start as
``metadata["scan_start"]``. After a retune a fresh scan starts, at the
window's low end. The last subscan of a visit takes up to
``max_scan_duration`` of what the visit has left. A stretch the planner or
the booking screen refuses is booked as one subscan instead, and a visit
ends at a subscan boundary where no retune is due and no stretch runs on.

Timing a pass by its sidereal window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A constant-elevation patch with ``scan_params["lsa_window"]`` is booked on
the window and the side of the meridian the planner scans it in, so its
blocks rebuild to the pass the planner plans; a ``rising`` given beside
the window is not read. The scheduler books the next window that opens
after it reaches the patch: a window already open then is not booked that
day.

Asking for one side of the meridian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A constant-elevation patch asks for the rising or the setting pass with
``scan_params["rising"]``, or with an azimuth window on that side. It is
held to that side by the scheduler's crossing check alone: only that
side's pass is solved, and the patch is booked when that pass is due,
including a setting pass that opens before the field centre transits, as
it does for a field that culminates not far above the scan elevation.

.. _overhead-timeline-visit:

How a Visit Is Booked
---------------------

A science visit is booked block by block: a slew to the first sample of the
visit's first subscan, a retune when one is due, an idle with reason
``waiting_for_pass`` at the slew's end pose while a constant-elevation pass
has not opened yet, then the science subscans, with a retune between two of
them whenever one is due (see :ref:`overhead-integration-slew-onto-scan`).
The subscans are booked back to back, with no lead reserved for dispatching
each one.

.. _overhead-timeline-booking-screen:

The booking screen
~~~~~~~~~~~~~~~~~~

A science subscan, a scan-mode planet pass, and, before its slew is
booked, a visit's shortest first subscan are each built and put to the
checks the control software (PCS) makes before it sends a scan to the
Go TCS, in this order:

1. The Sun, when the site's Sun avoidance is enabled: every sample lies
   outside the site's exclusion radius (a sample at the radius is inside)
   and, when the night is given a ``sun_safe`` model, outside that model's
   zone, a check PCS does not make.
2. The axes: no axis speed or acceleration past a Go TCS limit (table
   below). A pong, daisy or planet pass is judged on the gradients of its
   samples, a constant-elevation scan on the azimuth speed it records, and
   a constant-elevation scan or planet pass also on its turnaround, whose
   quintic profile peaks at 1.5 × ``az_accel``. A value at a limit
   passes.
3. The sample interval: no two samples closer than
   ``GO_TCS_MIN_SAMPLE_INTERVAL_SEC`` (0.05 s), judged as
   :func:`~fyst_trajectories.trajectory_utils.to_path_format` judges it.
   A grid built at a ``timestep`` of exactly 0.05 s can fall just under it
   by round-off, so at that ``timestep`` a scan is booked only where
   ``to_path_format`` serialises it.

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Constant (:mod:`~fyst_trajectories.trajectory_utils`)
     - Axis
     - Limit
   * - ``GO_TCS_AZ_MAX_VELOCITY``
     - azimuth speed
     - 3.0 deg/s
   * - ``GO_TCS_EL_MAX_VELOCITY``
     - elevation speed
     - 1.5 deg/s
   * - ``GO_TCS_AZ_MAX_ACCELERATION``
     - azimuth acceleration
     - 6.0 deg/s²
   * - ``GO_TCS_EL_MAX_ACCELERATION``
     - elevation acceleration
     - 1.5 deg/s²

A refused subscan is tried shorter: a pong with one whole period fewer,
down to one; a daisy or constant-elevation subscan at half the previous
length while that is longer than its shortest subscan, then at the
shortest; a window subscan with the whole legs of those lengths. A visit
whose shortest first subscan the screen refuses is not slewed to: the tick
idles at the unmoved pose with reason ``sun_scan`` (a sample inside the
Sun zone), ``axis_dynamics`` (an axis past a limit) or ``unplannable``
(two samples closer than the minimum interval). A tick also idles
``unplannable`` when the planner cannot build the scan or the subscan no
longer fits by the slew's arrival. A refused planet pass leaves the
calibration due.

The limit a fast pattern meets first is the elevation acceleration,
1.5 deg/s². A daisy turns at about its speed squared over its
``turn_radius``, 2 deg/s² for the default 0.5° at 1 deg/s, so it is not
booked at that speed, and a 4° pong at 1 deg/s is not booked either; both
are booked at 0.5 deg/s. A pattern whose field passes close to the zenith
meets the azimuth speed limit instead.

.. _overhead-timeline-held-poses:

Held poses
~~~~~~~~~~

An idle, a calibration that does not scan (a retune among them), the wait
for a pass and the stretch after the last block hold the telescope still.
Each is booked only at a pose that stays outside the Sun zone for the
whole block, checked every 30 s and at its end. A pose the zone would
reach first is left with a slew named ``sun_escape``
(:func:`~fyst_trajectories.overhead.plan_escape`); where no move clears
it, the idle carries the reason ``no_escape``.

What the simulator does not model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Inside a long visit no calibration is inserted but the retune.
* The move between two subscans separated by a retune, and the move
  between two planet passes, are not booked as slews: the telescope makes
  them inside the retune or the wait.
* Efficiency is the science blocks' time over the whole window, so it
  counts the turnarounds inside each scan as science.
* A calibration due at the tick before a pass opens, the startup
  calibrations included, can hold a visit back, so that it starts inside
  its pass, by a few minutes on some nights.
* A long constant-elevation visit is cut into subscans that are slices of
  one planned pass, while the control software's field form builds a whole
  pass and cannot start inside one, so how such a visit is dispatched is
  outside the simulator. A stretch of a window visit, by contrast, is one
  scan the control software can run.
* A slew is planned on a direct path only, and a Sun escape is chosen
  without regard to the next patch. After an escape the direct path to a
  patch can cross the Sun zone until the Sun has moved, so a daytime night
  can lose a visit that another escape pose would have reached.

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
    # ObservingTimeline: 2026-09-15 00:00:00.000 to 2026-09-15 12:00:00.000
    # Science: 1.2h ( 9.6%), 4 blocks

Time Budget
-----------

:func:`~fyst_trajectories.overhead.compute_budget` returns a dict with time breakdowns::

    from fyst_trajectories.overhead import compute_budget

    stats = compute_budget(timeline)

    # Top-level stats
    print(f"Total time:   {stats['total_time'] / 3600:.1f}h")  # Total time: 12.0h
    print(f"Efficiency:   {stats['efficiency']:.1%}")  # Efficiency: 9.6%
    print(f"Science:      {stats['science_time'] / 3600:.1f}h")  # Science: 1.2h
    print(f"Calibration:  {stats['calibration_time'] / 3600:.1f}h")  # Calibration: 3.0h

    # Per-patch breakdown
    for name, pstats in stats['per_patch'].items():
        print(f"{name}: {pstats['science_time'] / 3600:.1f}h, {pstats['n_scans']} scans")
    # CMB: 1.2h, 4 scans

    # Calibration type breakdown
    for cal_type, cstats in stats['calibration_breakdown'].items():
        print(f"{cal_type}: {cstats['count']}x, {cstats['total_time']:.0f}s total")
    # retune: 5x, 1500s total
    # pointing_cal: 21x, 3780s total
    # focus: 11x, 3300s total
    # skydip: 6x, 1800s total
    # planet_cal: 1x, 600s total

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
|                   | observable, a pass not yet open or a window   |
|                   | too short for ``min_scan_duration`` (none of  |
|                   | these carries a reason), a refused slew, a    |
|                   | subscan that cannot be built, whose samples   |
|                   | lie closer than the Go TCS minimum interval,  |
|                   | or that no longer fits by the slew's arrival  |
|                   | (``unplannable``), a scan with a sample       |
|                   | inside the Sun zone (``sun_scan``: the site's |
|                   | exclusion radius, which the control software  |
|                   | (PCS) enforces, and the night's injected Sun  |
|                   | model when one is given), a scan PCS refuses  |
|                   | for an axis speed or acceleration past a Go   |
|                   | TCS limit (``axis_dynamics``), a pose the Sun |
|                   | zone holds (``no_escape``), or the stretch    |
|                   | after the last block; the last six name       |
|                   | themselves in ``metadata["reason"]``; also    |
|                   | the wait for a pass to open                   |
|                   | (``waiting_for_pass``), at the pose the       |
|                   | telescope holds                               |
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
A timeline written to a file stores each block's start and stop to the
millisecond (see :doc:`overhead_io`), and the rebuild cuts a block's
samples at those stored times, so a block rebuilds to the same samples
from memory and from a file.

::

    from fyst_trajectories.overhead import schedule_to_trajectories

    results = schedule_to_trajectories(timeline)
    for timeline_block, scan_block in results:
        traj = scan_block.trajectory
        # feed traj.az, traj.el, traj.times into coverage code

.. _overhead-timeline-window-scan:

A constant-elevation window scan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A scan planned with the window form of
:func:`~fyst_trajectories.planning.plan_constant_el_scan` (``az_window`` with
``duration``) is written to a timeline with :meth:`TimelineBlock.from_window_scan()
<fyst_trajectories.overhead.TimelineBlock.from_window_scan>`,
which records the planner's inputs under ``metadata["window_scan"]``, and
:func:`~fyst_trajectories.overhead.schedule_to_trajectories` rebuilds the
block with the same form of the planner. The ``field`` and ``detector_offset`` given to the
writer are the ones given to the planner; a scan planned without a field is
written without one. The writer rebuilds the block once and refuses a scan it
would not reproduce. With ``span=(t_start, t_stop)`` the writer records a
stretch of the scan instead, so one scan can be booked as consecutive
blocks; a block that starts after the scan records the scan's start as
``metadata["scan_start"]``. Here module ``IM6`` sweeps azimuth 59.7° to
67.0° at elevation 43° for 0.33 h, at an on-sky speed of 0.5 deg/s::

    import math

    from astropy.time import Time

    from fyst_trajectories import get_primecam_offset
    from fyst_trajectories.overhead import ObservingTimeline, TimelineBlock
    from fyst_trajectories.planning import plan_constant_el_scan

    im6_offset = get_primecam_offset("IM6")
    window_scan = plan_constant_el_scan(
        az_window=(59.7, 67.0),
        elevation=43.0,
        velocity=0.5 / math.cos(math.radians(43.0)),  # mount-frame azimuth rate
        duration=0.33 * 3600.0,
        site=site,
        start_time=Time("2026-11-15T08:01:44", scale="utc"),
        detector_offset=im6_offset,
    )
    window_block = TimelineBlock.from_window_scan(
        window_scan,
        site=site,
        patch_name="COSMOS_CE",
        scan_index=0,
        detector_offset=im6_offset,
    )
    window_timeline = ObservingTimeline(
        blocks=[window_block],
        site=site,
        start_time=window_block.t_start,
        end_time=window_block.t_stop,
        overhead_model=OverheadModel(),
        calibration_policy=CalibrationPolicy(),
    )
    window_pairs = schedule_to_trajectories(window_timeline)
    window_traj = window_pairs[0][1].trajectory

    print(window_block.metadata["window_scan"]["az_window"])  # [59.7, 67.0]
    print(f"{window_block.az_start:.2f} to {window_block.az_end:.2f}")  # 58.79 to 66.67
    print(f"{window_block.elevation:.3f}")  # 44.723
    print((window_traj.az == window_scan.trajectory.az).all())  # True

The block's ``az_start`` and ``az_end`` are the boresight's azimuth envelope,
turnaround overshoot included, so they lie outside the recorded window, and
its ``elevation`` is the boresight's; with a ``detector_offset`` the record
holds the module's window and elevation.

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
        # "Timeline is clean"
