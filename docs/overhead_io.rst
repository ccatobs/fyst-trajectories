Timeline I/O
============

:func:`~fyst_trajectories.overhead.write_timeline` and :func:`~fyst_trajectories.overhead.read_timeline`
provide round-trip serialization in TOAST-compatible ECSV format.

Writing a Timeline
------------------

Simulate a night, then write it::

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.overhead import (
        ObservingPatch,
        generate_timeline,
        write_timeline,
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
    ]
    timeline = generate_timeline(
        patches=patches,
        site=site,
        start_time="2026-06-15T06:00:00",
        end_time="2026-06-15T10:00:00",
    )

    write_timeline(timeline, "schedule.ecsv")

Reading a Timeline
------------------

Read it back and check what it holds::

    from fyst_trajectories.overhead import read_timeline

    timeline = read_timeline("schedule.ecsv")
    print(f"Loaded {len(timeline)} blocks")  # Loaded 30 blocks
    print(f"Efficiency: {timeline.efficiency:.1%}")  # Efficiency: 45.2%

The window holds the patch's constant-elevation pass, so the night has
science in it. A window that closes before the pass opens reports
``0.0%``: the honest schedule of calibrations and idle, not a failure.

ECSV Format
-----------

The ECSV file uses TOAST ``GroundSchedule`` canonical column names
(``start_time`` / ``stop_time`` as ISO strings, ``scan_index`` /
``subscan_index``) plus FYST-specific extension columns for block type and
scan pattern:

+----------------------+--------+----------------------------------------------+
| Column               | Type   | Description                                  |
+======================+========+==============================================+
| ``start_time``       | str    | Start time as ISO-8601 UTC string            |
+----------------------+--------+----------------------------------------------+
| ``stop_time``        | str    | Stop time as ISO-8601 UTC string             |
+----------------------+--------+----------------------------------------------+
| ``name``             | str    | Patch name or calibration type               |
+----------------------+--------+----------------------------------------------+
| ``azmin``            | float  | Minimum azimuth (deg); slew: from-azimuth    |
+----------------------+--------+----------------------------------------------+
| ``azmax``            | float  | Maximum azimuth (deg); slew: to-azimuth      |
+----------------------+--------+----------------------------------------------+
| ``el``               | float  | Elevation (deg)                              |
+----------------------+--------+----------------------------------------------+
| ``scan_index``       | int    | Scan counter                                 |
+----------------------+--------+----------------------------------------------+
| ``subscan_index``    | int    | Sub-scan index                               |
+----------------------+--------+----------------------------------------------+
| ``boresight_angle``  | float  | Boresight rotation angle (deg)               |
+----------------------+--------+----------------------------------------------+
| ``ra_center``        | float  | FYST extension: patch RA centre (deg)        |
+----------------------+--------+----------------------------------------------+
| ``dec_center``       | float  | FYST extension: patch Dec centre (deg)       |
+----------------------+--------+----------------------------------------------+
| ``width``            | float  | FYST extension: patch width (deg)            |
+----------------------+--------+----------------------------------------------+
| ``height``           | float  | FYST extension: patch height (deg)           |
+----------------------+--------+----------------------------------------------+
| ``velocity``         | float  | FYST extension: scan velocity (deg/s); on-sky|
|                      |        | for pong/daisy, azimuth rate for constant-el |
+----------------------+--------+----------------------------------------------+
| ``scan_params_json`` | str    | FYST extension: pattern parameters (JSON)    |
+----------------------+--------+----------------------------------------------+
| ``block_type``       | str    | FYST extension: science/calibration/slew/idle|
+----------------------+--------+----------------------------------------------+
| ``scan_type``        | str    | FYST extension: pattern or calibration type  |
+----------------------+--------+----------------------------------------------+
| ``rising``           | bool   | FYST extension: rising-side flag             |
+----------------------+--------+----------------------------------------------+
| ``az_final``         | float  | FYST extension: azimuth the block ends at    |
|                      |        | (deg) when that differs from ``azmax``;      |
|                      |        | ``nan`` otherwise (see below)                |
+----------------------+--------+----------------------------------------------+
| ``block_meta_json``  | str    | FYST extension: JSON-encoded bag of any      |
|                      |        | ``TimelineBlock.metadata`` keys (see below)  |
|                      |        | not promoted to a dedicated column above;    |
|                      |        | carries retune events (:doc:`retune_events`) |
+----------------------+--------+----------------------------------------------+

For ``slew`` rows, ``azmin`` / ``azmax`` hold the from / to azimuths of
the move and may be unordered (``azmin`` greater than ``azmax`` for a
negative-direction slew); they are true minimum / maximum bounds only
for science and calibration rows. A consumer that needs ordered bounds
(for example ``azmax - azmin`` as a scan width) must filter on
``block_type`` first. On a row that sweeps, the bounds are the azimuth
envelope the block executes; the field geometry is in the ``ra_center`` /
``dec_center`` / ``width`` / ``height`` columns instead.

``az_final`` records the pose a swept block is actually left at, which is
generally neither bound; read it through
:attr:`~fyst_trajectories.overhead.TimelineBlock.end_pose_az`, which
falls back to ``azmax`` wherever the column is ``nan``, including a file
written before the column existed. See
:class:`~fyst_trajectories.overhead.TimelineBlock` for the full
envelope-versus-pose distinction.

The set of FYST extension columns may grow over time; any
:attr:`~fyst_trajectories.overhead.TimelineBlock.metadata` field not surfaced
as a dedicated column lands in ``block_meta_json`` so the round-trip remains
lossless.

Header metadata carries the declared timeline window, the site (coordinates,
Nasmyth port, plate scale, sun-avoidance radii), and every
:class:`~fyst_trajectories.overhead.OverheadModel` and
:class:`~fyst_trajectories.overhead.CalibrationPolicy` field. Any key missing
from a file falls back to that dataclass's default on read, so a partial or
hand-written header still loads. Telescope axis limits are not persisted: a
non-FYST site reloads with the FYST limits and a
:class:`~fyst_trajectories.exceptions.PointingWarning`.

A timeline with no blocks, which is what
:func:`~fyst_trajectories.overhead.plan_calibration_night` returns when the
Sun never sets inside the requested window, writes one placeholder row, because ECSV
cannot express a typed table with no rows. The header flags it with
``timeline_is_empty`` and :func:`~fyst_trajectories.overhead.read_timeline`
drops the row again, so such a file reads back with no blocks rather than
with one dated 2000-01-01.

Retune events
-------------

Per-block retune events (:doc:`retune_events`) persist through the round
trip via the ``block_meta_json`` channel. The write side encodes each
:class:`~fyst_trajectories.trajectory.RetuneEvent` as a JSON-native
``[t_start, duration]`` pair; the read side decodes them back into a
tuple, matching what
:attr:`~fyst_trajectories.trajectory.Trajectory.retune_events` exposes.
Attach them to a block before writing, then read them back:

.. code-block:: python

    from fyst_trajectories import RetuneEvent
    from fyst_trajectories.overhead import write_timeline

    timeline.blocks[0].metadata["retune_events"] = [
        RetuneEvent(t_start=30.0, duration=5.0),
        RetuneEvent(t_start=300.0, duration=5.0),
    ]
    write_timeline(timeline, "night.ecsv")

    loaded = read_timeline("night.ecsv")
    events = loaded.blocks[0].metadata["retune_events"]
    # events is a tuple[RetuneEvent, ...]

Plumbing :func:`~fyst_trajectories.retune.inject_retune`'s
output (``trajectory.retune_events``) into
``TimelineBlock.metadata["retune_events"]`` is manual: the scheduler does
not propagate a generated event list into each science block.

TOAST Compatibility
-------------------

Files written by fyst-trajectories use TOAST canonical column names for the
common fields **and** attach ``u.deg`` units to the angle columns
(``azmin``, ``azmax``, ``el``, ``boresight_angle``), with the site
coordinates stored as ``Quantity`` header metadata. TOAST's
``GroundSchedule`` v5 reader can therefore read them directly, ignoring
the FYST extension columns.

Standard TOAST schedule files (without ``block_type``, ``scan_type``,
``rising``, or the patch-geometry extension columns) are also supported on
read. They are interpreted as all-science timelines with sensible defaults
for the missing FYST extension columns.

To hand TOAST a schedule with no calibration, slew, or idle rows, filter to
science blocks before writing (the FYST extension columns are still written;
TOAST ignores them). Replace the block list and nothing else, so the
timeline's window, models and metadata travel with it; the gaps where the
removed rows were are what ``validate()`` then reports, by construction. A
night with no science would write the placeholder row described above, which
only :func:`~fyst_trajectories.overhead.read_timeline` knows to drop, so
check that there is something to hand over first::

    import dataclasses

    from fyst_trajectories.overhead import write_timeline

    science_only = dataclasses.replace(timeline, blocks=timeline.science_blocks)
    if science_only.blocks:
        write_timeline(science_only, "toast_schedule.ecsv")
