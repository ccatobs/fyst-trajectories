Timeline I/O
============

:func:`~fyst_trajectories.overhead.write_timeline` and :func:`~fyst_trajectories.overhead.read_timeline`
provide round-trip serialization in TOAST-compatible ECSV format.

Writing a Timeline
------------------

::

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
        start_time="2026-06-15T00:00:00",
        end_time="2026-06-15T04:00:00",
    )

    write_timeline(timeline, "schedule.ecsv")

Reading a Timeline
------------------

::

    from fyst_trajectories.overhead import read_timeline

    timeline = read_timeline("schedule.ecsv")
    print(f"Loaded {len(timeline)} blocks")
    print(f"Efficiency: {timeline.efficiency:.1%}")

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
| ``azmin``            | float  | Minimum azimuth (deg)                        |
+----------------------+--------+----------------------------------------------+
| ``azmax``            | float  | Maximum azimuth (deg)                        |
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
| ``velocity``         | float  | FYST extension: scan velocity (deg/s)        |
+----------------------+--------+----------------------------------------------+
| ``scan_params_json`` | str    | FYST extension: pattern parameters (JSON)    |
+----------------------+--------+----------------------------------------------+
| ``block_type``       | str    | FYST extension: science/calibration/slew/idle|
+----------------------+--------+----------------------------------------------+
| ``scan_type``        | str    | FYST extension: pattern or calibration type  |
+----------------------+--------+----------------------------------------------+
| ``rising``           | bool   | FYST extension: rising-side flag             |
+----------------------+--------+----------------------------------------------+
| ``block_meta_json``  | str    | FYST extension: JSON-encoded bag of any      |
|                      |        | ``TimelineBlock.metadata`` keys (see below)  |
|                      |        | not promoted to a dedicated column above     |
+----------------------+--------+----------------------------------------------+

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
TOAST ignores them)::

    from fyst_trajectories.overhead import ObservingTimeline, write_timeline

    science_only = ObservingTimeline(
        blocks=timeline.science_blocks,
        site=timeline.site,
        start_time=timeline.start_time,
        end_time=timeline.end_time,
        overhead_model=timeline.overhead_model,
        calibration_policy=timeline.calibration_policy,
    )
    write_timeline(science_only, "toast_schedule.ecsv")
