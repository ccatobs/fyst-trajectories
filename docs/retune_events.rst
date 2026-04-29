Retune Events
=============

:func:`~fyst_trajectories.inject_retune` has two modes: a uniform-cadence
path that schedules retunes every ``retune_interval`` seconds, and an
event-list path that applies a caller-supplied sequence of
:class:`~fyst_trajectories.RetuneEvent` instances. Both paths populate
:attr:`~fyst_trajectories.Trajectory.retune_events` on the returned
trajectory, so introspection and ECSV round-trip work identically
regardless of which mode produced the retunes.

Dual-mode API
-------------

Uniform cadence:

.. code-block:: python

    from fyst_trajectories import inject_retune

    retuned = inject_retune(
        traj,
        retune_interval=300.0,
        retune_duration=5.0,
    )

Explicit event list:

.. code-block:: python

    from fyst_trajectories import RetuneEvent, inject_retune

    events = [
        RetuneEvent(t_start=30.0, duration=5.0),
        RetuneEvent(t_start=300.0, duration=5.0),
        RetuneEvent(t_start=600.0, duration=5.0),
    ]
    retuned = inject_retune(traj, retune_events=events)
    assert retuned.retune_events == tuple(events)

``t_start`` is measured in seconds from the trajectory start
(``trajectory.times[0]``). Events are sorted and validated for overlap
by :func:`~fyst_trajectories.inject_retune`; events past the trajectory
end are skipped with a :class:`~fyst_trajectories.PointingWarning`.
Per-module staggering in event-list mode is handled by composition —
call :func:`~fyst_trajectories.inject_retune` once per module with its
own event list.

CSV schema
----------

The canonical on-disk shape is a two-or-three-column CSV. Downstream
consumers such as the ``primecam_camera_mapping_simulations``
``--retune_events`` CLI flag parse this format directly.

- Header row required. Column names are compared case-insensitively.
- Required columns: ``t_start_s`` (float, seconds from trajectory
  start) and ``duration_s`` (float, positive seconds).
- Optional column: ``module_index`` (integer, 0-based, non-negative).
  When absent, all rows are treated as ``module_index == 0``.
- Any other columns are ignored with a single
  :class:`~fyst_trajectories.PointingWarning` listing the unused
  column names.

Example::

    t_start_s,duration_s,module_index
    30.0,5.0,0
    300.0,5.0,0
    600.0,8.0,0

Reading a CSV into a list of :class:`~fyst_trajectories.RetuneEvent`
is a two-line job using the standard library:

.. code-block:: python

    import csv

    from fyst_trajectories import RetuneEvent, inject_retune

    with open("retunes.csv", newline="") as handle:
        reader = csv.DictReader(handle)
        events = [
            RetuneEvent(
                t_start=float(row["t_start_s"]),
                duration=float(row["duration_s"]),
            )
            for row in reader
        ]

    retuned = inject_retune(traj, retune_events=events)

ECSV round-trip
---------------

Per-block retune events persist through
:func:`~fyst_trajectories.overhead.write_timeline` /
:func:`~fyst_trajectories.overhead.read_timeline` via the existing
``block_meta_json`` extra-payload channel on
:class:`~fyst_trajectories.overhead.TimelineBlock`. The write side
encodes each :class:`~fyst_trajectories.RetuneEvent` as a JSON-native
``[t_start, duration]`` pair; the read side decodes those back into a
tuple of :class:`~fyst_trajectories.RetuneEvent`, matching what
:attr:`~fyst_trajectories.Trajectory.retune_events` exposes.

Attach retune events to a block before writing:

.. code-block:: python

    from fyst_trajectories import RetuneEvent
    from fyst_trajectories.overhead import write_timeline

    events = [
        RetuneEvent(t_start=30.0, duration=5.0),
        RetuneEvent(t_start=300.0, duration=5.0),
    ]
    timeline.blocks[0].metadata["retune_events"] = events

    write_timeline(timeline, "night.ecsv")

Read the timeline back and inspect the decoded tuple:

.. code-block:: python

    from fyst_trajectories.overhead import read_timeline

    loaded = read_timeline("night.ecsv")
    events = loaded.blocks[0].metadata["retune_events"]
    # events is a tuple[RetuneEvent, ...]

.. note::

    Plumbing from :func:`~fyst_trajectories.inject_retune`'s output
    (``trajectory.retune_events``) into
    ``TimelineBlock.metadata["retune_events"]`` is currently manual; the
    overhead scheduler does not auto-propagate the generated event list
    into each science block's metadata.
