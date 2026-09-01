Retune Events
=============

:func:`~fyst_trajectories.trajectory_utils.inject_retune` has two modes: a
uniform-cadence path that schedules retunes every ``retune_interval``
seconds, and an event-list path that applies a caller-supplied sequence of
:class:`~fyst_trajectories.trajectory.RetuneEvent` instances. Both paths
populate :attr:`~fyst_trajectories.trajectory.Trajectory.retune_events` on
the returned trajectory, so introspection and ECSV round-trip work
identically regardless of which mode produced the retunes.

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

Two more uniform-cadence knobs, both off by default:

- ``prefer_turnarounds=True`` snaps each due retune to a turnaround
  region within ``turnaround_window`` seconds of its scheduled time.
  This saves only a sliver of science time (~0.04% in the
  configurations measured) and concentrates the coverage gaps at
  turnaround positions; the default time-based placement keeps coverage
  uniform.
- ``module_index`` / ``n_modules`` stagger the cadence per readout
  module: each module's first retune is offset by
  ``module_index * retune_interval / n_modules``, so with
  ``n_modules=7`` only one module is retuning at any given time. The
  per-module duty cost is unchanged; see
  :func:`~fyst_trajectories.trajectory_utils.inject_retune` for what
  staggering does and does not buy, and the instrument-team premise it
  rests on.

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

Either mode overwrites only ``SCAN_FLAG_SCIENCE`` samples with
``SCAN_FLAG_RETUNE``; turnaround flags are never modified, and
``Trajectory.science_mask`` excludes the retuned samples.

``t_start`` is measured in seconds from the trajectory start
(``trajectory.times[0]``). Events are sorted and validated for overlap
by :func:`~fyst_trajectories.trajectory_utils.inject_retune`; events past
the trajectory end are skipped with a
:class:`~fyst_trajectories.exceptions.PointingWarning`. Per-module
staggering in event-list mode is handled by composition: call
:func:`~fyst_trajectories.trajectory_utils.inject_retune` once per module
with its own event list.

Sampled event lists
-------------------

:func:`~fyst_trajectories.trajectory_utils.sample_retune_events` draws a
non-overlapping event list from caller-supplied samplers, for Monte Carlo
studies of retune overhead. No distribution is baked in:

.. code-block:: python

    import numpy as np

    from fyst_trajectories import inject_retune, sample_retune_events

    rng = np.random.default_rng(seed=42)
    events = sample_retune_events(
        duration=traj.duration,
        interval_sampler=lambda r: r.uniform(60.0, 120.0),
        duration_sampler=lambda r: r.uniform(3.0, 8.0),
        rng=rng,
    )
    retuned = inject_retune(traj, retune_events=events)

CSV schema
----------

Retune schedules are commonly stored as a two-or-three-column CSV.
fyst-trajectories ships no CSV reader - the format is a convention that
consumers implement. The full convention:

- Header row required. Column names are compared case-insensitively.
- Required columns: ``t_start_s`` (float, seconds from trajectory
  start) and ``duration_s`` (float, positive seconds).
- Optional column: ``module_index`` (integer, 0-based, non-negative).
  When absent, all rows are treated as ``module_index == 0``.
- Any other column is ignored; a reader following this convention
  should warn once, naming the unused columns.

The minimal snippet below is enough for well-formed files: it reads the
two required columns by exact name and implements none of the
convention's robustness (no case folding, no ``module_index`` handling,
no unused-column warning). A production consumer should add those.

Example::

    t_start_s,duration_s,module_index
    30.0,5.0,0
    300.0,5.0,0
    600.0,8.0,0

Read one into a list of :class:`~fyst_trajectories.trajectory.RetuneEvent`
with the standard library:

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
:class:`~fyst_trajectories.overhead.TimelineBlock` (the column schema
is in :doc:`overhead_io`). The write side
encodes each :class:`~fyst_trajectories.trajectory.RetuneEvent` as a
JSON-native ``[t_start, duration]`` pair; the read side decodes those back
into a tuple of :class:`~fyst_trajectories.trajectory.RetuneEvent`, matching
what :attr:`~fyst_trajectories.trajectory.Trajectory.retune_events` exposes.

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

    Plumbing from
    :func:`~fyst_trajectories.trajectory_utils.inject_retune`'s output
    (``trajectory.retune_events``) into
    ``TimelineBlock.metadata["retune_events"]`` is currently manual; the
    overhead scheduler does not auto-propagate the generated event list
    into each science block's metadata.
