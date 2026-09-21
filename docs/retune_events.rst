Retune Events
=============

Stamp detector-retune gaps into a built trajectory.
:func:`~fyst_trajectories.trajectory_utils.inject_retune` has two modes, a
uniform cadence and a caller-supplied event list, and both populate
:attr:`~fyst_trajectories.trajectory.Trajectory.retune_events`, so
introspection and the ECSV round trip work the same either way.
What the field records differs by mode: the event list records the caller's
request (an event past the trajectory end stays in the tuple, unapplied), the
uniform cadence records the placements it made. ``scan_flag`` is the single
record of what was applied.

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

- ``prefer_turnarounds=True`` snaps each due retune to a nearby
  turnaround, trading uniform coverage for a sliver of science time.
- ``module_index`` / ``n_modules`` stagger the cadence per readout
  module: each module's first retune is offset by
  ``module_index * retune_interval / n_modules``, so with
  ``n_modules=7`` only one module is retuning at a time, as long as
  ``retune_duration`` is shorter than ``retune_interval / n_modules``
  (it is, at the defaults). The per-module duty cost is unchanged;
  :func:`~fyst_trajectories.trajectory_utils.inject_retune` states what
  staggering buys, the instrument-team premise it rests on, and how to
  compose it in event-list mode.

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
(``trajectory.times[0]``).

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

Retune schedules from a CSV
---------------------------

Retune schedules are commonly kept as ``t_start_s,duration_s``
CSV. fyst-trajectories ships no reader for them; read one into a list of
:class:`~fyst_trajectories.trajectory.RetuneEvent` with the standard library:

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

A file that also carries a module column is applied per module: group the rows
and call :func:`~fyst_trajectories.trajectory_utils.inject_retune` once for
each, since event-list mode requires the default ``module_index=0`` /
``n_modules=1``.

Retune events survive the ECSV round trip; see "Retune events" in
:doc:`overhead_io`.
