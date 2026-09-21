Overhead I/O
============

Writing a simulated night to a TOAST-compatible ECSV file, reading one
back, and drawing it. See :doc:`../overhead_io` for the column schema,
TOAST compatibility, and a round-trip example.

.. autofunction:: fyst_trajectories.overhead.write_timeline

.. autofunction:: fyst_trajectories.overhead.read_timeline

Night-Level Figures
-------------------

Render a recorded night from its ECSV timeline (both figures need the
``plotting`` extra)::

    from fyst_trajectories.overhead import read_timeline
    from fyst_trajectories.visualization import plot_sky_coverage, plot_timeline_gantt

    timeline = read_timeline("timeline.ecsv")

    fig = plot_timeline_gantt(timeline, show=False)
    fig.savefig("night_gantt.png", dpi=140, bbox_inches="tight")

    fig = plot_sky_coverage(timeline, show=False)
    fig.savefig("sky_coverage.png", dpi=140, bbox_inches="tight")

.. autofunction:: fyst_trajectories.visualization.plot_timeline_gantt

.. autofunction:: fyst_trajectories.visualization.plot_sky_coverage
