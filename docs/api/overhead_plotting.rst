Overhead Plotting
=================

Night-level visualization of observing timelines: a per-lane Gantt of the
night and the science scans' RA/Dec sky coverage. Requires ``matplotlib``
(install via ``pip install fyst-trajectories[plotting]``).

Render a recorded night from its ECSV timeline::

    from fyst_trajectories.overhead import (
        plot_sky_coverage,
        plot_timeline_gantt,
        read_timeline,
    )

    timeline = read_timeline("timeline.ecsv")

    fig = plot_timeline_gantt(timeline, show=False)
    fig.savefig("night_gantt.png", dpi=140, bbox_inches="tight")

    fig = plot_sky_coverage(timeline, show=False)
    fig.savefig("sky_coverage.png", dpi=140, bbox_inches="tight")

Pass ``ax=`` to compose a panel into an existing figure (the function then
draws into that axes, returns ``ax.get_figure()``, and never calls
``plt.show()`` or ``fig.tight_layout()`` on the caller's figure).

.. autofunction:: fyst_trajectories.overhead.plot_timeline_gantt

.. autofunction:: fyst_trajectories.overhead.plot_sky_coverage
