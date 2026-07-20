Visualization
=============

.. module:: fyst_trajectories.visualization

All matplotlib rendering for fyst-trajectories lives in the
``fyst_trajectories.visualization`` subpackage: trajectory diagnostics, RA/Dec
hit-density maps, and night-level overhead-timeline figures.

.. note::

   This subpackage requires the ``plotting`` extra::

       pip install "fyst-trajectories[plotting]"

   This installs ``matplotlib``. Importing ``fyst_trajectories`` (or any
   of its subpackages) never imports matplotlib; each plot function
   loads it lazily on first call.

Trajectory Diagnostics
----------------------

.. autofunction:: fyst_trajectories.visualization.plot_trajectory

**Example**::

    from fyst_trajectories.visualization import plot_trajectory

    # Plot 3-panel figure (Az vs Time, El vs Time, Sky Track)
    fig = plot_trajectory(trajectory, show=True)

    # Get figure without displaying
    fig = plot_trajectory(trajectory, show=False)
    fig.savefig("trajectory.png")

Hit Map Visualization
---------------------

.. autofunction:: fyst_trajectories.visualization.plot_hit_map

Generate hit-density maps in RA/Dec for multiple detector modules::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import PongScanConfig, TrajectoryBuilder
    from fyst_trajectories.primecam import get_primecam_offset
    from fyst_trajectories.visualization import plot_hit_map

    site = get_fyst_site()
    start_time = Time("2026-03-15T04:00:00", scale="utc")

    # Generate a Pong scan trajectory
    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=180.0, dec=-30.0)
        .with_config(PongScanConfig(
            timestep=0.1, width=2.0, height=2.0, spacing=0.1,
            velocity=0.5, num_terms=4, angle=0.0,
        ))
        .duration(300.0)
        .starting_at(start_time)
        .build()
    )

    # Plot detector-center tracks for two PrimeCam modules
    offsets = [
        (get_primecam_offset("i1"), "f280"),
        (get_primecam_offset("i6"), "f350"),
    ]
    fig = plot_hit_map(trajectory, offsets, site, show=True)

**With module footprint convolution**::

    fig = plot_hit_map(
        trajectory, offsets, site,
        module_fov=1.3,      # PrimeCam module FOV in degrees (2 x 0.65 deg radius)
        show=True,
    )

**Save figure**::

    fig = plot_hit_map(trajectory, offsets, site, show=False)
    fig.savefig("coverage_map.png", dpi=300)

Night-Level Overhead Figures
----------------------------

Render a recorded night from its ECSV timeline::

    from fyst_trajectories.overhead import read_timeline
    from fyst_trajectories.visualization import plot_sky_coverage, plot_timeline_gantt

    timeline = read_timeline("timeline.ecsv")

    fig = plot_timeline_gantt(timeline, show=False)
    fig.savefig("night_gantt.png", dpi=140, bbox_inches="tight")

    fig = plot_sky_coverage(timeline, show=False)
    fig.savefig("sky_coverage.png", dpi=140, bbox_inches="tight")

Pass ``ax=`` to compose a panel into an existing figure (the function
then draws into that axes, returns ``ax.get_figure()``, and never calls
``plt.show()`` or ``fig.tight_layout()`` on the caller's figure).

.. autofunction:: fyst_trajectories.visualization.plot_timeline_gantt

.. autofunction:: fyst_trajectories.visualization.plot_sky_coverage
