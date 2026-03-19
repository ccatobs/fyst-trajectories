Plotting Functions
==================

Visualization utilities for trajectories and observational data.

.. note::

   This module requires the ``plotting`` extra::

       pip install "fyst-trajectories[plotting]"

   This installs ``matplotlib`` and ``scipy``.

.. automodule:: fyst_trajectories.plotting
   :members:
   :undoc-members:
   :show-inheritance:

Hit Map Visualization
---------------------

Generate hit-density maps in RA/Dec for multiple detector modules::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import PongScanConfig, TrajectoryBuilder
    from fyst_trajectories.plotting import plot_hit_map
    from fyst_trajectories.primecam import get_primecam_offset

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

The ``plot_hit_map()`` function returns a matplotlib ``Figure`` with one
panel per offset.  See the full parameter list in the auto-generated
API docs above.

Examples
--------

**With module footprint convolution**::

    fig = plot_hit_map(
        trajectory, offsets, site,
        module_fov=1.1,      # PrimeCam module FOV in degrees
        show=True,
    )

**Save figure**::

    fig = plot_hit_map(trajectory, offsets, site, show=False)
    fig.savefig("coverage_map.png", dpi=300)
