Planning Module
===============

Astronomer-friendly wrappers that translate field coordinates, elevation
constraints, and scan velocities into full pattern configurations. Planning
functions exist only where there is non-trivial computation bridging the
astronomer's inputs and the pattern config:

- **Pong** -- computes the Pong period from field dimensions, spacing, and
  velocity.
- **Constant-El** -- finds RA-edge elevation crossings to determine timing,
  derives the azimuth range and ``n_scans`` automatically.
- **Daisy** -- convenience wrapper; parameters map nearly 1:1 to the config.

Sidereal, planet, and linear patterns have no non-trivial planning step;
:class:`~fyst_trajectories.patterns.TrajectoryBuilder` can be used directly.

Quick Start
-----------

Plan a Pong survey scan over a 2x2 degree field::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import FieldRegion, plan_pong_scan

    site = get_fyst_site()

    field = FieldRegion(ra_center=180.0, dec_center=-30.0, width=2.0, height=2.0)
    block = plan_pong_scan(
        field=field,
        velocity=0.5,        # deg/s
        spacing=0.1,         # deg between scan lines
        num_terms=4,         # Fourier terms for smooth turnarounds
        site=site,
        start_time=Time("2026-03-15T04:00:00", scale="utc"),
        timestep=0.1,
    )

    print(block.summary)
    print(f"Duration: {block.duration:.1f}s ({block.duration / 3600:.1f}h)")
    print(f"Trajectory: {block.trajectory.n_points} points")

Plan a constant-elevation scan over a field with auto-computed timing::

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import FieldRegion, plan_constant_el_scan

    site = get_fyst_site()

    field = FieldRegion(ra_center=0.0, dec_center=-2.0, width=60.0, height=14.0)
    block = plan_constant_el_scan(
        field=field,
        elevation=50.0,
        velocity=0.5,
        site=site,
        start_time="2026-09-15T00:00:00",
        rising=True,
    )

    print(block.summary)
    print(f"Duration: {block.duration:.0f}s")

Field Regions
-------------

A :class:`~fyst_trajectories.planning.FieldRegion` defines a rectangular sky area
by its center coordinates and angular extent::

    from fyst_trajectories.planning import FieldRegion

    field = FieldRegion(
        ra_center=0.0,     # deg
        dec_center=-2.0,   # deg
        width=60.0,        # RA extent in degrees
        height=14.0,       # Dec extent in degrees
    )

    # Dec boundaries are computed automatically
    print(f"Dec range: [{field.dec_min}, {field.dec_max}]")
    # Dec range: [-9.0, 5.0]

Planning a Pong Scan
--------------------

:func:`~fyst_trajectories.planning.plan_pong_scan` converts a field region into a
Pong scan trajectory. It automatically computes the Pong period from the field
dimensions, spacing, and velocity, then generates one full period by default.

Basic usage::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import FieldRegion, plan_pong_scan

    site = get_fyst_site()

    field = FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7)
    block = plan_pong_scan(
        field=field,
        velocity=0.5,
        spacing=0.08,
        num_terms=4,
        site=site,
        start_time=Time("2026-03-15T22:12:00", scale="utc"),
        timestep=0.1,
        angle=170.0,     # rotation angle (degrees)
    )

Multiple cycles::

    block = plan_pong_scan(
        field=field,
        velocity=0.5,
        spacing=0.1,
        num_terms=4,
        site=site,
        start_time=Time("2026-03-15T22:12:00", scale="utc"),
        timestep=0.1,
        n_cycles=3,      # observe 3 full Pong periods
    )

With a detector offset (for off-axis PrimeCam modules)::

    from fyst_trajectories.primecam import get_primecam_offset

    offset = get_primecam_offset("i1")
    block = plan_pong_scan(
        field=field,
        velocity=0.5,
        spacing=0.1,
        num_terms=4,
        site=site,
        start_time=Time("2026-03-15T22:12:00", scale="utc"),
        timestep=0.1,
        detector_offset=offset,
    )

Multi-Rotation Pong Tiling
--------------------------

:func:`~fyst_trajectories.planning.plan_pong_rotation_sequence` returns
``n_rotations`` copies of a base
:class:`~fyst_trajectories.patterns.PongScanConfig` with the ``angle``
field overridden to a uniform ``180° / n_rotations`` sequence. Each
returned config is passed individually through
:func:`~fyst_trajectories.planning.plan_pong_scan`::

    from astropy.time import Time, TimeDelta

    from fyst_trajectories import PongScanConfig, get_fyst_site
    from fyst_trajectories.planning import (
        FieldRegion,
        plan_pong_rotation_sequence,
        plan_pong_scan,
    )

    site = get_fyst_site()
    base = PongScanConfig(
        timestep=0.1, width=2.0, height=2.0,
        spacing=0.1, velocity=0.5, num_terms=4, angle=0.0,
    )

    # 8 rotations at 22.5 deg spacing.
    configs = plan_pong_rotation_sequence(base, n_rotations=8)
    [c.angle for c in configs]
    # [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]

    # Schedule each rotation back-to-back.
    field = FieldRegion(ra_center=180.0, dec_center=-30.0, width=2.0, height=2.0)
    t0 = Time("2026-03-15T04:00:00", scale="utc")
    blocks = []
    for i, cfg in enumerate(configs):
        block = plan_pong_scan(
            field=field,
            velocity=cfg.velocity,
            spacing=cfg.spacing,
            num_terms=cfg.num_terms,
            site=site,
            start_time=t0 + TimeDelta(i * 600.0, format="sec"),
            timestep=cfg.timestep,
            angle=cfg.angle,
        )
        blocks.append(block)

Planning a Constant-Elevation Scan
-----------------------------------

:func:`~fyst_trajectories.planning.plan_constant_el_scan` auto-computes
the azimuth range, observation duration, and number of scans from a
``FieldRegion``, target elevation, and approximate start time:

1. Finds when the RA edges of the field cross the target elevation (determines
   start/end time and total duration).
2. Computes the azimuth range that covers the entire field at that elevation
   at the midpoint of the observation.
3. Derives ``n_scans`` from the duration and single-leg sweep time.
4. Builds and returns a :class:`~fyst_trajectories.planning.ScanBlock`.

Basic usage::

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import FieldRegion, plan_constant_el_scan

    site = get_fyst_site()

    field = FieldRegion(ra_center=0.0, dec_center=-2.0, width=60.0, height=14.0)
    block = plan_constant_el_scan(
        field=field,
        elevation=50.0,          # fixed elevation in degrees
        velocity=0.5,            # az scan speed in deg/s
        site=site,
        start_time="2026-09-15T00:00:00",
        rising=True,             # use rising crossing
    )

    print(block.summary)
    print(f"Duration: {block.duration:.0f}s")
    print(f"Az range: [{block.computed_params['az_start']:.1f}, "
          f"{block.computed_params['az_stop']:.1f}]")

With a detector offset::

    from fyst_trajectories.primecam import get_primecam_offset

    offset = get_primecam_offset("i1")
    block = plan_constant_el_scan(
        field=field,
        elevation=50.0,
        velocity=0.5,
        site=site,
        start_time="2026-09-15T00:00:00",
        detector_offset=offset,
    )

Planning a Daisy Scan
---------------------

:func:`~fyst_trajectories.planning.plan_daisy_scan` takes a single RA/Dec
position rather than a ``FieldRegion``::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import plan_daisy_scan

    site = get_fyst_site()

    block = plan_daisy_scan(
        ra=83.633,
        dec=22.014,
        radius=0.5,             # characteristic radius R0 (degrees)
        velocity=0.3,           # scan velocity (deg/s)
        turn_radius=0.2,        # curvature radius for turns (degrees)
        avoidance_radius=0.0,   # avoid center within this radius
        start_acceleration=0.5, # ramp-up acceleration (deg/s^2)
        site=site,
        start_time=Time("2026-01-15T02:00:00", scale="utc"),
        timestep=0.1,
        duration=300.0,         # 5 minutes
    )

    print(block.summary)

Scan Block Output
-----------------

All planning functions return a :class:`~fyst_trajectories.planning.ScanBlock`
containing:

``trajectory``
    The generated :class:`~fyst_trajectories.trajectory.Trajectory`, ready for
    telescope upload or further analysis.

``config``
    The underlying pattern configuration object (``PongScanConfig``,
    ``ConstantElScanConfig``, ``DaisyScanConfig``, etc.).

``duration``
    The total observation duration in seconds.

``computed_params``
    A dictionary of computed parameters specific to the scan type. The
    exact key set is documented by a :class:`typing.TypedDict` schema
    per scan type:

    - **Pong** -- :class:`~fyst_trajectories.planning.PongComputedParams`
      (``period``, ``x_numvert``, ``y_numvert``, ``n_cycles``).
    - **Constant-El (auto)** --
      :class:`~fyst_trajectories.planning.ConstantElComputedParams`
      (``az_start``, ``az_stop``, ``az_throw``, ``n_scans``,
      ``start_time_iso``, ``end_time_iso``, ``duration``).
    - **Daisy** --
      :class:`~fyst_trajectories.planning.DaisyComputedParams`
      (``duration``).

    Access the computed parameters as a standard ``dict``.

``summary``
    A human-readable string summarizing the planned observation.

Example of inspecting a scan block::

    block = plan_pong_scan(...)

    # Access the trajectory
    traj = block.trajectory
    print(f"Points: {traj.n_points}, Duration: {traj.duration:.0f}s")

    # Inspect computed parameters
    print(f"Pong period: {block.computed_params['period']:.0f}s")
    print(f"Vertices: {block.computed_params['x_numvert']} x "
          f"{block.computed_params['y_numvert']}")

    # Print summary
    print(block.summary)

    # Validate trajectory against telescope limits
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.trajectory_utils import validate_trajectory
    validate_trajectory(traj, get_fyst_site())
