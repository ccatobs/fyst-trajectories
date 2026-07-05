Planning Module
===============

Astronomer-friendly wrappers that translate field coordinates, elevation
constraints, and scan velocities into full pattern configurations. Planning
functions exist only where there is non-trivial computation bridging the
astronomer's inputs and the pattern config:

- **Pong** - computes the Pong period from field dimensions, spacing, and
  velocity.
- **Constant-El** - finds RA-edge elevation crossings to determine timing,
  derives the azimuth range and ``n_scans`` automatically; also accepts an
  explicit ``lsa_window`` to pin timing to a Local Sidereal Angle window.
- **Daisy** - convenience wrapper; parameters map nearly 1:1 to the config.
- **Source CES** - drags a moving source (planet or sidereal point) across an
  instrument-array footprint at fixed boresight elevation, solving for the
  azimuth drift rate.
- **AltAz Pong / Daisy** - run the Pong and Daisy patterns about a fixed
  horizon-frame center (no RA/Dec tracking).

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

Planning an AltAz Pong Scan
---------------------------

:func:`~fyst_trajectories.planning.plan_pong_altaz_scan` runs the same
Curvy-Pong pattern as :func:`~fyst_trajectories.planning.plan_pong_scan`, but
about a fixed horizon-frame center (``az_center``, ``el_center``) with no sky
tracking. The on-sky tangent-plane offsets are mapped into telescope
coordinates by::

    az = x_offset / cos(radians(el_center)) + az_center
    el = y_offset + el_center

so ``width``, ``height``, ``spacing``, and ``velocity`` keep their on-sky
meaning from the celestial Pong. The azimuth coordinate is stretched by
``1 / cos(el_center)``: the azimuth-coordinate extent is
``width / cos(el_center)`` and the azimuth-coordinate speed exceeds the on-sky
``velocity`` by the same factor (budget ``velocity`` against the mount azimuth
rate limit accordingly).

Basic usage::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import plan_pong_altaz_scan

    site = get_fyst_site()

    block = plan_pong_altaz_scan(
        az_center=120.0,     # deg
        el_center=60.0,      # deg (fixed; no sky tracking)
        width=2.0,           # deg on-sky
        height=2.0,          # deg on-sky
        spacing=0.1,         # deg between scan lines
        velocity=0.5,        # deg/s on-sky
        site=site,
        start_time=Time("2026-03-15T04:00:00", scale="utc"),
    )

    print(block.summary)
    print(f"Duration: {block.duration:.1f}s")

The duration defaults to one full Pong period; pass ``n_cycles`` to observe
several. ``num_terms``, ``angle``, ``timestep``, and ``detector_offset`` behave
as in :func:`~fyst_trajectories.planning.plan_pong_scan`. ``start_time`` is
used to anchor the trajectory timestamp and to convert the center to RA/Dec for
the (warn-only) sun-safety pre-flight check.

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

LSA-windowed timing
~~~~~~~~~~~~~~~~~~~~

Instead of deriving timing from RA-edge elevation crossings, pass
``lsa_window=(min_lsa, max_lsa)`` (degrees) to pin the scan to a Local
Sidereal Angle window. The duration is ``((max_lsa - min_lsa) mod 360) / 15``
hours. Wrap-around windows (``max_lsa < min_lsa``, e.g. ``(310.0, 10.0)``) are
supported, and ``rising`` still selects the azimuth half::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import FieldRegion, plan_constant_el_scan

    site = get_fyst_site()

    field = FieldRegion(ra_center=0.0, dec_center=-2.0, width=60.0, height=14.0)
    block = plan_constant_el_scan(
        field=field,
        elevation=50.0,
        velocity=0.5,
        site=site,
        start_time=Time("2026-09-15T00:00:00", scale="utc"),
        rising=True,
        lsa_window=(310.0, 10.0),   # 60 deg / 15 = 4 h scan across LSA = 0
    )

    print(f"Duration: {block.duration / 3600:.1f}h")

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

Planning an AltAz Daisy Scan
----------------------------

:func:`~fyst_trajectories.planning.plan_daisy_altaz_scan` runs the same
Constant-Velocity Daisy pattern as
:func:`~fyst_trajectories.planning.plan_daisy_scan`, but about a fixed
horizon-frame center (``az_center``, ``el_center``) with no sky tracking. The
on-sky tangent-plane offsets are mapped into telescope coordinates by::

    az = x_offset / cos(radians(el_center)) + az_center
    el = y_offset + el_center

so ``radius``, ``velocity``, ``turn_radius``, ``avoidance_radius``, and
``start_acceleration`` keep their on-sky meaning from the celestial Daisy. The
azimuth coordinate is stretched by ``1 / cos(el_center)``: the
azimuth-coordinate extent is ``2 * radius / cos(el_center)`` and the
azimuth-coordinate speed exceeds the on-sky ``velocity`` by the same factor
(budget ``velocity`` against the mount azimuth rate limit accordingly).

Basic usage::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import plan_daisy_altaz_scan

    site = get_fyst_site()

    block = plan_daisy_altaz_scan(
        az_center=120.0,        # deg
        el_center=60.0,         # deg (fixed; no sky tracking)
        radius=0.5,             # characteristic radius R0 (deg on-sky)
        velocity=0.3,           # deg/s on-sky
        turn_radius=0.2,        # curvature radius for turns (deg on-sky)
        avoidance_radius=0.0,   # avoid center within this radius (deg on-sky)
        start_acceleration=0.5, # ramp-up acceleration (deg/s^2 on-sky)
        site=site,
        start_time=Time("2026-03-15T04:00:00", scale="utc"),
        timestep=0.1,
        duration=300.0,
    )

    print(block.summary)
    print(f"Duration: {block.duration:.1f}s")

``timestep``, ``duration``, ``y_offset``, and ``detector_offset`` behave as in
:func:`~fyst_trajectories.planning.plan_daisy_scan`. ``start_time`` is used to
anchor the trajectory timestamp and to convert the center to RA/Dec for the
(warn-only) sun-safety pre-flight check.

Planning a Source CES (Planet / Sidereal Drift)
------------------------------------------------

:func:`~fyst_trajectories.planning.plan_source_ces` drags a moving source
(planet or sidereal point) across an instrument-array footprint at a fixed
boresight elevation ``el_bore``, solving for the azimuth drift rate ``v_az``
that sweeps the source over the array. It mirrors Simons Observatory's
``schedlib.source.make_source_ces``.

Worked example, Jupiter rising across the full PrimeCam array::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site, PRIMECAM_MODULES
    from fyst_trajectories.planning import plan_source_ces

    site = get_fyst_site()
    modules = [PRIMECAM_MODULES[k] for k in ("c", "i1", "i2", "i3", "i4", "i5", "i6")]

    block = plan_source_ces(
        body="jupiter",
        footprint=modules,
        el_bore=35.0,
        night=Time("2026-03-15T00:00:00", scale="utc"),
        mode="rising",
        site=site,
    )

    print(block.summary)
    cp = block.computed_params
    print(f"Source pass: {cp['t0_iso'][:19]} → {cp['t1_iso'][:19]}")
    print(f"Az drift:    {cp['v_az']:+.5f} deg/s")
    print(f"Az range:    [{cp['az_start']:.2f}, {cp['az_start'] + cp['az_throw']:.2f}] deg")

The ``footprint`` argument accepts a named module string (``"c"``,
``"i1"`` .. ``"i6"``), a single :class:`~fyst_trajectories.InstrumentOffset`,
a sequence of offsets (one per module), or an explicit
:class:`~fyst_trajectories.ArrayFootprint`. For a multi-module footprint,
:func:`~fyst_trajectories.resolve_module_tag` expands an SO-style tag
(``resolve_module_tag("i1,i2")``, or ``"all"``) into that sequence.

Select the time window with either ``night`` + ``mode`` (``"rising"`` or
``"setting"``; searches the next 24 h) or an explicit
``window=(t_start, t_end)``.

Other knobs: ``boresight_rot`` (mechanical boresight rotation, deg),
``v_az`` (override the solved drift rate), ``az_padding`` (extra azimuth
margin, default 0.5 deg), ``az_branch`` (centre of the azimuth wrap branch),
and ``allow_partial`` (clip to the observable arc and warn instead of
raising when the source does not fully cover the footprint at ``el_bore``).

Params-only mode (emit-time)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~fyst_trajectories.planning.compute_source_ces_params` takes the same
arguments as :func:`~fyst_trajectories.planning.plan_source_ces` (minus
``timestep``) and returns just the scalar
:class:`~fyst_trajectories.planning.SourceCESComputedParams` dict, skipping
trajectory generation. This is the emit-time entry point: a scheduler can
price many candidate scans cheaply (feasibility, duration, azimuth throw)
without building a trajectory, which the execution layer generates once at
dispatch.

Worked example, the same Jupiter input as the section above, scalars
only::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site, PRIMECAM_MODULES
    from fyst_trajectories.planning import compute_source_ces_params

    site = get_fyst_site()
    modules = [PRIMECAM_MODULES[k] for k in ("c", "i1", "i2", "i3", "i4", "i5", "i6")]

    params = compute_source_ces_params(
        body="jupiter",
        footprint=modules,
        el_bore=35.0,
        night=Time("2026-03-15T00:00:00", scale="utc"),
        mode="rising",
        site=site,
    )

    # No trajectory; just the scalars a scheduler needs at emit time.
    print(f"az_start={params['az_start']:.2f}  az_throw={params['az_throw']:.2f}")
    print(f"v_az={params['v_az']:+.5f} deg/s  el_bore={params['el_bore']:.2f}")
    print(f"t0={params['t0_iso'][:19]}  t1={params['t1_iso'][:19]}")

The returned dict is identical to ``plan_source_ces(...).computed_params``
for the same inputs.

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

    - **Pong** - :class:`~fyst_trajectories.planning.PongComputedParams`
      (``period``, ``x_numvert``, ``y_numvert``, ``n_cycles``).
    - **AltAz Pong** -
      :class:`~fyst_trajectories.planning.PongAltAzComputedParams`
      (``period``, ``x_numvert``, ``y_numvert``, ``n_cycles``,
      ``az_center``, ``el_center``).
    - **Constant-El (auto)** -
      :class:`~fyst_trajectories.planning.ConstantElComputedParams`
      (``az_start``, ``az_stop``, ``az_throw``, ``n_scans``,
      ``start_time_iso``, ``end_time_iso``, ``duration``).
    - **Daisy** -
      :class:`~fyst_trajectories.planning.DaisyComputedParams`
      (``duration``).
    - **AltAz Daisy** -
      :class:`~fyst_trajectories.planning.DaisyAltAzComputedParams`
      (``duration``, ``az_center``, ``el_center``).
    - **Source CES** -
      :class:`~fyst_trajectories.planning.SourceCESComputedParams`
      (``az_start``, ``az_throw``, ``v_az``, ``el_bore``,
      ``boresight_rot``, ``t0_iso``, ``t1_iso``, ``duration``,
      ``mode``, ``n_scans``).

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
