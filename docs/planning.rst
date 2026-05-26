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

Planning a Source CES (Planet / Sidereal Drift)
------------------------------------------------

:func:`~fyst_trajectories.planning.plan_source_ces` plans a constant-
elevation scan that *drags a moving source* (planet or sidereal point)
across an instrument-array footprint -- use it when the target is a
point source you want to sweep across the array (planet calibration,
beam map) rather than a sky rectangle to survey
(:func:`~fyst_trajectories.planning.plan_constant_el_scan`'s job).

Where ``plan_constant_el_scan`` aims at a fixed RA/Dec rectangle and
lets sidereal motion fill the time axis, ``plan_source_ces`` aims at a
single moving source and solves for an *additional* azimuth drift rate
``v_az`` so the source sweeps across the full array footprint while
the boresight stays at a fixed elevation ``el_bore``. It is the
fyst-trajectories analogue of Simons Observatory's
``schedlib.source.make_source_ces`` (intended consumer: a future
``schedlib/policies/fyst.py``).
Two implementation differences worth knowing:

- ``az_bore`` is recovered in closed form via the spherical inverse
  (:func:`~fyst_trajectories.detector_to_boresight`) instead of SO's
  Nelder-Mead solve.
- The monotonic-arc selection enumerates *all* arcs between local
  elevation extrema, picking the first arc of the requested direction
  whose elevation range covers ``el_bore``. SO picks the global
  ``argmin``/``argmax`` pair, which silently fails when the search
  window straddles a culmination (e.g. a 24 h window on a planet near
  opposition).

Worked example -- Jupiter rising across the full PrimeCam array::

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

Footprint specification
~~~~~~~~~~~~~~~~~~~~~~~

The ``footprint`` argument accepts four shapes:

- **Named module string** (``"c"``, ``"i1"`` … ``"i6"``) -- resolved via
  :func:`~fyst_trajectories.get_primecam_offset` and inscribed as a
  50-vertex circle of radius
  :data:`~fyst_trajectories.MODULE_FOV_RADIUS_DEG` (currently 0.65°).
- **Single** :class:`~fyst_trajectories.InstrumentOffset` -- as above,
  inscribed as a circular cover.
- **Sequence of** :class:`~fyst_trajectories.InstrumentOffset` -- one
  per module; cover is the union of per-module circles; aggregate
  centre is the arithmetic mean of per-module ``(dx, dy)``.
- **Explicit** :class:`~fyst_trajectories.ArrayFootprint` -- direct
  ``(centre, cover polygon)`` specification, mirroring SO's
  ``array_info`` dict.

Time-window selection
~~~~~~~~~~~~~~~~~~~~~

- ``night`` + ``mode`` (``"rising"`` or ``"setting"``): the planner
  searches the next 24 h for a monotonic source arc of the requested
  direction whose elevation range covers ``el_bore``.
- ``window=(t_start, t_end)``: explicit search window. ``mode`` is
  auto-detected from the longest monotonic arc inside the window
  unless overridden.

When the search window straddles a culmination or anti-culmination the
planner enumerates *all* monotonic arcs (not just the global min→max
pair) and picks the first directional arc that reaches ``el_bore`` --
this matters for sources near opposition where naïve ``argmin``/
``argmax`` selection produces an empty slice.

Tuning knobs
~~~~~~~~~~~~

- ``boresight_rot`` -- mechanical boresight rotation in degrees, added
  to the focal-plane rotation when projecting the cover. ``None``
  (default) means no rotation is commanded.
- ``v_az`` -- override the solved drift rate. Useful for repeatable
  observations and as a cross-check against the optimiser. When given,
  the Nelder-Mead solve is skipped.
- ``az_padding`` -- extra padding (deg) on each side of the solved
  ``[az_start, az_start + az_throw]`` interval. Default 0.5.
- ``az_branch`` -- centre of the azimuth wrap branch. When given,
  ``az_start`` is re-expressed in ``[az_branch − 180, az_branch + 180)``.
- ``allow_partial`` -- if ``False`` (default), raise
  :class:`~fyst_trajectories.TargetNotObservableError` when the
  source's elevation span does not fully cover the footprint at
  ``el_bore``; if ``True``, clip ``(t0, t1)`` to the available source
  el-range and emit a :class:`~fyst_trajectories.PointingWarning`.

Conventions and known unknowns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Focal-plane axes.** ``ArrayFootprint``'s ``(xi, eta)`` correspond
  to :class:`~fyst_trajectories.InstrumentOffset` ``(dx, dy)`` -- ``xi``
  is the cross-elevation direction and ``eta`` is the elevation
  direction at zero field rotation. The parity with the SO
  ``schedlib`` xi/eta convention is a known open question; cross-validate
  against ``schedlib`` before treating the convention as confirmed (note
  also that ``schedlib`` ``array_info`` uses radians while
  :class:`~fyst_trajectories.InstrumentOffset` uses degrees -- see
  :meth:`~fyst_trajectories.ArrayFootprint.from_array_info`).
- **MODULE_FOV_RADIUS_DEG.** The 0.65° constant is a conservative
  Prime-Cam default (wafer ~0.39° at the FYST plate scale). Pass an
  explicit ``ArrayFootprint`` to override.
- **Boresight rotation sign.** The current implementation *adds*
  ``boresight_rot`` to the focal-plane rotation; SO ``schedlib`` uses
  ``quat.euler(2, -np.deg2rad(boresight_rot))`` in its quaternion
  pipeline. Signs agree at ``boresight_rot=0`` (the default); non-zero
  values are a known open question pending a parity check against SO.
- **Refraction.** Like the rest of the planning subpackage, the
  default is vacuum (the FYST ACU applies refraction at execution).
  Pass ``atmosphere=AtmosphericConditions.for_fyst()`` for visibility-
  check style calculations where the output is not sent to the ACU.

.. note::

   ``plan_source_ces`` is **planner-only**: its output is intended for
   Simons Observatory's ``schedlib`` (via a future
   ``schedlib/policies/fyst.py``), not for the in-tree
   :func:`~fyst_trajectories.overhead.generate_timeline` simulator.
   The simulator handles planet calibrations as fixed-duration blocks
   without invoking this planner, so ``"source_ces"`` is intentionally
   not registered with
   :func:`~fyst_trajectories.planning.validate_computed_params` or
   with the overhead-side scan dispatch.

Params-only mode for SO schedlib
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~fyst_trajectories.planning.compute_source_ces_params` is a
params-only sibling of :func:`~fyst_trajectories.planning.plan_source_ces`.
It accepts the same keyword arguments (minus ``timestep`` — only the
trajectory builder consumes it) and returns just the scalar
:class:`~fyst_trajectories.planning.SourceCESComputedParams` dict,
skipping the per-sample trajectory generation. The downstream consumer
that motivates it is a future ``schedlib/policies/fyst.py`` emitting a
``run.acu.source_scan(az_start, az_throw, v_az, el_bore, ...)`` line
for Simons Observatory's Nextline: schedlib needs the scan scalars but
discards the trajectory. On a 15-minute Jupiter scan at
``timestep=0.1`` this saves ~370 KB of trajectory arrays and ~10-20 ms
of vectorised compute per call; at ~30 source-CES blocks per tactical
scheduling pass, that is ~11 MB and ~300-600 ms avoided.

Worked example -- the same Jupiter input as the section above, scalars
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

    # No trajectory; just the scalars the schedlib emitter needs.
    print(f"az_start={params['az_start']:.2f}  az_throw={params['az_throw']:.2f}")
    print(f"v_az={params['v_az']:+.5f} deg/s  el_bore={params['el_bore']:.2f}")
    print(f"t0={params['t0_iso'][:19]}  t1={params['t1_iso'][:19]}")

The returned dict is identical to ``plan_source_ces(...).computed_params``
for the same inputs. ``compute_source_ces_params`` runs an envelope-only
azimuth bounds check (``[az_start, az_start + az_throw]`` widened by
``|v_az| * (t1 - t0)``) before returning, raising
:class:`~fyst_trajectories.AzimuthBoundsError` for clearly out-of-range
scans without building the trajectory. :func:`plan_source_ces` runs the
stricter per-sample bounds check via
:func:`~fyst_trajectories.trajectory_utils.validate_trajectory_bounds`
after building.

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
    - **Source CES** --
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
