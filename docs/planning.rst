Planning Module
===============

Astronomer-friendly wrappers that translate field coordinates, elevation
constraints, and scan velocities into full pattern configurations.

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
        velocity=0.4,        # deg/s
        spacing=0.1,         # deg between scan lines
        num_terms=4,         # Fourier terms for smooth turnarounds
        site=site,
        start_time=Time("2026-03-15T01:00:00", scale="utc"),
        timestep=0.1,
    )

    print(block.summary)
    print(f"Duration: {block.duration:.1f}s ({block.duration / 3600:.1f}h)")
    print(f"Trajectory: {block.trajectory.n_points} points")

A planning function wraps each scan type an astronomer specifies by sky
geometry; how much it computes between those inputs and the pattern config
varies:

- **Pong** - computes the Pong period from field dimensions, spacing, and
  velocity.
- **Constant-El** - derives the timing, azimuth range and ``n_scans`` from
  an elevation crossing or a Local Sidereal Angle window.
- **Daisy** - convenience wrapper; parameters map nearly 1:1 to the config.
- **Source CES** - drags a moving source across an instrument-array
  footprint at fixed boresight elevation.
- **AltAz Pong / Daisy** - run the Pong and Daisy patterns about a fixed
  horizon-frame center (no RA/Dec tracking).

Sidereal, planet, satellite, and linear patterns have no non-trivial planning
step; use :class:`~fyst_trajectories.patterns.TrajectoryBuilder` directly.

Field Regions
-------------

A :class:`~fyst_trajectories.planning.FieldRegion` defines a rectangular sky area
by its center coordinates and angular extent::

    from fyst_trajectories.planning import FieldRegion

    field = FieldRegion(
        ra_center=0.0,     # deg
        dec_center=-2.0,   # deg
        width=10.0,        # RA extent in degrees
        height=6.0,        # Dec extent in degrees
    )

    # Dec boundaries are computed automatically
    print(f"Dec range: [{field.dec_min}, {field.dec_max}]")
    # Dec range: [-5.0, 1.0]

Planning a Pong Scan
--------------------

:func:`~fyst_trajectories.planning.plan_pong_scan` converts a field region into a
Pong scan trajectory, generating one full Pong period by default. Beyond
the Quick Start call it takes a pattern rotation and a cycle count::

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
        start_time=Time("2026-03-15T23:30:00", scale="utc"),
        timestep=0.1,
        angle=170.0,     # rotation angle (degrees)
        n_cycles=3,      # observe 3 full Pong periods
    )

``detector_offset`` shifts the boresight so an off-axis PrimeCam module
tracks the field::

    from fyst_trajectories.primecam import get_primecam_offset

    block = plan_pong_scan(
        field=field, velocity=0.5, spacing=0.1, num_terms=4, site=site,
        start_time=Time("2026-03-15T01:00:00", scale="utc"), timestep=0.1,
        detector_offset=get_primecam_offset("i1"),
    )

Multi-Rotation Pong Tiling
--------------------------

:func:`~fyst_trajectories.planning.plan_pong_rotation_sequence` returns
``n_rotations`` copies of a base
:class:`~fyst_trajectories.patterns.PongScanConfig` with the ``angle``
field overridden to a uniform ``180° / n_rotations`` sequence. Each
returned config is passed individually through
:func:`~fyst_trajectories.planning.plan_pong_scan`::

    from fyst_trajectories import PongScanConfig, get_fyst_site
    from fyst_trajectories.planning import plan_pong_rotation_sequence

    site = get_fyst_site()
    base = PongScanConfig(
        timestep=0.1, width=2.0, height=2.0,
        spacing=0.1, velocity=0.35, num_terms=4, angle=0.0,
    )

    # 8 rotations at 22.5 deg spacing, each scheduled by its own
    # plan_pong_scan(..., angle=c.angle) call.
    configs = plan_pong_rotation_sequence(base, n_rotations=8)
    print([c.angle for c in configs])
    # [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]

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
meaning from the celestial Pong. Budget ``velocity`` against the mount
azimuth rate limit accordingly.

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
        velocity=0.4,        # deg/s on-sky
        site=site,
        start_time=Time("2026-03-15T04:00:00", scale="utc"),
    )

    print(block.summary)
    print(f"Duration: {block.duration:.1f}s")

The duration defaults to one full Pong period; pass ``n_cycles`` to observe
several. ``num_terms``, ``angle``, ``timestep`` and ``detector_offset``
behave as in :func:`~fyst_trajectories.planning.plan_pong_scan`;
``start_time`` anchors the trajectory and fixes the instant of the warn-only
Sun pre-flight check, which is taken on the horizon-frame center directly.

Planning a Constant-Elevation Scan
-----------------------------------

:func:`~fyst_trajectories.planning.plan_constant_el_scan` auto-computes
the azimuth range, observation duration, and number of scans from a
``FieldRegion``, target elevation, and approximate start time, and
returns a :class:`~fyst_trajectories.planning.ScanBlock`. Timing comes
from the next crossing of the target elevation by the field's RA edges
at or after ``start_time``, so the scan begins at that crossing rather
than literally at ``start_time``.

Other knobs: ``az_padding`` (extra azimuth margin on each side, default
2.0 deg here; the source-CES planner's same-named knob defaults to
0.5), ``az_accel`` (the turnaround acceleration, in mount-frame azimuth
degrees per second squared rather than on-sky; default 1.0), and
``max_search_hours`` (how far past ``start_time`` the crossing search
looks, 12 h by default; a field whose crossing lies beyond it raises
``ValueError``).

Basic usage::

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import FieldRegion, plan_constant_el_scan

    site = get_fyst_site()

    field = FieldRegion(ra_center=0.0, dec_center=-2.0, width=10.0, height=6.0)
    block = plan_constant_el_scan(
        field=field,
        elevation=45.0,          # fixed elevation in degrees
        velocity=0.5,            # az scan speed in deg/s
        site=site,
        start_time="2026-09-15T00:00:00",
        rising=True,             # use rising crossing
    )

    print(block.summary)
    print(f"Duration: {block.duration:.0f}s")
    print(f"Az range: [{block.computed_params['az_start']:.1f}, "
          f"{block.computed_params['az_stop']:.1f}]")

``detector_offset`` behaves as in
:func:`~fyst_trajectories.planning.plan_pong_scan`.

LSA-windowed timing
~~~~~~~~~~~~~~~~~~~~

Instead of deriving timing from RA-edge elevation crossings, pass
``lsa_window=(min_lsa, max_lsa)`` (degrees) to pin the scan to a Local
Sidereal Angle window. The scan spans ``((max_lsa - min_lsa) mod 360) / 15``
hours of UTC, about 0.3 percent longer than the sidereal window it names,
and ``block.duration`` is that span quantised to whole azimuth legs.
Wrap-around windows (``max_lsa < min_lsa``, e.g. ``(310.0, 10.0)``) are
supported. The window fixes the timing and the azimuth range together, so
there is no crossing half left to choose and ``rising`` is not accepted
beside it::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import FieldRegion, plan_constant_el_scan

    site = get_fyst_site()

    field = FieldRegion(ra_center=30.0, dec_center=-47.0, width=10.0, height=6.0)
    block = plan_constant_el_scan(
        field=field,
        elevation=45.0,
        velocity=0.5,
        site=site,
        start_time=Time("2026-09-15T00:00:00", scale="utc"),
        lsa_window=(310.0, 330.0),   # 20 deg / 15 = 1.33 h scan
    )

    print(f"Duration: {block.duration / 3600:.1f}h")  # Duration: 1.3h

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
:func:`~fyst_trajectories.planning.plan_daisy_scan` about a fixed
horizon-frame center, under the same ``1 / cos(el_center)`` azimuth
mapping; the azimuth-coordinate extent is about ``2 * r_max /
cos(el_center)``, where the petal's reach ``r_max = sqrt(radius**2 +
turn_radius**2) + turn_radius``::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import plan_daisy_altaz_scan

    site = get_fyst_site()

    block = plan_daisy_altaz_scan(
        az_center=120.0,        # deg
        el_center=60.0,         # deg (fixed; no sky tracking)
        radius=0.5,
        velocity=0.3,
        turn_radius=0.2,
        avoidance_radius=0.0,
        start_acceleration=0.5,
        site=site,
        start_time=Time("2026-03-15T04:00:00", scale="utc"),
        timestep=0.1,
        duration=300.0,
    )

    print(block.summary)
    print(f"Duration: {block.duration:.1f}s")

``timestep``, ``duration``, ``y_offset``, ``detector_offset`` and
``start_time`` behave as in
:func:`~fyst_trajectories.planning.plan_daisy_scan`.

Planning a Source CES (Planet / Sidereal Drift)
------------------------------------------------

:func:`~fyst_trajectories.planning.plan_source_ces` drags a moving source
(planet or sidereal point) across an instrument-array footprint at a fixed
boresight elevation ``el_bore``, solving for the azimuth drift rate ``v_az``
that keeps the swept window on the source while its own elevation motion
carries it across the array. It mirrors Simons Observatory's
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
``"i1"`` .. ``"i6"``), a single
:class:`~fyst_trajectories.offsets.InstrumentOffset`, a sequence of offsets
(one per module), or an explicit
:class:`~fyst_trajectories.planning.ArrayFootprint`. For a multi-module
footprint, :func:`~fyst_trajectories.primecam.resolve_module_tag` expands a
comma-separated tag (``resolve_module_tag("i1,i2")``, or ``"all"``) into that
sequence; see :doc:`instrument_offsets`.

Select the time window with either ``night`` + ``mode`` (``"rising"`` or
``"setting"``; searches the next 24 h) or an explicit
``window=(t_start, t_end)``.

Other knobs: ``boresight_rot`` (mechanical boresight rotation, deg),
``v_az`` (override the solved drift rate), ``az_padding`` (extra azimuth
margin, default 0.5 deg), ``az_branch`` (centre of the azimuth wrap branch),
and ``allow_partial`` (clip to the observable arc and warn instead of
raising when the source does not fully cover the footprint at ``el_bore``).
A sidereal source takes ``ra=`` / ``dec=`` (optionally ``pm_ra`` /
``pm_dec`` / ``ref_epoch``) in place of ``body=``.

By default the pass is a slow drag: the telescope sweeps the solved window
once down and up over the whole footprint crossing. Three optional inputs
turn it into a faster, narrower or shorter measurement while keeping the
same drift solve and Sun check: ``az_speed`` (deg/s) sets the per-leg
speed of the sweep, distinct from the drift rate ``v_az`` that keeps the
window on the source; ``az_throw`` (deg) replaces the solved, padded
window with an explicit one re-centred on it (not combinable with an
explicit ``az_padding``; narrower than the footprint crossing warns); and
``dwell`` (s) narrows the pass symmetrically about the crossing midpoint
(longer than the crossing is rejected, shorter warns as a partial pass).
``computed_params`` records the speed used as ``az_speed`` and the full
crossing as ``crossing_seconds`` whether or not the inputs were given.

To see where the source actually travelled across the focal plane during a
planned pass, :func:`~fyst_trajectories.planning.source_ces_focal_plane_track`
returns its ``(xi, eta)`` coordinates per trajectory sample, and
:func:`~fyst_trajectories.visualization.plot_source_track` draws them over
the module layout (see :doc:`api/visualization`).

Anchoring to an approximate start time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A scheduler often knows only that it is now *T* and the telescope is free,
not which boresight elevation to use. Pass ``start_time`` (mutually
exclusive with ``night`` and ``window``) to plan a pass beginning near that
time. With ``el_bore`` omitted the planner derives it so the pass starts at
or just after the anchor; with ``mode`` omitted it reads the rising or
setting direction from the source's elevation slope there::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import plan_source_ces

    site = get_fyst_site()

    block = plan_source_ces(
        body="jupiter",
        footprint="c",
        start_time=Time("2026-03-15T21:41:00", scale="utc"),
        site=site,
    )
    cp = block.computed_params
    print(f"mode={cp['mode']}  el_bore={cp['el_bore']:.2f}  start={cp['t0_iso'][:19]}")

The resolved start typically lands within about a minute after
``start_time`` and never before it. Supplying ``el_bore`` alongside instead
forward-searches from the anchor for that elevation. An anchor near transit
is rejected with
:class:`~fyst_trajectories.exceptions.TargetNotObservableError`, the
elevation-crossing inversion being ill-conditioned there; anchor away from
transit or pass ``el_bore``.
:func:`~fyst_trajectories.planning.compute_source_ces_params` and
:func:`~fyst_trajectories.planning.plan_source_ces_passes` accept
``start_time`` on the same terms, the anchor applying to the first pass in
time.

Multiple passes for full focal-plane coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A single :func:`~fyst_trajectories.planning.plan_source_ces` paints one band
of the focal plane, so covering every detector takes several passes stepped
through different rows of the array.

:func:`~fyst_trajectories.planning.plan_source_ces_passes` builds that
sequence. It offsets the *footprint* along the focal-plane elevation (eta)
axis to reach a new row, because stepping ``el_bore`` alone does not move
the coverage: a source-tracking scan re-centres on the source at every
elevation, so each ``el_bore`` reproduces the same band. Independently it
steps ``el_bore`` to sequence the passes in time so they do not overlap.
The result is a time-ordered ``list`` of ordinary source-CES blocks::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import plan_source_ces_passes

    site = get_fyst_site()

    passes = plan_source_ces_passes(
        body="jupiter",
        footprint="c",          # one PrimeCam module
        el_bore=35.0,
        n_passes=3,             # three drift passes tiling the module in eta
        night=Time("2026-03-15T00:00:00", scale="utc"),
        mode="rising",
        site=site,
    )

    for block in passes:
        pp = block.trajectory.metadata.pattern_params
        print(
            f"pass {pp['pass_index']}: eta_offset={pp['pass_eta_offset_deg']:+.2f} deg, "
            f"el_bore={pp['pass_el_bore_deg']:.2f} deg, {block.duration:.0f}s"
        )

Specify the pass grid with either ``n_passes`` (plus an optional ``step``,
defaulting to the footprint eta extent divided by ``n_passes``) or an
explicit ``eta_offsets`` list in degrees. Each pass covers more than its own
step, since focal-plane rotation mixes the azimuth throw into eta, so
successive passes interleave rather than painting disjoint bands. The
``el_step`` knob (default: the footprint eta extent) sets how far
``el_bore`` moves between passes; below the extent the passes pack closer,
their source windows overlap in time, and the planner warns. Each block
carries its ``pass_index``, ``pass_eta_offset_deg`` and
``pass_el_bore_deg`` in ``trajectory.metadata.pattern_params``.

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

    from fyst_trajectories.planning import compute_source_ces_params

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

The returned dict is identical to ``plan_source_ces(...).computed_params``
for the same inputs.

Shared Parameters: Sun Safety and Refraction
--------------------------------------------

Every planner on this page except ``plan_pong_rotation_sequence``
(which only produces configs) accepts two cross-cutting keyword
parameters.

``sun_safe=`` injects a Sun-safety predicate
(:class:`~fyst_trajectories.dispatch.SunSafePredicate`) into the
planner's pre-flight check in place of the built-in scalar radius. This
is the injection point for the directional CAD-derived model::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import FieldRegion, plan_pong_scan
    from fyst_trajectories.sun_models import make_sun_safe

    site = get_fyst_site()
    field = FieldRegion(ra_center=180.0, dec_center=-30.0, width=2.0, height=2.0)
    block = plan_pong_scan(
        field=field,
        velocity=0.4,
        spacing=0.1,
        num_terms=4,
        site=site,
        start_time=Time("2026-03-15T01:00:00", scale="utc"),
        timestep=0.1,
        sun_safe=make_sun_safe("cad"),  # directional pre-flight, not the scalar
    )

The pre-flight warns (``PointingWarning``), it never refuses; the
refusing gate lives at dispatch. See :doc:`sun_avoidance` for the
policies and every place the predicate travels.

``atmosphere=`` sets the refraction model of the planner's underlying
:class:`~fyst_trajectories.coordinates.Coordinates`. The default ``None``
means vacuum; pass ``AtmosphericConditions.for_fyst()`` only for output
that never reaches the telescope.
``plan_pong_altaz_scan`` and ``plan_daisy_altaz_scan`` accept it for parity but
run no coordinate transform, so it does not change their output. See
:ref:`quickstart-planning-refraction` for why that default is
load-bearing.

Scan Block Output
-----------------

The six single-scan planners (``plan_pong_scan``, ``plan_constant_el_scan``,
``plan_daisy_scan``, ``plan_pong_altaz_scan``, ``plan_daisy_altaz_scan`` and
``plan_source_ces``) return a
:class:`~fyst_trajectories.planning.ScanBlock`. The other three entry
points do not: ``plan_source_ces_passes`` returns a list of these blocks
(one per pass), ``plan_pong_rotation_sequence`` returns a list of configs,
and ``compute_source_ces_params`` returns a bare dict, as shown in their
sections above.

A block carries the built trajectory, the pattern config behind it, the
duration, the derived ``computed_params`` and a human-readable
``summary``; :doc:`api/planning` documents each field and the per-scan-type
key set of ``computed_params``.
