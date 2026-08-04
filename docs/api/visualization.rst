Visualization
=============

.. module:: fyst_trajectories.visualization

All matplotlib rendering for fyst-trajectories lives in the
``fyst_trajectories.visualization`` subpackage: target-visibility planning
figures, the instantaneous all-sky view, the focal-plane footprint on sky,
trajectory diagnostics, RA/Dec hit-density maps, and night-level
overhead-timeline figures.

.. note::

   This subpackage requires the ``plotting`` extra::

       pip install "fyst-trajectories[plotting]"

   This installs ``matplotlib``. Importing ``fyst_trajectories`` (or any
   of its subpackages) never imports matplotlib; each plot function
   loads it lazily on first call.

Target Visibility Planning
--------------------------

One call renders the calibration-planning figure for an observing night:
per-target elevation and azimuth curves, the Sun's own track, night and
astronomical-night shading with sunrise/sunset and twilight markers (from
:func:`~fyst_trajectories.observability.sun_events`), the telescope
elevation floor, and sun-proximity highlighting on each target curve.

.. autofunction:: fyst_trajectories.visualization.plot_visibility

.. py:data:: DEFAULT_VISIBILITY_TARGETS
   :type: tuple[str, ...]

   Default target list for :func:`plot_visibility` and
   :func:`plot_sky_view`: every BODY entry of
   :data:`~fyst_trajectories.observability.FLUX_CALIBRATORS` (the planets
   and the Moon; satellites are excluded because they duplicate their
   parent-body proxy curve).

**Tonight's calibrators from FYST** (planets + Moon, elevation and
azimuth panels, sun zones from the site configuration)::

    from astropy.time import Time

    from fyst_trajectories.visualization import plot_visibility

    fig = plot_visibility(Time("2026-11-15T16:00:00", scale="utc"), show=False)
    fig.savefig("visibility.png", dpi=140, bbox_inches="tight")

**Chilean local time, chosen targets, and a Sun-separation panel**::

    from zoneinfo import ZoneInfo

    fig = plot_visibility(
        Time("2026-11-15T16:00:00", scale="utc"),
        ["jupiter", "uranus", "neptune"],
        tz=ZoneInfo("America/Santiago"),
        panels=("elevation", "azimuth", "sun_separation"),
        show=False,
    )
    fig.savefig("visibility_local.png", dpi=140, bbox_inches="tight")

Computation is always UTC; ``tz`` changes only the axis labels. Fixed
RA/Dec sources enter through ``extra_targets``, exactly as in
:func:`~fyst_trajectories.observability.check_observability`.

**Selectable sun-avoidance policy.** The overlay model is injectable:
with the shared sun-avoidance library installed, ``sun_model="cad"``
drives the overlays and the separation panel from FYST's directional
CAD zone, so each target is over-drawn where *that* policy marks it
unsafe and carries its own direction-dependent minimum-separation
curve::

    fig = plot_visibility(
        Time("2026-11-15T16:00:00", scale="utc"),
        sun_model="cad",
        panels=("elevation", "azimuth", "sun_separation"),
        show=False,
    )
    fig.savefig("visibility_cad.png", dpi=140, bbox_inches="tight")

See :doc:`../sun_avoidance` for the model catalog and how to choose one.
The site's scalar radii are today's default policy; FYST's directional
CAD zone is opt-in via
``fyst_trajectories.sun_models.make_sun_safe("cad")`` and is expected to
become the default in a future release.

Observability Windows
---------------------

The Gantt view of the same question: one lane per target, a bar per
contiguous interval where every criterion passes (elevation limits, the
selected sun policy, any avoid zones), drawn directly from
:func:`~fyst_trajectories.observability.check_observability`'s
``windows``. A target with no window keeps its empty lane, so "never
observable tonight" is visible rather than missing, and night shading
plus sunrise/sunset markers carry the solar context.

.. autofunction:: fyst_trajectories.visualization.plot_observability_windows

**Which chunks of tonight can be used for which calibrator** (elevation
floor 30°, default sun policy)::

    from astropy.time import Time

    from fyst_trajectories.visualization import plot_observability_windows

    fig = plot_observability_windows(
        Time("2026-11-29T00:00:00", scale="utc"),
        el_min=30.0,
        show=False,
    )
    fig.savefig("windows.png", dpi=140, bbox_inches="tight")

Pass ``sun_model="cad"`` to compute the windows under the directional
policy instead; the legend always states the criteria in force.

Instantaneous All-Sky View
--------------------------

The whole sky at one moment as a polar chart: zenith at the center,
horizon on the rim, north up and east to the left. The projection is
azimuthal equidistant, exact radially (the radius is literally the zenith
angle) and tangentially stretched toward the rim, so measure on-sky sizes
with :func:`plot_array_footprint`, not here.

.. autofunction:: fyst_trajectories.visualization.plot_sky_view

**This afternoon's sky with the array on the Moon** (default scalar
policy from the site configuration)::

    from astropy.time import Time

    from fyst_trajectories.visualization import plot_sky_view

    fig = plot_sky_view(
        Time("2026-11-15T18:00:00", scale="utc"),
        boresight="moon",
        show=False,
    )
    fig.savefig("sky_view.png", dpi=140, bbox_inches="tight")

**The same sky under the directional CAD policy** (requires the shared
sun-avoidance library; compare the asymmetric zone against the scalar
circle)::

    fig = plot_sky_view(
        Time("2026-11-15T18:00:00", scale="utc"),
        sun_model="cad",
        boresight="moon",
        show=False,
    )
    fig.savefig("sky_view_cad.png", dpi=140, bbox_inches="tight")

Pass a polar axes via ``ax=`` (``fig.add_subplot(..., projection="polar")``)
to compose side-by-side policy panels into one figure. Two layers other
all-sky viewers draw are deliberately absent, blocked on data rather than
code: the surveyed landscape horizon (no Cerro Chajnantor skyline survey
exists, and the telescope elevation floor dominates the terrain from the
summit) and site-structure occlusion (no FYST as-built model).

Focal-Plane Footprint on Sky
----------------------------

The detector-array-on-sky view: all seven PrimeCam modules drawn at their
true on-sky positions and 0.65° FOV radii for a given boresight elevation,
rotated by the mechanical Nasmyth field rotation. The axes are to scale
(equal aspect), so the figure answers "which module lands on the source at
this elevation?" honestly, and makes the elevation-dependent field
rotation directly visible.

.. autofunction:: fyst_trajectories.visualization.plot_array_footprint

**The array at two elevations** (compare the Nasmyth rotation)::

    from fyst_trajectories.visualization import plot_array_footprint

    fig = plot_array_footprint(el=30.0, show=False)
    fig.savefig("footprint_el30.png", dpi=140)

    fig = plot_array_footprint(el=70.0, show=False)
    fig.savefig("footprint_el70.png", dpi=140)

Trajectory Diagnostics
----------------------

.. autofunction:: fyst_trajectories.visualization.plot_trajectory

Any built ``Trajectory`` (see :doc:`patterns`) renders as the three
diagnostic panels::

    from fyst_trajectories.visualization import plot_trajectory

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
        (get_primecam_offset("i1"), "module i1"),
        (get_primecam_offset("i6"), "module i6"),
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

.. autofunction:: fyst_trajectories.visualization.plot_timeline_gantt

.. autofunction:: fyst_trajectories.visualization.plot_sky_coverage
