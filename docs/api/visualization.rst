Visualization
=============

.. module:: fyst_trajectories.visualization

All matplotlib rendering for fyst-trajectories lives in the
``fyst_trajectories.visualization`` subpackage: target-visibility planning
figures and the observability-window Gantt, the instantaneous all-sky view,
the focal-plane footprint on sky and a source's track across it, trajectory
diagnostics and RA/Dec hit-density maps.

.. note::

   This subpackage requires the ``plotting`` extra:

   .. code-block:: bash

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

   Default target list for :func:`plot_visibility`, :func:`plot_sky_view`
   and :func:`plot_observability_windows`: every ``BODY`` entry of
   :data:`~fyst_trajectories.observability.FLUX_CALIBRATORS` (the planets
   and the Moon; satellites are left out because they duplicate their
   parent body's curve).

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

Omitting ``sun_model`` uses the site's scalar radii; see
:doc:`../sun_avoidance` for the model catalog and how to choose one.

Observability Windows
---------------------

The Gantt view of the same question: one lane per target, a bar per
contiguous interval where every criterion passes (elevation limits, the
selected sun policy, any avoid zones), drawn directly from
:func:`~fyst_trajectories.observability.check_observability`'s
``windows``. A target with no window keeps its empty lane, and night
shading plus sunrise/sunset markers carry the solar context.

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
to compose side-by-side policy panels into one figure.

Focal-Plane Footprint on Sky
----------------------------

The detector-array-on-sky view: all seven Prime-Cam modules drawn at their
true on-sky positions and 0.65° FOV radii for a given boresight elevation,
rotated by the mechanical focal-plane rotation. The axes are to scale
(equal aspect), so the figure answers "which module lands on the source at
this elevation?" and makes the elevation-dependent Nasmyth rotation directly
visible.

.. autofunction:: fyst_trajectories.visualization.plot_array_footprint

**The array at two elevations** (compare the Nasmyth rotation)::

    from fyst_trajectories.visualization import plot_array_footprint

    fig = plot_array_footprint(el=30.0, show=False)
    fig.savefig("footprint_el30.png", dpi=140)

    fig = plot_array_footprint(el=70.0, show=False)
    fig.savefig("footprint_el70.png", dpi=140)

Source Track
------------

Where a source travels across the focal plane during one source-CES pass
(see :doc:`planning`): the module layout in the focal-plane frame, to
scale, with the source's sawtooth track over it. The sweep legs carry the
source back and forth across the swept window while the drift moves it
through the array, so the envelope shows which modules the source crossed
and for how much of the pass.

.. autofunction:: fyst_trajectories.visualization.plot_source_track

**One Jupiter pass through the centre module**::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site, plan_source_ces
    from fyst_trajectories.visualization import plot_source_track

    site = get_fyst_site()
    block = plan_source_ces(
        body="jupiter",
        footprint="c",
        el_bore=35.0,
        night=Time("2026-03-15T00:00:00", scale="utc"),
        mode="rising",
        site=site,
    )
    fig = plot_source_track(block, site=site, show=False)
    fig.savefig("source_track.png", dpi=140)

The track itself comes from
:func:`~fyst_trajectories.planning.source_ces_focal_plane_track`, which
returns the source's ``(xi, eta)`` focal-plane coordinates per trajectory
sample.

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

Generate hit-density maps in RA/Dec for multiple detector modules, from
any built ``Trajectory`` with a ``start_time``::

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.primecam import get_primecam_offset
    from fyst_trajectories.visualization import plot_hit_map

    site = get_fyst_site()

    # Coverage of two PrimeCam modules, each averaged over its field of view
    modules = {
        "module i1": get_primecam_offset("i1"),
        "module i6": get_primecam_offset("i6"),
    }
    fig = plot_hit_map(trajectory, modules=modules, site=site, show=False)
    fig.savefig("coverage_map.png", dpi=300)

**Raw detector-centre tracks**, without the field-of-view average::

    fig = plot_hit_map(
        trajectory, modules=modules, site=site,
        fov_radius_deg=None,  # the default is the 0.65 deg module radius
        show=False,
    )

The two night-level figures, ``plot_timeline_gantt`` and
``plot_sky_coverage``, draw a simulated observing night and are
documented with the timeline they read, in :doc:`overhead_io`.
