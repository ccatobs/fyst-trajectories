Sun Avoidance
=============

FYST's Sun check is one seam. Every entry point that can care about the
Sun takes the same injectable predicate, so a policy chosen once travels
from visibility reporting through night simulation to the dispatch-time
slew gate. The default everywhere is the site's isotropic exclusion
radius (45°, warning at 50°) and needs no extra install. A position
exactly at the exclusion radius counts as unsafe.

Is my target sun-safe?
----------------------

The fastest answer needs two lines and the default policy::

    from astropy.time import Time

    from fyst_trajectories import check_observability

    reports = check_observability(
        ["jupiter", "moon"], Time("2027-01-05T04:00:00", scale="utc")
    )
    for r in reports:
        print(r.name, r.sun_clear, f"{r.sun_separation_deg:.1f} deg")
    # jupiter True 138.4 deg
    # moon False 29.1 deg

``sun_separation_deg`` is always the geometric separation, whichever
policy produced the verdict; the Sun result is also folded into
``r.observable`` and ``r.reasons`` as ``SUN_TOO_CLOSE``. See
:doc:`api/observability` for the full report, observability windows, and
the separate caller-specified bright-source avoid list.

Choosing a policy
-----------------

Three models, one constructor
(:func:`~fyst_trajectories.sun_models.make_sun_safe`). The scalar model
is built in; the other two bind the observatory's shared
`ccatobs/sun-avoidance <https://github.com/ccatobs/sun-avoidance>`_
library, the common home of FYST's Sun-zone geometry:

- ``"scalar"``: the site's isotropic exclusion radius (45°, warning 50°,
  from ``site.sun_avoidance``). The default anywhere ``sun_safe``,
  ``sun_model``, or ``slew_safe`` is omitted. Needs nothing beyond
  fyst-trajectories.
- ``"cone"``: the shared library's isotropic cone at any radius you name.
- ``"cad"``: the same library's directional CAD-derived zone. The minimum
  Sun separation runs 50-90° with the Sun's direction in the mount frame,
  which is FYST's own hardware model. Opt-in today, and expected to
  become the FYST default in a future release.

The shared library is an optional dependency, deliberately not bundled
with fyst-trajectories while the scalar model is the default and the
library has no packaged release (requesting ``"cone"`` or ``"cad"``
without it raises with this exact command); it is expected to ship
automatically once the directional model becomes the default. Install it
from git at the pinned revision::

    pip install "git+https://github.com/ccatobs/sun-avoidance@e6fa12aa53ce5f5f76d50f8b753e7fe4b4ad8e18"

That repository is CCAT-internal, so the command above requires
collaboration access. The default ``"scalar"`` model requires none of
it and is what you get if you omit ``sun_model`` entirely.

The scalar radius is an observing policy, not the directional zone's
inscribed cone: at 45° it is more permissive than the CAD model in every
direction, since that model requires 50-90°. Choose ``"cad"`` explicitly
when a scan needs mirror-illumination protection rather than the
observing baseline. (Note :func:`~fyst_trajectories.sun_models.make_sun_safe`
itself defaults to ``model="cad"``; the library-wide default when you
inject nothing is the scalar.)

The models disagree wherever the directional zone exceeds the scalar
radius::

    from astropy.time import Time

    from fyst_trajectories.sun_models import make_sun_safe

    t = Time("2026-11-15T20:30:00", scale="utc")  # Sun at az 261, el 31
    models = (
        make_sun_safe("scalar"),
        make_sun_safe("cone", radius=50.0),
        make_sun_safe("cad"),
    )
    for model in models:
        print(model.describe, model(150.0, 45.0, t))
    # scalar 45°        True
    # cone 50°          True
    # CAD zone 50-90°   False

Planning a night
----------------

Over a horizon, the injected policy shapes the observable windows (here:
which calibrators are observable tonight under the CAD policy)::

    from astropy.time import Time

    from fyst_trajectories import check_observability
    from fyst_trajectories.sun_models import make_sun_safe

    sun_safe = make_sun_safe("cad")
    reports = check_observability(
        ["jupiter", "uranus", "moon"],
        Time("2026-11-15T16:00:00", scale="utc"),
        horizon_hours=24.0,
        sun_safe=sun_safe,
    )
    for r in reports:
        print(r.name, r.sun_clear, f"{r.total_observable_hours:.1f} h")

For the Gantt view of those windows (one bar lane per target, the
"which chunks of tonight" chart), see
:func:`~fyst_trajectories.visualization.plot_observability_windows` in
:doc:`api/visualization`.

The offline overhead simulator takes the same predicate; it drives both
the patch-selection Sun constraint and the mid-scan duration clips, so a
science scan is cut short rather than run into the zone::

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.overhead import ObservingPatch, generate_timeline
    from fyst_trajectories.sun_models import make_sun_safe

    patch = ObservingPatch(
        name="field", ra_center=24.0, dec_center=-32.0, width=10.0,
        height=10.0, scan_type="pong", velocity=1.0, elevation=50.0,
    )
    night = generate_timeline(
        [patch], get_fyst_site(),
        "2026-11-15T02:00:00", "2026-11-15T06:00:00",
        sun_safe=make_sun_safe("cad"),
    )

See :doc:`overhead_quickstart` for the simulator itself.

Gating a slew at dispatch
-------------------------

Dispatch is the one place the Sun check refuses instead of warns. A
correct point predicate is invariant under ``az -> az + 360`` (the same
sky direction), so it can never *choose* an azimuth wrap; the two wraps
differ in the path swept between them, and that is a path question,
answered by sweeping the point model along the trapezoidal slew
(:func:`~fyst_trajectories.sun_models.make_slew_safe`)::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.dispatch import choose_encoder_solution
    from fyst_trajectories.sun_models import make_slew_safe, make_sun_safe

    t = Time("2026-11-15T20:30:00", scale="utc")
    az_cmd, el_cmd = choose_encoder_solution(
        65.0, 45.0, 100.0, 45.0, t, get_fyst_site(),
        sun_safe=make_sun_safe("cad"), slew_safe=make_slew_safe("cad"),
    )
    # command the slew to (az_cmd, el_cmd), then POST the trajectory.

When no wrap has a clear direct path, dispatch raises ``PointingError``
rather than rerouting. A caller who wants a two-leg detour plans it
explicitly with
:func:`~fyst_trajectories.sun_models.find_sun_safe_detour`, which
returns ``None`` more often than not under FYST's own policies (the
zones span most of the elevation range and the azimuth axis outruns
elevation about 3 to 1), so the wrap choice above, or waiting, is
usually the real recourse. See :doc:`api/dispatch`.

Seeing the zone
---------------

One call draws the whole sky with the policy's own verdicts shaded on an
az/el grid, so the directional zone renders its true asymmetric shape
instead of a circle::

    from astropy.time import Time

    from fyst_trajectories.visualization import plot_sky_view

    fig = plot_sky_view(
        Time("2026-11-15T18:00:00", scale="utc"),
        sun_model="cad",
        boresight="moon",
        show=False,
    )
    fig.savefig("sky_view_cad.png", dpi=140, bbox_inches="tight")

For a whole night rather than an instant,
``plot_visibility(..., sun_model="cad")`` overdraws each target where
the policy marks it unsafe and adds a per-target minimum-separation
curve. Both require the ``plotting`` extra. See :doc:`api/visualization`.

Where the check runs
--------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Entry point
     - What it does
   * - ``plan_*_scan(..., sun_safe=)``
     - Pre-flight on the field center at the start time. Warns
       (``PointingWarning``), never refuses.
   * - ``validate_sun_avoidance(site, az, el, times)``
     - Checks the whole trajectory span (subsampled). Warns once: at or
       inside the exclusion radius, otherwise inside the warning radius.
       Advisory only.
   * - ``check_observability(..., sun_safe=)``
     - Reports ``sun_clear`` and ``SUN_TOO_CLOSE``. Never raises.
   * - ``generate_timeline(..., sun_safe=)``
     - The offline night simulator. Excludes unsafe patches and clips
       scans that would run into the zone.
   * - ``choose_encoder_solution(..., sun_safe=, slew_safe=)``
     - Dispatch. Raises ``PointingError`` when no azimuth wrap is clear.
   * - ``get_fyst_site(sun_exclusion_radius=..., sun_warning_radius=...)``
     - Overrides the scalar radii; ``sun_avoidance_enabled=False``
       disables the check entirely (tests and engineering only).

The telescope control system enforces its own hard limits independently.
Nothing here is an interlock.

See also
--------

- :doc:`api/sun_models` - the module reference.
- :doc:`api/observability` - reports, windows, and avoid zones.
- :doc:`api/dispatch` - the encoder-choice gate.
- :doc:`api/visualization` - visibility curves and the all-sky view.
