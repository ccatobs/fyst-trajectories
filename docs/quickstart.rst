Quickstart
==========

The shortest path from the FYST site constants to a trajectory the
telescope can execute. This page walks the site, a coordinate transform,
two scan patterns and the request body for the FYST telescope control
system's (Go TCS) ``/path`` endpoint; :doc:`planning` and
:doc:`trajectory_examples` go deeper.

The site
--------

Get the FYST site configuration::

    from fyst_trajectories import get_fyst_site

    site = get_fyst_site()
    print(f"FYST is at {site.latitude}, {site.longitude}")  # FYST is at -22.985639, -67.740278

Coordinate Transformations
--------------------------

Convert RA/Dec to Az/El. For trajectory generation, use bare
``Coordinates(site)``. Refraction belongs downstream, in exactly one of
the Go TCS or the ACU; P-INCM-ICD-0003-A section 6 leaves the choice
between them open, so vacuum (geometric) coordinates are correct either
way::

    from astropy.time import Time

    from fyst_trajectories import Coordinates, get_fyst_site

    site = get_fyst_site()
    coords = Coordinates(site)

    # Orion Nebula
    obstime = Time("2026-01-15T02:00:00", scale="utc")
    az, el = coords.radec_to_altaz(ra=83.82, dec=-5.39, obstime=obstime)
    print(f"Orion is at Az={az:.1f}, El={el:.1f}")  # Orion is at Az=23.0, El=71.0

**Frame name translation** (string alias resolution)::

    from fyst_trajectories import FRAME_ALIASES

    print(sorted(FRAME_ALIASES))     # ['B1950', 'FK5', 'HORIZON', 'J2000']

**Proper motion support** (for high proper motion stars)::

    from astropy.time import Time

    from fyst_trajectories import Coordinates, get_fyst_site

    coords = Coordinates(get_fyst_site())

    # Barnard's Star, J2000 catalogue position and proper motion
    az, el = coords.radec_to_altaz_with_pm(
        ra=269.452, dec=4.693,
        pm_ra=-798.58, pm_dec=10328.12,  # mas/yr
        ref_epoch=Time("J2000.0"),
        obstime=Time("2026-06-15T04:00:00"),
    )

See :doc:`coordinate_systems` for more details on supported coordinate systems.

.. _quickstart-planning-refraction:

Planning with Refraction
------------------------

For planning and simulation (visibility calculations, observability
checks, hitmap simulations) where the output is NOT sent to the ACU,
pass :meth:`~fyst_trajectories.site.AtmosphericConditions.for_fyst` to
apply submillimetre-appropriate refraction at typical Cerro Chajnantor
conditions::

    from astropy.time import Time

    from fyst_trajectories import AtmosphericConditions, Coordinates, get_fyst_site

    site = get_fyst_site()
    coords = Coordinates(site, atmosphere=AtmosphericConditions.for_fyst())

    # Visibility check: where is this source right now?
    obstime = Time("2026-01-15T02:00:00", scale="utc")
    az, el = coords.radec_to_altaz(ra=83.82, dec=-5.39, obstime=obstime)

``AtmosphericConditions.no_refraction()`` is an explicit synonym for
vacuum; bare ``Coordinates(site)`` is equivalent.

Trajectory Generation
---------------------

A planner in ``fyst_trajectories.planning`` derives a scan from a field
or a source and returns a ``ScanBlock`` carrying its trajectory;
``TrajectoryBuilder`` and the config classes in the ``patterns`` package
build a trajectory from explicit pattern parameters, in the form the ACU
ProgramTrack mode executes.

The builder infers the pattern type from the config class. Available
patterns: ``constant_el``, ``daisy``, ``daisy_altaz``, ``linear``,
``planet``, ``pong``, ``pong_altaz``, ``satellite``, ``sidereal``. The
first example below goes through a planner and the second through the
builder. :doc:`trajectory_examples` works through the celestial and
AltAz-frame trackers, :doc:`planning` covers the two AltAz-native scans,
and :doc:`api/patterns` documents every config field.

**Constant elevation scan** (auto-computed from a field region, recommended)::

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import FieldRegion, plan_constant_el_scan

    site = get_fyst_site()

    field = FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7)
    block = plan_constant_el_scan(
        field=field,
        elevation=50.0,
        velocity=0.5,
        site=site,
        start_time="2026-03-15T17:00:00",
        az_accel=0.5,
    )
    trajectory = block.trajectory

**Pong scan** (curvy box pattern)::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import PongScanConfig, TrajectoryBuilder

    site = get_fyst_site()
    start_time = Time("2026-03-15T01:00:00", scale="utc")

    config = PongScanConfig(
        timestep=0.1, width=2.0, height=2.0, spacing=0.1,
        velocity=0.4, num_terms=4, angle=0.0,
    )

    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=180.0, dec=-30.0)
        .with_config(config)
        .duration(300.0)
        .starting_at(start_time)
        .build()
    )

**Build the Go TCS /path request body**::

    from fyst_trajectories.trajectory_utils import to_path_payload

    # The three-key body {"start_time", "coordsys", "points"}. The
    # points rows are [t_rel_s, az, el, az_vel, el_vel] with t_rel_s in
    # seconds from trajectory.start_time. Prefer this over assembling the
    # dict by hand: the Go TCS receiver rejects a body with a missing key or
    # an unknown one. to_path_format() returns just the points rows.
    payload = to_path_payload(trajectory)

**Print formatted summary**::

    from fyst_trajectories.trajectory_utils import print_trajectory

    print_trajectory(trajectory)  # Shows first 5 and last 5 points

**Dynamics safety checks** (the builder flags scans that exceed limits)::

    import warnings

    from astropy.time import Time

    from fyst_trajectories import PointingWarning, get_fyst_site
    from fyst_trajectories.patterns import PongScanConfig, TrajectoryBuilder

    site = get_fyst_site()

    # cos(el) inflates the mount-frame azimuth rate at high elevation, so
    # the builder warns and still returns the trajectory.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trajectory = (
            TrajectoryBuilder(site)
            .at(ra=180.0, dec=-30.0)
            .with_config(PongScanConfig(
                timestep=0.1, width=2.0, height=2.0, spacing=0.1,
                velocity=0.5, num_terms=4, angle=0.0,
            ))
            .duration(300.0)
            .starting_at(Time("2026-03-15T04:00:00", scale="utc"))  # high elevation
            .build()
        )

    flagged = [w.message for w in caught if issubclass(w.category, PointingWarning)]
    # `flagged` lists the high-elevation azimuth-rate / acceleration warnings.
    # Observe at a lower elevation or reduce the scan velocity to clear them.

The ``plan_*_scan`` planners run a Sun pre-flight that warns and never
refuses; how much of the scan it covers depends on the planner, from the
field center at the start time to the whole resolved pass.
``validate_sun_avoidance`` checks a built trajectory; the builder itself
performs no Sun check. See :doc:`sun_avoidance` for the check each entry
point runs, the selectable policies, and the dispatch-time gate.

Instrument Offsets
------------------

When an off-axis detector should track the target, ``.for_detector()``
offsets the boresight in the opposite direction, accounting for the
mechanical focal-plane rotation (``nasmyth_sign * elevation +
instrument_rotation``), not the celestial field rotation.

See :doc:`instrument_offsets` for the two rotations, custom offsets from
angular or focal-plane coordinates, and the Prime-Cam module layout.
