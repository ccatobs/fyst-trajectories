Quickstart
==========

Basic Usage
-----------

Get the FYST site configuration::

    from fyst_trajectories import get_fyst_site

    site = get_fyst_site()
    print(f"FYST is at {site.latitude}, {site.longitude}")

Coordinate Transformations
--------------------------

Convert RA/Dec to Az/El. For trajectory generation, use bare
``Coordinates(site)`` -- the FYST ACU applies atmospheric refraction
downstream, so vacuum (geometric) coordinates are correct::

    from astropy.time import Time

    from fyst_trajectories import Coordinates, get_fyst_site

    site = get_fyst_site()
    coords = Coordinates(site)

    # Orion Nebula
    obstime = Time("2026-01-15T02:00:00", scale="utc")
    az, el = coords.radec_to_altaz(ra=83.82, dec=-5.39, obstime=obstime)
    print(f"Orion is at Az={az:.1f}, El={el:.1f}")

.. note::

   ``Coordinates(site)`` defaults to vacuum because the FYST ACU
   applies atmospheric refraction at execution time. Applying
   refraction here would cause double-refraction errors in the
   telescope pointing. For planning and simulation (visibility
   calculations, observability checks) where results are NOT sent to
   the ACU, use ``AtmosphericConditions.for_fyst()`` -- see
   :ref:`quickstart-planning-refraction` below.

**Frame name translation** (string alias resolution)::

    from fyst_trajectories import FRAME_ALIASES, normalize_frame

    # Translate common frame names to astropy equivalents
    astropy_frame = normalize_frame("J2000")    # Returns "icrs"
    astropy_frame = normalize_frame("B1950")    # Returns "fk4"

**Proper motion support** (for high proper motion stars)::

    from astropy.time import Time

    from fyst_trajectories import Coordinates, get_fyst_site

    coords = Coordinates(get_fyst_site())

    # Barnard's Star with proper motion
    az, el = coords.radec_to_altaz_with_pm(
        ra=269.452, dec=4.693,
        pm_ra=-798.58, pm_dec=10328.12,  # mas/yr
        ref_epoch=Time("J2015.5"),
        obstime=Time("2026-06-15T04:00:00"),
        distance=1.8,  # parsecs
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

``AtmosphericConditions.no_refraction()`` is available as an explicit
synonym for vacuum when you want to be self-documenting about the
choice, but bare ``Coordinates(site)`` is equivalent and cleaner.

Trajectory Generation
---------------------

The ``patterns`` package provides ``TrajectoryBuilder`` and config classes
for generating telescope trajectories compatible with the ACU ProgramTrack mode.

The pattern type is automatically inferred from the config class you provide.
Available patterns: ``constant_el``, ``daisy``, ``linear``, ``planet``, ``pong``, ``sidereal``.

**Track a celestial source** (sidereal tracking)::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import SiderealTrackConfig, TrajectoryBuilder

    site = get_fyst_site()
    start_time = Time("2026-01-15T02:00:00", scale="utc")

    # Track the Crab Nebula for 5 minutes
    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=83.633, dec=22.014)
        .with_config(SiderealTrackConfig(timestep=0.1))
        .duration(300.0)
        .starting_at(start_time)
        .build()
    )
    print(f"Generated {trajectory.n_points} points")

**Track a planet**::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import PlanetTrackConfig, TrajectoryBuilder

    site = get_fyst_site()
    start_time = Time("2026-03-15T16:00:00", scale="utc")

    trajectory = (
        TrajectoryBuilder(site)
        .with_config(PlanetTrackConfig(timestep=0.1, body="mars"))
        .duration(600.0)
        .starting_at(start_time)
        .build()
    )

**Constant elevation scan** (auto-computed from a field region --- recommended)::

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

**Constant elevation scan** (manual parameters --- for engineering or known az ranges)::

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import ConstantElScanConfig, TrajectoryBuilder

    site = get_fyst_site()

    config = ConstantElScanConfig(
        timestep=0.1,
        az_start=120.0,
        az_stop=150.0,
        elevation=45.0,
        az_speed=1.0,
        az_accel=0.5,
    )

    trajectory = (
        TrajectoryBuilder(site)
        .with_config(config)
        .duration(300.0)
        .build()
    )

**Pong scan** (curvy box pattern for wide-field mapping)::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import PongScanConfig, TrajectoryBuilder

    site = get_fyst_site()
    start_time = Time("2026-03-15T02:00:00", scale="utc")

    config = PongScanConfig(
        timestep=0.1, width=2.0, height=2.0, spacing=0.1,
        velocity=0.3, num_terms=4, angle=0.0,
    )

    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=180.0, dec=-30.0)
        .with_config(config)
        .duration(300.0)
        .starting_at(start_time)
        .build()
    )

**Daisy scan** (petal pattern for point sources)::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import DaisyScanConfig, TrajectoryBuilder

    site = get_fyst_site()
    start_time = Time("2026-03-15T02:00:00", scale="utc")

    config = DaisyScanConfig(
        timestep=0.1, radius=0.5, velocity=0.3, turn_radius=0.2,
        avoidance_radius=0.0, start_acceleration=0.5, y_offset=0.0,
    )

    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=180.0, dec=-30.0)
        .with_config(config)
        .duration(300.0)
        .starting_at(start_time)
        .build()
    )

**Dynamics safety checks** (the builder flags scans that exceed limits)::

    import warnings

    from astropy.time import Time

    from fyst_trajectories import PointingWarning, get_fyst_site
    from fyst_trajectories.patterns import PongScanConfig, TrajectoryBuilder

    site = get_fyst_site()

    # A 2 deg field scanned at 0.5 deg/s with the target near zenith: cos(el)
    # inflates the azimuth rate and acceleration past the telescope limits, so
    # the builder emits PointingWarnings (it still returns the trajectory).
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
            .starting_at(Time("2026-03-15T04:00:00", scale="utc"))  # near zenith
            .build()
        )

    flagged = [w.message for w in caught if issubclass(w.category, PointingWarning)]
    # `flagged` lists the high-elevation azimuth-rate / acceleration warnings.
    # Observe at a lower elevation or reduce the scan velocity to clear them.

**Convert trajectory for OCS /path endpoint**::

    from fyst_trajectories.trajectory_utils import to_path_format

    # List of [time, az, el, az_vel, el_vel]
    points = to_path_format(trajectory)

**Print formatted summary**::

    from fyst_trajectories.trajectory_utils import print_trajectory

    print_trajectory(trajectory)  # Shows first 5 and last 5 points

Instrument Offsets
------------------

When using off-axis detectors, use ``.for_detector()`` to adjust trajectories
so the detector (not the boresight) tracks the target::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import PongScanConfig, TrajectoryBuilder
    from fyst_trajectories.primecam import get_primecam_offset

    site = get_fyst_site()
    start_time = Time("2026-03-15T02:00:00", scale="utc")

    # Use a PrimeCam module offset
    offset = get_primecam_offset("i1")

    # Boresight adjusted so detector tracks the target
    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=180.0, dec=-30.0)
        .with_config(PongScanConfig(
            timestep=0.1, width=1.0, height=1.0, spacing=0.1,
            velocity=0.2, num_terms=4, angle=0.0,
        ))
        .for_detector(offset)
        .duration(60.0)
        .starting_at(start_time)
        .build()
    )

**Custom detector offset (from angular values)**::

    from fyst_trajectories import InstrumentOffset

    offset = InstrumentOffset(dx=30.0, dy=15.0, name="CustomDetector")

**Custom detector offset (from focal plane coordinates)**::

    from fyst_trajectories import InstrumentOffset, get_fyst_site

    site = get_fyst_site()

    # Convert physical mm position to angular offset
    offset = InstrumentOffset.from_focal_plane(
        x_mm=100.0, y_mm=200.0,
        plate_scale=site.plate_scale,  # 13.89 arcsec/mm
        name="CustomDetector",
    )

See :doc:`api/offsets` for full details on instrument offset handling.

For comprehensive examples of each pattern, see :doc:`trajectory_examples`.
