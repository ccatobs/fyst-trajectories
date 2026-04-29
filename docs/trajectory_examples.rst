Trajectory Generation Examples
==============================

Examples for generating telescope trajectories using the patterns package.
Trajectories are compatible with the ACU ProgramTrack mode and OCS ``/path`` endpoint.

Setup
-----

::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import TrajectoryBuilder

    site = get_fyst_site()
    start_time = Time("2026-03-15T04:00:00", scale="utc")

The ``Trajectory`` Object
-------------------------

Pattern generation returns a ``Trajectory`` containing:

- ``times`` - Seconds from start (numpy array)
- ``az``, ``el`` - Positions in degrees (numpy arrays)
- ``az_vel``, ``el_vel`` - Velocities in deg/s (numpy arrays)
- ``start_time`` - Absolute start (astropy Time)
- ``scan_flag`` - Per-sample flags: 0=unclassified, 1=science, 2=turnaround, 3=retune
- ``science_mask`` - Boolean property: True for science-quality samples
- ``pattern_type``, ``pattern_params`` - Metadata (from ``TrajectoryMetadata``)
- ``duration``, ``n_points`` - Computed properties

**Export for OCS**::

    from fyst_trajectories.trajectory_utils import to_path_format

    # List of [time, az, el, az_vel, el_vel]
    points = to_path_format(trajectory)

    payload = {
        "start_time": trajectory.start_time.unix,
        "coordsys": "Horizon",
        "points": points,
    }

**Print formatted summary**::

    from fyst_trajectories.trajectory_utils import print_trajectory

    print_trajectory(trajectory)              # First 5 and last 5 points
    print_trajectory(trajectory, head=10)     # First 10 and last 5 points
    print_trajectory(trajectory, tail=None)   # Only first 5 points

Sidereal Track
--------------

Track a fixed RA/Dec position::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import SiderealTrackConfig, TrajectoryBuilder

    site = get_fyst_site()
    start_time = Time("2026-01-15T02:00:00", scale="utc")

    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=83.633, dec=22.014)  # Crab Nebula
        .with_config(SiderealTrackConfig(timestep=0.1))
        .duration(300.0)
        .starting_at(start_time)
        .build()
    )

Planet Track
------------

Track solar system bodies using astropy ephemeris::

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

Supported bodies: mercury, venus, mars, jupiter, saturn, uranus, neptune, moon, sun.

Constant Elevation Scan
-----------------------

For field-based observations, :func:`~fyst_trajectories.planning.plan_constant_el_scan`
is the recommended approach. It auto-computes the azimuth range, observation
duration, and number of scans from the field geometry::

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import FieldRegion, plan_constant_el_scan

    site = get_fyst_site()

    field = FieldRegion(ra_center=0.0, dec_center=-2.0, width=60.0, height=14.0)
    block = plan_constant_el_scan(
        field=field,
        elevation=50.0,         # Fixed elevation (deg)
        velocity=0.5,           # Az scan speed (deg/s)
        site=site,
        start_time="2026-09-15T00:00:00",
        rising=True,            # Use rising crossing
    )
    trajectory = block.trajectory

For manual control (engineering tests, known azimuth ranges),
``ConstantElScanConfig`` + ``TrajectoryBuilder`` can be used directly::

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import ConstantElScanConfig, TrajectoryBuilder

    site = get_fyst_site()

    config = ConstantElScanConfig(
        timestep=0.1,       # Time between points (s)
        az_start=120.0,     # Starting azimuth (deg)
        az_stop=180.0,      # Ending azimuth (deg)
        elevation=45.0,     # Fixed elevation (deg)
        az_speed=1.0,       # Scan speed (deg/s)
        az_accel=0.5,       # Acceleration (deg/s^2)
        n_scans=2,          # Back-and-forth count
    )

    trajectory = (
        TrajectoryBuilder(site)
        .with_config(config)
        .duration(120.0)
        .build()
    )

Pong Scan
---------

::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import PongScanConfig, TrajectoryBuilder

    site = get_fyst_site()
    start_time = Time("2026-03-15T04:00:00", scale="utc")

    config = PongScanConfig(
        timestep=0.1,       # Time between points (s)
        width=2.0,          # Width (deg)
        height=2.0,         # Height (deg)
        spacing=0.1,        # Space between scan lines (deg)
        velocity=0.5,       # Total scan velocity (deg/s)
        num_terms=4,        # Fourier terms for smoothing
        angle=0.0,          # Rotation angle (deg)
    )

    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=180.0, dec=-30.0)
        .with_config(config)
        .duration(600.0)
        .starting_at(start_time)
        .build()
    )

Daisy Scan
----------

::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import DaisyScanConfig, TrajectoryBuilder

    site = get_fyst_site()
    start_time = Time("2026-01-15T02:00:00", scale="utc")

    config = DaisyScanConfig(
        timestep=0.1,           # Time between points (s)
        radius=0.5,             # Characteristic radius R0 (deg)
        velocity=0.3,           # Scan velocity (deg/s)
        turn_radius=0.2,        # Radius of curvature for turns (deg)
        avoidance_radius=0.0,   # Avoid center radius (0 = pass through)
        start_acceleration=0.5, # Ramp-up acceleration (deg/s^2)
        y_offset=0.0,           # Initial y offset (deg)
    )

    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=83.82, dec=-5.39)  # Orion Nebula
        .with_config(config)
        .duration(300.0)
        .starting_at(start_time)
        .build()
    )

Linear Motion
-------------

Constant velocity motion in Az/El::

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import LinearMotionConfig, TrajectoryBuilder

    site = get_fyst_site()

    config = LinearMotionConfig(
        timestep=0.1,           # Time between points (s)
        az_start=100.0,         # Starting azimuth (deg)
        el_start=45.0,          # Starting elevation (deg)
        az_velocity=0.5,        # Az velocity (deg/s)
        el_velocity=0.1,        # El velocity (deg/s)
    )

    trajectory = (
        TrajectoryBuilder(site)
        .with_config(config)
        .duration(60.0)
        .build()
    )

ACU Upload
----------

In production the PCS ACU agent receives scan parameters from the OCS
scheduler, calls fyst-trajectories to build a trajectory, and uploads it
via ``aculib``::

    OCS Scheduler --[scan config]--> agent.execute_scan()
                                        |
                                        v
                                    fyst-trajectories  (trajectory planning)
                                        |
                                        v
                                    aculib.scan_pattern()  (HTTP client)
                                        |
                                        v
                                    TCS (Go server) --> ACU Hardware

fyst-trajectories is a library dependency of the PCS agent, not an OCS
agent itself. Inside an agent task, a typical scan looks like::

    # Inside ACUAgent.execute_pong_scan() -- an OCS task
    from astropy.time import Time

    from fyst_trajectories import get_fyst_site, resolve_offset
    from fyst_trajectories.planning import FieldRegion, plan_pong_scan
    from fyst_trajectories.trajectory_utils import to_path_format

    site = get_fyst_site()

    field = FieldRegion(
        ra_center=params["ra_center"],
        dec_center=params["dec_center"],
        width=params["width"],
        height=params["height"],
    )
    offset = resolve_offset(module=params.get("detector"))

    block = plan_pong_scan(
        field=field,
        velocity=params.get("velocity", 0.5),
        spacing=params.get("spacing", 0.1),
        num_terms=4,
        site=site,
        start_time=Time.now(),
        detector_offset=offset,
    )

    data = {
        "start_time": float(block.trajectory.start_time.unix),
        "coordsys": "Horizon",
        "points": to_path_format(block.trajectory),
    }
    tcs = self._get_tcs_client()
    tcs.scan_pattern(data)

For local testing without the PCS agent, POST the same payload directly::

    import requests

    from fyst_trajectories.trajectory_utils import to_path_format

    payload = {
        "start_time": trajectory.start_time.unix,
        "coordsys": "Horizon",
        "points": to_path_format(trajectory),
    }
    response = requests.post("http://localhost:8000/path", json=payload)

The planning and simulation pipelines import the same library, so
trajectories used for coverage analysis match what the telescope executes
at runtime.

.. tip::

   For field-based constant-elevation observations, use
   :func:`~fyst_trajectories.planning.plan_constant_el_scan` instead of manually
   constructing ``ConstantElScanConfig``. It auto-computes the azimuth range,
   duration, and number of scans from a ``FieldRegion``. See :doc:`planning`.

Drift Scan (Planet Calibration)
-------------------------------

A constant elevation scan where a planet drifts through the field of view
due to Earth's rotation.

**Simple constant-el scan centered on planet position**::

    from astropy.time import Time

    from fyst_trajectories import Coordinates, get_fyst_site
    from fyst_trajectories.patterns import ConstantElScanConfig, TrajectoryBuilder

    site = get_fyst_site()
    coords = Coordinates(site)
    observation_time = Time("2026-03-15T00:00:00", scale="utc")

    # Get Jupiter's position at observation time
    jupiter_az, jupiter_el = coords.get_body_altaz("jupiter", observation_time)

    # Simple constant-el scan centered on planet position
    config = ConstantElScanConfig(
        timestep=0.1,
        az_start=jupiter_az - 5.0,  # scan +/-5 deg around Jupiter
        az_stop=jupiter_az + 5.0,
        elevation=jupiter_el,
        az_speed=0.5,
        az_accel=0.3,
        n_scans=4,
    )

    trajectory = (
        TrajectoryBuilder(site)
        .with_config(config)
        .duration(600.0)
        .starting_at(observation_time)
        .build()
    )

**Have planet drift through a specific detector (e.g., I1)**::

    from astropy.time import Time

    from fyst_trajectories import Coordinates, get_fyst_site
    from fyst_trajectories.offsets import compute_focal_plane_rotation, detector_to_boresight
    from fyst_trajectories.patterns import ConstantElScanConfig, TrajectoryBuilder
    from fyst_trajectories.primecam import get_primecam_offset

    site = get_fyst_site()
    coords = Coordinates(site)
    observation_time = Time("2026-03-15T00:00:00", scale="utc")

    # Get planet position using ephemeris
    planet_az, planet_el = coords.get_body_altaz("jupiter", observation_time)
    planet_ra, planet_dec = coords.get_body_radec("jupiter", observation_time)

    # Compute focal plane rotation (mechanical only, no parallactic angle for planets)
    offset = get_primecam_offset("i1")
    parallactic_angle = coords.get_parallactic_angle(planet_ra, planet_dec, observation_time)
    field_rotation = compute_focal_plane_rotation(
        el=planet_el, site=site, offset=offset, parallactic_angle=parallactic_angle
    )

    # Compute boresight position so detector I1 sees the planet
    bore_az, bore_el = detector_to_boresight(
        det_az=planet_az, det_el=planet_el,
        offset=offset,
        field_rotation=field_rotation,
    )

    # Set up scan centered on boresight position
    config = ConstantElScanConfig(
        timestep=0.1,
        az_start=bore_az - 5.0,
        az_stop=bore_az + 5.0,
        elevation=bore_el,
        az_speed=0.5,
        az_accel=0.3,
        n_scans=4,
    )

    trajectory = (
        TrajectoryBuilder(site)
        .with_config(config)
        .duration(600.0)
        .starting_at(observation_time)
        .build()
    )

Advanced: Pattern Discovery
---------------------------

For interactive exploration or dynamic scenarios where the pattern name is
determined at runtime, you can use the registry functions::

    from astropy.time import Time

    from fyst_trajectories import list_patterns, get_pattern, get_fyst_site

    # List available pattern names
    print(list_patterns())
    # ['constant_el', 'daisy', 'linear', 'planet', 'pong', 'sidereal']

    # Get a pattern class by name (useful for plugins or config-driven selection)
    pattern_name = "pong"  # e.g., from user input or config file
    PatternClass = get_pattern(pattern_name)

    # Instantiate and generate
    from fyst_trajectories.patterns import PongScanConfig

    site = get_fyst_site()
    start_time = Time("2026-03-15T04:00:00", scale="utc")
    config = PongScanConfig(
        timestep=0.1, width=2.0, height=2.0, spacing=0.1,
        velocity=0.5, num_terms=4, angle=0.0,
    )
    pattern = PatternClass(ra=180.0, dec=-30.0, config=config)
    trajectory = pattern.generate(site, duration=300.0, start_time=start_time)
