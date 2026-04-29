"""Test all code examples from RST documentation files.

This test suite verifies that every code example in the documentation
actually runs without errors. Each test is named after the doc section
it validates.
"""

import pytest
from astropy.time import Time

from fyst_trajectories import (
    Coordinates,
    InstrumentOffset,
    get_fyst_site,
    normalize_frame,
    print_trajectory,
    to_path_format,
    validate_trajectory,
)
from fyst_trajectories.exceptions import (
    ElevationBoundsError,
    PointingWarning,
    TargetNotObservableError,
)
from fyst_trajectories.offsets import (
    apply_detector_offset,
    boresight_to_detector,
    compute_focal_plane_rotation,
    detector_to_boresight,
)
from fyst_trajectories.patterns import (
    ConstantElScanConfig,
    DaisyScanConfig,
    PlanetTrackConfig,
    PongScanConfig,
    SiderealTrackConfig,
    TrajectoryBuilder,
)
from fyst_trajectories.primecam import (
    PRIMECAM_I1,
    PRIMECAM_MODULES,
    get_primecam_offset,
)

# NOTE: Many test functions below contain function-level imports that duplicate
# the module-level imports above.  This is intentional -- each test mirrors a
# code snippet from the RST documentation, so the imports inside the function
# must match what the docs show the user.  Do not hoist them to module level.

# ============================================================================
# quickstart.rst examples
# ============================================================================


def test_quickstart_get_site():
    """Test basic site retrieval from quickstart.rst."""
    site = get_fyst_site()
    print(f"FYST is at {site.latitude}, {site.longitude}")
    assert site is not None


def test_quickstart_radec_to_altaz():
    """Test RA/Dec to Az/El conversion from quickstart.rst."""
    from fyst_trajectories import get_fyst_site

    site = get_fyst_site()
    coords = Coordinates(site)

    # Orion Nebula
    obstime = Time("2026-01-15T02:00:00", scale="utc")
    az, el = coords.radec_to_altaz(ra=83.82, dec=-5.39, obstime=obstime)
    print(f"Orion is at Az={az:.1f}, El={el:.1f}")
    assert isinstance(az, float)
    assert isinstance(el, float)


def test_quickstart_frame_translation():
    """Test coordinate frame name translation from quickstart.rst."""
    # Translate common frame names to astropy equivalents
    normalize_frame("J2000")  # Returns "icrs"
    normalize_frame("GALACTIC")  # Returns "galactic"

    assert normalize_frame("J2000") == "icrs"
    assert normalize_frame("GALACTIC") == "galactic"


def test_quickstart_proper_motion():
    """Test proper motion support from quickstart.rst."""
    from fyst_trajectories import get_fyst_site

    coords = Coordinates(get_fyst_site())

    # Barnard's Star with proper motion
    az, el = coords.radec_to_altaz_with_pm(
        ra=269.452,
        dec=4.693,
        pm_ra=-798.58,
        pm_dec=10328.12,  # mas/yr
        ref_epoch=Time("J2015.5"),
        obstime=Time("2026-06-15T04:00:00"),
        distance=1.8,  # parsecs
    )
    assert isinstance(az, float)
    assert isinstance(el, float)


def test_quickstart_sidereal_tracking():
    """Test sidereal tracking example from quickstart.rst."""
    from fyst_trajectories import get_fyst_site

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
    assert trajectory.n_points > 0
    assert abs(trajectory.duration - 300.0) < 1.0


def test_quickstart_planet_tracking():
    """Test planet tracking example from quickstart.rst."""
    from fyst_trajectories import get_fyst_site

    site = get_fyst_site()
    start_time = Time("2026-03-15T16:00:00", scale="utc")

    trajectory = (
        TrajectoryBuilder(site)
        .with_config(PlanetTrackConfig(timestep=0.1, body="mars"))
        .duration(600.0)
        .starting_at(start_time)
        .build()
    )
    assert trajectory.n_points > 0
    assert abs(trajectory.duration - 600.0) < 1.0


def test_quickstart_constant_el_scan():
    """Test constant elevation scan from quickstart.rst."""
    site = get_fyst_site()

    config = ConstantElScanConfig(
        timestep=0.1,
        az_start=120.0,
        az_stop=180.0,
        elevation=45.0,
        az_speed=1.0,
        az_accel=0.5,
        n_scans=2,
    )

    trajectory = TrajectoryBuilder(site).with_config(config).duration(300.0).build()
    assert trajectory.n_points > 0


def test_quickstart_pong_scan():
    """Test pong scan example from quickstart.rst."""
    from fyst_trajectories import get_fyst_site

    site = get_fyst_site()
    start_time = Time("2026-03-15T04:00:00", scale="utc")

    config = PongScanConfig(
        timestep=0.1,
        width=2.0,
        height=2.0,
        spacing=0.1,
        velocity=0.5,
        num_terms=4,
        angle=0.0,
    )

    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=180.0, dec=-30.0)
        .with_config(config)
        .duration(300.0)
        .starting_at(start_time)
        .build()
    )
    assert trajectory.n_points > 0


def test_quickstart_daisy_scan():
    """Test daisy scan example from quickstart.rst."""
    from fyst_trajectories import get_fyst_site

    site = get_fyst_site()
    start_time = Time("2026-03-15T04:00:00", scale="utc")

    config = DaisyScanConfig(
        timestep=0.1,
        radius=0.5,
        velocity=0.3,
        turn_radius=0.2,
        avoidance_radius=0.0,
        start_acceleration=0.5,
        y_offset=0.0,
    )

    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=180.0, dec=-30.0)
        .with_config(config)
        .duration(300.0)
        .starting_at(start_time)
        .build()
    )
    assert trajectory.n_points > 0


def test_quickstart_to_path_format():
    """Test trajectory to path format conversion from quickstart.rst."""
    from fyst_trajectories import get_fyst_site

    site = get_fyst_site()
    start_time = Time("2026-01-15T02:00:00", scale="utc")

    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=83.633, dec=22.014)
        .with_config(SiderealTrackConfig(timestep=0.1))
        .duration(10.0)
        .starting_at(start_time)
        .build()
    )

    points = to_path_format(trajectory)  # List of [time, az, el, az_vel, el_vel]

    assert isinstance(points, list)
    assert len(points) > 0
    assert len(points[0]) == 5  # [time, az, el, az_vel, el_vel]


def test_quickstart_print_trajectory():
    """Test print_trajectory from quickstart.rst."""
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import SiderealTrackConfig, TrajectoryBuilder

    site = get_fyst_site()
    start_time = Time("2026-01-15T02:00:00", scale="utc")

    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=83.633, dec=22.014)
        .with_config(SiderealTrackConfig(timestep=0.1))
        .duration(10.0)
        .starting_at(start_time)
        .build()
    )

    print_trajectory(trajectory)  # Shows first 5 and last 5 points
    # Just verify it doesn't crash


def test_quickstart_detector_offset():
    """Test instrument offset example from quickstart.rst."""
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.primecam import get_primecam_offset

    site = get_fyst_site()
    start_time = Time("2026-03-15T04:00:00", scale="utc")

    # Use a PrimeCam module offset
    offset = get_primecam_offset("i1")

    # Boresight adjusted so detector tracks the target
    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=180.0, dec=-30.0)
        .with_config(
            PongScanConfig(
                timestep=0.1,
                width=1.0,
                height=1.0,
                spacing=0.1,
                velocity=0.5,
                num_terms=4,
                angle=0.0,
            )
        )
        .for_detector(offset)
        .duration(60.0)
        .starting_at(start_time)
        .build()
    )
    assert trajectory.n_points > 0


def test_quickstart_custom_offset_angular():
    """Test custom detector offset from angular values from quickstart.rst."""
    offset = InstrumentOffset(dx=30.0, dy=15.0, name="CustomDetector")
    assert offset.dx == 30.0
    assert offset.dy == 15.0


def test_quickstart_custom_offset_focal_plane():
    """Test custom detector offset from focal plane coords from quickstart.rst."""
    site = get_fyst_site()

    # Convert physical mm position to angular offset
    offset = InstrumentOffset.from_focal_plane(
        x_mm=100.0,
        y_mm=200.0,
        plate_scale=site.plate_scale,
        name="CustomDetector",
    )
    assert offset.name == "CustomDetector"
    assert isinstance(offset.dx, float)
    assert isinstance(offset.dy, float)


# ============================================================================
# trajectory_examples.rst examples
# ============================================================================


def test_pipeline_stage3_trajectory_generation():
    """Test Stage 3 trajectory generation from trajectory_examples.rst."""
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import PongScanConfig, TrajectoryBuilder
    from fyst_trajectories.primecam import get_primecam_offset

    # Simulate scheduled block (from Stage 2)
    # Note: Using 22:00 UTC instead of 04:00 UTC because Crab Nebula is observable then
    scheduled_block = {
        "target_ra": 83.633,
        "target_dec": 22.014,
        "start_time": "2026-03-15T22:00:00",
        "duration": 200.0,
        "config": {
            "width": 1.0,
            "height": 1.0,
            "spacing": 0.05,
            "velocity": 0.5,
            "num_terms": 4,
            "angle": 0.0,
            "timestep": 0.1,
        },
    }

    # Initialize site
    site = get_fyst_site()

    # Get the I1 module offset (280 GHz, inner ring)
    offset = get_primecam_offset("i1")

    # Build the trajectory from scheduled observation parameters
    start_time = Time(scheduled_block["start_time"], scale="utc")
    config = PongScanConfig(**scheduled_block["config"])

    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=scheduled_block["target_ra"], dec=scheduled_block["target_dec"])
        .with_config(config)
        .for_detector(offset)
        .duration(scheduled_block["duration"])
        .starting_at(start_time)
        .build()
    )

    # Validate against telescope limits
    # This scan configuration exceeds acceleration limits, which is expected behavior
    with pytest.warns(PointingWarning):
        validate_trajectory(trajectory, site)

    # Inspect the result
    print_trajectory(trajectory)
    print(f"Points: {trajectory.n_points}")
    print(f"Duration: {trajectory.duration:.1f}s")
    print(f"Az range: [{trajectory.az.min():.2f}, {trajectory.az.max():.2f}] deg")
    print(f"El range: [{trajectory.el.min():.2f}, {trajectory.el.max():.2f}] deg")

    assert trajectory.n_points > 0


def test_pipeline_stage4_to_path_format():
    """Test Stage 4 to_path_format from trajectory_examples.rst."""
    from fyst_trajectories import get_fyst_site

    site = get_fyst_site()
    # Use a time when the Crab Nebula (RA=83.6, Dec=+22) is above the horizon
    start_time = Time("2026-01-15T06:00:00", scale="utc")

    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=83.633, dec=22.014)
        .with_config(SiderealTrackConfig(timestep=0.1))
        .duration(10.0)
        .starting_at(start_time)
        .build()
    )

    # Convert trajectory to the /path endpoint format.
    # Each point is [time_offset, az, el, az_vel, el_vel].
    points = to_path_format(trajectory)

    # Verify format
    assert isinstance(points, list)
    assert len(points) > 0
    assert len(points[0]) == 5
    assert isinstance(points[0][0], float)  # time_offset
    assert isinstance(points[0][1], float)  # az
    assert isinstance(points[0][2], float)  # el
    assert isinstance(points[0][3], float)  # az_vel
    assert isinstance(points[0][4], float)  # el_vel

    # Don't actually make HTTP request, just verify format
    # response = requests.post(...)


def test_pipeline_error_target_not_observable():
    """Test target not observable error from trajectory_examples.rst."""
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import PongScanConfig, TrajectoryBuilder

    site = get_fyst_site()
    start_time = Time("2026-03-15T04:00:00", scale="utc")
    config = PongScanConfig(
        timestep=0.1,
        width=2.0,
        height=2.0,
        spacing=0.1,
        velocity=0.5,
        num_terms=4,
        angle=0.0,
    )

    with pytest.raises(TargetNotObservableError) as exc_info:
        (
            TrajectoryBuilder(site)
            .at(ra=180.0, dec=80.0)  # Dec +80 deg never visible from FYST
            .with_config(config)
            .duration(200.0)
            .starting_at(start_time)
            .build()
        )

    exc = exc_info.value
    assert "180.000" in exc.target
    assert "80.000" in exc.target


def test_pipeline_error_elevation_bounds():
    """Test elevation bounds error from trajectory_examples.rst."""
    from fyst_trajectories.patterns import ConstantElScanConfig, TrajectoryBuilder

    site = get_fyst_site()

    with pytest.raises(ElevationBoundsError) as exc_info:
        (
            TrajectoryBuilder(site)
            .with_config(
                ConstantElScanConfig(
                    timestep=0.1,
                    az_start=120.0,
                    az_stop=180.0,
                    elevation=15.0,  # Below minimum of 20 deg
                    az_speed=1.0,
                    az_accel=0.5,
                    n_scans=4,
                )
            )
            .duration(120.0)
            .build()
        )

    exc = exc_info.value
    assert exc.actual_min == 15.0
    assert exc.limit_min == 20.0


# ============================================================================
# instrument_offsets.rst examples
# ============================================================================


def test_offsets_quick_example():
    """Test quick example from instrument_offsets.rst."""
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.primecam import get_primecam_offset

    site = get_fyst_site()
    start_time = Time("2026-03-15T04:00:00", scale="utc")

    # Use predefined PrimeCam offset
    offset = get_primecam_offset("i1")

    # Boresight adjusted so detector I1 tracks the target
    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=180.0, dec=-30.0)
        .with_config(
            PongScanConfig(
                timestep=0.1,
                width=1.0,
                height=1.0,
                spacing=0.1,
                velocity=0.5,
                num_terms=4,
                angle=0.0,
            )
        )
        .for_detector(offset)
        .duration(60.0)
        .starting_at(start_time)
        .build()
    )
    assert trajectory.n_points > 0


def test_offsets_compute_focal_plane_rotation():
    """Test compute_focal_plane_rotation from instrument_offsets.rst."""
    site = get_fyst_site()
    offset = InstrumentOffset(dx=5.0, dy=3.0, instrument_rotation=10.0)

    rotation = compute_focal_plane_rotation(
        el=45.0, site=site, offset=offset, parallactic_angle=20.0
    )
    # rotation = +1 * 45.0 + 10.0 + 20.0 = 75.0
    assert abs(rotation - 75.0) < 0.01


def test_offsets_boresight_to_detector():
    """Test boresight_to_detector from instrument_offsets.rst."""
    offset = InstrumentOffset(dx=5.0, dy=3.0)  # arcmin

    det_az, det_el = boresight_to_detector(
        az=180.0,
        el=45.0,
        offset=offset,
        field_rotation=30.0,  # degrees
    )
    assert isinstance(det_az, float)
    assert isinstance(det_el, float)


def test_offsets_detector_to_boresight():
    """Test detector_to_boresight from instrument_offsets.rst."""
    offset = InstrumentOffset(dx=5.0, dy=3.0)

    det_az, det_el = boresight_to_detector(az=180.0, el=45.0, offset=offset, field_rotation=30.0)

    bore_az, bore_el = detector_to_boresight(
        det_az=det_az, det_el=det_el, offset=offset, field_rotation=30.0
    )

    # Should get back original boresight position
    assert abs(bore_az - 180.0) < 0.001
    assert abs(bore_el - 45.0) < 0.001


def test_offsets_apply_detector_offset():
    """Test apply_detector_offset from instrument_offsets.rst."""
    from fyst_trajectories import InstrumentOffset, get_fyst_site
    from fyst_trajectories.patterns import PongScanConfig, TrajectoryBuilder

    site = get_fyst_site()
    start_time = Time("2026-03-15T04:00:00", scale="utc")

    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=180.0, dec=-30.0)
        .with_config(
            PongScanConfig(
                timestep=0.1,
                width=1.0,
                height=1.0,
                spacing=0.1,
                velocity=0.5,
                num_terms=4,
                angle=0.0,
            )
        )
        .duration(60.0)
        .starting_at(start_time)
        .build()
    )

    offset = InstrumentOffset(dx=30.0, dy=0.0)
    adjusted = apply_detector_offset(trajectory, offset, site)

    assert adjusted.n_points == trajectory.n_points


def test_offsets_primecam_access():
    """Test PrimeCam offset access from instrument_offsets.rst."""
    offset = get_primecam_offset("i1")  # Case-insensitive
    offset_direct = PRIMECAM_I1  # Direct access

    assert offset.dx == offset_direct.dx
    assert offset.dy == offset_direct.dy

    for name, offset in PRIMECAM_MODULES.items():
        print(f"{name}: dx={offset.dx:.1f}', dy={offset.dy:.1f}'")


def test_offsets_custom_angular():
    """Test custom offset from angular values from instrument_offsets.rst."""
    offset = InstrumentOffset(dx=10.0, dy=5.0, name="MyDetector")

    # Values are in arcminutes; properties provide degrees
    print(f"{offset.dx_deg:.4f} x {offset.dy_deg:.4f} degrees")

    # With instrument rotation (e.g., dewar rotated 15 degrees)
    offset = InstrumentOffset(dx=10.0, dy=5.0, name="RotatedDetector", instrument_rotation=15.0)
    assert offset.instrument_rotation == 15.0


def test_offsets_custom_focal_plane():
    """Test custom offset from focal plane coords from instrument_offsets.rst."""
    site = get_fyst_site()

    # Convert physical position to angular offset using plate scale
    offset = InstrumentOffset.from_focal_plane(
        x_mm=230.65,  # Cross-elevation position (mm)
        y_mm=399.5,  # Elevation position (mm)
        plate_scale=site.plate_scale,  # 13.89 arcsec/mm
        name="Module-A2",
    )
    print(f"Angular offset: {offset.dx:.1f}' x {offset.dy:.1f}'")

    # With instrument rotation (e.g., dewar at 15 degree angle)
    offset = InstrumentOffset.from_focal_plane(
        x_mm=100.0,
        y_mm=200.0,
        plate_scale=site.plate_scale,
        name="RotatedModule",
        instrument_rotation=15.0,
    )
    assert offset.instrument_rotation == 15.0


def test_offsets_large_offset_calculation():
    """Test large offset calculation from instrument_offsets.rst."""
    # Works for any offset size (small or large)
    offset = InstrumentOffset(dx=180.0, dy=90.0, name="OuterRing")  # 3 degrees

    det_az, det_el = boresight_to_detector(
        az=180.0,
        el=45.0,
        offset=offset,
        field_rotation=30.0,
    )
    assert isinstance(det_az, float)
    assert isinstance(det_el, float)


# ============================================================================
# coordinate_systems.rst examples
# ============================================================================


def test_coordsys_frame_aliases():
    """Test frame alias usage from coordinate_systems.rst."""
    # Case-insensitive lookup
    astropy_frame = normalize_frame("J2000")  # Returns "icrs"
    astropy_frame = normalize_frame("galactic")  # Returns "galactic"

    assert normalize_frame("J2000") == "icrs"
    assert normalize_frame("galactic") == "galactic"

    # Unknown frames are lowercased for astropy compatibility
    astropy_frame = normalize_frame("MyFrame")  # Returns "myframe"
    assert astropy_frame == "myframe"


def test_coordsys_trajectory_metadata():
    """Test trajectory coordinate fields from coordinate_systems.rst."""
    from fyst_trajectories import get_fyst_site

    # Use a specific time when target is observable
    start_time = Time("2026-03-15T04:00:00", scale="utc")

    trajectory = (
        TrajectoryBuilder(get_fyst_site())
        .at(ra=180.0, dec=-30.0)  # Input in ICRS
        .with_config(
            PongScanConfig(
                timestep=0.1,
                width=2.0,
                height=2.0,
                spacing=0.1,
                velocity=0.5,
                num_terms=4,
                angle=0.0,
            )
        )
        .duration(300.0)
        .starting_at(start_time)
        .build()
    )

    print(trajectory.coordsys)  # "altaz"
    print(trajectory.metadata.input_frame)  # "icrs"

    assert trajectory.coordsys == "altaz"
    assert trajectory.metadata.input_frame == "icrs"


def test_coordsys_proper_motion():
    """Test proper motion from coordinate_systems.rst."""
    from fyst_trajectories import get_fyst_site

    coords = Coordinates(get_fyst_site())

    # Barnard's Star (moves ~10 arcsec/year)
    az, el = coords.radec_to_altaz_with_pm(
        ra=269.452,
        dec=4.693,
        pm_ra=-798.58,
        pm_dec=10328.12,  # mas/yr (pm_ra includes cos(dec))
        ref_epoch=Time("J2015.5"),
        obstime=Time("2026-06-15T04:00:00", scale="utc"),
        distance=1.8,  # parsecs, optional
    )
    assert isinstance(az, float)
    assert isinstance(el, float)


# ============================================================================
# planning.rst examples
# ============================================================================
#
# These tests exercise every code example in ``docs/planning.rst``.  They exist
# to close a coverage gap: until v0.3.0, ``planning.rst`` had zero tests, which
# allowed two broken ``plan_pong_scan`` examples (using an unobservable
# ``start_time`` for the Chandra Deep Field South) to reach release.  The
# regression test ``test_planning_plan_pong_scan_chandra_deep_field_observable``
# locks in the corrected time.


def test_planning_quickstart_pong():
    """Test Quick Start plan_pong_scan example from planning.rst."""
    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import FieldRegion, plan_pong_scan

    site = get_fyst_site()

    field = FieldRegion(ra_center=180.0, dec_center=-30.0, width=2.0, height=2.0)
    block = plan_pong_scan(
        field=field,
        velocity=0.5,  # deg/s
        spacing=0.1,  # deg between scan lines
        num_terms=4,  # Fourier terms for smooth turnarounds
        site=site,
        start_time=Time("2026-03-15T04:00:00", scale="utc"),
        timestep=0.1,
    )

    print(block.summary)
    print(f"Duration: {block.duration:.1f}s ({block.duration / 3600:.1f}h)")
    print(f"Trajectory: {block.trajectory.n_points} points")

    assert block.trajectory.n_points > 0
    assert block.trajectory.scan_flag is not None
    assert len(block.trajectory.scan_flag) == block.trajectory.n_points
    # Trajectory should stay within FYST telescope limits.
    el_limits = site.telescope_limits.elevation
    assert block.trajectory.el.min() >= el_limits.min
    assert block.trajectory.el.max() <= el_limits.max


def test_planning_quickstart_constant_el():
    """Test Quick Start plan_constant_el_scan example from planning.rst."""
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import FieldRegion, plan_constant_el_scan

    site = get_fyst_site()

    field = FieldRegion(ra_center=0.0, dec_center=-2.0, width=60.0, height=14.0)
    with pytest.warns(PointingWarning):
        # Scan width of 60 deg is above the advisory MAX_REASONABLE_SCAN_WIDTH_DEG.
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

    assert block.trajectory.n_points > 0
    assert block.duration > 0


def test_planning_field_region_cmb():
    """Test FieldRegion construction example from planning.rst."""
    from fyst_trajectories.planning import FieldRegion

    # Stripe 82 CMB field: 60 deg RA x 14 deg Dec
    cmb_field = FieldRegion(
        ra_center=0.0,  # deg (0h RA)
        dec_center=-2.0,  # deg
        width=60.0,  # RA extent in degrees
        height=14.0,  # Dec extent in degrees
    )

    # Dec boundaries are computed automatically
    print(f"Dec range: [{cmb_field.dec_min}, {cmb_field.dec_max}]")
    assert cmb_field.dec_min == pytest.approx(-9.0)
    assert cmb_field.dec_max == pytest.approx(5.0)


def test_planning_plan_pong_scan_basic_chandra_deep_field():
    """Test basic plan_pong_scan example from planning.rst (Chandra Deep Field).

    This is the regression-critical test: the previous docs used
    ``start_time="2026-03-15T04:00:00"`` which left the Chandra Deep Field
    South below the horizon, raising ``TargetNotObservableError``.  The
    corrected example uses ``"2026-03-15T22:12:00"``.
    """
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
        angle=170.0,  # rotation angle (degrees)
    )

    assert block.trajectory.n_points > 0
    assert block.trajectory.scan_flag is not None
    assert len(block.trajectory.scan_flag) == block.trajectory.n_points
    el_limits = site.telescope_limits.elevation
    assert block.trajectory.el.min() >= el_limits.min
    assert block.trajectory.el.max() <= el_limits.max


def test_planning_plan_pong_scan_multiple_cycles():
    """Test multi-cycle plan_pong_scan example from planning.rst."""
    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import FieldRegion, plan_pong_scan

    site = get_fyst_site()
    field = FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7)

    block = plan_pong_scan(
        field=field,
        velocity=0.5,
        spacing=0.1,
        num_terms=4,
        site=site,
        start_time=Time("2026-03-15T22:12:00", scale="utc"),
        timestep=0.1,
        n_cycles=3,  # observe 3 full Pong periods
    )

    assert block.trajectory.n_points > 0
    assert block.computed_params["n_cycles"] == 3
    # Duration should equal 3 periods.
    assert block.duration == pytest.approx(block.computed_params["period"] * 3)


def test_planning_plan_pong_scan_with_detector_offset():
    """Test detector-offset plan_pong_scan example from planning.rst."""
    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import FieldRegion, plan_pong_scan
    from fyst_trajectories.primecam import get_primecam_offset

    site = get_fyst_site()
    field = FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7)

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

    assert block.trajectory.n_points > 0
    el_limits = site.telescope_limits.elevation
    assert block.trajectory.el.min() >= el_limits.min
    assert block.trajectory.el.max() <= el_limits.max


def test_planning_plan_pong_scan_stripe82_survey():
    """Test 'Stripe 82 / Deep56 survey' real-world example from planning.rst."""
    import astropy.units as u
    import numpy as np
    from astropy.time import Time, TimeDelta

    from fyst_trajectories import (
        AtmosphericConditions,
        Coordinates,
        get_fyst_site,
    )
    from fyst_trajectories.planning import FieldRegion, plan_pong_scan

    site = get_fyst_site()
    coords = Coordinates(site, atmosphere=AtmosphericConditions.for_fyst())

    # CMB field: RA 23h-3h, Dec -9 to +5
    cmb_field = FieldRegion(ra_center=0.0, dec_center=-2.0, width=60.0, height=14.0)

    # Find when field center reaches el=50 (rising side)
    search_start = Time("2026-03-15T00:00:00", scale="utc")
    dt = np.arange(0, 24 * 3600, 60)
    times = search_start + TimeDelta(dt * u.s)
    _, el = coords.radec_to_altaz(
        np.full(len(times), 0.0),
        np.full(len(times), -2.0),
        times,
    )
    crossing = np.where(np.diff((el >= 50.0).astype(int)))[0][0]
    start_cmb = times[crossing]

    with pytest.warns(PointingWarning):
        # Scan width of 60 deg is above the advisory MAX_REASONABLE_SCAN_WIDTH_DEG.
        cmb_block = plan_pong_scan(
            field=cmb_field,
            velocity=0.5,
            spacing=0.25,
            num_terms=4,
            site=site,
            start_time=start_cmb,
            timestep=0.5,
        )
    print(cmb_block.summary)

    assert cmb_block.trajectory.n_points > 0
    assert cmb_block.trajectory.scan_flag is not None
    assert len(cmb_block.trajectory.scan_flag) == cmb_block.trajectory.n_points


def test_planning_plan_constant_el_scan_basic():
    """Test basic plan_constant_el_scan example from planning.rst."""
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import FieldRegion, plan_constant_el_scan

    site = get_fyst_site()

    field = FieldRegion(ra_center=0.0, dec_center=-2.0, width=60.0, height=14.0)
    with pytest.warns(PointingWarning):
        # Scan width of 60 deg is above the advisory MAX_REASONABLE_SCAN_WIDTH_DEG.
        block = plan_constant_el_scan(
            field=field,
            elevation=50.0,  # fixed elevation in degrees
            velocity=0.5,  # az scan speed in deg/s
            site=site,
            start_time="2026-09-15T00:00:00",
            rising=True,  # use rising crossing
        )

    print(block.summary)
    print(f"Duration: {block.duration:.0f}s")
    print(
        f"Az range: [{block.computed_params['az_start']:.1f}, "
        f"{block.computed_params['az_stop']:.1f}]"
    )

    assert block.trajectory.n_points > 0
    assert "az_start" in block.computed_params
    assert "az_stop" in block.computed_params


def test_planning_plan_constant_el_scan_with_detector_offset():
    """Test plan_constant_el_scan with detector offset from planning.rst."""
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import FieldRegion, plan_constant_el_scan
    from fyst_trajectories.primecam import get_primecam_offset

    site = get_fyst_site()
    field = FieldRegion(ra_center=0.0, dec_center=-2.0, width=60.0, height=14.0)

    offset = get_primecam_offset("i1")
    with pytest.warns(PointingWarning):
        # Scan width of 60 deg is above the advisory MAX_REASONABLE_SCAN_WIDTH_DEG.
        block = plan_constant_el_scan(
            field=field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time="2026-09-15T00:00:00",
            detector_offset=offset,
        )

    assert block.trajectory.n_points > 0


def test_planning_plan_daisy_scan():
    """Test plan_daisy_scan example from planning.rst."""
    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import plan_daisy_scan

    site = get_fyst_site()

    block = plan_daisy_scan(
        ra=83.633,  # Crab Nebula RA
        dec=22.014,  # Crab Nebula Dec
        radius=0.5,  # characteristic radius R0 (degrees)
        velocity=0.3,  # scan velocity (deg/s)
        turn_radius=0.2,  # curvature radius for turns (degrees)
        avoidance_radius=0.0,  # avoid center within this radius
        start_acceleration=0.5,  # ramp-up acceleration (deg/s^2)
        site=site,
        start_time=Time("2026-01-15T02:00:00", scale="utc"),
        timestep=0.1,
        duration=300.0,  # 5 minutes
    )

    print(block.summary)

    assert block.trajectory.n_points > 0
    el_limits = site.telescope_limits.elevation
    assert block.trajectory.el.min() >= el_limits.min
    assert block.trajectory.el.max() <= el_limits.max


def test_planning_inspect_scan_block():
    """Test 'Example of inspecting a scan block' from planning.rst."""
    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.planning import FieldRegion, plan_pong_scan

    site = get_fyst_site()
    field = FieldRegion(ra_center=180.0, dec_center=-30.0, width=2.0, height=2.0)
    block = plan_pong_scan(
        field=field,
        velocity=0.5,
        spacing=0.1,
        num_terms=4,
        site=site,
        start_time=Time("2026-03-15T04:00:00", scale="utc"),
        timestep=0.1,
    )

    # Access the trajectory
    traj = block.trajectory
    print(f"Points: {traj.n_points}, Duration: {traj.duration:.0f}s")

    # Inspect computed parameters
    print(f"Pong period: {block.computed_params['period']:.0f}s")
    print(f"Vertices: {block.computed_params['x_numvert']} x {block.computed_params['y_numvert']}")

    # Print summary
    print(block.summary)

    # Validate trajectory against telescope limits
    validate_trajectory(traj, get_fyst_site())

    assert traj.n_points > 0
    assert "period" in block.computed_params
    assert "x_numvert" in block.computed_params
    assert "y_numvert" in block.computed_params


@pytest.mark.parametrize(
    "start_iso",
    [
        "2026-03-15T22:12:00",  # the corrected, observable time
    ],
)
def test_planning_plan_pong_scan_chandra_deep_field_observable(start_iso):
    """Regression test for docs/planning.rst start_time bug.

    The Chandra Deep Field South is below the horizon at FYST at
    2026-03-15T04:00:00 (the previously-broken example time).  It is
    observable at 2026-03-15T22:12:00 (the corrected time).  This
    parametrized test locks the corrected time in place -- if someone
    reverts it to an unobservable value, this test will fail loudly.
    """
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
        start_time=Time(start_iso, scale="utc"),
        timestep=0.1,
        angle=170.0,
    )
    assert block.trajectory.n_points > 0


# ============================================================================
# Source docstring regression tests
# ============================================================================
#
# These tests guard the three docstring examples that the v0.3.0 verification
# pass found broken and then fixed.  They intentionally mirror the shape of the
# fixed docstring snippets so a regression in the source docstring would
# immediately break a test.


def test_get_rise_set_times_handles_no_set_within_window():
    """Regression test for coordinates.py ``get_rise_set_times`` docstring fix.

    Some sources rise within the search window but do not set within it.
    The fixed docstring example uses an explicit ``None`` check on
    ``set_`` before dereferencing ``set_.iso``.  This test follows the
    same pattern so a regression that removed the guard would crash here.
    """
    from astropy.time import Time

    from fyst_trajectories import Coordinates, get_fyst_site

    coords = Coordinates(get_fyst_site())
    start = Time("2026-03-15T00:00:00", scale="utc")
    rise, set_ = coords.get_rise_set_times(
        ra=83.633,
        dec=22.014,  # Crab Nebula / Orion neighborhood
        start_time=start,
        horizon=0.0,
        max_search_hours=24.0,
        step_hours=0.1,
    )
    # The exact guard pattern from the fixed docstring:
    if rise is not None and set_ is not None:
        rise_iso = rise.iso  # would crash if set_ check fired but rise didn't
        set_iso = set_.iso
        assert isinstance(rise_iso, str)
        assert isinstance(set_iso, str)
    else:
        # At least one of rise or set_ is None -- that's allowed and must
        # not raise.
        pass


def test_constant_el_pattern_docstring_example():
    """Regression test for ``ConstantElScanPattern`` docstring example.

    The docstring shows ``pattern.generate(site, duration=60.0)`` without
    ``start_time``.  This only works after the signature gained a default
    of ``start_time: Time | None = None``.
    """
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import ConstantElScanConfig, ConstantElScanPattern

    config = ConstantElScanConfig(
        timestep=0.1,
        az_start=120.0,
        az_stop=180.0,
        elevation=45.0,
        az_speed=0.5,
        az_accel=1.0,
        n_scans=10,
    )
    pattern = ConstantElScanPattern(config)
    trajectory = pattern.generate(get_fyst_site(), duration=60.0)
    assert trajectory.n_points > 0


def test_linear_motion_pattern_docstring_example():
    """Regression test for ``LinearMotionPattern`` docstring example.

    The docstring shows ``pattern.generate(site, duration=60.0)`` without
    ``start_time``.  This only works after the signature gained a default
    of ``start_time: Time | None = None``.
    """
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import LinearMotionConfig, LinearMotionPattern

    config = LinearMotionConfig(
        timestep=0.1,
        az_start=100.0,
        el_start=45.0,
        az_velocity=0.5,
        el_velocity=0.1,
    )
    pattern = LinearMotionPattern(config)
    trajectory = pattern.generate(get_fyst_site(), duration=60.0)
    assert trajectory.n_points > 0


# ============================================================================
# overhead_quickstart.rst examples
# ============================================================================


def test_overhead_quickstart_basic_usage():
    """Test the full Basic Usage walk-through from overhead_quickstart.rst.

    Generates an 8-hour timeline with two patches and explicit
    ``OverheadModel`` and ``CalibrationPolicy`` arguments. Verifies the
    timeline contains blocks of every expected type (the same regression
    class as the v0.3.0 Chandra Deep Field test for ``planning.rst``).
    """
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.overhead import (
        CalibrationPolicy,
        ObservingPatch,
        OverheadModel,
        compute_budget,
        generate_timeline,
    )

    site = get_fyst_site()

    patches = [
        ObservingPatch(
            name="Deep56",
            ra_center=24.0,
            dec_center=-32.0,
            width=40.0,
            height=10.0,
            scan_type="constant_el",
            velocity=1.0,
            elevation=50.0,
        ),
        ObservingPatch(
            name="Wide01",
            ra_center=180.0,
            dec_center=-30.0,
            width=20.0,
            height=10.0,
            scan_type="pong",
            velocity=0.5,
        ),
    ]

    overhead_model = OverheadModel(
        retune_duration=5.0,
        pointing_cal_duration=180.0,
        focus_duration=300.0,
        skydip_duration=300.0,
        planet_cal_duration=600.0,
        beam_map_duration=600.0,
        settle_time=5.0,
        min_scan_duration=60.0,
        max_scan_duration=3600.0,
    )

    calibration_policy = CalibrationPolicy(
        retune_cadence=0.0,
        pointing_cadence=3600.0,
        focus_cadence=7200.0,
        skydip_cadence=10800.0,
        planet_cal_cadence=43200.0,
        beam_map_cadence=None,
        planet_targets=("jupiter", "saturn", "mars", "uranus", "neptune"),
        planet_min_elevation=20.0,
    )

    timeline = generate_timeline(
        patches=patches,
        site=site,
        start_time="2026-06-15T02:00:00",
        end_time="2026-06-15T10:00:00",
        overhead_model=overhead_model,
        calibration_policy=calibration_policy,
    )

    # The doc snippet ends with ``print(f"{len(timeline)} blocks scheduled")``;
    # we verify the produced timeline is structurally sound too.
    assert len(timeline) > 0
    print(f"{len(timeline)} blocks scheduled")

    stats = compute_budget(timeline)
    assert "efficiency" in stats
    assert stats["science_time"] > 0
    assert 0.0 <= stats["efficiency"] <= 1.0
    print(f"Efficiency: {stats['efficiency']:.1%}")


def test_overhead_quickstart_save_and_load(tmp_path):
    """Test the Saving a Timeline ECSV round-trip from overhead_quickstart.rst."""
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.overhead import (
        ObservingPatch,
        generate_timeline,
        read_timeline,
        write_timeline,
    )

    site = get_fyst_site()
    patches = [
        ObservingPatch(
            name="Deep56",
            ra_center=24.0,
            dec_center=-32.0,
            width=40.0,
            height=10.0,
            scan_type="constant_el",
            velocity=1.0,
            elevation=50.0,
        ),
    ]
    timeline = generate_timeline(
        patches=patches,
        site=site,
        start_time="2026-06-15T02:00:00",
        end_time="2026-06-15T05:00:00",
    )

    out_path = tmp_path / "my_timeline.ecsv"
    write_timeline(timeline, str(out_path))
    loaded = read_timeline(str(out_path))
    print(f"Loaded {len(loaded)} blocks")
    assert len(loaded) == len(timeline)


# ============================================================================
# overhead_model.rst examples
# ============================================================================


def test_overhead_model_explicit_construction():
    """Test the OverheadModel construction snippet from overhead_model.rst."""
    from fyst_trajectories.overhead import OverheadModel

    model = OverheadModel(
        retune_duration=5.0,
        pointing_cal_duration=180.0,
        focus_duration=300.0,
        skydip_duration=300.0,
        planet_cal_duration=600.0,
        beam_map_duration=600.0,
        settle_time=5.0,
        min_scan_duration=60.0,
        max_scan_duration=3600.0,
    )
    assert model.retune_duration == 5.0
    assert model.beam_map_duration == 600.0
    assert model.max_scan_duration == 3600.0


def test_overhead_model_calibration_policy():
    """Test the CalibrationPolicy construction snippet from overhead_model.rst."""
    from fyst_trajectories.overhead import CalibrationPolicy

    policy = CalibrationPolicy(
        retune_cadence=0.0,
        pointing_cadence=3600.0,
        focus_cadence=7200.0,
        skydip_cadence=10800.0,
        planet_cal_cadence=43200.0,
        beam_map_cadence=None,
        planet_targets=("jupiter", "saturn", "mars", "uranus", "neptune"),
        planet_min_elevation=20.0,
    )
    assert policy.beam_map_cadence is None
    assert policy.pointing_cadence == 3600.0


def test_overhead_model_beam_map_cadence_opt_in():
    """Test the opt-in beam-map cadence example from overhead_model.rst."""
    from fyst_trajectories.overhead import CalibrationPolicy

    policy = CalibrationPolicy(beam_map_cadence=21600.0)
    assert policy.beam_map_cadence == 21600.0


def test_overhead_model_quick_commissioning_strategy():
    """Test the 'Quick commissioning' configuration from overhead_model.rst."""
    from fyst_trajectories.overhead import CalibrationPolicy, OverheadModel

    commissioning_policy = CalibrationPolicy(
        retune_cadence=0.0,
        pointing_cadence=900.0,
        focus_cadence=1800.0,
        skydip_cadence=3600.0,
    )
    commissioning_overhead = OverheadModel(max_scan_duration=600.0)
    assert commissioning_policy.pointing_cadence == 900.0
    assert commissioning_overhead.max_scan_duration == 600.0


def test_overhead_model_deep_survey_strategy():
    """Test the 'Deep survey' configuration from overhead_model.rst."""
    from fyst_trajectories.overhead import CalibrationPolicy, OverheadModel

    survey_policy = CalibrationPolicy(
        retune_cadence=0.0,
        pointing_cadence=7200.0,
        focus_cadence=14400.0,
        skydip_cadence=21600.0,
    )
    survey_overhead = OverheadModel(max_scan_duration=3600.0, min_scan_duration=120.0)
    assert survey_policy.skydip_cadence == 21600.0
    assert survey_overhead.min_scan_duration == 120.0


# ============================================================================
# overhead_timeline.rst examples
# ============================================================================


def test_overhead_timeline_observing_patch_setup():
    """Test the ObservingPatch construction examples from overhead_timeline.rst."""
    from fyst_trajectories.overhead import ObservingPatch

    deep_field = ObservingPatch(
        name="Deep56",
        ra_center=24.0,
        dec_center=-32.0,
        width=40.0,
        height=10.0,
        scan_type="constant_el",
        velocity=1.0,
        elevation=50.0,
    )
    assert deep_field.scan_type == "constant_el"

    wide_field = ObservingPatch(
        name="Wide01",
        ra_center=180.0,
        dec_center=-30.0,
        width=20.0,
        height=10.0,
        scan_type="pong",
        velocity=0.5,
        scan_params={"spacing": 0.1, "num_terms": 4},
    )
    assert wide_field.scan_type == "pong"
    assert wide_field.scan_params["spacing"] == 0.1


def test_overhead_timeline_from_field_region():
    """Test the ObservingPatch.from_field_region snippet from overhead_timeline.rst."""
    from fyst_trajectories.overhead import ObservingPatch
    from fyst_trajectories.planning import FieldRegion

    field = FieldRegion(ra_center=0.0, dec_center=-2.0, width=60.0, height=14.0)
    patch = ObservingPatch.from_field_region(
        field,
        name="Stripe82",
        scan_type="constant_el",
        velocity=1.0,
        elevation=50.0,
    )
    assert patch.name == "Stripe82"
    assert patch.ra_center == 0.0
    assert patch.elevation == 50.0


def test_overhead_timeline_custom_calibration_policy():
    """Test the Custom CalibrationPolicy walk-through from overhead_timeline.rst."""
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.overhead import (
        CalibrationPolicy,
        ObservingPatch,
        OverheadModel,
        generate_timeline,
    )

    site = get_fyst_site()

    policy = CalibrationPolicy(
        retune_cadence=0.0,
        pointing_cadence=1800.0,
        focus_cadence=3600.0,
        skydip_cadence=7200.0,
        planet_cal_cadence=43200.0,
    )
    overhead = OverheadModel(retune_duration=3.0, max_scan_duration=1800.0)

    patches = [
        ObservingPatch(
            name="CMB",
            ra_center=0.0,
            dec_center=-2.0,
            width=60.0,
            height=14.0,
            scan_type="constant_el",
            velocity=1.0,
            elevation=50.0,
        ),
    ]

    timeline = generate_timeline(
        patches=patches,
        site=site,
        start_time="2026-09-15T00:00:00",
        end_time="2026-09-15T12:00:00",
        overhead_model=overhead,
        calibration_policy=policy,
    )
    assert len(timeline) > 0
    print(timeline)


def test_overhead_timeline_schedule_to_trajectories():
    """Test the schedule_to_trajectories snippet from overhead_timeline.rst."""
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.overhead import (
        ObservingPatch,
        generate_timeline,
        schedule_to_trajectories,
    )

    site = get_fyst_site()
    patches = [
        ObservingPatch(
            name="CMB",
            ra_center=0.0,
            dec_center=-2.0,
            width=60.0,
            height=14.0,
            scan_type="constant_el",
            velocity=1.0,
            elevation=50.0,
        ),
    ]
    timeline = generate_timeline(
        patches=patches,
        site=site,
        start_time="2026-09-15T00:00:00",
        end_time="2026-09-15T03:00:00",
    )

    results = list(schedule_to_trajectories(timeline))
    assert len(results) > 0
    # At least one science block should have produced a trajectory.
    for timeline_block, scan_block in results:
        traj = scan_block.trajectory
        assert traj.n_points > 0


def test_overhead_timeline_validate():
    """Test the timeline.validate() snippet from overhead_timeline.rst."""
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.overhead import ObservingPatch, generate_timeline

    site = get_fyst_site()
    patches = [
        ObservingPatch(
            name="CMB",
            ra_center=0.0,
            dec_center=-2.0,
            width=60.0,
            height=14.0,
            scan_type="constant_el",
            velocity=1.0,
            elevation=50.0,
        ),
    ]
    timeline = generate_timeline(
        patches=patches,
        site=site,
        start_time="2026-09-15T00:00:00",
        end_time="2026-09-15T03:00:00",
    )

    warnings = timeline.validate()
    # validate() returns a list of strings (possibly empty); a freshly
    # generated timeline should be clean.
    assert isinstance(warnings, list)


# ============================================================================
# overhead_io.rst examples
# ============================================================================


def test_overhead_io_writing_a_timeline(tmp_path):
    """Test the Writing a Timeline snippet from overhead_io.rst."""
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.overhead import (
        ObservingPatch,
        generate_timeline,
        write_timeline,
    )

    site = get_fyst_site()
    patches = [
        ObservingPatch(
            name="Deep56",
            ra_center=24.0,
            dec_center=-32.0,
            width=40.0,
            height=10.0,
            scan_type="constant_el",
            velocity=1.0,
            elevation=50.0,
        ),
    ]
    timeline = generate_timeline(
        patches=patches,
        site=site,
        start_time="2026-06-15T00:00:00",
        end_time="2026-06-15T04:00:00",
    )

    out_path = tmp_path / "schedule.ecsv"
    write_timeline(timeline, str(out_path))
    assert out_path.exists()


def test_overhead_io_reading_a_timeline(tmp_path):
    """Test the Reading a Timeline snippet from overhead_io.rst."""
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.overhead import (
        ObservingPatch,
        generate_timeline,
        read_timeline,
        write_timeline,
    )

    site = get_fyst_site()
    patches = [
        ObservingPatch(
            name="Deep56",
            ra_center=24.0,
            dec_center=-32.0,
            width=40.0,
            height=10.0,
            scan_type="constant_el",
            velocity=1.0,
            elevation=50.0,
        ),
    ]
    timeline = generate_timeline(
        patches=patches,
        site=site,
        start_time="2026-06-15T00:00:00",
        end_time="2026-06-15T04:00:00",
    )
    out_path = tmp_path / "schedule.ecsv"
    write_timeline(timeline, str(out_path))

    loaded = read_timeline(str(out_path))
    print(f"Loaded {len(loaded)} blocks")
    print(f"Efficiency: {loaded.efficiency:.1%}")
    assert len(loaded) == len(timeline)


def test_overhead_io_toast_only_filter(tmp_path):
    """Test the science-only filter recipe from the TOAST Compatibility section."""
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.overhead import (
        ObservingPatch,
        ObservingTimeline,
        generate_timeline,
        write_timeline,
    )

    site = get_fyst_site()
    patches = [
        ObservingPatch(
            name="Deep56",
            ra_center=24.0,
            dec_center=-32.0,
            width=40.0,
            height=10.0,
            scan_type="constant_el",
            velocity=1.0,
            elevation=50.0,
        ),
    ]
    timeline = generate_timeline(
        patches=patches,
        site=site,
        start_time="2026-06-15T00:00:00",
        end_time="2026-06-15T04:00:00",
    )

    science_only = ObservingTimeline(
        blocks=timeline.science_blocks,
        site=timeline.site,
        start_time=timeline.start_time,
        end_time=timeline.end_time,
        overhead_model=timeline.overhead_model,
        calibration_policy=timeline.calibration_policy,
    )
    out_path = tmp_path / "toast_schedule.ecsv"
    write_timeline(science_only, str(out_path))
    assert out_path.exists()


# ============================================================================
# New top-level helpers (plan_pong_rotation_sequence, no_refraction)
# ============================================================================


def test_plan_pong_rotation_sequence_doc_example():
    """Test the ``plan_pong_rotation_sequence`` example from planning.rst.

    The 8-rotation case should produce angles at 22.5° spacing covering
    [0°, 180°). Verifies the doc claim
    ``[0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]``.
    """
    from fyst_trajectories import PongScanConfig
    from fyst_trajectories.planning import plan_pong_rotation_sequence

    base = PongScanConfig(
        timestep=0.1,
        width=2.0,
        height=2.0,
        spacing=0.1,
        velocity=0.5,
        num_terms=4,
        angle=0.0,
    )
    configs = plan_pong_rotation_sequence(base, n_rotations=8)
    angles = [c.angle for c in configs]
    expected = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]
    assert len(angles) == 8
    for got, want in zip(angles, expected):
        assert abs(got - want) < 1e-9


def test_plan_pong_rotation_sequence_full_planning_example():
    """Test the full planning.rst snippet that schedules each rotation back-to-back."""
    from astropy.time import TimeDelta

    from fyst_trajectories import PongScanConfig, get_fyst_site
    from fyst_trajectories.planning import (
        FieldRegion,
        plan_pong_rotation_sequence,
        plan_pong_scan,
    )

    site = get_fyst_site()
    base = PongScanConfig(
        timestep=0.1,
        width=2.0,
        height=2.0,
        spacing=0.1,
        velocity=0.5,
        num_terms=4,
        angle=0.0,
    )
    configs = plan_pong_rotation_sequence(base, n_rotations=8)

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
    assert len(blocks) == 8


def test_no_refraction_atmosphere_pattern():
    """Test that ``Coordinates(site)`` produces vacuum coordinates without warning.

    Bare ``Coordinates(site)`` defaults to vacuum (no refraction) because
    the FYST ACU applies atmospheric refraction downstream. No warning
    is emitted. ``AtmosphericConditions.no_refraction()`` is available as
    an explicit opt-in synonym for the same behaviour.
    """
    from fyst_trajectories import AtmosphericConditions, Coordinates, get_fyst_site

    site = get_fyst_site()

    # Bare construction: vacuum, no warning.
    coords_bare = Coordinates(site)
    assert coords_bare.atmosphere.pressure_hpa == 0

    # Explicit no_refraction: identical result.
    coords_explicit = Coordinates(site, atmosphere=AtmosphericConditions.no_refraction())
    assert coords_explicit.atmosphere.pressure_hpa == 0

    obstime = Time("2026-01-15T02:00:00", scale="utc")
    az, el = coords_bare.radec_to_altaz(83.633, 22.014, obstime=obstime)
    assert isinstance(az, float)
    assert isinstance(el, float)


def test_print_trajectory_tail_none():
    """Test ``print_trajectory(trajectory, tail=None)`` from trajectory_examples.rst.

    Closes Finding 12 from final_docs_freshness_review (the doc claims
    ``tail=None`` shows only the first 5 points; this test exercises that
    variant).
    """
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import SiderealTrackConfig, TrajectoryBuilder

    site = get_fyst_site()
    start_time = Time("2026-01-15T02:00:00", scale="utc")

    trajectory = (
        TrajectoryBuilder(site)
        .at(ra=83.633, dec=22.014)
        .with_config(SiderealTrackConfig(timestep=0.1))
        .duration(10.0)
        .starting_at(start_time)
        .build()
    )

    # Should print only the first 5 points (tail=None means "skip tail").
    print_trajectory(trajectory, tail=None)


# ============================================================================
# retune_events.rst examples
# ============================================================================


def _build_tiny_trajectory_for_retune_docs():
    """Build a minimal trajectory that is long enough for the doc examples."""
    import numpy as np

    from fyst_trajectories import Trajectory

    # A simple 1000-second fixed-position trajectory is sufficient for
    # exercising ``inject_retune`` in both modes and writing an ECSV
    # round-trip; no pattern machinery is needed.
    times = np.arange(0.0, 1001.0, 0.5)
    return Trajectory(
        times=times,
        az=np.full_like(times, 100.0),
        el=np.full_like(times, 50.0),
        az_vel=np.zeros_like(times),
        el_vel=np.zeros_like(times),
    )


def test_retune_events_uniform_mode():
    """Uniform-cadence example from retune_events.rst."""
    from fyst_trajectories import inject_retune

    traj = _build_tiny_trajectory_for_retune_docs()
    retuned = inject_retune(
        traj,
        retune_interval=300.0,
        retune_duration=5.0,
    )
    assert len(retuned.retune_events) > 0


def test_retune_events_event_list_mode():
    """Event-list example from retune_events.rst."""
    from fyst_trajectories import RetuneEvent, inject_retune

    traj = _build_tiny_trajectory_for_retune_docs()
    events = [
        RetuneEvent(t_start=30.0, duration=5.0),
        RetuneEvent(t_start=300.0, duration=5.0),
        RetuneEvent(t_start=600.0, duration=5.0),
    ]
    retuned = inject_retune(traj, retune_events=events)
    assert retuned.retune_events == tuple(events)


def test_retune_events_csv_parse_snippet(tmp_path):
    """CSV-parsing snippet from retune_events.rst."""
    import csv

    from fyst_trajectories import RetuneEvent, inject_retune

    csv_path = tmp_path / "retunes.csv"
    csv_path.write_text(
        "t_start_s,duration_s,module_index\n30.0,5.0,0\n300.0,5.0,0\n600.0,8.0,0\n",
        encoding="ascii",
    )

    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        events = [
            RetuneEvent(
                t_start=float(row["t_start_s"]),
                duration=float(row["duration_s"]),
            )
            for row in reader
        ]
    assert len(events) == 3

    traj = _build_tiny_trajectory_for_retune_docs()
    retuned = inject_retune(traj, retune_events=events)
    assert retuned.retune_events == tuple(events)


def test_retune_events_ecsv_round_trip(tmp_path):
    """ECSV round-trip example from retune_events.rst.

    Attach a list of ``RetuneEvent`` to the first block's metadata,
    write the timeline, read it back, and verify the decoded tuple
    matches (modulo ``RetuneEvent`` equality).
    """
    from fyst_trajectories import RetuneEvent, get_fyst_site
    from fyst_trajectories.overhead import (
        CalibrationPolicy,
        ObservingPatch,
        OverheadModel,
        generate_timeline,
        read_timeline,
        write_timeline,
    )

    site = get_fyst_site()
    patches = [
        ObservingPatch(
            name="Deep56",
            ra_center=24.0,
            dec_center=-32.0,
            width=40.0,
            height=10.0,
            scan_type="constant_el",
            velocity=1.0,
            elevation=50.0,
        ),
    ]
    timeline = generate_timeline(
        patches=patches,
        site=site,
        start_time="2026-06-15T02:00:00",
        end_time="2026-06-15T03:00:00",
        overhead_model=OverheadModel(),
        calibration_policy=CalibrationPolicy(),
    )
    events = [
        RetuneEvent(t_start=30.0, duration=5.0),
        RetuneEvent(t_start=300.0, duration=5.0),
    ]
    timeline.blocks[0].metadata["retune_events"] = events

    out = tmp_path / "night.ecsv"
    write_timeline(timeline, out)

    loaded = read_timeline(out)
    loaded_events = loaded.blocks[0].metadata["retune_events"]
    assert loaded_events == tuple(events)
