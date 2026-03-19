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

    points = trajectory.to_path_format()  # List of [time, az, el, az_vel, el_vel]

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
        trajectory.validate(site)

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
    points = trajectory.to_path_format()

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
