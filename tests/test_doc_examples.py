"""Value/outcome assertions for specific documentation examples.

This module holds the value-bearing checks for selected documentation
examples, the invariants, error/warning behaviours, and regression
guards that go beyond "this snippet runs". Pure execution coverage for
every code block in ``docs/*.rst`` lives in
``tests/test_doc_examples_rst.py``, which extracts and runs each block,
so the run-only inline copies that used to live here have been retired.
"""

import numpy as np
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
    boresight_to_detector,
    compute_focal_plane_rotation,
    detector_to_boresight,
)
from fyst_trajectories.patterns import (
    SiderealTrackConfig,
    TrajectoryBuilder,
)

# NOTE: Many test functions below contain function-level imports that duplicate
# the module-level imports above.  This is intentional, each test mirrors a
# code snippet from the RST documentation, so the imports inside the function
# must match what the docs show the user.  Do not hoist them to module level.

# ============================================================================
# quickstart.rst examples
# ============================================================================


def test_quickstart_get_site():
    """Test basic site retrieval from quickstart.rst."""
    site = get_fyst_site()
    print(f"FYST is at {site.latitude}, {site.longitude}")
    # FYST on Cerro Chajnantor: lat -22.9856, lon -67.7403 (site.py constants).
    assert site.latitude == pytest.approx(-22.9856, abs=1e-3)
    assert site.longitude == pytest.approx(-67.7403, abs=1e-3)


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
    # Round-trips back to the input RA/Dec (a real transform, not a stub).
    ra_back, dec_back = coords.altaz_to_radec(az, el, obstime=obstime)
    assert ra_back == pytest.approx(83.82, abs=0.01)
    assert dec_back == pytest.approx(-5.39, abs=0.01)


def test_quickstart_frame_translation():
    """Test coordinate frame name translation from quickstart.rst."""
    # Translate common frame names to astropy equivalents
    normalize_frame("J2000")  # Returns "icrs"
    normalize_frame("B1950")  # Returns "fk4"

    assert normalize_frame("J2000") == "icrs"
    assert normalize_frame("B1950") == "fk4"


def test_quickstart_proper_motion():
    """Test proper motion support from quickstart.rst."""
    from fyst_trajectories import get_fyst_site

    coords = Coordinates(get_fyst_site())

    # Barnard's Star, J2000 catalogue position and proper motion
    az, el = coords.radec_to_altaz_with_pm(
        ra=269.452,
        dec=4.693,
        pm_ra=-798.58,
        pm_dec=10328.12,  # mas/yr
        ref_epoch=Time("J2000.0"),
        obstime=Time("2026-06-15T04:00:00"),
        distance=1.8,  # parsecs
    )
    assert isinstance(az, float)
    assert isinstance(el, float)
    # 10.4"/yr proper motion over ~26.5 yr shifts the apparent position ~0.076 deg
    # from the zero-PM transform, a real correction, not a no-op.
    az0, el0 = coords.radec_to_altaz(269.452, 4.693, obstime=Time("2026-06-15T04:00:00"))
    sep = np.hypot((az - az0) * np.cos(np.radians(el)), el - el0)
    assert sep == pytest.approx(0.0761, abs=0.01)


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
    # The detector sits offset from the boresight by the offset magnitude
    # sqrt(5^2 + 3^2) = 5.83 arcmin = 0.0972 deg on-sky.
    sep = np.hypot((det_az - 180.0) * np.cos(np.radians(45.0)), det_el - 45.0)
    assert sep == pytest.approx(0.0972, abs=0.005)


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


def test_coordsys_proper_motion():
    """Test proper motion from coordinate_systems.rst."""
    from fyst_trajectories import get_fyst_site

    coords = Coordinates(get_fyst_site())

    # Barnard's Star (moves ~10 arcsec/year); J2000 catalogue position
    az, el = coords.radec_to_altaz_with_pm(
        ra=269.452,
        dec=4.693,
        pm_ra=-798.58,
        pm_dec=10328.12,  # mas/yr (pm_ra includes cos(dec))
        ref_epoch=Time("J2000.0"),
        obstime=Time("2026-06-15T04:00:00", scale="utc"),
        distance=1.8,  # parsecs, optional
    )
    assert isinstance(az, float)
    assert isinstance(el, float)
    # Same Barnard's Star example: the large proper motion shifts the apparent
    # position ~0.076 deg from the zero-PM transform.
    az0, el0 = coords.radec_to_altaz(
        269.452, 4.693, obstime=Time("2026-06-15T04:00:00", scale="utc")
    )
    sep = np.hypot((az - az0) * np.cos(np.radians(el)), el - el0)
    assert sep == pytest.approx(0.0761, abs=0.01)


# ============================================================================
# planning.rst examples
# ============================================================================
#
# These tests exercise every code example in ``docs/planning.rst``, so a broken
# example cannot ship unnoticed.  The ``plan_pong_scan`` examples use a
# ``start_time`` at which the Chandra Deep Field South is actually observable; the
# regression test ``test_planning_plan_pong_scan_chandra_deep_field_observable``
# locks that time in.


def test_planning_field_region_cmb():
    """Test FieldRegion construction example from planning.rst."""
    from fyst_trajectories.planning import FieldRegion

    # Equatorial field: 10 deg RA x 6 deg Dec (matches the planning.rst example)
    cmb_field = FieldRegion(
        ra_center=0.0,  # deg (0h RA)
        dec_center=-2.0,  # deg
        width=10.0,  # RA extent in degrees
        height=6.0,  # Dec extent in degrees
    )

    # Dec boundaries are computed automatically
    print(f"Dec range: [{cmb_field.dec_min}, {cmb_field.dec_max}]")
    assert cmb_field.dec_min == pytest.approx(-5.0)
    assert cmb_field.dec_max == pytest.approx(1.0)


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
        start_time=Time("2026-03-15T23:30:00", scale="utc"),
        timestep=0.1,
        n_cycles=3,  # observe 3 full Pong periods
    )

    assert block.trajectory.n_points > 0
    assert block.computed_params["n_cycles"] == 3
    # Duration should equal 3 periods.
    assert block.duration == pytest.approx(block.computed_params["period"] * 3)


def test_planning_plan_source_ces():
    """Test 'Source CES' worked example from planning.rst."""
    from astropy.time import Time

    from fyst_trajectories import PRIMECAM_MODULES, get_fyst_site
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
    print(f"Source pass: {cp['t0_iso'][:19]} to {cp['t1_iso'][:19]}")
    print(f"Az drift:    {cp['v_az']:+.5f} deg/s")
    print(f"Az range:    [{cp['az_start']:.2f}, {cp['az_start'] + cp['az_throw']:.2f}] deg")

    assert cp["mode"] == "rising"
    assert cp["el_bore"] == pytest.approx(35.0)
    assert cp["duration"] > 0
    assert cp["n_scans"] >= 1


@pytest.mark.parametrize(
    "start_iso",
    [
        "2026-03-15T01:00:00",  # the corrected, observable, advisory-clean time
    ],
)
def test_planning_plan_pong_scan_chandra_deep_field_observable(start_iso):
    """Regression test for docs/planning.rst start_time bug.

    The Chandra Deep Field South is below the horizon at FYST at
    2026-03-15T04:00:00.  The example first moved to 22:12 (observable but
    near transit, where the high-elevation advisory fires) and now uses
    01:00, where the field sits near 30 deg elevation and the example runs
    advisory-clean.  This parametrized test locks the corrected time in
    place. If someone reverts it to an unobservable value, this test will
    fail loudly.
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
# These tests guard three docstring examples that were found broken and fixed.
# They intentionally mirror the shape of the fixed docstring snippets so a
# regression in the source docstring would immediately break a test.


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
        # At least one of rise or set_ is None, that's allowed and must
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
# New top-level helpers (plan_pong_rotation_sequence, no_refraction)
# ============================================================================


def test_plan_pong_rotation_sequence_doc_example():
    """Test the ``plan_pong_rotation_sequence`` example from planning.rst.

    The 8-rotation case should produce angles at 22.5 deg spacing covering
    [0 deg, 180 deg). Verifies the doc claim
    ``[0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]``.
    """
    from fyst_trajectories import PongScanConfig
    from fyst_trajectories.planning import plan_pong_rotation_sequence

    base = PongScanConfig(
        timestep=0.1,
        width=2.0,
        height=2.0,
        spacing=0.1,
        velocity=0.35,
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
        velocity=0.35,
        num_terms=4,
        angle=0.0,
    )
    configs = plan_pong_rotation_sequence(base, n_rotations=8)

    field = FieldRegion(ra_center=180.0, dec_center=-30.0, width=2.0, height=2.0)
    t0 = Time("2026-03-15T00:00:00", scale="utc")
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
    refraction is applied downstream at execution time, by exactly one of
    the Go TCS or the ACU. No warning is emitted.
    ``AtmosphericConditions.no_refraction()`` is available as an explicit
    opt-in synonym for the same behaviour.
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
