"""Tests for instrument offset functionality."""

import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import TETE, SkyCoord
from astropy.time import Time

from fyst_trajectories.coordinates import Coordinates
from fyst_trajectories.exceptions import TargetNotObservableError
from fyst_trajectories.offsets import (
    _INVERSE_EARLY_EXIT_THRESHOLD,
    _INVERSE_FAILURE_THRESHOLD,
    InstrumentOffset,
    apply_detector_offset,
    boresight_to_detector,
    compute_focal_plane_rotation,
    detector_to_boresight,
)
from fyst_trajectories.patterns import (
    ConstantElScanConfig,
    PongScanConfig,
    TrajectoryBuilder,
)
from fyst_trajectories.primecam import (
    PRIMECAM_I1,
    PRIMECAM_MODULES,
    get_primecam_offset,
)
from fyst_trajectories.site import (
    AtmosphericConditions,
    AxisLimits,
    Site,
    SunAvoidanceConfig,
    TelescopeLimits,
    get_fyst_site,
)
from fyst_trajectories.trajectory import RetuneEvent, Trajectory
from fyst_trajectories.trajectory_utils import get_absolute_times, inject_retune


class TestBoresightToDetector:
    """Tests for boresight_to_detector function."""

    def test_zero_offset_no_change(self):
        """Test that zero offset produces no change."""
        offset = InstrumentOffset(dx=0.0, dy=0.0)
        det_az, det_el = boresight_to_detector(180.0, 45.0, offset, field_rotation=0.0)
        assert det_az == pytest.approx(180.0, abs=1e-12)
        assert det_el == pytest.approx(45.0, abs=1e-12)

    def test_x_offset_increases_azimuth(self):
        """Test that positive x offset increases azimuth."""
        offset = InstrumentOffset(dx=60.0, dy=0.0)  # 1 degree in arcmin
        det_az, _det_el = boresight_to_detector(180.0, 45.0, offset, field_rotation=0.0)

        assert det_az > 180.0

    def test_y_offset_increases_elevation(self):
        """Test that positive y offset increases elevation."""
        offset = InstrumentOffset(dx=0.0, dy=60.0)  # 1 degree in arcmin
        det_az, det_el = boresight_to_detector(180.0, 45.0, offset, field_rotation=0.0)

        # Pure elevation offset: spherical gives same result
        assert det_az == pytest.approx(180.0, abs=1e-10)
        assert det_el == pytest.approx(46.0, abs=1e-10)

    def test_field_rotation_90_degrees(self):
        """Test that 90 degree field rotation swaps x and y."""
        offset = InstrumentOffset(dx=60.0, dy=0.0)  # 1 degree x offset

        # With 90 degree rotation, x offset becomes y offset
        det_az_90, det_el_90 = boresight_to_detector(180.0, 45.0, offset, field_rotation=90.0)

        # x offset rotated by 90 deg -> pure elevation offset
        assert det_az_90 == pytest.approx(180.0, abs=1e-6)
        assert det_el_90 == pytest.approx(46.0, rel=1e-4)

    def test_field_rotation_180_degrees(self):
        """Test that 180 degree field rotation approximately inverts offsets.

        On the sphere, the inversion is not exact because great-circle
        offsets are nonlinear. Both azimuth and elevation components
        invert approximately, with small residuals due to the curvature.
        """
        offset = InstrumentOffset(dx=60.0, dy=30.0)

        det_az_0, det_el_0 = boresight_to_detector(180.0, 45.0, offset, field_rotation=0.0)
        det_az_180, det_el_180 = boresight_to_detector(180.0, 45.0, offset, field_rotation=180.0)

        el_diff_0 = det_el_0 - 45.0
        el_diff_180 = det_el_180 - 45.0
        az_diff_0 = det_az_0 - 180.0
        az_diff_180 = det_az_180 - 180.0

        # Both components approximately invert (within ~4% for 1 degree offsets)
        assert az_diff_180 == pytest.approx(-az_diff_0, rel=0.02)
        assert el_diff_180 == pytest.approx(-el_diff_0, rel=0.04)

    def test_array_input(self):
        """Test with array inputs."""
        offset = InstrumentOffset(dx=30.0, dy=30.0)
        az = np.array([100.0, 150.0, 200.0])
        el = np.array([30.0, 45.0, 60.0])

        det_az, det_el = boresight_to_detector(az, el, offset, field_rotation=0.0)

        assert len(det_az) == 3
        assert len(det_el) == 3
        assert all(det_el > el)  # All elevations should increase

    def test_array_field_rotation(self):
        """Test with array field rotation values."""
        offset = InstrumentOffset(dx=60.0, dy=0.0)
        field_rotation = np.array([0.0, 90.0, 180.0])

        det_az, det_el = boresight_to_detector(180.0, 45.0, offset, field_rotation=field_rotation)

        assert len(det_az) == 3
        assert len(det_el) == 3


class TestDetectorToBoresight:
    """Tests for detector_to_boresight function."""

    def test_zero_offset_no_change(self):
        """Test that zero offset produces no change."""
        offset = InstrumentOffset(dx=0.0, dy=0.0)
        bore_az, bore_el = detector_to_boresight(180.0, 45.0, offset, field_rotation=0.0)
        assert bore_az == pytest.approx(180.0, abs=1e-12)
        assert bore_el == pytest.approx(45.0, abs=1e-12)

    def test_inverse_relationship(self):
        """Test that detector_to_boresight is inverse of boresight_to_detector."""
        offset = InstrumentOffset(dx=30.0, dy=20.0)
        bore_az, bore_el = 180.0, 45.0

        det_az, det_el = boresight_to_detector(bore_az, bore_el, offset, field_rotation=0.0)
        bore_az_recovered, bore_el_recovered = detector_to_boresight(
            det_az, det_el, offset, field_rotation=0.0
        )

        assert bore_az_recovered == pytest.approx(bore_az, abs=0.01 / 3600.0)
        assert bore_el_recovered == pytest.approx(bore_el, abs=0.01 / 3600.0)

    def test_inverse_with_field_rotation(self):
        """Test inverse relationship with field rotation."""
        offset = InstrumentOffset(dx=30.0, dy=20.0)
        field_rotation = 45.0
        bore_az, bore_el = 180.0, 45.0

        det_az, det_el = boresight_to_detector(
            bore_az,
            bore_el,
            offset,
            field_rotation=field_rotation,
        )
        bore_az_recovered, bore_el_recovered = detector_to_boresight(
            det_az,
            det_el,
            offset,
            field_rotation=field_rotation,
        )

        assert bore_az_recovered == pytest.approx(bore_az, abs=0.01 / 3600.0)
        assert bore_el_recovered == pytest.approx(bore_el, abs=0.01 / 3600.0)

    def test_inverse_with_large_offset(self):
        """Test inverse relationship with large offset."""
        offset = InstrumentOffset(dx=120.0, dy=60.0)  # 2 deg, 1 deg
        bore_az, bore_el = 200.0, 50.0

        det_az, det_el = boresight_to_detector(bore_az, bore_el, offset, field_rotation=0.0)
        bore_az_recovered, bore_el_recovered = detector_to_boresight(
            det_az, det_el, offset, field_rotation=0.0
        )

        assert bore_az_recovered == pytest.approx(bore_az, abs=0.01 / 3600.0)
        assert bore_el_recovered == pytest.approx(bore_el, abs=0.01 / 3600.0)

    def test_array_input_inverse(self):
        """Test inverse with array inputs."""
        offset = InstrumentOffset(dx=30.0, dy=30.0)
        bore_az = np.array([100.0, 150.0, 200.0])
        bore_el = np.array([30.0, 45.0, 60.0])

        det_az, det_el = boresight_to_detector(bore_az, bore_el, offset, field_rotation=0.0)
        bore_az_recovered, bore_el_recovered = detector_to_boresight(
            det_az, det_el, offset, field_rotation=0.0
        )

        np.testing.assert_allclose(bore_az_recovered, bore_az, atol=0.01 / 3600.0)
        np.testing.assert_allclose(bore_el_recovered, bore_el, atol=0.01 / 3600.0)


class TestApplyDetectorOffset:
    """Tests for apply_detector_offset function."""

    def test_no_start_time_required(self, site):
        """Mechanical (horizon-frame) rotation needs no timestamps.

        Regression for the pa-in-horizon-frame fix: ``start_time`` was only
        ever required to evaluate the parallactic angle, which does not
        belong in this az/el projection. A trajectory without ``start_time``
        must be accepted and produce the same boresight as the identical
        trajectory with ``start_time`` set.
        """
        offset = InstrumentOffset(dx=5.0, dy=3.0)

        def _traj(start_time):
            return Trajectory(
                times=np.array([0.0, 1.0, 2.0]),
                az=np.array([180.0, 181.0, 182.0]),
                el=np.array([45.0, 45.0, 45.0]),
                az_vel=np.array([1.0, 1.0, 1.0]),
                el_vel=np.array([0.0, 0.0, 0.0]),
                start_time=start_time,
            )

        adj_no_time = apply_detector_offset(_traj(None), offset, site)
        adj_with_time = apply_detector_offset(
            _traj(Time("2026-03-15T04:00:00", scale="utc")), offset, site
        )

        np.testing.assert_allclose(adj_no_time.az, adj_with_time.az)
        np.testing.assert_allclose(adj_no_time.el, adj_with_time.el)

    def test_zero_offset_preserves_trajectory(self, site):
        """Test that zero offset returns same positions."""
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

        offset = InstrumentOffset(dx=0.0, dy=0.0)
        adjusted = apply_detector_offset(trajectory, offset, site)

        np.testing.assert_allclose(adjusted.az, trajectory.az, rtol=1e-10)
        np.testing.assert_allclose(adjusted.el, trajectory.el, rtol=1e-10)

    def test_offset_changes_positions(self, site):
        """Test that non-zero offset changes positions."""
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

        offset = InstrumentOffset(dx=30.0, dy=30.0)  # 0.5 deg offset
        adjusted = apply_detector_offset(trajectory, offset, site)

        assert not np.allclose(adjusted.az, trajectory.az)
        # Inverse offset: boresight shifts opposite to detector, so elevation drops
        assert np.mean(adjusted.el) < np.mean(trajectory.el)

    def test_preserves_metadata(self, site):
        """Test that metadata is preserved."""
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

        offset = InstrumentOffset(dx=30.0, dy=30.0)
        adjusted = apply_detector_offset(trajectory, offset, site)

        assert adjusted.metadata is not None
        assert adjusted.pattern_type == trajectory.pattern_type
        assert adjusted.center_ra == trajectory.center_ra
        assert adjusted.center_dec == trajectory.center_dec

    def test_preserves_start_time(self, site):
        """Test that start_time is preserved."""
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

        offset = InstrumentOffset(dx=30.0, dy=30.0)
        adjusted = apply_detector_offset(trajectory, offset, site)

        assert adjusted.start_time == start_time

    def test_preserves_scan_flag_with_offset(self, site):
        """Test that scan_flag is preserved when applying a non-zero offset."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        n = 10
        scan_flag = np.array([1, 1, 1, 2, 2, 2, 1, 1, 1, 2], dtype=np.int8)
        trajectory = Trajectory(
            times=np.linspace(0, 9, n),
            az=np.full(n, 180.0),
            el=np.full(n, 45.0),
            az_vel=np.zeros(n),
            el_vel=np.zeros(n),
            start_time=start_time,
            scan_flag=scan_flag,
        )

        offset = InstrumentOffset(dx=30.0, dy=30.0)
        adjusted = apply_detector_offset(trajectory, offset, site)

        assert adjusted.scan_flag is not None
        np.testing.assert_array_equal(adjusted.scan_flag, scan_flag)

    def test_preserves_scan_flag_with_zero_offset(self, site):
        """Test that scan_flag is preserved for the zero-offset early-exit path."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        n = 5
        scan_flag = np.array([1, 2, 1, 2, 1], dtype=np.int8)
        trajectory = Trajectory(
            times=np.linspace(0, 4, n),
            az=np.full(n, 180.0),
            el=np.full(n, 45.0),
            az_vel=np.zeros(n),
            el_vel=np.zeros(n),
            start_time=start_time,
            scan_flag=scan_flag,
        )

        offset = InstrumentOffset(dx=0.0, dy=0.0)
        adjusted = apply_detector_offset(trajectory, offset, site)

        assert adjusted.scan_flag is not None
        np.testing.assert_array_equal(adjusted.scan_flag, scan_flag)


class TestPrimeCamOffsets:
    """Tests for predefined PrimeCam offsets.

    Module-lookup, center-is-zero, and inner-ring-equidistant coverage lives in
    test_primecam.py (TestGetPrimecamOffset / TestCenterModule / the hexagonal
    symmetry tests); only the I1 direction sign is checked here.
    """

    def test_i1_offset_direction(self):
        """Test I1 module is in correct direction."""
        # I1 is at theta=-90, which means dy=-inner_ring_distance
        assert PRIMECAM_I1.dx == pytest.approx(0.0, abs=1e-10)
        assert PRIMECAM_I1.dy < 0  # Negative y direction


class TestBuilderForDetector:
    """Tests for TrajectoryBuilder.for_detector() integration."""

    def test_for_detector_changes_positions(self, site):
        """Test that for_detector changes trajectory positions."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        offset = InstrumentOffset(dx=30.0, dy=30.0)

        pong_config = PongScanConfig(
            timestep=0.1,
            width=1.0,
            height=1.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        )

        # Build without offset
        trajectory_without = (
            TrajectoryBuilder(site)
            .at(ra=180.0, dec=-30.0)
            .with_config(pong_config)
            .duration(60.0)
            .starting_at(start_time)
            .build()
        )

        # Build with offset
        trajectory_with = (
            TrajectoryBuilder(site)
            .at(ra=180.0, dec=-30.0)
            .with_config(pong_config)
            .for_detector(offset)
            .duration(60.0)
            .starting_at(start_time)
            .build()
        )

        assert not np.allclose(trajectory_with.az, trajectory_without.az)
        assert not np.allclose(trajectory_with.el, trajectory_without.el)

    def test_for_detector_with_primecam(self, site):
        """PrimeCam predefined offset displaces pointing by the module distance."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = PongScanConfig(
            timestep=0.1,
            width=1.0,
            height=1.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        )

        traj_boresight = (
            TrajectoryBuilder(site)
            .at(ra=180.0, dec=-30.0)
            .with_config(config)
            .duration(60.0)
            .starting_at(start_time)
            .build()
        )
        traj_i1 = (
            TrajectoryBuilder(site)
            .at(ra=180.0, dec=-30.0)
            .with_config(config)
            .for_detector(get_primecam_offset("i1"))
            .duration(60.0)
            .starting_at(start_time)
            .build()
        )

        assert traj_i1.n_points == traj_boresight.n_points
        # I1 sits ~1.78 deg off-axis, so applying it displaces the pointing from
        # the boresight pointing by that distance on-sky (a no-op offset would
        # leave the two trajectories identical).
        d_el = traj_i1.el - traj_boresight.el
        d_az = (traj_i1.az - traj_boresight.az) * np.cos(np.radians(traj_i1.el))
        sep = np.hypot(d_az, d_el)
        assert np.median(sep) == pytest.approx(1.78, abs=0.05)


class TestOffsetRoundTrips:
    """Comprehensive round-trip tests for offset transformations."""

    @pytest.mark.parametrize(
        "dx,dy",
        [
            (0.0, 0.0),  # Zero offset
            (30.0, 0.0),  # X only
            (0.0, 30.0),  # Y only
            (30.0, 30.0),  # Both positive
            (-30.0, 30.0),  # Mixed signs
            (30.0, -30.0),  # Mixed signs
            (-30.0, -30.0),  # Both negative
            (60.0, 60.0),  # 1 degree offset
            (120.0, 60.0),  # Large offset
        ],
    )
    def test_round_trip_various_offsets(self, dx, dy):
        """Test round-trip for various offset values."""
        offset = InstrumentOffset(dx=dx, dy=dy)
        bore_az, bore_el = 180.0, 45.0

        det_az, det_el = boresight_to_detector(bore_az, bore_el, offset, field_rotation=0.0)

        bore_az_back, bore_el_back = detector_to_boresight(
            det_az, det_el, offset, field_rotation=0.0
        )

        assert bore_az_back == pytest.approx(bore_az, abs=0.01 / 3600.0)
        assert bore_el_back == pytest.approx(bore_el, abs=0.01 / 3600.0)

    @pytest.mark.parametrize(
        "az,el",
        [
            (0.0, 30.0),  # North
            (90.0, 30.0),  # East
            (180.0, 30.0),  # South
            (270.0, 30.0),  # West
            (180.0, 20.0),  # Low elevation
            (180.0, 60.0),  # High elevation
            (180.0, 85.0),  # Near zenith
            (45.0, 45.0),  # Intermediate
            (315.0, 50.0),  # Another quadrant
        ],
    )
    def test_round_trip_various_positions(self, az, el):
        """Test round-trip at various telescope positions."""
        offset = InstrumentOffset(dx=30.0, dy=20.0)

        det_az, det_el = boresight_to_detector(az, el, offset, field_rotation=0.0)

        az_back, el_back = detector_to_boresight(det_az, det_el, offset, field_rotation=0.0)

        assert az_back == pytest.approx(az, abs=0.01 / 3600.0)
        assert el_back == pytest.approx(el, abs=0.01 / 3600.0)

    @pytest.mark.parametrize(
        "field_rotation",
        [0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 270.0, -45.0, -90.0],
    )
    def test_round_trip_various_field_rotations(self, field_rotation):
        """Test round-trip at various field rotation angles."""
        offset = InstrumentOffset(dx=30.0, dy=20.0)
        bore_az, bore_el = 180.0, 45.0

        det_az, det_el = boresight_to_detector(
            bore_az,
            bore_el,
            offset,
            field_rotation=field_rotation,
        )

        bore_az_back, bore_el_back = detector_to_boresight(
            det_az,
            det_el,
            offset,
            field_rotation=field_rotation,
        )

        assert bore_az_back == pytest.approx(bore_az, abs=0.01 / 3600.0)
        assert bore_el_back == pytest.approx(bore_el, abs=0.01 / 3600.0)

    def test_round_trip_with_arrays(self):
        """Test round-trip with array inputs."""
        offset = InstrumentOffset(dx=30.0, dy=20.0)

        bore_az = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
        bore_el = np.array([25.0, 35.0, 45.0, 55.0, 65.0])
        field_rotation = np.array([0.0, 30.0, 60.0, 90.0, 120.0])

        det_az, det_el = boresight_to_detector(
            bore_az,
            bore_el,
            offset,
            field_rotation=field_rotation,
        )

        bore_az_back, bore_el_back = detector_to_boresight(
            det_az,
            det_el,
            offset,
            field_rotation=field_rotation,
        )

        np.testing.assert_allclose(bore_az_back, bore_az, atol=0.01 / 3600.0)
        np.testing.assert_allclose(bore_el_back, bore_el, atol=0.01 / 3600.0)

    @pytest.mark.parametrize(
        "offset_arcmin,el,field_rotation",
        [
            (6.0, 30.0, 0.0),  # Small offset, low el
            (60.0, 45.0, 45.0),  # 1 deg offset, mid el
            (106.8, 45.0, 90.0),  # PrimeCam inner ring
            (180.0, 60.0, 120.0),  # 3 deg offset
            (300.0, 45.0, 0.0),  # 5 deg offset
            (300.0, 80.0, 60.0),  # 5 deg offset, high el
            (60.0, 20.0, 270.0),  # 1 deg offset, low el
        ],
    )
    def test_round_trip_large_offsets(self, offset_arcmin, el, field_rotation):
        """Test round-trip accuracy for various offset/elevation/rotation combos."""
        offset = InstrumentOffset(dx=offset_arcmin, dy=offset_arcmin * 0.5)
        bore_az = 200.0

        det_az, det_el = boresight_to_detector(
            bore_az,
            el,
            offset,
            field_rotation=field_rotation,
        )
        bore_az_back, bore_el_back = detector_to_boresight(
            det_az,
            det_el,
            offset,
            field_rotation=field_rotation,
        )

        # Round-trip should be accurate to < 0.01 arcsec for all cases
        assert bore_az_back == pytest.approx(bore_az, abs=0.01 / 3600.0)
        assert bore_el_back == pytest.approx(el, abs=0.01 / 3600.0)


class TestOffsetKnownGeometry:
    """Tests with known geometric relationships."""

    def test_90_degree_rotation_swaps_axes(self):
        """Test that 90 degree field rotation swaps x and y offsets."""
        offset = InstrumentOffset(dx=60.0, dy=0.0)
        az, el = 180.0, 0.0  # At horizon, cos(el)=1

        det_az, det_el = boresight_to_detector(az, el, offset, field_rotation=90.0)

        assert det_az == pytest.approx(az, abs=1e-10)
        assert det_el == pytest.approx(el + 1.0, rel=1e-6)

    def test_90_degree_rotation_with_y_offset(self):
        """Test 90 degree rotation with y offset becomes negative x."""
        offset = InstrumentOffset(dx=0.0, dy=60.0)
        az, el = 180.0, 0.0

        det_az, det_el = boresight_to_detector(az, el, offset, field_rotation=90.0)

        assert det_az == pytest.approx(az - 1.0, rel=1e-6)
        assert det_el == pytest.approx(el, abs=1e-6)

    def test_180_degree_rotation_inverts_offsets(self):
        """Test that 180 degree rotation inverts the offset direction."""
        offset = InstrumentOffset(dx=60.0, dy=30.0)
        az, el = 180.0, 0.0

        det_az_0, det_el_0 = boresight_to_detector(az, el, offset, field_rotation=0.0)
        det_az_180, det_el_180 = boresight_to_detector(az, el, offset, field_rotation=180.0)

        az_offset_0 = det_az_0 - az
        az_offset_180 = det_az_180 - az
        el_offset_0 = det_el_0 - el
        el_offset_180 = det_el_180 - el

        assert az_offset_180 == pytest.approx(-az_offset_0, rel=1e-4)
        assert el_offset_180 == pytest.approx(-el_offset_0, rel=1e-4)

    def test_pure_elevation_offset_is_exact(self):
        """Test that pure elevation offset adds directly to elevation."""
        offset = InstrumentOffset(dx=0.0, dy=60.0)
        az, el = 180.0, 45.0

        det_az, det_el = boresight_to_detector(az, el, offset, field_rotation=0.0)

        assert det_az == pytest.approx(az, abs=1e-10)
        assert det_el == pytest.approx(el + 1.0, abs=1e-10)

    def test_offset_direction_with_zero_field_rotation(self):
        """Test that positive dx increases azimuth with zero field rotation."""
        offset = InstrumentOffset(dx=30.0, dy=0.0)
        az, el = 180.0, 0.0

        det_az, _det_el = boresight_to_detector(az, el, offset, field_rotation=0.0)

        assert det_az > az

    def test_offset_direction_with_zero_field_rotation_y(self):
        """Test that positive dy increases elevation with zero field rotation."""
        offset = InstrumentOffset(dx=0.0, dy=30.0)
        az, el = 180.0, 45.0

        _det_az, det_el = boresight_to_detector(az, el, offset, field_rotation=0.0)

        assert det_el > el


class TestFieldRotationEffects:
    """Tests verifying that offset direction rotates with parallactic angle."""

    def test_offset_rotates_continuously(self):
        """Test that offset direction rotates smoothly with field rotation."""
        offset = InstrumentOffset(dx=60.0, dy=0.0)
        az, el = 180.0, 45.0

        results = []
        for fr in np.linspace(0, 360, 13)[:-1]:
            det_az, det_el = boresight_to_detector(az, el, offset, field_rotation=fr)
            results.append((fr, det_az - az, det_el - el))

        # Verify the offset magnitude is approximately constant
        # (on the sphere, it won't be exactly constant in projected coords,
        # but the angular separation should be constant)
        magnitudes = []
        for _, daz, de in results:
            # Approximate angular distance
            cos_el = np.cos(np.deg2rad(el))
            mag = np.sqrt((daz * cos_el) ** 2 + de**2)
            magnitudes.append(mag)

        np.testing.assert_allclose(magnitudes, magnitudes[0], rtol=5e-3)

    def test_field_rotation_period_360(self):
        """Test that field rotation has 360 degree period."""
        offset = InstrumentOffset(dx=30.0, dy=20.0)
        az, el = 180.0, 45.0

        det_az_0, det_el_0 = boresight_to_detector(az, el, offset, field_rotation=0.0)
        det_az_360, det_el_360 = boresight_to_detector(az, el, offset, field_rotation=360.0)

        assert det_az_360 == pytest.approx(det_az_0, rel=1e-10)
        assert det_el_360 == pytest.approx(det_el_0, rel=1e-10)

    def test_negative_field_rotation(self):
        """Test that negative field rotation is handled correctly."""
        offset = InstrumentOffset(dx=30.0, dy=20.0)
        az, el = 180.0, 45.0

        det_az_neg, det_el_neg = boresight_to_detector(az, el, offset, field_rotation=-45.0)
        det_az_pos, det_el_pos = boresight_to_detector(az, el, offset, field_rotation=315.0)

        assert det_az_pos == pytest.approx(det_az_neg, rel=1e-10)
        assert det_el_pos == pytest.approx(det_el_neg, rel=1e-10)

    def test_vectorized_round_trip(self):
        """Test that spherical method works with numpy arrays."""
        offset = InstrumentOffset(dx=60.0, dy=30.0)
        az = np.array([100.0, 150.0, 200.0, 250.0])
        el = np.array([25.0, 35.0, 45.0, 55.0])
        fr = np.array([0.0, 30.0, 60.0, 90.0])

        det_az, det_el = boresight_to_detector(az, el, offset, field_rotation=fr)

        assert isinstance(det_az, np.ndarray)
        assert isinstance(det_el, np.ndarray)
        assert len(det_az) == 4
        assert len(det_el) == 4
        assert np.all(np.isfinite(det_az))
        assert np.all(np.isfinite(det_el))

        # Round-trip
        bore_az, bore_el = detector_to_boresight(det_az, det_el, offset, field_rotation=fr)
        np.testing.assert_allclose(bore_az, az, atol=0.01 / 3600.0)
        np.testing.assert_allclose(bore_el, el, atol=0.01 / 3600.0)


class TestComputeFocalPlaneRotation:
    """Tests for compute_focal_plane_rotation helper."""

    def test_right_nasmyth_positive(self, site):
        """Test that right Nasmyth gives positive sign on elevation."""
        offset = InstrumentOffset(dx=0.0, dy=0.0)
        rot = compute_focal_plane_rotation(45.0, site, offset)
        # site.nasmyth_sign = +1, so rotation = +1 * 45 + 0 + 0 = 45
        assert rot == pytest.approx(45.0)

    def test_with_parallactic_angle(self, site):
        """Test that parallactic angle is added correctly."""
        offset = InstrumentOffset(dx=0.0, dy=0.0)
        rot = compute_focal_plane_rotation(45.0, site, offset, parallactic_angle=10.0)
        assert rot == pytest.approx(55.0)

    def test_with_instrument_rotation(self, site):
        """Test that instrument_rotation is included."""
        offset = InstrumentOffset(dx=0.0, dy=0.0, instrument_rotation=15.0)
        rot = compute_focal_plane_rotation(45.0, site, offset)
        # +1 * 45 + 15 + 0 = 60
        assert rot == pytest.approx(60.0)

    def test_array_input(self, site):
        """Test with array elevation input."""
        offset = InstrumentOffset(dx=0.0, dy=0.0)
        el = np.array([30.0, 45.0, 60.0])
        rot = compute_focal_plane_rotation(el, site, offset)
        np.testing.assert_allclose(rot, el)


class TestApplyDetectorOffsetFieldRotation:
    """Tests for the field rotation decomposition in apply_detector_offset."""

    def test_altaz_trajectory_nonzero_rotation(self, site):
        """Test that AltAz trajectory (no RA/Dec) uses mechanical rotation."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")

        # ConstantEl has no RA/Dec metadata
        trajectory = (
            TrajectoryBuilder(site)
            .with_config(
                ConstantElScanConfig(
                    timestep=0.1,
                    az_start=120.0,
                    az_stop=180.0,
                    elevation=45.0,
                    az_speed=1.0,
                    az_accel=0.5,
                )
            )
            .duration(60.0)
            .starting_at(start_time)
            .build()
        )

        assert trajectory.center_ra is None
        assert trajectory.center_dec is None

        # Use an asymmetric offset so the rotation effect is visible in both axes
        offset = InstrumentOffset(dx=60.0, dy=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adjusted = apply_detector_offset(trajectory, offset, site)

        # With mechanical rotation = +1 * 45 = 45 degrees, the dx=1 degree
        # offset is rotated into both az and el components.
        # Verify positions changed (offset is applied with non-zero rotation)
        assert not np.allclose(adjusted.az, trajectory.az)
        assert not np.allclose(adjusted.el, trajectory.el)

    def test_altaz_trajectory_no_warning(self, site):
        """AltAz trajectories must not warn: mechanical-only IS the model.

        The pre-fix code warned that the parallactic angle was unavailable;
        with the pa-in-horizon-frame fix the mechanical rotation is the
        correct and complete rotation for every az/el projection, so there
        is nothing to warn about.
        """
        start_time = Time("2026-03-15T04:00:00", scale="utc")

        trajectory = (
            TrajectoryBuilder(site)
            .with_config(
                ConstantElScanConfig(
                    timestep=0.1,
                    az_start=120.0,
                    az_stop=180.0,
                    elevation=45.0,
                    az_speed=1.0,
                    az_accel=0.5,
                )
            )
            .duration(60.0)
            .starting_at(start_time)
            .build()
        )

        offset = InstrumentOffset(dx=30.0, dy=30.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            adjusted = apply_detector_offset(trajectory, offset, site)
        assert adjusted.n_points == trajectory.n_points

    def test_left_nasmyth_sign_flip(self):
        """Test that nasmyth_port='left' flips the sign of elevation rotation."""
        right_site = get_fyst_site()

        # Create a left-nasmyth site by loading and modifying config
        left_site = Site(
            name=right_site.name,
            description=right_site.description,
            latitude=right_site.latitude,
            longitude=right_site.longitude,
            elevation=right_site.elevation,
            atmosphere=None,
            telescope_limits=right_site.telescope_limits,
            sun_avoidance=right_site.sun_avoidance,
            nasmyth_port="left",
        )

        start_time = Time("2026-03-15T04:00:00", scale="utc")
        offset = InstrumentOffset(dx=30.0, dy=30.0)

        # Use ConstantEl so we test mechanical rotation only
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=120.0,
            az_stop=180.0,
            elevation=45.0,
            az_speed=1.0,
            az_accel=0.5,
        )

        traj_right = (
            TrajectoryBuilder(right_site)
            .with_config(config)
            .duration(60.0)
            .starting_at(start_time)
            .build()
        )

        traj_left = (
            TrajectoryBuilder(left_site)
            .with_config(config)
            .duration(60.0)
            .starting_at(start_time)
            .build()
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adj_right = apply_detector_offset(traj_right, offset, right_site)
            adj_left = apply_detector_offset(traj_left, offset, left_site)

        # The offsets should differ because the sign of el in rotation is flipped
        assert not np.allclose(adj_right.az, adj_left.az)

    def test_nonzero_instrument_rotation(self, site):
        """Test that instrument_rotation affects the trajectory."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")

        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=120.0,
            az_stop=180.0,
            elevation=45.0,
            az_speed=1.0,
            az_accel=0.5,
        )

        trajectory = (
            TrajectoryBuilder(site)
            .with_config(config)
            .duration(60.0)
            .starting_at(start_time)
            .build()
        )

        offset_no_rot = InstrumentOffset(dx=30.0, dy=30.0, instrument_rotation=0.0)
        offset_with_rot = InstrumentOffset(dx=30.0, dy=30.0, instrument_rotation=15.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adj_no_rot = apply_detector_offset(trajectory, offset_no_rot, site)
            adj_with_rot = apply_detector_offset(trajectory, offset_with_rot, site)

        # Different instrument_rotation should produce different trajectories
        assert not np.allclose(adj_no_rot.az, adj_with_rot.az)

    def test_celestial_metadata_does_not_change_projection(self, site):
        """Same az/el in, same az/el out: celestial metadata is irrelevant.

        Frame-invariance regression for the pa-in-horizon-frame fix: the
        focal-plane-to-az/el projection depends only on (az, el, offset,
        mechanical rotation). Two trajectories with identical az/el paths
        must produce identical boresights whether or not ``center_ra`` /
        ``center_dec`` metadata is present. The pre-fix code added the
        parallactic angle when RA/Dec was available, making the two paths
        diverge by degrees for an off-axis module.
        """
        start_time = Time("2026-03-15T04:00:00", scale="utc")

        celestial = (
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
        assert celestial.center_ra is not None

        # Identical az/el path, but no celestial metadata.
        bare = Trajectory(
            times=celestial.times.copy(),
            az=celestial.az.copy(),
            el=celestial.el.copy(),
            az_vel=celestial.az_vel.copy(),
            el_vel=celestial.el_vel.copy(),
            start_time=celestial.start_time,
        )
        assert bare.center_ra is None

        offset = InstrumentOffset(dx=30.0, dy=30.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # neither path may warn
            adj_celestial = apply_detector_offset(celestial, offset, site)
            adj_bare = apply_detector_offset(bare, offset, site)

        np.testing.assert_allclose(adj_celestial.az, adj_bare.az, atol=1e-12)
        np.testing.assert_allclose(adj_celestial.el, adj_bare.el, atol=1e-12)

    def test_unobservable_target_raises(self, site):
        """Test catching TargetNotObservableError for invalid celestial target."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")

        # Dec=+80 is never visible from FYST (latitude -22.96)
        with pytest.raises(TargetNotObservableError):
            TrajectoryBuilder(site).at(ra=180.0, dec=80.0).with_config(
                PongScanConfig(
                    timestep=0.1,
                    width=2.0,
                    height=2.0,
                    spacing=0.1,
                    velocity=0.5,
                    num_terms=4,
                    angle=0.0,
                )
            ).duration(300.0).starting_at(start_time).build()


class TestInstrumentRotationRepr:
    """Tests for InstrumentOffset repr with instrument_rotation."""

    def test_repr_without_instrument_rotation(self):
        """Test repr when instrument_rotation is default (0.0)."""
        offset = InstrumentOffset(dx=5.0, dy=3.0, name="Test")
        r = repr(offset)
        assert "instrument_rotation" not in r
        assert "dx=5.0'" in r
        assert "dy=3.0'" in r
        assert "name='Test'" in r

    def test_repr_with_instrument_rotation(self):
        """Test repr when instrument_rotation is non-zero."""
        offset = InstrumentOffset(dx=5.0, dy=3.0, instrument_rotation=15.0)
        r = repr(offset)
        assert "instrument_rotation=15.0" in r


class TestComputeFocalPlaneRotationExtended:
    """Extended tests for compute_focal_plane_rotation."""

    def test_cassegrain_elevation_does_not_contribute(self):
        """Test that cassegrain (nasmyth_sign=0) ignores elevation."""
        cass_site = Site(
            name="Test",
            description="",
            latitude=-23.0,
            longitude=-67.0,
            elevation=5000.0,
            atmosphere=None,
            telescope_limits=TelescopeLimits(
                azimuth=AxisLimits(
                    min=-270,
                    max=270,
                    max_velocity=3,
                    max_acceleration=1,
                ),
                elevation=AxisLimits(
                    min=20,
                    max=90,
                    max_velocity=1,
                    max_acceleration=0.5,
                ),
            ),
            sun_avoidance=SunAvoidanceConfig(
                enabled=True,
                exclusion_radius=45,
                warning_radius=50,
            ),
            nasmyth_port="cassegrain",
        )
        assert cass_site.nasmyth_sign == 0

        offset = InstrumentOffset(dx=5.0, dy=3.0)
        # At various elevations, rotation should be the same (0*el + 0 + 0 = 0)
        rot_30 = compute_focal_plane_rotation(30.0, cass_site, offset)
        rot_60 = compute_focal_plane_rotation(60.0, cass_site, offset)
        rot_85 = compute_focal_plane_rotation(85.0, cass_site, offset)

        assert rot_30 == pytest.approx(0.0)
        assert rot_60 == pytest.approx(0.0)
        assert rot_85 == pytest.approx(0.0)

    def test_cassegrain_with_parallactic_angle(self):
        """Test cassegrain with parallactic angle (elevation still ignored)."""
        cass_site = Site(
            name="Test",
            description="",
            latitude=-23.0,
            longitude=-67.0,
            elevation=5000.0,
            atmosphere=None,
            telescope_limits=TelescopeLimits(
                azimuth=AxisLimits(
                    min=-270,
                    max=270,
                    max_velocity=3,
                    max_acceleration=1,
                ),
                elevation=AxisLimits(
                    min=20,
                    max=90,
                    max_velocity=1,
                    max_acceleration=0.5,
                ),
            ),
            sun_avoidance=SunAvoidanceConfig(
                enabled=True,
                exclusion_radius=45,
                warning_radius=50,
            ),
            nasmyth_port="cassegrain",
        )
        offset = InstrumentOffset(dx=5.0, dy=3.0)
        rot = compute_focal_plane_rotation(
            45.0,
            cass_site,
            offset,
            parallactic_angle=25.0,
        )
        # 0 * 45 + 0 + 25 = 25
        assert rot == pytest.approx(25.0)

    def test_all_three_components(self, site):
        """Test combining nasmyth_sign * el + instrument_rotation + pa."""
        # site is FYST with nasmyth_sign = +1
        offset = InstrumentOffset(dx=5.0, dy=3.0, instrument_rotation=15.0)
        el = 45.0
        pa = 20.0

        rot = compute_focal_plane_rotation(el, site, offset, parallactic_angle=pa)
        # +1 * 45 + 15 + 20 = 80
        assert rot == pytest.approx(80.0)

    def test_left_nasmyth_all_components(self):
        """Test left nasmyth with all three components."""
        left_site = Site(
            name="Test",
            description="",
            latitude=-23.0,
            longitude=-67.0,
            elevation=5000.0,
            atmosphere=None,
            telescope_limits=TelescopeLimits(
                azimuth=AxisLimits(
                    min=-270,
                    max=270,
                    max_velocity=3,
                    max_acceleration=1,
                ),
                elevation=AxisLimits(
                    min=20,
                    max=90,
                    max_velocity=1,
                    max_acceleration=0.5,
                ),
            ),
            sun_avoidance=SunAvoidanceConfig(
                enabled=True,
                exclusion_radius=45,
                warning_radius=50,
            ),
            nasmyth_port="left",
        )
        offset = InstrumentOffset(dx=5.0, dy=3.0, instrument_rotation=10.0)
        rot = compute_focal_plane_rotation(
            45.0,
            left_site,
            offset,
            parallactic_angle=20.0,
        )
        # -1 * 45 + 10 + 20 = -15
        assert rot == pytest.approx(-15.0)


class TestEarlyExitZeroOffset:
    """Tests for the early-exit optimization with zero offsets."""

    def test_zero_offset_returns_same_object(self, site):
        """Test that zero offset returns the exact same trajectory object."""
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

        offset = InstrumentOffset(dx=0.0, dy=0.0)
        result = apply_detector_offset(trajectory, offset, site)

        # Early exit should return a copy to avoid aliasing mutable arrays
        assert result is not trajectory
        np.testing.assert_array_equal(result.az, trajectory.az)
        np.testing.assert_array_equal(result.el, trajectory.el)

    def test_zero_offset_with_instrument_rotation_not_early_exit(self, site):
        """Test that zero dx/dy but non-zero instrument_rotation is NOT skipped."""
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

        # dx=dy=0 but instrument_rotation != 0, should NOT early-exit
        offset = InstrumentOffset(dx=0.0, dy=0.0, instrument_rotation=15.0)
        result = apply_detector_offset(trajectory, offset, site)

        # Should be a different object (new trajectory was computed)
        assert result is not trajectory

    def test_nonzero_offset_not_early_exit(self, site):
        """Test that non-zero offset does NOT early-exit."""
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

        offset = InstrumentOffset(dx=30.0, dy=30.0)
        result = apply_detector_offset(trajectory, offset, site)

        assert result is not trajectory


class TestFromFocalPlane:
    """Tests for InstrumentOffset.from_focal_plane factory method."""

    def test_basic_conversion(self):
        """Test basic conversion from mm to arcmin."""
        offset = InstrumentOffset.from_focal_plane(x_mm=0.0, y_mm=-461.3, plate_scale=13.89)
        # 461.3 mm * 13.89 arcsec/mm / 60 = 106.79 arcmin
        assert offset.dx == pytest.approx(0.0, abs=1e-10)
        assert offset.dy == pytest.approx(-106.79, abs=0.01)

    def test_name_passed_through(self):
        """Test that name is passed through correctly."""
        offset = InstrumentOffset.from_focal_plane(
            x_mm=0.0, y_mm=-461.3, plate_scale=13.89, name="TestModule"
        )
        assert offset.name == "TestModule"

    def test_instrument_rotation_passed_through(self):
        """Test that instrument_rotation is passed through correctly."""
        offset = InstrumentOffset.from_focal_plane(
            x_mm=0.0, y_mm=-461.3, plate_scale=13.89, instrument_rotation=15.0
        )
        assert offset.instrument_rotation == pytest.approx(15.0)

    def test_zero_position_returns_zero_offset(self):
        """Test that (0, 0) position produces zero offset."""
        offset = InstrumentOffset.from_focal_plane(x_mm=0.0, y_mm=0.0, plate_scale=13.89)
        assert offset.dx == pytest.approx(0.0, abs=1e-10)
        assert offset.dy == pytest.approx(0.0, abs=1e-10)

    def test_consistency_with_manual_calculation(self):
        """Test that from_focal_plane matches manual calculation."""
        x_mm, y_mm, plate_scale = 100.0, 200.0, 13.89

        # Manual calculation
        dx_arcmin_manual = x_mm * plate_scale / 60.0
        dy_arcmin_manual = y_mm * plate_scale / 60.0

        # Via factory
        offset = InstrumentOffset.from_focal_plane(x_mm=x_mm, y_mm=y_mm, plate_scale=plate_scale)

        assert offset.dx == pytest.approx(dx_arcmin_manual, abs=1e-10)
        assert offset.dy == pytest.approx(dy_arcmin_manual, abs=1e-10)

    def test_symmetric_positions(self):
        """Test that symmetric positions produce expected symmetric offsets."""
        plate_scale = 13.89

        offset_pos = InstrumentOffset.from_focal_plane(
            x_mm=100.0, y_mm=100.0, plate_scale=plate_scale
        )
        offset_neg = InstrumentOffset.from_focal_plane(
            x_mm=-100.0, y_mm=-100.0, plate_scale=plate_scale
        )

        assert offset_neg.dx == pytest.approx(-offset_pos.dx, abs=1e-10)
        assert offset_neg.dy == pytest.approx(-offset_pos.dy, abs=1e-10)


class TestPrimeCamFromFocalPlane:
    """Tests verifying PRIMECAM_MODULES values from from_focal_plane."""

    def test_inner_ring_distance_matches_expected(self):
        """Test that inner ring modules are at expected angular distance."""
        plate_scale = get_fyst_site().plate_scale
        # Expected distance: 461.3 mm * 13.89 arcsec/mm / 60 = 106.79 arcmin
        expected_distance = 461.3 * plate_scale / 60.0

        inner_ring_offsets = [
            PRIMECAM_MODULES["i1"],
            PRIMECAM_MODULES["i2"],
            PRIMECAM_MODULES["i3"],
            PRIMECAM_MODULES["i4"],
            PRIMECAM_MODULES["i5"],
            PRIMECAM_MODULES["i6"],
        ]

        for offset in inner_ring_offsets:
            distance = np.sqrt(offset.dx**2 + offset.dy**2)
            assert distance == pytest.approx(expected_distance, rel=1e-6)

    def test_conversion_consistent_with_plate_scale(self):
        """Test that PRIMECAM modules use site plate_scale correctly."""
        plate_scale = get_fyst_site().plate_scale
        # I1 is at (0, -461.3) mm
        i1 = PRIMECAM_MODULES["i1"]
        expected_dy = -461.3 * plate_scale / 60.0

        assert i1.dx == pytest.approx(0.0, abs=1e-10)
        assert i1.dy == pytest.approx(expected_dy, rel=1e-6)


class TestComputeFocalPlaneRotationArray:
    """The array-input path of compute_focal_plane_rotation (used by live PCS).

    Every other caller passes scalars; the PCS scan tasks pass per-sample
    arrays, so the broadcasting path had no regression guard.
    """

    def test_array_el_and_pa_broadcast_elementwise(self):
        site = get_fyst_site()  # right Nasmyth -> nasmyth_sign = +1
        offset = InstrumentOffset(dx=0.0, dy=0.0, instrument_rotation=10.0)
        el = np.array([20.0, 45.0, 70.0])
        pa = np.array([5.0, -3.0, 12.0])

        rot = compute_focal_plane_rotation(el, site, offset, parallactic_angle=pa)

        assert isinstance(rot, np.ndarray)
        assert rot.shape == (3,)
        np.testing.assert_allclose(rot, site.nasmyth_sign * el + 10.0 + pa)
        # Concrete spot-check: +1*45 + 10 + (-3) = 52.
        assert rot[1] == pytest.approx(52.0)


def _independent_apparent_pa(coords, ra, dec, times):
    """Parallactic angle from an independent apparent-place transform (no lib PA).

    Brings the ICRS centre to the apparent equinox of date (TETE) and forms
    ``HA = LAST - RA_apparent`` before applying the IAU spherical-triangle
    formula. Independent of ``Coordinates.get_parallactic_angle`` so it can be
    used as ground truth for the offset path.
    """
    loc = coords.location
    lat_rad = np.deg2rad(coords.site.latitude)
    app = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").transform_to(
        TETE(obstime=times, location=loc)
    )
    last = times.sidereal_time("apparent", longitude=loc.lon).to_value(u.deg)
    ha = np.deg2rad(((last - app.ra.deg + 180.0) % 360.0) - 180.0)
    dr = np.deg2rad(app.dec.deg)
    return np.rad2deg(
        np.arctan2(np.sin(ha), np.tan(lat_rad) * np.cos(dr) - np.sin(dr) * np.cos(ha))
    )


class TestOffsetPathLandsOnTarget:
    """The named detector observes the original target after the offset.

    ``apply_detector_offset`` is a horizon-frame (az/el) projection, so it
    must place an off-axis module using the MECHANICAL focal-plane rotation
    only (``nasmyth_sign * el + instrument_rotation``). The ground truth here
    is an *independent* flat-sky (KOSMA-style) forward projection of the
    rotated offset from the library's adjusted boresight, plain numpy, no
    library projection functions. Peer references for the pa-free az/el
    projection: SO ``make_source_ces`` (static rotation only), NIKA2
    A&A 637 A71 Sec. 5.1 Eq. 2 (elevation-only Nasmyth-to-altaz matrix), and the
    KOSMA focal-plane model (``+/-el`` only; the parallactic angle lives in a
    separate celestial pipeline stage).

    The companion test asserts the PRE-FIX model (mechanical + parallactic
    angle) misses the target by degrees at this geometry, proving the oracle
    discriminates between the two frame models rather than passing vacuously.
    """

    @staticmethod
    def _build_trajectory(site, start_time):
        # ra/dec 180/-30 at 09:00 UTC -> el ~ 36 deg (setting), where the
        # flat-sky truncation error is small and |pa| is large (~100 deg).
        return (
            TrajectoryBuilder(site)
            .at(ra=180.0, dec=-30.0)
            .with_config(
                PongScanConfig(
                    timestep=0.5,
                    width=0.5,
                    height=0.5,
                    spacing=0.1,
                    velocity=0.3,
                    num_terms=4,
                    angle=0.0,
                )
            )
            .duration(20.0)
            .starting_at(start_time)
            .build()
        )

    @staticmethod
    def _flat_project(bore_az, bore_el, offset, rho_deg):
        """Independent KOSMA-style flat-sky projection of a rotated offset."""
        rho = np.radians(rho_deg)
        dxr = offset.dx_deg * np.cos(rho) - offset.dy_deg * np.sin(rho)
        dyr = offset.dx_deg * np.sin(rho) + offset.dy_deg * np.cos(rho)
        det_el = bore_el + dyr
        det_az = bore_az + dxr / np.cos(np.radians(bore_el + dyr / 2.0))
        return det_az, det_el

    def test_inner_ring_module_lands_on_target(self):
        site = get_fyst_site()
        start_time = Time("2026-03-15T09:00:00", scale="utc")
        trajectory = self._build_trajectory(site, start_time)

        # PRIMECAM_I1 is an inner-ring module ~106.8 arcmin (1.78 deg) off-axis.
        boresight = apply_detector_offset(trajectory, PRIMECAM_I1, site)

        # Independent mechanical (horizon-frame) rotation. Evaluated at the
        # input (detector) elevation, matching the library's documented
        # convention.
        rho_mech = site.nasmyth_sign * trajectory.el + PRIMECAM_I1.instrument_rotation
        actual_az, actual_el = self._flat_project(boresight.az, boresight.el, PRIMECAM_I1, rho_mech)

        target = SkyCoord(trajectory.az * u.deg, trajectory.el * u.deg, frame="altaz")
        actual = SkyCoord(actual_az * u.deg, actual_el * u.deg, frame="altaz")
        miss_deg = target.separation(actual).to_value(u.deg)

        # Flat-vs-spherical truncation at rho ~ 1.78 deg and el ~ 36 deg is
        # well under this bound; a frame-model error is > 1 deg (companion test).
        assert miss_deg.max() < 0.05, (
            f"inner-ring module misses target by up to {miss_deg.max():.3f} deg; "
            "the az/el projection is not using the mechanical rotation"
        )

    def test_pa_in_horizon_frame_would_miss(self):
        """The pre-fix (mechanical + pa) model misses grossly: the oracle discriminates."""
        site = get_fyst_site()
        coords = Coordinates(site)
        start_time = Time("2026-03-15T09:00:00", scale="utc")
        trajectory = self._build_trajectory(site, start_time)

        boresight = apply_detector_offset(trajectory, PRIMECAM_I1, site)

        abs_times = get_absolute_times(trajectory)
        pa = _independent_apparent_pa(
            coords, trajectory.center_ra, trajectory.center_dec, abs_times
        )
        # Geometry guard: |pa| must be large here or this test is vacuous.
        assert np.abs(pa).min() > 25.0

        rho_wrong = site.nasmyth_sign * trajectory.el + pa
        wrong_az, wrong_el = self._flat_project(boresight.az, boresight.el, PRIMECAM_I1, rho_wrong)

        target = SkyCoord(trajectory.az * u.deg, trajectory.el * u.deg, frame="altaz")
        wrong = SkyCoord(wrong_az * u.deg, wrong_el * u.deg, frame="altaz")
        miss_deg = target.separation(wrong).to_value(u.deg)

        assert miss_deg.min() > 0.5, (
            "pa-rotated projection should miss by degrees; if it lands on "
            "target the oracle no longer discriminates the frame models"
        )


class TestInverseThresholdMagnitudes:
    """Pin the falsifiable magnitudes in the threshold comments.

    The inline labels on the two private thresholds previously read
    "~3.6 microarcsec" / "~3.6 arcsec"; both were off by 1000x. The values
    are deg, so deg*3600 = arcsec. Pin the true magnitudes so the comments
    cannot drift unnoticed again.
    """

    def test_early_exit_threshold_is_nanoarcsec(self):
        # 1e-12 deg * 3600 = 3.6e-9 arcsec = 3.6 nanoarcsec.
        arcsec = _INVERSE_EARLY_EXIT_THRESHOLD * 3600.0
        assert arcsec == pytest.approx(3.6e-9, rel=1e-9)

    def test_failure_threshold_is_milliarcsec(self):
        # 1e-6 deg * 3600 = 3.6e-3 arcsec = 3.6 milliarcsec.
        arcsec = _INVERSE_FAILURE_THRESHOLD * 3600.0
        assert arcsec == pytest.approx(3.6e-3, rel=1e-9)


class TestInverseZenithDegeneracy:
    """The inverse must not silently return a wrong azimuth at the pole.

    At the zenith pole, azimuth is degenerate: every boresight azimuth maps a
    pole-elevation detector to the same position, so the forward-residual
    convergence check reports success while the recovered azimuth is arbitrary.
    The pole guard raises a clear RuntimeError instead.
    """

    def test_zenith_offset_raises_instead_of_wrong_azimuth(self):
        # bore=(180, 89), dx=60', fr=90 lands the detector at el=90 (the pole).
        offset = InstrumentOffset(dx=60.0, dy=0.0)
        det_az, det_el = boresight_to_detector(180.0, 89.0, offset, field_rotation=90.0)
        assert det_el == pytest.approx(90.0, abs=1e-3)

        with pytest.raises(RuntimeError, match="azimuth"):
            detector_to_boresight(det_az, det_el, offset, field_rotation=90.0)

    def test_operational_envelope_still_round_trips(self):
        """The pole guard must not fire inside the real PrimeCam envelope."""
        offset = InstrumentOffset(dx=106.8, dy=0.0)  # inner ring ~1.78 deg
        for el in [20.0, 45.0, 70.0, 85.0]:
            for fr in [0.0, 90.0, 180.0, 270.0]:
                det_az, det_el = boresight_to_detector(200.0, el, offset, field_rotation=fr)
                bore_az, bore_el = detector_to_boresight(det_az, det_el, offset, field_rotation=fr)
                assert bore_az == pytest.approx(200.0, abs=0.01 / 3600.0)
                assert bore_el == pytest.approx(el, abs=0.01 / 3600.0)


class TestApplyDetectorOffsetSingleSample:
    """A length-1 trajectory must not raise an opaque IndexError."""

    def test_single_sample_trajectory_zero_velocities(self, site):
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        trajectory = Trajectory(
            times=np.array([0.0]),
            az=np.array([180.0]),
            el=np.array([45.0]),
            az_vel=np.array([0.0]),
            el_vel=np.array([0.0]),
            start_time=start_time,
        )

        offset = InstrumentOffset(dx=30.0, dy=30.0)
        adjusted = apply_detector_offset(trajectory, offset, site)

        assert adjusted.n_points == 1
        # Boresight velocity is undefined for a single sample -> zeros, not a crash.
        assert adjusted.az_vel[0] == 0.0
        assert adjusted.el_vel[0] == 0.0


class TestApplyDetectorOffsetRetuneEvents:
    """The offset must preserve retune_events alongside scan_flag==3."""

    def test_retune_events_preserved_after_offset(self, site):
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        trajectory = (
            TrajectoryBuilder(site)
            .at(ra=180.0, dec=-30.0)
            .with_config(
                PongScanConfig(
                    timestep=0.5,
                    width=1.0,
                    height=1.0,
                    spacing=0.1,
                    velocity=0.3,
                    num_terms=4,
                    angle=0.0,
                )
            )
            .duration(120.0)
            .starting_at(start_time)
            .build()
        )

        retuned = inject_retune(
            trajectory,
            retune_events=[RetuneEvent(t_start=10.0, duration=2.0), RetuneEvent(40.0, 2.0)],
        )
        assert len(retuned.retune_events) == 2
        n_retune_samples = int(np.sum(retuned.scan_flag == 3))
        assert n_retune_samples > 0

        offset = InstrumentOffset(dx=30.0, dy=30.0)
        adjusted = apply_detector_offset(retuned, offset, site)

        # Both the per-sample flags and the event-level provenance must survive.
        assert len(adjusted.retune_events) == len(retuned.retune_events)
        assert int(np.sum(adjusted.scan_flag == 3)) == n_retune_samples


class TestApplyDetectorOffsetFrameConsistency:
    """Frame-varying regression for a refracted-el input.

    Decision (after empirical measurement): keep the physically-correct
    per-sample ``trajectory.el`` for the mechanical term: substituting a
    single center-vacuum-el would regress the vacuum/live path by ~30-200"
    for extended patterns, far more than the residual leak it would remove.
    Since the pa-in-horizon-frame fix the only frame leak left is the
    mechanical term itself: a ``for_fyst()`` (refracted) input evaluates
    ``nasmyth_sign * el`` at the apparent elevation, differing from vacuum
    by ``nasmyth_sign * (refraction bump)``, a sub-arcsec boresight effect
    at PrimeCam offset radii. This test documents and bounds that leak; the
    vacuum path remains the reference.
    """

    def _build(self, site, start_time, atmosphere):
        builder = TrajectoryBuilder(site).at(ra=180.0, dec=-30.0)
        if atmosphere is not None:
            builder = builder.with_atmosphere(atmosphere)
        return (
            builder.with_config(
                PongScanConfig(
                    timestep=0.5,
                    width=1.0,
                    height=1.0,
                    spacing=0.1,
                    velocity=0.3,
                    num_terms=4,
                    angle=0.0,
                )
            )
            .duration(60.0)
            .starting_at(start_time)
            .build()
        )

    def test_refracted_input_leak_is_bounded_arcsec(self, site):
        # el ~ 36 deg at this epoch, near the worst case for the leak.
        start_time = Time("2026-03-15T09:00:00", scale="utc")
        offset = PRIMECAM_I1  # inner ring rho ~ 1.78 deg

        traj_vac = self._build(site, start_time, atmosphere=None)
        traj_ref = self._build(site, start_time, atmosphere=AtmosphericConditions.for_fyst())

        adj_vac = apply_detector_offset(traj_vac, offset, site)
        adj_ref = apply_detector_offset(traj_ref, offset, site)

        # The refracted az/el differ from vacuum by the refraction bump itself
        # (tens of arcsec in el); to isolate the *rotation* leak we compare the
        # boresight the offset produces for each, removing the input el offset by
        # comparing the detector->boresight *shift* (boresight - input position).
        shift_vac_az = adj_vac.az - traj_vac.az
        shift_vac_el = adj_vac.el - traj_vac.el
        shift_ref_az = adj_ref.az - traj_ref.az
        shift_ref_el = adj_ref.el - traj_ref.el

        # The offset-induced boresight shift should be nearly identical between
        # the vacuum and refracted inputs; the only difference is the mechanical
        # rotation evaluated at apparent vs vacuum el, bounded to a few arcsec.
        d_az = (shift_ref_az - shift_vac_az) * np.cos(np.radians(traj_vac.el))
        d_el = shift_ref_el - shift_vac_el
        leak_arcsec = np.hypot(d_az, d_el) * 3600.0

        assert leak_arcsec.max() < 5.0, (
            f'refracted-input frame leak {leak_arcsec.max():.2f}" exceeds the '
            "documented ~arcsec bound"
        )

    def test_vacuum_path_lands_on_target(self, site):
        """The vacuum (live) path is the reference: the module lands on target."""
        start_time = Time("2026-03-15T09:00:00", scale="utc")
        offset = PRIMECAM_I1

        traj_vac = self._build(site, start_time, atmosphere=None)
        boresight = apply_detector_offset(traj_vac, offset, site)

        # Mechanical (horizon-frame) rotation, forward/inverse consistency;
        # the independent frame-model oracle lives in TestOffsetPathLandsOnTarget.
        phi = compute_focal_plane_rotation(traj_vac.el, site, offset)
        actual_az, actual_el = boresight_to_detector(boresight.az, boresight.el, offset, phi)

        target = SkyCoord(traj_vac.az * u.deg, traj_vac.el * u.deg, frame="altaz")
        actual = SkyCoord(actual_az * u.deg, actual_el * u.deg, frame="altaz")
        miss_arcsec = target.separation(actual).to_value(u.arcsec)
        assert miss_arcsec.max() < 0.01
