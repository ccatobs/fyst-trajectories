"""Tests for instrument offset functionality."""

import warnings

import numpy as np
import pytest
from astropy.time import Time

from fyst_pointing.exceptions import PointingWarning, TargetNotObservableError
from fyst_pointing.offsets import (
    InstrumentOffset,
    apply_detector_offset,
    boresight_to_detector,
    compute_focal_plane_rotation,
    detector_to_boresight,
)
from fyst_pointing.patterns import (
    ConstantElScanConfig,
    PongScanConfig,
    TrajectoryBuilder,
)
from fyst_pointing.primecam import (
    PRIMECAM_CENTER,
    PRIMECAM_I1,
    PRIMECAM_MODULES,
    get_primecam_offset,
)
from fyst_pointing.site import (
    AxisLimits,
    Site,
    SunAvoidanceConfig,
    TelescopeLimits,
    get_fyst_site,
)
from fyst_pointing.trajectory import Trajectory


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

    def test_requires_start_time(self, site):
        """Test that start_time is required."""
        offset = InstrumentOffset(dx=5.0, dy=3.0)
        trajectory = Trajectory(
            times=np.array([0, 1, 2]),
            az=np.array([180.0, 181.0, 182.0]),
            el=np.array([45.0, 45.0, 45.0]),
            az_vel=np.array([1.0, 1.0, 1.0]),
            el_vel=np.array([0.0, 0.0, 0.0]),
            start_time=None,  # No start time
        )

        with pytest.raises(ValueError, match="start_time"):
            apply_detector_offset(trajectory, offset, site)

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
    """Tests for predefined PrimeCam offsets."""

    def test_center_is_zero(self):
        """Test that center module has zero offset."""
        assert PRIMECAM_CENTER.dx == 0.0
        assert PRIMECAM_CENTER.dy == 0.0

    def test_i1_offset_direction(self):
        """Test I1 module is in correct direction."""
        # I1 is at theta=-90, which means dy=-inner_ring_distance
        assert PRIMECAM_I1.dx == pytest.approx(0.0, abs=1e-10)
        assert PRIMECAM_I1.dy < 0  # Negative y direction

    def test_get_primecam_offset(self):
        """Test get_primecam_offset function."""
        offset = get_primecam_offset("c")
        assert offset is PRIMECAM_CENTER

        offset = get_primecam_offset("i1")
        assert offset is PRIMECAM_I1

    def test_get_primecam_offset_case_insensitive(self):
        """Test that module names are case-insensitive."""
        offset_lower = get_primecam_offset("center")
        offset_upper = get_primecam_offset("CENTER")
        offset_mixed = get_primecam_offset("Center")

        assert offset_lower is offset_upper is offset_mixed

    def test_get_primecam_offset_unknown_raises(self):
        """Test that unknown module raises KeyError."""
        with pytest.raises(KeyError, match="Unknown PrimeCam module"):
            get_primecam_offset("nonexistent")

    def test_inner_ring_modules_equidistant(self):
        """Test that all inner ring modules are same distance from center."""
        inner_ring_offsets = [
            PRIMECAM_MODULES["i1"],
            PRIMECAM_MODULES["i2"],
            PRIMECAM_MODULES["i3"],
            PRIMECAM_MODULES["i4"],
            PRIMECAM_MODULES["i5"],
            PRIMECAM_MODULES["i6"],
        ]

        distances = []
        for offset in inner_ring_offsets:
            dist = np.sqrt(offset.dx**2 + offset.dy**2)
            distances.append(dist)

        # All should be same distance
        np.testing.assert_allclose(distances, distances[0], rtol=1e-10)


class TestBuilderForDetector:
    """Tests for TrajectoryBuilder.for_detector() integration."""

    def test_for_detector_in_chain(self, site):
        """Test for_detector in full builder chain."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        offset = InstrumentOffset(dx=30.0, dy=30.0)

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
        assert trajectory.pattern_type == "pong"

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
        """Test using PrimeCam predefined offset."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        offset = get_primecam_offset("i1")

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
                    n_scans=2,
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

    def test_altaz_trajectory_emits_warning(self, site):
        """Test that AltAz trajectory emits a warning about mechanical-only rotation."""
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
                    n_scans=2,
                )
            )
            .duration(60.0)
            .starting_at(start_time)
            .build()
        )

        offset = InstrumentOffset(dx=30.0, dy=30.0)
        with pytest.warns(PointingWarning, match="Parallactic angle unavailable"):
            apply_detector_offset(trajectory, offset, site)

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
            n_scans=2,
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
            n_scans=2,
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

    def test_celestial_trajectory_uses_parallactic_angle(self, site):
        """Test that celestial trajectory includes parallactic angle."""
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

        assert trajectory.center_ra is not None

        offset = InstrumentOffset(dx=30.0, dy=30.0)
        # Should not warn (celestial trajectory has RA/Dec)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            adjusted = apply_detector_offset(trajectory, offset, site)

        assert adjusted.n_points == trajectory.n_points

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

        # dx=dy=0 but instrument_rotation != 0 -- should NOT early-exit
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


class TestPointingWarningEmitted:
    """Tests that PointingWarning (not bare UserWarning) is emitted."""

    def test_altaz_offset_emits_pointing_warning(self, site):
        """Test that AltAz trajectory offset emits PointingWarning specifically."""
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
                    n_scans=2,
                )
            )
            .duration(60.0)
            .starting_at(start_time)
            .build()
        )

        offset = InstrumentOffset(dx=30.0, dy=30.0)

        # Should emit PointingWarning, not bare UserWarning
        with pytest.warns(PointingWarning, match="Parallactic angle unavailable"):
            apply_detector_offset(trajectory, offset, site)


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
