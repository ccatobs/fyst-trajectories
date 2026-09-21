"""Tests for instrument offset functionality."""

import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import TETE, SkyCoord
from astropy.time import Time

from fyst_trajectories.coordinates import Coordinates
from fyst_trajectories.exceptions import (
    OffsetInversionError,
    PointingError,
)
from fyst_trajectories.offsets import (
    _INVERSE_EARLY_EXIT_THRESHOLD,
    _INVERSE_FAILURE_THRESHOLD,
    InstrumentOffset,
    apply_detector_offset,
    boresight_to_detector,
    compute_focal_plane_rotation,
    detector_to_boresight,
    sky_to_focal_plane,
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
    """Forward projection: offset directions, field-rotation angles, and array input."""

    def test_zero_offset_no_change(self):
        offset = InstrumentOffset(dx=0.0, dy=0.0)
        det_az, det_el = boresight_to_detector(180.0, 45.0, offset, field_rotation=0.0)
        assert det_az == pytest.approx(180.0, abs=1e-12)
        assert det_el == pytest.approx(45.0, abs=1e-12)

    def test_x_offset_increases_azimuth(self):
        offset = InstrumentOffset(dx=60.0, dy=0.0)  # 1 degree in arcmin
        det_az, _det_el = boresight_to_detector(180.0, 45.0, offset, field_rotation=0.0)

        assert det_az > 180.0

    def test_y_offset_increases_elevation(self):
        offset = InstrumentOffset(dx=0.0, dy=60.0)  # 1 degree in arcmin
        det_az, det_el = boresight_to_detector(180.0, 45.0, offset, field_rotation=0.0)

        # Pure elevation offset: spherical gives same result
        assert det_az == pytest.approx(180.0, abs=1e-10)
        assert det_el == pytest.approx(46.0, abs=1e-10)

    def test_field_rotation_90_degrees(self):
        """A 90 degree field rotation swaps x and y."""
        offset = InstrumentOffset(dx=60.0, dy=0.0)  # 1 degree x offset

        # With 90 degree rotation, x offset becomes y offset
        det_az_90, det_el_90 = boresight_to_detector(180.0, 45.0, offset, field_rotation=90.0)

        # x offset rotated by 90 deg -> pure elevation offset
        assert det_az_90 == pytest.approx(180.0, abs=1e-6)
        assert det_el_90 == pytest.approx(46.0, rel=1e-4)

    def test_field_rotation_180_degrees(self):
        """A 180 degree field rotation approximately inverts the offset.

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

        # Both components approximately invert: azimuth within 2%, elevation within 4%.
        assert az_diff_180 == pytest.approx(-az_diff_0, rel=0.02)
        assert el_diff_180 == pytest.approx(-el_diff_0, rel=0.04)

    def test_array_input(self):
        offset = InstrumentOffset(dx=30.0, dy=30.0)
        az = np.array([100.0, 150.0, 200.0])
        el = np.array([30.0, 45.0, 60.0])

        det_az, det_el = boresight_to_detector(az, el, offset, field_rotation=0.0)

        assert len(det_az) == 3
        assert len(det_el) == 3
        assert all(det_el > el)  # All elevations should increase

    def test_array_field_rotation(self):
        offset = InstrumentOffset(dx=60.0, dy=0.0)
        field_rotation = np.array([0.0, 90.0, 180.0])

        det_az, det_el = boresight_to_detector(180.0, 45.0, offset, field_rotation=field_rotation)

        assert len(det_az) == 3
        assert len(det_el) == 3


class TestDetectorToBoresight:
    """The inverse recovers the boresight to 0.01 arcsec, with rotation and arrays."""

    def test_zero_offset_no_change(self):
        offset = InstrumentOffset(dx=0.0, dy=0.0)
        bore_az, bore_el = detector_to_boresight(180.0, 45.0, offset, field_rotation=0.0)
        assert bore_az == pytest.approx(180.0, abs=1e-12)
        assert bore_el == pytest.approx(45.0, abs=1e-12)

    def test_inverse_relationship(self):
        """``detector_to_boresight`` inverts ``boresight_to_detector``."""
        offset = InstrumentOffset(dx=30.0, dy=20.0)
        bore_az, bore_el = 180.0, 45.0

        det_az, det_el = boresight_to_detector(bore_az, bore_el, offset, field_rotation=0.0)
        bore_az_recovered, bore_el_recovered = detector_to_boresight(
            det_az, det_el, offset, field_rotation=0.0
        )

        assert bore_az_recovered == pytest.approx(bore_az, abs=0.01 / 3600.0)
        assert bore_el_recovered == pytest.approx(bore_el, abs=0.01 / 3600.0)

    def test_inverse_with_field_rotation(self):
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
        offset = InstrumentOffset(dx=120.0, dy=60.0)  # 2 deg, 1 deg
        bore_az, bore_el = 200.0, 50.0

        det_az, det_el = boresight_to_detector(bore_az, bore_el, offset, field_rotation=0.0)
        bore_az_recovered, bore_el_recovered = detector_to_boresight(
            det_az, det_el, offset, field_rotation=0.0
        )

        assert bore_az_recovered == pytest.approx(bore_az, abs=0.01 / 3600.0)
        assert bore_el_recovered == pytest.approx(bore_el, abs=0.01 / 3600.0)

    def test_array_input_inverse(self):
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
    """Trajectory-level offsets: no timestamps needed, every field carried through."""

    def test_no_start_time_required(self, site):
        """Mechanical (horizon-frame) rotation needs no timestamps.

        ``start_time`` is only ever needed to evaluate the parallactic angle,
        which does not belong in this az/el projection. A trajectory without
        ``start_time`` must be accepted and produce the same boresight as the
        identical trajectory with ``start_time`` set.
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
        """``scan_flag`` survives the zero-offset early-exit path."""
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


class TestBuilderForDetector:
    """``for_detector`` moves the built trajectory by the module's on-sky distance."""

    def test_for_detector_changes_positions(self, site):
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

        trajectory_without = (
            TrajectoryBuilder(site)
            .at(ra=180.0, dec=-30.0)
            .with_config(pong_config)
            .duration(60.0)
            .starting_at(start_time)
            .build()
        )

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
    """Forward then inverse returns the boresight to 0.01 arcsec over the whole grid."""

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
    """Hand-checkable cases: 90 deg swaps the axes, 180 deg inverts, pure el is exact."""

    def test_90_degree_rotation_swaps_axes(self):
        offset = InstrumentOffset(dx=60.0, dy=0.0)
        az, el = 180.0, 0.0  # At horizon, cos(el)=1

        det_az, det_el = boresight_to_detector(az, el, offset, field_rotation=90.0)

        assert det_az == pytest.approx(az, abs=1e-10)
        assert det_el == pytest.approx(el + 1.0, rel=1e-6)

    def test_90_degree_rotation_with_y_offset(self):
        """A 90 degree rotation turns a y offset into a negative x offset."""
        offset = InstrumentOffset(dx=0.0, dy=60.0)
        az, el = 180.0, 0.0

        det_az, det_el = boresight_to_detector(az, el, offset, field_rotation=90.0)

        assert det_az == pytest.approx(az - 1.0, rel=1e-6)
        assert det_el == pytest.approx(el, abs=1e-6)

    def test_180_degree_rotation_inverts_offsets(self):
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
        """A pure elevation offset adds directly to elevation, with no azimuth term."""
        offset = InstrumentOffset(dx=0.0, dy=60.0)
        az, el = 180.0, 45.0

        det_az, det_el = boresight_to_detector(az, el, offset, field_rotation=0.0)

        assert det_az == pytest.approx(az, abs=1e-10)
        assert det_el == pytest.approx(el + 1.0, abs=1e-10)

    def test_offset_direction_with_zero_field_rotation(self):
        offset = InstrumentOffset(dx=30.0, dy=0.0)
        az, el = 180.0, 0.0

        det_az, _det_el = boresight_to_detector(az, el, offset, field_rotation=0.0)

        assert det_az > az

    def test_offset_direction_with_zero_field_rotation_y(self):
        offset = InstrumentOffset(dx=0.0, dy=30.0)
        az, el = 180.0, 45.0

        _det_az, det_el = boresight_to_detector(az, el, offset, field_rotation=0.0)

        assert det_el > el


class TestFieldRotationEffects:
    """The offset sweeps a constant-separation circle, with a 360 deg period."""

    def test_offset_rotates_continuously(self):
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
        offset = InstrumentOffset(dx=30.0, dy=20.0)
        az, el = 180.0, 45.0

        det_az_0, det_el_0 = boresight_to_detector(az, el, offset, field_rotation=0.0)
        det_az_360, det_el_360 = boresight_to_detector(az, el, offset, field_rotation=360.0)

        assert det_az_360 == pytest.approx(det_az_0, rel=1e-10)
        assert det_el_360 == pytest.approx(det_el_0, rel=1e-10)

    def test_negative_field_rotation(self):
        offset = InstrumentOffset(dx=30.0, dy=20.0)
        az, el = 180.0, 45.0

        det_az_neg, det_el_neg = boresight_to_detector(az, el, offset, field_rotation=-45.0)
        det_az_pos, det_el_pos = boresight_to_detector(az, el, offset, field_rotation=315.0)

        assert det_az_pos == pytest.approx(det_az_neg, rel=1e-10)
        assert det_el_pos == pytest.approx(det_el_neg, rel=1e-10)

    def test_vectorized_round_trip(self):
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
    """The rotation is ``nasmyth_sign * el + instrument_rotation + pa``, scalar or array."""

    def test_right_nasmyth_positive(self, site):
        offset = InstrumentOffset(dx=0.0, dy=0.0)
        rot = compute_focal_plane_rotation(45.0, site, offset)
        # site.nasmyth_sign = +1, so rotation = +1 * 45 + 0 + 0 = 45
        assert rot == pytest.approx(45.0)

    def test_with_parallactic_angle(self, site):
        offset = InstrumentOffset(dx=0.0, dy=0.0)
        rot = compute_focal_plane_rotation(45.0, site, offset, parallactic_angle=10.0)
        assert rot == pytest.approx(55.0)

    def test_with_instrument_rotation(self, site):
        offset = InstrumentOffset(dx=0.0, dy=0.0, instrument_rotation=15.0)
        rot = compute_focal_plane_rotation(45.0, site, offset)
        # +1 * 45 + 15 + 0 = 60
        assert rot == pytest.approx(60.0)

    def test_array_input(self, site):
        offset = InstrumentOffset(dx=0.0, dy=0.0)
        el = np.array([30.0, 45.0, 60.0])
        rot = compute_focal_plane_rotation(el, site, offset)
        np.testing.assert_allclose(rot, el)


class TestApplyDetectorOffsetFieldRotation:
    """The trajectory path uses the mechanical rotation only, and never warns about it."""

    def test_altaz_trajectory_nonzero_rotation(self, site):
        """An AltAz trajectory (no RA/Dec) uses the mechanical rotation."""
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
        assert not np.allclose(adjusted.az, trajectory.az)
        assert not np.allclose(adjusted.el, trajectory.el)

    def test_altaz_trajectory_no_warning(self, site):
        """AltAz trajectories must not warn: mechanical-only IS the model.

        The mechanical rotation is the correct and complete rotation for every
        az/el projection, so an unavailable parallactic angle is not something
        to warn about.
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
        """``nasmyth_port='left'`` flips the sign of the elevation rotation."""
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

        Frame-invariance regression: the focal-plane-to-az/el projection
        depends only on (az, el, offset, mechanical rotation). Two trajectories
        with identical az/el paths must produce identical boresights whether or
        not ``center_ra`` / ``center_dec`` metadata is present. Adding the
        parallactic angle when RA/Dec is available would make the two paths
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


class TestInstrumentRotationRepr:
    """``repr`` shows ``instrument_rotation`` only when it is non-zero."""

    def test_repr_without_instrument_rotation(self):
        offset = InstrumentOffset(dx=5.0, dy=3.0, name="Test")
        r = repr(offset)
        assert "instrument_rotation" not in r
        assert "dx=5.0'" in r
        assert "dy=3.0'" in r
        assert "name='Test'" in r

    def test_repr_with_instrument_rotation(self):
        offset = InstrumentOffset(dx=5.0, dy=3.0, instrument_rotation=15.0)
        r = repr(offset)
        assert "instrument_rotation=15.0" in r


class TestComputeFocalPlaneRotationExtended:
    """The Cassegrain and left-Nasmyth sign cases, and all three terms together."""

    def test_cassegrain_elevation_does_not_contribute(self):
        """Cassegrain (``nasmyth_sign=0``) drops the elevation term entirely."""
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
        """Cassegrain still ignores elevation when a parallactic angle is supplied."""
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
        """The three terms combine as ``nasmyth_sign * el + instrument_rotation + pa``."""
        # site is FYST with nasmyth_sign = +1
        offset = InstrumentOffset(dx=5.0, dy=3.0, instrument_rotation=15.0)
        el = 45.0
        pa = 20.0

        rot = compute_focal_plane_rotation(el, site, offset, parallactic_angle=pa)
        # +1 * 45 + 15 + 20 = 80
        assert rot == pytest.approx(80.0)

    def test_left_nasmyth_all_components(self):
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
    """A zero dx/dy offset returns a copy; a non-zero rotation still recomputes."""

    def test_zero_offset_returns_a_copy(self, site):
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

        # The early exit returns a copy that shares every array; a recomputed
        # trajectory carries new ones. That is what distinguishes the branches.
        assert result.az is not trajectory.az
        # A zero dx/dy offset still moves nothing, whichever branch computes it.
        np.testing.assert_allclose(result.az, trajectory.az, rtol=0, atol=1e-12)
        np.testing.assert_allclose(result.el, trajectory.el, rtol=0, atol=1e-12)

    def test_nonzero_offset_not_early_exit(self, site):
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

        # A recomputed trajectory carries new arrays, and this offset is large
        # enough that it also moves the pointing by degrees.
        assert result.az is not trajectory.az
        assert np.max(np.abs(result.az - trajectory.az)) > 1.0
        assert np.max(np.abs(result.el - trajectory.el)) > 0.5


class TestFromFocalPlane:
    """The mm-to-arcmin factory: the plate-scale conversion and the pass-through fields."""

    def test_basic_conversion(self):
        """The factory converts mm to arcmin through the plate scale."""
        offset = InstrumentOffset.from_focal_plane(x_mm=0.0, y_mm=-461.3, plate_scale=13.89)
        # 461.3 mm * 13.89 arcsec/mm / 60 = 106.79 arcmin
        assert offset.dx == pytest.approx(0.0, abs=1e-10)
        assert offset.dy == pytest.approx(-106.79, abs=0.01)

    def test_name_passed_through(self):
        offset = InstrumentOffset.from_focal_plane(
            x_mm=0.0, y_mm=-461.3, plate_scale=13.89, name="TestModule"
        )
        assert offset.name == "TestModule"

    def test_instrument_rotation_passed_through(self):
        offset = InstrumentOffset.from_focal_plane(
            x_mm=0.0, y_mm=-461.3, plate_scale=13.89, instrument_rotation=15.0
        )
        assert offset.instrument_rotation == pytest.approx(15.0)

    def test_zero_position_returns_zero_offset(self):
        offset = InstrumentOffset.from_focal_plane(x_mm=0.0, y_mm=0.0, plate_scale=13.89)
        assert offset.dx == pytest.approx(0.0, abs=1e-10)
        assert offset.dy == pytest.approx(0.0, abs=1e-10)

    def test_consistency_with_manual_calculation(self):
        x_mm, y_mm, plate_scale = 100.0, 200.0, 13.89

        # Manual calculation
        dx_arcmin_manual = x_mm * plate_scale / 60.0
        dy_arcmin_manual = y_mm * plate_scale / 60.0

        # Via factory
        offset = InstrumentOffset.from_focal_plane(x_mm=x_mm, y_mm=y_mm, plate_scale=plate_scale)

        assert offset.dx == pytest.approx(dx_arcmin_manual, abs=1e-10)
        assert offset.dy == pytest.approx(dy_arcmin_manual, abs=1e-10)

    def test_symmetric_positions(self):
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
    """The shipped module constants match the plate-scale conversion of their mm positions."""

    def test_conversion_consistent_with_plate_scale(self):
        plate_scale = get_fyst_site().plate_scale
        # I1 is at (0, -461.3) mm
        i1 = PRIMECAM_MODULES["i1"]
        expected_dy = -461.3 * plate_scale / 60.0

        assert i1.dx == pytest.approx(0.0, abs=1e-10)
        assert i1.dy == pytest.approx(expected_dy, rel=1e-6)


class TestComputeFocalPlaneRotationArray:
    """The array-input path of ``compute_focal_plane_rotation``.

    Most callers pass scalars; a per-sample trajectory passes arrays, so
    elevation and parallactic angle must broadcast elementwise.
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

    The values are deg, so deg*3600 = arcsec. Pin the true magnitudes so the
    comments cannot drift.
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
    The pole guard raises a clear ``OffsetInversionError`` instead, a type
    inside the ``PointingError`` hierarchy the callers' Raises sections
    advertise, so geometric infeasibility is catchable with the rest.
    """

    def test_zenith_offset_raises_instead_of_wrong_azimuth(self):
        # bore=(180, 89), dx=60', fr=90 lands the detector at el=90 (the pole).
        offset = InstrumentOffset(dx=60.0, dy=0.0)
        det_az, det_el = boresight_to_detector(180.0, 89.0, offset, field_rotation=90.0)
        assert det_el == pytest.approx(90.0, abs=1e-3)

        with pytest.raises(OffsetInversionError, match="azimuth") as excinfo:
            detector_to_boresight(det_az, det_el, offset, field_rotation=90.0)
        # It is in the library's hierarchy, so a caller catching PointingError
        # (or ValueError) sees it.
        assert isinstance(excinfo.value, PointingError)

    def test_array_call_names_the_offending_samples(self):
        """An array call reports which samples tripped the pole guard.

        The whole call still refuses: a partially-inverted trajectory would be
        silently wrong at the degenerate samples. What the caller gains is the
        indices, so a long trajectory is diagnosable.
        """
        offset = InstrumentOffset(dx=60.0, dy=0.0)
        pole_az, pole_el = boresight_to_detector(180.0, 89.0, offset, field_rotation=90.0)
        safe_az, safe_el = boresight_to_detector(180.0, 45.0, offset, field_rotation=90.0)
        det_az = np.array([safe_az, pole_az, safe_az, pole_az])
        det_el = np.array([safe_el, pole_el, safe_el, pole_el])

        with pytest.raises(OffsetInversionError) as excinfo:
            detector_to_boresight(det_az, det_el, offset, field_rotation=90.0)
        assert excinfo.value.indices == (1, 3)
        assert "[1, 3]" in str(excinfo.value)

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
    The only frame leak is then the mechanical term itself: a ``for_fyst()``
    (refracted) input evaluates
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


try:
    import scanning as _scanning  # noqa: F401

    HAS_SCANNING = True
except ImportError:
    HAS_SCANNING = False


@pytest.mark.slow
@pytest.mark.skipif(not HAS_SCANNING, reason="requires the scanning (scan_patterns) package")
class TestScanningModuleProjectionParity:
    """scanning's module projection matches ours, absolute sign included.

    The boresight-level parity tests never apply a module offset, so the
    field-rotation sign does not enter them; a round trip cancels a
    consistent sign error, and a flip-sensitivity test passes under
    either sign. This compares the two packages' detector placement
    directly, at an off-axis module and nonzero elevations, where a
    flipped mechanical rotation displaces the module by ``2 r sin(el)``
    (2.5 deg at el 45). Exactly that flip has shipped undetected once.
    """

    # The elevations below (25 and 80) sit outside the oracle's declared
    # 30-75 deg validity band, deliberately: the displacement under a flipped
    # sign scales as sin(el), so the ends of the range are where it is largest.
    # The comparison is exact (atol 1e-9), so an oracle that degraded outside
    # its band would fail this test rather than pass it silently.
    @pytest.mark.filterwarnings("ignore:elevation has values outside of 30 to 75 range")
    def test_from_boresight_matches_mechanical_rotation(self, site):
        import scanning

        # The same construction scanning's own behavioral tests use; its
        # private projection method is the exact seam under test.
        pong = scanning.Pong(num_term=4, width=2, height=2, spacing=0.1, velocity=0.5)
        tp = scanning.TelescopePattern(pong, start_ra=180, start_dec=-30, start_hrang=-1)

        dist, theta = 1.77985, 30.0
        az = np.array([0.0, 90.0, 180.0, 300.0])
        el = np.array([25.0, 45.0, 65.0, 80.0])

        their_az, their_el = tp._transform_from_boresight(az, el, dist, theta)

        offset = InstrumentOffset(
            dx=dist * 60.0 * np.cos(np.radians(theta)),
            dy=dist * 60.0 * np.sin(np.radians(theta)),
        )
        our_az, our_el = boresight_to_detector(
            az, el, offset, field_rotation=site.nasmyth_sign * el
        )

        # The oracle revision pinned by CI still applies the flipped sign this
        # test guards against. An oracle that reproduces the flip exactly is
        # that known state, not a new disagreement, so report it as an expected
        # failure rather than hide it. Remove this gate when the pin is bumped.
        flip_az, flip_el = boresight_to_detector(
            az, el, offset, field_rotation=-site.nasmyth_sign * el
        )
        flip_delta = (np.asarray(their_az) - np.asarray(flip_az) + 180.0) % 360.0 - 180.0
        if np.allclose(flip_delta, 0.0, atol=1e-9) and np.allclose(their_el, flip_el, atol=1e-9):
            pytest.xfail(
                "the installed scan_patterns revision carries the field-rotation sign flip; "
                "bump the oracle pin in .github/workflows/tests.yml once the fix is pushed"
            )

        az_delta = (np.asarray(their_az) - np.asarray(our_az) + 180.0) % 360.0 - 180.0
        np.testing.assert_allclose(az_delta, 0.0, atol=1e-9)
        np.testing.assert_allclose(their_el, our_el, atol=1e-9)


class TestSkyToFocalPlane:
    """The inverse projection recovers the focal-plane offset a source sits at."""

    ARCSEC = 1.0 / 3600.0

    @pytest.mark.parametrize("el", [20.0, 45.0, 70.0, 89.0])
    @pytest.mark.parametrize("rotation", [0.0, 37.0, -120.0, 200.0])
    @pytest.mark.parametrize("dx, dy", [(5.0, 3.0), (-40.0, 10.0), (0.0, 0.0), (90.0, -90.0)])
    def test_round_trips_boresight_to_detector(self, el, rotation, dx, dy):
        offset = InstrumentOffset(dx=dx, dy=dy)
        det_az, det_el = boresight_to_detector(180.0, el, offset, field_rotation=rotation)
        xi, eta = sky_to_focal_plane(180.0, el, det_az, det_el, field_rotation=rotation)
        assert xi == pytest.approx(offset.dx_deg, abs=0.01 * self.ARCSEC)
        assert eta == pytest.approx(offset.dy_deg, abs=0.01 * self.ARCSEC)

    def test_axis_conventions_against_the_flat_sky(self):
        """An independent small-angle check: xi follows azimuth, eta follows elevation."""
        # Offsets small enough that the second-order spherical terms (which
        # scale as the offset squared) sit below the tolerance.
        bore_az, bore_el = 100.0, 40.0
        d_az, d_el = 0.02, 0.01
        xi, eta = sky_to_focal_plane(bore_az, bore_el, bore_az + d_az, bore_el + d_el, 0.0)
        assert xi == pytest.approx(d_az * np.cos(np.radians(bore_el)), abs=5e-6)
        assert eta == pytest.approx(d_el, abs=5e-6)
        # The rotation is applied the way the forward map applies it: a source
        # straight "up" in the horizon frame lands on the focal-plane axis that
        # the rotation carries onto the elevation direction.
        xi_r, eta_r = sky_to_focal_plane(bore_az, bore_el, bore_az, bore_el + d_el, 90.0)
        assert xi_r == pytest.approx(d_el, abs=5e-6)
        assert eta_r == pytest.approx(0.0, abs=5e-6)

    def test_coincident_positions_land_on_the_origin(self):
        assert sky_to_focal_plane(123.0, 33.0, 123.0, 33.0, 12.0) == (0.0, 0.0)

    def test_broadcasts_over_a_trajectory(self):
        offset = InstrumentOffset(dx=12.0, dy=-7.0)
        az = np.linspace(100.0, 140.0, 25)
        el = np.linspace(30.0, 60.0, 25)
        rotation = -el
        det_az, det_el = boresight_to_detector(az, el, offset, field_rotation=rotation)
        xi, eta = sky_to_focal_plane(az, el, det_az, det_el, rotation)
        np.testing.assert_allclose(xi, offset.dx_deg, atol=0.01 * self.ARCSEC)
        np.testing.assert_allclose(eta, offset.dy_deg, atol=0.01 * self.ARCSEC)
        assert isinstance(xi, np.ndarray) and xi.shape == az.shape

    def test_scalar_inputs_return_floats(self):
        xi, eta = sky_to_focal_plane(10.0, 50.0, 10.5, 50.2, 15.0)
        assert isinstance(xi, float) and isinstance(eta, float)
