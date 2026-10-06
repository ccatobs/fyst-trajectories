"""Tests for apply_detector_offset and the builder's for_detector path."""

import warnings

import numpy as np
import pytest
from _pa_oracle import _apparent_pa
from _site_ports import _site_with_port
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from fyst_trajectories.coordinates import Coordinates
from fyst_trajectories.offsets import (
    InstrumentOffset,
    apply_detector_offset,
    boresight_to_detector,
    compute_focal_plane_rotation,
)
from fyst_trajectories.patterns import (
    ConstantElScanConfig,
    PongScanConfig,
    TrajectoryBuilder,
)
from fyst_trajectories.primecam import (
    PRIMECAM_I1,
    get_primecam_offset,
)
from fyst_trajectories.retune import inject_retune
from fyst_trajectories.site import (
    AtmosphericConditions,
    get_fyst_site,
)
from fyst_trajectories.trajectory import RetuneEvent, Trajectory
from fyst_trajectories.trajectory_utils import get_absolute_times


@pytest.fixture(scope="module")
def pong_trajectory():
    """Return the 60 s, 1 x 1 deg pong at (180, -30) the offset tests share.

    Its build trips the dynamics advisories, and a module-scoped fixture is built
    inside whichever test requests it first, so every test that requests it
    carries the same filterwarnings mark.
    """
    return (
        TrajectoryBuilder(get_fyst_site())
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
        .starting_at(Time("2026-03-15T04:00:00", scale="utc"))
        .build()
    )


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

        adj_no_time = apply_detector_offset(_traj(None), offset, site=site)
        adj_with_time = apply_detector_offset(
            _traj(Time("2026-03-15T04:00:00", scale="utc")), offset, site=site
        )

        np.testing.assert_allclose(adj_no_time.az, adj_with_time.az)
        np.testing.assert_allclose(adj_no_time.el, adj_with_time.el)

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_offset_changes_positions(self, site, pong_trajectory):
        offset = InstrumentOffset(dx=30.0, dy=30.0)  # 0.5 deg offset
        adjusted = apply_detector_offset(pong_trajectory, offset, site=site)

        assert not np.allclose(adjusted.az, pong_trajectory.az)
        # Inverse offset: boresight shifts opposite to detector, so elevation drops
        assert np.mean(adjusted.el) < np.mean(pong_trajectory.el)

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_preserves_metadata(self, site, pong_trajectory):
        offset = InstrumentOffset(dx=30.0, dy=30.0)
        adjusted = apply_detector_offset(pong_trajectory, offset, site=site)

        assert adjusted.metadata is pong_trajectory.metadata
        assert adjusted.pattern_type == pong_trajectory.pattern_type
        assert adjusted.center_ra == pong_trajectory.center_ra
        assert adjusted.center_dec == pong_trajectory.center_dec

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_preserves_start_time(self, site, pong_trajectory):
        offset = InstrumentOffset(dx=30.0, dy=30.0)
        adjusted = apply_detector_offset(pong_trajectory, offset, site=site)

        assert adjusted.start_time == pong_trajectory.start_time

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
        adjusted = apply_detector_offset(trajectory, offset, site=site)

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
        adjusted = apply_detector_offset(trajectory, offset, site=site)

        assert adjusted.scan_flag is not None
        np.testing.assert_array_equal(adjusted.scan_flag, scan_flag)


class TestBuilderForDetector:
    """``for_detector`` moves the built trajectory by the module's on-sky distance."""

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
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

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
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
        adjusted = apply_detector_offset(trajectory, offset, site=site)

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
            adjusted = apply_detector_offset(trajectory, offset, site=site)
        assert adjusted.n_points == trajectory.n_points

    def test_left_nasmyth_sign_flip(self):
        """``nasmyth_port='left'`` flips the sign of the elevation rotation."""
        right_site = _site_with_port("right")
        left_site = _site_with_port("left")

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

        adj_right = apply_detector_offset(traj_right, offset, site=right_site)
        adj_left = apply_detector_offset(traj_left, offset, site=left_site)

        # The offsets should differ because the sign of el in rotation is flipped
        assert not np.allclose(adj_right.az, adj_left.az)
        # The left port rotates the focal plane by -el at the boresight's own
        # elevation, so pushing the left boresight forward at that rotation
        # lands the module back on the commanded path.
        det_az, det_el = boresight_to_detector(adj_left.az, adj_left.el, offset, -adj_left.el)
        np.testing.assert_allclose(det_az, traj_left.az, rtol=0, atol=1e-9)
        np.testing.assert_allclose(det_el, traj_left.el, rtol=0, atol=1e-9)

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

        adj_no_rot = apply_detector_offset(trajectory, offset_no_rot, site=site)
        adj_with_rot = apply_detector_offset(trajectory, offset_with_rot, site=site)

        # Different instrument_rotation should produce different trajectories
        assert not np.allclose(adj_no_rot.az, adj_with_rot.az)

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_celestial_metadata_does_not_change_projection(self, site, pong_trajectory):
        """Same az/el in, same az/el out: celestial metadata is irrelevant.

        Frame-invariance regression: the focal-plane-to-az/el projection
        depends only on (az, el, offset, mechanical rotation). Two trajectories
        with identical az/el paths must produce identical boresights whether or
        not ``center_ra`` / ``center_dec`` metadata is present. Adding the
        parallactic angle when RA/Dec is available would make the two paths
        diverge by degrees for an off-axis module.
        """
        celestial = pong_trajectory
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
            adj_celestial = apply_detector_offset(celestial, offset, site=site)
            adj_bare = apply_detector_offset(bare, offset, site=site)

        np.testing.assert_allclose(adj_celestial.az, adj_bare.az, atol=1e-12)
        np.testing.assert_allclose(adj_celestial.el, adj_bare.el, atol=1e-12)


class TestEarlyExitZeroOffset:
    """A zero dx/dy offset returns a copy, whatever its instrument rotation."""

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_zero_offset_returns_a_copy(self, site, pong_trajectory):
        offset = InstrumentOffset(dx=0.0, dy=0.0)
        result = apply_detector_offset(pong_trajectory, offset, site=site)

        # The early exit returns a new Trajectory object that shares every array.
        assert result is not pong_trajectory
        np.testing.assert_array_equal(result.az, pong_trajectory.az)
        np.testing.assert_array_equal(result.el, pong_trajectory.el)

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_zero_offset_with_instrument_rotation_returns_a_copy(self, site, pong_trajectory):
        # A rotated zero vector is still zero, so the rotation alone moves nothing.
        offset = InstrumentOffset(dx=0.0, dy=0.0, instrument_rotation=15.0)
        result = apply_detector_offset(pong_trajectory, offset, site=site)

        assert result is not pong_trajectory
        assert result.az is pong_trajectory.az
        assert result.az_vel is pong_trajectory.az_vel

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_nonzero_offset_not_early_exit(self, site, pong_trajectory):
        offset = InstrumentOffset(dx=30.0, dy=30.0)
        result = apply_detector_offset(pong_trajectory, offset, site=site)

        # A recomputed pong_trajectory carries new arrays, and this offset is large
        # enough that it also moves the pointing by degrees.
        assert result.az is not pong_trajectory.az
        assert np.max(np.abs(result.az - pong_trajectory.az)) > 1.0
        assert np.max(np.abs(result.el - pong_trajectory.el)) > 0.5


class TestOffsetPathLandsOnTarget:
    """The named detector observes the original target after the offset.

    ``apply_detector_offset`` is a horizon-frame (az/el) projection, so it
    must place an off-axis module using the MECHANICAL focal-plane rotation
    only (``nasmyth_sign * el + instrument_rotation``), evaluated at the
    boresight's own elevation, the axis the Nasmyth rotation follows. The
    ground truth here is an *independent* spherical forward projection of the
    rotated offset from the library's adjusted boresight (astropy's
    ``SkyCoord.directional_offset_by``), no library projection functions.
    Peer references for the pa-free az/el
    projection: SO ``make_source_ces`` (static rotation only), NIKA2
    (Perotto et al. 2020, "Calibration and performance of the NIKA2 camera at
    the IRAM 30-m Telescope", A&A 637, A71, doi:10.1051/0004-6361/201936220,
    sec. 5.1, eq. 2: an elevation-only Nasmyth-to-altaz matrix), and the
    KOSMA focal-plane model (``+/-el`` only; the parallactic angle lives in a
    separate celestial pipeline stage).

    The companion tests assert that two wrong models miss the target at this
    geometry: the rotation taken at the detector elevation misses by minutes
    of arc, and the mechanical rotation plus the parallactic angle by degrees.
    The oracle therefore discriminates between the models rather than passing
    vacuously.
    """

    @staticmethod
    def _build_trajectory(site):
        # ra/dec 180/-30 at 09:00 UTC -> el ~ 36 deg (setting), where |pa| is
        # large (~100 deg).
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
            .starting_at(Time("2026-03-15T09:00:00", scale="utc"))
            .build()
        )

    @staticmethod
    def _miss_arcsec(trajectory, boresight, offset, rho_deg):
        """Separation of the target from an independent spherical projection.

        The offset is rotated by ``rho_deg`` and laid off from the boresight
        along a great circle, its position angle measured from the elevation
        direction toward increasing azimuth.
        """
        rho = np.radians(rho_deg)
        dxr = offset.dx_deg * np.cos(rho) - offset.dy_deg * np.sin(rho)
        dyr = offset.dx_deg * np.sin(rho) + offset.dy_deg * np.cos(rho)
        bore = SkyCoord(boresight.az * u.deg, boresight.el * u.deg, frame="altaz")
        actual = bore.directional_offset_by(
            np.arctan2(dxr, dyr) * u.rad, np.hypot(dxr, dyr) * u.deg
        )
        target = SkyCoord(trajectory.az * u.deg, trajectory.el * u.deg, frame="altaz")
        return target.separation(actual).to_value(u.arcsec)

    def test_inner_ring_module_lands_on_target(self):
        site = get_fyst_site()
        trajectory = self._build_trajectory(site)

        # PRIMECAM_I1 is an inner-ring module ~106.8 arcmin (1.78 deg) off-axis.
        boresight = apply_detector_offset(trajectory, PRIMECAM_I1, site=site)

        # Independent mechanical (horizon-frame) rotation at the boresight's
        # own elevation.
        rho_mech = site.nasmyth_sign * boresight.el + PRIMECAM_I1.instrument_rotation
        miss = self._miss_arcsec(trajectory, boresight, PRIMECAM_I1, rho_mech)

        assert miss.max() < 1.0, (
            f"inner-ring module misses target by up to {miss.max():.3f} arcsec; "
            "the az/el projection is not the mechanical rotation at the boresight elevation"
        )

    def test_detector_elevation_rotation_would_miss(self):
        """A rotation taken at the detector (target) elevation misses by arcminutes."""
        site = get_fyst_site()
        trajectory = self._build_trajectory(site)

        boresight = apply_detector_offset(trajectory, PRIMECAM_I1, site=site)

        rho_detector = site.nasmyth_sign * trajectory.el + PRIMECAM_I1.instrument_rotation
        miss = self._miss_arcsec(trajectory, boresight, PRIMECAM_I1, rho_detector)

        assert miss.min() > 60.0, (
            "a detector-elevation rotation should miss by arcminutes; if it lands "
            "on target the 1 arcsec bound no longer discriminates the two elevations"
        )

    def test_pa_in_horizon_frame_would_miss(self):
        """A mechanical + pa model misses grossly: the oracle discriminates."""
        site = get_fyst_site()
        coords = Coordinates(site)
        trajectory = self._build_trajectory(site)

        boresight = apply_detector_offset(trajectory, PRIMECAM_I1, site=site)

        abs_times = get_absolute_times(trajectory)
        pa = _apparent_pa(coords, trajectory.center_ra, trajectory.center_dec, abs_times)
        # Geometry guard: |pa| must be large here or this test is vacuous.
        assert np.abs(pa).min() > 25.0

        rho_wrong = site.nasmyth_sign * boresight.el + pa
        miss = self._miss_arcsec(trajectory, boresight, PRIMECAM_I1, rho_wrong)

        assert miss.min() > 0.5 * 3600.0, (
            "pa-rotated projection should miss by degrees; if it lands on "
            "target the oracle no longer discriminates the frame models"
        )


class TestApplyDetectorOffsetSignature:
    """``site`` is keyword-only, so it cannot be passed where ``offset`` belongs."""

    def test_positional_site_is_rejected(self, site):
        trajectory = Trajectory(
            times=np.array([0.0, 1.0]),
            az=np.array([180.0, 180.1]),
            el=np.array([45.0, 45.0]),
            az_vel=np.array([0.1, 0.1]),
            el_vel=np.zeros(2),
        )
        with pytest.raises(TypeError):
            apply_detector_offset(trajectory, InstrumentOffset(dx=5.0, dy=3.0), site)


class TestApplyDetectorOffsetSingleSample:
    """A length-1 trajectory must not raise an opaque IndexError."""

    def test_single_sample_trajectory_carries_its_velocities(self, site):
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        trajectory = Trajectory(
            times=np.array([0.0]),
            az=np.array([180.0]),
            el=np.array([45.0]),
            az_vel=np.array([0.5]),
            el_vel=np.array([-0.2]),
            start_time=start_time,
        )

        offset = InstrumentOffset(dx=30.0, dy=30.0)
        adjusted = apply_detector_offset(trajectory, offset, site=site)

        assert adjusted.n_points == 1
        # The correction's rate is unknown for one sample: the input's pass through.
        assert adjusted.az_vel[0] == 0.5
        assert adjusted.el_vel[0] == -0.2


class TestApplyDetectorOffsetVelocities:
    """The input's velocities are carried through; only the correction is differenced."""

    @staticmethod
    def _ce(site, timestep):
        return (
            TrajectoryBuilder(site)
            .with_config(
                ConstantElScanConfig(
                    timestep=timestep,
                    az_start=100.0,
                    az_stop=110.0,
                    elevation=50.0,
                    az_speed=1.5,
                    az_accel=1.0,
                )
            )
            .duration(120.0)
            .build()
        )

    def test_rotation_only_offset_returns_the_input(self, site):
        trajectory = self._ce(site, 1.0)
        adjusted = apply_detector_offset(
            trajectory, InstrumentOffset(dx=0.0, dy=0.0, instrument_rotation=30.0), site=site
        )
        for field in ("az", "el", "az_vel", "el_vel"):
            np.testing.assert_array_equal(getattr(adjusted, field), getattr(trajectory, field))

    @pytest.mark.parametrize("timestep", [0.1, 1.0])
    def test_constant_el_keeps_its_analytic_velocities(self, site, timestep):
        # At constant elevation the correction is constant, so the boresight
        # moves exactly as the target does. Differencing the boresight instead
        # missed the analytic turnaround velocity by 0.2 deg/s at a 1 s step.
        trajectory = self._ce(site, timestep)
        adjusted = apply_detector_offset(trajectory, PRIMECAM_I1, site=site)
        np.testing.assert_allclose(adjusted.az_vel, trajectory.az_vel, rtol=0, atol=1e-12)
        np.testing.assert_allclose(adjusted.el_vel, trajectory.el_vel, rtol=0, atol=1e-12)

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory (azimuth|elevation) acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_differenced_input_matches_differencing_the_boresight(self, site, pong_trajectory):
        # A celestial pattern's velocities are already a finite difference of
        # its positions, and np.gradient is linear, so the two forms agree.
        adjusted = apply_detector_offset(pong_trajectory, PRIMECAM_I1, site=site)
        np.testing.assert_allclose(
            adjusted.az_vel, np.gradient(adjusted.az, adjusted.times), rtol=0, atol=1e-10
        )
        np.testing.assert_allclose(
            adjusted.el_vel, np.gradient(adjusted.el, adjusted.times), rtol=0, atol=1e-10
        )


class TestApplyDetectorOffsetRetuneEvents:
    """The offset must preserve retune_events alongside scan_flag==3."""

    @pytest.mark.filterwarnings(
        "ignore:High elevation reduces on-sky azimuth speed:"
        "fyst_trajectories.exceptions.PointingWarning",
        "ignore:Trajectory azimuth acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
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
        adjusted = apply_detector_offset(retuned, offset, site=site)

        # Both the per-sample flags and the event-level provenance must survive.
        assert len(adjusted.retune_events) == len(retuned.retune_events)
        assert int(np.sum(adjusted.scan_flag == 3)) == n_retune_samples
        assert adjusted.retune_events is retuned.retune_events


class TestApplyDetectorOffsetFrameConsistency:
    """Frame-varying regression for a refracted-el input.

    The mechanical term is evaluated at the boresight elevation solved from
    each sample's own ``trajectory.el``. A ``for_fyst()`` (refracted) input
    therefore evaluates
    ``nasmyth_sign * el`` at the apparent elevation, differing from vacuum
    by ``nasmyth_sign * (refraction bump)``, which moves the boresight of an
    inner-ring module by about 2 arcsec at el ~ 36 deg. This test bounds that
    leak; the vacuum path remains the reference.
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

        adj_vac = apply_detector_offset(traj_vac, offset, site=site)
        adj_ref = apply_detector_offset(traj_ref, offset, site=site)

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
        boresight = apply_detector_offset(traj_vac, offset, site=site)

        # Mechanical (horizon-frame) rotation at the boresight's own elevation,
        # forward/inverse consistency; the independent frame-model oracle lives
        # in TestOffsetPathLandsOnTarget.
        phi = compute_focal_plane_rotation(boresight.el, site=site, offset=offset)
        actual_az, actual_el = boresight_to_detector(boresight.az, boresight.el, offset, phi)

        target = SkyCoord(traj_vac.az * u.deg, traj_vac.el * u.deg, frame="altaz")
        actual = SkyCoord(actual_az * u.deg, actual_el * u.deg, frame="altaz")
        miss_arcsec = target.separation(actual).to_value(u.arcsec)
        assert miss_arcsec.max() < 0.01
