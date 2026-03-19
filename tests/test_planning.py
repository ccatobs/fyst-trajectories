"""Tests for fyst_trajectories.planning module."""

import math

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories.exceptions import (
    TargetNotObservableError,
)
from fyst_trajectories.offsets import InstrumentOffset
from fyst_trajectories.patterns.configs import (
    ConstantElScanConfig,
    DaisyScanConfig,
    PongScanConfig,
)
from fyst_trajectories.planning import (
    FieldRegion,
    ScanBlock,
    _field_region_corners,
    plan_constant_el_scan,
    plan_daisy_scan,
    plan_pong_scan,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def start_time():
    """Provide a standard start time when the target is observable."""
    return Time("2026-03-15T04:00:00", scale="utc")


@pytest.fixture
def field():
    """Provide a standard field region for testing."""
    return FieldRegion(ra_center=180.0, dec_center=-30.0, width=2.0, height=2.0)


@pytest.fixture
def small_field():
    """Provide a small field region for faster tests."""
    return FieldRegion(ra_center=180.0, dec_center=-30.0, width=1.0, height=1.0)


# ---------------------------------------------------------------------------
# FieldRegion tests
# ---------------------------------------------------------------------------


class TestFieldRegion:
    """Tests for the FieldRegion dataclass."""

    def test_dec_boundaries(self):
        """Verify dec_min and dec_max are computed from center and height."""
        field = FieldRegion(ra_center=180.0, dec_center=-30.0, width=2.0, height=4.0)
        assert field.dec_min == pytest.approx(-32.0)
        assert field.dec_max == pytest.approx(-28.0)

    def test_zero_width_raises(self):
        """Test that zero width raises ValueError."""
        with pytest.raises(ValueError, match="width must be positive"):
            FieldRegion(ra_center=0.0, dec_center=0.0, width=0.0, height=1.0)

    def test_negative_width_raises(self):
        """Test that negative width raises ValueError."""
        with pytest.raises(ValueError, match="width must be positive"):
            FieldRegion(ra_center=0.0, dec_center=0.0, width=-1.0, height=1.0)

    def test_zero_height_raises(self):
        """Test that zero height raises ValueError."""
        with pytest.raises(ValueError, match="height must be positive"):
            FieldRegion(ra_center=0.0, dec_center=0.0, width=1.0, height=0.0)

    def test_negative_height_raises(self):
        """Test that negative height raises ValueError."""
        with pytest.raises(ValueError, match="height must be positive"):
            FieldRegion(ra_center=0.0, dec_center=0.0, width=1.0, height=-2.0)

    def test_frozen(self):
        """Test that FieldRegion is immutable."""
        field = FieldRegion(ra_center=180.0, dec_center=-30.0, width=2.0, height=2.0)
        with pytest.raises(AttributeError):
            field.ra_center = 0.0


# ---------------------------------------------------------------------------
# plan_pong_scan tests
# ---------------------------------------------------------------------------


class TestPlanPongScan:
    """Tests for plan_pong_scan."""

    def test_basic_plan(self, site, start_time, small_field):
        """plan_pong_scan returns a ScanBlock with pong config and metadata."""
        block = plan_pong_scan(
            field=small_field,
            velocity=0.5,
            spacing=0.1,
            num_terms=4,
            site=site,
            start_time=start_time,
            timestep=0.1,
        )

        assert isinstance(block, ScanBlock)
        assert isinstance(block.config, PongScanConfig)
        assert block.duration > 0
        assert block.trajectory.n_points > 0
        assert "period" in block.computed_params
        assert "x_numvert" in block.computed_params
        assert "y_numvert" in block.computed_params
        assert "Pong scan" in block.summary

    def test_duration_equals_period(self, site, start_time, small_field):
        """Test that default duration is one full period."""
        block = plan_pong_scan(
            field=small_field,
            velocity=0.5,
            spacing=0.1,
            num_terms=4,
            site=site,
            start_time=start_time,
            timestep=0.1,
        )

        expected_period = block.computed_params["period"]
        assert block.duration == pytest.approx(expected_period)

    def test_multiple_cycles(self, site, start_time, small_field):
        """Test that n_cycles multiplies the duration."""
        block1 = plan_pong_scan(
            field=small_field,
            velocity=0.5,
            spacing=0.1,
            num_terms=4,
            site=site,
            start_time=start_time,
            timestep=0.1,
            n_cycles=1,
        )
        block2 = plan_pong_scan(
            field=small_field,
            velocity=0.5,
            spacing=0.1,
            num_terms=4,
            site=site,
            start_time=start_time,
            timestep=0.1,
            n_cycles=2,
        )

        assert block2.duration == pytest.approx(block1.duration * 2)

    def test_invalid_n_cycles_raises(self, site, start_time, small_field):
        """Test that n_cycles < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_cycles must be at least 1"):
            plan_pong_scan(
                field=small_field,
                velocity=0.5,
                spacing=0.1,
                num_terms=4,
                site=site,
                start_time=start_time,
                timestep=0.1,
                n_cycles=0,
            )

    def test_config_matches_field(self, site, start_time, small_field):
        """Test that the generated config uses field width/height."""
        block = plan_pong_scan(
            field=small_field,
            velocity=0.5,
            spacing=0.1,
            num_terms=4,
            site=site,
            start_time=start_time,
            timestep=0.1,
        )

        assert block.config.width == small_field.width
        assert block.config.height == small_field.height

    def test_trajectory_has_valid_bounds(self, site, start_time, small_field):
        """Test that trajectory stays within telescope limits."""
        block = plan_pong_scan(
            field=small_field,
            velocity=0.5,
            spacing=0.1,
            num_terms=4,
            site=site,
            start_time=start_time,
            timestep=0.1,
        )

        traj = block.trajectory
        limits = site.telescope_limits
        assert traj.el.min() >= limits.elevation.min
        assert traj.el.max() <= limits.elevation.max
        assert traj.az.min() >= limits.azimuth.min
        assert traj.az.max() <= limits.azimuth.max

    def test_unobservable_target_raises(self, site, start_time):
        """Test that an unobservable target raises TargetNotObservableError."""
        # Dec = +80 is never visible from FYST (latitude ~ -23)
        field = FieldRegion(ra_center=180.0, dec_center=80.0, width=1.0, height=1.0)
        with pytest.raises(TargetNotObservableError):
            plan_pong_scan(
                field=field,
                velocity=0.5,
                spacing=0.1,
                num_terms=4,
                site=site,
                start_time=start_time,
                timestep=0.1,
            )

    def test_with_angle(self, site, start_time, small_field):
        """Test that angle parameter is passed through correctly."""
        block = plan_pong_scan(
            field=small_field,
            velocity=0.5,
            spacing=0.1,
            num_terms=4,
            site=site,
            start_time=start_time,
            timestep=0.1,
            angle=45.0,
        )

        assert block.config.angle == 45.0

    def test_with_detector_offset(self, site, start_time, small_field):
        """Test that detector offset is applied."""
        block_no_offset = plan_pong_scan(
            field=small_field,
            velocity=0.5,
            spacing=0.1,
            num_terms=4,
            site=site,
            start_time=start_time,
            timestep=0.1,
        )

        offset = InstrumentOffset(dx=5.0, dy=3.0, name="TestDet")
        block_with_offset = plan_pong_scan(
            field=small_field,
            velocity=0.5,
            spacing=0.1,
            num_terms=4,
            site=site,
            start_time=start_time,
            timestep=0.1,
            detector_offset=offset,
        )

        # Trajectories should differ when offset is applied
        assert not np.allclose(block_no_offset.trajectory.az, block_with_offset.trajectory.az)

    def test_scan_block_is_frozen(self, site, start_time, small_field):
        """ScanBlock should be immutable after creation."""
        block = plan_pong_scan(
            field=small_field,
            velocity=0.5,
            spacing=0.1,
            num_terms=4,
            site=site,
            start_time=start_time,
            timestep=0.1,
        )
        with pytest.raises(AttributeError):
            block.duration = 999.0


# ---------------------------------------------------------------------------
# plan_daisy_scan tests
# ---------------------------------------------------------------------------


class TestPlanDaisyScan:
    """Tests for plan_daisy_scan."""

    def test_basic_plan(self, site, start_time):
        """plan_daisy_scan returns a ScanBlock with daisy config."""
        block = plan_daisy_scan(
            ra=180.0,
            dec=-30.0,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            site=site,
            start_time=start_time,
            timestep=0.1,
            duration=60.0,
        )

        assert isinstance(block, ScanBlock)
        assert isinstance(block.config, DaisyScanConfig)
        assert block.duration == pytest.approx(60.0)
        assert block.trajectory.n_points > 0
        assert "Daisy scan" in block.summary

    def test_trajectory_has_valid_bounds(self, site, start_time):
        """Generated trajectory must stay within telescope elevation limits."""
        block = plan_daisy_scan(
            ra=180.0,
            dec=-30.0,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            site=site,
            start_time=start_time,
            timestep=0.1,
            duration=60.0,
        )

        traj = block.trajectory
        limits = site.telescope_limits
        assert traj.el.min() >= limits.elevation.min
        assert traj.el.max() <= limits.elevation.max

    def test_unobservable_target_raises(self, site, start_time):
        """Test that an unobservable target raises TargetNotObservableError."""
        with pytest.raises(TargetNotObservableError):
            plan_daisy_scan(
                ra=180.0,
                dec=80.0,  # Not visible from FYST
                radius=0.5,
                velocity=0.3,
                turn_radius=0.2,
                avoidance_radius=0.0,
                start_acceleration=0.5,
                site=site,
                start_time=start_time,
                timestep=0.1,
                duration=60.0,
            )

    def test_with_detector_offset(self, site, start_time):
        """Test that detector offset is applied."""
        block_no_offset = plan_daisy_scan(
            ra=180.0,
            dec=-30.0,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            site=site,
            start_time=start_time,
            timestep=0.1,
            duration=60.0,
        )

        offset = InstrumentOffset(dx=5.0, dy=3.0, name="TestDet")
        block_with_offset = plan_daisy_scan(
            ra=180.0,
            dec=-30.0,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            site=site,
            start_time=start_time,
            timestep=0.1,
            duration=60.0,
            detector_offset=offset,
        )

        # Trajectories should differ when offset is applied
        assert not np.allclose(block_no_offset.trajectory.az, block_with_offset.trajectory.az)


# ---------------------------------------------------------------------------
# _field_region_corners tests
# ---------------------------------------------------------------------------


class TestFieldRegionCorners:
    """Tests for the _field_region_corners helper."""

    def test_no_rotation(self):
        """With angle=0, corners are axis-aligned around center."""
        corners = _field_region_corners(10.0, -30.0, 4.0, 6.0, 0.0)
        assert len(corners) == 4
        ra_vals = [c[0] for c in corners]
        dec_vals = [c[1] for c in corners]
        # RA offsets are divided by cos(dec) to account for convergence of meridians
        cos_dec = math.cos(math.radians(-30.0))
        assert min(ra_vals) == pytest.approx(10.0 - 2.0 / cos_dec)
        assert max(ra_vals) == pytest.approx(10.0 + 2.0 / cos_dec)
        assert min(dec_vals) == pytest.approx(-33.0)
        assert max(dec_vals) == pytest.approx(-27.0)

    def test_90_degree_rotation_swaps_axes(self):
        """A 90-degree rotation swaps width and height extents."""
        corners = _field_region_corners(0.0, 0.0, 4.0, 2.0, 90.0)
        ra_vals = [c[0] for c in corners]
        dec_vals = [c[1] for c in corners]
        # After 90 deg rotation: width (4.0) appears in Dec, height (2.0) in RA
        assert max(abs(r) for r in ra_vals) == pytest.approx(1.0, abs=0.01)
        assert max(abs(d) for d in dec_vals) == pytest.approx(2.0, abs=0.01)


# ---------------------------------------------------------------------------
# plan_constant_el_scan tests
# ---------------------------------------------------------------------------


class TestPlanConstantElScan:
    """Tests for plan_constant_el_scan."""

    @pytest.fixture
    def ecdfs_field(self):
        """E-CDF-S field region used in the scan strategy script."""
        return FieldRegion(
            ra_center=53.117,
            dec_center=-27.808,
            width=5.0,
            height=6.7,
        )

    @pytest.fixture
    def search_time(self):
        """Provide a base search time for E-CDF-S CE scans."""
        return Time("2026-03-15T17:00:00", scale="utc")

    def test_basic_plan(self, site, ecdfs_field, search_time):
        """plan_constant_el_scan returns a ScanBlock with correct types."""
        block = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=True,
            angle=170.0,
        )

        assert isinstance(block, ScanBlock)
        assert isinstance(block.config, ConstantElScanConfig)
        assert block.duration > 0
        assert block.trajectory.n_points > 0
        assert "az_start" in block.computed_params
        assert "az_stop" in block.computed_params
        assert "az_throw" in block.computed_params
        assert "n_scans" in block.computed_params
        assert "Constant-El scan" in block.summary

    def test_elevation_in_trajectory(self, site, ecdfs_field, search_time):
        """Trajectory should be at the requested elevation."""
        block = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=True,
            angle=170.0,
        )

        assert np.allclose(block.trajectory.el, 50.0)

    def test_rising_vs_setting_different_times(self, site, ecdfs_field, search_time):
        """Rising and setting passes should have different start times."""
        rising = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=True,
            angle=170.0,
        )
        setting = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=False,
            angle=170.0,
        )

        assert rising.computed_params["start_time_iso"] != setting.computed_params["start_time_iso"]

    def test_string_start_time(self, site, ecdfs_field):
        """start_time as a string should work."""
        block = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time="2026-03-15T17:00:00",
            rising=True,
            angle=170.0,
        )

        assert block.duration > 0

    def test_azimuth_throw_positive(self, site, ecdfs_field, search_time):
        """Computed azimuth throw should be positive."""
        block = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=True,
            angle=170.0,
        )

        assert block.computed_params["az_throw"] > 0

    def test_unreachable_field_raises(self, site):
        """A field that never reaches the target elevation should raise."""
        # Dec = +70 is never reachable at el=50 from FYST
        field = FieldRegion(ra_center=180.0, dec_center=70.0, width=1.0, height=1.0)
        with pytest.raises(ValueError, match="Could not find elevation crossing"):
            plan_constant_el_scan(
                field=field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=Time("2026-03-15T00:00:00", scale="utc"),
                rising=True,
            )
