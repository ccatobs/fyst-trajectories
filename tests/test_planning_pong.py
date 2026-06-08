"""Tests for plan_pong_scan."""

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories.exceptions import TargetNotObservableError
from fyst_trajectories.offsets import InstrumentOffset
from fyst_trajectories.patterns.configs import PongScanConfig
from fyst_trajectories.planning import (
    FieldRegion,
    ScanBlock,
    plan_pong_rotation_sequence,
    plan_pong_scan,
)


@pytest.fixture
def start_time():
    """Provide a standard start time when the target is observable."""
    return Time("2026-03-15T04:00:00", scale="utc")


@pytest.fixture
def small_field():
    """Provide a small field region for faster tests."""
    return FieldRegion(ra_center=180.0, dec_center=-30.0, width=1.0, height=1.0)


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
        assert "Pong scan" in block.summary

        # Pin the Lissajous params to hand-derived values for this 1x1 deg,
        # spacing=0.1, v=0.5 field. vert_spacing = sqrt(2)*0.1, so
        # x_numvert = y_numvert = ceil(1 / vert_spacing) = 8; the opposite-parity
        # bump lifts y_numvert to 9 (8 and 9 coprime); and
        # period = 4*x*y*spacing/velocity = 4*8*9*0.1/0.5 = 57.6 s. Wrong period,
        # swapped vertex counts, or a half-length trajectory all fail here.
        assert block.computed_params["x_numvert"] == 8
        assert block.computed_params["y_numvert"] == 9
        assert block.computed_params["period"] == pytest.approx(57.6)
        assert block.duration == pytest.approx(57.6)

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


class TestPlanPongRotationSequence:
    """Tests for the multi-rotation Pong helper."""

    @pytest.fixture
    def base_config(self):
        return PongScanConfig(
            timestep=0.1,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=37.5,  # arbitrary; the helper ignores this
        )

    def test_returns_n_configs(self, base_config):
        configs = plan_pong_rotation_sequence(base_config, n_rotations=4)
        assert len(configs) == 4
        assert all(isinstance(c, PongScanConfig) for c in configs)

    def test_angles_evenly_spaced(self, base_config):
        configs = plan_pong_rotation_sequence(base_config, n_rotations=4)
        angles = [c.angle for c in configs]
        assert angles == [0.0, 45.0, 90.0, 135.0]

    def test_other_fields_preserved(self, base_config):
        configs = plan_pong_rotation_sequence(base_config, n_rotations=3)
        for c in configs:
            assert c.width == base_config.width
            assert c.height == base_config.height
            assert c.spacing == base_config.spacing
            assert c.velocity == base_config.velocity
            assert c.num_terms == base_config.num_terms
            assert c.timestep == base_config.timestep

    def test_n_rotations_one_returns_zero_angle(self, base_config):
        configs = plan_pong_rotation_sequence(base_config, n_rotations=1)
        assert len(configs) == 1
        assert configs[0].angle == 0.0

    def test_n_rotations_zero_raises(self, base_config):
        with pytest.raises(ValueError, match="n_rotations must be at least 1"):
            plan_pong_rotation_sequence(base_config, n_rotations=0)

    def test_jcmt_typical_eleven_rotations(self, base_config):
        """JCMT/SCUBA-2 uses ~11 rotations spaced ~16 deg."""
        configs = plan_pong_rotation_sequence(base_config, n_rotations=11)
        angles = [c.angle for c in configs]
        # Last angle should be 10 * 180/11 ~ 163.6
        assert angles[-1] == pytest.approx(180.0 * 10 / 11)
        # Spacing between consecutive is constant
        diffs = np.diff(angles)
        assert np.allclose(diffs, 180.0 / 11)
