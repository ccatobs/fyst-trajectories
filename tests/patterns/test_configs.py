"""Tests for pattern configuration classes.

These tests focus on validation logic and ensuring invalid configurations
are rejected with appropriate error messages.
"""

import pytest

from fyst_trajectories import PointingWarning
from fyst_trajectories.patterns import (
    ConstantElScanConfig,
    DaisyScanConfig,
    PlanetTrackConfig,
    PongScanConfig,
    ScanConfig,
)


class TestScanConfig:
    """Tests for base ScanConfig class."""

    def test_invalid_timestep_negative(self):
        """Test that negative timestep raises."""
        with pytest.raises(ValueError, match="timestep must be positive"):
            ScanConfig(timestep=-0.1)

    def test_invalid_timestep_zero(self):
        """Test that zero timestep raises."""
        with pytest.raises(ValueError, match="timestep must be positive"):
            ScanConfig(timestep=0.0)

    def test_config_is_frozen(self):
        """Test that config is immutable."""
        config = ScanConfig(timestep=0.1)
        with pytest.raises(Exception):  # FrozenInstanceError
            config.timestep = 0.2


class TestConstantElScanConfig:
    """Tests for ConstantElScanConfig."""

    def test_invalid_az_speed(self):
        """Test that invalid az_speed raises."""
        with pytest.raises(ValueError, match="az_speed must be positive"):
            ConstantElScanConfig(
                timestep=0.1,
                az_start=0,
                az_stop=10,
                elevation=45,
                az_speed=-1.0,
                az_accel=0.5,
            )

        with pytest.raises(ValueError, match="az_speed must be positive"):
            ConstantElScanConfig(
                timestep=0.1,
                az_start=0,
                az_stop=10,
                elevation=45,
                az_speed=0.0,
                az_accel=0.5,
            )

    def test_invalid_az_accel(self):
        """Test that invalid az_accel raises."""
        with pytest.raises(ValueError, match="az_accel must be positive"):
            ConstantElScanConfig(
                timestep=0.1,
                az_start=0,
                az_stop=10,
                elevation=45,
                az_speed=1.0,
                az_accel=-1.0,
            )

        with pytest.raises(ValueError, match="az_accel must be positive"):
            ConstantElScanConfig(
                timestep=0.1,
                az_start=0,
                az_stop=10,
                elevation=45,
                az_speed=1.0,
                az_accel=0.0,
            )

    def test_n_scans_keyword_rejected(self):
        """``n_scans`` was removed as a dead/false-contract field.

        Guard against accidental re-introduction: the constant-elevation scan
        length is set by ``duration`` at generate time, never by a config field.
        """
        with pytest.raises(TypeError):
            ConstantElScanConfig(
                timestep=0.1,
                az_start=0,
                az_stop=10,
                elevation=45,
                az_speed=1.0,
                az_accel=0.5,
                n_scans=1,
            )


class TestPongScanConfig:
    """Tests for PongScanConfig."""

    def test_invalid_width(self):
        """Test that invalid width raises."""
        with pytest.raises(ValueError, match="width must be positive"):
            PongScanConfig(
                timestep=0.1,
                width=0.0,
                height=2.0,
                spacing=0.1,
                velocity=0.5,
                num_terms=4,
                angle=0.0,
            )
        with pytest.raises(ValueError, match="width must be positive"):
            PongScanConfig(
                timestep=0.1,
                width=-1.0,
                height=2.0,
                spacing=0.1,
                velocity=0.5,
                num_terms=4,
                angle=0.0,
            )

    def test_invalid_height(self):
        """Test that invalid height raises."""
        with pytest.raises(ValueError, match="height must be positive"):
            PongScanConfig(
                timestep=0.1,
                width=2.0,
                height=0.0,
                spacing=0.1,
                velocity=0.5,
                num_terms=4,
                angle=0.0,
            )
        with pytest.raises(ValueError, match="height must be positive"):
            PongScanConfig(
                timestep=0.1,
                width=2.0,
                height=-1.0,
                spacing=0.1,
                velocity=0.5,
                num_terms=4,
                angle=0.0,
            )

    def test_invalid_spacing(self):
        """Test that invalid spacing raises."""
        with pytest.raises(ValueError, match="spacing must be positive"):
            PongScanConfig(
                timestep=0.1,
                width=2.0,
                height=2.0,
                spacing=0.0,
                velocity=0.5,
                num_terms=4,
                angle=0.0,
            )
        with pytest.raises(ValueError, match="spacing must be positive"):
            PongScanConfig(
                timestep=0.1,
                width=2.0,
                height=2.0,
                spacing=-0.1,
                velocity=0.5,
                num_terms=4,
                angle=0.0,
            )

    def test_invalid_velocity(self):
        """Test that invalid velocity raises."""
        with pytest.raises(ValueError, match="velocity must be positive"):
            PongScanConfig(
                timestep=0.1,
                width=2.0,
                height=2.0,
                spacing=0.1,
                velocity=0.0,
                num_terms=4,
                angle=0.0,
            )
        with pytest.raises(ValueError, match="velocity must be positive"):
            PongScanConfig(
                timestep=0.1,
                width=2.0,
                height=2.0,
                spacing=0.1,
                velocity=-0.5,
                num_terms=4,
                angle=0.0,
            )

    def test_invalid_num_terms(self):
        """Test that invalid num_terms raises."""
        with pytest.raises(ValueError, match="num_terms must be at least 1"):
            PongScanConfig(
                timestep=0.1,
                width=2.0,
                height=2.0,
                spacing=0.1,
                velocity=0.5,
                num_terms=0,
                angle=0.0,
            )


class TestDaisyScanConfig:
    """Tests for DaisyScanConfig."""

    def test_invalid_radius(self):
        """Test that invalid radius raises."""
        with pytest.raises(ValueError, match="radius must be positive"):
            DaisyScanConfig(
                timestep=0.1,
                radius=0.0,
                velocity=0.3,
                turn_radius=0.2,
                avoidance_radius=0.0,
                start_acceleration=0.5,
                y_offset=0.0,
            )
        with pytest.raises(ValueError, match="radius must be positive"):
            DaisyScanConfig(
                timestep=0.1,
                radius=-0.5,
                velocity=0.3,
                turn_radius=0.2,
                avoidance_radius=0.0,
                start_acceleration=0.5,
                y_offset=0.0,
            )

    def test_invalid_velocity(self):
        """Test that invalid velocity raises."""
        with pytest.raises(ValueError, match="velocity must be positive"):
            DaisyScanConfig(
                timestep=0.1,
                radius=0.5,
                velocity=0.0,
                turn_radius=0.2,
                avoidance_radius=0.0,
                start_acceleration=0.5,
                y_offset=0.0,
            )
        with pytest.raises(ValueError, match="velocity must be positive"):
            DaisyScanConfig(
                timestep=0.1,
                radius=0.5,
                velocity=-0.3,
                turn_radius=0.2,
                avoidance_radius=0.0,
                start_acceleration=0.5,
                y_offset=0.0,
            )

    def test_invalid_turn_radius(self):
        """Test that invalid turn_radius raises."""
        with pytest.raises(ValueError, match="turn_radius must be positive"):
            DaisyScanConfig(
                timestep=0.1,
                radius=0.5,
                velocity=0.3,
                turn_radius=0.0,
                avoidance_radius=0.0,
                start_acceleration=0.5,
                y_offset=0.0,
            )
        with pytest.raises(ValueError, match="turn_radius must be positive"):
            DaisyScanConfig(
                timestep=0.1,
                radius=0.5,
                velocity=0.3,
                turn_radius=-0.2,
                avoidance_radius=0.0,
                start_acceleration=0.5,
                y_offset=0.0,
            )

    def test_invalid_avoidance_radius(self):
        """Test that negative avoidance_radius raises."""
        with pytest.raises(ValueError, match="avoidance_radius must be non-negative"):
            DaisyScanConfig(
                timestep=0.1,
                radius=0.5,
                velocity=0.3,
                turn_radius=0.2,
                avoidance_radius=-0.1,
                start_acceleration=0.5,
                y_offset=0.0,
            )


class TestPlanetTrackConfig:
    """Tests for PlanetTrackConfig."""

    def test_invalid_body(self):
        """Test that invalid body raises."""
        with pytest.raises(ValueError, match="Unknown body"):
            PlanetTrackConfig(timestep=0.1, body="pluto")


class TestConfigWarnings:
    """Advisory ``PointingWarning`` paths in config ``__post_init__`` routines.

    These warn-on-unusual-value branches (and the constant-el
    turnaround-exceeds-throw branch) had no regression guard, a mutant
    deleting them passed.
    """

    def test_constant_el_az_throw_warns(self):
        with pytest.warns(PointingWarning, match="Azimuth throw"):
            ConstantElScanConfig(
                timestep=0.1,
                az_start=0.0,
                az_stop=40.0,
                elevation=45.0,
                az_speed=1.0,
                az_accel=1.0,
            )

    def test_constant_el_speed_warns(self):
        with pytest.warns(PointingWarning, match="Azimuth speed"):
            ConstantElScanConfig(
                timestep=0.1,
                az_start=0.0,
                az_stop=10.0,
                elevation=45.0,
                az_speed=6.0,
                az_accel=3.0,
            )

    def test_constant_el_turnaround_exceeds_throw_warns(self):
        # d_half_turn = 5*v^2/(8*a) = 5*4/(8*0.5) = 5.0 deg, well over the 2 deg throw.
        with pytest.warns(PointingWarning, match="Turnaround distance"):
            ConstantElScanConfig(
                timestep=0.1,
                az_start=0.0,
                az_stop=2.0,
                elevation=45.0,
                az_speed=2.0,
                az_accel=0.5,
            )

    def test_pong_width_warns(self):
        with pytest.warns(PointingWarning, match="Scan width"):
            PongScanConfig(
                timestep=0.1,
                width=40.0,
                height=2.0,
                spacing=0.1,
                velocity=0.5,
                num_terms=4,
                angle=0.0,
            )

    def test_daisy_radius_warns(self):
        with pytest.warns(PointingWarning, match="Daisy radius"):
            DaisyScanConfig(
                timestep=0.1,
                radius=20.0,
                velocity=0.5,
                turn_radius=0.2,
                avoidance_radius=0.0,
                start_acceleration=0.5,
                y_offset=0.0,
            )
