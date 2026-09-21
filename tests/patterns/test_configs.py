"""Tests for pattern configuration classes.

These tests focus on validation logic and ensuring invalid configurations
are rejected with appropriate error messages.
"""

from dataclasses import FrozenInstanceError

import pytest

from fyst_trajectories import PointingWarning
from fyst_trajectories.patterns import (
    ConstantElScanConfig,
    DaisyScanConfig,
    LinearMotionConfig,
    PlanetTrackConfig,
    PongScanConfig,
    ScanConfig,
)


class TestScanConfig:
    """The base config validates ``timestep`` and is frozen."""

    def test_invalid_timestep_negative(self):
        with pytest.raises(ValueError, match="timestep must be positive"):
            ScanConfig(timestep=-0.1)

    def test_invalid_timestep_zero(self):
        with pytest.raises(ValueError, match="timestep must be positive"):
            ScanConfig(timestep=0.0)

    def test_config_is_frozen(self):
        config = ScanConfig(timestep=0.1)
        with pytest.raises(FrozenInstanceError):
            config.timestep = 0.2


class TestConstantElScanConfig:
    """Speed and acceleration must be positive, and ``n_scans`` is not a field."""

    def test_invalid_az_speed(self):
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
        """``ConstantElScanConfig`` has no ``n_scans`` field; the leg count is derived.

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
    """Every Lissajous geometry field is rejected at zero or negative."""

    def test_invalid_width(self):
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
    """Rosette geometry must be positive; ``avoidance_radius`` may be zero but not negative."""

    def test_invalid_radius(self):
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
    """An unknown body name is refused at construction."""

    def test_invalid_body(self):
        with pytest.raises(ValueError, match="Unknown body"):
            PlanetTrackConfig(timestep=0.1, body="pluto")


class TestConfigWarnings:
    """Advisory ``PointingWarning`` paths in config ``__post_init__`` routines.

    These warn-on-unusual-value branches (and the constant-el
    turnaround-exceeds-throw branch) are guarded here, so a mutant deleting
    them fails.
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


class TestConfigsRejectNonFiniteValues:
    """A NaN or infinite field is refused at construction, not much later.

    A bare ``value <= 0`` check lets both through: every comparison with NaN
    is False, and infinity is positive. The bad value then surfaces far away,
    a NaN velocity dying inside the leg quantiser as ``cannot convert float
    NaN to integer``, with nothing pointing back at the config.
    """

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_pong_rejects_non_finite_geometry(self, bad):
        for field in ("width", "height", "spacing", "velocity"):
            params = dict(
                timestep=0.1,
                width=2.0,
                height=2.0,
                spacing=0.1,
                velocity=0.5,
                num_terms=4,
                angle=0.0,
            )
            params[field] = bad
            with pytest.raises(ValueError, match=f"{field} must be positive"):
                PongScanConfig(**params)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_constant_el_rejects_non_finite_kinematics(self, bad):
        for field in ("az_speed", "az_accel", "timestep"):
            params = dict(
                timestep=0.1,
                az_start=100.0,
                az_stop=110.0,
                elevation=45.0,
                az_speed=0.5,
                az_accel=1.0,
            )
            params[field] = bad
            with pytest.raises(ValueError, match=f"{field} must be positive"):
                ConstantElScanConfig(**params)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_daisy_rejects_non_finite_geometry(self, bad):
        for field in ("radius", "velocity", "turn_radius", "start_acceleration"):
            params = dict(
                timestep=0.1,
                radius=0.5,
                velocity=0.5,
                turn_radius=0.2,
                avoidance_radius=0.0,
                start_acceleration=0.5,
                y_offset=0.0,
            )
            params[field] = bad
            with pytest.raises(ValueError, match=f"{field} must be positive"):
                DaisyScanConfig(**params)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_daisy_rejects_non_finite_avoidance_radius(self, bad):
        with pytest.raises(ValueError, match="avoidance_radius must be non-negative"):
            DaisyScanConfig(
                timestep=0.1,
                radius=0.5,
                velocity=0.5,
                turn_radius=0.2,
                avoidance_radius=bad,
                start_acceleration=0.5,
                y_offset=0.0,
            )


class TestLinearMotionConfigValidation:
    """``LinearMotionConfig`` validates its own four motion fields.

    Without a ``__post_init__`` of its own it would inherit only the timestep
    check and let a NaN start position or velocity through to the generator,
    which produces an all-NaN trajectory that only the position-bounds check
    refuses, naming no field.
    """

    @staticmethod
    def _kwargs(**overrides):
        """Return a valid keyword set with the named fields replaced."""
        kwargs = dict(timestep=0.1, az_start=180.0, el_start=45.0, az_velocity=0.1, el_velocity=0.0)
        kwargs.update(overrides)
        return kwargs

    @pytest.mark.parametrize("field", ["az_start", "el_start", "az_velocity", "el_velocity"])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_field_is_refused(self, field, bad):
        """Each motion field rejects NaN and infinity, naming itself."""
        with pytest.raises(ValueError, match=f"{field} must be a finite number"):
            LinearMotionConfig(**self._kwargs(**{field: bad}))

    def test_zero_and_negative_velocities_are_legal(self):
        """A velocity is signed and may be zero; only finiteness is required."""
        config = LinearMotionConfig(**self._kwargs(az_velocity=-0.2, el_velocity=0.0))
        assert config.az_velocity == -0.2
        assert config.el_velocity == 0.0

    def test_timestep_check_is_still_inherited(self):
        """The base class's own timestep check applies alongside the motion checks."""
        with pytest.raises(ValueError, match="timestep must be positive"):
            LinearMotionConfig(**self._kwargs(timestep=0.0))


class TestAdvisoryWarningAttribution:
    """A config advisory points at the code that built the config.

    ``warnings.warn`` counts frames from the function that calls it, so an
    advisory one short (``stacklevel=3`` in the helper, ``stacklevel=2`` in
    the turnaround advisory) lands on the dataclass-generated ``__init__``,
    whose filename astropy reports as ``<string>``. A caller filtering
    warnings by module then cannot match it.
    """

    def test_unusual_value_advisory_names_this_file(self, recwarn):
        """The 'unusually large' advisory is attributed to the construction site."""
        PongScanConfig(
            timestep=0.1,
            width=200.0,
            height=1.0,
            spacing=0.5,
            velocity=0.5,
            num_terms=3,
            angle=0.0,
        )
        advisories = [w for w in recwarn.list if issubclass(w.category, PointingWarning)]
        assert advisories, "expected an 'unusually large' advisory"
        assert all(w.filename == __file__ for w in advisories), [w.filename for w in advisories]

    def test_turnaround_advisory_names_this_file(self, recwarn):
        """The turnaround-dominates advisory is attributed to the construction site."""
        ConstantElScanConfig(
            timestep=0.1,
            az_start=180.0,
            az_stop=180.1,
            elevation=45.0,
            az_speed=1.0,
            az_accel=0.5,
        )
        advisories = [
            w
            for w in recwarn.list
            if issubclass(w.category, PointingWarning) and "Turnaround distance" in str(w.message)
        ]
        assert advisories, "expected a turnaround advisory"
        assert all(w.filename == __file__ for w in advisories)
