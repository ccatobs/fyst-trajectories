"""Regression tests for degenerate-duration guards in pattern generation.

Every pattern samples on ``n_points = round(duration / timestep) + 1``. A
duration that is zero, negative, or shorter than ``timestep`` collapses to a
single sample. Unguarded, that either fails opaquely in ``np.gradient``
(an unhelpful ``IndexError``) or, for the AltAz patterns that set
velocities directly (linear, constant_el), *silently* returns a wrong
1-point trajectory. These tests pin the contract: such durations raise a
clear ``PointingError`` (a ``ValueError`` subclass), via both the public
``TrajectoryBuilder`` path and the direct ``.generate()`` /
``.generate_offsets()`` path, while a normal duration generates.

Six of the nine registered patterns are driven below. The AltAz Pong and Daisy
patterns raise the same error on the same durations and are not repeated here.
"""

import pytest
from astropy.time import Time

from fyst_trajectories.exceptions import PointingError
from fyst_trajectories.patterns import (
    ConstantElScanConfig,
    ConstantElScanPattern,
    DaisyScanConfig,
    DaisyScanPattern,
    LinearMotionConfig,
    LinearMotionPattern,
    PlanetTrackConfig,
    PlanetTrackPattern,
    PongScanConfig,
    PongScanPattern,
    SiderealTrackConfig,
    SiderealTrackPattern,
    TrajectoryBuilder,
)

_START = Time("2026-06-15T04:00:00", scale="utc")
# A time/body where Jupiter is well above the FYST horizon, for the
# normal-duration sanity checks on the planet pattern.
_JUPITER_UP = Time("2026-06-15T17:30:00", scale="utc")
_TIMESTEP = 0.1
# Observable (RA, Dec) from FYST at _START for celestial sanity checks
# (southern, high-culminating so the trajectory stays within el limits).
_OBS_RA = 270.0
_OBS_DEC = -40.0


def _pong():
    return PongScanPattern(
        ra=180.0,
        dec=-30.0,
        config=PongScanConfig(
            timestep=_TIMESTEP,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        ),
    )


def _daisy():
    return DaisyScanPattern(
        ra=180.0,
        dec=-30.0,
        config=DaisyScanConfig(
            timestep=_TIMESTEP,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.0,
        ),
    )


def _linear():
    return LinearMotionPattern(
        LinearMotionConfig(
            timestep=_TIMESTEP,
            az_start=100.0,
            el_start=45.0,
            az_velocity=0.5,
            el_velocity=0.1,
        )
    )


def _constant_el():
    return ConstantElScanPattern(
        ConstantElScanConfig(
            timestep=_TIMESTEP,
            az_start=120.0,
            az_stop=180.0,
            elevation=45.0,
            az_speed=0.5,
            az_accel=1.0,
        )
    )


def _sidereal(ra=_OBS_RA, dec=_OBS_DEC):
    return SiderealTrackPattern(ra=ra, dec=dec, config=SiderealTrackConfig(timestep=_TIMESTEP))


def _planet(body="jupiter"):
    return PlanetTrackPattern(PlanetTrackConfig(timestep=_TIMESTEP, body=body))


# (pattern factory, config, needs start_time,
#  observable start time for the "normal works" checks).
_PATTERNS = [
    pytest.param(_pong, _pong().config, True, _START, id="pong"),
    pytest.param(_daisy, _daisy().config, True, _START, id="daisy"),
    pytest.param(_linear, _linear().config, False, None, id="linear"),
    pytest.param(_constant_el, _constant_el().config, False, None, id="constant_el"),
    pytest.param(_sidereal, _sidereal().config, True, _START, id="sidereal"),
    pytest.param(_planet, _planet().config, True, _JUPITER_UP, id="planet"),
]


def _generate(pattern, site, duration, needs_start_time, start=_START):
    start_time = start if needs_start_time else None
    return pattern.generate(site, duration, start_time)


class TestDegenerateDurationDirect:
    """Degenerate durations raise via the direct ``.generate()`` path."""

    @pytest.mark.parametrize("pattern_factory, config, needs_start, obs_start", _PATTERNS)
    @pytest.mark.parametrize("duration", [0.0, -1.0])
    def test_nonpositive_duration_raises(
        self, pattern_factory, config, needs_start, obs_start, duration, site
    ):
        """Zero or negative duration raises a clear PointingError, not IndexError."""
        pattern = pattern_factory()
        with pytest.raises(PointingError, match="fewer than 2 samples"):
            _generate(pattern, site, duration, needs_start)

    @pytest.mark.parametrize("pattern_factory, config, needs_start, obs_start", _PATTERNS)
    def test_sub_timestep_duration_raises(
        self, pattern_factory, config, needs_start, obs_start, site
    ):
        """A sub-timestep duration raises a clear PointingError, not IndexError."""
        pattern = pattern_factory()
        with pytest.raises(PointingError, match="fewer than 2 samples"):
            _generate(pattern, site, config.timestep / 2.0, needs_start)

    @pytest.mark.parametrize("pattern_factory, config, needs_start, obs_start", _PATTERNS)
    def test_normal_duration_still_works(
        self, pattern_factory, config, needs_start, obs_start, site
    ):
        """A normal multi-sample duration generates a >= 2-point trajectory."""
        pattern = pattern_factory()
        traj = _generate(pattern, site, 30.0, needs_start, obs_start)
        assert traj.n_points >= 2


class TestDegenerateDurationBuilder:
    """Degenerate durations are rejected at the public builder entry point."""

    @pytest.mark.parametrize("pattern_factory, config, needs_start, obs_start", _PATTERNS)
    def test_sub_timestep_duration_raises(
        self, pattern_factory, config, needs_start, obs_start, site
    ):
        """The builder rejects a sub-timestep duration with its >= 2-sample message."""
        builder = TrajectoryBuilder(site).with_config(config).duration(config.timestep / 2.0)
        # The builder.duration() guard rejects <= 0 before .build(); a positive
        # sub-timestep duration is caught by build()'s sample-count check.
        if needs_start:
            # PlanetTrackConfig is an AltAz pattern (ignores ra/dec); only call
            # .at() for patterns that consume sky coordinates.
            if not isinstance(config, PlanetTrackConfig):
                builder = builder.at(ra=_OBS_RA, dec=_OBS_DEC)
            builder = builder.starting_at(_START)
        with pytest.raises(ValueError, match="fewer than 2 samples"):
            builder.build()

    @pytest.mark.parametrize("pattern_factory, config, needs_start, obs_start", _PATTERNS)
    def test_normal_duration_still_works(
        self, pattern_factory, config, needs_start, obs_start, site
    ):
        """A normal duration builds a >= 2-point trajectory through the builder."""
        builder = TrajectoryBuilder(site).with_config(config).duration(30.0)
        if needs_start:
            if not isinstance(config, PlanetTrackConfig):
                builder = builder.at(ra=_OBS_RA, dec=_OBS_DEC)
            builder = builder.starting_at(obs_start)
        traj = builder.build()
        assert traj.n_points >= 2


class TestDaisyEqualsTimestepRaises:
    """Daisy uniquely loses a sample at duration == timestep via downsampling.

    On the nominal grid ``duration == timestep`` yields exactly 2 samples
    (valid for pong/linear/constant_el), but daisy's ``[::sample_every]``
    downsampling drops the final partial step, collapsing to 1 sample. That
    is the case daisy's half-guard lets through into ``np.gradient``, so it
    must raise a clear PointingError.
    """

    def test_daisy_generate_offsets_equals_timestep_raises(self):
        pattern = _daisy()
        with pytest.raises(PointingError, match="fewer than 2 samples"):
            pattern.generate_offsets(pattern.config.timestep)

    def test_daisy_generate_equals_timestep_raises(self, site):
        pattern = _daisy()
        with pytest.raises(PointingError, match="fewer than 2 samples"):
            pattern.generate(site, pattern.config.timestep, _START)


class TestAltAzNoSilentOnePoint:
    """linear and constant_el must not silently return a 1-point trajectory.

    These two patterns set velocities directly (no ``np.gradient``), so an
    unguarded degenerate duration produces a wrong 1-point Trajectory with no
    error, the worst failure mode. They must raise instead.
    """

    @pytest.mark.parametrize(
        "pattern_factory", [_linear, _constant_el], ids=["linear", "constant_el"]
    )
    @pytest.mark.parametrize("duration", [0.0, -1.0, _TIMESTEP / 2.0])
    def test_no_silent_one_point_trajectory(self, pattern_factory, duration, site):
        pattern = pattern_factory()
        with pytest.raises(PointingError, match="fewer than 2 samples"):
            pattern.generate(site, duration)
