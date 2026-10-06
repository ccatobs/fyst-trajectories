"""Tests for the custom exception hierarchy.

Verifies:
- Exception inheritance (every error is catchable as ValueError)
- Structured data on all exception types
- Pattern-level error wrapping (TargetNotObservableError)
- AltAz pattern direct bounds errors
- validate_trajectory() bounds errors
- pickle / copy round trips (the errors cross process boundaries)
- the rule that splits ``PointingError`` from plain ``ValueError``
"""

import copy
import pickle
import sys

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories import Coordinates, compute_source_ces_params
from fyst_trajectories.exceptions import (
    AccelerationLimitWarning,
    AzimuthBoundsError,
    DwellExceedsCrossingError,
    ElevationBoundsError,
    EncoderSolutionError,
    OffsetInversionError,
    PointingError,
    PointingWarning,
    TargetNotObservableError,
    TrajectoryBoundsError,
    VelocityLimitWarning,
)
from fyst_trajectories.observability import Target, TargetKind
from fyst_trajectories.patterns import (
    ConstantElScanConfig,
    ConstantElScanPattern,
    DaisyScanConfig,
    DaisyScanPattern,
    LinearMotionConfig,
    LinearMotionPattern,
    PongScanConfig,
    PongScanPattern,
    TrajectoryBuilder,
)
from fyst_trajectories.patterns.registry import register_pattern
from fyst_trajectories.patterns.utils import rewrap_trajectory_azimuth, validate_sample_count
from fyst_trajectories.planning import FieldRegion, plan_constant_el_scan
from fyst_trajectories.trajectory import Trajectory
from fyst_trajectories.trajectory_utils import validate_trajectory
from fyst_trajectories.visualization.sky_view import _boresight_altaz


class TestExceptionStructuredData:
    """Test that exceptions carry correct structured data."""

    def test_target_not_observable_attributes(self):
        """Test TargetNotObservableError has all structured attributes."""
        bounds = ElevationBoundsError(
            actual_min=-10.0,
            actual_max=15.0,
            limit_min=20.0,
            limit_max=90.0,
        )
        exc = TargetNotObservableError(
            target="RA=180.000 Dec=80.000",
            time_info="2026-03-15T04:00:00.000",
            bounds_error=bounds,
        )
        assert exc.target == "RA=180.000 Dec=80.000"
        assert exc.time_info == "2026-03-15T04:00:00.000"
        assert exc.bounds_error is bounds
        assert exc.bounds_error.axis == "elevation"
        assert exc.bounds_error.actual_min == -10.0

    def test_azimuth_bounds_error_message(self):
        """Test AzimuthBoundsError has a meaningful message."""
        exc = AzimuthBoundsError(
            actual_min=-300.0,
            actual_max=200.0,
            limit_min=-180.0,
            limit_max=360.0,
        )
        msg = str(exc)
        assert "azimuth" in msg
        assert "-300.00" in msg
        assert "200.00" in msg
        assert "[-180.0, 360.0]" in msg

    def test_elevation_bounds_error_message(self):
        """Test ElevationBoundsError has a meaningful message."""
        exc = ElevationBoundsError(
            actual_min=10.0,
            actual_max=85.0,
            limit_min=20.0,
            limit_max=90.0,
        )
        msg = str(exc)
        assert "elevation" in msg
        assert "10.00" in msg

    def test_target_not_observable_message(self):
        """Test TargetNotObservableError has a meaningful message."""
        bounds = ElevationBoundsError(
            actual_min=-10.0,
            actual_max=15.0,
            limit_min=20.0,
            limit_max=90.0,
        )
        exc = TargetNotObservableError(
            target="Mars",
            time_info="2026-03-15T04:00:00",
            bounds_error=bounds,
        )
        msg = str(exc)
        assert "Mars" in msg
        assert "2026-03-15T04:00:00" in msg
        assert "elevation" in msg

    def test_target_not_observable_wraps_original(self):
        """Test that TargetNotObservableError preserves the original bounds error."""
        bounds = AzimuthBoundsError(
            actual_min=-280.0,
            actual_max=100.0,
            limit_min=-180.0,
            limit_max=360.0,
        )
        exc = TargetNotObservableError(
            target="RA=350.0 Dec=-30.0",
            time_info="2026-06-15",
            bounds_error=bounds,
        )
        # The wrapped error is accessible and has the right type
        assert isinstance(exc.bounds_error, AzimuthBoundsError)
        assert isinstance(exc.bounds_error, TrajectoryBoundsError)
        assert exc.bounds_error.axis == "azimuth"
        assert exc.bounds_error.actual_min == -280.0


class TestLimitWarningHierarchy:
    """Lock the category contract for the structured limit warnings.

    A dispatch-time escalation filters on ``VelocityLimitWarning`` by
    *category* (``issubclass``), so these must remain ``PointingWarning``
    subclasses or every existing ``except PointingWarning`` /
    ``issubclass(.., PointingWarning)`` handler would silently stop catching
    them.
    """

    def test_velocity_limit_warning_is_pointing_warning(self):
        assert issubclass(VelocityLimitWarning, PointingWarning)

    def test_acceleration_limit_warning_is_pointing_warning(self):
        assert issubclass(AccelerationLimitWarning, PointingWarning)


class TestPongRaisesTargetNotObservable:
    """Test that PongScanPattern raises TargetNotObservableError for unobservable targets."""

    def test_pong_unobservable_high_dec(self, site):
        """Test that Pong raises TargetNotObservableError for target below horizon."""
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
        # Dec=+80 never visible from FYST (lat -22.99)
        pattern = PongScanPattern(ra=180.0, dec=80.0, config=config)

        with pytest.raises(TargetNotObservableError) as exc_info:
            pattern.generate(site, duration=300.0, start_time=start_time)

        exc = exc_info.value
        assert "RA=180.000" in exc.target
        assert "Dec=80.000" in exc.target
        assert exc.time_info == start_time.iso
        assert isinstance(exc.bounds_error, TrajectoryBoundsError)

    def test_pong_unobservable_is_catchable_as_valueerror(self, site):
        """A TargetNotObservableError is catchable as a plain ValueError."""
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
        pattern = PongScanPattern(ra=180.0, dec=80.0, config=config)

        with pytest.raises(ValueError):
            pattern.generate(site, duration=300.0, start_time=start_time)


class TestDaisyRaisesTargetNotObservable:
    """Test that DaisyScanPattern raises TargetNotObservableError for unobservable targets."""

    def test_daisy_unobservable_high_dec(self, site):
        """Test that Daisy raises TargetNotObservableError for target below horizon."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.0,
        )
        # Dec=+80 never visible from FYST (lat -22.99)
        pattern = DaisyScanPattern(ra=180.0, dec=80.0, config=config)

        with pytest.raises(TargetNotObservableError) as exc_info:
            pattern.generate(site, duration=300.0, start_time=start_time)

        exc = exc_info.value
        assert "RA=180.000" in exc.target
        assert "Dec=80.000" in exc.target
        assert isinstance(exc.bounds_error, TrajectoryBoundsError)


class TestConstantElRaisesBoundsError:
    """Test that ConstantElScanPattern raises bounds errors directly."""

    def test_elevation_below_limit(self, site):
        """Test ElevationBoundsError for elevation below minimum.

        A ``TargetNotObservableError`` would fail the ``raises`` check: it is not an
        ``ElevationBoundsError``.
        """
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=120.0,
            az_stop=180.0,
            elevation=15.0,
            az_speed=1.0,
            az_accel=0.5,
        )
        pattern = ConstantElScanPattern(config)

        with pytest.raises(ElevationBoundsError) as exc_info:
            pattern.generate(site, duration=120.0, start_time=None)

        exc = exc_info.value
        assert exc.axis == "elevation"
        assert exc.actual_min == 15.0
        assert exc.actual_max == 15.0
        assert exc.limit_min == site.telescope_limits.elevation.min

    def test_azimuth_out_of_range(self, site):
        """Test AzimuthBoundsError for azimuth exceeding limits."""
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=-280.0,
            az_stop=-260.0,
            elevation=45.0,
            az_speed=1.0,
            az_accel=0.5,
        )
        pattern = ConstantElScanPattern(config)

        with pytest.raises(AzimuthBoundsError) as exc_info:
            pattern.generate(site, duration=120.0, start_time=None)

        exc = exc_info.value
        assert exc.axis == "azimuth"
        assert exc.actual_min < site.telescope_limits.azimuth.min


class TestLinearRaisesBoundsError:
    """Test that LinearMotionPattern raises bounds errors directly."""

    def test_elevation_exceeds_limit(self, site):
        """Test ElevationBoundsError when linear motion goes above max elevation."""
        config = LinearMotionConfig(
            timestep=0.1,
            az_start=100.0,
            el_start=85.0,
            az_velocity=0.0,
            el_velocity=1.0,
        )
        pattern = LinearMotionPattern(config)

        start_time = Time("2026-03-15T04:00:00", scale="utc")
        with pytest.raises(ElevationBoundsError) as exc_info:
            pattern.generate(site, duration=60.0, start_time=start_time)

        exc = exc_info.value
        assert exc.axis == "elevation"
        assert exc.actual_max > site.telescope_limits.elevation.max


class TestBuilderRaisesExceptions:
    """Test that TrajectoryBuilder propagates exceptions correctly."""

    def test_builder_propagates_target_not_observable(self, site):
        """Test that builder propagates TargetNotObservableError from celestial patterns."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")

        with pytest.raises(TargetNotObservableError) as exc_info:
            TrajectoryBuilder(site).at(
                ra=180.0,
                dec=80.0,
            ).with_config(
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
        assert "180.000" in exc_info.value.target
        assert "80.000" in exc_info.value.target

    def test_builder_propagates_elevation_bounds_error(self, site):
        """Test that builder propagates ElevationBoundsError from AltAz patterns."""
        with pytest.raises(ElevationBoundsError) as exc_info:
            TrajectoryBuilder(site).with_config(
                ConstantElScanConfig(
                    timestep=0.1,
                    az_start=120.0,
                    az_stop=180.0,
                    elevation=15.0,
                    az_speed=1.0,
                    az_accel=0.5,
                )
            ).duration(120.0).build()
        assert exc_info.value.actual_min == 15.0
        assert exc_info.value.limit_min == 20.0


class TestTrajectoryValidateExceptions:
    """Test that validate_trajectory() raises the correct exceptions."""

    def test_validate_azimuth_out_of_bounds(self, site):
        """Test that validate raises AzimuthBoundsError for out-of-range azimuth."""
        traj = Trajectory(
            times=np.array([0.0, 1.0, 2.0]),
            az=np.array([100.0, 110.0, 400.0]),
            el=np.full(3, 45.0),
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
        )
        with pytest.raises(AzimuthBoundsError) as exc_info:
            validate_trajectory(traj, site)

        exc = exc_info.value
        assert exc.axis == "azimuth"
        assert exc.actual_max == 400.0

    def test_validate_elevation_out_of_bounds(self, site):
        """Test that validate raises ElevationBoundsError for out-of-range elevation."""
        traj = Trajectory(
            times=np.array([0.0, 1.0, 2.0]),
            az=np.array([100.0, 110.0, 120.0]),
            el=np.array([10.0, 45.0, 50.0]),
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
        )
        with pytest.raises(ElevationBoundsError) as exc_info:
            validate_trajectory(traj, site)

        exc = exc_info.value
        assert exc.axis == "elevation"
        assert exc.actual_min == 10.0

    def test_validate_catchable_as_pointing_error(self, site):
        """Test that validate errors are catchable as PointingError."""
        traj = Trajectory(
            times=np.array([0.0, 1.0, 2.0]),
            az=np.array([100.0, 110.0, 400.0]),
            el=np.full(3, 45.0),
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
        )
        with pytest.raises(PointingError):
            validate_trajectory(traj, site)


class TestExceptionChaining:
    """Test that exception chaining suppresses noisy tracebacks."""

    def test_target_not_observable_suppresses_chained_traceback(self, site):
        """Test that TargetNotObservableError suppresses the chained traceback.

        The ``raise ... from None`` pattern suppresses the inner
        TrajectoryBoundsError traceback to keep error output clean.
        The original error is still accessible via the ``bounds_error``
        attribute for programmatic inspection.
        """
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
        pattern = PongScanPattern(ra=180.0, dec=80.0, config=config)

        with pytest.raises(TargetNotObservableError) as exc_info:
            pattern.generate(site, duration=300.0, start_time=start_time)

        exc = exc_info.value
        # __cause__ is None because we use "raise ... from None" to suppress
        # the chained traceback for cleaner error output
        assert exc.__cause__ is None
        # __suppress_context__ is True when "from None" is used
        assert exc.__suppress_context__ is True
        # The original bounds error is still available via the attribute
        assert isinstance(exc.bounds_error, TrajectoryBoundsError)


def _structured_errors():
    """One instance of every structured exception, with distinctive fields."""
    bounds = AzimuthBoundsError(-190.0, 370.0, -180.0, 360.0)
    return [
        TrajectoryBoundsError("elevation", 10.0, 95.0, 20.0, 90.0),
        bounds,
        ElevationBoundsError(10.0, 95.0, 20.0, 90.0),
        TargetNotObservableError("mars", "2026-06-15T04:00:00", bounds),
        EncoderSolutionError(
            "sun_blocked",
            "every wrap is blocked",
            goal_az=200.0,
            goal_el=45.0,
            current_az=10.0,
            current_el=30.0,
            candidates=[-160.0, 200.0],
            time_iso="2026-06-15T04:00:00",
        ),
        OffsetInversionError("degenerate at the pole", indices=[2, 7]),
        DwellExceedsCrossingError(
            "dwell must not exceed the solved footprint crossing",
            dwell=641.35,
            crossing_seconds=581.35,
        ),
    ]


def _assert_same_error(restored, original):
    """Assert two exception instances carry the same type, message and fields."""
    assert type(restored) is type(original)
    assert str(restored) == str(original)
    for name, value in vars(original).items():
        got = getattr(restored, name)
        if isinstance(value, BaseException):
            # Exceptions have no value equality; compare type and message.
            assert type(got) is type(value)
            assert str(got) == str(value)
        else:
            assert got == value, name


@pytest.mark.parametrize("error", _structured_errors(), ids=lambda e: type(e).__name__)
def test_structured_errors_survive_pickle(error):
    """Every structured exception reconstructs through ``pickle``.

    ``BaseException`` reconstructs by calling the class with ``self.args``,
    which for these is just the composed message; the extra constructor
    arguments made that a ``TypeError`` on unpickle. Each class defines
    ``__reduce__`` instead. This matters wherever an error crosses a process
    boundary, which is what a control system marshalling a task failure does.
    """
    _assert_same_error(pickle.loads(pickle.dumps(error)), error)


@pytest.mark.parametrize("error", _structured_errors(), ids=lambda e: type(e).__name__)
def test_structured_errors_survive_deepcopy(error):
    """The same reconstruction path serves ``copy.deepcopy``."""
    _assert_same_error(copy.deepcopy(error), error)


@pytest.mark.parametrize("error", _structured_errors(), ids=lambda e: type(e).__name__)
def test_structured_errors_survive_copy(error):
    """The same reconstruction path serves ``copy.copy``."""
    _assert_same_error(copy.copy(error), error)


_ROUND_TRIPS = {
    "pickle": lambda error: pickle.loads(pickle.dumps(error)),
    "copy": copy.copy,
    "deepcopy": copy.deepcopy,
}


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="BaseException.add_note is new in Python 3.11"
)
@pytest.mark.parametrize("round_trip", sorted(_ROUND_TRIPS))
@pytest.mark.parametrize(
    "index",
    range(len(_structured_errors())),
    ids=[type(e).__name__ for e in _structured_errors()],
)
def test_structured_errors_keep_notes(index, round_trip):
    """Notes added with ``add_note`` survive ``pickle``, ``copy`` and ``deepcopy``."""
    error = _structured_errors()[index]
    error.add_note("dispatched from task 7")
    restored = _ROUND_TRIPS[round_trip](error)
    assert restored.__notes__ == ["dispatched from task 7"]
    _assert_same_error(restored, error)


@pytest.mark.parametrize("error", _structured_errors(), ids=lambda e: type(e).__name__)
def test_structured_errors_are_pointing_errors(error):
    """Every structured error is a ``PointingError`` and so a ``ValueError``.

    Downstream ``except ValueError`` handlers and the catch order that takes
    ``PointingError`` first both depend on this hierarchy.
    """
    assert isinstance(error, PointingError)
    assert isinstance(error, ValueError)


# ---------------------------------------------------------------------------
# The rule: PointingError for a well-formed request that cannot be satisfied
# for this site, target and time; plain ValueError for a malformed request.
# Every row is a raise site whose side the rule decides.
# ---------------------------------------------------------------------------

_NIGHT = Time("2026-03-15T00:00:00", scale="utc")
_ECDFS = FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7)


class _RuleTablePattern:
    """A stand-in pattern class for the registration rows; never registered."""


def _rewrap_by_a_partial_turn(site):
    traj = Trajectory(
        times=np.arange(3, dtype=float),
        az=np.array([100.0, 101.0, 102.0]),
        el=np.full(3, 45.0),
        az_vel=np.ones(3),
        el_vel=np.zeros(3),
    )
    rewrap_trajectory_azimuth(traj, 90.0)


def _daisy_offsets_one_timestep(site):
    pattern = DaisyScanPattern(
        ra=180.0,
        dec=-30.0,
        config=DaisyScanConfig(
            timestep=0.1,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            y_offset=0.0,
        ),
    )
    pattern.generate_offsets(pattern.config.timestep)


def _constant_el(site, field, **kwargs):
    kwargs.setdefault("start_time", _NIGHT)
    kwargs.setdefault("rising", True)
    plan_constant_el_scan(field=field, elevation=50.0, velocity=0.5, site=site, **kwargs)


def _below_horizon_boresight(site):
    coords = Coordinates(site)
    ra, dec = coords.altaz_to_radec(0.0, -45.0, _NIGHT)
    down = Target("down_under", TargetKind.FIXED, ra_deg=float(ra), dec_deg=float(dec))
    _boresight_altaz(down, coords, _NIGHT, None)


_MALFORMED = [
    pytest.param(
        lambda site: validate_sample_count(10.0, 0.0),
        "timestep must be positive",
        id="sample_count_timestep",
    ),
    pytest.param(
        lambda site: validate_sample_count(float("nan"), 0.1),
        "duration must be finite",
        id="sample_count_duration",
    ),
    pytest.param(
        lambda site: validate_sample_count(0.01, 0.1),
        "fewer than 2 samples",
        id="sample_count_too_short",
    ),
    pytest.param(_daisy_offsets_one_timestep, "fewer than 2 samples", id="daisy_offsets_too_short"),
    pytest.param(_rewrap_by_a_partial_turn, "whole multiple of 360", id="rewrap_partial_turn"),
    pytest.param(lambda site: register_pattern(""), "non-blank string", id="register_blank_name"),
    pytest.param(
        lambda site: register_pattern("pong")(_RuleTablePattern),
        "already registered",
        id="register_duplicate_name",
    ),
    pytest.param(
        lambda site: register_pattern("rule_table_pattern", config=PongScanConfig)(
            _RuleTablePattern
        ),
        "already mapped",
        id="register_duplicate_config",
    ),
    pytest.param(
        lambda site: TrajectoryBuilder(site).with_config(3.0),
        "^Unknown config type",
        id="builder_unknown_config",
    ),
]

_UNSATISFIABLE = [
    pytest.param(
        lambda site: _constant_el(
            site, FieldRegion(ra_center=100.0, dec_center=-89.6, width=0.5, height=0.2)
        ),
        "too close to celestial pole",
        id="constant_el_near_pole",
    ),
    pytest.param(
        lambda site: _constant_el(
            site, FieldRegion(ra_center=180.0, dec_center=70.0, width=1.0, height=1.0)
        ),
        "Could not find elevation crossing",
        id="constant_el_no_crossing",
    ),
    pytest.param(
        lambda site: _constant_el(
            site,
            _ECDFS,
            start_time=Time("2026-03-15T17:34:26.717", scale="utc"),
            angle=170.0,
            max_search_hours=30.0,
        ),
        "different passes",
        id="constant_el_different_passes",
    ),
    pytest.param(
        lambda site: compute_source_ces_params(
            body="jupiter",
            footprint="c",
            el_bore=35.0,
            night=_NIGHT,
            mode="rising",
            site=site,
            dwell=1.0e5,
        ),
        "dwell must not exceed the solved footprint crossing",
        id="source_ces_dwell_exceeds_crossing",
    ),
    pytest.param(
        _below_horizon_boresight, "below the horizon", id="sky_view_boresight_below_horizon"
    ),
]


@pytest.mark.parametrize("call, match", _MALFORMED)
def test_a_malformed_request_raises_plain_value_error(call, match, site):
    """An argument the caller got wrong is a plain ``ValueError``.

    A handler that catches ``PointingError`` to defer an infeasible request
    must not also swallow a programming error, so none of these is a
    ``PointingError``.
    """
    with pytest.raises(ValueError, match=match) as exc_info:
        call(site)
    assert not isinstance(exc_info.value, PointingError)


@pytest.mark.parametrize("call, match", _UNSATISFIABLE)
def test_an_unsatisfiable_request_raises_pointing_error(call, match, site):
    """A well-formed request this site, target and time cannot meet is a ``PointingError``."""
    with pytest.raises(PointingError, match=match):
        call(site)
