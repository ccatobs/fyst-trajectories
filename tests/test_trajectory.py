"""Tests for the Trajectory container and its export and validation helpers."""

import ast
import copy
import dataclasses
import io
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.time import Time

import fyst_trajectories
import fyst_trajectories.trajectory
from fyst_trajectories import (
    SCAN_FLAG_SCIENCE,
    SCAN_FLAG_TURNAROUND,
    SCAN_FLAG_UNCLASSIFIED,
    Trajectory,
    TrajectoryBuilder,
    get_fyst_site,
    print_trajectory,
)
from fyst_trajectories.exceptions import (
    AzimuthBoundsError,
    PointingWarning,
    VelocityLimitWarning,
)
from fyst_trajectories.patterns import ConstantElScanConfig, TrajectoryMetadata
from fyst_trajectories.trajectory_utils import (
    _format_trajectory,
    get_absolute_times,
    to_arrays,
    to_path_format,
    to_path_payload,
    to_trackpoint_format,
    validate_trajectory,
)


class TestTrajectory:
    """Construction, metadata access, the export formats, and the validators."""

    def test_trajectory_creation(self):
        times = np.array([0, 1, 2, 3, 4], dtype=float)
        az = np.array([100, 101, 102, 101, 100], dtype=float)
        el = np.full(5, 45.0)
        az_vel = np.array([1, 1, 0, -1, -1], dtype=float)
        el_vel = np.zeros(5)

        traj = Trajectory(
            times=times,
            az=az,
            el=el,
            az_vel=az_vel,
            el_vel=el_vel,
        )

        assert traj.n_points == 5
        assert traj.duration == 4.0

    def test_trajectory_with_metadata(self):
        times = np.array([0, 1, 2], dtype=float)
        metadata = TrajectoryMetadata(
            pattern_type="test_pattern",
            pattern_params={"width": 2.0, "height": 1.0},
            center_ra=180.0,
            center_dec=-30.0,
        )

        traj = Trajectory(
            times=times,
            az=np.zeros(3),
            el=np.full(3, 45.0),
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
            metadata=metadata,
        )

        assert traj.pattern_type == "test_pattern"
        assert traj.pattern_params == {"width": 2.0, "height": 1.0}
        assert traj.center_ra == 180.0
        assert traj.center_dec == -30.0

    def test_absolute_times_with_start(self):
        times = np.array([0, 1, 2], dtype=float)
        start = Time("2026-03-15T04:00:00", scale="utc")

        traj = Trajectory(
            times=times,
            az=np.zeros(3),
            el=np.zeros(3),
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
            start_time=start,
        )

        abs_times = get_absolute_times(traj)
        assert len(abs_times) == 3
        assert abs_times[0] == start

    def test_absolute_times_without_start_raises(self):
        traj = Trajectory(
            times=np.array([0, 1, 2], dtype=float),
            az=np.zeros(3),
            el=np.zeros(3),
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
        )

        with pytest.raises(ValueError, match="start_time not set"):
            get_absolute_times(traj)

    def test_to_arrays(self):
        """Exported arrays are copies: mutating one leaves the trajectory unchanged."""
        times = np.array([0.0, 1.0, 2.0])
        az = np.array([100.0, 110.0, 120.0])
        el = np.array([45.0, 46.0, 47.0])

        traj = Trajectory(
            times=times,
            az=az,
            el=el,
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
        )

        t_out, az_out, el_out = to_arrays(traj)

        np.testing.assert_array_equal(t_out, times)
        np.testing.assert_array_equal(az_out, az)
        np.testing.assert_array_equal(el_out, el)

        t_out[0] = 999  # Modifying copy should not affect original
        assert traj.times[0] == 0.0

    def test_to_path_format(self):
        times = np.array([0.0, 1.0])
        az = np.array([100.0, 110.0])
        el = np.array([45.0, 46.0])
        az_vel = np.array([10.0, 10.0])
        el_vel = np.array([1.0, 1.0])

        traj = Trajectory(
            times=times,
            az=az,
            el=el,
            az_vel=az_vel,
            el_vel=el_vel,
        )

        path = to_path_format(traj)

        assert len(path) == 2
        assert path[0] == [0.0, 100.0, 45.0, 10.0, 1.0]
        assert path[1] == [1.0, 110.0, 46.0, 10.0, 1.0]

    def test_to_path_format_nonzero_time_origin(self):
        """Row times are re-zeroed to ``times[0]`` so the first row lands on start_time.

        A trajectory whose clock does not begin at 0 (a sliced or re-based
        one) must not be commanded ``times[0]`` late: the /path body's
        ``start_time`` is absolute and the rows are relative to it.
        """
        traj = Trajectory(
            times=np.array([100.0, 101.0]),
            az=np.array([100.0, 110.0]),
            el=np.array([45.0, 46.0]),
            az_vel=np.array([10.0, 10.0]),
            el_vel=np.array([1.0, 1.0]),
        )
        path = to_path_format(traj)
        assert path[0][0] == 0.0
        assert path[1][0] == 1.0

    def _payload_traj(self, with_start_time=True):
        """Build a small 2-point trajectory for /path payload tests."""
        kwargs = dict(
            times=np.array([0.0, 1.0]),
            az=np.array([100.0, 110.0]),
            el=np.array([45.0, 46.0]),
            az_vel=np.array([10.0, 10.0]),
            el_vel=np.array([1.0, 1.0]),
        )
        if with_start_time:
            kwargs["start_time"] = Time("2026-05-28T00:00:00", scale="utc")
        return Trajectory(**kwargs)

    def test_to_path_payload(self):
        traj = self._payload_traj()

        payload = to_path_payload(traj)

        assert set(payload) == {"start_time", "coordsys", "points"}
        assert payload["coordsys"] == "Horizon"
        assert payload["start_time"] == traj.start_time.unix
        assert payload["points"] == to_path_format(traj)

    def test_to_path_payload_rejects_unknown_coordsys(self):
        with pytest.raises(ValueError, match="coordsys must be"):
            to_path_payload(self._payload_traj(), coordsys="galactic")

    def test_to_path_payload_requires_start_time(self):
        with pytest.raises(ValueError, match="start_time not set"):
            to_path_payload(self._payload_traj(with_start_time=False))

    def test_to_path_payload_warns_on_icrs(self):
        """coordsys='ICRS' warns: Go TCS ICRS /path velocities are unimplemented."""
        with pytest.warns(PointingWarning, match="ICRS"):
            payload = to_path_payload(self._payload_traj(), coordsys="ICRS")
        assert payload["coordsys"] == "ICRS"

    def _fine_traj(self, dt, with_start_time=False):
        """Build a 3-point trajectory with sample spacing ``dt`` seconds."""
        kwargs = dict(
            times=np.array([0.0, dt, 2.0 * dt]),
            az=np.array([100.0, 100.1, 100.2]),
            el=np.array([45.0, 45.0, 45.0]),
            az_vel=np.array([5.0, 5.0, 5.0]),
            el_vel=np.array([0.0, 0.0, 0.0]),
        )
        if with_start_time:
            kwargs["start_time"] = Time("2026-05-28T00:00:00", scale="utc")
        return Trajectory(**kwargs)

    def test_to_path_format_rejects_sub_50ms_interval(self):
        """A sample interval below the Go TCS /path 50 ms minimum raises."""
        traj = self._fine_traj(dt=0.02)  # 20 ms < 50 ms
        with pytest.raises(ValueError, match="Go TCS .*minimum"):
            to_path_format(traj)

    def test_to_path_payload_rejects_sub_50ms_interval(self):
        """``to_path_payload`` inherits the 50 ms interval check."""
        traj = self._fine_traj(dt=0.02, with_start_time=True)
        with pytest.raises(ValueError, match="Go TCS .*minimum"):
            to_path_payload(traj)

    def test_to_path_format_accepts_50ms_interval(self):
        """Exactly 50 ms is allowed (boundary)."""
        traj = self._fine_traj(dt=0.05)
        assert len(to_path_format(traj)) == 3

    def test_builder_grid_at_50ms_is_refused_with_its_true_interval(self):
        """A builder grid at ``timestep=0.05`` carries round-off just below 50 ms.

        Go TCS refuses it too, so the export does; the message prints the
        interval at full precision rather than rounding it back to 0.05.
        """
        config = ConstantElScanConfig(
            timestep=0.05,
            az_start=100.0,
            az_stop=110.0,
            elevation=45.0,
            az_speed=1.0,
            az_accel=1.0,
        )
        traj = TrajectoryBuilder(get_fyst_site()).with_config(config).duration(600.0).build()
        with pytest.raises(ValueError, match=r"interval 0\.04999999\d* s is below"):
            to_path_format(traj)

    def _flagged_traj(self, scan_flag, start_time=True):
        """Build a trajectory with an explicit scan_flag pattern for TrackPoint tests."""
        n = len(scan_flag)
        kwargs = dict(
            times=np.arange(n, dtype=float),
            az=np.arange(n, dtype=float) + 100.0,
            el=np.full(n, 45.0),
            az_vel=np.full(n, 10.0),
            el_vel=np.zeros(n),
            scan_flag=np.array(scan_flag, dtype=np.int8),
        )
        if start_time:
            kwargs["start_time"] = Time("2026-05-28T00:00:00", scale="utc")
        return Trajectory(**kwargs)

    def test_to_trackpoint_format_flags_and_timestamps(self):
        """az_flag/group_flag follow the SO ACU convention; timestamps are absolute Unix."""
        # A 5-point science leg, a turnaround, a 2-point science leg (shorter
        # than TRACKPOINT_NEW_LEG_GROUP_SIZE), then two more turnaround points.
        # The short leg is deliberately NOT last: with a trailing sample the
        # group countdown has somewhere to leak to, so the "clear at the leg
        # end" rule is actually exercised.
        sf = [
            SCAN_FLAG_SCIENCE,
            SCAN_FLAG_SCIENCE,
            SCAN_FLAG_SCIENCE,
            SCAN_FLAG_SCIENCE,
            SCAN_FLAG_SCIENCE,
            SCAN_FLAG_TURNAROUND,
            SCAN_FLAG_SCIENCE,
            SCAN_FLAG_SCIENCE,
            SCAN_FLAG_TURNAROUND,
            SCAN_FLAG_TURNAROUND,
        ]
        traj = self._flagged_traj(sf)
        rows = to_trackpoint_format(traj)

        assert len(rows) == 10
        assert set(rows[0]) == {
            "timestamp",
            "az",
            "el",
            "az_vel",
            "el_vel",
            "az_flag",
            "el_flag",
            "group_flag",
        }
        # az_flag: interior science=1, final point of a science leg=2, non-science=0.
        assert [r["az_flag"] for r in rows] == [1, 1, 1, 1, 2, 0, 1, 2, 0, 0]
        # group_flag: the first TRACKPOINT_NEW_LEG_GROUP_SIZE (4) points of each
        # science leg, and nothing outside a science leg.
        assert [r["group_flag"] for r in rows] == [1, 1, 1, 1, 0, 0, 1, 1, 0, 0]
        assert all(r["el_flag"] == 0 for r in rows)
        # Absolute Unix timestamps = start_time.unix + relative trajectory times.
        t0 = traj.start_time.unix
        assert rows[0]["timestamp"] == pytest.approx(t0 + 0.0)
        assert rows[5]["timestamp"] == pytest.approx(t0 + 5.0)

    def test_to_trackpoint_format_nonzero_time_origin(self):
        """A nonzero ``times[0]`` maps the first row exactly onto start_time.unix."""
        traj = Trajectory(
            times=np.array([100.0, 101.0]),
            az=np.array([100.0, 110.0]),
            el=np.array([45.0, 46.0]),
            az_vel=np.array([10.0, 10.0]),
            el_vel=np.array([1.0, 1.0]),
            start_time=Time("2026-05-28T00:00:00", scale="utc"),
        )
        rows = to_trackpoint_format(traj)
        assert rows[0]["timestamp"] == pytest.approx(traj.start_time.unix)
        assert rows[1]["timestamp"] == pytest.approx(traj.start_time.unix + 1.0)

    def test_to_trackpoint_format_requires_start_time(self):
        """A missing start_time raises ValueError (timestamps are absolute)."""
        traj = self._flagged_traj([SCAN_FLAG_SCIENCE, SCAN_FLAG_SCIENCE], start_time=False)
        with pytest.raises(ValueError, match="start_time not set"):
            to_trackpoint_format(traj)

    def test_to_trackpoint_format_unflagged_all_zero(self):
        traj = Trajectory(
            times=np.array([0.0, 1.0, 2.0]),
            az=np.array([100.0, 101.0, 102.0]),
            el=np.full(3, 45.0),
            az_vel=np.full(3, 10.0),
            el_vel=np.zeros(3),
            start_time=Time("2026-05-28T00:00:00", scale="utc"),
        )
        rows = to_trackpoint_format(traj)
        assert all(r["az_flag"] == 0 and r["group_flag"] == 0 for r in rows)

    def test_get_absolute_times_nonzero_origin(self):
        """get_absolute_times maps the first sample to start_time even when times[0] != 0."""
        t0 = Time("2026-05-28T00:00:00", scale="utc")
        traj = Trajectory(
            times=np.array([100.0, 101.0, 102.0]),  # clock does not start at 0
            az=np.array([10.0, 11.0, 12.0]),
            el=np.full(3, 45.0),
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
            start_time=t0,
        )
        abs_times = get_absolute_times(traj)
        # First sample == start_time (not start_time + 100 s); spacing preserved.
        assert float((abs_times[0] - t0).sec) == pytest.approx(0.0, abs=1e-6)
        assert float((abs_times[-1] - t0).sec) == pytest.approx(2.0, abs=1e-6)

    def test_array_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Array length mismatch"):
            Trajectory(
                times=np.array([0.0, 1.0, 2.0]),
                az=np.array([100.0, 110.0]),  # 2 instead of 3
                el=np.full(3, 45.0),
                az_vel=np.zeros(3),
                el_vel=np.zeros(3),
            )

    def test_validate_within_limits(self):
        site = get_fyst_site()
        traj = Trajectory(
            times=np.array([0.0, 1.0, 2.0, 3.0]),
            az=np.array([100.0, 101.0, 102.0, 103.0]),
            el=np.full(4, 45.0),
            az_vel=np.full(4, 1.0),
            el_vel=np.zeros(4),
        )
        validate_trajectory(traj, site)

    def test_validate_out_of_bounds_raises(self):
        site = get_fyst_site()
        traj = Trajectory(
            times=np.array([0.0, 1.0, 2.0]),
            az=np.array([100.0, 110.0, 400.0]),  # 400 > 360 limit
            el=np.full(3, 45.0),
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
        )
        with pytest.raises(AzimuthBoundsError, match="azimuth"):
            validate_trajectory(traj, site)

    def test_validate_warns_on_high_velocity(self):
        site = get_fyst_site()
        traj = Trajectory(
            times=np.linspace(0, 10, 100),
            az=100.0 + 10.0 * np.linspace(0, 10, 100),  # 10 deg/s
            el=np.full(100, 45.0),
            az_vel=np.full(100, 10.0),
            el_vel=np.zeros(100),
        )
        with pytest.warns(VelocityLimitWarning, match="azimuth velocity"):
            validate_trajectory(traj, site)

    def test_trajectory_is_frozen(self):
        """Trajectory is immutable: rebinding raises, replace works, dtype coerces."""
        traj = Trajectory(
            times=np.array([0, 1, 2], dtype=float),
            az=np.array([100, 101, 102], dtype=float),
            el=np.full(3, 45.0),
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            traj.az = np.zeros(traj.n_points)
        start = Time("2026-03-15T04:00:00", scale="utc")
        with pytest.raises(dataclasses.FrozenInstanceError):
            traj.start_time = start

        assert dataclasses.replace(traj, start_time=start).start_time is start

        flagged = Trajectory(
            times=np.array([0, 1, 2], dtype=float),
            az=np.array([100, 101, 102], dtype=float),
            el=np.full(3, 45.0),
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
            scan_flag=np.array([1, 0, 2]),
        )
        assert flagged.scan_flag.dtype == np.int8

    def test_trajectory_field_order(self):
        """The field order fixes which field a positional argument binds to."""
        assert [f.name for f in dataclasses.fields(Trajectory)] == [
            "times",
            "az",
            "el",
            "az_vel",
            "el_vel",
            "start_time",
            "metadata",
            "scan_flag",
            "retune_events",
        ]

    def test_trajectory_compares_and_hashes_by_identity(self):
        """Equality and hashing are by identity, on every interpreter.

        The generated ``==`` compared the arrays: it raised ``ValueError`` for
        two separately built instances, returned ``True`` for a replace copy
        on Python 3.10 to 3.12 and raised on 3.13, and ``hash`` raised.
        """
        traj = Trajectory(
            times=np.array([0, 1, 2], dtype=float),
            az=np.array([100, 101, 102], dtype=float),
            el=np.full(3, 45.0),
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
        )
        twin = Trajectory(
            times=traj.times,
            az=traj.az,
            el=traj.el,
            az_vel=traj.az_vel,
            el_vel=traj.el_vel,
        )

        assert isinstance(hash(traj), int)
        assert traj in {traj}
        assert traj == traj
        assert traj != dataclasses.replace(traj)
        assert traj != twin
        assert len({traj, twin}) == 2


class TestAccelerationJerkProperties:
    """Shapes, and the accelerations and jerks recovered from known velocity profiles."""

    def _make_constant_velocity_trajectory(self):
        """Create a trajectory with constant velocity (zero acceleration)."""
        times = np.linspace(0, 10, 101)
        az = 100.0 + 1.0 * times  # 1 deg/s constant
        el = 45.0 + 0.5 * times  # 0.5 deg/s constant
        az_vel = np.full_like(times, 1.0)
        el_vel = np.full_like(times, 0.5)
        return Trajectory(times=times, az=az, el=el, az_vel=az_vel, el_vel=el_vel)

    def _make_accelerating_trajectory(self):
        """Create a trajectory with constant acceleration."""
        times = np.linspace(0, 10, 101)
        # Constant acceleration of 0.2 deg/s^2 in az, 0.1 in el
        az_vel = 1.0 + 0.2 * times
        el_vel = 0.5 + 0.1 * times
        az = 100.0 + 1.0 * times + 0.5 * 0.2 * times**2
        el = 45.0 + 0.5 * times + 0.5 * 0.1 * times**2
        return Trajectory(times=times, az=az, el=el, az_vel=az_vel, el_vel=el_vel)

    def test_accel_returns_correct_shape(self):
        traj = self._make_constant_velocity_trajectory()
        assert traj.az_accel.shape == traj.times.shape
        assert traj.el_accel.shape == traj.times.shape

    def test_jerk_returns_correct_shape(self):
        traj = self._make_constant_velocity_trajectory()
        assert traj.az_jerk.shape == traj.times.shape
        assert traj.el_jerk.shape == traj.times.shape

    def test_constant_velocity_has_zero_acceleration(self):
        traj = self._make_constant_velocity_trajectory()
        np.testing.assert_allclose(traj.az_accel, 0.0, atol=1e-10)
        np.testing.assert_allclose(traj.el_accel, 0.0, atol=1e-10)

    def test_constant_velocity_has_zero_jerk(self):
        traj = self._make_constant_velocity_trajectory()
        np.testing.assert_allclose(traj.az_jerk, 0.0, atol=1e-10)
        np.testing.assert_allclose(traj.el_jerk, 0.0, atol=1e-10)

    def test_constant_acceleration_value(self):
        traj = self._make_accelerating_trajectory()
        np.testing.assert_allclose(traj.az_accel, 0.2, atol=1e-10)
        np.testing.assert_allclose(traj.el_accel, 0.1, atol=1e-10)

    def test_constant_acceleration_has_zero_jerk(self):
        traj = self._make_accelerating_trajectory()
        np.testing.assert_allclose(traj.az_jerk, 0.0, atol=1e-10)
        np.testing.assert_allclose(traj.el_jerk, 0.0, atol=1e-10)

    def test_jerk_matches_the_analytic_value(self):
        # Use a trajectory where acceleration varies (quadratic velocity)
        times = np.linspace(0, 5, 501)
        az_vel = 0.1 * times**2  # accel = 0.2*t, jerk = 0.2
        el_vel = 0.05 * times**2  # accel = 0.1*t, jerk = 0.1
        az = np.cumsum(az_vel) * (times[1] - times[0])
        el = 45.0 + np.cumsum(el_vel) * (times[1] - times[0])
        traj = Trajectory(times=times, az=az, el=el, az_vel=az_vel, el_vel=el_vel)

        # np.gradient is second-order in the interior and first-order at the two
        # edge samples, so the analytic value holds away from the edges.
        np.testing.assert_allclose(traj.az_jerk[2:-2], 0.2, atol=1e-10)
        np.testing.assert_allclose(traj.el_jerk[2:-2], 0.1, atol=1e-10)

    def test_single_sample_derivatives_are_zeros(self):
        """A one-sample trajectory has no time step, so its derivatives are zeros."""
        traj = Trajectory(
            times=np.array([5.0]),
            az=np.array([10.0]),
            el=np.array([40.0]),
            az_vel=np.array([1.0]),
            el_vel=np.array([0.5]),
        )
        for name in ("az_accel", "el_accel", "az_jerk", "el_jerk"):
            value = getattr(traj, name)
            assert value.shape == (1,), name
            np.testing.assert_array_equal(value, np.zeros(1), err_msg=name)

    @pytest.mark.parametrize("n", [2, 3, 101])
    def test_multi_sample_derivatives_are_np_gradient(self, n):
        """Two or more samples give exactly ``np.gradient`` of the velocities."""
        times = np.linspace(0.0, 10.0, n) ** 1.5
        az_vel = np.sin(times)
        el_vel = 0.1 * times**2
        traj = Trajectory(
            times=times,
            az=100.0 + az_vel,
            el=45.0 + el_vel,
            az_vel=az_vel,
            el_vel=el_vel,
        )
        az_accel = np.gradient(az_vel, times)
        el_accel = np.gradient(el_vel, times)
        np.testing.assert_array_equal(traj.az_accel, az_accel)
        np.testing.assert_array_equal(traj.el_accel, el_accel)
        np.testing.assert_array_equal(traj.az_jerk, np.gradient(az_accel, times))
        np.testing.assert_array_equal(traj.el_jerk, np.gradient(el_accel, times))


class TestFormatTrajectory:
    """Table rendering: headers, the head/tail ellipsis, the UTC column, file output."""

    def _make_trajectory(self, n_points=20, start_time=None):
        """Create a test trajectory."""
        times = np.linspace(0, n_points - 1, n_points)
        az = 100.0 + np.arange(n_points, dtype=float)
        el = np.full(n_points, 45.0)
        az_vel = np.ones(n_points)
        el_vel = np.zeros(n_points)
        return Trajectory(
            times=times,
            az=az,
            el=el,
            az_vel=az_vel,
            el_vel=el_vel,
            start_time=start_time,
        )

    def test_basic_formatting(self):
        traj = self._make_trajectory(5)
        output = _format_trajectory(traj, head=5, tail=5)

        lines = output.strip().split("\n")
        assert any("t (s)" in line for line in lines)
        assert any("az" in line for line in lines)
        assert "..." not in output

    def test_ellipsis_for_long_trajectory(self):
        traj = self._make_trajectory(100)
        output = _format_trajectory(traj, head=3, tail=3)
        assert "..." in output

    def test_head_tail_combinations(self):
        traj = self._make_trajectory(20)

        output = _format_trajectory(traj, head=3, tail=None)
        assert "..." not in output

        output = _format_trajectory(traj, head=None, tail=3)
        assert "..." not in output

        output = _format_trajectory(traj, head=3, tail=3)
        assert "..." in output

        # head + tail >= n_points: no ellipsis needed
        output = _format_trajectory(traj, head=15, tail=15)
        assert "..." not in output

    def test_with_absolute_times(self):
        start = Time("2026-03-15T04:00:00", scale="utc")
        traj = self._make_trajectory(5, start_time=start)
        output = _format_trajectory(traj, head=5, tail=5)

        # Should contain UTC column
        assert "UTC" in output

    def test_print_trajectory_writes_to_file(self):
        traj = self._make_trajectory(5)
        buf = io.StringIO()
        print_trajectory(traj, head=3, tail=2, file=buf)

        output = buf.getvalue()
        assert len(output) > 0
        assert "t (s)" in output


class TestScanFlagValidation:
    """scan_flag length checking, and what ``science_mask`` counts as science."""

    def test_scan_flag_length_mismatch_raises(self):
        times = np.array([0, 1, 2], dtype=float)
        with pytest.raises(ValueError, match="scan_flag"):
            Trajectory(
                times=times,
                az=np.zeros(3),
                el=np.zeros(3),
                az_vel=np.zeros(3),
                el_vel=np.zeros(3),
                scan_flag=np.zeros(5, dtype=np.int8),
            )

    def test_science_mask_default_all_true(self):
        traj = Trajectory(
            times=np.array([0, 1, 2], dtype=float),
            az=np.zeros(3),
            el=np.zeros(3),
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
        )
        assert traj.scan_flag is None
        mask = traj.science_mask
        assert mask.dtype == bool
        assert np.all(mask)

    def test_science_mask_with_flags(self):
        flags = np.array(
            [SCAN_FLAG_SCIENCE, SCAN_FLAG_TURNAROUND, SCAN_FLAG_SCIENCE],
            dtype=np.int8,
        )
        traj = Trajectory(
            times=np.array([0, 1, 2], dtype=float),
            az=np.zeros(3),
            el=np.zeros(3),
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
            scan_flag=flags,
        )
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(traj.science_mask, expected)

    def test_science_mask_excludes_unclassified(self):
        flags = np.array([SCAN_FLAG_UNCLASSIFIED, SCAN_FLAG_SCIENCE], dtype=np.int8)
        traj = Trajectory(
            times=np.array([0, 1], dtype=float),
            az=np.zeros(2),
            el=np.zeros(2),
            az_vel=np.zeros(2),
            el_vel=np.zeros(2),
            scan_flag=flags,
        )
        expected = np.array([False, True])
        np.testing.assert_array_equal(traj.science_mask, expected)


class TestArrayCoercion:
    """The five arrays are stored one-dimensional float64; scan_flag one-dimensional int8."""

    _FIELDS = ("times", "az", "el", "az_vel", "el_vel")

    @staticmethod
    def _kwargs(n=3):
        return {
            "times": np.arange(n, dtype=np.float64),
            "az": np.linspace(100.0, 102.0, n),
            "el": np.full(n, 45.0),
            "az_vel": np.ones(n),
            "el_vel": np.zeros(n),
        }

    def test_list_input_is_stored_as_float64(self):
        traj = Trajectory(
            times=[0, 1, 2],
            az=[100, 101, 102],
            el=[45.0, 45.0, 45.0],
            az_vel=[1, 1, 1],
            el_vel=[0, 0, 0],
        )
        assert repr(traj) == (
            "Trajectory(n_points=3, duration=2.0s, az=[100.0, 102.0]deg, el=[45.0, 45.0]deg)"
        )
        for name in self._FIELDS:
            value = getattr(traj, name)
            assert isinstance(value, np.ndarray), name
            assert value.dtype == np.float64, name

    def test_integer_input_is_stored_as_float64_with_equal_values(self):
        kwargs = {name: np.asarray(value, dtype=np.int64) for name, value in self._kwargs().items()}
        traj = Trajectory(**kwargs)
        for name in self._FIELDS:
            value = getattr(traj, name)
            assert value.dtype == np.float64, name
            np.testing.assert_array_equal(value, kwargs[name], err_msg=name)
        # Integer storage would wrap under arithmetic; float storage does not.
        small = Trajectory(**{name: np.asarray(v, dtype=np.int8) for name, v in kwargs.items()})
        np.testing.assert_array_equal(small.az + 100, [200.0, 201.0, 202.0])

    def test_float64_input_is_not_copied(self):
        kwargs = self._kwargs()
        traj = Trajectory(**kwargs)
        for name in self._FIELDS:
            assert getattr(traj, name) is kwargs[name], name

    @pytest.mark.parametrize("name", _FIELDS)
    def test_two_dimensional_array_raises_naming_the_field(self, name):
        kwargs = self._kwargs(2)
        kwargs[name] = kwargs[name][:, None] + np.array([0.0, 0.5])  # shape (2, 2), rows increasing
        with pytest.raises(ValueError, match=rf"'{name}' must be one-dimensional, got shape"):
            Trajectory(**kwargs)

    def test_zero_dimensional_times_raises(self):
        kwargs = self._kwargs(1)
        kwargs["times"] = np.float64(0.0)
        with pytest.raises(ValueError, match="'times' must be one-dimensional, got shape"):
            Trajectory(**kwargs)

    def test_list_scan_flag_is_stored_as_int8(self):
        traj = Trajectory(**self._kwargs(), scan_flag=[1, 2, 1])
        assert isinstance(traj.scan_flag, np.ndarray)
        assert traj.scan_flag.dtype == np.int8
        np.testing.assert_array_equal(traj.scan_flag, [1, 2, 1])

    def test_two_dimensional_scan_flag_raises(self):
        with pytest.raises(ValueError, match="'scan_flag' must be one-dimensional, got shape"):
            Trajectory(**self._kwargs(), scan_flag=np.ones((3, 2), dtype=np.int8))


class TestNonFiniteRejection:
    """Trajectory rejects NaN/Inf in its coordinate arrays at construction."""

    @staticmethod
    def _kwargs():
        return {
            "times": np.array([0.0, 1.0]),
            "az": np.array([10.0, 11.0]),
            "el": np.array([45.0, 45.0]),
            "az_vel": np.zeros(2),
            "el_vel": np.zeros(2),
        }

    def test_nan_in_az_raises(self):
        kwargs = self._kwargs()
        kwargs["az"] = np.array([10.0, np.nan])
        with pytest.raises(ValueError, match="Non-finite"):
            Trajectory(**kwargs)

    def test_inf_in_el_raises(self):
        kwargs = self._kwargs()
        kwargs["el"] = np.array([45.0, np.inf])
        with pytest.raises(ValueError, match="Non-finite"):
            Trajectory(**kwargs)


class TestMonotonicTimes:
    """Trajectory times must be strictly increasing.

    Every derived quantity divides by a time step: the acceleration and jerk
    properties and the dynamics validator. A repeated sample would make those
    silently ``inf``/``nan`` (the constructor's finiteness check looks at the
    stored arrays, not at their differences), and only
    ``validate_trajectory_dynamics`` would notice, and only if it were called.
    """

    @staticmethod
    def _kwargs(times):
        n = len(times)
        return {
            "times": np.asarray(times, dtype=float),
            "az": np.linspace(10.0, 11.0, n),
            "el": np.full(n, 45.0),
            "az_vel": np.zeros(n),
            "el_vel": np.zeros(n),
        }

    def test_repeated_timestamp_raises(self):
        with pytest.raises(ValueError, match="strictly increasing") as excinfo:
            Trajectory(**self._kwargs([0.0, 1.0, 1.0, 2.0]))
        # The message names the offending pair, not just the rule.
        assert "times[2]" in str(excinfo.value)

    def test_decreasing_timestamp_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            Trajectory(**self._kwargs([0.0, 2.0, 1.0]))

    def test_single_sample_is_accepted(self):
        """A one-sample trajectory has no step to check."""
        assert Trajectory(**self._kwargs([0.0])).n_points == 1

    def test_increasing_times_still_construct(self):
        traj = Trajectory(**self._kwargs([0.0, 0.1, 0.2, 0.3]))
        assert traj.n_points == 4
        assert np.all(np.isfinite(traj.az_accel))
        assert np.all(np.isfinite(traj.az_jerk))


class TestTrajectoryMetadataHome:
    """``TrajectoryMetadata`` is defined beside ``Trajectory``, in the container module.

    The container module imports nothing from the package but the private read-only mapping,
    whose module imports nothing at all, so the class its ``metadata`` field names is defined
    there and the patterns import it from below.
    """

    def test_import_paths_return_one_class(self):
        cls = fyst_trajectories.trajectory.TrajectoryMetadata
        assert fyst_trajectories.TrajectoryMetadata is cls
        assert fyst_trajectories.patterns.TrajectoryMetadata is cls
        assert cls.__module__ == "fyst_trajectories.trajectory"

    def test_trajectory_module_imports_only_the_readonly_leaf(self):
        source = Path(fyst_trajectories.trajectory.__file__).read_text(encoding="utf-8")
        intra = [
            ast.unparse(node)
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom)
            and (node.level > 0 or (node.module or "").startswith("fyst_trajectories"))
            or isinstance(node, ast.Import)
            and any(alias.name.startswith("fyst_trajectories") for alias in node.names)
        ]
        # tests/test_readonly.py asserts that ``_readonly`` imports nothing.
        assert intra == ["from ._readonly import ReadOnlyDict"]


class TestPatternParamsReadOnly:
    """``pattern_params`` is a read-only ``dict`` that a trajectory and its copies share."""

    @staticmethod
    def _trajectory():
        metadata = TrajectoryMetadata(
            pattern_type="test_pattern",
            pattern_params={"width": 2.0, "height": 1.0},
            center_ra=180.0,
            center_dec=-30.0,
        )
        return Trajectory(
            times=np.array([0.0, 1.0, 2.0]),
            az=np.zeros(3),
            el=np.full(3, 45.0),
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
            metadata=metadata,
        )

    def test_item_assignment_raises(self):
        traj = self._trajectory()
        with pytest.raises(TypeError, match="read-only"):
            traj.pattern_params["width"] = 9.0
        with pytest.raises(TypeError, match="read-only"):
            traj.metadata.pattern_params.update(width=9.0)
        assert traj.pattern_params == {"width": 2.0, "height": 1.0}

    def test_construction_copies_the_callers_dict(self):
        params = {"width": 2.0}
        metadata = TrajectoryMetadata(pattern_type="test_pattern", pattern_params=params)
        params["width"] = 9.0
        assert metadata.pattern_params == {"width": 2.0}

    def test_metadata_is_hashable(self):
        traj = self._trajectory()
        same = dataclasses.replace(traj.metadata)
        assert same == traj.metadata
        assert hash(same) == hash(traj.metadata)

    def test_pickle_deepcopy_asdict_and_json(self):
        traj = self._trajectory()
        restored = pickle.loads(pickle.dumps(traj))
        assert restored.metadata == traj.metadata
        assert copy.deepcopy(traj.metadata) == traj.metadata
        assert dataclasses.asdict(traj.metadata)["pattern_params"] == {"width": 2.0, "height": 1.0}
        assert json.loads(json.dumps(traj.pattern_params)) == {"width": 2.0, "height": 1.0}
        assert json.loads(json.dumps(dataclasses.asdict(traj.metadata)))["pattern_type"] == (
            "test_pattern"
        )

    def test_yaml_after_dict(self):
        traj = self._trajectory()
        dumped = yaml.safe_dump(dict(traj.pattern_params))
        assert yaml.safe_load(dumped) == {"width": 2.0, "height": 1.0}

    def test_replace_copy_shares_the_metadata_and_neither_edits_it(self):
        traj = self._trajectory()
        copy_ = dataclasses.replace(traj, az=traj.az + 1.0)
        assert copy_.metadata is traj.metadata
        for owner in (traj, copy_):
            with pytest.raises(TypeError, match="read-only"):
                owner.pattern_params["width"] = 9.0
        assert copy_.pattern_params == {"width": 2.0, "height": 1.0}

    def test_edited_copy_through_replace(self):
        traj = self._trajectory()
        edited = dataclasses.replace(
            traj.metadata, pattern_params={**traj.pattern_params, "width": 3.0}
        )
        assert edited.pattern_params == {"width": 3.0, "height": 1.0}
        assert traj.pattern_params == {"width": 2.0, "height": 1.0}
        with pytest.raises(TypeError, match="read-only"):
            edited.pattern_params["width"] = 9.0
