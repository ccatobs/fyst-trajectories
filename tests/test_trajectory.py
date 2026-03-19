"""Tests for the Trajectory container class."""

import io
import warnings

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories import (
    SCAN_FLAG_SCIENCE,
    SCAN_FLAG_TURNAROUND,
    SCAN_FLAG_UNCLASSIFIED,
    Trajectory,
    get_fyst_site,
    print_trajectory,
)
from fyst_trajectories.exceptions import AzimuthBoundsError
from fyst_trajectories.patterns import TrajectoryMetadata
from fyst_trajectories.trajectory_utils import _format_trajectory


class TestTrajectory:
    """Tests for Trajectory container class."""

    def test_trajectory_creation(self):
        """Test creating a Trajectory object."""
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
        """Test creating a Trajectory with metadata and accessing via properties."""
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
        """Test computing absolute timestamps."""
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

        abs_times = traj.get_absolute_times()
        assert len(abs_times) == 3
        assert abs_times[0] == start

    def test_absolute_times_without_start_raises(self):
        """Test that get_absolute_times raises without start_time."""
        traj = Trajectory(
            times=np.array([0, 1, 2], dtype=float),
            az=np.zeros(3),
            el=np.zeros(3),
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
        )

        with pytest.raises(ValueError, match="start_time not set"):
            traj.get_absolute_times()

    def test_to_arrays(self):
        """Test exporting trajectory to arrays returns copies."""
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

        t_out, az_out, el_out = traj.to_arrays()

        np.testing.assert_array_equal(t_out, times)
        np.testing.assert_array_equal(az_out, az)
        np.testing.assert_array_equal(el_out, el)

        t_out[0] = 999  # Modifying copy should not affect original
        assert traj.times[0] == 0.0

    def test_to_path_format(self):
        """Test converting trajectory to path format for OCS."""
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

        path = traj.to_path_format()

        assert len(path) == 2
        assert path[0] == [0.0, 100.0, 45.0, 10.0, 1.0]
        assert path[1] == [1.0, 110.0, 46.0, 10.0, 1.0]

    def test_array_length_mismatch_raises(self):
        """Test that mismatched array lengths raise ValueError."""
        with pytest.raises(ValueError, match="Array length mismatch"):
            Trajectory(
                times=np.array([0.0, 1.0, 2.0]),
                az=np.array([100.0, 110.0]),  # 2 instead of 3
                el=np.full(3, 45.0),
                az_vel=np.zeros(3),
                el_vel=np.zeros(3),
            )

    def test_validate_within_limits(self):
        """Test that validate passes for trajectory within limits."""
        site = get_fyst_site()
        traj = Trajectory(
            times=np.array([0.0, 1.0, 2.0, 3.0]),
            az=np.array([100.0, 101.0, 102.0, 103.0]),
            el=np.full(4, 45.0),
            az_vel=np.full(4, 1.0),
            el_vel=np.zeros(4),
        )
        traj.validate(site)

    def test_validate_out_of_bounds_raises(self):
        """Test that validate raises for trajectory outside limits."""
        site = get_fyst_site()
        traj = Trajectory(
            times=np.array([0.0, 1.0, 2.0]),
            az=np.array([100.0, 110.0, 400.0]),  # 400 > 360 limit
            el=np.full(3, 45.0),
            az_vel=np.zeros(3),
            el_vel=np.zeros(3),
        )
        with pytest.raises(AzimuthBoundsError, match="azimuth"):
            traj.validate(site)

    def test_validate_warns_on_high_velocity(self):
        """Test that validate warns for excessive velocity."""
        site = get_fyst_site()
        traj = Trajectory(
            times=np.linspace(0, 10, 100),
            az=100.0 + 10.0 * np.linspace(0, 10, 100),  # 10 deg/s
            el=np.full(100, 45.0),
            az_vel=np.full(100, 10.0),
            el_vel=np.zeros(100),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            traj.validate(site)
            vel_warnings = [x for x in w if "velocity" in str(x.message).lower()]
            assert len(vel_warnings) >= 1


class TestAccelerationJerkProperties:
    """Tests for the computed acceleration and jerk properties."""

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
        """Test that acceleration properties return arrays matching trajectory length."""
        traj = self._make_constant_velocity_trajectory()
        assert traj.az_accel.shape == traj.times.shape
        assert traj.el_accel.shape == traj.times.shape

    def test_jerk_returns_correct_shape(self):
        """Test that jerk properties return arrays matching trajectory length."""
        traj = self._make_constant_velocity_trajectory()
        assert traj.az_jerk.shape == traj.times.shape
        assert traj.el_jerk.shape == traj.times.shape

    def test_constant_velocity_has_zero_acceleration(self):
        """Test that a constant-velocity trajectory has approximately zero acceleration."""
        traj = self._make_constant_velocity_trajectory()
        np.testing.assert_allclose(traj.az_accel, 0.0, atol=1e-10)
        np.testing.assert_allclose(traj.el_accel, 0.0, atol=1e-10)

    def test_constant_velocity_has_zero_jerk(self):
        """Test that a constant-velocity trajectory has approximately zero jerk."""
        traj = self._make_constant_velocity_trajectory()
        np.testing.assert_allclose(traj.az_jerk, 0.0, atol=1e-10)
        np.testing.assert_allclose(traj.el_jerk, 0.0, atol=1e-10)

    def test_constant_acceleration_value(self):
        """Test that constant acceleration is recovered correctly."""
        traj = self._make_accelerating_trajectory()
        np.testing.assert_allclose(traj.az_accel, 0.2, atol=1e-10)
        np.testing.assert_allclose(traj.el_accel, 0.1, atol=1e-10)

    def test_constant_acceleration_has_zero_jerk(self):
        """Test that constant acceleration produces approximately zero jerk."""
        traj = self._make_accelerating_trajectory()
        np.testing.assert_allclose(traj.az_jerk, 0.0, atol=1e-10)
        np.testing.assert_allclose(traj.el_jerk, 0.0, atol=1e-10)

    def test_jerk_is_derivative_of_acceleration(self):
        """Test that jerk equals the numerical derivative of acceleration."""
        # Use a trajectory where acceleration varies (quadratic velocity)
        times = np.linspace(0, 5, 501)
        az_vel = 0.1 * times**2  # accel = 0.2*t, jerk = 0.2
        el_vel = 0.05 * times**2
        az = np.cumsum(az_vel) * (times[1] - times[0])
        el = 45.0 + np.cumsum(el_vel) * (times[1] - times[0])
        traj = Trajectory(times=times, az=az, el=el, az_vel=az_vel, el_vel=el_vel)

        # Compute jerk two ways: via property and manually
        jerk_via_property = traj.az_jerk
        accel_manual = np.gradient(traj.az_vel, traj.times)
        jerk_manual = np.gradient(accel_manual, traj.times)

        np.testing.assert_allclose(jerk_via_property, jerk_manual, atol=1e-10)

    def test_properties_recompute_each_call(self):
        """Test that properties are computed fresh (not cached stale values)."""
        traj = self._make_constant_velocity_trajectory()
        accel1 = traj.az_accel
        accel2 = traj.az_accel
        # They should be equal but not the same object (recomputed)
        np.testing.assert_array_equal(accel1, accel2)
        assert accel1 is not accel2


class TestFormatTrajectory:
    """Tests for _format_trajectory and print_trajectory."""

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
        """Test basic formatting of a small trajectory."""
        traj = self._make_trajectory(5)
        output = _format_trajectory(traj, head=5, tail=5)

        lines = output.strip().split("\n")
        assert any("t (s)" in line for line in lines)
        assert any("az" in line for line in lines)
        assert "..." not in output

    def test_ellipsis_for_long_trajectory(self):
        """Test that long trajectories show ellipsis."""
        traj = self._make_trajectory(100)
        output = _format_trajectory(traj, head=3, tail=3)
        assert "..." in output

    def test_head_tail_combinations(self):
        """Test various head/tail combinations."""
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
        """Test formatting with absolute times."""
        start = Time("2026-03-15T04:00:00", scale="utc")
        traj = self._make_trajectory(5, start_time=start)
        output = _format_trajectory(traj, head=5, tail=5)

        # Should contain UTC column
        assert "UTC" in output

    def test_print_trajectory_writes_to_file(self):
        """Test print_trajectory writes to a file object."""
        traj = self._make_trajectory(5)
        buf = io.StringIO()
        print_trajectory(traj, head=3, tail=2, file=buf)

        output = buf.getvalue()
        assert len(output) > 0
        assert "t (s)" in output


class TestScanFlagValidation:
    """Tests for scan_flag field on Trajectory."""

    def test_scan_flag_length_mismatch_raises(self):
        """scan_flag must match times length if provided."""
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
        """science_mask should be all True when scan_flag is None."""
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
        """science_mask should reflect scan_flag values."""
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
        """Unclassified samples are NOT treated as science."""
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
