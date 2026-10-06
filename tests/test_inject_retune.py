"""Tests for inject_retune() in uniform-cadence mode on synthetic trajectories.

Covers basic placement, turnaround snapping, a trajectory without turnarounds,
the science mask, edge cases, the per-module staggered mode, theoretical
efficiency parametric checks, the zero-velocity defensive guard, and the
placements the uniform path records in ``retune_events``.
"""

import warnings as _warnings

import numpy as np
import pytest
from _retune_stubs import _group_retune_events, _make_trajectory

from fyst_trajectories.exceptions import PointingWarning
from fyst_trajectories.retune import inject_retune
from fyst_trajectories.trajectory import (
    SCAN_FLAG_RETUNE,
    SCAN_FLAG_SCIENCE,
    SCAN_FLAG_TURNAROUND,
    RetuneEvent,
    Trajectory,
)


class TestInjectRetuneBasic:
    """Uniform placement: the next retune is due one interval after the last one ends."""

    def test_retune_flags_placed_at_correct_intervals(self):
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        result = inject_retune(
            traj, retune_interval=30.0, retune_duration=5.0, prefer_turnarounds=False
        )

        # Check that retune flags exist
        retune_mask = result.scan_flag == SCAN_FLAG_RETUNE
        assert retune_mask.any()

        # Find the start times of retune events
        retune_times = result.times[retune_mask]
        events = _group_retune_events(retune_times)

        # Retune at ~30s, then interval measured from retune_end (35s),
        # so next at ~65s, then ~100s.
        assert len(events) == 3
        np.testing.assert_allclose(events, [30.0, 65.0, 100.0], atol=0.15)

    def test_retune_duration_correct(self):
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        result = inject_retune(
            traj, retune_interval=30.0, retune_duration=5.0, prefer_turnarounds=False
        )

        retune_mask = result.scan_flag == SCAN_FLAG_RETUNE
        retune_times = result.times[retune_mask]

        # Group into events
        events_samples = []
        current = [retune_times[0]]
        for i in range(1, len(retune_times)):
            if retune_times[i] - retune_times[i - 1] > 0.2:
                events_samples.append(current)
                current = [retune_times[i]]
            else:
                current.append(retune_times[i])
        events_samples.append(current)

        for event in events_samples:
            event_duration = event[-1] - event[0] + 0.1  # +timestep
            assert abs(event_duration - 5.0) < 0.2


class TestInjectRetuneTurnaroundSnapping:
    """Snapping to a turnaround inside the window, the time-based fallback outside it."""

    def test_snaps_to_nearby_turnaround(self):
        # Turnaround at 28-31s (3s), near the 30s due time.
        # Retune duration is 5s, so retune covers 28-33s.
        # Turnaround occupies 28-31, so RETUNE flags appear at 31-33 (science region).
        traj = _make_trajectory(
            duration=120.0,
            timestep=0.1,
            turnaround_intervals=[(28.0, 31.0)],
        )
        result = inject_retune(
            traj,
            retune_interval=30.0,
            retune_duration=5.0,
            prefer_turnarounds=True,
            turnaround_window=5.0,
        )

        retune_mask = result.scan_flag == SCAN_FLAG_RETUNE
        assert retune_mask.any()

        # The retune flags should start right after the turnaround ends (at ~31s)
        # because the turnaround samples are not overwritten
        first_retune_time = result.times[retune_mask][0]
        assert abs(first_retune_time - 31.0) < 0.15

        # Turnaround flags should be preserved
        ta_mask = result.scan_flag == SCAN_FLAG_TURNAROUND
        ta_count = ta_mask.sum()
        original_ta_count = (traj.scan_flag == SCAN_FLAG_TURNAROUND).sum()
        assert ta_count == original_ta_count

    def test_no_turnaround_nearby_falls_back_to_time_based(self):
        # Turnaround far from the 30s due time
        traj = _make_trajectory(
            duration=120.0,
            timestep=0.1,
            turnaround_intervals=[(10.0, 15.0)],
        )
        result = inject_retune(
            traj,
            retune_interval=30.0,
            retune_duration=5.0,
            prefer_turnarounds=True,
            turnaround_window=5.0,
        )

        retune_mask = result.scan_flag == SCAN_FLAG_RETUNE
        assert retune_mask.any()
        first_retune_time = result.times[retune_mask][0]
        # Should be at ~30s (time-based), not snapped
        assert abs(first_retune_time - 30.0) < 0.15

    def test_synthetic_turnaround_overlap_count(self):
        """With turnarounds at retune due times, snapping should use them.

        Creates a synthetic trajectory with turnarounds at 28-31s,
        58-61s, 88-91s, near the 30s, 60s, 90s due times. With
        prefer_turnarounds=True, retunes should snap to these and
        preserve more science samples than time-based placement.
        """
        turnarounds = [(28.0, 31.0), (58.0, 61.0), (88.0, 91.0)]
        traj = _make_trajectory(
            duration=120.0,
            timestep=0.1,
            turnaround_intervals=turnarounds,
        )

        # With snapping
        result_snap = inject_retune(
            traj,
            retune_interval=30.0,
            retune_duration=5.0,
            prefer_turnarounds=True,
            turnaround_window=5.0,
        )
        # Without snapping
        result_time = inject_retune(
            traj,
            retune_interval=30.0,
            retune_duration=5.0,
            prefer_turnarounds=False,
        )

        # With snapping, retunes overlap turnaround positions, so the
        # additional science lost should be less
        snap_science = result_snap.science_mask.sum()
        time_science = result_time.science_mask.sum()

        # Snapping should preserve strictly more science samples
        assert snap_science > time_science, (
            f"Snapping preserved {snap_science} science samples vs {time_science} without snapping"
        )


class TestInjectRetuneNoTurnarounds:
    """A scan with no turnaround flags gets exactly the time-based retunes."""

    def test_no_turnarounds_uses_time_based(self):
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        snapped = inject_retune(
            traj, retune_interval=30.0, retune_duration=5.0, prefer_turnarounds=True
        )
        time_based = inject_retune(
            traj, retune_interval=30.0, retune_duration=5.0, prefer_turnarounds=False
        )

        assert len(snapped.retune_events) == 3
        np.testing.assert_array_equal(snapped.scan_flag, time_based.scan_flag)
        assert snapped.retune_events == time_based.retune_events


class TestInjectRetuneScienceMask:
    """``science_mask`` excludes retune samples: 30 s of science per 35 s cadence."""

    def test_science_mask_excludes_retune(self):
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        result = inject_retune(
            traj, retune_interval=30.0, retune_duration=5.0, prefer_turnarounds=False
        )

        science = result.science_mask
        retune = result.scan_flag == SCAN_FLAG_RETUNE

        # No overlap: science_mask should be False wherever retune is True
        assert not np.any(science & retune)

    def test_efficiency_calculation(self):
        """30s interval / 5s duration should give ~85.7% science fraction."""
        traj = _make_trajectory(duration=300.0, timestep=0.1)
        result = inject_retune(
            traj, retune_interval=30.0, retune_duration=5.0, prefer_turnarounds=False
        )

        science_fraction = result.science_mask.sum() / len(result.times)
        # ~85.7% (30/35); edge effects lift a 300 s trajectory to ~86.7%.
        assert science_fraction == pytest.approx(0.8667, abs=1e-3)


class TestInjectRetuneEdgeCases:
    """Short trajectories, absent flags, and purity: no retune, no crash, no mutation."""

    def test_interval_longer_than_trajectory(self):
        traj = _make_trajectory(duration=20.0, timestep=0.1)
        result = inject_retune(traj, retune_interval=30.0, retune_duration=5.0)

        # No retune should be placed
        retune_mask = result.scan_flag == SCAN_FLAG_RETUNE
        assert not retune_mask.any()

    def test_very_short_trajectory(self):
        """Very short trajectory should not crash and have no retune flags."""
        times = np.array([0.0, 0.1, 0.2])
        az = np.array([100.0, 100.1, 100.2])
        el = np.full(3, 45.0)
        az_vel = np.array([1.0, 1.0, 1.0])
        el_vel = np.zeros(3)
        scan_flag = np.full(3, SCAN_FLAG_SCIENCE, dtype=np.int8)
        traj = Trajectory(
            times=times, az=az, el=el, az_vel=az_vel, el_vel=el_vel, scan_flag=scan_flag
        )

        result = inject_retune(traj, retune_interval=30.0, retune_duration=5.0)
        # Should not crash and no retune placed
        assert not (result.scan_flag == SCAN_FLAG_RETUNE).any()

    def test_only_science_flags_overwritten(self):
        traj = _make_trajectory(
            duration=120.0,
            timestep=0.1,
            turnaround_intervals=[(29.0, 36.0)],
        )
        original_turnaround_count = (traj.scan_flag == SCAN_FLAG_TURNAROUND).sum()

        result = inject_retune(
            traj, retune_interval=30.0, retune_duration=5.0, prefer_turnarounds=False
        )

        new_turnaround_count = (result.scan_flag == SCAN_FLAG_TURNAROUND).sum()
        assert new_turnaround_count == original_turnaround_count

    def test_returns_new_trajectory(self):
        """inject_retune should return a new Trajectory, not mutate the original."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        original_flags = traj.scan_flag.copy()

        result = inject_retune(traj, retune_interval=30.0, retune_duration=5.0)

        assert result is not traj
        np.testing.assert_array_equal(traj.scan_flag, original_flags)

    def test_no_scan_flag_array(self):
        """Trajectory with scan_flag=None should work (treated as all-science)."""
        times = np.arange(0, 120.0, 0.1)
        n = len(times)
        traj = Trajectory(
            times=times,
            az=np.linspace(100, 200, n),
            el=np.full(n, 45.0),
            az_vel=np.ones(n),
            el_vel=np.zeros(n),
            scan_flag=None,
        )
        result = inject_retune(traj, retune_interval=30.0, retune_duration=5.0)
        assert result.scan_flag is not None
        assert (result.scan_flag == SCAN_FLAG_RETUNE).any()


class TestInjectRetuneStaggered:
    """Staggering offsets each module's retunes by ``retune_interval / n_modules``.

    Per-module retune independence is UNCONFIRMED by the FYST instrument
    team. These tests verify the staggering mechanism works correctly if
    modules can retune independently.
    """

    def test_staggered_retune_offset(self):
        """Different module_index values should produce retunes at different times."""
        traj = _make_trajectory(duration=300.0, timestep=0.1)

        def _first_retune_time(module_index: int) -> float:
            result = inject_retune(
                traj,
                retune_interval=30.0,
                retune_duration=5.0,
                module_index=module_index,
                n_modules=7,
            )
            retune_mask = result.scan_flag == SCAN_FLAG_RETUNE
            return float(result.times[retune_mask][0])

        # Each module should start its first retune at a different time
        first_times = [_first_retune_time(i) for i in range(7)]

        # All first retune times should be distinct
        for i in range(len(first_times)):
            for j in range(i + 1, len(first_times)):
                assert abs(first_times[i] - first_times[j]) > 1.0, (
                    f"Module {i} and {j} retune at the same time: "
                    f"{first_times[i]:.1f} vs {first_times[j]:.1f}"
                )

        # The offsets should be spaced by retune_interval / n_modules = 30/7 ~= 4.29s
        expected_spacing = 30.0 / 7
        for i in range(1, 7):
            expected = first_times[0] + i * expected_spacing
            np.testing.assert_allclose(first_times[i], expected, atol=0.15)

    def test_staggered_retune_coverage(self):
        """Combined science_mask from all 7 modules should have better coverage."""
        traj = _make_trajectory(duration=300.0, timestep=0.1)

        # Single module (no staggering), baseline
        single = inject_retune(traj, retune_interval=30.0, retune_duration=5.0, n_modules=1)
        single_fraction = single.science_mask.sum() / len(single.times)

        # 7 staggered modules, combined mask is True where ANY module is observing
        module_masks = []
        for i in range(7):
            result = inject_retune(
                traj,
                retune_interval=30.0,
                retune_duration=5.0,
                module_index=i,
                n_modules=7,
            )
            module_masks.append(result.science_mask)

        # For each sample, count how many modules are doing science
        # The "combined" fraction: a sample is lost only if ALL modules are retuning
        all_retuning = np.ones(len(traj.times), dtype=bool)
        for mask in module_masks:
            all_retuning &= ~mask
        combined_fraction = 1.0 - all_retuning.sum() / len(traj.times)

        # Single module: ~86.7% (a 5 s gap every 35 s). Staggered: the seven
        # offsets are spread across the interval, so at most two modules retune
        # at once and some module is always observing.
        assert combined_fraction > single_fraction
        assert combined_fraction > 0.97

    def test_staggered_defaults_unchanged(self):
        """module_index=0, n_modules=1 should produce identical output."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)

        result_default = inject_retune(traj, retune_interval=30.0, retune_duration=5.0)
        result_explicit = inject_retune(
            traj,
            retune_interval=30.0,
            retune_duration=5.0,
            module_index=0,
            n_modules=1,
        )

        np.testing.assert_array_equal(result_default.scan_flag, result_explicit.scan_flag)

    def test_staggered_invalid_module_index(self):
        """module_index >= n_modules should raise ValueError."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)

        with pytest.raises(ValueError, match="module_index"):
            inject_retune(traj, module_index=7, n_modules=7)

        with pytest.raises(ValueError, match="module_index"):
            inject_retune(traj, module_index=-1, n_modules=7)

    def test_staggered_invalid_n_modules(self):
        """n_modules < 1 should raise ValueError."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)

        with pytest.raises(ValueError, match="n_modules"):
            inject_retune(traj, n_modules=0)

    @pytest.mark.parametrize("duration", [30.0, 45.0], ids=["equal", "longer"])
    def test_gap_not_shorter_than_interval_raises(self, duration):
        """A gap at least as long as the interval is refused in uniform-cadence mode.

        Under time-based placement such a gap would flag
        ``duration / (interval + duration)``, at least half, of a long
        trajectory's science samples (``TestTheoreticalEfficiency`` pins that
        steady state for the shorter gaps the function accepts), which almost
        certainly means the two arguments are swapped.
        """
        traj = _make_trajectory(duration=120.0, timestep=0.1)

        with pytest.raises(ValueError, match="shorter than retune_interval"):
            inject_retune(traj, retune_interval=30.0, retune_duration=duration)

    def test_gap_guard_is_uniform_mode_only(self):
        """Event-list mode ignores the scalar knobs, so the guard must not fire there."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        events = [RetuneEvent(t_start=30.0, duration=5.0)]

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", PointingWarning)
            result = inject_retune(
                traj, retune_events=events, retune_interval=30.0, retune_duration=45.0
            )
        assert result.retune_events == tuple(events)


class TestInjectRetuneScalarValidation:
    """Non-finite and negative scalars raise with the argument's own name."""

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
    def test_non_finite_interval_raises(self, bad):
        """A NaN or infinite interval is refused, naming the argument.

        Unchecked, an infinite interval would inject nothing and a NaN one would
        fail later, in ``RetuneEvent``.
        """
        traj = _make_trajectory(duration=120.0, timestep=0.1)

        with pytest.raises(ValueError, match="retune_interval must be finite and positive"):
            inject_retune(traj, retune_interval=bad)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
    def test_non_finite_duration_raises(self, bad):
        """A non-finite gap is refused before the shorter-than-interval check."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)

        with pytest.raises(ValueError, match="retune_duration must be finite and positive"):
            inject_retune(traj, retune_interval=30.0, retune_duration=bad)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -5.0], ids=["nan", "inf", "negative"])
    def test_bad_turnaround_window_raises(self, bad):
        """A negative or NaN window used to turn snapping off without a word."""
        traj = _make_trajectory(duration=120.0, timestep=0.1, turnaround_intervals=[(30.0, 35.0)])

        with pytest.raises(ValueError, match="turnaround_window must be finite and non-negative"):
            inject_retune(
                traj, retune_interval=28.0, prefer_turnarounds=True, turnaround_window=bad
            )

    def test_bad_turnaround_window_raises_without_snapping(self):
        """The window is validated whether or not prefer_turnarounds is set."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)

        with pytest.raises(ValueError, match="turnaround_window must be finite and non-negative"):
            inject_retune(traj, retune_interval=30.0, turnaround_window=-5.0)

    def test_zero_turnaround_window_is_accepted(self):
        """A zero window snaps only a due time that falls on a turnaround start."""
        traj = _make_trajectory(duration=120.0, timestep=0.1, turnaround_intervals=[(32.0, 35.0)])

        result = inject_retune(
            traj,
            retune_interval=30.0,
            retune_duration=5.0,
            prefer_turnarounds=True,
            turnaround_window=0.0,
        )
        assert [event.t_start for event in result.retune_events] == pytest.approx(
            [30.0, 65.0, 100.0]
        )


class TestTheoreticalEfficiency:
    """Verify inject_retune efficiency matches theoretical predictions.

    The next retune is due one interval after the previous one ends, so the
    repeat period is ``interval + duration`` and the steady-state efficiency is
    ``interval / (interval + duration)``. This assumes long trajectories where
    edge effects are negligible.
    """

    @pytest.mark.parametrize(
        "interval, duration, expected",
        [
            (30.0, 5.0, 0.857),  # 30/35 = 85.7%
            (60.0, 5.0, 0.923),  # 60/65 = 92.3%
            (30.0, 2.0, 0.938),  # 30/32 = 93.8%
        ],
        ids=["30s/5s", "60s/5s", "30s/2s"],
    )
    def test_efficiency_matches_theory(self, interval, duration, expected):
        """Long trajectory efficiency should match theoretical value within 0.5%.

        The band is tight enough to reject the repeat period the steady state
        would have if the next retune were due one interval after the previous
        one STARTED rather than ended.
        """
        traj = _make_trajectory(duration=600.0, timestep=0.1)
        result = inject_retune(
            traj,
            retune_interval=interval,
            retune_duration=duration,
            prefer_turnarounds=False,
        )

        science_frac = result.science_mask.sum() / len(result.times)
        assert abs(science_frac - expected) < 0.005, (
            f"Science fraction {science_frac:.4f} deviates from "
            f"theoretical {expected:.4f} by more than 0.5%"
        )

    def test_longer_trajectory_closer_to_theory(self):
        """Longer trajectories should have smaller edge effects.

        A 1200s trajectory should be closer to 85.7% than a 120s one
        with 30s/5s retune.
        """
        expected = 30.0 / 35.0

        short = _make_trajectory(duration=120.0, timestep=0.1)
        long = _make_trajectory(duration=1200.0, timestep=0.1)

        short_result = inject_retune(
            short,
            retune_interval=30.0,
            retune_duration=5.0,
            prefer_turnarounds=False,
        )
        long_result = inject_retune(
            long,
            retune_interval=30.0,
            retune_duration=5.0,
            prefer_turnarounds=False,
        )

        short_err = abs(short_result.science_mask.sum() / len(short_result.times) - expected)
        long_err = abs(long_result.science_mask.sum() / len(long_result.times) - expected)

        assert long_err <= short_err + 0.001, (
            f"Long trajectory error ({long_err:.4f}) should be <= "
            f"short trajectory error ({short_err:.4f})"
        )


class TestZeroVelocityGuard:
    """Defensive guard for zero-velocity + prefer_turnarounds=True.

    Snapping reads turnaround starts from ``scan_flag``. A planned trajectory
    derives those flags from its velocity profile, so identically zero
    velocities mean the flags carry no reversal information: the guard warns
    and falls back to time-based placement, even when flags were set by hand.
    """

    def test_warns_on_zero_velocities_with_turnaround_snap(self):
        """Zero velocities + prefer_turnarounds=True must warn and fall back."""
        # Build a 300s trajectory whose az/el velocities are exactly zero.
        duration = 300.0
        timestep = 0.1
        times = np.arange(0, duration, timestep)
        n = len(times)
        # A turnaround 2 s before the first due time: a snap would move the
        # first retune to 28 s, the time-based fallback keeps it at 30 s.
        scan_flag = np.full(n, SCAN_FLAG_SCIENCE, dtype=np.int8)
        scan_flag[(times >= 28.0) & (times < 31.0)] = SCAN_FLAG_TURNAROUND
        traj = Trajectory(
            times=times,
            az=np.linspace(100.0, 200.0, n),
            el=np.full(n, 45.0),
            az_vel=np.zeros(n),
            el_vel=np.zeros(n),
            scan_flag=scan_flag,
        )

        with pytest.warns(PointingWarning, match="zero velocities"):
            result = inject_retune(
                traj,
                retune_interval=30.0,
                retune_duration=5.0,
                prefer_turnarounds=True,
            )

        # The fallback placed the retune time-based, not on the turnaround.
        assert result.retune_events[0].t_start == pytest.approx(30.0)
        # The input trajectory must remain unchanged (inject_retune is pure).
        assert traj.scan_flag is not None
        assert not (traj.scan_flag == SCAN_FLAG_RETUNE).any()

    def test_no_warning_when_prefer_turnarounds_false(self):
        """Zero velocities + prefer_turnarounds=False must NOT warn.

        Verifies the guard is scoped to the turnaround-snapping path and
        does not emit spurious warnings for the default time-based path.
        """
        duration = 120.0
        timestep = 0.1
        times = np.arange(0, duration, timestep)
        n = len(times)
        traj = Trajectory(
            times=times,
            az=np.linspace(100.0, 200.0, n),
            el=np.full(n, 45.0),
            az_vel=np.zeros(n),
            el_vel=np.zeros(n),
            scan_flag=np.full(n, SCAN_FLAG_SCIENCE, dtype=np.int8),
        )

        with _warnings.catch_warnings(record=True) as records:
            _warnings.simplefilter("always")
            inject_retune(
                traj,
                retune_interval=30.0,
                retune_duration=5.0,
                prefer_turnarounds=False,
            )

        matches = [
            r
            for r in records
            if issubclass(r.category, PointingWarning) and "zero velocities" in str(r.message)
        ]
        assert not matches, (
            f"Zero-velocity warning leaked into prefer_turnarounds=False path: {matches}"
        )

    def test_no_warning_with_real_velocities(self):
        """Real velocities + prefer_turnarounds=True must NOT warn about zero vel."""
        traj = _make_trajectory(
            duration=300.0,
            timestep=0.1,
            turnaround_intervals=[(28.0, 31.0), (58.0, 61.0)],
        )

        # az_vel is computed from np.gradient, so it is nonzero.
        assert not np.all(traj.az_vel == 0.0)

        with _warnings.catch_warnings(record=True) as records:
            _warnings.simplefilter("always")
            inject_retune(
                traj,
                retune_interval=30.0,
                retune_duration=5.0,
                prefer_turnarounds=True,
            )

        matches = [
            r
            for r in records
            if issubclass(r.category, PointingWarning) and "zero velocities" in str(r.message)
        ]
        assert not matches, (
            f"Zero-velocity guard fired for trajectory with real velocities: {matches}"
        )


class TestUniformPathPopulatesRetuneEvents:
    """The uniform-cadence path must populate ``Trajectory.retune_events``."""

    def test_uniform_path_retune_events_populated(self):
        """Calling uniform-cadence inject_retune sets retune_events symmetrically."""
        # 300 s at a 60 s interval + 5 s gap = a 65 s cadence: four events, at
        # t = 60, 125, 190, 255.
        duration = 300.0
        interval = 60.0
        dur = 5.0
        traj = _make_trajectory(duration=duration, timestep=0.1)
        result = inject_retune(
            traj, retune_interval=interval, retune_duration=dur, prefer_turnarounds=False
        )

        assert [e.t_start for e in result.retune_events] == pytest.approx(
            [60.0, 125.0, 190.0, 255.0]
        )
        # Every event has duration == the requested retune_duration.
        for ev in result.retune_events:
            assert ev.duration == pytest.approx(dur)
