"""Tests for inject_retune() trajectory utility.

Covers basic placement, turnaround snapping, edge cases, the per-module
staggered mode, the per-pattern efficiency cross-validation against
real planner outputs (CE / Pong / Daisy), turnaround-overlap dead-time
reduction, theoretical efficiency parametric checks, and the N-6
zero-velocity defensive guard.
"""

import warnings as _warnings

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories.exceptions import PointingWarning
from fyst_trajectories.planning import (
    FieldRegion,
    plan_constant_el_scan,
    plan_daisy_scan,
    plan_pong_scan,
)
from fyst_trajectories.trajectory import (
    SCAN_FLAG_RETUNE,
    SCAN_FLAG_SCIENCE,
    SCAN_FLAG_TURNAROUND,
    Trajectory,
)
from fyst_trajectories.trajectory_utils import (
    RetuneEvent,
    inject_retune,
    sample_retune_events,
)

# ``site`` fixture is provided by ``conftest.py``; tests in this module
# use that shared definition.


@pytest.fixture
def start_time():
    """Nighttime start at FYST."""
    return Time("2026-06-15T04:00:00", scale="utc")


def _make_trajectory(
    duration: float = 120.0,
    timestep: float = 0.1,
    turnaround_intervals: list[tuple[float, float]] | None = None,
) -> Trajectory:
    """Create a synthetic trajectory for inject_retune tests.

    Parameters
    ----------
    duration : float
        Total duration in seconds.
    timestep : float
        Time step in seconds.
    turnaround_intervals : list of (start, end) tuples
        Time intervals to flag as turnaround.
    """
    times = np.arange(0, duration, timestep)
    n = len(times)
    az = np.linspace(100, 200, n)
    el = np.full(n, 45.0)
    az_vel = np.gradient(az, times)
    el_vel = np.zeros(n)
    scan_flag = np.full(n, SCAN_FLAG_SCIENCE, dtype=np.int8)

    if turnaround_intervals:
        for t_start, t_end in turnaround_intervals:
            mask = (times >= t_start) & (times < t_end)
            scan_flag[mask] = SCAN_FLAG_TURNAROUND

    return Trajectory(times=times, az=az, el=el, az_vel=az_vel, el_vel=el_vel, scan_flag=scan_flag)


def _group_retune_events(retune_times: np.ndarray) -> list[float]:
    """Group retune flag timestamps into distinct events by start time.

    Returns the start time of each distinct retune event.
    """
    if len(retune_times) == 0:
        return []

    events = [retune_times[0]]
    for i in range(1, len(retune_times)):
        # Gap > 0.2s means a new event
        if retune_times[i] - retune_times[i - 1] > 0.2:
            events.append(retune_times[i])
    return events


class TestInjectRetuneBasic:
    """Basic retune flag placement tests."""

    def test_retune_flags_placed_at_correct_intervals(self):
        """Retune events should appear at the expected interval positions."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        result = inject_retune(
            traj, retune_interval=30.0, retune_duration=5.0, prefer_turnarounds=False
        )

        # Check that retune flags exist
        retune_mask = result.scan_flag == SCAN_FLAG_RETUNE
        assert retune_mask.any()

        # Find the start times of retune events
        retune_times = result.times[retune_mask]
        # Group into distinct retune events by finding gaps
        events = []
        current_start = retune_times[0]
        for i in range(1, len(retune_times)):
            if retune_times[i] - retune_times[i - 1] > 0.2:
                events.append(current_start)
                current_start = retune_times[i]
        events.append(current_start)

        # Retune at ~30s, then interval measured from retune_end (35s),
        # so next at ~65s, then ~100s.
        assert len(events) == 3
        np.testing.assert_allclose(events, [30.0, 65.0, 100.0], atol=0.15)

    def test_retune_duration_correct(self):
        """Each retune event should span the configured duration."""
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
    """Tests for turnaround snapping behavior."""

    def test_snaps_to_nearby_turnaround(self):
        """Retune should snap to a nearby turnaround when within window."""
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
        """Without a nearby turnaround, retune falls back to time-based placement."""
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
        58-61s, 88-91s -- near the 30s, 60s, 90s due times. With
        prefer_turnarounds=True, retunes should snap to these and
        preserve more (or equal) science samples than time-based
        placement.
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

        # Snapping should preserve more (or equal) science samples
        assert snap_science >= time_science, (
            f"Snapping preserved {snap_science} science samples vs {time_science} without snapping"
        )


class TestInjectRetuneDaisyContinuous:
    """Tests for continuous scan (no turnarounds), like daisy patterns."""

    def test_no_turnarounds_uses_time_based(self):
        """Continuous scan with no turnarounds should still place retunes."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        result = inject_retune(
            traj, retune_interval=30.0, retune_duration=5.0, prefer_turnarounds=True
        )

        retune_mask = result.scan_flag == SCAN_FLAG_RETUNE
        assert retune_mask.any()


class TestInjectRetuneScienceMask:
    """Tests for science_mask interaction."""

    def test_science_mask_excludes_retune(self):
        """science_mask should be False for all retune-flagged samples."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        result = inject_retune(
            traj, retune_interval=30.0, retune_duration=5.0, prefer_turnarounds=False
        )

        science = result.science_mask
        retune = result.scan_flag == SCAN_FLAG_RETUNE

        # No overlap: science_mask should be False wherever retune is True
        assert not np.any(science & retune)

    def test_efficiency_calculation(self):
        """30s interval / 5s duration should give ~83.3% science fraction."""
        traj = _make_trajectory(duration=300.0, timestep=0.1)
        result = inject_retune(
            traj, retune_interval=30.0, retune_duration=5.0, prefer_turnarounds=False
        )

        science_fraction = result.science_mask.sum() / len(result.times)
        # ~83.3% (25/30), allow some tolerance for edge effects
        assert 0.80 < science_fraction < 0.87


class TestInjectRetuneEdgeCases:
    """Edge case tests."""

    def test_interval_longer_than_trajectory(self):
        """No retune should be placed if interval exceeds trajectory duration."""
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
        """Turnaround flags should never be changed to retune."""
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
    """Tests for per-module staggered retune scheduling.

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

        # Single module (no staggering) -- baseline
        single = inject_retune(traj, retune_interval=30.0, retune_duration=5.0, n_modules=1)
        single_fraction = single.science_mask.sum() / len(single.times)

        # 7 staggered modules -- combined mask is True where ANY module is observing
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

        # Combined coverage should be much better than single-module
        # Single module: ~83.3% (5/30 lost). Staggered: >97% since retune
        # windows don't overlap when interval/n_modules > duration.
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


class TestPerPatternEfficiency:
    """Verify science fraction after inject_retune for each scan pattern.

    Cross-validates the efficiency-vs-pattern interaction using actual
    planner outputs (CE / Pong / Daisy) instead of the synthetic
    trajectories used elsewhere in this file.
    """

    def test_ce_scan_efficiency(self, site, start_time):
        """CE scan with 30s/5s retune should have ~80-87% science fraction."""
        field = FieldRegion(ra_center=24.0, dec_center=-32.0, width=40.0, height=10.0)
        block = plan_constant_el_scan(
            field=field,
            elevation=50.0,
            velocity=1.0,
            site=site,
            start_time=start_time,
            rising=True,
            timestep=0.1,
        )
        traj = block.trajectory
        result = inject_retune(traj, retune_interval=30.0, retune_duration=5.0)

        science_frac = result.science_mask.sum() / len(result.times)
        # CE scans have turnarounds that can absorb some retune time,
        # so efficiency should be in the 80-87% range
        assert 0.78 <= science_frac <= 0.90, (
            f"CE science fraction {science_frac:.3f} outside expected range [0.78, 0.90]"
        )

        # Retune flags should exist
        retune_count = (result.scan_flag == SCAN_FLAG_RETUNE).sum()
        assert retune_count > 0

    def test_pong_scan_efficiency(self, site, start_time):
        """Pong scan with 30s/5s retune should have ~80-87% science fraction.

        Pong scans have no turnaround flags (scan_flag is None),
        so inject_retune treats all samples as science and places
        retunes purely by time.
        """
        field = FieldRegion(ra_center=180.0, dec_center=-30.0, width=2.0, height=2.0)
        block = plan_pong_scan(
            field=field,
            velocity=0.5,
            spacing=0.1,
            num_terms=4,
            site=site,
            start_time=start_time,
            timestep=0.1,
        )
        traj = block.trajectory
        result = inject_retune(traj, retune_interval=30.0, retune_duration=5.0)

        science_frac = result.science_mask.sum() / len(result.times)
        # With scan_flag turnaround flagging + retune injection, science
        # fraction is lower than for patterns without turnaround flags.
        assert 0.65 <= science_frac <= 0.90, (
            f"Pong science fraction {science_frac:.3f} outside expected range [0.65, 0.90]"
        )

    def test_daisy_scan_efficiency(self, site, start_time):
        """Daisy scan with 30s/5s retune should have ~80-87% science fraction.

        Daisy scans are continuous (no turnarounds), so retune events
        always consume science time.
        """
        block = plan_daisy_scan(
            ra=180.0,
            dec=-30.0,
            radius=1.0,
            velocity=0.5,
            turn_radius=0.5,
            avoidance_radius=0.1,
            start_acceleration=0.5,
            site=site,
            start_time=start_time,
            timestep=0.1,
            duration=300.0,
        )
        traj = block.trajectory
        result = inject_retune(traj, retune_interval=30.0, retune_duration=5.0)

        science_frac = result.science_mask.sum() / len(result.times)
        assert 0.78 <= science_frac <= 0.90, (
            f"Daisy science fraction {science_frac:.3f} outside expected range [0.78, 0.90]"
        )

    def test_retune_flags_at_correct_intervals(self, site, start_time):
        """Retune events should appear at approximately the configured interval.

        Uses a daisy scan (no turnarounds) for clean interval verification.
        """
        block = plan_daisy_scan(
            ra=180.0,
            dec=-30.0,
            radius=1.0,
            velocity=0.5,
            turn_radius=0.5,
            avoidance_radius=0.1,
            start_acceleration=0.5,
            site=site,
            start_time=start_time,
            timestep=0.1,
            duration=300.0,
        )
        traj = block.trajectory
        result = inject_retune(
            traj, retune_interval=30.0, retune_duration=5.0, prefer_turnarounds=False
        )

        # Find retune event start times
        retune_mask = result.scan_flag == SCAN_FLAG_RETUNE
        retune_times = result.times[retune_mask]
        events = _group_retune_events(retune_times)

        # With 300s duration, 30s interval + 5s duration, effective
        # spacing is ~35s (next retune measured from end of previous).
        assert len(events) >= 6
        assert len(events) <= 10

        expected_gap = 30.0 + 5.0
        for i in range(1, len(events)):
            gap = events[i] - events[i - 1]
            assert abs(gap - expected_gap) < 1.0, (
                f"Retune interval {gap:.1f}s deviates from expected {expected_gap:.1f}s"
            )


class TestTurnaroundOverlap:
    """Verify turnaround snapping reduces dead time for CE scans."""

    def test_ce_turnaround_snapping_reduces_dead_time(self, site, start_time):
        """CE scan: snapping should preserve more science than time-based."""
        field = FieldRegion(ra_center=24.0, dec_center=-32.0, width=40.0, height=10.0)
        block = plan_constant_el_scan(
            field=field,
            elevation=50.0,
            velocity=1.0,
            site=site,
            start_time=start_time,
            rising=True,
            timestep=0.1,
        )
        traj = block.trajectory

        # Skip if scan is too short for meaningful comparison
        if traj.duration < 120.0:
            pytest.skip("CE scan too short for turnaround overlap test")

        result_snap = inject_retune(
            traj,
            retune_interval=30.0,
            retune_duration=5.0,
            prefer_turnarounds=True,
            turnaround_window=5.0,
        )
        result_time = inject_retune(
            traj,
            retune_interval=30.0,
            retune_duration=5.0,
            prefer_turnarounds=False,
        )

        frac_snap = result_snap.science_mask.sum() / len(result_snap.times)
        frac_time = result_time.science_mask.sum() / len(result_time.times)

        # Turnaround snapping should preserve at least as much science
        # time (>=) since it can overlap retunes with existing dead time.
        # Allow tiny tolerance for floating-point edge effects.
        assert frac_snap >= frac_time - 0.005, (
            f"Snapping ({frac_snap:.4f}) should be >= time-based ({frac_time:.4f})"
        )


class TestTheoreticalEfficiency:
    """Verify inject_retune efficiency matches theoretical predictions.

    Theoretical formula: efficiency = (interval - duration) / interval
    This assumes long trajectories where edge effects are negligible.
    """

    @pytest.mark.parametrize(
        "interval, duration, expected",
        [
            (30.0, 5.0, 0.833),  # 25/30 = 83.3%
            (60.0, 5.0, 0.917),  # 55/60 = 91.7%
            (30.0, 2.0, 0.933),  # 28/30 = 93.3%
        ],
        ids=["30s/5s", "60s/5s", "30s/2s"],
    )
    def test_efficiency_matches_theory(self, interval, duration, expected):
        """Long trajectory efficiency should match theoretical value within 3%."""
        traj = _make_trajectory(duration=600.0, timestep=0.1)
        result = inject_retune(
            traj,
            retune_interval=interval,
            retune_duration=duration,
            prefer_turnarounds=False,
        )

        science_frac = result.science_mask.sum() / len(result.times)
        assert abs(science_frac - expected) < 0.03, (
            f"Science fraction {science_frac:.4f} deviates from "
            f"theoretical {expected:.4f} by more than 3%"
        )

    def test_longer_trajectory_closer_to_theory(self):
        """Longer trajectories should have smaller edge effects.

        A 1200s trajectory should be closer to 83.3% than a 120s one
        with 30s/5s retune.
        """
        expected = 0.833

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
    """N-6: defensive guard for zero-velocity + prefer_turnarounds=True.

    The turnaround-snapping path in ``inject_retune`` scans for
    ``SCAN_FLAG_TURNAROUND`` samples in the input trajectory, but it
    still relies on the assumption that the velocity profile is
    meaningful.  The PrimeCam wrapper historically supplies
    identically-zero velocities, which would silently collapse all
    turnaround detection and produce wrong results.  The guard warns
    and falls back to time-based placement so callers notice.
    """

    def test_warns_on_zero_velocities_with_turnaround_snap(self):
        """Zero velocities + prefer_turnarounds=True must warn and fall back."""
        # Build a 300s trajectory whose az/el velocities are exactly zero.
        duration = 300.0
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

        with pytest.warns(PointingWarning, match="zero velocities"):
            result = inject_retune(
                traj,
                retune_interval=30.0,
                retune_duration=5.0,
                prefer_turnarounds=True,
            )

        # The fallback must still produce retune flags via the time-based
        # placement branch; the original scan_flag must not be mutated.
        retune_count = int((result.scan_flag == SCAN_FLAG_RETUNE).sum())
        assert retune_count > 0, "Time-based fallback should still place retune samples"
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


class TestRetuneEventDataclass:
    """Validation rules for the RetuneEvent dataclass itself."""

    def test_event_list_negative_duration_raises(self):
        """RetuneEvent(duration=-1.0) is rejected at construction."""
        with pytest.raises(ValueError, match="duration must be positive"):
            RetuneEvent(t_start=10.0, duration=-1.0)

    def test_event_list_zero_duration_raises(self):
        """RetuneEvent(duration=0.0) is rejected at construction."""
        with pytest.raises(ValueError, match="duration must be positive"):
            RetuneEvent(t_start=10.0, duration=0.0)

    def test_event_list_negative_tstart_raises(self):
        """RetuneEvent(t_start=-1.0) is rejected at construction."""
        with pytest.raises(ValueError, match="t_start must be non-negative"):
            RetuneEvent(t_start=-1.0, duration=5.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_event_list_non_finite_tstart_raises(self, bad):
        """Non-finite t_start values are rejected."""
        with pytest.raises(ValueError, match="t_start must be finite"):
            RetuneEvent(t_start=bad, duration=5.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_event_list_non_finite_duration_raises(self, bad):
        """Non-finite duration values are rejected."""
        with pytest.raises(ValueError, match="duration must be positive"):
            RetuneEvent(t_start=10.0, duration=bad)


class TestInjectRetuneEventList:
    """Event-list mode (``retune_events=...``) for inject_retune."""

    def test_event_list_roundtrip(self):
        """Three explicit events each land on the timeline; retune_events field preserves them."""
        traj = _make_trajectory(duration=300.0, timestep=0.1)
        events = [
            RetuneEvent(t_start=30.0, duration=5.0),
            RetuneEvent(t_start=120.0, duration=3.0),
            RetuneEvent(t_start=200.0, duration=8.0),
        ]
        result = inject_retune(traj, retune_events=events)

        # The Trajectory.retune_events field carries the validated, sorted tuple verbatim.
        assert result.retune_events == tuple(events)

        # Each event's window marks some samples as RETUNE.
        retune_times = result.times[result.scan_flag == SCAN_FLAG_RETUNE]
        for ev in events:
            window_mask = (retune_times >= ev.t_start) & (retune_times < ev.t_start + ev.duration)
            assert window_mask.any(), f"Event at t_start={ev.t_start} produced no retune samples"

    def test_event_list_empty(self):
        """Empty event list is a no-op; scan_flag unchanged; retune_events is empty."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        original_flags = traj.scan_flag.copy()

        result = inject_retune(traj, retune_events=[])

        # scan_flag is copied but not mutated.
        np.testing.assert_array_equal(result.scan_flag, original_flags)
        assert result.retune_events == ()

    def test_event_at_trajectory_boundary_skipped(self):
        """Event with t_start == times[-1] is skipped with PointingWarning."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        # Trajectory-relative end is times[-1] - times[0]; for this fixture
        # times[0] = 0, so t_end_rel ~= 119.9.
        t_end_rel = float(traj.times[-1] - traj.times[0])
        events = [RetuneEvent(t_start=t_end_rel, duration=5.0)]

        with pytest.warns(PointingWarning, match="past trajectory end"):
            result = inject_retune(traj, retune_events=events)

        # No retune samples produced.
        assert not (result.scan_flag == SCAN_FLAG_RETUNE).any()
        # retune_events field still carries the event tuple (the request was recorded).
        assert result.retune_events == tuple(events)

    def test_event_clipped_at_trajectory_end(self):
        """An event whose duration overruns the end is clipped, not truncated."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        # Event starts at t=115 (well inside), lasts 20 s -> would end at 135.
        events = [RetuneEvent(t_start=115.0, duration=20.0)]

        # Must not warn: the event starts inside the trajectory.
        with _warnings.catch_warnings(record=True) as records:
            _warnings.simplefilter("always")
            result = inject_retune(traj, retune_events=events)

        skips = [
            r
            for r in records
            if issubclass(r.category, PointingWarning) and "past trajectory end" in str(r.message)
        ]
        assert not skips

        # At least some retune samples must be present -- clipped but not empty.
        retune_mask = result.scan_flag == SCAN_FLAG_RETUNE
        assert retune_mask.any()
        # And they must all lie within the trajectory bounds.
        retune_times = result.times[retune_mask]
        assert retune_times.min() >= 115.0 - 1e-6
        assert retune_times.max() <= float(traj.times[-1])

    def test_event_overlap_same_module_raises(self):
        """Two overlapping events raise ValueError naming both indices."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        events = [
            RetuneEvent(t_start=30.0, duration=10.0),  # ends at 40
            RetuneEvent(t_start=35.0, duration=5.0),  # starts before previous ends
        ]
        with pytest.raises(ValueError, match="Overlapping retune events"):
            inject_retune(traj, retune_events=events)

    def test_event_in_turnaround_preserves_turnaround(self):
        """Event that falls entirely inside a turnaround window consumes no science."""
        # Turnaround at 29-45s. Event 30-40s sits inside.
        traj = _make_trajectory(
            duration=120.0,
            timestep=0.1,
            turnaround_intervals=[(29.0, 45.0)],
        )
        original_ta_count = int((traj.scan_flag == SCAN_FLAG_TURNAROUND).sum())
        events = [RetuneEvent(t_start=30.0, duration=10.0)]
        result = inject_retune(traj, retune_events=events)

        # No SCAN_FLAG_RETUNE samples -- all would-be retune samples were
        # already SCAN_FLAG_TURNAROUND and remain so.
        assert not (result.scan_flag == SCAN_FLAG_RETUNE).any()
        # Turnaround count unchanged.
        new_ta_count = int((result.scan_flag == SCAN_FLAG_TURNAROUND).sum())
        assert new_ta_count == original_ta_count

    def test_event_list_with_prefer_turnarounds_snaps(self):
        """Event 1 s before a turnaround start snaps to the turnaround."""
        # Turnaround at 30-35s. Event due at 29 -> should snap to 30 with
        # window=5.0.
        traj = _make_trajectory(
            duration=120.0,
            timestep=0.1,
            turnaround_intervals=[(30.0, 35.0)],
        )
        events = [RetuneEvent(t_start=29.0, duration=5.0)]
        result = inject_retune(
            traj,
            retune_events=events,
            prefer_turnarounds=True,
            turnaround_window=5.0,
        )
        retune_times = result.times[result.scan_flag == SCAN_FLAG_RETUNE]
        # Snap to 30 -> but samples 30..35 are TURNAROUND and don't get
        # overwritten; so the effective retune flags live at the tail of
        # the 30-35s window where the science region resumes. Since the
        # snapped window ends at 35s and the turnaround occupies 30-35s,
        # zero SCAN_FLAG_RETUNE samples should exist in this configuration.
        assert not retune_times.size, (
            "Snapped event should overlap the turnaround and leave zero new retune samples."
        )

    def test_event_list_with_prefer_turnarounds_no_turnaround_nearby(self):
        """Event far from any turnaround uses caller's t_start verbatim."""
        # Turnaround at 10-12s. Event at 70s -- window=5.0 should find
        # no turnaround nearby, so placement is literal.
        traj = _make_trajectory(
            duration=120.0,
            timestep=0.1,
            turnaround_intervals=[(10.0, 12.0)],
        )
        events = [RetuneEvent(t_start=70.0, duration=5.0)]
        result = inject_retune(
            traj,
            retune_events=events,
            prefer_turnarounds=True,
            turnaround_window=5.0,
        )
        retune_times = result.times[result.scan_flag == SCAN_FLAG_RETUNE]
        assert retune_times.size > 0
        # First retune sample should be at ~70s (not snapped to 10s).
        assert abs(float(retune_times[0]) - 70.0) < 0.15

    def test_event_list_applied_in_sorted_order_regardless_of_input_order(self):
        """Out-of-order input is sorted before application and before metadata attach."""
        traj = _make_trajectory(duration=300.0, timestep=0.1)
        events_unsorted = [
            RetuneEvent(t_start=200.0, duration=5.0),
            RetuneEvent(t_start=30.0, duration=5.0),
            RetuneEvent(t_start=120.0, duration=5.0),
        ]
        result_unsorted = inject_retune(traj, retune_events=events_unsorted)

        # retune_events tuple is sorted.
        stored = result_unsorted.retune_events
        assert [e.t_start for e in stored] == sorted(e.t_start for e in events_unsorted)

        # scan_flag is identical to passing the sorted list directly.
        events_sorted = sorted(events_unsorted, key=lambda e: e.t_start)
        result_sorted = inject_retune(traj, retune_events=events_sorted)
        np.testing.assert_array_equal(result_unsorted.scan_flag, result_sorted.scan_flag)


class TestInjectRetuneEventListMutualExclusion:
    """Guardrails for mixing ``retune_events`` with uniform-cadence kwargs."""

    def test_both_uniform_and_events_supplied_raises_on_module_index(self):
        """retune_events + module_index=1 -> ValueError."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        events = [RetuneEvent(t_start=30.0, duration=5.0)]
        with pytest.raises(ValueError, match="module_index"):
            inject_retune(traj, retune_events=events, module_index=1, n_modules=7)

    def test_both_uniform_and_events_supplied_raises_on_n_modules(self):
        """retune_events + n_modules=7 -> ValueError."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        events = [RetuneEvent(t_start=30.0, duration=5.0)]
        with pytest.raises(ValueError, match="n_modules"):
            inject_retune(traj, retune_events=events, n_modules=7)

    def test_both_uniform_and_events_supplied_warns_on_custom_interval(self):
        """retune_events + retune_interval=100.0 -> PointingWarning; events applied."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        events = [RetuneEvent(t_start=30.0, duration=5.0)]
        with pytest.warns(PointingWarning, match="retune_interval is ignored"):
            result = inject_retune(traj, retune_events=events, retune_interval=100.0)
        # Events were still applied.
        assert (result.scan_flag == SCAN_FLAG_RETUNE).any()
        assert result.retune_events == tuple(events)

    def test_both_uniform_and_events_supplied_warns_on_custom_duration(self):
        """retune_events + retune_duration=10.0 -> PointingWarning; events applied."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        events = [RetuneEvent(t_start=30.0, duration=5.0)]
        with pytest.warns(PointingWarning, match="retune_duration is ignored"):
            result = inject_retune(traj, retune_events=events, retune_duration=10.0)
        assert (result.scan_flag == SCAN_FLAG_RETUNE).any()

    def test_retune_events_with_defaults_does_not_warn(self):
        """Silent when retune_events is supplied with defaults for the uniform kwargs."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        events = [RetuneEvent(t_start=30.0, duration=5.0)]
        with _warnings.catch_warnings(record=True) as records:
            _warnings.simplefilter("always")
            inject_retune(traj, retune_events=events)
        ignored = [
            r
            for r in records
            if issubclass(r.category, PointingWarning)
            and ("is ignored" in str(r.message) or "retune_interval" in str(r.message))
        ]
        assert not ignored


class TestSampleRetuneEvents:
    """Unit tests for the ``sample_retune_events`` helper."""

    def test_sample_retune_events_seeded_reproducible(self):
        """Same seed -> identical event list across two calls."""

        def _make_rng():
            return np.random.default_rng(seed=12345)

        def interval(r):
            return float(r.uniform(30.0, 90.0))

        def dur(r):
            return float(r.uniform(2.0, 8.0))

        first = sample_retune_events(
            duration=1000.0,
            interval_sampler=interval,
            duration_sampler=dur,
            rng=_make_rng(),
        )
        second = sample_retune_events(
            duration=1000.0,
            interval_sampler=interval,
            duration_sampler=dur,
            rng=_make_rng(),
        )

        assert first == second
        # Sanity: non-overlapping, chronological.
        for i in range(1, len(first)):
            assert first[i].t_start >= first[i - 1].t_start + first[i - 1].duration

    def test_sample_retune_events_negative_interval_raises(self):
        """Sampler returning negative interval -> ValueError naming the sampler."""
        rng = np.random.default_rng(seed=1)
        with pytest.raises(ValueError, match="interval_sampler"):
            sample_retune_events(
                duration=100.0,
                interval_sampler=lambda r: -5.0,
                duration_sampler=lambda r: 5.0,
                rng=rng,
            )

    def test_sample_retune_events_negative_duration_raises(self):
        """Sampler returning negative duration -> ValueError naming the sampler."""
        rng = np.random.default_rng(seed=1)
        with pytest.raises(ValueError, match="duration_sampler"):
            sample_retune_events(
                duration=100.0,
                interval_sampler=lambda r: 20.0,
                duration_sampler=lambda r: -3.0,
                rng=rng,
            )

    def test_sample_retune_events_zero_window_returns_empty(self):
        """Zero-duration window yields zero events without raising."""
        rng = np.random.default_rng(seed=1)
        events = sample_retune_events(
            duration=0.0,
            interval_sampler=lambda r: 10.0,
            duration_sampler=lambda r: 5.0,
            rng=rng,
        )
        assert events == []

    def test_sample_retune_events_feeds_inject_retune(self):
        """End-to-end: sampled events can drive inject_retune without error."""
        rng = np.random.default_rng(seed=7)
        events = sample_retune_events(
            duration=600.0,
            interval_sampler=lambda r: float(r.uniform(60.0, 120.0)),
            duration_sampler=lambda r: float(r.uniform(3.0, 6.0)),
            rng=rng,
        )
        traj = _make_trajectory(duration=600.0, timestep=0.1)
        result = inject_retune(traj, retune_events=events)
        assert (result.scan_flag == SCAN_FLAG_RETUNE).any()
        assert result.retune_events == tuple(events)


class TestInjectRetuneMetadataPreservation:
    """Round-2 B1 regression: ``inject_retune`` must not mutate ``metadata``.

    The Phase-1 implementation stashed ``retune_events`` inside
    ``trajectory.metadata`` as a dict key, which silently broke
    ``Trajectory.pattern_type`` / ``.center_ra`` / ``.pattern_params``
    accessors. The new design promotes ``retune_events`` to a first-class
    field on :class:`Trajectory`; ``metadata`` stays untouched.
    """

    def test_typed_metadata_survives_event_injection(self):
        """A ``TrajectoryMetadata`` instance round-trips through inject_retune."""
        from fyst_trajectories import TrajectoryMetadata

        meta = TrajectoryMetadata(
            pattern_type="pong",
            pattern_params={"width": 2.0, "height": 2.0},
            center_ra=180.0,
            center_dec=-30.0,
        )
        times = np.arange(0, 300.0, 0.1)
        n = len(times)
        traj = Trajectory(
            times=times,
            az=np.linspace(100, 200, n),
            el=np.full(n, 45.0),
            az_vel=np.gradient(np.linspace(100, 200, n), times),
            el_vel=np.zeros(n),
            metadata=meta,
            scan_flag=np.full(n, SCAN_FLAG_SCIENCE, dtype=np.int8),
        )

        events = [
            RetuneEvent(t_start=30.0, duration=5.0),
            RetuneEvent(t_start=150.0, duration=4.0),
        ]
        result = inject_retune(traj, retune_events=events)

        # Pattern accessors must still work — the B1 blocker.
        assert result.pattern_type == "pong"
        assert result.center_ra == 180.0
        assert result.center_dec == -30.0
        assert result.pattern_params == {"width": 2.0, "height": 2.0}

        # Metadata is the exact same object — we don't mutate it.
        assert result.metadata is traj.metadata

        # The new first-class field carries the sorted event tuple.
        assert result.retune_events == tuple(events)


class TestInjectRetuneOutOfBoundsReporting:
    """Round-2 review nit: OOB warning must name *sorted* indices, not input order."""

    def test_multiple_oob_events_single_warning_with_sorted_indices(self):
        """Three out-of-bounds events produce exactly one warning naming all three."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        t_end_rel = float(traj.times[-1] - traj.times[0])
        # Three events, all past the trajectory end. Input order differs from
        # sorted order so we can also verify the warning speaks in sorted terms.
        events = [
            RetuneEvent(t_start=t_end_rel + 200.0, duration=5.0),
            RetuneEvent(t_start=t_end_rel + 50.0, duration=5.0),
            RetuneEvent(t_start=t_end_rel + 100.0, duration=5.0),
        ]

        with pytest.warns(PointingWarning, match="past trajectory end") as record:
            result = inject_retune(traj, retune_events=events)

        # Exactly one warning captured.
        assert len(record) == 1
        msg = str(record[0].message)
        # All three sorted indices (0, 1, 2) should appear in the message.
        for idx in (0, 1, 2):
            assert f"sorted_index={idx}" in msg, f"expected sorted_index={idx} in {msg!r}"
        # The message must explicitly mention that the indices are sorted.
        assert "sorted" in msg.lower()
        # No retune samples produced.
        assert not (result.scan_flag == SCAN_FLAG_RETUNE).any()


class TestInjectRetuneTimesOffset:
    """Round-2 review nit: ``times[0] != 0`` must map correctly in event-list mode."""

    def test_times_offset_event_placement(self):
        """Events are trajectory-relative; times[0] offset is respected."""
        times = np.arange(100.0, 200.0, 0.1)
        n = len(times)
        scan_flag = np.full(n, SCAN_FLAG_SCIENCE, dtype=np.int8)
        traj = Trajectory(
            times=times,
            az=np.linspace(0, 10, n),
            el=np.full(n, 45.0),
            az_vel=np.gradient(np.linspace(0, 10, n), times),
            el_vel=np.zeros(n),
            scan_flag=scan_flag,
        )

        events = [RetuneEvent(t_start=10.0, duration=5.0)]
        result = inject_retune(traj, retune_events=events)

        retune_times = result.times[result.scan_flag == SCAN_FLAG_RETUNE]
        assert retune_times.size > 0
        # Events are trajectory-relative, so t_start=10 maps to times[0] + 10 = 110.
        assert retune_times.min() >= 110.0 - 1e-6
        assert retune_times.max() < 115.0


class TestInjectRetuneUniformEquivalence:
    """Round-2 review nit: event-list mode with uniform-matching events is identical."""

    def test_uniform_and_equivalent_event_list_produce_same_scan_flag(self):
        """Hand-built events matching the uniform cadence produce identical scan_flag."""
        traj = _make_trajectory(duration=300.0, timestep=0.1)

        # Uniform path.
        uniform_result = inject_retune(
            traj, retune_interval=60.0, retune_duration=5.0, prefer_turnarounds=False
        )

        # Reconstruct the same events the uniform path generates and pass them in.
        # The uniform path anchors on times[0] and advances by
        # ``max(retune_end, due_time)``, so the first event starts at
        # times[0] + retune_interval (relative: retune_interval), the next at
        # retune_end + retune_interval (relative: retune_interval + retune_duration
        # + retune_interval), and so on.
        uniform_events_for_event_mode = tuple(uniform_result.retune_events)
        event_result = inject_retune(traj, retune_events=list(uniform_events_for_event_mode))

        np.testing.assert_array_equal(uniform_result.scan_flag, event_result.scan_flag)
        assert uniform_result.retune_events == event_result.retune_events


class TestInjectRetuneAdjacentAndOverlap:
    """Round-2 review nit: boundary-touching events OK; same-start-different-duration is overlap."""

    def test_adjacent_events_accepted(self):
        """Two events where ``a.t_start + a.duration == b.t_start`` must not raise."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        events = [
            RetuneEvent(t_start=30.0, duration=5.0),  # ends at 35
            RetuneEvent(t_start=35.0, duration=4.0),  # starts exactly where a ends
        ]
        # Must not raise.
        result = inject_retune(traj, retune_events=events)

        # Both events should produce retune samples.
        retune_times = result.times[result.scan_flag == SCAN_FLAG_RETUNE]
        first_window = retune_times[(retune_times >= 30.0) & (retune_times < 35.0)]
        second_window = retune_times[(retune_times >= 35.0) & (retune_times < 39.0)]
        assert first_window.size > 0, "First (pre-boundary) event produced no retune samples"
        assert second_window.size > 0, "Second (post-boundary) event produced no retune samples"

    def test_same_tstart_different_durations_overlap_raises(self):
        """Two events sharing ``t_start`` with different durations overlap."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        events = [
            RetuneEvent(t_start=30.0, duration=5.0),
            RetuneEvent(t_start=30.0, duration=3.0),
        ]
        with pytest.raises(ValueError, match="Overlapping retune events"):
            inject_retune(traj, retune_events=events)


class TestUniformPathPopulatesRetuneEvents:
    """Round-2 design: uniform-cadence path must populate ``Trajectory.retune_events``."""

    def test_uniform_path_retune_events_populated(self):
        """Calling uniform-cadence inject_retune sets retune_events symmetrically."""
        # 300 s / 60 s interval = 5 retunes at approximately t=60, 125, 190, 255, ... but
        # the anchor advances by max(retune_end, due_time), so cadence is
        # interval + duration = 65. Expected events: 5 (60, 125, 190, 255, ...).
        duration = 300.0
        interval = 60.0
        dur = 5.0
        traj = _make_trajectory(duration=duration, timestep=0.1)
        result = inject_retune(
            traj, retune_interval=interval, retune_duration=dur, prefer_turnarounds=False
        )

        # retune_events is non-empty and sorted.
        assert len(result.retune_events) > 0
        t_starts = [e.t_start for e in result.retune_events]
        assert t_starts == sorted(t_starts)

        # First event starts at t=interval (first due_time after times[0]=0).
        assert result.retune_events[0].t_start == pytest.approx(interval)

        # Every event has duration == the requested retune_duration.
        for ev in result.retune_events:
            assert ev.duration == pytest.approx(dur)

        # Cross-check count: expected ~ (duration - interval) / (interval + dur) + 1.
        # For 300, 60, 5: (300 - 60) / 65 + 1 = ~4.69 -> 4 or 5 events. Accept either.
        expected_min = int((duration - interval) / (interval + dur))
        expected_max = expected_min + 2
        assert expected_min <= len(result.retune_events) <= expected_max
