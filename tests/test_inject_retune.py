"""Tests for inject_retune() trajectory utility.

Covers basic placement, turnaround snapping, edge cases, the per-module
staggered mode, the per-pattern efficiency cross-validation against
real planner outputs (CE / Pong / Daisy), turnaround-overlap dead-time
reduction, theoretical efficiency parametric checks, and the
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
    """A continuous scan with no turnaround flags gets time-based retunes."""

    def test_no_turnarounds_uses_time_based(self):
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        result = inject_retune(
            traj, retune_interval=30.0, retune_duration=5.0, prefer_turnarounds=True
        )

        retune_mask = result.scan_flag == SCAN_FLAG_RETUNE
        assert retune_mask.any()


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
        assert 0.80 < science_fraction < 0.87


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
        """A gap at least as long as its cadence would flag every science sample."""
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
        """Pong scan with 30s/5s retune should have ~76% science fraction.

        The planned pong carries turnaround flags (~13% of samples), which
        inject_retune never overwrites, so its science fraction sits below the
        30/35 steady state of a flag-free trajectory.
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
        assert 0.65 <= science_frac <= 0.90, (
            f"Pong science fraction {science_frac:.3f} outside expected range [0.65, 0.90]"
        )

    def test_daisy_scan_efficiency(self, site, start_time):
        """Daisy scan with 30s/5s retune should have ~80-87% science fraction.

        Daisy scans are nearly continuous (a handful of turnaround samples), so
        retune events consume science time almost everywhere.
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

        Uses a daisy scan (only a handful of turnaround samples) for clean
        interval verification.
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

    The turnaround-snapping path in ``inject_retune`` scans for
    ``SCAN_FLAG_TURNAROUND`` samples in the input trajectory, but it
    still relies on the assumption that the velocity profile is
    meaningful.  A caller that supplies identically-zero velocities would
    silently collapse all turnaround detection and produce wrong results.
    The guard warns and falls back to time-based placement so callers
    notice.
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


def test_event_clipped_to_end_flags_final_sample():
    """A retune event clipped to the trajectory end flags the final science sample."""
    times = np.arange(0.0, 11.0, 1.0)  # 0..10 s, 11 samples
    n = len(times)
    traj = Trajectory(
        times=times,
        az=np.zeros(n),
        el=np.full(n, 45.0),
        az_vel=np.zeros(n),
        el_vel=np.zeros(n),
    )
    # Event starts at t=8 and outlasts the trajectory, so it clips to t_end=10;
    # the sample at exactly t_end must still be flagged (an off-by-one drops it).
    result = inject_retune(traj, retune_events=[RetuneEvent(t_start=8.0, duration=10.0)])
    assert result.scan_flag[-1] == SCAN_FLAG_RETUNE
    flagged = result.scan_flag[times >= 8.0]
    assert (flagged == SCAN_FLAG_RETUNE).all()


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
    """Event-list mode: placement, clipping, overlap refusal, and sorting of the request."""

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

        # At least some retune samples must be present, clipped but not empty.
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

        # No SCAN_FLAG_RETUNE samples: all would-be retune samples were
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
        # Turnaround at 10-12s. Event at 70s, window=5.0 should find
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
    """Seeded sampling is reproducible, non-overlapping, and feeds ``inject_retune``."""

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
    """Regression: ``inject_retune`` must not mutate ``metadata``.

    Stashing ``retune_events`` inside ``trajectory.metadata`` as a dict key
    would silently break the ``Trajectory.pattern_type`` / ``.center_ra`` /
    ``.pattern_params`` accessors. ``retune_events`` is instead a first-class
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

        # Pattern accessors must still work.
        assert result.pattern_type == "pong"
        assert result.center_ra == 180.0
        assert result.center_dec == -30.0
        assert result.pattern_params == {"width": 2.0, "height": 2.0}

        # Metadata is the exact same object, we don't mutate it.
        assert result.metadata is traj.metadata

        # The first-class field carries the sorted event tuple.
        assert result.retune_events == tuple(events)


class TestInjectRetuneOutOfBoundsReporting:
    """Regression: OOB warning must name *sorted* indices, not input order."""

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
    """Regression: ``times[0] != 0`` must map correctly in event-list mode."""

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
    """Regression: event-list mode with uniform-matching events is identical."""

    def test_uniform_and_equivalent_event_list_produce_same_scan_flag(self):
        """The uniform path's own events, replayed through event-list mode, match."""
        traj = _make_trajectory(duration=300.0, timestep=0.1)

        # Uniform path.
        uniform_result = inject_retune(
            traj, retune_interval=60.0, retune_duration=5.0, prefer_turnarounds=False
        )

        # Feed the uniform path's own emitted events back through the
        # event-list path.
        uniform_events_for_event_mode = tuple(uniform_result.retune_events)
        event_result = inject_retune(traj, retune_events=list(uniform_events_for_event_mode))

        np.testing.assert_array_equal(uniform_result.scan_flag, event_result.scan_flag)
        assert uniform_result.retune_events == event_result.retune_events


class TestInjectRetuneAdjacentAndOverlap:
    """Regression: boundary-touching events OK; same-start-different-duration is overlap."""

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


class TestEventListElementTypes:
    """Event-list mode says what it wanted when given something else.

    The sort runs first, so an unchecked tuple or dict dies as an
    ``AttributeError`` from inside the sort key, naming ``t_start`` and no
    argument.
    """

    @pytest.mark.parametrize("bad", [(10.0, 5.0), {"t_start": 10.0, "duration": 5.0}, 10.0, None])
    def test_non_event_element_is_refused(self, bad):
        """A non-``RetuneEvent`` element raises a TypeError naming its index."""
        traj = _make_trajectory(duration=120.0)
        events = [RetuneEvent(t_start=10.0, duration=5.0), bad]
        with pytest.raises(TypeError, match=r"retune_events\[1\] is a"):
            inject_retune(traj, retune_events=events)


class TestSnappingCannotSilentlyMergeEvents:
    """Overlap is re-checked against the placements, not only the request.

    Checking overlap before ``prefer_turnarounds`` snapping lets two events
    that do not overlap as asked be pulled onto the same turnaround: the
    second then paints samples the first already owns and disappears, with
    the caller told nothing.
    """

    def test_two_events_snapped_onto_one_turnaround_raise(self):
        """Both events land on the same turnaround and the merge is refused."""
        traj = _make_trajectory(duration=120.0, turnaround_intervals=[(30.0, 31.0), (60.0, 61.0)])
        # 28 s and 33 s both snap to the turnaround starting at 30 s within a
        # 10 s window, and a 5 s gap at 30 s runs into a second one placed there.
        events = [RetuneEvent(t_start=28.0, duration=5.0), RetuneEvent(t_start=33.0, duration=5.0)]
        with pytest.raises(ValueError, match="overlap after snapping to turnarounds"):
            inject_retune(
                traj, retune_events=events, prefer_turnarounds=True, turnaround_window=10.0
            )

    def test_events_snapping_to_different_turnarounds_are_fine(self):
        """Well-separated events still snap and apply."""
        traj = _make_trajectory(duration=120.0, turnaround_intervals=[(30.0, 31.0), (60.0, 61.0)])
        events = [RetuneEvent(t_start=28.0, duration=5.0), RetuneEvent(t_start=62.0, duration=5.0)]
        result = inject_retune(
            traj, retune_events=events, prefer_turnarounds=True, turnaround_window=10.0
        )
        assert np.count_nonzero(result.scan_flag == SCAN_FLAG_RETUNE) > 0
        assert result.retune_events == tuple(events)

    def test_without_snapping_the_same_events_apply(self):
        """The snapping check is snapping-specific; the request itself does not overlap."""
        traj = _make_trajectory(duration=120.0, turnaround_intervals=[(30.0, 31.0), (60.0, 61.0)])
        events = [RetuneEvent(t_start=28.0, duration=5.0), RetuneEvent(t_start=33.0, duration=5.0)]
        result = inject_retune(traj, retune_events=events)
        assert result.retune_events == tuple(events)
