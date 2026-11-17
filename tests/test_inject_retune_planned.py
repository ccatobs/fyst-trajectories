"""Tests for inject_retune() on trajectories built by the planners.

Flag accounting on a planned CE trajectory (the science_mask reduction and
its efficiency ratio), per-pattern efficiency against real CE, pong and daisy
planner outputs, and turnaround-overlap dead-time reduction.
"""

import numpy as np
import pytest
from _retune_stubs import _group_retune_events
from astropy.time import Time

from fyst_trajectories import get_fyst_site
from fyst_trajectories.patterns.turnarounds import turnaround_duration_sec
from fyst_trajectories.planning import (
    FieldRegion,
    plan_constant_el_scan,
    plan_daisy_scan,
    plan_pong_scan,
)
from fyst_trajectories.retune import inject_retune
from fyst_trajectories.trajectory import (
    SCAN_FLAG_RETUNE,
    SCAN_FLAG_SCIENCE,
    SCAN_FLAG_TURNAROUND,
)


@pytest.fixture
def ce_trajectory():
    """Plan a CE trajectory over a 30 x 8 deg field centred on E-CDF-S (RA 53.1, Dec -27.8)."""
    site = get_fyst_site()
    field = FieldRegion(ra_center=53.1, dec_center=-27.8, width=30.0, height=8.0)
    block = plan_constant_el_scan(
        field=field,
        elevation=50.0,
        velocity=1.0,
        site=site,
        start_time=Time("2026-06-15T04:00:00", scale="utc"),
        rising=True,
        timestep=0.1,
    )
    return block.trajectory


def _turnaround_fraction(trajectory) -> float:
    """Return the share of a trajectory's samples flagged as turnaround."""
    return float((trajectory.scan_flag == SCAN_FLAG_TURNAROUND).mean())


def _expected_turnaround_fraction(trajectory, velocity: float, az_accel: float) -> float:
    """Return the share of a constant-elevation pass its turnarounds take, from its legs.

    The legs reverse at the sweep's ends, so a pass turns at every sign change
    of its azimuth velocity, each turnaround lasting ``2 v / a``.
    """
    signs = np.sign(trajectory.az_vel[np.abs(trajectory.az_vel) > 0.5 * velocity])
    n_turnarounds = int(np.count_nonzero(np.diff(signs)))
    duration = float(trajectory.times[-1] - trajectory.times[0])
    return n_turnarounds * turnaround_duration_sec(velocity, az_accel) / duration


# Retunes start at a fixed cadence whatever the legs do, so a 5 s window in every
# 35 s covers science and turnaround samples in proportion, and only the science
# ones are overwritten: the science share is (30 / 35) times the share the
# turnarounds leave. The 0.002 allowed covers retunes landing on turnarounds
# unevenly (measured within 5e-4); the turnaround share is held to its geometry
# to 1e-3 (measured within 2e-5).
_RETUNE_SHARE_TOLERANCE = 0.002
_TURNAROUND_SHARE_TOLERANCE = 1e-3


class TestScienceMaskReduction:
    """Verify science_mask correctly tracks retune overhead."""

    def test_science_mask_reduces_sample_count(self, ce_trajectory):
        original_science = ce_trajectory.science_mask.sum()
        result = inject_retune(
            ce_trajectory,
            retune_interval=30.0,
            retune_duration=5.0,
        )
        reduced_science = result.science_mask.sum()

        assert reduced_science < original_science, (
            f"Expected fewer science samples after retune injection: "
            f"original={original_science}, after={reduced_science}"
        )

    def test_science_ratio_matches_efficiency(self, ce_trajectory):
        """Ratio of science samples to total should match expected efficiency.

        For 30s/5s retune, theoretical efficiency is ~85.7% (30 s of science
        per 35 s cadence).
        CE scans have turnarounds that reduce effective science fraction
        further, so the ratio sits below that steady state.
        """
        result = inject_retune(
            ce_trajectory,
            retune_interval=30.0,
            retune_duration=5.0,
        )

        ratio = result.science_mask.sum() / len(result.times)

        # The turnarounds take the share the pass's legs imply, so
        # dropping their flags fails this, not only the ratio below.
        turnaround = _turnaround_fraction(ce_trajectory)
        assert turnaround == pytest.approx(
            _expected_turnaround_fraction(ce_trajectory, 1.0, 1.0),
            abs=_TURNAROUND_SHARE_TOLERANCE,
        )
        assert ratio == pytest.approx(
            (30.0 / 35.0) * (1.0 - turnaround), abs=_RETUNE_SHARE_TOLERANCE
        )

    def test_retune_and_turnaround_exclusive(self, ce_trajectory):
        """Retune flags should never overwrite turnaround flags."""
        result = inject_retune(
            ce_trajectory,
            retune_interval=30.0,
            retune_duration=5.0,
        )

        # Count turnaround samples before and after
        original_ta = (ce_trajectory.scan_flag == SCAN_FLAG_TURNAROUND).sum()
        result_ta = (result.scan_flag == SCAN_FLAG_TURNAROUND).sum()

        assert result_ta == original_ta, f"Turnaround count changed: {original_ta} -> {result_ta}"

    def test_flag_values_partition(self, ce_trajectory):
        """Every sample should have exactly one of: science, turnaround, retune."""
        result = inject_retune(
            ce_trajectory,
            retune_interval=30.0,
            retune_duration=5.0,
        )

        n_sci = (result.scan_flag == SCAN_FLAG_SCIENCE).sum()
        n_ta = (result.scan_flag == SCAN_FLAG_TURNAROUND).sum()
        n_ret = (result.scan_flag == SCAN_FLAG_RETUNE).sum()
        total = len(result.scan_flag)

        assert n_sci + n_ta + n_ret == total, (
            f"Flag partition mismatch: "
            f"science={n_sci} + turnaround={n_ta} + retune={n_ret} "
            f"!= total={total}"
        )


@pytest.fixture
def start_time():
    """Nighttime start at FYST."""
    return Time("2026-06-15T04:00:00", scale="utc")


class TestPerPatternEfficiency:
    """Verify science fraction after inject_retune for each scan pattern.

    Cross-validates the efficiency-vs-pattern interaction on CE, Pong and
    Daisy planner outputs.
    """

    def test_ce_scan_efficiency(self, site, start_time):
        """CE with a 30 s / 5 s retune: science is 30/35 of the share the turnarounds leave."""
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
        # The planned CE carries turnaround flags that inject_retune never
        # overwrites, so its science fraction is the 30/35 steady state of a
        # flag-free trajectory times the share the turnarounds leave.
        turnaround = _turnaround_fraction(traj)
        assert turnaround == pytest.approx(
            _expected_turnaround_fraction(traj, 1.0, 1.0), abs=_TURNAROUND_SHARE_TOLERANCE
        )
        assert science_frac == pytest.approx(
            (30.0 / 35.0) * (1.0 - turnaround), abs=_RETUNE_SHARE_TOLERANCE
        )

        # Retune flags should exist
        retune_count = (result.scan_flag == SCAN_FLAG_RETUNE).sum()
        assert retune_count > 0

    @pytest.mark.filterwarnings(
        "ignore:Trajectory elevation acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
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
        assert 0.65 <= science_frac < 30.0 / 35.0, (
            f"Pong science fraction {science_frac:.3f} outside expected range [0.65, 30/35)"
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
        assert len(events) == 8  # 30, 65, ..., 275 s: a 35 s cadence in 300 s

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

        # Snapping moves retunes onto turnaround samples that are already lost
        # to science, so it keeps strictly more science than time-based placement.
        assert frac_snap > frac_time, (
            f"Snapping ({frac_snap:.4f}) should be > time-based ({frac_time:.4f})"
        )
