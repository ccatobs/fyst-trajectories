"""Regression tests for timeline generation.

Expected values were computed once from a known-good run and are
hardcoded here as regression anchors.

Any change to these values means the timeline generation algorithm
changed, which requires explicit acknowledgment and updating the
anchors.

Two properties of this fixture night explain the shape of the schedule.
Constant-elevation science is corridor-gated (see
tests/overhead/test_ce_corridor.py), so the schedule idles until Deep56's el=50
rising pass is imminent and books ~1.8 h of science rather than pointing at
empty sky beforehand. And a cadence-0 retune is scan-coupled: it fires once at
startup and then immediately before every science subscan, never on an idle
tick, so each of the three subscans pays a 300 s whole-array retune out of the
crossing corridor.
"""

import pytest

from fyst_trajectories import get_fyst_site
from fyst_trajectories.overhead import (
    BudgetStats,
    CalibrationBudget,
    CalibrationPolicy,
    ObservingPatch,
    OverheadModel,
    PatchBudget,
    compute_budget,
    generate_timeline,
)


@pytest.fixture(scope="module")
def regression_timeline():
    """Generate a timeline with fixed, known inputs for regression testing.

    Uses two patches (one CE scan, one Pong scan) with an 8-hour
    nighttime window. The COSMOS patch is below elevation limits
    during this window, so only Deep56 is scheduled. This is
    intentional and tests the constraint system.
    """
    site = get_fyst_site()
    patches = [
        ObservingPatch(
            name="Deep56",
            ra_center=24.0,
            dec_center=-32.0,
            width=40.0,
            height=10.0,
            scan_type="constant_el",
            velocity=1.0,
            elevation=50.0,
        ),
        ObservingPatch(
            name="COSMOS",
            ra_center=150.0,
            dec_center=2.2,
            width=4.0,
            height=4.0,
            scan_type="pong",
            velocity=0.5,
        ),
    ]
    overhead = OverheadModel()
    policy = CalibrationPolicy(
        retune_cadence=0.0,
        pointing_cadence=3600.0,
        focus_cadence=7200.0,
        skydip_cadence=10800.0,
        planet_cal_cadence=43200.0,
    )
    timeline = generate_timeline(
        patches=patches,
        site=site,
        start_time="2026-06-15T02:00:00",
        end_time="2026-06-15T10:00:00",
        overhead_model=overhead,
        calibration_policy=policy,
    )
    return timeline


class TestRegressionTimeline:
    """Verify that timeline generation produces known-good outputs.

    These anchors were computed from the initial extraction and serve
    as regression baselines. Tolerances are tight, any significant
    change signals an algorithm or API change.
    """

    def test_block_count(self, regression_timeline):
        """Total number of blocks should be stable."""
        assert len(regression_timeline.blocks) == 80

    def test_science_scan_count(self, regression_timeline):
        """Number of science scans should be stable."""
        assert regression_timeline.n_science_scans == 3

    def test_calibration_block_count(self, regression_timeline):
        """Number of calibration blocks should be stable."""
        assert len(regression_timeline.calibration_blocks) == 16

    def test_science_time(self, regression_timeline):
        """Total science time should match within 1 second."""
        assert abs(regression_timeline.total_science_time - 6391.3) < 1.0

    def test_calibration_time(self, regression_timeline):
        """Total calibration time should match exactly (deterministic)."""
        assert abs(regression_timeline.total_calibration_time - 4380.0) < 0.1

    def test_efficiency(self, regression_timeline):
        """Science efficiency should match within 0.1%."""
        assert abs(regression_timeline.efficiency - 0.2219) < 0.001

    def test_block_type_distribution(self, regression_timeline):
        """Block type counts should match expected distribution."""
        from collections import Counter

        type_counts = Counter(b.block_type for b in regression_timeline.blocks)
        assert type_counts["calibration"] == 16
        assert type_counts["slew"] == 1
        assert type_counts["science"] == 3
        assert type_counts.get("idle", 0) == 60

    def test_only_deep56_scheduled(self, regression_timeline):
        """COSMOS is below elevation limits; only Deep56 should be scheduled."""
        patch_names = {b.patch_name for b in regression_timeline.science_blocks}
        assert patch_names == {"Deep56"}

    def test_calibration_breakdown(self, regression_timeline):
        """Calibration types and counts should match regression anchors."""
        stats = compute_budget(regression_timeline)
        cal = stats["calibration_breakdown"]

        # Scan-coupled retunes (cadence 0): one at startup plus one
        # immediately before each of the 3 subscans, none during idle.
        assert cal["retune"]["count"] == 4
        assert abs(cal["retune"]["total_time"] - 1200.0) < 0.1

        assert cal["pointing_cal"]["count"] == 6
        assert abs(cal["pointing_cal"]["total_time"] - 1080.0) < 0.1

        assert cal["focus"]["count"] == 3
        assert abs(cal["focus"]["total_time"] - 900.0) < 0.1

        assert cal["skydip"]["count"] == 2
        assert abs(cal["skydip"]["total_time"] - 600.0) < 0.1

        assert cal["planet_cal"]["count"] == 1
        assert abs(cal["planet_cal"]["total_time"] - 600.0) < 0.1

    def test_slew_time(self, regression_timeline):
        """Total slew time should match within 2 seconds."""
        slew_time = sum(b.duration for b in regression_timeline.blocks if b.block_type == "slew")
        # One slew: the single CE visit starts once the pass is imminent.
        assert abs(slew_time - 28.7) < 2.0

    def test_idle_time_is_the_pre_pass_wait(self, regression_timeline):
        """Idle time equals the honest wait before Deep56's crossing pass.

        Deep56's el=50 rising pass opens ~08:02 UTC and the corridor gate
        admits it one tick earlier, at 07:58; until then nothing on this
        fixture night is observable, and the corridor-gated scheduler reports
        that as idle instead of booking dead-air science.
        """
        idle_time = sum(b.duration for b in regression_timeline.blocks if b.block_type == "idle")
        assert abs(idle_time - 18000.0) < 1.0

    def test_timeline_validates_clean(self, regression_timeline):
        """Timeline should pass internal validation with no warnings."""
        warnings = regression_timeline.validate()
        assert warnings == []

    def test_compute_budget_keys(self, regression_timeline):
        """compute_budget() returns exactly the keys its schema declares.

        The expected set is derived from ``BudgetStats`` rather than written
        out again: the return type used to be a bare ``dict`` whose shape was
        pinned only by a hand-maintained literal here, so a key added to the
        summary and not to this list went unnoticed.
        """
        stats = compute_budget(regression_timeline)
        assert set(stats.keys()) == set(BudgetStats.__annotations__)
        assert set(stats.keys()) == {
            "total_time",
            "science_time",
            "calibration_time",
            "slew_time",
            "idle_time",
            "efficiency",
            "n_science_scans",
            "n_calibration_blocks",
            "per_patch",
            "calibration_breakdown",
        }

    def test_compute_budget_nested_keys_match_their_schemas(self, regression_timeline):
        """The per-patch and per-calibration entries match their own TypedDicts."""
        stats = compute_budget(regression_timeline)
        for entry in stats["per_patch"].values():
            assert set(entry) == set(PatchBudget.__annotations__)
        for entry in stats["calibration_breakdown"].values():
            assert set(entry) == set(CalibrationBudget.__annotations__)

    def test_total_time_conservation(self, regression_timeline):
        """All block durations should sum to less than total timeline span.

        Blocks may not cover the entire timeline (gaps at the end),
        but should never exceed it.
        """
        block_total = sum(b.duration for b in regression_timeline.blocks)
        timeline_span = regression_timeline.total_time
        assert block_total <= timeline_span + 1.0  # 1s tolerance

    def test_block_ordering(self, regression_timeline):
        """Blocks should be in chronological order."""
        blocks = regression_timeline.blocks
        for i in range(len(blocks) - 1):
            assert blocks[i].t_start.unix <= blocks[i + 1].t_start.unix

    def test_initial_calibration_sequence(self, regression_timeline):
        """First blocks should be the initial calibration burst.

        The scheduler performs the due in-place calibrations at startup:
        retune, pointing_cal, focus, skydip. The planet calibration waits
        for a planet above ``planet_min_elevation``, which on this night is
        not until ~06:45 UTC.
        """
        initial_cals = []
        for b in regression_timeline.blocks:
            if b.block_type != "calibration":
                break
            initial_cals.append(str(b.scan_type))

        assert initial_cals == ["retune", "pointing_cal", "focus", "skydip"]

    def test_ce_visit_reconstruction_tiles_blocks(self, regression_timeline):
        """CE subscans rebuild as slices of one shared crossing solve.

        All three Deep56 subscans share ``metadata["t0_scan"]``, so each
        rebuilt trajectory must cover only its own block window rather
        than the full pass (which triple-counted the visit at 4.75x the
        scheduled science time). The first subscan legitimately starts up
        to a tick plus the slew allowance before the re-solved crossing,
        so its slice may be shorter than its block.
        """
        from fyst_trajectories.overhead import schedule_to_trajectories

        pairs = schedule_to_trajectories(regression_timeline)
        assert len(pairs) == 3

        starts = []
        total_span = 0.0
        for sblock, scan_block in pairs:
            traj = scan_block.trajectory
            span = float(traj.times[-1] - traj.times[0])
            total_span += span
            starts.append(traj.start_time.unix)
            assert traj.times[0] == 0.0
            assert traj.start_time.unix >= sblock.t_start.unix - 1e-3
            end_unix = traj.start_time.unix + span
            assert end_unix <= sblock.t_stop.unix + 1e-3
            assert span <= sblock.duration + 1e-3

        # Distinct ordered starts; the pre-fix bug returned three
        # identical full passes.
        assert starts == sorted(starts)
        assert len(set(starts)) == 3

        science_time = regression_timeline.total_science_time
        # The only deficit is the first subscan's acquisition lead, at
        # most one scheduler tick (300 s) plus the 180 s slew allowance.
        assert science_time - 500.0 < total_span <= science_time + 1e-3
        # The slice, not the full pass: duration diverges from the
        # computed_params of the solved pass on purpose.
        assert pairs[0][1].duration < pairs[0][1].computed_params["duration"]

    def test_science_blocks_record_their_executed_envelope(self, regression_timeline):
        """A science block's azimuth bounds are what its trajectory sweeps.

        The bounds used to be a scalar estimate of the field width at the
        tick time, ``(83.71, 145.94)`` here, while the crossing pass these
        blocks belong to drifts across ``(102.33, 215.62)``. They are read
        from the block's own rebuilt trajectory now, so the two cannot
        diverge again. The corridor is pinned as well: agreement alone
        would also hold if both numbers were wrong together.
        """
        from fyst_trajectories.overhead import schedule_to_trajectories

        pairs = schedule_to_trajectories(regression_timeline)
        assert len(pairs) == 3

        for sblock, scan_block in pairs:
            az = scan_block.trajectory.az
            assert sblock.az_start == pytest.approx(float(az.min()), abs=1e-9)
            assert sblock.az_end == pytest.approx(float(az.max()), abs=1e-9)
            assert sblock.end_pose_az == pytest.approx(float(az[-1]), abs=1e-9)
            assert sblock.az_start == pytest.approx(102.33, abs=0.01)
            assert sblock.az_end == pytest.approx(215.62, abs=0.01)
