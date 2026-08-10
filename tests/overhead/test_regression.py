"""Regression tests for timeline generation.

Expected values were computed once from a known-good run and are
hardcoded here as regression anchors.

Any change to these values means the timeline generation algorithm
changed, which requires explicit acknowledgment and updating the
anchors.

Re-anchored for the CE crossing-corridor gate: the scheduler now emits
constant-elevation science only while the pass is plannable and imminent
(see tests/overhead/test_ce_corridor.py). On this fixture night Deep56's
el=50 rising pass opens ~07:50 UTC, so the honest schedule idles until
then and books ~2 h of science instead of the old ~6.6 h (most of which
pointed at empty sky before the pass and could not be reconstructed as
scheduled).

Also re-anchored for scan-coupled retunes: a cadence-0 retune now fires
only at scan boundaries (once at startup, then immediately before every
science subscan), never on idle ticks, so the idle wait no longer books
a retune every 300 s.
"""

import pytest

from fyst_trajectories import get_fyst_site
from fyst_trajectories.overhead import (
    CalibrationPolicy,
    ObservingPatch,
    OverheadModel,
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
        assert len(regression_timeline.blocks) == 81

    def test_science_scan_count(self, regression_timeline):
        """Number of science scans should be stable."""
        assert regression_timeline.n_science_scans == 3

    def test_calibration_block_count(self, regression_timeline):
        """Number of calibration blocks should be stable."""
        assert len(regression_timeline.calibration_blocks) == 16

    def test_science_time(self, regression_timeline):
        """Total science time should match within 1 second."""
        assert abs(regression_timeline.total_science_time - 7271.3) < 1.0

    def test_calibration_time(self, regression_timeline):
        """Total calibration time should match exactly (deterministic)."""
        assert abs(regression_timeline.total_calibration_time - 3200.0) < 0.1

    def test_efficiency(self, regression_timeline):
        """Science efficiency should match within 0.1%."""
        assert abs(regression_timeline.efficiency - 0.2524) < 0.001

    def test_block_type_distribution(self, regression_timeline):
        """Block type counts should match expected distribution."""
        from collections import Counter

        type_counts = Counter(b.block_type for b in regression_timeline.blocks)
        assert type_counts["calibration"] == 16
        assert type_counts["slew"] == 1
        assert type_counts["science"] == 3
        assert type_counts.get("idle", 0) == 61

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
        assert abs(cal["retune"]["total_time"] - 20.0) < 0.1

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

        Deep56's el=50 rising pass opens ~07:50 UTC; until then nothing on
        this fixture night is observable, and the corridor-gated scheduler
        reports that as idle instead of booking dead-air science.
        """
        idle_time = sum(b.duration for b in regression_timeline.blocks if b.block_type == "idle")
        assert abs(idle_time - 18300.0) < 1.0

    def test_timeline_validates_clean(self, regression_timeline):
        """Timeline should pass internal validation with no warnings."""
        warnings = regression_timeline.validate()
        assert warnings == []

    def test_compute_budget_keys(self, regression_timeline):
        """compute_budget() should return all expected keys."""
        stats = compute_budget(regression_timeline)
        expected_keys = {
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
        assert set(stats.keys()) == expected_keys

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

        The scheduler performs all due calibrations at startup:
        retune, pointing_cal, focus, skydip, planet_cal.
        """
        initial_cals = []
        for b in regression_timeline.blocks:
            if b.block_type != "calibration":
                break
            initial_cals.append(str(b.scan_type))

        # Planet cal may be absent if no planet is visible at the test time.
        expected_base = ["retune", "pointing_cal", "focus", "skydip"]
        assert initial_cals[:4] == expected_base
        if len(initial_cals) > 4:
            assert initial_cals[4] == "planet_cal"
