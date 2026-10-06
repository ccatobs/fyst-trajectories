"""Tests for the timeline generator."""

import numpy as np
import pytest

from fyst_trajectories import Coordinates, get_fyst_site
from fyst_trajectories.overhead import (
    BlockType,
    CalibrationPolicy,
    ElevationConstraint,
    ObservingPatch,
    OverheadModel,
    SunAvoidanceConstraint,
    generate_timeline,
    schedule_to_trajectories,
)
from fyst_trajectories.patterns import PongScanConfig, compute_pong_period


def _pong_period(patch):
    """One period of ``patch``'s pattern at the rebuild's default spacing."""
    config = PongScanConfig(
        timestep=0.1,
        width=patch.width,
        height=patch.height,
        spacing=0.1,
        velocity=patch.velocity,
        num_terms=4,
        angle=0.0,
    )
    return compute_pong_period(config)[0]


def assert_boundary_retunes_precede_science(timeline):
    """Assert every retune after the startup burst opens a science subscan.

    At the default ``retune_cadence=0.0`` a retune is booked only together
    with the science subscan it precedes, so none may be left dangling.
    """
    blocks = timeline.blocks
    for i, block in enumerate(blocks):
        if block.block_type != BlockType.CALIBRATION or block.scan_type != "retune":
            continue
        if block.t_start.unix == timeline.start_time.unix:
            continue  # the startup burst's retune
        following = blocks[i + 1]
        assert following.block_type == BlockType.SCIENCE, f"dangling retune at {block.t_start}"
        assert following.t_start.unix == pytest.approx(block.t_stop.unix, abs=1e-6)


def assert_timeline_valid(timeline, site):
    """Assert a generated timeline is sound.

    Delegates the block-level invariants (overlap, window containment,
    azimuth ordering, pose continuity) to ``ObservingTimeline.validate``
    so every generated-timeline test inherits them, then adds the
    elevation-limit and efficiency checks validate() does not cover.
    """
    assert timeline.validate() == []

    # 0.1 deg slack on the elevation-limit comparisons below.
    for b in timeline.blocks:
        if b.block_type == "science":
            assert b.elevation >= site.telescope_limits.elevation.min - 0.1
            assert b.elevation <= site.telescope_limits.elevation.max + 0.1

    if timeline.n_science_scans > 0:
        assert timeline.efficiency > 0.0
        assert timeline.efficiency <= 1.0


class TestGenerateTimeline:
    """A generated night validates: priority order, cal blocks, scan caps, sun, horizon."""

    def test_single_patch_one_night(self):
        site = get_fyst_site()
        patches = [
            ObservingPatch(
                name="test_field",
                ra_center=180.0,
                dec_center=-30.0,
                width=4.0,
                height=4.0,
                scan_type="pong",
                velocity=0.5,
            ),
        ]
        timeline = generate_timeline(
            patches=patches,
            site=site,
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T10:00:00",
        )
        assert_timeline_valid(timeline, site)
        assert timeline.n_science_scans > 0
        assert len(timeline.calibration_blocks) > 0

    def test_no_patches(self):
        site = get_fyst_site()
        timeline = generate_timeline(
            patches=[],
            site=site,
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T04:00:00",
        )
        assert timeline.n_science_scans == 0
        idle_blocks = [b for b in timeline.blocks if b.block_type == "idle"]
        assert len(idle_blocks) > 0

    def test_multiple_patches_priority(self):
        site = get_fyst_site()
        patches = [
            ObservingPatch(
                name="high_priority",
                ra_center=180.0,
                dec_center=-30.0,
                width=4.0,
                height=4.0,
                scan_type="pong",
                velocity=0.5,
                priority=1.0,
            ),
            ObservingPatch(
                name="low_priority",
                ra_center=200.0,
                dec_center=-30.0,
                width=4.0,
                height=4.0,
                scan_type="pong",
                velocity=0.5,
                priority=10.0,
            ),
        ]
        timeline = generate_timeline(
            patches=patches,
            site=site,
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T10:00:00",
        )
        assert_timeline_valid(timeline, site)

        high_time = sum(
            b.duration for b in timeline.science_blocks if b.patch_name == "high_priority"
        )
        low_time = sum(
            b.duration for b in timeline.science_blocks if b.patch_name == "low_priority"
        )
        assert high_time >= low_time

    def test_calibration_blocks_present(self):
        site = get_fyst_site()
        patches = [
            ObservingPatch(
                name="test",
                ra_center=180.0,
                dec_center=-30.0,
                width=4.0,
                height=4.0,
                scan_type="pong",
                velocity=0.5,
            ),
        ]
        timeline = generate_timeline(
            patches=patches,
            site=site,
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T10:00:00",
            calibration_policy=CalibrationPolicy(retune_cadence=0.0),
        )
        cal_types = {b.scan_type for b in timeline.calibration_blocks}
        assert "retune" in cal_types

    def test_max_scan_duration_splits(self):
        site = get_fyst_site()
        patches = [
            ObservingPatch(
                name="test",
                ra_center=180.0,
                dec_center=-30.0,
                width=4.0,
                height=4.0,
                scan_type="pong",
                velocity=0.5,
            ),
        ]
        overhead = OverheadModel(max_scan_duration=1800.0)  # 30 min max
        timeline = generate_timeline(
            patches=patches,
            site=site,
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T10:00:00",
            overhead_model=overhead,
        )
        assert_timeline_valid(timeline, site)
        assert_boundary_retunes_precede_science(timeline)
        assert timeline.science_blocks, "expected the timeline to carry science blocks"
        period = _pong_period(patches[0])
        for b in timeline.science_blocks:
            # +1.0 s: a whole-second cushion (seconds) on the max-scan-duration cap.
            assert b.duration <= overhead.max_scan_duration + 1.0
            # Filled with whole pattern periods, recorded for the rebuild.
            n_cycles = b.metadata["scan_params"]["n_cycles"]
            assert n_cycles >= 1
            assert b.duration == pytest.approx(n_cycles * period, abs=1e-6)

    def test_sun_avoidance_respected(self):
        """The Sun constraint keeps a Sun-side field out of a daytime schedule."""
        site = get_fyst_site()
        # Afternoon at FYST, the Sun at el ~51 falling to ~11: near_sun sits
        # ~23 deg from it (inside the 45 deg zone) and outranks far_from_sun
        # (~88 deg away), so only the constraint keeps near_sun unscheduled.
        patches = [
            ObservingPatch(
                name="near_sun",
                ra_center=255.0,
                dec_center=-22.0,
                width=4.0,
                height=4.0,
                scan_type="pong",
                velocity=0.5,
                priority=1.0,
            ),
            ObservingPatch(
                name="far_from_sun",
                ra_center=330.0,
                dec_center=-30.0,
                width=4.0,
                height=4.0,
                scan_type="pong",
                velocity=0.5,
                priority=10.0,
            ),
        ]
        constraints = [
            ElevationConstraint(el_min=30.0, el_max=80.0),
            SunAvoidanceConstraint(min_angle=45.0),
        ]
        timeline = generate_timeline(
            patches=patches,
            site=site,
            start_time="2026-11-15T19:00:00",
            end_time="2026-11-15T22:00:00",
            constraints=constraints,
        )
        assert_timeline_valid(timeline, site)
        # Selection never offers near_sun, so no tick idles on a refused slew.
        assert {b.patch_name for b in timeline.science_blocks} == {"far_from_sun"}
        assert not any(b.metadata.get("reason") for b in timeline.blocks)

        coords = Coordinates(site)
        for b in timeline.science_blocks:
            mid_time = b.t_start + (b.t_stop - b.t_start) / 2
            az_mid = (b.az_start + b.az_end) / 2.0
            assert coords.is_sun_safe(az_mid, b.elevation, mid_time)

    def test_short_timeline(self):
        site = get_fyst_site()
        patches = [
            ObservingPatch(
                name="test",
                ra_center=180.0,
                dec_center=-30.0,
                width=4.0,
                height=4.0,
                scan_type="pong",
                velocity=0.5,
            ),
        ]
        timeline = generate_timeline(
            patches=patches,
            site=site,
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T02:10:00",
        )
        assert_timeline_valid(timeline, site)

    def test_pong_scan_clipped_to_observability(self):
        """Pong scans should not extend past the source setting time."""
        site = get_fyst_site()
        coords = Coordinates(site)

        # RA=60, Dec=-30 is descending through this window and crosses the
        # 20 deg floor at ~20:13, so the last scan must be clipped.
        patches = [
            ObservingPatch(
                name="setting_source",
                ra_center=60.0,
                dec_center=-30.0,
                width=4.0,
                height=4.0,
                scan_type="pong",
                velocity=0.5,
            ),
        ]
        timeline = generate_timeline(
            patches=patches,
            site=site,
            start_time="2026-06-15T17:00:00",
            end_time="2026-06-15T22:00:00",
            overhead_model=OverheadModel(max_scan_duration=3600.0),
        )
        assert_timeline_valid(timeline, site)

        el_min = site.telescope_limits.elevation.min
        assert timeline.science_blocks, "expected the timeline to carry science blocks"
        for b in timeline.science_blocks:
            # Verify the source is above el_min at both start and end of scan.
            _, el_start = coords.radec_to_altaz(np.array([60.0]), np.array([-30.0]), b.t_start)
            _, el_end = coords.radec_to_altaz(np.array([60.0]), np.array([-30.0]), b.t_stop)
            # The duration clip lands on the el_min crossing to well under
            # 0.01 deg (the last scan ends at 19.9997 deg on this night).
            assert float(el_start[0]) >= el_min - 0.01, (
                f"Source below el_min at scan start: {float(el_start[0]):.1f} deg"
            )
            assert float(el_end[0]) >= el_min - 0.01, (
                f"Source below el_min at scan end: {float(el_end[0]):.1f} deg"
            )


class TestTimeStep:
    """``generate_timeline`` refuses a time step that cannot advance the clock."""

    @pytest.mark.parametrize("time_step", [0.0, -1.0, float("nan")])
    def test_a_step_that_is_not_positive_is_refused(self, time_step):
        with pytest.raises(ValueError, match="time_step must be positive"):
            generate_timeline(
                patches=[],
                site=get_fyst_site(),
                start_time="2026-06-15T02:00:00",
                end_time="2026-06-15T02:30:00",
                time_step=time_step,
            )


class TestPongPeriodRefusal:
    """``generate_timeline`` refuses a pong patch whose period no subscan can hold."""

    def test_a_period_no_subscan_can_hold_is_refused(self):
        """A 20 x 10 deg pong at 0.5 deg/s takes 8179.2 s per period."""
        patch = ObservingPatch(
            name="Wide01",
            ra_center=180.0,
            dec_center=-30.0,
            width=20.0,
            height=10.0,
            scan_type="pong",
            velocity=0.5,
        )
        with pytest.raises(ValueError, match=r"3300\.0 s.*Wide01 \(8179\.2 s\)"):
            generate_timeline(
                patches=[patch],
                site=get_fyst_site(),
                start_time="2026-06-15T02:00:00",
                end_time="2026-06-15T02:30:00",
            )

    @pytest.mark.parametrize("retune_cadence", [0.0, 3600.0])
    def test_the_boundary_retune_counts_only_at_cadence_zero(self, retune_cadence):
        """An 8 x 8 deg pong (2644.8 s) fits a 2800 s subscan unless a retune precedes it."""
        patch = ObservingPatch(
            name="Mid",
            ra_center=180.0,
            dec_center=-30.0,
            width=8.0,
            height=8.0,
            scan_type="pong",
            velocity=0.5,
        )
        kwargs = dict(
            patches=[patch],
            site=get_fyst_site(),
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T02:30:00",
            overhead_model=OverheadModel(max_scan_duration=2800.0),
            calibration_policy=CalibrationPolicy(retune_cadence=retune_cadence),
        )
        if retune_cadence == 0.0:
            with pytest.raises(ValueError, match=r"2500\.0 s.*Mid \(2644\.8 s\)"):
                generate_timeline(**kwargs)
        else:
            assert generate_timeline(**kwargs).validate() == []


class TestQuickstartNight:
    """The simulator quickstart's night rebuilds every science block it books."""

    def test_every_science_block_rebuilds_to_its_own_length(self):
        site = get_fyst_site()
        wide = ObservingPatch(
            name="Wide01",
            ra_center=180.0,
            dec_center=-30.0,
            width=4.0,
            height=4.0,
            scan_type="pong",
            velocity=0.5,
        )
        deep = ObservingPatch(
            name="Deep56",
            ra_center=24.0,
            dec_center=-32.0,
            width=40.0,
            height=10.0,
            scan_type="constant_el",
            velocity=1.0,
            elevation=50.0,
        )
        timeline = generate_timeline(
            patches=[deep, wide],
            site=site,
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T10:00:00",
        )
        assert_timeline_valid(timeline, site)
        assert_boundary_retunes_precede_science(timeline)

        pairs = schedule_to_trajectories(timeline)
        assert len(pairs) == len(timeline.science_blocks)
        pongs = [(block, scan) for block, scan in pairs if block.scan_type == "pong"]
        assert pongs, "expected the night to scan Wide01"
        for block, scan in pongs:
            n_cycles = block.metadata["scan_params"]["n_cycles"]
            assert n_cycles >= 1
            assert block.duration == pytest.approx(n_cycles * _pong_period(wide), abs=1e-6)
            # The rebuild runs the whole block, to within one 0.1 s timestep.
            assert abs(float(scan.trajectory.times[-1]) - block.duration) <= 0.1
