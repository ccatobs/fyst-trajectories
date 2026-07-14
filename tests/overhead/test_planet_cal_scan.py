"""Tests for opt-in source-CES multi-pass planet calibrations.

When ``CalibrationPolicy.planet_cal_scan`` is set, a due ``planet_cal`` is
planned as a multi-pass source-CES sequence anchored at the scheduler
clock (one CALIBRATION block per pass) instead of a single fixed-duration
parked block. These tests cover the emit path, the failure/truncation
semantics, and the ECSV round-trip of the recorded pass parameters.

Jupiter rises over Cerro Chajnantor across roughly 20:00-23:30 UTC on
2026-03-15 (the date the planning tests use), so the calibration anchors
below sit inside that rising arc.
"""

import logging

import numpy as np
import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories import get_fyst_site
from fyst_trajectories.exceptions import TargetNotObservableError
from fyst_trajectories.overhead import (
    BlockType,
    CalibrationPolicy,
    CalibrationState,
    ObservingPatch,
    ObservingTimeline,
    OverheadModel,
    accumulate_hitmaps,
    compute_budget,
    generate_timeline,
    read_timeline,
    schedule_to_trajectories,
    validate_scan_params,
    write_timeline,
)
from fyst_trajectories.overhead.scheduler import (
    CalibrationPhase,
    SchedulerContext,
    SchedulerState,
)
from fyst_trajectories.overhead.simulation import _SOURCE_CES_WINDOW_BUFFER_SEC
from fyst_trajectories.planning import plan_source_ces
from fyst_trajectories.planning.source_ces import _offset_footprint_eta, _resolve_footprint

# Anchor inside Jupiter's rising arc where a 3-pass sequence is feasible.
_ANCHOR = "2026-03-15T20:45:00"
_END = "2026-03-15T23:30:00"

_SCAN_PARAM_KEYS = {
    "body",
    "footprint",
    "el_bore",
    "mode",
    "window",
    "boresight_rot",
    "timestep",
    "eta_offset_deg",
    "pass_index",
    "n_passes",
}


def _scan_policy(**overrides):
    """Policy with only ``planet_cal`` due so the passes are isolated.

    Every non-planet cadence is set well above the test window so, paired
    with a :func:`_only_planet_cal_due` state, ``needs_calibration`` returns
    ``planet_cal`` alone.
    """
    params = dict(
        retune_cadence=1.0e9,
        pointing_cadence=1.0e9,
        focus_cadence=1.0e9,
        skydip_cadence=1.0e9,
        planet_cal_cadence=1.0e9,
        planet_cal_scan=True,
        planet_cal_passes=3,
        planet_targets=("jupiter",),
        planet_min_elevation=15.0,
    )
    params.update(overrides)
    return CalibrationPolicy(**params)


def _only_planet_cal_due(anchor):
    """CalibrationState with every ``last_*`` recent except ``planet_cal``."""
    return CalibrationState(
        last_retune=anchor,
        last_pointing_cal=anchor,
        last_focus=anchor,
        last_skydip=anchor,
        last_planet_cal=None,
    )


def _ctx(policy, *, start=_ANCHOR, end=_END):
    return SchedulerContext.build(
        patches=[],
        site=get_fyst_site(),
        start_time=Time(start, scale="utc"),
        end_time=Time(end, scale="utc"),
        calibration_policy=policy,
    )


# A field observable across the Jupiter-rising window whose science blocks
# reconstruct cleanly, so a flag-on timeline carries both science blocks and
# source-CES planet-cal passes (needed to exercise science + calibration
# reconstruction together in schedule_to_trajectories).
_RECON_PATCH = ObservingPatch(
    name="ReconField",
    ra_center=140.0,
    dec_center=-23.0,
    width=4.0,
    height=4.0,
    scan_type="pong",
    velocity=0.5,
)


@pytest.fixture(scope="module")
def flag_on_timeline():
    """Flag-on timeline carrying both science blocks and three planet-cal passes."""
    return generate_timeline(
        patches=[_RECON_PATCH],
        site=get_fyst_site(),
        start_time=_ANCHOR,
        end_time=_END,
        overhead_model=OverheadModel(),
        calibration_policy=CalibrationPolicy(
            planet_cal_cadence=43200.0,
            planet_cal_scan=True,
            planet_cal_passes=3,
            planet_targets=("jupiter",),
            planet_min_elevation=15.0,
        ),
    )


@pytest.fixture(scope="module")
def flag_on_recon(flag_on_timeline):
    """Reconstruction of the flag-on timeline, computed once for the module.

    Returns ``(site, science_only_pairs, science_plus_cal_pairs)``.
    """
    site = flag_on_timeline.site
    sci_only = schedule_to_trajectories(flag_on_timeline, science_only=True)
    with_cals = schedule_to_trajectories(flag_on_timeline, science_only=False)
    return site, sci_only, with_cals


def _reference_pass(scan_params, site):
    """Independently plan the reference source-CES pass for a recorded block.

    Built from the same recorded parameters as the reconstruction
    (eta-shifted footprint, widened window), so a reconstruction that
    skipped the eta shift would not match it.
    """
    base = _resolve_footprint(scan_params["footprint"])
    fp = _offset_footprint_eta(base, scan_params["eta_offset_deg"])
    t0 = Time(scan_params["window"][0], scale="utc")
    t1 = Time(scan_params["window"][1], scale="utc")
    buf = TimeDelta(_SOURCE_CES_WINDOW_BUFFER_SEC, format="sec")
    return plan_source_ces(
        body=scan_params["body"],
        footprint=fp,
        el_bore=scan_params["el_bore"],
        boresight_rot=scan_params["boresight_rot"],
        timestep=scan_params["timestep"],
        window=(t0 - buf, t1 + buf),
        mode=scan_params["mode"],
        site=site,
    )


def _assert_reconstruction_faithful(block, scan_block, site, *, atol=1e-3):
    """Assert a rebuilt cal pass matches its window and a reference plan.

    Checks the two contracts the reconstruction must honour: the re-solved
    ``t0_iso`` / ``t1_iso`` land on the recorded window (within a coarse
    sampling step), and the trajectory matches an independently-planned
    reference pass built from the same recorded parameters.
    """
    sp = block.metadata["scan_params"]
    w0 = Time(sp["window"][0], scale="utc")
    w1 = Time(sp["window"][1], scale="utc")
    rebuilt_t0 = Time(scan_block.computed_params["t0_iso"], scale="utc")
    rebuilt_t1 = Time(scan_block.computed_params["t1_iso"], scale="utc")
    assert abs((rebuilt_t0 - w0).sec) <= 2.0
    assert abs((rebuilt_t1 - w1).sec) <= 2.0

    ref = _reference_pass(sp, site)
    assert scan_block.trajectory.az.shape == ref.trajectory.az.shape
    assert np.allclose(scan_block.trajectory.az, ref.trajectory.az, atol=atol)
    assert np.allclose(scan_block.trajectory.el, ref.trajectory.el, atol=atol)


class TestPlanetCalScanEmit:
    """The multi-pass source-CES emit path and its recorded parameters."""

    def test_emits_one_block_per_pass_anchored_and_contiguous(self):
        anchor = Time(_ANCHOR, scale="utc")
        policy = _scan_policy(planet_cal_passes=3)
        ctx = _ctx(policy)
        state = SchedulerState(
            current_time=anchor,
            current_az=180.0,
            current_el=50.0,
            cal_state=_only_planet_cal_due(anchor),
            scan_counter=0,
        )

        result = CalibrationPhase().run(state, ctx)
        blocks = result.blocks

        # One CALIBRATION block per requested pass.
        assert len(blocks) == 3
        assert all(b.block_type == BlockType.CALIBRATION for b in blocks)
        assert all(b.scan_type == "planet_cal" for b in blocks)

        # The first block starts exactly at the pre-cal clock; the blocks
        # then tile with no gaps (each t_stop is the next t_start).
        assert blocks[0].t_start.unix == anchor.unix
        for a, b in zip(blocks, blocks[1:]):
            assert abs(a.t_stop.unix - b.t_start.unix) < 1e-6

        # State advances to the last block's stop and marks the cadence.
        assert abs(result.state.current_time.unix - blocks[-1].t_stop.unix) < 1e-6
        assert result.state.cal_state.last_planet_cal is not None
        assert result.state.cal_state.last_planet_cal.unix == anchor.unix

    def test_scan_params_recorded_and_valid(self):
        anchor = Time(_ANCHOR, scale="utc")
        ctx = _ctx(_scan_policy(planet_cal_passes=3))
        state = SchedulerState(
            current_time=anchor,
            current_az=180.0,
            current_el=50.0,
            cal_state=_only_planet_cal_due(anchor),
            scan_counter=0,
        )

        blocks = CalibrationPhase().run(state, ctx).blocks

        for idx, block in enumerate(blocks):
            meta = block.metadata
            assert set(meta) == {"cal_type", "target", "t0_scan", "scan_params"}
            sp = meta["scan_params"]
            # Full replay-grade parameter set, and it validates.
            assert set(sp) == _SCAN_PARAM_KEYS
            validate_scan_params(sp, "source_ces")
            assert sp["body"] == "jupiter"
            assert sp["footprint"] == "c"
            assert sp["mode"] == "rising"
            assert sp["n_passes"] == 3
            assert sp["pass_index"] == idx
            # The recorded window is the pass extent [t0, t1]; its start is
            # the scan start.
            assert len(sp["window"]) == 2
            assert sp["window"][0] == meta["t0_scan"]
            # el_bore matches the block elevation and is the pass value.
            assert sp["el_bore"] == block.elevation
            # The true scan start is at or after the block start (acquisition
            # and inter-pass repointing fold into the block).
            assert Time(meta["t0_scan"]).unix >= block.t_start.unix - 1e-6

    def test_el_bore_steps_monotonically_with_mode(self):
        anchor = Time(_ANCHOR, scale="utc")
        ctx = _ctx(_scan_policy(planet_cal_passes=3))
        state = SchedulerState(
            current_time=anchor,
            current_az=180.0,
            current_el=50.0,
            cal_state=_only_planet_cal_due(anchor),
            scan_counter=0,
        )

        blocks = CalibrationPhase().run(state, ctx).blocks

        # Jupiter is rising, so passes are ordered by increasing el_bore.
        assert all(b.metadata["scan_params"]["mode"] == "rising" for b in blocks)
        el_bores = [b.elevation for b in blocks]
        assert el_bores == sorted(el_bores)
        assert all(a < b for a, b in zip(el_bores, el_bores[1:]))

    def test_setting_planet_steps_el_bore_down(self):
        """A setting anchor produces a descending, contiguous setting sequence.

        Jupiter sets over roughly 00:00-04:00 UTC on the same night, so an
        anchor at 01:30 sits inside the setting arc: the passes step the
        boresight elevation strictly downward and every block observes the
        setting side.
        """
        anchor = Time("2026-03-15T01:30:00", scale="utc")
        ctx = _ctx(
            _scan_policy(planet_cal_passes=3),
            start="2026-03-15T01:30:00",
            end="2026-03-15T03:30:00",
        )
        state = SchedulerState(
            current_time=anchor,
            current_az=180.0,
            current_el=50.0,
            cal_state=_only_planet_cal_due(anchor),
            scan_counter=0,
        )

        blocks = CalibrationPhase().run(state, ctx).blocks

        assert len(blocks) == 3
        # Every pass runs on the setting arc, and the blocks say so.
        assert all(b.metadata["scan_params"]["mode"] == "setting" for b in blocks)
        assert all(b.rising is False for b in blocks)
        # A setting source crosses higher elevations first: strictly
        # decreasing el_bore across the sequence.
        el_bores = [b.elevation for b in blocks]
        assert all(a > b for a, b in zip(el_bores, el_bores[1:]))
        # Contiguous tiling from the anchor, same as the rising path.
        assert blocks[0].t_start.unix == anchor.unix
        for a, b in zip(blocks, blocks[1:]):
            assert abs(a.t_stop.unix - b.t_start.unix) < 1e-6

    def test_total_time_conserved_and_budget_counts_planet_cal(self):
        anchor = Time(_ANCHOR, scale="utc")
        ctx = _ctx(_scan_policy(planet_cal_passes=3))
        state = SchedulerState(
            current_time=anchor,
            current_az=180.0,
            current_el=50.0,
            cal_state=_only_planet_cal_due(anchor),
            scan_counter=0,
        )

        blocks = CalibrationPhase().run(state, ctx).blocks
        timeline = ObservingTimeline(
            blocks=blocks,
            site=ctx.site,
            start_time=anchor,
            end_time=blocks[-1].t_stop,
            overhead_model=ctx.overhead_model,
            calibration_policy=ctx.calibration_policy,
        )

        # Blocks tile [anchor, last t_stop] with no holes.
        span = (blocks[-1].t_stop - anchor).sec
        block_total = sum(b.duration for b in blocks)
        assert abs(block_total - span) < 1e-3
        assert timeline.validate() == []

        breakdown = compute_budget(timeline)["calibration_breakdown"]
        assert breakdown["planet_cal"]["count"] == 3


class TestPlanetCalScanIntegration:
    """The flag drives through the public ``generate_timeline`` entry point."""

    def test_generate_timeline_flag_on(self):
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
            )
        ]
        policy = CalibrationPolicy(
            planet_cal_cadence=43200.0,
            planet_cal_scan=True,
            planet_cal_passes=3,
            planet_targets=("jupiter",),
            planet_min_elevation=15.0,
        )
        timeline = generate_timeline(
            patches=patches,
            site=site,
            start_time=_ANCHOR,
            end_time=_END,
            overhead_model=OverheadModel(),
            calibration_policy=policy,
        )

        planet_blocks = [b for b in timeline.blocks if b.scan_type == "planet_cal"]
        # The opening burst fires one planet cal as a 3-pass sequence.
        assert len(planet_blocks) == 3
        for b in planet_blocks:
            assert "scan_params" in b.metadata
            validate_scan_params(b.metadata["scan_params"], "source_ces")

        # Standard conservation + validation invariants still hold.
        block_total = sum(b.duration for b in timeline.blocks)
        assert block_total <= timeline.total_time + 1.0
        assert timeline.validate() == []
        assert compute_budget(timeline)["calibration_breakdown"]["planet_cal"]["count"] == 3


class TestPlanetCalScanFlagOff:
    """With the flag off the planet cal stays a single parked block."""

    def test_default_policy_emits_single_parked_planet_cal(self):
        anchor = Time(_ANCHOR, scale="utc")
        # planet_cal_scan defaults to False.
        policy = _scan_policy(planet_cal_scan=False)
        ctx = _ctx(policy)
        state = SchedulerState(
            current_time=anchor,
            current_az=175.0,
            current_el=48.0,
            cal_state=_only_planet_cal_due(anchor),
            scan_counter=0,
        )

        blocks = CalibrationPhase().run(state, ctx).blocks

        assert len(blocks) == 1
        block = blocks[0]
        assert block.scan_type == "planet_cal"
        # Parked at the current pose with the legacy 2-key metadata.
        assert block.az_start == block.az_end == 175.0
        assert block.elevation == 48.0
        assert set(block.metadata) == {"cal_type", "target"}
        assert block.metadata["cal_type"] == "planet_cal"
        assert block.metadata["target"] == "jupiter"
        assert abs(block.duration - OverheadModel().planet_cal_duration) < 1e-6


class TestPlanetCalScanDeferOnFailure:
    """An infeasible sequence is skipped and left due, other cals still run."""

    def test_planning_failure_defers_without_marking(self, monkeypatch):
        def _raise(**kwargs):
            raise TargetNotObservableError(
                target="jupiter",
                time_info=_ANCHOR,
                bounds_error=None,
                message="forced infeasible for test",
            )

        monkeypatch.setattr(
            "fyst_trajectories.overhead.scheduler.phases.plan_source_ces_passes",
            _raise,
        )

        anchor = Time(_ANCHOR, scale="utc")
        policy = _scan_policy(skydip_cadence=1.0e9)
        ctx = _ctx(policy)
        # skydip AND planet_cal both due; retune/pointing/focus recent.
        state = SchedulerState(
            current_time=anchor,
            current_az=180.0,
            current_el=50.0,
            cal_state=CalibrationState(
                last_retune=anchor,
                last_pointing_cal=anchor,
                last_focus=anchor,
                last_skydip=None,
                last_planet_cal=None,
            ),
            scan_counter=0,
        )

        result = CalibrationPhase().run(state, ctx)

        # The in-place skydip still emits; no planet_cal block appears.
        scan_types = [b.scan_type for b in result.blocks]
        assert "skydip" in scan_types
        assert "planet_cal" not in scan_types
        # planet_cal was not marked, so it stays due for a later iteration.
        assert result.state.cal_state.last_planet_cal is None
        assert result.state.cal_state.last_skydip is not None


class TestPlanetCalScanTruncation:
    """End-of-night keeps only the passes that finish before the window closes."""

    def test_truncates_to_fitting_prefix(self):
        anchor = Time("2026-03-15T20:30:00", scale="utc")
        # Only the first pass (finishing ~20:37) fits before this end time.
        ctx = _ctx(
            _scan_policy(planet_cal_passes=3),
            start="2026-03-15T20:30:00",
            end="2026-03-15T20:40:00",
        )
        state = SchedulerState(
            current_time=anchor,
            current_az=180.0,
            current_el=50.0,
            cal_state=_only_planet_cal_due(anchor),
            scan_counter=0,
        )

        blocks = CalibrationPhase().run(state, ctx).blocks

        assert len(blocks) == 1
        # The requested total is preserved so truncation is visible.
        assert blocks[0].metadata["scan_params"]["n_passes"] == 3
        assert blocks[0].metadata["scan_params"]["pass_index"] == 0
        assert blocks[0].t_stop.unix <= ctx.end_time.unix


class TestPlanetCalScanECSVRoundTrip:
    """The recorded pass parameters survive a TOAST-ECSV write/read."""

    def test_scan_params_and_t0_scan_round_trip(self, tmp_path):
        anchor = Time(_ANCHOR, scale="utc")
        ctx = _ctx(_scan_policy(planet_cal_passes=3))
        state = SchedulerState(
            current_time=anchor,
            current_az=180.0,
            current_el=50.0,
            cal_state=_only_planet_cal_due(anchor),
            scan_counter=0,
        )
        blocks = CalibrationPhase().run(state, ctx).blocks
        timeline = ObservingTimeline(
            blocks=blocks,
            site=ctx.site,
            start_time=anchor,
            end_time=blocks[-1].t_stop,
            overhead_model=ctx.overhead_model,
            calibration_policy=ctx.calibration_policy,
        )

        path = tmp_path / "planet_cal_scan_rt.ecsv"
        write_timeline(timeline, path)
        loaded = read_timeline(path)

        cal_blocks = [b for b in loaded.blocks if b.block_type == BlockType.CALIBRATION]
        assert len(cal_blocks) == 3
        for original, restored in zip(blocks, cal_blocks):
            assert restored.metadata["t0_scan"] == original.metadata["t0_scan"]
            assert restored.metadata["scan_params"] == original.metadata["scan_params"]
            # Restored params still validate.
            validate_scan_params(restored.metadata["scan_params"], "source_ces")


class TestScheduleToTrajectoriesScienceOnly:
    """The ``science_only`` flag on :func:`schedule_to_trajectories`."""

    def test_flag_on_timeline_default_returns_science_only(self, flag_on_recon, flag_on_timeline):
        """Default ``science_only=True`` returns science pairs and no calibration."""
        _site, sci_only, _with_cals = flag_on_recon

        assert sci_only, "expected the flag-on timeline to carry science blocks"
        assert all(b.block_type == BlockType.SCIENCE for b, _ in sci_only)
        # Every science block reconstructs; calibration passes are excluded.
        assert len(sci_only) == len(flag_on_timeline.science_blocks)
        # The planet-cal passes exist in the timeline but are not returned here.
        assert any(b.scan_type == "planet_cal" for b in flag_on_timeline.blocks)

    def test_default_policy_skips_calibrations_silently(self, caplog):
        """``science_only=False`` skips parked cals/retunes with no logs; science unchanged."""
        site = get_fyst_site()
        patch = ObservingPatch(
            name="Deep56",
            ra_center=180.0,
            dec_center=-30.0,
            width=4.0,
            height=4.0,
            scan_type="pong",
            velocity=0.5,
        )
        # Default policy: parked planet cal + retunes, none carrying scan_params.
        timeline = generate_timeline(
            patches=[patch],
            site=site,
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T06:00:00",
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )
        # There really are calibration blocks with no scan_params, so the
        # science_only=False path exercises the silent-skip branch.
        assert any(
            b.block_type == BlockType.CALIBRATION and "scan_params" not in b.metadata
            for b in timeline.blocks
        )

        sci_true = schedule_to_trajectories(timeline, science_only=True)
        with caplog.at_level(logging.WARNING, logger="fyst_trajectories.overhead.simulation"):
            all_false = schedule_to_trajectories(timeline, science_only=False)

        # The parked/retune/idle blocks were skipped silently: no reconstruction
        # failure was logged for them.
        sim_records = [
            r for r in caplog.records if r.name == "fyst_trajectories.overhead.simulation"
        ]
        assert sim_records == []

        # With no reconstructable calibration blocks, False adds nothing: the
        # science pairs are identical to the science_only=True result.
        assert len(all_false) == len(sci_true)
        assert all(b.block_type == BlockType.SCIENCE for b, _ in all_false)
        for (b_false, sb_false), (b_true, sb_true) in zip(all_false, sci_true):
            assert b_false is b_true
            assert np.array_equal(sb_false.trajectory.az, sb_true.trajectory.az)
            assert np.array_equal(sb_false.trajectory.el, sb_true.trajectory.el)


class TestPlanetCalScanReconstruction:
    """``schedule_to_trajectories(science_only=False)`` rebuilds source-CES passes."""

    def test_reconstructs_science_plus_one_pair_per_pass(self, flag_on_recon, flag_on_timeline):
        site, sci_only, with_cals = flag_on_recon

        cal_pairs = [(b, sb) for b, sb in with_cals if b.block_type == BlockType.CALIBRATION]
        sci_pairs = [(b, sb) for b, sb in with_cals if b.block_type == BlockType.SCIENCE]
        planet_blocks = [b for b in flag_on_timeline.blocks if b.scan_type == "planet_cal"]

        # One reconstructed pair per planet-cal pass, plus the science pairs.
        assert len(planet_blocks) == 3
        assert len(cal_pairs) == len(planet_blocks)
        assert all(b.scan_type == "planet_cal" for b, _ in cal_pairs)

        # The science subset is exactly the science_only=True result.
        assert len(sci_pairs) == len(sci_only)
        for (b_f, sb_f), (b_t, sb_t) in zip(sci_pairs, sci_only):
            assert b_f is b_t
            assert np.array_equal(sb_f.trajectory.az, sb_t.trajectory.az)

        # Each rebuilt pass lands on its recorded window and matches a reference
        # plan built from the same shifted footprint and widened window.
        for block, scan_block in cal_pairs:
            _assert_reconstruction_faithful(block, scan_block, site)

    def test_full_chain_ecsv_roundtrip_feeds_reconstruction(self, flag_on_timeline, tmp_path):
        path = tmp_path / "flag_on_rt.ecsv"
        write_timeline(flag_on_timeline, path)
        loaded = read_timeline(path)

        pairs = schedule_to_trajectories(loaded, science_only=False)
        cal_pairs = [(b, sb) for b, sb in pairs if b.block_type == BlockType.CALIBRATION]
        assert len(cal_pairs) == 3
        # The scan_params survive the ECSV JSON round-trip and still rebuild the
        # same geometry, matched against a reference planned from the loaded
        # parameters.
        for block, scan_block in cal_pairs:
            _assert_reconstruction_faithful(block, scan_block, loaded.site)

    def test_truncated_pass_reconstructs(self):
        """A truncated emission (``n_passes`` > emitted) still rebuilds per block."""
        anchor = Time("2026-03-15T20:30:00", scale="utc")
        ctx = _ctx(
            _scan_policy(planet_cal_passes=3),
            start="2026-03-15T20:30:00",
            end="2026-03-15T20:40:00",
        )
        state = SchedulerState(
            current_time=anchor,
            current_az=180.0,
            current_el=50.0,
            cal_state=_only_planet_cal_due(anchor),
            scan_counter=0,
        )
        blocks = CalibrationPhase().run(state, ctx).blocks
        assert len(blocks) == 1
        assert blocks[0].metadata["scan_params"]["n_passes"] == 3

        timeline = ObservingTimeline(
            blocks=blocks,
            site=ctx.site,
            start_time=anchor,
            end_time=blocks[-1].t_stop,
            overhead_model=ctx.overhead_model,
            calibration_policy=ctx.calibration_policy,
        )
        pairs = schedule_to_trajectories(timeline, science_only=False)
        assert len(pairs) == 1
        block, scan_block = pairs[0]
        assert block.block_type == BlockType.CALIBRATION
        _assert_reconstruction_faithful(block, scan_block, ctx.site)

    def test_setting_pass_reconstructs(self):
        """A setting-direction pass rebuilds onto its recorded window and geometry."""
        anchor = Time("2026-03-15T01:30:00", scale="utc")
        ctx = _ctx(
            _scan_policy(planet_cal_passes=1),
            start="2026-03-15T01:30:00",
            end="2026-03-15T03:30:00",
        )
        state = SchedulerState(
            current_time=anchor,
            current_az=180.0,
            current_el=50.0,
            cal_state=_only_planet_cal_due(anchor),
            scan_counter=0,
        )
        blocks = CalibrationPhase().run(state, ctx).blocks
        assert len(blocks) == 1
        assert blocks[0].metadata["scan_params"]["mode"] == "setting"

        timeline = ObservingTimeline(
            blocks=blocks,
            site=ctx.site,
            start_time=anchor,
            end_time=blocks[-1].t_stop,
            overhead_model=ctx.overhead_model,
            calibration_policy=ctx.calibration_policy,
        )
        pairs = schedule_to_trajectories(timeline, science_only=False)
        assert len(pairs) == 1
        block, scan_block = pairs[0]
        assert block.block_type == BlockType.CALIBRATION
        assert scan_block.computed_params["mode"] == "setting"
        _assert_reconstruction_faithful(block, scan_block, ctx.site)

    def test_hitmap_gains_hits_from_reconstructed_cals(self, flag_on_recon):
        """Feeding science+cal pairs into accumulate_hitmaps adds the cal hits."""
        pytest.importorskip("healpy")
        site, sci_only, with_cals = flag_on_recon

        hm_science = accumulate_hitmaps(sci_only, site, nside=16)
        hm_all = accumulate_hitmaps(with_cals, site, nside=16)
        # The reconstructed planet-cal passes contribute additional samples.
        assert hm_all.sum() > hm_science.sum()
