"""Tests for timeline simulation pipeline."""

import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories import get_fyst_site
from fyst_trajectories.overhead import (
    ObservingPatch,
    TimelineBlock,
    compute_budget,
    generate_timeline,
    validate_scan_params,
)
from fyst_trajectories.overhead.simulation import (
    _generate_trajectory_for_block,
    _slice_to_block_window,
)


@pytest.fixture(scope="module")
def one_night_timeline():
    """Generate a short timeline for testing."""
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
    return generate_timeline(
        patches=patches,
        site=site,
        start_time="2026-06-15T02:00:00",
        end_time="2026-06-15T06:00:00",
    )


class TestComputeBudget:
    """Tests for compute_budget."""

    def test_returns_expected_keys(self, one_night_timeline):
        stats = compute_budget(one_night_timeline)
        assert "total_time" in stats
        assert "science_time" in stats
        assert "calibration_time" in stats
        assert "slew_time" in stats
        assert "idle_time" in stats
        assert "efficiency" in stats
        assert "n_science_scans" in stats
        assert "per_patch" in stats
        assert "calibration_breakdown" in stats

    def test_time_adds_up(self, one_night_timeline):
        stats = compute_budget(one_night_timeline)
        accounted = (
            stats["science_time"]
            + stats["calibration_time"]
            + stats["slew_time"]
            + stats["idle_time"]
        )
        # Accounted time should be <= total time (gaps are possible)
        assert accounted <= stats["total_time"] + 1.0

    def test_per_patch_breakdown(self, one_night_timeline):
        stats = compute_budget(one_night_timeline)
        if one_night_timeline.n_science_scans > 0:
            assert "test_field" in stats["per_patch"]
            patch_stats = stats["per_patch"]["test_field"]
            assert patch_stats["science_time"] > 0
            assert patch_stats["n_scans"] > 0

    def test_calibration_breakdown(self, one_night_timeline):
        stats = compute_budget(one_night_timeline)
        if one_night_timeline.calibration_blocks:
            assert len(stats["calibration_breakdown"]) > 0
            for cal_type, cal_info in stats["calibration_breakdown"].items():
                assert cal_info["count"] > 0
                assert cal_info["total_time"] > 0

    def test_efficiency_matches_timeline(self, one_night_timeline):
        stats = compute_budget(one_night_timeline)
        assert abs(stats["efficiency"] - one_night_timeline.efficiency) < 0.01

    def test_empty_timeline(self):
        from fyst_trajectories.overhead.models import (
            CalibrationPolicy,
            ObservingTimeline,
            OverheadModel,
        )

        site = get_fyst_site()
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        timeline = ObservingTimeline(
            blocks=[],
            site=site,
            start_time=t0,
            end_time=t0,
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )
        stats = compute_budget(timeline)
        assert stats["n_science_scans"] == 0
        assert stats["science_time"] == 0.0


class TestGenerateTrajectoryForBlock:
    """Defensive tests for the simulation bridge."""

    def test_raises_on_missing_metadata(self):
        """Missing geometry keys should raise ValueError loudly."""
        site = get_fyst_site()
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        block = TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(300, format="sec"),
            block_type="science",
            patch_name="no_meta",
            az_start=170.0,
            az_end=190.0,
            elevation=50.0,
            scan_index=0,
            scan_type="pong",
            metadata={},  # intentionally empty
        )
        with pytest.raises(ValueError, match="missing required keys"):
            _generate_trajectory_for_block(block, site)

    def test_raises_lists_missing_keys(self):
        """The error message should list all missing keys."""
        site = get_fyst_site()
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        block = TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(300, format="sec"),
            block_type="science",
            patch_name="partial_meta",
            az_start=170.0,
            az_end=190.0,
            elevation=50.0,
            scan_index=0,
            scan_type="pong",
            metadata={"ra_center": 180.0, "dec_center": -30.0},
        )
        with pytest.raises(ValueError, match="width.*height.*velocity"):
            _generate_trajectory_for_block(block, site)

    def test_succeeds_with_full_metadata(self):
        """With all required keys, generation should succeed."""
        site = get_fyst_site()
        t0 = Time("2026-06-15T05:00:00", scale="utc")
        block = TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(120, format="sec"),
            block_type="science",
            patch_name="full_meta",
            az_start=170.0,
            az_end=190.0,
            elevation=60.0,
            scan_index=0,
            scan_type="pong",
            metadata={
                "ra_center": 200.0,
                "dec_center": -25.0,
                "width": 3.0,
                "height": 2.0,
                "velocity": 0.5,
                "scan_params": {"spacing": 0.1, "num_terms": 4},
            },
        )
        sb = _generate_trajectory_for_block(block, site)
        assert sb.config.width == pytest.approx(3.0)
        assert sb.config.height == pytest.approx(2.0)

    def test_raises_on_bad_scan_params_key(self):
        """A typo in ``scan_params`` should raise KeyError, not fall through."""
        site = get_fyst_site()
        t0 = Time("2026-06-15T05:00:00", scale="utc")
        block = TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(120, format="sec"),
            block_type="science",
            patch_name="typo_meta",
            az_start=170.0,
            az_end=190.0,
            elevation=60.0,
            scan_index=0,
            scan_type="daisy",
            metadata={
                "ra_center": 200.0,
                "dec_center": -25.0,
                "width": 3.0,
                "height": 2.0,
                "velocity": 0.5,
                # "radiu" is a typo for "radius", belongs to DaisyScanParams.
                "scan_params": {"radiu": 1.0},
            },
        )
        with pytest.raises(KeyError, match="radiu"):
            _generate_trajectory_for_block(block, site)


class TestValidateScanParams:
    """Unit tests for the ``scan_params`` consumer-side validator."""

    def test_accepts_empty_dict(self):
        # Every scan-params TypedDict is total=False, so {} is always valid.
        validate_scan_params({}, "constant_el")
        validate_scan_params({}, "pong")
        validate_scan_params({}, "daisy")

    def test_accepts_known_keys_per_scan_type(self):
        validate_scan_params({"az_padding": 1.0, "az_accel": 0.5}, "constant_el")
        validate_scan_params({"spacing": 0.1, "num_terms": 4}, "pong")
        validate_scan_params({"radius": 1.0, "turn_radius": 0.5}, "daisy")

    def test_rejects_unknown_key(self):
        with pytest.raises(KeyError, match="unknown keys"):
            validate_scan_params({"radius": 1.0}, "constant_el")  # Daisy key on CE
        with pytest.raises(KeyError, match="spacing"):
            validate_scan_params({"spacing": 0.1}, "daisy")  # Pong key on Daisy

    def test_rejects_unknown_scan_type(self):
        with pytest.raises(KeyError, match="Unknown scan_type"):
            validate_scan_params({}, "sidereal")

    def test_rising_key_allowed_only_for_constant_el(self):
        # "rising" is a CEScanParams key: valid on constant_el, rejected
        # on pong and daisy (which have no such key).
        validate_scan_params({"rising": True}, "constant_el")
        with pytest.raises(KeyError, match="unknown keys"):
            validate_scan_params({"rising": True}, "pong")
        with pytest.raises(KeyError, match="unknown keys"):
            validate_scan_params({"rising": True}, "daisy")


class TestSliceToBlockWindow:
    """Rebuilt science trajectories cover only their own block window."""

    @staticmethod
    def _pong_block(t0, window_sec):
        return TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(window_sec, format="sec"),
            block_type="science",
            patch_name="slice_me",
            az_start=170.0,
            az_end=190.0,
            elevation=60.0,
            scan_index=0,
            scan_type="pong",
            metadata={
                "ra_center": 200.0,
                "dec_center": -25.0,
                "width": 3.0,
                "height": 2.0,
                "velocity": 0.5,
                "scan_params": {"spacing": 0.1, "num_terms": 4},
            },
        )

    def test_pong_rebuild_sliced_to_block_window(self):
        """A pong rebuild longer than its block is cut to [t_start, t_stop)."""
        site = get_fyst_site()
        t0 = Time("2026-06-15T05:00:00", scale="utc")
        block = self._pong_block(t0, 120.0)

        sb = _generate_trajectory_for_block(block, site)
        traj = sb.trajectory

        # A 3x2 deg pong at 0.5 deg/s runs far longer than 120 s per
        # period, so the slice must engage.
        assert traj.times[0] == 0.0
        assert traj.times[-1] < 120.0
        assert traj.start_time.unix >= t0.unix - 1e-6
        end = traj.start_time + TimeDelta(float(traj.times[-1]), format="sec")
        assert end.unix < block.t_stop.unix
        assert sb.duration == pytest.approx(float(traj.times[-1]))

    def test_disjoint_window_raises(self):
        """A block window past the re-solved scan raises (caller logs + skips)."""
        site = get_fyst_site()
        t0 = Time("2026-06-15T05:00:00", scale="utc")
        sb = _generate_trajectory_for_block(self._pong_block(t0, 120.0), site)

        late = self._pong_block(t0 + TimeDelta(1e6, format="sec"), 120.0)
        with pytest.raises(ValueError, match="no longer overlaps"):
            _slice_to_block_window(sb, late)

    def test_daisy_rebuild_already_inside_window_unchanged(self):
        """The daisy branch plans with the block duration; no slicing occurs."""
        site = get_fyst_site()
        t0 = Time("2026-06-15T05:00:00", scale="utc")
        block = TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(120.0, format="sec"),
            block_type="science",
            patch_name="daisy_field",
            az_start=170.0,
            az_end=190.0,
            elevation=60.0,
            scan_index=0,
            scan_type="daisy",
            metadata={
                "ra_center": 200.0,
                "dec_center": -25.0,
                "width": 3.0,
                "height": 2.0,
                "velocity": 0.5,
                "scan_params": {"radius": 1.0, "turn_radius": 0.5},
            },
        )

        sb = _generate_trajectory_for_block(block, site)

        # The daisy plan spans duration - timestep, so it fits unsliced.
        assert sb.trajectory.times[0] == 0.0
        assert sb.trajectory.times[-1] == pytest.approx(120.0 - 0.1)
        assert sb.trajectory.start_time.unix == pytest.approx(t0.unix)
