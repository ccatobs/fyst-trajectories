"""Tests for timeline simulation pipeline."""

import inspect
import json

import numpy as np
import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories import get_fyst_site
from fyst_trajectories.overhead import (
    BlockNotReconstructableError,
    ObservingPatch,
    ScanParamsSchemaError,
    SourceCESScanParams,
    TimelineBlock,
    compute_budget,
    generate_timeline,
    validate_scan_params,
)
from fyst_trajectories.overhead import simulation as sim
from fyst_trajectories.overhead.simulation import (
    _generate_trajectory_for_block,
    _slice_to_block_window,
)
from fyst_trajectories.overhead.utils import _search_start_record, _search_start_time
from fyst_trajectories.planning import plan_source_ces
from fyst_trajectories.planning.footprints import resolve_footprint


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
    """The budget's totals and its per-patch and per-cal breakdowns."""

    def test_time_adds_up(self, one_night_timeline):
        stats = compute_budget(one_night_timeline)
        accounted = (
            stats["science_time"]
            + stats["calibration_time"]
            + stats["slew_time"]
            + stats["idle_time"]
        )
        # The schedule tiles its window, tail included, so no time is unaccounted.
        assert accounted == pytest.approx(stats["total_time"], abs=0.01)

    def test_per_patch_breakdown(self, one_night_timeline):
        stats = compute_budget(one_night_timeline)
        assert one_night_timeline.n_science_scans > 0
        assert "test_field" in stats["per_patch"]
        patch_stats = stats["per_patch"]["test_field"]
        assert patch_stats["science_time"] > 0
        assert patch_stats["n_scans"] > 0

    def test_calibration_breakdown(self, one_night_timeline):
        stats = compute_budget(one_night_timeline)
        assert one_night_timeline.calibration_blocks
        assert len(stats["calibration_breakdown"]) > 0
        for cal_type, cal_info in stats["calibration_breakdown"].items():
            assert cal_info["count"] > 0
            assert cal_info["total_time"] > 0

    def test_efficiency_matches_timeline(self, one_night_timeline):
        stats = compute_budget(one_night_timeline)
        assert stats["efficiency"] == one_night_timeline.efficiency

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
    """Rebuilding a block: full metadata succeeds, missing geometry and typos refuse."""

    def test_raises_on_missing_metadata(self):
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
        with pytest.raises(ScanParamsSchemaError, match="missing required keys"):
            _generate_trajectory_for_block(block, site)

    def test_raises_lists_missing_keys(self):
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

    @pytest.mark.filterwarnings(
        "ignore:Trajectory elevation acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_succeeds_with_full_metadata(self):
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
        """A typo in ``scan_params`` raises ScanParamsSchemaError instead of falling through."""
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
        with pytest.raises(ScanParamsSchemaError, match="radiu"):
            _generate_trajectory_for_block(block, site)


class TestValidateScanParams:
    """An empty dict always passes; unknown keys and scan types are refused."""

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

    @pytest.mark.filterwarnings(
        "ignore:Trajectory elevation acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
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

    @pytest.mark.filterwarnings(
        "ignore:Trajectory elevation acceleration:"
        "fyst_trajectories.exceptions.AccelerationLimitWarning",
    )
    def test_disjoint_window_raises(self):
        """A block window past the re-solved scan raises (caller logs + skips)."""
        site = get_fyst_site()
        t0 = Time("2026-06-15T05:00:00", scale="utc")
        sb = _generate_trajectory_for_block(self._pong_block(t0, 120.0), site)

        late = self._pong_block(t0 + TimeDelta(1e6, format="sec"), 120.0)
        with pytest.raises(BlockNotReconstructableError, match="no longer overlaps"):
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


class TestSourceCESRebuildForwardsRecordedKeys:
    """Every recorded source-CES key reaches the rebuild, by kwarg or by transform.

    The expected set is derived from ``SourceCESScanParams`` itself, so a key
    added to the schema without a matching forward in
    ``_generate_source_ces_trajectory`` fails here instead of being dropped
    silently on rebuild.
    """

    # Sequence bookkeeping only. No geometry, so the rebuild never reads them.
    PROVENANCE_ONLY = frozenset({"pass_index", "n_passes"})
    # Consumed through a transform rather than forwarded verbatim: the base
    # module tag is resolved, inflated by footprint_margin and shifted by
    # eta_offset_deg, and, on a block without ``search_start`` (as here), the
    # recorded window is widened by the re-solve buffer.
    TRANSFORMED = frozenset({"footprint", "footprint_margin", "eta_offset_deg", "window"})

    @staticmethod
    def _full_params() -> dict:
        return {
            "body": "jupiter",
            "footprint": "c",
            "el_bore": 35.0,
            "mode": "rising",
            "window": ["2026-03-15T22:00:00.000", "2026-03-15T22:10:00.000"],
            "boresight_rot": 0.0,
            "timestep": 0.1,
            "eta_offset_deg": 0.4,
            "az_accel": 1.5,
            "az_padding": 0.25,
            "v_az": 0.012,
            "az_speed": 1.5,
            "az_throw": 2.44,
            "dwell": 300.0,
            "footprint_margin": 0.4,
            "pass_index": 1,
            "n_passes": 3,
        }

    @staticmethod
    def _spy_kernel(monkeypatch) -> tuple[dict, object]:
        seen: dict = {}
        sentinel = object()

        def fake_plan_source_ces(**kwargs):
            seen.update(kwargs)
            return sentinel

        monkeypatch.setattr(sim, "plan_source_ces", fake_plan_source_ces)
        return seen, sentinel

    def test_fixture_covers_the_whole_schema(self):
        """Extend ``_full_params`` whenever ``SourceCESScanParams`` gains a key."""
        assert set(self._full_params()) == set(SourceCESScanParams.__optional_keys__)

    def test_schema_partitions_into_forwarded_transformed_and_provenance(self, monkeypatch, site):
        params = self._full_params()
        seen, sentinel = self._spy_kernel(monkeypatch)
        meta = {"cal_type": "planet_cal", "target": "jupiter", "scan_params": params}

        assert sim._generate_source_ces_trajectory(meta, site) is sentinel

        kwarg_keys = set(seen) - {"site"}
        consumed = kwarg_keys | self.TRANSFORMED
        expected = set(SourceCESScanParams.__optional_keys__) - self.PROVENANCE_ONLY
        assert consumed == expected, (
            f"rebuild consumes {sorted(consumed)} but the schema minus provenance is "
            f"{sorted(expected)}; forward the missing key or classify it explicitly"
        )
        assert not (kwarg_keys & self.PROVENANCE_ONLY)

        # Verbatim keys arrive unchanged (the kernel overrides included).
        for key in kwarg_keys - self.TRANSFORMED:
            assert seen[key] == params[key], key

        # The transforms: the inflated then shifted footprint and the widened window.
        base = resolve_footprint("c")
        fp = seen["footprint"]
        assert fp.center_eta_deg == pytest.approx(base.center_eta_deg + 0.4)
        assert fp.center_xi_deg == pytest.approx(base.center_xi_deg)
        radius = np.hypot(fp.cover_xi_deg - fp.center_xi_deg, fp.cover_eta_deg - fp.center_eta_deg)
        base_radius = np.hypot(
            base.cover_xi_deg - base.center_xi_deg, base.cover_eta_deg - base.center_eta_deg
        )
        np.testing.assert_allclose(radius, base_radius + 0.4)
        w0, w1 = seen["window"]
        assert (Time(params["window"][0], scale="utc") - w0).to_value("s") == pytest.approx(
            sim._SOURCE_CES_WINDOW_BUFFER_SEC
        )
        assert (w1 - Time(params["window"][1], scale="utc")).to_value("s") == pytest.approx(
            sim._SOURCE_CES_WINDOW_BUFFER_SEC
        )

    def test_forwarded_override_keys_are_kernel_parameters(self):
        """A spy accepts any kwarg; the real kernel must accept each forwarded key."""
        accepted = set(inspect.signature(plan_source_ces).parameters)
        assert set(sim._SOURCE_CES_OVERRIDE_KEYS) <= accepted

    def test_absent_overrides_leave_the_kernel_defaults_in_force(self, monkeypatch, site):
        """Passes recorded without overrides rebuild on the kernel's own defaults."""
        params = {
            key: value
            for key, value in self._full_params().items()
            if key not in sim._SOURCE_CES_OVERRIDE_KEYS
        }
        seen, _ = self._spy_kernel(monkeypatch)
        meta = {"cal_type": "planet_cal", "target": "jupiter", "scan_params": params}

        sim._generate_source_ces_trajectory(meta, site)

        assert not (set(seen) & set(sim._SOURCE_CES_OVERRIDE_KEYS))

    def test_a_missing_required_key_is_a_schema_error(self, site):
        """A recorded pass without a key the rebuild reads is refused by type."""
        params = self._full_params()
        del params["eta_offset_deg"]
        meta = {"cal_type": "planet_cal", "target": "jupiter", "scan_params": params}

        with pytest.raises(ScanParamsSchemaError, match="eta_offset_deg"):
            sim._generate_source_ces_trajectory(meta, site)


class TestSourceCESRebuildRepeatsThePlannersSearch:
    """A pass block that records ``search_start`` is rebuilt from the planner's own search.

    The planner solves every pass of a visit in the 24 h window that opens at
    the visit's anchor, and the block records that anchor beside its dict. The
    rebuild hands it to the source-CES planner's anchored form with the
    recorded ``el_bore`` and ``mode``; a block without it keeps the recorded
    pass widened by the re-solve buffer.
    """

    # Not on a millisecond, so a record rounded anywhere would show.
    START = Time("2026-03-15T21:41:07.123456", scale="utc") + TimeDelta(3.3e-8, format="sec")

    @staticmethod
    def _meta(params, **extra):
        return {"cal_type": "planet_cal", "target": "jupiter", "scan_params": params, **extra}

    def test_the_recorded_start_reaches_the_planner_exactly(self, monkeypatch, site):
        """The anchored form gets the instant itself, and the recorded window is not used."""
        params = TestSourceCESRebuildForwardsRecordedKeys._full_params()
        seen, sentinel = TestSourceCESRebuildForwardsRecordedKeys._spy_kernel(monkeypatch)
        meta = self._meta(params, search_start=_search_start_record(self.START))

        assert sim._generate_source_ces_trajectory(meta, site) is sentinel

        assert "window" not in seen
        start = seen["start_time"]
        assert (start.jd1, start.jd2) == (self.START.jd1, self.START.jd2)
        assert (seen["el_bore"], seen["mode"]) == (params["el_bore"], params["mode"])

    def test_the_planner_searches_the_window_it_planned_the_visit_in(self, monkeypatch, site):
        """The rebuild's search is the one ``plan_source_ces_passes`` gave each pass.

        Both calls stop at the kernel, which records the window it was
        handed; the two windows agree to the bit at both ends.
        """
        from fyst_trajectories.planning import plan_source_ces_passes
        from fyst_trajectories.planning import source_ces as kernel_module

        windows = []

        class _Stop(Exception):
            pass

        def record(**kwargs):
            windows.append(kwargs["window"])
            raise _Stop

        monkeypatch.setattr(kernel_module, "_compute_source_ces_core", record)
        params = TestSourceCESRebuildForwardsRecordedKeys._full_params()
        with pytest.raises(_Stop):
            plan_source_ces_passes(
                body=params["body"],
                footprint="c",
                el_bore=params["el_bore"],
                mode=params["mode"],
                start_time=self.START,
                n_passes=1,
                site=site,
            )
        meta = self._meta(params, search_start=_search_start_record(self.START))
        with pytest.raises(_Stop):
            sim._generate_source_ces_trajectory(meta, site)

        planned, rebuilt = windows
        for got, want in zip(rebuilt, planned):
            assert (got.jd1, got.jd2) == (want.jd1, want.jd2)
        assert (planned[1] - planned[0]).to_value("s") == 86400.0

    def test_a_block_without_it_keeps_the_widened_pass(self, monkeypatch, site):
        """A relative dict without ``search_start`` re-solves around the block's own bounds."""
        params = {
            key: value
            for key, value in TestSourceCESRebuildForwardsRecordedKeys._full_params().items()
            if key != "window"
        }
        seen, _ = TestSourceCESRebuildForwardsRecordedKeys._spy_kernel(monkeypatch)
        t0 = Time("2026-03-15T22:00:00", scale="utc")
        t1 = t0 + TimeDelta(600.0, format="sec")

        sim._generate_source_ces_trajectory(self._meta(params), site, fallback_window=(t0, t1))

        assert "start_time" not in seen
        w0, w1 = seen["window"]
        buffer = sim._SOURCE_CES_WINDOW_BUFFER_SEC
        assert (t0 - w0).to_value("s") == pytest.approx(buffer)
        assert (w1 - t1).to_value("s") == pytest.approx(buffer)

    def test_a_block_without_it_or_any_window_is_not_reconstructable(self, site):
        params = {
            key: value
            for key, value in TestSourceCESRebuildForwardsRecordedKeys._full_params().items()
            if key != "window"
        }
        with pytest.raises(BlockNotReconstructableError, match="no search_start"):
            sim._generate_source_ces_trajectory(self._meta(params), site)

    @pytest.mark.parametrize("record", [[True, False], [2461295.0], None], ids=str)
    def test_a_record_of_the_wrong_form_is_refused_before_the_planner(
        self, monkeypatch, site, record
    ):
        """The rebuild names the key and never reaches the planner."""
        params = TestSourceCESRebuildForwardsRecordedKeys._full_params()
        seen, _ = TestSourceCESRebuildForwardsRecordedKeys._spy_kernel(monkeypatch)
        with pytest.raises(ScanParamsSchemaError, match="search_start"):
            sim._generate_source_ces_trajectory(self._meta(params, search_start=record), site)
        assert seen == {}

    def test_the_record_is_the_instant_to_the_bit(self):
        """``[jd1, jd2]`` through JSON restores the instant exactly; ISO would not."""
        record = json.loads(json.dumps(_search_start_record(self.START)))
        back = _search_start_time(record)
        assert (back.jd1, back.jd2) == (self.START.jd1, self.START.jd2)
        nanosecond = self.START.copy()
        nanosecond.precision = 9
        iso = Time(nanosecond.isot, scale="utc")
        assert (iso.jd1, iso.jd2) != (self.START.jd1, self.START.jd2)


class TestSimulatorErrorTaxonomy:
    """The rebuild path's own refusals are typed, and stay ``ValueError``.

    The two types live in ``overhead/`` (simulator tier) and subclass
    ``PointingError``, so a library-tier consumer never imports a
    simulator-only concept and an existing ``except ValueError`` is
    unaffected.
    """

    def test_both_types_are_pointing_errors(self):
        from fyst_trajectories.exceptions import PointingError

        for cls in (ScanParamsSchemaError, BlockNotReconstructableError):
            assert issubclass(cls, PointingError)
            assert issubclass(cls, ValueError)

    def test_unbuildable_scan_type_is_a_schema_error(self):
        """A science block naming a scan type the rebuild cannot dispatch.

        ``source_ces`` is a valid scan-parameter schema but is only rebuilt on
        the calibration route, so a science block carrying it falls through
        the dispatch chain.
        """
        site = get_fyst_site()
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        block = TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(300, format="sec"),
            block_type="science",
            patch_name="p",
            az_start=170.0,
            az_end=190.0,
            elevation=50.0,
            scan_index=0,
            scan_type="source_ces",
            metadata={
                "ra_center": 24.0,
                "dec_center": -32.0,
                "width": 5.0,
                "height": 5.0,
                "velocity": 0.5,
            },
        )
        with pytest.raises(ScanParamsSchemaError, match="Unknown scan type"):
            _generate_trajectory_for_block(block, site)
