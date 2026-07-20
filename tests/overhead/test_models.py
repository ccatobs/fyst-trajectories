"""Tests for scheduling data models."""

import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories.overhead.models import (
    BlockType,
    CalibrationPolicy,
    CalibrationSpec,
    CalibrationType,
    ObservingPatch,
    ObservingTimeline,
    OverheadModel,
    TimelineBlock,
)

# Block durations come from a Time subtraction converted to seconds; 0.01 s
# (10 ms) absorbs the float round-off in that conversion.
_DURATION_TOL_SEC = 0.01


class TestObservingPatch:
    """Tests for ObservingPatch dataclass."""

    def test_dec_bounds(self):
        patch = ObservingPatch(
            name="test",
            ra_center=0.0,
            dec_center=-30.0,
            width=10.0,
            height=20.0,
            scan_type="constant_el",
            velocity=1.0,
        )
        assert patch.dec_min == -40.0
        assert patch.dec_max == -20.0

    def test_invalid_width(self):
        with pytest.raises(ValueError, match="width must be positive"):
            ObservingPatch(
                name="bad",
                ra_center=0.0,
                dec_center=0.0,
                width=-1.0,
                height=1.0,
                scan_type="pong",
                velocity=1.0,
            )

    def test_invalid_scan_type(self):
        with pytest.raises(ValueError, match="scan_type"):
            ObservingPatch(
                name="bad",
                ra_center=0.0,
                dec_center=0.0,
                width=1.0,
                height=1.0,
                scan_type="invalid",
                velocity=1.0,
            )

    def test_invalid_velocity(self):
        with pytest.raises(ValueError, match="velocity must be positive"):
            ObservingPatch(
                name="bad",
                ra_center=0.0,
                dec_center=0.0,
                width=1.0,
                height=1.0,
                scan_type="pong",
                velocity=-1.0,
            )


class TestCalibrationSpec:
    """Tests for CalibrationSpec dataclass."""

    def test_valid_types(self):
        for name in ("retune", "pointing_cal", "focus", "skydip", "planet_cal", "beam_map"):
            spec = CalibrationSpec(name=name, duration=10.0)
            assert spec.name == name

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown calibration type"):
            CalibrationSpec(name="invalid", duration=10.0)

    def test_invalid_duration(self):
        with pytest.raises(ValueError, match="duration must be positive"):
            CalibrationSpec(name="retune", duration=-1.0)


class TestCalibrationTypeFieldMappings:
    """Tests for the ``duration_field`` / ``state_field`` mappings."""

    def test_duration_field_resolves_on_overhead_model(self):
        """Every CalibrationType's duration_field must name an OverheadModel attribute."""
        model = OverheadModel()
        for cal_type in CalibrationType:
            # property returns an attribute name, getattr must resolve.
            value = getattr(model, cal_type.duration_field)
            assert value == model.get_calibration_duration(cal_type)

    def test_state_field_round_trip(self):
        """``state.update(CAL, t)`` must populate the attribute named by state_field."""
        from fyst_trajectories.overhead import CalibrationState

        t = Time("2026-06-15T02:00:00", scale="utc")
        for cal_type in CalibrationType:
            state = CalibrationState()
            new_state = state.update(cal_type, t)
            assert getattr(new_state, cal_type.state_field) == t


class TestTimelineBlock:
    """Tests for TimelineBlock dataclass."""

    def test_duration(self):
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        t1 = t0 + TimeDelta(300, format="sec")
        block = TimelineBlock(
            t_start=t0,
            t_stop=t1,
            block_type="science",
            patch_name="test",
            az_start=100.0,
            az_end=200.0,
            elevation=50.0,
            scan_index=0,
        )
        assert abs(block.duration - 300.0) < _DURATION_TOL_SEC

    def test_invalid_block_type(self):
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        with pytest.raises(ValueError, match="block_type"):
            TimelineBlock(
                t_start=t0,
                t_stop=t0,
                block_type="invalid",
                patch_name="test",
                az_start=0.0,
                az_end=0.0,
                elevation=0.0,
                scan_index=0,
            )

    def test_string_block_type_coerced(self):
        """String block_type should be coerced to BlockType enum."""
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        block = TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(60, format="sec"),
            block_type="science",
            patch_name="test",
            az_start=0.0,
            az_end=0.0,
            elevation=0.0,
            scan_index=0,
        )
        assert block.block_type is BlockType.SCIENCE
        # Equality with the string value must still hold (str subclass).
        assert block.block_type == "science"

    def test_enum_block_type(self):
        """Passing the enum directly should work without coercion."""
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        block = TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(60, format="sec"),
            block_type=BlockType.CALIBRATION,
            patch_name="test",
            az_start=0.0,
            az_end=0.0,
            elevation=0.0,
            scan_index=0,
        )
        assert block.block_type is BlockType.CALIBRATION


class TestTimelineBlockFactories:
    """Tests for the TimelineBlock factory classmethods."""

    def _t0(self) -> Time:
        return Time("2026-06-15T02:00:00", scale="utc")

    def test_calibration_factory(self, site):
        t0 = self._t0()
        block = TimelineBlock.calibration(
            cal_type="pointing_cal",
            t_start=t0,
            duration=180.0,
            az=120.0,
            el=45.0,
            site=site,
            scan_index=3,
            target=None,
        )
        assert block.block_type is BlockType.CALIBRATION
        assert block.patch_name == "pointing_cal"
        assert block.scan_type == "pointing_cal"
        assert block.az_start == block.az_end == 120.0
        assert block.elevation == 45.0
        assert block.scan_index == 3
        assert abs(block.duration - 180.0) < _DURATION_TOL_SEC
        assert block.metadata == {"cal_type": "pointing_cal", "target": None}

    def test_calibration_factory_accepts_enum_and_target(self, site):
        t0 = self._t0()
        block = TimelineBlock.calibration(
            cal_type=CalibrationType.PLANET_CAL,
            t_start=t0,
            duration=600.0,
            az=90.0,
            el=30.0,
            site=site,
            scan_index=0,
            target="jupiter",
        )
        assert block.patch_name == "planet_cal"
        assert block.metadata == {"cal_type": "planet_cal", "target": "jupiter"}

    def test_idle_factory(self, site):
        t0 = self._t0()
        block = TimelineBlock.idle(
            t_start=t0,
            duration=60.0,
            az=10.0,
            el=20.0,
            site=site,
            scan_index=7,
        )
        assert block.block_type is BlockType.IDLE
        assert block.patch_name == "no_target"
        assert block.scan_type == "idle"
        assert block.az_start == block.az_end == 10.0
        assert block.elevation == 20.0
        assert block.scan_index == 7
        assert abs(block.duration - 60.0) < _DURATION_TOL_SEC

    def test_slew_factory(self, site):
        t0 = self._t0()
        block = TimelineBlock.slew(
            t_start=t0,
            duration=25.0,
            az_start=100.0,
            az_end=160.0,
            el=40.0,
            site=site,
            scan_index=2,
            patch_name="slew_to_deep56",
        )
        assert block.block_type is BlockType.SLEW
        assert block.patch_name == "slew_to_deep56"
        assert block.scan_type == "slew"
        assert block.az_start == 100.0
        assert block.az_end == 160.0
        assert block.elevation == 40.0
        assert abs(block.duration - 25.0) < _DURATION_TOL_SEC

    def test_science_factory(self, site):
        t0 = self._t0()
        patch = ObservingPatch(
            name="deep56",
            ra_center=55.0,
            dec_center=-27.0,
            width=5.0,
            height=6.0,
            scan_type="constant_el",
            velocity=1.0,
        )
        block = TimelineBlock.science(
            patch=patch,
            t_start=t0,
            duration=600.0,
            az_start=80.0,
            az_end=120.0,
            el=55.0,
            site=site,
            scan_index=4,
            subscan_index=2,
            rising=False,
        )
        assert block.block_type is BlockType.SCIENCE
        assert block.patch_name == "deep56"
        assert block.scan_type == "constant_el"
        assert block.az_start == 80.0
        assert block.az_end == 120.0
        assert block.elevation == 55.0
        assert block.scan_index == 4
        assert block.subscan_index == 2
        assert block.rising is False
        assert block.metadata["ra_center"] == 55.0
        assert block.metadata["dec_center"] == -27.0
        assert block.metadata["width"] == 5.0
        assert block.metadata["height"] == 6.0
        assert block.metadata["velocity"] == 1.0

    def test_retune_factory(self, site):
        t0 = self._t0()
        block = TimelineBlock.retune(
            t_start=t0,
            duration=5.0,
            az_start=100.0,
            az_end=160.0,
            el=45.0,
            site=site,
            scan_index=1,
        )
        assert block.block_type is BlockType.CALIBRATION
        assert block.patch_name == "retune"
        assert block.scan_type == "retune"
        assert block.az_start == 100.0
        assert block.az_end == 160.0
        assert abs(block.duration - 5.0) < _DURATION_TOL_SEC

    def test_calibration_factory_source_ces_kwargs(self, site):
        """The optional source-CES kwargs are honored and recorded in metadata."""
        t0 = self._t0()
        scan_params = {
            "body": "jupiter",
            "footprint": "c",
            "el_bore": 35.0,
            "mode": "rising",
            "window": ["2026-06-15 02:00:00", "2026-06-15 02:07:00"],
            "boresight_rot": 0.0,
            "timestep": 0.1,
            "eta_offset_deg": -0.43,
            "pass_index": 0,
            "n_passes": 3,
        }
        block = TimelineBlock.calibration(
            cal_type="planet_cal",
            t_start=t0,
            duration=420.0,
            az=48.0,
            el=35.0,
            site=site,
            scan_index=2,
            target="jupiter",
            az_end=52.0,
            scan_params=scan_params,
            t0_scan="2026-06-15 02:00:16",
            rising=True,
        )
        assert block.block_type is BlockType.CALIBRATION
        assert block.patch_name == "planet_cal"
        assert block.scan_type == "planet_cal"
        # az_end honored: the block spans a swept azimuth range, not parked.
        assert block.az_start == 48.0
        assert block.az_end == 52.0
        assert block.elevation == 35.0
        assert block.rising is True
        assert block.metadata["cal_type"] == "planet_cal"
        assert block.metadata["target"] == "jupiter"
        assert block.metadata["t0_scan"] == "2026-06-15 02:00:16"
        assert block.metadata["scan_params"] == scan_params

    def test_calibration_factory_default_no_key_leakage(self, site):
        """The parked default path emits exactly the legacy 2-key metadata.

        No ``scan_params`` / ``t0_scan`` keys leak in when the source-CES
        kwargs are omitted, and the block stays parked (az_start == az_end).
        """
        t0 = self._t0()
        block = TimelineBlock.calibration(
            cal_type="planet_cal",
            t_start=t0,
            duration=600.0,
            az=90.0,
            el=30.0,
            site=site,
            scan_index=0,
            target="saturn",
        )
        assert set(block.metadata.keys()) == {"cal_type", "target"}
        assert block.az_start == block.az_end == 90.0
        assert block.rising is True


class TestBlockType:
    """Tests for the BlockType enum."""

    def test_equals_string(self):
        """BlockType inherits from str so == string comparisons work."""
        assert BlockType.SCIENCE == "science"
        assert BlockType.CALIBRATION == "calibration"


class TestOverheadModel:
    """Tests for OverheadModel dataclass."""

    def test_negative_duration(self):
        with pytest.raises(ValueError, match="non-negative"):
            OverheadModel(retune_duration=-1.0)

    def test_min_ge_max(self):
        with pytest.raises(ValueError, match="min_scan_duration"):
            OverheadModel(min_scan_duration=4000.0, max_scan_duration=3600.0)

    def test_get_calibration_duration(self):
        model = OverheadModel(retune_duration=5.0, focus_duration=300.0)
        assert model.get_calibration_duration("retune") == 5.0
        assert model.get_calibration_duration("focus") == 300.0

    def test_get_calibration_duration_unknown(self):
        model = OverheadModel()
        with pytest.raises(ValueError, match="Unknown calibration type"):
            model.get_calibration_duration("invalid")


class TestCalibrationPolicy:
    """Tests for CalibrationPolicy dataclass."""

    def test_negative_cadence(self):
        with pytest.raises(ValueError, match="non-negative"):
            CalibrationPolicy(pointing_cadence=-1.0)

    def test_beam_map_cadence_default_is_none(self):
        """``beam_map_cadence`` defaults to ``None`` (manual-only)."""
        policy = CalibrationPolicy()
        assert policy.beam_map_cadence is None

    def test_beam_map_cadence_negative_rejected(self):
        with pytest.raises(ValueError, match="beam_map_cadence"):
            CalibrationPolicy(beam_map_cadence=-1.0)

    def test_beam_map_cadence_zero_allowed(self):
        """A zero cadence is allowed (always due), same as the others."""
        policy = CalibrationPolicy(beam_map_cadence=0.0)
        assert policy.beam_map_cadence == 0.0

    def test_unknown_planet_cal_footprint_rejected(self):
        """An unknown module tag is rejected at construction, regardless of ``planet_cal_scan``."""
        with pytest.raises(ValueError, match="planet_cal_footprint.*'zzz'.*Available"):
            CalibrationPolicy(planet_cal_footprint="zzz")

    def test_valid_planet_cal_footprint_accepted(self):
        """Known module tags construct cleanly, with or without the scan flag."""
        assert CalibrationPolicy(planet_cal_footprint="i1").planet_cal_footprint == "i1"
        policy = CalibrationPolicy(planet_cal_footprint="center", planet_cal_scan=True)
        assert policy.planet_cal_footprint == "center"


class TestBeamMapScheduling:
    """Tests for BEAM_MAP first-class scheduling behaviour."""

    def test_default_policy_skips_beam_map(self):
        """With ``beam_map_cadence=None`` no automatic beam map is scheduled."""
        from fyst_trajectories.overhead import CalibrationState

        state = CalibrationState()
        policy = CalibrationPolicy()  # beam_map_cadence is None by default
        overhead = OverheadModel()
        t = Time("2026-06-15T02:00:00", scale="utc")
        needed = state.needs_calibration(t, policy, overhead, coords=None)
        assert all(spec.name != CalibrationType.BEAM_MAP for spec in needed)

    def test_explicit_cadence_schedules_beam_map(self):
        """Setting ``beam_map_cadence`` adds beam maps to ``needs_calibration``."""
        from fyst_trajectories.overhead import CalibrationState

        state = CalibrationState()
        policy = CalibrationPolicy(beam_map_cadence=86400.0)
        overhead = OverheadModel()
        t = Time("2026-06-15T02:00:00", scale="utc")
        needed = state.needs_calibration(t, policy, overhead, coords=None)
        beam_specs = [spec for spec in needed if spec.name == CalibrationType.BEAM_MAP]
        assert len(beam_specs) == 1
        assert beam_specs[0].duration == overhead.beam_map_duration

    def test_beam_map_state_round_trip(self):
        """``CalibrationState.update("beam_map", t)`` populates ``last_beam_map``."""
        from fyst_trajectories.overhead import CalibrationState

        state = CalibrationState()
        t = Time("2026-06-15T02:00:00", scale="utc")
        new_state = state.update("beam_map", t)
        assert new_state.last_beam_map == t


class TestObservingTimeline:
    """Tests for ObservingTimeline dataclass."""

    def test_empty_timeline(self, site):
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        t1 = t0 + TimeDelta(3600, format="sec")
        tl = ObservingTimeline(
            blocks=[],
            site=site,
            start_time=t0,
            end_time=t1,
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )
        assert tl.n_science_scans == 0
        assert tl.efficiency == 0.0
        assert tl.total_science_time == 0.0

    def test_efficiency(self, site):
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        t1 = t0 + TimeDelta(1000, format="sec")
        science = TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(500, format="sec"),
            block_type="science",
            patch_name="test",
            az_start=100.0,
            az_end=200.0,
            elevation=50.0,
            scan_index=0,
        )
        cal = TimelineBlock(
            t_start=t0 + TimeDelta(500, format="sec"),
            t_stop=t0 + TimeDelta(600, format="sec"),
            block_type="calibration",
            patch_name="retune",
            az_start=100.0,
            az_end=100.0,
            elevation=50.0,
            scan_index=0,
        )
        tl = ObservingTimeline(
            blocks=[science, cal],
            site=site,
            start_time=t0,
            end_time=t1,
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )
        assert tl.n_science_scans == 1
        # efficiency is a dimensionless science/total ratio; 0.01 = one percentage point.
        assert abs(tl.efficiency - 0.5) < 0.01

    def test_validate_clean(self, site):
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        t1 = t0 + TimeDelta(1000, format="sec")
        block = TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(500, format="sec"),
            block_type="science",
            patch_name="test",
            az_start=100.0,
            az_end=200.0,
            elevation=50.0,
            scan_index=0,
        )
        tl = ObservingTimeline(
            blocks=[block],
            site=site,
            start_time=t0,
            end_time=t1,
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )
        warnings = tl.validate()
        assert warnings == []

    def test_validate_overlap(self, site):
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        t1 = t0 + TimeDelta(1000, format="sec")
        b1 = TimelineBlock(
            t_start=t0,
            t_stop=t0 + TimeDelta(600, format="sec"),
            block_type="science",
            patch_name="a",
            az_start=100.0,
            az_end=200.0,
            elevation=50.0,
            scan_index=0,
        )
        b2 = TimelineBlock(
            t_start=t0 + TimeDelta(500, format="sec"),
            t_stop=t0 + TimeDelta(800, format="sec"),
            block_type="science",
            patch_name="b",
            az_start=100.0,
            az_end=200.0,
            elevation=50.0,
            scan_index=1,
        )
        tl = ObservingTimeline(
            blocks=[b1, b2],
            site=site,
            start_time=t0,
            end_time=t1,
            overhead_model=OverheadModel(),
            calibration_policy=CalibrationPolicy(),
        )
        warnings = tl.validate()
        assert len(warnings) == 1
        assert "Overlap" in warnings[0]
