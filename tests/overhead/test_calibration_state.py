"""Tests for calibration cadence tracking (CalibrationState)."""

import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories import get_fyst_site
from fyst_trajectories.overhead import CalibrationState, ObservingPatch, generate_timeline
from fyst_trajectories.overhead.models import CalibrationPolicy, OverheadModel


class TestCalibrationState:
    """What is due when, in priority order, plus immutable updates and planet picks."""

    def test_retune_always_due_with_zero_cadence(self):
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        state = CalibrationState(last_retune=t0)
        t1 = t0 + TimeDelta(1, format="sec")
        policy = CalibrationPolicy(retune_cadence=0.0)
        overhead = OverheadModel()
        needed = state.needs_calibration(t1, policy, overhead)
        names = [c.name for c in needed]
        assert "retune" in names

    def test_nothing_due_when_recent(self):
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        state = CalibrationState(
            last_retune=t0,
            last_pointing_cal=t0,
            last_focus=t0,
            last_skydip=t0,
            last_planet_cal=t0,
        )
        t1 = t0 + TimeDelta(10, format="sec")
        policy = CalibrationPolicy(
            retune_cadence=3600.0,
            pointing_cadence=3600.0,
            focus_cadence=7200.0,
            skydip_cadence=28800.0,
            planet_cal_cadence=43200.0,
        )
        overhead = OverheadModel()
        needed = state.needs_calibration(t1, policy, overhead)
        assert len(needed) == 0

    def test_pointing_due_after_cadence(self):
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        state = CalibrationState(
            last_retune=t0,
            last_pointing_cal=t0,
            last_focus=t0,
            last_skydip=t0,
            last_planet_cal=t0,
        )
        t1 = t0 + TimeDelta(7200, format="sec")
        policy = CalibrationPolicy(
            retune_cadence=86400.0,
            pointing_cadence=3600.0,
            focus_cadence=86400.0,
            skydip_cadence=86400.0,
            planet_cal_cadence=86400.0,
        )
        overhead = OverheadModel()
        needed = state.needs_calibration(t1, policy, overhead)
        names = [c.name for c in needed]
        assert "pointing_cal" in names
        assert "retune" not in names

    def test_priority_order(self):
        """A fresh state is due everything, in the documented priority order.

        Beam maps stay off the automatic schedule (``beam_map_cadence`` is
        None by default), so five calibrations come back.
        """
        state = CalibrationState()
        t = Time("2026-06-15T02:00:00", scale="utc")
        needed = state.needs_calibration(t, CalibrationPolicy(), OverheadModel())
        assert [c.name for c in needed] == [
            "retune",
            "pointing_cal",
            "focus",
            "skydip",
            "planet_cal",
        ]

    def test_update_returns_new_state(self):
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        state = CalibrationState()
        new_state = state.update("retune", t0)
        assert new_state.last_retune == t0
        assert state.last_retune is None

    def test_update_unknown_type(self):
        t0 = Time("2026-06-15T02:00:00", scale="utc")
        state = CalibrationState()
        with pytest.raises(ValueError, match="Unknown calibration type"):
            state.update("invalid", t0)

    def test_planet_cal_has_target(self):
        state = CalibrationState()
        t = Time("2026-06-15T02:00:00", scale="utc")
        policy = CalibrationPolicy(planet_targets=("jupiter", "saturn"))
        overhead = OverheadModel()
        needed = state.needs_calibration(t, policy, overhead)
        planet_cals = [c for c in needed if c.name == "planet_cal"]
        assert len(planet_cals) == 1
        assert planet_cals[0].target == "jupiter"

    def test_planet_choice_follows_target_order_above_min_elevation(self):
        """The first listed planet above ``planet_min_elevation`` is the target."""

        class MockCoords:
            def get_body_altaz(self, body: str, time):
                return 0.0, {"jupiter": 15.0, "saturn": 25.0}.get(body, -10.0)

        t = Time("2026-06-15T02:00:00", scale="utc")
        for min_el, expected in ((20.0, "saturn"), (10.0, "jupiter")):
            policy = CalibrationPolicy(
                planet_targets=("jupiter", "saturn"), planet_min_elevation=min_el
            )
            needed = CalibrationState().needs_calibration(
                t, policy, OverheadModel(), coords=MockCoords()
            )
            planet_cals = [c for c in needed if c.name == "planet_cal"]
            assert [c.target for c in planet_cals] == [expected]

    def test_planet_min_elevation_threaded_through_needs_calibration(self):
        class MockCoords:
            def get_body_altaz(self, body: str, time):
                return 0.0, 15.0

        state = CalibrationState()
        t = Time("2026-06-15T02:00:00", scale="utc")
        coords = MockCoords()

        policy = CalibrationPolicy(
            planet_targets=("jupiter",),
            planet_min_elevation=20.0,
        )
        overhead = OverheadModel()
        needed = state.needs_calibration(t, policy, overhead, coords=coords)
        planet_cals = [c for c in needed if c.name == "planet_cal"]
        assert len(planet_cals) == 0

        policy_low = CalibrationPolicy(
            planet_targets=("jupiter",),
            planet_min_elevation=10.0,
        )
        needed = state.needs_calibration(t, policy_low, overhead, coords=coords)
        planet_cals = [c for c in needed if c.name == "planet_cal"]
        assert len(planet_cals) == 1
        assert planet_cals[0].target == "jupiter"


class TestZeroDurationCalibrations:
    """A zero duration in the overhead model schedules no such operation."""

    def test_a_zero_duration_calibration_is_not_due(self):
        t = Time("2026-06-15T02:00:00", scale="utc")
        overhead = OverheadModel(retune_duration=0.0, skydip_duration=0.0)
        needed = CalibrationState().needs_calibration(t, CalibrationPolicy(), overhead)
        assert [c.name for c in needed] == ["pointing_cal", "focus", "planet_cal"]

    def test_a_scan_mode_planet_calibration_is_sized_by_its_passes(self):
        t = Time("2026-06-15T02:00:00", scale="utc")
        overhead = OverheadModel(planet_cal_duration=0.0)
        parked = CalibrationState().needs_calibration(
            t, CalibrationPolicy(planet_targets=("jupiter",)), overhead
        )
        scanned = CalibrationState().needs_calibration(
            t, CalibrationPolicy(planet_targets=("jupiter",), planet_cal_scan=True), overhead
        )
        assert "planet_cal" not in [c.name for c in parked]
        assert [(c.name, c.duration, c.target) for c in scanned if c.name == "planet_cal"] == [
            ("planet_cal", 0.0, "jupiter")
        ]

    def test_the_survey_scheduler_runs_without_them(self):
        """The model that holds no retune or skydip on a calibration night does here too."""
        deep56 = ObservingPatch(
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
            patches=[deep56],
            site=get_fyst_site(),
            start_time="2026-06-15T00:00:00",
            end_time="2026-06-15T01:00:00",
            overhead_model=OverheadModel(retune_duration=0.0, skydip_duration=0.0),
        )
        names = {b.patch_name for b in timeline.calibration_blocks}
        assert names
        assert not names & {"retune", "skydip"}
