"""Tests for calibration overhead tracking."""

import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories.overhead import CalibrationState
from fyst_trajectories.overhead.models import CalibrationPolicy, OverheadModel


class TestCalibrationState:
    """What is due when, in priority order, plus immutable updates and planet picks."""

    def test_all_due_initially(self):
        state = CalibrationState()
        t = Time("2026-06-15T02:00:00", scale="utc")
        policy = CalibrationPolicy()
        overhead = OverheadModel()
        needed = state.needs_calibration(t, policy, overhead)
        names = [c.name for c in needed]
        assert "retune" in names
        assert "pointing_cal" in names
        assert "focus" in names
        assert "skydip" in names
        assert "planet_cal" in names

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
        state = CalibrationState()
        t = Time("2026-06-15T02:00:00", scale="utc")
        policy = CalibrationPolicy()
        overhead = OverheadModel()
        needed = state.needs_calibration(t, policy, overhead)
        assert needed[0].name == "retune"

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

    def test_find_visible_planet_uses_min_elevation(self):
        class MockCoords:
            def __init__(self, altitudes: dict[str, float]):
                self._altitudes = altitudes

            def get_body_altaz(self, body: str, time):
                return 0.0, self._altitudes.get(body, -10.0)

        t = Time("2026-06-15T02:00:00", scale="utc")
        coords = MockCoords({"jupiter": 15.0, "saturn": 25.0})
        result = CalibrationState._find_visible_planet(
            ("jupiter", "saturn"), t, coords, min_elevation=20.0
        )
        assert result == "saturn"

        result = CalibrationState._find_visible_planet(
            ("jupiter", "saturn"), t, coords, min_elevation=10.0
        )
        assert result == "jupiter"

        result = CalibrationState._find_visible_planet(("jupiter",), t, coords, min_elevation=30.0)
        assert result is None

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
