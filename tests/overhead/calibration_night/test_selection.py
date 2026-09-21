"""Tests for the selection rules."""

import pytest
from astropy.time import Time

from fyst_trajectories.overhead import (
    Candidate,
    DeferralReason,
    NightState,
    ScanOverrides,
    ScriptedSelection,
    SelectionRule,
    select_priority,
)

T0 = Time("2026-09-11T06:30:00", scale="utc")


def _candidate(body, reason=None):
    return Candidate(
        body=body,
        el_bore_estimate=40.0,
        az_throw=2.5,
        bin=None,
        sun_ok=True,
        moon_separation=None,
        time_left_in_band=3600.0,
        reason=reason,
    )


class TestSelectPriority:
    """First available body in the offered order."""

    def test_picks_the_first_available(self):
        candidates = (
            _candidate("jupiter", DeferralReason.BELOW_BAND),
            _candidate("saturn"),
            _candidate("neptune"),
        )
        assert select_priority(candidates, NightState.initial(T0)) == ("saturn", ScanOverrides())

    def test_none_when_nothing_is_available(self):
        candidates = (_candidate("jupiter", DeferralReason.SUN_POINT),)
        assert select_priority(candidates, NightState.initial(T0)) is None

    def test_satisfies_the_protocol(self):
        assert isinstance(select_priority, SelectionRule)
        assert isinstance(ScriptedSelection(["saturn"]), SelectionRule)


class TestScriptedSelection:
    """A fixed script whose cursor lives in the state."""

    def test_normalises_entries(self):
        rule = ScriptedSelection(["Saturn", ("Uranus", ScanOverrides(az_speed=1.0))])
        assert rule.entries[0] == ("saturn", ScanOverrides())
        assert rule.entries[1] == ("uranus", ScanOverrides(az_speed=1.0))
        with pytest.raises(ValueError, match="must not be empty"):
            ScriptedSelection([])

    def test_returns_the_current_entry_only_when_its_body_is_available(self):
        rule = ScriptedSelection(["saturn", "uranus"])
        state = NightState.initial(T0)
        both = (_candidate("saturn"), _candidate("uranus"))
        assert rule(both, state) == ("saturn", ScanOverrides())
        blocked = (_candidate("saturn", DeferralReason.SUN_POINT), _candidate("uranus"))
        assert rule(blocked, state) is None
        later = state.advanced(script_index=1)
        assert rule(both, later) == ("uranus", ScanOverrides())
        spent = state.advanced(script_index=2)
        assert rule.current(spent) is None
        assert rule(both, spent) is None

    def test_rule_holds_no_cursor_of_its_own(self):
        rule = ScriptedSelection(["saturn", "saturn"])
        state = NightState.initial(T0)
        first = rule((_candidate("saturn"),), state)
        second = rule((_candidate("saturn"),), state)
        assert first == second == ("saturn", ScanOverrides())
