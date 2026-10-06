"""Tests for the ``sun_events`` almanac and its threshold-crossing search."""

import numpy as np
import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories import Coordinates
from fyst_trajectories.coordinates import _build_time_grid, _threshold_crossings
from fyst_trajectories.observability import (
    ASTRONOMICAL_TWILIGHT_ALTITUDE_DEG,
    CIVIL_TWILIGHT_ALTITUDE_DEG,
    NAUTICAL_TWILIGHT_ALTITUDE_DEG,
    SUN_RISE_SET_ALTITUDE_DEG,
    SunEventKind,
    sun_events,
)


# sun_events: one FYST day from local noon yields the full 8-event
# sequence, dusk side first, in strict time order with sane times.
def test_sun_events_full_day_sequence():
    t = Time("2026-11-15T16:00:00", scale="utc")  # ~13:00 Chile local
    events = sun_events(t)
    kinds = [e.kind for e in events]
    assert kinds == [
        SunEventKind.SUNSET,
        SunEventKind.CIVIL_DUSK,
        SunEventKind.NAUTICAL_DUSK,
        SunEventKind.ASTRONOMICAL_DUSK,
        SunEventKind.ASTRONOMICAL_DAWN,
        SunEventKind.NAUTICAL_DAWN,
        SunEventKind.CIVIL_DAWN,
        SunEventKind.SUNRISE,
    ]
    assert [e.rising for e in events] == [False] * 4 + [True] * 4
    mjds = [e.time.mjd for e in events]
    assert mjds == sorted(mjds)
    sunset = events[0]
    sunrise = events[-1]
    # Measured with the vendored IERS table: set 22:52:27, rise 09:38:41 UTC.
    # The windows are two minutes either side, deliberately tight enough to
    # exclude the geometric (0 deg) crossings at 22:48:35 and 09:42:33, so a
    # revert from the almanac convention to the geometric one fails here.
    assert Time("2026-11-15T22:50:30", scale="utc") <= sunset.time
    assert sunset.time <= Time("2026-11-15T22:54:30", scale="utc")
    assert Time("2026-11-16T09:36:40", scale="utc") <= sunrise.time
    assert sunrise.time <= Time("2026-11-16T09:40:40", scale="utc")


# The four published altitude constants hold their almanac values, and the
# solver is wired to them. test_sun_events_altitude_invariant below cannot see
# either: it compares the Sun's altitude at an event against the threshold the
# solver was handed, so it holds for any constants. Mutating any of the four
# leaves the rest of the suite green.
def test_sun_event_altitude_constants_are_the_almanac_conventions():
    assert SUN_RISE_SET_ALTITUDE_DEG == -0.8333  # -50': refraction + solar semidiameter
    assert CIVIL_TWILIGHT_ALTITUDE_DEG == -6.0
    assert NAUTICAL_TWILIGHT_ALTITUDE_DEG == -12.0
    assert ASTRONOMICAL_TWILIGHT_ALTITUDE_DEG == -18.0
    # Wiring, not only values: astronomical dusk on the same night is the
    # -18 deg crossing, measured 2026-11-16 00:15:03. A threshold paired with
    # the wrong event kind moves this without touching a constant.
    events = sun_events(Time("2026-11-15T16:00:00", scale="utc"))
    dusk = next(e for e in events if e.kind is SunEventKind.ASTRONOMICAL_DUSK)
    assert dusk.altitude_deg == ASTRONOMICAL_TWILIGHT_ALTITUDE_DEG
    assert Time("2026-11-16T00:13:00", scale="utc") <= dusk.time
    assert dusk.time <= Time("2026-11-16T00:17:00", scale="utc")


# sun_events: the Sun's geometric altitude at each event time equals the
# event's threshold (locks the interpolation and the vacuum convention).
def test_sun_events_altitude_invariant(site):
    events = sun_events(Time("2026-11-15T16:00:00", scale="utc"), site=site)
    coords = Coordinates(site)  # vacuum, matching the implementation
    assert events
    for event in events:
        _, el = coords.get_sun_altaz(event.time)
        # Locks the "seconds level" interpolation claim: 1e-3 deg is ~0.3 s
        # of solar altitude motion at FYST twilight rates.
        assert el == pytest.approx(event.altitude_deg, abs=1e-3)


# sun_events: parameter validation (incl. the NaN/inf hole: NaN passes
# a bare `<= 0` and would silently return an empty tuple).
def test_sun_events_validation():
    t = Time("2026-11-15T16:00:00", scale="utc")
    for bad_horizon in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            sun_events(t, horizon_hours=bad_horizon)
    for bad_step in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            sun_events(t, step_minutes=bad_step)


# _threshold_crossings: synthetic arrays lock the crossing partition,
# the interpolation, and the clipped-final-cell handling (no ephemeris).
def test_threshold_crossings_synthetic():
    t0 = Time("2026-06-15T00:00:00", scale="utc")
    grid4 = t0 + TimeDelta(np.arange(4) * 600.0, format="sec")

    # Plain interior crossing: linear interpolation between samples.
    up = _threshold_crossings(np.array([-2.0, -1.0, 0.5, 2.0]), grid4, 0.0, rising=True)
    assert len(up) == 1
    assert (up[0] - t0).to_value("s") == pytest.approx(600.0 + 600.0 * (1.0 / 1.5), abs=1e-6)

    # A value exactly AT the threshold on a grid sample: exactly one event,
    # landing exactly on that sample (frac = 1 in the preceding cell).
    grid3 = t0 + TimeDelta(np.arange(3) * 600.0, format="sec")
    exact = _threshold_crossings(np.array([-1.0, 0.0, 1.0]), grid3, 0.0, rising=True)
    assert len(exact) == 1
    assert abs((exact[0] - grid3[1]).to_value("s")) < 1e-9

    # Plateau at the threshold: still a single event, not one per sample.
    plateau = _threshold_crossings(np.array([-1.0, 0.0, 0.0, 1.0]), grid4, 0.0, rising=True)
    assert len(plateau) == 1

    # Tangential touch: landing exactly ON the threshold yields no events
    # (the >=/< partition), while dipping infinitesimally below yields a
    # set+rise pair. Pins the boundary semantics.
    assert _threshold_crossings(np.array([1.0, 0.0, 1.0]), grid3, 0.0, rising=True) == []
    assert _threshold_crossings(np.array([1.0, 0.0, 1.0]), grid3, 0.0, rising=False) == []
    dip = np.array([1.0, -1e-9, 1.0])
    assert len(_threshold_crossings(dip, grid3, 0.0, rising=False)) == 1
    assert len(_threshold_crossings(dip, grid3, 0.0, rising=True)) == 1

    # No crossings => empty.
    assert _threshold_crossings(np.array([1.0, 2.0, 3.0]), grid3, 0.0, rising=True) == []

    # Clipped final cell from _build_time_grid (horizon 9 min @ 4 min step =>
    # cells of 240/240/60 s): interpolation must use the actual 60 s spacing.
    grid_clip = _build_time_grid(t0, horizon_hours=0.15, step_minutes=4.0)
    assert (grid_clip[-1] - grid_clip[-2]).to_value("s") == pytest.approx(60.0)
    clipped = _threshold_crossings(np.array([-3.0, -2.0, -1.0, 1.0]), grid_clip, 0.0, rising=True)
    assert len(clipped) == 1
    assert (clipped[0] - t0).to_value("s") == pytest.approx(480.0 + 0.5 * 60.0, abs=1e-6)


# sun_events: a sub-day evening span returns only the dusk side.
def test_sun_events_subday_dusk_only():
    events = sun_events(Time("2026-11-15T21:00:00", scale="utc"), horizon_hours=4.0)
    assert [e.kind for e in events] == [
        SunEventKind.SUNSET,
        SunEventKind.CIVIL_DUSK,
        SunEventKind.NAUTICAL_DUSK,
        SunEventKind.ASTRONOMICAL_DUSK,
    ]
    assert all(e.rising is False for e in events)


# sun_events: a 48 h horizon returns two full days of events, sorted,
# in alternating dusk-block / dawn-block order.
def test_sun_events_two_days():
    events = sun_events(Time("2026-11-15T16:00:00", scale="utc"), horizon_hours=48.0)
    assert len(events) == 16
    mjds = [e.time.mjd for e in events]
    assert mjds == sorted(mjds)
    assert [e.rising for e in events] == [False] * 4 + [True] * 4 + [False] * 4 + [True] * 4
