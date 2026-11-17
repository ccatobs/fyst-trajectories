"""A Sun model's ``batch`` solves the Sun once per distinct instant.

The offline scheduler's Sun clip asks an injected model about a grid that
repeats every sampled time across each azimuth column of a span, so the
Sun's position is solved for each distinct instant once and spread over
the grid. These tests pin that the verdicts are the ones each instant gets
on its own, and that the ephemeris is asked about each instant once. They
need no shared library: every model reaches the ephemeris through the same
base-class path, exercised here with the scalar model.
"""

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time, TimeDelta

from fyst_trajectories import get_fyst_site
from fyst_trajectories.sun_models import make_sun_safe

# Sunrise at the site: the Sun climbs from about 3 to 16 deg over the hour,
# so at el 40 the grid holds both safe and unsafe samples.
_T0 = Time("2026-12-15T10:00:00", scale="utc")
_TIMES = _T0 + TimeDelta(np.linspace(0.0, 3600.0, 61), format="sec")
_COLUMNS = np.arange(0.0, 360.0, 2.0)
_EL = 40.0


def _grid(times):
    """Return every column at every time, the layout the duration clip asks about."""
    n_t = len(times)
    az = np.tile(_COLUMNS, n_t)
    return az, np.full(az.size, _EL), times[np.repeat(np.arange(n_t), _COLUMNS.size)]


def _spy_on_the_ephemeris(model, monkeypatch):
    """Record every time array the model hands its ephemeris."""
    seen = []
    solve = model._coords.get_sun_altaz

    def spy(obstime):
        seen.append(obstime)
        return solve(obstime)

    monkeypatch.setattr(model._coords, "get_sun_altaz", spy)
    return seen


class TestOneSolvePerInstant:
    """A grid that repeats its instants is solved once per instant, to the same verdicts."""

    def test_repeated_instants_get_the_verdicts_each_gets_alone(self):
        model = make_sun_safe("scalar")
        got = model.batch(*_grid(_TIMES))
        alone = [model.batch(_COLUMNS, np.full(_COLUMNS.size, _EL), t) for t in _TIMES]
        assert np.array_equal(got, np.concatenate(alone))
        assert got.any() and not got.all()
        _, _, times = _grid(_TIMES)
        spread, whole = model._sun_at_instants(times), model._coords.get_sun_altaz(times)
        assert np.array_equal(spread[0], whole[0]) and np.array_equal(spread[1], whole[1])

    def test_the_ephemeris_is_asked_about_each_instant_once(self, monkeypatch):
        model = make_sun_safe("scalar")
        seen = _spy_on_the_ephemeris(model, monkeypatch)
        model.batch(*_grid(_TIMES))
        (asked,) = seen
        assert len(asked) == len(_TIMES)
        assert sorted(zip(asked.jd1, asked.jd2, strict=True)) == sorted(
            zip(_TIMES.jd1, _TIMES.jd2, strict=True)
        )

    def test_distinct_instants_are_solved_as_given(self, monkeypatch):
        model = make_sun_safe("scalar")
        seen = _spy_on_the_ephemeris(model, monkeypatch)
        n_t = len(_TIMES)
        model.batch(np.full(n_t, 120.0), np.full(n_t, _EL), _TIMES)
        (asked,) = seen
        assert len(asked) == n_t

    def test_an_instant_at_two_locations_is_solved_at_both(self, monkeypatch):
        """A location per element enters the conversion to TDB, so equal dates stay apart."""
        site = get_fyst_site()
        locations = EarthLocation.from_geodetic(
            lon=[site.longitude, 0.0], lat=[site.latitude, 0.0], height=[site.elevation, 0.0]
        )
        times = Time([_T0.isot, _T0.isot], scale="utc", location=locations)
        model = make_sun_safe("scalar")
        seen = _spy_on_the_ephemeris(model, monkeypatch)
        model.batch([120.0, 120.0], [_EL, _EL], times)
        (asked,) = seen
        assert len(asked) == 2
