"""Sun clip of the offline scheduler's constant-elevation visits.

A constant-elevation visit sweeps its whole azimuth corridor on every leg,
so its Sun duration clip has to hold at every azimuth of that corridor at
the scan elevation, not only along the field centre. These tests pin:

- the ``az_span`` option of ``_time_until_sun_unsafe``. In scalar mode it
  is exact, checked against a dense brute-force azimuth grid, wrapped spans
  included. An injected model is asked about a grid of azimuth columns.
  Without the option the result is unchanged.
- the constant-elevation branch of ``_compute_scan_duration``, which
  refuses a visit whose corridor holds the Sun's azimuth while its field
  centre is clear.
- (slow) a 24-hour five-patch night whose rebuilt constant-elevation
  science clears the exclusion radius at every sample.
"""

import warnings

import numpy as np
import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories.overhead import (
    ObservingPatch,
    OverheadModel,
    generate_timeline,
    schedule_to_trajectories,
)
from fyst_trajectories.overhead.scheduler.helpers import (
    _SPAN_AZ_STEP_DEG,
    _ce_swept_az_envelope,
    _ce_visit_plan,
    _compute_scan_duration,
    _first_crossing,
    _span_columns,
    _time_until_sun_unsafe,
)

# Sunrise at the site: the Sun climbs from 3 deg (az ~114) to ~16 deg over
# the hour, so a span holding its azimuth at el 60 turns unsafe mid-window.
_MORNING = Time("2026-12-15T10:00:00", scale="utc")
_EL = 60.0
_WINDOW = 3600.0
# A field whose azimuth (about 167 to 155 deg over that hour) keeps it clear
# of the Sun at el 60. It is below the horizon, but under ``fixed_el`` only
# its azimuth matters.
_FAR_FIELD = (330.0, -40.0)


def _brute_force(coords, t0, el, span, radius, max_duration, step=60.0, az_step=0.05):
    """Crossing time from the minimum separation over a dense azimuth grid."""
    n_steps = max(2, int(max_duration / step) + 1)
    dt = np.linspace(0.0, max_duration, n_steps)
    times = t0 + TimeDelta(dt, format="sec")
    sun_az, sun_el = coords.get_sun_altaz(times)
    grid = np.arange(span[0], span[1] + az_step / 2, az_step)
    az = np.tile(grid, n_steps)
    sep = coords.angular_separation(
        az,
        np.full(az.size, el),
        np.repeat(np.asarray(sun_az), grid.size),
        np.repeat(np.asarray(sun_el), grid.size),
    )
    min_sep = np.asarray(sep).reshape(n_steps, grid.size).min(axis=1)
    return _first_crossing(dt, min_sep, radius, min_sep <= radius, max_duration)


class TestScalarSpan:
    """The scalar branch finds the span's closest point analytically."""

    def _sun_az(self, coords):
        sun_az, _ = coords.get_sun_altaz(_MORNING)
        return float(sun_az)

    def test_span_holding_the_sun_clips_earlier_than_the_centre(self, coordinates, site):
        radius = site.sun_avoidance.exclusion_radius
        ra, dec = _FAR_FIELD
        sun_az = self._sun_az(coordinates)
        centre = _time_until_sun_unsafe(
            ra, dec, _MORNING, _WINDOW, coordinates, radius, fixed_el=_EL
        )
        holding = _time_until_sun_unsafe(
            ra,
            dec,
            _MORNING,
            _WINDOW,
            coordinates,
            radius,
            fixed_el=_EL,
            az_span=(sun_az - 30.0, sun_az + 30.0),
        )
        far_side = _time_until_sun_unsafe(
            ra,
            dec,
            _MORNING,
            _WINDOW,
            coordinates,
            radius,
            fixed_el=_EL,
            az_span=(sun_az + 140.0, sun_az + 220.0),
        )
        assert centre == _WINDOW
        assert 0.0 < holding < centre
        assert far_side == _WINDOW

    @pytest.mark.parametrize(
        "offsets",
        [
            (-30.0, 30.0),  # holds the Sun's azimuth
            (8.0, 80.0),  # the lower edge is nearest all hour
            (-100.0, -8.0),  # the upper edge is nearest all hour
            (-390.0, -330.0),  # the first span, one turn down
            (15.0 - 360.0, 80.0 + 360.0),  # wider than a turn: every azimuth
        ],
    )
    def test_matches_a_dense_brute_force_grid(self, coordinates, site, offsets):
        radius = site.sun_avoidance.exclusion_radius
        sun_az = self._sun_az(coordinates)
        span = (sun_az + offsets[0], sun_az + offsets[1])
        got = _time_until_sun_unsafe(
            *_FAR_FIELD, _MORNING, _WINDOW, coordinates, radius, fixed_el=_EL, az_span=span
        )
        brute_span = span if span[1] - span[0] < 360.0 else (span[0], span[0] + 360.0)
        expected = _brute_force(coordinates, _MORNING, _EL, brute_span, radius, _WINDOW)
        assert 0.0 < got < _WINDOW  # every case crosses inside the hour
        assert got == pytest.approx(expected, abs=2.0)

    def test_without_a_span_the_centre_track_decides(self, coordinates, site):
        """``az_span=None`` is the centre-track computation, unchanged."""
        radius = site.sun_avoidance.exclusion_radius
        ra, dec = 210.0, -30.0  # tracks the Sun's azimuth this hour
        got = _time_until_sun_unsafe(ra, dec, _MORNING, _WINDOW, coordinates, radius, fixed_el=_EL)
        n_steps = int(_WINDOW / 60.0) + 1
        dt = np.linspace(0.0, _WINDOW, n_steps)
        times = _MORNING + TimeDelta(dt, format="sec")
        az, _ = coordinates.radec_to_altaz(np.full(n_steps, ra), np.full(n_steps, dec), times)
        el = np.full(n_steps, _EL)
        sun_az, sun_el = coordinates.get_sun_altaz(times)
        sep = np.asarray(coordinates.angular_separation(az, el, sun_az, sun_el), dtype=float)
        assert got == _first_crossing(dt, sep, radius, sep <= radius, _WINDOW)
        assert 0.0 < got < _WINDOW

    def test_at_the_zenith_every_azimuth_is_one_point(self, coordinates, site):
        radius = site.sun_avoidance.exclusion_radius
        near = _time_until_sun_unsafe(
            *_FAR_FIELD, _MORNING, _WINDOW, coordinates, radius, fixed_el=90.0, az_span=(0, 10)
        )
        centre = _time_until_sun_unsafe(
            *_FAR_FIELD, _MORNING, _WINDOW, coordinates, radius, fixed_el=90.0
        )
        assert near == pytest.approx(centre, abs=1e-6)

    @pytest.mark.parametrize("span", [(10.0, 5.0), (float("nan"), 5.0), (0.0, float("inf"))])
    def test_rejects_an_unordered_or_non_finite_span(self, coordinates, span):
        with pytest.raises(ValueError, match="az_span"):
            _time_until_sun_unsafe(
                *_FAR_FIELD, _MORNING, 600.0, coordinates, 45.0, fixed_el=_EL, az_span=span
            )


class TestInjectedModelSpan:
    """An injected model is asked about every azimuth column of the span."""

    @staticmethod
    def _wedge_model(t0, lo, hi, with_batch):
        """Unsafe inside the azimuth wedge ``[lo, hi]`` (mod 360) after 600 s."""

        def verdict(az, t):
            late = np.asarray((t - t0).sec) >= 600.0
            in_wedge = np.mod(np.asarray(az, dtype=float) - lo, 360.0) <= hi - lo
            return ~(late & in_wedge)

        class _Model:
            def __call__(self, az, el, t):
                return bool(verdict(az, t))

        if with_batch:
            _Model.batch = lambda self, az, el, t: verdict(az, t)
        return _Model()

    @pytest.mark.parametrize("with_batch", [True, False])
    @pytest.mark.parametrize(
        ("span", "wedge"),
        [((0.0, 100.0), (70.0, 73.0)), ((-40.0, 20.0), (330.0, 333.0))],
    )
    def test_a_wedge_inside_the_span_clips(self, coordinates, with_batch, span, wedge):
        model = self._wedge_model(_MORNING, *wedge, with_batch)
        # The far field's own azimuth (about 167 to 162 deg here) never enters the wedge.
        centre = _time_until_sun_unsafe(
            *_FAR_FIELD, _MORNING, 1200.0, coordinates, 45.0, sun_safe=model, fixed_el=_EL
        )
        spanned = _time_until_sun_unsafe(
            *_FAR_FIELD,
            _MORNING,
            1200.0,
            coordinates,
            45.0,
            sun_safe=model,
            fixed_el=_EL,
            az_span=span,
        )
        assert centre == 1200.0
        # Verdict bisection returns the last verified-safe time.
        assert spanned == pytest.approx(600.0, abs=0.1)

    def test_columns_hold_both_edges_and_the_centre(self):
        cols = _span_columns((10.0, 10.5))
        assert cols.tolist() == [10.0, 10.25, 10.5]
        wide = _span_columns((-30.0, 125.0))
        assert wide[0] == 330.0 and wide[-1] == 125.0
        assert 47.5 in wide
        steps = np.degrees(np.diff(np.unwrap(np.radians(wide))))
        assert np.all(steps <= _SPAN_AZ_STEP_DEG + 1e-9)

    def test_batch_shape_guard(self, coordinates):
        class _Short:
            def __call__(self, az, el, t):
                return True

            def batch(self, az, el, t):
                return np.array([True])

        with pytest.raises(ValueError, match="azimuth-span grid"):
            _time_until_sun_unsafe(
                *_FAR_FIELD,
                _MORNING,
                300.0,
                coordinates,
                45.0,
                sun_safe=_Short(),
                fixed_el=_EL,
                az_span=(0.0, 10.0),
            )


# The field is at el 75 in azimuth ~240 while the Sun, near el 55, sits at
# azimuth ~98: the centre is clear, but the visit's corridor at el 55 runs
# from ~105 to ~261 deg and crosses the Sun's side of the sky.
_OVERHEAD_SUN = Time("2026-12-15T13:55:00", scale="utc")
_W4 = ObservingPatch(
    name="W4",
    ra_center=210.0,
    dec_center=-30.0,
    width=40.0,
    height=10.0,
    scan_type="constant_el",
    velocity=1.0,
    elevation=55.0,
)


def test_ce_visit_is_refused_when_its_corridor_meets_the_zone(coordinates, site):
    end_time = _OVERHEAD_SUN + TimeDelta(12 * 3600, format="sec")
    plan = _ce_visit_plan(_W4, 55.0, _OVERHEAD_SUN, end_time, coordinates, {}, 480.0)
    assert plan is not None
    _, t_open, t_close = plan
    corridor = min((t_close - _OVERHEAD_SUN).sec, (end_time - _OVERHEAD_SUN).sec)

    centre_only = _time_until_sun_unsafe(
        _W4.ra_center,
        _W4.dec_center,
        _OVERHEAD_SUN,
        corridor,
        coordinates,
        site.sun_avoidance.exclusion_radius,
        fixed_el=55.0,
    )
    assert centre_only > 3600.0  # the field centre alone would book the hours

    lo, hi = _ce_swept_az_envelope(_W4, t_open, t_close, coordinates)
    sun_az, sun_el = coordinates.get_sun_altaz(_OVERHEAD_SUN)
    edge_sep = coordinates.angular_separation(
        np.array([lo, hi]), 55.0, float(sun_az), float(sun_el)
    )
    assert float(np.min(edge_sep)) < site.sun_avoidance.exclusion_radius

    dur = _compute_scan_duration(
        _W4, _OVERHEAD_SUN, end_time, site, coordinates, OverheadModel(), 55.0
    )
    assert dur == 0.0


def _sun_unit_vectors(coords, times):
    """Sun direction per trajectory sample, from a 10 s ephemeris grid.

    The Sun moves under 0.05 deg in 10 s, so linear interpolation of its
    unit vector is exact to about 1e-7 rad, far below any margin tested.
    """
    t_rel = (times - times[0]).sec
    grid = np.arange(0.0, t_rel[-1] + 10.0, 10.0)
    sun_az, sun_el = coords.get_sun_altaz(times[0] + TimeDelta(grid, format="sec"))
    az, el = np.radians(np.asarray(sun_az)), np.radians(np.asarray(sun_el))
    xyz = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    out = np.stack([np.interp(t_rel, grid, xyz[i]) for i in range(3)])
    return out / np.linalg.norm(out, axis=0)


@pytest.mark.slow
def test_five_patch_night_ce_science_clears_the_zone(site, coordinates):
    """Every rebuilt constant-elevation science sample clears the radius.

    A 24-hour night with the Sun up for half of it: five 40 x 10 deg fields
    pinned at 45-55 deg elevation, whose corridors run tens of degrees wider
    than the fields.
    """
    spec = [
        ("W1", 24.0, -32.0, 50.0),
        ("W2", 150.0, 2.2, 45.0),
        ("W3", 60.0, -50.0, 50.0),
        ("W4", 210.0, -30.0, 55.0),
        ("W5", 330.0, -40.0, 50.0),
    ]
    patches = [
        ObservingPatch(
            name=name,
            ra_center=ra,
            dec_center=dec,
            width=40.0,
            height=10.0,
            scan_type="constant_el",
            velocity=1.0,
            elevation=el,
        )
        for name, ra, dec, el in spec
    ]
    timeline = generate_timeline(
        patches=patches,
        site=site,
        start_time="2026-12-15T09:00:00",
        end_time="2026-12-16T09:00:00",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pairs = schedule_to_trajectories(timeline)
    assert len(pairs) >= 5  # the night still schedules constant-elevation science

    radius = site.sun_avoidance.exclusion_radius
    for block, scan in pairs:
        traj = scan.trajectory
        times = traj.start_time + TimeDelta(traj.times, format="sec")
        sun = _sun_unit_vectors(coordinates, times)
        az, el = np.radians(traj.az), np.radians(traj.el)
        pose = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        sep = np.degrees(np.arccos(np.clip(np.sum(pose * sun, axis=0), -1.0, 1.0)))
        assert sep.min() > radius, (block.patch_name, block.t_start.isot, float(sep.min()))
