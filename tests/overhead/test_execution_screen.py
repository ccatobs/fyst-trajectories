"""The simulator books no scan the booking screen refuses.

PCS refuses to POST a scan with a sample inside the Sun exclusion radius,
whose axis speed or acceleration exceeds the Go TCS command layer's limits,
or with two samples closer than the ``/path`` minimum interval;
:func:`~fyst_trajectories.overhead._execution_screen._execution_refusal`
asks the same questions of a built trajectory, with its own Sun arithmetic,
and asks an injected Sun model too when the night is given one, which PCS
does not. The scheduler books no subscan, planet pass or slew onto a visit
it refuses. The duration clip of a pong or daisy keeps the whole pattern,
not only its centre, outside the Sun zone, so the screen is a backstop there.

The guards at the end hold every block of every shared night to those limits,
computed here from the rebuilt arrays with PCS's own formulas.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest
from _sun_stubs import fake_sun_model
from astropy import units as u
from astropy.time import Time, TimeDelta

from fyst_trajectories import (
    GO_TCS_AZ_MAX_ACCELERATION,
    GO_TCS_AZ_MAX_VELOCITY,
    GO_TCS_EL_MAX_ACCELERATION,
    GO_TCS_EL_MAX_VELOCITY,
    Coordinates,
    get_fyst_site,
    to_path_format,
)
from fyst_trajectories.exceptions import (
    AccelerationLimitWarning,
    PointingWarning,
    VelocityLimitWarning,
)
from fyst_trajectories.overhead import (
    BlockType,
    DeferralReason,
    ObservingPatch,
    generate_timeline,
    schedule_to_trajectories,
)
from fyst_trajectories.overhead._execution_screen import _execution_refusal
from fyst_trajectories.overhead.scheduler.helpers import _time_until_sun_unsafe
from fyst_trajectories.planning import FieldRegion, plan_pong_scan
from fyst_trajectories.planning._sun_safety import _trajectory_sun_separation
from fyst_trajectories.trajectory import Trajectory, TrajectoryMetadata

from ._nights import NIGHTS, night, rebuilt

T_DAY = Time("2026-11-15T16:00:00", scale="utc")  # Sun up at FYST

_SUN_SCAN = str(DeferralReason.SUN_SCAN)
_AXIS = str(DeferralReason.AXIS_DYNAMICS)

#: A site with no Sun check, for the axis tests.
_NO_SUN = get_fyst_site(sun_avoidance_enabled=False)

#: The factor by which a quintic turnaround's peak acceleration exceeds its
#: nominal ``az_accel``.
_QUINTIC_PEAK = 1.5


def _trajectory(times, az, el, *, az_vel=None, el_vel=None, start=T_DAY, metadata=None):
    times = np.asarray(times, dtype=float)
    az = np.asarray(az, dtype=float)
    el = np.asarray(el, dtype=float)
    return Trajectory(
        times=times,
        az=az,
        el=el,
        az_vel=np.zeros_like(times) if az_vel is None else np.asarray(az_vel, dtype=float),
        el_vel=np.zeros_like(times) if el_vel is None else np.asarray(el_vel, dtype=float),
        start_time=start,
        metadata=metadata,
    )


def _ce(az_speed, az_accel, *, site_az=180.0, el=45.0):
    """Return a short constant-elevation leg recording its speed and acceleration."""
    times = np.arange(0.0, 2.0, 0.5)
    meta = TrajectoryMetadata(
        pattern_type="constant_el",
        pattern_params={
            "az_start": site_az,
            "az_stop": site_az + 1.0,
            "elevation": el,
            "az_speed": az_speed,
            "az_accel": az_accel,
        },
    )
    return _trajectory(times, site_az + 0.1 * times, np.full(times.size, el), metadata=meta)


# ---------------------------------------------------------------------------
# The screen: axis limits, on PCS's side of each tie (strictly above refuses)
# ---------------------------------------------------------------------------


def _three_sides(value):
    """``value`` one step below, on and one step above, with the verdict PCS gives."""
    return [
        (np.nextafter(value, -np.inf), None),
        (value, None),
        (np.nextafter(value, np.inf), DeferralReason.AXIS_DYNAMICS),
    ]


@pytest.mark.parametrize("scan_type", ["pong", "daisy", "source_ces"])
@pytest.mark.parametrize("axis", ["az", "el"])
@pytest.mark.parametrize("side", [0, 1, 2], ids=["below", "on", "above"])
def test_speed_from_the_positions_ties_on_pcs_side(scan_type, axis, side):
    """A gradient speed at the Go TCS ceiling passes; one step above it is refused."""
    ceiling = GO_TCS_AZ_MAX_VELOCITY if axis == "az" else GO_TCS_EL_MAX_VELOCITY
    rate, expected = _three_sides(ceiling)[side]
    times = np.array([0.0, 1.0, 2.0])
    moving = rate * times
    still = np.full(times.size, 40.0)
    az, el = (moving, still) if axis == "az" else (still, moving)
    # The construction itself: the gradient lands on the value under test.
    assert np.gradient(moving, times).max() == rate
    traj = _trajectory(times, az, el)
    assert _execution_refusal(traj, scan_type, {}, _NO_SUN, None) is expected


@pytest.mark.parametrize("axis", ["az", "el"])
@pytest.mark.parametrize("side", [0, 1, 2], ids=["below", "on", "above"])
def test_speed_from_the_velocity_column_ties_on_pcs_side(axis, side):
    """The recorded speed column counts as well as the positions' gradient."""
    ceiling = GO_TCS_AZ_MAX_VELOCITY if axis == "az" else GO_TCS_EL_MAX_VELOCITY
    rate, expected = _three_sides(ceiling)[side]
    times = np.arange(0.0, 2.0, 0.5)
    column = np.full(times.size, -rate)
    flat = np.full(times.size, 40.0)
    kwargs = {"az_vel": column} if axis == "az" else {"el_vel": column}
    traj = _trajectory(times, flat, flat, **kwargs)
    assert _execution_refusal(traj, "pong", {}, _NO_SUN, None) is expected


@pytest.mark.parametrize("axis", ["az", "el"])
@pytest.mark.parametrize("side", [0, 1, 2], ids=["below", "on", "above"])
def test_acceleration_ties_on_pcs_side(axis, side):
    """The second gradient of the positions at the ceiling passes; above it is refused.

    On a grid ``h`` apart, positions ``(0, 0, 0, 2 a h^2)`` have a second
    gradient peaking at ``a``, exactly for ``h`` a power of two.
    """
    ceiling = GO_TCS_AZ_MAX_ACCELERATION if axis == "az" else GO_TCS_EL_MAX_ACCELERATION
    target, expected = _three_sides(ceiling)[side]
    step = 0.125
    times = np.arange(4) * step
    moving = np.array([0.0, 0.0, 0.0, 2.0 * target * step**2])
    still = np.full(times.size, 40.0)
    assert np.abs(np.gradient(np.gradient(moving, times), times)).max() == target
    az, el = (moving, still) if axis == "az" else (still, moving)
    traj = _trajectory(times, az, el)
    assert _execution_refusal(traj, "daisy", {}, _NO_SUN, None) is expected


def test_acceleration_needs_four_samples():
    """Three samples give a speed but no acceleration, as PCS reads them."""
    step = 2.0**-4  # at or above the Go TCS minimum sample interval
    times = np.arange(3) * step
    az = np.array([0.0, 0.0, 16.0 * step**2])  # a second gradient of 8 deg/s^2, 1 deg/s at most
    traj = _trajectory(times, az, np.full(3, 40.0))
    assert _execution_refusal(traj, "pong", {}, _NO_SUN, None) is None


def test_one_sample_has_no_dynamics():
    traj = _trajectory([0.0], [30.0], [40.0])
    assert _execution_refusal(traj, "pong", {}, _NO_SUN, None) is None


def test_azimuth_through_north_is_unwrapped():
    """A sweep across 0/360 deg moves at its own speed, not a turn per sample."""
    times = np.arange(0.0, 2.0, 0.5)
    az = (359.5 + 1.0 * times) % 360.0
    traj = _trajectory(times, az, np.full(times.size, 40.0))
    assert _execution_refusal(traj, "pong", {}, _NO_SUN, None) is None


@pytest.mark.parametrize("side", [0, 1, 2], ids=["below", "on", "above"])
def test_constant_elevation_speed_ties_on_pcs_side(side):
    """A constant-elevation scan is refused when its recorded speed exceeds the ceiling."""
    speed, expected = _three_sides(GO_TCS_AZ_MAX_VELOCITY)[side]
    traj = _ce(speed, 1.0)
    assert _execution_refusal(traj, "constant_el", {}, _NO_SUN, None) is expected


@pytest.mark.parametrize("side", [0, 1, 2], ids=["below", "on", "above"])
def test_constant_elevation_turnaround_ties_on_pcs_side(side):
    """A turnaround peaks at 1.5 times ``az_accel``, refused only above the ceiling."""
    accel, expected = _three_sides(GO_TCS_AZ_MAX_ACCELERATION / _QUINTIC_PEAK)[side]
    assert (_QUINTIC_PEAK * accel > GO_TCS_AZ_MAX_ACCELERATION) == (side == 2)
    traj = _ce(1.0, accel)
    assert _execution_refusal(traj, "constant_el", {}, _NO_SUN, None) is expected


@pytest.mark.parametrize("side", [0, 1, 2], ids=["below", "on", "above"])
def test_source_ces_turnaround_ties_on_pcs_side(side):
    """A planet pass's turnaround is judged on the ``az_accel`` it records."""
    accel, expected = _three_sides(GO_TCS_AZ_MAX_ACCELERATION / _QUINTIC_PEAK)[side]
    times = np.arange(0.0, 2.0, 0.5)
    traj = _trajectory(times, 30.0 + 0.5 * times, np.full(times.size, 40.0))
    params = {"az_accel": accel}
    assert _execution_refusal(traj, "source_ces", params, _NO_SUN, None) is expected


def test_source_ces_without_az_accel_reads_the_planner_default():
    """No recorded ``az_accel`` is the planner's 1.0 deg/s^2, inside the ceiling."""
    times = np.arange(0.0, 2.0, 0.5)
    traj = _trajectory(times, 30.0 + 0.5 * times, np.full(times.size, 40.0))
    assert _execution_refusal(traj, "source_ces", {}, _NO_SUN, None) is None


def test_unknown_scan_type_raises():
    traj = _trajectory([0.0, 1.0], [30.0, 30.5], [40.0, 40.0])
    with pytest.raises(ValueError, match="scan_type"):
        _execution_refusal(traj, "raster", {}, _NO_SUN, None)


# ---------------------------------------------------------------------------
# The screen: the Sun
# ---------------------------------------------------------------------------


def _parked(n=5, step=30.0, az=180.0, el=45.0):
    """Return a trajectory holding one pose for ``(n - 1) * step`` s from ``T_DAY``."""
    times = np.arange(n) * step
    return _trajectory(times, np.full(n, az), np.full(n, el))


@pytest.mark.parametrize("side", [0, 1, 2], ids=["below", "on", "above"])
def test_a_sample_at_the_exclusion_radius_is_inside_the_zone(side):
    """At the screen's own separation, a sample at the radius is inside and refused.

    The tie is the screen's: PCS computes the separation with its own
    arithmetic, which differs by up to about 1e-5 deg (the next test).
    """
    traj = _parked()
    closest = float(_trajectory_sun_separation(get_fyst_site(), traj).min())
    radius = [np.nextafter(closest, -np.inf), closest, np.nextafter(closest, np.inf)][side]
    site = get_fyst_site(sun_exclusion_radius=radius, sun_warning_radius=radius + 5.0)
    expected = None if side == 0 else DeferralReason.SUN_SCAN
    assert _execution_refusal(traj, "pong", {}, site, None) is expected


def _parked_hour():
    times = np.arange(36000) * 0.1
    start = Time("2026-12-15T16:00:00", scale="utc")
    return _trajectory(times, np.full(times.size, 180.0), np.full(times.size, 45.0), start=start)


def _daytime_pong():
    field = FieldRegion(ra_center=200.0, dec_center=-20.0, width=6.0, height=6.0)
    with warnings.catch_warnings():
        # Near transit the pattern exceeds the site's advisory azimuth limits;
        # only its Sun separation is measured here.
        warnings.filterwarnings("ignore", message="Trajectory azimuth", category=PointingWarning)
        warnings.filterwarnings("ignore", message="High elevation", category=PointingWarning)
        block = plan_pong_scan(
            field,
            velocity=0.5,
            spacing=0.1,
            num_terms=4,
            site=get_fyst_site(),
            start_time=Time("2026-12-15T13:00:00", scale="utc"),
        )
    return block.trajectory


@pytest.mark.parametrize("build", [_parked_hour, _daytime_pong], ids=["parked_hour", "pong"])
def test_the_screen_and_pcs_measure_the_sun_to_within_1e_5_deg(build):
    """The two separations agree to about 1e-5 deg, and are not the same numbers.

    An hour parked at one pose, and a 6 x 6 deg pong by day; a sample this
    close to the radius can be judged on either side by the two. PCS's
    arithmetic is written out in ``_pcs_sun_separation`` below.
    """
    traj = build()
    site = get_fyst_site()
    band = np.abs(_trajectory_sun_separation(site, traj) - _pcs_sun_separation(traj, site)).max()
    assert 0.0 < band < 2e-5


def test_a_trajectory_clear_of_the_sun_and_inside_the_limits_passes():
    traj = _parked()
    closest = float(_trajectory_sun_separation(get_fyst_site(), traj).min())
    site = get_fyst_site(sun_exclusion_radius=closest - 1.0, sun_warning_radius=closest)
    assert _execution_refusal(traj, "daisy", {}, site, fake_sun_model(True)) is None


def test_an_injected_model_refuses_where_the_radius_does_not():
    traj = _parked()
    closest = float(_trajectory_sun_separation(get_fyst_site(), traj).min())
    site = get_fyst_site(sun_exclusion_radius=closest - 1.0, sun_warning_radius=closest)
    model = fake_sun_model(False)
    assert _execution_refusal(traj, "pong", {}, site, model) is DeferralReason.SUN_SCAN


def test_the_radius_refuses_where_an_injected_model_does_not():
    traj = _parked()
    closest = float(_trajectory_sun_separation(get_fyst_site(), traj).min())
    site = get_fyst_site(sun_exclusion_radius=closest + 1.0, sun_warning_radius=closest + 5.0)
    model = fake_sun_model(True)
    assert _execution_refusal(traj, "pong", {}, site, model) is DeferralReason.SUN_SCAN


def test_disabled_sun_avoidance_skips_both_sun_checks():
    model = fake_sun_model(False)
    assert _execution_refusal(_parked(), "pong", {}, _NO_SUN, model) is None
    assert model.batch_calls == 0 and model.scalar_calls == 0


def test_an_injected_model_is_asked_about_strided_samples_and_the_last():
    """At most 600 strided samples plus the last are put to an injected model."""
    n = 1301
    traj = _parked(n=n, step=0.1)
    asked = []

    def unsafe_at_the_end(az, el, times):
        asked.append(np.atleast_1d((times - T_DAY).sec))
        return np.atleast_1d((times - T_DAY).sec) < 129.95

    site = get_fyst_site(sun_exclusion_radius=1.0, sun_warning_radius=2.0)
    model = fake_sun_model(unsafe_at_the_end)
    assert _execution_refusal(traj, "pong", {}, site, model) is DeferralReason.SUN_SCAN
    offsets = np.concatenate(asked)
    assert offsets.size <= 601
    assert offsets[-1] == pytest.approx(130.0, abs=1e-6)
    assert offsets[-2] < 129.95
    assert offsets[0] == pytest.approx(0.0, abs=1e-6)


def test_the_sun_is_checked_before_the_axes():
    """A scan both inside the zone and too fast is refused for the Sun."""
    times = np.arange(0.0, 2.0, 0.5)
    traj = _trajectory(times, 30.0 + 4.0 * times, np.full(times.size, 40.0))
    site = get_fyst_site(sun_exclusion_radius=179.0, sun_warning_radius=179.5)
    assert _execution_refusal(traj, "pong", {}, site, None) is DeferralReason.SUN_SCAN


# ---------------------------------------------------------------------------
# The sample interval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("second", "refused"), [(0.0499, True), (0.05, False)], ids=["under", "at"]
)
def test_a_sample_interval_under_the_go_tcs_minimum_is_refused(second, refused):
    """Two samples closer than the ``/path`` minimum refuse the scan; the minimum passes."""
    traj = _trajectory([0.0, second, 0.1], [180.0, 180.0, 180.0], [45.0, 45.0, 45.0])
    reason = _execution_refusal(traj, "pong", {}, _NO_SUN, None)
    assert reason is (DeferralReason.UNPLANNABLE if refused else None)


@pytest.mark.parametrize("duration", [1.0, 7.3, 60.0, 600.0, 1577.0])
@pytest.mark.parametrize("step", [0.05, 0.0500001, 0.051])
def test_the_screen_judges_the_interval_as_to_path_format_does(duration, step):
    """A grid built at the minimum can fall below it by round-off; both judge it the same."""
    n = round(duration / step) + 1
    times = np.linspace(0.0, duration, n)
    traj = _trajectory(times, np.full(n, 180.0), np.full(n, 45.0))
    try:
        to_path_format(traj)
        serialised = True
    except ValueError:
        serialised = False
    refused = _execution_refusal(traj, "pong", {}, _NO_SUN, None) is DeferralReason.UNPLANNABLE
    assert refused is not serialised


def _fine(scan_type, timestep):
    params = {"timestep": timestep}
    if scan_type == "constant_el":
        return ObservingPatch(
            name="Fine",
            ra_center=24.0,
            dec_center=-32.0,
            width=40.0,
            height=10.0,
            scan_type="constant_el",
            velocity=1.0,
            elevation=50.0,
            scan_params=params,
        )
    return ObservingPatch(
        name="Fine",
        ra_center=140.0,
        dec_center=-23.0,
        width=4.0,
        height=4.0,
        scan_type=scan_type,
        velocity=0.5,
        scan_params=params,
    )


@pytest.mark.parametrize("scan_type", ["pong", "daisy", "constant_el"])
def test_a_patch_sampled_under_the_go_tcs_minimum_is_refused_when_the_night_is_built(scan_type):
    """No time of night changes the sample interval, so the patch is refused at once."""
    with pytest.raises(ValueError, match="sample interval is under the Go TCS") as err:
        generate_timeline(
            patches=[_fine(scan_type, 0.02)],
            site=get_fyst_site(),
            start_time=Time("2026-06-15T05:00:00", scale="utc"),
            end_time=Time("2026-06-15T06:00:00", scale="utc"),
        )
    assert "under the Go TCS /path minimum of 0.05 s: Fine (timestep 0.02 s)" in str(err.value)


#: The nights of the patches above: a pong and a daisy rising to the zenith
#: before dawn on 2026-03-01, and the constant-elevation field's rising pass
#: on 2026-06-15.
_FINE_NIGHTS = {
    "pong": ("2026-03-01T00:30:00", "2026-03-01T01:35:00"),
    "daisy": ("2026-03-01T00:30:00", "2026-03-01T01:35:00"),
    "constant_el": ("2026-06-15T05:00:00", "2026-06-15T09:00:00"),
}


@pytest.mark.filterwarnings(
    "ignore:High elevation reduces on-sky azimuth speed:"
    "fyst_trajectories.exceptions.PointingWarning"
)
@pytest.mark.parametrize(
    ("scan_type", "booked"),
    [("pong", False), ("daisy", False), ("constant_el", True)],
)
def test_at_the_go_tcs_minimum_a_scan_is_booked_only_where_it_serialises(scan_type, booked):
    """At exactly 0.05 s the tie falls where ``to_path_format`` puts each scan.

    A pong's and a daisy's samples at 0.05 s fall a few 1e-13 s under the
    minimum by round-off, so the screen refuses them (``unplannable``) as
    ``to_path_format`` would, and the slew phase, which screens the same
    shortest subscan before it books a slew, never slews to them; the
    constant-elevation field's pass is spread over a whole number of samples
    a little over 0.05 s apart, and is booked.
    """
    start, end = _FINE_NIGHTS[scan_type]
    timeline = generate_timeline(
        patches=[_fine(scan_type, 0.05)],
        site=get_fyst_site(),
        start_time=Time(start, scale="utc"),
        end_time=Time(end, scale="utc"),
    )
    pairs = list(schedule_to_trajectories(timeline))
    assert bool(pairs) is booked
    for _, scan in pairs:
        to_path_format(scan.trajectory)
    _assert_no_slew_onto_a_refusal(timeline.blocks)
    if not booked:
        slews = [b.t_start.isot for b in timeline.blocks if b.patch_name == "slew_to_Fine"]
        assert slews == []
        assert str(DeferralReason.UNPLANNABLE) in _reasons(timeline)


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------


def test_the_refusals_are_deferral_reasons():
    assert str(DeferralReason.SUN_SCAN) == "sun_scan"
    assert str(DeferralReason.AXIS_DYNAMICS) == "axis_dynamics"
    assert DeferralReason("sun_scan") is DeferralReason.SUN_SCAN
    assert DeferralReason("axis_dynamics") is DeferralReason.AXIS_DYNAMICS


# ---------------------------------------------------------------------------
# The pong and daisy Sun clip covers the pattern's reach
# ---------------------------------------------------------------------------


def _field_at(coordinates, az=180.0, el=45.0, t=T_DAY):
    ra, dec = coordinates.altaz_to_radec(az, el, t)
    sun_az, sun_el = coordinates.get_sun_altaz(t)
    return float(ra), float(dec), float(coordinates.angular_separation(az, el, sun_az, sun_el))


def _closing_threshold(coordinates, ra, dec, span=600.0):
    """Return a separation the field centre crosses within ``span`` s of ``T_DAY``."""
    times = T_DAY + TimeDelta([0.0, span], format="sec")
    az, el = coordinates.radec_to_altaz(np.full(2, ra), np.full(2, dec), times)
    sun_az, sun_el = coordinates.get_sun_altaz(times)
    sep = coordinates.angular_separation(az, el, sun_az, sun_el)
    assert sep[1] < sep[0]  # the centre closes on the Sun
    return float(0.5 * (sep[0] + sep[1]))


def test_the_reach_widens_the_scalar_radius(coordinates):
    """With ``reach`` the scalar clip stops where ``radius + reach`` would."""
    ra, dec, _ = _field_at(coordinates)
    reach = 2.5
    radius = _closing_threshold(coordinates, ra, dec) - reach
    widened = _time_until_sun_unsafe(ra, dec, T_DAY, 600.0, coordinates, radius, reach=reach)
    reference = _time_until_sun_unsafe(ra, dec, T_DAY, 600.0, coordinates, radius + reach)
    plain = _time_until_sun_unsafe(ra, dec, T_DAY, 600.0, coordinates, radius)
    assert 0.0 < widened < 600.0
    assert widened == reference
    assert plain == 600.0


def test_the_reach_tie_is_inside(coordinates):
    """A field whose reach exactly touches the radius at the start has no time left.

    The separation lies in [64, 128) deg, where subtracting and adding 0.5
    returns it exactly.
    """
    ra, dec, sep = _field_at(coordinates, el=30.0)
    assert 64.0 <= sep < 128.0
    radius = sep - 0.5
    assert radius + 0.5 == sep
    left = _time_until_sun_unsafe(ra, dec, T_DAY, 600.0, coordinates, radius, reach=0.5)
    assert left == 0.0


def test_a_zoned_model_is_widened_by_the_reach(coordinates):
    """A zoned model clips where the centre's separation meets ``threshold + reach``."""
    ra, dec, _ = _field_at(coordinates)
    reach = 2.5
    required = _closing_threshold(coordinates, ra, dec) - reach
    zoned = fake_sun_model(True, threshold=required)
    clipped = _time_until_sun_unsafe(
        ra, dec, T_DAY, 600.0, coordinates, 1.0, sun_safe=zoned, reach=reach
    )
    scalar = _time_until_sun_unsafe(ra, dec, T_DAY, 600.0, coordinates, required + reach)
    assert 0.0 < scalar < 600.0
    assert clipped == pytest.approx(scalar, abs=0.2)
    assert clipped <= scalar + 0.01


def test_a_bare_model_is_asked_about_the_centre_and_eight_points_on_the_reach(coordinates):
    """Without a threshold the model is asked about the centre and eight points at ``reach``."""
    ra, dec, _ = _field_at(coordinates)
    reach = 2.5
    asked = []

    def record(az, el, times):
        asked.append((np.atleast_1d(az).copy(), np.atleast_1d(el).copy(), times))
        return np.ones(np.atleast_1d(az).shape, dtype=bool)

    model = fake_sun_model(record)
    out = _time_until_sun_unsafe(
        ra, dec, T_DAY, 600.0, coordinates, 45.0, sun_safe=model, reach=reach
    )
    assert out == 600.0
    az, el, times = asked[0]
    n_t = az.size // 9
    assert az.size == 9 * n_t and n_t >= 2
    az = az.reshape(n_t, 9)
    el = el.reshape(n_t, 9)
    for row_az, row_el in zip(az, el, strict=True):
        distance = coordinates.angular_separation(row_az[1:], row_el[1:], row_az[0], row_el[0])
        np.testing.assert_allclose(distance, reach, atol=1e-9)
        bearing = np.degrees(
            np.arctan2(
                np.sin(np.radians(row_az[1:] - row_az[0])) * np.cos(np.radians(row_el[1:])),
                np.cos(np.radians(row_el[0])) * np.sin(np.radians(row_el[1:]))
                - np.sin(np.radians(row_el[0]))
                * np.cos(np.radians(row_el[1:]))
                * np.cos(np.radians(row_az[1:] - row_az[0])),
            )
        )
        np.testing.assert_allclose(np.diff(np.unwrap(bearing, period=360.0)), 45.0, atol=1e-6)


def test_a_bare_model_refusing_on_the_reach_clips(coordinates):
    """A zone the pattern's reach enters, but the centre does not, clips the visit."""
    ra, dec, _ = _field_at(coordinates)
    cut = 290.0  # between two 60 s grid samples

    def ring_after_cut(az, el, times):
        late = np.atleast_1d((times - T_DAY).sec) >= cut
        centre_az, centre_el = coordinates.radec_to_altaz(ra, dec, times)
        off = coordinates.angular_separation(az, el, centre_az, centre_el)
        return ~(late & (np.atleast_1d(off) > 1.0))

    model = fake_sun_model(ring_after_cut)
    clipped = _time_until_sun_unsafe(
        ra, dec, T_DAY, 600.0, coordinates, 45.0, sun_safe=model, reach=2.0
    )
    unclipped = _time_until_sun_unsafe(ra, dec, T_DAY, 600.0, coordinates, 45.0, sun_safe=model)
    assert unclipped == 600.0
    assert clipped == pytest.approx(cut, abs=0.2)
    assert clipped < cut


# ---------------------------------------------------------------------------
# Nights
# ---------------------------------------------------------------------------


def _reasons(timeline):
    return [b.metadata.get("reason") for b in timeline.blocks if b.block_type == BlockType.IDLE]


def test_a_pong_too_fast_for_the_go_tcs_idles_without_a_slew():
    """Near the zenith the pong's azimuth speed exceeds 3 deg/s: it idles, unslewed.

    Before its transit four periods of the pong fit its window and PCS would
    run two of them; it is booked those two and then idles for the axes.
    Every slew onto it is followed by its scan.
    """
    timeline = night("zenith")
    blocks = timeline.blocks
    assert _AXIS in _reasons(timeline)
    science = [b for b in blocks if b.block_type == BlockType.SCIENCE]
    assert [b.patch_name for b in science] == ["ZenP"]
    assert science[0].metadata["scan_params"]["n_cycles"] == 2
    for i, block in enumerate(blocks):
        if block.patch_name == "slew_to_ZenP":
            after = next(b for b in blocks[i + 1 :] if b.block_type == BlockType.SCIENCE)
            assert after.patch_name == "ZenP"


def test_a_daisy_too_fast_at_full_length_is_booked_shorter():
    """Before its transit the daisy runs at a shorter length, and every slew onto it scans.

    The full-length daisy reaches past the Go TCS limits as its field nears
    the zenith; a shorter one from the same start does not, and is booked.
    """
    timeline = night("zenith_daisy")
    blocks = timeline.blocks
    science = [b for b in blocks if b.block_type == BlockType.SCIENCE]
    assert science, "no daisy was booked"
    assert all(b.patch_name == "ZenD" for b in science)
    for i, block in enumerate(blocks):
        if block.patch_name == "slew_to_ZenD":
            after = blocks[i + 1 :]
            first = next(b for b in after if b.block_type != BlockType.CALIBRATION)
            assert first.block_type == BlockType.SCIENCE, (block.t_start.isot, first.metadata)


#: The idle reasons that say the visit just slewed onto cannot run.
_REFUSED_VISIT = {
    _SUN_SCAN,
    _AXIS,
    str(DeferralReason.SUN_POINT),
    str(DeferralReason.UNPLANNABLE),
}


def _assert_no_slew_onto_a_refusal(blocks):
    """Assert that no slew onto a patch is followed directly by an idle refusing the visit."""
    for before, after in itertools.pairwise(blocks):
        if before.block_type != BlockType.SLEW or not before.patch_name.startswith("slew_to_"):
            continue
        reason = after.metadata.get("reason") if after.block_type == BlockType.IDLE else None
        assert reason not in _REFUSED_VISIT, (before.patch_name, before.t_start.isot, reason)


@pytest.mark.parametrize("name", sorted(NIGHTS))
def test_no_slew_onto_a_patch_is_followed_by_its_refusal(name):
    """A visit is slewed to only when its first subscan can be booked.

    The slew phase screens the first subscan before it books the slew, and
    the science phase books at least that subscan, so no slew onto a patch
    is followed directly by an idle saying the visit cannot run. A slew onto
    a planet is followed by its passes.
    """
    _assert_no_slew_onto_a_refusal(night(name).blocks)


#: A 10 x 10 deg pong whose centre stays outside the 45 deg Sun zone while
#: its corners reach inside it.
_SUN_PONG = ObservingPatch(
    name="SunP",
    ra_center=136.0,
    dec_center=23.0,
    width=10.0,
    height=10.0,
    scan_type="pong",
    velocity=1.0,
)


def _built(patches, start, end, **kwargs):
    timeline = generate_timeline(
        patches=patches, site=get_fyst_site(), start_time=start, end_time=end, **kwargs
    )
    # The site's advisory elevation limits are below the Go TCS limits, and
    # the samples' Sun separations are what the test asserts.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Trajectory elevation velocity", category=VelocityLimitWarning
        )
        warnings.filterwarnings(
            "ignore", message="Trajectory elevation acceleration", category=AccelerationLimitWarning
        )
        warnings.filterwarnings("ignore", message="EXCLUSION ZONE: ", category=PointingWarning)
        pairs = schedule_to_trajectories(timeline, science_only=False)
    return timeline, pairs


def test_the_pong_clip_keeps_the_whole_pattern_out_of_the_sun():
    """A pong whose corners would enter the zone is not booked into it."""
    timeline, pairs = _built([_SUN_PONG], "2026-06-15T18:00:00", "2026-06-15T19:15:00")
    site = timeline.site
    radius = site.sun_avoidance.exclusion_radius
    for _, scan_block in pairs:
        assert _trajectory_sun_separation(site, scan_block.trajectory).min() > radius
    # The clip stopped it, not the screen behind it.
    assert _SUN_SCAN not in _reasons(timeline)


class _RingModel:
    """Unsafe in the half of a ring 0.8 to 1.2 deg about a field's centre above it.

    The pong crosses the half ring; its centre and the eight points on its
    reach the duration clip asks about stay clear of it, and so does the
    slew onto the centre, which comes from below.
    """

    def __init__(self, ra, dec, coords):
        self._ra, self._dec, self._coords = ra, dec, coords

    def __call__(self, az, el, time):
        return bool(self.batch(np.atleast_1d(az), np.atleast_1d(el), time)[0])

    def batch(self, az, el, times):
        az = np.atleast_1d(np.asarray(az, dtype=float))
        el = np.atleast_1d(np.asarray(el, dtype=float))
        times = times if times.shape == az.shape else Time(np.full(az.shape, times.jd), format="jd")
        c_az, c_el = self._coords.radec_to_altaz(
            np.full(az.shape, self._ra), np.full(az.shape, self._dec), times
        )
        off = np.asarray(self._coords.angular_separation(az, el, c_az, c_el), dtype=float)
        return ~((off > 0.8) & (off < 1.2) & (el > c_el))


def test_a_pong_the_clip_passes_and_the_screen_refuses_idles_for_the_sun():
    """A Sun model the clip's nine points miss refuses the pong at the screen."""
    site = get_fyst_site()
    patch = ObservingPatch(
        name="RingP",
        ra_center=150.0,
        dec_center=2.2,
        width=4.0,
        height=4.0,
        scan_type="pong",
        velocity=0.5,
    )
    model = _RingModel(patch.ra_center, patch.dec_center, Coordinates(site))
    timeline = generate_timeline(
        patches=[patch],
        site=site,
        start_time="2026-12-15T09:30:00",
        end_time="2026-12-15T10:30:00",
        sun_safe=model,
    )
    names = [b.patch_name for b in timeline.blocks]
    assert "RingP" not in names
    assert "slew_to_RingP" not in names
    assert _SUN_SCAN in _reasons(timeline)


# ---------------------------------------------------------------------------
# Guards: every booked block of every shared night runs on PCS
# ---------------------------------------------------------------------------


def _pcs_sun_separation(trajectory, site):
    """Every sample's Sun separation, as PCS computes it before a POST."""
    rel = np.asarray(trajectory.times, dtype=float)
    span = float(rel[-1])
    n_grid = int(np.ceil(span / 60.0)) + 1
    grid = np.linspace(0.0, max(span, 60.0), max(n_grid, 2))
    sun_az, sun_el = Coordinates(site).get_sun_altaz(trajectory.start_time + grid * u.s)
    az_r, el_r = np.radians(sun_az), np.radians(sun_el)
    sun = np.vstack(
        [
            np.interp(rel, grid, comp)
            for comp in (np.cos(el_r) * np.cos(az_r), np.cos(el_r) * np.sin(az_r), np.sin(el_r))
        ]
    )
    sun /= np.linalg.norm(sun, axis=0)
    t_az, t_el = np.radians(trajectory.az), np.radians(trajectory.el)
    pointing = np.vstack([np.cos(t_el) * np.cos(t_az), np.cos(t_el) * np.sin(t_az), np.sin(t_el)])
    return np.degrees(np.arccos(np.clip(np.sum(pointing * sun, axis=0), -1.0, 1.0)))


def _pcs_peaks(trajectory):
    """Peak axis speeds and accelerations, as PCS computes them before a POST."""
    times = np.asarray(trajectory.times, dtype=float)
    az = np.unwrap(np.asarray(trajectory.az, dtype=float), period=360.0)
    el = np.asarray(trajectory.el, dtype=float)
    if times.size < 2:
        return 0.0, 0.0, 0.0, 0.0
    v_az, v_el = np.gradient(az, times), np.gradient(el, times)
    speeds = (
        max(np.abs(v_az).max(), np.abs(trajectory.az_vel).max()),
        max(np.abs(v_el).max(), np.abs(trajectory.el_vel).max()),
    )
    if times.size < 4:
        return (*speeds, 0.0, 0.0)
    return (
        *speeds,
        np.abs(np.gradient(v_az, times)).max(),
        np.abs(np.gradient(v_el, times)).max(),
    )


def _scan_type(block):
    return "source_ces" if block.block_type == BlockType.CALIBRATION else block.scan_type


@pytest.mark.parametrize("name", sorted(NIGHTS))
def test_every_booked_block_is_outside_the_sun_zone(name):
    timeline = night(name)
    site = timeline.site
    radius = site.sun_avoidance.exclusion_radius
    for block, scan_block in rebuilt(name):
        sep = _pcs_sun_separation(scan_block.trajectory, site)
        assert sep.min() > radius, (block.patch_name, block.t_start.isot, float(sep.min()))


@pytest.mark.parametrize("name", sorted(NIGHTS))
def test_every_booked_block_is_inside_the_go_tcs_limits(name):
    limits = (
        GO_TCS_AZ_MAX_VELOCITY,
        GO_TCS_EL_MAX_VELOCITY,
        GO_TCS_AZ_MAX_ACCELERATION,
        GO_TCS_EL_MAX_ACCELERATION,
    )
    for block, scan_block in rebuilt(name):
        peaks = _pcs_peaks(scan_block.trajectory)
        for peak, limit in zip(peaks, limits, strict=True):
            assert peak <= limit, (block.patch_name, block.t_start.isot, peaks)
        if _scan_type(block) == "constant_el":
            az_accel = scan_block.trajectory.metadata.pattern_params["az_accel"]
            assert _QUINTIC_PEAK * az_accel <= GO_TCS_AZ_MAX_ACCELERATION
        elif _scan_type(block) == "source_ces":
            az_accel = block.metadata["scan_params"].get("az_accel", 1.0)
            assert _QUINTIC_PEAK * az_accel <= GO_TCS_AZ_MAX_ACCELERATION


@pytest.mark.parametrize("name", sorted(NIGHTS))
def test_every_booked_block_passes_the_screen(name):
    site = night(name).site
    for block, scan_block in rebuilt(name):
        refusal = _execution_refusal(
            scan_block.trajectory, _scan_type(block), block.metadata["scan_params"], site, None
        )
        assert refusal is None, (block.patch_name, block.t_start.isot, refusal)


def test_the_guards_cover_a_pass_and_every_scan_type():
    """The shared nights book constant-elevation, pong, daisy and planet-pass blocks."""
    kinds = {_scan_type(block) for name in NIGHTS for block, _ in rebuilt(name)}
    assert kinds == {"constant_el", "pong", "daisy", "source_ces"}
