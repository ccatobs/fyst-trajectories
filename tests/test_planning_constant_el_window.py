"""Tests for plan_constant_el_window_scan."""

import math
import re
import warnings
from decimal import Decimal

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import FK5, SkyCoord
from astropy.time import Time

import fyst_trajectories
from fyst_trajectories import get_fyst_site, planning
from fyst_trajectories.exceptions import (
    AzimuthBoundsError,
    ElevationBoundsError,
    OffsetInversionError,
    PointingError,
    PointingWarning,
)
from fyst_trajectories.offsets import InstrumentOffset, apply_detector_offset
from fyst_trajectories.patterns.configs import ConstantElScanConfig
from fyst_trajectories.patterns.constant_el import _ce_az_reach
from fyst_trajectories.patterns.turnarounds import (
    swept_az_envelope,
    turnaround_duration_sec,
    turnaround_overshoot_deg,
)
from fyst_trajectories.planning import (
    ConstantElComputedParams,
    FieldRegion,
    ScanBlock,
    plan_constant_el_scan,
    plan_constant_el_window_scan,
    validate_computed_params,
)
from fyst_trajectories.planning._ce_geometry import (
    _boresight_az_offset,
    _ce_branch_shift,
    _window_side,
)
from fyst_trajectories.primecam import get_primecam_offset
from fyst_trajectories.site import AtmosphericConditions
from fyst_trajectories.trajectory_utils import get_absolute_times

_FIELD = FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7)

# The docstring example: E-CDF-S's rising crossing of el 50 on 2026-03-15 as a
# catalogue azimuth-range row states one, the azimuths at which the field lies
# within 0.2 deg of the elevation, rounded out to 0.1 deg, from the minute
# before the field reaches that band until it leaves it.
_EXAMPLE = dict(
    az_window=(101.0, 111.6),
    elevation=50.0,
    velocity=0.5,
    duration=1680.0,
    start_time=Time("2026-03-15T17:20:00", scale="utc"),
)

# A night instant, the Sun far below the horizon, for the tests whose subject
# is not the Sun.
_NIGHT = Time("2026-03-15T03:00:00", scale="utc")

# The field form's search anchors for E-CDF-S's two crossings of el 50 on the
# night of 2026-11-15/16 (the passes start at 01:16 and 07:08 UTC, both in the
# dark).
_PASS_ANCHORS = {"rising": "2026-11-15T22:00:00", "setting": "2026-11-16T04:00:00"}

# The wide-field survey's source as the catalogue seeds it, the whole RA circle
# between Dec -61 and -18 deg: a band with no RA edges.
_BAND = FieldRegion(
    ra_center=180.0,
    dec_center=-39.5,
    width=360.0 * math.cos(math.radians(-39.5)),
    height=43.0,
)


# The Earth's rotation in RA degrees per solar hour.
_SIDEREAL_DEG_PER_HOUR = 15.0411


def _wide_field(ra_span):
    """Return E-CDF-S widened to ``ra_span`` deg of RA at its central Dec."""
    cos_dec = math.cos(math.radians(_FIELD.dec_center))
    return FieldRegion(
        ra_center=_FIELD.ra_center,
        dec_center=_FIELD.dec_center,
        width=ra_span * cos_dec,
        height=_FIELD.height,
    )


def _wide_pass(block, ra_span):
    """Return when a field of ``ra_span`` deg of RA centred on E-CDF-S starts and ends its pass.

    Read off the field form's E-CDF-S pass, which starts as its leading edge
    crosses: an edge ``x`` deg of RA later crosses ``x / 15.0411`` h later.
    """
    half_ecdfs = 0.5 * _FIELD.width / math.cos(math.radians(_FIELD.dec_center))
    start = block.trajectory.start_time
    return (
        start - (0.5 * ra_span - half_ecdfs) / _SIDEREAL_DEG_PER_HOUR * u.hour,
        start + (0.5 * ra_span + half_ecdfs) / _SIDEREAL_DEG_PER_HOUR * u.hour,
    )


def _field_ending_before(ra_span, hours):
    """Return a field of ``ra_span`` deg of RA whose pass ends ``hours`` before E-CDF-S's starts.

    At E-CDF-S's Dec, with its trailing edge ``hours`` of the Earth's rotation
    in RA before E-CDF-S's leading edge, so on either side it crosses that
    much earlier.
    """
    cos_dec = math.cos(math.radians(_FIELD.dec_center))
    ra_trail = _FIELD.ra_center - 0.5 * _FIELD.width / cos_dec - _SIDEREAL_DEG_PER_HOUR * hours
    return FieldRegion(
        ra_center=(ra_trail - 0.5 * ra_span) % 360.0,
        dec_center=_FIELD.dec_center,
        width=ra_span * cos_dec,
        height=_FIELD.height,
    )


def _pointing_warnings(caught):
    """Return the messages of the PointingWarnings among recorded warnings."""
    return [str(w.message) for w in caught if issubclass(w.category, PointingWarning)]


def _plan(site, field=None, **overrides):
    """Plan a 1680 s window at el 50 on the night of 2026-03-15, with no field by default."""
    kwargs = dict(
        az_window=(101.0, 111.6),
        elevation=50.0,
        velocity=0.5,
        duration=1680.0,
        site=site,
        start_time=_NIGHT,
    )
    kwargs.update(overrides)
    return plan_constant_el_window_scan(field, **kwargs)


def _replay(block, **overrides):
    """Plan the window form over a field-form pass: its window, exact start and crossing time."""
    cp = block.computed_params
    crossing = (Time(cp["end_time_iso"], scale="utc") - Time(cp["start_time_iso"], scale="utc")).sec
    kwargs = dict(
        az_window=(cp["az_start"], cp["az_stop"]),
        elevation=50.0,
        velocity=0.5,
        duration=crossing,
        site=get_fyst_site(),
        start_time=block.trajectory.start_time,
    )
    kwargs.update(overrides)
    return plan_constant_el_window_scan(**kwargs)


@pytest.fixture(scope="module")
def example_block():
    """Return the docstring example's block."""
    return plan_constant_el_window_scan(_FIELD, site=get_fyst_site(), **_EXAMPLE)


@pytest.fixture(scope="module", params=["rising", "setting"])
def field_pass(request):
    """Return a side and E-CDF-S's unpadded crossing of el 50 on it, from the field form."""
    side = request.param
    block = plan_constant_el_scan(
        _FIELD,
        elevation=50.0,
        velocity=0.5,
        site=get_fyst_site(),
        start_time=_PASS_ANCHORS[side],
        rising=side == "rising",
        az_padding=0.0,
    )
    return side, block


class TestPlanConstantElWindowScan:
    """Block shape, the literal start, the window as commanded, and the summary."""

    def test_basic_plan(self, example_block):
        block = example_block
        assert isinstance(block, ScanBlock)
        assert isinstance(block.config, ConstantElScanConfig)
        assert set(block.computed_params) == set(ConstantElComputedParams.__annotations__)
        validate_computed_params(block.computed_params, "constant_el")
        assert block.duration == block.computed_params["duration"]
        np.testing.assert_array_equal(block.trajectory.el, 50.0)

    def test_summary(self, example_block):
        assert example_block.summary == (
            "Constant-El window scan: 5.00 x 6.70 deg field at RA=53.117, Dec=-27.808\n"
            "  Elevation: 50.00 deg, Az window: [101.00, 111.60] deg (throw: 10.60 deg)\n"
            "  Velocity: 0.500 deg/s, Acceleration: 1.000 deg/s^2\n"
            "  Requested: 2026-03-15 17:20:00 to 2026-03-15 17:48:00\n"
            "  Scans: 76, Duration: 1686.2s (28.1min), Trajectory points: 16863"
        )

    def test_the_window_is_commanded_as_given(self, example_block):
        """The window's ends are the legs' ends; only the turnarounds reach past them."""
        cp = example_block.computed_params
        assert (cp["az_start"], cp["az_stop"]) == (101.0, 111.6)
        assert cp["az_throw"] == pytest.approx(10.6)
        assert (example_block.config.az_start, example_block.config.az_stop) == (101.0, 111.6)
        overshoot = turnaround_overshoot_deg(0.5, 1.0)
        assert example_block.trajectory.az.min() == pytest.approx(101.0 - overshoot, abs=1e-4)
        assert example_block.trajectory.az.max() == pytest.approx(111.6 + overshoot, abs=1e-4)

    def test_a_row_with_no_sky_box_plans_with_or_without_its_band(self, site):
        """The wide-field survey's el 50 rising row (az 26 to 154 deg) needs no field."""
        row = dict(az_window=(26.0, 154.0), elevation=50.0, velocity=1.0, duration=1800.0)
        bare = _plan(site, **row)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            banded = _plan(site, _BAND, **row)
        assert _pointing_warnings(caught) == []
        assert np.array_equal(banded.trajectory.az, bare.trajectory.az)
        cp = bare.computed_params
        assert (cp["az_start"], cp["az_stop"]) == (26.0, 154.0)

    def test_the_example_window_is_the_fields_band_crossing(self, coordinates):
        """E-CDF-S lies within 0.2 deg of el 50 in the window, from 17:21 to 17:48 UTC."""
        xi, eta = np.meshgrid(np.linspace(-2.5, 2.5, 11), np.linspace(-3.35, 3.35, 15))
        dec = _FIELD.dec_center + eta.ravel()
        ra = _FIELD.ra_center + xi.ravel() / np.cos(np.radians(dec))
        offsets = np.repeat(np.arange(0.0, _EXAMPLE["duration"] + 1.0, 10.0), ra.size)
        az, el = coordinates.radec_to_altaz(
            np.resize(ra, offsets.size),
            np.resize(dec, offsets.size),
            _EXAMPLE["start_time"] + offsets * u.s,
        )
        in_band = np.abs(np.asarray(el, dtype=float) - 50.0) <= 0.2
        az = np.asarray(az, dtype=float)[in_band]
        assert 60.0 <= offsets[in_band].min() <= 120.0
        assert _EXAMPLE["duration"] - 60.0 <= offsets[in_band].max() < _EXAMPLE["duration"]
        window = (math.floor(az.min() * 10.0) / 10.0, math.ceil(az.max() * 10.0) / 10.0)
        assert window == _EXAMPLE["az_window"]

    def test_the_example_field_rises_through_the_window(self, coordinates):
        """E-CDF-S's centre rises through el 50 inside the example's window during the example."""
        offsets = np.arange(0.0, _EXAMPLE["duration"] + 1.0, 10.0)
        az, el = coordinates.radec_to_altaz(
            np.full(offsets.size, _FIELD.ra_center),
            np.full(offsets.size, _FIELD.dec_center),
            _EXAMPLE["start_time"] + offsets * u.s,
        )
        az, el = np.asarray(az, dtype=float), np.asarray(el, dtype=float)
        (crossing,) = np.flatnonzero((el[:-1] < 50.0) & (el[1:] >= 50.0))
        az_lo, az_hi = _EXAMPLE["az_window"]
        assert az_lo < az[crossing] < az_hi

    def test_the_start_is_literal(self, example_block):
        """The first sample is ``start_time``; the end is the start plus the requested duration."""
        start = _EXAMPLE["start_time"]
        assert abs((example_block.trajectory.start_time - start).sec) < 1e-9
        assert example_block.computed_params["start_time_iso"] == "2026-03-15 17:20:00.000"
        assert example_block.computed_params["end_time_iso"] == "2026-03-15 17:48:00.000"

    def test_a_later_start_sweeps_the_same_window_later(self, site):
        early = _plan(site)
        late = _plan(site, start_time=_NIGHT + 600.0 * u.s)
        assert late.computed_params["n_scans"] == early.computed_params["n_scans"]
        np.testing.assert_array_equal(late.trajectory.times, early.trajectory.times)
        np.testing.assert_array_equal(late.trajectory.az, early.trajectory.az)
        shift = (late.trajectory.start_time - early.trajectory.start_time).sec
        assert shift == pytest.approx(600.0, abs=1e-6)

    def test_the_field_never_shapes_the_trajectory(self, site, example_block):
        bare = plan_constant_el_window_scan(None, site=site, **_EXAMPLE)
        assert bare.computed_params == example_block.computed_params
        assert bare.config == example_block.config
        np.testing.assert_array_equal(bare.trajectory.az, example_block.trajectory.az)
        lines = bare.summary.splitlines()
        assert lines[0] == "Constant-El window scan: no field region"
        assert lines[1:] == example_block.summary.splitlines()[1:]

    def test_a_tt_start_time_records_utc_times(self, site, example_block):
        """The recorded times and the summary are UTC whatever scale ``start_time`` is in."""
        start = _EXAMPLE["start_time"].tt
        block = plan_constant_el_window_scan(_FIELD, site=site, **{**_EXAMPLE, "start_time": start})
        for key in ("start_time_iso", "end_time_iso"):
            assert block.computed_params[key] == example_block.computed_params[key]
        assert block.summary == example_block.summary

    def test_a_string_start_and_a_list_window_plan_as_the_time_and_the_tuple(
        self, site, example_block
    ):
        """An ISO string and a JSON-shaped list plan exactly what a Time and a tuple do."""
        block = plan_constant_el_window_scan(
            _FIELD,
            site=site,
            **{**_EXAMPLE, "start_time": "2026-03-15T17:20:00", "az_window": [101.0, 111.6]},
        )
        assert block.computed_params == example_block.computed_params
        np.testing.assert_array_equal(block.trajectory.az, example_block.trajectory.az)

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            pytest.param("az_window", (Decimal("101.0"), Decimal("111.6")), id="az_window"),
            pytest.param("elevation", Decimal("50.0"), id="elevation"),
            pytest.param("velocity", Decimal("0.5"), id="velocity"),
            pytest.param("duration", Decimal("1680.0"), id="duration"),
            pytest.param("az_accel", Decimal("1.0"), id="az_accel"),
            pytest.param("az_padding", Decimal("0.0"), id="az_padding"),
        ],
    )
    def test_a_decimal_argument_plans_as_its_float(self, site, example_block, name, value):
        block = plan_constant_el_window_scan(_FIELD, site=site, **{**_EXAMPLE, name: value})
        assert block.computed_params == example_block.computed_params
        assert block.config == example_block.config
        np.testing.assert_array_equal(block.trajectory.az, example_block.trajectory.az)

    def test_a_detector_offset_moves_the_boresight(self, site):
        """The offset trajectory is the boresight one with the offset applied."""
        offset = InstrumentOffset(dx=5.0, dy=3.0, name="TestDet")
        bare = _plan(site)
        block = _plan(site, detector_offset=offset)
        expected = apply_detector_offset(bare.trajectory, offset, site=site)
        np.testing.assert_array_equal(block.trajectory.az, expected.az)
        np.testing.assert_array_equal(block.trajectory.el, expected.el)
        assert not np.allclose(block.trajectory.az, bare.trajectory.az)
        assert block.computed_params == bare.computed_params

    def test_the_atmosphere_never_refracts_the_commanded_window(self, site):
        bare = _plan(site)
        refracted = _plan(site, atmosphere=AtmosphericConditions.for_fyst())
        np.testing.assert_array_equal(refracted.trajectory.az, bare.trajectory.az)
        np.testing.assert_array_equal(refracted.trajectory.el, bare.trajectory.el)

    def test_science_mask_excludes_turnarounds(self, example_block):
        """Each of the 75 one-second turnarounds holds 9 samples; its seams are science."""
        science = example_block.trajectory.science_mask
        assert np.sum(~science) == 75 * 9
        assert science[0] and science[-1]

    def test_the_planner_is_public(self):
        assert fyst_trajectories.plan_constant_el_window_scan is plan_constant_el_window_scan
        assert "plan_constant_el_window_scan" in fyst_trajectories.__all__
        assert "plan_constant_el_window_scan" in planning.__all__


class TestWindowQuantisation:
    """The duration rounds to whole legs, by the same rule as the field form's pass."""

    def test_the_duration_rounds_to_whole_legs(self, example_block):
        """10.6 deg legs at 0.5 deg/s take 21.2 s, and each turnaround 2 v / a = 1 s.

        So 1680 s holds (1680 + 1) / 22.2 = 75.7 legs, which rounds to 76 legs
        and 75 turnarounds.
        """
        assert example_block.computed_params["n_scans"] == 76
        assert example_block.duration == pytest.approx(76 * 21.2 + 75 * 1.0)
        assert example_block.trajectory.n_points == 16863

    def test_padding_widens_the_window_on_each_side(self, site):
        """14.6 deg legs take 29.2 s, so (1680 + 1) / 30.2 = 55.7 legs round to 56."""
        block = _plan(site, az_padding=2.0)
        cp = block.computed_params
        assert (cp["az_start"], cp["az_stop"]) == pytest.approx((99.0, 113.6))
        assert cp["n_scans"] == 56
        assert block.duration == pytest.approx(56 * 29.2 + 55 * 1.0)

    def test_the_field_forms_pass_replays_bit_for_bit(self, field_pass):
        """Given a pass's window, start and crossing time, the window form builds that pass.

        Both planners round the crossing to whole legs with one quantiser and
        build from one start, so the leg count, the rounded duration and every
        sample agree.
        """
        _, block = field_pass
        replay = _replay(block)
        assert replay.computed_params["n_scans"] == block.computed_params["n_scans"]
        assert replay.duration == block.duration
        assert replay.config == block.config
        for name in ("times", "az", "el", "az_vel", "el_vel", "scan_flag"):
            np.testing.assert_array_equal(
                getattr(replay.trajectory, name), getattr(block.trajectory, name)
            )


class TestWindowBranchPlacement:
    """A window moves a whole turn only when the azimuths it or the boresight sweeps overrun."""

    @pytest.mark.parametrize(
        ("window", "placed"),
        [
            pytest.param((355.0, 370.0), (-5.0, 10.0), id="across-north-from-above"),
            pytest.param((-5.0, 10.0), (-5.0, 10.0), id="across-north-in-range"),
            pytest.param((-200.0, -170.0), (160.0, 190.0), id="below-the-minimum"),
            pytest.param((-170.0, -150.0), (-170.0, -150.0), id="negative-branch-kept"),
            pytest.param((190.0, 210.0), (190.0, 210.0), id="positive-branch-kept"),
        ],
    )
    def test_placement(self, site, window, placed):
        block = _plan(site, az_window=window, duration=600.0)
        cp = block.computed_params
        assert (cp["az_start"], cp["az_stop"]) == pytest.approx(placed)
        limits = site.telescope_limits.azimuth
        assert limits.min <= block.trajectory.az.min()
        assert block.trajectory.az.max() <= limits.max

    def test_turnarounds_past_a_limit_move_the_window_a_turn(self, site):
        """At 3 deg/s a turnaround overshoots by 5.6 deg, so a window ending at 359.9 moves down.

        The window itself fits on its own branch; its 23 legs reach both
        overshoots, which fit only a turn lower.
        """
        window = (300.0, 359.9)
        assert all(site.telescope_limits.azimuth.is_in_range(az) for az in window)
        block = _plan(site, az_window=window, elevation=40.0, velocity=3.0, duration=600.0)
        cp = block.computed_params
        assert (cp["az_start"], cp["az_stop"]) == pytest.approx((-60.0, -0.1))
        overshoot = turnaround_overshoot_deg(3.0, 1.0)
        assert overshoot == pytest.approx(5.625)
        assert block.trajectory.az.min() == pytest.approx(-60.0 - overshoot, abs=1e-3)
        assert block.trajectory.az.max() == pytest.approx(-0.1 + overshoot, abs=1e-3)

    def test_a_window_that_fits_no_branch_is_refused(self, site):
        with pytest.raises(AzimuthBoundsError):
            _plan(site, az_window=(100.0, 400.0))

    @pytest.mark.parametrize(
        ("window", "legs", "placed"),
        [
            pytest.param((-179.0, -100.0), 1, (-179.0, -100.0), id="one-leg-no-turnaround"),
            pytest.param((-178.0, -6.0), 2, (-178.0, -6.0), id="two-legs-turn-only-high"),
            pytest.param((-178.0, -6.0), 3, (182.0, 354.0), id="three-legs-turn-at-both"),
            pytest.param((182.0, 358.0), 2, (-178.0, -2.0), id="two-legs-high-turn-past-360"),
        ],
    )
    def test_the_legs_decide_which_turnarounds_must_fit(self, site, window, legs, placed):
        """The first leg runs up from the low edge, so the legs fix the overshoots reached.

        At 3 deg/s a turnaround overshoots its edge by 5.6 deg. One leg makes
        no turnaround, two turn only at the high edge, three or more at both;
        a window moves only when the overshoots it reaches would overrun.
        """
        velocity = 3.0
        t_turn = turnaround_duration_sec(velocity, 1.0)
        duration = legs * (window[1] - window[0]) / velocity + (legs - 1) * t_turn
        block = _plan(site, az_window=window, elevation=40.0, velocity=velocity, duration=duration)
        cp = block.computed_params
        assert cp["n_scans"] == legs
        assert (cp["az_start"], cp["az_stop"]) == pytest.approx(placed)
        overshoot = turnaround_overshoot_deg(velocity, 1.0)
        reach_lo = placed[0] - (overshoot if legs >= 3 else 0.0)
        reach_hi = placed[1] + (overshoot if legs >= 2 else 0.0)
        # Samples 0.1 s apart come within 0.002 deg of a turnaround's apex.
        assert block.trajectory.az.min() == pytest.approx(reach_lo, abs=5e-3)
        assert block.trajectory.az.max() == pytest.approx(reach_hi, abs=5e-3)

    @pytest.mark.parametrize(
        ("legs", "shift"),
        [(1, 0.0), (2, -360.0), (3, 0.0)],
    )
    def test_the_branch_rule_counts_the_legs(self, site, legs, shift):
        """(182, 358) at 3 deg/s fits as one leg, two fit a turn down, three fit no branch."""
        duration = legs * 176.0 / 3.0 + (legs - 1) * turnaround_duration_sec(3.0, 1.0)
        assert (
            _ce_branch_shift(
                182.0,
                358.0,
                n_scans=legs,
                duration=duration,
                timestep=0.1,
                velocity=3.0,
                az_accel=1.0,
                boresight_az_offset=0.0,
                limits=site.telescope_limits.azimuth,
            )
            == shift
        )

    @pytest.mark.parametrize(
        ("module", "window", "legs", "placed"),
        [
            pytest.param("i2", (184.79, 359.9), 3, (-175.21, -0.1), id="i2-low-apex"),
            pytest.param("i5", (-180.5, -4.7855), 2, (179.5, 355.2145), id="i5-high-apex"),
        ],
    )
    def test_a_sweep_is_placed_where_its_samples_fit(self, site, module, window, legs, placed):
        """Where no branch holds the whole sweep, its samples, which stop short of an apex, decide.

        At 2 deg/s a turnaround's apex lies 2.5 deg past its edge. Each
        window as given overruns a limit (i2's high turnaround passes 360 deg,
        i5's low edge lies below -180 deg), and one turn away the boresight's
        low apex under module i2 (2.29 deg lower in azimuth than the window)
        lies just below -180 deg, and its high apex under i5 (2.29 deg higher)
        just above 360 deg, so no branch holds either sweep whole. The pattern
        and the builder check samples, and those 0.1 s apart stop short of
        both apexes, so each window moves a turn and plans.
        """
        velocity = 2.0
        t_turn = turnaround_duration_sec(velocity, 1.0)
        duration = legs * (window[1] - window[0]) / velocity + (legs - 1) * t_turn
        offset = get_primecam_offset(module)
        block = _plan(
            site,
            az_window=window,
            elevation=40.0,
            velocity=velocity,
            duration=duration,
            detector_offset=offset,
        )
        cp = block.computed_params
        assert cp["n_scans"] == legs
        assert (cp["az_start"], cp["az_stop"]) == pytest.approx(placed)
        limits = site.telescope_limits.azimuth
        assert limits.min <= block.trajectory.az.min()
        assert block.trajectory.az.max() <= limits.max
        apex_lo, apex_hi = np.add(
            swept_az_envelope(*placed, velocity, 1.0), _boresight_az_offset(40.0, offset, site)
        )
        assert apex_lo < limits.min or apex_hi > limits.max

    def test_a_sweep_takes_the_branch_that_holds_all_of_it(self, site):
        """Where one branch holds the whole sweep and another only its samples, the first is taken.

        Two legs at 2 deg/s across (300, 357.5005) turn once, 2.5 deg past the
        high edge, at 360.0005 deg. Samples 0.1 s apart straddle that apex and
        stop 0.0019 deg short of it, inside the limit, but the mount runs the
        path between them too, so the window moves a turn down, where the
        whole sweep fits.
        """
        window = (300.0, 357.5005)
        overshoot = turnaround_overshoot_deg(2.0, 1.0)
        limits = site.telescope_limits.azimuth
        _, sample_hi = _ce_az_reach(*window, 2.0, 1.0, 61.5005, 0.1)
        assert sample_hi <= limits.max < window[1] + overshoot
        block = _plan(
            site,
            az_window=window,
            elevation=40.0,
            velocity=2.0,
            duration=61.5005,
            timestep=0.1,
        )
        cp = block.computed_params
        assert cp["n_scans"] == 2
        assert (cp["az_start"], cp["az_stop"]) == pytest.approx((-60.0, -2.4995))
        assert cp["az_stop"] + overshoot <= limits.max
        assert limits.min <= block.trajectory.az.min()
        assert block.trajectory.az.max() <= limits.max

    def test_a_sweep_stays_where_the_mount_path_fits(self, site):
        """The module's own path can pass a limit that the mount's does not; the window stays.

        The window above, swept by module i2: the module's path still turns
        at 360.0005 deg, but the boresight, which the mount follows, sits
        2.29 deg lower in azimuth at el 40, so its whole path fits, and the
        module's samples 0.1 s apart stop short of 360. The window keeps its
        branch.
        """
        offset = get_primecam_offset("i2")
        window = (300.0, 357.5005)
        block = _plan(
            site,
            az_window=window,
            elevation=40.0,
            velocity=2.0,
            duration=61.5005,
            timestep=0.1,
            detector_offset=offset,
        )
        cp = block.computed_params
        assert cp["n_scans"] == 2
        assert (cp["az_start"], cp["az_stop"]) == pytest.approx(window)
        overshoot = turnaround_overshoot_deg(2.0, 1.0)
        bore_apex = window[1] + _boresight_az_offset(40.0, offset, site) + overshoot
        limits = site.telescope_limits.azimuth
        assert bore_apex <= limits.max < window[1] + overshoot
        assert limits.min <= block.trajectory.az.min()
        assert block.trajectory.az.max() <= limits.max

    def test_a_sweep_moves_to_where_the_mount_path_fits(self, site):
        """Where both branches hold the samples, the one that holds the mount's whole path is taken.

        Three legs at 2 deg/s across (182.4995, 355.2146) turn at both edges,
        2.5 deg past each. Swept by module i5, whose boresight sits 2.29 deg
        higher in azimuth at el 40, the boresight's high apex lies at
        360.0005 deg. The samples 0.1 s apart stop short of it, so the window
        could be built as given, but the mount, which follows the boresight,
        would pass the limit between them. One turn down the boresight's
        whole path fits, and the module's samples stop short of its low apex
        at -180.0005, so the window moves there.
        """
        offset = get_primecam_offset("i5")
        d = _boresight_az_offset(40.0, offset, site)
        window = (182.4995, 355.2146)
        block = _plan(
            site,
            az_window=window,
            elevation=40.0,
            velocity=2.0,
            duration=267.0,
            timestep=0.1,
            detector_offset=offset,
        )
        cp = block.computed_params
        assert cp["n_scans"] == 3
        placed = (cp["az_start"], cp["az_stop"])
        assert placed == pytest.approx((window[0] - 360.0, window[1] - 360.0))
        limits = site.telescope_limits.azimuth
        lo, hi = _ce_az_reach(*window, 2.0, 1.0, float(block.trajectory.times[-1]), 0.1)
        assert all(limits.is_in_range(az) for az in (lo, hi, lo + d, hi + d))
        assert swept_az_envelope(*window, 2.0, 1.0)[1] + d > limits.max
        apex_lo, apex_hi = swept_az_envelope(*placed, 2.0, 1.0)
        assert limits.min <= apex_lo + d
        assert apex_hi + d <= limits.max
        assert limits.min <= block.trajectory.az.min()
        assert block.trajectory.az.max() <= limits.max

    def test_each_branch_is_judged_where_the_sweep_would_be_placed(self, site):
        """A whole turn can change the last bit of an apex, so a branch is judged as it is built.

        Four legs at 0.948 deg/s across (-179.46, -0.56) turn 0.56 deg past
        each edge: the low apex passes -180 deg, though the samples 0.63 s
        apart stop 0.05 deg short of it, and the high apex lies 2.4e-14 deg
        above 0. That apex moved a turn up lands on 360.0, inside the limit,
        but the pattern places the window first and then adds the overshoot,
        which puts the apex, and the sample at it, at 360.00000000000006. So
        the window keeps the branch where its samples fit.
        """
        window = (-179.46060402025557, -0.5618678898567565)
        velocity = 0.9481501061387106
        assert swept_az_envelope(*window, velocity, 1.0)[1] + 360.0 == 360.0
        assert swept_az_envelope(window[0] + 360.0, window[1] + 360.0, velocity, 1.0)[1] > 360.0
        block = _plan(
            site,
            az_window=window,
            elevation=40.0,
            velocity=velocity,
            duration=760.416385123246,
            timestep=0.6321000707591404,
        )
        cp = block.computed_params
        assert cp["n_scans"] == 4
        assert (cp["az_start"], cp["az_stop"]) == window
        limits = site.telescope_limits.azimuth
        assert limits.min <= block.trajectory.az.min()
        assert block.trajectory.az.max() <= limits.max

    def test_a_tie_keeps_the_branch_it_was_given(self, site):
        """Where both branches hold the samples and neither the whole path, the window stays.

        Three legs at 2 deg/s across (182.49, 357.51) turn 2.5 deg past each
        edge, so the path runs from 179.99 to 360.01 deg: its high apex passes
        360 where the window is, and one turn down its low apex passes -180.
        Samples 0.3 s apart stop short of both apexes, so either branch could
        be built, and the window keeps its own.
        """
        window = (182.49, 357.51)
        block = _plan(
            site,
            az_window=window,
            elevation=40.0,
            velocity=2.0,
            duration=270.53,
            timestep=0.3,
        )
        cp = block.computed_params
        assert cp["n_scans"] == 3
        assert (cp["az_start"], cp["az_stop"]) == window
        limits = site.telescope_limits.azimuth
        duration = float(block.trajectory.times[-1])
        for turn in (0.0, -360.0):
            lo, hi = window[0] + turn, window[1] + turn
            sample_lo, sample_hi = _ce_az_reach(lo, hi, 2.0, 1.0, duration, 0.3)
            assert limits.min <= sample_lo
            assert sample_hi <= limits.max
            apex_lo, apex_hi = swept_az_envelope(lo, hi, 2.0, 1.0)
            assert apex_lo < limits.min or apex_hi > limits.max
        assert limits.min <= block.trajectory.az.min()
        assert block.trajectory.az.max() <= limits.max

    @pytest.mark.parametrize(
        ("module", "window", "placed"),
        [
            pytest.param("i2", (186.0, 358.0), (-174.0, -2.0), id="i2-high-turn-past-360"),
            pytest.param("i5", (-178.0, -6.0), (-178.0, -6.0), id="i5-no-low-turn"),
        ],
    )
    def test_an_off_axis_two_leg_sweep_is_placed_by_the_turnarounds_it_makes(
        self, site, module, window, placed
    ):
        """A 100 s sweep at 3 deg/s is two legs, which turn only at the high edge.

        With module i2 the window's high turnaround reaches 363.62 deg, so the
        window moves a turn down, where the boresight, 2.29 deg lower, fits
        too. With module i5 nothing turns at the low edge, -178 deg, and the
        high turnaround and the boresight, 2.29 deg higher, end below 2 deg, so
        the window keeps its branch.
        """
        block = _plan(
            site,
            az_window=window,
            elevation=40.0,
            velocity=3.0,
            duration=100.0,
            detector_offset=get_primecam_offset(module),
        )
        cp = block.computed_params
        assert cp["n_scans"] == 2
        assert (cp["az_start"], cp["az_stop"]) == pytest.approx(placed)
        limits = site.telescope_limits.azimuth
        assert limits.min <= block.trajectory.az.min()
        assert block.trajectory.az.max() <= limits.max

    @pytest.mark.parametrize(
        ("module", "window", "placed"),
        [
            pytest.param("i5", (345.14, 354.78), (-14.86, -5.22), id="i5-past-360"),
            pytest.param("i2", (-176.0, -168.0), (184.0, 192.0), id="i2-past-minus-180"),
        ],
    )
    def test_an_off_axis_module_takes_the_branch_its_boresight_holds(
        self, site, module, window, placed
    ):
        """A detector offset moves the boresight's turnarounds, and the branch must hold them.

        At el 40 module i5 puts the boresight 2.29 deg higher in azimuth than
        the window and i2 2.29 deg lower. At 2.2 deg/s each window's own
        turnaround envelope, 3.03 deg beyond each end, fits its branch, but
        the boresight's runs past 360 deg or below -180 deg, so the window
        moves a whole turn.
        """
        block = _plan(
            site,
            az_window=window,
            elevation=40.0,
            velocity=2.2,
            duration=600.0,
            detector_offset=get_primecam_offset(module),
        )
        cp = block.computed_params
        assert (cp["az_start"], cp["az_stop"]) == pytest.approx(placed)
        limits = site.telescope_limits.azimuth
        assert limits.min <= block.trajectory.az.min()
        assert block.trajectory.az.max() <= limits.max

    def test_the_commanded_window_must_fit_as_well(self, site):
        """The pattern checks the commanded window and the builder the boresight, so both must fit.

        Module i2 puts the boresight 2.29 deg lower in azimuth than the window.
        At 3 deg/s the window's turnaround envelope ends at 360.41, past the
        limit, while the boresight's ends at 358.11, inside it; the window
        still moves a turn down, since the pattern checks the window's
        samples, which pass the limit, before the offset is applied.
        """
        block = _plan(
            site,
            az_window=(345.14, 354.78),
            elevation=40.0,
            velocity=3.0,
            duration=600.0,
            detector_offset=get_primecam_offset("i2"),
        )
        cp = block.computed_params
        assert (cp["az_start"], cp["az_stop"]) == pytest.approx((-14.86, -5.22))

    @pytest.mark.parametrize("module", ["i1", "i2", "i3", "i4", "i5", "i6"])
    def test_the_boresight_sits_a_constant_azimuth_from_the_window(self, site, module):
        """At a fixed elevation the offset moves the boresight by one azimuth, the rule's input."""
        offset = get_primecam_offset(module)
        bare = _plan(site, duration=600.0)
        block = _plan(site, duration=600.0, detector_offset=offset)
        shift = block.trajectory.az - bare.trajectory.az
        assert np.ptp(shift) < 1e-9
        assert shift[0] == pytest.approx(_boresight_az_offset(50.0, offset, site), abs=1e-9)

    @pytest.mark.parametrize(
        ("interval", "shift"),
        [
            ((100.0, 200.0), 0.0),
            ((300.0, 370.0), -360.0),
            ((-200.0, -170.0), 360.0),
            ((-170.0, -150.0), 0.0),
            ((100.0, 400.0), 0.0),
            ((300.0, 359.9), 0.0),
            ((-179.9, -120.0), 0.0),
        ],
    )
    def test_one_leg_moves_only_when_a_turn_away_holds_it(self, site, interval, shift):
        """One leg makes no turnaround, so its interval alone decides its branch.

        The last two intervals end 0.1 deg inside a limit, within the
        0.625 deg a turnaround at 1 deg/s would overshoot, and stay.
        """
        lo, hi = interval
        assert (
            _ce_branch_shift(
                lo,
                hi,
                n_scans=1,
                duration=hi - lo,
                timestep=0.1,
                velocity=1.0,
                az_accel=1.0,
                boresight_az_offset=0.0,
                limits=site.telescope_limits.azimuth,
            )
            == shift
        )


class TestWindowRefusals:
    """Refusals: a ValueError naming the argument, or the trajectory validation's bounds error."""

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            pytest.param({"az_window": (111.6, 101.0)}, r"az_hi > az_lo", id="inverted"),
            pytest.param({"az_window": (101.0, 101.0)}, r"az_hi > az_lo", id="equal-ends"),
            pytest.param({"az_window": 101.0}, r"must be a pair", id="scalar"),
            pytest.param({"az_window": (100.0, 105.0, 110.0)}, r"must be a pair", id="triple"),
            pytest.param({"az_window": None}, r"must be a pair", id="none"),
            pytest.param({"az_window": (math.nan, 111.6)}, r"az_window\[0\] must", id="nan-lo"),
            pytest.param({"az_window": (101.0, math.inf)}, r"az_window\[1\] must", id="inf-hi"),
            pytest.param({"az_window": (True, 111.6)}, r"az_window\[0\] must", id="bool-lo"),
            pytest.param({"az_window": ("101", 111.6)}, r"az_window\[0\] must", id="text-lo"),
            pytest.param(
                {"az_window": (0.0, 350.0), "az_padding": 5.001},
                r"more than a full turn",
                id="padded-past-a-turn",
            ),
            pytest.param({"elevation": math.nan}, r"elevation must", id="elevation-nan"),
            pytest.param({"velocity": 0.0}, r"velocity must", id="velocity-zero"),
            pytest.param({"velocity": -0.5}, r"velocity must", id="velocity-negative"),
            pytest.param({"velocity": True}, r"velocity must", id="velocity-bool"),
            pytest.param({"duration": 0.0}, r"duration must", id="duration-zero"),
            pytest.param({"duration": -60.0}, r"duration must", id="duration-negative"),
            pytest.param({"duration": None}, r"duration must", id="duration-none"),
            pytest.param({"az_accel": 0.0}, r"az_accel must", id="az_accel-zero"),
            pytest.param({"az_padding": -1.0}, r"az_padding must", id="padding-negative"),
            pytest.param({"az_padding": 361.0}, r"az_padding must", id="padding-past-a-turn"),
            pytest.param({"az_padding": "2.0"}, r"az_padding must", id="padding-text"),
            pytest.param({"timestep": 0.0}, r"timestep must", id="timestep-zero"),
        ],
    )
    def test_a_malformed_argument_is_a_value_error(self, site, overrides, match):
        with pytest.raises(ValueError, match=match) as excinfo:
            _plan(site, **overrides)
        assert not isinstance(excinfo.value, PointingError)

    def test_a_window_one_full_turn_wide_is_accepted(self, site):
        """The width check is inclusive: a padded window of exactly 360 deg is one long leg."""
        block = _plan(site, az_window=(0.0, 350.0), az_padding=5.0, duration=900.0)
        cp = block.computed_params
        assert (cp["az_start"], cp["az_stop"]) == (-5.0, 355.0)
        assert cp["n_scans"] == 1

    @pytest.mark.parametrize("field", [None, _FIELD], ids=["no-field", "field"])
    @pytest.mark.parametrize("module", [None, "i1"])
    @pytest.mark.parametrize("elevation", [-30.0, -5.0, 5.0, 10.0, 95.0])
    def test_an_elevation_outside_the_limits_is_refused(self, site, elevation, module, field):
        """The build refuses the elevation before the drift checks run, so nothing warns."""
        offset = None if module is None else get_primecam_offset(module)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ElevationBoundsError):
                _plan(site, field, elevation=elevation, detector_offset=offset)
        assert _pointing_warnings(caught) == []

    def test_a_module_at_the_zenith_is_refused_for_every_sample(self, site):
        """The boresight solve fails at el 90, and the builder's refusal names the samples."""
        with pytest.raises(OffsetInversionError) as excinfo:
            _plan(site, elevation=90.0, detector_offset=get_primecam_offset("i1"))
        assert excinfo.value.indices

    def test_a_module_whose_boresight_solve_does_not_converge_is_refused(self, site):
        """At el 88.5, inside the elevation limits, the boresight solve for i1 does not converge."""
        with pytest.raises(OffsetInversionError, match="failed to converge"):
            _plan(site, elevation=88.5, detector_offset=get_primecam_offset("i1"))

    @pytest.mark.filterwarnings(
        "ignore:Turnaround distance:fyst_trajectories.exceptions.PointingWarning"
    )
    def test_a_sweep_shorter_than_two_samples_is_refused(self, site):
        """A 0.01 deg window at 0.5 deg/s is one 0.02 s leg, under two 0.1 s samples."""
        with pytest.raises(ValueError, match="fewer than 2 samples") as excinfo:
            _plan(site, az_window=(101.0, 101.01), duration=0.01)
        assert not isinstance(excinfo.value, PointingError)

    def test_an_array_start_time_is_refused(self, site):
        times = Time(["2026-03-15T03:00:00", "2026-03-15T04:00:00"], scale="utc")
        with pytest.raises(ValueError, match="single instant"):
            _plan(site, start_time=times)

    def test_a_start_time_that_is_not_a_time_is_refused(self, site):
        with pytest.raises(TypeError, match="start_time"):
            _plan(site, start_time=1773543600.0)


class TestWindowDriftChecks:
    """Warn-only checks that the named field drifts through the window during the scan."""

    def test_the_passes_start_in_the_dark(self, field_pass, coordinates):
        """The anchors' passes start at the minutes their comment states, the Sun below -18 deg."""
        side, block = field_pass
        start = block.trajectory.start_time
        expected = {"rising": "2026-11-16 01:16", "setting": "2026-11-16 07:08"}[side]
        assert start.utc.iso[:16] == expected
        _, sun_el = coordinates.get_sun_altaz(start + np.array([0.0, block.duration]) * u.s)
        assert np.all(np.asarray(sun_el, dtype=float) < -18.0)

    def test_the_field_forms_pass_is_silent(self, field_pass):
        _, block = field_pass
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _replay(block, field=_FIELD)
        assert _pointing_warnings(caught) == []

    @pytest.mark.parametrize("offset", [-600.0, 600.0], ids=["10-min-early", "10-min-late"])
    def test_a_scan_over_part_of_the_pass_is_silent(self, field_pass, offset):
        _, block = field_pass
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _replay(block, field=_FIELD, start_time=block.trajectory.start_time + offset * u.s)
        assert _pointing_warnings(caught) == []

    def test_a_start_after_the_field_has_drifted_through_warns(self, field_pass):
        """The warning names when the trailing edge crossed, as the field form solves it."""
        side, block = field_pass
        late = block.trajectory.start_time + 3600.0 * u.s
        with pytest.warns(PointingWarning, match="starts after the field has drifted") as record:
            _replay(block, field=_FIELD, start_time=late)
        (warning,) = [w for w in record if issubclass(w.category, PointingWarning)]
        message = str(warning.message)
        assert f"has already crossed el=50.00 deg on the {side} side, at " in message
        crossed = Time(message.rsplit(", at ", 1)[1].rstrip("."), scale="utc")
        expected = Time(block.computed_params["end_time_iso"], scale="utc")
        assert abs((crossed - expected).sec) < 1.0
        assert warning.filename == __file__

    def test_an_end_before_the_field_arrives_warns(self, field_pass):
        """The warning names when the leading edge crosses, as the field form solves it."""
        side, block = field_pass
        early = block.trajectory.start_time - 3600.0 * u.s
        with pytest.warns(PointingWarning, match="ends before the field arrives") as record:
            _replay(block, field=_FIELD, start_time=early, duration=1200.0)
        (warning,) = [w for w in record if issubclass(w.category, PointingWarning)]
        message = str(warning.message)
        assert f"has not crossed el=50.00 deg on the {side} side by the requested end" in message
        arrives = Time(message.rsplit("it crosses at ", 1)[1].rstrip("."), scale="utc")
        expected = Time(block.computed_params["start_time_iso"], scale="utc")
        assert abs((arrives - expected).sec) < 1.0
        assert warning.filename == __file__

    @pytest.mark.parametrize(
        "atmosphere",
        [None, AtmosphericConditions.for_fyst()],
        ids=["vacuum", "refracted"],
    )
    def test_the_pass_has_the_field_forms_bounds(self, field_pass, site, atmosphere):
        """Two seconds either side of each edge's crossing, refracted or not.

        A scan that starts before the trailing edge crosses, or ends after the
        leading edge does, is silent; two seconds the other way it warns. The
        five-hour scans start with the leading edge far from its crossing,
        below the horizon on the rising side.
        """
        side, _ = field_pass
        block = plan_constant_el_scan(
            _FIELD,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=_PASS_ANCHORS[side],
            rising=side == "rising",
            az_padding=0.0,
            atmosphere=atmosphere,
        )
        lead = Time(block.computed_params["start_time_iso"], scale="utc")
        trail = Time(block.computed_params["end_time_iso"], scale="utc")
        cases = [
            (trail - 2.0 * u.s, 600.0, None),
            (trail + 2.0 * u.s, 600.0, "starts after the field has drifted through"),
            (lead - 602.0 * u.s, 600.0, "ends before the field arrives"),
            (lead - 598.0 * u.s, 600.0, None),
            (lead - 18002.0 * u.s, 18000.0, "ends before the field arrives"),
            (lead - 17998.0 * u.s, 18000.0, None),
        ]
        for start, duration, expected in cases:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _replay(
                    block,
                    field=_FIELD,
                    start_time=start,
                    duration=duration,
                    atmosphere=atmosphere,
                    timestep=1.0,
                )
            messages = _pointing_warnings(caught)
            if expected is None:
                assert messages == [], start.iso
            else:
                assert len(messages) == 1 and expected in messages[0], start.iso

    @pytest.mark.parametrize(
        ("dec", "culminations"),
        [(60.0, "el=7.01 and el=-52.99"), (-80.0, "el=32.99 and el=12.99")],
        ids=["never-high-enough", "never-low-enough"],
    )
    def test_a_field_that_never_crosses_the_elevation_warns(self, site, dec, culminations):
        field = FieldRegion(ra_center=53.0, dec_center=dec, width=2.0, height=2.0)
        with pytest.warns(PointingWarning, match="field never crosses el=50.00 deg") as record:
            _plan(site, field, duration=600.0)
        (warning,) = [w for w in record if issubclass(w.category, PointingWarning)]
        assert f"culminates at {culminations} deg" in str(warning.message)
        assert warning.filename == __file__

    @pytest.mark.parametrize(
        ("field", "window"),
        [
            pytest.param(None, None, id="no-field"),
            pytest.param(_FIELD, (170.0, 190.0), id="window-across-the-meridian"),
            pytest.param(_FIELD, (170.0, 180.0), id="window-ending-on-the-meridian"),
            pytest.param(_BAND, None, id="field-around-the-ra-circle"),
        ],
    )
    def test_the_checks_are_skipped(self, field_pass, field, window):
        """A start an hour after the pass warns for E-CDF-S; these cases are not checked."""
        _, block = field_pass
        overrides = {"field": field, "start_time": block.trajectory.start_time + 3600.0 * u.s}
        if window is not None:
            overrides["az_window"] = window
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _replay(block, **overrides)
        assert _pointing_warnings(caught) == []

    def test_the_side_is_read_from_the_padded_window(self, field_pass):
        """A window that reaches the meridian only once padded is not checked."""
        _, block = field_pass
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _replay(
                block,
                field=_FIELD,
                az_window=(170.0, 179.0),
                az_padding=2.0,
                start_time=block.trajectory.start_time + 3600.0 * u.s,
            )
        assert _pointing_warnings(caught) == []

    @pytest.mark.parametrize(
        ("window", "side"),
        [
            ((101.0, 111.6), "rising"),
            ((1.0, 179.0), "rising"),
            ((248.0, 260.0), "setting"),
            ((-170.0, -150.0), "setting"),
            ((-60.0, -0.1), "setting"),
            ((170.0, 190.0), None),
            ((170.0, 180.0), None),
            ((180.0, 190.0), None),
            ((-5.0, 10.0), None),
            ((355.0, 370.0), None),
        ],
    )
    def test_the_window_side(self, window, side):
        """East of the meridian a field rises, west of it it sets; the meridian has no side."""
        assert _window_side(*window) == side

    @pytest.mark.parametrize("near", ["start", "end"])
    def test_a_wide_field_is_silent_inside_its_pass(self, field_pass, near):
        """A 150 deg field draws no advisory 0.3 h from either end of its 10 h pass."""
        _, block = field_pass
        pass_start, pass_end = _wide_pass(block, 150.0)
        if near == "start":
            start = pass_start + 0.3 * u.hour
        else:
            start = pass_end - 0.3 * u.hour - 1200.0 * u.s
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _replay(block, field=_wide_field(150.0), start_time=start, duration=1200.0)
        assert _pointing_warnings(caught) == []

    def test_a_wide_field_is_silent_while_its_tail_drifts_through(self, field_pass):
        """A 130 deg field is swept for an hour from 0.3 h before its trailing edge crosses.

        On the setting side its leading edge passes its lower culmination
        during the scan; the scan still started inside the pass.
        """
        _, block = field_pass
        _, pass_end = _wide_pass(block, 130.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _replay(
                block,
                field=_wide_field(130.0),
                start_time=pass_end - 0.3 * u.hour,
                duration=3600.0,
            )
        assert _pointing_warnings(caught) == []

    def test_a_long_scan_from_inside_a_pass_near_the_pole_is_silent(self, site):
        """A 1 deg field at Dec -80 sets through el 22 in about 23 min.

        Swept for 5.9 h from 3 min into its pass, by when its leading edge has
        passed its lower culmination, the scan draws no advisory.
        """
        field = FieldRegion(ra_center=315.0, dec_center=-80.0, width=1.0, height=1.0)
        block = plan_constant_el_scan(
            field,
            elevation=22.0,
            velocity=0.5,
            site=site,
            start_time="2026-11-16T00:00:00",
            rising=False,
            az_padding=0.0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _replay(
                block,
                field=field,
                elevation=22.0,
                start_time=block.trajectory.start_time + 180.0 * u.s,
                duration=5.9 * 3600.0,
                timestep=1.0,
            )
        assert _pointing_warnings(caught) == []

    @pytest.mark.parametrize("ra_span", [135.0, 136.5, 200.0, 300.0])
    def test_a_start_after_a_wide_fields_pass_warns(self, field_pass, ra_span):
        """An hour after a field's pass has ended, however wide the field, the scan starts late.

        The field's pass ends 2 h before E-CDF-S's begins, in the dark.
        """
        _, block = field_pass
        with pytest.warns(PointingWarning, match="starts after the field has drifted") as record:
            _replay(
                block,
                field=_field_ending_before(ra_span, 2.0),
                start_time=block.trajectory.start_time - 1.0 * u.hour,
                duration=1200.0,
            )
        (warning,) = [w for w in record if issubclass(w.category, PointingWarning)]
        assert warning.filename == __file__

    @pytest.mark.parametrize(
        ("hours", "expected"),
        [
            (0.5, "starts after the field has drifted through"),
            (2.5, "ends before the field arrives"),
            (3.5, None),
        ],
    )
    def test_a_scan_between_passes_names_the_nearer_one(self, field_pass, hours, expected):
        """A 300 deg field leaves a 4 h gap between passes; each scan starts 1 h into it.

        Ending 2.5 h before the next pass, the scan is nearer the pass before;
        ending 0.5 h before it, nearer the next one; lasting 3.5 h, it reaches it.
        """
        _, block = field_pass
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _replay(
                block,
                field=_field_ending_before(300.0, 2.0),
                start_time=block.trajectory.start_time - 1.0 * u.hour,
                duration=hours * 3600.0,
            )
        messages = _pointing_warnings(caught)
        if expected is None:
            assert messages == []
        else:
            assert len(messages) == 1 and expected in messages[0]

    def test_precession_moves_a_declination_by_under_0p2_deg_before_2035(self):
        """The bound the drift check states for judging culminations from the catalogue Dec.

        A declination moves by at most the angle between the two poles, about
        0.195 deg from J2000 to J2035.
        """
        ra, dec = np.meshgrid(np.linspace(0.0, 360.0, 73), np.linspace(-89.0, 89.0, 90))
        j2000 = SkyCoord(ra=ra.ravel() * u.deg, dec=dec.ravel() * u.deg, frame=FK5(equinox="J2000"))
        moved = np.abs(j2000.transform_to(FK5(equinox="J2035")).dec.deg - dec.ravel())
        assert 0.19 < moved.max() < 0.2


class TestWindowLongBlock:
    """A block longer than 6 h draws the same advisory as the field form's long pass."""

    def test_a_block_over_six_hours_warns(self, site):
        with pytest.warns(PointingWarning, match=r"duration 6\.5h is unusually long") as record:
            _plan(site, duration=6.5 * 3600.0, timestep=1.0)
        (warning,) = [w for w in record if issubclass(w.category, PointingWarning)]
        assert warning.filename == __file__

    def test_a_block_of_six_hours_is_silent(self, site):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _plan(site, duration=6.0 * 3600.0, timestep=1.0)
        assert _pointing_warnings(caught) == []


# 2026-03-15 at dawn: the Sun stands at az 89.9, el 5.4 deg at 11:00 UTC and
# climbs about 14 deg in the hour, towards a window fixed above it.
_DAWN = Time("2026-03-15T11:00:00", scale="utc")
_DAWN_MESSAGE = (
    "EXCLUSION ZONE: planned constant-elevation window scan over az [80.00, 100.00] at el=55.00"
)


def _dawn_block(site, **overrides):
    """Plan an hour over az 80 to 100 deg at el 55, from 2026-03-15 11:00 UTC.

    Sampled at 1 s, so the test's own per-sample Sun solve stays short.
    """
    kwargs = dict(
        az_window=(80.0, 100.0), elevation=55.0, duration=3600.0, start_time=_DAWN, timestep=1.0
    )
    kwargs.update(overrides)
    return _plan(site, **kwargs)


def _sample_separations(coordinates, trajectory):
    """Return each sample's time and separation from the Sun, solved at every sample."""
    times = get_absolute_times(trajectory)
    sun_az, sun_el = coordinates.get_sun_altaz(times)
    separations = coordinates.angular_separation(trajectory.az, trajectory.el, sun_az, sun_el)
    return times, np.asarray(separations, dtype=float)


class TestWindowSunScreen:
    """The built trajectory is screened against the Sun over the whole block (warn-only)."""

    def test_a_window_the_sun_closes_on_warns(self, site, coordinates):
        """The block starts outside the 45 deg zone and ends 36 deg from the Sun.

        The warning names the closest sample the per-sample ephemeris finds,
        and it is attributed to the planner's caller.
        """
        with pytest.warns(PointingWarning, match=re.escape(_DAWN_MESSAGE)) as record:
            block = _dawn_block(site)
        (warning,) = [w for w in record if issubclass(w.category, PointingWarning)]
        assert warning.filename == __file__
        message = str(warning.message)
        match = re.search(r"passes ([0-9.]+) deg from the Sun at (\S+ \S+) \(", message)
        assert match, message
        times, separations = _sample_separations(coordinates, block.trajectory)
        closest = int(np.argmin(separations))
        # The message rounds the separation to 0.1 deg and the time to 1 ms.
        assert float(match.group(1)) == pytest.approx(separations[closest], abs=0.05)
        assert abs((Time(match.group(2), scale="utc") - times[closest]).sec) < 0.5
        assert separations[0] > 45.0
        assert separations[-1] == pytest.approx(36.0, abs=0.5)
        # The dawn the comment above _DAWN states.
        sun_az, sun_el = coordinates.get_sun_altaz(_DAWN + np.array([0.0, 3600.0]) * u.s)
        assert float(sun_az[0]) == pytest.approx(89.9, abs=0.05)
        assert float(sun_el[0]) == pytest.approx(5.4, abs=0.05)
        assert float(sun_el[1] - sun_el[0]) == pytest.approx(14.0, abs=0.5)

    def test_disabled_avoidance_and_a_permissive_model_are_silent(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _dawn_block(get_fyst_site(sun_avoidance_enabled=False))
            _dawn_block(get_fyst_site(), sun_safe=lambda az, el, t: True)
        assert _pointing_warnings(caught) == []

    def test_an_injected_model_that_turns_unsafe_late_warns(self, site):
        """A model that turns unsafe halfway through the block is heard at its first unsafe probe.

        The 16863 samples are probed every 29th, 2.9 s apart, so the warning
        names a time within one stride after the turn.
        """
        midpoint = _NIGHT.unix + 0.5 * _plan(site).duration

        def late_unsafe(az, el, t):
            return t.unix < midpoint

        with pytest.warns(PointingWarning, match="enters the Sun avoidance zone") as record:
            _plan(site, sun_safe=late_unsafe)
        (warning,) = [w for w in record if issubclass(w.category, PointingWarning)]
        assert warning.filename == __file__
        match = re.search(r"\) at (\S+ \S+)\.$", str(warning.message))
        assert match, str(warning.message)
        assert 0.0 <= Time(match.group(1), scale="utc").unix - midpoint < 2.9

    @pytest.mark.parametrize("with_batch", [False, True], ids=["predicate", "batch"])
    def test_an_injected_model_sees_at_most_601_samples(self, site, with_batch):
        """An injected model is asked about at most 600 strided samples plus the last one."""
        seen_jd: list[float] = []

        class Spy:
            def __call__(self, az, el, t):
                seen_jd.append(float(t.jd))
                return True

        class BatchSpy(Spy):
            def batch(self, az, el, times):
                seen_jd.extend(np.atleast_1d(times.jd).tolist())
                return np.ones(np.size(az), dtype=bool)

        block = _plan(site, sun_safe=BatchSpy() if with_batch else Spy())
        assert block.trajectory.n_points > 601
        assert len(seen_jd) <= 601
        times = get_absolute_times(block.trajectory)
        for edge in (times[0], times[-1]):
            assert np.any(np.isclose(seen_jd, float(edge.jd), rtol=0.0, atol=1e-8))
