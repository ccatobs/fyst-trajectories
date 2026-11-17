"""Tests for plan_constant_el_scan."""

import functools
import inspect
import math
import warnings
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from _off_axis_offsets import off_axis_offset
from astropy import units as u
from astropy.time import Time, TimeDelta
from scipy.optimize import brentq

from fyst_trajectories import get_fyst_site
from fyst_trajectories.coordinates import Coordinates
from fyst_trajectories.exceptions import (
    AzimuthBoundsError,
    ElevationBoundsError,
    OffsetInversionError,
    PointingError,
    PointingWarning,
)
from fyst_trajectories.patterns.configs import ConstantElScanConfig
from fyst_trajectories.patterns.constant_el import _ce_az_reach
from fyst_trajectories.patterns.turnarounds import swept_az_envelope, turnaround_overshoot_deg
from fyst_trajectories.planning import (
    FieldRegion,
    ScanBlock,
    plan_constant_el_scan,
)
from fyst_trajectories.planning._ce_geometry import (
    _boresight_az_offset,
    _ce_field_az_range,
    _compute_ce_az_range,
    _compute_ce_duration,
    _compute_ce_duration_from_lsa,
    _field_region_corners,
    _quantize_ce_duration,
)
from fyst_trajectories.primecam import get_primecam_offset
from fyst_trajectories.site import AtmosphericConditions
from fyst_trajectories.trajectory_utils import get_absolute_times

# One turn of the Earth relative to the stars, in seconds.
_SIDEREAL_DAY_SEC = 86164.0905


def _unit_vectors(az_deg, el_deg):
    """Horizon-frame unit vectors of azimuths and elevations in degrees."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def _sun_separation_of_every_sample(site, trajectory):
    """Angle in degrees between each sample and the Sun.

    The Sun is solved once a second and its unit vector interpolated to the
    samples; it moves about 0.004 deg in a second, so the interpolation is
    exact to far below the 0.1 deg an advisory prints.
    """
    times = get_absolute_times(trajectory)
    rel = (times - times[0]).sec
    grid = np.arange(0.0, rel[-1] + 1.0, 1.0)
    sun_az, sun_el = Coordinates(site).get_sun_altaz(times[0] + TimeDelta(grid, format="sec"))
    sun = np.stack([np.interp(rel, grid, c) for c in _unit_vectors(sun_az, sun_el)])
    sun /= np.linalg.norm(sun, axis=0)
    cosine = np.sum(sun * _unit_vectors(trajectory.az, trajectory.el), axis=0)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


@pytest.fixture(scope="module")
def ecdfs_block():
    """Return the E-CDF-S rising pass at el = 50 the block-shape tests share."""
    return plan_constant_el_scan(
        field=FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7),
        elevation=50.0,
        velocity=0.5,
        site=get_fyst_site(),
        start_time=Time("2026-03-15T17:00:00", scale="utc"),
        rising=True,
        angle=170.0,
    )


class TestPlanConstantElScan:
    """Block shape, the held elevation, the rising/setting split, and the refusals."""

    @pytest.fixture
    def ecdfs_field(self):
        """E-CDF-S field region used in the scan strategy script."""
        return FieldRegion(
            ra_center=53.117,
            dec_center=-27.808,
            width=5.0,
            height=6.7,
        )

    @pytest.fixture
    def search_time(self):
        """Provide a base search time for E-CDF-S CE scans."""
        return Time("2026-03-15T17:00:00", scale="utc")

    def test_basic_plan(self, ecdfs_block):
        """plan_constant_el_scan returns a ScanBlock with correct types."""
        block = ecdfs_block

        assert isinstance(block, ScanBlock)
        assert isinstance(block.config, ConstantElScanConfig)
        assert block.duration > 0
        assert block.trajectory.n_points > 0
        assert "az_start" in block.computed_params
        assert "az_stop" in block.computed_params
        assert "az_throw" in block.computed_params
        assert "n_scans" in block.computed_params
        assert "Constant-El scan" in block.summary
        assert block.computed_params["az_throw"] > 0

    def test_elevation_in_trajectory(self, ecdfs_block):
        assert np.allclose(ecdfs_block.trajectory.el, 50.0)

    def test_rising_vs_setting_different_times(self, site, ecdfs_field, search_time):
        rising = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=True,
            angle=170.0,
        )
        setting = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=False,
            angle=170.0,
        )

        # The rising crossing comes first and lies east of the meridian, the
        # setting one west of it, so a swapped flag fails every line below.
        t_rise = Time(rising.computed_params["start_time_iso"], scale="utc")
        t_set = Time(setting.computed_params["start_time_iso"], scale="utc")
        assert t_rise < t_set
        assert rising.computed_params["az_stop"] < 180.0
        assert setting.computed_params["az_start"] > 180.0

    def test_a_tt_start_time_records_utc_times(self, site, ecdfs_field, search_time, ecdfs_block):
        """The recorded times and the summary are UTC whatever scale ``start_time`` is in."""
        block = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time.tt,
            rising=True,
            angle=170.0,
        )

        for key in ("start_time_iso", "end_time_iso"):
            assert block.computed_params[key] == ecdfs_block.computed_params[key]
        assert block.summary == ecdfs_block.summary

    def test_string_start_time(self, site, ecdfs_field):
        block = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time="2026-03-15T17:00:00",
            rising=True,
            angle=170.0,
        )

        assert block.duration > 0

    def test_unreachable_field_raises(self, site):
        # Dec = +70 is never reachable at el=50 from FYST
        field = FieldRegion(ra_center=180.0, dec_center=70.0, width=1.0, height=1.0)
        with pytest.raises(PointingError, match="Could not find elevation crossing"):
            plan_constant_el_scan(
                field=field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=Time("2026-03-15T00:00:00", scale="utc"),
                rising=True,
            )

    @pytest.mark.parametrize(
        "timing",
        [{"rising": True}, {"rising": False}, {"lsa_window": (100.0, 110.0)}],
        ids=["rising", "setting", "lsa_window"],
    )
    def test_a_field_with_a_corner_past_a_pole_raises(self, site, timing):
        """A field at Dec -80 that is 25 deg tall reaches Dec -92.5, so it is refused."""
        field = FieldRegion(ra_center=100.0, dec_center=-80.0, width=2.0, height=25.0)
        with pytest.raises(PointingError, match="corners pass a celestial pole") as excinfo:
            plan_constant_el_scan(
                field=field,
                elevation=25.0,
                velocity=0.5,
                site=site,
                start_time=Time("2026-03-15T00:00:00", scale="utc"),
                max_search_hours=24.0,
                **timing,
            )
        assert "Dec=-80.000, height=25.000, angle=0.0 deg" in str(excinfo.value)

    @pytest.mark.parametrize("elevation", [95.0, -5.0])
    @pytest.mark.parametrize(
        "timing",
        [{"rising": True}, {"lsa_window": (100.0, 105.0)}],
        ids=["crossing", "lsa_window"],
    )
    def test_an_elevation_outside_the_limits_is_refused(
        self, site, ecdfs_field, search_time, elevation, timing
    ):
        """The elevation is refused before any search, so nothing warns first."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ElevationBoundsError) as excinfo:
                plan_constant_el_scan(
                    field=ecdfs_field,
                    elevation=elevation,
                    velocity=0.5,
                    site=site,
                    start_time=search_time,
                    **timing,
                )
        assert excinfo.value.actual_min == excinfo.value.actual_max == elevation
        assert [w for w in caught if issubclass(w.category, PointingWarning)] == []

    def test_anchor_inside_a_pass_raises(self, site, ecdfs_field):
        """A search anchored mid-pass refuses to join edges from different passes.

        Anchored at the midpoint of the rising E-CDF-S pass with a 30 h horizon,
        the leading edge's next crossing falls a day after the trailing edge's,
        so the planner refuses instead of returning a pass of about a day.
        """
        with pytest.raises(PointingError, match="different passes"):
            plan_constant_el_scan(
                field=ecdfs_field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=Time("2026-03-15T17:34:26.717", scale="utc"),
                rising=True,
                angle=170.0,
                max_search_hours=30.0,
            )

    def test_negative_az_padding_raises(self, site, ecdfs_field, search_time):
        """A negative padding would narrow the scan below the field's own extent."""
        with pytest.raises(ValueError, match="az_padding"):
            plan_constant_el_scan(
                field=ecdfs_field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=search_time,
                rising=True,
                angle=170.0,
                az_padding=-1.0,
            )

    @pytest.mark.parametrize(
        "bad",
        [float("nan"), "2.0", None, 9e307, 10**400, 361.0],
        ids=["nan", "text", "none", "9e307", "int-1e400", "past-a-turn"],
    )
    def test_malformed_az_padding_raises(self, site, ecdfs_field, search_time, bad):
        """A padding that is not a real number in [0, 360] is a ValueError that names it."""
        with pytest.raises(ValueError, match="az_padding") as excinfo:
            plan_constant_el_scan(
                field=ecdfs_field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=search_time,
                rising=True,
                angle=170.0,
                az_padding=bad,
            )
        assert not isinstance(excinfo.value, PointingError)

    @pytest.mark.parametrize(
        ("name", "value", "lsa_window"),
        [
            pytest.param("az_padding", Decimal("0.0"), None, id="az_padding"),
            pytest.param("velocity", Decimal("0.5"), None, id="velocity"),
            pytest.param("az_accel", Decimal("1.0"), None, id="az_accel"),
            pytest.param("elevation", Decimal("50.0"), None, id="elevation"),
            pytest.param("max_search_hours", Decimal("12.0"), None, id="max_search_hours"),
            pytest.param("step_seconds", Decimal("30.0"), None, id="step_seconds"),
            pytest.param(
                "field",
                FieldRegion(
                    ra_center=53.117,
                    dec_center=-27.808,
                    width=Decimal("5.0"),
                    height=Decimal("6.7"),
                ),
                None,
                id="field",
            ),
            # A window inside the E-CDF-S crossing pass, which spans LST 5.8 to 12.6 deg.
            pytest.param(
                "max_search_hours", Decimal("12.0"), (7.0, 12.0), id="lsa-max_search_hours"
            ),
            pytest.param("step_seconds", Decimal("30.0"), (7.0, 12.0), id="lsa-step_seconds"),
        ],
    )
    def test_a_decimal_argument_plans_as_its_float(
        self, site, ecdfs_field, search_time, ecdfs_block, name, value, lsa_window
    ):
        """A ``Decimal`` argument, or a field sized in ``Decimal``, plans as the float does.

        The search, by elevation crossing or by sidereal window, the leg
        quantiser and the Sun sweep compute with the float each check
        returns, so a ``Decimal`` never meets float arithmetic. The shared
        block is the crossing pass planned from floats.
        """
        kwargs = dict(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            angle=170.0,
        )
        if lsa_window is None:
            kwargs["rising"] = True
            expected = ecdfs_block
        else:
            kwargs["lsa_window"] = lsa_window
            expected = plan_constant_el_scan(**kwargs)
        block = plan_constant_el_scan(**{**kwargs, name: value})
        assert block.computed_params == expected.computed_params
        assert block.config == expected.config
        np.testing.assert_array_equal(block.trajectory.az, expected.trajectory.az)

    def test_a_full_turn_of_padding_reaches_the_azimuth_limits(
        self, site, ecdfs_field, search_time
    ):
        """360 deg is the widest padding accepted, and its range exceeds the azimuth limits."""
        with pytest.raises(AzimuthBoundsError):
            plan_constant_el_scan(
                field=ecdfs_field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=search_time,
                rising=True,
                angle=170.0,
                az_padding=360.0,
            )

    def test_science_mask_excludes_turnarounds(self, ecdfs_block):
        traj = ecdfs_block.trajectory
        assert traj.scan_flag is not None
        science = traj.science_mask
        assert science.sum() < traj.n_points
        assert science.sum() > traj.n_points * 0.5


# The branch-placement tests sweep this field west of the meridian at el 40, at 2 deg/s
# and 1 deg/s^2 (so each turnaround overshoots its edge by 2.5 deg), over a sidereal window.
_BRANCH_FIELD = FieldRegion(ra_center=70.0, dec_center=-10.0, width=50.0, height=60.0)
_BRANCH_START = Time("2026-09-15T00:00:00", scale="utc")
_BRANCH_TIMESTEPS = np.linspace(0.8, 1.2, 81)


def _plan_branch_case(site, lsa_window, **kwargs) -> ScanBlock:
    """Plan the branch-placement field over ``lsa_window``."""
    return plan_constant_el_scan(
        field=_BRANCH_FIELD,
        elevation=40.0,
        velocity=2.0,
        site=site,
        start_time=_BRANCH_START,
        lsa_window=lsa_window,
        **kwargs,
    )


def _tune_apex_past_limit(site, lsa_window, *, high, shift, offset_deg):
    """Return a padding and timestep that put one turnaround apex just past an azimuth limit.

    The planner's own range for the field, planned once without padding, is
    padded so that, moved by ``shift``, the apex of the turnaround at its
    high (``high``) or low edge, moved by ``offset_deg`` (the boresight's
    azimuth minus the swept range's), lies past the limit on that side by
    half the distance the nearest sample stops short of it (at most
    ``3 a dt**2 / 16``), by the pattern's own arithmetic. Of the candidate
    timesteps the one that leaves both halves widest is taken, so the case
    holds whatever the range. Returns ``(unpadded_range, padding, timestep,
    depth, inside)``: how far the apex lies past the limit and the nearest
    sample inside it, in degrees.
    """
    bare = _plan_branch_case(site, lsa_window, timestep=1.0).computed_params
    lo0, hi0 = bare["az_start"], bare["az_stop"]
    assert 180.0 < lo0 < hi0 < 360.0
    duration = _compute_ce_duration_from_lsa(lsa_window, Coordinates(site), _BRANCH_START)[2]
    limits = site.telescope_limits.azimuth
    overshoot = turnaround_overshoot_deg(2.0, 1.0)
    sign = 1.0 if high else -1.0
    limit = limits.max if high else limits.min
    edge = (hi0 if high else lo0) + shift

    def place(depth, timestep):
        padding = sign * (limit - edge - offset_deg) - overshoot + depth
        lo, hi = lo0 - padding + shift, hi0 + padding + shift
        _, built = _quantize_ce_duration(
            az_throw=hi - lo, velocity=2.0, duration=duration, az_accel=1.0
        )
        sample = _ce_az_reach(lo, hi, 2.0, 1.0, built, timestep)[1 if high else 0] + offset_deg
        return padding, sign * (limit - sample)

    best = None
    for timestep in _BRANCH_TIMESTEPS:
        # The sample stops short of the apex by depth + inside; split that gap evenly.
        depth = 3.0 * 1.0 * timestep**2 / 32.0
        _, inside = place(depth, timestep)
        depth = 0.5 * (depth + inside)
        padding, inside = place(depth, timestep)
        if best is None or min(depth, inside) > min(best[3], best[4]):
            best = ((lo0, hi0), padding, float(timestep), depth, inside)
    return best


class TestCEGeometryWrapHandling:
    """The RA = 0 and az = 0/360 wraps, and a pass's branch at the azimuth limits.

    The azimuth wrap is reached at FYST's -23 deg latitude by any source that
    transits north of the zenith, that is any declination between the site
    latitude (-22.99 deg) and the +67 deg above which nothing rises. The RA
    wrap cases include fields wider than half a turn of RA and fields spanning
    a whole turn or more.
    """

    @pytest.mark.parametrize(
        ("rising", "expected"),
        [(True, (22.337, 31.937)), (False, (328.063, 337.663))],
    )
    def test_az_range_handles_north_transit(self, site, rising, expected):
        """``_compute_ce_az_range`` returns a contiguous range for north-transiting sources.

        At FYST (lat = -22.99 deg), a 4 x 4 deg field at Dec +35 transits
        north of the zenith, its corners culminating between el 30.0 and
        34.0. At el 25 every corner crosses, east of north as the field
        rises and west of it as it sets, so each side's range is one
        contiguous interval of about 10 deg on that side, padded by 0.5 deg.
        """
        coords = Coordinates(site)
        field = FieldRegion(ra_center=0.0, dec_center=35.0, width=4.0, height=4.0)
        az_min, az_max, _, _ = _compute_ce_az_range(
            field,
            0.0,
            25.0,
            rising,
            coords,
            Time("2026-09-16T04:30:00", scale="utc"),
            0.5,
            velocity=0.5,
            az_accel=1.0,
            duration=1200.0,
            timestep=0.1,
        )
        assert (az_min, az_max) == pytest.approx(expected, abs=1e-3)
        side = (0.0, 180.0) if rising else (180.0, 360.0)
        assert side[0] < az_min < az_max < side[1]

    def test_az_range_normal_field_unchanged(self, site):
        """A field crossing well away from the meridian gets a modest range inside [0, 360).

        The rotated E-CDF-S box of the docstring example rises through el 50
        east of the meridian; with 0.5 deg of padding its range is
        (99.95, 112.61), placed on the branch it lies on.
        """
        coords = Coordinates(site)
        field = FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7)
        az_min, az_max, _, _ = _compute_ce_az_range(
            field,
            170.0,
            50.0,
            True,
            coords,
            Time("2026-03-15T17:20:00", scale="utc"),
            0.5,
            velocity=0.5,
            az_accel=1.0,
            duration=1650.0,
            timestep=0.1,
        )
        assert (az_min, az_max) == pytest.approx((99.952, 112.614), abs=1e-3)
        assert 0.0 <= az_min < az_max < 180.0

    def test_ra_wrap_handled_for_field_near_ra_zero(self, site):
        """``_compute_ce_duration`` correctly identifies edges for RA ~ 0 fields.

        A 3-deg-wide field centred at RA = 1 deg has corners at RA ~ -0.5 deg
        and ~ +2.5 deg. After ``% 360``, naive ``min``/``max`` would return
        the wrong leading/trailing edges. The corners are taken as unwrapped
        offsets from the field centre, so the edges stay -0.5 and 2.5 deg.
        """
        coords = Coordinates(site)
        # Use a southern-hemisphere field that transits comfortably
        # above the requested elevation at FYST (lat = -23 deg).
        field = FieldRegion(ra_center=1.0, dec_center=-25.0, width=3.0, height=3.0)
        # Pick a search start time before the field rises through el=40
        # (transit happens ~03:30 UTC at this RA on this date).
        base_time = Time("2026-09-15T00:00:00", scale="utc")

        t_start, t_end, duration = _compute_ce_duration(
            field,
            angle=0.0,
            elevation=40.0,
            coords_obj=coords,
            base_search_time=base_time,
            rising=True,
        )

        # A naive ``min(ra_vals)``/``max(ra_vals)`` would be 2.5 and 359.5;
        # the leading edge would be searched at RA = 359.5 (which crosses
        # el=40 hours later than the true RA = -0.5 edge) and the reported
        # duration would be a few-hour overestimate. Unwrapped, the duration
        # is the short interval between the two true RA edges crossing
        # el=40.
        assert duration > 0
        assert duration < 30 * 60  # 30 min; the true value is ~13 min for a 3 deg width

    @pytest.mark.parametrize("argument", ["max_search_hours", "step_seconds"])
    def test_compute_ce_duration_rejects_a_non_positive_horizon_or_step(self, site, argument):
        coords = Coordinates(site)
        field = FieldRegion(ra_center=1.0, dec_center=-25.0, width=3.0, height=3.0)
        base_time = Time("2026-09-15T00:00:00", scale="utc")
        with pytest.raises(ValueError, match=f"{argument} must be positive"):
            _compute_ce_duration(
                field,
                angle=0.0,
                elevation=40.0,
                coords_obj=coords,
                base_search_time=base_time,
                rising=True,
                **{argument: 0},
            )

    def test_north_transit_planning_succeeds(self, site):
        """End-to-end: ``plan_constant_el_scan`` works for a north-transiting source.

        The 2 x 2 deg field at Dec +35 sets through el 30 just west of north,
        its border meeting that elevation over (341.63, 350.01): a short
        throw on the west side, not one wrapped round the 0/360 azimuth.
        """
        field = FieldRegion(ra_center=0.0, dec_center=35.0, width=2.0, height=2.0)
        block = plan_constant_el_scan(
            field=field,
            elevation=30.0,
            velocity=0.5,
            site=site,
            start_time=Time("2026-09-16T04:30:00", scale="utc"),
            rising=False,
            angle=0.0,
        )
        cp = block.computed_params
        assert (cp["az_start"], cp["az_stop"]) == pytest.approx((341.627, 350.005), abs=1e-3)
        assert cp["az_throw"] < 30.0

    def test_setting_north_crossing_plans_in_range(self, site):
        """A setting scan whose range ends at the northern meridian plans inside the limits.

        The field's northern corners, at Dec +36, culminate at el 31.0, short
        of el 31.9, so they never reach it and the setting range runs to the
        meridian: (347.59, 360.0). The turnarounds overshoot 360 deg, so the
        planned trajectory must come out on the in-limits near-zero branch,
        (-12.41, 0.0), not on the equivalent branch extending past the 360 deg
        azimuth limit.
        """
        field = FieldRegion(ra_center=180.0, dec_center=35.0, width=4.0, height=2.0)
        block = plan_constant_el_scan(
            field=field,
            elevation=31.9,
            velocity=0.5,
            site=site,
            start_time=Time("2027-03-15T04:19:00", scale="utc"),
            rising=False,
            angle=0.0,
        )
        lim = site.telescope_limits.azimuth
        assert block.trajectory.az.min() >= lim.min
        assert block.trajectory.az.max() <= lim.max
        cp = block.computed_params
        assert cp["az_start"] == pytest.approx(-12.411, abs=1e-3)
        assert cp["az_stop"] == 0.0

    def test_setting_north_crossing_az_range_within_limits(self, site):
        """``_compute_ce_az_range`` keeps a range that ends at the meridian within limits.

        Same regime as the end-to-end test above, exercised at the helper level:
        the setting range runs to the northern meridian, (345.63, 362.0) with
        the 2 deg padding, and the returned interval must be the 360 deg branch
        that fits the telescope range, (-14.37, 2.0), not the equivalent one
        extending past 360.
        """
        coords = Coordinates(site)
        field = FieldRegion(ra_center=180.0, dec_center=35.0, width=4.0, height=2.0)
        t_start, _, duration = _compute_ce_duration(
            field,
            angle=0.0,
            elevation=31.9,
            coords_obj=coords,
            base_search_time=Time("2026-09-15T16:13:00", scale="utc"),
            rising=False,
        )
        az_min, az_max, _, _ = _compute_ce_az_range(
            field,
            0.0,
            31.9,
            False,
            coords,
            t_start,
            2.0,
            velocity=0.5,
            az_accel=1.0,
            duration=duration,
            timestep=0.1,
        )
        lim = site.telescope_limits.azimuth
        assert az_min >= lim.min
        assert az_max <= lim.max
        assert az_min == pytest.approx(345.627 - 360.0, abs=1e-3)
        assert az_max == 2.0

    def test_setting_padding_overflow_az_range_within_limits(self, site):
        """Azimuth padding alone must not push the returned interval past 360.

        A wide field at Dec +20.85 sets through el 40 with every corner
        crossing it, its border meeting that elevation over (316.30, 358.62),
        short of the northern meridian; its turnarounds at 0.5 deg/s reach
        358.78, inside the limit, but the 2 deg padding pushes the maximum to
        360.62. The returned interval must sit on the branch that fits the
        telescope range.
        """
        coords = Coordinates(site)
        field = FieldRegion(ra_center=0.0, dec_center=20.85, width=12.0, height=12.0)
        t_start, _, duration = _compute_ce_duration(
            field,
            angle=0.0,
            elevation=40.0,
            coords_obj=coords,
            base_search_time=Time("2026-09-15T05:21:00", scale="utc"),
            rising=False,
        )
        args = (field, 0.0, 40.0, False, coords, t_start)
        raw_min, raw_max = _ce_field_az_range(*args, 0.0)
        assert (raw_min, raw_max) == pytest.approx((316.298, 358.622), abs=1e-3)
        az_min, az_max, _, _ = _compute_ce_az_range(
            *args,
            2.0,
            velocity=0.5,
            az_accel=1.0,
            duration=duration,
            timestep=0.1,
        )
        lim = site.telescope_limits.azimuth
        assert az_min >= lim.min
        assert az_max <= lim.max
        assert (az_min, az_max) == pytest.approx((raw_min - 362.0, raw_max - 358.0), abs=1e-9)

    def test_rising_north_crossing_control_on_near_zero_branch(self, site):
        """Rising-pass control: the already-in-range branch is not shifted.

        Same field, elevation, and anchor as the setting regression above but
        rising = True. The rising range starts at the northern meridian, 0 deg,
        and runs east to 12.41 deg; its turnarounds reach -0.16 deg, inside the
        telescope limits, so the branch placement must leave it on the
        near-zero branch; a spurious whole-turn shift would move the endpoints
        by 360 deg.
        """
        field = FieldRegion(ra_center=180.0, dec_center=35.0, width=4.0, height=2.0)
        block = plan_constant_el_scan(
            field=field,
            elevation=31.9,
            velocity=0.5,
            site=site,
            start_time=Time("2027-03-15T04:19:00", scale="utc"),
            rising=True,
            angle=0.0,
        )
        lim = site.telescope_limits.azimuth
        az = block.trajectory.az
        assert az.min() >= lim.min
        assert az.max() <= lim.max
        cp = block.computed_params
        assert cp["az_start"] == 0.0
        assert cp["az_stop"] == pytest.approx(12.411, abs=1e-3)

    def test_a_setting_pass_takes_the_branch_that_holds_its_turnarounds(self, site):
        """The azimuth branch is chosen for the turnaround envelope, not the padded range.

        A 2 x 2 deg field at Dec +25.8 sets through el 40 just west of north,
        its border meeting that elevation over (342.25, 355.59). The
        turnarounds overshoot that range by 5 v^2 / (8 a) on each side:
        0.16 deg at 0.5 deg/s, which stays below the 360 deg limit, and
        5.625 deg at 3 deg/s, which reaches 361.21, so the faster pass sweeps
        the same range one turn down, where its envelope (-23.38, 1.21) is
        inside the limits.
        """
        field = FieldRegion(ra_center=60.0, dec_center=25.8, width=2.0, height=2.0)
        start = Time("2026-11-15T18:00:00", scale="utc")
        slow, fast = (
            plan_constant_el_scan(
                field=field,
                elevation=40.0,
                velocity=velocity,
                site=site,
                start_time=start,
                rising=False,
            )
            for velocity in (0.5, 3.0)
        )
        assert slow.computed_params["az_start"] == pytest.approx(342.249, abs=1e-3)
        assert slow.computed_params["az_stop"] == pytest.approx(355.588, abs=1e-3)
        for key in ("az_start", "az_stop"):
            assert fast.computed_params[key] == slow.computed_params[key] - 360.0
        assert fast.trajectory.az.max() == pytest.approx(1.213, abs=0.01)
        lim = site.telescope_limits.azimuth
        for block in (slow, fast):
            assert lim.min <= block.trajectory.az.min()
            assert block.trajectory.az.max() <= lim.max

    def test_an_off_axis_module_takes_the_branch_its_boresight_holds(self, site):
        """The setting pass above, swept by the detector at 150 deg at 2.2 deg/s, moves a turn down.

        The detector's turnaround envelope, (342.25, 355.59) widened by 3.03 deg
        on each side, ends below 360 deg, but at el 40 the boresight sits
        2.29 deg higher in azimuth than the detector, so its envelope ends at
        360.90; the pass is placed one turn down, where both fit.
        """
        block = plan_constant_el_scan(
            field=FieldRegion(ra_center=60.0, dec_center=25.8, width=2.0, height=2.0),
            elevation=40.0,
            velocity=2.2,
            site=site,
            start_time=Time("2026-11-15T18:00:00", scale="utc"),
            rising=False,
            detector_offset=off_axis_offset(150),
        )
        assert block.computed_params["az_start"] == pytest.approx(342.249 - 360.0, abs=1e-3)
        assert block.computed_params["az_stop"] == pytest.approx(355.588 - 360.0, abs=1e-3)
        assert block.trajectory.az.max() == pytest.approx(0.90, abs=0.01)
        lim = site.telescope_limits.azimuth
        assert lim.min <= block.trajectory.az.min()
        assert block.trajectory.az.max() <= lim.max

    def test_the_commanded_range_must_fit_as_well(self, site):
        """The pattern checks the detector's range and the builder the boresight, so both must fit.

        At 3 deg/s, swept by the detector at -30 deg, the range's turnaround
        envelope ends at 361.21 deg, past the limit, while the boresight,
        2.29 deg lower, ends at 358.92, inside it; the pass still moves a turn
        down, since the pattern checks the detector's samples, which pass the
        limit, before the offset is applied.
        """
        block = plan_constant_el_scan(
            field=FieldRegion(ra_center=60.0, dec_center=25.8, width=2.0, height=2.0),
            elevation=40.0,
            velocity=3.0,
            site=site,
            start_time=Time("2026-11-15T18:00:00", scale="utc"),
            rising=False,
            detector_offset=off_axis_offset(-30),
        )
        assert block.computed_params["az_start"] == pytest.approx(342.249 - 360.0, abs=1e-3)
        assert block.computed_params["az_stop"] == pytest.approx(355.588 - 360.0, abs=1e-3)
        lim = site.telescope_limits.azimuth
        assert lim.min <= block.trajectory.az.min()
        assert block.trajectory.az.max() <= lim.max

    def test_a_pass_is_placed_where_its_samples_fit(self, site):
        """Where no branch holds the whole sweep, its samples, which stop short of an apex, decide.

        The field's border meets el 40 west of the meridian; over this
        sidereal window the pass makes three legs at 2 deg/s, so it turns at
        both edges, 2.5 deg past each. Swept by the detector at -30 deg, whose
        boresight sits about 2.29 deg lower in azimuth at el 40, the padding
        (:func:`_tune_apex_past_limit`) puts the boresight's low apex, one turn
        down, just below -180 deg while its samples stop short of it, inside
        the limit; on its own branch the range runs past 360 deg. So no branch
        holds the whole sweep, the branch a turn down holds its samples (the
        placement's grade 1) and its own branch does not (grade 0): the pass
        moves a turn down and plans. Measured margins: the apex lies 0.10 deg
        past -180 and the nearest sample 0.10 deg inside it.
        """
        offset = off_axis_offset(-30)
        bore = _boresight_az_offset(40.0, offset, site)
        lsa_window = (100.0, 101.25)
        (lo0, hi0), padding, timestep, depth, inside = _tune_apex_past_limit(
            site, lsa_window, high=False, shift=-360.0, offset_deg=bore
        )
        assert min(depth, inside) > 0.05, "no timestep leaves the apex and its samples apart"
        block = _plan_branch_case(
            site, lsa_window, az_padding=padding, timestep=timestep, detector_offset=offset
        )
        cp = block.computed_params
        lim = site.telescope_limits.azimuth
        built = float(block.trajectory.times[-1])
        assert cp["n_scans"] == 3, "the pass no longer turns at both edges"
        # Its own branch: the commanded samples run past 360 deg.
        assert _ce_az_reach(lo0 - padding, hi0 + padding, 2.0, 1.0, built, timestep)[1] > lim.max
        # A turn down: the boresight's low apex lies past -180, its samples inside.
        apex_lo = swept_az_envelope(lo0 - padding - 360.0, hi0 + padding - 360.0, 2.0, 1.0)[0]
        assert apex_lo + bore < lim.min, "the boresight's whole path fits a turn down"
        assert cp["az_start"] == pytest.approx(lo0 - padding - 360.0, abs=1e-9)
        assert cp["az_stop"] == pytest.approx(hi0 + padding - 360.0, abs=1e-9)
        assert lim.min <= block.trajectory.az.min()
        assert block.trajectory.az.max() <= lim.max

    def test_a_pass_takes_the_branch_that_holds_all_of_it(self, site):
        """Where one branch holds the whole sweep and another only its samples, the first is taken.

        The field's border meets el 40 west of the meridian; over this
        sidereal window the pass makes two legs at 2 deg/s, so it turns once,
        2.5 deg past the high edge. The padding (:func:`_tune_apex_past_limit`)
        puts that apex just past 360 deg while the samples stop short of it,
        inside the limit (the placement's grade 1 on its own branch), but the
        mount runs the path between them too, so the pass moves a turn down,
        where the whole sweep fits (grade 2). Measured margins: the apex lies
        0.13 deg past 360 and the nearest sample 0.13 deg inside it.
        """
        lsa_window = (100.0, 100.73)
        (lo0, hi0), padding, timestep, depth, inside = _tune_apex_past_limit(
            site, lsa_window, high=True, shift=0.0, offset_deg=0.0
        )
        assert min(depth, inside) > 0.05, "no timestep leaves the apex and its samples apart"
        block = _plan_branch_case(site, lsa_window, az_padding=padding, timestep=timestep)
        cp = block.computed_params
        lim = site.telescope_limits.azimuth
        overshoot = turnaround_overshoot_deg(2.0, 1.0)
        built = float(block.trajectory.times[-1])
        assert cp["n_scans"] == 2, "the pass no longer turns at one edge only"
        # Its own branch: the samples fit, the apex between them does not.
        sample_hi = _ce_az_reach(lo0 - padding, hi0 + padding, 2.0, 1.0, built, timestep)[1]
        assert sample_hi <= lim.max < hi0 + padding + overshoot
        # A turn down the whole sweep fits.
        assert lim.min <= lo0 - padding - 360.0
        assert hi0 + padding - 360.0 + overshoot <= lim.max
        assert cp["az_start"] == pytest.approx(lo0 - padding - 360.0, abs=1e-9)
        assert cp["az_stop"] == pytest.approx(hi0 + padding - 360.0, abs=1e-9)
        assert lim.min <= block.trajectory.az.min()
        assert block.trajectory.az.max() <= lim.max

    def test_a_pass_stays_where_the_mount_path_fits(self, site):
        """The detector's own path can pass a limit that the mount's does not; the pass stays.

        The pass above, swept by the detector at -30 deg: the detector's path
        still turns just past 360 deg and its samples stop short of it, but the
        boresight, which the mount follows, sits about 2.29 deg lower in
        azimuth at el 40, so its whole path fits (the placement's grade 2 on
        its own branch). The pass keeps its branch. Measured margins: the
        boresight's apex lies 2.16 deg inside the limit, the detector's apex
        0.13 deg past it and its nearest sample 0.13 deg inside it.
        """
        offset = off_axis_offset(-30)
        bore = _boresight_az_offset(40.0, offset, site)
        lsa_window = (100.0, 100.73)
        (lo0, hi0), padding, timestep, depth, inside = _tune_apex_past_limit(
            site, lsa_window, high=True, shift=0.0, offset_deg=0.0
        )
        assert min(depth, inside) > 0.05, "no timestep leaves the apex and its samples apart"
        block = _plan_branch_case(
            site, lsa_window, az_padding=padding, timestep=timestep, detector_offset=offset
        )
        cp = block.computed_params
        lim = site.telescope_limits.azimuth
        overshoot = turnaround_overshoot_deg(2.0, 1.0)
        built = float(block.trajectory.times[-1])
        assert cp["n_scans"] == 2, "the pass no longer turns at one edge only"
        # The detector's samples fit and its path does not; the boresight's path fits.
        sample_hi = _ce_az_reach(lo0 - padding, hi0 + padding, 2.0, 1.0, built, timestep)[1]
        assert sample_hi <= lim.max < hi0 + padding + overshoot
        assert hi0 + padding + overshoot + bore <= lim.max
        assert cp["az_start"] == pytest.approx(lo0 - padding, abs=1e-9)
        assert cp["az_stop"] == pytest.approx(hi0 + padding, abs=1e-9)
        assert lim.min <= block.trajectory.az.min()
        assert block.trajectory.az.max() <= lim.max

    def test_a_module_the_boresight_solve_cannot_place_is_refused_by_the_build(self, site):
        """Where the boresight solve fails, the branch is left to the commanded range.

        The sidereal window fixes the elevation without a crossing search, so
        the pass reaches the build at el 90, where the builder's solve fails
        and its refusal names the samples. The field at Dec -23 holds the
        site's zenith declination, so its border reaches el 90.
        """
        with pytest.raises(OffsetInversionError) as excinfo:
            plan_constant_el_scan(
                field=FieldRegion(ra_center=0.0, dec_center=-23.0, width=4.0, height=4.0),
                elevation=90.0,
                velocity=0.5,
                site=site,
                start_time=Time("2026-09-15T00:00:00", scale="utc"),
                lsa_window=(0.0, 5.0),
                detector_offset=get_primecam_offset("IM2"),
                timestep=1.0,
            )
        assert excinfo.value.indices

    def test_a_module_whose_boresight_solve_does_not_converge_is_refused(self, site):
        """At el 88.5, inside the elevation limits, IM2's boresight solve does not converge."""
        with pytest.raises(OffsetInversionError, match="failed to converge"):
            plan_constant_el_scan(
                field=FieldRegion(ra_center=0.0, dec_center=-23.0, width=4.0, height=4.0),
                elevation=88.5,
                velocity=0.5,
                site=site,
                start_time=Time("2026-09-15T00:00:00", scale="utc"),
                lsa_window=(0.0, 5.0),
                detector_offset=get_primecam_offset("IM2"),
                timestep=1.0,
            )

    @pytest.mark.parametrize(
        ("ra_center", "start"),
        [(180.0, "2026-11-15T18:00:00"), (10.0, "2026-11-15T12:00:00")],
    )
    def test_a_field_wider_than_half_a_turn_of_ra_keeps_its_own_width(self, site, ra_center, start):
        """A field 200 deg of RA wide is timed by its own edges, wherever it is centred.

        Rising through el 50 at Dec -27.8, the pass runs from the leading RA
        edge's crossing to the trailing edge's, 200 deg of RA later, so it
        lasts 200/360 of a sidereal day (13.30 h) to within the minute or so
        that apparent place moves the two edges apart. Centred at RA 10, the
        corners lie either side of RA 0/360, and taken modulo 360 they would
        bound the 160 deg complement, whose pass is the rest of the sidereal
        day; this anchor falls inside that pass, about 4 h before the field's
        own pass opens.
        """
        dec = -27.8
        field = FieldRegion(
            ra_center=ra_center,
            dec_center=dec,
            width=200.0 * math.cos(math.radians(dec)),
            height=2.0,
        )
        block = plan_constant_el_scan(
            field=field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=Time(start, scale="utc"),
            rising=True,
            max_search_hours=36.0,
            # The pass runs through the day; the Sun screens are not the subject.
            sun_safe=lambda az, el, t: True,
        )
        cp = block.computed_params
        crossing = Time(cp["end_time_iso"], scale="utc") - Time(cp["start_time_iso"], scale="utc")
        assert crossing.sec == pytest.approx(200.0 / 360.0 * _SIDEREAL_DAY_SEC, abs=120.0)

    @pytest.mark.parametrize(("dec", "ra_span"), [(-27.8, 370.0), (-43.3, 360.0)])
    def test_a_field_spanning_a_whole_turn_of_ra_is_refused(self, site, dec, ra_span):
        """A field a whole turn of RA wide or more has no RA edges to time a pass by.

        At 370 deg its edges are 10 deg of RA apart on the sky, the trailing
        one past the leading one's meridian; a pass running from one to the
        other would last more than a sidereal day, which no single crossing
        describes. At Dec -43.3 the corners of a field 360 deg of RA wide span
        359.99999999999994 deg, a whole turn to rounding.
        """
        field = FieldRegion(
            ra_center=0.0,
            dec_center=dec,
            width=ra_span * math.cos(math.radians(dec)),
            height=2.0,
        )
        with pytest.raises(PointingError, match="a whole turn or more"):
            plan_constant_el_scan(
                field=field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=Time("2026-11-15T22:00:00", scale="utc"),
                rising=True,
                max_search_hours=36.0,
            )

    @pytest.mark.parametrize(
        ("start", "refusal"),
        [
            ("2026-11-15T12:00:00", "Could not find elevation crossing"),
            ("2026-11-16T04:30:00", "different passes"),
        ],
    )
    def test_a_pass_longer_than_the_search_says_so(self, site, start, refusal):
        """A refusal names a pass that cannot fit in the search horizon.

        The 200 deg field's pass lasts about 13.3 h, longer than the default
        12 h ``max_search_hours``, so no anchor finds it whole: from 12:00 the
        trailing edge crosses after the horizon ends, and from 04:30 the next
        day, inside the pass, the trailing edge crosses before the leading
        edge comes round again.
        """
        dec = -27.8
        field = FieldRegion(
            ra_center=10.0,
            dec_center=dec,
            width=200.0 * math.cos(math.radians(dec)),
            height=2.0,
        )
        with pytest.raises(PointingError, match=refusal) as exc_info:
            plan_constant_el_scan(
                field=field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=Time(start, scale="utc"),
                rising=True,
            )
        message = str(exc_info.value)
        assert "its pass lasts about 13.3 h, longer than max_search_hours=12.0" in message


class TestPlanConstantElLsaWindow:
    """``lsa_window`` places the pass by sidereal angle, not by an elevation crossing.

    Covers no-wrap, wrap-around, equal-endpoint validation, not-found
    handling, the refusal of ``rising`` beside the window, and that an
    explicit ``None`` plans exactly what omitting the argument does.
    """

    @pytest.fixture
    def deep56_field(self):
        """Deep56-like field centred near RA = 0, dec ~ -2 deg.

        A full Deep56-style patch spans 60 deg in RA (23:00 -> 03:00 wraps)
        and 14 deg in Dec, and plans at el = 50 in every window these tests
        plan (``plan_constant_el_scan``'s own example plans it). These tests
        shrink it to 4 x 4 deg because the sidereal window alone fixes the
        timing they pin; the field only sets the azimuth range, which must
        fit the limits.
        """
        return FieldRegion(ra_center=0.0, dec_center=-2.0, width=4.0, height=4.0)

    @pytest.fixture
    def deep56_search_time(self):
        """Search anchor for Deep56 LSA tests."""
        return Time("2026-09-15T00:00:00", scale="utc")

    def test_lsa_window_no_wrap(self, site, deep56_field, deep56_search_time):
        """A no-wrap window (22 deg, 82 deg) produces a 4-hour scan.

        Duration = (82 - 22) / 15 = 4 h. The start lies between
        ``start_time`` and ``start_time + max_search_hours``.
        """
        block = plan_constant_el_scan(
            field=deep56_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=deep56_search_time,
            lsa_window=(22.0, 82.0),
        )

        assert block.duration > 0
        # Reported obs_end - obs_start (from computed_params duration)
        # tracks the LSA-derived 4-hour duration; the recomputed
        # n_scans-driven ``actual_duration`` is allowed to differ by
        # at most a leg/turnaround pair.
        obs_start = Time(block.computed_params["start_time_iso"], scale="utc")
        obs_end = Time(block.computed_params["end_time_iso"], scale="utc")
        lsa_duration_h = (obs_end - obs_start).to_value(u.hour)
        assert lsa_duration_h == pytest.approx(4.0, abs=1e-3)

        # Start lies inside the search horizon.
        assert obs_start >= deep56_search_time
        assert obs_start <= deep56_search_time + 12.0 * u.hour

    def test_lsa_window_with_wrap(self, site, deep56_field, deep56_search_time):
        """A wrap-around window (310 deg, 10 deg) produces a 4-hour scan."""
        block = plan_constant_el_scan(
            field=deep56_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=deep56_search_time,
            lsa_window=(310.0, 10.0),
        )

        obs_start = Time(block.computed_params["start_time_iso"], scale="utc")
        obs_end = Time(block.computed_params["end_time_iso"], scale="utc")
        lsa_duration_h = (obs_end - obs_start).to_value(u.hour)
        # (10 - 310) mod 360 / 15 = 60 / 15 = 4 h.
        assert lsa_duration_h == pytest.approx(4.0, abs=1e-3)

        # LST at obs_start should be very close to 310 deg.
        coords = Coordinates(site)
        lst_at_start = coords.get_lst(obs_start)
        diff = (lst_at_start - 310.0 + 180.0) % 360.0 - 180.0
        assert abs(diff) < 0.05  # 0.05 deg ~ 12 s of sidereal time

    def test_lsa_window_deep56_pattern(self, site, deep56_field, deep56_search_time):
        """Exercise the two elevation-50 Deep56 LSA windows, one wrap-around and one not."""
        block_early = plan_constant_el_scan(
            field=deep56_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=deep56_search_time,
            lsa_window=(310.0, 10.0),
        )
        block_late = plan_constant_el_scan(
            field=deep56_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=deep56_search_time,
            lsa_window=(22.0, 82.0),
        )

        # Both windows are 4 hours wide.
        for block in (block_early, block_late):
            obs_start = Time(block.computed_params["start_time_iso"], scale="utc")
            obs_end = Time(block.computed_params["end_time_iso"], scale="utc")
            assert (obs_end - obs_start).to_value(u.hour) == pytest.approx(4.0, abs=1e-3)

        # The 22->82 window starts after the 310->10 one (both anchored at
        # the same start_time; 310 deg comes up first, then 22 deg).
        t_early = Time(block_early.computed_params["start_time_iso"], scale="utc")
        t_late = Time(block_late.computed_params["start_time_iso"], scale="utc")
        assert t_early < t_late

    def test_lsa_window_equal_endpoints_raises(self, site, deep56_field, deep56_search_time):
        """Equal endpoints produce a zero-duration window, refused."""
        with pytest.raises(ValueError, match="zero-duration"):
            plan_constant_el_scan(
                field=deep56_field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=deep56_search_time,
                lsa_window=(45.0, 45.0),
            )

    def test_lsa_window_out_of_range_raises(self, site, deep56_field, deep56_search_time):
        """Endpoints outside [0, 360) are rejected."""
        with pytest.raises(ValueError, match=r"\[0, 360\)"):
            plan_constant_el_scan(
                field=deep56_field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=deep56_search_time,
                lsa_window=(-10.0, 30.0),
            )

    @pytest.mark.parametrize("window", [(330.0,), (330.0, 350.0, 5.0), "x"])
    def test_lsa_window_that_is_not_a_pair_raises(
        self, site, deep56_field, deep56_search_time, window
    ):
        """A window of one or three values is refused, not read in part."""
        with pytest.raises(ValueError, match="lsa_window must be a pair"):
            plan_constant_el_scan(
                field=deep56_field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=deep56_search_time,
                lsa_window=window,
            )

    def test_lsa_window_not_found_raises(self, site, deep56_field):
        """A tiny search horizon that never crosses min_lsa raises PointingError.

        Pick a start_time where LST is just past 100 deg and search only
        0.5 hours forward (~ 7.5 deg of LSA travel) for a target of
        100 deg, the increasing-direction crossing will already be in
        the past.
        """
        coords = Coordinates(site)
        # The first time LST lies just past 100.5 deg, so the next 0.5 h spans
        # LSA ~100.5-108 deg and never re-crosses 100 deg.
        anchor = Time("2026-09-15T00:00:00", scale="utc")
        dt = np.arange(0, 24 * 3600, 30.0)
        times = anchor + dt * u.s
        lsa = np.asarray(coords.get_lst(times))
        idx_arr = np.flatnonzero((lsa > 100.5) & (lsa < 105.0))
        assert len(idx_arr) > 0
        start_time = times[idx_arr[0]]

        with pytest.raises(PointingError, match="not reached in increasing direction"):
            plan_constant_el_scan(
                field=deep56_field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=start_time,
                lsa_window=(100.0, 200.0),
                max_search_hours=0.5,
            )

    @pytest.mark.parametrize("rising", [True, False])
    def test_lsa_window_rejects_rising(self, site, deep56_field, deep56_search_time, rising):
        """``rising`` beside ``lsa_window`` is refused, not silently ignored.

        The sidereal window fixes the timing, and the side of the meridian
        whose range is swept is the one the field centre stands on at the
        window's middle, so a ``rising`` flag beside it would either repeat
        that choice or contradict it. Refusing the combination is the honest
        contract; the summary names the sidereal window instead (see
        ``test_lsa_window_summary_names_the_window``).
        """
        with pytest.raises(ValueError, match="rising is not accepted with lsa_window"):
            plan_constant_el_scan(
                field=deep56_field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=deep56_search_time,
                rising=rising,
                lsa_window=(22.0, 82.0),
            )

    def test_lsa_window_summary_names_the_window(self, site, deep56_field, deep56_search_time):
        """An LSA-windowed block's summary claims no crossing half."""
        block = plan_constant_el_scan(
            field=deep56_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=deep56_search_time,
            lsa_window=(22.0, 82.0),
        )

        assert "LSA-window pass:" in block.summary
        assert "Rising pass" not in block.summary
        assert "Setting pass" not in block.summary

    @pytest.mark.parametrize(
        ("rising", "label"), [(None, "Rising pass"), (True, "Rising pass"), (False, "Setting pass")]
    )
    def test_crossing_summary_names_the_half(self, site, rising, label):
        """On the elevation-crossing path the summary still names the half.

        ``rising=None`` is the omitted-argument default and means the
        rising crossing, exactly as an explicit ``rising=True`` does.
        """
        ecdfs_field = FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7)
        kwargs = {} if rising is None else {"rising": rising}
        block = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=Time("2026-03-15T17:00:00", scale="utc"),
            angle=170.0,
            **kwargs,
        )

        assert label in block.summary

    def test_lsa_window_none_equals_omitting_it(self, site):
        """Default ``lsa_window=None`` is byte-identical to omitting the kwarg.

        Guards against accidental coupling between the LSA branch and
        the elevation-crossing path.
        """
        ecdfs_field = FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7)
        search_time = Time("2026-03-15T17:00:00", scale="utc")
        block_default = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=True,
            angle=170.0,
        )
        block_explicit_none = plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=search_time,
            rising=True,
            angle=170.0,
            lsa_window=None,
        )

        cp_default = block_default.computed_params
        cp_explicit = block_explicit_none.computed_params
        assert cp_default["start_time_iso"] == cp_explicit["start_time_iso"]
        assert cp_default["end_time_iso"] == cp_explicit["end_time_iso"]
        assert cp_default["az_start"] == cp_explicit["az_start"]
        assert cp_default["az_stop"] == cp_explicit["az_stop"]
        assert cp_default["n_scans"] == cp_explicit["n_scans"]
        assert cp_default["duration"] == cp_explicit["duration"]
        assert block_default.duration == block_explicit_none.duration
        assert np.array_equal(block_default.trajectory.az, block_explicit_none.trajectory.az)
        assert np.array_equal(block_default.trajectory.el, block_explicit_none.trajectory.el)

    def test_lsa_window_at_lst_zero_crossing(self, site):
        """``min_lsa = 0`` works when LST is currently just below 360 deg.

        Regression for the wrap-edge straddle detection: with a
        ``min_lsa = 0`` and consecutive samples ``(359.9, 0.1)``, a plain
        ``(lsa - 0) * (lsa_next - 0) < 0`` product test fails
        because both factors are positive (the 0/360 boundary is not a
        true sign change in the wrapped-modulo representation). The
        unwrapped LSA series locates the crossing correctly and the
        planner produces a valid 4-hour scan starting at LST ~ 0 deg.

        The field is positioned at RA = 60 deg so the LST = 0 crossing
        doesn't place the field at meridian (which would put the field
        near zenith for low |dec| and break the az-range computation).
        """
        # Find an anchor at which LST is currently ~358 deg so the first
        # increasing crossing of 0 deg (== 360 deg unwrapped) lies within a
        # short search horizon.
        coords = Coordinates(site)
        sample_anchor = Time("2026-09-15T00:00:00", scale="utc")
        dt = np.arange(0, 24 * 3600, 30.0)
        times = sample_anchor + dt * u.s
        lsa = np.asarray(coords.get_lst(times))
        # First index where LSA is in [358, 359.5], gives ~2-7 minutes
        # before the wrap crossing.
        idx_arr = np.flatnonzero((lsa > 358.0) & (lsa < 359.5))
        assert len(idx_arr) > 0
        start_time = times[idx_arr[0]]

        # Field at RA = 60 deg, dec = -25 deg: when LST crosses 0 deg the field
        # is at HA = -60 deg (rising side), well clear of meridian/zenith.
        field = FieldRegion(ra_center=60.0, dec_center=-25.0, width=4.0, height=4.0)

        block = plan_constant_el_scan(
            field=field,
            elevation=30.0,
            velocity=0.5,
            site=site,
            start_time=start_time,
            lsa_window=(0.0, 60.0),
        )

        obs_start = Time(block.computed_params["start_time_iso"], scale="utc")
        obs_end = Time(block.computed_params["end_time_iso"], scale="utc")
        lsa_duration_h = (obs_end - obs_start).to_value(u.hour)
        # (60 - 0) / 15 = 4 hours.
        assert lsa_duration_h == pytest.approx(4.0, abs=1e-3)

        # LST at obs_start should be very close to 0 deg (== 360 deg).
        lst_at_start = coords.get_lst(obs_start)
        diff = (lst_at_start - 0.0 + 180.0) % 360.0 - 180.0
        assert abs(diff) < 0.05  # 0.05 deg ~ 12 s of sidereal time

    def test_lsa_window_min_just_above_zero(self, site):
        """``min_lsa ~ 0.0001`` is in the wrap dead-zone.

        With a plain straddle test ``(lsa - 0.0001) * (lsa_next - 0.0001) < 0``,
        consecutive samples like ``(359.9, 0.1)`` produce a negative product
        only if both factors have opposite sign, but both are positive after
        the ``% 360`` wrap. The unwrapped series recognises the crossing.

        Uses the helper directly to keep the test focused on the
        wrap-detection geometry (the end-to-end planner is exercised by
        ``test_lsa_window_at_lst_zero_crossing``).
        """
        coords = Coordinates(site)
        sample_anchor = Time("2026-09-15T00:00:00", scale="utc")
        dt = np.arange(0, 24 * 3600, 30.0)
        times = sample_anchor + dt * u.s
        lsa = np.asarray(coords.get_lst(times))
        idx_arr = np.flatnonzero((lsa > 358.0) & (lsa < 359.5))
        assert len(idx_arr) > 0
        base_time = times[idx_arr[0]]

        t_start, t_end, duration = _compute_ce_duration_from_lsa(
            lsa_window=(0.0001, 60.0001),
            coords_obj=coords,
            base_search_time=base_time,
        )
        # 60 deg LSA window / 15 deg per hr = 4 h = 14400 s exactly; abs=1e-6 s absorbs round-off.
        assert duration == pytest.approx(4.0 * 3600.0, abs=1e-6)
        # The crossing of LST = 0.0001 deg lies just past the wrap.
        lst_at_start = coords.get_lst(t_start)
        diff = (lst_at_start - 0.0001 + 180.0) % 360.0 - 180.0
        assert abs(diff) < 0.05

    # The one test in this file that runs the simulator tier's dispatcher;
    # marking the test rather than the module keeps the planning tests in the
    # library-tier job.
    @pytest.mark.offline
    def test_lsa_window_json_roundtrip(self, site, deep56_field, deep56_search_time):
        """``lsa_window`` as a *list* (post-JSON shape) works through the dispatcher.

        ECSV serialisation in :func:`fyst_trajectories.overhead.io.write_timeline`
        passes ``scan_params`` through ``json.dumps``, which converts
        Python tuples to JSON arrays. The corresponding ``json.loads`` on
        read returns lists, so a round-tripped ``lsa_window`` is
        ``[310.0, 10.0]`` (list), not the tuple form. The simulation
        dispatcher must accept both and produce identical trajectories.
        """
        import json

        from fyst_trajectories.overhead.models import ObservingPatch, TimelineBlock
        from fyst_trajectories.overhead.simulation import _generate_trajectory_for_block

        scan_params_tuple = {"lsa_window": (310.0, 10.0)}
        # Simulate ECSV write -> read: tuple becomes list via JSON.
        scan_params_list = json.loads(json.dumps(scan_params_tuple))
        assert isinstance(scan_params_list["lsa_window"], list)

        patch_tuple = ObservingPatch(
            name="deep56_tuple",
            ra_center=deep56_field.ra_center,
            dec_center=deep56_field.dec_center,
            width=deep56_field.width,
            height=deep56_field.height,
            scan_type="constant_el",
            velocity=0.5,
            elevation=50.0,
            scan_params=scan_params_tuple,
        )
        patch_list = ObservingPatch(
            name="deep56_list",
            ra_center=deep56_field.ra_center,
            dec_center=deep56_field.dec_center,
            width=deep56_field.width,
            height=deep56_field.height,
            scan_type="constant_el",
            velocity=0.5,
            elevation=50.0,
            scan_params=scan_params_list,
        )

        # A 4-hour duration matches the LSA-derived window; az_start / az_end
        # are placeholders for the SCIENCE block factory (the LSA-window
        # branch recomputes both internally).
        t_start = deep56_search_time
        block_tuple = TimelineBlock.science(
            patch=patch_tuple,
            t_start=t_start,
            duration=4.0 * 3600.0,
            az_start=0.0,
            az_end=10.0,
            el=50.0,
            scan_index=0,
        )
        block_list = TimelineBlock.science(
            patch=patch_list,
            t_start=t_start,
            duration=4.0 * 3600.0,
            az_start=0.0,
            az_end=10.0,
            el=50.0,
            scan_index=0,
        )

        scan_tuple = _generate_trajectory_for_block(block_tuple, site)
        scan_list = _generate_trajectory_for_block(block_list, site)

        assert (
            scan_tuple.computed_params["start_time_iso"]
            == (scan_list.computed_params["start_time_iso"])
        )
        assert (
            scan_tuple.computed_params["end_time_iso"]
            == (scan_list.computed_params["end_time_iso"])
        )
        assert scan_tuple.computed_params["duration"] == scan_list.computed_params["duration"]
        assert np.array_equal(scan_tuple.trajectory.az, scan_list.trajectory.az)

    def test_lsa_window_narrow_window(self, site, deep56_field, deep56_search_time):
        """A narrow 0.5 deg window produces a valid 2-minute scan."""
        block = plan_constant_el_scan(
            field=deep56_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=deep56_search_time,
            lsa_window=(50.0, 50.5),
        )
        # 0.5 deg / 15 = 0.0333 h = 120 s.
        obs_start = Time(block.computed_params["start_time_iso"], scale="utc")
        obs_end = Time(block.computed_params["end_time_iso"], scale="utc")
        assert (obs_end - obs_start).to_value(u.s) == pytest.approx(120.0, abs=1e-2)
        # Trajectory has positive duration and a non-empty az array.
        assert block.duration > 0
        assert block.trajectory.az.size > 0

    def test_lsa_window_invalid_max_search_hours(self, site, deep56_field, deep56_search_time):
        """Non-positive ``max_search_hours`` is rejected up front."""
        with pytest.raises(ValueError, match="max_search_hours"):
            plan_constant_el_scan(
                field=deep56_field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=deep56_search_time,
                lsa_window=(22.0, 82.0),
                max_search_hours=0.0,
            )
        with pytest.raises(ValueError, match="max_search_hours"):
            plan_constant_el_scan(
                field=deep56_field,
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=deep56_search_time,
                lsa_window=(22.0, 82.0),
                max_search_hours=-1.0,
            )

    def test_lsa_window_long_duration_warning(self, site, deep56_search_time):
        """Sustained > 6 h LSA windows emit a ``PointingWarning``.

        Exercises the helper directly: the warning is an
        LSA-duration-only signal, independent of field geometry, and a
        wide-enough planner call would also need its azimuth range to
        fit inside telescope limits (a separate concern).
        """
        coords = Coordinates(site)
        # 120 deg / 15 = 8 hours, past the 6 h threshold.
        with pytest.warns(PointingWarning, match="long"):
            _compute_ce_duration_from_lsa(
                lsa_window=(22.0, 142.0),
                coords_obj=coords,
                base_search_time=deep56_search_time,
                max_search_hours=24.0,
            )

    def test_lsa_window_sun_safety_recheck_invoked(
        self, site, deep56_field, deep56_search_time, monkeypatch
    ):
        """``_check_field_sun_safety`` is called twice in the LSA branch.

        The first call is at ``start_time`` (search anchor); the second
        is at the LSA-resolved ``obs_start``. The two times should be
        distinct in this scenario because the LSA window starts > 1 h
        after ``deep56_search_time``.
        """
        from fyst_trajectories.planning import constant_el as ce_module

        recorded: list[Time] = []
        original = ce_module._check_field_sun_safety

        def spy(ra, dec, t, s, sun_safe=None):
            recorded.append(t)
            return original(ra, dec, t, s, sun_safe=sun_safe)

        monkeypatch.setattr(ce_module, "_check_field_sun_safety", spy)

        plan_constant_el_scan(
            field=deep56_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=deep56_search_time,
            lsa_window=(22.0, 82.0),
        )

        assert len(recorded) == 2, f"expected 2 sun-safety calls, got {len(recorded)}"
        # The two calls should be at distinct times (start_time vs.
        # resolved obs_start, which is hours later for this window).
        dt_sec = (recorded[1] - recorded[0]).to_value(u.s)
        assert dt_sec > 60.0, (
            f"obs_start should differ from start_time by minutes-to-hours, got {dt_sec} s"
        )

    def test_lsa_window_sun_safety_single_call_on_the_crossing_path(self, site, monkeypatch):
        """Without ``lsa_window``, only one sun-safety call fires.

        Guards the gating property: re-checking sun safety is
        gated on the LSA branch (a regression that would also call the
        check at obs_start in the elevation-crossing path is the kind
        of surprise we want to catch early).
        """
        from fyst_trajectories.planning import constant_el as ce_module

        recorded: list[Time] = []
        original = ce_module._check_field_sun_safety

        def spy(ra, dec, t, s, sun_safe=None):
            recorded.append(t)
            return original(ra, dec, t, s, sun_safe=sun_safe)

        monkeypatch.setattr(ce_module, "_check_field_sun_safety", spy)

        ecdfs_field = FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7)
        plan_constant_el_scan(
            field=ecdfs_field,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=Time("2026-03-15T17:00:00", scale="utc"),
            rising=True,
            angle=170.0,
        )

        assert len(recorded) == 1


class TestConstantElSweptSunCheck:
    """The Sun screen sweeps the whole resolved pass, not just its opening.

    A constant-elevation block can run for hours while the Sun closes at
    about 15 degrees per hour. Screening only the field centre at the search
    anchor (and, on the LSA path, at the resolved opening) leaves the rest of
    the pass unscreened. The planner screens the built boresight trajectory
    over the whole run.
    """

    _FIELD = FieldRegion(ra_center=180.0, dec_center=-30.0, width=20.0, height=10.0)
    _ANCHOR = Time("2026-12-21T00:00:00", scale="utc")

    def _plan(self, site, **overrides):
        kwargs = dict(
            field=self._FIELD,
            elevation=50.0,
            velocity=0.5,
            site=site,
            start_time=self._ANCHOR,
            rising=True,
            angle=170.0,
        )
        kwargs.update(overrides)
        return plan_constant_el_scan(**kwargs)

    def test_sweep_reaches_the_end_of_the_pass(self, site):
        """The sampled times span the block, so its last sample is screened."""
        seen: list[Time] = []

        def spy(az, el, t):
            seen.append(t)
            return True

        block = self._plan(site, sun_safe=spy)
        cp = block.computed_params
        t_start = Time(cp["start_time_iso"], scale="utc")
        t_last = get_absolute_times(block.trajectory)[-1]
        sweep = Time([t.jd for t in seen], format="jd", scale="utc")
        assert sweep.min().unix <= t_start.unix + 1e-6
        # The spy's instants pass through a single-float Julian date, good to
        # tens of microseconds.
        assert sweep.max().unix >= t_last.unix - 1e-3

    def test_a_pass_that_ends_inside_the_zone_warns(self, site):
        """A late Sun approach is caught even though the opening is clear.

        On 2026-12-21 this field's boresight starts 87 deg from the Sun and
        closes to 67 deg by the end of its 99-minute pass, so an exclusion
        radius of 75 deg is clear at the opening and violated at the close.
        """
        from fyst_trajectories import get_fyst_site

        narrow = get_fyst_site(sun_exclusion_radius=75.0, sun_warning_radius=80.0)
        with pytest.warns(PointingWarning, match="EXCLUSION ZONE"):
            block = self._plan(narrow)

        # The opening itself is clear, so the field-centre pre-flight alone
        # would have been silent: this is the sweep talking, not the anchor.
        coords = Coordinates(narrow)
        cp = block.computed_params
        t_start = Time(cp["start_time_iso"], scale="utc")
        az_mid = 0.5 * (cp["az_start"] + cp["az_stop"])
        sun_az, sun_el = coords.get_sun_altaz(t_start)
        assert float(coords.angular_separation(az_mid, 50.0, sun_az, sun_el)) > 75.0

    def test_a_late_unsafe_verdict_from_an_injected_model_warns(self, site):
        """An injected model that turns unsafe late in the pass is honoured.

        The scalar case above depends on real geometry; this one pins the
        mechanism directly: a predicate that answers ``True`` for the first
        half of the run and ``False`` afterwards must still produce a
        warning, which it cannot if only the opening is consulted.
        """
        clear = self._plan(site)
        cp = clear.computed_params
        t_start = Time(cp["start_time_iso"], scale="utc")
        midpoint = t_start.unix + 0.5 * cp["duration"]

        def late_unsafe(az, el, t):
            return t.unix < midpoint

        with pytest.warns(PointingWarning, match="EXCLUSION ZONE"):
            self._plan(site, sun_safe=late_unsafe)

        # An always-clear predicate is silent over the same pass, so the warning
        # above comes from the late verdict, not from this pass's own geometry.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._plan(site, sun_safe=lambda az, el, t: True)
        assert not [w for w in caught if "EXCLUSION ZONE" in str(w.message)]

    def test_each_field_form_advisory_keeps_its_attribution(self, site):
        """The centre pre-flight names the planner; the other field-form advisories the caller.

        The other three are the swept-pass screen and the two long-pass
        advisories: a 110 deg wide field that crosses el 50 deg over 8.4 h, and
        an LSA window of 120 deg, an 8 h block.
        """
        long_pass = dict(
            elevation=50.0,
            velocity=1.0,
            site=site,
            start_time=Time("2026-06-15T03:00:00", scale="utc"),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan_constant_el_scan(
                FieldRegion(358.0, -2.0, 5.0, 5.0),
                elevation=40.0,
                velocity=0.5,
                site=site,
                start_time=Time("2026-03-20T10:00:00", scale="utc"),
            )
            plan_constant_el_scan(FieldRegion(0.0, -30.0, 110.0, 10.0), **long_pass)
            plan_constant_el_scan(
                FieldRegion(0.0, -30.0, 10.0, 10.0),
                lsa_window=(300.0, 60.0),
                max_search_hours=24.0,
                **long_pass,
            )
        labels = {
            "Field center passes": "centre",
            "planned constant-elevation": "sweep",
            "Computed observation duration": "crossing",
            "LSA-derived duration": "lsa",
        }
        files = sorted(
            (
                next(label for text, label in labels.items() if text in str(w.message)),
                Path(w.filename).name,
            )
            for w in caught
        )
        here = Path(__file__).name
        assert files == [
            ("centre", "constant_el.py"),
            ("crossing", here),
            ("lsa", here),
            ("sweep", here),
        ]

    @pytest.mark.parametrize(
        ("field", "elevation", "velocity", "anchor", "module", "warns"),
        [
            # A tall field whose throw of 64 deg puts the Sun between the envelope's
            # edges and its middle near the end of the pass, 0.6 deg above the horizon.
            (
                FieldRegion(ra_center=45.0, dec_center=-10.0, width=10.0, height=40.0),
                45.0,
                1.0,
                "2026-06-15T09:00:00",
                None,
                True,
            ),
            # IM6 sits below the boresight, so the boresight scans 1.66 deg higher
            # than the detector's elevation, and nearer the Sun.
            (
                FieldRegion(ra_center=42.0, dec_center=0.0, width=2.0, height=2.0),
                50.0,
                0.5,
                "2026-03-15T13:00:00",
                "IM6",
                True,
            ),
            # IM3 sits above it, so the boresight scans lower and stays clear.
            (
                FieldRegion(ra_center=41.0, dec_center=0.0, width=2.0, height=2.0),
                50.0,
                0.5,
                "2026-03-15T13:00:00",
                "IM3",
                False,
            ),
        ],
        ids=["tall-field", "IM6", "IM3"],
    )
    def test_the_screen_judges_every_sample_of_the_boresight_path(
        self, site, field, elevation, velocity, anchor, module, warns
    ):
        """The field form warns exactly when a sample of the built trajectory is inside the zone.

        The advisory names the closest sample, so its distance is the minimum
        separation of the boresight path from the Sun over every sample.
        """
        kwargs = dict(
            elevation=elevation,
            velocity=velocity,
            site=site,
            start_time=Time(anchor, scale="utc"),
            detector_offset=None if module is None else get_primecam_offset(module),
        )
        if warns:
            with pytest.warns(PointingWarning, match="EXCLUSION ZONE: planned constant") as rec:
                block = plan_constant_el_scan(field, **kwargs)
            (message,) = [str(w.message) for w in rec if "EXCLUSION ZONE" in str(w.message)]
        else:
            block = plan_constant_el_scan(field, **kwargs)

        closest = float(_sun_separation_of_every_sample(site, block.trajectory).min())
        radius = site.sun_avoidance.exclusion_radius
        assert (closest <= radius) == warns
        if warns:
            stated = float(message.split(" passes ")[1].split(" deg")[0])
            assert stated == pytest.approx(closest, abs=0.06)


class TestAnchorAtOrInsideAPass:
    """A field-form anchor at or after the pass's opening is refused, and the refusal says why.

    The search looks forward for the leading edge's crossing; from inside a
    pass that edge has already crossed, so the next crossing is about a
    sidereal day later, beyond the search.
    """

    _FIELD = FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7)

    def _plan(self, site, start_time):
        return plan_constant_el_scan(
            self._FIELD, elevation=50.0, velocity=0.5, site=site, start_time=start_time
        )

    def test_an_anchor_just_before_the_opening_returns_the_pass(self, site):
        """One millisecond before the opening, the same pass is found.

        The search grid starts from the anchor, so the interpolated crossings
        move in their last bits; the pass and its legs do not.
        """
        opened = self._plan(site, Time("2026-03-15T17:00:00", scale="utc"))
        opening = Time(opened.computed_params["start_time_iso"], scale="utc")

        again = self._plan(site, opening - TimeDelta(1e-3, format="sec"))

        before, after = opened.computed_params, again.computed_params
        for key in ("start_time_iso", "end_time_iso", "n_scans"):
            assert after[key] == before[key]
        for key in ("az_start", "az_stop", "duration"):
            assert after[key] == pytest.approx(before[key], abs=1e-9)

    @pytest.mark.parametrize("into_pass", [0.0, 0.5], ids=["opening", "mid-pass"])
    def test_an_anchor_inside_the_pass_is_refused_as_inside_it(self, site, into_pass):
        """At the opening and mid-pass the refusal names the anchor as inside the pass."""
        opened = self._plan(site, Time("2026-03-15T17:00:00", scale="utc"))
        opening = Time(opened.computed_params["start_time_iso"], scale="utc")
        anchor = opening + TimeDelta(into_pass * opened.duration, format="sec")

        with pytest.raises(PointingError, match="falls inside a pass") as err:
            self._plan(site, anchor)

        assert anchor.iso in str(err.value)
        assert "about a sidereal day later" in str(err.value)

    def test_a_field_never_below_the_elevation_is_not_called_inside_a_pass(self, site):
        """A field above the elevation all day has no crossing, and the refusal does not claim one.

        At Dec -88 the field stays between about 21 and 25 deg from FYST, so it
        never rises through el 20.
        """
        with pytest.raises(PointingError, match="Could not find elevation crossing") as err:
            plan_constant_el_scan(
                FieldRegion(ra_center=0.0, dec_center=-88.0, width=2.0, height=2.0),
                elevation=20.0,
                velocity=0.5,
                site=site,
                start_time=Time("2026-03-15T00:00:00", scale="utc"),
            )

        assert "inside a pass" not in str(err.value)


class TestLsaSearchStepValidation:
    """The LSA crossing search validates the step it walks forward on.

    Unvalidated, a zero step divides by zero building the sample grid and a
    negative one produces an empty grid, which the search then reports as
    "no crossing found" over a horizon it never sampled.
    """

    @pytest.fixture
    def search_time(self):
        """Search anchor for the step-validation calls."""
        return Time("2026-09-15T00:00:00", scale="utc")

    def test_zero_step_is_refused(self, site, search_time):
        """A zero step raises a message naming the argument."""
        with pytest.raises(ValueError, match="step_seconds must be positive"):
            _compute_ce_duration_from_lsa(
                lsa_window=(22.0, 82.0),
                coords_obj=Coordinates(site),
                base_search_time=search_time,
                step_seconds=0.0,
            )

    def test_negative_step_is_refused(self, site, search_time):
        """A negative step raises rather than silently searching nothing."""
        with pytest.raises(ValueError, match="step_seconds must be positive"):
            _compute_ce_duration_from_lsa(
                lsa_window=(22.0, 82.0),
                coords_obj=Coordinates(site),
                base_search_time=search_time,
                step_seconds=-30.0,
            )


def _ra_dec_box(ra_lo: float, ra_hi: float, dec_lo: float, dec_hi: float) -> FieldRegion:
    """Return the FieldRegion of an RA/Dec box, its width measured at the central Dec."""
    dec_c = 0.5 * (dec_lo + dec_hi)
    return FieldRegion(
        ra_center=0.5 * (ra_lo + ra_hi),
        dec_center=dec_c,
        width=(ra_hi - ra_lo) * math.cos(math.radians(dec_c)),
        height=dec_hi - dec_lo,
    )


# Orion A: a 6 x 2.5 deg box turned by -63.2 deg about RA 05:38:28.55, Dec -06:57:02.34.
_ORION_A = FieldRegion(
    ra_center=(5 + 38 / 60 + 28.55 / 3600) * 15.0,
    dec_center=-(6 + 57 / 60 + 2.34 / 3600),
    width=6.0,
    height=2.5,
)

# name: (field, angle, elevation, rising, the range's two ends to a tenth of a degree, the
# coarse time step of the trace in seconds). The ends only centre the trace's azimuth
# windows, which the trace checks it did not clip; they are not what is asserted.
_BORDER_CASES = {
    # COSMOS: RA 09:50:30 to 10:10:30, Dec +00:05 to +04:15.
    "cosmos-43-rising": (
        _ra_dec_box(147.625, 152.625, 5.0 / 60.0, 4.25),
        0.0,
        43.0,
        True,
        (59.8, 66.8),
        60.0,
    ),
    # Deep56: RA 23:00 to 03:00, Dec -09 to +05, a field 60 deg long in RA.
    "deep56-50-setting": (
        _ra_dec_box(-15.0, 45.0, -9.0, 5.0),
        0.0,
        50.0,
        False,
        (284.1, 311.1),
        120.0,
    ),
    "orion-a-45-rising": (_ORION_A, -63.2, 45.0, True, (71.0, 81.2), 60.0),
    "orion-a-45-setting": (_ORION_A, -63.2, 45.0, False, (278.8, 289.0), 60.0),
}
_BORDER_NIGHT = Time("2026-11-15T00:00:00", scale="utc")

# The trace's fine grid: 2 s in time and 0.002 deg in azimuth.
_TRACE_FINE_STEP_SEC = 2.0
_TRACE_FINE_AZ_STEP_DEG = 0.002


@functools.cache
def _border_block(name: str, atmosphere: bool = False) -> ScanBlock:
    """Plan one of the border cases with the planner's default padding, vacuum or refracted."""
    field, angle, elevation, rising, _, _ = _BORDER_CASES[name]
    return plan_constant_el_scan(
        field,
        elevation=elevation,
        velocity=0.5,
        site=get_fyst_site(),
        start_time=_BORDER_NIGHT,
        rising=rising,
        angle=angle,
        atmosphere=AtmosphericConditions.for_fyst() if atmosphere else None,
    )


def _pass_span(block: ScanBlock, margin_sec: float = 2700.0) -> tuple[Time, Time]:
    """Return the planner's timed pass widened by ``margin_sec`` on each side."""
    cp = block.computed_params
    margin = TimeDelta(margin_sec, format="sec")
    return (
        Time(cp["start_time_iso"], scale="utc") - margin,
        Time(cp["end_time_iso"], scale="utc") + margin,
    )


def _in_box(ra, dec, field: FieldRegion, angle: float) -> np.ndarray:
    """Return which RA/Dec points lie in the box, by the inverse of the planner's corner rule."""
    a = math.radians(angle)
    cos_dec = math.cos(math.radians(field.dec_center))
    dx = ((np.asarray(ra) - field.ra_center + 180.0) % 360.0 - 180.0) * cos_dec
    dy = np.asarray(dec) - field.dec_center
    x = dx * math.cos(a) + dy * math.sin(a)
    y = -dx * math.sin(a) + dy * math.cos(a)
    return (np.abs(x) <= 0.5 * field.width) & (np.abs(y) <= 0.5 * field.height)


def _border_mask(coords, field, angle, elevation, times: Time, az: np.ndarray) -> np.ndarray:
    """Return, per time and grid azimuth, whether that point of the elevation circle is in it."""
    n_t, n_az = len(times), az.size
    ra, dec = coords.altaz_to_radec(
        np.tile(az, n_t),
        np.full(n_t * n_az, elevation),
        times[np.repeat(np.arange(n_t), n_az)],
    )
    return _in_box(ra, dec, field, angle).reshape(n_t, n_az)


def _per_time_extreme(mask: np.ndarray, az: np.ndarray, low: bool) -> np.ndarray:
    """Return, per time, the lowest (or highest) in-box grid azimuth; +-inf where none."""
    if low:
        return np.where(mask, az, np.inf).min(axis=1)
    return np.where(mask, az, -np.inf).max(axis=1)


def _trace_end(coords, name: str, times: Time, guess_az: float, low: bool) -> float:
    """Trace one end of the border's azimuths over ``times``, in three passes.

    A coarse pass (0.02 deg in a 1.2 deg window about ``guess_az``) finds
    the extreme over all times; a second (0.002 deg) re-reads every time
    whose coarse value lies within one coarse step of it, since a slowly
    drifting end ties on the coarse grid for hours; a third refines the
    best of those in time (2 s over one coarse step either side, 0.002 deg).
    Each pass checks that its extreme is not at its window's outer edge.
    """
    field, angle, elevation, _, _, coarse_step = _BORDER_CASES[name]
    sign = 1.0 if low else -1.0

    def best(times_, grid):
        per_time = _per_time_extreme(
            _border_mask(coords, field, angle, elevation, times_, grid), grid, low
        )
        k = int(np.argmin(sign * per_time))
        value = float(per_time[k])
        assert grid.min() < value < grid.max()
        return value, per_time, k

    def window(value, below, above):
        # ``below`` deg past the end being traced, ``above`` deg back into the range.
        lo, hi = (value - below, value + above) if low else (value - above, value + below)
        return np.arange(lo, hi, _TRACE_FINE_AZ_STEP_DEG)

    coarse = np.arange(guess_az - 0.6, guess_az + 0.6, 0.02)
    value, per_time, _ = best(times, coarse)
    near = np.flatnonzero(sign * (per_time - value) <= 0.02 + 1e-9)
    value, _, k = best(times[near], window(value, 0.03, 0.004))
    half = coarse_step + 10.0
    fine_times = times[near[k]] + TimeDelta(
        np.arange(-half, half, _TRACE_FINE_STEP_SEC), format="sec"
    )
    value, _, _ = best(fine_times, window(value, 0.2, 0.004))
    return value


def _trace_border_range(coords, name: str, t_from: Time, t_to: Time) -> tuple[float, float]:
    """Trace where the box's border meets the scan elevation over its whole crossing.

    Points of the scan-elevation circle are followed through the crossing
    with ``altaz_to_radec`` and tested for membership in the box, and the
    lowest and highest azimuth any of them takes at any time are returned
    (:func:`_trace_end`, sampled at the case's coarse time step). The
    elevation circle on the pass's side holds no point of the box at
    ``t_from`` or ``t_to``, so the span holds the whole crossing.
    """
    field, angle, elevation, rising, guess, coarse_step = _BORDER_CASES[name]
    side = np.arange(0.0, 180.0, 0.05) + (0.0 if rising else 180.0)
    ends = t_from + TimeDelta([0.0, (t_to - t_from).sec], format="sec")
    assert not _border_mask(coords, field, angle, elevation, ends, side).any()

    times = t_from + TimeDelta(np.arange(0.0, (t_to - t_from).sec, coarse_step), format="sec")
    return (
        _trace_end(coords, name, times, guess[0], low=True),
        _trace_end(coords, name, times, guess[1], low=False),
    )


def _corner_crossing_range(coords, name: str, t_from: Time, t_to: Time) -> tuple[float, float]:
    """Return the azimuths of the box's corners each at its own crossing of the scan elevation.

    Each corner's elevation is sampled every 60 s over the span, the one
    crossing on the pass's side (upward when rising, downward when setting)
    is bracketed and solved with ``brentq`` on the transform's own
    elevation, and the corner's azimuth is read at that instant. The lowest
    and highest of the four azimuths are returned.
    """
    field, angle, elevation, rising, _, _ = _BORDER_CASES[name]
    corners = _field_region_corners(
        field.ra_center, field.dec_center, field.width, field.height, angle
    )
    offsets = np.arange(0.0, (t_to - t_from).sec, 60.0)
    times = t_from + TimeDelta(offsets, format="sec")
    azimuths = []
    for ra, dec in corners:
        _, el = coords.radec_to_altaz(np.full(offsets.size, ra), np.full(offsets.size, dec), times)
        gap = np.asarray(el) - elevation
        if rising:
            found = np.flatnonzero((gap[:-1] < 0.0) & (gap[1:] >= 0.0))
        else:
            found = np.flatnonzero((gap[:-1] > 0.0) & (gap[1:] <= 0.0))
        assert found.size == 1

        def gap_at(seconds, ra=ra, dec=dec):
            at = t_from + TimeDelta(seconds, format="sec")
            return coords.radec_to_altaz(ra, dec, at)[1] - elevation

        k = int(found[0])
        root = brentq(gap_at, offsets[k], offsets[k + 1], xtol=1e-4)
        az, _ = coords.radec_to_altaz(ra, dec, t_from + TimeDelta(root, format="sec"))
        azimuths.append(az)
    return min(azimuths), max(azimuths)


def _near(az: float, ref: float) -> float:
    """Return ``az`` moved by whole turns to within half a turn of ``ref``."""
    return az + 360.0 * round((ref - az) / 360.0)


def _crossing_azimuths(coords, ra, dec, elevation, rising, at: Time) -> np.ndarray:
    """Return each point's azimuth as it crosses the scan elevation, found by a search in time.

    Each RA/Dec point's elevation is sampled every 900 s for one day from
    ``at``; its first crossing on the pass's side (upward when rising,
    downward when setting) is bracketed, bisected to 1 ms on the transform's
    own elevation, and the point's azimuth is read at that instant. A point
    with no such crossing gets NaN.
    """
    ra, dec = np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)
    offsets = np.arange(0.0, 86400.0 + 900.0, 900.0)
    n, m = ra.size, offsets.size
    _, el = coords.radec_to_altaz(
        np.repeat(ra, m), np.repeat(dec, m), at + TimeDelta(np.tile(offsets, n), format="sec")
    )
    gap = np.asarray(el).reshape(n, m) - elevation
    if rising:
        bracket = (gap[:, :-1] < 0.0) & (gap[:, 1:] >= 0.0)
    else:
        bracket = (gap[:, :-1] > 0.0) & (gap[:, 1:] <= 0.0)
    k = bracket.argmax(axis=1)
    lo, hi = offsets[k], offsets[k + 1]
    for _ in range(20):  # 900 s halved 20 times is under 1 ms
        mid = 0.5 * (lo + hi)
        _, el_mid = coords.radec_to_altaz(ra, dec, at + TimeDelta(mid, format="sec"))
        later = (np.asarray(el_mid) < elevation) == rising
        lo, hi = np.where(later, mid, lo), np.where(later, hi, mid)
    az, _ = coords.radec_to_altaz(ra, dec, at + TimeDelta(0.5 * (lo + hi), format="sec"))
    return np.where(bracket.any(axis=1), np.asarray(az, dtype=float), np.nan)


def _box_points(field: FieldRegion, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Return points of the box: 49 along each edge, corners included, and a 3 x 3 interior grid.

    The box is the parallelogram spanned by the planner's corners
    (:func:`_field_region_corners`, which go round it) in unwrapped RA and Dec.
    """
    c0, c1, _, c3 = np.array(
        _field_region_corners(field.ra_center, field.dec_center, field.width, field.height, angle)
    )
    s = np.linspace(0.0, 1.0, 49)
    g = np.linspace(0.25, 0.75, 3)
    along, across = np.concatenate(
        [
            np.stack([s, np.zeros_like(s)]),
            np.stack([s, np.ones_like(s)]),
            np.stack([np.zeros_like(s), s]),
            np.stack([np.ones_like(s), s]),
            np.stack([np.repeat(g, 3), np.tile(g, 3)]),
        ],
        axis=1,
    )
    points = c0 + along[:, None] * (c1 - c0) + across[:, None] * (c3 - c0)
    return points[:, 0], points[:, 1]


class TestCEAzimuthRangeIsTheBorderAtTheElevation:
    """The range is where the field's border meets the scan elevation over its whole crossing.

    On the pass's side of the meridian it runs between the extreme azimuths
    at which points of the box's border cross the scan elevation, and it
    holds every azimuth at which any part of the box is ever at that
    elevation, on the planner's timed pass or outside it.
    """

    # Every azimuth the trace keeps is that of a point of the box at the scan
    # elevation, so the range must hold it, to within one step of the trace's
    # fine azimuth grid (0.002 deg); the step also covers the range reading
    # apparent places at one instant where the trace reads each at its own
    # time (about 1e-4 deg here). The grid stops short of each end by up to
    # that step and by the 2 s time step: near a corner of a turned box the
    # border's meeting point moves along the elevation circle at up to about
    # 0.005 deg/s, so up to 0.01 deg at the 2 s step (unturned boxes meet the
    # elevation along a whole edge of constant Dec, so their ends are reached
    # on a plateau). So the range may exceed the trace by up to 0.02 deg at
    # each end. Deep56's north edge, 60 deg of RA long, crosses 0.012 deg past
    # its corners at el 50 (its apparent Dec bows above theirs near RA 0), so
    # its case fails a range read at the corners alone.
    _EXCESS_TOLERANCE_DEG = 0.02
    _CONTAIN_TOLERANCE_DEG = _TRACE_FINE_AZ_STEP_DEG

    @pytest.mark.parametrize("name", list(_BORDER_CASES))
    def test_the_range_holds_the_traced_border_and_little_more(self, coordinates, name):
        block = _border_block(name)
        traced_lo, traced_hi = _trace_border_range(coordinates, name, *_pass_span(block))
        lo = _near(block.computed_params["az_start"], traced_lo)
        hi = _near(block.computed_params["az_stop"], traced_hi)
        assert lo <= traced_lo + self._CONTAIN_TOLERANCE_DEG
        assert hi >= traced_hi - self._CONTAIN_TOLERANCE_DEG
        assert traced_lo - lo <= self._EXCESS_TOLERANCE_DEG
        assert hi - traced_hi <= self._EXCESS_TOLERANCE_DEG

    @pytest.mark.parametrize("name", list(_BORDER_CASES))
    def test_each_corner_crosses_inside_the_range(self, coordinates, name):
        """Each corner's azimuth, read as that corner crosses, lies inside the range.

        The corners' crossings are solved to 1e-4 s, so the 0.001 deg allowed
        covers only the range reading apparent places at one instant.
        """
        block = _border_block(name)
        corner_lo, corner_hi = _corner_crossing_range(coordinates, name, *_pass_span(block))
        assert _near(block.computed_params["az_start"], corner_lo) <= corner_lo + 0.001
        assert _near(block.computed_params["az_stop"], corner_hi) >= corner_hi - 0.001

    # Fields many degrees long in RA, and one near the south celestial pole, whose
    # border crosses the elevation past every corner: the corners alone fall short of
    # the searched range by the amount given.
    @pytest.mark.parametrize(
        ("field", "elevation", "rising", "corner_shortfall"),
        [
            pytest.param(FieldRegion(0.0, -10.0, 120.0, 4.0), 45.0, True, 0.117, id="120x4-rising"),
            pytest.param(
                FieldRegion(0.0, -10.0, 150.0, 4.0), 45.0, False, 0.174, id="150x4-setting"
            ),
            pytest.param(FieldRegion(0.0, -35.0, 250.0, 6.0), 45.0, True, 0.402, id="250x6-rising"),
            pytest.param(
                FieldRegion(0.0, -86.0, 10.0, 4.0), 23.0, True, 0.111, id="near-pole-rising"
            ),
        ],
    )
    def test_a_long_or_polar_field_reaches_past_its_corners(
        self, coordinates, field, elevation, rising, corner_shortfall
    ):
        """The range is the border's crossings searched in time, not its corners'.

        Each point of the box (:func:`_box_points`) is followed through a day
        and its azimuth read as it crosses (:func:`_crossing_azimuths`). The
        range matches the searched extremes to 0.002 deg (measured within
        5e-4: the search reads 49 points per edge and each point's apparent
        place at its own crossing), and no interior point is more extreme.
        """
        at = Time("2026-06-15T00:00:00", scale="utc")
        ra, dec = _box_points(field, 0.0)
        azimuths = _crossing_azimuths(coordinates, ra, dec, elevation, rising, at)
        assert not np.isnan(azimuths).any()
        searched = (azimuths.min(), azimuths.max())
        corners = azimuths[[0, 48, 49, 97]]  # the first and last points of two opposite edges
        shortfall = (corners.min() - searched[0]) + (searched[1] - corners.max())
        assert shortfall == pytest.approx(corner_shortfall, abs=0.001)
        interior = azimuths[4 * 49 :]
        assert searched[0] < interior.min() and interior.max() < searched[1]
        lo, hi = _ce_field_az_range(field, 0.0, elevation, rising, coordinates, at, 0.0)
        assert lo == pytest.approx(searched[0], abs=0.002)
        assert hi == pytest.approx(searched[1], abs=0.002)

    @pytest.mark.parametrize(
        ("dec", "height", "rising"),
        [(-10.0, 4.0, True), (-35.0, 6.0, False)],
        ids=["dec-10-rising", "dec-35-setting"],
    )
    def test_a_denser_read_of_the_border_barely_moves_the_range(
        self, coordinates, monkeypatch, dec, height, rising
    ):
        """20000 points per edge move a 359 deg wide field's range by less than 1e-4 deg."""
        field = FieldRegion(0.0, dec, 359.0, height)
        at = Time("2026-06-15T03:00:00", scale="utc")
        as_shipped = _ce_field_az_range(field, 0.0, 45.0, rising, coordinates, at, 0.0)
        monkeypatch.setattr(
            "fyst_trajectories.planning._ce_geometry._CE_BORDER_SAMPLES_PER_EDGE", 20000
        )
        dense = _ce_field_az_range(field, 0.0, 45.0, rising, coordinates, at, 0.0)
        assert as_shipped == pytest.approx(dense, abs=1e-4)

    @pytest.mark.parametrize(
        ("field", "angle", "elevation", "rising", "start", "expected"),
        [
            pytest.param(
                *_BORDER_CASES["cosmos-43-rising"][:4],
                _BORDER_NIGHT,
                ("2026-11-15 08:01:44.284", "2026-11-15 08:21:39.692"),
                id="cosmos",
            ),
            pytest.param(
                *_BORDER_CASES["deep56-50-setting"][:4],
                _BORDER_NIGHT,
                ("2026-11-15 02:15:43.581", "2026-11-15 06:15:10.128"),
                id="deep56",
            ),
            pytest.param(
                FieldRegion(ra_center=53.117, dec_center=-27.808, width=5.0, height=6.7),
                170.0,
                50.0,
                True,
                Time("2026-03-15T17:00:00", scale="utc"),
                ("2026-03-15 17:20:43.488", "2026-03-15 17:48:09.281"),
                id="docstring-example",
            ),
        ],
    )
    def test_the_pass_timing_does_not_depend_on_the_range(
        self, site, field, angle, elevation, rising, start, expected
    ):
        """The pass is still timed by the RA edges' crossings, as recorded before the range rule."""
        block = plan_constant_el_scan(
            field,
            elevation=elevation,
            velocity=0.5,
            site=site,
            start_time=start,
            rising=rising,
            angle=angle,
        )
        cp = block.computed_params
        assert (cp["start_time_iso"], cp["end_time_iso"]) == expected

    def test_padding_widens_each_side_by_its_value(self, site):
        field, angle, elevation, rising, _, _ = _BORDER_CASES["cosmos-43-rising"]
        bare, padded = (
            plan_constant_el_scan(
                field,
                elevation=elevation,
                velocity=0.5,
                site=site,
                start_time=_BORDER_NIGHT,
                rising=rising,
                angle=angle,
                az_padding=padding,
            )
            for padding in (0.0, 1.5)
        )
        assert padded.computed_params["az_start"] == pytest.approx(
            bare.computed_params["az_start"] - 1.5, abs=1e-12
        )
        assert padded.computed_params["az_stop"] == pytest.approx(
            bare.computed_params["az_stop"] + 1.5, abs=1e-12
        )

    def test_the_default_padding_is_zero(self):
        assert inspect.signature(plan_constant_el_scan).parameters["az_padding"].default == 0.0

    def test_the_two_forms_build_one_sweep(self, site):
        """The window form, handed the field form's range, start and pass length, builds its sweep.

        COSMOS at el 43 rising: its range lies inside ``[0, 360)`` and is not
        moved by a whole turn, so the window is the range as computed.
        """
        field, angle, elevation, rising, _, _ = _BORDER_CASES["cosmos-43-rising"]
        block = plan_constant_el_scan(
            field,
            elevation=elevation,
            velocity=0.5,
            site=site,
            start_time=_BORDER_NIGHT,
            rising=rising,
            angle=angle,
            az_accel=1.0,
            timestep=0.1,
        )
        cp = block.computed_params
        assert 0.0 <= cp["az_start"] < cp["az_stop"] < 360.0
        pass_length = (
            Time(cp["end_time_iso"], scale="utc") - Time(cp["start_time_iso"], scale="utc")
        ).sec
        window = plan_constant_el_scan(
            az_window=(cp["az_start"], cp["az_stop"]),
            elevation=elevation,
            velocity=0.5,
            duration=pass_length,
            site=site,
            start_time=block.trajectory.start_time,
            az_accel=1.0,
            timestep=0.1,
        )
        assert window.computed_params["n_scans"] == cp["n_scans"]
        for name in ("times", "az", "el", "scan_flag"):
            assert np.array_equal(getattr(window.trajectory, name), getattr(block.trajectory, name))

    # At el 40 a declination culminates at the scan elevation at lat + 50 deg
    # (+27.01) north of the zenith and lat - 50 deg (-72.99) south of it. Each
    # field reaches past that declination, so the part of it beyond never
    # reaches el 40 and the range stops at the meridian on that side.
    @pytest.mark.parametrize(
        ("dec_center", "height", "rising", "end", "meridian"),
        [
            pytest.param(26.0, 4.0, True, "az_start", 0.0, id="north-rising"),
            pytest.param(26.0, 4.0, False, "az_stop", 0.0, id="north-setting"),
            pytest.param(-72.0, 6.0, True, "az_stop", 180.0, id="south-rising"),
            pytest.param(-72.0, 6.0, False, "az_start", 180.0, id="south-setting"),
        ],
    )
    def test_a_declination_short_of_the_elevation_ends_the_range_at_the_meridian(
        self, site, dec_center, height, rising, end, meridian
    ):
        block = plan_constant_el_scan(
            FieldRegion(ra_center=60.0, dec_center=dec_center, width=4.0, height=height),
            elevation=40.0,
            velocity=0.5,
            site=site,
            start_time=_BORDER_NIGHT,
            rising=rising,
        )
        cp = block.computed_params
        assert math.remainder(cp[end] - meridian, 360.0) == 0.0
        # The other end lies strictly on the pass's side.
        lo, hi = cp["az_start"] % 360.0, cp["az_stop"] % 360.0 or 360.0
        assert (0.0 <= lo < hi <= 180.0) if rising else (180.0 <= lo < hi <= 360.0)

    @pytest.mark.parametrize(
        ("lsa_window", "east"),
        [
            # LST 50 at the window's middle puts the centre (RA 60) 10 deg east of the
            # meridian, LST 70 10 deg west of it; each window holds the transit.
            pytest.param((20.0, 80.0), True, id="centre-east-at-the-middle"),
            pytest.param((40.0, 100.0), False, id="centre-west-at-the-middle"),
        ],
    )
    def test_an_lsa_window_takes_the_side_of_the_centre_at_its_middle(self, site, lsa_window, east):
        block = plan_constant_el_scan(
            FieldRegion(ra_center=60.0, dec_center=-25.0, width=4.0, height=4.0),
            elevation=30.0,
            velocity=0.5,
            site=site,
            start_time=_BORDER_NIGHT,
            lsa_window=lsa_window,
        )
        cp = block.computed_params
        lo = cp["az_start"] % 360.0
        hi = lo + cp["az_throw"]
        assert (0.0 < lo < hi < 180.0) if east else (180.0 < lo < hi < 360.0)

    def test_an_lsa_window_over_a_field_that_never_reaches_the_elevation_raises(self, site):
        """The field's south edge, Dec +58, culminates at el 9 at FYST, so none of it reaches 50."""
        with pytest.raises(PointingError, match=r"never crosses el=50\.00 deg"):
            plan_constant_el_scan(
                FieldRegion(ra_center=60.0, dec_center=60.0, width=4.0, height=4.0),
                elevation=50.0,
                velocity=0.5,
                site=site,
                start_time=_BORDER_NIGHT,
                lsa_window=(20.0, 80.0),
            )

    def test_with_refraction_the_border_crosses_at_its_refracted_elevation(self, site):
        """With an atmosphere the ends are where the border's refracted elevation is the scan's.

        COSMOS's edges are 5 deg long, so its ends are crossings of corners
        (as the corner test shows). The refracted ends are matched to the
        corners' refracted crossings to 0.002 deg, and the vacuum range,
        which refraction moves by about 0.008 to 0.009 deg here, misses them
        by more than twice that.
        """
        name = "cosmos-43-rising"
        refracted = _border_block(name, atmosphere=True)
        vacuum = _border_block(name)
        observed = Coordinates(site, atmosphere=AtmosphericConditions.for_fyst())
        corner_lo, corner_hi = _corner_crossing_range(observed, name, *_pass_span(refracted))
        for block, close in ((refracted, True), (vacuum, False)):
            cp = block.computed_params
            misses = (abs(cp["az_start"] - corner_lo), abs(cp["az_stop"] - corner_hi))
            for miss in misses:
                assert (miss <= 0.002) if close else (0.004 < miss < 0.05)
