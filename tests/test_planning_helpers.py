"""Tests for the private planning helpers and the computed-params validator."""

import math

import pytest

from fyst_trajectories.exceptions import PointingError, PointingWarning
from fyst_trajectories.planning import validate_computed_params
from fyst_trajectories.planning._ce_geometry import _field_region_corners, _quantize_ce_duration


class TestFieldRegionCorners:
    """Corner placement: the cos(dec) RA scaling, rotation, and the near-pole refusal."""

    def test_no_rotation(self):
        """With angle=0, corners are axis-aligned around center."""
        corners = _field_region_corners(10.0, -30.0, 4.0, 6.0, 0.0)
        assert len(corners) == 4
        ra_vals = [c[0] for c in corners]
        dec_vals = [c[1] for c in corners]
        # RA offsets are divided by cos(dec) to account for convergence of meridians
        cos_dec = math.cos(math.radians(-30.0))
        assert min(ra_vals) == pytest.approx(10.0 - 2.0 / cos_dec)
        assert max(ra_vals) == pytest.approx(10.0 + 2.0 / cos_dec)
        assert min(dec_vals) == pytest.approx(-33.0)
        assert max(dec_vals) == pytest.approx(-27.0)

    def test_90_degree_rotation_swaps_axes(self):
        corners = _field_region_corners(0.0, 0.0, 4.0, 2.0, 90.0)
        ra_vals = [c[0] for c in corners]
        dec_vals = [c[1] for c in corners]
        # After 90 deg rotation: width (4.0) appears in Dec, height (2.0) in RA
        assert max(abs(r) for r in ra_vals) == pytest.approx(1.0, abs=0.01)
        assert max(abs(d) for d in dec_vals) == pytest.approx(2.0, abs=0.01)

    def test_near_pole_raises(self):
        """A field within ~0.57 deg of a celestial pole raises (cos(dec) -> 0)."""
        with pytest.raises(PointingError, match="too close to celestial pole"):
            _field_region_corners(10.0, 89.5, 4.0, 6.0, 0.0)


class TestValidateComputedParams:
    """Error paths of the computed_params validator.

    Producer-side success paths (each ``plan_*_scan`` invokes the
    validator before returning) are exercised implicitly by every
    other planner test in this package.
    """

    def test_missing_keys_raise_key_error(self):
        """Missing required keys raise KeyError with a helpful message."""
        with pytest.raises(KeyError, match="missing required keys"):
            validate_computed_params({"period": 60.0}, "pong")

    def test_unknown_scan_type_raises(self):
        """An unknown scan_type raises KeyError."""
        with pytest.raises(KeyError, match="Unknown scan_type"):
            validate_computed_params({}, "sidereal")

    def test_extra_keys_emit_warning(self):
        """Extra keys trigger a PointingWarning but do not raise."""
        params = {"duration": 60.0, "extra_key": 1.0}
        with pytest.warns(PointingWarning, match="unexpected keys"):
            validate_computed_params(params, "daisy")

    def test_scan_type_keys_invariant_non_empty(self):
        """Each scan-type's required-key set must be non-empty.

        ``_SCAN_TYPE_TO_KEYS`` derives its entries from each TypedDict's
        ``__required_keys__``, which is non-empty only because the
        planning TypedDicts use the implicit ``total=True``. If a future
        contributor flips one of them to ``total=False`` (or migrates
        keys to ``NotRequired``) without updating the validator, the
        runtime guard would silently accept ``{}``. This test pins the
        invariant so the regression fails here rather than passing silently.
        """
        from fyst_trajectories.planning._types import _SCAN_TYPE_TO_KEYS

        assert _SCAN_TYPE_TO_KEYS, "_SCAN_TYPE_TO_KEYS must not be empty"
        for scan_type, keys in _SCAN_TYPE_TO_KEYS.items():
            assert keys, (
                f"{scan_type} required-key set is empty; the corresponding "
                f"TypedDict was probably flipped to total=False without updating "
                f"validate_computed_params."
            )


class TestQuantizeCEDuration:
    """The shared CE quantiser counts legs against cruise plus turnaround time."""

    @pytest.mark.parametrize(
        "az_throw, velocity, az_accel, duration",
        [
            (2.44, 1.5, 1.5, 300.0),  # fast drag: 1.6 s legs, 2.0 s turnarounds
            (1.3, 0.5, 0.3, 600.0),  # short legs, slow acceleration: turnaround > leg
            (40.0, 1.0, 1.0, 3600.0),  # the offline scheduler's regime: 40 s legs, 2 s turns
            (2.6, 0.05, 1.0, 581.0),  # slow drag: turnaround negligible against a leg
        ],
        ids=["fast-drag", "turn-longer-than-leg", "scheduler-ce", "slow-drag"],
    )
    def test_actual_duration_tracks_the_request(self, az_throw, velocity, az_accel, duration):
        n_scans, actual = _quantize_ce_duration(
            az_throw=az_throw, velocity=velocity, duration=duration, az_accel=az_accel
        )
        t_cruise = az_throw / velocity
        t_turn = 2.0 * velocity / az_accel
        assert n_scans >= 1
        assert actual == pytest.approx(n_scans * t_cruise + (n_scans - 1) * t_turn)
        # Rounding to whole legs moves the window by at most half a leg-plus-turnaround.
        assert abs(actual - duration) <= 0.5 * (t_cruise + t_turn) + 1e-9

    def test_counting_legs_by_cruise_time_alone_would_overshoot(self):
        """A cruise-only leg count turns a 300 s window into 184 legs and 848 s."""
        az_throw, velocity, az_accel, duration = 2.44, 1.5, 1.0, 300.0
        t_cruise = az_throw / velocity
        t_turn = 2.0 * velocity / az_accel
        cruise_only = round(duration / t_cruise)
        assert cruise_only == 184
        assert cruise_only * t_cruise + (cruise_only - 1) * t_turn == pytest.approx(848.3, abs=0.1)

        n_scans, actual = _quantize_ce_duration(
            az_throw=az_throw, velocity=velocity, duration=duration, az_accel=az_accel
        )
        assert n_scans == 65
        assert actual == pytest.approx(297.7, abs=0.1)

    def test_single_leg_minimum(self):
        n_scans, actual = _quantize_ce_duration(
            az_throw=10.0, velocity=1.0, duration=1.0, az_accel=1.0
        )
        assert n_scans == 1
        assert actual == pytest.approx(10.0)


class TestQuantiserDegenerateAndTieCases:
    """Two ways the leg quantiser could answer without saying anything.

    A zero azimuth throw would make every leg instantaneous, so the window
    would fill with turnarounds and the caller would get a plausible duration
    for a scan that sweeps no sky; and a tie between two leg counts would go
    to the *even* one, because the built-in ``round`` is banker's rounding, so
    an exact 2.5 legs would quantise down while an exact 3.5 quantised up.
    """

    @pytest.mark.parametrize("az_throw", [0.0, -1.0])
    def test_non_positive_throw_is_refused(self, az_throw):
        """A scan with no azimuth sweep covers nothing and is rejected."""
        with pytest.raises(ValueError, match="az_throw must be positive"):
            _quantize_ce_duration(az_throw=az_throw, velocity=1.0, duration=30.0, az_accel=1.0)

    @pytest.mark.parametrize(
        "duration, expected",
        # t_cruise = 10 s, t_turnaround = 2 s. legs = (duration + 2) / 12.
        [(28.0, 3), (40.0, 4)],
        ids=["exact-2.5-legs", "exact-3.5-legs"],
    )
    def test_a_tie_rounds_up_not_to_even(self, duration, expected):
        """Both half-leg ties take the larger count, as ``floor(x + 0.5)`` does."""
        n_scans, _ = _quantize_ce_duration(
            az_throw=10.0, velocity=1.0, duration=duration, az_accel=1.0
        )
        assert n_scans == expected
