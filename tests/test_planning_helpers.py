"""Tests for private planning helpers and cross-plan integration."""

import math

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories.exceptions import PointingWarning
from fyst_trajectories.planning import (
    FieldRegion,
    plan_daisy_scan,
    plan_pong_scan,
    validate_computed_params,
)
from fyst_trajectories.planning._ce_geometry import _field_region_corners


class TestFieldRegionCorners:
    """Tests for the _field_region_corners helper."""

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
        """A 90-degree rotation swaps width and height extents."""
        corners = _field_region_corners(0.0, 0.0, 4.0, 2.0, 90.0)
        ra_vals = [c[0] for c in corners]
        dec_vals = [c[1] for c in corners]
        # After 90 deg rotation: width (4.0) appears in Dec, height (2.0) in RA
        assert max(abs(r) for r in ra_vals) == pytest.approx(1.0, abs=0.01)
        assert max(abs(d) for d in dec_vals) == pytest.approx(2.0, abs=0.01)

    def test_near_pole_raises(self):
        """A field within ~0.57 deg of a celestial pole raises (cos(dec) -> 0)."""
        with pytest.raises(ValueError, match="too close to celestial pole"):
            _field_region_corners(10.0, 89.5, 4.0, 6.0, 0.0)


class TestCrossPlanIntegration:
    """Cross-plan tests: the area-mapping planners (pong, daisy) on one field."""

    @pytest.fixture
    def shared_field(self):
        """Field region usable by all three plan functions."""
        return FieldRegion(ra_center=180.0, dec_center=-30.0, width=1.0, height=1.0)

    @pytest.fixture
    def shared_time(self):
        return Time("2026-03-15T04:00:00", scale="utc")

    def test_pong_and_daisy_produce_finite_trajectories(self, site, shared_field, shared_time):
        """The pong and daisy planners produce trajectories with finite az/el values."""
        pong = plan_pong_scan(
            field=shared_field,
            velocity=0.5,
            spacing=0.1,
            num_terms=4,
            site=site,
            start_time=shared_time,
            timestep=0.1,
        )
        daisy = plan_daisy_scan(
            ra=shared_field.ra_center,
            dec=shared_field.dec_center,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            site=site,
            start_time=shared_time,
            timestep=0.1,
            duration=60.0,
        )

        for block in [pong, daisy]:
            assert np.all(np.isfinite(block.trajectory.az))
            assert np.all(np.isfinite(block.trajectory.el))


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
        invariant so the regression fails closed at import-time.
        """
        from fyst_trajectories.planning._types import _SCAN_TYPE_TO_KEYS

        assert _SCAN_TYPE_TO_KEYS, "_SCAN_TYPE_TO_KEYS must not be empty"
        for scan_type, keys in _SCAN_TYPE_TO_KEYS.items():
            assert keys, (
                f"{scan_type} required-key set is empty; the corresponding "
                f"TypedDict was probably flipped to total=False without updating "
                f"validate_computed_params."
            )
