"""Tests for plan_pong_altaz_scan."""

import math

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories.exceptions import PointingWarning, TrajectoryBoundsError
from fyst_trajectories.offsets import InstrumentOffset
from fyst_trajectories.patterns.configs import PongAltAzScanConfig
from fyst_trajectories.planning import (
    PongAltAzComputedParams,
    ScanBlock,
    plan_pong_altaz_scan,
)
from fyst_trajectories.planning._types import validate_computed_params

# The parity test compares against the ``scanning`` package (scan_patterns).
# It is optional; gate on its availability with the same HAS_* + skipif
# precedent used elsewhere in the suite (e.g. test_inject_retune_hitmap.py).
try:
    from scanning import Pong as _ScanningPong  # noqa: F401

    HAS_SCANNING = True
except ImportError:
    HAS_SCANNING = False


@pytest.fixture
def start_time():
    """Provide a standard start time."""
    return Time("2026-03-15T04:00:00", scale="utc")


class TestPlanPongAltAzScan:
    """Tests for plan_pong_altaz_scan."""

    def test_basic_plan(self, site, start_time):
        """Returns a ScanBlock with a pong_altaz config and trajectory."""
        block = plan_pong_altaz_scan(
            az_center=120.0,
            el_center=60.0,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            site=site,
            start_time=start_time,
        )

        assert isinstance(block, ScanBlock)
        assert isinstance(block.config, PongAltAzScanConfig)
        assert block.duration > 0
        assert block.trajectory.n_points > 0
        assert block.trajectory.pattern_type == "pong_altaz"
        assert "AltAz Pong scan" in block.summary

    def test_computed_params_schema_validates(self, site, start_time):
        """The returned computed_params matches the PongAltAz schema."""
        block = plan_pong_altaz_scan(
            az_center=120.0,
            el_center=60.0,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            site=site,
            start_time=start_time,
        )

        params = block.computed_params
        # Exactly the PongAltAzComputedParams keys, no more, no less.
        assert set(params) == set(PongAltAzComputedParams.__required_keys__)
        assert params["az_center"] == 120.0
        assert params["el_center"] == 60.0
        assert params["n_cycles"] == 1
        assert params["period"] > 0.0
        # Round-trips through the runtime validator without raising or warning.
        validate_computed_params(params, "pong_altaz")

    def test_known_square_field_period(self, site, start_time):
        """A 2x2 deg, 0.1 deg-spacing pong has the hand-derived period.

        Same geometry as the celestial Pong period test: x_numvert=15,
        y_numvert=16, period = 4*15*16*0.1/0.5 = 192.0 s. The AltAz mapping
        does not change the period (it is an on-sky-geometry quantity).
        """
        block = plan_pong_altaz_scan(
            az_center=120.0,
            el_center=60.0,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            site=site,
            start_time=start_time,
        )
        assert block.computed_params["x_numvert"] == 15
        assert block.computed_params["y_numvert"] == 16
        assert block.computed_params["period"] == pytest.approx(192.0)
        assert block.duration == pytest.approx(192.0)

    def test_duration_scales_with_n_cycles(self, site, start_time):
        """Duration equals n_cycles times the period."""
        block1 = plan_pong_altaz_scan(
            az_center=120.0,
            el_center=60.0,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            site=site,
            start_time=start_time,
            n_cycles=1,
        )
        block3 = plan_pong_altaz_scan(
            az_center=120.0,
            el_center=60.0,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            site=site,
            start_time=start_time,
            n_cycles=3,
        )
        assert block3.duration == pytest.approx(3.0 * block1.duration)
        assert block3.computed_params["period"] == pytest.approx(block1.computed_params["period"])

    def test_invalid_n_cycles_raises(self, site, start_time):
        """n_cycles < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_cycles must be at least 1"):
            plan_pong_altaz_scan(
                az_center=120.0,
                el_center=60.0,
                width=2.0,
                height=2.0,
                spacing=0.1,
                velocity=0.5,
                site=site,
                start_time=start_time,
                n_cycles=0,
            )

    def test_el_center_above_range_raises_config_message(self, site, start_time):
        """el_center > 90 raises the config's message, not an astropy latitude error."""
        with pytest.raises(ValueError, match="el_center"):
            plan_pong_altaz_scan(
                az_center=120.0,
                el_center=95.0,
                width=2.0,
                height=2.0,
                spacing=0.1,
                velocity=0.5,
                site=site,
                start_time=start_time,
            )

    def test_accepts_iso_start_time(self, site):
        """A start_time string is accepted, like the other planners."""
        block = plan_pong_altaz_scan(
            az_center=120.0,
            el_center=60.0,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            site=site,
            start_time="2026-03-15T04:00:00",
        )
        assert block.trajectory.n_points > 0

    def test_bounds_error_for_high_el_center(self, site, start_time):
        """A high el_center drives the elevation extent past the 90 deg limit.

        With el_center=85 and a 12 deg on-sky height the upper elevation
        extent reaches ~90.8 deg (the realized Pong y-extent is a little
        under the nominal height, so 12 rather than exactly 10 is needed to
        clear 90), so bounds validation in the AltAz builder path must raise.
        """
        with pytest.raises(TrajectoryBoundsError):
            plan_pong_altaz_scan(
                az_center=120.0,
                el_center=85.0,
                width=10.0,
                height=12.0,
                spacing=0.2,
                velocity=0.5,
                site=site,
                start_time=start_time,
            )

    def test_detector_offset_changes_trajectory(self, site, start_time):
        """A detector offset shifts the trajectory (smoke)."""
        block_no_offset = plan_pong_altaz_scan(
            az_center=120.0,
            el_center=60.0,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            site=site,
            start_time=start_time,
        )
        offset = InstrumentOffset(dx=5.0, dy=3.0, name="TestDet")
        block_with_offset = plan_pong_altaz_scan(
            az_center=120.0,
            el_center=60.0,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            site=site,
            start_time=start_time,
            detector_offset=offset,
        )
        assert not np.allclose(block_no_offset.trajectory.az, block_with_offset.trajectory.az)


class TestPlanPongAltAzSunSafety:
    """Sun-safety pre-flight behavior (warn-only)."""

    def test_sun_adjacent_center_warns(self, site, coordinates):
        """A center pointed at the Sun emits the EXCLUSION ZONE warning.

        The planner converts (az_center, el_center) to RA/Dec at the start
        time for the sun-safety pre-flight, so aiming the center straight at
        the Sun's az/el must trip the warning. Pick a start time where the
        Sun is well above the horizon so the pattern still builds.
        """
        obstime = Time("2026-03-15T17:00:00", scale="utc")
        sun_az, sun_alt = coordinates.get_sun_altaz(obstime)
        if not 20.0 < sun_alt < 80.0:
            pytest.skip(f"Sun elevation {sun_alt:.1f} not in a convenient test band")

        with pytest.warns(PointingWarning, match="EXCLUSION ZONE"):
            plan_pong_altaz_scan(
                az_center=float(sun_az),
                el_center=float(sun_alt),
                width=1.0,
                height=1.0,
                spacing=0.1,
                velocity=0.5,
                site=site,
                start_time=obstime,
            )

    def test_sun_safety_is_warn_only(self, site, coordinates):
        """The sun-adjacent center still returns a valid ScanBlock."""
        obstime = Time("2026-03-15T17:00:00", scale="utc")
        sun_az, sun_alt = coordinates.get_sun_altaz(obstime)
        if not 20.0 < sun_alt < 80.0:
            pytest.skip(f"Sun elevation {sun_alt:.1f} not in a convenient test band")

        with pytest.warns(PointingWarning, match="EXCLUSION ZONE"):
            block = plan_pong_altaz_scan(
                az_center=float(sun_az),
                el_center=float(sun_alt),
                width=1.0,
                height=1.0,
                spacing=0.1,
                velocity=0.5,
                site=site,
                start_time=obstime,
            )
        assert block.trajectory.n_points > 0


@pytest.mark.slow
@pytest.mark.skipif(not HAS_SCANNING, reason="requires the scanning (scan_patterns) package")
class TestLegacyMappingParity:
    """End-to-end parity with the legacy scanning.Pong + horizon mapping.

    scanning.Pong delegates its offset generation to this library's
    PongScanPattern, so this validates that plan_pong_altaz_scan reproduces
    the legacy horizon-frame mapping (x/cos(el0)+az0, y+el0) and the
    surrounding plumbing end to end, not the offset math itself.
    """

    def test_trajectory_matches_legacy_mapping(self, site):
        """plan_pong_altaz_scan az/el equals the inline legacy mapping."""
        from scanning import Pong

        az_center = 130.0
        el_center = 55.0
        width = 2.0
        height = 2.0
        spacing = 0.1
        velocity = 0.5
        num_terms = 4
        angle = 0.0
        timestep = 0.1

        block = plan_pong_altaz_scan(
            az_center=az_center,
            el_center=el_center,
            width=width,
            height=height,
            spacing=spacing,
            velocity=velocity,
            site=site,
            start_time=Time("2026-03-15T04:00:00", scale="utc"),
            num_terms=num_terms,
            angle=angle,
            timestep=timestep,
        )
        traj = block.trajectory

        # Build scanning.Pong with matching parameters and apply the legacy
        # horizon-frame mapping inline, exactly as the sims repo did:
        #   coscorr = cos(radians(el_center))
        #   az = x/coscorr + az_center ; el = y + el_center
        pong = Pong(
            num_term=num_terms,
            width=width,
            height=height,
            spacing=spacing,
            velocity=velocity,
            angle=angle,
            sample_interval=timestep,
            max_scan_duration=block.duration,
        )
        x_off = pong.x_coord.value
        y_off = pong.y_coord.value

        coscorr = math.cos(math.radians(el_center))
        legacy_az = x_off / coscorr + az_center
        legacy_el = y_off + el_center

        # Guard: the two paths must sample the same number of points.
        assert len(legacy_az) == traj.n_points

        np.testing.assert_allclose(traj.az, legacy_az, atol=1e-6)
        np.testing.assert_allclose(traj.el, legacy_el, atol=1e-6)
