"""Tests for plan_daisy_altaz_scan."""

import math

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories.exceptions import PointingWarning, TrajectoryBoundsError
from fyst_trajectories.offsets import InstrumentOffset
from fyst_trajectories.patterns.configs import DaisyAltAzScanConfig
from fyst_trajectories.planning import (
    DaisyAltAzComputedParams,
    ScanBlock,
    plan_daisy_altaz_scan,
)
from fyst_trajectories.planning._types import validate_computed_params

# The parity test compares against the ``scanning`` package (scan_patterns).
# It is optional; gate on its availability with the same HAS_* + skipif
# precedent used elsewhere in the suite (e.g. test_planning_pong_altaz.py).
try:
    from scanning import Daisy as _ScanningDaisy  # noqa: F401

    HAS_SCANNING = True
except ImportError:
    HAS_SCANNING = False


@pytest.fixture
def start_time():
    """Provide a standard start time."""
    return Time("2026-03-15T04:00:00", scale="utc")


def _plan(site, start_time, **overrides):
    """Call plan_daisy_altaz_scan with sensible test defaults."""
    params = dict(
        az_center=120.0,
        el_center=60.0,
        radius=0.5,
        velocity=0.3,
        turn_radius=0.2,
        avoidance_radius=0.0,
        start_acceleration=0.5,
        site=site,
        start_time=start_time,
        timestep=0.1,
        duration=100.0,
    )
    params.update(overrides)
    return plan_daisy_altaz_scan(**params)


class TestPlanDaisyAltAzScan:
    """Tests for plan_daisy_altaz_scan."""

    def test_basic_plan(self, site, start_time):
        """Returns a ScanBlock with a daisy_altaz config and trajectory."""
        block = _plan(site, start_time)

        assert isinstance(block, ScanBlock)
        assert isinstance(block.config, DaisyAltAzScanConfig)
        assert block.duration > 0
        assert block.trajectory.n_points > 0
        assert block.trajectory.pattern_type == "daisy_altaz"
        assert "AltAz Daisy scan" in block.summary

    def test_computed_params_schema_validates(self, site, start_time):
        """The returned computed_params matches the DaisyAltAz schema."""
        block = _plan(site, start_time)

        params = block.computed_params
        # Exactly the DaisyAltAzComputedParams keys, no more, no less.
        assert set(params) == set(DaisyAltAzComputedParams.__required_keys__)
        assert params["az_center"] == 120.0
        assert params["el_center"] == 60.0
        assert params["duration"] == pytest.approx(100.0)
        # Round-trips through the runtime validator without raising or warning.
        validate_computed_params(params, "daisy_altaz")

    def test_duration_honored(self, site, start_time):
        """The explicit duration is passed through to the ScanBlock."""
        block = _plan(site, start_time, duration=250.0)
        assert block.duration == pytest.approx(250.0)
        assert block.computed_params["duration"] == pytest.approx(250.0)

    def test_accepts_iso_start_time(self, site):
        """A start_time string is accepted, like the other planners."""
        block = _plan(site, "2026-03-15T04:00:00")
        assert block.trajectory.n_points > 0

    def test_el_center_above_range_raises_config_message(self, site, start_time):
        """el_center > 90 raises the config's message, not an astropy latitude error."""
        with pytest.raises(ValueError, match="el_center"):
            _plan(site, start_time, el_center=95.0)

    def test_bounds_error_when_elevation_exceeds_limit(self, site, start_time):
        """A high el_center plus a large radius drives el past the 90 deg limit.

        With el_center=87 and radius=6 the realized Daisy y-extent reaches
        ~5.7 deg, so the upper elevation extent hits ~92.7 deg and bounds
        validation in the AltAz builder path must raise.
        """
        with pytest.raises(TrajectoryBoundsError):
            _plan(
                site,
                start_time,
                el_center=87.0,
                radius=6.0,
                turn_radius=0.5,
                duration=400.0,
            )

    def test_detector_offset_changes_trajectory(self, site, start_time):
        """A detector offset shifts the trajectory (smoke)."""
        block_no_offset = _plan(site, start_time)
        offset = InstrumentOffset(dx=5.0, dy=3.0, name="TestDet")
        block_with_offset = _plan(site, start_time, detector_offset=offset)
        assert not np.allclose(block_no_offset.trajectory.az, block_with_offset.trajectory.az)


class TestPlanDaisyAltAzSunSafety:
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
            _plan(
                site,
                obstime,
                az_center=float(sun_az),
                el_center=float(sun_alt),
            )

    def test_sun_safety_is_warn_only(self, site, coordinates):
        """The sun-adjacent center still returns a valid ScanBlock."""
        obstime = Time("2026-03-15T17:00:00", scale="utc")
        sun_az, sun_alt = coordinates.get_sun_altaz(obstime)
        if not 20.0 < sun_alt < 80.0:
            pytest.skip(f"Sun elevation {sun_alt:.1f} not in a convenient test band")

        with pytest.warns(PointingWarning, match="EXCLUSION ZONE"):
            block = _plan(
                site,
                obstime,
                az_center=float(sun_az),
                el_center=float(sun_alt),
            )
        assert block.trajectory.n_points > 0


@pytest.mark.slow
@pytest.mark.skipif(not HAS_SCANNING, reason="requires the scanning (scan_patterns) package")
class TestLegacyMappingParity:
    """End-to-end parity with the legacy scanning.Daisy + horizon mapping.

    scanning.Daisy delegates its offset generation to this library's
    DaisyScanPattern, so this validates that plan_daisy_altaz_scan reproduces
    the legacy horizon-frame mapping (x/cos(el0)+az0, y+el0) and the
    surrounding plumbing end to end, not the offset math itself.
    """

    def test_trajectory_matches_legacy_mapping(self, site):
        """plan_daisy_altaz_scan az/el equals the inline legacy mapping."""
        from scanning import Daisy

        az_center = 130.0
        el_center = 55.0
        radius = 0.5
        velocity = 0.3
        turn_radius = 0.2
        avoidance_radius = 0.0
        start_acceleration = 0.5
        y_offset = 0.0
        timestep = 0.1
        duration = 100.0

        block = plan_daisy_altaz_scan(
            az_center=az_center,
            el_center=el_center,
            radius=radius,
            velocity=velocity,
            turn_radius=turn_radius,
            avoidance_radius=avoidance_radius,
            start_acceleration=start_acceleration,
            site=site,
            start_time=Time("2026-03-15T04:00:00", scale="utc"),
            timestep=timestep,
            duration=duration,
            y_offset=y_offset,
        )
        traj = block.trajectory

        # Build scanning.Daisy with matching parameters and apply the legacy
        # horizon-frame mapping inline, exactly as the sims repo did:
        #   coscorr = cos(radians(el_center))
        #   az = x/coscorr + az_center ; el = y + el_center
        daisy = Daisy(
            velocity=velocity,
            start_acc=start_acceleration,
            R0=radius,
            Rt=turn_radius,
            Ra=avoidance_radius,
            T=duration,
            sample_interval=timestep,
            y_offset=y_offset,
        )
        x_off = daisy.x_coord.value
        y_off = daisy.y_coord.value

        coscorr = math.cos(math.radians(el_center))
        legacy_az = x_off / coscorr + az_center
        legacy_el = y_off + el_center

        # Guard: the two paths must sample the same number of points.
        assert len(legacy_az) == traj.n_points

        # AltAz azimuth is used as provided (no normalization), so a direct
        # comparison is expected.
        np.testing.assert_allclose(traj.az, legacy_az, atol=1e-6)
        np.testing.assert_allclose(traj.el, legacy_el, atol=1e-6)
