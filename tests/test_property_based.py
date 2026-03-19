"""Property-based tests using Hypothesis.

This module uses Hypothesis to test properties that should hold for all valid
inputs. Property-based testing is excellent for finding edge cases that would
be missed by example-based tests.

Properties tested:
1. Coordinate transforms produce valid outputs for valid inputs
2. Forward/inverse transforms are inverses within tolerance
3. Results are deterministic (same input -> same output)
4. Offset transforms are reversible
5. Trajectory properties (matching array lengths, monotonic times)
"""

import numpy as np
import pytest
from astropy.time import Time
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from fyst_trajectories import Coordinates, InstrumentOffset, get_fyst_site
from fyst_trajectories.offsets import boresight_to_detector, detector_to_boresight
from fyst_trajectories.patterns import (
    ConstantElScanConfig,
    ConstantElScanPattern,
    LinearMotionConfig,
    LinearMotionPattern,
    PongScanConfig,
    PongScanPattern,
    TrajectoryBuilder,
)

# -----------------------------------------------------------------------------
# Strategy definitions for generating valid astronomical inputs
# -----------------------------------------------------------------------------

# Right Ascension: 0 to 360 degrees
ra_strategy = st.floats(min_value=0.0, max_value=360.0, allow_nan=False, allow_infinity=False)

# Declination: -90 to 90 degrees
dec_strategy = st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False)

# Azimuth: 0 to 360 degrees (telescope limits may be tighter)
az_strategy = st.floats(min_value=0.0, max_value=360.0, allow_nan=False, allow_infinity=False)

# Elevation: 20 to 87 degrees (typical telescope limits)
el_strategy = st.floats(min_value=20.0, max_value=87.0, allow_nan=False, allow_infinity=False)

# Observable declination from FYST site (~-23 deg latitude)
# Roughly -90 to +67 can be observed above horizon
observable_dec_strategy = st.floats(
    min_value=-85.0, max_value=60.0, allow_nan=False, allow_infinity=False
)

# Offset values in arcminutes (reasonable range for focal plane)
offset_strategy = st.floats(
    min_value=-120.0,
    max_value=120.0,
    allow_nan=False,
    allow_infinity=False,
)

# Field rotation angles
field_rotation_strategy = st.floats(
    min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False
)

# Scan dimensions (degrees)
scan_dim_strategy = st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)

# Scan velocity (degrees/second)
scan_velocity_strategy = st.floats(
    min_value=0.05, max_value=2.0, allow_nan=False, allow_infinity=False
)

# Duration in seconds
duration_strategy = st.floats(
    min_value=10.0,
    max_value=300.0,
    allow_nan=False,
    allow_infinity=False,
)

# Timestep in seconds
timestep_strategy = st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False)


# -----------------------------------------------------------------------------
# Coordinate Transform Properties
# -----------------------------------------------------------------------------


class TestCoordinateTransformProperties:
    """Property-based tests for coordinate transformations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.site = get_fyst_site()
        self.coords = Coordinates(self.site)
        # Use a fixed time to ensure reproducibility
        self.obstime = Time("2026-06-15T04:00:00", scale="utc")

    @given(ra=ra_strategy, dec=observable_dec_strategy)
    @settings(max_examples=100, deadline=None)
    def test_radec_to_altaz_produces_valid_output(self, ra, dec):
        """Test that radec_to_altaz produces valid Az/El for any valid RA/Dec."""
        az, el = self.coords.radec_to_altaz(ra, dec, obstime=self.obstime)

        assert -360 < az < 720, f"Azimuth {az} outside valid range"
        assert -90 <= el <= 90, f"Elevation {el} outside valid range"
        assert np.isfinite(az), f"Azimuth {az} is not finite"
        assert np.isfinite(el), f"Elevation {el} is not finite"

    @given(az=az_strategy, el=el_strategy)
    @settings(max_examples=100, deadline=None)
    def test_altaz_to_radec_produces_valid_output(self, az, el):
        """Test that altaz_to_radec produces valid RA/Dec for any valid Az/El."""
        ra, dec = self.coords.altaz_to_radec(az, el, obstime=self.obstime)

        assert 0 <= ra < 360, f"RA {ra} outside valid range"
        assert -90 <= dec <= 90, f"Dec {dec} outside valid range"
        assert np.isfinite(ra), f"RA {ra} is not finite"
        assert np.isfinite(dec), f"Dec {dec} is not finite"

    @given(ra=ra_strategy, dec=observable_dec_strategy)
    @settings(max_examples=50, deadline=None)
    def test_radec_altaz_round_trip(self, ra, dec):
        """Test that RA/Dec -> Az/El -> RA/Dec returns the original values."""
        az, el = self.coords.radec_to_altaz(ra, dec, obstime=self.obstime)
        assume(el > 5.0)

        ra_back, dec_back = self.coords.altaz_to_radec(az, el, obstime=self.obstime)

        ra_diff = abs(ra_back - ra)
        ra_diff = min(ra_diff, 360 - ra_diff)

        assert ra_diff < 0.01, f"RA round-trip error: {ra} -> {ra_back}"
        assert abs(dec_back - dec) < 0.01, f"Dec round-trip error: {dec} -> {dec_back}"

    @given(az=az_strategy, el=el_strategy)
    @settings(max_examples=50, deadline=None)
    def test_altaz_radec_round_trip(self, az, el):
        """Test that Az/El -> RA/Dec -> Az/El returns the original values."""
        ra, dec = self.coords.altaz_to_radec(az, el, obstime=self.obstime)
        az_back, el_back = self.coords.radec_to_altaz(ra, dec, obstime=self.obstime)

        az_diff = abs(az_back - az)
        az_diff = min(az_diff, 360 - az_diff)

        assert az_diff < 0.01, f"Az round-trip error: {az} -> {az_back}"
        assert abs(el_back - el) < 0.01, f"El round-trip error: {el} -> {el_back}"

    @given(ra=ra_strategy, dec=observable_dec_strategy)
    @settings(max_examples=30, deadline=None)
    def test_radec_to_altaz_deterministic(self, ra, dec):
        """Test that radec_to_altaz is deterministic (same input -> same output)."""
        az1, el1 = self.coords.radec_to_altaz(ra, dec, obstime=self.obstime)
        az2, el2 = self.coords.radec_to_altaz(ra, dec, obstime=self.obstime)

        assert az1 == az2, f"Non-deterministic azimuth: {az1} != {az2}"
        assert el1 == el2, f"Non-deterministic elevation: {el1} != {el2}"


# -----------------------------------------------------------------------------
# Offset Transform Properties
# -----------------------------------------------------------------------------


class TestOffsetTransformProperties:
    """Property-based tests for instrument offset transformations."""

    @given(
        dx=offset_strategy,
        dy=offset_strategy,
        az=az_strategy,
        el=el_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_boresight_detector_round_trip(self, dx, dy, az, el):
        """Test that boresight -> detector -> boresight returns original.

        Spherical round-trip with Newton refinement achieves sub-milliarcsecond
        precision. Skip cases where offset would push elevation past 89 deg
        (zenith singularity causes 1/cos(el) amplification in azimuth).
        """
        # Offsets that push elevation past 89 deg hit the zenith singularity
        max_el_offset = abs(dy) / 60.0 + abs(dx) / 60.0
        assume(el + max_el_offset < 89.0)

        offset = InstrumentOffset(dx=dx, dy=dy)

        det_az, det_el = boresight_to_detector(az, el, offset, field_rotation=0.0)
        bore_az, bore_el = detector_to_boresight(det_az, det_el, offset, field_rotation=0.0)

        az_diff = abs(bore_az - az)
        az_diff = min(az_diff, 360 - az_diff)

        assert az_diff < 1e-5, f"Az round-trip error: {az} -> {bore_az}"
        assert abs(bore_el - el) < 1e-5, f"El round-trip error: {el} -> {bore_el}"

    @given(
        dx=offset_strategy,
        dy=offset_strategy,
        az=az_strategy,
        el=el_strategy,
        field_rotation=field_rotation_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_round_trip_with_field_rotation(self, dx, dy, az, el, field_rotation):
        """Test round-trip with field rotation applied.

        Spherical round-trip with Newton refinement achieves sub-milliarcsecond
        precision. Skip cases where offset would push elevation past 89 deg
        (zenith singularity causes 1/cos(el) amplification in azimuth).
        """
        # Offsets that push elevation past 89 deg hit the zenith singularity
        max_el_offset = abs(dy) / 60.0 + abs(dx) / 60.0
        assume(el + max_el_offset < 89.0)

        offset = InstrumentOffset(dx=dx, dy=dy)

        det_az, det_el = boresight_to_detector(az, el, offset, field_rotation=field_rotation)
        bore_az, bore_el = detector_to_boresight(
            det_az, det_el, offset, field_rotation=field_rotation
        )

        az_diff = abs(bore_az - az)
        az_diff = min(az_diff, 360 - az_diff)

        assert az_diff < 1e-5, f"Az round-trip error with FR={field_rotation}"
        assert abs(bore_el - el) < 1e-5, f"El round-trip error with FR={field_rotation}"

    @given(az=az_strategy, el=el_strategy)
    @settings(max_examples=50, deadline=None)
    def test_zero_offset_no_change(self, az, el):
        """Test that zero offset produces no change in coordinates."""
        offset = InstrumentOffset(dx=0.0, dy=0.0)

        det_az, det_el = boresight_to_detector(az, el, offset, field_rotation=0.0)

        assert det_az == pytest.approx(az)
        assert det_el == pytest.approx(el)

    @given(
        dx=offset_strategy,
        dy=offset_strategy,
        az=az_strategy,
        el=el_strategy,
    )
    @settings(max_examples=50, deadline=None)
    def test_offset_produces_finite_results(self, dx, dy, az, el):
        """Test that offset transforms produce finite results."""
        offset = InstrumentOffset(dx=dx, dy=dy)

        det_az, det_el = boresight_to_detector(az, el, offset, field_rotation=0.0)

        assert np.isfinite(det_az), f"Detector azimuth {det_az} is not finite"
        assert np.isfinite(det_el), f"Detector elevation {det_el} is not finite"


# -----------------------------------------------------------------------------
# Trajectory Properties
# -----------------------------------------------------------------------------


class TestTrajectoryProperties:
    """Property-based tests for trajectory generation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.site = get_fyst_site()
        self.start_time = Time("2026-03-15T04:00:00", scale="utc")

    @given(duration=duration_strategy)
    @settings(max_examples=20, deadline=None)
    def test_linear_trajectory_array_lengths_match(self, duration):
        """Test that all trajectory arrays have the same length."""
        config = LinearMotionConfig(
            timestep=0.1,
            az_start=150.0,
            el_start=45.0,
            az_velocity=0.1,
            el_velocity=0.0,
        )
        pattern = LinearMotionPattern(config)
        trajectory = pattern.generate(self.site, duration=duration, start_time=self.start_time)

        n = len(trajectory.times)
        assert len(trajectory.az) == n
        assert len(trajectory.el) == n
        assert len(trajectory.az_vel) == n
        assert len(trajectory.el_vel) == n

    @given(duration=duration_strategy)
    @settings(max_examples=20, deadline=None)
    def test_trajectory_times_monotonic(self, duration):
        """Test that trajectory times are monotonically increasing."""
        config = LinearMotionConfig(
            timestep=0.1,
            az_start=150.0,
            el_start=45.0,
            az_velocity=0.1,
            el_velocity=0.0,
        )
        pattern = LinearMotionPattern(config)
        trajectory = pattern.generate(self.site, duration=duration, start_time=self.start_time)

        time_diffs = np.diff(trajectory.times)
        assert np.all(time_diffs > 0), "Times are not monotonically increasing"

    @given(duration=duration_strategy)
    @settings(max_examples=20, deadline=None)
    def test_trajectory_duration_matches(self, duration):
        """Test that trajectory duration matches expected value."""
        config = LinearMotionConfig(
            timestep=0.1,
            az_start=150.0,
            el_start=45.0,
            az_velocity=0.1,
            el_velocity=0.0,
        )
        pattern = LinearMotionPattern(config)
        trajectory = pattern.generate(self.site, duration=duration, start_time=self.start_time)

        actual_duration = trajectory.times[-1] - trajectory.times[0]
        assert abs(actual_duration - duration) < 0.5, (
            f"Duration mismatch: requested {duration}, got {actual_duration}"
        )

    @given(
        width=st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False),
        height=st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False),
        velocity=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_pong_trajectory_arrays_finite(self, width, height, velocity):
        """Test that Pong trajectory arrays contain only finite values."""
        config = PongScanConfig(
            timestep=0.1,
            width=width,
            height=height,
            spacing=0.1,
            velocity=velocity,
            num_terms=4,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)
        trajectory = pattern.generate(self.site, duration=60.0, start_time=self.start_time)

        assert np.all(np.isfinite(trajectory.times)), "Times contain non-finite values"
        assert np.all(np.isfinite(trajectory.az)), "Azimuth contains non-finite values"
        assert np.all(np.isfinite(trajectory.el)), "Elevation contains non-finite values"
        assert np.all(np.isfinite(trajectory.az_vel)), "Az velocity contains non-finite values"
        assert np.all(np.isfinite(trajectory.el_vel)), "El velocity contains non-finite values"

    @given(
        az_start=st.floats(min_value=50.0, max_value=150.0, allow_nan=False, allow_infinity=False),
        az_stop=st.floats(min_value=200.0, max_value=300.0, allow_nan=False, allow_infinity=False),
        elevation=st.floats(min_value=30.0, max_value=70.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_constant_el_elevation_truly_constant(self, az_start, az_stop, elevation):
        """Test that ConstantElScan maintains constant elevation."""
        config = ConstantElScanConfig(
            timestep=0.1,
            az_start=az_start,
            az_stop=az_stop,
            elevation=elevation,
            az_speed=1.0,
            az_accel=0.5,
            n_scans=1,
        )
        pattern = ConstantElScanPattern(config)
        trajectory = pattern.generate(self.site, duration=60.0, start_time=self.start_time)

        np.testing.assert_allclose(
            trajectory.el,
            np.full_like(trajectory.el, elevation),
            atol=1e-10,
            err_msg="Elevation is not constant",
        )


class TestTrajectoryBuilderProperties:
    """Property-based tests for TrajectoryBuilder."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.site = get_fyst_site()
        self.start_time = Time("2026-03-15T04:00:00", scale="utc")

    @given(
        ra=st.floats(min_value=150.0, max_value=210.0, allow_nan=False, allow_infinity=False),
        dec=st.floats(min_value=-50.0, max_value=-10.0, allow_nan=False, allow_infinity=False),
        width=st.floats(min_value=0.5, max_value=1.5, allow_nan=False, allow_infinity=False),
        height=st.floats(min_value=0.5, max_value=1.5, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=15, deadline=None)
    def test_builder_produces_valid_trajectory(self, ra, dec, width, height):
        """Test that TrajectoryBuilder produces valid trajectories."""
        coords = Coordinates(self.site)
        center_az, center_el = coords.radec_to_altaz(ra, dec, obstime=self.start_time)
        limits = self.site.telescope_limits
        margin = max(width, height) * 2
        assume(center_az >= limits.azimuth.min + margin)
        assume(center_az <= limits.azimuth.max - margin)
        # Exclude azimuths near the 0/360 wrap; pattern offsets can
        # straddle the boundary and produce values outside [-180, 360].
        assume(center_az > margin)
        assume(center_az < 360.0 - margin)
        assume(center_el >= limits.elevation.min + margin)
        assume(center_el <= limits.elevation.max - margin)

        trajectory = (
            TrajectoryBuilder(self.site)
            .at(ra=ra, dec=dec)
            .with_config(
                PongScanConfig(
                    timestep=0.1,
                    width=width,
                    height=height,
                    spacing=0.1,
                    velocity=0.3,
                    num_terms=4,
                    angle=0.0,
                )
            )
            .duration(30.0)
            .starting_at(self.start_time)
            .build()
        )

        n = len(trajectory.times)
        assert len(trajectory.az) == n
        assert len(trajectory.el) == n
        assert len(trajectory.az_vel) == n
        assert len(trajectory.el_vel) == n

        assert np.all(np.isfinite(trajectory.az))
        assert np.all(np.isfinite(trajectory.el))
        assert np.all(np.isfinite(trajectory.az_vel))
        assert np.all(np.isfinite(trajectory.el_vel))

        assert trajectory.center_ra == ra
        assert trajectory.center_dec == dec
        assert trajectory.pattern_type == "pong"


# -----------------------------------------------------------------------------
# Idempotence and Consistency Tests
# -----------------------------------------------------------------------------


class TestConsistencyProperties:
    """Tests for consistency and idempotence properties."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.site = get_fyst_site()
        self.coords = Coordinates(self.site)
        self.start_time = Time("2026-03-15T04:00:00", scale="utc")

    @given(ra=ra_strategy, dec=observable_dec_strategy)
    @settings(max_examples=30, deadline=None)
    def test_parallactic_angle_finite(self, ra, dec):
        """Test that parallactic angle is always finite."""
        pa = self.coords.get_parallactic_angle(ra, dec, obstime=self.start_time)
        assert np.isfinite(pa), f"Parallactic angle {pa} is not finite for ({ra}, {dec})"

    @given(ra=ra_strategy, dec=observable_dec_strategy)
    @settings(max_examples=30, deadline=None)
    def test_field_rotation_finite(self, ra, dec):
        """Test that field rotation is always finite."""
        fr = self.coords.get_field_rotation(ra, dec, obstime=self.start_time)
        assert np.isfinite(fr), f"Field rotation {fr} is not finite for ({ra}, {dec})"

    @given(ra=ra_strategy)
    @settings(max_examples=30, deadline=None)
    def test_hour_angle_in_range(self, ra):
        """Test that hour angle is in valid range."""
        ha = self.coords.get_hour_angle(ra, obstime=self.start_time)
        assert -180 <= ha <= 180, f"Hour angle {ha} outside valid range"

    @given(ra=ra_strategy, dec=observable_dec_strategy)
    @settings(max_examples=20, deadline=None)
    def test_lst_consistent_with_hour_angle(self, ra, dec):
        """Test that LST, RA, and HA are consistent."""
        lst = self.coords.get_lst(obstime=self.start_time)
        ha = self.coords.get_hour_angle(ra, obstime=self.start_time)

        expected_ha = lst - ra
        expected_ha = ((expected_ha + 180) % 360) - 180

        assert abs(ha - expected_ha) < 1e-10, (
            f"HA inconsistency: computed {ha}, expected {expected_ha}"
        )
