"""Tests for PongScanPattern."""

import math

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories.patterns import PongScanConfig, PongScanPattern, compute_pong_period


class TestPongScanPattern:
    """Tests for Pong scan pattern."""

    def test_basic_pong_scan(self, site):
        """Test generating a basic Pong scan pattern."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = PongScanConfig(
            timestep=0.1,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        assert trajectory.n_points > 0
        assert trajectory.duration == pytest.approx(60.0, abs=0.2)
        assert trajectory.start_time == start_time
        assert trajectory.pattern_type == "pong"
        assert trajectory.center_ra == 180.0
        assert trajectory.center_dec == -30.0
        assert trajectory.coordsys == "altaz"
        assert trajectory.metadata.input_frame == "icrs"

    def test_pong_covers_expected_region(self, site):
        """Test that Pong pattern covers approximately the expected region."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = PongScanConfig(
            timestep=0.1,
            width=1.0,
            height=1.0,
            spacing=0.1,
            velocity=0.3,
            num_terms=4,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=300.0, start_time=start_time)

        az_range = trajectory.az.max() - trajectory.az.min()
        el_range = trajectory.el.max() - trajectory.el.min()

        # Sanity: the projected trajectory is genuinely 2-D (not a point). The raw
        # azimuth range is intentionally NOT upper-bounded -- near high elevation
        # cos(el) inflates the azimuth coordinate, so a ~1 deg on-sky pattern can
        # span several degrees of azimuth.
        assert az_range > 0.5
        assert el_range > 0.5

        # Precise coverage check in the offset frame (decoupled from cos(el) and
        # field rotation): for this field x_numvert=8, y_numvert=9, so the pattern
        # spans 2*amp = numvert*sqrt(2)*spacing -- 1.131 deg in x, 1.273 deg in y.
        _, x_off, y_off = pattern.generate_offsets(300.0)
        assert np.ptp(x_off) == pytest.approx(8 * np.sqrt(2) * 0.1, abs=0.1)
        assert np.ptp(y_off) == pytest.approx(9 * np.sqrt(2) * 0.1, abs=0.1)

    def test_pong_smooth_velocities(self, site):
        """Test that Pong pattern has smooth velocities."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = PongScanConfig(
            timestep=0.1,
            width=1.0,
            height=1.0,
            spacing=0.1,
            velocity=0.3,
            num_terms=4,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        assert np.abs(trajectory.az_vel).max() < 2.0
        assert np.abs(trajectory.el_vel).max() < 2.0

        dt = trajectory.times[1] - trajectory.times[0]
        az_accel = np.diff(trajectory.az_vel) / dt
        el_accel = np.diff(trajectory.el_vel) / dt

        assert np.abs(az_accel).max() < 10.0
        assert np.abs(el_accel).max() < 10.0

    def test_pong_with_rotation(self, site):
        """Test Pong pattern with non-zero rotation angle."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config_no_rot = PongScanConfig(
            timestep=0.1,
            width=1.0,
            height=1.0,
            spacing=0.1,
            velocity=0.3,
            num_terms=4,
            angle=0.0,
        )
        config_with_rot = PongScanConfig(
            timestep=0.1,
            width=1.0,
            height=1.0,
            spacing=0.1,
            velocity=0.3,
            num_terms=4,
            angle=45.0,
        )

        pattern_no_rot = PongScanPattern(ra=180.0, dec=-30.0, config=config_no_rot)
        pattern_with_rot = PongScanPattern(ra=180.0, dec=-30.0, config=config_with_rot)

        traj_no_rot = pattern_no_rot.generate(site, duration=60.0, start_time=start_time)
        traj_with_rot = pattern_with_rot.generate(site, duration=60.0, start_time=start_time)

        assert not np.allclose(traj_no_rot.az, traj_with_rot.az)

    def test_pong_metadata_stored(self, site):
        """Test that Pong pattern stores metadata correctly."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = PongScanConfig(
            timestep=0.1,
            width=2.0,
            height=1.5,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=30.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        assert trajectory.pattern_params is not None
        params = trajectory.pattern_params
        assert params["width"] == 2.0
        assert params["height"] == 1.5
        assert params["spacing"] == 0.1
        assert params["velocity"] == 0.5
        assert params["num_terms"] == 4
        assert params["angle"] == 30.0
        assert "period" in params
        assert "x_numvert" in params
        assert "y_numvert" in params

    def test_pong_narrow_pattern(self, site):
        """Test Pong pattern with very different width and height."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = PongScanConfig(
            timestep=0.1,
            width=3.0,
            height=0.5,
            spacing=0.1,
            velocity=0.3,
            num_terms=4,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)

        trajectory = pattern.generate(site, duration=120.0, start_time=start_time)

        assert trajectory.n_points > 0
        assert np.all(np.isfinite(trajectory.az))
        assert np.all(np.isfinite(trajectory.el))


class TestPongVertexComputation:
    """Tests for Pong vertex computation algorithm."""

    def test_vertices_are_coprime(self):
        """Test that computed vertex counts are coprime."""
        config = PongScanConfig(
            timestep=0.1,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)

        x_numvert, y_numvert, _, _ = pattern._compute_vertices()

        assert math.gcd(x_numvert, y_numvert) == 1

    def test_vertices_have_opposite_parity(self):
        """Test that vertex counts have opposite parity."""
        config = PongScanConfig(
            timestep=0.1,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)

        x_numvert, y_numvert, _, _ = pattern._compute_vertices()

        assert (x_numvert % 2) != (y_numvert % 2)


class TestPongScanFlags:
    """Tests for scan flag behavior on Pong trajectories."""

    def test_pong_trajectory_has_science_and_turnaround_flags(self, site):
        """Pong trajectory carries SCIENCE and TURNAROUND scan flags (not None)."""
        start_time = Time("2026-03-15T04:00:00", scale="utc")
        config = PongScanConfig(
            timestep=0.1,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        )
        pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)
        trajectory = pattern.generate(site, duration=60.0, start_time=start_time)

        assert trajectory.scan_flag is not None
        assert np.any(trajectory.scan_flag == 1)  # SCAN_FLAG_SCIENCE
        assert np.any(trajectory.scan_flag == 2)  # SCAN_FLAG_TURNAROUND
        # Majority should be science
        science_frac = (trajectory.scan_flag == 1).sum() / len(trajectory.scan_flag)
        assert science_frac > 0.7


class TestComputePongPeriod:
    """Ground-truth tests for the public ``compute_pong_period`` helper.

    ``compute_pong_period`` is exported in ``__all__`` as the canonical entry
    point for external code (e.g. the scan_patterns cross-validation
    reference). The Lissajous ``period`` and the two vertex counts it returns
    were previously never value-checked anywhere in the suite, so a regression
    in the period math would have passed silently.
    """

    def test_known_square_field_period(self):
        """A 2x2 deg, 0.1 deg-spacing pong has a hand-derivable period.

        ``vert_spacing = sqrt(2) * 0.1``; ``x_numvert = y_numvert =
        ceil(2 / vert_spacing) = 15`` before the opposite-parity bump pushes
        ``y_numvert`` to 16 (15 and 16 are coprime, so no further bump). The
        sqrt(2) factors cancel, leaving
        ``period = 4 * x_numvert * y_numvert * spacing / velocity =
        4 * 15 * 16 * 0.1 / 0.5 = 192.0`` s.
        """
        config = PongScanConfig(
            timestep=0.1,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        )

        period, x_numvert, y_numvert = compute_pong_period(config)

        assert x_numvert == 15
        assert y_numvert == 16
        assert period == pytest.approx(192.0)

    @pytest.mark.parametrize(
        "width, height, spacing, velocity",
        [
            (2.0, 2.0, 0.1, 0.5),  # square
            (3.0, 1.0, 0.1, 0.5),  # wide
            (1.0, 4.0, 0.05, 0.3),  # tall, fine spacing
            (5.0, 5.0, 0.25, 1.0),  # large, coarse
        ],
    )
    def test_period_invariants(self, width, height, spacing, velocity):
        """Vertex counts are coprime + opposite-parity and the period is positive.

        The Pong pattern only closes (and so tiles uniformly) when the two axes'
        vertex counts are coprime with opposite parity; the helper guarantees
        both by construction. A positive period is required for the downstream
        ``duration >= period`` coverage checks.
        """
        config = PongScanConfig(
            timestep=0.1,
            width=width,
            height=height,
            spacing=spacing,
            velocity=velocity,
            num_terms=4,
            angle=0.0,
        )

        period, x_numvert, y_numvert = compute_pong_period(config)

        assert period > 0.0
        assert math.gcd(x_numvert, y_numvert) == 1
        assert (x_numvert % 2) != (y_numvert % 2)

    def test_matches_pattern_metadata(self):
        """``compute_pong_period`` agrees with ``PongScanPattern.get_metadata``.

        The helper and the pattern compute the period from independent copies
        of the same formula; pin them together so they cannot drift apart.
        """
        config = PongScanConfig(
            timestep=0.1,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        )

        period, x_numvert, y_numvert = compute_pong_period(config)
        meta = PongScanPattern(ra=180.0, dec=-30.0, config=config).get_metadata()

        assert meta.pattern_params["period"] == pytest.approx(period)
        assert meta.pattern_params["x_numvert"] == x_numvert
        assert meta.pattern_params["y_numvert"] == y_numvert
