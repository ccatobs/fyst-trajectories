"""Tests for PongScanPattern."""

import math

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories.patterns import PongScanConfig, PongScanPattern, compute_pong_period
from fyst_trajectories.patterns.pong import _pong_peak_offsets


class TestPongScanPattern:
    """Pong generation: metadata, offset-frame coverage extent, smoothness, and rotation."""

    def test_basic_pong_scan(self, site):
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

        assert trajectory.duration == pytest.approx(60.0, abs=0.2)
        assert trajectory.start_time == start_time
        assert trajectory.pattern_type == "pong"
        assert trajectory.center_ra == 180.0
        assert trajectory.center_dec == -30.0
        assert trajectory.metadata.input_frame == "icrs"

    def test_pong_covers_expected_region(self, site):
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
        # azimuth range is intentionally NOT upper-bounded, near high elevation
        # cos(el) inflates the azimuth coordinate, so a ~1 deg on-sky pattern can
        # span several degrees of azimuth.
        assert az_range > 0.5
        assert el_range > 0.5

        # Precise coverage check in the offset frame (decoupled from cos(el) and
        # field rotation): for this field x_numvert=8, y_numvert=9, so the ideal
        # triangle wave spans numvert*sqrt(2)*spacing, 1.131 deg in x and 1.273 deg
        # in y. Four Fourier terms reach 8/pi^2 * (1 + 1/9 + 1/25 + 1/49) = 0.950
        # of each vertex, so the path spans 1.074 deg and 1.209 deg.
        _, x_off, y_off = pattern.generate_offsets(300.0)
        truncation = 8.0 / np.pi**2 * (1.0 + 1.0 / 9.0 + 1.0 / 25.0 + 1.0 / 49.0)
        assert np.ptp(x_off) == pytest.approx(truncation * 8 * np.sqrt(2) * 0.1, abs=1e-6)
        assert np.ptp(y_off) == pytest.approx(truncation * 9 * np.sqrt(2) * 0.1, abs=1e-6)

    def test_pong_smooth_velocities(self, site):
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

        pattern.generate(site, duration=120.0, start_time=start_time)

        # A 3 x 0.5 deg field stays elongated: 2.95 deg by 0.67 deg in the offset frame.
        _, x_off, y_off = pattern.generate_offsets(120.0)
        assert np.ptp(x_off) > 3.0 * np.ptp(y_off)


class TestPongScanFlags:
    """Pong trajectories carry both science and turnaround flags, with science dominant."""

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
    reference); the Lissajous ``period`` and the two vertex counts it returns
    are pinned here to hand-derived values.
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


class TestPongVertexSpeed:
    """The science-speed threshold sits above the dip at the vertices."""

    def test_vertex_speed_dips_to_about_one_over_root_two(self):
        """At an x vertex the offset-frame speed falls to about ``velocity / sqrt(2)``.

        One axis passes through its turning point there, so the diagonal speed
        is the other axis's speed alone. The pattern flags science above
        ``0.8 * velocity``; this pins the dip that threshold is set against.
        """
        config = PongScanConfig(
            timestep=0.01,
            width=2.0,
            height=2.0,
            spacing=0.1,
            velocity=0.5,
            num_terms=4,
            angle=0.0,
        )
        period, _, _ = compute_pong_period(config)
        pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)
        times, x_off, y_off = pattern.generate_offsets(period)

        x_vel = np.gradient(x_off, times)
        y_vel = np.gradient(y_off, times)
        vertex = np.flatnonzero(np.sign(x_vel[:-1]) != np.sign(x_vel[1:]))
        assert vertex.size > 0
        median_fraction = np.median(np.hypot(x_vel[vertex], y_vel[vertex])) / config.velocity

        # Measured 0.747 of ``velocity``; the ideal corner gives 1 / sqrt(2) = 0.707.
        assert median_fraction == pytest.approx(1.0 / math.sqrt(2.0), abs=0.1)


class TestPongPeakOffsets:
    """``_pong_peak_offsets`` is the box the pattern fills: it reaches each side and no further.

    The offline scheduler bounds a pong visit by the elevation of this box's
    corners, so an offset past it would let a booked scan leave the limits.
    """

    @pytest.mark.parametrize(
        "width, height, spacing, num_terms",
        [
            (4.0, 4.0, 0.1, 4),  # the rebuild's defaults
            (4.0, 3.0, 0.5, 5),  # coarse spacing, five terms
            (6.0, 3.0, 0.3, 2),
            (2.0, 2.0, 0.1, 1),  # one term: the furthest from a triangle
        ],
    )
    def test_the_pattern_reaches_its_peaks_and_never_passes_them(
        self, width, height, spacing, num_terms
    ):
        config = PongScanConfig(
            timestep=0.01,
            width=width,
            height=height,
            spacing=spacing,
            velocity=0.5,
            num_terms=num_terms,
            angle=0.0,
        )
        period, _, _ = compute_pong_period(config)
        _, x_off, y_off = PongScanPattern(ra=0.0, dec=0.0, config=config).generate_offsets(period)

        peak_x, peak_y = _pong_peak_offsets(config)
        assert np.abs(x_off).max() <= peak_x * (1.0 + 1e-12)
        assert np.abs(y_off).max() <= peak_y * (1.0 + 1e-12)
        # Each axis peaks at a quarter of its own period, which the 0.01 s
        # grid samples to well under a micro-degree.
        assert np.abs(x_off).max() == pytest.approx(peak_x, abs=1e-6)
        assert np.abs(y_off).max() == pytest.approx(peak_y, abs=1e-6)
