"""Smoke tests for plotting functions.

These tests verify the plotting functions execute end-to-end and return
the expected types. They do not verify pixel-perfect output. Matplotlib
is an optional dependency, so the entire module is skipped if it is
unavailable.
"""

import numpy as np
import pytest
from astropy.time import Time

# Skip the entire file if matplotlib isn't installed (optional extra).
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from fyst_trajectories import (  # noqa: E402
    InstrumentOffset,
    Trajectory,
    get_fyst_site,
)
from fyst_trajectories.visualization import plot_hit_map, plot_trajectory  # noqa: E402
from fyst_trajectories.visualization.hitmap import (  # noqa: E402
    _format_dec_deg,
    _format_ra_hm,
    _make_disk_kernel,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to keep memory bounded."""
    yield
    plt.close("all")


def _make_simple_trajectory(with_start_time: bool = False) -> Trajectory:
    """Build a minimal synthetic Trajectory suitable for plotting tests.

    Creates a 100-point trajectory slewing in azimuth at constant
    elevation. Optionally attaches a ``start_time`` for tests that
    require absolute times (e.g. ``plot_hit_map``).
    """
    n = 100
    times = np.linspace(0.0, 10.0, n)
    az = np.linspace(180.0, 181.0, n)
    el = np.full(n, 50.0)
    az_vel = np.gradient(az, times)
    el_vel = np.gradient(el, times)
    start_time = Time("2026-06-15T04:00:00", scale="utc") if with_start_time else None
    return Trajectory(
        times=times,
        az=az,
        el=el,
        az_vel=az_vel,
        el_vel=el_vel,
        start_time=start_time,
    )


# === plot_hit_map ===


class TestPlotHitMap:
    """Smoke tests for the public ``plot_hit_map`` function."""

    def test_returns_figure_with_single_offset(self):
        """plot_hit_map runs end-to-end on a short trajectory with one offset."""
        site = get_fyst_site()
        trajectory = _make_simple_trajectory(with_start_time=True)
        offsets = [(InstrumentOffset(dx=0.0, dy=0.0, name="boresight"), "boresight")]

        fig = plot_hit_map(trajectory, offsets, site, show=False)

        assert isinstance(fig, Figure)
        assert len(fig.axes) >= 1  # at least one panel + possibly colorbar axes

    def test_returns_figure_with_multiple_offsets(self):
        """plot_hit_map produces one panel per offset when multiple are given."""
        site = get_fyst_site()
        trajectory = _make_simple_trajectory(with_start_time=True)
        offsets = [
            (InstrumentOffset(dx=0.0, dy=0.0, name="center"), "center"),
            (InstrumentOffset(dx=5.0, dy=0.0, name="right"), "right"),
        ]

        fig = plot_hit_map(trajectory, offsets, site, show=False)

        assert isinstance(fig, Figure)
        # Two offsets -> at least two panels (plus possible colorbar axes).
        assert len(fig.axes) >= 2

    def test_without_start_time_raises(self):
        """plot_hit_map should raise ValueError if trajectory has no start_time."""
        site = get_fyst_site()
        trajectory = _make_simple_trajectory(with_start_time=False)
        offsets = [(InstrumentOffset(dx=0.0, dy=0.0, name="boresight"), "boresight")]

        with pytest.raises(ValueError, match="start_time"):
            plot_hit_map(trajectory, offsets, site, show=False)

    def test_module_fov_coverage_mode(self):
        """plot_hit_map in coverage mode (module_fov set) runs without errors."""
        scipy = pytest.importorskip("scipy")  # noqa: F841
        site = get_fyst_site()
        trajectory = _make_simple_trajectory(with_start_time=True)
        offsets = [(InstrumentOffset(dx=0.0, dy=0.0, name="boresight"), "boresight")]

        fig = plot_hit_map(
            trajectory,
            offsets,
            site,
            module_fov=1.1,
            show=False,
        )

        assert isinstance(fig, Figure)

    def test_smooth_sigma(self):
        """plot_hit_map with Gaussian smoothing runs without errors."""
        scipy = pytest.importorskip("scipy")  # noqa: F841
        site = get_fyst_site()
        trajectory = _make_simple_trajectory(with_start_time=True)
        offsets = [(InstrumentOffset(dx=0.0, dy=0.0, name="boresight"), "boresight")]

        fig = plot_hit_map(
            trajectory,
            offsets,
            site,
            smooth_sigma=1.0,
            show=False,
        )

        assert isinstance(fig, Figure)


# === plot_trajectory ===


class TestPlotTrajectory:
    """Smoke tests for the public ``plot_trajectory`` function."""

    def test_returns_figure_for_simple_trajectory(self):
        """plot_trajectory runs end-to-end on a minimal synthetic trajectory."""
        trajectory = _make_simple_trajectory(with_start_time=False)

        fig = plot_trajectory(trajectory, show=False)

        assert isinstance(fig, Figure)
        # plot_trajectory creates a 3-panel figure (az/t, el/t, sky track).
        assert len(fig.axes) == 3

    def test_returns_figure_with_start_time(self):
        """plot_trajectory ignores start_time and still returns a Figure."""
        trajectory = _make_simple_trajectory(with_start_time=True)

        fig = plot_trajectory(trajectory, show=False)

        assert isinstance(fig, Figure)


# === _make_disk_kernel ===


class TestMakeDiskKernel:
    """Tests for the internal disk-kernel helper."""

    def test_returns_2d_array(self):
        kernel = _make_disk_kernel(radius_bins=3.0)
        assert isinstance(kernel, np.ndarray)
        assert kernel.ndim == 2

    def test_shape_matches_radius(self):
        """The kernel is (2*ceil(r)+1) x (2*ceil(r)+1) bins."""
        kernel = _make_disk_kernel(radius_bins=3.0)
        assert kernel.shape == (7, 7)

    def test_non_integer_radius(self):
        """Non-integer radii use ceil() for the bounding box size."""
        kernel = _make_disk_kernel(radius_bins=2.5)
        assert kernel.shape == (7, 7)  # 2*ceil(2.5) + 1 = 7

    def test_normalized_to_unit_sum(self):
        """The kernel should be normalized so its elements sum to 1."""
        kernel = _make_disk_kernel(radius_bins=5.0)
        assert kernel.sum() == pytest.approx(1.0)

    def test_disk_is_symmetric(self):
        """A disk kernel should be symmetric across both axes."""
        kernel = _make_disk_kernel(radius_bins=5.0)
        np.testing.assert_array_equal(kernel, kernel[::-1, :])
        np.testing.assert_array_equal(kernel, kernel[:, ::-1])

    def test_values_are_non_negative(self):
        kernel = _make_disk_kernel(radius_bins=4.0)
        assert np.all(kernel >= 0.0)

    def test_center_inside_disk(self):
        """The central element should be inside the disk (non-zero)."""
        kernel = _make_disk_kernel(radius_bins=3.0)
        center = kernel.shape[0] // 2
        assert kernel[center, center] > 0.0


# === _format_ra_hm ===


class TestFormatRaHm:
    """Tests for the RA -> hour-angle label formatter."""

    def test_format_returns_string(self):
        result = _format_ra_hm(0.0, None)
        assert isinstance(result, str)

    def test_format_zero(self):
        """0 degrees is 0h00m."""
        result = _format_ra_hm(0.0, None)
        # Exact LaTeX token: hours=0, minutes=00.
        assert result == "0$^{h}$00$^{m}$"

    def test_format_180_is_12h(self):
        """180 degrees = 12h on the RA hour scale."""
        result = _format_ra_hm(180.0, None)
        # Starts with the 12-hour value, not merely "12" somewhere in the string.
        assert result.startswith("12$^{h}$")
        assert result == "12$^{h}$00$^{m}$"

    def test_format_15_is_1h(self):
        """15 degrees = 1h on the RA hour scale."""
        result = _format_ra_hm(15.0, None)
        assert result == "1$^{h}$00$^{m}$"

    def test_wraps_at_360(self):
        """RA values at or beyond 360 degrees wrap back to 0h."""
        result_360 = _format_ra_hm(360.0, None)
        result_0 = _format_ra_hm(0.0, None)
        assert result_360 == result_0


# === _format_dec_deg ===


class TestFormatDecDeg:
    """Tests for the Dec -> degree label formatter."""

    def test_format_returns_string(self):
        result = _format_dec_deg(0.0, None)
        assert isinstance(result, str)

    def test_format_zero(self):
        """0 degrees should render a '0' with a degree marker."""
        result = _format_dec_deg(0.0, None)
        assert "0" in result
        # LaTeX degree marker should be present.
        assert "circ" in result

    def test_format_positive(self):
        """Positive dec values should render the integer degrees."""
        result = _format_dec_deg(30.0, None)
        assert "30" in result

    def test_format_negative(self):
        """Negative dec values should include a minus sign."""
        result = _format_dec_deg(-30.0, None)
        assert "-" in result or "\u2212" in result
        assert "30" in result
