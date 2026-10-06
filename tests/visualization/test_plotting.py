"""Smoke tests for plotting functions.

These tests verify the plotting functions execute end-to-end and return
the expected types. They do not verify pixel-perfect output. Matplotlib
is an optional dependency, so the entire module is skipped if it is
unavailable.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.time import Time

# Skip the entire file if matplotlib isn't installed (optional extra).
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import QuadMesh  # noqa: E402
from matplotlib.contour import ContourSet  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from fyst_trajectories import (  # noqa: E402
    InstrumentOffset,
    Trajectory,
    get_fyst_site,
)
from fyst_trajectories.planning import (  # noqa: E402
    FieldRegion,
    plan_daisy_scan,
    plan_pong_scan,
)
from fyst_trajectories.visualization import plot_hit_map, plot_trajectory  # noqa: E402
from fyst_trajectories.visualization.hitmap import (  # noqa: E402
    _format_dec_deg,
    _format_ra_hm,
    _format_ra_hms,
    _make_disk_kernel,
    _ra_tick_step_deg,
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


def _hit_total(ax):
    """Sum of the hit map drawn on one panel (its single QuadMesh)."""
    (mesh,) = [c for c in ax.collections if isinstance(c, QuadMesh)]
    return float(np.asarray(mesh.get_array()).sum())


# === plot_hit_map ===


def _pong_2deg_at_ra_12h() -> Trajectory:
    """One period of the planning page's 2 x 2 deg Pong at RA 12h."""
    return plan_pong_scan(
        field=FieldRegion(ra_center=180.0, dec_center=-30.0, width=2.0, height=2.0),
        velocity=0.4,
        spacing=0.1,
        num_terms=4,
        site=get_fyst_site(),
        start_time=Time("2026-03-15T01:00:00", scale="utc"),
    ).trajectory


def _rendered_ra_labels(ax):
    """Return the RA tick labels drawn inside the panel, with their window extents."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    lo, hi = sorted(ax.get_xlim())
    labels = [
        t
        for t in ax.get_xticklabels()
        if t.get_text() and t.get_visible() and lo <= t.get_position()[0] <= hi
    ]
    return [t.get_text() for t in labels], [t.get_window_extent(renderer) for t in labels]


class TestPlotHitMap:
    """One panel per module, the start_time requirement, and refusals before a figure opens."""

    def test_returns_figure_with_single_offset(self):
        site = get_fyst_site()
        trajectory = _make_simple_trajectory(with_start_time=True)
        modules = {"boresight": InstrumentOffset(dx=0.0, dy=0.0, name="boresight")}

        fig = plot_hit_map(trajectory, modules=modules, site=site, fov_radius_deg=None, show=False)

        assert isinstance(fig, Figure)
        assert len(fig.axes) >= 1  # at least one panel + possibly colorbar axes
        assert _hit_total(fig.axes[0]) == pytest.approx(100.0)  # every sample binned once

    def test_returns_figure_with_multiple_offsets(self):
        """plot_hit_map produces one panel per module when several are given."""
        site = get_fyst_site()
        trajectory = _make_simple_trajectory(with_start_time=True)
        modules = {
            "center": InstrumentOffset(dx=0.0, dy=0.0, name="center"),
            "right": InstrumentOffset(dx=5.0, dy=0.0, name="right"),
        }

        fig = plot_hit_map(trajectory, modules=modules, site=site, fov_radius_deg=None, show=False)

        assert isinstance(fig, Figure)
        # Two modules -> at least two panels (plus possible colorbar axes).
        assert len(fig.axes) >= 2

    def test_default_modules_draw_one_key_labelled_panel_per_module(self):
        """Without ``modules`` the seven PrimeCam modules fill a 2 x 4 grid.

        The alias ``"center"`` is drawn once, under its first key ``"c"``,
        and the eighth axes of the grid is hidden.
        """
        pytest.importorskip("scipy")
        trajectory = _make_simple_trajectory(with_start_time=True)

        fig = plot_hit_map(trajectory, title="All modules", show=False)

        titles = [ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")]
        assert titles == ["c", "i1", "i2", "i3", "i4", "i5", "i6"]
        assert sum(not ax.get_visible() for ax in fig.axes) == 1
        assert tuple(fig.get_size_inches()) == pytest.approx((24.0, 10.0))
        assert fig.get_suptitle() == "All modules"

    def test_caller_axes_are_drawn_into_without_layout_or_show(self, monkeypatch):
        """With ``axes`` the caller's figure comes back, neither laid out nor shown."""
        trajectory = _make_simple_trajectory(with_start_time=True)
        modules = {"boresight": InstrumentOffset(dx=0.0, dy=0.0)}
        fig, ax = plt.subplots()
        show_calls: list[int] = []
        layout_calls: list[int] = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: show_calls.append(1))
        monkeypatch.setattr(Figure, "tight_layout", lambda self, *a, **k: layout_calls.append(1))

        out = plot_hit_map(
            trajectory, modules=modules, fov_radius_deg=None, title="composed", axes=[ax]
        )

        assert out is fig
        assert plt.get_fignums() == [fig.number]
        assert _hit_total(ax) == pytest.approx(100.0)
        assert ax.get_title(loc="left") == "composed"
        assert show_calls == []
        assert layout_calls == []

    def test_wrong_number_of_axes_is_refused_before_drawing(self):
        trajectory = _make_simple_trajectory(with_start_time=True)
        modules = {
            "a": InstrumentOffset(dx=0.0, dy=0.0),
            "b": InstrumentOffset(dx=5.0, dy=0.0),
        }
        fig, ax = plt.subplots()

        with pytest.raises(ValueError, match="axes has 1 entries but 2 modules"):
            plot_hit_map(trajectory, modules=modules, axes=[ax], show=False)

        assert plt.get_fignums() == [fig.number]
        assert not ax.collections

    def test_old_call_is_refused(self):
        """The removed positional form and ``module_fov`` fail loudly, leaving no figure."""
        site = get_fyst_site()
        trajectory = _make_simple_trajectory(with_start_time=True)
        offsets = [(InstrumentOffset(dx=0.0, dy=0.0), "boresight")]

        plt.close("all")
        with pytest.raises(TypeError, match="positional"):
            plot_hit_map(trajectory, offsets, site, show=False)
        with pytest.raises(TypeError, match="module_fov"):
            plot_hit_map(trajectory, module_fov=1.3, show=False)
        assert plt.get_fignums() == []

    def test_without_start_time_raises(self):
        site = get_fyst_site()
        trajectory = _make_simple_trajectory(with_start_time=False)
        modules = {"boresight": InstrumentOffset(dx=0.0, dy=0.0, name="boresight")}

        with pytest.raises(ValueError, match="start_time"):
            plot_hit_map(trajectory, modules=modules, site=site, show=False)

    def test_bad_inputs_raise_before_a_figure_exists(self):
        """An empty module mapping, a non-positive bin size or FOV radius are refused.

        All are checked before ``plt.subplots``: an empty mapping asks for a
        zero-column figure, a zero bin size for an empty histogram grid, and
        a negative FOV radius for a padding that drops samples.
        Either way the call would otherwise fail with an open figure left
        behind, so a loop over many plots would leak one per failure.
        """
        site = get_fyst_site()
        trajectory = _make_simple_trajectory(with_start_time=True)
        modules = {"boresight": InstrumentOffset(dx=0.0, dy=0.0, name="boresight")}

        plt.close("all")
        with pytest.raises(ValueError, match="modules must not be empty"):
            plot_hit_map(trajectory, modules={}, site=site, show=False)
        with pytest.raises(ValueError, match="bin_size must be positive"):
            plot_hit_map(trajectory, modules=modules, site=site, bin_size=0.0, show=False)
        for fov_radius_deg in (-0.3, 0.0, float("nan")):
            with pytest.raises(ValueError, match="fov_radius_deg"):
                plot_hit_map(
                    trajectory,
                    modules=modules,
                    site=site,
                    fov_radius_deg=fov_radius_deg,
                    show=False,
                )
        assert plt.get_fignums() == []

    def test_fov_radius_coverage_mode(self):
        pytest.importorskip("scipy")
        site = get_fyst_site()
        trajectory = _make_simple_trajectory(with_start_time=True)
        modules = {"boresight": InstrumentOffset(dx=0.0, dy=0.0, name="boresight")}

        fig = plot_hit_map(
            trajectory,
            modules=modules,
            site=site,
            fov_radius_deg=0.55,
            show=False,
        )

        assert isinstance(fig, Figure)
        assert _hit_total(fig.axes[0]) == pytest.approx(100.0)  # every sample binned once

    def test_footprint_contour_is_drawn_on_filled_maps_only(self):
        """A raw track gets no contour, which would outline every sample."""
        pytest.importorskip("scipy")
        site = get_fyst_site()
        trajectory = _make_simple_trajectory(with_start_time=True)
        modules = {"boresight": InstrumentOffset(dx=0.0, dy=0.0, name="boresight")}

        raw = plot_hit_map(trajectory, modules=modules, site=site, fov_radius_deg=None, show=False)
        filled = plot_hit_map(
            trajectory, modules=modules, site=site, fov_radius_deg=0.55, show=False
        )

        assert not any(isinstance(c, ContourSet) for c in raw.axes[0].collections)
        assert any(isinstance(c, ContourSet) for c in filled.axes[0].collections)

    def test_smooth_sigma(self):
        """plot_hit_map with Gaussian smoothing runs without errors."""
        pytest.importorskip("scipy")
        site = get_fyst_site()
        trajectory = _make_simple_trajectory(with_start_time=True)
        modules = {"boresight": InstrumentOffset(dx=0.0, dy=0.0, name="boresight")}

        fig = plot_hit_map(
            trajectory,
            modules=modules,
            site=site,
            fov_radius_deg=None,
            smooth_sigma=1.0,
            show=False,
        )

        assert isinstance(fig, Figure)
        assert _hit_total(fig.axes[0]) == pytest.approx(100.0)  # every sample binned once


class TestHitMapRaTicks:
    """RA ticks sit on whole minutes, or whole seconds on a narrow axis, and never touch."""

    def test_wide_field_labels_are_distinct_and_clear_of_each_other(self):
        """A filled 2 x 2 deg Pong at RA 12h: two-digit-hour labels do not run together.

        Adjacent label extents must leave a visible gap (2 px at the default
        100 dpi), not merely avoid overlapping by a fraction of a pixel.
        """
        pytest.importorskip("scipy")
        trajectory = _pong_2deg_at_ra_12h()
        modules = {"centre": InstrumentOffset(dx=0.0, dy=0.0)}

        fig = plot_hit_map(
            trajectory,
            modules=modules,
            fov_radius_deg=0.65,
            bin_size=0.01,
            smooth_sigma=1.0,
            show=False,
        )

        texts, boxes = _rendered_ra_labels(fig.axes[0])
        assert len(texts) >= 2
        assert len(set(texts)) == len(texts), texts
        boxes = sorted(boxes, key=lambda b: b.x0)
        gaps = [b.x0 - a.x1 for a, b in zip(boxes, boxes[1:])]
        assert min(gaps) >= 2.0, (texts, gaps)

    def test_narrow_field_ticks_are_distinct_whole_minutes(self):
        """About a degree of RA: no label repeats, every tick is a whole minute (0.25 deg)."""
        pytest.importorskip("scipy")
        block = plan_daisy_scan(
            ra=83.633,
            dec=22.014,
            radius=0.5,
            velocity=0.3,
            turn_radius=0.2,
            avoidance_radius=0.0,
            start_acceleration=0.5,
            site=get_fyst_site(),
            start_time=Time("2026-01-15T02:00:00", scale="utc"),
            duration=300.0,
        )
        modules = {"boresight": InstrumentOffset(dx=0.0, dy=0.0)}

        fig = plot_hit_map(
            block.trajectory,
            modules=modules,
            fov_radius_deg=None,
            bin_size=0.01,
            smooth_sigma=1.0,
            show=False,
        )

        ax = fig.axes[0]
        texts, _ = _rendered_ra_labels(ax)
        assert len(texts) >= 2
        assert len(set(texts)) == len(texts), texts
        lo, hi = sorted(ax.get_xlim())
        ticks = [t for t in ax.get_xticks() if lo <= t <= hi]
        assert np.allclose(np.asarray(ticks) / 0.25, np.round(np.asarray(ticks) / 0.25))

    @pytest.mark.parametrize(
        ("ra", "dec", "radius", "velocity", "turn_radius", "start", "bin_size"),
        [
            (83.633, 22.014, 0.05, 0.1, 0.02, "2026-01-15T02:00:00", 0.004),
            # Two-digit hours, the longest labels, on an axis of about 10 s of RA.
            (180.0, -30.0, 0.01, 0.03, 0.005, "2026-03-15T01:00:00", 0.002),
        ],
    )
    def test_sub_minute_axis_is_labelled_in_seconds(
        self, ra, dec, radius, velocity, turn_radius, start, bin_size
    ):
        """A raw daisy under a minute of RA wide: distinct second-resolution labels, apart."""
        block = plan_daisy_scan(
            ra=ra,
            dec=dec,
            radius=radius,
            velocity=velocity,
            turn_radius=turn_radius,
            avoidance_radius=0.0,
            start_acceleration=0.2,
            site=get_fyst_site(),
            start_time=Time(start, scale="utc"),
            timestep=0.1,
            duration=200.0,
        )
        modules = {"boresight": InstrumentOffset(dx=0.0, dy=0.0)}

        fig = plot_hit_map(
            block.trajectory, modules=modules, bin_size=bin_size, fov_radius_deg=None, show=False
        )

        ax = fig.axes[0]
        assert abs(np.diff(ax.get_xlim())[0]) < 0.25  # under one minute of RA
        texts, boxes = _rendered_ra_labels(ax)
        assert len(texts) >= 2, texts
        assert len(set(texts)) == len(texts), texts
        assert all(text.endswith("$^{s}$") for text in texts), texts
        boxes = sorted(boxes, key=lambda b: b.x0)
        gaps = [b.x0 - a.x1 for a, b in zip(boxes, boxes[1:])]
        assert min(gaps) > 0.0, (texts, gaps)

    def test_axis_with_two_whole_minutes_keeps_minute_ticks(self):
        """An axis holding two whole minutes of RA is ticked in whole minutes, as before."""
        # 5h33m50s to 5h35m10s: 80 s of RA holding 5h34m and 5h35m.
        assert _ra_tick_step_deg(83.25 + 50 / 240, 83.75 + 10 / 240) == 0.25


def _load_make_figures():
    """Return the ``docs/make_figures.py`` module, loading it once."""
    if "make_figures" in sys.modules:
        return sys.modules["make_figures"]
    script = Path(__file__).resolve().parents[2] / "docs" / "make_figures.py"
    spec = importlib.util.spec_from_file_location("make_figures", script)
    module = importlib.util.module_from_spec(spec)
    # Registered first: the frozen dataclass in the script resolves its
    # string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _hit_map_panels(fig):
    """Return the visible, titled panels of a figure that draw a hit map."""
    return [
        ax
        for ax in fig.axes
        if ax.get_visible()
        and ax.get_title(loc="left")
        and any(isinstance(c, QuadMesh) for c in ax.collections)
    ]


def _assert_statistics_outside_the_map(fig):
    """Each panel's statistics sit above its data area, clear of its title and inside the figure."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panels = _hit_map_panels(fig)
    assert panels
    for ax in panels:
        texts = [t for t in ax.findobj(Text) if "footprint" in t.get_text()]
        assert len(texts) == 1, [t.get_text() for t in texts]
        stats = texts[0].get_window_extent(renderer)
        (title,) = [
            t for t in ax.findobj(Text) if t.get_text() == ax.get_title(loc="left") and t.get_text()
        ]
        data_area = ax.bbox
        assert stats.y0 >= data_area.y1, (ax.get_title(loc="left"), stats, data_area)
        assert not stats.overlaps(title.get_window_extent(renderer)), ax.get_title(loc="left")
        assert fig.bbox.contains(stats.x0, stats.y0) and fig.bbox.contains(stats.x1, stats.y1)


class TestHitMapStatistics:
    """The statistics are drawn above the map, never over it."""

    @pytest.mark.parametrize("name", ["pong_scan", "daisy_scan", "constant_el_scan"])
    def test_documentation_figures(self, name):
        pytest.importorskip("scipy")
        make_figures = _load_make_figures()
        (spec,) = [spec for spec in make_figures.REGISTRY if spec.name == name]

        with plt.rc_context(make_figures.RC_PARAMS):
            fig = spec.build(spec.size_in)
            _assert_statistics_outside_the_map(fig)

    def test_default_seven_panel_figure(self):
        pytest.importorskip("scipy")
        trajectory = _make_simple_trajectory(with_start_time=True)

        fig = plot_hit_map(trajectory, show=False)

        assert len(_hit_map_panels(fig)) == 7
        _assert_statistics_outside_the_map(fig)


# === plot_trajectory ===


class TestPlotTrajectory:
    """The three-panel figure builds with or without a ``start_time``."""

    def test_returns_figure_for_simple_trajectory(self):
        trajectory = _make_simple_trajectory(with_start_time=False)

        fig = plot_trajectory(trajectory, show=False)

        assert isinstance(fig, Figure)
        # plot_trajectory creates a 3-panel figure (az/t, el/t, sky track).
        assert len(fig.axes) == 3

    def test_returns_figure_with_start_time(self):
        """plot_trajectory ignores ``start_time`` and returns a Figure."""
        trajectory = _make_simple_trajectory(with_start_time=True)

        fig = plot_trajectory(trajectory, show=False)

        assert isinstance(fig, Figure)

    def test_az_el_panel_is_drawn_at_the_true_angular_shape(self):
        """At el 60 one degree of azimuth spans half a degree on sky."""
        n = 50
        times = np.linspace(0.0, 10.0, n)
        az = np.linspace(180.0, 182.0, n)
        el = np.full(n, 60.0)
        trajectory = Trajectory(
            times=times,
            az=az,
            el=el,
            az_vel=np.gradient(az, times),
            el_vel=np.gradient(el, times),
        )

        fig = plot_trajectory(trajectory, show=False)

        assert fig.axes[2].get_aspect() == pytest.approx(1.0 / np.cos(np.radians(60.0)))

    def test_title_is_the_suptitle_of_its_own_figure(self):
        trajectory = _make_simple_trajectory(with_start_time=False)

        fig = plot_trajectory(trajectory, title="One period", show=False)

        assert fig.get_suptitle() == "One period"
        assert fig.axes[0].get_title() == "Az vs Time"

    def test_caller_axes_take_exactly_three_panels(self, monkeypatch):
        """Three caller axes are drawn into, unlaid-out and unshown; two are refused."""
        trajectory = _make_simple_trajectory(with_start_time=False)
        fig, axes = plt.subplots(1, 3)
        show_calls: list[int] = []
        layout_calls: list[int] = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: show_calls.append(1))
        monkeypatch.setattr(Figure, "tight_layout", lambda self, *a, **k: layout_calls.append(1))

        out = plot_trajectory(trajectory, title="composed", axes=axes)

        assert out is fig
        assert plt.get_fignums() == [fig.number]
        assert [ax.get_title() for ax in axes] == ["composed", "El vs Time", "Az/El Track"]
        assert show_calls == []
        assert layout_calls == []

        two_fig, two_axes = plt.subplots(1, 2)
        with pytest.raises(ValueError, match="axes has 2 entries"):
            plot_trajectory(trajectory, axes=two_axes, show=False)
        assert all(not ax.lines for ax in two_axes)
        assert plt.get_fignums() == [fig.number, two_fig.number]


# === _make_disk_kernel ===


class TestMakeDiskKernel:
    """The disk kernel is a normalized, symmetric (2*ceil(r)+1)-square array."""

    def test_shape_matches_radius(self):
        """The kernel is (2*ceil(r)+1) x (2*ceil(r)+1) bins."""
        kernel = _make_disk_kernel(radius_bins=3.0)
        assert kernel.shape == (7, 7)

    def test_non_integer_radius(self):
        """Non-integer radii use ceil() for the bounding box size."""
        kernel = _make_disk_kernel(radius_bins=2.5)
        assert kernel.shape == (7, 7)  # 2*ceil(2.5) + 1 = 7

    def test_normalized_to_unit_sum(self):
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
    """Degrees render as hours and minutes, wrapping at 360."""

    def test_format_zero(self):
        """0 degrees is 0h00m."""
        result = _format_ra_hm(0.0, None)
        # Exact LaTeX token: hours=0, minutes=00.
        assert result == "0$^{h}$00$^{m}$"

    def test_format_180_is_12h(self):
        """180 degrees = 12h on the RA hour scale."""
        result = _format_ra_hm(180.0, None)
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

    def test_a_tick_just_below_an_hour_rounds_up(self):
        """The minutes round with the hour, so no label reads 60 minutes."""
        assert _format_ra_hm(44.9, None) == "3$^{h}$00$^{m}$"
        assert _format_ra_hm(179.999, None) == "12$^{h}$00$^{m}$"


class TestFormatRaHms:
    """Degrees render as hours, minutes and seconds, in the hour-minute style."""

    def test_format_seconds(self):
        """83.625 deg is 5h34m30s, and 180 deg + 1 s of RA is 12h00m01s."""
        assert _format_ra_hms(83.625, None) == "5$^{h}$34$^{m}$30$^{s}$"
        assert _format_ra_hms(180.0 + 1.0 / 240.0, None) == "12$^{h}$00$^{m}$01$^{s}$"

    def test_a_tick_just_below_a_minute_rounds_up(self):
        """The seconds round with the minute and hour, so no label reads 60 seconds."""
        assert _format_ra_hms(44.9999, None) == "3$^{h}$00$^{m}$00$^{s}$"
        assert _format_ra_hms(359.9999, None) == "0$^{h}$00$^{m}$00$^{s}$"


# === _format_dec_deg ===


class TestFormatDecDeg:
    """Dec renders in degrees with a LaTeX degree marker and a sign."""

    def test_format_zero(self):
        """0 degrees should render a '0' with a degree marker."""
        assert _format_dec_deg(0.0, None) == "0$^\\circ$"

    def test_format_positive(self):
        """Positive dec values should render the degrees."""
        assert _format_dec_deg(30.0, None) == "30$^\\circ$"

    def test_format_negative(self):
        """Negative dec values should include a minus sign."""
        assert _format_dec_deg(-30.0, None) == "-30$^\\circ$"

    def test_sub_degree_ticks_stay_distinct(self):
        """Half-degree ticks a degree apart get distinct labels."""
        assert _format_dec_deg(-29.5, None) != _format_dec_deg(-30.5, None)


# === package hygiene ===


def test_import_isolation():
    """Importing the package (incl. overhead and visualization) must not import matplotlib."""
    code = (
        "import sys\n"
        "import fyst_trajectories\n"
        "import fyst_trajectories.overhead\n"
        "import fyst_trajectories.visualization\n"
        "leaked = sorted(m for m in sys.modules if m.startswith('matplotlib'))\n"
        "sys.exit(1 if leaked else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, (
        f"package import eagerly loaded matplotlib\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
