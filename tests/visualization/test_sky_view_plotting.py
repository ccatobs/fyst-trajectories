"""Tests for the instantaneous all-sky view renderer.

Structure-level checks (figure/axes contents, mask math, footprint
geometry, guards), not pixel-perfect output, following the conventions of
the sibling plotting tests. The whole file skips when matplotlib is not
installed; the import-isolation test in test_overhead_plotting.py covers
this module too.
"""

import numpy as np
import pytest
from astropy.time import Time

# Skip the entire file if matplotlib isn't installed (optional extra).
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.contour import QuadContourSet  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from fyst_trajectories import Coordinates, get_fyst_site  # noqa: E402
from fyst_trajectories.observability import Target, TargetKind  # noqa: E402
from fyst_trajectories.primecam import (  # noqa: E402
    MODULE_FOV_RADIUS_DEG,
    PRIMECAM_MODULES,
)
from fyst_trajectories.sun_models import make_sun_safe  # noqa: E402
from fyst_trajectories.visualization import plot_sky_view  # noqa: E402
from fyst_trajectories.visualization.sky_view import (  # noqa: E402
    FOOTPRINT_COLOR,
    _footprint_on_sky,
    _policy_masks,
)
from fyst_trajectories.visualization.visibility import SUN_COLOR  # noqa: E402

# Sun well up (el ~65) and the Moon up (el ~43) at FYST; deterministic ephemeris.
T0 = Time("2026-11-15T18:00:00", scale="utc")
NIGHT = Time("2026-11-15T06:00:00", scale="utc")  # Sun el ~-41
STEP = 6.0  # coarse policy grid keeps every render cheap


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to keep memory bounded."""
    yield
    plt.close("all")


def _zone_fills(ax):
    return [c for c in ax.collections if isinstance(c, QuadContourSet)]


def _footprint_lines(ax):
    return [line for line in ax.get_lines() if line.get_color() == FOOTPRINT_COLOR]


def _legend_texts(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


def _grid(step=STEP):
    r_edges = np.linspace(0.0, 90.0, round(90.0 / step) + 1)
    az_edges = np.linspace(0.0, 360.0, round(360.0 / step) + 1)
    az_c = 0.5 * (az_edges[:-1] + az_edges[1:])
    el_c = 90.0 - 0.5 * (r_edges[:-1] + r_edges[1:])
    return np.meshgrid(az_c, el_c)


class _StubModel:
    """Minimal injected model: everything unsafe, recognizable describe."""

    describe = "stub policy"

    def __call__(self, az_deg, el_deg, time):
        return False

    def batch(self, az_deg, el_deg, times):
        return np.zeros(np.shape(np.asarray(az_deg)), dtype=bool)


class _BadShapeModel:
    describe = "bad shape"

    def batch(self, az_deg, el_deg, times):
        return np.ones(3, dtype=bool)


# ---------------------------------------------------------------------------
# Figure structure and legends
# ---------------------------------------------------------------------------


def test_default_returns_polar_figure_with_scalar_shading():
    fig = plot_sky_view(T0, grid_step_deg=STEP, show=False)
    assert isinstance(fig, Figure)
    (ax,) = fig.axes
    assert ax.name == "polar"
    # Scalar default draws two zone fills: exclusion + warning band.
    assert len(_zone_fills(ax)) == 2
    labels = _legend_texts(ax)
    assert any("exclusion" in text for text in labels)
    assert any("warning" in text for text in labels)
    # The site radii appear in the legend, not hardcoded numbers.
    cfg = get_fyst_site().sun_avoidance
    assert any(f"< {cfg.exclusion_radius:.0f}" in text for text in labels)
    # The Sun and every drawn body are identified in the legend.
    assert "Sun" in labels
    assert "moon" in labels


def test_sun_disabled_site_draws_no_zone():
    site = get_fyst_site(sun_avoidance_enabled=False)
    fig = plot_sky_view(T0, site=site, grid_step_deg=STEP, show=False)
    (ax,) = fig.axes
    assert len(_zone_fills(ax)) == 0
    assert not any("exclusion" in text for text in _legend_texts(ax))


def test_labels_off_keeps_legend_identification():
    fig = plot_sky_view(T0, labels=False, grid_step_deg=STEP, show=False)
    (ax,) = fig.axes
    assert len(ax.texts) == 0  # no on-chart annotations
    labels = _legend_texts(ax)
    assert "Sun" in labels
    assert "moon" in labels


def test_night_sun_reported_in_legend_not_drawn():
    fig = plot_sky_view(NIGHT, grid_step_deg=STEP, show=False)
    (ax,) = fig.axes
    # No Sun marker on the chart, but the zone shading remains (no night
    # waiver in the default policy) and the legend states the Sun is down.
    assert not [line for line in ax.get_lines() if line.get_color() == SUN_COLOR]
    assert len(_zone_fills(ax)) == 2
    assert any("Sun below horizon" in text for text in _legend_texts(ax))


def test_injected_model_drives_shading_and_legend():
    fig = plot_sky_view(T0, sun_model=_StubModel(), grid_step_deg=STEP, show=False)
    (ax,) = fig.axes
    assert len(_zone_fills(ax)) == 1  # no warning band for injected models
    assert any("stub policy" in text for text in _legend_texts(ax))


def test_injected_model_shape_guard():
    with pytest.raises(ValueError, match="sun_model.batch returned shape"):
        plot_sky_view(T0, sun_model=_BadShapeModel(), grid_step_deg=STEP, show=False)


# ---------------------------------------------------------------------------
# Zone mask math (probed directly, independent of the artists)
# ---------------------------------------------------------------------------


def test_scalar_zone_tracks_true_separation():
    """Cells at the Sun are unsafe; a far low-elevation cell is clear.

    The far probe sits at the opposite azimuth at LOW elevation: at the
    Sun's own elevation the azimuth circle is compressed by cos(el), so
    90 deg of azimuth is not 90 deg of separation.
    """
    site = get_fyst_site()
    coords = Coordinates(site)
    sun_az, sun_el = coords.get_sun_altaz(T0)
    az_grid, el_grid = _grid()
    unsafe, warn = _policy_masks(coords, site, None, az_grid, el_grid, T0, sun_az, sun_el)
    at_sun = np.unravel_index(
        np.argmin(
            np.asarray(coords.angular_separation(az_grid.ravel(), el_grid.ravel(), sun_az, sun_el))
        ),
        az_grid.shape,
    )
    assert bool(unsafe[at_sun])
    far = (
        int(np.argmin(np.abs(el_grid[:, 0] - 25.0))),
        int(np.argmin(np.abs(az_grid[0, :] - (sun_az + 180.0) % 360.0))),
    )
    sep_far = coords.angular_separation(az_grid[far], el_grid[far], sun_az, sun_el)
    assert sep_far > site.sun_avoidance.warning_radius
    assert not bool(unsafe[far])
    assert not bool(warn[far])
    # The warning band is an annulus: nonempty, disjoint from unsafe.
    assert warn.any()
    assert not (warn & unsafe).any()


def test_scalar_string_model_matches_default_mask():
    """make_sun_safe("scalar") and the sun_model=None path agree cell-for-cell."""
    site = get_fyst_site()
    coords = Coordinates(site)
    sun_az, sun_el = coords.get_sun_altaz(T0)
    az_grid, el_grid = _grid()
    default_unsafe, _ = _policy_masks(coords, site, None, az_grid, el_grid, T0, sun_az, sun_el)
    model_unsafe, model_warn = _policy_masks(
        coords, site, make_sun_safe("scalar", site=site), az_grid, el_grid, T0, sun_az, sun_el
    )
    assert model_warn is None
    np.testing.assert_array_equal(model_unsafe, default_unsafe)


def test_disabled_site_masks_are_none():
    site = get_fyst_site(sun_avoidance_enabled=False)
    coords = Coordinates(site)
    az_grid, el_grid = _grid()
    assert _policy_masks(coords, site, None, az_grid, el_grid, T0, 0.0, 0.0) == (None, None)


def test_cad_model_renders():
    pytest.importorskip("sun_avoidance")
    fig = plot_sky_view(T0, sun_model="cad", grid_step_deg=STEP, show=False)
    (ax,) = fig.axes
    assert len(_zone_fills(ax)) == 1
    assert any("unsafe" in text for text in _legend_texts(ax))


# ---------------------------------------------------------------------------
# Footprint geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("el_b", [20.0, 45.0, 89.0, 90.0])
def test_footprint_outlines_are_exact_on_sky(el_b):
    """Every outline's true separation from the boresight matches the focal plane.

    Independent oracle (angular_separation): the center module's ring sits
    at exactly the FOV radius; each off-axis ring spans [rho - fov,
    rho + fov] around its module's radial distance. Valid at every
    elevation INCLUDING 90; a project-then-flatten implementation or a
    parallactic-angle contamination fails this.
    """
    site = get_fyst_site()
    coords = Coordinates(site)
    outlines = _footprint_on_sky(30.0, el_b, site, PRIMECAM_MODULES, MODULE_FOV_RADIUS_DEG)
    assert len(outlines) == 7
    spans = sorted(
        (float(np.min(sep)), float(np.max(sep)))
        for sep in (
            np.asarray(coords.angular_separation(ring_az, ring_el, 30.0, el_b))
            for ring_az, ring_el in outlines
        )
    )
    assert spans[0] == pytest.approx((MODULE_FOV_RADIUS_DEG, MODULE_FOV_RADIUS_DEG), abs=1e-6)
    rhos = sorted(
        np.hypot(offset.dx, offset.dy) / 60.0
        for offset in {id(o): o for o in PRIMECAM_MODULES.values()}.values()
        if offset.dx or offset.dy
    )
    for (lo, hi), rho in zip(spans[1:], rhos):
        assert (lo, hi) == pytest.approx(
            (rho - MODULE_FOV_RADIUS_DEG, rho + MODULE_FOV_RADIUS_DEG), abs=2e-3
        )


def test_boresight_tuple_draws_seven_module_outlines():
    fig = plot_sky_view(T0, boresight=(120.0, 45.0), grid_step_deg=STEP, show=False)
    (ax,) = fig.axes
    assert len(_footprint_lines(ax)) == 7
    assert any("footprint" in text for text in _legend_texts(ax))


def test_no_boresight_no_footprint():
    fig = plot_sky_view(T0, grid_step_deg=STEP, show=False)
    (ax,) = fig.axes
    assert len(_footprint_lines(ax)) == 0


def test_boresight_by_name_and_below_horizon_raises():
    fig = plot_sky_view(T0, boresight="moon", grid_step_deg=STEP, show=False)
    assert len(_footprint_lines(fig.axes[0])) == 7

    # A fixed source at a below-horizon pointing, by construction.
    coords = Coordinates(get_fyst_site())
    ra, dec = coords.altaz_to_radec(0.0, -45.0, T0)
    down = Target("down_under", TargetKind.FIXED, ra_deg=float(ra), dec_deg=float(dec))
    with pytest.raises(ValueError, match="below the horizon"):
        plot_sky_view(T0, boresight=down, grid_step_deg=STEP, show=False)


def test_boresight_pair_validation():
    with pytest.raises(ValueError, match="boresight elevation"):
        plot_sky_view(T0, boresight=(120.0, 95.0), grid_step_deg=STEP, show=False)
    with pytest.raises(ValueError, match="boresight elevation"):
        plot_sky_view(T0, boresight=(120.0, 0.0), grid_step_deg=STEP, show=False)
    with pytest.raises(ValueError, match="must be \\(az, el\\)"):
        plot_sky_view(T0, boresight=(120.0, 45.0, 1.0), grid_step_deg=STEP, show=False)
    # An ndarray pair is accepted like a tuple.
    fig = plot_sky_view(T0, boresight=np.array([120.0, 45.0]), grid_step_deg=STEP, show=False)
    assert len(_footprint_lines(fig.axes[0])) == 7


# ---------------------------------------------------------------------------
# Composition contract and guards
# ---------------------------------------------------------------------------


def test_ax_composition_requires_polar_and_reuses_figure(monkeypatch):
    fig, rect_ax = plt.subplots()
    with pytest.raises(ValueError, match="polar axes"):
        plot_sky_view(T0, grid_step_deg=STEP, ax=rect_ax)

    show_calls = []
    layout_calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: show_calls.append(1))
    monkeypatch.setattr(Figure, "tight_layout", lambda self, *a, **k: layout_calls.append(1))
    fig2 = plt.figure()
    polar_ax = fig2.add_subplot(1, 1, 1, projection="polar")
    out = plot_sky_view(T0, grid_step_deg=STEP, ax=polar_ax, title="composed")
    assert out is fig2
    assert polar_ax.get_title() == "composed"
    # Composition mode never calls plt.show() or layouts the caller's figure.
    assert show_calls == []
    assert layout_calls == []


def test_no_figure_leak_on_bad_target():
    with pytest.raises(ValueError, match="Unknown target"):
        plot_sky_view(T0, targets=["not_a_body"], grid_step_deg=STEP, show=False)
    assert plt.get_fignums() == []
    with pytest.raises(ValueError):
        plot_sky_view(T0, boresight=(0.0, -5.0), grid_step_deg=STEP, show=False)
    assert plt.get_fignums() == []


def test_input_validation():
    with pytest.raises(ValueError, match="scalar Time"):
        plot_sky_view(Time(["2026-11-15T18:00:00"], scale="utc"), show=False)
    with pytest.raises(ValueError, match="grid_step_deg"):
        plot_sky_view(T0, grid_step_deg=0.0, show=False)
    with pytest.raises(ValueError, match="targets must not be empty"):
        plot_sky_view(T0, targets=[], grid_step_deg=STEP, show=False)
    with pytest.raises(ValueError, match="el_min"):
        plot_sky_view(T0, el_min=120.0, grid_step_deg=STEP, show=False)
    with pytest.raises(ValueError, match="el_min"):
        plot_sky_view(T0, el_min=float("nan"), grid_step_deg=STEP, show=False)


def test_tz_accepts_non_utc_time_scale():
    from zoneinfo import ZoneInfo

    tai = Time("2026-11-15T18:00:00", scale="tai")
    fig = plot_sky_view(tai, tz=ZoneInfo("America/Santiago"), grid_step_deg=STEP, show=False)
    assert "America/Santiago" in fig.axes[0].get_title()
