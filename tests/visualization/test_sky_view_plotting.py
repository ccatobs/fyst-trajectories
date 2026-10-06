"""Tests for the instantaneous all-sky view renderer.

Structure-level checks (figure/axes contents, zone geometry, footprint
geometry, guards), not pixel-perfect output, following the conventions of
the sibling plotting tests. The whole file skips when matplotlib is not
installed; the import-isolation test in test_plotting.py covers
this module too.
"""

import numpy as np
import pytest
from astropy.time import Time

# Skip the entire file if matplotlib isn't installed (optional extra).
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402
from _sun_stubs import HAVE_SUN_AVOIDANCE, fake_sun_model  # noqa: E402
from matplotlib.contour import QuadContourSet  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.projections.polar import PolarAxes  # noqa: E402

from fyst_trajectories import Coordinates, get_fyst_site  # noqa: E402
from fyst_trajectories.exceptions import PointingError  # noqa: E402
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
    _policy_zones,
)
from fyst_trajectories.visualization.visibility import SUN_COLOR  # noqa: E402

# Sun well up (el ~65) and the Moon up (el ~43) at FYST; deterministic ephemeris.
T0 = Time("2026-11-15T18:00:00", scale="utc")
NIGHT = Time("2026-11-15T06:00:00", scale="utc")  # Sun el ~-41
EVENING = Time("2026-11-15T22:00:00", scale="utc")  # Sun el ~11
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


def _edges(step=STEP):
    """Return the renderer's grid edges: azimuth in radians, ``r = 90 - el`` in degrees."""
    theta_edges = np.deg2rad(np.linspace(0.0, 360.0, round(360.0 / step) + 1))
    r_edges = np.linspace(0.0, 90.0, round(90.0 / step) + 1)
    return theta_edges, r_edges


def _nodes(step=STEP):
    """Azimuth and elevation, in degrees, of every node of the grid."""
    theta_edges, r_edges = _edges(step)
    return np.meshgrid(np.rad2deg(theta_edges), 90.0 - r_edges)


def _zones(sun_model, *, time=T0, site=None):
    site = get_fyst_site() if site is None else site
    coords = Coordinates(site)
    sun_az, sun_el = coords.get_sun_altaz(time)
    theta_edges, r_edges = _edges()
    return _policy_zones(coords, site, sun_model, theta_edges, r_edges, time, sun_az, sun_el)


class _BadShapeModel:
    """Deliberately breaks the batch shape contract, so it stays hand-written."""

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
    assert any(f"<= {cfg.exclusion_radius:.0f}" in text for text in labels)
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
    fig = plot_sky_view(
        T0,
        sun_model=fake_sun_model(False, describe="stub policy"),
        grid_step_deg=STEP,
        show=False,
    )
    (ax,) = fig.axes
    assert len(_zone_fills(ax)) == 1  # no warning band for injected models
    assert any("stub policy" in text for text in _legend_texts(ax))


def test_injected_model_shape_guard():
    with pytest.raises(ValueError, match="sun_model.batch returned shape"):
        plot_sky_view(T0, sun_model=_BadShapeModel(), grid_step_deg=STEP, show=False)


# ---------------------------------------------------------------------------
# Zone geometry (probed directly, independent of the artists)
# ---------------------------------------------------------------------------


def test_scalar_zone_tracks_true_separation():
    """The default zones are the true Sun separation on the nodes, cut at the site radii.

    Each layer shades ``lower < values <= upper``, so a node exactly at a
    radius falls in the zone inside it (the at-radius-is-unsafe boundary).
    """
    site = get_fyst_site()
    coords = Coordinates(site)
    sun_az, sun_el = coords.get_sun_altaz(T0)
    unsafe, warn = _zones(None, site=site)
    az_nodes, el_nodes = _nodes()
    separation = np.asarray(
        coords.angular_separation(az_nodes.ravel(), el_nodes.ravel(), sun_az, sun_el)
    ).reshape(az_nodes.shape)
    cfg = site.sun_avoidance
    assert unsafe.on_nodes and warn.on_nodes
    np.testing.assert_allclose(unsafe.values, separation)
    np.testing.assert_allclose(warn.values, separation)
    assert unsafe.lower < unsafe.values.min()
    assert unsafe.upper == cfg.exclusion_radius
    assert (warn.lower, warn.upper) == (cfg.exclusion_radius, cfg.warning_radius)
    # Both zones are nonempty at this instant, and the band lies outside the zone.
    assert (separation <= cfg.exclusion_radius).any()
    assert ((separation > cfg.exclusion_radius) & (separation <= cfg.warning_radius)).any()


def test_scalar_string_model_matches_default_zone():
    """make_sun_safe("scalar") and the sun_model=None path draw the same edge."""
    site = get_fyst_site()
    default_unsafe, _ = _zones(None, site=site)
    model_unsafe, model_warn = _zones(make_sun_safe("scalar", site=site), site=site)
    assert model_warn is None
    assert model_unsafe.on_nodes
    np.testing.assert_allclose(
        model_unsafe.values - model_unsafe.upper,
        default_unsafe.values - default_unsafe.upper,
        atol=1e-9,
    )


def test_disabled_site_zones_are_none():
    site = get_fyst_site(sun_avoidance_enabled=False)
    coords = Coordinates(site)
    theta_edges, r_edges = _edges()
    assert _policy_zones(coords, site, None, theta_edges, r_edges, T0, 0.0, 0.0) == (None, None)


def _separation_verdict(radius):
    """Build a point verdict that is safe beyond ``radius`` deg of the Sun at ``T0``."""
    coords = Coordinates(get_fyst_site())
    sun_az, sun_el = coords.get_sun_altaz(T0)

    def verdict(az, el, time):
        return np.asarray(coords.angular_separation(az, el, sun_az, sun_el)) > radius

    return verdict


@pytest.mark.parametrize(
    ("threshold", "on_nodes"),
    [(40.0, True), (30.0, False), (None, False)],
    ids=["consistent-threshold", "inconsistent-threshold", "no-threshold"],
)
def test_injected_model_uses_field_only_when_threshold_reproduces_verdicts(threshold, on_nodes):
    """The field is drawn only where the model's threshold reproduces its own verdicts.

    A model whose threshold disagrees with its verdicts, or that has none,
    is drawn from its cell verdicts.
    """
    model = fake_sun_model(_separation_verdict(40.0), threshold=threshold)
    unsafe, warn = _zones(model)
    assert warn is None
    assert unsafe.on_nodes is on_nodes
    if on_nodes:
        assert unsafe.upper == 0.0
        assert unsafe.lower < unsafe.values.min()
    else:
        assert set(np.unique(unsafe.values)) <= {0.0, 1.0}
        assert (unsafe.lower, unsafe.upper) == (0.5, 1.5)
    fig = plot_sky_view(T0, sun_model=model, grid_step_deg=STEP, show=False)
    assert len(_zone_fills(fig.axes[0])) == 1


@pytest.mark.skipif(not HAVE_SUN_AVOIDANCE, reason="needs the shared sun-avoidance library")
def test_cad_zone_uses_field_unless_its_threshold_disagrees():
    """The CAD model is contoured; with its island check on it can fall back.

    The figure's instant takes the field. At ``EVENING`` the island check
    clears nodes its threshold calls unsafe, so the zone is drawn from the
    cell verdicts there.
    """
    site = get_fyst_site()
    coords = Coordinates(site)
    unsafe, _ = _zones(make_sun_safe("cad", site=site), site=site)
    assert unsafe.on_nodes

    island = make_sun_safe("cad", site=site, island_check=True)
    az_nodes, el_nodes = _nodes()
    sun_az, sun_el = coords.get_sun_altaz(EVENING)
    field = np.asarray(
        coords.angular_separation(az_nodes.ravel(), el_nodes.ravel(), sun_az, sun_el)
    ) - np.asarray(island.threshold(az_nodes.ravel(), el_nodes.ravel(), EVENING))
    verdicts = np.asarray(island.batch(az_nodes.ravel(), el_nodes.ravel(), EVENING), dtype=bool)
    assert (((field > 0.0) != verdicts) & (np.abs(field) >= 1e-9)).any()
    unsafe, _ = _zones(island, time=EVENING, site=site)
    assert not unsafe.on_nodes


@pytest.fixture
def recorded_fills(monkeypatch):
    """Record the arguments of every ``contourf`` call made on a polar axes."""
    calls = []
    original = PolarAxes.contourf

    def recording(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PolarAxes, "contourf", recording)
    return calls


def _boundary_vertices(call):
    """Vertices ``(az, el)`` of the level lines bounding one recorded fill.

    The lines are contoured from the same field the fill was drawn from, at
    the fill's levels inside the field's range, so they are the zone edge
    alone: the seam and rim segments of the fill polygon are not on them.
    """
    (theta, r, values), levels = call[0][:3], call[1]["levels"]
    values = np.asarray(values, dtype=float)
    inside = [lv for lv in levels if values.min() < lv < values.max()]
    if not inside:
        return np.empty(0), np.empty(0)
    scratch_fig, scratch_ax = plt.subplots()
    lines = scratch_ax.contour(theta, r, values, levels=inside)
    vertices = [seg for segs in lines.allsegs for seg in segs if len(seg)]
    plt.close(scratch_fig)
    stacked = np.concatenate(vertices) if vertices else np.empty((0, 2))
    return np.rad2deg(stacked[:, 0]) % 360.0, 90.0 - stacked[:, 1]


def test_scalar_zone_edges_lie_on_the_radii(recorded_fills):
    """The drawn exclusion and warning edges sit on the site radii, not on cell edges.

    Measured on the boundary lines between elevation 1 and 89 deg at the
    coarse test grid; a fill traced from cell verdicts is off by a large
    fraction of a cell.
    """
    site = get_fyst_site()
    coords = Coordinates(site)
    sun_az, sun_el = coords.get_sun_altaz(T0)
    radii = np.array([site.sun_avoidance.exclusion_radius, site.sun_avoidance.warning_radius])
    plot_sky_view(T0, grid_step_deg=STEP, show=False)
    assert len(recorded_fills) == 2
    for call in recorded_fills:
        az, el = _boundary_vertices(call)
        keep = (el > 1.0) & (el < 89.0)
        assert keep.sum() > 10
        sep = np.asarray(coords.angular_separation(az[keep], el[keep], sun_az, sun_el))
        error = np.min(np.abs(sep[:, None] - radii[None, :]), axis=1)
        assert error.max() <= 0.05, f"edge off the radius by up to {error.max():.3f} deg"


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
    with pytest.raises(PointingError, match="below the horizon"):
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
