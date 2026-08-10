"""Tests for the target-visibility and array-footprint renderers.

Structure-level checks (figure/axes contents, curve masks, guards), not
pixel-perfect output, following the conventions of the sibling plotting
tests. The whole file skips when matplotlib is not installed; the
import-isolation test in test_overhead_plotting.py covers this module too.
"""

import dataclasses
from datetime import timedelta, timezone

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

# Skip the entire file if matplotlib isn't installed (optional extra).
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from fyst_trajectories import Coordinates, get_fyst_site  # noqa: E402
from fyst_trajectories.observability import Target, TargetKind  # noqa: E402
from fyst_trajectories.primecam import (  # noqa: E402
    MODULE_FOV_RADIUS_DEG,
    PRIMECAM_MODULES,
)
from fyst_trajectories.visualization import (  # noqa: E402
    DEFAULT_VISIBILITY_TARGETS,
    plot_array_footprint,
    plot_observability_windows,
    plot_visibility,
)
from fyst_trajectories.visualization.visibility import (  # noqa: E402
    EVENT_LINE_COLOR,
    EXCLUSION_COLOR,
    SUN_COLOR,
    WARNING_COLOR,
)

T0 = Time("2026-11-15T16:00:00", scale="utc")  # ~13:00 Chile local


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to keep memory bounded."""
    yield
    plt.close("all")


def _lines_with_color(ax, color):
    return [line for line in ax.get_lines() if line.get_color() == color]


# ---------------------------------------------------------------------------
# plot_visibility
# ---------------------------------------------------------------------------


def test_visibility_default_returns_two_panel_figure():
    fig = plot_visibility(T0, show=False)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    el_ax = fig.axes[0]
    labels = [t.get_text() for t in el_ax.get_legend().get_texts()]
    for name in DEFAULT_VISIBILITY_TARGETS:
        assert name in labels
    assert "Sun" in labels
    # The Sun curve is drawn on both panels.
    assert _lines_with_color(el_ax, SUN_COLOR)
    assert _lines_with_color(fig.axes[1], SUN_COLOR)


def test_default_targets_constant():
    assert DEFAULT_VISIBILITY_TARGETS == (
        "mars",
        "jupiter",
        "saturn",
        "uranus",
        "neptune",
        "moon",
    )


def test_visibility_sun_zone_masks_track_true_separation():
    """The red/orange overlays key on TRUE angular separation, per target.

    A fixed source riding at the Sun's position must be flagged for the
    whole span; the anti-solar source must never be. This is the defence
    against the per-axis band-around-the-Sun mistake.
    """
    site = get_fyst_site()
    coords = Coordinates(site)
    mid = T0 + 12 * u.hour
    sun_ra, sun_dec = coords.get_body_radec("sun", mid)
    at_sun = Target("at_sun", TargetKind.FIXED, ra_deg=float(sun_ra), dec_deg=float(sun_dec))
    anti_sun = Target(
        "anti_sun",
        TargetKind.FIXED,
        ra_deg=float((sun_ra + 180.0) % 360.0),
        dec_deg=float(-sun_dec),
    )
    fig = plot_visibility(T0, [at_sun, anti_sun], panels=("elevation",), show=False)
    ax = fig.axes[0]

    exclusion_overlays = _lines_with_color(ax, EXCLUSION_COLOR)
    warning_overlays = _lines_with_color(ax, WARNING_COLOR)
    assert len(exclusion_overlays) == 2  # one per target, in target order
    assert len(warning_overlays) == 2
    at_sun_y = np.asarray(exclusion_overlays[0].get_ydata(), dtype=float)
    anti_sun_y = np.asarray(exclusion_overlays[1].get_ydata(), dtype=float)
    assert np.isfinite(at_sun_y).sum() > 0  # riding the Sun => flagged
    assert np.isfinite(anti_sun_y).sum() == 0  # anti-solar => never flagged


def test_visibility_radii_come_from_the_site():
    """Legend text and separation-panel guide lines read site.sun_avoidance."""
    site = get_fyst_site(sun_exclusion_radius=30.0, sun_warning_radius=37.0)
    fig = plot_visibility(
        T0, ["mars"], site=site, panels=("elevation", "sun_separation"), show=False
    )
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("< 30°" in label for label in labels)
    assert any("< 37°" in label for label in labels)
    sep_ax = fig.axes[1]
    hline_ys = {
        line.get_ydata()[0]
        for line in sep_ax.get_lines()
        if len(set(np.round(np.asarray(line.get_ydata(), dtype=float), 9))) == 1
    }
    assert 30.0 in hline_ys
    assert 37.0 in hline_ys


def test_visibility_sun_disabled_omits_overlays():
    site = get_fyst_site(sun_avoidance_enabled=False)
    fig = plot_visibility(T0, ["mars"], site=site, panels=("elevation",), show=False)
    ax = fig.axes[0]
    assert _lines_with_color(ax, EXCLUSION_COLOR) == []
    assert _lines_with_color(ax, WARNING_COLOR) == []
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert not any("exclusion" in label for label in labels)


def test_visibility_night_shading_and_event_lines():
    fig = plot_visibility(T0, ["mars"], show=False)
    for ax in fig.axes:
        assert len(ax.collections) >= 1  # night / astronomical-night shading
    # One sunset + one sunrise in this 24 h span; the azimuth panel has no
    # el_min line, so the only dashed event-colored lines are those two.
    az_ax = fig.axes[1]
    event_lines = [
        line for line in _lines_with_color(az_ax, EVENT_LINE_COLOR) if line.get_linestyle() == "--"
    ]
    assert len(event_lines) == 2


def test_visibility_multiday_lines_repeat_labels_do_not():
    fig = plot_visibility(T0, ["mars"], horizon_hours=48.0, show=False)
    el_ax = fig.axes[0]
    texts = [t.get_text() for t in el_ax.texts]
    assert texts.count("sunset") == 1
    assert texts.count("sunrise") == 1
    dashed = [
        line for line in _lines_with_color(el_ax, EVENT_LINE_COLOR) if line.get_linestyle() == "--"
    ]
    assert len(dashed) == 4  # two sunsets + two sunrises in 48 h


def test_visibility_timezone_labels():
    chile = timezone(timedelta(hours=-3))
    fig = plot_visibility(T0, ["mars"], tz=chile, show=False)
    assert "UTC-03:00" in fig.axes[-1].get_xlabel()
    fig2 = plot_visibility(T0, ["mars"], show=False)
    assert "(UTC)" in fig2.axes[-1].get_xlabel()


def test_visibility_into_caller_axes(monkeypatch):
    calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))
    fig, axs = plt.subplots(2, 1)  # deliberately NOT sharex
    fig.suptitle("caller suptitle")
    out = plot_visibility(T0, ["mars"], axes=list(axs))  # default show=True
    assert out is fig
    assert calls == []
    assert len(plt.get_fignums()) == 1
    # Composition contract: the caller's figure-level suptitle is never touched.
    assert fig.get_suptitle() == "caller suptitle"
    # Every supplied axes is date-formatted and aligned, even unshared ones.
    assert axs[0].get_xlim() == axs[1].get_xlim()
    fig.canvas.draw()
    for ax in axs:
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert any(":" in label for label in tick_labels)


def test_visibility_explicit_title_with_caller_axes_goes_on_first_axes():
    fig, axs = plt.subplots(2, 1)
    fig.suptitle("caller suptitle")
    plot_visibility(T0, ["mars"], axes=list(axs), title="my panel title", show=False)
    assert fig.get_suptitle() == "caller suptitle"
    assert axs[0].get_title() == "my panel title"


def test_visibility_validation():
    with pytest.raises(ValueError, match="panel"):
        plot_visibility(T0, ["mars"], panels=("elevation", "bogus"), show=False)
    with pytest.raises(ValueError, match="panels"):
        plot_visibility(T0, ["mars"], panels=(), show=False)
    with pytest.raises(ValueError, match="targets"):
        plot_visibility(T0, [], show=False)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="horizon_hours"):
            plot_visibility(T0, ["mars"], horizon_hours=bad, show=False)
        if bad != 0.0:
            with pytest.raises(ValueError, match="step_minutes"):
                plot_visibility(T0, ["mars"], step_minutes=bad, show=False)
    _, axs = plt.subplots(3, 1)
    with pytest.raises(ValueError, match="axes"):
        plot_visibility(T0, ["mars"], panels=("elevation",), axes=list(axs), show=False)


def test_visibility_no_figure_leak_on_bad_target():
    with pytest.raises(ValueError, match="Unknown target"):
        plot_visibility(T0, ["not_a_body"], show=False)
    assert plt.get_fignums() == []


class _StubSunModel:
    """Duck-typed sun model: unsafe above el 40, flat 60 deg threshold."""

    describe = "stub 60°"

    def __call__(self, az, el, t):
        return bool(self.batch(az, el, t)[0])

    def batch(self, az, el, times):
        az_b, el_b = np.broadcast_arrays(
            np.atleast_1d(np.asarray(az, dtype=float)),
            np.atleast_1d(np.asarray(el, dtype=float)),
        )
        return el_b <= 40.0

    def threshold(self, az, el, times):
        az_b, _ = np.broadcast_arrays(
            np.atleast_1d(np.asarray(az, dtype=float)),
            np.atleast_1d(np.asarray(el, dtype=float)),
        )
        return np.full(az_b.shape, 60.0)


def test_visibility_sun_model_object_drives_overlays():
    """An injected model replaces the radius masks.

    One red overlay keyed to the model's verdicts, no orange warning tier,
    and per-target threshold curves instead of the fixed guide lines.
    """
    fig = plot_visibility(
        T0,
        ["mars"],
        sun_model=_StubSunModel(),
        panels=("elevation", "sun_separation"),
        show=False,
    )
    el_ax, sep_ax = fig.axes
    red = _lines_with_color(el_ax, EXCLUSION_COLOR)
    assert len(red) == 1
    assert _lines_with_color(el_ax, WARNING_COLOR) == []
    # The stub marks el > 40 unsafe: every finite overlay sample sits above 40.
    red_y = np.asarray(red[0].get_ydata(), dtype=float)
    assert np.isfinite(red_y).any()
    assert np.nanmin(red_y) > 40.0
    # Legend names the model; separation panel carries the dashed threshold
    # curve at 60 and no fixed 45/50 guide lines.
    labels = [t.get_text() for t in el_ax.get_legend().get_texts()]
    assert any("unsafe (stub 60°)" in label for label in labels)
    assert any("min safe sep" in label for label in labels)
    dashed = [
        line
        for line in sep_ax.get_lines()
        if line.get_linestyle() == "--" and np.all(np.asarray(line.get_ydata()) == 60.0)
    ]
    assert len(dashed) == 1
    hline_ys = {
        line.get_ydata()[0]
        for line in sep_ax.get_lines()
        if len(set(np.asarray(line.get_ydata(), dtype=float))) == 1
    }
    assert 45.0 not in hline_ys and 50.0 not in hline_ys


def test_visibility_sun_model_scalar_string():
    """sun_model="scalar" works without the shared library."""
    fig = plot_visibility(T0, ["mars"], sun_model="scalar", panels=("sun_separation",), show=False)
    sep_ax = fig.axes[0]
    dashed_45 = [
        line
        for line in sep_ax.get_lines()
        if line.get_linestyle() == "--" and np.all(np.asarray(line.get_ydata()) == 45.0)
    ]
    assert len(dashed_45) == 1  # the scalar model's flat threshold curve (site default)


def test_visibility_sun_model_cad_renders():
    """The real CAD model renders end-to-end (skipped without the library)."""
    pytest.importorskip("sun_avoidance", exc_type=ImportError)
    fig = plot_visibility(
        T0,
        ["jupiter", "moon"],
        sun_model="cad",
        panels=("elevation", "azimuth", "sun_separation"),
        show=False,
    )
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("CAD zone 50-90°" in label for label in labels)


# ---------------------------------------------------------------------------
# plot_array_footprint
# ---------------------------------------------------------------------------


def _circles(fig):
    return [p for p in fig.axes[0].patches if isinstance(p, Circle)]


def test_footprint_draws_each_module_once():
    fig = plot_array_footprint(el=45.0, show=False)
    assert isinstance(fig, Figure)
    circles = _circles(fig)
    # 8 dict keys, but "c"/"center" alias one offset: 7 unique modules.
    assert len(circles) == 7
    assert len({id(offset) for offset in PRIMECAM_MODULES.values()}) == 7
    for circle in circles:
        assert circle.get_radius() == pytest.approx(MODULE_FOV_RADIUS_DEG)


@pytest.mark.parametrize("el", [0.0, 20.0, 45.0, 70.0, 85.0, 90.0])
def test_footprint_is_to_scale(el):
    """Inner-ring radial distances are EXACT at every elevation, including 90.

    The tangent-plane rotation preserves each module's radius; a
    project-through-the-sphere-then-flatten implementation fails this from
    ~el=80 and collapses entirely at el=90.
    """
    ring = PRIMECAM_MODULES["i1"]
    expected = float(np.hypot(ring.dx_deg, ring.dy_deg))
    fig = plot_array_footprint(el=el, show=False)
    radial = sorted(np.hypot(*c.get_center()) for c in _circles(fig))
    assert radial[0] == pytest.approx(0.0, abs=1e-9)  # center module
    for r in radial[1:]:
        assert r == pytest.approx(expected, abs=1e-9)
    # The docstring's falsifiable spans: ~3.6 deg between opposite module
    # centres, ~4.9 deg edge to edge.
    assert 2.0 * expected == pytest.approx(3.56, abs=0.01)
    assert 2.0 * (expected + MODULE_FOV_RADIUS_DEG) == pytest.approx(4.86, abs=0.01)


def test_footprint_rotates_with_elevation():
    """The Nasmyth rotation moves off-axis modules as elevation changes."""
    lo = {
        tuple(np.round(c.get_center(), 6))
        for c in _circles(plot_array_footprint(el=20.0, show=False))
    }
    hi = {
        tuple(np.round(c.get_center(), 6))
        for c in _circles(plot_array_footprint(el=70.0, show=False))
    }
    assert lo != hi
    # The center module is rotation-invariant: the origin is in both sets.
    assert (0.0, 0.0) in lo and (0.0, 0.0) in hi


def test_footprint_nasmyth_port_mirrors_layout():
    """Flipping the Nasmyth port negates the rotation: the layout mirrors in cross-el."""
    site = get_fyst_site()
    other_port = "left" if site.nasmyth_port == "right" else "right"
    flipped = dataclasses.replace(site, nasmyth_port=other_port)
    assert flipped.nasmyth_sign == -site.nasmyth_sign  # precondition
    base = sorted(
        c.get_center() for c in _circles(plot_array_footprint(el=50.0, site=site, show=False))
    )
    mirror = sorted(
        c.get_center() for c in _circles(plot_array_footprint(el=50.0, site=flipped, show=False))
    )
    mirrored_base = sorted((-x, y) for x, y in base)
    for (x1, y1), (x2, y2) in zip(mirrored_base, mirror):
        assert x1 == pytest.approx(x2, abs=1e-6)
        assert y1 == pytest.approx(y2, abs=1e-6)


def test_footprint_into_caller_axes(monkeypatch):
    calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))
    fig, ax = plt.subplots()
    out = plot_array_footprint(el=45.0, ax=ax)  # default show=True
    assert out is fig
    assert calls == []
    assert len(plt.get_fignums()) == 1


def test_footprint_validation():
    for bad_el in (-1.0, 90.5, float("nan")):
        with pytest.raises(ValueError, match="el"):
            plot_array_footprint(el=bad_el, show=False)
    with pytest.raises(ValueError, match="fov_radius_deg"):
        plot_array_footprint(el=45.0, fov_radius_deg=0.0, show=False)
    with pytest.raises(ValueError, match="modules"):
        plot_array_footprint(el=45.0, modules={}, show=False)


def test_footprint_title_reports_rotation():
    fig = plot_array_footprint(el=50.0, show=False)
    title = fig.axes[0].get_title()
    assert "el = 50.0" in title
    assert "Nasmyth rotation" in title


# ---------------------------------------------------------------------------
# plot_observability_windows
# ---------------------------------------------------------------------------


def _bar_collections(ax):
    """Window-bar collections: the only zorder-3 collections on the axes.

    Selected by an intrinsic property rather than position so a future
    change to the night-shading fills cannot silently reclassify bars.
    """
    return [c for c in ax.collections if c.get_zorder() == 3]


def test_windows_default_lanes_and_labels():
    fig = plot_observability_windows(T0, show=False)
    assert isinstance(fig, Figure)
    (ax,) = fig.axes
    labels = [t.get_text() for t in ax.get_yticklabels()]
    # First requested target on the TOP lane; yticks run bottom-up.
    assert labels == list(reversed(DEFAULT_VISIBILITY_TARGETS))
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    cfg = get_fyst_site().sun_avoidance
    assert any(f"Sun > {cfg.exclusion_radius:.0f}" in t for t in legend_texts)


def test_windows_bars_match_report():
    """Every drawn bar reproduces exactly the report's window intervals."""
    from fyst_trajectories.observability import check_observability

    fig = plot_observability_windows(T0, ["jupiter"], el_min=30.0, show=False)
    (ax,) = fig.axes
    bars = _bar_collections(ax)
    assert len(bars) == 1
    (report,) = check_observability(["jupiter"], T0, horizon_hours=24.0, el_min=30.0)
    assert report.windows  # the probe date must actually have windows
    paths = bars[0].get_paths()
    assert len(paths) == len(report.windows)
    for path, window in zip(paths, report.windows):
        xs = path.vertices[:, 0]
        assert float(xs.min()) == pytest.approx(window.start.plot_date, abs=1e-9)
        assert float(xs.max()) == pytest.approx(window.end.plot_date, abs=1e-9)


def test_windows_empty_lane_is_kept_and_lanes_pair_with_targets():
    """A never-observable target keeps its lane, and bars land on the right lane.

    First requested target on the TOP lane: with the observable target
    requested first its bars sit on the high-y lane, and swapping the
    request order moves them to the low-y lane. Guards the reversed-label
    / len-1-lane pairing against a silent flip.
    """
    site = get_fyst_site()
    coords = Coordinates(site)
    mid = T0 + 12 * u.hour
    sun_ra, sun_dec = coords.get_body_radec("sun", mid)
    at_sun = Target("at_sun", TargetKind.FIXED, ra_deg=float(sun_ra), dec_deg=float(sun_dec))

    def bar_lane_centers(fig):
        (ax,) = fig.axes
        centers = set()
        for coll in _bar_collections(ax):
            for path in coll.get_paths():
                ys = path.vertices[:, 1]
                centers.add(round(float((ys.min() + ys.max()) / 2.0), 6))
        return centers

    fig = plot_observability_windows(T0, ["jupiter", at_sun], el_min=30.0, show=False)
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["at_sun", "jupiter"]
    assert bar_lane_centers(fig) == {1.0}  # jupiter first => top lane (y=1)

    fig2 = plot_observability_windows(T0, [at_sun, "jupiter"], el_min=30.0, show=False)
    assert [t.get_text() for t in fig2.axes[0].get_yticklabels()] == ["jupiter", "at_sun"]
    assert bar_lane_centers(fig2) == {0.0}  # jupiter second => bottom lane (y=0)


def test_windows_injected_model_drives_verdicts_and_legend():
    class _AllUnsafe:
        describe = "stub policy"

        def __call__(self, az_deg, el_deg, time):
            return False

        def batch(self, az_deg, el_deg, times):
            return np.zeros(np.shape(np.asarray(az_deg)), dtype=bool)

    fig = plot_observability_windows(T0, ["jupiter"], sun_model=_AllUnsafe(), show=False)
    (ax,) = fig.axes
    assert len(_bar_collections(ax)) == 0  # everything unsafe => no windows
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("stub policy" in t for t in legend_texts)


def test_windows_validation():
    with pytest.raises(ValueError, match="horizon_hours"):
        plot_observability_windows(T0, horizon_hours=0.0, show=False)
    with pytest.raises(ValueError, match="targets must not be empty"):
        plot_observability_windows(T0, [], show=False)
    for bad_step in (0.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="window_step_minutes"):
            plot_observability_windows(T0, window_step_minutes=bad_step, show=False)
    assert plt.get_fignums() == []  # no half-built figure leaked by any raise


def test_windows_ax_composition(monkeypatch):
    show_calls = []
    layout_calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: show_calls.append(1))
    monkeypatch.setattr(Figure, "tight_layout", lambda self, *a, **k: layout_calls.append(1))
    fig, ax = plt.subplots()
    out = plot_observability_windows(T0, ["jupiter"], ax=ax, title="composed")
    assert out is fig
    assert ax.get_title() == "composed"
    assert show_calls == []
    assert layout_calls == []


def test_windows_tz_axis_label():
    fig = plot_observability_windows(T0, ["jupiter"], tz=timezone(timedelta(hours=-3)), show=False)
    assert "UTC-03:00" in fig.axes[0].get_xlabel()
