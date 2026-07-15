"""Tests for the overhead night-level plotting functions.

Structure-level checks (figure/axes contents, lane order, guards), not
pixel-perfect output. The whole file skips when matplotlib is not installed
(optional extra); the import-isolation test additionally asserts the package
itself never imports matplotlib eagerly.
"""

import subprocess
import sys
import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

# Skip the entire file if matplotlib isn't installed (optional extra).
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from fyst_trajectories import Trajectory, get_fyst_site  # noqa: E402
from fyst_trajectories.overhead import (  # noqa: E402
    CalibrationPolicy,
    ObservingPatch,
    ObservingTimeline,
    OverheadModel,
    TimelineBlock,
    generate_timeline,
    plot_sky_coverage,
    plot_timeline_gantt,
)
from fyst_trajectories.patterns.configs import ScanConfig  # noqa: E402
from fyst_trajectories.planning import ScanBlock  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to keep memory bounded."""
    yield
    plt.close("all")


def _block(t0, minutes, block_type, patch="", scan_type="", scan_index=0):
    return TimelineBlock(
        t_start=t0,
        t_stop=t0 + minutes * u.min,
        block_type=block_type,
        patch_name=patch,
        az_start=180.0,
        az_end=185.0,
        elevation=50.0,
        scan_index=scan_index,
        scan_type=scan_type,
    )


def _make_timeline(blocks=None, start_iso="2026-06-15T02:00:00", hours=2.0):
    """Hand-build a small timeline (no scheduler run)."""
    start = Time(start_iso, scale="utc")
    if blocks is None:
        # Calibration types deliberately appear in NON-canonical order
        # (pointing_cal before retune) so the lane-order test proves the
        # canonical reordering.
        blocks = [
            _block(start, 10, "science", patch="PatchA"),
            _block(start + 10 * u.min, 3, "calibration", scan_type="pointing_cal"),
            _block(start + 13 * u.min, 1, "calibration", scan_type="retune"),
            _block(start + 14 * u.min, 10, "science", patch="PatchA", scan_index=1),
            _block(start + 24 * u.min, 2, "slew"),
            _block(start + 26 * u.min, 10, "science", patch="PatchB", scan_index=2),
            _block(start + 36 * u.min, 5, "idle"),
        ]
    return ObservingTimeline(
        blocks=blocks,
        site=get_fyst_site(),
        start_time=start,
        end_time=start + hours * u.hour,
        overhead_model=OverheadModel(),
        calibration_policy=CalibrationPolicy(),
        metadata={},
    )


def _make_sky_pairs(n=2):
    """Synthetic (TimelineBlock, ScanBlock) pairs; no planner involved."""
    start = Time("2026-06-15T03:00:00", scale="utc")
    pairs = []
    for i in range(n):
        times = np.arange(0.0, 60.0, 0.5)
        traj = Trajectory(
            times=times,
            az=np.linspace(170.0 + 5.0 * i, 190.0 + 5.0 * i, times.size),
            el=np.full(times.size, 50.0),
            az_vel=np.zeros(times.size),
            el_vel=np.zeros(times.size),
            start_time=start + i * 0.5 * u.hour,
        )
        scan_block = ScanBlock(trajectory=traj, config=ScanConfig(timestep=0.5), duration=60.0)
        block = _block(start, 1, "science", patch=("PatchA", "PatchB")[i % 2], scan_index=i)
        pairs.append((block, scan_block))
    return pairs


# ---------------------------------------------------------------------------
# plot_timeline_gantt
# ---------------------------------------------------------------------------


def test_gantt_returns_figure():
    fig = plot_timeline_gantt(_make_timeline(), show=False)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


def test_gantt_into_caller_axes_returns_that_figure():
    fig, ax = plt.subplots()
    out = plot_timeline_gantt(_make_timeline(), ax=ax, show=False)
    assert out is ax.get_figure()
    assert len(plt.get_fignums()) == 1  # no second figure created


def test_gantt_show_suppressed_with_ax(monkeypatch):
    calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))
    _, ax = plt.subplots()
    plot_timeline_gantt(_make_timeline(), ax=ax)  # default show=True
    assert calls == []


def test_gantt_auto_title_neutral():
    fig = plot_timeline_gantt(_make_timeline(), show=False)
    title = fig.axes[0].get_title()
    assert "2026-06-15" in title
    assert "efficiency" in title
    assert "commissioning" not in title.lower()

    fig2 = plot_timeline_gantt(_make_timeline(), title="Custom title", show=False)
    assert fig2.axes[0].get_title() == "Custom title"


def test_gantt_empty_timeline_raises():
    with pytest.raises(ValueError, match="no blocks"):
        plot_timeline_gantt(_make_timeline(blocks=[]), show=False)


def test_gantt_lane_order_contract():
    """Lane order top-to-bottom: science (first appearance), canonical cals, slew, idle."""
    fig = plot_timeline_gantt(_make_timeline(), show=False)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    positions = list(ax.get_yticks())
    top_to_bottom = [label for _, label in sorted(zip(positions, labels), reverse=True)]
    # retune precedes pointing_cal despite appearing later in the blocks.
    assert top_to_bottom == ["PatchA", "PatchB", "retune", "pointing_cal", "slew", "idle"]


def test_gantt_bars_sit_in_their_labeled_lane():
    """Each lane's bars occupy y+0.1..y+0.9 around that lane's y-tick.

    Locks the bar half of the documented annotation contract (the label
    half is locked by test_gantt_lane_order_contract): a mutation that
    desyncs bar positions from lane labels must fail here.
    """
    fig = plot_timeline_gantt(_make_timeline(), show=False)
    ax = fig.axes[0]
    ticks = list(ax.get_yticks())  # top-to-bottom lane order, as set
    # Every lane in the default timeline is non-empty, so there is one
    # broken_barh collection per lane, drawn in the same order as the ticks.
    assert len(ax.collections) == len(ticks)
    for tick, collection in zip(ticks, ax.collections):
        ys = np.concatenate([path.vertices[:, 1] for path in collection.get_paths()])
        assert ys.min() == pytest.approx(tick - 0.4, abs=1e-6)  # y + 0.1
        assert ys.max() == pytest.approx(tick + 0.4, abs=1e-6)  # y + 0.9


def test_gantt_midnight_crossing_ticks():
    """A night straddling UTC midnight labels hours mod 24, never 24:00+."""
    start = Time("2026-06-15T23:00:00", scale="utc")
    blocks = [
        _block(start, 30, "science", patch="PatchA"),
        _block(start + 1 * u.hour, 30, "science", patch="PatchA", scan_index=1),
    ]
    timeline = _make_timeline(blocks=blocks, start_iso="2026-06-15T23:00:00", hours=2.0)
    fig = plot_timeline_gantt(timeline, show=False)
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert "23:00" in labels
    assert "00:00" in labels
    assert "01:00" in labels
    for label in labels:
        assert int(label.split(":")[0]) < 24


# ---------------------------------------------------------------------------
# plot_sky_coverage
# ---------------------------------------------------------------------------


def test_sky_returns_figure_with_injected_pairs():
    pairs = _make_sky_pairs()
    fig = plot_sky_coverage(_make_timeline(), pairs=pairs, show=False)
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].lines) == len(pairs)  # one track per pair


def test_sky_zero_pairs_raises():
    with pytest.raises(ValueError, match="no science"):
        plot_sky_coverage(_make_timeline(), pairs=[], show=False)


def test_sky_stride_zero_and_negative_raise():
    pairs = _make_sky_pairs()
    with pytest.raises(ValueError, match="stride"):
        plot_sky_coverage(_make_timeline(), pairs=pairs, stride=0, show=False)
    with pytest.raises(ValueError, match="stride"):
        plot_sky_coverage(_make_timeline(), pairs=pairs, stride=-1, show=False)


def test_sky_into_caller_axes(monkeypatch):
    calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))
    fig, ax = plt.subplots()
    out = plot_sky_coverage(_make_timeline(), pairs=_make_sky_pairs(), ax=ax)
    assert out is ax.get_figure()
    assert calls == []
    assert len(plt.get_fignums()) == 1


def test_sky_ra_wrapped_range():
    fig = plot_sky_coverage(_make_timeline(), pairs=_make_sky_pairs(), show=False)
    ax = fig.axes[0]
    ra = np.concatenate([line.get_xdata() for line in ax.lines])
    assert ra.size > 0
    assert np.all(ra >= -180.0)
    assert np.all(ra < 180.0)
    assert ax.xaxis_inverted()


def test_sky_legend_counts_scans():
    fig = plot_sky_coverage(_make_timeline(), pairs=_make_sky_pairs(4), show=False)
    legend_texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "PatchA  (2 scans)" in legend_texts
    assert "PatchB  (2 scans)" in legend_texts


def test_sky_missing_start_time_raises_without_figure_leak():
    """The start_time guard fires before any figure is created (no leak)."""
    times = np.arange(0.0, 10.0, 0.5)
    traj = Trajectory(
        times=times,
        az=np.linspace(170.0, 180.0, times.size),
        el=np.full(times.size, 50.0),
        az_vel=np.zeros(times.size),
        el_vel=np.zeros(times.size),
    )
    block = _block(Time("2026-06-15T03:00:00", scale="utc"), 1, "science", patch="PatchA")
    pair = (block, ScanBlock(trajectory=traj, config=ScanConfig(timestep=0.5), duration=10.0))
    with pytest.raises(ValueError, match="start_time"):
        plot_sky_coverage(_make_timeline(), pairs=[pair], show=False)
    assert plt.get_fignums() == []


# ---------------------------------------------------------------------------
# package hygiene + end-to-end
# ---------------------------------------------------------------------------


def test_import_isolation():
    """Importing the package (incl. overhead) must not import matplotlib."""
    code = (
        "import sys\n"
        "import fyst_trajectories\n"
        "import fyst_trajectories.overhead\n"
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


def test_sky_auto_pairs_end_to_end():
    """The pairs=None branch: run a small night and rebuild internally."""
    patch = ObservingPatch(
        name="Deep56",
        ra_center=24.0,
        dec_center=-32.0,
        width=40.0,
        height=10.0,
        scan_type="constant_el",
        velocity=1.0,
        elevation=50.0,
    )
    timeline = generate_timeline(
        patches=[patch],
        site=get_fyst_site(),
        start_time="2026-06-15T00:00:00",
        end_time="2026-06-15T12:00:00",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = plot_sky_coverage(timeline, show=False)
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].lines) > 0
