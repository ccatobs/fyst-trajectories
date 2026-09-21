"""Tests for ``plot_source_track``.

The whole file skips when matplotlib is not installed; the import-isolation
test in test_overhead_plotting.py covers this module too.
"""

import warnings

import numpy as np
import pytest
from astropy.time import Time

# Skip the entire file if matplotlib isn't installed (optional extra).
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from fyst_trajectories import (  # noqa: E402
    FieldRegion,
    get_fyst_site,
    plan_pong_scan,
    plan_source_ces,
    source_ces_focal_plane_track,
)
from fyst_trajectories.primecam import PRIMECAM_MODULES, get_primecam_offset  # noqa: E402
from fyst_trajectories.visualization import plot_source_track  # noqa: E402


@pytest.fixture(scope="module")
def jupiter_pass():
    """One single-module Jupiter-rising pass on the 2026-03-15 night."""
    site = get_fyst_site()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        block = plan_source_ces(
            body="jupiter",
            footprint="c",
            el_bore=35.0,
            night=Time("2026-03-15T00:00:00", scale="utc"),
            mode="rising",
            site=site,
        )
    return site, block


class TestPlotSourceTrack:
    """Layout from the registry, track from the planner, composition rules."""

    def test_returns_figure_with_the_registry_layout(self, jupiter_pass):
        site, block = jupiter_pass
        fig = plot_source_track(block, site=site, show=False)
        try:
            assert isinstance(fig, Figure)
            ax = fig.axes[0]
            circles = [p for p in ax.patches if isinstance(p, Circle)]
            unique = {id(o) for o in PRIMECAM_MODULES.values()}
            assert len(circles) == len(unique)
            centres = {(round(c.center[0], 6), round(c.center[1], 6)) for c in circles}
            i1 = get_primecam_offset("i1")
            assert (round(i1.dx_deg, 6), round(i1.dy_deg, 6)) in centres
            labels = {t.get_text() for t in ax.texts}
            assert {"Center", "I1", "I6"} <= labels
        finally:
            plt.close(fig)

    def test_track_is_the_planners_track(self, jupiter_pass):
        site, block = jupiter_pass
        xi, eta = source_ces_focal_plane_track(block, site=site)
        fig = plot_source_track(block, site=site, show=False)
        try:
            ax = fig.axes[0]
            track = next(line for line in ax.lines if line.get_label() == "source track")
            np.testing.assert_allclose(track.get_xdata(), xi)
            np.testing.assert_allclose(track.get_ydata(), eta)
            start = next(line for line in ax.lines if line.get_label() == "pass start")
            assert (start.get_xdata()[0], start.get_ydata()[0]) == (xi[0], eta[0])
            # A rising pass climbs through the array: the end sits above the start.
            assert eta[-1] > eta[0]
        finally:
            plt.close(fig)

    def test_title_states_the_projection_assumptions(self, jupiter_pass):
        site, block = jupiter_pass
        fig = plot_source_track(block, site=site, show=False)
        try:
            title = fig.axes[0].get_title()
            assert "Jupiter" in title and "rising" in title and "35.0" in title
            assert f"Nasmyth sign {site.nasmyth_sign:+d}" in title
        finally:
            plt.close(fig)

    def test_ax_composition_and_no_labels(self, jupiter_pass):
        site, block = jupiter_pass
        fig, ax = plt.subplots()
        try:
            out = plot_source_track(block, site=site, ax=ax, labels=False, title="custom")
            assert out is fig
            assert ax.get_title() == "custom"
            assert not ax.texts
        finally:
            plt.close(fig)

    def test_rejects_a_non_source_ces_block(self, jupiter_pass):
        site, _ = jupiter_pass
        field = FieldRegion(ra_center=180.0, dec_center=-30.0, width=2.0, height=2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pong = plan_pong_scan(
                field=field,
                velocity=0.5,
                spacing=0.1,
                num_terms=4,
                timestep=0.5,
                start_time=Time("2026-03-15T01:00:00", scale="utc"),
                site=site,
            )
        with pytest.raises(ValueError, match="plan_source_ces"):
            source_ces_focal_plane_track(pong, site=site)
        with pytest.raises(ValueError, match="plan_source_ces"):
            plot_source_track(pong, site=site, show=False)

    def test_bad_radius_and_empty_modules(self, jupiter_pass):
        site, block = jupiter_pass
        with pytest.raises(ValueError, match="fov_radius_deg"):
            plot_source_track(block, site=site, fov_radius_deg=0.0, show=False)
        with pytest.raises(ValueError, match="modules must not be empty"):
            plot_source_track(block, site=site, modules={}, show=False)
