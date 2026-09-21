"""Regression anchors for planned calibration nights.

The fast anchor is a one-hour two-body night on 2026-09-11 (Saturn near
transit with Uranus rising); the full four-body night from dusk to dawn
runs under ``--run-slow``. Counts and causes are anchored exactly, geometry
tolerantly. Every value was re-run, not derived.
"""

import warnings

import pytest
from astropy.time import Time

from fyst_trajectories import choose_encoder_solution, get_fyst_site
from fyst_trajectories.overhead import (
    NightContext,
    NightState,
    list_candidates,
    plan_calibration_night,
    read_calibration_night_metadata,
    schedule_to_trajectories,
    summarize_calibration_night,
)
from fyst_trajectories.sun_models import make_sun_safe


def _plan(targets, start, end):
    site = get_fyst_site()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return site, plan_calibration_night(targets, site, start, end)


def _passes(timeline):
    return [b for b in timeline.blocks if b.scan_type == "planet_cal"]


def _check_rebuild_and_regate(site, timeline):
    """Every pass rebuilds from its relative dict and its wrap clears the Sun gate."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pairs = schedule_to_trajectories(timeline, science_only=False)
    passes = _passes(timeline)
    assert len(pairs) == len(passes)
    sun_safe = make_sun_safe("scalar", site=site)
    for block, rebuilt in pairs:
        # Measured on this fixture: 0.007 to 0.014 s, the reconstruction
        # widening the window and re-landing on the same crossing. A 2.0 s
        # tolerance would be wide enough to absorb a real anchoring drift.
        assert abs((rebuilt.trajectory.start_time - block.t_start).to_value("s")) < 0.1
        assert rebuilt.computed_params["az_throw"] == pytest.approx(
            block.metadata["applied"]["az_throw"], abs=1e-6
        )
        wrap = block.metadata["transition"]["wrap"]
        t0 = Time(block.metadata["t0_scan"], scale="utc")
        az, _ = choose_encoder_solution(
            wrap,
            block.elevation,
            wrap,
            block.elevation,
            t0,
            site,
            sun_safe=sun_safe,
            goal_az_span=(block.az_start, block.az_end),
        )
        assert az == pytest.approx(wrap, abs=1e-6)


class TestShortNight:
    """Three Saturn passes in an hour: the fast anchor."""

    def test_block_counts(self, short_night):
        _, timeline = short_night
        kinds = [b.scan_type for b in timeline.blocks]
        assert len(kinds) == 15
        assert kinds.count("planet_cal") == 3
        assert kinds.count("slew") == 3
        assert kinds.count("skydip") == 1
        assert kinds.count("retune") == 4  # one find_detectors plus one retune per pass
        assert kinds.count("idle") == 4
        assert timeline.validate() == []

    def test_visit_order_and_causes(self, short_night):
        _, timeline = short_night
        assert [b.metadata["target"] for b in _passes(timeline)] == ["saturn"] * 3
        meta = read_calibration_night_metadata(timeline)
        assert meta["drops"] == []
        assert {d["reason"] for d in meta["deferrals"]} == {"window_closed"}
        assert {d["body"] for d in meta["deferrals"]} == {"saturn", "uranus"}

    def test_one_retune_per_pass_at_the_target_pose(self, short_night):
        _, timeline = short_night
        blocks = timeline.blocks
        for i, block in enumerate(blocks):
            if block.scan_type != "planet_cal":
                continue
            j = i - 1
            while blocks[j].scan_type == "idle":
                j -= 1
            retune = blocks[j]
            assert retune.scan_type == "retune" and "operation" not in retune.metadata
            assert retune.az_start == pytest.approx(block.metadata["transition"]["wrap"])
            assert retune.elevation == pytest.approx(block.elevation)
            assert retune.duration == pytest.approx(300.0)

    def test_geometry_tolerant(self, short_night):
        _, timeline = short_night
        first = _passes(timeline)[0]
        meta = first.metadata
        assert first.elevation == pytest.approx(61.5, abs=0.5)
        assert meta["solved"]["crossing_seconds"] == pytest.approx(736.0, abs=10.0)
        assert meta["applied"]["az_throw"] == pytest.approx(4.5, abs=0.1)
        assert meta["applied"]["az_speed"] == 1.5 and meta["applied"]["az_accel"] == 1.5
        assert 0.55 < meta["science_fraction"] < 0.65
        assert meta["n_legs"] == pytest.approx(148, abs=3)
        summary = summarize_calibration_night(timeline)
        assert summary.bodies[0].minutes_on_source == pytest.approx(30.8, abs=0.5)
        assert any("2.2" in w and "exceeds limit" in w for w in summary.warnings)

    def test_rebuilds_and_regates(self, short_night):
        site, timeline = short_night
        _check_rebuild_and_regate(site, timeline)


@pytest.mark.slow
class TestFullNight:
    """The four-body night from dusk to dawn, 2026-09-10/11."""

    @pytest.fixture(scope="class")
    def full_night(self):
        return _plan(
            ["jupiter", "saturn", "neptune", "uranus"],
            "2026-09-10T22:00:00",
            "2026-09-11T10:30:00",
        )

    def test_shape(self, full_night):
        _, timeline = full_night
        assert timeline.validate() == []
        summary = summarize_calibration_night(timeline)
        by_body = {b.body: b for b in summary.bodies}
        assert by_body["jupiter"].passes == 0  # only 11 deg up when dawn ends the window
        assert by_body["saturn"].passes >= 20
        assert by_body["neptune"].passes >= 3
        assert by_body["uranus"].passes >= 1
        assert summary.usable_interval[0].startswith("2026-09-10 22:2")

    def test_contention_and_extrapolation(self, full_night):
        site, timeline = full_night
        # Three bodies are available at 08:30 UTC; the priority rule resolves the
        # contention to Saturn, the first in the caller's order, every time.
        ctx = NightContext.build(
            ["jupiter", "saturn", "neptune", "uranus"],
            site,
            "2026-09-10T22:00:00",
            "2026-09-11T10:30:00",
        )
        state = NightState.initial(Time("2026-09-11T08:30:00", scale="utc"), (300.0, 45.0))
        available = {c.body for c in list_candidates(state, ctx) if c.available}
        assert available == {"saturn", "neptune", "uranus"}
        passes = _passes(timeline)
        window = {
            p.metadata["target"]
            for p in passes
            if "2026-09-11 08:00" <= p.t_start.iso[:16] <= "2026-09-11 09:00"
        }
        assert window == {"saturn"}
        uranus = [p for p in passes if p.metadata["target"] == "uranus"]
        assert any(p.elevation > 40.0 for p in uranus)
        assert all(p.metadata["requested"]["az_throw"] > 2.61 for p in uranus if p.elevation > 40.0)

    def test_every_pass_rebuilds_and_regates(self, full_night):
        site, timeline = full_night
        _check_rebuild_and_regate(site, timeline)
