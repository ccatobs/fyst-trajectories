"""Regression: every scheduled CE science block must reconstruct.

The scheduler gates CE emission on the planner's own crossing solve
(``_ce_visit_plan``), falls over to the setting half once the rising pass
is spent, and stamps each subscan with the visit anchor
(``metadata["t0_scan"]``) so every slice re-solves from the anchor the
gate guaranteed. Without that gate a constant-elevation patch scheduled
across its transit emits rising-half blocks for hours past the crossing
pass's opening, and ``schedule_to_trajectories`` can rebuild only the
blocks anchored before it. The gate reads a ``PointingError`` from the solve
as "no plannable pass" and lets any other error through, and leaves the
solve's advisories to the rebuild.
"""

import warnings

import numpy as np
import pytest
from astropy.time import Time

from fyst_trajectories import Coordinates, get_fyst_site
from fyst_trajectories.exceptions import PointingError, PointingWarning
from fyst_trajectories.overhead import (
    BlockType,
    ObservingPatch,
    generate_timeline,
    read_timeline,
    schedule_to_trajectories,
    write_timeline,
)
from fyst_trajectories.overhead.scheduler import helpers as _helpers
from fyst_trajectories.overhead.scheduler.helpers import (
    _ce_crossing_corridor,
    _ce_swept_az_envelope,
)
from fyst_trajectories.planning import FieldRegion, plan_constant_el_scan

_AUGUST_PATCH = ObservingPatch(
    name="AugustCES",
    ra_center=330.0,
    dec_center=-50.0,
    width=30.0,
    height=8.0,
    scan_type="constant_el",
    velocity=1.0,
    elevation=50.0,
)


@pytest.fixture(scope="module")
def august_ce_timeline():
    """Build the night this file pins: one CE patch observed across its transit."""
    return generate_timeline(
        patches=[_AUGUST_PATCH],
        site=get_fyst_site(),
        start_time="2026-08-01T23:00:00",
        end_time="2026-08-02T07:00:00",
    )


def _science(timeline):
    return [b for b in timeline.blocks if b.block_type == BlockType.SCIENCE]


def test_every_scheduled_ce_block_reconstructs(august_ce_timeline):
    science = _science(august_ce_timeline)
    assert science, "expected science blocks on this night"
    pairs = schedule_to_trajectories(august_ce_timeline)
    assert len(pairs) == len(science), (
        f"{len(science) - len(pairs)} of {len(science)} scheduled science "
        f"blocks were not reconstructable"
    )


def test_ce_blocks_carry_visit_anchor(august_ce_timeline):
    """Every CE subscan stamps t0_scan, at or before its own start.

    The 5 ms tolerance absorbs the ISO-millisecond rounding of the stamp.
    Each visit's FIRST science block must additionally start within one
    boundary retune of its anchor (the leading scan-coupled retune is the
    only thing between the anchor and the first subscan).
    """
    from fyst_trajectories.overhead import OverheadModel

    retune_dur = OverheadModel().retune_duration
    science = _science(august_ce_timeline)
    assert science, "expected science blocks on the CE corridor night"
    by_anchor: dict[str, list] = {}
    for block in science:
        assert "t0_scan" in block.metadata
        anchor = Time(block.metadata["t0_scan"], scale="utc")
        assert anchor.unix <= block.t_start.unix + 5e-3
        by_anchor.setdefault(block.metadata["t0_scan"], []).append(block)
    for anchor_iso, group in by_anchor.items():
        anchor = Time(anchor_iso, scale="utc")
        first_start = min(b.t_start.unix for b in group)
        assert first_start - anchor.unix <= retune_dur + 5e-3


def test_default_half_falls_over_to_setting(august_ce_timeline):
    """Once the rising pass is spent, emission continues on the setting half."""
    science = _science(august_ce_timeline)
    halves = {bool(b.rising) for b in science}
    assert halves == {True, False}, f"expected both halves on this night, got {halves}"
    # The rising visit must not outlive its pass: every rising block starts
    # before the first setting block (the fall-over point).
    first_setting = min(b.t_start.unix for b in science if not b.rising)
    assert all(b.t_start.unix < first_setting for b in science if b.rising)


def test_ce_visits_end_inside_their_corridor(august_ce_timeline):
    """A visit's boundary retunes share its corridor, they do not extend it.

    The observable duration a CE visit is scored on ends when the pass
    closes. Booking the retunes between subscans on top of that budget
    would overrun the close by one retune per subscan (300 s each), and
    those trailing blocks point at a field whose RA edges have already
    crossed the scan elevation. The visit's blocks share one wall-clock
    budget instead, so the last subscan is clipped.

    The 10 ms tolerance covers the ISO-millisecond stamp of the anchor the
    crossing search is re-run from here.
    """
    coords = Coordinates(get_fyst_site())
    by_anchor: dict[str, list] = {}
    for block in _science(august_ce_timeline):
        by_anchor.setdefault(block.metadata["t0_scan"], []).append(block)
    assert len(by_anchor) >= 2, "expected more than one CE visit on this night"

    for anchor_iso, group in by_anchor.items():
        anchor = Time(anchor_iso, scale="utc")
        window = _ce_crossing_corridor(
            _AUGUST_PATCH, 50.0, bool(group[0].rising), anchor, coords, {}
        )
        assert window is not None, f"the visit anchored at {anchor_iso} has no plannable pass"
        _, t_close = window
        visit_end = max(b.t_stop.unix for b in group)
        overrun = visit_end - t_close.unix
        assert overrun <= 0.01, (
            f"the visit anchored at {anchor_iso} runs {overrun:.3f} s past its corridor close"
        )


def test_reconstructs_after_ecsv_round_trip(august_ce_timeline, tmp_path):
    """t0_scan survives ECSV and the rebuilt count still matches."""
    path = tmp_path / "august_ce.ecsv"
    write_timeline(august_ce_timeline, str(path))
    loaded = read_timeline(str(path))
    science = _science(loaded)
    assert all("t0_scan" in b.metadata for b in science)
    pairs = schedule_to_trajectories(loaded)
    assert len(pairs) == len(science)


def test_a_night_given_in_tt_is_the_utc_night_held_in_utc(august_ce_timeline, tmp_path):
    """The window given as TT ``Time`` objects plans the UTC night, held and recorded in UTC.

    Every block falls at the UTC night's instants, each CE subscan records the
    UTC night's ``t0_scan``, the file keeps the instants, and every subscan
    rebuilds as the UTC night's does.
    """
    night = generate_timeline(
        patches=[_AUGUST_PATCH],
        site=get_fyst_site(),
        start_time=Time("2026-08-01T23:00:00", scale="utc").tt,
        end_time=Time("2026-08-02T07:00:00", scale="utc").tt,
    )
    assert (night.start_time.scale, night.end_time.scale) == ("utc", "utc")
    assert len(night.blocks) == len(august_ce_timeline.blocks)
    for block, utc_block in zip(night.blocks, august_ce_timeline.blocks):
        assert (block.t_start.scale, block.t_stop.scale) == ("utc", "utc")
        assert abs((block.t_start - utc_block.t_start).to_value("s")) < 1e-6
        assert block.metadata.get("t0_scan") == utc_block.metadata.get("t0_scan")

    path = tmp_path / "august_ce_tt.ecsv"
    write_timeline(night, str(path))
    loaded = read_timeline(str(path))
    for block, back in zip(night.blocks, loaded.blocks, strict=True):
        assert abs((back.t_start - block.t_start).to_value("s")) <= 5e-4 + 1e-9
    pairs = schedule_to_trajectories(night)
    utc_pairs = schedule_to_trajectories(august_ce_timeline)
    assert len(schedule_to_trajectories(loaded)) == len(_science(loaded))
    assert len(pairs) == len(utc_pairs) == len(_science(night))
    for (_, rebuilt), (_, utc_rebuilt) in zip(pairs, utc_pairs):
        assert np.array_equal(rebuilt.trajectory.times, utc_rebuilt.trajectory.times)
        assert np.array_equal(rebuilt.trajectory.az, utc_rebuilt.trajectory.az)


def test_a_patch_beside_a_celestial_pole_is_skipped_not_raised():
    """The crossing solve refuses a field within 0.57 deg of a pole; the night goes on.

    ``ObservingPatch`` accepts such a declination, and the solve's flat-sky
    corner projection cannot plan it, so the refusal is a ``PointingError``
    the corridor gate reads as "no plannable pass".
    """
    polar = ObservingPatch(
        name="polar",
        ra_center=100.0,
        dec_center=-89.6,
        width=0.5,
        height=0.2,
        scan_type="constant_el",
        velocity=1.0,
        elevation=30.0,
    )
    timeline = generate_timeline(
        patches=[polar],
        site=get_fyst_site(),
        start_time="2026-04-01T00:00:00",
        end_time="2026-04-01T01:00:00",
    )
    assert timeline.blocks
    assert not _science(timeline)


def test_the_corridor_is_the_envelope_of_the_pass_the_planner_builds_at_the_limit():
    """At an azimuth limit the corridor is the rebuilt pass's envelope, a turn apart.

    At 3 deg/s this field's setting pass sweeps (345.14, 354.78) with a
    5.625 deg turnaround overshoot on each side, past the 360 deg limit, so
    the planner sweeps it one turn down. The corridor the scheduler slews to
    and clips against the Sun is the envelope of that trajectory on the
    field's own branch: the slew re-wraps it to the mount's azimuth, and the
    Sun clip reads it modulo 360 deg.
    """
    site = get_fyst_site()
    coords = Coordinates(site)
    start = Time("2026-11-15T18:00:00", scale="utc")
    patch = ObservingPatch(
        name="NorthSetting",
        ra_center=60.0,
        dec_center=26.25,
        width=2.0,
        height=2.0,
        scan_type="constant_el",
        velocity=3.0,
        elevation=40.0,
    )
    window = _ce_crossing_corridor(patch, 40.0, False, start, coords, {})
    assert window is not None
    lo, hi = _ce_swept_az_envelope(patch, *window, coords)

    block = plan_constant_el_scan(
        field=FieldRegion(ra_center=60.0, dec_center=26.25, width=2.0, height=2.0),
        elevation=40.0,
        velocity=3.0,
        site=site,
        start_time=start,
        rising=False,
    )
    az = block.trajectory.az
    assert lo - 360.0 == pytest.approx(az.min(), abs=0.01)
    assert hi - 360.0 == pytest.approx(az.max(), abs=0.01)
    lim = site.telescope_limits.azimuth
    assert lim.min <= az.min()
    assert az.max() <= lim.max


def test_a_pass_advisory_from_the_gate_is_raised_by_the_rebuild_not_the_night():
    """The crossing solve's advisory on a long pass is left to the rebuild.

    A 110 deg wide field crosses el 50 deg over 8.4 h, which the solve
    flags as unusually long. The gate runs that solve while the timeline is
    generated, and the advisory describes the pass the science block
    records, so it is raised where a science subscan's are: by the rebuild
    of the finished timeline, which runs the same solve.
    """
    wide = ObservingPatch(
        name="Wide",
        ra_center=0.0,
        dec_center=-30.0,
        width=110.0,
        height=10.0,
        scan_type="constant_el",
        velocity=1.0,
        elevation=50.0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PointingWarning)
        timeline = generate_timeline(
            patches=[wide],
            site=get_fyst_site(),
            start_time="2026-06-15T03:00:00",
            end_time="2026-06-15T04:00:00",
        )
    assert [str(w.message) for w in caught if issubclass(w.category, PointingWarning)] == []
    assert len(_science(timeline)) == 1

    with pytest.warns(PointingWarning, match=r"Computed observation duration 8\.4h is unusually"):
        pairs = schedule_to_trajectories(timeline)
    assert len(pairs) == 1


class TestCorridorSolveRefusals:
    """The corridor gate caches an infeasible solve as a miss and lets a bug through."""

    _START = Time("2026-08-01T23:00:00", scale="utc")

    def _corridor(self, monkeypatch, error):
        def _refuse(*args, **kwargs):
            raise error

        monkeypatch.setattr(_helpers, "_compute_ce_duration", _refuse)
        cache: dict = {}
        coords = Coordinates(get_fyst_site())
        window = _ce_crossing_corridor(_AUGUST_PATCH, 50.0, True, self._START, coords, cache)
        return window, cache

    def test_pointing_error_is_cached_as_a_miss(self, monkeypatch):
        window, cache = self._corridor(monkeypatch, PointingError("no crossing"))
        assert window is None
        assert cache == {("AugustCES", 50.0, True): ("miss", self._START)}

    def test_plain_value_error_propagates(self, monkeypatch):
        """A malformed argument is a caller's bug, not an unplannable pass."""
        with pytest.raises(ValueError, match="step_seconds"):
            self._corridor(monkeypatch, ValueError("step_seconds must be positive"))
