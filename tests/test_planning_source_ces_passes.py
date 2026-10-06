"""Tests for :func:`fyst_trajectories.plan_source_ces_passes`.

The multi-pass sequence steps the footprint in focal-plane eta so that the
passes together tile the footprint's extent; each pass is one source-CES
block planned with real astronomy (FYST site, real planet ephemerides).
"""

from __future__ import annotations

import numpy as np
import pytest
from _source_ces_helpers import _JUPITER_NIGHT
from astropy import units as u
from astropy.time import Time, TimeDelta

from fyst_trajectories import (
    FYST_AZ_MAX_VELOCITY,
    Coordinates,
    ElevationBoundsError,
    InstrumentOffset,
    PointingWarning,
    ScanBlock,
    SourceCESComputedParams,
    TargetNotObservableError,
    compute_focal_plane_rotation,
    get_fyst_site,
    plan_source_ces_passes,
)

# ---------------------------------------------------------------------------
# plan_source_ces_passes (multi-pass full-coverage sequence)
# ---------------------------------------------------------------------------


def _source_focalplane_eta_mean(block, site, coords, body="jupiter"):
    """Mean focal-plane eta of the source over a pass's source window.

    Recovers the source's position in the pass's focal-plane frame by
    un-rotating the (source - boresight) sky offset by the mechanical
    focal-plane rotation (the same horizon-frame convention the planner
    uses). For a footprint offset by ``eta`` this mean tracks ``eta``,
    which is what proves the offset moves the coverage 1:1.
    """
    traj = block.trajectory
    cp = block.computed_params
    el_bore = float(cp["el_bore"])
    t0 = (Time(cp["t0_iso"]) - traj.start_time).to_value(u.s)
    t1 = (Time(cp["t1_iso"]) - traj.start_time).to_value(u.s)
    ts = np.linspace(t0, t1, 60)
    times = traj.start_time + TimeDelta(ts * u.s)
    src_az, src_el = coords.get_body_altaz(body, times)
    src_az = np.asarray(src_az, dtype=float)
    src_el = np.asarray(src_el, dtype=float)
    bore_az = np.interp(ts, traj.times, traj.az)
    bore_el = np.interp(ts, traj.times, traj.el)
    # Wrap the azimuth difference into [-180, 180] so a coordinate that
    # straddles the 0/360 boundary does not blow up the cross-el term.
    d_az = ((src_az - bore_az + 180.0) % 360.0) - 180.0
    dxi_sky = d_az * np.cos(np.deg2rad(el_bore))
    deta_sky = src_el - bore_el
    rot = np.deg2rad(
        compute_focal_plane_rotation(el=el_bore, site=site, offset=InstrumentOffset(dx=0.0, dy=0.0))
    )
    eta = -dxi_sky * np.sin(rot) + deta_sky * np.cos(rot)
    return float(np.mean(eta))


@pytest.fixture(scope="module")
def jupiter_three_passes():
    """Plan the 3-pass Jupiter-rising sequence once for the module.

    The tests that read it only inspect the blocks, which are frozen. It
    builds its own site because the ``site`` fixture is function-scoped.
    """
    return plan_source_ces_passes(
        body="jupiter",
        footprint="c",
        el_bore=35.0,
        n_passes=3,
        night=_JUPITER_NIGHT,
        mode="rising",
        site=get_fyst_site(),
    )


def test_passes_time_ordered_and_non_overlapping(jupiter_three_passes):
    """A 3-pass Jupiter-rising sequence is time-ordered and non-overlapping."""
    blocks = jupiter_three_passes
    assert len(blocks) == 3
    assert all(isinstance(b, ScanBlock) for b in blocks)

    # Full-block occupancy windows [start, start + duration].
    occ = [
        (b.trajectory.start_time.unix, b.trajectory.start_time.unix + b.duration) for b in blocks
    ]
    # Strictly time-ordered by start.
    assert all(occ[k][0] < occ[k + 1][0] for k in range(len(occ) - 1))
    # Non-overlapping: each pass starts at or after the previous one ends.
    assert all(occ[k + 1][0] >= occ[k][1] - 1e-6 for k in range(len(occ) - 1)), (
        f"passes overlap in time: {occ}"
    )
    # pass_index metadata matches the returned (time) order.
    assert [b.trajectory.metadata.pattern_params["pass_index"] for b in blocks] == [0, 1, 2]


def test_passes_tile_footprint_extent(site, jupiter_three_passes):
    """The passes' eta offsets tile the footprint extent, and coverage tracks them."""
    coords = Coordinates(site)
    n_passes = 3
    blocks = jupiter_three_passes

    # Footprint eta extent, computed exactly as the wrapper does (the
    # 50-vertex circular cover inscribes slightly inside 2 * radius).
    from fyst_trajectories.planning.footprints import resolve_footprint

    base_fp = resolve_footprint("c")
    extent = float(base_fp.cover_eta_deg.max() - base_fp.cover_eta_deg.min())
    step = extent / n_passes  # the documented default step

    offsets = sorted(b.trajectory.metadata.pattern_params["pass_eta_offset_deg"] for b in blocks)
    # Distinct and symmetric about 0.
    assert len(set(offsets)) == n_passes
    # The n bands of width ``step`` centred on the offsets tile
    # [-extent/2, +extent/2] edge to edge.
    assert offsets[0] - step / 2.0 == pytest.approx(-extent / 2.0, abs=1e-6)
    assert offsets[-1] + step / 2.0 == pytest.approx(extent / 2.0, abs=1e-6)

    # The offset is not a cosmetic label: the source's mean focal-plane
    # eta actually tracks each pass's offset (this is what a bare el_bore
    # step would fail to do). Sort by offset and check monotonic tracking.
    by_offset = sorted(
        blocks, key=lambda b: b.trajectory.metadata.pattern_params["pass_eta_offset_deg"]
    )
    measured = [_source_focalplane_eta_mean(b, site, coords) for b in by_offset]
    assert measured == sorted(measured), f"coverage centres not monotonic: {measured}"
    for b, m in zip(by_offset, measured):
        expected = b.trajectory.metadata.pattern_params["pass_eta_offset_deg"]
        assert m == pytest.approx(expected, abs=0.1), (
            f"coverage centre {m:.3f} does not track eta offset {expected:.3f}"
        )
    # The measured coverage centres span ~the full offset range (tiling).
    assert (measured[-1] - measured[0]) == pytest.approx(offsets[-1] - offsets[0], abs=0.1)


def test_each_pass_is_valid_source_ces(jupiter_three_passes):
    """Every pass validates exactly like a standalone plan_source_ces block."""
    blocks = jupiter_three_passes
    for b in blocks:
        # Same computed_params schema as a single source-CES block.
        assert set(b.computed_params) == set(SourceCESComputedParams.__required_keys__)
        assert b.computed_params["mode"] == "rising"
        assert b.computed_params["n_scans"] >= 1
        assert b.duration > 0
        # Constant elevation at this pass's stepped el_bore.
        el_bore = b.computed_params["el_bore"]
        assert np.allclose(b.trajectory.el, el_bore, atol=1e-6)
        # Azimuth velocity within the hardware limit (bounds were validated
        # inside plan_source_ces).
        assert np.all(np.abs(b.trajectory.az_vel) <= FYST_AZ_MAX_VELOCITY)
    # The stepped boresight elevations are symmetric about el_bore.
    el_bores = sorted(b.computed_params["el_bore"] for b in blocks)
    assert el_bores[1] == pytest.approx(35.0)
    assert (el_bores[2] - el_bores[1]) == pytest.approx(el_bores[1] - el_bores[0])


def test_every_pass_records_the_source(jupiter_three_passes):
    """Each pass block carries the source identity in ``pattern_params["body"]``."""
    assert [b.trajectory.metadata.pattern_params["body"] for b in jupiter_three_passes] == [
        "jupiter"
    ] * 3


def test_explicit_eta_offsets_honored(site):
    """An explicit eta_offsets list produces one pass per row, coverage tracking."""
    coords = Coordinates(site)
    requested = [0.3, -0.3, 0.0]  # deliberately unsorted
    blocks = plan_source_ces_passes(
        body="jupiter",
        footprint="c",
        el_bore=35.0,
        eta_offsets=requested,
        night=_JUPITER_NIGHT,
        mode="rising",
        site=site,
    )
    assert len(blocks) == 3
    offsets = sorted(b.trajectory.metadata.pattern_params["pass_eta_offset_deg"] for b in blocks)
    assert offsets == pytest.approx(sorted(requested))
    # Coverage tracks the explicit rows.
    for b in blocks:
        m = _source_focalplane_eta_mean(b, site, coords)
        expected = b.trajectory.metadata.pattern_params["pass_eta_offset_deg"]
        assert m == pytest.approx(expected, abs=0.1)


def test_passes_setting_source_time_ordered(site):
    """A setting source (coverage order reversed vs time) still returns time-ordered."""
    blocks = plan_source_ces_passes(
        ra=180.0,
        dec=-30.0,
        footprint="c",
        el_bore=40.0,
        n_passes=3,
        night=_JUPITER_NIGHT,
        mode="setting",
        site=site,
    )
    occ = [
        (b.trajectory.start_time.unix, b.trajectory.start_time.unix + b.duration) for b in blocks
    ]
    assert all(occ[k][0] < occ[k + 1][0] for k in range(len(occ) - 1))
    assert all(occ[k + 1][0] >= occ[k][1] - 1e-6 for k in range(len(occ) - 1)), (
        f"setting-source passes overlap in time: {occ}"
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param(dict(n_passes=0), "n_passes must be at least 1", id="n_passes-zero"),
        pytest.param(dict(n_passes=-1), "n_passes must be at least 1", id="n_passes-negative"),
        pytest.param(
            dict(),
            "must specify either 'n_passes' or 'eta_offsets'",
            id="neither-n_passes-nor-eta_offsets",
        ),
        pytest.param(
            dict(n_passes=3, eta_offsets=[0.0, 0.5]),
            "'n_passes' or 'eta_offsets', not both",
            id="both-n_passes-and-eta_offsets",
        ),
        pytest.param(dict(eta_offsets=[]), "eta_offsets cannot be empty", id="empty-eta_offsets"),
        pytest.param(
            dict(eta_offsets=[0.0, 0.5], step=0.2),
            "'step' is only valid together with 'n_passes'",
            id="step-without-n_passes",
        ),
        pytest.param(dict(n_passes=3, step=-0.1), "step must be positive", id="negative-step"),
        pytest.param(dict(n_passes=3, el_step=0.0), "el_step must be positive", id="zero-el_step"),
    ],
)
def test_passes_invalid_controls_raise_value_error(site, kwargs, match):
    """Degenerate pass-control combinations raise ValueError before astronomy runs."""
    full = dict(
        body="jupiter",
        footprint="c",
        el_bore=35.0,
        night=_JUPITER_NIGHT,
        mode="rising",
        site=site,
    )
    full.update(kwargs)
    with pytest.raises(ValueError, match=match):
        plan_source_ces_passes(**full)


def test_passes_offset_beyond_reach_raises(site):
    """An eta offset that steps a pass past the source's reachable arc raises cleanly."""
    # A +30 deg eta offset drives one pass's footprint (and its stepped
    # el_bore) far above Jupiter's accessible arc, so the underlying
    # plan_source_ces gate rejects it.
    with pytest.raises((TargetNotObservableError, ElevationBoundsError)):
        plan_source_ces_passes(
            body="jupiter",
            footprint="c",
            el_bore=35.0,
            eta_offsets=[0.0, 30.0],
            night=_JUPITER_NIGHT,
            mode="rising",
            site=site,
        )


def test_passes_duplicate_eta_offsets_raise(site):
    """Duplicate eta offsets are rejected: identical passes are never intended."""
    with pytest.raises(ValueError, match="unique"):
        plan_source_ces_passes(
            body="jupiter",
            footprint="c",
            el_bore=35.0,
            eta_offsets=[0.0, 0.0],
            night=_JUPITER_NIGHT,
            mode="rising",
            site=site,
        )


def test_passes_small_el_step_overlap_warns(site):
    """Shrinking el_step below the footprint extent overlaps pass windows and warns."""
    with pytest.warns(PointingWarning, match="overlap"):
        blocks = plan_source_ces_passes(
            body="jupiter",
            footprint="c",
            el_bore=35.0,
            n_passes=2,
            el_step=0.05,
            night=_JUPITER_NIGHT,
            mode="rising",
            site=site,
        )
    # Still time-ordered even when the occupancy windows overlap.
    starts = [Time(b.computed_params["t0_iso"]).unix for b in blocks]
    assert starts == sorted(starts)
