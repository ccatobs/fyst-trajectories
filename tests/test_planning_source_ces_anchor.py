"""Tests for the approximate ``start_time`` anchor of the source-CES planners.

An anchored call plans a pass that starts about now: at or shortly after
the anchor, never before it, with ``el_bore`` and ``mode`` derived when
they are not given.
"""

from __future__ import annotations

import numpy as np
import pytest
from _source_ces_helpers import _JUPITER_NIGHT, _JUPITER_RISING_ANCHOR
from astropy import units as u
from astropy.time import Time, TimeDelta

import fyst_trajectories.planning.source_ces._anchor as _anchor_module
import fyst_trajectories.planning.source_ces._kernel as _kernel_module
from fyst_trajectories import (
    Coordinates,
    ElevationBoundsError,
    PointingError,
    ScanBlock,
    SourceCESComputedParams,
    TargetNotObservableError,
    compute_source_ces_params,
    get_fyst_site,
    plan_source_ces,
    plan_source_ces_passes,
)
from fyst_trajectories.planning import inflate_footprint, resolve_footprint

# ---------------------------------------------------------------------------
# Approximate start_time anchor ("plan a pass starting about now")
# ---------------------------------------------------------------------------

# Tolerance for the "anchored pass starts near the anchor" assertions. The
# derivation leads by _ANCHOR_START_LEAD_DEG of elevation, which at the minimum
# permitted drift rate crosses in _ANCHOR_START_LEAD_DEG / _MIN_ANCHOR_EL_DRIFT_DEG_S
# = 60 s; doubled to cover crossing-solver slack.
_ANCHOR_START_TOL_SEC = 120.0


def _jupiter_el_track(site):
    """Sample Jupiter (az, el) across the test night at 60 s cadence."""
    coords = Coordinates(site)
    dt = np.arange(0.0, 24 * 3600.0, 60.0)
    times = _JUPITER_NIGHT + TimeDelta(dt * u.s)
    _, el = coords.get_body_altaz("jupiter", times)
    return times, np.asarray(el, dtype=float)


def _jupiter_transit_anchor(site):
    """Time of Jupiter's culmination (elevation maximum) on the test night."""
    times, el = _jupiter_el_track(site)
    return times[int(np.argmax(el))]


def _jupiter_setting_anchor(site, target_el=40.0):
    """First post-transit time Jupiter descends through ``target_el``."""
    times, el = _jupiter_el_track(site)
    i_max = int(np.argmax(el))
    after = np.arange(len(el)) > i_max
    idx = np.where(after & (el <= target_el))[0]
    assert len(idx), "no setting Jupiter sample found"
    return times[idx[0]]


def test_anchored_plan_source_ces_rising(site):
    """Anchored plan_source_ces derives a rising pass starting near the anchor."""
    coords = Coordinates(site)
    anchor = _JUPITER_RISING_ANCHOR
    _, el_at_anchor = coords.get_body_altaz("jupiter", anchor)
    el_at_anchor = float(el_at_anchor)

    block = plan_source_ces(body="jupiter", footprint="c", start_time=anchor, site=site)
    cp = block.computed_params

    assert cp["mode"] == "rising"
    t0 = Time(cp["t0_iso"])
    delta = (t0 - anchor).to_value(u.s)
    # Anchor, not literal start: t0 lands at or just after the anchor.
    assert delta >= -1e-6, f"t0 must be >= anchor, got {delta:+.3f}s"
    assert delta <= _ANCHOR_START_TOL_SEC, (
        f"t0 should land within 120 s of the anchor, got {delta:+.1f}s"
    )

    el_limits = site.telescope_limits.elevation
    assert el_limits.min <= cp["el_bore"] <= el_limits.max
    # For a centred module the boresight sits a little above the source
    # elevation at the anchor (roughly the cover half-height plus the lead).
    assert el_at_anchor < cp["el_bore"] < el_at_anchor + 1.5


def test_anchored_plan_source_ces_setting(site):
    """A setting anchor resolves mode='setting' and starts at or after the anchor."""
    anchor = _jupiter_setting_anchor(site)
    block = plan_source_ces(body="jupiter", footprint="c", start_time=anchor, site=site)
    cp = block.computed_params

    assert cp["mode"] == "setting"
    delta = (Time(cp["t0_iso"]) - anchor).to_value(u.s)
    assert delta >= -1e-6, f"t0 must be >= anchor, got {delta:+.3f}s"
    assert delta <= _ANCHOR_START_TOL_SEC, (
        f"t0 should land within 120 s of the anchor, got {delta:+.1f}s"
    )


def test_anchored_explicit_el_bore_is_forward_search(site):
    """Explicit el_bore + start_time forward-searches from the anchor, el_bore respected."""
    anchor = _JUPITER_RISING_ANCHOR
    block = plan_source_ces(
        body="jupiter", footprint="c", el_bore=35.0, start_time=anchor, site=site
    )
    cp = block.computed_params
    # el_bore is honoured exactly (no derivation).
    assert cp["el_bore"] == pytest.approx(35.0)
    # Jupiter is below 35 deg at the anchor and climbs to it later, so the
    # forward search lands the pass strictly after the anchor.
    assert (Time(cp["t0_iso"]) - anchor).to_value(u.s) >= -1e-6


def test_anchored_matches_classic_window(site):
    """An anchored call equals the classic window call with its derived params."""
    anchor = _JUPITER_RISING_ANCHOR
    block = plan_source_ces(body="jupiter", footprint="c", start_time=anchor, site=site)
    cp = block.computed_params

    # Rebuild the window and el_bore the anchor resolved to and run the classic
    # form; the two must agree bit-for-bit on the pass endpoints.
    horizon = _anchor_module._DEFAULT_SEARCH_HORIZON_HOURS * 3600.0
    window = (anchor, anchor + TimeDelta(horizon * u.s))
    classic = plan_source_ces(
        body="jupiter",
        footprint="c",
        el_bore=cp["el_bore"],
        window=window,
        mode=cp["mode"],
        site=site,
    )
    assert classic.computed_params["t0_iso"] == cp["t0_iso"]
    assert classic.computed_params["t1_iso"] == cp["t1_iso"]


def test_anchored_compute_params_matches_plan(site):
    """compute_source_ces_params anchored path matches plan_source_ces's derived t0."""
    anchor = _JUPITER_RISING_ANCHOR
    params = compute_source_ces_params(body="jupiter", footprint="c", start_time=anchor, site=site)
    block = plan_source_ces(body="jupiter", footprint="c", start_time=anchor, site=site)

    assert set(params) == set(SourceCESComputedParams.__required_keys__)
    for key in SourceCESComputedParams.__required_keys__:
        expected = block.computed_params[key]
        actual = params[key]
        if isinstance(expected, float):
            assert actual == pytest.approx(expected), f"mismatch on key {key!r}"
        else:
            assert actual == expected, f"mismatch on key {key!r}"


def test_anchored_passes_first_pass_near_anchor(site):
    """Anchored plan_source_ces_passes starts the first pass near the anchor."""
    anchor = _JUPITER_RISING_ANCHOR
    blocks = plan_source_ces_passes(
        body="jupiter", footprint="c", n_passes=3, start_time=anchor, site=site
    )
    assert len(blocks) == 3
    assert all(isinstance(b, ScanBlock) for b in blocks)

    # The first pass in time is anchored; later passes follow.
    delta0 = (Time(blocks[0].computed_params["t0_iso"]) - anchor).to_value(u.s)
    assert delta0 >= -1e-6, f"first pass t0 must be >= anchor, got {delta0:+.3f}s"
    assert delta0 <= _ANCHOR_START_TOL_SEC, (
        f"first pass should start within 120 s of anchor, got {delta0:+.1f}s"
    )

    # Blocks time-ordered by start, with intact per-pass metadata.
    starts = [Time(b.computed_params["t0_iso"]).unix for b in blocks]
    assert starts == sorted(starts)
    assert [b.trajectory.metadata.pattern_params["pass_index"] for b in blocks] == [0, 1, 2]
    for b in blocks:
        assert set(b.computed_params) == set(SourceCESComputedParams.__required_keys__)
        assert b.computed_params["mode"] == "rising"


def test_anchored_mutual_exclusion_with_night_and_window(site):
    """start_time may not be combined with night or window."""
    with pytest.raises(ValueError, match="'start_time' or 'night'"):
        plan_source_ces(
            body="jupiter",
            footprint="c",
            start_time=_JUPITER_RISING_ANCHOR,
            night=_JUPITER_NIGHT,
            mode="rising",
            site=site,
        )
    with pytest.raises(ValueError, match="'start_time' or 'window'"):
        plan_source_ces(
            body="jupiter",
            footprint="c",
            start_time=_JUPITER_RISING_ANCHOR,
            window=(_JUPITER_NIGHT, _JUPITER_NIGHT + TimeDelta(1 * u.hour)),
            site=site,
        )


def test_missing_el_bore_without_start_time_raises(site):
    """Omitting el_bore in the classic (night/window) form raises a clear ValueError."""
    with pytest.raises(ValueError, match="el_bore is required"):
        plan_source_ces(
            body="jupiter",
            footprint="c",
            night=_JUPITER_NIGHT,
            mode="rising",
            site=site,
        )


def test_anchored_near_transit_guard(site):
    """Anchoring at transit (near-zero elevation drift) raises mentioning drift."""
    transit = _jupiter_transit_anchor(site)
    with pytest.raises(TargetNotObservableError, match="drift"):
        plan_source_ces(body="jupiter", footprint="c", start_time=transit, site=site)


def test_anchored_passes_first_pass_near_anchor_setting(site):
    """A setting anchor puts the highest pass first, starting near the anchor."""
    anchor = _jupiter_setting_anchor(site)
    blocks = plan_source_ces_passes(
        body="jupiter", footprint="c", n_passes=3, start_time=anchor, site=site
    )
    assert len(blocks) == 3
    assert all(b.computed_params["mode"] == "setting" for b in blocks)

    # The first pass in time is anchored.
    delta0 = (Time(blocks[0].computed_params["t0_iso"]) - anchor).to_value(u.s)
    assert delta0 >= -1e-6, f"first pass t0 must be >= anchor, got {delta0:+.3f}s"
    assert delta0 <= _ANCHOR_START_TOL_SEC, (
        f"first pass should start within 120 s of anchor, got {delta0:+.1f}s"
    )

    # Blocks time-ordered with intact per-pass metadata.
    starts = [Time(b.computed_params["t0_iso"]).unix for b in blocks]
    assert starts == sorted(starts)
    assert [b.trajectory.metadata.pattern_params["pass_index"] for b in blocks] == [0, 1, 2]

    # A setting source crosses higher elevations first, so the anchored first
    # pass must carry the highest boresight elevation and the top coverage row
    # (the largest eta offset of the default symmetric grid).
    etas = [b.trajectory.metadata.pattern_params["pass_eta_offset_deg"] for b in blocks]
    el_bores = [b.trajectory.metadata.pattern_params["pass_el_bore_deg"] for b in blocks]
    assert etas[0] == pytest.approx(max(etas))
    assert etas[0] > 0.0  # the top row of a symmetric grid is strictly positive
    assert el_bores[0] == pytest.approx(max(el_bores))


# Neptune rises through this anchor 2.9 deg below its 66.96 deg culmination. The
# el_bore derivation's probe lifts the i2 module so high that Neptune's arc reaches
# only the lowest of the module's 50 cover vertices. The anchors that do so form a
# band from about 04:35:48.71 to 04:35:49.90, and this is its middle.
_NEPTUNE_GRAZING_ANCHOR = Time("2026-09-11T04:35:49.303", scale="utc")
_NEPTUNE_I2 = dict(body="neptune", footprint="i2", az_speed=1.5, az_accel=1.0)


@pytest.fixture(scope="module")
def neptune_i2_refusal_with_padding():
    """Return the refusal of the grazing anchored call under the default padding."""
    with pytest.raises(TargetNotObservableError) as excinfo:
        plan_source_ces(start_time=_NEPTUNE_GRAZING_ANCHOR, site=get_fyst_site(), **_NEPTUNE_I2)
    return str(excinfo.value)


@pytest.mark.parametrize(
    "entry", [compute_source_ces_params, plan_source_ces, plan_source_ces_passes]
)
def test_an_anchor_whose_probe_grazes_one_vertex_ignores_the_padding(
    site, entry, neptune_i2_refusal_with_padding
):
    """With no padding the grazing anchor ends exactly as it does with the default.

    The probe reads only the start of its crossing, so the caller's padding has no
    say in the el_bore it derives, and the call reaches the real solve, which
    refuses: the derived el_bore lies below Neptune's elevation at the anchor, so
    the forward search takes the next day's arc, which the 24 h window cuts off
    below the module's top edge.
    """
    kwargs = dict(n_passes=1) if entry is plan_source_ces_passes else {}
    with pytest.raises(TargetNotObservableError) as excinfo:
        entry(
            start_time=_NEPTUNE_GRAZING_ANCHOR,
            site=site,
            az_padding=0.0,
            **_NEPTUNE_I2,
            **kwargs,
        )
    assert str(excinfo.value) == neptune_i2_refusal_with_padding


@pytest.mark.parametrize(
    "entry", [compute_source_ces_params, plan_source_ces, plan_source_ces_passes]
)
def test_the_anchor_probe_sweeps_with_the_kernel_padding_whatever_the_callers(
    site, monkeypatch, entry
):
    """The probe that derives el_bore never takes the caller's padding.

    Its start does not depend on the padding, and a caller's 0 would leave the
    probe's sweep without width wherever the source meets the lifted cover at a
    single vertex. Checked at an ordinary anchor, so it holds whether or not the
    call falls in such a band.
    """
    probe_calls = []
    real_kernel = _anchor_module._compute_source_ces_core

    def spy(**kwargs):
        probe_calls.append(kwargs)
        return real_kernel(**kwargs)

    monkeypatch.setattr(_anchor_module, "_compute_source_ces_core", spy)
    kwargs = dict(n_passes=1) if entry is plan_source_ces_passes else {}
    entry(
        body="jupiter",
        footprint="c",
        start_time=_JUPITER_RISING_ANCHOR,
        site=site,
        az_padding=0.0,
        **kwargs,
    )
    (probe,) = probe_calls
    default = _kernel_module._DEFAULT_AZ_PADDING_DEG
    assert probe.get("az_padding", default) == default


# Anchors at which the probe refuses: Neptune at 15.0 deg, below the elevation
# floor, on c; and just past the i2 band above, where it reaches no cover vertex.
@pytest.mark.parametrize(
    ("footprint", "anchor"),
    [("c", "2026-09-11T00:31:00"), ("i2", "2026-09-11T04:36:30")],
    ids=["below-the-floor", "past-the-band"],
)
@pytest.mark.parametrize(
    "entry", [compute_source_ces_params, plan_source_ces, plan_source_ces_passes]
)
def test_a_malformed_padding_is_refused_before_the_anchor_probe(site, entry, footprint, anchor):
    """A malformed az_padding is a ValueError even where the probe refuses the anchor.

    The probe never receives the caller's padding, so the padding is checked
    before the probe runs; otherwise the probe's refusal would report a malformed
    argument as an infeasibility.
    """
    kwargs = dict(n_passes=1) if entry is plan_source_ces_passes else {}
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="az_padding must be non-negative") as excinfo:
            entry(
                body="neptune",
                footprint=footprint,
                start_time=Time(anchor, scale="utc"),
                site=site,
                az_padding=bad,
                **kwargs,
            )
        assert not isinstance(excinfo.value, PointingError)


# The middles of bands of such anchors, on three modules, for planets and a
# fixed source, one, three and seven passes and two margins. At the first the
# real solve plans a pass; at the others it refuses.
_GRAZING_BAND_MIDDLES = [
    pytest.param(dict(body="saturn"), "i4", 1, 0.0, "2026-12-15T22:39:19.912", id="saturn-i4"),
    pytest.param(dict(ra=30.0, dec=-45.0), "i3", 1, 0.0, "2026-09-11T06:06:32.629", id="dec-45-i3"),
    pytest.param(
        dict(body="jupiter"),
        "i3",
        1,
        0.0,
        "2026-09-11T13:11:47.316",
        id="jupiter-i3",
        marks=pytest.mark.slow,
    ),
    pytest.param(dict(body="mars"), "i4", 3, 0.0, "2026-09-11T11:34:02.414", id="mars-i4-3-passes"),
    pytest.param(
        dict(body="saturn"),
        "i3",
        7,
        0.6,
        "2026-09-11T05:10:15.604",
        id="saturn-i3-7-passes-margin",
        marks=pytest.mark.slow,
    ),
]


@pytest.mark.filterwarnings(
    "ignore:High elevation reduces on-sky azimuth speed:"
    "fyst_trajectories.exceptions.PointingWarning"
)
@pytest.mark.parametrize(
    ("source", "module", "n_passes", "margin", "anchor"), _GRAZING_BAND_MIDDLES
)
def test_grazing_anchors_end_as_they_do_with_the_default_padding(
    site, source, module, n_passes, margin, anchor
):
    """At every grazing anchor, no padding gives the default padding's passes or refusal.

    Each call is the one the calibration night makes: the margined module, its
    pass count, a fast drag and no padding beside the solved throw.
    """
    kwargs = dict(
        source,
        footprint=inflate_footprint(resolve_footprint(module), margin),
        start_time=Time(anchor, scale="utc"),
        n_passes=n_passes,
        site=site,
        az_speed=1.5,
        az_accel=1.0,
    )

    def outcome(az_padding):
        try:
            blocks = plan_source_ces_passes(az_padding=az_padding, **kwargs)
        except PointingError as exc:
            return type(exc), str(exc)
        return [(b.computed_params["el_bore"], b.computed_params["t0_iso"]) for b in blocks]

    assert outcome(0.0) == outcome(0.5)


def test_anchored_below_elevation_floor_raises_target_not_observable(site):
    """Anchoring a source below the elevation floor raises an anchor-relative error."""
    times, el = _jupiter_el_track(site)
    floor = site.telescope_limits.elevation.min
    i_max = int(np.argmax(el))
    after = np.arange(len(el)) > i_max
    idx = np.where(after & (el <= floor - 5.0))[0]
    assert len(idx), "no below-floor Jupiter sample found on the test night"
    anchor = times[idx[0]]

    with pytest.raises(TargetNotObservableError, match="floor") as excinfo:
        plan_source_ces(body="jupiter", footprint="c", start_time=anchor, site=site)
    msg = str(excinfo.value)
    assert "Jupiter" in msg
    assert str(anchor.iso) in msg
    # The message reports the telescope floor and the source's elevation at
    # the anchor, not the internal probe boresight the derivation attempted.
    assert f"floor {floor}" in msg
    assert f"{float(el[idx[0]]):.2f}" in msg
    # The kernel's original bounds rejection is preserved for structured access.
    assert isinstance(excinfo.value.__cause__, ElevationBoundsError)
