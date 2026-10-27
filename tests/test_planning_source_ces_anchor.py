"""Tests for the approximate ``start_time`` anchor of the source-CES planners.

An anchored call plans a pass that starts about now: at or shortly after
the anchor, never before it, with ``el_bore`` and ``mode`` derived when
they are not given.
"""

from __future__ import annotations

import numpy as np
import pytest
from _source_ces_helpers import (
    _JUPITER_NIGHT,
    _JUPITER_RISING_ANCHOR,
    _MALFORMED_AZ_PADDINGS,
    _assert_same_pass,
)
from astropy import units as u
from astropy.time import Time, TimeDelta

import fyst_trajectories.planning.source_ces._anchor as _anchor_module
import fyst_trajectories.planning.source_ces._kernel as _kernel_module
from fyst_trajectories import (
    Coordinates,
    ElevationBoundsError,
    OffsetInversionError,
    PointingError,
    PointingWarning,
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


def _computed_params(result):
    """Return the computed parameters of an entry point's result (the only pass's, for a list)."""
    if isinstance(result, ScanBlock):
        return result.computed_params
    if isinstance(result, list):
        (block,) = result
        return block.computed_params
    return result


@pytest.fixture(scope="module")
def neptune_i2_pass_with_padding():
    """Return the boresight elevation and start of the grazing anchored pass, default padding."""
    cp = compute_source_ces_params(
        start_time=_NEPTUNE_GRAZING_ANCHOR, site=get_fyst_site(), **_NEPTUNE_I2
    )
    return cp["el_bore"], cp["t0_iso"]


@pytest.mark.filterwarnings(
    "ignore:High elevation reduces on-sky azimuth speed:"
    "fyst_trajectories.exceptions.PointingWarning"
)
@pytest.mark.parametrize(
    "entry", [compute_source_ces_params, plan_source_ces, plan_source_ces_passes]
)
def test_an_anchor_whose_probe_grazes_one_vertex_ignores_the_padding(
    site, entry, neptune_i2_pass_with_padding
):
    """With no padding the grazing anchor plans the pass the default padding plans.

    The probe reads only the start of its crossing, so the caller's padding has no
    say in the el_bore it derives. The module's cover sits above the boresight
    there, so the probe's value would lie below Neptune's elevation at the anchor;
    the derivation anchors el_bore on Neptune instead, and the pass starts when
    Neptune reaches the module, 3.3 min after the anchor.
    """
    kwargs = dict(n_passes=1) if entry is plan_source_ces_passes else {}
    result = entry(
        start_time=_NEPTUNE_GRAZING_ANCHOR,
        site=site,
        az_padding=0.0,
        **_NEPTUNE_I2,
        **kwargs,
    )
    cp = _computed_params(result)
    assert (cp["el_bore"], cp["t0_iso"]) == neptune_i2_pass_with_padding
    delay = Time(cp["t0_iso"], scale="utc") - _NEPTUNE_GRAZING_ANCHOR
    assert delay.to_value(u.s) == pytest.approx(196.9, abs=0.5)


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
# floor, on c; and just past the i2 band above, where it reaches no cover vertex
# (the call then derives el_bore from the cover's projection and plans).
@pytest.mark.parametrize(
    ("footprint", "anchor"),
    [("c", "2026-09-11T00:31:00"), ("i2", "2026-09-11T04:36:30")],
    ids=["below-the-floor", "past-the-band"],
)
@pytest.mark.parametrize(
    "entry", [compute_source_ces_params, plan_source_ces, plan_source_ces_passes]
)
def test_a_malformed_padding_is_refused_before_the_anchor_probe(site, entry, footprint, anchor):
    """A malformed az_padding is a ValueError naming it even where the probe refuses the anchor.

    A negative padding, every padding the planners refuse as malformed (a NaN,
    an infinity, text, past a full turn or past the float range) is
    refused at anchors where the probe would refuse. The probe never receives
    the caller's padding, so the padding is checked before the probe runs;
    otherwise the probe's refusal would report a malformed argument as an
    infeasibility.
    """
    kwargs = dict(n_passes=1) if entry is plan_source_ces_passes else {}
    for bad in (-0.1, float("inf"), *_MALFORMED_AZ_PADDINGS.values()):
        with pytest.raises(ValueError, match="az_padding") as excinfo:
            entry(
                body="neptune",
                footprint=footprint,
                start_time=Time(anchor, scale="utc"),
                site=site,
                az_padding=bad,
                **kwargs,
            )
        assert not isinstance(excinfo.value, PointingError), repr(bad)


# The middles of bands of such anchors, on three modules, for planets and a
# fixed source, one, three and seven passes and two margins. At the first three
# the real solve plans the pass; at the other two it refuses a later pass, whose
# cover reaches above the source's culmination.
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
        # A daytime pass, which the Sun screen reports.
        marks=[
            pytest.mark.slow,
            pytest.mark.filterwarnings(
                "ignore:EXCLUSION ZONE:fyst_trajectories.exceptions.PointingWarning"
            ),
        ],
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


# ---------------------------------------------------------------------------
# A cover that lies beyond the boresight along the source's motion
# ---------------------------------------------------------------------------

# Jupiter rises through 23.5 deg at this anchor. The i3 and i4 covers sit above
# the boresight there, so the probe's value put the boresight elevation behind
# Jupiter, which the search from the anchor meets only on the next day's arc;
# and between the probe's el_bore and the derived one the focal-plane rotation
# turns i2's lowest vertex down by more than the 0.05 deg lead, so its pass would
# have opened before the anchor.
_LEAD_RISING = Time("2026-03-15T20:45:00", scale="utc")
# Jupiter sets through 42.0 deg at this anchor, an hour past its culmination.
# The i1 and i6 covers sit below the boresight there, the mirror of i3 and i4;
# an hour later i2's highest vertex turns up past the lead, the mirror of i2
# rising.
_LEAD_SETTING = Time("2026-03-16T01:00:00", scale="utc")
_LEAD_SETTING_LATER = Time("2026-03-16T02:00:00", scale="utc")
_LEAD = _anchor_module._ANCHOR_START_LEAD_DEG


def _jupiter_el_at(site, t):
    """Jupiter's elevation in degrees at ``t``."""
    _, el = Coordinates(site).get_body_altaz("jupiter", t)
    return float(el)


# (anchor, module, mode, which check placed el_bore, seconds from the anchor to
# the pass start).
_PLACED_BY_THE_CHECKS = [
    pytest.param(_LEAD_RISING, "i2", "rising", "edge", 17.0, id="rising-i2"),
    pytest.param(_LEAD_RISING, "i3", "rising", "source", 283.6, id="rising-i3"),
    pytest.param(_LEAD_RISING, "i4", "rising", "source", 353.3, id="rising-i4"),
    pytest.param(_LEAD_SETTING, "i1", "setting", "source", 575.4, id="setting-i1"),
    pytest.param(
        _LEAD_SETTING, "i6", "setting", "source", 831.8, id="setting-i6", marks=pytest.mark.slow
    ),
    pytest.param(_LEAD_SETTING_LATER, "i2", "setting", "edge", 23.2, id="setting-i2"),
]


@pytest.mark.parametrize(("anchor", "module", "mode", "placed_by", "delay"), _PLACED_BY_THE_CHECKS)
def test_an_anchored_ring_module_pass_the_checks_place_starts_after_the_anchor(
    site, anchor, module, mode, placed_by, delay
):
    """A ring-module pass placed by one of the derivation's checks starts after the anchor.

    "source": the cover lies beyond the boresight along Jupiter's motion, so
    el_bore is anchored on Jupiter, the lead past its elevation at the anchor,
    and the pass starts minutes later, when Jupiter reaches the module. "edge":
    the cover straddles the boresight, and el_bore is re-solved to put its near
    edge the lead past Jupiter. Either way the anchored search with the
    boresight elevation and mode the pass records, which is what a rebuild from
    the pass's search start runs, returns the pass bit for bit.
    """
    block = plan_source_ces(body="jupiter", footprint=module, start_time=anchor, site=site)
    cp = block.computed_params
    assert cp["mode"] == mode
    sign = 1.0 if mode == "rising" else -1.0
    el_at_anchor = _jupiter_el_at(site, anchor)
    t0 = Time(cp["t0_iso"], scale="utc")
    assert (t0 - anchor).to_value(u.s) == pytest.approx(delay, abs=0.5)
    moved = sign * (_jupiter_el_at(site, t0) - el_at_anchor)
    if placed_by == "source":
        assert cp["el_bore"] == pytest.approx(el_at_anchor + sign * _LEAD, abs=1e-9)
        assert moved > _LEAD
    else:
        assert sign * (cp["el_bore"] - el_at_anchor) > _LEAD
        assert moved == pytest.approx(_LEAD, abs=1e-3)

    replay = plan_source_ces(
        body="jupiter",
        footprint=module,
        start_time=anchor,
        el_bore=cp["el_bore"],
        mode=mode,
        site=site,
    )
    _assert_same_pass(replay, block)


def test_allow_partial_leaves_a_derived_pass_whole(site):
    """allow_partial plans the same pass: the derivation puts the near edge ahead either way.

    This is the rising-i2 case above, where the second check lifts el_bore so the
    pass starts 17 s after the anchor with the whole cover crossed. allow_partial
    tolerates a source that turns before it crosses the cover, so it changes none
    of the parameters and raises no partial-cover advisory (an error in this
    suite); it does not clip the pass at the anchor to start it sooner.
    """
    kwargs = dict(body="jupiter", footprint="i2", start_time=_LEAD_RISING, site=site)
    whole = compute_source_ces_params(**kwargs)
    assert compute_source_ces_params(allow_partial=True, **kwargs) == whole
    delay = Time(whole["t0_iso"], scale="utc") - _LEAD_RISING
    assert delay.to_value(u.s) == pytest.approx(17.0, abs=0.5)


# ---------------------------------------------------------------------------
# An anchor within the probe's lift of the source's culmination
# ---------------------------------------------------------------------------

# Jupiter rises through 41.62 deg at this anchor, 2.45 deg below its culmination,
# within the 2.53 deg the probe lifts a ring module's el_bore (the cover's full
# reach from the boresight plus the buffer), so the probe refuses and el_bore is
# derived from the module's projected cover. The centre module's probe, lifted
# 0.75 deg, still reaches.
_NEAR_CULMINATION = Time("2026-03-15T23:00:00", scale="utc")

# (module, which check placed el_bore, seconds from the anchor to the pass start).
# The i3 and i4 covers lead the boresight there; the others trail it.
_PLANNED_NEAR_CULMINATION = [
    pytest.param("i1", "edge", 39.4, id="i1", marks=pytest.mark.slow),
    pytest.param("i2", "edge", 39.4, id="i2", marks=pytest.mark.slow),
    pytest.param("i3", "source", 966.8, id="i3"),
    pytest.param("i4", "source", 608.5, id="i4", marks=pytest.mark.slow),
    pytest.param("i5", "edge", 39.3, id="i5", marks=pytest.mark.slow),
    # The boresight 0.04 deg below the culmination.
    pytest.param("i6", "edge", 39.4, id="i6"),
]


@pytest.mark.parametrize(("module", "placed_by", "delay"), _PLANNED_NEAR_CULMINATION)
def test_a_ring_module_whose_probe_is_out_of_reach_plans_from_its_cover(
    site, monkeypatch, module, placed_by, delay
):
    """Near the culmination the probe refuses, and el_bore comes from the cover's projection.

    "edge": the cover trails the boresight, and el_bore is lifted until its near
    edge sits the lead ahead of Jupiter, so the pass starts about 39 s after the
    anchor (Jupiter climbs 1.3e-3 deg/s there). "source": the cover leads it,
    el_bore sits the lead ahead of Jupiter, and the pass starts when Jupiter
    reaches the module. Either way the anchored search with the boresight
    elevation and mode the pass records returns it bit for bit.
    """
    refused = []
    real_kernel = _anchor_module._compute_source_ces_core

    def probe(**kwargs):
        try:
            return real_kernel(**kwargs)
        except PointingError as exc:
            refused.append(exc)
            raise

    monkeypatch.setattr(_anchor_module, "_compute_source_ces_core", probe)
    block = plan_source_ces(
        body="jupiter", footprint=module, start_time=_NEAR_CULMINATION, site=site
    )
    (exc,) = refused
    assert isinstance(exc, TargetNotObservableError)

    cp = block.computed_params
    assert cp["mode"] == "rising"
    el_at_anchor = _jupiter_el_at(site, _NEAR_CULMINATION)
    t0 = Time(cp["t0_iso"], scale="utc")
    assert (t0 - _NEAR_CULMINATION).to_value(u.s) == pytest.approx(delay, abs=0.5)
    if placed_by == "source":
        assert cp["el_bore"] == pytest.approx(el_at_anchor + _LEAD, abs=1e-9)
    else:
        edge = _anchor_module._near_cover_edge(
            resolve_footprint(module), cp["el_bore"], None, site, 1.0
        )
        assert edge - el_at_anchor == pytest.approx(_LEAD, abs=1e-4)

    replay = plan_source_ces(
        body="jupiter",
        footprint=module,
        start_time=_NEAR_CULMINATION,
        el_bore=cp["el_bore"],
        mode="rising",
        site=site,
    )
    _assert_same_pass(replay, block)


def test_a_cover_taller_than_the_climb_left_is_refused_or_clipped_where_the_source_turns(site):
    """Where the cover does not fit below the culmination, the pass is refused or clipped.

    Saturn rises through 63.17 deg at this anchor, 1.16 deg below its culmination,
    less than a module's 1.3 deg extent. The real solve refuses the pass el_bore
    is derived for; allow_partial clips its far side instead, where Saturn turns,
    and the pass still starts after the anchor with the near side whole.
    """
    anchor = Time("2026-09-11T05:30:00", scale="utc")
    kwargs = dict(body="saturn", footprint="i2", start_time=anchor, site=site)
    with pytest.raises(TargetNotObservableError) as refused:
        compute_source_ces_params(**kwargs)
    # The cover-vs-arc guard: the module's extent against Saturn's arc.
    bounds = refused.value.bounds_error
    assert bounds.limit_max - bounds.limit_min == pytest.approx(1.3, abs=0.01)
    assert bounds.actual_max < bounds.limit_max
    with pytest.warns(PointingWarning, match="does not cover the footprint extent"):
        cp = compute_source_ces_params(allow_partial=True, **kwargs)
    assert Time(cp["t0_iso"], scale="utc") > anchor


def test_against_the_motion_a_probe_out_of_reach_is_still_a_refusal(site):
    """The fallback reads the cover at the anchor, which does not bound an arc against the motion.

    Jupiter sets through 42.0 deg at this anchor, 2.1 deg below its culmination,
    and the probe for a rising i1 pass lies above it. The rising arc asked for
    starts at Jupiter's next turn, so the probe's refusal stands rather than a
    pass on the next day's arc.
    """
    with pytest.raises(TargetNotObservableError):
        compute_source_ces_params(
            body="jupiter", footprint="i1", start_time=_LEAD_SETTING, mode="rising", site=site
        )


def test_an_offset_inversion_error_from_the_probe_reaches_the_caller(site, monkeypatch):
    """The fallback takes over only where the source turns before the probe's reach.

    This source culminates near the zenith, and at this anchor the probe's
    off-centre inverse for i3 does not converge. That refusal is not the
    source turning, so it reaches the caller rather than handing el_bore to
    the fallback.
    """
    refused = []
    real_kernel = _anchor_module._compute_source_ces_core

    def probe(**kwargs):
        try:
            return real_kernel(**kwargs)
        except PointingError as exc:
            refused.append(exc)
            raise

    monkeypatch.setattr(_anchor_module, "_compute_source_ces_core", probe)
    with pytest.raises(OffsetInversionError) as raised:
        compute_source_ces_params(
            ra=30.0,
            dec=-23.4,
            footprint="i3",
            start_time=Time("2026-09-11T06:58:20", scale="utc"),
            site=site,
        )
    (exc,) = refused
    assert raised.value is exc


# The one test in this file that runs the simulator tier's rebuild; marking the
# test rather than the module keeps the planning tests in the library-tier job.
@pytest.mark.offline
@pytest.mark.parametrize("module", ["i2", "i3"])
def test_a_ring_module_pass_rebuilds_from_its_calibration_night_record(site, module):
    """The pass the calibration night plans on the module rebuilds from its block's record.

    The pass is planned with the night's keywords and recorded as the night
    records it, with the anchor as ``search_start``; the simulator's rebuild
    repeats that search with the recorded boresight elevation and returns the
    planned pass, sample for sample.
    """
    from fyst_trajectories.overhead.models import CalibrationType, TimelineBlock
    from fyst_trajectories.overhead.simulation import _generate_trajectory_for_block

    anchor = _LEAD_RISING
    (planned,) = plan_source_ces_passes(
        body="jupiter",
        footprint=module,
        n_passes=1,
        start_time=anchor,
        site=site,
        az_speed=1.5,
        az_accel=1.0,
        az_padding=0.0,
    )
    cp = planned.computed_params
    traj = planned.trajectory
    scan_params = {
        "body": "jupiter",
        "footprint": module,
        "el_bore": cp["el_bore"],
        "mode": cp["mode"],
        "boresight_rot": None,
        "timestep": float(planned.config.timestep),
        "eta_offset_deg": 0.0,
        "pass_index": 0,
        "n_passes": 1,
        "az_speed": 1.5,
        "az_accel": 1.0,
        "az_padding": 0.0,
    }
    block = TimelineBlock.calibration(
        CalibrationType.PLANET_CAL,
        t_start=traj.start_time,
        duration=float(planned.duration),
        az=float(traj.az.min()),
        el=cp["el_bore"],
        scan_index=0,
        target="jupiter",
        az_end=float(traj.az.max()),
        az_final=float(traj.az[-1]),
        scan_params=scan_params,
        t0_scan=str(traj.start_time.iso),
        rising=True,
        extra_metadata={"search_start": [anchor.utc.jd1, anchor.utc.jd2]},
    )
    _assert_same_pass(_generate_trajectory_for_block(block, site), planned)


def test_passes_whose_eta_offsets_lift_the_centre_module_plan_after_the_anchor(site):
    """Eta offsets that lift the centre module's cover above the boresight plan from the anchor.

    The first pass in time drags Jupiter through the row 1.0 deg above the module
    centre, where the cover lies wholly above the boresight, so its el_bore is
    anchored on Jupiter and the pass starts when Jupiter reaches the cover; the
    second follows at the default el_step, after the first has ended.
    """
    anchor = _LEAD_RISING
    blocks = plan_source_ces_passes(
        body="jupiter", footprint="c", eta_offsets=[1.0, 1.6], start_time=anchor, site=site
    )
    etas = [b.trajectory.metadata.pattern_params["pass_eta_offset_deg"] for b in blocks]
    assert etas == [1.0, 1.6]
    first, second = (b.computed_params for b in blocks)
    assert first["el_bore"] == pytest.approx(_jupiter_el_at(site, anchor) + _LEAD, abs=1e-9)
    delays = [(Time(cp["t0_iso"], scale="utc") - anchor).to_value(u.s) for cp in (first, second)]
    assert delays == pytest.approx([108.0, 742.9], abs=0.5)
    assert Time(second["t0_iso"], scale="utc") > Time(first["t1_iso"], scale="utc")


# Anchored calls whose derived el_bore already put the boresight elevation and the
# near edge ahead of the source, recorded before the derivation checked either:
# the centre module and straddling and trailing ring covers, both directions. The
# checks leave each alone, so each derives the el_bore and the start it did.
_ALREADY_AHEAD = [
    pytest.param(
        "2026-03-15T20:45:00", "c", 24.19668429604154, "2026-03-15 20:45:17.042", id="rising-c"
    ),
    pytest.param(
        "2026-03-15T20:45:00",
        "i1",
        25.798616205686614,
        "2026-03-15 20:45:16.005",
        id="rising-i1",
        marks=pytest.mark.slow,
    ),
    pytest.param(
        "2026-03-15T20:45:00", "i5", 24.086523160850863, "2026-03-15 20:45:37.826", id="rising-i5"
    ),
    pytest.param(
        "2026-03-15T20:45:00",
        "i6",
        25.676682917811778,
        "2026-03-15 20:45:19.086",
        id="rising-i6",
        marks=pytest.mark.slow,
    ),
    pytest.param(
        "2026-03-16T01:00:00",
        "c",
        41.27546238551126,
        "2026-03-16 01:00:41.839",
        id="setting-c",
        marks=pytest.mark.slow,
    ),
    pytest.param(
        "2026-03-16T01:00:00", "i2", 41.00573883898531, "2026-03-16 01:00:02.621", id="setting-i2"
    ),
    pytest.param(
        "2026-03-16T01:00:00",
        "i3",
        39.611639361148455,
        "2026-03-16 01:00:40.273",
        id="setting-i3",
        marks=pytest.mark.slow,
    ),
    pytest.param(
        "2026-03-16T01:00:00", "i4", 39.910324796305325, "2026-03-16 01:00:49.856", id="setting-i4"
    ),
    pytest.param(
        "2026-03-16T01:00:00",
        "i5",
        41.58953634624911,
        "2026-03-16 01:01:37.947",
        id="setting-i5",
        marks=pytest.mark.slow,
    ),
]


@pytest.mark.parametrize(("iso", "module", "el_bore", "t0_iso"), _ALREADY_AHEAD)
def test_a_derivation_already_ahead_of_the_source_is_unchanged(site, iso, module, el_bore, t0_iso):
    """Where the probe's el_bore puts the pass after the anchor, it is the el_bore derived."""
    cp = compute_source_ces_params(
        body="jupiter", footprint=module, start_time=Time(iso, scale="utc"), site=site
    )
    assert cp["el_bore"] == pytest.approx(el_bore, rel=1e-12)
    assert cp["t0_iso"] == t0_iso


# Anchored calls whose explicit mode runs against Jupiter's motion at the
# anchor, recorded before the derivation checked anything: the requested arc
# starts at Jupiter's next turn, which the anchor does not bound. Applied
# regardless of the motion, the checks would move all three: i1's cover leads
# the boresight while Jupiter sets and i3's while it rises, and i2's near edge
# lies behind Jupiter's elevation at the anchor.
_AGAINST_THE_MOTION = [
    pytest.param(
        "2026-03-15T20:45:00",
        "setting",
        "i1",
        24.460825215970086,
        "2026-03-16 03:19:46.951",
        id="rising-anchor-setting-i1",
    ),
    pytest.param(
        "2026-03-15T20:45:00",
        "setting",
        "i2",
        23.087104301540105,
        "2026-03-16 03:19:39.495",
        id="rising-anchor-setting-i2",
    ),
    pytest.param(
        "2026-03-16T03:00:00",
        "rising",
        "i3",
        26.05690119850367,
        "2026-03-16 21:00:49.836",
        id="setting-anchor-rising-i3",
        marks=pytest.mark.slow,
    ),
]


@pytest.mark.parametrize(("iso", "mode", "module", "el_bore", "t0_iso"), _AGAINST_THE_MOTION)
def test_an_explicit_mode_against_the_motion_is_left_to_the_probe(
    site, iso, mode, module, el_bore, t0_iso
):
    """The checks apply only to a source that moves in the requested direction at the anchor."""
    cp = compute_source_ces_params(
        body="jupiter", footprint=module, start_time=Time(iso, scale="utc"), mode=mode, site=site
    )
    assert cp["mode"] == mode
    assert cp["el_bore"] == pytest.approx(el_bore, rel=1e-12)
    assert cp["t0_iso"] == t0_iso


@pytest.mark.slow
def test_passes_on_eta_shifted_centre_covers_already_ahead_are_unchanged(site):
    """A two-pass sequence whose first cover straddles the boresight derives what it did."""
    blocks = plan_source_ces_passes(
        body="jupiter",
        footprint="c",
        eta_offsets=[-0.3, 0.3],
        start_time=_LEAD_RISING,
        site=site,
    )
    got = [(b.computed_params["el_bore"], b.computed_params["t0_iso"]) for b in blocks]
    assert [e for e, _ in got] == pytest.approx([24.46966600471914, 25.767100751675894], rel=1e-12)
    assert [t for _, t in got] == ["2026-03-15 20:45:16.984", "2026-03-15 20:55:52.717"]
