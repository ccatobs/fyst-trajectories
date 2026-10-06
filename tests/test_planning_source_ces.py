"""Tests for :func:`fyst_trajectories.plan_source_ces`.

Exercises the source-tracking constant-elevation planner end-to-end
using real astronomy (FYST site, real planet ephemerides, real UTC
dates). Mocking is restricted to the optimiser and to forcing a
convergence-failure code path; the underlying coordinate transforms
and footprint geometry are exercised against astropy throughout.
"""

from __future__ import annotations

import copy
import dataclasses
import pickle
import warnings

import numpy as np
import pytest
from _source_ces_helpers import _FULL_PRIMECAM_MODULES, _JUPITER_NIGHT, _full_primecam_block
from _sun_stubs import allow_everything, block_everything
from astropy import units as u
from astropy.time import Time, TimeDelta

import fyst_trajectories.planning.source_ces as _source_ces_module
import fyst_trajectories.planning.source_ces._kernel as _source_ces_kernel
from fyst_trajectories import (
    FYST_AZ_MAX_VELOCITY,
    MODULE_FOV_RADIUS_DEG,
    ArrayFootprint,
    AzimuthBoundsError,
    ConstantElScanConfig,
    Coordinates,
    InstrumentOffset,
    PointingError,
    PointingWarning,
    ScanBlock,
    SourceCESComputedParams,
    TargetNotObservableError,
    Trajectory,
    boresight_to_detector,
    compute_focal_plane_rotation,
    compute_source_ces_params,
    get_primecam_offset,
    plan_source_ces,
    plan_source_ces_passes,
)
from fyst_trajectories.planning._types import validate_computed_params

# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_jupiter_rising_full_primecam(site):
    """Full PrimeCam Jupiter-rising CES returns a usable ScanBlock."""
    block = _full_primecam_block(site)

    assert isinstance(block, ScanBlock)
    cp = block.computed_params
    assert cp["mode"] == "rising"
    assert cp["el_bore"] == pytest.approx(35.0)
    assert cp["duration"] > 0
    assert cp["n_scans"] >= 1

    # Constant elevation: the underlying ConstantEl pattern holds el
    # fixed to within numerical noise.
    assert np.allclose(block.trajectory.el, 35.0, atol=1e-6)

    # v_az is finite and well below the FYST hardware limit.
    assert np.isfinite(cp["v_az"])
    assert abs(cp["v_az"]) < FYST_AZ_MAX_VELOCITY

    # Trajectory azimuth velocity stays within axis limits.
    assert np.all(np.isfinite(block.trajectory.az_vel))
    assert np.all(np.abs(block.trajectory.az_vel) <= FYST_AZ_MAX_VELOCITY)

    # Summary mentions the source and mode.
    assert "Jupiter" in block.summary
    assert "rising" in block.summary


def test_sidereal_setting_single_module(site):
    """Sidereal target on a non-centered module (PrimeCam-I1)."""
    block = plan_source_ces(
        ra=180.0,
        dec=-30.0,
        footprint="i1",
        el_bore=40.0,
        night=Time("2026-03-15T00:00:00", scale="utc"),
        mode="setting",
        site=site,
    )

    cp = block.computed_params
    assert cp["mode"] == "setting"
    assert cp["az_throw"] > 0
    assert np.isfinite(cp["az_start"])

    # PrimeCam-I1 sits ~1.78 deg off-axis (dy = -106.8 arcmin), so the boresight
    # azimuth that places the I1 detector on the source must differ measurably
    # from the boresight for a CENTRED footprint at the same target, not just be
    # "positive and finite". A silent fallback to az_bore = source_az (the bug the
    # gold test_off_centre_module_lands_on_source_during_pass guards) would
    # collapse this offset to ~0. Here the off-centre boresight lands ~1.5 deg
    # from the centred one.
    block_centre = plan_source_ces(
        ra=180.0,
        dec=-30.0,
        footprint="c",
        el_bore=40.0,
        night=Time("2026-03-15T00:00:00", scale="utc"),
        mode="setting",
        site=site,
    )
    assert 1.0 < abs(cp["az_start"] - block_centre.computed_params["az_start"]) < 2.0


def test_source_identity_recorded_in_pattern_params(site):
    """``pattern_params["body"]`` is the lower-case body, or None for an RA/Dec source."""
    common = dict(footprint="c", night=_JUPITER_NIGHT, site=site)
    body_block = plan_source_ces(body="JUPITER", el_bore=35.0, mode="rising", **common)
    radec_block = plan_source_ces(ra=180.0, dec=-30.0, el_bore=40.0, mode="setting", **common)

    assert body_block.trajectory.metadata.pattern_params["body"] == "jupiter"
    assert radec_block.trajectory.metadata.pattern_params["body"] is None
    # target_name stays the display label.
    assert body_block.trajectory.metadata.target_name == "Jupiter"
    assert radec_block.trajectory.metadata.target_name.startswith("RA=")


def test_explicit_window_auto_mode_detect(site):
    """When ``window`` is given and no ``mode``, the planner auto-detects."""
    # Window straddles a Jupiter rising arc identified empirically above.
    t1 = Time("2026-03-15T21:00:00", scale="utc")
    t2 = t1 + TimeDelta(2 * 3600 * u.s)
    block = plan_source_ces(
        body="jupiter",
        footprint="c",
        el_bore=35.0,
        window=(t1, t2),
        site=site,
    )
    assert block.computed_params["mode"] == "rising"


def test_v_az_override_skips_optimisation(site, monkeypatch):
    """Passing ``v_az`` short-circuits the optimiser."""
    called = {"n": 0}

    real_minimize = _source_ces_kernel.minimize

    def _spy(*args, **kwargs):
        called["n"] += 1
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr("fyst_trajectories.planning.source_ces._kernel.minimize", _spy)

    block = _full_primecam_block(site, v_az=0.005)

    assert called["n"] == 0
    assert block.computed_params["v_az"] == pytest.approx(0.005)


def test_explicit_array_footprint(site):
    """A user-supplied ``ArrayFootprint`` is accepted as-is."""
    fp = ArrayFootprint(
        center_xi_deg=0.0,
        center_eta_deg=0.0,
        cover_xi_deg=np.array([-0.1, 0.1, 0.1, -0.1]),
        cover_eta_deg=np.array([-0.1, -0.1, 0.1, 0.1]),
    )
    block = plan_source_ces(
        body="jupiter",
        footprint=fp,
        el_bore=35.0,
        night=_JUPITER_NIGHT,
        mode="rising",
        site=site,
    )

    # The cover polygon (4 vertices) flows through the planner; throw
    # is bounded by the source pass duration and the cover extent.
    assert block.computed_params["az_throw"] > 0


def test_computed_params_schema(site):
    """computed_params matches the SourceCESComputedParams schema exactly."""
    block = _full_primecam_block(site)
    assert set(block.computed_params) == set(SourceCESComputedParams.__required_keys__)


def test_source_ces_computed_params_message_signposts():
    """A source_ces computed_params check points the caller at the right validator.

    Passing ``"source_ces"`` raises ``KeyError`` whose message names the public
    ``validate_scan_params`` entry point, so a caller who reaches this corner is
    directed to the validator that does accept source-CES params.
    """
    with pytest.raises(KeyError, match="validate_scan_params"):
        validate_computed_params({}, "source_ces")


# ---------------------------------------------------------------------------
# Error / partial-coverage paths
# ---------------------------------------------------------------------------


def test_source_below_el_bore_raises(site):
    """A southern source that never reaches a high el_bore raises."""
    # Dec = -85 deg is below FYST's horizon for el_bore = 50 (FYST is
    # at lat -23, so the southern circumpolar cap is dec < -67).
    with pytest.raises(TargetNotObservableError):
        plan_source_ces(
            ra=0.0,
            dec=-85.0,
            footprint="c",
            el_bore=50.0,
            night=_JUPITER_NIGHT,
            mode="rising",
            site=site,
        )


def test_partial_coverage_allow_true_warns(site):
    """``allow_partial=True`` downgrades partial-cover error to a warning."""
    # Sidereal source at dec=-50 culminates at el ~ 90 - |-50 - (-23)| = 63 deg
    # from FYST (lat = -23 deg). With a 1 deg-radius circular cover centred on
    # el_bore=62.5 deg, ``el_cover_max = 63.5 deg`` exceeds the source's
    # ``el_src_max ~ 63 deg``, a textbook partial-coverage scenario where
    # the cover cap is just out of reach but the az track stays well
    # inside FYST limits.
    theta = np.linspace(0.0, 2.0 * np.pi, 50, endpoint=False)
    R = 1.0
    fp = ArrayFootprint(
        center_xi_deg=0.0,
        center_eta_deg=0.0,
        cover_xi_deg=R * np.cos(theta),
        cover_eta_deg=R * np.sin(theta),
    )

    with pytest.raises(TargetNotObservableError):
        plan_source_ces(
            ra=180.0,
            dec=-50.0,
            footprint=fp,
            el_bore=62.5,
            night=_JUPITER_NIGHT,
            mode="rising",
            site=site,
            allow_partial=False,
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        block = plan_source_ces(
            ra=180.0,
            dec=-50.0,
            footprint=fp,
            el_bore=62.5,
            night=_JUPITER_NIGHT,
            mode="rising",
            site=site,
            allow_partial=True,
        )
    assert isinstance(block, ScanBlock)
    assert any(
        issubclass(w.category, PointingWarning) and "does not cover the footprint" in str(w.message)
        for w in caught
    )


# A source at Dec -75 culminates at 37.84 deg. At el_bore 37.7 the first vertex of
# this cover sits on the boresight, inside the source's arc, and the second, a
# degree away in eta, projects 0.79 deg higher, above the culmination.
_TWO_VERTEX_COVER = ArrayFootprint(
    center_xi_deg=0.0,
    center_eta_deg=0.0,
    cover_xi_deg=np.array([0.0, 0.0]),
    cover_eta_deg=np.array([0.0, 1.0]),
)
_POINT_COVER = ArrayFootprint(
    center_xi_deg=0.0,
    center_eta_deg=0.0,
    cover_xi_deg=np.array([0.0]),
    cover_eta_deg=np.array([0.0]),
)
_DEC_M75_RISING = dict(ra=180.0, dec=-75.0, el_bore=37.7, night=_JUPITER_NIGHT, mode="rising")


def _computed(result):
    """Return the computed parameters of a result of any of the three planners."""
    if isinstance(result, list):
        return result[0].computed_params
    return result if isinstance(result, dict) else result.computed_params


@pytest.mark.filterwarnings(
    "ignore:Source RA=180.000, Dec=-75.000 elevation span:"
    "fyst_trajectories.exceptions.PointingWarning"
)
@pytest.mark.parametrize(
    "entry", [compute_source_ces_params, plan_source_ces, plan_source_ces_passes]
)
def test_a_partial_cover_reached_at_one_vertex_has_no_sweep_without_padding(site, entry):
    """The crossing of a single vertex has no azimuth extent to sweep.

    At this elevation the source reaches one vertex of the partial cover, so
    with no padding the swept window has no width. The request is well formed
    (at a lower ``el_bore`` the source crosses the whole cover), so it is a
    ``PointingError``; the default padding sweeps the same crossing at twice
    the padding.
    """
    kwargs = dict(_DEC_M75_RISING, footprint=_TWO_VERTEX_COVER, site=site, allow_partial=True)
    if entry is plan_source_ces_passes:
        kwargs["n_passes"] = 1
    with pytest.raises(PointingError, match="swept azimuth window has no width"):
        entry(az_padding=0.0, **kwargs)
    assert _computed(entry(**kwargs))["az_throw"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "entry", [compute_source_ces_params, plan_source_ces, plan_source_ces_passes]
)
def test_a_point_cover_with_no_padding_is_refused_as_malformed(site, entry):
    """A cover of one point has no crossing to sweep at any site or time.

    So with no padding and no throw the request is malformed, a plain
    ``ValueError`` raised before any computation; a padding or a throw gives
    the sweep its width.
    """
    kwargs = dict(_DEC_M75_RISING, footprint=_POINT_COVER, site=site)
    if entry is plan_source_ces_passes:
        kwargs["n_passes"] = 1
    with pytest.raises(ValueError, match="footprint cover is a single point") as excinfo:
        entry(az_padding=0.0, **kwargs)
    assert not isinstance(excinfo.value, PointingError)
    assert _computed(entry(**kwargs))["az_throw"] == pytest.approx(1.0)
    assert _computed(entry(az_throw=0.8, **kwargs))["az_throw"] == pytest.approx(0.8)


def test_sun_avoidance_warns_not_raises(site):
    """A planet near the Sun emits a PointingWarning but still returns."""
    # Mercury on 2026-05-15 lies inside FYST's 45 deg sun exclusion at
    # mid-Chilean-day (verified by hand).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        block = plan_source_ces(
            body="mercury",
            footprint="c",
            el_bore=40.0,
            night=Time("2026-05-15T00:00:00", scale="utc"),
            mode="rising",
            site=site,
        )

    assert isinstance(block, ScanBlock)
    assert any(issubclass(w.category, PointingWarning) and "Sun" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# Injectable sun_safe predicate (injectable seam) on the source-CES arc check.
# ---------------------------------------------------------------------------


def test_plan_source_ces_honors_injected_predicate(site):
    """An injected False predicate warns on an otherwise sun-safe Jupiter arc.

    The Jupiter-rising arc at ``_JUPITER_NIGHT`` clears FYST's scalar Sun zone
    (45 deg exclusion, 50 deg warning; the happy-path tests above run it
    silently), so an EXCLUSION ZONE warning here proves the injected directional
    model, not the scalar radius, drives the arc verdict end-to-end.
    """
    # Precondition: the default (scalar) arc check is silent for this arc.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _full_primecam_block(site)
        assert not [
            w for w in caught if issubclass(w.category, PointingWarning) and "Sun" in str(w.message)
        ]

    with pytest.warns(PointingWarning, match="EXCLUSION ZONE"):
        block = _full_primecam_block(site, sun_safe=block_everything)
    assert isinstance(block, ScanBlock)


def test_plan_source_ces_injected_predicate_receives_arc_samples(site):
    """The arc predicate is consulted per-sample with (az, el_bore, time)."""
    seen = []

    def spy(az, el, t):
        seen.append((float(az), float(el)))
        return True

    _full_primecam_block(site, sun_safe=spy)

    assert seen, "arc sun_safe predicate was never consulted"
    # The arc is probed at el_bore across many az positions/times.
    assert len(seen) > 1
    assert all(el == pytest.approx(35.0) for _, el in seen)


def test_arc_warning_names_the_earliest_unsafe_sample(site):
    """The swept-arc warning names the earliest unsafe time, not the first probe.

    The probe grid lists every low envelope edge, then every midpoint, then
    every high edge, so the first unsafe array position can be a late
    low-edge sample while a high-edge sample is unsafe at the pass start.
    """
    times = []

    def record(az, el, t):
        times.append(t)
        return True

    _full_primecam_block(site, sun_safe=record)
    n = len(times) // 3
    # The model is consulted in grid order. Unsafe: the last low-edge probe
    # (the latest time) and the first high-edge probe (the earliest time).
    unsafe = {n - 1, 2 * n}
    calls = iter(range(3 * n))

    def two_unsafe_probes(az, el, t):
        return next(calls) not in unsafe

    with pytest.warns(PointingWarning, match="enters the Sun") as caught:
        _full_primecam_block(site, sun_safe=two_unsafe_probes)
    message = next(str(w.message) for w in caught if "enters the Sun" in str(w.message))
    assert f"at {times[2 * n].iso}." in message
    assert times[n - 1].iso not in message


def test_plan_source_ces_allow_predicate_overrides_sun(site):
    """A permissive predicate suppresses the warning for Mercury at the Sun."""
    mercury_kwargs = dict(
        body="mercury",
        footprint="c",
        el_bore=40.0,
        night=Time("2026-05-15T00:00:00", scale="utc"),
        mode="rising",
        site=site,
    )

    # Precondition: scalar default warns (Mercury inside the 45 deg zone).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan_source_ces(**mercury_kwargs)
        assert [
            w for w in caught if issubclass(w.category, PointingWarning) and "Sun" in str(w.message)
        ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan_source_ces(**mercury_kwargs, sun_safe=allow_everything)
        assert not [
            w
            for w in caught
            if issubclass(w.category, PointingWarning) and "source-CES" in str(w.message)
        ]


def test_compute_source_ces_params_honors_injected_predicate(site):
    """compute_source_ces_params threads sun_safe through the shared core."""
    base_kwargs = dict(
        body="jupiter",
        footprint="c",
        el_bore=35.0,
        night=_JUPITER_NIGHT,
        mode="rising",
        site=site,
    )

    # Default (scalar) path is silent for this arc.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        params_default = compute_source_ces_params(**base_kwargs)
        assert not [
            w for w in caught if issubclass(w.category, PointingWarning) and "Sun" in str(w.message)
        ]

    with pytest.warns(PointingWarning, match="EXCLUSION ZONE"):
        params_blocked = compute_source_ces_params(**base_kwargs, sun_safe=block_everything)

    # The injected predicate is advisory only: the returned scalars are
    # identical regardless of the sun verdict.
    assert params_blocked == params_default


def test_convergence_failure_falls_back(site, monkeypatch):
    """A non-converging optimiser triggers the median-az-speed fallback."""

    class _BadResult:
        success = False
        x = np.array([0.0])

    def _bad_minimize(*_args, **_kwargs):
        return _BadResult()

    monkeypatch.setattr("fyst_trajectories.planning.source_ces._kernel.minimize", _bad_minimize)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        block = _full_primecam_block(site)

    assert isinstance(block, ScanBlock)
    assert any(
        issubclass(w.category, PointingWarning) and "did not converge" in str(w.message)
        for w in caught
    )
    # Fallback v_az = median source az speed, which is finite.
    assert np.isfinite(block.computed_params["v_az"])


def test_az_branch_wraps(site):
    """``az_branch`` re-expresses az_start in the chosen half-turn branch."""
    # A source setting in the west (az_start ~248 deg) has an image inside
    # FYST's [-180, 360] limits in both the 180 and the 0 branch, so the
    # branch choice is observable: 180 keeps ~248, 0 moves it to ~-112.
    common = dict(
        ra=180.0,
        dec=-30.0,
        footprint="c",
        el_bore=40.0,
        night=_JUPITER_NIGHT,
        mode="setting",
        site=site,
    )
    high = plan_source_ces(az_branch=180.0, **common).computed_params["az_start"]
    low = plan_source_ces(az_branch=0.0, **common).computed_params["az_start"]
    assert 0.0 <= high < 360.0
    assert -180.0 <= low < 180.0
    assert high - low == pytest.approx(360.0)

    # The wrap math for ``az_branch=-180`` produces az_start in [-360, 0),
    # which falls outside FYST's [-180, 360] hardware limits. The
    # post-build bounds check correctly rejects it with
    # AzimuthBoundsError. This pins both the wrap algebra and the
    # downstream safety net.
    with pytest.raises(AzimuthBoundsError):
        _full_primecam_block(site, az_branch=-180.0)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param(
            dict(body="jupiter", ra=180.0, dec=-30.0),
            "'body' or 'ra'/'dec', not both",
            id="both-body-and-radec",
        ),
        pytest.param(
            dict(), "must specify 'body' or both 'ra' and 'dec'", id="neither-body-nor-radec"
        ),
        pytest.param(
            dict(
                body="jupiter",
                window=(_JUPITER_NIGHT, _JUPITER_NIGHT + TimeDelta(1 * u.hour)),
                night=_JUPITER_NIGHT,
                mode="rising",
            ),
            "'window' or 'night', not both",
            id="both-window-and-night",
        ),
        pytest.param(
            dict(body="jupiter"),
            "must specify either 'window' or 'night'",
            id="neither-window-nor-night",
        ),
        pytest.param(
            dict(body="jupiter", night=_JUPITER_NIGHT),
            "'mode' is required when using 'night'",
            id="night-without-mode",
        ),
        pytest.param(
            dict(body="jupiter", night=_JUPITER_NIGHT, mode="upwards"),
            "mode must be 'rising' or 'setting'",
            id="invalid-mode-string",
        ),
    ],
)
def test_invalid_arg_combos_raise_value_error(site, kwargs, match):
    """Invalid keyword combinations raise ValueError before any astronomy runs."""
    full = dict(footprint="c", el_bore=40.0, site=site)
    full.update(kwargs)
    with pytest.raises(ValueError, match=match):
        plan_source_ces(**full)


def test_invalid_footprint_type_raises(site):
    """Footprints that are neither InstrumentOffset/str/sequence/ArrayFootprint reject."""
    with pytest.raises(TypeError):
        plan_source_ces(
            body="jupiter",
            footprint=42,  # type: ignore[arg-type]
            el_bore=35.0,
            night=_JUPITER_NIGHT,
            mode="rising",
            site=site,
        )


@pytest.mark.parametrize(
    "entry", [compute_source_ces_params, plan_source_ces, plan_source_ces_passes]
)
def test_negative_az_padding_is_refused(site, entry):
    """A negative ``az_padding`` would narrow the sweep below the footprint crossing."""
    kwargs = dict(n_passes=2) if entry is plan_source_ces_passes else {}
    with pytest.raises(ValueError, match="az_padding must be non-negative"):
        entry(
            body="jupiter",
            footprint="c",
            el_bore=35.0,
            night=_JUPITER_NIGHT,
            mode="rising",
            site=site,
            az_padding=-0.3,
            **kwargs,
        )


def test_empty_footprint_sequence_raises(site):
    with pytest.raises(ValueError, match="footprint sequence cannot be empty"):
        plan_source_ces(
            body="jupiter",
            footprint=[],
            el_bore=35.0,
            night=_JUPITER_NIGHT,
            mode="rising",
            site=site,
        )


def test_single_module_footprint_inscribes_module_fov_radius():
    """A single-module footprint's cover vertices lie at MODULE_FOV_RADIUS_DEG."""
    from fyst_trajectories.planning.footprints import resolve_footprint

    fp = resolve_footprint("c")
    radii = np.hypot(
        fp.cover_xi_deg - fp.center_xi_deg,
        fp.cover_eta_deg - fp.center_eta_deg,
    )
    assert np.allclose(radii, MODULE_FOV_RADIUS_DEG)


def test_invalid_offset_type_in_sequence(site):
    """A sequence with mixed types raises TypeError."""
    with pytest.raises(TypeError):
        plan_source_ces(
            body="jupiter",
            footprint=[InstrumentOffset(dx=0.0, dy=0.0), "not_an_offset"],  # type: ignore[list-item]
            el_bore=35.0,
            night=_JUPITER_NIGHT,
            mode="rising",
            site=site,
        )


def test_boresight_rot_changes_az_bore_for_off_centre_footprint(site):
    """Non-zero ``boresight_rot`` shifts the recovered ``az_start`` for an off-centre module."""
    # Use module I1 (centre offset = (0.0, -106.8 arcmin) ~ (0.0, -1.78 deg))
    # so the boresight recovery exercises the spherical inverse and is
    # actually sensitive to focal-plane rotation. Compare az_start
    # between boresight_rot=None (treated as 0) and boresight_rot=30.
    common = dict(
        ra=180.0,
        dec=-30.0,
        footprint="i1",
        el_bore=40.0,
        night=_JUPITER_NIGHT,
        mode="setting",
        site=site,
    )
    block_zero = plan_source_ces(**common, boresight_rot=None)
    block_rot = plan_source_ces(**common, boresight_rot=30.0)

    az_zero = block_zero.computed_params["az_start"]
    az_rot = block_rot.computed_params["az_start"]
    # The 30 deg boresight rotation should shift az_start by well more than
    # 0.1 deg for an off-centre module. Anything tighter than that would
    # mean the parameter has no observable effect on the geometry.
    assert abs(az_rot - az_zero) > 0.1, (
        f"boresight_rot did not affect az_start: {az_rot=} vs {az_zero=}"
    )


def test_proper_motion_requires_ref_epoch(site):
    """Non-zero proper motion without ref_epoch is rejected, not silently mis-propagated."""
    with pytest.raises(ValueError, match="ref_epoch"):
        compute_source_ces_params(
            ra=83.6,
            dec=22.0,
            pm_ra=100.0,
            pm_dec=-50.0,
            footprint="c",
            el_bore=45.0,
            night=Time("2026-03-15T00:00:00", scale="utc"),
            mode="rising",
            site=site,
        )


def test_proper_motion_path_runs(site):
    """A non-zero proper-motion call takes the proper-motion propagation path."""
    # Large PM (1000 mas/yr = 1 arcsec/yr in RA, about a tenth of Barnard's Star)
    # so the displacement at the planning epoch is unambiguously
    # different from the no-PM case.
    common = dict(
        ra=180.0,
        dec=-30.0,
        footprint="c",
        el_bore=40.0,
        night=_JUPITER_NIGHT,
        mode="setting",
        site=site,
    )
    ref = Time("2000-01-01T00:00:00", scale="utc")
    block_pm = plan_source_ces(
        **common,
        pm_ra=1000.0,
        pm_dec=500.0,
        ref_epoch=ref,
    )
    block_nopm = plan_source_ces(**common)

    assert isinstance(block_pm, ScanBlock)
    # The proper-motion and fixed-position paths produce different az
    # solutions for the same source at the same epoch.
    assert (
        block_pm.computed_params["az_start"] != block_nopm.computed_params["az_start"]
        or block_pm.computed_params["az_throw"] != block_nopm.computed_params["az_throw"]
    )

    # Magnitude guard: 1000 mas/yr (RA) + 500 mas/yr (dec) over ~26 yr from the
    # J2000 ref epoch is ~0.008 deg of on-sky motion; projected to the boresight
    # azimuth it shifts az_start by ~0.0048 deg (the throw is unchanged). Bounding
    # the shift rules out a units error (mas read as arcsec -> ~1000x too large)
    # and a silently-zero displacement, both of which the bare "differs" check
    # above would miss.
    daz = abs(block_pm.computed_params["az_start"] - block_nopm.computed_params["az_start"])
    assert 0.002 < daz < 0.02


def test_array_footprint_from_array_info_round_trip():
    """``ArrayFootprint.from_array_info`` round-trips a small SO-style dict."""
    # Use degree-units input for an easy comparison; verify the produced
    # arrays match the input numerically.
    cover_xi_deg = np.array([-0.1, 0.1, 0.1, -0.1])
    cover_eta_deg = np.array([-0.1, -0.1, 0.1, 0.1])
    info = {
        "center": (0.05, -0.05),
        "cover": (cover_xi_deg, cover_eta_deg),
    }
    fp = ArrayFootprint.from_array_info(info, units="deg")
    assert fp.center_xi_deg == pytest.approx(0.05)
    assert fp.center_eta_deg == pytest.approx(-0.05)
    np.testing.assert_allclose(fp.cover_xi_deg, cover_xi_deg)
    np.testing.assert_allclose(fp.cover_eta_deg, cover_eta_deg)

    # And the radian path applies the rad2deg scaling.
    info_rad = {
        "center": (np.deg2rad(0.05), np.deg2rad(-0.05)),
        "cover": (np.deg2rad(cover_xi_deg), np.deg2rad(cover_eta_deg)),
    }
    fp_rad = ArrayFootprint.from_array_info(info_rad, units="rad")
    assert fp_rad.center_xi_deg == pytest.approx(0.05)
    np.testing.assert_allclose(fp_rad.cover_xi_deg, cover_xi_deg)


@pytest.mark.parametrize("bad", ["arcmin", "RAD", "Deg", "", None])
def test_array_footprint_from_array_info_rejects_unknown_units(bad):
    """An unrecognised ``units`` is refused, not silently treated as degrees.

    The scale was chosen with ``"rad" if units == "rad" else 1.0``, so every
    other spelling fell through to degrees. A radian-valued cover labelled
    ``"RAD"`` was therefore read 57.3x too small, with no error anywhere.
    """
    info = {"center": (0.0, 0.0), "cover": (np.array([0.01]), np.array([0.0]))}
    with pytest.raises(ValueError, match="units must be 'rad' or 'deg'"):
        ArrayFootprint.from_array_info(info, units=bad)


def _square_footprint() -> ArrayFootprint:
    return ArrayFootprint(
        center_xi_deg=0.05,
        center_eta_deg=-0.05,
        cover_xi_deg=np.array([-0.1, 0.1, 0.1, -0.1]),
        cover_eta_deg=np.array([-0.1, -0.1, 0.1, 0.1]),
    )


def test_array_footprint_copies_cover_arrays():
    """Editing the arrays passed in leaves the footprint unchanged."""
    cover_xi = np.array([-0.1, 0.1, 0.1, -0.1])
    cover_eta = np.array([-0.1, -0.1, 0.1, 0.1])
    fp = ArrayFootprint(
        center_xi_deg=0.0,
        center_eta_deg=0.0,
        cover_xi_deg=cover_xi,
        cover_eta_deg=cover_eta,
    )

    cover_xi[0] = 42.0
    cover_eta[0] = 42.0

    assert fp.cover_xi_deg[0] == -0.1
    assert fp.cover_eta_deg[0] == -0.1


def test_array_footprint_cover_arrays_are_read_only():
    """An in-place write to a stored cover array raises."""
    fp = _square_footprint()

    with pytest.raises(ValueError, match="read-only"):
        fp.cover_xi_deg[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        fp.cover_eta_deg[0] = 1.0


@pytest.mark.parametrize(
    "duplicate",
    [
        lambda fp: pickle.loads(pickle.dumps(fp)),
        copy.deepcopy,
        copy.copy,
    ],
    ids=["pickle", "deepcopy", "copy"],
)
def test_array_footprint_copies_keep_read_only_arrays_and_every_field(duplicate):
    """Pickle, deepcopy and copy rebuild through the constructor.

    NumPy hands back writeable arrays from all three, so a footprint that
    did not rebuild through its constructor would lose the read-only flag;
    every field is compared so a field the rebuild dropped is caught.
    """
    fp = _square_footprint()

    dup = duplicate(fp)

    assert dup is not fp
    assert not dup.cover_xi_deg.flags.writeable
    assert not dup.cover_eta_deg.flags.writeable
    for f in dataclasses.fields(fp):
        assert np.array_equal(getattr(dup, f.name), getattr(fp, f.name)), f.name


def test_array_footprint_compares_and_hashes_by_identity():
    """Two footprints built from the same module are unequal and hashable."""
    from fyst_trajectories.planning.footprints import resolve_footprint

    fp_a = resolve_footprint("c")
    fp_b = resolve_footprint("c")

    assert isinstance(hash(fp_a), int)
    assert fp_a == fp_a
    assert fp_a != fp_b
    assert fp_a != dataclasses.replace(fp_a)
    assert len({fp_a, fp_b}) == 2


def test_scan_block_compares_and_hashes_by_identity():
    """A block is hashable and unequal to its replace copy, on every interpreter.

    A replace copy shares the trajectory object, so the generated ``==``
    returned ``True`` for it on every interpreter, while ``hash`` raised.
    """
    trajectory = Trajectory(
        times=np.array([0.0, 1.0, 2.0]),
        az=np.array([100.0, 101.0, 102.0]),
        el=np.full(3, 45.0),
        az_vel=np.ones(3),
        el_vel=np.zeros(3),
    )
    block = ScanBlock(
        trajectory=trajectory,
        config=ConstantElScanConfig(
            az_start=100.0,
            az_stop=102.0,
            elevation=45.0,
            az_speed=1.0,
            az_accel=1.0,
            timestep=0.1,
        ),
        duration=2.0,
    )

    assert isinstance(hash(block), int)
    assert block in {block}
    assert block == block
    assert block != dataclasses.replace(block)


# ---------------------------------------------------------------------------
# compute_source_ces_params (params-only sibling)
# ---------------------------------------------------------------------------


def test_compute_params_matches_plan_block(site):
    """compute_source_ces_params returns the same scalars as plan_source_ces."""
    kwargs = dict(
        body="jupiter",
        footprint=_FULL_PRIMECAM_MODULES,
        el_bore=35.0,
        night=_JUPITER_NIGHT,
        mode="rising",
        site=site,
    )
    params = compute_source_ces_params(**kwargs)
    block = plan_source_ces(**kwargs, timestep=0.1)

    assert set(params) == set(SourceCESComputedParams.__required_keys__)
    # All scalar invariants must match between the two paths.
    for key in SourceCESComputedParams.__required_keys__:
        expected = block.computed_params[key]
        actual = params[key]
        if isinstance(expected, float):
            assert actual == pytest.approx(expected), f"mismatch on key {key!r}"
        else:
            assert actual == expected, f"mismatch on key {key!r}"


def test_compute_params_no_trajectory_built(site, monkeypatch):
    """compute_source_ces_params must NOT call the trajectory builder."""

    def _explode(*_args, **_kwargs):
        raise AssertionError("trajectory builder must not be called in params-only path")

    monkeypatch.setattr(
        "fyst_trajectories.planning.source_ces._build_altaz_trajectory",
        _explode,
    )

    params = compute_source_ces_params(
        body="jupiter",
        footprint=_FULL_PRIMECAM_MODULES,
        el_bore=35.0,
        night=_JUPITER_NIGHT,
        mode="rising",
        site=site,
    )
    # Sanity: the function returned a populated dict, not a placeholder.
    assert params["el_bore"] == pytest.approx(35.0)
    assert params["mode"] == "rising"


def test_compute_params_az_envelope_validation(site, monkeypatch):
    """An out-of-range envelope raises AzimuthBoundsError before any trajectory work."""

    # If the envelope check ever fell through, the patched builder would fire.
    def _explode(*_args, **_kwargs):
        raise AssertionError(
            "trajectory builder must not run when envelope is already out of range"
        )

    monkeypatch.setattr(
        "fyst_trajectories.planning.source_ces._build_altaz_trajectory",
        _explode,
    )

    # Pick a v_az override that, multiplied by the source-pass duration,
    # pushes the envelope past the FYST azimuth max (360 deg). The source
    # pass for a centred footprint at el_bore=35 lasts about ten minutes, so
    # v_az=10 deg/s drives the drift-extended endpoint far past 360.
    with (
        pytest.warns(PointingWarning, match="EXCLUSION ZONE"),
        pytest.warns(PointingWarning, match="Required peak azimuth speed"),
        pytest.raises(AzimuthBoundsError),
    ):
        compute_source_ces_params(
            body="jupiter",
            footprint="c",
            el_bore=35.0,
            night=_JUPITER_NIGHT,
            mode="rising",
            site=site,
            v_az=10.0,
        )


def test_compute_params_az_envelope_directional_parity(site):
    """Emit-time envelope and full plan agree on az-bounds.

    The emit-time envelope check widens the static sweep only in the
    drift direction (sign of ``v_az``), matching the one-signed drift the
    executed trajectory applies (``az + v_az*times``). A Jupiter rising
    CES (el_bore=35) has a negative ``v_az`` so the real trajectory spans
    ``[33.37, 37.70]`` while the over-wide symmetric envelope would reach
    ~40.03 on the +az side the track never visits.

    Limits ``[-180, 39.0]`` bracket the real trajectory but fall inside
    the symmetric envelope, so ``compute_source_ces_params`` must NOT raise
    where ``plan_source_ces`` succeeds. Tightening the max below the real
    trajectory (``36.0 < 37.70``) must make BOTH raise.
    """
    base_az = site.telescope_limits.azimuth

    def _site_with_az_max(az_max):
        az = dataclasses.replace(base_az, min=-180.0, max=az_max)
        tl = dataclasses.replace(site.telescope_limits, azimuth=az)
        return dataclasses.replace(site, telescope_limits=tl)

    kwargs = dict(
        body="jupiter",
        footprint="c",
        el_bore=35.0,
        night=_JUPITER_NIGHT,
        mode="rising",
    )

    # Limits bracket the real [33.37, 37.70] track but not the symmetric
    # over-wide envelope (~40.03). plan_source_ces succeeds, so the
    # emit-time check must agree and NOT raise.
    site_ok = _site_with_az_max(39.0)
    block = plan_source_ces(site=site_ok, **kwargs)
    assert float(np.max(block.trajectory.az)) < 39.0
    # No raise => the params are returned (parity with plan_source_ces).
    params = compute_source_ces_params(site=site_ok, **kwargs)
    assert params["mode"] == "rising"

    # Genuine violation: max below the real trajectory's reach. Both the
    # emit-time check and the full plan must raise, in agreement.
    site_bad = _site_with_az_max(36.0)
    with pytest.raises(AzimuthBoundsError):
        compute_source_ces_params(site=site_bad, **kwargs)
    with pytest.raises(AzimuthBoundsError):
        plan_source_ces(site=site_bad, **kwargs)


def test_compute_params_v_az_override(site):
    """Explicit v_az is passed through to the returned params unchanged."""
    params = compute_source_ces_params(
        body="jupiter",
        footprint=_FULL_PRIMECAM_MODULES,
        el_bore=35.0,
        night=_JUPITER_NIGHT,
        mode="rising",
        site=site,
        v_az=0.005,
    )
    assert params["v_az"] == 0.005


def test_compute_params_raises_target_not_observable(site):
    """A southern source that never reaches el_bore raises TargetNotObservableError."""
    with pytest.raises(TargetNotObservableError):
        compute_source_ces_params(
            ra=0.0,
            dec=-85.0,
            footprint="c",
            el_bore=50.0,
            night=_JUPITER_NIGHT,
            mode="rising",
            site=site,
        )


def test_compute_params_no_timestep_kwarg(site):
    """compute_source_ces_params must reject the trajectory-only ``timestep`` kwarg."""
    import inspect

    sig = inspect.signature(compute_source_ces_params)
    assert "timestep" not in sig.parameters

    with pytest.raises(TypeError):
        compute_source_ces_params(
            body="jupiter",
            footprint="c",
            el_bore=35.0,
            night=_JUPITER_NIGHT,
            mode="rising",
            site=site,
            timestep=0.1,  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# Slow / cross-validation
# ---------------------------------------------------------------------------


def _so_chosen_rising_block(schedlib_source, source_name, el_bore):
    """Generate a rising SourceBlock from SO schedlib whose el span covers el_bore.

    Returns the first rising ``SourceBlock`` (SO PyEphem-backed) over a
    two-day window starting 2026-03-15 whose elevation range contains
    ``el_bore``. ``schedlib.source.get_site`` must already be patched to
    return the FYST site so the ephemeris resolves at FYST.
    """
    import datetime as _dt

    t0w = _dt.datetime(2026, 3, 15, tzinfo=_dt.timezone.utc)
    t1w = _dt.datetime(2026, 3, 17, tzinfo=_dt.timezone.utc)
    blocks = schedlib_source.source_gen_seq(source_name, t0w, t1w)
    for b in blocks:
        if b.mode != "rising":
            continue
        _, _, alt = b.get_az_alt()
        if alt.min() <= el_bore <= alt.max():
            return b
    raise AssertionError(f"no rising {source_name} block covers el_bore={el_bore}")


@pytest.mark.slow
@pytest.mark.filterwarnings(
    "ignore:EXCLUSION ZONE. planned source-CES on Saturn:"
    "fyst_trajectories.exceptions.PointingWarning",
)
def test_cross_validate_so_make_source_ces(site, monkeypatch):
    """Cross-validate source-CES geometry against SO ``schedlib.source.make_source_ces``.

    Resolves the xi/eta axis pairing and the boresight_rot sign
    by feeding **identical** inputs (same FYST site, same source +
    observing window, same ``array_info`` footprint, same
    ``boresight_rot``) to both Simons Observatory's quaternion
    implementation (``so3g.proj.quat`` via ``make_source_ces``) and
    fyst-trajectories' spherical-trig :func:`compute_source_ces_params`.

    Requires ``so3g`` (Linux only; no Windows wheel) and upstream ``schedlib``.
    Upstream schedlib ships SO sites only, so the test registers FYST in
    ``schedlib.source.SITES`` from this library's site constants. Skipped when
    either import is missing.
    Run with::

        pytest tests/test_planning_source_ces.py --run-slow -k cross_validate_so

    Convention findings (verified against so3g 0.2.7 / schedlib 0.4.0):

    * **xi/eta pairing: CORRECT.** ``ArrayFootprint`` pairs
      ``xi -> cross-elevation`` and ``eta -> elevation`` the same way SO's
      ``quat.rotation_xieta`` does, with no 90 deg axis swap. The projected
      cover lands at the same on-sky (az, el).
    * **boresight_rot sign: CORRECT.** fyst's additive
      ``+boresight_rot`` produces the same cover rotation as SO's
      ``quat.euler(2, -np.deg2rad(boresight_rot))``. The signs agree
      (they do **not** flip) at +/-20 deg and +/-45 deg.

    The one *expected* divergence: SO ``make_source_ces`` projects the
    cover in a frame with **no field rotation** (SO LAT has a corotator
    that holds the array fixed in az/el, so ``boresight_rot`` is already
    the net focal-plane angle), whereas fyst-trajectories, modelling
    Prime-Cam on a Nasmyth port, adds the **mechanical**
    ``nasmyth_sign*el`` term (the az_bore recovery and cover projection
    in ``_compute_source_ces_core``). The two therefore agree exactly
    for a **centred** footprint (where that rotation is a no-op on a
    symmetric circle) and for an **off-centre** footprint once the
    rotation term is reconciled by the documented bridge
    ``boresight_rot_fyst = boresight_rot_SO - nasmyth_sign*el``.
    This is a platform-physics difference (corotator vs Nasmyth), **not**
    a bug. (The bridge deliberately carries no parallactic angle: pa
    describes the horizon-to-celestial rotation, so adding it inside this
    az/el projection is unphysical, and a bridge that absorbed it would
    cancel it algebraically rather than validate it.)
    """
    so3g = pytest.importorskip("so3g")  # noqa: F841  Linux-only; gates the test
    schedlib_source = pytest.importorskip("schedlib.source")
    schedlib_instrument = pytest.importorskip("schedlib.instrument")

    source_name = "saturn"  # reaches el ~67 deg from FYST in the chosen window
    el_bore = 50.0

    # --- Make BOTH sides use the SAME site (FYST). SO's PyEphem path
    # resolves the source via ``schedlib.source.get_site()``, which
    # hard-defaults to 'lat'; register 'fyst', point get_site at it, and
    # clear the precomputed-source cache so the FYST site is actually used.
    monkeypatch.setitem(
        schedlib_source.SITES,
        "fyst",
        schedlib_source.Location(lat=site.latitude, lon=site.longitude, elev=site.elevation),
    )
    _orig_get_site = schedlib_source.get_site
    monkeypatch.setattr(schedlib_source, "get_site", lambda s="fyst": _orig_get_site(s))
    schedlib_source.PRECOMPUTED_SOURCES.clear()

    chosen = _so_chosen_rising_block(schedlib_source, source_name, el_bore)
    window = (Time(chosen.t0), Time(chosen.t1))

    radius_deg = float(MODULE_FOV_RADIUS_DEG)
    rots = [0.0, 20.0, -20.0, 45.0, -45.0]

    # Tolerances. az_start within 0.05 deg, az_throw within 0.05 deg, times
    # within sampling_step_seconds (30 s). We PIN ``v_az`` on both sides
    # rather than let each side solve it independently: the drift-rate
    # objective (minimise az_throw over v_az) is extremely shallow for a
    # near-stationary source, so SO's bare Nelder-Mead and fyst's
    # (xatol=1e-5, fatol=1e-4) Nelder-Mead converge to points ~1e-4 deg/s
    # apart, pure optimiser-tolerance noise, not a convention difference.
    # Pinning v_az removes that noise and makes az_start/az_throw a
    # deterministic function of the projection convention (exactly what
    # these checks exercise). Two pinned values exercise the no-drift and
    # with-drift code paths.
    AZ_TOL = 0.05
    THROW_TOL = 0.05
    SAMPLING_STEP = 30.0
    PINNED_VAZ = [0.0, -0.0035]  # deg/s: pure-tracking and a small drift

    def _fy_params(footprint, boresight_rot, v_az):
        # az_padding=0 matches SO (which does not pad); az_accel=1 and
        # sampling_step_seconds=30 match SO's defaults.
        return compute_source_ces_params(
            body=source_name,
            footprint=footprint,
            el_bore=el_bore,
            boresight_rot=boresight_rot,
            window=window,
            mode="rising",
            site=site,
            sampling_step_seconds=SAMPLING_STEP,
            az_accel=1.0,
            az_padding=0.0,
            az_branch=None,
            allow_partial=False,
            v_az=v_az,
        )

    # =====================================================================
    # Part 1 - common case: CENTRED footprint.
    #
    # For a symmetric circular cover centred on the boresight, fyst's
    # mechanical ``nasmyth_sign*el`` rotation is a no-op, so the
    # AS-SHIPPED planner must match SO directly at every boresight_rot.
    # A swapped xi/eta axis or a flipped boresight_rot sign
    # would shift az_start here.
    # =====================================================================
    ai_c = schedlib_instrument.make_circular_cover(0.0, 0.0, radius_deg, degree=True)
    fp_c = ArrayFootprint.from_array_info(ai_c, units="rad")
    for v_az in PINNED_VAZ:
        for rot in rots:
            so_block = schedlib_source.make_source_ces(
                chosen, array_info=ai_c, el_bore=el_bore, v_az=v_az, boresight_rot=rot
            )
            assert so_block is not None, f"SO returned None (centred, rot={rot})"
            fy = _fy_params(fp_c, rot, v_az)

            assert fy["az_start"] == pytest.approx(float(so_block.az), abs=AZ_TOL), (
                f"centred az_start mismatch at boresight_rot={rot}, v_az={v_az}: "
                f"fyst={fy['az_start']:.4f} SO={float(so_block.az):.4f}"
            )
            assert fy["az_throw"] == pytest.approx(float(so_block.throw), abs=THROW_TOL), (
                f"centred az_throw mismatch at boresight_rot={rot}, v_az={v_az}: "
                f"fyst={fy['az_throw']:.4f} SO={float(so_block.throw):.4f}"
            )
            # v_az is pinned identically on both sides, so it round-trips.
            assert fy["v_az"] == pytest.approx(v_az, abs=1e-9)
            dt0 = abs((Time(fy["t0_iso"]) - Time(so_block.t0)).to_value(u.s))
            dt1 = abs((Time(fy["t1_iso"]) - Time(so_block.t1)).to_value(u.s))
            assert dt0 <= SAMPLING_STEP, f"centred t0 off by {dt0:.1f}s at rot={rot}"
            assert dt1 <= SAMPLING_STEP, f"centred t1 off by {dt1:.1f}s at rot={rot}"

    # =====================================================================
    # Part 2 - discriminating case: OFF-CENTRE i1 module.
    #
    # i1 sits at (xi=0, eta~-1.78 deg), asymmetric, so a 90 deg axis swap or a
    # mirrored boresight rotation would NOT cancel. SO and fyst encode
    # different telescopes here (SO corotator vs FYST Nasmyth), so the
    # only way to prove the xi/eta pairing and boresight sign are correct
    # is to reconcile the documented mechanical-rotation convention bridge:
    #
    #     boresight_rot_fyst = boresight_rot_SO - nasmyth_sign*el
    #
    # If the axis pairing or the boresight sign were wrong, parity would NOT
    # hold even after this bridge.
    # =====================================================================
    i1_eta_deg = float(get_primecam_offset("i1").dy_deg)
    ai_i1 = schedlib_instrument.make_circular_cover(0.0, i1_eta_deg, radius_deg, degree=True)
    fp_i1 = ArrayFootprint.from_array_info(ai_i1, units="rad")

    # Mechanical focal-plane rotation at el_bore, computed the same way the
    # planner does (horizon-frame term only, no parallactic angle).
    field_rot = float(
        compute_focal_plane_rotation(
            el=el_bore,
            site=site,
            offset=InstrumentOffset(dx=0.0, dy=0.0),
        )
    )  # = nasmyth_sign*el_bore

    for v_az in PINNED_VAZ:
        for rot in rots:
            so_block = schedlib_source.make_source_ces(
                chosen, array_info=ai_i1, el_bore=el_bore, v_az=v_az, boresight_rot=rot
            )
            assert so_block is not None, f"SO returned None (i1, rot={rot})"
            # Bridge: hand fyst the boresight_rot that cancels its extra
            # field-rotation term, leaving the same net cover rotation SO uses.
            fy = _fy_params(fp_i1, rot - field_rot, v_az)

            assert fy["az_start"] == pytest.approx(float(so_block.az), abs=AZ_TOL), (
                f"i1 az_start mismatch at boresight_rot_SO={rot}, v_az={v_az} "
                f"(a wrong axis pairing or boresight sign would break this): "
                f"fyst={fy['az_start']:.4f} SO={float(so_block.az):.4f}"
            )
            assert fy["az_throw"] == pytest.approx(float(so_block.throw), abs=THROW_TOL), (
                f"i1 az_throw mismatch at boresight_rot_SO={rot}, v_az={v_az}: "
                f"fyst={fy['az_throw']:.4f} SO={float(so_block.throw):.4f}"
            )
            assert fy["v_az"] == pytest.approx(v_az, abs=1e-9)


# ---------------------------------------------------------------------------
# Tier-1 integration smoke tests
#
# These two tests are the local stand-in for the schedlib cross-validation
# (test_cross_validate_so_make_source_ces above). They do not need
# schedlib + so3g installed but they verify the same end-to-end property:
# the planner's trajectory + cover geometry actually places the source on
# the focal-plane footprint at the times it claims to. They sit at the
# integration layer (planner output x Coordinates x offset math) and would
# catch the high-impact class of bug (xi/eta sign convention, focal-plane
# rotation sign, drift-rate solve, off-centre boresight recovery) that a
# schedlib parity check would catch.
# ---------------------------------------------------------------------------


def test_source_lies_inside_swept_cover_along_trajectory(site):
    """Verify the source lies inside the *swept* cover over the source pass.

    For a full-PrimeCam Jupiter-rising CES, sample the source-pass window
    ``[t0, t1]`` from ``computed_params`` at 20 uniform times. At each time,
    recover the commanded boresight (az, el) from the trajectory and project
    the 350 cover-polygon vertices to on-sky (az, el) through that boresight.
    Take the UNION of all projected covers across all sample times. This is
    the on-sky footprint actually swept by the array while the source is
    crossing it. Assert that the source's instantaneous (az, el) at *each*
    sample time lies inside the convex hull of that swept union.

    The integration property being verified is: "the source's track over
    [t0, t1] lies inside the union of all swept covers", the actual
    coverage claim of a back-and-forth CES. Two important geometric notes:

    * Per-instant containment is the wrong check: in a multi-leg CES the
      boresight oscillates and the instantaneous cover need not contain
      the source at every leg endpoint. Only the swept union must.
    * We sample within ``[t0, t1]``, NOT over the full trajectory duration.
      The trajectory continues to scan after ``t1`` because ``n_scans`` is
      rounded up; the source genuinely exits the cover at ``t1`` by design,
      and sampling past ``t1`` is uninformative.

    What this pins is end-to-end self-consistency: the trajectory the planner
    actually built sweeps a cover that contains the source track over its own
    ``[t0, t1]``. Conventions are NOT pinned here. The union of the ring's
    swept covers is nearly rotation-symmetric and the azimuth envelope is
    widened to follow the drift, so adding 90, -90 or 180 deg to the cover
    rotation, swapping xi/eta, or planning with a wrong ``v_az`` all still
    land a containing hull. ``test_cross_validate_so_make_source_ces``
    resolves the xi/eta pairing and the ``boresight_rot`` sign against SO's
    quaternion implementation; this case is centred, so ``az_bore`` is just
    the source azimuth and the off-centre recovery is exercised by
    ``test_off_centre_module_lands_on_source_during_pass``.

    For the centred full-PrimeCam case we additionally require the source
    to sit near the swept-envelope centroid at the trajectory midpoint, a
    coarse check: it tolerates offsets up to half the swept envelope's
    median radius, which is larger than the footprint's own radius.
    """
    from scipy.spatial import ConvexHull, Delaunay, QhullError

    block = _full_primecam_block(site)
    traj = block.trajectory
    cp = block.computed_params
    el_bore = float(cp["el_bore"])

    coords = Coordinates(site)

    assert traj.start_time is not None, "source-CES trajectory must carry start_time"
    t_start = traj.start_time

    # 20 uniform samples within the *source-pass* window [t0, t1].
    # NOTE: the trajectory's actual duration extends beyond t1 because
    # ``n_scans`` is rounded up (the back-and-forth CES continues to scan
    # after the source has left the cover). Sampling beyond t1 is
    # uninformative. The source is genuinely outside the cover there
    # by design.
    t0_iso = cp["t0_iso"]
    t1_iso = cp["t1_iso"]
    t0_rel = (Time(t0_iso) - t_start).to_value(u.s)
    t1_rel = (Time(t1_iso) - t_start).to_value(u.s)
    n_samples = 20
    t_secs_samples = np.linspace(t0_rel, t1_rel, n_samples)
    sample_times = t_start + TimeDelta(t_secs_samples * u.s)

    # The footprint that flowed into the planner, re-resolved here so the
    # test is self-contained.
    from fyst_trajectories.planning.footprints import resolve_footprint

    fp = resolve_footprint(_FULL_PRIMECAM_MODULES)
    assert fp.cover_xi_deg.size == 350  # 7 modules x 50 vertices

    # Buffer (degrees) added to the convex-hull containment test. The
    # window ``[t0, t1]`` is bounded by the source entering and leaving the
    # cover, so the first and last samples sit ON the swept hull boundary;
    # the buffer keeps that tangency from failing on rounding, and 0.1 deg
    # is well below the per-leg az_padding (0.5 deg).
    HULL_BUFFER_DEG = 0.1

    # Pass 1: collect the swept cover (union of all projected covers).
    swept_az = []
    swept_el = []
    source_pts = []  # (src_az, src_el) at each sample time
    bore_pts = []  # (bore_az, bore_el) at each sample time, for diagnostics

    for t_k_sec, t_k in zip(t_secs_samples, sample_times):
        src_az_k, src_el_k = coords.get_body_altaz("jupiter", t_k)
        src_az_k = float(src_az_k)
        src_el_k = float(src_el_k)
        source_pts.append((src_az_k, src_el_k))

        bore_az = float(np.interp(t_k_sec, traj.times, traj.az))
        bore_el = float(np.interp(t_k_sec, traj.times, traj.el))
        bore_pts.append((bore_az, bore_el))

        # Mechanical (horizon-frame) rotation, matching the planner's
        # az/el projection convention (no parallactic angle).
        fp_rot_k = float(
            compute_focal_plane_rotation(
                el=el_bore,
                site=site,
                offset=InstrumentOffset(dx=0.0, dy=0.0),
            )
        )  # boresight_rot defaulted to 0 for this case.

        # Project all 350 cover vertices through the boresight.
        for xi_deg, eta_deg in zip(fp.cover_xi_deg, fp.cover_eta_deg):
            vertex_offset = InstrumentOffset(dx=xi_deg * 60.0, dy=eta_deg * 60.0)
            az_v, el_v = boresight_to_detector(
                az=bore_az,
                el=bore_el,
                offset=vertex_offset,
                focal_plane_rotation=fp_rot_k,
            )
            swept_az.append(float(az_v))
            swept_el.append(float(el_v))

    swept_points = np.column_stack([swept_az, swept_el])
    try:
        swept_hull = ConvexHull(swept_points)
    except QhullError as exc:  # pragma: no cover - geometric guard
        raise AssertionError(f"Swept cover polygon degenerate: {exc}") from exc
    swept_vertices = swept_points[swept_hull.vertices]

    # Inflate the swept hull by pushing each vertex outward
    # HULL_BUFFER_DEG along its own centroid-to-vertex unit normal. This
    # uniform-thickness expansion is robust for elongated polygons (the
    # swept cover is wide in az, narrow in el) where centroid-relative
    # scaling would inflate the long axis far more than the short axis.
    swept_centroid = swept_vertices.mean(axis=0)
    vecs = swept_vertices - swept_centroid
    unit = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    inflated = swept_vertices + HULL_BUFFER_DEG * unit
    swept_tri = Delaunay(inflated)

    # Pass 2: assert each source position lies inside the swept hull.
    for (src_az_k, src_el_k), (bore_az, bore_el), t_k in zip(source_pts, bore_pts, sample_times):
        source_point = np.array([src_az_k, src_el_k])
        assert swept_tri.find_simplex(source_point) >= 0, (
            f"source ({src_az_k:.4f}, {src_el_k:.4f}) lies outside the swept "
            f"cover at t_k={t_k.iso} (boresight=({bore_az:.4f}, {bore_el:.4f}))"
        )

    # Centred-case sanity: at the trajectory midpoint, the source should
    # sit near the swept-envelope centroid (within half the median cover
    # radius). This is a coarse check on systematic offsets; it does not
    # discriminate the cover-rotation convention (see the docstring).
    mid_idx = len(source_pts) // 2
    mid_src = np.array(source_pts[mid_idx])
    median_radius = float(np.median(np.linalg.norm(swept_vertices - swept_centroid, axis=1)))
    d_mid = float(np.linalg.norm(mid_src - swept_centroid))
    assert d_mid < 0.5 * median_radius, (
        f"source at trajectory midpoint is not near swept-cover centroid: "
        f"d_mid={d_mid:.4f} deg, median radius={median_radius:.4f} deg"
    )


def test_off_centre_module_lands_on_source_during_pass(site):
    """Verify off-centre az_bore recovery places the I1 module on the source.

    For a single-module (PrimeCam-I1, dy = -106.8 arcmin off-axis) sidereal
    source CES, scan along the planned trajectory and find the time at
    which the forward-projected I1 module position is closest to the
    source's instantaneous (az, el). Assert that the minimum miss
    distance is sub-arcmin.

    Important geometric note: the planner's reference time
    ``t_at_el_bore`` (when the source's elevation equals ``el_bore``)
    is NOT, in general, inside the trajectory window for an off-centre
    footprint. The cover is offset below the boresight by the module's
    ``dy``, so projected from a boresight at ``el_bore=40`` deg the cover
    sits at lower elevations. The trajectory window
    ``[t0, t1]`` is bounded by the source el span overlapping the cover
    el span, which for I1 puts source el ~ 41 (above el_bore=40) during
    the actual scan. The I1 module's centre lands on the source somewhere
    near the trajectory midpoint, NOT at ``t_at_el_bore``.

    This test catches the case where the planner's Step-6 spherical
    inverse (:func:`detector_to_boresight`) silently fails and falls back
    to ``az_bore = source_az``, which is *wrong* for any off-centre
    footprint (the I1 module would then sit ~1.8 deg away from the source,
    not on it).
    """
    # Sidereal source, off-centre module. Match the existing
    # test_sidereal_setting_single_module setup so the geometry is known.
    ra_deg = 180.0
    dec_deg = -30.0
    el_bore = 40.0
    night = Time("2026-03-15T00:00:00", scale="utc")

    block = plan_source_ces(
        ra=ra_deg,
        dec=dec_deg,
        footprint="i1",
        el_bore=el_bore,
        night=night,
        mode="setting",
        site=site,
    )
    traj = block.trajectory
    cp = block.computed_params
    coords = Coordinates(site)
    i1_offset = get_primecam_offset("i1")

    assert traj.start_time is not None

    # Walk the trajectory at a moderate cadence (~100 samples) and
    # find the time at which the projected I1 module position is closest
    # to the source. By symmetry this is near the trajectory midpoint;
    # we don't hardcode "midpoint" because the back-and-forth scan
    # pattern means the boresight crosses the source-track multiple
    # times, but the *closest-approach* is the canonical pinch-point
    # where the planner's az_bore-recovery is tested.
    n_walk = 100
    t_walk_sec = np.linspace(traj.times[0], traj.times[-1], n_walk)
    misses_deg = np.empty(n_walk)
    diagnostics = []
    for k, t_k_sec in enumerate(t_walk_sec):
        t_k = traj.start_time + TimeDelta(t_k_sec * u.s)
        src_az_k, src_el_k = coords.radec_to_altaz(ra_deg, dec_deg, t_k)
        src_az_k = float(src_az_k)
        src_el_k = float(src_el_k)
        bore_az = float(np.interp(t_k_sec, traj.times, traj.az))
        bore_el = float(np.interp(t_k_sec, traj.times, traj.el))
        # Mechanical (horizon-frame) rotation, matching the planner's
        # az/el projection convention (no parallactic angle).
        fp_rot_k = float(
            compute_focal_plane_rotation(
                el=el_bore,
                site=site,
                offset=InstrumentOffset(dx=0.0, dy=0.0),
            )
        )  # boresight_rot defaulted to None -> treated as 0.
        det_az, det_el = boresight_to_detector(
            az=bore_az,
            el=bore_el,
            offset=i1_offset,
            focal_plane_rotation=fp_rot_k,
        )
        det_az = float(det_az)
        det_el = float(det_el)
        miss_deg = float(
            np.hypot(
                (det_az - src_az_k) * np.cos(np.deg2rad(src_el_k)),
                det_el - src_el_k,
            )
        )
        misses_deg[k] = miss_deg
        diagnostics.append((t_k.iso, src_az_k, src_el_k, bore_az, bore_el, det_az, det_el))

    k_best = int(np.argmin(misses_deg))
    best_miss = float(misses_deg[k_best])

    # Tolerance: the closed-form spherical inverse converges to sub-udeg,
    # but the boresight is held at ``el_bore`` while the source sweeps in
    # elevation, so the module's closest approach to the source is about
    # an arcmin, and the 100-sample walk above resolves it only to a few
    # arcmin. 6 arcmin is well below the module's 0.65 deg (39 arcmin)
    # FOV radius, well inside the module, and an order of magnitude
    # smaller than the dy offset (107 arcmin) a true az_bore-recovery bug
    # would expose (the planner would fall back to ``az_bore =
    # source_az``, missing by the full ``dy``).
    TOL_ARCMIN = 6.0
    assert best_miss * 60.0 < TOL_ARCMIN, (
        f"I1 module never lands on source: closest approach {best_miss * 60:.2f} arcmin "
        f"at sample {k_best}/{n_walk} (t={diagnostics[k_best][0]}). "
        f"diagnostics={diagnostics[k_best]}. computed_params={dict(cp)}"
    )


# ---------------------------------------------------------------------------
# Core-parameter guards
# ---------------------------------------------------------------------------


class TestNumericParameterGuards:
    """The shared source-CES core rejects non-positive algorithm parameters."""

    @pytest.mark.parametrize("bad", [0, -30])
    def test_non_positive_sampling_step_raises(self, site, bad):
        with pytest.raises(ValueError, match="sampling_step_seconds must be positive"):
            _full_primecam_block(site, sampling_step_seconds=bad)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_az_accel_raises(self, site, bad):
        # A negative az_accel would otherwise yield a silently wrong duration.
        with pytest.raises(ValueError, match="az_accel must be positive"):
            _full_primecam_block(site, az_accel=bad)


class TestPostDriftDynamicsCheck:
    """The returned (drifted) trajectory is dynamics-checked exactly once."""

    @staticmethod
    def _block(site, az_accel):
        # One module keeps the solve short; the check is on the trajectory
        # shape, not the footprint.
        return _full_primecam_block(site, footprint="c", az_accel=az_accel)

    def test_default_acceleration_is_advisory_clean(self, site):
        from fyst_trajectories.exceptions import AccelerationLimitWarning, VelocityLimitWarning

        with warnings.catch_warnings():
            warnings.simplefilter("error", AccelerationLimitWarning)
            warnings.simplefilter("error", VelocityLimitWarning)
            self._block(site, az_accel=1.0)

    def test_dynamics_check_runs_once_on_the_drifted_trajectory(self, site, monkeypatch):
        """One call, on the arrays the caller receives; the builder's own check is skipped.

        The drift is added after the base trajectory is built, so a check
        inside the builder would judge the wrong azimuth profile and a
        second check here would report every violation twice.
        """
        from fyst_trajectories.patterns import builder as builder_module

        kernel_calls: list[tuple] = []
        builder_calls: list[tuple] = []
        monkeypatch.setattr(
            _source_ces_module,
            "validate_trajectory_dynamics",
            lambda *args: kernel_calls.append(args),
        )
        monkeypatch.setattr(
            builder_module,
            "validate_trajectory_dynamics",
            lambda *args: builder_calls.append(args),
        )

        block = self._block(site, az_accel=1.0)

        assert builder_calls == []
        assert len(kernel_calls) == 1
        called_site, az, el, times = kernel_calls[0]
        assert called_site is site
        assert az is block.trajectory.az
        assert el is block.trajectory.el
        assert times is block.trajectory.times
        # The drifted azimuth is what was checked: it is not the constant
        # elevation pattern's undrifted profile.
        drift = block.computed_params["v_az"] * times
        assert np.abs(drift).max() > 0.01


class TestWindowOrdering:
    """A window that does not run forwards is refused as a bad argument.

    Unchecked, a reversed or zero-length window leaves the forward search
    grid empty, and the first numpy reduction over it raises a
    zero-size-reduction message from inside an error constructor, naming
    neither edge.
    """

    T0 = Time("2026-03-15T04:00:00", scale="utc")

    def test_zero_length_window_is_refused(self, site):
        """Equal edges raise a message quoting both times."""
        with pytest.raises(ValueError, match="window end .* must be later than window start"):
            plan_source_ces(
                body="jupiter", window=(self.T0, self.T0), el_bore=45.0, footprint="c", site=site
            )

    def test_reversed_window_is_refused(self, site):
        """A window given end-first raises the same message."""
        later = self.T0 + TimeDelta(3600.0, format="sec")
        with pytest.raises(ValueError, match="window end .* must be later than window start"):
            plan_source_ces(
                body="jupiter", window=(later, self.T0), el_bore=45.0, footprint="c", site=site
            )


def test_computed_params_record_utc(site):
    """``t0_iso``/``t1_iso`` are UTC, and re-parsing them as UTC round-trips.

    The recorded strings carry no scale, and every reader (the pass
    sequencer included) parses them as UTC, so the conversion has to be
    made where they are written.
    """
    block = plan_source_ces(
        body="jupiter",
        night=Time("2026-03-15T00:00:00", scale="utc"),
        mode="rising",
        el_bore=40.0,
        footprint="c",
        site=site,
    )
    params = block.computed_params
    t0 = Time(params["t0_iso"], scale="utc")
    t1 = Time(params["t1_iso"], scale="utc")
    assert t1 > t0
    # The trajectory carries the same instant as a real ``Time``; reading
    # the recorded string back as UTC has to land on it. A TAI-scaled
    # write would sit 37 s away.
    assert abs((t0 - block.trajectory.start_time).sec) < 1.0
