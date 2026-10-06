"""Tests for the source-CES scan-geometry inputs.

Covers ``az_speed``, ``az_throw`` and ``dwell``, the refusal of non-scalar
times, the swept azimuth envelope the kernel screens, and the caller-side
margin transform.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest
from _source_ces_helpers import (
    _FULL_PRIMECAM_MODULES,
    _JUPITER_NIGHT,
    _JUPITER_RISING_ANCHOR,
    _full_primecam_block,
)
from astropy import units as u
from astropy.time import Time

import fyst_trajectories.planning.source_ces._kernel as _source_ces_kernel
from fyst_trajectories import (
    FYST_AZ_MAX_VELOCITY,
    MODULE_FOV_RADIUS_DEG,
    AzimuthBoundsError,
    DwellExceedsCrossingError,
    PointingError,
    PointingWarning,
    compute_source_ces_params,
    plan_source_ces,
    plan_source_ces_passes,
)
from fyst_trajectories.patterns.turnarounds import turnaround_overshoot_deg

# ---------------------------------------------------------------------------
# Scan-geometry inputs: az_speed, az_throw, dwell, and the margin transform
# ---------------------------------------------------------------------------


def _one_module_params(site, **overrides):
    """compute_source_ces_params for a single-module Jupiter-rising pass."""
    kwargs = dict(
        body="jupiter",
        footprint="c",
        el_bore=35.0,
        night=_JUPITER_NIGHT,
        mode="rising",
        site=site,
    )
    kwargs.update(overrides)
    return compute_source_ces_params(**kwargs)


class TestAzSpeedInput:
    """``az_speed`` replaces the derived slow-drag leg speed."""

    def test_default_records_the_derived_slow_drag_speed(self, site):
        cp = _one_module_params(site)
        # One module crosses in ~10 min at ~2.6 deg of throw: the derived
        # speed sits on the slow-drag floor, and it is recorded either way.
        assert cp["az_speed"] == pytest.approx(0.05)

    def test_explicit_speed_is_used_and_recorded(self, site):
        cp = _one_module_params(site, az_speed=1.5)
        assert cp["az_speed"] == 1.5
        block = _full_primecam_block(site, footprint="c", az_speed=1.5)
        assert block.computed_params["az_speed"] == 1.5
        assert block.config.az_speed == 1.5
        # The legs cruise at the requested speed plus the small drift.
        peak = float(np.abs(block.trajectory.az_vel).max())
        assert peak == pytest.approx(1.5 + abs(block.computed_params["v_az"]), abs=0.02)

    def test_speed_feeds_the_peak_speed_advisory(self, site):
        with pytest.warns(PointingWarning, match="exceeds site limit"):
            _one_module_params(site, az_speed=FYST_AZ_MAX_VELOCITY + 0.5)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_non_positive_speed_raises(self, site, bad):
        with pytest.raises(ValueError, match="az_speed must be positive"):
            _one_module_params(site, az_speed=bad)

    def test_fast_drag_at_hot_acceleration_warns_at_the_quintic_peak(self, site):
        """1.5 deg/s at 1.5 deg/s^2: the turnaround peaks at 2.25 deg/s^2, over the limit."""
        from fyst_trajectories.exceptions import AccelerationLimitWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            block = _full_primecam_block(
                site, footprint="c", az_speed=1.5, az_accel=1.5, az_throw=2.44
            )
        accel = [w for w in caught if issubclass(w.category, AccelerationLimitWarning)]
        assert len(accel) == 1
        traj = block.trajectory
        az_vel = np.gradient(np.unwrap(traj.az, period=360.0), traj.times)
        peak = float(np.abs(np.gradient(az_vel, traj.times)).max())
        assert peak == pytest.approx(2.25, abs=0.05)


class TestAzThrowInput:
    """``az_throw`` replaces the padded solved throw, re-centred on the window."""

    def test_explicit_throw_is_recorded_and_recentred(self, site):
        default = _one_module_params(site)
        cp = _one_module_params(site, az_throw=2.44)
        assert cp["az_throw"] == 2.44
        centre_default = default["az_start"] + 0.5 * default["az_throw"]
        centre = cp["az_start"] + 0.5 * cp["az_throw"]
        assert centre == pytest.approx(centre_default, abs=1e-9)
        # Timing and drift are untouched by the throw.
        assert cp["t0_iso"] == default["t0_iso"]
        assert cp["v_az"] == pytest.approx(default["v_az"])

    def test_throw_with_explicit_padding_raises(self, site):
        with pytest.raises(ValueError, match="cannot be combined with an explicit az_padding"):
            _one_module_params(site, az_throw=2.44, az_padding=0.2)

    def test_narrow_throw_warns(self, site):
        default = _one_module_params(site, az_padding=0.0)
        crossing = default["az_throw"]
        with pytest.warns(PointingWarning, match="narrower than the .* footprint crossing"):
            cp = _one_module_params(site, az_throw=0.5 * crossing)
        assert cp["az_throw"] == pytest.approx(0.5 * crossing)

    @pytest.mark.parametrize("bad", [0.0, -2.0])
    def test_non_positive_throw_raises(self, site, bad):
        with pytest.raises(ValueError, match="az_throw must be positive"):
            _one_module_params(site, az_throw=bad)

    def test_arc_sun_check_sees_the_overridden_envelope(self, site, monkeypatch):
        """The Sun sweep runs on the final swept window, so a narrower throw shrinks it."""
        seen: list[np.ndarray] = []
        real = _source_ces_kernel._check_arc_sun_safety

        def spy(coords, site_, arc_az, arc_el, arc_times, source_label, **kw):
            seen.append(np.asarray(arc_az))
            return real(coords, site_, arc_az, arc_el, arc_times, source_label, **kw)

        monkeypatch.setattr(_source_ces_kernel, "_check_arc_sun_safety", spy)
        default = _one_module_params(site)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PointingWarning)
            narrow = _one_module_params(site, az_throw=1.0)
        assert len(seen) == 2
        span_default = seen[0].max() - seen[0].min()
        span_narrow = seen[1].max() - seen[1].min()
        assert span_default - span_narrow == pytest.approx(default["az_throw"] - narrow["az_throw"])


class TestNonScalarTimeInputs:
    """A time grid where one instant belongs is refused, clearly.

    Unchecked, an array-valued ``Time`` fails far downstream: ``night`` as a
    numpy broadcast error, ``window`` as "the truth value of an array is
    ambiguous", and ``start_time`` worst of all, as a
    ``TargetNotObservableError`` naming a whole grid of times, which reads
    as an astronomy verdict rather than a malformed argument.
    """

    _GRID = Time(["2026-03-15T00:00:00", "2026-03-15T01:00:00"], scale="utc")

    def test_array_night_raises(self, site):
        with pytest.raises(ValueError, match="night must be a single instant"):
            plan_source_ces(
                body="jupiter",
                footprint="c",
                el_bore=35.0,
                site=site,
                night=self._GRID,
                mode="rising",
            )

    def test_array_window_edge_raises(self, site):
        with pytest.raises(ValueError, match="window start must be a single instant"):
            plan_source_ces(
                body="jupiter",
                footprint="c",
                el_bore=35.0,
                site=site,
                window=(self._GRID, self._GRID[0]),
            )

    def test_array_start_time_raises(self, site):
        with pytest.raises(ValueError, match="start_time must be a single instant"):
            plan_source_ces(body="jupiter", footprint="c", site=site, start_time=self._GRID)


class TestSweptEnvelope:
    """Every envelope the kernel reasons about covers the commanded motion.

    A constant-elevation sweep cruises across the science window and then
    overshoots each edge by the quintic turnaround peak
    ``5 * az_speed**2 / (8 * az_accel)`` before coming back. Two places
    could reason about the science window instead: the arc Sun check and the
    emit-time azimuth-bounds check. Both widen through the shared
    ``swept_az_envelope`` helper, so the screened range contains what the
    builder produces rather than sitting inside it.
    """

    def _captured_arc(self, site, monkeypatch, **overrides):
        """Plan a pass and return (arc azimuths seen by the Sun check, block)."""
        seen: list[np.ndarray] = []
        real = _source_ces_kernel._check_arc_sun_safety

        def spy(coords, site_, arc_az, arc_el, arc_times, label, **kw):
            seen.append(np.asarray(arc_az, dtype=float))
            return real(coords, site_, arc_az, arc_el, arc_times, label, **kw)

        monkeypatch.setattr(_source_ces_kernel, "_check_arc_sun_safety", spy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PointingWarning)
            block = _full_primecam_block(site, footprint="c", **overrides)
        assert len(seen) == 1
        return seen[0], block

    @pytest.mark.parametrize(
        ("az_speed", "az_accel"),
        [(1.5, 1.0), (1.5, 1.5), (2.0, 1.0)],
    )
    def test_arc_sun_check_covers_the_built_trajectory(self, site, monkeypatch, az_speed, az_accel):
        """The sampled arc contains the trajectory's own azimuth extremes.

        An arc that is only the science window misses the turnaround
        overshoot on each side, so a scan could put samples inside the
        exclusion zone with no warning at all.
        """
        arc_az, block = self._captured_arc(site, monkeypatch, az_speed=az_speed, az_accel=az_accel)
        assert arc_az.min() <= block.trajectory.az.min()
        assert arc_az.max() >= block.trajectory.az.max()

    def test_arc_widens_by_exactly_the_turnaround_overshoot(self, site, monkeypatch):
        """The widening is the shared constant, not an arbitrary pad."""
        arc_az, block = self._captured_arc(site, monkeypatch, az_speed=1.5, az_accel=1.0)
        cp = block.computed_params
        overshoot = turnaround_overshoot_deg(cp["az_speed"], 1.0)
        assert overshoot == pytest.approx(1.40625)
        # The arc spans the commanded window plus the drift the pass
        # accumulates between t0 and t1: throw + 2 * overshoot + |v_az| * dt.
        # The recorded window is an ISO string truncated to milliseconds, so
        # reconstructing the drift from it carries about a microdegree of
        # slack; that is far below the 1.4 deg the widening adds.
        window_sec = (Time(cp["t1_iso"]) - Time(cp["t0_iso"])).to_value(u.s)
        expected = cp["az_throw"] + 2 * overshoot + abs(cp["v_az"]) * window_sec
        assert arc_az.max() - arc_az.min() == pytest.approx(expected, abs=1e-4)

    @pytest.mark.parametrize("az_speed", [1.0, 2.0])
    def test_emit_time_envelope_covers_the_built_trajectory(self, site, az_speed):
        """The turnaround-overshoot envelope contains the built trajectory.

        This is the envelope ``compute_source_ces_params`` screens at emit
        time; the refusal itself is pinned by the test below.
        """
        limits = site.telescope_limits.azimuth
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PointingWarning)
            block = _full_primecam_block(site, footprint="c", az_speed=az_speed)
            cp = compute_source_ces_params(
                body="jupiter",
                footprint="c",
                el_bore=35.0,
                night=_JUPITER_NIGHT,
                mode="rising",
                site=site,
                az_speed=az_speed,
            )
        overshoot = turnaround_overshoot_deg(cp["az_speed"], 1.0)
        env_lo = min(cp["az_start"], cp["az_start"] + cp["az_throw"]) - overshoot
        env_hi = max(cp["az_start"], cp["az_start"] + cp["az_throw"]) + overshoot
        drift = cp["v_az"] * cp["duration"]
        env_lo += min(0.0, drift)
        env_hi += max(0.0, drift)
        assert env_lo <= block.trajectory.az.min()
        assert env_hi >= block.trajectory.az.max()
        # Sanity: this geometry is comfortably inside the limits, so the
        # widening is what is being pinned, not a bounds refusal.
        assert limits.min < env_lo and env_hi < limits.max

    def test_emit_time_envelope_refuses_a_trajectory_that_leaves_the_limits(self, site):
        """A sweep whose overshoot crosses the limit is refused at emit time.

        Built by squeezing the azimuth limits around the solved window so
        only the turnaround overshoot pushes past them: without the widening
        the same call returns a params dict a dispatcher would then reject.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PointingWarning)
            cp = _one_module_params(site, az_speed=1.5)
        overshoot = turnaround_overshoot_deg(cp["az_speed"], 1.0)
        drift = cp["v_az"] * cp["duration"]
        sci_hi = max(cp["az_start"], cp["az_start"] + cp["az_throw"]) + max(0.0, drift)
        # A limit between the science edge and the commanded edge.
        tight = dataclasses.replace(
            site,
            telescope_limits=dataclasses.replace(
                site.telescope_limits,
                azimuth=dataclasses.replace(
                    site.telescope_limits.azimuth, max=sci_hi + 0.5 * overshoot
                ),
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PointingWarning)
            with pytest.raises(AzimuthBoundsError):
                _one_module_params(tight, az_speed=1.5)


class TestDwellInput:
    """``dwell`` narrows the pass about the crossing midpoint."""

    def test_default_records_the_full_crossing(self, site):
        cp = _one_module_params(site)
        span = (Time(cp["t1_iso"]) - Time(cp["t0_iso"])).to_value(u.s)
        assert cp["crossing_seconds"] == pytest.approx(span, abs=1e-3)

    def test_dwell_narrows_symmetrically_and_keeps_the_crossing(self, site):
        default = _one_module_params(site)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PointingWarning)
            cp = _one_module_params(site, dwell=300.0)
        t0, t1 = Time(cp["t0_iso"]), Time(cp["t1_iso"])
        assert (t1 - t0).to_value(u.s) == pytest.approx(300.0, abs=1e-3)
        mid_default = Time(default["t0_iso"]) + 0.5 * (
            Time(default["t1_iso"]) - Time(default["t0_iso"])
        )
        mid = t0 + 0.5 * (t1 - t0)
        assert abs((mid - mid_default).to_value(u.s)) < 1e-3
        assert cp["crossing_seconds"] == pytest.approx(default["crossing_seconds"])
        assert cp["crossing_seconds"] > 300.0
        # The quantised trajectory length tracks the dwell, not the crossing.
        # The bound is derived, not picked: rounding the window to whole legs
        # can land at most half a leg-plus-turnaround either side of it. A
        # flat 60 s slop would be 2.3x that and would pass a floor that
        # silently widened the pass.
        half_cycle = 0.5 * (cp["az_throw"] / cp["az_speed"] + 2.0 * cp["az_speed"] / 1.0)
        assert abs(cp["duration"] - 300.0) <= half_cycle

    def test_partial_dwell_warns(self, site):
        with pytest.warns(PointingWarning, match="shorter than the .* footprint crossing"):
            _one_module_params(site, dwell=300.0)

    def test_dwell_longer_than_the_crossing_raises(self, site):
        """The refusal is typed and carries both numbers, so a caller needs no message match."""
        crossing = _one_module_params(site)["crossing_seconds"]
        with pytest.raises(
            DwellExceedsCrossingError, match="dwell must not exceed the solved footprint crossing"
        ) as exc_info:
            _one_module_params(site, dwell=crossing + 60.0)
        exc = exc_info.value
        assert isinstance(exc, PointingError)
        assert exc.crossing_seconds == crossing
        assert exc.dwell == crossing + 60.0

    @pytest.mark.parametrize("bad", [0.0, -30.0])
    def test_non_positive_dwell_raises(self, site, bad):
        with pytest.raises(ValueError, match="dwell must be positive"):
            _one_module_params(site, dwell=bad)

    def test_dwell_below_the_sampling_step_raises_naming_both(self, site):
        """A dwell shorter than one sample is refused, not silently widened.

        The narrowed window is resampled on the ``sampling_step_seconds``
        grid, so a shorter dwell would be floored back up to one step: a
        dwell of 8 s would build a 30 s scan while ``t0_iso``/``t1_iso`` still
        reported 8 s, and changing only the sampling step would change the
        scan duration by 3.75x. The message names both numbers so the caller
        can see which knob to move.
        """
        with pytest.raises(ValueError, match=r"dwell must be at least sampling_step_seconds"):
            _one_module_params(site, dwell=8.0)

        with pytest.raises(ValueError) as excinfo:
            _one_module_params(site, dwell=8.0, sampling_step_seconds=30.0)
        assert "dwell=8.0" in str(excinfo.value)
        assert "sampling_step_seconds=30.0" in str(excinfo.value)

    def test_dwell_at_the_sampling_step_is_honoured_exactly(self, site):
        """At the floor the pass is exactly the requested length, not one step wider."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PointingWarning)
            cp = _one_module_params(site, dwell=10.0, sampling_step_seconds=10.0)
        assert (Time(cp["t1_iso"]) - Time(cp["t0_iso"])).to_value(u.s) == pytest.approx(
            10.0, abs=1e-3
        )

    def test_dwell_with_multiple_passes_raises(self, site):
        with pytest.raises(ValueError, match="accepted only for a single pass"):
            plan_source_ces_passes(
                body="jupiter",
                footprint="c",
                el_bore=35.0,
                n_passes=2,
                night=_JUPITER_NIGHT,
                mode="rising",
                site=site,
                dwell=120.0,
            )

    def test_single_pass_sequence_accepts_dwell(self, site):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PointingWarning)
            (block,) = plan_source_ces_passes(
                body="jupiter",
                footprint="c",
                el_bore=35.0,
                n_passes=1,
                night=_JUPITER_NIGHT,
                mode="rising",
                site=site,
                dwell=300.0,
            )
        cp = block.computed_params
        assert (Time(cp["t1_iso"]) - Time(cp["t0_iso"])).to_value(u.s) == pytest.approx(
            300.0, abs=1e-3
        )

    def test_anchored_dwell_pass_starts_half_the_cut_after_the_anchor(self, site):
        """The anchor places the full crossing; the dwell narrows about its midpoint."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PointingWarning)
            full = plan_source_ces(
                body="jupiter", footprint="c", start_time=_JUPITER_RISING_ANCHOR, site=site
            )
            narrowed = plan_source_ces(
                body="jupiter",
                footprint="c",
                start_time=_JUPITER_RISING_ANCHOR,
                site=site,
                dwell=300.0,
            )
        assert narrowed.computed_params["el_bore"] == pytest.approx(full.computed_params["el_bore"])
        crossing = full.computed_params["crossing_seconds"]
        expected_delay = 0.5 * (crossing - 300.0)
        delay = (narrowed.trajectory.start_time - full.trajectory.start_time).to_value(u.s)
        assert delay == pytest.approx(expected_delay, abs=1.0)
        cp = narrowed.computed_params
        assert (Time(cp["t1_iso"]) - Time(cp["t0_iso"])).to_value(u.s) == pytest.approx(
            300.0, abs=1e-3
        )


class TestInflateFootprint:
    """The caller-side margin transform pushes the cover out from the centre."""

    def test_single_module_circle_grows_by_the_margin(self):
        from fyst_trajectories.planning.footprints import inflate_footprint, resolve_footprint

        base = resolve_footprint("c")
        grown = inflate_footprint(base, 0.4)
        r_base = np.hypot(
            base.cover_xi_deg - base.center_xi_deg, base.cover_eta_deg - base.center_eta_deg
        )
        r_grown = np.hypot(
            grown.cover_xi_deg - grown.center_xi_deg, grown.cover_eta_deg - grown.center_eta_deg
        )
        np.testing.assert_allclose(r_grown, r_base + 0.4)
        assert grown.center_xi_deg == base.center_xi_deg
        assert grown.center_eta_deg == base.center_eta_deg
        assert r_base.max() == pytest.approx(MODULE_FOV_RADIUS_DEG)

    def test_off_centre_module_keeps_its_own_centre(self):
        from fyst_trajectories.planning.footprints import inflate_footprint, resolve_footprint

        base = resolve_footprint("i1")
        grown = inflate_footprint(base, 0.25)
        assert (grown.center_xi_deg, grown.center_eta_deg) == (
            base.center_xi_deg,
            base.center_eta_deg,
        )
        r = np.hypot(
            grown.cover_xi_deg - base.center_xi_deg, grown.cover_eta_deg - base.center_eta_deg
        )
        np.testing.assert_allclose(r, MODULE_FOV_RADIUS_DEG + 0.25)

    def test_zero_margin_is_an_equal_copy(self):
        from fyst_trajectories.planning.footprints import inflate_footprint, resolve_footprint

        base = resolve_footprint(_FULL_PRIMECAM_MODULES)
        same = inflate_footprint(base, 0.0)
        np.testing.assert_allclose(same.cover_xi_deg, base.cover_xi_deg)
        np.testing.assert_allclose(same.cover_eta_deg, base.cover_eta_deg)
        assert same is not base

    def test_negative_margin_raises(self):
        from fyst_trajectories.planning.footprints import inflate_footprint, resolve_footprint

        with pytest.raises(ValueError, match="margin_deg must be non-negative"):
            inflate_footprint(resolve_footprint("c"), -0.1)

    def test_margin_lengthens_the_crossing(self, site):
        """One module plus 0.4 deg per side reproduces a wider, longer pass."""
        from fyst_trajectories.planning.footprints import inflate_footprint, resolve_footprint

        plain = _one_module_params(site, az_padding=0.0)
        wide = _one_module_params(
            site, footprint=inflate_footprint(resolve_footprint("c"), 0.4), az_padding=0.0
        )
        assert wide["crossing_seconds"] > plain["crossing_seconds"]
        # The on-sky throw grows by twice the margin, up to the drift term.
        grown = (wide["az_throw"] - plain["az_throw"]) * np.cos(np.radians(35.0))
        assert grown == pytest.approx(0.8, abs=0.1)
