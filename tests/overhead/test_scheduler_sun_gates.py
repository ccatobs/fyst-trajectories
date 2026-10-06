"""Sun-gate tests for the offline scheduler: the Sun-checked slew and the escape."""

import pytest
from _scheduler_helpers import _initial_state, _make_ctx
from _sun_stubs import pose_blocker
from astropy.time import Time, TimeDelta

from fyst_trajectories import get_fyst_site
from fyst_trajectories.overhead import ElevationConstraint, ObservingPatch
from fyst_trajectories.overhead.scheduler import PatchSelectionPhase, PhaseResult, SlewPhase
from fyst_trajectories.overhead.scheduler.helpers import _compute_az_range


def _sky_band_blocker(lo: float, hi: float, until: Time | None = None):
    """Build a point predicate that is unsafe inside a sky-azimuth band.

    The band is open, in degrees on the sky (any encoder wrap maps into
    it); with ``until`` the band clears at that time, so a refusal can be
    retried against a Sun that has moved on.
    """

    def predicate(az, el, t):
        if until is not None and t >= until:
            return True
        return not (lo < float(az) % 360.0 < hi)

    return predicate


class TestSlewTransition:
    """The slew is Sun-checked; a refusal idles the tick instead of raising."""

    def _pong_patch(self, **overrides):
        params = dict(
            name="field",
            ra_center=24.0,
            dec_center=-32.0,
            width=10.0,
            height=10.0,
            scan_type="pong",
            velocity=1.0,
        )
        params.update(overrides)
        return ObservingPatch(**params)

    def test_blocked_path_idles_with_the_reason(self):
        """A blocked direct path emits one idle tick, labelled, at the unmoved pose."""
        patch = self._pong_patch()
        # Sky azimuth 115 has a single encoder image, so the blocked band
        # between the bootstrap pose (180) and the goal leaves no wrap.
        ctx = _make_ctx(patches=[patch], sun_safe=_sky_band_blocker(140.0, 160.0))
        state = _initial_state(ctx)
        sel = PhaseResult(state=state, blocks=[], selection=patch, best_az=115.0, best_el=50.0)

        result = SlewPhase().run(state, ctx, selection=sel)

        assert result.skip_to_next_iter is True
        assert result.stop is False
        assert result.selection is None
        (block,) = result.blocks
        assert str(block.block_type) == "idle"
        assert block.metadata["reason"] == "sun_path"
        assert block.duration == pytest.approx(ctx.time_step)
        assert block.az_start == state.current_az and block.elevation == state.current_el
        assert result.state.current_az == state.current_az
        assert result.state.current_time.unix == pytest.approx(
            state.current_time.unix + ctx.time_step
        )

    def test_sun_blocked_goal_idles_as_sun_point(self):
        """A goal inside the zone in every wrap is refused as ``sun_point``."""
        patch = self._pong_patch()
        ctx = _make_ctx(patches=[patch], sun_safe=_sky_band_blocker(100.0, 130.0))
        state = _initial_state(ctx)
        sel = PhaseResult(state=state, blocks=[], selection=patch, best_az=115.0, best_el=50.0)

        result = SlewPhase().run(state, ctx, selection=sel)

        assert result.blocks[0].metadata["reason"] == "sun_point"
        assert result.skip_to_next_iter is True

    def test_selection_idle_carries_no_reason(self):
        """The nothing-observable idle keeps its bare metadata."""
        ctx = _make_ctx(patches=[])
        state = _initial_state(ctx)

        result = PatchSelectionPhase().run(state, ctx)

        assert result.blocks[0].metadata == {}

    def test_far_wrap_carries_the_science_frame(self):
        """When only the far wrap has a clear path, best_az follows it.

        Sky azimuth 200 has encoder images 200 and -160. The band blocks
        the short path from 180 to 200 but not the long way round, so the
        transition lands on -160 and the science range downstream must be
        derived on that wrap, not a full turn away from the telescope. The
        schedule starts at 09:00 UTC, when the field is up with room for a
        whole visit: the slew is booked only if the visit's shortest
        subscan still fits on arrival.
        """
        patch = self._pong_patch()
        ctx = _make_ctx(
            patches=[patch],
            sun_safe=_sky_band_blocker(185.0, 195.0),
            start_time="2026-06-15T09:00:00",
        )
        state = _initial_state(ctx)
        sel = PhaseResult(state=state, blocks=[], selection=patch, best_az=200.0, best_el=50.0)

        result = SlewPhase().run(state, ctx, selection=sel)

        (block,) = result.blocks
        assert str(block.block_type) == "slew"
        assert block.az_end == pytest.approx(-160.0)
        assert result.state.current_az == pytest.approx(-160.0)
        assert result.best_az == pytest.approx(-160.0)
        lo, hi = _compute_az_range(patch, result.best_az, result.best_el, ctx.site)
        assert lo <= result.state.current_az <= hi
        # The long way round is priced as such.
        assert block.duration > 100.0

    def test_explicit_window_cannot_follow_the_far_wrap(self):
        """An explicit window pins the range, so a far-wrap-only slew idles."""
        patch = self._pong_patch(
            scan_type="constant_el",
            elevation=50.0,
            scan_params={"az_min": 200.0, "az_max": 220.0},
        )
        ctx = _make_ctx(patches=[patch], sun_safe=_sky_band_blocker(185.0, 195.0))
        state = _initial_state(ctx)
        sel = PhaseResult(state=state, blocks=[], selection=patch, best_az=210.0, best_el=50.0)

        result = SlewPhase().run(state, ctx, selection=sel)

        assert result.skip_to_next_iter is True
        assert result.blocks[0].metadata["reason"] == "no_wrap"
        assert result.state.current_az == state.current_az

    def test_scheduler_retries_once_the_path_clears(self):
        """The loop idles on a refusal and slews once the Sun has moved.

        The band between the bootstrap pose and the field clears at a
        fixed time; every refusal must precede it, each labelled and at
        the unmoved pose, and the first slew and the science must follow.
        """
        from fyst_trajectories.overhead import generate_timeline

        patch = self._pong_patch()
        clear_at = Time("2026-06-15T08:35:00", scale="utc")
        timeline = generate_timeline(
            patches=[patch],
            site=get_fyst_site(),
            start_time="2026-06-15T07:50:00",
            end_time="2026-06-15T09:30:00",
            sun_safe=_sky_band_blocker(140.0, 160.0, until=clear_at),
        )

        refused = [b for b in timeline.blocks if b.metadata.get("reason") == "sun_path"]
        assert len(refused) >= 2
        assert all(b.t_start.unix < clear_at.unix for b in refused)
        assert all(b.az_start == 180.0 and str(b.block_type) == "idle" for b in refused)
        slews = [b for b in timeline.blocks if str(b.block_type) == "slew"]
        science = [b for b in timeline.blocks if str(b.block_type) == "science"]
        assert slews and science
        assert slews[0].t_start.unix >= clear_at.unix
        assert science[0].t_start.unix > slews[0].t_start.unix
        assert timeline.validate() == []


class TestSunEscape:
    """A pose the zone has overtaken is moved out before the telescope idles or slews."""

    def _pong_patch(self):
        return ObservingPatch(
            name="field",
            ra_center=24.0,
            dec_center=-32.0,
            width=10.0,
            height=10.0,
            scan_type="pong",
            velocity=1.0,
        )

    def test_overtaken_pose_escapes_before_the_slew(self):
        """The bootstrap pose is inside the zone: the slew phase escapes and restarts."""
        patch = self._pong_patch()
        ctx = _make_ctx(patches=[patch], sun_safe=pose_blocker(180.0, 50.0))
        state = _initial_state(ctx)
        sel = PhaseResult(state=state, blocks=[], selection=patch, best_az=115.0, best_el=50.0)

        result = SlewPhase().run(state, ctx, selection=sel)

        (block,) = result.blocks
        assert str(block.block_type) == "slew" and block.patch_name == "sun_escape"
        assert block.az_start == 180.0
        assert result.skip_to_next_iter is True and result.selection is None
        assert (result.state.current_az, result.state.current_el) == (block.az_end, block.elevation)
        moved_az = abs(result.state.current_az - 180.0) >= 5.0
        moved_el = abs(result.state.current_el - 50.0) >= 5.0
        assert moved_az or moved_el
        arrival = state.current_time.unix + block.duration
        assert result.state.current_time.unix == pytest.approx(arrival)

    def test_idle_path_escapes_instead_of_parking_in_the_zone(self):
        """With nothing observable, the selection idle still moves an overtaken pose."""
        ctx = _make_ctx(patches=[], sun_safe=pose_blocker(180.0, 50.0))
        state = _initial_state(ctx)

        result = PatchSelectionPhase().run(state, ctx)

        (block,) = result.blocks
        assert block.patch_name == "sun_escape"
        assert result.skip_to_next_iter is True
        # The next tick idles at the escape pose, with no reason label.
        again = PatchSelectionPhase().run(result.state, ctx)
        assert str(again.blocks[0].block_type) == "idle"
        assert again.blocks[0].az_start == result.state.current_az
        assert again.blocks[0].metadata == {}

    def test_escape_searches_down_to_the_schedule_elevation_floor(self):
        """The escape stops at the observing floor, not at the mount limit.

        Only sky at or below 35 deg is clear here, and the schedule's
        elevation constraint floors it at 40, so there is nowhere to go
        and the tick idles labelled ``no_escape``. Searching down to the
        mount limit (20 deg at FYST) instead would park the telescope
        5 deg below the sky the selection phase is willing to use.
        """
        ctx = _make_ctx(
            patches=[],
            sun_safe=lambda az, el, t: float(el) <= 35.0,
            constraints=[ElevationConstraint(el_min=40.0, el_max=90.0)],
        )
        assert ctx.el_floor == 40.0
        assert ctx.slew_safe is not None
        state = _initial_state(ctx)

        result = PatchSelectionPhase().run(state, ctx)

        (block,) = result.blocks
        assert str(block.block_type) == "idle"
        assert block.metadata["reason"] == "no_escape"
        assert (result.state.current_az, result.state.current_el) == (
            state.current_az,
            state.current_el,
        )

    def test_trapped_pose_idles_with_no_escape(self):
        ctx = _make_ctx(patches=[], sun_safe=lambda az, el, t: False)
        state = _initial_state(ctx)

        result = PatchSelectionPhase().run(state, ctx)

        (block,) = result.blocks
        assert str(block.block_type) == "idle"
        assert block.metadata["reason"] == "no_escape"
        assert result.state.current_az == state.current_az

    def test_escape_past_the_window_end_stops_the_loop(self):
        patch = self._pong_patch()
        ctx = _make_ctx(
            patches=[patch],
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T02:00:10",
            sun_safe=pose_blocker(180.0, 50.0),
        )
        state = _initial_state(ctx)
        sel = PhaseResult(state=state, blocks=[], selection=patch, best_az=115.0, best_el=50.0)

        result = SlewPhase().run(state, ctx, selection=sel)

        assert result.stop is True and result.blocks == []

    def test_escape_past_the_window_end_on_the_idle_path_stops_cleanly(self):
        """The idle path's escape can end past the window; the loop must stop, not raise.

        The pose is safe when the loop-top check runs, the opening
        calibrations then consume the window down to a sliver, and the
        selection idle finds the pose overtaken with too little time for
        the escape. That result carries ``stop`` and no selection, which
        the loop must not hand on to the slew phase.
        """
        from fyst_trajectories.overhead import generate_timeline

        start = Time("2026-06-15T02:00:00", scale="utc")
        overtaken_at = start + TimeDelta(1000.0, format="sec")

        def sun_safe(az, el, t):
            if t < overtaken_at:
                return True
            return not (abs(float(az) - 180.0) < 5.0 and abs(float(el) - 50.0) < 5.0)

        # The four opening calibrations take 1080 s; 20 s then remain,
        # less than the escape's 35 s.
        timeline = generate_timeline(
            patches=[],
            site=get_fyst_site(),
            start_time=start.isot,
            end_time=(start + TimeDelta(1100.0, format="sec")).isot,
            sun_safe=sun_safe,
        )

        assert timeline.blocks
        assert all(b.t_stop.unix <= timeline.end_time.unix + 1e-6 for b in timeline.blocks)
        assert not any(b.patch_name == "sun_escape" for b in timeline.blocks)
        assert timeline.validate() == []

    def test_slew_is_refused_when_the_goal_will_not_hold_for_a_tick(self):
        """A pose the next tick would escape from is not slewed to at all.

        Without the hold the wrap choice sees only the slew instant, so
        the loop would command a pose, find it overtaken one tick later
        and escape again: three round trips and no science. The slew
        phase asks for one scheduler tick of dwell instead, and refuses.
        """
        patch = self._pong_patch()
        start = Time("2026-06-15T02:00:00", scale="utc")
        closes_at = start + TimeDelta(120.0, format="sec")

        def sun_safe(az, el, t):
            # The parked pose stays clear; the science pose closes after
            # two minutes, well inside the 300 s tick.
            if abs(float(az) - 180.0) < 5.0 and abs(float(el) - 50.0) < 5.0:
                return True
            return bool(t < closes_at)

        ctx = _make_ctx(patches=[patch], sun_safe=sun_safe, time_step=300.0)
        state = _initial_state(ctx)
        sel = PhaseResult(state=state, blocks=[], selection=patch, best_az=115.0, best_el=50.0)

        result = SlewPhase().run(state, ctx, selection=sel)

        (block,) = result.blocks
        assert str(block.block_type) == "idle"
        assert block.metadata["reason"] == "sun_point"
        assert result.skip_to_next_iter is True
        assert (result.state.current_az, result.state.current_el) == (180.0, 50.0)

    def test_one_pose_and_time_is_searched_once(self, monkeypatch):
        """One (pose, time) question is searched at most once.

        The loop top, the idle emitter and the slew phase all have to ask,
        because the calibration phase can advance the clock inside a tick;
        the context's memo makes the repeats free. Nothing is observable on
        this night, so the loop top and the idle emitter are the sites that
        ask here, and every search the planner sees must be a new question.
        """
        from fyst_trajectories.overhead import _moves, generate_timeline

        seen = []
        original = _moves.plan_escape

        def counting(az, el, t, site, **kwargs):
            seen.append((round(float(az), 6), round(float(el), 6), round(t.unix, 6)))
            return original(az, el, t, site, **kwargs)

        monkeypatch.setattr(_moves, "plan_escape", counting)

        timeline = generate_timeline(
            patches=[self._pong_patch()],
            site=get_fyst_site(),
            start_time="2026-06-15T02:00:00",
            end_time="2026-06-15T04:00:00",
        )

        assert timeline.blocks
        assert seen  # the check runs at all
        assert len(seen) == len(set(seen))

    def test_night_escapes_once_then_observes(self):
        """The loop escapes the overtaken bootstrap pose and the night proceeds normally."""
        from fyst_trajectories.overhead import generate_timeline

        patch = self._pong_patch()
        clear_at = Time("2026-06-15T08:35:00", scale="utc")
        timeline = generate_timeline(
            patches=[patch],
            site=get_fyst_site(),
            start_time="2026-06-15T07:50:00",
            end_time="2026-06-15T09:30:00",
            sun_safe=pose_blocker(180.0, 50.0, until=clear_at),
        )

        escapes = [b for b in timeline.blocks if b.patch_name == "sun_escape"]
        assert len(escapes) == 1
        assert escapes[0].t_start.unix == timeline.start_time.unix
        assert not any(b.metadata.get("reason") for b in timeline.blocks)
        kinds = [str(b.block_type) for b in timeline.blocks]
        assert "science" in kinds
        assert timeline.validate() == []
