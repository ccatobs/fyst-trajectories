"""Tests for inject_retune() in event-list mode, RetuneEvent and sample_retune_events.

Covers the RetuneEvent dataclass, caller-supplied event lists (clipping,
out-of-bounds reporting, time offsets, adjacency and overlap, element types,
snapping that must not merge events), the exclusion of the uniform-cadence
kwargs, sample_retune_events, metadata preservation, and equivalence with the
uniform path.
"""

import warnings as _warnings

import numpy as np
import pytest
from _retune_stubs import _make_trajectory

from fyst_trajectories.exceptions import PointingWarning
from fyst_trajectories.retune import inject_retune, sample_retune_events
from fyst_trajectories.trajectory import (
    SCAN_FLAG_RETUNE,
    SCAN_FLAG_SCIENCE,
    SCAN_FLAG_TURNAROUND,
    RetuneEvent,
    Trajectory,
)


def test_event_clipped_to_end_flags_final_sample():
    """A retune event clipped to the trajectory end flags the final science sample."""
    times = np.arange(0.0, 11.0, 1.0)  # 0..10 s, 11 samples
    n = len(times)
    traj = Trajectory(
        times=times,
        az=np.zeros(n),
        el=np.full(n, 45.0),
        az_vel=np.zeros(n),
        el_vel=np.zeros(n),
    )
    # Event starts at t=8 and outlasts the trajectory, so it clips to t_end=10;
    # the sample at exactly t_end must still be flagged (an off-by-one drops it).
    result = inject_retune(traj, retune_events=[RetuneEvent(t_start=8.0, duration=10.0)])
    assert result.scan_flag[-1] == SCAN_FLAG_RETUNE
    flagged = result.scan_flag[times >= 8.0]
    assert (flagged == SCAN_FLAG_RETUNE).all()


class TestRetuneEventDataclass:
    """Validation rules for the RetuneEvent dataclass itself."""

    def test_event_list_negative_duration_raises(self):
        """RetuneEvent(duration=-1.0) is rejected at construction."""
        with pytest.raises(ValueError, match="duration must be positive"):
            RetuneEvent(t_start=10.0, duration=-1.0)

    def test_event_list_zero_duration_raises(self):
        """RetuneEvent(duration=0.0) is rejected at construction."""
        with pytest.raises(ValueError, match="duration must be positive"):
            RetuneEvent(t_start=10.0, duration=0.0)

    def test_event_list_negative_tstart_raises(self):
        """RetuneEvent(t_start=-1.0) is rejected at construction."""
        with pytest.raises(ValueError, match="t_start must be non-negative"):
            RetuneEvent(t_start=-1.0, duration=5.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_event_list_non_finite_tstart_raises(self, bad):
        """Non-finite t_start values are rejected."""
        with pytest.raises(ValueError, match="t_start must be finite"):
            RetuneEvent(t_start=bad, duration=5.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_event_list_non_finite_duration_raises(self, bad):
        """Non-finite duration values are rejected."""
        with pytest.raises(ValueError, match="duration must be positive"):
            RetuneEvent(t_start=10.0, duration=bad)


class TestInjectRetuneEventList:
    """Event-list mode: placement, clipping, overlap refusal, and sorting of the request."""

    def test_event_list_roundtrip(self):
        """Three explicit events each land on the timeline; retune_events field preserves them."""
        traj = _make_trajectory(duration=300.0, timestep=0.1)
        events = [
            RetuneEvent(t_start=30.0, duration=5.0),
            RetuneEvent(t_start=120.0, duration=3.0),
            RetuneEvent(t_start=200.0, duration=8.0),
        ]
        result = inject_retune(traj, retune_events=events)

        # The Trajectory.retune_events field carries the validated, sorted tuple verbatim.
        assert result.retune_events == tuple(events)

        # Each event's window marks some samples as RETUNE.
        retune_times = result.times[result.scan_flag == SCAN_FLAG_RETUNE]
        for ev in events:
            window_mask = (retune_times >= ev.t_start) & (retune_times < ev.t_start + ev.duration)
            assert window_mask.any(), f"Event at t_start={ev.t_start} produced no retune samples"

    def test_event_list_empty(self):
        """Empty event list is a no-op; scan_flag unchanged; retune_events is empty."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        original_flags = traj.scan_flag.copy()

        result = inject_retune(traj, retune_events=[])

        # scan_flag is copied but not mutated.
        np.testing.assert_array_equal(result.scan_flag, original_flags)
        assert result.retune_events == ()

    def test_event_at_trajectory_boundary_skipped(self):
        """Event with t_start == times[-1] is skipped with PointingWarning."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        # Trajectory-relative end is times[-1] - times[0]; for this fixture
        # times[0] = 0, so t_end_rel ~= 119.9.
        t_end_rel = float(traj.times[-1] - traj.times[0])
        events = [RetuneEvent(t_start=t_end_rel, duration=5.0)]

        with pytest.warns(PointingWarning, match="past trajectory end"):
            result = inject_retune(traj, retune_events=events)

        # No retune samples produced.
        assert not (result.scan_flag == SCAN_FLAG_RETUNE).any()
        # retune_events field still carries the event tuple (the request was recorded).
        assert result.retune_events == tuple(events)

    def test_event_clipped_at_trajectory_end(self):
        """An event whose duration overruns the end is clipped, not truncated."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        # Event starts at t=115 (well inside), lasts 20 s -> would end at 135.
        events = [RetuneEvent(t_start=115.0, duration=20.0)]

        # Must not warn: the event starts inside the trajectory.
        with _warnings.catch_warnings(record=True) as records:
            _warnings.simplefilter("always")
            result = inject_retune(traj, retune_events=events)

        skips = [
            r
            for r in records
            if issubclass(r.category, PointingWarning) and "past trajectory end" in str(r.message)
        ]
        assert not skips

        # At least some retune samples must be present, clipped but not empty.
        retune_mask = result.scan_flag == SCAN_FLAG_RETUNE
        assert retune_mask.any()
        # None may precede the event start; the final-sample flag is pinned by
        # test_event_clipped_to_end_flags_final_sample.
        retune_times = result.times[retune_mask]
        assert retune_times.min() >= 115.0 - 1e-6

    def test_event_overlap_same_module_raises(self):
        """Two overlapping events raise ValueError naming both indices."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        events = [
            RetuneEvent(t_start=30.0, duration=10.0),  # ends at 40
            RetuneEvent(t_start=35.0, duration=5.0),  # starts before previous ends
        ]
        with pytest.raises(ValueError, match="Overlapping retune events"):
            inject_retune(traj, retune_events=events)

    def test_event_in_turnaround_preserves_turnaround(self):
        """Event that falls entirely inside a turnaround window consumes no science."""
        # Turnaround at 29-45s. Event 30-40s sits inside.
        traj = _make_trajectory(
            duration=120.0,
            timestep=0.1,
            turnaround_intervals=[(29.0, 45.0)],
        )
        original_ta_count = int((traj.scan_flag == SCAN_FLAG_TURNAROUND).sum())
        events = [RetuneEvent(t_start=30.0, duration=10.0)]
        result = inject_retune(traj, retune_events=events)

        # No SCAN_FLAG_RETUNE samples: all would-be retune samples were
        # already SCAN_FLAG_TURNAROUND and remain so.
        assert not (result.scan_flag == SCAN_FLAG_RETUNE).any()
        # Turnaround count unchanged.
        new_ta_count = int((result.scan_flag == SCAN_FLAG_TURNAROUND).sum())
        assert new_ta_count == original_ta_count

    def test_event_list_with_prefer_turnarounds_snaps(self):
        """Event 1 s before a turnaround start snaps to the turnaround."""
        # Turnaround at 30-35s. Event due at 29 -> should snap to 30 with
        # window=5.0.
        traj = _make_trajectory(
            duration=120.0,
            timestep=0.1,
            turnaround_intervals=[(30.0, 35.0)],
        )
        events = [RetuneEvent(t_start=29.0, duration=5.0)]
        result = inject_retune(
            traj,
            retune_events=events,
            prefer_turnarounds=True,
            turnaround_window=5.0,
        )
        retune_times = result.times[result.scan_flag == SCAN_FLAG_RETUNE]
        # Snapped to 30 s, the 5 s event covers exactly the 30-35 s turnaround,
        # whose samples are never overwritten, so no RETUNE sample remains.
        # Unsnapped, the 29.0-29.9 s science samples would be flagged.
        assert not retune_times.size, (
            "Snapped event should overlap the turnaround and leave zero new retune samples."
        )

    def test_event_list_with_prefer_turnarounds_no_turnaround_nearby(self):
        """Event far from any turnaround uses caller's t_start verbatim."""
        # Turnaround at 10-12s. Event at 70s, window=5.0 should find
        # no turnaround nearby, so placement is literal.
        traj = _make_trajectory(
            duration=120.0,
            timestep=0.1,
            turnaround_intervals=[(10.0, 12.0)],
        )
        events = [RetuneEvent(t_start=70.0, duration=5.0)]
        result = inject_retune(
            traj,
            retune_events=events,
            prefer_turnarounds=True,
            turnaround_window=5.0,
        )
        retune_times = result.times[result.scan_flag == SCAN_FLAG_RETUNE]
        assert retune_times.size > 0
        # First retune sample should be at ~70s (not snapped to 10s).
        assert abs(float(retune_times[0]) - 70.0) < 0.15

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -5.0], ids=["nan", "inf", "negative"])
    def test_event_list_bad_turnaround_window_raises(self, bad):
        """The window is validated in event-list mode too, before any snapping."""
        traj = _make_trajectory(
            duration=120.0,
            timestep=0.1,
            turnaround_intervals=[(30.0, 35.0)],
        )
        events = [RetuneEvent(t_start=29.0, duration=5.0)]

        with pytest.raises(ValueError, match="turnaround_window must be finite and non-negative"):
            inject_retune(
                traj,
                retune_events=events,
                prefer_turnarounds=True,
                turnaround_window=bad,
            )

    def test_event_list_applied_in_sorted_order_regardless_of_input_order(self):
        """Out-of-order input is sorted before application and before metadata attach."""
        traj = _make_trajectory(duration=300.0, timestep=0.1)
        events_unsorted = [
            RetuneEvent(t_start=200.0, duration=5.0),
            RetuneEvent(t_start=30.0, duration=5.0),
            RetuneEvent(t_start=120.0, duration=5.0),
        ]
        result_unsorted = inject_retune(traj, retune_events=events_unsorted)

        # retune_events tuple is sorted.
        stored = result_unsorted.retune_events
        assert [e.t_start for e in stored] == sorted(e.t_start for e in events_unsorted)

        # scan_flag is identical to passing the sorted list directly.
        events_sorted = sorted(events_unsorted, key=lambda e: e.t_start)
        result_sorted = inject_retune(traj, retune_events=events_sorted)
        np.testing.assert_array_equal(result_unsorted.scan_flag, result_sorted.scan_flag)


class TestInjectRetuneEventListMutualExclusion:
    """Guardrails for mixing ``retune_events`` with uniform-cadence kwargs."""

    def test_both_uniform_and_events_supplied_raises_on_module_index(self):
        """retune_events + module_index=1 -> ValueError."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        events = [RetuneEvent(t_start=30.0, duration=5.0)]
        with pytest.raises(ValueError, match="module_index"):
            inject_retune(traj, retune_events=events, module_index=1, n_modules=7)

    def test_both_uniform_and_events_supplied_raises_on_n_modules(self):
        """retune_events + n_modules=7 -> ValueError."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        events = [RetuneEvent(t_start=30.0, duration=5.0)]
        with pytest.raises(ValueError, match="n_modules"):
            inject_retune(traj, retune_events=events, n_modules=7)

    def test_both_uniform_and_events_supplied_warns_on_custom_interval(self):
        """retune_events + retune_interval=100.0 -> PointingWarning; events applied."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        events = [RetuneEvent(t_start=30.0, duration=5.0)]
        with pytest.warns(PointingWarning, match="retune_interval is ignored"):
            result = inject_retune(traj, retune_events=events, retune_interval=100.0)
        # Events were still applied.
        assert (result.scan_flag == SCAN_FLAG_RETUNE).any()
        assert result.retune_events == tuple(events)

    def test_both_uniform_and_events_supplied_warns_on_custom_duration(self):
        """retune_events + retune_duration=10.0 -> PointingWarning; events applied."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        events = [RetuneEvent(t_start=30.0, duration=5.0)]
        with pytest.warns(PointingWarning, match="retune_duration is ignored"):
            result = inject_retune(traj, retune_events=events, retune_duration=10.0)
        assert (result.scan_flag == SCAN_FLAG_RETUNE).any()

    def test_retune_events_with_defaults_does_not_warn(self):
        """Silent when retune_events is supplied with defaults for the uniform kwargs."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        events = [RetuneEvent(t_start=30.0, duration=5.0)]
        with _warnings.catch_warnings(record=True) as records:
            _warnings.simplefilter("always")
            inject_retune(traj, retune_events=events)
        ignored = [
            r
            for r in records
            if issubclass(r.category, PointingWarning)
            and ("is ignored" in str(r.message) or "retune_interval" in str(r.message))
        ]
        assert not ignored


class TestSampleRetuneEvents:
    """Seeded sampling is reproducible, non-overlapping, and feeds ``inject_retune``."""

    def test_sample_retune_events_seeded_reproducible(self):
        """Same seed -> identical event list across two calls."""

        def _make_rng():
            return np.random.default_rng(seed=12345)

        def interval(r):
            return float(r.uniform(30.0, 90.0))

        def dur(r):
            return float(r.uniform(2.0, 8.0))

        first = sample_retune_events(
            duration=1000.0,
            interval_sampler=interval,
            duration_sampler=dur,
            rng=_make_rng(),
        )
        second = sample_retune_events(
            duration=1000.0,
            interval_sampler=interval,
            duration_sampler=dur,
            rng=_make_rng(),
        )

        assert first == second
        # Sanity: non-overlapping, chronological.
        for i in range(1, len(first)):
            assert first[i].t_start >= first[i - 1].t_start + first[i - 1].duration

    def test_sample_retune_events_negative_interval_raises(self):
        """Sampler returning negative interval -> ValueError naming the sampler."""
        rng = np.random.default_rng(seed=1)
        with pytest.raises(ValueError, match="interval_sampler"):
            sample_retune_events(
                duration=100.0,
                interval_sampler=lambda r: -5.0,
                duration_sampler=lambda r: 5.0,
                rng=rng,
            )

    def test_sample_retune_events_negative_duration_raises(self):
        """Sampler returning negative duration -> ValueError naming the sampler."""
        rng = np.random.default_rng(seed=1)
        with pytest.raises(ValueError, match="duration_sampler"):
            sample_retune_events(
                duration=100.0,
                interval_sampler=lambda r: 20.0,
                duration_sampler=lambda r: -3.0,
                rng=rng,
            )

    def test_sample_retune_events_zero_window_returns_empty(self):
        """Zero-duration window yields zero events without raising."""
        rng = np.random.default_rng(seed=1)
        events = sample_retune_events(
            duration=0.0,
            interval_sampler=lambda r: 10.0,
            duration_sampler=lambda r: 5.0,
            rng=rng,
        )
        assert events == []

    def test_sample_retune_events_feeds_inject_retune(self):
        """End-to-end: sampled events can drive inject_retune without error."""
        rng = np.random.default_rng(seed=7)
        events = sample_retune_events(
            duration=600.0,
            interval_sampler=lambda r: float(r.uniform(60.0, 120.0)),
            duration_sampler=lambda r: float(r.uniform(3.0, 6.0)),
            rng=rng,
        )
        traj = _make_trajectory(duration=600.0, timestep=0.1)
        result = inject_retune(traj, retune_events=events)
        assert (result.scan_flag == SCAN_FLAG_RETUNE).any()
        assert result.retune_events == tuple(events)


class TestInjectRetuneMetadataPreservation:
    """Regression: ``inject_retune`` must not mutate ``metadata``.

    Stashing ``retune_events`` inside ``trajectory.metadata`` as a dict key
    would silently break the ``Trajectory.pattern_type`` / ``.center_ra`` /
    ``.pattern_params`` accessors. ``retune_events`` is instead a first-class
    field on :class:`Trajectory`; ``metadata`` stays untouched.
    """

    def test_typed_metadata_survives_event_injection(self):
        """A ``TrajectoryMetadata`` instance round-trips through inject_retune."""
        from fyst_trajectories import TrajectoryMetadata

        meta = TrajectoryMetadata(
            pattern_type="pong",
            pattern_params={"width": 2.0, "height": 2.0},
            center_ra=180.0,
            center_dec=-30.0,
        )
        times = np.arange(0, 300.0, 0.1)
        n = len(times)
        traj = Trajectory(
            times=times,
            az=np.linspace(100, 200, n),
            el=np.full(n, 45.0),
            az_vel=np.gradient(np.linspace(100, 200, n), times),
            el_vel=np.zeros(n),
            metadata=meta,
            scan_flag=np.full(n, SCAN_FLAG_SCIENCE, dtype=np.int8),
        )

        events = [
            RetuneEvent(t_start=30.0, duration=5.0),
            RetuneEvent(t_start=150.0, duration=4.0),
        ]
        result = inject_retune(traj, retune_events=events)

        # Pattern accessors must still work.
        assert result.pattern_type == "pong"
        assert result.center_ra == 180.0
        assert result.center_dec == -30.0
        assert result.pattern_params == {"width": 2.0, "height": 2.0}

        # Metadata is the exact same object, we don't mutate it.
        assert result.metadata is traj.metadata

        # The first-class field carries the sorted event tuple.
        assert result.retune_events == tuple(events)


class TestInjectRetuneOutOfBoundsReporting:
    """Regression: OOB warning must name *sorted* indices, not input order."""

    def test_multiple_oob_events_single_warning_with_sorted_indices(self):
        """Three out-of-bounds events produce exactly one warning naming all three."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        t_end_rel = float(traj.times[-1] - traj.times[0])
        # Three events, all past the trajectory end. Input order differs from
        # sorted order so we can also verify the warning speaks in sorted terms.
        events = [
            RetuneEvent(t_start=t_end_rel + 200.0, duration=5.0),
            RetuneEvent(t_start=t_end_rel + 50.0, duration=5.0),
            RetuneEvent(t_start=t_end_rel + 100.0, duration=5.0),
        ]

        with pytest.warns(PointingWarning, match="past trajectory end") as record:
            result = inject_retune(traj, retune_events=events)

        # Exactly one warning captured.
        assert len(record) == 1
        msg = str(record[0].message)
        # All three sorted indices (0, 1, 2) should appear in the message.
        for idx in (0, 1, 2):
            assert f"sorted_index={idx}" in msg, f"expected sorted_index={idx} in {msg!r}"
        # The message must explicitly mention that the indices are sorted.
        assert "sorted" in msg.lower()
        # No retune samples produced.
        assert not (result.scan_flag == SCAN_FLAG_RETUNE).any()


class TestInjectRetuneTimesOffset:
    """Regression: ``times[0] != 0`` must map correctly in event-list mode."""

    def test_times_offset_event_placement(self):
        """Events are trajectory-relative; times[0] offset is respected."""
        times = np.arange(100.0, 200.0, 0.1)
        n = len(times)
        scan_flag = np.full(n, SCAN_FLAG_SCIENCE, dtype=np.int8)
        traj = Trajectory(
            times=times,
            az=np.linspace(0, 10, n),
            el=np.full(n, 45.0),
            az_vel=np.gradient(np.linspace(0, 10, n), times),
            el_vel=np.zeros(n),
            scan_flag=scan_flag,
        )

        events = [RetuneEvent(t_start=10.0, duration=5.0)]
        result = inject_retune(traj, retune_events=events)

        retune_times = result.times[result.scan_flag == SCAN_FLAG_RETUNE]
        assert retune_times.size > 0
        # Events are trajectory-relative, so t_start=10 maps to times[0] + 10 = 110.
        assert retune_times.min() >= 110.0 - 1e-6
        assert retune_times.max() < 115.0


class TestInjectRetuneUniformEquivalence:
    """Regression: event-list mode with uniform-matching events is identical."""

    def test_uniform_and_equivalent_event_list_produce_same_scan_flag(self):
        """The uniform path's own events, replayed through event-list mode, match."""
        traj = _make_trajectory(duration=300.0, timestep=0.1)

        # Uniform path.
        uniform_result = inject_retune(
            traj, retune_interval=60.0, retune_duration=5.0, prefer_turnarounds=False
        )

        # Feed the uniform path's own emitted events back through the
        # event-list path.
        uniform_events_for_event_mode = tuple(uniform_result.retune_events)
        event_result = inject_retune(traj, retune_events=list(uniform_events_for_event_mode))

        np.testing.assert_array_equal(uniform_result.scan_flag, event_result.scan_flag)
        assert uniform_result.retune_events == event_result.retune_events


class TestInjectRetuneAdjacentAndOverlap:
    """Regression: boundary-touching events OK; same-start-different-duration is overlap."""

    def test_adjacent_events_accepted(self):
        """Two events where ``a.t_start + a.duration == b.t_start`` must not raise."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        events = [
            RetuneEvent(t_start=30.0, duration=5.0),  # ends at 35
            RetuneEvent(t_start=35.0, duration=4.0),  # starts exactly where a ends
        ]
        # Must not raise.
        result = inject_retune(traj, retune_events=events)

        # Both events should produce retune samples.
        retune_times = result.times[result.scan_flag == SCAN_FLAG_RETUNE]
        first_window = retune_times[(retune_times >= 30.0) & (retune_times < 35.0)]
        second_window = retune_times[(retune_times >= 35.0) & (retune_times < 39.0)]
        assert first_window.size > 0, "First (pre-boundary) event produced no retune samples"
        assert second_window.size > 0, "Second (post-boundary) event produced no retune samples"

    def test_same_tstart_different_durations_overlap_raises(self):
        """Two events sharing ``t_start`` with different durations overlap."""
        traj = _make_trajectory(duration=120.0, timestep=0.1)
        events = [
            RetuneEvent(t_start=30.0, duration=5.0),
            RetuneEvent(t_start=30.0, duration=3.0),
        ]
        with pytest.raises(ValueError, match="Overlapping retune events"):
            inject_retune(traj, retune_events=events)


class TestEventListElementTypes:
    """Event-list mode says what it wanted when given something else.

    The sort runs first, so an unchecked tuple or dict dies as an
    ``AttributeError`` from inside the sort key, naming ``t_start`` and no
    argument.
    """

    @pytest.mark.parametrize("bad", [(10.0, 5.0), {"t_start": 10.0, "duration": 5.0}, 10.0, None])
    def test_non_event_element_is_refused(self, bad):
        """A non-``RetuneEvent`` element raises a TypeError naming its index."""
        traj = _make_trajectory(duration=120.0)
        events = [RetuneEvent(t_start=10.0, duration=5.0), bad]
        with pytest.raises(TypeError, match=r"retune_events\[1\] is a"):
            inject_retune(traj, retune_events=events)


class TestSnappingCannotSilentlyMergeEvents:
    """Overlap is re-checked against the placements, not only the request.

    Checking overlap before ``prefer_turnarounds`` snapping lets two events
    that do not overlap as asked be pulled onto the same turnaround: the
    second then paints samples the first already owns and disappears, with
    the caller told nothing.
    """

    def test_two_events_snapped_onto_one_turnaround_raise(self):
        """Both events land on the same turnaround and the merge is refused."""
        traj = _make_trajectory(duration=120.0, turnaround_intervals=[(30.0, 31.0), (60.0, 61.0)])
        # 28 s and 33 s both snap to the turnaround starting at 30 s within a
        # 10 s window, and a 5 s gap at 30 s runs into a second one placed there.
        events = [RetuneEvent(t_start=28.0, duration=5.0), RetuneEvent(t_start=33.0, duration=5.0)]
        with pytest.raises(ValueError, match="overlap after snapping to turnarounds"):
            inject_retune(
                traj, retune_events=events, prefer_turnarounds=True, turnaround_window=10.0
            )

    def test_events_snapping_to_different_turnarounds_are_fine(self):
        """Well-separated events still snap and apply."""
        traj = _make_trajectory(duration=120.0, turnaround_intervals=[(30.0, 31.0), (60.0, 61.0)])
        events = [RetuneEvent(t_start=28.0, duration=5.0), RetuneEvent(t_start=62.0, duration=5.0)]
        result = inject_retune(
            traj, retune_events=events, prefer_turnarounds=True, turnaround_window=10.0
        )
        retune_times = result.times[result.scan_flag == SCAN_FLAG_RETUNE]
        assert retune_times.min() == pytest.approx(31.0)  # snapped past the 28 s request
        assert result.retune_events == tuple(events)

    def test_without_snapping_the_same_events_apply(self):
        """The snapping check is snapping-specific; the request itself does not overlap."""
        traj = _make_trajectory(duration=120.0, turnaround_intervals=[(30.0, 31.0), (60.0, 61.0)])
        events = [RetuneEvent(t_start=28.0, duration=5.0), RetuneEvent(t_start=33.0, duration=5.0)]
        result = inject_retune(traj, retune_events=events)
        assert result.retune_events == tuple(events)
