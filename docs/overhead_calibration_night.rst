Calibration Nights
==================

:func:`~fyst_trajectories.overhead.plan_calibration_night` lays out one
night of solar-system calibration passes back to back: it walks the
night, picks whichever planet is up next, plans each visit with the
source-CES kernel (see :doc:`planning`) on a per-body scan table, checks
the slew between visits for Sun safety, reserves the detector operations
before each visit, and returns an ordinary
:class:`~fyst_trajectories.overhead.ObservingTimeline`. It is an offline
planning tool for commissioning nights, a sibling of the survey-night
simulator :func:`~fyst_trajectories.overhead.generate_timeline` that
shares its block model and outputs but walks a body queue instead of a
cadence loop.

Quick Start
-----------

Plan the first hour of a night and read it back::

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.overhead import (
        dispatch_sheet,
        plan_calibration_night,
        summarize_calibration_night,
    )

    site = get_fyst_site()
    timeline = plan_calibration_night(
        ["saturn", "uranus"],          # priority order
        site,
        "2026-09-11T06:30:00",         # the window is clipped to when the Sun is down
        "2026-09-11T07:15:00",
    )

    summary = summarize_calibration_night(timeline)
    print(summary.bodies[0].passes, summary.bodies[1].passes)
    # 2 0

    row = next(line for line in dispatch_sheet(timeline).splitlines() if "source_scan" in line)
    print(row[:53])
    #      6  2026-09-11 06:46:28     12.3 min  source_scan

The summary carries each body's visits, passes, minutes on source, duty
cycle and geometry; the sheet has one numbered row per block, and the
pass rows carry the literal ``scan_params`` dict for the execution
layer's source-scan task. Beside the base keys that dict carries the
geometry the visit asked for: ``az_speed`` and ``az_accel`` always, and
``az_throw``, ``dwell`` or ``footprint_margin`` when the table, the
policy or a per-visit override sets them. Confirm the receiving task
forwards those keys before dispatching, because a task that reads only
the base keys drops the rest without a message and runs a different
scan. The sheet repeats that check on every affected row. Both views are
rendered from the timeline's blocks and metadata, so a sheet printed
from a timeline read back from ECSV is byte-identical.

Scan Tables and Policy
----------------------

Each body's scan geometry comes from an elevation-binned table: the
azimuth throw to sweep in each bin, plus a reference dwell that the planner
shows beside its own solved crossing time and applies only when asked. The
shipped tables are instrument-team commissioning defaults; a body without
its own table uses ``"default"``, and above a table's top bin the throw is
extrapolated at constant on-sky width::

    from fyst_trajectories.overhead import DEFAULT_SCAN_TABLES

    shared = DEFAULT_SCAN_TABLES["default"]
    print(shared.el_range, shared.bins[0].az_throw, shared.bins[0].dwell_reference)
    # (30.0, 50.0) 2.44 600.0
    print(f"{shared.az_throw_at(62.0):.2f}")
    # 4.48

Your own table replaces the defaults without a release. The loader reads a
CSV with a body column, the bin bounds, the scan time in minutes and the
scan width in degrees of azimuth::

    from pathlib import Path

    from fyst_trajectories.overhead import load_scan_tables

    Path("tables.csv").write_text(
        "body,el_lo,el_hi,scan_time_min,az_throw\n"
        "default,30,40,10,2.5\n"
        "default,40,50,13,2.9\n"
        "uranus,30,40,15,2.5\n"
    )
    tables = load_scan_tables("tables.csv")
    print(sorted(tables), tables["uranus"].el_range)
    # ['default', 'uranus'] (30.0, 40.0)

The throw is one of three kernel inputs the planner sets per pass; the
sweep speed and acceleration come from the
:class:`~fyst_trajectories.overhead.CalibrationNightPolicy` (1.5 deg/s
and 1.5 deg/s^2 by default), and the dwell is solved from the footprint
crossing unless the policy applies the table's reference or a visit
overrides it. The policy also holds the observing floor (30 deg), the
longest crossing accepted (20 min), the footprint and the on-sky margin
around it (``footprint_margin``, which a rebuild has to re-apply to land
on the same crossing), the idle tick and the retry interval, and the tuning policy
for the detector operations. The quintic turnaround peaks at 1.5 times
the nominal acceleration, so the default 1.5 deg/s^2 reaches an analytic
2.25 deg/s^2 against the site's 1.5 deg/s^2 advisory ceiling; the planner
records the warning on every pass rather than raising it, and the summary
lists the slightly lower peak the sampled trajectory measures::

    print(next(w for w in summary.warnings if "acceleration" in w))
    # Trajectory azimuth acceleration (2.23 deg/s^2) exceeds limit (1.50 deg/s^2).

A 2.44 deg leg, the throw the shared table gives at the bottom of its
range, spends about 45 percent of its samples in science at those defaults
(the rest in turnarounds); a wider throw higher up spends more, and the
``science_fraction`` recorded on every pass block reports the actual
figure.

Selection Rules
---------------

The default rule visits the first available body in the caller's order.
A :class:`~fyst_trajectories.overhead.ScriptedSelection` walks a fixed list
of entries instead, each with optional per-visit
:class:`~fyst_trajectories.overhead.ScanOverrides`; consecutive entries on
one body chain back to back, which is how a parameter sweep is written::

    from fyst_trajectories.overhead import ScanOverrides, ScriptedSelection

    sweep = ScriptedSelection(
        [("saturn", ScanOverrides(az_speed=v)) for v in (0.5, 1.0, 1.5)]
    )
    swept = plan_calibration_night(
        ["saturn"], site, "2026-09-11T06:30:00", "2026-09-11T07:30:00", selection=sweep
    )
    passes = [b for b in swept.blocks if b.scan_type == "planet_cal"]
    print([b.metadata["applied"]["az_speed"] for b in passes])
    # [0.5, 1.0, 1.5]

An entry whose body is not available waits up to the policy's
``max_wait_seconds`` and is then set aside; the summary lists such
unplaced entries. Any callable with the
:class:`~fyst_trajectories.overhead.SelectionRule` signature is accepted.

Stepping Through a Night by Hand
--------------------------------

The driver loops over four public functions, and a person at an
interactive session can call them directly to plan a visit, inspect it,
discard it and re-plan with other overrides. Time comes from the state,
never from a clock, so a night planned twice is identical::

    from fyst_trajectories.overhead import (
        NightContext,
        NightState,
        commit_visit,
        list_candidates,
        plan_visit,
    )

    ctx = NightContext.build(["saturn", "uranus"], site, "2026-09-11T06:30:00", "2026-09-11T08:00:00")
    state = NightState.initial(ctx.start_time)

    candidates = list_candidates(state, ctx)
    print([(c.body, c.available, round(c.el_bore_estimate, 1)) for c in candidates])
    # [('saturn', True, 62.2), ('uranus', True, 31.3)]

    plan = plan_visit(state, ctx, "saturn", ScanOverrides(az_speed=1.0))
    print(plan.feasible, plan.transition.path, [b.scan_type for b in plan.blocks])
    # True direct ['slew', 'retune', 'skydip', 'retune', 'idle', 'planet_cal']

    state = commit_visit(state, plan)
    print(round((state.t - ctx.start_time).to_value("s") / 60.0, 1))
    # 28.7

An infeasible visit is returned, never raised: ``plan.reason`` names why
(the kernel could not plan the anchor, below or above the band, the
crossing too slow, no whole pass left before the night ends, the retune
pose or the pass inside the Sun zone, no Sun-safe slew, no azimuth wrap,
the goal outside the axis limits, or no way out of the zone), and
``commit_visit`` records a deferral or, for the two geometry reasons (no
wrap and the axis limits), a drop for the night. A telescope the Sun zone has
overtaken while it sat still is moved out first, by ``plan_visit`` and
``advance_idle`` alike (:func:`~fyst_trajectories.overhead.plan_escape`;
a slew named ``sun_escape`` leads the blocks and the pass records the
pose it escaped to). The night's first ``retune`` block stands for the
detector-finding operation, named in ``metadata["operation"]``, and a
skydip is reserved whenever the calibration policy's cadence says one is
due.

What a Pass Block Carries
-------------------------

Each pass is a ``planet_cal`` calibration block whose ``scan_params`` is
a relative dispatch dict (body, mode, boresight elevation, the geometry
and override keys, never an absolute window), alongside the pass anchor
``t0_scan``, three geometry records (``requested``, ``applied`` and
``solved``), the science fraction, the number of legs, the fraction of the
pass each module saw the source (``module_crossings``) and the transition
that preceded it. Every pass rebuilds from its block, and the track of the
source across the focal plane can be drawn. Rebuilding re-runs the kernel,
so here the acceleration advisory and, at this elevation, the on-sky
azimuth speed advisory are raised as warnings rather than recorded::

    from fyst_trajectories.overhead import schedule_to_trajectories
    from fyst_trajectories.visualization import plot_source_track, plot_timeline_gantt

    pairs = schedule_to_trajectories(timeline, science_only=False)
    print(len(pairs), sorted(pairs[0][0].metadata["solved"]))
    # 2 ['az_throw', 'crossing_seconds']

    fig = plot_timeline_gantt(timeline, show=False)
    fig.savefig("calibration_night.png", dpi=120)
    fig = plot_source_track(pairs[0][1], site=site, show=False)
    fig.savefig("first_pass_track.png", dpi=120)

Lateness
--------

A pass can absorb only the slack the plan left in front of it: the
``waiting_for_pass`` idle that runs from the end of the detector operations
to ``t0_scan``. The kernel's anchor lead sets that idle, so it is usually
tens of seconds; the sheet prints the figure on each pass row, and the idle
itself is a row of its own, so read the night's own number there rather than
assuming a fixed one. Once ``t0_scan`` has passed the crossing has opened
without the telescope on it, and re-planning the recorded dict from a later
anchor raises
:class:`~fyst_trajectories.exceptions.TargetNotObservableError` rather than
returning a shifted pass: re-plan the next block from
now with ``plan_visit`` on a state whose time is the actual time, from
the saved ECSV if the session is gone
(:func:`~fyst_trajectories.overhead.read_calibration_night_metadata`
rehydrates the night's inputs).
