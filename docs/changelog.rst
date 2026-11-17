Changelog
=========

Each release lists the breaking changes a caller is likely to meet, each
with a migration, its major additions and the fixes that change a result on
a path in use. Removed names that no known consumer used are listed
together, and every other change is recorded in the repository's commit
history.
While the version is 0.x, breaking changes arrive only in minor releases,
and a patch release never changes behaviour or the public API.

v0.14.0 (2026-11-17)
--------------------

Breaking changes
~~~~~~~~~~~~~~~~

- **The Prime-Cam ring is the bays IM1 to IM6, and the labels i1 to i6
  are removed.** The module positions are the instrument team's bays,
  ``IM0`` on the optical axis and ``IM1`` .. ``IM6`` on the ring, which has
  moved (see the fix below), so no ``i`` label names a bay or is kept as
  an alias: everything that takes a module name refuses ``i1`` .. ``i6``
  with a message that says so. ``PRIMECAM_MODULES`` and
  ``primecam_geometry_dict()`` are keyed ``IM0`` .. ``IM6``, the constants
  ``PRIMECAM_IM0`` .. ``PRIMECAM_IM6`` replace ``PRIMECAM_CENTER`` and
  ``PRIMECAM_I1`` .. ``PRIMECAM_I6``, and the offsets are named
  ``PrimeCam-IM0`` .. ``PrimeCam-IM6``. The name lookups still accept
  ``c`` and ``center`` for ``IM0``, and a pass's dispatch dict still names
  the on-axis bay ``c``. A timeline whose planet-calibration footprint is
  an ``i`` label no longer reads, and a calibration night recorded for one
  cannot be reported or resumed.

  **Migration:** choose the bay by where it sits (the table on
  :doc:`instrument_offsets`), not by its number: ``i1`` is not ``IM1``,
  and no bay sits where an ``i`` label did. Look a module up with
  ``get_primecam_offset(name)``, which ignores case, rather than indexing
  ``PRIMECAM_MODULES`` with a lower-cased name.
- **The window planner is now a form of plan_constant_el_scan, and
  plan_constant_el_window_scan is removed.** ``plan_constant_el_scan``
  takes either a ``field``, as before, or ``az_window`` with ``duration``,
  which it sweeps from a literal ``start_time`` as the removed planner did.
  The field form's ``rising``, ``angle``, ``lsa_window``,
  ``max_search_hours`` and ``step_seconds`` default to ``None``, with the
  effect of their former defaults; any of them given beside ``az_window``
  is refused, as ``duration`` is without it, with a ``ValueError``. No
  result changes, except in the last digit, or by one sample, for a window
  the planner moves up a whole turn.

  **Migration:** call ``plan_constant_el_scan`` with the same keywords you
  gave ``plan_constant_el_window_scan``.
- **az_min and az_max on a constant-elevation patch are the azimuth window
  the visit sweeps.** ``generate_timeline`` scans such a patch with the
  window form of ``plan_constant_el_scan``: the subscans of a visit up to
  its next retune are one window scan in whole legs, and the visit is timed
  by the field's pass on the window's side of the meridian (see
  :ref:`overhead-timeline-window-patch`). The previous release used the
  pair only to choose the cable-wrap turn of the slew and swept the field's
  own range: Deep56 at 50° with the window (242°, 252°) on 2026-06-15 from
  13:20 to 15:50 UTC swept 145.1° to 257.2° and booked 6393.7 s of science;
  it now sweeps 241.8° to 252.2° and books 6550.2 s. ``ObservingPatch``
  refuses one key without the other, ``az_min`` not below ``az_max``, the
  pair beside ``lsa_window``, a window that reaches the meridian without
  ``rising``, and a ``rising`` that names the other side.

  **Migration:** give the window the visit is to sweep, written unwrapped
  across 0°/360° as (355, 370); drop the pair to sweep the field's own
  range.
- **A pong or daisy patch takes no elevation.** ``ObservingPatch`` refuses
  an ``elevation`` on a pong or daisy patch with a ``ValueError``. Such a
  pattern is scanned where its field is on the sky, and the previous
  release judged, slewed to and recorded the visit at the given elevation,
  a pose the scan never took. A constant-elevation patch still needs one.

  **Migration:** leave ``elevation`` as ``None`` on a pong or daisy patch.
- **generate_timeline refuses a constant-elevation patch that can never
  have a pass, and a patch sampled faster than the Go TCS accepts.** Before
  it schedules anything, ``generate_timeline`` raises one ``ValueError``
  naming every constant-elevation patch with no pass over a sidereal day
  from the night's start, on the side the patch asks for (either side when
  it names none, inside its window for an ``lsa_window``): a field
  spanning a whole turn of RA, one reaching a celestial pole, a scan
  elevation the field never crosses, or a pass longer than the planner's
  12 h search. It also refuses by name a patch whose
  ``scan_params["timestep"]`` is under 0.05 s, the Go TCS ``/path``
  minimum sample interval (``GO_TCS_MIN_SAMPLE_INTERVAL_SEC``). The
  previous release returned a night that idled through a patch without a
  pass, and booked a patch sampled too fast, every scan of which
  ``to_path_format`` refused.

  **Migration:** drop the patch the message names, or give it an elevation
  its field crosses; sample at 0.05 s or more (the planners' default is
  0.1 s), and resample a rebuilt trajectory where a finer grid is needed.

Additions
~~~~~~~~~

- **A timeline block for a constant-elevation window scan.**
  ``TimelineBlock.from_window_scan`` writes the scan that the window form
  of ``plan_constant_el_scan`` returned as a science block, with or
  without an RA/Dec field, and ``schedule_to_trajectories`` rebuilds it
  with that form, from memory and after an ECSV round trip. The writer
  refuses a scan it would not reproduce sample for sample; with
  ``span=(t_start, t_stop)`` it writes one stretch of the scan, so one scan
  can be booked as consecutive blocks (see
  :ref:`overhead-timeline-window-scan`).
- **A patch can be observed with a Prime-Cam module.** ``ObservingPatch``
  takes ``detector_offset``, as ``get_primecam_offset`` returns it for a
  bay, and ``generate_timeline`` schedules the boresight that holds that
  module on the field: every pose it scores, checks against the elevation
  limits and the Sun, slews to, holds or records is the boresight's. A
  pong or daisy is booked only while both the module's pattern and the
  boresight's stay inside the elevation limits. On a constant-elevation
  patch ``elevation`` is the module's. Every science block records the
  offset as ``metadata["detector_offset"]``, and a constant-elevation
  block the module's elevation as ``metadata["detector_elevation"]``;
  ``schedule_to_trajectories`` rebuilds the block with them. A patch with
  the on-axis bay ``IM0`` is scheduled as one without an offset (see
  :ref:`overhead-timeline-module`). Reported by Yihong Zhou.

Fixes that change results
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Every Prime-Cam ring position moves by 0.92° on the sky.** The bays
  now sit where the instrument team's optical model puts them (the table
  on :doc:`instrument_offsets`), 30° round the ring from the old
  positions, so everything aimed with a ring module moves; the centre
  module is unchanged. The layout still awaits an on-sky check (see
  :ref:`index-pending-verification`).
- **A constant-elevation pass sweeps where the field's border meets the
  scan elevation.** ``plan_constant_el_scan`` sweeps from the smallest to
  the largest azimuth at which the field's border crosses the scan
  elevation (the equation is on :doc:`planning`), and ``az_padding``
  defaults to 0° (``az_padding=2.0`` restores the old margin). A field
  60° of RA by 14° at Dec -2° setting through 50° sweeps 26.95° of
  azimuth where it swept 167.65°. The pass opens and closes when it did,
  and a timeline written by an earlier release rebuilds with the new
  range.
- **A simulated slew ends where the scan starts.** The offline scheduler
  slews to the first sample of the visit's first subscan, where it slewed
  to the field centre, so block times move by up to a few seconds.
- **A simulated pong or daisy visit leaves the telescope where its scan
  ends.** Such a block records the elevation its trajectory ends at as
  ``TimelineBlock.el_final``, and ``write_timeline`` writes it as a new
  ``el_final`` column; a file without the column reads as before.
- **A constant-elevation plan's Sun advisory screens the path it plans.**
  The field form of ``plan_constant_el_scan`` screens the built boresight
  trajectory, as the window form does, so the advisory can appear or
  disappear. No trajectory changes.
- **scan_flag separates two legs whose turnaround falls between samples.**
  Where a constant-elevation or source-CES turnaround is no longer than a
  timestep, the first sample after the reversal is flagged as turnaround,
  so consecutive legs are separate science runs. Scans whose turnarounds
  hold a sample do not change.
- **A daisy plan reports the duration it built.** ``ScanBlock.duration``,
  ``computed_params["duration"]`` and the summary give the built length
  (299.9 s for a 300 s request at the default 0.1 s). The trajectory is
  unchanged.
- **A simulated constant-elevation visit starts when its pass opens.** The
  first science block starts at the pass's first sample, the wait before
  it is an idle block with reason ``waiting_for_pass``, and the last block
  ends one sample after the pass's last. In the previous release the
  block could begin before the pass's first sample, or minutes after the
  pass had opened. Reported by Yihong Zhou.
- **Scan-mode planet calibrations record each pass as it runs.** With
  ``CalibrationPolicy(planet_cal_scan=True)`` a pass block spans its
  samples, the wait before it is an idle block with reason
  ``waiting_for_pass``, a pass that would open before the previous one
  ends is dropped, and the recorded ``boresight_rot`` is ``None``.
- **A simulated scan is booked only where the control software would run
  it.** ``generate_timeline`` puts every science subscan and scan-mode
  planet pass to the checks the control software makes before it sends a
  scan to the Go TCS: the Sun on every sample, the axis limits and the
  minimum sample interval. A visit refused even at its shortest subscan
  idles with reason ``sun_scan``, ``axis_dynamics`` or ``unplannable``
  (see :ref:`overhead-timeline-booking-screen`). A small pattern scanned
  fast fails the elevation acceleration limit of 1.5 deg/s²: a 4° pong or
  a daisy of the default shape at 1 deg/s books no science, and both are
  booked at 0.5 deg/s.
- **A simulated telescope holds no pose inside the Sun zone.** An idle, a
  calibration that does not scan, a wait or a retune is booked only at a
  pose that stays outside the zone for the whole block; otherwise the
  pose is left with a ``sun_escape`` slew, or the idle is labelled
  ``no_escape``. A daytime night can lose a visit, since an escape is
  chosen without regard to the next patch. A night with the Sun below the
  horizon is not affected.
- **A constant-elevation patch is booked on the pass it asks for.** A
  patch with ``lsa_window`` is booked in its window, on the side
  ``plan_constant_el_scan`` scans; a window already open when the
  scheduler reaches the patch is not booked that day, and a ``rising``
  beside the window is not read. A patch that asks for a side with
  ``rising`` is booked on that side's pass even when it opens before the
  field centre transits.
- **A simulated night rebuilds the same samples from a file as from
  memory.** ``schedule_to_trajectories`` cuts each block's samples at its
  start and stop as ``write_timeline`` stores them, to the millisecond, so
  a block rebuilds to equal arrays from memory and after an ECSV round
  trip.

v0.13.0 (2026-11-09)
--------------------

Additions
~~~~~~~~~

- **Constant-elevation scans over a given azimuth window.**
  ``plan_constant_el_window_scan`` sweeps ``az_window`` at one elevation
  from a literal ``start_time`` for a given duration, the inputs of a
  constant-elevation ``azimuth_range`` row in OpsDB (its duration in hours
  there, in seconds here). The duration is rounded to whole azimuth legs as
  ``plan_constant_el_scan`` rounds its pass.
  The window is commanded as given and moves by a whole turn only when the
  azimuths the sweep reaches, its turnarounds included, or the boresight's
  under a ``detector_offset``, would overrun an azimuth limit. The built
  trajectory is screened against the Sun, and an optional ``field`` is
  checked for drifting through the window during the scan; both checks warn
  and never refuse.

Fixes that change results
~~~~~~~~~~~~~~~~~~~~~~~~~

- **A constant-elevation pass takes the azimuth branch that holds its
  turnarounds and its boresight.** ``plan_constant_el_scan`` placed its
  azimuth range on a branch that held the module's samples but not the
  turnaround overshoot or, with ``detector_offset``, the boresight's path,
  and refused the pass with ``AzimuthBoundsError`` when either ran past a
  limit although the branch one turn away holds the whole sweep. Such a
  pass now plans one turn over: a 2 × 2° field at RA 60°, Dec +26.25°
  setting through 40° at 3 deg/s, refused before, plans on
  (-14.86°, -5.22°). A pass whose own branch holds everything does not move
  (the rule is on :doc:`planning`). The offline scheduler books the passes
  it idled through as unplannable: that field's pass on 2026-11-16 gains
  353 s of science.
- **A field wider than 180° of RA is timed by its own extent.**
  ``plan_constant_el_scan`` took the field's leading and trailing RA edges
  modulo 360°, so a field across RA 0° wider than 180° of RA was timed by
  the complementary arc: a field 200° of RA wide at Dec -27.8°, centred at
  RA 10° and rising through 50°, was planned as a 10.65 h pass, or refused
  when the search began inside it. Its own pass lasts 13.29 h, which
  ``max_search_hours=36`` plans and the default 12 h refuses, naming the
  pass's length. A field spanning a whole turn of RA raises
  ``PointingError`` instead of planning a pass.
- **A constant-elevation sample on the edge of the science range is
  science.** The flag of a sample where a leg meets a turnaround depended on
  the last bits of the inputs: the same recorded source-CES pass ended on a
  turnaround sample under SciPy 1.17.0 and on a science sample under
  SciPy 1.18.1. ``ConstantElScanPattern`` now flags a sample within 1e-9° of
  the science range as science and places it on the range, so a block
  quantised to whole legs ends on a science sample; 15 of the 21 recorded
  test passes gain one. The Go TCS ``/path`` body carries no flags.

v0.12.0 (2026-10-27)
--------------------

Breaking changes
~~~~~~~~~~~~~~~~

- **The ECSV boresight_angle column is the roll TOAST applies.**
  ``write_timeline`` writes ``-nasmyth_sign * el`` for each row: the roll
  TOAST's ``SimGround`` applies to the boresight in the az/el frame before
  it adds the parallactic rotation itself, with a positive sense opposite
  to this library's rotation. The column held the celestial field rotation
  ``nasmyth_sign * el + q``, with ``q`` the parallactic angle, at the row's
  mid azimuth. The header key
  ``boresight_angle_convention`` marks the new files, and ``read_timeline``
  ignores the column. ``TimelineBlock.boresight_angle``, which nothing
  read, and the ``site`` argument of the five ``TimelineBlock`` factories,
  which served only that field, are removed.

  **Migration:** the celestial field rotation of a row is
  ``Coordinates(site).get_field_rotation_from_altaz(0.5 * (azmin + azmax), el)``;
  drop ``site=`` from the ``TimelineBlock.calibration`` / ``retune`` /
  ``idle`` / ``slew`` / ``science`` calls.

- **The validators and the dispatch helpers take site by keyword.**
  ``validate_trajectory``, ``validate_trajectory_bounds``,
  ``validate_trajectory_dynamics``, ``validate_sun_avoidance``,
  ``estimate_slew_time`` and ``choose_encoder_solution`` take the data they
  check or price positionally and ``site``, with every option after it, by
  keyword, as the planners do. ``site`` was the first, second, fifth or
  sixth positional argument, by function, and a call that passes it
  positionally, such as a dispatcher's
  ``validate_trajectory(trajectory, site)``, raises ``TypeError``.

  **Migration:** name ``site=`` and put the data first:
  ``validate_trajectory(trajectory, site=site)``,
  ``validate_trajectory_bounds(az, el, site=site)``,
  ``validate_trajectory_dynamics(az, el, times, site=site)``,
  ``validate_sun_avoidance(az, el, times, site=site)``,
  ``estimate_slew_time(az1, el1, az2, el2, site=site)`` and
  ``choose_encoder_solution(current_az, current_el, goal_az, goal_el, obstime, site=site)``;
  ``check_sun``, ``coords`` and ``sun_safe`` go by keyword too.

- **The source-CES planners refuse any explicit az_padding beside
  az_throw.** ``az_padding`` defaults to ``None``, which pads by 0.5° as
  before, and ``plan_source_ces``, ``plan_source_ces_passes`` and
  ``compute_source_ces_params`` refuse ``az_throw`` combined with any padding
  the caller passes, 0.5 included, with ``ValueError``. An explicit 0.5 was
  accepted beside ``az_throw`` because it equalled the default, while 0.6 was
  refused.

  **Migration:** leave ``az_padding`` unset when passing ``az_throw``; code
  that fills a missing padding with 0.5 passes ``scan_params.get("az_padding")``
  instead.

Fixes that change results
~~~~~~~~~~~~~~~~~~~~~~~~~

- **TOAST simulates the focal plane at its mechanical orientation.** A
  ``SimGround`` run from a fyst-trajectories schedule turned the focal
  plane to ``-nasmyth_sign * el`` on the sky instead of
  ``nasmyth_sign * el + q``, an error of ``2 * nasmyth_sign * el + q``
  (67° in the constant-elevation case reported). Reported and diagnosed
  by Yihong Zhou.
- **The source-CES Sun screen covers the whole pass.** ``plan_source_ces``,
  ``plan_source_ces_passes`` and ``compute_source_ces_params`` checked the
  Sun over the source's crossing window only, while the trajectory runs for
  the leg-quantised duration from the crossing start, so the last seconds of
  a pass that outlasts its window (up to 24 s on the recorded test passes)
  went unchecked. The check now runs to the trajectory's end, and a pass that
  enters the exclusion zone only in those seconds warns. No recorded pass
  changes.
- **A survey-scheduler subscan is recorded in the cable wrap the slew
  reached.** On a night with a field reached from the other cable wrap (a
  field at azimuth 180° to 360° approached across north), the offline
  scheduler recorded the subscan's azimuths, and so the pose the next slew
  started from, a full turn from where the slew had left the mount, and
  priced and Sun-checked that slew from the wrong pose. The block and the
  calibrations and idle ticks parked after it now carry the slew's wrap; a
  rebuilt trajectory still returns the planner's. No block's time changes on
  the shipped example nights; on the 12-hour night of the timeline page, the
  two subscans from 07:20 UTC and the 52 blocks after them are now recorded
  360° lower, in the slew's wrap.

v0.11.0 (2026-10-21)
--------------------

Breaking changes
~~~~~~~~~~~~~~~~

- **A scan block's computed parameters are read-only.**
  ``ScanBlock.computed_params`` is a read-only mapping, as
  ``TrajectoryMetadata.pattern_params`` and ``ObservingPatch.scan_params``
  have been since v0.10.0; adding a key in place raises ``TypeError``.

  **Migration:** ``params = dict(block.computed_params)`` and add to the copy.

- **ECSV times carry an explicit UTC offset.** ``write_timeline`` writes
  ``2026-11-01T09:00:00.123+00:00`` where it wrote ``2026-11-01 09:00:00.123``,
  and astropy's ``Time`` does not parse a string with an offset: code that
  read the ``start_time`` and ``stop_time`` columns, or the window keys in
  the header, with it raises ``ValueError``.

  **Migration:** ``read_timeline`` reads both forms; for a column on its own,
  ``datetime.fromisoformat(s)`` or ``Time(s.removesuffix("+00:00"), scale="utc")``.

Additions
~~~~~~~~~

- **One record per visit for the execution layer.** ``dispatch_records(timeline)``
  returns a planned calibration night as a JSON-ready record: for each visit
  its slews, its detector operations in execution order and each pass's
  ``scan_params`` with its ``scheduled_t0_unix``, duration and slack.
  ``NightRecord`` and its parts declare the shape, whose keys are stable and
  may gain additions.

Fixes that change results
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Calibration-night passes wait for their dispatch lead, and the sheet
  hands the execution layer the search start.** Every pass now starts at
  least ``CalibrationNightPolicy.dispatch_lead_seconds`` (20 s) after the
  detector operations before it, and a pass row's ``scheduled_t0_unix`` is
  the planner's search start, not the pass start: the execution layer
  searches from the later of that instant and 10 s after the dispatch, so
  handed the pass start it refused every rising pass. Passes start later,
  by about 26 s for each visit that directly follows another, so a pass near
  the end of its elevation band or of the night can be lost. A night
  recorded before the lead existed resumes with it at zero.

- **A pong or daisy visit stays clear of the zenith.** The offline scheduler
  admits, slews to and times a pong or daisy visit only while the circle
  that bounds its pattern excludes the zenith, where it used to slew and
  then idle as unplannable; a field whose pattern passes over the zenith
  near transit now idles in place through that interval and loses those
  subscans.

- **Anchored source-CES passes on the inner-ring modules.** Given
  ``start_time`` and no ``el_bore``, ``plan_source_ces``,
  ``plan_source_ces_passes`` and ``compute_source_ces_params`` refused a
  ring module whose cover lies beyond the boresight along the source's
  motion, and often ``i2``. They now plan the pass, which starts when the
  source reaches the module, and within 2.53° of the culmination derive a
  ring module's ``el_bore`` from its projected cover. Every anchored call
  that planned before plans the same pass, and a calibration-night visit
  deferred as ``unplannable`` on such a module is now planned.

- **A schedule handed to TOAST keeps its UTC times on any host.** TOAST's
  ``GroundSchedule`` parses the ECSV times with ``datetime.fromisoformat``
  and takes their ``timestamp()``, so the offset-free strings v0.10.0 wrote
  became host-local times; the ``+00:00`` offset above makes them UTC
  everywhere. Fixed by Yihong Zhou.

v0.10.0 (2026-10-06)
--------------------

Breaking changes
~~~~~~~~~~~~~~~~

- **The focal-plane rotation has one name.** ``boresight_to_detector``,
  ``detector_to_boresight`` and ``sky_to_focal_plane`` take their angle as
  ``focal_plane_rotation``. The old name, ``field_rotation``, is also the
  name of the celestial field rotation, a different angle that includes the
  parallactic angle. ``compute_focal_plane_rotation`` takes everything after
  ``el`` by keyword, and ``apply_detector_offset`` everything after
  ``offset``.

  **Migration:** rename the keyword, and pass every argument after ``el``
  or ``offset`` by name.

- **The planners take only their target positionally.** ``plan_pong_scan``,
  ``plan_daisy_scan``, ``plan_constant_el_scan``, ``plan_pong_altaz_scan``
  and ``plan_daisy_altaz_scan`` take only their target positionally, and
  ``plan_pong_rotation_sequence`` only its config, so two of their numbers
  can no longer be swapped unnoticed.

  **Migration:** pass every other argument by name.

- **One exception rule.** ``PointingError`` means a well-formed request
  that cannot be met for this site, target and time, and a plain
  ``ValueError`` a malformed one, so a caller can treat the first as an
  ordinary outcome. ``register_pattern``, ``TrajectoryBuilder.with_config``,
  ``validate_sample_count`` and ``rewrap_trajectory_azimuth`` now raise a
  plain ``ValueError`` (all four raised ``PointingError`` in v0.9.0),
  ``plan_constant_el_scan``'s no-crossing and near-pole refusals raise
  ``PointingError``, and a ``dwell`` longer than its crossing raises the
  new ``DwellExceedsCrossingError``, a ``PointingError``.

  **Migration:** catch ``ValueError`` for argument errors, and
  ``PointingError`` (still a ``ValueError``) first where the two are
  handled differently.

- **Value types are frozen and the shared tables read-only.**
  ``Trajectory``, ``ArrayFootprint`` and ``ScanBlock`` compare and hash by
  identity; ``==`` on two trajectories raised, and on a copy its result
  depended on the Python version. ``pattern_params``,
  ``ObservingPatch.scan_params``, ``SOLAR_SYSTEM_BODIES``,
  ``PRIMECAM_MODULES`` and ``FLUX_CALIBRATORS`` are read-only, so one
  caller cannot change them for another.

  **Migration:** compare contents with ``numpy.array_equal``, and copy with
  ``dict(...)`` or ``list(...)`` before editing or passing to
  ``yaml.safe_dump``.

- **A calibration night sweeps the solved throw at 1.0 deg/s².**
  ``CalibrationNightPolicy.az_accel`` defaults to 1.0 deg/s² (was 1.5), so
  the turnaround peaks at 1.5 deg/s², the instrument team's peak value. A
  pass sweeps the throw solved from the footprint, not the scan table's
  width: the planner times each pass to the source's crossing of the
  module, so the table's allowance for an unknown start offset is not
  needed. The dispatch dict carries ``az_padding`` in place of
  ``az_throw``.

  **Migration:** ``CalibrationNightPolicy(az_accel=1.5,
  use_table_throw=True)`` keeps the earlier acceleration and width.

- **The offline scheduler books only what it can run.** A science subscan
  is booked only when the planner can build its trajectory; otherwise the
  scheduler idles with the reason ``unplannable``, where it had booked
  blocks that no trajectory could run. A pong subscan is booked as whole
  pattern periods (recorded as ``n_cycles``), a pong patch whose period
  can never fit is refused, and so is a constant-elevation patch without a
  pinned ``elevation``, which had been scanned at whatever elevation the
  field centre had when selected. Science-scan counts and efficiencies
  fall to what the trajectories run.

  **Migration:** keep a pong period under ``max_scan_duration`` (less
  ``retune_duration`` when ``retune_cadence`` is 0), pin ``elevation`` on
  every constant-elevation patch, and re-baseline stored counts and
  efficiencies.

- **Hit maps take a module mapping.** ``plot_hit_map`` takes ``modules=``
  (a mapping), ``site`` by keyword and the module field as a radius,
  ``fov_radius_deg`` (default 0.65°, so a filled map is the default).

  **Migration:** ``modules={"label": offset}``, ``site=site``,
  ``fov_radius_deg=module_fov / 2``, and ``fov_radius_deg=None`` for the
  raw track.

- **Moved and removed names.** ``inject_retune``, ``sample_retune_events``
  and ``DEFAULT_RETUNE_DURATION_SEC`` move from ``trajectory_utils`` to
  ``fyst_trajectories.retune``, and ``estimate_slew_time`` from
  ``fyst_trajectories.overhead`` to ``fyst_trajectories.dispatch``, so
  pricing a slew no longer imports the offline simulator. Removed, none
  used by a known consumer: ``Site.atmosphere`` (it read as if it enabled
  refraction, but ``Coordinates`` never used it; pass an
  ``AtmosphericConditions`` to ``Coordinates`` or the planner),
  ``Trajectory.coordsys`` and ``Trajectory.epoch``,
  ``TrajectoryMetadata.epoch``, ``CalibrationSpec.elevation``,
  ``overhead.get_observable_windows`` (use ``check_observability``, with
  ``el_min=30.0`` for its old 30° floor),
  ``overhead.compute_nasmyth_rotation`` (now
  ``Coordinates.get_field_rotation_from_altaz``), and five names without
  a successor: ``coordinates.AltAzCoord``,
  ``patterns.utils.generate_time_array``,
  ``overhead.utils.circular_mean_deg``, ``overhead.get_transit_time`` and
  ``overhead.get_max_elevation``.

  **Migration:** import the moved names, and ``RetuneEvent``, from
  ``fyst_trajectories``; for a removed name, drop the argument or call the
  replacement named.

Additions
~~~~~~~~~

- **Multi-rotation Pong tilings.** ``plan_pong_rotation_scans`` plans a
  multi-rotation Pong tiling as back-to-back blocks.
- **A Sun-avoidance seam.** ``fyst_trajectories.sun_protocols`` defines the
  seam's protocols, with runtime-checkable batch, zoned and path extensions
  that the ``make_sun_safe`` and ``make_slew_safe`` models implement.
- **Calibration nights resume.** ``NightContext.from_timeline`` and
  ``NightState.from_timeline`` resume a calibration night from its
  timeline, in memory or read back from ECSV.

Fixes that change results
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Inner-ring module pointing.** ``apply_detector_offset``, and so every
  ``detector_offset=`` path and ``TrajectoryBuilder.for_detector``, takes
  the Nasmyth rotation at the boresight elevation it solves for; inner-ring
  module pointing moves by up to 3.3 arcmin.
- **Constant-elevation velocities through the turnarounds.**
  ``apply_detector_offset`` keeps a constant-elevation scan's analytic
  velocities; its velocity columns were off in the turnarounds, by up to
  0.28 deg/s at a 1 s timestep and the default 1.0 deg/s² acceleration.
- **Dispatch checks.** ``choose_encoder_solution`` refuses a NaN or
  infinite position; a lost position read had chosen the most negative
  azimuth wrap. ``validate_trajectory`` checks the commanded velocity
  columns that ``to_path_format`` uploads, and a constant-elevation scan at
  the azimuth velocity ceiling no longer earns a limit warning from
  rounding, which a dispatcher that escalates warnings turned into a
  refusal.
- **The high-elevation advisory has one rule.** It judges the highest
  sample that moves at half the top azimuth speed or more and quotes its
  numbers; it had judged the single fastest sample, which rounding picked
  when the azimuth rate was constant. A 2° ``plan_pong_altaz_scan`` at
  0.4 deg/s centred at 59° warns where it was silent, and a
  ``plan_pong_scan`` at 80° elevation reports 0.35 deg/s on sky where it
  reported 0.57 deg/s.
- **The CAD Sun model works on Linux.** ``make_sun_safe("cad")``,
  ``make_slew_safe`` and ``load_avoidance_data("cad")`` had been refused on
  every Linux install by the CAD table's pinned digest.
- **Planner Sun and crossing checks.** ``plan_pong_altaz_scan`` and
  ``plan_daisy_altaz_scan`` warn when the realised block enters the Sun's
  exclusion zone, not only its start, and ``plan_constant_el_scan`` raises
  ``PointingError`` when the two field edges cross on different passes,
  where it had joined them into a day-long scan.
- **The offline scheduler's Sun and elevation gates.** The
  constant-elevation Sun clip checks the whole swept corridor, not the
  field centre, and with ``planet_cal_scan=True`` the scheduler slews to
  each planet calibration with ``plan_transition`` and skips a planet
  inside the Sun zone; daytime visits and planet-calibration passes had
  been booked inside the zone. A pong or daisy visit is admitted and sized
  by its whole scan pattern (a pong's box, a daisy's petals), not the field
  centre, so a visit near rising or setting starts later or ends sooner;
  visits had been booked whose trajectories went below the elevation
  limit, by more than 5° for a 10 × 10° pong.
- **Calibration nights.** ``plan_calibration_night`` checks the moves and
  waits between the passes of a visit against the Sun, and chooses the
  azimuth wrap for every pass, so a later pass no longer runs past an
  azimuth limit. A policy that names the centre module ``"IM0"``, ``"C"``
  or ``"Center"`` records ``"c"`` in every pass's dispatch dict, which the
  execution layer's centred-only check accepts; the dict had carried the
  spelling given, and the check refused every pass.
- **Hit maps.** ``accumulate_hitmaps`` bins a trajectory whose clock does
  not start at zero at the right sky position.

v0.9.0 (2026-09-21)
-------------------

Breaking changes
~~~~~~~~~~~~~~~~

- **No rising flag with a sidereal window.** ``plan_constant_el_scan``
  refuses ``rising`` together with ``lsa_window``, and ``rising`` defaults
  to ``None`` (the rising crossing); the sidereal window fixes the timing,
  so the flag did nothing on that path.

  **Migration:** drop ``rising`` from an ``lsa_window`` call.

- **Two new required source-CES keys.** ``SourceCESComputedParams`` gains
  the required keys ``az_speed`` and ``crossing_seconds``.

  **Migration:** include both where code builds the dict or compares it
  against a fixed key list.

- **A 300 s retune between blocks.** ``OverheadModel.retune_duration``
  defaults to 300 s (was 5 s), the whole-array retune between scan blocks;
  on the 8 h regression night efficiency falls from 25.2% to 22.2%.

  **Migration:** pass ``OverheadModel(retune_duration=5.0)`` to reproduce
  earlier timelines.

- **Science blocks record the executed azimuth envelope.** A science
  block's ``az_start`` / ``az_end`` (the ECSV ``azmin`` / ``azmax``) record
  the azimuth envelope its trajectory executes, not an estimate of the
  field's width; on the reference eight-hour night they move from
  (83.71°, 145.94°) to (102.33°, 215.62°).

  **Migration:** derive a science window from the recorded field geometry.

- **A typed error from the focal-plane inverse.** The focal-plane inverse
  (``detector_to_boresight``, ``apply_detector_offset``) raises
  ``OffsetInversionError``, a ``PointingError``, instead of
  ``RuntimeError`` at the pole or when the refinement does not converge.

  **Migration:** catch ``OffsetInversionError`` or ``PointingError``.

Additions
~~~~~~~~~

- **Encoder solutions.** ``choose_encoder_solution`` returns an
  ``EncoderSolution``, an ``(az, el)`` tuple whose ``az_shift`` is the
  multiple of 360° from the caller's goal azimuth to the chosen wrap;
  ``rewrap_trajectory_azimuth`` applies that shift to a trajectory, and the
  five refusals raise ``EncoderSolutionError`` with a typed ``cause``.
- **Footprint transforms.** ``fyst_trajectories.planning.footprints``
  publishes the three footprint transforms a source-CES pass is solved
  against: ``resolve_footprint``, ``inflate_footprint`` and
  ``offset_footprint_eta``.
- **The calibration-night planner.** ``overhead.plan_calibration_night``
  plans one night of solar-system calibration passes from the instrument
  team's per-body scan tables and returns an ``ObservingTimeline`` whose
  blocks carry relative dispatch dicts; its step functions let a visit be
  planned, inspected and re-planned interactively, and ``dispatch_sheet``
  and ``summarize_calibration_night`` read a night back.
- **Source tracks through the focal plane.** ``sky_to_focal_plane``,
  ``planning.source_ces_focal_plane_track`` and
  ``visualization.plot_source_track`` trace a source through the focal
  plane over a planned source-CES pass.
- **Transitions.** ``overhead.plan_transition`` checks a slew between
  observing poses (the wrap choice, the Sun path sweep and the slew
  estimate) and returns a ``Transition`` whose ``DeferralReason`` cause
  says why a slew was refused.
- **Escapes.** ``overhead.plan_escape`` finds the move out of the Sun zone
  for a telescope the zone has overtaken; the offline scheduler and the
  calibration-night planner escape before they park, idle or slew there.

Fixes that change results
~~~~~~~~~~~~~~~~~~~~~~~~~

- **The Sun checks cover what the mount sweeps.** The source-CES Sun
  screen and azimuth-bounds check include the turnaround overshoot (1.41°
  per side at 1.5 deg/s and 1.0 deg/s²), and ``plan_constant_el_scan``
  sweeps the whole resolved pass. The offline scheduler checks a patch with
  a pinned ``elevation`` against the Sun at that elevation and plans every
  slew with ``plan_transition``, recording a slew that the Sun zone or the
  wrap refuses as a labelled idle. A visit could be booked with its
  recorded pose inside the zone, and a slew recorded through it.
- **Blocks record where they end.** Every swept block records
  ``az_final``, its trajectory's last azimuth, and the next slew starts
  there; the pose handed forward had been up to 11.2° (planet
  calibrations) and 29.7° (science) from where the scan ended.
- **Retunes inside the pass.** A constant-elevation visit fits the retunes
  before its subscans inside the time its pass lasts, where they had been
  added on top; on the 12 h night of the ``fyst_trajectories.overhead``
  example, efficiency falls from 26.5% to 23.7%.
- **The leg quantiser counts turnarounds.** ``n_scans`` and ``duration``
  change for short, fast constant-elevation legs (a 300 s request with
  2.44° legs at 1.5 deg/s had quantised to 848 s).
- **Source-CES times in UTC.** A source-CES pass records ``t0_iso`` /
  ``t1_iso`` in UTC; a window given on another time scale was recorded in
  that scale and read back as UTC.
- **Every RA/Dec alias accepted.** ``radec_to_altaz``, ``altaz_to_radec``
  and ``radec_to_altaz_with_pm`` accept every alias in ``FRAME_ALIASES``;
  ``frame="J2000"`` and ``frame="ICRS"`` raised.

v0.8.0 (2026-08-30)
-------------------

Breaking changes
~~~~~~~~~~~~~~~~

- **Rebuilt science trajectories are sliced to their block.**
  ``schedule_to_trajectories`` slices each rebuilt science trajectory to
  its block's ``[t_start, t_stop)`` window, so a constant-elevation visit's
  subscans come back as consecutive slices rather than one full pass each.

  **Migration:** rebuild from ``metadata["t0_scan"]`` with
  ``plan_constant_el_scan`` for a full pass.

- **The scalar Sun defaults return to 45° and 50°.** The scalar
  Sun-avoidance defaults revert to 45° exclusion and 50° warning (0.7.0
  shipped 50° and 55°).

  **Migration:** ``get_fyst_site(sun_exclusion_radius=50.0,
  sun_warning_radius=55.0)`` keeps the 0.7.0 values.

Fixes that change results
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Row times start at zero.** ``to_path_format`` and
  ``to_trackpoint_format`` re-zero row times to ``times[0]``; a trajectory
  whose clock did not start at zero was serialized ``times[0]`` late.
- **Scheduler poses and slews.** The offline scheduler updates its pose
  when it emits a slew, keeps azimuth ranges ordered across the cable-wrap
  seam and evaluates a slew block's ``boresight_angle`` at its true
  mid-travel azimuth, so poses, slew estimates and that angle move on
  affected nights.
- **ECSV keeps its angle units.** Timeline ECSV output keeps its angle
  units, restoring TOAST ``GroundSchedule`` compatibility.

Earlier releases
----------------

Earlier releases are listed as tags at
https://github.com/ccatobs/fyst-trajectories/tags.
