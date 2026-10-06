Changelog
=========

Each release lists the breaking changes a caller is likely to meet, each
with a migration, its major additions and the fixes that change a result on
a path in use. Removed names that no known consumer used are listed
together, and every other change is recorded in the repository's commit
history.
While the version is 0.x, breaking changes arrive only in minor releases,
and a patch release never changes behaviour or the public API.

Unreleased
----------

Planned: the module labels ``c`` / ``i1`` .. ``i6`` will be renamed to
the instrument team's ``IM0`` .. ``IM6`` in the first minor release after
the correspondence to the as-built focal plane is confirmed; the two
schemes do not correspond index-for-index (see :doc:`instrument_offsets`).

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
