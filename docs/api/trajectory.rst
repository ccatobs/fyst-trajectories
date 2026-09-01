Trajectory Container
====================

Container for telescope trajectory data with time-stamped position and
velocity setpoints for Az/El axes. Worked examples are in
:doc:`../trajectory_examples`.

.. automodule:: fyst_trajectories.trajectory
   :members:
   :undoc-members:
   :show-inheritance:

Derived Dynamics Properties
---------------------------

``az_accel``/``el_accel`` and ``az_jerk``/``el_jerk`` are read-only
properties derived from the velocity arrays with ``np.gradient``::

    import numpy as np

    accel = trajectory.az_accel          # np.ndarray, same shape as times
    max_jerk = np.abs(trajectory.el_jerk).max()

Coordinate System Fields
------------------------

- ``coordsys``: Coordinate system of trajectory points (``"altaz"`` for patterns)
- ``epoch``: Optional epoch annotation (e.g., ``"J2000"``)
- ``metadata.input_frame``: Input coordinate frame used for generation
- ``metadata.epoch``: Epoch of input coordinates

Scan Flags
----------

Each trajectory sample can be classified with a scan flag indicating
whether it is science data, a turnaround, or a retune pause.  Four
constants are exported from ``fyst_trajectories``:

- ``SCAN_FLAG_UNCLASSIFIED`` (0) - default when no classification is available.
- ``SCAN_FLAG_SCIENCE`` (1) - science-quality samples.
- ``SCAN_FLAG_TURNAROUND`` (2) - turnaround or slew samples.
- ``SCAN_FLAG_RETUNE`` (3) - KID retune pause (injected by ``inject_retune()``).

The ``science_mask`` property returns a boolean mask that is ``True``
for science samples, making it easy to filter trajectory data::

    import numpy as np
    from fyst_trajectories import SCAN_FLAG_SCIENCE, SCAN_FLAG_TURNAROUND

    # Build a constant-elevation (AltAz) trajectory; no start_time needed.
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import ConstantElScanConfig, TrajectoryBuilder

    traj = (
        TrajectoryBuilder(get_fyst_site())
        .with_config(
            ConstantElScanConfig(
                timestep=0.1,
                az_start=120.0,
                az_stop=145.0,
                elevation=45.0,
                az_speed=1.0,
                az_accel=0.5,
            )
        )
        .duration(3600.0)
        .build()
    )

    # Get only science samples (excludes turnarounds)
    science_data = traj.az[traj.science_mask]

    # Check flag values directly
    n_turnaround = np.sum(traj.scan_flag == SCAN_FLAG_TURNAROUND)

If ``scan_flag`` is ``None`` (no flagging information), ``science_mask``
returns all ``True``.

Usage Examples
--------------

**Manual creation**::

    import numpy as np
    from fyst_trajectories import Trajectory

    trajectory = Trajectory(
        times=np.array([0, 1, 2, 3, 4]),
        az=np.array([100, 101, 102, 101, 100]),
        el=np.full(5, 45.0),
        az_vel=np.array([1, 1, 0, -1, -1]),
        el_vel=np.zeros(5),
    )

**Pattern generation** (recommended)::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import PongScanConfig, TrajectoryBuilder

    start_time = Time("2026-03-15T01:00:00", scale="utc")

    trajectory = (
        TrajectoryBuilder(get_fyst_site())
        .at(ra=180.0, dec=-30.0)
        .with_config(PongScanConfig(
            timestep=0.1, width=2.0, height=2.0, spacing=0.1,
            velocity=0.4, num_terms=4, angle=0.0,
        ))
        .duration(300.0)
        .starting_at(start_time)
        .build()
    )

    print(trajectory.pattern_type)   # "pong"
    print(trajectory.center_ra)      # 180.0
    print(trajectory.pattern_params) # {'width': 2.0, ...}

**Export**::

    from fyst_trajectories.trajectory_utils import to_arrays, to_path_payload

    # Ready-to-POST Go TCS /path body: {"start_time", "coordsys", "points"}
    payload = to_path_payload(trajectory)

    # Simple arrays
    times, az, el = to_arrays(trajectory)

Absolute-time conversion, validation, and the formatted-table printer are
covered in :doc:`trajectory_utils`.

**Plot trajectory**::

    from fyst_trajectories.visualization import plot_trajectory

    # Display interactive plot
    fig = plot_trajectory(trajectory, show=True)

    # Get figure for saving
    fig = plot_trajectory(trajectory, show=False)
    fig.savefig("trajectory.png")
