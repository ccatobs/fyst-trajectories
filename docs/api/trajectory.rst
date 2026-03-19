Trajectory Container
====================

Container for telescope trajectory data with time-stamped position and
velocity setpoints for Az/El axes.

.. automodule:: fyst_trajectories.trajectory
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Derived Dynamics Properties
---------------------------

The following read-only properties compute higher-order derivatives on
demand from the stored velocity arrays using ``np.gradient``. They are
not stored fields -- each access recomputes the result.

- ``az_accel`` / ``el_accel``: Acceleration in degrees/second^2
  (gradient of velocity with respect to time).
- ``az_jerk`` / ``el_jerk``: Jerk in degrees/second^3
  (gradient of acceleration with respect to time).

Example::

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
whether it is science data or a turnaround.  Three constants are
exported from ``fyst_trajectories``:

- ``SCAN_FLAG_UNCLASSIFIED`` (0) -- default when no classification is available.
- ``SCAN_FLAG_SCIENCE`` (1) -- science-quality samples.
- ``SCAN_FLAG_TURNAROUND`` (2) -- turnaround or slew samples.

The ``science_mask`` property returns a boolean mask that is ``True``
for science samples, making it easy to filter trajectory data::

    import numpy as np
    from fyst_trajectories import SCAN_FLAG_SCIENCE, SCAN_FLAG_TURNAROUND

    # After generating a CE trajectory:
    traj = pattern.generate(site, duration=3600.0, start_time=t0)

    # Get only science samples (excludes turnarounds)
    science_data = traj.azimuths[traj.science_mask]

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

    start_time = Time("2026-03-15T04:00:00", scale="utc")

    trajectory = (
        TrajectoryBuilder(get_fyst_site())
        .at(ra=180.0, dec=-30.0)
        .with_config(PongScanConfig(
            timestep=0.1, width=2.0, height=2.0, spacing=0.1,
            velocity=0.5, num_terms=4, angle=0.0,
        ))
        .duration(300.0)
        .starting_at(start_time)
        .build()
    )

    print(trajectory.pattern_type)   # "pong"
    print(trajectory.center_ra)      # 180.0
    print(trajectory.pattern_params) # {'width': 2.0, ...}

**Export**::

    from fyst_trajectories.trajectory_utils import to_path_format, to_arrays

    # For OCS /path endpoint (preferred: free function)
    points = to_path_format(trajectory)
    payload = {
        "start_time": trajectory.start_time.unix,
        "coordsys": "Horizon",
        "points": points,
    }

    # Simple arrays (preferred: free function)
    times, az, el = to_arrays(trajectory)

    # Method-style calls also work (thin wrappers)
    points = trajectory.to_path_format()
    times, az, el = trajectory.to_arrays()

**Absolute times**::

    from astropy.time import Time

    from fyst_trajectories.trajectory_utils import get_absolute_times

    trajectory.start_time = Time("2026-03-15T04:00:00", scale="utc")

    # Preferred: free function
    abs_times = get_absolute_times(trajectory)

    # Method-style call also works (thin wrapper)
    abs_times = trajectory.get_absolute_times()

**Validation**::

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.trajectory_utils import validate_trajectory

    site = get_fyst_site()

    # Preferred: free function
    validate_trajectory(trajectory, site)

    # Method-style call also works (thin wrapper)
    trajectory.validate(site)

**Print formatted summary**::

    from fyst_trajectories.trajectory_utils import print_trajectory

    print_trajectory(trajectory)              # First 5 and last 5 points
    print_trajectory(trajectory, head=10)     # Customize head count
    print_trajectory(trajectory, tail=None)   # Skip tail section

**Plot trajectory**::

    from fyst_trajectories.trajectory_utils import plot_trajectory

    # Display interactive plot
    fig = plot_trajectory(trajectory, show=True)

    # Get figure for saving
    fig = plot_trajectory(trajectory, show=False)
    fig.savefig("trajectory.png")
