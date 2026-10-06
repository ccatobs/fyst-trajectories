Trajectory Utilities
====================

Validation, export, and formatted display for
:class:`~fyst_trajectories.trajectory.Trajectory` objects.

.. automodule:: fyst_trajectories.trajectory_utils
   :members:
   :undoc-members:
   :show-inheritance:

Common Operations
-----------------

**Export formats**::

    from fyst_trajectories.trajectory_utils import (
        to_arrays,
        to_path_format,
        to_path_payload,
        to_trackpoint_format,
    )

    # Ready-to-POST Go TCS /path body: {"start_time", "coordsys", "points"}
    payload = to_path_payload(trajectory)

    # Just the point rows: List[List[float]] with [time, az, el, az_vel, el_vel]
    points = to_path_format(trajectory)

    # Simple numpy arrays
    times, az, el = to_arrays(trajectory)

    # Direct-ACU ProgramTrack rows (dicts keyed as the socs TrackPoint)
    rows = to_trackpoint_format(trajectory)

**Absolute times**::

    import dataclasses

    from astropy.time import Time

    from fyst_trajectories.trajectory_utils import get_absolute_times

    trajectory = dataclasses.replace(trajectory, start_time=Time("2026-03-15T01:00:00", scale="utc"))
    abs_times = get_absolute_times(trajectory)  # Returns Time array

**Formatted display**::

    from fyst_trajectories.trajectory_utils import print_trajectory

    # Print formatted table
    print_trajectory(trajectory)              # First 5 and last 5 points
    print_trajectory(trajectory, head=10)     # Customize display

Plotting lives in the :doc:`visualization subpackage <visualization>`
(``fyst_trajectories.visualization.plot_trajectory``), and retune injection
in :doc:`retune`, which has its own topic page, :doc:`../retune_events`.

Validation Functions
--------------------

``validate_trajectory`` is the recommended entry point: it runs the bounds
check, the dynamics check, and (when ``check_sun=True``) the advisory sun
check. The low-level checks can also be called directly::

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.trajectory_utils import (
        validate_trajectory,
        validate_trajectory_bounds,
        validate_trajectory_dynamics,
    )

    site = get_fyst_site()
    validate_trajectory(trajectory, site)  # raises out of bounds, warns on dynamics or Sun

    # Check only position bounds (raises exception if out of range)
    validate_trajectory_bounds(site, trajectory.az, trajectory.el)

    # Check only dynamics (emits warning if limits exceeded)
    validate_trajectory_dynamics(site, trajectory.az, trajectory.el, trajectory.times)

Sun avoidance is advisory: a violation warns and never raises, and the check
is subsampled at roughly 60 s, so a fast scan can cross the exclusion zone
between samples without a warning. Telescope control systems must enforce
their own hard sun-avoidance limits independently::

    from fyst_trajectories import get_fyst_site, validate_sun_avoidance
    from fyst_trajectories.trajectory_utils import get_absolute_times

    site = get_fyst_site()
    # trajectory must have start_time set
    abs_times = get_absolute_times(trajectory)
    validate_sun_avoidance(site, trajectory.az, trajectory.el, abs_times)
