Trajectory Utilities
====================

Utility functions for working with ``Trajectory`` objects. These free functions
are the primary API for trajectory operations.

.. automodule:: fyst_trajectories.trajectory_utils
   :members:
   :undoc-members:
   :show-inheritance:

Common Operations
-----------------

**Validation**::

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.trajectory_utils import validate_trajectory

    site = get_fyst_site()
    validate_trajectory(trajectory, site)  # Raises if out of bounds, warns if dynamics exceeded

**Export formats**::

    from fyst_trajectories.trajectory_utils import to_path_format, to_arrays

    # For OCS /path endpoint
    points = to_path_format(trajectory)
    # Returns: List[List[float]] with [time, az, el, az_vel, el_vel]

    # Simple numpy arrays
    times, az, el = to_arrays(trajectory)

**Absolute times**::

    from astropy.time import Time

    from fyst_trajectories.trajectory_utils import get_absolute_times

    trajectory.start_time = Time("2026-03-15T04:00:00", scale="utc")
    abs_times = get_absolute_times(trajectory)  # Returns Time array

**Visualization**::

    from fyst_trajectories.trajectory_utils import plot_trajectory, print_trajectory

    # Print formatted table
    print_trajectory(trajectory)              # First 5 and last 5 points
    print_trajectory(trajectory, head=10)     # Customize display

    # Plot 3-panel figure (Az vs Time, El vs Time, Sky Track)
    fig = plot_trajectory(trajectory, show=True)

    # Get figure without displaying
    fig = plot_trajectory(trajectory, show=False)
    fig.savefig("trajectory.png")

Convenience Methods
-------------------

``Trajectory`` methods (``validate()``, ``to_path_format()``, ``to_arrays()``,
``get_absolute_times()``, ``plot()``) are thin wrappers around these free
functions.

Validation Functions
--------------------

Low-level validation utilities:

.. autofunction:: fyst_trajectories.trajectory_utils.validate_trajectory_bounds
   :no-index:

.. autofunction:: fyst_trajectories.trajectory_utils.validate_trajectory_dynamics
   :no-index:

**Example usage**::

    from fyst_trajectories.trajectory_utils import (
        validate_trajectory_bounds,
        validate_trajectory_dynamics,
    )

    # Check only position bounds (raises exception if out of range)
    validate_trajectory_bounds(site, az_array, el_array)

    # Check only dynamics (emits warning if limits exceeded)
    validate_trajectory_dynamics(site, az_array, el_array, times_array)

.. autofunction:: fyst_trajectories.trajectory_utils.validate_sun_avoidance
   :no-index:

**Sun avoidance validation**::

    from fyst_trajectories import get_fyst_site, validate_sun_avoidance
    from fyst_trajectories.trajectory_utils import get_absolute_times

    site = get_fyst_site()
    # trajectory must have start_time set
    abs_times = get_absolute_times(trajectory)
    validate_sun_avoidance(site, trajectory.az, trajectory.el, abs_times)

.. note::

   ``validate_sun_avoidance`` is advisory only -- it emits warnings but
   never raises exceptions.  Telescope control systems must enforce their
   own hard sun-avoidance limits independently.

.. note::

   The high-level ``validate_trajectory()`` function calls both of these
   internally and is the recommended API for most use cases.
