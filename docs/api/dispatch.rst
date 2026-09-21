Dispatch-Time Helpers
=====================

Turn a goal sky position into a concrete, sun-safe encoder command at *dispatch*
time, just before a scan task slews to its start point.

.. automodule:: fyst_trajectories.dispatch
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Read the telescope's current encoder position from the live broadcast, then
choose the sun-safe azimuth-wrap / encoder ``(az, el)`` to slew to for a scan's
first sample::

    from astropy.time import Time
    from fyst_trajectories import get_fyst_site, rewrap_trajectory_azimuth
    from fyst_trajectories.dispatch import choose_encoder_solution

    site = get_fyst_site()
    obstime = Time("2026-03-15T18:00:00", scale="utc")

    solution = choose_encoder_solution(
        current_az=120.0,   # from the live ACU position broadcast
        current_el=45.0,
        goal_az=200.0,      # first sample of the scan trajectory
        goal_el=50.0,
        obstime=obstime,
        site=site,
    )
    # Shift the whole trajectory onto the chosen wrap, then command the
    # slew to (solution.az, solution.el) and POST the shifted trajectory.
    commanded = rewrap_trajectory_azimuth(trajectory, solution.az_shift)

The Sun test is injectable at two levels: ``sun_safe`` judges the goal point
(default: the site's scalar exclusion radius; a position exactly at the
exclusion radius counts as unsafe), and ``slew_safe`` judges the direct slew
path to it (default: no path check). Build either from
:func:`~fyst_trajectories.sun_models.make_sun_safe` /
:func:`~fyst_trajectories.sun_models.make_slew_safe` to run FYST's directional
model instead, without touching a call site. See :doc:`sun_models`.
