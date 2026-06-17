Dispatch-Time Helpers
=====================

Turn a goal sky position into a concrete, sun-safe encoder command at *dispatch*
time -- just before a scan task slews to its start point.

.. automodule:: fyst_trajectories.dispatch
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

Read the telescope's current encoder position from the live broadcast, then
choose the sun-safe azimuth-wrap / encoder ``(az, el)`` to slew to for a scan's
first sample::

    from astropy.time import Time
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.dispatch import choose_encoder_solution

    site = get_fyst_site()
    obstime = Time("2026-03-15T18:00:00", scale="utc")

    enc_az, enc_el = choose_encoder_solution(
        current_az=120.0,   # from the 200 Hz position broadcast
        current_el=45.0,
        goal_az=200.0,      # first sample of the scan trajectory
        goal_el=50.0,
        obstime=obstime,
        site=site,
    )
    # command the slew to (enc_az, enc_el), then POST the trajectory.

The sun-safety test is injectable via ``sun_safe`` (a
:class:`~fyst_trajectories.dispatch.SunSafePredicate`), defaulting to the scalar
exclusion check -- a directional sun-avoidance model can be supplied without
changing the call sites.
