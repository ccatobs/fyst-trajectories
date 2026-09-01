Sun-Avoidance Models
====================

Task-oriented walkthrough: :doc:`../sun_avoidance`. This page is the
module reference for :mod:`fyst_trajectories.sun_models`.

:func:`~fyst_trajectories.sun_models.make_sun_safe` builds the predicate
that every ``sun_safe=`` seam accepts;
:func:`~fyst_trajectories.sun_models.make_slew_safe` lifts a point model
to the path level for dispatch. The ``"cone"`` and ``"cad"`` models bind
the shared `ccatobs/sun-avoidance
<https://github.com/ccatobs/sun-avoidance>`_ library (an optional git
dependency; install instructions in :doc:`../sun_avoidance`), and only
its *point geometry* is bound (``calc_sun_distance`` +
``get_mask_fixed_pos``); Sun positions, site constants, and kinematics
stay fyst-trajectories'.

.. automodule:: fyst_trajectories.sun_models
   :members:
   :undoc-members:
   :show-inheritance:

Vectorized use
--------------

Every model exposes the optional ``batch`` / ``threshold`` extension
(grids and trajectories; consumers with a time grid use it
automatically)::

    import numpy as np
    from astropy.time import Time

    from fyst_trajectories.sun_models import make_sun_safe

    cad = make_sun_safe("cad")
    t = Time("2026-11-15T16:00:00", scale="utc")
    az = np.linspace(0.0, 355.0, 72)
    el = np.full_like(az, 45.0)
    verdicts = cad.batch(az, el, t)            # ndarray[bool]
    min_separation = cad.threshold(az, el, t)  # deg, directional

Path-level slew safety
----------------------

:func:`~fyst_trajectories.sun_models.make_slew_safe` sweeps any point
model along the direct trapezoidal slew path (the FYST axis
velocity/acceleration limits, the Sun advanced along the motion) to build
a :class:`~fyst_trajectories.dispatch.SlewSafePredicate` for
:func:`~fyst_trajectories.dispatch.choose_encoder_solution`. When no wrap
has a clear direct path, dispatch **raises** (by design: reject, never
auto-reroute); the caller may then plan a two-leg detour explicitly::

    from astropy.time import Time

    from fyst_trajectories.sun_models import find_sun_safe_detour, make_slew_safe

    t = Time("2026-11-15T20:30:00", scale="utc")
    detour = find_sun_safe_detour(180.0, 45.0, 260.0, 45.0, t, make_slew_safe("cad"))
    # (az_mid, el_mid) with el_mid inside the telescope limits, or None.

``find_sun_safe_detour`` bounds every candidate elevation to the
commandable range, and the honest answer is often ``None`` under FYST's
own policies; see :doc:`../sun_avoidance` for why, and for the wrap
choice that is usually the real recourse.
