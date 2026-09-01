Site Configuration
==================

FYST physical constants, the frozen :class:`~fyst_trajectories.site.Site`
dataclass, telescope limits, and atmospheric conditions.

.. automodule:: fyst_trajectories.site
   :members:
   :undoc-members:
   :show-inheritance:

Overriding Defaults
-------------------

Only the sun-avoidance parameters are overridable; location, optics, and
mechanical limits are fixed constants::

    from fyst_trajectories import get_fyst_site

    # Default FYST site
    site = get_fyst_site()

    # Override sun avoidance radius (default is 45 degrees)
    site_custom = get_fyst_site(sun_exclusion_radius=30.0)

    # Disable sun avoidance entirely (for testing)
    site_no_sun = get_fyst_site(sun_avoidance_enabled=False)

The scalar radii configure one of three selectable Sun policies; see
:doc:`../sun_avoidance` for the directional alternatives.

Convenience Constants
---------------------

.. py:currentmodule:: fyst_trajectories

.. py:data:: FYST_LOCATION

   Pre-computed :class:`~astropy.coordinates.EarthLocation` for the FYST
   telescope.  Equivalent to ``get_fyst_site().location``.  Useful for
   quick calculations where a full :class:`~fyst_trajectories.site.Site`
   object is not needed.

   ::

       from fyst_trajectories import FYST_LOCATION

       print(FYST_LOCATION.lat)   # -22d59m08.3004s
       print(FYST_LOCATION.lon)   # -67d44m25.0008s
