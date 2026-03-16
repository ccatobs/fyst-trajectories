Site Configuration
==================

.. automodule:: fyst_pointing.site
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: fyst_pointing.get_fyst_site

Overriding Defaults
-------------------

``get_fyst_site()`` accepts keyword arguments to override Tier 3
parameters (sun avoidance).  Tier 1 and Tier 2 parameters (location,
optics, mechanical limits) are fixed constants::

    from fyst_pointing import get_fyst_site

    # Default FYST site
    site = get_fyst_site()

    # Override sun avoidance radius (default is 45 degrees)
    site_custom = get_fyst_site(sun_exclusion_radius=30.0)

    # Disable sun avoidance entirely (for testing)
    site_no_sun = get_fyst_site(sun_avoidance_enabled=False)

Convenience Constants
---------------------

.. py:data:: fyst_pointing.FYST_LOCATION

   Pre-computed :class:`~astropy.coordinates.EarthLocation` for the FYST
   telescope.  Equivalent to ``get_fyst_site().location``.  Useful for
   quick calculations where a full :class:`Site` object is not needed.

   ::

       from fyst_pointing import FYST_LOCATION

       print(FYST_LOCATION.lat)   # -22d59m08.3004s
       print(FYST_LOCATION.lon)   # -67d44m25.0008s
