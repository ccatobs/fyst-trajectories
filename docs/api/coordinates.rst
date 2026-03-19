Coordinate Transformations
==========================

Coordinate transformation utilities with solar system ephemeris.
Atmospheric refraction is applied only when an explicit
``AtmosphericConditions`` is provided; the default is vacuum (no refraction).

.. automodule:: fyst_trajectories.coordinates
   :members:
   :undoc-members:
   :show-inheritance:

Frame Aliases
-------------

+---------------+----------------------------+
| Alias         | Astropy Frame              |
+===============+============================+
| ``J2000``     | ``icrs``                   |
+---------------+----------------------------+
| ``FK5``       | ``fk5``                    |
+---------------+----------------------------+
| ``B1950``     | ``fk4``                    |
+---------------+----------------------------+
| ``GALACTIC``  | ``galactic``               |
+---------------+----------------------------+
| ``ECLIPTIC``  | ``geocentrictrueecliptic`` |
+---------------+----------------------------+
| ``HORIZON``   | ``altaz``                  |
+---------------+----------------------------+

::

    from fyst_trajectories import FRAME_ALIASES, normalize_frame

    normalize_frame("J2000")    # "icrs"
    normalize_frame("galactic") # "galactic"

Usage Examples
--------------

**Basic transformation**::

    from astropy.time import Time

    from fyst_trajectories import Coordinates, get_fyst_site

    coords = Coordinates(get_fyst_site())

    obstime = Time("2026-03-15T04:00:00", scale="utc")
    az, el = coords.radec_to_altaz(ra=83.633, dec=22.014, obstime=obstime)

**Observational parameters**::

    lst = coords.get_lst(obstime)
    ha = coords.get_hour_angle(ra=180.0, obstime=obstime)
    pa = coords.get_parallactic_angle(ra=180.0, dec=-30.0, obstime=obstime)
    # Simplified field rotation (nasmyth_sign * el + pa, no instrument rotation)
    fr = coords.get_field_rotation(ra=180.0, dec=-30.0, obstime=obstime)

.. note::

   ``get_field_rotation()`` returns ``nasmyth_sign * elevation + parallactic_angle``
   using the Nasmyth port from the site configuration. For the full focal-plane
   rotation (including instrument rotation), use
   ``compute_focal_plane_rotation()`` from :doc:`offsets`.

**Solar system bodies**::

    obstime = Time("2026-03-15T16:00:00", scale="utc")
    az, el = coords.get_body_altaz("mars", obstime)
    ra, dec = coords.get_body_radec("jupiter", obstime)
    sun_az, sun_el = coords.get_sun_altaz(obstime)

.. note::

   The list of supported solar system bodies is available as
   ``SOLAR_SYSTEM_BODIES``::

       from fyst_trajectories import SOLAR_SYSTEM_BODIES
       print(SOLAR_SYSTEM_BODIES)
       # ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn', ...]

AltAzCoord
~~~~~~~~~~

``AltAzCoord`` is a frozen dataclass holding ``az`` and ``alt`` fields
in degrees.  ``el`` is available as a property alias for ``alt``.
It is returned by several :class:`Coordinates` helper methods::

    from fyst_trajectories import AltAzCoord

    coord = AltAzCoord(az=180.0, alt=45.0)
    print(coord.az, coord.el)  # .el is an alias for .alt

**Safety checks**::

    obstime = Time("2026-03-15T04:00:00", scale="utc")
    observable, reason = coords.is_position_observable(az=180, el=45, obstime=obstime)
    is_safe = coords.is_sun_safe(az=180, el=45, obstime=obstime)

**Proper motion** (for high PM stars)::

    az, el = coords.radec_to_altaz_with_pm(
        ra=269.452, dec=4.693,
        pm_ra=-798.58, pm_dec=10328.12,  # mas/yr
        ref_epoch=Time("J2015.5"),
        obstime=Time("2025-06-15T04:00:00", scale="utc"),
        distance=1.8,  # parsecs
    )
