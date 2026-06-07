Coordinate Transformations
==========================

Coordinate transformation utilities with solar system ephemeris.
``Coordinates(site)`` defaults to vacuum (geometric) coordinates --
this is the correct default for trajectory generation because the
FYST ACU applies atmospheric refraction downstream. Pass
:meth:`~fyst_trajectories.site.AtmosphericConditions.for_fyst` for
planning and simulation (visibility calculations, observability
checks) where the output is NOT sent to the ACU.

.. automodule:: fyst_trajectories.coordinates
   :members:
   :undoc-members:
   :show-inheritance:

Frame Aliases
-------------

+-------------------+----------------------------+
| Alias             | Astropy Frame              |
+===================+============================+
| ``J2000`` [#j2k]_ | ``icrs``                   |
+-------------------+----------------------------+
| ``FK5``           | ``fk5``                    |
+-------------------+----------------------------+
| ``B1950``         | ``fk4``                    |
+-------------------+----------------------------+
| ``HORIZON``       | ``altaz``                  |
+-------------------+----------------------------+

Only spherical RA/Dec frames (``J2000``/``FK5``/``B1950``) work with the
``radec_to_altaz`` / ``altaz_to_radec`` transform methods; ``GALACTIC`` and
``ECLIPTIC`` are intentionally not aliased (they use ``l``/``b`` and
``lon``/``lat`` and would raise).

.. [#j2k] ``J2000`` is a label of convenience: this library maps it to
   ``icrs``, but ICRS and FK5(J2000) differ by ~22 mas (the IAU 1997
   alignment). Sub-arcsecond catalogue work should use ``FK5`` if the
   inputs are FK5 J2000.0; for telescope pointing the offset is well
   below the beam and is harmless.

::

    from fyst_trajectories import FRAME_ALIASES, normalize_frame

    normalize_frame("J2000")    # "icrs"
    normalize_frame("B1950")    # "fk4"

Usage Examples
--------------

**Basic transformation** (vacuum -- ACU applies refraction)::

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

.. note::

   Planetary satellites (e.g. Titan) are addressed separately, listed in the
   public ``SATELLITE_BODIES`` tuple. Unlike the builtin bodies they require a JPL
   satellite SPK kernel, supplied via ``Coordinates(satellite_kernel=...)`` or the
   ``FYST_SATELLITE_KERNEL`` environment variable (the optional ``[ephemeris]``
   extra). See :class:`fyst_trajectories.patterns.SatelliteTrackConfig`.

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
