Coordinate Transformations
==========================

Coordinate transformations between RA/Dec and Az/El, with solar system
ephemeris, sidereal time, hour angle, parallactic angle, field rotation,
and sun-safety checks.

.. automodule:: fyst_trajectories.coordinates
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: SOLAR_SYSTEM_BODIES, SATELLITE_BODIES, FRAME_ALIASES

Data Catalogs
-------------

.. autodata:: fyst_trajectories.coordinates.SOLAR_SYSTEM_BODIES

.. autodata:: fyst_trajectories.coordinates.SATELLITE_BODIES

.. autodata:: fyst_trajectories.coordinates.FRAME_ALIASES
   :no-value:

The full alias table, with the frames that are deliberately *not*
aliased, is documented in :doc:`../coordinate_systems`.

Usage Examples
--------------

**Basic transformation** (vacuum; refraction is applied downstream)::

    from astropy.time import Time

    from fyst_trajectories import Coordinates, get_fyst_site

    coords = Coordinates(get_fyst_site())

    obstime = Time("2026-03-15T04:00:00", scale="utc")
    az, el = coords.radec_to_altaz(ra=83.633, dec=22.014, obstime=obstime)

**Observational parameters**::

    lst = coords.get_lst(obstime)
    ha = coords.get_hour_angle(ra=180.0, obstime=obstime)
    pa = coords.get_parallactic_angle(ra=180.0, dec=-30.0, obstime=obstime)
    # Celestial-frame field rotation (nasmyth_sign * el + pa, no instrument rotation)
    fr = coords.get_field_rotation(ra=180.0, dec=-30.0, obstime=obstime)

.. note::

   ``get_field_rotation()`` returns the *celestial-frame* orientation,
   ``nasmyth_sign * elevation + parallactic angle``, using the Nasmyth port
   from the site configuration: map orientation, image rotation and
   polarization angles. The rotation the az/el projections use is the
   *mechanical* one, ``nasmyth_sign * elevation + instrument_rotation``,
   returned by ``compute_focal_plane_rotation()`` from :doc:`offsets` (its
   ``parallactic_angle`` argument defaults to 0). The parallactic angle does
   not belong in an az/el projection.

**Solar system bodies**::

    obstime = Time("2026-03-15T16:00:00", scale="utc")
    az, el = coords.get_body_altaz("mars", obstime)
    ra, dec = coords.get_body_radec("jupiter", obstime)
    sun_az, sun_el = coords.get_sun_altaz(obstime)

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

``radec_to_altaz_with_pm`` propagates a catalogue position by its proper
motion first; :doc:`../quickstart` runs it on a high-proper-motion star.
