Coordinate Systems
==================

fyst-trajectories supports celestial and horizontal coordinate systems via astropy,
with ``FRAME_ALIASES`` for compatibility with telescope control systems.

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

Only spherical RA/Dec frames (``J2000``/``FK5``/``B1950``) are usable with
:meth:`~fyst_trajectories.Coordinates.radec_to_altaz` /
:meth:`~fyst_trajectories.Coordinates.altaz_to_radec`. ``GALACTIC`` and
``ECLIPTIC`` are intentionally not aliased: those frames use ``l``/``b`` and
``lon``/``lat`` and would raise in the transform methods.

.. [#j2k] ``J2000`` is a label of convenience: this library maps it to
   ``icrs``, but ICRS and FK5(J2000) differ by ~22 mas (the IAU 1997
   alignment). Sub-arcsecond catalogue work should use ``FK5`` if the
   inputs are FK5 J2000.0; for telescope pointing the offset is well
   below the beam and is harmless.

**Usage**::

    from fyst_trajectories import FRAME_ALIASES, normalize_frame

    # Case-insensitive lookup
    astropy_frame = normalize_frame("J2000")    # Returns "icrs"
    astropy_frame = normalize_frame("b1950")    # Returns "fk4"

    # Unknown frames are lowercased for astropy compatibility
    astropy_frame = normalize_frame("MyFrame")  # Returns "myframe"

Trajectory Coordinate Fields
----------------------------

Pattern-generated trajectories track coordinate provenance:

- ``trajectory.coordsys``: Always ``"altaz"`` (output is Az/El)
- ``trajectory.metadata.input_frame``: Input frame (e.g., ``"icrs"``)

::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import PongScanConfig, TrajectoryBuilder

    # Use a specific time when target is observable
    start_time = Time("2026-03-15T04:00:00", scale="utc")

    trajectory = (
        TrajectoryBuilder(get_fyst_site())
        .at(ra=180.0, dec=-30.0)  # Input in ICRS
        .with_config(PongScanConfig(
            timestep=0.1, width=2.0, height=2.0,
            spacing=0.1, velocity=0.5, num_terms=4, angle=0.0,
        ))
        .duration(300.0)
        .starting_at(start_time)
        .build()
    )

    print(trajectory.coordsys)            # "altaz"
    print(trajectory.metadata.input_frame) # "icrs"

Proper Motion
-------------

High proper motion stars are handled by ``radec_to_altaz_with_pm()``.
For trajectory generation (output sent to the ACU), bare
``Coordinates(site)`` is appropriate -- the ACU applies atmospheric
refraction downstream. For planning/simulation,
``AtmosphericConditions.for_fyst()`` can be passed instead::

    from astropy.time import Time

    from fyst_trajectories import Coordinates, get_fyst_site

    coords = Coordinates(get_fyst_site())

    # Barnard's Star (moves ~10 arcsec/year)
    az, el = coords.radec_to_altaz_with_pm(
        ra=269.452, dec=4.693,
        pm_ra=-798.58, pm_dec=10328.12,  # mas/yr (pm_ra includes cos(dec))
        ref_epoch=Time("J2015.5"),
        obstime=Time("2026-06-15T04:00:00", scale="utc"),
        distance=1.8,  # parsecs, optional
    )

Field Rotation vs. Focal Plane Rotation
----------------------------------------

``Coordinates.get_field_rotation()`` returns the sky rotation component
(``nasmyth_sign * elevation + parallactic_angle``) with no instrument rotation.
The Nasmyth sign is determined by ``site.nasmyth_port`` (+1 for Right, -1 for
Left, 0 for Cassegrain).

For the full focal-plane rotation that includes Nasmyth port sign convention
and instrument rotation, use ``compute_focal_plane_rotation()``:

    ``rotation = nasmyth_sign * elevation + instrument_rotation + parallactic_angle``

See :doc:`instrument_offsets` for details on the decomposition and usage.

.. note::

   Sources whose declination is close to the site latitude
   (``|dec − lat| < 5°``) transit very near the zenith, where the
   parallactic-angle *rate* diverges. FYST's lat = −22.99° puts sources
   with dec ≈ −18° to −28° in this regime; field rotation can swing
   through 180° in a few seconds at transit. See
   :meth:`~fyst_trajectories.coordinates.Coordinates.get_parallactic_angle` Notes
   for the full discussion.
