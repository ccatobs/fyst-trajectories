Coordinate Systems
==================

fyst-trajectories supports celestial and horizontal coordinate systems via astropy,
with ``FRAME_ALIASES`` for compatibility with telescope control systems.

All transforms are vacuum (unrefracted) by default: refraction is
applied downstream at execution time, so ``Coordinates(site)`` emits
geometric coordinates unless an atmosphere is passed explicitly. See
:ref:`quickstart-planning-refraction` for when (and when not) to
enable refraction.

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
:meth:`~fyst_trajectories.coordinates.Coordinates.radec_to_altaz` /
:meth:`~fyst_trajectories.coordinates.Coordinates.altaz_to_radec`.
``GALACTIC`` and ``ECLIPTIC`` are intentionally not aliased: those frames use
``l``/``b`` and ``lon``/``lat`` and would raise in the transform methods.
For a star whose proper motion has moved it by more than the beam since
the catalogue epoch, use
:meth:`~fyst_trajectories.coordinates.Coordinates.radec_to_altaz_with_pm`,
which propagates the catalogue position to the observation time first;
:doc:`quickstart` runs it on Barnard's Star.

.. [#j2k] ``J2000`` is a label of convenience: this library maps it to
   ``icrs``, but ICRS and FK5(J2000) differ at the tens-of-mas level: the
   FK5 equinox sits -22.9 +/- 2.3 mas from the ICRS right-ascension origin
   (IERS Conventions 2010, TN36 section 2.1.2). Sub-arcsecond catalogue
   work should use ``FK5`` if the
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
- ``trajectory.metadata.input_frame``: ``"icrs"`` for the RA/Dec patterns
  (pong, daisy, sidereal), ``None`` for every AltAz-frame pattern, which
  includes planet and satellite tracking (no other value is produced)

::

    from astropy.time import Time

    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.patterns import PongScanConfig, TrajectoryBuilder

    # Use a specific time when target is observable
    start_time = Time("2026-03-15T01:00:00", scale="utc")

    trajectory = (
        TrajectoryBuilder(get_fyst_site())
        .at(ra=180.0, dec=-30.0)  # Input in ICRS
        .with_config(PongScanConfig(
            timestep=0.1, width=2.0, height=2.0,
            spacing=0.1, velocity=0.4, num_terms=4, angle=0.0,
        ))
        .duration(300.0)
        .starting_at(start_time)
        .build()
    )

    print(trajectory.coordsys)            # "altaz"
    print(trajectory.metadata.input_frame) # "icrs"

Field Rotation vs. Focal Plane Rotation
----------------------------------------

``Coordinates.get_field_rotation()`` returns the **celestial-frame**
orientation of the focal plane
(``nasmyth_sign * elevation + parallactic_angle``, no instrument
rotation), the quantity needed for sky-map orientation, image rotation and
polarization angles. The az/el projections use the mechanical
(horizon-frame) rotation instead,
``nasmyth_sign * elevation + instrument_rotation``, and
``compute_focal_plane_rotation()`` computes either frame. See
:doc:`instrument_offsets` for the distinction and its usage.

.. note::

   Sources whose declination is close to the site latitude
   (``|dec - lat| < 5°``) transit very near the zenith, where the
   parallactic-angle *rate* diverges. FYST's lat = -22.99° puts dec
   -18° to -28° in that band. See
   :meth:`~fyst_trajectories.coordinates.Coordinates.get_parallactic_angle` Notes
   for how fast the angle swings across that band.
