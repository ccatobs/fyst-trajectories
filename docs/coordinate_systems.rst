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

Only RA/Dec frames are usable with
:meth:`~fyst_trajectories.coordinates.Coordinates.radec_to_altaz` /
:meth:`~fyst_trajectories.coordinates.Coordinates.altaz_to_radec`: the
three RA/Dec aliases above, or an astropy RA/Dec frame name such as
``icrs`` (the default) or ``gcrs``; ``HORIZON`` is refused by name.
``GALACTIC`` and ``ECLIPTIC`` are intentionally not aliased: those frames use
``l``/``b`` and ``lon``/``lat`` and would raise in the transform methods.
For a star whose proper motion has moved it by more than the beam since
the catalogue epoch, use
:meth:`~fyst_trajectories.coordinates.Coordinates.radec_to_altaz_with_pm`,
which propagates the catalogue position to the observation time first;
:doc:`quickstart` runs it on Barnard's Star.

.. [#j2k] ``J2000`` is a label of convenience: this library maps it to
   ``icrs``, but ICRS and FK5(J2000) differ at the tens-of-mas level: the
   FK5 equinox sits -22.9 ± 2.3 mas from the ICRS right-ascension origin
   (IERS Conventions 2010, TN36 section 2.1.2). Catalogue work at the
   milliarcsecond level should use ``FK5`` if the inputs are FK5 J2000.0;
   for telescope pointing the offset is well below the beam and is
   harmless.

**Usage**::

    from fyst_trajectories import FRAME_ALIASES, normalize_frame

    # Case-insensitive lookup
    astropy_frame = normalize_frame("J2000")    # Returns "icrs"
    astropy_frame = normalize_frame("b1950")    # Returns "fk4"

    # Unknown frames are lowercased for astropy compatibility
    astropy_frame = normalize_frame("MyFrame")  # Returns "myframe"

Trajectory Coordinate Fields
----------------------------

A pattern-generated trajectory is always in Az/El, whatever frame its
centre was given in. The input frame is recorded in
``trajectory.metadata.input_frame``: ``"icrs"`` for the RA/Dec patterns
(pong, daisy, sidereal), ``None`` for every AltAz-frame pattern, which
includes planet and satellite tracking (no other value is produced).

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

    print(trajectory.metadata.input_frame) # "icrs"

Field Rotation vs. Focal Plane Rotation
----------------------------------------

``Coordinates.get_field_rotation()`` returns the **celestial-frame**
orientation of the focal plane,
``nasmyth_sign * elevation + parallactic_angle``, with no instrument
rotation: the quantity needed for sky-map orientation, image rotation and
polarization angles. It is not the rotation the az/el projections use.
See :doc:`instrument_offsets` for the mechanical/celestial decomposition
and ``compute_focal_plane_rotation()``, which computes either frame.

.. note::

   Sources whose declination is within 5° of the site latitude transit
   within 5° of the zenith, where the parallactic angle swings through
   180° during transit, faster the closer the transit is to the zenith.
   FYST's latitude of -22.99° puts declinations from -18° to -28° in that
   band. See
   :meth:`~fyst_trajectories.coordinates.Coordinates.get_parallactic_angle` Notes
   for the rate.

.. _earth-orientation-data:

Earth-orientation data
----------------------

Every transform between the sky and the horizon needs UT1-UTC and polar
motion, which astropy reads from its IERS table
(:mod:`astropy.utils.iers`). An error of one second in UT1-UTC turns the
sky about the Earth's axis by 15.04 arcsec, which moves a source at
declination δ by 15.04 cos δ arcsec.

By default astropy's table refreshes itself: when a transform needs predicted
values and the table's predictions are more than 30 days old, astropy downloads
a new table from the IERS servers. On a host that cannot reach them, astropy
falls back to a table it already holds, and what follows depends on its version:

- Astropy 7 and later fall back to the IERS-A table bundled in the
  ``astropy-iers-data`` package. Once that table's predictions are more than
  30 days old, a transform that needs them raises astropy's ``ValueError``
  ("interpolating from IERS_Auto using predictive values that are more than
  30.0 days old").
- Astropy 6 and earlier fall back to an IERS-B table, which holds no
  predictions: a transform of the present warns and takes UT1-UTC as zero and
  polar motion as its 50-year mean. With an earlier download in its cache,
  astropy uses that table and raises the same ``ValueError`` once it is stale.

Such a host also waits on downloads that fail, each attempt up to its network
timeout: every version tries again at each transform once its table is stale,
and astropy 7 and earlier also while they have no downloaded table.

:func:`~fyst_trajectories.coordinates.check_earth_orientation_data` raises
:class:`~fyst_trajectories.exceptions.PointingError` when the active table does
not cover the span a caller will plan in, or astropy refuses its predictions
there. Called before the first plan, it also has astropy load the table, and
download a new one if needed, at that point::

    from astropy.time import Time

    from fyst_trajectories import check_earth_orientation_data

    start = Time("2026-03-15T04:00:00", scale="utc")  # for example Time.now()
    covered_until = check_earth_orientation_data(start, days=2.0)
    print(covered_until > start)  # True

Astropy judges the age of its predictions by the host's clock at each
transform, so a table that passes the check can be refused later.

A host that cannot reach the IERS servers has two ways to give astropy a table:

- A mirror of the IERS ``finals2000A.all`` file that the host can reach, named
  in ``iers.conf.iers_auto_url`` (in code, or in astropy's configuration file).
  Astropy then refreshes its table from the mirror as it does from the IERS
  servers.
- A table set by hand (with ``from astropy.utils import iers``):
  ``iers.conf.auto_download = False``, then
  ``iers.earth_orientation_table.set(iers.IERS_A.open(path))``, with ``path``
  such a file; astropy 6 and later bundle one, which ``iers.IERS_A.open()``
  reads when given no path. Astropy makes no download attempt for a table set
  by hand and does not check its age; its predictions lose accuracy as they age.
