Observability Reporting
=======================

Assess-only visibility for solar-system flux calibrators and fixed sources,
plus the site's sunrise/sunset/twilight almanac. Nothing here builds a
trajectory or raises for an unobservable target: the reason is reported.
Jump to `Usage Examples`_ for the common calls.

.. automodule:: fyst_trajectories.observability
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: FLUX_CALIBRATORS

Data Catalogs
-------------

.. autodata:: fyst_trajectories.observability.FLUX_CALIBRATORS
   :no-value:

Usage Examples
--------------

**Instant verdict** (``horizon_hours=0``, the default, evaluate at one time)::

    from astropy.time import Time

    from fyst_trajectories.observability import check_observability

    reports = check_observability(
        ["uranus", "neptune", "mars"],
        Time("2026-06-15T05:00:00", scale="utc"),
    )
    for r in reports:
        print(r.summary)            # one-line human-readable verdict
    observable_now = [r.name for r in reports if r.observable]

**With a query horizon** (find every observable window within *N* hours)::

    reports = check_observability(
        ["uranus", "neptune"],
        Time("2026-06-15T05:00:00", scale="utc"),
        horizon_hours=24.0,
    )
    for r in reports:
        for w in r.windows or ():
            print(r.name, w.start.iso, "→", w.end.iso,
                  f"({w.duration_hours:.1f} h)")
        print(r.name, "total:", f"{r.total_observable_hours:.1f} h")

A target can have several disjoint windows in one horizon (an evening pass
and a pre-dawn pass are common); ``windows`` reports all of them in time
order. It is ``None`` when no horizon was requested (``horizon_hours=0``)
and an empty tuple when a horizon was evaluated and no window exists.

**With caller-specified bright-source avoidance** (the AVOID list)::

    from fyst_trajectories.observability import AvoidZone, check_observability

    reports = check_observability(
        ["uranus", "neptune", "mars"],
        Time("2026-06-15T05:00:00", scale="utc"),
        avoid=[AvoidZone("jupiter", 3.0), AvoidZone("moon", 5.0)],
    )
    # A scheduler wrapper parsing ('jupiter', '3deg') pairs can use
    # AvoidZone.from_pair(("jupiter", "3deg")) instead.

**Satellite targets** (Titan). A ``SATELLITE`` in the built-in
:data:`~fyst_trajectories.observability.FLUX_CALIBRATORS` catalog is evaluated
at its parent body's position (Titan at Saturn, within ~3 arcmin), which is
ample for an up/down/sun-safe verdict and is flagged on the report::

    reports = check_observability(
        ["titan"],
        Time("2026-06-15T05:00:00", scale="utc"),
    )
    assert reports[0].position_approximate   # Saturn-proxy position

.. note::

   Visibility (*is Titan up and sun-safe?*) needs **no** ephemeris kernel: the
   Saturn proxy is computed from astropy's built-in ephemeris and works offline.
   Arcsecond-accurate **tracking** (pointing at Titan) is a separate concern
   handled by :class:`~fyst_trajectories.patterns.SatelliteTrackConfig`, which
   does require a JPL satellite SPK kernel. See :doc:`coordinates` and
   :doc:`patterns`.

**Fixed RA/Dec sources and bodies outside the catalog**::

    from fyst_trajectories.observability import Target, TargetKind, check_observability

    extra = {
        "3c279": Target("3c279", TargetKind.FIXED, ra_deg=194.046, dec_deg=-5.789),
    }
    reports = check_observability(
        ["3c279"],
        Time("2026-06-15T05:00:00", scale="utc"),
        extra_targets=extra,
    )

**Sunrise, sunset, and twilight**
(:func:`~fyst_trajectories.observability.sun_events`)::

    from fyst_trajectories.observability import SunEventKind, sun_events

    events = sun_events(Time("2026-11-15T16:00:00", scale="utc"))
    sunset = next(e for e in events if e.kind == SunEventKind.SUNSET)
    dusk_side = [e.kind.value for e in events if not e.rising]
