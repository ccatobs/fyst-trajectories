Observability Reporting
=======================

The observability module answers a single operational question — *"which of
these targets can we observe now (or over the next* N *hours), and why not?"* —
for a list of solar-system flux calibrators. It is the **assess-only** sibling
of the trajectory builders
(:class:`~fyst_trajectories.patterns.PlanetTrackPattern`, ``plan_*_scan``):
:func:`~fyst_trajectories.observability.check_observability` returns a per-target
:class:`~fyst_trajectories.observability.ObservabilityReport`, never builds a
trajectory, and **never raises** for an unobservable target — the reason is
reported, not excepted.

It is importable in isolation: it depends only on
:mod:`fyst_trajectories.coordinates` and :mod:`fyst_trajectories.site` and does
**not** import the offline ``overhead`` simulator.

Two physically distinct kinds of avoidance are kept structurally separate:

- **Sun** — always-on thermal/hardware safety, read from
  ``site.sun_avoidance`` (45° by default). Reported in the dedicated
  ``sun_separation_deg`` / ``sun_clear`` fields with a
  :attr:`~fyst_trajectories.observability.ReasonCode.SUN_TOO_CLOSE` reason. It is
  never an :class:`~fyst_trajectories.observability.AvoidZone` and cannot be
  weakened via the ``avoid`` list.
- **Bright-source contamination** — caller-specified, variable, per-body
  exclusion zones (the Moon, Jupiter, …). Reported as
  :class:`~fyst_trajectories.observability.AvoidSeparation` entries with an
  :attr:`~fyst_trajectories.observability.ReasonCode.AVOID_TOO_CLOSE` reason.
  There are **no library default zones**: every
  :class:`~fyst_trajectories.observability.AvoidZone` carries its own
  caller-supplied radius, so ``AvoidZone("moon")`` is a ``TypeError``.

.. note::

   The orchestration that turns ``schedule(OBSERVE=[...], AVOID=[...])`` into
   selected, sequenced observing blocks lives **one layer up** — in the
   scheduling layer, not here. This module supplies the stateless
   observability primitive that the scheduler calls; it does not select,
   sequence, or trim blocks.

.. automodule:: fyst_trajectories.observability
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

**Instant verdict** (``horizon_hours=0``, the default — evaluate at one time)::

    from astropy.time import Time

    from fyst_trajectories.observability import check_observability

    reports = check_observability(
        ["uranus", "neptune", "mars"],
        Time("2026-06-15T05:00:00", scale="utc"),
    )
    for r in reports:
        print(r.summary)            # one-line human-readable verdict
    observable_now = [r.name for r in reports if r.observable]

**With a query horizon** (find the first observable window within *N* hours)::

    reports = check_observability(
        ["uranus", "neptune"],
        Time("2026-06-15T05:00:00", scale="utc"),
        horizon_hours=24.0,
    )
    for r in reports:
        if r.window is not None:
            print(r.name, r.window.start.iso, "→", r.window.end.iso,
                  f"({r.window.duration_hours:.1f} h)")

**With caller-specified bright-source avoidance** (the AVOID list)::

    from fyst_trajectories.observability import AvoidZone, check_observability

    reports = check_observability(
        ["uranus", "neptune", "mars"],
        Time("2026-06-15T05:00:00", scale="utc"),
        avoid=[AvoidZone("jupiter", 3.0), AvoidZone("moon", 5.0)],
    )
    # A scheduler wrapper parsing ('jupiter', '3deg') pairs can use
    # AvoidZone.from_pair(("jupiter", "3deg")) instead.

**Satellite targets** (Titan). Titan is carried in the built-in
:data:`~fyst_trajectories.observability.FLUX_CALIBRATORS` catalog as a
``SATELLITE`` proxied by its parent body (Saturn). Its observability position is
the parent-body position (≲ ~3 arcmin), which is more than adequate for an
up/down/sun-safe verdict and is flagged on the report::

    reports = check_observability(
        ["titan"],
        Time("2026-06-15T05:00:00", scale="utc"),
    )
    assert reports[0].position_approximate   # Saturn-proxy position

.. note::

   Visibility (*is Titan up and sun-safe?*) needs **no** ephemeris kernel — the
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
