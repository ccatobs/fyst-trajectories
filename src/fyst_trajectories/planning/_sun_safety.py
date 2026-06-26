"""Pre-flight sun-safety check shared by all planner entry points."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from astropy.time import Time

from ..coordinates import Coordinates
from ..exceptions import PointingWarning
from ..site import Site

if TYPE_CHECKING:
    # Annotation-only import to avoid an import cycle: ``dispatch`` imports
    # ``coordinates``/``site``/``exceptions`` at runtime, so importing it here
    # at module level could cycle. The predicate is invoked structurally, so
    # only the type hint needs the symbol.
    from ..dispatch import SunSafePredicate


def _check_field_sun_safety(
    ra: float,
    dec: float,
    start_time: Time,
    site: Site,
    sun_safe: SunSafePredicate | None = None,
) -> None:
    """Quick pre-flight check that a field center is not near the sun.

    This is a lightweight check that warns before expensive trajectory
    generation. It never blocks trajectory generation. Violations are
    reported as warnings.

    Parameters
    ----------
    ra : float
        Right Ascension of the field center in degrees.
    dec : float
        Declination of the field center in degrees.
    start_time : Time
        Observation start time.
    site : Site
        Site configuration with sun avoidance settings.
    sun_safe : SunSafePredicate, optional
        Sun-safety predicate implementing the
        :class:`~fyst_trajectories.dispatch.SunSafePredicate` contract,
        ``(az_deg, el_deg, time) -> bool`` returning ``True`` when the field
        center is clear of the Sun. ``None`` (default) keeps the built-in
        scalar exclusion-radius check (the field center's angular separation
        from the Sun against ``site.sun_avoidance.exclusion_radius``). When a
        predicate is injected it is consulted in place of the scalar check, so
        the directional sun-avoidance model (future shared library) is honored
        end-to-end. See :class:`~fyst_trajectories.dispatch.SunSafePredicate`.

    Warns
    -----
    PointingWarning
        If the field center is within the sun exclusion radius (default) or
        the injected ``sun_safe`` predicate reports it unsafe.
    """
    if not site.sun_avoidance.enabled:
        return
    coords = Coordinates(site)
    az, el = coords.radec_to_altaz(ra, dec, start_time)
    if sun_safe is None:
        sun_az, sun_alt = coords.get_sun_altaz(start_time)
        sep = coords.angular_separation(az, el, sun_az, sun_alt)
        if sep <= site.sun_avoidance.exclusion_radius:
            warnings.warn(
                f"EXCLUSION ZONE: Field center passes {sep:.1f}\u00b0 from the Sun "
                f"(exclusion radius: {site.sun_avoidance.exclusion_radius}\u00b0) "
                f"at {start_time.iso}. The telescope hardware may refuse this trajectory.",
                PointingWarning,
                stacklevel=2,
            )
    elif not sun_safe(float(az), float(el), start_time):
        warnings.warn(
            f"EXCLUSION ZONE: Field center at (az={float(az):.1f}\u00b0, "
            f"el={float(el):.1f}\u00b0) is inside the Sun avoidance zone at "
            f"{start_time.iso}. The telescope hardware may refuse this trajectory.",
            PointingWarning,
            stacklevel=2,
        )
