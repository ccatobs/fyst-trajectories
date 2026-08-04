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
    stacklevel_offset: int = 0,
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
        the directional sun-avoidance model (see
        :func:`~fyst_trajectories.sun_models.make_sun_safe`) is honored
        end-to-end. See :class:`~fyst_trajectories.dispatch.SunSafePredicate`.
    stacklevel_offset : int, optional
        Added to the base ``stacklevel=2`` of the emitted warning so a wrapper
        that calls this check one frame deeper (for example
        :func:`_check_altaz_center_sun_safety`) still attributes the warning to
        the originating planner module. Default is 0, which direct callers use.

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
                stacklevel=2 + stacklevel_offset,
            )
    elif not sun_safe(float(az), float(el), start_time):
        warnings.warn(
            f"EXCLUSION ZONE: Field center at (az={float(az):.1f}\u00b0, "
            f"el={float(el):.1f}\u00b0) is inside the Sun avoidance zone at "
            f"{start_time.iso}. The telescope hardware may refuse this trajectory.",
            PointingWarning,
            stacklevel=2 + stacklevel_offset,
        )


def _check_altaz_center_sun_safety(
    *,
    site: Site,
    az_center: float,
    el_center: float,
    start_time: Time,
    sun_safe: SunSafePredicate | None = None,
) -> None:
    """Sun-safety pre-flight for a fixed AltAz-center scan.

    The AltAz planners fix a horizon-frame center, but the shared field
    check works in RA/Dec, so convert the center to RA/Dec at ``start_time``
    and defer to :func:`_check_field_sun_safety`. Passing
    ``stacklevel_offset=1`` keeps any warning attributed to the calling
    planner module, matching the celestial planners' direct calls.

    Parameters
    ----------
    site : Site
        Site configuration with sun avoidance settings.
    az_center, el_center : float
        Azimuth and elevation of the fixed pattern center in degrees.
    start_time : Time
        Observation start time.
    sun_safe : SunSafePredicate, optional
        Forwarded to :func:`_check_field_sun_safety`; see that function.

    Warns
    -----
    PointingWarning
        If the converted center is within the sun exclusion radius (default)
        or the injected ``sun_safe`` predicate reports it unsafe.
    """
    # Sun-safety pre-flight works in RA/Dec (like the other planners), so
    # convert the fixed AltAz center to RA/Dec at the start time. Vacuum
    # (default) Coordinates matches the geometry the trajectory is built in.
    coords = Coordinates(site)
    ra_center, dec_center = coords.altaz_to_radec(az_center, el_center, start_time)
    _check_field_sun_safety(
        float(ra_center),
        float(dec_center),
        start_time,
        site,
        sun_safe=sun_safe,
        stacklevel_offset=1,
    )
