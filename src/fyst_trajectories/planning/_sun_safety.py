"""Pre-flight sun-safety checks shared by all planner entry points."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from astropy.time import Time

from ..coordinates import Coordinates
from ..exceptions import PointingWarning
from ..patterns.turnarounds import swept_az_envelope
from ..site import Site

if TYPE_CHECKING:
    # Annotation-only import to avoid an import cycle: ``dispatch`` imports
    # ``coordinates``/``site``/``exceptions`` at runtime, so importing it here
    # at module level could cycle. The predicate is invoked structurally, so
    # only the type hint needs the symbol.
    from ..dispatch import SunSafePredicate

# Number of times sampled along a planned constant-elevation pass for the
# sun-safety sweep. 60 samples over a typical 10-minute arc gives ~10 s
# resolution, finer than the sun's apparent motion (~15"/s) and the array's
# footprint extent; a multi-hour pass gets proportionally coarser sampling,
# still well inside the exclusion radius the check screens against.
_SUN_SAFETY_ARC_N_SAMPLES = 60


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
        the directional sun-avoidance model (see
        :func:`~fyst_trajectories.sun_models.make_sun_safe`) is honored
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
    _warn_if_center_unsafe(
        float(az),
        float(el),
        start_time,
        site,
        sun_safe=sun_safe,
        coords=coords,
        stacklevel=3,
    )


def _warn_if_center_unsafe(
    az: float,
    el: float,
    start_time: Time,
    site: Site,
    *,
    sun_safe: SunSafePredicate | None,
    coords: Coordinates,
    stacklevel: int,
) -> None:
    """Warn when a horizon position is inside the Sun zone at one instant.

    The verdict half of the field-centre pre-flight, taken on the horizon
    position the caller already holds. Callers that start from a horizon
    position use it directly; the RA/Dec entry point converts first.

    Parameters
    ----------
    az, el : float
        Position under test in degrees.
    start_time : Time
        Instant to evaluate at.
    site : Site
        Site configuration supplying the scalar exclusion radius.
    sun_safe : SunSafePredicate, optional
        Injected point model; ``None`` uses the scalar radius.
    coords : Coordinates
        Transformer for the Sun ephemeris on the scalar branch.
    stacklevel : int
        Frames out to attribute the warning to, counted from this function.
    """
    if sun_safe is None:
        sun_az, sun_alt = coords.get_sun_altaz(start_time)
        sep = coords.angular_separation(az, el, sun_az, sun_alt)
        if sep <= site.sun_avoidance.exclusion_radius:
            warnings.warn(
                f"EXCLUSION ZONE: Field center passes {sep:.1f}\u00b0 from the Sun "
                f"(exclusion radius: {site.sun_avoidance.exclusion_radius}\u00b0) "
                f"at {start_time.iso}. This violates the configured Sun avoidance "
                f"policy; nothing downstream is guaranteed to reject it.",
                PointingWarning,
                stacklevel=stacklevel,
            )
    elif not sun_safe(az, el, start_time):
        warnings.warn(
            f"EXCLUSION ZONE: Field center at (az={az:.1f}\u00b0, "
            f"el={el:.1f}\u00b0) is inside the Sun avoidance zone at "
            f"{start_time.iso}. This violates the configured Sun avoidance "
            f"policy; nothing downstream is guaranteed to reject it.",
            PointingWarning,
            stacklevel=stacklevel,
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

    The AltAz planners fix a horizon-frame center, which is already the frame
    the verdict is taken in, so the centre goes straight to the shared check.
    Any warning is attributed to the calling planner module, matching the
    celestial planners' direct calls.

    Parameters
    ----------
    site : Site
        Site configuration with sun avoidance settings.
    az_center, el_center : float
        Azimuth and elevation of the fixed pattern center in degrees.
    start_time : Time
        Observation start time.
    sun_safe : SunSafePredicate, optional
        Forwarded to the shared verdict; see :func:`_check_field_sun_safety`.

    Warns
    -----
    PointingWarning
        If the center is within the sun exclusion radius (default)
        or the injected ``sun_safe`` predicate reports it unsafe.
    """
    # The centre is already a horizon position, which is the frame the
    # verdict is taken in, so it goes straight to the check. Converting it to
    # RA/Dec first only to have the sibling convert it back cost two
    # transforms and could not change the answer. Vacuum (default)
    # Coordinates matches the geometry the trajectory is built in.
    if not site.sun_avoidance.enabled:
        return
    _warn_if_center_unsafe(
        float(az_center),
        float(el_center),
        start_time,
        site,
        sun_safe=sun_safe,
        coords=Coordinates(site),
        stacklevel=3,
    )


def _swept_arc_samples(
    *,
    az_min: np.ndarray,
    az_throw: float,
    el_deg: float,
    times: Time,
    az_speed: float,
    az_accel: float,
) -> tuple[np.ndarray, np.ndarray, Time]:
    """Build the sample grid covering a constant-elevation pass's commanded envelope.

    The scan sweeps back and forth across ``[az_min, az_min + az_throw]``
    while the mount overshoots each science edge by the quintic turnaround
    peak, so the range it actually occupies is
    :func:`~fyst_trajectories.patterns.turnarounds.swept_az_envelope` of the
    science window. This helper probes three azimuth positions per time
    sample - the two envelope edges and the midpoint - so a pass whose
    midpoint clears the Sun but whose edges do not is still caught. Sun
    motion within a single sweep is negligible (~15"/s, far less than one
    azimuth throw), so reusing the same time at all three positions is
    sound.

    Parameters
    ----------
    az_min : np.ndarray
        Low edge of the science window at each time sample, in degrees.
        An array so a drifting window (source-CES) can vary per sample.
    az_throw : float
        Science throw in degrees, constant across the pass.
    el_deg : float
        Boresight elevation in degrees, constant across the pass.
    times : Time
        Time samples spanning the pass; one entry per ``az_min`` element.
    az_speed : float
        Cruise azimuth speed in degrees/second.
    az_accel : float
        Turnaround acceleration in degrees/second^2.

    Returns
    -------
    az_arr, el_arr : np.ndarray
        Azimuth and elevation of every probe, length ``3 * len(times)``.
    times_arr : Time
        The matching times, each input time repeated once per probe.
    """
    az_lo = np.asarray(az_min, dtype=float)
    env_lo, env_hi = swept_az_envelope(az_lo, az_lo + float(az_throw), az_speed, az_accel)
    az_arr = np.concatenate([env_lo, 0.5 * (env_lo + env_hi), env_hi])
    el_arr = np.full(az_arr.size, float(el_deg))
    times_arr = times[np.tile(np.arange(len(times)), 3)]
    return az_arr, el_arr, times_arr


def _check_arc_sun_safety(
    coords: Coordinates,
    site: Site,
    az_arr: np.ndarray,
    el_arr: np.ndarray,
    times: Time,
    scan_label: str,
    sun_safe: SunSafePredicate | None = None,
    stacklevel: int = 4,
) -> None:
    """Check sun safety along an arc of samples; warns only.

    Computes sun separation at every sample and emits a single warning
    naming the closest approach if any point falls inside the exclusion
    radius. The caller chooses the sampling: the constant-elevation
    planners pass the grid :func:`_swept_arc_samples` builds, so the check
    covers the azimuth envelope the mount is commanded through rather than
    the boresight track alone. Boresight-level within each sample (no
    focal-plane extent), which is appropriate while the footprint is small
    relative to the exclusion radius.

    When ``sun_safe`` is ``None`` (default) the built-in vectorised
    scalar-radius check runs unchanged. When a predicate is injected it
    is consulted per-sample ``(az_i, el_i, time_i)`` instead, so the
    directional sun-avoidance model
    (see :func:`~fyst_trajectories.sun_models.make_sun_safe`) is honored;
    the warn-only semantics are preserved either way.

    Parameters
    ----------
    coords : Coordinates
        Coordinate helper used for the Sun ephemeris.
    site : Site
        Site whose ``sun_avoidance`` policy is screened against. A
        disabled policy makes this a no-op.
    az_arr, el_arr : np.ndarray
        Azimuth and elevation of every sample, in degrees.
    times : Time
        Time of every sample; same length as ``az_arr``.
    scan_label : str
        Phrase naming the planned scan in the warning text, for example
        ``"source-CES on jupiter"``.
    sun_safe : SunSafePredicate, optional
        Injected sun-safety model; see the description above.
    stacklevel : int, optional
        ``stacklevel`` for the emitted warning, counted from this
        function's own frame. Default 4 attributes the warning to the
        caller of a planner that reaches this check through one
        intermediate frame.

    Warns
    -----
    PointingWarning
        If any sample falls inside the sun exclusion radius (default) or
        the injected ``sun_safe`` predicate reports it unsafe.
    """
    if not site.sun_avoidance.enabled:
        return

    if sun_safe is None:
        sun_az_arr, sun_el_arr = coords.get_sun_altaz(times)
        # Vectorised haversine on the sphere (in degrees) so the whole sample
        # grid costs one call rather than a Python loop over every probe.
        az_rad = np.deg2rad(az_arr)
        el_rad = np.deg2rad(el_arr)
        sun_az_rad = np.deg2rad(np.asarray(sun_az_arr, dtype=float))
        sun_el_rad = np.deg2rad(np.asarray(sun_el_arr, dtype=float))
        cos_sep = np.sin(el_rad) * np.sin(sun_el_rad) + np.cos(el_rad) * np.cos(
            sun_el_rad
        ) * np.cos(az_rad - sun_az_rad)
        seps_deg = np.rad2deg(np.arccos(np.clip(cos_sep, -1.0, 1.0)))

        excl = site.sun_avoidance.exclusion_radius
        inside = seps_deg <= excl
        if not np.any(inside):
            return
        closest = int(np.argmin(seps_deg))
        # ``Time.__getitem__`` returns ``Time``; pyright's stubs sometimes
        # narrow it to ``Time | None`` because the dunder is generic. Coerce
        # via ``str()`` and silence the spurious optional-access warning.
        closest_iso = str(times[closest].iso)  # type: ignore[union-attr]
        warnings.warn(
            f"EXCLUSION ZONE: planned {scan_label} passes "
            f"{seps_deg[closest]:.1f} deg from the Sun at "
            f"{closest_iso} (exclusion radius {excl} deg).",
            PointingWarning,
            stacklevel=stacklevel,
        )
        return

    # Injected directional model: ``False`` marks an unsafe (inside-the-zone)
    # sample. Warn once, naming the first unsafe sample, mirroring the scalar
    # branch's single-warning semantics. A model exposing the vectorised
    # ``batch`` extension answers the whole grid in one call, which matters
    # here because this runs inside a dispatch-time plan.
    batch = getattr(sun_safe, "batch", None)
    if callable(batch):
        verdicts = np.asarray(batch(az_arr, el_arr, times), dtype=bool)
        if verdicts.shape != az_arr.shape:
            raise ValueError(
                f"sun_safe.batch returned shape {verdicts.shape}, expected {az_arr.shape}"
            )
        unsafe_idx = np.flatnonzero(~verdicts).tolist()
    else:
        unsafe_idx = [
            i
            for i in range(len(az_arr))
            if not sun_safe(float(az_arr[i]), float(el_arr[i]), times[i])  # type: ignore[index]
        ]
    if not unsafe_idx:
        return
    first = int(unsafe_idx[0])
    first_iso = str(times[first].iso)  # type: ignore[union-attr]
    warnings.warn(
        f"EXCLUSION ZONE: planned {scan_label} enters the Sun "
        f"avoidance zone at (az={float(az_arr[first]):.1f} deg, "
        f"el={float(el_arr[first]):.1f} deg) at {first_iso}.",
        PointingWarning,
        stacklevel=stacklevel,
    )
