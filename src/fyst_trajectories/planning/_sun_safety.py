"""Pre-flight sun-safety checks shared by all planner entry points."""

from __future__ import annotations

import math
import warnings

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

from ..coordinates import Coordinates
from ..exceptions import PointingWarning
from ..patterns.turnarounds import swept_az_envelope
from ..site import Site
from ..sun_protocols import SunSafePredicate, _sun_verdicts
from ..trajectory import Trajectory
from ..trajectory_utils import get_absolute_times

# Number of times sampled along a planned constant-elevation pass for the
# sun-safety sweep. 60 samples over a typical 10-minute arc gives ~10 s
# resolution, finer than the sun's apparent motion (~15"/s) and the array's
# footprint extent; a multi-hour pass gets proportionally coarser sampling,
# still well inside the exclusion radius the check screens against.
_SUN_SAFETY_ARC_N_SAMPLES = 60

# Spacing of the Sun ephemeris grid behind the built-in block screen. The Sun
# moves about 0.25 deg per minute, so its unit vector interpolated between grid
# points stays far inside the warning's 0.1 deg rounding, zenith transits
# included.
_SUN_SAFETY_EPHEMERIS_STEP_SEC = 60.0

# Most samples of a built trajectory an injected sun_safe model is asked
# about; the model solves its own Sun ephemeris per call, so every sample of
# an hour-long block would cost seconds.
_SUN_SAFETY_TRAJECTORY_MAX_PROBES = 600


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
        :class:`~fyst_trajectories.sun_protocols.SunSafePredicate` contract,
        ``(az_deg, el_deg, time) -> bool`` returning ``True`` when the field
        center is clear of the Sun. ``None`` (default) keeps the built-in
        scalar exclusion-radius check (the field center's angular separation
        from the Sun against ``site.sun_avoidance.exclusion_radius``). When a
        predicate is injected it is consulted in place of the scalar check, so
        the directional sun-avoidance model (see
        :func:`~fyst_trajectories.sun_models.make_sun_safe`) is honored
        end-to-end. See :class:`~fyst_trajectories.sun_protocols.SunSafePredicate`.

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
                f"EXCLUSION ZONE: Field center passes {sep:.1f} deg from the Sun "
                f"(exclusion radius: {site.sun_avoidance.exclusion_radius} deg) "
                f"at {start_time.iso}. This violates the configured Sun avoidance "
                f"policy; nothing downstream is guaranteed to reject it.",
                PointingWarning,
                stacklevel=stacklevel,
            )
    elif not sun_safe(az, el, start_time):
        warnings.warn(
            f"EXCLUSION ZONE: Field center at (az={az:.1f} deg, "
            f"el={el:.1f} deg) is inside the Sun avoidance zone at "
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
    *,
    sun_altaz: tuple[np.ndarray, np.ndarray] | None = None,
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
    sun_altaz : tuple of np.ndarray, optional
        The Sun's azimuth and elevation in degrees at every sample, used
        in place of solving the ephemeris at ``times`` when the caller
        already holds it. Read on the built-in branch only; an injected
        ``sun_safe`` model solves its own.

    Warns
    -----
    PointingWarning
        If any sample falls inside the sun exclusion radius (default) or
        the injected ``sun_safe`` predicate reports it unsafe.
    """
    if not site.sun_avoidance.enabled:
        return

    if sun_safe is None:
        if sun_altaz is None:
            sun_az_arr, sun_el_arr = coords.get_sun_altaz(times)
        else:
            sun_az_arr, sun_el_arr = sun_altaz
        # The same separation the centre pre-flight uses, over the whole
        # sample grid in one vectorised call.
        seps_deg = np.asarray(
            coords.angular_separation(az_arr, el_arr, sun_az_arr, sun_el_arr), dtype=float
        )

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
    # sample. Warn once, naming the earliest unsafe sample, mirroring the scalar
    # branch's single-warning semantics. A model exposing the vectorised
    # ``batch`` extension answers the whole grid in one call, which matters
    # here because this runs inside a dispatch-time plan.
    verdicts = _sun_verdicts(sun_safe, az_arr, el_arr, times, what="swept arc")
    unsafe_idx = np.flatnonzero(~verdicts).tolist()
    if not unsafe_idx:
        return
    jd = times.jd  # type: ignore[union-attr]
    first = min(unsafe_idx, key=lambda i: jd[i])
    first_iso = str(times[first].iso)  # type: ignore[union-attr]
    warnings.warn(
        f"EXCLUSION ZONE: planned {scan_label} enters the Sun "
        f"avoidance zone at (az={float(az_arr[first]):.1f} deg, "
        f"el={float(el_arr[first]):.1f} deg) at {first_iso}.",
        PointingWarning,
        stacklevel=stacklevel,
    )


def _check_trajectory_sun_safety(
    *,
    site: Site,
    trajectory: Trajectory,
    scan_label: str,
    sun_safe: SunSafePredicate | None = None,
) -> None:
    """Screen a built trajectory against the Sun over its whole block; warns only.

    A fixed horizon-frame scan holds its position while the Sun moves about
    15 deg per hour, so a block whose centre is clear at the start can end
    inside the zone, and the pattern's own extent can reach in where its
    centre does not. This screens the realised boresight trajectory, so it
    covers both, and emits :func:`_check_arc_sun_safety`'s single warning.

    With the built-in scalar check every sample is screened: the Sun is
    solved on a grid ``_SUN_SAFETY_EPHEMERIS_STEP_SEC`` apart and its unit
    vector interpolated to each sample, which is exact to well under the
    warning's 0.1 deg rounding. An injected ``sun_safe`` model solves its own
    ephemeris, so it is asked about at most
    ``_SUN_SAFETY_TRAJECTORY_MAX_PROBES`` evenly strided samples plus the
    last one; that is a sampled screen, not a proof, and a graze between
    probes can pass unwarned.

    Parameters
    ----------
    site : Site
        Site whose ``sun_avoidance`` policy is screened against. A disabled
        policy makes this a no-op.
    trajectory : Trajectory
        The built trajectory. Without a ``start_time`` there is no Sun to
        place, and the screen is skipped.
    scan_label : str
        Phrase naming the planned scan in the warning text.
    sun_safe : SunSafePredicate, optional
        Injected sun-safety model; ``None`` uses the scalar radius.

    Warns
    -----
    PointingWarning
        If any screened sample falls inside the sun exclusion radius
        (default) or the injected ``sun_safe`` model reports it unsafe. The
        warning is attributed to the caller of the planner that calls this.
    """
    if not site.sun_avoidance.enabled or trajectory.start_time is None:
        return
    coords = Coordinates(site)
    if sun_safe is None:
        rel = trajectory.times - trajectory.times[0]
        grid = np.append(np.arange(0.0, rel[-1], _SUN_SAFETY_EPHEMERIS_STEP_SEC), rel[-1])
        grid_az, grid_el = coords.get_sun_altaz(trajectory.start_time + TimeDelta(grid * u.s))
        az_rad = np.radians(np.asarray(grid_az, dtype=float))
        el_rad = np.radians(np.asarray(grid_el, dtype=float))
        # Interpolate the Sun's unit vector rather than its azimuth, which
        # swings through tens of degrees a minute near a zenith transit.
        x, y, z = (
            np.interp(rel, grid, component)
            for component in (
                np.cos(el_rad) * np.cos(az_rad),
                np.cos(el_rad) * np.sin(az_rad),
                np.sin(el_rad),
            )
        )
        sun_az = np.degrees(np.arctan2(y, x)) % 360.0
        sun_el = np.degrees(np.arcsin(z / np.sqrt(x * x + y * y + z * z)))
        _check_arc_sun_safety(
            coords,
            site,
            trajectory.az,
            trajectory.el,
            get_absolute_times(trajectory),
            scan_label,
            stacklevel=4,
            sun_altaz=(sun_az, sun_el),
        )
        return
    n = trajectory.n_points
    step = max(1, math.ceil(n / _SUN_SAFETY_TRAJECTORY_MAX_PROBES))
    index = np.arange(0, n, step)
    if index[-1] != n - 1:
        index = np.append(index, n - 1)
    _check_arc_sun_safety(
        coords,
        site,
        trajectory.az[index],
        trajectory.el[index],
        get_absolute_times(trajectory)[index],
        scan_label,
        sun_safe=sun_safe,
        stacklevel=4,
    )
