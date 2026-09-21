"""Turnaround profile generators for scan patterns."""

from typing import TypeVar

import numpy as np

_AzLike = TypeVar("_AzLike", float, np.ndarray)


def quintic_turnaround(
    t: np.ndarray,
    v: float,
    T: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute position and velocity for a smooth polynomial turnaround.

    Uses a degree-4 polynomial that satisfies six boundary conditions:
    - p(0) = 0, p(T) = 0  (returns to entry position)
    - p'(0) = +v, p'(T) = -v  (reverses velocity)
    - p''(0) = 0, p''(T) = 0  (zero acceleration at boundaries)

    This is the standard turnaround of the Simons Observatory ACU agent
    (``socs.agents.acu.turnarounds``), which generates it to mimic the
    spline-based turnaround its antenna control unit produces. It is the
    fifth-order ("quintic") polynomial fixed by these six conditions,
    the motion meeting them with the least integrated squared jerk; for
    this symmetric reversal the t^5 coefficient is zero, so the result
    is degree 4.

    Key properties:
    - Peak displacement: 5*v*T/16 at t=T/2
    - Peak acceleration: 3*v/T = 1.5 * a_avg
    - Velocity passes through zero at t=T/2

    Parameters
    ----------
    t : np.ndarray
        Time array within the turnaround, 0 <= t <= T.
    v : float
        Entry speed (positive). Exit velocity will be -v.
    T : float
        Total turnaround duration in seconds.

    Returns
    -------
    position : np.ndarray
        Position offset from entry point.
    velocity : np.ndarray
        Velocity at each time point.
    """
    tau = np.clip(t / T, 0.0, 1.0)
    tau2 = tau * tau
    tau3 = tau2 * tau

    # p(t) = v*T*(tau - 2*tau^3 + tau^4)
    position = v * T * (tau - 2.0 * tau3 + tau3 * tau)

    # p'(t) = v*(1 - 6*tau^2 + 4*tau^3)
    velocity = v * (1.0 - 6.0 * tau2 + 4.0 * tau3)

    return position, velocity


def turnaround_duration_sec(az_speed: float, az_accel: float) -> float:
    """Return how long one quintic turnaround between azimuth legs takes.

    The turnaround lasts ``2 * az_speed / az_accel``, the time a uniform
    deceleration and re-acceleration at ``az_accel`` would take; the
    quintic keeps that duration but not that shape, so ``az_accel`` is its
    average acceleration. The generator that emits the turnaround and the
    quantiser that counts how many legs fit in a window must agree on it,
    which is why it is one shared helper.

    Parameters
    ----------
    az_speed : float
        Cruise azimuth speed in degrees/second (positive).
    az_accel : float
        Average turnaround acceleration in degrees/second^2 (positive).
        The quintic profile's peak acceleration is 1.5x this value.

    Returns
    -------
    float
        Turnaround duration in seconds.

    Examples
    --------
    >>> from fyst_trajectories.patterns.turnarounds import turnaround_duration_sec
    >>> turnaround_duration_sec(1.5, 1.0)
    3.0
    """
    return 2.0 * az_speed / az_accel


def turnaround_overshoot_deg(az_speed: float, az_accel: float) -> float:
    """Return how far a quintic turnaround overshoots the science edge.

    A constant-elevation leg cruises at ``az_speed`` to the science edge
    and then reverses through a :func:`quintic_turnaround` of duration
    :func:`turnaround_duration_sec`, ``T = 2 * az_speed / az_accel``. The
    turnaround's peak displacement is ``5 * az_speed * T / 16``, so the
    telescope travels ``5 * az_speed**2 / (8 * az_accel)`` degrees past the
    science edge before coming back to it.

    The commanded azimuth envelope of a sweep over ``[az_min, az_max]`` is
    therefore ``[az_min - d, az_max + d]`` for the ``d`` returned here.
    Any envelope reasoning about a constant-elevation scan (axis-limit
    checks, Sun screens) has to use that wider range, not the science
    window, which is why this is one shared helper rather than a repeated
    expression.

    Parameters
    ----------
    az_speed : float
        Cruise azimuth speed in degrees/second (positive).
    az_accel : float
        Average turnaround acceleration in degrees/second^2 (positive).
        The quintic profile's peak acceleration is 1.5x this value.

    Returns
    -------
    float
        Overshoot distance past each science edge, in degrees.

    Examples
    --------
    >>> from fyst_trajectories.patterns.turnarounds import turnaround_overshoot_deg
    >>> round(turnaround_overshoot_deg(1.5, 1.0), 6)
    1.40625
    """
    return 5.0 * az_speed**2 / (8.0 * az_accel)


def swept_az_envelope(
    az_min: _AzLike,
    az_max: _AzLike,
    az_speed: float,
    az_accel: float,
) -> tuple[_AzLike, _AzLike]:
    """Return the commanded azimuth envelope of a constant-elevation sweep.

    The single definition of the range a back-and-forth azimuth sweep
    occupies: the science window ``[az_min, az_max]`` widened on both
    sides by :func:`turnaround_overshoot_deg`. Every consumer that
    reasons about where the mount goes - axis-limit checks, Sun screens,
    the pattern generator's own motion bounds, the offline scheduler's
    cable-wrap choice - has to use this range and not the science window,
    which is narrower by one overshoot per side.

    Widening the *right* window matters as much as widening it. A
    drifting constant-elevation pass sweeps the corridor its field
    crosses over the whole pass, which is far wider than the field's
    instantaneous width, so the window handed in has to be that corridor.

    Parameters
    ----------
    az_min, az_max : float or np.ndarray
        Low and high edge of the science window in degrees. Arrays are
        supported so a drifting window can be widened per time sample.
    az_speed : float
        Cruise azimuth speed in degrees/second (positive).
    az_accel : float
        Average turnaround acceleration in degrees/second^2 (positive).

    Returns
    -------
    env_min, env_max : float or np.ndarray
        Low and high edge of the commanded envelope, in degrees, matching
        the input types.

    Examples
    --------
    >>> from fyst_trajectories.patterns.turnarounds import swept_az_envelope
    >>> lo, hi = swept_az_envelope(100.0, 150.0, 1.5, 1.0)
    >>> round(lo, 5), round(hi, 5)
    (98.59375, 151.40625)
    """
    overshoot = turnaround_overshoot_deg(az_speed, az_accel)
    return az_min - overshoot, az_max + overshoot
