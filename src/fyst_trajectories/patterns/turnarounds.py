"""Turnaround profile generators for scan patterns."""

import numpy as np


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

    The name "quintic" follows the SO convention even though the
    boundary conditions yield a degree-4 polynomial (the t^5
    coefficient is zero).

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
    tau = t / T
    tau2 = tau * tau
    tau3 = tau2 * tau

    # p(t) = v*T*(tau - 2*tau^3 + tau^4)
    position = v * T * (tau - 2.0 * tau3 + tau3 * tau)

    # p'(t) = v*(1 - 6*tau^2 + 4*tau^3)
    velocity = v * (1.0 - 6.0 * tau2 + 4.0 * tau3)

    return position, velocity
