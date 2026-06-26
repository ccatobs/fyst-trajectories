"""Scheduling constraints.

Each constraint scores a candidate observation from 0.0 (infeasible) to 1.0
(optimal). Scores are multiplied together by the scheduler.
"""

from abc import ABC, abstractmethod

from astropy.time import Time, TimeDelta

from ..coordinates import Coordinates
from .models import ObservingPatch

__all__ = [
    "Constraint",
    "ElevationConstraint",
    "MinDurationConstraint",
    "MoonAvoidanceConstraint",
    "SunAvoidanceConstraint",
]


class Constraint(ABC):
    """Base class for scheduling constraints.

    Subclasses implement ``score()`` to evaluate a candidate observation.
    A score of 0 means the observation is infeasible; 1 means fully
    acceptable. Values between 0 and 1 express preference.
    """

    @abstractmethod
    def score(
        self,
        patch: ObservingPatch,
        time: Time,
        az: float,
        el: float,
        coords: Coordinates,
    ) -> float:
        """Score a candidate observation.

        Parameters
        ----------
        patch : ObservingPatch
            The candidate patch.
        time : Time
            Candidate observation start time.
        az : float
            Azimuth of the patch center at this time (degrees).
        el : float
            Elevation of the patch center at this time (degrees).
        coords : Coordinates
            Coordinate transformer for the site.

        Returns
        -------
        float
            Score from 0.0 (infeasible) to 1.0 (optimal).
        """


class ElevationConstraint(Constraint):
    """Enforce elevation bounds on observations.

    Returns 1.0 if the target elevation is within ``[el_min, el_max]``,
    0.0 otherwise.

    Defaults match ``FYST_EL_MIN = 20`` and ``FYST_EL_MAX = 90`` so that
    users who instantiate ``ElevationConstraint()`` without arguments get
    the full FYST-allowed range rather than a more restrictive subset.
    For other observatories, override these values explicitly or pull
    them from ``site.telescope_limits.elevation``. The internal scheduler
    in :mod:`fyst_trajectories.overhead.timeline` already constructs its
    default constraint from the site limits, so this default only
    matters for standalone use.

    Parameters
    ----------
    el_min : float
        Minimum allowed elevation in degrees. Defaults to 20.0
        (``FYST_EL_MIN``).
    el_max : float
        Maximum allowed elevation in degrees. Defaults to 90.0
        (``FYST_EL_MAX``).
    """

    def __init__(self, el_min: float = 20.0, el_max: float = 90.0) -> None:
        if el_min >= el_max:
            raise ValueError(f"el_min ({el_min}) must be less than el_max ({el_max})")
        self.el_min = el_min
        self.el_max = el_max

    def score(
        self,
        patch: ObservingPatch,
        time: Time,
        az: float,
        el: float,
        coords: Coordinates,
    ) -> float:
        """Return 1.0 if elevation is within bounds, 0.0 otherwise."""
        if self.el_min <= el <= self.el_max:
            return 1.0
        return 0.0


class SunAvoidanceConstraint(Constraint):
    """Enforce minimum angular separation from the Sun.

    Returns 0.0 if the target is closer than ``min_angle`` degrees
    to the Sun, 1.0 otherwise.

    Parameters
    ----------
    min_angle : float
        Minimum angular distance from the Sun in degrees.
    """

    def __init__(self, min_angle: float = 45.0) -> None:
        if min_angle < 0:
            raise ValueError(f"min_angle must be non-negative, got {min_angle}")
        self.min_angle = min_angle

    def score(
        self,
        patch: ObservingPatch,
        time: Time,
        az: float,
        el: float,
        coords: Coordinates,
    ) -> float:
        """Return 0.0 if too close to Sun, 1.0 otherwise."""
        sun_az, sun_el = coords.get_sun_altaz(time)
        sep = coords.angular_separation(az, el, sun_az, sun_el)
        if sep < self.min_angle:
            return 0.0
        return 1.0


class MoonAvoidanceConstraint(Constraint):
    """Enforce minimum angular separation from the Moon.

    Notes
    -----
    Moon avoidance is asymmetric across the library by design:
    the scheduler (this constraint) gates patch selection on lunar
    proximity, but the planning helpers (``plan_pong_scan`` etc.)
    do not run a moon-safety pre-flight check the way they do for
    the Sun. This is intentional — at submillimetre wavelengths the
    Moon is a useful calibration source (it is a bright, well-modelled
    extended target), so total avoidance is not always desirable.
    Callers who want a hard pre-flight moon check should query
    ``coords.get_body_altaz("moon", obstime)`` and apply their own
    threshold before constructing a trajectory.

    Parameters
    ----------
    min_angle : float
        Minimum angular distance from the Moon in degrees.
    """

    def __init__(self, min_angle: float = 20.0) -> None:
        if min_angle < 0:
            raise ValueError(f"min_angle must be non-negative, got {min_angle}")
        self.min_angle = min_angle

    def score(
        self,
        patch: ObservingPatch,
        time: Time,
        az: float,
        el: float,
        coords: Coordinates,
    ) -> float:
        """Return 0.0 if too close to Moon, 1.0 otherwise."""
        moon_az, moon_el = coords.get_body_altaz("moon", time)
        sep = coords.angular_separation(az, el, moon_az, moon_el)
        if sep < self.min_angle:
            return 0.0
        return 1.0


class MinDurationConstraint(Constraint):
    """Reject observations where the remaining observable window is too short.

    This is a heuristic: if the target will set (or enter Sun exclusion)
    within ``min_duration`` seconds, skip it.

    Parameters
    ----------
    min_duration : float
        Minimum required observable time in seconds.
    """

    def __init__(self, min_duration: float = 60.0) -> None:
        if min_duration < 0:
            raise ValueError(f"min_duration must be non-negative, got {min_duration}")
        self.min_duration = min_duration

    def score(
        self,
        patch: ObservingPatch,
        time: Time,
        az: float,
        el: float,
        coords: Coordinates,
    ) -> float:
        """Return 0.0 if target sets or enters Sun exclusion too soon, 1.0 otherwise.

        Forward check: verify the target is still above the elevation limit
        after ``min_duration`` seconds, and, when the site has sun
        avoidance enabled, that it is still outside the Sun exclusion
        radius then. The sun forward-check matches the class docstring's
        promise (it previously checked elevation only).
        """
        future_time = time + TimeDelta(self.min_duration, format="sec")
        future_az, future_el = coords.radec_to_altaz(patch.ra_center, patch.dec_center, future_time)
        el_min = coords.site.telescope_limits.elevation.min
        if future_el < el_min:
            return 0.0
        sun_avoidance = coords.site.sun_avoidance
        if sun_avoidance.enabled:
            sun_az, sun_el = coords.get_sun_altaz(future_time)
            sep = coords.angular_separation(future_az, future_el, sun_az, sun_el)
            if sep < sun_avoidance.exclusion_radius:
                return 0.0
        return 1.0
