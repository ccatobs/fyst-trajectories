"""Daisy (Constant Velocity petal) scan pattern.

See Friberg 2012, "CV Daisy - JCMT small area scanning pattern", Joint
Astronomy Centre, JCMT TCS/UN/005, for algorithm details.

Performance
-----------
The inner loop runs at an internal timestep of at most 1/150 s (it
follows the output timestep when that is finer), so a 300-second scan
runs 45,000 to 90,000 iterations at any output timestep of 1/150 s or
coarser, and ``duration / timestep`` at finer ones.

With numba (``pip install fyst-trajectories[performance]``):
    JIT-compiled; the inner loop runs one to two orders of magnitude
    faster than the pure-Python fallback.

Without numba (pure Python fallback):
    Each iteration executes Python-level floating-point math. A typical
    scan still generates in well under a second on a current interpreter,
    so offline planning is comfortable either way; the fallback only
    becomes limiting at fine output timesteps, very long scans, or
    repeated high-volume generation.

Install numba for production use.
"""

import math

import numpy as np
from astropy.time import Time

from ..math_utils import SMALL_DISTANCE_EPSILON
from ..site import AtmosphericConditions, Site
from ..trajectory import Trajectory, TrajectoryMetadata
from .base import CelestialPattern
from .configs import DaisyAltAzScanConfig, DaisyScanConfig
from .registry import register_pattern
from .utils import (
    _SCIENCE_SPEED_FRACTION,
    _celestial_offsets_to_trajectory,
    _flag_by_offset_speed,
    validate_sample_count,
)

try:
    import numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Internal timestep ceiling for the Daisy pattern, ensuring smooth curve
# approximation (the loop runs at the output timestep when that is finer).
# Capped at 1/150 s (~6.67 ms) rather than user-configurable because the
# Taylor-series position updates during curved segments assume small arc
# lengths per step.  At typical velocities (~0.3 deg/s) and turn radii
# (~0.2 deg), this gives adequate sampling.  Extreme parameter combinations
# (very high velocity with very tight turn radius) may need a finer timestep
# for accurate results; see DaisyScanConfig docstring.
_DAISY_INTERNAL_TIMESTEP = 1.0 / 150.0


def _daisy_loop_python(
    n_internal: int,
    dt: float,
    r0: float,
    rt: float,
    ra_avoid: float,
    target_speed: float,
    start_acc: float,
    y_offset: float,
    small_dist_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inner loop for daisy pattern generation (pure Python).

    When numba is available, this function is JIT-compiled for performance.
    Without numba, it runs as plain Python (slower but functionally identical).

    Parameters
    ----------
    n_internal : int
        Number of internal time steps.
    dt : float
        Time step in seconds.
    r0 : float
        Characteristic radius in degrees.
    rt : float
        Turn radius in degrees.
    ra_avoid : float
        Avoidance radius in degrees.
    target_speed : float
        Target velocity in sky-offset degrees/second.
    start_acc : float
        Start acceleration in sky-offset degrees/second^2.
    y_offset : float
        Initial y offset in degrees.
    small_dist_eps : float
        Epsilon for detecting near-zero distances.

    Returns
    -------
    x_coords : np.ndarray
        X coordinates in degrees.
    y_coords : np.ndarray
        Y coordinates in degrees.
    """
    x, y = 0.0, y_offset
    vx, vy = 1.0, 0.0

    x_coords = np.empty(n_internal)
    y_coords = np.empty(n_internal)

    speed = 0.0
    for step in range(n_internal):
        speed += start_acc * dt
        if speed >= target_speed:
            speed = target_speed

        r = math.sqrt(x * x + y * y)

        if r < small_dist_eps:
            x += vx * speed * dt
            y += vy * speed * dt
            x_coords[step] = x
            y_coords[step] = y
            continue

        if r < r0:
            x += vx * speed * dt
            y += vy * speed * dt
        else:
            xn = x / r
            yn = y / r

            dot_product = -xn * vx - yn * vy

            if r > ra_avoid:
                threshold = math.sqrt(1 - (ra_avoid * ra_avoid) / (r * r))
            else:
                threshold = 0.0

            if dot_product > threshold:
                x += vx * speed * dt
                y += vy * speed * dt
            else:
                cross = -xn * vy + yn * vx
                if cross > 0:
                    nx = vy
                    ny = -vx
                else:
                    nx = -vy
                    ny = vx

                s = speed * dt
                s2 = s * s
                s3 = s2 * s
                rt2 = rt * rt
                rt3 = rt2 * rt

                x += (s - s3 / (rt2 * 6)) * vx + (s2 / (rt * 2)) * nx
                y += (s - s3 / (rt2 * 6)) * vy + (s2 / (rt * 2)) * ny

                vx += (-s2 / (rt2 * 2)) * vx + (s / rt - s3 / (rt3 * 6)) * nx
                vy += (-s2 / (rt2 * 2)) * vy + (s / rt - s3 / (rt3 * 6)) * ny

                v_mag = math.sqrt(vx * vx + vy * vy)
                vx /= v_mag
                vy /= v_mag

        x_coords[step] = x
        y_coords[step] = y

    return x_coords, y_coords


if HAS_NUMBA:
    _daisy_loop = numba.jit(nopython=True)(_daisy_loop_python)
else:
    _daisy_loop = _daisy_loop_python


def _daisy_reach(
    radius: float,
    turn_radius: float,
    velocity: float,
    timestep: float,
    y_offset: float = 0.0,
) -> float:
    """Return the farthest a Daisy pattern goes from its centre, in on-sky degrees.

    A petal starts its turn on the first integrator step at or past
    ``radius``, no more than one step of at most
    ``velocity * min(timestep, 1/150 s)`` past it, and turns toward the
    centre on a circle of radius ``turn_radius``. That circle's centre lies
    no farther than ``hypot(r, turn_radius)`` from the pattern's centre,
    where ``r`` is where the turn began, so no point of the turn is farther
    than that plus ``turn_radius``. A petal that leaves the centre radially,
    as the first does from a zero ``y_offset``, reaches this to within the
    step, so the pattern goes at least ``turn_radius`` beyond ``radius``. A
    pattern that starts outside ``radius`` (a ``y_offset`` beyond it) turns
    at once, heading across the radius, so that first turn stays within the
    larger of ``abs(y_offset)`` and the bound above. One more step is added
    for the integrator itself: with an ``avoidance_radius`` wider than the
    pattern a turn can end tangent at its farthest point and the steps that
    follow creep outward by microdegrees, and a step comparable to
    ``turn_radius`` overshoots the turn.

    Parameters
    ----------
    radius, turn_radius : float
        The pattern's characteristic and turn radii, in degrees.
    velocity : float
        Scan velocity in sky-offset degrees/second.
    timestep : float
        Output timestep in seconds; the integrator steps at most 1/150 s.
    y_offset : float, optional
        Initial y offset in degrees.

    Returns
    -------
    float
        An upper bound on the distance of every pattern offset from the
        centre, in degrees.
    """
    step = velocity * min(timestep, _DAISY_INTERNAL_TIMESTEP)
    return max(abs(y_offset), math.hypot(radius + step, turn_radius) + turn_radius) + step


@register_pattern("daisy", config=DaisyScanConfig)
class DaisyScanPattern(CelestialPattern):
    """Daisy (Constant Velocity petal) scan about a tracked RA/Dec center.

    Parameters
    ----------
    ra : float
        Right Ascension of pattern center in degrees.
    dec : float
        Declination of pattern center in degrees.
    config : DaisyScanConfig or DaisyAltAzScanConfig
        Pattern configuration. A :class:`DaisyAltAzScanConfig` is accepted
        for the fields it shares with :class:`DaisyScanConfig`; its
        horizon-frame centre (``az_center``, ``el_center``) is not read, and
        the pattern is centred on ``ra`` and ``dec``.

    Attributes
    ----------
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    config : DaisyScanConfig or DaisyAltAzScanConfig
        The configuration for this pattern.

    Examples
    --------
    >>> from astropy.time import Time
    >>> from fyst_trajectories.patterns import DaisyScanPattern, DaisyScanConfig
    >>> start_time = Time("2026-03-15T04:00:00", scale="utc")
    >>> config = DaisyScanConfig(
    ...     timestep=0.1,
    ...     radius=0.5,
    ...     velocity=0.3,
    ...     turn_radius=0.2,
    ...     avoidance_radius=0.0,
    ...     start_acceleration=0.5,
    ...     y_offset=0.0,
    ... )
    >>> pattern = DaisyScanPattern(ra=180.0, dec=-30.0, config=config)
    >>> trajectory = pattern.generate(site, duration=300.0, start_time=start_time)
    """

    def __init__(
        self,
        ra: float,
        dec: float,
        config: DaisyScanConfig | DaisyAltAzScanConfig,
    ):
        super().__init__(ra, dec)
        self.config = config

    def generate_offsets(self, duration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate scan pattern offsets without coordinate conversion.

        Returns pure sky-plane offsets suitable for use by external libraries
        that handle their own coordinate transforms.

        Parameters
        ----------
        duration : float
            Total duration of the scan in seconds.

        Returns
        -------
        times : np.ndarray
            Time array in seconds from scan start.
        x_offsets : np.ndarray
            X offsets in the sky-plane tangent frame, in degrees.
        y_offsets : np.ndarray
            Y offsets in the sky-plane tangent frame, in degrees.

        Raises
        ------
        ValueError
            If ``duration`` yields fewer than two samples at the config
            timestep.
        """
        timestep = self.config.timestep
        validate_sample_count(duration, timestep)

        if timestep > _DAISY_INTERNAL_TIMESTEP:
            sample_every = math.ceil(timestep / _DAISY_INTERNAL_TIMESTEP)
            dt = timestep / sample_every
        else:
            sample_every = 1
            dt = timestep

        x_coords, y_coords = self._generate_daisy_pattern(
            duration=duration,
            dt=dt,
            r0=self.config.radius,
            rt=self.config.turn_radius,
            ra_avoid=self.config.avoidance_radius,
            target_speed=self.config.velocity,
            start_acc=self.config.start_acceleration,
            y_offset=self.config.y_offset,
        )

        if sample_every > 1:
            x_coords = x_coords[::sample_every]
            y_coords = y_coords[::sample_every]

        n_points = len(x_coords)
        # The ``[::sample_every]`` downsampling drops the final partial step, so
        # a duration of exactly one ``timestep`` (which passes the up-front
        # ``validate_sample_count`` guard) can still collapse to a single kept
        # sample. Reject that here so daisy fails loud and consistently with
        # the other patterns rather than feeding 1 point into ``np.gradient``.
        if n_points < 2:
            raise ValueError(
                f"duration {duration}s yields fewer than 2 samples at timestep "
                f"{timestep}s; a scan shorter than one sample is degenerate."
            )
        # Label samples on the integrator's own grid: each kept sample is
        # ``sample_every`` internal steps apart, i.e. exactly ``timestep``
        # (``sample_every * dt == timestep`` by construction above). The
        # integrator spans ``[0, duration - dt]``, so ``linspace(0, duration,
        # n_points)`` would stretch the axis and bias every velocity derived
        # as ``gradient(position, times)`` (~1% at a 10 s scan).
        times = np.arange(n_points) * (sample_every * dt)

        return times, x_coords, y_coords

    def generate(
        self,
        site: Site,
        duration: float,
        start_time: Time | None,
        atmosphere: AtmosphericConditions | None = None,
    ) -> Trajectory:
        """Generate the Daisy scan trajectory.

        Parameters
        ----------
        site : Site
            Telescope site configuration.
        duration : float
            Total duration of the scan in seconds.
        start_time : Time
            Start time for the trajectory. Required for coordinate
            transforms (RA/Dec to AltAz).
        atmosphere : AtmosphericConditions or None, optional
            Atmospheric conditions for refraction correction.
            If None, no refraction is applied.

        Returns
        -------
        Trajectory
            The generated trajectory.

        Raises
        ------
        ValueError
            If ``start_time`` is None, or if ``duration`` yields fewer than
            two samples at the config timestep.
        TargetNotObservableError
            If the target is below the horizon or outside telescope
            limits at the requested time (bounds violations are
            wrapped into this error).

        Warns
        -----
        PointingWarning
            If no whole-turn shift places the azimuth track inside the
            telescope's azimuth range.
        """
        if start_time is None:
            raise ValueError(
                "start_time is required for DaisyScanPattern (celestial pattern). "
                "Provide an astropy Time object."
            )

        times, x_offsets, y_offsets = self.generate_offsets(duration)
        scan_flag = _flag_by_offset_speed(
            times, x_offsets, y_offsets, self.config.velocity, _SCIENCE_SPEED_FRACTION
        )
        return _celestial_offsets_to_trajectory(
            site,
            self.ra,
            self.dec,
            times,
            x_offsets,
            y_offsets,
            scan_flag,
            start_time,
            atmosphere,
            self.get_metadata(),
        )

    def get_metadata(self) -> TrajectoryMetadata:
        """Get pattern metadata.

        Returns
        -------
        TrajectoryMetadata
            Metadata including pattern type and parameters.
        """
        return TrajectoryMetadata(
            pattern_type=self.name,
            pattern_params={
                "radius": self.config.radius,
                "velocity": self.config.velocity,
                "turn_radius": self.config.turn_radius,
                "avoidance_radius": self.config.avoidance_radius,
                "start_acceleration": self.config.start_acceleration,
                "y_offset": self.config.y_offset,
            },
            center_ra=self.ra,
            center_dec=self.dec,
            input_frame="icrs",
        )

    def _generate_daisy_pattern(
        self,
        duration: float,
        dt: float,
        r0: float,
        rt: float,
        ra_avoid: float,
        target_speed: float,
        start_acc: float,
        y_offset: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate the daisy pattern x,y coordinates.

        Delegates to the inner loop (JIT-compiled if numba is available).

        Parameters
        ----------
        duration : float
            Total duration in seconds.
        dt : float
            Time step in seconds.
        r0 : float
            Characteristic radius in degrees.
        rt : float
            Turn radius in degrees.
        ra_avoid : float
            Avoidance radius in degrees.
        target_speed : float
            Target velocity in sky-offset degrees/second.
        start_acc : float
            Start acceleration in sky-offset degrees/second^2.
        y_offset : float
            Initial y offset in degrees.

        Returns
        -------
        x_coords : np.ndarray
            X coordinates in degrees.
        y_coords : np.ndarray
            Y coordinates in degrees.
        """
        n_internal = int(round(duration / dt))
        return _daisy_loop(
            n_internal,
            dt,
            r0,
            rt,
            ra_avoid,
            target_speed,
            start_acc,
            y_offset,
            SMALL_DISTANCE_EPSILON,
        )
