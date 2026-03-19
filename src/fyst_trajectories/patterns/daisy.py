"""Daisy (Constant Velocity petal) scan pattern.

The Daisy pattern is a constant-velocity pattern optimized for point sources.
The telescope moves in curved petal-shaped paths that repeatedly cross
the center of the field, ensuring good coverage of the central region.

See "CV Daisy - JCMT small area scanning pattern" (JCMT TCS/UN/005)
for algorithm details.

Performance
-----------
The inner loop runs at 150 Hz internal timestep regardless of the output
timestep. A 300-second scan produces 45,000 iterations.

With numba (``pip install fyst-trajectories[performance]``):
    JIT-compiled; generation is fast (sub-second for typical scans).

Without numba (pure Python fallback):
    Each iteration executes Python-level floating-point math. A 300-second
    scan may take several seconds on a typical machine. This is acceptable
    for offline planning but is not recommended for production or repeated
    high-volume use.

Install numba for production use.
"""

import math

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

from ..coordinates import Coordinates
from ..exceptions import TargetNotObservableError, TrajectoryBoundsError
from ..math_utils import SMALL_DISTANCE_EPSILON
from ..site import AtmosphericConditions, Site
from ..trajectory import Trajectory
from ..trajectory_utils import validate_trajectory_bounds
from .base import CelestialPattern, TrajectoryMetadata
from .configs import DaisyScanConfig
from .registry import register_pattern
from .utils import compute_velocities, normalize_azimuth, sky_offsets_to_altaz

try:
    import numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Internal timestep for Daisy pattern ensuring smooth curve approximation.
# Fixed at 1/150 s (~6.67 ms) rather than user-configurable because the
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


# Use numba JIT-compiled version when available, plain Python otherwise
if HAS_NUMBA:
    _daisy_loop = numba.jit(nopython=True)(_daisy_loop_python)
else:
    _daisy_loop = _daisy_loop_python


@register_pattern("daisy")
class DaisyScanPattern(CelestialPattern):
    """Daisy scan pattern for point source observations.

    The Daisy pattern moves in curved petal-shaped paths that repeatedly
    cross the center of the field. It is optimized for point source
    observations with good central coverage.

    Parameters
    ----------
    ra : float
        Right Ascension of pattern center in degrees.
    dec : float
        Declination of pattern center in degrees.
    config : DaisyScanConfig
        Pattern configuration.

    Attributes
    ----------
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    config : DaisyScanConfig
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
        config: DaisyScanConfig,
    ):
        super().__init__(ra, dec)
        self.config = config

    @property
    def name(self) -> str:
        """Return pattern identifier."""
        return "daisy"

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
        """
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")

        timestep = self.config.timestep

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
        times = np.linspace(0, duration, n_points)

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
            If ``start_time`` is None.
        TrajectoryBoundsError
            If the trajectory exceeds telescope limits.
        TargetNotObservableError
            If the target is below the horizon or outside telescope
            limits at the requested time.
        """
        if start_time is None:
            raise ValueError(
                "start_time is required for DaisyScanPattern (celestial pattern). "
                "Provide an astropy Time object."
            )

        coords = Coordinates(site, atmosphere=atmosphere)

        times, x_offsets, y_offsets = self.generate_offsets(duration)

        obstimes = start_time + TimeDelta(times * u.s)

        az, el = sky_offsets_to_altaz(
            x_offsets,
            y_offsets,
            self.ra,
            self.dec,
            obstimes,
            coords,
        )
        az = normalize_azimuth(az, site)

        az_vel = compute_velocities(az, times, is_angular=True)
        el_vel = compute_velocities(el, times, is_angular=False)

        try:
            validate_trajectory_bounds(site, az, el)
        except TrajectoryBoundsError as exc:
            raise TargetNotObservableError(
                target=f"RA={self.ra:.3f} Dec={self.dec:.3f}",
                time_info=start_time.iso,
                bounds_error=exc,
            ) from None

        return Trajectory(
            times=times,
            az=az,
            el=el,
            az_vel=az_vel,
            el_vel=el_vel,
            start_time=start_time,
            metadata=self.get_metadata(),
            coordsys="altaz",
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
