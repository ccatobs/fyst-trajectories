"""Pong (curvy box) scan pattern.

See "Scan Mode Strategies for SCUBA-2" (SCUBA-2 Project SC2/ANA/S210/005,
Scott & Van Engelen 2005) for algorithm details.
"""

import functools
import math

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

from ..coordinates import Coordinates
from ..site import AtmosphericConditions, Site
from ..trajectory import SCAN_FLAG_SCIENCE, SCAN_FLAG_TURNAROUND, Trajectory
from ..trajectory_utils import validate_trajectory_bounds
from .base import CelestialPattern, TrajectoryMetadata
from .configs import PongScanConfig
from .registry import register_pattern
from .utils import (
    compute_velocities,
    normalize_azimuth,
    sky_offsets_to_altaz,
    wrap_bounds_error,
)

_TURNAROUND_SPEED_THRESHOLD: float = 0.8
"""Fraction of nominal velocity below which samples are flagged as turnaround.

Samples with offset-frame speed below ``threshold * config.velocity`` are
classified as SCAN_FLAG_TURNAROUND.

The 0.8 threshold is empirical and matches the natural speed reduction
at the vertices of a Fourier-truncated triangle wave with the default
``num_terms = 4`` (harmonics 1, 3, 5, 7 give a peak velocity at the
midpoint and a ~80% velocity at the vertex). Increasing ``num_terms``
sharpens the corners and may require this threshold to be lowered;
decreasing it does the opposite.

No published cross-facility standard exists for this exact value.
SO classifies turnarounds geometrically (samples outside the science
azimuth range — see Hoang et al. 2024); JCMT uses a similar geometric
criterion. The speed-based criterion here is closer to ACT's
time-window approach but tuned for FYST/Prime-Cam scan dynamics.
"""


# Cache size 32 comfortably covers a planning session that iterates over many
# distinct (width, height, spacing) tuples (e.g. an outer loop over patches
# with different field geometries, or a multi-rotation pong sequence with
# differing footprints). Each cache entry is a 4-tuple of ints/floats, so the
# memory cost is negligible; raising the bound trades small cache footprint
# for a guaranteed hit rate across realistic workflows.
@functools.lru_cache(maxsize=32)
def _compute_pong_vertices(
    width: float, height: float, spacing: float
) -> tuple[int, int, float, float]:
    """Compute vertex counts ensuring coprime and opposite parity.

    This is a module-level pure function so the LRU cache is keyed on
    (width, height, spacing) instead of on a bound ``self``, avoiding
    the GC-preventing reference that ``lru_cache`` creates when used on
    instance methods (Ruff B019).

    Parameters
    ----------
    width : float
        Scan width in degrees.
    height : float
        Scan height in degrees.
    spacing : float
        Row spacing in degrees.

    Returns
    -------
    x_numvert : int
        Number of vertices along x axis.
    y_numvert : int
        Number of vertices along y axis.
    amp_x : float
        X amplitude (half-width) in degrees.
    amp_y : float
        Y amplitude (half-height) in degrees.
    """
    vert_spacing = math.sqrt(2) * spacing

    x_numvert = math.ceil(width / vert_spacing)
    y_numvert = math.ceil(height / vert_spacing)

    if x_numvert % 2 == y_numvert % 2:
        if x_numvert >= y_numvert:
            y_numvert += 1
        else:
            x_numvert += 1

    num_vert = [x_numvert, y_numvert]
    most_i = num_vert.index(max(x_numvert, y_numvert))

    while math.gcd(num_vert[0], num_vert[1]) != 1:
        num_vert[most_i] += 2

    x_numvert = num_vert[0]
    y_numvert = num_vert[1]

    amp_x = x_numvert * vert_spacing / 2
    amp_y = y_numvert * vert_spacing / 2

    return x_numvert, y_numvert, amp_x, amp_y


def compute_pong_period(config: PongScanConfig) -> tuple[float, int, int]:
    """Compute the fundamental period of a Pong scan and its vertex counts.

    The Pong pattern uses two Fourier-approximated triangle waves whose
    periods are coprime, so the pattern repeats only after
    ``x_numvert * y_numvert`` turnarounds of the faster axis. This helper
    computes that period (and the vertex counts) without instantiating
    a :class:`PongScanPattern`.

    This is the canonical entry point for external code (e.g. the
    scan_patterns cross-validation reference) that needs the period
    implied by a :class:`PongScanConfig`.

    Parameters
    ----------
    config : PongScanConfig
        The Pong scan configuration.

    Returns
    -------
    period : float
        The fundamental period of the Pong pattern in seconds.
    x_numvert : int
        Number of vertices along the x axis.
    y_numvert : int
        Number of vertices along the y axis.

    Examples
    --------
    >>> from fyst_trajectories.patterns import PongScanConfig, compute_pong_period
    >>> cfg = PongScanConfig(
    ...     timestep=0.1,
    ...     width=2.0,
    ...     height=2.0,
    ...     spacing=0.1,
    ...     velocity=0.5,
    ...     num_terms=4,
    ...     angle=0.0,
    ... )
    >>> period, nx, ny = compute_pong_period(cfg)
    """
    x_numvert, y_numvert, _, _ = _compute_pong_vertices(config.width, config.height, config.spacing)
    vert_spacing = math.sqrt(2) * config.spacing
    vavg = config.velocity / math.sqrt(2)
    period = x_numvert * y_numvert * vert_spacing * 2 / vavg
    return period, x_numvert, y_numvert


@register_pattern("pong", config=PongScanConfig)
class PongScanPattern(CelestialPattern):
    """Pong scan pattern for uniform rectangular coverage.

    Parameters
    ----------
    ra : float
        Right Ascension of pattern center in degrees.
    dec : float
        Declination of pattern center in degrees.
    config : PongScanConfig
        Pattern configuration.

    Attributes
    ----------
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    config : PongScanConfig
        The configuration for this pattern.

    Examples
    --------
    >>> from astropy.time import Time
    >>> from fyst_trajectories.patterns import PongScanPattern, PongScanConfig
    >>> start_time = Time("2026-03-15T04:00:00", scale="utc")
    >>> config = PongScanConfig(
    ...     timestep=0.1,
    ...     width=2.0,
    ...     height=2.0,
    ...     spacing=0.1,
    ...     velocity=0.5,
    ...     num_terms=4,
    ...     angle=0.0,
    ... )
    >>> pattern = PongScanPattern(ra=180.0, dec=-30.0, config=config)
    >>> trajectory = pattern.generate(site, duration=300.0, start_time=start_time)
    """

    def __init__(
        self,
        ra: float,
        dec: float,
        config: PongScanConfig,
    ):
        super().__init__(ra, dec)
        self.config = config

    @property
    def name(self) -> str:
        return "pong"

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

        x_numvert, y_numvert, amp_x, amp_y = self._compute_vertices()

        vert_spacing = math.sqrt(2) * self.config.spacing
        vavg = self.config.velocity / math.sqrt(2)

        peri_x = x_numvert * vert_spacing * 2 / vavg
        peri_y = y_numvert * vert_spacing * 2 / vavg

        n_points = int(round(duration / self.config.timestep)) + 1
        times = np.linspace(0, duration, n_points)

        x_offsets = self._fourier_triangle_wave(self.config.num_terms, amp_x, times, peri_x)
        y_offsets = self._fourier_triangle_wave(self.config.num_terms, amp_y, times, peri_y)

        if self.config.angle != 0.0:
            angle_rad = math.radians(self.config.angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            x_rot = x_offsets * cos_a - y_offsets * sin_a
            y_rot = x_offsets * sin_a + y_offsets * cos_a
            x_offsets = x_rot
            y_offsets = y_rot

        return times, x_offsets, y_offsets

    def generate(
        self,
        site: Site,
        duration: float,
        start_time: Time | None,
        atmosphere: AtmosphericConditions | None = None,
    ) -> Trajectory:
        """Generate the Pong scan trajectory.

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
                "start_time is required for PongScanPattern (celestial pattern). "
                "Provide an astropy Time object."
            )

        coords = Coordinates(site, atmosphere=atmosphere)

        times, x_offsets, y_offsets = self.generate_offsets(duration)

        # Flag turnaround samples using offset-frame speed so the threshold
        # is independent of elevation (az coordinate velocity varies with cos(el)).
        x_vel = np.gradient(x_offsets, times)
        y_vel = np.gradient(y_offsets, times)
        speed = np.sqrt(x_vel**2 + y_vel**2)
        scan_flag = np.full(len(times), SCAN_FLAG_TURNAROUND, dtype=np.int8)
        scan_flag[speed >= _TURNAROUND_SPEED_THRESHOLD * self.config.velocity] = SCAN_FLAG_SCIENCE

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

        with wrap_bounds_error(f"RA={self.ra:.3f} Dec={self.dec:.3f}", start_time.iso):
            validate_trajectory_bounds(site, az, el)

        return Trajectory(
            times=times,
            az=az,
            el=el,
            az_vel=az_vel,
            el_vel=el_vel,
            start_time=start_time,
            metadata=self.get_metadata(),
            coordsys="altaz",
            scan_flag=scan_flag,
        )

    def get_metadata(self) -> TrajectoryMetadata:
        """Get pattern metadata.

        Returns
        -------
        TrajectoryMetadata
            Metadata including pattern type and parameters.
        """
        x_numvert, y_numvert, _, _ = self._compute_vertices()
        vert_spacing = math.sqrt(2) * self.config.spacing
        vavg = self.config.velocity / math.sqrt(2)
        period = x_numvert * y_numvert * vert_spacing * 2 / vavg

        return TrajectoryMetadata(
            pattern_type=self.name,
            pattern_params={
                "width": self.config.width,
                "height": self.config.height,
                "spacing": self.config.spacing,
                "velocity": self.config.velocity,
                "num_terms": self.config.num_terms,
                "angle": self.config.angle,
                "period": period,
                "x_numvert": x_numvert,
                "y_numvert": y_numvert,
            },
            center_ra=self.ra,
            center_dec=self.dec,
            input_frame="icrs",
        )

    def _compute_vertices(self) -> tuple[int, int, float, float]:
        """Compute vertex counts ensuring coprime and opposite parity.

        Delegates to the module-level ``_compute_pong_vertices()`` so the
        LRU cache is keyed on config values, not on ``self``.

        Returns
        -------
        x_numvert : int
            Number of vertices along x axis.
        y_numvert : int
            Number of vertices along y axis.
        amp_x : float
            X amplitude (half-width) in degrees.
        amp_y : float
            Y amplitude (half-height) in degrees.
        """
        return _compute_pong_vertices(self.config.width, self.config.height, self.config.spacing)

    def _fourier_triangle_wave(
        self,
        num_terms: int,
        amplitude: float,
        t: np.ndarray,
        period: float,
    ) -> np.ndarray:
        """Compute Fourier series approximation of triangle wave.

        Parameters
        ----------
        num_terms : int
            Number of Fourier terms to use.
        amplitude : float
            Peak amplitude of the wave.
        t : np.ndarray
            Time values at which to evaluate.
        period : float
            Period of the wave.

        Returns
        -------
        np.ndarray
            Wave values at each time point.
        """
        n_harmonics = num_terms * 2 - 1

        a = (8 * amplitude) / (math.pi**2)
        b = 2 * math.pi / period

        result = np.zeros_like(t)
        for n in range(1, n_harmonics + 1, 2):
            c = ((-1) ** ((n - 1) // 2)) / (n**2)
            result += c * np.sin(b * n * t)

        result *= a
        return result
