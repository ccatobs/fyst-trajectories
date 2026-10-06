"""Pong (curvy box) scan pattern.

See Scott & van Engelen 2005, "SCAN Mode Strategies for SCUBA-2", SCUBA-2
Project document SC2/ANA/S210/005, section 4.3, for algorithm details.
"""

import math

import numpy as np
from astropy.time import Time

from ..site import AtmosphericConditions, Site
from ..trajectory import Trajectory, TrajectoryMetadata
from .base import CelestialPattern
from .configs import PongAltAzScanConfig, PongScanConfig
from .registry import register_pattern
from .utils import (
    _SCIENCE_SPEED_FRACTION,
    _celestial_offsets_to_trajectory,
    _flag_by_offset_speed,
    validate_sample_count,
)


def _compute_pong_vertices(
    width: float, height: float, spacing: float
) -> tuple[int, int, float, float]:
    """Compute vertex counts ensuring coprime and opposite parity.

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


def compute_pong_period(
    config: PongScanConfig | PongAltAzScanConfig,
) -> tuple[float, int, int]:
    """Compute the fundamental period of a Pong scan and its vertex counts.

    The Pong pattern uses two Fourier-approximated triangle waves whose vertex
    counts are coprime, so the pattern repeats only after the x axis has
    completed ``y_numvert`` full cycles and the y axis ``x_numvert``. This
    helper computes that period (and the vertex counts) without instantiating
    a :class:`PongScanPattern`.

    This is the canonical entry point for external code (e.g. the
    scan_patterns cross-validation reference) that needs the period
    implied by a :class:`PongScanConfig` or a :class:`PongAltAzScanConfig`.

    Parameters
    ----------
    config : PongScanConfig or PongAltAzScanConfig
        The Pong scan configuration. Only the on-sky fields the two
        configurations share (``width``, ``height``, ``spacing`` and
        ``velocity``) are read, so both give the same period for the same
        on-sky geometry.

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
    ...     velocity=0.4,
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


def _pong_peak_offsets(config: PongScanConfig | PongAltAzScanConfig) -> tuple[float, float]:
    """Return the largest offsets the pattern reaches along its own x and y axes, in degrees.

    Each axis follows the truncated Fourier series of a triangle wave whose
    amplitude :func:`_compute_pong_vertices` sets. At a quarter period every
    odd harmonic is at its positive peak, so the series peaks there at
    ``amplitude * 8 / pi**2 * sum(1 / n**2)`` over its ``num_terms`` odd
    harmonics ``n``, short of the amplitude, and nowhere exceeds that. The
    pattern touches these offsets on the sides of its box and stays inside
    the box they span, which ``angle`` then rotates.
    """
    _, _, amp_x, amp_y = _compute_pong_vertices(config.width, config.height, config.spacing)
    harmonics = np.arange(1, 2 * config.num_terms, 2)
    peak = 8.0 / math.pi**2 * float(np.sum(1.0 / harmonics**2))
    return amp_x * peak, amp_y * peak


@register_pattern("pong", config=PongScanConfig)
class PongScanPattern(CelestialPattern):
    """Pong (curvy box) scan about a tracked RA/Dec center.

    Parameters
    ----------
    ra : float
        Right Ascension of pattern center in degrees.
    dec : float
        Declination of pattern center in degrees.
    config : PongScanConfig or PongAltAzScanConfig
        Pattern configuration. A :class:`PongAltAzScanConfig` is accepted
        for the fields it shares with :class:`PongScanConfig`; its
        horizon-frame centre (``az_center``, ``el_center``) is not read, and
        the pattern is centred on ``ra`` and ``dec``.

    Attributes
    ----------
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    config : PongScanConfig or PongAltAzScanConfig
        The configuration for this pattern.

    Examples
    --------
    >>> from astropy.time import Time
    >>> from fyst_trajectories.patterns import PongScanPattern, PongScanConfig
    >>> start_time = Time("2026-03-15T01:00:00", scale="utc")
    >>> config = PongScanConfig(
    ...     timestep=0.1,
    ...     width=2.0,
    ...     height=2.0,
    ...     spacing=0.1,
    ...     velocity=0.4,
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
        config: PongScanConfig | PongAltAzScanConfig,
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
        n_points = validate_sample_count(duration, self.config.timestep)

        x_numvert, y_numvert, amp_x, amp_y = self._compute_vertices()

        vert_spacing = math.sqrt(2) * self.config.spacing
        vavg = self.config.velocity / math.sqrt(2)

        peri_x = x_numvert * vert_spacing * 2 / vavg
        peri_y = y_numvert * vert_spacing * 2 / vavg

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
                "start_time is required for PongScanPattern (celestial pattern). "
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
        period, x_numvert, y_numvert = compute_pong_period(self.config)

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

        Delegates to the module-level ``_compute_pong_vertices()``.

        Returns
        -------
        tuple
            ``(x_numvert, y_numvert, amp_x, amp_y)``; see
            ``_compute_pong_vertices()``.
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
