"""Configuration dataclasses for scan patterns.

Each scan pattern has an associated config class that holds its
parameters. All configs inherit from ScanConfig which provides
the common timestep parameter.

All config classes are frozen (immutable) after creation.
"""

import warnings
from dataclasses import dataclass
from typing import ClassVar

from ..coordinates import SOLAR_SYSTEM_BODIES
from ..exceptions import PointingWarning

# ---------------------------------------------------------------------------
# Soft validation thresholds
#
# These are advisory upper bounds used to emit PointingWarning when a
# parameter value looks unusually large.  They are NOT hard telescope
# limits (those live in Site.telescope_limits, built from site.py constants).
# Exceeding these values is allowed. The warning is informational only.
# ---------------------------------------------------------------------------
MAX_REASONABLE_SCAN_WIDTH_DEG: float = 30.0
"""Maximum scan width/height (or azimuth throw) before a warning is issued."""

MAX_REASONABLE_DAISY_RADIUS_DEG: float = 15.0
"""Maximum Daisy scan radius before a warning is issued."""

MAX_REASONABLE_VELOCITY_DEG_S: float = 5.0
"""Maximum scan velocity before a warning is issued."""

MAX_REASONABLE_ACCELERATION_DEG_S2: float = 3.0
"""Maximum scan acceleration before a warning is issued."""


@dataclass(frozen=True)
class ScanConfig:
    """Base configuration for all scan patterns.

    Parameters
    ----------
    timestep : float
        Time between trajectory points in seconds.

    Raises
    ------
    ValueError
        If timestep is not positive.
    """

    timestep: float

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.timestep <= 0:
            raise ValueError(f"timestep must be positive, got {self.timestep}")


@dataclass(frozen=True)
class ConstantElScanConfig(ScanConfig):
    """Configuration for constant elevation scan.

    Parameters
    ----------
    az_start : float
        Starting azimuth in degrees.
    az_stop : float
        Ending azimuth in degrees.
    elevation : float
        Fixed elevation in degrees.
    az_speed : float
        Azimuth scan speed in azimuth coordinate degrees/second
        (not on-sky). The on-sky speed is
        ``az_speed * cos(elevation)``. This is the value sent
        directly to the Vertex ACU.
    az_accel : float
        Azimuth acceleration in azimuth coordinate
        degrees/second^2 (not on-sky).
    n_scans : int
        Number of one-way scan sweeps (legs).
    timestep : float
        Time between trajectory points in seconds.

    Raises
    ------
    ValueError
        If az_speed or az_accel is not positive.
    """

    pattern_name: ClassVar[str] = "constant_el"

    az_start: float
    az_stop: float
    elevation: float
    az_speed: float
    az_accel: float
    n_scans: int

    def __post_init__(self) -> None:
        """Validate constant elevation scan parameters."""
        # Call parent validation
        super().__post_init__()
        if self.az_speed <= 0:
            raise ValueError(f"az_speed must be positive, got {self.az_speed}")
        if self.az_accel <= 0:
            raise ValueError(f"az_accel must be positive, got {self.az_accel}")
        if self.n_scans < 1:
            raise ValueError(f"n_scans must be at least 1, got {self.n_scans}")
        az_throw = abs(self.az_stop - self.az_start)
        if az_throw > MAX_REASONABLE_SCAN_WIDTH_DEG:
            warnings.warn(
                f"Azimuth throw {az_throw:.1f} deg is unusually large "
                f"(> {MAX_REASONABLE_SCAN_WIDTH_DEG} deg).",
                PointingWarning,
                stacklevel=2,
            )
        if self.az_speed > MAX_REASONABLE_VELOCITY_DEG_S:
            warnings.warn(
                f"Azimuth speed {self.az_speed} deg/s is unusually high "
                f"(> {MAX_REASONABLE_VELOCITY_DEG_S} deg/s).",
                PointingWarning,
                stacklevel=2,
            )
        if self.az_accel > MAX_REASONABLE_ACCELERATION_DEG_S2:
            warnings.warn(
                f"Azimuth acceleration {self.az_accel} deg/s^2 is unusually high "
                f"(> {MAX_REASONABLE_ACCELERATION_DEG_S2} deg/s^2).",
                PointingWarning,
                stacklevel=2,
            )


@dataclass(frozen=True)
class PongScanConfig(ScanConfig):
    """Configuration for Pong (curvy box) scan.

    The Pong pattern is a closed-path scan optimized for uniformly covering
    rectangular regions. It uses Fourier-approximated triangle waves to
    create smooth turnarounds at the edges while maintaining efficient
    coverage.

    Parameters
    ----------
    width : float
        Width of scan region in degrees. Must be positive.
    height : float
        Height of scan region in degrees. Must be positive.
    spacing : float
        Space between scan lines in degrees. Must be positive.
    velocity : float
        Total scan velocity in sky-offset degrees/second. This is
        the speed in the tangent plane, not azimuth coordinate
        velocity. Must be positive.
    num_terms : int
        Fourier terms for triangle wave approximation.
    angle : float
        Rotation angle of pattern in degrees.
    timestep : float
        Time between trajectory points in seconds.

    Raises
    ------
    ValueError
        If width, height, spacing, or velocity is not positive.
        If num_terms is less than 1.

    Notes
    -----
    The scan geometry is computed using a flat-sky (tangent-plane)
    approximation.  This is accurate for scan dimensions up to about
    10 degrees; beyond that, distortion at the field edges becomes
    significant. For very large scans, consider tiling with smaller
    overlapping fields.
    """

    pattern_name: ClassVar[str] = "pong"

    width: float
    height: float
    spacing: float
    velocity: float
    num_terms: int
    angle: float

    def __post_init__(self) -> None:
        """Validate pong scan parameters."""
        super().__post_init__()
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")
        if self.spacing <= 0:
            raise ValueError(f"spacing must be positive, got {self.spacing}")
        if self.velocity <= 0:
            raise ValueError(f"velocity must be positive, got {self.velocity}")
        if self.num_terms < 1:
            raise ValueError(f"num_terms must be at least 1, got {self.num_terms}")
        if self.width > MAX_REASONABLE_SCAN_WIDTH_DEG:
            warnings.warn(
                f"Scan width {self.width} deg is unusually large "
                f"(> {MAX_REASONABLE_SCAN_WIDTH_DEG} deg).",
                PointingWarning,
                stacklevel=2,
            )
        if self.height > MAX_REASONABLE_SCAN_WIDTH_DEG:
            warnings.warn(
                f"Scan height {self.height} deg is unusually large "
                f"(> {MAX_REASONABLE_SCAN_WIDTH_DEG} deg).",
                PointingWarning,
                stacklevel=2,
            )
        if self.velocity > MAX_REASONABLE_VELOCITY_DEG_S:
            warnings.warn(
                f"Scan velocity {self.velocity} deg/s is unusually high "
                f"(> {MAX_REASONABLE_VELOCITY_DEG_S} deg/s).",
                PointingWarning,
                stacklevel=2,
            )


@dataclass(frozen=True)
class DaisyScanConfig(ScanConfig):
    """Configuration for Daisy (Constant Velocity petal) scan.

    The Daisy pattern is a constant-velocity pattern optimized for point sources.
    The telescope moves in curved petal-shaped paths that repeatedly cross
    the center of the field, ensuring good coverage of the central region.

    Parameters
    ----------
    radius : float
        Characteristic radius R0 in degrees. Must be positive.
    velocity : float
        Scan velocity in sky-offset degrees/second. This is the
        speed in the tangent plane, not azimuth coordinate
        velocity. Must be positive.
    turn_radius : float
        Radius of curvature for turns in degrees. Must be positive.
    avoidance_radius : float
        Radius to avoid near center in degrees. Must be non-negative.
    start_acceleration : float
        Ramp-up acceleration in degrees/second^2. Must be positive.
    y_offset : float
        Initial y offset in degrees.
    timestep : float
        Time between trajectory points in seconds.

    Raises
    ------
    ValueError
        If radius, velocity, or turn_radius is not positive.
        If avoidance_radius is negative.
        If start_acceleration is not positive.

    Notes
    -----
    The internal simulation uses a fixed timestep of ~1/150 s for accurate
    curve approximation during turns. Extreme parameter combinations (very
    high velocity with very tight turn_radius) may produce inaccurate curves
    because the Taylor series approximation assumes small arc lengths per step.
    If the arc length per internal step (velocity / 150) approaches the
    turn_radius, consider reducing velocity or increasing turn_radius.
    """

    pattern_name: ClassVar[str] = "daisy"

    radius: float
    velocity: float
    turn_radius: float
    avoidance_radius: float
    start_acceleration: float
    y_offset: float

    def __post_init__(self) -> None:
        """Validate daisy scan parameters."""
        super().__post_init__()
        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}")
        if self.velocity <= 0:
            raise ValueError(f"velocity must be positive, got {self.velocity}")
        if self.turn_radius <= 0:
            raise ValueError(f"turn_radius must be positive, got {self.turn_radius}")
        if self.avoidance_radius < 0:
            raise ValueError(f"avoidance_radius must be non-negative, got {self.avoidance_radius}")
        if self.start_acceleration <= 0:
            raise ValueError(f"start_acceleration must be positive, got {self.start_acceleration}")
        if self.radius > MAX_REASONABLE_DAISY_RADIUS_DEG:
            warnings.warn(
                f"Daisy radius {self.radius} deg is unusually large "
                f"(> {MAX_REASONABLE_DAISY_RADIUS_DEG} deg).",
                PointingWarning,
                stacklevel=2,
            )
        if self.velocity > MAX_REASONABLE_VELOCITY_DEG_S:
            warnings.warn(
                f"Scan velocity {self.velocity} deg/s is unusually high "
                f"(> {MAX_REASONABLE_VELOCITY_DEG_S} deg/s).",
                PointingWarning,
                stacklevel=2,
            )
        if self.start_acceleration > MAX_REASONABLE_ACCELERATION_DEG_S2:
            warnings.warn(
                f"Start acceleration {self.start_acceleration} deg/s^2 is unusually high "
                f"(> {MAX_REASONABLE_ACCELERATION_DEG_S2} deg/s^2).",
                PointingWarning,
                stacklevel=2,
            )


@dataclass(frozen=True)
class SiderealTrackConfig(ScanConfig):
    """Configuration for sidereal tracking.

    Sidereal tracking follows a fixed RA/Dec position as it moves
    across the sky due to Earth's rotation.

    Parameters
    ----------
    timestep : float
        Time between trajectory points in seconds.
    """

    pattern_name: ClassVar[str] = "sidereal"


@dataclass(frozen=True)
class PlanetTrackConfig(ScanConfig):
    """Configuration for planet tracking.

    Planet tracking follows a solar system body as it moves
    across the sky.

    Parameters
    ----------
    body : str
        Name of solar system body to track.
    timestep : float
        Time between trajectory points in seconds.

    Raises
    ------
    ValueError
        If body is not a valid solar system body name.
    """

    pattern_name: ClassVar[str] = "planet"

    body: str

    def __post_init__(self) -> None:
        """Validate planet tracking parameters."""
        super().__post_init__()
        if self.body.lower() not in SOLAR_SYSTEM_BODIES:
            raise ValueError(f"Unknown body '{self.body}'. Valid: {sorted(SOLAR_SYSTEM_BODIES)}")


@dataclass(frozen=True)
class LinearMotionConfig(ScanConfig):
    """Configuration for linear motion.

    Linear motion moves at constant velocity in Az/El space.

    Parameters
    ----------
    az_start : float
        Starting azimuth in degrees.
    el_start : float
        Starting elevation in degrees.
    az_velocity : float
        Azimuth velocity in azimuth coordinate degrees/second
        (not on-sky). The on-sky component is
        ``az_velocity * cos(elevation)``.
    el_velocity : float
        Elevation velocity in degrees/second.
    timestep : float
        Time between trajectory points in seconds.
    """

    pattern_name: ClassVar[str] = "linear"

    az_start: float
    el_start: float
    az_velocity: float
    el_velocity: float


# Mapping from config classes to pattern names for automatic pattern inference.
# This allows TrajectoryBuilder.with_config() to infer the pattern type.
# Derived from each config's pattern_name ClassVar attribute.
CONFIG_TO_PATTERN: dict[type, str] = {
    cls: cls.pattern_name
    for cls in [
        ConstantElScanConfig,
        PongScanConfig,
        DaisyScanConfig,
        SiderealTrackConfig,
        PlanetTrackConfig,
        LinearMotionConfig,
    ]
}
