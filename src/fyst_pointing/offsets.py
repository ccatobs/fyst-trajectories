"""Instrument and detector offset transformations.

This module provides utilities for handling instrument and detector offsets
from the telescope boresight. When pointing the telescope's boresight at a
target, different instruments/detectors see different parts of the sky based
on their offset from the boresight and field rotation during observation.

Offsets are projected using spherical trigonometry (great-circle offset
formulas), which is accurate for any offset size.

The use cases are:
1. Given boresight pointing, compute where a detector observes (boresight_to_detector)
2. Given where you want a detector to point, compute boresight pointing (detector_to_boresight)
3. Apply detector offsets to entire trajectories (apply_detector_offset)

Examples
--------
Basic offset transformation:

>>> from fyst_pointing.offsets import InstrumentOffset, boresight_to_detector
>>> offset = InstrumentOffset(dx=5.0, dy=3.0, name="Module-1")
>>> det_az, det_el = boresight_to_detector(az=180.0, el=45.0, offset=offset, field_rotation=0.0)

Compute boresight for a detector target:

>>> from fyst_pointing.offsets import detector_to_boresight
>>> bore_az, bore_el = detector_to_boresight(
...     det_az=180.0, det_el=45.0, offset=offset, field_rotation=0.0
... )
"""

import warnings
from dataclasses import dataclass

import numpy as np

from .coordinates import Coordinates
from .exceptions import PointingWarning
from .site import Site
from .trajectory import Trajectory


@dataclass(frozen=True)
class InstrumentOffset:
    """Offset of an instrument/detector from telescope boresight.

    Represents the position of an instrument or detector relative to the
    telescope boresight in the focal plane coordinate system. The offsets
    (dx, dy) are defined in the focal plane frame. When projecting onto
    the sky, the offsets are rotated by the field_rotation angle (which
    combines elevation and parallactic angle for alt-az telescopes). At
    zero field rotation, dx corresponds to the cross-elevation direction
    and dy to the elevation direction.

    Parameters
    ----------
    dx : float
        X offset in arcminutes in the focal plane. At zero field rotation,
        this is the cross-elevation direction (positive = increasing azimuth).
    dy : float
        Y offset in arcminutes in the focal plane. At zero field rotation,
        this is the elevation direction (positive = increasing elevation).
    name : str, optional
        Name of the instrument/detector for identification.
    instrument_rotation : float, optional
        Fixed rotation of the instrument relative to the Nasmyth flange,
        in degrees. This accounts for instruments that are mounted at a
        rotational offset from the default orientation. Default is 0.0.

    Attributes
    ----------
    dx_deg : float
        X offset in degrees.
    dy_deg : float
        Y offset in degrees.

    Examples
    --------
    Create an offset for a detector module:

    >>> offset = InstrumentOffset(dx=5.0, dy=3.0, name="SFH-Module")
    >>> print(f"Offset: {offset.dx}' x {offset.dy}'")
    Offset: 5.0' x 3.0'

    Access offset in degrees:

    >>> print(f"Offset in deg: {offset.dx_deg:.4f} x {offset.dy_deg:.4f}")

    With instrument rotation:

    >>> offset = InstrumentOffset(dx=5.0, dy=3.0, instrument_rotation=15.0)
    """

    dx: float
    dy: float
    name: str | None = None
    instrument_rotation: float = 0.0

    @property
    def dx_deg(self) -> float:
        """X offset in degrees."""
        return self.dx / 60.0

    @property
    def dy_deg(self) -> float:
        """Y offset in degrees."""
        return self.dy / 60.0

    @classmethod
    def from_focal_plane(
        cls,
        x_mm: float,
        y_mm: float,
        plate_scale: float,
        name: str | None = None,
        instrument_rotation: float = 0.0,
    ) -> "InstrumentOffset":
        """Create an offset from focal plane physical coordinates.

        Converts physical positions in millimeters on the focal plane to
        angular offsets using the telescope plate scale.

        Parameters
        ----------
        x_mm : float
            X position on focal plane in millimeters relative to optical axis.
        y_mm : float
            Y position on focal plane in millimeters relative to optical axis.
        plate_scale : float
            Plate scale in arcsec/mm.
        name : str, optional
            Name of the instrument/detector.
        instrument_rotation : float, optional
            Instrument rotation in degrees. Default 0.0.

        Returns
        -------
        InstrumentOffset
            Offset with dx, dy converted to arcminutes.

        Examples
        --------
        >>> offset = InstrumentOffset.from_focal_plane(
        ...     x_mm=0.0,
        ...     y_mm=-461.3,
        ...     plate_scale=13.89,
        ...     name="PrimeCam-I1",
        ... )
        >>> print(f"{offset.dy:.1f} arcmin")
        -106.8 arcmin
        """
        dx_arcmin = x_mm * plate_scale / 60.0
        dy_arcmin = y_mm * plate_scale / 60.0
        return cls(
            dx=dx_arcmin,
            dy=dy_arcmin,
            name=name,
            instrument_rotation=instrument_rotation,
        )

    def __repr__(self) -> str:
        """Return string representation of the offset."""
        name_str = f", name='{self.name}'" if self.name else ""
        rot_str = (
            f", instrument_rotation={self.instrument_rotation}°"
            if self.instrument_rotation != 0.0
            else ""
        )
        return f"InstrumentOffset(dx={self.dx}', dy={self.dy}'{name_str}{rot_str})"


def _offset_forward(
    az: float | np.ndarray,
    el: float | np.ndarray,
    dx_rot_deg: float | np.ndarray,
    dy_rot_deg: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    r"""Apply spherical offset to Az/El position.

    Computes the detector position on the celestial sphere given a
    boresight position and an offset that has already been rotated by
    field rotation. Uses exact spherical trigonometry (great-circle
    offset formulas).

    Parameters
    ----------
    az : float or array
        Azimuth in degrees.
    el : float or array
        Elevation in degrees.
    dx_rot_deg : float or array
        Cross-elevation offset in degrees (after field rotation).
    dy_rot_deg : float or array
        Elevation offset in degrees (after field rotation).

    Returns
    -------
    new_az : float or array
        Offset azimuth in degrees.
    new_el : float or array
        Offset elevation in degrees.

    Notes
    -----
    The offset is parameterized by angular distance ``rho`` and position
    angle ``phi`` (measured from the elevation direction toward increasing
    azimuth):

    .. math::

        \sin(El_1) = \sin(El_0) \cos(\rho)
                     + \cos(El_0) \sin(\rho) \cos(\phi)

        \Delta Az = \arctan2(\sin(\rho) \sin(\phi),
                     \cos(El_0) \cos(\rho)
                     - \sin(El_0) \sin(\rho) \cos(\phi))

    where ``rho = sqrt(dx^2 + dy^2)`` and ``phi = atan2(dx, dy)``.

    For numerical stability, the formulas are rewritten using
    ``sinc(rho) = sin(rho) / rho`` to avoid division by zero when
    ``rho = 0``.
    """
    dx_rad = np.deg2rad(dx_rot_deg)
    dy_rad = np.deg2rad(dy_rot_deg)

    rho = np.sqrt(dx_rad**2 + dy_rad**2)

    # sinc_rho = sin(rho) / rho, with safe handling for rho near zero.
    # Use np.where with out parameter to avoid evaluating division at zero.
    safe_rho = np.where(rho < 1e-15, 1.0, rho)
    sinc_rho = np.where(rho < 1e-15, 1.0, np.sin(safe_rho) / safe_rho)

    el_rad = np.deg2rad(el)
    sin_el = np.sin(el_rad)
    cos_el = np.cos(el_rad)
    cos_rho = np.cos(rho)

    # Elevation component: sin(rho)*cos(phi) = dy_rad * sinc_rho * rho / rho
    # Simplified: sin(rho)*cos(phi) = dy_rad * sinc_rho (since phi = atan2(dx,dy))
    # More precisely: sin(rho)*cos(phi) = dy_rad * (sin(rho)/rho)
    sin_new_el = sin_el * cos_rho + cos_el * dy_rad * sinc_rho
    # Clamp to [-1, 1] to avoid numerical issues with arcsin
    sin_new_el = np.clip(sin_new_el, -1.0, 1.0)
    new_el_rad = np.arcsin(sin_new_el)

    # Azimuth component: sin(rho)*sin(phi) = dx_rad * sinc_rho
    delta_az_rad = np.arctan2(
        dx_rad * sinc_rho,
        cos_el * cos_rho - sin_el * dy_rad * sinc_rho,
    )

    new_az = az + np.rad2deg(delta_az_rad)
    new_el = np.rad2deg(new_el_rad)

    return new_az, new_el


def _offset_inverse(
    det_az: float | np.ndarray,
    det_el: float | np.ndarray,
    dx_rot_deg: float | np.ndarray,
    dy_rot_deg: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Invert spherical offset to recover original Az/El.

    Given a detector position and the (already field-rotation-rotated)
    offset, compute the boresight position. Uses the forward formula with
    negated offsets plus ten Newton refinement iterations for
    sub-microarcsecond round-trip precision.

    Parameters
    ----------
    det_az : float or array
        Detector azimuth in degrees.
    det_el : float or array
        Detector elevation in degrees.
    dx_rot_deg : float or array
        Cross-elevation offset in degrees (after field rotation).
    dy_rot_deg : float or array
        Elevation offset in degrees (after field rotation).

    Returns
    -------
    bore_az : float or array
        Boresight azimuth in degrees.
    bore_el : float or array
        Boresight elevation in degrees.

    Notes
    -----
    The closed-form inverse (negated offsets applied via the forward
    formula) has round-trip error of order ``rho^3``. Each Newton
    iteration squares the relative error. Up to 20 iterations are
    performed, with early exit when the correction drops below
    1e-12 degrees (~3.6 microarcsec). A ``RuntimeError`` is raised
    if the residual exceeds 1e-6 degrees after all iterations.
    """
    # Initial estimate: apply forward formula with negated offsets
    bore_az, bore_el = _offset_forward(det_az, det_el, -dx_rot_deg, -dy_rot_deg)

    # Newton refinement iterations. Each iteration computes the forward
    # transform from the current estimate and applies the residual as a
    # correction. Early exit at 1e-12 deg; raises RuntimeError if the
    # residual exceeds 1e-6 deg after all iterations.
    _EARLY_EXIT_THRESHOLD = 1e-12  # degrees (~3.6 microarcsec)
    _FAILURE_THRESHOLD = 1e-6  # degrees (~3.6 arcsec)
    _MAX_ITERATIONS = 20
    for _ in range(_MAX_ITERATIONS):
        det_az_check, det_el_check = _offset_forward(bore_az, bore_el, dx_rot_deg, dy_rot_deg)
        d_az = det_az - det_az_check
        d_el = det_el - det_el_check
        bore_az = bore_az + d_az
        bore_el = bore_el + d_el
        if np.all(np.abs(d_az) < _EARLY_EXIT_THRESHOLD) and np.all(
            np.abs(d_el) < _EARLY_EXIT_THRESHOLD
        ):
            break
    else:
        max_err = max(float(np.max(np.abs(d_az))), float(np.max(np.abs(d_el))))
        if max_err > _FAILURE_THRESHOLD:
            raise RuntimeError(
                f"_offset_inverse Newton iteration failed to converge after "
                f"{_MAX_ITERATIONS} iterations (max residual: {max_err:.2e} deg). "
                f"This may indicate an extreme offset or near-zenith elevation."
            )

    return bore_az, bore_el


def boresight_to_detector(
    az: float | np.ndarray,
    el: float | np.ndarray,
    offset: InstrumentOffset,
    field_rotation: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Compute detector Az/El given boresight Az/El and offset.

    Applies the instrument offset with field rotation to compute
    the actual sky position that a detector is observing given where the
    telescope boresight is pointed. Uses spherical trigonometry
    (great-circle offset formulas) for accuracy at any offset size.

    For an alt-az mounted telescope, the field of view rotates as objects
    are tracked across the sky. The field_rotation parameter accounts for
    this rotation when computing detector positions.

    Parameters
    ----------
    az : float or array
        Boresight azimuth in degrees.
    el : float or array
        Boresight elevation in degrees.
    offset : InstrumentOffset
        Detector offset from boresight.
    field_rotation : float or array
        Field rotation angle in degrees.
        For alt-az telescopes, this is typically elevation + parallactic angle.

    Returns
    -------
    det_az : float or array
        Detector azimuth in degrees.
    det_el : float or array
        Detector elevation in degrees.

    Examples
    --------
    >>> offset = InstrumentOffset(dx=5.0, dy=0.0)
    >>> det_az, det_el = boresight_to_detector(180.0, 45.0, offset, field_rotation=0.0)
    >>> print(f"Detector at Az={det_az:.3f}, El={det_el:.3f}")
    """
    dx_deg = offset.dx_deg
    dy_deg = offset.dy_deg

    rot_rad = np.deg2rad(field_rotation)
    cos_rot = np.cos(rot_rad)
    sin_rot = np.sin(rot_rad)

    dx_rot = dx_deg * cos_rot - dy_deg * sin_rot
    dy_rot = dx_deg * sin_rot + dy_deg * cos_rot

    det_az, det_el = _offset_forward(az, el, dx_rot, dy_rot)

    if np.isscalar(az) and np.isscalar(el) and np.isscalar(field_rotation):
        return float(det_az), float(det_el)
    return det_az, det_el


def detector_to_boresight(
    det_az: float | np.ndarray,
    det_el: float | np.ndarray,
    offset: InstrumentOffset,
    field_rotation: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Compute boresight Az/El to place detector at given position.

    Given where you want a detector to point on the sky, compute where
    the telescope boresight should be pointed. This is the inverse of
    boresight_to_detector. Uses spherical trigonometry with Newton
    refinement for sub-milliarcsecond round-trip precision.

    Parameters
    ----------
    det_az : float or array
        Desired detector azimuth in degrees.
    det_el : float or array
        Desired detector elevation in degrees.
    offset : InstrumentOffset
        Detector offset from boresight.
    field_rotation : float or array
        Field rotation angle in degrees.

    Returns
    -------
    bore_az : float or array
        Required boresight azimuth in degrees.
    bore_el : float or array
        Required boresight elevation in degrees.

    Examples
    --------
    >>> offset = InstrumentOffset(dx=5.0, dy=0.0)
    >>> bore_az, bore_el = detector_to_boresight(180.0, 45.0, offset, field_rotation=0.0)
    >>> print(f"Boresight at Az={bore_az:.3f}, El={bore_el:.3f}")

    Verify inverse relationship:

    >>> det_az2, det_el2 = boresight_to_detector(bore_az, bore_el, offset, field_rotation=0.0)
    >>> assert abs(det_az2 - 180.0) < 1e-6
    >>> assert abs(det_el2 - 45.0) < 1e-6
    """
    dx_deg = offset.dx_deg
    dy_deg = offset.dy_deg

    # Same rotation as forward transform
    rot_rad = np.deg2rad(field_rotation)
    cos_rot = np.cos(rot_rad)
    sin_rot = np.sin(rot_rad)

    dx_rot = dx_deg * cos_rot - dy_deg * sin_rot
    dy_rot = dx_deg * sin_rot + dy_deg * cos_rot

    bore_az, bore_el = _offset_inverse(det_az, det_el, dx_rot, dy_rot)

    if np.isscalar(det_az) and np.isscalar(det_el) and np.isscalar(field_rotation):
        return float(bore_az), float(bore_el)
    return bore_az, bore_el


def compute_focal_plane_rotation(
    el: float | np.ndarray,
    site: Site,
    offset: InstrumentOffset,
    parallactic_angle: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """Compute the total focal-plane rotation angle.

    Decomposes the rotation into mechanical (Nasmyth) and sky components:

        rotation = nasmyth_sign * elevation + instrument_rotation + parallactic_angle

    The mechanical part (nasmyth_sign * el + instrument_rotation) is always
    available from the elevation axis. The parallactic angle adds the sky
    rotation component and requires celestial coordinates.

    Parameters
    ----------
    el : float or array
        Elevation in degrees.
    site : Site
        Telescope site (provides nasmyth_sign).
    offset : InstrumentOffset
        Instrument offset (provides instrument_rotation).
    parallactic_angle : float or array, optional
        Parallactic angle in degrees. Default is 0.0.

    Returns
    -------
    float or array
        Total focal-plane rotation in degrees.

    See Also
    --------
    fyst_pointing.coordinates.Coordinates.get_field_rotation :
        Convenience method that computes ``nasmyth_sign * el + parallactic_angle``
        from RA/Dec (no instrument_rotation).  Use ``compute_focal_plane_rotation``
        when instrument_rotation is needed.
    """
    mechanical = site.nasmyth_sign * el + offset.instrument_rotation
    return mechanical + parallactic_angle


def apply_detector_offset(
    trajectory: Trajectory,
    offset: InstrumentOffset,
    site: Site,
    validate: bool = False,
) -> Trajectory:
    """Apply detector offset to trajectory, accounting for field rotation.

    Returns a new trajectory with boresight positions adjusted so that the
    specified detector observes the original target positions. This is useful
    when you have generated a trajectory for a celestial target but want a
    specific off-axis detector to track that target instead of the boresight.

    The field rotation is decomposed into:

    - **Mechanical rotation**: ``nasmyth_sign * elevation + instrument_rotation``
      (always available from the elevation axis).
    - **Parallactic angle**: adds sky rotation when celestial coordinates
      (RA/Dec) are available in the trajectory metadata.

    For celestial patterns (with RA/Dec), both components are used. For
    AltAz patterns (without RA/Dec), only mechanical rotation is used,
    which is physically correct for focal-plane-to-AltAz conversion.

    Parameters
    ----------
    trajectory : Trajectory
        Original trajectory (assumed to be for the desired detector pointing).
    offset : InstrumentOffset
        Detector offset from boresight.
    site : Site
        Telescope site configuration (needed for field rotation calculation).
    validate : bool, optional
        If True, run ``validate_trajectory_bounds`` on the adjusted
        trajectory and raise on violations. Default is False (no
        post-adjustment validation).

    Returns
    -------
    Trajectory
        New trajectory with adjusted boresight positions.

    Raises
    ------
    ValueError
        If trajectory has no start_time set (needed for field rotation).
    AzimuthBoundsError
        If ``validate=True`` and the adjusted trajectory exceeds azimuth limits.
    ElevationBoundsError
        If ``validate=True`` and the adjusted trajectory exceeds elevation limits.

    Warns
    -----
    PointingWarning
        When no celestial coordinates are available for parallactic angle
        computation (AltAz trajectories).

    Notes
    -----
    The returned trajectory has the same metadata as the input, but with
    the az/el positions adjusted so that when the telescope follows this
    trajectory, the detector observes the original target.

    The parallactic angle is computed at the pattern center coordinates
    (``trajectory.center_ra``, ``trajectory.center_dec``) for all timesteps.
    This is an approximation; for very large scan patterns (many degrees),
    the actual parallactic angle varies across the field.

    Examples
    --------
    >>> from astropy.time import Time
    >>> from fyst_pointing import get_fyst_site
    >>> from fyst_pointing.patterns import TrajectoryBuilder, PongScanConfig
    >>> from fyst_pointing.offsets import InstrumentOffset, apply_detector_offset
    >>>
    >>> site = get_fyst_site()
    >>> start_time = Time("2026-03-15T04:00:00", scale="utc")
    >>> offset = InstrumentOffset(dx=5.0, dy=3.0, name="Mod2")
    >>>
    >>> # Generate trajectory for target (start_time required for celestial patterns)
    >>> trajectory = (
    ...     TrajectoryBuilder(site)
    ...     .at(ra=180.0, dec=-30.0)
    ...     .with_config(
    ...         PongScanConfig(
    ...             timestep=0.1,
    ...             width=1.0,
    ...             height=1.0,
    ...             spacing=0.1,
    ...             velocity=0.5,
    ...             num_terms=4,
    ...             angle=0.0,
    ...         )
    ...     )
    ...     .duration(60.0)
    ...     .starting_at(start_time)
    ...     .build()
    ... )
    >>>
    >>> # Adjust so Mod2 observes the target instead of boresight
    >>> adjusted = apply_detector_offset(trajectory, offset, site)
    """
    if trajectory.start_time is None:
        raise ValueError("Trajectory must have start_time set for field rotation calculation")

    # Early exit: zero offset with zero instrument rotation is a no-op.
    # Return a shallow copy to avoid aliasing mutable arrays.
    if offset.dx == 0.0 and offset.dy == 0.0 and offset.instrument_rotation == 0.0:
        return Trajectory(
            times=trajectory.times.copy(),
            az=trajectory.az.copy(),
            el=trajectory.el.copy(),
            az_vel=trajectory.az_vel.copy(),
            el_vel=trajectory.el_vel.copy(),
            start_time=trajectory.start_time,
            coordsys=trajectory.coordsys,
            epoch=trajectory.epoch,
            metadata=trajectory.metadata,
            scan_flag=trajectory.scan_flag,
        )

    coords = Coordinates(site, atmosphere=None)

    abs_times = trajectory.get_absolute_times()

    # Mechanical rotation: always available from elevation
    mechanical_rotation = site.nasmyth_sign * trajectory.el + offset.instrument_rotation

    if trajectory.center_ra is not None and trajectory.center_dec is not None:
        # Celestial patterns: add parallactic angle for full accuracy
        ra_arr = np.full(len(trajectory.times), trajectory.center_ra)
        dec_arr = np.full(len(trajectory.times), trajectory.center_dec)
        pa = coords.get_parallactic_angle(ra_arr, dec_arr, obstime=abs_times)
        field_rotation = mechanical_rotation + pa
    else:
        # AltAz/planet patterns: mechanical rotation only
        # This is physically correct per NIKA2/KOSMA models
        warnings.warn(
            "Parallactic angle unavailable (no RA/Dec in trajectory metadata). "
            "Using mechanical rotation only "
            "(nasmyth_sign * elevation + instrument_rotation). "
            "This is correct for AltAz and planet-tracking patterns where the "
            "sky rotation is already embedded in the Az/El coordinates. For "
            "celestial patterns, ensure center_ra/center_dec metadata is set so "
            "the full field rotation (including parallactic angle) is applied.",
            PointingWarning,
            stacklevel=2,
        )
        field_rotation = mechanical_rotation

    bore_az, bore_el = detector_to_boresight(
        trajectory.az,
        trajectory.el,
        offset,
        field_rotation,
    )

    # Numerical differentiation required as offset transformation is nonlinear,
    # so original analytical velocities don't apply after the spherical offset.
    az_vel = np.gradient(bore_az, trajectory.times)
    el_vel = np.gradient(bore_el, trajectory.times)

    result = Trajectory(
        times=trajectory.times.copy(),
        az=bore_az,
        el=bore_el,
        az_vel=az_vel,
        el_vel=el_vel,
        start_time=trajectory.start_time,
        metadata=trajectory.metadata,
        coordsys=trajectory.coordsys,
        epoch=trajectory.epoch,
        scan_flag=trajectory.scan_flag,
    )

    if validate:
        from .trajectory_utils import (
            validate_trajectory_bounds,  # pylint: disable=import-outside-toplevel
        )

        validate_trajectory_bounds(site, result.az, result.el)

    return result
