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

>>> from fyst_trajectories.offsets import InstrumentOffset, boresight_to_detector
>>> offset = InstrumentOffset(dx=5.0, dy=3.0, name="Module-1")
>>> det_az, det_el = boresight_to_detector(az=180.0, el=45.0, offset=offset, field_rotation=0.0)

Compute boresight for a detector target:

>>> from fyst_trajectories.offsets import detector_to_boresight
>>> bore_az, bore_el = detector_to_boresight(
...     det_az=180.0, det_el=45.0, offset=offset, field_rotation=0.0
... )
"""

import dataclasses
from dataclasses import dataclass

import numpy as np

from .site import Site
from .trajectory import Trajectory


@dataclass(frozen=True)
class InstrumentOffset:
    """Offset of an instrument/detector from telescope boresight.

    Represents the position of an instrument or detector relative to the
    telescope boresight in the focal plane coordinate system. The offsets
    (dx, dy) are defined in the focal plane frame. When projecting onto
    the sky, the offsets are rotated by the caller-supplied field_rotation
    angle; for the az/el projections this is the mechanical Nasmyth
    rotation (see :func:`compute_focal_plane_rotation`). At zero field
    rotation, dx corresponds to the cross-elevation direction and dy to
    the elevation direction.

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

    Examples
    --------
    Create an offset for a detector module:

    >>> offset = InstrumentOffset(dx=5.0, dy=3.0, name="SFH-Module")
    >>> print(f"Offset: {offset.dx}' x {offset.dy}'")
    Offset: 5.0' x 3.0'

    Access offset in degrees:

    >>> print(f"Offset in deg: {offset.dx_deg:.4f} x {offset.dy_deg:.4f}")
    Offset in deg: 0.0833 x 0.0500

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

    safe_rho = np.where(rho < 1e-15, 1.0, rho)
    sinc_rho = np.where(rho < 1e-15, 1.0, np.sin(safe_rho) / safe_rho)

    el_rad = np.deg2rad(el)
    sin_el = np.sin(el_rad)
    cos_el = np.cos(el_rad)
    cos_rho = np.cos(rho)

    sin_new_el = sin_el * cos_rho + cos_el * dy_rad * sinc_rho
    sin_new_el = np.clip(sin_new_el, -1.0, 1.0)
    new_el_rad = np.arcsin(sin_new_el)

    delta_az_rad = np.arctan2(
        dx_rad * sinc_rho,
        cos_el * cos_rho - sin_el * dy_rad * sinc_rho,
    )

    new_az = az + np.rad2deg(delta_az_rad)
    new_el = np.rad2deg(new_el_rad)

    return new_az, new_el


_INVERSE_EARLY_EXIT_THRESHOLD: float = 1e-12
"""Iterative refinement convergence threshold in degrees (~3.6 nanoarcsec)."""

_INVERSE_FAILURE_THRESHOLD: float = 1e-6
"""Degrees (~3.6 milliarcsec) above which _offset_inverse raises RuntimeError."""

_INVERSE_MAX_ITERATIONS: int = 20
"""Maximum refinement iterations in _offset_inverse."""

_POLE_GUARD_DEG: float = 1e-6
"""Elevation within this many degrees of +/-90 makes azimuth indeterminate.

At the pole the forward map collapses every boresight azimuth onto the same
detector position, so the residual-based convergence check in
:func:`_offset_inverse` reports success while the recovered azimuth is
arbitrary. The guard raises instead of returning a silently-wrong azimuth.
"""


def _offset_inverse(
    det_az: float | np.ndarray,
    det_el: float | np.ndarray,
    dx_rot_deg: float | np.ndarray,
    dy_rot_deg: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Invert spherical offset to recover original Az/El.

    Given a detector position and the (already field-rotation-rotated)
    offset, compute the boresight position. Uses the forward formula with
    negated offsets plus iterative refinement; the round-trip precision is
    typically sub-microarcsecond in practice and enforced below the
    ~3.6 mas failure threshold.

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

    Raises
    ------
    RuntimeError
        If the detector or boresight elevation is within
        :data:`_POLE_GUARD_DEG` of the pole (+/-90 deg), where azimuth is
        degenerate and the residual check cannot validate it; or if the
        iterative refinement residual exceeds
        :data:`_INVERSE_FAILURE_THRESHOLD` after all iterations.

    Notes
    -----
    The closed-form inverse (negated offsets applied via the forward
    formula) has round-trip error of order ``rho^2``. The refinement is
    fixed-point iteration (the residual is added back to the previous
    estimate; no Jacobian is computed); measured at PrimeCam-scale
    offsets it drops the residual by several orders of magnitude per
    iteration.

    The forward residual ``(d_az, d_el)`` is degenerate at the pole: every
    boresight azimuth maps a pole-elevation detector to the same position,
    so the residual can be ~0 while the recovered azimuth is arbitrary. A
    near-pole guard (:data:`_POLE_GUARD_DEG`) catches this before the residual
    check can report a false convergence.
    """
    bore_az, bore_el = _offset_forward(det_az, det_el, -dx_rot_deg, -dy_rot_deg)

    # Pole guard: azimuth is indeterminate when either the detector or the
    # boresight lands within _POLE_GUARD_DEG of +/-90 deg. The residual-based
    # convergence check below cannot detect this (it is ~0 for any azimuth at
    # the pole), so it would otherwise return a silently-wrong azimuth.
    near_pole = (np.abs(np.abs(det_el) - 90.0) < _POLE_GUARD_DEG) | (
        np.abs(np.abs(bore_el) - 90.0) < _POLE_GUARD_DEG
    )
    if np.any(near_pole):
        raise RuntimeError(
            "_offset_inverse cannot resolve azimuth at the pole: detector or "
            "boresight elevation is within "
            f"{_POLE_GUARD_DEG:g} deg of +/-90 deg, where azimuth is degenerate "
            "(every boresight azimuth maps to the same pole position, so the "
            "residual check cannot validate it). This requires an extreme offset "
            "placing the detector at the zenith and is far outside any realistic "
            "PrimeCam pointing envelope."
        )

    # Track the worst residual ever seen so the diagnostic on failure can
    # report the full convergence history. The failure check itself uses
    # only the last-iteration residual: for the contractive map underlying
    # this fixed-point iteration, a converged endpoint is the user-visible
    # answer regardless of early-iteration transients. (Non-monotone
    # convergence does occur for unrealistic-large offsets near the zenith
    # singularity, e.g. dx=117 deg at el=86 deg under property-based fuzzing,
    # but the iteration still lands sub-microarcsecond at the end.)
    worst_err = 0.0

    for _ in range(_INVERSE_MAX_ITERATIONS):
        det_az_check, det_el_check = _offset_forward(bore_az, bore_el, dx_rot_deg, dy_rot_deg)
        d_az = det_az - det_az_check
        d_el = det_el - det_el_check
        bore_az = bore_az + d_az
        bore_el = bore_el + d_el
        worst_err = max(worst_err, float(np.max(np.abs(d_az))), float(np.max(np.abs(d_el))))
        if np.all(np.abs(d_az) < _INVERSE_EARLY_EXIT_THRESHOLD) and np.all(
            np.abs(d_el) < _INVERSE_EARLY_EXIT_THRESHOLD
        ):
            break
    else:
        last_err = max(float(np.max(np.abs(d_az))), float(np.max(np.abs(d_el))))
        if last_err > _INVERSE_FAILURE_THRESHOLD:
            raise RuntimeError(
                f"_offset_inverse iterative refinement failed to converge after "
                f"{_INVERSE_MAX_ITERATIONS} iterations "
                f"(last residual: {last_err:.2e} deg, worst residual: {worst_err:.2e} deg). "
                f"This may indicate an extreme offset or near-zenith elevation."
            )

    return bore_az, bore_el


def _rotate_offset(
    offset: InstrumentOffset,
    field_rotation: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Rotate offset by field rotation angle.

    Parameters
    ----------
    offset : InstrumentOffset
        Detector offset from boresight.
    field_rotation : float or array
        Field rotation angle in degrees.

    Returns
    -------
    dx_rot : float or array
        Rotated cross-elevation offset in degrees.
    dy_rot : float or array
        Rotated elevation offset in degrees.
    """
    dx_deg = offset.dx_deg
    dy_deg = offset.dy_deg
    rot_rad = np.deg2rad(field_rotation)
    cos_rot = np.cos(rot_rad)
    sin_rot = np.sin(rot_rad)
    return dx_deg * cos_rot - dy_deg * sin_rot, dx_deg * sin_rot + dy_deg * cos_rot


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

    For a Nasmyth-mounted instrument on an alt-az telescope, the focal
    plane rotates relative to the (az, el) axes as the elevation changes.
    The field_rotation parameter accounts for this rotation when computing
    detector positions.

    Parameters
    ----------
    az : float or array
        Boresight azimuth in degrees.
    el : float or array
        Boresight elevation in degrees.
    offset : InstrumentOffset
        Detector offset from boresight.
    field_rotation : float or array
        Orientation of the focal plane relative to the horizon (az/el)
        axes, in degrees. For a Nasmyth-mounted instrument this is the
        mechanical ``nasmyth_sign * elevation + instrument_rotation``
        (plus any commanded rotator angle).

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
    Detector at Az=180.118, El=45.000
    """
    dx_rot, dy_rot = _rotate_offset(offset, field_rotation)
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
    boresight_to_detector. Uses spherical trigonometry with iterative
    refinement; the round trip is enforced to better than the ~3.6
    milliarcsecond failure threshold and typically converges far below
    that.

    Parameters
    ----------
    det_az : float or array
        Desired detector azimuth in degrees.
    det_el : float or array
        Desired detector elevation in degrees.
    offset : InstrumentOffset
        Detector offset from boresight.
    field_rotation : float or array
        Orientation of the focal plane relative to the horizon (az/el)
        axes, in degrees; see :func:`boresight_to_detector`.

    Returns
    -------
    bore_az : float or array
        Required boresight azimuth in degrees.
    bore_el : float or array
        Required boresight elevation in degrees.

    Raises
    ------
    RuntimeError
        If the requested detector position cannot be produced by any
        boresight (offset larger than the pole distance) or the
        iterative refinement fails to converge.

    Examples
    --------
    >>> offset = InstrumentOffset(dx=5.0, dy=0.0)
    >>> bore_az, bore_el = detector_to_boresight(180.0, 45.0, offset, field_rotation=0.0)
    >>> print(f"Boresight at Az={bore_az:.3f}, El={bore_el:.3f}")
    Boresight at Az=179.882, El=45.000

    Verify inverse relationship:

    >>> det_az2, det_el2 = boresight_to_detector(bore_az, bore_el, offset, field_rotation=0.0)
    >>> assert abs(det_az2 - 180.0) < 1e-6
    >>> assert abs(det_el2 - 45.0) < 1e-6
    """
    dx_rot, dy_rot = _rotate_offset(offset, field_rotation)
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
    """Compute the focal-plane rotation angle.

    Decomposes the rotation into mechanical (Nasmyth) and sky components:

        rotation = nasmyth_sign * elevation + instrument_rotation + parallactic_angle

    The mechanical part (the default, with ``parallactic_angle=0.0``) is
    the orientation of the focal plane relative to the horizon (az/el)
    axes, the rotation used by the az/el projections
    (:func:`boresight_to_detector`, :func:`detector_to_boresight`,
    :func:`apply_detector_offset`). Adding the parallactic angle gives
    the orientation relative to the celestial (equatorial) axes, used
    for sky-map orientation, image rotation, and polarization angles.

    Parameters
    ----------
    el : float or array
        Elevation in degrees.
    site : Site
        Telescope site (provides nasmyth_sign).
    offset : InstrumentOffset
        Instrument offset (provides instrument_rotation).
    parallactic_angle : float or array, optional
        Parallactic angle in degrees. Default is 0.0 (the mechanical,
        horizon-frame rotation); pass a value to obtain the
        celestial-frame orientation.

    Returns
    -------
    float or array
        Focal-plane rotation in degrees.

    See Also
    --------
    ~fyst_trajectories.coordinates.Coordinates.get_field_rotation :
        Computes the celestial-frame quantity
        ``nasmyth_sign * el + parallactic_angle`` from RA/Dec
        (no instrument_rotation).
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

    The adjustment is a horizon-frame (az/el) projection: the focal-plane
    offset is rotated by the mechanical rotation
    ``nasmyth_sign * elevation + instrument_rotation`` and inverted to a
    boresight path. Celestial and AltAz patterns behave identically (the
    trajectory's ``center_ra``/``center_dec`` metadata is not consumed).
    For the focal plane's orientation on the celestial sky (map
    orientation, image rotation, polarization angles), use
    :meth:`~fyst_trajectories.coordinates.Coordinates.get_field_rotation`.

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
    AzimuthBoundsError
        If ``validate=True`` and the adjusted trajectory exceeds azimuth limits.
    ElevationBoundsError
        If ``validate=True`` and the adjusted trajectory exceeds elevation limits.
    RuntimeError
        If the offset inversion cannot reach the requested detector
        position or fails to converge (see :func:`detector_to_boresight`).

    Notes
    -----
    **Precondition: the input trajectory must be in geometric (vacuum)
    coordinates.** This holds on every live path: ``Coordinates(site)``
    defaults to vacuum, and refraction is applied downstream at
    execution time (by exactly one of the Go TCS or the ACU). The mechanical Nasmyth term consumes
    ``trajectory.el`` directly; if the trajectory was instead built with
    ``AtmosphericConditions.for_fyst()`` (refracted, a planning/sim-only
    path), its ``el`` is in the apparent frame and the mechanical rotation
    differs from the vacuum one by ``nasmyth_sign * (refraction bump)``,
    a small boresight effect (a few arcseconds at worst, at the lowest
    elevations) at PrimeCam offset radii. ``Trajectory``
    carries no refraction flag, so a refracted input cannot be detected
    here; pair detector offsets with vacuum trajectories.

    The mechanical term ``nasmyth_sign * el`` is evaluated at the input
    trajectory's elevation (the detector/target elevation), not the returned
    boresight elevation. The two differ by the offset's elevation component
    (up to the offset radius), so a consumer that re-derives the field rotation
    from the *boresight* elevation will get a slightly different value; use the
    input (detector) elevation to reproduce it.

    Examples
    --------
    >>> from astropy.time import Time
    >>> from fyst_trajectories import get_fyst_site
    >>> from fyst_trajectories.patterns import TrajectoryBuilder, PongScanConfig
    >>> from fyst_trajectories.offsets import InstrumentOffset, apply_detector_offset
    >>>
    >>> site = get_fyst_site()
    >>> start_time = Time("2026-03-15T01:00:00", scale="utc")
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
    ...             velocity=0.3,
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
    if offset.dx == 0.0 and offset.dy == 0.0 and offset.instrument_rotation == 0.0:
        return dataclasses.replace(trajectory)

    # Horizon-frame projection: the rotation is mechanical only; the
    # parallactic angle is a horizon-to-celestial quantity and has no
    # place in an az/el projection.
    field_rotation = compute_focal_plane_rotation(trajectory.el, site, offset)

    bore_az, bore_el = detector_to_boresight(
        trajectory.az,
        trajectory.el,
        offset,
        field_rotation,
    )

    if len(trajectory.times) < 2:
        # np.gradient needs >=2 samples; boresight velocities are undefined for
        # a single sample. Sibling np.gradient sites (daisy.py, trajectory_utils)
        # guard the same way, and the builder tolerates <2-point trajectories.
        az_vel = np.zeros_like(bore_az)
        el_vel = np.zeros_like(bore_el)
    else:
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
        retune_events=trajectory.retune_events,
    )

    if validate:
        from .trajectory_utils import (
            validate_trajectory_bounds,  # pylint: disable=import-outside-toplevel
        )

        validate_trajectory_bounds(site, result.az, result.el)

    return result
