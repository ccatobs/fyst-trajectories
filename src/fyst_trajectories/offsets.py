"""Instrument and detector offset transformations.

This module provides utilities for handling instrument and detector offsets
from the telescope boresight. When pointing the telescope's boresight at a
target, different instruments/detectors see different parts of the sky based
on their offset from the boresight and the focal-plane rotation during observation.

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
>>> det_az, det_el = boresight_to_detector(
...     az=180.0, el=45.0, offset=offset, focal_plane_rotation=0.0
... )

Compute boresight for a detector target:

>>> from fyst_trajectories.offsets import detector_to_boresight
>>> bore_az, bore_el = detector_to_boresight(
...     det_az=180.0, det_el=45.0, offset=offset, focal_plane_rotation=0.0
... )
"""

import dataclasses
from dataclasses import dataclass

import numpy as np

from .exceptions import OffsetInversionError
from .site import Site
from .trajectory import Trajectory
from .trajectory_utils import validate_trajectory_bounds


@dataclass(frozen=True)
class InstrumentOffset:
    """Offset of an instrument/detector from telescope boresight.

    Represents the position of an instrument or detector relative to the
    telescope boresight in the focal plane coordinate system. The offsets
    (dx, dy) are defined in the focal plane frame. When projecting onto
    the sky, the offsets are rotated by the caller-supplied
    focal_plane_rotation angle; for the az/el projections this is the
    mechanical Nasmyth rotation (see :func:`compute_focal_plane_rotation`).
    At zero rotation, dx corresponds to the cross-elevation direction and
    dy to the elevation direction.

    Parameters
    ----------
    dx : float
        X offset in arcminutes in the focal plane. At zero focal-plane rotation,
        this is the cross-elevation direction (positive = increasing azimuth).
    dy : float
        Y offset in arcminutes in the focal plane. At zero focal-plane rotation,
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
            f", instrument_rotation={self.instrument_rotation} deg"
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
    the focal-plane rotation. Uses exact spherical trigonometry
    (great-circle offset formulas).

    Parameters
    ----------
    az : float or array
        Azimuth in degrees.
    el : float or array
        Elevation in degrees.
    dx_rot_deg : float or array
        Cross-elevation offset in degrees (after the focal-plane rotation).
    dy_rot_deg : float or array
        Elevation offset in degrees (after the focal-plane rotation).

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
"""Degrees (~3.6 milliarcsec) above which _offset_inverse refuses to converge."""

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

    Given a detector position and the offset (already rotated by the
    focal-plane rotation), compute the boresight position. Uses the
    forward formula with negated offsets plus iterative refinement; the
    round-trip precision is typically sub-microarcsecond in practice and
    enforced below the ~3.6 mas failure threshold.

    Parameters
    ----------
    det_az : float or array
        Detector azimuth in degrees.
    det_el : float or array
        Detector elevation in degrees.
    dx_rot_deg : float or array
        Cross-elevation offset in degrees (after the focal-plane rotation).
    dy_rot_deg : float or array
        Elevation offset in degrees (after the focal-plane rotation).

    Returns
    -------
    bore_az : float or array
        Boresight azimuth in degrees.
    bore_el : float or array
        Boresight elevation in degrees.

    Raises
    ------
    OffsetInversionError
        If the detector or boresight elevation is within
        :data:`_POLE_GUARD_DEG` of the pole (+/-90 deg), where azimuth is
        degenerate and the residual check cannot validate it; or if the
        iterative refinement residual exceeds
        :data:`_INVERSE_FAILURE_THRESHOLD`, or is not a number, after all
        iterations. The pole
        case carries the offending sample indices on ``.indices``.

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
        # Report which samples are degenerate: an array call that trips the
        # guard on a handful of samples is a caller-diagnosable geometry
        # problem, and the whole call still refuses (a partially-inverted
        # trajectory would be silently wrong at the bad samples).
        bad = np.flatnonzero(np.atleast_1d(near_pole))
        where = "" if near_pole.ndim == 0 else f" Offending sample indices: {bad.tolist()}."
        raise OffsetInversionError(
            "_offset_inverse cannot resolve azimuth at the pole: detector or "
            "boresight elevation is within "
            f"{_POLE_GUARD_DEG:g} deg of +/-90 deg, where azimuth is degenerate "
            "(every boresight azimuth maps to the same pole position, so the "
            "residual check cannot validate it). This requires an extreme offset "
            "placing the detector at the zenith and is far outside any realistic "
            f"PrimeCam pointing envelope.{where}",
            indices=bad.tolist() if near_pole.ndim else (),
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
        # Written so a NaN residual fails closed: one NaN sample makes the
        # maximum NaN, and ``nan > threshold`` is False, which would return
        # every other sample unchecked, an unconverged one included. The
        # element-wise maximum propagates a NaN from either axis.
        last_err = float(np.max(np.maximum(np.abs(d_az), np.abs(d_el))))
        if not last_err <= _INVERSE_FAILURE_THRESHOLD:
            raise OffsetInversionError(
                f"_offset_inverse iterative refinement failed to converge after "
                f"{_INVERSE_MAX_ITERATIONS} iterations "
                f"(last residual: {last_err:.2e} deg, worst residual: {worst_err:.2e} deg). "
                f"This may indicate an extreme offset or near-zenith elevation."
            )

    return bore_az, bore_el


def _rotate_offset(
    offset: InstrumentOffset,
    focal_plane_rotation: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Rotate offset by the focal-plane rotation angle.

    Parameters
    ----------
    offset : InstrumentOffset
        Detector offset from boresight.
    focal_plane_rotation : float or array
        Focal-plane rotation angle in degrees.

    Returns
    -------
    dx_rot : float or array
        Rotated cross-elevation offset in degrees.
    dy_rot : float or array
        Rotated elevation offset in degrees.
    """
    dx_deg = offset.dx_deg
    dy_deg = offset.dy_deg
    rot_rad = np.deg2rad(focal_plane_rotation)
    cos_rot = np.cos(rot_rad)
    sin_rot = np.sin(rot_rad)
    return dx_deg * cos_rot - dy_deg * sin_rot, dx_deg * sin_rot + dy_deg * cos_rot


def boresight_to_detector(
    az: float | np.ndarray,
    el: float | np.ndarray,
    offset: InstrumentOffset,
    focal_plane_rotation: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Compute detector Az/El given boresight Az/El and offset.

    Applies the instrument offset with the focal-plane rotation to compute
    the actual sky position that a detector is observing given where the
    telescope boresight is pointed. Uses spherical trigonometry
    (great-circle offset formulas) for accuracy at any offset size.

    For a Nasmyth-mounted instrument on an alt-az telescope, the focal
    plane rotates relative to the (az, el) axes as the elevation changes.
    The focal_plane_rotation parameter accounts for this rotation when
    computing detector positions.

    Parameters
    ----------
    az : float or array
        Boresight azimuth in degrees.
    el : float or array
        Boresight elevation in degrees.
    offset : InstrumentOffset
        Detector offset from boresight.
    focal_plane_rotation : float or array
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
    >>> det_az, det_el = boresight_to_detector(180.0, 45.0, offset, focal_plane_rotation=0.0)
    >>> print(f"Detector at Az={det_az:.3f}, El={det_el:.3f}")
    Detector at Az=180.118, El=45.000
    """
    dx_rot, dy_rot = _rotate_offset(offset, focal_plane_rotation)
    det_az, det_el = _offset_forward(az, el, dx_rot, dy_rot)

    if np.isscalar(az) and np.isscalar(el) and np.isscalar(focal_plane_rotation):
        return float(det_az), float(det_el)
    return det_az, det_el


def detector_to_boresight(
    det_az: float | np.ndarray,
    det_el: float | np.ndarray,
    offset: InstrumentOffset,
    focal_plane_rotation: float | np.ndarray,
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
    focal_plane_rotation : float or array
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
    OffsetInversionError
        If the detector or the resulting boresight elevation lies within
        1e-6 deg of the pole (+/-90 deg), where azimuth is degenerate, or
        if the iterative refinement fails to converge.
    ValueError
        If ``det_az``, ``det_el``, ``focal_plane_rotation`` or the offset's
        ``dx`` or ``dy`` holds a NaN or an infinity.

    Examples
    --------
    >>> offset = InstrumentOffset(dx=5.0, dy=0.0)
    >>> bore_az, bore_el = detector_to_boresight(180.0, 45.0, offset, focal_plane_rotation=0.0)
    >>> print(f"Boresight at Az={bore_az:.3f}, El={bore_el:.3f}")
    Boresight at Az=179.882, El=45.000

    Verify inverse relationship:

    >>> det_az2, det_el2 = boresight_to_detector(bore_az, bore_el, offset, focal_plane_rotation=0.0)
    >>> assert abs(det_az2 - 180.0) < 1e-6
    >>> assert abs(det_el2 - 45.0) < 1e-6
    """
    for name, value in (
        ("det_az", det_az),
        ("det_el", det_el),
        ("focal_plane_rotation", focal_plane_rotation),
        ("offset.dx", offset.dx),
        ("offset.dy", offset.dy),
    ):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"detector_to_boresight: {name} must be finite (no NaN or infinity)")

    dx_rot, dy_rot = _rotate_offset(offset, focal_plane_rotation)
    bore_az, bore_el = _offset_inverse(det_az, det_el, dx_rot, dy_rot)

    if np.isscalar(det_az) and np.isscalar(det_el) and np.isscalar(focal_plane_rotation):
        return float(bore_az), float(bore_el)
    return bore_az, bore_el


def sky_to_focal_plane(
    bore_az: float | np.ndarray,
    bore_el: float | np.ndarray,
    sky_az: float | np.ndarray,
    sky_el: float | np.ndarray,
    focal_plane_rotation: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    r"""Locate a sky position in the focal plane of a boresight pointing.

    The inverse projection of :func:`boresight_to_detector`: given where the
    boresight points and where a source is on the sky, return the focal-plane
    offset ``(xi, eta)`` at which that source appears, in degrees. Feeding the
    result back as an :class:`InstrumentOffset` reproduces ``(sky_az, sky_el)``
    from the boresight, after converting to the arcminutes that constructor
    takes (the example below multiplies by 60). All inputs broadcast, so a
    whole trajectory can be traced in one call.

    Parameters
    ----------
    bore_az, bore_el : float or array
        Boresight azimuth and elevation in degrees.
    sky_az, sky_el : float or array
        Azimuth and elevation of the sky position in degrees.
    focal_plane_rotation : float or array
        Orientation of the focal plane relative to the horizon axes in
        degrees, the same angle :func:`boresight_to_detector` takes
        (``nasmyth_sign * elevation + instrument_rotation`` plus any
        commanded rotator angle).

    Returns
    -------
    xi_deg : float or array
        Cross-elevation focal-plane coordinate in degrees (the
        ``InstrumentOffset.dx_deg`` axis).
    eta_deg : float or array
        Elevation focal-plane coordinate in degrees (the
        ``InstrumentOffset.dy_deg`` axis).

    Notes
    -----
    The great-circle separation ``rho`` and the position angle ``phi``
    of the sky position about the boresight (from the elevation direction
    toward increasing azimuth) come from the spherical triangle with the
    zenith:

    .. math::

        \sin\rho \sin\phi = \cos(El_1) \sin\Delta Az

        \sin\rho \cos\phi = \sin(El_1) \cos(El_0) - \cos(El_1) \sin(El_0) \cos\Delta Az

        \cos\rho = \sin(El_0) \sin(El_1) + \cos(El_0) \cos(El_1) \cos\Delta Az

    The horizon-frame offset ``(rho sin phi, rho cos phi)`` is then rotated
    back by ``focal_plane_rotation`` into the focal plane. At a boresight
    elevation of exactly 90 degrees the azimuth direction is undefined and
    ``phi`` is measured from an arbitrary meridian.

    Examples
    --------
    >>> offset = InstrumentOffset(dx=5.0, dy=3.0)
    >>> det_az, det_el = boresight_to_detector(180.0, 45.0, offset, focal_plane_rotation=30.0)
    >>> xi, eta = sky_to_focal_plane(180.0, 45.0, det_az, det_el, focal_plane_rotation=30.0)
    >>> print(f"{xi * 60:.6f} {eta * 60:.6f}")
    5.000000 3.000000
    """
    el0 = np.deg2rad(bore_el)
    el1 = np.deg2rad(sky_el)
    daz = np.deg2rad(np.asarray(sky_az, dtype=float) - np.asarray(bore_az, dtype=float))
    sin_el0, cos_el0 = np.sin(el0), np.cos(el0)
    sin_el1, cos_el1 = np.sin(el1), np.cos(el1)

    x = cos_el1 * np.sin(daz)
    y = sin_el1 * cos_el0 - cos_el1 * sin_el0 * np.cos(daz)
    cos_rho = sin_el0 * sin_el1 + cos_el0 * cos_el1 * np.cos(daz)
    rho = np.arctan2(np.hypot(x, y), cos_rho)
    phi = np.arctan2(x, y)

    dx_rot = np.rad2deg(rho * np.sin(phi))
    dy_rot = np.rad2deg(rho * np.cos(phi))

    rot = np.deg2rad(focal_plane_rotation)
    cos_rot, sin_rot = np.cos(rot), np.sin(rot)
    xi = dx_rot * cos_rot + dy_rot * sin_rot
    eta = -dx_rot * sin_rot + dy_rot * cos_rot

    scalar = all(np.isscalar(v) for v in (bore_az, bore_el, sky_az, sky_el, focal_plane_rotation))
    if scalar:
        return float(xi), float(eta)
    return xi, eta


def compute_focal_plane_rotation(
    el: float | np.ndarray,
    *,
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
        Telescope (boresight) elevation in degrees; the Nasmyth rotation
        follows the elevation axis.
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
    *,
    site: Site,
    validate: bool = False,
) -> Trajectory:
    """Apply detector offset to trajectory, accounting for focal-plane rotation.

    Returns a new trajectory with boresight positions adjusted so that the
    specified detector observes the original target positions.

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
        Telescope site configuration (needed for the focal-plane rotation).
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
    OffsetInversionError
        If the inversion hits the near-pole azimuth degeneracy or fails to
        converge (see :func:`detector_to_boresight`), or the boresight
        elevation the Nasmyth term is evaluated at does not settle.
    ValueError
        If the offset's ``dx`` or ``dy``, or the ``instrument_rotation`` of
        a nonzero offset, holds a NaN or an infinity.

    Notes
    -----
    **Precondition: the input trajectory must be in geometric (vacuum)
    coordinates.** This holds on every live path: ``Coordinates(site)``
    defaults to vacuum, and refraction is applied downstream at
    execution time (by exactly one of the Go TCS or the ACU). The mechanical Nasmyth term is
    solved from ``trajectory.el``; if the trajectory was instead built with
    ``AtmosphericConditions.for_fyst()`` (refracted, a planning/sim-only
    path), its ``el`` is in the apparent frame and the mechanical rotation
    differs from the vacuum one by ``nasmyth_sign * (refraction bump)``,
    a small boresight effect (a few arcseconds at worst, at the lowest
    elevations) at PrimeCam offset radii. ``Trajectory``
    carries no refraction flag, so a refracted input cannot be detected
    here; pair detector offsets with vacuum trajectories.

    The mechanical term ``nasmyth_sign * el`` is evaluated at the returned
    boresight elevation, the telescope's own elevation axis, which is what
    the Nasmyth rotation follows. That elevation is the unknown being
    solved for, so the rotation and the inversion are iterated together
    until it settles.

    The returned velocities are the input's plus the rate of change of the
    boresight correction, so a pattern's analytic velocities (the
    constant-elevation turnarounds, linear motion) survive the adjustment;
    only the correction, which varies slowly, is differentiated
    numerically. The input's velocity columns must therefore be the
    derivative of its positions, as they are in every trajectory the
    library builds.

    The returned trajectory shares some arrays with the input: ``metadata``,
    ``scan_flag`` and ``retune_events`` are the same objects, and a
    zero offset (no shift, whatever its instrument rotation, since a
    rotated zero vector is still zero) returns a copy that shares every
    array, since there is nothing to recompute. ``Trajectory``
    is frozen but its arrays are not read-only, so mutate one only when
    you mean to reach the other.

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
    >>> adjusted = apply_detector_offset(trajectory, offset, site=site)
    """
    if offset.dx == 0.0 and offset.dy == 0.0:
        return dataclasses.replace(trajectory)

    # Horizon-frame projection: the rotation is mechanical only; the
    # parallactic angle is a horizon-to-celestial quantity and has no
    # place in an az/el projection. The Nasmyth term follows the telescope's
    # own (boresight) elevation, the unknown being solved for, so iterate:
    # each pass shrinks the elevation error by about the offset radius in
    # radians (0.03 for the inner ring).
    bore_el = trajectory.el
    for _ in range(_INVERSE_MAX_ITERATIONS):
        rotation = compute_focal_plane_rotation(bore_el, site=site, offset=offset)
        bore_az, new_bore_el = detector_to_boresight(
            trajectory.az,
            trajectory.el,
            offset,
            rotation,
        )
        step = float(np.max(np.abs(new_bore_el - bore_el)))
        bore_el = new_bore_el
        if step < _INVERSE_EARLY_EXIT_THRESHOLD:
            break
    else:
        if not step <= _INVERSE_FAILURE_THRESHOLD:
            raise OffsetInversionError(
                f"apply_detector_offset: the boresight elevation did not settle after "
                f"{_INVERSE_MAX_ITERATIONS} iterations (last step {step:.2e} deg); the "
                "offset is far outside the PrimeCam envelope."
            )

    if len(trajectory.times) < 2:
        # np.gradient needs >=2 samples, so the correction's rate is unknown
        # for a single sample (the builder tolerates <2-point trajectories);
        # carry the input velocities through rather than fail.
        az_vel = trajectory.az_vel.copy()
        el_vel = trajectory.el_vel.copy()
    else:
        az_vel = trajectory.az_vel + np.gradient(bore_az - trajectory.az, trajectory.times)
        el_vel = trajectory.el_vel + np.gradient(bore_el - trajectory.el, trajectory.times)

    result = Trajectory(
        times=trajectory.times.copy(),
        az=bore_az,
        el=bore_el,
        az_vel=az_vel,
        el_vel=el_vel,
        start_time=trajectory.start_time,
        metadata=trajectory.metadata,
        scan_flag=trajectory.scan_flag,
        retune_events=trajectory.retune_events,
    )

    if validate:
        validate_trajectory_bounds(site, result.az, result.el)

    return result
