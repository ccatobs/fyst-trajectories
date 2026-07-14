"""Dataclasses and typed schemas for planning.

Contains:

* :class:`FieldRegion`, :class:`ArrayFootprint`, and :class:`ScanBlock` —
  the public data containers consumed and returned by the planner
  functions (``plan_pong_scan``, ``plan_constant_el_scan``,
  ``plan_daisy_scan``, ``plan_source_ces``).
* :class:`PongComputedParams`, :class:`PongAltAzComputedParams`,
  :class:`ConstantElComputedParams`, :class:`DaisyComputedParams`,
  :class:`DaisyAltAzComputedParams`, :class:`SourceCESComputedParams` —
  schemas that describe the shape of :attr:`ScanBlock.computed_params`
  returned by each planner.

The dataclasses and schemas are re-exported from
:mod:`fyst_trajectories.planning` and :mod:`fyst_trajectories`.
"""

import dataclasses
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

import numpy as np

from ..exceptions import PointingWarning
from ..patterns.configs import ScanConfig
from ..trajectory import Trajectory


class PongComputedParams(TypedDict):
    """Computed parameters returned by :func:`plan_pong_scan`.

    Attributes
    ----------
    period : float
        Pattern period in seconds for one full Pong cycle.
    x_numvert : int
        Number of vertices along the x-axis of the Lissajous lattice.
    y_numvert : int
        Number of vertices along the y-axis of the Lissajous lattice.
    n_cycles : int
        Number of full pattern cycles in the planned observation.
    """

    period: float
    x_numvert: int
    y_numvert: int
    n_cycles: int


class PongAltAzComputedParams(TypedDict):
    """Computed parameters returned by :func:`plan_pong_altaz_scan`.

    Mirrors :class:`PongComputedParams` and adds the fixed horizon-frame
    center the pattern was executed about.

    Attributes
    ----------
    period : float
        Pattern period in seconds for one full Pong cycle.
    x_numvert : int
        Number of vertices along the x-axis of the Lissajous lattice.
    y_numvert : int
        Number of vertices along the y-axis of the Lissajous lattice.
    n_cycles : int
        Number of full pattern cycles in the planned observation.
    az_center : float
        Azimuth of the fixed pattern center in degrees.
    el_center : float
        Elevation of the fixed pattern center in degrees.
    """

    period: float
    x_numvert: int
    y_numvert: int
    n_cycles: int
    az_center: float
    el_center: float


class ConstantElComputedParams(TypedDict):
    """Computed parameters returned by :func:`plan_constant_el_scan`.

    Attributes
    ----------
    az_start : float
        Lower azimuth bound of the scan in degrees.
    az_stop : float
        Upper azimuth bound of the scan in degrees.
    az_throw : float
        Total azimuth throw (``az_stop - az_start``) in degrees.
    n_scans : int
        Number of azimuth sweeps (legs) in the scan.
    start_time_iso : str
        ISO-format UTC start time of the observation.
    end_time_iso : str
        ISO-format UTC end time of the observation.
    duration : float
        Total observation duration in seconds.
    """

    az_start: float
    az_stop: float
    az_throw: float
    n_scans: int
    start_time_iso: str
    end_time_iso: str
    duration: float


class DaisyComputedParams(TypedDict):
    """Computed parameters returned by :func:`plan_daisy_scan`.

    Attributes
    ----------
    duration : float
        Observation duration in seconds.
    """

    duration: float


class DaisyAltAzComputedParams(TypedDict):
    """Computed parameters returned by :func:`plan_daisy_altaz_scan`.

    Mirrors :class:`DaisyComputedParams` and adds the fixed horizon-frame
    center the pattern was executed about.

    Attributes
    ----------
    duration : float
        Observation duration in seconds.
    az_center : float
        Azimuth of the fixed pattern center in degrees.
    el_center : float
        Elevation of the fixed pattern center in degrees.
    """

    duration: float
    az_center: float
    el_center: float


class SourceCESComputedParams(TypedDict):
    """Computed parameters returned by :func:`plan_source_ces`.

    Attributes
    ----------
    az_start : float
        Lower azimuth bound of the scan in degrees (after padding and
        ``az_branch`` re-wrapping).
    az_throw : float
        Total azimuth throw in degrees.
    v_az : float
        Solved (or user-supplied) azimuth drift rate in deg/s.
    el_bore : float
        Fixed boresight elevation in degrees.
    boresight_rot : float
        Mechanical boresight rotation in degrees (0.0 when not supplied).
    t0_iso : str
        ISO UTC time at which the source enters the footprint.
    t1_iso : str
        ISO UTC time at which the source exits the footprint.
    duration : float
        Actual trajectory duration in seconds (may differ slightly from
        ``t1 - t0`` because of leg/turnaround quantisation in the
        underlying ConstantEl pattern).
    mode : str
        Either ``"rising"`` or ``"setting"``.
    n_scans : int
        Number of azimuth sweeps (legs) in the scan.
    """

    az_start: float
    az_throw: float
    v_az: float
    el_bore: float
    boresight_rot: float
    t0_iso: str
    t1_iso: str
    duration: float
    mode: str
    n_scans: int


# Umbrella alias used by :attr:`ScanBlock.computed_params`. The concrete
# dict shape depends on which ``plan_*`` function produced the block.
ComputedParams = (
    PongComputedParams
    | PongAltAzComputedParams
    | ConstantElComputedParams
    | DaisyComputedParams
    | DaisyAltAzComputedParams
    | SourceCESComputedParams
)


@dataclass(frozen=True)
class FieldRegion:
    """Astronomer's specification of a rectangular field on the sky.

    Parameters
    ----------
    ra_center : float
        Right Ascension of the field center in degrees.
    dec_center : float
        Declination of the field center in degrees.
    width : float
        Angular width of the field in degrees (cross-scan direction).
        This is the physical angular extent, not the RA span. The
        cos(dec) projection is applied internally when computing
        RA boundaries. Must be positive.
    height : float
        Angular height of the field in degrees (Dec extent). Must be
        positive.

    Raises
    ------
    ValueError
        If width or height is not positive.

    Examples
    --------
    >>> field = FieldRegion(ra_center=180.0, dec_center=-30.0, width=2.0, height=2.0)
    """

    ra_center: float
    dec_center: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")

    @property
    def dec_min(self) -> float:
        """Minimum declination of the field in degrees."""
        return self.dec_center - self.height / 2.0

    @property
    def dec_max(self) -> float:
        """Maximum declination of the field in degrees."""
        return self.dec_center + self.height / 2.0


@dataclass(frozen=True)
class ArrayFootprint:
    """Explicit array footprint (focal-plane center + cover polygon).

    Used as input to :func:`plan_source_ces` to describe the on-sky
    extent the source must traverse. Coordinates are focal-plane
    degrees in the (xi, eta) convention where ``xi`` is the
    cross-elevation axis and ``eta`` is the elevation axis (matching
    :class:`~fyst_trajectories.InstrumentOffset` ``dx``/``dy`` axes).

    Mirrors the ``array_info`` dict consumed by Simons Observatory's
    ``schedlib.source.make_source_ces``, which the SO scheduler uses
    to project per-wafer geometries onto the sky.

    Parameters
    ----------
    center_xi_deg : float
        Cross-elevation coordinate of the footprint center, in degrees.
    center_eta_deg : float
        Elevation coordinate of the footprint center, in degrees.
    cover_xi_deg : np.ndarray
        Cross-elevation coordinates of polygon vertices, in degrees.
        Must be 1-D and have the same length as ``cover_eta_deg``.
    cover_eta_deg : np.ndarray
        Elevation coordinates of polygon vertices, in degrees.
        Must be 1-D and have the same length as ``cover_xi_deg``.

    Raises
    ------
    ValueError
        If the cover arrays are not 1-D, are different lengths, or
        are empty.

    Examples
    --------
    A 50-vertex circular footprint of radius 0.65 degrees centered on
    the focal-plane origin:

    >>> import numpy as np
    >>> theta = np.linspace(0.0, 2 * np.pi, 50, endpoint=False)
    >>> radius = 0.65
    >>> footprint = ArrayFootprint(
    ...     center_xi_deg=0.0,
    ...     center_eta_deg=0.0,
    ...     cover_xi_deg=radius * np.cos(theta),
    ...     cover_eta_deg=radius * np.sin(theta),
    ... )
    """

    center_xi_deg: float
    center_eta_deg: float
    cover_xi_deg: np.ndarray = field()
    cover_eta_deg: np.ndarray = field()

    def __post_init__(self) -> None:
        cover_xi = np.asarray(self.cover_xi_deg, dtype=float)
        cover_eta = np.asarray(self.cover_eta_deg, dtype=float)
        if cover_xi.ndim != 1 or cover_eta.ndim != 1:
            raise ValueError(
                f"cover_xi_deg and cover_eta_deg must be 1-D arrays, "
                f"got shapes {cover_xi.shape} and {cover_eta.shape}"
            )
        if cover_xi.shape != cover_eta.shape:
            raise ValueError(
                f"cover_xi_deg and cover_eta_deg must have the same length, "
                f"got {cover_xi.shape[0]} and {cover_eta.shape[0]}"
            )
        if cover_xi.size == 0:
            raise ValueError("ArrayFootprint cover polygon must have at least one vertex")
        # frozen=True precludes normal assignment; use object.__setattr__ to
        # canonicalise to float arrays (cheap and avoids defensive copies in
        # downstream code).
        object.__setattr__(self, "cover_xi_deg", cover_xi)
        object.__setattr__(self, "cover_eta_deg", cover_eta)

    @classmethod
    def from_array_info(
        cls,
        array_info: Mapping[str, Any],
        *,
        units: Literal["rad", "deg"] = "rad",
    ) -> "ArrayFootprint":
        """Build from SO-style ``{'center': (xi, eta), 'cover': (xi_arr, eta_arr)}``.

        Bridges the ``array_info`` dict schema consumed by Simons
        Observatory's ``schedlib.source.make_source_ces`` to
        fyst-trajectories' :class:`ArrayFootprint`. This is the
        recommended entry point for SO ``schedlib`` integrators: one
        call converts the dict and its radian-valued xi/eta arrays
        into the degree-valued ``ArrayFootprint`` that
        :func:`~fyst_trajectories.plan_source_ces` accepts.

        Parameters
        ----------
        array_info : mapping
            Dict with two entries: ``'center'`` as a length-2
            ``(xi, eta)`` sequence and ``'cover'`` as a length-2
            sequence whose first element is a 1-D array of vertex xi
            values and second element is the matching eta values.
        units : {'rad', 'deg'}, optional
            Angular units of the input. Default ``'rad'`` matches SO's
            convention; pass ``'deg'`` if your data is already in
            degrees.

        Returns
        -------
        ArrayFootprint
            Equivalent fyst-trajectories footprint, with all internal
            arrays in degrees.

        Examples
        --------
        Convert an SO ``array_info`` dict (radian xi/eta) to a footprint
        ready for :func:`~fyst_trajectories.plan_source_ces`:

        >>> import numpy as np
        >>> from fyst_trajectories.planning import ArrayFootprint
        >>> theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        >>> array_info = {
        ...     "center": (0.0, 0.0),
        ...     "cover": (0.01 * np.cos(theta), 0.01 * np.sin(theta)),
        ... }
        >>> fp = ArrayFootprint.from_array_info(array_info)  # radians by default
        """
        scale = float(np.rad2deg(1.0)) if units == "rad" else 1.0
        center = array_info["center"]
        cover = array_info["cover"]
        return cls(
            center_xi_deg=float(center[0]) * scale,
            center_eta_deg=float(center[1]) * scale,
            cover_xi_deg=np.asarray(cover[0], dtype=float) * scale,
            cover_eta_deg=np.asarray(cover[1], dtype=float) * scale,
        )


@dataclass(frozen=True)
class ScanBlock:
    """Complete observation specification produced by a planning function.

    Contains the generated trajectory, the pattern configuration used, and
    computed parameters that help the astronomer understand the observation.

    Parameters
    ----------
    trajectory : Trajectory
        The generated trajectory ready for telescope upload. Treat this
        as read-only after planning; downstream code should not mutate
        its arrays or metadata.
    config : ScanConfig
        The pattern configuration used to generate the trajectory.
    duration : float
        Observation duration in seconds.
    computed_params : ComputedParams
        A dict of computed parameters whose shape depends on the planner
        that produced the block: :class:`PongComputedParams`,
        :class:`PongAltAzComputedParams`,
        :class:`ConstantElComputedParams`, :class:`DaisyComputedParams`,
        :class:`DaisyAltAzComputedParams`, or
        :class:`SourceCESComputedParams`.
    summary : str
        Human-readable summary of the planned observation.

    Examples
    --------
    >>> block = plan_pong_scan(...)
    >>> print(block.summary)
    >>> print(f"Duration: {block.duration:.1f}s")
    >>> print(f"Points: {block.trajectory.n_points}")
    """

    trajectory: Trajectory
    config: ScanConfig
    duration: float
    # Runtime is a plain ``dict``; the TypedDict union is advisory for
    # static checkers. mypy can't match ``dict`` to any union member.
    computed_params: ComputedParams = dataclasses.field(default_factory=dict)  # type: ignore[assignment]
    summary: str = ""


# Expected keys per scan type, derived from each TypedDict's
# ``__required_keys__`` so the table cannot drift from the declared
# schemas (each TypedDict is ``total=True`` with no ``NotRequired``).
#
# NOTE: ``source_ces`` is intentionally NOT registered here.
# :class:`SourceCESComputedParams` exists as a static-type schema for
# :func:`~fyst_trajectories.plan_source_ces` returns, and the planner
# self-checks its return value directly against
# :attr:`SourceCESComputedParams.__required_keys__`, so this table is
# never consulted for source-CES. ``ObservingPatch`` still rejects
# ``"source_ces"`` as a science scan type. The
# :mod:`fyst_trajectories.overhead` simulator does now consume
# source-CES, but through calibration-block ``scan_params``
# (planet-cal passes emitted by ``CalibrationPolicy.planet_cal_scan``
# and rebuilt by ``schedule_to_trajectories(science_only=False)``),
# which validate against the overhead-side registry
# ``overhead/models.py:_SCAN_TYPE_TO_SCAN_PARAM_KEYS``
# (``SourceCESScanParams``) rather than this computed-params table.
# If a future use case wants source-CES *science* blocks in
# :func:`~fyst_trajectories.overhead.generate_timeline`, add the
# entry here AND wire it through
# ``overhead/simulation.py:_generate_trajectory_for_block`` AND
# ``overhead/models.py:ObservingPatch``.
_SCAN_TYPE_TO_KEYS: dict[str, frozenset[str]] = {
    "pong": PongComputedParams.__required_keys__,
    "pong_altaz": PongAltAzComputedParams.__required_keys__,
    "constant_el": ConstantElComputedParams.__required_keys__,
    "daisy": DaisyComputedParams.__required_keys__,
    "daisy_altaz": DaisyAltAzComputedParams.__required_keys__,
}


def validate_computed_params(params: Mapping[str, object], scan_type: str) -> None:
    """Validate the shape of a ``computed_params`` dict at runtime.

    Checks that the dict contains the keys expected for the given
    scan type. Missing required keys raise :class:`KeyError`;
    unexpected extra keys emit a
    :class:`~fyst_trajectories.exceptions.PointingWarning`.

    Parameters
    ----------
    params : mapping of str to object
        The candidate computed_params dict.
    scan_type : str
        One of ``"pong"``, ``"pong_altaz"``, ``"constant_el"``,
        ``"daisy"``, or ``"daisy_altaz"``.
        ``"source_ces"`` is intentionally NOT accepted — see the note
        on :data:`_SCAN_TYPE_TO_KEYS`. :func:`plan_source_ces`
        self-validates against
        :attr:`SourceCESComputedParams.__required_keys__` directly.

    Raises
    ------
    KeyError
        If ``scan_type`` is unknown or ``params`` is missing any key
        required by that scan type.
    """
    if scan_type not in _SCAN_TYPE_TO_KEYS:
        raise KeyError(
            f"Unknown scan_type {scan_type!r}; expected one of {sorted(_SCAN_TYPE_TO_KEYS)}"
        )
    expected = _SCAN_TYPE_TO_KEYS[scan_type]
    actual = set(params)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise KeyError(f"{scan_type} computed_params missing required keys: {sorted(missing)}")
    if extra:
        warnings.warn(
            f"{scan_type} computed_params has unexpected keys: {sorted(extra)}",
            PointingWarning,
            stacklevel=2,
        )
