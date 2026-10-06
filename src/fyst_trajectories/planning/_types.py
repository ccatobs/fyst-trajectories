"""Dataclasses and typed schemas for planning.

Contains:

* :class:`FieldRegion`, :class:`ArrayFootprint`, and :class:`ScanBlock`:
  the public data containers consumed and returned by the planner
  functions (``plan_pong_scan``, ``plan_constant_el_scan``,
  ``plan_daisy_scan``, ``plan_source_ces``).
* :class:`PongComputedParams`, :class:`PongAltAzComputedParams`,
  :class:`ConstantElComputedParams`, :class:`DaisyComputedParams`,
  :class:`DaisyAltAzComputedParams`, :class:`SourceCESComputedParams`:
  schemas that describe the shape of :attr:`ScanBlock.computed_params`
  returned by each planner.

The dataclasses and schemas are re-exported from
:mod:`fyst_trajectories.planning` and :mod:`fyst_trajectories`.
"""

import dataclasses
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypedDict, TypeVar

import numpy as np

from .._validation import _require_positive
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
        Number of vertices along the x-axis of the Pong vertex lattice.
    y_numvert : int
        Number of vertices along the y-axis of the Pong vertex lattice.
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
        Number of vertices along the x-axis of the Pong vertex lattice.
    y_numvert : int
        Number of vertices along the y-axis of the Pong vertex lattice.
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
        ISO-format UTC time at which the solved pass window closes (the
        elevation crossing, or the LSA window when one was given).
    duration : float
        Trajectory duration in seconds: that window rounded to a whole number
        of azimuth legs, so it differs from ``end_time_iso - start_time_iso``
        in either direction by up to half of one leg-plus-turnaround, and by
        more when the window is shorter than the single leg that is always
        built. Book a slot on this value.
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
        Total azimuth throw in degrees (the solved footprint crossing plus
        padding, or the explicit ``az_throw`` when one was given).
    az_speed : float
        Per-leg azimuth speed in deg/s the sweep was planned with: the
        explicit ``az_speed`` when one was given, otherwise the derived
        slow-drag speed. A mount-frame azimuth coordinate rate.
    v_az : float
        Solved (or user-supplied) azimuth drift rate in deg/s. A
        mount-frame azimuth *coordinate* rate, not an on-sky speed.
    el_bore : float
        Fixed boresight elevation in degrees.
    boresight_rot : float
        Mechanical boresight rotation in degrees (0.0 when not supplied).
    t0_iso : str
        ISO UTC time at which the scanned window opens: when the source
        enters the footprint, or later by half the cut when a ``dwell``
        narrowed the pass.
    t1_iso : str
        ISO UTC time at which the scanned window closes: when the source
        exits the footprint, or earlier by half the cut when a ``dwell``
        narrowed the pass.
    duration : float
        Actual trajectory duration in seconds. Leg/turnaround
        quantisation in the underlying ConstantEl pattern rounds the
        window to whole legs, so this differs from ``t1 - t0`` by up to
        half a leg plus turnaround in either direction: a few seconds at
        a fast drag, about 26 s on the slow-drag default. Book a slot on
        this value, not on the window.
    crossing_seconds : float
        Duration in seconds of the full footprint crossing, before any
        ``dwell`` narrowing. Equals ``t1 - t0`` unless a ``dwell`` was
        given.
    mode : {"rising", "setting"}
        Direction of the source arc the pass was planned on.
    n_scans : int
        Number of azimuth sweeps (legs) in the scan.
    """

    az_start: float
    az_throw: float
    az_speed: float
    v_az: float
    el_bore: float
    boresight_rot: float
    t0_iso: str
    t1_iso: str
    duration: float
    crossing_seconds: float
    mode: Literal["rising", "setting"]
    n_scans: int


# Umbrella alias used by :attr:`ScanBlock.computed_params`. The concrete
# dict shape depends on which ``plan_*`` function produced the block.
#
# Deliberately INCLUDES ``SourceCESComputedParams``: the planner returns it.
# Do not equalize with the overhead-side scan-type vocabularies.
ComputedParams = (
    PongComputedParams
    | PongAltAzComputedParams
    | ConstantElComputedParams
    | DaisyComputedParams
    | DaisyAltAzComputedParams
    | SourceCESComputedParams
)

# The type parameter of :class:`ScanBlock`: the computed-parameter schema of
# the planner that produced the block. Covariant because a block is frozen, so
# a block of any one planner is also a ``ScanBlock[ComputedParams]``.
_ParamsT_co = TypeVar("_ParamsT_co", bound=ComputedParams, covariant=True)


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
        _require_positive(self.width, "width")
        _require_positive(self.height, "height")

    @property
    def dec_min(self) -> float:
        """Minimum declination of the field in degrees."""
        return self.dec_center - self.height / 2.0

    @property
    def dec_max(self) -> float:
        """Maximum declination of the field in degrees."""
        return self.dec_center + self.height / 2.0


@dataclass(frozen=True, eq=False)
class ArrayFootprint:
    """Explicit array footprint (focal-plane center + cover polygon).

    Used as input to :func:`plan_source_ces` to describe the on-sky
    extent the source must traverse. Coordinates are focal-plane
    degrees in the (xi, eta) convention where ``xi`` is the
    cross-elevation axis and ``eta`` is the elevation axis (matching
    :class:`~fyst_trajectories.offsets.InstrumentOffset` ``dx``/``dy`` axes).

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

    Notes
    -----
    The cover arrays are copied at construction and stored read-only, so
    editing the arrays passed in leaves the footprint unchanged and an
    in-place write to ``cover_xi_deg`` or ``cover_eta_deg`` raises
    ``ValueError``; copy an array with ``np.array(footprint.cover_xi_deg)``
    before editing it. Pickling and copying rebuild the footprint through
    its constructor, so the copies are read-only too.

    Footprints compare and hash by identity: ``==`` is ``True`` only for
    the same object, so a :func:`dataclasses.replace` copy is unequal to its
    original, and a footprint can be a set member or a dictionary key.
    Compare two footprints field by field with :func:`numpy.array_equal`.

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
    cover_xi_deg: np.ndarray
    cover_eta_deg: np.ndarray

    def __post_init__(self) -> None:
        cover_xi = np.array(self.cover_xi_deg, dtype=float)
        cover_eta = np.array(self.cover_eta_deg, dtype=float)
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
        # frozen=True precludes normal assignment, so object.__setattr__
        # stores the float-canonicalised arrays. np.array always copies, so
        # the footprint owns its arrays and can make them read-only.
        cover_xi.setflags(write=False)
        cover_eta.setflags(write=False)
        object.__setattr__(self, "cover_xi_deg", cover_xi)
        object.__setattr__(self, "cover_eta_deg", cover_eta)

    def __reduce__(self) -> tuple[type["ArrayFootprint"], tuple[Any, ...]]:
        # NumPy returns writeable arrays from pickle, deepcopy and copy, so
        # rebuild through the constructor, which copies and freezes them again.
        return (type(self), tuple(getattr(self, f.name) for f in dataclasses.fields(self)))

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
        :func:`~fyst_trajectories.planning.plan_source_ces` accepts.

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
            degrees. Any other value is rejected.

        Returns
        -------
        ArrayFootprint
            Equivalent fyst-trajectories footprint, with all internal
            arrays in degrees.

        Raises
        ------
        ValueError
            If ``units`` is neither ``'rad'`` nor ``'deg'``.

        Examples
        --------
        Convert an SO ``array_info`` dict (radian xi/eta) to a footprint
        ready for :func:`~fyst_trajectories.planning.plan_source_ces`:

        >>> import numpy as np
        >>> from fyst_trajectories.planning import ArrayFootprint
        >>> theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        >>> array_info = {
        ...     "center": (0.0, 0.0),
        ...     "cover": (0.01 * np.cos(theta), 0.01 * np.sin(theta)),
        ... }
        >>> fp = ArrayFootprint.from_array_info(array_info)  # radians by default
        """
        if units not in ("rad", "deg"):
            raise ValueError(f"units must be 'rad' or 'deg', got {units!r}")
        scale = float(np.rad2deg(1.0)) if units == "rad" else 1.0
        center = array_info["center"]
        cover = array_info["cover"]
        return cls(
            center_xi_deg=float(center[0]) * scale,
            center_eta_deg=float(center[1]) * scale,
            cover_xi_deg=np.asarray(cover[0], dtype=float) * scale,
            cover_eta_deg=np.asarray(cover[1], dtype=float) * scale,
        )


@dataclass(frozen=True, eq=False)
class ScanBlock(Generic[_ParamsT_co]):
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
        :class:`SourceCESComputedParams`. The block's type parameter is
        that schema, so ``plan_pong_scan`` returns a
        ``ScanBlock[PongComputedParams]``.
    summary : str
        Human-readable summary of the planned observation.

    Notes
    -----
    Blocks compare and hash by identity: ``==`` is ``True`` only for the
    same object, so a :func:`dataclasses.replace` copy is unequal to its
    original even though it shares the trajectory, and a block can be a
    set member or a dictionary key.

    Examples
    --------
    >>> block = plan_pong_scan(...)  # doctest: +SKIP
    >>> print(block.summary)  # doctest: +SKIP
    >>> print(f"Duration: {block.duration:.1f}s")  # doctest: +SKIP
    >>> print(f"Points: {block.trajectory.n_points}")  # doctest: +SKIP
    """

    trajectory: Trajectory
    config: ScanConfig
    duration: float
    # Runtime is a plain ``dict``; the schema type is advisory for static
    # checkers, which cannot match the ``dict`` default to it.
    computed_params: _ParamsT_co = dataclasses.field(default_factory=dict)  # type: ignore[assignment]
    summary: str = ""


# Expected keys per scan type, derived from each TypedDict's
# ``__required_keys__`` so the table cannot drift from the declared
# schemas (each TypedDict is ``total=True`` with no ``NotRequired``).
#
# Deliberately EXCLUDES ``source_ces``: the planner self-checks its return
# against :attr:`SourceCESComputedParams.__required_keys__`, so this table is
# never consulted for it, and the simulator reaches source-CES only through
# calibration-block ``scan_params``, validated by the overhead-side
# ``_SCAN_TYPE_TO_SCAN_PARAM_KEYS``. Do not equalize the two.
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
        ``"source_ces"`` is intentionally not accepted:
        :func:`plan_source_ces` self-validates against
        ``SourceCESComputedParams.__required_keys__`` directly.

    Raises
    ------
    KeyError
        If ``scan_type`` is unknown or ``params`` is missing any key
        required by that scan type.

    Warns
    -----
    PointingWarning
        If ``params`` carries keys the scan type does not declare.
    """
    if scan_type == "source_ces":
        raise KeyError(
            "source_ces computed_params are not validated by "
            "validate_computed_params; plan_source_ces self-checks its return "
            "against SourceCESComputedParams.__required_keys__, and overhead "
            "source-CES scan_params validate via "
            "fyst_trajectories.overhead.validate_scan_params. This validator "
            f"accepts only {sorted(_SCAN_TYPE_TO_KEYS)}."
        )
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
