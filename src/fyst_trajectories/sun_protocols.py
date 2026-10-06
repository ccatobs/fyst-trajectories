"""The contracts of the Sun-avoidance seam.

Every function that accepts an injected Sun model types it with one of
these protocols. A model implements one of the two base contracts:
:class:`SunSafePredicate` judges a point and :class:`SlewSafePredicate`
judges the direct slew to one. Three runtime-checkable extensions name the
optional methods a consumer uses when a model has them:
:class:`BatchSunSafePredicate` adds ``batch`` (verdicts for a whole grid in
one call), :class:`ZonedSunSafePredicate` adds ``threshold`` on top of it
(the Sun separation the zone requires) and :class:`PathSlewSafePredicate`
adds ``evaluate`` (the sampled slew path). The models
:func:`~fyst_trajectories.sun_models.make_sun_safe` builds are zoned, and
those :func:`~fyst_trajectories.sun_models.make_slew_safe` builds are path
predicates.

The library discovers an extension with ``isinstance``, which tests only
that the methods exist: any callable satisfies both base contracts whatever
its arity, and on Python 3.12 and later a model that exposes ``batch``
only through ``__getattr__`` delegation is not recognised as a batch model
and is consulted per sample.

The module imports nothing from the package, so every module can import
it.
"""

from typing import Protocol, runtime_checkable

import numpy as np
from astropy.time import Time


@runtime_checkable
class SunSafePredicate(Protocol):
    """The build-to contract for FYST's pluggable sun-avoidance model.

    This is **the** interface a sun-avoidance check must implement to be injected
    into :func:`~fyst_trajectories.dispatch.choose_encoder_solution` via its
    ``sun_safe`` parameter. It is a structural :class:`typing.Protocol`, so any
    callable with the matching signature satisfies it; no base class or
    registration is required.

    The default binding is the site's scalar exclusion radius (one isotropic
    radius, a position exactly at it counting as unsafe): the model
    ``make_sun_safe("scalar")`` builds, whose verdicts are those of
    :meth:`fyst_trajectories.coordinates.Coordinates.is_sun_safe`. The
    directional alternative, ``make_sun_safe("cad")``
    (:mod:`fyst_trajectories.sun_models`, FYST's own 50-90 deg
    direction-dependent zone from the shared sun-avoidance library), implements
    this same signature, so swapping models changes no call site.

    The query is instantaneous, a single ``(az, el, time)`` point. A caller may
    query it at several instants to cover a dwell window (see
    :func:`~fyst_trajectories.dispatch.choose_encoder_solution`'s array-valued
    ``obstime``); implementations stay single-instant. Dwell / exit-window ("how
    soon does the Sun enter this wrap") logic is *not* part of this contract; it
    belongs to the directional model's internal state, not its per-point verdict.

    **Optional extensions.** An implementation MAY additionally expose
    ``batch(az_deg, el_deg, times) -> numpy.ndarray[bool]`` (vectorized
    verdicts over broadcastable inputs, :class:`BatchSunSafePredicate`) and,
    on top of it, ``threshold`` (:class:`ZonedSunSafePredicate`). The
    library's grid consumers use ``batch`` when the model has it and fall
    back to per-sample scalar calls otherwise; the visibility
    renderers' ``sun_model`` parameter *requires* the zoned extension. The
    predicates built by :func:`~fyst_trajectories.sun_models.make_sun_safe`
    implement all of it.

    This contract judges a POINT. Whether the *slew path* to that point
    stays clear is the separate :class:`SlewSafePredicate` contract.
    """

    def __call__(self, az_deg: float, el_deg: float, time: Time) -> bool:
        """Return whether an encoder position is clear of the Sun.

        Parameters
        ----------
        az_deg : float
            Encoder azimuth in degrees (telescope range, not astropy
            ``[0, 360)`` sky range).
        el_deg : float
            Encoder elevation in degrees.
        time : Time
            Time at which to locate the Sun.

        Returns
        -------
        bool
            ``True`` when the position is clear of the Sun (safe to command),
            ``False`` when it is inside the avoidance zone.
        """
        ...


@runtime_checkable
class SlewSafePredicate(Protocol):
    """The path-level sibling of :class:`SunSafePredicate`.

    Judges whether the telescope's DIRECT slew from the current encoder
    position to a goal encoder position stays clear of the Sun for the
    whole motion, with the Sun advanced along the path. A correct
    single-point predicate is necessarily invariant under ``az -> az +
    360`` (same sky direction), so it can never *choose* an azimuth wrap;
    the paths to two wraps differ, which is exactly what this contract
    evaluates. Built by
    :func:`~fyst_trajectories.sun_models.make_slew_safe`, which sweeps a
    point model along the trapezoidal kinematic path using the FYST axis
    velocity/acceleration limits.

    Consumed by :func:`~fyst_trajectories.dispatch.choose_encoder_solution`'s
    optional ``slew_safe`` parameter to rank admissible wraps by path safety.
    When no direct path is clear the dispatch layer raises rather than
    auto-rerouting; a caller wanting a two-leg reroute plans it explicitly via
    :func:`~fyst_trajectories.sun_models.find_sun_safe_detour`, which needs
    the :class:`PathSlewSafePredicate` extension.
    """

    def __call__(
        self,
        current_az: float,
        current_el: float,
        goal_az: float,
        goal_el: float,
        time: Time,
    ) -> bool:
        """Return whether the direct slew is clear of the Sun throughout.

        Parameters
        ----------
        current_az, current_el : float
            Current encoder position in degrees.
        goal_az, goal_el : float
            Goal encoder position in degrees (``goal_az`` in the
            telescope encoder range; the path is the literal encoder
            travel, no wrapping).
        time : Time
            Slew start time (scalar); implementations advance the Sun
            along the path from here.

        Returns
        -------
        bool
            ``True`` when the whole path is clear of the Sun.
        """
        ...


@runtime_checkable
class BatchSunSafePredicate(SunSafePredicate, Protocol):
    """A :class:`SunSafePredicate` that also answers a whole grid in one call.

    A consumer holding many samples (a dwell grid, a trajectory, a horizon
    grid) asks ``batch`` once rather than calling the predicate per sample,
    so the Sun ephemeris is solved once for the grid.
    """

    def batch(
        self, az_deg: float | np.ndarray, el_deg: float | np.ndarray, times: Time
    ) -> np.ndarray:
        """Return the verdicts for a grid of positions and times.

        Parameters
        ----------
        az_deg, el_deg : float or np.ndarray
            Encoder azimuths and elevations in degrees, scalar or
            one-dimensional, broadcast against each other and ``times``.
        times : Time
            One time per sample, or a scalar time shared by every sample.

        Returns
        -------
        np.ndarray
            Boolean array, one verdict per sample, ``True`` where the
            position is clear of the Sun.
        """
        ...


@runtime_checkable
class ZonedSunSafePredicate(BatchSunSafePredicate, Protocol):
    """A :class:`BatchSunSafePredicate` that also states the separation it requires.

    ``threshold`` reports, per sample, the Sun separation the zone demands.
    It lets a consumer measure how far inside or outside the zone a pose
    is against the model's own requirement rather than against raw Sun
    separation, which matters for a directional zone whose requirement
    varies with the Sun's direction. Every model
    :func:`~fyst_trajectories.sun_models.make_sun_safe` builds is zoned.
    """

    def threshold(
        self, az_deg: float | np.ndarray, el_deg: float | np.ndarray, times: Time
    ) -> np.ndarray:
        """Return the minimum safe Sun separation in degrees for each sample.

        Parameters
        ----------
        az_deg, el_deg : float or np.ndarray
            Encoder azimuths and elevations in degrees, as for ``batch``.
        times : Time
            One time per sample, or a scalar time shared by every sample.

        Returns
        -------
        np.ndarray
            Float array, one required separation per sample.
        """
        ...


@runtime_checkable
class PathSlewSafePredicate(SlewSafePredicate, Protocol):
    """A :class:`SlewSafePredicate` that also returns the path it sampled.

    ``evaluate`` hands back the swept path with its verdict, so a caller can
    chain a second leg from the first leg's arrival time or read the Sun
    separations along the path. Every model
    :func:`~fyst_trajectories.sun_models.make_slew_safe` builds has it.
    """

    def evaluate(
        self,
        current_az: float,
        current_el: float,
        goal_az: float,
        goal_el: float,
        time: Time,
    ) -> tuple[bool, np.ndarray, np.ndarray, Time]:
        """Return the verdict with the sampled path.

        Parameters
        ----------
        current_az, current_el : float
            Current encoder position in degrees.
        goal_az, goal_el : float
            Goal encoder position in degrees.
        time : Time
            Slew start time (scalar).

        Returns
        -------
        tuple
            ``(safe, az_path, el_path, times)``: the path verdict, the
            sampled encoder azimuths and elevations in degrees and their
            times; ``times[-1]`` is the arrival.
        """
        ...


def _sun_verdicts(
    sun_safe: SunSafePredicate,
    az: float | np.ndarray,
    el: float | np.ndarray,
    times: Time,
    *,
    what: str = "grid",
) -> np.ndarray:
    """Return one Sun verdict per sample, through ``batch`` when the model has it.

    A :class:`BatchSunSafePredicate` is asked once for the whole grid and
    its answer must hold one verdict per sample: a short or scalar answer
    would broadcast one verdict over the grid, so it raises instead. Any
    other predicate is called once per sample.

    Parameters
    ----------
    sun_safe : SunSafePredicate
        The point model to consult.
    az, el : float or np.ndarray
        Encoder azimuths and elevations in degrees, one-dimensional and of
        one length (a scalar is one sample).
    times : Time
        One time per sample, or a scalar time shared by every sample.
    what : str, optional
        What the grid is, named in the shape error. Default ``"grid"``.

    Returns
    -------
    np.ndarray
        Boolean array with one verdict per sample (shape ``(1,)`` for a
        scalar position), ``True`` where the sample is clear of the Sun.

    Raises
    ------
    ValueError
        If ``batch`` returns a result of the wrong shape.
    """
    az = np.atleast_1d(np.asarray(az, dtype=float))
    el = np.atleast_1d(np.asarray(el, dtype=float))
    if isinstance(sun_safe, BatchSunSafePredicate):
        verdicts = np.atleast_1d(np.asarray(sun_safe.batch(az, el, times), dtype=bool))
        if verdicts.shape != az.shape:
            raise ValueError(
                f"sun_safe.batch returned shape {verdicts.shape}, expected {az.shape} "
                f"verdicts for the {what}"
            )
        return verdicts
    return np.array(
        [
            bool(sun_safe(float(az[i]), float(el[i]), times if times.isscalar else times[i]))
            for i in range(az.size)
        ],
        dtype=bool,
    )
