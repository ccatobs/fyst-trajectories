"""Selectable sun-avoidance models, including the shared ``sun-avoidance`` library.

This module is the single place where a caller chooses *which* sun-avoidance
model drives the ``sun_safe`` seam (:class:`~fyst_trajectories.dispatch.SunSafePredicate`)
threaded through :func:`~fyst_trajectories.observability.check_observability`,
the planners, and :func:`~fyst_trajectories.dispatch.choose_encoder_solution`:

- ``"scalar"``: the library's own isotropic exclusion radius, read from
  ``site.sun_avoidance`` (today's default behaviour, no extra dependency).
- ``"cone"``: the shared `ccatobs/sun-avoidance
  <https://github.com/ccatobs/sun-avoidance>`_ library's cone mode at a
  caller-chosen radius.
- ``"cad"``: the same library's full directional CAD-derived zone
  (``sa_safe_CAD_20231030.csv``, minimum-separation 50-90 deg depending on
  the Sun's clock angle). The site's scalar radii are today's default
  policy; FYST's directional CAD zone is opt-in via
  ``make_sun_safe("cad")`` and is expected to become the default in a
  future release. The scalar also remains the commissioning fallback.

The shared library is an *optional* dependency, imported lazily and pinned:
:data:`SUN_AVOIDANCE_PINNED_SHA` records the revision the parity fixtures
were generated against, and :data:`CAD_TABLE_SHA256` guards the CAD table
bytes. ``"scalar"`` never needs it.

Deviations from the shared library's own defaults (deliberate, documented):

1. ``min_solar_altitude`` defaults to ``-90.0`` (the constraint always
   applies), not the library's ``0.0`` (which exempts the whole sky the
   moment the Sun's geometric altitude is negative, while refraction keeps
   the Sun visible to about -0.83 deg). Pass ``min_solar_altitude=0.0`` to
   reproduce the library's behaviour.
2. The "forbidden island" check defaults OFF (``island_check=False``). It
   answers a reachability question ("could this position become isolated by
   the moving zone"), not a point-safety one, and belongs to the path-level
   seam (enable it there via :func:`make_slew_safe`'s ``island_check``);
   enabling it here makes the point verdict conservative for
   high-elevation positions on the Sun's side of the sky.
3. Elevations at or within 0.001 deg of the zenith are evaluated at
   89.999 deg. At exactly 90 the library's clock-angle branch does not fire
   (the threshold silently collapses to the table floor), and in the final
   float ULPs below 90 its bearing computation is numerically degenerate
   (the clock angle quantises to multiples of 90), so a working margin is
   required, not an infinitesimal nudge. The 0.001 deg position change is
   4-5 orders of magnitude below the 50-90 deg threshold scale.
4. Sun positions come from this library's own vacuum ephemeris
   (:meth:`~fyst_trajectories.coordinates.Coordinates.get_sun_altaz`),
   matching the library-wide vacuum convention, not from the shared
   library's refraction-forced ``FYST`` observer.

Only the shared library's *point geometry* is bound here. Its high-level
checks (``check_fixed_target``'s silent over-the-top substitution), slew
kinematics, and detour search are deliberately not used. Path-level slew
safety is built here instead: :func:`make_slew_safe` sweeps a point model
along the FYST trapezoidal kinematic path (the FYST axis limits, the Sun
ephemeris advanced along the motion) to satisfy the
:class:`~fyst_trajectories.dispatch.SlewSafePredicate` contract, and
:func:`find_sun_safe_detour` plans elevation-bounded two-leg reroutes for
the caller (dispatch itself rejects rather than auto-rerouting, by
design).

Examples
--------
>>> from astropy.time import Time
>>> from fyst_trajectories import check_observability
>>> from fyst_trajectories.sun_models import make_sun_safe
>>> sun_safe = make_sun_safe("cad")
>>> reports = check_observability(
...     ["jupiter", "uranus"],
...     Time("2026-11-15T16:00:00", scale="utc"),
...     horizon_hours=24.0,
...     sun_safe=sun_safe,
... )
"""

from __future__ import annotations

import hashlib
import math
import pathlib
import warnings
from typing import TYPE_CHECKING

import numpy as np
from astropy.time import Time, TimeDelta

from .coordinates import Coordinates
from .dispatch import SlewSafePredicate, SunSafePredicate
from .exceptions import PointingWarning
from .site import (
    FYST_AZ_MAX_ACCELERATION,
    FYST_AZ_MAX_VELOCITY,
    FYST_EL_MAX_ACCELERATION,
    FYST_EL_MAX_VELOCITY,
    Site,
    get_fyst_site,
)

if TYPE_CHECKING:
    from sun_avoidance import AvoidanceData

__all__ = [
    "CAD_TABLE_SHA256",
    "SUN_AVOIDANCE_PINNED_SHA",
    "find_sun_safe_detour",
    "load_avoidance_data",
    "make_slew_safe",
    "make_sun_safe",
]

SUN_AVOIDANCE_PINNED_SHA = "e6fa12aa53ce5f5f76d50f8b753e7fe4b4ad8e18"
"""ccatobs/sun-avoidance revision the adapter and parity fixtures are built against."""

CAD_TABLE_SHA256 = "ecdf86e2d9d091c01cf1d9d67490e46edeb77af5bafd08c0ab1e7bc18a4d4a46"
"""SHA-256 of ``sa_safe_CAD_20231030.csv`` at the pinned revision.

The loader refuses a table whose bytes differ: a revised CAD model must
arrive as a deliberate re-pin (new SHA, regenerated parity fixtures), never
as a silent behaviour change.
"""

#: Module names the pinned library maps to a nonzero boresight-to-module
#: padding; anything else silently resolves to zero padding upstream.
_KNOWN_PADDED_MODULES = ("primecam_f280", "primecam_f350", "primecam_eorspec")

#: Elevations above this are evaluated here (deviation 3): the library's
#: clock-angle bearing is numerically degenerate in the final ULPs below 90
#: and its branch logic skips exactly 90, so a real working margin is needed.
_EL_ZENITH_CLAMP = 89.999

#: Sun-position cache entries kept per predicate before the cache resets.
_SUN_CACHE_MAX = 2048


def _import_sun_avoidance():
    """Import the optional shared library, with an actionable error when absent."""
    try:
        import sun_avoidance
    except ImportError:
        raise RuntimeError(
            "the shared sun-avoidance library is required for the 'cone' and 'cad' "
            "sun models. Install the pinned revision:\n"
            "  pip install git+https://github.com/ccatobs/sun-avoidance@"
            f"{SUN_AVOIDANCE_PINNED_SHA}"
        ) from None
    return sun_avoidance


def load_avoidance_data(model: str = "cad", *, radius: float | None = None) -> AvoidanceData:
    """Load the shared library's avoidance-zone table for a named model.

    Parameters
    ----------
    model : str, optional
        ``"cad"`` (default): the directional CAD-derived table shipped with
        the library, integrity-checked against :data:`CAD_TABLE_SHA256`.
        ``"cone"``: a circularly symmetric cone; requires ``radius``.
    radius : float, optional
        Cone radius in degrees (``model="cone"`` only; must be finite and
        positive).

    Returns
    -------
    sun_avoidance.AvoidanceData
        The loaded zone.

    Raises
    ------
    RuntimeError
        If the shared library is not installed, or the CAD table's bytes do
        not match the pinned SHA-256.
    ValueError
        On an unknown ``model``, a missing/invalid cone ``radius``, or a
        ``radius`` given with ``model="cad"``.
    """
    # Argument validation precedes the optional import so a bad call raises the
    # same ValueError whether or not the shared library is installed.
    if model not in ("cad", "cone"):
        raise ValueError(f"Unknown avoidance model {model!r}; expected 'cad' or 'cone'.")
    if model == "cad" and radius is not None:
        raise ValueError("radius applies to model='cone' only; the CAD zone is fixed.")
    if model == "cone" and (radius is None or not np.isfinite(radius) or not 0 < radius <= 180.0):
        raise ValueError(f"model='cone' requires a finite radius in (0, 180], got {radius}")

    lib = _import_sun_avoidance()

    if model == "cad":
        table = pathlib.Path(lib.__file__).parent / "data" / "sa_safe_CAD_20231030.csv"
        digest = hashlib.sha256(table.read_bytes()).hexdigest()
        if digest != CAD_TABLE_SHA256:
            raise RuntimeError(
                f"CAD table {table} has SHA-256 {digest}, expected {CAD_TABLE_SHA256} "
                f"(pinned at sun-avoidance {SUN_AVOIDANCE_PINNED_SHA}). The installed "
                "library revision differs from the pin; re-pin deliberately (update "
                "SUN_AVOIDANCE_PINNED_SHA / CAD_TABLE_SHA256 and regenerate the parity "
                "fixtures) rather than mixing revisions."
            )
        data = lib.AvoidanceData()
        data.loadFromFile(table)
        return data

    data = lib.AvoidanceData()
    data.calculateFromCone(float(radius))
    return data


class _BaseSunModel:
    """Shared predicate machinery for every sun model.

    Provides the scalar-call contract, the input guards (rank, elevation
    range), a per-predicate Sun-position cache for scalar times, and the
    disabled-site passthrough shape helper.
    """

    def __init__(self, site: Site):
        self._site = site
        self._coords = Coordinates(site)
        self._enabled = site.sun_avoidance.enabled
        self._sun_cache: dict[float, tuple[float, float]] = {}

    def __call__(self, az_deg: float, el_deg: float, time: Time) -> bool:
        """Scalar :class:`~fyst_trajectories.dispatch.SunSafePredicate` verdict."""
        verdict = self.batch(az_deg, el_deg, time)
        if verdict.size != 1:
            raise ValueError(
                "the scalar predicate call takes one (az, el, time) point; use batch() for arrays."
            )
        return bool(verdict[0])

    def _broadcast_with_sun(
        self, az_deg, el_deg, times: Time
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Validate and broadcast (az, el) with the Sun's position over ``times``.

        Supports the three call shapes the seam consumers use: same-length
        trajectory arrays with a matching time grid, a fixed position over a
        time grid, and a position array at a single instant. Higher-rank
        inputs are rejected (numpy broadcasting would silently outer-product
        them into wrong pairings) and mismatched 1-D lengths raise. Sun
        positions for scalar times are cached per predicate: dispatch and
        planning loops revisit the same instants across wraps and field
        corners, and the ephemeris call dominates the per-point cost.
        """
        az, el = self._validated_arrays(az_deg, el_deg)
        if times.isscalar:
            # Scale-invariant key: two Times in different scales can share the
            # same numeric jd while naming instants ~37 s apart (utc vs tai);
            # keying on the TAI jd both de-collides them and merges one
            # instant expressed in two scales into a single entry.
            key = float(times.tai.jd)
            cached = self._sun_cache.get(key)
            if cached is None:
                cached = self._coords.get_sun_altaz(times)
                if len(self._sun_cache) >= _SUN_CACHE_MAX:
                    self._sun_cache.clear()
                self._sun_cache[key] = cached
            sun_az, sun_el = cached
        else:
            sun_az, sun_el = self._coords.get_sun_altaz(times)
        sun_az = np.atleast_1d(np.asarray(sun_az, dtype=float))
        sun_el = np.atleast_1d(np.asarray(sun_el, dtype=float))
        az, el, sun_az, sun_el = np.broadcast_arrays(az, el, sun_az, sun_el)
        # Real copies: the pinned library does not mutate its inputs today,
        # but it indexes and reshapes them; cheap insurance against a future
        # revision writing into a 0-stride broadcast view.
        return az.copy(), el.copy(), sun_az.copy(), sun_el.copy()

    @staticmethod
    def _validated_arrays(az_deg, el_deg) -> tuple[np.ndarray, np.ndarray]:
        """Apply the rank and elevation-range guards on every entry path."""
        if np.ndim(az_deg) > 1 or np.ndim(el_deg) > 1:
            raise ValueError(
                "az_deg/el_deg must be scalar or 1-D (the SunSafePredicate call "
                f"shapes); got shapes {np.shape(az_deg)} and {np.shape(el_deg)}."
            )
        az = np.atleast_1d(np.asarray(az_deg, dtype=float))
        el = np.atleast_1d(np.asarray(el_deg, dtype=float))
        if el.size and (np.any(el > 90.0) or np.any(el < -90.0)):
            raise ValueError(
                f"el_deg must lie within [-90, 90], got values in [{el.min()}, {el.max()}]; "
                "over-the-top encoder elevations are not part of the point-predicate contract."
            )
        return az, el

    def _result_shape(self, az_deg, el_deg, times: Time) -> tuple[int, ...]:
        """Broadcast result shape without computing the Sun (disabled path).

        Runs the same input guards as the compute path, so a disabled site
        rejects exactly what an enabled one does.
        """
        az, el = self._validated_arrays(az_deg, el_deg)
        time_shape = () if times.isscalar else (times.size,)
        return np.broadcast_shapes(az.shape, el.shape, time_shape)


class _ScalarSunModel(_BaseSunModel):
    """The site's isotropic exclusion radius as a batch-capable predicate.

    Reproduces :meth:`~fyst_trajectories.coordinates.Coordinates.is_sun_safe`
    exactly (strict ``>``, disabled-site passthrough), adding the vectorized
    ``batch``/``threshold`` surface shared by every model from
    :func:`make_sun_safe`.
    """

    model = "scalar"

    def __init__(self, site: Site):
        super().__init__(site)
        self.describe = (
            f"scalar {site.sun_avoidance.exclusion_radius:g}°"
            if self._enabled
            else "scalar (disabled)"
        )

    def batch(self, az_deg, el_deg, times: Time) -> np.ndarray:
        """Vectorized verdicts; True = clear of the Sun."""
        if not self._enabled:
            return np.ones(self._result_shape(az_deg, el_deg, times), dtype=bool)
        az, el, sun_az, sun_el = self._broadcast_with_sun(az_deg, el_deg, times)
        sep = np.atleast_1d(self._coords.angular_separation(az, el, sun_az, sun_el))
        return sep > self._site.sun_avoidance.exclusion_radius

    def threshold(self, az_deg, el_deg, times: Time) -> np.ndarray:
        """Minimum safe Sun separation (deg) per sample; constant for this model."""
        shape = self._result_shape(az_deg, el_deg, times)
        if not self._enabled:
            return np.zeros(shape, dtype=float)
        return np.full(shape, self._site.sun_avoidance.exclusion_radius, dtype=float)


class _LibrarySunModel(_BaseSunModel):
    """The shared library's zone geometry as a batch-capable predicate."""

    def __init__(
        self,
        model: str,
        data: AvoidanceData,
        site: Site,
        *,
        min_solar_altitude: float,
        island_check: bool,
        maxoffset: float,
        tracking_module: str,
    ):
        super().__init__(site)
        # Hoisted so a predicate keeps working (and fails at construction,
        # with the actionable message) regardless of later import-state games.
        lib = _import_sun_avoidance()
        self._calc_sun_distance = lib.calc_sun_distance
        self._get_mask_fixed_pos = lib.get_mask_fixed_pos
        self._get_module_offset = lib.get_distance_tracking_module
        self.model = model
        self._data = data
        self._min_solar_altitude = min_solar_altitude
        self._island_check = island_check
        self._maxoffset = maxoffset
        self._tracking_module = tracking_module
        zone = (
            f"cone {data.deltaMin:g}°"
            if model == "cone"
            else f"CAD zone {data.deltaMin:g}-{data.deltaMax:g}°"
        )
        self.describe = zone if self._enabled else f"{zone} (disabled)"

    def batch(self, az_deg, el_deg, times: Time) -> np.ndarray:
        """Vectorized verdicts; True = clear of the Sun."""
        if not self._enabled:
            return np.ones(self._result_shape(az_deg, el_deg, times), dtype=bool)
        mask, _ = self._evaluate(az_deg, el_deg, times)
        return mask

    def threshold(self, az_deg, el_deg, times: Time) -> np.ndarray:
        """Minimum safe geometric Sun separation (deg) per sample.

        The library's directional table threshold at each sample's clock
        angle, plus the ``maxoffset`` / ``tracking_module`` paddings, so the
        value is directly comparable to the geometric boresight-Sun
        separation (as plotted by ``plot_visibility``'s separation panel).

        Reflects the zone geometry only: the ``min_solar_altitude`` waiver
        and the island check change verdicts without changing this value,
        so ``verdict == (separation > threshold)`` holds exactly only under
        the default never-waive, island-off configuration (zeros when the
        site has Sun avoidance disabled).
        """
        if not self._enabled:
            return np.zeros(self._result_shape(az_deg, el_deg, times), dtype=float)
        _, threshold = self._evaluate(az_deg, el_deg, times)
        return threshold

    def _evaluate(self, az_deg, el_deg, times: Time) -> tuple[np.ndarray, np.ndarray]:
        """Shared verdict + geometric-units threshold computation."""
        az, el, sun_az, sun_el = self._broadcast_with_sun(az_deg, el_deg, times)
        # Deviation 3: evaluate near-zenith elevations at 89.999. At exactly
        # 90 the library's clock-angle branch does not fire (threshold
        # collapses to the table floor), and in the final float ULPs below 90
        # its bearing computation is numerically degenerate (gamma quantises
        # to multiples of 90), so a real working margin is required. The
        # 0.001 deg position change is negligible against 50-90 deg
        # thresholds.
        el = np.minimum(el, _EL_ZENITH_CLAMP)

        delta, gamma = self._calc_sun_distance(
            el,
            az,
            sun_el,
            sun_az,
            maxoffset=self._maxoffset,
            tracking_module=self._tracking_module,
        )
        # Starred unpack: upstream a69b8a3 grew the return from 2 to 4 values
        # (component masks appended); this accepts both revisions.
        mask, delta_closest, *_ = self._get_mask_fixed_pos(
            delta,
            gamma,
            self._data.delta,
            self._data.gamma,
            sun_el,
            self._min_solar_altitude,
            alt=el,
            az=az,
            solar_azimuth=sun_az,
            avoidance_shape=self._data.avoidance_shape,
            ignore_island_check=not self._island_check,
        )
        padding = self._maxoffset + self._get_module_offset(self._tracking_module, el, gamma)
        threshold = np.asarray(delta_closest, dtype=float) + padding
        return np.asarray(mask, dtype=bool), threshold


def make_sun_safe(
    model: str = "cad",
    *,
    radius: float | None = None,
    site: Site | None = None,
    min_solar_altitude: float = -90.0,
    island_check: bool = False,
    maxoffset: float = 0.0,
    tracking_module: str = "center",
) -> SunSafePredicate:
    """Build a sun-safety predicate for a named avoidance model.

    The returned object satisfies the scalar
    :class:`~fyst_trajectories.dispatch.SunSafePredicate` contract
    (``(az_deg, el_deg, time) -> bool``) so it can be passed to every
    ``sun_safe=`` seam unchanged, and additionally exposes
    ``batch(az, el, times) -> ndarray[bool]`` and
    ``threshold(az, el, times) -> ndarray[float]`` for grid consumers:
    :func:`~fyst_trajectories.observability.check_observability` uses
    ``batch`` automatically when present, and the visibility renderer's
    ``sun_model`` parameter requires both extensions.

    Parameters
    ----------
    model : str, optional
        ``"cad"`` (default): the shared library's directional CAD zone.
        ``"cone"``: the shared library's cone at ``radius``.
        ``"scalar"``: this library's own isotropic site radius (no extra
        dependency; reproduces today's default behaviour).
    radius : float, optional
        Cone radius in degrees, required for ``model="cone"`` and invalid
        otherwise.
    site : Site, optional
        Observing site (Sun ephemeris and, for ``"scalar"``, the radii).
        Defaults to :func:`~fyst_trajectories.site.get_fyst_site`.
    min_solar_altitude : float, optional
        Sun altitude (deg) below which the constraint is waived entirely.
        Default ``-90.0``: never waived (deviation 1 in the module
        docstring; the shared library's own default is ``0.0``). Ignored
        for ``model="scalar"``.
    island_check : bool, optional
        Apply the shared library's "forbidden island" reachability check on
        top of the zone test. Default False (deviation 2): the point
        predicate answers point safety; reachability belongs to the
        path-level seam (:func:`make_slew_safe`). Ignored for
        ``model="scalar"``.
    maxoffset : float, optional
        Map half-extent padding in degrees, subtracted from the separation
        before the zone test so the verdict covers the whole map footprint.
        Default ``0.0``. Ignored for ``model="scalar"``.
    tracking_module : str, optional
        Module name for the shared library's boresight-to-module padding
        (``"<instrument>_<module>"``). Default ``"center"`` (no padding).
        The pinned library recognises only ``primecam_f280`` / ``f350`` /
        ``eorspec`` (1.78 deg); any other name silently resolves to zero
        padding upstream, so passing one emits a
        :class:`~fyst_trajectories.exceptions.PointingWarning` here.
        Ignored for ``model="scalar"``.

    Returns
    -------
    SunSafePredicate
        A predicate satisfying the scalar contract, additionally exposing
        ``batch``/``threshold``/``model``/``describe`` attributes. When the
        site has Sun avoidance disabled, every model reports all-safe and
        appends ``(disabled)`` to ``describe``.

    Raises
    ------
    RuntimeError
        If ``model`` needs the shared library and it is not installed, or
        the installed CAD table fails the SHA pin.
    ValueError
        On an unknown model, an invalid ``radius`` combination, a
        ``min_solar_altitude`` outside [-90, 90], or a negative/non-finite
        ``maxoffset``.

    Examples
    --------
    >>> sun_safe = make_sun_safe("cone", radius=50.0)
    >>> sun_safe.describe
    'cone 50°'
    """
    site = get_fyst_site() if site is None else site
    # Bounded, not merely finite: a value above the Sun's own altitude range
    # (e.g. a typo'd 91) would silently disable the constraint everywhere.
    if not np.isfinite(min_solar_altitude) or not -90.0 <= min_solar_altitude <= 90.0:
        raise ValueError(f"min_solar_altitude must lie within [-90, 90], got {min_solar_altitude}")
    if not np.isfinite(maxoffset) or maxoffset < 0:
        raise ValueError(f"maxoffset must be finite and >= 0, got {maxoffset}")

    if model == "scalar":
        if radius is not None:
            raise ValueError(
                "radius applies to model='cone'; the scalar model reads "
                "site.sun_avoidance.exclusion_radius."
            )
        return _ScalarSunModel(site)

    data = load_avoidance_data(model, radius=radius)
    if tracking_module != "center" and tracking_module not in _KNOWN_PADDED_MODULES:
        warnings.warn(
            f"tracking_module={tracking_module!r} is not recognised by the pinned "
            f"sun-avoidance library and silently resolves to ZERO boresight-to-module "
            f"padding (known padded names: {_KNOWN_PADDED_MODULES}).",
            PointingWarning,
            stacklevel=2,
        )
    return _LibrarySunModel(
        model,
        data,
        site,
        min_solar_altitude=min_solar_altitude,
        island_check=island_check,
        maxoffset=maxoffset,
        tracking_module=tracking_module,
    )


# ── Path-level slew safety (Phase 6) ─────────────────────────────────────────
def _axis_slew_duration(distance: float, vmax: float, amax: float) -> float:
    """Trapezoidal-profile travel time (s) for one axis over ``distance`` deg.

    Same profile as ``overhead.utils._axis_slew_time`` (deliberate
    duplication: the overhead subpackage sits above this module in the
    import graph); agreement is locked by
    ``tests/test_slew_safety.py::test_axis_profile_endpoints_and_duration``.
    """
    if distance <= 0.0:
        return 0.0
    t_accel = vmax / amax
    d_accel = vmax * t_accel  # accelerate + decelerate distance
    if distance <= d_accel:
        return 2.0 * math.sqrt(distance / amax)
    return 2.0 * t_accel + (distance - d_accel) / vmax


def _axis_positions(delta: float, vmax: float, amax: float, t: np.ndarray) -> np.ndarray:
    """Signed axis positions (deg from start) at times ``t`` for a trapezoidal slew.

    The axis accelerates at ``amax`` to at most ``vmax``, cruises, and
    decelerates symmetrically; positions for ``t`` beyond the axis's own
    arrival time hold at ``delta`` (the other axis may still be moving).
    """
    d = abs(float(delta))
    if d == 0.0:
        return np.zeros_like(t, dtype=float)
    sign = 1.0 if delta >= 0 else -1.0
    t_accel = vmax / amax
    d_accel = vmax * t_accel
    if d <= d_accel:
        # Triangular profile: peak speed reached at t_peak < t_accel.
        t_peak = math.sqrt(d / amax)
        total = 2.0 * t_peak
        rising = 0.5 * amax * np.square(np.clip(t, 0.0, t_peak))
        falling = d - 0.5 * amax * np.square(np.clip(total - t, 0.0, t_peak))
        pos = np.where(t <= t_peak, rising, falling)
    else:
        t_cruise = (d - d_accel) / vmax
        total = 2.0 * t_accel + t_cruise
        d_half = 0.5 * amax * t_accel**2
        accel = 0.5 * amax * np.square(np.clip(t, 0.0, t_accel))
        cruise = d_half + vmax * (np.clip(t, t_accel, t_accel + t_cruise) - t_accel)
        decel = d - 0.5 * amax * np.square(np.clip(total - t, 0.0, t_accel))
        pos = np.select([t <= t_accel, t <= t_accel + t_cruise], [accel, cruise], default=decel)
    pos = np.clip(pos, 0.0, d)
    return sign * np.where(t >= _axis_slew_duration(d, vmax, amax), d, pos)


class _SlewSafeModel:
    """Direct-slew path evaluator sweeping a point model along the motion.

    Satisfies :class:`~fyst_trajectories.dispatch.SlewSafePredicate`. The
    path is the literal encoder travel on each axis under the trapezoidal
    kinematic profile (simultaneous axis starts, each axis holding at its
    goal once arrived), sampled at ``step_seconds``, with the Sun advanced
    along the path; one vectorized point-model ``batch`` call decides.
    """

    def __init__(
        self,
        point_model,
        *,
        az_speed: float,
        az_accel: float,
        el_speed: float,
        el_accel: float,
        step_seconds: float,
    ):
        self._model = point_model
        self._az_speed = az_speed
        self._az_accel = az_accel
        self._el_speed = el_speed
        self._el_accel = el_accel
        self._step = step_seconds
        self.describe = f"slew({getattr(point_model, 'describe', 'point model')})"

    def __call__(
        self,
        current_az: float,
        current_el: float,
        goal_az: float,
        goal_el: float,
        time: Time,
    ) -> bool:
        """Path verdict per the :class:`SlewSafePredicate` contract."""
        safe, _, _, _ = self.evaluate(current_az, current_el, goal_az, goal_el, time)
        return safe

    def evaluate(
        self,
        current_az: float,
        current_el: float,
        goal_az: float,
        goal_el: float,
        time: Time,
    ) -> tuple[bool, np.ndarray, np.ndarray, Time]:
        """Full evaluation: ``(safe, az_path, el_path, times)``.

        ``times[-1]`` is the arrival time, which a detour planner uses as
        the second leg's start.
        """
        if not time.isscalar:
            raise ValueError("slew evaluation takes a scalar start time")
        d_az = float(goal_az) - float(current_az)
        d_el = float(goal_el) - float(current_el)
        duration = max(
            _axis_slew_duration(abs(d_az), self._az_speed, self._az_accel),
            _axis_slew_duration(abs(d_el), self._el_speed, self._el_accel),
        )
        n = max(2, int(math.ceil(duration / self._step)) + 1)
        dt = np.linspace(0.0, duration, n)
        az_path = float(current_az) + _axis_positions(d_az, self._az_speed, self._az_accel, dt)
        el_path = float(current_el) + _axis_positions(d_el, self._el_speed, self._el_accel, dt)
        times = time + TimeDelta(dt, format="sec")
        if hasattr(self._model, "batch"):
            verdicts = np.atleast_1d(
                np.asarray(self._model.batch(az_path, el_path, times), dtype=bool)
            )
            if verdicts.shape != (n,):
                raise ValueError(
                    f"point model batch returned shape {verdicts.shape}, expected ({n},) "
                    "verdicts for the slew path"
                )
        else:
            # Plain SunSafePredicate: consult it per path sample.
            verdicts = np.array(
                [
                    bool(self._model(float(az_path[i]), float(el_path[i]), times[i]))
                    for i in range(n)
                ],
                dtype=bool,
            )
        return bool(np.all(verdicts)), az_path, el_path, times


def make_slew_safe(
    model="cad",
    *,
    radius: float | None = None,
    site: Site | None = None,
    min_solar_altitude: float = -90.0,
    island_check: bool = False,
    maxoffset: float = 0.0,
    tracking_module: str = "center",
    az_speed: float = FYST_AZ_MAX_VELOCITY,
    az_accel: float = FYST_AZ_MAX_ACCELERATION,
    el_speed: float = FYST_EL_MAX_VELOCITY,
    el_accel: float = FYST_EL_MAX_ACCELERATION,
    step_seconds: float = 1.0,
) -> SlewSafePredicate:
    """Build a path-level slew-safety predicate for a named avoidance model.

    Sweeps a point model (:func:`make_sun_safe`) along the direct encoder
    slew path from the current to the goal position, using this library's
    kinematics (the FYST operational axis velocity/acceleration limits,
    overridable), its own Sun ephemeris advanced along the path, and one
    vectorized zone evaluation. The shared library's own ``slew``/detour
    machinery is deliberately not used (mismatched kinematics; unbounded
    intermediate elevations).

    Parameters
    ----------
    model : str or point predicate, optional
        ``"cad"`` (default) / ``"cone"`` / ``"scalar"`` built via
        :func:`make_sun_safe`, or an already-built point predicate (its own
        configuration then applies and the model-related keyword arguments
        here must be left at their defaults; a predicate exposing ``batch``
        is evaluated in one vectorized call per path, a plain
        :class:`~fyst_trajectories.dispatch.SunSafePredicate` per sample).
    radius, site, min_solar_altitude, island_check, maxoffset, tracking_module
        Forwarded to :func:`make_sun_safe` when ``model`` is a name; see
        there. ``island_check`` may be turned on here: reachability is a
        legitimate path-level concern.
    az_speed, az_accel, el_speed, el_accel : float, optional
        Axis kinematics in deg/s and deg/s^2. Default: the FYST
        operational limits
        (:data:`~fyst_trajectories.site.FYST_AZ_MAX_VELOCITY` etc.).
    step_seconds : float, optional
        Path sampling cadence in seconds. Default ``1.0`` (3 deg of
        azimuth travel per sample at the az speed limit, far below the
        50 deg zone floor).

    Returns
    -------
    SlewSafePredicate
        A predicate with ``__call__(current_az, current_el, goal_az,
        goal_el, time) -> bool`` plus an ``evaluate`` method returning
        ``(safe, az_path, el_path, times)``.

    Raises
    ------
    ValueError
        On non-positive/non-finite kinematics or ``step_seconds``, or a
        point predicate passed together with model-building keywords.
    """
    for name, value in (
        ("az_speed", az_speed),
        ("az_accel", az_accel),
        ("el_speed", el_speed),
        ("el_accel", el_accel),
        ("step_seconds", step_seconds),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite value > 0, got {value}")

    if not isinstance(model, str):
        model_kwargs_touched = (
            radius is not None
            or site is not None
            or min_solar_altitude != -90.0
            or island_check
            or maxoffset != 0.0
            or tracking_module != "center"
        )
        if model_kwargs_touched:
            raise ValueError(
                "model is an already-built point predicate; its own configuration "
                "applies, so the model-building keyword arguments must be left at "
                "their defaults."
            )
        point_model = model
    else:
        point_model = make_sun_safe(
            model,
            radius=radius,
            site=site,
            min_solar_altitude=min_solar_altitude,
            island_check=island_check,
            maxoffset=maxoffset,
            tracking_module=tracking_module,
        )
    return _SlewSafeModel(
        point_model,
        az_speed=az_speed,
        az_accel=az_accel,
        el_speed=el_speed,
        el_accel=el_accel,
        step_seconds=step_seconds,
    )


def find_sun_safe_detour(
    current_az: float,
    current_el: float,
    goal_az: float,
    goal_el: float,
    time: Time,
    slew_safe: SlewSafePredicate,
    *,
    site: Site | None = None,
    el_min: float | None = None,
    el_max: float | None = None,
    coarse_step: float = 10.0,
    fine_step: float = 2.0,
) -> tuple[float, float] | None:
    """Find a two-leg detour when the direct slew crosses the Sun zone.

    Searches for an intermediate encoder position ``(az_mid, el_mid)`` at
    the azimuth midpoint of the travel, sweeping the intermediate
    elevation over the commandable range (coarse then fine, preferring
    elevations closest to the direct path's midpoint), such that BOTH
    legs are clear under ``slew_safe`` - the second leg evaluated at the
    first leg's arrival time. Every candidate elevation is bounded to the
    telescope's operational range, so the result is directly commandable
    (unlike the shared library's detour, which returns elevations far
    below the horizon).

    This is a caller-invoked planner: by design,
    :func:`~fyst_trajectories.dispatch.choose_encoder_solution` rejects
    blocked paths rather than auto-rerouting, and the caller decides
    whether to command the two legs this function returns.

    Parameters
    ----------
    current_az, current_el : float
        Current encoder position in degrees.
    goal_az, goal_el : float
        Goal encoder position in degrees (a specific wrap; run the wrap
        choice first).
    time : Time
        Slew start time (scalar).
    slew_safe : SlewSafePredicate
        The path evaluator (from :func:`make_slew_safe`); it must expose
        ``evaluate`` for arrival-time propagation.
    site : Site, optional
        Site providing the default elevation bounds. Defaults to
        :func:`~fyst_trajectories.site.get_fyst_site`.
    el_min, el_max : float, optional
        Intermediate-elevation bounds in degrees. Default: the site's
        telescope elevation limits.
    coarse_step, fine_step : float, optional
        Elevation search steps in degrees. Defaults 10.0 and 2.0.

    Returns
    -------
    tuple of (float, float) or None
        ``(az_mid, el_mid)`` for the intermediate point, or ``None`` when
        no clear two-leg route exists at this azimuth midpoint.

    Raises
    ------
    ValueError
        On invalid steps/bounds, a non-scalar ``time``, or a ``slew_safe``
        without ``evaluate``.
    """
    if not time.isscalar:
        raise ValueError("find_sun_safe_detour takes a scalar start time")
    if not hasattr(slew_safe, "evaluate"):
        raise ValueError(
            "slew_safe must expose evaluate() (build it with make_slew_safe) so the "
            "second leg can start at the first leg's arrival time."
        )
    if (
        not np.isfinite(coarse_step)
        or coarse_step <= 0
        or not np.isfinite(fine_step)
        or fine_step <= 0
    ):
        raise ValueError(
            f"coarse_step/fine_step must be finite and > 0, got {coarse_step}/{fine_step}"
        )

    site = get_fyst_site() if site is None else site
    el_limits = site.telescope_limits.elevation
    lo = el_limits.min if el_min is None else el_min
    hi = el_limits.max if el_max is None else el_max
    if lo > hi:
        raise ValueError(f"el_min ({lo}) must be <= el_max ({hi})")

    az_mid = (float(current_az) + float(goal_az)) / 2.0
    el_direct_mid = (float(current_el) + float(goal_el)) / 2.0

    def _legs_clear(el_mid: float) -> bool:
        ok1, _, _, times1 = slew_safe.evaluate(current_az, current_el, az_mid, el_mid, time)
        if not ok1:
            return False
        return bool(slew_safe(az_mid, el_mid, goal_az, goal_el, times1[-1]))

    coarse = np.arange(lo, hi + 1e-9, coarse_step)
    if hi not in coarse:
        coarse = np.append(coarse, hi)
    coarse = coarse[np.argsort(np.abs(coarse - el_direct_mid), kind="stable")]
    el_coarse_hit = next((float(el) for el in coarse if _legs_clear(float(el))), None)
    if el_coarse_hit is None:
        return None

    # Refine toward the direct-path elevation: among the fine candidates
    # between the coarse hit and the direct midpoint, take the clear one
    # closest to the direct path (least detour).
    fine_lo = min(el_coarse_hit, el_direct_mid)
    fine_hi = max(el_coarse_hit, el_direct_mid)
    fine = np.arange(fine_lo, fine_hi + 1e-9, fine_step)
    fine = np.clip(fine, lo, hi)
    fine = fine[np.argsort(np.abs(fine - el_direct_mid), kind="stable")]
    for el_candidate in fine:
        if _legs_clear(float(el_candidate)):
            return az_mid, float(el_candidate)
    return az_mid, el_coarse_hit
