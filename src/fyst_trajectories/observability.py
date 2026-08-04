"""Observability reporting for solar-system targets (OBSERVE / EXCLUDE primitives).

This module answers "which of these targets can we observe now (or over the
next *N* hours), and why not?" for a list of solar-system flux calibrators. It
is the assess-only sibling of the trajectory builders
(:class:`~fyst_trajectories.patterns.PlanetTrackPattern`, ``plan_*_scan``): it
returns a per-target :class:`ObservabilityReport`, it never builds a trajectory,
and it never raises for an unobservable target (the reason is reported, not
excepted).

It also owns the site-level Sun almanac: :func:`sun_events` reports sunrise,
sunset, and twilight crossings for an observing site, independent of any
target list.

It is intentionally importable in isolation: it depends only on
:mod:`fyst_trajectories.coordinates` and :mod:`fyst_trajectories.site` (astropy +
numpy underneath) and does **not** import the offline ``overhead`` simulator.

Two physically distinct kinds of avoidance are kept structurally separate:

* **Sun**: always-on thermal/hardware safety, read from
  ``site.sun_avoidance`` (a 50 deg exclusion radius by default: the inscribed
  cone (floor) of FYST's directional CAD-derived avoidance zone, whose minimum
  Sun separation runs 50-90 deg with the Sun's direction in the mount frame;
  pass ``sun_safe=`` to swap in that directional model, see
  :mod:`fyst_trajectories.sun_models`). Reported in the dedicated
  ``sun_separation_deg`` / ``sun_clear`` fields with a
  :attr:`ReasonCode.SUN_TOO_CLOSE` reason. It is never an :class:`AvoidZone` and
  cannot be weakened via the ``avoid`` list.
* **Bright-source contamination**: caller-specified, variable, per-body
  exclusion zones (the Moon, Jupiter, ...). Reported as
  :class:`AvoidSeparation` entries with an :attr:`ReasonCode.AVOID_TOO_CLOSE`
  reason. There are **no library default zones**: every :class:`AvoidZone`
  carries its own caller-supplied radius.

The orchestration that turns ``schedule(OBSERVE=[...], AVOID=[...])`` into
selected blocks lives one layer up (the scheduling layer), not here; this
module supplies the observability primitive the scheduler calls.

Examples
--------
>>> from astropy.time import Time
>>> from fyst_trajectories.observability import AvoidZone, check_observability
>>> reports = check_observability(
...     ["uranus", "neptune", "mars", "titan"],
...     Time("2026-06-15T05:00:00", scale="utc"),
...     avoid=[AvoidZone("jupiter", 3.0), AvoidZone("moon", 5.0)],
...     horizon_hours=24.0,
... )
>>> observable_now = [r.name for r in reports if r.observable]
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

from .coordinates import SOLAR_SYSTEM_BODIES, Coordinates
from .site import AtmosphericConditions, Site, get_fyst_site

if TYPE_CHECKING:
    # Annotation-only import to avoid an import cycle: ``dispatch`` imports
    # ``coordinates``/``site``/``exceptions`` at runtime, and ``__init__``
    # imports ``dispatch`` before ``observability``. The predicate is invoked
    # structurally, so only the type hint needs the symbol.
    from .dispatch import SunSafePredicate


# ── Target catalog ───────────────────────────────────────────────────────────
class TargetKind(str, enum.Enum):
    """How a target's position is resolved."""

    BODY = "body"
    """A major body resolved directly by astropy (planet, Moon, Sun)."""
    SATELLITE = "satellite"
    """A planetary moon. Its position is approximated by its
    ``parent_body`` (e.g. Titan -> Saturn), which is adequate (<= ~3 arcmin)
    for observability but flagged via ``ObservabilityReport.position_approximate``."""
    FIXED = "fixed"
    """A fixed ICRS RA/Dec source."""


@dataclass(frozen=True)
class Target:
    """A resolvable observing target.

    Parameters
    ----------
    name : str
        Target name (canonicalised to lower-case at construction).
    kind : TargetKind
        How the position is resolved.
    parent_body : str or None, optional
        For ``SATELLITE`` targets, the major body whose position is used as the
        proxy. Must be a member of
        :data:`~fyst_trajectories.coordinates.SOLAR_SYSTEM_BODIES`.
    ra_deg, dec_deg : float or None, optional
        For ``FIXED`` targets, the ICRS coordinates in degrees.
    aliases : tuple of str, optional
        Alternative names accepted by :func:`resolve_target`
        (case-insensitive).

    Raises
    ------
    ValueError
        If the per-kind required fields are missing or invalid.
    """

    name: str
    kind: TargetKind
    parent_body: str | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", self.name.lower())
        if self.parent_body is not None:
            object.__setattr__(self, "parent_body", self.parent_body.lower())

        if self.kind == TargetKind.BODY:
            if self.name not in SOLAR_SYSTEM_BODIES:
                raise ValueError(
                    f"BODY target '{self.name}' is not a supported body. "
                    f"Supported: {SOLAR_SYSTEM_BODIES}"
                )
        elif self.kind == TargetKind.SATELLITE:
            if self.parent_body is None or self.parent_body not in SOLAR_SYSTEM_BODIES:
                raise ValueError(
                    f"SATELLITE target '{self.name}' requires parent_body in "
                    f"{SOLAR_SYSTEM_BODIES} (proxy position)."
                )
        elif self.kind == TargetKind.FIXED:
            if self.ra_deg is None or self.dec_deg is None:
                raise ValueError(f"FIXED target '{self.name}' requires both ra_deg and dec_deg.")


# Built-in submm flux-calibrator catalog. This resolves names -> positions only;
# it carries NO avoidance defaults. Titan is a SATELLITE proxied by Saturn
# (a real satellite ephemeris is a possible future addition).
FLUX_CALIBRATORS: dict[str, Target] = {
    "mars": Target("mars", TargetKind.BODY),
    "jupiter": Target("jupiter", TargetKind.BODY),
    "saturn": Target("saturn", TargetKind.BODY),
    "uranus": Target("uranus", TargetKind.BODY),
    "neptune": Target("neptune", TargetKind.BODY),
    "moon": Target("moon", TargetKind.BODY, aliases=("luna",)),
    "titan": Target("titan", TargetKind.SATELLITE, parent_body="saturn"),
}
"""Built-in submm flux-calibrator catalog mapping names to :class:`Target` positions.

Resolution only; it carries no avoidance defaults. Titan is a ``SATELLITE``
proxied by its parent body. Looked up (after any caller ``extra``) by
``resolve_target`` and ``check_observability``.
"""


def resolve_target(
    name_or_target: str | Target,
    *,
    extra: dict[str, Target] | None = None,
) -> Target:
    """Resolve a target name (or pass a :class:`Target` through unchanged).

    Lookup is case-insensitive and checks ``extra`` (caller-supplied) before
    :data:`FLUX_CALIBRATORS`, matching canonical names first and then aliases.

    Parameters
    ----------
    name_or_target : str or Target
        A catalog name/alias, or an already-built :class:`Target`.
    extra : dict of str to Target, optional
        Additional catalog (e.g. fixed RA/Dec sources) searched first.

    Returns
    -------
    Target
        The resolved target.

    Raises
    ------
    ValueError
        If the name is not found in any catalog.
    """
    if isinstance(name_or_target, Target):
        return name_or_target

    key = name_or_target.lower()
    for catalog in (extra, FLUX_CALIBRATORS):
        if not catalog:
            continue
        if key in catalog:
            return catalog[key]
        for target in catalog.values():
            if key in tuple(alias.lower() for alias in target.aliases):
                return target

    raise ValueError(
        f"Unknown target '{name_or_target}'. Known calibrators: "
        f"{sorted(FLUX_CALIBRATORS)}. Pass a Target (or extra_targets=) for "
        f"fixed sources or bodies outside the built-in catalog."
    )


def _resolve_avoid_body(name: str, *, extra: dict[str, Target] | None = None) -> str:
    """Resolve an AVOID body name to a canonical ``SOLAR_SYSTEM_BODIES`` body.

    AVOID bodies are alias- and satellite-aware, exactly like targets: ``"luna"``
    resolves to ``"moon"`` and a SATELLITE name (``"titan"``) resolves to its
    parent-body proxy (``"saturn"``). The result is always a member of
    :data:`~fyst_trajectories.coordinates.SOLAR_SYSTEM_BODIES`, so it can be
    passed straight to :meth:`Coordinates.get_body_altaz`. A FIXED catalog entry
    has no body identity and is rejected.

    Parameters
    ----------
    name : str
        AVOID body name or alias.
    extra : dict of str to Target, optional
        Caller-supplied catalog searched before the built-in calibrators.

    Returns
    -------
    str
        The canonical body name to evaluate the zone against.

    Raises
    ------
    ValueError
        If ``name`` is not a resolvable body (unknown name, or a FIXED source
        with no body position).
    """
    key = name.lower()
    if key in SOLAR_SYSTEM_BODIES:
        return key
    try:
        target = resolve_target(name, extra=extra)
    except ValueError as exc:
        raise ValueError(
            f"AVOID body '{name}' is not a resolvable solar-system body. "
            f"Supported: {SOLAR_SYSTEM_BODIES} (plus catalog aliases/satellites)."
        ) from exc
    if target.kind == TargetKind.SATELLITE:
        return target.parent_body  # already canonical + lower-cased
    if target.kind == TargetKind.BODY:
        return target.name
    raise ValueError(
        f"AVOID body '{name}' resolves to a FIXED source with no body position; "
        f"AVOID bodies must be solar-system bodies ({SOLAR_SYSTEM_BODIES})."
    )


# ── AVOID variable-zone model (no default zone) ──────────────────────────────
@dataclass(frozen=True)
class AvoidZone:
    """A caller-specified bright-source exclusion zone.

    A target fails this zone when its separation from ``body`` is less than
    ``zone_deg``. There is intentionally **no default** for ``zone_deg``: it is
    a required field, so ``AvoidZone("moon")`` is a ``TypeError``.

    Parameters
    ----------
    body : str
        Contaminating body name (lower-cased at construction). Must be a major
        body resolvable by
        :meth:`~fyst_trajectories.coordinates.Coordinates.get_body_altaz`.
    zone_deg : float
        Exclusion radius in degrees (must be non-negative).

    Raises
    ------
    ValueError
        If ``zone_deg`` is negative, or if ``body`` is ``"sun"`` (the Sun is
        always-on hardware safety, never an AVOID zone).
    """

    body: str
    zone_deg: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "body", self.body.lower())
        if self.body == "sun":
            raise ValueError(
                "the Sun is always-on hardware safety, not an AvoidZone; the report "
                "carries it in the sun_clear / sun_separation_deg fields."
            )
        # Finite check first: ``nan < 0`` is False, so a non-finite zone would
        # otherwise slip past the ``< 0`` guard and silently break separations.
        if not np.isfinite(self.zone_deg):
            raise ValueError(f"zone_deg must be finite, got {self.zone_deg}")
        if self.zone_deg < 0:
            raise ValueError(f"zone_deg must be >= 0, got {self.zone_deg}")

    @classmethod
    def from_pair(cls, pair: tuple[str, float | str]) -> AvoidZone:
        """Build from a ``(body, zone)`` pair, accepting ``'3deg'``/``'3'``/``3.0``.

        Convenience for a scheduler wrapper parsing the ``AVOID`` list. A pair
        with a missing or empty zone raises ``ValueError``; it never silently
        defaults.

        Parameters
        ----------
        pair : tuple of (str, float or str)
            The ``(body, zone)`` pair.

        Returns
        -------
        AvoidZone
            The parsed zone.

        Raises
        ------
        ValueError
            If the pair is malformed (wrong shape), or the zone is
            missing/empty, non-numeric, or non-finite.
        """
        # ``isinstance`` guard precedes ``len`` so a 2-char str (e.g. "xy")
        # does not pass ``len == 2`` and unpack into characters.
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"AVOID entry {pair!r} must be a (body, zone) pair with a radius.")
        body, zone = pair
        if isinstance(zone, str):
            cleaned = zone.strip().lower().removesuffix("deg").removesuffix("°").strip()
            if not cleaned:
                raise ValueError(f"AVOID entry for '{body}' has no zone radius.")
            zone = cleaned
        try:
            zone_deg = float(zone)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"AVOID entry for '{body}' has a non-numeric zone radius {zone!r}."
            ) from exc
        return cls(body, zone_deg)


# ── Report schema ────────────────────────────────────────────────────────────
class ReasonCode(str, enum.Enum):
    """Why a target is (not) observable.

    Currently emits ``BELOW_EL_MIN``, ``ABOVE_EL_MAX``, ``SUN_TOO_CLOSE``, and
    ``AVOID_TOO_CLOSE``. ``OK``, ``BELOW_HORIZON``, ``NEVER_RISES``, and
    ``WINDOW_TOO_SHORT`` are reserved and are not yet produced.
    """

    OK = "ok"
    BELOW_HORIZON = "below_horizon"
    BELOW_EL_MIN = "below_el_min"
    ABOVE_EL_MAX = "above_el_max"
    SUN_TOO_CLOSE = "sun_too_close"
    AVOID_TOO_CLOSE = "avoid_too_close"
    NEVER_RISES = "never_rises"
    WINDOW_TOO_SHORT = "window_too_short"


@dataclass(frozen=True)
class AvoidSeparation:
    """Separation of a target from one AVOID body at the report instant.

    Parameters
    ----------
    body : str
        The contaminating body, the caller-supplied body name (not the
        resolved proxy/canonical body).
    zone_deg : float
        The caller-supplied exclusion radius (degrees).
    separation_deg : float
        Angular separation between the target and the body (degrees).
    clear : bool
        ``True`` when ``separation_deg >= zone_deg``.
    """

    body: str
    zone_deg: float
    separation_deg: float
    clear: bool


@dataclass(frozen=True)
class ObservabilityWindow:
    """One contiguous interval (within the horizon) where all criteria pass.

    Parameters
    ----------
    start, end : Time
        Window bounds.
    duration_hours : float
        ``end - start`` in hours.
    truncated_start : bool
        ``True`` if the window abuts the start of the query horizon (it may
        extend earlier than ``start``).
    truncated_end : bool
        ``True`` if the window abuts the end of the query horizon (it may
        extend later than ``end``).

    Notes
    -----
    Window endpoints land on grid samples spaced at most
    ``window_step_minutes`` apart (see :func:`check_observability`); a true
    rise/set is resolved only to that granularity and a sub-step excursion
    above/below the criteria between two samples may be missed. No
    interpolation is performed.
    """

    start: Time
    end: Time
    duration_hours: float
    truncated_start: bool = False
    truncated_end: bool = False


@dataclass(frozen=True)
class ObservabilityReport:
    """Per-target observability verdict.

    Parameters
    ----------
    name : str
        Canonical target name.
    target : Target
        The resolved target.
    time : Time
        The instant the report was evaluated at.
    observable : bool
        ``True`` iff ``reasons`` is empty.
    az_deg, el_deg : float
        Target azimuth (raw astropy ``[0, 360)``, unnormalised) and elevation
        at ``time``.
    ra_deg, dec_deg : float
        Target ICRS coordinates at ``time``.
    sun_separation_deg : float
        Angular separation from the Sun at ``time``.
    sun_clear : bool
        ``True`` when the target is outside the Sun exclusion radius (or Sun
        avoidance is disabled).
    avoid_separations : tuple of AvoidSeparation
        One entry per evaluated AVOID zone, in input order (self-exclusion
        entries omitted; duplicate bodies each get their own entry).
    reasons : tuple of ReasonCode
        Empty iff ``observable``.
    windows : tuple of ObservabilityWindow, or None
        Every observable window within the horizon, in time order. ``None``
        in instant-only mode (``horizon_hours == 0``: windows were not
        evaluated); an empty tuple when a horizon was evaluated and no
        window exists. A target can have several disjoint windows in one
        horizon (e.g. an evening pass and a pre-dawn pass), so summing
        or selecting from this tuple is the caller's choice;
        :attr:`total_observable_hours` sums the durations (a lower bound,
        see its docstring).
    position_approximate : bool
        ``True`` when a ``SATELLITE`` target used a parent-body proxy position
        (Titan is evaluated at Saturn).
    """

    name: str
    target: Target
    time: Time
    observable: bool
    az_deg: float
    el_deg: float
    ra_deg: float
    dec_deg: float
    sun_separation_deg: float
    sun_clear: bool
    avoid_separations: tuple[AvoidSeparation, ...]
    reasons: tuple[ReasonCode, ...]
    windows: tuple[ObservabilityWindow, ...] | None = None
    position_approximate: bool = False

    @property
    def summary(self) -> str:
        """One-line human-readable verdict."""
        if self.observable:
            return f"{self.name}: observable (el={self.el_deg:.1f}°)"
        reasons = ", ".join(r.value for r in self.reasons)
        return f"{self.name}: NOT observable - {reasons}"

    @property
    def total_observable_hours(self) -> float:
        """Sum of all window durations in hours (0.0 when none, or not evaluated).

        A lower bound: window endpoints land on grid samples (see
        :class:`ObservabilityWindow`), so each window underestimates the true
        interval by up to two ``window_step_minutes`` (one per endpoint; one
        step for a window truncated at a horizon edge) and the bias
        accumulates across windows.
        """
        return float(sum(w.duration_hours for w in self.windows or ()))


# ── Internal helpers ─────────────────────────────────────────────────────────
def _build_time_grid(time: Time, horizon_hours: float, step_minutes: float) -> Time:
    """Build the sample grid. Length 1 (just ``time``) when ``horizon_hours <= 0``."""
    if horizon_hours and horizon_hours > 0:
        horizon_s = horizon_hours * 3600.0
        step_s = step_minutes * 60.0
        # Cover [time, time + horizon] with uniform step_s spacing. ceil so the
        # interval is fully covered; n >= 2 so a positive horizon is always a real
        # interval (never a degenerate length-1 grid). The final sample is clipped
        # to land exactly on time + horizon (no sample past the horizon), so the
        # last cell may be shorter than step_s. Endpoints therefore land on grid
        # samples spaced <= step_minutes apart.
        n = max(2, int(np.ceil(horizon_s / step_s)) + 1)
        offsets_s = np.minimum(np.arange(n) * step_s, horizon_s)
    else:
        n = 1
        offsets_s = np.arange(1) * (step_minutes * 60.0)
    return time + TimeDelta(offsets_s, format="sec")


def _all_windows(ok: np.ndarray, grid: Time) -> tuple[ObservabilityWindow, ...]:
    """Every contiguous ``True`` run of ``ok`` as an :class:`ObservabilityWindow`, in time order."""
    ok = np.asarray(ok, dtype=bool)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return ()
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    last_i = len(ok) - 1
    windows: list[ObservabilityWindow] = []
    for run in runs:
        start_i, end_i = int(run[0]), int(run[-1])
        start_t, end_t = grid[start_i], grid[end_i]
        windows.append(
            ObservabilityWindow(
                start=start_t,
                end=end_t,
                duration_hours=float((end_t - start_t).to_value(u.hour)),
                truncated_start=start_i == 0,
                truncated_end=end_i == last_i,
            )
        )
    return tuple(windows)


def _target_altaz_grid(
    coords: Coordinates, target: Target, grid: Time
) -> tuple[np.ndarray, np.ndarray]:
    """Target (az, el) over the grid; SATELLITE uses its parent-body proxy."""
    if target.kind == TargetKind.FIXED:
        az, el = coords.radec_to_altaz(target.ra_deg, target.dec_deg, grid)
    else:
        body = target.parent_body if target.kind == TargetKind.SATELLITE else target.name
        az, el = coords.get_body_altaz(body, grid)
    return np.atleast_1d(az), np.atleast_1d(el)


# ── Public entry point ───────────────────────────────────────────────────────
def check_observability(
    targets: list[str | Target],
    time: Time,
    *,
    avoid: list[AvoidZone] | None = None,
    site: Site | None = None,
    horizon_hours: float = 0.0,
    el_min: float | None = None,
    el_max: float | None = None,
    atmosphere: AtmosphericConditions | None = None,
    window_step_minutes: float = 5.0,
    extra_targets: dict[str, Target] | None = None,
    sun_safe: SunSafePredicate | None = None,
) -> list[ObservabilityReport]:
    """Assess each target's observability now (and optionally over a horizon).

    Returns one :class:`ObservabilityReport` per input target, in input order.
    It never raises for an unobservable target (the reason is reported); it
    raises only on a genuinely unknown target name or a malformed AVOID body.

    The Sun check is always-on (thermal/hardware safety, independent of
    ``avoid``); the ``avoid`` list is exclusively for caller-specified
    bright-source contamination zones, each carrying its own radius. A target
    is never excluded by its own glare (self-exclusion: an :class:`AvoidZone`
    whose body matches the target's resolved position body, ``parent_body``
    for a SATELLITE, ``name`` for a BODY, is skipped).

    Parameters
    ----------
    targets : list of (str or Target)
        Names (resolved via :func:`resolve_target`) or explicit targets.
    time : Time
        The instant to evaluate (and the start of the horizon window).
    avoid : list of AvoidZone, optional
        Bright-source exclusion zones. Each body must be a major body
        resolvable by ``get_body_altaz``.
    site : Site, optional
        Observing site. Defaults to :func:`~fyst_trajectories.site.get_fyst_site`.
    horizon_hours : float, optional
        If ``> 0``, also compute every observable window over
        ``[time, time + horizon_hours]``. Default ``0.0`` (instant only).
    el_min, el_max : float, optional
        Elevation limits in degrees. Default to the site telescope limits.
    atmosphere : AtmosphericConditions, optional
        Refraction model. Default ``None`` (vacuum), matching the library-wide
        vacuum-by-default convention; pass
        ``AtmosphericConditions.for_fyst()`` for refraction-aware planning.
    window_step_minutes : float, optional
        Horizon sampling cadence in minutes. Default ``5.0``. Window endpoints
        land on these grid samples (no interpolation): a rise/set is resolved
        only to this granularity and sub-step excursions may be missed.
    extra_targets : dict of str to Target, optional
        Additional catalog (e.g. fixed RA/Dec sources) searched before the
        built-in calibrators.
    sun_safe : SunSafePredicate, optional
        Sun-safety predicate implementing the
        :class:`~fyst_trajectories.dispatch.SunSafePredicate` contract,
        ``(az_deg, el_deg, time) -> bool`` returning ``True`` when the
        position is clear of the Sun. ``None`` (default) keeps the built-in
        scalar check, ``separation > site.sun_avoidance.exclusion_radius``
        (a position exactly at the exclusion radius counts as unsafe). An
        injected predicate drives the ``sun_clear`` /
        :attr:`ReasonCode.SUN_TOO_CLOSE` verdict end-to-end: one exposing
        the optional ``batch`` extension (e.g. from
        :func:`~fyst_trajectories.sun_models.make_sun_safe`) is evaluated
        in a single vectorized call over the grid, and any other predicate
        is consulted per grid sample ``(az_i, el_i, time_i)``. The reported
        ``sun_separation_deg`` is still the geometric Sun separation
        regardless of the predicate. Has no effect when Sun avoidance is
        disabled on the site.

    Returns
    -------
    list of ObservabilityReport
        One report per input target, in order.
    """
    site = get_fyst_site() if site is None else site
    coords = Coordinates(site, atmosphere=atmosphere)
    avoid = list(avoid) if avoid else []

    el_limits = site.telescope_limits.elevation
    el_min = el_limits.min if el_min is None else el_min
    el_max = el_limits.max if el_max is None else el_max
    if el_min > el_max:
        raise ValueError(f"el_min ({el_min}) must be <= el_max ({el_max})")
    if horizon_hours and horizon_hours > 0 and window_step_minutes <= 0:
        raise ValueError(f"window_step_minutes must be > 0, got {window_step_minutes}")

    grid = _build_time_grid(time, horizon_hours, window_step_minutes)
    n = len(grid)
    want_window = bool(horizon_hours and horizon_hours > 0)

    # Sun and AVOID-body positions are target-independent over a shared grid:
    # compute each once and reuse across all targets.
    sun_az, sun_el = coords.get_sun_altaz(grid)
    sun_az, sun_el = np.atleast_1d(sun_az), np.atleast_1d(sun_el)
    sun_enabled = site.sun_avoidance.enabled
    sun_radius = site.sun_avoidance.exclusion_radius

    # Resolve every AVOID body to a canonical body up front (alias/satellite
    # aware, like targets) so an unresolvable body raises a clear error here
    # rather than as a deep get_body_altaz failure inside the grid loop.
    resolved_bodies = [_resolve_avoid_body(zone.body, extra=extra_targets) for zone in avoid]
    avoid_pos: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for resolved in resolved_bodies:
        if resolved not in avoid_pos:
            b_az, b_el = coords.get_body_altaz(resolved, grid)
            avoid_pos[resolved] = (np.atleast_1d(b_az), np.atleast_1d(b_el))

    reports: list[ObservabilityReport] = []
    for name in targets:
        target = resolve_target(name, extra=extra_targets)
        # Body whose position the target is evaluated AT (for self-exclusion):
        # a SATELLITE is proxied by its parent; a FIXED source has no body identity.
        if target.kind == TargetKind.FIXED:
            position_body = None
        else:
            position_body = (
                target.parent_body if target.kind == TargetKind.SATELLITE else target.name
            )
        az_grid, el_grid = _target_altaz_grid(coords, target, grid)
        az0, el0 = float(az_grid[0]), float(el_grid[0])
        if target.kind == TargetKind.FIXED:
            ra0, dec0 = float(target.ra_deg), float(target.dec_deg)
        else:
            # Report RA/Dec consistent with the computed Az/El by inverting the
            # apparent position. (Equivalent to get_body_radec, which also returns
            # the apparent place; inverting here avoids a second ephemeris call.)
            ra0, dec0 = coords.altaz_to_radec(az0, el0, time)

        sun_sep_grid = np.atleast_1d(coords.angular_separation(az_grid, el_grid, sun_az, sun_el))
        if not sun_enabled:
            sun_ok_grid = np.ones(n, dtype=bool)
        elif sun_safe is None:
            # Strict `>`: conservative thermal/hardware stance, matching is_sun_safe
            # (a target exactly at the exclusion radius is NOT clear).
            sun_ok_grid = sun_sep_grid > sun_radius
        elif hasattr(sun_safe, "batch"):
            # Batch-capable predicate (e.g. from sun_models.make_sun_safe):
            # one vectorized evaluation over the whole grid instead of a
            # per-sample ephemeris call each iteration.
            sun_ok_grid = np.atleast_1d(
                np.asarray(sun_safe.batch(az_grid, el_grid, grid), dtype=bool)
            )
            if sun_ok_grid.shape != (n,):
                # A scalar/short result would silently broadcast one verdict
                # over the whole grid; fail loudly instead.
                raise ValueError(
                    f"sun_safe.batch returned shape {sun_ok_grid.shape}, expected "
                    f"({n},) verdicts for the horizon grid"
                )
        else:
            # Injected directional model: consult it per grid sample. ``False``
            # marks an unsafe (inside-the-zone) sample. The reported
            # ``sun_separation_deg`` above is still the geometric separation.
            sun_ok_grid = np.array(
                [bool(sun_safe(float(az_grid[i]), float(el_grid[i]), grid[i])) for i in range(n)],
                dtype=bool,
            )

        el_ok_grid = (el_grid >= el_min) & (el_grid <= el_max)

        avoid_ok_grid = np.ones(n, dtype=bool)
        instant_avoid: list[AvoidSeparation] = []
        for zone, resolved_body in zip(avoid, resolved_bodies):
            # Self-exclusion compares RESOLVED positions: AVOIDing the target's
            # own (alias/satellite-resolved) body is skipped.
            if resolved_body == position_body:
                continue
            b_az, b_el = avoid_pos[resolved_body]
            sep_grid = np.atleast_1d(coords.angular_separation(az_grid, el_grid, b_az, b_el))
            # `>=`: an AvoidZone radius is a caller-supplied minimum separation, so
            # a target exactly at the radius is clear (cf. the Sun's conservative `>`).
            clear_grid = sep_grid >= zone.zone_deg
            avoid_ok_grid &= clear_grid
            instant_avoid.append(
                AvoidSeparation(
                    body=zone.body,  # echo the caller's name, not the resolved proxy
                    zone_deg=zone.zone_deg,
                    separation_deg=float(sep_grid[0]),
                    clear=bool(clear_grid[0]),
                )
            )

        ok_grid = el_ok_grid & sun_ok_grid & avoid_ok_grid

        reasons: list[ReasonCode] = []
        if el0 < el_min:
            reasons.append(ReasonCode.BELOW_EL_MIN)
        if el0 > el_max:
            reasons.append(ReasonCode.ABOVE_EL_MAX)
        if not bool(sun_ok_grid[0]):
            reasons.append(ReasonCode.SUN_TOO_CLOSE)
        if any(not sep.clear for sep in instant_avoid):
            reasons.append(ReasonCode.AVOID_TOO_CLOSE)

        reports.append(
            ObservabilityReport(
                name=target.name,
                target=target,
                time=time,
                observable=not reasons,
                az_deg=az0,
                el_deg=el0,
                ra_deg=ra0,
                dec_deg=dec0,
                sun_separation_deg=float(sun_sep_grid[0]),
                sun_clear=bool(sun_ok_grid[0]),
                avoid_separations=tuple(instant_avoid),
                reasons=tuple(reasons),
                windows=_all_windows(ok_grid, grid) if want_window else None,
                position_approximate=target.kind == TargetKind.SATELLITE,
            )
        )

    return reports


# ── Sun events (sunrise / sunset / twilight) ─────────────────────────────────
SUN_RISE_SET_ALTITUDE_DEG: float = -0.8333
"""Sun-centre altitude defining sunrise and sunset, in degrees.

The standard almanac convention: -50 arcmin, folding mean horizon refraction
(~34') and the solar semidiameter (~16') into the threshold. The altitudes it
is compared against are geometric (vacuum), so refraction must not be applied
twice.

Horizon dip is deliberately excluded (the sea-level convention): at FYST's
5611.8 m the dip to a sea-level horizon is ~2 deg, so the Sun remains visible
from the summit for ~10 minutes after the reported sunset (and before the
reported sunrise). These events are the standard twilight-scheduling almanac
times, not "when direct sunlight stops hitting the dish".
"""

CIVIL_TWILIGHT_ALTITUDE_DEG: float = -6.0
"""Sun-centre altitude defining civil dawn/dusk, in degrees (geometric)."""

NAUTICAL_TWILIGHT_ALTITUDE_DEG: float = -12.0
"""Sun-centre altitude defining nautical dawn/dusk, in degrees (geometric)."""

ASTRONOMICAL_TWILIGHT_ALTITUDE_DEG: float = -18.0
"""Sun-centre altitude defining astronomical dawn/dusk, in degrees (geometric)."""


class SunEventKind(str, enum.Enum):
    """Kind of solar altitude-crossing event."""

    SUNRISE = "sunrise"
    SUNSET = "sunset"
    CIVIL_DAWN = "civil_dawn"
    CIVIL_DUSK = "civil_dusk"
    NAUTICAL_DAWN = "nautical_dawn"
    NAUTICAL_DUSK = "nautical_dusk"
    ASTRONOMICAL_DAWN = "astronomical_dawn"
    ASTRONOMICAL_DUSK = "astronomical_dusk"


@dataclass(frozen=True)
class SunEvent:
    """A solar altitude-crossing event.

    Parameters
    ----------
    kind : SunEventKind
        Which threshold was crossed, and in which direction.
    time : Time
        The crossing time (linear interpolation between grid samples).
    altitude_deg : float
        The threshold altitude that was crossed, in degrees.
    rising : bool
        ``True`` for the dawn-side events (sunrise and the dawns), ``False``
        for the dusk side. Derived from ``kind``; carried as a field so
        callers can filter without enumerating kinds.
    """

    kind: SunEventKind
    time: Time
    altitude_deg: float
    rising: bool


# (threshold, rising kind, setting kind) per tracked altitude.
_SUN_EVENT_THRESHOLDS: tuple[tuple[float, SunEventKind, SunEventKind], ...] = (
    (SUN_RISE_SET_ALTITUDE_DEG, SunEventKind.SUNRISE, SunEventKind.SUNSET),
    (CIVIL_TWILIGHT_ALTITUDE_DEG, SunEventKind.CIVIL_DAWN, SunEventKind.CIVIL_DUSK),
    (NAUTICAL_TWILIGHT_ALTITUDE_DEG, SunEventKind.NAUTICAL_DAWN, SunEventKind.NAUTICAL_DUSK),
    (
        ASTRONOMICAL_TWILIGHT_ALTITUDE_DEG,
        SunEventKind.ASTRONOMICAL_DAWN,
        SunEventKind.ASTRONOMICAL_DUSK,
    ),
)


def _threshold_crossings(
    values: np.ndarray, grid: Time, threshold: float, *, rising: bool
) -> list[Time]:
    """Linearly interpolated times where ``values`` crosses ``threshold``.

    ``rising=True`` finds upward crossings (``values[i] < threshold <=
    values[i+1]``); ``rising=False`` downward ones (``values[i] >= threshold >
    values[i+1]``). The two masks partition each grid cell, so one cell yields
    at most one crossing and a value exactly at the threshold is never counted
    twice. Interpolation uses the actual per-cell spacing, so a clipped final
    grid cell (see :func:`_build_time_grid`) is handled exactly.
    """
    below, above = values[:-1], values[1:]
    if rising:
        mask = (below < threshold) & (above >= threshold)
    else:
        mask = (below >= threshold) & (above < threshold)
    times: list[Time] = []
    for i in np.flatnonzero(mask):
        denom = values[i + 1] - values[i]
        frac = 0.0 if abs(denom) < 1e-12 else (threshold - values[i]) / denom
        times.append(grid[i] + frac * (grid[i + 1] - grid[i]))
    return times


def sun_events(
    time: Time,
    *,
    site: Site | None = None,
    horizon_hours: float = 24.0,
    step_minutes: float = 1.0,
) -> tuple[SunEvent, ...]:
    """Find sunrise, sunset, and twilight events over a time span.

    Solves the real solar ephemeris on a uniform grid (one vectorized
    :meth:`~fyst_trajectories.coordinates.Coordinates.get_sun_altaz` call)
    and locates every crossing of the standard thresholds: sunrise/sunset
    at :data:`SUN_RISE_SET_ALTITUDE_DEG` (-0.8333 deg, the almanac
    convention folding refraction and semidiameter into the threshold) and
    civil/nautical/astronomical dawn/dusk at -6/-12/-18 deg geometric.
    Crossing times are linearly interpolated between grid samples; at the
    default 1-minute step the interpolation error is at the seconds level.
    The error grows roughly quadratically with the step, and a threshold
    touch contained entirely between two samples is missed, so keep
    ``step_minutes`` small relative to the twilight timescale (the default
    resolves every event).

    The Sun altitudes are geometric (vacuum transform). Do not feed
    refracted altitudes to these thresholds: the rise/set convention
    already accounts for refraction, so applying it twice would bias the
    times.

    Parameters
    ----------
    time : Time
        Start of the search span.
    site : Site, optional
        Observing site. Defaults to :func:`~fyst_trajectories.site.get_fyst_site`.
    horizon_hours : float, optional
        Span to search, in hours. Default ``24.0`` (one full day: each
        event kind typically occurs once).
    step_minutes : float, optional
        Grid cadence in minutes. Default ``1.0``.

    Returns
    -------
    tuple of SunEvent
        Every event in ``[time, time + horizon_hours]``, sorted by time.
        A threshold the Sun never crosses in the span contributes no
        events.

    Raises
    ------
    ValueError
        If ``horizon_hours`` or ``step_minutes`` is not positive.

    Examples
    --------
    >>> from astropy.time import Time
    >>> from fyst_trajectories.observability import SunEventKind, sun_events
    >>> events = sun_events(Time("2026-11-15T00:00:00", scale="utc"))
    >>> sunset = next(e for e in events if e.kind == SunEventKind.SUNSET)
    """
    # Finite checks first: NaN slips past a bare ``<= 0`` (both comparisons are
    # False) and would silently return an empty event tuple.
    if not np.isfinite(horizon_hours) or horizon_hours <= 0:
        raise ValueError(f"horizon_hours must be a finite value > 0, got {horizon_hours}")
    if not np.isfinite(step_minutes) or step_minutes <= 0:
        raise ValueError(f"step_minutes must be a finite value > 0, got {step_minutes}")

    site = get_fyst_site() if site is None else site
    coords = Coordinates(site)  # vacuum: the thresholds are geometric conventions

    grid = _build_time_grid(time, horizon_hours, step_minutes)
    _, sun_el = coords.get_sun_altaz(grid)
    sun_el = np.atleast_1d(np.asarray(sun_el, dtype=float))

    events: list[SunEvent] = []
    for threshold, rise_kind, set_kind in _SUN_EVENT_THRESHOLDS:
        for kind, rising in ((rise_kind, True), (set_kind, False)):
            for t_cross in _threshold_crossings(sun_el, grid, threshold, rising=rising):
                events.append(
                    SunEvent(kind=kind, time=t_cross, altitude_deg=threshold, rising=rising)
                )

    events.sort(key=lambda e: e.time.mjd)
    return tuple(events)
