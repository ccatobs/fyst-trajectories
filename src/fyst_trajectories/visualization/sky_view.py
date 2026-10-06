"""Instantaneous all-sky chart with sun-policy shading and the array on sky.

:func:`plot_sky_view` renders one moment in time as a polar all-sky chart
(zenith at the center, the horizon on the rim, north up, east to the left:
the astronomer's looking-up convention): every visible catalog body, the
region of sky that is unsafe under the selected sun-avoidance policy
(drawn to the policy's own boundary, so the directional CAD zone renders
its true shape), the band below the telescope elevation floor, and
optionally the PrimeCam footprint projected onto the sky at a boresight, at
honest angular scale.

Two layers are deliberately absent: the surveyed landscape horizon and
site-structure occlusion. Neither model ships with this package.

Requires ``matplotlib`` (``pip install fyst-trajectories[plotting]``).

Examples
--------
This afternoon's sky with the default policy and the array on the Moon:

>>> from astropy.time import Time
>>> from fyst_trajectories.visualization import plot_sky_view
>>> fig = plot_sky_view(
...     Time("2026-11-15T18:00:00", scale="utc"),
...     boresight="moon",
...     show=False,
... )
>>> fig.savefig("sky_view.png", dpi=140, bbox_inches="tight")
"""

from datetime import tzinfo
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from astropy.time import Time

from ..coordinates import Coordinates
from ..exceptions import PointingError
from ..observability import Target, _target_altaz_grid, resolve_target
from ..offsets import _offset_forward, _rotate_offset, compute_focal_plane_rotation
from ..primecam import MODULE_FOV_RADIUS_DEG, PRIMECAM_MODULES
from ..site import Site, get_fyst_site
from ..sun_models import make_sun_safe
from ..sun_protocols import ZonedSunSafePredicate
from ._common import FOOTPRINT_COLOR, _palette, _unique_offsets
from .visibility import (
    DEFAULT_VISIBILITY_TARGETS,
    EXCLUSION_COLOR,
    SUN_COLOR,
    WARNING_COLOR,
    _tz_label,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ..offsets import InstrumentOffset
    from ..sun_protocols import BatchSunSafePredicate

__all__ = ["plot_sky_view"]

_EL_LIMIT_SHADE_COLOR = "0.55"
_RING_SAMPLES = 73


def _boresight_altaz(
    boresight,
    coords: Coordinates,
    time: Time,
    extra_targets: "dict[str, Target] | None",
) -> tuple[float, float]:
    """Resolve ``boresight`` to a single (az, el), raising if it is unusable."""
    if isinstance(boresight, (tuple, list, np.ndarray)):
        if len(boresight) != 2:
            raise ValueError(f"a boresight tuple must be (az, el), got {boresight!r}")
        az_b, el_b = (float(v) for v in boresight)
        # (0, 90]: el = 0 is excluded for the same reason a below-horizon
        # named target is, so both boresight forms agree at the horizon.
        if not np.isfinite(az_b):
            raise ValueError(f"boresight azimuth must be finite, got {az_b}")
        if not 0.0 < el_b <= 90.0:
            raise ValueError(f"boresight elevation must be within (0, 90] degrees, got {el_b}")
        return az_b % 360.0, el_b
    target = resolve_target(boresight, extra=extra_targets)
    az, el = _target_altaz_grid(coords, target, time)
    az_b, el_b = float(az[0]) % 360.0, float(el[0])
    if el_b <= 0.0:
        raise PointingError(
            f"boresight target '{target.name}' is below the horizon at "
            f"{time.utc.iso[:16]} UTC (el {el_b:.1f} deg)"
        )
    return az_b, el_b


def _footprint_on_sky(
    az_b: float,
    el_b: float,
    site: Site,
    modules: "Mapping[str, InstrumentOffset]",
    fov_radius_deg: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sky outlines (az, el arrays) of each unique module's FOV circle.

    Each module's focal-plane FOV circle is rotated by the mechanical
    focal-plane rotation in the tangent plane (rotating the center and
    phase-shifting the circle are the same point set) and mapped through
    the exact spherical forward offset, so outlines are honest at any
    elevation and offset size.
    """
    unique = _unique_offsets(modules)
    ring = np.linspace(0.0, 2.0 * np.pi, _RING_SAMPLES)
    outlines = []
    for offset in unique:
        rotation = compute_focal_plane_rotation(el_b, site=site, offset=offset)
        dx_c, dy_c = _rotate_offset(offset, rotation)
        ring_az, ring_el = _offset_forward(
            az_b,
            el_b,
            dx_c + fov_radius_deg * np.cos(ring),
            dy_c + fov_radius_deg * np.sin(ring),
        )
        outlines.append((np.asarray(ring_az), np.asarray(ring_el)))
    return outlines


class _ZoneLayer(NamedTuple):
    """One shaded layer: the sky where ``lower < values <= upper``.

    ``values`` is sampled on the grid nodes (the cell edges, ``r`` rows by
    ``theta`` columns) when ``on_nodes`` is true, else on the cell centers.
    """

    values: np.ndarray
    lower: float
    upper: float
    on_nodes: bool


def _below(values: np.ndarray, upper: float) -> float:
    """Return a level below every value, so that a layer fills ``values <= upper``."""
    return min(float(np.min(values)), upper) - 1.0


def _batch_on(sun_model, az: np.ndarray, el: np.ndarray, time: Time, what: str) -> np.ndarray:
    """Ask ``sun_model.batch`` for one verdict per grid point, shaped like the grid."""
    verdicts = np.asarray(sun_model.batch(az.ravel(), el.ravel(), time), dtype=bool)
    if verdicts.shape != az.ravel().shape:
        raise ValueError(
            f"sun_model.batch returned shape {verdicts.shape}, expected "
            f"{az.ravel().shape} verdicts for the {what}"
        )
    return verdicts.reshape(az.shape)


def _policy_zones(
    coords: Coordinates,
    site: Site,
    sun_model: "BatchSunSafePredicate | None",
    theta_edges: np.ndarray,
    r_edges: np.ndarray,
    time: Time,
    sun_az: float,
    sun_el: float,
) -> "tuple[_ZoneLayer | None, _ZoneLayer | None]":
    """Return the ``(unsafe, warn_band)`` layers to shade on the polar grid.

    ``(None, None)`` when Sun avoidance is disabled on the site. The default
    path contours the true angular separation on the grid nodes, the zone
    at ``exclusion_radius`` and the warning band up to ``warning_radius``,
    with the at-radius-is-UNSAFE ``<=`` boundary of
    ``Coordinates.is_sun_safe``. An injected model has no warning band. Its
    zone is the contour of separation minus its own ``threshold`` when that
    field reproduces the model's ``batch`` verdicts at every node (nodes
    within 1e-9 deg of the boundary are ties); otherwise, or without a
    ``threshold``, it is the model's ``batch`` verdicts on the cell
    centers, so a zone the threshold does not describe keeps its verdict
    shape.
    """
    sun_cfg = site.sun_avoidance
    if not sun_cfg.enabled:
        return None, None
    if sun_model is not None and not isinstance(sun_model, ZonedSunSafePredicate):
        return _cell_verdicts(sun_model, theta_edges, r_edges, time), None
    az_nodes, el_nodes = np.meshgrid(np.rad2deg(theta_edges), 90.0 - r_edges)
    separation = np.asarray(
        coords.angular_separation(az_nodes.ravel(), el_nodes.ravel(), sun_az, sun_el)
    ).reshape(az_nodes.shape)
    if sun_model is None:
        exclusion, warning = sun_cfg.exclusion_radius, sun_cfg.warning_radius
        return (
            _ZoneLayer(separation, _below(separation, exclusion), exclusion, True),
            _ZoneLayer(separation, exclusion, warning, True),
        )
    verdicts = _batch_on(sun_model, az_nodes, el_nodes, time, "grid nodes")
    threshold = np.asarray(
        sun_model.threshold(az_nodes.ravel(), el_nodes.ravel(), time), dtype=float
    )
    if threshold.shape != az_nodes.ravel().shape:
        raise ValueError(
            f"sun_model.threshold returned shape {threshold.shape}, expected "
            f"{az_nodes.ravel().shape} values for the grid nodes"
        )
    field = separation - threshold.reshape(az_nodes.shape)
    agrees = ((field > 0.0) == verdicts) | (np.abs(field) < 1e-9)
    if np.all(np.isfinite(field)) and np.all(agrees):
        return _ZoneLayer(field, _below(field, 0.0), 0.0, True), None
    return _cell_verdicts(sun_model, theta_edges, r_edges, time), None


def _cell_verdicts(
    sun_model: "BatchSunSafePredicate", theta_edges: np.ndarray, r_edges: np.ndarray, time: Time
) -> _ZoneLayer:
    """Return the model's unsafe cells, from its ``batch`` verdicts at the cell centers."""
    theta_c = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    r_c = 0.5 * (r_edges[:-1] + r_edges[1:])
    az_cells, el_cells = np.meshgrid(np.rad2deg(theta_c), 90.0 - r_c)
    unsafe = ~_batch_on(sun_model, az_cells, el_cells, time, "evaluation grid")
    return _ZoneLayer(unsafe.astype(float), 0.5, 1.5, False)


def _fill_zone(ax, theta_edges: np.ndarray, r_edges: np.ndarray, layer: _ZoneLayer, rgba) -> None:
    """Fill one layer as a single smooth translucent region.

    A semi-transparent ``pcolormesh`` composites every shared cell edge
    twice, rendering the zone as a seam moire instead of a flat wash;
    ``contourf`` draws each connected region as a single filled path. A
    layer on the nodes is contoured directly: the node grid includes the
    zenith row and both seam columns, so the fill covers the whole sky. A
    layer on the cell centers is padded with a wrapped azimuth column
    (closes the theta seam at north) and duplicated pole/rim rows (covers
    the sky inside the first center ring and outside the last).
    """
    if layer.on_nodes:
        theta, r, values = theta_edges, r_edges, layer.values
    else:
        theta_c = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        r_c = 0.5 * (r_edges[:-1] + r_edges[1:])
        theta = np.concatenate([theta_c, theta_c[:1] + 2.0 * np.pi])
        r = np.concatenate([[r_edges[0]], r_c, [r_edges[-1]]])
        values = np.concatenate([layer.values, layer.values[:, :1]], axis=1)
        values = np.vstack([values[:1, :], values, values[-1:, :]])
    ax.contourf(theta, r, values, levels=[layer.lower, layer.upper], colors=[rgba], zorder=1)


def plot_sky_view(
    time: Time,
    targets: "Sequence[str | Target] | None" = None,
    *,
    site: Site | None = None,
    boresight: "str | Target | tuple[float, float] | None" = None,
    sun_model: "str | ZonedSunSafePredicate | None" = None,
    el_min: float | None = None,
    extra_targets: "dict[str, Target] | None" = None,
    modules: "Mapping[str, InstrumentOffset] | None" = None,
    fov_radius_deg: float = MODULE_FOV_RADIUS_DEG,
    grid_step_deg: float = 1.5,
    labels: bool = True,
    tz: tzinfo | None = None,
    title: str | None = None,
    ax: "Axes | None" = None,
    show: bool = True,
) -> "Figure":
    """Plot the whole sky at one instant: bodies, sun-policy zone, array footprint.

    Polar all-sky chart with the zenith at the center and the horizon on
    the rim, north up and east to the left. Draws every target that is
    above the horizon at ``time`` (targets below it are silently omitted),
    the Sun, the sky region unsafe under the selected sun-avoidance policy,
    the band below the telescope elevation floor, and, when ``boresight``
    is given, the PrimeCam module FOV outlines projected onto the sky
    there, at true angular scale with the mechanical focal-plane rotation
    applied.

    With the default ``sun_model=None`` the unsafe region is the sky
    within ``exclusion_radius`` of the Sun (plus the warning annulus out
    to ``warning_radius``), its edges contoured from the true angular
    separation, so they lie on the radii rather than on the grid's cells
    at any ``grid_step_deg``. An injected model's zone is contoured the
    same way from the separation minus its ``threshold`` when that field
    reproduces the model's own ``batch`` verdicts at every grid node, so a
    directional model (e.g. the shared library's CAD zone) renders its
    true, asymmetric shape rather than a circle. A model without
    ``threshold``, or whose threshold does not reproduce its verdicts (as
    the CAD model's may not with its island check or a solar-altitude
    waiver switched on), is drawn from its ``batch`` verdicts at the cell
    centers, and its edge follows the cells. All shading is omitted when
    Sun avoidance is disabled on the site. A below-horizon Sun is reported
    in the legend rather than drawn; its zone shading remains, because the
    default policies carry no night waiver.

    Parameters
    ----------
    time : Time
        The instant to render (scalar, UTC).
    targets : sequence of (str or Target), optional
        Bodies/sources to mark, resolved via
        :func:`~fyst_trajectories.observability.resolve_target`. Default
        :data:`~fyst_trajectories.visualization.DEFAULT_VISIBILITY_TARGETS`.
    site : Site, optional
        Observing site. Defaults to
        :func:`~fyst_trajectories.site.get_fyst_site`.
    boresight : str, Target, or (az, el) pair, optional
        Where to project the PrimeCam footprint: a target name or
        :class:`~fyst_trajectories.observability.Target` (must be above
        the horizon at ``time``) or an explicit ``(az_deg, el_deg)`` pair
        (tuple, list, or array; elevation in ``(0, 90]``). Default
        ``None`` draws no footprint.
    sun_model : str or predicate, optional
        Sun-avoidance model shading the unsafe sky, with the same contract
        as ``plot_visibility``: default ``None`` uses the site's scalar
        radii; pass a :func:`~fyst_trajectories.sun_models.make_sun_safe`
        name (``"cad"``, ``"scalar"``) or any predicate exposing ``batch``
        (and ``threshold``, for a contoured edge).
    el_min : float, optional
        Elevation floor for the shaded rim band, in ``[0, 90]``. Defaults
        to the site telescope elevation minimum.
    extra_targets : dict of str to Target, optional
        Additional catalog searched before the built-in calibrators, for
        both ``targets`` and a named ``boresight``.
    modules : mapping of str to InstrumentOffset, optional
        Modules to draw at the boresight. Default
        :data:`~fyst_trajectories.primecam.PRIMECAM_MODULES` (alias keys
        drawn once).
    fov_radius_deg : float, optional
        Per-module on-sky FOV radius in degrees. Default
        :data:`~fyst_trajectories.primecam.MODULE_FOV_RADIUS_DEG`.
    grid_step_deg : float, optional
        Requested cell size (degrees) of the policy-evaluation grid; each
        axis uses the nearest integer cell count, so the effective steps
        can differ slightly from the request (and between axes). A
        contoured edge is interpolated between the grid's nodes, so the
        step sets how finely its curvature is followed, and where a
        directional model's required separation jumps between table
        levels, the jump is placed to within one cell; a model drawn
        from its cell verdicts has an edge that follows the cells
        throughout. Default ``1.5``.
    labels : bool, optional
        Annotate each drawn body with its name on the chart. The legend
        always identifies the Sun and every drawn body regardless.
        Default True.
    tz : datetime.tzinfo, optional
        Timezone appended to the auto-generated title as a local-time
        stamp. Computation is always UTC.
    title : str, optional
        Axes title. Default is an auto-generated summary.
    ax : matplotlib.axes.Axes, optional
        Draw into this **polar** axes (``fig.add_subplot(...,
        projection="polar")``) instead of creating a new figure. When
        given, ``show`` is ignored and no layout call is made on the
        caller's figure.
    show : bool, optional
        Call ``plt.show()`` after rendering (only when the function
        created the figure). Default True.

    Returns
    -------
    Figure
        The figure containing the chart (``ax.get_figure()`` when ``ax``
        was supplied).

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If ``time`` is not scalar, ``targets`` is empty, ``modules`` is
        empty while a ``boresight`` is given, ``grid_step_deg`` or
        ``fov_radius_deg`` is not a finite positive value
        (``grid_step_deg`` at most 30), ``el_min`` is outside [0, 90],
        ``ax`` is not a polar axes, the boresight is malformed (an
        explicit pair at or below the horizon included), or an injected
        ``sun_model``'s ``batch`` or ``threshold`` returns the wrong shape.
    PointingError
        If a ``boresight`` given as a name or a
        :class:`~fyst_trajectories.observability.Target` is at or below
        the horizon at ``time``.
    AttributeError
        If an injected ``sun_model`` has no ``batch`` method. The base
        :class:`~fyst_trajectories.sun_protocols.SunSafePredicate` contract
        makes ``batch`` optional, so a plain predicate such as
        ``Coordinates.is_sun_safe`` is refused here; build the model with
        :func:`~fyst_trajectories.sun_models.make_sun_safe`.
    """
    try:
        import matplotlib.colors as mcolors  # pylint: disable=import-outside-toplevel
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        from matplotlib.patches import Patch  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_sky_view(). "
            "Install it with: pip install fyst-trajectories[plotting]"
        ) from None

    if not time.isscalar:
        raise ValueError("time must be a scalar Time (one instant); got an array")
    if not np.isfinite(grid_step_deg) or not 0.0 < grid_step_deg <= 30.0:
        raise ValueError(f"grid_step_deg must be a finite value in (0, 30], got {grid_step_deg}")
    if not np.isfinite(fov_radius_deg) or fov_radius_deg <= 0:
        raise ValueError(f"fov_radius_deg must be a finite value > 0, got {fov_radius_deg}")
    if ax is not None and getattr(ax, "name", "") != "polar":
        raise ValueError(
            "ax must be a polar axes; create it with projection='polar' "
            '(e.g. fig.add_subplot(1, 1, 1, projection="polar"))'
        )

    site = get_fyst_site() if site is None else site
    coords = Coordinates(site)
    target_list = list(targets) if targets is not None else list(DEFAULT_VISIBILITY_TARGETS)
    if not target_list:
        raise ValueError("targets must not be empty")
    el_floor = site.telescope_limits.elevation.min if el_min is None else el_min
    if not np.isfinite(el_floor) or not 0.0 <= el_floor <= 90.0:
        raise ValueError(f"el_min must be within [0, 90] degrees, got {el_floor}")
    modules = PRIMECAM_MODULES if modules is None else modules

    sun_az, sun_el = coords.get_sun_altaz(time)
    if isinstance(sun_model, str):
        sun_model = make_sun_safe(sun_model, site=site)

    # Everything below is computed BEFORE the figure exists so a resolution
    # or model error cannot leak a half-built figure. The palette index is
    # the position in the REQUESTED list, so a body keeps its color
    # regardless of which other targets happen to be up.
    visible = []
    for index, entry in enumerate(target_list):
        target = resolve_target(entry, extra=extra_targets)
        t_az, t_el = _target_altaz_grid(coords, target, time)
        if float(t_el[0]) > 0.0:
            visible.append((index, target, float(t_az[0]) % 360.0, float(t_el[0])))

    # Policy-evaluation grid: r = 90 - el (zenith at the center). The zones
    # are contoured on its nodes (the edges, zenith row and both seam
    # columns included); the cell-verdict fallback samples the cell
    # centers, the top one below 90 deg, clear of the zenith azimuth
    # degeneracy.
    r_edges = np.linspace(0.0, 90.0, max(1, round(90.0 / grid_step_deg)) + 1)
    theta_edges = np.deg2rad(np.linspace(0.0, 360.0, max(1, round(360.0 / grid_step_deg)) + 1))

    sun_cfg = site.sun_avoidance
    unsafe, warn_band = _policy_zones(
        coords, site, sun_model, theta_edges, r_edges, time, sun_az, sun_el
    )

    outlines = None
    if boresight is not None:
        az_b, el_b = _boresight_altaz(boresight, coords, time, extra_targets)
        outlines = _footprint_on_sky(az_b, el_b, site, modules, fov_radius_deg)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8.0, 8.0), subplot_kw={"projection": "polar"})
    else:
        fig = ax.get_figure()

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)  # east to the left: the looking-up convention

    if warn_band is not None:
        _fill_zone(ax, theta_edges, r_edges, warn_band, mcolors.to_rgba(WARNING_COLOR, 0.22))
    if unsafe is not None:
        _fill_zone(ax, theta_edges, r_edges, unsafe, mcolors.to_rgba(EXCLUSION_COLOR, 0.30))

    rim = np.linspace(0.0, 2.0 * np.pi, 241)
    ax.fill_between(
        rim, 90.0 - el_floor, 90.0, color=_EL_LIMIT_SHADE_COLOR, alpha=0.35, lw=0, zorder=2
    )

    # A below-horizon Sun would land outside rlim and clip silently; its
    # state goes in the legend instead so night charts stay unambiguous
    # (the zone shading is still drawn: the scalar policy has no night
    # waiver, so a set Sun legitimately keeps excluding sky).
    sun_up = sun_el > 0.0
    if sun_up:
        ax.plot(
            np.deg2rad(sun_az), 90.0 - sun_el, "o", ms=14, color=SUN_COLOR, mec="#b8860b", zorder=6
        )
        if labels:
            ax.annotate(
                f"Sun (el {sun_el:.0f} deg)",
                (np.deg2rad(sun_az), 90.0 - sun_el),
                textcoords="offset points",
                xytext=(9, 8),
                fontsize=9,
                zorder=7,
            )
    # Body-marker palette: the rc prop cycle MINUS the reserved semantic
    # colors (Sun, zone overlays, footprint), so a body marker can never
    # masquerade as one of them; same defence as plot_visibility.
    palette = _palette((SUN_COLOR, EXCLUSION_COLOR, WARNING_COLOR, FOOTPRINT_COLOR), "#2ca02c")
    body_handles = []
    for index, target, t_az, t_el in visible:
        color = palette[index % len(palette)]
        ax.plot(np.deg2rad(t_az), 90.0 - t_el, "o", ms=7, color=color, mec="k", mew=0.4, zorder=6)
        body_handles.append(
            plt.Line2D([], [], ls="", marker="o", ms=6, color=color, label=target.name)
        )
        if labels:
            ax.annotate(
                target.name,
                (np.deg2rad(t_az), 90.0 - t_el),
                textcoords="offset points",
                xytext=(7, 5),
                fontsize=8,
                zorder=7,
            )

    if outlines is not None:
        for ring_az, ring_el in outlines:
            ax.plot(
                np.deg2rad(ring_az), 90.0 - ring_el, "-", color=FOOTPRINT_COLOR, lw=1.1, zorder=5
            )
        ax.plot(np.deg2rad(az_b), 90.0 - el_b, "+", color="black", ms=9, mew=1.4, zorder=6)

    ax.set_xticks(np.deg2rad([0.0, 90.0, 180.0, 270.0]))
    ax.set_xticklabels(["N", "E", "S", "W"])
    # No tick at r = 0: it would label the zenith POINT and collide with a
    # near-zenith footprint; the center is unambiguous anyway.
    ax.set_rticks([30.0, 60.0, 90.0])
    ax.set_yticklabels(
        ["el 60 deg", "30 deg", "0 deg"],
        fontsize=7,
    )
    ax.set_rlabel_position(202.5)
    ax.set_rlim(0.0, 90.0)
    ax.grid(ls=":", alpha=0.4)

    # The Sun and every drawn body are always identified in the legend, so
    # the chart stays readable with labels=False (annotations off).
    sun_label = "Sun" if sun_up else f"Sun below horizon (el {sun_el:.0f} deg)"
    handles = [plt.Line2D([], [], ls="", marker="o", ms=8, color=SUN_COLOR, label=sun_label)]
    handles.extend(body_handles)
    if unsafe is not None:
        if sun_model is not None:
            zone_label = f"unsafe ({getattr(sun_model, 'describe', 'injected model')})"
        else:
            radius = f"{sun_cfg.exclusion_radius:.0f} deg"
            zone_label = f"<= {radius} from Sun (exclusion)"
        handles.append(Patch(facecolor=mcolors.to_rgba(EXCLUSION_COLOR, 0.30), label=zone_label))
    if warn_band is not None:
        handles.append(
            Patch(
                facecolor=mcolors.to_rgba(WARNING_COLOR, 0.22),
                label=(
                    f"{sun_cfg.exclusion_radius:.0f}-{sun_cfg.warning_radius:.0f} deg (warning)"
                ),
            )
        )
    handles.append(
        Patch(
            facecolor=_EL_LIMIT_SHADE_COLOR,
            alpha=0.35,
            label=f"below el limit ({el_floor:.0f} deg)",
        )
    )
    if outlines is not None:
        handles.append(
            plt.Line2D([], [], color=FOOTPRINT_COLOR, lw=1.4, label="Prime-Cam footprint")
        )
    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=8,
        framealpha=0.85,
        ncol=2 if len(handles) > 7 else 1,
    )

    if title is None:
        site_name = site.name.strip() or "site"
        title = f"Sky view from {site_name} - {time.utc.iso[:16]} UTC"
        if tz is not None:
            # via .utc: to_datetime(timezone=) rejects non-UTC Time scales.
            local = time.utc.to_datetime(timezone=tz)
            title += f" ({local:%H:%M} {_tz_label(tz)})"
    ax.set_title(title, fontsize=11.5)

    if own_fig:
        fig.tight_layout()
        if show:
            plt.show()
    return fig
