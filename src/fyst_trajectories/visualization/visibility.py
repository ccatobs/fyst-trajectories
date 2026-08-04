"""Night-level target-visibility and focal-plane footprint figures.

Planning-side renderers for calibration-target work:

- :func:`plot_visibility`: multi-target elevation / azimuth (and optionally
  Sun-separation) vs. time for one observing span, with night/twilight
  shading, sunrise/sunset markers, and sun-proximity highlighting computed
  from the true angular separation (never from per-axis distance to the
  Sun's own curve, which is not a separation test).
- :func:`plot_observability_windows`: the Gantt view of the same span, one
  bar lane per target from ``check_observability``'s windows.
- :func:`plot_array_footprint`: the instantaneous PrimeCam module layout
  projected onto the sky at a given elevation, at honest angular scale,
  showing the Nasmyth field rotation.

These functions require ``matplotlib`` (install via
``pip install fyst-trajectories[plotting]``). They never call
``matplotlib.use()``; backend selection is the caller's business.

Pass ``axes=`` / ``ax=`` to compose panels into an existing figure: the
function then draws into the given axes and returns their figure, and
neither ``plt.show()`` nor ``fig.tight_layout()`` is invoked (both apply
only to figures the function itself creates).

Examples
--------
Render tonight's calibrator visibility in Chilean local time:

>>> from zoneinfo import ZoneInfo
>>> from astropy.time import Time
>>> from fyst_trajectories.visualization import plot_visibility
>>> fig = plot_visibility(
...     Time("2026-11-15T16:00:00", scale="utc"),
...     tz=ZoneInfo("America/Santiago"),
...     show=False,
... )
>>> fig.savefig("visibility.png", dpi=140, bbox_inches="tight")
"""

from datetime import tzinfo
from typing import TYPE_CHECKING

import numpy as np
from astropy.time import Time

from ..coordinates import Coordinates
from ..observability import (
    ASTRONOMICAL_TWILIGHT_ALTITUDE_DEG,
    FLUX_CALIBRATORS,
    SUN_RISE_SET_ALTITUDE_DEG,
    AvoidZone,
    SunEventKind,
    Target,
    TargetKind,
    _build_time_grid,
    _target_altaz_grid,
    check_observability,
    resolve_target,
    sun_events,
)
from ..offsets import _rotate_offset, compute_focal_plane_rotation
from ..primecam import MODULE_FOV_RADIUS_DEG, PRIMECAM_MODULES
from ..site import AtmosphericConditions, Site, get_fyst_site
from ..sun_models import make_sun_safe

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ..offsets import InstrumentOffset

__all__ = [
    "DEFAULT_VISIBILITY_TARGETS",
    "plot_array_footprint",
    "plot_observability_windows",
    "plot_visibility",
]

DEFAULT_VISIBILITY_TARGETS: tuple[str, ...] = tuple(
    name for name, target in FLUX_CALIBRATORS.items() if target.kind is TargetKind.BODY
)
"""Default ``plot_visibility`` target list: every BODY entry of
:data:`~fyst_trajectories.observability.FLUX_CALIBRATORS` (the planets and
the Moon; satellites are excluded because they duplicate their parent-body
proxy curve)."""

#: Valid ``panels`` entries for :func:`plot_visibility`, in canonical order.
VISIBILITY_PANELS = ("elevation", "azimuth", "sun_separation")

#: Color of the Sun's own curve (excluded from the target palette).
SUN_COLOR = "#e6a817"
#: Overlay color for samples inside the exclusion radius.
EXCLUSION_COLOR = "#d62728"
#: Overlay color for samples inside the warning radius.
WARNING_COLOR = "#ff7f0e"
#: Color of the sunrise/sunset event lines and their labels.
EVENT_LINE_COLOR = "0.25"
_NIGHT_SHADE_COLOR = "0.35"

#: Per-panel height (inches) when the function creates its own figure.
_PANEL_HEIGHT = 3.4
_PANEL_WIDTH = 12.0


def _tz_label(tz: tzinfo | None) -> str:
    """Human-readable axis label for a timezone (``"UTC"`` when None)."""
    return str(tz) if tz is not None else "UTC"


def _wrap_break_indices(az: np.ndarray) -> np.ndarray:
    """Return the insertion indices at azimuth jumps larger than 180 deg.

    Fires on 0/360 wraps and on genuine fast swings (a near-zenith transit
    can step the azimuth by ~180 deg between samples); both draw as
    meaningless vertical lines without a break. Computed from the UNMASKED
    curve so that a masked overlay of the same curve gets byte-identical
    insertions (a mask's NaNs would otherwise change the jump pattern and
    desynchronise the x/y lengths).
    """
    return np.flatnonzero(np.abs(np.diff(np.asarray(az, dtype=float))) > 180.0) + 1


def _with_breaks(values: np.ndarray, break_indices: np.ndarray) -> np.ndarray:
    """Copy of ``values`` with NaN inserted at ``break_indices`` (line breaks)."""
    values = np.asarray(values, dtype=float)
    if break_indices.size == 0:
        return values
    return np.insert(values, break_indices, np.nan)


def _shade_night(ax: "Axes", x: np.ndarray, sun_el: np.ndarray) -> None:
    """Shade night (Sun below the rise/set threshold) and astronomical night.

    Thresholds are the same exported constants ``sun_events`` uses, so the
    shading edges coincide with the event lines by construction.
    """
    for threshold in (SUN_RISE_SET_ALTITUDE_DEG, ASTRONOMICAL_TWILIGHT_ALTITUDE_DEG):
        ax.fill_between(
            x,
            0,
            1,
            where=sun_el < threshold,
            transform=ax.get_xaxis_transform(),
            color=_NIGHT_SHADE_COLOR,
            alpha=0.08,
            linewidth=0,
            zorder=0,
        )


def _draw_sun_event_lines(ax: "Axes", events, *, annotate: bool) -> None:
    """Draw sunrise/sunset (dashed, labelled) and astronomical twilight (dotted) lines.

    Multi-day spans repeat the lines but each label is written once (the
    first occurrence) to avoid a picket fence of duplicate text.
    """
    labelled = {SunEventKind.SUNRISE: "sunrise", SunEventKind.SUNSET: "sunset"}
    dotted = (SunEventKind.ASTRONOMICAL_DAWN, SunEventKind.ASTRONOMICAL_DUSK)
    seen_labels: set[SunEventKind] = set()
    for event in events:
        if event.kind in labelled:
            ax.axvline(event.time.plot_date, color=EVENT_LINE_COLOR, ls="--", lw=1.0, zorder=1)
            if annotate and event.kind not in seen_labels:
                seen_labels.add(event.kind)
                # Bottom of the panel: the top-right corner belongs to the legend.
                ax.annotate(
                    labelled[event.kind],
                    xy=(event.time.plot_date, 0.02),
                    xycoords=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=EVENT_LINE_COLOR,
                )
        elif event.kind in dotted:
            ax.axvline(event.time.plot_date, color="0.45", ls=":", lw=0.9, zorder=1)


def plot_visibility(
    time: Time,
    targets: "Sequence[str | Target] | None" = None,
    *,
    site: Site | None = None,
    horizon_hours: float = 24.0,
    step_minutes: float = 2.0,
    el_min: float | None = None,
    atmosphere: AtmosphericConditions | None = None,
    extra_targets: "dict[str, Target] | None" = None,
    tz: tzinfo | None = None,
    sun_model=None,
    panels: "Sequence[str]" = ("elevation", "azimuth"),
    title: str | None = None,
    axes: "Sequence[Axes] | None" = None,
    show: bool = True,
) -> "Figure":
    """Plot target elevation/azimuth (and Sun separation) over an observing span.

    One curve per target on every panel, the Sun as its own distinguished
    curve, night and astronomical-night background shading, sunrise/sunset
    and astronomical-twilight markers from
    :func:`~fyst_trajectories.observability.sun_events`, and a dashed line
    at the elevation floor.

    Sun proximity is highlighted on the target curves themselves. By
    default samples whose **true angular separation** from the Sun is
    inside the site's warning radius are over-drawn in orange, and samples
    inside the exclusion radius in red (radii from ``site.sun_avoidance``).
    The avoidance model is selectable via ``sun_model``: with the shared
    library's directional CAD zone, each target is over-drawn in red where
    *that* model marks it unsafe and the separation panel shows each
    target's own direction-dependent minimum-separation curve. Every
    overlay is omitted when Sun avoidance is disabled on the site. A band
    drawn around the Sun's own elevation or azimuth curve is *not* an
    angular-separation test (azimuth distance degenerates near zenith, and
    per-axis elevation distance ignores azimuth entirely); this function
    deliberately does not draw one.

    Parameters
    ----------
    time : Time
        Start of the span (UTC).
    targets : sequence of (str or Target), optional
        Targets to plot, resolved via
        :func:`~fyst_trajectories.observability.resolve_target`. Default
        :data:`DEFAULT_VISIBILITY_TARGETS` (the planets and the Moon).
    site : Site, optional
        Observing site. Defaults to
        :func:`~fyst_trajectories.site.get_fyst_site`.
    horizon_hours : float, optional
        Span length in hours. Default ``24.0``.
    step_minutes : float, optional
        Sampling cadence in minutes. Default ``2.0``.
    el_min : float, optional
        Elevation floor drawn on the elevation panel and used nowhere
        else. Defaults to the site telescope elevation minimum.
    atmosphere : AtmosphericConditions, optional
        Refraction model for the target/Sun positions. Default ``None``
        (vacuum), matching the library-wide convention; pass
        ``AtmosphericConditions.for_fyst()`` for refraction-aware planning
        (invisible at plot scale above ~20 deg elevation). Night shading
        and the sunrise/sunset/twilight markers always use the vacuum
        almanac convention regardless of this argument.
    extra_targets : dict of str to Target, optional
        Additional catalog (e.g. fixed RA/Dec sources) searched before the
        built-in calibrators, exactly as in ``check_observability``.
    tz : datetime.tzinfo, optional
        Timezone for the x-axis labels (e.g.
        ``zoneinfo.ZoneInfo("America/Santiago")``). Default ``None`` = UTC.
        Computation is always UTC; only the labels change.
    sun_model : str or predicate, optional
        Sun-avoidance model driving the proximity overlays. Default
        ``None``: the site's scalar radii (orange warning + red exclusion
        overlays, fixed guide lines on the separation panel). Pass a model
        name accepted by :func:`~fyst_trajectories.sun_models.make_sun_safe`
        (``"cad"`` or ``"scalar"``; ``"cone"`` needs a radius, so build it
        with ``make_sun_safe("cone", radius=...)`` and pass the object) or
        any predicate exposing ``batch`` / ``threshold`` / ``describe``.
        The ``"cad"`` and ``"cone"`` models require the shared
        sun-avoidance library.
    panels : sequence of str, optional
        Which panels to draw, top to bottom, from
        ``("elevation", "azimuth", "sun_separation")``. Default
        ``("elevation", "azimuth")``.
    title : str, optional
        Title text. On a figure this function creates it is drawn as the
        figure suptitle and defaults to an auto-generated summary. With
        caller-supplied ``axes`` an explicit title is set on the first
        axes instead (the caller's suptitle is never touched) and there is
        no auto-generated default.
    axes : sequence of matplotlib.axes.Axes, optional
        Draw into these axes (one per panel, same order) instead of
        creating a new figure. When given, ``show`` is ignored and no
        layout call is made on the caller's figure.
    show : bool, optional
        Call ``plt.show()`` after rendering (only when the function
        created the figure). Default True.

    Returns
    -------
    Figure
        The figure containing the panels (``axes[0].get_figure()`` when
        ``axes`` was supplied).

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If ``targets`` or ``panels`` is empty, a panel name is unknown,
        ``axes`` does not match ``panels`` in length, or
        ``horizon_hours`` / ``step_minutes`` is not a finite positive
        value.
    """
    try:
        import matplotlib.colors as mcolors  # pylint: disable=import-outside-toplevel
        import matplotlib.dates as mdates  # pylint: disable=import-outside-toplevel
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_visibility(). "
            "Install it with: pip install fyst-trajectories[plotting]"
        ) from None

    if not np.isfinite(horizon_hours) or horizon_hours <= 0:
        raise ValueError(f"horizon_hours must be a finite value > 0, got {horizon_hours}")
    if not np.isfinite(step_minutes) or step_minutes <= 0:
        raise ValueError(f"step_minutes must be a finite value > 0, got {step_minutes}")
    panels = tuple(panels)
    if not panels:
        raise ValueError("panels must not be empty")
    unknown = [p for p in panels if p not in VISIBILITY_PANELS]
    if unknown:
        raise ValueError(f"Unknown panel(s) {unknown}; valid panels: {VISIBILITY_PANELS}")
    if axes is not None and len(axes) != len(panels):
        raise ValueError(f"axes has {len(axes)} entries but {len(panels)} panels were requested")

    site = get_fyst_site() if site is None else site
    coords = Coordinates(site, atmosphere=atmosphere)
    target_list = list(targets) if targets is not None else list(DEFAULT_VISIBILITY_TARGETS)
    if not target_list:
        raise ValueError("targets must not be empty")
    el_floor = site.telescope_limits.elevation.min if el_min is None else el_min

    grid = _build_time_grid(time, horizon_hours, step_minutes)
    x = grid.plot_date
    sun_az, sun_el = (np.atleast_1d(v) for v in coords.get_sun_altaz(grid))
    events = sun_events(time, site=site, horizon_hours=horizon_hours)
    # Night shading follows the vacuum almanac convention, consistent with
    # the sun_events lines; the drawn Sun curve keeps the caller's atmosphere.
    if atmosphere is None:
        sun_el_shading = sun_el
    else:
        _, sun_el_shading = (np.atleast_1d(v) for v in Coordinates(site).get_sun_altaz(grid))

    sun_cfg = site.sun_avoidance
    if isinstance(sun_model, str):
        sun_model = make_sun_safe(sun_model, site=site)
    resolved = [resolve_target(t, extra=extra_targets) for t in target_list]

    # Compute every track, its true Sun separation, and its overlay masks
    # BEFORE creating the figure so a resolution or model error cannot leak
    # a half-built figure. Overlay masks use `<=` (a sample exactly at a
    # radius is NOT clear), matching the conservative boundary of
    # Coordinates.is_sun_safe.
    tracks = []
    for target in resolved:
        t_az, t_el = _target_altaz_grid(coords, target, grid)
        separation = np.atleast_1d(coords.angular_separation(t_az, t_el, sun_az, sun_el))
        threshold = None
        if not sun_cfg.enabled:
            overlays = ()
        elif sun_model is not None:
            unsafe = ~np.atleast_1d(np.asarray(sun_model.batch(t_az, t_el, grid), dtype=bool))
            if unsafe.shape != np.shape(t_el):
                raise ValueError(
                    f"sun_model.batch returned shape {unsafe.shape}, expected "
                    f"{np.shape(t_el)} verdicts for the time grid"
                )
            overlays = ((unsafe, EXCLUSION_COLOR),)
            if "sun_separation" in panels:
                threshold = np.atleast_1d(
                    np.asarray(sun_model.threshold(t_az, t_el, grid), dtype=float)
                )
                if threshold.shape != np.shape(t_el):
                    raise ValueError(
                        f"sun_model.threshold returned shape {threshold.shape}, "
                        f"expected {np.shape(t_el)}"
                    )
        else:
            overlays = (
                (separation <= sun_cfg.warning_radius, WARNING_COLOR),
                (separation <= sun_cfg.exclusion_radius, EXCLUSION_COLOR),
            )
        tracks.append((target, t_az, t_el, separation, overlays, threshold))

    own_fig = axes is None
    if own_fig:
        fig, axs = plt.subplots(
            len(panels),
            1,
            figsize=(_PANEL_WIDTH, _PANEL_HEIGHT * len(panels)),
            sharex=True,
            squeeze=False,
        )
        axes = [row[0] for row in axs]
    else:
        axes = list(axes)
        fig = axes[0].get_figure()

    # Target palette: the rc prop cycle MINUS the reserved semantic colors
    # (the default matplotlib cycle contains both the warning orange and the
    # exclusion red, and a target curve must never masquerade as a sun-zone
    # overlay). Compare in normalized hex so named cycles ('tab:red') and
    # tuple entries are excluded too. Modulo indexing (not zip) so more
    # targets than palette colors are still all drawn, merely reusing colors.
    reserved = {mcolors.to_hex(c) for c in (SUN_COLOR, WARNING_COLOR, EXCLUSION_COLOR)}
    cycle = [
        c
        for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]
        if mcolors.to_hex(c) not in reserved
    ] or ["#1f77b4"]
    colors = [cycle[i % len(cycle)] for i in range(len(tracks))]

    def draw_target_curve(ax: "Axes", y: np.ndarray, overlays, color: str) -> None:
        """One target curve with its precomputed unsafe overlays, azimuth-wrap aware."""
        jumps = _wrap_break_indices(y)
        bx = _with_breaks(x, jumps)
        ax.plot(bx, _with_breaks(y, jumps), color=color, lw=1.4, zorder=3)
        for mask, zone_color in overlays:
            masked_y = _with_breaks(np.where(mask, y, np.nan), jumps)
            ax.plot(bx, masked_y, color=zone_color, lw=2.6, zorder=4)

    for panel, ax in zip(panels, axes):
        _shade_night(ax, x, sun_el_shading)
        _draw_sun_event_lines(ax, events, annotate=panel == panels[0])
        if panel == "elevation":
            for (target, _, t_el, _, overlays, _), color in zip(tracks, colors):
                draw_target_curve(ax, t_el, overlays, color)
            ax.plot(x, sun_el, color=SUN_COLOR, lw=2.2, zorder=2)
            ax.axhline(el_floor, color="0.3", ls="--", lw=1.0, zorder=1)
            ax.annotate(
                f"el_min = {el_floor:.0f}°",
                xy=(0.012, el_floor),
                xycoords=("axes fraction", "data"),
                va="bottom",
                fontsize=8,
                color="0.3",
            )
            ax.set_ylim(0.0, 90.0)
            ax.set_ylabel("Elevation [deg]", fontsize=10)
        elif panel == "azimuth":
            for (target, t_az, _, _, overlays, _), color in zip(tracks, colors):
                draw_target_curve(ax, t_az, overlays, color)
            sun_jumps = _wrap_break_indices(sun_az)
            ax.plot(
                _with_breaks(x, sun_jumps),
                _with_breaks(sun_az, sun_jumps),
                color=SUN_COLOR,
                lw=2.2,
                zorder=2,
            )
            ax.set_ylim(0.0, 360.0)
            ax.set_yticks(np.arange(0.0, 361.0, 90.0))
            ax.set_ylabel("Azimuth [deg]", fontsize=10)
        else:  # sun_separation
            for (target, _, _, sep, _, threshold), color in zip(tracks, colors):
                ax.plot(x, sep, color=color, lw=1.4, zorder=3)
                if threshold is not None:
                    # The target's own (direction-dependent) minimum-safe
                    # separation under the injected model.
                    ax.plot(x, threshold, color=color, lw=1.0, ls="--", alpha=0.75, zorder=2)
            if sun_cfg.enabled and sun_model is None:
                ax.axhline(sun_cfg.exclusion_radius, color=EXCLUSION_COLOR, ls="--", lw=1.2)
                ax.axhline(sun_cfg.warning_radius, color=WARNING_COLOR, ls=":", lw=1.2)
            ax.set_ylim(0.0, 180.0)
            ax.set_ylabel("Sun separation [deg]", fontsize=10)
        ax.grid(ls=":", alpha=0.4)
        ax.set_axisbelow(True)

    # Format EVERY axes, not only the last: caller-supplied axes need not
    # share an x-axis, and unformatted upper panels would show raw date
    # floats. (Fresh locator/formatter instances per axis, as matplotlib
    # requires.)
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=13, tz=tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
        ax.set_xlim(x[0], x[-1])
    axes[-1].set_xlabel(f"Time ({_tz_label(tz)})", fontsize=11)

    handles = [
        plt.Line2D([], [], color=color, lw=1.6, label=track[0].name)
        for track, color in zip(tracks, colors)
    ]
    handles.append(plt.Line2D([], [], color=SUN_COLOR, lw=2.2, label="Sun"))
    if sun_cfg.enabled:
        if sun_model is not None:
            label = f"unsafe ({getattr(sun_model, 'describe', 'injected model')})"
            handles.append(plt.Line2D([], [], color=EXCLUSION_COLOR, lw=2.6, label=label))
            if "sun_separation" in panels:
                handles.append(
                    plt.Line2D(
                        [], [], color="0.4", lw=1.0, ls="--", label="min safe sep (per target)"
                    )
                )
        else:
            for color, label in (
                (EXCLUSION_COLOR, f"< {sun_cfg.exclusion_radius:.0f}° from Sun (exclusion)"),
                (WARNING_COLOR, f"< {sun_cfg.warning_radius:.0f}° (warning)"),
            ):
                handles.append(plt.Line2D([], [], color=color, lw=2.6, label=label))
    axes[0].legend(handles=handles, ncol=4, fontsize=8.5, loc="upper right", framealpha=0.85)

    if own_fig:
        if title is None:
            site_name = site.name.strip() or "site"
            utc_iso = time.utc.iso
            title = (
                f"Target visibility from {site_name} - {utc_iso[:10]} "
                f"({horizon_hours:.0f} h from {utc_iso[11:16]} UTC)"
            )
        fig.suptitle(title, fontsize=13)
    elif title is not None:
        # Composition mode: never touch the caller's figure-level suptitle;
        # an explicitly requested title goes on the first supplied axes.
        axes[0].set_title(title, fontsize=12)

    if own_fig:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        if show:
            plt.show()
    return fig


def plot_observability_windows(
    time: Time,
    targets: "Sequence[str | Target] | None" = None,
    *,
    site: Site | None = None,
    horizon_hours: float = 24.0,
    el_min: float | None = None,
    el_max: float | None = None,
    window_step_minutes: float = 5.0,
    avoid: "list[AvoidZone] | None" = None,
    atmosphere: AtmosphericConditions | None = None,
    extra_targets: "dict[str, Target] | None" = None,
    sun_model=None,
    tz: tzinfo | None = None,
    title: str | None = None,
    ax: "Axes | None" = None,
    show: bool = True,
) -> "Figure":
    """Plot each target's observable windows as one bar lane per target.

    The Gantt-style answer to "which chunks of tonight can I use for which
    target": one horizontal lane per requested target, a bar per
    contiguous interval where every criterion passes (elevation limits,
    the selected sun-avoidance policy, and any ``avoid`` zones), computed
    by :func:`~fyst_trajectories.observability.check_observability` on its
    sampled grid. A target with no window keeps its (empty) lane, so
    "never observable in this horizon" is visible rather than missing.
    Night and astronomical-night shading plus sunrise/sunset and twilight
    markers give the solar context, so no artificial "sun row" is needed.

    Window endpoints land on ``window_step_minutes`` grid samples (no
    interpolation), exactly as reported by ``check_observability``.

    Parameters
    ----------
    time : Time
        Start of the span (UTC).
    targets : sequence of (str or Target), optional
        Targets, one lane each, resolved via
        :func:`~fyst_trajectories.observability.resolve_target`. Default
        :data:`DEFAULT_VISIBILITY_TARGETS`.
    site : Site, optional
        Observing site. Defaults to
        :func:`~fyst_trajectories.site.get_fyst_site`.
    horizon_hours : float, optional
        Span length in hours; must be positive (instant-only mode has no
        windows to draw). Default ``24.0``.
    el_min, el_max : float, optional
        Elevation criteria forwarded to ``check_observability`` (defaults:
        the site telescope limits).
    window_step_minutes : float, optional
        Sampling cadence in minutes. Default ``5.0``.
    avoid : list of AvoidZone, optional
        Bright-source exclusion zones, forwarded unchanged.
    atmosphere : AtmosphericConditions, optional
        Refraction model, forwarded unchanged. Default ``None`` (vacuum).
        Night shading and the sunrise/sunset markers always use the vacuum
        almanac convention regardless of this argument.
    extra_targets : dict of str to Target, optional
        Additional catalog searched before the built-in calibrators.
    sun_model : str or predicate, optional
        Sun-avoidance model deciding the windows, with the same contract
        as ``plot_visibility``: default ``None`` uses the site's scalar
        radii; pass a :func:`~fyst_trajectories.sun_models.make_sun_safe`
        name or any predicate exposing ``batch``.
    tz : datetime.tzinfo, optional
        Timezone for the x-axis labels. Computation is always UTC.
    title : str, optional
        Title text; same semantics as ``plot_visibility`` (suptitle on an
        own figure with an auto-generated default, axes title in
        composition mode).
    ax : matplotlib.axes.Axes, optional
        Draw into this axes instead of creating a new figure. When given,
        ``show`` is ignored and no layout call is made on the caller's
        figure.
    show : bool, optional
        Call ``plt.show()`` after rendering (only when the function
        created the figure). Default True.

    Returns
    -------
    Figure
        The figure containing the lanes (``ax.get_figure()`` when ``ax``
        was supplied).

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If ``targets`` is empty or ``horizon_hours`` is not a finite
        positive value.
    """
    try:
        import matplotlib.colors as mcolors  # pylint: disable=import-outside-toplevel
        import matplotlib.dates as mdates  # pylint: disable=import-outside-toplevel
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_observability_windows(). "
            "Install it with: pip install fyst-trajectories[plotting]"
        ) from None

    if not np.isfinite(horizon_hours) or horizon_hours <= 0:
        raise ValueError(
            f"horizon_hours must be a finite value > 0 (windows need a horizon), "
            f"got {horizon_hours}"
        )
    if not np.isfinite(window_step_minutes) or window_step_minutes <= 0:
        raise ValueError(
            f"window_step_minutes must be a finite value > 0, got {window_step_minutes}"
        )
    site = get_fyst_site() if site is None else site
    target_list = list(targets) if targets is not None else list(DEFAULT_VISIBILITY_TARGETS)
    if not target_list:
        raise ValueError("targets must not be empty")
    if isinstance(sun_model, str):
        sun_model = make_sun_safe(sun_model, site=site)

    # Everything computed BEFORE the figure exists so an error cannot leak
    # a half-built figure.
    reports = check_observability(
        target_list,
        time,
        avoid=avoid,
        site=site,
        horizon_hours=horizon_hours,
        el_min=el_min,
        el_max=el_max,
        atmosphere=atmosphere,
        window_step_minutes=window_step_minutes,
        extra_targets=extra_targets,
        sun_safe=sun_model,
    )
    grid = _build_time_grid(time, horizon_hours, window_step_minutes)
    _, sun_el_shading = (np.atleast_1d(v) for v in Coordinates(site).get_sun_altaz(grid))
    events = sun_events(time, site=site, horizon_hours=horizon_hours)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(_PANEL_WIDTH, max(2.4, 0.55 * len(reports) + 1.2)))
    else:
        fig = ax.get_figure()

    x = grid.plot_date
    _shade_night(ax, x, sun_el_shading)
    _draw_sun_event_lines(ax, events, annotate=True)

    reserved = {mcolors.to_hex(c) for c in (SUN_COLOR, WARNING_COLOR, EXCLUSION_COLOR)}
    cycle = [
        c
        for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]
        if mcolors.to_hex(c) not in reserved
    ] or ["#1f77b4"]

    # First requested target on the TOP lane, reading order.
    for lane, report in enumerate(reports):
        y = len(reports) - 1 - lane
        color = cycle[lane % len(cycle)]
        spans = [
            (w.start.plot_date, w.end.plot_date - w.start.plot_date) for w in (report.windows or ())
        ]
        if spans:
            # Edges drawn too: a single-grid-sample window has zero width and
            # would otherwise render zero pixels, indistinguishable from
            # "never observable".
            ax.broken_barh(
                spans,
                (y - 0.3, 0.6),
                facecolors=color,
                edgecolors=color,
                linewidths=0.8,
                zorder=3,
            )

    ax.set_yticks(range(len(reports)))
    ax.set_yticklabels([r.name for r in reversed(reports)], fontsize=10)
    # Headroom above the top lane so the legend box never covers its bars,
    # and a slightly deeper floor so the sunrise/sunset labels clear the
    # bottom lane.
    ax.set_ylim(-0.75, len(reports) + 0.2)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=13, tz=tz))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel(f"Time ({_tz_label(tz)})", fontsize=11)
    ax.grid(axis="x", ls=":", alpha=0.4)
    ax.set_axisbelow(True)

    el_floor = site.telescope_limits.elevation.min if el_min is None else el_min
    if sun_model is not None:
        policy = getattr(sun_model, "describe", "injected model")
    elif site.sun_avoidance.enabled:
        policy = f"Sun > {site.sun_avoidance.exclusion_radius:.0f}\N{DEGREE SIGN}"
    else:
        policy = "sun avoidance disabled"
    criteria = (
        f"observable: el \N{GREATER-THAN OR EQUAL TO} {el_floor:.0f}\N{DEGREE SIGN}, {policy}"
    )
    if el_max is not None:
        criteria += f", el \N{LESS-THAN OR EQUAL TO} {el_max:.0f}\N{DEGREE SIGN}"
    if avoid:
        criteria += f", {len(avoid)} avoid zone{'s' if len(avoid) != 1 else ''}"
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, facecolor="0.55", label=criteria)],
        loc="upper right",
        fontsize=8.5,
        framealpha=0.85,
    )

    if own_fig:
        if title is None:
            site_name = site.name.strip() or "site"
            utc_iso = time.utc.iso
            title = (
                f"Observability windows from {site_name} - {utc_iso[:10]} "
                f"({horizon_hours:.0f} h from {utc_iso[11:16]} UTC)"
            )
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        if show:
            plt.show()
    elif title is not None:
        ax.set_title(title, fontsize=12)
    return fig


def plot_array_footprint(
    el: float,
    *,
    site: Site | None = None,
    modules: "Mapping[str, InstrumentOffset] | None" = None,
    fov_radius_deg: float = MODULE_FOV_RADIUS_DEG,
    labels: bool = True,
    title: str | None = None,
    ax: "Axes | None" = None,
    show: bool = True,
) -> "Figure":
    """Plot the instantaneous focal-plane module layout on sky at an elevation.

    Rotates each module's boresight offset by the mechanical Nasmyth field
    rotation (:func:`~fyst_trajectories.offsets.compute_focal_plane_rotation`
    with ``parallactic_angle=0``, i.e. ``nasmyth_sign * el +
    instrument_rotation``) and draws it as a circle of its on-sky FOV
    radius in the boresight tangent plane: cross-elevation offset against
    elevation offset, in degrees. The rotation is exact at every elevation
    (offsets are rotated in the tangent plane, never projected through the
    spherical forward map and flattened back, which degenerates toward
    ``el = 90``). The axes are to scale (equal aspect), unlike schematic
    focal-plane rosettes; at FYST opposite PrimeCam module centres sit
    ~3.6 deg apart and the full footprint spans ~4.9 deg edge to edge.

    The layout depends on elevation only through the Nasmyth rotation;
    azimuth plays no role in the boresight-relative frame.

    Parameters
    ----------
    el : float
        Boresight elevation in degrees (0-90).
    site : Site, optional
        Observing site (provides ``nasmyth_sign``). Defaults to
        :func:`~fyst_trajectories.site.get_fyst_site`.
    modules : mapping of str to InstrumentOffset, optional
        Modules to draw. Default
        :data:`~fyst_trajectories.primecam.PRIMECAM_MODULES` (alias keys
        pointing at the same offset are drawn once).
    fov_radius_deg : float, optional
        Per-module on-sky FOV radius in degrees. Default
        :data:`~fyst_trajectories.primecam.MODULE_FOV_RADIUS_DEG` (0.65).
    labels : bool, optional
        Label each module circle with its offset name. Default True.
    title : str, optional
        Axes title. Default is an auto-generated summary including the
        field-rotation angle.
    ax : matplotlib.axes.Axes, optional
        Draw into this axes instead of creating a new figure. When given,
        ``show`` is ignored and no layout call is made on the caller's
        figure.
    show : bool, optional
        Call ``plt.show()`` after rendering (only when the function
        created the figure). Default True.

    Returns
    -------
    Figure
        The figure containing the footprint (``ax.get_figure()`` when
        ``ax`` was supplied).

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If ``el`` is outside [0, 90], ``fov_radius_deg`` is not positive,
        or ``modules`` is empty.
    """
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        from matplotlib.patches import Circle  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_array_footprint(). "
            "Install it with: pip install fyst-trajectories[plotting]"
        ) from None

    if not np.isfinite(el) or not 0.0 <= el <= 90.0:
        raise ValueError(f"el must be within [0, 90] degrees, got {el}")
    if not np.isfinite(fov_radius_deg) or fov_radius_deg <= 0:
        raise ValueError(f"fov_radius_deg must be a finite value > 0, got {fov_radius_deg}")

    site = get_fyst_site() if site is None else site
    modules = PRIMECAM_MODULES if modules is None else modules

    # Alias keys ("c"/"center") reference one offset; draw each offset once.
    unique: list[InstrumentOffset] = []
    for offset in modules.values():
        if not any(offset is seen for seen in unique):
            unique.append(offset)
    if not unique:
        raise ValueError("modules must not be empty")

    # Exact tangent-plane rotation: preserves each module's radial distance
    # at every elevation (a project-through-the-sphere-then-flatten round
    # trip would distort at high elevation and collapse at el = 90).
    placements = []
    for offset in unique:
        rotation = compute_focal_plane_rotation(el, site, offset)
        dx, dy = _rotate_offset(offset, rotation)
        placements.append((offset, float(dx), float(dy)))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
    else:
        fig = ax.get_figure()

    for offset, dx, dy in placements:
        ax.add_patch(
            Circle(
                (dx, dy),
                fov_radius_deg,
                facecolor="#1f77b4",
                alpha=0.25,
                edgecolor="#1f77b4",
                lw=1.4,
            )
        )
        if labels:
            name = (offset.name or "").removeprefix("PrimeCam-") or "?"
            ax.annotate(name, xy=(dx, dy), ha="center", va="center", fontsize=9)

    ax.plot(0.0, 0.0, "+", color="black", ms=10, mew=1.5, zorder=5)

    extent = max(np.hypot(dx, dy) for _, dx, dy in placements) + fov_radius_deg
    limit = 1.15 * extent
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.grid(ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlabel("Cross-elevation offset from boresight [deg]", fontsize=10)
    ax.set_ylabel("Elevation offset from boresight [deg]", fontsize=10)

    if title is None:
        rot_text = f"Nasmyth rotation {site.nasmyth_sign * el:+.1f}°"
        if any(offset.instrument_rotation != 0.0 for offset in unique):
            rot_text += " + per-module instrument rotation"
        title = f"PrimeCam footprint at el = {el:.1f}°  ({rot_text}, to scale)"
    ax.set_title(title, fontsize=11.5)

    if own_fig:
        fig.tight_layout()
        if show:
            plt.show()
    return fig
