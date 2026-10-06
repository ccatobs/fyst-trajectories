"""Trajectory diagnostics: the 3-panel az/el/sky-track figure."""

from typing import TYPE_CHECKING

import numpy as np

from ..trajectory import Trajectory

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "plot_trajectory",
]


def plot_trajectory(
    trajectory: Trajectory,
    *,
    title: str | None = None,
    axes: "Sequence[Axes] | None" = None,
    show: bool = True,
) -> "Figure":
    """Plot trajectory az/el vs time and the az/el track.

    Creates a 3-panel figure showing azimuth vs time, elevation vs time,
    and azimuth vs elevation. The az/el panel is drawn at the true angular
    shape about the track's mean elevation: one degree of azimuth spans
    ``cos(el)`` degrees on sky. ``cos(el)`` is floored at 0.05, so above a
    mean elevation of about 87.1 deg the panel is stretched in azimuth.

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory to plot.
    title : str, optional
        Title text. On a figure this function creates it is drawn as the
        figure suptitle; with caller-supplied ``axes`` it replaces the
        first panel's title (the caller's suptitle is never touched).
        Default None, no title beyond the three panel titles.
    axes : sequence of matplotlib.axes.Axes, optional
        Draw into these three axes (azimuth, elevation, track, in that
        order) instead of creating a new figure. When given, ``show`` is
        ignored and no layout call is made on the caller's figure.
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
        If ``axes`` does not hold exactly three axes.
    """
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_trajectory(). "
            "Install it with: pip install fyst-trajectories[plotting]"
        ) from None

    own_fig = axes is None
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    else:
        axes = list(axes)
        if len(axes) != 3:
            raise ValueError(f"axes has {len(axes)} entries but plot_trajectory draws 3 panels")
        fig = axes[0].get_figure()

    axes[0].plot(trajectory.times, trajectory.az)
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Azimuth [deg]")
    axes[0].set_title("Az vs Time")

    axes[1].plot(trajectory.times, trajectory.el)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Elevation [deg]")
    axes[1].set_title("El vs Time")

    axes[2].plot(trajectory.az, trajectory.el)
    axes[2].set_xlabel("Azimuth [deg]")
    axes[2].set_ylabel("Elevation [deg]")
    axes[2].set_title("Az/El Track")
    # Floor the cosine so a track through the zenith still gets a finite aspect.
    cos_el = max(float(np.cos(np.radians(np.mean(trajectory.el)))), 0.05)
    axes[2].set_aspect(1.0 / cos_el)

    if own_fig:
        if title is not None:
            fig.suptitle(title)
        fig.tight_layout()
        if show:
            plt.show()
    elif title is not None:
        axes[0].set_title(title)

    return fig
