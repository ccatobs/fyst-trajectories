"""Trajectory diagnostics: the 3-panel az/el/sky-track figure."""

from typing import TYPE_CHECKING

from ..trajectory import Trajectory

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "plot_trajectory",
]


def plot_trajectory(trajectory: Trajectory, *, show: bool = True) -> "Figure":
    """Plot trajectory az/el vs time and sky track.

    Creates a 3-panel figure showing azimuth vs time, elevation vs time,
    and azimuth vs elevation (sky track).

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory to plot.
    show : bool, optional
        Whether to call ``plt.show()`` after creating the figure.
        Default True.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_trajectory(). "
            "Install it with: pip install fyst-trajectories[plotting]"
        ) from None

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(trajectory.times, trajectory.az)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Azimuth (deg)")
    axes[0].set_title("Az vs Time")

    axes[1].plot(trajectory.times, trajectory.el)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Elevation (deg)")
    axes[1].set_title("El vs Time")

    axes[2].plot(trajectory.az, trajectory.el)
    axes[2].set_xlabel("Azimuth (deg)")
    axes[2].set_ylabel("Elevation (deg)")
    axes[2].set_title("Sky Track")
    axes[2].set_aspect("equal")

    fig.tight_layout()

    if show:
        plt.show()

    return fig
