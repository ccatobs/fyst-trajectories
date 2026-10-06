"""Synthetic trajectories shared by the inject_retune() test modules."""

import numpy as np

from fyst_trajectories.trajectory import (
    SCAN_FLAG_SCIENCE,
    SCAN_FLAG_TURNAROUND,
    Trajectory,
)


def _make_trajectory(
    duration: float = 120.0,
    timestep: float = 0.1,
    turnaround_intervals: list[tuple[float, float]] | None = None,
) -> Trajectory:
    """Create a synthetic trajectory for inject_retune tests.

    Parameters
    ----------
    duration : float
        Total duration in seconds.
    timestep : float
        Time step in seconds.
    turnaround_intervals : list of (start, end) tuples
        Time intervals to flag as turnaround.
    """
    times = np.arange(0, duration, timestep)
    n = len(times)
    az = np.linspace(100, 200, n)
    el = np.full(n, 45.0)
    az_vel = np.gradient(az, times)
    el_vel = np.zeros(n)
    scan_flag = np.full(n, SCAN_FLAG_SCIENCE, dtype=np.int8)

    if turnaround_intervals:
        for t_start, t_end in turnaround_intervals:
            mask = (times >= t_start) & (times < t_end)
            scan_flag[mask] = SCAN_FLAG_TURNAROUND

    return Trajectory(times=times, az=az, el=el, az_vel=az_vel, el_vel=el_vel, scan_flag=scan_flag)


def _group_retune_events(retune_times: np.ndarray) -> list[float]:
    """Group retune flag timestamps into distinct events by start time.

    Returns the start time of each distinct retune event.
    """
    if len(retune_times) == 0:
        return []

    events = [retune_times[0]]
    for i in range(1, len(retune_times)):
        # Gap > 0.2s means a new event
        if retune_times[i] - retune_times[i - 1] > 0.2:
            events.append(retune_times[i])
    return events
