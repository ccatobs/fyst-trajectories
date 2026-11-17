"""The booking screen: the checks the offline scheduler makes on a built scan.

The offline scheduler books a science subscan or a planet pass only when
:func:`_execution_refusal` refuses nothing: no sample inside the Sun zone,
no axis speed or acceleration past the limits of the Go TCS command
layer, and no two samples closer than its ``/path`` minimum interval.
The Sun zone is the site's exclusion radius, which PCS enforces on every
sample before it POSTs a scan, and, when the night is given a
``sun_safe`` model, that model's zone as well, which PCS does not ask. The
axis limits are the ones PCS enforces. The site's own telescope limits are
advisory and play no part here.
"""

from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ..planning import plan_source_ces
from ..planning._sun_safety import _SUN_SAFETY_TRAJECTORY_MAX_PROBES, _trajectory_sun_separation
from ..site import Site
from ..sun_protocols import _sun_verdicts
from ..trajectory import Trajectory
from ..trajectory_utils import (
    GO_TCS_AZ_MAX_ACCELERATION,
    GO_TCS_AZ_MAX_VELOCITY,
    GO_TCS_EL_MAX_ACCELERATION,
    GO_TCS_EL_MAX_VELOCITY,
    GO_TCS_MIN_SAMPLE_INTERVAL_SEC,
    get_absolute_times,
)
from .transitions import DeferralReason

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..sun_protocols import SunSafePredicate

#: The peak acceleration of a quintic turnaround as a multiple of its nominal
#: ``az_accel``.
_QUINTIC_PEAK_ACCEL_FACTOR = 1.5

#: The ``az_accel`` a planet pass is planned with when its ``scan_params``
#: record none: :func:`~fyst_trajectories.planning.plan_source_ces`'s default.
_SOURCE_CES_AZ_ACCEL = inspect.signature(plan_source_ces).parameters["az_accel"].default


def _execution_refusal(
    trajectory: Trajectory,
    scan_type: str,
    scan_params: Mapping[str, Any],
    site: Site,
    sun_safe: SunSafePredicate | None,
) -> DeferralReason | None:
    """Return why the booking screen refuses a built scan, or None when it passes.

    The Sun comes first, when the site's Sun avoidance is enabled: every
    sample must lie outside the site's exclusion radius (a separation at
    the radius is inside), and an injected ``sun_safe`` model must clear
    at most ``_SUN_SAFETY_TRAJECTORY_MAX_PROBES`` evenly strided samples
    and the last one. PCS checks the radius on every sample too, with its
    own arithmetic (the Sun solved on another grid and interpolated), and
    the two separations agree to about 1e-5 deg; a sample within that of
    the radius can be inside for one and outside for the other. An
    injected model's zone is the scheduler's own and is not asked by PCS.

    The axes follow. A pong, daisy or planet pass is read as PCS reads it:
    each axis's speed is the larger of the positions' gradient over the
    sample times (azimuth unwrapped) and its recorded velocity column, and
    with four samples or more its acceleration is the gradient of that
    gradient. A constant-elevation scan's azimuth speed is the one it
    records. A constant-elevation scan or a planet pass turns around with a
    quintic profile whose peak is ``_QUINTIC_PEAK_ACCEL_FACTOR`` times its
    ``az_accel``: the one the trajectory records, or for a planet pass the
    one in ``scan_params``, else the planner's default. A value past a
    Go TCS limit refuses the scan; one at the limit does not.

    Last, the sample interval: two samples closer than
    ``GO_TCS_MIN_SAMPLE_INTERVAL_SEC``, judged on the times relative to the
    first sample as :func:`~fyst_trajectories.trajectory_utils.to_path_format`
    judges them, refuse the scan, which PCS could not serialise. A grid
    built at exactly that interval can fall a few 1e-13 s below it.

    Parameters
    ----------
    trajectory : Trajectory
        The built scan, with a ``start_time``.
    scan_type : str
        ``"constant_el"``, ``"pong"``, ``"daisy"`` or ``"source_ces"``.
    scan_params : Mapping
        The block's recorded ``scan_params``.
    site : Site
        The site whose Sun avoidance policy applies.
    sun_safe : SunSafePredicate or None
        An injected Sun model, asked in addition to the exclusion radius.

    Returns
    -------
    DeferralReason or None
        ``SUN_SCAN`` for a sample inside the Sun zone (the exclusion
        radius, or an injected model's zone), ``AXIS_DYNAMICS`` for an axis
        past a Go TCS limit, which PCS refuses too, ``UNPLANNABLE`` for a
        sample interval under the ``/path`` minimum; None when none holds.

    Raises
    ------
    ValueError
        If ``scan_type`` is none of the four, or the Sun is checked on a
        trajectory without a ``start_time``.
    """
    if scan_type not in ("constant_el", "pong", "daisy", "source_ces"):
        raise ValueError(f"no execution checks for scan_type {scan_type!r}")
    if site.sun_avoidance.enabled and _inside_sun_zone(trajectory, site, sun_safe):
        return DeferralReason.SUN_SCAN
    if scan_type == "constant_el":
        params = trajectory.metadata.pattern_params if trajectory.metadata else {}
        if float(params["az_speed"]) > GO_TCS_AZ_MAX_VELOCITY:
            return DeferralReason.AXIS_DYNAMICS
        az_accel = float(params["az_accel"])
    else:
        if _past_axis_limits(trajectory):
            return DeferralReason.AXIS_DYNAMICS
        az_accel = (
            float(scan_params.get("az_accel", _SOURCE_CES_AZ_ACCEL))
            if scan_type == "source_ces"
            else None
        )
    if az_accel is not None and _QUINTIC_PEAK_ACCEL_FACTOR * abs(az_accel) > (
        GO_TCS_AZ_MAX_ACCELERATION
    ):
        return DeferralReason.AXIS_DYNAMICS
    if _below_sample_interval(trajectory):
        return DeferralReason.UNPLANNABLE
    return None


def _inside_sun_zone(trajectory: Trajectory, site: Site, sun_safe: SunSafePredicate | None) -> bool:
    """Whether a sample is inside the exclusion radius, or refused by ``sun_safe``."""
    separation = _trajectory_sun_separation(site, trajectory)
    if np.any(separation <= site.sun_avoidance.exclusion_radius):
        return True
    if sun_safe is None:
        return False
    n = trajectory.n_points
    step = max(1, math.ceil(n / _SUN_SAFETY_TRAJECTORY_MAX_PROBES))
    index = np.arange(0, n, step)
    if index[-1] != n - 1:
        index = np.append(index, n - 1)
    verdicts = _sun_verdicts(
        sun_safe,
        trajectory.az[index],
        trajectory.el[index],
        get_absolute_times(trajectory)[index],
        what="execution screen",
    )
    return not bool(np.all(verdicts))


def _below_sample_interval(trajectory: Trajectory) -> bool:
    """Whether two samples lie closer than the Go TCS ``/path`` minimum interval.

    The test :func:`~fyst_trajectories.trajectory_utils.to_path_format`
    makes before it serialises a scan, on the same values: the sample times
    relative to the first.
    """
    times = np.asarray(trajectory.times, dtype=float)
    if times.size < 2:
        return False
    return float(np.diff(times - times[0]).min()) < GO_TCS_MIN_SAMPLE_INTERVAL_SEC


def _past_axis_limits(trajectory: Trajectory) -> bool:
    """Whether an axis's speed or acceleration exceeds the Go TCS limit."""
    times = np.asarray(trajectory.times, dtype=float)
    if times.size < 2:
        return False
    az = np.unwrap(np.asarray(trajectory.az, dtype=float), period=360.0)
    el = np.asarray(trajectory.el, dtype=float)
    az_vel = np.gradient(az, times)
    el_vel = np.gradient(el, times)
    az_speed = max(np.abs(az_vel).max(), np.abs(trajectory.az_vel).max())
    el_speed = max(np.abs(el_vel).max(), np.abs(trajectory.el_vel).max())
    if az_speed > GO_TCS_AZ_MAX_VELOCITY or el_speed > GO_TCS_EL_MAX_VELOCITY:
        return True
    if times.size < 4:
        return False
    az_acc = np.abs(np.gradient(az_vel, times)).max()
    el_acc = np.abs(np.gradient(el_vel, times)).max()
    return bool(az_acc > GO_TCS_AZ_MAX_ACCELERATION or el_acc > GO_TCS_EL_MAX_ACCELERATION)
