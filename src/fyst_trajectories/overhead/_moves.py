"""The shared move vocabulary of the two offline loops.

The survey-night scheduler and the calibration-night sequencer both have
to answer one question before they park, slew or plan anything: has the
Sun zone overtaken the pose the telescope is at, and if so where does it
go? This module answers it once, for both.

:func:`plan_escape_move` decides and returns the decision as a value:
the escape itself, the ``sun_escape`` slew block that records it, and the
label an idle must carry when the telescope cannot move. What to do with
that decision stays with the caller, because the two loops genuinely
differ there: one restarts its tick, the other plans a visit from the
escape pose or shortens an idle by the move.

:func:`sweep_sun_safe` is the other check both loops share: the
fail-closed Sun sweep over a planned pass, which certifies the pass before
it is emitted.

Both loops plan an escape under the path-level safety model and the
schedule's own elevation floor, and both leave the telescope where it is
when the escape does not fit in what is left of the window, labelling the
idle ``no_escape``: the pose stays inside the zone whether it is the zone
or the clock that holds it. Only what follows that differs, and by
design, the survey loop stopping where the calibration night fills its
remaining sliver.

Where the two loops still differ, deliberately:

- **The identity of a multi-pass calibration's passes.** Both give each
  pass one, by different but internally consistent conventions: the
  survey scheduler keeps the sequence as a single scan and numbers the
  passes in ``subscan_index``, while the calibration night treats each
  pass as its own scan. Either way ``(scan_index, subscan_index)`` is
  unique.
- **The gap between consecutive passes.** The calibration night records
  it as an idle labelled ``waiting_for_pass``; the survey scheduler
  folds it into the following pass block as acquisition time. Both are
  gap-free and honest about the pose; the difference is what each loop
  is for, so it stands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from astropy.time import Time

from ..sun_protocols import BatchSunSafePredicate
from .models import TimelineBlock
from .transitions import DeferralReason, Transition, plan_escape

if TYPE_CHECKING:
    from ..site import Site
    from ..sun_protocols import SlewSafePredicate, SunSafePredicate

__all__ = ["ESCAPE_BLOCK_NAME", "EscapeMove", "plan_escape_move", "sweep_sun_safe"]

#: ``patch_name`` of the SLEW block that records an escape, in both loops.
ESCAPE_BLOCK_NAME = "sun_escape"

# Per-sample fallback budget for a predicate without ``batch``: a 10 min pass
# at 0.1 s has 6000 samples; 600 keeps the fail-closed sweep at ~1 s spacing.
_SWEEP_MAX_SAMPLES = 600


@dataclass(frozen=True)
class EscapeMove:
    """What a loop should do about the pose it is sitting at.

    Four outcomes, distinguished by which fields are populated: the pose
    is clear (nothing set), the telescope moves (``transition`` and
    ``block``), the zone holds it (``transition`` and ``label``), or an
    escape exists but does not complete before the window closes
    (``transition`` and ``label``, with ``fits`` false).

    Attributes
    ----------
    transition : Transition or None
        The escape, or ``None`` when the pose is already outside the
        zone and nothing has to happen.
    block : TimelineBlock or None
        The SLEW block that records the move, ready to emit; ``None``
        whenever the telescope does not actually move.
    label : DeferralReason or None
        The reason an idle emitted instead of the move must carry.
        ``None`` exactly when the move is emitted or the pose is clear.
    fits : bool
        Whether the escape completes inside the remaining window. A
        caller that must fill its window idles out the remainder under
        ``label``; one that stops at the end of its window stops.
    """

    transition: Transition | None = None
    block: TimelineBlock | None = None
    label: DeferralReason | None = None
    fits: bool = True

    @property
    def clear(self) -> bool:
        """Whether the pose is outside the zone, so nothing has to move."""
        return self.transition is None


def plan_escape_move(
    az: float,
    el: float,
    time: Time,
    site: Site,
    *,
    sun_safe: SunSafePredicate | None = None,
    slew_safe: SlewSafePredicate | None = None,
    settle_time: float = 0.0,
    el_floor: float | None = None,
    remaining: float,
    scan_index: int = 0,
    cache: dict | None = None,
) -> EscapeMove:
    """Decide what to do about a pose the Sun zone may have overtaken.

    Wraps :func:`~fyst_trajectories.overhead.plan_escape` with the two
    pieces every caller needs after it: the block that records the move,
    and the verdict on whether the move fits in the time that is left.

    Parameters
    ----------
    az, el : float
        The pose in question, in degrees.
    time : Time
        Scalar time the move would start.
    site : Site
        Telescope site; supplies the axis limits, the kinematic estimate
        and the default safety models.
    sun_safe : SunSafePredicate, optional
        Point-level Sun predicate; ``None`` (default) builds the scalar
        model from the site's avoidance radii.
    slew_safe : SlewSafePredicate, optional
        Path-level model; ``None`` (default) builds one from the site's
        axis limits on every call, so a loop should pass its own.
    settle_time : float, optional
        Seconds added to the kinematic estimate after arrival. Default
        ``0.0``.
    el_floor : float, optional
        Lowest elevation the escape may use, in degrees. Default the
        site's elevation limit; pass the observing floor so the escape
        does not park below the sky the loop is willing to use.
    remaining : float
        Seconds left in the window the caller is filling. An escape
        longer than this is reported with ``fits`` false.
    scan_index : int, optional
        Scan counter stamped on the emitted block. Default ``0``.
    cache : dict, optional
        Per-run memo of the search, keyed by pose, time, elevation floor
        and settle time. Both loops ask about one pose and time from
        several places in a tick, and the search costs a Sun ephemeris
        solve even when it answers "clear", so a loop passes one dict for
        the whole run. Valid only while the safety models are fixed, which
        they are for one run: pass a fresh dict per set of models.

    Returns
    -------
    EscapeMove
        The decision. ``EscapeMove.clear`` is true when the pose needs
        no move at all.
    """
    key = (float(az), float(el), float(time.unix), el_floor, float(settle_time))
    if cache is not None and key in cache:
        escape = cache[key]
    else:
        escape = plan_escape(
            az,
            el,
            time,
            site,
            sun_safe=sun_safe,
            slew_safe=slew_safe,
            settle_time=settle_time,
            el_floor=el_floor,
        )
        if cache is not None:
            cache[key] = escape
    if escape is None:
        return EscapeMove()
    if not escape.safe:
        # The zone holds the telescope: it stays where it is, labelled.
        return EscapeMove(transition=escape, label=escape.cause)
    if escape.duration >= remaining:
        # An escape exists but there is no time to make it, so the
        # telescope is held by the window rather than by the zone. The
        # label is the same, since the pose stays inside the zone either
        # way.
        return EscapeMove(transition=escape, label=DeferralReason.NO_ESCAPE, fits=False)
    block = TimelineBlock.slew(
        t_start=time,
        duration=escape.duration,
        az_start=az,
        az_end=escape.az_to,
        el=escape.el_to,
        site=site,
        scan_index=scan_index,
        patch_name=ESCAPE_BLOCK_NAME,
    )
    return EscapeMove(transition=escape, block=block)


def sweep_sun_safe(sun_safe: SunSafePredicate, az: np.ndarray, el: np.ndarray, times: Time) -> bool:
    """Whether a trajectory's samples are clear of the Sun, fail closed.

    Uses the predicate's vectorised ``batch`` when it has one, which checks
    every sample; otherwise evaluates the predicate per sample on an even
    subsample of at most :data:`_SWEEP_MAX_SAMPLES` points, both ends
    included, so that path is a sampled gate rather than a proof.

    An empty trajectory answers ``False``. There is nothing to screen, so a
    ``True`` would be a vacuous pass of the Sun gate rather than a verdict,
    the same hole the dispatch-time wrap gate fails closed on.
    """
    az = np.asarray(az, dtype=float)
    el = np.asarray(el, dtype=float)
    if az.size == 0:
        return False
    if isinstance(sun_safe, BatchSunSafePredicate):
        verdicts = np.asarray(sun_safe.batch(az, el, times), dtype=bool)
        if verdicts.shape != az.shape:
            raise ValueError(f"sun_safe.batch returned shape {verdicts.shape}, expected {az.shape}")
        return bool(verdicts.all())
    n = az.size
    stride = max(1, int(np.ceil(n / _SWEEP_MAX_SAMPLES)))
    index = list(range(0, n, stride))
    if index[-1] != n - 1:
        index.append(n - 1)
    return all(bool(sun_safe(float(az[i]), float(el[i]), times[i])) for i in index)
