"""Shared Sun-model test doubles, and the gate for the optional shared library.

Several test files share one fake Sun-safety model rather than each
re-implementing the ``__call__`` / ``batch`` / ``threshold`` surface and its
array-shape contract. The fakes are deliberately unequal (each exposes only
the capabilities the call site under test may discover), so the builder below
takes the capability subset as arguments rather than offering one all-capable
stub. A change to the shape contract is then one edit here.

The optional ``sun_avoidance`` gate lives here too: both documentation guards
skip the examples that select a shared-library model, and they must skip on
the same condition, ``"cad"`` or ``"cone"`` (the scalar model is the only one
that needs no library).
"""

from __future__ import annotations

import numpy as np

try:
    import sun_avoidance  # noqa: F401

    HAVE_SUN_AVOIDANCE = True
except ImportError:
    HAVE_SUN_AVOIDANCE = False

#: Model names that bind the shared library, in either quote style.
_LIBRARY_MODEL_MARKERS = ('"cad"', "'cad'", '"cone"', "'cone'")


def needs_sun_avoidance(source: str) -> bool:
    """Report whether source text selects a shared-library sun model."""
    return any(marker in source for marker in _LIBRARY_MODEL_MARKERS)


def fake_sun_model(verdict=True, *, batch=True, threshold=None, describe=None):
    """Build a fake Sun-safety model exposing a chosen capability subset.

    Parameters
    ----------
    verdict : bool or callable
        The point verdict: a constant, or ``f(az, el, time) -> bool`` applied
        element-wise (safe is ``True``).
    batch : bool, default True
        Whether the model exposes the vectorised ``batch`` extension. Set it
        ``False`` for a plain point predicate, so a call site that discovers
        ``batch`` takes its scalar path.
    threshold : float or callable, optional
        Adds the optional ``threshold`` extension, returning this value (or
        ``f(az, el, times)``) broadcast over the inputs. Omitted when None.
    describe : str, optional
        Value of the optional ``describe`` attribute, which the plots put in
        their legends. Omitted when None.

    Returns
    -------
    object
        A callable model. Every fake counts its consultations in
        ``scalar_calls`` and ``batch_calls``.
    """
    evaluate = verdict if callable(verdict) else (lambda az, el, time: verdict)

    def _call(self, az, el, time):
        self.scalar_calls += 1
        return bool(evaluate(az, el, time))

    def _batch(self, az, el, times):
        self.batch_calls += 1
        az_b, el_b = np.broadcast_arrays(
            np.atleast_1d(np.asarray(az, dtype=float)),
            np.atleast_1d(np.asarray(el, dtype=float)),
        )
        verdicts = np.asarray(evaluate(az_b, el_b, times), dtype=bool)
        return np.broadcast_to(verdicts, az_b.shape).copy()

    def _threshold(self, az, el, times):
        az_b, el_b = np.broadcast_arrays(
            np.atleast_1d(np.asarray(az, dtype=float)),
            np.atleast_1d(np.asarray(el, dtype=float)),
        )
        value = threshold(az_b, el_b, times) if callable(threshold) else threshold
        return np.full(az_b.shape, value, dtype=float)

    def _init(self):
        self.scalar_calls = 0
        self.batch_calls = 0

    namespace = {"__init__": _init, "__call__": _call}
    if batch:
        namespace["batch"] = _batch
    if threshold is not None:
        namespace["threshold"] = _threshold
    if describe is not None:
        namespace["describe"] = describe
    return type("FakeSunModel", (), namespace)()


def block_everything(az, el, t):
    """Report every sample unsafe, whatever the geometry."""
    return False


def allow_everything(az, el, t):
    """Report every sample clear of the Sun, whatever the geometry."""
    return True


def pose_blocker(az0, el0, radius=5.0, until=None):
    """Build a point predicate unsafe within ``radius`` deg of one pose.

    With ``until`` the pose becomes safe from that time onward, which is how
    a test makes the zone move off a telescope it has overtaken.
    """

    def predicate(az, el, t):
        if until is not None and t >= until:
            return True
        return not (abs(float(az) - az0) < radius and abs(float(el) - el0) < radius)

    return predicate
