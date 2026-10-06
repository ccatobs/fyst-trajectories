"""Tests for the Sun-avoidance seam's protocols (``fyst_trajectories.sun_protocols``).

The two base contracts test only ``__call__``, so any callable of the right
arity satisfies both; the three runtime-checkable extensions are what a
consumer discovers ``batch``, ``threshold`` and ``evaluate`` through. The
private ``_sun_verdicts`` helper is the one place a point model's verdicts
over a grid are asked for, through ``batch`` when the model has it.
"""

import numpy as np
import pytest
from _sun_stubs import HAVE_SUN_AVOIDANCE, fake_sun_model
from astropy.time import Time, TimeDelta

from fyst_trajectories import Coordinates, get_fyst_site
from fyst_trajectories.sun_models import make_slew_safe, make_sun_safe
from fyst_trajectories.sun_protocols import (
    BatchSunSafePredicate,
    PathSlewSafePredicate,
    SlewSafePredicate,
    SunSafePredicate,
    ZonedSunSafePredicate,
    _sun_verdicts,
)

T0 = Time("2026-03-15T12:00:00", scale="utc")

PROTOCOLS = (
    SunSafePredicate,
    SlewSafePredicate,
    BatchSunSafePredicate,
    ZonedSunSafePredicate,
    PathSlewSafePredicate,
)

_needs_library = pytest.mark.skipif(
    not HAVE_SUN_AVOIDANCE, reason="the shared sun-avoidance library is not installed"
)


def _membership(obj) -> set[str]:
    """Return the names of the protocols ``obj`` satisfies."""
    return {p.__name__ for p in PROTOCOLS if isinstance(obj, p)}


class _BatchOnly:
    """A point model with ``batch`` and no ``threshold``."""

    def __call__(self, az, el, time):
        return True

    def batch(self, az, el, times):
        return np.ones(np.shape(np.atleast_1d(az)), dtype=bool)


class _ThresholdOnly:
    """A point model with ``threshold`` and no ``batch``."""

    def __call__(self, az, el, time):
        return True

    def threshold(self, az, el, times):
        return np.full(np.shape(np.atleast_1d(az)), 45.0)


class _EvaluatingPath:
    """A path model exposing ``evaluate``, the way ``make_slew_safe`` builds one."""

    def __call__(self, current_az, current_el, goal_az, goal_el, time):
        return True

    def evaluate(self, current_az, current_el, goal_az, goal_el, time):
        times = time + TimeDelta([0.0, 1.0], format="sec")
        return True, np.array([current_az, goal_az]), np.array([current_el, goal_el]), times


class TestMembership:
    """Which object satisfies which protocol."""

    BASE = {"SunSafePredicate", "SlewSafePredicate"}

    @pytest.mark.parametrize(
        "model",
        ["scalar", pytest.param("cad", marks=_needs_library)],
    )
    def test_make_sun_safe_models_are_zoned(self, model):
        assert _membership(make_sun_safe(model)) == self.BASE | {
            "BatchSunSafePredicate",
            "ZonedSunSafePredicate",
        }

    def test_the_library_model_class_is_zoned_without_the_library(self):
        """The class behind "cad" and "cone" is zoned, checked where the library is absent too."""
        from fyst_trajectories.sun_models import _LibrarySunModel

        assert issubclass(_LibrarySunModel, ZonedSunSafePredicate)

    def test_make_slew_safe_models_are_path_predicates(self):
        assert _membership(make_slew_safe("scalar")) == self.BASE | {"PathSlewSafePredicate"}

    def test_a_bare_callable_satisfies_only_the_base_protocols(self):
        """Arity is not checked at run time, so either base protocol accepts any callable."""
        assert _membership(lambda az, el, time: True) == self.BASE
        assert _membership(lambda a, b, c, d, e: True) == self.BASE
        assert _membership(Coordinates(get_fyst_site()).is_sun_safe) == self.BASE

    def test_a_batch_only_model_is_batch_but_not_zoned(self):
        assert _membership(_BatchOnly()) == self.BASE | {"BatchSunSafePredicate"}

    def test_a_threshold_without_batch_is_not_zoned(self):
        """The zoned extension builds on ``batch``; ``threshold`` alone is not enough."""
        assert _membership(_ThresholdOnly()) == self.BASE

    def test_a_model_with_evaluate_is_a_path_predicate(self):
        assert _membership(_EvaluatingPath()) == self.BASE | {"PathSlewSafePredicate"}


class TestSunVerdicts:
    """``_sun_verdicts`` asks ``batch`` once, or the scalar call per sample."""

    AZ = np.array([10.0, 20.0, 30.0, 40.0])
    EL = np.full(4, 45.0)
    TIMES = T0 + TimeDelta(np.arange(4.0), format="sec")

    @staticmethod
    def _verdict(az, el, time):
        return np.asarray(az, dtype=float) < 25.0

    def test_a_batch_model_is_asked_once(self):
        model = fake_sun_model(self._verdict)
        verdicts = _sun_verdicts(model, self.AZ, self.EL, self.TIMES)
        assert verdicts.tolist() == [True, True, False, False]
        assert (model.batch_calls, model.scalar_calls) == (1, 0)

    def test_a_bare_predicate_is_asked_per_sample(self):
        model = fake_sun_model(self._verdict, batch=False)
        verdicts = _sun_verdicts(model, self.AZ, self.EL, self.TIMES)
        assert verdicts.tolist() == [True, True, False, False]
        assert model.scalar_calls == 4

    def test_each_sample_gets_its_own_time(self):
        seen = []

        def predicate(az, el, time):
            seen.append(float(time.unix))
            return True

        _sun_verdicts(predicate, self.AZ, self.EL, self.TIMES)
        assert seen == pytest.approx(list(self.TIMES.unix))

    @pytest.mark.parametrize("batch", [True, False])
    def test_a_scalar_time_is_accepted(self, batch):
        """A position array at one instant is a call shape the ``batch`` contract admits."""
        model = fake_sun_model(self._verdict, batch=batch)
        verdicts = _sun_verdicts(model, [float(self.AZ[0])], [float(self.EL[0])], T0)
        assert verdicts.tolist() == [True]

    def test_the_answer_is_one_boolean_per_sample(self):
        model = fake_sun_model(lambda az, el, time: 1, batch=False)
        verdicts = _sun_verdicts(model, self.AZ, self.EL, self.TIMES)
        assert verdicts.dtype == bool
        assert verdicts.shape == (4,)

    def test_a_wrong_batch_shape_raises_with_the_label(self):
        """A short answer would broadcast one verdict over the grid; it must raise."""

        class _Short:
            def __call__(self, az, el, time):
                return True

            def batch(self, az, el, times):
                return np.ones(1, dtype=bool)

        with pytest.raises(
            ValueError,
            match=r"sun_safe\.batch returned shape \(1,\), expected \(4,\) verdicts for the "
            r"slew path",
        ):
            _sun_verdicts(_Short(), self.AZ, self.EL, self.TIMES, what="slew path")

    def test_the_default_label_names_a_grid(self):
        class _Short:
            def __call__(self, az, el, time):
                return True

            def batch(self, az, el, times):
                return np.ones(2, dtype=bool)

        with pytest.raises(ValueError, match="verdicts for the grid$"):
            _sun_verdicts(_Short(), self.AZ, self.EL, self.TIMES)
