"""Guard the planners' calling convention.

Each planner takes only its target positionally and every other argument by
keyword, so two siblings that order the same parameters differently (the
pong pair's ``velocity`` and ``spacing``) cannot be transposed silently.
Every planner that builds a trajectory defaults ``timestep`` to the same
value. ``plan_source_ces`` is pinned fully keyword-only by
``tests/test_ac_schema_contract.py``, which guards the adapter contract.
"""

import inspect

import pytest
from astropy.time import Time

from fyst_trajectories.planning import (
    FieldRegion,
    compute_source_ces_params,
    plan_constant_el_scan,
    plan_daisy_altaz_scan,
    plan_daisy_scan,
    plan_pong_altaz_scan,
    plan_pong_rotation_sequence,
    plan_pong_scan,
    plan_source_ces,
    plan_source_ces_passes,
)

_POSITIONAL = inspect.Parameter.POSITIONAL_OR_KEYWORD
_KEYWORD = inspect.Parameter.KEYWORD_ONLY

# Each function with the parameters it takes positionally (its target group).
_TARGET_GROUPS = [
    (plan_pong_scan, ("field",)),
    (plan_daisy_scan, ("ra", "dec")),
    (plan_constant_el_scan, ("field",)),
    (plan_pong_altaz_scan, ("az_center", "el_center")),
    (plan_daisy_altaz_scan, ("az_center", "el_center")),
    (plan_pong_rotation_sequence, ("config",)),
    (plan_source_ces_passes, ()),
    (compute_source_ces_params, ()),
]

_TRAJECTORY_PLANNERS = [
    plan_pong_scan,
    plan_daisy_scan,
    plan_constant_el_scan,
    plan_pong_altaz_scan,
    plan_daisy_altaz_scan,
    plan_source_ces,
    plan_source_ces_passes,
]


@pytest.mark.parametrize(
    ("planner", "target"), _TARGET_GROUPS, ids=[f.__name__ for f, _ in _TARGET_GROUPS]
)
def test_only_the_target_is_positional(planner, target):
    """The target group is positional-or-keyword and every other parameter keyword-only."""
    params = list(inspect.signature(planner).parameters.values())
    assert tuple(p.name for p in params[: len(target)]) == target
    assert all(p.kind is _POSITIONAL for p in params[: len(target)])
    rest = params[len(target) :]
    assert rest, f"{planner.__name__} takes nothing after its target"
    not_keyword = [p.name for p in rest if p.kind is not _KEYWORD]
    assert not_keyword == [], f"{planner.__name__}: {not_keyword} are not keyword-only"


@pytest.mark.parametrize("planner", _TRAJECTORY_PLANNERS, ids=lambda f: f.__name__)
def test_timestep_defaults_to_one_tenth_second(planner):
    """Every planner that builds a trajectory defaults ``timestep`` to 0.1 s."""
    assert inspect.signature(planner).parameters["timestep"].default == 0.1


def test_pong_planners_share_num_terms_default():
    """The celestial and AltAz Pong planners default ``num_terms`` alike."""
    celestial = inspect.signature(plan_pong_scan).parameters["num_terms"].default
    altaz = inspect.signature(plan_pong_altaz_scan).parameters["num_terms"].default
    assert celestial == altaz == 4


@pytest.mark.filterwarnings(
    "ignore:High elevation reduces on-sky azimuth speed:"
    "fyst_trajectories.exceptions.PointingWarning",
    "ignore:Trajectory (azimuth|elevation) acceleration:"
    "fyst_trajectories.exceptions.AccelerationLimitWarning",
)
def test_positional_call_after_the_target_raises(site):
    """A call passing scan parameters positionally is refused before any planning."""
    field = FieldRegion(ra_center=180.0, dec_center=-30.0, width=1.0, height=1.0)
    start_time = Time("2026-03-15T04:00:00", scale="utc")
    with pytest.raises(TypeError, match="positional argument"):
        plan_pong_scan(field, 0.5, 0.1, 4, site, start_time, 0.1)
