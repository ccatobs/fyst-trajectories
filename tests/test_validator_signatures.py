"""Guard the calling convention of the validators and the dispatch helpers.

Each takes the data it checks or prices positionally and ``site``, with every
option after it, by keyword, as the planners take everything after their
target, so a call reads the same in every family.
"""

import inspect

import pytest

from fyst_trajectories.dispatch import choose_encoder_solution, estimate_slew_time
from fyst_trajectories.trajectory_utils import (
    validate_sun_avoidance,
    validate_trajectory,
    validate_trajectory_bounds,
    validate_trajectory_dynamics,
)

_POSITIONAL = inspect.Parameter.POSITIONAL_OR_KEYWORD
_KEYWORD = inspect.Parameter.KEYWORD_ONLY

# Each function with the parameters it takes positionally (its data group).
_DATA_GROUPS = [
    (validate_trajectory, ("trajectory",)),
    (validate_trajectory_bounds, ("az", "el")),
    (validate_trajectory_dynamics, ("az", "el", "times")),
    (validate_sun_avoidance, ("az", "el", "times")),
    (estimate_slew_time, ("az1", "el1", "az2", "el2")),
    (choose_encoder_solution, ("current_az", "current_el", "goal_az", "goal_el", "obstime")),
]


@pytest.mark.parametrize(
    ("function", "data"), _DATA_GROUPS, ids=[f.__name__ for f, _ in _DATA_GROUPS]
)
def test_only_the_data_is_positional(function, data):
    """The data group is positional-or-keyword; ``site`` and every option are keyword-only."""
    params = list(inspect.signature(function).parameters.values())
    assert tuple(p.name for p in params[: len(data)]) == data
    assert all(p.kind is _POSITIONAL for p in params[: len(data)])
    rest = params[len(data) :]
    assert rest[0].name == "site"
    not_keyword = [p.name for p in rest if p.kind is not _KEYWORD]
    assert not_keyword == [], f"{function.__name__}: {not_keyword} are not keyword-only"


@pytest.mark.parametrize(
    ("function", "data"), _DATA_GROUPS, ids=[f.__name__ for f, _ in _DATA_GROUPS]
)
def test_a_positional_site_is_refused(site, function, data):
    """``site`` passed positionally raises ``TypeError`` before anything is checked."""
    with pytest.raises(TypeError, match="positional argument"):
        function(*([None] * len(data)), site)
