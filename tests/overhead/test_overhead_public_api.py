"""Public API surface guard for the simulator tier.

The simulator-tier twin of ``test_library_subpackage_all_is_sound`` in
``tests/test_public_api.py``: each ``__all__`` list below resolves, holds no
private name and has no duplicate. The subpackages are imported by name so
``tests/conftest.py`` marks this module ``offline``.
"""

import pytest

from fyst_trajectories import overhead
from fyst_trajectories.overhead import calibration_night, scheduler


def _is_private(name: str) -> bool:
    """Return True for single-underscore private names (dunders are public)."""
    return name.startswith("_") and not name.startswith("__")


@pytest.mark.parametrize(
    "module", [overhead, scheduler, calibration_night], ids=lambda m: m.__name__
)
def test_simulator_subpackage_all_is_sound(module):
    """Each simulator-tier subpackage ``__all__`` resolves, is public, and has no duplicate."""
    names = module.__all__
    assert [n for n in names if not hasattr(module, n)] == []
    assert [n for n in names if _is_private(n)] == []
    assert len(names) == len(set(names))


def test_schema_names_are_the_overhead_exports():
    """Every ``overhead.schemas`` public name is the object ``overhead`` exports."""
    from fyst_trajectories.overhead import schemas

    assert len(schemas.__all__) == 12
    assert [n for n in schemas.__all__ if n not in overhead.__all__] == []
    assert [n for n in schemas.__all__ if getattr(overhead, n) is not getattr(schemas, n)] == []
