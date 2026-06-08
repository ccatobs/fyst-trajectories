"""Public API surface guard.

These tests constrain the ``__all__`` surface that downstream consumers
(``scan_patterns``, ``primecam_camera_mapping_simulations``, the OCS client,
the KOSMA translator) import from. They guarantee that every advertised
symbol is actually importable and that no private name leaks into the public
surface.

Scope and limits:

- :func:`test_top_level_all_symbols_importable` /
  :func:`test_patterns_all_symbols_importable` parametrize over the live
  ``__all__`` lists, so a re-export that drops a symbol leaves a *dangling*
  name (in ``__all__`` but not importable) and fails here.
- :func:`test_no_private_names_leak_into_all` catches a private (single
  leading-underscore) symbol leaking into the public surface.
- :func:`test_identity_across_import_paths` pins that the planning symbols are
  the *same object* whether imported from the top level or from
  :mod:`fyst_trajectories.planning`.

What these tests deliberately do **not** do is assert the exact *membership*
of ``__all__`` (an intentional add/remove is a normal change, not a
regression). Detecting accidental removals -- a symbol silently dropped from
``__all__`` entirely -- is the job of the ``check-api`` skill, which diffs the
full surface against the source. The hardcoded planning checks below cover the
five symbols downstream code is known to import.
"""

import pytest

import fyst_trajectories
from fyst_trajectories import patterns


def _is_private(name: str) -> bool:
    """Return True for single-underscore private names (dunders are public)."""
    return name.startswith("_") and not name.startswith("__")


@pytest.mark.parametrize("name", fyst_trajectories.__all__)
def test_top_level_all_symbols_importable(name):
    """Every name in top-level ``__all__`` resolves to a real attribute."""
    assert hasattr(fyst_trajectories, name), (
        f"{name!r} is in fyst_trajectories.__all__ but not importable"
    )


@pytest.mark.parametrize("name", patterns.__all__)
def test_patterns_all_symbols_importable(name):
    """Every name in ``patterns.__all__`` resolves to a real attribute."""
    assert hasattr(patterns, name), (
        f"{name!r} is in fyst_trajectories.patterns.__all__ but not importable"
    )


def test_no_private_names_leak_into_all():
    """No private (single-underscore) symbol is advertised in either ``__all__``."""
    leaked_top = [n for n in fyst_trajectories.__all__ if _is_private(n)]
    leaked_patterns = [n for n in patterns.__all__ if _is_private(n)]
    assert not leaked_top, f"private names leaked into fyst_trajectories.__all__: {leaked_top}"
    assert not leaked_patterns, f"private names leaked into patterns.__all__: {leaked_patterns}"


def test_no_duplicate_names_in_all():
    """``__all__`` lists are hand-maintained; guard against copy-paste duplicates."""
    top = fyst_trajectories.__all__
    pat = patterns.__all__
    assert len(top) == len(set(top)), "duplicate name(s) in fyst_trajectories.__all__"
    assert len(pat) == len(set(pat)), "duplicate name(s) in patterns.__all__"


def test_patterns_reexports_are_consistent_with_top_level():
    """Pattern symbols re-exported at the top level are the same objects.

    ``patterns/__init__`` defines the canonical pattern objects and the
    top-level ``__init__`` re-exports a subset of them. Any symbol present in
    both ``__all__`` lists must refer to the identical object so consumers get
    the same class/function regardless of import path.
    """
    shared = set(fyst_trajectories.__all__) & set(patterns.__all__)
    assert shared, "expected the top level to re-export pattern symbols"
    mismatched = [
        name for name in shared if getattr(fyst_trajectories, name) is not getattr(patterns, name)
    ]
    assert not mismatched, f"top-level re-exports diverge from patterns.__all__: {mismatched}"


def test_identity_across_import_paths():
    """Planning symbols imported from both paths are the same object."""
    from fyst_trajectories import FieldRegion as F1
    from fyst_trajectories import ScanBlock as S1
    from fyst_trajectories import plan_constant_el_scan as pce1
    from fyst_trajectories import plan_daisy_scan as pda1
    from fyst_trajectories import plan_pong_scan as pp1
    from fyst_trajectories.planning import FieldRegion as F2
    from fyst_trajectories.planning import ScanBlock as S2
    from fyst_trajectories.planning import plan_constant_el_scan as pce2
    from fyst_trajectories.planning import plan_daisy_scan as pda2
    from fyst_trajectories.planning import plan_pong_scan as pp2

    assert F1 is F2
    assert S1 is S2
    assert pp1 is pp2
    assert pce1 is pce2
    assert pda1 is pda2
