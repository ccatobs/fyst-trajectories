"""Public API surface guard.

These tests constrain the ``__all__`` surface that downstream consumers
import from. They guarantee that every advertised symbol is actually importable
and that no private name leaks into the public surface.

Scope and limits:

- :func:`test_top_level_all_symbols_importable` /
  :func:`test_patterns_all_symbols_importable` parametrize over the live
  ``__all__`` lists, so a re-export that drops a symbol leaves a *dangling*
  name (in ``__all__`` but not importable) and fails here.
- :func:`test_no_private_names_leak_into_all` catches a private (single
  leading-underscore) symbol leaking into the public surface.
- :func:`test_planning_reexports_are_consistent_with_top_level` pins that
  every planning symbol re-exported at the top level is the *same object*
  whether imported from the top level or from :mod:`fyst_trajectories.planning`.
- :func:`test_root_constants_are_their_module_attributes` pins the same
  identity for the defaults and vocabularies the root re-exports beside the
  functions that use them.

What these tests deliberately do **not** do is assert the exact *membership*
of ``__all__`` (an intentional add/remove is a normal change, not a
regression). Detecting accidental removals, a symbol silently dropped from
``__all__`` entirely, is left to an explicit review that diffs the full
surface against the source.
"""

import importlib

import pytest

import fyst_trajectories
from fyst_trajectories import patterns, planning, visualization


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


def test_planning_reexports_are_consistent_with_top_level():
    """Planning symbols re-exported at the top level are the same objects."""
    shared = set(fyst_trajectories.__all__) & set(planning.__all__)
    assert shared, "expected the top level to re-export planning symbols"
    mismatched = [
        name for name in shared if getattr(fyst_trajectories, name) is not getattr(planning, name)
    ]
    assert not mismatched, f"top-level re-exports diverge from planning.__all__: {mismatched}"


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("DEFAULT_RETUNE_DURATION_SEC", "fyst_trajectories.retune"),
        ("GO_TCS_MIN_SAMPLE_INTERVAL_SEC", "fyst_trajectories.trajectory_utils"),
        ("TRACKPOINT_NEW_LEG_GROUP_SIZE", "fyst_trajectories.trajectory_utils"),
        ("EncoderSolutionCause", "fyst_trajectories.exceptions"),
    ],
)
def test_root_constants_are_their_module_attributes(name, module):
    """The defaults and vocabularies the root re-exports are the defining module's objects."""
    assert getattr(fyst_trajectories, name) is getattr(importlib.import_module(module), name)


@pytest.mark.parametrize("module", [planning, visualization], ids=lambda m: m.__name__)
def test_library_subpackage_all_is_sound(module):
    """Each library-tier subpackage ``__all__`` resolves, is public, and has no duplicate."""
    names = module.__all__
    assert [n for n in names if not hasattr(module, n)] == []
    assert [n for n in names if _is_private(n)] == []
    assert len(names) == len(set(names))
