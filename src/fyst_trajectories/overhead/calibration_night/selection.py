"""Selection rules: which body to visit next.

A rule is a callable over the current candidates and the night state
that returns ``(body, overrides)`` or ``None`` to idle. Two ship:
:func:`select_priority` takes the first available body in the caller's
order, and :class:`ScriptedSelection` walks a fixed list of entries so a
person can lay out, for example, five passes on one body at five speeds.
The scripted rule keeps no cursor of its own: the cursor, the wait timer
and the entries it set aside live in the night state, so a visit that is
planned and discarded at an interactive session leaves nothing advanced.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .policy import ScanOverrides

if TYPE_CHECKING:
    from .state import NightState
    from .step import Candidate

__all__ = [
    "ScriptedSelection",
    "SelectionRule",
    "select_priority",
]


@runtime_checkable
class SelectionRule(Protocol):
    """Choose the next body from the current candidates.

    Implementations must be pure with respect to their own state: read
    the night state, return a choice, mutate nothing.
    """

    def __call__(
        self, candidates: tuple[Candidate, ...], state: NightState
    ) -> tuple[str, ScanOverrides] | None:
        """Return ``(body, overrides)`` for the next visit, or ``None`` to idle.

        ``body`` must be one of the candidates; the driver refuses a body
        it did not offer.
        """
        ...


def select_priority(
    candidates: tuple[Candidate, ...], state: NightState
) -> tuple[str, ScanOverrides] | None:
    """Select the first available candidate in the caller's target order.

    The default :class:`SelectionRule`. It applies no overrides, so every
    visit is planned from the scan tables and the policy alone.

    Parameters
    ----------
    candidates : tuple of Candidate
        This tick's candidates, in the order the caller listed the
        targets; ``list_candidates`` preserves that order.
    state : NightState
        Current night state. Unused here, and read by rules that carry a
        cursor or a wait timer.

    Returns
    -------
    tuple of (str, ScanOverrides) or None
        The body to visit next and an empty override set, or ``None``
        when no candidate is available this tick, which idles the night.
    """
    for candidate in candidates:
        if candidate.available:
            return candidate.body, ScanOverrides()
    return None


@dataclass(frozen=True)
class ScriptedSelection:
    """Visit a fixed list of entries in order, each an optional set of overrides.

    Parameters
    ----------
    entries : sequence of str or (str, ScanOverrides)
        The bodies to visit in order; a bare name means no overrides.
        Consecutive entries naming one body chain back to back, which is
        how a parameter sweep on a single source is written.

    Raises
    ------
    ValueError
        If ``entries`` is empty.

    Examples
    --------
    >>> from fyst_trajectories.overhead import ScanOverrides, ScriptedSelection
    >>> sweep = ScriptedSelection([("jupiter", ScanOverrides(az_speed=v)) for v in (0.5, 1.0, 1.5)])
    >>> len(sweep.entries)
    3
    """

    entries: tuple[tuple[str, ScanOverrides], ...]

    def __init__(self, entries: Sequence[str | tuple[str, ScanOverrides]]):
        normalised: list[tuple[str, ScanOverrides]] = []
        for entry in entries:
            if isinstance(entry, str):
                normalised.append((entry.lower(), ScanOverrides()))
            else:
                body, overrides = entry
                normalised.append((str(body).lower(), overrides))
        if not normalised:
            raise ValueError("entries must not be empty")
        object.__setattr__(self, "entries", tuple(normalised))

    def current(self, state: NightState) -> tuple[str, ScanOverrides] | None:
        """Return the entry at the state's cursor, or ``None`` when the script is spent."""
        if state.script_index >= len(self.entries):
            return None
        return self.entries[state.script_index]

    def __call__(
        self, candidates: tuple[Candidate, ...], state: NightState
    ) -> tuple[str, ScanOverrides] | None:
        """Return the current entry when its body is available, else ``None`` (wait)."""
        entry = self.current(state)
        if entry is None:
            return None
        body, overrides = entry
        for candidate in candidates:
            if candidate.body == body and candidate.available:
                return body, overrides
        return None
