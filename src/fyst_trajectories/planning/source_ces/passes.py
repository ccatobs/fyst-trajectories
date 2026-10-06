"""Helpers of ``plan_source_ces_passes``: the passes' eta offsets and their per-pass tags."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from ..._validation import _require_positive
from .._types import ScanBlock, SourceCESComputedParams


def _tag_pass_block(
    block: ScanBlock[SourceCESComputedParams],
    *,
    pass_index: int,
    n_passes: int,
    eta_offset_deg: float,
    el_bore_deg: float,
) -> ScanBlock[SourceCESComputedParams]:
    """Attach per-pass metadata to a source-CES ``ScanBlock``.

    Records the pass index, total pass count, focal-plane eta offset
    (the row this pass drags the source through), and the stepped
    ``el_bore`` in the trajectory metadata's ``pattern_params`` and
    prepends a one-line header to the summary, so a consumer can tell
    which stripe a pass covers without re-deriving it. The trajectory
    arrays, config, computed_params, and duration are untouched.
    """
    meta = block.trajectory.metadata
    new_params = dict(meta.pattern_params)
    new_params.update(
        {
            "pass_index": int(pass_index),
            "n_passes": int(n_passes),
            "pass_eta_offset_deg": float(eta_offset_deg),
            "pass_el_bore_deg": float(el_bore_deg),
        }
    )
    new_meta = dataclasses.replace(meta, pattern_params=new_params)
    new_traj = dataclasses.replace(block.trajectory, metadata=new_meta)
    header = (
        f"[Pass {pass_index + 1}/{n_passes}: eta_offset={eta_offset_deg:+.3f} deg "
        f"(focal-plane row), el_bore={el_bore_deg:.2f} deg]\n"
    )
    return dataclasses.replace(block, trajectory=new_traj, summary=header + block.summary)


def _resolve_pass_offsets(
    *,
    n_passes: int | None,
    eta_offsets: Sequence[float] | None,
    step: float | None,
    footprint_eta_extent: float,
) -> list[float]:
    """Resolve the pass controls into a sorted list of eta offsets.

    Either ``eta_offsets`` (explicit) or ``n_passes`` (+ optional
    ``step``) must be supplied, not both. When ``n_passes`` is used the
    offsets form the symmetric grid ``step * (k - (n_passes - 1) / 2)``;
    with the default ``step = footprint_eta_extent / n_passes`` the pass
    centers spread evenly across ``[-extent/2, +extent/2]``. The grid
    positions the track centers only: each pass's on-sky eta coverage is
    wider than ``step`` (focal-plane rotation mixes the azimuth throw
    into eta), so successive passes interleave and densify the coverage
    rather than painting disjoint bands.
    """
    has_n = n_passes is not None
    has_list = eta_offsets is not None
    if has_n and has_list:
        raise ValueError("specify either 'n_passes' or 'eta_offsets', not both")
    if not has_n and not has_list:
        raise ValueError("must specify either 'n_passes' or 'eta_offsets'")
    if step is not None and not has_n:
        raise ValueError("'step' is only valid together with 'n_passes'")

    if has_list:
        assert eta_offsets is not None  # narrow for type-checker
        offsets = [float(o) for o in eta_offsets]
        if not offsets:
            raise ValueError("eta_offsets cannot be empty")
        if not all(np.isfinite(o) for o in offsets):
            raise ValueError("eta_offsets must all be finite")
        if len(set(offsets)) != len(offsets):
            raise ValueError("eta_offsets must be unique")
        return sorted(offsets)

    assert n_passes is not None  # narrow for type-checker
    if n_passes < 1:
        raise ValueError(f"n_passes must be at least 1, got {n_passes}")
    if step is None:
        step_val = footprint_eta_extent / n_passes
    else:
        step_val = float(step)
        _require_positive(step_val, "step")
    return [step_val * (k - (n_passes - 1) / 2.0) for k in range(n_passes)]
