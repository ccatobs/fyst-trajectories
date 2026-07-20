"""Visualization subpackage: all matplotlib rendering for fyst-trajectories.

Plot functions live only here and are never re-exported from the package
root; the single public import path is ``from
fyst_trajectories.visualization import ...``. Every function lazy-imports
matplotlib inside its body (install via ``pip install
fyst-trajectories[plotting]``), so importing this subpackage never pulls
matplotlib; the import-isolation test enforces that nothing outside
``visualization/`` imports matplotlib at all.

Pre-1.0 stability stance: plot-function signatures are excluded from the
API stability promise that covers the planners and coordinate layers;
they may change between minor versions.

Placement rationale: ``docs/ecosystem/plotting_architecture_review_2026-07-16.md``.
"""

from .hitmap import plot_hit_map
from .overhead import plot_sky_coverage, plot_timeline_gantt
from .trajectory import plot_trajectory

__all__ = [
    "plot_hit_map",
    "plot_sky_coverage",
    "plot_timeline_gantt",
    "plot_trajectory",
]
