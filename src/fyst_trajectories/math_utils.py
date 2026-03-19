"""Math utilities and constants for numerical operations.

This module provides named constants for numerical tolerances and small
epsilon values used throughout the fyst_trajectories library.

Constants
---------
SMALL_DISTANCE_EPSILON : float
    Epsilon for detecting near-zero distances or radii.
    Used when checking if position is effectively at center/origin.
"""

SMALL_DISTANCE_EPSILON: float = 1e-10
