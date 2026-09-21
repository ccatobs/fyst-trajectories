"""Math utilities and constants for numerical operations.

One named tolerance constant, shared so the value is stated in a single
place.

Constants
---------
SMALL_DISTANCE_EPSILON : float
    Epsilon for detecting near-zero distances or radii.
    Used when checking if position is effectively at center/origin.
"""

SMALL_DISTANCE_EPSILON: float = 1e-10
