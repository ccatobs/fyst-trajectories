"""Error types raised by the offline observing-night simulator.

The simulator's own refusals live here rather than in the library tier's
:mod:`fyst_trajectories.exceptions`, so a library-tier consumer never
imports a simulator-only concept. Both classes subclass
:class:`~fyst_trajectories.exceptions.PointingError` (see that module's
docstring for the ``ValueError`` rationale), so a caller that wants to
tell a schema mismatch from an unreconstructable block can.

Exception hierarchy
-------------------
::

    PointingError (ValueError)
        ScanParamsSchemaError
        BlockNotReconstructableError
"""

from __future__ import annotations

from ..exceptions import PointingError


class ScanParamsSchemaError(PointingError):
    """Raised when recorded block metadata does not match the expected schema.

    The recorded ``scan_params`` or block metadata is missing a key the
    rebuild needs, or names a scan type the simulator cannot rebuild. The
    block itself may be perfectly well formed on the timeline; what failed
    is the contract between what was written and what the rebuild expects.
    """


class BlockNotReconstructableError(PointingError):
    """Raised when a block's schema is sound but its trajectory cannot be rebuilt.

    The recorded parameters are complete, yet re-solving them does not
    produce a trajectory that covers this block: the re-solved scan no
    longer overlaps the block's own time window, or the parameters carry no
    window and the caller supplied no fallback.
    """
